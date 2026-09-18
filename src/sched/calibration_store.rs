//! Per-host persistence of measured dispatch costs and device
//! capabilities, in a memory-mapped file shared by every process on the
//! host.
//!
//! Calibration is measured, and a measurement costs time and is only as
//! good as the host was quiet. A process that finds a table for its own
//! host reads it and measures nothing; one that does not takes the
//! writer lease beside it, measures, and publishes. The table is keyed
//! by a [`HostStamp`], so a different CPU, a different core count or a
//! different probe set is a different table rather than a stale one.
//!
//! The directory is `FLYNNEL_CALIBRATION_DIR` when set, otherwise
//! `%LOCALAPPDATA%\flynnel\calibration` on Windows and
//! `$XDG_CACHE_HOME/flynnel/calibration` or `~/.cache/flynnel/calibration`
//! elsewhere. A run that measures while the host is busy sets it to a
//! throwaway path; see [`calibration_dir`].
//!
//! # Layout
//!
//! ```text
//! +-------------------------------+
//! | Header  (64 bytes, aligned)   |  magic, versions, SeqLock,
//! |                               |  writer pid, heartbeat, stamp
//! +-------------------------------+
//! | CpuCalibration (64 bytes)     |  the three dispatch thresholds
//! +-------------------------------+
//! | AccelCalibration * MAX_ACCEL  |  one per device, 64 bytes each
//! +-------------------------------+
//! ```
//!
//! # What makes a read safe
//!
//! The magic is written last, after every other field is in place, so a
//! process attaching to a region another process is still laying out
//! sees a zero magic and waits rather than reading an unwritten record
//! as real.
//!
//! Readers take the record under a SeqLock: the writer raises
//! `seq_version` to an odd value before touching the payload and to the
//! next even value after, and a reader that observes an odd version, or
//! a different version either side of its copy, retries. So a record is
//! never half old and half new.
//!
//! A writer that dies mid-measurement leaves its pid in the header and
//! stops beating. A later start whose heartbeat has not advanced within
//! [`LEASE_GRACE_EPOCHS`] takes the lease from it. Without that, one
//! killed process would stop every later start on the host from ever
//! calibrating.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};

use memmap2::{MmapMut, MmapOptions};

/// ASCII "FLCB" then a layout byte. A file whose first eight bytes are
/// anything else is not this table.
pub const CALIBRATION_MAGIC: u64 = 0x464C_4342_0000_0001;

/// Raising this changes every host stamp, so the next start on any host
/// measures again. Raise it whenever a stored field changes meaning.
///
/// It is also what displaces a standing record that nothing else can.
/// A stored record is the cheapest dispatch cost seen for its stamp, so
/// a spuriously fast reading stands for the life of the stamp: nothing
/// later can be cheaper than a wrong floor, and neither a fresh draw nor
/// `FLYNNEL_CALIBRATION_MAX_AGE_S` ageing one out will beat it.
pub const LAYOUT_VERSION: u32 = 8;

/// Devices a table records. A host with more reports the first
/// [`MAX_ACCEL`] and the rest go unrecorded rather than overflowing.
pub const MAX_ACCEL: usize = 8;

/// Heartbeat ticks a writer may miss before a starting process takes
/// the lease. The writer beats before each probe, and a start ticks
/// once, so one missed beat is a writer that has not run a probe in the
/// time it takes another process to start and look.
pub const LEASE_GRACE_EPOCHS: u64 = 2;

/// No process holds the writer lease.
pub const NO_WRITER: u32 = 0;

/// Disagreement between two draws of one host, in parts per thousand of
/// their mean dispatch median, above which they have not confirmed each
/// other and the stored record stays provisional.
///
/// On a quiet 12-core host, twenty pairs: dispatch medians 0 to 206
/// apart, median 74. The within-draw spread over those same forty draws
/// ran 273 to 7429, median 1136.
pub const PROVISIONAL_SPREAD_PER_MILLE: u32 = 250;

/// FNV-1a over bytes. The stamp hashes a short canonical string, so the
/// only property needed is that different strings differ.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// What makes two starts the same host for calibration purposes.
///
/// Every field changes the numbers a probe would produce. The core
/// counts are in because the dispatch cost is a property of the pool,
/// and the pool is sized from them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostStamp {
    /// CPUID leaf 0 vendor string, or `unknown` off x86_64.
    pub vendor: String,
    /// CPUID leaf 1 EAX: stepping, model, family and their extensions.
    pub cpuid_signature: u32,
    /// Architecture the binary was built for.
    pub arch: &'static str,
    /// Operating system the binary was built for.
    pub os: &'static str,
    /// Always-active workers: one per physical core under the default
    /// sizing.
    pub primary_workers: u32,
    /// Every worker thread, SMT siblings among them.
    pub total_workers: u32,
    /// Layout and probe-set version.
    pub layout_version: u32,
}

impl HostStamp {
    /// The stamp of the running host, from the arena that is already
    /// built rather than from a fresh topology probe.
    pub fn detect() -> Self {
        let arena = crate::sched::arena::global_local_arena();
        let (vendor, cpuid_signature) = cpuid_identity();
        Self {
            vendor,
            cpuid_signature,
            arch: std::env::consts::ARCH,
            os: std::env::consts::OS,
            primary_workers: arena.primary_workers() as u32,
            total_workers: arena.total_workers() as u32,
            layout_version: LAYOUT_VERSION,
        }
    }

    /// The string the file name hashes.
    pub fn canonical(&self) -> String {
        format!(
            "vendor={};sig={:#010x};arch={};os={};primary={};total={};v={}",
            self.vendor,
            self.cpuid_signature,
            self.arch,
            self.os,
            self.primary_workers,
            self.total_workers,
            self.layout_version,
        )
    }

    /// FNV-1a of [`canonical`](Self::canonical).
    pub fn hash(&self) -> u64 {
        fnv1a_64(self.canonical().as_bytes())
    }
}

#[cfg(target_arch = "x86_64")]
fn cpuid_identity() -> (String, u32) {
    use std::arch::x86_64::__cpuid_count;
    // Leaves 0 and 1 are architectural on every x86_64 part, so both
    // reads are defined on any host this branch compiles for.
    let (leaf0, leaf1) = (__cpuid_count(0, 0), __cpuid_count(1, 0));
    let mut bytes = [0u8; 12];
    bytes[0..4].copy_from_slice(&leaf0.ebx.to_le_bytes());
    bytes[4..8].copy_from_slice(&leaf0.edx.to_le_bytes());
    bytes[8..12].copy_from_slice(&leaf0.ecx.to_le_bytes());
    let vendor = String::from_utf8_lossy(&bytes)
        .trim_end_matches('\0')
        .to_string();
    (vendor, leaf1.eax)
}

#[cfg(not(target_arch = "x86_64"))]
fn cpuid_identity() -> (String, u32) {
    ("unknown".to_string(), 0)
}

/// The three measured dispatch thresholds, with what the measurement
/// was worth.
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct CpuCalibration {
    /// The pool's cost of one dispatch.
    pub dispatch_cost_ns: u64,
    /// Total work below which a dispatch collapses to the caller.
    pub collapse_threshold_ns: u64,
    /// Total work below which the wake path costs more than polling.
    pub jec_wake_threshold_ns: u64,
    /// Seconds since the Unix epoch at which this was measured.
    pub measured_unix_s: u64,
    /// Spread of the calibration's own samples in parts per thousand of
    /// their median. A record above [`PROVISIONAL_SPREAD_PER_MILLE`]
    /// was taken on a busy host.
    pub spread_per_mille: u32,
    /// Samples the spread was computed from.
    pub samples: u32,
    /// Share of the measuring thread's interval spent on a core, in
    /// parts per mille, or [`OCCUPANCY_UNRECORDED`] where the platform
    /// reports no thread clock.
    ///
    /// The spread says whether the samples agreed with each other. It
    /// cannot say whether they agreed on the wrong number, which is what
    /// a draw taken while a neighbour held half the machine produces:
    /// every sample slow, and slow by about the same amount. This is the
    /// figure that separates those, and it gates nothing - a reader with
    /// two records can prefer the better-drawn one, which is a
    /// comparison rather than a cutoff.
    pub occupancy_per_mille: u32,
    /// Independent draws of this host whose dispatch median agreed with
    /// this record's, to within [`PROVISIONAL_SPREAD_PER_MILLE`].
    ///
    /// Zero makes the record provisional: stored, so the next draw has
    /// something to agree with, and not served, because one draw cannot
    /// say whether its own median reproduces.
    pub confirmations: u32,
    _pad: [u8; 16],
}

/// No thread clock on this platform, so nothing was recorded.
///
/// Distinct from zero, which is the share a thread that never reached a
/// core genuinely reports. A record that could not be measured must not
/// sort as the worst-drawn one.
pub const OCCUPANCY_UNRECORDED: u32 = u32::MAX;

impl CpuCalibration {
    /// A record stamped with the current time.
    ///
    /// `spread_per_mille` comes from
    /// [`crate::sched::par_iter::sample_spread_per_mille`] over the
    /// samples the median was taken from, and decides whether this
    /// stands as the host's calibration or as one loaded reading of it.
    ///
    /// `occupancy_per_mille` is `None` where the platform reports no
    /// thread clock, which is stored as [`OCCUPANCY_UNRECORDED`] rather
    /// than as a share, so a record nobody could measure does not sort
    /// beneath one measured on a saturated host.
    pub fn new(
        dispatch_cost_ns: u64,
        collapse_threshold_ns: u64,
        jec_wake_threshold_ns: u64,
        spread_per_mille: u32,
        samples: u32,
        occupancy_per_mille: Option<u32>,
    ) -> Self {
        Self {
            dispatch_cost_ns,
            collapse_threshold_ns,
            jec_wake_threshold_ns,
            measured_unix_s: now_unix_s(),
            spread_per_mille,
            samples,
            occupancy_per_mille: occupancy_per_mille.unwrap_or(OCCUPANCY_UNRECORDED),
            // Raised only where two records are in hand.
            confirmations: 0,
            _pad: [0; 16],
        }
    }

    /// The share of the machine this record's draw actually got, or
    /// `None` where the platform could not measure it.
    pub fn occupancy(&self) -> Option<u32> {
        match self.occupancy_per_mille {
            OCCUPANCY_UNRECORDED => None,
            share => Some(share),
        }
    }

    /// Whether this record reproduces well enough to stand as the
    /// host's calibration rather than as one reading of it.
    ///
    /// Reproducibility is a property of two draws. No statistic over a
    /// single draw's samples substitutes: `spread_per_mille` is the
    /// range of nine, which one scheduling hiccup sets, and it falls
    /// under load as every sample slows together.
    pub fn is_trustworthy(&self) -> bool {
        self.samples > 0 && self.confirmations > 0
    }
}

/// What kind of device an [`AccelCalibration`] describes.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AccelKind {
    /// The slot holds no device.
    None = 0,
    /// A CUDA device.
    Cuda = 1,
    /// A device reached through the GPU-peer shared region.
    GpuPeer = 2,
}

/// The doorbell handshake completed and was measured.
pub const ACCEL_DOORBELL_OK: u32 = 1 << 0;
/// The Fischer lock self-test passed at the recorded margin.
pub const ACCEL_TIMED_LOCK_OK: u32 = 1 << 1;
/// Cross-device compare-and-swap conserved claims, which a coherent
/// link allows and a PCIe one does not.
pub const ACCEL_SYS_ATOMICS_OK: u32 = 1 << 2;

/// The segmented-wave costs measured on one device at one team size.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WaveCostRecord {
    /// Team size the costs were measured at.
    pub width: u32,
    /// Cost of one cross-block generation barrier, ns.
    pub barrier_ns: u64,
    /// Host round trip of a wave slice apart from its segments, ns.
    pub fixed_ns: u64,
    /// Substrate cost of one segment, ps.
    pub segment_ps: u64,
    /// Fixed cost of one rebalance apart from the ids it moves, ns.
    pub rebalance_fixed_ns: u64,
    /// Rebalance cost of moving one pending id, ps.
    pub copy_ps_per_id: u64,
    /// First-barrier wait of a coupled slice, ns.
    pub skew_ns: u32,
    /// Longest generation of the calibration waves, ns.
    pub generation_ns: u64,
}

/// One device's capabilities and the timings measured against it.
///
/// The capability fields are read from the driver and do not vary with
/// what else is running. The timing fields are measured and do, which
/// is why the record carries the spread of the samples behind them on
/// the same scale the CPU record uses.
///
/// The fields mirror what the GPU peer already measures at every init:
/// a doorbell round trip at three points of its distribution, the
/// cross-device clock error, the visibility bound derived from them,
/// the validated Fischer margin, and a launch-and-synchronise baseline
/// the doorbell path is competing against.
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AccelCalibration {
    /// [`AccelKind`] as its discriminant; `None` marks an unused slot.
    pub kind: u32,
    /// Device ordinal as the driver enumerates it.
    pub ordinal: u32,
    /// Compute capability, major times ten plus minor.
    pub capability: u32,
    /// Multiprocessor count.
    pub multiprocessors: u32,
    /// Peak clock in kHz.
    pub clock_khz: u32,
    /// Device memory in mebibytes.
    pub memory_mib: u32,
    /// `ACCEL_*` capability bits established by the self-tests.
    pub flags: u32,
    /// Spread of the round-trip distribution in parts per thousand of
    /// its median, on the same scale as
    /// [`CpuCalibration::spread_per_mille`].
    pub spread_per_mille: u32,
    /// Doorbell round trip, minimum observed.
    pub rtt_min_ns: u64,
    /// Doorbell round trip, median.
    pub rtt_median_ns: u64,
    /// Doorbell round trip, 99th percentile.
    pub rtt_p99_ns: u64,
    /// One-way visibility bound.
    pub one_way_ns: u64,
    /// Cross-device clock alignment error.
    pub clock_err_ns: u64,
    /// Fischer margin the self-test validated.
    pub delta_ns: u64,
    /// Kernel launch and synchronise baseline.
    pub launch_ns: u64,
    /// Team size the wave costs below were measured at; 0 when none have
    /// been recorded.
    pub wave_width: u32,
    /// First-barrier wait of a coupled slice at `wave_width`, ns.
    pub wave_skew_ns: u32,
    /// Cost of one cross-block generation barrier at `wave_width`, ns.
    pub wave_barrier_ns: u64,
    /// Host round trip of a wave slice apart from its segments, ns.
    pub wave_fixed_ns: u64,
    /// Substrate cost of one segment, ps.
    pub wave_segment_ps: u64,
    /// Rebalance cost of moving one pending id, ps.
    pub wave_copy_ps_per_id: u64,
    /// Longest generation of the calibration waves, ns.
    pub wave_generation_ns: u64,
    /// Fixed cost of one rebalance at `wave_width` apart from the ids it
    /// moves, ns.
    pub wave_rebalance_fixed_ns: u64,
    _pad: [u8; 48],
}

impl Default for AccelCalibration {
    fn default() -> Self {
        Self {
            kind: AccelKind::None as u32,
            ordinal: 0,
            capability: 0,
            multiprocessors: 0,
            clock_khz: 0,
            memory_mib: 0,
            flags: 0,
            spread_per_mille: 0,
            rtt_min_ns: 0,
            rtt_median_ns: 0,
            rtt_p99_ns: 0,
            one_way_ns: 0,
            clock_err_ns: 0,
            delta_ns: 0,
            launch_ns: 0,
            wave_width: 0,
            wave_skew_ns: 0,
            wave_barrier_ns: 0,
            wave_fixed_ns: 0,
            wave_segment_ps: 0,
            wave_copy_ps_per_id: 0,
            wave_generation_ns: 0,
            wave_rebalance_fixed_ns: 0,
            _pad: [0; 48],
        }
    }
}

impl AccelCalibration {
    /// A device record whose spread is derived from the round-trip
    /// distribution it already carries.
    ///
    /// The peer measures the round trip at three points, so how far
    /// apart they fell is the device's equivalent of the CPU sweep's
    /// sample spread, and it costs nothing extra: a device with another
    /// process resident on it stretches the tail without moving the
    /// minimum, which is exactly what this reads.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: AccelKind,
        ordinal: u32,
        capability: u32,
        multiprocessors: u32,
        clock_khz: u32,
        memory_mib: u32,
        flags: u32,
        rtt_min_ns: u64,
        rtt_median_ns: u64,
        rtt_p99_ns: u64,
        one_way_ns: u64,
        clock_err_ns: u64,
        delta_ns: u64,
        launch_ns: u64,
    ) -> Self {
        let spread_per_mille = crate::sched::par_iter::sample_spread_per_mille(&[
            rtt_min_ns,
            rtt_median_ns,
            rtt_p99_ns,
        ]);
        Self {
            kind: kind as u32,
            ordinal,
            capability,
            multiprocessors,
            clock_khz,
            memory_mib,
            flags,
            spread_per_mille,
            rtt_min_ns,
            rtt_median_ns,
            rtt_p99_ns,
            one_way_ns,
            clock_err_ns,
            delta_ns,
            launch_ns,
            wave_width: 0,
            wave_skew_ns: 0,
            wave_barrier_ns: 0,
            wave_fixed_ns: 0,
            wave_segment_ps: 0,
            wave_copy_ps_per_id: 0,
            wave_generation_ns: 0,
            wave_rebalance_fixed_ns: 0,
            _pad: [0; 48],
        }
    }

    /// Whether this slot describes a device.
    pub fn is_present(&self) -> bool {
        self.kind != AccelKind::None as u32
    }

    /// Whether the device was idle enough for the timings to describe
    /// it rather than whatever else was resident on it.
    pub fn is_trustworthy(&self) -> bool {
        self.is_present() && self.spread_per_mille <= PROVISIONAL_SPREAD_PER_MILLE
    }

    /// Whether a capability bit is set.
    pub fn has(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }

    /// This record carrying `wave` as its wave costs.
    pub fn with_wave(mut self, wave: WaveCostRecord) -> Self {
        self.wave_width = wave.width;
        self.wave_barrier_ns = wave.barrier_ns;
        self.wave_fixed_ns = wave.fixed_ns;
        self.wave_segment_ps = wave.segment_ps;
        self.wave_copy_ps_per_id = wave.copy_ps_per_id;
        self.wave_skew_ns = wave.skew_ns;
        self.wave_rebalance_fixed_ns = wave.rebalance_fixed_ns;
        self.wave_generation_ns = wave.generation_ns;
        self
    }

    /// The wave costs this record carries, when any have been recorded.
    pub fn wave(&self) -> Option<WaveCostRecord> {
        if self.wave_width == 0 {
            return None;
        }
        Some(WaveCostRecord {
            width: self.wave_width,
            barrier_ns: self.wave_barrier_ns,
            fixed_ns: self.wave_fixed_ns,
            segment_ps: self.wave_segment_ps,
            copy_ps_per_id: self.wave_copy_ps_per_id,
            skew_ns: self.wave_skew_ns,
            rebalance_fixed_ns: self.wave_rebalance_fixed_ns,
            generation_ns: self.wave_generation_ns,
        })
    }
}

/// The mapped file's header. Every multi-process field is atomic; the
/// rest are written once while the magic is still zero.
#[repr(C, align(64))]
struct Header {
    /// [`CALIBRATION_MAGIC`] once the region is complete, zero while it
    /// is being laid out.
    magic: u64,
    /// [`LAYOUT_VERSION`] of the writer.
    layout_version: u32,
    /// `size_of::<CpuCalibration>()`, so a reader refuses a region laid
    /// out by a build whose record is a different size.
    record_bytes: u32,
    /// SeqLock: odd while a writer is in the payload, even otherwise.
    seq_version: AtomicU32,
    /// Pid of the process measuring, or [`NO_WRITER`].
    writer_pid: AtomicU32,
    /// Raised by the writer before each probe. A holder that stops
    /// beating is taken over.
    heartbeat_epoch: AtomicU64,
    /// [`HostStamp::hash`] this table was measured for.
    stamp_hash: u64,
    /// Devices described in the accelerator array.
    accel_count: AtomicU32,
    _pad: [u8; 20],
}

/// Bytes the mapped region occupies.
const FILE_SIZE: usize = size_of::<Header>()
    + size_of::<CpuCalibration>()
    + MAX_ACCEL * size_of::<AccelCalibration>();

/// What went wrong reaching the table. Every variant leaves the caller
/// able to measure for itself, which is what it would have done without
/// a table at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    /// The file exists but was not laid out by this build.
    LayoutMismatch,
    /// The table belongs to a different host stamp.
    StampMismatch,
    /// Another process holds the writer lease and is beating.
    WriterActive,
    /// The filesystem refused.
    Io(io::ErrorKind),
}

impl From<io::Error> for StoreError {
    fn from(e: io::Error) -> Self {
        Self::Io(e.kind())
    }
}

/// The calibration directory, from `FLYNNEL_CALIBRATION_DIR` or the
/// per-user cache location. Not created here.
///
/// A test suite, a gate or anything else running while the host is busy
/// should point this at a throwaway path of its own. Such a run
/// measures under its own load, and the trust check passes more readily
/// for a contended draw, whose samples agree because they were all
/// slowed together, so the draw likeliest to be published is the one
/// least worth publishing. A directory per run keeps it out of the
/// shared table, and the record it leaves behind is what that run
/// routed on.
pub fn calibration_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("FLYNNEL_CALIBRATION_DIR") {
        return Some(PathBuf::from(dir));
    }
    #[cfg(windows)]
    {
        std::env::var_os("LOCALAPPDATA")
            .map(|d| PathBuf::from(d).join("flynnel").join("calibration"))
    }
    #[cfg(not(windows))]
    {
        if let Some(x) = std::env::var_os("XDG_CACHE_HOME") {
            return Some(PathBuf::from(x).join("flynnel").join("calibration"));
        }
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".cache").join("flynnel").join("calibration"))
    }
}

/// The table's path for a stamp.
pub fn table_path(dir: &Path, stamp: &HostStamp) -> PathBuf {
    dir.join(format!("flynnel-{:016x}.cal", stamp.hash()))
}

/// A mapped calibration table for one host stamp.
pub struct CalibrationStore {
    _file: File,
    mmap: MmapMut,
    stamp_hash: u64,
}

// SAFETY: every cross-process field in the header is atomic, and the
// payload is only read or written under the header's SeqLock. The mmap
// handle is Send + Sync per memmap2.
unsafe impl Send for CalibrationStore {}
unsafe impl Sync for CalibrationStore {}

impl CalibrationStore {
    /// Open the table for `stamp`, creating and laying it out when the
    /// path does not yet exist.
    ///
    /// A file laid out by a build with a different record size, or
    /// carrying another host's stamp, is refused rather than reused:
    /// the numbers in it describe something else.
    pub fn open_or_create(dir: &Path, stamp: &HostStamp) -> Result<Self, StoreError> {
        std::fs::create_dir_all(dir)?;
        let path = table_path(dir, stamp);
        let stamp_hash = stamp.hash();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let fresh = file.metadata()?.len() < FILE_SIZE as u64;
        if fresh {
            file.set_len(FILE_SIZE as u64)?;
        }
        // SAFETY: the file is at least FILE_SIZE bytes; the mapping is
        // owned by this struct and outlives every accessor below, none
        // of which forms a reference over a word another process
        // writes except through an atomic.
        let mut mmap = unsafe { MmapOptions::new().len(FILE_SIZE).map_mut(&file)? };

        if fresh {
            // SAFETY: the mapping is FILE_SIZE bytes and this process
            // just created the file, so no other process can be mid-
            // layout in it.
            unsafe { lay_out(mmap.as_mut_ptr(), stamp_hash) };
        } else {
            // SAFETY: the mapping is FILE_SIZE bytes and the header is
            // at offset zero.
            let hdr = unsafe { &*(mmap.as_ptr() as *const Header) };
            if hdr.magic != CALIBRATION_MAGIC
                || hdr.layout_version != LAYOUT_VERSION
                || hdr.record_bytes as usize != size_of::<CpuCalibration>()
            {
                return Err(StoreError::LayoutMismatch);
            }
            if hdr.stamp_hash != stamp_hash {
                return Err(StoreError::StampMismatch);
            }
        }
        Ok(Self { _file: file, mmap, stamp_hash })
    }

    /// The stamp hash this table was laid out for.
    pub fn stamp_hash(&self) -> u64 {
        self.stamp_hash
    }

    fn header(&self) -> &Header {
        // SAFETY: the mapping is FILE_SIZE bytes and the header sits at
        // offset zero, laid out by `lay_out` before the magic was
        // published.
        unsafe { &*(self.mmap.as_ptr() as *const Header) }
    }

    fn cpu_ptr(&self) -> *const CpuCalibration {
        // SAFETY: the CPU record follows the header within the mapping.
        unsafe { self.mmap.as_ptr().add(size_of::<Header>()) as *const CpuCalibration }
    }

    fn accel_ptr(&self, i: usize) -> *const AccelCalibration {
        let off = size_of::<Header>()
            + size_of::<CpuCalibration>()
            + i * size_of::<AccelCalibration>();
        // SAFETY: `i` is below MAX_ACCEL at every call site, so the
        // offset stays inside the mapping.
        unsafe { self.mmap.as_ptr().add(off) as *const AccelCalibration }
    }

    /// Read the CPU record and the devices, retrying while a writer is
    /// in the payload.
    ///
    /// `None` means the read never caught the payload between writers,
    /// not that the table is empty: a fresh table reads back as a
    /// zeroed record, which `is_trustworthy` refuses.
    pub fn read(&self) -> Option<(CpuCalibration, Vec<AccelCalibration>)> {
        let hdr = self.header();
        // A writer holds the payload for the length of one memcpy, so a
        // bounded retry is enough; a reader that loses this many races
        // measures for itself rather than spinning.
        for _ in 0..1024 {
            let before = hdr.seq_version.load(Ordering::Acquire);
            if before & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let n = (hdr.accel_count.load(Ordering::Acquire) as usize).min(MAX_ACCEL);
            // SAFETY: both records sit inside the mapping and are plain
            // data; the SeqLock check below decides whether what was
            // read is a consistent snapshot.
            let cpu = unsafe { std::ptr::read_volatile(self.cpu_ptr()) };
            let mut accel = Vec::with_capacity(n);
            for i in 0..n {
                accel.push(unsafe { std::ptr::read_volatile(self.accel_ptr(i)) });
            }
            fence(Ordering::Acquire);
            if hdr.seq_version.load(Ordering::Acquire) == before {
                // An empty CPU record is returned rather than hidden: a
                // table may hold devices and no host profile, and
                // reporting nothing would lose them. Whether either
                // half is worth using is `is_trustworthy`'s answer, not
                // this one's.
                return Some((cpu, accel));
            }
        }
        None
    }

    /// Take the writer lease, or report who holds it.
    ///
    /// A holder whose heartbeat has not advanced within
    /// [`LEASE_GRACE_EPOCHS`] is taken over: it died mid-measurement,
    /// and without the takeover no later start on this host would ever
    /// calibrate.
    pub fn try_acquire_writer(&self) -> Result<WriterGuard<'_>, StoreError> {
        let hdr = self.header();
        let me = std::process::id();
        loop {
            let holder = hdr.writer_pid.load(Ordering::Acquire);
            if holder == me {
                return Ok(WriterGuard { store: self });
            }
            if holder != NO_WRITER {
                let first = hdr.heartbeat_epoch.load(Ordering::Acquire);
                // One tick of our own gives a live holder time to beat.
                std::thread::yield_now();
                let second = hdr.heartbeat_epoch.load(Ordering::Acquire);
                if second.wrapping_sub(first) > 0 || second == 0 {
                    return Err(StoreError::WriterActive);
                }
                if second.wrapping_sub(first) < LEASE_GRACE_EPOCHS
                    && hdr.writer_pid.load(Ordering::Acquire) == holder
                {
                    // The holder is quiet. Take it only by winning the
                    // same compare-exchange every other waiter races,
                    // so exactly one process succeeds.
                    if hdr
                        .writer_pid
                        .compare_exchange(holder, me, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        return Ok(WriterGuard { store: self });
                    }
                    continue;
                }
                return Err(StoreError::WriterActive);
            }
            if hdr
                .writer_pid
                .compare_exchange(NO_WRITER, me, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(WriterGuard { store: self });
            }
        }
    }
}

/// Holds the writer lease for as long as it lives. Dropping it releases
/// the lease, including while a panic unwinds, so a failed measurement
/// does not strand the next process.
pub struct WriterGuard<'a> {
    store: &'a CalibrationStore,
}

/// Whether a stored record should stand against one being offered, and
/// `None` where it should not.
///
/// The cheaper dispatch cost stands. Load only ever adds time, so the
/// smallest cost seen for a stamp is the quietest observation of that
/// host, and the table converges on it as more draws arrive. A tie goes
/// to the offer, since the fresher draw describes the host now.
///
/// Measured on a 12-core host, 16 draws at each level: dispatch costs
/// 1300 to 1500 ns idle against 3.2 to 7.0 million saturated. The two
/// conditions do not overlap, so the ordering needs no threshold and no
/// measure of spread.
///
/// The known weakness is that one spuriously fast reading stands for
/// the life of the stamp, because nothing later can be cheaper than a
/// wrong floor. Raising [`LAYOUT_VERSION`] is what displaces it.
///
/// Separate from the write so the decision can be exercised without a
/// mapped file, a writer lease, or the process-wide switch that gates
/// it: a switch read once per process cannot be moved between tests, so
/// a test that needed it on would fix the value for every test after it.
/// Per-mille disagreement between two draws of the same host, on each
/// of the three dispatch figures, in the order dispatch, collapse,
/// wake.
///
/// Taken against the pair's mean rather than against either record, so
/// the figure does not depend on which of the two is called the
/// incumbent. A figure both draws report as zero disagrees by zero.
///
/// [`confirmations_after`] decides on the dispatch figure. The other
/// two are reported beside it, so a pair that agrees on dispatch and
/// parts company on the thresholds derived from it stays visible.
fn two_draw_disagreement_per_mille(a: &CpuCalibration, b: &CpuCalibration) -> [u64; 3] {
    fn gap(x: u64, y: u64) -> u64 {
        let sum = x.saturating_add(y);
        if sum == 0 {
            return 0;
        }
        x.abs_diff(y).saturating_mul(2_000) / sum
    }
    [
        gap(a.dispatch_cost_ns, b.dispatch_cost_ns),
        gap(a.collapse_threshold_ns, b.collapse_threshold_ns),
        gap(a.jec_wake_threshold_ns, b.jec_wake_threshold_ns),
    ]
}

/// Agreements the record that ends up stored should carry.
///
/// Medians within [`PROVISIONAL_SPREAD_PER_MILLE`] raise the count,
/// whichever record is kept. Otherwise a dearer offer leaves the count
/// alone, since load only adds time, and a cheaper one resets it to
/// zero. A zeroed table is not a draw to agree with.
fn confirmations_after(incumbent: &CpuCalibration, offered: &CpuCalibration) -> u32 {
    if incumbent.samples == 0 {
        return 0;
    }
    let agreed = two_draw_disagreement_per_mille(incumbent, offered)[0]
        <= PROVISIONAL_SPREAD_PER_MILLE as u64;
    if agreed {
        incumbent.confirmations.saturating_add(1)
    } else if incumbent.dispatch_cost_ns < offered.dispatch_cost_ns {
        incumbent.confirmations
    } else {
        0
    }
}

fn prefers_incumbent(
    incumbent: &CpuCalibration,
    offered: &CpuCalibration,
) -> Option<PublishOutcome> {
    // A table nobody has published reads back zeroed, and a zero
    // dispatch cost would win every comparison forever.
    if incumbent.samples == 0 {
        return None;
    }
    let (held, new) = (incumbent.dispatch_cost_ns, offered.dispatch_cost_ns);
    (held < new).then_some(PublishOutcome::KeptIncumbent {
        incumbent: held,
        offered: new,
        confirmations: confirmations_after(incumbent, offered),
    })
}

/// What [`WriterGuard::publish_if_better`] did with the record it was
/// offered.
///
/// A refusal is reported rather than silent. A writer that measured,
/// found the table already better served and wrote nothing has done
/// something worth saying; returning unit would make that identical to
/// having published.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PublishOutcome {
    /// The record was written.
    Published {
        /// Agreements it carries. Zero is a provisional record: stored
        /// for the next draw to agree with, and not served.
        confirmations: u32,
    },
    /// The table held a cheaper dispatch cost, and it stands. Both
    /// figures are nanoseconds.
    KeptIncumbent {
        /// The stored record's dispatch cost.
        incumbent: u64,
        /// This draw's dispatch cost.
        offered: u64,
        /// Agreements the stored record carries once this draw is
        /// counted. A draw that is refused on cost can still confirm.
        confirmations: u32,
    },
}

impl WriterGuard<'_> {
    /// Raise the heartbeat. A writer calls this before each probe so a
    /// starting process can tell a slow measurement from a dead one.
    pub fn beat(&self) {
        self.store
            .header()
            .heartbeat_epoch
            .fetch_add(1, Ordering::Release);
    }

    /// Publish unless the table already holds a cheaper dispatch cost.
    ///
    /// An ordering between two records, not a threshold on either. No
    /// cost is called good or bad and no cutoff separates a contended
    /// draw from a clean one, because the share of free cores is a
    /// continuous property of a host rather than a state it is in. Two
    /// records can still be compared without any of that being settled.
    ///
    /// Load only ever adds time, so the cheapest cost seen for a stamp
    /// is the quietest observation of that host and the table converges
    /// on it. A tie publishes, since the fresher draw describes the host
    /// now. A table nobody has published reads back zeroed and is not
    /// allowed to defend its zero.
    ///
    /// Neither the spread nor the occupancy enters this. Both were
    /// measured across three load levels: the spread moves 1.3x between
    /// an idle host and a saturated one and the interquartile range
    /// 1.1x, while the dispatch cost moves by a factor of thousands.
    /// Every dispersion figure tried divides by the median, and the
    /// difference between the conditions is in the median.
    pub fn publish_if_better(
        &self,
        cpu: &CpuCalibration,
        accel: &[AccelCalibration],
    ) -> PublishOutcome {
        // Reported whether or not the refusal is on, because this is the
        // only point in a running process where two draws of one host
        // are both in hand. Reporting only when the refusal fires would
        // sample the half where the incumbent already won.
        // `samples > 0` because a table that has never been published
        // reads back zeroed, and a zeroed incumbent is not a draw. The
        // gap against it is the formula's maximum on every figure, so a
        // run against fresh directories reported 2000 per mille
        // forty-eight times and looked like a host that agrees with
        // itself about nothing.
        if std::env::var_os("FLYNNEL_OCCUPANCY").is_some()
            && let Some((incumbent, _)) = self.store.read()
            && incumbent.samples > 0
        {
            let [dispatch, collapse, wake] = two_draw_disagreement_per_mille(&incumbent, cpu);
            // The incumbent is the cheapest cost seen for this stamp, so
            // it is the host's known floor rather than merely another
            // draw, and what this run cost against that floor is the
            // load reading the campaign spent its length looking for. It
            // needs no bound: on a 12-core host a saturated draw reads
            // some thousands of times its own floor.
            //
            // The symmetric figure stays beside it because that is what
            // profile_draws_report computes over arbitrary pairs, and a
            // production line that cannot be read against the sweep is
            // two numbers rather than one measurement.
            let floor = incumbent.dispatch_cost_ns.max(1);
            let times_floor = cpu.dispatch_cost_ns as f64 / floor as f64;
            eprintln!(
                "flynnel: this draw costs {times_floor:.1}x the cheapest seen for this stamp \
                 ({} against {} ns); two draws disagree by {dispatch},{collapse},{wake} per \
                 mille on dispatch,collapse,wake; incumbent spread {} occupancy {:?}, \
                 offered spread {} occupancy {:?}",
                cpu.dispatch_cost_ns,
                incumbent.dispatch_cost_ns,
                incumbent.spread_per_mille,
                incumbent.occupancy(),
                cpu.spread_per_mille,
                cpu.occupancy(),
            );
        }
        if crate::sched::levers::calibration_refusal()
            && let Some((held, held_accel)) = self.store.read()
            && let Some(kept) = prefers_incumbent(&held, cpu)
        {
            // A draw refused on cost can still confirm, so a raised
            // count is written back. It carries the stored record's own
            // accelerators: `publish` takes the device count from what
            // it is handed.
            if let PublishOutcome::KeptIncumbent { confirmations, .. } = kept
                && confirmations != held.confirmations
            {
                let mut raised = held;
                raised.confirmations = confirmations;
                self.publish(&raised, &held_accel);
            }
            return kept;
        }
        // Computed whether or not the refusal is on: that lever decides
        // which record stands, this decides whether it has been seen to
        // reproduce.
        let mut record = *cpu;
        record.confirmations = match self.store.read() {
            Some((held, _)) => confirmations_after(&held, cpu),
            None => 0,
        };
        self.publish(&record, accel);
        PublishOutcome::Published { confirmations: record.confirmations }
    }

    /// Publish a measurement under the SeqLock.
    ///
    /// The version goes odd, the payload is written, then it goes to
    /// the next even value, so a concurrent reader either takes the
    /// whole previous record or the whole new one.
    pub fn publish(&self, cpu: &CpuCalibration, accel: &[AccelCalibration]) {
        let hdr = self.store.header();
        let n = accel.len().min(MAX_ACCEL);
        hdr.seq_version.fetch_add(1, Ordering::AcqRel);
        fence(Ordering::Release);
        // SAFETY: the lease makes this the only writer, and both
        // records sit inside the mapping. Readers are excluded by the
        // odd SeqLock version rather than by exclusion of the mapping.
        unsafe {
            let cpu_dst = self.store.cpu_ptr() as *mut CpuCalibration;
            std::ptr::write_volatile(cpu_dst, *cpu);
            for (i, a) in accel.iter().take(n).enumerate() {
                let dst = self.store.accel_ptr(i) as *mut AccelCalibration;
                std::ptr::write_volatile(dst, *a);
            }
        }
        hdr.accel_count.store(n as u32, Ordering::Release);
        fence(Ordering::Release);
        hdr.seq_version.fetch_add(1, Ordering::AcqRel);
    }
}

impl Drop for WriterGuard<'_> {
    fn drop(&mut self) {
        let hdr = self.store.header();
        let me = std::process::id();
        // Release only this process's own claim. A lease taken from us
        // as stale belongs to whoever took it, and clearing that would
        // hand a third process the same table to write at the same
        // time.
        if let Err(holder) =
            hdr.writer_pid
                .compare_exchange(me, NO_WRITER, Ordering::AcqRel, Ordering::Acquire)
        {
            eprintln!(
                "flynnel: the calibration lease held by pid {me} was taken by pid {holder} \
                 mid-measurement; leaving it with the new holder"
            );
        }
    }
}

/// Write the header and a zeroed payload, then publish the magic.
///
/// The magic goes last because it is what an attacher checks: a region
/// carrying the magic with an unwritten payload is one another process
/// would read as real.
///
/// # Safety
/// `ptr` addresses at least [`FILE_SIZE`] writable bytes that no other
/// process is laying out.
unsafe fn lay_out(ptr: *mut u8, stamp_hash: u64) {
    unsafe {
        let hdr = ptr as *mut Header;
        std::ptr::write(
            hdr,
            Header {
                magic: 0,
                layout_version: LAYOUT_VERSION,
                record_bytes: size_of::<CpuCalibration>() as u32,
                seq_version: AtomicU32::new(0),
                writer_pid: AtomicU32::new(NO_WRITER),
                heartbeat_epoch: AtomicU64::new(0),
                stamp_hash,
                accel_count: AtomicU32::new(0),
                _pad: [0; 20],
            },
        );
        let cpu = ptr.add(size_of::<Header>()) as *mut CpuCalibration;
        std::ptr::write(cpu, CpuCalibration::default());
        for i in 0..MAX_ACCEL {
            let off = size_of::<Header>()
                + size_of::<CpuCalibration>()
                + i * size_of::<AccelCalibration>();
            std::ptr::write(ptr.add(off) as *mut AccelCalibration, AccelCalibration::default());
        }
        fence(Ordering::Release);
        std::ptr::write_volatile(std::ptr::addr_of_mut!((*hdr).magic), CALIBRATION_MAGIC);
    }
}

/// Seconds since the Unix epoch, or zero when the clock is before it.
pub fn now_unix_s() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Remove a test's directory, saying so when something other than
    /// its absence stopped it: a table left behind is one the next run
    /// of this test would attach to instead of creating.
    fn cleanup(dir: &Path) {
        match std::fs::remove_dir_all(dir) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => eprintln!("calibration test left {} behind: {e}", dir.display()),
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("flynnel-cal-test-{name}-{}", std::process::id()));
        cleanup(&d);
        d
    }

    fn stamp(primary: u32, total: u32) -> HostStamp {
        HostStamp {
            vendor: "TestVendor".to_string(),
            cpuid_signature: 0x000A_0671,
            arch: "test-arch",
            os: "test-os",
            primary_workers: primary,
            total_workers: total,
            layout_version: LAYOUT_VERSION,
        }
    }

    fn sample_cpu() -> CpuCalibration {
        CpuCalibration {
            dispatch_cost_ns: 1_000,
            collapse_threshold_ns: 70_000,
            jec_wake_threshold_ns: 40_000,
            measured_unix_s: now_unix_s(),
            spread_per_mille: 41,
            samples: 9,
            occupancy_per_mille: 970,
            confirmations: 1,
            _pad: [0; 16],
        }
    }

    /// A draw another draw has already agreed with.
    fn confirmed(mut cpu: CpuCalibration) -> CpuCalibration {
        cpu.confirmations = 1;
        cpu
    }

    #[test]
    fn a_record_from_a_platform_with_no_thread_clock_reports_no_occupancy() {
        // Zero is the share a thread that never reached a core reports,
        // so an unmeasured record storing zero would sort beneath one
        // drawn on a saturated host.
        let unmeasured =
            CpuCalibration::new(1_000, 70_000, 40_000, 41, 9, None);
        assert_eq!(unmeasured.occupancy(), None);
        assert_eq!(unmeasured.occupancy_per_mille, OCCUPANCY_UNRECORDED);

        let starved = CpuCalibration::new(1_000, 70_000, 40_000, 41, 9, Some(0));
        assert_eq!(starved.occupancy(), Some(0));
    }

    #[test]
    fn occupancy_is_provenance_and_gates_nothing() {
        // The share of free cores is a continuous property of the host
        // rather than a state it is in, so no cutoff separates a
        // contended draw from a clean one. The figure records what a
        // draw ran under and decides nothing on its own; a reader
        // holding two records can prefer the better-drawn one.
        let quiet = confirmed(CpuCalibration::new(1_000, 70_000, 40_000, 41, 9, Some(990)));
        let loaded = confirmed(CpuCalibration::new(1_000, 70_000, 40_000, 41, 9, Some(210)));
        assert!(quiet.is_trustworthy());
        assert!(loaded.is_trustworthy());
        assert!(quiet.occupancy() > loaded.occupancy());
    }

    /// Publish `first`, then offer `second`, and report what happened
    /// along with what the table holds afterward.
    fn offer_after(
        tag: &str,
        first: CpuCalibration,
        second: CpuCalibration,
    ) -> (PublishOutcome, CpuCalibration, PathBuf) {
        let dir = temp_dir(tag);
        let s = stamp(12, 24);
        let store = CalibrationStore::open_or_create(&dir, &s).expect("create");
        {
            let w = store.try_acquire_writer().expect("no other writer");
            w.publish(&first, &[]);
        }
        let outcome = {
            let w = store.try_acquire_writer().expect("still the only writer");
            w.publish_if_better(&second, &[])
        };
        let (held, _) = store.read().expect("a record is present either way");
        (outcome, held, dir)
    }

    /// The decision alone, with no file, no lease and no process-wide
    /// switch: those are the write, and this is what to write.
    #[test]
    fn a_cheaper_draw_displaces_a_more_expensive_one() {
        let loaded = CpuCalibration::new(9_000, 70_000, 40_000, 41, 9, Some(210));
        let quiet = CpuCalibration::new(1_000, 70_000, 40_000, 41, 9, Some(990));
        assert_eq!(prefers_incumbent(&loaded, &quiet), None, "the cheaper draw wins");
    }

    #[test]
    fn a_more_expensive_draw_does_not_displace_a_cheaper_one() {
        // An ordering between two records, not a cutoff on either: 9000
        // is not called bad, it is called dearer than 1000.
        let quiet = CpuCalibration::new(1_000, 70_000, 40_000, 41, 9, Some(990));
        let loaded = CpuCalibration::new(9_000, 70_000, 40_000, 41, 9, Some(210));
        assert_eq!(
            prefers_incumbent(&quiet, &loaded),
            Some(PublishOutcome::KeptIncumbent {
                incumbent: 1_000,
                offered: 9_000,
                confirmations: 0,
            }),
            "the refusal names both costs so a caller can say why"
        );
    }

    #[test]
    fn a_scattered_incumbent_stands_while_it_is_the_cheapest_seen() {
        // The spread no longer participates. It was measured across
        // three load levels and moves 1.3x between an idle host and a
        // saturated one, so it cannot order two records; the cost moves
        // by thousands. A record whose samples disagreed is still the
        // cheapest observation of this host until something cheaper
        // arrives.
        let scattered = CpuCalibration::new(
            1_000,
            70_000,
            40_000,
            PROVISIONAL_SPREAD_PER_MILLE + 1,
            9,
            Some(990),
        );
        assert!(!scattered.is_trustworthy(), "nothing has agreed with it");
        let dearer = CpuCalibration::new(9_000, 70_000, 40_000, 41, 9, Some(210));
        assert_eq!(
            prefers_incumbent(&scattered, &dearer),
            Some(PublishOutcome::KeptIncumbent {
                incumbent: 1_000,
                offered: 9_000,
                confirmations: 0,
            }),
            "cost orders them, and the incumbent's spread does not enter it"
        );
    }

    #[test]
    fn a_record_with_no_occupancy_is_ordered_like_any_other() {
        // Occupancy no longer orders anything, so a platform with no
        // thread clock is not a special case and cannot freeze or lose
        // the table. Both directions are asserted, because a rule that
        // read the absent field would fail in exactly one of them.
        let dear = CpuCalibration::new(9_000, 70_000, 40_000, 41, 9, Some(990));
        let cheap_unmeasured = CpuCalibration::new(1_000, 70_000, 40_000, 41, 9, None);
        assert_eq!(
            prefers_incumbent(&cheap_unmeasured, &dear),
            Some(PublishOutcome::KeptIncumbent {
                incumbent: 1_000,
                offered: 9_000,
                confirmations: 0,
            }),
            "a cheaper incumbent stands without an occupancy of its own"
        );
        assert_eq!(
            prefers_incumbent(&dear, &cheap_unmeasured),
            None,
            "and a cheaper offer displaces without one either"
        );
    }

    #[test]
    fn an_equal_cost_publishes_so_a_fresher_measurement_wins_a_tie() {
        // Equal costs order nothing, and the newer measurement has the
        // better claim to describe the host now.
        let first = CpuCalibration::new(4_000, 70_000, 40_000, 41, 9, Some(700));
        let second = CpuCalibration::new(4_000, 12_000, 9_000, 41, 9, Some(700));
        assert_eq!(prefers_incumbent(&first, &second), None);
    }

    #[test]
    fn a_table_nobody_published_never_wins_on_its_zero() {
        // A zeroed record reads back from a fresh table, and zero is
        // cheaper than any measurement, so without this guard the first
        // comparison against a new table would keep the empty record
        // permanently.
        let empty = CpuCalibration::new(0, 0, 0, 0, 0, None);
        let measured = CpuCalibration::new(4_000, 70_000, 40_000, 41, 9, Some(990));
        assert_eq!(
            prefers_incumbent(&empty, &measured),
            None,
            "an unpublished table has no draw to defend"
        );
    }

    #[test]
    fn the_only_incumbent_the_caller_delivers_is_one_the_refusal_declines() {
        // Holds at the shipped default of no age bound. The caller in
        // par_iter returns the stored record whenever it clears
        // `is_trustworthy` and only measures and offers when it does
        // not, so the incumbent arriving here has failed that check,
        // and `prefers_incumbent` returns `None` for exactly those.
        //
        // One half now, not two. The guard used to reject an
        // untrustworthy incumbent on its first line, which was the other
        // half of the contradiction; cost orders the two records without
        // consulting the spread, so that half is gone and only the
        // caller's early return keeps the comparison unreached.
        // `an_aged_incumbent_is_ordered_rather_than_dismissed` covers the
        // case a configured bound opens.
        let passes = confirmed(CpuCalibration::new(1_000, 70_000, 40_000, 41, 9, Some(990)));
        assert!(
            passes.is_trustworthy(),
            "a record like this is returned by the caller and never offered"
        );
    }

    #[test]
    fn two_draws_disagree_by_one_figure_whichever_is_called_the_incumbent() {
        // Symmetry is the property that makes this reportable at the
        // publish site: the pair there is ordered by which draw happened
        // to be stored first, and a figure that changed with that order
        // would describe the order rather than the host.
        let quiet = CpuCalibration::new(1_000, 4_000, 4_200, 41, 9, Some(990));
        let loaded = CpuCalibration::new(2_000, 20_000, 6_500, 41, 9, Some(660));

        let forward = two_draw_disagreement_per_mille(&quiet, &loaded);
        let backward = two_draw_disagreement_per_mille(&loaded, &quiet);
        assert_eq!(forward, backward, "the figure must not depend on the argument order");

        // Against the pair's mean: 1000 apart on a mean of 1500 is 666,
        // 16000 on 12000 is 1333, 2300 on 5350 is 429.
        assert_eq!(forward, [666, 1333, 429]);

        let same = two_draw_disagreement_per_mille(&quiet, &quiet);
        assert_eq!(same, [0, 0, 0], "a draw against itself disagrees by nothing");

        // Both figures zero is agreement, not a division this cannot do.
        let empty = CpuCalibration::new(0, 0, 0, 41, 9, Some(990));
        assert_eq!(two_draw_disagreement_per_mille(&empty, &empty), [0, 0, 0]);
    }

    #[test]
    fn an_aged_incumbent_is_ordered_rather_than_dismissed() {
        // With `FLYNNEL_CALIBRATION_MAX_AGE_S` set, a record that clears
        // `is_trustworthy` but is older than the bound falls past the
        // caller's early return, so the incumbent reaching the guard is
        // trustworthy. The guard's first line stops discarding it and
        // the ordering it implements decides.
        //
        // Asserted against the same pair as the default case, so the
        // difference between the two tests is the incumbent's trust and
        // nothing else.
        let aged = confirmed(CpuCalibration::new(1_000, 70_000, 40_000, 41, 9, Some(990)));
        assert!(aged.is_trustworthy(), "an aged record is drawn again, not distrusted");

        let offered = CpuCalibration::new(9_000, 70_000, 40_000, 41, 9, Some(210));
        assert_eq!(
            prefers_incumbent(&aged, &offered),
            Some(PublishOutcome::KeptIncumbent {
                incumbent: 1_000,
                offered: 9_000,
                confirmations: 1,
            }),
            "the cheaper of the two stands, keeping its agreement against a dearer offer"
        );
    }

    #[test]
    fn the_write_path_publishes_while_the_switch_is_off() {
        // The shipped default. The decision above is exercised directly;
        // this checks the path that gates it, so a switch left on by a
        // careless edit fails a test rather than a release.
        let quiet = CpuCalibration::new(1_000, 70_000, 40_000, 41, 9, Some(990));
        let loaded = CpuCalibration::new(9_000, 70_000, 40_000, 41, 9, Some(210));
        let (outcome, held, dir) = offer_after("switch-off", quiet, loaded);
        if crate::sched::levers::calibration_refusal() {
            assert_eq!(
                outcome,
                PublishOutcome::KeptIncumbent {
                    incumbent: 1_000,
                    offered: 9_000,
                    confirmations: 0,
                },
                "the switch is on in this process, so the refusal must fire"
            );
            assert_eq!(held.dispatch_cost_ns, 1_000);
        } else {
            assert_eq!(
                outcome,
                PublishOutcome::Published { confirmations: 0 },
                "off publishes unconditionally, and the two costs did not agree"
            );
            assert_eq!(held.dispatch_cost_ns, 9_000, "the offer is what stands");
        }
        cleanup(&dir);
    }

    #[test]
    fn a_fresh_table_offers_a_record_nothing_will_use() {
        let dir = temp_dir("fresh");
        let s = stamp(12, 24);
        let store = CalibrationStore::open_or_create(&dir, &s).expect("create");
        let (cpu, accel) = store.read().expect("a fresh table reads back");
        assert_eq!(cpu.samples, 0, "nothing has been measured into it");
        assert!(
            !cpu.is_trustworthy(),
            "so a caller must measure rather than take this"
        );
        assert!(accel.is_empty(), "and it describes no devices");
        cleanup(&dir);
    }

    #[test]
    fn a_published_record_reads_back_whole() {
        let dir = temp_dir("roundtrip");
        let s = stamp(12, 24);
        let store = CalibrationStore::open_or_create(&dir, &s).expect("create");
        let cpu = sample_cpu();
        let accel = [AccelCalibration::new(
            AccelKind::GpuPeer,
            0,
            89,
            46,
            2_505_000,
            12_282,
            ACCEL_DOORBELL_OK | ACCEL_TIMED_LOCK_OK,
            3_000,
            3_400,
            3_700,
            1_970,
            120,
            19_700,
            41_000,
        )];
        {
            let w = store.try_acquire_writer().expect("no other writer");
            w.beat();
            w.publish(&cpu, &accel);
        }
        let (got_cpu, got_accel) = store.read().expect("a published record reads back");
        assert_eq!(got_cpu, cpu, "the CPU record survives the round trip");
        assert_eq!(got_accel.len(), 1, "one device was published");
        assert_eq!(got_accel[0], accel[0], "the device record survives too");
        assert!(got_cpu.is_trustworthy(), "41 per mille is a quiet host");
        assert!(
            got_accel[0].is_trustworthy(),
            "a round trip of 3000/3400/3700 is a device nothing else is resident on"
        );
        assert!(
            got_accel[0].has(ACCEL_DOORBELL_OK) && got_accel[0].has(ACCEL_TIMED_LOCK_OK),
            "the capability bits survive the round trip"
        );
        assert!(
            !got_accel[0].has(ACCEL_SYS_ATOMICS_OK),
            "a bit that was not set does not read as set"
        );
        cleanup(&dir);
    }

    /// A device carrying someone else's work stretches the tail of the
    /// round trip without moving its minimum, and that is what makes
    /// the record provisional rather than the host's calibration.
    #[test]
    fn a_device_timed_under_load_is_not_trustworthy() {
        let quiet = AccelCalibration::new(
            AccelKind::GpuPeer, 0, 89, 46, 2_505_000, 12_282,
            ACCEL_DOORBELL_OK, 3_000, 3_400, 3_700, 1_970, 120, 19_700, 41_000,
        );
        let loaded = AccelCalibration::new(
            AccelKind::GpuPeer, 0, 89, 46, 2_505_000, 12_282,
            ACCEL_DOORBELL_OK, 3_000, 3_400, 9_400, 4_820, 120, 48_200, 41_000,
        );
        assert!(quiet.is_trustworthy(), "a tight distribution describes the device");
        assert!(
            !loaded.is_trustworthy(),
            "a p99 nearly three times the minimum describes what else was running"
        );
        assert!(
            !AccelCalibration::default().is_trustworthy(),
            "an empty slot describes no device at all"
        );
    }

    #[test]
    fn wave_costs_ride_the_device_record_through_a_round_trip() {
        let dir = temp_dir("wave");
        let s = stamp(12, 24);
        let store = CalibrationStore::open_or_create(&dir, &s).expect("create");
        let wave = WaveCostRecord {
            width: 32,
            barrier_ns: 2_100,
            fixed_ns: 180_000,
            segment_ps: 450,
            copy_ps_per_id: 1_900,
            skew_ns: 490_000,
            rebalance_fixed_ns: 5_200,
            generation_ns: 1_800,
        };
        let device = AccelCalibration::new(
            AccelKind::GpuPeer, 0, 120, 48, 2_505_000, 12_227,
            ACCEL_DOORBELL_OK, 3_000, 3_400, 3_700, 1_970, 120, 19_700, 41_000,
        );
        assert_eq!(device.wave(), None, "a device record starts with no wave costs");
        let carried = device.with_wave(wave);
        {
            let w = store.try_acquire_writer().expect("no other writer");
            w.publish(&sample_cpu(), &[carried]);
        }
        let (_cpu, accel) = store.read().expect("a published record reads back");
        assert_eq!(accel[0].wave(), Some(wave), "the wave costs survive the round trip");
        assert_eq!(accel[0].rtt_median_ns, 3_400, "and leave the device timings as they were");
        cleanup(&dir);
    }

    #[test]
    fn a_second_process_reads_what_the_first_published() {
        let dir = temp_dir("reattach");
        let s = stamp(12, 24);
        let cpu = sample_cpu();
        {
            let store = CalibrationStore::open_or_create(&dir, &s).expect("create");
            let w = store.try_acquire_writer().expect("no other writer");
            w.publish(&cpu, &[]);
        }
        let reopened = CalibrationStore::open_or_create(&dir, &s).expect("attach");
        let (got, accel) = reopened.read().expect("the record persists across handles");
        assert_eq!(got, cpu, "a later attach reads the published record");
        assert!(accel.is_empty(), "no devices were published");
        cleanup(&dir);
    }

    #[test]
    fn a_different_stamp_is_a_different_table() {
        let dir = temp_dir("stamp");
        let a = stamp(12, 24);
        let b = stamp(8, 16);
        assert_ne!(a.hash(), b.hash(), "core counts change the stamp");
        assert_ne!(
            table_path(&dir, &a),
            table_path(&dir, &b),
            "a different stamp names a different file, so neither reads the other's numbers"
        );
        cleanup(&dir);
    }

    #[test]
    fn the_lease_releases_on_drop_and_is_retakeable() {
        let dir = temp_dir("lease");
        let s = stamp(12, 24);
        let store = CalibrationStore::open_or_create(&dir, &s).expect("create");
        {
            let _w = store.try_acquire_writer().expect("free lease");
        }
        let _again = store
            .try_acquire_writer()
            .expect("a dropped guard leaves the lease free");
        cleanup(&dir);
    }

    #[test]
    fn the_provisional_threshold_separates_a_quiet_host_from_a_loaded_one() {
        use crate::sched::par_iter::sample_spread_per_mille;
        // A calibration sweep on an idle host: 68200 to 71000 around a
        // median of 69600.
        assert!(
            sample_spread_per_mille(&[68_200, 69_600, 71_000]) < PROVISIONAL_SPREAD_PER_MILLE,
            "a four percent spread is a host whose numbers describe the machine"
        );
        // The same sweep on a host carrying other work: 42400 to 68600.
        assert!(
            sample_spread_per_mille(&[42_400, 55_000, 68_600]) > PROVISIONAL_SPREAD_PER_MILLE,
            "a forty-eight percent spread is a host describing its load"
        );
    }

    #[test]
    fn trust_follows_agreement_between_draws_not_the_spread_within_one() {
        // Both directions, because a rule reading the spread would pass
        // one of them on its own.
        let mut scattered = sample_cpu();
        scattered.spread_per_mille = PROVISIONAL_SPREAD_PER_MILLE * 20;
        assert!(
            scattered.is_trustworthy(),
            "a confirmed record stands however far its own nine samples scattered"
        );

        let mut tight = sample_cpu();
        tight.spread_per_mille = 0;
        tight.confirmations = 0;
        assert!(
            !tight.is_trustworthy(),
            "and one nothing has agreed with does not, however tightly it sampled"
        );
    }

    #[test]
    fn a_draw_confirms_the_stored_record_when_their_medians_agree() {
        let held = CpuCalibration::new(1_300, 70_000, 40_000, 41, 9, Some(990));
        // 1300 against 1400 is 74 per mille apart, the median of twenty
        // pairs measured on a quiet 12-core host.
        let near = CpuCalibration::new(1_400, 70_000, 40_000, 41, 9, Some(990));
        assert_eq!(confirmations_after(&held, &near), 1);

        let far = CpuCalibration::new(9_000, 70_000, 40_000, 41, 9, Some(210));
        assert_eq!(
            confirmations_after(&held, &far),
            0,
            "a dearer draw that disagrees leaves the count where it was"
        );

        let cheaper_and_apart = CpuCalibration::new(200, 70_000, 40_000, 41, 9, Some(990));
        assert_eq!(
            confirmations_after(&confirmed(held), &cheaper_and_apart),
            0,
            "a cheaper draw that disagrees replaces and starts over"
        );
        assert_eq!(
            confirmations_after(&confirmed(held), &far),
            1,
            "while a dearer one keeps the agreement already earned"
        );
    }
}

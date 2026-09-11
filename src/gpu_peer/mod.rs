//! GPU-peer substrate: the GPU joins the scheduler as a shared-memory
//! peer over a CUDA-registered memory-mapped file.
//!
//! One physical region is simultaneously (a) a plain mapped file this
//! process and any other process can open, and (b) device-visible
//! memory a resident GPU kernel polls and writes. Work flows through
//! Lamport single-producer/single-consumer lanes with doorbell
//! signalling - no kernel launch on the per-message path, no atomics
//! across the CPU/GPU boundary (measured unsafe over PCIe hosts
//! without native atomics), and no data copies besides the payload
//! writes themselves.
//!
//! Every timing constant is host-calibrated at [`GpuPeer::init`]:
//! doorbell round-trip, cross-device clock error, the Fischer
//! timed-lock margin (validated by a live contention self-test), the
//! launch baseline, and the system-atomics capability flag. Nothing
//! is baked from a reference machine; a host with a coherent CPU-GPU
//! link measures tighter constants and unlocks more capability
//! automatically. See [`PeerCalibration`].
//!
//! The consumer runs as a bounded-quantum persistent kernel
//! (watchdog-safe on display GPUs) that parks when idle and costs one
//! launch to wake; a continuously fed queue never pays the wake cost
//! (see the poller module docs).
//!
//! # What the peer does to a caller's CUDA state
//!
//! Streams: the peer creates every stream it uses - one per lane, one
//! for wide launches - and never makes a caller's stream current. Work
//! a consumer enqueues on [`GpuPeer::wide_stream`] is FIFO-ordered with
//! the peer's own wide launches; everything else it owns is separate.
//!
//! Context: the peer operates on the device primary context, and makes
//! it current on whatever thread reaches one of its bind points. That
//! matters to a consumer that built its own context - cudarc's
//! `new_non_primary` is how one is obtained - because such a context is
//! replaced on the calling thread, after which the consumer's launches
//! run on the primary context instead of the one it built.
//! [`GpuPeer::init`] reports this, to stderr and through
//! [`GpuPeer::displaced_foreign_context`], so a consumer can fall back
//! rather than discover it at a launch far from the cause. A consumer
//! holding the primary context, which is what `CudaContext::new`
//! returns, shares it with the peer and is unaffected.
//!
//! Device-wide state: [`l2_persist::L2Persist`] resets the persisting-L2
//! window and limit on drop. Those are context-wide rather than
//! stream-local, so a consumer that sets its own persisting-L2 window
//! will find it cleared.
//!
//! # Slot capacity
//!
//! A lane slot carries [`Geometry::payload_max`] payload bytes, and
//! both directions are bounded by that one figure: submitting more is
//! refused with [`GpuPeerError::PayloadTooLarge`], and so is reading
//! more, because the bytes past a slot belong to the next one. Read
//! that figure from the running geometry rather than deriving it from a
//! slot size.

pub mod calibration;
pub mod group;
pub mod hybrid;
pub mod l2_persist;
pub mod lanes;
pub mod linalg;
pub mod ozaki;
pub mod layout;
pub mod region;
pub mod timed_lock;
pub mod vram;
pub mod watchdog;
pub mod wave;

mod poller;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::sys as cu;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub use calibration::PeerCalibration;
pub use lanes::{LaneSet, RegionWords, Ticket};
pub use layout::{
    Geometry, OP_ADD1_F32, OP_ADD1_F32_V, OP_H2V, OP_NOP, OP_SUM_U32, OP_SUM_U32_V, OP_V2H,
    RESIDENT_PARAMS_BYTES, STATUS_DONE, STATUS_ERR, STATUS_TEAM_INCOMPLETE, USER_OP_YIELD,
};
pub use group::{GroupHandle, PeerGroup};
pub use l2_persist::{L2BenchReport, L2Capability, L2Persist};
// WideKernel is defined in this module; re-exported at the crate root
// alongside the other gpu_peer types via lib.rs.
pub use region::PeerRegion;
pub use vram::VramPool;

/// Pre-generated PTX for the peer kernels (driver-JIT'd at runtime;
/// regenerating after a kernels/gpu_peer.cu edit requires nvcc - see
/// that file's header).
pub(crate) const PEER_PTX: &str = include_str!("../../kernels/gpu_peer.ptx");

/// The kernel source, embedded so user opcodes can be NVRTC-composed
/// with the poller at init into one module (device-function linkage
/// requires a single compilation unit).
const PEER_CU: &str = include_str!("../../kernels/gpu_peer.cu");

/// Errors from the GPU-peer substrate.
#[derive(Debug)]
pub enum GpuPeerError {
    /// No usable CUDA device / driver.
    NoDevice(String),
    /// A CUDA driver call failed.
    Driver(String),
    /// Region file I/O failed.
    Io(std::io::Error),
    /// Payload exceeds the slot capacity for this geometry.
    PayloadTooLarge {
        /// Rejected payload length.
        len: usize,
        /// Slot payload capacity.
        max: usize,
    },
    /// Tickets must be reaped in submission order per lane.
    ReapOutOfOrder {
        /// Lane of the offending reap.
        lane: u32,
        /// The lane's oldest unreaped sequence.
        expected: u32,
        /// The sequence the caller tried to reap.
        got: u32,
    },
    /// A bounded wait expired.
    Timeout,
    /// The substrate cannot operate on this host (capability refused
    /// by calibration).
    Unavailable(&'static str),
}

impl core::fmt::Display for GpuPeerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoDevice(s) => write!(f, "no CUDA device: {s}"),
            Self::Driver(s) => write!(f, "CUDA driver error: {s}"),
            Self::Io(e) => write!(f, "region I/O error: {e}"),
            Self::PayloadTooLarge { len, max } => {
                write!(f, "payload {len} bytes exceeds slot capacity {max}")
            }
            Self::ReapOutOfOrder { lane, expected, got } => write!(
                f,
                "lane {lane} reap out of order: expected seq {expected}, got {got}"
            ),
            Self::Timeout => write!(f, "bounded wait expired"),
            Self::Unavailable(s) => write!(f, "substrate unavailable: {s}"),
        }
    }
}

impl std::error::Error for GpuPeerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Calibrate this peer, taking the timing half from the host's stored
/// record when there is one for this device and it was measured on a
/// device nothing else was resident on.
///
/// The margin and the atomics flag are re-established on every start
/// whichever path this takes, because what they assert is how this
/// device behaved under contention and only a run on it can say that.
///
/// Every way of failing to reach the table ends in a full measurement,
/// which is what this did before there was one. What it does not do is
/// fail quietly: a table that cannot be read and a device that has
/// never been measured produce the same calibration, and only the
/// diagnostic separates them.
#[cfg(feature = "persisted-calibration")]
fn calibrate_or_reuse(
    region: &PeerRegion,
    stream: &Arc<CudaStream>,
    kernels: &calibration::CalibKernels,
    ordinal: usize,
) -> Result<PeerCalibration, GpuPeerError> {
    use crate::sched::calibration_store::{
        ACCEL_DOORBELL_OK, ACCEL_SYS_ATOMICS_OK, ACCEL_TIMED_LOCK_OK, AccelCalibration, AccelKind,
        CalibrationStore, HostStamp, StoreError, calibration_dir,
    };

    let Some(dir) = calibration_dir() else {
        return calibration::calibrate(region, stream, kernels);
    };
    let stamp = HostStamp::detect();
    let store = match CalibrationStore::open_or_create(&dir, &stamp) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "flynnel gpu peer: the calibration table under {} is unusable ({e:?}); \
                 measuring this device",
                dir.display()
            );
            return calibration::calibrate(region, stream, kernels);
        }
    };

    let stored = store.read().and_then(|(_cpu, accel)| {
        accel
            .into_iter()
            .find(|a| a.ordinal as usize == ordinal && a.is_trustworthy())
    });

    let cal = match stored {
        Some(prior) => calibration::calibrate_with_prior(
            region,
            stream,
            kernels,
            PeerCalibration {
                rtt_min_ns: prior.rtt_min_ns,
                rtt_median_ns: prior.rtt_median_ns,
                rtt_p99_ns: prior.rtt_p99_ns,
                one_way_ns: prior.one_way_ns,
                clock_err_ns: prior.clock_err_ns,
                delta_ns: prior.delta_ns,
                launch_ns: prior.launch_ns,
                doorbell_ok: true,
                timed_lock_ok: false,
                sys_atomics_ok: false,
                lock_cpu_contended: 0,
                lock_gpu_contended: 0,
            },
        )?,
        None => calibration::calibrate(region, stream, kernels)?,
    };

    if !cal.doorbell_ok {
        // Nothing worth storing: the substrate did not come up.
        return Ok(cal);
    }
    let caps = device_capabilities(ordinal);
    let mut flags = ACCEL_DOORBELL_OK;
    if cal.timed_lock_ok {
        flags |= ACCEL_TIMED_LOCK_OK;
    }
    if cal.sys_atomics_ok {
        flags |= ACCEL_SYS_ATOMICS_OK;
    }
    let record = AccelCalibration::new(
        AccelKind::GpuPeer,
        ordinal as u32,
        caps.capability,
        caps.multiprocessors,
        caps.clock_khz,
        caps.memory_mib,
        flags,
        cal.rtt_min_ns,
        cal.rtt_median_ns,
        cal.rtt_p99_ns,
        cal.one_way_ns,
        cal.clock_err_ns,
        cal.delta_ns,
        cal.launch_ns,
    );
    match store.try_acquire_writer() {
        Ok(writer) => {
            writer.beat();
            // The CPU half is carried through untouched. This path
            // measured a device, not a host, and writing a host record
            // from here would claim a calibration nothing performed.
            let (cpu, mut slots) = match store.read() {
                Some((cpu, accel)) => (cpu, accel),
                None => (Default::default(), Vec::new()),
            };
            match slots.iter().position(|a| a.ordinal as usize == ordinal) {
                // The wave costs come from a separate calibration run, so a
                // start that re-measures the device timings keeps them.
                Some(i) => {
                    slots[i] = match slots[i].wave() {
                        Some(wave) => record.with_wave(wave),
                        None => record,
                    };
                }
                None => slots.push(record),
            }
            writer.publish(&cpu, &slots);
        }
        // Another process is measuring this host. Its record serves the
        // next start; this one keeps what it just established.
        Err(StoreError::WriterActive) => {}
        Err(e) => eprintln!(
            "flynnel gpu peer: calibrated device {ordinal} but could not publish it ({e:?})"
        ),
    }
    Ok(cal)
}

/// Calibrate this peer with nothing persisted.
#[cfg(not(feature = "persisted-calibration"))]
fn calibrate_or_reuse(
    region: &PeerRegion,
    stream: &Arc<CudaStream>,
    kernels: &calibration::CalibKernels,
    _ordinal: usize,
) -> Result<PeerCalibration, GpuPeerError> {
    calibration::calibrate(region, stream, kernels)
}

/// What the driver reports about a device, as opposed to what a probe
/// measures against it. Zero for anything the driver declines to answer.
#[cfg(feature = "persisted-calibration")]
struct DeviceCapabilities {
    capability: u32,
    multiprocessors: u32,
    clock_khz: u32,
    memory_mib: u32,
}

#[cfg(feature = "persisted-calibration")]
fn device_capabilities(ordinal: usize) -> DeviceCapabilities {
    let mut dev: cu::CUdevice = 0;
    // SAFETY: an out parameter and an ordinal; the driver validates the
    // ordinal and reports failure rather than writing on a bad one.
    let got = unsafe { cu::cuDeviceGet(&mut dev, ordinal as i32) };
    if got != cu::CUresult::CUDA_SUCCESS {
        return DeviceCapabilities {
            capability: 0,
            multiprocessors: 0,
            clock_khz: 0,
            memory_mib: 0,
        };
    }
    let attr = |a: cu::CUdevice_attribute| -> u32 {
        let mut v: i32 = 0;
        // SAFETY: valid device handle from cuDeviceGet.
        let r = unsafe { cu::cuDeviceGetAttribute(&mut v, a, dev) };
        if r == cu::CUresult::CUDA_SUCCESS && v > 0 { v as u32 } else { 0 }
    };
    let mut bytes: usize = 0;
    // SAFETY: an out parameter and a valid device handle.
    let mem = unsafe { cu::cuDeviceTotalMem_v2(&mut bytes, dev) };
    let memory_mib = if mem == cu::CUresult::CUDA_SUCCESS {
        (bytes / (1024 * 1024)) as u32
    } else {
        0
    };
    DeviceCapabilities {
        capability: attr(cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 10
            + attr(cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        multiprocessors: attr(cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        clock_khz: attr(cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE),
        memory_mib,
    }
}

/// Construction parameters. The defaults suit control-plane traffic
/// (4 KB slots); bulk streaming raises `slot_bytes`.
#[derive(Debug, Clone)]
pub struct GpuPeerConfig {
    /// Backing file for the shared region. Default: a per-process
    /// file in the OS temp directory, removed on drop. A caller-fixed
    /// path makes the region attachable by other processes.
    pub region_path: Option<PathBuf>,
    /// SPSC lane count (one consumer block each).
    pub lanes: u32,
    /// Slot size including the 16-byte descriptor.
    pub slot_bytes: u32,
    /// Ring depth per lane.
    pub slots_per_lane: u32,
    /// Poller quantum (bounded residency per launch).
    pub quantum_ns: u64,
    /// How long rank 0 waits for the rest of its block team before
    /// retiring the slot [`STATUS_TEAM_INCOMPLETE`].
    ///
    /// Only consulted when `blocks_per_lane > 1`. This is what a caller
    /// waits to learn that a team was lost, so it trades reporting
    /// latency against abandoning teams that would have assembled. A
    /// healthy team on a measured host needs 2.4 to 13 microseconds;
    /// the default leaves several hundred times that.
    pub barrier_deadline_ns: u64,
    /// Idle time after which a resident quantum parks.
    pub idle_exit_ns: u64,
    /// CUDA device ordinal.
    pub device_ordinal: usize,
    /// Resident-pool block size (bytes of VRAM per block).
    pub vram_block_bytes: u32,
    /// Resident-pool block count (0 disables the pool).
    pub vram_blocks: u32,
    /// Blocks serving each lane. 1 keeps a lane on one SM, which suits
    /// many small ops; above 1 a lane is worked by a team of
    /// consecutive blocks so a single doorbell op spreads across the
    /// device. Rank 0 owns the ring and retires the slot once the
    /// whole team has finished; the user op receives its rank and the
    /// team size and strides its work over them.
    ///
    /// [`GpuPeer::init`] clamps this to the device's streaming
    /// multiprocessor count when the driver reports one, because a team
    /// wider than the device loses ranks at its barrier, and says so on
    /// stderr when it does. [`GpuPeer::team_size`] is the size in use,
    /// and it is the `team_size` the user op receives.
    pub blocks_per_lane: u32,
    /// User opcode implementations as CUDA C source defining
    /// `extern "C" __device__ unsigned flynnel_user_op(unsigned op,
    /// unsigned char* block, unsigned count, volatile unsigned char*
    /// payload, unsigned team_rank, unsigned team_size)` - the
    /// six-argument team-aware hook the kernel forward-declares; a
    /// source defining any other arity fails the NVRTC compose with a
    /// duplicate-C-linkage error. When set, the poller is NVRTC-compiled at init
    /// together with this source and ops >= [`layout::OP_USER_BASE`]
    /// dispatch through it (called block-cooperatively by all 256
    /// threads). Requires the NVRTC runtime library on the host;
    /// `None` uses the pre-generated PTX and needs only the driver.
    ///
    /// The op's return value decides the slot. `0` retires it
    /// [`STATUS_DONE`]; [`USER_OP_YIELD`] (`FLYNNEL_USER_YIELD` in the
    /// source) keeps it in the ring so the same op runs again on the
    /// poller's next pass; anything else retires it [`STATUS_ERR`]. Only
    /// the value returned by thread 0 of rank 0 is read, so a failure on
    /// any other thread, or in another block of a team, reaches the slot
    /// only when the op carries it to that thread through shared device
    /// memory. A yielded slot runs again only while its lane has a
    /// resident quantum; [`GpuPeer::wait_status`] relaunches the lane as
    /// it waits.
    ///
    /// `__syncthreads` synchronizes the threads of one block. The blocks
    /// of a team meet only at atomic barriers, the kernel's or the wave
    /// helpers', and every thread of a block must reach the same number
    /// of `__syncthreads` in one op or the block deadlocks at its next
    /// barrier. Device `atomicAdd` returns the word's value from before
    /// the addition.
    ///
    /// The source is composed after the poller kernel and the wave
    /// helpers in `kernels/gpu_peer_wave.cu`, so an op may call
    /// `gtimer()`, return `FLYNNEL_USER_YIELD`, and run a segmented wave
    /// through the `flw_` helpers (see [`wave`]).
    ///
    /// A handle passed to [`GpuPeer::submit_user`] may name a span from
    /// [`GpuPeer::pin_bulk`]: the op's `count` may run to the end of the
    /// pool, not only to the end of its first block.
    pub user_ops_cuda: Option<String>,
}

impl Default for GpuPeerConfig {
    fn default() -> Self {
        Self {
            region_path: None,
            lanes: 4,
            slot_bytes: 4096,
            slots_per_lane: 64,
            quantum_ns: 250_000_000,
            barrier_deadline_ns: 5_000_000,
            idle_exit_ns: 2_000_000,
            device_ordinal: 0,
            vram_block_bytes: 65_536,
            vram_blocks: 1024,
            blocks_per_lane: 1,
            user_ops_cuda: None,
        }
    }
}

/// A full user kernel compiled for the wide-launch path.
///
/// The doorbell user-op ([`GpuPeer::submit_user`]) runs on a single block
/// of 256 threads - one SM - which is right for many small
/// latency-sensitive ops but caps a single large data-parallel op
/// (a big convolution, a full-image stencil) at one SM. A
/// `WideKernel` is the complement: a caller-authored `__global__`
/// launched across a full grid over a resident block, so every SM
/// works the op while the data stays resident. The kernel keeps its
/// own grid-stride loop; [`GpuPeer::launch_wide`] sets the grid.
pub struct WideKernel {
    // Keeps the module alive for the function's lifetime.
    _module: Arc<CudaModule>,
    func: CudaFunction,
}

impl WideKernel {
    pub(crate) fn new(module: Arc<CudaModule>, func: CudaFunction) -> Self {
        Self { _module: module, func }
    }
}

/// A block of device-resident data the scheduler owns by index. Data
/// pinned through [`GpuPeer::pin`] stays in the VRAM pool across any
/// number of tasks; each resident task moves only an 8-byte param
/// header over the bus. All tasks touching one handle ride the
/// handle's lane, so same-handle ordering (read-after-write,
/// write-after-write) is the lane's FIFO order - no extra
/// synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentHandle {
    block: u32,
    lane: u32,
    bytes: u32,
}

impl ResidentHandle {
    /// Bytes pinned in the block.
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes as usize
    }
    /// True when zero bytes are pinned.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
    /// The lane all of this handle's tasks ride (its dependency
    /// chain).
    #[inline]
    pub fn lane(&self) -> u32 {
        self.lane
    }
}

/// The GPU as a scheduler peer. Single-submitter surface (methods
/// take `&mut self`), matching the single-producer lane protocol.
pub struct GpuPeer {
    // Field order = drop order: poller shutdown flag is set in Drop
    // before region unregisters and the pool frees; module/stream/
    // context outlive all of them.
    region: PeerRegion,
    lane_set: LaneSet,
    poller: poller::Poller,
    pool: Option<VramPool>,
    calibration: PeerCalibration,
    /// Blocks serving each lane after the clamp to the SM count.
    team_size: u32,
    /// The poller quantum and the kernel's team barrier deadline, which a
    /// wave's slice budget is derived after.
    quantum_ns: u64,
    barrier_deadline_ns: u64,
    /// The device this peer runs on, which watchdog detection reads.
    device_ordinal: usize,
    _module: Arc<CudaModule>,
    _stream: Arc<CudaStream>,
    // Wide ops run on their own stream so they neither serialize behind
    // a resident poller quantum nor block doorbell traffic; the two run
    // concurrently, which is exactly when pause_poller matters.
    wide_stream: Arc<CudaStream>,
    _ctx: Arc<CudaContext>,
    /// Set when init found a different context current on its thread.
    displaced_foreign_context: bool,
}

/// The calling thread's current CUDA context, or `None` when the
/// driver is uninitialized or no context is current.
///
/// A failed query and an absent context are the same answer here:
/// both mean the caller had nothing for this peer to displace.
fn current_context() -> Option<cu::CUcontext> {
    let mut ctx: cu::CUcontext = core::ptr::null_mut();
    // SAFETY: out-parameter write to a local; the driver reports its
    // own uninitialized state through the return code rather than
    // touching the pointer.
    let rc: cu::CUresult = unsafe { cu::cuCtxGetCurrent(&mut ctx) };
    if rc == cu::CUresult::CUDA_SUCCESS && !ctx.is_null() { Some(ctx) } else { None }
}

/// The team size a lane runs: the requested blocks per lane, no wider
/// than the device's streaming multiprocessor count when that is known.
fn clamp_team_size(requested: u32, sm_count: Option<u32>) -> u32 {
    sm_count.map_or(requested, |sm| requested.min(sm))
}

/// Report a context this peer displaced on the calling thread.
///
/// Flynnel operates on the device primary context and makes it current
/// on whatever thread reaches its bind points. A caller holding a
/// different context - one built with cudarc's `new_non_primary`, say -
/// keeps working right up until a peer call rebinds its thread, after
/// which its launches land on the wrong context with nothing to say
/// so. That is undiagnosable from the far end, so it is named here, at
/// the one moment the mismatch can be introduced.
/// Returns whether a context was displaced, so a caller can act on it
/// rather than only read about it. The report is the default because
/// refusing would change `init`'s success contract for every consumer
/// over a configuration none is known to have;
/// [`GpuPeer::displaced_foreign_context`] is how a consumer that would
/// rather fall back chooses that for itself.
fn warn_on_foreign_context(prior: Option<cu::CUcontext>) -> bool {
    let Some(prior) = prior else { return false };
    match current_context() {
        Some(ours) if ours != prior => {
            eprintln!(
                "flynnel gpu_peer: a different CUDA context was current on this \
                 thread ({prior:?}) and the peer's primary context ({ours:?}) has \
                 replaced it. The peer operates on the device primary context; \
                 work the caller enqueues after this point runs on the primary \
                 context, not the one it built."
            );
            true
        }
        _ => false,
    }
}

impl GpuPeer {
    /// Initialize the substrate: CUDA context, region creation +
    /// registration, kernel load, and the full host calibration
    /// (doorbell, clocks, Fischer self-test, atomics probe).
    ///
    /// Returns `Err` - never panics - when no device is present, so
    /// callers can fall back to CPU-only dispatch.
    pub fn init(config: GpuPeerConfig) -> Result<Self, GpuPeerError> {
        // Read the caller's context before creating ours, because
        // creation binds: cudarc's CudaContext::new ends in
        // bind_to_thread, so afterwards the current context is always
        // the one this call retained and the comparison would be with
        // itself.
        let prior = current_context();
        let ctx = CudaContext::new(config.device_ordinal)
            .map_err(|e| GpuPeerError::NoDevice(format!("{e:?}")))?;
        let displaced = warn_on_foreign_context(prior);
        let stream = ctx.default_stream();
        let module = match &config.user_ops_cuda {
            None => match ctx.load_module(Ptx::from_src(PEER_PTX)) {
                Ok(m) => m,
                Err(ptx_err) => {
                    // A driver older than the toolchain that produced
                    // the checked-in PTX rejects it
                    // (CUDA_ERROR_UNSUPPORTED_PTX_VERSION); the host's
                    // own NVRTC emits PTX its driver accepts.
                    eprintln!(
                        "flynnel gpu_peer: checked-in PTX rejected ({ptx_err:?}); \
                         compiling the peer kernels with NVRTC instead"
                    );
                    let ptx = cudarc::nvrtc::compile_ptx(PEER_CU).map_err(|e| {
                        GpuPeerError::Driver(format!(
                            "PTX load: {ptx_err:?}; NVRTC fallback compile: {e:?}"
                        ))
                    })?;
                    ctx.load_module(ptx).map_err(|e| {
                        GpuPeerError::Driver(format!("NVRTC-fallback PTX load: {e:?}"))
                    })?
                }
            },
            Some(user_src) => {
                // Compose poller, wave helpers and user ops into a single
                // compilation unit so the device-function calls link, then JIT.
                let src = format!(
                    "#define FLYNNEL_USER_OPS 1\n{PEER_CU}\n{}\n{user_src}\n",
                    wave::WAVE_CU
                );
                let ptx = cudarc::nvrtc::compile_ptx(src).map_err(|e| {
                    GpuPeerError::Driver(format!("user-ops NVRTC compile: {e:?}"))
                })?;
                ctx.load_module(ptx)
                    .map_err(|e| GpuPeerError::Driver(format!("user-ops PTX load: {e:?}")))?
            }
        };
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| GpuPeerError::Driver(format!("kernel `{name}`: {e:?}")))
        };
        let f_poller = load("flynnel_peer_poller")?;
        let kernels = calibration::CalibKernels {
            calib_pong: load("flynnel_peer_calib_pong")?,
            fischer: load("flynnel_peer_fischer")?,
            cas_probe: load("flynnel_peer_cas_probe")?,
        };

        // The poller tracks each lane's launches and exits in a
        // fixed-size header array, so a region with more lanes than it
        // addresses could not be scheduled.
        if config.lanes as usize > layout::MAX_POLLER_LANES {
            return Err(GpuPeerError::Unavailable(
                "lane count exceeds the per-lane poller words in the region header",
            ));
        }
        let geometry = Geometry {
            lanes: config.lanes.max(1),
            slot_bytes: config.slot_bytes.max(64),
            slots_per_lane: config.slots_per_lane.max(2),
        };
        let (path, remove_on_drop) = match &config.region_path {
            Some(p) => (p.clone(), false),
            None => {
                // Unique per instance, not just per process: a peer
                // GROUP creates several regions in one process.
                static INSTANCE: core::sync::atomic::AtomicU32 =
                    core::sync::atomic::AtomicU32::new(0);
                let n = INSTANCE.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                (
                    std::env::temp_dir().join(format!(
                        "flynnel_gpu_peer_{}_{n}.bin",
                        std::process::id()
                    )),
                    true,
                )
            }
        };
        let region = PeerRegion::create(&ctx, &path, geometry, remove_on_drop)?;
        let calibration =
            calibrate_or_reuse(&region, &stream, &kernels, config.device_ordinal)?;
        if !calibration.doorbell_ok {
            return Err(GpuPeerError::Unavailable(
                "doorbell handshake failed during calibration",
            ));
        }

        let pool = if config.vram_blocks > 0 {
            Some(VramPool::new(
                &ctx,
                &stream,
                config.vram_block_bytes.max(64),
                config.vram_blocks,
            )?)
        } else {
            None
        };
        let (vbase, vbytes, vblocks) = match &pool {
            Some(p) => (p.base(), p.block_bytes(), p.blocks()),
            None => (0, 0, 0),
        };
        let lane_set = LaneSet::new(geometry);
        // One stream per lane. Kernels on a single stream run in
        // sequence, so a lane relaunched onto the stream of a lane that
        // is still working would not start until that work finished.
        let mut lane_streams = Vec::with_capacity(geometry.lanes as usize);
        for lane in 0..geometry.lanes {
            lane_streams.push(
                ctx.new_stream()
                    .map_err(|e| GpuPeerError::Driver(format!("lane {lane} stream: {e:?}")))?,
            );
        }
        let sm_count = crate::backend::detect::cuda_sm_count(config.device_ordinal);
        let requested_team = config.blocks_per_lane.max(1);
        let team_size = clamp_team_size(requested_team, sm_count);
        if team_size != requested_team {
            eprintln!(
                "flynnel gpu_peer: blocks_per_lane {requested_team} exceeds device \
                 {}'s {team_size} streaming multiprocessors; each lane runs a team \
                 of {team_size}",
                config.device_ordinal
            );
        }
        let poller = poller::Poller::new(
            lane_streams,
            f_poller,
            config.quantum_ns,
            config.idle_exit_ns,
            vbase,
            vbytes,
            vblocks,
            team_size,
            config.barrier_deadline_ns,
        );
        let wide_stream = ctx
            .new_stream()
            .map_err(|e| GpuPeerError::Driver(format!("wide stream: {e:?}")))?;
        Ok(Self {
            region,
            lane_set,
            poller,
            pool,
            calibration,
            team_size,
            quantum_ns: config.quantum_ns,
            barrier_deadline_ns: config.barrier_deadline_ns.max(1),
            device_ordinal: config.device_ordinal,
            _module: module,
            _stream: stream,
            wide_stream,
            _ctx: ctx,
            displaced_foreign_context: displaced,
        })
    }

    /// Whether [`Self::init`] replaced a different CUDA context that
    /// was current on the calling thread.
    ///
    /// The peer operates on the device primary context and makes it
    /// current where it binds. A consumer that builds its own context -
    /// cudarc's `new_non_primary` is the way to get one - can read this
    /// and decide for itself whether to keep using the peer or fall
    /// back, rather than inheriting a policy chosen here. `false` on
    /// every host where the caller had no context of its own, or had
    /// the same primary context, which is every consumer known today.
    pub fn displaced_foreign_context(&self) -> bool {
        self.displaced_foreign_context
    }

    /// The host-measured constants and capability flags.
    #[inline]
    pub fn calibration(&self) -> PeerCalibration {
        self.calibration
    }

    /// Region geometry.
    #[inline]
    pub fn geometry(&self) -> Geometry {
        self.region.geometry()
    }

    /// Blocks serving each lane: [`GpuPeerConfig::blocks_per_lane`],
    /// clamped to the device's streaming multiprocessor count when the
    /// driver reports one. This is the `team_size` a user op receives.
    #[inline]
    pub fn team_size(&self) -> u32 {
        self.team_size
    }

    /// The shared region (attachment path, offset accessors).
    #[inline]
    pub fn region(&self) -> &PeerRegion {
        &self.region
    }

    /// Submit `payload` under a built-in opcode. Blocks (bounded) on
    /// backpressure when every lane is full.
    pub fn submit(&mut self, op: u32, payload: &[u8]) -> Result<Ticket, GpuPeerError> {
        let t0 = Instant::now();
        loop {
            if let Some(t) = self.lane_set.try_submit(&self.region, op, payload)? {
                self.poller.ensure_running(&self.region, t.lane)?;
                return Ok(t);
            }
            // Every lane is full, so the drain could come from any of
            // them and there is no one lane to wake.
            self.poller.ensure_running_all(&self.region)?;
            if t0.elapsed() > Duration::from_secs(10) {
                return Err(GpuPeerError::Timeout);
            }
            std::thread::yield_now();
        }
    }

    /// True when the consumer has completed `ticket`.
    #[inline]
    pub fn is_done(&self, ticket: Ticket) -> bool {
        self.lane_set.is_done(&self.region, ticket)
    }

    /// Wait (bounded) for completion and return the slot's status
    /// word, which is always [`STATUS_DONE`].
    ///
    /// A slot that completed without doing its work returns
    /// `Err(Unavailable)` rather than its status, so a caller that
    /// only tests for an error cannot read an unfilled payload as an
    /// answer. `Err(Timeout)` still means the slot never completed.
    /// [`Self::wait_status`] returns the raw word instead.
    pub fn wait(&mut self, ticket: Ticket, timeout: Duration) -> Result<u32, GpuPeerError> {
        let status = self.wait_status(ticket, timeout)?;
        if status != STATUS_DONE {
            return Err(GpuPeerError::Unavailable("slot completed with a failed status"));
        }
        Ok(status)
    }

    /// [`Self::wait`] without the status check: returns whatever the
    /// slot's status word holds once it completes.
    pub fn wait_status(&mut self, ticket: Ticket, timeout: Duration) -> Result<u32, GpuPeerError> {
        let t0 = Instant::now();
        let mut spins = 0u32;
        while !self.lane_set.is_done(&self.region, ticket) {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(4096) {
                // Self-heal the exit-vs-new-work race: if the lane's
                // quantum idled out between our submit and its poll,
                // relaunch that lane.
                self.poller.ensure_running(&self.region, ticket.lane)?;
                if t0.elapsed() > timeout {
                    return Err(GpuPeerError::Timeout);
                }
            }
            core::hint::spin_loop();
        }
        Ok(self.lane_set.status(&self.region, ticket))
    }

    /// Copy a completed ticket's result payload into `dst`, filling it
    /// entirely.
    ///
    /// `dst.len()` may not exceed the slot's payload capacity, which
    /// is `self.region().geometry().payload_max()`. Read that figure
    /// rather than deriving it from a slot size: a longer buffer is
    /// refused with [`GpuPeerError::PayloadTooLarge`], because the
    /// bytes past a slot belong to the next one.
    ///
    /// # Where a user op's bytes land
    ///
    /// This copies the slot payload from its start, while a user op is
    /// handed that payload advanced past the [`RESIDENT_PARAMS_BYTES`]
    /// parameter block. So an op writing its own byte 0 appears at
    /// `dst[RESIDENT_PARAMS_BYTES]`, and a buffer sized for the op's
    /// own layout must add that prefix to fit what comes back.
    ///
    /// The two sides count from different places, which is easy to
    /// carry incorrectly into a size calculation: an op needing `n`
    /// bytes of its own requires `RESIDENT_PARAMS_BYTES + n` here, and
    /// that total is what must fit `payload_max()`.
    pub fn read_result(&self, ticket: Ticket, dst: &mut [u8]) -> Result<(), GpuPeerError> {
        self.lane_set.read_result(&self.region, ticket, dst)
    }

    /// Release the ticket's slot for reuse (in submission order per
    /// lane).
    pub fn reap(&mut self, ticket: Ticket) -> Result<(), GpuPeerError> {
        self.lane_set.reap(ticket)
    }

    /// Barrier expiries since this peer started, and the largest ring
    /// depth seen at one: `(count, max_depth)`.
    ///
    /// A count above zero means some block team did not fully arrive
    /// before rank 0's deadline, and those slots came back
    /// [`STATUS_TEAM_INCOMPLETE`]. Both are zero when
    /// `blocks_per_lane` is 1, since a lane of one block has no
    /// barrier to miss.
    ///
    /// This counter is the only thing that sees them. A consumer
    /// measuring 12500 queries found three and six stalls leaving no
    /// mark on either signal it had: the 99th percentile there is the
    /// 125th slowest query, which a handful of expiries cannot reach,
    /// and recall moved 0.0005 while the count went 0, 3, 6. The same
    /// expiries dominated the percentile in an earlier run of 300
    /// queries. So a latency tail exposes them only when the window is
    /// short enough for a few events to be most of it, which is the
    /// window least worth measuring.
    ///
    /// The depth is what separates the two ways a team can be split. A
    /// quantum boundary falling between two ranks' clock reads splits
    /// whatever the lane happened to be holding, so a lightly fed lane
    /// reports one or two. A lane relaunched late and draining a
    /// backlog claims from a full ring and reports a depth near
    /// `slots_per_lane`. The count says stalls happened; the depth says
    /// which kind, and the host cannot recover it afterwards because
    /// the ring has moved on by the time a status is read.
    pub fn barrier_stalls(&self) -> (u32, u32) {
        (
            self.region.load_u32(layout::HDR_STALL_COUNT_OFF),
            self.region.load_u32(layout::HDR_STALL_MAX_DEPTH_OFF),
        )
    }

    /// Longest time rank 0 spent at the team barrier on a slot the whole
    /// team reached, in nanoseconds.
    ///
    /// The deadline a team is given is a whole quantum, 250 ms by
    /// default. This is what a healthy team on this host actually needs,
    /// and the ratio between them is the margin that default spends. It
    /// is the figure to consult before shortening the deadline: a
    /// shorter one tells a caller sooner that a team was lost, at the
    /// cost of abandoning teams that would have assembled.
    ///
    /// Zero when `blocks_per_lane` is 1, since there is no barrier.
    pub fn barrier_wait_max_ns(&self) -> u32 {
        self.region.load_u32(layout::HDR_BARRIER_WAIT_MAX_OFF)
    }

    /// Submit on a SPECIFIC lane (bounded backpressure wait).
    fn submit_on_lane(
        &mut self,
        lane: u32,
        op: u32,
        payload: &[u8],
    ) -> Result<Ticket, GpuPeerError> {
        let t0 = Instant::now();
        loop {
            if let Some(t) = self.lane_set.try_submit_on(&self.region, lane, op, payload)? {
                self.poller.ensure_running(&self.region, t.lane)?;
                return Ok(t);
            }
            self.poller.ensure_running(&self.region, lane)?;
            if t0.elapsed() > Duration::from_secs(10) {
                return Err(GpuPeerError::Timeout);
            }
            std::thread::yield_now();
        }
    }

    /// Pin `data` into a device-resident block. Synchronous (waits
    /// for the upload); requires the assigned lane to have no
    /// unreaped tickets outstanding.
    ///
    /// The upload rides a lane slot, so `data` may be at most one pool
    /// block and at most the slot's payload capacity less the
    /// [`RESIDENT_PARAMS_BYTES`] header, `geometry().payload_max() - 8`.
    /// Anything longer is refused with [`GpuPeerError::PayloadTooLarge`].
    /// Resident state larger than that goes through [`Self::pin_bulk`],
    /// which copies straight to the device across as many consecutive
    /// blocks as it needs.
    pub fn pin(&mut self, data: &[u8]) -> Result<ResidentHandle, GpuPeerError> {
        let pool = self
            .pool
            .as_mut()
            .ok_or(GpuPeerError::Unavailable("resident pool disabled"))?;
        let block_bytes = pool.block_bytes() as usize;
        if data.len() > block_bytes
            || data.len() + RESIDENT_PARAMS_BYTES > self.region.geometry().payload_max()
        {
            return Err(GpuPeerError::PayloadTooLarge {
                len: data.len(),
                max: block_bytes.min(
                    self.region.geometry().payload_max() - RESIDENT_PARAMS_BYTES,
                ),
            });
        }
        let block = pool
            .alloc()
            .ok_or(GpuPeerError::Unavailable("resident pool exhausted"))?;
        let lane = block % self.region.geometry().lanes;
        let mut payload = Vec::with_capacity(RESIDENT_PARAMS_BYTES + data.len());
        payload.extend_from_slice(&block.to_le_bytes());
        payload.extend_from_slice(&(data.len() as u32).to_le_bytes());
        payload.extend_from_slice(data);
        let t = self.submit_on_lane(lane, OP_H2V, &payload)?;
        let status = self.wait(t, Duration::from_secs(10))?;
        self.reap(t)?;
        if status != STATUS_DONE {
            if let Some(p) = self.pool.as_mut() {
                p.release(block);
            }
            return Err(GpuPeerError::Unavailable("resident upload rejected"));
        }
        Ok(ResidentHandle { block, lane, bytes: data.len() as u32 })
    }

    /// Re-upload `data` into an EXISTING resident block (synchronous
    /// H2V on the handle's lane). The residence-flip primitive: a
    /// host-modified mirror pushes its bytes back to the device
    /// without re-allocating. `data` must not exceed the handle's
    /// pinned length.
    pub fn write_resident(
        &mut self,
        handle: &ResidentHandle,
        data: &[u8],
    ) -> Result<(), GpuPeerError> {
        if data.len() > handle.len() {
            return Err(GpuPeerError::PayloadTooLarge { len: data.len(), max: handle.len() });
        }
        let mut payload = Vec::with_capacity(RESIDENT_PARAMS_BYTES + data.len());
        payload.extend_from_slice(&handle.block.to_le_bytes());
        payload.extend_from_slice(&(data.len() as u32).to_le_bytes());
        payload.extend_from_slice(data);
        let t = self.submit_on_lane(handle.lane, layout::OP_H2V, &payload)?;
        let status = self.wait(t, Duration::from_secs(10))?;
        self.reap(t)?;
        if status != STATUS_DONE {
            return Err(GpuPeerError::Unavailable("resident re-upload rejected"));
        }
        Ok(())
    }

    /// Pin a buffer of any size straight into VRAM, bypassing the
    /// doorbell.
    ///
    /// [`Self::pin`] carries its data in a slot payload, so it is
    /// capped by `slot_bytes` - fine for the kilobyte operands a
    /// doorbell op takes, useless for a corpus. This copies host to
    /// device directly and spans as many consecutive pool blocks as
    /// the data needs, which is what a workload that must stay
    /// resident across many calls requires: upload once, query
    /// forever, and only the query's own arguments ever cross again.
    ///
    /// The returned handle names the first block; `resident_ptr` gives
    /// its device address and the span is contiguous by construction.
    pub fn pin_bulk(&mut self, data: &[u8]) -> Result<ResidentHandle, GpuPeerError> {
        let pool = self
            .pool
            .as_mut()
            .ok_or(GpuPeerError::Unavailable("resident pool disabled"))?;
        let block_bytes = pool.block_bytes() as usize;
        let need = data.len().div_ceil(block_bytes.max(1)).max(1);
        if need > pool.free_blocks() {
            return Err(GpuPeerError::Unavailable("resident pool exhausted"));
        }
        let first = pool
            .alloc_span(need as u32)
            .ok_or(GpuPeerError::Unavailable("no contiguous resident span"))?;
        let dst = pool.block_ptr(first);
        // SAFETY: `dst` is the pool's own device allocation and the
        // span was just claimed, so it covers `data.len()` bytes.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                dst,
                data,
                self.wide_stream.cu_stream() as _,
            )
            .map_err(|e| GpuPeerError::Driver(format!("pin_bulk htod: {e:?}")))?;
        }
        self.wide_stream
            .synchronize()
            .map_err(|e| GpuPeerError::Driver(format!("pin_bulk sync: {e:?}")))?;
        Ok(ResidentHandle {
            block: first,
            lane: first % self.region.geometry().lanes,
            bytes: data.len() as u32,
        })
    }

    /// Overwrite a [`Self::pin_bulk`] span host-to-device directly,
    /// for the small per-call operands a resident workload still
    /// changes (a query's needle, a result counter's reset).
    pub fn write_resident_bulk(
        &mut self,
        handle: &ResidentHandle,
        data: &[u8],
    ) -> Result<(), GpuPeerError> {
        let (dst, _) = self.resident_ptr(handle)?;
        // SAFETY: `dst` is the pool's own device span for this handle.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                dst,
                data,
                self.wide_stream.cu_stream() as _,
            )
            .map_err(|e| GpuPeerError::Driver(format!("write_resident_bulk: {e:?}")))?;
        }
        self.wide_stream
            .synchronize()
            .map_err(|e| GpuPeerError::Driver(format!("write_resident_bulk sync: {e:?}")))
    }

    /// Read a [`Self::pin_bulk`] span device-to-host directly.
    pub fn fetch_bulk(
        &mut self,
        handle: &ResidentHandle,
        out: &mut [u8],
    ) -> Result<(), GpuPeerError> {
        let (src, _) = self.resident_ptr(handle)?;
        // SAFETY: `src` is the pool's own device span for this handle.
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(
                out,
                src,
                self.wide_stream.cu_stream() as _,
            )
            .map_err(|e| GpuPeerError::Driver(format!("fetch_bulk: {e:?}")))?;
        }
        self.wide_stream
            .synchronize()
            .map_err(|e| GpuPeerError::Driver(format!("fetch_bulk sync: {e:?}")))
    }

    /// [`Self::pin`] without waiting: zero-synchronization prefetch.
    /// The upload rides the handle's lane, and lane FIFO order IS the
    /// dependency order - any task submitted on this handle
    /// afterwards executes after the data has landed, with no fence,
    /// no event, no wait anywhere. The returned upload ticket must be
    /// reaped first among the lane's tickets (in-order reap rule).
    pub fn pin_prefetch(
        &mut self,
        data: &[u8],
    ) -> Result<(ResidentHandle, Ticket), GpuPeerError> {
        let pool = self
            .pool
            .as_mut()
            .ok_or(GpuPeerError::Unavailable("resident pool disabled"))?;
        let block_bytes = pool.block_bytes() as usize;
        if data.len() > block_bytes
            || data.len() + RESIDENT_PARAMS_BYTES > self.region.geometry().payload_max()
        {
            return Err(GpuPeerError::PayloadTooLarge {
                len: data.len(),
                max: block_bytes
                    .min(self.region.geometry().payload_max() - RESIDENT_PARAMS_BYTES),
            });
        }
        let block = pool
            .alloc()
            .ok_or(GpuPeerError::Unavailable("resident pool exhausted"))?;
        let lane = block % self.region.geometry().lanes;
        let mut payload = Vec::with_capacity(RESIDENT_PARAMS_BYTES + data.len());
        payload.extend_from_slice(&block.to_le_bytes());
        payload.extend_from_slice(&(data.len() as u32).to_le_bytes());
        payload.extend_from_slice(data);
        let t = self.submit_on_lane(lane, layout::OP_H2V, &payload)?;
        Ok((ResidentHandle { block, lane, bytes: data.len() as u32 }, t))
    }

    /// Submit a user opcode (>= [`layout::OP_USER_BASE`], implemented
    /// by the CUDA source registered via
    /// [`GpuPeerConfig::user_ops_cuda`]). With a handle, the task
    /// rides the handle's lane (ordered with its other tasks) and the
    /// hook receives the resident block; without one it round-robins
    /// and the hook receives a null block. `args` land at payload+8
    /// (the hook's argument/result space).
    pub fn submit_user(
        &mut self,
        op: u32,
        handle: Option<&ResidentHandle>,
        args: &[u8],
    ) -> Result<Ticket, GpuPeerError> {
        if op < layout::OP_USER_BASE {
            return Err(GpuPeerError::Unavailable("op below OP_USER_BASE"));
        }
        let (block, count, lane) = match handle {
            Some(h) => (h.block, h.bytes, Some(h.lane)),
            None => (layout::NO_BLOCK, 0u32, None),
        };
        let mut payload = Vec::with_capacity(RESIDENT_PARAMS_BYTES + args.len());
        payload.extend_from_slice(&block.to_le_bytes());
        payload.extend_from_slice(&count.to_le_bytes());
        payload.extend_from_slice(args);
        match lane {
            Some(l) => self.submit_on_lane(l, op, &payload),
            None => self.submit(op, &payload),
        }
    }

    /// [`Self::submit_user`] pinned to `lane`, for diagnostics that
    /// need to control which lane's blocks stay warm.
    ///
    /// A handle's own lane owns its resident block, so passing a
    /// different lane here reads that block from another lane's
    /// team. Pass `None` unless the op does not touch resident data.
    /// `lane` is taken modulo the lane count.
    pub fn submit_user_on_lane(
        &mut self,
        op: u32,
        handle: Option<&ResidentHandle>,
        args: &[u8],
        lane: u32,
    ) -> Result<Ticket, GpuPeerError> {
        if op < layout::OP_USER_BASE {
            return Err(GpuPeerError::Unavailable("op below OP_USER_BASE"));
        }
        let (block, count) = match handle {
            Some(h) => (h.block, h.bytes),
            None => (layout::NO_BLOCK, 0u32),
        };
        let mut payload = Vec::with_capacity(RESIDENT_PARAMS_BYTES + args.len());
        payload.extend_from_slice(&block.to_le_bytes());
        payload.extend_from_slice(&count.to_le_bytes());
        payload.extend_from_slice(args);
        let lanes = self.region.geometry().lanes.max(1);
        self.submit_on_lane(lane % lanes, op, &payload)
    }

    /// Submit a resident-block task (`OP_ADD1_F32_V` / `OP_SUM_U32_V`).
    /// Only the 8-byte param header crosses the bus; the data stays
    /// in VRAM. Tasks on one handle execute in submission order (lane
    /// FIFO).
    pub fn submit_resident(
        &mut self,
        op: u32,
        handle: &ResidentHandle,
    ) -> Result<Ticket, GpuPeerError> {
        let mut params = [0u8; RESIDENT_PARAMS_BYTES];
        params[..4].copy_from_slice(&handle.block.to_le_bytes());
        params[4..].copy_from_slice(&handle.bytes.to_le_bytes());
        self.submit_on_lane(handle.lane, op, &params)
    }

    /// Download a resident block into `out` (synchronous; same lane
    /// discipline as [`Self::pin`]). `out` receives
    /// `min(out.len(), handle.len())` bytes.
    pub fn fetch(
        &mut self,
        handle: &ResidentHandle,
        out: &mut [u8],
    ) -> Result<(), GpuPeerError> {
        let mut params = [0u8; RESIDENT_PARAMS_BYTES];
        params[..4].copy_from_slice(&handle.block.to_le_bytes());
        params[4..].copy_from_slice(&handle.bytes.to_le_bytes());
        let t = self.submit_on_lane(handle.lane, OP_V2H, &params)?;
        let status = self.wait(t, Duration::from_secs(10))?;
        if status != STATUS_DONE {
            self.reap(t)?;
            return Err(GpuPeerError::Unavailable("resident download rejected"));
        }
        let n = out.len().min(handle.len());
        let mut buf = vec![0u8; RESIDENT_PARAMS_BYTES + n];
        let read = self.read_result(t, &mut buf);
        self.reap(t)?;
        read?;
        out[..n].copy_from_slice(&buf[RESIDENT_PARAMS_BYTES..]);
        Ok(())
    }

    /// Return the handle's block to the pool. The caller is done
    /// with the data (any still-queued tasks on the lane complete
    /// first by lane order before a new pin can reuse the block's
    /// lane slot).
    pub fn unpin(&mut self, handle: ResidentHandle) -> Result<(), GpuPeerError> {
        let pool = self
            .pool
            .as_mut()
            .ok_or(GpuPeerError::Unavailable("resident pool disabled"))?;
        // A pin_bulk handle spans every block its bytes cover.
        let span = (handle.bytes as usize)
            .div_ceil(pool.block_bytes().max(1) as usize)
            .max(1) as u32;
        for b in handle.block..handle.block + span {
            pool.release(b);
        }
        Ok(())
    }

    /// Resident-pool stats: (free blocks, total blocks); zeros when
    /// the pool is disabled.
    pub fn pool_stats(&self) -> (usize, u32) {
        match &self.pool {
            Some(p) => (p.free_blocks(), p.blocks()),
            None => (0, 0),
        }
    }

    /// Raw device address and byte length of a resident handle's
    /// block. A wide-launch kernel targets this base directly.
    pub fn resident_ptr(&self, handle: &ResidentHandle) -> Result<(u64, usize), GpuPeerError> {
        let pool = self
            .pool
            .as_ref()
            .ok_or(GpuPeerError::Unavailable("resident pool disabled"))?;
        Ok((pool.block_ptr(handle.block), handle.len()))
    }

    /// NVRTC-compile a caller-authored full `__global__` kernel for
    /// the wide-launch path. The source stands alone (it is not
    /// composed with the poller); `entry` names the `extern "C"`
    /// entry point. Requires the NVRTC runtime; returns a driver
    /// error where it is absent.
    ///
    /// The kernel's signature is `(T0* p0, .., Tn* pn, u32 s0, ..)` -
    /// pointer arguments first (device addresses from
    /// [`Self::resident_ptr`]), then u32 scalars - matching the
    /// `ptrs` and `scalars` passed to [`Self::launch_wide`]. Write a
    /// grid-stride loop so any grid size is correct.
    pub fn compile_wide_kernel(&self, src: &str, entry: &str) -> Result<WideKernel, GpuPeerError> {
        let ptx = cudarc::nvrtc::compile_ptx(src)
            .map_err(|e| GpuPeerError::Driver(format!("wide-kernel NVRTC compile: {e:?}")))?;
        self.load_wide_kernel(ptx, entry)
    }

    /// Load a wide-launch kernel from PTX text (pre-generated, driver-
    /// JIT'd; no NVRTC needed). Same signature contract as
    /// [`Self::compile_wide_kernel`].
    pub fn load_wide_kernel_ptx(&self, ptx: &str, entry: &str) -> Result<WideKernel, GpuPeerError> {
        self.load_wide_kernel(Ptx::from_src(ptx), entry)
    }

    fn load_wide_kernel(&self, ptx: Ptx, entry: &str) -> Result<WideKernel, GpuPeerError> {
        let module = self
            ._ctx
            .load_module(ptx)
            .map_err(|e| GpuPeerError::Driver(format!("wide-kernel PTX load: {e:?}")))?;
        let func = module
            .load_function(entry)
            .map_err(|e| GpuPeerError::Driver(format!("wide-kernel entry `{entry}`: {e:?}")))?;
        Ok(WideKernel { _module: module, func })
    }

    /// The CUDA context this peer runs on, for consumers that bind
    /// their own library handles or allocate device memory the wide
    /// kernels then read.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self._ctx
    }

    /// The stream wide launches run on. Work a consumer enqueues here
    /// (its own kernels, library calls) is FIFO-ordered with
    /// [`Self::launch_wide_async`] launches and fenced by
    /// [`Self::sync_wide`].
    pub fn wide_stream(&self) -> &Arc<CudaStream> {
        &self.wide_stream
    }

    /// Enqueue a [`WideKernel`] on the resident stream without
    /// synchronizing. Pointer arguments come first, then u32 scalars,
    /// matching the kernel signature; `grid_blocks` = 0 auto-sizes the
    /// grid from `scalars[0]` (an element count).
    fn launch_wide_inner(
        &self,
        kernel: &WideKernel,
        grid_blocks: u32,
        block_threads: u32,
        ptrs: &[u64],
        scalars: &[u32],
    ) -> Result<(), GpuPeerError> {
        let block = block_threads.clamp(1, 1024);
        let grid = if grid_blocks == 0 {
            scalars.first().copied().unwrap_or(1).div_ceil(block).max(1)
        } else {
            grid_blocks
        };
        let mut b = self.wide_stream.launch_builder(&kernel.func);
        for p in ptrs {
            b.arg(p);
        }
        for s in scalars {
            b.arg(s);
        }
        // SAFETY: the caller's kernel signature matches (ptrs as
        // device-pointer args, then u32 scalars); the arg values live
        // in the caller's slices for the duration of the launch.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| GpuPeerError::Driver(format!("wide launch: {e:?}")))?;
        Ok(())
    }

    /// Launch a [`WideKernel`] across `grid_blocks x block_threads`
    /// threads and block until it finishes. This is the full-device
    /// path a large resident op takes instead of the one-block
    /// doorbell op. One-off convenience: it pays a stream sync (a
    /// WDDM command-buffer flush) per call. For a chain of many small
    /// dependent kernels use [`Self::launch_wide_async`] +
    /// [`Self::sync_wide`], which pays a single flush for the whole batch.
    pub fn launch_wide(
        &self,
        kernel: &WideKernel,
        grid_blocks: u32,
        block_threads: u32,
        ptrs: &[u64],
        scalars: &[u32],
    ) -> Result<(), GpuPeerError> {
        self.launch_wide_inner(kernel, grid_blocks, block_threads, ptrs, scalars)?;
        self.sync_wide()
    }

    /// Enqueue a [`WideKernel`] on the resident stream and return
    /// immediately - no sync. The resident stream is FIFO, so kernels
    /// queued back to back run in order and a dependent chain is
    /// correct without a per-kernel sync. Queue the whole batch, then
    /// call [`Self::sync_wide`] once. This is the WDDM-friendly path
    /// for a many-small-dependent-kernel workload, where a sync per
    /// call would flush the command buffer N times.
    pub fn launch_wide_async(
        &self,
        kernel: &WideKernel,
        grid_blocks: u32,
        block_threads: u32,
        ptrs: &[u64],
        scalars: &[u32],
    ) -> Result<(), GpuPeerError> {
        self.launch_wide_inner(kernel, grid_blocks, block_threads, ptrs, scalars)
    }

    /// Block until every enqueued wide launch on the resident stream
    /// has finished. One flush for a whole [`Self::launch_wide_async`]
    /// batch.
    pub fn sync_wide(&self) -> Result<(), GpuPeerError> {
        self.wide_stream
            .synchronize()
            .map_err(|e| GpuPeerError::Driver(format!("wide sync: {e:?}")))
    }

    /// Quiesce the doorbell poller: force its resident quantum to exit
    /// and hold off relaunches until [`Self::resume_poller`]. The
    /// poller busy-polls its lanes while resident, so a live poller
    /// steals SM occupancy and L2 bandwidth from a concurrent wide op.
    /// Pause it around a heavy wide batch to hand the whole device to
    /// that batch, then resume. Small doorbell ops submitted while
    /// paused simply queue and are consumed after resume.
    pub fn pause_poller(&mut self) -> Result<(), GpuPeerError> {
        self.poller.pause(&self.region)
    }

    /// Undo [`Self::pause_poller`]; the next submit relaunches the
    /// poller.
    pub fn resume_poller(&mut self) {
        self.poller.resume();
    }

    /// Whether the doorbell poller is currently paused.
    #[inline]
    pub fn poller_paused(&self) -> bool {
        self.poller.is_paused()
    }

    /// The device's L2-persistence ceilings (set-aside + window max).
    pub fn l2_capability(&self) -> Result<L2Capability, GpuPeerError> {
        L2Capability::query(self._ctx_ref())
    }

    /// Measure L2 persistence on this device: the same hammer kernel
    /// timed with the hot working set pinned in L2 versus streaming.
    /// See [`l2_persist::benchmark`].
    pub fn l2_benchmark(
        &self,
        hot_bytes: usize,
        pol_bytes: usize,
        iters: u32,
        runs: u32,
    ) -> Result<L2BenchReport, GpuPeerError> {
        l2_persist::benchmark(self._ctx_ref(), hot_bytes, pol_bytes, iters, runs)
    }

    #[inline]
    fn _ctx_ref(&self) -> &Arc<CudaContext> {
        &self._ctx
    }

    /// Acquire the region's Fischer timed lock (cross-device mutual
    /// exclusion without atomics) at the CALIBRATED margin. Only
    /// available when the calibration self-test granted the
    /// capability.
    pub fn timed_lock_acquire(&self, timeout: Duration) -> Result<(), GpuPeerError> {
        if !self.calibration.timed_lock_ok {
            return Err(GpuPeerError::Unavailable("timed lock not validated on this host"));
        }
        if timed_lock::acquire(
            &self.region,
            Duration::from_nanos(self.calibration.delta_ns),
            timeout,
        ) {
            Ok(())
        } else {
            Err(GpuPeerError::Timeout)
        }
    }

    /// Release the region's Fischer timed lock.
    pub fn timed_lock_release(&self) {
        timed_lock::release(&self.region);
    }
}

impl Drop for GpuPeer {
    fn drop(&mut self) {
        self.poller.shutdown(&self.region);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The foreign-context detector fires, and the ordering that lets
    /// it fire is what this pins.
    ///
    /// cudarc's `CudaContext::new` ends in `bind_to_thread`, so a
    /// detector that read the current context once the peer's own had
    /// been created would compare that context against itself, report
    /// nothing on every host, and be indistinguishable from a working
    /// guard. Moving the read in `init` below the creation makes this
    /// test go quiet, which is the regression it exists to catch.
    ///
    /// Requires a CUDA device.
    #[test]
    fn init_reports_displacing_a_foreign_context() {
        // A non-primary context is the only thing the peer's primary
        // context can displace; every consumer today holds the primary
        // one, where there is nothing to report.
        let foreign = CudaContext::new_non_primary(0, 0)
            .expect("a CUDA device is required for this test");
        foreign.bind_to_thread().expect("make the foreign context current");
        let before = current_context().expect("the foreign context is current");

        let peer = GpuPeer::init(GpuPeerConfig::default()).expect("peer init");
        assert!(
            peer.displaced_foreign_context(),
            "init replaced a context the caller had current and must say so"
        );

        let after = current_context().expect("a context is current after init");
        assert_ne!(
            after, before,
            "the report is only meaningful if the context actually changed"
        );
    }

    /// A device pointer from a displaced context is rejected rather
    /// than silently misread.
    ///
    /// This is what decides how loud the displacement report has to be.
    /// If the driver refuses the pointer, a consumer whose context was
    /// replaced gets a CUDA error at its next launch - confusing and
    /// far from the cause, but not silent. If instead the pointer reads
    /// as valid and returns another context's memory, the report is the
    /// only warning that will ever arrive, and its absence would cost
    /// wrong numbers rather than a failed call.
    ///
    /// Allocates under a non-primary context, lets `init` rebind the
    /// thread to the primary one, then reads the pointer back through
    /// the raw driver API while the primary context is current.
    ///
    /// Requires a CUDA device.
    #[test]
    fn a_pointer_from_a_displaced_context_is_refused_not_misread() {
        const N: usize = 256;
        const PATTERN: u8 = 0xAB;

        let foreign = CudaContext::new_non_primary(0, 0)
            .expect("a CUDA device is required for this test");
        foreign.bind_to_thread().expect("make the foreign context current");

        let mut dptr: cu::CUdeviceptr = 0;
        // SAFETY: the foreign context is current; dptr is an out
        // parameter written only on success.
        let alloc = unsafe { cu::cuMemAlloc_v2(&mut dptr, N) };
        assert_eq!(alloc, cu::CUresult::CUDA_SUCCESS, "allocate under the foreign context");
        // SAFETY: dptr owns N bytes in the current context.
        let set = unsafe { cu::cuMemsetD8_v2(dptr, PATTERN, N) };
        assert_eq!(set, cu::CUresult::CUDA_SUCCESS);
        // SAFETY: no arguments; drains the fill before the rebind.
        let sync = unsafe { cu::cuCtxSynchronize() };
        assert_eq!(sync, cu::CUresult::CUDA_SUCCESS);

        let peer = GpuPeer::init(GpuPeerConfig::default()).expect("peer init");
        assert!(
            peer.displaced_foreign_context(),
            "the peer must have replaced the context this pointer belongs to, \
             or the test is not exercising a displacement at all"
        );

        let mut back = [0u8; N];
        // SAFETY: back is N bytes of writable host memory. dptr belongs
        // to the displaced context, which is the condition under test;
        // the driver reports what it thinks of that through the return
        // code rather than by writing.
        let copy = unsafe {
            cu::cuMemcpyDtoH_v2(back.as_mut_ptr().cast::<core::ffi::c_void>(), dptr, N)
        };

        // Which branch this takes is the finding, so it is reported
        // rather than only asserted: a one-sided assertion passes
        // whichever way the driver answers and would leave the question
        // open.
        if copy == cu::CUresult::CUDA_SUCCESS {
            let intact = back.iter().all(|&b| b == PATTERN);
            println!(
                "displaced-context pointer: driver ACCEPTED the read, bytes \
                 {} what was written",
                if intact { "match" } else { "DO NOT match" }
            );
            assert!(
                intact,
                "the driver accepted a pointer from the displaced context and \
                 returned bytes that are not the ones written to it. That is a \
                 silent cross-context read, and the displacement report is then \
                 the only warning a consumer will ever get"
            );
        } else {
            println!("displaced-context pointer: driver REFUSED the read with {copy:?}");
        }

        // SAFETY: freeing under whichever context now owns it; a
        // failure here is the same class of refusal being measured and
        // is not a reason to fail the test.
        let _free: cu::CUresult = unsafe { cu::cuMemFree_v2(dptr) };
    }

    /// The detector stays quiet when there is nothing to displace.
    ///
    /// Without this, a detector that reported unconditionally would
    /// pass the test above while being useless.
    #[test]
    fn init_reports_nothing_when_it_displaces_nothing() {
        let peer = GpuPeer::init(GpuPeerConfig::default())
            .expect("a CUDA device is required for this test");
        assert!(
            !peer.displaced_foreign_context(),
            "the peer's own primary context is not a foreign one"
        );
    }
}

//! KHL (K_inner=3 per-slot Vyukov gating) backed by a memory-
//! mapped file for cross-process work-stealing dispatch.
//!
//! The K_inner=3 batching + per-slot Vyukov seq pattern of the
//! in-process `crate::sched::khl_local`, on the cross-process
//! MMF substrate. Each slot carries up to 3
//! [`LineItem`]s (closure_id + 8-byte inline args) in one 64-byte
//! cache line; one cross-CCX / cross-socket / cross-host steal
//! delivers 3 jobs per coherence transfer.
//!
//! ## Why MMF KHL exists alongside Chase-Lev / KHPD / LOH / URD
//!
//! The cross-process deque-variants matrix measured KHPD (shared
//! head, per-slot per-line atomic) at 2.4x faster than Chase-Lev
//! K=1 on Zen+ producer-fast K=64. KHL (owner-private bottom,
//! per-slot Vyukov seq, K_inner=3) measured at 3.0x faster than
//! Chase-Lev K=1 - the headline win on the design cube. This
//! module is that shape's entry in the flynnel cross-process
//! backend matrix.
//!
//! ## K_radius / MOVDIR64B
//!
//! The module exposes MOVDIR64B as a pair of standalone helpers,
//! [`movdir64b`] and [`movdir64b_available`], for consumers that
//! own the publish path themselves (e.g., a custom KHL routing
//! layer sitting on top of this substrate). Callers detect the
//! feature with [`movdir64b_available`] (CPUID leaf 7 ECX bit 28;
//! Sapphire Rapids+ / Zen 5+) and use the intrinsic wrapper to
//! bypass the producer L1d when the consumer is at coherence
//! distance d >= 3 (cross-CCD / cross-socket / cross-host). The
//! body cache line goes straight to memory without entering the
//! producer's cache hierarchy, saving the read-for-ownership
//! upgrade and the eventual dirty-line eviction back to memory
//! that a cached publish would pay.
//!
//! The internal [`MmfKhlDeque::publish`] path uses regular cached
//! stores and does not invoke MOVDIR64B; callers wanting the
//! direct-store path build it on top of the intrinsic wrapper
//! themselves.
//!
//! ## Layout
//!
//! ```text
//! +-----------------------------+
//! | KhlMmfHeader (64B aligned)  |  magic, capacity, owner_pid,
//! |                             |  bottom, head (each on its
//! |                             |  own cache line)
//! +-----------------------------+
//! | KhlMmfSlot[0]   (64B)       |  seq + n_items + pad + 3 items
//! | KhlMmfSlot[1]               |
//! | ...                         |
//! | KhlMmfSlot[capacity - 1]    |
//! +-----------------------------+
//! ```

#![allow(clippy::missing_errors_doc)]

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use memmap2::{MmapMut, MmapOptions};

use super::khpd::LineItem;

/// Magic byte sequence marking a valid MMF-KHL file. Reads as
/// ASCII "FKHL" then a version byte.
pub const KHL_MMF_MAGIC: u64 = 0x464B_484C_0000_0001;

/// One slot is exactly one cache line.
pub const KHL_MMF_SLOT_SIZE: usize = 64;

/// Items per slot.
pub const KHL_MMF_LINE_ITEMS: usize = 3;

/// Header sits at file offset 0; cache-line aligned.
#[repr(C, align(64))]
pub struct KhlMmfHeader {
    /// Magic constant set to [`KHL_MMF_MAGIC`] on create.
    pub magic: u64,
    /// Number of slots; always a power of two.
    pub capacity: u64,
    /// PID of the owner process; cleared to 0 by `close_owner`.
    pub owner_pid: AtomicU64,
    /// Epoch counter advanced on owner shutdown.
    pub epoch: AtomicU64,
    /// Padding so bottom lands on its own cache line.
    pub _pad_meta: [u8; 24],
    /// Owner-private bottom counter. Owner stores Relaxed; thieves
    /// Acquire-load as an emptiness hint.
    pub bottom: AtomicI64,
    /// Padding so head lands on its own cache line.
    pub _pad_bottom: [u8; 56],
    /// Thief-side head counter. Thieves CAS this to claim a slot.
    pub head: AtomicI64,
    /// Padding rounding the header to two whole cache lines after
    /// head.
    pub _pad_head: [u8; 56],
}

/// One slot stored in the MMF buffer. 64 bytes exact.
#[repr(C, align(64))]
pub struct KhlMmfSlot {
    /// Vyukov publication sequence. Initialized to slot index `i`;
    /// producer publishes `b+1`; consumer releases `t+capacity`.
    pub seq: AtomicU64,
    /// Number of valid items in `items` (1..=3).
    pub n_items: u8,
    /// Padding to 16-byte boundary so `items` starts at offset 16.
    pub _pad: [u8; 7],
    /// Inline payload (only `items[..n_items]` are valid).
    pub items: [LineItem; KHL_MMF_LINE_ITEMS],
}

/// Total file size for a deque with `capacity` slots.
pub const fn khl_mmf_file_size(capacity: usize) -> usize {
    std::mem::size_of::<KhlMmfHeader>() + capacity * KHL_MMF_SLOT_SIZE
}

/// Outcome of [`MmfKhlDeque::publish`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishError {
    /// Items count exceeds [`KHL_MMF_LINE_ITEMS`].
    TooManyItems,
}

/// Outcome of [`MmfKhlDeque::pop`] / [`MmfKhlDeque::steal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Steal {
    /// Got a slot.
    Success(StealResult),
    /// Deque was empty.
    Empty,
    /// CAS-loss; caller should retry.
    Retry,
}

/// Snapshot of a popped or stolen slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StealResult {
    /// Number of valid items.
    pub n_items: u8,
    /// Inline payload (only `items[..n_items]` are valid).
    pub items: [LineItem; KHL_MMF_LINE_ITEMS],
}

/// MMF-backed KHL deque.
pub struct MmfKhlDeque {
    _file: File,
    mmap: MmapMut,
    capacity: usize,
    capacity_mask: i64,
}

// SAFETY: same justification as other MMF-backed types.
unsafe impl Send for MmfKhlDeque {}
unsafe impl Sync for MmfKhlDeque {}

impl MmfKhlDeque {
    /// Create a fresh deque file at `path` with `capacity` slots
    /// (rounded up to next pow2; minimum 2). Truncates any
    /// existing file.
    pub fn create<P: AsRef<Path>>(path: P, capacity: usize) -> io::Result<Self> {
        let capacity = capacity.max(2).next_power_of_two();
        let size = khl_mmf_file_size(capacity);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())?;
        file.set_len(size as u64)?;

        // SAFETY: protocol-gated access only; file size fixed.
        let mut mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };

        let header_ptr = mmap.as_mut_ptr() as *mut KhlMmfHeader;
        // SAFETY: mmap page-aligned, alignment satisfied.
        unsafe {
            (*header_ptr).magic = KHL_MMF_MAGIC;
            (*header_ptr).capacity = capacity as u64;
            (*header_ptr).owner_pid = AtomicU64::new(std::process::id() as u64);
            (*header_ptr).epoch = AtomicU64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_meta.as_mut_ptr(), 0, 24);
            (*header_ptr).bottom = AtomicI64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_bottom.as_mut_ptr(), 0, 56);
            (*header_ptr).head = AtomicI64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_head.as_mut_ptr(), 0, 56);
        }

        // Zero the slot buffer, then initialize seq[i] = i so the
        // first round of publishes (b = 0..cap) sees seq == b and
        // proceeds without waiting on a non-existent prior consumer.
        let slots_start = std::mem::size_of::<KhlMmfHeader>();
        // SAFETY: slot region is the unwritten tail of the map.
        unsafe {
            std::ptr::write_bytes(
                mmap.as_mut_ptr().add(slots_start),
                0,
                capacity * KHL_MMF_SLOT_SIZE,
            );
        }
        for i in 0..capacity {
            let slot_off = slots_start + i * KHL_MMF_SLOT_SIZE;
            // SAFETY: slot AtomicU64 at the head of each slot.
            unsafe {
                let seq_ptr = mmap.as_mut_ptr().add(slot_off) as *mut AtomicU64;
                (*seq_ptr).store(i as u64, Ordering::Relaxed);
            }
        }

        mmap.flush()?;

        Ok(Self {
            _file: file,
            mmap,
            capacity,
            capacity_mask: (capacity as i64) - 1,
        })
    }

    /// Open an existing deque file at `path`.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path.as_ref())?;
        let size = file.metadata()?.len() as usize;
        if size < std::mem::size_of::<KhlMmfHeader>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "khl-mmf file too small to contain header",
            ));
        }
        // SAFETY: same as `create`.
        let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };

        let header_ptr = mmap.as_ptr() as *const KhlMmfHeader;
        // SAFETY: map covers header.
        let (magic, capacity) = unsafe {
            ((*header_ptr).magic, (*header_ptr).capacity as usize)
        };
        if magic != KHL_MMF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("khl-mmf magic mismatch: got {magic:#x}, want {KHL_MMF_MAGIC:#x}"),
            ));
        }
        if !capacity.is_power_of_two() || capacity < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("khl-mmf capacity {capacity} is not a power of two >= 2"),
            ));
        }
        if size < khl_mmf_file_size(capacity) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("khl-mmf file size {size} below expected {}",
                    khl_mmf_file_size(capacity)),
            ));
        }

        Ok(Self {
            _file: file,
            mmap,
            capacity,
            capacity_mask: (capacity as i64) - 1,
        })
    }

    /// Slot count (always a power of two).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// PID of the owner process at create time, or 0 if cleared.
    pub fn owner_pid(&self) -> u64 {
        self.header().owner_pid.load(Ordering::Acquire)
    }

    /// Current epoch (informational; advanced on owner shutdown).
    pub fn epoch(&self) -> u64 {
        self.header().epoch.load(Ordering::Acquire)
    }

    /// Advance the epoch and zero the owner pid. Called by the
    /// owner before dropping the deque so peers can detect
    /// abandonment.
    pub fn close_owner(&self) {
        self.header().owner_pid.store(0, Ordering::Release);
        self.header().epoch.fetch_add(1, Ordering::Release);
    }

    fn header(&self) -> &KhlMmfHeader {
        // SAFETY: map sized to cover header.
        unsafe { &*(self.mmap.as_ptr() as *const KhlMmfHeader) }
    }

    fn slot_ptr(&self, idx: i64) -> *mut KhlMmfSlot {
        let slot_idx = (idx & self.capacity_mask) as usize;
        let off = std::mem::size_of::<KhlMmfHeader>() + slot_idx * KHL_MMF_SLOT_SIZE;
        // SAFETY: slot_idx in [0, capacity); off within mapped region.
        unsafe { self.mmap.as_ptr().add(off) as *mut KhlMmfSlot }
    }

    /// Snapshot the current `(head, bottom, size)` for tests / debug.
    /// Both loads are Acquire for a consistent point-in-time view.
    pub fn snapshot_size(&self) -> (i64, i64, i64) {
        let h = self.header();
        let t = h.head.load(Ordering::Acquire);
        let b = h.bottom.load(Ordering::Acquire);
        (t, b, b - t)
    }

    /// Owner-side publish. **Only the owner process may call this.**
    /// Spins on slot-release wait with bounded spin + yield.
    pub fn publish(
        &self,
        n_items: u8,
        items: [LineItem; KHL_MMF_LINE_ITEMS],
    ) -> Result<(), PublishError> {
        if n_items as usize > KHL_MMF_LINE_ITEMS {
            return Err(PublishError::TooManyItems);
        }
        let h = self.header();
        let b = h.bottom.load(Ordering::Relaxed);
        let slot = self.slot_ptr(b);
        // SAFETY: slot pointer is in-mapped; seq protocol gates body access.
        let slot_ref = unsafe { &*slot };
        let mut spins: u32 = 0;
        while slot_ref.seq.load(Ordering::Acquire) != b as u64 {
            spins = spins.wrapping_add(1);
            if spins & 63 == 0 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: we own the slot for this round per the seq
        // invariant; no consumer reads body until our seq Release-
        // store below.
        unsafe {
            (*slot).n_items = n_items;
            (*slot).items = items;
        }
        slot_ref.seq.store((b as u64) + 1, Ordering::Release);
        h.bottom.store(b + 1, Ordering::Relaxed);
        Ok(())
    }

    /// Owner-side pop. Mirrors thief steal but called by the
    /// owner process (lets the owner drain its own deque).
    pub fn pop(&self) -> Steal {
        self.steal_or_pop()
    }

    /// Thief-side steal (FIFO). Any process / thread may call this.
    pub fn steal(&self) -> Steal {
        self.steal_or_pop()
    }

    fn steal_or_pop(&self) -> Steal {
        let h = self.header();
        let t = h.head.load(Ordering::Acquire);
        let b = h.bottom.load(Ordering::Acquire);
        if t >= b {
            return Steal::Empty;
        }
        if h.head
            .compare_exchange(t, t + 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Steal::Retry;
        }
        let slot = self.slot_ptr(t);
        // SAFETY: claimed slot via head CAS; seq invariant.
        let slot_ref = unsafe { &*slot };
        let mut spins: u32 = 0;
        while slot_ref.seq.load(Ordering::Acquire) != (t as u64) + 1 {
            spins = spins.wrapping_add(1);
            if spins & 63 == 0 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: published; sole consumer.
        let n_items = unsafe { (*slot).n_items };
        let items = unsafe { (*slot).items };
        slot_ref.seq.store(
            (t as u64) + (self.capacity as u64),
            Ordering::Release,
        );
        Steal::Success(StealResult { n_items, items })
    }

    /// Force dirty pages to disk. Meaningful for durable deques;
    /// tmpfs / disposable deques no-op.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }

    /// Hint that a steal at current head is upcoming.
    #[inline]
    pub fn prefetch_for_steal(&self) {
        let t = self.header().head.load(Ordering::Relaxed);
        let slot = self.slot_ptr(t);
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: prefetch is no-fault hint.
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    slot as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::hint::black_box(slot);
        }
    }
}

/// CPUID-detected: does this host expose MOVDIR64B (CPUID leaf 7
/// subleaf 0 ECX bit 28)? Cached at first call. Returns false on
/// non-x86_64 targets.
///
/// MOVDIR64B is available on Sapphire Rapids+ (Intel) and Zen 5+
/// (AMD). It writes 64 bytes from src to dst without dirtying the
/// producer's L1d.
#[inline]
pub fn movdir64b_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static CACHE: OnceLock<bool> = OnceLock::new();
        *CACHE.get_or_init(|| {
            let res = std::arch::x86_64::__cpuid_count(7, 0);
            (res.ecx & (1 << 28)) != 0
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// MOVDIR64B intrinsic wrapper. Writes 64 bytes from `src` to
/// `dst` via a direct-store that bypasses producer L1d.
///
/// # Safety
///
/// - `dst` must be 64-byte aligned and point to 64 bytes of valid
///   writable memory.
/// - `src` must point to 64 bytes of valid readable memory (no
///   alignment requirement).
/// - Caller must have verified [`movdir64b_available`] returned
///   true on this host; calling this on a host without MOVDIR64B
///   support raises #UD.
///
/// Direct-stores have weak ordering vs other stores in the same
/// thread; if a publication signal must be visible AFTER the
/// direct-store, emit `SFENCE` between MOVDIR64B and the signal
/// store.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn movdir64b(dst: *mut u8, src: *const u8) {
    // Intel syntax: `MOVDIR64B r64, m512` - r64 holds the linear
    // address (dst) and m512 is the source memory reference (src).
    unsafe {
        core::arch::asm!(
            "movdir64b {dst}, [{src}]",
            dst = in(reg) dst,
            src = in(reg) src,
            options(nostack, preserves_flags),
        );
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn movdir64b(_dst: *mut u8, _src: *const u8) {
    unreachable!("movdir64b not supported on this target");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("flynnel_khl_mmf_{pid}_{nonce}_{name}.bin"));
        p
    }

    fn make_item(closure_id: u32, args: &[u8]) -> LineItem {
        LineItem::new(closure_id, 0, args).expect("build item")
    }

    fn batch(items: Vec<LineItem>) -> (u8, [LineItem; KHL_MMF_LINE_ITEMS]) {
        let n = items.len();
        let mut arr = [LineItem { closure_id: 0, latch_offset: 0, args_inline: [0u8; 8] }; KHL_MMF_LINE_ITEMS];
        for (i, item) in items.into_iter().enumerate() {
            arr[i] = item;
        }
        (n as u8, arr)
    }

    #[test]
    fn slot_is_exactly_one_cache_line() {
        assert_eq!(core::mem::size_of::<KhlMmfSlot>(), 64,
            "KhlMmfSlot must be exactly 64 bytes");
        assert_eq!(core::mem::align_of::<KhlMmfSlot>(), 64);
    }

    #[test]
    fn create_then_open_round_trips_header() {
        let path = temp_path("create_open");
        let _d = MmfKhlDeque::create(&path, 8).expect("create");
        let opened = MmfKhlDeque::open(&path).expect("open");
        assert_eq!(opened.capacity(), 8);
        assert_eq!(opened.owner_pid(), std::process::id() as u64);
        assert_eq!(opened.epoch(), 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn open_rejects_bad_magic() {
        let path = temp_path("bad_magic");
        std::fs::write(&path, vec![0u8; 16384]).expect("seed");
        let r = MmfKhlDeque::open(&path);
        assert!(r.is_err(), "open must reject bad-magic file");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn publish_then_steal_fifo() {
        let path = temp_path("publish_steal_fifo");
        let d = MmfKhlDeque::create(&path, 4).expect("create");
        let (n1, b1) = batch(vec![make_item(1, b"a")]);
        d.publish(n1, b1).expect("p1");
        let (n2, b2) = batch(vec![make_item(2, b"b"), make_item(3, b"c")]);
        d.publish(n2, b2).expect("p2");
        match d.steal() {
            Steal::Success(r) => {
                assert_eq!(r.n_items, 1);
                assert_eq!(r.items[0].closure_id, 1);
            }
            other => panic!("expected Success, got {other:?}"),
        }
        match d.steal() {
            Steal::Success(r) => {
                assert_eq!(r.n_items, 2);
                assert_eq!(r.items[0].closure_id, 2);
                assert_eq!(r.items[1].closure_id, 3);
            }
            other => panic!("expected Success, got {other:?}"),
        }
        assert!(matches!(d.steal(), Steal::Empty));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn snapshot_size_tracks_publish_steal() {
        let path = temp_path("snapshot");
        let d = MmfKhlDeque::create(&path, 4).expect("create");
        let (t0, b0, sz0) = d.snapshot_size();
        assert_eq!((t0, b0, sz0), (0, 0, 0));
        let (n1, b1) = batch(vec![make_item(1, b"")]);
        d.publish(n1, b1).expect("p1");
        let (n2, b2) = batch(vec![make_item(2, b"")]);
        d.publish(n2, b2).expect("p2");
        let (_, _, sz) = d.snapshot_size();
        assert_eq!(sz, 2);
        d.steal();
        let (_, _, sz) = d.snapshot_size();
        assert_eq!(sz, 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_steal_returns_empty() {
        let path = temp_path("empty_steal");
        let d = MmfKhlDeque::create(&path, 4).expect("create");
        assert!(matches!(d.steal(), Steal::Empty));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prefetch_for_steal_is_safe_on_empty() {
        let path = temp_path("prefetch_empty");
        let d = MmfKhlDeque::create(&path, 4).expect("create");
        d.prefetch_for_steal();
        d.prefetch_for_steal();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn close_owner_zeros_pid_and_advances_epoch() {
        let path = temp_path("close_owner");
        let d = MmfKhlDeque::create(&path, 2).expect("create");
        assert_eq!(d.owner_pid(), std::process::id() as u64);
        assert_eq!(d.epoch(), 0);
        d.close_owner();
        assert_eq!(d.owner_pid(), 0);
        assert_eq!(d.epoch(), 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn movdir64b_detection_does_not_fault() {
        // The CPUID probe is no-fault on any x86_64 host; on
        // non-x86_64 it returns false. Calling it confirms the
        // CPUID path itself does not raise.
        let available = movdir64b_available();
        // Result is hardware-dependent; just observe it.
        std::hint::black_box(available);
    }

    #[test]
    fn concurrent_owner_publish_and_thieves_no_double_take() {
        // Stress: producer publishes N batches; multiple thieves
        // race to steal. Total items consumed should equal total
        // items published.
        let path = temp_path("concurrent_no_double_take");
        let d = Arc::new(MmfKhlDeque::create(&path, 64).expect("create"));
        let n_batches = 500;
        let consumed = Arc::new(AtomicUsize::new(0));

        let mut thieves = Vec::new();
        for _ in 0..4 {
            let d = Arc::clone(&d);
            let consumed = Arc::clone(&consumed);
            thieves.push(thread::spawn(move || {
                let target = n_batches * 3;
                while consumed.load(Ordering::Relaxed) < target {
                    match d.steal() {
                        Steal::Success(r) => {
                            consumed.fetch_add(r.n_items as usize, Ordering::Relaxed);
                        }
                        Steal::Empty | Steal::Retry => std::thread::yield_now(),
                    }
                }
            }));
        }

        for i in 0..n_batches as u32 {
            let (n, b) = batch(vec![
                make_item(i, b""),
                make_item(i, b""),
                make_item(i, b""),
            ]);
            d.publish(n, b).expect("publish");
        }

        for h in thieves {
            h.join().expect("thief joined");
        }

        let total = n_batches * 3;
        assert_eq!(consumed.load(Ordering::Relaxed), total,
            "each item must be consumed exactly once");
        std::fs::remove_file(&path).ok();
    }
}

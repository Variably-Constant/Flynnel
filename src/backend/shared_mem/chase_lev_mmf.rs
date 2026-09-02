//! Fixed-capacity Chase-Lev work-stealing deque backed by a
//! memory-mapped file.
//!
//! Mirrors the classic Chase-Lev protocol (PPoPP 2005) - one owner
//! pushes and pops one end without atomics, any number of thieves
//! CAS the other end - except the underlying storage is an MMF so
//! a thief can live in a different OS process from the owner.
//!
//! The asymmetry is load-bearing: the owner pushes / pops one end
//! with `Relaxed` / `Release` stores on `bottom`; only thieves CAS
//! the other end (`top`), so the owner's hot path never touches a
//! contended atomic. The owner is whichever process created the
//! deque. Atomic ordering propagates across the process boundary
//! because the MMF pages are the same physical cache lines under
//! every mapping.
//!
//! ## Layout
//!
//! ```text
//! +-----------------------------+
//! | DequeHeader  (64B aligned)  |  magic, capacity, top, bottom,
//! |                             |  owner_pid, epoch
//! +-----------------------------+
//! | Slot[0]      (64B aligned)  |  closure_id, args_len,
//! |                             |  latch_offset, args_inline[48]
//! | Slot[1]                     |
//! | ...                         |
//! | Slot[capacity - 1]          |
//! +-----------------------------+
//! ```
//!
//! ## Protocol (single owner, many thieves)
//!
//! Owner push (no atomic on bottom; only Release-store):
//! ```text
//!   b = bottom.load(Relaxed)
//!   buffer[b mod capacity] = slot
//!   bottom.store(b + 1, Release)
//! ```
//!
//! Owner pop (LIFO end; races with thieves at b == t):
//! ```text
//!   b = bottom.load(Relaxed) - 1
//!   bottom.store(b, Relaxed)
//!   atomic::fence(SeqCst)
//!   t = top.load(Relaxed)
//!   if t > b: bottom = b + 1; return Empty
//!   if t == b:                                  // last item: race
//!       if !top.cas(t, t+1, SeqCst, Relaxed):   // thief beat us
//!           bottom = b + 1
//!           return Empty
//!       bottom = b + 1
//!       return Ok(buffer[b])                    // we got it
//!   return Ok(buffer[b])                        // no race possible
//! ```
//!
//! Thief steal (FIFO end; CAS-on-top):
//! ```text
//!   t = top.load(Acquire)
//!   atomic::fence(SeqCst)
//!   b = bottom.load(Acquire)
//!   if t >= b: return Empty
//!   slot = buffer[t mod capacity]
//!   if !top.cas(t, t+1, SeqCst, Relaxed): return Retry
//!   return Ok(slot)
//! ```
//!
//! The thief's Acquire-load of `bottom` synchronizes-with the
//! owner's Release-store, which happens-after the buffer write, so
//! stolen slot bytes are the bytes the owner wrote; the single CAS
//! on `top` linearizes competing thieves in any process.
//!
//! Not provided: resize (push returns [`PushError::Full`] at
//! capacity), environment-capturing closures (slots carry
//! `(closure_id, args)` by value; peers register handlers in
//! [`super::pass_registry`] first), and wake notification (the
//! [`crate::sched::dual_deque`] integration handles waking).

#![allow(clippy::missing_errors_doc)]

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering, fence};

use memmap2::{MmapMut, MmapOptions};

/// Magic byte sequence marking a valid Chase-Lev deque file. Reads
/// as ASCII "FLCL" then a version byte. Distinct from the latch
/// arena magic so a file confusion is rejected at open time.
pub const CHASE_LEV_MAGIC: u64 = 0x464C_434C_0000_0001;

/// One slot is exactly one cache line so adjacent slots never share
/// a coherence-traffic line.
pub const SLOT_SIZE: usize = 64;

/// Inline args capacity per slot = slot size minus the fixed-shape
/// header fields (closure_id u32 + args_len u32 + latch_offset u32
/// + reserved_tag u32 = 16 bytes; see [`RemoteJobSlot`]).
pub const ARGS_INLINE_BYTES: usize = SLOT_SIZE - 16;

/// Header sits at file offset 0; cache-line aligned so `top` and
/// `bottom` get dedicated lines and never share with the magic /
/// capacity / pid metadata.
#[repr(C, align(64))]
pub struct DequeHeader {
    /// Magic constant set to [`CHASE_LEV_MAGIC`] on `create`.
    pub magic: u64,
    /// Number of slots in the buffer; always a power of two.
    pub capacity: u64,
    /// Pid of the owner process; informational. Cleared to 0 on
    /// `close_owner` so peers can observe abandonment.
    pub owner_pid: AtomicU64,
    /// Epoch counter advanced by the owner on shutdown to invalidate
    /// in-flight thief observations. Peers may consult this in their
    /// own watchdog logic; the deque protocol itself ignores it.
    pub epoch: AtomicU64,
    /// Padding so `top` and `bottom` land on independent cache lines
    /// below the metadata block.
    pub _pad_meta: [u8; 24],
    /// Chase-Lev `top` counter. Thieves CAS this to claim a slot.
    pub top: AtomicI64,
    /// Padding so `bottom` lands on its own cache line.
    pub _pad_top: [u8; 56],
    /// Chase-Lev `bottom` counter. Owner stores this with Release on
    /// push (no atomic on the hot path).
    pub bottom: AtomicI64,
    /// Padding to round the header out to two whole cache lines after
    /// `bottom`.
    pub _pad_bottom: [u8; 56],
}

/// Job slot stored in the Chase-Lev buffer. Fixed-shape, 64 bytes,
/// process-portable.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteJobSlot {
    /// Pass-registry id; resolved on the receiving side.
    pub closure_id: u32,
    /// Length of `args_inline` actually used.
    pub args_len: u32,
    /// Byte offset within the companion latch arena where this job's
    /// result should be published; or `u32::MAX` for fire-and-forget.
    pub latch_offset: u32,
    /// Reserved 32-bit tag for caller use (variant / numa hint / etc).
    pub reserved_tag: u32,
    /// Inline argument payload. Decoded by the registered handler.
    pub args_inline: [u8; ARGS_INLINE_BYTES],
}

impl RemoteJobSlot {
    /// Construct a slot from caller fields. Rejects oversized args.
    pub fn new(
        closure_id: u32,
        latch_offset: u32,
        args: &[u8],
    ) -> Result<Self, PushError> {
        if args.len() > ARGS_INLINE_BYTES {
            return Err(PushError::PayloadTooLarge);
        }
        let mut s = Self {
            closure_id,
            args_len: args.len() as u32,
            latch_offset,
            reserved_tag: 0,
            args_inline: [0u8; ARGS_INLINE_BYTES],
        };
        s.args_inline[..args.len()].copy_from_slice(args);
        Ok(s)
    }

    /// Return the live argument bytes as a slice.
    pub fn args(&self) -> &[u8] {
        &self.args_inline[..self.args_len as usize]
    }
}

/// Total file size for a deque with `capacity` slots, including
/// header.
pub const fn deque_file_size(capacity: usize) -> usize {
    std::mem::size_of::<DequeHeader>() + capacity * SLOT_SIZE
}

/// Outcome of [`MmfChaseLevDeque::push`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushError {
    /// Deque at capacity; consumer hasn't caught up. Caller can
    /// spin, back off, or overflow into an in-heap deque.
    Full,
    /// Args payload exceeds [`ARGS_INLINE_BYTES`].
    PayloadTooLarge,
}

/// Outcome of [`MmfChaseLevDeque::pop`] / [`MmfChaseLevDeque::steal`].
/// Same three-arm Success / Empty / Retry shape as the in-process
/// [`crate::sched::chase_lev_local::Steal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Steal {
    /// Got a slot.
    Success(RemoteJobSlot),
    /// Deque was empty (no race; nothing to do).
    Empty,
    /// CAS-on-top lost to a competing thief; the caller should
    /// retry the steal loop.
    Retry,
}

/// MMF-backed Chase-Lev deque. Single owner (the process that
/// created the file), arbitrarily many thieves across processes.
pub struct MmfChaseLevDeque {
    _file: File,
    mmap: MmapMut,
    capacity: usize,
    capacity_mask: i64,
}

// SAFETY: Same justification as the other MMF-backed types in this
// module - the underlying mmap handle is Send+Sync per memmap2, and
// every slot/header access is gated by the Chase-Lev protocol's
// per-atomic ordering pairs.
unsafe impl Send for MmfChaseLevDeque {}
// SAFETY: Same justification as the `Send` impl directly above.
unsafe impl Sync for MmfChaseLevDeque {}

impl MmfChaseLevDeque {
    /// Create a fresh deque file at `path` with `capacity` slots
    /// (rounded up to the next power of two; minimum 2). Truncates
    /// any existing file. The current process is recorded as owner.
    pub fn create<P: AsRef<Path>>(path: P, capacity: usize) -> io::Result<Self> {
        let capacity = capacity.max(2).next_power_of_two();
        let size = deque_file_size(capacity);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())?;
        file.set_len(size as u64)?;

        // SAFETY: `map_mut` is unsafe because the kernel cannot
        // prevent another process from truncating or mutating the
        // backing file in ways that violate Rust's aliasing rules.
        // This call site upholds the soundness contract by writing
        // only through the Chase-Lev protocol's per-atomic ordering
        // pairs on `top` and `bottom`, which every accessor (owner
        // push / pop, thief steal) is required to follow; the file
        // size is fixed by `file.set_len` immediately above and never
        // shrunk for the lifetime of any mapping.
        let mut mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };

        let header_ptr = mmap.as_mut_ptr() as *mut DequeHeader;
        // SAFETY: mmap is page-aligned (well above the 64-byte
        // alignment DequeHeader requires); the map covers
        // `deque_file_size(capacity)` bytes by construction.
        unsafe {
            (*header_ptr).magic = CHASE_LEV_MAGIC;
            (*header_ptr).capacity = capacity as u64;
            (*header_ptr).owner_pid = AtomicU64::new(std::process::id() as u64);
            (*header_ptr).epoch = AtomicU64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_meta.as_mut_ptr(), 0, 24);
            (*header_ptr).top = AtomicI64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_top.as_mut_ptr(), 0, 56);
            (*header_ptr).bottom = AtomicI64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_bottom.as_mut_ptr(), 0, 56);
        }

        // Zero the slot buffer. The Chase-Lev protocol never reads a
        // slot whose index is not in [top, bottom), so zeroing is
        // defence-in-depth rather than load-bearing.
        let slots_start = std::mem::size_of::<DequeHeader>();
        // SAFETY: slots_start..slots_start + capacity*SLOT_SIZE is
        // exactly the unwritten tail of the map; write_bytes through
        // mmap pointer is sound for the full region.
        unsafe {
            std::ptr::write_bytes(
                mmap.as_mut_ptr().add(slots_start),
                0,
                capacity * SLOT_SIZE,
            );
        }

        mmap.flush()?;

        Ok(Self {
            _file: file,
            mmap,
            capacity,
            capacity_mask: (capacity as i64) - 1,
        })
    }

    /// Open an existing deque file at `path`. Validates magic +
    /// capacity headers. Use this in peer processes that want to
    /// steal from a deque some other process created.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path.as_ref())?;
        let size = file.metadata()?.len() as usize;
        if size < std::mem::size_of::<DequeHeader>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "deque file too small to contain header",
            ));
        }

        // SAFETY: Same justification as `create` - protocol access only.
        let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };

        let header_ptr = mmap.as_ptr() as *const DequeHeader;
        // SAFETY: map size verified to cover header; mmap alignment
        // exceeds header alignment.
        let (magic, capacity) = unsafe {
            ((*header_ptr).magic, (*header_ptr).capacity as usize)
        };
        if magic != CHASE_LEV_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("chase-lev magic mismatch: got {magic:#x}, want {CHASE_LEV_MAGIC:#x}"),
            ));
        }
        if !capacity.is_power_of_two() || capacity < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("deque capacity {capacity} is not a power of two >= 2"),
            ));
        }
        if size < deque_file_size(capacity) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("deque file size {size} below expected {}", deque_file_size(capacity)),
            ));
        }

        Ok(Self {
            _file: file,
            mmap,
            capacity,
            capacity_mask: (capacity as i64) - 1,
        })
    }

    /// Slot count of this deque (always a power of two).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Pid of the owner process at create time, or 0 if cleared.
    pub fn owner_pid(&self) -> u64 {
        self.header().owner_pid.load(Ordering::Acquire)
    }

    /// Current epoch (informational; advanced by the owner on
    /// shutdown).
    pub fn epoch(&self) -> u64 {
        self.header().epoch.load(Ordering::Acquire)
    }

    /// Advance the epoch and zero the owner pid. Called by the
    /// owner before dropping the deque to let peers detect
    /// abandonment.
    pub fn close_owner(&self) {
        self.header().owner_pid.store(0, Ordering::Release);
        self.header().epoch.fetch_add(1, Ordering::Release);
    }

    fn header(&self) -> &DequeHeader {
        // SAFETY: map is sized to cover the header; alignment
        // satisfied by mmap's page alignment.
        unsafe { &*(self.mmap.as_ptr() as *const DequeHeader) }
    }

    fn slot_ptr(&self, idx: i64) -> *mut RemoteJobSlot {
        let slot_idx = (idx & self.capacity_mask) as usize;
        let off = std::mem::size_of::<DequeHeader>() + slot_idx * SLOT_SIZE;
        // SAFETY: `slot_idx` is in [0, capacity); `off` is within
        // the mapped region and 64-byte aligned.
        unsafe { self.mmap.as_ptr().add(off) as *mut RemoteJobSlot }
    }

    /// Snapshot the current `(top, bottom, size)` for tests / debug.
    /// Both loads are `Acquire` for consistency with the steal path.
    pub fn snapshot_size(&self) -> (i64, i64, i64) {
        let h = self.header();
        let t = h.top.load(Ordering::Acquire);
        let b = h.bottom.load(Ordering::Acquire);
        (t, b, b - t)
    }

    /// Owner-side push. **Only the owner process may call this.**
    /// Calling it from a non-owner process violates Chase-Lev's
    /// single-pusher invariant and corrupts the deque.
    pub fn push(&self, slot: RemoteJobSlot) -> Result<(), PushError> {
        let h = self.header();
        let b = h.bottom.load(Ordering::Relaxed);
        let t = h.top.load(Ordering::Acquire);
        let size = b - t;
        if size >= self.capacity as i64 {
            return Err(PushError::Full);
        }
        // Write the slot bytes, then Release-store `bottom` so a
        // thief's Acquire-load of `bottom` synchronizes-with this
        // store and sees the slot bytes.
        //
        // SAFETY: `slot_ptr(b)` is the in-bounds aligned pointer to
        // an UnsafeCell-equivalent slot whose access is gated by the
        // Chase-Lev protocol. Until the Release-store of `b+1`
        // below, no thief observes `bottom > b` and therefore no
        // thief reads this slot.
        unsafe {
            std::ptr::write(self.slot_ptr(b), slot);
        }
        h.bottom.store(b + 1, Ordering::Release);
        Ok(())
    }

    /// Owner-side pop (LIFO; same end the owner pushes to). Races
    /// with thieves at `b == t`; the embedded SeqCst fence + CAS
    /// linearize the race.
    ///
    /// **Only the owner process may call this.** Non-owner callers
    /// must use [`Self::steal`].
    pub fn pop(&self) -> Steal {
        let h = self.header();
        let b = h.bottom.load(Ordering::Relaxed) - 1;
        // Reserve our slot by writing `bottom = b` (the to-be-popped
        // index). Thieves see `top..bottom` shrinking by one.
        h.bottom.store(b, Ordering::Relaxed);
        // The SeqCst fence linearizes with thieves' SeqCst-loads on
        // top below (in the steal path) so this fence + the
        // subsequent top.load see a consistent (top, bottom) pair.
        fence(Ordering::SeqCst);
        let t = h.top.load(Ordering::Relaxed);
        if t > b {
            // Deque was empty; restore bottom and report.
            h.bottom.store(b + 1, Ordering::Relaxed);
            return Steal::Empty;
        }
        // SAFETY: `slot_ptr(b)` is in-bounds; the byte-copy below is
        // the standard Chase-Lev "owner reads its own write" path.
        // Until we either succeed the CAS (t == b case) or simply
        // return Success (t < b case), no other writer can touch
        // the slot.
        let slot = unsafe { std::ptr::read(self.slot_ptr(b)) };
        if t < b {
            // Multiple items in the deque: no race possible; we own
            // the slot at b.
            return Steal::Success(slot);
        }
        // t == b: single-item race against thieves. Try to claim
        // it by CAS-ing top from t -> t+1.
        let won = h
            .top
            .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok();
        // In every case below, restore bottom to b+1 (we either took
        // the slot via CAS, or a thief took it via their own CAS).
        h.bottom.store(b + 1, Ordering::Relaxed);
        if won {
            Steal::Success(slot)
        } else {
            // The slot we read may be stale (the winning thief read
            // its own copy too); we still return Empty because the
            // logical owner-side pop yielded nothing.
            Steal::Empty
        }
    }

    /// Thief-side steal (FIFO; opposite end from owner push/pop).
    /// Any process / thread may call this. Returns
    /// [`Steal::Retry`] when a competing thief beat us on the CAS;
    /// the caller's outer steal loop should retry.
    pub fn steal(&self) -> Steal {
        let h = self.header();
        let t = h.top.load(Ordering::Acquire);
        fence(Ordering::SeqCst);
        let b = h.bottom.load(Ordering::Acquire);
        if t >= b {
            return Steal::Empty;
        }
        // SAFETY: t is in [top, bottom); slot_ptr(t) is in-bounds.
        // The Acquire load of `bottom` above synchronizes-with the
        // owner's Release-store after writing the slot, so the
        // bytes we read here are the bytes the owner wrote.
        let slot = unsafe { std::ptr::read(self.slot_ptr(t)) };
        let won = h
            .top
            .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok();
        if won {
            Steal::Success(slot)
        } else {
            Steal::Retry
        }
    }

    /// Force any dirty pages to disk. Only meaningful for durable
    /// deques; tmpfs / disposable deques no-op.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }

    /// Hint that a steal at the current `top` is upcoming. Issues
    /// `PREFETCHT0` (`_mm_prefetch(_MM_HINT_T0)`) for the slot bytes
    /// the next `steal()` call will read. Best-effort: the CPU may
    /// ignore the hint; correctness is not affected.
    ///
    /// The intent is that a caller about to issue a steal can call
    /// `prefetch_for_steal` first, perform some unrelated work (a
    /// `K_inflight` ROB-window-worth of cycles), then call `steal()`.
    /// The unrelated work overlaps with the slot-line coherence
    /// transfer, hiding the ~60-80 ns cross-CCX miss that the steal's
    /// slot read would otherwise pay.
    ///
    /// On non-x86_64 targets this compiles to a `top.load(Relaxed)`
    /// plus pointer arithmetic only - the actual prefetch hint is a
    /// no-op intrinsic.
    #[inline]
    pub fn prefetch_for_steal(&self) {
        // Relaxed load: we are issuing a hint, not synchronizing. If
        // the read races with a concurrent thief's CAS the worst case
        // is that we prefetch a slot that gets claimed by someone
        // else - still a no-side-effect hint.
        let t = self.header().top.load(Ordering::Relaxed);
        let slot = self.slot_ptr(t);
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: `_mm_prefetch` is a no-side-effect hint
            // intrinsic that accepts any pointer value without
            // architectural fault on x86_64. `slot_ptr` returns an
            // in-bounds aligned pointer; even if `t` advances between
            // the load and the prefetch, the prefetched line is
            // either the right slot or the slot that wrapped into
            // that ring index - both are valid mapped memory inside
            // the deque file.
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    slot as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Reference the pointer so the load + index aren't
            // dead-code-eliminated; the architectural prefetch hint
            // has no stable cross-platform intrinsic.
            std::hint::black_box(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as O};
    use std::thread;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("flynnel_chase_lev_{pid}_{nonce}_{name}.bin"));
        p
    }

    fn dummy_slot(id: u32, args_len: u8) -> RemoteJobSlot {
        let args: Vec<u8> = (0..args_len).collect();
        RemoteJobSlot::new(id, u32::MAX, &args).expect("build slot")
    }

    #[test]
    fn create_then_open_round_trips_header() {
        let path = temp_path("create_open");
        let _d = MmfChaseLevDeque::create(&path, 8).expect("create");
        let opened = MmfChaseLevDeque::open(&path).expect("open");
        assert_eq!(opened.capacity(), 8);
        assert_eq!(opened.owner_pid(), std::process::id() as u64);
        assert_eq!(opened.epoch(), 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn open_rejects_bad_magic() {
        let path = temp_path("bad_magic");
        std::fs::write(&path, vec![0u8; 16384]).expect("seed");
        let r = MmfChaseLevDeque::open(&path);
        assert!(r.is_err(), "open must reject bad-magic file");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn push_then_pop_lifo_owner_side() {
        let path = temp_path("push_pop_lifo");
        let d = MmfChaseLevDeque::create(&path, 4).expect("create");
        d.push(dummy_slot(1, 1)).expect("push 1");
        d.push(dummy_slot(2, 1)).expect("push 2");
        d.push(dummy_slot(3, 1)).expect("push 3");
        // LIFO: 3, then 2, then 1.
        let s = d.pop();
        assert!(matches!(s, Steal::Success(slot) if slot.closure_id == 3),
            "expected closure_id 3, got {s:?}");
        let s = d.pop();
        assert!(matches!(s, Steal::Success(slot) if slot.closure_id == 2));
        let s = d.pop();
        assert!(matches!(s, Steal::Success(slot) if slot.closure_id == 1));
        assert!(matches!(d.pop(), Steal::Empty));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn full_deque_reports_push_error() {
        let path = temp_path("full");
        let d = MmfChaseLevDeque::create(&path, 2).expect("create");
        d.push(dummy_slot(1, 0)).expect("push 1");
        d.push(dummy_slot(2, 0)).expect("push 2");
        let err = d.push(dummy_slot(3, 0)).expect_err("expected full");
        assert_eq!(err, PushError::Full);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn oversize_args_rejected() {
        let big = vec![0u8; ARGS_INLINE_BYTES + 1];
        let err = RemoteJobSlot::new(0, u32::MAX, &big).expect_err("expected oversize");
        assert_eq!(err, PushError::PayloadTooLarge);
    }

    #[test]
    fn steal_drains_owner_pushed_items_fifo() {
        let path = temp_path("steal_fifo");
        let d = MmfChaseLevDeque::create(&path, 8).expect("create");
        d.push(dummy_slot(1, 0)).expect("push 1");
        d.push(dummy_slot(2, 0)).expect("push 2");
        d.push(dummy_slot(3, 0)).expect("push 3");
        // Steal pulls from the FIFO end (1, 2, 3).
        let s = d.steal();
        assert!(matches!(s, Steal::Success(slot) if slot.closure_id == 1),
            "expected closure_id 1, got {s:?}");
        let s = d.steal();
        assert!(matches!(s, Steal::Success(slot) if slot.closure_id == 2));
        let s = d.steal();
        assert!(matches!(s, Steal::Success(slot) if slot.closure_id == 3));
        assert!(matches!(d.steal(), Steal::Empty));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn args_round_trip_through_slot() {
        let path = temp_path("args_round_trip");
        let d = MmfChaseLevDeque::create(&path, 2).expect("create");
        let payload: Vec<u8> = (0..32u8).collect();
        let slot = RemoteJobSlot::new(0xCAFE_F00D, 0x1234_5678, &payload)
            .expect("build slot");
        d.push(slot).expect("push");
        match d.pop() {
            Steal::Success(got) => {
                assert_eq!(got.closure_id, 0xCAFE_F00D);
                assert_eq!(got.latch_offset, 0x1234_5678);
                assert_eq!(got.args(), &payload[..]);
            }
            other => panic!("expected Success, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_deque_pop_returns_empty() {
        let path = temp_path("empty_pop");
        let d = MmfChaseLevDeque::create(&path, 4).expect("create");
        assert!(matches!(d.pop(), Steal::Empty));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_deque_steal_returns_empty() {
        let path = temp_path("empty_steal");
        let d = MmfChaseLevDeque::create(&path, 4).expect("create");
        assert!(matches!(d.steal(), Steal::Empty));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn snapshot_size_tracks_push_pop() {
        let path = temp_path("snapshot");
        let d = MmfChaseLevDeque::create(&path, 4).expect("create");
        let (t, b, sz) = d.snapshot_size();
        assert_eq!((t, b, sz), (0, 0, 0));
        d.push(dummy_slot(1, 0)).expect("push 1");
        d.push(dummy_slot(2, 0)).expect("push 2");
        let (_, _, sz) = d.snapshot_size();
        assert_eq!(sz, 2);
        // Drop the popped value via bare expression statement; the
        // value isn't bound because the test only cares that the
        // size shrinks by one after one pop.
        d.pop();
        let (_, _, sz) = d.snapshot_size();
        assert_eq!(sz, 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prefetch_for_steal_is_safe_on_empty_deque() {
        let path = temp_path("prefetch_empty");
        let d = MmfChaseLevDeque::create(&path, 4).expect("create");
        // No pushes yet; top.load reads 0; slot[0] is a zeroed line
        // inside the map. Prefetch must not fault.
        d.prefetch_for_steal();
        d.prefetch_for_steal();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prefetch_for_steal_does_not_disturb_state() {
        let path = temp_path("prefetch_no_disturb");
        let d = MmfChaseLevDeque::create(&path, 8).expect("create");
        d.push(dummy_slot(1, 0)).expect("push 1");
        d.push(dummy_slot(2, 0)).expect("push 2");
        let (t0, b0, sz0) = d.snapshot_size();
        // Prefetch is purely a CPU hint; it MUST NOT change the
        // observable deque state.
        for _ in 0..16 {
            d.prefetch_for_steal();
        }
        let (t1, b1, sz1) = d.snapshot_size();
        assert_eq!((t0, b0, sz0), (t1, b1, sz1),
            "prefetch_for_steal must be observable-state-neutral");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prefetch_followed_by_steal_returns_same_slot() {
        let path = temp_path("prefetch_then_steal");
        let d = MmfChaseLevDeque::create(&path, 4).expect("create");
        d.push(dummy_slot(0xAA, 0)).expect("push");
        d.prefetch_for_steal();
        match d.steal() {
            Steal::Success(s) => assert_eq!(s.closure_id, 0xAA),
            other => panic!("expected Success, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn close_owner_zeros_pid_and_advances_epoch() {
        let path = temp_path("close_owner");
        let d = MmfChaseLevDeque::create(&path, 2).expect("create");
        assert_eq!(d.owner_pid(), std::process::id() as u64);
        assert_eq!(d.epoch(), 0);
        d.close_owner();
        assert_eq!(d.owner_pid(), 0);
        assert_eq!(d.epoch(), 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn owner_push_with_concurrent_thieves_no_double_take() {
        // The hardest invariant: across N owner pushes and many
        // thief steals, every slot is taken exactly once (between
        // owner pop + thief steal). Run a stress loop and check
        // sums match.
        let path = temp_path("concurrent_no_double_take");
        let d = Arc::new(MmfChaseLevDeque::create(&path, 64).expect("create"));
        let n = 5_000usize;

        let consumed = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));

        // Two thief threads.
        let mut thieves = Vec::new();
        for _ in 0..2 {
            let d = Arc::clone(&d);
            let consumed = Arc::clone(&consumed);
            let sum = Arc::clone(&sum);
            thieves.push(thread::spawn(move || {
                while consumed.load(O::Relaxed) < n {
                    match d.steal() {
                        Steal::Success(slot) => {
                            consumed.fetch_add(1, O::Relaxed);
                            sum.fetch_add(slot.closure_id as usize, O::Relaxed);
                        }
                        Steal::Empty | Steal::Retry => std::thread::yield_now(),
                    }
                }
            }));
        }

        // Owner pushes; periodically pops a few to interleave.
        for i in 0..n {
            loop {
                match d.push(dummy_slot(i as u32, 0)) {
                    Ok(()) => break,
                    Err(PushError::Full) => {
                        // Drain locally to keep the pipe moving;
                        // record the pop into the same counters
                        // so the invariant check below is total.
                        if let Steal::Success(slot) = d.pop() {
                            consumed.fetch_add(1, O::Relaxed);
                            sum.fetch_add(slot.closure_id as usize, O::Relaxed);
                        }
                    }
                    Err(other) => panic!("unexpected: {other:?}"),
                }
            }
        }
        // Drain whatever is left in the deque from the owner side.
        while consumed.load(O::Relaxed) < n {
            match d.pop() {
                Steal::Success(slot) => {
                    consumed.fetch_add(1, O::Relaxed);
                    sum.fetch_add(slot.closure_id as usize, O::Relaxed);
                }
                Steal::Empty => std::thread::yield_now(),
                Steal::Retry => std::thread::yield_now(),
            }
        }
        for h in thieves {
            h.join().expect("thief joined");
        }

        let expected: usize = (0..n).sum();
        assert_eq!(sum.load(O::Relaxed), expected,
            "sum invariant violated: each slot should be consumed exactly once");
        assert_eq!(consumed.load(O::Relaxed), n);
        std::fs::remove_file(&path).ok();
    }
}

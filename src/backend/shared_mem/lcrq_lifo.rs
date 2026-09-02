//! LOH: LCRQ-on-LIFO hybrid deque, MMF-backed.
//!
//! Owner-side LIFO of `LcrqJobSlot`s (no atomic on push / pop on the
//! hot path) plus an MMF-backed LCRQ-style ring as the cross-process
//! transport. Migration drains a batch of LIFO entries into the ring's
//! tail at the caller's chosen flush boundary.
//!
//! ## Why this shape
//!
//! Chase-Lev's owner-side push always Release-stores `bottom` and the
//! steal-side pays a CAS on `top` per claimed item. For a single-
//! item-in-flight request-reply workload that ~matches the cost of one
//! cache-line bounce; for a workload that publishes many items per
//! coherence interval (parallel-for fan-out, fork-join leaves) the
//! per-item bookkeeping is unamortized. LOH amortizes by letting the
//! owner stage many items in its private heap (no atomic) and pay one
//! ring tail update per migration batch.
//!
//! Trade-offs vs Chase-Lev MMF:
//! - **Owner push** drops from one Release-store on `bottom` per item
//!   to a plain `Vec::push` (`~3 ns`).
//! - **Migration** is one `tail.fetch_add(batch)` plus `batch` Release-
//!   stores on per-slot sequence numbers.
//! - **Thief steal** is one CAS on `head` (same shape as Chase-Lev's
//!   `top` CAS) plus a sequence-number check on the slot. The wasted-
//!   ticket race that pure-XADD LCRQ exhibits is avoided by gating
//!   the CAS on `head < tail`.
//! - **Inline args drop from 48 to 40 bytes** because each slot
//!   reserves 8 bytes for its Vyukov sequence number. Caller payloads
//!   that don't fit must use a separate transport (or
//!   [`super::chase_lev_mmf::MmfChaseLevDeque`] which has the wider
//!   inline payload).
//!
//! Where LOH wins per the cost model: bursty dispatch (parallel-for,
//! fan-out, fork-join leaves) where the per-burst migration amortizes
//! over many items per cache-line bounce. Where LOH does NOT win:
//! single-item request-reply, because there's no batching to amortize
//! against.

#![allow(clippy::missing_errors_doc)]

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering, fence};

use memmap2::{MmapMut, MmapOptions};

/// Magic byte sequence marking a valid LOH deque file. Reads as
/// ASCII "FLLO" then a version byte. Distinct from the Chase-Lev
/// deque magic and the latch arena magic so a file-confusion is
/// rejected at open time.
pub const LOH_MAGIC: u64 = 0x464C_4C4F_0000_0001;

/// One slot is exactly one cache line.
pub const LOH_SLOT_SIZE: usize = 64;

/// Inline args capacity per slot. Slot layout is one cache line:
/// `sequence: AtomicI64` (8 B) + `closure_id` (4 B) + `args_len`
/// (4 B) + `latch_offset` (4 B) + `reserved_tag` (4 B) + `args_inline`
/// (40 B) = 64 B.
pub const LOH_ARGS_INLINE_BYTES: usize = 40;

/// Header sits at file offset 0; cache-line aligned. `tail` and
/// `head` each get their own cache line so the producer-side
/// `tail.fetch_add` doesn't invalidate the consumer-side `head` line.
#[repr(C, align(64))]
pub struct LohHeader {
    /// Magic constant.
    pub magic: u64,
    /// Number of ring slots; always a power of two.
    pub capacity: u64,
    /// Pid of the owner process; informational. Cleared on
    /// `close_owner()`.
    pub owner_pid: AtomicU64,
    /// Epoch counter advanced by the owner on shutdown.
    pub epoch: AtomicU64,
    /// Padding to push `tail` onto its own cache line.
    pub _pad_meta: [u8; 24],
    /// Producer counter. Owner `fetch_add(batch_size)` during
    /// migration to claim a contiguous block of slots.
    pub tail: AtomicI64,
    /// Padding to push `head` onto its own cache line.
    pub _pad_tail: [u8; 56],
    /// Consumer counter. Thieves CAS this to claim a slot.
    pub head: AtomicI64,
    /// Padding round to two whole cache lines after `head`.
    pub _pad_head: [u8; 56],
}

/// Ring slot: sequence + Marshal-shaped payload. Fixed-shape,
/// 64 bytes, process-portable.
#[repr(C, align(64))]
pub struct LcrqJobSlot {
    /// Vyukov-style sequence number gating payload access:
    /// - On creation: `seq == idx` (slot is empty, ready to publish)
    /// - After producer Release-store: `seq == idx + 1` (published,
    ///   consumer may read)
    /// - After consumer Release-store: `seq == idx + capacity`
    ///   (consumed, ready for next round at `idx + capacity`)
    pub sequence: AtomicI64,
    /// Pass-registry id; resolved on the receiving side.
    pub closure_id: u32,
    /// Length of `args_inline` actually used.
    pub args_len: u32,
    /// Byte offset within the companion latch arena where this job's
    /// result should be published, or `u32::MAX` for fire-and-forget.
    pub latch_offset: u32,
    /// Reserved 32-bit tag for caller use (variant / numa hint / etc).
    pub reserved_tag: u32,
    /// Inline argument payload.
    pub args_inline: [u8; LOH_ARGS_INLINE_BYTES],
}

/// Owner-side LIFO entry. Equivalent to [`LcrqJobSlot`] without the
/// atomic sequence (the sequence lives in the ring; the LIFO is
/// process-private).
#[derive(Debug, Clone, Copy)]
pub struct LohLifoEntry {
    /// Pass-registry id.
    pub closure_id: u32,
    /// Length of `args_inline` actually used.
    pub args_len: u32,
    /// Latch-arena offset or `u32::MAX`.
    pub latch_offset: u32,
    /// Caller tag.
    pub reserved_tag: u32,
    /// Inline args.
    pub args_inline: [u8; LOH_ARGS_INLINE_BYTES],
}

impl LohLifoEntry {
    /// Build a LIFO entry; reject oversized args.
    pub fn new(closure_id: u32, latch_offset: u32, args: &[u8]) -> Result<Self, PushError> {
        if args.len() > LOH_ARGS_INLINE_BYTES {
            return Err(PushError::PayloadTooLarge);
        }
        let mut s = Self {
            closure_id,
            args_len: args.len() as u32,
            latch_offset,
            reserved_tag: 0,
            args_inline: [0u8; LOH_ARGS_INLINE_BYTES],
        };
        s.args_inline[..args.len()].copy_from_slice(args);
        Ok(s)
    }

    /// Live arg bytes.
    pub fn args(&self) -> &[u8] {
        &self.args_inline[..self.args_len as usize]
    }
}

/// Total file size for a ring with `capacity` slots, including
/// header.
pub const fn loh_file_size(capacity: usize) -> usize {
    std::mem::size_of::<LohHeader>() + capacity * LOH_SLOT_SIZE
}

/// Outcome of [`LohDeque::push`] / [`LohDeque::flush`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushError {
    /// Ring at capacity; consumer hasn't caught up. Caller may spin,
    /// back off, or report upstream pressure.
    Full,
    /// Args payload exceeds [`LOH_ARGS_INLINE_BYTES`].
    PayloadTooLarge,
    /// Owner-side LIFO at its soft cap; caller must `flush()`
    /// or back off before pushing more.
    LifoFull,
}

/// Outcome of [`LohDeque::steal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Steal {
    /// Got a slot.
    Success(StealResult),
    /// Ring was empty (no published item past `head`).
    Empty,
    /// CAS lost to a competing thief; outer loop should retry.
    Retry,
}

/// Slot payload returned by a successful steal. Equivalent to
/// [`LohLifoEntry`] but without trait noise; carries the same
/// fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StealResult {
    /// Pass-registry id.
    pub closure_id: u32,
    /// Length of `args_inline` actually used.
    pub args_len: u32,
    /// Latch-arena offset.
    pub latch_offset: u32,
    /// Caller tag.
    pub reserved_tag: u32,
    /// Inline args.
    pub args_inline: [u8; LOH_ARGS_INLINE_BYTES],
}

impl StealResult {
    /// Live arg bytes.
    pub fn args(&self) -> &[u8] {
        &self.args_inline[..self.args_len as usize]
    }
}

/// Max LIFO depth (soft cap; back-pressure boundary).
pub const DEFAULT_LIFO_CAP: usize = 256;

/// MMF-backed LOH deque. Single owner (the process that created the
/// file); arbitrarily many thieves across processes.
pub struct LohDeque {
    _file: File,
    mmap: MmapMut,
    capacity: usize,
    capacity_mask: i64,
    flush_threshold: usize,
    lifo_cap: usize,
    /// Owner-side LIFO. `Mutex` is uncontended on the hot path
    /// because, by protocol, only the originator thread pushes; the
    /// Mutex exists to satisfy `Sync` for the surrounding
    /// `Arc<LohDeque>` shape that `DispatchBackend` consumers want.
    local_lifo: Mutex<Vec<LohLifoEntry>>,
}

// SAFETY: All fields are Send. Mmap handle is Send+Sync per memmap2;
// every ring access goes through the LCRQ sequence-number protocol
// (per-slot Acquire/Release pair) so concurrent producers and
// consumers see a consistent view. The Mutex around the LIFO
// linearizes owner-side accesses across any thread the originator
// happens to schedule the push on.
unsafe impl Send for LohDeque {}
// SAFETY: Same justification as the `Send` impl directly above.
unsafe impl Sync for LohDeque {}

impl LohDeque {
    /// Create a fresh LOH deque file. `capacity` rounds up to the
    /// next power of two (min 2). `flush_threshold` is the LIFO
    /// length at which an auto-flush fires on the next push.
    pub fn create<P: AsRef<Path>>(
        path: P,
        capacity: usize,
        flush_threshold: usize,
    ) -> io::Result<Self> {
        let capacity = capacity.max(2).next_power_of_two();
        let size = loh_file_size(capacity);

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
        // only through the LCRQ per-slot sequence-number protocol;
        // the file size is fixed by `file.set_len` immediately
        // above and never shrunk for the lifetime of any mapping.
        let mut mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };

        let header_ptr = mmap.as_mut_ptr() as *mut LohHeader;
        // SAFETY: mmap is page-aligned (>= 64-byte alignment that
        // LohHeader requires); the map covers `loh_file_size(capacity)`
        // bytes by construction.
        unsafe {
            (*header_ptr).magic = LOH_MAGIC;
            (*header_ptr).capacity = capacity as u64;
            (*header_ptr).owner_pid = AtomicU64::new(std::process::id() as u64);
            (*header_ptr).epoch = AtomicU64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_meta.as_mut_ptr(), 0, 24);
            (*header_ptr).tail = AtomicI64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_tail.as_mut_ptr(), 0, 56);
            (*header_ptr).head = AtomicI64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_head.as_mut_ptr(), 0, 56);
        }

        // Initialize each slot's sequence to its index. On first
        // producer touch, `sequence == idx`, so the publisher knows
        // the slot is ready to publish (it writes payload, then
        // Release-stores `idx + 1`).
        let slots_start = std::mem::size_of::<LohHeader>();
        for i in 0..capacity {
            let off = slots_start + i * LOH_SLOT_SIZE;
            // SAFETY: `off + LOH_SLOT_SIZE <= loh_file_size(capacity)`
            // by construction; the cast to `*mut LcrqJobSlot` is sound
            // because the slot is `repr(C, align(64))` and `off` is a
            // multiple of 64.
            let slot_ptr = unsafe { mmap.as_mut_ptr().add(off) as *mut LcrqJobSlot };
            // SAFETY: `slot_ptr` is in-bounds + aligned; the cell
            // payload bytes are valid for any bit pattern.
            unsafe {
                (*slot_ptr).sequence = AtomicI64::new(i as i64);
                (*slot_ptr).closure_id = 0;
                (*slot_ptr).args_len = 0;
                (*slot_ptr).latch_offset = u32::MAX;
                (*slot_ptr).reserved_tag = 0;
                std::ptr::write_bytes(
                    (*slot_ptr).args_inline.as_mut_ptr(),
                    0,
                    LOH_ARGS_INLINE_BYTES,
                );
            }
        }

        mmap.flush()?;

        let flush_threshold = flush_threshold.max(1);
        Ok(Self {
            _file: file,
            mmap,
            capacity,
            capacity_mask: (capacity as i64) - 1,
            flush_threshold,
            lifo_cap: DEFAULT_LIFO_CAP,
            local_lifo: Mutex::new(Vec::with_capacity(DEFAULT_LIFO_CAP)),
        })
    }

    /// Open an existing LOH deque file. Validates magic + capacity.
    pub fn open<P: AsRef<Path>>(path: P, flush_threshold: usize) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path.as_ref())?;
        let size = file.metadata()?.len() as usize;
        if size < std::mem::size_of::<LohHeader>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "loh deque file too small to contain header",
            ));
        }

        // SAFETY: Same justification as `create` - protocol-only access.
        let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };

        let header_ptr = mmap.as_ptr() as *const LohHeader;
        // SAFETY: map size verified to cover header; mmap alignment
        // exceeds header alignment.
        let (magic, capacity) = unsafe { ((*header_ptr).magic, (*header_ptr).capacity as usize) };
        if magic != LOH_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("loh magic mismatch: got {magic:#x}, want {LOH_MAGIC:#x}"),
            ));
        }
        if !capacity.is_power_of_two() || capacity < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("loh capacity {capacity} is not a power of two >= 2"),
            ));
        }
        if size < loh_file_size(capacity) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "loh file size {size} below expected {}",
                    loh_file_size(capacity)
                ),
            ));
        }

        let flush_threshold = flush_threshold.max(1);
        Ok(Self {
            _file: file,
            mmap,
            capacity,
            capacity_mask: (capacity as i64) - 1,
            flush_threshold,
            lifo_cap: DEFAULT_LIFO_CAP,
            local_lifo: Mutex::new(Vec::with_capacity(DEFAULT_LIFO_CAP)),
        })
    }

    /// Slot count of the ring (always a power of two).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Configured flush threshold.
    pub fn flush_threshold(&self) -> usize {
        self.flush_threshold
    }

    /// Pid of the owner process at create time, or 0 if cleared.
    pub fn owner_pid(&self) -> u64 {
        self.header().owner_pid.load(Ordering::Acquire)
    }

    /// Owner shutdown: zero pid + advance epoch.
    pub fn close_owner(&self) {
        self.header().owner_pid.store(0, Ordering::Release);
        self.header().epoch.fetch_add(1, Ordering::Release);
    }

    fn header(&self) -> &LohHeader {
        // SAFETY: map covers the header; alignment is page-aligned.
        unsafe { &*(self.mmap.as_ptr() as *const LohHeader) }
    }

    fn slot_ptr(&self, idx: i64) -> *mut LcrqJobSlot {
        let slot_idx = (idx & self.capacity_mask) as usize;
        let off = std::mem::size_of::<LohHeader>() + slot_idx * LOH_SLOT_SIZE;
        // SAFETY: `slot_idx` is in [0, capacity); `off` is within the
        // mapped region and 64-byte aligned.
        unsafe { self.mmap.as_ptr().add(off) as *mut LcrqJobSlot }
    }

    /// Snapshot the current `(head, tail, ring_size, lifo_len)`.
    /// Loads are independent; the tuple is not a linearizable
    /// snapshot - useful for debug / introspection only.
    pub fn snapshot_size(&self) -> (i64, i64, i64, usize) {
        let h = self.header();
        let head = h.head.load(Ordering::Acquire);
        let tail = h.tail.load(Ordering::Acquire);
        let lifo_len = self
            .local_lifo
            .try_lock()
            .map(|g| g.len())
            .unwrap_or(0);
        (head, tail, tail - head, lifo_len)
    }

    /// Owner-side push. Stages the item in the local LIFO; when the
    /// LIFO reaches `flush_threshold` an automatic [`Self::flush`]
    /// fires that drains the LIFO into the ring tail.
    ///
    /// **Only the owner process may call this.**
    pub fn push(&self, entry: LohLifoEntry) -> Result<(), PushError> {
        let mut lifo = self
            .local_lifo
            .lock()
            .expect("LOH local LIFO mutex poisoned");
        if lifo.len() >= self.lifo_cap {
            return Err(PushError::LifoFull);
        }
        lifo.push(entry);
        if lifo.len() >= self.flush_threshold {
            // Flush from inside the lock to keep the LIFO consistent
            // with the migration count. If the flush fails (ring at
            // capacity), undo the push so the caller can retry with
            // a clean LIFO state (otherwise a retried `push(same i)`
            // would duplicate the entry, since `entry` is supposed
            // to denote "this one item to enqueue").
            if let Err(e) = self.flush_locked(&mut lifo) {
                lifo.pop();
                return Err(e);
            }
        }
        Ok(())
    }

    /// Owner-side explicit flush. Drains the local LIFO into the
    /// ring's tail in one batch (one `tail.fetch_add(N)` + N Release-
    /// stores). Returns the number of items migrated.
    pub fn flush(&self) -> Result<usize, PushError> {
        let mut lifo = self
            .local_lifo
            .lock()
            .expect("LOH local LIFO mutex poisoned");
        self.flush_locked(&mut lifo)
    }

    fn flush_locked(&self, lifo: &mut Vec<LohLifoEntry>) -> Result<usize, PushError> {
        let n = lifo.len();
        if n == 0 {
            return Ok(0);
        }
        // Reserve a contiguous tail block. If the ring would overflow,
        // back off without consuming the tail.
        let h = self.header();
        let head_snapshot = h.head.load(Ordering::Acquire);
        let tail_snapshot = h.tail.load(Ordering::Relaxed);
        if (tail_snapshot - head_snapshot + n as i64) > self.capacity as i64 {
            // Ring would overflow; report Full so caller can back off.
            // Items remain in the LIFO for next flush attempt.
            return Err(PushError::Full);
        }
        let base = h.tail.fetch_add(n as i64, Ordering::AcqRel);

        // Drain LIFO in FIFO order (oldest first) so the ring sees
        // items in their original push order. drain() avoids the
        // O(N) shift cost of pop()-into-reverse.
        for (i, entry) in lifo.drain(..).enumerate() {
            let idx = base + i as i64;
            let slot = self.slot_ptr(idx);
            // Spin-wait until the slot is publishable (sequence == idx).
            // For the owner path this should usually be already true
            // because head <= tail always and slot.sequence advances
            // past idx only when a consumer has taken it.
            // SAFETY: slot_ptr returns an in-bounds aligned pointer.
            unsafe {
                loop {
                    let seq = (*slot).sequence.load(Ordering::Acquire);
                    let diff = seq - idx;
                    if diff == 0 {
                        // Slot ready: consumer released the prior
                        // round (or this is the first publish, where
                        // init set sequence == idx).
                        break;
                    }
                    if diff < 0 {
                        // Prior round's consumer has not yet
                        // released. Spin until they do.
                        std::hint::spin_loop();
                        continue;
                    }
                    // diff > 0: the slot's sequence is for a future
                    // round. With a single producer and the
                    // capacity-check guard in flush_locked, this is
                    // unreachable; if it does happen it indicates a
                    // protocol invariant violation. Loud panic so the
                    // cause can be diagnosed instead of silently
                    // overwriting an unconsumed slot.
                    panic!(
                        "LOH producer protocol violation: slot[{}] seq={} ahead of idx={}",
                        idx & self.capacity_mask,
                        seq,
                        idx
                    );
                }
                (*slot).closure_id = entry.closure_id;
                (*slot).args_len = entry.args_len;
                (*slot).latch_offset = entry.latch_offset;
                (*slot).reserved_tag = entry.reserved_tag;
                std::ptr::write_bytes(
                    (*slot).args_inline.as_mut_ptr(),
                    0,
                    LOH_ARGS_INLINE_BYTES,
                );
                std::ptr::copy_nonoverlapping(
                    entry.args_inline.as_ptr(),
                    (*slot).args_inline.as_mut_ptr(),
                    entry.args_len as usize,
                );
                (*slot).sequence.store(idx + 1, Ordering::Release);
            }
        }
        Ok(n)
    }

    /// Owner-side pop from the local LIFO. Items not yet flushed to
    /// the ring may be retrieved locally without round-tripping
    /// through the ring. Used by the dual-deque worker that owns the
    /// originator side of the deque.
    pub fn pop_local(&self) -> Option<LohLifoEntry> {
        let mut lifo = self
            .local_lifo
            .lock()
            .expect("LOH local LIFO mutex poisoned");
        lifo.pop()
    }

    /// Thief-side steal. Race-free CAS-on-head with sequence-number
    /// validation on the slot. Returns [`Steal::Retry`] when a
    /// competing thief beat us on the head CAS.
    pub fn steal(&self) -> Steal {
        let h = self.header();
        let head = h.head.load(Ordering::Acquire);
        fence(Ordering::SeqCst);
        let tail = h.tail.load(Ordering::Acquire);
        if head >= tail {
            return Steal::Empty;
        }
        let slot = self.slot_ptr(head);
        // Check the sequence ahead of the CAS. The producer Release-
        // stores `head + 1` after writing the slot bytes; a value
        // less than that means the publisher hasn't reached the
        // Release yet, and a value greater than that means the ring
        // has wrapped and the producer has re-published this slot
        // for a future round (the head we loaded is stale). In both
        // cases the thief must re-load head and try again.
        //
        // SAFETY: slot is in-bounds + aligned.
        let seq = unsafe { (*slot).sequence.load(Ordering::Acquire) };
        if seq != head + 1 {
            return Steal::Retry;
        }
        // Try to claim head. Once we win the CAS we own slot[head &
        // mask] for this round: the producer cannot re-publish the
        // slot until we release the sequence to `head + capacity`,
        // and the seq-check above already confirmed the publisher
        // released `head + 1`. The slot bytes we read below are the
        // bytes the producer wrote for this round.
        let won = h
            .head
            .compare_exchange(head, head + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok();
        if !won {
            return Steal::Retry;
        }
        // SAFETY: same as above; head is now ours and the producer's
        // Release on slot.sequence happens-before our Acquire load
        // of slot.sequence above.
        let result = unsafe {
            StealResult {
                closure_id: (*slot).closure_id,
                args_len: (*slot).args_len,
                latch_offset: (*slot).latch_offset,
                reserved_tag: (*slot).reserved_tag,
                args_inline: (*slot).args_inline,
            }
        };
        // Release the slot for the next round at `head + capacity`.
        // SAFETY: still our slot; the Release synchronizes with the
        // next producer's Acquire-spin in flush_locked.
        unsafe {
            (*slot)
                .sequence
                .store(head + self.capacity as i64, Ordering::Release);
        }
        Steal::Success(result)
    }

    /// Force any dirty pages to disk.
    pub fn flush_to_disk(&self) -> io::Result<()> {
        self.mmap.flush()
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
        p.push(format!("flynnel_loh_{pid}_{nonce}_{name}.bin"));
        p
    }

    fn entry(id: u32, args_len: u8) -> LohLifoEntry {
        let args: Vec<u8> = (0..args_len).collect();
        LohLifoEntry::new(id, u32::MAX, &args).expect("build entry")
    }

    #[test]
    fn create_then_open_round_trips_header() {
        let path = temp_path("create_open");
        let _d = LohDeque::create(&path, 8, 4).expect("create");
        let o = LohDeque::open(&path, 4).expect("open");
        assert_eq!(o.capacity(), 8);
        assert_eq!(o.owner_pid(), std::process::id() as u64);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn open_rejects_bad_magic() {
        let path = temp_path("bad_magic");
        std::fs::write(&path, vec![0xCDu8; 8192]).expect("seed");
        let r = LohDeque::open(&path, 4);
        assert!(r.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn push_and_explicit_flush_migrates() {
        let path = temp_path("flush");
        // flush_threshold = usize::MAX so auto-flush never fires;
        // the explicit `flush()` is the only path to the ring.
        let d = LohDeque::create(&path, 8, usize::MAX).expect("create");
        for i in 0..3u32 {
            d.push(entry(i, 1)).expect("push");
        }
        // Ring is still empty before flush.
        let (head, tail, sz, lifo_len) = d.snapshot_size();
        assert_eq!(head, 0);
        assert_eq!(tail, 0);
        assert_eq!(sz, 0);
        assert_eq!(lifo_len, 3);
        // Flush: 3 items migrate.
        let n = d.flush().expect("flush");
        assert_eq!(n, 3);
        let (_, tail, sz, lifo_len) = d.snapshot_size();
        assert_eq!(tail, 3);
        assert_eq!(sz, 3);
        assert_eq!(lifo_len, 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn push_auto_flushes_at_threshold() {
        let path = temp_path("autoflush");
        let d = LohDeque::create(&path, 8, 4).expect("create");
        for i in 0..4u32 {
            d.push(entry(i, 0)).expect("push");
        }
        // The 4th push triggers auto-flush.
        let (_, tail, sz, lifo_len) = d.snapshot_size();
        assert_eq!(tail, 4);
        assert_eq!(sz, 4);
        assert_eq!(lifo_len, 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn steal_drains_in_fifo_order_after_flush() {
        let path = temp_path("fifo");
        let d = LohDeque::create(&path, 8, usize::MAX).expect("create");
        for i in 1..=3u32 {
            d.push(entry(i, 0)).expect("push");
        }
        d.flush().expect("flush");
        for expected in 1..=3u32 {
            loop {
                match d.steal() {
                    Steal::Success(slot) => {
                        assert_eq!(slot.closure_id, expected);
                        break;
                    }
                    Steal::Empty | Steal::Retry => std::thread::yield_now(),
                }
            }
        }
        assert!(matches!(d.steal(), Steal::Empty));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn pop_local_drains_lifo_in_lifo_order() {
        let path = temp_path("pop_local_lifo");
        let d = LohDeque::create(&path, 4, usize::MAX).expect("create");
        for i in 1..=3u32 {
            d.push(entry(i, 0)).expect("push");
        }
        // Owner pops in LIFO order (newest first).
        for expected in (1..=3u32).rev() {
            let e = d.pop_local().expect("pop_local");
            assert_eq!(e.closure_id, expected);
        }
        assert!(d.pop_local().is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn ring_full_at_capacity() {
        let path = temp_path("full");
        let d = LohDeque::create(&path, 2, usize::MAX).expect("create");
        d.push(entry(1, 0)).expect("push");
        d.push(entry(2, 0)).expect("push");
        let n = d.flush().expect("flush");
        assert_eq!(n, 2);
        // Ring is at capacity; pushing more + flushing should report
        // Full. (Auto-flush is off via usize::MAX threshold.)
        d.push(entry(3, 0)).expect("push to lifo");
        let err = d.flush().expect_err("flush past capacity");
        assert_eq!(err, PushError::Full);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn oversize_args_rejected() {
        let big = vec![0u8; LOH_ARGS_INLINE_BYTES + 1];
        let err = LohLifoEntry::new(0, 0, &big).expect_err("oversize");
        assert_eq!(err, PushError::PayloadTooLarge);
    }

    #[test]
    fn close_owner_zeros_pid_and_advances_epoch() {
        let path = temp_path("close");
        let d = LohDeque::create(&path, 2, 1).expect("create");
        assert_eq!(d.owner_pid(), std::process::id() as u64);
        let h = d.header();
        let before = h.epoch.load(O::Acquire);
        d.close_owner();
        assert_eq!(d.owner_pid(), 0);
        assert_eq!(h.epoch.load(O::Acquire), before + 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn concurrent_thieves_no_double_take() {
        // Stress: owner pushes + auto-flushes; two thief threads
        // race to drain. Every slot must be consumed exactly once.
        let path = temp_path("stress");
        let d = Arc::new(LohDeque::create(&path, 128, 8).expect("create"));
        let n = 5_000usize;

        let consumed = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));

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

        for i in 0..n {
            loop {
                match d.push(entry(i as u32, 0)) {
                    Ok(()) => break,
                    Err(PushError::LifoFull) | Err(PushError::Full) => {
                        std::thread::yield_now();
                        d.flush().ok();
                    }
                    Err(other) => panic!("push: {other:?}"),
                }
            }
        }
        d.flush().ok();
        for h in thieves {
            h.join().expect("thief");
        }
        let expected: usize = (0..n).sum();
        assert_eq!(sum.load(O::Relaxed), expected, "every slot consumed once");
        std::fs::remove_file(&path).ok();
    }
}

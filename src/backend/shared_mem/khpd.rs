//! KHPD: K-axis Hierarchical Publication Deque, MMF-backed.
//!
//! Owner publishes
//! to a ring of fixed-size **publication lines**; each line holds
//! `LINE_ITEMS = 3` items inline and is claimed atomically as a
//! group by a single thief CAS. The amortization win vs the
//! per-item LOH / Chase-Lev shape: one Release-store on the line's
//! state byte publishes K items together; one cache-line transfer
//! delivers K items to the claiming thief.
//!
//! ## Layout
//!
//! ```text
//! +-----------------------------+
//! | KhpdHeader (64B)            |  magic, capacity, owner_pid,
//! |                             |  tail, head (each on its own line)
//! +-----------------------------+
//! | PublicationLine[0]  (64B)   |  state (8B) + 3 LineItems (48B)
//! | PublicationLine[1]          |
//! | ...                         |
//! | PublicationLine[capacity-1] |
//! +-----------------------------+
//! ```
//!
//! Each `PublicationLine` is exactly one cache line so adjacent
//! lines never share a coherence-traffic line. `state` is an
//! `AtomicU64` packed as `(epoch: u32, n_items: u16, claim: u16)`;
//! the publisher increments `epoch` and stores `n_items` + claim =
//! READY in one store, the claimer CAS-takes by setting `claim =
//! CLAIMED`, and the consumer's release advances `state` to the
//! next round's epoch.
//!
//! ## Per-item line layout
//!
//! `LineItem` is 16 bytes: `closure_id` (4 B) plus `latch_offset`
//! (4 B) plus `args_inline` (8 B). For payloads larger than 8
//! bytes, the caller routes through
//! [`super::chase_lev_mmf::MmfChaseLevDeque`] (48 B per slot) or
//! [`super::lcrq_lifo::LohDeque`] (40 B per slot).
//!
//! This is the single-tier shape: one publication ring shared by
//! all thieves regardless of coherence distance. The in-process
//! per-tier equivalent is [`crate::sched::deque_tier`].

#![allow(clippy::missing_errors_doc)]

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering, fence};

use memmap2::{MmapMut, MmapOptions};

/// Magic byte sequence marking a valid KHPD file. ASCII "FKHP".
pub const KHPD_MAGIC: u64 = 0x464B_4850_0000_0001;

/// Cache-line size; one publication line per cache line.
pub const KHPD_LINE_SIZE: usize = 64;

/// Items per publication line. State (8 B) + 3 * 16 = 56 B; 8 B
/// trailing padding rounds the line to 64.
pub const LINE_ITEMS: usize = 3;

/// Bytes per inline args payload in one [`LineItem`].
pub const KHPD_ARGS_INLINE_BYTES: usize = 8;

/// `state` packed-bit-field layout: epoch in the top 32 bits,
/// `n_items` in the next 16, `claim` in the bottom 16.
const STATE_EMPTY: u64 = 0;
const CLAIM_BIT: u64 = 1;

/// Header sits at file offset 0; cache-line aligned. `head` and
/// `tail` each get their own cache line to prevent producer and
/// consumer counters from invalidating each other.
#[repr(C, align(64))]
pub struct KhpdHeader {
    /// Magic constant.
    pub magic: u64,
    /// Number of publication lines; always a power of two.
    pub capacity: u64,
    /// Pid of the owner process; informational. Cleared on
    /// `close_owner()`.
    pub owner_pid: AtomicU64,
    /// Epoch counter advanced on owner shutdown.
    pub epoch: AtomicU64,
    /// Padding to push `tail` to its own cache line.
    pub _pad_meta: [u8; 24],
    /// Producer counter. Owner `fetch_add(1)` per published line.
    pub tail: AtomicI64,
    /// Padding to push `head` to its own line.
    pub _pad_tail: [u8; 56],
    /// Consumer counter. Thieves CAS this to claim a line.
    pub head: AtomicI64,
    /// Padding to round the header to two whole cache lines after
    /// `head`.
    pub _pad_head: [u8; 56],
}

/// One item carried in a publication line. 16 bytes.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineItem {
    /// Pass-registry id.
    pub closure_id: u32,
    /// Latch arena offset.
    pub latch_offset: u32,
    /// Inline args (up to 8 bytes).
    pub args_inline: [u8; KHPD_ARGS_INLINE_BYTES],
}

impl LineItem {
    /// Build a line item from caller fields.
    pub fn new(closure_id: u32, latch_offset: u32, args: &[u8]) -> Result<Self, PushError> {
        if args.len() > KHPD_ARGS_INLINE_BYTES {
            return Err(PushError::PayloadTooLarge);
        }
        let mut item = Self {
            closure_id,
            latch_offset,
            args_inline: [0u8; KHPD_ARGS_INLINE_BYTES],
        };
        item.args_inline[..args.len()].copy_from_slice(args);
        Ok(item)
    }

    /// Live arg bytes.
    pub fn args(&self) -> &[u8] {
        // Without a separate `args_len` field, we treat the whole
        // 8-byte payload as the args. Callers that need a length
        // must encode it in the args themselves (the bench's adder
        // does exactly this: two u32s = 8 bytes; both halves are
        // meaningful).
        &self.args_inline
    }
}

/// One publication line: state + LINE_ITEMS items + padding.
#[repr(C, align(64))]
pub struct PublicationLine {
    /// `(epoch: u32) << 32 | (n_items: u16) << 16 | (claim: u16)`.
    /// `claim` = 0 (empty), 1 (READY for claim).
    pub state: AtomicU64,
    /// Inline items.
    pub items: [LineItem; LINE_ITEMS],
    /// Trailing padding to round to 64 bytes.
    pub _pad: [u8; 8],
}

/// Total file size for a KHPD with `capacity` publication lines.
pub const fn khpd_file_size(capacity: usize) -> usize {
    std::mem::size_of::<KhpdHeader>() + capacity * KHPD_LINE_SIZE
}

/// Outcome of [`KhpdDeque::publish`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushError {
    /// Ring at capacity; consumer hasn't caught up.
    Full,
    /// Items count exceeds [`LINE_ITEMS`].
    TooManyItems,
    /// Args payload exceeds [`KHPD_ARGS_INLINE_BYTES`].
    PayloadTooLarge,
}

/// Outcome of [`KhpdDeque::steal_line`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Steal {
    /// Got a publication line; carries up to LINE_ITEMS items.
    Success(StealResult),
    /// Ring was empty (head >= tail).
    Empty,
    /// Lost the CAS race on `head` to a competing thief.
    Retry,
}

/// Result of a successful steal: the publication line's items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StealResult {
    /// How many of the `items` slots are filled.
    pub n_items: usize,
    /// The items (only `items[..n_items]` are valid).
    pub items: [LineItem; LINE_ITEMS],
}

/// MMF-backed KHPD deque. Single owner, arbitrarily many thieves.
pub struct KhpdDeque {
    _file: File,
    mmap: MmapMut,
    capacity: usize,
    capacity_mask: i64,
    /// Owner-side staging buffer. Items accumulate here until the
    /// caller calls `publish()` to flush the buffer into a single
    /// publication line. `Mutex` is uncontended on the hot path
    /// (only the originator stages).
    pending: Mutex<Vec<LineItem>>,
}

// SAFETY: All fields are Send. Mmap handle is Send+Sync per memmap2.
// Every line access goes through the per-line state-atomic protocol;
// the pending Mutex linearizes owner-side accesses.
unsafe impl Send for KhpdDeque {}
// SAFETY: same justification as the Send impl directly above.
unsafe impl Sync for KhpdDeque {}

impl KhpdDeque {
    /// Create a fresh KHPD file. `capacity` rounds to the next
    /// power of two; min 2.
    pub fn create<P: AsRef<Path>>(path: P, capacity: usize) -> io::Result<Self> {
        let capacity = capacity.max(2).next_power_of_two();
        let size = khpd_file_size(capacity);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())?;
        file.set_len(size as u64)?;

        // SAFETY: `map_mut` is unsafe because the kernel cannot
        // prevent another process from truncating the file. This
        // call site upholds the soundness contract by writing only
        // through the KHPD per-line state-atomic protocol; file
        // size is fixed by `set_len` above and never shrunk for the
        // lifetime of any mapping.
        let mut mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };

        let header_ptr = mmap.as_mut_ptr() as *mut KhpdHeader;
        // SAFETY: mmap is page-aligned (well above the 64-byte
        // alignment KhpdHeader requires); the map covers
        // `khpd_file_size(capacity)` bytes by construction.
        unsafe {
            (*header_ptr).magic = KHPD_MAGIC;
            (*header_ptr).capacity = capacity as u64;
            (*header_ptr).owner_pid = AtomicU64::new(std::process::id() as u64);
            (*header_ptr).epoch = AtomicU64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_meta.as_mut_ptr(), 0, 24);
            (*header_ptr).tail = AtomicI64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_tail.as_mut_ptr(), 0, 56);
            (*header_ptr).head = AtomicI64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_head.as_mut_ptr(), 0, 56);
        }

        // Zero the lines (state == 0 == STATE_EMPTY).
        let lines_start = std::mem::size_of::<KhpdHeader>();
        // SAFETY: lines_start..lines_start + capacity*KHPD_LINE_SIZE
        // is the unwritten tail of the map.
        unsafe {
            std::ptr::write_bytes(
                mmap.as_mut_ptr().add(lines_start),
                0,
                capacity * KHPD_LINE_SIZE,
            );
        }

        mmap.flush()?;

        Ok(Self {
            _file: file,
            mmap,
            capacity,
            capacity_mask: (capacity as i64) - 1,
            pending: Mutex::new(Vec::with_capacity(LINE_ITEMS)),
        })
    }

    /// Open an existing KHPD file.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())?;
        let size = file.metadata()?.len() as usize;
        if size < std::mem::size_of::<KhpdHeader>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "khpd file too small",
            ));
        }
        // SAFETY: same protocol-only-access justification as `create`.
        let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };
        let header_ptr = mmap.as_ptr() as *const KhpdHeader;
        // SAFETY: map size verified to cover header.
        let (magic, capacity) =
            unsafe { ((*header_ptr).magic, (*header_ptr).capacity as usize) };
        if magic != KHPD_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("khpd magic mismatch {magic:#x}"),
            ));
        }
        if !capacity.is_power_of_two() || capacity < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("khpd capacity {capacity} not pow2 >= 2"),
            ));
        }
        if size < khpd_file_size(capacity) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "khpd file size {size} below expected {}",
                    khpd_file_size(capacity)
                ),
            ));
        }
        Ok(Self {
            _file: file,
            mmap,
            capacity,
            capacity_mask: (capacity as i64) - 1,
            pending: Mutex::new(Vec::with_capacity(LINE_ITEMS)),
        })
    }

    /// Capacity in publication lines (always a power of two).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Owner pid at create time, or 0 after `close_owner()`.
    pub fn owner_pid(&self) -> u64 {
        self.header().owner_pid.load(Ordering::Acquire)
    }

    /// Advance epoch + zero pid on shutdown.
    pub fn close_owner(&self) {
        self.header().owner_pid.store(0, Ordering::Release);
        self.header().epoch.fetch_add(1, Ordering::Release);
    }

    fn header(&self) -> &KhpdHeader {
        // SAFETY: header is at the start of the map; mmap is
        // page-aligned.
        unsafe { &*(self.mmap.as_ptr() as *const KhpdHeader) }
    }

    fn line_ptr(&self, idx: i64) -> *mut PublicationLine {
        let line_idx = (idx & self.capacity_mask) as usize;
        let off = std::mem::size_of::<KhpdHeader>() + line_idx * KHPD_LINE_SIZE;
        // SAFETY: `line_idx` < capacity; `off` is in-bounds + 64-byte aligned.
        unsafe { self.mmap.as_ptr().add(off) as *mut PublicationLine }
    }

    /// Snapshot `(head, tail, ring_size_lines, pending_items)`.
    pub fn snapshot_size(&self) -> (i64, i64, i64, usize) {
        let h = self.header();
        let head = h.head.load(Ordering::Acquire);
        let tail = h.tail.load(Ordering::Acquire);
        let pending = self
            .pending
            .try_lock()
            .map(|g| g.len())
            .unwrap_or(0);
        (head, tail, tail - head, pending)
    }

    /// Owner-side stage. Adds one item to the pending buffer.
    /// Returns the running pending count (so the caller can decide
    /// to flush at LINE_ITEMS). **Only the owner process may stage.**
    pub fn stage(&self, item: LineItem) -> Result<usize, PushError> {
        let mut p = self.pending.lock().expect("KHPD pending poisoned");
        p.push(item);
        Ok(p.len())
    }

    /// Owner-side publish. Drains the pending buffer into one or
    /// more publication lines (`LINE_ITEMS` items per line). Reserves
    /// the whole batch with ONE `tail.fetch_add(n_lines)` up front,
    /// then per-line: waits for `state == STATE_EMPTY`, fills items,
    /// and Release-stores the packed state. Returns the number of
    /// LINES published.
    pub fn publish(&self) -> Result<usize, PushError> {
        let mut p = self.pending.lock().expect("KHPD pending poisoned");
        if p.is_empty() {
            return Ok(0);
        }
        // How many lines do we need?
        let total = p.len();
        let n_lines = total.div_ceil(LINE_ITEMS);
        // Reserve `n_lines` ring slots. If the ring would overflow,
        // back off without consuming tail.
        let h = self.header();
        let head_snap = h.head.load(Ordering::Acquire);
        let tail_snap = h.tail.load(Ordering::Relaxed);
        if (tail_snap - head_snap + n_lines as i64) > self.capacity as i64 {
            return Err(PushError::Full);
        }
        let base = h.tail.fetch_add(n_lines as i64, Ordering::AcqRel);

        let mut item_iter = p.drain(..);
        for line_i in 0..n_lines {
            let idx = base + line_i as i64;
            let line = self.line_ptr(idx);
            // Spin-wait for the slot to be reusable. The line's
            // state has top-32-bit epoch == idx (initially 0; the
            // consumer's release advances it to `idx + 1` for
            // round 2). Producer's expected pre-publish state =
            // (idx_round << 32) where idx_round = idx / capacity.
            // For the first round at this physical slot, state
            // should be 0 (initial). For subsequent rounds, state
            // should be `(idx - capacity) << 32 | RELEASED`.
            //
            // Simpler invariant: consumer's release stores
            // `(idx + 1) << 32` (= empty for the next round). So
            // producer at `idx` spins until state's top 32 bits ==
            // `idx_top` where `idx_top = (idx / capacity) << 32` -
            // ah this gets tangled. The cleanest version:
            // consumer releases by storing `0` (empty). Producer
            // spins until state == 0.
            //
            // SAFETY: line is in-bounds + aligned.
            unsafe {
                loop {
                    let st = (*line).state.load(Ordering::Acquire);
                    if st == STATE_EMPTY {
                        break;
                    }
                    std::hint::spin_loop();
                }

                // Fill items.
                let mut n_filled = 0usize;
                for i in 0..LINE_ITEMS {
                    match item_iter.next() {
                        Some(item) => {
                            (*line).items[i] = item;
                            n_filled += 1;
                        }
                        None => break,
                    }
                }
                // Pack state: (epoch:32 from base+line_i) | n_filled
                // | CLAIM_BIT marker (=1 means READY).
                let new_state =
                    ((idx as u64) << 32) | ((n_filled as u64) << 16) | CLAIM_BIT;
                (*line).state.store(new_state, Ordering::Release);
            }
        }
        Ok(n_lines)
    }

    /// Thief-side. Claim one publication line.
    pub fn steal_line(&self) -> Steal {
        let h = self.header();
        let head = h.head.load(Ordering::Acquire);
        fence(Ordering::SeqCst);
        let tail = h.tail.load(Ordering::Acquire);
        if head >= tail {
            return Steal::Empty;
        }
        let line = self.line_ptr(head);
        // SAFETY: line in-bounds + aligned.
        let state = unsafe { (*line).state.load(Ordering::Acquire) };
        // Validate: state's epoch matches our head and CLAIM_BIT
        // is set.
        let expected_epoch = (head as u64) << 32;
        if state & 0xFFFF_FFFF_0000_0000 != expected_epoch {
            // Publisher hasn't written this line for our round yet.
            return Steal::Retry;
        }
        if state & CLAIM_BIT == 0 {
            // Line empty (no items for this round). Skip.
            return Steal::Retry;
        }
        // CAS head to claim.
        let won = h
            .head
            .compare_exchange(head, head + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok();
        if !won {
            return Steal::Retry;
        }
        // We own the line; read items + release state to EMPTY for
        // the next round at this physical slot.
        // SAFETY: line is in-bounds + aligned; CAS established
        // exclusive read access.
        let result = unsafe {
            let n_items = ((state >> 16) & 0xFFFF) as usize;
            let n_items = n_items.min(LINE_ITEMS);
            StealResult {
                n_items,
                items: (*line).items,
            }
        };
        // Release the line: store STATE_EMPTY so the next round's
        // producer (at idx = head + capacity) sees the slot ready.
        // SAFETY: still our line; the Release synchronizes with
        // the next producer's Acquire-spin in `publish`.
        unsafe {
            (*line).state.store(STATE_EMPTY, Ordering::Release);
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
        p.push(format!("flynnel_khpd_{pid}_{nonce}_{name}.bin"));
        p
    }

    fn item(id: u32) -> LineItem {
        LineItem::new(id, u32::MAX, &id.to_le_bytes()).expect("item")
    }

    #[test]
    fn create_open_round_trips_header() {
        let path = temp_path("create_open");
        let _d = KhpdDeque::create(&path, 8).expect("create");
        let o = KhpdDeque::open(&path).expect("open");
        assert_eq!(o.capacity(), 8);
        assert_eq!(o.owner_pid(), std::process::id() as u64);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn open_rejects_bad_magic() {
        let path = temp_path("badmagic");
        std::fs::write(&path, vec![0u8; 8192]).expect("seed");
        assert!(KhpdDeque::open(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn stage_then_publish_writes_one_line() {
        let path = temp_path("stage_publish");
        let d = KhpdDeque::create(&path, 4).expect("create");
        d.stage(item(1)).expect("stage 1");
        d.stage(item(2)).expect("stage 2");
        let lines = d.publish().expect("publish");
        assert_eq!(lines, 1);
        let (_, tail, sz, pending) = d.snapshot_size();
        assert_eq!(tail, 1);
        assert_eq!(sz, 1);
        assert_eq!(pending, 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn publish_spans_multiple_lines() {
        let path = temp_path("multi_line");
        let d = KhpdDeque::create(&path, 4).expect("create");
        // 7 items: 3 + 3 + 1 = 3 lines.
        for i in 1..=7u32 {
            d.stage(item(i)).expect("stage");
        }
        let lines = d.publish().expect("publish");
        assert_eq!(lines, 3);
        let (_, tail, sz, _) = d.snapshot_size();
        assert_eq!(tail, 3);
        assert_eq!(sz, 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn steal_returns_items_in_publication_order() {
        let path = temp_path("fifo");
        let d = KhpdDeque::create(&path, 4).expect("create");
        for i in 1..=5u32 {
            d.stage(item(i)).expect("stage");
        }
        d.publish().expect("publish");
        // Steal returns line 0 first (items 1,2,3), then line 1 (4,5).
        loop {
            match d.steal_line() {
                Steal::Success(r) => {
                    assert_eq!(r.n_items, 3);
                    assert_eq!(r.items[0].closure_id, 1);
                    assert_eq!(r.items[1].closure_id, 2);
                    assert_eq!(r.items[2].closure_id, 3);
                    break;
                }
                Steal::Empty | Steal::Retry => std::thread::yield_now(),
            }
        }
        loop {
            match d.steal_line() {
                Steal::Success(r) => {
                    assert_eq!(r.n_items, 2);
                    assert_eq!(r.items[0].closure_id, 4);
                    assert_eq!(r.items[1].closure_id, 5);
                    break;
                }
                Steal::Empty | Steal::Retry => std::thread::yield_now(),
            }
        }
        // Now empty.
        loop {
            match d.steal_line() {
                Steal::Empty => break,
                Steal::Retry => continue,
                Steal::Success(_) => panic!("unexpected success after drain"),
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn oversize_args_rejected() {
        let big = vec![0u8; KHPD_ARGS_INLINE_BYTES + 1];
        let err = LineItem::new(0, 0, &big).expect_err("oversize");
        assert_eq!(err, PushError::PayloadTooLarge);
    }

    #[test]
    fn ring_full_at_capacity_returns_full() {
        let path = temp_path("full");
        let d = KhpdDeque::create(&path, 2).expect("create");
        // Fill the ring (2 publication lines * 3 items = 6 items).
        for i in 1..=6u32 {
            d.stage(item(i)).expect("stage");
        }
        d.publish().expect("publish 2 lines");
        // Now stage more + publish; ring is full.
        d.stage(item(7)).expect("stage 7");
        let err = d.publish().expect_err("publish past capacity");
        assert_eq!(err, PushError::Full);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn close_owner_zeros_pid_and_advances_epoch() {
        let path = temp_path("close");
        let d = KhpdDeque::create(&path, 2).expect("create");
        let before = d.header().epoch.load(O::Acquire);
        d.close_owner();
        assert_eq!(d.owner_pid(), 0);
        assert_eq!(d.header().epoch.load(O::Acquire), before + 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn concurrent_thieves_no_double_take() {
        // Stress: 5000 items via repeated stage+publish; 2 thieves
        // race to drain. Every item must be consumed exactly once.
        let path = temp_path("stress");
        let d = Arc::new(KhpdDeque::create(&path, 64).expect("create"));
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
                    match d.steal_line() {
                        Steal::Success(r) => {
                            for i in 0..r.n_items {
                                consumed.fetch_add(1, O::Relaxed);
                                sum.fetch_add(r.items[i].closure_id as usize, O::Relaxed);
                            }
                        }
                        Steal::Empty | Steal::Retry => std::thread::yield_now(),
                    }
                }
            }));
        }

        // Publisher: stage LINE_ITEMS, publish, repeat until n items
        // total.
        let mut pushed = 0usize;
        while pushed < n {
            let want = LINE_ITEMS.min(n - pushed);
            for _ in 0..want {
                d.stage(item(pushed as u32)).expect("stage");
                pushed += 1;
            }
            loop {
                match d.publish() {
                    Ok(_) => break,
                    Err(PushError::Full) => {
                        std::thread::yield_now();
                    }
                    Err(other) => panic!("publish: {other:?}"),
                }
            }
        }

        for t in thieves {
            t.join().expect("thief");
        }
        let expected: usize = (0..n).sum();
        assert_eq!(
            sum.load(O::Relaxed),
            expected,
            "stress sum mismatch (expected every item consumed exactly once)"
        );
        std::fs::remove_file(&path).ok();
    }
}

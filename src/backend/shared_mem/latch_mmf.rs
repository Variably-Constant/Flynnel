//! Bump-allocated arena of fixed-size latch cells, backed by a
//! memory-mapped file.
//!
//! Each `LatchCell` is one 64-byte cache line: one `AtomicU8` state
//! byte plus a small result-bytes inline buffer. The arena's bump
//! pointer is itself in MMF so cross-process producers can allocate
//! latch cells without coordinating out-of-band.
//!
//! ## Role
//!
//! Carries the result-publication leg of the cross-process dispatch
//! shape paired with the MMF Chase-Lev deque
//! ([`super::chase_lev_mmf`]). The originator allocates a cell,
//! stamps its offset into a deque slot, then polls the cell's
//! `state` byte until the peer publishes the result inline; the
//! peer Release-stores `SET` after copying its reply bytes into the
//! cell, and the originator's `Acquire` load + read sees them.
//!
//! ## Layout
//!
//! ```text
//! +-----------------------------+
//! | LatchArenaHeader (64B)      |  magic, capacity, next_free
//! +-----------------------------+
//! | LatchCell[0]   (64B)        |  state + result_bytes[55]
//! | LatchCell[1]                |
//! | ...                         |
//! | LatchCell[capacity - 1]     |
//! +-----------------------------+
//! ```
//!
//! ## Protocol
//!
//! 1. **Originator** calls [`MmfLatchArena::alloc`] - bumps
//!    `next_free` with a `fetch_add` and returns the offset of a
//!    fresh `UNSET` cell.
//! 2. **Originator** writes the cell offset into the Chase-Lev
//!    deque slot's `latch_offset` field, then pushes the slot.
//! 3. **Thief** drains the slot, executes the handler, then calls
//!    [`MmfLatchArena::publish`] - copies the result bytes into the
//!    cell and Release-stores `state = SET`.
//! 4. **Originator** spins on [`MmfLatchArena::is_set`] (Acquire);
//!    once set, reads the result bytes with [`MmfLatchArena::read_result`].
//! 5. **Originator** optionally calls [`MmfLatchArena::reset`] to
//!    return the cell to `UNSET` for reuse (no free-list; cells age
//!    out as `next_free` wraps).
//!
//! Cells are 64 bytes wide so different requesters never share a
//! cache line; the per-cell `state` byte lives in its own line and
//! the originator's polling load never invalidates a sibling cell.

#![allow(clippy::missing_errors_doc)]

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use memmap2::{MmapMut, MmapOptions};

/// Magic byte sequence marking a valid latch arena file. Reads as
/// ASCII "FLLA" then a version byte. Distinct from the Chase-Lev
/// deque magic so a file confusion is rejected at open time.
pub const LATCH_ARENA_MAGIC: u64 = 0x464C_4C41_0000_0001;

/// Each latch cell is exactly one cache line so different requesters
/// poll on disjoint lines.
pub const LATCH_CELL_SIZE: usize = 64;

/// Result-bytes capacity per cell = [`LATCH_CELL_SIZE`] minus the
/// state byte and its alignment padding.
pub const RESULT_BYTES: usize = LATCH_CELL_SIZE - 8;

/// Cell state: not yet published.
pub const UNSET: u8 = 0;
/// Cell state: publisher has written `result_bytes` and Release-stored
/// this flag. Originator may Acquire-load and read the result.
pub const SET: u8 = 1;
/// Cell state: publisher reports an error condition. `result_bytes`
/// holds a UTF-8 diagnostic if `result_len > 0`.
pub const ERR: u8 = 2;

/// Header sits at file offset 0; cache-line aligned so `next_free`
/// has its own coherence-traffic line away from the cells.
#[repr(C, align(64))]
pub struct LatchArenaHeader {
    /// Magic constant set to [`LATCH_ARENA_MAGIC`] on `create`.
    pub magic: u64,
    /// Number of cells in the arena.
    pub capacity: u64,
    /// Bump-allocator counter; cells [0..next_free.load()) have been
    /// handed out at some point. Modulo `capacity` for wrap-around
    /// reuse.
    pub next_free: AtomicU64,
    /// Reserved for future per-arena policy bits.
    pub _reserved: [u8; 40],
}

/// One latch cell. Exactly 64 bytes; the layout is hand-packed to
/// keep the polled `state` byte at offset 0 (lowest cache-line byte)
/// so the load address is the cell's natural alignment.
#[repr(C, align(64))]
pub struct LatchCell {
    /// One of [`UNSET`], [`SET`], [`ERR`]. Originator polls with
    /// Acquire; publisher transitions with Release.
    pub state: AtomicU8,
    /// Length of the published result in `result_bytes`. Valid only
    /// when `state != UNSET`.
    pub result_len: u8,
    /// Padding so `result_bytes` starts on an 8-byte boundary.
    pub _pad: [u8; 6],
    /// Inline result payload. Read-after-Acquire is sound because
    /// the publisher Release-stored `state` after writing here.
    pub result_bytes: [u8; RESULT_BYTES],
}

/// Total file size for an arena with `capacity` cells, including
/// the header.
pub const fn arena_file_size(capacity: usize) -> usize {
    std::mem::size_of::<LatchArenaHeader>() + capacity * LATCH_CELL_SIZE
}

/// Errors from arena operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatchError {
    /// `alloc()` requested but arena's bump counter has wrapped past
    /// the cell with the given offset and the cell has not been
    /// reset. Caller should pick a different cell or wait.
    NotResetable(u32),
    /// `read_result` / `publish` called with an offset that is not
    /// 64-byte aligned within the arena.
    BadOffset(u32),
    /// `publish` payload larger than `RESULT_BYTES`.
    PayloadTooLarge,
}

/// Cross-thread / cross-process bump-allocated latch arena.
pub struct MmfLatchArena {
    _file: File,
    mmap: MmapMut,
    capacity: usize,
}

// SAFETY: The underlying mmap handle is Send + Sync per memmap2, and
// every cell access is gated by the per-cell Acquire/Release pair on
// `state`. The `UnsafeCell`-equivalent raw-pointer writes through the
// mmap are sound because exactly one publisher writes to a given cell
// between an `alloc()` that returns its offset and the subsequent
// `reset()` or wrap-around reuse.
unsafe impl Send for MmfLatchArena {}
// SAFETY: Same justification as the `Send` impl directly above.
unsafe impl Sync for MmfLatchArena {}

impl MmfLatchArena {
    /// Create a fresh arena file at `path` with `capacity` cells
    /// (rounded up to the next power of two; minimum 2). Truncates
    /// any existing file.
    pub fn create<P: AsRef<Path>>(path: P, capacity: usize) -> io::Result<Self> {
        let capacity = capacity.max(2).next_power_of_two();
        let size = arena_file_size(capacity);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())?;
        file.set_len(size as u64)?;

        // SAFETY: `map_mut` is unsafe because the kernel cannot
        // prevent another process from truncating or mutating the
        // backing file. This call site upholds the soundness contract
        // by writing only through the per-cell Acquire/Release
        // protocol on `state` (which every accessor follows); the
        // file size is fixed by `file.set_len` immediately above and
        // never shrunk for the lifetime of any mapping.
        let mut mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };

        let header_ptr = mmap.as_mut_ptr() as *mut LatchArenaHeader;
        // SAFETY: `mmap.as_mut_ptr()` is page-aligned (well above the
        // 64-byte alignment LatchArenaHeader requires) and the map
        // covers `arena_file_size(capacity)` bytes by construction.
        unsafe {
            (*header_ptr).magic = LATCH_ARENA_MAGIC;
            (*header_ptr).capacity = capacity as u64;
            (*header_ptr).next_free = AtomicU64::new(0);
            std::ptr::write_bytes((*header_ptr)._reserved.as_mut_ptr(), 0, 40);
        }

        let cells_start = std::mem::size_of::<LatchArenaHeader>();
        for i in 0..capacity {
            let off = cells_start + i * LATCH_CELL_SIZE;
            // SAFETY: `off + LATCH_CELL_SIZE <= arena_file_size(capacity)`
            // by construction; cast to `*mut LatchCell` is sound
            // because `LatchCell` is `repr(C, align(64))` and `off`
            // is a multiple of 64.
            let cell_ptr = unsafe { mmap.as_mut_ptr().add(off) as *mut LatchCell };
            // SAFETY: `cell_ptr` is the in-bounds aligned pointer
            // computed immediately above; the cell payload bytes are
            // valid for any bit pattern.
            unsafe {
                (*cell_ptr).state = AtomicU8::new(UNSET);
                (*cell_ptr).result_len = 0;
                std::ptr::write_bytes((*cell_ptr)._pad.as_mut_ptr(), 0, 6);
                std::ptr::write_bytes(
                    (*cell_ptr).result_bytes.as_mut_ptr(),
                    0,
                    RESULT_BYTES,
                );
            }
        }

        mmap.flush()?;

        Ok(Self {
            _file: file,
            mmap,
            capacity,
        })
    }

    /// Open an existing arena file at `path`. Validates magic +
    /// capacity headers.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path.as_ref())?;
        let size = file.metadata()?.len() as usize;
        if size < std::mem::size_of::<LatchArenaHeader>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "latch arena file too small to contain header",
            ));
        }

        // SAFETY: Same justification as the `create` path above -
        // protocol-only access. Other processes may concurrently
        // mutate the cells through the per-cell Acquire/Release
        // protocol on `state`; we honor the same protocol here.
        let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };

        let header_ptr = mmap.as_ptr() as *const LatchArenaHeader;
        // SAFETY: map size verified to cover header; mmap alignment
        // exceeds header alignment.
        let (magic, capacity) = unsafe {
            ((*header_ptr).magic, (*header_ptr).capacity as usize)
        };
        if magic != LATCH_ARENA_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("latch arena magic mismatch: got {magic:#x}, want {LATCH_ARENA_MAGIC:#x}"),
            ));
        }
        if !capacity.is_power_of_two() || capacity < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("latch arena capacity {capacity} is not a power of two >= 2"),
            ));
        }
        if size < arena_file_size(capacity) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("latch arena file size {size} below expected {}", arena_file_size(capacity)),
            ));
        }

        Ok(Self {
            _file: file,
            mmap,
            capacity,
        })
    }

    /// Number of cells in this arena (always a power of two).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn header(&self) -> &LatchArenaHeader {
        // SAFETY: map is sized to cover the header; alignment
        // satisfied by mmap's page alignment.
        unsafe { &*(self.mmap.as_ptr() as *const LatchArenaHeader) }
    }

    fn cell(&self, idx: usize) -> &LatchCell {
        let off = std::mem::size_of::<LatchArenaHeader>() + idx * LATCH_CELL_SIZE;
        // SAFETY: idx < capacity by caller contract; `off` is within
        // the mapped region and 64-byte aligned.
        unsafe { &*(self.mmap.as_ptr().add(off) as *const LatchCell) }
    }

    /// Allocate a fresh cell. Bumps `next_free` and returns the
    /// cell's byte offset within the file (suitable for stashing in
    /// a Chase-Lev slot's `latch_offset` field). The returned cell
    /// has `state = UNSET` (Release-stored) and `result_len = 0`.
    ///
    /// Note: `result_bytes` is NOT zeroed by `alloc`; it may still
    /// hold data from the previous publisher when a wrap-around
    /// reuses the cell. Safety of subsequent reads relies on
    /// `result_len` bounding the copy in [`Self::read_result`] and
    /// on [`Self::publish`] zeroing before writing new bytes.
    ///
    /// Cells wrap modulo capacity; reuse of an in-flight cell is
    /// caller-detectable via [`Self::is_set`] returning `true` for a
    /// cell that the caller did not just publish (impossible if the
    /// caller publishes promptly and `capacity` exceeds steady-state
    /// in-flight latch count).
    pub fn alloc(&self) -> u32 {
        let n = self.header().next_free.fetch_add(1, Ordering::Relaxed);
        let idx = (n & (self.capacity as u64 - 1)) as usize;
        let cell = self.cell(idx);
        // Reset the cell so the new owner observes a clean UNSET
        // even if the wrap-around case is hit.
        cell.state.store(UNSET, Ordering::Release);
        // SAFETY: Mutating `result_len` requires &mut access on the
        // cell, but we only hold &. The protocol guarantees no other
        // thread reads `result_len` until `state != UNSET`, and the
        // Release-store of UNSET above happens-before this write
        // (single thread, single function). Using the raw pointer
        // path is sound for the same reason `_pad` is initialized
        // through `write_bytes`: the cell layout is process-local
        // POD until the state Release is observed.
        unsafe {
            let cell_ptr = cell as *const _ as *mut LatchCell;
            (*cell_ptr).result_len = 0;
        }
        (std::mem::size_of::<LatchArenaHeader>() + idx * LATCH_CELL_SIZE) as u32
    }

    /// Translate a byte offset back to a cell index. Returns
    /// [`LatchError::BadOffset`] if the offset is not 64-byte
    /// aligned or falls outside the arena.
    fn offset_to_idx(&self, offset: u32) -> Result<usize, LatchError> {
        let header_size = std::mem::size_of::<LatchArenaHeader>();
        if (offset as usize) < header_size {
            return Err(LatchError::BadOffset(offset));
        }
        let rel = offset as usize - header_size;
        if !rel.is_multiple_of(LATCH_CELL_SIZE) {
            return Err(LatchError::BadOffset(offset));
        }
        let idx = rel / LATCH_CELL_SIZE;
        if idx >= self.capacity {
            return Err(LatchError::BadOffset(offset));
        }
        Ok(idx)
    }

    /// Test whether the cell at `offset` has been published. Acquire
    /// ordered so a subsequent read of `result_bytes` sees the
    /// publisher's writes.
    pub fn is_set(&self, offset: u32) -> Result<bool, LatchError> {
        let idx = self.offset_to_idx(offset)?;
        let s = self.cell(idx).state.load(Ordering::Acquire);
        Ok(s != UNSET)
    }

    /// Publisher path. Copies `payload` into the cell's
    /// `result_bytes` and Release-stores `state = SET`. Returns
    /// [`LatchError::PayloadTooLarge`] if the payload exceeds
    /// [`RESULT_BYTES`].
    pub fn publish(&self, offset: u32, payload: &[u8]) -> Result<(), LatchError> {
        if payload.len() > RESULT_BYTES {
            return Err(LatchError::PayloadTooLarge);
        }
        let idx = self.offset_to_idx(offset)?;
        let cell = self.cell(idx);
        // SAFETY: writes go through the protocol: we copy bytes then
        // Release-store state. Until that Release, no other thread
        // reads `result_bytes` (originator polls `state` with
        // Acquire). Casting to mut is sound because (a) we are the
        // only writer to this cell by protocol convention - exactly
        // one publisher per cell allocation - and (b) the cell
        // layout is POD outside the atomic.
        unsafe {
            let cell_ptr = cell as *const _ as *mut LatchCell;
            std::ptr::write_bytes(
                (*cell_ptr).result_bytes.as_mut_ptr(),
                0,
                RESULT_BYTES,
            );
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                (*cell_ptr).result_bytes.as_mut_ptr(),
                payload.len(),
            );
            (*cell_ptr).result_len = payload.len() as u8;
        }
        cell.state.store(SET, Ordering::Release);
        Ok(())
    }

    /// Publish an error condition into the cell. `state` transitions
    /// to [`ERR`]; the caller's `read_result` returns the diagnostic
    /// bytes.
    pub fn publish_err(&self, offset: u32, diagnostic: &[u8]) -> Result<(), LatchError> {
        if diagnostic.len() > RESULT_BYTES {
            return Err(LatchError::PayloadTooLarge);
        }
        let idx = self.offset_to_idx(offset)?;
        let cell = self.cell(idx);
        // SAFETY: same justification as `publish` - we hold the
        // single-publisher slot, Release-store gates the bytes.
        unsafe {
            let cell_ptr = cell as *const _ as *mut LatchCell;
            std::ptr::write_bytes(
                (*cell_ptr).result_bytes.as_mut_ptr(),
                0,
                RESULT_BYTES,
            );
            std::ptr::copy_nonoverlapping(
                diagnostic.as_ptr(),
                (*cell_ptr).result_bytes.as_mut_ptr(),
                diagnostic.len(),
            );
            (*cell_ptr).result_len = diagnostic.len() as u8;
        }
        cell.state.store(ERR, Ordering::Release);
        Ok(())
    }

    /// Read the published result bytes into `dst`. Returns the cell
    /// state (`SET` or `ERR`); call [`Self::is_set`] first to gate
    /// the read.
    pub fn read_result(&self, offset: u32, dst: &mut Vec<u8>) -> Result<u8, LatchError> {
        let idx = self.offset_to_idx(offset)?;
        let cell = self.cell(idx);
        let state = cell.state.load(Ordering::Acquire);
        let len = cell.result_len as usize;
        dst.clear();
        dst.resize(len, 0);
        // SAFETY: `result_bytes` is a fixed-size inline array of
        // length RESULT_BYTES; `len <= RESULT_BYTES` because
        // result_len is a u8 and we bounds-checked in publish. The
        // Acquire above synchronizes-with the publisher's Release
        // so the bytes we read are the bytes the publisher wrote.
        unsafe {
            std::ptr::copy_nonoverlapping(
                cell.result_bytes.as_ptr(),
                dst.as_mut_ptr(),
                len,
            );
        }
        Ok(state)
    }

    /// Reset a cell to `UNSET`. Used by the originator after
    /// consuming the result, freeing the cell for reuse before the
    /// bump counter wraps. No fence is needed beyond the Release
    /// because subsequent reuse will Release-store again on publish.
    pub fn reset(&self, offset: u32) -> Result<(), LatchError> {
        let idx = self.offset_to_idx(offset)?;
        self.cell(idx).state.store(UNSET, Ordering::Release);
        Ok(())
    }

    /// Force any dirty pages to disk. Only meaningful for durable
    /// arenas; tmpfs / disposable arenas no-op.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("flynnel_latch_arena_{pid}_{nonce}_{name}.bin"));
        p
    }

    #[test]
    fn create_then_open_validates_header() {
        let path = temp_path("create_open");
        let _arena = MmfLatchArena::create(&path, 16).expect("create");
        let reopen = MmfLatchArena::open(&path).expect("open");
        assert_eq!(reopen.capacity(), 16);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn open_rejects_bad_magic() {
        let path = temp_path("bad_magic");
        std::fs::write(&path, vec![0u8; 8192]).expect("seed");
        let r = MmfLatchArena::open(&path);
        assert!(r.is_err(), "open must reject bad-magic file");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn alloc_publish_read_round_trip() {
        let path = temp_path("alloc_publish_read");
        let arena = MmfLatchArena::create(&path, 4).expect("create");
        let off = arena.alloc();
        assert!(!arena.is_set(off).expect("is_set"));
        let payload = b"hello-latch";
        arena.publish(off, payload).expect("publish");
        assert!(arena.is_set(off).expect("is_set after publish"));
        let mut out = Vec::new();
        let s = arena.read_result(off, &mut out).expect("read");
        assert_eq!(s, SET);
        assert_eq!(&out[..], payload);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn publish_err_round_trips() {
        let path = temp_path("publish_err");
        let arena = MmfLatchArena::create(&path, 4).expect("create");
        let off = arena.alloc();
        arena.publish_err(off, b"oops").expect("publish_err");
        let mut out = Vec::new();
        let s = arena.read_result(off, &mut out).expect("read");
        assert_eq!(s, ERR);
        assert_eq!(&out[..], b"oops");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn reset_returns_cell_to_unset() {
        let path = temp_path("reset");
        let arena = MmfLatchArena::create(&path, 4).expect("create");
        let off = arena.alloc();
        arena.publish(off, b"x").expect("publish");
        arena.reset(off).expect("reset");
        assert!(!arena.is_set(off).expect("is_set after reset"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bad_offset_rejected() {
        let path = temp_path("bad_offset");
        let arena = MmfLatchArena::create(&path, 4).expect("create");
        // Offset inside the header is invalid.
        let r = arena.is_set(8);
        assert!(matches!(r, Err(LatchError::BadOffset(_))));
        // Misaligned offset is invalid.
        let r = arena.is_set((std::mem::size_of::<LatchArenaHeader>() + 1) as u32);
        assert!(matches!(r, Err(LatchError::BadOffset(_))));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn payload_too_large_rejected() {
        let path = temp_path("oversize");
        let arena = MmfLatchArena::create(&path, 2).expect("create");
        let off = arena.alloc();
        let big = vec![0u8; RESULT_BYTES + 1];
        let err = arena.publish(off, &big).expect_err("expected oversize");
        assert_eq!(err, LatchError::PayloadTooLarge);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn alloc_wraps_modulo_capacity() {
        let path = temp_path("wrap");
        let arena = MmfLatchArena::create(&path, 4).expect("create");
        let header_size = std::mem::size_of::<LatchArenaHeader>() as u32;
        let mut seen = Vec::new();
        for _ in 0..8 {
            seen.push(arena.alloc());
        }
        // First 4 allocs map to cells 0..3 ; next 4 wrap.
        assert_eq!(seen[0], header_size);
        assert_eq!(seen[4], header_size);
        assert_eq!(seen[1], header_size + LATCH_CELL_SIZE as u32);
        assert_eq!(seen[5], header_size + LATCH_CELL_SIZE as u32);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn two_threads_publish_and_observe() {
        let path = temp_path("two_threads");
        // Capacity must exceed `n` so the bump allocator does not
        // wrap and reuse cells before the publisher finishes -
        // wrap-around reuse would clobber in-flight values. Use
        // `n.next_power_of_two() * 2` for headroom.
        let n = 200usize;
        let capacity = (n * 2).next_power_of_two();
        let arena = Arc::new(MmfLatchArena::create(&path, capacity).expect("create"));

        // Originator side allocs cells, hands offsets to the publisher.
        let mut offsets = Vec::with_capacity(n);
        for _ in 0..n {
            offsets.push(arena.alloc());
        }

        let pub_arena = Arc::clone(&arena);
        let pub_offsets = offsets.clone();
        let pub_handle = thread::spawn(move || {
            for (i, off) in pub_offsets.iter().enumerate() {
                let bytes = (i as u32).to_le_bytes();
                pub_arena.publish(*off, &bytes).expect("publish");
            }
        });

        let mut buf = Vec::new();
        for (i, off) in offsets.iter().enumerate() {
            // Spin until publisher has set the cell.
            loop {
                if arena.is_set(*off).expect("is_set") {
                    break;
                }
                std::thread::yield_now();
            }
            let s = arena.read_result(*off, &mut buf).expect("read");
            assert_eq!(s, SET);
            let mut arr = [0u8; 4];
            arr.copy_from_slice(&buf[..4]);
            assert_eq!(u32::from_le_bytes(arr) as usize, i, "value mismatch at i={i}");
        }
        pub_handle.join().expect("publisher joined");
        std::fs::remove_file(&path).ok();
    }
}

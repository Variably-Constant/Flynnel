//! Background memory zeroing + first-touch for next-op allocation.
//!
//! Large BigFloat allocations at extreme precision (`K >= 14`,
//! many MB of mantissa) hit OS page-zeroing on first touch.
//! The thread that writes first pays the kernel's page-zero
//! cost in line with its own computation. A background thread
//! can pre-allocate + zero + first-touch the NEXT expected
//! operand on the right NUMA node ahead of compute so the
//! compute thread only sees an already-resident buffer.
//!
//! Lazy zero-fill costs ~200 ns per 4 KB page on first touch
//! (Linux / macOS; ~500 us for 10 MB), and Windows pays the eager
//! `MEM_COMMIT` zero in the allocating thread either way.
//!
//! [`prepare`] submits an allocate + first-touch closure to the
//! [`crate::sched::io_pool`] IoPool and returns a `Handle`; `wait`
//! returns immediately when the background work finished, blocks
//! otherwise, and runs inline when the IoPool is disabled. Callers
//! opt in explicitly; each `prepare` is a fresh allocation (no
//! cache), and NUMA placement beyond the IoPool worker's inherited
//! affinity is [`crate::sched::numa_alloc`]'s job.

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::sched::io_pool::global_io_pool;

/// Handle returned from [`prepare`]. Polls / waits the
/// background allocation.
pub struct Handle {
    inner: Arc<HandleInner>,
}

struct HandleInner {
    buffer: Mutex<Option<Vec<u8>>>,
    ready: AtomicBool,
}

impl Handle {
    /// `true` if the background allocation has already
    /// completed.
    pub fn is_ready(&self) -> bool {
        self.inner.ready.load(Ordering::Acquire)
    }

    /// Wait for the background allocation to complete and
    /// return the zero-initialized buffer. Blocks the caller
    /// if the worker has not finished yet (spin + short
    /// `thread::yield_now`).
    ///
    /// Returns `None` if the buffer was already taken by a
    /// prior `wait` call (handle is single-use).
    pub fn wait(self) -> Option<Vec<u8>> {
        while !self.inner.ready.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        self.inner.buffer.lock().ok()?.take()
    }
}

/// Submit a background zeroing request for `n_bytes`. If the
/// IoPool is enabled (`FLYNNEL_SCHED_SMT_AS_IO=on`), the worker
/// thread does the allocation + first-touch. Otherwise the
/// allocation runs inline in the caller's thread and the
/// returned handle is immediately ready (the caller pays the
/// page-zero cost up front, but the API contract is preserved).
///
/// First-touch is done via a single byte-write at every page
/// offset (the OS resolves the lazy zero-page fault on read,
/// not on write, so the inner loop walks the buffer at 4 KB
/// stride).
pub fn prepare(n_bytes: usize) -> Handle {
    let inner = Arc::new(HandleInner {
        buffer: Mutex::new(None),
        ready: AtomicBool::new(false),
    });

    let inner_for_worker = Arc::clone(&inner);
    let work = move || {
        let buf = allocate_and_first_touch(n_bytes);
        if let Ok(mut slot) = inner_for_worker.buffer.lock() {
            *slot = Some(buf);
        }
        inner_for_worker.ready.store(true, Ordering::Release);
    };

    match global_io_pool() {
        Some(pool) => {
            pool.submit(work);
        }
        None => {
            // No IoPool: do the work inline on the caller's
            // thread so the API still produces a usable handle.
            // The caller does not benefit from background
            // overlap in this configuration.
            work();
        }
    }

    Handle { inner }
}

/// Allocate `n_bytes` and write a single zero byte at every
/// page offset to force first-touch. On Linux / macOS this
/// resolves the lazy-zero page fault per page; on Windows
/// MEM_COMMIT already eager-zeroes so the first-touch step is
/// effectively a no-op.
fn allocate_and_first_touch(n_bytes: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n_bytes];
    let page_size = page_size_bytes();
    let mut offset = 0;
    while offset < buf.len() {
        // Writing 0 to an already-zero byte is a no-op
        // semantically but forces the page to be resident.
        // Use volatile so the optimizer does not elide.
        let ptr = buf.as_mut_ptr();
        // SAFETY: offset < buf.len() per the loop condition;
        // buf is a valid Vec<u8>.
        unsafe {
            std::ptr::write_volatile(ptr.add(offset), 0u8);
        }
        offset = offset.saturating_add(page_size);
    }
    buf
}

/// Conservative page-size estimate. 4 KB on every common
/// production platform; reading the actual value via syscall
/// is unnecessary for the first-touch loop (over-touching is
/// cheap; under-touching just leaves a few pages lazy).
fn page_size_bytes() -> usize {
    4096
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_returns_zero_filled_buffer_inline() {
        // With no IoPool the work runs inline.
        let handle = prepare(8192);
        assert!(handle.is_ready(), "inline path must be ready immediately");
        let buf = handle.wait().expect("buffer must be present");
        assert_eq!(buf.len(), 8192);
        assert!(buf.iter().all(|&b| b == 0), "buffer must be zero-filled");
    }

    #[test]
    fn prepare_zero_size_succeeds() {
        let handle = prepare(0);
        let buf = handle.wait().expect("zero-size buffer is still a buffer");
        assert!(buf.is_empty());
    }

    #[test]
    fn prepare_small_buffer_below_page_size() {
        let handle = prepare(128);
        let buf = handle.wait().unwrap();
        assert_eq!(buf.len(), 128);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn prepare_large_multi_page_buffer() {
        // 10 MB - representative of a K>=14 BigFloat operand.
        let handle = prepare(10 * 1024 * 1024);
        let buf = handle.wait().unwrap();
        assert_eq!(buf.len(), 10 * 1024 * 1024);
        // Check zero at a few offsets without scanning every
        // byte (the all-zero invariant is already established
        // by Vec::vec![0; n] semantics).
        for &offset in &[0usize, 4096, 1_048_576, 9_999_999] {
            assert_eq!(buf[offset], 0, "byte at offset {} must be zero", offset);
        }
    }

    #[test]
    fn handle_wait_returns_none_on_double_wait() {
        // The Handle API consumes the inner buffer on wait;
        // a second wait via the same Arc would return None.
        // We test this by cloning the inner Arc (the Handle
        // itself is consumed by wait).
        let handle = prepare(1024);
        let inner = Arc::clone(&handle.inner);
        let _ = handle.wait();
        // Now access via the cloned Arc: buffer slot is empty.
        let second = inner.buffer.lock().unwrap();
        assert!(second.is_none(), "buffer must be taken by the first wait");
    }

    #[test]
    fn page_size_estimate_is_reasonable() {
        // Sanity: the estimate is in the expected range and a
        // power of two.
        let p = page_size_bytes();
        assert!(p.is_power_of_two());
        assert!((4096..=65536).contains(&p));
    }

    #[test]
    fn first_touch_does_not_corrupt_buffer() {
        // After allocate_and_first_touch, the buffer must
        // remain zero-filled. We touched every page offset
        // with a write_volatile(0), which is semantically a
        // no-op on the value but forces residency.
        let buf = allocate_and_first_touch(16 * 4096);
        for (i, &b) in buf.iter().enumerate() {
            assert_eq!(b, 0, "first-touch must not corrupt byte {i}");
        }
    }
}

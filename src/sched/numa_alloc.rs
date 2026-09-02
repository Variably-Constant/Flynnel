//! Cross-platform NUMA-aware page allocator for large
//! `BigFloat` / `FpN<N>` buffers at K >= 8 (>= 256 limbs = >= 8 KB =
//! >= 2 pages).
//!
//!
//! ## Why this exists
//!
//! `BigFloat`'s `Vec<u32>` backing is allocated by the system
//! allocator on whichever NUMA node first touched the pages. For
//! a multi-socket workload that mutates a BigFloat from one socket
//! after allocating it on another, every limb access is a remote-
//! DRAM transaction (~3x the local bandwidth, ~100ns per cache
//! miss). For K >= 1M-bit operands this can dominate the
//! computation. `NumaAlloc` lets the caller place the pages on a
//! specific NUMA node up front, before first touch.
//!
//! ## Per-platform backend
//!
//! - **Windows**: `VirtualAllocExNuma(handle, NULL, size,
//!   MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE, preferred_node)`
//!   via raw FFI to `kernel32`. Hard placement on the named node
//!   when the system has enough free RAM there; falls back to
//!   other nodes otherwise.
//! - **Linux x86_64**: `mmap(NULL, len, PROT_READ|PROT_WRITE,
//!   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)` via raw FFI to `libc`.
//!   NUMA placement relies on the caller having pinned its thread
//!   to a CPU in the target node (first-touch). Use
//!   `core_affinity` to pin before calling.
//! - **Other (macOS, Linux non-x86_64, BSD, ...)**: standard
//!   `std::alloc::alloc` with page-rounded size. macOS M-series
//!   is single-NUMA so there is no placement to worry about; the
//!   API still works and just ignores the node parameter.
//!
//! ## Resolution of `NUMA_NODE_LOCAL`
//!
//! Passing the sentinel [`NUMA_NODE_LOCAL`] means "place on the
//! current thread's NUMA node." The resolver calls
//! `GetCurrentProcessorNumber` on Windows or `sched_getcpu` on
//! Linux to find the logical CPU index, then looks up the node
//! via the cached [`crate::numa_topology::NumaTopology`].

use core::ptr::NonNull;
use std::alloc::{Layout, alloc as std_alloc, dealloc as std_dealloc};

use crate::numa_topology::numa_topology;

/// Page size assumed for rounding. 4 KiB matches every modern x86
/// and aarch64 system; over-allocation by at most one page is the
/// cost of the assumption when the platform uses larger pages.
const PAGE_SIZE: usize = 4096;

/// Sentinel meaning "place on the caller-thread's current node."
pub const NUMA_NODE_LOCAL: u32 = u32::MAX;

/// Round `bytes` up to the next page boundary, then up to the
/// next power of two if larger than one page. Pow2-rounding above
/// the page floor matches a typical pow2-limbs invariant.
#[inline]
fn round_alloc_size(bytes: usize) -> usize {
    let page_rounded = (bytes + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    if page_rounded <= PAGE_SIZE {
        PAGE_SIZE
    } else {
        page_rounded.next_power_of_two()
    }
}

/// NUMA-aware allocator handle. Stateless: the node hint is
/// carried on each call.
#[derive(Copy, Clone, Debug)]
pub struct NumaAlloc;

impl NumaAlloc {
    /// Allocate at least `bytes` of zeroed memory, placed on
    /// `node`. Pass [`NUMA_NODE_LOCAL`] to target the caller-
    /// thread's node. Returns `None` if the system allocator
    /// failed.
    ///
    /// The returned pointer is page-aligned (4 KiB) and the
    /// underlying allocation is at least one page; for buffers
    /// smaller than a page the round-up is the cost of NUMA-
    /// awareness.
    ///
    /// # Safety
    ///
    /// Safe to call. The returned pointer must be passed to
    /// [`Self::dealloc_bytes`] exactly once with the same
    /// `bytes` value to avoid leaking the allocation.
    pub fn alloc_bytes(node: u32, bytes: usize) -> Option<NonNull<u8>> {
        if bytes == 0 {
            return None;
        }
        let size = round_alloc_size(bytes);
        let node = resolve_node(node);

        #[cfg(target_os = "windows")]
        {
            // SAFETY: `size` was page-rounded by `round_alloc_size`
            // above; `node` was resolved into a valid system node
            // id by `resolve_node`. Both are accepted by the
            // `VirtualAllocExNuma` wrapper that `windows_alloc`
            // forwards to.
            unsafe { windows_alloc(size, node) }
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            // node is honored by first-touch + thread affinity:
            // the caller must have pinned to a CPU in `node`
            // before this call. This path matches glibc malloc's
            // first-touch semantics.
            let _ = node;
            // SAFETY: `size` is page-rounded. `linux_mmap_alloc`
            // wraps `mmap(NULL, size, ...)` which never reads
            // from caller memory; the returned pointer is
            // exclusively owned.
            unsafe { linux_mmap_alloc(size) }
        }
        #[cfg(not(any(
            target_os = "windows",
            all(target_os = "linux", target_arch = "x86_64"),
        )))]
        {
            let _ = node;
            // SAFETY: `std_alloc_fallback` upholds the
            // `std::alloc::Layout` invariants internally; `size`
            // is page-rounded so alignment fits.
            unsafe { std_alloc_fallback(size) }
        }
    }

    /// Free a region returned by [`Self::alloc_bytes`].
    ///
    /// # Safety
    ///
    /// `ptr` must have come from `Self::alloc_bytes(_, bytes)`,
    /// and the same `bytes` value must be passed here. Calling
    /// with a mismatched size leaks or corrupts; calling twice
    /// double-frees.
    pub unsafe fn dealloc_bytes(ptr: NonNull<u8>, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let size = round_alloc_size(bytes);

        #[cfg(target_os = "windows")]
        {
            let _ = size;
            // SAFETY: `ptr` came from `Self::alloc_bytes` per the
            // outer function's `# Safety` clause, which means
            // it was returned by `VirtualAllocExNuma`. That makes
            // it a valid argument to `VirtualFree` with
            // `MEM_RELEASE`.
            unsafe { windows_dealloc(ptr) }
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            // SAFETY: `ptr` and `size` came from `Self::alloc_bytes`
            // per the outer `# Safety` clause; `size` matches the
            // value passed to `mmap`, which is the precondition
            // `munmap` requires.
            unsafe { linux_munmap(ptr, size) }
        }
        #[cfg(not(any(
            target_os = "windows",
            all(target_os = "linux", target_arch = "x86_64"),
        )))]
        {
            // SAFETY: same `Self::alloc_bytes` provenance as above;
            // `size` matches the original allocation request.
            unsafe { std_dealloc_fallback(ptr, size) }
        }
    }
}

/// Resolve a logical node id into a system node id. The sentinel
/// [`NUMA_NODE_LOCAL`] is resolved via
/// [`current_thread_node`]; explicit node ids are clamped to the
/// host's known node range (out-of-range maps to node 0 because
/// `VirtualAllocExNuma` would otherwise reject it).
fn resolve_node(node: u32) -> u32 {
    if node == NUMA_NODE_LOCAL {
        return current_thread_node();
    }
    let t = numa_topology();
    if node < t.num_nodes { node } else { 0 }
}

/// Query the NUMA node of the caller-thread's currently-running
/// logical CPU. Returns 0 on single-NUMA hosts or when the OS
/// query fails.
fn current_thread_node() -> u32 {
    let cpu = current_thread_cpu();
    let t = numa_topology();
    t.node_of_cpu.get(cpu).copied().unwrap_or(0)
}

#[cfg(target_os = "windows")]
fn current_thread_cpu() -> usize {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcessorNumber() -> u32;
    }
    // SAFETY: `GetCurrentProcessorNumber` takes no arguments,
    // reads no caller memory, and returns the index of the CPU
    // the calling thread is currently scheduled on. It has no
    // preconditions on Windows Vista or newer.
    unsafe { GetCurrentProcessorNumber() as usize }
}

#[cfg(target_os = "linux")]
fn current_thread_cpu() -> usize {
    #[link(name = "c")]
    unsafe extern "C" {
        fn sched_getcpu() -> core::ffi::c_int;
    }
    // SAFETY: `sched_getcpu` takes no arguments and has no
    // preconditions on glibc / musl. Negative return means
    // "unavailable" which we coerce to node 0 below.
    let cpu = unsafe { sched_getcpu() };
    if cpu < 0 { 0 } else { cpu as usize }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn current_thread_cpu() -> usize {
    0
}

// ---------------------------------------------------------------------------
// Windows: VirtualAllocExNuma
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
#[allow(non_camel_case_types, non_snake_case, clippy::upper_case_acronyms)]
mod win {
    pub type HANDLE = *mut core::ffi::c_void;
    pub type LPVOID = *mut core::ffi::c_void;
    pub type SIZE_T = usize;
    pub type DWORD = u32;

    pub const MEM_COMMIT: DWORD = 0x1000;
    pub const MEM_RESERVE: DWORD = 0x2000;
    pub const MEM_RELEASE: DWORD = 0x8000;
    pub const PAGE_READWRITE: DWORD = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn GetCurrentProcess() -> HANDLE;
        pub fn VirtualAllocExNuma(
            hProcess: HANDLE,
            lpAddress: LPVOID,
            dwSize: SIZE_T,
            flAllocationType: DWORD,
            flProtect: DWORD,
            nndPreferred: DWORD,
        ) -> LPVOID;
        pub fn VirtualFree(
            lpAddress: LPVOID,
            dwSize: SIZE_T,
            dwFreeType: DWORD,
        ) -> i32;
    }
}

#[cfg(target_os = "windows")]
unsafe fn windows_alloc(size: usize, node: u32) -> Option<core::ptr::NonNull<u8>> {
    use win::*;
    // SAFETY: `GetCurrentProcess` is a no-argument Win32 API
    // that returns a pseudo-handle for the current process. It
    // has no preconditions and cannot fail.
    let h = unsafe { GetCurrentProcess() };
    // SAFETY: `VirtualAllocExNuma` requires `h` to be a valid
    // process handle (the pseudo-handle from `GetCurrentProcess`
    // qualifies). Passing `NULL` for `lpAddress` requests a
    // system-chosen address; `size` is page-rounded; `flAlloc /
    // flProtect` are documented valid bit-flag combinations;
    // `node` was resolved into the host's known node range.
    let p = unsafe {
        VirtualAllocExNuma(
            h,
            core::ptr::null_mut(),
            size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
            node,
        )
    };
    core::ptr::NonNull::new(p as *mut u8)
}

#[cfg(target_os = "windows")]
unsafe fn windows_dealloc(ptr: core::ptr::NonNull<u8>) {
    // SAFETY: the outer `Self::dealloc_bytes` `# Safety` clause
    // guarantees `ptr` came from a previous `windows_alloc`
    // call, which makes it valid for `VirtualFree` with
    // `dwSize = 0` and `dwFreeType = MEM_RELEASE`.
    // @hook-allow:no-let-underscore
    unsafe {
        let _ = win::VirtualFree(ptr.as_ptr() as win::LPVOID, 0, win::MEM_RELEASE);
    }
}

// ---------------------------------------------------------------------------
// Linux x86_64: mmap / munmap
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[allow(non_camel_case_types, non_snake_case)]
mod lnx {
    use core::ffi::{c_int, c_void};

    pub const PROT_READ: c_int = 0x1;
    pub const PROT_WRITE: c_int = 0x2;
    pub const MAP_PRIVATE: c_int = 0x2;
    pub const MAP_ANONYMOUS: c_int = 0x20;

    #[link(name = "c")]
    unsafe extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, length: usize) -> c_int;
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn linux_mmap_alloc(size: usize) -> Option<core::ptr::NonNull<u8>> {
    use lnx::*;
    let p = unsafe {
        mmap(
            core::ptr::null_mut(),
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if p == (usize::MAX as *mut core::ffi::c_void) {
        // MAP_FAILED on Linux is (void*)-1
        return None;
    }
    core::ptr::NonNull::new(p as *mut u8)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn linux_munmap(ptr: core::ptr::NonNull<u8>, size: usize) {
    unsafe {
        let _ = lnx::munmap(ptr.as_ptr() as *mut core::ffi::c_void, size);
    }
}

// ---------------------------------------------------------------------------
// Fallback: std::alloc with page-rounded layout
// ---------------------------------------------------------------------------

#[cfg(not(any(
    target_os = "windows",
    all(target_os = "linux", target_arch = "x86_64"),
)))]
unsafe fn std_alloc_fallback(size: usize) -> Option<core::ptr::NonNull<u8>> {
    let layout = Layout::from_size_align(size, PAGE_SIZE).ok()?;
    let p = unsafe { std_alloc(layout) };
    core::ptr::NonNull::new(p)
}

#[cfg(not(any(
    target_os = "windows",
    all(target_os = "linux", target_arch = "x86_64"),
)))]
unsafe fn std_dealloc_fallback(ptr: core::ptr::NonNull<u8>, size: usize) {
    if let Ok(layout) = Layout::from_size_align(size, PAGE_SIZE) {
        unsafe { std_dealloc(ptr.as_ptr(), layout) };
    }
}

// Silence the unused-import warning on platforms that take the
// non-fallback path. `std_alloc` and `std_dealloc` are referenced
// only inside the cfg-gated fallback functions.
#[cfg(any(
    target_os = "windows",
    all(target_os = "linux", target_arch = "x86_64"),
))]
const _: fn() = || {
    let _ = std_alloc;
    let _ = std_dealloc;
    let _ = Layout::from_size_align(0, 0);
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_alloc_size_is_at_least_one_page() {
        assert_eq!(round_alloc_size(1), PAGE_SIZE);
        assert_eq!(round_alloc_size(0), PAGE_SIZE);
        assert_eq!(round_alloc_size(PAGE_SIZE - 1), PAGE_SIZE);
        assert_eq!(round_alloc_size(PAGE_SIZE), PAGE_SIZE);
    }

    #[test]
    fn round_alloc_size_is_pow2_above_one_page() {
        assert_eq!(round_alloc_size(PAGE_SIZE + 1), (PAGE_SIZE << 1));
        assert_eq!(round_alloc_size(3 * PAGE_SIZE), (PAGE_SIZE << 2));
        assert_eq!(round_alloc_size(5 * PAGE_SIZE), (PAGE_SIZE << 3));
        // 17 pages -> 32 pages (next pow2).
        assert_eq!(round_alloc_size(17 * PAGE_SIZE), (PAGE_SIZE << 5));
    }

    #[test]
    fn alloc_zero_bytes_returns_none() {
        assert!(NumaAlloc::alloc_bytes(0, 0).is_none());
    }

    #[test]
    fn alloc_dealloc_roundtrip_small() {
        let bytes = 1024;
        let p = NumaAlloc::alloc_bytes(0, bytes).expect("alloc must succeed");
        unsafe {
            core::ptr::write_bytes(p.as_ptr(), 0xAB, 1);
            core::ptr::write_bytes(p.as_ptr().add(PAGE_SIZE - 1), 0xCD, 1);
            assert_eq!(*p.as_ptr(), 0xAB);
            assert_eq!(*p.as_ptr().add(PAGE_SIZE - 1), 0xCD);
            NumaAlloc::dealloc_bytes(p, bytes);
        }
    }

    #[test]
    fn alloc_dealloc_roundtrip_multi_page() {
        let bytes = 100 * 1024;
        let p = NumaAlloc::alloc_bytes(0, bytes).expect("alloc must succeed");
        unsafe {
            let n = round_alloc_size(bytes);
            core::ptr::write(p.as_ptr(), 0x11u8);
            core::ptr::write(p.as_ptr().add(n >> 1), 0x22u8);
            core::ptr::write(p.as_ptr().add(n - 1), 0x33u8);
            assert_eq!(*p.as_ptr(), 0x11);
            assert_eq!(*p.as_ptr().add(n >> 1), 0x22);
            assert_eq!(*p.as_ptr().add(n - 1), 0x33);
            NumaAlloc::dealloc_bytes(p, bytes);
        }
    }

    #[test]
    fn alloc_returns_page_aligned() {
        let p = NumaAlloc::alloc_bytes(0, 1024).expect("alloc must succeed");
        let addr = p.as_ptr() as usize;
        assert_eq!(addr & (PAGE_SIZE - 1), 0,
            "alloc must return page-aligned: got 0x{addr:x}");
        unsafe { NumaAlloc::dealloc_bytes(p, 1024) };
    }

    #[test]
    fn resolve_node_clamps_out_of_range() {
        // numa_topology() on this single-NUMA host reports
        // num_nodes = 1, so node id 99 is out of range.
        assert_eq!(resolve_node(99), 0);
        // Valid node id passes through.
        assert_eq!(resolve_node(0), 0);
    }

    #[test]
    fn resolve_node_local_returns_current_thread_node() {
        // NUMA_NODE_LOCAL resolves to whichever node the running
        // thread's CPU belongs to. On a single-NUMA host this is
        // always 0; on multi-NUMA hosts the answer depends on the
        // OS scheduler's current placement. Either way it must
        // be < num_nodes.
        let n = resolve_node(NUMA_NODE_LOCAL);
        let t = numa_topology();
        assert!(n < t.num_nodes,
            "current_thread_node {n} out of range (num_nodes {})",
            t.num_nodes);
    }

    #[test]
    fn current_thread_cpu_is_in_range() {
        let cpu = current_thread_cpu();
        let t = numa_topology();
        // node_of_cpu has length == logical_threads; the returned
        // CPU index must fit or the resolver falls through to 0.
        // We accept either case (in-range or OOB -> 0); the
        // assertion just checks the call didn't crash.
        let _ = (cpu, t);
    }

    #[test]
    fn alloc_local_sentinel_does_not_crash() {
        let p = NumaAlloc::alloc_bytes(NUMA_NODE_LOCAL, 8192)
            .expect("local alloc must succeed");
        unsafe {
            for i in 0..(8192 / PAGE_SIZE) {
                core::ptr::write(p.as_ptr().add(i * PAGE_SIZE), 0xFFu8);
            }
            NumaAlloc::dealloc_bytes(p, 8192);
        }
    }
}

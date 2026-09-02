//! Async prefetch: warm cache for the NEXT chunk while the compute
//! pool processes the current one.
//!
//! ## Why
//!
//! Large NTT / Karatsuba kernels (K>=12 = >=4096 limbs = >=16 KB
//! per operand on `[u32]`) stall on RAM latency between chunks
//! because the working set exceeds L2 / L3. The standard streaming
//! pattern hides RAM latency by issuing prefetch hints ahead of
//! compute: while the compute thread processes chunk `N`, the
//! prefetch thread brings chunk `N+1` into the cache hierarchy.
//!
//! On modern x86_64 with SMT-2, the prefetch thread can run on the
//! SMT sibling of the compute core. SMT siblings share L1d/L2
//! (Zen+/Zen 3/4 split L1d but share L2 per CCX; Intel Skylake+
//! has shared L1d/L2 per physical core). A prefetch issued on the
//! sibling thread populates the cache line in a state the compute
//! thread can read with a hit instead of paying RAM latency.
//!
//! ## API
//!
//! [`prefetch_into_l2`] / [`prefetch_into_l3`] take a slice and
//! submit a prefetch-walk task to the
//! [`crate::sched::io_pool::IoPool`]. When the pool is disabled,
//! the calls are no-ops (prefetch is best-effort by design - missing
//! it is a perf miss, not a correctness bug).
//!
//! On non-x86_64 targets the prefetch hint compiles to a no-op
//! intrinsic; the cache-warm benefit isn't realized but the code
//! remains portable.
//!

use crate::sched::io_pool::global_io_pool;

/// Stride between prefetch hints: 64 bytes = one cache line on
/// x86_64. ARM uses 64 or 128 depending on the implementation;
/// 64 is a safe lower bound that issues at-most-2x the necessary
/// hints on ARM but still hits every line.
const PREFETCH_LINE_BYTES: usize = 64;

/// Issue a single prefetch hint targeting the L3 (NTA on Intel,
/// PREFETCHT2 on most x86). For data the consumer will use ONCE
/// soon (the streaming-NTT case); not displacing useful L1/L2
/// content.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch_hint_l3(ptr: *const u8) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T2};
    // SAFETY: `_mm_prefetch` is a no-side-effect hint intrinsic.
    // It accepts any pointer value (including dangling) without
    // architectural fault on x86_64; the CPU may ignore the hint
    // but never produces UB. The caller has marked this function
    // `unsafe` for ABI symmetry with the non-x86_64 stub.
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T2);
    }
}

/// Issue a single prefetch hint targeting L2 + L1 (PREFETCHT1).
/// For data the consumer will reuse multiple times in the near
/// future.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch_hint_l2(ptr: *const u8) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T1};
    // SAFETY: same reasoning as `prefetch_hint_l3` above.
    // `_mm_prefetch` is a no-side-effect CPU hint that accepts
    // any pointer value without architectural fault.
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T1);
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn prefetch_hint_l3(_ptr: *const u8) {
    // No-op on non-x86_64. ARM has PRFM but it's not exposed
    // through stable intrinsics; can be added under cfg(target_arch
    // = "aarch64") if a host-specific micro-bench shows it matters.
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn prefetch_hint_l2(_ptr: *const u8) {
}

/// Submit a prefetch-into-L3 task for the byte range covering
/// `slice` to the IoPool. Returns immediately; the prefetch
/// happens asynchronously on an SMT-sibling thread.
///
/// Use this for "I'll consume `slice` once soon" patterns - NTT
/// streaming, Karatsuba scratch buffers, BigFloat scratch.
///
/// No-op when the IoPool is disabled (`FLYNNEL_SCHED_SMT_AS_IO`
/// not set). Prefetch is best-effort; missing it is a perf miss
/// only.
///
/// # Safety
///
/// The pointer / length captured by the closure must remain valid
/// for the duration of the prefetch task. Since prefetch is a
/// no-side-effect hint, the worst case if the pointer is dangling
/// is a bogus prefetch that the CPU may or may not honor - no
/// memory safety violation. We document this with the API but
/// don't require unsafe.
pub fn prefetch_into_l3<T>(slice: &[T]) {
    let Some(pool) = global_io_pool() else { return };
    let addr = slice.as_ptr() as usize;
    let bytes = std::mem::size_of_val(slice);
    pool.submit(move || {
        let base = addr as *const u8;
        // SAFETY: caller-API doc requires the slice to remain
        // valid for the duration of the prefetch. Even if it
        // doesn't, prefetch hints are no-side-effect.
        unsafe {
            let mut off = 0usize;
            while off < bytes {
                prefetch_hint_l3(base.add(off));
                off = off.saturating_add(PREFETCH_LINE_BYTES);
            }
        }
    });
}

/// Submit a prefetch-into-L2 task for the byte range covering
/// `slice` to the IoPool. Use for data the consumer will reuse
/// multiple times in the near future - typically inner-loop
/// constants like Karatsuba twiddle factors.
///
/// No-op when the IoPool is disabled.
pub fn prefetch_into_l2<T>(slice: &[T]) {
    let Some(pool) = global_io_pool() else { return };
    let addr = slice.as_ptr() as usize;
    let bytes = std::mem::size_of_val(slice);
    pool.submit(move || {
        let base = addr as *const u8;
        // SAFETY: same reasoning as the `prefetch_into_l3`
        // closure above. The pointer / length pair was captured
        // from a live slice; prefetch hints are no-side-effect
        // even if the caller invalidates the slice before this
        // task runs.
        unsafe {
            let mut off = 0usize;
            while off < bytes {
                prefetch_hint_l2(base.add(off));
                off = off.saturating_add(PREFETCH_LINE_BYTES);
            }
        }
    });
}

/// Issue prefetch hints synchronously on the calling thread, no
/// async dispatch. Use when you're already running on the compute
/// thread and want to warm cache for an upcoming hot loop on the
/// same thread.
///
/// Unlike [`prefetch_into_l3`] this runs immediately and does NOT
/// depend on the IoPool. Use for in-loop prefetch where the
/// channel-submit overhead would dominate the prefetch cost.
#[inline]
pub fn prefetch_into_l3_inline<T>(slice: &[T]) {
    let base = slice.as_ptr() as *const u8;
    let bytes = std::mem::size_of_val(slice);
    // SAFETY: `base` and `bytes` describe the live `slice`
    // passed in, so every `base.add(off)` for `off < bytes`
    // points inside the slice. `prefetch_hint_l3` is itself a
    // no-side-effect hint intrinsic.
    unsafe {
        let mut off = 0usize;
        while off < bytes {
            prefetch_hint_l3(base.add(off));
            off = off.saturating_add(PREFETCH_LINE_BYTES);
        }
    }
}

/// Inline form of [`prefetch_into_l2`].
#[inline]
pub fn prefetch_into_l2_inline<T>(slice: &[T]) {
    let base = slice.as_ptr() as *const u8;
    let bytes = std::mem::size_of_val(slice);
    // SAFETY: same reasoning as `prefetch_into_l3_inline`.
    unsafe {
        let mut off = 0usize;
        while off < bytes {
            prefetch_hint_l2(base.add(off));
            off = off.saturating_add(PREFETCH_LINE_BYTES);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefetch_inline_does_not_crash_on_small_slice() {
        let v: Vec<u32> = (0..16).collect();
        prefetch_into_l3_inline(&v);
        prefetch_into_l2_inline(&v);
    }

    #[test]
    fn prefetch_inline_does_not_crash_on_large_slice() {
        // 1 MB - exceeds L2 on most CPUs, exercises the inner loop.
        let v: Vec<u32> = (0..256 * 1024).collect();
        prefetch_into_l3_inline(&v);
    }

    #[test]
    fn prefetch_inline_does_not_crash_on_empty_slice() {
        let v: Vec<u32> = Vec::new();
        prefetch_into_l3_inline(&v);
        prefetch_into_l2_inline(&v);
    }

    #[test]
    fn prefetch_async_noop_when_pool_disabled() {
        // FLYNNEL_SCHED_SMT_AS_IO not set in test env - returns
        // immediately without submitting anything.
        let v: Vec<u32> = (0..1024).collect();
        let t0 = std::time::Instant::now();
        prefetch_into_l3(&v);
        let elapsed = t0.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(10),
            "async prefetch should be ~immediate when pool disabled"
        );
    }
}

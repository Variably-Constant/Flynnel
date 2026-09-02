//! Const-generic K-recursion driver: `k_join<const K>(a, b)`.
//!
//! The bridge from K-recursive algorithm code (Karatsuba, NTT,
//! Burnikel-Ziegler, transcendental Newton) into the scheduler.
//! The const-generic parameter `K = log2(n_limbs)` drives compile-
//! time dispatch:
//!
//! - At `K <= 4` (microsizes, sub-microsecond per-op work) the
//!   function monomorphises to literally `(a(), b())` - no
//!   scheduler call, no JobPlan allocation, zero overhead.
//! - At `K >= 5` it builds a [`JobPlan`] and delegates to
//!   [`crate::sched::join`], which dispatches by tier.
//!
//! Consumed by K-recursive Karatsuba, K-recursive NTT, and
//! K-recursive Burnikel-Ziegler division drivers in downstream
//! crates.

use crate::sched::plan::JobPlan;

/// K-recursion fork-join. `K` is the **compile-time** recursion
/// level (= log2 of the operand limb count); at `K <= 4` the call
/// folds to inline serial execution with no scheduler involvement.
///
/// # Why const-generic K
///
/// Rust's const-generic dispatch resolves `K <= 4` at compile
/// time, so the `if` branch is a free zero-cost classifier. A
/// runtime `if k <= 4` would still pay the branch + the
/// `JobPlan::new` ctor unconditionally; the const form pays
/// neither at small K.
pub fn k_join<const K: u32, A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    if K <= 4 {
        let ra = a();
        let rb = b();
        (ra, rb)
    } else {
        let plan = JobPlan::new(K as u8, 1);
        crate::sched::arena::join(&plan, a, b)
    }
}

/// Variant of [`k_join`] that takes an explicit [`JobPlan`].
/// Useful when the caller needs to pin `hw_class` / `variant` /
/// `numa_hint` (the default plan picks `Scalar` / `Faithful` /
/// no hint).
pub fn k_join_with_plan<const K: u32, A, B, RA, RB>(
    plan: &JobPlan,
    a: A,
    b: B,
) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    if K <= 4 {
        let ra = a();
        let rb = b();
        (ra, rb)
    } else {
        crate::sched::arena::join(plan, a, b)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HwClass;
    use crate::Variant;

    #[test]
    fn k_join_at_k2_runs_both_halves() {
        // K = 2 (Inline band, no scheduler).
        let (ra, rb) = k_join::<2, _, _, _, _>(|| 10u32, || 20u32);
        assert_eq!((ra, rb), (10, 20));
    }

    #[test]
    fn k_join_at_k3_runs_both_halves() {
        let (ra, rb) = k_join::<3, _, _, _, _>(|| 1u64, || 2u64);
        assert_eq!((ra, rb), (1, 2));
    }

    #[test]
    fn k_join_at_k6_runs_both_halves_local_band() {
        // K = 6 (Local band) - routes through sched::join which
        // currently dispatches inline.
        let (ra, rb) = k_join::<6, _, _, _, _>(|| 10i32, || -20i32);
        assert_eq!((ra, rb), (10, -20));
    }

    #[test]
    fn k_join_at_k9_runs_both_halves_hierarchical_band() {
        let (ra, rb) = k_join::<9, _, _, _, _>(
            || "left".to_string(),
            || "right".to_string(),
        );
        assert_eq!(ra, "left");
        assert_eq!(rb, "right");
    }

    #[test]
    fn k_join_at_k13_runs_both_halves_federated_band() {
        let (ra, rb) = k_join::<13, _, _, _, _>(|| 1.5_f64, || 2.5_f64);
        assert_eq!((ra, rb), (1.5, 2.5));
    }

    #[test]
    fn k_join_preserves_left_first_serial_order() {
        // K <= 4 path runs a then b. At K=2, observable side
        // effects must show this ordering.
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};
        let counter = Arc::new(AtomicU32::new(0));
        let ca = Arc::clone(&counter);
        let cb = Arc::clone(&counter);
        let (ra, rb) = k_join::<2, _, _, _, _>(
            move || ca.fetch_add(1, Ordering::SeqCst),
            move || cb.fetch_add(1, Ordering::SeqCst),
        );
        // a saw 0, b saw 1.
        assert_eq!((ra, rb), (0, 1));
    }

    #[test]
    fn k_join_with_plan_at_k8_uses_custom_hw_class() {
        // Verify the plan-taking variant accepts a customised plan
        // without dropping any fields.
        let plan = JobPlan::new(8, 1024)
            .with_hw_class(HwClass::Avx2)
            .with_variant(Variant::Correct);
        let (ra, rb) = k_join_with_plan::<8, _, _, _, _>(&plan, || 0u32, || 1u32);
        assert_eq!((ra, rb), (0, 1));
    }

    #[test]
    fn k_join_with_plan_at_k2_skips_scheduler_even_with_plan() {
        // At K <= 4 the plan is ignored and the call folds inline.
        // The custom hw_class on the plan is irrelevant; we just
        // check the call returns.
        let plan = JobPlan::new(2, 1).with_hw_class(HwClass::Avx512f);
        let (ra, rb) = k_join_with_plan::<2, _, _, _, _>(&plan, || 7u32, || 11u32);
        assert_eq!((ra, rb), (7, 11));
    }

    #[test]
    fn k_join_propagates_panic_from_left_half() {
        let r = std::panic::catch_unwind(|| {
            k_join::<2, _, _, u32, u32>(|| panic!("left panic"), || 0u32)
        });
        assert!(r.is_err(), "left-half panic must propagate");
    }

    #[test]
    fn k_join_threaded_recursive_simulated_karatsuba_shape() {
        // Simulate a 4-way Karatsuba-style fork: top-level k_join
        // splits two halves; each half k_joins again. Verify all
        // four leaves run and their results combine correctly.
        let ((u11, u12), (u21, u22)) = k_join::<8, _, _, _, _>(
            || k_join::<7, _, _, _, _>(|| 1u32, || 2u32),
            || k_join::<7, _, _, _, _>(|| 4u32, || 8u32),
        );
        assert_eq!(u11, 1);
        assert_eq!(u12, 2);
        assert_eq!(u21, 4);
        assert_eq!(u22, 8);
        // Combined as a Karatsuba-style XOR-checksum.
        assert_eq!(u11 ^ u12 ^ u21 ^ u22, 1 ^ 2 ^ 4 ^ 8);
    }
}

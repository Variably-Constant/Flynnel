//! Plan-free rayon-shaped surface.
//!
//! Provides the ergonomics of rayon's `join` and `par_iter_mut`
//! without requiring callers to construct a [`JobPlan`]. Each
//! function here delegates to the same underlying scheduler the
//! [`crate::sched::join`] / [`crate::sched::par_iter::for_each_chunk`]
//! paths use, with a default `JobPlan::new(0, n)` so consumers who
//! do not care about per-call tuning (hw_class / variant /
//! numa_hint / SMT activation / oversubscription) can call into
//! flynnel with the same shape rayon uses.
//!
//! When per-call tuning IS needed, use the typed
//! [`crate::sched::JobPlan`] surface directly (it is the same
//! scheduler underneath; the flat surface is purely an ergonomic
//! wrapper).
//!
//! # Examples
//!
//! Rayon-style `join`:
//!
//! ```no_run
//! let (left, right) = flynnel::flat::join(
//!     || (0..1000).sum::<u32>(),
//!     || (1000..2000).sum::<u32>(),
//! );
//! assert_eq!(left + right, (0..2000).sum::<u32>());
//! ```
//!
//! Rayon-style `par_iter_mut().for_each(...)`:
//!
//! ```no_run
//! let mut data: Vec<u32> = (0..1_000_000).collect();
//! flynnel::flat::par_for_each_mut(&mut data, |x| *x = x.wrapping_mul(3));
//! ```
//!
//! Slice-chunk shape (closure receives a `&mut [T]` slice instead
//! of one element at a time):
//!
//! ```no_run
//! let mut data: Vec<f64> = (1..=1_000_000).map(|i| i as f64).collect();
//! flynnel::flat::par_for_each_chunk_mut(&mut data, |slice| {
//!     for x in slice {
//!         *x = x.sqrt();
//!     }
//! });
//! ```

use crate::sched::JobPlan;

/// Plan-free `join`: runs `a` and `b` potentially in parallel and
/// returns their results. Same semantics as `rayon::join`.
///
/// Internally constructs a default [`JobPlan`] sized to a
/// 2-element batch. Consumers that need to set `hw_class`,
/// `variant`, `numa_hint`, `use_smt`, or any of the per-call
/// tuning knobs should construct a [`JobPlan`] and call
/// [`crate::sched::join`] directly.
pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    let plan = JobPlan::new(0, 2);
    crate::sched::join(&plan, a, b)
}

/// Plan-free per-element parallel for-each on a mutable slice.
/// Equivalent in shape to `slice.par_iter_mut().for_each(op)` in
/// rayon. The closure runs once per element, receiving `&mut T`.
///
/// Internally constructs a default [`JobPlan`] sized to the input
/// length and dispatches through [`crate::sched::par_iter::for_each_chunk`],
/// which routes work via the scheduler's hybrid-JEC wake protocol.
///
/// The slice-chunk variant [`par_for_each_chunk_mut`] is faster
/// when the per-element work is tight and you want SIMD-friendly
/// loop bodies; use this per-element form for closures whose
/// natural shape is a single-element operation.
pub fn par_for_each_mut<T, F>(items: &mut [T], op: F)
where
    T: Send,
    F: Fn(&mut T) + Sync,
{
    let plan = JobPlan::new(0, items.len() as u32);
    crate::sched::par_iter::for_each_chunk(&plan, items, move |slice: &mut [T]| {
        for x in slice.iter_mut() {
            op(x);
        }
    });
}

/// Plan-free slice-chunk parallel for-each on a mutable slice.
/// The closure receives a contiguous `&mut [T]` chunk per leaf;
/// this is the natural shape for SIMD-friendly inner loops and is
/// what most flynnel users want.
///
/// Equivalent to the `&mut [T]`-closure shape of
/// [`crate::sched::par_iter::for_each_chunk`], with a default
/// [`JobPlan`] constructed for you.
pub fn par_for_each_chunk_mut<T, F>(items: &mut [T], op: F)
where
    T: Send,
    F: Fn(&mut [T]) + Sync,
{
    let plan = JobPlan::new(0, items.len() as u32);
    crate::sched::par_iter::for_each_chunk(&plan, items, op);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_join_returns_both_results() {
        let (a, b) = join(|| 42u32, || 7u32);
        assert_eq!(a, 42);
        assert_eq!(b, 7);
    }

    #[test]
    fn flat_par_for_each_mut_touches_every_element() {
        let n = 10_000usize;
        let mut v: Vec<u32> = (0..n as u32).collect();
        par_for_each_mut(&mut v, |x| *x = x.wrapping_mul(3).wrapping_add(7));
        for (i, &val) in v.iter().enumerate() {
            assert_eq!(val, (i as u32).wrapping_mul(3).wrapping_add(7));
        }
    }

    #[test]
    fn flat_par_for_each_chunk_mut_processes_full_slice() {
        let n = 10_000usize;
        let mut v: Vec<u64> = (0..n as u64).collect();
        par_for_each_chunk_mut(&mut v, |slice| {
            for x in slice.iter_mut() {
                *x = x.wrapping_mul(5);
            }
        });
        for (i, &val) in v.iter().enumerate() {
            assert_eq!(val, (i as u64).wrapping_mul(5));
        }
    }

    #[test]
    fn flat_par_for_each_mut_empty_slice_is_noop() {
        let mut v: Vec<u32> = Vec::new();
        par_for_each_mut(&mut v, |x| *x += 1);
        assert!(v.is_empty());
    }
}

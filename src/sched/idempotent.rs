//! Idempotent job contract: a marker trait
//! whose impl asserts the job body is safe to execute more than
//! once. Pairs with fence-free work-stealing variants where the
//! cross-worker race can lose a steal-vs-pop conflict by
//! executing the same job twice rather than blocking.
//!
//! The [`crate::sched::arena_local`] worker loop enforces
//! exactly-once execution via Chase-Lev's atomic counters; this
//! trait is the opt-in contract for fence-free variants that trade
//! brief multiplicity windows for one fewer Acquire fence per
//! local pop.
//!
//! Most op bodies are NOT idempotent; they mutate a
//! Vec output, increment a counter, or read+write a shared
//! accumulator. The few that ARE idempotent (pure indexed-collect
//! over a `MaybeUninit` buffer where each slot is written once;
//! pure read-only fold) can opt in via this trait.
//!
//! # Examples
//!
//! ```
//! use flynnel::sched::idempotent::IdempotentJob;
//!
//! // Pure-read filter is idempotent: running the closure on the
//! // same range produces the same result with no side effects.
//! struct PureFilter<'a>(&'a [u32]);
//! impl<'a> IdempotentJob for PureFilter<'a> {
//!     type Output = u64;
//!     fn run(&self, start: usize, end: usize) -> u64 {
//!         self.0[start..end].iter().map(|&x| x as u64).sum()
//!     }
//! }
//! ```

/// Marker for a job whose body is safe to execute more than once
/// without producing incorrect output or violating invariants.
///
/// A job is **idempotent** when:
/// - The body is a pure function of the input range (no side
///   effects that depend on call count).
/// - Writes go through `MaybeUninit` slots that each closure call
///   would write to the same value, OR the writes never observe
///   their own prior state.
///
/// Examples:
/// - Pure `Fn(usize) -> R` indexed-collect - idempotent IF the
///   closure has no captured mutable state.
/// - Read-only fold like `sum`, `max`, `count` - idempotent.
/// - Per-tile matmul that writes into a fresh accumulator buffer -
///   idempotent IF the buffer is zeroed before each call attempt.
///
/// Counter-examples (NOT idempotent):
/// - Anything that `fetch_add`s an atomic counter - running twice
///   double-counts.
/// - Pushing onto a shared `Vec` - running twice double-pushes.
/// - Newton iteration where each step reads its previous output -
///   running twice perturbs convergence.
pub trait IdempotentJob: Send + Sync {
    /// The output type produced by one execution.
    type Output: Send;

    /// Execute the job body over the given index range. May be
    /// called multiple times with the same range; implementations
    /// MUST return the same output for the same input each time
    /// AND MUST NOT depend on previous calls' side effects.
    fn run(&self, start: usize, end: usize) -> Self::Output;
}

/// Dispatch helper for idempotent jobs: runs the body, returning
/// its output. This is the entry point that an idempotent-aware
/// worker loop would call. Provided in the trait surface so the
/// production wiring path (when added) has a stable API target.
#[inline]
pub fn run_idempotent<J: IdempotentJob>(job: &J, start: usize, end: usize) -> J::Output {
    job.run(start, end)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct Sum<'a>(&'a [u32]);
    impl<'a> IdempotentJob for Sum<'a> {
        type Output = u64;
        fn run(&self, start: usize, end: usize) -> u64 {
            self.0[start..end].iter().map(|&x| x as u64).sum()
        }
    }

    #[test]
    fn idempotent_run_returns_same_value_across_calls() {
        let data: Vec<u32> = (0..1000).collect();
        let job = Sum(&data);
        let a = run_idempotent(&job, 0, 1000);
        let b = run_idempotent(&job, 0, 1000);
        let c = run_idempotent(&job, 0, 1000);
        assert_eq!(a, b);
        assert_eq!(b, c);
        let expected: u64 = (0..1000u64).sum();
        assert_eq!(a, expected);
    }

    #[test]
    fn idempotent_run_partial_range() {
        let data: Vec<u32> = (0..100).collect();
        let job = Sum(&data);
        let total: u64 = run_idempotent(&job, 0, 100);
        let lo: u64 = run_idempotent(&job, 0, 50);
        let hi: u64 = run_idempotent(&job, 50, 100);
        assert_eq!(lo + hi, total);
    }
}

//! The fork-join entries and their determinism contracts.
//!
//! The documented promise of every one of these is that the caller's
//! ordering survives the scheduling: `join` hands back the left half
//! first whichever thread ran it, and the cooperative entries return
//! results in the order the closures were supplied rather than the
//! order they finished. A scheduler that got this wrong would produce
//! a different answer for a non-commutative reduction depending on
//! machine load, which is the kind of wrong that does not reproduce.

use std::sync::atomic::{AtomicUsize, Ordering};

use flynnel::sched::cooperative::{
    cooperative_join_n_flat, cooperative_join_n_flat_mailbox, cooperative_join_n_tree,
};
use flynnel::{JobPlan, cooperative_join_n, join, join_context, join_default, k_join};
use flynnel::sched::k_join::k_join_with_plan;

/// Takes the inline tier: a micro K band with a batch small enough
/// that dispatch cannot pay. A batch of 32 or under classifies as
/// latency-bound instead, which disables the inline fallback, so the
/// batch here is deliberately above that.
fn tiny() -> JobPlan {
    JobPlan::new(2, 100)
}

/// Large enough to reach the pool.
fn wide() -> JobPlan {
    JobPlan::new(8, 1_000_000)
}

#[test]
fn join_hands_back_the_left_half_first_on_both_tiers() {
    // String concatenation does not commute, so a swapped pair is
    // visible in the value rather than only in the types.
    for p in [tiny(), wide()] {
        let (a, b) = join(&p, || "left".to_string(), || "right".to_string());
        assert_eq!(a, "left", "the first closure's result comes back first");
        assert_eq!(b, "right");
        assert_eq!(format!("{a}{b}"), "leftright");
    }
}

#[test]
fn join_runs_both_halves_exactly_once() {
    for p in [tiny(), wide()] {
        let left = AtomicUsize::new(0);
        let right = AtomicUsize::new(0);
        let (ra, rb) = join(
            &p,
            || {
                left.fetch_add(1, Ordering::Relaxed);
                1u32
            },
            || {
                right.fetch_add(1, Ordering::Relaxed);
                2u32
            },
        );
        assert_eq!((ra, rb), (1, 2));
        assert_eq!(left.load(Ordering::Relaxed), 1, "the left half ran once");
        assert_eq!(right.load(Ordering::Relaxed), 1, "and so did the right");
    }
}

#[test]
fn join_propagates_a_panic_from_either_half() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    for p in [tiny(), wide()] {
        let left_panicked = std::panic::catch_unwind(|| {
            join(&p, || panic!("left half"), || 1u32);
        });
        assert!(left_panicked.is_err(), "a panic in the left half reaches the caller");

        let right_panicked = std::panic::catch_unwind(|| {
            join(&p, || 1u32, || panic!("right half"));
        });
        assert!(right_panicked.is_err(), "and so does one in the right half");
    }

    std::panic::set_hook(previous);
}

#[test]
fn the_inline_tier_reports_neither_injection_nor_a_steal() {
    // The inline tier runs both halves in the caller, so by definition
    // nothing was injected and nothing was stolen.
    let (injected, stolen) = join_context(&tiny(), |injected| injected, |stolen| stolen);
    assert!(!injected, "an inline join was not cold-injected");
    assert!(!stolen, "and its right half was not taken by a peer");
}

#[test]
fn join_context_flags_do_not_change_the_answer() {
    // Whatever the flags read, the results and their order are fixed.
    for p in [tiny(), wide()] {
        let (a, b) = join_context(&p, |_| "left", |_| "right");
        assert_eq!((a, b), ("left", "right"));
    }
}

#[test]
fn join_default_builds_a_working_plan() {
    let (a, b) = join_default(6, 1024, || 10u32, || 32u32);
    assert_eq!((a, b), (10, 32));
    // And agrees with the explicit-plan form it is a convenience for.
    let (c, d) = join(&JobPlan::new(6, 1024), || 10u32, || 32u32);
    assert_eq!((a, b), (c, d));
}

#[test]
fn a_small_k_join_folds_to_serial_execution_on_the_calling_thread() {
    // K <= 4 is documented as folding to inline serial with no
    // scheduler involvement, which is observable in the thread ids.
    let calling = std::thread::current().id();
    let (a, b) = k_join::<2, _, _, _, _>(
        || std::thread::current().id(),
        || std::thread::current().id(),
    );
    assert_eq!(a, calling, "the left half stayed on the calling thread");
    assert_eq!(b, calling, "and so did the right");
}

#[test]
fn a_large_k_join_still_returns_both_halves_in_order() {
    let (a, b) = k_join::<8, _, _, _, _>(|| "left".to_string(), || "right".to_string());
    assert_eq!(a, "left");
    assert_eq!(b, "right");
}

#[test]
fn k_join_with_plan_agrees_with_k_join_on_results() {
    let plan = JobPlan::new(8, 4096);
    let (a, b) = k_join_with_plan::<8, _, _, _, _>(&plan, || 3u32, || 4u32);
    assert_eq!((a, b), (3, 4));
    let (c, d) = k_join::<8, _, _, _, _>(|| 3u32, || 4u32);
    assert_eq!((a, b), (c, d));
    // And the small-K fold ignores the plan the same way.
    let (e, f) = k_join_with_plan::<2, _, _, _, _>(&plan, || 3u32, || 4u32);
    assert_eq!((e, f), (3, 4));
}

fn boxed(n: usize) -> Vec<Box<dyn FnOnce() -> usize + Send>> {
    (0..n).map(|i| Box::new(move || i * 7 + 1) as Box<dyn FnOnce() -> usize + Send>).collect()
}

fn expected(n: usize) -> Vec<usize> {
    (0..n).map(|i| i * 7 + 1).collect()
}

#[test]
fn cooperative_join_n_returns_caller_order_at_every_size() {
    // The sizes straddle every internal fast path: empty, one, two,
    // the three that turns on the tree, and a dozen.
    for n in [0usize, 1, 2, 3, 5, 12] {
        let plan = JobPlan::new(8, n.max(1) as u32);
        let got = cooperative_join_n(&plan, boxed(n));
        assert_eq!(got, expected(n), "caller-supplied order at n = {n}");
    }
}

#[test]
fn every_cooperative_variant_agrees_with_the_others() {
    // Three shapes with the same documented ordering contract: a
    // divergence between them is a silent reordering for whichever
    // caller happens to route to the odd one.
    for n in [0usize, 1, 2, 3, 5, 12] {
        let plan = JobPlan::new(8, n.max(1) as u32);
        let want = expected(n);
        assert_eq!(cooperative_join_n_tree(&plan, boxed(n)), want, "tree at n = {n}");
        assert_eq!(cooperative_join_n_flat(&plan, boxed(n)), want, "flat at n = {n}");
        assert_eq!(
            cooperative_join_n_flat_mailbox(&plan, boxed(n)),
            want,
            "mailbox at n = {n}"
        );
    }
}

#[test]
fn cooperative_join_n_runs_every_closure_exactly_once() {
    let n = 12usize;
    let plan = JobPlan::new(8, n as u32);
    let counter = std::sync::Arc::new(AtomicUsize::new(0));
    let closures: Vec<Box<dyn FnOnce() -> usize + Send>> = (0..n)
        .map(|i| {
            let c = std::sync::Arc::clone(&counter);
            Box::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
                i
            }) as Box<dyn FnOnce() -> usize + Send>
        })
        .collect();
    let got = cooperative_join_n(&plan, closures);
    assert_eq!(got, (0..n).collect::<Vec<_>>());
    assert_eq!(counter.load(Ordering::Relaxed), n, "each closure ran once, not zero or twice");
}

#[test]
fn a_cooperative_result_order_is_stable_across_repeats() {
    // Execution order varies with load; the returned order must not.
    let n = 12usize;
    let plan = JobPlan::new(8, n as u32);
    let first = cooperative_join_n(&plan, boxed(n));
    for _ in 0..16 {
        assert_eq!(cooperative_join_n(&plan, boxed(n)), first, "order is not load-dependent");
    }
}

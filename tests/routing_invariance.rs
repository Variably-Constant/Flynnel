//! A walker's answer does not depend on which route it took.
//!
//! Every data-parallel entry can run its body on the calling thread or
//! hand it to the pool, decided per call from the plan's estimate
//! against this host's measured collapse threshold. That decision is
//! about cost and must never be about results. The indexed entries are
//! the ones with something to get wrong: a collapsed body sees the
//! whole range at once where a dispatched one sees pieces with
//! offsets, so an absolute index computed correctly on one path can be
//! wrong on the other and nothing would say so.
//!
//! Each test runs one entry twice over identical input, once with an
//! estimate that forces the collapse and once with an estimate that
//! forces dispatch, and compares the two answers.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use flynnel::sched::par_iter::{
    collect_indexed, for_each_chunk_indexed_min_leaf, for_each_chunk_triple_min_leaf, reduce_chunks,
};
use flynnel::{JobPlan, for_each_chunk, for_each_chunk_ref, for_each_indexed};

/// A per-item cost small enough that any size in this file lands under
/// the measured collapse threshold.
const COLLAPSES: u32 = 1;
/// A per-item cost large enough that no size in this file does.
const DISPATCHES: u32 = 10_000_000;

fn plan(n: usize, per_item_ns: u32) -> JobPlan {
    JobPlan::new(6, n as u32).with_estimated_per_item_ns(per_item_ns)
}

/// Both routes are actually taken, so the tests below are not
/// comparing one route against itself.
#[test]
fn the_two_plans_really_do_route_differently() {
    let n = 64usize;
    let calling = std::thread::current().id();

    let inline_off = AtomicUsize::new(0);
    let mut v: Vec<u32> = (0..n as u32).collect();
    for_each_chunk_indexed_min_leaf(&plan(n, COLLAPSES), &mut v, 1, |_, _| {
        if std::thread::current().id() != calling {
            inline_off.fetch_add(1, Ordering::Relaxed);
        }
    });
    assert_eq!(
        inline_off.load(Ordering::Relaxed),
        0,
        "the collapsing plan must stay on the calling thread"
    );

    let pool_off = AtomicUsize::new(0);
    let mut v2: Vec<u32> = (0..n as u32).collect();
    for_each_chunk_indexed_min_leaf(&plan(n, DISPATCHES), &mut v2, 1, |_, _| {
        if std::thread::current().id() != calling {
            pool_off.fetch_add(1, Ordering::Relaxed);
        }
    });
    assert!(
        pool_off.load(Ordering::Relaxed) > 0,
        "the dispatching plan must reach the pool, or these tests compare nothing"
    );
}

fn squares(n: usize, per_item_ns: u32) -> Vec<u64> {
    let mut v: Vec<u64> = (0..n as u64).collect();
    for_each_chunk(&plan(n, per_item_ns), &mut v, |c| {
        for x in c {
            *x = *x * *x + 1;
        }
    });
    v
}

#[test]
fn for_each_chunk_answers_the_same_either_way() {
    let n = 200usize;
    assert_eq!(squares(n, COLLAPSES), squares(n, DISPATCHES));
}

fn indexed_min_leaf(n: usize, per_item_ns: u32) -> Vec<u64> {
    let mut v = vec![0u64; n];
    for_each_chunk_indexed_min_leaf(&plan(n, per_item_ns), &mut v, 1, |start, chunk| {
        for (i, slot) in chunk.iter_mut().enumerate() {
            // The absolute index is the whole point: a collapsed body
            // gets start = 0 and the entire slice, a dispatched one
            // gets a piece and its offset.
            *slot = (start + i) as u64 * 3 + 7;
        }
    });
    v
}

#[test]
fn the_indexed_walker_computes_the_same_absolute_indices_either_way() {
    let n = 200usize;
    let collapsed = indexed_min_leaf(n, COLLAPSES);
    assert_eq!(collapsed, indexed_min_leaf(n, DISPATCHES));
    // And against the closed form, so both routes being wrong the same
    // way would still fail.
    let expect: Vec<u64> = (0..n as u64).map(|i| i * 3 + 7).collect();
    assert_eq!(collapsed, expect);
}

fn triple(n: usize, per_item_ns: u32) -> Vec<u64> {
    let a: Vec<u64> = (0..n as u64).collect();
    let b: Vec<u64> = (0..n as u64).map(|i| i * 2).collect();
    let mut out = vec![0u64; n];
    for_each_chunk_triple_min_leaf(&plan(n, per_item_ns), &mut out, &a, &b, 1, |o, x, y| {
        for ((o, x), y) in o.iter_mut().zip(x).zip(y) {
            *o = *x + *y;
        }
    });
    out
}

#[test]
fn the_triple_walker_stays_in_lockstep_either_way() {
    let n = 200usize;
    let collapsed = triple(n, COLLAPSES);
    assert_eq!(collapsed, triple(n, DISPATCHES));
    let expect: Vec<u64> = (0..n as u64).map(|i| i * 3).collect();
    assert_eq!(collapsed, expect, "out[i] = a[i] + b[i] with the slices aligned");
}

fn chunk_ref_sums(n: usize, width: usize, per_item_ns: u32) -> Vec<(usize, u64)> {
    let items: Vec<u64> = (0..n as u64).collect();
    let seen = Mutex::new(Vec::new());
    for_each_chunk_ref(&plan(n, per_item_ns), &items, width, |start, chunk| {
        let sum: u64 = chunk.iter().sum();
        seen.lock().unwrap_or_else(|e| e.into_inner()).push((start, sum));
    });
    let mut v = seen.into_inner().unwrap_or_else(|e| e.into_inner());
    v.sort_unstable();
    v
}

#[test]
fn the_ref_walker_tiles_the_slice_the_same_either_way() {
    let (n, width) = (200usize, 16usize);
    let collapsed = chunk_ref_sums(n, width, COLLAPSES);
    assert_eq!(collapsed, chunk_ref_sums(n, width, DISPATCHES));
    assert_eq!(collapsed.len(), n.div_ceil(width), "the chunks tile the slice exactly once");
    let total: u64 = collapsed.iter().map(|&(_, s)| s).sum();
    assert_eq!(total, (0..n as u64).sum::<u64>(), "and cover every element once");
}

fn indexed_marks(n: usize, per_item_ns: u32) -> Vec<u64> {
    let marks: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    for_each_indexed(&plan(n, per_item_ns), n, 1, |i| {
        marks[i].store(i as u64 + 1, Ordering::Relaxed);
    });
    marks.iter().map(|m| m.load(Ordering::Relaxed)).collect()
}

#[test]
fn for_each_indexed_visits_every_index_once_either_way() {
    let n = 200usize;
    let collapsed = indexed_marks(n, COLLAPSES);
    assert_eq!(collapsed, indexed_marks(n, DISPATCHES));
    let expect: Vec<u64> = (1..=n as u64).collect();
    assert_eq!(collapsed, expect, "every index was visited exactly once");
}

#[test]
fn collect_indexed_builds_the_same_vector_either_way() {
    let n = 200usize;
    let collapsed = collect_indexed(&plan(n, COLLAPSES), n, 1, |i| i as u64 * 5 + 2);
    let dispatched = collect_indexed(&plan(n, DISPATCHES), n, 1, |i| i as u64 * 5 + 2);
    assert_eq!(collapsed, dispatched);
    let expect: Vec<u64> = (0..n as u64).map(|i| i * 5 + 2).collect();
    assert_eq!(collapsed, expect, "and in index order, not completion order");
}

#[test]
fn reduce_chunks_folds_to_the_same_value_either_way() {
    let n = 2000usize;
    let items: Vec<u64> = (0..n as u64).collect();
    let run = |per_item_ns: u32| {
        reduce_chunks(
            &plan(n, per_item_ns),
            &items,
            || 0u64,
            |acc, chunk| acc + chunk.iter().sum::<u64>(),
            |a, b| a + b,
        )
    };
    let collapsed = run(COLLAPSES);
    assert_eq!(collapsed, run(DISPATCHES));
    assert_eq!(collapsed, (0..n as u64).sum::<u64>(), "and equals the serial fold");
}

#[test]
fn a_non_associative_reduce_is_bit_exact_across_routes() {
    // Float addition is not associative, so a reduction that regrouped
    // its operands would answer differently. The documented contract is
    // that the algebraic order holds whichever thread ran which half.
    let n = 4096usize;
    let items: Vec<f64> = (0..n).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let run = |per_item_ns: u32| {
        reduce_chunks(
            &plan(n, per_item_ns),
            &items,
            || 0.0f64,
            |acc, chunk| chunk.iter().fold(acc, |a, &x| a + x),
            |a, b| a + b,
        )
    };
    let collapsed = run(COLLAPSES);
    let dispatched = run(DISPATCHES);
    assert_eq!(
        collapsed.to_bits(),
        dispatched.to_bits(),
        "the two routes must agree bit for bit, not merely closely: {collapsed:?} against {dispatched:?}"
    );
}

#[test]
fn a_worker_cap_of_one_answers_the_same_as_the_pool() {
    // A third route: the cap is read from the plan alone, without
    // consulting the host profile at all, so it reaches the calling
    // thread by a different path than the collapse does.
    let n = 200usize;
    let capped = {
        let mut v = vec![0u64; n];
        let p = JobPlan::new(6, n as u32)
            .with_estimated_per_item_ns(DISPATCHES)
            .with_workers(1);
        for_each_chunk_indexed_min_leaf(&p, &mut v, 1, |start, chunk| {
            for (i, slot) in chunk.iter_mut().enumerate() {
                *slot = (start + i) as u64 * 3 + 7;
            }
        });
        v
    };
    assert_eq!(capped, indexed_min_leaf(n, DISPATCHES));
    let expect: Vec<u64> = (0..n as u64).map(|i| i * 3 + 7).collect();
    assert_eq!(capped, expect);
}

#[test]
fn a_site_that_latches_after_an_overrun_keeps_answering_correctly() {
    // A tiny estimate admits the collapse; a body far slower than that
    // estimate makes the site stop trusting itself, so a later call
    // from the same source location routes to the pool instead. The
    // routing changes underneath the caller and the answer must not.
    let n = 64usize;
    let calling = std::thread::current().id();
    let expect: Vec<u64> = (0..n as u64).collect();

    let run = || {
        let off = AtomicUsize::new(0);
        let mut v = vec![0u64; n];
        for_each_chunk_indexed_min_leaf(&plan(n, COLLAPSES), &mut v, 1, |start, chunk| {
            if std::thread::current().id() != calling {
                off.fetch_add(1, Ordering::Relaxed);
            }
            for (i, slot) in chunk.iter_mut().enumerate() {
                // Far past any measured collapse threshold, so the
                // first collapsed body overruns and latches the site.
                std::thread::sleep(std::time::Duration::from_micros(200));
                *slot = (start + i) as u64;
            }
        });
        (v, off.load(Ordering::Relaxed))
    };

    let (first, first_off) = run();
    assert_eq!(first, expect, "the collapsed call is correct");
    assert_eq!(first_off, 0, "and it did collapse");

    let (second, second_off) = run();
    assert_eq!(second, expect, "and so is the call after the site latched");
    assert!(
        second_off > 0,
        "the site should have stopped collapsing after the overrun, so this call reaches the pool"
    );
}

#[test]
fn a_panic_in_a_pool_leaf_reaches_the_caller() {
    // A leaf that panics must not be swallowed into a wrong answer or
    // a hang: the caller sees the panic.
    let n = 200usize;
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(|| {
        let mut v = vec![0u64; n];
        for_each_chunk_indexed_min_leaf(&plan(n, DISPATCHES), &mut v, 1, |start, _chunk| {
            assert_ne!(start, n / 2, "the leaf that was asked to fail");
        });
    });
    std::panic::set_hook(previous);
    assert!(
        outcome.is_err(),
        "a panic inside a dispatched leaf must reach the caller rather than be lost"
    );
}

#[test]
fn a_walker_nested_in_a_walker_completes_and_is_correct() {
    // An outer dispatch whose leaves each dispatch again: the inner
    // calls run on workers that are already inside the pool, which is
    // where a scheduler deadlocks if a worker blocks instead of
    // stealing.
    let (outer, inner) = (8usize, 32usize);
    let mut rows = vec![vec![0u64; inner]; outer];
    for_each_chunk_indexed_min_leaf(&plan(outer, DISPATCHES), &mut rows, 1, |start, chunk| {
        for (r, row) in chunk.iter_mut().enumerate() {
            let row_index = start + r;
            for_each_chunk_indexed_min_leaf(&plan(inner, DISPATCHES), row, 1, |s, c| {
                for (i, slot) in c.iter_mut().enumerate() {
                    *slot = (row_index * 1000 + s + i) as u64;
                }
            });
        }
    });
    for (r, row) in rows.iter().enumerate() {
        for (c, &got) in row.iter().enumerate() {
            assert_eq!(got, (r * 1000 + c) as u64, "row {r} column {c}");
        }
    }
}

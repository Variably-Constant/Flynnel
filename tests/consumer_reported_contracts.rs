//! Properties consumers named as load-bearing that nothing was
//! checking.
//!
//! Every test here exists because someone building on Flynnel said, in
//! answer to a direct question, that they depend on it and can point at
//! no documentation for it. That provenance is the point: these are not
//! properties derived from reading the export surface, which is how the
//! rest of the suite was written and why none of it covered these. An
//! API documents what it checks, so a suite derived from an API tests
//! the guarded direction and is silent on the rest.
//!
//! Each test names its source. When one fails, the question is not only
//! whether the change was intended but whether that consumer was told.

use std::collections::HashSet;
use std::sync::Mutex;
use std::thread::ThreadId;

use flynnel::JobPlan;
use flynnel::sched::arena::global_local_arena;
use flynnel::sched::par_iter::{for_each_chunk_indexed, for_each_chunk_indexed_min_leaf};

/// Refuse to report a pass on a host that cannot make the claim.
///
/// The two spread tests below need more than one worker to say anything
/// at all. Returning early there would report a pass having asserted
/// nothing, which is the one outcome worse than a failure: it looks
/// like coverage.
fn require_more_than_one_worker() {
    let workers = global_local_arena().total_workers();
    assert!(
        workers > 1,
        "these tests observe how many threads a dispatch reached and need \
         more than one worker; the arena has {workers}"
    );
}

/// splitmix64 step, folded so the spin cannot be optimized away.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// One heavy item: dependent steps, so the leaf costs microseconds
/// rather than nanoseconds and is worth splitting.
fn heavy(seed: u64) -> u64 {
    let mut acc = seed;
    for _ in 0..20_000 {
        acc = splitmix64(acc);
    }
    acc
}

/// `total_workers()` counts primaries and SMT siblings together.
///
/// Reported by an arbitrary-precision orchestrator as an undocumented
/// dependency: its gemm reads this number as a routing input, gating a
/// cellblock kernel on `workers <= 16`, not merely as a capacity hint.
/// Siblings park unless an SMT request is live, so a reasonable person
/// could implement this either way; a host with 8 physical cores
/// answers 8 under one reading and 16 under the other, which silently
/// selects a different kernel with a 1.5x swing between them.
///
/// Callers wanting the narrower figure have `primary_count()`. What
/// they need is for the meaning not to move underneath them.
#[test]
fn total_workers_counts_primaries_and_smt_siblings_together() {
    let arena = global_local_arena();
    let nodes = arena.node_count();
    assert!(nodes >= 1, "a host always has at least one node");

    let mut primaries = 0usize;
    let mut siblings = 0usize;
    for node in 0..nodes {
        let local = arena.node_arc(Some(node as u32));
        let (primary, sibling) = (local.primary_count(), local.smt_extension_count());
        // Pinned per node as well as in total. The sum is the figure
        // that cannot drift; the split is the figure a caller reasons
        // about, and a change that preserved the sum while moving the
        // boundary would leave the total assertion below untouched.
        assert_eq!(
            local.worker_count(),
            primary + sibling,
            "node {node}: the worker slice is exactly its primary head \
             plus its sibling tail"
        );
        assert!(primary >= 1, "node {node}: at least one always-active worker");
        primaries += primary;
        siblings += sibling;
    }

    assert_eq!(
        arena.total_workers(),
        primaries + siblings,
        "total_workers is the whole worker slice: {primaries} primary + \
         {siblings} sibling. A consumer routing on this number sees a \
         different value the moment that stops being true, and no error"
    );
}

/// `min_leaf = 1` keeps a heavy batch parallel at both the size
/// consumers dispatch and the size where the default floor does not.
///
/// This is the contract consumers rely on: every heavy-per-item site
/// they own passes `min_leaf = 1`, and losing the split there costs
/// them all parallelism on batches whose items are milliseconds to
/// seconds of work.
///
/// The companion test records why they pass it.
#[test]
fn min_leaf_one_keeps_a_heavy_batch_parallel_at_both_sizes() {
    require_more_than_one_worker();
    for n in [32usize, 256usize] {
        let spread = spread_of(n, Some(1));
        assert!(
            spread > 1,
            "min_leaf=1 must let {n} heavy items reach more than one \
             thread; saw {spread}"
        );
    }
}

/// The default floor serializes a heavy batch at 256 items and does not
/// at 32.
///
/// Consumers pass `min_leaf = 1` citing a scheduler fork's documentation
/// that at 256 items against a 256-item floor the bisect runs serial.
/// That claim traces to one source and reached here twice only because a
/// second consumer had read the same document, so it arrived as folklore
/// rather than as a measurement. It is correct: at 256 the default entry
/// runs on one thread.
///
/// It is also narrower than it sounds. At 32 items the same entry spread
/// across every worker, so the floor is not simply serializing every
/// small batch, and a consumer reasoning from the claim alone would
/// expect a serial run it does not get.
///
/// Both halves are pinned because either moving is worth knowing: if 256
/// starts spreading, `min_leaf = 1` has become belt-and-braces; if 32
/// stops, consumers at that size lose parallelism they currently have.
#[test]
fn the_default_floor_serializes_at_the_floor_size_but_not_below_it() {
    require_more_than_one_worker();
    assert_eq!(
        spread_of(256, None),
        1,
        "256 heavy items against a 256-item floor is one leaf"
    );
    let small = spread_of(32, None);
    assert!(
        small > 1,
        "32 heavy items reach many threads at the default floor; saw \
         {small}. If this becomes 1 the floor has started governing \
         below its own size and consumers should be told"
    );
}

/// Runs `n` heavy items through the indexed entry, with `min_leaf` when
/// given and the default floor otherwise, and reports how many threads
/// the body ran on.
fn spread_of(n: usize, min_leaf: Option<usize>) -> usize {
    let threads: Mutex<HashSet<ThreadId>> = Mutex::new(HashSet::new());
    let mut items: Vec<u64> = (0..n as u64).collect();
    let plan = JobPlan::new(0, n as u32);
    let expected: Vec<u64> = (0..n as u64).map(heavy).collect();

    let body = |start: usize, slots: &mut [u64]| {
        for (i, slot) in slots.iter_mut().enumerate() {
            *slot = heavy((start + i) as u64);
        }
        threads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(std::thread::current().id());
    };
    match min_leaf {
        Some(m) => for_each_chunk_indexed_min_leaf(&plan, &mut items, m, body),
        None => for_each_chunk_indexed(&plan, &mut items, body),
    }

    // The answers are checked on every route, so a spread count can
    // never come from a run that computed the wrong thing.
    assert_eq!(items, expected, "n={n} min_leaf={min_leaf:?} computed the wrong answers");
    threads.lock().unwrap_or_else(std::sync::PoisonError::into_inner).len()
}

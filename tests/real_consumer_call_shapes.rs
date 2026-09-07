//! Call shapes taken from consumers that exist, at the sizes they use.
//!
//! Every test here was written from a call site in a program that
//! depends on this crate, not from the export surface and not from a
//! guess about how the crate might be used. Where a consumer's per-item
//! cost is known, the test spends roughly that; where the size is
//! known, it uses that size. The consumers are described by what they
//! do rather than named, because the shape is the reusable part.
//!
//! This exists because the rest of the suite did not cover it. Tests
//! derived from reading the API cover the directions the API checks;
//! several defects here were found instead by asking a consumer what
//! they call and then checking whether it behaved. The shapes below are
//! the answers to that question, made executable so the next change has
//! to keep them working.
//!
//! Two properties are asserted throughout, and neither is a timing:
//! the answers are correct, and the routing decision the caller asked
//! for is the one the plan carries. Wall-clock speedup is deliberately
//! absent - the host these run on is shared, and a timing assertion
//! there measures the neighbour.
//!
//! How many threads a dispatch actually reached is the same kind of
//! measurement and is absent for the same reason. The bisect collapses
//! the remaining slice inline when it observes no steal pressure, which
//! is the right answer on a quiet host, so a spread of one is correct
//! behavior rather than a dropped hint. What the plan carries is the
//! decision; where the work lands is the scheduler's to make.

use flynnel::sched::par_iter::{collect_indexed, for_each_chunk_indexed_min_leaf};
use flynnel::{DispatchProfile, JobPlan, LeafShape, for_each_chunk, join};

/// splitmix64 step; the kernel every heavy leaf spins on.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Dependent steps, so a leaf costs real time and cannot be elided.
fn work(seed: u64, steps: u64) -> u64 {
    let mut acc = seed;
    for _ in 0..steps {
        acc = splitmix64(acc);
    }
    acc
}


// ---------------------------------------------------------------------
// A consumer offloading batches to a device, four sites of one shape
// ---------------------------------------------------------------------

/// `JobPlan::new(0, n).with_leaf_shape(LatencyCompute)` then
/// `for_each_chunk_indexed_min_leaf(.., 1, ..)`, at 32 to a few hundred
/// items whose per-item cost is a whole batched device launch.
///
/// All four of that consumer's sites are this exact sequence and none
/// passes a cost estimate. That combination is what makes the shape
/// hint reachable by the fine-grain guard: `new` seeds an estimate from
/// the active profile, and a guard that reads it without asking whether
/// the caller supplied it discards the shape below roughly 4200 items,
/// which is every size dispatched here.
#[test]
fn a_latency_compute_shape_survives_at_small_batch_sizes() {
    for n in [32usize, 64, 200] {
        let plan = JobPlan::new(0, n as u32).with_leaf_shape(LeafShape::LatencyCompute);

        assert!(
            plan.use_smt,
            "n={n}: the shape the caller names must survive to the plan; \
             siblings parked here leaves the wall clock in device waits, \
             measured at 38 to 48 percent for this workload"
        );
        assert_eq!(plan.leaf_shape, LeafShape::LatencyCompute);

        let mut out = vec![0u64; n];
        let expected: Vec<u64> = (0..n as u64).map(|i| work(i, 20_000)).collect();
        for_each_chunk_indexed_min_leaf(&plan, &mut out, 1, |start, slots| {
            for (i, s) in slots.iter_mut().enumerate() {
                *s = work((start + i) as u64, 20_000);
            }
        });

        assert_eq!(out, expected, "n={n}: the answers must not depend on the split");
    }
}

/// `flynnel::join` over a CPU half and a device half, both built from
/// the same plan shape. The same consumer, at a site that is easy to
/// overlook because it is a join rather than a chunked helper, and the
/// leaf-shape precedence rule reaches it too.
#[test]
fn a_join_splits_a_cpu_half_from_a_device_half() {
    let n = 64u32;
    let plan = JobPlan::new(0, n).with_leaf_shape(LeafShape::LatencyCompute);
    let (cpu, dev) = join(&plan, || work(1, 40_000), || work(2, 40_000));
    assert_eq!(cpu, work(1, 40_000));
    assert_eq!(dev, work(2, 40_000));
    assert!(plan.use_smt, "the join arms inherit the same plan and the same hint");
}

// ---------------------------------------------------------------------
// A streaming-audio consumer, fanning out per file and per chunk
// ---------------------------------------------------------------------

/// `JobPlan::new(0, n)` then `collect_indexed(&plan, n, 1, ..)` with no
/// shape and no estimate - the plainest constructor any consumer uses.
#[test]
fn collect_indexed_with_min_leaf_one_and_no_hint_at_all() {
    for n in [16usize, 120] {
        let plan = JobPlan::new(0, n as u32);
        let got = collect_indexed(&plan, n, 1, |i| work(i as u64, 15_000));
        let expected: Vec<u64> = (0..n as u64).map(|i| work(i, 15_000)).collect();
        assert_eq!(got, expected, "n={n}: collect_indexed must preserve index order");
    }
}

// ---------------------------------------------------------------------
// A nearest-neighbour index, at its vector dimensions
// ---------------------------------------------------------------------

/// `JobPlan::new(0, c).with_leaf_shape(PortCompute)` then
/// `for_each_chunk_indexed_min_leaf(.., 1, ..)`. The same sequence as
/// the device-offload consumer above with the other shape, and the
/// reason a site like this is unaffected in outcome by the fine-grain
/// guard: PortCompute and FineGrain both map to PortBound, so
/// discarding the shape changes nothing here.
///
/// Which makes it the wrong place to look for that defect, and worth
/// keeping for exactly that reason.
#[test]
fn a_port_compute_shape_at_a_vector_dimension() {
    let c = 96usize;
    let plan = JobPlan::new(0, c as u32).with_leaf_shape(LeafShape::PortCompute);
    assert!(!plan.use_smt, "port-saturating work keeps the siblings parked");

    let mut lists = vec![0u64; c];
    let expected: Vec<u64> = (0..c as u64).map(|i| work(i, 8_000)).collect();
    for_each_chunk_indexed_min_leaf(&plan, &mut lists, 1, |start, slots| {
        for (i, s) in slots.iter_mut().enumerate() {
            *s = work((start + i) as u64, 8_000);
        }
    });
    assert_eq!(lists, expected);
}

// ---------------------------------------------------------------------
// A full-text index scanner, over a shared mapping
// ---------------------------------------------------------------------

/// `JobPlan::set_profile(k_outer, batch, Streaming)` where k_outer is
/// `(usize::BITS - count.leading_zeros()).min(24)` - a k_outer far
/// above anything else in the suite, capped at 24 by the caller.
#[test]
fn a_streaming_profile_at_a_large_k_outer() {
    let count = 1_000_000usize;
    let k_outer = (usize::BITS - count.leading_zeros()).min(24) as u8;
    assert_eq!(k_outer, 20, "the consumer's own expression, pinned so a change is visible");

    let plan = JobPlan::set_profile(
        k_outer,
        count.min(u32::MAX as usize) as u32,
        DispatchProfile::Streaming,
    );
    assert!(!plan.use_smt, "streaming keeps siblings parked, which the caller relies on");
    assert_eq!(plan.k_outer, k_outer, "a k_outer of 20 must survive construction");
}

/// `flynnel::join` bisecting an index range rather than a slice,
/// recursing to a leaf.
///
/// A scanner whose source is a shared mapping has no slice to take, and
/// the chunked helpers all want `&mut`, so `join` over indices is the
/// only route open to it. Nothing else in the suite drives `join`
/// recursively over indices, so this is the only cover for a consumer
/// that cannot use the slice helpers at all.
#[test]
fn a_join_bisects_an_index_range_rather_than_a_slice() {
    fn scan_range(start: usize, end: usize, plan: &JobPlan) -> u64 {
        const LEAF: usize = 256;
        if end - start <= LEAF {
            let mut acc = 0u64;
            for i in start..end {
                acc ^= work(i as u64, 400);
            }
            return acc;
        }
        let mid = start + (end - start) / 2;
        let (a, b) = join(
            plan,
            || scan_range(start, mid, plan),
            || scan_range(mid, end, plan),
        );
        a ^ b
    }

    let n = 4096usize;
    let k_outer = (usize::BITS - n.leading_zeros()).min(24) as u8;
    let plan = JobPlan::set_profile(k_outer, n as u32, DispatchProfile::Streaming);

    let got = scan_range(0, n, &plan);
    let expected = (0..n as u64).fold(0u64, |acc, i| acc ^ work(i, 400));
    assert_eq!(got, expected, "a recursive index-range bisect must fold to the serial answer");
}

// ---------------------------------------------------------------------
// A desktop compositor, drawing art and stepping a simulation
// ---------------------------------------------------------------------

/// `set_profile(k_outer, n, MemoryBound)` then `for_each_chunk`, where
/// the leaf blocks on a store fetch. The only production use of
/// `MemoryBound` among the consumers surveyed.
#[test]
fn a_memory_bound_profile_over_for_each_chunk() {
    const ART_PX: usize = 256;
    let k_outer = ((ART_PX * ART_PX) as f64).log2() as u8;
    assert_eq!(k_outer, 16, "the consumer's own expression");

    let n = 512usize;
    let mut slots: Vec<(u64, bool)> = (0..n as u64).map(|i| (i, false)).collect();
    let plan = JobPlan::set_profile(k_outer, n as u32, DispatchProfile::MemoryBound);
    assert!(plan.use_smt, "MemoryBound engages siblings to cover the stalls");

    for_each_chunk(&plan, &mut slots, |chunk| {
        for (seed, done) in chunk.iter_mut() {
            *done = work(*seed, 3_000) != 0;
        }
    });
    assert!(slots.iter().all(|(_, done)| *done), "every slot must be visited exactly once");
}

/// `set_profile(0, n, LatencyBound)` then `for_each_chunk` over a body
/// that steps a simulation, at k_outer 0 with a batch of tens.
#[test]
fn a_latency_bound_profile_at_k_outer_zero() {
    let n = 48usize;
    let plan = JobPlan::set_profile(0, n as u32, DispatchProfile::LatencyBound);
    assert!(plan.use_smt);

    let mut bodies: Vec<u64> = (0..n as u64).collect();
    let expected: Vec<u64> = (0..n as u64).map(|i| work(i, 6_000)).collect();
    for_each_chunk(&plan, &mut bodies, |chunk| {
        for b in chunk.iter_mut() {
            *b = work(*b, 6_000);
        }
    });
    assert_eq!(bodies, expected);
}

// ---------------------------------------------------------------------
// An indexing daemon and a launcher, both on the plan-free API
// ---------------------------------------------------------------------

/// The `flat` API, which takes no plan at all: `par_for_each_mut` for
/// per-slot indexing, and `flat::join` for a two-way directory walk.
/// Only `flat::join` had any integration cover before this.
#[test]
fn the_flat_api_runs_without_a_plan() {
    let mut jobs: Vec<u64> = (0..300u64).collect();
    let expected: Vec<u64> = (0..300u64).map(|i| work(i, 2_000)).collect();
    flynnel::flat::par_for_each_mut(&mut jobs, |job| *job = work(*job, 2_000));
    assert_eq!(jobs, expected);

    let (a, b) = flynnel::flat::join(|| work(7, 9_000), || work(11, 9_000));
    assert_eq!((a, b), (work(7, 9_000), work(11, 9_000)));
}

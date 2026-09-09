//! What seed-depth hysteresis and estimate smoothing cost, and what
//! they buy, measured against each other and against neither.
//!
//! ## The shape being reproduced
//!
//! `adaptive_seed_depth` turns a per-item estimate into a leaf count by
//! rounding up to a power of two, so a workload whose `est * items`
//! sits near a boundary seeds a different number of leaves depending on
//! where the estimate happened to land. One measured cell read 233,
//! 233, 365 and 382 ns for the same operation across four runs and
//! seeded 32, 32, 64 and 64 leaves.
//!
//! Each iteration here alternates the estimate across such a boundary,
//! which is the worst case rather than the typical one: a caller whose
//! estimate is stable pays nothing for either stabiliser, and a caller
//! whose estimate straddles a boundary pays the full cost of the flip.
//!
//! ## The four arms
//!
//! - `neither` - what ships today.
//! - `hysteresis` - a change of depth needs two agreeing calls.
//! - `smoothing` - the estimate is averaged against the site's history.
//! - `both`.
//!
//! Every arm runs in one process, back to back, because the quantity
//! under test is a decision driven by a measured estimate: arms split
//! across processes carry whatever else differed between them, and on
//! a shared host that is the larger effect.
//!
//! Each arm owns a `static CallSiteState`, so the depth one arm settled
//! on and the average it learned do not reach the next.

#![allow(clippy::missing_docs_in_private_items)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::JobPlan;
use flynnel::sched::call_site::{CallSiteState, SiteRef};
use flynnel::sched::par_iter::{for_each_chunk, set_estimate_smoothing, set_seed_hysteresis};

/// Items per dispatch. With the 1 ms leaf target and 24 workers the
/// leaf count crosses 32 at an estimate near 976 ns, so the two hints
/// below straddle it.
const ITEMS: usize = 32_768;

/// The two estimates the caller alternates between, one either side of
/// the boundary. Their ratio is 1.5, inside the 1.71 spread one
/// operation actually showed across runs.
const EST_LOW_NS: u32 = 800;
const EST_HIGH_NS: u32 = 1_200;

/// Per-item work: a dependent chain, so the cost is the chain rather
/// than anything the optimiser can vectorise away.
#[inline(never)]
fn item_work(seed: u64) -> u64 {
    let mut x = seed;
    for _ in 0..220 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x = x.wrapping_mul(0x100000001B3);
    }
    x
}

/// One dispatch at the given estimate.
fn dispatch(site: SiteRef, buf: &mut [u64], est_ns: u32) {
    let plan = JobPlan::new(0, buf.len() as u32)
        .with_site(site)
        .with_estimated_per_item_ns(est_ns);
    for_each_chunk(&plan, buf, |slice| {
        for x in slice {
            *x = item_work(*x);
        }
    });
}

/// A pair of dispatches straddling the boundary: the unit of work every
/// arm is timed over, so the arms differ only in how the scheduler
/// reacts to the second estimate.
fn straddling_pair(site: SiteRef, buf: &mut [u64]) {
    dispatch(site, buf, EST_LOW_NS);
    dispatch(site, buf, EST_HIGH_NS);
}

fn bench_stabilisers(c: &mut Criterion) {
    let mut group = c.benchmark_group("seed_depth_stability");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(20);

    let mut buf: Vec<u64> = (0..ITEMS as u64).collect();

    static NEITHER: CallSiteState = CallSiteState::new();
    static HYSTERESIS: CallSiteState = CallSiteState::new();
    static SMOOTHING: CallSiteState = CallSiteState::new();
    static BOTH: CallSiteState = CallSiteState::new();

    group.bench_function("neither", |b| {
        set_seed_hysteresis(false);
        set_estimate_smoothing(false);
        b.iter(|| {
            straddling_pair(SiteRef::new(&NEITHER), &mut buf);
            black_box(buf[0]);
        });
    });

    group.bench_function("hysteresis", |b| {
        set_seed_hysteresis(true);
        set_estimate_smoothing(false);
        b.iter(|| {
            straddling_pair(SiteRef::new(&HYSTERESIS), &mut buf);
            black_box(buf[0]);
        });
    });

    group.bench_function("smoothing", |b| {
        set_seed_hysteresis(false);
        set_estimate_smoothing(true);
        b.iter(|| {
            straddling_pair(SiteRef::new(&SMOOTHING), &mut buf);
            black_box(buf[0]);
        });
    });

    group.bench_function("both", |b| {
        set_seed_hysteresis(true);
        set_estimate_smoothing(true);
        b.iter(|| {
            straddling_pair(SiteRef::new(&BOTH), &mut buf);
            black_box(buf[0]);
        });
    });

    // Leave the process as it was found, so a later group in the same
    // binary is not measured under whatever the last arm set.
    set_seed_hysteresis(false);
    set_estimate_smoothing(false);

    group.finish();
}

/// The control the four arms above need: the same pair of dispatches
/// with an estimate that does not move, where neither stabiliser has
/// anything to do. A cost here is a cost paid by every caller, not only
/// by one whose estimate straddles a boundary.
fn bench_stable_estimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("seed_depth_stable_estimate");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(20);

    let mut buf: Vec<u64> = (0..ITEMS as u64).collect();

    static NEITHER: CallSiteState = CallSiteState::new();
    static BOTH: CallSiteState = CallSiteState::new();

    group.bench_function("neither", |b| {
        set_seed_hysteresis(false);
        set_estimate_smoothing(false);
        b.iter(|| {
            dispatch(SiteRef::new(&NEITHER), &mut buf, EST_LOW_NS);
            dispatch(SiteRef::new(&NEITHER), &mut buf, EST_LOW_NS);
            black_box(buf[0]);
        });
    });

    group.bench_function("both", |b| {
        set_seed_hysteresis(true);
        set_estimate_smoothing(true);
        b.iter(|| {
            dispatch(SiteRef::new(&BOTH), &mut buf, EST_LOW_NS);
            dispatch(SiteRef::new(&BOTH), &mut buf, EST_LOW_NS);
            black_box(buf[0]);
        });
    });

    set_seed_hysteresis(false);
    set_estimate_smoothing(false);

    group.finish();
}

criterion_group!(benches, bench_stabilisers, bench_stable_estimate);
criterion_main!(benches);

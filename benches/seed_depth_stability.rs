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
//! ## Three regimes, because callers live in all of them
//!
//! - `straddle` - the estimate alternates across a boundary every
//!   dispatch. The worst case, and the only one either mechanism can
//!   help.
//! - `stable` - the estimate is fixed at a target the worker floor
//!   barely binds. Both mechanisms are pure cost here.
//! - `pinned` - the estimate is so far under the floor that `max()`
//!   supplies the whole target and the estimate reaches nothing. A
//!   consumer's sites are mostly here.
//!
//! The two cost regimes decide whether either mechanism is shippable: a
//! cost there is paid by every caller, including those that can never
//! cross a boundary.
//!
//! ## Why every regime is registered twice, in opposite order
//!
//! Criterion runs arms sequentially, so a load that arrives or departs
//! during a group lands on some arms and not others. A measured
//! position effect of 2.6x on a first arm has been seen on this host,
//! which is larger than anything being measured here.
//!
//! Registering each regime forward and reversed makes that visible
//! rather than assumed: an effect present in one order and absent in
//! the other is the box, and one whose ratio survives both is the code.
//! This is what lets the bench be read on a host that cannot be made
//! quiet, which is the only kind available.
//!
//! Every arm owns a distinct `CallSiteState`, so the depth one arm
//! settles on and the average it learns never become another's starting
//! condition.

#![allow(clippy::missing_docs_in_private_items)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::JobPlan;
use flynnel::sched::call_site::{CallSiteState, SiteRef};
use flynnel::sched::par_iter::{for_each_chunk, set_estimate_smoothing, set_seed_hysteresis};

/// One site per arm per ordering, so nothing a site learns crosses
/// between them.
static SITES: [CallSiteState; 16] = [const { CallSiteState::new() }; 16];

/// Which stabilisers an arm runs with.
#[derive(Copy, Clone)]
struct Arm {
    name: &'static str,
    hysteresis: bool,
    smoothing: bool,
}

const NEITHER: Arm = Arm { name: "neither", hysteresis: false, smoothing: false };
const HYSTERESIS: Arm = Arm { name: "hysteresis", hysteresis: true, smoothing: false };
const SMOOTHING: Arm = Arm { name: "smoothing", hysteresis: false, smoothing: true };
const BOTH: Arm = Arm { name: "both", hysteresis: true, smoothing: true };

/// Per-item work: a dependent chain, so the cost is the chain rather
/// than anything the optimizer can vectorize away.
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

/// Register one regime's arms in the given order.
///
/// `estimates` is the pair a single iteration dispatches with: two
/// different values straddle a boundary, two equal ones hold still.
/// `site_base` indexes this regime's block of [`SITES`], and `order`
/// distinguishes the forward registration from the reversed one so the
/// two never share a site.
fn register(
    c: &mut Criterion,
    group_name: &str,
    items: usize,
    estimates: (u32, u32),
    arms: &[Arm],
    site_base: usize,
) {
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(20);

    let mut buf: Vec<u64> = (0..items as u64).collect();

    for (i, arm) in arms.iter().enumerate() {
        let site = SiteRef::new(&SITES[site_base + i]);
        group.bench_function(arm.name, |b| {
            set_seed_hysteresis(arm.hysteresis);
            set_estimate_smoothing(arm.smoothing);
            b.iter(|| {
                dispatch(site, &mut buf, estimates.0);
                dispatch(site, &mut buf, estimates.1);
                black_box(buf[0]);
            });
        });
    }

    // Leave the process as found, so a later group is not measured
    // under whatever the last arm set.
    set_seed_hysteresis(false);
    set_estimate_smoothing(false);
    group.finish();
}

/// 32768 items on a 24-worker host crosses a boundary near 976 ns, so
/// 800 and 1200 straddle it. Their ratio is 1.5, inside the 1.71 one
/// operation actually showed across runs.
fn bench_straddle(c: &mut Criterion) {
    let fwd = [NEITHER, HYSTERESIS, SMOOTHING, BOTH];
    let rev = [BOTH, SMOOTHING, HYSTERESIS, NEITHER];
    register(c, "straddle", 32_768, (800, 1_200), &fwd, 0);
    register(c, "straddle_rev", 32_768, (800, 1_200), &rev, 4);
}

/// The same dispatches with the estimate held still, where neither
/// mechanism has anything to do.
fn bench_stable(c: &mut Criterion) {
    let fwd = [NEITHER, BOTH];
    let rev = [BOTH, NEITHER];
    register(c, "stable", 32_768, (800, 800), &fwd, 8);
    register(c, "stable_rev", 32_768, (800, 800), &rev, 10);
}

/// 4096 items at 100 ns gives a target of 0.41, which the worker floor
/// lifts to the worker count: the estimate reaches the decision not at
/// all. A cost that appears only when the floor binds shows here.
fn bench_pinned(c: &mut Criterion) {
    let fwd = [NEITHER, BOTH];
    let rev = [BOTH, NEITHER];
    register(c, "pinned", 4_096, (100, 100), &fwd, 12);
    register(c, "pinned_rev", 4_096, (100, 100), &rev, 14);
}

criterion_group!(benches, bench_straddle, bench_stable, bench_pinned);
criterion_main!(benches);

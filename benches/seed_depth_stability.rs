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
//! during a group lands on some arms and not others. Two of this
//! bench's six groups have read 61 and 68 percent apart between their
//! two orders, against arms whose own intervals are near one percent -
//! so a shifting load moves a group by far more than this resolves.
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
//!
//! ## What the reported line says
//!
//! `flips` counts dispatches at that arm's site which seeded a
//! different leaf count from the dispatch before them. That is the
//! quantity the two mechanisms exist to reduce, and the timings do not
//! express it: a flip between two adjacent depths costs little either
//! way, so an arm can be fast and unstable or slow and steady. Reading
//! cost without it says what each mechanism charges and not what it
//! buys.
//!
//! `pct` is the fraction of the arm's last dispatch that its measuring
//! thread spent on a core. Nothing consumes it; it is reported so the
//! distribution across quiet and loaded runs can be read off real runs
//! rather than assumed.
//!
//! It answers a different question from the two orders, and neither
//! subsumes the other. Alternation catches load that arrives or departs
//! during a group, because that lands on some arms and not others; load
//! present across both orders moves them equally and they agree. So a
//! pair of orders can agree while both were measured under the same
//! contention, and the occupancy figure is what would say so.

#![allow(clippy::missing_docs_in_private_items)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::JobPlan;
use flynnel::sched::call_site::{CallSiteState, SiteRef};
use flynnel::sched::par_iter::{for_each_chunk, set_seed_hysteresis};

/// One site per arm per ordering, so nothing a site learns crosses
/// between them.
static SITES: [CallSiteState; 16] = [const { CallSiteState::new() }; 16];

/// Whether an arm runs with seed-depth hysteresis.
#[derive(Copy, Clone)]
struct Arm {
    name: &'static str,
    hysteresis: bool,
}

const OFF: Arm = Arm { name: "off", hysteresis: false };
const ON: Arm = Arm { name: "on", hysteresis: true };

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
        let state = &SITES[site_base + i];
        let site = SiteRef::new(state);
        group.bench_function(arm.name, |b| {
            set_seed_hysteresis(arm.hysteresis);
            b.iter(|| {
                dispatch(site, &mut buf, estimates.0);
                dispatch(site, &mut buf, estimates.1);
                black_box(buf[0]);
            });
        });
        // An arm whose dispatches never reported reads "none" rather
        // than a number, so a missing measurement is not a measurement.
        let pct = match state.recent_occupancy() {
            Some(p) => p.to_string(),
            None => "none".to_string(),
        };
        eprintln!(
            "occupancy {}/{} pct={} flips={}",
            group_name,
            arm.name,
            pct,
            state.seed_depth_flips(),
        );
    }

    // Leave the process at the shipped default, so a later group is not
    // measured under whatever the last arm set.
    set_seed_hysteresis(true);
    group.finish();
}

/// 32768 items on a 24-worker host crosses a boundary near 976 ns, so
/// 800 and 1200 straddle it. Their ratio is 1.5, inside the 1.71 one
/// operation actually showed across runs.
fn bench_straddle(c: &mut Criterion) {
    register(c, "straddle", 32_768, (800, 1_200), &[OFF, ON], 0);
    register(c, "straddle_rev", 32_768, (800, 1_200), &[ON, OFF], 4);
}

/// The same dispatches with the estimate held still, where the
/// mechanism has nothing to do.
fn bench_stable(c: &mut Criterion) {
    register(c, "stable", 32_768, (800, 800), &[OFF, ON], 8);
    register(c, "stable_rev", 32_768, (800, 800), &[ON, OFF], 10);
}

/// 4096 items at 100 ns gives a target of 0.41, which the worker floor
/// lifts to the worker count: the estimate reaches the decision not at
/// all. A cost that appears only when the floor binds shows here.
fn bench_pinned(c: &mut Criterion) {
    register(c, "pinned", 4_096, (100, 100), &[OFF, ON], 12);
    register(c, "pinned_rev", 4_096, (100, 100), &[ON, OFF], 14);
}

criterion_group!(benches, bench_straddle, bench_stable, bench_pinned);
criterion_main!(benches);

//! What an input size costs for crossing a fan-out boundary, at sizes
//! close enough together that only the leaf count differs.
//!
//! ## The shape being measured
//!
//! `adaptive_seed_depth` computes
//! `max(workers, items * est / TARGET_LEAF_WORK_NS)`, caps it at
//! `items`, and rounds up to a power of two. The rounding is the last
//! step, so the seeded leaf count is a step function of the input size:
//! two calls whose sizes differ by one percent can seed 32 leaves and
//! 64. Nothing is noisy here and nothing flips - the same size always
//! gets the same shape. It is a sawtooth in `n`.
//!
//! A consumer sweeping sizes sees that discontinuity in its own numbers
//! and reads it as its kernel changing behavior.
//!
//! ## Where the boundary is, and why this does not have to hunt for it
//!
//! The division is integer and the worker floor is the pool width, so
//! with 24 workers the count is 32 until the quotient reaches 33:
//!
//! ```text
//! 32 -> 64 leaves when floor(items * est / 1e6) >= 33
//!                 i.e. items >= 33e6 / est
//! ```
//!
//! At `est = 50` that is 660_000 items. The sizes below bracket it:
//! 600_000 and 640_000 seed 32, 665_000 and upward seed 64, and 655_000
//! is the last size on the low side. The per-item work is tuned to the
//! same 50 ns the plan is told, so the estimate describes the workload
//! rather than steering it somewhere the work does not go.
//!
//! On a host whose worker count is not 24 the floor moves and so does
//! the boundary; the printed leaf count per size is what says where it
//! actually fell.
//!
//! ## Registered in both size orders
//!
//! Criterion measures sequentially, so a load arriving during a group
//! lands on some sizes and not others. Sweeping the sizes forward and
//! reversed makes that visible: a step present in one order and absent
//! in the other is the machine. Reading the step against the
//! within-size spread is the whole measurement, and a step buried under
//! that spread is a fact about the code with no consequence.

#![allow(clippy::missing_docs_in_private_items)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::JobPlan;
use flynnel::sched::call_site::{CallSiteState, SiteRef};
use flynnel::sched::par_iter::for_each_chunk;

/// One site per size per ordering, so what one size settles on is never
/// another's starting condition.
static SITES: [CallSiteState; 12] = [const { CallSiteState::new() }; 12];

/// The per-item estimate handed to the plan, in nanoseconds. The work
/// below is tuned to cost about this.
const EST_NS: u32 = 50;

/// Sizes bracketing the 32-to-64 boundary at `EST_NS` on a 24-worker
/// host, which sits at 660_000.
const SIZES: [usize; 6] = [600_000, 640_000, 655_000, 665_000, 700_000, 760_000];

/// About fifty nanoseconds of dependent work, so the estimate the plan
/// is given describes what the item actually costs.
#[inline(never)]
fn item_work(seed: u64) -> u64 {
    let mut x = seed;
    for _ in 0..11 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x = x.wrapping_mul(0x100000001B3);
    }
    x
}

fn dispatch(site: SiteRef, buf: &mut [u64]) {
    let plan = JobPlan::new(0, buf.len() as u32)
        .with_site(site)
        .with_estimated_per_item_ns(EST_NS);
    for_each_chunk(&plan, buf, |slice| {
        for x in slice {
            *x = item_work(*x);
        }
    });
}

/// Register one pass over the sizes, in the order given.
fn register(c: &mut Criterion, group_name: &str, order: &[usize], site_base: usize) {
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(20);

    for (i, &items) in order.iter().enumerate() {
        let state = &SITES[site_base + i];
        let site = SiteRef::new(state);
        let mut buf: Vec<u64> = (0..items as u64).collect();
        // Per item, so sizes are comparable to each other rather than
        // to the total work each does.
        group.throughput(criterion::Throughput::Elements(items as u64));
        group.bench_function(format!("n{items}"), |b| {
            b.iter(|| {
                dispatch(site, &mut buf);
                black_box(buf[0]);
            });
        });
        // A size whose dispatches never reported reads "none" rather
        // than a number, so a missing measurement is not a measurement.
        let pct = match state.recent_occupancy() {
            Some(p) => p.to_string(),
            None => "none".to_string(),
        };
        eprintln!("smoothness {group_name}/n{items} pct={pct}");
    }

    group.finish();
}

fn bench_smoothness(c: &mut Criterion) {
    let mut reversed = SIZES;
    reversed.reverse();
    register(c, "smoothness", &SIZES, 0);
    register(c, "smoothness_rev", &reversed, 6);
}

criterion_group!(benches, bench_smoothness);
criterion_main!(benches);

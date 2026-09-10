//! What running under the wrong workload class costs, on a quiet box.
//!
//! ## The question
//!
//! Preemption lands on some leaves and not others, so a neighbour's load
//! reaches the classifier as variance rather than as uniform slowdown. A
//! site whose leaves are uniform can therefore read as irregular on a
//! loaded host and migrate class, and the class selects a fan-out shape
//! and an SMT setting. The choice outlives the load that caused it.
//!
//! A slowdown that ends when the load ends costs what it costs. A
//! misclassification that persists goes on charging after the box is
//! quiet, and that is the quantity here.
//!
//! ## Both arms run quiet, which is the part that is easy to invert
//!
//! Load is what causes the misclassification. It must not be present
//! while the cost is measured, or the reading prices contention instead.
//! Nothing in this file generates load, and a run taken while a
//! neighbour is up measures something else.
//!
//! ## Every profile, rather than one pair
//!
//! Which migration a loaded host actually produces is a separate
//! measurement. Timing only the pair it produces would make this bench
//! wait on that answer, and a guessed pair is likelier to be wrong than
//! right with five classes in the vocabulary. Every profile is timed
//! against the same work instead, so the cost of any pair is a
//! subtraction on one table.
//!
//! Arms are `DispatchProfile` rather than `WorkloadClass` because the
//! profile is what reaches the plan. The class vocabulary has one more
//! variant than the profile: `FineGrain` and `PortBound` both map to
//! `PortBound`, so confusing those two changes nothing the scheduler
//! does, and that is a result rather than an omission.
//!
//! ## Why every shape is registered twice, in opposite order
//!
//! Criterion runs arms sequentially, so a load arriving or departing
//! during a group lands on some arms and not others. An effect present
//! in one order and absent in the other is the box; one whose ratio
//! survives both is the code. The precedent for reading this bench that
//! way is `seed_depth_stability`, where two groups read 61 and 68
//! percent apart between orders against arms whose own intervals were
//! near one percent.
//!
//! Every arm owns a distinct `CallSiteState`, so the leaf statistics,
//! policy arm and placement one arm learns never become another's
//! starting condition.
//!
//! ## What the reported line says
//!
//! `pct` is the fraction of the arm's last dispatch that its measuring
//! thread spent on a core, and reads `none` when no dispatch reported
//! one, so a missing measurement is not mistaken for a quiet reading.
//! `global` is the process-wide class after the arm ran: each arm pins
//! its profile on the plan, and a global that has moved would mean the
//! pin leaked and the arms are no longer independent.

#![allow(clippy::missing_docs_in_private_items)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::JobPlan;
use flynnel::dispatch_profile::DispatchProfile;
use flynnel::sched::adaptive_profile::active_workload_class;
use flynnel::sched::call_site::{CallSiteState, SiteRef};
use flynnel::sched::par_iter::for_each_chunk;

/// One site per arm per ordering, so nothing a site learns crosses
/// between them. Three shapes, five profiles, two orders.
static SITES: [CallSiteState; 30] = [const { CallSiteState::new() }; 30];

/// The profiles a plan can carry, in a fixed order. Reversed for the
/// second registration of each shape.
const PROFILES: [(&str, DispatchProfile); 5] = [
    ("port", DispatchProfile::PortBound),
    ("latency", DispatchProfile::LatencyBound),
    ("memory", DispatchProfile::MemoryBound),
    ("streaming", DispatchProfile::Streaming),
    ("unspecified", DispatchProfile::Unspecified),
];

/// Dependent xorshift chain. The cost is the dependency chain rather
/// than anything the optimizer can vectorize away, and `rounds` is what
/// moves an item between the classifier's per-item bands.
#[inline(never)]
fn chain(seed: u64, rounds: u32) -> u64 {
    let mut x = seed | 1;
    for _ in 0..rounds {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x = x.wrapping_mul(0x100000001B3);
    }
    x
}

/// Dependent gather over a table larger than last-level cache, so each
/// item pays a miss that a sibling thread could overlap. This is the
/// shape the SMT setting is supposed to discriminate, so it is where a
/// wrong profile has the most room to cost something.
#[inline(never)]
fn gather(seed: u64, table: &[u64]) -> u64 {
    let mask = table.len() - 1;
    let mut idx = (seed as usize) & mask;
    let mut acc = 0u64;
    for _ in 0..64 {
        let v = table[idx];
        acc = acc.wrapping_add(v);
        idx = (v as usize) & mask;
    }
    acc
}

/// The three shapes, spanning the per-item bands the classifier splits
/// on. `Gather` is the one whose cost is a stall rather than a port.
#[derive(Copy, Clone)]
enum Shape {
    Fine,
    Heavy,
    Gather,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Self::Fine => "fine",
            Self::Heavy => "heavy",
            Self::Gather => "gather",
        }
    }

    /// Items per dispatch. The gather shape uses fewer because each
    /// item costs a chain of misses.
    fn items(self) -> usize {
        match self {
            Self::Fine => 262_144,
            Self::Heavy => 16_384,
            Self::Gather => 32_768,
        }
    }

    fn apply(self, slice: &mut [u64], table: &[u64]) {
        match self {
            Self::Fine => {
                for x in slice {
                    *x = chain(*x, 2);
                }
            }
            Self::Heavy => {
                for x in slice {
                    *x = chain(*x, 400);
                }
            }
            Self::Gather => {
                for x in slice {
                    *x = gather(*x, table);
                }
            }
        }
    }
}

/// 8 MiB of u64, past last-level cache on the hosts this runs on, and a
/// power of two so the index mask is exact.
const TABLE_LEN: usize = 1 << 20;

fn build_table() -> Vec<u64> {
    let mut t = vec![0u64; TABLE_LEN];
    let mut x = 0x243F_6A88_85A3_08D3u64;
    for (i, slot) in t.iter_mut().enumerate() {
        x = chain(x ^ i as u64, 1);
        *slot = x;
    }
    t
}

/// Register one shape's arms in the given order.
///
/// `site_base` indexes this shape's block of [`SITES`], so the forward
/// registration and the reversed one never share a site.
fn register(
    c: &mut Criterion,
    group_name: &str,
    shape: Shape,
    profiles: &[(&str, DispatchProfile)],
    site_base: usize,
    table: &[u64],
) {
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(20);

    let items = shape.items();
    let mut buf: Vec<u64> = (0..items as u64).collect();

    for (i, (arm_name, profile)) in profiles.iter().enumerate() {
        let state = &SITES[site_base + i];
        let site = SiteRef::new(state);
        // Built once, outside the timed loop. set_profile marks the
        // profile explicit, which is what pins it: a site's own learning
        // cannot override a profile the caller supplied. Constructing it
        // per iteration would put the classifier's work inside the
        // measurement, where it is not what the arm is comparing.
        let plan = JobPlan::set_profile(0, items as u32, *profile).with_site(site);

        group.bench_function(*arm_name, |b| {
            b.iter(|| {
                for_each_chunk(&plan, &mut buf, |slice| shape.apply(slice, table));
                black_box(buf[0]);
            });
        });
        let pct = match state.recent_occupancy() {
            Some(p) => p.to_string(),
            None => "none".to_string(),
        };
        eprintln!(
            "class cost {}/{} pct={} global={:?}",
            group_name,
            arm_name,
            pct,
            active_workload_class(),
        );
    }

    group.finish();
}

fn bench_fine(c: &mut Criterion) {
    let table = build_table();
    let mut rev = PROFILES;
    rev.reverse();
    register(c, "fine", Shape::Fine, &PROFILES, 0, &table);
    register(c, "fine_rev", Shape::Fine, &rev, 5, &table);
}

fn bench_heavy(c: &mut Criterion) {
    let table = build_table();
    let mut rev = PROFILES;
    rev.reverse();
    register(c, "heavy", Shape::Heavy, &PROFILES, 10, &table);
    register(c, "heavy_rev", Shape::Heavy, &rev, 15, &table);
}

fn bench_gather(c: &mut Criterion) {
    let table = build_table();
    let mut rev = PROFILES;
    rev.reverse();
    register(c, "gather", Shape::Gather, &PROFILES, 20, &table);
    register(c, "gather_rev", Shape::Gather, &rev, 25, &table);
}

criterion_group!(benches, bench_fine, bench_heavy, bench_gather);
criterion_main!(benches);

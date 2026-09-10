//! A/B microbench for the SIMC primitive
//! ([`cooperative_join_n_flat`]) against rayon's closest equivalent
//! (`rayon::scope::spawn` fan-out).
//!
//! ## Why this bench exists
//!
//! The Flynn-axis taxonomy in `crate::backend::mod`'s doc table
//! assigns `cooperative_join_n` as the SIMC (Single Instruction,
//! Multiple Cores) primitive. Its win zone is N independent uniform-
//! cost closures fanned out across N cores as one logical mega-
//! SIMD vector. The flat-shape variant
//! [`cooperative_join_n_flat`] pushes each closure directly to a
//! specific peer worker's mailbox (URD-style owner-directed
//! distribution); the target worker drains its mailbox ahead of any
//! deque in `find_work`, so each closure starts on its assigned core with
//! no CAS contention on a shared deque head.
//!
//! Rayon's `scope::spawn` fans out via the shared deque + random
//! peer-steal. Every closure pushed by the calling thread lands
//! on one shared deque; thieves race to grab them. Cross-CCX peers
//! can pull a closure that was pushed from a core that shares L1d
//! with a different sibling - the cache hit-rate is random.
//!
//! ## Bench-audit (HARD RULE 3)
//!
//! - **Same payload**: each closure does the same fixed amount of
//!   pure CPU work (a 1000-iteration u64 xorshift mixer) so the
//!   comparison measures dispatch + steal latency, not workload
//!   variance.
//! - **Same N**: 8 closures for the canonical "one per physical
//!   core" SIMC case on the development host (Zen+ R7 2700:
//!   8 physical / 16 logical).
//! - **Same result-collection**: both halves materialise a Vec<u64>
//!   in caller order so the bench measures equivalent total work
//!   including the result-gather phase.
//! - **The primitive's named feature is exercised**:
//!   cooperative_join_n_flat's mailbox-distribute path fires at
//!   N >= the worker count. The bench calls cooperative_join_n_flat
//!   directly; its
//!   internal fan_out_external path wraps in a parent StackJob
//!   submitted onto the global arena so the inner fan_out_in_worker
//!   call runs on a Flynnel worker thread (per current_worker_ctx).
//!
//! ## Every size is registered twice, in opposite arm order
//!
//! Criterion measures arms sequentially, so a load that arrives or
//! departs partway through a group shifts the means of the arms it
//! covers and not the others. Confidence intervals do not defend
//! against this: they describe spread within an arm, so two arms can
//! have tight disjoint intervals and still be separated by the machine
//! rather than by the code.
//!
//! A ratio that holds in both orders is the code. One that appears in
//! one order and is absent or reversed in the other is the host, and
//! the group is unreadable rather than merely noisy. This is what lets
//! the sweep be read on a shared host, which is the only kind
//! available here.

#![allow(clippy::missing_docs_in_private_items)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::sched::cooperative::{
    cooperative_join_n, cooperative_join_n_flat, cooperative_join_n_flat_mailbox,
};
use flynnel::sched::plan::JobPlan;

/// Closure body: deterministic, fixed-cost CPU work. Each call
/// runs a 1000-iter xorshift mixer + returns the final value so
/// the optimizer can't elide the loop.
#[inline(never)]
fn fixed_cost_work(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..1000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x = x.wrapping_mul(0x100000001B3);
    }
    x
}

/// One fan-out shape.
///
/// `Deque` pushes N-1 closures onto the calling worker's local deque
/// and lets random-victim peer-steal distribute them. `Mailbox` pushes
/// each closure to one worker's mailbox, behind a gate comparing N
/// against the worker count: below it the fan-out demotes to the
/// shared deque, so `Deque` and `Mailbox` run the same code there.
/// `Routed` is the entry point a caller uses, whose `Auto` arm picks
/// between the tree and mailbox variants. `Rayon` is the closest
/// equivalent outside the crate.
#[derive(Copy, Clone)]
enum Arm {
    Deque,
    Mailbox,
    Routed,
    Rayon,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Deque => "flynnel_baseline_deque",
            Self::Mailbox => "flynnel_mailbox",
            Self::Routed => "flynnel_routed",
            Self::Rayon => "rayon_scope_spawn",
        }
    }
}

/// Build the closure set one iteration fans out.
///
/// Rebuilt per iteration because each variant consumes it, so the
/// allocation is inside every arm's timed region and not only some.
fn closures(n: usize) -> Vec<Box<dyn FnOnce() -> u64 + Send>> {
    (0..n)
        .map(|i| {
            let b: Box<dyn FnOnce() -> u64 + Send> = Box::new(move || fixed_cost_work(i as u64));
            b
        })
        .collect()
}

/// Fan out `n` closures through one shape and materialize the results.
///
/// Every arm produces a `Vec<u64>` in caller order, so the timed region
/// covers the result-gather phase equally.
fn run_arm(arm: Arm, plan: &JobPlan, n: usize) {
    match arm {
        Arm::Deque => {
            black_box(cooperative_join_n_flat::<u64>(plan, closures(n)));
        }
        Arm::Mailbox => {
            black_box(cooperative_join_n_flat_mailbox::<u64>(plan, closures(n)));
        }
        Arm::Routed => {
            black_box(cooperative_join_n::<u64>(plan, closures(n)));
        }
        Arm::Rayon => {
            let results: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(vec![0u64; n]);
            rayon::scope(|s| {
                for i in 0..n {
                    let results_ref = &results;
                    s.spawn(move |_| {
                        let r = fixed_cost_work(i as u64);
                        results_ref.lock().unwrap()[i] = r;
                    });
                }
            });
            black_box(results.into_inner().unwrap());
        }
    }
}

/// Register one group's arms at `n_closures`, in the order given.
fn register(c: &mut Criterion, group_name: String, n_closures: usize, arms: &[Arm]) {
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    let plan = JobPlan::new(2, 1);

    for arm in arms {
        group.bench_function(arm.name(), |b| {
            b.iter(|| run_arm(*arm, &plan, n_closures));
        });
    }

    group.finish();
}

/// Run the 4-way A/B at a specific N, forward and reversed.
fn bench_n(c: &mut Criterion, n_closures: usize) {
    const FWD: [Arm; 4] = [Arm::Deque, Arm::Mailbox, Arm::Routed, Arm::Rayon];
    const REV: [Arm; 4] = [Arm::Rayon, Arm::Routed, Arm::Mailbox, Arm::Deque];
    register(c, format!("simc_cooperative_n{n_closures}"), n_closures, &FWD);
    register(c, format!("simc_cooperative_n{n_closures}_rev"), n_closures, &REV);
}

fn bench_simc_cooperative(c: &mut Criterion) {
    // The sweep straddles the gate, which sits at the worker count.
    // Below it both flynnel shape arms run the same deque path.
    bench_n(c, 8);
    bench_n(c, 12);
    bench_n(c, 16);
    // At or above the worker count on hosts up to twenty-four, where
    // the mailbox arm reaches owner-directed distribution. A wider
    // pool moves the crossing up and these sizes back below it.
    bench_n(c, 24);
    bench_n(c, 32);
    bench_n(c, 56);
    bench_n(c, 64);
    // Past the widths where mailbox has been measured losing to deque
    // by 10 to 22 percent. The design's own argument is that
    // owner-directed placement eventually beats random peer-steal, so
    // the gate belongs wherever that crossing is rather than at the
    // worker count; these are what say whether there is one.
    bench_n(c, 128);
    bench_n(c, 256);
}

criterion_group!(benches, bench_simc_cooperative);
criterion_main!(benches);

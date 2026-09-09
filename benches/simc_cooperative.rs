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

/// Run the 4-way A/B at a specific N (closure count).
///
/// Which path the mailbox arm takes depends on the gate inside
/// [`cooperative_join_n_flat_mailbox`], which compares N against the
/// worker count. At or above it each closure is pushed to one
/// worker's mailbox; below it the fan-out demotes to the shared
/// deque, so both flynnel shape arms run the same code there.
fn bench_n(c: &mut Criterion, n_closures: usize) {
    let mut group = c.benchmark_group(format!("simc_cooperative_n{n_closures}"));
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    let plan = JobPlan::new(2, 1);

    // ---- flynnel cooperative_join_n_flat (BASELINE: deque) ----
    // Existing behavior: N-1 closures pushed onto the calling
    // worker's local deque (Public tier). Peer-steal via random
    // victim selection distributes them across workers.
    group.bench_function("flynnel_baseline_deque", |b| {
        b.iter(|| {
            let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..n_closures)
                .map(|i| {
                    let b: Box<dyn FnOnce() -> u64 + Send> =
                        Box::new(move || fixed_cost_work(i as u64));
                    b
                })
                .collect();
            let results = cooperative_join_n_flat::<u64>(&plan, closures);
            black_box(results);
        });
    });

    // ---- flynnel cooperative_join_n_flat_mailbox ----
    // SIMC owner-directed distribution behind the architectural gate.
    // The gate demotes to deque mode below the worker count, where a
    // fan-out narrower than the pool would leave the parent and every
    // untargeted worker idle for the wait: peer-steal reads deques,
    // so it cannot reach a closure sitting in someone's mailbox.
    group.bench_function("flynnel_mailbox", |b| {
        b.iter(|| {
            let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..n_closures)
                .map(|i| {
                    let b: Box<dyn FnOnce() -> u64 + Send> =
                        Box::new(move || fixed_cost_work(i as u64));
                    b
                })
                .collect();
            let results = cooperative_join_n_flat_mailbox::<u64>(&plan, closures);
            black_box(results);
        });
    });

    // ---- flynnel cooperative_join_n, the shape the routing picks ----
    // The entry point a caller uses. Its Auto arm routes below
    // `total_workers` to the tree variant and at or above it to the
    // mailbox variant, which then applies its own gate. On a 24-worker
    // host N=8 and N=16 reach the tree variant, which neither of the
    // arms above measures.
    group.bench_function("flynnel_routed", |b| {
        b.iter(|| {
            let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..n_closures)
                .map(|i| {
                    let b: Box<dyn FnOnce() -> u64 + Send> =
                        Box::new(move || fixed_cost_work(i as u64));
                    b
                })
                .collect();
            let results = cooperative_join_n::<u64>(&plan, closures);
            black_box(results);
        });
    });

    // ---- rayon scope::spawn equivalent ----
    group.bench_function("rayon_scope_spawn", |b| {
        b.iter(|| {
            let results: std::sync::Mutex<Vec<u64>> =
                std::sync::Mutex::new(vec![0u64; n_closures]);
            rayon::scope(|s| {
                for i in 0..n_closures {
                    let results_ref = &results;
                    s.spawn(move |_| {
                        let r = fixed_cost_work(i as u64);
                        results_ref.lock().unwrap()[i] = r;
                    });
                }
            });
            black_box(results.into_inner().unwrap());
        });
    });

    group.finish();
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
}

criterion_group!(benches, bench_simc_cooperative);
criterion_main!(benches);

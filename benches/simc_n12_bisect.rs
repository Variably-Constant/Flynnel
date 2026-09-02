//! SIMC N=12 cooperative-path bisect bench.
//!
//! ## Why this bench exists
//!
//! [`benches/flynn_axes.rs::bench_simc`] benches the SIMC axis as
//! `cooperative_join_n(plan, closures)` vs `rayon::scope` +
//! `Arc<Mutex<Vec<_>>>` with `n_closures = 2 * physical_cores`. On
//! the 12-logical / 6-physical Xeon Cascade Lake Colab host that
//! lands at N = 12 which matches the host's total worker count
//! (primaries + SMT extensions), and on the 16-logical / 8-physical
//! Zen+ R7 2700 host at N = 12 < 16 workers (under-populated).
//!
//! [`flynnel::sched::cooperative_join_n`] picks its dispatch shape
//! against the host worker pool: `N < n_workers` routes through
//! `cooperative_join_n_tree`, `N >= n_workers` routes through
//! `cooperative_join_n_flat_mailbox`. The bisect bench keeps
//! `n_closures = 12` fixed and the body identical to
//! `flynn_axes::bench_simc` (a 2_000_000-iter wrapping_mul +
//! xorshift mixer on a u64) so the numbers here are byte-comparable
//! against the headline `simc_nway_fork_12_closures` row in the
//! Flynn-axes bench output. The four flynnel variants exercised:
//!
//! - `flynnel_adaptive` - calls `cooperative_join_n` (the production
//!   surface; picks per host: tree on Zen+ at N=12 < 16 workers,
//!   mailbox on Xeon at N=12 == 12 workers).
//! - `flynnel_tree` - calls `cooperative_join_n_tree` directly.
//! - `flynnel_flat_deque` - calls `cooperative_join_n_flat` directly
//!   (the raw deque fan-out; baseline for the mailbox comparison).
//! - `flynnel_flat_mailbox` - calls `cooperative_join_n_flat_mailbox`
//!   (owner-directed distribution via per-worker mailboxes; the SIMC
//!   primitive's structural fit per `cooperative.rs`'s module doc).
//!
//! Plus the rayon baseline (`rayon::scope` + `Arc<Mutex<Vec>>`)
//! reproduced byte-equivalent to `flynn_axes::bench_simc`'s
//! `rayon_scope_mutex_vec` contender so the comparison is apples-
//! to-apples.
//!
//! ## Bench-audit (HARD RULE 3)
//!
//! - **Does the bench invoke the primitive's named feature?** Yes -
//!   each variant calls its own cooperative entry point with the
//!   same closure list. The mailbox variant's owner-directed
//!   distribution path fires when N >= 3, matched here.
//! - **Does it impose surplus locks / allocs / indirection vs the
//!   baseline?** No - rayon's row uses `Arc<Mutex<Vec<Option<u64>>>>`
//!   because that is the literal pattern in `flynn_axes::bench_simc`;
//!   flynnel's four variants own their result-collection via
//!   `Vec<u64>` return because that is how the surface ships. The
//!   per-call allocation cost (one `Vec` build + one `Box::new` per
//!   closure) is shared by every flynnel variant identically, so the
//!   inter-flynnel comparison stays clean.
//! - **Is the primitive sized / configured for the workload?** Yes -
//!   `JobPlan::new(8, 1024)` matches the plan `flynn_axes::bench_simc`
//!   constructs, and the body is `#[inline(never)]` so the closure
//!   does not specialize differently per call site.

#![allow(clippy::missing_docs_in_private_items)]

use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::sched::cooperative::cooperative_join_n_flat_mailbox;
use flynnel::sched::{
    JobPlan, cooperative_join_n, cooperative_join_n_flat, cooperative_join_n_tree,
};

/// Per-closure body. Identical to `flynn_axes::bench_simc::body`:
/// 2 000 000 iterations of `wrapping_mul + xorshift` on a u64. The
/// inline-never attribute prevents the optimizer from specializing
/// the body per call site.
#[inline(never)]
fn body(seed: u64) -> u64 {
    let mut v: u64 = seed | 1;
    for _ in 0..2_000_000u64 {
        v = v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        v ^= v >> 31;
    }
    v
}

const N_CLOSURES: usize = 12;

fn build_closures() -> Vec<Box<dyn FnOnce() -> u64 + Send>> {
    (0..N_CLOSURES)
        .map(|i| {
            let b: Box<dyn FnOnce() -> u64 + Send> = Box::new(move || body(i as u64));
            b
        })
        .collect()
}

fn bench_simc_n12_bisect(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("simc_n{N_CLOSURES}_cooperative_bisect"));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    let plan = JobPlan::new(8, 1024);

    // Adaptive: production surface. At N = 12 this routes to
    // `_flat` (cooperative.rs FLAT_CROSSOVER_N == 12).
    group.bench_function("flynnel_adaptive", |b| {
        b.iter(|| {
            let results = cooperative_join_n::<u64>(&plan, build_closures());
            black_box(results);
        });
    });

    // Tree: balanced binary bisect, depth ceil(log2(12)) = 4.
    group.bench_function("flynnel_tree", |b| {
        b.iter(|| {
            let results = cooperative_join_n_tree::<u64>(&plan, build_closures());
            black_box(results);
        });
    });

    // Flat deque: all 12 closures fanned onto the dispatching
    // worker's local Chase-Lev deque, peer-stolen by other workers.
    group.bench_function("flynnel_flat_deque", |b| {
        b.iter(|| {
            let results = cooperative_join_n_flat::<u64>(&plan, build_closures());
            black_box(results);
        });
    });

    // Flat mailbox: owner-directed distribution; each closure
    // pushed to a specific peer's mailbox, drained mailbox-first.
    group.bench_function("flynnel_flat_mailbox", |b| {
        b.iter(|| {
            let results = cooperative_join_n_flat_mailbox::<u64>(&plan, build_closures());
            black_box(results);
        });
    });

    // Rayon baseline, byte-equivalent to
    // `flynn_axes::bench_simc::rayon_scope_mutex_vec`.
    group.bench_function("rayon_scope_mutex_vec", |b| {
        b.iter(|| {
            let results: Arc<Mutex<Vec<Option<u64>>>> =
                Arc::new(Mutex::new(vec![None; N_CLOSURES]));
            rayon::scope(|s| {
                for i in 0..N_CLOSURES {
                    let r = Arc::clone(&results);
                    s.spawn(move |_| {
                        let val = body(i as u64);
                        r.lock().unwrap()[i] = Some(val);
                    });
                }
            });
            let final_vec: Vec<u64> = results
                .lock()
                .unwrap()
                .iter()
                .map(|x| x.unwrap())
                .collect();
            black_box(final_vec);
        });
    });

    group.finish();
}

criterion_group!(simc_n12_bisect, bench_simc_n12_bisect);
criterion_main!(simc_n12_bisect);

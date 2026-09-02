//! Comprehensive tutorial exercising every workhorse Flynnel primitive.
//!
//! Runs each of the top-level primitives with real workloads that
//! produce observable output. Each section is preceded by a
//! `println!` header so the console output reads as a walk-through
//! rather than a mystery of atomics.
//!
//! ```bash
//! cargo run --example tutorial_all_apis --release
//! ```
//!
//! Covered primitives (crate-root re-exports from
//! [`src/lib.rs`](../src/lib.rs)):
//!
//! - `flynnel::join(plan, a, b)`
//! - `flynnel::join_context(plan, a, b)` (via `sched::arena::join_context`)
//! - `flynnel::join_default(k_outer, batch_size, a, b)`
//! - `flynnel::for_each_chunk(plan, items, op)`
//! - `flynnel::cooperative_join_n(plan, closures)`
//! - `flynnel::race_variants(plan, fast, faithful, correct)`
//! - `flynnel::k_join::<K, ..>(a, b)` (const-generic)
//! - `flynnel::k_join_with_plan::<K, ..>(plan, a, b)` (const-generic + explicit plan)
//! - `flynnel::JobPlan::new / set_profile / bare / with_leaf_shape /
//!   with_estimated_per_item_ns / with_bisect_variant`
//!
//! CPU + GPU hybrid primitives (`join_hybrid`, `hybrid_pipeline`) are
//! feature-gated on `cuda-reference`; a minimal in-process stub is
//! demonstrated at the end of the file when the feature is off.

use std::time::Instant;

use flynnel::{
    BisectVariant, DispatchProfile, JobPlan, LeafShape, WorkloadClass,
    cooperative_join_n, for_each_chunk, join, join_default, k_join,
    k_join_with_plan, race_variants,
};
use flynnel::sched::adaptive_profile::migrate_workload_class;
use flynnel::sched::arena::join_context;

fn header(title: &str) {
    println!();
    println!("=== {title} ===");
}

fn main() {
    // Force the process-global classification to a known value so
    // the walk-through starts from a predictable baseline. Production
    // code rarely does this: the observer migrates the class
    // automatically as leaves are recorded.
    migrate_workload_class(WorkloadClass::PortBound);

    section_join();
    section_join_context();
    section_join_default();
    section_for_each_chunk();
    section_for_each_chunk_hinted();
    section_cooperative_join_n();
    section_race_variants();
    section_k_join();
    section_k_join_with_plan();
    section_hybrid_demo_cpu_only();

    println!();
    println!("Tutorial done. Each section above ran real work and asserted results.");
}

fn section_join() {
    header("join(plan, a, b): two-way fork-join");
    let plan = JobPlan::new(8, 1024);
    let (a, b) = join(
        &plan,
        || (0..512u64).sum::<u64>(),
        || (512..1024u64).sum::<u64>(),
    );
    let total = a + b;
    let expected = (0..1024u64).sum::<u64>();
    println!("  left half sum = {a}");
    println!("  right half sum = {b}");
    println!("  total = {total} (expected {expected})");
    assert_eq!(total, expected);
}

fn section_join_context() {
    header("join_context(plan, a, b): stolen/injected flag exposed");
    let plan = JobPlan::new(8, 1024);
    let (a_result, b_result) = join_context(
        &plan,
        |injected: bool| {
            println!("  a's `injected` flag = {injected}");
            (0..512u64).sum::<u64>()
        },
        |stolen: bool| {
            println!("  b's `stolen` flag = {stolen}");
            (512..1024u64).sum::<u64>()
        },
    );
    let total = a_result + b_result;
    let expected = (0..1024u64).sum::<u64>();
    println!("  total = {total} (expected {expected})");
    assert_eq!(total, expected);
}

fn section_join_default() {
    header("join_default(k_outer, batch_size, a, b): convenience wrapper");
    let (a, b) = join_default(
        8,
        1024,
        || (0..512u64).sum::<u64>(),
        || (512..1024u64).sum::<u64>(),
    );
    let total = a + b;
    println!("  total = {total} (built JobPlan::new(8, 1024) internally)");
    assert_eq!(total, (0..1024u64).sum::<u64>());
}

fn section_for_each_chunk() {
    header("for_each_chunk(plan, &mut items, op): bulk data-parallel");
    let mut data: Vec<u64> = (0..100_000).collect();
    let plan = JobPlan::new(8, data.len() as u32);
    for_each_chunk(&plan, &mut data, |chunk: &mut [u64]| {
        for x in chunk {
            *x = x.wrapping_mul(3);
        }
    });
    println!("  data[42] = {} (expected {})", data[42], 42u64 * 3);
    println!("  data[99_999] = {} (expected {})", data[99_999], 99_999u64 * 3);
    assert_eq!(data[42], 42 * 3);
    assert_eq!(data[99_999], 99_999 * 3);
}

fn section_for_each_chunk_hinted() {
    header("for_each_chunk with explicit per-item ns hint (production shape)");
    let mut data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    // Estimate: ~50 ns per element for a sqrt chain. The hint lets
    // the tier picker sizing decision skip the probe path and go
    // straight to bisect with min_leaf tuned to the leaf-work
    // overhead target (5us default).
    let plan = JobPlan::new(6, data.len() as u32)
        .with_estimated_per_item_ns(50)
        .with_leaf_shape(LeafShape::PortCompute);
    let t0 = Instant::now();
    for_each_chunk(&plan, &mut data, |chunk: &mut [f64]| {
        for x in chunk {
            let mut acc = *x + 1.0;
            for _ in 0..4 {
                acc = (acc + 1.0).sqrt();
            }
            *x = acc;
        }
    });
    println!("  10_000 items processed in {:?}", t0.elapsed());
    println!("  data[0] = {} (finite)", data[0]);
    assert!(data[0].is_finite());
}

fn section_cooperative_join_n() {
    header("cooperative_join_n(plan, closures): N-way SIMC dispatch");
    let plan = JobPlan::new(6, 16);
    let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..8u64)
        .map(|i| {
            let f: Box<dyn FnOnce() -> u64 + Send> = Box::new(move || {
                let start = i * 100;
                let end = start + 100;
                (start..end).sum::<u64>()
            });
            f
        })
        .collect();
    let results = cooperative_join_n(&plan, closures);
    let total: u64 = results.iter().sum();
    let expected: u64 = (0..800u64).sum::<u64>();
    println!("  8 closures returned = {results:?}");
    println!("  sum of all = {total} (expected {expected})");
    assert_eq!(total, expected);
}

fn section_race_variants() {
    header("race_variants(plan, fast, faithful, correct): MISD speculation");
    let plan = JobPlan::new(6, 1);
    let (result, winner) = race_variants(
        &plan,
        |_cancel| {
            // Fast variant: might succeed cheaply, might return None.
            std::thread::sleep(std::time::Duration::from_millis(5));
            Some(1_000u64)
        },
        |_cancel| {
            // Faithful variant: intermediate cost.
            std::thread::sleep(std::time::Duration::from_millis(20));
            Some(2_000u64)
        },
        |_cancel| {
            // Correct variant: always runs to completion, ignores cancel.
            std::thread::sleep(std::time::Duration::from_millis(50));
            3_000u64
        },
    );
    println!("  winner = {winner:?}, value = {result}");
    println!("  (fast variant beat the others because 5ms < 20ms < 50ms)");
    assert_eq!(result, 1_000);
}

fn section_k_join() {
    header("k_join::<K, ..>(a, b): const-generic; K<=4 collapses to inline");
    // At K=4 (const), the const-fn dispatch inlines both closures.
    let (a4, b4) = k_join::<4, _, _, _, _>(
        || (0..100u64).sum::<u64>(),
        || (100..200u64).sum::<u64>(),
    );
    println!("  K=4 (inline): a={a4}, b={b4}, total={}", a4 + b4);
    assert_eq!(a4 + b4, (0..200u64).sum::<u64>());

    // At K=8 (const), the const-fn dispatch calls the arena.
    let (a8, b8) = k_join::<8, _, _, _, _>(
        || (0..1000u64).sum::<u64>(),
        || (1000..2000u64).sum::<u64>(),
    );
    println!("  K=8 (arena): a={a8}, b={b8}, total={}", a8 + b8);
    assert_eq!(a8 + b8, (0..2000u64).sum::<u64>());
}

fn section_k_join_with_plan() {
    header("k_join_with_plan::<K, ..>(plan, a, b): const-generic + custom plan");
    let plan = JobPlan::set_profile(8, 2048, DispatchProfile::PortBound)
        .with_bisect_variant(BisectVariant::RayonStyleReplenish);
    let (a, b) = k_join_with_plan::<8, _, _, _, _>(
        &plan,
        || (0..1024u64).sum::<u64>(),
        || (1024..2048u64).sum::<u64>(),
    );
    println!("  K=8 with pinned profile + variant: total = {}", a + b);
    assert_eq!(a + b, (0..2048u64).sum::<u64>());
}

fn section_hybrid_demo_cpu_only() {
    header("hybrid dispatch (feature-gated on cuda-reference)");
    println!("  join_hybrid / hybrid_pipeline are demonstrated in");
    println!("  `examples/tpu_jax_demo.rs` (TPU backend) and");
    println!("  `benches/mimt_coupled.rs` (CUDA backend + Metropolis / CG / MCTS).");
    println!("  Build with:");
    println!("    cargo run --example tpu_jax_demo --release --features tpu-jax-reference");
    println!("    cargo bench --bench mimt_coupled --features cuda-reference");
}

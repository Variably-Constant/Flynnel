//! Manual walk-through of `AdaptiveDispatcher` used for each Flynn axis.
//!
//! Companion to `wiki/content/docs/explanation/Extended-Flynn-Taxonomy.md`
//! (section: "Using each axis through `AdaptiveDispatcher`"). Every
//! block in that doc has a matching section here.
//!
//! ```bash
//! cargo run --example dispatcher_per_axis --release
//! ```
//!
//! Each section prints a header, runs real work through the
//! dispatcher, and asserts the result. If the run finishes with
//! "Dispatcher-per-axis walk-through done." you have verified the
//! `AdaptiveDispatcher` surface for every axis that carries an
//! `execute_*` method.

use flynnel::sched::dispatch::AdaptiveDispatcher;
use flynnel::sched::workload_shape::WorkloadShape;

fn header(title: &str) {
    println!();
    println!("=== {title} ===");
}

fn main() {
    section_sisd_streaming();
    section_mimd_for_each();
    section_simc_cooperative();
    section_simc_cooperative_mailbox();
    section_mimc_heterogeneous();
    section_simt_indexed_backend_adaptive();
    section_dispatcher_observation_and_migration();

    println!();
    println!("Dispatcher-per-axis walk-through done. All assertions passed.");
    println!("MISD (race_variants) and MIMT (join_hybrid, hybrid_pipeline) are");
    println!("exercised in `examples/tutorial_all_apis.rs` because their type");
    println!("signatures are best expressed via the direct primitives.");
}

fn section_sisd_streaming() {
    header("SISD: execute_streaming (runs closure on caller thread)");
    let sum: u64 = AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::Streaming)
        .execute_streaming(|| (0..1_000u64).sum());
    let expected = (0..1_000u64).sum::<u64>();
    println!("  sum = {sum} (expected {expected})");
    assert_eq!(sum, expected);
}

fn section_mimd_for_each() {
    header("MIMD: execute_for_each (bulk data-parallel via for_each_chunk)");
    let mut data: Vec<u64> = (0..100_000).collect();
    AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::WorkSteal {
            n_consumers: 8,
            batch_size: data.len() as u32,
        })
        .with_k_outer(8)
        .execute_for_each(&mut data, |chunk: &mut [u64]| {
            for x in chunk {
                *x = x.wrapping_mul(3);
            }
        });
    println!("  data[0] = {} (expected 0)", data[0]);
    println!("  data[42] = {} (expected {})", data[42], 42u64 * 3);
    println!(
        "  data[99_999] = {} (expected {})",
        data[99_999],
        99_999u64 * 3
    );
    assert_eq!(data[0], 0);
    assert_eq!(data[42], 42 * 3);
    assert_eq!(data[99_999], 99_999 * 3);
}

fn section_simc_cooperative() {
    header("SIMC: execute_cooperative (N-way flat fan-out)");
    let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..8u64)
        .map(|i| {
            let f: Box<dyn FnOnce() -> u64 + Send> = Box::new(move || {
                let start = i * 1_000;
                (start..start + 1_000).sum::<u64>()
            });
            f
        })
        .collect();
    let results: Vec<u64> = AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::Cooperative { n_cores: 8 })
        .execute_cooperative(closures);
    let total: u64 = results.iter().sum();
    let expected: u64 = (0..8_000u64).sum::<u64>();
    println!("  8 closures returned = {results:?}");
    println!("  total = {total} (expected {expected})");
    assert_eq!(total, expected);
}

fn section_simc_cooperative_mailbox() {
    header("SIMC (mailbox): execute_cooperative_mailbox (URD owner-directed)");
    let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..8u64)
        .map(|i| {
            let f: Box<dyn FnOnce() -> u64 + Send> = Box::new(move || {
                let start = i * 100;
                (start..start + 100).sum::<u64>()
            });
            f
        })
        .collect();
    let results: Vec<u64> = AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::Cooperative { n_cores: 8 })
        .execute_cooperative_mailbox(closures);
    let total: u64 = results.iter().sum();
    let expected: u64 = (0..800u64).sum::<u64>();
    println!("  8 closures via mailbox routing = {results:?}");
    println!("  total = {total} (expected {expected})");
    assert_eq!(total, expected);
}

fn section_mimc_heterogeneous() {
    header("MIMC: execute_cooperative with heterogeneous closures");
    // 4 closures with DIFFERENT bodies (heterogeneity is intrinsic;
    // the dispatcher does not know or care about it).
    let mut closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = Vec::new();
    closures.push(Box::new(|| {
        // Role A: sum of first 100 integers.
        (0..100u64).sum()
    }));
    closures.push(Box::new(|| {
        // Role B: sum of first 100 squares.
        (0..100u64).map(|x| x * x).sum()
    }));
    closures.push(Box::new(|| {
        // Role C: sum of first 100 cubes.
        (0..100u64).map(|x| x * x * x).sum()
    }));
    closures.push(Box::new(|| {
        // Role D: max of first 100 sqrt-chained values.
        let mut best = 0u64;
        for i in 0..100u64 {
            let v = (i as f64).sqrt().to_bits();
            if v > best {
                best = v;
            }
        }
        best
    }));
    let results: Vec<u64> = AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::Cooperative { n_cores: 4 })
        .execute_cooperative(closures);
    println!("  role A (sum) = {}", results[0]);
    println!("  role B (sum of squares) = {}", results[1]);
    println!("  role C (sum of cubes) = {}", results[2]);
    println!("  role D (max sqrt bits) = {}", results[3]);
    // Verify order preservation: results[i] is closure[i]'s return.
    assert_eq!(results[0], (0..100u64).sum::<u64>());
    assert_eq!(results[1], (0..100u64).map(|x| x * x).sum::<u64>());
    assert_eq!(results[2], (0..100u64).map(|x| x * x * x).sum::<u64>());
    assert!(results[3] > 0);
}

fn section_simt_indexed_backend_adaptive() {
    header("SIMT: execute_indexed (backend-adaptive parallel-for)");
    // Sums via an atomic accumulator so we can observe every index
    // actually got called. In production the work closure is where
    // the real per-index work lives (kernel launch on GPU, chunk
    // work on CPU).
    use std::sync::atomic::{AtomicU64, Ordering};
    let acc = AtomicU64::new(0);
    let dispatcher = AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::WorkSteal {
            n_consumers: 8,
            batch_size: 1024,
        });
    // No .migrate_backend() call means the active backend stays
    // CPU (the always-registered default). To route to CUDA:
    //   dispatcher.migrate_backend(flynnel::Backend::Cuda { device_id: 0 });
    let fell_back = dispatcher.execute_indexed(1024, |i: u32| {
        acc.fetch_add(i as u64, Ordering::Relaxed);
    });
    let total = acc.load(Ordering::Relaxed);
    let expected: u64 = (0..1024u64).sum();
    println!("  1024 items dispatched; fell_back_to_cpu = {fell_back}");
    println!("  sum of indices = {total} (expected {expected})");
    assert_eq!(total, expected);
}

fn section_dispatcher_observation_and_migration() {
    header("Migration surface: k_gating / dispatch_profile / backend");
    let d = AdaptiveDispatcher::new();
    let active_profile = d.active_dispatch_profile();
    let active_backend = d.active_backend_id();
    let (_backend, fell_back) = d.resolve_active_backend();
    println!("  active dispatch profile = {active_profile:?}");
    println!("  active backend id = {active_backend:?}");
    println!("  resolve_active_backend fell_back = {fell_back}");
    // Runtime migration is a single AtomicU8 Release-store; subsequent
    // dispatches read the new value on the next plan-construction.
    d.migrate_workload_class(flynnel::WorkloadClass::PortBound);
    println!(
        "  post migration: active dispatch profile = {:?}",
        d.active_dispatch_profile()
    );
}

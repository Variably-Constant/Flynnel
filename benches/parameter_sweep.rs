//! Parameter sweep across (N, per_item_ns) covering the small-N +
//! heavy-per-item corner of the parameter space (the region where
//! commits d63aac8 + a054d9b fixed the bisect MIN_LEAF floor and
//! the probe-and-decide tail-consumption after external users
//! reported serial-shaped runs at parallelizable workloads).
//!
//! This sweep covers:
//!   N           = 1, 2, 4, 8, 16, 32, 64, 128, 1024, 10000
//!   per_item_ns = ~10ns, ~1us, ~100us, ~10ms  (sqrt-chain compute)
//!   plan        = default, with_estimated_per_item_ns
//!
//! Per cell: serial / rayon / flynnel_default / flynnel_hinted

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use flynnel::sched::par_iter::for_each_chunk;
use flynnel::JobPlan;
use rayon::prelude::*;

/// Per-item compute load. Tunes the chained-sqrt-count to produce
/// roughly the target nanoseconds on x86_64 (Zen3 ~4 GHz: ~12ns/sqrt).
/// Returns the FLOAT result so the compiler can not elide the chain.
#[inline(never)]
fn sqrt_chain(seed: f64, iters: u32) -> f64 {
    let mut x = seed;
    for _ in 0..iters {
        x = (x + 1.0).sqrt();
    }
    x
}

/// Workload profile: (label, sqrt_iters_to_reach_target_ns, nominal_ns).
/// Calibrated for Zen3 ~4 GHz where one sqrt ~= 12 ns.
struct WorkloadProfile {
    label: &'static str,
    sqrt_iters: u32,
    nominal_ns: u32,
}

const PROFILES: &[WorkloadProfile] = &[
    WorkloadProfile { label: "10ns",   sqrt_iters: 1,        nominal_ns: 12 },
    WorkloadProfile { label: "1us",    sqrt_iters: 83,       nominal_ns: 1_000 },
    WorkloadProfile { label: "100us",  sqrt_iters: 8_333,    nominal_ns: 100_000 },
    WorkloadProfile { label: "10ms",   sqrt_iters: 833_333,  nominal_ns: 10_000_000 },
];

const SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 1024, 10_000];

fn bench_sweep(c: &mut Criterion) {
    for prof in PROFILES {
        for &n in SIZES {
            // Skip cells where total work would be over ~5 seconds (criterion would
            // run too few iterations for stable timing).
            let total_ns: u64 = (prof.nominal_ns as u64) * (n as u64);
            if total_ns > 5_000_000_000 {
                continue;
            }
            let mut group = c.benchmark_group(format!("sweep_{}_{}items", prof.label, n));
            group.sample_size(15);
            // Larger workloads need longer measurement windows.
            let meas_ms = (total_ns / 1_000_000).clamp(50, 5_000);
            group.measurement_time(std::time::Duration::from_millis(meas_ms * 2));
            group.warm_up_time(std::time::Duration::from_millis(meas_ms));

            let template: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
            let iters = prof.sqrt_iters;

            // 1. serial baseline
            group.bench_function("serial", |b| {
                b.iter_batched_ref(
                    || template.clone(),
                    |v| {
                        for x in v.iter_mut() {
                            *x = sqrt_chain(*x, iters);
                        }
                    },
                    BatchSize::LargeInput,
                );
            });

            // 2. rayon par_iter
            group.bench_function("rayon", |b| {
                b.iter_batched_ref(
                    || template.clone(),
                    |v| {
                        v.par_iter_mut().for_each(|x| {
                            *x = sqrt_chain(*x, iters);
                        });
                    },
                    BatchSize::LargeInput,
                );
            });

            // 3. flynnel default (no per-item hint)
            group.bench_function("flynnel_default", |b| {
                b.iter_batched_ref(
                    || template.clone(),
                    |v| {
                        let plan = JobPlan::new(6, n as u32);
                        for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [f64]| {
                            for x in slice {
                                *x = sqrt_chain(*x, iters);
                            }
                        });
                    },
                    BatchSize::LargeInput,
                );
            });

            // 4. flynnel with explicit per-item-ns hint (matches the
            //    workload actual per-item cost so the adaptive
            //    min_leaf computation has authoritative input)
            group.bench_function("flynnel_hinted", |b| {
                b.iter_batched_ref(
                    || template.clone(),
                    |v| {
                        let plan = JobPlan::new(6, n as u32)
                            .with_estimated_per_item_ns(prof.nominal_ns);
                        for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [f64]| {
                            for x in slice {
                                *x = sqrt_chain(*x, iters);
                            }
                        });
                    },
                    BatchSize::LargeInput,
                );
            });
            group.finish();
        }
    }
}

criterion_group!(benches, bench_sweep);
criterion_main!(benches);

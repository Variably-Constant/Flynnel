//! Criterion bench for the three scheduler-decision regimes.
//!
//! The bench exists to demonstrate Flynnel's three distinct dispatch
//! decisions, each measured at N=10k and N=100k:
//!
//! - **Fine-Grain** (~20 ns/elem, 3 sqrt iters): aggregate ~200 us at
//!   N=10k, well under the parallel dispatch floor. The tier picker
//!   collapses to inline serial execution (no fork, no steal, no latch
//!   wait); the calling thread runs the whole slice. Rayon's
//!   `par_iter_mut` unconditionally pays the parallel dispatch cost
//!   even though the workload is too small to amortize it. Headline
//!   inline-collapse demonstration.
//! - **Latency-Bound** (~600 ns/elem, 100 sqrt iters chained): long
//!   FP dependency chain; SMT siblings hide pipeline stalls. Aggregate
//!   ~6 ms / ~60 ms; parallel dispatch IS justified at both sizes,
//!   and `DispatchProfile::LatencyBound` (SMT-on, oversubscribe 4x) is
//!   the canonical winning shape.
//! - **Port-Bound** (~12-15 ns/elem, chained u128 mul): saturates the
//!   single IMUL execution port; SMT siblings compete for the same
//!   port and hurt throughput. Aggregate ~150 us / ~1.5 ms; sits near
//!   the dispatch crossover, and `DispatchProfile::PortBound` (SMT-off,
//!   oversubscribe 2x) is the canonical winning shape.
//!
//! Each parallel workload also includes a `set_profile(LatencyBound)` /
//! `set_profile(PortBound)` plan that shows how `JobPlan::set_profile`
//! toggles SMT, cost estimate, and oversubscription together based on
//! the `DispatchProfile` classification.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench inline_collapse
//! ```
//!
//! Reports land under `target/criterion/<group>/<bench>/report/index.html`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use flynnel::sched::par_iter::for_each_chunk;
use flynnel::{DispatchProfile, BisectVariant, JobPlan};

// ===========================================================================
// Op definitions: one canonical op per scheduler-decision regime.
// ===========================================================================

/// Latency-bound op: 100-deep sqrt dependency chain. ~600 ns per
/// element. Each sqrt has ~14-cycle latency on Zen / Skylake; the
/// chain is strictly sequential so the FP pipeline stalls between
/// sqrts. SMT siblings hide those stalls, so `LatencyBound` profile
/// (SMT-on, 4x oversubscribe) is the optimal dispatch shape.
#[inline(never)]
fn latency_bound_op(x: f64) -> f64 {
    let mut v = x;
    for _ in 0..100 {
        v = v.sqrt() * 1.0000001_f64;
    }
    v
}

/// Port-bound op: 50-deep u128 wrapping-multiply chain. ~12-15 ns
/// per element. u128 IMUL goes to a single execution port (port 1
/// on Zen); the pipeline is full with multiplies. SMT siblings
/// compete for the same port and reduce throughput, so `PortBound`
/// profile (SMT-off, 2x oversubscribe) is the optimal dispatch shape.
#[inline(never)]
fn port_bound_op(x: u64) -> u64 {
    let mut v = x | 1;
    for _ in 0..50 {
        let prod = (v as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15u128);
        v = (prod as u64) ^ ((prod >> 64) as u64).wrapping_add(1);
    }
    v
}

/// Fine-grain op: 3-deep sqrt chain. ~20 ns per element. Tests the
/// inline-collapse decision: at N=10k aggregate ~200 us is below the
/// parallel-dispatch crossover, and the tier picker should drop to
/// Inline serial rather than pay the fork overhead. At N=100k
/// aggregate ~2 ms is large enough that parallel-with-lazy-steal
/// still beats both serial and eager-fork.
#[inline(never)]
fn fine_grain_op(x: f64) -> f64 {
    let mut v = x;
    for _ in 0..3 {
        v = v.sqrt() * 1.0000001_f64;
    }
    v
}

// ===========================================================================
// Fine-Grain workload: the canonical inline-collapse demonstration.
// ===========================================================================
fn bench_fine_grain(c: &mut Criterion) {
    let mut g = c.benchmark_group("fine_grain_3sqrt");
    g.sample_size(50);

    for &n in &[10_000usize, 100_000] {
        let template: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        g.bench_with_input(BenchmarkId::new("serial", n), &template, |b, tpl| {
            b.iter_batched_ref(
                || tpl.clone(),
                |v| {
                    for x in v.iter_mut() {
                        *x = fine_grain_op(*x);
                    }
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("rayon_par_iter_mut", n), &template, |b, tpl| {
            use rayon::prelude::*;
            b.iter_batched_ref(
                || tpl.clone(),
                |v| {
                    v.par_iter_mut().for_each(|x| *x = fine_grain_op(*x));
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("flynnel_for_each_chunk", n), &template, |b, tpl| {
            b.iter_batched_ref(
                || tpl.clone(),
                |v| {
                    let plan = JobPlan::new(6, n as u32);
                    for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [f64]| {
                        for x in slice {
                            *x = fine_grain_op(*x);
                        }
                    });
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }

    g.finish();
}

// ===========================================================================
// Latency-Bound workload: parallel-dispatch IS justified; `LatencyBound`
// profile (SMT-on, 4x oversubscribe) wins.
// ===========================================================================
fn bench_latency_bound(c: &mut Criterion) {
    let mut g = c.benchmark_group("latency_bound_100sqrt");
    g.sample_size(20);

    for &n in &[10_000usize, 100_000] {
        let template: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        g.bench_with_input(BenchmarkId::new("serial", n), &template, |b, tpl| {
            b.iter_batched_ref(
                || tpl.clone(),
                |v| {
                    for x in v.iter_mut() {
                        *x = latency_bound_op(*x);
                    }
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("rayon_par_iter_mut", n), &template, |b, tpl| {
            use rayon::prelude::*;
            b.iter_batched_ref(
                || tpl.clone(),
                |v| {
                    v.par_iter_mut().for_each(|x| *x = latency_bound_op(*x));
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("flynnel_default", n), &template, |b, tpl| {
            b.iter_batched_ref(
                || tpl.clone(),
                |v| {
                    let plan = JobPlan::new(6, n as u32);
                    for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [f64]| {
                        for x in slice {
                            *x = latency_bound_op(*x);
                        }
                    });
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(
            BenchmarkId::new("flynnel_for_profile_LatencyBound_smt_on", n),
            &template,
            |b, tpl| {
                b.iter_batched_ref(
                    || tpl.clone(),
                    |v| {
                        let plan = JobPlan::set_profile(6, n as u32, DispatchProfile::LatencyBound);
                        for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [f64]| {
                            for x in slice {
                                *x = latency_bound_op(*x);
                            }
                        });
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        // Experiment variants: each routes through a different
        // `for_each_chunk` policy. See the architectural directions
        // in `colab/HANDOFF-scheduler-tuning.md`.
        g.bench_with_input(
            BenchmarkId::new("flynnel_v_producer_max_len_workers", n),
            &template,
            |b, tpl| {
                b.iter_batched_ref(
                    || tpl.clone(),
                    |v| {
                        let plan = JobPlan::new(6, n as u32)
                            .with_bisect_variant(BisectVariant::ProducerMaxLenWorkers);
                        for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [f64]| {
                            for x in slice {
                                *x = latency_bound_op(*x);
                            }
                        });
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        g.bench_with_input(
            BenchmarkId::new("flynnel_v_rayon_style_replenish", n),
            &template,
            |b, tpl| {
                b.iter_batched_ref(
                    || tpl.clone(),
                    |v| {
                        let plan = JobPlan::new(6, n as u32)
                            .with_bisect_variant(BisectVariant::RayonStyleReplenish);
                        for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [f64]| {
                            for x in slice {
                                *x = latency_bound_op(*x);
                            }
                        });
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

    }

    g.finish();
}

// ===========================================================================
// Port-Bound workload: IMUL-port-saturated; `PortBound` profile (SMT-off,
// 2x oversubscribe) wins.
// ===========================================================================
fn bench_port_bound(c: &mut Criterion) {
    let mut g = c.benchmark_group("port_bound_u128imul");
    g.sample_size(30);

    for &n in &[10_000usize, 100_000] {
        let template: Vec<u64> = (1..=n as u64).collect();

        g.bench_with_input(BenchmarkId::new("serial", n), &template, |b, tpl| {
            b.iter_batched_ref(
                || tpl.clone(),
                |v| {
                    for x in v.iter_mut() {
                        *x = port_bound_op(*x);
                    }
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("rayon_par_iter_mut", n), &template, |b, tpl| {
            use rayon::prelude::*;
            b.iter_batched_ref(
                || tpl.clone(),
                |v| {
                    v.par_iter_mut().for_each(|x| *x = port_bound_op(*x));
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("flynnel_default", n), &template, |b, tpl| {
            b.iter_batched_ref(
                || tpl.clone(),
                |v| {
                    let plan = JobPlan::new(6, n as u32);
                    for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [u64]| {
                        for x in slice {
                            *x = port_bound_op(*x);
                        }
                    });
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(
            BenchmarkId::new("flynnel_for_profile_PortBound_smt_off", n),
            &template,
            |b, tpl| {
                b.iter_batched_ref(
                    || tpl.clone(),
                    |v| {
                        let plan = JobPlan::set_profile(6, n as u32, DispatchProfile::PortBound);
                        for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [u64]| {
                            for x in slice {
                                *x = port_bound_op(*x);
                            }
                        });
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        // Experiment variants (same set as the Latency-Bound bench;
        // see `colab/HANDOFF-scheduler-tuning.md`).
        g.bench_with_input(
            BenchmarkId::new("flynnel_v_producer_max_len_workers", n),
            &template,
            |b, tpl| {
                b.iter_batched_ref(
                    || tpl.clone(),
                    |v| {
                        let plan = JobPlan::new(6, n as u32)
                            .with_bisect_variant(BisectVariant::ProducerMaxLenWorkers);
                        for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [u64]| {
                            for x in slice {
                                *x = port_bound_op(*x);
                            }
                        });
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        g.bench_with_input(
            BenchmarkId::new("flynnel_v_rayon_style_replenish", n),
            &template,
            |b, tpl| {
                b.iter_batched_ref(
                    || tpl.clone(),
                    |v| {
                        let plan = JobPlan::new(6, n as u32)
                            .with_bisect_variant(BisectVariant::RayonStyleReplenish);
                        for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [u64]| {
                            for x in slice {
                                *x = port_bound_op(*x);
                            }
                        });
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

    }

    g.finish();
}

criterion_group!(
    inline_collapse,
    bench_fine_grain,
    bench_latency_bound,
    bench_port_bound
);
criterion_main!(inline_collapse);

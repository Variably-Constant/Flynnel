//! Bench: isolate pure scheduler overhead from leaf-body work.
//!
//! Runs `for_each_chunk` and `par_iter_mut().for_each` with bodies of
//! varying weight at fixed N, AND at varying N with fixed body weight.
//! The "no-op body" rows give pure scheduler+dispatch cost. The
//! difference between flynnel and rayon at the no-op body is the
//! pure scheduling gap.
//!
//! If flynnel's no-op-body is 2x rayon's, the gap is in the scheduler.
//! If they're equal, the gap is at the leaf-body boundary
//! (record_leaf bracket cost, closure shape difference).
//!
//! Run with:
//!   cargo bench --bench sched_overhead_isolation

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use flynnel::sched::par_iter::for_each_chunk;
use flynnel::JobPlan;

// Body weights, expressed as iterations of a sqrt chain per element.
const WEIGHTS: &[(&str, u32)] = &[
    ("noop", 0),       // pure scheduler cost (closure is touched but does no real work)
    ("w5", 5),         // ~30 ns/elem - closer to Light tier
    ("w20", 20),       // ~120 ns/elem
    ("w100", 100),     // ~600 ns/elem - Heavy tier (matches inline_collapse)
];

// Sizes - picked to cover the gap-scaling pattern.
const SIZES: &[usize] = &[10_000, 100_000];

#[inline(never)]
fn body_chain(x: f64, weight: u32) -> f64 {
    if weight == 0 {
        // Touch the value so the compiler doesn't optimize the closure
        // away entirely. black_box keeps the read live; the value
        // returned to the slot is the unchanged input.
        return black_box(x);
    }
    let mut v = x;
    for _ in 0..weight {
        v = v.sqrt() * 1.0000001_f64;
    }
    v
}

fn bench_scheduler_overhead(c: &mut Criterion) {
    let mut g = c.benchmark_group("sched_overhead_isolation");
    g.sample_size(30);

    for &(weight_name, weight) in WEIGHTS {
        for &n in SIZES {
            let template: Vec<f64> = (1..=n).map(|i| i as f64).collect();

            // serial reference for absolute-vs-ideal calculation
            let id_serial = format!("serial_{weight_name}");
            g.bench_with_input(BenchmarkId::new(id_serial, n), &template, |b, tpl| {
                b.iter_batched_ref(
                    || tpl.clone(),
                    |v| {
                        for x in v.iter_mut() {
                            *x = body_chain(*x, weight);
                        }
                    },
                    criterion::BatchSize::LargeInput,
                );
            });

            // rayon contender
            let id_rayon = format!("rayon_{weight_name}");
            g.bench_with_input(BenchmarkId::new(id_rayon, n), &template, |b, tpl| {
                use rayon::prelude::*;
                b.iter_batched_ref(
                    || tpl.clone(),
                    |v| {
                        v.par_iter_mut().for_each(|x| *x = body_chain(*x, weight));
                    },
                    criterion::BatchSize::LargeInput,
                );
            });

            // flynnel default path
            let id_flynnel = format!("flynnel_{weight_name}");
            g.bench_with_input(BenchmarkId::new(id_flynnel, n), &template, |b, tpl| {
                let plan = JobPlan::new(6, n as u32);
                b.iter_batched_ref(
                    || tpl.clone(),
                    |v| {
                        for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [f64]| {
                            for x in slice {
                                *x = body_chain(*x, weight);
                            }
                        });
                    },
                    criterion::BatchSize::LargeInput,
                );
            });

            // flynnel with PER-ELEMENT closure shape (matches rayon's
            // body shape: closure invoked per element, not per leaf).
            // Isolates whether the per-leaf vs per-element closure
            // shape matters for the gap.
            let id_flynnel_per_elem = format!("flynnel_per_elem_{weight_name}");
            g.bench_with_input(
                BenchmarkId::new(id_flynnel_per_elem, n),
                &template,
                |b, tpl| {
                    let plan = JobPlan::new(6, n as u32);
                    b.iter_batched_ref(
                        || tpl.clone(),
                        |v| {
                            for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [f64]| {
                                // Per-element call shape inside the
                                // leaf - same as rayon's for_each
                                // pattern. Lets us factor out the
                                // closure-call-shape difference.
                                slice
                                    .iter_mut()
                                    .for_each(|x| *x = body_chain(*x, weight));
                            });
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );

            // flynnel with OVERSUBSCRIPTION=0 (i.e. 1 leaf per worker,
            // 16 leaves total on Zen+/16T - approximately matches
            // rayon's LengthSplitter initial leaf count). Tests
            // whether the gap is caused by flynnel's default 2x
            // oversubscription producing twice as many leaves as
            // rayon, each with cumulative coordination cost.
            let id_flynnel_1leaf = format!("flynnel_1leaf_per_worker_{weight_name}");
            g.bench_with_input(
                BenchmarkId::new(id_flynnel_1leaf, n),
                &template,
                |b, tpl| {
                    let plan = JobPlan::new(6, n as u32).with_oversubscription_log2(0);
                    b.iter_batched_ref(
                        || tpl.clone(),
                        |v| {
                            for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [f64]| {
                                for x in slice {
                                    *x = body_chain(*x, weight);
                                }
                            });
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    g.finish();
}

criterion_group!(sched_overhead, bench_scheduler_overhead);
criterion_main!(sched_overhead);

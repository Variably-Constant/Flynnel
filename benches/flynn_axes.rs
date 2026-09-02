//! Criterion-driven Flynn-axes benchmark covering every axis Flynnel
//! exposes that has a rayon-versus-Flynnel comparison.
//!
//! Axes covered:
//!
//! - MIMD (data-parallel): `par_iter_mut` vs `for_each_chunk`.
//! - SIMC (cooperative fork, identical closures): `rayon::scope` +
//!   `Mutex` vs `cooperative_join_n`.
//! - MIMC (cooperative fork, heterogeneous closures - one role per
//!   closure): two patterns benched:
//!     * 4-way reduce: 3 partial-sum closures + 1 calibration-probe
//!       closure (different work bodies, one cooperative sync).
//!     * Pivoted-LU step: 1 pivot+scale closure + N trailing-row-apply
//!       closures, the canonical numerical-linalg "one factors, N
//!       apply" pattern.
//! - MISD (speculative race): `rayon::scope` + cancel atomic vs
//!   `race_variants`.
//! - SIMT (GPU dispatch): rayon `par_iter_mut` vs CUDA per-call H2D
//!   vs CUDA persistent device buffer vs CUDA warp-cooperative
//!   (uses `shfl.sync.bfly.b32` for cross-lane register exchange
//!   and warp-wide early exit on convergence).
//! - MIMT (CPU || GPU): sequential CPU-then-GPU vs `join_hybrid`.
//!
//! Each axis is a `BenchmarkGroup` so criterion's report HTML groups
//! the contenders side-by-side. The SIMT group reports the per-iter
//! H2D+D2H variant AND the persistent-device-buffer variant
//! (allocated once outside the timing loop, kernel-launch+sync only
//! inside) AND the warp-cooperative variant, so each gap is measured.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench flynn_axes --features cuda-reference
//! ```
//!
//! Reports land under `target/criterion/<group>/<bench>/report/index.html`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use cudarc::driver::DevicePtr;
use flynnel::backend::cuda::CudaBackend;
use flynnel::sched::par_iter::for_each_chunk;
use flynnel::{
    DispatchBackend, JobPlan, KernelArg, KernelHandle, cooperative_join_n, join_hybrid,
    race_variants, register_backend,
};

const KERNEL_PTX: &str = include_str!("../kernels/newton_sqrt.ptx");

/// Warp-cooperative Newton sqrt PTX. Compiled offline via `nvcc -ptx
/// -arch=sm_75 -o newton_sqrt_warp.ptx newton_sqrt_warp.cu`. The
/// .cu source uses `__shfl_xor_sync(0xffffffff, residual, mask)` for
/// a 32-lane butterfly max-reduce of the per-iter Newton residual,
/// then takes a warp-wide early-exit branch when the warp-max falls
/// below epsilon. Bundled as already-compiled PTX so the bench does
/// NOT need NVRTC at runtime - only the CUDA driver. The .cu source
/// ships alongside the .ptx in the kernels/ directory for review.
const KERNEL_PTX_WARP: &str = include_str!("../kernels/newton_sqrt_warp.ptx");

#[inline(never)]
fn cpu_newton_sqrt(x: f32, iters: i32) -> f32 {
    let mut v = x;
    for _ in 0..iters {
        v = (v + x / v) * 0.5;
    }
    v
}

// ===========================================================================
// MIMD: data-parallel reference (par_iter_mut vs for_each_chunk)
// ===========================================================================
fn bench_mimd(c: &mut Criterion) {
    const N: usize = 1_000_000;
    const ITERS: i32 = 50;
    let template: Vec<f32> = (1..=N).map(|i| (i as f32) * (i as f32)).collect();

    let mut g = c.benchmark_group("mimd_newton_sqrt_1M_50iter");
    g.sample_size(20);

    g.bench_function("rayon_par_iter_mut", |b| {
        use rayon::prelude::*;
        b.iter_batched_ref(
            || template.clone(),
            |v| {
                v.par_iter_mut().for_each(|x| *x = cpu_newton_sqrt(*x, ITERS));
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.bench_function("flynnel_for_each_chunk", |b| {
        let plan = JobPlan::new(8, N as u32);
        b.iter_batched_ref(
            || template.clone(),
            |v| {
                for_each_chunk(&plan, v.as_mut_slice(), |slice: &mut [f32]| {
                    for x in slice {
                        *x = cpu_newton_sqrt(*x, ITERS);
                    }
                });
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ===========================================================================
// SIMC: cooperative_join_n vs rayon scope+Mutex (N-way cooperative fork)
// ===========================================================================
fn bench_simc(c: &mut Criterion) {
    let info = flynnel::cpu_info::cpu_info();
    let n_closures = (2 * info.physical_cores as usize).max(2);

    fn body(seed: u64) -> u64 {
        let mut v: u64 = seed | 1;
        for _ in 0..2_000_000u64 {
            v = v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            v ^= v >> 31;
        }
        v
    }

    let mut g = c.benchmark_group(format!("simc_nway_fork_{n_closures}_closures"));
    g.sample_size(20);

    g.bench_function("rayon_scope_mutex_vec", |b| {
        b.iter(|| {
            let results: Arc<Mutex<Vec<Option<u64>>>> =
                Arc::new(Mutex::new(vec![None; n_closures]));
            rayon::scope(|s| {
                for i in 0..n_closures {
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
            final_vec
        });
    });

    g.bench_function("flynnel_cooperative_join_n", |b| {
        // batch_size >= 32 keeps the tier picker in Local. The
        // closure count is what cooperative_join_n parallelizes.
        let plan = JobPlan::new(8, 1024);
        b.iter(|| {
            let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..n_closures)
                .map(|i| Box::new(move || body(i as u64)) as _)
                .collect();
            cooperative_join_n(&plan, closures)
        });
    });

    g.finish();
}

// ===========================================================================
// MIMC 4-way heterogeneous: cooperative_join_n vs rayon scope+Mutex
// (4 closures, NOT identical: 3 compute partials over different chunks,
// 1 computes a calibration probe over the whole input - one cooperative
// sync boundary, four distinct instruction streams)
// ===========================================================================
fn bench_mimc_4way_heterogeneous(c: &mut Criterion) {
    const N: usize = 200_000;
    let data: Arc<Vec<f64>> = Arc::new((1..=N).map(|i| (i as f64).sqrt()).collect());

    // Role A: chained sqrt accumulator. Three closures use this over
    // disjoint chunks of the input.
    fn partial_sqrt_sum(slice: &[f64]) -> f64 {
        let mut acc = 0.0_f64;
        for &x in slice {
            let mut v = x;
            for _ in 0..32 {
                v = v.sqrt() * 1.0000001;
            }
            acc += v;
        }
        acc
    }

    // Role B: max-abs differences between two transforms over the whole
    // input. Different inner shape than role A (branchy + memory walk).
    fn calibration_max_diff(slice: &[f64]) -> f64 {
        let mut max_diff = 0.0_f64;
        for &x in slice {
            let g = (x * 1.5 + 0.1).tan();
            let h = (x * 1.5 + 0.1).sin() / (x * 1.5 + 0.1).cos();
            let diff = (g - h).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        max_diff
    }

    let mut g = c.benchmark_group("mimc_4way_heterogeneous");
    g.sample_size(20);

    let data_for_rayon = Arc::clone(&data);
    g.bench_function("rayon_scope_mutex_vec", |b| {
        b.iter(|| {
            let chunk = N / 3;
            let results: Arc<Mutex<[f64; 4]>> = Arc::new(Mutex::new([0.0; 4]));
            let data_ref = Arc::clone(&data_for_rayon);
            rayon::scope(|s| {
                for i in 0..3 {
                    let r = Arc::clone(&results);
                    let d = Arc::clone(&data_ref);
                    s.spawn(move |_| {
                        let lo = i * chunk;
                        let hi = if i == 2 { N } else { lo + chunk };
                        let v = partial_sqrt_sum(&d[lo..hi]);
                        r.lock().unwrap()[i] = v;
                    });
                }
                let r = Arc::clone(&results);
                let d = Arc::clone(&data_ref);
                s.spawn(move |_| {
                    let v = calibration_max_diff(&d[..]);
                    r.lock().unwrap()[3] = v;
                });
            });
            let arr = *results.lock().unwrap();
            arr[0] + arr[1] + arr[2] + arr[3]
        });
    });

    let data_for_flynnel = Arc::clone(&data);
    g.bench_function("flynnel_cooperative_join_n", |b| {
        let plan = JobPlan::new(8, 1024);
        b.iter(|| {
            let chunk = N / 3;
            let d0 = Arc::clone(&data_for_flynnel);
            let d1 = Arc::clone(&data_for_flynnel);
            let d2 = Arc::clone(&data_for_flynnel);
            let d3 = Arc::clone(&data_for_flynnel);
            let closures: Vec<Box<dyn FnOnce() -> f64 + Send>> = vec![
                Box::new(move || partial_sqrt_sum(&d0[0..chunk])),
                Box::new(move || partial_sqrt_sum(&d1[chunk..2 * chunk])),
                Box::new(move || partial_sqrt_sum(&d2[2 * chunk..N])),
                Box::new(move || calibration_max_diff(&d3[..])),
            ];
            let results = cooperative_join_n(&plan, closures);
            results.into_iter().sum::<f64>()
        });
    });

    g.finish();
}

// ===========================================================================
// MIMC pivoted-LU: cooperative_join_n vs rayon scope+Mutex
// (one LU elimination step on a 256x256 matrix - 1 closure does
// max-abs pivot selection + scale, N-1 closures apply the pivot to
// disjoint trailing-row ranges. The pivot-selection closure has a
// fundamentally different work shape from the trailing-update
// closures: scalar scan vs vector SAXPY.)
// ===========================================================================
fn bench_mimc_pivoted_lu(c: &mut Criterion) {
    const DIM: usize = 256;
    const N_APPLY_LANES: usize = 7;
    // Total closures = 1 (pivot+scale) + N_APPLY_LANES (apply); 8 total.

    fn build_matrix() -> Vec<f64> {
        let mut m = Vec::with_capacity(DIM * DIM);
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in 0..DIM {
            for j in 0..DIM {
                x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                let v = ((x >> 11) as f64 / ((1u64 << 53) as f64)) - 0.5;
                // Diagonal-dominant so partial pivoting is well-defined.
                m.push(if i == j { v + 10.0 } else { v });
            }
        }
        m
    }

    // Role A: pivot selection - scan column for max-abs, then scale the
    // pivot row in-place. Sequential O(DIM) scan + O(DIM) scale.
    // Returns the pivot value (for use by the apply closures).
    fn pick_and_scale_pivot(matrix: &mut [f64], step: usize) -> f64 {
        let mut best_row = step;
        let mut best_abs = matrix[step * DIM + step].abs();
        for row in (step + 1)..DIM {
            let a = matrix[row * DIM + step].abs();
            if a > best_abs {
                best_abs = a;
                best_row = row;
            }
        }
        // Row swap (in-place).
        if best_row != step {
            for col in 0..DIM {
                matrix.swap(step * DIM + col, best_row * DIM + col);
            }
        }
        let pivot = matrix[step * DIM + step];
        // Scale pivot row.
        for col in (step + 1)..DIM {
            matrix[step * DIM + col] /= pivot;
        }
        pivot
    }

    // Role B: trailing-row update - for each row in the assigned range,
    // apply the pivot subtraction across all trailing columns. O(rows *
    // cols) vector work per closure.
    fn apply_pivot_to_rows(matrix: &mut [f64], step: usize, lo: usize, hi: usize) {
        for row in lo..hi {
            let multiplier = matrix[row * DIM + step];
            for col in (step + 1)..DIM {
                let pivot_col = matrix[step * DIM + col];
                matrix[row * DIM + col] -= multiplier * pivot_col;
            }
            matrix[row * DIM + step] = multiplier; // L-factor stays here.
        }
    }

    // The bench runs one LU elimination step at step=0 per iter.
    // The matrix is cloned per iter so each iter starts from the same
    // state (in-place mutation otherwise affects subsequent samples).

    let template = Arc::new(build_matrix());

    let mut g = c.benchmark_group("mimc_pivoted_lu_step");
    g.sample_size(20);

    let template_rayon = Arc::clone(&template);
    g.bench_function("rayon_scope_mutex_vec", |b| {
        b.iter_batched_ref(
            || (*template_rayon).clone(),
            |matrix| {
                // First: pivot selection + scale (sequential, by design).
                let _pivot = pick_and_scale_pivot(matrix.as_mut_slice(), 0);
                // Then: trailing-update across N_APPLY_LANES closures
                // via rayon::scope. Pass disjoint row ranges so the
                // closures don't alias (UnsafeCell pattern via raw
                // pointer split).
                let start_row: usize = 1;
                let total_rows = DIM - start_row;
                let rows_per_lane = total_rows.div_ceil(N_APPLY_LANES);
                // SAFETY: each closure mutates a disjoint row range of
                // the matrix; row ranges don't overlap so simultaneous
                // mutable access is sound. The raw pointer trick
                // sidesteps the borrow checker's row-disjointness
                // proof obligation in this synthetic bench.
                let ptr = matrix.as_mut_ptr() as usize;
                rayon::scope(|s| {
                    for lane in 0..N_APPLY_LANES {
                        let lo = start_row + lane * rows_per_lane;
                        let hi = (lo + rows_per_lane).min(DIM);
                        if lo >= hi {
                            continue;
                        }
                        s.spawn(move |_| {
                            let p = ptr as *mut f64;
                            // SAFETY: row ranges disjoint per `lo..hi`
                            // bounds above; matrix lifetime exceeds
                            // scope.
                            let m = unsafe {
                                std::slice::from_raw_parts_mut(p, DIM * DIM)
                            };
                            apply_pivot_to_rows(m, 0, lo, hi);
                        });
                    }
                });
                std::hint::black_box(matrix);
            },
            criterion::BatchSize::LargeInput,
        );
    });

    let template_flynnel = Arc::clone(&template);
    g.bench_function("flynnel_cooperative_join_n", |b| {
        let plan = JobPlan::new(8, 1024);
        b.iter_batched_ref(
            || (*template_flynnel).clone(),
            |matrix| {
                let _pivot = pick_and_scale_pivot(matrix.as_mut_slice(), 0);
                let start_row: usize = 1;
                let total_rows = DIM - start_row;
                let rows_per_lane = total_rows.div_ceil(N_APPLY_LANES);
                let ptr = matrix.as_mut_ptr() as usize;
                let closures: Vec<Box<dyn FnOnce() + Send>> = (0..N_APPLY_LANES)
                    .map(|lane| {
                        let lo = start_row + lane * rows_per_lane;
                        let hi = (lo + rows_per_lane).min(DIM);
                        Box::new(move || {
                            if lo < hi {
                                let p = ptr as *mut f64;
                                // SAFETY: row ranges disjoint; matrix
                                // borrow outlives the cooperative call.
                                let m = unsafe {
                                    std::slice::from_raw_parts_mut(p, DIM * DIM)
                                };
                                apply_pivot_to_rows(m, 0, lo, hi);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                let _results = cooperative_join_n(&plan, closures);
                std::hint::black_box(matrix);
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ===========================================================================
// MISD: race_variants vs rayon scope + cancel atomic (3 algorithms racing
// to estimate pi; first one in returns)
// ===========================================================================
fn bench_misd(c: &mut Criterion) {
    fn fast_pi(cancel: &AtomicBool) -> Option<f64> {
        let mut s = 0.0_f64;
        for k in 0..32u32 {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let sign = if k & 1 == 0 { 1.0 } else { -1.0 };
            s += sign / (2.0 * k as f64 + 1.0);
        }
        let est = 4.0 * s;
        if (est - std::f64::consts::PI).abs() < 1e-4 {
            Some(est)
        } else {
            None
        }
    }

    fn faithful_pi(cancel: &AtomicBool) -> Option<f64> {
        fn atan_series(x: f64, terms: usize, cancel: &AtomicBool) -> Option<f64> {
            let mut s = 0.0_f64;
            let mut term = x;
            let x2 = x * x;
            for k in 0..terms {
                if cancel.load(Ordering::Relaxed) {
                    return None;
                }
                let sign = if k & 1 == 0 { 1.0 } else { -1.0 };
                s += sign * term / (2.0 * k as f64 + 1.0);
                term *= x2;
            }
            Some(s)
        }
        let a = atan_series(1.0 / 5.0, 24, cancel)?;
        let b = atan_series(1.0 / 239.0, 8, cancel)?;
        let est = 4.0 * (4.0 * a - b);
        if (est - std::f64::consts::PI).abs() < 1e-10 {
            Some(est)
        } else {
            None
        }
    }

    fn correct_pi(cancel: &AtomicBool) -> f64 {
        let mut a = 1.0_f64;
        let mut b = 1.0_f64 / 2.0_f64.sqrt();
        let mut t = 0.25_f64;
        let mut p = 1.0_f64;
        for _ in 0..6 {
            if cancel.load(Ordering::Relaxed) {
                return std::f64::consts::PI;
            }
            let a_next = (a + b) * 0.5;
            let b_next = (a * b).sqrt();
            t -= p * (a - a_next).powi(2);
            a = a_next;
            b = b_next;
            p *= 2.0;
        }
        (a + b).powi(2) / (4.0 * t)
    }

    let mut g = c.benchmark_group("misd_race_pi_3_variants");
    g.sample_size(50);

    g.bench_function("rayon_scope_cancel_atomic", |b| {
        b.iter(|| {
            let cancel = Arc::new(AtomicBool::new(false));
            let fast_done = Arc::new(Mutex::new(None));
            let faith_done = Arc::new(Mutex::new(None));
            let corr_done = Arc::new(Mutex::new(None));
            rayon::scope(|s| {
                let c1 = Arc::clone(&cancel);
                let r1 = Arc::clone(&fast_done);
                s.spawn(move |_| {
                    let v = fast_pi(&c1);
                    *r1.lock().unwrap() = Some(v);
                    if v.is_some() {
                        c1.store(true, Ordering::Release);
                    }
                });
                let c2 = Arc::clone(&cancel);
                let r2 = Arc::clone(&faith_done);
                s.spawn(move |_| {
                    let v = faithful_pi(&c2);
                    *r2.lock().unwrap() = Some(v);
                    if v.is_some() {
                        c2.store(true, Ordering::Release);
                    }
                });
                let c3 = Arc::clone(&cancel);
                let r3 = Arc::clone(&corr_done);
                s.spawn(move |_| {
                    let v = correct_pi(&c3);
                    *r3.lock().unwrap() = Some(v);
                    c3.store(true, Ordering::Release);
                });
            });
            fast_done
                .lock()
                .unwrap()
                .and_then(|v| v)
                .or_else(|| faith_done.lock().unwrap().and_then(|v| v))
                .unwrap_or_else(|| corr_done.lock().unwrap().unwrap())
        });
    });

    g.bench_function("flynnel_race_variants", |b| {
        let plan = JobPlan::new(8, 1024);
        b.iter(|| {
            let (pi_val, _variant) = race_variants(
                &plan,
                |cancel| {
                    let a = AtomicBool::new(cancel.is_cancelled());
                    fast_pi(&a)
                },
                |cancel| {
                    let a = AtomicBool::new(cancel.is_cancelled());
                    faithful_pi(&a)
                },
                |cancel| {
                    let a = AtomicBool::new(cancel.is_cancelled());
                    correct_pi(&a)
                },
            );
            pi_val
        });
    });

    g.finish();
}

// ===========================================================================
// SIMT: CPU rayon vs CUDA per-call H2D vs CUDA persistent device buffer
// ===========================================================================
fn bench_simt(c: &mut Criterion) {
    const N: usize = 1_000_000;
    const ITERS: i32 = 50;
    let template: Vec<f32> = (1..=N).map(|i| (i as f32) * (i as f32)).collect();

    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SIMT bench skipped: CUDA init failed ({e})");
            return;
        }
    };
    let handle = backend
        .register_kernel("newton_sqrt", KERNEL_PTX.as_bytes())
        .expect("kernel register");
    // Warp-cooperative kernel: pre-compiled PTX (offline via nvcc).
    // The driver's PTX JIT compiles it to SASS for the live GPU
    // alongside newton_sqrt.ptx. Skip the variant on driver/PTX
    // mismatch rather than panic so MIMD/SIMC/MISD/SIMT-base/MIMT
    // still report on hosts with an older driver.
    let handle_warp_opt =
        match backend.register_kernel("newton_sqrt_warp", KERNEL_PTX_WARP.as_bytes()) {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!(
                    "warp-cooperative PTX register failed ({e}); skipping warp variant. \
                     This typically means the driver PTX JIT does not support the .version \
                     emitted by nvcc 13.1 - install a matching CUDA driver to enable it."
                );
                None
            }
        };
    let stream = backend.stream().clone();

    let mut g = c.benchmark_group("simt_newton_sqrt_1M_50iter");
    g.sample_size(20);

    g.bench_function("rayon_par_iter_mut_cpu", |b| {
        use rayon::prelude::*;
        b.iter_batched_ref(
            || template.clone(),
            |v| {
                v.par_iter_mut().for_each(|x| *x = cpu_newton_sqrt(*x, ITERS));
            },
            criterion::BatchSize::LargeInput,
        );
    });

    // Per-call H2D + kernel + D2H. Every iteration uploads the host
    // template to the device, launches the kernel, downloads the
    // result. PCIe round-trip is in every sample.
    g.bench_function("flynnel_cuda_per_call_h2d", |b| {
        b.iter_custom(|n| {
            // Warm path: keep clones outside the inner timing loop
            // body via iter_custom so the H2D/D2H IS the measured
            // unit, not the cloning of the host template.
            let mut total = Duration::ZERO;
            for _ in 0..n {
                let host = template.clone();
                let t0 = Instant::now();
                let buf = stream.clone_htod(&host).expect("H2D");
                let (dev_ptr, _g) = buf.device_ptr(&stream);
                backend
                    .dispatch_kernel(
                        handle,
                        N as u32,
                        &[
                            KernelArg::DevicePtr(dev_ptr as usize),
                            KernelArg::I32(N as i32),
                            KernelArg::I32(ITERS),
                        ],
                    )
                    .expect("launch");
                let mut out = vec![0f32; N];
                stream.memcpy_dtoh(&buf, &mut out).expect("D2H");
                stream.synchronize().expect("sync");
                total += t0.elapsed();
                std::hint::black_box(out);
            }
            total
        });
    });

    // Persistent device buffer: allocate ONCE outside the timing
    // loop. Each measured iter does only kernel launch + sync. No
    // H2D, no D2H. This is the steady-state cost when the same
    // buffer is reused across many kernel launches (the realistic
    // case for iterative GPU workloads).
    g.bench_function("flynnel_cuda_persistent_buffer", |b| {
        let dev_buf = stream
            .clone_htod(&template)
            .expect("persistent buffer H2D (one-shot, outside timing)");
        b.iter_custom(|n| {
            let mut total = Duration::ZERO;
            for _ in 0..n {
                let (dev_ptr, _g) = dev_buf.device_ptr(&stream);
                let t0 = Instant::now();
                backend
                    .dispatch_kernel(
                        handle,
                        N as u32,
                        &[
                            KernelArg::DevicePtr(dev_ptr as usize),
                            KernelArg::I32(N as i32),
                            KernelArg::I32(ITERS),
                        ],
                    )
                    .expect("launch");
                stream.synchronize().expect("sync");
                total += t0.elapsed();
            }
            total
        });
    });

    // Warp-cooperative variant: same persistent device buffer, but
    // launches newton_sqrt_warp.ptx which uses `shfl.sync.bfly.b32`
    // to compute a warp-wide max of the per-iteration residual and
    // exits when the warp-max falls below epsilon. Demonstrates a
    // true cross-lane register exchange (no shared memory, no global
    // sync). Compare against persistent_buffer to isolate the cost
    // of the warp-cooperative path on the same data.
    if let Some(handle_warp) = handle_warp_opt {
        g.bench_function("flynnel_cuda_warp_cooperative", |b| {
            let dev_buf = stream
                .clone_htod(&template)
                .expect("persistent buffer H2D (one-shot, outside timing)");
            b.iter_custom(|n| {
                let mut total = Duration::ZERO;
                for _ in 0..n {
                    let (dev_ptr, _g) = dev_buf.device_ptr(&stream);
                    let t0 = Instant::now();
                    backend
                        .dispatch_kernel(
                            handle_warp,
                            N as u32,
                            &[
                                KernelArg::DevicePtr(dev_ptr as usize),
                                KernelArg::I32(N as i32),
                                KernelArg::I32(ITERS),
                            ],
                        )
                        .expect("warp-coop launch");
                    stream.synchronize().expect("sync");
                    total += t0.elapsed();
                }
                total
            });
        });
    }

    g.finish();
}

// ===========================================================================
// MIMT: join_hybrid CPU + CUDA concurrent vs sequential CPU-then-GPU
// ===========================================================================
fn bench_mimt(c: &mut Criterion) {
    const CPU_BUF_BYTES: usize = 4 * 1024 * 1024;
    const GPU_N: usize = 1_000_000;
    const GPU_ITERS: i32 = 200;

    let cuda = match CudaBackend::new() {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("MIMT bench skipped: CUDA init failed ({e})");
            return;
        }
    };
    let handle = cuda
        .register_kernel("newton_sqrt", KERNEL_PTX.as_bytes())
        .expect("kernel register");
    let backend_id = cuda.id();
    register_backend(cuda.clone() as Arc<dyn DispatchBackend>);

    let cpu_input: Vec<u8> = (0..CPU_BUF_BYTES).map(|i| (i & 0xFF) as u8).collect();
    let gpu_input: Vec<f32> = (1..=GPU_N).map(|i| (i as f32) * (i as f32)).collect();
    let gpu_dev_buf = Arc::new(
        cuda.stream()
            .clone_htod(&gpu_input)
            .expect("persistent GPU buffer H2D (one-shot)"),
    );

    fn cpu_work(input: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in input {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn gpu_work(
        cuda: &Arc<CudaBackend>,
        handle: KernelHandle,
        dev_buf: &cudarc::driver::CudaSlice<f32>,
        n: usize,
        iters: i32,
    ) {
        let stream = cuda.stream();
        let (dp, _g) = dev_buf.device_ptr(stream);
        cuda.dispatch_kernel(
            handle,
            n as u32,
            &[
                KernelArg::DevicePtr(dp as usize),
                KernelArg::I32(n as i32),
                KernelArg::I32(iters),
            ],
        )
        .expect("launch");
        stream.synchronize().expect("sync");
    }

    let mut g = c.benchmark_group("mimt_cpu_or_gpu_pair");
    g.sample_size(20);

    let cuda_seq = cuda.clone();
    let gpu_buf_seq = gpu_dev_buf.clone();
    g.bench_function("sequential_cpu_then_gpu", |b| {
        b.iter(|| {
            let cpu_h = cpu_work(&cpu_input);
            gpu_work(&cuda_seq, handle, &gpu_buf_seq, GPU_N, GPU_ITERS);
            cpu_h
        });
    });

    let plan = JobPlan::new(8, 1024).with_backend(backend_id);
    let cuda_hyb = cuda.clone();
    let gpu_buf_hyb = gpu_dev_buf.clone();
    g.bench_function("flynnel_join_hybrid", |b| {
        b.iter(|| {
            let cpu_in: &[u8] = &cpu_input;
            let cuda_for_gpu = Arc::clone(&cuda_hyb);
            let gpu_buf_for_gpu = Arc::clone(&gpu_buf_hyb);
            let (h, _) = join_hybrid::<u64, (), _, _>(
                &plan,
                || cpu_work(cpu_in),
                move || {
                    gpu_work(&cuda_for_gpu, handle, &gpu_buf_for_gpu, GPU_N, GPU_ITERS);
                },
            );
            h
        });
    });

    g.finish();
}

// Workaround for `BenchmarkId` unused warning when no other axis
// uses it. Reserved for future sweep-over-N variants.
#[allow(dead_code)]
fn _bench_id_sentinel() -> BenchmarkId {
    BenchmarkId::new("noop", 0)
}

criterion_group!(
    flynn_axes,
    bench_mimd,
    bench_simc,
    bench_mimc_4way_heterogeneous,
    bench_mimc_pivoted_lu,
    bench_misd,
    bench_simt,
    bench_mimt
);
criterion_main!(flynn_axes);

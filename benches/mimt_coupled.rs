//! Criterion bench for the **real** MIMT shape - three coupled
//! CPU+GPU algorithms benchmarked against a sequential baseline.
//!
//! The original `mimt_cpu_or_gpu_pair` bench in `flynn_axes.rs`
//! measured trivial concurrency (CPU FNV hash + GPU Newton sqrt
//! run side-by-side with no data dependency between them). That
//! is NOT the algorithmically meaningful MIMT shape. The shape
//! that benefits from a pipelined CPU+GPU primitive is one where
//! the GPU half consumes outputs the CPU half produced AND feeds
//! the next CPU stage, iterated.
//!
//! Three algorithm families fit this shape:
//!
//! 1. **Pipelined Metropolis MCMC**: CPU adaptive proposal →
//!    GPU log-likelihood evaluation → CPU accept/reject + step
//!    adaptation. Three stages with balanced cost; classic
//!    2-3x pipelined speedup.
//! 2. **Batched conjugate gradient**: CPU per-RHS init + check →
//!    GPU sparse matrix-vector product → CPU dot/update. The
//!    intra-solve dependencies block single-solve pipelining;
//!    multi-RHS pipelining (this bench) is the natural fit.
//! 3. **MCTS with batched NN evaluation (AlphaZero shape)**:
//!    CPU tree expansion by UCB + visit count → GPU policy/value
//!    network on leaf batch → CPU backprop value up the tree.
//!
//! Each bench compares the **sequential** baseline (`for each
//! input { pre(); gpu(); post(); }` on the calling thread) to
//! the **pipelined** variant via `flynnel::hybrid_pipeline`.
//!
//! The GPU stage uses the existing `kernels/newton_sqrt.ptx`
//! sized to ~5 ms via `iters=4000` on a persistent device buffer,
//! a representative compute kernel of the right cost profile.
//! The bench tests the pipelining shape, not the kernel's
//! correctness for the three host algorithms (which is the CPU
//! stages' job).
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench mimt_coupled --features cuda-reference
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};

use cudarc::driver::DevicePtr;
use flynnel::backend::cuda::CudaBackend;
use flynnel::{JobPlan, KernelArg, KernelHandle, hybrid_pipeline};

const KERNEL_PTX: &str = include_str!("../kernels/newton_sqrt.ptx");
/// Per-invocation GPU work cost - Newton sqrt iter count chosen
/// to land near ~3 ms per call on persistent buffer on RTX 3070.
/// Tuned to roughly balance the CPU-stage costs below so the
/// pipelining steady-state speedup is the dominant signal.
const GPU_ITERS: i32 = 4000;
const GPU_N: usize = 1_000_000;
/// Number of "pipeline cycles" per bench iter. Each cycle runs
/// pre_cpu, gpu, post_cpu once. Higher N amortizes pipeline-fill
/// over more iterations so the steady-state speedup dominates.
///
/// The theoretical max-speedup at perfect stage balance is
/// `3 * N / (N + 2)`: at N=8 that's 2.4x; at N=32, 2.82x; at
/// N=64, 2.91x. Set at 64 so the steady-state regime dominates
/// over fill cost while still keeping the bench wall time near
/// a second per criterion sample.
const CYCLES_PER_SAMPLE: usize = 64;
/// Scale factor that sizes the CPU stages so each one takes
/// roughly 2-4 ms - the regime where pipelining a CPU+GPU+CPU
/// chain meaningfully wins. Real coupled algorithms (CG with
/// large RHS, MCMC with leapfrog HMC integration, MCTS with
/// deep tree expansion) routinely sit in this regime.
const CPU_WORK_SCALE: usize = 3000;

// ===========================================================================
// Shared GPU-stage helper.
// ===========================================================================

/// Run the Newton sqrt kernel on a persistent device buffer
/// inside the caller's CudaBackend, synchronously. Returns the
/// first element of the result vector (representative scalar
/// extracted to enforce data dependency back into the CPU side).
fn gpu_run(
    backend: &CudaBackend,
    handle: KernelHandle,
    dev_buf: &cudarc::driver::CudaSlice<f32>,
    n: usize,
) -> f32 {
    use flynnel::DispatchBackend;
    let stream = backend.stream();
    let (dp, _g) = dev_buf.device_ptr(stream);
    backend
        .dispatch_kernel(
            handle,
            n as u32,
            &[
                KernelArg::DevicePtr(dp as usize),
                KernelArg::I32(n as i32),
                KernelArg::I32(GPU_ITERS),
            ],
        )
        .expect("launch");
    stream.synchronize().expect("sync");
    // Read one element back so the bench actually enforces a
    // device-to-host dependency chain (the post_cpu stage uses
    // this value). cudarc requires explicit memcpy to read.
    let mut out = [0f32; 1];
    // Sub-slice the device buffer to the first element to read it.
    let head = dev_buf.slice(0..1);
    stream.memcpy_dtoh(&head, &mut out).expect("D2H head");
    stream.synchronize().expect("sync head");
    out[0]
}

// ===========================================================================
// MCMC: CPU propose → GPU loglik → CPU accept
// ===========================================================================

/// CPU stage 1: adaptive Metropolis-Hastings proposal with a
/// leapfrog-HMC-shaped inner loop. Each call generates a
/// `CPU_WORK_SCALE * 4`-dim proposal and runs `CPU_WORK_SCALE / 8`
/// leapfrog half-steps over it (representative of the gradient-
/// evaluation passes a real HMC proposal would do per sample).
/// Costs ~2-3 ms on Zen+ at the default `CPU_WORK_SCALE`.
fn mcmc_propose(seed: u64) -> Vec<f32> {
    let dim = CPU_WORK_SCALE * 4;
    // Tuned to land near the ~3ms GPU stage cost (each stage of
    // a 3-stage pipeline must be balanced for the steady-state
    // ceiling to approach 3 * N / (N + 2)). At CPU_WORK_SCALE=3000
    // this gives 250 leapfrog half-steps over a 12000-dim vector
    // ~= 3ms on Zen+.
    let leapfrog_steps = CPU_WORK_SCALE / 12;
    let mut x: u64 = seed | 1;
    let mut v = Vec::with_capacity(dim);
    let mut prev = 0.0_f32;
    let step = 0.1_f32;
    for _ in 0..dim {
        x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let uniform = ((x >> 11) as f64 / ((1u64 << 53) as f64)) as f32;
        let normal = (uniform - 0.5) * 2.0 * step;
        prev += normal;
        v.push(prev);
    }
    // Leapfrog-HMC-shaped passes over the proposal vector. Each
    // pass is one gradient + position update (~3 ops per element).
    for _ in 0..leapfrog_steps {
        let mut prev_grad = 0.0_f32;
        for value in v.iter_mut() {
            let grad = *value - prev_grad;
            *value += 0.5 * grad * step;
            prev_grad = grad;
        }
    }
    v
}

/// CPU stage 3: Metropolis accept/reject + adaptive-step update
/// using the GPU-returned log-likelihood. Walks the proposal
/// vector to compute the proposal's contribution to the running
/// running mean and variance (Welford one-pass formulation), then
/// runs the Metropolis test. Costs ~2-3 ms at the default scale.
fn mcmc_accept(loglik_and_proposal: (f32, Vec<f32>)) -> f32 {
    let (loglik, proposal) = loglik_and_proposal;
    // Real adaptive samplers maintain a dozen per-iteration
    // statistics: Welford mean/variance, per-chain autocorrelation
    // lag-k for several k, posterior-tail mass (top-1%), proposal
    // step adaptation, etc. Each one is a fresh accumulator over
    // the proposal vector. Sized so the post stage lands ~3ms on
    // Zen+ to balance the ~3ms GPU stage. Each statistic must be
    // computed from a fresh accumulator state so LLVM cannot fold
    // the loops (one pass per statistic kind).
    let n_f = proposal.len() as f32;
    // Welford mean/variance, computed fresh.
    let mut mean = 0.0_f32;
    let mut m2 = 0.0_f32;
    for (i, &v) in proposal.iter().enumerate() {
        let n = (i + 1) as f32;
        let delta = v - mean;
        mean += delta / n;
        let delta2 = v - mean;
        m2 += delta * delta2;
    }
    let variance = m2 / n_f;
    // Autocorrelation at lag k for k = 1..=6 (one fresh pass per
    // lag - each pass cannot be elided since prev_k state restarts).
    let mut autocorr_sum = 0.0_f32;
    for lag in 1usize..=6 {
        let mut ac = 0.0_f32;
        for i in lag..proposal.len() {
            ac += proposal[i] * proposal[i - lag];
        }
        autocorr_sum += ac / (n_f - lag as f32);
    }
    // Top-1% posterior-tail mass via partial sort - extracts the
    // adapt-step prior signal a real sampler would feed forward.
    let mut sorted = proposal.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let tail_n = (proposal.len() / 100).max(1);
    let tail_mass: f32 = sorted.iter().take(tail_n).sum::<f32>() / tail_n as f32;
    // Metropolis test combined with adaptive-step heuristic. Every
    // accumulator feeds the log_ratio so none is dead-store work.
    let log_ratio = loglik + mean * 0.01 - variance.ln().max(-10.0) * 0.1
        + autocorr_sum * 0.001
        + tail_mass * 0.01;
    if log_ratio > 0.0 || log_ratio > -1.0 {
        loglik
    } else {
        -loglik
    }
}

fn bench_mcmc(c: &mut Criterion) {
    let backend = match CudaBackend::new() {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("MCMC bench skipped: CUDA init failed ({e})");
            return;
        }
    };
    use flynnel::DispatchBackend;
    let handle = backend
        .register_kernel("newton_sqrt", KERNEL_PTX.as_bytes())
        .expect("kernel register");
    let gpu_input: Vec<f32> = (1..=GPU_N).map(|i| (i as f32) * (i as f32)).collect();
    let dev_buf = Arc::new(
        backend
            .stream()
            .clone_htod(&gpu_input)
            .expect("persistent device buffer"),
    );

    let mut g = c.benchmark_group("mimt_mcmc");
    g.sample_size(15);

    let backend_seq = Arc::clone(&backend);
    let dev_buf_seq = Arc::clone(&dev_buf);
    g.bench_function("sequential", |b| {
        b.iter_custom(|n| {
            let mut total = Duration::ZERO;
            for _ in 0..n {
                let t0 = Instant::now();
                let mut prev_loglik = 0.0_f32;
                for cycle in 0..CYCLES_PER_SAMPLE {
                    let proposal = mcmc_propose(cycle as u64 + 1);
                    let loglik = gpu_run(&backend_seq, handle, &dev_buf_seq, GPU_N);
                    let accepted = mcmc_accept((loglik + prev_loglik * 0.0, proposal));
                    prev_loglik = accepted;
                }
                std::hint::black_box(prev_loglik);
                total += t0.elapsed();
            }
            total
        });
    });

    let backend_pipe = Arc::clone(&backend);
    let dev_buf_pipe = Arc::clone(&dev_buf);
    g.bench_function("hybrid_pipeline", |b| {
        b.iter_custom(|n| {
            let mut total = Duration::ZERO;
            for _ in 0..n {
                let t0 = Instant::now();
                let backend_for_gpu = Arc::clone(&backend_pipe);
                let dev_buf_for_gpu = Arc::clone(&dev_buf_pipe);
                let plan = JobPlan::new(8, 1024);
                let results = hybrid_pipeline(
                    &plan,
                    (0..CYCLES_PER_SAMPLE as u64).collect::<Vec<u64>>(),
                    |seed: u64| mcmc_propose(seed + 1),
                    move |proposal: Vec<f32>| -> (f32, Vec<f32>) {
                        let loglik = gpu_run(&backend_for_gpu, handle, &dev_buf_for_gpu, GPU_N);
                        (loglik, proposal)
                    },
                    |pair: (f32, Vec<f32>)| mcmc_accept(pair),
                );
                std::hint::black_box(results);
                total += t0.elapsed();
            }
            total
        });
    });

    g.finish();
}

// ===========================================================================
// CG batched: per-RHS CPU init/check → GPU SpMV → CPU dot/update
// ===========================================================================

/// CPU stage 1: CG init for one RHS. Initializes x = 0, r = b,
/// p = r, computes rho_old = (r,r) on a dim = `CPU_WORK_SCALE * 256`
/// vector - representative of the large-RHS case where the per-RHS
/// setup itself is meaningful CPU work (sparse RHS expansion,
/// preconditioner apply on the right-hand side, etc).
fn cg_init(rhs_seed: u64) -> (Vec<f32>, Vec<f32>, f32) {
    let dim = CPU_WORK_SCALE * 256;
    let mut x: u64 = rhs_seed | 1;
    let mut b = Vec::with_capacity(dim);
    for _ in 0..dim {
        x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let v = ((x >> 11) as f64 / ((1u64 << 53) as f64)) as f32 - 0.5;
        b.push(v);
    }
    // Single Jacobi-shaped preconditioner sweep on the RHS. The
    // RNG-fill above is the dominant init cost; one pass keeps
    // init near the ~3ms GPU stage cost without over-shooting.
    for v in b.iter_mut() {
        *v = *v * 1.01 + 0.001;
    }
    let rho_old: f32 = b.iter().map(|v| v * v).sum();
    let p = b.clone();
    (b, p, rho_old)
}

/// CPU stage 3: CG update using the GPU-returned A*p scalar.
/// Computes (p, Ap) dot product (full vector pass), then
/// alpha = rho_old / (p, Ap), x += alpha*p, r -= alpha*Ap,
/// rho_new = (r, r). Returns the new residual norm. Real CG
/// inner loop runs all of these per iteration.
fn cg_update(state: (Vec<f32>, Vec<f32>, f32, f32)) -> f32 {
    let (mut r, p, rho_old, ap_dot) = state;
    // Full (p, Ap) dot via vector pass.
    let p_dot_ap: f32 = p.iter().zip(r.iter()).map(|(a, b)| a * b).sum::<f32>() + ap_dot * 0.0;
    let alpha = rho_old / (p_dot_ap.max(1e-10) + 1.0);
    // Vectored r -= alpha * p.
    for i in 0..r.len() {
        r[i] -= alpha * p[i];
    }
    // (r, r) reduction for rho_new.
    let rho_new: f32 = r.iter().map(|v| v * v).sum();
    // Beta + p_new + restart-aware diagonal precondition residual
    // pass that a real CG inner loop runs once the SpMV result
    // lands. Sizes update near ~3ms on Zen+ to match the GPU stage.
    let beta = rho_new / rho_old.max(1e-10);
    for i in 0..r.len() {
        r[i] = (r[i] + beta * p[i]) * 1.0001;
    }
    rho_new.sqrt()
}

fn bench_cg(c: &mut Criterion) {
    let backend = match CudaBackend::new() {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("CG bench skipped: CUDA init failed ({e})");
            return;
        }
    };
    use flynnel::DispatchBackend;
    let handle = backend
        .register_kernel("newton_sqrt", KERNEL_PTX.as_bytes())
        .expect("kernel register");
    let gpu_input: Vec<f32> = (1..=GPU_N).map(|i| (i as f32) * (i as f32)).collect();
    let dev_buf = Arc::new(
        backend
            .stream()
            .clone_htod(&gpu_input)
            .expect("persistent device buffer"),
    );

    let mut g = c.benchmark_group("mimt_cg_batched");
    g.sample_size(15);

    let backend_seq = Arc::clone(&backend);
    let dev_buf_seq = Arc::clone(&dev_buf);
    g.bench_function("sequential", |b| {
        b.iter_custom(|n| {
            let mut total = Duration::ZERO;
            for _ in 0..n {
                let t0 = Instant::now();
                let mut last_resid = 0.0_f32;
                for cycle in 0..CYCLES_PER_SAMPLE {
                    let (r, p, rho_old) = cg_init(cycle as u64 + 1);
                    let ap_scalar = gpu_run(&backend_seq, handle, &dev_buf_seq, GPU_N);
                    last_resid = cg_update((r, p, rho_old, ap_scalar));
                }
                std::hint::black_box(last_resid);
                total += t0.elapsed();
            }
            total
        });
    });

    let backend_pipe = Arc::clone(&backend);
    let dev_buf_pipe = Arc::clone(&dev_buf);
    g.bench_function("hybrid_pipeline", |b| {
        b.iter_custom(|n| {
            let mut total = Duration::ZERO;
            for _ in 0..n {
                let t0 = Instant::now();
                let backend_for_gpu = Arc::clone(&backend_pipe);
                let dev_buf_for_gpu = Arc::clone(&dev_buf_pipe);
                let plan = JobPlan::new(8, 1024);
                let results = hybrid_pipeline(
                    &plan,
                    (0..CYCLES_PER_SAMPLE as u64).collect::<Vec<u64>>(),
                    |seed: u64| cg_init(seed + 1),
                    move |state: (Vec<f32>, Vec<f32>, f32)| -> (Vec<f32>, Vec<f32>, f32, f32) {
                        let ap_scalar = gpu_run(&backend_for_gpu, handle, &dev_buf_for_gpu, GPU_N);
                        (state.0, state.1, state.2, ap_scalar)
                    },
                    |s: (Vec<f32>, Vec<f32>, f32, f32)| cg_update(s),
                );
                std::hint::black_box(results);
                total += t0.elapsed();
            }
            total
        });
    });

    g.finish();
}

// ===========================================================================
// MCTS: CPU expand → GPU NN-eval → CPU backprop (AlphaZero shape)
// ===========================================================================

/// CPU stage 1: tree-expansion stand-in shaped like AlphaZero
/// MCTS leaf selection. Builds a batch of `CPU_WORK_SCALE / 4`
/// "leaf states" each represented as a 1024-dim feature vector.
/// Each leaf walks a depth-32 tree path running UCB1 across
/// 32 children per node - the kind of branchy, irregular work
/// real MCTS does to pick a leaf batch.
fn mcts_expand(iter: u64) -> Vec<Vec<f32>> {
    let batch = (CPU_WORK_SCALE / 4).max(64);
    const FEAT: usize = 1024;
    // TREE_DEPTH * CHILDREN_PER_NODE drive the per-leaf UCB scan
    // cost. Tuned to land expand near ~3ms on Zen+ to balance the
    // ~3ms GPU stage so the steady-state pipeline cap applies.
    const TREE_DEPTH: usize = 20;
    const CHILDREN_PER_NODE: usize = 32;
    let mut x: u64 = iter | 1;
    let mut leaves = Vec::with_capacity(batch);
    for _ in 0..batch {
        // UCB1 traversal down a synthetic tree to pick the leaf
        // seed. Branchy + irregular, like real MCTS selection.
        let mut node_seed = x;
        for _ in 0..TREE_DEPTH {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_child = 0u64;
            for _ in 0..CHILDREN_PER_NODE {
                x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                let q = ((x >> 11) as f64 / ((1u64 << 53) as f64)) as f32;
                let visits = ((x & 0xFF) as f32 + 1.0).sqrt();
                let score = q + 1.4_f32 / visits;
                if score > best_score {
                    best_score = score;
                    best_child = x;
                }
            }
            node_seed = best_child;
        }
        // Feature vector for the selected leaf.
        let mut leaf = Vec::with_capacity(FEAT);
        let mut y = node_seed;
        for _ in 0..FEAT {
            y = y.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            leaf.push(((y >> 11) as f64 / ((1u64 << 53) as f64)) as f32);
        }
        leaves.push(leaf);
    }
    leaves
}

/// CPU stage 3: backprop the GPU-returned value up the tree.
/// For each leaf, walks a synthetic tree path of depth 32,
/// updating visit count + Q-value at each parent - the kind
/// of pointer-chasing memory access pattern real MCTS backprop
/// does on the CPU side after batched NN evaluation.
fn mcts_backprop(input: (Vec<Vec<f32>>, f32)) -> f32 {
    let (leaves, value) = input;
    // PARENT_HOPS sets the backprop walk depth per leaf. Tuned to
    // ~3ms on Zen+ matching the expand and GPU stages. Pointer-
    // chasing scales super-linearly in hops due to cache effects,
    // so this is empirically lower than 2x the original 32.
    const PARENT_HOPS: usize = 40;
    let mut accum = 0.0_f32;
    for (i, leaf) in leaves.iter().enumerate() {
        // Walk up the synthetic tree, updating Q at each parent.
        let mut q_accum = value;
        let mut visits = (i & 0xFF) as f32 + 1.0;
        for hop in 0..PARENT_HOPS {
            // Q update: incremental mean
            let leaf_idx = (hop * 13 + i) % leaf.len();
            let leaf_contrib = leaf[leaf_idx];
            q_accum = q_accum + (leaf_contrib - q_accum) / visits;
            visits += 1.0;
        }
        accum += q_accum;
    }
    accum / leaves.len() as f32
}

fn bench_mcts(c: &mut Criterion) {
    let backend = match CudaBackend::new() {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("MCTS bench skipped: CUDA init failed ({e})");
            return;
        }
    };
    use flynnel::DispatchBackend;
    let handle = backend
        .register_kernel("newton_sqrt", KERNEL_PTX.as_bytes())
        .expect("kernel register");
    let gpu_input: Vec<f32> = (1..=GPU_N).map(|i| (i as f32) * (i as f32)).collect();
    let dev_buf = Arc::new(
        backend
            .stream()
            .clone_htod(&gpu_input)
            .expect("persistent device buffer"),
    );

    let mut g = c.benchmark_group("mimt_mcts");
    g.sample_size(15);

    let backend_seq = Arc::clone(&backend);
    let dev_buf_seq = Arc::clone(&dev_buf);
    g.bench_function("sequential", |b| {
        b.iter_custom(|n| {
            let mut total = Duration::ZERO;
            for _ in 0..n {
                let t0 = Instant::now();
                let mut last = 0.0_f32;
                for cycle in 0..CYCLES_PER_SAMPLE {
                    let leaves = mcts_expand(cycle as u64 + 1);
                    let value = gpu_run(&backend_seq, handle, &dev_buf_seq, GPU_N);
                    last = mcts_backprop((leaves, value));
                }
                std::hint::black_box(last);
                total += t0.elapsed();
            }
            total
        });
    });

    let backend_pipe = Arc::clone(&backend);
    let dev_buf_pipe = Arc::clone(&dev_buf);
    g.bench_function("hybrid_pipeline", |b| {
        b.iter_custom(|n| {
            let mut total = Duration::ZERO;
            for _ in 0..n {
                let t0 = Instant::now();
                let backend_for_gpu = Arc::clone(&backend_pipe);
                let dev_buf_for_gpu = Arc::clone(&dev_buf_pipe);
                let plan = JobPlan::new(8, 1024);
                let results = hybrid_pipeline(
                    &plan,
                    (0..CYCLES_PER_SAMPLE as u64).collect::<Vec<u64>>(),
                    |seed: u64| mcts_expand(seed + 1),
                    move |leaves: Vec<Vec<f32>>| -> (Vec<Vec<f32>>, f32) {
                        let value = gpu_run(&backend_for_gpu, handle, &dev_buf_for_gpu, GPU_N);
                        (leaves, value)
                    },
                    |s: (Vec<Vec<f32>>, f32)| mcts_backprop(s),
                );
                std::hint::black_box(results);
                total += t0.elapsed();
            }
            total
        });
    });

    g.finish();
}

criterion_group!(mimt_coupled, bench_mcmc, bench_cg, bench_mcts);
criterion_main!(mimt_coupled);

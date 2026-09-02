---
title: Sched Module Reference
weight: 3
---

Every primitive in the `flynnel::sched` module, organised by Flynn axis. The workhorses (`join`, `for_each_chunk`, `cooperative_join_n`, `join_hybrid`, `hybrid_pipeline`, `race_variants`, `k_join`) are also exposed at the crate root (`flynnel::*`).

## Index by Flynn axis

| Axis | Primitive | Module |
|------|-----------|--------|
| MIMD | `join`, `join_context`, `join_default` | [arena](#arena) |
| MIMD | `for_each_chunk` family | [par_iter](#par_iter) |
| MIMD const-K | `k_join`, `k_join_with_plan` | [k_join](#k_join) |
| SIMC | `cooperative_join_n` (identical closures) | [cooperative](#cooperative) |
| MIMC | `cooperative_join_n` (heterogeneous closures, one role per closure) | [cooperative](#cooperative) |
| MIMT single-pair | `join_hybrid` | [hybrid](#hybrid) |
| MIMT pipelined | `hybrid_pipeline` | [hybrid](#hybrid) |
| MIMT learned placement | `hybrid_auto`, `hybrid_auto_split`, `SplitReport`, `Placement` | [hybrid](#hybrid) |
| MISD | `race_variants` (first tolerable result wins, losers cancel) | [race](#race) |
| MIMD explore + select | `explore_select` (all explorers finish, best-by-comparator wins) | [race](#race) |
| Racing family | `race_any` (hedged), `race_quorum` (k-of-n), `race_refute` (duel), `race_agree` (consensus), `race_deadline` (anytime), `race_tournament` (successive halving), `race_statistical` (Hoeffding) | [race](#the-racing-family) |
| One-task-per-element | `par_map_in_place`, `par_zip_apply` | [par_iter](#par_iter) |
| Map + serial fold | `par_map_serial_reduce` (parallel map overlapped with in-order serial combine) | [pipeline](#pipeline) |
| Pipelines | `run`, `run_dyn`, `PipelineStage`, `FnStage` | [pipeline](#pipeline) |
| HW regions | `run_in_region`, `MatrixModeBackend`, `ScalarFallback` | [mode_region](#mode_region) |
| Async helpers | `IoPool`, `submit_io_or_inline`, `global_io_pool` | [io_pool](#io_pool) |
| Calibration | `spawn_calibration`, `timed_avg_ns` | [bg_calibration](#bg_calibration) |
| Prefetch | `prefetch_into_l2`, `prefetch_into_l3`, inline variants | [prefetch](#prefetch) |
| NUMA alloc | `NumaAlloc`, `NUMA_NODE_LOCAL` | [numa_alloc](#numa_alloc) |
| Memory prep | `bg_zero::prepare`, `bg_zero::Handle` | [bg_zero](#bg_zero) |
| Verification | `VerifyChain`, `VerifyHasher`, `default_hasher` (feature `verify-chain`) | [verify_chain](#verify_chain) |
| Observability | `split_multiplier`, `set_split_multiplier`, `spawn_observer`, `LeafStats` | [split_observer](#split_observer) |
| Tier policy | `JobPlan`, `SchedTier`, `HwClass`, `kband_for`, `pick_tier` | [plan](JobPlan-Reference.md) |
| Idempotent jobs | `IdempotentJob`, `run_idempotent` | [idempotent](#idempotent) |
| Unified user-facing | `AdaptiveDispatcher`, `execute_streaming` / `execute_cooperative` / `execute_for_each` / `execute_indexed` | [dispatch](#dispatch) |
| Declarative shape hint | `WorkloadShape` enum + `hints()` mapping | [workload_shape](#workload_shape) |
| Runtime profile migration | `WorkloadClass`, `migrate_workload_class`, `active_dispatch_profile` | [adaptive_profile](#adaptive_profile) |
| Per-call-site adaptive state | `CallSiteState`, `SiteRef`, `PolicyArm`, `Placement` | [call_site](#call_site) |
| Threshold calibration | `ClassThresholds`, `class_thresholds`, `calibrate_class_thresholds`, `spawn_class_threshold_calibration` | [adaptive_profile](#adaptive_profile) |
| Runtime backing migration | `AdaptiveWorker`, `AdaptiveStealer`, `KGating` tag swap | [adaptive_worker](#adaptive_worker), [k_gating](#k_gating) |
| Runtime backend migration | `active_backend_id`, `migrate_backend`, `resolve_active_backend` | [adaptive_backend](#adaptive_backend) |
| In-house Chase-Lev | `Worker`, `Stealer`, `Steal`, `new_chase_lev` (replaces crossbeam) | [chase_lev_local](#chase_lev_local) |
| In-house MPMC ring | `FlynnelRing`, `PushResult`, `PopResult` | [flynnel_ring](#flynnel_ring) |
| In-house SPSC ring | `new_spsc`, `Producer`, `Consumer` (Lamport zero-CAS) | [flynnel_ring_spsc](#flynnel_ring_spsc) |
| In-house MPSC ring | `new_mpsc`, `MpscProducer`, `Consumer` | [flynnel_ring_mpsc](#flynnel_ring_mpsc) |
| Composed N-by-M grid | `ComposedMpsc`, `ComposedMpmc` (per-producer FIFO; 1.98x-2.15x over Vyukov) | [flynnel_ring_composed](#flynnel_ring_composed) |
| Global injector | `Injector`, `InjectorSteal` (drop-in for crossbeam Injector) | [injector](#injector) |
| Blocking notify channel | `NotifyHub`, `NotifySender`, `NotifyReceiver`, `NotifyShutdownOnDrop` | [notify_ring](#notify_ring) |
| Per-NUMA worker pool | `LocalArena`, `NumaArena` | [arena_local](#arena_local), [arena_numa](#arena_numa) |
| Topology latency | `TopologyLatencyTable`, `topology_latency_table` | [numa_latency](#numa_latency) |

## `arena`

The MIMD entry points. Defined in [`src/sched/arena.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/arena.rs).

### `join`

```rust
pub fn join<A, B, RA, RB>(plan: &JobPlan, a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
```

Two-way fork-join. Both closures run concurrently on the work-stealing arena when `pick_tier` selects `Local` (or higher); otherwise they run serially in caller order. The returned tuple is in caller-supplied order regardless of which thread executed which half.

### `join_context`

```rust
pub fn join_context<A, B, RA, RB>(plan: &JobPlan, a: A, b: B) -> (RA, RB)
where
    A: FnOnce(bool) -> RA + Send,
    B: FnOnce(bool) -> RB + Send,
    ...
```

Variant of `join` that exposes the migrated / stolen flag to each closure. `a` is called with `injected: bool` (true exactly when this entire `join_context` was cold-injected from outside the worker pool); `b` is called with `stolen: bool` (true exactly when `b` was dequeued and executed by a peer worker). The flag is the key signal for adaptive splitters.

### `join_default`

```rust
pub fn join_default<A, B, RA, RB>(
    k_outer: u8,
    batch_size: u32,
    a: A,
    b: B,
) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
```

Convenience: builds `JobPlan::new(k_outer, batch_size)` and calls `join`. Use this when you do not need to customize `hw_class` / `variant` / `numa_hint`.

### `global_local_arena`

```rust
pub fn global_local_arena() -> &'static Arc<NumaArena>
```

Lazily-initialized process-global NUMA-aware arena. On single-NUMA hosts this is a single sub-arena; on multi-NUMA hosts (Genoa, dual-socket Xeon / Threadripper) it has one sub-arena per NUMA node, each pinned to its node's CPUs.

## `par_iter`

The bisecting parallel-iterator family. Defined in [`src/sched/par_iter.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/par_iter.rs).

### `for_each_chunk`

```rust
pub fn for_each_chunk<T, F>(plan: &JobPlan, items: &mut [T], op: F)
where
    T: Send,
    F: Fn(&mut [T]) + Sync,
```

Recursively bisect `items` and apply `op` to each leaf chunk in parallel. Leaf-size floor is `MIN_LEAF_ITEMS = 256`. Uses the SLAW ("Scalable Locality-aware Adaptive Work-stealing", Guo-Zhao-Cavé-Sarkar, IPDPS 2010) adaptive splitter: more splits where there is observed contention, fewer where workers are saturated.

### `for_each_fixed_chunk`

```rust
pub fn for_each_fixed_chunk<T, F>(plan: &JobPlan, items: &mut [T], chunk_size: usize, op: F)
```

Variant with a fixed chunk size instead of the adaptive floor. Useful when the caller knows a SIMD-optimal chunk shape (e.g., AVX-512 16-lane FpN<16> = 64 items per chunk to fill 16 ZMM registers x 4 lanes each).

### `for_each_chunk_triple` and `for_each_chunk_triple_min_leaf`

```rust
pub fn for_each_chunk_triple<T1, T2, T3, F>(
    plan: &JobPlan,
    out: &mut [T1],
    a: &[T2],
    b: &[T3],
    op: F,
)
```

Apply `op` to every chunk-triple of `(out, a, b)` in parallel. All three slices must have the same length. Used for `out = f(a, b)` slice kernels (mul_slice / add_slice / sub_slice). The `_min_leaf` variant takes a caller-supplied leaf floor (use `min_leaf = 1` for heavy per-element work like row-update or per-row spmv).

### `for_each_chunk_indexed` and `for_each_chunk_indexed_min_leaf`

```rust
pub fn for_each_chunk_indexed<T, F>(plan: &JobPlan, items: &mut [T], op: F)
```

Indexed-collect pattern: closure receives `(start_idx, &mut [T])` so the body can know the absolute slot index. The `_min_leaf` variant takes a caller-supplied floor (use `min_leaf = 1` for heavy per-element work like matmul O(k), spmv O(nnz_per_row), LU row update, Jacobi rotation, etc.).

### `collect_indexed` and variants

```rust
pub fn collect_indexed<R, F>(plan: &JobPlan, n: usize, min_leaf: usize, f: F) -> Vec<R>
```

Parallel `(0..n).map(f).collect()` with `MaybeUninit` backing. Variants:

- `collect_indexed_heartbeat`: rdtsc-polling heartbeat splitter.
- `collect_indexed_token_bucket`: token-bucket promotion (work amplifies as steals fire).
- `collect_indexed_tiny_tasks`: Tiny-Tasks (Acar 2013) cost-model splitter, uses `plan.optimal_chunk_count`.

### `reduce_chunks`

```rust
pub fn reduce_chunks<T, A, F, R, I>(
    plan: &JobPlan,
    items: &[T],
    init: I,
    fold: F,
    reduce: R,
) -> A
```

Map-fold-reduce: each leaf chunk folds into an accumulator `A`; chunk-level accumulators reduce pairwise into a single result. The familiar Rayon-shape parallel reduction.

## `k_join`

Const-generic K-recursion driver. Defined in [`src/sched/k_join.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/k_join.rs).

### `k_join`

```rust
pub fn k_join<const K: u32, A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where ...
```

At `K <= 4` the function monomorphizes to inline serial `(a(), b())` - no scheduler call, no `JobPlan` allocation, zero overhead. At `K >= 5` it builds a default `JobPlan::new(K as u8, 1)` and delegates to [`arena::join`](#join).

### `k_join_with_plan`

```rust
pub fn k_join_with_plan<const K: u32, A, B, RA, RB>(
    plan: &JobPlan, a: A, b: B,
) -> (RA, RB)
```

Same const-K dispatch but takes an explicit `JobPlan`. Useful when the caller needs to pin `hw_class` / `variant` / `numa_hint`.

Use these when the K parameter is compile-time-known (Karatsuba / NTT / Burnikel-Ziegler recursion): the `K <= 4` branch folds to zero overhead via const propagation.

## `cooperative`

The SIMC and MIMC entry point. Defined in [`src/sched/cooperative.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/cooperative.rs).

### `cooperative_join_n`

```rust
pub fn cooperative_join_n<R>(
    plan: &JobPlan,
    closures: Vec<Box<dyn FnOnce() -> R + Send>>,
) -> Vec<R>
```

N-way fork-join. Returns results in caller-supplied order, invariant of which thread executed which closure.

Implementation: at `N = 1` runs inline; at `N = 2` invokes [`arena::join`](#join); at `N >= 3` splits the list in half (left-biased on odd N) and recurses via `arena::join` on the two subtrees. The shape is deterministic given N; it does not depend on available worker count.

#### SIMC vs MIMC usage

This single primitive covers two distinct Flynn axes depending on what the caller passes in:

- **SIMC (Single Instruction, Multiple Cores)** - all N closures call the same function body over different inputs. Canonical reduction pattern; every closure produces the same kind of partial result that aggregates trivially.
- **MIMC (Multiple Instruction, Multiple Cores)** - the N closures carry different function bodies, one per role. The "Multiple Instruction" axis of Flynn's taxonomy. Canonical pattern in numerical linear algebra ("one closure factors the pivot, N closures apply it"), in MCTS-style search ("one closure picks the leaf, N closures backprop value"), in Bayesian inference ("one closure proposes, N closures evaluate likelihood components in parallel").

The primitive does not distinguish the two; it accepts any `Vec<Box<dyn FnOnce() -> R + Send>>` and runs them as one cooperative call. The Flynn-axis label is determined by what the caller chose to put inside the boxes.

#### Example: SIMC (identical closures, partial-sum reduce)

```rust
use flynnel::{JobPlan, cooperative_join_n};

let data: Vec<f64> = (0..1_000_000).map(|i| (i as f64).sqrt()).collect();
let plan = JobPlan::new(8, 1024);
let n_lanes = 16;
let chunk = data.len() / n_lanes;
let arc = std::sync::Arc::new(data);

let closures: Vec<Box<dyn FnOnce() -> f64 + Send>> = (0..n_lanes)
    .map(|i| {
        let d = std::sync::Arc::clone(&arc);
        let lo = i * chunk;
        let hi = if i == n_lanes - 1 { d.len() } else { lo + chunk };
        Box::new(move || d[lo..hi].iter().sum::<f64>())
    })
    .collect();

let partials = cooperative_join_n(&plan, closures);
let total: f64 = partials.iter().sum();
```

All 16 closures run the same function body (`d[lo..hi].iter().sum`) over different ranges. SIMC.

#### Example: MIMC (heterogeneous closures, role specialization)

```rust
use flynnel::{JobPlan, cooperative_join_n};

let plan = JobPlan::new(8, 1024);
let data = std::sync::Arc::new(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);

let d_a = std::sync::Arc::clone(&data);
let d_b = std::sync::Arc::clone(&data);
let d_c = std::sync::Arc::clone(&data);
let d_probe = std::sync::Arc::clone(&data);

// Wrap each role's result in an enum so the closures can return
// different concrete values via a single shared return type.
#[derive(Debug)]
enum Role { Sum(f64), Max(f64), Count(usize), Probe(bool) }

let closures: Vec<Box<dyn FnOnce() -> Role + Send>> = vec![
    Box::new(move || Role::Sum(d_a.iter().sum())),                       // role A
    Box::new(move || Role::Max(d_b.iter().cloned().fold(f64::MIN, f64::max))), // role B
    Box::new(move || Role::Count(d_c.iter().filter(|&&x| x > 2.0).count())),   // role C
    Box::new(move || Role::Probe(d_probe.iter().all(|&x| x.is_finite()))),     // role D (calibration probe)
];

let results = cooperative_join_n(&plan, closures);
// results[0] = Role::Sum(15.0), results[1] = Role::Max(5.0), etc.
```

Four closures, four distinct roles, one cooperative sync boundary. MIMC.

See [`benches/flynn_axes.rs`](https://github.com/markusmcnugen/flynnel/blob/main/benches/flynn_axes.rs) for the criterion benches that exercise the 4-way heterogeneous reduce and the pivoted-LU step MIMC shapes, and [`examples/dispatcher_per_axis.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/dispatcher_per_axis.rs) for a runnable MIMC walk-through through the AdaptiveDispatcher.

## `hybrid`

The MIMT entry point. Defined in [`src/sched/hybrid.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/hybrid.rs).

### `join_hybrid`

```rust
pub fn join_hybrid<RA, RB, A, B>(plan: &JobPlan, cpu_work: A, gpu_work: B) -> (RA, RB)
where
    A: FnOnce() -> RA,
    B: FnOnce() -> RB + Send + 'static,
    RA: 'static,
    RB: Send + 'static,
```

Runs `cpu_work` on the calling thread and `gpu_work` on whichever backend [`JobPlan::pick_backend`](JobPlan-Reference.md#pick_backend) selects. Results return as `(cpu_result, gpu_result)` in caller-supplied order.

Use this for ONE pair of independent CPU + GPU closures that should run concurrently. For iterated coupled workloads where the GPU half consumes outputs the CPU half produced AND feeds the next CPU stage, see [`hybrid_pipeline`](#hybrid_pipeline) instead.

Panic propagation: either half panicking propagates through. A panic from the GPU half is captured on the spawned thread and re-raised on the calling thread after the CPU half completes.

#### Example

```rust
use flynnel::{JobPlan, Backend, join_hybrid};

let plan = JobPlan::new(8, 1024).with_backend(Backend::Cuda { device_id: 0 });
let (cpu_sum, gpu_sum) = join_hybrid(
    &plan,
    || (0..512u64).sum::<u64>(),
    || (512..1024u64).sum::<u64>(),
);
assert_eq!(cpu_sum + gpu_sum, (0..1024u64).sum::<u64>());
```

### `hybrid_pipeline`

```rust
pub fn hybrid_pipeline<I, F1, F2, F3, A, B, R>(
    plan: &JobPlan,
    inputs: I,
    pre_cpu: F1,
    gpu: F2,
    post_cpu: F3,
) -> Vec<R>
where
    I: IntoIterator + Send + 'static,
    I::Item: Send + 'static,
    F1: FnMut(I::Item) -> A + Send + 'static,
    F2: FnMut(A) -> B + Send + 'static,
    F3: FnMut(B) -> R + Send + 'static,
    A: Send + 'static,
    B: Send + 'static,
    R: Send + 'static,
```

Streaming three-stage CPU - GPU - CPU pipeline. Each input flows through `pre_cpu` -> `gpu` -> `post_cpu`, and the three stages run on dedicated OS threads connected by depth-2 bounded channels so stage[N+1] of an earlier pipeline position overlaps stage[N] of a later one. After pipeline-fill the steady-state throughput is `1 / max(t_pre_cpu, t_gpu, t_post_cpu)` per input - the smaller stages hide entirely behind the largest.

This is the algorithmically meaningful MIMT shape: the GPU half consumes outputs the prior CPU stage produced AND feeds the next CPU stage. Compare against [`join_hybrid`](#join_hybrid), which only handles one (cpu, gpu) pair and does not pipeline across iterations.

Channel depth: each inter-stage channel is bounded at 2 (the "ping-pong" depth). The producer stays one item ahead of the consumer without unbounded buffering. Pipeline-fill cost is one stage time; steady-state throughput is bounded by the slowest stage.

Backend hint: `plan.backend_hint` is informational only here - the `gpu` closure body decides which backend (and which CUDA stream) it actually invokes. Pass an `Arc<YourBackend>` into the closure via `move` capture.

Panic propagation: a panic on any stage thread propagates to the calling thread when that stage's `join` is awaited. The other stages run to completion before the panic is re-raised so partial results up to the panic point are not lost in-flight.

#### Theoretical speedup

The ceiling for a 3-stage pipeline with stages of cost `a, b, c` is `(a+b+c) / max(a, b, c)`. For perfectly balanced stages (`a = b = c`) that's 3x. For an extreme one-stage dominator (e.g., `c >> a, b`), the speedup approaches 1x. Pushing toward 3x is purely a question of stage balance: match the CPU stage cost to the GPU stage cost.

#### When to use

- WIN: the algorithm has a CPU -> GPU -> CPU dependency chain repeated per iteration (MCMC, batched CG, MCTS-with-NN-eval).
- WIN: the three stages are within ~3x of each other in cost.
- WIN: the number of iterations is large enough that pipeline-fill (one stage time) amortizes (~8+ iterations).
- LOSS: when the GPU stage dominates by 10x or more, the pipelining win shrinks to ~10%.
- LOSS: when iterations have intra-iter cross-stage dependencies (the CPU stage cannot start until the GPU returns AND the GPU cannot start until the CPU returns for the SAME iteration). Use [`join_hybrid`](#join_hybrid) for that one-pair shape.

#### Example (Metropolis MCMC shape)

```rust
use std::sync::Arc;
use flynnel::{JobPlan, hybrid_pipeline};

let plan = JobPlan::new(8, 1024);

// CPU stage 1: adaptive proposal generation (branchy).
let pre_cpu = |seed: u64| -> Vec<f32> {
    (0..1024)
        .map(|i| ((seed * 1103515245 + 12345 + i as u64) & 0xFFFF) as f32 / 65536.0)
        .collect()
};

// GPU stage: capture backend handle here and dispatch a kernel.
// Synthetic stand-in (real call would invoke CudaBackend).
let gpu = |proposal: Vec<f32>| -> (Vec<f32>, f32) {
    let loglik: f32 = proposal.iter().map(|x| -x * x).sum();
    (proposal, loglik)
};

// CPU stage 3: Metropolis accept/reject + adaptive step update.
let post_cpu = |state: (Vec<f32>, f32)| -> f32 {
    let (_proposal, loglik) = state;
    if loglik > -100.0 { loglik } else { -100.0 }
};

let accepted: Vec<f32> = hybrid_pipeline(&plan, 0..16u64, pre_cpu, gpu, post_cpu);
assert_eq!(accepted.len(), 16);
```

See [`benches/mimt_coupled.rs`](https://github.com/markusmcnugen/flynnel/blob/main/benches/mimt_coupled.rs) for three full coupled-algorithm bench files (MCMC, batched CG, AlphaZero-shape MCTS) and the [Benchmarks page](Benchmarks.md#mimt-pipelined---coupled-algorithm-benchmarks) for measured speedups (Metropolis MCMC 1.96x, Batched CG 2.35x, MCTS 1.84x on Zen+/16T + RTX 3070).

### `hybrid_auto`

```rust
pub fn hybrid_auto<R, C, G>(plan: &JobPlan, cpu_impl: C, gpu_impl: G) -> (R, Placement)
where
    C: FnOnce() -> R,
    G: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
```

Learned CPU-vs-backend placement for two implementations of the SAME computation. The decision keys on the caller-location-resolved `CallSiteState` (or the caller's `plan.site` attachment) bucketed by log2 of `batch_size`:

- **Cold bucket**: races both sides via [`join_hybrid`](#join_hybrid), timing each end-to-end (queueing + launch + execution). Racing IS the calibration; the CPU result is returned, so the race costs one redundant computation rather than a stalled pipeline.
- **Warm bucket**: routes to whichever side holds the lower EWMA and times just that side, keeping the model current.
- **Reprobe**: every 32nd call in a bucket re-races so a shifted workload or freed-up device gets re-measured.

Returns `(result, Placement)` where [`Placement`](#call_site) reports `Cpu`, `Backend`, or `Race`. End-to-end wall time on the calling side is the ONLY signal; there is no data-residency model, so transfer cost is captured implicitly in the measured backend time. E2E walkthrough: [`examples/hybrid_auto_demo.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/hybrid_auto_demo.rs).

### `hybrid_auto_split`

```rust
pub fn hybrid_auto_split<T, CF, GF>(
    plan: &JobPlan,
    items: &mut [T],
    cpu_impl: CF,
    backend_impl: GF,
) -> SplitReport
where
    T: Send + 'static,
    CF: FnOnce(&mut [T]),
    GF: FnOnce(&mut [T]) + Send + 'static,

pub struct SplitReport {
    pub cpu_items: usize,
    pub backend_items: usize,
    pub cpu_ns: u64,
    pub backend_ns: u64,
    pub cpu_share_per_mille: u32,
}
```

Data-parallel variant: splits `items` between the CPU and the backend at a LEARNED share (per-mille of items to the CPU, clamped to 50..=950; 500 when cold), runs both halves concurrently, and updates per-item throughput EWMAs from the measured halves so the next call's split tracks the observed speed ratio. Blocks until both halves finish (the backend half borrows from the same slice). The returned `SplitReport` carries the realized split and per-side timings.

## `race`

The MISD entry point. Defined in [`src/sched/race.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/race.rs).

### `race_variants`

```rust
pub fn race_variants<R, Ff, Fa, Fc>(
    plan: &JobPlan,
    fast: Ff,
    faithful: Fa,
    correct: Fc,
) -> (R, Variant)
where
    R: Send + Sync + 'static,
    Ff: FnOnce(&CancelToken) -> Option<R> + Send,
    Fa: FnOnce(&CancelToken) -> Option<R> + Send,
    Fc: FnOnce(&CancelToken) -> R + Send,
```

Race three variant implementations in parallel:

- `fast`: returns `None` if the fast tier cannot meet its tolerance for this input.
- `faithful`: returns `None` if the faithful tier cannot meet its tolerance.
- `correct`: always succeeds; the final safety net.

Each closure receives a [`CancelToken`](#canceltoken) it can poll at checkpoints (between Newton iterations, between NTT butterfly layers, between Karatsuba recursion levels) to abandon early when a peer has won.

Returns `(R, Variant)` where the `Variant` tag identifies which tier produced the result. The Correct closure always runs to completion (it ignores cancel); cancel only affects Fast / Faithful losers.

### `explore_select`

```rust
pub fn explore_select<R, F, B>(
    plan: &JobPlan,
    n: usize,
    explore: F,
    better: B,
) -> Option<(usize, R)>
where
    R: Send,
    F: Fn(usize) -> R + Sync,
    B: Fn(&R, &R) -> bool,
```

The COMPLEMENT of [`race_variants`]. Where `race_variants` is first-past-the-post (fastest tolerable result cancels the losers - MISD speculation), `explore_select` runs EVERY explorer to completion (nothing cancels) and picks the winner by RESULT QUALITY. This is the correct contract for episode racing / population search: N independent explorers each run a full trajectory and the best-scoring one is kept - fewest actions, highest reward - regardless of which finished first. First-past-the-post would discard a better-but-slower explorer; `explore_select` keeps it.

- `explore(i)` produces explorer `i`'s result for `i` in `0..n`; the index seeds per-explorer state (clone id, RNG stream).
- `better(a, b)` returns `true` when `a` is STRICTLY better than `b`, fully defining "best" (argmin, argmax, lexicographic, tie-break) without float-`Ord` friction. Ties keep the earlier index, so the winner is deterministic given deterministic explorers.

Returns the winning `(index, result)`, or `None` when `n == 0`. Each explorer is one leaf (`min_leaf = 1`) so N heavy trajectories fan out fully even at small N. E2E walkthrough: [`examples/explore_select_demo.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/explore_select_demo.rs).

### The racing family

`race_variants` and `explore_select` are two answers to one question: launch several trials, then decide something from how they turn out. Racing has more answers than those two. Each primitive below fixes a different combination of three choices - when the ensemble stops, what it returns, and what the losers do - and each matches a distinct kind of work. The [`racing_zoo_demo`](https://github.com/markusmcnugen/flynnel/blob/main/examples/racing_zoo_demo.rs) runs all of them against known-correct answers.

### `race_any`

```rust
pub fn race_any<P, F>(plan: &JobPlan, n: usize, attempt: F) -> Option<(usize, P)>
where P: Send + Sync + 'static, F: Fn(usize, &CancelToken) -> P + Sync,
```

The tail-latency move: fire `n` interchangeable attempts and keep the first that finishes. When latency is variable - one of several replicas, mirrors, or routes - firing a few and taking whichever returns first trims the slow tail. There is no tolerability predicate and no safety net, which is what sets it apart from `race_variants`; speed is the only thing separating the attempts. Losers poll their [`CancelToken`](#canceltoken) and quit. Returns the winning `(index, result)`.

### `race_quorum`

```rust
pub fn race_quorum<P, F>(plan: &JobPlan, n: usize, k: usize, attempt: F) -> Vec<(usize, P)>
where P: Send + 'static, F: Fn(usize, &CancelToken) -> P + Sync,
```

The quorum read: ask `n` replicas, act on the first `k` to answer, drop the stragglers (they see cancel and stop). `k` clamps to `n`. The `k` winners come back in COMPLETION order, not index order, so the caller learns which replicas were fast.

### `race_refute`

```rust
pub enum Settled<P, R> { Proved(P), Refuted(R), Unsettled }

pub fn race_refute<P, R, FP, FR>(plan: &JobPlan, prove: FP, refute: FR) -> Settled<P, R>
where P: Send + Sync + 'static, R: Send + Sync + 'static,
      FP: FnOnce(&CancelToken) -> Option<P> + Send,
      FR: FnOnce(&CancelToken) -> Option<R> + Send,
```

A duel: two sides chase opposite verdicts and the first to settle wins, cancelling the other. A SAT portfolio is the clean case - one engine hunts a model, the other a proof of unsatisfiability, and whichever lands first ends it. The same shape drives a capability probe: certify a property absent versus witness it present. The two sides return different types (a model is not a refutation), so the verdict carries both; if both give up, it is `Unsettled`.

### `race_agree`

```rust
pub enum Agreement<R> {
    Consensus { value: R, agree: usize, total: usize },
    Split { plurality: usize, total: usize },
}

pub fn race_agree<R, F>(plan: &JobPlan, n: usize, threshold: usize, explore: F) -> Agreement<R>
where R: PartialEq + Send, F: Fn(usize) -> R + Sync,
```

Trust by agreement. Compute a result several ways and believe it only when at least `threshold` of them concur. Where `race_variants` trusts the first tolerable answer, `race_agree` trusts nothing until the votes line up - and it reports when they do not, so a silent divergence between two supposedly-equivalent routines surfaces here instead of downstream. Every explorer runs to completion; you cannot count votes you did not collect. Equality drives the tally, so `R: PartialEq` suffices.

### `race_deadline`

```rust
pub struct Anytime<R> { /* is_expired(), submit(score, value) */ }

pub fn race_deadline<R, F>(plan: &JobPlan, budget: Duration, n: usize, explore: F) -> Option<(f64, R)>
where R: Send, F: Fn(usize, &Anytime<R>) + Sync + Send,
```

Time is the terminator, which is what sets this apart from every other race here. The explorers do not finish; they improve. Think tree search under a move budget or an iterative solver with a latency SLA: you spend the whole budget and take the best answer found within it. Each explorer loops - improve, [`Anytime::submit`], check [`Anytime::is_expired`] - until the clock flips, then returns. Returns the highest-scored published `(score, value)`. One worker parks on the timer for the budget, so the budget being the point is what makes that trade deliberate.

### `race_tournament`

```rust
pub fn race_tournament<R, F, B>(plan: &JobPlan, n: usize, eta: usize, base_budget: u32, run: F, better: B) -> Option<(usize, R)>
where R: Send, F: Fn(usize, u32) -> R + Sync, B: Fn(&R, &R) -> bool,
```

Successive halving. What if running every candidate to completion is too expensive, but the first to finish throws away quality? Spend a little on everyone, prune the losers by interim score, and pour the freed budget into the survivors. Each round keeps `ceil(survivors / eta)` candidates and multiplies the budget by `eta`, so work per round stays roughly flat while budget-per-survivor climbs. `run(id, budget)` runs a candidate fresh at the given budget (Hyperband-style); `better(a, b)` is `true` when `a` beats `b`. Returns the final survivor's `(id, result)`.

### `race_statistical`

```rust
pub struct StatOpts { pub value_range: f64, pub delta: f64, pub batch: usize, pub max_samples: usize, pub maximize: bool }
pub struct StatOutcome { pub winner: usize, pub mean: f64, pub samples_each: usize, pub survivors: usize }

pub fn race_statistical<F>(plan: &JobPlan, n: usize, opts: StatOpts, sample: F) -> Option<StatOutcome>
where F: Fn(usize) -> f64 + Sync,
```

The trials are noisy, so wall-clock and single-result selection both lie - one lucky sample means nothing. Each candidate accumulates samples in rounds, and after each round a Hoeffding bound asks whether a candidate's optimistic estimate is already worse than the leader's pessimistic one. If so, cut it. This picks among noisy estimators - a Monte Carlo variant, a stochastic policy, an A/B arm - without paying the full sample budget for every one. The radius is `value_range * sqrt(ln(2/delta) / (2 * n_samples))`; the race stops at one survivor or when `max_samples` is reached.

### `CancelToken`

```rust
pub struct CancelToken { /* opaque */ }

impl CancelToken {
    pub fn is_cancelled(&self) -> bool;  // cheap atomic load
}
```

### `par_map_in_place`

```rust
pub fn par_map_in_place<T, F>(plan: &JobPlan, items: &mut [T], op: F)
where T: Send, F: Fn(&mut T) + Sync,
```

Apply `op` to each element in parallel, with one dispatched task per element (leaf chunk size = 1). Use this when each element's `op` is large enough (~10 us+) to amortize per-task dispatch overhead. For small per-element work use [`for_each_chunk`](#for_each_chunk) instead, which groups multiple elements per leaf via the bisect splitter.

Examples of one-task-per-element work: per-block high-precision arithmetic, per-row matrix factorisation, per-particle PDE step, per-image GPU dispatch coordinator. The common shape is "few large units" rather than "many small units".

### `par_zip_apply`

```rust
pub fn par_zip_apply<T, U, F>(plan: &JobPlan, lhs: &mut [T], rhs: &[U], op: F)
where T: Send, U: Sync, F: Fn(&mut T, &U) + Sync,
```

Parallel zip-apply: for each index `i`, call `op(&mut lhs[i], &rhs[i])`. Mutates `lhs` in place; `rhs` is read-only. Panics if `lhs.len() != rhs.len()`. One task per index by default (matches `par_map_in_place`'s granule).

Useful when paired indices need disjoint mutable access that the standard slice borrow checker cannot prove (independent per-element ops with no cross-element dependency). Internally routes through `for_each_fixed_chunk` over an index-list with raw-pointer arithmetic to produce disjoint `&mut T` per index.

## `pipeline`

Generic N-stage pipeline. Defined in [`src/sched/pipeline.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/pipeline.rs).

### `PipelineStage` trait

```rust
pub trait PipelineStage<In, Out>: Send + Sync {
    fn process(&self, item: In) -> Out;
}
```

Single transformation `In -> Out`. The trait takes `&self` so stages can hold precomputed state without per-item allocation.

### `FnStage`

```rust
pub struct FnStage<F>;
impl<F: Fn(In) -> Out + Send + Sync> PipelineStage<In, Out> for FnStage<F>
```

Lifts an `Fn(In) -> Out + Send + Sync` closure into a `PipelineStage`. Use `FnStage::new(|x| x + 1)` for ad-hoc stages without defining a struct.

### `run`

```rust
pub fn run<S, T>(stages: &[S], inputs: Vec<T>) -> Vec<T>
where S: PipelineStage<T, T>, T: Send + 'static,
```

Run a sequence of stages over a batch of inputs. Allocates one bounded SPSC crossbeam channel per inter-stage edge and one scoped thread per stage. Returns a `Vec<T>` in caller-supplied input order.

Panics if `stages.is_empty()`. A zero-stage pipeline has no meaningful semantics.

### `run_dyn`

```rust
pub fn run_dyn<S>(
    stages: &[S],
    inputs: Vec<Box<dyn Any + Send>>,
) -> Vec<Box<dyn Any + Send>>
```

Heterogeneous-type variant: every item moves through stages as `Box<dyn Any + Send>`, so each stage may produce a different type. Useful when stages 0..N change the underlying type but the dispatcher does not statically know the chain.

### `par_map_serial_reduce`

```rust
pub fn par_map_serial_reduce<T, U, R, FOp, FCombine>(
    plan: &JobPlan,
    lhs: &mut [T],
    rhs: &[U],
    initial: R,
    op: FOp,
    combine: FCombine,
) -> R
```

Two-stage pipeline: parallel per-element op overlapped with serial in-order reduction. For each index `i`, the parallel stage runs `op(&mut lhs[i], &rhs[i])`; the serial combine stage threads an accumulator through `combine(acc, &lhs[i])` in index order. Wall-clock cost is `max(parallel_total, combine_total)` instead of the naive `parallel_total + combine_total` of a fork-then-fold layout.

Use when the per-element op is independent and the combine is a sequential left-fold that can't itself parallelize (Two-Sum-chain, exact-integer-add-chain, sequential hash absorb, sequential running statistics). For associative combines use [`reduce_chunks`](#reduce_chunks) instead - that path is faster because no sequential dependency blocks the combine.

Panics if `lhs.len() != rhs.len()`.

## `mode_region`

Matrix-extension mode-region wrappers (Intel AMX, ARM SME, NVIDIA Tensor Cores). Defined in [`src/sched/mode_region.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/mode_region.rs).

### `MatrixModeBackend` trait

```rust
pub trait MatrixModeBackend {
    type Config;
    type Context;
    unsafe fn enter(config: &Self::Config) -> Self::Context;
    unsafe fn exit(ctx: Self::Context);
}
```

Per-platform matrix-extension hooks. Implementors provide concrete `enter` / `exit` for their hardware. Both methods are `unsafe` because they invoke instruction-level state changes (LDTILECFG, SMSTART, etc.).

### `run_in_region`

```rust
pub fn run_in_region<B, F, R>(config: &B::Config, op: F) -> R
where B: MatrixModeBackend, F: FnOnce(&mut B::Context) -> R,
```

Safe wrapper that enters the region, runs the closure, and exits via RAII (`Drop` runs even on panic). Callers stay in safe code; the unsafe enter/exit is encapsulated.

### `ScalarFallback` / `ScalarContext` / `ScalarConfig`

The always-available no-op backend so callers can write `run_in_region` code that compiles on every platform, degrading gracefully to plain scalar ops inside the region body.

## `io_pool`

SMT-sibling thread pool for non-compute roles. Defined in [`src/sched/io_pool.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/io_pool.rs).

### `IoPool`

```rust
pub struct IoPool { /* ... */ }
impl IoPool {
    pub fn submit<F: FnOnce() + Send + 'static>(&self, task: F);
    pub fn worker_count(&self) -> usize;
}
```

Off by default. Set `FLYNNEL_SCHED_SMT_AS_IO=on|1|true` to enable. When enabled, parks one worker per physical core on the SMT sibling and uses it for non-compute roles (background calibration, BLAKE3 verification of stripe outputs, prefetch sweeps, GPU event polling, cross-node send/recv).

### `global_io_pool`

```rust
pub fn global_io_pool() -> Option<&'static Arc<IoPool>>
```

Returns `Some(pool)` when `FLYNNEL_SCHED_SMT_AS_IO` is set; `None` otherwise. Callers should run their async work inline on the caller thread when this returns `None`.

### `submit_io_or_inline`

```rust
pub fn submit_io_or_inline<F: FnOnce() + Send + 'static>(task: F)
```

Submit to the IO pool if enabled; run inline on the caller thread otherwise. Sound for tasks independent of the caller's subsequent work.

## Idle-spin window control

An idle worker spins `yield_now` for a window before parking on its
condvar. The default window (500 rounds, ~500us) is tuned to win on
throughput: when the next dispatch lands inside the window the producer
skips the unpark syscall and the worker finds the work on its own. But
for a bursty-idle workload - a short burst, then a long idle - that
spin is wasted CPU, the `sched_yield` a flamegraph flags.

So the window is controllable, off the crate root:

- `set_spin_window(rounds)` forces a short window and stops the
  controller - the explicit lever for a workload known to be
  bursty-idle and latency-insensitive between bursts, so idle workers
  park promptly instead of spinning. This is the CPU analog of pausing
  the GPU poller.
- `set_spin_adaptive(true)` (or `FLYNNEL_ADAPTIVE_SPIN=1`) turns on a
  controller that shrinks the window toward a floor when it sees
  workers parking (the spin was wasted) and grows it back toward the
  default when it sees them rescued mid-spin (the spin paid off). A
  throughput workload keeps the long window because its spins are
  rescued; a bursty one loses it.
- `spin_window()` reads the current window; `total_idle_yields()` and
  `reset_spin_stats()` expose and reset the idle-yield count for
  measuring a phase.

Both are off by default: the default stays exactly the tuned 500 with
no controller, so existing throughput code does not regress; a bursty
workload opts in. Measured (RTX-host 16-thread, 120-burst bursty
workload): the forced short window cut idle yields 3.2x, the adaptive
controller 2.1x while converging the window to its floor. E2E:
[`examples/adaptive_spin_demo.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/adaptive_spin_demo.rs).

## `bg_calibration`

Background per-host calibration on the IO pool. Defined in [`src/sched/bg_calibration.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/bg_calibration.rs).

### `timed_avg_ns`

```rust
pub fn timed_avg_ns<F: FnMut()>(op: F, iters: u32) -> f64
```

Run `op` 8 warmup + `iters` measured times; return average per-call wall time in nanoseconds.

### `spawn_calibration`

```rust
pub fn spawn_calibration(closures: Vec<Box<dyn FnOnce() + Send + 'static>>)
```

Submit a list of calibration microbenches to the IO pool. No-op when the IO pool is disabled.

## `prefetch`

Software prefetch hints. Defined in [`src/sched/prefetch.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/prefetch.rs).

```rust
pub fn prefetch_into_l3<T>(slice: &[T])
pub fn prefetch_into_l2<T>(slice: &[T])
pub fn prefetch_into_l3_inline<T>(slice: &[T])
pub fn prefetch_into_l2_inline<T>(slice: &[T])
```

The non-inline variants submit to the IO pool (async prefetch on the SMT sibling); inline variants emit prefetch instructions on the calling thread. Used by the worker pool to warm captured-state cache lines on successful steals.

## `numa_alloc`

Cross-platform NUMA-aware page allocator. Defined in [`src/sched/numa_alloc.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/numa_alloc.rs).

### `NumaAlloc`

Stateless allocator helper. Reads `numa_topology()` and emits per-node allocation requests via `numa_alloc_onnode` (Linux) or `VirtualAllocExNuma` (Windows). macOS and other platforms fall back to plain heap allocation.

### `NUMA_NODE_LOCAL`

Sentinel constant (`u32::MAX`) meaning "use the current thread's NUMA node."

## `bg_zero`

Background memory zeroing for next-op allocation. Defined in [`src/sched/bg_zero.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/bg_zero.rs).

### `prepare`

```rust
pub fn prepare(n_bytes: usize) -> Handle
```

Submit a request to allocate + first-touch a `Vec<u8>` of size `n_bytes` on the current thread's NUMA node. Returns a `Handle` the caller can later `.take()` to receive the prepared buffer. When the IO pool is enabled the work runs in the background; otherwise it runs inline on caller demand.

## `verify_chain` (feature `verify-chain`)

BLAKE3-rooted hash chain for per-stripe verification. Defined in [`src/sched/verify_chain.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/verify_chain.rs).

### `VerifyHasher` trait

```rust
pub trait VerifyHasher: Send + 'static {
    fn update(&mut self, chunk: &[u8]);
    fn finalize(self: Box<Self>) -> [u8; 32];
}
```

### `default_hasher`

Returns `Box::new(Blake3Hasher::new())` when `verify-chain` is enabled; otherwise returns `Box::new(FxFallbackHasher::new())` (a fast non-cryptographic FxHash-style fallback).

### `VerifyChain`

```rust
pub struct VerifyChain { /* ... */ }
impl VerifyChain {
    pub fn new() -> Self;
    pub fn submit_chunk(&self, chunk: Vec<u8>);
    pub fn finalize(self) -> [u8; 32];
}
```

Running hash chain over a sequence of stripe outputs. `submit_chunk` is non-blocking when the IO pool is enabled (the hash update happens on the SMT sibling). `finalize` blocks until every submitted chunk has been hashed and returns the 32-byte root.

## `split_observer`

Runtime tuning of the SLAW split-budget multiplier. Defined in [`src/sched/split_observer.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/split_observer.rs).

### `split_multiplier` / `set_split_multiplier`

```rust
pub fn split_multiplier() -> u32
pub fn set_split_multiplier(value: u32)
```

The multiplier used by [`for_each_chunk`](#for_each_chunk) for the initial split budget (`workers * multiplier`). Defaults to 2. Higher means more aggressive subdivision (better steal granularity at the cost of dispatch overhead).

### `record_leaf_time_ns` / `snapshot_leaf_stats` / `reset_leaf_stats`

Per-leaf timing observability. Workers call `record_leaf_time_ns` after each leaf chunk completes; the observer reads `snapshot_leaf_stats` periodically.

### `spawn_observer`

```rust
pub fn spawn_observer()
```

Spawn the background observer on the IO pool (no-op when IO pool disabled). The observer samples steal-rate and leaf-time CV every interval and updates `split_multiplier` accordingly.

### `LeafStats`

```rust
pub struct LeafStats { /* mean, variance, sample count */ }
```

Aggregate leaf-time statistics produced by `snapshot_leaf_stats`.

## `idempotent`

Marker trait for jobs safe to execute more than once. Defined in [`src/sched/idempotent.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/idempotent.rs).

```rust
pub trait IdempotentJob: Send + Sync {
    type Output;
    fn run(&self, start: usize, end: usize) -> Self::Output;
}
pub fn run_idempotent<J: IdempotentJob>(job: &J, start: usize, end: usize) -> J::Output
```

Pairs with fence-free work-stealing variants where the cross-worker race can lose a steal-vs-pop conflict by executing the same job twice rather than blocking. Most op bodies are not idempotent (they mutate output buffers or increment counters); pure-read folds and write-once `MaybeUninit` indexed-collect bodies can opt in.

## `chase_lev_local`

In-house Chase-Lev work-stealing deque. Defined in [`src/sched/chase_lev_local.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/chase_lev_local.rs). Replaces the prior dependency on `crossbeam::deque::{Worker, Stealer, Steal}`; same wait-free single-owner LIFO + thief-side FIFO steal per Vafeiadis et al. (arXiv:2309.03642), generic over the slot type `T`.

### `new_chase_lev`

```rust
pub fn new_chase_lev<T>(capacity: usize) -> (Worker<T>, Stealer<T>)
```

Construct a deque with `capacity` slots (rounded up to next power of two, minimum 2). Returns the owner half + one thief handle; call `Worker::stealer()` to clone additional thief handles.

### `Worker<T>`

Owner half. Single-owner-writer: only ONE thread may hold a `Worker<T>` at a time.

- `Worker::push(item) -> Result<(), T>` - LIFO push. Returns `Err(item)` on capacity overflow (the in-house deque is BOUNDED; the prior crossbeam Worker grew dynamically).
- `Worker::pop() -> Steal<T>` - LIFO pop with the SeqCst-fence + CAS-on-top single-item race resolution.
- `Worker::stealer() -> Stealer<T>` - clone a thief handle.
- `Worker::is_empty() -> bool` - approximate snapshot.
- `Worker::len() -> usize` - approximate `bottom - top` snapshot.
- `Worker::capacity() -> usize` - rounded-up capacity (always power of two).
- `unsafe fn Worker::slot_ptr(idx) -> *const T` - raw slot pointer for prefetch wiring. The in-house implementation exposes this; the prior crossbeam Worker did not.

### `Stealer<T>`

Thief handle. Clonable; any number of thieves can hold one. Concurrent `steal()` calls race via the per-CAS linearization on `top`.

### `Steal<T>` enum

Three-arm outcome of `Worker::pop` and `Stealer::steal`: `Success(T)` / `Empty` / `Retry`. Mirrors the prior `crossbeam::deque::Steal` 1:1 so call sites swap with no semantic change.

## `flynnel_ring`

Bounded MPMC Vyukov per-slot-sequence ring. Defined in [`src/sched/flynnel_ring.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/flynnel_ring.rs).

```rust
pub struct FlynnelRing<T: Send> { /* opaque */ }

pub enum PushResult<T> { Ok, Full(T) }
pub enum PopResult<T>  { Ok(T), Empty }
```

- `FlynnelRing::new(capacity) -> Self` - rounded up to next pow2, minimum 2.
- `FlynnelRing::push(item) -> PushResult<T>` - try-push; CAS-loop on slot sequence.
- `FlynnelRing::pop() -> PopResult<T>` - try-pop; CAS-loop on slot sequence.
- `FlynnelRing::push_blocking(item)` - infallible push; spins via `spin_loop` on full.
- `FlynnelRing::pop_blocking() -> T` - blocking pop; spins on empty.
- `FlynnelRing::len() -> usize`, `is_empty() -> bool`, `capacity() -> usize` - snapshot helpers.

Used as the per-worker mailbox in [`arena_local`](#arena_local) (replaced `crossbeam::queue::ArrayQueue`) and as the backing for [`injector::Injector`](#injector) and [`notify_ring::NotifyHub`](#notify_ring).

## `flynnel_ring_spsc`

Single-producer single-consumer Lamport ring with zero CAS. Defined in [`src/sched/flynnel_ring_spsc.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/flynnel_ring_spsc.rs).

```rust
pub fn new_spsc<T: Send>(capacity: usize) -> (Producer<T>, Consumer<T>)

pub enum SpscPushResult<T> { Ok, Full(T) }
pub enum SpscPopResult<T>  { Ok(T), Empty }
```

Producer Release-stores `tail`; consumer Release-stores `head`; counters live on separate cache lines. The Acquire/Release pair synchronizes the slot data write with the consumer's slot data read. Faster than `FlynnelRing` for SPSC because the MPMC ring's per-slot CAS is pure overhead when only one producer and one consumer race.

## `flynnel_ring_mpsc`

Multi-producer single-consumer ring. CAS on the producer side, no CAS on the consumer side. Defined in [`src/sched/flynnel_ring_mpsc.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/flynnel_ring_mpsc.rs).

```rust
pub fn new_mpsc<T: Send>(capacity: usize) -> (MpscProducer<T>, Consumer<T>)

pub enum MpscPushResult<T> { Ok, Full(T) }
pub enum MpscPopResult<T>  { Ok(T), Empty }
```

`MpscProducer<T>` is `Clone` and `Send`; any number of producer threads can hold one. The single consumer reads via `Release-store` on `head` with no contention.

## `flynnel_ring_composed`

N-by-M Lamport SPSC grid: MPMC built from `N * M` per-pair SPSC rings instead of one Vyukov MPMC. Per-producer FIFO preserved; global FIFO traded for per-op throughput. Defined in [`src/sched/flynnel_ring_composed.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/flynnel_ring_composed.rs).

```rust
pub fn new_composed_mpsc<T: Send>(n_producers: usize, capacity_per_producer: usize) -> ComposedMpsc<T>
pub fn new_composed_mpmc<T: Send>(n_producers: usize, n_consumers: usize, capacity_per_pair: usize) -> ComposedMpmc<T>

pub struct ComposedMpsc<T: Send> {
    pub producers: Vec<SpscProducer<T>>,
    pub consumer: ComposedMpscConsumer<T>,
}

pub struct ComposedMpmc<T: Send> {
    pub producers: Vec<GridProducer<T>>,
    pub consumers: Vec<GridConsumer<T>>,
}
```

The MPSC variant fits the per-worker mailbox shape (N producers -> 1 consumer); the MPMC grid fits the cross-process dispatch surface (N producers -> M consumers, each producer picks a target consumer round-robin).

Headline measured win on 44T Genoa Heavy/100k MPMC: composed grid 21.08 M/s vs Vyukov 9.81 M/s = **2.15x faster**; buffer-normalized 19.44 M/s vs 9.81 M/s = **1.98x faster** (proves the architectural win, not buffer-disparity). See [Benchmarks](Benchmarks.md#scheduler-primitive-isolation-injector-hot-path) for the full table.

## `injector`

Global MPMC fork queue. External submitters push jobs here; arena workers steal from it when their local deque is empty. Defined in [`src/sched/injector.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/injector.rs).

```rust
pub struct Injector<T: Send> { /* wraps FlynnelRing<T> */ }

pub enum InjectorSteal<T> { Success(T), Empty, Retry }

pub const DEFAULT_INJECTOR_CAPACITY: usize = 4096;
```

- `Injector::new() -> Self` - default 4096 slots.
- `Injector::with_capacity(cap) -> Self` - explicit capacity (rounded up to pow2).
- `Injector::push(item)` - infallible; spins via `spin_loop` on capacity overflow.
- `Injector::try_push(item) -> Result<(), T>` - non-blocking push; `Err(item)` on full.
- `Injector::steal() -> InjectorSteal<T>` - same three-arm Success / Empty / Retry shape as the local deque steal.
- `Injector::len() -> usize`, `is_empty() -> bool`, `capacity() -> usize` - snapshot helpers.

In-house implementation of the global injector. Same three-arm `Success` / `Empty` / `Retry` steal contract as the local Chase-Lev deque. Reduced wrapper overhead vs the prior `crossbeam::deque::Injector` dev-dependency; the `sched_overhead_isolation` bench measures this path on the `flynnel_1leaf_per_worker_w100/100000` cell.

## `notify_ring`

Blocking notify-wrapper over `FlynnelRing` + per-consumer `Parker`. Gives the standard channel surface (`send` / `recv` / `close`) without depending on `crossbeam::channel` or `std::sync::mpsc`. Defined in [`src/sched/notify_ring.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/notify_ring.rs).

```rust
pub struct NotifyHub<T: Send> { /* opaque */ }
pub struct NotifySender<T: Send> { /* opaque, Clone */ }
pub struct NotifyReceiver<T: Send> { /* opaque, single-owner */ }
pub struct NotifyShutdownOnDrop<T: Send> { /* RAII panic-safety guard */ }

pub enum NotifySendResult<T>    { Ok, Closed(T) }
pub enum NotifyTrySendResult<T> { Ok, Full(T), Closed(T) }
```

- `NotifyHub::new(capacity, n_consumers) -> Self` - allocates the ring + a fixed-size `Box<[OnceLock<Arc<Parker>>]>` of length `n_consumers`. The fixed-size pre-allocation means the wake path is Mutex-free.
- `NotifyHub::sender() -> NotifySender` - clone a producer handle.
- `NotifyHub::register_consumer() -> NotifyReceiver` - call from the consumer thread; allocates a `Parker` capturing `thread::current()`, claims the next parker slot, returns the receiver.
- `NotifyHub::shutdown_on_drop() -> NotifyShutdownOnDrop` - RAII guard that calls `shutdown` on drop. Hold one inside each stage thread so panic-unwind triggers shutdown automatically (replaces the panic-safety crossbeam channels gave via `Sender::Drop`).
- `NotifySender::send(item) -> NotifySendResult` - spin-on-full push + wake one consumer (round-robin).
- `NotifyReceiver::recv() -> Option<T>` - blocking pop; spins via `Parker::park_until` then enters the kernel park path. Returns `None` when the hub is shut down AND drained.

Used by the IO pool ([`io_pool`](#io_pool)), the hybrid pipeline ([`hybrid::hybrid_pipeline`](#hybrid_pipeline)), and the reference backends' per-worker dispatch ring (CUDA, WASM).

## `call_site`

Per-call-site adaptive state. Defined in [`src/sched/call_site.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/call_site.rs).

```rust
pub struct CallSiteState { /* all-atomic, const-init */ }
pub struct SiteRef(/* &'static CallSiteState with pointer-identity Eq/Hash */);
pub enum PolicyArm { Default, Alternative }
pub enum Placement { Cpu, Backend, Race }

pub fn caller_site() -> SiteRef;  // #[track_caller]
pub fn site_for_location(loc: &'static std::panic::Location<'static>) -> SiteRef;
```

`CallSiteState` is the identity that makes the adaptive machinery PER-WORKLOAD instead of process-global. Every generic dispatch entry (`for_each_chunk` and family) is `#[track_caller]`; at entry it resolves `std::panic::Location::caller()` - the (file, line, column) of the USER's call - and maps it to a `&'static CallSiteState` via `caller_site()` / `site_for_location()`. The registry is a read-mostly `RwLock<HashMap>` fronted by a per-thread one-slot cache keyed on the `Location` address, so a hot dispatch loop pays two thread-local loads per call after the first. (A `static` inside a generic function CANNOT provide this identity: Rust instantiates ONE shared static across all monomorphizations, which would merge every caller into a single learning pool. The `track_caller` attribute chains through the delegating wrappers, so `for_each_chunk_indexed` and `reduce_chunks` resolve to the outermost user call site.)

What each site learns, all atomically and lock-free:

- **Classifier**: a learned `WorkloadClass` tag with hysteresis (two consecutive agreeing quanta to switch) plus a fast-adapt path (bucket distance >= 2 with >= 64 samples switches immediately). Read via `learned_class()`; `JobPlan::apply_site_class` re-derives routing knobs from it when the caller pinned nothing.
- **Leaf statistics**: cumulative and delta-window count / sum / sum-of-squares of leaf nanoseconds, exposing `cv2_per_mille()` (squared coefficient of variation) and `leaf_count()`. Leaf batches ALSO flow into the process-global stats, which stay the cold-start prior for site-less plans.
- **Policy arms**: EWMA per arm (`Slaw` vs `Heartbeat`) with a trial cadence, so irregular workloads converge on the scheduling policy that measures faster at THAT site.
- **Hybrid placement**: per-log2-size-bucket CPU vs backend EWMAs feeding [`hybrid_auto`](#hybrid_auto), plus learned per-item split throughputs feeding [`hybrid_auto_split`](#hybrid_auto_split).

`SiteRef::new(&STATIC_SITE)` wraps a caller-owned static for explicit attachment via [`JobPlan::with_site`](JobPlan-Reference.md#builder-methods); an outer attachment always wins over the entry's own location-resolved site. E2E walkthrough: [`examples/site_classifier_demo.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/site_classifier_demo.rs).

## `adaptive_profile`

Process-global `WorkloadClass` / `DispatchProfile` migration via a single `AtomicU8` tag, plus the calibrated classification thresholds. Defined in [`src/sched/adaptive_profile.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/adaptive_profile.rs).

```rust
pub enum WorkloadClass { Light, Compute, Heavy, Memory }

pub fn active_dispatch_profile()   -> DispatchProfile;   // Acquire-load
pub fn migrate_dispatch_profile(p: DispatchProfile);     // Release-store
pub fn active_workload_class()     -> WorkloadClass;     // wraps above
pub fn migrate_workload_class(c: WorkloadClass);         // Release-store
```

The global `ACTIVE_PROFILE_TAG: AtomicU8` is read by the `AdaptiveDispatcher`'s plan construction (~1 ns), zero per-op cost on the dispatch hot path. [`JobPlan::new`](JobPlan-Reference.md#new) routes from its own static classifier instead, so the global reflects dispatcher-surface policy rather than every plan. Applications observe their workload and call `migrate_workload_class(...)` when the active class no longer matches; the observer's `tick_auto_classify()` does the same automatically from measured leaf times.

See [`WorkloadClass`](Foundation-Types-Reference.md#workloadclass) for the user-facing enum + the mapping to `DispatchProfile`; migration behavior is covered by the unit tests in [`src/sched/adaptive_profile.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/adaptive_profile.rs).

### `ClassThresholds` and in-binary calibration

```rust
pub struct ClassThresholds {
    pub fine_grain_ns: AtomicU64,          // default 50
    pub port_heavy_ns: AtomicU64,          // default 500
    pub memory_latency_ns: AtomicU64,      // default 2000
    pub cv2_low_per_mille: AtomicU64,      // default 50
    pub cv2_high_per_mille: AtomicU64,     // default 500
    pub trivial_reduce_cycles: AtomicU64,  // default 30000
}

pub fn class_thresholds() -> &'static ClassThresholds;
pub fn calibrate_class_thresholds() -> ThresholdCalibration;
pub fn calibrate_class_thresholds_into(target: &ClassThresholds) -> ThresholdCalibration;
pub fn spawn_class_threshold_calibration();
pub fn classify_observed(mean_ns: u64, cv2_per_mille: u64) -> WorkloadClass;
```

The classification boundaries used by `classify_observed` and the static hint branch are host-calibrated atomics, not hard-coded constants. `calibrate_class_thresholds()` measures the RUNNING binary on the RUNNING host: an empty-join round trip sets `fine_grain_ns` (median / 256, clamped 25..=200), a 256-element merge sets `trivial_reduce_cycles` (measured x8, clamped 5000..=200000), and a paired sqrt probe with and without SMT siblings nudges `memory_latency_ns` (+-250, clamped 1000..=4000). `spawn_class_threshold_calibration()` runs the same pass once on a background thread; the returned `ThresholdCalibration` report carries the raw measurements alongside the installed values.

## `adaptive_worker`

Per-worker K_inner=3 deque-backing tag swap. Defined in [`src/sched/adaptive_worker.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/adaptive_worker.rs).

```rust
pub struct AdaptiveWorker { /* holds an AtomicU32 tag + two backings */ }
pub struct AdaptiveStealer { /* mirrors AdaptiveWorker on the thief side */ }
pub struct AdaptiveStash { /* per-thief K_inner=3 pop-stash */ }

pub fn new_adaptive(slot_capacity: usize, initial: KGating) -> (AdaptiveWorker, AdaptiveStealer);
pub fn steal_via_stash(stealer: &AdaptiveStealer, stash: &mut AdaptiveStash) -> AdaptiveSteal2<JobRef>;
```

The `AdaptiveWorker` holds both a KHL (per-slot Vyukov) and an Fcl (counter-only Chase-Lev) backing internally. Per push / pop it reads its `AtomicU32` tag with one `Relaxed` load and branches to the active backing. Migration is one `AtomicU32::Release-store` per worker; `LocalArena::migrate_all_workers_k_gating(KGating::*)` walks every worker's tag in one pass.

See [`KGating`](Foundation-Types-Reference.md#kgating) for the user-facing enum.

## `adaptive_backend`

Process-global active-backend tag swap. Defined in [`src/sched/adaptive_backend.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/adaptive_backend.rs).

```rust
pub fn active_backend_id() -> Backend;
pub fn migrate_backend(b: Backend);
pub fn resolve_active_backend() -> (BackendRef, bool /* fell_back */);
```

`ACTIVE_BACKEND_TAG: AtomicU32` holds the encoded `Backend` enum tag. `migrate_backend(Backend::Cuda { device_id: 0 })` re-points dispatch through a different registered backend in one `AtomicU32::Release-store`. `resolve_active_backend()` looks up the registered backend for the active tag and falls back to `cpu_backend()` when the requested backend is not registered, returning `(BackendRef, fell_back: bool)` so the caller knows which path landed.

Wired by [`AdaptiveDispatcher`](#dispatch) for `migrate_backend` / `active_backend_id` / `resolve_active_backend` exposure on the user-facing surface.

## `dispatch`

Unified user-facing dispatcher: [`AdaptiveDispatcher`](#dispatch). Defined in [`src/sched/dispatch.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/dispatch.rs). Picks the Flynn-axis entry point (SISD / MIMD / SIMC / MIMC / MISD) from a single [`WorkloadShape`](Foundation-Types-Reference.md#workloadshape) hint.

```rust
pub struct AdaptiveDispatcher { /* opaque */ }

impl AdaptiveDispatcher {
    pub fn new() -> Self;

    // Builders (per-call hints).
    pub fn with_shape(self, shape: WorkloadShape) -> Self;
    pub fn with_workload_class(self, class: WorkloadClass) -> Self;
    pub fn with_smt(self) -> Self;
    pub fn with_variant(self, variant: Variant) -> Self;

    // Execute entry points (one per Flynn axis).
    pub fn execute_streaming<R, F: FnOnce() -> R>(self, op: F) -> R;
    pub fn execute_cooperative<R: Send + 'static>(self, closures: Vec<Box<dyn FnOnce() -> R + Send>>) -> Vec<R>;
    pub fn execute_cooperative_mailbox<R: Send + 'static>(self, closures: Vec<Box<dyn FnOnce() -> R + Send>>) -> Vec<R>;
    pub fn execute_for_each<T: Send, F: Fn(&mut [T]) + Sync + Send>(self, items: &mut [T], op: F);
    pub fn execute_indexed<F: Fn(u32) + Send + Sync>(self, count: u32, work: F) -> bool;

    // Runtime migration (atomic tag swaps).
    pub fn migrate_k_gating(&self, gating: KGating);
    pub fn migrate_workload_class(&self, class: WorkloadClass);
    pub fn migrate_dispatch_profile(&self, profile: DispatchProfile);
    pub fn migrate_backend(&self, backend: Backend);

    // State inspection.
    pub fn active_dispatch_profile(&self) -> DispatchProfile;
    pub fn active_backend_id(&self) -> Backend;
    pub fn resolve_active_backend(&self) -> (BackendRef, bool);
}
```

End-to-end demo covering all four Flynn-axis dispatches + all four migration surfaces (K_gating, WorkloadClass, DispatchProfile, Backend) sits at [`examples/adaptive_dispatcher_demo.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/adaptive_dispatcher_demo.rs):

```sh
cargo run --release --example adaptive_dispatcher_demo
cargo run --release --features cuda-reference,wasm-reference --example adaptive_dispatcher_demo
```

## `workload_shape`

Declarative shape API. Defined in [`src/sched/workload_shape.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/workload_shape.rs). The enum lives at the call site; the dispatcher maps shape -> `(k_gating, mailbox, oversubscription)` knob triples once at plan construction, and the per-call dispatch path stays direct atomic ops.

```rust
pub enum WorkloadShape {
    Streaming,
    ProducerFast { burst: u32 },
    WorkSteal { n_consumers: u32, batch_size: u32 },
    Cooperative { n_cores: u32 },
    VariantRace,
}

impl WorkloadShape {
    pub fn hints(&self) -> WorkloadShapeHints;
}

pub struct WorkloadShapeHints {
    pub k_gating: KGating,
    pub use_mailbox_routing: bool,
    pub oversubscription_log2: Option<u8>,
}
```

Consumed by [`JobPlan::with_workload_shape(shape)`](JobPlan-Reference.md#builder-methods) and by [`AdaptiveDispatcher::with_shape(shape)`](#dispatch). See [`WorkloadShape`](Foundation-Types-Reference.md#workloadshape) for the per-variant axis mapping.

## `k_gating`

Per-worker K_inner=3 deque-backing selector + per-host calibration. Defined in [`src/sched/k_gating.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/k_gating.rs).

```rust
pub enum KGating { Auto, CounterOnly, PerSlot }

pub fn calibrate_k_gating() -> KGating;
pub static CALIBRATED_GATING: OnceLock<KGating>;
```

`calibrate_k_gating()` runs a short probe at startup (push/pop bursts against both backings) and picks the winner for this host. The result is cached in `CALIBRATED_GATING` so subsequent `KGating::Auto` reads land on the winner without re-calibrating.

See [`KGating`](Foundation-Types-Reference.md#kgating) for the user-facing enum and per-variant trade-offs.

## `arena_local`

Single-NUMA-node work-stealing thread pool. Defined in [`src/sched/arena_local.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/arena_local.rs).

```rust
pub struct LocalArena { /* opaque */ }

impl LocalArena {
    pub fn new(n_workers: usize) -> Arc<Self>;
    pub fn with_cpu_set(n_workers: usize, cpu_set: Option<Vec<CoreId>>) -> Arc<Self>;
    pub fn with_smt_extension(primary_count: usize, smt_extension: usize, cpu_set: Option<Vec<CoreId>>) -> Arc<Self>;
    pub fn submit(&self, job: JobRef);
    pub fn try_run_one(&self) -> bool;
    pub fn acquire_smt(&self) -> SmtGuard;
    pub fn migrate_all_workers_k_gating(&self, gating: KGating);
    pub fn global_burst_ratio(&self) -> f32;
    pub fn injector_len(&self) -> usize;
    pub fn wait_injector_drained(&self);
}
```

Each worker holds an [`AdaptiveWorker`](#adaptive_worker) per-tier stack (4 tiers from `SmtLocal` to `Public`). The arena's `injector` is the in-house [`Injector<JobRef>`](#injector); per-worker mailboxes are [`FlynnelRing<JobRef>`](#flynnel_ring) (capacity 16, used by `push_to_mailbox` for cross-worker hand-offs).

## `arena_numa`

Multi-NUMA composition. Defined in [`src/sched/arena_numa.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/arena_numa.rs).

```rust
pub struct NumaArena { /* opaque */ }
```

Composes one `LocalArena` per NUMA node. On single-NUMA hosts (most desktops) it collapses to a single underlying `LocalArena`; cross-node code paths are dead branches with zero overhead. On multi-NUMA hosts it routes work to the caller's current-thread node by default and rebalances via cross-node steal when one node is idle.

## `numa_latency`

Cross-core cache-line round-trip latency table. Defined in [`src/sched/numa_latency.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/numa_latency.rs).

```rust
pub struct TopologyLatencyTable { /* opaque */ }
pub fn topology_latency_table() -> &'static TopologyLatencyTable;
```

Ping-pong measured at startup between pinned cores; lets the dispatcher pick peer-steal victims by coherence distance instead of random selection.

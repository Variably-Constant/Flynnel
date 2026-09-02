---
title: Extended Flynn Taxonomy
weight: 2
---

The architectural framing Flynnel speaks. This page explains why the crate is named after Michael J. Flynn and how the eight-axis mapping below organises every primitive in the crate into a single coherent vocabulary.

## Michael J. Flynn and the 1966 taxonomy

[Michael J. Flynn](https://en.wikipedia.org/wiki/Michael_J._Flynn) is a Stanford computer architect (born 1934) who, in his 1966 paper *Very high-speed computing systems*, proposed a four-class taxonomy of computer architectures based on independent axes:

- **I-axis**: how many *instruction streams* the machine fetches simultaneously (single vs multiple).
- **D-axis**: how many *data streams* those instructions act on simultaneously (single vs multiple).

The Cartesian product gives four classes:

| Acronym | Expansion                            | Canonical example                              |
|---------|--------------------------------------|------------------------------------------------|
| SISD    | Single Instruction, Single Data      | a classical scalar CPU executing one operation |
| SIMD    | Single Instruction, Multiple Data    | a vector ALU (SSE, AVX, NEON)                  |
| MISD    | Multiple Instruction, Single Data    | rarely realized; speculative variant racing fits this shape |
| MIMD    | Multiple Instruction, Multiple Data  | a multicore CPU running independent threads    |

Six decades later, computing has acquired axes Flynn's original taxonomy did not anticipate: GPUs with thousands of simultaneous threads driven by one instruction stream (SIMT), cores cooperating as a single logical vector across a coherence boundary, hybrid CPU-plus-accelerator dispatch where different instruction streams target different hardware classes simultaneously.

Flynnel extends the original taxonomy by four axes to cover this modern landscape. Every primitive in the crate is positioned on one of the eight axes below; the crate name is a pun on Flynn's surname and the lineage of [Cilk](https://en.wikipedia.org/wiki/Cilk) and [rayon](https://github.com/rayon-rs/rayon).

## The eight axes Flynnel covers

| Acronym | Expansion                                  | Flynnel entry point                                |
|---------|--------------------------------------------|---------------------------------------------------|
| SISD    | Single Instruction, Single Data            | `K_core = 0` (inline execution in caller thread)  |
| SIMD    | Single Instruction, Multiple Data          | `K_hardware >= 1` (vector lanes within a kernel)  |
| MISD    | Multiple Instruction, Single Data          | [`flynnel::race_variants`](Sched-Module-Reference.md#race_variants) |
| MIMD    | Multiple Instruction, Multiple Data        | [`flynnel::join`](Sched-Module-Reference.md#join) + work-stealing arena |
| SIMT    | Single Instruction, Multiple Threads       | [`DispatchBackend::dispatch_parallel_for`](Backend-System.md#dispatch_parallel_for) |
| MIMT    | Multiple Instruction, Multiple Threads     | [`flynnel::join_hybrid`](Sched-Module-Reference.md#join_hybrid) |
| SIMC    | Single Instruction, Multiple Cores         | [`flynnel::cooperative_join_n`](Sched-Module-Reference.md#cooperative_join_n) |
| MIMC    | Multiple Instruction, Multiple Cores       | `K_class` within `K_unified` (heterogeneous roles inside one cooperative call) |

The four rows from Flynn's original 1966 paper (SISD / SIMD / MISD / MIMD) are at the top; the four extensions (SIMC / MIMC / SIMT / MIMT) span the bottom.

### SISD: inline execution

When `pick_tier(plan, topo)` returns `SchedTier::Inline`, the scheduler does nothing: the calling thread runs the closure(s) serially. This is the dispatch-cost floor and is correct any time per-call work is below the scheduler's overhead. `JobPlan` with `k_outer <= 4` and `batch_size < 256` lands here.

### SIMD: vector lanes inside one kernel

Flynnel does not produce SIMD instructions itself; the closure body you pass to `for_each_chunk` is what hits the SIMD lanes. The crate carries the `HwClass` enum (`Sse2`, `Avx2`, `Avx512f`, `Neon`, `Avx512Bf16`, `Avx512Vnni`) so consumers can communicate which lane width their kernel expects. The arena treats SIMD as opaque per-leaf work and never assumes anything about lane layout.

### MISD: variant racing

[`race_variants(plan, fast, faithful, correct)`](Sched-Module-Reference.md#race_variants) is the MISD primitive. Three closures compute the same logical result at three different precision tiers; the dispatcher submits all three concurrently and returns the first one that succeeds (i.e., its accuracy contract is met). This is the Ziv speculative-widening pattern: the cheap variant usually wins, the expensive variant is the safety net.

### MIMD: work-stealing fork-join

The bread-and-butter axis. [`flynnel::join(plan, a, b)`](Sched-Module-Reference.md#join) and the broader [`for_each_chunk`](Sched-Module-Reference.md#for_each_chunk) family run independent work on independent cores. Implementation uses a Chase-Lev work-stealing deque per worker, a four-state `CoreLatch`, and a JEC-protected two-phase sleep protocol (the rayon-core lineage).

### SIMT: parallel-for on a GPU / TPU

[`DispatchBackend::dispatch_parallel_for(count, work)`](Backend-System.md#dispatch_parallel_for) is the SIMT entry point. CPU backends fan out to the work-stealing arena; GPU and TPU backends launch `count` work-items as a single kernel invocation that runs in lockstep across the device's SIMT lanes (32-thread warp on NVIDIA, 64-thread wave on AMD, MXU lane on TPU).

The closure body remains CPU-runnable (an arbitrary Rust closure cannot codegen to PTX). For real GPU compute, consumers use the [`dispatch_kernel`](Backend-System.md#dispatch_kernel) handle path with a pre-built PTX / Python kernel body.

Two flavours of SIMT kernel ship in the bench surface, demonstrating the distinction:

- **Per-thread SIMT** (`kernels/newton_sqrt.ptx`): one thread per element, no explicit warp-level primitives. The GPU groups threads into warps of 32 in hardware, but the kernel body does not exchange register values across lanes. This is the canonical embarrassingly-parallel shape and is correct for any per-element-independent workload.
- **Warp-cooperative SIMT** (CUDA C source inlined in `benches/flynn_axes.rs` as `KERNEL_CUDA_C_WARP`, compiled via `cudarc::nvrtc::compile_ptx`): threads use `__shfl_xor_sync(0xffffffff, residual, mask)` (butterfly warp shuffle) to do a 32-lane max-reduce of the per-iteration residual at the end of each Newton iteration, then take a warp-wide early-exit branch when the warp-max falls below epsilon. This is genuine cross-lane register exchange without going through shared memory and demonstrates a warp-level ballot pattern. It pays back on workloads where convergence is bursty enough that early exit fires routinely.

Both ship as separate benchmark functions in the `simt_*` criterion group so the gap between them is measured rather than asserted. See [Benchmarks](Benchmarks.md) for the numbers.

### MIMT: hybrid CPU + accelerator dispatch

Flynnel ships two MIMT primitives covering distinct shapes:

- [`join_hybrid(plan, cpu_work, gpu_work)`](Sched-Module-Reference.md#join_hybrid) runs ONE pair of distinct closures concurrently - CPU half on the calling thread, GPU half on whatever `JobPlan::pick_backend()` selects. Returns `(cpu_result, gpu_result)`. Use this when the two halves are independent and the caller only needs one round of overlap.
- [`hybrid_pipeline(plan, inputs, pre_cpu, gpu, post_cpu)`](Sched-Module-Reference.md#hybrid_pipeline) runs a streaming three-stage pipeline where each input flows through `pre_cpu → gpu → post_cpu` and stage[N+1] of an earlier pipeline position overlaps stage[N] of a later one. After pipeline fill, steady-state throughput is `1 / max(t_pre, t_gpu, t_post)` per input - the smaller stages hide entirely behind the largest.

The pipeline primitive is the algorithmically meaningful MIMT shape: the GPU half consumes outputs the CPU produced AND feeds the next CPU stage. Coupled algorithms that fit this shape include:

- **Metropolis-Hastings MCMC**: CPU adaptive proposal → GPU log-likelihood evaluation → CPU accept/reject + step adaptation.
- **Batched conjugate gradient** (CG with many RHS): CPU per-RHS init/check → GPU sparse matrix-vector product → CPU dot product + scalar update.
- **MCTS with batched NN evaluation** (the AlphaZero/MuZero shape): CPU tree expansion by UCB + visit count → GPU policy/value network on the leaf batch → CPU backprop value up the tree.

`join_hybrid` is the single-pair primitive; `hybrid_pipeline` is the streamed primitive. Use the single-pair one for one-shot overlap with no data feedback; use the pipeline one for any loop where successive iterations depend on each other through the CPU + GPU stages. See [Benchmarks](Benchmarks.md) for measured speedups on all three coupled-algorithm shapes.

This is the highest-rank Flynn axis Flynnel exposes. Below it the taxonomy fans out into per-device specialization; above it lies cluster-scale dispatch (distributed orchestration), which is a different concern and a different crate.

### SIMC: cooperative cross-core vector

[`cooperative_join_n(plan, closures)`](Sched-Module-Reference.md#cooperative_join_n) takes a list of N closures and runs them as one logical mega-vector across N cores. Unlike MIMD `join` (which treats the two halves as independent tasks), the cooperative path commits to running all N together with a single sync boundary so the closures can compute partial results that combine deterministically. The cooperative shape matches reductions across a cache-coherent complex (intra-CCX on AMD Zen, intra-cluster on Apple silicon).

### MIMC: heterogeneous roles inside one cooperative call

When the closures passed to `cooperative_join_n` are not identical, the call is MIMC (Multiple Instruction, Multiple Cores). The classic case: in a 4-way cooperative reduce, three closures compute partials and the fourth computes a calibration probe; all four run concurrently and the result combines deterministically. Flynnel's `K_class` axis names this pattern (`K_class = 2` for two distinct roles, etc.); the cooperative_join_n primitive carries the partition shape directly.

Two workload shapes ship as MIMC benchmark contenders in [`benches/flynn_axes.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/benches/flynn_axes.rs):

- **4-way heterogeneous reduce**: 3 closures compute partial chained-sqrt sums over disjoint chunks (role A), 1 closure computes a max-abs calibration probe over the whole input (role B). Total work at this cell ~9 ms.
- **Pivoted-LU step**: 1 closure does pivot selection + row scale (role A: scalar column scan), 7 closures apply the pivot to disjoint trailing-row ranges (role B: vector SAXPY). The canonical "one role factors, N roles apply" pattern in numerical linear algebra. Fine-grain coordination scale ~130 us per step; the cooperative primitive's single-sync-boundary matches the algorithm's dependency graph directly.

See [Benchmarks](Benchmarks.md) for the measured harness.

## Why this matters in practice

The eight-axis framing is not bureaucracy. It tells you which primitive to call:

- "I have a low-precision fast path and a high-precision safe path, take whichever finishes first with the contract met": [`race_variants`](Sched-Module-Reference.md#race_variants) (MISD).
- "I have two halves that don't share data": [`join`](Sched-Module-Reference.md#join) (MIMD).
- "I have a large data-parallel loop body to run on a GPU kernel": [`DispatchBackend::dispatch_kernel`](Backend-System.md#dispatch_kernel) (SIMT).
- "I have a CPU pre-processing step and a GPU compute step, run them concurrently (one pair)": [`join_hybrid`](Sched-Module-Reference.md#join_hybrid) (MIMT, single pair).
- "I have an iterative algorithm where CPU and GPU stages alternate and each iteration depends on the prior, and I want consecutive iterations to overlap": [`hybrid_pipeline`](Sched-Module-Reference.md#hybrid_pipeline) (MIMT, streamed).
- "I have N halves that need to commit together with one sync boundary": [`cooperative_join_n`](../reference/Sched-Module-Reference.md#cooperative_join_n) (SIMC).

Every axis above is also reachable through the unified [`AdaptiveDispatcher`](#using-each-axis-through-adaptivedispatcher) surface when your call site prefers the shape-hint routing over selecting a primitive by name.

It also gives you a vocabulary for talking about your workload's dispatch shape with collaborators, and a stable mapping to anchor papers and benchmarks against.

## K-axes: the data-side companion

The Flynn axes describe the *control* shape of a computation. Flynnel also carries a set of `K_*` axes that describe the *data* shape: `K_outer` (operand size), `K_inner` (sub-unit bits), `K_hardware` (lanes per instruction), `K_class` (distinct execution regimes per call), `K_core` (cores dispatched), `K_unified` (cores cooperating as one mega-vector).

The two systems are orthogonal. A job is positioned on one Flynn axis (the *kind* of parallelism) and several K-axes (the *amount* of data + compute). `JobPlan` carries both: `k_outer`, `batch_size`, `hw_class` (the data side); `use_smt`, `backend_hint` (the dispatch side).

See [JobPlan Reference](JobPlan-Reference.md) for the full field list.

## Using each axis through `AdaptiveDispatcher`

[`AdaptiveDispatcher`](../reference/Sched-Module-Reference.md#dispatch) is the unified user-facing surface. Every Flynn axis Flynnel exposes is reachable through it via a matching `execute_*` method + a `WorkloadShape` hint. The hint routes the dispatch to the low-level primitive AND tunes the K-axis knobs (K_gating, mailbox routing, oversubscription) via [`WorkloadShape::hints()`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/workload_shape.rs).

| Axis | Dispatcher call | `WorkloadShape` hint | Underlying primitive |
|---|---|---|---|
| SISD | `.execute_streaming(op)` | `WorkloadShape::Streaming` | Direct closure call on caller thread |
| MIMD | `.execute_for_each(items, op)` | `WorkloadShape::WorkSteal { n_consumers, batch_size }` | [`for_each_chunk`](../reference/Sched-Module-Reference.md#for_each_chunk) |
| SIMC | `.execute_cooperative(closures)` | `WorkloadShape::Cooperative { n_cores }` | [`cooperative_join_n_flat`](../reference/Sched-Module-Reference.md#cooperative_join_n) |
| SIMC (mailbox) | `.execute_cooperative_mailbox(closures)` | same as SIMC | `cooperative_join_n_flat_mailbox` (URD owner-directed) |
| MIMC | `.execute_cooperative(closures)` with heterogeneous closures | `WorkloadShape::Cooperative { n_cores }` | Same as SIMC; heterogeneity is intrinsic to the closures |
| SIMT | `.execute_indexed(count, work)` | `WorkloadShape::WorkSteal { .. }` | [`DispatchBackend::dispatch_parallel_for`](../reference/Backend-System.md) via active backend (CPU / CUDA / TPU / Metal / ROCm) |
| MIMT single-pair | Direct [`join_hybrid(plan, cpu, gpu)`](../reference/Sched-Module-Reference.md#join_hybrid) call | (dispatcher does not carry a MIMT execute_ method) | `join_hybrid` |
| MIMT pipelined | Direct [`hybrid_pipeline(plan, ..)`](../reference/Sched-Module-Reference.md#hybrid_pipeline) call | (dispatcher does not carry a MIMT execute_ method) | `hybrid_pipeline` |
| MISD | Direct [`race_variants(plan, ..)`](../reference/Sched-Module-Reference.md#race_variants) call OR `WorkloadShape::VariantRace { n_variants }` hint | `WorkloadShape::VariantRace` | `race_variants` |

### Manual walk-through: dispatcher for each axis

#### SISD via dispatcher

```rust
use flynnel::sched::dispatch::AdaptiveDispatcher;
use flynnel::sched::workload_shape::WorkloadShape;

let sum: u64 = AdaptiveDispatcher::new()
    .with_shape(WorkloadShape::Streaming)
    .execute_streaming(|| (0..1_000_000u64).sum());
```

`execute_streaming` runs the closure directly on the caller's thread (no scheduler involvement). The `Streaming` shape hint carries the K-axis defaults suited to sequential work.

#### MIMD via dispatcher (data-parallel bulk work)

```rust
use flynnel::sched::dispatch::AdaptiveDispatcher;
use flynnel::sched::workload_shape::WorkloadShape;

let mut data: Vec<u64> = (0..1_000_000).collect();
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
```

`execute_for_each` builds a `JobPlan` from the shape + k_outer + variant, then routes to `for_each_chunk`. The `WorkSteal` shape tunes the K_gating / oversubscription / mailbox knobs for data-parallel work.

#### SIMC via dispatcher (N-way cooperative)

```rust
use flynnel::sched::dispatch::AdaptiveDispatcher;
use flynnel::sched::workload_shape::WorkloadShape;

let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..8u64)
    .map(|i| {
        let f: Box<dyn FnOnce() -> u64 + Send> = Box::new(move || {
            let start = i * 1000;
            (start..start + 1000).sum::<u64>()
        });
        f
    })
    .collect();

let results: Vec<u64> = AdaptiveDispatcher::new()
    .with_shape(WorkloadShape::Cooperative { n_cores: 8 })
    .execute_cooperative(closures);
```

The `Cooperative` shape hint switches to the flat fan-out variant of `cooperative_join_n`. For the mailbox-routed variant use `.execute_cooperative_mailbox(closures)` instead; the internal gating demotes to deque mode when N is below the worker count.

#### MIMC via dispatcher (heterogeneous closures)

MIMC is not a separate `execute_` method: pass CLOSURES OF DIFFERENT SHAPES to `execute_cooperative`. The dispatcher does not know or care whether the closures are identical; heterogeneity is intrinsic to the call.

```rust
use flynnel::sched::dispatch::AdaptiveDispatcher;
use flynnel::sched::workload_shape::WorkloadShape;

// Pivoted-LU step: 1 pivot-select closure + 3 apply closures.
let mut closures: Vec<Box<dyn FnOnce() -> f64 + Send>> = Vec::new();
closures.push(Box::new(|| pivot_select_and_scale()));
closures.push(Box::new(|| apply_pivot_row_range_a()));
closures.push(Box::new(|| apply_pivot_row_range_b()));
closures.push(Box::new(|| apply_pivot_row_range_c()));

let results: Vec<f64> = AdaptiveDispatcher::new()
    .with_shape(WorkloadShape::Cooperative { n_cores: 4 })
    .execute_cooperative(closures);
```

The result vector preserves caller-supplied order: `results[0]` is the pivot-select closure's return regardless of which worker executed it.

#### SIMT via dispatcher (backend-adaptive parallel-for)

```rust
use flynnel::sched::dispatch::AdaptiveDispatcher;
use flynnel::sched::workload_shape::WorkloadShape;

let dispatcher = AdaptiveDispatcher::new()
    .with_shape(WorkloadShape::WorkSteal {
        n_consumers: 1024,
        batch_size: 1024,
    });

// Optionally point at a specific backend (defaults to CPU).
dispatcher.migrate_backend(flynnel::Backend::Cuda { device_id: 0 });

let fell_back = dispatcher.execute_indexed(1024, |i: u32| {
    // Work item i. On CPU, this runs on the work-stealing pool.
    // On CUDA, this dispatches through the registered CUDA backend
    // (falls back to CPU if the backend is not registered).
    do_work(i);
});
println!("Backend fell back to CPU: {}", fell_back);
```

`execute_indexed` consults the process-global active backend via `resolve_active_backend()`. If the requested backend is not registered, the return value indicates the fallback. This is the same routing decision `hybrid_pipeline` makes for its `gpu` stage.

#### MISD via dispatcher (variant racing)

MISD is best expressed as a direct [`race_variants`](../reference/Sched-Module-Reference.md#race_variants) call because the three-arm shape (`fast` / `faithful` / `correct`) has a specific type contract. The `WorkloadShape::VariantRace { n_variants }` hint on `AdaptiveDispatcher` sets the K-axis knobs for a racing dispatch, and callers hand the plan built via `dispatcher.build_plan(1)` to `race_variants` directly:

```rust
use flynnel::{JobPlan, race_variants};

let plan = JobPlan::new(6, 1);
let (result, winner) = race_variants(
    &plan,
    |_cancel| Some(fast_path()),          // may return None to fail out
    |_cancel| Some(faithful_path()),
    |_cancel| correct_path(),             // always returns
);
```

#### MIMT via direct primitive (hybrid CPU + accelerator)

MIMT is likewise best expressed via the direct primitive because the CPU / accelerator role separation is part of the type signature:

```rust
use flynnel::{Backend, JobPlan, join_hybrid, hybrid_pipeline};

// Single-pair MIMT: one CPU + one GPU stage overlap.
let plan = JobPlan::new(8, 1024)
    .with_backend(Backend::Cuda { device_id: 0 });
let (cpu_result, gpu_result) = join_hybrid(
    &plan,
    || cpu_side_work(),
    || gpu_side_work(),
);

// Pipelined MIMT: streamed three-stage overlap.
let outputs: Vec<f32> = hybrid_pipeline(
    &plan,
    0..16u64,
    |seed| cpu_pre_stage(seed),
    |prepared| gpu_stage(prepared),
    |gpu_out| cpu_post_stage(gpu_out),
);
```

### Migration methods (runtime axis-swap)

`AdaptiveDispatcher` exposes the process-global migration surface so a single call handle can flip every subsequent dispatch's routing:

| Method | Effect |
|---|---|
| `.migrate_workload_class(WorkloadClass)` | Set the global active class. Next `JobPlan::new` calls read the new class. |
| `.migrate_dispatch_profile(DispatchProfile)` | Set the active profile directly (bypasses the class-to-profile map). |
| `.migrate_k_gating(KGating)` | Flip every worker's per-tier deque backing (KHL PerSlot vs Fcl CounterOnly). |
| `.migrate_backend(Backend)` | Retarget the SIMT / MIMT accelerator dispatches. |
| `.active_dispatch_profile()` / `.active_backend_id()` / `.resolve_active_backend()` | Observation methods for the currently-active state. |

Each migration is a single atomic Release-store; the per-op deque hot path is untouched. A full walk-through of all three (K_gating + WorkloadClass + Backend) lives at [`examples/adaptive_dispatcher_demo.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/adaptive_dispatcher_demo.rs). A per-axis walk-through of every `execute_*` method above (SISD / MIMD / SIMC / SIMC-mailbox / MIMC / SIMT / migration surface) lives at [`examples/dispatcher_per_axis.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/dispatcher_per_axis.rs); run with `cargo run --example dispatcher_per_axis --release`.

## Reading the source

Each axis maps to one or more modules:

| Axis | Module |
|------|--------|
| SISD | [`sched::arena::inline_join_context`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/arena.rs) (private; reached via `pick_tier == Inline`) |
| SIMD | [`HwClass`](Foundation-Types-Reference.md#hwclass) classifier + consumer kernel bodies |
| MISD | [`sched::race`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/race.rs) |
| MIMD | [`sched::arena`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/arena.rs), [`sched::arena_local`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/arena_local.rs), [`sched::par_iter`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/par_iter.rs) |
| SIMT | [`backend::DispatchBackend`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/mod.rs) |
| MIMT | [`sched::hybrid::join_hybrid`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/hybrid.rs) |
| SIMC | [`sched::cooperative`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/cooperative.rs) |
| MIMC | `K_class` partitioning inside `cooperative_join_n` |

---
title: Getting Started
weight: 1
---

A 5-minute walkthrough that gets you from `cargo add flynnel` to running parallel work.

## Install

```toml
[dependencies]
flynnel = "0.2"
```

Optional features:

```toml
flynnel = { version = "0.2", features = ["verify-chain"] }            # BLAKE3 verify-chain
flynnel = { version = "0.2", features = ["cuda-reference"] }          # NVIDIA reference backend
flynnel = { version = "0.2", features = ["tpu-jax-reference"] }       # Google TPU reference backend
```

Minimum Rust: **1.96** (edition 2024, let-chains).

## First call: `join`

`flynnel::join` is the two-way fork-join primitive. It runs two closures concurrently on the work-stealing arena and returns their results.

```rust
use flynnel::{JobPlan, join};

fn main() {
    let plan = JobPlan::new(8, 1024);
    let (a, b) = join(
        &plan,
        || (0..512).sum::<u64>(),
        || (512..1024).sum::<u64>(),
    );
    assert_eq!(a + b, (0..1024).sum::<u64>());
}
```

What's happening:

- `JobPlan::new(k_outer, batch_size)` constructs a default plan. `k_outer = 8` means *"this job's per-operand data has `2^8` logical sub-units"*; `batch_size = 1024` is the total item count. The pair tells the arena's tier picker whether to run inline, on the local pool, or hierarchically.
- `join(plan, a, b)` invokes both closures. The two halves run concurrently when the tier picker selects the `Local` tier (which it does at `k_outer >= 5` with `batch_size >= 32`).
- The returned tuple is `(RA, RB)` - **caller-supplied order**, invariant of which thread executed which half. This is the determinism contract bit-exact reductions rely on.

## Bulk parallel: `for_each_chunk`

For data-parallel work over a slice, use `flynnel::for_each_chunk`:

```rust
use flynnel::{JobPlan, for_each_chunk};

fn main() {
    let mut data: Vec<u64> = (0..1_000_000).collect();
    let plan = JobPlan::new(8, data.len() as u32);

    for_each_chunk(&plan, &mut data, |chunk: &mut [u64]| {
        for v in chunk {
            *v = v.wrapping_mul(3);
        }
    });

    assert_eq!(data[42], 42 * 3);
}
```

What's happening:

- The arena recursively bisects the slice into chunks, runs each chunk's `op` on a worker, and returns when every chunk is done.
- Bisection has a SLAW-style adaptive split budget: when steal pressure is observed on a sub-job, the budget refills so the subtree splits more aggressively.
- Leaf chunks are at least `MIN_LEAF_ITEMS` items (a tuned floor); below that the chunk runs serially in one worker without further splitting.
- `op` is `Fn(&mut [T]) + Sync` - the same closure body is called from multiple workers concurrently, each with a disjoint sub-slice.

## Picking the right plan

For most call sites `JobPlan::new(K, batch)` is enough. The default plan is adaptive: a static classifier on `(K, batch)` picks the `WorkloadClass` for call-1 routing, `KGating::Auto` lets the scheduler pick between the KHL (per-slot Vyukov) and Fcl (counter-only Chase-Lev) K_inner=3 backings based on host calibration, and per-call-site learned classes refine unpinned plans on later calls. The process-global class drives the `AdaptiveDispatcher` surface; [`flynnel::sched::adaptive_profile::migrate_workload_class`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/adaptive_profile.rs) flips it in one atomic store, free on the per-op path.

```rust
use flynnel::JobPlan;

// Adaptive default - this is the right call for ~every typical use.
let plan = JobPlan::new(8, 1024);
```

When you know the profile better than the calibrated default - for example a per-call site you've already measured - use `set_profile` to pin SMT activation, cost estimate, oversubscription, and locality routing for **this one call**:

```rust
use flynnel::{JobPlan, DispatchProfile};

// User override for one call site: pin SMT-siblings parked because
// the work saturates execution ports.
let plan = JobPlan::set_profile(8, 1024, DispatchProfile::PortBound);
```

`DispatchProfile` is the classification: `LatencyBound` (SMT siblings active to hide stall bubbles), `PortBound` (SMT siblings parked because they would contest the execution unit), `MemoryBound` (SMT siblings active to interleave cache misses).

For your own domain enums, implement [`OpClass`](../reference/Foundation-Types-Reference.md#opclass-trait) and use `for_op_generic` to inherit the same `is_latency_bound()` classification path. For a declarative shape-level override (streaming pipeline vs producer-fast burst vs work-steal fan-out vs cooperative reduction vs variant race), use `JobPlan::new(K, batch).with_workload_shape(WorkloadShape::*)`.

If you need bare defaults independent of the global active class (for example a unit test that asserts specific plan fields, or a call site that wants the historical pre-adaptive defaults), use [`JobPlan::bare`](../reference/JobPlan-Reference.md#bare) - it constructs a plan with `use_smt = false`, `estimated_per_item_ns = None`, `oversubscription_log2 = None`, and `K_gating = Auto`, ignoring the process-global `WorkloadClass`. Most production call sites want the adaptive `::new`, not `::bare`.

## Routing to a GPU / TPU backend

For ONE independent CPU + GPU pair:

```rust
use flynnel::{Backend, JobPlan, join_hybrid};

let plan = JobPlan::new(8, 1024)
    .with_backend(Backend::Cuda { device_id: 0 });

let (cpu_result, gpu_result) = join_hybrid(
    &plan,
    || cpu_side_work(),
    || gpu_side_work(),
);
```

For an iterated coupled algorithm where each iteration's GPU stage consumes outputs the CPU stage produced AND feeds the next CPU stage (MCMC, batched CG, MCTS with NN evaluation, etc.) use the streamed pipeline so consecutive iterations overlap:

```rust
use flynnel::{JobPlan, hybrid_pipeline};

let plan = JobPlan::new(8, 1024);
let outputs: Vec<f32> = hybrid_pipeline(
    &plan,
    0..16u64,                         // iteration seeds
    |seed| cpu_pre_stage(seed),       // CPU: prepare for GPU
    |prepared| gpu_stage(prepared),   // GPU: kernel + sync
    |gpu_out| cpu_post_stage(gpu_out),// CPU: consume GPU result
);
```

After pipeline fill, throughput is `1 / max(t_pre, t_gpu, t_post)` per iteration: the smaller stages hide behind the largest. See [Sched Module Reference - hybrid_pipeline](../reference/Sched-Module-Reference.md#hybrid_pipeline) and [Benchmarks - MIMT pipelined](../reference/Benchmarks.md#mimt-pipelined---coupled-algorithm-benchmarks) for the full API. Measured speedups on Zen+/16T + RTX 3070 at `CYCLES_PER_SAMPLE = 64`: Metropolis MCMC 1.96x, Batched CG 2.35x, MCTS 1.84x.

`pick_backend()` honors the hint when a backend with that id is registered, else falls back to the always-available `CpuBackend`. See [Backend System](Backend-System.md) for the full registration story.

## Running the demos

```bash
cargo run --example backend_dispatch_demo --release
cargo run --example tpu_jax_demo --release --features tpu-jax-reference
```

The first prints detection probe results and exercises every backend trait method against an in-process stub backend. The second drives the TPU JAX bridge and shows graceful degradation on hosts without Python+JAX.

## Where to go next

- [JobPlan Reference](JobPlan-Reference.md) - every field on the plan struct.
- [Sched Module Reference](Sched-Module-Reference.md) - `race`, `cooperative_join_n`, `join_hybrid`, `hybrid_pipeline`, `pipeline::run`, `par_map_in_place`, `par_zip_apply`, `par_map_serial_reduce`, and more.
- [Backend System](Backend-System.md) - the `DispatchBackend` trait + registry.
- [Architecture Overview](Architecture-Overview.md) - how the layers fit together.

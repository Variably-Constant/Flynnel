---
title: All Primitives Tutorial
weight: 2
---

A walk-through of every workhorse Flynnel primitive with runnable examples. Every code block in this tutorial matches a section of [`examples/tutorial_all_apis.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/tutorial_all_apis.rs); build and run it end-to-end with:

```bash
cargo run --example tutorial_all_apis --release
```

The example asserts every result. If the run finishes with "Tutorial done." you have verified the crate is installed correctly and every primitive works on your host.

## 1. `join(plan, a, b)`: two-way fork-join

The simplest fork-join primitive. Runs `a` and `b` concurrently on the work-stealing arena and returns their results.

```rust
use flynnel::{JobPlan, join};

let plan = JobPlan::new(8, 1024);
let (a, b) = join(
    &plan,
    || (0..512u64).sum::<u64>(),
    || (512..1024u64).sum::<u64>(),
);
assert_eq!(a + b, (0..1024u64).sum::<u64>());
```

The returned tuple is caller-supplied order regardless of which thread executed which half; that is the determinism contract bit-exact reductions rely on.

## 2. `join_context(plan, a, b)`: stolen/injected flag exposed

Same as `join` but each closure receives a `bool` telling it whether it was migrated across worker threads. The flag drives adaptive splitters that subdivide more aggressively when work is being stolen.

```rust
use flynnel::JobPlan;
use flynnel::sched::arena::join_context;

let plan = JobPlan::new(8, 1024);
let (a, b) = join_context(
    &plan,
    |injected: bool| {
        println!("a's injected flag = {injected}");
        (0..512u64).sum::<u64>()
    },
    |stolen: bool| {
        println!("b's stolen flag = {stolen}");
        (512..1024u64).sum::<u64>()
    },
);
```

`injected` is true exactly when the entire `join_context` was cold-injected from outside the worker pool. `stolen` is true exactly when the right-half was dequeued and executed by a peer.

## 3. `join_default(k_outer, batch_size, a, b)`: convenience wrapper

Convenience for the common case where you do not need to customize `hw_class` / `variant` / `numa_hint`. Builds `JobPlan::new(k_outer, batch_size)` internally.

```rust
use flynnel::join_default;

let (a, b) = join_default(
    8, 1024,
    || (0..512u64).sum::<u64>(),
    || (512..1024u64).sum::<u64>(),
);
```

## 4. `for_each_chunk(plan, &mut items, op)`: bulk data-parallel

The bisecting parallel-iterator for slice-shaped workloads. `op` is called from multiple workers concurrently, each with a disjoint sub-slice.

```rust
use flynnel::{JobPlan, for_each_chunk};

let mut data: Vec<u64> = (0..100_000).collect();
let plan = JobPlan::new(8, data.len() as u32);
for_each_chunk(&plan, &mut data, |chunk: &mut [u64]| {
    for x in chunk {
        *x = x.wrapping_mul(3);
    }
});
```

## 5. `for_each_chunk` with explicit per-item ns hint (production shape)

For call sites where you have measured per-item cost, supply it via `with_estimated_per_item_ns`. The tier picker sizes the leaf floor from the hint, skipping the probe path that a hint-less call would run.

```rust
use flynnel::{JobPlan, LeafShape, for_each_chunk};

let mut data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
let plan = JobPlan::new(6, data.len() as u32)
    .with_estimated_per_item_ns(50)          // ~50 ns per element
    .with_leaf_shape(LeafShape::PortCompute); // FMA-bound pipeline

for_each_chunk(&plan, &mut data, |chunk: &mut [f64]| {
    for x in chunk {
        let mut acc = *x + 1.0;
        for _ in 0..4 {
            acc = (acc + 1.0).sqrt();
        }
        *x = acc;
    }
});
```

`LeafShape` is the strongest static signal: when set it maps directly to a [`WorkloadClass`](../reference/Foundation-Types-Reference.md#workloadclass) and the (ns, K, N) heuristics are bypassed. Available values: `PortCompute`, `LatencyCompute`, `Streaming`, `Gather`, `Unknown`.

## 6. `cooperative_join_n(plan, closures)`: N-way SIMC dispatch

Fan out N closures across the worker pool with one sync boundary. Each closure runs on a different worker; the results are collected in the input order.

```rust
use flynnel::{JobPlan, cooperative_join_n};

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
```

For fewer than 3 closures, `cooperative_join_n` delegates to the tree variant which uses inline / 2-way fast paths.

## 7. `race_variants(plan, fast, faithful, correct)`: MISD speculation

Runs three algorithm variants in parallel; the first to return `Some(r)` wins. The `correct` variant always runs to completion and ignores cancellation, so a value is always returned.

```rust
use flynnel::{JobPlan, race_variants};

let plan = JobPlan::new(6, 1);
let (result, winner) = race_variants(
    &plan,
    |_cancel| Some(fast_approximation()),         // fastest, may fail
    |_cancel| Some(faithful_within_1_ulp()),      // intermediate, may fail
    |_cancel| correct_ieee_rounded(),             // always returns
);
```

Use this when you have an algorithm with a fast approximation that occasionally needs to fall back to a slow safety net.

## 8. `k_join::<K, ...>(a, b)`: const-generic fork-join

Const-generic variant of `join` that resolves the `K <= 4` branch at compile time. Small-K joins skip the plan-construction and arena dispatch entirely; larger-K joins fall through to the arena.

```rust
use flynnel::k_join;

// K=4 (const): compiler inlines both closures, zero scheduler cost.
let (a4, b4) = k_join::<4, _, _, _, _>(
    || (0..100u64).sum::<u64>(),
    || (100..200u64).sum::<u64>(),
);

// K=8 (const): compiler emits an arena call.
let (a8, b8) = k_join::<8, _, _, _, _>(
    || (0..1000u64).sum::<u64>(),
    || (1000..2000u64).sum::<u64>(),
);
```

## 9. `k_join_with_plan::<K, ...>(plan, a, b)`: const-generic + explicit plan

Same as `k_join` but takes an explicit `JobPlan`. Use this when the const-generic dispatch matters but you also want to pin `hw_class` / `variant` / `numa_hint` / bisect variant.

```rust
use flynnel::{BisectVariant, DispatchProfile, JobPlan, k_join_with_plan};

let plan = JobPlan::set_profile(8, 2048, DispatchProfile::PortBound)
    .with_bisect_variant(BisectVariant::RayonStyleReplenish);
let (a, b) = k_join_with_plan::<8, _, _, _, _>(
    &plan,
    || (0..1024u64).sum::<u64>(),
    || (1024..2048u64).sum::<u64>(),
);
```

## Hybrid CPU + GPU (`join_hybrid`, `hybrid_pipeline`)

The extended-Flynn MIMT axes are demonstrated with real GPU kernels in:

- [`examples/tpu_jax_demo.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/tpu_jax_demo.rs): TPU JAX backend integration
- [`benches/mimt_coupled.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/benches/mimt_coupled.rs): Metropolis MCMC, batched CG, MCTS with CUDA

Build with the appropriate backend feature:

```bash
# TPU backend:
cargo run --example tpu_jax_demo --release --features tpu-jax-reference

# CUDA backend (mimt_coupled bench):
cargo bench --bench mimt_coupled --features cuda-reference
```

The shape of the API:

```rust
use flynnel::{JobPlan, join_hybrid};

let plan = JobPlan::new(8, 1024)
    .with_backend(flynnel::Backend::Cuda { device_id: 0 });

let (cpu_result, gpu_result) = join_hybrid(
    &plan,
    || cpu_side_work(),
    || gpu_side_work(),
);
```

And for the iterated pipeline shape:

```rust
use flynnel::{JobPlan, hybrid_pipeline};

let plan = JobPlan::new(8, 1024);
let outputs: Vec<f32> = hybrid_pipeline(
    &plan,
    0..16u64,                          // iteration seeds
    |seed| cpu_pre_stage(seed),        // CPU: prepare for GPU
    |prepared| gpu_stage(prepared),    // GPU: kernel + sync
    |gpu_out| cpu_post_stage(gpu_out), // CPU: consume GPU result
);
```

After pipeline fill, throughput approaches `1 / max(t_pre, t_gpu, t_post)` per iteration: the smaller stages hide behind the largest.

## `JobPlan` builders

Every primitive on this page takes a `&JobPlan`. Available constructors + builders:

| Method | Purpose |
|---|---|
| `JobPlan::new(k_outer, batch_size)` | Adaptive default. Reads `active_dispatch_profile()` at construction. |
| `JobPlan::set_profile(k_outer, batch_size, DispatchProfile)` | Pins a specific profile (`LatencyBound` / `PortBound` / `MemoryBound` / `Streaming` / `Unspecified`). |
| `JobPlan::bare(k_outer, batch_size)` | Profile-independent baseline: `use_smt=false`, `estimated_per_item_ns=None`, `oversubscription_log2=None`. |
| `JobPlan::for_op_generic(k_outer, batch_size, op)` | Classify from an `OpClass` impl. Used by domain enums. |
| `.with_estimated_per_item_ns(ns)` | Set the per-item cost hint. Marks it authoritative; disables the probe path. |
| `.with_leaf_shape(LeafShape)` | Strongest static hint. Bypasses (ns, K, N) heuristics. |
| `.with_bisect_variant(BisectVariant)` | Pin the bisect variant. Default: `None` (adaptive routing decides). |
| `.with_backend(Backend)` | Route to a registered backend by id. Falls back to `CpuBackend` if unregistered. |
| `.with_numa_hint(node_id)` | Constrain to a NUMA node. `NUMA_NODE_LOCAL` for the caller's node. |
| `.with_hw_class(HwClass)` | Set the target hardware class. Default: `Scalar`. |
| `.with_variant(Variant)` | Set the quality-of-result variant: `Approx` / `Faithful` / `Correct`. |

Full field-level reference at [JobPlan Reference](../reference/JobPlan-Reference.md).

## Runtime observation and migration

The scheduler observes leaf execution times and can migrate the process-global classification atomically:

```rust
use flynnel::WorkloadClass;
use flynnel::sched::adaptive_profile::migrate_workload_class;

// Force the classification to LatencyBound for the next dispatches:
migrate_workload_class(WorkloadClass::LatencyBound);
```

The migration is a single `AtomicU8::Release-store`; the `AdaptiveDispatcher` surface reads the new value on its next plan construction via a single `AtomicU8::Acquire-load`. Plain `JobPlan::new` routes from its own static classifier on `(K, batch)` and refines per call site, so a global flip steers the dispatcher surface without contaminating unrelated call sites.

## Where to go next

- [JobPlan Reference](../reference/JobPlan-Reference.md): every field on the plan struct
- [Sched Module Reference](../reference/Sched-Module-Reference.md): all `sched::*` primitives including specialized variants (`for_each_chunk_indexed`, `for_each_chunk_triple`, `par_map_serial_reduce`, `run_in_region`, etc.)
- [Backend System](../reference/Backend-System.md): registering CPU / GPU / TPU / custom backends
- [Architecture Overview](../explanation/Architecture-Overview.md): how the layers fit together

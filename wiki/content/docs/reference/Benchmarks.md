---
title: Benchmarks
weight: 7
---

Reference guide to Flynnel's bench harness. All benches live under `benches/` and use [criterion 0.8.2](https://crates.io/crates/criterion) unless the `[[bench]]` entry sets `harness = false` (in which case they are standalone binaries with `fn main()`).

## Reproducing the numbers

```bash
# Data-parallel + inline-collapse
cargo bench --bench inline_collapse

# Cold-cache one-off dispatch latency (harness=false)
cargo bench --bench cold_workloads

# Cross-batch-size, cross-per-item-cost sweep
cargo bench --bench parameter_sweep

# Scheduler primitive isolation
cargo bench --bench sched_overhead_isolation

# Full Flynn-axes (needs CUDA)
cargo bench --bench flynn_axes --features cuda-reference

# Pipelined MIMT coupled algorithms
cargo bench --bench mimt_coupled --features cuda-reference
```

Criterion writes per-group HTML reports to `target/criterion/<group>/<bench>/report/index.html`; open `target/criterion/report/index.html` for the dashboard. The standalone-binary benches (`cold_workloads`, others tagged `harness = false` in `Cargo.toml`) write their table to stdout.

## Bench categories

Flynnel's benches split into four categories with different purposes:

| Category | Bench files | What it measures |
|---|---|---|
| Dispatch overhead isolation | `sched_overhead_isolation`, `join_overhead`, `dispatcher_routing` | Per-call scheduler cost (nanoseconds) on empty / near-empty closures. Isolates the plumbing cost from the workload cost. |
| Data-parallel workloads | `inline_collapse`, `parameter_sweep`, `cold_workloads` | End-to-end wall-clock on realistic parallel workloads across per-item-cost and batch-size axes. |
| Cross-process backend | `chase_lev_mmf` | Per-op cost of the memory-mapped Chase-Lev deque that backs the `shared-memory-worker-reference` cross-process dispatch. |
| Wait-strategy A/B | `parker_wait_strategy` | Parker wake-latency across yield / park / WAITPKG modes; the input to `WaitStrategy::pick`. |
| Cross-mode dispatch | `flynn_axes`, `mimt_coupled`, `simc_cooperative`, `simc_n12_bisect` | The SIMD / SIMC / MIMD / MIMT / MISD extended-Flynn-taxonomy path costs, including CPU+GPU hybrid dispatch (`join_hybrid`, `hybrid_pipeline`, `cooperative_join_n`, `race_variants`). |

## Bench audit (per the hard rule)

Every bench in this tree was audited against three questions before publication:

1. Does the bench invoke the primitive's named feature? Example: a "join_overhead" bench must actually call `join()` on the arena, not a hand-rolled variant.
2. Does it impose surplus locks / allocs / indirection vs baseline? Example: using `Vec::new()` inside the measured body inflates times relative to a hoisted allocation.
3. Is the primitive sized / configured for the workload? Example: a `MIN_LEAF_ITEMS=256` bench floor kills granularity for a 4-item workload, understating parallelism.

Audit comments live inline in the bench source above each contender.

## Which `JobPlan` the published bench data uses

Bench transparency: every flynnel cell in this tree measures the adaptive default (`JobPlan::new(K, batch)`) unless the row name says otherwise. Concretely:

- Adaptive rows (headline cells): `JobPlan::new(K, n)` constructed inside `iter_batched_ref`. `JobPlan::new` consults the process-global `active_dispatch_profile()` via one `AtomicU8::Acquire-load` at construction; `KGating::Auto` lets the AdaptiveWorker pick KHL vs Fcl per worker.
- `flynnel_for_profile_*` rows (variant cells): `JobPlan::set_profile(K, n, DispatchProfile::*)` constructed inside `iter_batched_ref`. These pin the override for the row's named profile (`LatencyBound` activates SMT siblings; `PortBound` parks them; `MemoryBound` activates them with cache-miss interleaving defaults).
- `flynnel_v_*` rows (per-variant cells): `JobPlan::new(K, n).with_bisect_variant(v)` for one of the two pinned [`BisectVariant`](Foundation-Types-Reference.md#bisectvariant) entries. On AMD + `PortBound` + the right batch size the adaptive routing resolves to the same variant automatically (see [`adaptive_variant_routing`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/adaptive_variant_routing.rs)).

Adaptive-switching coverage. The `JobPlan::new` to `active_dispatch_profile()` read AND the runtime `migrate_workload_class()` propagation path are verified by [`tests/adaptive_jobplan_streaming.rs`](https://github.com/markusmcnugen/flynnel/blob/main/tests/adaptive_jobplan_streaming.rs):

1. `jobplan_new_reads_active_dispatch_profile_at_construction`: cycle the global through PortBound to LatencyBound to MemoryBound to PortBound and assert `JobPlan::new`'s `use_smt` + `estimated_per_item_ns` fields match the active profile's defaults on each transition.
2. `jobplan_bare_ignores_active_profile`: `JobPlan::bare(K, batch)` returns the profile-independent baseline regardless of the global state.
3. `mid_stream_migration_propagates_within_bounded_iters`: a producer thread constructs `JobPlan::new` in a tight loop while an observer thread calls `migrate_workload_class(LatencyBound)` mid-stream. Asserts the producer observes the new `use_smt = true` plan within a bounded number of post-migration iterations.
4. `concurrent_producers_both_observe_oscillation`: two producer threads stream `JobPlan::new` while a third oscillates the class between `PortBound` and `LatencyBound` every 5 ms for 100 ms; each producer must observe BOTH SMT-on AND SMT-off plans.

## Default thread count

| Config | Default thread count on 16-logical-thread host |
|------|------------------------------------------------|
| `JobPlan::new()` (adaptive default) | 16 (all logical threads) |
| `FLYNNEL_SCHED_PHYSICAL_ONLY=on` | 8 (physical cores only) |
| `FLYNNEL_SCHED_WORKERS=N` | N per NUMA node |

Flynnel's out-of-the-box default uses all logical threads. Users with IMUL-saturated workloads where SMT siblings contest the execution port can opt into physical-cores-only via `FLYNNEL_SCHED_PHYSICAL_ONLY=on`.

## Bench harness details

### `cold_workloads` (standalone binary)

`benches/cold_workloads.rs` runs each workload shape 10 times with a mandatory 100 ms sleep between samples. The sleep forces the JEC sleep coordinator to park workers between calls, matching the real one-off dispatch latency that CLI tools / notebook cells / API handlers see, as opposed to criterion's iter-back-to-back pattern that keeps workers hot and hides the wake-on-push latency.

Shapes covered:
- `nmfd_5x100ms`: 5 items x 100ms each (small-N heavy-per-item)
- `shallow_4x10ms`, `shallow_8x10ms`, `shallow_16x10ms`: shallow bisect depth x 10ms items
- `medium_32x1ms`, `medium_128x500us`: balanced medium workloads
- `deep_1024x100us`: deep-recursion bisect
- `stream_16k_10us`: streaming many-light-items

Each cell reports median + p10 + p90 durations to stderr and appends one markdown row to stdout.

### `parameter_sweep` (criterion)

`benches/parameter_sweep.rs` sweeps 4 per-item cost profiles (`10ns`, `1us`, `100us`, `10ms`) x 10 sizes (`[1, 2, 4, 8, 16, 32, 64, 128, 1024, 10000]`) x 3 contenders (serial baseline, `flynnel_default`, `flynnel_hinted`). Skips cells where total work would exceed ~5 seconds. Uses `sqrt_chain(seed, iters)` as the workload primitive (compiler can not elide because the result feeds the next iteration's input).

### `sched_overhead_isolation` (criterion)

`benches/sched_overhead_isolation.rs` measures the near-zero-work case: `join(&plan, || (), || ())` and single-item `for_each_chunk`. Isolates per-dispatch plumbing cost from the workload cost. Numbers here are the floor below which no scheduler primitive can go.

### `join_overhead` (criterion)

`benches/join_overhead.rs` covers `join()`, `join_context()`, `join_default()` with light bodies. Isolates the per-call cost of each `join` surface at the shipping default so the overhead each entry point pays is directly comparable.

### Cross-process backend microbench

`benches/chase_lev_mmf.rs` measures per-op push / steal / pop cost of the memory-mapped Chase-Lev deque that backs the [`shared-memory-worker-reference`](https://github.com/markusmcnugen/flynnel/blob/main/src/backend/shared_mem/) cross-process backend. The numbers characterize the mmap-backed deque under the same push / steal patterns the in-process `chase_lev_local` deque handles.

### `parker_wait_strategy` (criterion)

Compares the three worker wait strategies: pure `thread::yield_now` spin, `thread::park` with wake counter, and Intel `WAITPKG` (UMWAIT) on hosts where it is available. Used to justify the sleep-tier defaults in [`sched::sleep`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/sleep.rs).

### Flynn-axes bench (`flynn_axes` + `simc_*` + `mimt_coupled`)

The full extended-Flynn-taxonomy coverage:

- MIMD: baseline `for_each_chunk` data-parallel path
- SIMC: `cooperative_join_n` cross-core cooperative SIMD (SIMT-style within a CPU worker set)
- MISD: `race_variants` speculative variant racing (approx / faithful / correct arms racing)
- SIMT: per-call H2D, persistent device buffer, warp-cooperative CUDA kernel
- MIMT single-pair: `join_hybrid` for one CPU + one GPU stage
- MIMT pipelined: `hybrid_pipeline` for iterated coupled CPU+GPU algorithms (MCMC / batched CG / MCTS-NN)

## Configuration overrides

| Env var | Effect |
|---|---|
| `FLYNNEL_SCHED_PHYSICAL_ONLY=on` | Cap worker pool to physical cores (no SMT siblings). |
| `FLYNNEL_SCHED_WORKERS=N` | Set worker count per NUMA node to `N`. |
| `FLYNNEL_SCHED_STRATEGY=yield\|park\|waitpkg` | Force worker wait strategy. Default: auto-picked by [`WaitStrategy::pick`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/sleep.rs). |
| `FLYNNEL_TRACE_DISPATCH=1` | Emit per-join dispatch trace to stderr (JOIN_CALL_COUNT / JOIN_A_BODY_NS / JOIN_WAIT_NS accumulators). |
| `FLYNNEL_LOCKLATCH_DIAGNOSE=1` | Diagnostic wait/exit tracing on `LockLatch`. |

Full list at [Environment Variables reference](Environment-Variables.md).

## Bench hosts

The recorded cross-host numbers on this page and in the README come from sweeps on a local Windows 4C/8T box, a Linux VM on Zen3 (16 threads), a Linux VPS on EPYC (32 threads), a Colab Xeon Cascade Lake (12T + L4), and a Zen+ Ryzen 7 2700 (16T + RTX 3070). Raw sweep artifacts are not kept in the repository; rerun the named bench on the target host to reproduce a number.

---
title: Architecture Overview
weight: 1
---

Flynnel is a layered work-stealing scheduler. This page walks the layers top-down so you can see how a call to `flynnel::join` flows through the crate.

## The layers

```text
                          .---------------------------------.
  Public top-level API    |  flynnel::join,                 |
                          |  flynnel::for_each_chunk,       |
                          |  flynnel::cooperative_join_n,   |
                          |  flynnel::join_hybrid, ...      |
                          '---------------+-----------------'
                                          |
                          .---------------v-----------------.
  JobPlan + tier picker   |  flynnel::JobPlan               |
  (adaptive by default;   |   - K_gating = Auto (KHL vs Fcl)|
   overrides per-call)    |   - static classifier picks the |
                          |     call-1 WorkloadClass        |
                          |   - set_profile / with_workload_|
                          |     shape pin a single call     |
                          |  sched::pick_tier               |
                          '---------------+-----------------'
                                          |
                .-------------------------+-------------------.
                |                         |                   |
                v                         v                   v
        .---------------.       .------------------.   .--------------.
  Tier  |  Inline       |       |  Local           |   |  Hierarchic. |
        |  (serial)     |       |  (1 NUMA node)   |   |  / Federated |
        '---------------'       '---------+--------'   '------+-------'
                                          |                   |
                                          v                   v
                                .--------------------------.
                                |  NumaArena               |
                                |   .--- LocalArena (n0) --.
                                |   '--- LocalArena (n1) --|
                                '---------+----------------'
                                          |
                                          v
                                .---------------------------.
                                |  WorkerCtx + Parker       |
                                |   - AdaptiveWorker (KHL / |
                                |     Fcl K_inner=3, swap   |
                                |     via AtomicU32 tag)    |
                                |   - FlynnelRing mailbox   |
                                |   - Injector (Vyukov MPMC)|
                                |   - 4-state CoreLatch     |
                                '---------------------------'
                                          |
                                          v
                                .---------------------------.
                                | In-house primitives only: |
                                |   chase_lev_local         |
                                |   + flynnel_ring          |
                                |   + injector              |
                                |   + notify_ring (Parker)  |
                                |   + std::thread workers   |
                                '---------------------------'

  Orthogonal: Backend system
        flynnel::Backend / DispatchBackend / registry
          - CpuBackend (default, wraps the arena above)
          - CudaBackend (cuda-reference feature)
          - TpuJaxBackend (tpu-jax-reference feature)
          - WasmBackend (wasm-reference feature)
          - SharedMemoryChaseLevBackend (shared-memory-worker-reference feature)
          - consumer-supplied backends via register_backend
```

## Walkthrough: `flynnel::join(&plan, a, b)`

1. **Top-level alias** resolves to `sched::arena::join` (one `pub use` indirection).
2. `join` is a thin wrapper over `join_context` that drops the `injected` / `stolen` callback parameters.
3. `join_context` calls [`pick_tier(plan, numa_topology())`](Sched-Module-Reference.md#pick_tier) to pick a [`SchedTier`](Foundation-Types-Reference.md#schedtier) from `plan.k_outer` + `plan.batch_size` + the cached NUMA topology.
4. For `SchedTier::Inline` the call resolves to `inline_join_context(a, b)`: closure `a` runs first on the calling thread, then closure `b`. Returns immediately.
5. For `SchedTier::Local` (or `Hierarchical` / `Federated`, which collapse to local on single-NUMA hosts) the call dispatches into `local_join_context`. Two paths:
   - **In-worker fast path** (calling thread is a Flynnel worker, observable via `current_worker_ctx()`): push the right-half `StackJob` onto the worker's own Chase-Lev deque (LIFO, single-owner-writer, ~5 ns), run `a` inline, then pop the right-half back. If it's still there, run it inline with `stolen = false`. If a peer thief took it, drain the local deque and steal until the latch flips.
   - **External path** (calling thread is not a worker): push the right-half job to the `NumaArena`'s `Injector` (the in-house FlynnelRing-backed global MPMC queue), run `a` on the calling thread, then call `arena.try_run_one()` cooperatively until the latch flips.
6. The right-half `StackJob` lives on the caller's stack; the worker that executes it writes the result into the embedded slot and then sets the `CoreLatch` with a `swap(SET, AcqRel)`. The caller waits on the latch before returning, so the stack-resident job is never freed while a worker is still reading it.
7. The return tuple `(RA, RB)` is in caller-supplied order regardless of which thread executed which half. This is the determinism contract bit-exact reductions rely on.

## Walkthrough: `flynnel::for_each_chunk(&plan, &mut slice, op)`

1. Top-level alias resolves to `sched::par_iter::for_each_chunk`.
2. The function queries `global_local_arena().total_workers()` and reads the observer-tuned `split_multiplier()` to compute `max_budget = workers * multiplier` (default multiplier is 2: 2x oversubscription so steals have headroom).
3. `bisect(plan, items, &op, max_budget, max_budget, migrated=false)` recurses: at each level it splits the slice at the midpoint and calls `join_context`.
4. The recursion bottoms out when:
   - `items.len() <= MIN_LEAF_ITEMS` (256), or
   - `splits == 0` and `migrated == false` (no steal pressure, budget exhausted).
5. **SLAW adaptive replenish**: each `join_context` reports back via the `stolen` flag whether the right-half was peer-stolen. If yes, `migrated = true` propagates to the recursive call, and that subtree replenishes its budget to `max_budget`. This is the SLAW ("Scalable Locality-aware Adaptive Work-stealing") pattern from Guo-Zhao-Cavé-Sarkar (IPDPS 2010): more splits where there is observed contention, fewer where workers are saturated.

## Backend dispatch (orthogonal to the arena)

The [`flynnel::backend`](Backend-System.md) module is an orthogonal layer for routing work off-CPU. The arena above is what `CpuBackend` wraps; CUDA / ROCm / Metal / TPU / ANE / WASM / SharedMemoryWorker / Custom backends are pluggable via the registry. The shared-memory worker backend is the off-process equivalent of `CpuBackend`: same trait, same routing surface, peer-process execution over a lock-free MMF ring. See [Shared-Memory Worker Backend](Shared-Memory-Worker-Backend.md).

`JobPlan::pick_backend()` is the single resolution point:

1. If `plan.backend_hint` is `Some(b)` AND a backend with id `b` is registered, return that backend.
2. Otherwise return `cpu_backend()` (the always-available default).

`flynnel::join_hybrid(plan, cpu_work, gpu_work)` runs the CPU half on the calling thread and the GPU half on whatever `pick_backend()` returns. This is the MIMT entry point.

## Topology probes

Three small modules probe host hardware at startup; each caches its result in a `OnceLock`:

| Module | What it probes | Surface |
|--------|---------------|---------|
| [`numa_topology`](NUMA-And-Topology.md#numatopology) | `/sys/devices/system/node/*` (Linux) or `GetLogicalProcessorInformationEx` (Windows) for per-CPU node membership and SLIT distances | `NumaTopology`, `numa_topology()` |
| [`cpu_info`](NUMA-And-Topology.md#cpuinfo) | `std::thread::available_parallelism` + CPUID HTT bit for SMT factor | `CpuInfo`, `cpu_info()` |
| [`numa_latency`](NUMA-And-Topology.md#numalatencytable) | Ping-pong cache-line round-trip between pinned cores (calibrated) | `TopologyLatencyTable`, `topology_latency_table()` |

## Why these layers exist

- **Tier picker** separates "what work is this?" from "how should it run?". The same primitive code calls work at K=4 and K=12 without conditionalizing on size; the picker handles the per-call dispatch decision.
- **Adaptive plan** separates "what shape is this workload right now?" from "what knobs does the scheduler turn?". `JobPlan::new` is adaptive: a static classifier on `(K, batch)` picks the `WorkloadClass` (FineGrain / PortBound / LatencyBound / MemoryBound / Streaming) for call-1 routing, K_gating (KHL vs Fcl K_inner=3 backing) is `Auto` and resolved per-host, and per-call-site learned classes refine unpinned plans on later calls. The process-global class drives the `AdaptiveDispatcher` surface; `migrate_workload_class()` flips it in one atomic store. `JobPlan::set_profile(K, batch, profile)` and `with_workload_shape(WorkloadShape::*)` are explicit per-call overrides for sites where the caller knows the profile better than calibration would.
- **Worker arena** separates parallelism mechanics (deque, latch, sleep) from the scheduling policy. The arena exposes only `submit`, `try_run_one`, `acquire_smt`; the call sites above pick the policy.
- **Backend system** separates "where does this run?" from "how does it run?". A CUDA backend reuses the same `JobPlan` plumbing; a Custom consumer backend slots in alongside.

## Where to look next

| Question | Page |
|----------|------|
| What primitives can I call? | [Sched Module Reference](Sched-Module-Reference.md) |
| What does the `JobPlan` actually contain? | [JobPlan Reference](JobPlan-Reference.md) |
| How do I write my own backend? | [How To Write A Backend](How-To-Write-A-Backend.md) |
| What's the Chase-Lev deque / latch / sleep state machine? | [Internals: Work-Stealing](Internals-Work-Stealing.md) |
| Why this design? Comparisons to other schedulers? | [Comparison To Other Schedulers](Comparison-To-Other-Schedulers.md) |

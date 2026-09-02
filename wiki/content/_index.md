---
title: Flynnel Wiki
toc: false
---

<p align="center">
  <img src="flynnel-logo.png" alt="Flynnel logo" width="200" />
</p>

A K-aware, NUMA-aware work-stealing scheduler for Rust with extended-Flynn-taxonomy dispatch. Pun-named for [Michael J. Flynn](https://en.wikipedia.org/wiki/Michael_J._Flynn) - the Stanford computer architect whose 1966 taxonomy of computer architectures (SISD / SIMD / MISD / MIMD) underlies the per-call execution-class plan this crate exposes.

{{< cards >}}
  {{< card link="docs/tutorials/" title="Tutorials" subtitle="Learning-oriented guides. Start with Getting Started for the 5-minute join + for_each_chunk quickstart." icon="academic-cap" >}}
  {{< card link="docs/how-to/" title="How-to" subtitle="Task-oriented recipes. Write a backend, drive the CUDA or TPU reference targets." icon="cog" >}}
  {{< card link="docs/explanation/" title="Explanation" subtitle="Background and rationale. Architecture, extended Flynn taxonomy, NUMA, work-stealing internals, comparison vs other schedulers." icon="book-open" >}}
  {{< card link="docs/reference/" title="Reference" subtitle="API surface for lookup. JobPlan, sched primitives, foundation types, backend system, env vars, glossary, benchmarks." icon="document-text" >}}
{{< /cards >}}

## Quick links

- **New here?** [Getting Started](Getting-Started.md) - 5-minute quickstart with `join` + `for_each_chunk`.
- **Architecture orientation?** [Architecture Overview](Architecture-Overview.md) - the layered design from `JobPlan` down to Chase-Lev deques.
- **Picking a tier or backend?** [JobPlan Reference](JobPlan-Reference.md) and [Backend System](Backend-System.md).
- **Comparing schedulers?** [Comparison To Other Schedulers](Comparison-To-Other-Schedulers.md).
- **Need numbers?** [Benchmarks](Benchmarks.md) - the internal bench harness organized by category (dispatch overhead / data-parallel workloads / deque backing / cross-mode dispatch).

## What Flynnel is

A single dispatch crate that covers the full extended-Flynn axis space in one cohesive surface:

| Acronym | Expansion                                  | Flynnel entry point                        |
|---------|--------------------------------------------|--------------------------------------------|
| SISD    | Single Instruction, Single Data            | `K_core = 0` inline execution              |
| SIMD    | Single Instruction, Multiple Data          | `K_hardware >= 1` vector lanes (in-kernel) |
| MISD    | Multiple Instruction, Single Data          | `flynnel::race_variants`                   |
| MIMD    | Multiple Instruction, Multiple Data        | `flynnel::join` + work-stealing            |
| SIMC    | Single Instruction, Multiple Cores         | `flynnel::cooperative_join_n`              |
| MIMC    | Multiple Instruction, Multiple Cores       | `flynnel::cooperative_join_n` with heterogeneous closures (different role per closure) |
| SIMT    | Single Instruction, Multiple Threads       | `flynnel::DispatchBackend::dispatch_parallel_for` |
| MIMT    | Multiple Instruction, Multiple Threads     | `flynnel::join_hybrid` (single-pair) / `flynnel::hybrid_pipeline` (streamed) |

See [Extended Flynn Taxonomy](Extended-Flynn-Taxonomy.md) for the full mapping with rationale.

## What's adaptive at runtime

Every `JobPlan::new` call runs a static classifier on `(K, batch)` that picks a `WorkloadClass` controlling SMT activation, per-element cost estimate, and oversubscription factor - correct routing on call 1 without any per-call user code. Later calls at the same call site refine from that site's learned class. The process-global active class drives the `AdaptiveDispatcher` surface; `flynnel::sched::adaptive_profile::migrate_workload_class(...)` flips it in one atomic store, and the closing-loop observer migrates it automatically from measured leaf times.

Three adaptive surfaces compose without user coordination:

- **K_gating** - every worker's per-tier deque backing (KHL per-slot Vyukov vs Fcl counter-only Chase-Lev) swaps via `AtomicU32::Release-store`. Calibrated at startup by the host probe; swappable at runtime via `migrate_all_workers_k_gating`.
- **WorkloadClass** - process-global atomic that drives the default `DispatchProfile` for `AdaptiveDispatcher` plans. `set_profile(...)` and `with_workload_class(...)` override it per-call.
- **Backend** - `migrate_backend(Backend::Cuda { device_id: 0 })` re-points dispatch through a different registered backend; `resolve_active_backend()` falls back to CPU when the requested backend is not registered.

A full end-to-end demo covering all three sits at [`examples/adaptive_dispatcher_demo.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/adaptive_dispatcher_demo.rs); run with `cargo run --release --example adaptive_dispatcher_demo` (add `--features cuda-reference,wasm-reference` to engage real backends).

## What Flynnel is not

- **Not an async runtime.** No `Future`, no `async`/`await`, no I/O reactor. For async I/O use `tokio` or `async-std`.
- **Not a job scheduler service.** No persistent queues, no cross-process work distribution, no retry storage. The scope is in-process compute parallelism.
- **Not a GPU kernel codegen toolkit.** The `cuda-reference` and `tpu-jax-reference` backends launch consumer-supplied PTX / Python kernels; Flynnel doesn't compile Rust to GPU code.

## Wiki contents

The pages below are organised under the four Diataxis sections (cards above). Each page is also reachable directly:

### Tutorials

- [Getting Started](Getting-Started.md) - 5-minute walkthrough that gets you from `cargo add flynnel` to running parallel work.
- [All Primitives Tutorial](All-Primitives-Tutorial.md) - runnable walk-through of every workhorse Flynnel primitive (`join`, `join_context`, `join_default`, `for_each_chunk`, `cooperative_join_n`, `race_variants`, `k_join`, `k_join_with_plan`, plus hybrid dispatch). Matches [`examples/tutorial_all_apis.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/tutorial_all_apis.rs) end-to-end.

### How-to

- [How To Write A Backend](How-To-Write-A-Backend.md) - implementing your own `DispatchBackend` and registering it with the Flynnel registry.
- [Reference Backends: CUDA, TPU, and WASM](Reference-Backends-CUDA-And-TPU.md) - the three in-process reference backends Flynnel ships.
- [How To Use The Shared-Memory Worker Backend](How-To-Use-The-Shared-Memory-Worker-Backend.md) - cross-process dispatch into peer workers over an MMF Chase-Lev deque + latch arena.

### Explanation

- [Architecture Overview](Architecture-Overview.md) - the layered design top-down from `JobPlan` to Chase-Lev deques.
- [Extended Flynn Taxonomy](Extended-Flynn-Taxonomy.md) - why the crate is named after Flynn and how the eight-axis mapping organises every primitive.
- [Comparison To Other Schedulers](Comparison-To-Other-Schedulers.md) - Flynnel vs rayon, Cilk, tokio, std::thread.
- [Internals: Work-Stealing Algorithm](Internals-Work-Stealing.md) - the low-level mechanics inside the CPU arena.
- [NUMA and Topology](NUMA-And-Topology.md) - how Flynnel probes the host hardware.
- [Shared-Memory Worker Backend](Shared-Memory-Worker-Backend.md) - the off-process equivalent of `CpuBackend`.

### Reference

- [JobPlan Reference](JobPlan-Reference.md) - every field on the plan struct.
- [Foundation Types Reference](Foundation-Types-Reference.md) - `Variant`, `SchedTier`, `HwClass`, `DispatchProfile`, `OpClass`.
- [Sched Module Reference](Sched-Module-Reference.md) - every primitive in `flynnel::sched`, organised by Flynn axis.
- [Backend System](Backend-System.md) - the `DispatchBackend` trait + registry.
- [Environment Variables](Environment-Variables.md) - every env var Flynnel honors at startup.
- [Glossary](Glossary.md) - terms specific to Flynnel and the broader extended-Flynn vocabulary.
- [Benchmarks](Benchmarks.md) - the internal bench harness organized by category (dispatch overhead / data-parallel workloads / deque backing / cross-mode dispatch).

## License

MIT - see [LICENSE](https://github.com/markusmcnugen/flynnel/blob/main/LICENSE) on the repository.

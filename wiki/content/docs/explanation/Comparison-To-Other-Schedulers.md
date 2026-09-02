---
title: Comparison To Other Schedulers
weight: 3
---

Flynnel exists alongside several well-established Rust and non-Rust schedulers. This page positions Flynnel against each respectfully: what each crate is good at, what shape of workload fits, and where Flynnel's choices differ.

## TL;DR

| Scheduler | Best at | Flynnel relationship |
|-----------|---------|---------------------|
| [rayon](https://github.com/rayon-rs/rayon) | Idiomatic data-parallel iteration; great `par_iter` ergonomics | Direct lineage - Flynnel borrows rayon's `JobRef` shape, latch state machine, and JEC sleep handshake. Extends with K-axis tier picker, hybrid CPU+GPU dispatch, optional reference GPU / TPU backends. |
| [tokio](https://tokio.rs) | Async I/O, network services | Orthogonal - tokio is an async runtime for I/O; Flynnel is a CPU compute scheduler. They compose: run tokio for the I/O reactor and Flynnel for the parallel CPU work the reactor hands off. |
| [crossbeam](https://docs.rs/crossbeam) | Lock-free primitives, channels, scoped threads | Sibling - Flynnel ships its own in-house equivalents (Chase-Lev deque, FlynnelRing Vyukov MPMC, ComposedMpsc/Mpmc Lamport rings, NotifyHub) so users never pull crossbeam transitively; it reaches the dev-only graph through the `rayon` / `criterion` dev-dependencies. |
| [std::thread](https://doc.rust-lang.org/std/thread) | Long-lived OS threads | Underneath - Flynnel spawns `std::thread::Builder` workers and uses `std::thread::{park, current().unpark()}` for the sleep primitive. |
| [Cilk](https://en.wikipedia.org/wiki/Cilk) (research) | Fork-join with continuation-stealing | Conceptual ancestor - Cilk introduced the work-stealing model Flynnel and rayon both use. Cilk's `cilk_for` is structurally similar to `for_each_chunk`. |

## rayon

[rayon](https://github.com/rayon-rs/rayon) is the standard Rust crate for data-parallel iteration. Excellent ergonomics, mature implementation, broad ecosystem adoption. If you're writing `data.par_iter().map(|x| f(x)).collect()` and not thinking about K-axis dispatch, GPU offload, or NUMA-aware arenas, **rayon is the right choice** and Flynnel is not.

### What Flynnel borrows from rayon

The CPU arena's primitives are direct adaptations of rayon-core 1.13:

- **`JobRef` two-word vtable** - `pointer + execute_fn + tag bytes`. Same shape as `rayon_core::Job` for the same reasons (no `Box<dyn Trait>` allocation; thieves classify jobs by tag without dereferencing).
- **4-state `CoreLatch`** - `UNSET / SLEEPY / SLEEPING / SET` with the two-phase sleep handshake. Verbatim state machine.
- **JEC-protected sleep** - the Jobs Event Counter protocol at `src/sched/jec_sleep.rs` is a verbatim port of `rayon-core-1.13.0::sleep::{counters,mod}`; the SMT-sibling gate additionally rides a permit-based `std::thread::park` Parker.
- **Chase-Lev work-stealing deque** - shape adapted from `crossbeam::deque::{Worker, Stealer}` but reimplemented in-house at [`src/sched/chase_lev_local.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/chase_lev_local.rs) so we can expose `slot_ptr` for prefetch wiring and back the deque with the K_inner=3 adaptive backings (KHL per-slot Vyukov / Fcl counter-only Chase-Lev). Same wait-free single-owner LIFO + thief-side FIFO steal protocol per Vafeiadis et al.
- **In-worker fast path for `join`** - when `join` is called from inside a worker, push the right-half to that worker's own local deque (LIFO, ~5 ns) instead of the global injector (~5-30 us). This is rayon's central perf win; Flynnel inherits it.
- **`join_context` migrated / stolen flag** - drives the adaptive splitter the same way.

### What Flynnel adds on top

- **`JobPlan` and the tier picker.** Every call site carries a `JobPlan` rather than relying on the iterator's implicit sizing. `pick_tier(plan, topo)` picks `Inline` / `Local` / `Hierarchical` / `Federated` before any scheduling work fires; small jobs pay zero scheduler overhead instead of paying it and discovering they were too small. rayon's `par_iter` always engages the pool.
- **K-axis classifications, adaptive by default.** `JobPlan::new(K, batch)` is the adaptive entry point: `KGating::Auto` lets the scheduler pick the K_inner=3 deque backing (KHL per-slot Vyukov vs Fcl counter-only) and the process-global `WorkloadClass` is consulted per plan via a single AtomicU8 Acquire-load. `JobPlan::set_profile(K, batch, DispatchProfile::*)` is the explicit-override surface for call sites where the caller knows the profile better than calibration (pins SMT activation, per-element cost, oversubscription factor, mailbox routing, and deque tier hint for one call). rayon does not classify ops; the pool is one fixed size.
- **NUMA-aware arena composition.** `NumaArena` runs one `LocalArena` per NUMA node with per-node CPU pinning and leader-driven cross-node steals. rayon's pool is single global; multi-NUMA distribution depends on OS scheduling.
- **Backend system.** `flynnel::join_hybrid(plan, cpu_work, gpu_work)` routes one half to CPU and the other to a registered GPU / TPU backend. rayon has no GPU dispatch.
- **MISD variant racing.** `flynnel::race_variants(plan, fast, faithful, correct)` is a first-class primitive. rayon has no shape that fits.
- **Cooperative cross-core vector.** `cooperative_join_n` runs N closures as one mega-vector with a single sync boundary. rayon has nested `join` but no first-class N-way cooperative shape.

### When to prefer rayon over Flynnel

- You want `par_iter().map().collect()` ergonomics on existing iterator chains.
- Your workload is uniform data-parallel slice work and the per-call dispatch cost is invisible to you.
- You want the largest possible community and ecosystem integration.
- You don't need GPU / TPU dispatch.

### When to prefer Flynnel over rayon

- You have a mix of micro-jobs (sub-microsecond) and macro-jobs (millisecond+) and want the tier picker to suppress dispatch on the micro side.
- You have latency-bound vs IMUL-saturated ops in the same workload and want SMT activation per op.
- You're running on a multi-NUMA host and want per-node arenas.
- You want a uniform CPU + GPU + TPU dispatch surface.
- You have a variant-racing pattern (e.g., a fast path that sometimes fails and a slow safety net).

## tokio

[tokio](https://tokio.rs) is the canonical Rust async runtime. It owns the async I/O, timer, and futures-executor surfaces.

Tokio and Flynnel solve different problems and compose:

- **Tokio** runs the I/O reactor, async timers, and the futures executor for network services, file I/O, channel-based message passing.
- **Flynnel** runs the parallel compute work the reactor hands off - image processing, matrix math, simulation steps, etc.

A typical full-stack app:

```rust
#[tokio::main]
async fn main() {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Job>(1024);

    // Spawn an HTTP server that pushes Jobs onto the channel.
    tokio::spawn(http_server(tx));

    while let Some(job) = rx.recv().await {
        // Hand off to Flynnel for parallel CPU work.
        let result = tokio::task::spawn_blocking(move || {
            flynnel::for_each_chunk(
                &flynnel::JobPlan::new(8, job.data.len() as u32),
                &mut job.data,
                |chunk| { process(chunk); },
            );
            job
        }).await.unwrap();

        send_response(result).await;
    }
}
```

Flynnel does NOT compete with tokio's reactor; using Flynnel for async I/O is a category error (Flynnel has no `Future` runtime, no I/O selector, no async timer wheel). Conversely, tokio's `spawn_blocking` pool is sized for blocking syscalls (typically 512 threads idle); using it for CPU-bound parallel work wastes threads. The composition above hands off cleanly.

## crossbeam

[crossbeam](https://docs.rs/crossbeam) is the canonical lock-free primitives crate for Rust. Flynnel's production code does NOT depend on it: `cargo tree -e normal` shows zero crossbeam nodes, so consumers never pull it transitively. It does appear in the dev-only graph (`cargo tree -e all`) through the `rayon` and `criterion` dev-dependencies that power the A/B benches. Flynnel ships in-house equivalents of every crossbeam primitive its own production code needs, listed in the table below.

Flynnel ships its own equivalents of every crossbeam primitive its production code needs:

| Flynnel primitive | Surface | Crossbeam analogue | Module |
|---|---|---|---|
| `chase_lev_local::{Worker, Stealer, Steal}` | Single-owner LIFO + multi-thief FIFO steal | `crossbeam::deque::{Worker, Stealer, Steal}` | [`src/sched/chase_lev_local.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/chase_lev_local.rs) |
| `injector::Injector<T>` | Global MPMC fork queue with Success/Empty/Retry steal | `crossbeam::deque::Injector<T>` | [`src/sched/injector.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/injector.rs) |
| `flynnel_ring::FlynnelRing<T>` | Bounded MPMC (Vyukov per-slot sequence) | `crossbeam::queue::ArrayQueue<T>` | [`src/sched/flynnel_ring.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/flynnel_ring.rs) |
| `flynnel_ring_spsc::{Producer, Consumer}` | Bounded SPSC (Lamport zero-CAS) | none (specialized) | [`src/sched/flynnel_ring_spsc.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/flynnel_ring_spsc.rs) |
| `flynnel_ring_mpsc::{MpscProducer, Consumer}` | Bounded MPSC (CAS on producer, no CAS on consumer) | none (specialized) | [`src/sched/flynnel_ring_mpsc.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/flynnel_ring_mpsc.rs) |
| `flynnel_ring_composed::{ComposedMpsc, ComposedMpmc}` | N-by-M Lamport SPSC grid; per-producer FIFO preserved | none | [`src/sched/flynnel_ring_composed.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/flynnel_ring_composed.rs) |
| `notify_ring::{NotifyHub, NotifySender, NotifyReceiver}` | Blocking send / recv on top of FlynnelRing + Parker | `crossbeam::channel::{bounded, unbounded, Sender, Receiver}` | [`src/sched/notify_ring.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/notify_ring.rs) |
| `adaptive_worker::AdaptiveWorker` | AtomicU32-tag dispatch between KHL and Fcl backings (zero per-op overhead) | none (Flynnel-specific) | [`src/sched/adaptive_worker.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/adaptive_worker.rs) |

The in-house primitives are tuned for flynnel's K-axis dispatch: Chase-Lev exposes `slot_ptr` for prefetch wiring on the steal path; FlynnelRing carries the K_inner=3 batching contract; ComposedMpsc / ComposedMpmc give per-producer FIFO with zero-CAS hot paths; NotifyHub gives crossbeam-channel's blocking semantics without a Mutex on the wake hot path (pre-allocated `Box<[OnceLock<Arc<Parker>>]>`).

If you want lower-level lock-free primitives without the full scheduler, use crossbeam directly. If you need fork-join + tier picking + NUMA + backend dispatch + the in-house primitives in one crate, use flynnel.

## std::thread

`std::thread` is Rust's OS-thread surface. Flynnel uses it for:

- Spawning workers via `std::thread::Builder` with named threads (`"flynnel-sched-{idx}"` for compute workers, `"flynnel-io-{idx}"` for IO pool workers).
- Per-worker park / unpark via `std::thread::{park, current().unpark()}`.
- Scoped threads via `std::thread::scope` for the pipeline stages and the TPU JAX bridge child management.

If your workload spawns one or two long-lived threads and needs nothing else, `std::thread` is correct and Flynnel is overkill. Flynnel's value-add is at the 10+-task fork-join scale where work stealing actually matters.

## std::sync::mpsc, pipes, sockets (cross-process IPC)

Flynnel is a *scheduler*, but its [`SharedMemoryChaseLevBackend`](Shared-Memory-Worker-Backend.md) (feature `shared-memory-worker-reference`) overlaps with the part of the design space that cross-process IPC tools occupy. Where it sits relative to common choices, measured per round-trip dispatch on Zen+ R7 2700:

| Mechanism                                    | Per-call (median) | What it costs                                                  |
|----------------------------------------------|-------------------|----------------------------------------------------------------|
| `flynnel::flat::join` (in-process)           | 16.9 ns           | Atomics-only fork-join inside one process                      |
| `SharedMemoryChaseLevBackend` (SMT-siblings) | 342 ns            | Chase-Lev push + steal + latch publish, shared L1d             |
| `SharedMemoryChaseLevBackend` (intra-CCX)    | 424 ns            | Same protocol, shared L3, two different physical cores         |
| `SharedMemoryChaseLevBackend` (unpinned)     | 533 ns            | OS-scheduled placement                                         |
| `SharedMemoryChaseLevBackend` (cross-CCX)    | 881 ns            | Cross-die coherence bounce on every round-trip                 |
| `std::sync::mpsc::sync_channel` (same proc)  | 909.5 ns          | `Mutex<Condvar>` parking path on contention                    |
| OS pipe / Unix socket round-trip (same host) | ~20-50 us         | Kernel-mediated transfer + scheduling                          |

The shared-memory backend lands in a gap nothing else fills cleanly: faster than in-process `std::sync::mpsc` in every coherence tier, AND the work runs in a different process. If your workload already fits in one address space, use the in-process scheduler. If you need process isolation but you also need many dispatches per second, the Chase-Lev MMF backend beats every kernel-mediated alternative by 25-60x. Where Flynnel does NOT compete: cross-host transport. The mmap-aliasing trick only works when both peers share a kernel page cache.

## Cilk

[Cilk](https://en.wikipedia.org/wiki/Cilk) (MIT, 1990s+) introduced the work-stealing fork-join model both rayon and Flynnel inherit from. Cilk's `cilk_for`, `cilk_spawn`, `cilk_sync` were the original ergonomics; modern descendants (Intel Cilk Plus, OpenCilk, Tapir) keep the model.

Cilk's key architectural ideas Flynnel inherits:

- **Continuation stealing** as the work-stealing target (Cilk's classical model). Flynnel currently ships **child stealing** (rayon's model) where the child task is the one that gets stolen; continuation stealing is the more recent research direction (see [Libfork](https://arxiv.org/abs/2402.18480)) Flynnel's architecture leaves room for.
- **Work-first principle** - owner runs its local work first, steals only when local is empty. The `WorkerCtx::find_work` order (local pop -> injector -> peer steal -> park) is exactly this.
- **THE protocol** (Top-Half / Exchange) for the steal-vs-pop race. Crossbeam's Chase-Lev wait-free deque implements the same idea with different atomic primitives.

Cilk research papers are still the canonical references for work-stealing analysis (greedy scheduling, span complexity, work-time bounds). Flynnel's design choices that depart from Cilk:

- **K-axis tier picker.** Cilk classes assume the work-stealing pool is always the right answer once you've forked. Flynnel adds the "or just run it inline" tier.
- **NUMA awareness.** Cilk on a single shared-memory multiprocessor predates the NUMA proliferation of the 2010s+. Flynnel's `NumaArena` composes per-node arenas with the cross-node steal pattern from [ARCAS (2025)](https://arxiv.org/abs/2503.11460) and Olivier-Prins (ROSS '11).
- **Heterogeneous dispatch.** Cilk targets CPU; Flynnel's `Backend` system extends the model to GPU / TPU / custom accelerators.

## Why Flynnel exists despite rayon

Three answers depending on your workload:

1. **You're building a compute library where the K-axis matters.** If you've designed your numerics around log2(operand-size) dispatch decisions (Karatsuba recursion, NTT, BigFloat, multi-precision arithmetic), the `JobPlan::set_profile(K, batch, DispatchProfile::*)` pattern fits naturally and the tier picker collapses sub-microsecond ops to zero overhead. rayon's `par_iter` always pays the pool-entry cost.
2. **You need uniform CPU + GPU + TPU dispatch.** rayon is CPU-only. Flynnel's `join_hybrid` + `DispatchBackend` registry is the same call site whether the GPU half runs on a real CUDA backend, a stub backend in tests, or falls back to CPU on a host without a GPU.
3. **You have a workload that fits one of the extended-Flynn axes.** Variant racing (MISD), cooperative cross-core mega-vector (SIMC), or hybrid CPU + accelerator (MIMT) are first-class primitives in Flynnel; getting them in rayon means rolling your own coordination over `rayon::scope`.

If none of the three apply, rayon is the right tool. Flynnel's design accepts being a more specialized choice in exchange for covering the dispatch axes its target workloads need.

## A note on respect

Every scheduler listed here was built by people who solved real problems thoughtfully. Flynnel's existence is not a critique of any of them; it's an exploration of what becomes possible when you take work-stealing primitives and extend them across the extended-Flynn taxonomy. The borrowing is direct and acknowledged; the divergence is documented and reasoned about. The [Benchmarks](Benchmarks.md) reference documents the internal bench harness measuring Flynnel's own dispatch overhead, cold-cache one-off latency, deque backing choices, and cross-mode extended-Flynn-axis path costs.

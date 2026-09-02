---
title: Glossary
weight: 6
---

Terms specific to Flynnel and the broader extended-Flynn-taxonomy vocabulary.

## Flynn-axis acronyms

| Acronym | Expansion | Meaning |
|---------|-----------|---------|
| SISD | Single Instruction, Single Data | One instruction stream, one data element. Scalar serial execution. |
| SIMD | Single Instruction, Multiple Data | One instruction stream, many data lanes. Vector ALUs. |
| MISD | Multiple Instruction, Single Data | Many instruction streams, one data element. Variant racing fits this shape. |
| MIMD | Multiple Instruction, Multiple Data | Many instruction streams, many data elements. Classical multicore. |
| SIMC | Single Instruction, Multiple Cores | One logical instruction across N cooperating cores. Cross-core mega-vector. |
| MIMC | Multiple Instruction, Multiple Cores | Heterogeneous instruction streams across N cooperating cores. Same primitive as SIMC (`flynnel::cooperative_join_n`) but the N closures carry different function bodies, one per role. Canonical pattern: numerical linalg "one factors, N apply"; MCTS-style "one picks, N backprop"; Bayesian inference "one proposes, N evaluate". |
| SIMT | Single Instruction, Multiple Threads | One kernel invocation, many GPU threads in lockstep. |
| MIMT | Multiple Instruction, Multiple Threads | Hybrid CPU + accelerator dispatch with different instruction streams concurrently. |

## K-axis vocabulary

| Term | Meaning |
|------|---------|
| `K_outer` | `log2(n_limbs)` for the job's operand. Primary axis the tier picker reads. |
| `K_inner` | Bits per sub-unit within an operand. Typically 5 for `u32` limbs. |
| `K_hardware` | Multiply-add ops per single instruction (lanes per instruction). |
| `K_tower` | Summed pow2 blocks in a tower decomposition (e.g., 288-bit = 256 + 32, K_tower = 2). |
| `K_class` | Distinct execution regimes per batched call. |
| `K_core` | Cores dispatched per call. |
| `K_unified` | Cores cooperating as one logical mega-vector. |

## Scheduling vocabulary

| Term | Meaning |
|------|---------|
| Tier | One of `SchedTier::{Inline, Local, Hierarchical, Federated}`. Picked per call by `pick_tier`. |
| Plan | A [`JobPlan`](JobPlan-Reference.md) struct. Carries `k_outer`, `batch_size`, `hw_class`, `variant`, `backend_hint`, and tuning hints. |
| Variant | One of `Variant::{Correct, Faithful, Fast}`. Precision contract a primitive offers. |
| HwClass | Hardware class a primitive targets (`Scalar`, `Avx512f`, `AmxBf16`, `TensorCoreHopper`, etc.). |
| Backend | One of `Backend::{Cpu, Cuda, Rocm, Metal, Tpu, Ane, Wasm, SharedMemoryWorker, Custom(u32)}`. Dispatch-target taxonomy. |
| DispatchProfile | Scheduler-native work classification (`LatencyBound`, `PortBound`, `MemoryBound`, `Unspecified`). Drives SMT activation, per-element cost estimate, and leaf-count oversubscription together. |
| OpClass | Trait every domain-specific op enum implements; carries `is_latency_bound()`. Map your enum to a `DispatchProfile` to inherit cost / oversubscription defaults via `JobPlan::set_profile`. |
| WorkloadClass | User-facing classification (`FineGrain` / `PortBound` / `LatencyBound` / `MemoryBound`) that maps to a `DispatchProfile`. The process-global active class drives `JobPlan::new`'s defaults; flip at runtime via `migrate_workload_class`. |
| WorkloadShape | Declarative shape hint (`Streaming` / `ProducerFast` / `WorkSteal` / `Cooperative` / `VariantRace`) the dispatcher maps to `(k_gating, mailbox, oversubscription)` knob triples at plan-construction time. |
| KGating | Per-worker K_inner=3 deque-backing selector (`Auto` / `CounterOnly` / `PerSlot`). `Auto` lets the per-host calibration pick KHL vs Fcl; flip every worker at runtime via `LocalArena::migrate_all_workers_k_gating`. |
| AdaptiveDispatcher | Unified user-facing dispatcher in [`flynnel::sched::dispatch`](Sched-Module-Reference.md#dispatch). Picks the Flynn-axis entry point (SISD / MIMD / SIMC / MIMC / MISD) from one `WorkloadShape` hint; exposes runtime migration for K_gating, WorkloadClass, DispatchProfile, and Backend. |
| AdaptiveWorker | Per-worker K_inner=3 deque holder. AtomicU32 tag selects between the KHL and Fcl backings; per push / pop the worker reads the tag (~1 ns) and branches. Migration is one `AtomicU32::Release-store` per worker. |

## In-house primitive vocabulary

| Term | Meaning |
|------|---------|
| Vyukov ring | Bounded MPMC queue using per-slot sequence numbers (Dmitry Vyukov, 2010). The base of `FlynnelRing<T>` and `crossbeam::queue::ArrayQueue`. |
| Lamport ring | Bounded SPSC queue with zero CAS (Leslie Lamport, 1983). Producer Release-stores `tail`; consumer Release-stores `head`; counters on separate cache lines. The base of `flynnel_ring_spsc`. |
| FlynnelRing | In-house Vyukov MPMC ring (`flynnel::sched::flynnel_ring::FlynnelRing<T>`). Used as the per-worker mailbox and the backing for `Injector` + `NotifyHub`. |
| Composed N-by-M ring | MPMC built from `N * M` per-pair Lamport SPSC rings instead of one shared Vyukov ring. Per-producer FIFO preserved; ~2x faster than Vyukov at 4P/4C buffer-normalized. Lives in `flynnel::sched::flynnel_ring_composed`. |
| Injector | Global MPMC fork queue at the arena level. External submitters push here; arena workers steal when their local deque is empty. `flynnel::sched::injector::Injector<T>` wraps `FlynnelRing<T>`; replaced the prior `crossbeam::deque::Injector` dep. |
| NotifyHub | Blocking notify-wrapper over `FlynnelRing` + per-consumer `Parker`. Standard channel surface (`send` / `recv` / `close`) without depending on `crossbeam::channel` or `std::sync::mpsc`. |
| KHL | K_inner=3 per-slot Vyukov backing of the in-house Chase-Lev deque (`flynnel::sched::khl_local`). Publication contention spread across an array of per-slot `seq` atomics; wins on store-buffer-rich cores. |
| Fcl | K_inner=3 counter-only Chase-Lev backing (`flynnel::sched::fcl_local`). Single `bottom` counter for both ordering and publication; wins on smaller-store-buffer cores. |
| K_inner = 3 | Per-slot batching factor on the in-house Chase-Lev family. Each slot holds up to 3 jobs, packing 3 jobs per cache-line transfer when a thief steals. |

## Implementation vocabulary

| Term | Meaning |
|------|---------|
| Chase-Lev deque | Wait-free single-owner double-ended queue used per worker. Owner pushes/pops at one end; thieves steal at the other. |
| Injector | MPMC queue at the arena level for cross-thread submissions. |
| Latch | One-shot signalling primitive (`CoreLatch`) transitioning monotonically `UNSET -> SLEEPY -> SLEEPING -> SET`. |
| JEC | Jobs-Event-Counter sleep-protocol: workers count observed-job events before falling asleep so a producer's submit-then-wake handshake avoids lost wakeups. |
| Parker | Per-worker sleep primitive built on `std::thread::park` with a yield-N-then-park spin floor. |
| SLAW | "Scalable Locality-aware Adaptive Work-stealing" (Guo-Zhao-Cavé-Sarkar, IPDPS 2010, pp. 1-12). The adaptive splitter pattern Flynnel uses. |
| Migrated / stolen | Flags passed by `join_context` indicating whether a sub-job was dequeued by a peer (true) or popped by the originating worker (false). Drives the adaptive replenish. |
| NUMA | Non-Uniform Memory Access. A multi-socket / multi-CCD topology where some memory addresses are faster to access from some cores. |
| CCX | Cache-Coherent Complex (AMD Zen terminology). A cluster of cores sharing one L3 slice. Detected by Flynnel via CPUID `0x8000_001D` sub-leaf 3 on `AuthenticAMD` parts. |
| CCD | Core Complex Die. A physical Zen chiplet, typically containing one or two CCXs. |
| Module / Tile / Die | Intel CPUID `1Fh` v2-extended-topology domain types (numeric 3 / 4 / 5 in `ECX[15:8]`). On Sapphire Rapids+ the Module domain size is the chiplet-tile size - cores sharing one L3 slice - and is the analogue of AMD's CCX for Flynnel's cluster-detection probe. |
| DSU | DynamIQ Shared Unit. ARMv8+ cluster construct - cores within a DSU share L2/L3 cache. Flynnel detects DSU size on aarch64 Linux via `/sys/devices/system/cpu/cpuN/topology/cluster_id`. |
| perflevel | Apple Silicon performance-level grouping. `perflevel0` = P-cores, `perflevel1` = E-cores (on M-series). Queried via `sysctl hw.perflevel0.physicalcpu` for P-cluster size on macOS aarch64. |
| SLIT | System Locality Information Table. ACPI's per-pair NUMA distance matrix. |
| MXU | Matrix Multiply Unit. The systolic array inside a Google TPU. |

## Backend vocabulary

| Term | Meaning |
|------|---------|
| SIMT width | Lanes that execute in lockstep within one launch. 1 (CPU), 32 (NVIDIA warp), 64 (AMD wave), 128 (TPU MXU lane). |
| Launch latency | Cost of dispatching a single empty kernel. CPU ~100 ns; CUDA ~10 us; TPU JAX ~100 us. |
| H2D | Host-to-Device memory transfer (CPU to GPU / TPU). |
| Kernel handle | Opaque [`KernelHandle`](Backend-System.md#kernelhandle) returned by a backend after `register_kernel`. Per-backend; not portable. |
| PTX | NVIDIA's intermediate kernel format. CUDA backend's `register_kernel` accepts PTX text. |
| dlopen probe | Detection via runtime shared-library loading (`libloading::Library::new`). Lets Flynnel verify a runtime is present without linking it at build time. |

## Reductions vocabulary

| Term | Meaning |
|------|---------|
| Tower | High-precision value represented as a sum of pow2 blocks. |
| Two-Sum | Dekker's error-free transform for floating-point addition; returns `(sum, error)` exactly. |
| Componentwise op | Per-block operation in a `Tower<T>` reduction. Independent per-block, trivially parallel. |
| Combine | Sequential fold across `Tower<T>` blocks that produces the final accumulator. |

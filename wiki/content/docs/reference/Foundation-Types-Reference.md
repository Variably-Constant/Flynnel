---
title: Foundation Types Reference
weight: 2
---

The vocabulary every Flynnel call site speaks. Nine small types: [`Variant`](#variant), [`SchedTier`](#schedtier), [`HwClass`](#hwclass), [`DispatchProfile`](#dispatchprofile), [`OpClass`](#opclass-trait), [`WorkloadClass`](#workloadclass), [`WorkloadShape`](#workloadshape), [`KGating`](#kgating), [`BisectVariant`](#bisectvariant). All are dependency-free, `Copy`, `Eq`, `Hash`.

## `Variant`

Defined in [`src/foundation.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/foundation.rs).

```rust
pub enum Variant {
    Correct,    // default
    Faithful,
    Fast,
}
```

Three-tier precision discipline a primitive may offer.

| Variant | Meaning | When to pick |
|---------|---------|--------------|
| `Correct` | Bit-exact correctly-rounded result. The verification chain (BLAKE3 over per-stripe outputs xor Merkle-root agreement across replicas) requires this variant. | Reproducibility-critical workloads; CPU/GPU bit-parity checks; distributed-replica verification. |
| `Faithful` | Within 1 ulp of the exact answer but not necessarily correctly rounded. | Default for most workloads - pays ~2x throughput tax to upgrade to `Correct`. |
| `Fast` | Best-effort result with bounded but unspecified error. | Inner-loop iterates of refinement schemes that recover precision externally (Newton-Raphson seeds, AGM start points). |

API:

- `Variant::ALL: [Variant; 3]` - ordered highest accuracy first.
- `variant.requires_bit_exact() -> bool` - true only for `Correct`.
- `Display` impl writes `"correct"` / `"faithful"` / `"fast"`.

`Variant::default() == Variant::Correct` - the safest default. `JobPlan::new` overrides to `Faithful` because most production workloads can absorb the 1-ulp envelope.

## `SchedTier`

Defined in [`src/foundation.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/foundation.rs).

```rust
pub enum SchedTier {
    Inline,        // default
    Local,
    Hierarchical,
    Federated,
}
```

Which scheduler tier runs a given job. Picked per call by [`pick_tier`](Sched-Module-Reference.md#pick_tier) from `(k_outer, batch_size, numa_topology, hw_class)`.

| Tier | Execution | When picked |
|------|-----------|-------------|
| `Inline` | Serial in caller. No scheduler overhead. | `K_outer <= 4` and `batch_size < 256`. Sub-microsecond per op; scheduling overhead dominates the actual work at this size. |
| `Local` | Single-arena work-stealing inside one NUMA node. Child-stealing, randomized victim selection, Chase-Lev deque. | `K_outer = 5..7` (or Inline-band promoted by large batch). Most production CPU work lands here. |
| `Hierarchical` | Multi-arena (one per NUMA node) with leader-driven cross-arena steals. | `K_outer = 8..10` on multi-NUMA hosts. Collapses to `Local` on single-NUMA hosts. |
| `Federated` | Multi-pool federation: per-NUMA arenas + tiered storage + per-NUMA constant replication. FLINT-style pull-pool. | `K_outer >= 11`. Massive operands where data-locality dominates dispatch cost. |

API:

- `SchedTier::ALL: [SchedTier; 4]` - ascending parallelism cost.
- `tier.spin_rounds() -> u32` - recommended `thread::yield_now()` count before parking on a condvar. `Inline=0`, `Local=8`, `Hierarchical=32`, `Federated=0`. Empirical from Zen+ and Skylake-X measurements.
- `Display` impl: `"inline"` / `"local"` / `"hierarchical"` / `"federated"`.

`SchedTier::default() == SchedTier::Inline` - the safest default (no dispatch).

## `HwClass`

Defined in [`src/foundation.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/foundation.rs).

```rust
pub enum HwClass {
    Scalar,    // default
    Sse2,
    Avx2,
    Avx512f,
    Avx512Bf16,
    Avx512Vnni,
    Neon,
    Sme,
    AmxBf16,
    AmxInt8,
    AmxFp16,
    TensorCoreHopper,
    TensorCoreBlackwell,
}
```

Hardware class a primitive may target. Orthogonal to [`SchedTier`](#schedtier). Maps the K-axis hardware regime: vector SIMD at `K_R = 0..6` and matrix-extension regime at `K_R = 10..16`.

| Class | Regime | Typical silicon |
|-------|--------|-----------------|
| `Scalar` | scalar | always available |
| `Sse2` | vector SIMD (K_R=1) | every x86_64 |
| `Avx2` | vector SIMD + FMA (K_R=2) | Haswell+, Zen 2+ |
| `Avx512f` | vector SIMD (K_R=3) | Skylake-X+, Zen 4+ |
| `Avx512Bf16` | vector SIMD bf16 (K_R=5) | Sapphire Rapids+, Zen 4 client subset |
| `Avx512Vnni` | vector SIMD int8 (K_R=6) | Cascade Lake+, Zen 4 desktop |
| `Neon` | vector SIMD (K_R=2) | ARMv8 |
| `Sme` | matrix extension (K_R=14) | ARMv9-A SME, Apple M4+ |
| `AmxBf16` | matrix extension (K_R=13) | Intel AMX BF16 (Sapphire Rapids+) |
| `AmxInt8` | matrix extension (K_R=14) | Intel AMX INT8 |
| `AmxFp16` | matrix extension (K_R=13) | Intel AMX FP16 (Granite Rapids+) |
| `TensorCoreHopper` | matrix extension (K_R=13) | NVIDIA sm_90 + WGMMA + TMA |
| `TensorCoreBlackwell` | matrix extension (K_R=15) | NVIDIA sm_100 + tcgen05 + dual-SM 256x256 MMA |

API:

- `class.is_matrix_extension() -> bool` - true for `Sme`, `Amx*`, `TensorCore*`. Matrix-extension regimes require mode-region batching (see [`mode_region::run_in_region`](Sched-Module-Reference.md#mode_region)) because kernel-entry costs amortize per region, not per op.
- `Display` impl: short lowercase names (`"avx512f"`, `"amx-bf16"`, `"tc-hopper"`, etc.).

`HwClass::default() == HwClass::Scalar`.

## `DispatchProfile`

Defined in [`src/dispatch_profile.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/dispatch_profile.rs).

```rust
pub enum DispatchProfile {
    LatencyBound,
    PortBound,
    MemoryBound,
    Unspecified,
}
```

Scheduler-native classification of a dispatch. The scheduler needs two facts about a job to dispatch it well: (1) does SMT help (latency-bound, memory-bound) or hurt (port-bound), and (2) roughly how expensive is each element. `DispatchProfile` names both together so per-call tuning (leaf count, inline-collapse, SMT activation) has the inputs it needs without the caller knowing about the underlying knobs.

The defaults per variant:

| Variant | `use_smt` | default `ns_per_elem` | `oversubscription_log2` (leaves per worker) | When to pick |
|---------|-----------|----------------------|----------------------------------------------|--------------|
| `LatencyBound` | `true` | 600 | 2 (4x) | Long dependency chains (chained sqrt / div, recursive Newton, FMA chains with reg deps, branchy adaptive integrators) |
| `PortBound` | `false` | 12 | 1 (2x) | Saturated issue port (chained IMUL, throughput-bound packed SIMD with no inter-element dependency) |
| `MemoryBound` | `true` | 50 | 1 (2x) | Cache-miss-bound (sparse matvec, large-stride memory walks, pointer-chasing, gather/scatter) |
| `Unspecified` | `false` | None | 1 (2x) | Caller has no profile info; conservative defaults; cost-derived oversubscription stays off |

`DispatchProfile` implements [`OpClass`](#opclass-trait) so it works directly with [`JobPlan::for_op_generic`](JobPlan-Reference.md#for_op_generic). The canonical constructor [`JobPlan::set_profile`](JobPlan-Reference.md#set_profile) sets all three knobs (use_smt, ns_per_elem, oversubscription_log2) in one call.

Downstream crates that want richer op classification (a math crate with kernel-specific cost estimates, a graphics crate with shader categories) implement `OpClass` on their own enum and map each variant to a `DispatchProfile` to inherit the same per-call tuning behavior.

## `OpClass` trait

Defined in [`src/op_class.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/op_class.rs).

```rust
pub trait OpClass: Copy + Clone + Eq + Hash + Debug + 'static {
    fn is_latency_bound(&self) -> bool;
}
```

Marker trait every domain-specific kernel-op enum implements. The trait surface is deliberately minimal: [`JobPlan::for_op_generic`](JobPlan-Reference.md#for_op_generic) consumes the op only to read this one boolean and store it on the plan. For the full profile defaults (cost estimate + oversubscription factor), have the domain enum map to a `DispatchProfile` and use `set_profile` instead.

### Defining a custom op enum

```rust
use flynnel::{JobPlan, OpClass};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum MyOp {
    Compress,
    Encrypt,
    HashLookup,
}

impl OpClass for MyOp {
    fn is_latency_bound(&self) -> bool {
        matches!(self, MyOp::HashLookup)   // pointer-chase = stall-heavy
    }
}

let plan = JobPlan::for_op_generic(8, 1024, MyOp::Encrypt);
// plan.use_smt == false (Encrypt is FMA/AES-saturated)
let plan2 = JobPlan::for_op_generic(8, 1024, MyOp::HashLookup);
// plan2.use_smt == true (HashLookup is latency-bound)
```

The op value is consumed at plan construction; subsequent dispatch (`join`, `for_each_chunk`, `cooperative_join_n`) reads `plan.use_smt`, not the original op.

## `WorkloadClass`

Defined in [`src/sched/adaptive_profile.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/adaptive_profile.rs).

```rust
pub enum WorkloadClass {
    FineGrain,
    PortBound,
    LatencyBound,
    MemoryBound,
    Streaming,
}
```

User-facing classification that maps to a [`DispatchProfile`](#dispatchprofile). The application observes its workload (via measured per-item elapsed time or a-priori knowledge) and calls [`migrate_workload_class(class)`](Sched-Module-Reference.md#adaptive_profile) when the active class no longer matches.

| Class | Maps to `DispatchProfile` | When to pick |
|---|---|---|
| `FineGrain` | `PortBound` | tiny per-item cost (~< 50 ns/item); dispatch overhead dominates and the inline-collapse tier picker should make the serial-vs-fork call |
| `PortBound` | `PortBound` | medium per-item cost (~50-500 ns/item); port-saturated pipeline (typically integer multiply or FMA on a single execution port) |
| `LatencyBound` | `LatencyBound` | large per-item cost (~> 500 ns/item); long FP dependency chains stall the pipeline; SMT siblings active to fill the stall bubbles |
| `MemoryBound` | `MemoryBound` | memory-bandwidth-bound irregular access (sparse matvec, pointer chase, hash probes); SMT siblings active to interleave cache misses |
| `Streaming` | `Streaming` | sequential bandwidth-bound scans (byte scan, image kernels, histogram); SMT siblings parked because both threads on a core contest the same L2/L3 bandwidth |

The mapping is intentional: `FineGrain` and `PortBound` both go through `DispatchProfile::PortBound` because the inline-collapse tier picker handles the per-call sizing decision separately from the SMT-activation decision.

API:

- `class.to_dispatch_profile() -> DispatchProfile` - the deterministic mapping.
- [`migrate_workload_class(class)`](Sched-Module-Reference.md#adaptive_profile) - flip the process-global active class via one `AtomicU8::Release-store`. Subsequent `JobPlan::new` calls anywhere in the process pick up the new class on their next construction.
- [`active_workload_class()`](Sched-Module-Reference.md#adaptive_profile) - read the current class via one `AtomicU8::Acquire-load`.

Cost: one atomic store per migration; one Acquire-load per plan construction; zero per-op cost on the dispatch hot path.

Streaming-migration behavior (consultation at plan construction, per-call escape hatch, mid-stream propagation) is covered by the unit tests in [`src/sched/adaptive_profile.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/adaptive_profile.rs).

## `WorkloadShape`

Defined in [`src/sched/workload_shape.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/workload_shape.rs).

```rust
pub enum WorkloadShape {
    Streaming,
    ProducerFast { burst: u32 },
    WorkSteal { n_consumers: u32, batch_size: u32 },
    Cooperative { n_cores: u32 },
    VariantRace,
}
```

Declarative shape hint that the dispatcher maps to `(k_gating, use_mailbox_routing, oversubscription_log2)` knob triples at plan-construction time. The application names what the workload IS rather than which knobs to turn; the scheduler maps shape -> knobs once at `JobPlan::with_workload_shape(...)` time and the per-call dispatch path stays direct atomic ops.

| Shape | Flynn axis | Mapped knobs (per `WorkloadShape::hints()`) |
|---|---|---|
| `Streaming` | SISD | minimal hints; falls through to inline execution |
| `ProducerFast { burst }` | SIMC | sets `k_gating = PerSlot` (KHL backing) so burst pushes pack 3 jobs per cache-line transfer |
| `WorkSteal { n_consumers, batch_size }` | MIMD | sets `k_gating = Auto`; lets the splitter pick leaves per the standard SLAW path |
| `Cooperative { n_cores }` | SIMC / MIMC | enables mailbox routing once `n_cores` >= a documented threshold |
| `VariantRace` | MISD | sets `k_gating = PerSlot`; disables burst flushing (each variant runs independently) |

API:

- [`JobPlan::with_workload_shape(shape)`](JobPlan-Reference.md#builder-methods) - consume the shape and overwrite the three knobs.

Calls to other `with_*` builders that touch the same knobs (`with_k_gating`, `with_mailbox_routing`, `with_oversubscription_log2`) should come AFTER `with_workload_shape` so they win.

## `KGating`

Defined in [`src/sched/k_gating.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/k_gating.rs).

```rust
pub enum KGating {
    Auto,         // default
    CounterOnly,  // Fcl backing
    PerSlot,      // KHL backing
}
```

Per-worker K_inner=3 deque-backing selector. Every `AdaptiveWorker` holds an `AtomicU32` tag whose value selects between the Fcl (counter-only Chase-Lev) and KHL (per-slot Vyukov) backings; the swap is one `AtomicU32::Release-store` per worker per migration. Per push/pop the worker reads the tag with one `Relaxed` load (~1 ns) before branching to the active backing.

| Variant | Backing | Wins on |
|---|---|---|
| `Auto` | per-host startup calibration picks Fcl vs KHL | default; matches the empirical winner on the host |
| `CounterOnly` | Fcl (single `bottom` counter for both ordering and publication; classic Chase-Lev family) | smaller-store-buffer cores (in-order ARM, embedded) where the counter pattern wins |
| `PerSlot` | KHL (per-slot Vyukov sequence; publication contention spread across an array of `seq` atomics) | store-buffer-rich cores (Zen+, Sapphire Rapids); measured 3.0x faster than Chase-Lev K=1 on producer-fast K=64 |

API:

- [`JobPlan::with_k_gating(KGating)`](JobPlan-Reference.md#builder-methods) - pin the choice for one dispatch.
- `migrate_all_workers_k_gating(KGating)` (on `LocalArena`) - flip every worker's tag globally at runtime.
- [`calibrate_k_gating()`](Sched-Module-Reference.md#k_gating) - run the per-host calibration probe; cached in `CALIBRATED_GATING`.

## `BisectVariant`

Defined in [`src/sched/plan.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/plan.rs).

```rust
pub enum BisectVariant {
    ProducerMaxLenWorkers,
    RayonStyleReplenish,
}
```

Per-call selector for one of two production-validated bisect-policy variants. The default behavior (field `None`) uses the continuation-steal-lazy bisect: first level always splits to seed initial fanout (`workers` eager leaves), subsequent levels run serially inline unless real steal pressure is detected on the dispatching worker's deque.

The two variants are auto-routed by [`adaptive_variant_routing`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/adaptive_variant_routing.rs) at `JobPlan::set_profile` construction time: AMD vendors with `DispatchProfile::PortBound` (the profile [`WorkloadClass::PortBound`](#workloadclass) and [`WorkloadClass::FineGrain`](#workloadclass) map to) get `ProducerMaxLenWorkers` for `batch_size >= 50_000` and `RayonStyleReplenish` for smaller batches; all other profiles + vendors get `None`. Per-call overrides via [`JobPlan::with_bisect_variant(v)`](JobPlan-Reference.md#builder-methods) win over the auto-routing.

| Variant | What it changes | Empirical win cell |
|---|---|---|
| `ProducerMaxLenWorkers` | clamp upfront leaf count to `workers * 1` (matches rayon's initial `LengthSplitter` count) | Zen3 5700G Compute/100k: +19.6% over default |
| `RayonStyleReplenish` | start with `leaves_per_worker = 1`; on observed steal, replenish to `max(workers, splits / 2)` (mirrors rayon-1.12's `Splitter::try_split`) | Zen3 5700G Compute/10k: +37.6% over default |

---
title: JobPlan Reference
weight: 1
---

The plan struct attached to every dispatch. Defined in [`src/sched/plan.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/plan.rs); re-exported as `flynnel::JobPlan`.

```rust
pub struct JobPlan {
    pub k_outer: u8,
    pub batch_size: u32,
    pub hw_class: HwClass,
    pub variant: Variant,
    pub numa_hint: Option<u32>,
    pub use_smt: bool,
    pub estimated_per_item_ns: Option<u32>,
    pub estimated_per_item_ns_explicit: bool,
    pub task_overhead_ns: Option<u32>,
    pub task_span_ns: Option<u32>,
    pub effective_task_count: Option<u32>,
    pub k_inner_log2: Option<u8>,
    pub backend_hint: Option<Backend>,
    pub oversubscription_log2: Option<u8>,
    pub worker_cap: Option<u32>,
    pub bisect_variant: Option<BisectVariant>,
    pub use_mailbox_routing: bool,
    pub leaf_shape: crate::sched::adaptive_profile::LeafShape,
    pub deque_tier_hint: Option<crate::sched::deque_tier::DequeTier>,
    pub k_gating: crate::sched::k_gating::KGating,
    pub cooperative_routing: crate::sched::adaptive_cooperative::CooperativeRouting,
    pub site: Option<crate::sched::call_site::SiteRef>,
    pub profile_explicit: bool,
}
```

`Copy`, `Clone`, `Debug`, `PartialEq`, `Eq`, `Hash`. Every field is `pub` for direct construction; the builder methods below are the ergonomic surface.

## Fields

### `k_outer: u8`

`K_outer = log2(n_limbs)` for the job's operands. The primary axis the tier picker reads. Larger means more data per operand, which justifies higher-overhead scheduling tiers.

The [tier classifier](Foundation-Types-Reference.md#schedtier):

| `k_outer` band | Base tier |
|---------------|-----------|
| `0..=4` | `Inline` |
| `5..=7` | `Local` |
| `8..=10` | `Hierarchical` |
| `11..` | `Federated` |

### `batch_size: u32`

Number of independent operations being scheduled together. The tier picker promotes `Inline`-band jobs to `Local` once `batch_size >= 256` (because aggregate parallel work dominates per-call dispatch cost), and demotes `Local`-band jobs back to `Inline` if `batch_size < 32`.

### `hw_class: HwClass`

Target hardware class. See [HwClass](Foundation-Types-Reference.md#hwclass). Defaults to `Scalar`. Matrix-extension classes (AMX, SME, tensor cores) trigger the [mode-region batching](Sched-Module-Reference.md#mode_region) path.

### `variant: Variant`

Precision contract. See [Variant](Foundation-Types-Reference.md#variant). Defaults to `Faithful`.

### `numa_hint: Option<u32>`

`None` means "current thread's NUMA node"; `Some(n)` requests placement on a specific node (e.g., keep the job on the same node as a referenced buffer).

### `use_smt: bool`

SMT-sibling activation hint. `false` (default) means this work runs on primary (physical) workers only; SMT-sibling workers stay parked. `true` activates the SMT siblings for the duration of this submit so the pool runs at full logical-thread width.

Workload classification (the source's own guidance):

| Workload | `use_smt` |
|----------|-----------|
| BigFloat schoolbook / Karatsuba (IMUL-pipe saturating) | `false` |
| NTT butterflies (FMA-saturating) | `false` |
| `Tower<T>` combine (Two-Sum FMA) | `false` |
| FP sqrt / div / transcendental chains (latency-bound) | `true` (SMT-2 fills dispatch bubbles) |
| Mixed-shape kernels | `false` unless benchmarked |

[`set_profile`](#set_profile) sets this automatically from [`DispatchProfile`](Foundation-Types-Reference.md#dispatchprofile) (true for `LatencyBound` and `MemoryBound`).

### `estimated_per_item_ns: Option<u32>`

Caller-supplied estimate of per-item cost in nanoseconds. Lets dispatchers (e.g. `collect_indexed_heartbeat`) make a static serial-vs-parallel decision at entry without rdtsc-polling.

When set, [`estimated_total_ns`](#estimated_total_ns) returns `estimated_per_item_ns * batch_size`. Dispatchers gate behavior on the predicted total:

- Total below the small-N gate (~50 us, the shared `INLINE_COLLAPSE_THRESHOLD_NS`): go fully serial, zero scheduling overhead.
- Total at or above the gate: SLAW-bisect with full parallelism.

Per-op typical values from the source (Zen3 / FpN<8> baseline):

| Op | ns/item |
|----|---------|
| `add_slice` / `sub_slice` | ~10 (SIMD-vectorised) |
| `mul_slice` | ~10 to 20 |
| `spmv` per row | ~50 ns multiplied by `nnz_per_row` |
| `gemm` per output element | ~32 ns multiplied by `k` (inner dim) |
| Karatsuba recursive sub-product | `O(N^1.585)`, not flat |

### `estimated_per_item_ns_explicit: bool`

`true` when [`estimated_per_item_ns`](#estimated_per_item_ns-optionu32) was set explicitly by the caller via [`with_estimated_per_item_ns`](#with_estimated_per_item_ns) or the probe-derived path in `for_each_chunk`. `false` when the value is a classifier default from [`new`](#new) / [`set_profile`](#set_profile) (which writes the per-profile default 12 / 50 / 600 ns).

The probe-and-decide path in `for_each_chunk` consults this flag: classifier defaults are routing hints (good enough for most workloads) but not authoritative per-item-cost truth. When the caller did not give an explicit hint AND N is small enough that the wrong default would cost real wall-clock, the probe path measures actual per-element cost and overrides.

### `task_overhead_ns: Option<u32>`

Per-task scheduler overhead in nanoseconds (Tiny-Tasks model from Acar 2013): the fixed cost incurred each time a chunk is dispatched (deque push, atomic, latch init). Used together with `estimated_per_item_ns` to compute the optimal chunk count via [`optimal_chunk_count`](#optimal_chunk_count).

Typical: ~200 to 500 ns for the Flynnel join path with the adaptive splitter.

### `task_span_ns: Option<u32>`

Per-task critical-path span (Tiny-Tasks model). The portion of a task that cannot be further parallelized (e.g., serial sub-step before SIMD fan-out). For uniform-leaf workloads (matmul, slice ops) this is 0; for heterogeneous or recursive bodies (binsplit, FFT butterfly), it represents the inner-serial cost that bounds speedup. Currently unused in `optimal_chunk_count`; reserved for span-aware extensions.

### `effective_task_count: Option<u32>`

Caller-supplied effective task count when the natural item count over-states the parallelism budget. Example: a 1 M-item slice op where SIMD packs 16 items per leaf has an effective task count of `1 000 000 / 16 = 62 500`. [`optimal_chunk_count`](#optimal_chunk_count) consults this instead of `batch_size` when present.

### `k_inner_log2: Option<u8>`

BLIS K_inner axis: SIMD-lane fanout within a single matmul leaf. `Some(log2)` requests the kernel to process `2^log2` output cells per k-iteration via slice SIMD primitives (`mul_slice` + `add_slice`) instead of scalar `mul_scalar` + `add_scalar` per cell.

Recommended values:

- `Some(3)` (M=8) matches AVX-512 / Zen 4 wide registers for FpN<8> (Fp256). Default for matmul ops on capable hosts.
- `Some(2)` (M=4) AVX2 / Zen 3 fallback.
- `None` scalar inner loop. Use when the matmul shape is too small for SIMD batching to amortize.

Gating: callers set this only when `b.cols >= 2^log2` AND the `FpN<N>` width supports SIMD slice ops (N in `{4, 8, 16, 32}`). Otherwise the kernel falls back to scalar.

### `backend_hint: Option<Backend>`

Which dispatch target to route this job to. `None` means use the CPU backend. `Some(b)` requests routing to the matching registered backend ([CUDA / ROCm / Metal / TPU / ANE / Custom](Backend-System.md)). [`pick_backend`](#pick_backend) honors the hint when a backend with that id is registered; otherwise it falls back to the always-available CPU backend.

Pairs with the SIMT and MIMT axes of the [extended Flynn taxonomy](Extended-Flynn-Taxonomy.md):

- `Some(Backend::Cuda { .. })` routes the job to a SIMT device.
- [`join_hybrid`](Sched-Module-Reference.md#join_hybrid) reads this hint to decide which half runs where in the MIMT case.

### `oversubscription_log2: Option<u8>`

Per-call leaf-count multiplier expressed as a log2. `None` resolves to the conservative default 1 (2x) via `effective_oversubscription_log2()`; the probe path in `for_each_chunk` additionally consults the observer-tuned `split_multiplier`. `Some(log2)` caps the bisect leaves at `workers * 2^log2` for this dispatch only.

| Value | Leaves per worker | Used by |
|---|---|---|
| `Some(0)` | 1 (one leaf per worker) | tight uniform-cost loops that should not over-split |
| `Some(1)` | 2 | `set_profile(PortBound)` and `set_profile(MemoryBound)` defaults |
| `Some(2)` | 4 | `set_profile(LatencyBound)` default - headroom for steal-driven rebalance on long-chain stalls |
| `Some(3)` | 8 | rarely used; matches the source `clamp(.., 3)` ceiling |

The clamp ceiling is `3` (8x oversubscription) so accidental misuse cannot blow the leaf count past the dispatcher's safe envelope.

### `worker_cap: Option<u32>`

Hard cap on the worker count for this dispatch. `None` means use every available worker; `Some(1)` forces serial execution on the calling thread; `Some(N)` runs at most N workers. Lets a per-call site request a smaller subset of the pool than the arena holds (for example, a probe path that wants 4 workers regardless of the host's 44-thread allocation).

### `bisect_variant: Option<BisectVariant>`

Selects one of two production-validated bisect-policy variants (`ProducerMaxLenWorkers` or `RayonStyleReplenish`), documented in [`BisectVariant`](Foundation-Types-Reference.md#bisectvariant). `None` runs the default continuation-steal-lazy bisect path. The field is auto-resolved by [`adaptive_variant_routing::pick_variant_for_profile`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/adaptive_variant_routing.rs) at [`set_profile`](#set_profile) construction time based on CPUID vendor + `DispatchProfile` + `batch_size`; per-call overrides via [`with_bisect_variant(v)`](#builder-methods) win over the auto-routing.

### `use_mailbox_routing: bool`

Owner-directed hand-off hint. `false` (default) means recursive splits push the right-half to the global injector or the worker's own deque; `true` means push directly to the SMT-sibling's mailbox so the sibling sees the work without going through the steal probe.

The realistic_bench measurement on Zen+ R7 2700 (2026-06-06) showed no production profile wins from blanket mailbox routing (Compute/100k regressed 7x, Heavy/100k regressed 6x because pinning to the SMT pair starves cross-CCX peers). [`set_profile`](#set_profile) leaves this `false` for every variant. Power users opt in for the SIMC fan-out shape: single producer, locality-warm consumer, parallelism-limited workload. Set via [`with_mailbox_routing(true)`](#builder-methods) or via the declarative [`with_workload_shape(WorkloadShape::Cooperative { .. })`](#builder-methods) hint at large `n_cores`.

### `leaf_shape: crate::sched::adaptive_profile::LeafShape`

Caller-supplied structural shape hint. `LeafShape::Unknown` (default) means "no explicit shape; infer from `(ns, K, N)` heuristics in the static classifier." Any other value (`PortCompute`, `LatencyCompute`, `Streaming`, `Gather`) is the strongest static signal for [`infer_class_static_with_shape`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/adaptive_profile.rs); the shape maps directly to a [`WorkloadClass`](Foundation-Types-Reference.md#workloadclass) and the ns + K + N heuristics are bypassed.

Set via [`with_leaf_shape(shape)`](#builder-methods); the builder re-classifies and re-applies the profile-derived knobs (`use_smt`, `oversubscription_log2`, `use_mailbox_routing`, `deque_tier_hint`) from the new class while preserving any explicit `estimated_per_item_ns` hint.

### `deque_tier_hint: Option<crate::sched::deque_tier::DequeTier>`

Per-tier deque the recursive-split right-half is pushed to. `None` (default) means the splitter picks the widest tier (`Public`, any peer may steal); `Some(tier)` pins to a narrower coherence neighborhood (`SmtLocal` = SMT sibling, `IntraCcx` = same CCX, `CrossCcx` = cross-CCX same socket, `Public` = any peer). Same as `use_mailbox_routing`, the realistic_bench finding showed that narrowing the tier produces SMT-pair load concentration regressions on the four classified profiles. [`set_profile`](#set_profile) leaves this `None`; opt in via [`with_deque_tier_hint(tier)`](#builder-methods) when the workload's locality structure matches the chosen tier.

### `k_gating: crate::sched::k_gating::KGating`

Per-call K_inner=3 deque-backing selector. Default = `KGating::Auto` (the scheduler picks per host between KHL per-slot Vyukov and Fcl counter-only Chase-Lev based on the startup calibration). `KGating::CounterOnly` forces the Fcl backing; `KGating::PerSlot` forces KHL. See [`KGating`](Foundation-Types-Reference.md#kgating) for the trade-offs.

[`new`](#new), [`bare`](#bare), [`set_profile`](#set_profile), and [`for_op_generic`](#for_op_generic) all default this to `Auto`; the runtime-swappable `migrate_all_workers_k_gating(KGating::*)` flips every worker's active backing without disturbing per-call plans.

### `cooperative_routing: crate::sched::adaptive_cooperative::CooperativeRouting`

Per-call routing decision for `cooperative_join_n`. Defaults to the process-global active routing (resolved via a single `AtomicU8::Acquire-load` at plan construction), which is initially `CooperativeRouting::Auto`. `Auto` lets `cooperative_join_n` pick between the tree fan-out and flat fan-out shapes based on the closure count and worker topology.

Runtime-swappable via `migrate_cooperative_routing(CooperativeRouting::*)`; the flip is a single `AtomicU8::Release-store` visible to every subsequent `JobPlan::new`.

### `site: Option<crate::sched::call_site::SiteRef>`

Per-call-site adaptive state (`CallSiteState`) this dispatch records leaf timings into and reads learned decisions from. Each generic dispatch entry (`for_each_chunk`, `collect_indexed`, and the rest of that family) is `#[track_caller]` and attaches the state resolved from the CALLER's source location (`call_site::caller_site()`) via `with_site_if_none`, so distinct call sites accumulate INDEPENDENT classifier, variance, policy-arm, and hybrid-placement history; interleaved heterogeneous workloads no longer contaminate one another through the process-global tag (which remains the cold-start prior and the fallback for site-less plans). Attach a caller-owned `static CallSiteState` via `with_site` to pin one identity across multiple entries, or to read the learned state back (see `examples/site_classifier_demo.rs`).

### `profile_explicit: bool`

`true` when the profile-derived knobs came from an explicit caller decision (`set_profile` / `for_op_generic`), `false` on the adaptive default paths (`new` / `bare`). `apply_site_class` (called by the generic dispatch entries) re-derives the routing knobs from the site's LEARNED class only when the caller pinned nothing: `caller_pinned()` is `profile_explicit || estimated_per_item_ns_explicit || leaf_shape != Unknown`.

## Constructors

### `new`

```rust
pub fn new(k_outer: u8, batch_size: u32) -> Self
```

**Adaptive default.** The static initial classifier (`infer_class_static`) picks a `WorkloadClass` from `(k_outer, batch_size)` and pre-populates `use_smt`, `estimated_per_item_ns`, `oversubscription_log2`, `use_mailbox_routing`, and `deque_tier_hint` from that class's profile, so call 1 routes correctly without any measurement. Construction also resolves the process-global cooperative-routing and bisect-variant tags (one `AtomicU8::Acquire-load` each, ~1 ns). `k_gating` defaults to `Auto`.

Refinement after call 1 is per call site: the generic dispatch entries attach a `CallSiteState` and `apply_site_class` re-derives an unpinned plan's knobs from that site's learned class. The process-global active class ([`migrate_workload_class(WorkloadClass::*)`](Sched-Module-Reference.md#adaptive_profile), or the observer's `tick_auto_classify`) drives the `AdaptiveDispatcher` surface; its startup default is `PortBound` (SMT siblings parked, 12 ns/elem estimate, 2x oversubscription).

To bypass the global and get bare defaults independent of the active class (for example a unit test that asserts specific plan fields), use [`bare`](#bare). To pin a different profile for one call without mutating the global, use [`set_profile`](#set_profile).

### `bare`

```rust
pub fn bare(k_outer: u8, batch_size: u32) -> Self
```

**Profile-independent default.** Does NOT consult the global active class. Constructs a plan with `use_smt = false`, `estimated_per_item_ns = None`, `oversubscription_log2 = None`, `use_mailbox_routing = false`, `deque_tier_hint = None`, `Scalar` hw_class, `Faithful` variant, no NUMA hint, `k_gating = Auto`.

Use this from:

- Unit tests that assert specific plan field values regardless of the test-binary's global state.
- Call sites that explicitly want the pre-adaptive defaults (no SMT, no cost estimate, no oversubscription floor).
- Bench files that need a stable baseline plan unaffected by `migrate_workload_class` flips in other tests of the same binary.

Most production call sites want [`new`](#new) (adaptive default) or [`set_profile`](#set_profile) (per-call profile pin), not `bare`.

### `set_profile`

```rust
pub fn set_profile(k_outer: u8, batch_size: u32, profile: DispatchProfile) -> Self
```

Sets `use_smt`, `estimated_per_item_ns`, and `oversubscription_log2` together from a [`DispatchProfile`](Foundation-Types-Reference.md#dispatchprofile) variant:

| Profile | `use_smt` | default `ns_per_elem` | `oversubscription_log2` (leaves per worker) |
|---------|-----------|----------------------|----------------------------------------------|
| `LatencyBound` | `true` | 600 | 2 (4x) |
| `PortBound` | `false` | 12 | 1 (2x) |
| `MemoryBound` | `true` | 50 | 1 (2x) |
| `Streaming` | `false` | 50 | 1 (2x) |
| `Unspecified` | `false` | `None` | 1 (2x) |

**This is the recommended constructor for every call site that has a known dispatch profile.** Override individual knobs after construction via the builder methods.

### `for_op_generic`

```rust
pub fn for_op_generic<O: OpClass>(k_outer: u8, batch_size: u32, op: O) -> Self
```

Generic over any [`OpClass`](Foundation-Types-Reference.md#opclass-trait) impl. Use this from domain-specific dispatchers with their own op enums (a math crate's kernel enum, a graphics crate's shader-class enum, a string crate's operation enum). Only `use_smt` is set from the trait; for the full profile defaults, have the domain enum map to a `DispatchProfile` and call `set_profile` instead.

## Builder methods

All builders take `self` by value and return `Self`, supporting chains like `JobPlan::new(8, 1024).with_smt().with_backend(...)`.

| Method | Effect |
|--------|--------|
| `with_hw_class(HwClass)` | Sets `hw_class`. |
| `with_variant(Variant)` | Sets `variant`. |
| `with_numa_hint(node: u32)` | Sets `numa_hint = Some(node)`. |
| `with_smt()` | Sets `use_smt = true`. |
| `with_cost_ns_per_elem(ns: u32)` | Sets `estimated_per_item_ns = Some(ns)`. Drives leaf-count derivation in `for_each_chunk` and the inline-collapse fast path. Canonical name in the scheduler-tuning vocabulary; `with_estimated_per_item_ns` is the legacy alias. |
| `with_estimated_per_item_ns(ns: u32)` | Same as `with_cost_ns_per_elem`. |
| `with_oversubscription_log2(log2: u8)` | Sets `oversubscription_log2 = Some(log2)`. Overrides the per-call leaf-count multiplier (`log2 = 0` means 1 leaf per worker, `log2 = 3` means 8). Clamps to `[0, 3]`. |
| `with_workers(n: u32)` | Sets `worker_cap = Some(n)`. Caps the worker count for this dispatch; `1` forces serial execution on the calling thread. |
| `with_task_overhead_ns(ns: u32)` | Sets `task_overhead_ns = Some(ns)`. |
| `with_task_span_ns(ns: u32)` | Sets `task_span_ns = Some(ns)`. |
| `with_effective_task_count(count: u32)` | Sets `effective_task_count = Some(count)`. |
| `with_k_inner_log2(log2: u8)` | Sets `k_inner_log2 = Some(log2)`. |
| `with_backend(Backend)` | Sets `backend_hint = Some(backend)`. |
| `with_k_gating(KGating)` | Sets `k_gating`. Overrides the `Auto` default so this dispatch lands on a pinned backing (`CounterOnly` = Fcl, `PerSlot` = KHL) regardless of the per-host startup calibration. |
| `with_mailbox_routing(bool)` | Sets `use_mailbox_routing`. The realistic_bench finding is that blanket mailbox routing regresses Compute / Heavy; opt in only when the call site's locality structure justifies SMT-pair concentration. |
| `with_deque_tier_hint(DequeTier)` | Sets `deque_tier_hint = Some(tier)`. Pins the recursive-split right-half to a narrower coherence neighborhood than `Public`. Same trade-off as `with_mailbox_routing`. |
| `with_workload_shape(WorkloadShape)` | Overwrites `k_gating`, `use_mailbox_routing`, and `oversubscription_log2` from a declarative shape ([`WorkloadShape`](Foundation-Types-Reference.md#workloadshape)). Call this BEFORE other `with_*` builders that touch those fields, since it overwrites them. |
| `with_site(SiteRef)` | Attaches a caller-owned per-call-site state, replacing any prior attachment. Declare `static SITE: CallSiteState = CallSiteState::new();` and pass `SiteRef::new(&SITE)`. |
| `with_site_if_none(SiteRef)` | Attaches only when no site is present; the generic dispatch entries use this so an outer attachment always wins. |
| `with_bisect_variant(BisectVariant)` | Sets `bisect_variant = Some(v)`. Selects an in-tree scheduler-policy variant for bench-driven A/B research. Production code leaves this `None`. |

## Methods

### `estimated_total_ns`

```rust
pub fn estimated_total_ns(&self) -> Option<u64>
```

Returns `estimated_per_item_ns * batch_size` saturating, or `None` if no estimate is set.

### `effective_ns_per_elem`

```rust
pub fn effective_ns_per_elem(&self) -> Option<u32>
```

Returns the per-element cost estimate this dispatch should use, accounting for the active class. When the plan's `estimated_per_item_ns` is `Some(ns)`, that value wins; otherwise falls back to the active-profile default (12 ns for PortBound, 600 ns for LatencyBound, 50 ns for MemoryBound, `None` for Unspecified). The bisect splitter consults this when choosing a leaf count.

### `effective_oversubscription_log2`

```rust
pub fn effective_oversubscription_log2(&self) -> u8
```

Returns the per-call leaf-count multiplier this dispatch should use. Same precedence as above: explicit `Some(log2)` wins; otherwise falls back to the active-profile default (2 for LatencyBound, 1 for PortBound / MemoryBound / Unspecified). Always returns a value, never `None` - the splitter always needs SOME multiplier.

### `effective_use_smt`

```rust
pub fn effective_use_smt(&self) -> bool
```

Returns whether SMT siblings should activate for this dispatch, accounting for both the plan's `use_smt` field AND the variance-driven SMT-suppression observer. When the per-leaf cv-squared sampler indicates uniform-cost work (low variance), this returns `false` even if `plan.use_smt` is `true`, because SMT-2 siblings help when they can fill stall bubbles in heterogeneous workloads but hurt when every leaf is the same shape (the siblings would contest the same execution unit). See the [load-bearing invariants](../explanation/Architecture-Overview.md) for the full mechanism.

### `pick_backend`

```rust
pub fn pick_backend(&self) -> BackendRef
```

Resolves the backend this plan should dispatch to:

1. If `backend_hint` is `Some(b)` AND a backend with id `b` is registered: returns that backend.
2. Otherwise: returns the always-available CPU backend ([`cpu_backend()`](Backend-System.md#cpu_backend)).

This is the single resolution point every routing helper goes through; consumers don't handle hint / fallback logic themselves.

### `optimal_chunk_count`

```rust
pub fn optimal_chunk_count(&self, workers: usize) -> Option<u32>
```

Tiny-Tasks model (Acar 2013):

```text
C_opt = clamp(sqrt(W * P / O), 1, N)
```

Where `W` is total estimated work in ns (`estimated_per_item_ns * batch_size`), `P` is `workers`, `O` is `task_overhead_ns`, and `N` is `effective_task_count` (falling back to `batch_size`).

Returns `None` if `estimated_per_item_ns` or `task_overhead_ns` is unset, in which case callers fall back to the SLAW splitter's budget heuristic.

Derivation: total-time = `serial_work / C + overhead * C`. `d/dC = -W/C^2 + O = 0`, giving `C = sqrt(W / O)`. Including parallelism (`P` workers running concurrently) bumps the optimal `C` by `sqrt(P)` because each worker's overhead is paid once but the work is divided P-ways.

## Example

```rust
use flynnel::{JobPlan, DispatchProfile, Backend, Variant};

let plan = JobPlan::set_profile(8, 1024, DispatchProfile::LatencyBound)
    // use_smt = true, ns_per_elem = 600, oversubscription_log2 = 2
    .with_variant(Variant::Correct)                      // bit-exact requirement
    .with_cost_ns_per_elem(2500)                          // override with measured cost
    .with_task_overhead_ns(300)                          // ~300 ns join cost
    .with_backend(Backend::Cuda { device_id: 0 });       // prefer GPU when registered

assert_eq!(plan.estimated_total_ns(), Some(2_500 * 1024));
let opt = plan.optimal_chunk_count(16).unwrap();   // 16 workers
```

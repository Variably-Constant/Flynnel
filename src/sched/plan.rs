//! Plan types: which scheduler tier, which hardware class, and the
//! `(K_outer, batch_size)` -> tier mapping used by `pick_tier`.
//!
//! Tier boundaries:
//!
//! | Tier         | K_outer band            | Why                                           |
//! |--------------|-------------------------|-----------------------------------------------|
//! | Inline       | K <= 4 (<= 16 limbs)    | Sub-microsecond per op; scheduler overhead dominates |
//! | Local        | K = 5..7 (32-128 limbs) | Single-NUMA work-steal; rayon-style           |
//! | Hierarchical | K = 8..10 (256-1024 limbs) | Per-NUMA arenas + leader-driven cross-arena |
//! | Federated    | K >= 11 (>= 2048 limbs) | Multi-pool + tiered storage + constant replication |
//!
//! These defaults can be overridden by a per-host calibration table
//! populated at install time.

use crate::backend::{Backend, BackendRef, cpu_backend};
use crate::dispatch_profile::DispatchProfile;
use crate::foundation::Variant;
use crate::numa_topology::NumaTopology;

/// Per-call selector for one of the in-tree bisect-policy variants.
/// The default behavior (no variant set) uses the
/// continuation-steal-lazy bisect: first level always splits to seed
/// initial fanout (`workers` leaves), subsequent levels run serially
/// inline unless real steal pressure is detected on the dispatching
/// worker's deque.
///
/// Variants exist for the workloads where the default is measurably
/// beaten by a different policy. Routing is automatic when the
/// process-global [`crate::sched::adaptive_variant_routing`] tag is
/// set (which happens at startup via the CPUID-based default table);
/// callers can also pin a variant per-plan via
/// [`JobPlan::with_bisect_variant`].
///
/// Empirical justification (Xeon Cascade Lake 12T / EPYC 9B14 44T /
/// Zen3 5700G 16T, criterion 0.8.2 medians, see
/// `benches/inline_collapse.rs`):
///
/// - `ProducerMaxLenWorkers`: +19.6% Zen3 Compute/100k vs default;
///   tied on Genoa; small +5.7% Xeon Heavy/10k win.
/// - `RayonStyleReplenish`: +37.6% Zen3 Compute/10k vs default;
///   tied on Genoa Compute (both sizes).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BisectVariant {
    /// **Producer max-len-equals-workers**: clamp the upfront leaf
    /// count to `workers * 1` (matching rayon's `LengthSplitter`
    /// default initial split count = `current_num_threads()`).
    /// Replenish target stays at the default `workers * 1` so the
    /// bisect tree is shallower than the production baseline.
    ///
    /// Pareto cell: AMD Compute at large N (>= 50_000). Routed
    /// automatically when the CPUID-resolved variant routing is
    /// `ComputeBatchAdaptive` and `batch_size >= 50_000`.
    ProducerMaxLenWorkers,
    /// **Rayon-style replenish**: start with `leaves_per_worker = 1`
    /// AND, on each observed steal in `bisect`, replenish to
    /// `MAX(workers, splits / 2)` instead of the default
    /// `max_budget`. Mirrors `rayon-1.12.0`'s `Splitter::try_split`
    /// formula (`splits = max(thread_count, splits / 2)`).
    ///
    /// Pareto cell: AMD Compute at small N (< 50_000). Routed
    /// automatically when the CPUID-resolved variant routing is
    /// `ComputeBatchAdaptive` and `batch_size < 50_000`.
    RayonStyleReplenish,
}

// `SchedTier` and `HwClass` live in [`crate::foundation`] so the same
// types travel across an entire dispatch stack: scheduler tier
// classification on the parent side, kernel-variant selection on the
// child side, no namespace churn between them.
pub use crate::foundation::{HwClass, SchedTier};

/// A scheduling plan attached to one job submission. The four-tuple
/// `(k_outer, batch_size, hw_class, variant)` together with the cached
/// NUMA topology produces a [`SchedTier`] via [`pick_tier`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct JobPlan {
    /// `K_outer = log2(n_limbs)` for the job's operands.
    pub k_outer: u8,
    /// Batch size: number of independent operations being scheduled
    /// together. Important for amortizing tier-dispatch overhead.
    pub batch_size: u32,
    /// Target hardware class.
    pub hw_class: HwClass,
    /// Quality tier.
    pub variant: Variant,
    /// Optional NUMA node hint. `None` means "current thread's node;"
    /// `Some(n)` requests placement on a specific node (e.g., to keep
    /// the job on the same node as a referenced buffer).
    pub numa_hint: Option<u32>,
    /// SMT-sibling activation hint. `false` (default) means this work
    /// runs on the primary (physical) workers only; the 8 SMT
    /// siblings stay parked. `true` activates the SMT siblings for
    /// the duration of this submit so the pool runs at full
    /// `logical_threads` width.
    ///
    /// Workload guide:
    /// - BigFloat schoolbook / Karatsuba (IMUL pipe saturating): `false`
    /// - NTT butterflies (FMA-saturating): `false`
    /// - `Tower<T>` combine (Two-Sum FMA): `false`
    /// - FP sqrt / div / transcendental chains (latency-bound):
    ///   `true` (SMT-2 fills dispatch bubbles)
    /// - Mixed-shape kernels: prefer `false` unless benchmarked
    pub use_smt: bool,
    /// Optional caller-supplied estimate of the per-item cost in
    /// nanoseconds. Lets dispatchers (e.g. `collect_indexed_heartbeat`)
    /// make a static serial-vs-parallel decision at entry without
    /// rdtsc-polling the hot loop. `None` means "no estimate;
    /// dispatcher falls through to its default heuristic."
    ///
    /// When set, dispatchers compute
    /// `estimated_per_item_ns * batch_size` to predict the total
    /// compute cost and gate dispatch behavior accordingly:
    ///
    /// - Total below the small-N gate (~50µs, the shared
    ///   INLINE_COLLAPSE_THRESHOLD_NS): go fully serial, zero
    ///   scheduling overhead.
    /// - Total at or above the gate: SLAW-bisect with full
    ///   parallelism.
    ///
    /// Per-op typical values (Zen3 / FpN<8> baseline):
    /// - `add_slice` / `sub_slice`: ~10ns/elem (SIMD-vectorised)
    /// - `mul_slice`: ~10-20ns/elem
    /// - `spmv` per row: ~50ns * nnz_per_row
    /// - `gemm` per output element: ~32ns * k (k = inner dim)
    /// - `Karatsuba` recursive sub-product: O(N^1.585), not flat
    pub estimated_per_item_ns: Option<u32>,
    /// `true` when [`Self::estimated_per_item_ns`] was set explicitly
    /// by the caller via [`Self::with_estimated_per_item_ns`]. `false`
    /// when the value is a classifier default from
    /// [`Self::new`] / [`Self::set_profile`] (which writes the
    /// per-profile default 12 / 50 / 600 ns).
    ///
    /// The probe-and-decide path in `for_each_chunk` consults this
    /// flag: classifier defaults are routing hints (good enough for
    /// most workloads) but not authoritative per-item-cost truth.
    /// When the caller did not give an explicit hint AND N is small
    /// enough that the wrong default would cost real wall-clock,
    /// the probe path measures actual per-element cost and overrides.
    pub estimated_per_item_ns_explicit: bool,
    /// Per-task scheduler overhead in nanoseconds (Tiny-Tasks model
    /// from Acar 2013): the fixed cost incurred each time a chunk is
    /// dispatched (deque push, atomic, latch init). Used together
    /// with `estimated_per_item_ns` to compute the optimal chunk
    /// count via `optimal_chunk_count`. Typical: ~200-500ns for the
    /// Flynnel join path with the adaptive splitter. Set
    /// to `None` to let helpers fall back to the SLAW default.
    pub task_overhead_ns: Option<u32>,
    /// Per-task critical-path span in nanoseconds (Tiny-Tasks model).
    /// The portion of a task that cannot be further parallelized
    /// (e.g., serial sub-step before SIMD fan-out). For uniform-leaf
    /// workloads (matmul, slice ops) this is 0; for heterogeneous
    /// or recursive bodies (binsplit, FFT butterfly), it represents
    /// the inner-serial cost that bounds speedup. Set to `None` to
    /// assume zero span.
    pub task_span_ns: Option<u32>,
    /// Caller-supplied effective task count when the natural item
    /// count over-states the parallelism budget. Example: a 1M-item
    /// slice op where SIMD packs 16 items per leaf has an effective
    /// task count of `1M / 16 = 62.5k`. `optimal_chunk_count`
    /// consults this instead of `batch_size` when present.
    pub effective_task_count: Option<u32>,
    /// BLIS K_inner axis: SIMD-lane fanout within
    /// a single matmul leaf. `Some(log2)` requests the kernel to
    /// process `2^log2` output cells per k-iteration via slice
    /// SIMD primitives (`mul_slice` + `add_slice`) instead of one
    /// scalar `mul_scalar` + `add_scalar` per cell.
    ///
    /// Recommended values:
    /// - `Some(3)` (M=8): matches AVX-512 / Zen 4 wide registers
    ///   for FpN<8> (Fp256). Default for matmul ops on capable hosts.
    /// - `Some(2)` (M=4): AVX2 / Zen 3 fallback.
    /// - `None`: scalar inner loop (current behavior). Use when
    ///   the matmul shape is too small for SIMD batching to amortize.
    ///
    /// Gating: callers should set this only when `b.cols >= 2^log2`
    /// AND the `FpN<N>` width supports SIMD slice ops (N in
    /// {4, 8, 16, 32}). Otherwise the kernel falls back to scalar.
    pub k_inner_log2: Option<u8>,
    /// Backend hint: which dispatch target to route this job to.
    /// `None` means "use the CPU backend"; `Some(b)` requests
    /// routing to the matching registered backend (CUDA / ROCm /
    /// Metal / TPU / ANE / Custom). [`Self::pick_backend`] honors
    /// the hint when a backend with that id is registered, else
    /// falls back to the always-available CPU backend.
    ///
    /// Pairs with the extended-Flynn-taxonomy SIMT and MIMT axes:
    /// `Some(Backend::Cuda { .. })` etc. routes the job to a SIMT
    /// device; [`crate::sched::hybrid::join_hybrid`] reads this
    /// hint to decide which half runs where in the MIMT case.
    pub backend_hint: Option<Backend>,
    /// Direct override of the per-dispatch leaf-count oversubscription
    /// factor as `log2(leaves_per_worker)`. `None` falls back to the
    /// `DispatchProfile`'s default (or the global split-multiplier
    /// when no profile is set).
    ///
    /// Concrete effect: `for_each_chunk`'s bisect targets `workers *
    /// 2^oversubscription_log2` leaves. Set to a small value (0 or 1)
    /// for uniform-cost ops to reduce per-leaf dispatch overhead;
    /// set higher (2 or 3) for variable-cost ops to give the work-
    /// stealing pool more leaves to rebalance across.
    ///
    /// Clamps to `[0, 3]` internally (1..8 leaves per worker; a
    /// host with 16 workers gets up to 128 leaves at log2=3).
    pub oversubscription_log2: Option<u8>,
    /// Direct override of the worker count used for this dispatch.
    /// `None` uses the full per-arena worker count (16 logical
    /// threads on a typical Zen3 host).
    ///
    /// Concrete effect: caps the `max_budget` passed to the bisect
    /// at `workers * oversubscription`. Set to `1` to force serial
    /// execution on the calling thread; set to `physical_cores` for
    /// IMUL-bound work to skip the SMT siblings even when
    /// `use_smt = true`.
    pub worker_cap: Option<u32>,
    /// Select an alternate `for_each_chunk` policy variant for
    /// bench A/B comparison. `None` (default) runs the production
    /// probe-and-decide baseline; `Some(v)` routes through the
    /// matching variant in [`BisectVariant`].
    ///
    /// Only `for_each_chunk` reads this; other parallel primitives
    /// (`for_each_chunk_triple`, `for_each_fixed_chunk`) ignore it.
    pub bisect_variant: Option<BisectVariant>,
    /// SIMC/MIMC mailbox routing hint. When `true`, the
    /// `sched::arena::join` right-half-push attempts to route the
    /// right-half job to the SMT sibling worker's mailbox before
    /// falling back to a regular tiered deque push. When `false`
    /// (default), the right-half always goes to the regular deque
    /// so the broad pool can steal.
    ///
    /// Workload guide (Zen+ R7 2700 realistic_bench, 2026-06-06):
    /// - Latency-bound chains (sqrt, div, transcendentals): `false`.
    ///   These benefit from broad work-stealing across all primaries;
    ///   pinning the right-half to one SMT pair concentrates load
    ///   and starves the other 6 physical cores.
    /// - IMUL-saturated compute (Karatsuba, schoolbook mul,
    ///   port-saturating Compute): `false`. Same reason; broad
    ///   distribution wins.
    /// - Fine-grain locality-friendly ops (Light: tight inner
    ///   loops with shared SIMD registers, small per-element work):
    ///   `true`. The SMT-sibling pin keeps the right-half's
    ///   captured state in the L1d the pair shares.
    /// - Mixed / unknown: `false`.
    ///
    /// The mailbox path is FURTHER gated at the call site on the
    /// sibling's mailbox and SmtLocal deque both being empty - so
    /// even with `true`, a backlogged sibling never gets routed to.
    pub use_mailbox_routing: bool,
    /// Caller's leaf-shape hint, set via [`Self::with_leaf_shape`].
    /// Read by the static initial classifier
    /// (`infer_class_static_with_shape`) to skip the (ns + K + N)
    /// heuristics when the caller knows the leaf-level character
    /// of the work (compute-bound, gather-bound, streaming, etc.).
    /// Default `Unknown` lets the classifier fall back to
    /// heuristics.
    pub leaf_shape: crate::sched::adaptive_profile::LeafShape,
    /// Per-call deque-tier override for the right-half push in
    /// `sched::arena::join`. `None` (default) routes the right-half
    /// to the default tier
    /// ([`crate::sched::deque_tier::DequeTier::Public`] - any peer
    /// reachable; broad steal). `Some(tier)` pins the push to the
    /// caller-specified tier.
    ///
    /// Workload guide (paired with the steal discipline in
    /// [`crate::sched::deque_tier::thief_may_steal`]):
    ///
    /// - `Some(DequeTier::SmtLocal)`: pin to SMT-sibling local
    ///   deque. Only the SMT sibling can steal. Use when the
    ///   right-half captured state is L1d-warm AND the sibling is
    ///   expected to be the consumer (recursive splits handing off
    ///   to the same physical core).
    /// - `Some(DequeTier::IntraCcx)`: pin to intra-CCX. Cluster
    ///   peers (same Zen CCX / Intel module) can steal. Use for
    ///   moderate-locality work that should stay within one cache
    ///   cluster but tolerate non-sibling-CCX-peer pickup.
    /// - `Some(DequeTier::CrossCcx)`: pin to cross-CCX. Most peers
    ///   reachable; reserves the Public deque for cross-NUMA work.
    /// - `Some(DequeTier::Public)`: explicit default. Broad steal.
    /// - `None`: broad steal (same as Public). Recommended for
    ///   unhinted ops.
    ///
    /// This is the call-site half of the unified dispatcher; the
    /// cross-process side is the `WorkloadShape` type in
    /// `crate::backend::shared_mem::variant_dispatch` (gated on
    /// the `shared-memory-worker-reference` feature).
    pub deque_tier_hint: Option<crate::sched::deque_tier::DequeTier>,
    /// Per-call K_gating axis hint. See
    /// [`crate::sched::k_gating::KGating`] for the publication-
    /// signal axis: counter-only (Chase-Lev / Fcl - one shared
    /// bottom atomic) vs per-slot (KHL / KHPD - distributed slot
    /// seq atomics). `Auto` (the default) resolves to the host's
    /// calibrated winner via
    /// [`crate::sched::k_gating::calibrate_k_gating`] - PerSlot on
    /// store-buffer-rich cores (Zen+, Sapphire Rapids+),
    /// CounterOnly on smaller-store-buffer cores (in-order ARM,
    /// embedded).
    ///
    /// The current `WorkerCtx` is KHL-backed (PerSlot) per the
    /// Zen+ measurement; this hint is consumed by the
    /// cooperative-dispatch family that retains the option to
    /// route through alternative substrates on other host classes.
    pub k_gating: crate::sched::k_gating::KGating,
    /// Per-call cooperative-routing override consumed by
    /// [`crate::sched::cooperative::cooperative_join_n`]. `Auto`
    /// (the default) defers to the process-global tag via
    /// [`crate::sched::adaptive_cooperative::active_cooperative_routing`];
    /// when that is also `Auto`, the call falls through to the
    /// population heuristic (`N < n_workers` -> tree, `N >= n_workers`
    /// -> mailbox). Non-`Auto` values pin the dispatch shape for
    /// this plan only.
    ///
    /// Workload guide:
    /// - `ForceTree`: short closures (sub-100us) where the
    ///   tree's amortized setup wins over flat fan-out's
    ///   per-StackJob cost.
    /// - `ForceMailbox`: uniform-cost closures sized to the
    ///   worker pool (N >= n_workers) where each closure should
    ///   land on a specific peer's mailbox.
    /// - `ForceDeque`: heterogeneous-cost closures where broad
    ///   random peer-steal load balance is preferred over
    ///   owner-directed mailbox routing.
    pub cooperative_routing: crate::sched::adaptive_cooperative::CooperativeRouting,
    /// Per-call-site adaptive state
    /// ([`crate::sched::call_site::CallSiteState`]) this dispatch
    /// records into and reads learned decisions from. Generic
    /// dispatch entries attach the state resolved from the CALLER's
    /// source location (`track_caller` chain +
    /// [`crate::sched::call_site::caller_site`]) via
    /// [`Self::with_site_if_none`]; callers can pin an explicit
    /// identity via [`Self::with_site`]. `None` means the dispatch
    /// only feeds the process-global observer.
    pub site: Option<crate::sched::call_site::SiteRef>,
    /// `true` when the profile-derived knobs came from an explicit
    /// caller decision ([`Self::set_profile`] / [`Self::for_op_generic`])
    /// rather than the adaptive default path ([`Self::new`] /
    /// [`Self::bare`]). The per-site class override in
    /// [`Self::apply_site_class`] respects explicit callers and
    /// never overrides them.
    pub profile_explicit: bool,
}

impl JobPlan {
    /// Builder: install a `deque_tier_hint` and return the modified
    /// plan. Use to pin the right-half of an in-worker `join` to a
    /// specific coherence tier; see [`Self::deque_tier_hint`] for
    /// the workload guide.
    pub fn with_deque_tier_hint(
        mut self,
        tier: crate::sched::deque_tier::DequeTier,
    ) -> Self {
        self.deque_tier_hint = Some(tier);
        self
    }

    /// Builder: opt this plan into SIMC/MIMC mailbox routing for
    /// the right-half of `join`. See [`Self::use_mailbox_routing`]
    /// for the workload guide.
    pub fn with_mailbox_routing(mut self, enable: bool) -> Self {
        self.use_mailbox_routing = enable;
        self
    }

    /// Builder: override the K_gating axis for this dispatch.
    /// See [`Self::k_gating`] for the trade-offs. The default
    /// (`KGating::Auto`) resolves to the host's calibrated winner.
    pub fn with_k_gating(mut self, gating: crate::sched::k_gating::KGating) -> Self {
        self.k_gating = gating;
        self
    }

    /// Builder: override the cooperative-routing axis for this
    /// dispatch. See [`Self::cooperative_routing`] for the
    /// workload guide. The default ([`crate::sched::adaptive_cooperative::CooperativeRouting::Auto`])
    /// defers to the process-global tag (set via
    /// [`crate::sched::adaptive_cooperative::migrate_cooperative_routing`])
    /// and, when that is also `Auto`, to the population heuristic.
    pub fn with_cooperative_routing(
        mut self,
        routing: crate::sched::adaptive_cooperative::CooperativeRouting,
    ) -> Self {
        self.cooperative_routing = routing;
        self
    }

    /// Builder: install a declarative [`crate::sched::workload_shape::WorkloadShape`]
    /// hint. Resolves to a bundle of low-level K-axis hints
    /// ([`Self::k_gating`], [`Self::use_mailbox_routing`],
    /// [`Self::oversubscription_log2`]) via
    /// [`crate::sched::workload_shape::WorkloadShape::hints`]. The
    /// shape API is the convenience surface for callers that
    /// describe their workload in Flynn-axis terms instead of
    /// learning every K-axis knob individually.
    ///
    /// The `use_burst` hint inside
    /// [`crate::sched::workload_shape::WorkloadShapeHints`] is
    /// informational at the plan level: cooperative fan-out sites
    /// (cooperative_join_n_flat) always run in burst mode;
    /// single-push sites (`sched::join`) always run in auto-flush
    /// mode. The shape hint helps the user pick which user-facing
    /// primitive to call, not which internal flag the plan carries.
    ///
    /// Power users that have already set specific knobs via other
    /// `with_*` builders should call this BEFORE those builders -
    /// this method overwrites k_gating / use_mailbox_routing /
    /// oversubscription_log2 with shape-derived values.
    pub fn with_workload_shape(
        mut self,
        shape: crate::sched::workload_shape::WorkloadShape,
    ) -> Self {
        let h = shape.hints();
        self.k_gating = h.k_gating;
        self.use_mailbox_routing = h.use_mailbox_routing;
        if let Some(over) = h.oversubscription_log2 {
            self.oversubscription_log2 = Some(over);
        }
        self
    }

    /// Construct an **adaptive default** plan. The static initial
    /// classifier ([`crate::sched::adaptive_profile::infer_class_static`])
    /// picks a [`crate::sched::adaptive_profile::WorkloadClass`]
    /// from `(k_outer, batch_size)` and pre-populates `use_smt`,
    /// `estimated_per_item_ns`, `oversubscription_log2`,
    /// `use_mailbox_routing`, and `deque_tier_hint` from that
    /// class's profile. Construction also resolves the process-
    /// global cooperative-routing and bisect-variant tags (one
    /// Acquire-load each, ~1 ns). The `k_gating` axis stays
    /// `Auto`, so the K_inner=3 deque-backing choice (KHL vs Fcl)
    /// remains runtime-swappable via `AdaptiveWorker`'s AtomicU32
    /// tag.
    ///
    /// Refinement after call 1 is per call site: the generic
    /// dispatch entries attach a [`crate::sched::call_site::CallSiteState`]
    /// and [`Self::apply_site_class`] re-derives an unpinned
    /// plan's knobs from that site's learned class. The process-
    /// global active class
    /// ([`crate::sched::adaptive_profile::migrate_workload_class`]
    /// / the observer's `tick_auto_classify`) drives the
    /// [`crate::sched::dispatch::AdaptiveDispatcher`] surface.
    ///
    /// To pin the profile for a single call, use
    /// [`Self::set_profile`] or [`Self::with_workload_shape`]
    /// instead. To get bare defaults (no `use_smt`, no cost
    /// estimate, no oversubscription), use [`Self::bare`] or
    /// `JobPlan::set_profile(K, batch, DispatchProfile::Unspecified)`.
    pub fn new(k_outer: u8, batch_size: u32) -> Self {
        // Static initial classifier: pick a WorkloadClass from
        // (k_outer, batch_size) at construction time so the FIRST
        // dispatch already uses the right routing. The observer
        // refines via atomic migration if the classifier guessed
        // wrong; static gets us call-1 correctness, observer
        // handles call-N correction.
        //
        // The static guess takes precedence over the active global
        // profile because the caller's JobPlan parameters carry
        // workload-specific information that the global atomic
        // (which reflects the LAST workload's observation) does
        // not. Cross-workload contamination of the global was the
        // main failure mode of the observer-only architecture for
        // workloads that ran one-after-another with different
        // shapes (e.g., criterion bench cells).
        let class = crate::sched::adaptive_profile::infer_class_static(
            k_outer, batch_size, None,
        );
        let mut plan =
            Self::set_profile_with(k_outer, batch_size, class.to_dispatch_profile(), false);
        plan.leaf_shape = crate::sched::adaptive_profile::LeafShape::Unknown;
        plan
    }

    /// Builder: tag the plan with the caller's leaf-shape hint.
    /// The shape is the strongest static signal: when set it maps
    /// directly to a [`crate::sched::adaptive_profile::WorkloadClass`]
    /// and the (ns, K, N) heuristics are bypassed.
    ///
    /// Use this when you know your workload's leaf character at
    /// compile time:
    /// - `Streaming` for byte scans, image kernels, sequential reduce
    /// - `Gather` for sparse matvec, graph adjacency, hash probes
    /// - `PortCompute` for FMA-bound, IMUL-bound, integer-pipeline work
    /// - `LatencyCompute` for sqrt chains, Newton iterations, branchy FP
    ///
    /// The static classifier picks the right [`DispatchProfile`]
    /// from the shape on call 1; the observer remains active as
    /// a safety net if the workload changes shape mid-run.
    pub fn with_leaf_shape(
        mut self,
        shape: crate::sched::adaptive_profile::LeafShape,
    ) -> Self {
        self.leaf_shape = shape;
        // Re-classify using the shape as the strongest signal.
        let class = crate::sched::adaptive_profile::infer_class_static_with_shape(
            self.k_outer,
            self.batch_size,
            self.estimated_per_item_ns,
            shape,
        );
        let profile = class.to_dispatch_profile();
        self.use_smt = profile.is_latency_bound();
        self.oversubscription_log2 = Some(profile.default_oversubscription_log2());
        self.use_mailbox_routing = profile.use_mailbox_routing();
        self.deque_tier_hint = profile.deque_tier_hint();
        // estimated_per_item_ns updated only if shape-derived
        // profile has a default and caller hadn't pinned one.
        if self.estimated_per_item_ns.is_none() {
            self.estimated_per_item_ns = profile.default_ns_per_elem();
        }
        self
    }

    /// Construct a bare default plan (no profile consultation):
    /// scalar hw_class, faithful variant, no NUMA hint, no SMT
    /// activation, no cost estimate, no oversubscription, K_gating
    /// = Auto. Used by call sites that explicitly want
    /// profile-independent defaults (most callers should use
    /// [`Self::new`] which reads the active dispatch profile).
    pub fn bare(k_outer: u8, batch_size: u32) -> Self {
        Self {
            k_outer,
            batch_size,
            hw_class: HwClass::Scalar,
            variant: Variant::Faithful,
            numa_hint: None,
            use_smt: false,
            estimated_per_item_ns: None,
            estimated_per_item_ns_explicit: false,
            task_overhead_ns: None,
            task_span_ns: None,
            effective_task_count: None,
            k_inner_log2: None,
            backend_hint: None,
            oversubscription_log2: None,
            worker_cap: None,
            bisect_variant: None,
            use_mailbox_routing: false,
            leaf_shape: crate::sched::adaptive_profile::LeafShape::Unknown,
            deque_tier_hint: None,
            k_gating: crate::sched::k_gating::KGating::Auto,
            cooperative_routing: crate::sched::adaptive_cooperative::CooperativeRouting::Auto,
            site: None,
            profile_explicit: false,
        }
    }

    /// Construct a plan from a [`DispatchProfile`]. The profile sets
    /// SMT activation, per-element cost estimate, and oversubscription
    /// together so the scheduler has the inputs it needs for per-call
    /// tuning. Use this at any call site where the work fits a known
    /// profile; the SMT routing + leaf-count cap are derived
    /// automatically.
    ///
    /// Override any specific knob via the `with_*` builders after
    /// construction (e.g. `set_profile(K, batch, LatencyBound)
    /// .with_cost_ns_per_elem(measured_ns)`).
    pub fn set_profile(k_outer: u8, batch_size: u32, profile: DispatchProfile) -> Self {
        Self::set_profile_with(k_outer, batch_size, profile, true)
    }

    /// Shared constructor behind [`Self::set_profile`] (explicit =
    /// true) and [`Self::new`] (explicit = false). The flag lands in
    /// [`Self::profile_explicit`] so the per-site class override
    /// knows whether the caller expressed an opinion about the
    /// profile or accepted the adaptive default.
    pub(crate) fn set_profile_with(
        k_outer: u8,
        batch_size: u32,
        profile: DispatchProfile,
        explicit: bool,
    ) -> Self {
        Self {
            k_outer,
            batch_size,
            hw_class: HwClass::Scalar,
            variant: Variant::Faithful,
            numa_hint: None,
            use_smt: profile.is_latency_bound(),
            estimated_per_item_ns: profile.default_ns_per_elem(),
            estimated_per_item_ns_explicit: false,
            task_overhead_ns: None,
            task_span_ns: None,
            effective_task_count: None,
            k_inner_log2: None,
            backend_hint: None,
            oversubscription_log2: Some(profile.default_oversubscription_log2()),
            worker_cap: None,
            // Resolve the bisect-variant routing from the process-global
            // adaptive tag (initial value: CPUID-resolved per vendor).
            // AMD + PortBound + batch >= 50_000 -> ProducerMaxLenWorkers
            // AMD + PortBound + batch <  50_000 -> RayonStyleReplenish
            // Otherwise None (default lazy-steal bisect). The flip is
            // free on the per-op path; only one AtomicU8 Acquire-load
            // here at construction time.
            bisect_variant:
                crate::sched::adaptive_variant_routing::pick_variant_for_profile(
                    profile, batch_size,
                ),
            // Auto-routing from profile classification: the scheduler
            // picks the locality knobs for the caller. Power users can
            // still override via the with_* builders.
            use_mailbox_routing: profile.use_mailbox_routing(),
            leaf_shape: crate::sched::adaptive_profile::LeafShape::Unknown,
            deque_tier_hint: profile.deque_tier_hint(),
            k_gating: crate::sched::k_gating::KGating::Auto,
            // Resolve from the process-global active tag at construction
            // time (one AtomicU8 Acquire-load, ~1 ns). When the global
            // is also Auto (the default), the call falls through to the
            // population heuristic inside cooperative_join_n.
            cooperative_routing:
                crate::sched::adaptive_cooperative::active_cooperative_routing(),
            site: None,
            profile_explicit: explicit,
        }
    }

    /// Generic constructor over any [`crate::op_class::OpClass`].
    /// Accepts [`DispatchProfile`] as well as any domain-specific
    /// op enum a downstream crate defines (string substrates, GPU
    /// dispatch, signal-processing kernels, ...). The classification
    /// (`is_latency_bound`) drives `use_smt`; downstream enums that
    /// map to a `DispatchProfile` get the full cost / oversubscription
    /// defaults by wrapping `set_profile`.
    pub fn for_op_generic<O>(k_outer: u8, batch_size: u32, op: O) -> Self
    where
        O: crate::op_class::OpClass,
    {
        Self {
            k_outer,
            batch_size,
            hw_class: HwClass::Scalar,
            variant: Variant::Faithful,
            numa_hint: None,
            use_smt: op.is_latency_bound(),
            estimated_per_item_ns: None,
            estimated_per_item_ns_explicit: false,
            task_overhead_ns: None,
            task_span_ns: None,
            effective_task_count: None,
            k_inner_log2: None,
            backend_hint: None,
            oversubscription_log2: None,
            worker_cap: None,
            bisect_variant: None,
            use_mailbox_routing: false,
            leaf_shape: crate::sched::adaptive_profile::LeafShape::Unknown,
            deque_tier_hint: None,
            k_gating: crate::sched::k_gating::KGating::Auto,
            cooperative_routing:
                crate::sched::adaptive_cooperative::active_cooperative_routing(),
            site: None,
            // The op classification is a caller-supplied signal, so
            // the per-site override must not second-guess it.
            profile_explicit: true,
        }
    }

    /// Builder: attach a per-call-site adaptive state, replacing any
    /// prior attachment. Power-user surface: declare a
    /// `static SITE: CallSiteState = CallSiteState::new();` and
    /// attach it to every plan that should share one learned
    /// identity. Most callers rely on the generic dispatch entries
    /// attaching the caller-location-resolved state
    /// ([`crate::sched::call_site::caller_site`]) via
    /// [`Self::with_site_if_none`] instead.
    pub fn with_site(mut self, site: crate::sched::call_site::SiteRef) -> Self {
        self.site = Some(site);
        self
    }

    /// Attach `site` only when no site is present yet. Used by the
    /// generic dispatch entries so an outer attachment (a caller's
    /// explicit [`Self::with_site`], or an outer entry like
    /// `collect_indexed_heartbeat` delegating to `collect_indexed`)
    /// always wins over the inner entry's own static.
    pub fn with_site_if_none(mut self, site: crate::sched::call_site::SiteRef) -> Self {
        if self.site.is_none() {
            self.site = Some(site);
        }
        self
    }

    /// True when the caller pinned authoritative tuning on this plan:
    /// an explicit profile ([`Self::set_profile`] /
    /// [`Self::for_op_generic`]), an explicit per-item cost hint, or
    /// a leaf-shape declaration. The per-site class override defers
    /// to pinned callers.
    #[inline]
    pub fn caller_pinned(&self) -> bool {
        self.profile_explicit
            || self.estimated_per_item_ns_explicit
            || self.leaf_shape != crate::sched::adaptive_profile::LeafShape::Unknown
    }

    /// Re-derive the profile-driven knobs from this plan's site's
    /// LEARNED class, when (a) the caller pinned nothing and (b) the
    /// site has classified itself. Returns the plan unchanged
    /// otherwise. Called at the top of the generic dispatch entries
    /// after [`Self::with_site_if_none`], so repeat dispatches from
    /// the same call site run with that site's own learned routing
    /// instead of the process-global prior.
    pub fn apply_site_class(mut self) -> Self {
        if self.caller_pinned() {
            return self;
        }
        let Some(site) = self.site else { return self };
        let Some(class) = site.get().learned_class() else {
            return self;
        };
        let profile = class.to_dispatch_profile();
        self.use_smt = profile.is_latency_bound();
        self.oversubscription_log2 = Some(profile.default_oversubscription_log2());
        self.use_mailbox_routing = profile.use_mailbox_routing();
        self.deque_tier_hint = profile.deque_tier_hint();
        // Keep any existing classifier-default estimate unless the
        // learned profile carries one; the caller gave no explicit
        // hint (caller_pinned checked above).
        if let Some(ns) = profile.default_ns_per_elem() {
            self.estimated_per_item_ns = Some(ns);
        }
        self
    }

    /// Builder: set the hardware class.
    pub fn with_hw_class(mut self, hw: HwClass) -> Self {
        self.hw_class = hw;
        self
    }

    /// Builder: set the variant.
    pub fn with_variant(mut self, v: Variant) -> Self {
        self.variant = v;
        self
    }

    /// Builder: set the NUMA hint.
    pub fn with_numa_hint(mut self, node: u32) -> Self {
        self.numa_hint = Some(node);
        self
    }

    /// Builder: activate the SMT sibling workers for this submit.
    /// See [`Self::use_smt`] for the workload-classification guide.
    pub fn with_smt(mut self) -> Self {
        self.use_smt = true;
        self
    }

    /// Builder: tag the plan with a caller-supplied estimate of the
    /// per-item cost in nanoseconds. Dispatchers consult this via
    /// [`Self::estimated_total_ns`] to make static serial-vs-parallel
    /// decisions at entry, bypassing rdtsc-polling in the hot loop.
    ///
    /// Also re-runs the static initial classifier
    /// ([`crate::sched::adaptive_profile::infer_class_static`]) with
    /// the new hint and updates the plan's `use_smt` /
    /// `oversubscription_log2` / `bisect_variant` based on the
    /// new classification. The caller's hint is authoritative -- it
    /// overrides the (k_outer, batch_size)-only guess that
    /// [`Self::new`] made before the hint was available.
    pub fn with_estimated_per_item_ns(mut self, ns_per_item: u32) -> Self {
        self.estimated_per_item_ns = Some(ns_per_item);
        self.estimated_per_item_ns_explicit = true;
        let class = crate::sched::adaptive_profile::infer_class_static_with_shape(
            self.k_outer,
            self.batch_size,
            Some(ns_per_item),
            self.leaf_shape,
        );
        let profile = class.to_dispatch_profile();
        // Re-apply the profile-derived knobs but preserve the new
        // ns hint (set_profile would overwrite estimated_per_item_ns
        // with the profile default).
        self.use_smt = profile.is_latency_bound();
        self.oversubscription_log2 = Some(profile.default_oversubscription_log2());
        self.use_mailbox_routing = profile.use_mailbox_routing();
        self.deque_tier_hint = profile.deque_tier_hint();
        self
    }

    /// Compute the predicted total compute cost in nanoseconds from
    /// `estimated_per_item_ns * batch_size`. Returns `None` if no
    /// per-item estimate has been set. Used by
    /// `collect_indexed_heartbeat` and similar dispatchers to gate
    /// serial-vs-parallel at entry without rdtsc-polling.
    #[inline]
    pub fn estimated_total_ns(&self) -> Option<u64> {
        self.estimated_per_item_ns
            .map(|per_item| (per_item as u64).saturating_mul(self.batch_size as u64))
    }

    /// Builder: per-task scheduler overhead (Tiny-Tasks). See field
    /// docs on [`Self::task_overhead_ns`].
    pub fn with_task_overhead_ns(mut self, overhead_ns: u32) -> Self {
        self.task_overhead_ns = Some(overhead_ns);
        self
    }

    /// Builder: per-task critical-path span (Tiny-Tasks). See field
    /// docs on [`Self::task_span_ns`].
    pub fn with_task_span_ns(mut self, span_ns: u32) -> Self {
        self.task_span_ns = Some(span_ns);
        self
    }

    /// Builder: caller-supplied effective task count (Tiny-Tasks).
    /// See field docs on [`Self::effective_task_count`].
    pub fn with_effective_task_count(mut self, count: u32) -> Self {
        self.effective_task_count = Some(count);
        self
    }

    /// Builder: SIMD lane fanout for matmul kernels (BLIS K_inner).
    /// See field docs on [`Self::k_inner_log2`].
    pub fn with_k_inner_log2(mut self, log2: u8) -> Self {
        self.k_inner_log2 = Some(log2);
        self
    }

    /// Builder: route this job to a specific backend. See field
    /// docs on [`Self::backend_hint`].
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend_hint = Some(backend);
        self
    }

    /// Builder: caller-supplied per-element cost estimate. Same
    /// shape as [`Self::with_estimated_per_item_ns`] but the
    /// canonical name in the scheduler-tuning vocabulary.
    /// `for_each_chunk` reads this to derive the optimal leaf
    /// count (cap dispatch overhead at a fraction of per-leaf
    /// work) and the inline-collapse threshold.
    pub fn with_cost_ns_per_elem(mut self, ns_per_elem: u32) -> Self {
        self.estimated_per_item_ns = Some(ns_per_elem);
        self.estimated_per_item_ns_explicit = true;
        self
    }

    /// Builder: override the leaf-count oversubscription factor as
    /// `log2(leaves_per_worker)`. Clamps to `[0, 3]` internally
    /// (1..8 leaves per worker). Set to 0 to force one leaf per
    /// worker (no oversubscription), 1 for the default 2x, 2 for
    /// 4x (latency-bound default), 3 for 8x (maximum steal headroom).
    pub fn with_oversubscription_log2(mut self, log2: u8) -> Self {
        self.oversubscription_log2 = Some(log2.min(3));
        self
    }

    /// Builder: cap the worker count for this dispatch. `1` forces
    /// serial execution on the calling thread; `physical_cores`
    /// keeps the SMT siblings parked even when `use_smt = true`;
    /// the default (None) uses the full per-arena worker count.
    pub fn with_workers(mut self, workers: u32) -> Self {
        self.worker_cap = Some(workers.max(1));
        self
    }

    /// Builder: select an experimental `for_each_chunk` policy
    /// variant for bench A/B comparison. See [`BisectVariant`]
    /// for the variant catalogue.
    pub fn with_bisect_variant(mut self, variant: BisectVariant) -> Self {
        self.bisect_variant = Some(variant);
        self
    }

    /// Effective per-element cost in nanoseconds: returns the
    /// explicit override if set, else `None`. `for_each_chunk`
    /// uses this for oversubscription tuning when set, falling
    /// back to the global split-multiplier when unset.
    #[inline]
    pub fn effective_ns_per_elem(&self) -> Option<u32> {
        self.estimated_per_item_ns
    }

    /// Effective leaf-count oversubscription `log2` for this plan.
    /// Returns the explicit override when set, otherwise the
    /// conservative default (`1` = 2x oversubscription).
    #[inline]
    pub fn effective_oversubscription_log2(&self) -> u8 {
        self.oversubscription_log2.unwrap_or(1)
    }

    /// Variance-corrected SMT activation decision. Combines the
    /// plan's `use_smt` prior with the observer's measured per-
    /// leaf execution-time variance:
    ///
    /// - If the plan explicitly sets `use_smt = false`, returns
    ///   `false` (no observer signal re-enables SMT).
    /// - If the plan sets `use_smt = true` AND the observer has
    ///   recorded measured per-leaf time variance with low cv^2
    ///   (per-mille < 50, i.e. nearly uniform leaves), returns
    ///   `false`. SMT siblings contest the same execution unit
    ///   on uniform-cost work and produce no gain.
    /// - Otherwise returns `true`, trusting the plan's prior.
    ///
    /// The variance signal prefers this plan's per-call-site
    /// history when a site is attached and has recorded at least 4
    /// leaves, so a uniform-cost closure gets SMT suppressed from
    /// ITS OWN evidence even when other closures in the process are
    /// high-variance. Site-less plans consult the process-wide
    /// counters. The first call at any site sees no variance
    /// history and pays the SMT cost; subsequent calls converge
    /// within a few samples.
    #[inline]
    pub fn effective_use_smt(&self) -> bool {
        if !self.use_smt {
            return false;
        }
        let threshold = crate::sched::adaptive_profile::class_thresholds()
            .cv2_low_per_mille
            .load(core::sync::atomic::Ordering::Relaxed);
        if let Some(site) = self.site
            && let Some(cv2) = site.get().cv2_per_mille()
        {
            return cv2 >= threshold;
        }
        let stats = crate::sched::split_observer::snapshot_leaf_stats();
        !matches!(
            crate::sched::split_observer::leaf_cv_squared_per_mille(stats),
            Some(cv2) if cv2 < threshold
        )
    }

    /// Resolve the backend this plan should dispatch to. If
    /// [`Self::backend_hint`] is `Some(b)` and a matching backend
    /// is registered, returns that. Otherwise returns the always-
    /// available CPU backend.
    ///
    /// This is the single resolution point every routing helper
    /// goes through; consumers do not need to handle the hint /
    /// fallback logic themselves.
    pub fn pick_backend(&self) -> BackendRef {
        if let Some(hint) = self.backend_hint
            && let Some(backend) = crate::backend::backend_by_id(&hint)
        {
            return backend;
        }
        cpu_backend()
    }

    /// Optimal chunk count via the Tiny-Tasks model (Acar 2013):
    ///
    /// ```text
    /// C_opt = clamp(sqrt(W * P / O), 1, N)
    /// ```
    ///
    /// where `W` is total estimated work in ns, `P` is worker count,
    /// `O` is per-task overhead, and `N` is the effective task count
    /// (`effective_task_count` if set, else `batch_size`).
    ///
    /// Returns `None` if `estimated_per_item_ns` or `task_overhead_ns`
    /// are unset - callers must fall back to the SLAW splitter's
    /// budget heuristic.
    ///
    /// Derivation sketch: total-time = serial_work / C + overhead * C.
    /// d/dC = -W/C^2 + O = 0 ⇒ C = sqrt(W / O). Including parallelism
    /// (P workers running in parallel) bumps the optimal C up by a
    /// factor of sqrt(P) because each worker's overhead is paid
    /// once but the work is divided P-ways.
    #[inline]
    pub fn optimal_chunk_count(&self, workers: usize) -> Option<u32> {
        let per_item = self.estimated_per_item_ns? as u64;
        let overhead = self.task_overhead_ns? as u64;
        let n = self.effective_task_count.unwrap_or(self.batch_size) as u64;
        if n == 0 || overhead == 0 || per_item == 0 {
            return Some(n as u32);
        }
        let w_total = per_item.saturating_mul(n);
        let p = (workers as u64).max(1);
        // C = sqrt(W * P / O). Use isqrt to stay in integer space.
        let radicand = w_total
            .saturating_mul(p)
            .checked_div(overhead)
            .unwrap_or(0);
        let c_opt = integer_sqrt(radicand);
        let c = c_opt.max(1).min(n);
        // Account for span: if the workload has a serial span, the
        // floor is `W / (span + overhead)` (Amdahl-style). Currently
        // unused in the formula because span estimates are rare; left
        // as forward-compat for when ops thread it through.
        Some(c as u32)
    }
}

/// Integer square root for u64. Used by `optimal_chunk_count` to
/// stay out of f64 (which loses precision past 2^53). Newton's
/// method bottoms out in O(log log W) iterations from the seed.
#[inline]
fn integer_sqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    // Seed with f64 sqrt for the initial guess; refine via Newton.
    let mut x = (n as f64).sqrt() as u64;
    // Two Newton steps are enough at u64 scale.
    for _ in 0..2 {
        if x == 0 { return 0; }
        let y = (x + n / x) >> 1;
        if y >= x { break; }
        x = y;
    }
    while x * x > n { x -= 1; }
    while (x + 1) * (x + 1) <= n { x += 1; }
    x
}

/// Classify a K_outer into its tier band before considering
/// batch-size or NUMA topology. Returns the tier the job would land
/// at if batch_size and topology were "typical."
///
/// - K <= 4  -> Inline
/// - K = 5..7 -> Local
/// - K = 8..10 -> Hierarchical
/// - K >= 11 -> Federated
pub fn kband_for(k_outer: u8) -> SchedTier {
    match k_outer {
        0..=4 => SchedTier::Inline,
        5..=7 => SchedTier::Local,
        8..=10 => SchedTier::Hierarchical,
        _ => SchedTier::Federated,
    }
}

/// Pick the scheduler tier for a job given its plan and the host's
/// NUMA topology. Honors the K-band classification with two
/// adjustments:
///
/// 1. Local / Hierarchical work with `batch_size < 32` falls back to
///    Inline because tier-dispatch overhead dominates one or two
///    items.
/// 2. K_outer in the Hierarchical band collapses to Local when the
///    host is single-NUMA (no benefit from per-node arenas).
///
/// Marked `#[inline(always)]` so the tier dispatch collapses at
/// every call site where `k_outer` or `batch_size` is a compile-time
/// constant (typical for `FpN<const N: usize>` const-generic call
/// chains). LLVM eliminates the kband_for match + the
/// batch_size threshold branches via const propagation.
#[inline(always)]
pub fn pick_tier(plan: &JobPlan, topo: &NumaTopology) -> SchedTier {
    // Caller-explicit parallelism opt-in: if the caller has set a
    // leaf-shape hint (LeafShape::PortCompute, LatencyCompute,
    // etc.) via the `with_leaf_shape` builder, the workload is
    // NOT a BigFloat-tier op for which K_outer characterizes per-
    // op cost. Treat the leaf-shape signal as authoritative:
    // route to the worker pool whenever the batch can plausibly
    // amortize dispatch (>= 8 items as a floor; per-item shape
    // signals the actual cost) regardless of K_outer. Without
    // this, callers that supply K=0 because their workload has no
    // number-system precision tier (lexers, generic data-parallel
    // ops) are silently routed to inline serial execution,
    // defeating the entire point of constructing the plan via the
    // closing-loop classifier.
    //
    // We deliberately check ONLY `leaf_shape`. The other plan
    // fields that look like they signal intent (`bisect_variant`,
    // `estimated_per_item_ns`) are auto-populated by
    // `JobPlan::new` via the static classifier and the per-arch
    // variant routing; they reflect classifier defaults rather
    // than caller assertion. `with_leaf_shape` is the only
    // builder that sets `leaf_shape` to a non-`Unknown` value, so
    // it is the unambiguous opt-in signal.
    let has_explicit_shape =
        plan.leaf_shape != crate::sched::adaptive_profile::LeafShape::Unknown;
    if has_explicit_shape && plan.batch_size >= 8 {
        // Caller said "this is parallel work." Honor it; fall
        // through to the per-tier batch-size adjustments below
        // by re-binding base to Local for the K=0..=4 case so the
        // tier-fallback chain still respects the multi-NUMA
        // promotion logic for Hierarchical-band K values.
        let base = match kband_for(plan.k_outer) {
            SchedTier::Inline => SchedTier::Local,
            other => other,
        };
        return match base {
            SchedTier::Local => SchedTier::Local,
            SchedTier::Hierarchical => {
                if topo.is_multi_node() {
                    SchedTier::Hierarchical
                } else {
                    SchedTier::Local
                }
            }
            SchedTier::Federated => SchedTier::Federated,
            // Unreachable: the match above promoted Inline to Local.
            SchedTier::Inline => SchedTier::Local,
        };
    }
    // Heavy-per-item small-N override: when the caller has supplied a
    // per-item cost hint AND predicted total work exceeds the
    // dispatch breakeven (~50us), promote out of Inline regardless of
    // batch_size. The N >= 256 / N >= 32 floors below were tuned for
    // workloads where per_item ~ 10ns to ~1us. For per_item ~ 10us+
    // (BigFloat verify, sqrt-chains, FMA-heavy kernels) even N=4 is
    // worth dispatching. Without this check `pick_tier` would route
    // 4-item-of-10ms-each workloads to Inline, the join would
    // serialize the bisect, and `for_each_chunk`'s adaptive
    // min_leaf would never get a chance to fire.
    //
    // Threshold matches `for_each_chunk::INLINE_COLLAPSE_THRESHOLD_NS`
    // so the two layers agree on what counts as "worth the pool".
    const HEAVY_OVERRIDE_THRESHOLD_NS: u64 = 50_000;
    // Small hosts (< 4 physical cores) have no steal-parallelism
    // headroom; demand 4x more predicted work before promoting out
    // of Inline.
    let heavy_override_threshold_ns = HEAVY_OVERRIDE_THRESHOLD_NS
        .saturating_mul(crate::cpu_info::small_host_dispatch_factor());
    // Only trust the per-item-cost estimate when the caller marked it
    // authoritative (via with_estimated_per_item_ns or the probe-path
    // amended_plan). Classifier defaults are routing hints and would
    // produce false positives (12ns * 32 = 384ns -> never fires) or
    // miss false negatives in the heavy direction without measurement.
    let heavy_override = plan.estimated_per_item_ns_explicit
        && plan.estimated_per_item_ns
            .map(|ns| (ns as u64).saturating_mul(plan.batch_size as u64) >= heavy_override_threshold_ns)
            .unwrap_or(false);

    let base = kband_for(plan.k_outer);
    match base {
        SchedTier::Inline => {
            // K <= 4 (n_limbs <= 16) per-op cost is ~50ns-2us.
            // Promote to Local when the BATCH is large enough that
            // aggregate parallel-iteration work amortizes dispatch,
            // OR when the caller's heavy-per-item override fires.
            //
            // The 256 threshold assumes the in-worker fast path's
            // ~50ns per-join dispatch (WorkerThread fast path +
            // adaptive splitter + pin-default-off + adaptive
            // peer-probe together); the breakeven is ~256 items
            // for BigFloat-sized ops. Below 256 the per-call
            // wrapper-inject cost (~10us) still exceeds the
            // work-time savings; above, parallel wins. At ~60us
            // per-join dispatch the breakeven measures ~1024.
            //
            // For lighter ops (e.g., u32 limb-add at sub-100ns
            // per item) the breakeven is higher, but the per-call
            // cost is small enough that 256-with-serial-fallback
            // is the right default for hint-less callers.
            //
            // use_smt promotes for the same reason as in the Local
            // band below: with_smt, set_profile(LatencyBound), and
            // the hint-less batch <= 32 classifier all set it for
            // heavy-per-item work. The par_iter probes reclassify
            // light hint-less work to FineGrain (use_smt off), so
            // tiny light batches stay inline.
            if plan.batch_size >= 256 || heavy_override || plan.use_smt {
                SchedTier::Local
            } else {
                SchedTier::Inline
            }
        }
        SchedTier::Local => {
            // Small batch falls back to Inline UNLESS one of:
            //   - heavy_override fired (explicit ns hint says total
            //     work >= 50us)
            //   - classifier set use_smt=true (LatencyBound profile,
            //     which means "heavy per-item with long FP chains" --
            //     for K=5..7 with no ns hint, the no-hint fallback
            //     route in infer_class_static maps batch_size<=32 to
            //     LatencyBound on the assumption that small-batch
            //     hint-less calls are usually NMFD-shape heavy items)
            //
            // Without the use_smt gate, NMFD-like 5-item-x-100ms
            // workloads from JobPlan::new (no ns hint) route to
            // Inline -> serial 500ms instead of parallel ~125ms,
            // measured as 2x rayon-slowdown on cold_workloads bench
            // `flynnel_def` column. Adding the use_smt check makes
            // the static classifier's class choice authoritative
            // for the inline-vs-parallel decision.
            if plan.batch_size < 32 && !heavy_override && !plan.use_smt {
                SchedTier::Inline
            } else {
                SchedTier::Local
            }
        }
        SchedTier::Hierarchical => {
            if !topo.is_multi_node() {
                if plan.batch_size < 32 && !heavy_override && !plan.use_smt {
                    SchedTier::Inline
                } else {
                    SchedTier::Local
                }
            } else {
                SchedTier::Hierarchical
            }
        }
        SchedTier::Federated => SchedTier::Federated,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::Variant;
    use crate::numa_topology::NumaTopology;

    #[test]
    fn for_profile_sets_smt_cost_and_oversubscription_together() {
        // LatencyBound: SMT on, 600 ns/elem, 4x oversubscription.
        let plan = JobPlan::set_profile(8, 1024, DispatchProfile::LatencyBound);
        assert!(plan.use_smt);
        assert_eq!(plan.estimated_per_item_ns, Some(600));
        assert_eq!(plan.oversubscription_log2, Some(2));

        // PortBound: SMT off, 12 ns/elem, 2x oversubscription.
        let plan = JobPlan::set_profile(8, 1024, DispatchProfile::PortBound);
        assert!(!plan.use_smt);
        assert_eq!(plan.estimated_per_item_ns, Some(12));
        assert_eq!(plan.oversubscription_log2, Some(1));

        // MemoryBound: SMT on, 50 ns/elem, 2x oversubscription.
        let plan = JobPlan::set_profile(8, 1024, DispatchProfile::MemoryBound);
        assert!(plan.use_smt);
        assert_eq!(plan.estimated_per_item_ns, Some(50));
        assert_eq!(plan.oversubscription_log2, Some(1));

        // Unspecified: SMT off, no cost estimate, 2x oversubscription.
        let plan = JobPlan::set_profile(8, 1024, DispatchProfile::Unspecified);
        assert!(!plan.use_smt);
        assert_eq!(plan.estimated_per_item_ns, None);
        assert_eq!(plan.oversubscription_log2, Some(1));
    }

    #[test]
    fn direct_knobs_override_profile_defaults() {
        let plan = JobPlan::set_profile(8, 1024, DispatchProfile::LatencyBound)
            .with_cost_ns_per_elem(42)
            .with_oversubscription_log2(0)
            .with_workers(4);
        assert_eq!(plan.estimated_per_item_ns, Some(42));
        assert_eq!(plan.oversubscription_log2, Some(0));
        assert_eq!(plan.worker_cap, Some(4));
        // use_smt still inherited from the profile (LatencyBound).
        assert!(plan.use_smt);
    }

    #[test]
    fn with_workers_enforces_minimum_of_one() {
        let plan = JobPlan::new(8, 1024).with_workers(0);
        assert_eq!(plan.worker_cap, Some(1));
    }

    #[test]
    fn with_oversubscription_log2_clamps_to_max_three() {
        let plan = JobPlan::new(8, 1024).with_oversubscription_log2(99);
        assert_eq!(plan.oversubscription_log2, Some(3));
    }

    #[test]
    fn effective_use_smt_respects_plan_off_regardless_of_variance() {
        // When the plan explicitly says SMT off, no variance
        // measurement can re-enable it. Use `bare` because
        // `JobPlan::new` now runs the static initial classifier
        // which may pick a SMT-on profile (LatencyBound for K>=7).
        let plan = JobPlan::bare(8, 1024);
        assert!(!plan.use_smt);
        assert!(!plan.effective_use_smt());
    }

    #[test]
    fn sched_tier_all_covers_four_tiers() {
        assert_eq!(SchedTier::ALL.len(), 4);
        // Every variant appears exactly once.
        for &t in &SchedTier::ALL {
            assert_eq!(SchedTier::ALL.iter().filter(|&&x| x == t).count(), 1);
        }
    }

    #[test]
    fn spin_rounds_match_tier_design() {
        assert_eq!(SchedTier::Inline.spin_rounds(), 0);
        assert_eq!(SchedTier::Local.spin_rounds(), 8);
        assert_eq!(SchedTier::Hierarchical.spin_rounds(), 32);
        assert_eq!(SchedTier::Federated.spin_rounds(), 0);
    }

    #[test]
    fn kband_inline_covers_micro_sizes() {
        for k in 0..=4u8 {
            assert_eq!(kband_for(k), SchedTier::Inline, "K={k} should be Inline");
        }
    }

    #[test]
    fn kband_local_covers_small_sizes() {
        for k in 5..=7u8 {
            assert_eq!(kband_for(k), SchedTier::Local, "K={k} should be Local");
        }
    }

    #[test]
    fn kband_hierarchical_covers_mid_sizes() {
        for k in 8..=10u8 {
            assert_eq!(kband_for(k), SchedTier::Hierarchical, "K={k} should be Hierarchical");
        }
    }

    #[test]
    fn kband_federated_covers_large_sizes() {
        for k in 11..=20u8 {
            assert_eq!(kband_for(k), SchedTier::Federated, "K={k} should be Federated");
        }
    }

    #[test]
    fn hw_class_matrix_extension_classification() {
        // Vector-SIMD regime: K_R = 0..6
        assert!(!HwClass::Scalar.is_matrix_extension());
        assert!(!HwClass::Sse2.is_matrix_extension());
        assert!(!HwClass::Avx2.is_matrix_extension());
        assert!(!HwClass::Avx512f.is_matrix_extension());
        assert!(!HwClass::Avx512Bf16.is_matrix_extension());
        assert!(!HwClass::Avx512Vnni.is_matrix_extension());
        assert!(!HwClass::Neon.is_matrix_extension());
        // Matrix-extension regime: K_R = 10..16
        assert!(HwClass::Sme.is_matrix_extension());
        assert!(HwClass::AmxBf16.is_matrix_extension());
        assert!(HwClass::AmxInt8.is_matrix_extension());
        assert!(HwClass::AmxFp16.is_matrix_extension());
        assert!(HwClass::TensorCoreHopper.is_matrix_extension());
        assert!(HwClass::TensorCoreBlackwell.is_matrix_extension());
    }

    #[test]
    fn job_plan_builder_chains() {
        let p = JobPlan::new(8, 100)
            .with_hw_class(HwClass::Avx2)
            .with_variant(Variant::Correct)
            .with_numa_hint(1);
        assert_eq!(p.k_outer, 8);
        assert_eq!(p.batch_size, 100);
        assert_eq!(p.hw_class, HwClass::Avx2);
        assert_eq!(p.variant, Variant::Correct);
        assert_eq!(p.numa_hint, Some(1));
    }

    #[test]
    fn pick_tier_inline_for_micro_k_small_batch() {
        let topo = NumaTopology::fallback();
        // Small K + small batch: serial wins.
        let plan = JobPlan::new(2, 100);
        assert_eq!(pick_tier(&plan, &topo), SchedTier::Inline);
    }

    #[test]
    fn pick_tier_inline_k_promotes_to_local_on_large_batch() {
        let topo = NumaTopology::fallback();
        // Small K but huge batch: aggregate work dominates;
        // parallel iteration pays off.
        let plan = JobPlan::new(2, 1_000_000);
        assert_eq!(pick_tier(&plan, &topo), SchedTier::Local);
    }

    #[test]
    fn pick_tier_inline_for_small_batch_local_k() {
        let topo = NumaTopology::fallback();
        // PortBound profile (NOT LatencyBound) so the small-batch
        // inline-fallback fires. JobPlan::new(6, 16) without an ns
        // hint now routes through the no-hint classifier fallback
        // which maps batch_size<=32 to LatencyBound on the
        // assumption that small-batch hint-less calls are usually
        // NMFD-shape heavy items; LatencyBound disables the
        // inline-fallback gate (use_smt=true). To exercise the
        // fallback path, use an explicit PortBound profile that
        // sets use_smt=false.
        let plan = JobPlan::set_profile(6, 16, DispatchProfile::PortBound);
        assert_eq!(pick_tier(&plan, &topo), SchedTier::Inline,
            "PortBound + batch_size 16 in Local-K band should fall back to Inline");
    }

    #[test]
    fn pick_tier_local_for_small_batch_latency_bound_no_hint() {
        let topo = NumaTopology::fallback();
        // JobPlan::new(6, 5) without ns hint -- the no-hint
        // classifier routes batch_size<=32 to LatencyBound, which
        // sets use_smt=true and bypasses the small-batch inline
        // fallback. NMFD-shape workloads (5 heavy items) get the
        // parallel worker pool on call 1 without any caller-supplied
        // ns hint, avoiding the 2x slowdown measured on the
        // cold_workloads flynnel_def NMFD-like cells when such
        // workloads route to Inline.
        let plan = JobPlan::new(6, 5);
        assert_eq!(pick_tier(&plan, &topo), SchedTier::Local,
            "small-batch LatencyBound (classifier no-hint default) stays Local");
    }

    #[test]
    fn pick_tier_local_for_normal_batch_local_k() {
        let topo = NumaTopology::fallback();
        let plan = JobPlan::new(6, 1024);
        assert_eq!(pick_tier(&plan, &topo), SchedTier::Local);
    }

    #[test]
    fn pick_tier_hierarchical_k_collapses_to_local_on_single_node() {
        let topo = NumaTopology::fallback();
        // Single-NUMA host: Hierarchical band should collapse to Local.
        let plan = JobPlan::new(9, 1024);
        assert_eq!(pick_tier(&plan, &topo), SchedTier::Local);
    }

    #[test]
    fn pick_tier_hierarchical_k_stays_hierarchical_on_multi_node() {
        // Build a synthetic 2-node topology.
        let mut topo = NumaTopology::fallback();
        topo.num_nodes = 2;
        topo.node_of_cpu = vec![0, 0, 1, 1];
        topo.distances = vec![vec![10, 20], vec![20, 10]];
        let plan = JobPlan::new(9, 1024);
        assert_eq!(pick_tier(&plan, &topo), SchedTier::Hierarchical);
    }

    #[test]
    fn integer_sqrt_matches_f64_for_small_values() {
        for n in 0u64..1000 {
            let s = super::integer_sqrt(n);
            assert!(s * s <= n);
            assert!((s + 1) * (s + 1) > n);
        }
    }

    #[test]
    fn integer_sqrt_at_pow2_boundaries() {
        for shift in 0..63 {
            let n = 1u64 << shift;
            let s = super::integer_sqrt(n);
            assert!(s * s <= n, "n=2^{shift} integer_sqrt={s} squared exceeds n");
            assert!(
                (s + 1) * (s + 1) > n,
                "n=2^{shift} integer_sqrt={s}+1 squared below n",
            );
        }
    }

    #[test]
    fn optimal_chunk_count_none_without_estimates() {
        let p = JobPlan::new(6, 1024);
        assert!(p.optimal_chunk_count(8).is_none());
        let p2 = JobPlan::new(6, 1024).with_estimated_per_item_ns(50);
        // task_overhead_ns still None.
        assert!(p2.optimal_chunk_count(8).is_none());
    }

    #[test]
    fn optimal_chunk_count_scales_with_workers_and_inversely_with_overhead() {
        let base = JobPlan::new(6, 1024)
            .with_estimated_per_item_ns(50)
            .with_task_overhead_ns(500);
        let c1 = base.optimal_chunk_count(1).unwrap();
        let c8 = base.optimal_chunk_count(8).unwrap();
        // More workers means more chunks make sense.
        assert!(c8 >= c1, "c8={c8} should be >= c1={c1}");

        // Higher overhead -> fewer chunks (each chunk costs more).
        let heavy_overhead = JobPlan::new(6, 1024)
            .with_estimated_per_item_ns(50)
            .with_task_overhead_ns(5000);
        let c_heavy = heavy_overhead.optimal_chunk_count(8).unwrap();
        assert!(c_heavy <= c8, "heavier overhead should reduce chunks");
    }

    #[test]
    fn optimal_chunk_count_clamps_to_effective_task_count() {
        let p = JobPlan::new(6, 1_000_000)
            .with_estimated_per_item_ns(1)
            .with_task_overhead_ns(1_000_000_000)
            .with_effective_task_count(16);
        // With huge overhead the formula collapses to "1 chunk";
        // clamp to effective_task_count=16.
        let c = p.optimal_chunk_count(8).unwrap();
        assert!(c <= 16, "chunk count must clamp at effective_task_count");
    }

    #[test]
    fn pick_tier_federated_for_large_k() {
        let topo = NumaTopology::fallback();
        let plan = JobPlan::new(13, 1);
        // Federated does NOT collapse on batch_size; large K is its
        // own concern (single op may take milliseconds).
        assert_eq!(pick_tier(&plan, &topo), SchedTier::Federated);
    }
}

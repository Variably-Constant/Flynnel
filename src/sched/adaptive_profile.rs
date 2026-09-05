//! Adaptive DispatchProfile + WorkloadClass migration.
//!
//! The K_gating adaptive pattern proven in
//! [`crate::sched::adaptive_worker`] generalizes to ANY routing
//! decision that's consulted ONCE per dispatch (not per push/pop).
//! DispatchProfile is the canonical example: it drives `use_smt`,
//! `oversubscription_log2`, `estimated_per_item_ns`,
//! `use_mailbox_routing`, and `deque_tier_hint` - all of which are
//! read at `JobPlan::set_profile` construction time, not at the
//! deque hot path.
//!
//! This module exposes:
//! - [`WorkloadClass`]: the user-facing Light / Compute / Heavy /
//!   Memory vocabulary that maps to a [`DispatchProfile`]
//! - [`active_dispatch_profile`] / [`migrate_dispatch_profile`]:
//!   global active-profile read / swap via a single AtomicU8 tag
//! - [`migrate_workload_class`]: high-level swap via WorkloadClass
//!
//! ## Cost analysis
//!
//! - Per-op cost on the deque hot path: **zero** (profile is
//!   consulted at plan-construction time, never per push/pop)
//! - Migration cost: **one AtomicU8 Release-store** (~1 ns)
//! - Subsequent dispatch reads the new profile via one AtomicU8
//!   Acquire-load at `JobPlan::set_profile` time (~1 ns)
//!
//! Applications observe per-call elapsed-vs-expected ratio and
//! call [`migrate_workload_class`] when the observed pattern
//! doesn't match the active class. The flip is FREE on the per-op
//! path; only the application's classification cost matters.

#![allow(clippy::missing_errors_doc)]

use core::sync::atomic::{AtomicU8, Ordering};

use crate::dispatch_profile::DispatchProfile;

/// User-facing workload-class vocabulary. Maps to a
/// [`DispatchProfile`] via [`Self::to_dispatch_profile`].
///
/// The application is expected to classify its workload (via
/// observed per-item elapsed time or a-priori knowledge) and
/// migrate the active class when the workload shifts. Cost: one
/// atomic store per migration; zero on the per-op path.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum WorkloadClass {
    /// Tiny per-item cost (~< 50 ns/item); dispatch overhead
    /// dominates. Maps to [`DispatchProfile::PortBound`] because the
    /// inline-collapse decision is the dominant scheduler concern
    /// at this scale and the leaf body is typically port-bound when
    /// it does parallelize.
    FineGrain,
    /// Medium per-item cost (~50-500 ns/item); port-saturated
    /// pipeline (typically integer multiply or FMA on a single
    /// execution port). SMT siblings parked because they would
    /// contest the same port. Maps to [`DispatchProfile::PortBound`].
    PortBound,
    /// Large per-item cost (~> 500 ns/item); long FP dependency
    /// chains stall the pipeline. SMT siblings active to fill the
    /// stall bubbles. Maps to [`DispatchProfile::LatencyBound`].
    LatencyBound,
    /// Memory work with irregular access (pointer-chase, gather/
    /// scatter, hash probes, graph traversal). SMT siblings active
    /// to interleave cache misses. Maps to
    /// [`DispatchProfile::MemoryBound`].
    MemoryBound,
    /// Sequential streaming where per-core memory bandwidth is the
    /// bottleneck (byte scan, image kernels, prefix-sum block sums,
    /// histogram). SMT siblings parked - both threads on the same
    /// physical core would compete for the same L2/L3 bandwidth
    /// instead of helping. Maps to [`DispatchProfile::Streaming`].
    Streaming,
}

impl WorkloadClass {
    /// Map the user-facing class to the scheduler's
    /// [`DispatchProfile`].
    #[inline]
    pub fn to_dispatch_profile(self) -> DispatchProfile {
        match self {
            WorkloadClass::FineGrain => DispatchProfile::PortBound,
            WorkloadClass::PortBound => DispatchProfile::PortBound,
            WorkloadClass::LatencyBound => DispatchProfile::LatencyBound,
            WorkloadClass::MemoryBound => DispatchProfile::MemoryBound,
            WorkloadClass::Streaming => DispatchProfile::Streaming,
        }
    }
}

/// Caller-supplied hint about the structural shape of a workload.
/// When set on a [`crate::JobPlan`] via
/// [`crate::JobPlan::with_workload_shape`], the static initial
/// classifier consults it directly instead of inferring from
/// (k_outer, batch_size, estimated_per_item_ns). Use this when you
/// know your workload's pattern at compile time -- it lets the
/// scheduler get the right routing on call 1 with no observer
/// refinement needed.
///
/// Map to [`WorkloadClass`] via [`Self::to_workload_class`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LeafShape {
    /// Compute-bound work whose bottleneck is the issue port
    /// (IMUL, FMA, integer compare). SMT siblings on the same
    /// physical core would contest the same port and produce
    /// no net throughput.
    ///
    /// Examples: branchy compare-and-swap sort, RNG IMUL loops,
    /// kmeans FMA-bound distance computation, Conway stencil
    /// add+match.
    PortCompute,
    /// Compute-bound work whose bottleneck is a long FP
    /// dependency chain (sqrt, div, chained Newton iterations).
    /// SMT siblings active to fill the dispatch bubbles between
    /// latency stalls.
    ///
    /// Examples: n-body sqrt chain per pair, recursive Newton
    /// iteration, branchy adaptive integrators, FFT butterflies
    /// with register dependencies.
    LatencyCompute,
    /// Sequential streaming where per-core memory bandwidth is
    /// the bottleneck. SMT siblings parked because both threads
    /// would compete for the same L2/L3 bandwidth.
    ///
    /// Examples: byte scans (grep), image kernels (rgb->gray,
    /// box blur), CSV parsing, sequential prefix-sum stages,
    /// histogram on large arrays.
    Streaming,
    /// Irregular memory access (gather, scatter, pointer chase).
    /// SMT siblings active to interleave cache-miss loads.
    ///
    /// Examples: sparse matrix-vector multiply (col_idx gather),
    /// PageRank graph adjacency gather, hash table probes,
    /// merkle tree pair lookups, strided column reads.
    Gather,
    /// Mixed / unknown shape. Fall back to (k_outer, batch_size,
    /// estimated_per_item_ns) heuristics in
    /// [`infer_class_static`].
    Unknown,
}

impl LeafShape {
    /// Map the shape directly to a [`WorkloadClass`] when the
    /// shape is known. Returns `None` for `Unknown` -- the caller
    /// (typically the static classifier) falls back to ns + K + N
    /// heuristics.
    #[inline]
    pub fn to_workload_class(self) -> Option<WorkloadClass> {
        match self {
            Self::PortCompute => Some(WorkloadClass::PortBound),
            Self::LatencyCompute => Some(WorkloadClass::LatencyBound),
            Self::Streaming => Some(WorkloadClass::Streaming),
            Self::Gather => Some(WorkloadClass::MemoryBound),
            Self::Unknown => None,
        }
    }
}

/// Encoded active-profile tag stored in [`ACTIVE_PROFILE_TAG`].
const TAG_LATENCY_BOUND: u8 = 0;
const TAG_PORT_BOUND: u8 = 1;
const TAG_MEMORY_BOUND: u8 = 2;
const TAG_UNSPECIFIED: u8 = 3;
const TAG_STREAMING: u8 = 4;

/// Global active-profile tag. Read by the scheduler when
/// constructing default JobPlans via [`active_dispatch_profile`];
/// flipped by [`migrate_dispatch_profile`] /
/// [`migrate_workload_class`]. Initial value: PortBound (the
/// flynnel calibration default for Zen+ R7 2700 - see the
/// realistic_bench results in the WorkerCtx swap commit).
static ACTIVE_PROFILE_TAG: AtomicU8 = AtomicU8::new(TAG_PORT_BOUND);

/// Linkage confirmation marker. When the binary links this
/// module, `nm <bin> | grep __flynnel_marker` returns this
/// symbol, confirming the adaptive profile dispatch path is
/// present in the build.
#[unsafe(no_mangle)]
pub static __flynnel_marker_adaptive_profile: u8 = 0;

/// Read the active DispatchProfile via one AtomicU8 Acquire-load.
/// Used by the scheduler to pick default plan knobs when the
/// caller doesn't specify a profile explicitly.
#[inline]
pub fn active_dispatch_profile() -> DispatchProfile {
    match ACTIVE_PROFILE_TAG.load(Ordering::Acquire) {
        TAG_LATENCY_BOUND => DispatchProfile::LatencyBound,
        TAG_PORT_BOUND => DispatchProfile::PortBound,
        TAG_MEMORY_BOUND => DispatchProfile::MemoryBound,
        TAG_STREAMING => DispatchProfile::Streaming,
        _ => DispatchProfile::Unspecified,
    }
}

/// Migrate the global active DispatchProfile via one AtomicU8
/// Release-store. Subsequent dispatches that consult
/// [`active_dispatch_profile`] see the new value; per-op cost on
/// the deque hot path is unchanged.
#[inline]
pub fn migrate_dispatch_profile(profile: DispatchProfile) {
    let tag = match profile {
        DispatchProfile::LatencyBound => TAG_LATENCY_BOUND,
        DispatchProfile::PortBound => TAG_PORT_BOUND,
        DispatchProfile::MemoryBound => TAG_MEMORY_BOUND,
        DispatchProfile::Streaming => TAG_STREAMING,
        DispatchProfile::Unspecified => TAG_UNSPECIFIED,
    };
    ACTIVE_PROFILE_TAG.store(tag, Ordering::Release);
}

/// High-level WorkloadClass migration. Maps to the underlying
/// [`DispatchProfile`] then calls [`migrate_dispatch_profile`].
#[inline]
pub fn migrate_workload_class(class: WorkloadClass) {
    migrate_dispatch_profile(class.to_dispatch_profile());
}

/// Active WorkloadClass derived from [`active_dispatch_profile`].
/// The [`DispatchProfile`] enum has one fewer variant than
/// [`WorkloadClass`] (there is no PortBound-vs-FineGrain
/// distinction on the profile side because the inline-collapse
/// decision is a downstream policy question, not a per-op tuning
/// input). This function collapses:
/// - `PortBound` and `Unspecified` -> [`WorkloadClass::PortBound`]
///   (the calibration default; `FineGrain` is only reachable via
///   [`classify_observed`] or [`infer_class_static`] when a real
///   per-leaf cost signal is available).
/// - other profiles map straight through to their same-named
///   [`WorkloadClass`] variant.
#[inline]
pub fn active_workload_class() -> WorkloadClass {
    match active_dispatch_profile() {
        DispatchProfile::LatencyBound => WorkloadClass::LatencyBound,
        DispatchProfile::PortBound => WorkloadClass::PortBound,
        DispatchProfile::MemoryBound => WorkloadClass::MemoryBound,
        DispatchProfile::Streaming => WorkloadClass::Streaming,
        DispatchProfile::Unspecified => WorkloadClass::PortBound,
    }
}

/// Host-calibratable classifier boundaries. Every field is an
/// atomic so [`calibrate_class_thresholds`] can update the live
/// values with plain Release stores while dispatch paths read them
/// with Relaxed loads (one MOV each on x86).
///
/// Defaults are empirically-tuned Zen-family values; calibration
/// replaces them with quantities measured on the running host in
/// the running binary, clamped to sane bands so a noisy calibration
/// run cannot produce nonsense.
pub struct ClassThresholds {
    /// Per-item ns below which work is FineGrain (dispatch overhead
    /// dominates). Default 50.
    pub fine_grain_ns: AtomicU64,
    /// Per-item ns boundary between PortBound and the heavy classes.
    /// Default 500.
    pub port_heavy_ns: AtomicU64,
    /// Per-item ns boundary between MemoryBound and LatencyBound in
    /// the hint-driven static classifier. Default 2000.
    pub memory_latency_ns: AtomicU64,
    /// cv^2 (parts-per-1000) below which leaves count as uniform.
    /// Default 50.
    pub cv2_low_per_mille: AtomicU64,
    /// cv^2 (parts-per-1000) at or above which leaves count as
    /// high-variance. Default 500.
    pub cv2_high_per_mille: AtomicU64,
    /// Reduce-merge cycle count at or below which a reducer counts
    /// as trivial for the flat-fanout gate in
    /// [`crate::sched::par_iter::reduce_chunks`]. Default 30_000.
    pub trivial_reduce_cycles: AtomicU64,
}

impl ClassThresholds {
    /// A fresh threshold set holding the documented defaults.
    /// `const` so it can back both the process-global static and
    /// caller-owned instances for isolated calibration.
    pub const fn new_defaults() -> Self {
        Self {
            fine_grain_ns: AtomicU64::new(50),
            port_heavy_ns: AtomicU64::new(500),
            memory_latency_ns: AtomicU64::new(2000),
            cv2_low_per_mille: AtomicU64::new(50),
            cv2_high_per_mille: AtomicU64::new(500),
            trivial_reduce_cycles: AtomicU64::new(30_000),
        }
    }
}

static CLASS_THRESHOLDS: ClassThresholds = ClassThresholds::new_defaults();

use core::sync::atomic::AtomicU64;

/// Live classifier thresholds. Reads are single Relaxed loads.
#[inline]
pub fn class_thresholds() -> &'static ClassThresholds {
    &CLASS_THRESHOLDS
}

/// Report returned by [`calibrate_class_thresholds`] so callers can
/// log what the probes measured and what was installed.
#[derive(Debug, Clone, Copy)]
pub struct ThresholdCalibration {
    /// Measured in-pool empty-join round-trip in nanoseconds.
    pub join_ns: u64,
    /// Installed FineGrain boundary derived from `join_ns`.
    pub fine_grain_ns: u64,
    /// Measured reference trivial-reduce merge in TSC cycles.
    pub trivial_reduce_measured_cycles: u64,
    /// Installed trivial-reduce ceiling (8x measured, clamped).
    pub trivial_reduce_cycles: u64,
    /// SMT-on / SMT-off wall-time ratio (per-mille) of the sqrt-chain
    /// probe: below 950 means SMT helped, above 1050 means it hurt.
    pub smt_ratio_per_mille: u64,
    /// Installed MemoryBound/LatencyBound boundary after the SMT
    /// nudge.
    pub memory_latency_ns: u64,
}

/// Measure this host's classifier boundaries and install them.
///
/// Three probes, all executed inside the running binary so the
/// installed values are always in the running build profile's own
/// cost units (cycle counts and dispatch floors differ between
/// debug and release builds; measuring in-binary keeps the
/// thresholds consistent with whichever profile is live):
///
/// 1. **Dispatch floor**: median of 64 in-pool empty `join` calls.
///    `fine_grain_ns = clamp(join_ns / MIN_LEAF_ITEMS, 25..=200)`;
///    per-item work below that is dominated by scheduling cost even
///    at full leaf granularity.
/// 2. **Trivial-reduce reference**: a 256-lane u64 element-wise add
///    (the histogram-merge shape) timed in TSC cycles.
///    `trivial_reduce_cycles = clamp(8 * measured, 5_000..=200_000)`.
/// 3. **SMT-benefit probe**: a latency-bound sqrt-chain batch run
///    with SMT siblings parked vs active. When SMT helps by >= 5%
///    the MemoryBound/LatencyBound boundary drops toward 1500 (SMT
///    pays off earlier on this host); when it hurts by >= 5% the
///    boundary rises toward 3000. Clamped to 1_000..=4_000.
///
/// Wall-clock cost: a few milliseconds. Runs synchronously on the
/// calling thread; see [`spawn_class_threshold_calibration`] for the
/// IoPool variant.
pub fn calibrate_class_thresholds() -> ThresholdCalibration {
    calibrate_class_thresholds_into(class_thresholds())
}

/// [`calibrate_class_thresholds`] with the install target supplied
/// by the caller. The measured, clamped values land in `target`
/// instead of the process-global thresholds, so a caller can
/// calibrate into an isolated [`ClassThresholds`] (inspection,
/// comparison against the live set) without changing routing for
/// the rest of the process.
pub fn calibrate_class_thresholds_into(target: &ClassThresholds) -> ThresholdCalibration {
    use crate::sched::plan::JobPlan;

    // Probe 1: in-pool empty-join floor. Warm the pool first so the
    // measurement sees steady state, not spawn cost.
    let plan = JobPlan::bare(6, 2);
    for _ in 0..16 {
        crate::sched::arena::join(&plan, || (), || ());
    }
    let mut samples = [0u64; 64];
    for s in samples.iter_mut() {
        let t0 = std::time::Instant::now();
        crate::sched::arena::join(&plan, || (), || ());
        *s = t0.elapsed().as_nanos() as u64;
    }
    samples.sort_unstable();
    let join_ns = samples[samples.len() / 2].max(1);
    let fine_grain_ns = (join_ns / 256).clamp(25, 200);
    target.fine_grain_ns.store(fine_grain_ns, Ordering::Release);

    // Probe 2: trivial-reduce reference in cycles. Uses the same
    // 256-bin element-wise-add shape the reduce_chunks gate was
    // tuned against.
    let mut a = [1u64; 256];
    let b = [2u64; 256];
    let mut best_cycles = u64::MAX;
    for _ in 0..16 {
        let t0 = read_tsc_cal();
        for i in 0..256 {
            a[i] = a[i].wrapping_add(b[i]);
        }
        std::hint::black_box(&a);
        let dt = read_tsc_cal().wrapping_sub(t0);
        best_cycles = best_cycles.min(dt.max(1));
    }
    let trivial_reduce_cycles = (best_cycles.saturating_mul(8)).clamp(5_000, 200_000);
    target
        .trivial_reduce_cycles
        .store(trivial_reduce_cycles, Ordering::Release);

    // Probe 3: SMT benefit on a latency-bound sqrt chain. Runs the
    // same fixed workload twice through the pool, once with the SMT
    // extension parked (default) and once with the siblings raised
    // via acquire_smt. On hosts without an SMT extension the two
    // runs are identical and the nudge is a no-op.
    let sqrt_probe = |use_smt: bool| -> u64 {
        let arena = crate::sched::arena::global_local_arena();
        let _guards = if use_smt { arena.acquire_smt() } else { Vec::new() };
        let mut items: Vec<f64> = (1..=2048u32).map(f64::from).collect();
        let probe_plan = JobPlan::bare(6, items.len() as u32);
        let t0 = std::time::Instant::now();
        crate::sched::par_iter::for_each_chunk(&probe_plan, &mut items, |chunk| {
            for x in chunk.iter_mut() {
                let mut v = *x;
                for _ in 0..64 {
                    v = v.sqrt() + 1.0;
                }
                *x = v;
            }
        });
        std::hint::black_box(&items);
        t0.elapsed().as_nanos() as u64
    };
    let off_ns = sqrt_probe(false).max(1);
    let on_ns = sqrt_probe(true).max(1);
    let smt_ratio_per_mille = on_ns.saturating_mul(1000) / off_ns;
    let current = target.memory_latency_ns.load(Ordering::Relaxed);
    let nudged = if smt_ratio_per_mille <= 950 {
        // SMT helped: latency-bound behavior starts earlier here.
        current.saturating_sub(250)
    } else if smt_ratio_per_mille >= 1050 {
        // SMT hurt: demand heavier work before classifying latency.
        current.saturating_add(250)
    } else {
        current
    };
    let memory_latency_ns = nudged.clamp(1_000, 4_000);
    target
        .memory_latency_ns
        .store(memory_latency_ns, Ordering::Release);

    ThresholdCalibration {
        join_ns,
        fine_grain_ns,
        trivial_reduce_measured_cycles: best_cycles,
        trivial_reduce_cycles,
        smt_ratio_per_mille,
        memory_latency_ns,
    }
}

/// TSC read for the calibration probes. Mirrors the par_iter
/// read_tsc shape (which is private to that module).
#[inline]
fn read_tsc_cal() -> u64 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_rdtsc` is part of the base x86_64 ISA with no
    // CPU-feature or operand-state preconditions.
    unsafe {
        std::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::time::Instant::now().elapsed().as_nanos() as u64
    }
}

/// Run [`calibrate_class_thresholds`] on the SMT-sibling IoPool.
/// No-op when the pool is disabled (`FLYNNEL_SCHED_SMT_AS_IO`
/// unset): the defaults stay live and the caller can invoke the
/// synchronous variant explicitly instead.
pub fn spawn_class_threshold_calibration() {
    if let Some(pool) = crate::sched::io_pool::global_io_pool() {
        pool.submit(|| {
            let report = calibrate_class_thresholds();
            std::hint::black_box(report);
        });
    }
}

/// Classify an observed leaf-time distribution into a
/// [`WorkloadClass`]. Used by the closing-loop observer to turn
/// raw `(mean_ns, cv2_per_mille)` samples into a routing decision.
///
/// Threshold rationale (matches the doc-comments on each
/// [`WorkloadClass`] variant):
///
/// - `mean_ns < 50`: per-leaf cost is in the dispatch-overhead
///   range; treat as FineGrain (routes to PortBound, no SMT, no
///   extra split budget).
/// - `mean_ns >= 50 and cv2 < 50`: low variance. Either:
///   * `mean_ns < 500` -> PortBound (regular compute, SMT off
///     because siblings contest the issue port).
///   * `mean_ns >= 500` -> Streaming (steady high-bandwidth
///     sequential work; SMT off because siblings compete for
///     L2 bandwidth).
/// - `mean_ns >= 50 and cv2 >= 500`: high variance. Either:
///   * `mean_ns < 500` -> still PortBound (short bursts with
///     scheduling jitter; SMT off).
///   * `mean_ns >= 500` -> LatencyBound (long irregular
///     dependency chains; SMT on to fill stalls).
/// - `mean_ns >= 500 and 50 <= cv2 < 500`: moderate variance,
///   long-ish leaves -> MemoryBound (cache-hierarchy effects
///   producing some leaves slower than others; SMT on to
///   interleave misses).
#[inline]
pub fn classify_observed(mean_ns: u64, cv2_per_mille: u64) -> WorkloadClass {
    let t = class_thresholds();
    if mean_ns < t.fine_grain_ns.load(Ordering::Relaxed) {
        return WorkloadClass::FineGrain;
    }
    if mean_ns < t.port_heavy_ns.load(Ordering::Relaxed) {
        return WorkloadClass::PortBound;
    }
    // Heavy per-leaf work: variance picks between streaming /
    // memory-gather / latency.
    if cv2_per_mille < t.cv2_low_per_mille.load(Ordering::Relaxed) {
        WorkloadClass::Streaming
    } else if cv2_per_mille < t.cv2_high_per_mille.load(Ordering::Relaxed) {
        WorkloadClass::MemoryBound
    } else {
        WorkloadClass::LatencyBound
    }
}

/// Static initial classifier: pick a [`WorkloadClass`] from JobPlan
/// parameters ALONE, with no leaf-time observation. Called from
/// [`crate::JobPlan::new`] so the FIRST dispatch already uses the
/// right routing instead of waiting for the auto-classifier
/// observer to converge over several iterations.
///
/// Inputs:
/// - `k_outer`: precision tier (0..12). Low tier = small per-op
///   cost; high tier = long FP dependency chains.
/// - `batch_size`: total work-unit count for the dispatch.
/// - `estimated_per_item_ns`: caller's hint, if any. When present
///   it dominates the classification (caller knows their workload
///   better than the heuristic).
///
/// Heuristic with ns_hint (caller's signal is authoritative):
/// - `ns < 50` -> FineGrain (per-item below dispatch breakeven)
/// - `50 <= ns < 500` -> PortBound (regular compute)
/// - `ns >= 500` -> MemoryBound (heavy per-item; conservative
///   middle pick because we lack cv2 at construction time. The
///   observer refines to Streaming or LatencyBound after a few
///   iterations if the observed cv2 differs from the assumption.)
///
/// Heuristic without ns_hint (fall back to K_outer + batch_size):
/// - `k_outer >= 7`: high-precision tier -> LatencyBound (long
///   FP dependency chains stall the pipeline; SMT-on helps)
/// - `k_outer <= 3 and batch_size >= 16384`: low-tier with large
///   data -> Streaming (bandwidth-bound bulk processing)
/// - everything else: PortBound (the safe middle pick that the
///   observer can shift via classification migration)
///
/// The observer remains the authority: if leaf-time observation
/// contradicts the static guess over a few iterations, it migrates
/// the global active class. Static gives the right routing on
/// call 1; the observer corrects a wrong call-1 guess.
#[inline]
pub fn infer_class_static(
    k_outer: u8,
    batch_size: u32,
    estimated_per_item_ns: Option<u32>,
) -> WorkloadClass {
    infer_class_static_with_shape(
        k_outer,
        batch_size,
        estimated_per_item_ns,
        LeafShape::Unknown,
    )
}

/// Same as [`infer_class_static`] but with an explicit caller
/// shape hint. The shape is the STRONGEST static signal: when
/// known it maps directly to a [`WorkloadClass`] and the ns + K
/// + N heuristics are skipped.
#[inline]
pub fn infer_class_static_with_shape(
    k_outer: u8,
    batch_size: u32,
    estimated_per_item_ns: Option<u32>,
    shape: LeafShape,
) -> WorkloadClass {
    // Strongest signal: caller's structural shape hint maps
    // directly to a class. The ns and K + N heuristics below are
    // refinement scaffolding for when the caller doesn't know
    // (or hasn't told us).
    if let Some(class) = shape.to_workload_class() {
        // Handle the FineGrain corner case: even when the caller
        // says "PortCompute" or "Streaming", a tiny total batch
        // should inline-collapse instead of dispatching.
        if let Some(ns) = estimated_per_item_ns {
            let total_ns = (ns as u64).saturating_mul(batch_size as u64);
            if total_ns < 50_000 {
                return WorkloadClass::FineGrain;
            }
        }
        return class;
    }

    if let Some(ns) = estimated_per_item_ns {
        let total_ns = (ns as u64).saturating_mul(batch_size as u64);
        // Below dispatch breakeven (~50 us total): no point
        // parallelizing; treat as fine-grain so the scheduler
        // takes the inline-collapse fast path if available.
        if total_ns < 50_000 {
            return WorkloadClass::FineGrain;
        }
        // Very heavy per-item (>= 2000 ns) without a shape hint
        // is more often a long FP dependency chain (Newton
        // iteration, sqrt chain, branchy adaptive integrator)
        // than a single cache-miss gather; default to
        // LatencyBound. Callers with a gather pattern should pass
        // LeafShape::Gather.
        //
        // NOTE: tried routing batch>=16384+ns>=2000 to PortBound
        // (theory: FMA-bound streaming would benefit from SMT-off)
        // but the empirical regression was 2.76x on cold_workloads
        // 16k_10us (8.13ms LatencyBound vs 19.02ms PortBound).
        // PortBound's default oversubscription_log2 is lower so
        // fewer leaves and worse work-stealing balance on big
        // batches. LatencyBound stays the best class for heavy
        // per-item even on big batches.
        let t = class_thresholds();
        if (ns as u64) >= t.memory_latency_ns.load(Ordering::Relaxed) {
            return WorkloadClass::LatencyBound;
        }
        // Moderate-heavy per-item (between the port_heavy and
        // memory_latency boundaries) is the cache-miss-gather
        // signature (spmv, pagerank, hash probes). Pick
        // MemoryBound; observer can refine to LatencyBound if
        // observed cv^2 says otherwise.
        if (ns as u64) >= t.port_heavy_ns.load(Ordering::Relaxed) {
            return WorkloadClass::MemoryBound;
        }
        // Light per-item but BIG total batch: bulk bandwidth-bound
        // streaming work over a >= 64 KiB-equivalent batch is the
        // signature pattern of byte scans, image kernels, and
        // histogram-style accumulators.
        if (ns as u64) < t.fine_grain_ns.load(Ordering::Relaxed) && batch_size >= 65536 {
            return WorkloadClass::Streaming;
        }
        // Medium per-item: integer pipeline saturating compute.
        return WorkloadClass::PortBound;
    }
    // No caller hint: fall back to K_outer + batch_size as proxies
    // for workload shape.
    if k_outer >= 7 {
        return WorkloadClass::LatencyBound;
    }
    if k_outer <= 3 && batch_size >= 16384 {
        return WorkloadClass::Streaming;
    }
    // Small batch (<=32) without a ns hint is most often a small
    // collection of HEAVY items: NMFD batches, ML inference passes,
    // image-tile transforms, recursive algorithm fan-outs. The
    // failure mode of routing these to PortBound (SMT-off, primaries
    // only) is that the worker pool is half-width and the user sees
    // ~2x slower than rayon on cold-cache one-off dispatches.
    // LatencyBound enables SMT, doubling effective throughput on
    // hyperthreaded hosts. The observer remains the safety net: if
    // leaf-time observation reveals the items are actually light,
    // it migrates atomically to PortBound within a few iterations.
    // Cross-host cold-workload bench shows this lift takes
    // flynnel_def from ~2x slower to TIE on NMFD-like cells.
    if batch_size <= 32 {
        return WorkloadClass::LatencyBound;
    }
    WorkloadClass::PortBound
}

/// How many consecutive auto-classifier ticks must agree to
/// migrate when the observed class is adjacent to the active class
/// (bucket distance == 1). Larger bucket distances bypass this
/// via the fast-adapt path in [`tick_auto_classify`]. Set to 2
/// so adjacent-bucket migrations happen within ~8 leaves of
/// observed evidence, fast enough for criterion warm-up to
/// converge before the measurement phase.
pub const AUTO_MIGRATION_HYSTERESIS: u32 = 2;

/// "Bucket distance" between two WorkloadClass values. Adjacent
/// buckets in the FineGrain -> PortBound -> Streaming/MemoryBound
/// -> LatencyBound progression have distance 1; the extremes have
/// distance 3. Used by the fast-adapt path: when the observer sees
/// the workload sitting at bucket-distance >= 2 from the active
/// class, it migrates IMMEDIATELY (single tick, no hysteresis)
/// because the static initial guess was clearly wrong.
#[inline]
fn bucket_index(class: WorkloadClass) -> u8 {
    match class {
        WorkloadClass::FineGrain => 0,
        WorkloadClass::PortBound => 1,
        WorkloadClass::Streaming => 2,
        WorkloadClass::MemoryBound => 2,
        WorkloadClass::LatencyBound => 3,
    }
}

#[inline]
fn bucket_distance(a: WorkloadClass, b: WorkloadClass) -> u8 {
    bucket_index(a).abs_diff(bucket_index(b))
}

/// Encoded class tag stored in [`AUTO_PENDING_TAG`]. Same
/// encoding as [`ACTIVE_PROFILE_TAG`] for the migrated portion
/// (LatencyBound / PortBound / MemoryBound / Streaming /
/// Unspecified) plus a TAG_FINE_GRAIN value because the
/// classifier can produce WorkloadClass::FineGrain which the
/// underlying profile encoding rolls into PortBound.
const TAG_FINE_GRAIN: u8 = 5;

#[inline]
fn workload_class_to_tag(class: WorkloadClass) -> u8 {
    match class {
        WorkloadClass::FineGrain => TAG_FINE_GRAIN,
        WorkloadClass::PortBound => TAG_PORT_BOUND,
        WorkloadClass::LatencyBound => TAG_LATENCY_BOUND,
        WorkloadClass::MemoryBound => TAG_MEMORY_BOUND,
        WorkloadClass::Streaming => TAG_STREAMING,
    }
}

/// Encode a [`WorkloadClass`] into the shared u8 tag space. Used by
/// [`crate::sched::call_site::CallSiteState`] so per-site classifier
/// state stores the same encoding the process-global tag uses.
#[inline]
pub(crate) fn class_tag_encode(class: WorkloadClass) -> u8 {
    workload_class_to_tag(class)
}

/// Decode a u8 tag back to a [`WorkloadClass`]. Returns `None` for
/// any value outside the encoded range (notably the per-site
/// "uninitialized" sentinel 0xFF).
#[inline]
pub(crate) fn class_tag_decode(tag: u8) -> Option<WorkloadClass> {
    match tag {
        TAG_LATENCY_BOUND => Some(WorkloadClass::LatencyBound),
        TAG_PORT_BOUND => Some(WorkloadClass::PortBound),
        TAG_MEMORY_BOUND => Some(WorkloadClass::MemoryBound),
        TAG_STREAMING => Some(WorkloadClass::Streaming),
        TAG_FINE_GRAIN => Some(WorkloadClass::FineGrain),
        _ => None,
    }
}

/// Bucket distance between two classes on the FineGrain ->
/// PortBound -> Streaming/MemoryBound -> LatencyBound progression.
/// Shared with the per-site classifier so its fast-adapt rule uses
/// exactly the same geometry as the process-global observer.
#[inline]
pub(crate) fn class_bucket_distance(a: WorkloadClass, b: WorkloadClass) -> u8 {
    bucket_distance(a, b)
}

/// Tag of the WorkloadClass that the LAST auto-classifier tick
/// produced. When this matches for [`AUTO_MIGRATION_HYSTERESIS`]
/// consecutive ticks AND differs from the active class, the
/// observer fires [`migrate_workload_class`].
static AUTO_PENDING_TAG: AtomicU8 = AtomicU8::new(TAG_PORT_BOUND);

/// Count of consecutive auto-classifier ticks that produced
/// [`AUTO_PENDING_TAG`]. Reset to 0 when the classifier output
/// differs from the prior pending tag.
static AUTO_PENDING_RUN: core::sync::atomic::AtomicU32 =
    core::sync::atomic::AtomicU32::new(0);

/// Last-seen cumulative LEAF_COUNT / LEAF_TIME_SUM_NS /
/// LEAF_TIME_SUMSQ snapshot, captured at the prior tick. The
/// next tick computes its observation window as `current - last`
/// without RESETTING the global counters -- that way tests and
/// other consumers (e.g. cv2 reporting) see the full cumulative
/// stats, while the classifier sees only the incremental window.
static AUTO_LAST_COUNT: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);
static AUTO_LAST_SUM_NS: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);
static AUTO_LAST_SUMSQ: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);

/// One tick of the closing-loop observer. Reads the global leaf
/// stats, computes the DELTA since the last tick, runs
/// [`classify_observed`] on the delta window, and migrates the
/// active workload class once the same observation has been
/// confirmed [`AUTO_MIGRATION_HYSTERESIS`] times in a row.
///
/// Does NOT reset the global stats -- tests and cv^2 readers see
/// the full cumulative counters. The "fresh window per tick"
/// behavior comes from comparing the current snapshot against the
/// AUTO_LAST_* snapshot stored on the prior tick.
#[inline]
pub fn tick_auto_classify() {
    use crate::sched::split_observer::snapshot_leaf_stats;
    let stats = snapshot_leaf_stats();
    let prev_count = AUTO_LAST_COUNT.load(Ordering::Relaxed);
    let prev_sum = AUTO_LAST_SUM_NS.load(Ordering::Relaxed);
    let prev_sumsq = AUTO_LAST_SUMSQ.load(Ordering::Relaxed);

    // Window delta. If stats wrapped or were reset externally,
    // delta computations may saturate to 0; in that case skip.
    let dcount = stats.count.saturating_sub(prev_count);
    if dcount < 4 {
        return;
    }
    let dsum = stats.sum_ns.saturating_sub(prev_sum);
    let dsumsq = stats.sumsq_scaled.saturating_sub(prev_sumsq);

    AUTO_LAST_COUNT.store(stats.count, Ordering::Relaxed);
    AUTO_LAST_SUM_NS.store(stats.sum_ns, Ordering::Relaxed);
    AUTO_LAST_SUMSQ.store(stats.sumsq_scaled, Ordering::Relaxed);

    let mean_ns = dsum / dcount;
    // cv^2 = variance / mean^2 on the delta window.
    let scaled_mean = (dsum >> 8) / dcount;
    let cv2 = if scaled_mean == 0 {
        0
    } else {
        let sumsq_per_n = dsumsq / dcount;
        let mean_sq = scaled_mean.saturating_mul(scaled_mean);
        let var = sumsq_per_n.saturating_sub(mean_sq);
        var.saturating_mul(1000) / mean_sq.max(1)
    };
    let observed = classify_observed(mean_ns, cv2);
    let observed_tag = workload_class_to_tag(observed);

    let active = active_workload_class();
    // Fast-adapt path: if the observed class is FAR from active
    // (bucket distance >= 2) AND the delta window has enough
    // samples (>= 64) to be statistically meaningful, migrate
    // immediately. The sample-count gate prevents single noisy
    // windows from triggering the migration.
    if active != observed
        && bucket_distance(active, observed) >= 2
        && dcount >= 64
    {
        migrate_workload_class(observed);
        AUTO_PENDING_TAG.store(observed_tag, Ordering::Relaxed);
        AUTO_PENDING_RUN.store(0, Ordering::Relaxed);
        return;
    }

    let pending_tag = AUTO_PENDING_TAG.load(Ordering::Relaxed);
    if pending_tag == observed_tag {
        let prior_run = AUTO_PENDING_RUN.fetch_add(1, Ordering::Relaxed);
        let new_run = prior_run.saturating_add(1);
        if new_run >= AUTO_MIGRATION_HYSTERESIS && active != observed {
            migrate_workload_class(observed);
        }
    } else {
        AUTO_PENDING_TAG.store(observed_tag, Ordering::Relaxed);
        AUTO_PENDING_RUN.store(1, Ordering::Relaxed);
    }
}

/// Reset the auto-classifier hysteresis state: AUTO_PENDING_TAG,
/// AUTO_PENDING_RUN, and the AUTO_LAST_* snapshot counters that
/// [`tick_auto_classify`] reads to compute the delta window. After
/// reset, the next tick treats the entire current global leaf-stats
/// counter as one fresh delta window.
///
/// Public for observer-driven tests and for callers that need to
/// reset the closing-loop state at a workload-phase boundary
/// (e.g., the application starts a new bench cell with a different
/// expected workload shape and wants the observer to converge from
/// scratch rather than smooth across the old phase). The function
/// is idempotent and cheap (5 Relaxed atomic stores).
pub fn reset_auto_classify_state() {
    AUTO_PENDING_TAG.store(TAG_PORT_BOUND, Ordering::Relaxed);
    AUTO_PENDING_RUN.store(0, Ordering::Relaxed);
    AUTO_LAST_COUNT.store(0, Ordering::Relaxed);
    AUTO_LAST_SUM_NS.store(0, Ordering::Relaxed);
    AUTO_LAST_SUMSQ.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_calibration_installs_clamped_values() {
        // Calibrate into a LOCAL threshold set: the probes and the
        // clamp/install logic are the code under test, and routing
        // for concurrently-running tests must not shift mid-suite.
        let local = ClassThresholds::new_defaults();
        let report = calibrate_class_thresholds_into(&local);
        assert!(
            (25..=200).contains(&report.fine_grain_ns),
            "fine_grain_ns {} outside clamp band",
            report.fine_grain_ns
        );
        assert!(
            (5_000..=200_000).contains(&report.trivial_reduce_cycles),
            "trivial_reduce_cycles {} outside clamp band",
            report.trivial_reduce_cycles
        );
        assert!(
            (1_000..=4_000).contains(&report.memory_latency_ns),
            "memory_latency_ns {} outside clamp band",
            report.memory_latency_ns
        );
        // The install target must reflect the reported values.
        assert_eq!(
            local.fine_grain_ns.load(Ordering::Relaxed),
            report.fine_grain_ns
        );
        assert_eq!(
            local.trivial_reduce_cycles.load(Ordering::Relaxed),
            report.trivial_reduce_cycles
        );
        assert_eq!(
            local.memory_latency_ns.load(Ordering::Relaxed),
            report.memory_latency_ns
        );
    }

    fn restore_default_profile() {
        // Reset to PortBound between tests so cross-test state
        // doesn't leak via the global ACTIVE_PROFILE_TAG.
        migrate_dispatch_profile(DispatchProfile::PortBound);
    }

    #[test]
    fn workload_class_maps_to_dispatch_profile() {
        assert_eq!(WorkloadClass::FineGrain.to_dispatch_profile(), DispatchProfile::PortBound);
        assert_eq!(WorkloadClass::PortBound.to_dispatch_profile(), DispatchProfile::PortBound);
        assert_eq!(WorkloadClass::LatencyBound.to_dispatch_profile(), DispatchProfile::LatencyBound);
        assert_eq!(WorkloadClass::MemoryBound.to_dispatch_profile(), DispatchProfile::MemoryBound);
        assert_eq!(WorkloadClass::Streaming.to_dispatch_profile(), DispatchProfile::Streaming);
    }

    #[test]
    fn migration_changes_active_profile() {
        let _guard = TestGuard::new();
        // Default per static init.
        let initial = active_dispatch_profile();
        assert!(matches!(initial, DispatchProfile::PortBound),
            "default expected PortBound, got {initial:?}");

        migrate_dispatch_profile(DispatchProfile::LatencyBound);
        assert_eq!(active_dispatch_profile(), DispatchProfile::LatencyBound);

        migrate_dispatch_profile(DispatchProfile::MemoryBound);
        assert_eq!(active_dispatch_profile(), DispatchProfile::MemoryBound);

        migrate_dispatch_profile(DispatchProfile::Streaming);
        assert_eq!(active_dispatch_profile(), DispatchProfile::Streaming);

        migrate_dispatch_profile(DispatchProfile::PortBound);
        assert_eq!(active_dispatch_profile(), DispatchProfile::PortBound);
    }

    #[test]
    fn classifier_picks_fine_grain_for_tiny_leaves() {
        assert_eq!(classify_observed(10, 0), WorkloadClass::FineGrain);
        assert_eq!(classify_observed(49, 1000), WorkloadClass::FineGrain);
    }

    #[test]
    fn classifier_picks_port_bound_for_medium_leaves() {
        assert_eq!(classify_observed(100, 10), WorkloadClass::PortBound);
        assert_eq!(classify_observed(499, 5000), WorkloadClass::PortBound);
    }

    #[test]
    fn classifier_picks_streaming_for_heavy_uniform_leaves() {
        assert_eq!(classify_observed(500, 0), WorkloadClass::Streaming);
        assert_eq!(classify_observed(10_000, 49), WorkloadClass::Streaming);
    }

    #[test]
    fn classifier_picks_memory_bound_for_heavy_moderate_variance() {
        assert_eq!(classify_observed(500, 100), WorkloadClass::MemoryBound);
        assert_eq!(classify_observed(2000, 499), WorkloadClass::MemoryBound);
    }

    #[test]
    fn classifier_picks_latency_bound_for_heavy_high_variance() {
        assert_eq!(classify_observed(500, 500), WorkloadClass::LatencyBound);
        assert_eq!(classify_observed(5000, 10_000), WorkloadClass::LatencyBound);
    }

    // -- Static initial classifier (infer_class_static) -----------

    #[test]
    fn static_classifier_picks_streaming_for_big_byte_scan() {
        // grep-style: ns_hint=1, batch_size=16 MiB
        assert_eq!(
            infer_class_static(6, 16 * 1024 * 1024, Some(1)),
            WorkloadClass::Streaming,
        );
    }

    #[test]
    fn static_classifier_picks_streaming_for_big_image() {
        // rgb_to_gray: ns_hint=10, batch_size=4 M pixels
        assert_eq!(
            infer_class_static(6, 2048 * 2048, Some(10)),
            WorkloadClass::Streaming,
        );
    }

    #[test]
    fn static_classifier_picks_port_bound_for_medium_per_item() {
        // kmeans: ns_hint=100, batch_size=100K
        assert_eq!(
            infer_class_static(6, 100 * 1024, Some(100)),
            WorkloadClass::PortBound,
        );
    }

    #[test]
    fn static_classifier_picks_memory_bound_for_heavy_per_item() {
        // nbody-style: ns_hint=600 (per-particle force computation)
        assert_eq!(
            infer_class_static(6, 1024, Some(600)),
            WorkloadClass::MemoryBound,
        );
    }

    #[test]
    fn static_classifier_picks_fine_grain_for_tiny_total() {
        // Tiny workload below dispatch breakeven
        assert_eq!(
            infer_class_static(6, 100, Some(10)),
            WorkloadClass::FineGrain,
        );
    }

    #[test]
    fn static_classifier_picks_latency_for_high_precision_tier() {
        // No hint, K_outer=12 (high precision)
        assert_eq!(
            infer_class_static(12, 1024, None),
            WorkloadClass::LatencyBound,
        );
    }

    #[test]
    fn static_classifier_picks_streaming_for_low_tier_large_batch() {
        // No hint, K=2 (low tier), big batch
        assert_eq!(
            infer_class_static(2, 100_000, None),
            WorkloadClass::Streaming,
        );
    }

    #[test]
    fn static_classifier_falls_back_to_port_bound_when_unsure() {
        // No hint, medium tier, medium batch -> safe middle pick
        assert_eq!(
            infer_class_static(6, 1024, None),
            WorkloadClass::PortBound,
        );
    }

    #[test]
    fn auto_classify_migrates_after_hysteresis() {
        use crate::sched::split_observer::{
            acquire_test_lock, record_leaf_batch, reset_leaf_stats,
        };
        let _stats_lock = acquire_test_lock();
        let _guard = TestGuard::new();
        reset_leaf_stats();
        reset_auto_classify_state();
        // Start with PortBound active so a Streaming classification
        // counts as a disagreement that needs to migrate.
        migrate_dispatch_profile(DispatchProfile::PortBound);
        assert_eq!(active_workload_class(), WorkloadClass::PortBound);

        // Inject a batch matching the Streaming signature
        // (mean_ns >= 500, cv2 < 50). Use 64 samples of 1000 ns
        // each: sum_ns = 64 * 1000 = 64_000; sumsq_scaled =
        // (1000 >> 8)^2 * 64 = 3^2 * 64 = 576.
        // Mean = 1000, variance ~= 0 -> cv2 = 0.
        let sample_ns: u64 = 1000;
        let count: u64 = 64;
        let scaled = sample_ns >> 8;
        let sumsq = scaled.saturating_mul(scaled).saturating_mul(count);
        for _ in 0..AUTO_MIGRATION_HYSTERESIS {
            record_leaf_batch(sample_ns * count, sumsq, count);
        }
        // After HYSTERESIS consecutive Streaming classifications,
        // active class should have migrated.
        assert_eq!(
            active_workload_class(),
            WorkloadClass::Streaming,
            "auto-classifier did not migrate after {} consecutive Streaming observations",
            AUTO_MIGRATION_HYSTERESIS,
        );
    }

    #[test]
    fn auto_classify_does_not_migrate_on_a_single_outlier() {
        use crate::sched::split_observer::{
            acquire_test_lock, record_leaf_batch, reset_leaf_stats,
        };
        let _stats_lock = acquire_test_lock();
        let _guard = TestGuard::new();
        reset_leaf_stats();
        reset_auto_classify_state();
        migrate_dispatch_profile(DispatchProfile::PortBound);

        // One streaming batch is insufficient to migrate.
        let sample_ns: u64 = 1000;
        let count: u64 = 64;
        let scaled = sample_ns >> 8;
        let sumsq = scaled.saturating_mul(scaled).saturating_mul(count);
        record_leaf_batch(sample_ns * count, sumsq, count);
        assert_eq!(
            active_workload_class(),
            WorkloadClass::PortBound,
            "auto-classifier migrated on a single observation; hysteresis broken",
        );
    }

    #[test]
    fn workload_class_migration_propagates() {
        let _guard = TestGuard::new();
        migrate_workload_class(WorkloadClass::LatencyBound);
        assert_eq!(active_dispatch_profile(), DispatchProfile::LatencyBound);
        assert_eq!(active_workload_class(), WorkloadClass::LatencyBound);

        migrate_workload_class(WorkloadClass::MemoryBound);
        assert_eq!(active_dispatch_profile(), DispatchProfile::MemoryBound);
        assert_eq!(active_workload_class(), WorkloadClass::MemoryBound);
    }

    // RAII guard holding the global-profile test lock and restoring
    // the default profile on both ends, so a test that migrates the
    // process-wide profile neither races a test that reads it nor
    // leaks its last value.
    struct TestGuard {
        lock: std::sync::MutexGuard<'static, ()>,
    }
    impl TestGuard {
        fn new() -> Self {
            let lock = super::global_profile_test_lock();
            restore_default_profile();
            Self { lock }
        }
    }
    impl Drop for TestGuard {
        fn drop(&mut self) {
            // Restore while the lock is still held, then release it.
            restore_default_profile();
            let _still_held = &self.lock;
        }
    }
}

/// Serializes tests that migrate or depend on the process-wide
/// dispatch profile; poison-tolerant so one failing test does not
/// cascade.
#[cfg(test)]
pub(crate) fn global_profile_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

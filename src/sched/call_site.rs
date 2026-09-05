//! Per-call-site adaptive state, keyed by the caller's source
//! location.
//!
//! Every dispatch entry is `#[track_caller]` and maps
//! `std::panic::Location::caller()` to a `&'static CallSiteState`
//! via [`site_for_location`]: read-mostly `RwLock<HashMap>` fronted
//! by a per-thread one-slot cache, states `Box::leak`ed for process
//! lifetime. A `static` in a generic fn cannot provide this
//! identity (statics never monomorphize; every caller would share
//! one pool). `#[track_caller]` chains through wrapper entries, so
//! delegating helpers resolve to the outermost user call site.
//! Driver loops that funnel many workloads through one textual site
//! share one state; callers can pin their own via
//! [`crate::sched::JobPlan::with_site`].
//!
//! A site holds: a learned
//! [`crate::sched::adaptive_profile::WorkloadClass`] (delta-window
//! classifier, hysteresis 2 adjacent, fast-adapt at bucket
//! distance 2+ with 64+ samples), leaf-time statistics for
//! [`crate::sched::JobPlan::effective_use_smt`], policy-arm A/B
//! EWMAs for the heartbeat-vs-SLAW gate, and placement EWMAs
//! (per log2-size-bucket CPU vs backend wall times) for
//! [`crate::sched::hybrid::hybrid_auto`].
//!
//! Leaves feed both the process-global counters (cold-start prior
//! for site-less plans) and the site's own statistics. Serial-span
//! samples (heartbeat / token-bucket fillers) are site-only:
//! whole-span wall times would poison the global per-item-ns
//! boundaries.

#![allow(clippy::missing_errors_doc)]

use core::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

use crate::sched::adaptive_profile::{
    WorkloadClass, class_tag_decode, class_tag_encode, classify_observed,
};

/// Sentinel tag meaning "this site has not been classified yet".
const TAG_UNINIT: u8 = 0xFF;

/// Leaves between site classifier ticks. Matches the process-global
/// [`crate::sched::split_observer::AUTO_CLASSIFY_QUANTUM`] cadence so
/// per-site convergence speed is the same as the global observer's.
const SITE_CLASSIFY_QUANTUM: u64 = 16;

/// Consecutive agreeing ticks required before an adjacent-bucket
/// migration fires (same value as the global observer's hysteresis).
const SITE_MIGRATION_HYSTERESIS: u32 = 2;

/// Policy-arm trial cadence: every Nth arm selection returns the
/// non-preferred arm so its EWMA stays fresh enough to detect drift.
const ARM_TRIAL_CADENCE: u32 = 16;

/// Minimum samples per arm before the EWMA comparison is trusted.
const ARM_MIN_SAMPLES: u32 = 3;

/// Placement re-probe cadence: every Nth call in a warm size bucket
/// runs BOTH sides again so the model tracks drift (thermal
/// throttling, contention) instead of freezing on stale data.
const PLACEMENT_REPROBE_CADENCE: u32 = 32;

/// Number of log2-size buckets for the placement EWMAs. Bucket i
/// covers batch sizes in `[2^i, 2^(i+1))`; 40 buckets cover every
/// `u32` batch size and then some.
pub const PLACEMENT_BUCKETS: usize = 40;

/// EWMA update with alpha = 1/8: `new = old - old/8 + sample/8`.
/// Zero is the "empty" sentinel, so the first sample seeds directly.
/// Load/store (not CAS) is deliberate: concurrent updates may drop a
/// sample, which is acceptable for a smoothed statistic and keeps
/// the hot path at two relaxed atomics.
#[inline]
fn ewma_update(cell: &AtomicU64, sample_ns: u64) {
    let old = cell.load(Ordering::Relaxed);
    let new = if old == 0 {
        sample_ns.max(1)
    } else {
        (old - old / 8).saturating_add(sample_ns / 8).max(1)
    };
    cell.store(new, Ordering::Relaxed);
}

/// Which execution policy a site's A/B state currently prefers.
/// Arm meanings are defined by the consuming dispatch site (for the
/// heartbeat gate: arm 0 = SLAW bisect, arm 1 = heartbeat).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyArm {
    /// The default execution policy for the consuming site.
    Default,
    /// The alternative execution policy for the consuming site.
    Alternative,
}

impl PolicyArm {
    #[inline]
    fn idx(self) -> usize {
        match self {
            PolicyArm::Default => 0,
            PolicyArm::Alternative => 1,
        }
    }
}

/// Placement decision produced by the hybrid-dispatch model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Run only the CPU-side implementation.
    Cpu,
    /// Run only the backend-side implementation.
    Backend,
    /// Run both implementations concurrently and time each: the
    /// exploration/calibration mode (cold bucket or scheduled
    /// re-probe).
    Race,
}

/// Per-call-site adaptive state. Resolved automatically per caller
/// source location via [`caller_site`] / [`site_for_location`], or
/// declared as a caller-owned `static` and attached via
/// [`crate::sched::JobPlan::with_site`].
///
/// All fields are atomics with a `const fn new`, so the type is
/// directly usable in statics with zero lazy-init cost.
pub struct CallSiteState {
    // Classifier: learned class tag + adjacent-bucket hysteresis.
    active_tag: AtomicU8,
    pending_tag: AtomicU8,
    pending_run: AtomicU32,
    // Cumulative leaf statistics for this site.
    leaf_count: AtomicU64,
    leaf_sum_ns: AtomicU64,
    leaf_sumsq_scaled: AtomicU64,
    // Snapshot of the cumulative counters at the previous classifier
    // tick; each tick classifies the delta window since then.
    last_count: AtomicU64,
    last_sum_ns: AtomicU64,
    last_sumsq: AtomicU64,
    // Execution-policy A/B arms: per-arm EWMA wall time + sample
    // counts + a call counter driving the trial cadence.
    arm_ewma_ns: [AtomicU64; 2],
    arm_samples: [AtomicU32; 2],
    arm_calls: AtomicU32,
    // Hybrid-placement model: per-log2-size-bucket end-to-end EWMAs
    // for the CPU side and the backend side, plus a per-bucket call
    // counter driving the re-probe cadence.
    place_cpu_ns: [AtomicU64; PLACEMENT_BUCKETS],
    place_backend_ns: [AtomicU64; PLACEMENT_BUCKETS],
    place_calls: [AtomicU32; PLACEMENT_BUCKETS],
    // Split-throughput model: learned per-item cost on each side for
    // proportional slice splitting, site-wide and per log2-size
    // bucket (a batch's per-item cost changes with its size on both
    // sides, so the share is learned per bucket and falls back to
    // the site-wide value while a bucket is cold).
    split_cpu_ns_per_item: AtomicU64,
    split_backend_ns_per_item: AtomicU64,
    split_cpu_ns_per_item_by_size: [AtomicU64; PLACEMENT_BUCKETS],
    split_backend_ns_per_item_by_size: [AtomicU64; PLACEMENT_BUCKETS],
    /// Reduce-merge cost observer for
    /// [`crate::sched::par_iter::reduce_chunks`]: TSC-cycle sum and
    /// sample count of timed `reduce(a, b)` calls at THIS call
    /// site, so a cheap element-wise merge and a heavy
    /// collection-merge reducer each converge on their own average.
    reduce_cost_sum_cycles: AtomicU64,
    reduce_cost_samples: AtomicU32,
}

impl CallSiteState {
    /// Fresh, unclassified site. Usable in `static` position.
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Self {
        // Inline-const array elements: atomics are not Copy, so the
        // repeat expressions need per-element const evaluation.
        Self {
            active_tag: AtomicU8::new(TAG_UNINIT),
            pending_tag: AtomicU8::new(TAG_UNINIT),
            pending_run: AtomicU32::new(0),
            leaf_count: AtomicU64::new(0),
            leaf_sum_ns: AtomicU64::new(0),
            leaf_sumsq_scaled: AtomicU64::new(0),
            last_count: AtomicU64::new(0),
            last_sum_ns: AtomicU64::new(0),
            last_sumsq: AtomicU64::new(0),
            arm_ewma_ns: [const { AtomicU64::new(0) }; 2],
            arm_samples: [const { AtomicU32::new(0) }; 2],
            arm_calls: AtomicU32::new(0),
            place_cpu_ns: [const { AtomicU64::new(0) }; PLACEMENT_BUCKETS],
            place_backend_ns: [const { AtomicU64::new(0) }; PLACEMENT_BUCKETS],
            place_calls: [const { AtomicU32::new(0) }; PLACEMENT_BUCKETS],
            split_cpu_ns_per_item: AtomicU64::new(0),
            split_backend_ns_per_item: AtomicU64::new(0),
            split_cpu_ns_per_item_by_size: [const { AtomicU64::new(0) }; PLACEMENT_BUCKETS],
            split_backend_ns_per_item_by_size: [const { AtomicU64::new(0) }; PLACEMENT_BUCKETS],
            reduce_cost_sum_cycles: AtomicU64::new(0),
            reduce_cost_samples: AtomicU32::new(0),
        }
    }

    // -----------------------------------------------------------------
    // Classifier surface
    // -----------------------------------------------------------------

    /// The class this site has learned, or `None` while the site has
    /// not accumulated enough evidence to classify (fresh site, or
    /// fewer than 4 leaves in every delta window so far).
    #[inline]
    pub fn learned_class(&self) -> Option<WorkloadClass> {
        class_tag_decode(self.active_tag.load(Ordering::Acquire))
    }

    /// Record a batch of leaf-time samples for this site. Dual-writes
    /// to the process-global counters (keeping the global prior and
    /// the split-multiplier observer fed) and ticks the site's own
    /// classifier when the batch crosses the site quantum.
    pub fn record_batch(&'static self, sum_ns: u64, sumsq_scaled: u64, count: u64) {
        // Global dual-write first: preserves every existing consumer
        // of the process-wide stats (split multiplier, global
        // auto-classify, effective_use_smt fallback, tests).
        crate::sched::split_observer::record_leaf_batch(sum_ns, sumsq_scaled, count);
        self.record_batch_site_only(sum_ns, sumsq_scaled, count);
    }

    /// [`Self::record_batch`] without the process-global dual-write.
    /// For samples that are meaningful to THIS site's statistics but
    /// would poison the global classifier: heartbeat serial-span
    /// durations are whole-span wall times (tens of microseconds and
    /// up), not per-leaf costs, and feeding them to the global
    /// per-item-ns boundaries would migrate the process profile off
    /// unrelated workloads.
    pub(crate) fn record_batch_site_only(
        &'static self,
        sum_ns: u64,
        sumsq_scaled: u64,
        count: u64,
    ) {
        self.leaf_sum_ns.fetch_add(sum_ns, Ordering::Relaxed);
        self.leaf_sumsq_scaled.fetch_add(sumsq_scaled, Ordering::Relaxed);
        let prior = self.leaf_count.fetch_add(count, Ordering::Relaxed);
        let new_total = prior.wrapping_add(count);
        if (prior / SITE_CLASSIFY_QUANTUM) != (new_total / SITE_CLASSIFY_QUANTUM) {
            self.tick();
        }
    }

    /// Coefficient-of-variation squared (parts-per-1000) over this
    /// site's cumulative leaf history, or `None` below 4 samples.
    /// Same fixed-point convention as the global
    /// [`crate::sched::split_observer::leaf_cv_squared_per_mille`].
    pub fn cv2_per_mille(&self) -> Option<u64> {
        let n = self.leaf_count.load(Ordering::Relaxed);
        if n < 4 {
            return None;
        }
        let sum = self.leaf_sum_ns.load(Ordering::Relaxed);
        let sumsq = self.leaf_sumsq_scaled.load(Ordering::Relaxed);
        let mean_scaled = (sum >> 8) / n;
        if mean_scaled == 0 {
            return Some(0);
        }
        let sumsq_per_n = sumsq / n;
        let mean_sq = mean_scaled.saturating_mul(mean_scaled);
        let var = sumsq_per_n.saturating_sub(mean_sq);
        Some(var.saturating_mul(1000) / mean_sq.max(1))
    }

    /// Total leaves recorded against this site.
    #[inline]
    pub fn leaf_count(&self) -> u64 {
        self.leaf_count.load(Ordering::Relaxed)
    }

    /// One classifier tick over the delta window since the previous
    /// tick. Same algorithm as the process-global
    /// `tick_auto_classify`: hysteresis [`SITE_MIGRATION_HYSTERESIS`]
    /// for adjacent-bucket moves, immediate migration when the
    /// observation sits at bucket distance >= 2 with >= 64 samples.
    fn tick(&self) {
        let count = self.leaf_count.load(Ordering::Relaxed);
        let sum = self.leaf_sum_ns.load(Ordering::Relaxed);
        let sumsq = self.leaf_sumsq_scaled.load(Ordering::Relaxed);

        let prev_count = self.last_count.load(Ordering::Relaxed);
        let dcount = count.saturating_sub(prev_count);
        if dcount < 4 {
            return;
        }
        let dsum = sum.saturating_sub(self.last_sum_ns.load(Ordering::Relaxed));
        let dsumsq = sumsq.saturating_sub(self.last_sumsq.load(Ordering::Relaxed));
        self.last_count.store(count, Ordering::Relaxed);
        self.last_sum_ns.store(sum, Ordering::Relaxed);
        self.last_sumsq.store(sumsq, Ordering::Relaxed);

        let mean_ns = dsum / dcount;
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
        let observed_tag = class_tag_encode(observed);

        let active = class_tag_decode(self.active_tag.load(Ordering::Relaxed));
        match active {
            None => {
                // First classification of a fresh site: adopt
                // immediately; the static classifier's plan-level
                // guess governed calls up to this point.
                self.active_tag.store(observed_tag, Ordering::Release);
                self.pending_tag.store(observed_tag, Ordering::Relaxed);
                self.pending_run.store(0, Ordering::Relaxed);
            }
            Some(active_class) if active_class != observed => {
                if crate::sched::adaptive_profile::class_bucket_distance(
                    active_class, observed,
                ) >= 2
                    && dcount >= 64
                {
                    self.active_tag.store(observed_tag, Ordering::Release);
                    self.pending_tag.store(observed_tag, Ordering::Relaxed);
                    self.pending_run.store(0, Ordering::Relaxed);
                    return;
                }
                let pending = self.pending_tag.load(Ordering::Relaxed);
                if pending == observed_tag {
                    let run = self
                        .pending_run
                        .fetch_add(1, Ordering::Relaxed)
                        .saturating_add(1);
                    if run >= SITE_MIGRATION_HYSTERESIS {
                        self.active_tag.store(observed_tag, Ordering::Release);
                    }
                } else {
                    self.pending_tag.store(observed_tag, Ordering::Relaxed);
                    self.pending_run.store(1, Ordering::Relaxed);
                }
            }
            Some(_) => {
                // Observation agrees with the active class; reset any
                // stale pending streak toward a different class.
                self.pending_tag.store(observed_tag, Ordering::Relaxed);
                self.pending_run.store(0, Ordering::Relaxed);
            }
        }
    }

    // -----------------------------------------------------------------
    // Policy-arm A/B surface
    // -----------------------------------------------------------------

    /// Pick a policy arm for this dispatch. `alternative_allowed`
    /// gates the alternative arm behind the consuming site's
    /// precondition (e.g. the heartbeat gate requires high cv^2);
    /// when it is false the default arm is returned unconditionally
    /// and no trial fires.
    ///
    /// Selection order when the alternative is allowed:
    /// 1. Either arm below [`ARM_MIN_SAMPLES`]: pick the
    ///    lesser-sampled arm (bounded exploration).
    /// 2. Every [`ARM_TRIAL_CADENCE`]th call: pick the arm the EWMA
    ///    comparison does NOT prefer (drift detection).
    /// 3. Otherwise: the arm with the lower EWMA wall time.
    pub fn choose_arm(&self, alternative_allowed: bool) -> PolicyArm {
        if !alternative_allowed {
            return PolicyArm::Default;
        }
        let calls = self.arm_calls.fetch_add(1, Ordering::Relaxed);
        let s0 = self.arm_samples[0].load(Ordering::Relaxed);
        let s1 = self.arm_samples[1].load(Ordering::Relaxed);
        if s0 < ARM_MIN_SAMPLES || s1 < ARM_MIN_SAMPLES {
            return if s1 < s0 {
                PolicyArm::Alternative
            } else {
                PolicyArm::Default
            };
        }
        let e0 = self.arm_ewma_ns[0].load(Ordering::Relaxed);
        let e1 = self.arm_ewma_ns[1].load(Ordering::Relaxed);
        let best = if e1 < e0 {
            PolicyArm::Alternative
        } else {
            PolicyArm::Default
        };
        if calls % ARM_TRIAL_CADENCE == ARM_TRIAL_CADENCE - 1 {
            // Trial tick: run the non-preferred arm.
            return match best {
                PolicyArm::Default => PolicyArm::Alternative,
                PolicyArm::Alternative => PolicyArm::Default,
            };
        }
        best
    }

    /// Record one dispatch's wall time under `arm`.
    pub fn record_arm(&self, arm: PolicyArm, wall_ns: u64) {
        let i = arm.idx();
        ewma_update(&self.arm_ewma_ns[i], wall_ns);
        self.arm_samples[i].fetch_add(1, Ordering::Relaxed);
    }

    /// Current per-arm EWMA wall times `(default_ns, alternative_ns)`;
    /// zero means "no samples yet". Diagnostics + tests.
    pub fn arm_ewmas(&self) -> (u64, u64) {
        (
            self.arm_ewma_ns[0].load(Ordering::Relaxed),
            self.arm_ewma_ns[1].load(Ordering::Relaxed),
        )
    }

    // -----------------------------------------------------------------
    // Hybrid placement surface
    // -----------------------------------------------------------------

    /// Log2 size bucket for a batch size.
    #[inline]
    fn bucket(batch: u32) -> usize {
        (63 - (batch.max(1) as u64).leading_zeros() as usize).min(PLACEMENT_BUCKETS - 1)
    }

    /// Placement decision for a dispatch of `batch` items: `Race`
    /// while the bucket is cold (either side unmeasured) and on every
    /// [`PLACEMENT_REPROBE_CADENCE`]th call, otherwise whichever side
    /// has the lower end-to-end EWMA.
    pub fn choose_placement(&self, batch: u32) -> Placement {
        let b = Self::bucket(batch);
        let calls = self.place_calls[b].fetch_add(1, Ordering::Relaxed);
        let cpu = self.place_cpu_ns[b].load(Ordering::Relaxed);
        let dev = self.place_backend_ns[b].load(Ordering::Relaxed);
        if cpu == 0 || dev == 0 {
            return Placement::Race;
        }
        if calls % PLACEMENT_REPROBE_CADENCE == PLACEMENT_REPROBE_CADENCE - 1 {
            return Placement::Race;
        }
        if cpu <= dev { Placement::Cpu } else { Placement::Backend }
    }

    /// Record measured wall times for a dispatch of `batch` items.
    /// Either side may be `None` when only one side ran.
    pub fn record_placement(
        &self,
        batch: u32,
        cpu_ns: Option<u64>,
        backend_ns: Option<u64>,
    ) {
        let b = Self::bucket(batch);
        if let Some(ns) = cpu_ns {
            ewma_update(&self.place_cpu_ns[b], ns);
        }
        if let Some(ns) = backend_ns {
            ewma_update(&self.place_backend_ns[b], ns);
        }
    }

    /// Current placement EWMAs `(cpu_ns, backend_ns)` for the bucket
    /// covering `batch`; zero means unmeasured. Diagnostics + tests.
    pub fn placement_ewmas(&self, batch: u32) -> (u64, u64) {
        let b = Self::bucket(batch);
        (
            self.place_cpu_ns[b].load(Ordering::Relaxed),
            self.place_backend_ns[b].load(Ordering::Relaxed),
        )
    }

    /// CPU share of a divisible workload per the learned per-item
    /// throughputs, in parts-per-1000. 500 (an even split) until both
    /// sides have measurements. CPU share = backend_ns_per_item /
    /// (cpu_ns_per_item + backend_ns_per_item): the faster side gets
    /// the larger share.
    pub fn split_cpu_share_per_mille(&self) -> u32 {
        let c = self.split_cpu_ns_per_item.load(Ordering::Relaxed);
        let g = self.split_backend_ns_per_item.load(Ordering::Relaxed);
        if c == 0 || g == 0 {
            return 500;
        }
        let total = c.saturating_add(g).max(1);
        ((g.saturating_mul(1000)) / total).clamp(50, 950) as u32
    }

    /// Record per-item throughput observations from a split dispatch.
    pub fn record_split(
        &self,
        cpu_items: usize,
        cpu_ns: u64,
        backend_items: usize,
        backend_ns: u64,
    ) {
        if cpu_items > 0 {
            ewma_update(
                &self.split_cpu_ns_per_item,
                (cpu_ns / cpu_items as u64).max(1),
            );
        }
        if backend_items > 0 {
            ewma_update(
                &self.split_backend_ns_per_item,
                (backend_ns / backend_items as u64).max(1),
            );
        }
    }

    /// [`Self::split_cpu_share_per_mille`] for a dispatch of `n`
    /// items: the bucket covering `n` when it has measurements on
    /// both sides, the site-wide model otherwise.
    pub fn split_cpu_share_per_mille_for(&self, n: u32) -> u32 {
        let b = Self::bucket(n);
        let c = self.split_cpu_ns_per_item_by_size[b].load(Ordering::Relaxed);
        let g = self.split_backend_ns_per_item_by_size[b].load(Ordering::Relaxed);
        if c == 0 || g == 0 {
            return self.split_cpu_share_per_mille();
        }
        let total = c.saturating_add(g).max(1);
        ((g.saturating_mul(1000)) / total).clamp(50, 950) as u32
    }

    /// [`Self::record_split`] for a dispatch of `n` items: updates the
    /// bucket covering `n` and the site-wide model.
    pub fn record_split_for(
        &self,
        n: u32,
        cpu_items: usize,
        cpu_ns: u64,
        backend_items: usize,
        backend_ns: u64,
    ) {
        let b = Self::bucket(n);
        if cpu_items > 0 {
            ewma_update(
                &self.split_cpu_ns_per_item_by_size[b],
                (cpu_ns / cpu_items as u64).max(1),
            );
        }
        if backend_items > 0 {
            ewma_update(
                &self.split_backend_ns_per_item_by_size[b],
                (backend_ns / backend_items as u64).max(1),
            );
        }
        self.record_split(cpu_items, cpu_ns, backend_items, backend_ns);
    }

    /// Whether the reduce-cost observer still wants a calibration
    /// sample (bounded at 16; after convergence the caller skips
    /// the timed merge entirely).
    pub fn reduce_cost_wants_sample(&self) -> bool {
        self.reduce_cost_samples.load(Ordering::Relaxed) < REDUCE_COST_MAX_SAMPLES
    }

    /// Record one timed `reduce(a, b)` merge in TSC cycles.
    pub fn record_reduce_cost_sample(&self, cycles: u64) {
        self.reduce_cost_sum_cycles.fetch_add(cycles, Ordering::Relaxed);
        self.reduce_cost_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Average observed reduce-merge cost in TSC cycles, or `None`
    /// below 4 samples (cold: the caller defaults to the
    /// always-correct bisect path).
    pub fn reduce_cost_avg_cycles(&self) -> Option<u64> {
        let n = self.reduce_cost_samples.load(Ordering::Relaxed);
        if n < REDUCE_COST_MIN_SAMPLES {
            return None;
        }
        Some(self.reduce_cost_sum_cycles.load(Ordering::Relaxed) / n as u64)
    }
}

/// Reduce-cost observer bounds: sample until 16 merges are timed,
/// trust the average from 4.
const REDUCE_COST_MAX_SAMPLES: u32 = 16;
const REDUCE_COST_MIN_SAMPLES: u32 = 4;

impl core::fmt::Debug for CallSiteState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CallSiteState")
            .field("learned_class", &self.learned_class())
            .field("leaf_count", &self.leaf_count())
            .field("cv2_per_mille", &self.cv2_per_mille())
            .finish()
    }
}

/// Copyable handle to a `'static` [`CallSiteState`], with
/// pointer-identity equality/hash so [`crate::sched::JobPlan`] keeps
/// its `PartialEq / Eq / Hash` derives (the state's atomics have no
/// value equality; two sites are "equal" only when they are the same
/// static).
#[derive(Clone, Copy)]
pub struct SiteRef(&'static CallSiteState);

impl SiteRef {
    /// Wrap a `'static` site.
    #[inline]
    pub const fn new(site: &'static CallSiteState) -> Self {
        Self(site)
    }

    /// Access the underlying state.
    #[inline]
    pub fn get(self) -> &'static CallSiteState {
        self.0
    }
}

impl PartialEq for SiteRef {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.0, other.0)
    }
}
impl Eq for SiteRef {}
impl core::hash::Hash for SiteRef {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        (self.0 as *const CallSiteState as usize).hash(state);
    }
}
impl core::fmt::Debug for SiteRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "SiteRef({:p})", self.0 as *const CallSiteState)
    }
}

/// Location-keyed site registry. Value-keyed on (file, line,
/// column) rather than the `&'static Location` address: the same
/// textual call site inside a generic caller can surface as
/// distinct `Location` constants per instantiation, and the value
/// key merges those back into one site.
static SITE_REGISTRY: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashMap<u64, &'static CallSiteState>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));

fn location_key(loc: &'static std::panic::Location<'static>) -> u64 {
    use core::hash::{Hash, Hasher};
    let mut h = std::hash::DefaultHasher::new();
    loc.file().hash(&mut h);
    loc.line().hash(&mut h);
    loc.column().hash(&mut h);
    h.finish()
}

/// The site handle for the CALLER's source location. Every
/// dispatch entry uses this (via `#[track_caller]` chaining) to
/// attach automatic per-call-site identity; user code normally
/// never calls it directly, but it is the way to read back what a
/// specific call site learned without switching that site to an
/// explicit [`crate::sched::JobPlan::with_site`] attachment.
#[track_caller]
#[inline]
pub fn caller_site() -> SiteRef {
    site_for_location(std::panic::Location::caller())
}

/// Resolve a source location to its `'static` site, allocating the
/// state on first sight of the location.
///
/// Fast path: a per-thread one-slot cache keyed on the `Location`
/// address (two thread-local loads). Miss path: value-hash of
/// (file, line, column) into the read-mostly registry; the write
/// lock is taken only the first time a location is seen
/// process-wide.
pub fn site_for_location(loc: &'static std::panic::Location<'static>) -> SiteRef {
    thread_local! {
        static LAST: core::cell::Cell<(usize, usize)> = const { core::cell::Cell::new((0, 0)) };
    }
    let loc_addr = loc as *const _ as usize;
    let cached = LAST.with(|c| c.get());
    if cached.0 == loc_addr && cached.1 != 0 {
        // SAFETY: the cache only ever stores pointers to
        // `Box::leak`ed `CallSiteState` values inserted below, so
        // the referent is 'static and valid.
        return SiteRef::new(unsafe { &*(cached.1 as *const CallSiteState) });
    }
    let key = location_key(loc);
    let existing = SITE_REGISTRY
        .read()
        .ok()
        .and_then(|map| map.get(&key).copied());
    let site: &'static CallSiteState = match existing {
        Some(site) => site,
        None => match SITE_REGISTRY.write() {
            Ok(mut map) => map
                .entry(key)
                .or_insert_with(|| Box::leak(Box::new(CallSiteState::new()))),
            // Lock poisoned (a panic while inserting): fall back to
            // a leaked one-off state so dispatch keeps working; the
            // site just will not be shared with future calls.
            Err(_) => Box::leak(Box::new(CallSiteState::new())),
        },
    };
    LAST.with(|c| c.set((loc_addr, site as *const CallSiteState as usize)));
    SiteRef::new(site)
}

/// Number of distinct call sites the registry has materialised.
#[cfg(test)]
pub(crate) fn registry_len() -> usize {
    SITE_REGISTRY.read().map(|m| m.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_site_is_distinct_per_call_site_and_stable_per_site() {
        // Two textually distinct calls resolve to two different
        // states; repeating a call site resolves to the same state.
        // This is the regression guard for the identity mechanism:
        // a static inside a generic fn is SHARED across
        // monomorphizations, so identity must come from
        // track_caller locations, never from entry-local statics.
        let a = caller_site();
        let b = caller_site();
        assert_ne!(a, b, "distinct call sites must get distinct states");
        let mut repeats = Vec::new();
        for _ in 0..3 {
            repeats.push(caller_site());
        }
        assert_eq!(repeats[0], repeats[1]);
        assert_eq!(repeats[1], repeats[2]);
        assert_ne!(repeats[0], a);
        assert_ne!(repeats[0], b);
        assert!(registry_len() >= 3, "registry materialises one state per site");
    }

    #[test]
    fn fresh_site_is_unclassified() {
        static S: CallSiteState = CallSiteState::new();
        assert_eq!(S.learned_class(), None);
        assert_eq!(S.cv2_per_mille(), None);
        assert_eq!(S.leaf_count(), 0);
    }

    #[test]
    fn site_classifies_uniform_streaming_leaves() {
        static S: CallSiteState = CallSiteState::new();
        // 64 uniform heavy leaves (1ms each, zero variance) crosses
        // several quanta; mean >= 500ns + cv2 < low gives Streaming
        // per classify_observed.
        for _ in 0..4 {
            let per = 1_000_000u64;
            let scaled = per >> 8;
            // site_only: synthetic test samples must not leak into
            // the process-global stats other suite tests observe.
            S.record_batch_site_only(per * 16, scaled * scaled * 16, 16);
        }
        assert_eq!(S.learned_class(), Some(WorkloadClass::Streaming));
    }

    #[test]
    fn two_sites_classify_independently() {
        static LIGHT: CallSiteState = CallSiteState::new();
        static HEAVY: CallSiteState = CallSiteState::new();
        // Interleave the two shapes; each site must converge to its
        // own class with zero cross-talk between the sites.
        for _ in 0..4 {
            let l = 20u64; // 20ns leaves: FineGrain
            LIGHT.record_batch_site_only(l * 16, 0, 16);
            let h = 1_000_000u64; // 1ms uniform: Streaming
            let hs = h >> 8;
            HEAVY.record_batch_site_only(h * 16, hs * hs * 16, 16);
        }
        assert_eq!(LIGHT.learned_class(), Some(WorkloadClass::FineGrain));
        assert_eq!(HEAVY.learned_class(), Some(WorkloadClass::Streaming));
    }

    #[test]
    fn cv2_reflects_site_spread() {
        static UNIFORM: CallSiteState = CallSiteState::new();
        static SPREAD: CallSiteState = CallSiteState::new();
        let per = 10_000u64;
        let scaled = per >> 8;
        UNIFORM.record_batch_site_only(per * 8, scaled * scaled * 8, 8);
        // Spread: half 1us, half 100us.
        let a = 1_000u64;
        let b = 100_000u64;
        let asc = a >> 8;
        let bsc = b >> 8;
        SPREAD.record_batch_site_only(a * 4 + b * 4, asc * asc * 4 + bsc * bsc * 4, 8);
        assert!(UNIFORM.cv2_per_mille().unwrap() < 20);
        assert!(SPREAD.cv2_per_mille().unwrap() >= 500);
    }

    #[test]
    fn policy_arm_explores_then_adopts_faster_arm() {
        static S: CallSiteState = CallSiteState::new();
        // Feed samples: default arm slow (1ms), alternative fast
        // (100us). After both cross ARM_MIN_SAMPLES the chooser must
        // prefer Alternative on non-trial calls.
        for _ in 0..4 {
            S.record_arm(PolicyArm::Default, 1_000_000);
            S.record_arm(PolicyArm::Alternative, 100_000);
        }
        let mut alt = 0;
        for _ in 0..12 {
            if S.choose_arm(true) == PolicyArm::Alternative {
                alt += 1;
            }
        }
        assert!(alt >= 10, "faster arm must dominate; got {alt}/12");
        // Precondition gate: alternative disallowed forces Default.
        assert_eq!(S.choose_arm(false), PolicyArm::Default);
    }

    #[test]
    fn placement_races_cold_then_picks_faster_side() {
        static S: CallSiteState = CallSiteState::new();
        let batch = 4096u32;
        assert_eq!(S.choose_placement(batch), Placement::Race);
        S.record_placement(batch, Some(50_000), Some(5_000_000));
        // Warm bucket, CPU faster: exploit CPU (allowing the
        // scheduled re-probe tick).
        let mut cpu = 0;
        for _ in 0..8 {
            if S.choose_placement(batch) == Placement::Cpu {
                cpu += 1;
            }
        }
        assert!(cpu >= 7, "CPU side must dominate; got {cpu}/8");
        // A different bucket stays cold independently.
        assert_eq!(S.choose_placement(2), Placement::Race);
    }

    #[test]
    fn split_share_tracks_throughput_ratio() {
        static S: CallSiteState = CallSiteState::new();
        assert_eq!(S.split_cpu_share_per_mille(), 500);
        // CPU 10ns/item, backend 30ns/item: CPU should take ~750.
        S.record_split(1000, 10_000, 1000, 30_000);
        let share = S.split_cpu_share_per_mille();
        assert!((700..=800).contains(&share), "share {share}");
    }

    #[test]
    fn site_ref_identity_semantics() {
        static A: CallSiteState = CallSiteState::new();
        static B: CallSiteState = CallSiteState::new();
        assert_eq!(SiteRef::new(&A), SiteRef::new(&A));
        assert_ne!(SiteRef::new(&A), SiteRef::new(&B));
    }
}

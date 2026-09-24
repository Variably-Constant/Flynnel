//! Per-call-site adaptive state, keyed by the caller's source
//! location.
//!
//! Every dispatch entry is `#[track_caller]` and maps
//! `std::panic::Location::caller()` to a `&'static CallSiteState`
//! via [`site_for_location`]: an insert-once open-addressed table of
//! leaked nodes fronted by a per-thread one-slot cache, states
//! `Box::leak`ed for process lifetime. A `static` in a generic fn cannot provide this
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

use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicU32, AtomicU64, Ordering};

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

/// cv^2 per mille of per-item cost, from the recorder's summed per-item
/// squares (each leaf's `ns^2 / (items << 16)`), the integer mean per
/// item, and the item count those sums cover.
///
/// Formed in 128 bits from the scaled sums directly. An integer variance
/// in between would move in steps of `1000 / (mean^2 >> 16)` per mille -
/// 26 at 1.6 us per item, 333 at 500 ns - against class edges at 50 and
/// 500. The integer mean is the one rounding left, and it biases the
/// result by at most `2 / mean` per mille: under 4 at 500 ns per item,
/// the lowest cost the cv^2 edges apply to.
fn per_item_cv2(sumsq_per_item: u64, mean: u64, items: u64) -> u64 {
    let mean_sq = (mean as u128).saturating_mul(mean as u128);
    let expected = mean_sq.saturating_mul(items as u128);
    if expected == 0 {
        return 0;
    }
    let total = (sumsq_per_item as u128) << 16;
    let spread = total.saturating_sub(expected);
    (spread.saturating_mul(1000) / expected) as u64
}

/// The weight of a batch that spent its whole interval on a core.
const FULL_WEIGHT: u64 = 1000;

/// A batch's weight in parts per mille: the share of its interval the
/// worker spent on a core, from the tick pair the recorder reads at both
/// ends of the batch.
///
/// Formed from the two sums directly in 128 bits. The occupancy reported
/// beside a dispatch is in hundredths, and taking the weight from that
/// would move it in steps of ten per mille, which is the whole spread
/// between a batch that got 99 percent of its cores and one that got
/// 100.
///
/// Capped at [`FULL_WEIGHT`]. The two counts come from different clocks
/// on Windows - executed cycles against a fixed-rate timestamp counter -
/// so a boosted core reads over one, and a batch cannot count for more
/// than one batch because its host was generous.
///
/// `None` where the pair describes no interval: a batch whose ends did
/// not both carry an on-core count, or one whose elapsed count is zero.
/// A caller decides what an unweighted batch means to it, at the point
/// where it knows. Zero is not that answer: it is the weight of a batch
/// that never got a core, which is a reading rather than the absence of
/// one.
pub(crate) fn batch_weight_per_mille(on_core_ticks: u64, wall_ticks: u64) -> Option<u64> {
    if wall_ticks == 0 {
        return None;
    }
    let share = (on_core_ticks as u128).saturating_mul(FULL_WEIGHT as u128)
        / (wall_ticks as u128);
    Some((share as u64).min(FULL_WEIGHT))
}

/// Policy-arm trial cadence: every Nth arm selection returns the
/// non-preferred arm so its EWMA stays fresh enough to detect drift.
const ARM_TRIAL_CADENCE: u32 = 16;

/// Minimum samples per arm before the EWMA comparison is trusted.
const ARM_MIN_SAMPLES: u32 = 3;

/// Placement re-probe cadence: every Nth call in a warm size bucket
/// runs both sides again so the model tracks drift (thermal
/// throttling, contention) instead of freezing on stale data.
const PLACEMENT_REPROBE_CADENCE: u32 = 32;

/// Number of log2-size buckets for the placement EWMAs. Bucket i
/// covers batch sizes in `[2^i, 2^(i+1))`; 40 buckets cover every
/// `u32` batch size and then some.
pub const PLACEMENT_BUCKETS: usize = 40;

/// An averaged cell packs its sample count, saturating at
/// [`EWMA_WARM_SAMPLES`], into the top byte; the low 56 bits hold
/// the average in nanoseconds.
const EWMA_COUNT_SHIFT: u32 = 56;
const EWMA_VALUE_MASK: u64 = (1u64 << EWMA_COUNT_SHIFT) - 1;

/// Smoothing rate of the exponential phase, as the denominator of
/// alpha: each update keeps `1 - 1/EWMA_ALPHA_RECIP` of the old
/// average. The value is the reciprocal because the update is a
/// shift, not a multiply.
const EWMA_ALPHA_RECIP: u64 = 8;

/// Samples averaged with equal weight before the update turns
/// exponential. It is the smoothing rate's denominator, not a
/// separate choice: the equal-weight phase then lasts exactly as
/// long as the exponential's own memory, so the handoff neither
/// leaves a sample over-weighted nor discards one.
///
/// The phase exists because a site's first sample is a cold one
/// (pool start, page faults, a cold device). Seeded straight into
/// the exponential it keeps seven eighths of its weight in the
/// second average and half in the sixth, and a tandem split reading
/// the ratio of two such averages spent six rounds of the gemm
/// parity test short of the balance the eighth sample reaches.
const EWMA_WARM_SAMPLES: u64 = EWMA_ALPHA_RECIP;

/// Averaged-cell update: the running mean while fewer than
/// [`EWMA_WARM_SAMPLES`] samples are in, then exponential at
/// [`EWMA_ALPHA_RECIP`]. Zero is the "empty" sentinel, so the first
/// sample seeds directly. Load/store (not CAS) is deliberate:
/// concurrent updates may drop a sample, which is acceptable for a
/// smoothed statistic and keeps the hot path at two relaxed atomics.
#[inline]
fn ewma_update(cell: &AtomicU64, sample_ns: u64) {
    let packed = cell.load(Ordering::Relaxed);
    let count = packed >> EWMA_COUNT_SHIFT;
    let old = packed & EWMA_VALUE_MASK;
    let sample = sample_ns.max(1) & EWMA_VALUE_MASK;
    let (new, count) = if count == 0 {
        (sample, 1)
    } else if count < EWMA_WARM_SAMPLES {
        ((old * count + sample) / (count + 1), count + 1)
    } else {
        (
            (old - old / EWMA_ALPHA_RECIP).saturating_add(sample / EWMA_ALPHA_RECIP),
            count,
        )
    };
    cell.store((count << EWMA_COUNT_SHIFT) | (new.max(1) & EWMA_VALUE_MASK), Ordering::Relaxed);
}

/// The average held by an averaged cell, zero while empty.
#[inline]
fn ewma_value(cell: &AtomicU64) -> u64 {
    cell.load(Ordering::Relaxed) & EWMA_VALUE_MASK
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
    // Cumulative leaf statistics for this site, and beside them the
    // items those leaves covered with the summed per-item squared time.
    // The per-item pair is what a class can be decided from without the
    // split moving it: leaf times change when the scheduler splits
    // differently, and how it splits follows from the class.
    leaf_count: AtomicU64,
    leaf_sum_ns: AtomicU64,
    leaf_sumsq_scaled: AtomicU64,
    leaf_items: AtomicU64,
    leaf_sumsq_per_item: AtomicU64,
    // Summed weight of the batches those sums came from, in parts per
    // mille of a batch. The sums above carry the same factor, so this is
    // the count they divide by; `leaf_count` counts leaves and is what
    // the sample guards and the tick quantum read.
    leaf_weight_sum: AtomicU64,
    // The same leaves timed on the thread's own clock, which advances
    // only while the thread is on a core. Kept in raw counter ticks and
    // never converted: on Windows the thread clock counts executed
    // cycles against a fixed-rate elapsed counter, so a nanosecond
    // conversion would carry the achieved-to-base clock ratio into every
    // figure. A cv^2 is a ratio and a common factor cancels out of it,
    // so these feed the SPREAD only and the mean stays on wall time,
    // where the classifier's nanosecond boundaries are stated.
    //
    // Populated on the sampled path, where the cost of two extra clock
    // reads is divided by the stride. A site whose leaves are all
    // recorded off that path carries zero here and says so by its item
    // count rather than by a spread of zero.
    leaf_oncore_items: AtomicU64,
    leaf_oncore_sum: AtomicU64,
    leaf_oncore_sumsq_per_item: AtomicU64,
    // Snapshot of the cumulative counters at the previous classifier
    // tick; each tick classifies the delta window since then.
    last_count: AtomicU64,
    last_sum_ns: AtomicU64,
    last_sumsq: AtomicU64,
    last_items: AtomicU64,
    last_sumsq_per_item: AtomicU64,
    last_weight_sum: AtomicU64,
    last_oncore_items: AtomicU64,
    last_oncore_sum: AtomicU64,
    last_oncore_sumsq_per_item: AtomicU64,
    // Mean leaf time in nanoseconds and cv^2 per mille of the delta
    // window the latest tick classified, and how many windows have been
    // classified.
    window_mean_ns: AtomicU64,
    window_cv2: AtomicU64,
    window_ticks: AtomicU64,
    // Extremes of the per-window cv^2 across every tick, so a reader
    // gets the range the classifier acted over rather than whichever
    // tick happened to be last. A single tick's figure spans 0 to 527
    // across identical runs, so one reading of it cannot say which
    // regime a run was in.
    window_cv2_min: AtomicU64,
    window_cv2_max: AtomicU64,
    // The same window's spread on each clock apart, with the extremes of
    // each. `window_cv2` is whichever of the two the classifier used;
    // these let a reader set the one it used beside the one it did not.
    // The on-core figures are written only for a window whose spread the
    // classifier took from the on-core clock, which `window_oncore_ticks`
    // counts.
    window_wall_cv2: AtomicU64,
    window_wall_cv2_min: AtomicU64,
    window_wall_cv2_max: AtomicU64,
    window_oncore_cv2: AtomicU64,
    window_oncore_cv2_min: AtomicU64,
    window_oncore_cv2_max: AtomicU64,
    window_oncore_ticks: AtomicU64,
    // Execution-policy A/B arms: per-arm EWMA wall time + sample
    // counts + a call counter driving the trial cadence.
    arm_ewma_ns: [AtomicU64; 2],
    arm_samples: [AtomicU32; 2],
    arm_calls: AtomicU32,
    // The same shape for the routing decision, kept apart from the
    // execution-policy arms because the two consumers would otherwise
    // average each other's dispatches into one EWMA and neither could
    // read its own effect. Default is the plan the learned class
    // re-derives; Alternative is the plan as the caller built it.
    routing_ewma_ns: [AtomicU64; 2],
    routing_samples: [AtomicU32; 2],
    routing_calls: AtomicU32,
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
    /// True once a body this site ran on the calling thread took
    /// longer than the collapse threshold that admitted it. While
    /// set, the site's calls dispatch regardless of the caller's
    /// per-item estimate.
    collapse_overran: AtomicBool,
    /// Seed-depth hysteresis: the depth in force, the depth a recent
    /// call asked for instead, and how many consecutive calls have
    /// asked for it.
    active_depth: AtomicU32,
    pending_depth: AtomicU32,
    depth_run: AtomicU32,
    /// What fraction of its interval the most recent dispatch at this
    /// site spent on a core, in hundredths, or [`OCCUPANCY_UNREPORTED`]
    /// before any dispatch has said.
    ///
    /// A site with no dispatches and a site whose pool held its cores
    /// throughout are different findings, so they do not share a value.
    recent_occupancy_pct: AtomicU32,
    /// On-core ticks and elapsed ticks summed across every worker that
    /// has run a leaf for this site, in the same unit.
    ///
    /// Monotonic, so a dispatch takes the difference across itself: a
    /// caller's own window covers its join wait, during which it is
    /// deliberately not running, and reads low exactly when the work
    /// spread well. These describe the threads that ran the leaves.
    pool_thread_ticks: AtomicU64,
    pool_wall_ticks: AtomicU64,
    /// The seed depth this site last dispatched with, and how many
    /// times a dispatch has used a different one from the dispatch
    /// before it.
    ///
    /// This is the quantity the seed-depth stabilizers exist to reduce:
    /// the same workload seeding a different leaf count from one call
    /// to the next. Throughput does not express it, because a flip
    /// between two adjacent depths costs little either way.
    last_seed_depth: AtomicU32,
    seed_depth_flips: AtomicU32,
}

/// No seed depth is in force for a site yet.
const DEPTH_UNSET: u32 = u32::MAX;

/// No dispatch at a site has reported its pool's occupancy yet.
///
/// Distinct from every occupancy a dispatch can report, which are
/// hundredths and so at most 100. Not public: a caller reads
/// [`CallSiteState::recent_occupancy`], whose `None` says the same
/// thing without the caller having to know the encoding.
const OCCUPANCY_UNREPORTED: u32 = u32::MAX;

/// Consecutive calls that must agree on a different seed depth before
/// it takes effect.
///
/// The same value and the same reason as [`SITE_MIGRATION_HYSTERESIS`]:
/// the seeded leaf count is a bucketed decision driven by a measured
/// quantity, and one reading that lands the far side of a boundary
/// must not move it alone.
pub const SEED_DEPTH_HYSTERESIS: u32 = 2;

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
            leaf_items: AtomicU64::new(0),
            leaf_sumsq_per_item: AtomicU64::new(0),
            leaf_weight_sum: AtomicU64::new(0),
            leaf_oncore_items: AtomicU64::new(0),
            leaf_oncore_sum: AtomicU64::new(0),
            leaf_oncore_sumsq_per_item: AtomicU64::new(0),
            last_oncore_items: AtomicU64::new(0),
            last_oncore_sum: AtomicU64::new(0),
            last_oncore_sumsq_per_item: AtomicU64::new(0),
            last_count: AtomicU64::new(0),
            last_sum_ns: AtomicU64::new(0),
            last_sumsq: AtomicU64::new(0),
            last_items: AtomicU64::new(0),
            last_sumsq_per_item: AtomicU64::new(0),
            last_weight_sum: AtomicU64::new(0),
            window_mean_ns: AtomicU64::new(0),
            window_cv2: AtomicU64::new(0),
            window_ticks: AtomicU64::new(0),
            window_cv2_min: AtomicU64::new(u64::MAX),
            window_cv2_max: AtomicU64::new(0),
            window_wall_cv2: AtomicU64::new(0),
            window_wall_cv2_min: AtomicU64::new(u64::MAX),
            window_wall_cv2_max: AtomicU64::new(0),
            window_oncore_cv2: AtomicU64::new(0),
            window_oncore_cv2_min: AtomicU64::new(u64::MAX),
            window_oncore_cv2_max: AtomicU64::new(0),
            window_oncore_ticks: AtomicU64::new(0),
            arm_ewma_ns: [const { AtomicU64::new(0) }; 2],
            arm_samples: [const { AtomicU32::new(0) }; 2],
            arm_calls: AtomicU32::new(0),
            routing_ewma_ns: [const { AtomicU64::new(0) }; 2],
            routing_samples: [const { AtomicU32::new(0) }; 2],
            routing_calls: AtomicU32::new(0),
            place_cpu_ns: [const { AtomicU64::new(0) }; PLACEMENT_BUCKETS],
            place_backend_ns: [const { AtomicU64::new(0) }; PLACEMENT_BUCKETS],
            place_calls: [const { AtomicU32::new(0) }; PLACEMENT_BUCKETS],
            split_cpu_ns_per_item: AtomicU64::new(0),
            split_backend_ns_per_item: AtomicU64::new(0),
            split_cpu_ns_per_item_by_size: [const { AtomicU64::new(0) }; PLACEMENT_BUCKETS],
            split_backend_ns_per_item_by_size: [const { AtomicU64::new(0) }; PLACEMENT_BUCKETS],
            reduce_cost_sum_cycles: AtomicU64::new(0),
            reduce_cost_samples: AtomicU32::new(0),
            collapse_overran: AtomicBool::new(false),
            active_depth: AtomicU32::new(DEPTH_UNSET),
            pending_depth: AtomicU32::new(DEPTH_UNSET),
            depth_run: AtomicU32::new(0),
            recent_occupancy_pct: AtomicU32::new(OCCUPANCY_UNREPORTED),
            pool_thread_ticks: AtomicU64::new(0),
            pool_wall_ticks: AtomicU64::new(0),
            last_seed_depth: AtomicU32::new(DEPTH_UNSET),
            seed_depth_flips: AtomicU32::new(0),
        }
    }

    /// Returns the site to the state [`CallSiteState::new`] gives, so a
    /// second arm in the same process does not inherit what the first
    /// one taught the classifier.
    ///
    /// Site state is leaked for the life of the process, which is what
    /// makes a `SiteRef` `'static`. Without this, two arms measured in
    /// one process share a learned class, an EWMA per policy arm and a
    /// seed depth, and the second arm's numbers describe both. Measuring
    /// them in separate processes instead carries whatever else differed
    /// between those processes, which for a decision driven by a
    /// measured estimate is the thing being measured.
    ///
    /// Not atomic as a whole. A dispatch running at this site while the
    /// reset lands sees some counters cleared and some not, which skews
    /// that one window; call it between arms, not during one.
    pub fn reset(&self) {
        // Destructured rather than assigned field by field, so a field
        // added to the struct and not cleared here fails to compile. A
        // reset that silently skips a counter is the defect it exists to
        // prevent, and it would show up as the second arm agreeing
        // suspiciously with the first.
        let Self {
            active_tag,
            pending_tag,
            pending_run,
            leaf_count,
            leaf_sum_ns,
            leaf_sumsq_scaled,
            leaf_items,
            leaf_sumsq_per_item,
            leaf_weight_sum,
            leaf_oncore_items,
            leaf_oncore_sum,
            leaf_oncore_sumsq_per_item,
            last_count,
            last_sum_ns,
            last_sumsq,
            last_items,
            last_sumsq_per_item,
            last_weight_sum,
            last_oncore_items,
            last_oncore_sum,
            last_oncore_sumsq_per_item,
            window_mean_ns,
            window_cv2,
            window_ticks,
            window_cv2_min,
            window_cv2_max,
            window_wall_cv2,
            window_wall_cv2_min,
            window_wall_cv2_max,
            window_oncore_cv2,
            window_oncore_cv2_min,
            window_oncore_cv2_max,
            window_oncore_ticks,
            arm_ewma_ns,
            arm_samples,
            arm_calls,
            routing_ewma_ns,
            routing_samples,
            routing_calls,
            place_cpu_ns,
            place_backend_ns,
            place_calls,
            split_cpu_ns_per_item,
            split_backend_ns_per_item,
            split_cpu_ns_per_item_by_size,
            split_backend_ns_per_item_by_size,
            reduce_cost_sum_cycles,
            reduce_cost_samples,
            collapse_overran,
            active_depth,
            pending_depth,
            depth_run,
            recent_occupancy_pct,
            pool_thread_ticks,
            pool_wall_ticks,
            last_seed_depth,
            seed_depth_flips,
        } = self;

        for tag in [active_tag, pending_tag] {
            tag.store(TAG_UNINIT, Ordering::Relaxed);
        }
        for counter in [
            leaf_count,
            leaf_sum_ns,
            leaf_sumsq_scaled,
            leaf_items,
            leaf_sumsq_per_item,
            leaf_weight_sum,
            leaf_oncore_items,
            leaf_oncore_sum,
            leaf_oncore_sumsq_per_item,
            last_count,
            last_sum_ns,
            last_sumsq,
            last_items,
            last_sumsq_per_item,
            last_weight_sum,
            last_oncore_items,
            last_oncore_sum,
            last_oncore_sumsq_per_item,
            window_mean_ns,
            window_cv2,
            window_ticks,
            window_cv2_max,
            window_wall_cv2,
            window_wall_cv2_max,
            window_oncore_cv2,
            window_oncore_cv2_max,
            window_oncore_ticks,
            split_cpu_ns_per_item,
            split_backend_ns_per_item,
            reduce_cost_sum_cycles,
            pool_thread_ticks,
            pool_wall_ticks,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        for counter in [
            pending_run,
            arm_calls,
            routing_calls,
            reduce_cost_samples,
            depth_run,
            seed_depth_flips,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        for depth in [active_depth, pending_depth, last_seed_depth] {
            depth.store(DEPTH_UNSET, Ordering::Relaxed);
        }
        for bank in [
            &arm_ewma_ns[..],
            &routing_ewma_ns[..],
            &place_cpu_ns[..],
            &place_backend_ns[..],
            &split_cpu_ns_per_item_by_size[..],
            &split_backend_ns_per_item_by_size[..],
        ] {
            for cell in bank {
                cell.store(0, Ordering::Relaxed);
            }
        }
        for bank in [&arm_samples[..], &routing_samples[..], &place_calls[..]] {
            for cell in bank {
                cell.store(0, Ordering::Relaxed);
            }
        }
        // The extremes start at the identity for a min, so the first
        // window after a reset sets both rather than being weighed
        // against a range the previous arm established.
        for min in [window_cv2_min, window_wall_cv2_min, window_oncore_cv2_min] {
            min.store(u64::MAX, Ordering::Relaxed);
        }
        recent_occupancy_pct.store(OCCUPANCY_UNREPORTED, Ordering::Relaxed);
        collapse_overran.store(false, Ordering::Relaxed);
    }

    /// Note the seed depth a dispatch at this site is about to use, and
    /// count it when it differs from the one before.
    ///
    /// The first call at a site establishes the depth and counts
    /// nothing: there is no previous dispatch for it to differ from.
    pub fn record_seed_depth(&self, depth: usize) {
        let d = depth as u32;
        let previous = self.last_seed_depth.swap(d, Ordering::Relaxed);
        if previous != DEPTH_UNSET && previous != d {
            self.seed_depth_flips.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Dispatches at this site that seeded a different leaf count from
    /// the dispatch before them.
    ///
    /// The figure the seed-depth stabilizers are measured against: a
    /// mechanism that costs throughput and does not lower this is
    /// paying for nothing.
    ///
    /// Counts only dispatches that reach the adaptive seed depth, which
    /// a plan carrying an explicit `bisect_variant` does not - such a
    /// dispatch names its own shape and has no depth to flip. So zero
    /// at a site whose callers all name a variant means the question
    /// was never asked there, not that the answer was steady.
    pub fn seed_depth_flips(&self) -> u32 {
        self.seed_depth_flips.load(Ordering::Relaxed)
    }

    /// Add one worker's on-core and elapsed ticks for leaves it ran at
    /// this site.
    ///
    /// Called from the worker that ran them, so the sum is over the
    /// threads that did the work rather than over the thread that
    /// waited for it.
    pub(crate) fn add_pool_ticks(&self, thread_ticks: u64, wall_ticks: u64) {
        self.pool_thread_ticks
            .fetch_add(thread_ticks, Ordering::Relaxed);
        self.pool_wall_ticks.fetch_add(wall_ticks, Ordering::Relaxed);
    }

    /// This site's running pool totals, for a caller taking a
    /// difference across a dispatch.
    pub(crate) fn pool_ticks(&self) -> (u64, u64) {
        (
            self.pool_thread_ticks.load(Ordering::Relaxed),
            self.pool_wall_ticks.load(Ordering::Relaxed),
        )
    }

    /// Report what fraction of a dispatch's leaf time its workers spent
    /// on a core.
    ///
    /// A dispatch that ran no leaves through the buffered path reports
    /// nothing, leaving the previous figure rather than overwriting it
    /// with a zero that would read as total contention.
    pub fn record_occupancy(&self, percent: u32) {
        self.recent_occupancy_pct
            .store(percent.min(100), Ordering::Relaxed);
    }

    /// The occupancy of the most recent dispatch at this site, or
    /// `None` if no dispatch has reported one.
    pub fn recent_occupancy(&self) -> Option<u32> {
        match self.recent_occupancy_pct.load(Ordering::Relaxed) {
            OCCUPANCY_UNREPORTED => None,
            pct => Some(pct),
        }
    }

    /// The seed depth to dispatch with, given the depth this call's
    /// estimate asks for.
    ///
    /// The first call takes what it is given. After that a different
    /// depth must be asked for by [`SEED_DEPTH_HYSTERESIS`] consecutive
    /// calls before it takes effect, so one estimate landing the far
    /// side of a power-of-two boundary does not halve or double the
    /// leaf count on its own.
    pub fn stabilise_seed_depth(&self, observed: usize) -> usize {
        let obs = observed as u32;
        let active = self.active_depth.load(Ordering::Relaxed);
        if active == DEPTH_UNSET {
            self.active_depth.store(obs, Ordering::Relaxed);
            return observed;
        }
        if obs == active {
            // The site is asking for what it already has; any run
            // toward a change is broken.
            self.depth_run.store(0, Ordering::Relaxed);
            return observed;
        }
        let previous_pending = self.pending_depth.swap(obs, Ordering::Relaxed);
        let run = if previous_pending == obs {
            self.depth_run
                .fetch_add(1, Ordering::Relaxed)
                .saturating_add(1)
        } else {
            self.depth_run.store(1, Ordering::Relaxed);
            1
        };
        if run >= SEED_DEPTH_HYSTERESIS {
            self.active_depth.store(obs, Ordering::Relaxed);
            self.depth_run.store(0, Ordering::Relaxed);
            return observed;
        }
        active as usize
    }

    /// The seed depth in force, or `None` before this site's first
    /// dispatch. Diagnostics and tests.
    pub fn seeded_depth(&self) -> Option<u32> {
        match self.active_depth.load(Ordering::Relaxed) {
            DEPTH_UNSET => None,
            d => Some(d),
        }
    }

    /// True once this site ran a collapsed body slower than the
    /// collapse threshold, which means its estimate reads low enough
    /// to have made the wrong call.
    #[inline]
    pub fn collapse_overran(&self) -> bool {
        self.collapse_overran.load(Ordering::Relaxed)
    }

    /// Record that a collapsed body took `elapsed_ns` against a
    /// threshold of `threshold_ns`. Latches when it ran over.
    #[inline]
    pub fn note_collapsed_body(&self, elapsed_ns: u64, threshold_ns: u64) {
        if elapsed_ns > threshold_ns {
            self.collapse_overran.store(true, Ordering::Relaxed);
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
    pub fn record_batch(
        &'static self,
        sum_ns: u64,
        sumsq_scaled: u64,
        count: u64,
        items: u64,
        sumsq_per_item: u64,
    ) {
        // Global dual-write first: preserves every existing consumer
        // of the process-wide stats (split multiplier, global
        // auto-classify, effective_use_smt fallback, tests).
        crate::sched::split_observer::record_leaf_batch(
            sum_ns,
            sumsq_scaled,
            count,
            items,
            sumsq_per_item,
        );
        self.record_batch_site_only(sum_ns, sumsq_scaled, count, items, sumsq_per_item);
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
        items: u64,
        sumsq_per_item: u64,
    ) {
        self.record_batch_weighted(sum_ns, sumsq_scaled, count, items, sumsq_per_item, None)
    }

    /// [`Self::record_batch_site_only`] with the share of the batch's
    /// interval its worker spent on a core, from
    /// [`batch_weight_per_mille`].
    ///
    /// Every sum carries the weight as a factor and so does the count
    /// they are divided by, so each statistic is a ratio of weighted
    /// totals: a batch that held its cores counts for a whole batch and
    /// one that got a third of them counts for a third. The means are
    /// unchanged by the weighting and only the influence moves, which is
    /// what a sample of unknown extra spread is worth.
    ///
    /// `None` records the batch at full weight. That is what a platform
    /// with no thread clock, and a batch whose ends did not both carry a
    /// reading, must contribute: an unweighted sample rather than a
    /// suppressed one, because no reading is not evidence of contention.
    ///
    /// The sums scale by up to [`FULL_WEIGHT`] and are not divided back
    /// down, which keeps the ratios exact rather than truncating a small
    /// batch's items to zero. That costs ten bits of headroom on
    /// counters that already saturate, and `leaf_sumsq_scaled` is the
    /// one close enough to its ceiling for that to be reachable.
    ///
    /// `leaf_count` is left unweighted, because it is a count of leaves
    /// rather than a quantity the statistics divide: it drives the
    /// sample guards and the classifier quantum, and weighting it would
    /// put both out by whatever the host was doing.
    pub(crate) fn record_batch_weighted(
        &'static self,
        sum_ns: u64,
        sumsq_scaled: u64,
        count: u64,
        items: u64,
        sumsq_per_item: u64,
        weight_per_mille: Option<u64>,
    ) {
        let w = weight_per_mille.unwrap_or(FULL_WEIGHT).min(FULL_WEIGHT);
        let scale = |v: u64| v.saturating_mul(w);
        self.leaf_sum_ns.fetch_add(scale(sum_ns), Ordering::Relaxed);
        self.leaf_sumsq_scaled
            .fetch_add(scale(sumsq_scaled), Ordering::Relaxed);
        self.leaf_items.fetch_add(scale(items), Ordering::Relaxed);
        self.leaf_sumsq_per_item
            .fetch_add(scale(sumsq_per_item), Ordering::Relaxed);
        self.leaf_weight_sum
            .fetch_add(scale(count), Ordering::Relaxed);
        let prior = self.leaf_count.fetch_add(count, Ordering::Relaxed);
        let new_total = prior.wrapping_add(count);
        if (prior / SITE_CLASSIFY_QUANTUM) != (new_total / SITE_CLASSIFY_QUANTUM) {
            self.tick();
        }
    }

    /// Record a batch of leaves timed on the thread's own clock.
    ///
    /// Separate from [`Self::record_batch_weighted`] because these are
    /// the same leaves measured against a different clock, not more
    /// leaves: adding them to the leaf count would double it. They carry
    /// their own item count, so a site that reaches this path for some
    /// of its leaves and not others divides each figure by the items
    /// that figure actually covers.
    ///
    /// Raw counter ticks, unconverted. See the field comment: only a
    /// ratio is ever taken of these, and a ratio is what survives the
    /// unit being different from the elapsed clock's.
    pub(crate) fn record_oncore_batch(
        &'static self,
        oncore_sum: u64,
        oncore_sumsq_per_item: u64,
        items: u64,
    ) {
        if items == 0 {
            return;
        }
        self.leaf_oncore_sum.fetch_add(oncore_sum, Ordering::Relaxed);
        self.leaf_oncore_sumsq_per_item
            .fetch_add(oncore_sumsq_per_item, Ordering::Relaxed);
        self.leaf_oncore_items.fetch_add(items, Ordering::Relaxed);
    }

    /// cv^2 per mille of per-item cost measured on the thread's own
    /// clock, or `None` where no leaf carried an on-core reading.
    ///
    /// This is the figure a neighbour cannot move. Wall time rises both
    /// because the work is irregular and because the thread lost its
    /// core, and preemption lands on some leaves and not others, so it
    /// reaches a wall-time spread as variance that is indistinguishable
    /// from the work's own. A thread's own clock does not advance while
    /// the thread is off a core, so a preempted leaf reports what it
    /// cost rather than what it waited.
    ///
    /// `None` rather than zero: zero is the spread of perfectly uniform
    /// work, which is a reading, and a site whose leaves never reached
    /// the sampled path has no reading at all.
    pub fn per_item_oncore_cv2_per_mille(&self) -> Option<u64> {
        let items = self.leaf_oncore_items.load(Ordering::Relaxed);
        if items == 0 {
            return None;
        }
        let mean = self.leaf_oncore_sum.load(Ordering::Relaxed) / items;
        if mean == 0 {
            return None;
        }
        let sumsq = self.leaf_oncore_sumsq_per_item.load(Ordering::Relaxed);
        Some(per_item_cv2(sumsq, mean, items))
    }

    /// Leaves' worth of items that carried an on-core reading.
    ///
    /// Beside [`Self::per_item_oncore_cv2_per_mille`] so a reader can
    /// weigh how much the spread rests on, and distinct from
    /// [`Self::leaf_count`], which counts every leaf however it was
    /// timed.
    pub fn oncore_items(&self) -> u64 {
        self.leaf_oncore_items.load(Ordering::Relaxed)
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
        // The sums carry each batch's weight as a factor, so the count
        // they divide by has to carry it too. `n` above is leaves, which
        // is what the four-sample guard is about.
        let n = self.leaf_weight_sum.load(Ordering::Relaxed).max(1);
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

    /// Summed wall time of the leaves recorded against this site, in
    /// nanoseconds, over its cumulative history: the batch-weighted
    /// mean leaf time times the leaf count, so a batch measured at
    /// partial occupancy counts for less. Zero while no batch has
    /// carried weight.
    pub fn leaf_sum_ns(&self) -> u64 {
        let weight = self.leaf_weight_sum.load(Ordering::Relaxed);
        if weight == 0 {
            return 0;
        }
        let mean = self.leaf_sum_ns.load(Ordering::Relaxed) / weight;
        mean.saturating_mul(self.leaf_count.load(Ordering::Relaxed))
    }

    /// Mean cost of one item at this site, in nanoseconds, over its
    /// cumulative history. `None` below 4 leaves, and `None` while no
    /// recorded leaf carried an item count.
    ///
    /// This is the figure a class can be decided from without the split
    /// moving it. The mean leaf time doubles when the scheduler runs
    /// leaves twice the size, and how finely it splits follows from the
    /// class the leaf times produced.
    pub fn per_item_ns(&self) -> Option<u64> {
        if self.leaf_count.load(Ordering::Relaxed) < 4 {
            return None;
        }
        let items = self.leaf_items.load(Ordering::Relaxed);
        if items == 0 {
            return None;
        }
        Some(self.leaf_sum_ns.load(Ordering::Relaxed) / items)
    }

    /// cv^2 per mille of per-item cost at this site, weighted by the
    /// items each leaf covered. `None` on the same terms as
    /// [`Self::per_item_ns`].
    ///
    /// The recorder keeps each leaf's squared time over its item count,
    /// so the sum of those less the mean squared times the items is the
    /// item-weighted sum of squared deviations, and the variance is that
    /// over the items rather than over the leaves. Leaves of mixed sizes
    /// running identical items read near zero here and read high in
    /// [`Self::cv2_per_mille`].
    pub fn per_item_cv2_per_mille(&self) -> Option<u64> {
        let leaves = self.leaf_count.load(Ordering::Relaxed);
        if leaves < 4 {
            return None;
        }
        let items = self.leaf_items.load(Ordering::Relaxed);
        if items == 0 {
            return None;
        }
        // The mean squares in 128 bits and scales once, matching how the
        // recorder forms each leaf's term: scaling the mean first would
        // subtract a smaller square than the terms carry and report the
        // difference as spread.
        let mean_ns = self.leaf_sum_ns.load(Ordering::Relaxed) / items;
        let sumsq_per_item = self.leaf_sumsq_per_item.load(Ordering::Relaxed);
        Some(per_item_cv2(sumsq_per_item, mean_ns, items))
    }

    /// Mean cost of one item, in nanoseconds, over the delta window the
    /// latest classifier tick classified: the mean
    /// [`Self::learned_class`] was decided from. A window whose samples
    /// carried no item count reports its mean leaf time instead.
    /// `None` until a tick has classified a window. While ticks run it
    /// may come from a different tick than
    /// [`Self::window_cv2_per_mille`].
    pub fn window_mean_ns(&self) -> Option<u64> {
        if self.window_ticks.load(Ordering::Relaxed) == 0 {
            None
        } else {
            Some(self.window_mean_ns.load(Ordering::Relaxed))
        }
    }

    /// cv^2 per mille of per-item cost over the delta window the latest
    /// classifier tick classified, the variance [`Self::learned_class`]
    /// was decided from, as opposed to [`Self::cv2_per_mille`] over the
    /// site's whole life and over leaf times rather than items. It is the
    /// on-core clock's spread where the window carried one and wall
    /// time's otherwise; [`Self::window_wall_cv2_per_mille`] and
    /// [`Self::window_oncore_cv2_per_mille`] give the two apart. A window
    /// whose samples carried no item count reports the spread of its
    /// leaf times instead. `None` until a tick has classified a window.
    pub fn window_cv2_per_mille(&self) -> Option<u64> {
        if self.window_ticks.load(Ordering::Relaxed) == 0 {
            None
        } else {
            Some(self.window_cv2.load(Ordering::Relaxed))
        }
    }

    /// Delta windows the site's classifier has classified.
    pub fn window_ticks(&self) -> u64 {
        self.window_ticks.load(Ordering::Relaxed)
    }

    /// Smallest and largest per-window cv^2 the classifier has seen,
    /// over every tick rather than the latest one.
    ///
    /// [`Self::window_cv2_per_mille`] reports one tick of what is often
    /// thousands, and that figure spans 0 to 527 across identical runs,
    /// so it cannot say which regimes a run passed through. The range
    /// can: a maximum below the uniform edge says a spread-driven
    /// mechanism was never consulted in its own regime, whatever the
    /// last tick happened to hold. `None` until a tick has classified a
    /// window.
    pub fn window_cv2_range_per_mille(&self) -> Option<(u64, u64)> {
        if self.window_ticks.load(Ordering::Relaxed) == 0 {
            None
        } else {
            Some((
                self.window_cv2_min.load(Ordering::Relaxed),
                self.window_cv2_max.load(Ordering::Relaxed),
            ))
        }
    }

    /// cv^2 per mille of per-item cost on wall time over the delta window
    /// the latest tick classified, whichever clock the classifier took its
    /// spread from; a window whose samples carried no item count reports
    /// its leaf times' spread. Where [`Self::window_oncore_cv2_per_mille`]
    /// answers for the same tick, the two are one window on two clocks.
    /// `None` until a tick has classified a window.
    pub fn window_wall_cv2_per_mille(&self) -> Option<u64> {
        if self.window_ticks.load(Ordering::Relaxed) == 0 {
            None
        } else {
            Some(self.window_wall_cv2.load(Ordering::Relaxed))
        }
    }

    /// Smallest and largest per-window wall cv^2 over every tick. `None`
    /// until a tick has classified a window.
    pub fn window_wall_cv2_range_per_mille(&self) -> Option<(u64, u64)> {
        if self.window_ticks.load(Ordering::Relaxed) == 0 {
            None
        } else {
            Some((
                self.window_wall_cv2_min.load(Ordering::Relaxed),
                self.window_wall_cv2_max.load(Ordering::Relaxed),
            ))
        }
    }

    /// cv^2 per mille of per-item cost on the thread's own clock over the
    /// latest delta window whose spread the classifier took from that
    /// clock: [`crate::sched::levers::oncore_spread`] on, and the window's
    /// leaves timed on the sampled path. `None` until such a window has
    /// been classified, so a run that took no on-core timing reads as
    /// nothing rather than as a spread of zero.
    pub fn window_oncore_cv2_per_mille(&self) -> Option<u64> {
        if self.window_oncore_ticks.load(Ordering::Relaxed) == 0 {
            None
        } else {
            Some(self.window_oncore_cv2.load(Ordering::Relaxed))
        }
    }

    /// Smallest and largest on-core cv^2 over the windows whose spread the
    /// classifier took from the on-core clock. `None` until such a window
    /// has been classified.
    pub fn window_oncore_cv2_range_per_mille(&self) -> Option<(u64, u64)> {
        if self.window_oncore_ticks.load(Ordering::Relaxed) == 0 {
            None
        } else {
            Some((
                self.window_oncore_cv2_min.load(Ordering::Relaxed),
                self.window_oncore_cv2_max.load(Ordering::Relaxed),
            ))
        }
    }

    /// Delta windows whose spread the classifier took from the on-core
    /// clock, out of [`Self::window_ticks`].
    pub fn window_oncore_ticks(&self) -> u64 {
        self.window_oncore_ticks.load(Ordering::Relaxed)
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
        let items = self.leaf_items.load(Ordering::Relaxed);
        let sumsq_per_item = self.leaf_sumsq_per_item.load(Ordering::Relaxed);
        let dsum = sum.saturating_sub(self.last_sum_ns.load(Ordering::Relaxed));
        let dsumsq = sumsq.saturating_sub(self.last_sumsq.load(Ordering::Relaxed));
        let ditems = items.saturating_sub(self.last_items.load(Ordering::Relaxed));
        let dsumsq_per_item =
            sumsq_per_item.saturating_sub(self.last_sumsq_per_item.load(Ordering::Relaxed));
        let weight_sum = self.leaf_weight_sum.load(Ordering::Relaxed);
        let dweight = weight_sum
            .saturating_sub(self.last_weight_sum.load(Ordering::Relaxed))
            .max(1);
        self.last_weight_sum.store(weight_sum, Ordering::Relaxed);
        self.last_count.store(count, Ordering::Relaxed);
        self.last_sum_ns.store(sum, Ordering::Relaxed);
        self.last_sumsq.store(sumsq, Ordering::Relaxed);
        self.last_items.store(items, Ordering::Relaxed);
        self.last_sumsq_per_item.store(sumsq_per_item, Ordering::Relaxed);

        // Per item, not per leaf. A leaf's time scales with the items in
        // it, and how many that is comes from the split, which follows
        // from the class this decides: classifying leaf times lets the
        // class hold itself in place. Dividing by the items the window
        // covered leaves a figure the split cannot move.
        //
        // A window whose samples carried no item count - the heartbeat's
        // serial spans - is classified on its leaf times, which is all
        // such a sample can say.
        // The window's on-core deltas, taken over the same leaves on the
        // thread's own clock. A spread computed from these is the work's
        // own irregularity; the same spread computed from wall time also
        // carries every leaf that lost its core, because preemption
        // lands on some leaves and not others and so arrives as variance
        // rather than as a level shift.
        let oncore_items = self.leaf_oncore_items.load(Ordering::Relaxed);
        let oncore_sum = self.leaf_oncore_sum.load(Ordering::Relaxed);
        let oncore_sumsq = self.leaf_oncore_sumsq_per_item.load(Ordering::Relaxed);
        let d_oncore_items =
            oncore_items.saturating_sub(self.last_oncore_items.load(Ordering::Relaxed));
        let d_oncore_sum =
            oncore_sum.saturating_sub(self.last_oncore_sum.load(Ordering::Relaxed));
        let d_oncore_sumsq = oncore_sumsq
            .saturating_sub(self.last_oncore_sumsq_per_item.load(Ordering::Relaxed));
        self.last_oncore_items.store(oncore_items, Ordering::Relaxed);
        self.last_oncore_sum.store(oncore_sum, Ordering::Relaxed);
        self.last_oncore_sumsq_per_item
            .store(oncore_sumsq, Ordering::Relaxed);

        // Ticks, not nanoseconds, and only ever a ratio is taken of
        // them, so the clock's unit never reaches a threshold.
        let oncore_cv2 = d_oncore_sum
            .checked_div(d_oncore_items)
            .filter(|mean| *mean > 0)
            .map(|mean| per_item_cv2(d_oncore_sumsq, mean, d_oncore_items));

        let per_item = dsum.checked_div(ditems);
        // The mean stays on wall time, where the classifier's nanosecond
        // boundaries are stated, and so does the first spread here. The
        // classifier then takes its spread from the on-core clock where
        // the window carried one, and keeps the wall spread where it did
        // not, which is what a platform with no thread clock has always
        // had.
        let (mean_ns, wall_cv2) = if let Some(mean) = per_item {
            (mean, per_item_cv2(dsumsq_per_item, mean, ditems))
        } else {
            // Divided by the window's summed weight rather than its leaf
            // count, because the sums carry each batch's weight as a
            // factor and a leaf count does not.
            let mean = dsum / dweight;
            let scaled_mean = (dsum >> 8) / dweight;
            let spread = if scaled_mean == 0 {
                0
            } else {
                let sumsq_per_n = dsumsq / dweight;
                let mean_sq = scaled_mean.saturating_mul(scaled_mean);
                let var = sumsq_per_n.saturating_sub(mean_sq);
                var.saturating_mul(1000) / mean_sq.max(1)
            };
            (mean, spread)
        };
        let oncore_used = oncore_cv2.filter(|_| per_item.is_some());
        let cv2 = oncore_used.unwrap_or(wall_cv2);
        self.window_mean_ns.store(mean_ns, Ordering::Relaxed);
        self.window_cv2.store(cv2, Ordering::Relaxed);
        self.window_ticks.fetch_add(1, Ordering::Relaxed);
        self.window_cv2_min.fetch_min(cv2, Ordering::Relaxed);
        self.window_cv2_max.fetch_max(cv2, Ordering::Relaxed);
        self.window_wall_cv2.store(wall_cv2, Ordering::Relaxed);
        self.window_wall_cv2_min.fetch_min(wall_cv2, Ordering::Relaxed);
        self.window_wall_cv2_max.fetch_max(wall_cv2, Ordering::Relaxed);
        if let Some(oncore) = oncore_used {
            self.window_oncore_cv2.store(oncore, Ordering::Relaxed);
            self.window_oncore_cv2_min.fetch_min(oncore, Ordering::Relaxed);
            self.window_oncore_cv2_max.fetch_max(oncore, Ordering::Relaxed);
            self.window_oncore_ticks.fetch_add(1, Ordering::Relaxed);
        }
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
    ///    comparison does not prefer (drift detection).
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
        let e0 = ewma_value(&self.arm_ewma_ns[0]);
        let e1 = ewma_value(&self.arm_ewma_ns[1]);
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

    /// Pick whether this dispatch runs the routing the learned class
    /// re-derives ([`PolicyArm::Default`]) or the plan as the caller
    /// built it ([`PolicyArm::Alternative`]).
    ///
    /// Same selection as [`Self::choose_arm`] and for the same reason,
    /// on its own counters: explore until both arms have
    /// [`ARM_MIN_SAMPLES`], then take the lower EWMA, and every
    /// [`ARM_TRIAL_CADENCE`]th call run the other one so a routing that
    /// stopped being the better choice is found rather than assumed.
    ///
    /// A class is a description of the work and this is the check on
    /// what that description costs. Where the two disagree the
    /// measurement wins, which is what keeps a class that has gone
    /// wrong from being expensive as well as wrong.
    pub fn choose_routing_arm(&self) -> PolicyArm {
        let calls = self.routing_calls.fetch_add(1, Ordering::Relaxed);
        let s0 = self.routing_samples[0].load(Ordering::Relaxed);
        let s1 = self.routing_samples[1].load(Ordering::Relaxed);
        if s0 < ARM_MIN_SAMPLES || s1 < ARM_MIN_SAMPLES {
            return if s1 < s0 {
                PolicyArm::Alternative
            } else {
                PolicyArm::Default
            };
        }
        let e0 = ewma_value(&self.routing_ewma_ns[0]);
        let e1 = ewma_value(&self.routing_ewma_ns[1]);
        let best = if e1 < e0 {
            PolicyArm::Alternative
        } else {
            PolicyArm::Default
        };
        if calls % ARM_TRIAL_CADENCE == ARM_TRIAL_CADENCE - 1 {
            return match best {
                PolicyArm::Default => PolicyArm::Alternative,
                PolicyArm::Alternative => PolicyArm::Default,
            };
        }
        best
    }

    /// Record one dispatch's wall time under the routing arm it ran.
    pub fn record_routing_arm(&self, arm: PolicyArm, wall_ns: u64) {
        let i = arm.idx();
        ewma_update(&self.routing_ewma_ns[i], wall_ns);
        self.routing_samples[i].fetch_add(1, Ordering::Relaxed);
    }

    /// Current per-arm EWMA wall times for the routing decision,
    /// `(class_derived_ns, caller_plan_ns)`; zero means no samples yet.
    /// Diagnostics and tests.
    pub fn routing_ewmas(&self) -> (u64, u64) {
        (
            ewma_value(&self.routing_ewma_ns[0]),
            ewma_value(&self.routing_ewma_ns[1]),
        )
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
            ewma_value(&self.arm_ewma_ns[0]),
            ewma_value(&self.arm_ewma_ns[1]),
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
        let cpu = ewma_value(&self.place_cpu_ns[b]);
        let dev = ewma_value(&self.place_backend_ns[b]);
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
            ewma_value(&self.place_cpu_ns[b]),
            ewma_value(&self.place_backend_ns[b]),
        )
    }

    /// CPU share of a divisible workload per the learned per-item
    /// throughputs, in parts-per-1000. 500 (an even split) until both
    /// sides have measurements. CPU share = backend_ns_per_item /
    /// (cpu_ns_per_item + backend_ns_per_item): the faster side gets
    /// the larger share.
    pub fn split_cpu_share_per_mille(&self) -> u32 {
        let c = ewma_value(&self.split_cpu_ns_per_item);
        let g = ewma_value(&self.split_backend_ns_per_item);
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
        let c = ewma_value(&self.split_cpu_ns_per_item_by_size[b]);
        let g = ewma_value(&self.split_backend_ns_per_item_by_size[b]);
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
/// The location is kept beside the state so the registry can say where
/// each site is. The key is a hash and cannot be turned back into a
/// file and line, and `CallSiteState` holds no location of its own, so
/// without this a walk of the registry answers a list of anonymous
/// counters. One pointer per site, stored once when the location is
/// first seen; the dispatch path reads through a thread-local one-slot
/// cache and does not touch this map at all.
struct SiteNode {
    key: u64,
    location: &'static std::panic::Location<'static>,
    state: &'static CallSiteState,
}

/// Slot count of the site table. Call sites are source locations in
/// the binary, so the population is fixed at compile time and small;
/// 2048 slots is 16 KiB of zeroed bss and leaves the table under an
/// eighth full for a crate with a couple of hundred dispatch sites.
/// Linear probing wants the headroom: a table near capacity walks a
/// long run of occupied slots on every miss.
const SITE_SLOTS: usize = 2048;
const SITE_MASK: usize = SITE_SLOTS - 1;

/// Open-addressed, insert-once, never-removed table of leaked nodes.
///
/// A slot is empty exactly when its pointer is null, and a node is
/// fully built before the pointer that publishes it, so a non-null
/// slot is always a complete entry and a reader needs one atomic load
/// per probe and no lock. Nothing is ever removed or rehashed, so a
/// pointer a reader holds stays valid for the life of the process.
static SITE_TABLE: [AtomicPtr<SiteNode>; SITE_SLOTS] =
    [const { AtomicPtr::new(core::ptr::null_mut()) }; SITE_SLOTS];

/// The number of sites the table failed to hold. Non-zero means
/// [`SITE_SLOTS`] was reached and later sites are running on
/// unshared one-off state, which reads as every call being a first
/// call. Silence here would make that look like ordinary behavior.
static SITE_TABLE_OVERFLOW: AtomicU64 = AtomicU64::new(0);

/// The node for `key`, or `None` with the probe stopping at the first
/// empty slot, which is where an insert for this key would go.
fn site_lookup(key: u64) -> Option<&'static SiteNode> {
    let mut idx = (key as usize) & SITE_MASK;
    for _ in 0..SITE_SLOTS {
        let p = SITE_TABLE[idx].load(Ordering::Acquire);
        if p.is_null() {
            return None;
        }
        // SAFETY: a non-null slot holds a leaked SiteNode that is
        // never freed, moved or rehashed.
        let node = unsafe { &*p };
        if node.key == key {
            return Some(node);
        }
        idx = (idx + 1) & SITE_MASK;
    }
    None
}

/// The state for `key`, inserting one that records `loc` if the key
/// is not yet present. Racing inserters of the same key all return
/// the one node that won its slot.
fn site_insert(key: u64, loc: &'static std::panic::Location<'static>) -> &'static CallSiteState {
    // Built at most once and only on reaching an empty slot, so a
    // caller that finds the key already present allocates nothing. A
    // racer that loses its CAS and then finds its key further along
    // abandons the node it prepared; that is bounded by the number of
    // threads meeting one new location at the same moment.
    let mut prepared: Option<&'static SiteNode> = None;
    let mut idx = (key as usize) & SITE_MASK;
    for _ in 0..SITE_SLOTS {
        let p = SITE_TABLE[idx].load(Ordering::Acquire);
        if p.is_null() {
            let node = *prepared.get_or_insert_with(|| {
                &*Box::leak(Box::new(SiteNode {
                    key,
                    location: loc,
                    state: Box::leak(Box::new(CallSiteState::new())),
                }))
            });
            let fresh = node as *const SiteNode as *mut SiteNode;
            match SITE_TABLE[idx].compare_exchange(
                core::ptr::null_mut(),
                fresh,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return node.state,
                Err(taken) => {
                    // SAFETY: as in site_lookup.
                    let other = unsafe { &*taken };
                    if other.key == key {
                        return other.state;
                    }
                }
            }
        } else {
            // SAFETY: as in site_lookup.
            let node = unsafe { &*p };
            if node.key == key {
                return node.state;
            }
        }
        idx = (idx + 1) & SITE_MASK;
    }
    SITE_TABLE_OVERFLOW.fetch_add(1, Ordering::Relaxed);
    Box::leak(Box::new(CallSiteState::new()))
}

/// How many sites the table could not hold. Zero on every table that
/// has not reached [`SITE_SLOTS`] distinct call sites.
pub fn site_table_overflow() -> u64 {
    SITE_TABLE_OVERFLOW.load(Ordering::Relaxed)
}

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
/// (file, line, column) probed in the site table, one atomic load per
/// probe. A location seen for the first time process-wide is
/// published into an empty slot by a single compare-exchange; every
/// other caller only reads.
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
    let site: &'static CallSiteState = match site_lookup(key) {
        Some(node) => node.state,
        None => site_insert(key, loc),
    };
    LAST.with(|c| c.set((loc_addr, site as *const CallSiteState as usize)));
    SiteRef::new(site)
}

/// Number of distinct call sites the registry has materialised.
#[cfg(test)]
pub(crate) fn registry_len() -> usize {
    site_nodes().count()
}

/// Every node the table holds, in slot order.
///
/// Slot order is the hash's and carries no meaning. A scan can see an
/// insert that lands in an earlier slot after it has passed that slot,
/// so a walk answers the sites present at some point during it rather
/// than an instant's snapshot. Nothing is ever removed, so a site the
/// scan does report was really there.
fn site_nodes() -> impl Iterator<Item = &'static SiteNode> {
    SITE_TABLE.iter().filter_map(|slot| {
        let p = slot.load(Ordering::Acquire);
        if p.is_null() {
            None
        } else {
            // SAFETY: as in site_lookup.
            Some(unsafe { &*p })
        }
    })
}

/// One call site the registry has materialised, and where it is.
#[derive(Copy, Clone, Debug)]
pub struct RegisteredSite {
    /// The source location the site was first seen at.
    ///
    /// The same textual site inside a generic caller can surface as
    /// distinct `Location` constants per instantiation, and the
    /// registry merges those onto one entry, so this is whichever of
    /// them arrived first. They agree on file, line and column, which
    /// is what the entry is keyed on.
    pub location: &'static std::panic::Location<'static>,
    /// What that site has learned.
    pub site: SiteRef,
}

/// Every call site the registry holds, each with where it is.
///
/// Order is the table's and carries no meaning; sort on the location.
pub fn registered_sites() -> Vec<RegisteredSite> {
    site_nodes()
        .map(|node| RegisteredSite {
            location: node.location,
            site: SiteRef::new(node.state),
        })
        .collect()
}

/// Resets every site in the registry, answering how many.
///
/// For measuring two arms in one process: see [`CallSiteState::reset`]
/// for why the alternative, a process each, measures something else.
///
/// The sweep blocks nothing and is a few tens of atomic stores per
/// site. It is not an instant: a site registered while the walk is
/// past its slot keeps the counters it had, so reset before starting
/// the arm rather than alongside it.
pub fn reset_all_sites() -> usize {
    let mut swept = 0;
    for node in site_nodes() {
        node.state.reset();
        swept += 1;
    }
    swept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_that_held_its_cores_weighs_a_whole_batch() {
        assert_eq!(batch_weight_per_mille(8_000, 8_000), Some(FULL_WEIGHT));
        assert_eq!(batch_weight_per_mille(4_800, 8_000), Some(600));
    }

    #[test]
    fn a_weight_resolves_a_single_part_per_mille() {
        // The occupancy printed beside a dispatch is in hundredths, so a
        // weight taken from it would move in steps of ten per mille and
        // read both of these as zero. One percent of a batch is the
        // difference between a pool that held its cores and one that did
        // not, which is the distinction the weight exists to carry.
        assert_eq!(batch_weight_per_mille(1, 1_000), Some(1));
        assert_eq!(batch_weight_per_mille(999, 1_000), Some(999));
    }

    #[test]
    fn a_boosted_clock_pair_weighs_one_batch_and_not_more() {
        // On Windows the two counts come from different clocks: executed
        // cycles against a fixed-rate timestamp counter, so a core above
        // its base frequency reads over one. A generous host must not
        // make a batch count for more than a batch.
        assert_eq!(batch_weight_per_mille(12_000, 8_000), Some(FULL_WEIGHT));
    }

    #[test]
    fn an_interval_that_never_elapsed_has_no_weight_rather_than_none_of_one() {
        // Zero is the weight of a batch that got no core at all. Handing
        // it back for a pair that measured nothing would make a platform
        // without the clock read as permanently contended, which is the
        // defect the ThreadTicks type exists to prevent.
        assert_eq!(batch_weight_per_mille(0, 0), None);
        assert_eq!(batch_weight_per_mille(5_000, 0), None);
        assert_eq!(batch_weight_per_mille(0, 8_000), Some(0));
    }

    #[test]
    fn the_range_keeps_a_tick_the_latest_reading_has_already_lost() {
        // Two windows of different spread, in order. window_cv2 holds
        // the second and nothing recoverable from it says the first
        // happened, which is how a run through a high-variance regime
        // reports as uniform. The range holds both.
        static S: CallSiteState = CallSiteState::new();
        const ITEMS: u64 = 1024;
        let sq = |ns: u64| (ns >> 8).saturating_mul(ns >> 8);
        let per_item_sq =
            |ns: u64| ((ns as u128).saturating_mul(ns as u128) / ((ITEMS as u128) << 16)) as u64;

        // Spread: half the leaves at 1280 ns an item, half at 1920.
        let (fast, slow) = (1280u64 * ITEMS, 1920u64 * ITEMS);
        S.record_batch_site_only(
            8 * fast + 8 * slow,
            8 * sq(fast) + 8 * sq(slow),
            16,
            16 * ITEMS,
            8 * per_item_sq(fast) + 8 * per_item_sq(slow),
        );
        let spread_tick = S.window_cv2_per_mille().expect("a window was classified");
        assert!(spread_tick > 0, "the first window carries a spread");

        // Flat: every leaf the same, so this window's cv^2 is zero.
        let flat = 1600u64 * ITEMS;
        S.record_batch_site_only(
            16 * flat,
            16 * sq(flat),
            16,
            16 * ITEMS,
            16 * per_item_sq(flat),
        );
        assert_eq!(
            S.window_cv2_per_mille(),
            Some(0),
            "the latest reading is the flat window and says nothing of the first"
        );

        let (lo, hi) = S.window_cv2_range_per_mille().expect("two windows were classified");
        assert_eq!(lo, 0, "the flat window is the smallest seen");
        assert_eq!(hi, spread_tick, "and the spread one survives in the range");
    }

    #[test]
    fn a_per_item_spread_is_read_at_its_own_value() {
        // Sixteen leaves of 1024 items: eight at 1280 ns per item, eight
        // at 1920. Mean 1600, standard deviation 320, cv^2 exactly 40
        // per mille. The recorder's per-leaf term, ns^2 / (items << 16),
        // is 25600 and 57600 for the two leaf kinds, both exact. The
        // batch crosses the classify quantum, so it ticks once.
        static S: CallSiteState = CallSiteState::new();
        const ITEMS: u64 = 1024;
        let (fast, slow) = (1280u64 * ITEMS, 1920u64 * ITEMS);
        let sq = |ns: u64| (ns >> 8).saturating_mul(ns >> 8);
        let per_item_sq = |ns: u64| {
            ((ns as u128).saturating_mul(ns as u128) / ((ITEMS as u128) << 16)) as u64
        };
        S.record_batch_site_only(
            8 * fast + 8 * slow,
            8 * sq(fast) + 8 * sq(slow),
            16,
            16 * ITEMS,
            8 * per_item_sq(fast) + 8 * per_item_sq(slow),
        );
        assert_eq!(S.window_mean_ns(), Some(1600));
        assert_eq!(S.per_item_ns(), Some(1600));
        // An integer variance in between reads 25 here: one unit of
        // 1000 / (1600^2 >> 16), which is the wrong side of the 50 edge
        // from the value the leaves carry.
        assert_eq!(S.window_cv2_per_mille(), Some(40));
        assert_eq!(S.per_item_cv2_per_mille(), Some(40));
        let low = crate::sched::adaptive_profile::class_thresholds()
            .cv2_low_per_mille
            .load(Ordering::Relaxed);
        if 40 < low {
            assert_eq!(S.learned_class(), Some(WorkloadClass::Streaming));
        }
    }

    #[test]
    fn a_window_reports_its_spread_on_each_clock() {
        // The leaves of the test above on wall time, cv^2 40 per mille,
        // and the same leaves flat on the thread's own clock. The
        // classifier takes the on-core spread and the wall one stays
        // readable beside it.
        static S: CallSiteState = CallSiteState::new();
        const ITEMS: u64 = 1024;
        let (fast, slow) = (1280u64 * ITEMS, 1920u64 * ITEMS);
        let flat = 1600u64 * ITEMS;
        let sq = |ns: u64| (ns >> 8).saturating_mul(ns >> 8);
        let per_item_sq = |ns: u64| {
            ((ns as u128).saturating_mul(ns as u128) / ((ITEMS as u128) << 16)) as u64
        };
        S.record_oncore_batch(16 * flat, 16 * per_item_sq(flat), 16 * ITEMS);
        S.record_batch_site_only(
            8 * fast + 8 * slow,
            8 * sq(fast) + 8 * sq(slow),
            16,
            16 * ITEMS,
            8 * per_item_sq(fast) + 8 * per_item_sq(slow),
        );
        assert_eq!(S.window_wall_cv2_per_mille(), Some(40));
        assert_eq!(S.window_oncore_cv2_per_mille(), Some(0));
        assert_eq!(S.window_cv2_per_mille(), Some(0), "the classifier used the on-core spread");
        assert_eq!(S.window_oncore_ticks(), 1);
        assert_eq!(S.window_wall_cv2_range_per_mille(), Some((40, 40)));
        assert_eq!(S.window_oncore_cv2_range_per_mille(), Some((0, 0)));

        S.reset();
        assert_eq!(S.window_wall_cv2_per_mille(), None);
        assert_eq!(S.window_oncore_cv2_per_mille(), None);
        assert_eq!(S.window_oncore_ticks(), 0);
        // A flat window after the reset sets both ends of the wall range
        // rather than being weighed against the 40 before it.
        S.record_batch_site_only(16 * flat, 16 * sq(flat), 16, 16 * ITEMS, 16 * per_item_sq(flat));
        assert_eq!(S.window_wall_cv2_range_per_mille(), Some((0, 0)));
    }

    #[test]
    fn a_window_with_no_oncore_timing_reads_none_on_that_clock() {
        static S: CallSiteState = CallSiteState::new();
        const ITEMS: u64 = 1024;
        let (fast, slow) = (1280u64 * ITEMS, 1920u64 * ITEMS);
        let sq = |ns: u64| (ns >> 8).saturating_mul(ns >> 8);
        let per_item_sq = |ns: u64| {
            ((ns as u128).saturating_mul(ns as u128) / ((ITEMS as u128) << 16)) as u64
        };
        S.record_batch_site_only(
            8 * fast + 8 * slow,
            8 * sq(fast) + 8 * sq(slow),
            16,
            16 * ITEMS,
            8 * per_item_sq(fast) + 8 * per_item_sq(slow),
        );
        assert_eq!(S.window_wall_cv2_per_mille(), Some(40));
        assert_eq!(S.window_cv2_per_mille(), Some(40), "wall time is all the window had");
        assert_eq!(S.window_oncore_cv2_per_mille(), None);
        assert_eq!(S.window_oncore_cv2_range_per_mille(), None);
        assert_eq!(S.window_oncore_ticks(), 0);
    }

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
    fn a_reset_site_reads_as_a_fresh_one() {
        static S: CallSiteState = CallSiteState::new();
        // Teach it a depth flip, which is a counter with a reader, and
        // a depth, whose fresh value is not zero. A reset that wrote
        // zeros everywhere would pass on the counter and fail here.
        S.record_seed_depth(3);
        S.record_seed_depth(5);
        assert_eq!(S.seed_depth_flips(), 1, "the site must have learned something first");

        S.reset();

        assert_eq!(S.seed_depth_flips(), 0);
        assert_eq!(S.learned_class(), None);
        assert_eq!(S.cv2_per_mille(), None);
        assert_eq!(S.leaf_count(), 0);
        // The depth is unset again rather than zero, so the next
        // dispatch establishes one and counts no flip against the arm
        // before it.
        S.record_seed_depth(9);
        assert_eq!(S.seed_depth_flips(), 0, "the first depth after a reset flips nothing");
    }

    #[test]
    fn the_registry_says_where_each_site_is() {
        let mine = caller_site();
        let sites = registered_sites();
        let found = sites
            .iter()
            .find(|r| r.site == mine)
            .expect("a site resolved through caller_site is in the registry");
        assert!(
            found.location.file().ends_with("call_site.rs"),
            "the location is this file, not the binding's: {}",
            found.location.file()
        );
        assert!(found.location.line() > 0);
    }

    #[test]
    fn resetting_every_site_covers_the_one_just_made() {
        // Safe to run beside the other tests in this module even though
        // the sweep is process-wide: every one of them that asserts on
        // counters owns its state as a `static CallSiteState`, which is
        // never in the registry, and the one that does use caller_site
        // asserts identity and registry size, neither of which a reset
        // changes.
        let mine = caller_site();
        mine.get().record_seed_depth(2);
        mine.get().record_seed_depth(7);
        assert_eq!(mine.get().seed_depth_flips(), 1);

        let swept = reset_all_sites();
        assert!(swept >= 1, "the sweep counts the sites it reset");
        assert_eq!(mine.get().seed_depth_flips(), 0);
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
            // No item count: these samples assert on leaf-time
            // behavior, which the classifier keeps for windows that
            // carry none.
            S.record_batch_site_only(per * 16, scaled * scaled * 16, 16, 0, 0);
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
            LIGHT.record_batch_site_only(l * 16, 0, 16, 0, 0);
            let h = 1_000_000u64; // 1ms uniform: Streaming
            let hs = h >> 8;
            HEAVY.record_batch_site_only(h * 16, hs * hs * 16, 16, 0, 0);
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
        UNIFORM.record_batch_site_only(per * 8, scaled * scaled * 8, 8, 0, 0);
        // Spread: half 1us, half 100us.
        let a = 1_000u64;
        let b = 100_000u64;
        let asc = a >> 8;
        let bsc = b >> 8;
        SPREAD.record_batch_site_only(a * 4 + b * 4, asc * asc * 4 + bsc * bsc * 4, 8, 0, 0);
        assert!(UNIFORM.cv2_per_mille().unwrap() < 20);
        assert!(SPREAD.cv2_per_mille().unwrap() >= 500);
    }

    #[test]
    fn a_seed_depth_change_needs_corroboration() {
        static S: CallSiteState = CallSiteState::new();
        assert_eq!(S.seeded_depth(), None, "a fresh site has seeded nothing");
        assert_eq!(
            S.stabilise_seed_depth(5),
            5,
            "the first dispatch takes the depth its estimate asks for"
        );
        assert_eq!(
            S.stabilise_seed_depth(6),
            5,
            "one estimate the far side of a boundary does not move the depth"
        );
        assert_eq!(
            S.stabilise_seed_depth(6),
            6,
            "a second consecutive call asking the same thing does"
        );
        assert_eq!(S.seeded_depth(), Some(6));
    }

    #[test]
    fn a_run_toward_a_new_depth_restarts_when_a_third_value_intervenes() {
        static S: CallSiteState = CallSiteState::new();
        assert_eq!(S.stabilise_seed_depth(5), 5);
        assert_eq!(S.stabilise_seed_depth(6), 5);
        assert_eq!(
            S.stabilise_seed_depth(7),
            5,
            "a different candidate breaks the run toward 6"
        );
        assert_eq!(
            S.stabilise_seed_depth(6),
            5,
            "so 6 starts its run again rather than arriving already seconded"
        );
        assert_eq!(S.stabilise_seed_depth(6), 6);
    }

    #[test]
    fn a_depth_the_site_already_holds_clears_a_pending_run() {
        static S: CallSiteState = CallSiteState::new();
        assert_eq!(S.stabilise_seed_depth(5), 5);
        assert_eq!(S.stabilise_seed_depth(6), 5);
        assert_eq!(
            S.stabilise_seed_depth(5),
            5,
            "asking for what is already in force is not a change"
        );
        assert_eq!(
            S.stabilise_seed_depth(6),
            5,
            "and it broke the run, so 6 needs two agreeing calls again"
        );
        assert_eq!(S.stabilise_seed_depth(6), 6);
    }

    #[test]
    fn a_class_survives_the_same_work_split_into_different_leaf_sizes() {
        // One site, identical per-item cost throughout: 1 us an item.
        // The first windows run 1024-item leaves, the next run leaves
        // from 1 item to 512, which is what a site gets once its class
        // turns SMT on and the recursion floor drops. Leaf times then
        // span three orders of magnitude while the work has not changed
        // at all, and a classifier reading leaf times migrates on that
        // alone.
        static S: CallSiteState = CallSiteState::new();
        const PER_ITEM_NS: u64 = 1_000;

        let record = |items: u64, leaves: u64| {
            let leaf_ns = PER_ITEM_NS * items;
            let scaled = leaf_ns >> 8;
            let sq = scaled.saturating_mul(scaled);
            S.record_batch_site_only(
                leaf_ns * leaves,
                sq * leaves,
                leaves,
                items * leaves,
                (sq / items) * leaves,
            );
        };

        for _ in 0..8 {
            record(1_024, 16);
        }
        let uniform = S.learned_class();
        assert!(uniform.is_some(), "a site fed 128 leaves has classified");

        for _ in 0..8 {
            record(1, 8);
            record(64, 4);
            record(512, 4);
        }

        assert_eq!(
            S.learned_class(),
            uniform,
            "the same work in leaves of 1 to 512 items must hold the class the 1024-item \
             leaves produced; per-item cost never changed"
        );
        assert_eq!(
            S.per_item_ns(),
            Some(PER_ITEM_NS),
            "per-item cost reads the same whatever the leaves came out at"
        );
        // The scale that decides a class, not a comparison against
        // another statistic: classify_observed calls a site uniform
        // below cv2_low_per_mille and high-variance at or above
        // cv2_high_per_mille, so per-item work that never varied has to
        // read under the first of those.
        let per_item_cv2 = S.per_item_cv2_per_mille().expect("items were recorded");
        let low = crate::sched::adaptive_profile::class_thresholds()
            .cv2_low_per_mille
            .load(Ordering::Relaxed);
        assert!(
            per_item_cv2 < low,
            "per-item spread {per_item_cv2} must read as uniform, under {low}, on work whose \
             per-item cost never changed"
        );
    }

    #[test]
    fn routing_arm_adopts_the_caller_plan_when_the_class_routing_is_slower() {
        // The case the measurement exists for: a site whose class has
        // stopped describing its work, so the routing that class
        // re-derives costs more than the plan the caller built. The
        // class is not consulted here at all - only what each arm's
        // dispatches took.
        static S: CallSiteState = CallSiteState::new();
        for _ in 0..4 {
            S.record_routing_arm(PolicyArm::Default, 900_000);
            S.record_routing_arm(PolicyArm::Alternative, 300_000);
        }

        // Two cadences of calls, taken from the constant rather than a
        // number chosen here: the trial fires on every
        // ARM_TRIAL_CADENCE-th selection, so a shorter run can miss it
        // entirely and say nothing about whether it fires at all.
        let calls = 2 * ARM_TRIAL_CADENCE;
        let mut caller_plan = 0;
        let mut class_routing = 0;
        for _ in 0..calls {
            match S.choose_routing_arm() {
                PolicyArm::Alternative => caller_plan += 1,
                PolicyArm::Default => class_routing += 1,
            }
        }
        assert!(
            caller_plan > class_routing,
            "the faster arm must win the routing: caller plan {caller_plan}, class routing \
             {class_routing} over {calls} calls"
        );
        assert_eq!(
            class_routing, 2,
            "the slower arm runs once per cadence of {ARM_TRIAL_CADENCE} and no more often, \
             so a routing that becomes the better one again is found without paying for it \
             every dispatch"
        );
        let (default_ns, alternative_ns) = S.routing_ewmas();
        assert!(
            alternative_ns < default_ns,
            "the arms report what they cost: class routing {default_ns} ns, caller plan \
             {alternative_ns} ns"
        );
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
    fn split_share_outgrows_a_cold_first_sample_within_eight() {
        static S: CallSiteState = CallSiteState::new();
        // A cold first round: CPU 32 us/item, backend 18 us/item.
        S.record_split(512, 512 * 32_000, 512, 512 * 18_000);
        assert!(S.split_cpu_share_per_mille() < 400, "cold share {}", S.split_cpu_share_per_mille());
        // Seven warm rounds: CPU 10 us/item, backend 25 us/item.
        for _ in 0..7 {
            S.record_split(500, 500 * 10_000, 500, 500 * 25_000);
        }
        // The eighth average weights every sample equally: CPU 12.75,
        // backend 24.1, a share of 654; an exponential average from
        // the first sample would still read about 500 here.
        let share = S.split_cpu_share_per_mille();
        assert!((620..=700).contains(&share), "share {share}");
        // Later samples move exponentially toward the warm ratio.
        for _ in 0..8 {
            S.record_split(500, 500 * 10_000, 500, 500 * 25_000);
        }
        assert!(S.split_cpu_share_per_mille() > share, "share {}", S.split_cpu_share_per_mille());
    }

    #[test]
    fn site_ref_identity_semantics() {
        static A: CallSiteState = CallSiteState::new();
        static B: CallSiteState = CallSiteState::new();
        assert_eq!(SiteRef::new(&A), SiteRef::new(&A));
        assert_ne!(SiteRef::new(&A), SiteRef::new(&B));
    }
}

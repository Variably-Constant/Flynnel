//! Background steal-rate observer that tunes the `par_iter`
//! split-budget multiplier at runtime.
//!
//! A split budget of `workers * multiplier` is a heuristic, and this
//! multiplier is the one it scales by. When the pool is under heavy
//! contention, with a lot of cross-worker stealing, workers benefit
//! from a higher budget: more leaves spawned means finer steal
//! granularity. When the pool is mostly running owner-local work, a
//! higher budget only adds dispatch overhead with no distribution
//! benefit, so a lower one wins.
//!
//! The observer thread (hosted on the
//! [`crate::sched::io_pool::IoPool`]) samples per-worker stats
//! every 200ms, computes a global steal rate, and writes the
//! multiplier into a process-wide AtomicU32.
//!
//! What reads it is [`crate::sched::plan::JobPlan::effective_leaves_per_worker`],
//! and only for a plan whose caller did not pin a factor of their own.
//! That covers the budget-shaped dispatch entries: the triple, indexed
//! and collect helpers, the token-bucket and reduce paths, and
//! `for_each_chunk`'s probe and pinned-variant routes.
//!
//! It does not cover every dispatch. `for_each_chunk` and
//! `for_each_chunk_indexed` seed from `adaptive_seed_depth` when the
//! plan carries an authoritative per-item estimate, and that route
//! derives its leaf count from the estimate and from observed steals
//! without consulting this multiplier at all.
//!
//! ## Steal rate definition
//!
//! ```text
//! total_steals = sum over workers of stats.peer_steal_hits
//! total_pops   = sum over workers of stats.local_pops
//! steal_rate   = total_steals / (total_steals + total_pops)
//! ```
//!
//! Interpretation:
//! - `steal_rate >= 0.30`: heavy contention, peers are hungry.
//!   Recommend `multiplier = 4`.
//! - `steal_rate in [0.05, 0.30)`: moderate contention. Keep
//!   `multiplier = 2` (the baseline).
//! - `steal_rate < 0.05`: low contention. Recommend
//!   `multiplier = 1`.
//!
//! ## Activation
//!
//! Off when [`crate::sched::io_pool::global_io_pool`] is None
//! (`FLYNNEL_SCHED_SMT_AS_IO=on` to enable). Call
//! [`spawn_observer`] once at startup to begin sampling.
//!

use core::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use crate::sched::arena::global_local_arena;
use crate::sched::io_pool::global_io_pool;

/// Multiplier (in units of `workers`) used by
/// [`crate::sched::par_iter::for_each_chunk`] to size its initial
/// split budget. Updated by the observer; read on the hot path.
///
/// Default `2` matches the original constant. The observer picks
/// from `{1, 2, 4}` per the steal-rate and cv^2 thresholds in
/// [`sample_and_compute`], and the final value is clamped to
/// `[1, 8]` (the max of the two axes, capped at 8). Manual
/// callers via [`set_split_multiplier`] can set any value in the
/// same `[1, 8]` clamp range.
static SPLIT_MULTIPLIER: AtomicU32 = AtomicU32::new(2);

// ---------------------------------------------------------------------------
// Per-leaf execution-time variance tracking.
// ---------------------------------------------------------------------------
//
// Steal-rate is an indirect proxy for chunk-time variance: workers
// idling = peers waiting on long chunks. But it conflates "fewer
// chunks than workers" with "uneven chunks", and the proxy lags by
// 200ms (the observer's sample period). Direct per-leaf-time
// variance is a leading indicator: a single bisect call can pick up
// leaves taking 10x different time within its own lifetime.
//
// Three counters, summed atomically across workers:
//   LEAF_TIME_SUM_NS  - sum of leaf execution times (nanoseconds)
//   LEAF_TIME_SUMSQ   - sum of (leaf_time_ns / 256)^2 (scaled to
//                       avoid u64 overflow at large workloads)
//   LEAF_COUNT        - number of leaves recorded
//
// The observer derives coefficient of variation cv^2 =
// (sumsq/N - (sum/N)^2) / (sum/N)^2 in fixed-point and bumps the
// multiplier on high-cv runs.
//
// Cost on the hot path: 3 atomic Relaxed adds per leaf. At
// MIN_LEAF_ITEMS = 256 and a 1k-100k item workload that's 4-400
// atomic adds per bisect call, well below the leaf body cost.

use core::sync::atomic::AtomicU64;

static LEAF_TIME_SUM_NS: AtomicU64 = AtomicU64::new(0);
static LEAF_TIME_SUMSQ: AtomicU64 = AtomicU64::new(0);
static LEAF_COUNT: AtomicU64 = AtomicU64::new(0);

// Items the recorded leaves covered, and the summed per-item squared
// time. A leaf's time divided by its item count is what the classifier
// can read as a property of the work: leaf times alone move when the
// scheduler changes how finely it splits, and that split is a decision
// the classifier's own output drives.
static LEAF_ITEMS: AtomicU64 = AtomicU64::new(0);
static LEAF_SUMSQ_PER_ITEM: AtomicU64 = AtomicU64::new(0);


/// Record one leaf's execution time. Called by `par_iter` leaves
/// after the body runs. Cheap: 3 Relaxed atomic adds.
///
/// `nanos` is the wall-clock elapsed time. Implementations should
/// bracket the leaf body with `read_tsc()` and convert via the
/// cpu_topology's measured TSC-to-ns ratio, OR use Instant on
/// non-x86 hosts. The conversion need not be exact for the
/// variance to be useful: relative magnitude across leaves is what
/// matters.
///
/// Hot-path callers in `par_iter::record_leaf` accumulate samples
/// in a thread-local buffer and call [`record_leaf_batch`] every
/// `LocalLeafBuffer::FLUSH_THRESHOLD` leaves instead of calling
/// this directly per leaf (the threshold is currently 4; the
/// trade-off between per-leaf atomic cost and classifier-
/// convergence latency is documented on that constant). Batching
/// drops the per-leaf cost from ~100ns (3 contended atomic
/// fetch_adds on a 16-worker host) to ~30ns.
#[inline]
pub fn record_leaf_time_ns(nanos: u64) {
    LEAF_TIME_SUM_NS.fetch_add(nanos, Ordering::Relaxed);
    let scaled = nanos >> 8;
    LEAF_TIME_SUMSQ.fetch_add(scaled.saturating_mul(scaled), Ordering::Relaxed);
    LEAF_COUNT.fetch_add(1, Ordering::Relaxed);
    // A bare time carries no item count, so it feeds the leaf
    // statistics and leaves the per-item ones untouched rather than
    // entering them as one item costing the whole leaf.
}

/// Record a batch of leaf-time samples, pre-aggregated by a
/// caller-side accumulator. Used by the per-thread sample buffer
/// in `par_iter::record_leaf` to amortize the cost of the three
/// global atomic fetch_adds across many leaves.
///
/// Mathematically equivalent to calling [`record_leaf_time_ns`]
/// once per leaf, but with the same atomic-contention cost paid
/// once per batch instead of once per leaf.
#[inline]
pub fn record_leaf_batch(
    sum_ns: u64,
    sumsq_scaled: u64,
    count: u64,
    items: u64,
    sumsq_per_item: u64,
) {
    LEAF_TIME_SUM_NS.fetch_add(sum_ns, Ordering::Relaxed);
    LEAF_TIME_SUMSQ.fetch_add(sumsq_scaled, Ordering::Relaxed);
    LEAF_ITEMS.fetch_add(items, Ordering::Relaxed);
    LEAF_SUMSQ_PER_ITEM.fetch_add(sumsq_per_item, Ordering::Relaxed);
    let prior = LEAF_COUNT.fetch_add(count, Ordering::Relaxed);
    // Drive the auto-classify-and-migrate observer at every
    // AUTO_CLASSIFY_QUANTUM-th leaf so the routing decision tracks
    // the workload's actual shape without the application calling
    // migrate_workload_class. Cheap (~2 atomic loads + a few
    // arithmetic ops + maybe one Release-store when classification
    // disagrees with active class for K rounds).
    let new_total = prior.wrapping_add(count);
    if (prior / AUTO_CLASSIFY_QUANTUM) != (new_total / AUTO_CLASSIFY_QUANTUM) {
        crate::sched::adaptive_profile::tick_auto_classify();
    }
}

/// How many recorded leaves between auto-classifier ticks. Each
/// tick reads the global LeafStats snapshot, runs
/// [`crate::sched::adaptive_profile::classify_observed`], and
/// applies the K-consecutive-disagreement migration policy. Set
/// to 16 to give the classifier enough samples to compute a
/// stable cv^2 estimate per tick. Smaller quanta (4 leaves) made
/// the classifier flap on noise -- observed cv^2 of a 4-sample
/// window can swing 10x between adjacent buckets even when the
/// true workload shape is uniform.
pub const AUTO_CLASSIFY_QUANTUM: u64 = 16;

/// Process-global Mutex used by tests that mutate the LEAF_*
/// stats so concurrent tests serialize against each other. Both
/// `adaptive_profile` and `par_iter` test modules acquire this
/// before resetting / inspecting the global counters.
#[cfg(test)]
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the leaf-stats test serialization mutex. Returns a
/// guard that releases on drop. Poison-tolerant.
#[cfg(test)]
pub fn acquire_test_lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// Reset the variance counters. Used by tests and benches; the
/// observer also resets after each sample so consecutive samples
/// measure independent windows.
#[inline]
pub fn reset_leaf_stats() {
    LEAF_TIME_SUM_NS.store(0, Ordering::Relaxed);
    LEAF_TIME_SUMSQ.store(0, Ordering::Relaxed);
    LEAF_COUNT.store(0, Ordering::Relaxed);
    LEAF_ITEMS.store(0, Ordering::Relaxed);
    LEAF_SUMSQ_PER_ITEM.store(0, Ordering::Relaxed);
}

/// Snapshot of the variance counters. Used by the observer + tests.
#[derive(Copy, Clone, Debug)]
pub struct LeafStats {
    /// Number of leaves observed.
    pub count: u64,
    /// Sum of leaf wall-clock times (nanoseconds).
    pub sum_ns: u64,
    /// Sum of squared leaf times in fixed-point scale; the unit
    /// matches the `cv2_fixed` consumer's expected scaling.
    pub sumsq_scaled: u64,
    /// Items the observed leaves covered. Zero when no recorded leaf
    /// carried a count, which the per-item readings report as no
    /// reading rather than as zero cost.
    pub items: u64,
    /// Sum over leaves of a leaf's squared time divided by its item
    /// count, in the same fixed-point scale as `sumsq_scaled`.
    pub sumsq_per_item: u64,
}

/// Read the current variance counters atomically. Returned snapshot is
/// a point-in-time view; concurrent updates land in the next snapshot.
#[inline]
pub fn snapshot_leaf_stats() -> LeafStats {
    LeafStats {
        count: LEAF_COUNT.load(Ordering::Relaxed),
        sum_ns: LEAF_TIME_SUM_NS.load(Ordering::Relaxed),
        sumsq_scaled: LEAF_TIME_SUMSQ.load(Ordering::Relaxed),
        items: LEAF_ITEMS.load(Ordering::Relaxed),
        sumsq_per_item: LEAF_SUMSQ_PER_ITEM.load(Ordering::Relaxed),
    }
}

/// Mean observed leaf execution time in nanoseconds from recent
/// recorded leaves, or `None` when fewer than 4 leaves have been
/// recorded (statistically insignificant).
///
/// Used by `adaptive_min_leaf` in `crate::sched::par_iter` to size
/// the leaf floor for callers that did not supply an explicit
/// per-item ns hint. Observer-driven: the first dispatch on a
/// fresh process pays the classifier-default min_leaf, but
/// subsequent dispatches (criterion iter loops, multi-call
/// batches, NMFD repeated calls) use the observed mean to pick
/// a sharper granularity.
///
/// Returns the leaf-level mean rather than the per-item mean: observer
/// does not track items-per-leaf, so callers compare against
/// per-leaf overhead thresholds rather than dividing by leaf size.
#[inline]
pub fn observed_mean_leaf_ns() -> Option<u64> {
    let stats = snapshot_leaf_stats();
    if stats.count < 4 {
        return None;
    }
    Some(stats.sum_ns / stats.count)
}

/// Mean cost of one item across the recorded leaves, in nanoseconds, or
/// `None` below 4 leaves and when no leaf carried an item count.
///
/// The process-wide counterpart of
/// [`crate::sched::call_site::CallSiteState::per_item_ns`], and what
/// the classifier's per-item bands are stated in.
#[inline]
pub fn observed_per_item_ns(stats: LeafStats) -> Option<u64> {
    if stats.count < 4 || stats.items == 0 {
        return None;
    }
    Some(stats.sum_ns / stats.items)
}

/// cv^2 per mille of per-item cost, weighted by the items each leaf
/// covered, or `None` on the same terms as [`observed_per_item_ns`].
///
/// Unlike [`leaf_cv_squared_per_mille`] this does not move when the
/// scheduler splits the same work into leaves of different sizes, which
/// is what lets a class describe the workload rather than the split.
pub fn per_item_cv_squared_per_mille(stats: LeafStats) -> Option<u64> {
    if stats.count < 4 || stats.items == 0 {
        return None;
    }
    let mean_scaled = (stats.sum_ns >> 8) / stats.items;
    if mean_scaled == 0 {
        return Some(0);
    }
    let mean_sq = mean_scaled.saturating_mul(mean_scaled);
    let spread = stats
        .sumsq_per_item
        .saturating_sub(mean_sq.saturating_mul(stats.items));
    // Over the items, not the leaves: each leaf contributed its squared
    // time over its item count, so the subtraction leaves an
    // item-weighted sum of squared deviations.
    let var = spread / stats.items;
    Some(var.saturating_mul(1000) / mean_sq.max(1))
}

/// Coefficient of variation squared (cv^2 = var/mean^2) of leaf times
/// in fixed-point parts-per-1000. Returns `None` when fewer than 4
/// leaves have been recorded (statistically insignificant).
///
/// It is the spread of whole leaves, so it moves when the same work is
/// split into leaves of different sizes. That is what the split
/// multiplier wants to know, and it is what a class must not be decided
/// on, since the class decides the split;
/// [`per_item_cv_squared_per_mille`] is the reading that holds still.
///
/// cv^2 interpretation:
/// - `0..50` (cv < ~0.22): leaves are nearly uniform. Static SLAW
///   budget is optimal; multiplier=1 is fine.
/// - `50..500` (0.22 <= cv < ~0.71): moderate spread. Baseline
///   multiplier=2 is the right call.
/// - `>= 500` (cv >= ~0.71): high variance. Recommend multiplier=4
///   so steal pressure can rebalance long leaves.
pub fn leaf_cv_squared_per_mille(stats: LeafStats) -> Option<u64> {
    if stats.count < 4 {
        return None;
    }
    let n = stats.count;
    let mean_scaled = (stats.sum_ns >> 8) / n;
    if mean_scaled == 0 {
        return Some(0);
    }
    let sumsq_per_n = stats.sumsq_scaled / n;
    let mean_sq = mean_scaled.saturating_mul(mean_scaled);
    let var = sumsq_per_n.saturating_sub(mean_sq);
    Some(var.saturating_mul(1000) / mean_sq.max(1))
}

/// Read the current split-budget multiplier. Cheap atomic load
/// (single instruction on x86 with Relaxed ordering).
#[inline]
pub fn split_multiplier() -> u32 {
    SPLIT_MULTIPLIER.load(Ordering::Relaxed)
}

/// Set the split multiplier manually. Intended for tests or for
/// callers that want to lock the multiplier to a known value.
/// In production, [`spawn_observer`] handles updates.
pub fn set_split_multiplier(value: u32) {
    let clamped = value.clamp(1, 8);
    SPLIT_MULTIPLIER.store(clamped, Ordering::Relaxed);
}

/// Spawn the background observer. The first call schedules an
/// ever-self-respawning task on the IoPool; subsequent calls are
/// no-ops (the `OnceLock` ensures single-instance semantics).
///
/// No-op when the IoPool is disabled - the heuristic baseline
/// multiplier remains.
///
/// The observer task body:
/// 1. Sleep 200ms (the sampling interval).
/// 2. Sum stats across all workers via
///    [`crate::sched::arena_numa::NumaArena::iter_worker_stats`].
/// 3. Compute steal rate; update SPLIT_MULTIPLIER.
/// 4. Resubmit itself to the IoPool so the next sample fires in
///    200ms.
///
/// The resubmit pattern avoids holding an IoPool thread captive in
/// a loop (which would tie up that worker for the program's
/// lifetime). Each sample run is short (sleep + ~8 atomic loads +
/// 1 atomic store) so even if it lands on a different IoPool
/// thread each time, cache locality is minimal.
pub fn spawn_observer() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        if global_io_pool().is_none() {
            // SMT_AS_IO disabled; keep heuristic baseline.
            return;
        }
        schedule_next_sample();
    });
}

fn schedule_next_sample() {
    let Some(pool) = global_io_pool() else { return };
    pool.submit(|| {
        std::thread::sleep(Duration::from_millis(200));
        let new_mult = sample_and_compute();
        SPLIT_MULTIPLIER.store(new_mult, Ordering::Relaxed);
        // Tail-respawn: schedule the next sample.
        schedule_next_sample();
    });
}

/// One sampling pass. Reads counters, computes steal rate,
/// derives the new multiplier. Public for testing.
pub fn sample_and_compute() -> u32 {
    let arena = global_local_arena();
    let mut total_pops: u64 = 0;
    let mut total_steals: u64 = 0;
    for stats in arena.iter_worker_stats() {
        total_pops = total_pops.saturating_add(
            stats.local_pops.load(Ordering::Relaxed),
        );
        total_steals = total_steals.saturating_add(
            stats.peer_steal_hits.load(Ordering::Relaxed),
        );
    }
    let total = total_pops.saturating_add(total_steals);
    if total < 100 {
        // Insufficient data: keep current multiplier.
        return SPLIT_MULTIPLIER.load(Ordering::Relaxed);
    }
    // Compute steal_rate in fixed-point (parts per 1000 to avoid
    // float in the observer hot path).
    let steal_per_mille: u64 = (total_steals * 1000) / total;
    let from_steal = if steal_per_mille >= 300 {
        4u32
    } else if steal_per_mille >= 50 {
        2u32
    } else {
        1u32
    };

    // Per-leaf variance: orthogonal signal. High variance means
    // some leaves take much longer than others, even if the
    // steal-rate looks low because workers are individually busy.
    // Bumping the multiplier here gives more leaves, which lets
    // steals rebalance the long ones onto idle workers.
    let stats = snapshot_leaf_stats();
    let from_variance = match leaf_cv_squared_per_mille(stats) {
        Some(cv2) if cv2 >= 500 => 4u32,
        Some(cv2) if cv2 >= 50 => 2u32,
        Some(_) => 1u32,
        None => from_steal,
    };

    // Reset for the next sample window so we don't double-count.
    reset_leaf_stats();

    // Combine: take the max of the two signals so a high reading
    // on either axis triggers a bump. Cap at 8 to keep the budget
    // bounded.
    from_steal.max(from_variance).min(8)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_multiplier_starts_at_baseline_two() {
        // Test isolation: this test reads the process-global atomic so
        // run-order can affect it. Allow either 2 (baseline) or
        // whatever set_split_multiplier left it at from earlier
        // tests in the same process.
        let m = split_multiplier();
        assert!((1..=8).contains(&m), "multiplier should be in [1, 8]");
    }

    #[test]
    fn set_split_multiplier_clamps_to_valid_range() {
        set_split_multiplier(100);
        assert_eq!(split_multiplier(), 8);
        set_split_multiplier(0);
        assert_eq!(split_multiplier(), 1);
        // Restore baseline so we don't pollute other tests.
        set_split_multiplier(2);
    }

    #[test]
    fn spawn_observer_noop_when_pool_disabled() {
        // FLYNNEL_SCHED_SMT_AS_IO is not set in the test env;
        // global_io_pool() returns None and spawn_observer is
        // a no-op.
        let t0 = std::time::Instant::now();
        spawn_observer();
        assert!(
            t0.elapsed() < Duration::from_millis(50),
            "spawn_observer should return immediately"
        );
    }

    #[test]
    fn leaf_stats_record_and_reset() {
        reset_leaf_stats();
        record_leaf_time_ns(1_000);
        record_leaf_time_ns(2_000);
        record_leaf_time_ns(3_000);
        let s = snapshot_leaf_stats();
        assert_eq!(s.count, 3);
        assert_eq!(s.sum_ns, 6_000);
        reset_leaf_stats();
        let s2 = snapshot_leaf_stats();
        assert_eq!(s2.count, 0);
        assert_eq!(s2.sum_ns, 0);
        assert_eq!(s2.sumsq_scaled, 0);
    }

    #[test]
    fn leaf_cv_squared_handles_insufficient_samples() {
        reset_leaf_stats();
        for _ in 0..3 {
            record_leaf_time_ns(1_000);
        }
        let s = snapshot_leaf_stats();
        assert!(leaf_cv_squared_per_mille(s).is_none(),
            "fewer than 4 leaves should return None");
        reset_leaf_stats();
    }

    #[test]
    fn leaf_cv_squared_zero_for_uniform_leaves() {
        reset_leaf_stats();
        // 8 leaves, identical time = 0 variance.
        for _ in 0..8 {
            record_leaf_time_ns(10_000);
        }
        let s = snapshot_leaf_stats();
        let cv2 = leaf_cv_squared_per_mille(s).unwrap();
        // Allow small fixed-point rounding error from the >>8 scaling.
        assert!(cv2 < 20, "uniform leaves should have cv^2 ~ 0; got {cv2}");
        reset_leaf_stats();
    }

    #[test]
    fn leaf_cv_squared_high_for_spread_leaves() {
        reset_leaf_stats();
        // Mix of fast and slow leaves: half at 1us, half at 100us.
        // cv = sqrt(((100-50.5)^2 + (1-50.5)^2)/2) / 50.5 ~ 49.5/50.5 ~ 0.98.
        // cv^2 ~ 0.96 ~ 960 per mille.
        for _ in 0..4 {
            record_leaf_time_ns(1_000);
        }
        for _ in 0..4 {
            record_leaf_time_ns(100_000);
        }
        let s = snapshot_leaf_stats();
        let cv2 = leaf_cv_squared_per_mille(s).unwrap();
        assert!(cv2 >= 500, "spread leaves should have cv^2 >= 500; got {cv2}");
        reset_leaf_stats();
    }

    #[test]
    fn sample_returns_current_multiplier_when_no_data() {
        // Reset the global arena's per-worker counters first so we
        // genuinely have <100 total events at sample time. Earlier
        // parallel tests in the suite leave non-zero counters
        // behind otherwise (the arena is process-global), which
        // would push total past the <100 short-circuit.
        for stats in global_local_arena().iter_worker_stats() {
            stats.local_pops.store(0, Ordering::Relaxed);
            stats.peer_steal_hits.store(0, Ordering::Relaxed);
            stats.peer_steal_misses.store(0, Ordering::Relaxed);
        }
        set_split_multiplier(3);
        let computed = sample_and_compute();
        assert_eq!(computed, 3);
        set_split_multiplier(2);
    }
}

//! Parallel-iterator primitives: `for_each_chunk`, plus a thin
//! reduction helper.
//!
//! For fine-grain workloads (`FpN<N>` at K = 5..7, ~100 ns - 1 us
//! per op) submitting per-item jobs through `sched::join` would
//! pay scheduler overhead larger than the actual work. The right
//! pattern is to recursively bisect the input until chunks are at
//! a saturating "leaf size", then run a serial loop inside each
//! leaf so SIMD has a continuous run to operate on.
//!
//! Recursive bisection through [`crate::sched::join`] gives us:
//! - log2(n / leaf_size) recursion depth
//! - One `sched::join` per internal split
//! - Cooperative work-stealing during waits (free from the Local
//!   tier dispatch)
//! - Linear speedup up to `worker_count()` cores

use crate::sched::arena::{global_local_arena, join_context};
use crate::sched::plan::JobPlan;

/// Heartbeat interval in CPU cycles. ~20µs at 3 GHz, below the
/// 67µs canonical heartbeat value: the hybrid design hands the
/// tail off to SLAW on the first tick (no further heartbeat
/// recursion), so faster ticks mean an earlier handover and more
/// of the work runs in parallel.
///
/// At 20µs:
/// - matmul 16x16 (~25µs total serial) just barely fires one
///   heartbeat - fans the tiny tail through SLAW.
/// - matmul 32x32 (~1ms serial) fans out at iter ~20 of 1024;
///   97% of work runs through SLAW's parallel bisect.
/// - matmul 64x64 (~13ms serial) fans out at iter ~20 of 4096;
///   99% parallel.
const HEARTBEAT_CYCLES: u64 = 60_000;

/// How often to check the heartbeat timer. `iter_count & POLL_MASK
/// == 0` triggers an rdtsc read. 0x1F = every 32 iterations,
/// tight enough that the first tick lands soon after the
/// heartbeat quantum. rdtsc cost is
/// ~20-30 cycles on Zen3; checking every 32 iters of 1us work is
/// 0.05% overhead, well below the bench noise floor.
const POLL_MASK: usize = 0x1F;

/// Helper: RAII guard that emits a `DispatchExit` trace event when
/// dropped, regardless of which return path `for_each_chunk` takes.
fn scopeguard_dispatch_exit(n: usize) -> DispatchExitGuard {
    DispatchExitGuard { n: n as u32 }
}

struct DispatchExitGuard {
    n: u32,
}

impl Drop for DispatchExitGuard {
    fn drop(&mut self) {
        crate::sched::trace::emit(
            crate::sched::trace::TraceEvent::DispatchExit,
            self.n,
        );
    }
}

#[inline(always)]
fn read_tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_rdtsc` is part of the base x86_64 ISA (Pentium
    // and newer). The intrinsic has no CPU-feature preconditions
    // and no architectural preconditions on operand state.
    unsafe {
        std::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Non-x86 fallback: use Instant. Slower but portable. The
        // heartbeat scheduler will still work; only the
        // amortization-bound constants differ.
        std::time::Instant::now().elapsed().as_nanos() as u64
    }
}

/// Bracket a leaf-body invocation with TSC reads and record the
/// delta. Every bisect-leaf emit site uses this so the
/// [`crate::sched::split_observer`] and the site classifier can
/// derive per-leaf variance (cv^2 is unit-invariant, so the
/// approximate-ns TSC delta suffices). Samples batch in a
/// thread-local buffer ([`LocalLeafBuffer::FLUSH_THRESHOLD`] = 4)
/// before flushing: unbatched, the three global fetch_adds cost
/// ~100ns per leaf on a 16-worker host; batched, ~30ns (2 TSC
/// reads + thread-local access).
#[inline(always)]
fn record_leaf<F: FnOnce() -> R, R>(
    site: Option<crate::sched::call_site::SiteRef>,
    body: F,
) -> R {
    crate::sched::trace::emit(crate::sched::trace::TraceEvent::LeafStart, 0);
    let t0 = read_tsc();
    let out = body();
    let dt = read_tsc().wrapping_sub(t0);
    crate::sched::trace::emit(crate::sched::trace::TraceEvent::LeafEnd, 0);
    LOCAL_LEAF_BUFFER.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.add(site, dt);
    });
    out
}

/// Record a pre-measured serial-span duration against `site`
/// without bracketing a body. Used by the heartbeat and
/// token-bucket fillers, whose "leaves" are the serial spans
/// between promotions rather than closure invocations.
///
/// Spans feed the SITE's statistics only, never the process-global
/// counters: a span is a whole-stretch wall time (heartbeat quantum
/// and up), and mixing it into the global classifier's per-item-ns
/// boundaries would migrate the process profile off unrelated
/// workloads. With no site attached the sample is dropped.
#[inline(always)]
pub(crate) fn record_leaf_span_ns(
    site: Option<crate::sched::call_site::SiteRef>,
    nanos: u64,
) {
    if let Some(site) = site {
        let scaled = nanos >> 8;
        site.get()
            .record_batch_site_only(nanos, scaled.saturating_mul(scaled), 1);
    }
}

/// Per-thread accumulator that batches leaf-time samples before
/// flushing them onward. Two INDEPENDENT accumulations run side by
/// side:
///
/// - The **global half** feeds the process-wide observer counters
///   via [`crate::sched::split_observer::record_leaf_batch`] on a
///   fixed [`Self::FLUSH_THRESHOLD`]-leaf cadence, regardless of
///   which sites the samples belong to. Keeping this cadence
///   independent of site interleaving preserves the batch
///   granularity the global auto-classifier and the split-observer
///   statistics were tuned against.
/// - The **site half** feeds the per-call-site statistics via
///   [`crate::sched::call_site::CallSiteState::record_batch_site_only`],
///   flushing at the same threshold OR whenever a sample arrives
///   for a different site, so site batches never mix sites.
struct LocalLeafBuffer {
    global_sum_ns: u64,
    global_sumsq_scaled: u64,
    global_count: u64,
    site_sum_ns: u64,
    site_sumsq_scaled: u64,
    site_count: u64,
    /// Call site the buffered site-half samples belong to; null
    /// when the recent samples carried no site. Site changes are
    /// rare within one worker (only when it steals across
    /// concurrently-running dispatches), so the batching
    /// amortization survives keying.
    site: *const crate::sched::call_site::CallSiteState,
}

impl LocalLeafBuffer {
    // Per-worker flush threshold. 4 lets the closing-loop auto-
    // classifier observe a single workload-iteration's leaves on
    // small-N workloads (16-chunk grep, 16-chunk histogram, etc.).
    // With 1-chunk-per-worker workloads each worker produces 1
    // sample per iteration, so 4 iterations are enough to publish
    // a flush; the migration hysteresis then needs ~12 iterations
    // to converge on a stable classification -- well within
    // criterion's 3-second warm-up.
    const FLUSH_THRESHOLD: u64 = 4;

    const fn new() -> Self {
        Self {
            global_sum_ns: 0,
            global_sumsq_scaled: 0,
            global_count: 0,
            site_sum_ns: 0,
            site_sumsq_scaled: 0,
            site_count: 0,
            site: core::ptr::null(),
        }
    }

    #[inline(always)]
    fn add(&mut self, site: Option<crate::sched::call_site::SiteRef>, nanos: u64) {
        let scaled = nanos >> 8;
        let sq = scaled.saturating_mul(scaled);

        self.global_sum_ns = self.global_sum_ns.saturating_add(nanos);
        self.global_sumsq_scaled = self.global_sumsq_scaled.saturating_add(sq);
        self.global_count += 1;
        if self.global_count >= Self::FLUSH_THRESHOLD {
            self.flush_global();
        }

        if let Some(s) = site {
            let site_ptr: *const crate::sched::call_site::CallSiteState = s.get();
            if !core::ptr::eq(site_ptr, self.site) {
                self.flush_site();
                self.site = site_ptr;
            }
            self.site_sum_ns = self.site_sum_ns.saturating_add(nanos);
            self.site_sumsq_scaled = self.site_sumsq_scaled.saturating_add(sq);
            self.site_count += 1;
            if self.site_count >= Self::FLUSH_THRESHOLD {
                self.flush_site();
            }
        }
    }

    fn flush_global(&mut self) {
        if self.global_count == 0 {
            return;
        }
        crate::sched::split_observer::record_leaf_batch(
            self.global_sum_ns,
            self.global_sumsq_scaled,
            self.global_count,
        );
        self.global_sum_ns = 0;
        self.global_sumsq_scaled = 0;
        self.global_count = 0;
    }

    fn flush_site(&mut self) {
        if self.site_count == 0 || self.site.is_null() {
            self.site_count = 0;
            self.site_sum_ns = 0;
            self.site_sumsq_scaled = 0;
            return;
        }
        // SAFETY: `site` was captured from a `SiteRef`, which only
        // wraps `&'static CallSiteState`, so the pointee lives for
        // the program's lifetime.
        let site: &'static crate::sched::call_site::CallSiteState =
            unsafe { &*self.site };
        site.record_batch_site_only(
            self.site_sum_ns,
            self.site_sumsq_scaled,
            self.site_count,
        );
        self.site_sum_ns = 0;
        self.site_sumsq_scaled = 0;
        self.site_count = 0;
    }
}

impl Drop for LocalLeafBuffer {
    fn drop(&mut self) {
        // Flush any residual samples on thread exit so both stat
        // consumers stay current.
        self.flush_global();
        self.flush_site();
    }
}

thread_local! {
    static LOCAL_LEAF_BUFFER: std::cell::RefCell<LocalLeafBuffer>
        = const { std::cell::RefCell::new(LocalLeafBuffer::new()) };
}

/// Publishes the calling thread's buffered leaf samples on drop.
/// Dispatch entries hold one so short dispatches (a probe plus a
/// serial tail, or a small fan-out) feed the observer and the site
/// before returning instead of waiting for a later dispatch to
/// cross the batch threshold; without it a dispatch recording
/// fewer than FLUSH_THRESHOLD leaves on the calling thread
/// publishes nothing.
struct FlushLeafStatsOnExit;

impl Drop for FlushLeafStatsOnExit {
    fn drop(&mut self) {
        LOCAL_LEAF_BUFFER.with(|cell| {
            let mut buf = cell.borrow_mut();
            buf.flush_global();
            buf.flush_site();
        });
    }
}

/// Lower bound on chunk size for the recursive bisect. Below this
/// the bisect emits a leaf and runs `op` serially on the chunk.
///
/// Tuning evidence (rayon-crossover bench): 256 is the stable
/// point, winning at n=1k-100k and losing ~6% at n=1M (acceptable
/// trade). 64 wins 7% at n=1M heavy but regresses 5.6x at n=10k
/// heavy because the adaptive cascade (replenish-to-max on
/// migrated) creates 256-512 leaves with steal-traffic-dominated
/// coordination. 32 leaves `log2(N/32)` recursion depth to
/// dominate wherever per-join overhead is high.
const MIN_LEAF_ITEMS: usize = 256;

/// Sample stride for [`record_leaf_sampled`] used by the default
/// lazy-steal bisect path. The TSC bracket + global stat update
/// fires for every Nth leaf; the other N-1 just run the body.
/// Reduces per-leaf instrumentation from ~30ns amortized to ~4ns,
/// at the cost of cv^2 sampling granularity that the observer can
/// still compute meaningfully.
const LEAF_SAMPLE_STRIDE: u32 = 8;

/// Target per-leaf overhead used by [`adaptive_min_leaf`]. Matches
/// the probe-path budget in [`for_each_chunk`] so all wrappers agree.
const TARGET_PER_LEAF_OVERHEAD_NS_HELPER: u64 = 5_000;

/// Compute the adaptive recursion floor for bisects based on the
/// plan's per-item cost estimate. When the caller supplied an
/// authoritative hint AND each item is heavy enough that 1-item
/// leaves still amortize dispatch overhead (~5us), return 1 so the
/// bisect can split all the way down to one item per worker. For
/// non-authoritative or fine-grain hints, fall back to MIN_LEAF_ITEMS
/// (256) so chunked fine-grain workloads keep the existing per-leaf
/// amortization.
///
/// `caller_floor` is the caller-supplied minimum (typically
/// MIN_LEAF_ITEMS or 1). The returned value is `caller_floor.min(...)`
/// so callers that explicitly pass `min_leaf=1` (heavy-per-item
/// dispatch sites) always get that floor regardless of plan hints.
/// Target per-leaf wall-clock work used by [`adaptive_seed_depth`].
/// Smaller targets produce more leaves (better load balance, more
/// dispatch overhead). Tuned empirically across VM Zen3 16-thread
/// sweep + local Windows 4-thread:
///
/// 1ms target: VM Zen3 hinted scorecard 4W/5T/7L. Local Windows
/// showed 100us x 1024 splitting into 128 leaves of 8 items each
/// = 800us/leaf (mid-N regime), but VM data showed the cells
/// fluctuating in the noise floor between 1ms and 5ms targets.
/// 1ms keeps the same scorecard and matches the rayon default
/// chunk-size target more closely.
const TARGET_LEAF_WORK_NS: u64 = 1_000_000;

/// Compute the eager seed-split depth for bisect_lazy_steal_driven.
/// Returns the number of initial split levels before lazy mode
/// (steal-pressure-driven) kicks in.
///
/// Two-rule trade-off:
/// - LIGHT items (per_item < TARGET_LEAF_WORK_NS / 16):
///   log2(workers) levels. Targets workers initial leaves
///   (one per core) and minimizes dispatch overhead. Matches
///   rayon's LengthSplitter::default_count() behavior.
/// - HEAVY items (per_item >> TARGET_LEAF_WORK_NS / items.len()):
///   log2(items) levels. Targets one leaf per item for fine-grained
///   load balancing -- a slow worker only blocks one item's worth
///   of work before its remaining items can be stolen.
///
/// Formula: leaf_count = max(workers, items * per_item / TARGET_LEAF_WORK_NS),
/// capped at items.len(); seed_depth = ceil(log2(leaf_count)).
///
/// Example workloads (workers=16):
///   100us * 32 items: leaf_count = max(16, 32*100us/1ms) = max(16, 3.2) = 16
///                     -> 5 levels eager -> 16 initial leaves (good for light)
///   10ms * 128 items: leaf_count = max(16, 128*10ms/1ms) = max(16, 1280) = 128
///                     (capped at items.len()=128) -> 7 levels eager
///                     -> 128 initial leaves (good for heavy load-balance)
///
/// Without a per-item hint, fall back to log2(workers) (the
/// conservative fewest-leaves choice; lazy-steal mode kicks in
/// after to subdivide on demand).
#[inline]
fn adaptive_seed_depth(plan: &JobPlan, items: usize, workers: usize) -> usize {
    let workers = workers.max(1);
    let items = items.max(1);
    let workers_log2 = (workers as u64).next_power_of_two().trailing_zeros() as usize;
    let target_leaf_count = match plan.estimated_per_item_ns {
        Some(ns) if ns > 0 => {
            let total_work_ns = (ns as u64).saturating_mul(items as u64);
            let from_work = (total_work_ns / TARGET_LEAF_WORK_NS).max(workers as u64);
            (from_work as usize).min(items)
        }
        _ => workers,
    };
    let depth = (target_leaf_count as u64).next_power_of_two().trailing_zeros() as usize;
    depth.max(workers_log2)
}

#[inline]
fn adaptive_min_leaf(plan: &JobPlan, caller_floor: usize) -> usize {
    if !plan.estimated_per_item_ns_explicit {
        // No authoritative ns hint, but the static classifier may
        // have signaled heavy per-item work via use_smt=true
        // (LatencyBound profile). For heavy items each leaf is
        // its own item -- min_leaf=1 lets the bisect split N=5
        // NMFD-shape batches into 5 leaves so the worker pool
        // actually parallelizes. Without the min_leaf=1 floor the
        // caller_floor (typically 256) caps min_leaf above
        // batch_size and the bisect never splits: serial execution
        // on the wrap-and-park primary, measured 2x slower than
        // rayon on cold_workloads NMFD-like cells with
        // `flynnel_def` (no caller ns hint).
        if plan.use_smt {
            return 1;
        }
        // For hint-less PortBound batches, the for_each_chunk
        // probe path (lines 540+) already adaptively measures
        // per-item cost when n < workers*MIN_LEAF and feeds an
        // amended_plan with explicit ns into the bisect. Above
        // that threshold (n >= workers*MIN_LEAF, e.g. 16k items
        // on 8 workers) the static caller_floor=256 default
        // applies and the workload runs at 1.04-1.16x of rayon,
        // a bench-noise band that doesn't justify the per-leaf
        // bucketing overhead a finer adaptive sizing would
        // require. The probe path is the adaptation mechanism
        // for cells where it matters.
        return caller_floor;
    }
    match plan.estimated_per_item_ns {
        Some(ns) if ns > 0 => {
            let raw = (TARGET_PER_LEAF_OVERHEAD_NS_HELPER / ns as u64)
                .max(1) as usize;
            raw.min(caller_floor)
        }
        _ => caller_floor,
    }
}

/// Floor of [`inline_collapse_threshold_ns`]: no host collapses less
/// work than this on the calling thread.
pub const INLINE_COLLAPSE_FLOOR_NS: u64 = 50_000;

/// Cap of [`inline_collapse_threshold_ns`], sixteen floors: a host
/// whose pool never catches the serial body below it collapses up to
/// this much work and dispatches above it.
pub const INLINE_COLLAPSE_CAP_NS: u64 = INLINE_COLLAPSE_FLOOR_NS * 16;

/// Total work, in nanoseconds from the caller's explicit per-item
/// estimate, below which a data-parallel entry runs its body on the
/// calling thread instead of dispatching: this host's measured
/// crossover once [`calibrate_inline_collapse_threshold`] has run,
/// [`INLINE_COLLAPSE_FLOOR_NS`] until then. The first query starts
/// that calibration on a thread of its own and returns the floor, so
/// no call stalls on it; a process that wants the measured value
/// from its first call runs the calibration itself at start-up.
/// Classifier defaults never trigger the collapse; only
/// [`JobPlan::with_estimated_per_item_ns`] does.
pub fn inline_collapse_threshold_ns() -> u64 {
    use std::sync::atomic::Ordering;
    let measured = INLINE_COLLAPSE_MEASURED_NS.load(Ordering::Relaxed);
    if measured != 0 {
        return measured;
    }
    if !INLINE_COLLAPSE_CALIBRATING.swap(true, Ordering::AcqRel) {
        let spawned = std::thread::Builder::new()
            .name("flynnel-collapse-calibration".into())
            .spawn(|| {
                calibrate_inline_collapse_threshold();
            });
        if let Err(e) = spawned {
            // The next query tries again; until one succeeds the
            // floor stands.
            INLINE_COLLAPSE_CALIBRATING.store(false, Ordering::Release);
            eprintln!("flynnel: the collapse-threshold calibration thread did not start: {e}");
        }
    }
    INLINE_COLLAPSE_FLOOR_NS
}

/// The measured threshold, zero until the calibration has run.
static INLINE_COLLAPSE_MEASURED_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Set by whichever caller starts the calibration first.
static INLINE_COLLAPSE_CALIBRATING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Measure this host's collapse threshold now, on the calling
/// thread, and install it: a compute-bound body (eight dependent
/// multiply-rotate steps per item) over doubling item counts, serial
/// against dispatched through the pool (medians of five each), the
/// crossover being the serial time at which the dispatched body
/// finishes no later than the serial one, interpolated between the
/// last two counts; three sweeps after one discarded warm-up
/// dispatch, median taken, clamped to
/// [`INLINE_COLLAPSE_FLOOR_NS`]..=[`INLINE_COLLAPSE_CAP_NS`]. A pool
/// that never catches up below the cap yields the cap, so a
/// calibration on a loaded host collapses more work inline, never
/// less. A second call re-measures and replaces the value. Returns
/// the installed threshold; the test
/// `inline_collapse_threshold_is_in_its_band_and_cached` prints the
/// value and the cost on the running host.
pub fn calibrate_inline_collapse_threshold() -> u64 {
    use std::sync::atomic::Ordering;
    INLINE_COLLAPSE_CALIBRATING.store(true, Ordering::Release);
    let t = measure_inline_collapse_threshold_ns();
    INLINE_COLLAPSE_MEASURED_NS.store(t, Ordering::Relaxed);
    t
}

/// The sweep behind [`inline_collapse_threshold_ns`]. Dispatch goes
/// through a join bisect of its own with a fixed leaf, so the
/// measurement never consults the threshold it produces.
fn measure_inline_collapse_threshold_ns() -> u64 {
    const LEAF: usize = 256;
    const MAX_ITEMS: usize = 1 << 17;
    fn median5<F: FnMut()>(mut f: F) -> u64 {
        let mut t: [u64; 5] = [0; 5];
        for slot in t.iter_mut() {
            let t0 = std::time::Instant::now();
            f();
            *slot = t0.elapsed().as_nanos() as u64;
        }
        t.sort_unstable();
        t[2]
    }
    // Eight dependent multiply-rotate steps per item: compute-bound
    // at about eight nanoseconds an item, so the dispatched version
    // scales with cores the way the work callers estimate does. A
    // streaming body would measure the memory system instead, and on
    // a 24-thread host the pool never beat a serial stream below the
    // cap, which would have collapsed compute the pool runs eight
    // times faster.
    fn body(items: &mut [u64]) {
        for x in items.iter_mut() {
            let mut v = *x;
            for _ in 0..8 {
                v = v.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(13);
            }
            *x = v;
        }
        std::hint::black_box(&*items);
    }
    fn dispatched(plan: &JobPlan, items: &mut [u64]) {
        if items.len() <= LEAF {
            body(items);
            return;
        }
        let mid = items.len() >> 1;
        let (lo, hi) = items.split_at_mut(mid);
        join_context(plan, |_| dispatched(plan, lo), |_| dispatched(plan, hi));
    }
    // One sweep: doubling counts from four leaves; the crossover is
    // the serial time where the dispatched minus serial difference
    // reaches zero, interpolated linearly between the last two
    // counts so the doubling steps do not quantize it to a factor of
    // two. A sweep that never crosses below the cap reports the cap.
    fn sweep(v: &mut [u64]) -> u64 {
        let mut n = 4 * LEAF;
        let mut prev: Option<(u64, i64)> = None;
        loop {
            let plan = JobPlan::new(0, n as u32);
            let serial = median5(|| body(&mut v[..n]));
            let pool = median5(|| dispatched(&plan, &mut v[..n]));
            let diff = pool as i64 - serial as i64;
            if diff <= 0 {
                let crossing = match prev {
                    Some((s_prev, d_prev)) if d_prev > diff => {
                        let span = (serial - s_prev) as f64;
                        let frac = d_prev as f64 / (d_prev - diff) as f64;
                        s_prev + (span * frac) as u64
                    }
                    _ => serial,
                };
                return crossing.clamp(INLINE_COLLAPSE_FLOOR_NS, INLINE_COLLAPSE_CAP_NS);
            }
            if serial >= INLINE_COLLAPSE_CAP_NS || n >= v.len() {
                return INLINE_COLLAPSE_CAP_NS;
            }
            prev = Some((serial, diff));
            n <<= 1;
        }
    }
    let mut v: Vec<u64> = (0..MAX_ITEMS as u64).collect();
    // The first dispatches wake a cold pool; one is discarded.
    let warm = JobPlan::new(0, (4 * LEAF) as u32);
    dispatched(&warm, &mut v[..4 * LEAF]);
    let mut sweeps = [sweep(&mut v), sweep(&mut v), sweep(&mut v)];
    sweeps.sort_unstable();
    sweeps[1]
}

/// True when the caller's explicit estimate puts `n` items under
/// [`inline_collapse_threshold_ns`].
#[inline]
fn collapses_inline(plan: &JobPlan, n: usize) -> bool {
    plan.estimated_per_item_ns_explicit
        && plan.effective_ns_per_elem().is_some_and(|per| {
            (per as u64).saturating_mul(n as u64) < inline_collapse_threshold_ns()
        })
}

/// Apply `op` to every element of `items` in parallel by
/// recursively bisecting the slice. Each leaf chunk is processed
/// serially by `op`, which is the right granule for SIMD loops.
///
/// The number of leaves is bounded by `worker_count() * 2`:
/// log2-depth recursion of `sched::join` until either the chunk
/// is at most `n / target_leaves` items, or below
/// [`MIN_LEAF_ITEMS`]. Each level halves the remaining `splits`
/// budget; when the budget hits zero the recursion bottoms out
/// even if larger chunks would have been split further. This is
/// the simplified rayon `bridge` pattern: hand out roughly
/// `2 * worker_count` chunks so steals have headroom without
/// over-splitting.
///
/// `op` must be `Sync` because the same closure body is invoked
/// in parallel from multiple workers (each on its own chunk).
#[track_caller]
pub fn for_each_chunk<T, F>(plan: &JobPlan, items: &mut [T], op: F)
where
    T: Send,
    F: Fn(&mut [T]) + Sync,
{
    let n = items.len();
    crate::sched::trace::emit(
        crate::sched::trace::TraceEvent::DispatchEnter,
        n as u32,
    );
    let _trace_exit_guard = scopeguard_dispatch_exit(n);
    if n == 0 {
        return;
    }
    // Per-call-site identity from the caller's source location
    // (track_caller chain). An outer entry's attachment (or a
    // caller's explicit with_site) wins.
    let plan_owned = plan
        .with_site_if_none(crate::sched::call_site::caller_site())
        .apply_site_class();
    let plan = &plan_owned;
    let _flush_on_exit = FlushLeafStatsOnExit;
    // Hybrid JEC threshold: at small total work, the JEC counter
    // CAS + wake-from-condvar cost (~134us per dispatch on Genoa
    // 44T at small N, measured 2026-06-05) outweighs the
    // structural wake-cascade benefit. Below 200us estimated
    // total, skip JEC wake notifications and let workers find
    // pushed work via their spin-loop polling (safe because
    // `ROUNDS_UNTIL_SLEEPING` keeps workers spinning across
    // typical inter-dispatch gaps). Above 200us, take the full
    // JEC path - the wake-cascade win dominates.
    //
    const HYBRID_JEC_THRESHOLD_NS: u64 = 200_000;
    let estimated_total_ns: u64 = plan
        .effective_ns_per_elem()
        .map(|ns| (ns as u64).saturating_mul(n as u64))
        // Unknown cost: assume large workload, take JEC path
        // (safe default - JEC is correct for any size, just
        // higher overhead on small).
        .unwrap_or(u64::MAX);
    let use_jec_wake = estimated_total_ns >= HYBRID_JEC_THRESHOLD_NS;
    let _jec_scope = crate::sched::arena_local::DispatchScope::new_if_change(use_jec_wake);

    // Inline-collapse fast path: when the caller has supplied an
    // AUTHORITATIVE per-item cost (via with_estimated_per_item_ns or
    // with_cost_ns_per_elem) AND total work is under this host's
    // measured collapse threshold, skip the pool entirely.
    //
    // Gating on plan.estimated_per_item_ns_explicit (not just
    // effective_ns_per_elem) is critical: JobPlan::new auto-populates
    // estimated_per_item_ns with the classifier default 12 / 50 /
    // 600 ns. For N=4 BigFloat-heavy items the classifier default
    // (12 * 4 = 48ns < 50us) would falsely trigger inline-collapse
    // even though each item is 10ms of real compute. The probe path
    // downstream measures actual cost; let it run instead of
    // shortcutting here based on a guess.
    if collapses_inline(plan, n) {
        record_leaf(plan.site, || op(items));
        return;
    }

    let arena_workers = global_local_arena().total_workers();
    // Resolve worker count: cap to plan.worker_cap if set.
    let workers = plan
        .worker_cap
        .map(|cap| (cap as usize).min(arena_workers))
        .unwrap_or(arena_workers)
        .max(1);

    // Adaptive min_leaf. The default MIN_LEAF_ITEMS=256 floor caps
    // chunk count for fine-grain ops (~10ns/elem) where per-leaf
    // dispatch overhead dominates below 256-item chunks. For HEAVY
    // per-element work (BigFloat mul, sqrt-chains, FMA-heavy
    // kernels), a 256-item floor SERIALIZES the small-N case:
    // N=4 items each taking 10ms can fully saturate 4 cores at
    // 1-item leaves, but the 256-floor would force inline.
    //
    // Formula: effective_min_leaf = max(1, target_per_leaf_ns / per_item_ns),
    // capped at MIN_LEAF_ITEMS. When per-item-cost is unknown,
    // fall back to the conservative default.
    //
    // Examples (target_per_leaf_overhead_ns = 5us):
    //   per_item =     10ns -> floor = max(1, 500)   capped to 256
    //   per_item =     50ns -> floor = max(1, 100)   = 100
    //   per_item =    500ns -> floor = max(1, 10)    = 10
    //   per_item =  10000ns -> floor = max(1, 0)     = 1
    //
    // The cap-at-MIN_LEAF_ITEMS preserves the fine-grain-default
    // behavior; the lower-bound-at-1 unlocks the small-N + heavy
    // case where rayon already parallelizes and flynnel was
    // serializing.
    let effective_min_leaf = adaptive_min_leaf(plan, MIN_LEAF_ITEMS);

    // Probe-and-decide floor: workers * 5us. Empirically tuned for
    // Zen+ R7 2700 / Zen3 5700G / Xeon Cascade Lake / EPYC Genoa
    // (the four hosts in the bench matrix); higher floors regress
    // Genoa Compute/10k by 12.7x via under-dispatching the wide pool.
    const TARGET_PER_LEAF_OVERHEAD_NS: u64 = 5_000;
    let target_per_leaf_overhead_ns: u64 = TARGET_PER_LEAF_OVERHEAD_NS;

    // Probe-and-decide path: when the caller hasn't given us a cost
    // estimate AND the workload is small relative to the worker
    // count (n < workers * MIN_LEAF_ITEMS, meaning we can't even
    // make one MIN_LEAF-sized chunk per worker), we don't know
    // whether this work is dispatch-dominated (Light bench) or
    // genuinely parallelizable (Heavy bench).
    //
    // Probe a tiny leaf inline, measure actual per-element cost,
    // then decide:
    //   * estimated_tail_work < workers * 5us  -> finish inline
    //     serially (skip the pool entirely).
    //   * otherwise                            -> bisect the tail
    //     with the observed cost as the leaves_per_worker driver.
    //
    // Gated on `plan.oversubscription_log2.is_none()` too: if the
    // caller explicitly set an oversub override they opted in to
    // a specific dispatch shape and we honor it. The probe is
    // 32 items - small enough that the overhead is in the
    // single-digit-microsecond range even on Heavy ops (~19us
    // for 600ns/elem), but big enough that per-call function
    // overhead doesn't dominate the timing.
    const PROBE_SIZE: usize = 32;
    // Probe-and-decide fires when we lack AUTHORITATIVE per-item cost
    // information. JobPlan::new + set_profile auto-populate
    // `estimated_per_item_ns` with a classifier default (12 / 50 /
    // 600 ns); those values are routing hints, not measurements. When
    // the caller hasn't explicitly called with_estimated_per_item_ns,
    // we treat the auto-populated value as a guess and probe to learn
    // actual cost. Caller-explicit hints (the bench's
    // with_estimated_per_item_ns calls, or production code that
    // measured the workload) skip the probe -- they're authoritative.
    let cost_authoritative = plan.estimated_per_item_ns_explicit;
    // The probe path runs item 0 serially first to measure cost,
    // then bisects the tail. For heavy items this adds a full
    // item's runtime to the critical path: NMFD 5x100ms takes
    // 100ms-of-serial-probe + 50ms-of-parallel-tail = 150ms vs
    // rayon's 62ms (8-core ideal). That serial probe is the main
    // reason `flynnel_def` measured 2x slower than rayon on
    // cold_workloads NMFD-like cells.
    //
    // Skip the probe when the static classifier has already
    // signaled heavy per-item via use_smt=true (LatencyBound).
    // In that case the workload character is known; jumping
    // straight to the bisect with min_leaf=1 (from
    // adaptive_min_leaf's use_smt shortcut) gives full parallel
    // distribution from the very first call. When the classifier's
    // heavy guess is wrong, the observer migrates the global
    // class within a few iterations.
    if !cost_authoritative
        && !plan.use_smt
        && n < workers.saturating_mul(MIN_LEAF_ITEMS)
    {
        // Cold-path entry: production callers either supply
        // with_estimated_per_item_ns (cost_authoritative=true) or
        // hit the LatencyBound classifier shortcut (use_smt=true).
        // The probe path only fires for hint-less PortBound calls
        // at small N -- the rare measurement-detour case. Mark it
        // cold so LLVM (>= 21 since rust 1.96) keeps the
        // hint/SMT-classified fast paths icache-warm and reorders
        // this probe-and-amend-plan sequence into a cold section.
        core::hint::cold_path();
        // Probe must leave a non-empty tail so the bisect path has
        // items to parallelize after measurement. A `n.min(PROBE_SIZE)`
        // probe consumes the entire input when n <= PROBE_SIZE
        // (n=5 -> probe=5, tail=0, return before bisect),
        // serializing small-N heavy-per-item workloads (BigFloat
        // verify, NMFD with n<=8 instances) that rayon
        // parallelizes 3-4x.
        //
        // Formula: probe is at most 1/8 of the input, floored at 1,
        // capped at PROBE_SIZE. For n=5: probe=1, tail=4. For n=32:
        // probe=4, tail=28. For n=256: probe=32, tail=224. Tail
        // always has at least 7/8 of the items so the post-probe
        // bisect can drive parallel work-stealing.
        let probe_size = (n / 8).clamp(1, PROBE_SIZE);
        // Staged probe: time one item first. A single item at or
        // above the trust floor (20x the measured timer bracket,
        // bracket error <= ~5%) is a reliable heavy measurement,
        // and the remaining probe_size - 1 items skip the serial
        // detour: at 128 items x 500us the full n/8 probe costs
        // ~8 ms of serial work vs ~0.5 ms for one item. A light
        // first item is bracket-noise-dominated, so the rest of
        // the n/8 batch is probed for a reliable average (cheap,
        // the items are light).
        let (first, after_first) = items.split_at_mut(1);
        let start = std::time::Instant::now();
        record_leaf(plan.site, || op(first));
        let first_ns = start.elapsed().as_nanos() as u64;
        if after_first.is_empty() {
            // n=1: the single item was just processed inline.
            return;
        }
        let (probe_ns, probed, tail) = if first_ns >= probe_trust_floor_ns() || probe_size == 1 {
            (first_ns, 1usize, after_first)
        } else {
            let extra = (probe_size - 1).min(after_first.len().saturating_sub(1));
            if extra == 0 {
                (first_ns, 1usize, after_first)
            } else {
                let (more, rest) = after_first.split_at_mut(extra);
                let t2 = std::time::Instant::now();
                record_leaf(plan.site, || op(more));
                let more_ns = t2.elapsed().as_nanos() as u64;
                (first_ns + more_ns, 1 + extra, rest)
            }
        };
        let mut per_elem_ns = probe_ns.max(1) / probed as u64;
        let mut tail = tail;
        let dispatch_floor_ns = (workers as u64)
            .saturating_mul(target_per_leaf_overhead_ns)
            .saturating_mul(crate::cpu_info::small_host_dispatch_factor());
        // Confirmation probe: a probe preempted by the OS
        // mid-measurement overstates per-item cost by orders of
        // magnitude and would misroute a trivial workload to the
        // pool. When the probe says "dispatch" AND the light-path
        // batch ran (probed >= 8, so re-measurement is cheap
        // relative to it), time a tiny second probe and take the
        // MIN per-item estimate: preemption inflates a
        // measurement, never deflates one. The staged-heavy path
        // (probed == 1) skips confirmation: a second heavy item
        // on the critical path is the cost the staged probe
        // exists to avoid, and an inflated single-item reading
        // still routes to the pool, where the bisect's leaf
        // timing corrects the site.
        const CONFIRM_SIZE: usize = 4;
        // A single trusted item is confirmed when it could be a cold
        // first call rather than a heavy item: the first call at a
        // site pays one-time costs (code and site state faulting in)
        // that measured 30 us on a quiet 16-worker host and 79 us
        // under a saturated one for a one-add item, against an 80 us
        // dispatch floor there and a 40 us floor on 8 workers, so the
        // allowance is four floors. Up to three further items are
        // timed one at a time and the minimum taken, since preemption
        // only ever inflates a reading (a Windows quantum runs to
        // milliseconds); the loop stops at the first reading that
        // puts the tail under the floor, so a light item costs one
        // confirmation and a heavy one at most three, each at most
        // the allowance. It runs only when the tail is at least a
        // worker's width, where a misreading would dispatch a wide
        // tail of light items; a small tail costs little either way
        // and keeps the single-item critical path. An item beyond
        // the allowance is heavy on its own evidence.
        const CONFIRM_SINGLE_MAX: usize = 3;
        let cold_start_allowance_ns = dispatch_floor_ns.saturating_mul(4);
        if probed == 1 && first_ns < cold_start_allowance_ns && tail.len() >= workers {
            for _ in 0..CONFIRM_SINGLE_MAX {
                if per_elem_ns.saturating_mul(tail.len() as u64) < dispatch_floor_ns
                    || tail.len() <= 1
                {
                    break;
                }
                let (one, rest) = tail.split_at_mut(1);
                let start = std::time::Instant::now();
                record_leaf(plan.site, || op(one));
                per_elem_ns = per_elem_ns.min(start.elapsed().as_nanos().max(1) as u64);
                tail = rest;
            }
        }
        if per_elem_ns.saturating_mul(tail.len() as u64) >= dispatch_floor_ns
            && probed >= 8
            && tail.len() > CONFIRM_SIZE
        {
            let (confirm, rest) = tail.split_at_mut(CONFIRM_SIZE);
            let start = std::time::Instant::now();
            record_leaf(plan.site, || op(confirm));
            let confirm_ns = start.elapsed().as_nanos() as u64;
            per_elem_ns = per_elem_ns.min(confirm_ns.max(1) / CONFIRM_SIZE as u64);
            tail = rest;
        }
        let est_tail_ns = per_elem_ns.saturating_mul(tail.len() as u64);
        if est_tail_ns < dispatch_floor_ns {
            // Dispatch overhead would exceed remaining work; complete
            // the tail serially on the calling thread.
            record_leaf(plan.site, || op(tail));
            return;
        }
        // Tail is worth parallelizing. Match the observer-tuned
        // multiplier (default 2 leaves per worker) - the probe path
        // fires when no profile was provided, so we shouldn't
        // over-bisect to clamp(.., 8) like the cost-derived branch
        // would. Empirically (Genoa EPYC 9B14, run 2026-06-04) the
        // 8-clamp regressed Heavy/10k by 8% via probe-path
        // over-bisection vs the conservative observer-multiplier
        // choice.
        let leaves_per_worker = crate::sched::split_observer::split_multiplier() as usize;
        let max_budget = workers.saturating_mul(leaves_per_worker).max(1);
        // Probe path: use the per-element cost we just measured to
        // pick min_leaf adaptively (same formula as the main path
        // above). per_elem_ns came from the probe measurement so
        // it's authoritative for THIS workload's actual cost on
        // THIS host.
        let probe_min_leaf = target_per_leaf_overhead_ns
            .checked_div(per_elem_ns)
            .map(|raw| (raw.max(1) as usize).min(MIN_LEAF_ITEMS))
            .unwrap_or(MIN_LEAF_ITEMS);
        // Amend the plan with the measured per_item_ns so the bisect's
        // join_context -> pick_tier path observes a heavy-per-item
        // signal and routes to the worker pool. Without this, pick_tier
        // returns Inline for batch_size < 32 and the bisect runs
        // serially regardless of what min_leaf we computed. The probe
        // already learned the workload is heavy; record that knowledge
        // in the plan we hand down.
        let mut amended_plan = *plan;
        amended_plan.estimated_per_item_ns = Some(per_elem_ns.min(u32::MAX as u64) as u32);
        // Mark the probe-derived estimate as authoritative: it came
        // from a real measurement on this host, not a classifier
        // default. pick_tier's heavy_override consults this flag
        // when deciding small-N parallel dispatch.
        amended_plan.estimated_per_item_ns_explicit = true;
        amended_plan.batch_size = tail.len() as u32;
        bisect(&amended_plan, tail, &op, max_budget, max_budget, false, probe_min_leaf);
        return;
    }

    // Default path: continuation-steal-lazy bisect. First level
    // always splits to seed initial fanout (workers eager leaves);
    // subsequent levels run serially inline UNLESS the per-deque
    // steal counter on the dispatching worker has incremented since
    // the last check. Wins +12.7% Xeon Heavy/10k, +10.4% Xeon
    // Compute/100k, +5.1% Zen3 Compute/100k, never regresses across
    // the 12-cell (host x workload x size) bench matrix.
    if plan.bisect_variant.is_none() {
        let seed_depth = adaptive_seed_depth(plan, items.len(), workers);
        bisect_lazy_steal_driven(plan, items, &op, seed_depth, 0, effective_min_leaf);
        return;
    }

    // The two pinned BisectVariants drive an alternative
    // (leaves_per_worker, use_rayon_replenish) resolution into the
    // baseline bisect / bisect_rayon_style helpers. Routed
    // automatically by `adaptive_variant_routing` on AMD Compute;
    // callers can also pin per-plan via `with_bisect_variant`.
    let (leaves_per_worker, use_rayon_replenish) = match plan.bisect_variant {
        // ProducerMaxLenWorkers: upfront leaves exactly match rayon's
        // LengthSplitter default initial count = workers * 1. Wins
        // +19.6% on Zen3 Compute/100k.
        Some(crate::sched::plan::BisectVariant::ProducerMaxLenWorkers) => {
            (1usize, false)
        }
        // RayonStyleReplenish: start with one leaf per worker; on
        // each observed steal, replenish to max(workers, splits/2)
        // instead of the upfront budget. Wins +37.6% on Zen3
        // Compute/10k.
        Some(crate::sched::plan::BisectVariant::RayonStyleReplenish) => {
            (1usize, true)
        }
        // None is handled by the lazy-steal early-return above.
        None => unreachable!("None branch handled above"),
    };
    let max_budget = workers.saturating_mul(leaves_per_worker).max(1);
    if use_rayon_replenish {
        bisect_rayon_style(plan, items, &op, max_budget, workers, false, effective_min_leaf);
    } else {
        bisect(plan, items, &op, max_budget, max_budget, false, effective_min_leaf);
    }
}

/// Apply `op` to each chunk of size at most `chunk_size`, in
/// parallel. Chunks at the slice's tail may be smaller than
/// `chunk_size`. Useful when the caller knows a SIMD-optimal
/// chunk shape (e.g., AVX-512 16-lane FpN<16> = 64 items per
/// chunk to fill 16 ZMM registers x 4 lanes each).
#[track_caller]
pub fn for_each_fixed_chunk<T, F>(
    plan: &JobPlan,
    items: &mut [T],
    chunk_size: usize,
    op: F,
)
where
    T: Send,
    F: Fn(&mut [T]) + Sync,
{
    let n = items.len();
    if n == 0 {
        return;
    }
    // Per-call-site identity from the caller's source location
    // (track_caller chain); an outer attachment wins.
    let plan_owned = plan
        .with_site_if_none(crate::sched::call_site::caller_site())
        .apply_site_class();
    let plan = &plan_owned;
    let chunk = chunk_size.max(1);
    if n <= chunk {
        // Whole-input serial pass: one span for the SITE's
        // statistics only. It is not a chunk leaf, so it stays out
        // of the process-global per-leaf counters.
        let t0 = std::time::Instant::now();
        op(items);
        record_leaf_span_ns(plan.site, t0.elapsed().as_nanos() as u64);
        return;
    }
    let n_chunks = n.div_ceil(chunk);
    let max_budget = n_chunks.max(1);
    bisect_fixed(plan, items, &op, max_budget, max_budget, chunk, false);
}

/// Recursive bisection driver for `for_each_chunk`. Cuts the
/// slice in half on each level, dispatching the halves via
/// `sched::join_context`.
///
/// # Adaptive splitter (SLAW-style)
///
/// `splits` is the remaining split budget; `max_budget` is the
/// full budget to replenish to when migrated. On each call:
///
/// - If `migrated == true`, we observed steal pressure (some
///   peer dequeued this side from another worker's deque). The
///   pool is hungrier than the original split estimate; reset
///   `splits` to `max_budget` so this subtree can split log2(N)
///   more levels and produce many more leaves for distribution.
/// - Otherwise, the budget halves per level until it hits zero.
///
/// Bottoms out when either `items.len() <= MIN_LEAF_ITEMS` (too
/// small to split further) or `splits == 0` and `!migrated` (no
/// pressure, no budget).
///
/// This is the SLAW ("Scalable Locality-aware Adaptive Work-
/// stealing", Guo-Zhao-Cavé-Sarkar, IPDPS 2010) adaptive work-
/// stealing pattern: more splits where there is observed
/// contention, fewer where workers are saturated.
fn bisect<T, F>(
    plan: &JobPlan,
    items: &mut [T],
    op: &F,
    splits: usize,
    max_budget: usize,
    migrated: bool,
    min_leaf: usize,
)
where
    T: Send,
    F: Fn(&mut [T]) + Sync,
{
    if items.len() <= min_leaf {
        record_leaf(plan.site, || op(items));
        return;
    }
    // Replenish on stealing pressure.
    let cur_splits = if migrated { max_budget } else { splits };
    // Leaf when we're out of budget. The check uses `<= 1` rather
    // than `== 0`: a split with `cur_splits == 1` would emit two
    // children each holding `splits = 0`, and each would
    // immediately leaf. That produces 2*max_budget leaves total
    // instead of the intended max_budget, doubling per-leaf
    // dispatch overhead at fine-grain rapid-fire workloads. The
    // `<= 1` form stops splitting at the level where the budget
    // is consumed, capping the leaf count at the intended
    // max_budget.
    if cur_splits <= 1 {
        record_leaf(plan.site, || op(items));
        return;
    }
    let mid = items.len() / 2;
    let (left, right) = items.split_at_mut(mid);
    let half = cur_splits >> 1;
    join_context(
        plan,
        // `left_injected` propagates the migrated/injection signal
        // from the parent (left always runs inline, so it doesn't
        // observe its own steal flag - but if the parent itself
        // was stolen / injected, left should know).
        |left_injected| bisect(plan, left, op, half, max_budget, left_injected, min_leaf),
        // `right_stolen` is the freshly-observed stolen flag for
        // the right half. If a peer dequeued this job, we know
        // the pool is hungry, so this subtree splits more.
        |right_stolen| bisect(plan, right, op, half, max_budget, right_stolen, min_leaf),
    );
}

/// Rayon-style replenish variant of [`bisect`]. Differs from `bisect`
/// in the steal-replenish formula: on observed steal, the new
/// budget is `max(workers_floor, splits / 2)` instead of resetting
/// to the upfront `max_budget`. This mirrors rayon-1.12.0's
/// `Splitter::try_split` formula at
/// `src/iter/plumbing/mod.rs::258-283`:
///
/// ```text
/// if stolen { splits = max(thread_count, splits / 2) }
/// else if splits > 0 { splits /= 2 } else { return false }
/// ```
///
/// `workers_floor` plays the role of `current_num_threads()` in the
/// rayon formula. Used only when `for_each_chunk` is called with
/// [`crate::sched::plan::BisectVariant::RayonStyleReplenish`].
fn bisect_rayon_style<T, F>(
    plan: &JobPlan,
    items: &mut [T],
    op: &F,
    splits: usize,
    workers_floor: usize,
    migrated: bool,
    min_leaf: usize,
)
where
    T: Send,
    F: Fn(&mut [T]) + Sync,
{
    if items.len() <= min_leaf {
        record_leaf(plan.site, || op(items));
        return;
    }
    // Rayon's replenish formula: max(workers, splits/2).
    let cur_splits = if migrated {
        workers_floor.max(splits / 2)
    } else {
        splits
    };
    if cur_splits <= 1 {
        record_leaf(plan.site, || op(items));
        return;
    }
    let mid = items.len() / 2;
    let (left, right) = items.split_at_mut(mid);
    let half = cur_splits >> 1;
    join_context(
        plan,
        |left_injected| {
            bisect_rayon_style(plan, left, op, half, workers_floor, left_injected, min_leaf)
        },
        |right_stolen| {
            bisect_rayon_style(plan, right, op, half, workers_floor, right_stolen, min_leaf)
        },
    );
}

/// Sampled leaf recorder used by the default lazy-steal bisect
/// path. Only every Nth call goes through the full TSC bracket;
/// the other N-1 calls just run the body.
#[inline]
fn record_leaf_sampled<F: FnOnce() -> R, R>(
    site: Option<crate::sched::call_site::SiteRef>,
    body: F,
) -> R {
    thread_local! {
        static STRIDE_TICK: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    let should_sample = STRIDE_TICK.with(|c| {
        let v = c.get().wrapping_add(1);
        c.set(if v >= LEAF_SAMPLE_STRIDE { 0 } else { v });
        v >= LEAF_SAMPLE_STRIDE
    });
    if should_sample { record_leaf(site, body) } else { body() }
}

/// Direction B: continuation-steal lazy bisect.
///
/// Splits only when this worker observes that someone has stolen from
/// its OWN deque since the last check. Reads
/// `WorkerCtx::stats::times_stolen_from` as a Relaxed atomic - a
/// counter incremented by thieves at the steal site (see
/// `arena_local.rs::WorkerCtx::find_work` and the worker_loop steal
/// path).
///
/// The FIRST `seed_splits_remaining` calls always split to seed
/// initial fanout to ~`workers` leaves (so workers have something to
/// steal from before the lazy logic kicks in). Subsequent calls
/// consult the steal counter and only split if it advanced.
fn bisect_lazy_steal_driven<T, F>(
    plan: &JobPlan,
    items: &mut [T],
    op: &F,
    seed_splits_remaining: usize,
    last_seen_steals: u64,
    min_leaf: usize,
)
where
    T: Send,
    F: Fn(&mut [T]) + Sync,
{
    if items.len() <= min_leaf {
        record_leaf_sampled(plan.site, || op(items));
        return;
    }
    // Decide whether to split: always split during the seed phase
    // (so the deque fills with stealable work), then run lazy.
    let ctx_ptr = crate::sched::arena_local::current_worker_ctx();
    let (should_split, new_last_seen) = if seed_splits_remaining > 0 {
        (true, last_seen_steals)
    } else if ctx_ptr.is_null() {
        // External caller path: outside any worker, no ctx to read.
        // Split eagerly until execution lands inside a worker via
        // external_dispatch's wrapper job.
        (true, last_seen_steals)
    } else {
        // SAFETY: ctx_ptr set by worker_loop on this thread; valid
        // until worker_loop returns.
        let ctx = unsafe { &*ctx_ptr };
        let cur = ctx
            .stats
            .times_stolen_from
            .load(core::sync::atomic::Ordering::Relaxed);
        if cur != last_seen_steals {
            (true, cur)
        } else {
            (false, last_seen_steals)
        }
    };

    if !should_split {
        // No steal pressure observed: run the WHOLE remaining slice
        // inline as one leaf.
        record_leaf_sampled(plan.site, || op(items));
        return;
    }
    let mid = items.len() / 2;
    let (left, right) = items.split_at_mut(mid);
    let next_seed = seed_splits_remaining.saturating_sub(1);
    join_context(
        plan,
        |_| bisect_lazy_steal_driven(plan, left, op, next_seed, new_last_seen, min_leaf),
        |_| bisect_lazy_steal_driven(plan, right, op, next_seed, new_last_seen, min_leaf),
    );
}

/// Apply `op` to every chunk-triple of `(out, a, b)` in parallel by
/// recursively bisecting the three slices in lockstep. Each leaf
/// chunk is processed serially by `op(out_c, a_c, b_c)`, which is
/// the right granule for SIMD slice kernels (mul_slice, add_slice,
/// sub_slice in the consumer crate's SIMD primitives).
///
/// All three slices MUST have the same length (asserted). The
/// bisect always cuts at the midpoint so the three sub-slices
/// stay aligned by index.
///
/// Same SLAW splitter as [`for_each_chunk`]: budget halves per
/// level; replenishes to `max_budget` on observed steal.
#[track_caller]
pub fn for_each_chunk_triple<T1, T2, T3, F>(
    plan: &JobPlan,
    out: &mut [T1],
    a: &[T2],
    b: &[T3],
    op: F,
)
where
    T1: Send,
    T2: Send + Sync,
    T3: Send + Sync,
    F: Fn(&mut [T1], &[T2], &[T3]) + Sync,
{
    let leaf = adaptive_min_leaf(plan, MIN_LEAF_ITEMS);
    for_each_chunk_triple_min_leaf(plan, out, a, b, leaf, op)
}

/// Same as [`for_each_chunk_triple`] but the recursion floor is
/// caller-supplied. At small N (say n=1k with the default
/// `MIN_LEAF_ITEMS=256`), the bisect floor caps the chunk count at
/// `n / MIN_LEAF_ITEMS = 4` leaves - only 4-way parallelism on a
/// 16-thread host. Slice ops can pass a smaller floor (e.g.,
/// `PAR_MIN_ELEMS = 64`) so the SLAW budget (`workers * multiplier`)
/// caps chunk count instead, matching rayon's per-chunk fanout at
/// small N.
#[track_caller]
pub fn for_each_chunk_triple_min_leaf<T1, T2, T3, F>(
    plan: &JobPlan,
    out: &mut [T1],
    a: &[T2],
    b: &[T3],
    min_leaf: usize,
    op: F,
)
where
    T1: Send,
    T2: Send + Sync,
    T3: Send + Sync,
    F: Fn(&mut [T1], &[T2], &[T3]) + Sync,
{
    assert_eq!(out.len(), a.len(), "for_each_chunk_triple: out.len() != a.len()");
    assert_eq!(out.len(), b.len(), "for_each_chunk_triple: out.len() != b.len()");
    let n = out.len();
    if n == 0 {
        return;
    }
    // Per-call-site identity from the caller's source location
    // (track_caller chain); an outer attachment wins.
    let plan_owned = plan
        .with_site_if_none(crate::sched::call_site::caller_site())
        .apply_site_class();
    let plan = &plan_owned;
    // Under the dispatch floor by the caller's own estimate: the
    // body runs here, as in for_each_chunk.
    if collapses_inline(plan, n) {
        record_leaf(plan.site, || op(out, a, b));
        return;
    }
    let leaf = min_leaf.max(1);
    let workers = global_local_arena().total_workers();
    let multiplier = crate::sched::split_observer::split_multiplier() as usize;
    let max_budget = workers.saturating_mul(multiplier).max(1);
    bisect_triple(plan, out, a, b, &op, leaf, max_budget, max_budget, false);
}

#[allow(clippy::too_many_arguments)]
fn bisect_triple<T1, T2, T3, F>(
    plan: &JobPlan,
    out: &mut [T1],
    a: &[T2],
    b: &[T3],
    op: &F,
    min_leaf: usize,
    splits: usize,
    max_budget: usize,
    migrated: bool,
)
where
    T1: Send,
    T2: Send + Sync,
    T3: Send + Sync,
    F: Fn(&mut [T1], &[T2], &[T3]) + Sync,
{
    if out.len() <= min_leaf {
        record_leaf(plan.site, || op(out, a, b));
        return;
    }
    let cur_splits = if migrated { max_budget } else { splits };
    // Leaf at `<= 1`, not `== 0`. See `bisect` for the rationale:
    // splitting at cur_splits=1 emits two children each holding
    // 0 budget, both immediately leaf, doubling leaf count.
    if cur_splits <= 1 {
        record_leaf(plan.site, || op(out, a, b));
        return;
    }
    let mid = out.len() >> 1;
    let (out_lo, out_hi) = out.split_at_mut(mid);
    let (a_lo, a_hi) = a.split_at(mid);
    let (b_lo, b_hi) = b.split_at(mid);
    let half = cur_splits >> 1;
    join_context(
        plan,
        |left_inj| bisect_triple(plan, out_lo, a_lo, b_lo, op, min_leaf, half, max_budget, left_inj),
        |right_stolen| bisect_triple(plan, out_hi, a_hi, b_hi, op, min_leaf, half, max_budget, right_stolen),
    );
}

/// Apply `op(start_index, chunk)` to every chunk of `items` in
/// parallel by recursively bisecting the slice. The closure receives
/// the global starting index of the chunk so per-element work that
/// needs the absolute index (matmul row/col decode, SpMV row id,
/// LU trailing-update row id) can compute it from
/// `start + i_in_chunk`.
///
/// Same SLAW splitter as [`for_each_chunk`]: budget halves per
/// level; replenishes to `max_budget` on observed steal.
#[track_caller]
pub fn for_each_chunk_indexed<T, F>(plan: &JobPlan, items: &mut [T], op: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let leaf = adaptive_min_leaf(plan, MIN_LEAF_ITEMS);
    for_each_chunk_indexed_min_leaf(plan, items, leaf, op)
}

/// One-time measurement of the `Instant` bracket cost on this host
/// (Windows QPC measured ~91 ns on Zen+ R7 2700; rdtsc-backed
/// clocks run ~20-40 ns). Measured at first use: 4096 empty
/// brackets, ~0.4 ms once per process.
fn timer_bracket_ns() -> u64 {
    static BRACKET_NS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BRACKET_NS.get_or_init(|| {
        let iters = 4_096u32;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(std::time::Instant::now().elapsed());
        }
        ((t0.elapsed().as_nanos() as u64) / iters as u64).max(1)
    })
}

/// Trust floor for a single-item probe measurement: 20x the
/// measured timer bracket, bounding the bracket's contribution to
/// the reading at ~5%.
fn probe_trust_floor_ns() -> u64 {
    timer_bracket_ns().saturating_mul(20)
}

/// Probe size for the min_leaf-style entry probes, floored at the
/// caller's min_leaf.
const PROBE_SIZE_FLOOR: usize = 16;
/// Item count from which the min_leaf-style entries probe
/// `max(PROBE_SIZE_FLOOR, min_leaf)` items.
const PROBE_MIN_N: usize = 64;
/// Item count from which a one-min_leaf-unit probe fires (subject
/// to `n >= 2 * min_leaf` so the tail stays non-empty). Swept at
/// 2/4/8/16 on both bench hosts (PortBound heavy ~2-3ms items,
/// min_leaf 1, cold fresh-process sites, median of 5): 2 wins or
/// ties every cell. Zen+ R7 2700: n=3 heavy 5.70ms vs 8.19ms at
/// gate 4, n=4 heavy 5.70ms vs 11.60ms at gate 8. AVX-512 host:
/// n=3 heavy 3.80ms vs 5.06ms, n=8 heavy 4.0ms vs 13.4ms at gate
/// 16. Light items cost 4-16us at every gate on both hosts.
const PROBE_SMALL_MIN_N: usize = 2;

/// Same as [`for_each_chunk_indexed`] but the recursion floor is
/// supplied by the caller instead of [`MIN_LEAF_ITEMS`]. Use this
/// for coarse-grained workloads where each item carries enough work
/// that bisecting all the way to single-element leaves is justified
/// (e.g., per-row dense-GEMM strips where one strip is a full
/// `gemm_via_schemes` call, and the total strip count is small
/// relative to MIN_LEAF_ITEMS).
///
/// `min_leaf >= 1`. Setting `min_leaf == 1` produces one job per
/// item - the right choice when the body is heavy and the total
/// item count equals the worker count.
///
/// Probe sizing shared by the min_leaf-style entries (this and
/// [`collect_indexed`]): at `n >= PROBE_MIN_N` the probe is
/// `max(PROBE_SIZE_FLOOR, min_leaf)` items; at
/// `PROBE_SMALL_MIN_N.max(2 * min_leaf) <= n < PROBE_MIN_N` it is
/// one min_leaf unit, which is a complete work unit by the
/// min_leaf contract and leaves a non-empty tail. Heavy measured
/// cost promotes the tail to the pool; light measured cost
/// reclassifies to FineGrain and the tail runs inline.
#[track_caller]
pub fn for_each_chunk_indexed_min_leaf<T, F>(
    plan: &JobPlan,
    items: &mut [T],
    min_leaf: usize,
    op: F,
)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let n = items.len();
    if n == 0 {
        return;
    }
    // Per-call-site identity from the caller's source location
    // (track_caller chain); an outer attachment wins.
    let plan_owned = plan
        .with_site_if_none(crate::sched::call_site::caller_site())
        .apply_site_class();
    let plan = &plan_owned;
    let _flush_on_exit = FlushLeafStatsOnExit;
    // Under the dispatch floor by the caller's own estimate: the
    // body runs here, as in for_each_chunk.
    if collapses_inline(plan, n) {
        record_leaf(plan.site, || op(0, items));
        return;
    }
    let leaf = min_leaf.max(1);
    let workers = global_local_arena().total_workers();

    // Probe-and-decide: when the caller hasn't supplied a per-item
    // cost estimate AND the workload is big enough to amortize a
    // probe, run a tiny prefix through the closure to MEASURE the
    // actual per-item cost. The measurement feeds the static
    // classifier (via JobPlan::with_estimated_per_item_ns) so the
    // bulk dispatch already routes to the right WorkloadClass on
    // the first call -- no caller hint, no observer warm-up
    // iterations required.
    //
    // The probe runs on items[0..probe_n] with start_idx=0,
    // executes the work for those items (it isn't wasted), and the
    // bulk dispatch then operates on items[probe_n..] with
    // start_idx=probe_n to preserve absolute-index semantics for
    // the caller's closure.
    //
    // probe_n MUST be at least min_leaf. The caller's closure body
    // is sized around the leaf granularity: a stride-w row kernel,
    // a chunked accumulator over min_leaf-sized blocks, etc. A
    // probe shorter than min_leaf would feed the closure a
    // half-complete unit and produce a near-zero per-item estimate,
    // which would mis-classify the workload into FineGrain /
    // Streaming when the actual work is much heavier.
    let plan_holder: JobPlan;
    // Probe path consults estimated_per_item_ns_explicit (not just
    // is_none) so classifier-default 12/50/600 ns doesn't suppress
    // measurement for JobPlan::new callers. Same fix as for_each_chunk.
    // use_smt gate matches for_each_chunk's probe: when the
    // classifier already signals heavy per-item work, skip the
    // serial probe so every item starts concurrently (racing and
    // fan-out callers rely on this).
    let cost_authoritative = plan.estimated_per_item_ns_explicit;
    let probe_quota: usize = if cost_authoritative || plan.use_smt {
        0
    } else if n >= PROBE_MIN_N.max(leaf) {
        PROBE_SIZE_FLOOR.max(leaf).min(n)
    } else if n >= PROBE_SMALL_MIN_N.max(2 * leaf) {
        leaf
    } else {
        0
    };
    let (effective_plan, items_to_dispatch, start_offset): (&JobPlan, &mut [T], usize) =
        if probe_quota > 0 {
            // Staged: one leaf unit first. A unit reading at or
            // above the trust floor is reliable on its own and the
            // rest of the quota skips the serial detour; a light
            // unit is bracket-noise-dominated, so the remaining
            // quota is probed for a reliable average.
            let (first, rest) = items.split_at_mut(leaf.min(probe_quota));
            let first_n = first.len();
            let t0 = std::time::Instant::now();
            record_leaf(plan.site, || op(0, first));
            let first_ns = t0.elapsed().as_nanos() as u64;
            let (probe_ns, probed, tail): (u64, usize, &mut [T]) =
                if first_ns >= probe_trust_floor_ns() || probe_quota <= first_n {
                    (first_ns, first_n, rest)
                } else {
                    let extra = (probe_quota - first_n).min(rest.len().saturating_sub(1));
                    if extra == 0 {
                        (first_ns, first_n, rest)
                    } else {
                        let (more, rest2) = rest.split_at_mut(extra);
                        let t2 = std::time::Instant::now();
                        record_leaf(plan.site, || op(first_n, more));
                        let more_ns = t2.elapsed().as_nanos() as u64;
                        (first_ns + more_ns, first_n + extra, rest2)
                    }
                };
            let per_item_ns = (probe_ns.max(1) / probed as u64).min(u32::MAX as u64) as u32;
            plan_holder = (*plan).with_estimated_per_item_ns(per_item_ns);
            (&plan_holder, tail, probed)
        } else {
            (plan, items, 0)
        };

    if items_to_dispatch.is_empty() {
        return;
    }

    // Default: continuation-steal-lazy bisect with absolute-index
    // passthrough. Uses adaptive_seed_depth so heavy items get more
    // eager splits for load-balance headroom; light items get
    // log2(workers) to minimize dispatch overhead.
    if effective_plan.bisect_variant.is_none() {
        let seed_depth = adaptive_seed_depth(effective_plan, items_to_dispatch.len(), workers);
        bisect_lazy_steal_driven_indexed(
            effective_plan, items_to_dispatch, start_offset, &op, leaf, seed_depth, 0,
        );
        return;
    }
    // Pinned variant: fall back to the eager all-the-way-to-min_leaf
    // bisect_indexed for callers who explicitly opted into a fixed
    // split shape.
    let multiplier = crate::sched::split_observer::split_multiplier() as usize;
    let max_budget = workers.saturating_mul(multiplier).max(1);
    bisect_indexed(
        effective_plan, items_to_dispatch, start_offset, &op, leaf, max_budget, max_budget, false,
    );
}

/// Call `f(i)` for every `i` in `0..n` in parallel, each index
/// exactly once, with no slice to mutate: the read-only and
/// side-effect shapes (declare a cell per id through a shared
/// resolver, fault a page per index, fill a row of a buffer the
/// closure addresses itself). `min_leaf` is the bisect floor in
/// indices, as in [`for_each_chunk_indexed_min_leaf`], which this
/// runs over a zero-sized slice of length `n` so the probe, the
/// per-call-site statistics and the lazy-steal bisect all apply
/// unchanged. Nothing is allocated for the slice.
///
/// For a per-chunk body over a read-only slice use
/// [`for_each_chunk_ref`].
#[track_caller]
pub fn for_each_indexed<F>(plan: &JobPlan, n: usize, min_leaf: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if n == 0 {
        return;
    }
    // A Vec of unit is length without storage: the bisect halves
    // lengths and hands out absolute starts, which is all the
    // indexed walk needs.
    let mut units: Vec<()> = vec![(); n];
    for_each_chunk_indexed_min_leaf(plan, &mut units, min_leaf, |start, chunk| {
        for i in start..start + chunk.len() {
            f(i);
        }
    });
}

/// Call `f(start, chunk)` over consecutive read-only chunks of
/// `items` in parallel: `chunk` is `items[start..start + len]` with
/// `len == min_leaf` except on the trailing chunk, and the chunks
/// tile the slice exactly once. The read-only counterpart of
/// [`for_each_chunk_indexed_min_leaf`] for a body that needs a fixed
/// batch width (a tile kernel, a resolver that takes a run of ids)
/// and never writes the input. Built on [`for_each_indexed`] over
/// the chunk count, one chunk per index.
#[track_caller]
pub fn for_each_chunk_ref<T, F>(plan: &JobPlan, items: &[T], min_leaf: usize, f: F)
where
    T: Sync,
    F: Fn(usize, &[T]) + Sync,
{
    let n = items.len();
    if n == 0 {
        return;
    }
    let width = min_leaf.max(1);
    let n_chunks = n.div_ceil(width);
    for_each_indexed(plan, n_chunks, 1, |i| {
        let lo = i * width;
        let hi = (lo + width).min(n);
        f(lo, &items[lo..hi]);
    });
}

/// Continuation-steal-lazy bisect with absolute-index passthrough.
/// Mirrors [`bisect_lazy_steal_driven`] but threads the start
/// offset through so leaves know their absolute slot in the
/// original input. Used by [`for_each_chunk_indexed_min_leaf`] as
/// the default path; closes the per-dispatch overhead gap between
/// the indexed and non-indexed paths at fine-grain leaves.
#[allow(clippy::too_many_arguments)]
fn bisect_lazy_steal_driven_indexed<T, F>(
    plan: &JobPlan,
    items: &mut [T],
    start: usize,
    op: &F,
    min_leaf: usize,
    seed_splits_remaining: usize,
    last_seen_steals: u64,
)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    if items.len() <= min_leaf {
        // record_leaf (NOT _sampled): the closing-loop auto-classifier
        // observer needs every leaf timed so it can converge on the
        // workload's shape (mean_ns + cv^2) within the first few
        // iterations of a real workload, not after thousands. The
        // sampled variant rate-limited to 1-in-8, which never
        // accumulated enough flushes for the auto-migration to fire
        // on realistic small-N workloads (e.g. 16-chunk grep).
        record_leaf(plan.site, || op(start, items));
        return;
    }
    let ctx_ptr = crate::sched::arena_local::current_worker_ctx();
    let (should_split, new_last_seen) = if seed_splits_remaining > 0 {
        (true, last_seen_steals)
    } else if ctx_ptr.is_null() {
        // External caller path: outside any worker, no ctx to read.
        // Split eagerly until execution lands inside a worker via
        // external_dispatch's wrapper job.
        (true, last_seen_steals)
    } else {
        // SAFETY: ctx_ptr set by worker_loop on this thread; valid
        // until worker_loop returns.
        let ctx = unsafe { &*ctx_ptr };
        let cur = ctx
            .stats
            .times_stolen_from
            .load(core::sync::atomic::Ordering::Relaxed);
        if cur != last_seen_steals {
            (true, cur)
        } else {
            (false, last_seen_steals)
        }
    };

    if !should_split {
        // No steal pressure observed: run the WHOLE remaining slice
        // inline as one leaf. This is the rayon continuation-stealing
        // pattern: only fork further when somebody is starving.
        record_leaf(plan.site, || op(start, items));
        return;
    }
    let mid = items.len() / 2;
    let (left, right) = items.split_at_mut(mid);
    let right_start = start + mid;
    let next_seed = seed_splits_remaining.saturating_sub(1);
    join_context(
        plan,
        |_| bisect_lazy_steal_driven_indexed(plan, left, start, op, min_leaf, next_seed, new_last_seen),
        |_| bisect_lazy_steal_driven_indexed(plan, right, right_start, op, min_leaf, next_seed, new_last_seen),
    );
}

#[allow(clippy::too_many_arguments)]
fn bisect_indexed<T, F>(
    plan: &JobPlan,
    items: &mut [T],
    start: usize,
    op: &F,
    min_leaf: usize,
    splits: usize,
    max_budget: usize,
    migrated: bool,
)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    if items.len() <= min_leaf {
        record_leaf(plan.site, || op(start, items));
        return;
    }
    let cur_splits = if migrated { max_budget } else { splits };
    // Leaf at `<= 1` (see `bisect` for the doubling-bug rationale).
    if cur_splits <= 1 {
        record_leaf(plan.site, || op(start, items));
        return;
    }
    let mid = items.len() >> 1;
    let (left, right) = items.split_at_mut(mid);
    let right_start = start + mid;
    let half = cur_splits >> 1;
    join_context(
        plan,
        |left_inj| {
            bisect_indexed(plan, left, start, op, min_leaf, half, max_budget, left_inj)
        },
        |right_stolen| {
            bisect_indexed(
                plan, right, right_start, op, min_leaf, half, max_budget, right_stolen,
            )
        },
    );
}

/// Parallel indexed collect: compute `f(idx)` for `idx in 0..n` in
/// parallel and return the result `Vec<R>`. Matches the cost profile
/// of `rayon`'s `(0..n).into_par_iter().map(f).collect()` by
/// allocating the result buffer WITHOUT zero-init - uses
/// `MaybeUninit<R>` + ptr writes inside the SLAW bisect. This is the
/// right helper for indexed-collect ops (matmul, spmv, LU row update,
/// Jacobi rotation, eigenvalue rotation, FFT butterfly, block-sparse
/// GEMM tiles) where the pre-zero-init cost on the indexed slice
/// path dominated for small-to-mid N.
///
/// `min_leaf` controls the bisect floor:
/// - `1` for heavy-per-item work (matmul O(k), spmv O(nnz_per_row),
///   row updates) - the SLAW budget (`workers * multiplier`) is the
///   right cap, not the leaf floor.
/// - `MIN_LEAF_ITEMS` (~256) for fine-grain work where the floor
///   amortizes join overhead.
///
/// # Safety contract
///
/// `R` is dropped only on the success path. If `f` panics mid-bisect,
/// the partial result Vec leaks (no double-drop). This is sound for
/// `Copy` / POD result types (FpN, tuples of FpN, etc.); for
/// resource-holding `R` types, callers should ensure panics don't
/// occur or use a panic-safe wrapper.
#[track_caller]
pub fn collect_indexed<R, F>(plan: &JobPlan, n: usize, min_leaf: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync,
{
    if n == 0 {
        return Vec::new();
    }
    // Per-call-site identity from the caller's source location
    // (track_caller chain). An outer entry (heartbeat /
    // token-bucket / tiny-tasks) that already attached its own
    // site wins via with_site_if_none.
    let plan_owned = plan
        .with_site_if_none(crate::sched::call_site::caller_site())
        .apply_site_class();
    let plan = &plan_owned;
    let _flush_on_exit = FlushLeafStatsOnExit;

    let leaf = min_leaf.max(1);
    let workers = global_local_arena().total_workers();
    let multiplier = crate::sched::split_observer::split_multiplier() as usize;
    let max_budget = workers.saturating_mul(multiplier).max(1);

    // Allocate buffer of `MaybeUninit<R>` with exact capacity, NO
    // zero-init. We promise to write every slot via the probe +
    // collect_inner before transmuting back to `Vec<R>`.
    let mut buf: Vec<std::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: `MaybeUninit<R>` does not require initialization.
    // `set_len` is valid for any value <= capacity. After the probe
    // and `collect_inner` fill every slot, we transmute to `Vec<R>`.
    unsafe { buf.set_len(n); }

    // Entry probe, same sizing contract as
    // [`for_each_chunk_indexed_min_leaf`]: measures per-item cost
    // when the caller supplied none so the bulk dispatch routes on
    // a measurement instead of a classifier guess.
    // Same use_smt gate as the for_each_chunk probes: a
    // classifier-signaled heavy plan skips the serial probe so
    // every item starts concurrently (race_any / race_quorum
    // dispatch through here).
    let cost_authoritative = plan.estimated_per_item_ns_explicit;
    let probe_quota: usize = if cost_authoritative || plan.use_smt {
        0
    } else if n >= PROBE_MIN_N.max(leaf) {
        PROBE_SIZE_FLOOR.max(leaf).min(n)
    } else if n >= PROBE_SMALL_MIN_N.max(2 * leaf) {
        leaf
    } else {
        0
    };
    let plan_holder: JobPlan;
    let (effective_plan, start_offset): (&JobPlan, usize) = if probe_quota > 0 {
        let base = buf.as_mut_ptr() as *mut R;
        // Staged, matching for_each_chunk_indexed_min_leaf: one
        // leaf unit first; a trust-floor reading skips the rest of
        // the quota, a light reading probes the remainder for a
        // reliable average. Every probed slot is written exactly
        // once; the bulk dispatch below starts past them.
        let first_n = leaf.min(probe_quota);
        let t0 = std::time::Instant::now();
        record_leaf(plan.site, || {
            for i in 0..first_n {
                // SAFETY: `i < first_n <= n = buf.len()`; slots
                // below `probed` are written only here and in the
                // extra-probe loop over a disjoint range.
                unsafe {
                    base.add(i).write(f(i));
                }
            }
        });
        let first_ns = t0.elapsed().as_nanos() as u64;
        let (probe_ns, probed): (u64, usize) =
            if first_ns >= probe_trust_floor_ns() || probe_quota <= first_n {
                (first_ns, first_n)
            } else {
                let extra = (probe_quota - first_n).min(n - first_n - 1);
                if extra == 0 {
                    (first_ns, first_n)
                } else {
                    let t2 = std::time::Instant::now();
                    record_leaf(plan.site, || {
                        for i in first_n..first_n + extra {
                            // SAFETY: `i < first_n + extra < n`;
                            // disjoint from the first-unit range.
                            unsafe {
                                base.add(i).write(f(i));
                            }
                        }
                    });
                    let more_ns = t2.elapsed().as_nanos() as u64;
                    (first_ns + more_ns, first_n + extra)
                }
            };
        let per_item_ns = (probe_ns.max(1) / probed as u64).min(u32::MAX as u64) as u32;
        plan_holder = (*plan).with_estimated_per_item_ns(per_item_ns);
        (&plan_holder, probed)
    } else {
        (plan, 0)
    };

    if start_offset < n {
        collect_inner(
            effective_plan,
            &mut buf[start_offset..],
            start_offset,
            &f,
            leaf,
            max_budget,
            max_budget,
            false,
        );
    }

    let mut buf = std::mem::ManuallyDrop::new(buf);
    // SAFETY: every slot was written exactly once by
    // `collect_inner`. `MaybeUninit<R>` has the same layout as `R`,
    // so reconstructing the Vec with the same allocation is sound.
    unsafe { Vec::from_raw_parts(buf.as_mut_ptr() as *mut R, n, buf.capacity()) }
}

#[allow(clippy::too_many_arguments)]
fn collect_inner<R, F>(
    plan: &JobPlan,
    items: &mut [std::mem::MaybeUninit<R>],
    start: usize,
    f: &F,
    min_leaf: usize,
    splits: usize,
    max_budget: usize,
    migrated: bool,
)
where
    R: Send,
    F: Fn(usize) -> R + Sync,
{
    // Leaf fill via raw pointer arithmetic, matching rayon's
    // `CollectConsumer`-`consume_iter` byte-for-byte:
    // `self.start.0.add(i).write(item)`. This avoids the
    // `MaybeUninit::write` API which (per MICRO 2024 / rayon-demo
    // measurements) can occasionally inhibit LLVM auto-vec via
    // the `&mut MaybeUninit<R>` reference dance.
    #[inline(always)]
    fn fill_leaf<R, F>(items: &mut [std::mem::MaybeUninit<R>], start: usize, f: &F)
    where
        F: Fn(usize) -> R,
    {
        let len = items.len();
        let base = items.as_mut_ptr() as *mut R;
        for i in 0..len {
            // SAFETY: `base.add(i)` is in-bounds (`i < len`) and
            // points to a `MaybeUninit<R>` slot, which is valid
            // for writing an `R` by transmute (same layout).
            // Caller's contract: every slot is written exactly
            // once before `collect_indexed` transmutes back to
            // `Vec<R>`.
            unsafe { base.add(i).write(f(start + i)); }
        }
    }

    if items.len() <= min_leaf {
        record_leaf(plan.site, || fill_leaf(items, start, f));
        return;
    }
    let cur_splits = if migrated { max_budget } else { splits };
    // Leaf at `<= 1` (see `bisect` for the doubling-bug rationale).
    if cur_splits <= 1 {
        record_leaf(plan.site, || fill_leaf(items, start, f));
        return;
    }
    let mid = items.len() >> 1;
    let (left, right) = items.split_at_mut(mid);
    let right_start = start + mid;
    let half = cur_splits >> 1;
    join_context(
        plan,
        |left_inj| {
            collect_inner(plan, left, start, f, min_leaf, half, max_budget, left_inj)
        },
        |right_stolen| {
            collect_inner(
                plan, right, right_start, f, min_leaf, half, max_budget, right_stolen,
            )
        },
    );
}

/// Heartbeat-gate threshold for sites with **no cv^2 evidence
/// yet**: below this item count a fresh site falls through to
/// plain SLAW [`collect_indexed`] with `min_leaf=1`; at or above
/// it, heartbeat runs. Once the site has variance evidence the
/// cv^2 routing in [`collect_indexed_heartbeat`] replaces this
/// gate entirely (uniform leaves force SLAW at any n; irregular
/// leaves A/B via policy arms from [`HEARTBEAT_MIN_ITEMS`]).
/// Uniform-per-iter workloads (matmul, slice ops) regress 3-4x
/// under heartbeat at small/mid N, so the cold default stays
/// conservative; heartbeat's value is **heterogeneous** work
/// where per-iter cost varies by 10x+ - the rdtsc-driven
/// promotion adapts to actual elapsed time without prior
/// knowledge of the cost distribution.
const HEARTBEAT_GATE_ITEMS: usize = 100_000;

/// Parallel indexed collect using **heartbeat scheduling** (Acar,
/// Charguéraud, Guatto, Rainey, Sieczkowski, "Heartbeat Scheduling:
/// Provable Efficiency for Nested Parallelism", PLDI 2018, §4
/// "Native support for parallel loops"). On each rdtsc tick that
/// crosses [`HEARTBEAT_CYCLES`], the current loop's remaining range
/// is split in half and the upper half is forked via
/// [`crate::sched::join_context`]. Each forked sub-range runs its
/// own polling counter so the promotion tree builds incrementally.
///
/// # When to use
///
/// Call this for workloads whose per-item cost is **clustered or
/// irregular** (front-loaded heavy items, depth-dependent
/// recursion, work-list iteration) when no accurate per-item
/// weight is available. The heartbeat-vs-SLAW choice is then made
/// per call from the call site's learned statistics:
///
/// - Caller set [`JobPlan::estimated_per_item_ns`] explicitly:
///   entry-only static decision - fully serial when the estimated
///   total is below the ~20us heartbeat quantum, plain SLAW
///   [`collect_indexed`] otherwise. Site-learned profile defaults
///   do NOT trip this gate; only the explicit flag does.
/// - Site cv^2 known and below the calibrated
///   `cv2_high_per_mille` threshold (uniform leaves): SLAW at any
///   `n` - heartbeat loses 3-4x on uniform-per-iter work.
/// - Site cv^2 known and high (irregular leaves): the site's
///   policy arms A/B heartbeat against SLAW from
///   `n >= HEARTBEAT_MIN_ITEMS` (`4_096`), adopt whichever EWMA
///   wins, and re-trial on a fixed cadence.
/// - Site cv^2 unknown (fresh site): legacy item-count gate -
///   SLAW below [`HEARTBEAT_GATE_ITEMS`] (`100_000`), heartbeat
///   at or above it.
///
/// # Alternatives for other shapes
///
/// - Uniform-per-iter workloads (matmul, slice ops, FFT butterfly
///   groups, LU trailing updates, HODLR tile compress): use
///   [`collect_indexed`] - SLAW's static budget is optimal. The
///   cv^2 route above converges to the same choice after a few
///   calls; calling `collect_indexed` directly skips the learning
///   window.
/// - Heterogeneous with known per-item weight (spmv with
///   `nnz_per_row`, etc.): use [`collect_indexed_token_bucket`]
///   which routes via the supplied weight instead of discovering
///   cost via rdtsc polling.
#[track_caller]
pub fn collect_indexed_heartbeat<R, F>(plan: &JobPlan, n: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync,
{
    if n == 0 {
        return Vec::new();
    }
    // Per-call-site identity from the caller's source location
    // (track_caller chain): heartbeat's serial spans record into
    // this site, and the site's cv^2 + policy arms drive the
    // heartbeat-vs-SLAW choice below.
    let plan_owned = plan
        .with_site_if_none(crate::sched::call_site::caller_site())
        .apply_site_class();
    let plan = &plan_owned;

    // Plan-estimate gate (entry-only, no rdtsc-polling in hot loop).
    // When the caller supplied an AUTHORITATIVE per-item cost, the
    // dispatcher can compute total estimated cost and make a static
    // decision at entry: serial when below the heartbeat quantum,
    // SLAW-parallel otherwise. Gated on the explicit flag because
    // site-learned profile defaults also populate
    // `estimated_per_item_ns`; those are routing hints and must not
    // preempt the cv^2-driven policy choice below.
    if plan.estimated_per_item_ns_explicit
        && let Some(total_ns) = plan.estimated_total_ns()
    {
        // HEARTBEAT_CYCLES is ~20µs at 3 GHz = ~20_000ns.
        const HEARTBEAT_NS: u64 = 20_000;
        if total_ns < HEARTBEAT_NS {
            // Run fully serial: no scheduler involvement at all.
            return serial_collect_indexed(n, &f);
        }
        // Estimate says heartbeat would fire immediately - skip the
        // serial-prefix-then-handoff dance and go straight to SLAW.
        return collect_indexed(plan, n, 1, f);
    }

    // Heartbeat wins on IRREGULAR per-item cost (the Acar model's
    // target shape) and loses 3-4x on uniform cost (matmul-shaped
    // work), so the site's observed cv^2 is the routing signal:
    //
    // - cv^2 known and LOW (uniform leaves): force SLAW at any n.
    // - cv^2 known and HIGH (irregular leaves): let the site's
    //   policy arms A/B heartbeat (arm 1) against SLAW (arm 0) from
    //   n >= HEARTBEAT_MIN_ITEMS, adopting whichever EWMA wins and
    //   re-trialling on a fixed cadence.
    // - cv^2 unknown (fresh site): the legacy item-count gate.
    let site = plan.site.expect("attached above");
    let cv2 = site.get().cv2_per_mille();
    let cv2_high = crate::sched::adaptive_profile::class_thresholds()
        .cv2_high_per_mille
        .load(std::sync::atomic::Ordering::Relaxed);
    let arm = match cv2 {
        Some(c) if c < cv2_high => crate::sched::call_site::PolicyArm::Default,
        Some(_) => site
            .get()
            .choose_arm(n >= HEARTBEAT_MIN_ITEMS),
        None => {
            if n < HEARTBEAT_GATE_ITEMS {
                crate::sched::call_site::PolicyArm::Default
            } else {
                crate::sched::call_site::PolicyArm::Alternative
            }
        }
    };

    if arm == crate::sched::call_site::PolicyArm::Default {
        let t0 = std::time::Instant::now();
        let out = collect_indexed(plan, n, 1, f);
        if cv2.is_some() {
            site.get().record_arm(arm, t0.elapsed().as_nanos() as u64);
        }
        return out;
    }

    let t0 = std::time::Instant::now();
    let mut buf: Vec<std::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: identical contract to `collect_indexed` - every slot
    // is written by `heartbeat_fill` before transmute.
    unsafe { buf.set_len(n); }

    heartbeat_fill(plan, &mut buf[..], 0, &f);

    let mut buf = std::mem::ManuallyDrop::new(buf);
    if cv2.is_some() {
        site.get().record_arm(arm, t0.elapsed().as_nanos() as u64);
    }
    // SAFETY: `heartbeat_fill` writes every slot exactly once;
    // `MaybeUninit<R>` and `R` share layout, so reconstructing
    // the Vec from the raw parts is sound.
    unsafe { Vec::from_raw_parts(buf.as_mut_ptr() as *mut R, n, buf.capacity()) }
}

/// Floor for the heartbeat policy arm once a site's cv^2 evidence
/// says the workload is irregular. Well below
/// [`HEARTBEAT_GATE_ITEMS`]: with variance evidence in hand, the
/// per-fork amortization is the only remaining constraint.
const HEARTBEAT_MIN_ITEMS: usize = 4096;

/// Fully-serial collect for small batches where dispatch overhead
/// would exceed total compute cost. Uses the same `MaybeUninit + ptr
/// write` pattern as `collect_indexed` so it shares the no-zero-init
/// path; just runs single-threaded on the calling thread.
fn serial_collect_indexed<R, F>(n: usize, f: &F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync,
{
    let mut buf: Vec<std::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: every slot is written by the loop below before transmute.
    unsafe { buf.set_len(n); }
    let base = buf.as_mut_ptr() as *mut R;
    for i in 0..n {
        // SAFETY: 0 <= i < n = capacity; writing into MaybeUninit<R>
        // via ptr.add(i) is the documented pattern.
        unsafe { base.add(i).write(f(i)); }
    }
    let mut buf = std::mem::ManuallyDrop::new(buf);
    // SAFETY: the loop above wrote every slot exactly once.
    // `MaybeUninit<R>` and `R` share layout, so reconstructing
    // the Vec from the raw parts is sound.
    unsafe { Vec::from_raw_parts(buf.as_mut_ptr() as *mut R, n, buf.capacity()) }
}

fn heartbeat_fill<R, F>(
    plan: &JobPlan,
    items: &mut [std::mem::MaybeUninit<R>],
    start: usize,
    f: &F,
)
where
    R: Send,
    F: Fn(usize) -> R + Sync,
{
    let n = items.len();
    if n == 0 {
        return;
    }
    // Per-fork overhead exceeds any promotion benefit below
    // MIN_LEAF_ITEMS; run fully serial in that regime. The fill is
    // one span for the SITE's statistics only (not a chunk leaf,
    // so it stays out of the process-global per-leaf counters).
    if n <= MIN_LEAF_ITEMS {
        let t0 = std::time::Instant::now();
        fill_serial(items, start, f);
        record_leaf_span_ns(plan.site, t0.elapsed().as_nanos() as u64);
        return;
    }
    let base = items.as_mut_ptr() as *mut R;
    let last_tick = read_tsc();
    // Serial-span start for the site classifier: everything filled
    // between here and a promotion (or loop completion) is one
    // leaf-equivalent span. Same TSC-as-approximate-ns convention
    // as record_leaf.
    let span_t0 = last_tick;

    // Serial fill until rdtsc tick crosses HEARTBEAT_CYCLES; on tick
    // bisect the remaining tail in half and fork the far half via
    // sched::join_context. Both halves recurse through heartbeat_fill
    // so each forked sub-range maintains its own polling counter -
    // promotions form a binary tree over time matching Acar/Rainey
    // PLDI 2018 §4 "Native support for parallel loops".
    let mut i = 0;
    while i < n {
        // SAFETY: i < n, base+i is the i-th slot. Writing R into
        // MaybeUninit<R> via ptr is the documented pattern.
        unsafe { base.add(i).write(f(start + i)); }
        i += 1;

        // Heartbeat check: every POLL_MASK+1 iterations. Require
        // remaining tail >= 2 * MIN_LEAF_ITEMS so the bisect produces
        // halves at or above the serial-only threshold.
        if (i & POLL_MASK) == 0 && (n - i) >= 2 * MIN_LEAF_ITEMS {
            let now = read_tsc();
            if now.wrapping_sub(last_tick) >= HEARTBEAT_CYCLES {
                record_leaf_span_ns(plan.site, now.wrapping_sub(span_t0));
                let (_filled, tail) = items.split_at_mut(i);
                let mid = tail.len() >> 1;
                let (near, far) = tail.split_at_mut(mid);
                let near_start = start + i;
                let far_start = start + i + mid;
                join_context(
                    plan,
                    |_left_inj| heartbeat_fill(plan, near, near_start, f),
                    |_right_stolen| heartbeat_fill(plan, far, far_start, f),
                );
                return;
            }
        }
    }
    // Loop completed without promoting: the whole run was one
    // serial span.
    record_leaf_span_ns(plan.site, read_tsc().wrapping_sub(span_t0));
}

/// Pure serial fill into a `MaybeUninit<R>` slice. Used by
/// `heartbeat_fill` when the remaining work is below the
/// per-fork-amortization threshold.
#[inline(always)]
fn fill_serial<R, F>(items: &mut [std::mem::MaybeUninit<R>], start: usize, f: &F)
where
    F: Fn(usize) -> R,
{
    let len = items.len();
    let base = items.as_mut_ptr() as *mut R;
    for i in 0..len {
        // SAFETY: 0 <= i < len = capacity; ptr.add(i).write is the
        // documented pattern.
        unsafe { base.add(i).write(f(start + i)); }
    }
}

/// Token-bucket heartbeat: an alternative to the rdtsc-tick
/// heartbeat that accumulates **work tokens** between checks. Each
/// item contributes a caller-supplied token count; promote-to-SLAW
/// fires when accumulated tokens cross [`TOKEN_BUCKET_PROMOTE`] OR
/// when an rdtsc tick crosses [`HEARTBEAT_CYCLES`]. Use for
/// heterogeneous-per-iter workloads where the rdtsc-mask-based
/// `collect_indexed_heartbeat` polls too rarely (heavy items) or
/// too often (light items) for the actual cost distribution.
///
/// `tokens_fn(i)` returns the per-item token count for index `i`.
/// Tokens are an opaque unit: callers can use bytes-of-work, FpN
/// limbs accessed, or coarse 1/10/100 buckets. The threshold is
/// in the same unit; pick `TOKEN_BUCKET_PROMOTE` to match expected
/// "first promote moment" for the workload.
///
#[track_caller]
pub fn collect_indexed_token_bucket<R, F, W>(
    plan: &JobPlan,
    n: usize,
    work_per_item: W,
    f: F,
) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync,
    W: Fn(usize) -> u32 + Sync,
{
    if n == 0 {
        return Vec::new();
    }
    // Per-call-site identity from the caller's source location
    // (track_caller chain); an outer attachment wins.
    let plan_owned = plan
        .with_site_if_none(crate::sched::call_site::caller_site())
        .apply_site_class();
    let plan = &plan_owned;

    // Entry gate identical to `collect_indexed_heartbeat`: skip the
    // token-bucket machinery if the caller's AUTHORITATIVE estimate
    // says the whole batch fits in one heartbeat tick (explicit-only
    // for the same reason as the heartbeat entry: site-learned
    // profile defaults also populate the estimate field).
    if plan.estimated_per_item_ns_explicit
        && let Some(total_ns) = plan.estimated_total_ns()
    {
        const HEARTBEAT_NS: u64 = 20_000;
        if total_ns < HEARTBEAT_NS {
            return serial_collect_indexed(n, &f);
        }
        return collect_indexed(plan, n, 1, f);
    }

    if n < HEARTBEAT_GATE_ITEMS {
        return collect_indexed(plan, n, 1, f);
    }

    let mut buf: Vec<std::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: `MaybeUninit<R>` needs no initialization; `set_len`
    // up to capacity is sound. `token_bucket_fill` below writes
    // every slot before we transmute.
    unsafe { buf.set_len(n); }
    token_bucket_fill(plan, &mut buf[..], 0, &f, &work_per_item);
    let mut buf = std::mem::ManuallyDrop::new(buf);
    // SAFETY: `token_bucket_fill` wrote every slot exactly once.
    // `MaybeUninit<R>` and `R` share layout, so reconstructing
    // the Vec from the raw parts is sound.
    unsafe { Vec::from_raw_parts(buf.as_mut_ptr() as *mut R, n, buf.capacity()) }
}

/// Parallel indexed collect with **Tiny-Tasks model** chunk sizing.
/// Consults [`JobPlan::optimal_chunk_count`] to derive `min_leaf =
/// n / optimal_chunks` instead of using the caller-supplied floor.
/// Falls back to plain [`collect_indexed`] with `min_leaf=1` when
/// the plan lacks the required estimates (`estimated_per_item_ns`
/// AND `task_overhead_ns`).
#[track_caller]
pub fn collect_indexed_tiny_tasks<R, F>(plan: &JobPlan, n: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync,
{
    if n == 0 {
        return Vec::new();
    }
    let workers = global_local_arena().total_workers().max(1);
    let chunks = plan.optimal_chunk_count(workers);
    let min_leaf = match chunks {
        Some(c) if c > 0 => (n / c as usize).max(1),
        _ => 1,
    };
    collect_indexed(plan, n, min_leaf, f)
}

// NOTE: no `collect_indexed_idempotent` wire helper exists; the
// idempotent collect is a documented negative result. Bench on
// Zen+ AVX2:
//   n=10k:   standard 30.5us, idempotent 87.5us (2.87x slower)
//   n=100k:  standard 97.7us, idempotent 267us  (2.74x slower)
//   n=1M:    standard 2.06ms, idempotent 2.40ms (1.17x slower)
// The Relaxed-bitmap CAS overhead per slot exceeds any saved
// Acquire-fence benefit on x86-64 (TSO host: Acquire is
// essentially free, so there is no fence to save). The
// IdempotentJob trait stays in src/sched/idempotent.rs as a
// semantically useful marker trait.

/// Token-bucket promotion threshold in caller-defined work units.
/// Default sized so that ~20µs of work at "1 token = 1ns" accumulates
/// before promotion. Tunable per workload; the caller's `tokens_fn`
/// effectively chooses this value's meaning via the unit it picks.
const TOKEN_BUCKET_PROMOTE: u64 = 20_000;

fn token_bucket_fill<R, F, W>(
    plan: &JobPlan,
    items: &mut [std::mem::MaybeUninit<R>],
    start: usize,
    f: &F,
    tokens_fn: &W,
)
where
    R: Send,
    F: Fn(usize) -> R + Sync,
    W: Fn(usize) -> u32 + Sync,
{
    let n = items.len();
    if n == 0 {
        return;
    }
    let base = items.as_mut_ptr() as *mut R;
    let mut last_tick = read_tsc();
    // Serial-prefix span start for the site classifier (same
    // TSC-as-approximate-ns convention as record_leaf).
    let span_t0 = last_tick;
    let mut tokens_since_check: u64 = 0;

    let workers = global_local_arena().total_workers();
    let multiplier = crate::sched::split_observer::split_multiplier() as usize;
    let max_budget = workers.saturating_mul(multiplier).max(1);

    let mut i = 0;
    while i < n {
        let idx = start + i;
        // SAFETY: i < n; base.add(i) is a valid slot.
        unsafe { base.add(i).write(f(idx)); }
        tokens_since_check =
            tokens_since_check.saturating_add(tokens_fn(idx) as u64);
        i += 1;

        // Token threshold OR rdtsc tick: either signal triggers a
        // promote-to-SLAW. Token check is cheap (one branch); rdtsc
        // check fires only every POLL_MASK+1 items to keep cost
        // amortized even when each token is small.
        let token_trip = tokens_since_check >= TOKEN_BUCKET_PROMOTE;
        let tick_trip = (i & POLL_MASK) == 0 && {
            let now = read_tsc();
            let elapsed = now.wrapping_sub(last_tick);
            if elapsed >= HEARTBEAT_CYCLES {
                true
            } else {
                last_tick = now;
                false
            }
        };
        if (token_trip || tick_trip) && i + 1 < n {
            let remaining = n - i;
            if remaining >= 2 {
                record_leaf_span_ns(plan.site, read_tsc().wrapping_sub(span_t0));
                let (_filled, tail) = items.split_at_mut(i);
                collect_inner(
                    plan, tail, start + i, f, 1,
                    max_budget, max_budget, false,
                );
                return;
            }
            tokens_since_check = 0;
            last_tick = read_tsc();
        }
    }
    // Loop completed without promoting: one serial span.
    record_leaf_span_ns(plan.site, read_tsc().wrapping_sub(span_t0));
}

/// Recursive bisection with a fixed `chunk_size` floor + adaptive
/// splitter. Same SLAW pattern as `bisect`.
fn bisect_fixed<T, F>(
    plan: &JobPlan,
    items: &mut [T],
    op: &F,
    splits: usize,
    max_budget: usize,
    chunk_size: usize,
    migrated: bool,
)
where
    T: Send,
    F: Fn(&mut [T]) + Sync,
{
    if items.len() <= chunk_size {
        record_leaf(plan.site, || op(items));
        return;
    }
    let cur_splits = if migrated { max_budget } else { splits };
    // Leaf at `<= 1` (see `bisect` for the doubling-bug rationale).
    if cur_splits <= 1 {
        record_leaf(plan.site, || op(items));
        return;
    }
    let mid = items.len() / 2;
    let (left, right) = items.split_at_mut(mid);
    let half = cur_splits >> 1;
    join_context(
        plan,
        |left_inj| bisect_fixed(plan, left, op, half, max_budget, chunk_size, left_inj),
        |right_stolen| bisect_fixed(plan, right, op, half, max_budget, chunk_size, right_stolen),
    );
}

/// Parallel reduction: split `items` into chunks, fold each chunk
/// with `fold`, then combine pairs with `reduce` until one value
/// remains. The combine tree is binary and bit-exact for non-
/// commutative `reduce` only when called with associative
/// operations - the order of combination depends on the
/// recursion shape.
///
/// `init` is called once per chunk to seed its accumulator (so
/// each chunk gets a fresh `init` value, allowing thread-local
/// accumulators of types that aren't `Copy`).
///
/// # Strategy selection (adaptive)
///
/// Two dispatch shapes, picked automatically per call:
///
/// - **Flat fan-out**: pre-split into `workers * split_multiplier()`
///   contiguous chunks and dispatch via
///   [`crate::sched::cooperative_join_n_flat`] in one burst, then
///   parallel-tree-reduce the chunk results. Wins on uniform-cost
///   trivial-reduce workloads like histogram (256-bin element-wise
///   add, ~4us per merge) where the bisect's depth-first descent
///   ramp-up cost dominates.
/// - **Recursive bisect** (default): the [`reduce_inner`]
///   tree-bisect path. Wins when chunk costs vary (steal-driven
///   rebalancing absorbs variance) or when `reduce` is heavy
///   (word_count's HashMap merge at ~5ms per merge: the
///   serial-final-reduce cost in the flat path dominates).
///
/// Pick signal: observer-learned **reduce-cost**. The OOB
/// calibration block at the top of every reduce_chunks call
/// times one populated-accumulator `reduce(a, b)` merge on the
/// bench thread (bounded at 16 samples per call site) and
/// publishes the average to the caller's
/// [`crate::sched::call_site::CallSiteState`]. Subsequent calls
/// read it: if observed average is below the calibrated
/// trivial-reduce ceiling AND input is large enough for clean
/// chunks above MIN_LEAF_ITEMS, flat-fanout fires. Otherwise
/// bisect.
///
/// The reduce-cost observer lives on the caller's
/// [`crate::sched::call_site::CallSiteState`], so a histogram
/// merge and a HashMap merge observe their own costs even through
/// the same generic entry (see [`crate::sched::call_site`] for why
/// a `static` cannot provide that identity).
#[track_caller]
pub fn reduce_chunks<T, A, F, R, I>(
    plan: &JobPlan,
    items: &[T],
    init: I,
    fold: F,
    reduce: R,
) -> A
where
    T: Sync,
    A: Send + 'static,
    I: Fn() -> A + Sync,
    F: Fn(A, &[T]) -> A + Sync,
    R: Fn(A, A) -> A + Sync,
{
    use std::sync::atomic::Ordering::Relaxed;
    // Per-call-site identity from the caller's source location
    // (track_caller chain): hosts the reduce-cost observer and
    // flows into the inner collect/for_each dispatches.
    let plan_owned = plan.with_site_if_none(crate::sched::call_site::caller_site());
    let plan = &plan_owned;
    let site = plan.site.expect("attached above").get();
    // Trivial-reduce ceiling from the live calibratable thresholds
    // (default 30k cycles ~= 10us at 3 GHz; separates a 256-bin
    // element-wise add from a HashMap merge). Calibration measures
    // the reference merge in the running binary, so the value is
    // always in this build profile's own cycle units.
    let trivial_reduce_cycles_threshold: u64 =
        crate::sched::adaptive_profile::class_thresholds()
            .trivial_reduce_cycles
            .load(Relaxed);

    let workers = global_local_arena().total_workers();
    let multiplier = crate::sched::split_observer::split_multiplier() as usize;
    // Default chunk-count budget: workers * multiplier (multiplier=2
    // by default -> 32 chunks on a 16-worker host).
    //
    // Audit hook: FLYNNEL_REDUCE_CHUNKS_CHUNKS=<N> overrides the
    // default so chunk-count-parity-with-rayon experiments can run
    // without touching the bench code. Production callers never set
    // this; it's strictly for in-source investigation. Reference
    // rayon chunk counts on 16M-byte inputs (bench audit 2026-06-17):
    //   histogram   par_chunks(1MB)   -> 16 chunks
    //   word_count  par_chunks(256KB) -> 64 chunks
    //   kmeans      par_chunks(8KB)   -> 98 chunks
    let max_budget = match std::env::var("FLYNNEL_REDUCE_CHUNKS_CHUNKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(n) if n >= 1 => n,
        _ => workers.saturating_mul(multiplier).max(1),
    };

    // OOB CALIBRATION: time ONE reduce call on the bench thread
    // (never on workers, to avoid the wait-loop recursion +
    // measure_reduce stack-overflow problem documented in
    // reduce_inner). Bounded at MAX_SAMPLES; after convergence,
    // overhead drops to zero (the if-branch is skipped).
    //
    // CRITICAL: time reduce of representative folded accumulators,
    // not reduce of init() pairs. Naive reduce(init(), init())
    // measures EMPTY-input cost which is O(0) for HashMap-merge
    // reducers (word_count) and similar collection-shaped
    // accumulators -- it classifies them as TRIVIAL and routes
    // them to flat-fanout, which kills their perf because the
    // tree-reduce final stage merges fully-populated maps
    // serially-on-leaves rather than via bisect's natural
    // parallel-merge along the recursion unwind.
    //
    // Folding a small CAL_SLICE_LEN sample produces accumulators
    // with REAL populated state, then reduce times the actual
    // merge cost. Total bench-thread overhead: 16 samples * (fold
    // cost over CAL_SLICE_LEN items + reduce of two populated
    // values). For histogram: ~16 * 2us = 32us total. For
    // word_count: ~16 * 50us = 0.8ms total. Negligible against a
    // 3-second criterion warm-up.
    if site.reduce_cost_wants_sample() {
        let cal_slice_len = MIN_LEAF_ITEMS.min(items.len());
        if cal_slice_len > 0 {
            let cal_a = fold(init(), &items[..cal_slice_len]);
            let cal_b = fold(init(), &items[..cal_slice_len]);
            let t0 = read_tsc();
            let drained = reduce(cal_a, cal_b);
            let dt = read_tsc().wrapping_sub(t0);
            site.record_reduce_cost_sample(dt);
            drop(drained);
        }
    }

    // Read the observer signal. `None` = cold path: not enough
    // data, default to bisect (the always-correct path).
    let observed_trivial = site
        .reduce_cost_avg_cycles()
        .is_some_and(|avg_cycles| avg_cycles < trivial_reduce_cycles_threshold);

    // ADAPTIVE FAST PATH (gated): trivial reduce + large input
    // -> flat fan-out. Audit found this path consistently slower
    // than bisect for the three characterized reduce_chunks
    // workloads because cooperative_join_n_flat's external_dispatch
    // + LockLatch overhead per tree-reduce round (5+ rounds)
    // dominates the "all workers come online faster" benefit.
    // Bisect's in-worker join_context is cheaper.
    //
    // FLYNNEL_ENABLE_FLAT_FANOUT=1 re-enables the path for in-source
    // experiments (verifying the audit finding hasn't drifted, or
    // benching a future fan_out_in_worker-only refactor). Default
    // off: bisect-only.
    let flat_enabled = std::env::var_os("FLYNNEL_ENABLE_FLAT_FANOUT")
        .is_some_and(|v| v != "0" && v != "false" && !v.is_empty());
    if flat_enabled
        && observed_trivial
        && items.len() >= max_budget.saturating_mul(MIN_LEAF_ITEMS)
    {
        record_reduce_chunks_path(ReduceChunksPath::Flat);
        return reduce_chunks_flat(plan, items, &init, &fold, &reduce, max_budget);
    }
    record_reduce_chunks_path(ReduceChunksPath::Bisect);
    reduce_inner(plan, items, &init, &fold, &reduce, max_budget, max_budget, false)
}

/// Which dispatch path [`reduce_chunks`] last selected. Exposed
/// for the bench-audit test that verifies the observer-learned
/// router actually routes each consumer to the right path
/// (histogram -> Flat, word_count -> Bisect, etc.). Without this
/// surface, "the adaptive routing works" is an unfalsifiable
/// claim instead of a verified one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceChunksPath {
    /// `reduce_chunks_flat` (`cooperative_join_n_flat` + parallel
    /// tree-reduce). Picked when observer classifies reduce as
    /// trivial AND input is large enough for clean chunks.
    Flat,
    /// `reduce_inner` (recursive bisect). Default; picked when
    /// observer hasn't converged yet OR observed reduce is
    /// non-trivial OR input is too small.
    Bisect,
}

thread_local! {
    static LAST_REDUCE_CHUNKS_PATH: std::cell::Cell<Option<ReduceChunksPath>> =
        const { std::cell::Cell::new(None) };
}

#[inline(always)]
fn record_reduce_chunks_path(p: ReduceChunksPath) {
    LAST_REDUCE_CHUNKS_PATH.with(|c| c.set(Some(p)));
}

/// Read the dispatch path that [`reduce_chunks`] selected on its
/// most recent call on the CURRENT thread. Returns `None` if
/// reduce_chunks has not been called on this thread yet.
///
/// Bench-audit hook: a test that calls reduce_chunks with a
/// known-trivial reducer (e.g., sum) and asserts this returns
/// `Some(Flat)` after the observer warm-up converges; and
/// symmetrically for known-heavy reducers (sleep-spinning) ->
/// `Some(Bisect)`. Without this surface, the adaptive router's
/// correctness is unverifiable.
pub fn last_reduce_chunks_path() -> Option<ReduceChunksPath> {
    LAST_REDUCE_CHUNKS_PATH.with(|c| c.get())
}

/// Flat fan-out reduce: split `items` into `n_chunks` contiguous
/// slices, dispatch each as one closure via
/// [`crate::sched::cooperative_join_n_flat`], then parallel-tree-
/// reduce the chunk results. Used by [`reduce_chunks`] when its
/// observer signal classifies the call site as trivial-reduce.
///
/// # Safety
///
/// Uses raw-pointer arithmetic + `slice::from_raw_parts` to
/// reconstruct each chunk inside the dispatched closure, and
/// raw-pointer lifetime-laundering of `&I` and `&F` to fit the
/// `'static` closure requirement of `cooperative_join_n_flat`
/// without forcing reduce_chunks callers to provide `'static`
/// reducer closures.
///
/// Lifetime argument: `cooperative_join_n_flat` blocks the calling
/// thread until every closure has completed before returning, so
/// the caller's `items`, `init`, and `fold` references remain
/// valid for the entire closure lifetime. Pointer transport via
/// `usize` sidesteps the 2021+ disjoint-field-capture interaction
/// with raw pointer types.
///
/// Trait-bound argument: `T: Sync` makes `&[T]: Send` so the
/// reconstructed chunk reference is sound on a peer worker;
/// `I: Sync` and `F: Sync` make `&I` and `&F` `Send` so the
/// address-transport via `usize` is sound.
fn reduce_chunks_flat<T, A, F, R, I>(
    plan: &JobPlan,
    items: &[T],
    init: &I,
    fold: &F,
    reduce: &R,
    n_chunks: usize,
) -> A
where
    T: Sync,
    A: Send + 'static,
    I: Fn() -> A + Sync,
    F: Fn(A, &[T]) -> A + Sync,
    R: Fn(A, A) -> A + Sync,
{
    let len = items.len();
    debug_assert!(
        len >= n_chunks * MIN_LEAF_ITEMS,
        "reduce_chunks_flat called below chunk-size floor: len={len}, n_chunks={n_chunks}, MIN_LEAF_ITEMS={MIN_LEAF_ITEMS}",
    );
    let chunk_size = len.div_ceil(n_chunks);
    let n_real = len.div_ceil(chunk_size);
    let item_size = std::mem::size_of::<T>();
    let base_addr: usize = items.as_ptr() as usize;
    let fold_addr: usize = fold as *const F as usize;
    let init_addr: usize = init as *const I as usize;

    let closures: Vec<Box<dyn FnOnce() -> A + Send>> = (0..n_real)
        .map(|i| {
            let start = i * chunk_size;
            let chunk_len = chunk_size.min(len - start);
            let chunk_addr = base_addr + start * item_size;
            let boxed: Box<dyn FnOnce() -> A + Send> = Box::new(move || {
                // SAFETY: items, init, and fold are alive for the
                // entire cooperative_join_n_flat span (the call
                // blocks until every closure has completed).
                // start + chunk_len <= len by construction above.
                let chunk_ptr = chunk_addr as *const T;
                let chunk: &[T] = unsafe {
                    std::slice::from_raw_parts(chunk_ptr, chunk_len)
                };
                let fold_ref: &F = unsafe { &*(fold_addr as *const F) };
                let init_ref: &I = unsafe { &*(init_addr as *const I) };
                fold_ref(init_ref(), chunk)
            });
            boxed
        })
        .collect();

    let results = crate::sched::cooperative_join_n_flat(plan, closures);
    tree_reduce_in_parallel(plan, results, reduce)
}

/// Pair-merge `results` in parallel via [`crate::sched::
/// cooperative_join_n_flat`] until a single value remains. Used
/// by [`reduce_chunks_flat`] to combine its per-chunk results.
///
/// Matches the recursive bisect's natural tree-reduce shape: the
/// `reduce(la, ra)` call at each `join_context` level. Total work
/// = N-1 reduce calls (same as serial); wall-clock = ceil(log2(N))
/// * reduce_cost. For word_count's ~5ms HashMap merges over 16
///   results: 4 levels * 5ms = 20ms vs 15 * 5ms = 75ms serial.
fn tree_reduce_in_parallel<A, R>(plan: &JobPlan, mut results: Vec<A>, reduce: &R) -> A
where
    A: Send + 'static,
    R: Fn(A, A) -> A + Sync,
{
    while results.len() > 1 {
        let reduce_addr: usize = reduce as *const R as usize;
        // Pop an odd tail off so the remaining count is even.
        let odd_tail = if results.len() % 2 == 1 {
            results.pop()
        } else {
            None
        };
        let mut pairs: Vec<(A, A)> = Vec::with_capacity(results.len() / 2);
        let mut iter = results.into_iter();
        loop {
            let a = match iter.next() {
                Some(x) => x,
                None => break,
            };
            let b = iter
                .next()
                .expect("even results count after odd-tail pop");
            pairs.push((a, b));
        }
        let closures: Vec<Box<dyn FnOnce() -> A + Send>> = pairs
            .into_iter()
            .map(|(a, b)| {
                let boxed: Box<dyn FnOnce() -> A + Send> = Box::new(move || {
                    // SAFETY: reduce is alive for the entire
                    // tree_reduce_in_parallel call;
                    // cooperative_join_n_flat blocks until every
                    // closure completes. R: Sync means &R: Send.
                    let reduce_ref: &R = unsafe { &*(reduce_addr as *const R) };
                    reduce_ref(a, b)
                });
                boxed
            })
            .collect();
        let mut next_round = crate::sched::cooperative_join_n_flat(plan, closures);
        if let Some(extra) = odd_tail {
            next_round.push(extra);
        }
        results = next_round;
    }
    results.into_iter().next().expect("at least 1 result")
}

#[allow(clippy::too_many_arguments)]
fn reduce_inner<T, A, F, R, I>(
    plan: &JobPlan,
    items: &[T],
    init: &I,
    fold: &F,
    reduce: &R,
    splits: usize,
    max_budget: usize,
    migrated: bool,
) -> A
where
    T: Sync,
    A: Send,
    I: Fn() -> A + Sync,
    F: Fn(A, &[T]) -> A + Sync,
    R: Fn(A, A) -> A + Sync,
{
    if items.len() <= MIN_LEAF_ITEMS {
        return fold(init(), items);
    }
    let cur_splits = if migrated { max_budget } else { splits };
    // Leaf at `<= 1` (see `bisect` for the doubling-bug rationale).
    if cur_splits <= 1 {
        return fold(init(), items);
    }
    let mid = items.len() / 2;
    let (left, right) = items.split_at(mid);
    let half = cur_splits >> 1;
    let (la, ra) = join_context(
        plan,
        |left_inj| reduce_inner(plan, left, init, fold, reduce, half, max_budget, left_inj),
        |right_stolen| reduce_inner(plan, right, init, fold, reduce, half, max_budget, right_stolen),
    );
    reduce(la, ra)
}

/// Apply `op` to each element of `items` in parallel, with one
/// dispatched task per element. The leaf chunk size is 1: every
/// element is one unit of work for the scheduler.
///
/// Use this when each element's `op` is large enough (~10us+) to
/// amortize per-task dispatch overhead. For small per-element work
/// use [`for_each_chunk`] instead, which groups multiple elements
/// per leaf via the bisect splitter.
///
/// Examples of one-task-per-element work: per-block high-precision
/// arithmetic, per-row matrix factorization, per-particle PDE
/// step, per-image GPU dispatch coordinator. The common shape is
/// "few large units" rather than "many small units".
#[track_caller]
pub fn par_map_in_place<T, F>(plan: &JobPlan, items: &mut [T], op: F)
where
    T: Send,
    F: Fn(&mut T) + Sync,
{
    for_each_fixed_chunk(plan, items, 1, |slice| {
        for x in slice {
            op(x);
        }
    });
}

/// Parallel zip-apply: for each index `i`, call `op(&mut lhs[i],
/// &rhs[i])`. Mutates `lhs` in place; `rhs` is read-only. Panics
/// if `lhs.len() != rhs.len()`.
///
/// One task per index by default (matches [`par_map_in_place`]'s
/// granule). Useful when paired indices need disjoint mutable
/// access that the standard slice borrow checker cannot prove
/// (e.g., independent per-element ops with no cross-element
/// dependency).
///
/// Implementation routes through `for_each_fixed_chunk` over an
/// index-list and uses raw-pointer arithmetic internally to
/// produce disjoint `&mut T` per index. Each chunk owns a
/// disjoint contiguous index range so the pointer dereferences
/// never alias.
#[track_caller]
pub fn par_zip_apply<T, U, F>(plan: &JobPlan, lhs: &mut [T], rhs: &[U], op: F)
where
    T: Send,
    U: Sync,
    F: Fn(&mut T, &U) + Sync,
{
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "par_zip_apply requires matching slice lengths"
    );
    let n = lhs.len();
    if n == 0 {
        return;
    }
    // Address-as-usize sidesteps the edition-2021 disjoint-capture
    // inspection that sees through *mut T newtypes; routing via
    // primitive `usize` keeps the closure capture trivially Sync.
    let lhs_addr: usize = lhs.as_mut_ptr() as usize;
    let mut indices: Vec<usize> = (0..n).collect();
    for_each_fixed_chunk(plan, &mut indices, 1, move |chunk| {
        let lhs_ptr: *mut T = lhs_addr as *mut T;
        for &i in chunk.iter() {
            // SAFETY: each index `i` belongs to exactly one chunk
            // by construction, so the resulting `&mut T` does not
            // alias any other concurrent `&mut T` on lhs.
            let t: &mut T = unsafe { &mut *lhs_ptr.add(i) };
            op(t, &rhs[i]);
        }
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::sched::plan::JobPlan;

    #[test]
    fn for_each_chunk_zero_items_is_noop() {
        let mut v: Vec<u32> = Vec::new();
        let plan = JobPlan::new(6, 1);
        for_each_chunk(&plan, &mut v, |slice| {
            for x in slice {
                *x += 1;
            }
        });
        assert!(v.is_empty());
    }

    #[test]
    fn bisect_variants_all_produce_identical_result() {
        // Correctness baseline for the two BisectVariant entries:
        // same input through default (None, the lazy-steal path) and
        // through each variant must produce the same output. Forces n large enough that the probe-and-decide
        // path does NOT fire (`n >= workers * MIN_LEAF_ITEMS` with
        // workers ~16 and MIN_LEAF_ITEMS=256 -> n >= 4096). At n=20000
        // every variant exercises the bisect path it routes through.
        let n = 20_000usize;
        let template: Vec<u32> = (0..n as u32).collect();

        let run = |variant: Option<crate::sched::plan::BisectVariant>| -> Vec<u32> {
            let mut v = template.clone();
            let mut plan = JobPlan::new(6, n as u32);
            plan.bisect_variant = variant;
            for_each_chunk(&plan, &mut v, |slice| {
                for x in slice {
                    *x = x.wrapping_mul(3).wrapping_add(7);
                }
            });
            v
        };

        let baseline = run(None);
        assert_eq!(run(Some(crate::sched::plan::BisectVariant::ProducerMaxLenWorkers)), baseline);
        assert_eq!(run(Some(crate::sched::plan::BisectVariant::RayonStyleReplenish)), baseline);
    }

    #[test]
    fn for_each_chunk_touches_every_element_exactly_once() {
        let n = 10_000usize;
        let mut v: Vec<u32> = (0..n as u32).collect();
        let plan = JobPlan::new(6, n as u32);
        for_each_chunk(&plan, &mut v, |slice| {
            for x in slice {
                *x = x.wrapping_mul(2);
            }
        });
        for (i, &x) in v.iter().enumerate() {
            assert_eq!(x, (i as u32).wrapping_mul(2),
                "element {i} = {x}, expected {}", (i as u32).wrapping_mul(2));
        }
    }

    #[test]
    fn for_each_chunk_small_input_runs_serial() {
        // An input whose measured work is below the dispatch floor
        // (no explicit cost estimate, n < workers * MIN_LEAF_ITEMS)
        // runs serially on the calling thread without entering the
        // pool. A quarter of the leaf floor keeps the work below the
        // floor even when the suite saturates the host and a one-add
        // item measures 400 ns.
        //
        // The runtime may probe-and-decide (one small probe + tail) so
        // op CAN be called multiple times, but every call must come
        // from the calling thread (no pool dispatch). This is what
        // separates inline-collapse from pool dispatch.
        use std::sync::atomic::{AtomicUsize, Ordering};
        // The premise holds under the PortBound global profile: a
        // LatencyBound global activates SMT and a one-item leaf
        // floor, which dispatches even this size. Hold the profile
        // lock so the migration tests cannot move it underneath.
        let _profile = crate::sched::adaptive_profile::global_profile_test_lock();
        crate::sched::adaptive_profile::migrate_dispatch_profile(
            crate::DispatchProfile::PortBound,
        );
        let n = MIN_LEAF_ITEMS / 4;
        let mut v: Vec<u32> = (0..n as u32).collect();
        let plan = JobPlan::new(6, n as u32);
        let snap = format!(
            "use_smt={} est_pi={:?} explicit={} oversub={:?} global={:?}",
            plan.use_smt,
            plan.estimated_per_item_ns,
            plan.estimated_per_item_ns_explicit,
            plan.oversubscription_log2,
            crate::sched::adaptive_profile::active_dispatch_profile(),
        );
        let calling_thread = std::thread::current().id();
        let total_processed = AtomicUsize::new(0);
        // Violations are collected and asserted AFTER the dispatch
        // returns: a panic inside a pool-worker leaf is re-raised on
        // the test thread with its message swallowed by the pool's
        // propagation, which hides the diagnostic snapshot.
        let off_thread_calls = AtomicUsize::new(0);
        for_each_chunk(&plan, &mut v, |slice| {
            if std::thread::current().id() != calling_thread {
                off_thread_calls.fetch_add(1, Ordering::Relaxed);
            }
            total_processed.fetch_add(slice.len(), Ordering::Relaxed);
            for x in slice {
                *x += 1000;
            }
        });
        assert_eq!(
            off_thread_calls.load(Ordering::Relaxed),
            0,
            "small input must NOT dispatch to the pool [{snap}]"
        );
        assert_eq!(
            total_processed.load(Ordering::Relaxed),
            n,
            "all items must be processed exactly once across probe + tail"
        );
        for (i, &x) in v.iter().enumerate() {
            assert_eq!(x, i as u32 + 1000);
        }
    }

    #[test]
    fn inline_collapse_threshold_is_in_its_band_and_cached() {
        // Before or during the background calibration the query
        // answers the floor; once calibrated, the measured value.
        let q = inline_collapse_threshold_ns();
        assert!(
            q == INLINE_COLLAPSE_FLOOR_NS || (INLINE_COLLAPSE_FLOOR_NS..=INLINE_COLLAPSE_CAP_NS).contains(&q),
            "query before calibration: {q} ns"
        );
        let t0 = std::time::Instant::now();
        let t = calibrate_inline_collapse_threshold();
        let cost = t0.elapsed();
        assert!((INLINE_COLLAPSE_FLOOR_NS..=INLINE_COLLAPSE_CAP_NS).contains(&t), "threshold {t} ns");
        assert_eq!(inline_collapse_threshold_ns(), t, "the query hands out the installed value");
        eprintln!("inline collapse threshold on this host: {t} ns (measured in {cost:?})");
    }

    #[test]
    fn triple_min_leaf_small_explicit_estimate_runs_on_the_caller() {
        // 1000 items at 3 ns each is 3 us of work, under the
        // dispatch floor: the body runs once, on the calling thread.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let n = 1000usize;
        let a: Vec<u32> = (0..n as u32).collect();
        let b: Vec<u32> = vec![1; n];
        let mut out = vec![0u32; n];
        let plan = JobPlan::new(6, n as u32).with_estimated_per_item_ns(3);
        let caller = std::thread::current().id();
        let off_thread = AtomicUsize::new(0);
        let calls = AtomicUsize::new(0);
        for_each_chunk_triple_min_leaf(&plan, &mut out, &a, &b, 64, |o, x, y| {
            if std::thread::current().id() != caller {
                off_thread.fetch_add(1, Ordering::Relaxed);
            }
            calls.fetch_add(1, Ordering::Relaxed);
            for ((o, x), y) in o.iter_mut().zip(x).zip(y) {
                *o = x + y;
            }
        });
        assert_eq!(off_thread.load(Ordering::Relaxed), 0, "must not dispatch to the pool");
        assert_eq!(calls.load(Ordering::Relaxed), 1, "one body call over the whole slice");
        assert!(out.iter().enumerate().all(|(i, &v)| v == i as u32 + 1));
    }

    #[test]
    fn indexed_min_leaf_small_explicit_estimate_runs_on_the_caller() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let n = 1000usize;
        let mut v: Vec<u32> = vec![0; n];
        let plan = JobPlan::new(6, n as u32).with_estimated_per_item_ns(3);
        let caller = std::thread::current().id();
        let off_thread = AtomicUsize::new(0);
        let calls = AtomicUsize::new(0);
        for_each_chunk_indexed_min_leaf(&plan, &mut v, 1, |start, chunk| {
            if std::thread::current().id() != caller {
                off_thread.fetch_add(1, Ordering::Relaxed);
            }
            calls.fetch_add(1, Ordering::Relaxed);
            for (i, x) in chunk.iter_mut().enumerate() {
                *x = (start + i) as u32;
            }
        });
        assert_eq!(off_thread.load(Ordering::Relaxed), 0, "must not dispatch to the pool");
        assert_eq!(calls.load(Ordering::Relaxed), 1, "one body call over the whole slice");
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as u32));
    }

    #[test]
    fn for_each_fixed_chunk_respects_chunk_size() {
        let n = 1000usize;
        let chunk = 64;
        let mut v: Vec<u32> = vec![0; n];
        let plan = JobPlan::new(6, n as u32);
        let sizes_seen = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
        let sizes_clone = Arc::clone(&sizes_seen);
        for_each_fixed_chunk(&plan, &mut v, chunk, |slice| {
            sizes_clone.lock().unwrap().push(slice.len());
            for x in slice {
                *x = 7;
            }
        });
        // Every chunk passed to op must be <= chunk size.
        for size in sizes_seen.lock().unwrap().iter() {
            assert!(*size <= chunk,
                "chunk size {} exceeded the {chunk}-item ceiling", size);
        }
        // Every element was touched.
        assert!(v.iter().all(|&x| x == 7));
    }

    #[test]
    fn for_each_chunk_parallel_sum_matches_serial() {
        // Use atomic counter to verify total work matches expected
        // count. for_each_chunk must produce the same total work
        // as a serial loop.
        let n = 50_000usize;
        let mut v: Vec<u32> = (0..n as u32).collect();
        let counter = Arc::new(AtomicU64::new(0));
        let plan = JobPlan::new(6, n as u32);
        for_each_chunk(&plan, &mut v, |slice| {
            let local: u64 = slice.iter().map(|&x| x as u64).sum();
            counter.fetch_add(local, Ordering::Relaxed);
        });
        let expected: u64 = (0..n as u64).sum();
        assert_eq!(counter.load(Ordering::Relaxed), expected,
            "parallel sum {} != serial sum {expected}",
            counter.load(Ordering::Relaxed));
    }

    #[test]
    fn reduce_chunks_sum_matches_serial() {
        let n = 100_000usize;
        let v: Vec<u64> = (0..n as u64).collect();
        let plan = JobPlan::new(6, n as u32);
        let parallel = reduce_chunks(
            &plan,
            &v,
            || 0u64,
            |acc, chunk| acc + chunk.iter().sum::<u64>(),
            |a, b| a + b,
        );
        let serial: u64 = v.iter().sum();
        assert_eq!(parallel, serial);
    }

    #[test]
    fn reduce_chunks_large_trivial_reduce_matches_serial() {
        // 64 KiB items: above workers*MIN_LEAF_ITEMS (~4096) so
        // flat-fanout path can activate after observer converges.
        // Smaller than the bench input so debug-profile stack
        // usage stays well under the 8 MiB worker ceiling.
        let n = 64 * 1024usize;
        let v: Vec<u32> = (0..n as u32).collect();
        let plan = JobPlan::new(6, n as u32).with_estimated_per_item_ns(1);
        let mut last = [0u64; 256];
        for _ in 0..5 {
            last = reduce_chunks(
                &plan, &v,
                || [0u64; 256],
                |mut acc, chunk| {
                    for &x in chunk {
                        acc[(x & 0xFF) as usize] += 1;
                    }
                    acc
                },
                |mut a, b| {
                    for i in 0..256 {
                        a[i] += b[i];
                    }
                    a
                },
            );
        }
        let mut serial = [0u64; 256];
        for &x in &v {
            serial[(x & 0xFF) as usize] += 1;
        }
        assert_eq!(last, serial, "flat-fanout path output must match serial reference");
    }

    // Audit test infrastructure that was attempted here exposed
    // two real findings then deleted as unreliable methodology:
    //
    // 1. OOB calibration cycle reading is sensitive to whether
    //    the first sample catches worker-pool spawn-up. Histogram
    //    (trivial reducer) can read as heavy if its first
    //    calibration is the very first reduce_chunks call in the
    //    process. Subsequent runs of the same test see the
    //    cached-warm timing.
    //
    // 2. Debug-profile builds (cargo test default) measure
    //    significantly higher per-reduce cycles than release-
    //    profile builds (cargo bench default) for the same
    //    closure. A histogram-style 256-u64 element-wise add
    //    measures ~200 cycles in release (below the 30k-cycle
    //    threshold -> trivial) but ~30k+ cycles in debug (right
    //    at or above the threshold -> nondeterministic
    //    classification).
    //
    // The chunk-count investigation that produced these findings
    // moved to bench-driven validation (FLYNNEL_REDUCE_CHUNKS_CHUNKS=N
    // + multi-sample sweeps across the reduce_chunks call sites). The
    // [`ReduceChunksPath`] enum and [`last_reduce_chunks_path()`]
    // accessor are kept as infrastructure for any future audit
    // test that doesn't depend on calibration outcome (e.g., one
    // that directly drives the observer state via a test-only
    // setter).

    #[test]
    fn bench_audit_explicit_heavy_reduce_routes_to_bisect() {
        // EXPLICIT-COST audit: the reduce closure busy-spins for
        // ~50us, comfortably above the trivial-reduce ceiling
        // regardless of input shape, build profile (debug vs
        // release), or cold-vs-warm worker state. This is the only
        // routing-audit test that's reliable across all build/run
        // contexts. Trivial-reducer routing tests are unreliable
        // methodology here: the calibration cycle reading varies
        // with cold-warm pool state and with build profile, so no
        // such test exists in this module.
        //
        // Enable the flat-fanout gate (production keeps it off by
        // default since the bench audit found flat is structurally
        // slower for the three characterized reduce_chunks
        // consumers; the gate is for in-source experiments).
        // SAFETY: process-global env-var set; tests in this module
        // are expected to run with --test-threads=1 if they touch
        // shared process state.
        unsafe {
            std::env::set_var("FLYNNEL_ENABLE_FLAT_FANOUT", "1");
        }
        LAST_REDUCE_CHUNKS_PATH.with(|c| c.set(None));
        let n = 64 * 1024usize;
        let v: Vec<u32> = (0..n as u32).collect();
        let plan = JobPlan::new(6, n as u32).with_estimated_per_item_ns(1);
        let mut last_path = None;
        for _ in 0..8 {
            let _result = reduce_chunks(
                &plan, &v,
                || 0u64,
                |acc, chunk| acc + chunk.iter().map(|&x| x as u64).sum::<u64>(),
                |a, b| {
                    let t0 = read_tsc();
                    while read_tsc().wrapping_sub(t0) < 150_000 {
                        std::hint::spin_loop();
                    }
                    a.wrapping_add(b)
                },
            );
            last_path = last_reduce_chunks_path();
        }
        assert_eq!(
            last_path,
            Some(ReduceChunksPath::Bisect),
            "explicit-heavy reducer (>30k-cycle spin) must route to Bisect; got {last_path:?}"
        );
    }

    #[test]
    fn reduce_chunks_large_heavy_reduce_matches_serial() {
        // Same input-size reasoning as the trivial-reduce test.
        let n = 64 * 1024usize;
        let v: Vec<u32> = (0..n as u32).collect();
        let plan = JobPlan::new(6, n as u32).with_estimated_per_item_ns(1);
        let mut last_sum: u64 = 0;
        for _ in 0..3 {
            last_sum = reduce_chunks(
                &plan, &v,
                || 0u64,
                |acc, chunk| acc + chunk.iter().map(|&x| x as u64).sum::<u64>(),
                |a, b| {
                    // ~50 us simulated heavy reduce. Observer
                    // should classify > TRIVIAL threshold and
                    // keep call site on bisect path.
                    let t0 = read_tsc();
                    while read_tsc().wrapping_sub(t0) < 150_000 {
                        std::hint::spin_loop();
                    }
                    a.wrapping_add(b)
                },
            );
        }
        let serial: u64 = v.iter().map(|&x| x as u64).sum();
        assert_eq!(last_sum, serial, "bisect path output must match serial reference");
    }

    #[test]
    fn for_each_chunk_records_leaf_time_stats() {
        // 2way wire-in: `record_leaf` brackets every bisect leaf and
        // forwards the TSC delta into the variance counters. After a
        // for_each_chunk run with N > MIN_LEAF_ITEMS, the LEAF_COUNT
        // must be >= 1 (at least one leaf fires) and SUM_NS must be
        // strictly positive (every leaf body takes some non-zero
        // time). Run this AFTER a reset so the counts are this
        // call's contribution alone.
        use crate::sched::split_observer::{
            acquire_test_lock, reset_leaf_stats, snapshot_leaf_stats,
        };

        let _stats_lock = acquire_test_lock();
        reset_leaf_stats();
        let n = 5_000usize;
        let mut v: Vec<u32> = (0..n as u32).collect();
        let plan = JobPlan::new(6, n as u32);
        for_each_chunk(&plan, &mut v, |slice| {
            // Force a non-trivial body so the TSC delta is well
            // above the rdtsc-pair resolution (~20 cycles).
            let mut acc: u64 = 0;
            for &x in slice.iter() {
                acc = acc.wrapping_add((x as u64).wrapping_mul(0x9E3779B97F4A7C15));
            }
            std::hint::black_box(acc);
        });
        let stats = snapshot_leaf_stats();
        assert!(stats.count >= 1,
            "expected at least one leaf recorded, got {}", stats.count);
        assert!(stats.sum_ns > 0,
            "expected positive total TSC delta, got {}", stats.sum_ns);
        // Cleanup so this test's data doesn't pollute neighbours.
        reset_leaf_stats();
    }

    #[test]
    fn for_each_chunk_propagates_panic() {
        let mut v: Vec<u32> = (0..1000).collect();
        let plan = JobPlan::new(6, 1000);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for_each_chunk(&plan, &mut v, |slice| {
                if slice.first() == Some(&0) {
                    panic!("intentional panic from first chunk");
                }
                for x in slice {
                    *x += 1;
                }
            });
        }));
        assert!(r.is_err(), "panic in any chunk must propagate to caller");
    }

    #[test]
    fn par_map_in_place_touches_every_element() {
        let mut items: Vec<u64> = (0..256).map(|i| i as u64).collect();
        let plan = JobPlan::new(8, 256);
        par_map_in_place(&plan, &mut items, |b| *b = b.wrapping_mul(3));
        for (i, &b) in items.iter().enumerate() {
            assert_eq!(b, (i as u64).wrapping_mul(3));
        }
    }

    #[test]
    fn par_zip_apply_pairs_lhs_and_rhs_by_index() {
        let mut lhs: Vec<u64> = (0..1024).collect();
        let rhs: Vec<u64> = (0..1024).map(|i| i * 7).collect();
        let plan = JobPlan::new(8, 1024);
        par_zip_apply(&plan, &mut lhs, &rhs, |a, b| *a = a.wrapping_add(*b));
        for (i, &value) in lhs.iter().enumerate().take(1024) {
            let expected = (i as u64).wrapping_add((i as u64) * 7);
            assert_eq!(value, expected, "index {i}");
        }
    }

    #[test]
    #[should_panic(expected = "matching slice lengths")]
    fn par_zip_apply_panics_on_mismatched_lengths() {
        let mut lhs: Vec<u64> = vec![0; 10];
        let rhs: Vec<u64> = vec![0; 20];
        let plan = JobPlan::new(8, 10);
        par_zip_apply(&plan, &mut lhs, &rhs, |_a, _b| {});
    }

    #[test]
    fn par_zip_apply_empty_is_noop() {
        let mut lhs: Vec<u64> = Vec::new();
        let rhs: Vec<u64> = Vec::new();
        let plan = JobPlan::new(8, 0);
        par_zip_apply(&plan, &mut lhs, &rhs, |_a, _b| panic!("must not run"));
        assert!(lhs.is_empty());
    }

    #[test]
    fn for_each_indexed_visits_every_index_exactly_once() {
        let n = 10_000usize;
        let seen: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
        let plan = JobPlan::new(6, n as u32);
        for_each_indexed(&plan, n, 1, |i| {
            seen[i].fetch_add(1, Ordering::Relaxed);
        });
        for (i, s) in seen.iter().enumerate() {
            assert_eq!(s.load(Ordering::Relaxed), 1, "index {i} visited {} times", s.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn for_each_indexed_empty_is_noop() {
        let plan = JobPlan::new(6, 0);
        for_each_indexed(&plan, 0, 1, |_| panic!("must not run"));
    }

    #[test]
    fn for_each_chunk_ref_tiles_the_slice_once_at_the_requested_width() {
        let n = 4_099usize;
        let width = 64usize;
        let v: Vec<u32> = (0..n as u32).collect();
        let seen: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
        let widths = Arc::new(std::sync::Mutex::new(Vec::<(usize, usize)>::new()));
        let widths_clone = Arc::clone(&widths);
        let plan = JobPlan::new(6, n as u32);
        for_each_chunk_ref(&plan, &v, width, |start, chunk| {
            widths_clone.lock().unwrap().push((start, chunk.len()));
            for (k, &x) in chunk.iter().enumerate() {
                assert_eq!(x as usize, start + k, "chunk at {start} holds the wrong items");
                seen[start + k].fetch_add(1, Ordering::Relaxed);
            }
        });
        for (i, s) in seen.iter().enumerate() {
            assert_eq!(s.load(Ordering::Relaxed), 1, "item {i} covered {} times", s.load(Ordering::Relaxed));
        }
        let mut widths = widths.lock().unwrap().clone();
        widths.sort_unstable();
        assert_eq!(widths.len(), n.div_ceil(width), "one chunk per width-sized tile");
        for (k, &(start, len)) in widths.iter().enumerate() {
            assert_eq!(start, k * width, "chunk {k} starts at the wrong index");
            let expected = if k + 1 == widths.len() { n - k * width } else { width };
            assert_eq!(len, expected, "chunk {k} has the wrong width");
        }
    }

    #[test]
    fn for_each_chunk_ref_empty_is_noop() {
        let v: Vec<u32> = Vec::new();
        let plan = JobPlan::new(6, 0);
        for_each_chunk_ref(&plan, &v, 16, |_, _| panic!("must not run"));
    }
}

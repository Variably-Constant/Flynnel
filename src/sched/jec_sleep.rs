//! JEC (Jobs Event Counter) sleep protocol. Verbatim port of
//! `rayon-core-1.13.0::sleep::{counters,mod}`. The MIT copyright
//! notice carried by the upstream source is reproduced in
//! [`THIRD-PARTY-LICENSES.md`](../../../THIRD-PARTY-LICENSES.md)
//! at the repository root, per the terms of that license.
//!
//! The protocol tracks `awake_but_idle` and `sleeping` worker
//! counts separately so the producer can skip the unpark syscall
//! when enough workers are already spinning.
//!
//! # State machine
//!
//! Each worker iterates between four phases:
//!
//! 1. `ACTIVE`: running a job (not counted as inactive).
//! 2. `IDLE`: finished a job, spinning inside `no_work_found`, one
//!    `yield_now` a round, or one bounded monitor wait on the counters
//!    word where the spin monitor lever is on and the host has one;
//!    counted as `awake_but_idle`. After `ROUNDS_UNTIL_SLEEPY` rounds
//!    the worker transitions to:
//! 3. `SLEEPY`: announces itself by incrementing JEC (making it
//!    even); producers will see this and bump JEC back to odd if
//!    they post new work. Still counted as `awake_but_idle`.
//!    After `rounds_until_sleeping()` more yields the worker
//!    transitions to:
//! 4. `SLEEPING`: publishes that it is sleeping in its own atomic
//!    word, re-reads the shutdown flag, then parks; counted as both
//!    `inactive` AND `sleeping`. Awoken by `wake_specific_thread`,
//!    which claims that word (`SLEEPING` to `WAKING`), gives the
//!    sleeping count back, stores `AWAKE`, then unparks the thread.
//!    The worker stays parked until it reads `AWAKE`, so the count
//!    is back before it can take work and before a producer can
//!    count it as a sleeper still to wake. Exactly one party wins
//!    the claim, so exactly one decrements.
//!
//! Producers (`new_internal_jobs`):
//!   - Increment JEC if it is sleepy (signals sleepy workers to
//!     re-search before they sleep).
//!   - If queue was non-empty, wake `min(num_jobs, num_sleepers)`.
//!   - If queue was empty, wake `max(num_jobs - awake_but_idle, 0)`
//!     capped at num_sleepers.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::thread;

// ===========================================================================
// AtomicCounters - packed (sleeping | inactive | JEC) in one AtomicUsize
// ===========================================================================

#[cfg(target_pointer_width = "64")]
const THREADS_BITS: usize = 16;

#[cfg(target_pointer_width = "32")]
const THREADS_BITS: usize = 8;

#[allow(clippy::erasing_op)]
const SLEEPING_SHIFT: usize = 0 * THREADS_BITS;
#[allow(clippy::identity_op)]
const INACTIVE_SHIFT: usize = 1 * THREADS_BITS;
const JEC_SHIFT: usize = 2 * THREADS_BITS;

/// Maximum thread count the counter word can hold.
pub(crate) const THREADS_MAX: usize = (1 << THREADS_BITS) - 1;

const ONE_SLEEPING: usize = 1;
const ONE_INACTIVE: usize = 1 << INACTIVE_SHIFT;
const ONE_JEC: usize = 1 << JEC_SHIFT;

/// Process-shared atomic counter pack.
pub(crate) struct AtomicCounters {
    value: AtomicUsize,
}

/// Snapshot of the counter word for inspection without atomic
/// reads.
#[derive(Copy, Clone)]
pub(crate) struct Counters {
    word: usize,
}

/// The JEC value extracted from a counter snapshot. Even = sleepy
/// (the last increment was by a worker becoming sleepy). Odd =
/// active (the last increment was by a producer posting work).
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub(crate) struct JobsEventCounter(usize);

impl JobsEventCounter {
    pub(crate) const DUMMY: JobsEventCounter = JobsEventCounter(usize::MAX);

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn as_usize(self) -> usize {
        self.0
    }

    #[inline]
    pub(crate) fn is_sleepy(self) -> bool {
        (self.0 & 1) == 0
    }

    #[inline]
    pub(crate) fn is_active(self) -> bool {
        !self.is_sleepy()
    }
}

#[inline]
fn select_thread(word: usize, shift: usize) -> usize {
    (word >> shift) & THREADS_MAX
}

#[inline]
fn select_jec(word: usize) -> usize {
    word >> JEC_SHIFT
}

impl AtomicCounters {
    pub(crate) const fn new() -> Self {
        Self { value: AtomicUsize::new(0) }
    }

    #[inline]
    pub(crate) fn load(&self, ordering: Ordering) -> Counters {
        Counters { word: self.value.load(ordering) }
    }

    /// The packed word as it is, for a reader that only asks whether
    /// it has moved.
    #[inline]
    fn raw(&self, ordering: Ordering) -> usize {
        self.value.load(ordering)
    }

    /// The address of the word, for a monitor wait to watch the cache
    /// line it sits in. Every store a producer makes to announce work
    /// lands on this line.
    #[inline]
    fn line(&self) -> *const u8 {
        (&raw const self.value).cast::<u8>()
    }

    #[inline]
    fn try_exchange(&self, old: Counters, new: Counters, ordering: Ordering) -> bool {
        self.value
            .compare_exchange(old.word, new.word, ordering, Ordering::Relaxed)
            .is_ok()
    }

    /// Add one inactive thread. Invoked when a worker enters its
    /// idle loop looking for work.
    #[inline]
    pub(crate) fn add_inactive_thread(&self) {
        self.value.fetch_add(ONE_INACTIVE, Ordering::SeqCst);
    }

    /// Sub one inactive thread. Invoked when a worker finds work
    /// (transitions from idle to active). Returns the
    /// recommended number of sleepers to wake (up to 2 per
    /// rayon's heuristic).
    #[inline]
    pub(crate) fn sub_inactive_thread(&self) -> usize {
        let old = Counters {
            word: self.value.fetch_sub(ONE_INACTIVE, Ordering::SeqCst),
        };
        debug_assert!(old.inactive_threads() > 0);
        debug_assert!(old.sleeping_threads() <= old.inactive_threads());
        let sleepers = old.sleeping_threads();
        Ord::min(sleepers, 2)
    }

    /// Sub one sleeping thread. The caller must know that at least
    /// one sleeping thread exists: a waker that has claimed one, or a
    /// sleeper giving back its own before it parks.
    #[inline]
    pub(crate) fn sub_sleeping_thread(&self) {
        let old = Counters {
            word: self.value.fetch_sub(ONE_SLEEPING, Ordering::SeqCst),
        };
        debug_assert!(old.sleeping_threads() > 0);
    }

    /// Transition this worker from idle to sleeping. Will succeed
    /// only if no other counter change has happened since
    /// `old_value` was loaded.
    #[inline]
    pub(crate) fn try_add_sleeping_thread(&self, old: Counters) -> bool {
        debug_assert!(old.inactive_threads() > 0);
        debug_assert!(old.sleeping_threads() < THREADS_MAX);
        let mut new = old;
        new.word += ONE_SLEEPING;
        self.try_exchange(old, new, Ordering::SeqCst)
    }

    /// Increment the JEC if `pred` on the current value returns
    /// true. Used to flip JEC parity (sleepy <-> active). Returns
    /// the final snapshot for which `pred` is false.
    pub(crate) fn increment_jobs_event_counter_if(
        &self,
        pred: impl Fn(JobsEventCounter) -> bool,
    ) -> Counters {
        loop {
            let old = self.load(Ordering::SeqCst);
            if pred(old.jobs_counter()) {
                let new = Counters {
                    word: old.word.wrapping_add(ONE_JEC),
                };
                if self.try_exchange(old, new, Ordering::SeqCst) {
                    return new;
                }
            } else {
                return old;
            }
        }
    }
}

impl Counters {
    #[inline]
    pub(crate) fn jobs_counter(self) -> JobsEventCounter {
        JobsEventCounter(select_jec(self.word))
    }

    #[inline]
    pub(crate) fn inactive_threads(self) -> usize {
        select_thread(self.word, INACTIVE_SHIFT)
    }

    #[inline]
    pub(crate) fn sleeping_threads(self) -> usize {
        select_thread(self.word, SLEEPING_SHIFT)
    }

    #[inline]
    pub(crate) fn awake_but_idle_threads(self) -> usize {
        debug_assert!(self.sleeping_threads() <= self.inactive_threads());
        self.inactive_threads() - self.sleeping_threads()
    }
}

impl std::fmt::Debug for Counters {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("Counters")
            .field("word", &format!("{:016x}", self.word))
            .field("jobs", &self.jobs_counter().0)
            .field("inactive", &self.inactive_threads())
            .field("sleeping", &self.sleeping_threads())
            .finish()
    }
}

// ===========================================================================
// Sleep state machine
// ===========================================================================

/// Yield rounds before announcing sleepy intent.
const ROUNDS_UNTIL_SLEEPY: u32 = 32;

/// Default spin-window rounds added on top of `ROUNDS_UNTIL_SLEEPY`
/// before a sleepy worker parks. 500 rounds is
/// approximately a 500us spin window, sized to span both typical
/// inter-dispatch gaps (10-50us) AND the longer between-dispatch
/// pauses that smaller pools see on dispatches that complete in
/// hundreds of microseconds. The producer-side
/// `new_internal_jobs` can skip the unpark syscall when the next
/// dispatch lands inside that window.
///
/// Tuned by a `[100..800]` sweep across three hosts (Heavy/100k
/// on flynnel_default vs rayon_par_iter_mut):
///
/// | Host | Pool | 200 rounds | 500 rounds (default) | Delta |
/// |---|---|---|---|---|
/// | Zen+ R7 2700 | 8p / 16t | 5.66 ms | 5.66 ms | tied (noise) |
/// | Intel Xeon Cascade Lake | 6p / 12t | 6.94 ms | 6.34 ms | -9% |
/// | AMD EPYC Genoa 9B14 | 22p / 44t | 1.70 ms | 1.51 ms | -11% |
///
/// All three hosts pick 500 as best-or-tied; no host regresses
/// at 500 vs 200. The single global default holds across pool
/// sizes from 12 to 44 logical threads.
///
/// Override at process startup by setting the
/// `FLYNNEL_SPIN_WINDOW_ROUNDS` env var; re-tune a new host
/// class by sweeping that variable over `[100..800]` on a
/// representative workload.
const DEFAULT_SPIN_WINDOW_ROUNDS: u32 = 500;

/// Floor the adaptive controller will shrink the spin window to. A
/// bursty-idle workload parks after roughly this many yields (~8us)
/// instead of burning the full default window, which is the CPU
/// analog of quiescing the GPU poller.
const FLOOR_SPIN_WINDOW_ROUNDS: u32 = 8;

/// Consecutive non-parking returns from [`Sleep::sleep`] tolerated
/// before a worker waits on a timer instead of re-running the spin
/// window.
///
/// Four rather than one because all three of the single-miss cases
/// are legitimate and self-clearing: a producer posting work as the
/// worker goes sleepy, a job landing in the injector during the
/// same window, and a peer draining the deque first. Each resolves
/// on the next search. Four in a row does not resolve, and by then
/// the worker has spent four full spin windows finding nothing.
const SLEEPLESS_BEFORE_BACKOFF: u32 = 4;

/// First wait once a worker is backing off. A quarter of a
/// millisecond is longer than any search round and short enough that
/// a state clearing immediately costs no throughput worth measuring.
const SLEEPLESS_BACKOFF_FLOOR: std::time::Duration =
    std::time::Duration::from_micros(250);

/// Ceiling on the doubling. At 32ms a stuck worker still searches
/// about thirty times a second - fast enough that a pool recovering
/// on its own is back inside a frame - while costing roughly one
/// part in a thousand of a core instead of all of it.
const SLEEPLESS_BACKOFF_CAP: std::time::Duration =
    std::time::Duration::from_millis(32);

/// Times a worker waited on the timer rather than re-running the
/// spin window. Zero on a healthy pool; a climbing value is the
/// signal that workers are being denied a park.
static SLEEPLESS_BACKOFFS: AtomicU64 = AtomicU64::new(0);

/// How long to wait after `n` consecutive non-parking returns.
/// Doubles from the floor and clamps at the cap.
fn sleepless_backoff(n: u32) -> std::time::Duration {
    let steps = n.saturating_sub(SLEEPLESS_BEFORE_BACKOFF).min(16);
    SLEEPLESS_BACKOFF_FLOOR
        .saturating_mul(1u32 << steps)
        .min(SLEEPLESS_BACKOFF_CAP)
}

/// Times a worker backed off instead of spinning, since process
/// start. Stays at zero unless workers are being denied a park.
pub fn total_sleepless_backoffs() -> u64 {
    SLEEPLESS_BACKOFFS.load(Ordering::Relaxed)
}

/// Idle rounds spent in a monitor wait since process start. Zero
/// unless [`crate::sched::levers::spin_monitor`] is on and the host's
/// monitor holds; a harness reads it to tell an arm that engaged from
/// one that did not.
pub fn total_monitor_rounds() -> u64 {
    TOTAL_MONITOR_ROUNDS.load(Ordering::Relaxed)
}

/// TSC cycles one monitor-wait round lasts: the host's published
/// dispatch cost, so a round is one dispatch opportunity and the spin
/// window's round count keeps its meaning. Zero while no profile is
/// published, which the caller takes as a round to yield.
#[inline]
fn spin_round_cycles() -> u64 {
    round_cycles(
        crate::sched::par_iter::installed_dispatch_cost_ns(),
        crate::sched::par_iter::tsc_per_ns_16(),
    )
}

/// `dispatch_cost_ns` in TSC cycles at `per_ns_16` sixteenths of a
/// cycle per nanosecond.
#[inline]
fn round_cycles(dispatch_cost_ns: u64, per_ns_16: u64) -> u64 {
    dispatch_cost_ns.saturating_mul(per_ns_16) / 16
}

/// Effective spin-window rounds (on top of [`ROUNDS_UNTIL_SLEEPY`]),
/// adjusted at runtime by the adaptive controller. Starts at the
/// tuned default.
static SPIN_WINDOW: AtomicU32 = AtomicU32::new(DEFAULT_SPIN_WINDOW_ROUNDS);
/// Adaptation is OFF by default: the default window (500) is tuned to
/// win on throughput across three host classes, so the default
/// behavior stays exactly that, with zero regression risk. A
/// bursty-idle workload opts in via [`set_spin_window`] (explicit
/// short window) or [`set_spin_adaptive`] / `FLYNNEL_ADAPTIVE_SPIN=1`
/// (let the controller shrink it), the same opt-in model the GPU
/// poller's pause lever uses.
static ADAPTIVE: AtomicBool = AtomicBool::new(false);
/// Controller evidence since the last adjust: workers that parked
/// (the spin was wasted - work did not arrive in the window) versus
/// workers RESCUED mid-spin (the spin paid off - it avoided a
/// park/unpark syscall pair).
static PARK_EVENTS: AtomicU32 = AtomicU32::new(0);
static RESCUE_EVENTS: AtomicU32 = AtomicU32::new(0);
/// Total idle rounds, exposed for observability. With the spin monitor
/// off every one of them is a `yield_now`, the quantity a flamegraph
/// attributes to `sched_yield`; with it on, [`TOTAL_MONITOR_ROUNDS`] of
/// them were monitor waits instead.
static TOTAL_YIELDS: AtomicU64 = AtomicU64::new(0);
/// Idle rounds spent in a bounded monitor wait rather than a yield.
/// Zero unless [`crate::sched::levers::spin_monitor`] is on and the
/// host has a monitor that holds, which is what a harness reads to
/// tell an arm that engaged from one that did not.
static TOTAL_MONITOR_ROUNDS: AtomicU64 = AtomicU64::new(0);
/// Times [`maybe_adapt`] passed its event gate and reached a decision.
/// Counted because the window alone cannot report it: a rescue-dominated
/// workload grows and is clamped to the default it started at.
/// Monotonic for the process; [`reset_spin_stats`] leaves it alone, so a
/// reader can treat a rise as evidence without holding the reset.
static ADAPT_DECISIONS: AtomicU64 = AtomicU64::new(0);

/// Read the env once: a fixed `FLYNNEL_SPIN_WINDOW_ROUNDS` pins the
/// window (adaptation off); `FLYNNEL_ADAPTIVE_SPIN=0` pins the
/// default. Otherwise the controller adapts from the default.
fn spin_init() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        if let Some(v) = std::env::var("FLYNNEL_SPIN_WINDOW_ROUNDS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
        {
            SPIN_WINDOW.store(v, Ordering::Relaxed);
            ADAPTIVE.store(false, Ordering::Relaxed);
        }
        if std::env::var("FLYNNEL_ADAPTIVE_SPIN").as_deref() == Ok("1") {
            ADAPTIVE.store(true, Ordering::Relaxed);
        }
    });
}

/// Yield rounds before a sleepy worker parks. Reads the
/// runtime-adjustable [`SPIN_WINDOW`] so the adaptive controller (or
/// [`set_spin_window`]) can shorten it for a bursty-idle workload.
#[inline]
fn rounds_until_sleeping() -> u32 {
    spin_init();
    ROUNDS_UNTIL_SLEEPY + SPIN_WINDOW.load(Ordering::Relaxed)
}

/// The controller: after enough evidence, nudge the spin window. When
/// parks dominate (a bursty workload keeps missing the window), shrink
/// toward the floor so workers stop burning CPU on yields. When
/// rescues dominate (a throughput workload keeps landing work inside
/// the window), grow back toward the tuned default. Bounded by the
/// default, so a throughput workload never regresses and a bursty one
/// reclaims the idle spin. Called only when a worker is about to park,
/// off the hot path.
/// Parks plus rescues a controller needs before it will move the
/// window. Below it the sample is a moment of a workload rather than
/// its shape.
const EVIDENCE_FLOOR: u32 = 256;

/// Whether a controller in this state has enough to decide on.
///
/// Split out from the wiring so it can be checked as arithmetic. The
/// state it reads is process-global and every park writes to it, so a
/// test driving those statics has to exclude every other test that
/// dispatches; a test calling this does not.
#[inline]
fn should_adapt(adaptive: bool, park: u32, rescue: u32) -> bool {
    adaptive && park.saturating_add(rescue) >= EVIDENCE_FLOOR
}

/// The window a controller moves to, given the one it holds and the
/// evidence it has.
///
/// Parks dominating means the spin ran out before work arrived, so the
/// window halves toward the floor. Rescues dominating means work
/// landed inside the window and the spin saved a park and unpark pair,
/// so it grows by a quarter, clamped to the tuned default.
#[inline]
fn adapted_window(cur: u32, park: u32, rescue: u32) -> u32 {
    if park > rescue.saturating_mul(3) {
        (cur / 2).max(FLOOR_SPIN_WINDOW_ROUNDS)
    } else if rescue > park {
        (cur + cur / 4 + 1).min(DEFAULT_SPIN_WINDOW_ROUNDS)
    } else {
        cur
    }
}

fn maybe_adapt() {
    let park = PARK_EVENTS.load(Ordering::Relaxed);
    let rescue = RESCUE_EVENTS.load(Ordering::Relaxed);
    if !should_adapt(ADAPTIVE.load(Ordering::Relaxed), park, rescue) {
        return;
    }
    ADAPT_DECISIONS.fetch_add(1, Ordering::Relaxed);
    let cur = SPIN_WINDOW.load(Ordering::Relaxed);
    SPIN_WINDOW.store(adapted_window(cur, park, rescue), Ordering::Relaxed);
    PARK_EVENTS.store(0, Ordering::Relaxed);
    RESCUE_EVENTS.store(0, Ordering::Relaxed);
}

/// Current effective spin window (rounds on top of the sleepy
/// threshold). Shrinks toward the floor under a bursty-idle workload.
pub fn spin_window() -> u32 {
    SPIN_WINDOW.load(Ordering::Relaxed)
}

/// Total idle-yield rounds across all workers since process start (or
/// the last [`reset_spin_stats`]). This is the CPU the flamegraph
/// charges to `sched_yield`; a shorter window drops it.
pub fn total_idle_yields() -> u64 {
    TOTAL_YIELDS.load(Ordering::Relaxed)
}

/// Times the adaptive controller reached a decision, counted from
/// process start and never reset. Zero while [`spin_adaptive`] is true
/// means the workload never parked often enough to gather the evidence;
/// that is distinct from a controller that decided and left
/// [`spin_window`] where it found it.
pub fn spin_adapt_decisions() -> u64 {
    ADAPT_DECISIONS.load(Ordering::Relaxed)
}

/// Reset the yield and controller-evidence counters (for measuring a
/// specific phase).
pub fn reset_spin_stats() {
    TOTAL_YIELDS.store(0, Ordering::Relaxed);
    PARK_EVENTS.store(0, Ordering::Relaxed);
    RESCUE_EVENTS.store(0, Ordering::Relaxed);
}

/// Force the spin window to `rounds` and stop the adaptive
/// controller. The explicit lever for a workload known to be
/// bursty-idle and latency-insensitive between bursts: set a small
/// window so idle workers park promptly instead of spinning. Re-enable
/// auto-tuning with [`set_spin_adaptive`].
pub fn set_spin_window(rounds: u32) {
    spin_init();
    SPIN_WINDOW.store(rounds, Ordering::Relaxed);
    ADAPTIVE.store(false, Ordering::Relaxed);
}

/// Turn the adaptive controller on or off. When turned back on it
/// resumes from the current window.
pub fn set_spin_adaptive(on: bool) {
    ADAPTIVE.store(on, Ordering::Relaxed);
}

/// Whether the adaptive controller is running, after the environment
/// has been read.
///
/// A window still at its default says either that the controller is off
/// or that it is on and the evidence keeps it there, and a harness
/// reporting only the window cannot tell those apart.
pub fn spin_adaptive() -> bool {
    spin_init();
    ADAPTIVE.load(Ordering::Relaxed)
}

/// Per-worker sleep state held inside the global `Sleep` struct.
///
/// `#[repr(align(128))]` so two adjacent workers in the Vec never
/// share a 128-byte prefetched cache-line pair. Without this, worker
/// 0's write to its own state invalidates worker 1's cached copy.
/// Cilk's `CILK_CACHE_LINE = 128` is the same rationale.
#[repr(align(128))]
struct WorkerSleepState {
    /// [`SLEEPING`] from just before this worker parks until a waker
    /// claims it, [`WAKING`] while that waker gives the sleeping count
    /// back, [`AWAKE`] otherwise. Only [`AWAKE`] releases the worker
    /// from its park loop, so the count is back before it runs.
    state: AtomicU32,
    /// This worker's own thread handle, stored the first time it
    /// sleeps and reused after.
    ///
    /// `thread::park` and `unpark` carry a permit, so an unpark that
    /// lands before the park is remembered and the next park returns
    /// at once, so a wake that arrives between the publish below and
    /// the park is kept rather than dropped.
    handle: OnceLock<thread::Thread>,
}

/// [`WorkerSleepState::state`] for a worker that is running.
const AWAKE: u32 = 0;
/// [`WorkerSleepState::state`] for a worker that is parked, or has
/// committed to parking and is making its last checks.
const SLEEPING: u32 = 1;
/// [`WorkerSleepState::state`] for a parked worker a waker has
/// claimed and not yet released: the sleeping count is being given
/// back on the waker's thread, and the worker keeps parking until
/// it reads [`AWAKE`].
const WAKING: u32 = 2;

/// Per-worker idle bookkeeping carried across calls to
/// `no_work_found`. Initialized once when a worker enters its idle
/// loop, dropped when work is found.
pub(crate) struct IdleState {
    /// Worker index this idle state belongs to.
    pub worker_index: usize,
    /// Yield rounds elapsed since the worker entered its idle loop.
    pub rounds: u32,
    /// JEC snapshot taken when the worker entered sleepy state;
    /// used to detect a producer JEC bump that should rescue us.
    pub jobs_counter: JobsEventCounter,
    /// Set once this worker has actually parked in this
    /// idle episode, so a later find is not miscounted as a spin
    /// rescue (the spin did not save this worker - it parked).
    pub parked: bool,
    /// Consecutive [`Sleep::sleep`] calls that returned without
    /// parking, reset by any park. Both non-parking exits leave the
    /// worker searching again and neither yields, so a condition that
    /// persists turns the idle loop into a spin that holds a core.
    /// This counts them so the loop can back off.
    pub(crate) sleepless: u32,
}

impl IdleState {
    pub(crate) fn new(worker_index: usize) -> Self {
        Self {
            worker_index,
            rounds: 0,
            jobs_counter: JobsEventCounter::DUMMY,
            parked: false,
            sleepless: 0,
        }
    }

    fn wake_fully(&mut self) {
        self.rounds = 0;
        self.jobs_counter = JobsEventCounter::DUMMY;
    }

    fn wake_partly(&mut self) {
        self.rounds = ROUNDS_UNTIL_SLEEPY;
        self.jobs_counter = JobsEventCounter::DUMMY;
    }
}

/// Snapshot returned by [`Sleep::debug_state`].
pub(crate) struct SleepDebug {
    /// Workers registered as sleeping in the counter word.
    pub(crate) sleeping: usize,
    /// Workers in the idle loop (sleeping ones included).
    pub(crate) inactive: usize,
    /// Raw JEC value; even = sleepy, odd = active.
    pub(crate) jec: usize,
    /// Whether each worker is parked. Plain `bool` rather than an
    /// `Option`: the absent case meant its mutex was held at the
    /// moment of the read, and there is no mutex to hold now.
    pub(crate) blocked: Vec<bool>,
}

/// Process-global sleep coordinator. One instance per arena;
/// referenced by every worker and every producer.
pub(crate) struct Sleep {
    counters: AtomicCounters,
    worker_states: Vec<WorkerSleepState>,
    /// One bit per external slot, set while a caller holds the
    /// slot. A thief probes exactly the claimed slots' deques each
    /// round and draws its random victims from the workers alone,
    /// so an external job is found within a round instead of at
    /// the rate the slot count dilutes a random pick to.
    claimed_slots: AtomicU64,
    /// Set once by [`Sleep::wake_all_for_shutdown`], before it wakes
    /// anybody, and re-read by [`Sleep::sleep`] after that worker has
    /// published `SLEEPING`.
    ///
    /// The flag is what makes shutdown durable. Waking is not: a wake
    /// reaches only the workers that are parked at the moment it
    /// runs, and a worker still walking its tiers is not one of them,
    /// so without something left behind it parks afterwards and
    /// nothing ever wakes it again, which is `LocalArena::drop`
    /// joining a worker that will not return.
    ///
    /// The publish-then-read order is the load-bearing part rather
    /// than the flag itself. With every operation of the pair SeqCst,
    /// either the sweep's claim sees `SLEEPING` or the worker's read
    /// sees the flag.
    shutdown: AtomicBool,
}

impl Sleep {
    pub(crate) fn new(num_workers: usize) -> Self {
        assert!(num_workers <= THREADS_MAX, "too many workers");
        let mut states = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            states.push(WorkerSleepState {
                state: AtomicU32::new(AWAKE),
                handle: OnceLock::new(),
            });
        }
        Self {
            counters: AtomicCounters::new(),
            worker_states: states,
            claimed_slots: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    /// The number of pool workers, which is also the index of the
    /// first external slot in the arena's stealer table.
    #[inline]
    pub(crate) fn worker_count(&self) -> usize {
        self.worker_states.len()
    }

    /// Mark external slot `slot_id` (0-based within the slot pool)
    /// as held by a caller.
    #[inline]
    pub(crate) fn note_slot_claimed(&self, slot_id: usize) {
        self.claimed_slots.fetch_or(1u64 << slot_id, Ordering::Release);
    }

    /// Mark external slot `slot_id` as free again.
    #[inline]
    pub(crate) fn note_slot_released(&self, slot_id: usize) {
        self.claimed_slots.fetch_and(!(1u64 << slot_id), Ordering::Release);
    }

    /// The set of held external slots, one bit per slot id.
    #[inline]
    pub(crate) fn claimed_slots(&self) -> u64 {
        self.claimed_slots.load(Ordering::Acquire)
    }

    #[allow(dead_code)]
    pub(crate) fn num_workers(&self) -> usize {
        self.worker_states.len()
    }

    /// Diagnostic view of the counter word and each worker's
    /// parked flag, read straight out of each worker's atomic word
    /// at the instant of the read (worker mid-transition).
    pub(crate) fn debug_state(&self) -> SleepDebug {
        let c = self.counters.load(Ordering::SeqCst);
        let blocked = self
            .worker_states
            .iter()
            .map(|s| s.state.load(Ordering::Relaxed) != AWAKE)
            .collect();
        SleepDebug {
            sleeping: c.sleeping_threads(),
            inactive: c.inactive_threads(),
            jec: c.jobs_counter().as_usize(),
            blocked,
        }
    }

    /// Worker-side: called when a worker enters its idle loop for
    /// the first time after finishing work. Increments the
    /// inactive counter; balanced by `work_found`.
    #[inline]
    pub(crate) fn start_looking(&self, worker_index: usize) -> IdleState {
        self.counters.add_inactive_thread();
        IdleState::new(worker_index)
    }

    /// Worker-side: called when an idle worker found a job. Wakes
    /// up to 2 sleeping workers (rayon's heuristic) since the
    /// JEC churn / new work may have changed the equilibrium.
    #[inline]
    pub(crate) fn work_found(&self, idle: &IdleState) {
        // Rescued mid-spin (sleepy but never parked) means the spin
        // paid off - it avoided a park/unpark syscall. That is the
        // evidence the controller uses to keep the window long.
        if idle.rounds > ROUNDS_UNTIL_SLEEPY && !idle.parked {
            RESCUE_EVENTS.fetch_add(1, Ordering::Relaxed);
        }
        let wake_count = self.counters.sub_inactive_thread();
        if wake_count > 0 {
            self.wake_any_threads(wake_count as u32);
        }
    }

    /// One idle round: a bounded monitor wait on the counters word where
    /// [`crate::sched::levers::spin_monitor`] is on and the host has a
    /// monitor that holds, a `yield_now` otherwise.
    ///
    /// The wait watches the line a producer stores to when it posts
    /// work, so the round ends when work arrives rather than when the
    /// scheduler next picks this thread. It also ends on any other
    /// store to that line, which costs one more search round. A round
    /// lasts the host's published dispatch cost, so the spin window's
    /// round count keeps the scale it was tuned at; before a profile is
    /// published the round yields as before. The first monitor round of
    /// an episode is traced, so a traced dispatch counts the episodes
    /// that took the monitor.
    #[inline]
    fn idle_round(&self, idle: &IdleState) {
        TOTAL_YIELDS.fetch_add(1, Ordering::Relaxed);
        if crate::sched::levers::spin_monitor() {
            let budget = spin_round_cycles();
            if budget > 0 {
                let seen = self.counters.raw(Ordering::Relaxed);
                // SAFETY: the counters word lives as long as this
                // coordinator, which outlives every worker's idle loop.
                let waited = unsafe {
                    crate::sched::sleep::monitor_wait_once(self.counters.line(), budget, || {
                        self.counters.raw(Ordering::Relaxed) != seen
                    })
                };
                if waited {
                    TOTAL_MONITOR_ROUNDS.fetch_add(1, Ordering::Relaxed);
                    if idle.rounds == 0 && crate::sched::trace::is_enabled() {
                        crate::sched::trace::emit(
                            crate::sched::trace::TraceEvent::SpinMonitor,
                            if crate::cpu_info::has_waitpkg() { 1 } else { 2 },
                        );
                    }
                    return;
                }
            }
        }
        thread::yield_now();
    }

    /// Worker-side: called when one search round produced no
    /// work. Advances the idle state through yield -> sleepy ->
    /// sleeping. `has_injected_jobs` is called inside the sleep
    /// transition to recover from the race where a job was
    /// injected between us going sleepy and locking the mutex.
    pub(crate) fn no_work_found(
        &self,
        idle: &mut IdleState,
        has_reachable_work: impl Fn() -> bool,
    ) {
        if idle.rounds < ROUNDS_UNTIL_SLEEPY {
            self.idle_round(idle);
            idle.rounds += 1;
        } else if idle.rounds == ROUNDS_UNTIL_SLEEPY {
            idle.jobs_counter = self.announce_sleepy();
            idle.rounds += 1;
            self.idle_round(idle);
        } else if idle.rounds < rounds_until_sleeping() {
            idle.rounds += 1;
            self.idle_round(idle);
        } else {
            self.sleep(idle, has_reachable_work);
            // `sleep` returns without parking two ways: the JEC moved,
            // or the injector held a job. Both send the worker back to
            // searching and neither yields, so while the condition
            // holds the worker re-runs the spin window finding nothing
            // and keeps a core at one hundred percent. Past a few
            // consecutive misses the spin has stopped paying for
            // itself, so the worker waits on a timer instead. It still
            // runs a full search each time round, so work that becomes
            // takeable is picked up within one interval.
            if idle.sleepless >= SLEEPLESS_BEFORE_BACKOFF {
                SLEEPLESS_BACKOFFS.fetch_add(1, Ordering::Relaxed);
                thread::sleep(sleepless_backoff(idle.sleepless));
            }
        }
    }

    /// Bump JEC if currently active (odd), making it sleepy
    /// (even). The producer reads this on next `new_internal_jobs`
    /// and knows there is at least one sleepy worker that should
    /// be notified before it sleeps.
    fn announce_sleepy(&self) -> JobsEventCounter {
        self.counters
            .increment_jobs_event_counter_if(JobsEventCounter::is_active)
            .jobs_counter()
    }

    /// Worker-side: actually park after the sleepy phase.
    ///
    /// Returns without parking three ways: the JEC changed, so a
    /// producer posted work while this worker was sleepy;
    /// `has_reachable_work` answered true before the sleeping state
    /// was published; or it answered true after, in which case this
    /// worker takes itself back out of the sleeping count. The third
    /// is the one a waker cannot cover, because between the counter
    /// CAS and the state store a worker is counted as sleeping and is
    /// not yet claimable.
    ///
    /// `has_reachable_work` is asked twice and so is `Fn`, not
    /// `FnOnce`. It must report every queue this worker alone can
    /// drain, not merely the shared ones: a queue with other
    /// consumers survives a missed wake because another consumer
    /// takes the job, and a single-consumer queue does not.
    fn sleep(
        &self,
        idle: &mut IdleState,
        has_reachable_work: impl Fn() -> bool,
    ) {
        let state = &self.worker_states[idle.worker_index];
        debug_assert_eq!(state.state.load(Ordering::Relaxed), AWAKE);

        // Cheap exit before touching the counters. Not the load-bearing
        // check: that one is below, after this worker has published
        // that it is sleeping.
        if self.shutdown.load(Ordering::Acquire) {
            idle.wake_partly();
            return;
        }

        loop {
            let counters = self.counters.load(Ordering::SeqCst);
            debug_assert!(idle.jobs_counter.is_sleepy());
            if counters.jobs_counter() != idle.jobs_counter {
                // JEC changed: work posted since we went sleepy.
                // Bail out and resume searching.
                idle.sleepless = idle.sleepless.saturating_add(1);
                idle.wake_partly();
                return;
            }
            if self.counters.try_add_sleeping_thread(counters) {
                break;
            }
        }

        // Registered as sleeping. One last check for reachable
        // work (closes the deadlock race where work arrived
        // while we were sleepy and our JEC bump rolled over).
        std::sync::atomic::fence(Ordering::SeqCst);
        if has_reachable_work() {
            self.counters.sub_sleeping_thread();
            idle.sleepless = idle.sleepless.saturating_add(1);
        } else {
            // Publish first, then re-read. A waker arriving from here
            // on sees SLEEPING and claims it; one that arrived earlier
            // stored the flag this reads next. Both orders are covered
            // and neither needs a lock: SeqCst on the store here and
            // on the claim in wake_specific_thread puts all four
            // operations in one total order.
            //
            // Jobs need it too, and for the same reason. The counter
            // word orders only what arrives before
            // try_add_sleeping_thread, which has already succeeded
            // here, and the producer's bump is conditional on some
            // worker being sleepy, so a push landing in this window
            // moves nothing this worker reads. The producer does call
            // wake_any_threads, but wake_specific_thread claims a
            // worker only once its state reads SLEEPING, and in this
            // window it does not: the worker counts toward
            // sleeping_threads and is unclaimable at the same time.
            //
            // For the injector that is survivable, since any worker
            // drains it and another one takes the job. A mailbox is
            // popped only by its owner, so there is no other worker,
            // and the job waits for a wake that has already been
            // issued and landed nowhere. So the worker rescues itself
            // here rather than relying on a waker reaching it.
            state.handle.get_or_init(thread::current);
            state.state.store(SLEEPING, Ordering::SeqCst);
            if (self.shutdown.load(Ordering::SeqCst) || has_reachable_work())
                && state
                    .state
                    .compare_exchange(SLEEPING, AWAKE, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                // Shutdown or work landed, and no waker has claimed
                // this worker, so the sleeping count is this thread's
                // to give back. A claimed worker falls through to the
                // loop below and is released by its waker, which gives
                // the count back itself.
                self.counters.sub_sleeping_thread();
                // A rescue is a third way out of here without parking,
                // so it feeds the same counter the other two do. Work
                // this worker alone can drain is work it will get, but
                // work in the injector may be gone by the time it
                // looks; a rescue that keeps finding nothing would
                // re-run the spin window at one hundred percent of a
                // core, which is what the backoff exists to stop.
                idle.sleepless = idle.sleepless.saturating_add(1);
                idle.wake_fully();
                return;
            }

            // Committing to a park: neither the spin nor the recheck
            // above rescued this worker. Counted here rather than at
            // the decision to sleep, so a worker that rescued itself
            // is not counted as having parked and PARK_EVENTS means
            // what its name says. Feed the controller before blocking.
            idle.parked = true;
            idle.sleepless = 0;
            PARK_EVENTS.fetch_add(1, Ordering::Relaxed);
            crate::sched::trace::emit(
                crate::sched::trace::TraceEvent::PoolPark,
                idle.worker_index as u32,
            );
            maybe_adapt();

            // park returns on a permit, on an unpark, and spuriously,
            // and a claimed worker is still owed its release, so only
            // AWAKE ends the loop. The waker gave the sleeping count
            // back before it stored AWAKE; nothing is owed here.
            //
            // The park is the kernel's on every host. A monitor wait
            // here would be a running thread for as long as the worker
            // idled, holding its hardware thread from everything else
            // on the box; the monitor belongs in the bounded rounds
            // before this point, in `idle_round`.
            while state.state.load(Ordering::Acquire) != AWAKE {
                thread::park();
            }
        }
        idle.wake_fully();
    }

    /// Producer-side: called after pushing N new jobs. Decides
    /// whether to wake any sleeping workers based on the
    /// awake-but-idle / sleeping counters and whether the deque
    /// was empty before the push.
    #[inline]
    pub(crate) fn new_internal_jobs(&self, num_jobs: u32, queue_was_empty: bool) {
        // Flip JEC from sleepy (even) to active (odd) if any
        // worker is currently in the sleepy phase, so they bail
        // out before parking.
        let counters = self
            .counters
            .increment_jobs_event_counter_if(JobsEventCounter::is_sleepy);
        let awake_but_idle = counters.awake_but_idle_threads() as u32;
        let num_sleepers = counters.sleeping_threads() as u32;
        if num_sleepers == 0 {
            return;
        }
        if !queue_was_empty {
            // Queue was already non-empty: existing idle workers
            // aren't keeping up, wake more.
            let n = num_jobs.min(num_sleepers);
            self.wake_any_threads(n);
        } else if awake_but_idle < num_jobs {
            // Queue was empty: only wake if we don't already have
            // enough idle workers spinning.
            let n = (num_jobs - awake_but_idle).min(num_sleepers);
            self.wake_any_threads(n);
        }
    }

    /// Wake up to `num` sleeping workers. Walks the worker_states
    /// in order until enough have been woken (or all probed).
    fn wake_any_threads(&self, mut num: u32) {
        if num == 0 {
            return;
        }
        for i in 0..self.worker_states.len() {
            if self.wake_specific_thread(i) {
                crate::sched::trace::emit(crate::sched::trace::TraceEvent::PoolWake, i as u32);
                num -= 1;
                if num == 0 {
                    return;
                }
            }
        }
    }

    /// Wake one specific worker. Returns true if the worker was
    /// asleep and this call claimed it; false if it was awake or
    /// already claimed.
    fn wake_specific_thread(&self, idx: usize) -> bool {
        let state = &self.worker_states[idx];
        // SeqCst against the sleeper's publish-then-recheck. Either
        // this claim sees SLEEPING, or the sleeper's re-read sees what
        // this caller stored before calling; one of the two always
        // holds. A failed SeqCst compare-exchange is a SeqCst load, so
        // it sits in the same total order as the store it races.
        if state
            .state
            .compare_exchange(SLEEPING, WAKING, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        // The worker keeps parking while it reads WAKING, so this
        // decrement lands before it can take work and decrement
        // inactive. A producer reading the counters between the claim
        // and the store below is the only one that sees a sleeper that
        // is already spoken for.
        self.counters.sub_sleeping_thread();
        state.state.store(AWAKE, Ordering::SeqCst);
        if let Some(handle) = state.handle.get() {
            handle.unpark();
        }
        true
    }

    /// Wake every worker (for shutdown). Called once when the
    /// arena is being torn down.
    pub(crate) fn wake_all_for_shutdown(&self) {
        // Stored before the sweep, never after. A worker whose slot has
        // already been swept has to find the flag set, or it parks
        // behind the wake and stays there.
        //
        // SeqCst rather than Release, and the difference is the whole
        // guarantee. This store and the swap below are one half of a
        // Dekker pair whose other half is in `sleep`: publish, then
        // read what the other side published. The argument only holds
        // when all four operations are in the one total order. A
        // Release store is not in it, which leaves this interleaving
        // legal on a weakly ordered target:
        //
        //   waker    stores shutdown, not yet visible
        //   waker    swaps state, reads AWAKE, so unparks nobody
        //   sleeper  stores SLEEPING
        //   sleeper  reads shutdown, sees false
        //   sleeper  parks, and nothing will wake it
        //
        // which is the wedge this flag exists to prevent. x86 hides it
        // because the locked swap drains the store buffer; aarch64 does
        // not. Teardown runs once per arena, so the ordering costs
        // nothing anyone measures.
        self.shutdown.store(true, Ordering::SeqCst);
        for i in 0..self.worker_states.len() {
            self.wake_specific_thread(i);
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_monitor_round_is_the_dispatch_cost_in_cycles_and_zero_without_a_profile() {
        // 4 GHz is 64 sixteenths of a cycle per nanosecond.
        assert_eq!(round_cycles(400, 64), 1_600);
        assert_eq!(round_cycles(2_000, 55), 6_875);
        assert_eq!(round_cycles(0, 64), 0, "no profile is no wait");
        assert_eq!(round_cycles(400, 0), 0, "no rate is no wait");
        assert_eq!(
            round_cycles(u64::MAX, 64),
            u64::MAX / 16,
            "saturates rather than wraps"
        );
    }

    // These drive `should_adapt` and `adapted_window` rather than the
    // process-global statics those read in production. Every park
    // writes PARK_EVENTS, so a test driving the statics has to exclude
    // every other test that dispatches, and the only way to do that is
    // a lock. Against the functions there is nothing to exclude: the
    // arithmetic is the behavior, and two tests calling it at once
    // cannot see each other.

    #[test]
    fn a_window_that_keeps_being_missed_shrinks_toward_the_floor() {
        // Parks dominating means the spin ran out before work arrived,
        // which on a contended host is a worker burning slices a
        // neighbour could have used.
        let mut window = adapted_window(DEFAULT_SPIN_WINDOW_ROUNDS, 300, 4);
        assert_eq!(window, DEFAULT_SPIN_WINDOW_ROUNDS / 2);

        for _ in 0..12 {
            window = adapted_window(window, 300, 4);
        }
        assert_eq!(window, FLOOR_SPIN_WINDOW_ROUNDS);
    }

    #[test]
    fn a_window_that_keeps_paying_grows_back_but_never_past_the_tuned_default() {
        // Rescues dominating means work landed inside the window and the
        // spin saved a park and unpark pair.
        let mut window = FLOOR_SPIN_WINDOW_ROUNDS;
        for _ in 0..64 {
            window = adapted_window(window, 4, 300);
        }
        assert_eq!(window, DEFAULT_SPIN_WINDOW_ROUNDS);
    }

    #[test]
    fn one_burst_does_not_move_the_window() {
        // Below the evidence floor the controller has seen too little to
        // tell a workload's shape from a moment of it.
        assert!(!should_adapt(true, 200, 0), "200 parks is under the floor");
        assert!(
            should_adapt(true, 200, EVIDENCE_FLOOR - 200),
            "the floor counts parks and rescues together"
        );
    }

    #[test]
    fn a_held_window_tells_a_controller_that_ran_from_one_that_never_reached_the_gate() {
        // Rescues dominating grows the window and clamps it to the
        // default it started from, so the window comes back unmoved
        // while the controller did decide. The two are told apart by
        // asking whether it had the evidence to run, not by a counter:
        // the counter is process-wide and monotonic, so any test that
        // parks raises it between two readings.
        assert!(
            should_adapt(true, 4, 300),
            "this much evidence reaches the gate"
        );
        assert_eq!(
            adapted_window(DEFAULT_SPIN_WINDOW_ROUNDS, 4, 300),
            DEFAULT_SPIN_WINDOW_ROUNDS,
            "clamped to where it began"
        );
    }

    #[test]
    fn the_controller_stays_still_while_it_is_off() {
        // Off is the shipped default, and the window it holds is the one
        // tuned across three host classes. Asked of the gate rather
        // than by storing ADAPTIVE false, which is a process-global
        // write every other controller test would then read.
        assert!(
            !should_adapt(false, 1_000, 0),
            "evidence past the floor still decides nothing while it is off"
        );
        assert!(
            should_adapt(true, 1_000, 0),
            "and the same evidence decides once it is on"
        );
    }

    #[test]
    fn counters_initial_state_is_zero() {
        let c = AtomicCounters::new();
        let snap = c.load(Ordering::SeqCst);
        assert_eq!(snap.inactive_threads(), 0);
        assert_eq!(snap.sleeping_threads(), 0);
        assert_eq!(snap.awake_but_idle_threads(), 0);
    }

    #[test]
    fn add_then_sub_inactive_returns_to_zero() {
        let c = AtomicCounters::new();
        c.add_inactive_thread();
        assert_eq!(c.load(Ordering::SeqCst).inactive_threads(), 1);
        c.sub_inactive_thread();
        assert_eq!(c.load(Ordering::SeqCst).inactive_threads(), 0);
    }

    #[test]
    fn jec_starts_sleepy_and_flips_active_on_increment() {
        let c = AtomicCounters::new();
        let snap = c.load(Ordering::SeqCst);
        assert!(snap.jobs_counter().is_sleepy());
        let after = c.increment_jobs_event_counter_if(|_| true);
        assert!(after.jobs_counter().is_active());
    }

    #[test]
    fn sleep_struct_constructs_with_n_workers() {
        let s = Sleep::new(8);
        assert_eq!(s.num_workers(), 8);
    }
}

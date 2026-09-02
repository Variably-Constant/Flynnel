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
//! 1. ACTIVE: running a job (not counted as inactive).
//! 2. IDLE: finished a job, spinning `yield_now` inside
//!    `no_work_found`; counted as `awake_but_idle`. After
//!    `ROUNDS_UNTIL_SLEEPY` yields the worker transitions to:
//! 3. SLEEPY: announces itself by incrementing JEC (making it
//!    even); producers will see this and bump JEC back to odd if
//!    they post new work. Still counted as `awake_but_idle`.
//!    After `rounds_until_sleeping()` more yields the worker
//!    transitions to:
//! 4. SLEEPING: locks its Mutex, waits on Condvar; counted as
//!    both `inactive` AND `sleeping`. Awoken by
//!    `wake_specific_thread` (which clears the mutex and notifies).
//!
//! Producers (`new_internal_jobs`):
//!   - Increment JEC if it is sleepy (signals sleepy workers to
//!     re-search before they sleep).
//!   - If queue was non-empty, wake `min(num_jobs, num_sleepers)`.
//!   - If queue was empty, wake `max(num_jobs - awake_but_idle, 0)`
//!     capped at num_sleepers.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
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

    /// Sub one sleeping thread. Caller MUST know that at least
    /// one sleeping thread exists (typically because they just
    /// woke one via the condvar).
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
/// before a sleepy worker locks the condvar. 500 rounds is
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
/// Controller evidence since the last adjust: workers that PARKED
/// (the spin was wasted - work did not arrive in the window) versus
/// workers RESCUED mid-spin (the spin paid off - it avoided a
/// park/unpark syscall pair).
static PARK_EVENTS: AtomicU32 = AtomicU32::new(0);
static RESCUE_EVENTS: AtomicU32 = AtomicU32::new(0);
/// Total idle `yield_now` rounds, exposed for observability. This is
/// the quantity a flamegraph attributes to `sched_yield`.
static TOTAL_YIELDS: AtomicU64 = AtomicU64::new(0);

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

/// Yield rounds before a sleepy worker locks the condvar. Reads the
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
fn maybe_adapt() {
    if !ADAPTIVE.load(Ordering::Relaxed) {
        return;
    }
    let park = PARK_EVENTS.load(Ordering::Relaxed);
    let rescue = RESCUE_EVENTS.load(Ordering::Relaxed);
    if park + rescue < 256 {
        return;
    }
    let cur = SPIN_WINDOW.load(Ordering::Relaxed);
    let new = if park > rescue.saturating_mul(3) {
        (cur / 2).max(FLOOR_SPIN_WINDOW_ROUNDS)
    } else if rescue > park {
        (cur + cur / 4 + 1).min(DEFAULT_SPIN_WINDOW_ROUNDS)
    } else {
        cur
    };
    SPIN_WINDOW.store(new, Ordering::Relaxed);
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

/// Per-worker sleep state held inside the global `Sleep` struct.
///
/// `#[repr(align(128))]` so two adjacent workers in the Vec never
/// share a 128-byte prefetched cache-line pair. Without this, worker
/// 0's `is_blocked` mutex acquire invalidates worker 1's cached
/// state. Cilk's `CILK_CACHE_LINE = 128` is the same rationale.
#[repr(align(128))]
struct WorkerSleepState {
    is_blocked: Mutex<bool>,
    condvar: Condvar,
}

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
    /// Set once this worker has actually parked on the condvar in this
    /// idle episode, so a later find is not miscounted as a spin
    /// rescue (the spin did not save this worker - it parked).
    pub parked: bool,
}

impl IdleState {
    pub(crate) fn new(worker_index: usize) -> Self {
        Self {
            worker_index,
            rounds: 0,
            jobs_counter: JobsEventCounter::DUMMY,
            parked: false,
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

/// Process-global sleep coordinator. One instance per arena;
/// referenced by every worker and every producer.
pub(crate) struct Sleep {
    counters: AtomicCounters,
    worker_states: Vec<WorkerSleepState>,
}

impl Sleep {
    pub(crate) fn new(num_workers: usize) -> Self {
        assert!(num_workers <= THREADS_MAX, "too many workers");
        let mut states = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            states.push(WorkerSleepState {
                is_blocked: Mutex::new(false),
                condvar: Condvar::new(),
            });
        }
        Self {
            counters: AtomicCounters::new(),
            worker_states: states,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn num_workers(&self) -> usize {
        self.worker_states.len()
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

    /// Worker-side: called when one search round produced no
    /// work. Advances the idle state through yield -> sleepy ->
    /// sleeping. `has_injected_jobs` is called inside the sleep
    /// transition to recover from the race where a job was
    /// injected between us going sleepy and locking the mutex.
    pub(crate) fn no_work_found(
        &self,
        idle: &mut IdleState,
        has_injected_jobs: impl FnOnce() -> bool,
    ) {
        if idle.rounds < ROUNDS_UNTIL_SLEEPY {
            TOTAL_YIELDS.fetch_add(1, Ordering::Relaxed);
            thread::yield_now();
            idle.rounds += 1;
        } else if idle.rounds == ROUNDS_UNTIL_SLEEPY {
            idle.jobs_counter = self.announce_sleepy();
            idle.rounds += 1;
            TOTAL_YIELDS.fetch_add(1, Ordering::Relaxed);
            thread::yield_now();
        } else if idle.rounds < rounds_until_sleeping() {
            idle.rounds += 1;
            TOTAL_YIELDS.fetch_add(1, Ordering::Relaxed);
            thread::yield_now();
        } else {
            self.sleep(idle, has_injected_jobs);
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

    /// Worker-side: actually go to sleep on the condvar after the
    /// sleepy phase. Returns immediately if the JEC has changed
    /// (i.e. a producer posted work in the meantime).
    fn sleep(
        &self,
        idle: &mut IdleState,
        has_injected_jobs: impl FnOnce() -> bool,
    ) {
        let state = &self.worker_states[idle.worker_index];
        let mut is_blocked = state.is_blocked.lock().unwrap();
        debug_assert!(!*is_blocked);

        loop {
            let counters = self.counters.load(Ordering::SeqCst);
            debug_assert!(idle.jobs_counter.is_sleepy());
            if counters.jobs_counter() != idle.jobs_counter {
                // JEC changed: work posted since we went sleepy.
                // Bail out and resume searching.
                idle.wake_partly();
                return;
            }
            if self.counters.try_add_sleeping_thread(counters) {
                break;
            }
        }

        // Registered as sleeping. One last check for injected
        // jobs (closes the deadlock race where work was injected
        // while we were sleepy and our JEC bump rolled over).
        std::sync::atomic::fence(Ordering::SeqCst);
        if has_injected_jobs() {
            self.counters.sub_sleeping_thread();
        } else {
            // Committing to the condvar: the spin did not rescue this
            // worker. Feed the controller before blocking.
            idle.parked = true;
            PARK_EVENTS.fetch_add(1, Ordering::Relaxed);
            maybe_adapt();
            *is_blocked = true;
            while *is_blocked {
                is_blocked = state.condvar.wait(is_blocked).unwrap();
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
        // out before locking the condvar.
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
                num -= 1;
                if num == 0 {
                    return;
                }
            }
        }
    }

    /// Wake one specific worker. Returns true if the worker was
    /// actually asleep (and is now waking up); false if it was
    /// already awake.
    fn wake_specific_thread(&self, idx: usize) -> bool {
        let state = &self.worker_states[idx];
        let mut is_blocked = state.is_blocked.lock().unwrap();
        if *is_blocked {
            *is_blocked = false;
            state.condvar.notify_one();
            self.counters.sub_sleeping_thread();
            true
        } else {
            false
        }
    }

    /// Wake every worker (for shutdown). Called once when the
    /// arena is being torn down.
    pub(crate) fn wake_all_for_shutdown(&self) {
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

//! `Parker`: per-worker park / unpark primitive with a yield-N-then-
//! park spin floor.
//!
//! Built on `std::thread::{park, current().unpark()}`. The std
//! primitive provides the permit-based race resolution: if `unpark`
//! is called before `park`, the permit is stored and the next
//! `park` returns immediately. That eliminates the lost-wakeup
//! window the rayon JEC protocol exists to solve, in exchange for
//! the (cheap) cost of always calling `unpark` even when no one is
//! parked.
//!
//!
//! ## Spin floor policy
//!
//! - Local tier: 8 rounds of `thread::yield_now()` before parking
//!   (per [`crate::sched::SchedTier::spin_rounds`]). Sub-microsecond
//!   work avoids the syscall.
//! - Hierarchical tier: 32 rounds. Multi-microsecond work amortizes
//!   the park / unpark pair.
//! - Federated tier: 0 rounds (direct park). Federated jobs are
//!   millisecond-scale; throughput beats latency.
//!
//! `Parker` accepts the spin-round count at construction so it
//! works across tiers without conditional plumbing.

use core::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::thread::{self, Thread};

/// Selects how the [`Parker`] waits after the spin floor is exhausted.
///
/// Picked at construction time via [`WaitStrategy::pick`]:
/// - WAITPKG-capable silicon (Intel Tremont/Tiger Lake+, AMD Zen 5+)
///   -> [`WaitStrategy::Waitpkg`]: UMONITOR + UMWAIT halt the logical
///   CPU sub-100ns until the watched cache line transitions or the
///   TSC deadline fires. No kernel syscall.
/// - AMD silicon without WAITPKG (Excavator onward, so every Zen
///   before Zen 5) -> [`WaitStrategy::Monitorx`]: MONITORX + MWAITX,
///   the same wake-on-store shape from user mode.
/// - All other silicon -> [`WaitStrategy::StdPark`]: the original
///   `std::thread::park()` path (kernel condvar; ~1us syscall on Linux
///   futex / Windows WaitForSingleObject).
///
/// The strategy is observable via [`Parker::wait_strategy`] for
/// diagnostics + per-host bench gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitStrategy {
    /// `std::thread::park()`. Permits-based; cross-platform; always
    /// works. Kernel transition on the wait path (~1us).
    StdPark,
    /// `UMONITOR` + `UMWAIT`. Halts the logical CPU; wake-on-store
    /// to the monitored cache line. Sub-100ns wake; no syscall.
    /// Available if and only if [`crate::cpu_info::has_waitpkg`]
    /// is true.
    Waitpkg,
    /// `MONITORX` + `MWAITX`. AMD's user-mode monitor-wait, the same
    /// wake-on-store behavior as the WAITPKG pair. Available if and
    /// only if [`crate::cpu_info::has_monitorx`] is true.
    ///
    /// MWAITX takes a relative count of cycles in EBX where UMWAIT
    /// takes an absolute TSC deadline in EDX:EAX. The two are not the
    /// same quantity, and how far one EBX unit reaches is a property
    /// of the part rather than a constant: on a Ryzen 9 7900X one unit
    /// measures half an RDTSC cycle. Nothing in CPUID reports that
    /// ratio, and a busy host also returns from MWAITX early by an
    /// amount that varies between runs.
    /// [`Parker::wait_via_monitorx`] therefore re-arms toward its own
    /// deadline rather than trusting one instruction to reach it,
    /// which is what makes the arm indifferent to both.
    Monitorx,
}

/// Whether a monitor wait on this host suspends the core, which the
/// CPUID bit does not answer and only the length of a real wait
/// reveals. A Ryzen 7 2700 under load returns from MWAITX at once.
///
/// Never set back once false: a wrong false costs the kernel park
/// that was there anyway, a wrong true costs a spin over the
/// instruction pair on every park.
static MONITOR_HOLDS: AtomicBool = AtomicBool::new(true);

/// Separate waits that have found the monitor not holding.
///
/// The verdict is permanent and process-wide, so it is not taken on
/// one wait's evidence. A host whose monitor never holds supplies
/// these in microseconds; a host whose monitor holds has to be
/// unlucky this many times in separate waits.
static MONITOR_DOUBTS: AtomicU32 = AtomicU32::new(0);

/// Waits that must independently find the monitor not holding before
/// the process stops trying.
///
/// One was too few. On a 7900X, whose monitor holds, a single
/// unlucky wait condemned the process and every later group in that
/// run measured the kernel park under a MONITORX label; two runs of
/// one binary read 0.99 us and 5.08 us for the same cell depending
/// only on whether that happened before or after the cell ran.
const DOUBTS_BEFORE_GIVING_UP: u32 = 8;

/// Whether re-arming a monitor is still believed to be worth it.
fn monitor_holds() -> bool {
    MONITOR_HOLDS.load(Ordering::Relaxed)
}

/// How long the arms took in the most recent wait that doubted the
/// monitor, in RDTSC cycles.
///
/// The verdict says a wait got nowhere; this says how fast. Arms
/// totalling about four times the instruction pair mean the monitor
/// never armed at all, while hundreds of thousands of cycles mean it
/// armed and something ended it, and those want different answers.
/// Nothing in the verdict distinguishes them.
static LAST_DOUBT_CYCLES: AtomicU64 = AtomicU64::new(0);

/// Record that one wait did not suspend the core, and stop trying
/// once enough separate waits have said so.
fn note_monitor_does_not_hold(span_cycles: u64) {
    LAST_DOUBT_CYCLES.store(span_cycles, Ordering::Relaxed);
    if MONITOR_DOUBTS.fetch_add(1, Ordering::Relaxed) + 1 >= DOUBTS_BEFORE_GIVING_UP {
        MONITOR_HOLDS.store(false, Ordering::Relaxed);
    }
}

/// Waits that have found the monitor not holding, and how long the
/// arms took in the most recent of them.
///
/// For benches and diagnostics. A run that reports the monitor gave
/// up says nothing about why, and the two causes it cannot separate
/// need different fixes.
pub fn monitor_doubts() -> (u32, u64) {
    (
        MONITOR_DOUBTS.load(Ordering::Relaxed),
        LAST_DOUBT_CYCLES.load(Ordering::Relaxed),
    )
}

/// Whether any wait has found this host's monitor not to hold.
///
/// Exposed for the bench and for diagnostics: a MONITORX run whose
/// numbers look like the StdPark ones beside it has usually fallen
/// back, and without this that is indistinguishable from the wait
/// being no faster.
///
/// # True does not mean the monitor wait ran
///
/// This starts true and goes false only after
/// [`DOUBTS_BEFORE_GIVING_UP`] waits have each found the monitor not
/// holding, and those are counted inside the MONITORX path. A process
/// whose parker never chose that path executed no monitor wait,
/// recorded no doubt, and leaves this reading true having observed
/// nothing. True-after-none and true-after-thousands are the same
/// value.
///
/// So this answers "has the monitor been given up on", never "was the
/// monitor used". For the second, read
/// [`WaitController::report`]: `switches` above zero with `in_use` of
/// [`WaitStrategy::Monitorx`] is what says the parker moved, and
/// `challenger_samples` of zero says the controller never sampled the
/// arm at all. [`monitor_doubts`] beside them separates a monitor
/// that was tried and held from one that was never tried.
pub fn monitor_wait_held() -> bool {
    monitor_holds()
}

impl WaitStrategy {
    /// The strategy a parker built by [`Parker::new`] starts on, and
    /// the one it returns to whenever the controller has no verdict.
    ///
    /// WAITPKG where the silicon has it, because UMWAIT takes the
    /// deadline this parker wants directly. Otherwise the kernel
    /// park, which is what shipped before any of this and is
    /// therefore the floor a measurement has to beat.
    ///
    /// MONITORX is never the starting point even where present. It is
    /// reached only by [`wait_controller`] measuring it cheaper here,
    /// because whether it is cheaper depends on the load and not only
    /// on the part. Wake latency, unpark to observable return, in
    /// microseconds:
    ///
    /// ```text
    ///                 7900X                  2700
    ///            StdPark  MONITORX      StdPark  MONITORX
    ///   idle  50   6.0       1.4          19.6      3.4
    ///   idle 500   4.8       1.0          25.1      2.8
    ///   load  50   6.6       1.1          14.1    319.2
    ///   load 500   6.5       0.5          16.5    319.2
    /// ```
    ///
    /// Better on the 7900X everywhere, better on an idle 2700 by five
    /// to nine times, and twenty times worse on a loaded one. No
    /// property of the host settles that, which is why the choice is
    /// measured per process and revisited rather than decided here.
    pub fn baseline() -> Self {
        if crate::cpu_info::has_waitpkg() {
            Self::Waitpkg
        } else {
            Self::StdPark
        }
    }

    /// The strategy to use for the next park on this host.
    ///
    /// [`Self::baseline`] until the controller has measured something
    /// cheaper. Kept as the name the rest of the crate calls, so a
    /// caller that just wants "the right one" is unaffected by where
    /// the answer comes from.
    pub fn pick() -> Self {
        wait_controller().choose()
    }
}

/// Which wait is cheapest on this host right now, learned by timing
/// the waits the scheduler was going to do anyway.
///
/// # Why this is measured rather than decided
///
/// A monitor wait beats a kernel park on some parts and loses on
/// others, and on at least one part it does both depending on how
/// busy the host is. Nothing readable at construction separates
/// those: CPUID reports the instruction, not what it costs, and the
/// cost moves with the load. So the strategies are raced against each
/// other in the parks the pool performs anyway, and the cheaper one
/// is used until it stops being cheaper.
///
/// # Shape
///
/// The evidence gate the spin controller uses before it moves its
/// window, over a paired comparison rather than a running one.
///
/// A probe is two consecutive timed parks, one per arm, so every
/// baseline reading has a challenger reading beside it in time. That
/// pairing is the control: on this project's hosts a quiet draw and a
/// loaded draw of one quantity differ by hundreds to thousands of
/// times, which is far more than the two arms differ from each other,
/// so readings gathered over different stretches would compare the
/// machine's mood and not the waits.
///
/// **The pair is also what decides.** Each one credits a point to
/// whichever arm was cheaper in it, and the running count is the
/// verdict. Comparing the two means instead answers to their largest
/// draws, and a wake latency has a long tail: on a 7900X this
/// controller read its challenger at 17939 ns over 100 samples while
/// a pinned arm measured the same wait at 888 ns under the same
/// conditions, because a handful of draws near 340 us carried the
/// average, and it declined a wait nearly seven times cheaper. The
/// means are still kept, and reported, because the tail is worth
/// seeing; they are not what moves the process.
///
/// Probing never stops, so there is no verdict that outlives its
/// evidence. A host that gets busy is noticed because the pairs stop
/// falling the same way and the count walks back.
///
/// What does change is how often. Each probe spends one park on the
/// arm not in use, which on a host that settled against the monitor
/// wait is a park costing twenty times what the chosen one costs, so
/// a fixed rate is a standing tax for the life of the process. The
/// interval doubles toward [`PROBE_EVERY_MAX`] while the answer keeps
/// coming back the same and collapses to [`PROBE_EVERY_MIN`] the
/// moment the two means fall within the margin of each other, which
/// happens before the order flips rather than after.
///
/// # Cost on the path that does not use it
///
/// [`Self::choose`] is one relaxed load. Timing happens on sampled
/// parks only, and the sample flag is read by `unpark` out of a cache
/// line it is already writing, so an unsampled wake pays nothing it
/// was not already paying. A park is the slow path by construction:
/// the cheapest outcome measured here is about half a microsecond and
/// the timing costs two clock reads, so a timed park pays a few per
/// cent and one park in [`PROBE_EVERY_MIN`] is timed at the closest
/// the controller ever looks.
pub struct WaitController {
    /// Mean observed wake cost in nanoseconds, indexed by
    /// [`Self::slot`]. Zero means unmeasured.
    mean_ns: [AtomicU64; 2],
    /// Samples folded into each mean.
    samples: [AtomicU32; 2],
    /// Samples taken, which paces the re-probe. Counted here rather
    /// than counting parks, because this is touched only on a park
    /// that is already being timed.
    parks: AtomicU64,
    /// The strategy [`Self::choose`] currently returns, as a slot.
    current: AtomicU32,
    /// Times a verdict changed which strategy is in use.
    switches: AtomicU64,
    /// Parks a thread takes between probes, between
    /// [`PROBE_EVERY_MIN`] and [`PROBE_EVERY_MAX`].
    probe_every: AtomicU32,
    /// Completed pairs the challenger has won less those the
    /// baseline has won, saturating at [`SCORE_CAP`] either way.
    score: AtomicI32,
}

/// How far the running score can run in either direction.
///
/// It is what bounds recovery: a host that changes character has to
/// win this many pairs back before the count even reaches zero, so a
/// cap far above [`SWITCH_AT`] would buy confidence with a verdict
/// that takes too long to leave.
const SCORE_CAP: i32 = 64;

/// Net pairs one arm must be ahead by before the process moves to it.
///
/// A margin already keeps a pair from being scored at all unless the
/// two readings separate, so every point here is a pair where one
/// arm was clearly cheaper. Wanting this many of them is what stops
/// a short run of luck moving the process.
const SWITCH_AT: i32 = 24;

/// Parks between probes while a verdict is forming or contested.
///
/// Low enough that a verdict arrives inside a second of ordinary
/// scheduling and high enough that the two clock reads it costs are
/// spread thin.
const PROBE_EVERY_MIN: u32 = 64;

/// Parks between probes once the same answer has come back several
/// times running.
///
/// A probe spends one park on the arm not in use, and on a host where
/// that arm is far worse the difference lands on the pool's park
/// latency. The measured spread here reaches twenty times, so at the
/// close rate that is a sixth of every park, forever, on a host that
/// settled against the monitor wait in the first place. Widening to
/// this leaves a fortieth of that, which is under the run-to-run
/// spread of the thing being protected, while still bringing eight
/// probes inside a few tens of thousands of parks so a host that
/// changes character is noticed.
const PROBE_EVERY_MAX: u32 = 4096;

/// A thread's place in the probe cycle.
#[derive(Clone, Copy)]
struct ProbeCursor {
    /// Parks since this thread last completed a pair.
    since: u32,
    /// Interval it is counting to, refreshed when a pair starts.
    every: u32,
    /// Slots the open pair still owes a reading, as a bitmask. Zero
    /// when no pair is open.
    pending: u32,
    /// Slot of the pair's first reading, or [`NO_SLOT`].
    held_slot: u32,
    /// That reading, nanoseconds, waiting for its partner.
    held_ns: u64,
}

/// Both halves of a pair are owed.
const BOTH_SLOTS: u32 = (1 << SLOT_BASELINE) | (1 << SLOT_CHALLENGER);

/// No half of a pair is being held.
const NO_SLOT: u32 = u32::MAX;

thread_local! {
    /// Where this thread is in the probe cycle.
    ///
    /// Thread-local so the cadence costs no shared line. A worker
    /// that parks rarely probes rarely, which is correct: the
    /// controller wants samples in proportion to how much a thread
    /// actually parks. The interval is copied in when a pair starts
    /// rather than read per park, so an unsampled park reads nothing
    /// another worker writes.
    static PROBE: core::cell::Cell<ProbeCursor> = const {
        core::cell::Cell::new(ProbeCursor {
            since: 0,
            every: PROBE_EVERY_MIN,
            pending: 0,
            held_slot: NO_SLOT,
            held_ns: 0,
        })
    };
}

/// Samples after which an arm's mean stops accumulating and starts
/// following an exponential weight.
///
/// The mean is reported rather than obeyed, so this sets how quickly
/// the reported figure tracks a change in load and nothing about the
/// verdict, which comes from [`WaitController::score_pair`].
const SAMPLES_BEFORE_EWMA: u32 = 32;

/// How much cheaper the challenger must be before the process moves.
///
/// A switch costs nothing directly, but a strategy that flaps spends
/// its life exploring, and two means within noise of each other carry
/// no information worth acting on. A fifth is well outside the spread
/// seen between repeat runs on both hosts and well inside the four to
/// fourteen times the arms actually differ by when they differ.
const MARGIN_PER_CENT: u64 = 20;

static WAIT_CONTROLLER: WaitController = WaitController {
    mean_ns: [AtomicU64::new(0), AtomicU64::new(0)],
    samples: [AtomicU32::new(0), AtomicU32::new(0)],
    parks: AtomicU64::new(0),
    current: AtomicU32::new(SLOT_BASELINE),
    switches: AtomicU64::new(0),
    probe_every: AtomicU32::new(PROBE_EVERY_MIN),
    score: AtomicI32::new(0),
};

/// Slot of [`WaitStrategy::baseline`], whatever that resolves to.
const SLOT_BASELINE: u32 = 0;
/// Slot of the monitor wait being raced against it.
const SLOT_CHALLENGER: u32 = 1;

/// The process-wide wait controller.
pub fn wait_controller() -> &'static WaitController {
    &WAIT_CONTROLLER
}

impl WaitController {
    /// The strategy the next park should use.
    fn choose(&self) -> WaitStrategy {
        if self.current.load(Ordering::Relaxed) == SLOT_CHALLENGER
            && let Some(c) = challenger()
        {
            return c;
        }
        WaitStrategy::baseline()
    }

    /// Whether this park should be timed, and which slot it will
    /// report against. Called once per park, before the wait.
    ///
    /// This is also what drives exploration: the park it asks for is
    /// the arm being measured, not the arm the verdict prefers, so a
    /// probe is the only thing that ever runs the losing wait.
    ///
    /// # What a park between probes pays
    ///
    /// Nothing shared. The challenger check comes first, so a host
    /// with no second strategy leaves without touching the controller
    /// at all, and the cursor is thread-local, so the parks between
    /// probes neither read nor write a line another worker touches.
    /// The interval is copied into the cursor when a pair opens, so
    /// even the one relaxed load that sets it is paid per probe
    /// rather than per park.
    fn sample_plan(&self) -> Option<(WaitStrategy, u32)> {
        let challenger = challenger()?;

        // A probe is two consecutive timed parks, one per arm, and
        // the pair is the controller's control. It is not optional.
        // A quiet draw and a loaded draw of the same quantity differ
        // here by hundreds to thousands of times, far more than the
        // arms differ from each other, so two means gathered over
        // different stretches compare the load and not the waits.
        // Taking the two halves back to back keeps them inside the
        // same conditions however far apart the probes themselves
        // are, which is what lets the interval widen.
        let slot = PROBE.with(|p| {
            let mut c = p.get();
            if c.pending == 0 {
                c.since += 1;
                if c.since < c.every {
                    p.set(c);
                    return None;
                }
                c.pending = BOTH_SLOTS;
                c.every = self.probe_every.load(Ordering::Relaxed);
            }
            // A slot stays owed until it has produced a reading. A
            // park can end with nothing to report, because the wake
            // it was armed for never came, and only the kernel park
            // does that often: the monitor arms re-enter their wait
            // until the counter actually moves, while `thread::park`
            // is documented to return spuriously and the caller
            // re-enters. Dropping the slot there would sample the
            // baseline less often than the challenger, and the pair
            // is the whole control.
            let slot = if c.pending == BOTH_SLOTS {
                // Which arm leads alternates across probes, so
                // neither is always measured first out of a cold
                // cache.
                (self.parks.fetch_add(1, Ordering::Relaxed) % 2) as u32
            } else {
                u32::from(c.pending == (1 << SLOT_CHALLENGER))
            };
            p.set(c);
            Some(slot)
        })?;

        Some((
            if slot == SLOT_CHALLENGER {
                challenger
            } else {
                WaitStrategy::baseline()
            },
            slot,
        ))
    }

    /// Open a probe pair on this thread, so a test can drive
    /// [`Self::record`] down the path a probe takes.
    #[cfg(test)]
    fn open_pair() {
        PROBE.with(|p| {
            let mut c = p.get();
            c.pending = BOTH_SLOTS;
            c.held_slot = NO_SLOT;
            p.set(c);
        });
    }

    /// Note that a slot of the open pair has produced its reading,
    /// and hand back the partner reading once both are in.
    ///
    /// The pair closes only when both have reported, and the interval
    /// to the next one is counted from there.
    fn probe_served(slot: u32, wake_ns: u64) -> Option<u64> {
        PROBE.with(|p| {
            let mut c = p.get();
            c.pending &= !(1u32 << slot);
            if c.pending != 0 {
                // First half. Hold it for its partner rather than
                // comparing against a mean, so the two readings
                // being compared are the two this thread took next
                // to each other.
                c.held_slot = slot;
                c.held_ns = wake_ns;
                p.set(c);
                return None;
            }
            c.since = 0;
            let partner = (c.held_slot != NO_SLOT && c.held_slot != slot).then_some(c.held_ns);
            c.held_slot = NO_SLOT;
            c.held_ns = 0;
            p.set(c);
            partner
        })
    }

    /// Fold one timed wake into a slot's mean, score the pair it
    /// completes, and re-decide.
    fn record(&self, slot: u32, wake_ns: u64) {
        if let Some(partner_ns) = Self::probe_served(slot, wake_ns) {
            let (base, chal) = if slot == SLOT_CHALLENGER {
                (partner_ns, wake_ns)
            } else {
                (wake_ns, partner_ns)
            };
            self.score_pair(base, chal);
        }
        let i = slot as usize;
        let n = self.samples[i].fetch_add(1, Ordering::Relaxed) + 1;
        let prev = self.mean_ns[i].load(Ordering::Relaxed);
        // Cumulative mean while the sample count is small, then an
        // exponential one, so early samples are not swamped and a
        // later shift in load still moves the figure.
        //
        // The delta is signed. A sample under the mean has to pull it
        // down, and the cumulative phase is the one that establishes
        // the verdict, so a mean that can only rise ranks the arms by
        // the worst draw each happened to take.
        let next = if prev == 0 {
            wake_ns
        } else if n <= SAMPLES_BEFORE_EWMA {
            let prev_i = prev as i64;
            (prev_i + (wake_ns as i64 - prev_i) / i64::from(n)).max(0) as u64
        } else {
            (prev * 7 + wake_ns) / 8
        };
        self.mean_ns[i].store(next, Ordering::Relaxed);
        self.decide();
    }

    /// Credit one completed pair to whichever arm was cheaper in it.
    ///
    /// A pair is one comparison, not two measurements: both readings
    /// were taken by one thread moments apart, so the slower of them
    /// is the slower wait under the conditions that applied to both.
    /// Counting those outcomes is what keeps a tail from deciding.
    /// A wake latency has a long one, and a mean over it answers to
    /// its largest draws: on a 7900X the controller read its
    /// challenger at 17939 ns over 100 samples while a pinned arm
    /// measured the same wait at 888 ns under the same conditions,
    /// because a handful of draws near 340 us carried the average.
    /// It declined a wait almost seven times cheaper on that.
    fn score_pair(&self, base_ns: u64, chal_ns: u64) {
        let step = if chal_ns * 100 < base_ns * (100 - MARGIN_PER_CENT) {
            1
        } else if base_ns * 100 < chal_ns * (100 - MARGIN_PER_CENT) {
            -1
        } else {
            // Inside the margin this pair separates nothing, and
            // scoring it either way would be scoring noise.
            return;
        };
        self.score
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |s| {
                Some((s + step).clamp(-SCORE_CAP, SCORE_CAP))
            })
            .expect("the update closure returns Some on every call");
    }

    /// Adopt whichever side has won enough pairs to be worth moving
    /// to, and set how soon to look again.
    fn decide(&self) {
        let score = self.score.load(Ordering::Relaxed);

        if score >= SWITCH_AT || score <= -SWITCH_AT {
            let want = if score > 0 {
                SLOT_CHALLENGER
            } else {
                SLOT_BASELINE
            };
            if self.current.swap(want, Ordering::Relaxed) != want {
                self.switches.fetch_add(1, Ordering::Relaxed);
            }
        }

        // How soon to look again follows how settled the answer is,
        // because a probe spends one park on the arm not in use and
        // that park costs what it costs. A saturated score is an arm
        // that has won every recent pair, and a score back inside
        // the switching threshold is the order coming apart, which
        // is the moment to watch closely rather than after it flips.
        if score.abs() >= SCORE_CAP {
            let e = self.probe_every.load(Ordering::Relaxed);
            if e < PROBE_EVERY_MAX {
                self.probe_every
                    .store((e * 2).min(PROBE_EVERY_MAX), Ordering::Relaxed);
            }
        } else if score.abs() < SWITCH_AT {
            self.probe_every.store(PROBE_EVERY_MIN, Ordering::Relaxed);
        }
    }

    /// What the controller has measured: the two mean wake costs in
    /// nanoseconds, their sample counts, which slot is in use, and
    /// how many times that has changed.
    ///
    /// Zero samples on a side means it has never been tried, which is
    /// a different state from having been tried and found slow.
    pub fn report(&self) -> WaitControllerReport {
        WaitControllerReport {
            baseline_ns: self.mean_ns[SLOT_BASELINE as usize].load(Ordering::Relaxed),
            challenger_ns: self.mean_ns[SLOT_CHALLENGER as usize].load(Ordering::Relaxed),
            baseline_samples: self.samples[SLOT_BASELINE as usize].load(Ordering::Relaxed),
            challenger_samples: self.samples[SLOT_CHALLENGER as usize].load(Ordering::Relaxed),
            in_use: self.choose(),
            switches: self.switches.load(Ordering::Relaxed),
            probe_every: self.probe_every.load(Ordering::Relaxed),
            score: self.score.load(Ordering::Relaxed),
        }
    }
}

/// What [`WaitController::report`] answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitControllerReport {
    /// Mean wake cost of the baseline wait, nanoseconds, 0 if untried.
    pub baseline_ns: u64,
    /// Mean wake cost of the monitor wait, nanoseconds, 0 if untried.
    pub challenger_ns: u64,
    /// Timed wakes folded into `baseline_ns`.
    pub baseline_samples: u32,
    /// Timed wakes folded into `challenger_ns`.
    pub challenger_samples: u32,
    /// The strategy a park would use now.
    pub in_use: WaitStrategy,
    /// Times the verdict changed which strategy is in use.
    pub switches: u64,
    /// Parks a thread currently takes between probes. Sits at
    /// [`PROBE_EVERY_MIN`] while the answer is unsettled and reaches
    /// [`PROBE_EVERY_MAX`] once it stops changing.
    pub probe_every: u32,
    /// Pairs the monitor wait has won less those the baseline has
    /// won, saturating at [`SCORE_CAP`]. This is what decides; the
    /// two means above are what it cost, and they can disagree
    /// because a mean answers to its largest draws and this does
    /// not.
    pub score: i32,
}

/// A cycle counter reading, or 0 where the target has none.
///
/// Never the wall clock: this times a span of microseconds between
/// two threads on one host, which is what a cycle counter is for, and
/// the crate already assumes an invariant TSC elsewhere.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn tsc_now() -> u64 {
    // SAFETY: `_rdtsc` is a no-side-effect read of the TSC counter,
    // available on every x86_64 CPU produced this century.
    unsafe { core::arch::x86_64::_rdtsc() }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn tsc_now() -> u64 {
    0
}

/// Cycles as nanoseconds, against the rate this host measured for
/// itself rather than any figure derived from its model.
///
/// The controller compares two of its own readings, so a wrong rate
/// scales both and changes no verdict. It is converted anyway because
/// the report is read by people and a cycle count means nothing
/// without the part it was counted on.
fn cycles_to_ns(cycles: u64) -> u64 {
    let per_ns_16 = crate::sched::par_iter::tsc_per_ns_16();
    if per_ns_16 == 0 {
        return cycles;
    }
    cycles.saturating_mul(16) / per_ns_16
}

/// The monitor wait this host could race against the baseline, or
/// `None` where it has none or has been found not to hold.
fn challenger() -> Option<WaitStrategy> {
    // Capability before standing, and the order is the cost. Both
    // reads are cheap, but the CPUID answers are cached in a
    // `OnceLock` written once at startup, while `monitor_holds` is a
    // static this module writes when it gives up on the monitor and
    // shares a line with the counters that record the giving up.
    // This runs on every unpinned park, so a host that can never
    // reach the challenger leaves here having touched nothing that
    // any thread writes.
    if !crate::cpu_info::has_monitorx() || crate::cpu_info::has_waitpkg() {
        return None;
    }
    if !monitor_holds() {
        return None;
    }
    Some(WaitStrategy::Monitorx)
}

/// Per-worker park primitive. One `Parker` per worker thread; the
/// thread that owns it parks via [`Self::park_until`], and any
/// other thread wakes it via [`Self::unpark`].
///
/// On WAITPKG-capable hosts the wait path bypasses the kernel
/// condvar entirely - the producer's `unpark` increments
/// `wake_counter` (one cache-line store), and the parked thread's
/// `UMWAIT` returns as soon as the cache-line transition is observed
/// by the hardware monitor. Sub-100ns wake instead of ~1us syscall.
///
/// On non-WAITPKG hosts the wake_counter increment is still issued
/// (it costs one atomic add) but the wait path falls through to
/// `std::thread::park()` as before.
/// One cache line per parker, because a monitor wait watches the line
/// `wake_counter` sits in and wakes on any store to it. Without this
/// two parkers share a line and each one's unpark fires the other's
/// monitor, and an `Arc`'s refcounts sit immediately before the data
/// so every clone and drop fires it too. Both read as the monitor
/// failing to hold.
#[derive(Debug)]
#[repr(align(64))]
pub struct Parker {
    /// Cached `Thread` handle for cross-thread unpark.
    thread: Thread,
    /// Shutdown signal: set by the arena's drop / explicit
    /// shutdown path. When `true`, [`Self::park_until`] returns
    /// `false` to break the worker loop.
    shutdown: AtomicBool,
    /// How many `thread::yield_now()` rounds to spin before
    /// actually calling `thread::park()`. Picked per tier per
    /// [`crate::sched::SchedTier::spin_rounds`].
    spin_rounds: u32,
    /// Monotonic wake counter. Producers increment on `unpark`;
    /// the WAITPKG path snapshots before park + UMONITOR-watches
    /// the counter's cache line.
    wake_counter: AtomicU64,
    /// Strategy this parker is pinned to, or `None` to take whatever
    /// [`wait_controller`] currently measures cheapest.
    ///
    /// Pinned only by [`Self::with_strategy`], which exists so a
    /// bench can hold one arm still. A worker built by [`Self::new`]
    /// follows the controller, so a verdict reached while it is
    /// parked applies to its next park.
    pinned: Option<WaitStrategy>,
    /// When a park is being timed, the unparker stamps its TSC here
    /// and the waking thread differences it. [`SAMPLE_ARMED`] means a
    /// stamp is wanted, 0 means this park is not timed.
    ///
    /// The unparker reads this out of the cache line it is already
    /// writing `wake_counter` into, so an untimed wake pays one
    /// relaxed load and no clock read.
    wake_stamp: AtomicU64,
}

/// `wake_stamp` value meaning a timed park wants a stamp. Not a
/// plausible TSC, so it cannot be mistaken for one.
const SAMPLE_ARMED: u64 = 1;

impl Parker {
    /// Construct a Parker owned by the calling thread. Captures
    /// the current `Thread` handle for later cross-thread unpark.
    ///
    /// Follows [`wait_controller`] rather than fixing a strategy, so
    /// a verdict reached after this worker started applies to it.
    pub fn new(spin_rounds: u32) -> Self {
        Self {
            thread: thread::current(),
            shutdown: AtomicBool::new(false),
            spin_rounds,
            wake_counter: AtomicU64::new(0),
            pinned: None,
            wake_stamp: AtomicU64::new(0),
        }
    }

    /// Construct a Parker pinned to one wait strategy, which the
    /// controller will not move. Used by benches and tests that need
    /// one arm held still.
    ///
    /// Callers must not pass a strategy this host cannot execute:
    /// [`WaitStrategy::Waitpkg`] without
    /// [`crate::cpu_info::has_waitpkg`], or
    /// [`WaitStrategy::Monitorx`] without
    /// [`crate::cpu_info::has_monitorx`], raise `#UD`.
    pub fn with_strategy(spin_rounds: u32, wait_strategy: WaitStrategy) -> Self {
        Self {
            thread: thread::current(),
            shutdown: AtomicBool::new(false),
            spin_rounds,
            wake_counter: AtomicU64::new(0),
            pinned: Some(wait_strategy),
            wake_stamp: AtomicU64::new(0),
        }
    }

    /// The strategy this parker would use for a park starting now.
    ///
    /// Its pinned one, or the controller's current verdict. Not fixed
    /// for an unpinned parker, so two calls either side of a verdict
    /// legitimately differ.
    pub fn wait_strategy(&self) -> WaitStrategy {
        self.pinned.unwrap_or_else(WaitStrategy::pick)
    }

    /// Block the calling thread until `is_ready` returns `true`,
    /// shutdown is signalled, or the thread is unparked.
    ///
    /// Returns `true` when `is_ready()` was observed or the thread
    /// was unparked; returns `false` on shutdown.
    ///
    /// Polling sequence:
    /// 1. Loop `spin_rounds` times calling `thread::yield_now()`
    ///    between polls. Cheapest path: a worker about to receive
    ///    work via unpark stays out of the parker.
    /// 2. After the spin floor, `thread::park()` ONCE. If we wake
    ///    via unpark (regardless of predicate state) we return
    ///    `true` and let the caller re-attempt the work search.
    ///    This is important when the caller has out-of-band signals
    ///    (e.g., wake-on-push from a peer) that don't update the
    ///    predicate's observed state - the peer's deque might have
    ///    work but the predicate doesn't see it. Returning on
    ///    unpark hands control back to the caller, which then walks
    ///    the peer stealers in its main loop.
    pub fn park_until<F: FnMut() -> bool>(&self, mut is_ready: F) -> bool {
        // Snapshot wake_counter ahead of the spin floor so the waitpkg
        // path can detect any unpark that fires after this snapshot
        // (whether during the spin floor or during the UMWAIT itself).
        let initial_wake = self.wake_counter.load(Ordering::Acquire);

        for _ in 0..self.spin_rounds {
            if self.shutdown.load(Ordering::Acquire) {
                return false;
            }
            if is_ready() {
                return true;
            }
            thread::yield_now();
        }
        if self.shutdown.load(Ordering::Acquire) {
            return false;
        }
        if is_ready() {
            return true;
        }

        // The strategy for this park, and whether it is one of the
        // sampled ones. A pinned parker is never sampled: it exists
        // so a bench can hold an arm still, and feeding its timings
        // to the controller would let the bench move the default it
        // is measuring.
        let (strategy, sampling) = match self.pinned {
            Some(p) => (p, None),
            None => match wait_controller().sample_plan() {
                Some((s, slot)) => {
                    self.wake_stamp.store(SAMPLE_ARMED, Ordering::Relaxed);
                    (s, Some(slot))
                }
                None => (WaitStrategy::pick(), None),
            },
        };

        // Dispatch on wait strategy. Either path returns to the
        // caller on wake (real or spurious); the caller's loop
        // re-attempts the work search and re-enters park_until
        // when still empty.
        match strategy {
            WaitStrategy::StdPark => {
                thread::park();
            }
            // A monitor wait returns on its deadline as well as on a
            // wake, so the deadline re-reads state and the wait is
            // re-entered. Only a wake, a shutdown or a ready
            // predicate returns to the caller.
            WaitStrategy::Waitpkg | WaitStrategy::Monitorx => loop {
                match strategy {
                    WaitStrategy::Monitorx => self.wait_via_monitorx(initial_wake),
                    _ => self.wait_via_waitpkg(initial_wake),
                }
                if self.wake_counter.load(Ordering::Acquire) != initial_wake {
                    break;
                }
                if self.shutdown.load(Ordering::Acquire) {
                    break;
                }
                if is_ready() {
                    break;
                }
            },
        }

        // A sampled park reports what its wake cost, measured from
        // the unparker's stamp rather than from entering the wait, so
        // the figure is the wake and not how long there was nothing
        // to do. A park that ended without an unpark leaves the stamp
        // armed and reports nothing, because there is no wake to time.
        if let Some(slot) = sampling {
            let stamp = self.wake_stamp.swap(0, Ordering::Relaxed);
            if stamp > SAMPLE_ARMED {
                let now = tsc_now();
                wait_controller().record(slot, cycles_to_ns(now.wrapping_sub(stamp)));
            }
        }

        // Final shutdown check before returning so a shutdown
        // unpark surfaces cleanly.
        if self.shutdown.load(Ordering::Acquire) {
            return false;
        }
        true
    }

    /// Wake the parked thread if any. Unconditional: if no thread
    /// is parked, the permit is stored for the next park. This
    /// trades a no-op syscall on the empty case for not having to
    /// track an explicit "is this worker parked" flag.
    ///
    /// Increments [`Self::wake_counter`] before anything else, so the
    /// WAITPKG observer's monitor fires; then calls
    /// `thread::unpark()` so
    /// the [`WaitStrategy::StdPark`] path also wakes. Both are
    /// needed because the Parker is constructed knowing its
    /// strategy but the caller does not need to: this method works
    /// for both strategies uniformly.
    pub fn unpark(&self) {
        // Release-store on wake_counter happens-before the parked
        // observer's Acquire-load post-UMWAIT, so the observer sees
        // any state the producer published prior to unpark.
        // Stamped before the counter moves, so the figure covers the
        // whole wake rather than starting after the store the waiter
        // is watching for. The load is of a field in the line this is
        // about to write anyway, so an untimed unpark pays no clock
        // read and no extra line.
        if self.wake_stamp.load(Ordering::Relaxed) == SAMPLE_ARMED {
            self.wake_stamp.store(tsc_now(), Ordering::Relaxed);
        }
        self.wake_counter.fetch_add(1, Ordering::Release);
        self.thread.unpark();
    }

    /// Signal shutdown. The parked thread observes this on its
    /// next park return and exits its loop.
    ///
    /// Routes through [`Self::unpark`] so the wake_counter increments
    /// and the std::thread permit fires - the waitpkg observer
    /// returns from UMWAIT on the cache-line transition and then
    /// observes `shutdown == true` on its post-park check.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.unpark();
    }

    /// WAITPKG wait path: snapshot wake_counter (caller's), arm
    /// UMONITOR on its cache line, double-check the counter to
    /// catch a wake that landed between caller's snapshot and our
    /// UMONITOR setup, then UMWAIT until the line transitions OR
    /// a TSC deadline fires (~10ms).
    ///
    /// Returning here does not mean wake_counter actually changed -
    /// UMWAIT can return on signals, interrupts, or its hint
    /// expiry. The caller (`park_until`) re-checks `is_ready` and
    /// `shutdown` after wake_via_waitpkg returns and decides
    /// whether to re-park.
    #[cfg(target_arch = "x86_64")]
    fn wait_via_waitpkg(&self, initial_wake: u64) {
        // 10 ms deadline cap so a missed wake (e.g. shutdown raced
        // with a UMONITOR that armed after the shutdown unpark)
        // does not block forever. The deadline TSC is computed
        // assuming a ~2.5 GHz TSC frequency; off-by-2x error is
        // immaterial because the caller re-enters park_until on
        // spurious return anyway.
        const WAIT_DEADLINE_NS: u64 = 10_000_000;
        const TSC_HZ_ESTIMATE: u64 = 2_500_000_000;
        let cycles =
            WAIT_DEADLINE_NS.saturating_mul(TSC_HZ_ESTIMATE / 1_000_000_000);
        // SAFETY: `_rdtsc` is a no-side-effect read of the TSC
        // counter; available on every x86_64 CPU produced this
        // century.
        let now = unsafe { core::arch::x86_64::_rdtsc() };
        let deadline = now.wrapping_add(cycles);
        let lo = deadline as u32;
        let hi = (deadline >> 32) as u32;

        let addr = (&raw const self.wake_counter).cast::<u8>();

        // UMONITOR rax: arm hardware monitor on the cache line
        // containing the wake_counter. Any store to that line wakes
        // UMWAIT, including one to another field of this Parker;
        // `repr(align(64))` on the struct is what keeps a different
        // parker's stores out of it.
        //
        // SAFETY: caller (Parker::new -> WaitStrategy::pick) only
        // installs the Waitpkg strategy when has_waitpkg() returned
        // true, so UMONITOR is not a `#UD`. `addr` is a stable
        // pointer to a live AtomicU64 field of `self`.
        unsafe {
            core::arch::asm!(
                "umonitor rax",
                in("rax") addr,
                options(nostack, preserves_flags),
            );
        }

        // Race window: an unpark that fired between the caller's
        // initial_wake snapshot and the UMONITOR arming would not
        // wake UMWAIT (the monitor was not yet armed). Re-check
        // wake_counter; if it advanced, skip UMWAIT entirely.
        if self.wake_counter.load(Ordering::Acquire) != initial_wake {
            return;
        }

        // UMWAIT ecx, edx:eax with ecx = wake hint:
        //   1 = C0.1 (light wait, fastest wake)
        //   0 = C0.2 (deeper wait, lower power, slower wake)
        // We pick C0.1 for the scheduler's latency-sensitive
        // workload. EDX:EAX carries the absolute TSC deadline.
        //
        // SAFETY: same WAITPKG-available reasoning as the UMONITOR
        // above. UMWAIT modifies CF on return (timeout-vs-wake);
        // we drop preserves_flags accordingly.
        unsafe {
            core::arch::asm!(
                "umwait {hint:e}",
                hint = in(reg) 1u32,
                in("eax") lo,
                in("edx") hi,
                options(nostack),
            );
        }
    }

    /// Non-x86_64 stub. The WAITPKG strategy cannot be installed on
    /// non-x86_64 targets (the CPUID probe in `crate::cpu_info`
    /// returns false), so this branch is unreachable in practice.
    /// Fall through to `thread::park()` as a defensive default.
    #[cfg(not(target_arch = "x86_64"))]
    fn wait_via_waitpkg(&self, _initial_wake: u64) {
        thread::park();
    }

    /// MONITORX wait path: the same protocol
    /// [`Self::wait_via_waitpkg`] runs, over AMD's user-mode pair, and
    /// bounded to the same 10 ms.
    ///
    /// Re-arms toward the deadline instead of asking one instruction
    /// to reach it. MWAITX counts EBX units, and how far a unit
    /// reaches is not in CPUID: half an RDTSC cycle on a Ryzen 9
    /// 7900X, a whole one on a Ryzen 7 2700. A busy host also returns
    /// early by a varying amount. Both cost iterations here and
    /// neither is read.
    ///
    /// Returning does not mean `wake_counter` changed: the budget can
    /// run out and the monitor can fire on an unrelated store to the
    /// watched line. `park_until` re-checks `is_ready` and
    /// `shutdown`.
    #[cfg(target_arch = "x86_64")]
    fn wait_via_monitorx(&self, initial_wake: u64) {
        // Matches the WAITPKG arm's cap and its reasoning: a missed
        // wake must not block forever, and the estimate may be off by
        // a factor without mattering, because a short budget costs a
        // re-park and a long one is cut short by the shutdown and
        // wake checks below.
        const WAIT_DEADLINE_NS: u64 = 10_000_000;
        const TSC_HZ_ESTIMATE: u64 = 2_500_000_000;
        // A monitor that keeps firing on traffic to a neighbouring
        // address would otherwise spin here for the whole budget. The
        // count bounds that case on its own, without assuming any
        // iteration actually waits.
        const MAX_ARMS: u32 = 256;
        // This many arms inside `ARMS_TOO_FAST_CYCLES` means the
        // monitor is not holding, and the wait parks instead. Four
        // honoured budgets take tens of milliseconds; four unheld
        // ones take about four times the instruction pair, measured
        // at 2369 cycles on a 7900X and 1606 on a 2700.
        const ARMS_BEFORE_JUDGING: u32 = 4;
        // Below this, four arms are impossibly fast for a monitor
        // that armed at all. Unheld they cost about four times the
        // pair, 9,500 cycles on a 7900X and 6,400 on a 2700. Armed
        // and cut short by something else they measured 420,791 on a
        // 7900X under load, about 22 microseconds each. This sits
        // roughly eight times clear of both.
        //
        // A million was tried, and is why this figure is measured
        // rather than reasoned: it caught the interrupted case, so a
        // monitor that was working got condemned for being woken by
        // the load the arm exists to survive.
        const ARMS_TOO_FAST_CYCLES: u64 = 50_000;

        // Read before anything else this function does. Whether the
        // monitor holds is a property of the part, so a finding by any
        // parker in this process applies to all of them and is never
        // re-tested; and on a part that has given up, everything below
        // is cost paid on the way to a kernel park. A clock read alone
        // measures 206 ns on this project's Windows host and 973 on
        // its Linux guest.
        if !monitor_holds() {
            thread::park();
            return;
        }

        let budget = WAIT_DEADLINE_NS.saturating_mul(TSC_HZ_ESTIMATE / 1_000_000_000);
        // SAFETY: `_rdtsc` is a no-side-effect read of the TSC
        // counter; available on every x86_64 CPU produced this
        // century.
        let start = unsafe { core::arch::x86_64::_rdtsc() };

        let addr = (&raw const self.wake_counter).cast::<u8>();

        let mut arms = 0u32;
        for _ in 0..MAX_ARMS {
            arms += 1;
            // MONITORX rax: arm the monitor on the line holding
            // wake_counter. ECX carries extensions and EDX hints, both
            // zero, which is the only defined combination.
            //
            // SAFETY: Parker::new -> WaitStrategy::pick only installs
            // this strategy when has_monitorx() returned true, so the
            // opcode is not a `#UD`. `addr` points at a live AtomicU64
            // field of `self`. Encoded as bytes because the mnemonic
            // needs a target feature this crate does not set, and the
            // encoding is fixed.
            // The zeros arrive as operands rather than through `xor`,
            // because `xor` writes flags and this block promises not
            // to. MONITORX itself leaves them alone, so the promise
            // holds only while no instruction here breaks it.
            unsafe {
                core::arch::asm!(
                    ".byte 0x0f, 0x01, 0xfa",
                    in("rax") addr,
                    in("ecx") 0u32,
                    in("edx") 0u32,
                    options(nostack, preserves_flags),
                );
            }

            // An unpark between the caller's snapshot and the arming
            // above would not wake MWAITX, because the monitor was not
            // yet armed. Checked after every arm, not just the first,
            // since each iteration re-opens the window.
            if self.wake_counter.load(Ordering::Acquire) != initial_wake {
                return;
            }

            let spent = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(start);
            let Some(left) = budget.checked_sub(spent) else {
                return;
            };
            // EBX is 32 bits. A remaining budget past that asks for
            // the longest wait the register can express and the next
            // iteration asks for the rest.
            let ask = u32::try_from(left).unwrap_or(u32::MAX);

            // MWAITX: EAX = 0 requests C0, the light state matching
            // the WAITPKG arm's C0.1 hint; ECX bit 1 enables the EBX
            // timer; EBX carries the count.
            //
            // rbx is reserved by LLVM and cannot be an operand, so it
            // is saved and restored inside the block. That is why this
            // block does not claim `nostack`: it pushes, and a red
            // zone below rsp would be live under that promise.
            //
            // SAFETY: same MONITORX-available reasoning as above.
            // MWAITX reports timer-versus-wake exit in CF, so flags
            // are not preserved.
            unsafe {
                core::arch::asm!(
                    "push rbx",
                    "mov ebx, {ask:e}",
                    ".byte 0x0f, 0x01, 0xfb",
                    "pop rbx",
                    ask = in(reg) ask,
                    inout("eax") 0u32 => _,
                    inout("ecx") 2u32 => _,
                );
            }

            if self.wake_counter.load(Ordering::Acquire) != initial_wake {
                return;
            }
            if self.shutdown.load(Ordering::Acquire) {
                return;
            }

            // Arms and clock together, not the length of any single
            // return: an interrupt lengthens some returns, so a rule
            // over consecutive short ones never fires on a host whose
            // monitor is not holding.
            // The arm count gates the clock read, not the other way
            // round: the first arms cannot trip this and a clock read
            // costs 206 ns on the Windows host here and 973 on the
            // Linux guest.
            if arms >= ARMS_BEFORE_JUDGING {
                let span = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(start);
                if span < ARMS_TOO_FAST_CYCLES {
                    note_monitor_does_not_hold(span);
                    thread::park();
                    return;
                }
            }
        }
    }

    /// Non-x86_64 stub, for the same reason as the WAITPKG one: the
    /// CPUID probe cannot report MONITORX off x86_64, so this strategy
    /// is never installed there.
    #[cfg(not(target_arch = "x86_64"))]
    fn wait_via_monitorx(&self, _initial_wake: u64) {
        thread::park();
    }

    /// Test whether shutdown has been signalled. Workers can poll
    /// this between job executions to exit promptly.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn park_until_returns_immediately_when_ready() {
        let p = Parker::new(8);
        let t0 = Instant::now();
        let ok = p.park_until(|| true);
        let elapsed = t0.elapsed();
        assert!(ok);
        assert!(elapsed < Duration::from_millis(10),
            "park_until with ready=true must be fast; took {elapsed:?}");
    }

    #[test]
    fn park_until_returns_false_on_shutdown() {
        let p = Arc::new(Parker::new(8));
        let p_signal = Arc::clone(&p);
        let signal = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            p_signal.shutdown();
        });
        let ok = p.park_until(|| false);
        signal.join().unwrap();
        assert!(!ok, "park_until must return false after shutdown");
    }

    #[test]
    fn park_until_wakes_on_unpark_from_other_thread() {
        // Owner thread parks. Helper thread unparks after 50 ms.
        // The owner observes `is_ready` becoming true and returns.
        let ready = Arc::new(AtomicU32::new(0));
        let ready_clone = Arc::clone(&ready);

        let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();

        let owner = thread::spawn(move || {
            let p = Arc::new(Parker::new(8));
            tx.send(Arc::clone(&p)).unwrap();
            let t0 = Instant::now();
            let ok = p.park_until(|| ready.load(Ordering::Acquire) == 1);
            (ok, t0.elapsed())
        });

        let p_owner = rx.recv().expect("owner must send its parker");
        thread::sleep(Duration::from_millis(50));
        ready_clone.store(1, Ordering::Release);
        p_owner.unpark();

        let (ok, elapsed) = owner.join().unwrap();
        assert!(ok, "park_until must return true after ready becomes true");
        // Should wake within ~100 ms.
        assert!(elapsed < Duration::from_millis(500),
            "park_until took too long: {elapsed:?}");
    }

    #[test]
    fn park_until_spin_floor_eight_rounds_no_park() {
        // With spin_rounds=8 and is_ready becoming true on round 3,
        // park_until should return without ever calling park().
        // We can't observe park directly, but we can verify the
        // sequence completes quickly.
        let p = Parker::new(8);
        let mut polls = 0u32;
        let ok = p.park_until(|| {
            polls += 1;
            polls >= 3
        });
        assert!(ok);
        assert_eq!(polls, 3);
    }

    #[test]
    fn park_until_zero_spin_floor_goes_straight_to_park() {
        // With spin_rounds=0, park_until skips the yield loop. We
        // verify by setting ready=true synchronously - the call
        // returns at the loop's first iteration.
        let p = Parker::new(0);
        let ok = p.park_until(|| true);
        assert!(ok);
    }

    #[test]
    fn unpark_before_park_is_observable_via_permit() {
        // std::thread::park's permit semantics: unpark before park
        // stores a permit; next park returns immediately. We test
        // this through park_until: helper unparks before owner
        // calls park_until. The owner's first park sees the
        // permit and returns; the subsequent re-check observes
        // ready=true.
        let p = Arc::new(Parker::new(0)); // 0 spin so we go to park fast
        let ready = Arc::new(AtomicU32::new(0));
        let p_clone = Arc::clone(&p);
        let ready_clone = Arc::clone(&ready);

        // Pre-store an unpark permit on the owner thread before
        // it calls park_until. We do this by having the owner be
        // the main thread, and a helper that unparks then sets
        // ready.
        // (Easier: just stage the unpark via a delayed thread
        //  before the owner's park_until call.)

        let signal = thread::spawn(move || {
            // Caller's thread::current() is captured inside p_clone
            // when the main thread instantiates Parker. The unpark
            // targets the main thread (the parker's owner).
            ready_clone.store(1, Ordering::Release);
            p_clone.unpark();
        });
        signal.join().unwrap();

        let ok = p.park_until(|| ready.load(Ordering::Acquire) == 1);
        assert!(ok);
    }

    #[test]
    fn is_shutdown_reflects_shutdown_call() {
        let p = Parker::new(8);
        assert!(!p.is_shutdown());
        p.shutdown();
        assert!(p.is_shutdown());
    }

    #[test]
    fn the_baseline_is_what_the_silicon_supports_and_nothing_learned() {
        // baseline() is the floor a measurement has to beat, so it
        // reads CPUID and nothing else. pick() may differ from it
        // once the controller has evidence, which is the whole point,
        // and is asserted separately.
        let want = if crate::cpu_info::has_waitpkg() {
            WaitStrategy::Waitpkg
        } else {
            WaitStrategy::StdPark
        };
        assert_eq!(WaitStrategy::baseline(), want);
    }

    /// A controller of its own, so a test can drive the decision
    /// without moving the one the process parks on.
    fn fresh_controller() -> WaitController {
        WaitController {
            mean_ns: [AtomicU64::new(0), AtomicU64::new(0)],
            samples: [AtomicU32::new(0), AtomicU32::new(0)],
            parks: AtomicU64::new(0),
            current: AtomicU32::new(SLOT_BASELINE),
            switches: AtomicU64::new(0),
            probe_every: AtomicU32::new(PROBE_EVERY_MIN),
            score: AtomicI32::new(0),
        }
    }

    #[test]
    fn evidence_short_of_the_gate_decides_nothing() {
        // One arm looking wonderful over three samples must not move
        // a process off the wait it shipped with. A wake latency has
        // a long tail and a handful of samples is mostly tail.
        let c = fresh_controller();
        for _ in 0..3 {
            WaitController::open_pair();
            c.record(SLOT_CHALLENGER, 100);
            c.record(SLOT_BASELINE, 100_000);
        }
        assert_eq!(c.current.load(Ordering::Relaxed), SLOT_BASELINE);
        assert_eq!(c.switches.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_clear_win_moves_the_verdict_and_a_clear_loss_moves_it_back() {
        // The whole point: the choice follows the measurement in both
        // directions. A controller that could only adopt would be the
        // permanent verdict again, wearing a mean.
        let c = fresh_controller();
        for _ in 0..SWITCH_AT {
            WaitController::open_pair();
            c.record(SLOT_BASELINE, 6_000);
            c.record(SLOT_CHALLENGER, 1_000);
        }
        assert_eq!(c.current.load(Ordering::Relaxed), SLOT_CHALLENGER);

        // The host gets busy and the monitor wait stops paying. The
        // score has to be won back from wherever it saturated before
        // the process moves, which is the intent: one contrary pair
        // is not a change of character.
        for _ in 0..SCORE_CAP + SWITCH_AT {
            WaitController::open_pair();
            c.record(SLOT_CHALLENGER, 300_000);
            c.record(SLOT_BASELINE, 6_000);
        }
        assert_eq!(c.current.load(Ordering::Relaxed), SLOT_BASELINE);
        assert!(c.switches.load(Ordering::Relaxed) >= 2);
    }

    #[test]
    fn two_arms_within_the_margin_leave_the_verdict_alone() {
        // Switching on noise costs a process its exploitation and
        // buys nothing. Ten per cent apart is inside the twenty the
        // controller demands.
        let c = fresh_controller();
        for _ in 0..SCORE_CAP * 2 {
            WaitController::open_pair();
            c.record(SLOT_BASELINE, 5_000);
            c.record(SLOT_CHALLENGER, 4_600);
        }
        assert_eq!(c.current.load(Ordering::Relaxed), SLOT_BASELINE);
        assert_eq!(c.switches.load(Ordering::Relaxed), 0);
        assert_eq!(c.score.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_mean_follows_its_samples_down_as_well_as_up() {
        // Every other gate here feeds an arm one number, which a mean
        // that only rose would also satisfy, because a constant never
        // asks it to fall. An arm whose first timed park is slow and
        // whose next thirty-one are fast has to read as mostly fast.
        let c = fresh_controller();
        c.record(SLOT_CHALLENGER, 300_000);
        assert_eq!(
            c.mean_ns[SLOT_CHALLENGER as usize].load(Ordering::Relaxed),
            300_000
        );
        for _ in 0..SAMPLES_BEFORE_EWMA - 1 {
            c.record(SLOT_CHALLENGER, 1_000);
        }
        // One 300us draw among thirty-one of 1us averages near 10us.
        let m = c.mean_ns[SLOT_CHALLENGER as usize].load(Ordering::Relaxed);
        assert!(m < 20_000, "mean stayed high at {m}");
        assert!(m > 1_000, "mean dropped the slow draw at {m}");
    }

    #[test]
    fn a_win_survives_the_spread_a_real_wake_has() {
        // A wake latency arrives with a tail on both arms. The verdict
        // has to come from where the two sit against each other over
        // their samples, not from whichever drew the worst park.
        let c = fresh_controller();
        let base = [5_000u64, 6_500, 5_200, 40_000, 5_800, 6_100];
        let chal = [1_100u64, 1_400, 1_200, 30_000, 1_300, 1_250];
        for i in 0..(SWITCH_AT as usize) {
            WaitController::open_pair();
            c.record(SLOT_BASELINE, base[i % base.len()]);
            c.record(SLOT_CHALLENGER, chal[i % chal.len()]);
        }
        assert_eq!(c.current.load(Ordering::Relaxed), SLOT_CHALLENGER);
    }

    #[test]
    fn a_rare_enormous_draw_does_not_decide_against_an_arm() {
        // Measured on a 7900X: the controller read its challenger at
        // 17939 ns over 100 samples while a pinned arm measured the
        // same wait at 888 ns under the same conditions, and it
        // declined a wait nearly seven times cheaper. A handful of
        // draws near 340 us carried that average, which is what a
        // mean does with a long tail.
        //
        // The pair is what answers it: the challenger loses the rare
        // pair it stalls in and wins every other, and the count of
        // those outcomes does not care how large the loss was.
        let c = fresh_controller();
        for i in 0..SWITCH_AT * 2 {
            WaitController::open_pair();
            c.record(SLOT_BASELINE, 6_000);
            // One draw in twenty stalls, far above anything the
            // baseline does.
            let chal = if i % 20 == 19 { 340_000 } else { 900 };
            c.record(SLOT_CHALLENGER, chal);
        }
        assert_eq!(c.current.load(Ordering::Relaxed), SLOT_CHALLENGER);
        // The mean it reports still carries the tail, which is why
        // it is reported and not obeyed.
        assert!(
            c.mean_ns[SLOT_CHALLENGER as usize].load(Ordering::Relaxed)
                > c.mean_ns[SLOT_BASELINE as usize].load(Ordering::Relaxed),
            "the tail should still show in the reported mean"
        );
    }

    #[test]
    fn a_settled_verdict_probes_less_often_and_a_closing_gap_probes_more() {
        // A probe spends one park on the arm not in use. Where that
        // arm is twenty times worse, probing at the close rate is a
        // sixth of every park for the life of the process, which is
        // the cost this widening exists to remove.
        let c = fresh_controller();
        for _ in 0..SCORE_CAP * 2 {
            WaitController::open_pair();
            c.record(SLOT_BASELINE, 6_000);
            c.record(SLOT_CHALLENGER, 1_000);
        }
        assert_eq!(c.current.load(Ordering::Relaxed), SLOT_CHALLENGER);
        assert_eq!(c.score.load(Ordering::Relaxed), SCORE_CAP);
        assert_eq!(c.probe_every.load(Ordering::Relaxed), PROBE_EVERY_MAX);

        // The arms converge, so the pairs stop separating and the
        // score walks back toward zero. Crossing inside the
        // switching threshold is the order coming apart, and that is
        // the point to look often again rather than after it flips.
        for _ in 0..(SCORE_CAP - SWITCH_AT) + 1 {
            WaitController::open_pair();
            c.record(SLOT_CHALLENGER, 6_000);
            c.record(SLOT_BASELINE, 4_000);
        }
        assert_eq!(c.current.load(Ordering::Relaxed), SLOT_CHALLENGER);
        assert_eq!(c.probe_every.load(Ordering::Relaxed), PROBE_EVERY_MIN);
    }

    #[test]
    fn a_cold_controller_starts_on_the_baseline() {
        // Before anything is measured the process must behave as it
        // did before the controller existed, whatever the host can
        // execute. A monitor wait is reached by evidence or not at
        // all.
        let r = wait_controller().report();
        if r.switches == 0 {
            assert_eq!(r.in_use, WaitStrategy::baseline());
        }
    }

    #[test]
    fn the_controller_never_offers_a_wait_this_host_cannot_execute() {
        // Its verdict names an instruction, so a wrong one is a `#UD`
        // on the idle path. Read from the answer toward the probe,
        // which is the direction that catches a challenger chosen for
        // a host that does not have it.
        match WaitStrategy::pick() {
            WaitStrategy::Waitpkg => assert!(crate::cpu_info::has_waitpkg()),
            WaitStrategy::Monitorx => assert!(crate::cpu_info::has_monitorx()),
            WaitStrategy::StdPark => {}
        }
    }

    #[test]
    fn a_challenger_is_only_offered_where_it_could_win() {
        // No monitor wait, or one already found not to hold, leaves
        // nothing to race and the controller must not spend sampled
        // parks exploring it.
        match challenger() {
            Some(c) => {
                assert_eq!(c, WaitStrategy::Monitorx);
                assert!(crate::cpu_info::has_monitorx());
                assert!(monitor_wait_held());
            }
            None => assert!(
                !crate::cpu_info::has_monitorx()
                    || crate::cpu_info::has_waitpkg()
                    || !monitor_wait_held()
            ),
        }
    }

    #[test]
    fn a_pinned_parker_ignores_the_controller() {
        // The benches hold one arm still to measure it. A pinned
        // parker that drifted onto the controller's verdict would
        // measure whatever the controller had decided, which is the
        // thing the bench is trying to inform.
        let p = Parker::with_strategy(0, WaitStrategy::StdPark);
        assert_eq!(p.wait_strategy(), WaitStrategy::StdPark);

        if crate::cpu_info::has_monitorx() && !crate::cpu_info::has_waitpkg() {
            let m = Parker::with_strategy(0, WaitStrategy::Monitorx);
            assert_eq!(m.wait_strategy(), WaitStrategy::Monitorx);
        }
    }

    #[test]
    fn no_two_parkers_can_share_a_cache_line() {
        // A monitor wait watches the line wake_counter sits in, so a
        // neighbour sharing it turns that neighbour's every unpark
        // into a wake here. Pinned by alignment rather than by size,
        // because adding a field must not be able to undo it.
        assert_eq!(std::mem::align_of::<Parker>(), 64);
        assert_eq!(std::mem::size_of::<Parker>() % 64, 0);
    }

    #[test]
    fn the_parker_never_picks_a_wait_its_host_cannot_execute() {
        // Asserted through a Parker as well as through pick(),
        // because the parker is what actually executes the
        // instruction and it has its own pinned path.
        match Parker::new(0).wait_strategy() {
            WaitStrategy::Waitpkg => assert!(crate::cpu_info::has_waitpkg()),
            WaitStrategy::Monitorx => assert!(crate::cpu_info::has_monitorx()),
            WaitStrategy::StdPark => assert!(!crate::cpu_info::has_waitpkg()),
        }
    }

    #[test]
    fn the_hosts_own_strategy_wakes_on_unpark() {
        // The other wake tests name a strategy, so on a host whose
        // pick() differs from all of them the arm that actually runs
        // in production goes unexercised. This one parks on whatever
        // this host chose, which is the only test here that executes
        // the MONITORX path on a MONITORX host.
        // Constructed on the thread that parks, and handed back over
        // a channel. A Parker captures `thread::current()` at
        // construction and unparks that handle, so one built here and
        // parked over there sends the permit to this thread and the
        // parked one never wakes. The StdPark arm is where that
        // shows: a monitor wait returns on the cache-line store and
        // never reads the handle at all.
        let ready = Arc::new(AtomicU32::new(0));
        let woke = Arc::new(AtomicU32::new(0));
        let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();

        let ready_thread = Arc::clone(&ready);
        let woke_thread = Arc::clone(&woke);
        let owner = thread::spawn(move || {
            let p = Arc::new(Parker::new(0));
            tx.send(Arc::clone(&p)).expect("send parker");
            let ok = p.park_until(|| ready_thread.load(Ordering::Acquire) == 1);
            woke_thread.store(1, Ordering::Release);
            ok
        });

        let p = rx.recv().expect("recv parker");
        thread::sleep(Duration::from_millis(20));
        ready.store(1, Ordering::Release);
        p.unpark();

        assert!(
            owner.join().expect("owner thread joins"),
            "park_until must report a wake rather than a shutdown"
        );
        assert_eq!(woke.load(Ordering::Acquire), 1);
    }

    #[test]
    fn with_strategy_stdpark_round_trips_like_default() {
        // Explicitly construct a StdPark Parker; same semantics as
        // the original implementation.
        let p = Parker::with_strategy(8, WaitStrategy::StdPark);
        assert_eq!(p.wait_strategy(), WaitStrategy::StdPark);
        let ok = p.park_until(|| true);
        assert!(ok, "StdPark park_until must return true on ready=true");
    }

    #[test]
    fn unpark_increments_wake_counter() {
        // Verify the WAITPKG observer mechanism is wired: every
        // unpark MUST bump wake_counter so the WAITPKG path's
        // double-check after UMONITOR catches the wake.
        let p = Parker::new(8);
        let before = p.wake_counter.load(Ordering::Acquire);
        p.unpark();
        let after = p.wake_counter.load(Ordering::Acquire);
        assert_eq!(after, before + 1, "unpark must increment wake_counter");
    }

    #[test]
    fn shutdown_increments_wake_counter() {
        // shutdown routes through unpark so the WAITPKG observer
        // also wakes on shutdown (not just the std::thread permit).
        let p = Parker::new(8);
        let before = p.wake_counter.load(Ordering::Acquire);
        p.shutdown();
        let after = p.wake_counter.load(Ordering::Acquire);
        assert_eq!(after, before + 1, "shutdown must increment wake_counter via unpark");
        assert!(p.is_shutdown());
    }

    #[test]
    fn waitpkg_strategy_wakes_on_unpark_when_available() {
        // Skip on hosts without WAITPKG (the UMONITOR/UMWAIT opcodes
        // would #UD). Per-architecture cpuid check; on Zen+ R7 2700
        // this returns false and the test is a no-op.
        if !crate::cpu_info::has_waitpkg() {
            eprintln!(
                "skip waitpkg_strategy_wakes_on_unpark_when_available: \
                 host has no WAITPKG (cpuid leaf 7 ECX bit 5 = 0)"
            );
            return;
        }
        // WAITPKG-capable host: park with Waitpkg strategy + unpark
        // from helper thread. Owner thread must observe the wake
        // within the 10ms deadline.
        let ready = Arc::new(AtomicU32::new(0));
        let ready_clone = Arc::clone(&ready);
        let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();
        let owner = thread::spawn(move || {
            let p = Arc::new(Parker::with_strategy(8, WaitStrategy::Waitpkg));
            tx.send(Arc::clone(&p)).unwrap();
            let t0 = Instant::now();
            let ok = p.park_until(|| ready.load(Ordering::Acquire) == 1);
            (ok, t0.elapsed())
        });
        let p_owner = rx.recv().expect("owner must send its parker");
        thread::sleep(Duration::from_millis(20));
        ready_clone.store(1, Ordering::Release);
        p_owner.unpark();
        let (ok, elapsed) = owner.join().unwrap();
        assert!(ok, "Waitpkg park_until must return true on unpark");
        // Cap should be well under 100ms; the 10ms UMWAIT deadline
        // bounds the worst case to ~10ms even if UMWAIT misses the
        // wake.
        assert!(elapsed < Duration::from_millis(100),
            "Waitpkg park_until took {elapsed:?}, expected < 100ms");
    }

    #[test]
    fn shutdown_unparks_so_blocked_thread_exits() {
        // Parker MUST be constructed inside the thread that will
        // park on it, because `Parker::new` captures
        // `thread::current()` for the unpark target. A Parker
        // built in main and parked-on by a spawned thread would
        // unpark main, not the spawned thread, and deadlock.
        let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();
        let owner = thread::spawn(move || {
            let p = Arc::new(Parker::new(8));
            tx.send(Arc::clone(&p)).unwrap();
            p.park_until(|| false)
        });
        let p_owner = rx.recv().expect("owner must send its parker");
        thread::sleep(Duration::from_millis(50));
        p_owner.shutdown();
        let ok = owner.join().unwrap();
        assert!(!ok, "shutdown must surface as park_until -> false");
    }
}

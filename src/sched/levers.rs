//! Runtime switches for the noisy-host mechanisms.
//!
//! Every mechanism here has to earn its place against the same code
//! without it: never slower on a quiet host, faster on a contended one.
//! A switch is what makes that measurable. One binary runs both arms, so
//! the comparison carries no difference in commit, harness, build or
//! machine - only the mechanism.
//!
//! # Why a default is off
//!
//! Off is the behavior that shipped. A mechanism defaults on only when a
//! measurement says it should, and until then the crate behaves exactly
//! as it did, so carrying the code costs nothing but the branch it is on.
//! A switch whose default flips on the strength of an argument rather
//! than a reading is how the thing this campaign is fixing got in.
//!
//! Three have earned it: [`calibration_refusal`], [`oncore_spread`] and
//! [`allowed_width`] default on, and the reading is on each function.
//! The rest are off and say what would have to be measured for that to
//! change.
//!
//! # Why environment variables rather than features
//!
//! A cargo feature is chosen at build time, so an A/B across one needs
//! two binaries, and two binaries differ in more than the feature. These
//! are read once per process and cached, so a switch costs one relaxed
//! load at a call site and the two arms are the same bytes on disk.
//!
//! # What a switch must not do
//!
//! Change anything but the mechanism it names. A switch that also moves
//! a threshold, or that turns two mechanisms on together, makes its own
//! A/B unreadable and there is no way to tell from the result.

use std::sync::OnceLock;

/// Read a switch once. Absent, empty, `0`, `off` and `false` are off;
/// anything else is on.
///
/// Anything-else-is-on rather than only `1`: a caller who writes `yes`
/// or `true` meant on, and a switch that silently ignored them would run
/// the arm they did not ask for while printing nothing. The values that
/// mean off are the ones a script produces when it means off.
fn read(name: &str) -> bool {
    match std::env::var(name) {
        Ok(text) => {
            let v = text.trim().to_ascii_lowercase();
            !(v.is_empty() || v == "0" || v == "off" || v == "false")
        }
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(raw)) => {
            // Set to something unreadable. Reporting it matters because
            // the alternative is an arm that was asked for, did not run,
            // and produced a row indistinguishable from the other arm.
            eprintln!(
                "flynnel: {name} is set to something that is not UTF-8 ({raw:?}); \
                 treating it as off, which may not be the arm you asked for"
            );
            false
        }
    }
}

/// What a stored calibration has to satisfy before it is served.
///
/// Four arms rather than one, because which of them is right has not
/// been measured and the campaign has twice adopted a statistic on an
/// argument that later data refuted.
///
/// The shape they are judged on has two halves. A process that serves a
/// record skips a draw, measured at about 15 ms on a 12-core host. It
/// also routes on that record's thresholds rather than on a fresh
/// draw's, so an arm can win on start cost and lose on routing.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ServePolicy {
    /// The shipped arm: whatever
    /// `calibration_store::CpuCalibration::is_trustworthy` admits,
    /// which is a record some other draw has agreed with. Named in
    /// backticks rather than linked, because the item is behind
    /// `persisted-calibration` and a link to it is broken in a build
    /// without that feature.
    ///
    /// A fresh stamp therefore serves from its third start: the first
    /// stores a provisional record, the second agrees with it and
    /// confirms, and the rest read.
    Spread,
    /// Anything that was actually measured. The cheapest-cost ordering
    /// decides which record stands, and load only adds time, so a draw
    /// taken on a busy host cannot win it.
    Any,
    /// The share of its interval the measuring thread held a core,
    /// against [`occupancy_floor_per_mille`]. Reads 998 to 999 on a
    /// quiet draw here and 664 to 799 on a loaded one.
    Occupancy,
    /// The spread again, against a bound the caller supplies rather
    /// than the shipped one.
    SpreadAt(u32),
}

/// Carried only where the store is, because the record it judges is
/// `calibration_store::CpuCalibration` and that module sits behind
/// `persisted-calibration`. The one caller, `stored_record_serves` in
/// `par_iter`, is gated on the same feature, so a build without it has
/// nothing to ask and nothing to answer with.
///
/// The enum itself stays ungated: it names no gated type, and
/// `serve_policy()` reads the same switch in every configuration.
#[cfg(feature = "persisted-calibration")]
impl ServePolicy {
    /// Whether this record may be served.
    ///
    /// A record with no samples is refused by every arm: a table nobody
    /// has published reads back zeroed, and zero is not a measurement.
    pub fn admits(self, cpu: &crate::sched::calibration_store::CpuCalibration) -> bool {
        if cpu.samples == 0 {
            return false;
        }
        match self {
            Self::Spread => cpu.is_trustworthy(),
            Self::Any => true,
            Self::Occupancy => match cpu.occupancy() {
                // No thread clock means no reading, and a platform that
                // cannot measure occupancy must not be frozen out of
                // its own store by an arm that needs one.
                None => true,
                Some(share) => share >= occupancy_floor_per_mille(),
            },
            Self::SpreadAt(bound) => cpu.spread_per_mille <= bound,
        }
    }
}

/// The raw text of a variable, or `None` where it is genuinely unset.
///
/// A value that is not UTF-8 is reported and read as unset, rather than
/// dropped: an arm that failed to take effect produces rows that look
/// exactly like the arm that did.
fn raw(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(text) => Some(text),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(value)) => {
            eprintln!(
                "flynnel: {name} is set to something that is not UTF-8 ({value:?}); \
                 treating it as unset, which may not be the arm you asked for"
            );
            None
        }
    }
}

/// Which arm [`stored_record_serves`] applies, from
/// `FLYNNEL_SERVE_POLICY`: `spread` (the shipped behavior and the
/// default), `any`, `occupancy`, or `spread:<per-mille>`.
///
/// An unreadable value reports itself and falls back to the shipped
/// arm, because a rotation that silently ran the default while its log
/// claimed another arm is the failure this campaign exists to stop.
///
/// [`stored_record_serves`]: crate::sched::par_iter
pub fn serve_policy() -> ServePolicy {
    static V: OnceLock<ServePolicy> = OnceLock::new();
    *V.get_or_init(|| {
        let Some(text) = raw("FLYNNEL_SERVE_POLICY") else {
            return ServePolicy::Spread;
        };
        let v = text.trim().to_ascii_lowercase();
        if let Some(rest) = v.strip_prefix("spread:") {
            return match rest.parse::<u32>() {
                Ok(bound) => ServePolicy::SpreadAt(bound),
                Err(e) => {
                    eprintln!(
                        "flynnel: FLYNNEL_SERVE_POLICY spread: wants a whole number of \
                         parts per mille and got {rest:?} ({e}); running the shipped arm"
                    );
                    ServePolicy::Spread
                }
            };
        }
        match v.as_str() {
            "" | "spread" => ServePolicy::Spread,
            "any" => ServePolicy::Any,
            "occupancy" => ServePolicy::Occupancy,
            other => {
                eprintln!(
                    "flynnel: FLYNNEL_SERVE_POLICY is {other:?}, which is not spread, any, \
                     occupancy or spread:<per-mille>; running the shipped arm"
                );
                ServePolicy::Spread
            }
        }
    })
}

/// The floor [`ServePolicy::Occupancy`] admits at, from
/// `FLYNNEL_OCCUPANCY_FLOOR_PER_MILLE`.
///
/// 900 by default, which sits between the 998 to 999 a quiet draw reads
/// on the host measured and the 664 to 799 a loaded one reads. A
/// starting point for a rotation rather than a derived value, and the
/// rotation is what it exists for.
pub fn occupancy_floor_per_mille() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        let Some(text) = raw("FLYNNEL_OCCUPANCY_FLOOR_PER_MILLE") else {
            return 900;
        };
        match text.trim().parse::<u32>() {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "flynnel: FLYNNEL_OCCUPANCY_FLOOR_PER_MILLE wants a whole number and \
                     got {text:?} ({e}); using 900"
                );
                900
            }
        }
    })
}

/// Read a switch once, absent meaning on.
///
/// The same spellings mean off as in [`read`], so a caller who turns
/// something off gets it off by any of the words a script produces. The
/// difference is only what an unset variable means, which for a
/// mechanism a measurement has already settled is that it runs.
fn read_defaulting_on(name: &str) -> bool {
    match std::env::var_os(name) {
        None => true,
        Some(_) => read(name),
    }
}

/// Take the classifier's per-item spread from the thread's own clock
/// rather than from wall time.
///
/// Wall time rises both because the work is irregular and because the
/// thread lost its core, and preemption lands on some leaves and not
/// others, so it reaches a wall-time spread as variance indistinguishable
/// from the work's own. Costs two thread-clock reads on each sampled
/// leaf, which is why it is a switch rather than simply the behavior.
///
/// Read by two recorders, which between them cover the dispatch
/// entries. `record_leaf_sampled` serves the plain steal-driven bisect,
/// where one leaf in the stride is both wall-timed and bracketed.
/// `record_leaf_bracket_sampled` serves the indexed and triple entries,
/// which time every leaf and bracket one in the same stride. The two
/// cadences differ because those entries need a reading from every leaf
/// to converge, and that requirement is about the wall clock alone.
///
/// [`crate::sched::call_site::CallSiteState::oncore_items`] stays at
/// zero when no leaf was bracketed, so it reports whether this switch
/// reached the dispatch under measurement rather than leaving a switch
/// that never engaged to look like one that did not help.
/// On unless the variable turns it off. The bracket costs nothing
/// measurable: on a 24-thread host, three workload shapes each over
/// forty paired trials read 1.0033, 1.0000 and 1.0019 against the same
/// code with it off, at bounds of 0.64, 0.22 and 0.31 per cent with
/// controls resolving to 0.19 and tighter.
/// Measured again on three hosts, paired by trial, at 4096 reps of
/// uniform work: 0.9978 at a 0.22 per cent bound on a 24-thread Windows
/// bare-metal box, 32 clean pairs of 40, control 0.9988 at 0.38;
/// 0.9922 at 4.79 on a 16-core Linux guest, 0.9917 at 11.15 on a
/// 16-core FreeBSD guest. Not slower on any, no speed-up claimed on any.
///
/// The bare-metal cell was read twice. An earlier rotation gave 0.9964
/// at 1.90 per cent over 7 pairs of 40, taken while an unrelated process
/// held a core continuously: the box's idle floor was 1.81 cores against
/// the harness's 1.4-core gate, so most trials were dropped for a
/// condition none of them caused. The figures above are the repeat on a
/// quiet box, and the difference between the two bounds is the floor
/// rather than the lever.
///
/// The bracket is taken in all three - oncore_items reads zero on the
/// arm without the lever and millions on the arm with it - and on the
/// bare-metal run no class moves, so 1.90 per cent is what the
/// instrumentation costs when nothing downstream changes.
///
/// A correctness lever, judged on the class it produces rather than on
/// throughput. On-core timing is the thread clock and excludes
/// descheduled time, so what it changes in the classifier's input is
/// the descheduling there is: the learned class differs between the
/// arms in 29 trials of 40 on a Linux guest, 6 of 40 on a FreeBSD
/// guest, and 1 of 40 on bare metal. A host that does not deschedule
/// has nothing here to correct.
///
/// Its price on a guest, at an 8 second window over 40 trials: 0.9814
/// at a 2.52 per cent bound where the class moves, 0.9872 at 4.34 where
/// it holds. About the same either way. On quiet bare metal, where no
/// class moves at this size, 0.9978 at 0.22 over 32 pairs.
pub fn oncore_spread() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read_defaulting_on("FLYNNEL_LEVER_ONCORE_SPREAD"))
}

/// Weight a leaf batch by the share of its interval the pool spent on a
/// core, so a contended batch counts for less.
pub fn batch_weight() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_BATCH_WEIGHT"))
}

/// Give each thread its own wasmtime store per kernel, instantiated on
/// that thread's first dispatch of it, instead of sharing one store
/// behind a lock.
///
/// Off until measured. A wasmtime store is not `Sync` and its API
/// takes a mutable store, so two dispatches of one kernel on the
/// shared store take turns; that lock is the last one the crate has
/// that an external type forces. A store per thread removes the
/// sharing rather than the waiting, which is why it can remove the
/// lock at all, and it moves instantiation out of registration and
/// into each thread's first dispatch. An arm that measures only the
/// steady state reports a win a short-lived dispatch never sees, so
/// the instantiation has to be inside the measured span.
pub fn wasm_local_store() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_WASM_LOCAL_STORE"))
}

/// Wait for a foreign caller's latch through a spin floor, then bounded
/// monitor waits on the latch flag's own line, then the park, instead
/// of spinning the caller's whole window on `PAUSE`.
///
/// Off until measured. This is the one placement the monitor wait is
/// built for and the pool's idle search is not: the event waited on is
/// a single store to a known line, the waiter has nothing else it
/// could be doing, and the alternative it replaces is a spin rather
/// than a yield or a park. A monitor wait ends on the same store a
/// spin would, issues nothing into the pipeline while it waits, and
/// holds the logical processor no longer than the spin it replaces.
pub fn latch_monitor() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_LATCH_MONITOR"))
}

/// Serve a mailbox push with the generic wake and leave a worker's own
/// mailbox out of the predicate it checks before parking, as the
/// scheduler did before the single-consumer wake was added.
///
/// On reinstates a hang. A mailbox is drained only by its owner, and
/// the generic wake walks from worker zero and claims whoever is
/// parked, so a wide cooperative fan-out strands work in the mailboxes
/// of workers nothing woke: 11 of 12 runs of simc_cooperative_n1024
/// stop with fifteen of sixteen mailboxes still holding jobs. This is
/// not a fallback and nothing should run with it set.
///
/// It exists so the fix can be priced in one binary. The two
/// behaviours differ by a few instructions on the park and mailbox-push
/// paths, and comparing two builds instead would confound that
/// difference with code layout, which on this codebase has moved a
/// branchy cell by half. Off is the shipped behaviour, as for every
/// other lever here.
pub fn mailbox_wake_legacy() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_MAILBOX_WAKE_LEGACY"))
}

/// Give the outside caller's slot-wait parker no yield rounds, so a
/// caller whose spin budget is spent goes from the sleep handshake
/// straight to the park.
///
/// A yield while the process holds more runnable threads than cores
/// hands the core to a ready thread for the rest of that thread's time
/// slice, so a latch set during the yield rounds is seen late. On pc2 at
/// 13be54b, with 18 spinners beside 24 workers, 97 of 104 slow calls
/// ended their wait inside those rounds, a median 12.6 ms after their
/// job ended, and the callers that had reached the park woke in 2 to
/// 7 us.
///
/// Off: measured to change nothing. In four arms of the oversubscribed
/// caller, on pc2 at faa0d05 and on a 16-vCPU guest at 517dba9, this arm
/// sat with the arm that had no switch on, quiet and loaded, and adding
/// it to [`join_park`] moved nothing that arm had not.
pub fn slot_park_now() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_SLOT_PARK_NOW"))
}

/// Park a worker whose stolen right half is still running, once its
/// spin budget is spent and it finds nothing to steal, in a kernel wait
/// that the thief ends by setting the half's latch, instead of yielding
/// each round.
///
/// The same cause as [`slot_park_now`], inside the join: on pc2 at
/// 13be54b every slow call whose join ran on for a millisecond or more
/// after its last leaf had a waiter whose last yield covered most of
/// that time after its latch was set.
///
/// Off, and measured not to be never-slower. On pc2 at faa0d05, with 18
/// spinners beside 24 workers, it took a loaded dispatch's median from
/// 3.02 to 0.95 ms and its p99 from 31.3 to 1.8 ms, at no quiet cost.
/// On a 16-vCPU guest at 517dba9 it raised the quiet median from about
/// 1.3 to 3.1 ms in each of three rounds, with the serial control level
/// across the arms: a parked thread halts its vCPU, and the wake through
/// the hypervisor costs far more than the yield it replaced.
/// [`join_park_oversubscribed`] parks only where a yield would lose the
/// core.
pub fn join_park() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_JOIN_PARK"))
}

/// Park a join waiter whose spin budget is spent in the kernel, as
/// [`join_park`] does, but only while a yield somewhere in the process
/// has given its core away within the last
/// [`crate::sched::oversubscription::WINDOW`]; otherwise yield as the
/// shipped code does.
///
/// A yield loses the core only when another thread is ready to take it,
/// and when it does, it comes back a time slice later rather than in a
/// microsecond. So the pool's idle rounds, which yield with no latch
/// pending, time their yields while this is on, and a long one is the
/// reading. A quiet process never parks a join waiter, and a guest pays
/// no hypervisor wake for a yield that would have cost nothing. What it
/// adds while on is two clock reads around each idle yield.
///
/// Off until measured, quiet and loaded, on bare metal and on a guest.
pub fn join_park_oversubscribed() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_JOIN_PARK_OVERSUBSCRIBED"))
}

/// Spend an idle pool worker's spin rounds in a bounded monitor wait on
/// the sleep coordinator's counters word rather than in `yield_now`, on
/// a host with MONITORX or WAITPKG, so a producer's store wakes the
/// worker inside the round instead of at the scheduler's next pick. The
/// park that follows the window stays the kernel park.
///
/// Off until measured. A monitor wait is a running thread to the OS
/// for as long as it lasts, so the window bounds what it can hold: one
/// round is the host's calibrated dispatch cost, and the round count
/// is the spin window the coordinator already spends. Measured as the
/// pool park it must never be, a monitor wait held every hardware
/// thread the pool had and cost sixteen processor seconds per wall
/// second on a 24-thread host; the loaded arm of a cold-dispatch
/// measurement and the process's processor time beside it decide this
/// one.
pub fn spin_monitor() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_SPIN_MONITOR"))
}

/// Decide SMT from the window the site's classifier last read rather
/// than from its lifetime cv^2, which never decays.
///
/// Engaged and not slower on three hosts, paired by trial against the
/// same code with it off: a 24-thread Windows bare-metal box reads
/// 1.0012 at a 0.61 per cent bound over ten pairs, a 16-core Linux
/// guest sits inside its null at all three loads, and a 16-core FreeBSD
/// guest reads 1.0639 at 9.37 per cent over forty. The arms differ on
/// every host - off resolves SMT true, on resolves it false - and no
/// speed-up is claimed anywhere.
///
/// The bound tracks the host rather than the method. The guest controls
/// span 0.33 to 1.32 where bare metal spans 0.99 to 1.02, because a
/// guest cannot see its own vCPU being descheduled, so neither its
/// clocks nor its controls subtract it.
pub fn smt_from_window() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_SMT_WINDOW"))
}

/// Cap a plan's worker count by the CPUs the process may currently use,
/// re-read on a cadence, so a narrowed affinity mask or cgroup quota
/// reaches the sizing.
///
/// On unless the variable turns it off. What it buys is a host that
/// narrows after the pool is spawned: `tests/affinity_follows_process_mask`
/// holds that the plan's width follows the mask down and back on all
/// three platforms. Its price is the re-read, one affinity query every
/// 250 ms, paid for nothing on a host whose mask never changes. Measured
/// twice, paired by trial on a 24-thread Windows bare-metal box at 4096
/// reps of uniform work, 40 trials each, no decision moving between the
/// arms: 1.0006 at a 0.13 per cent bound over 40 clean pairs, retained
/// 1.0005 at 0.17, control 1.0000 at 0.10, on the code that ships; and
/// 1.0000 at 0.15 over 23 clean pairs, retained 1.0014 at 0.21, control
/// 1.0000 at 0.30, on a tree whose Windows probe was still
/// `available_parallelism`.
pub fn allowed_width() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read_defaulting_on("FLYNNEL_LEVER_ALLOWED_WIDTH"))
}

/// Decline to displace a stored calibration whose dispatch cost is
/// cheaper than the one being offered.
///
/// One of the three switches here that default on, with `oncore_spread`
/// and `allowed_width`. The rest default off because off is the behavior
/// that shipped and a measurement has to earn the flip; this one has the
/// measurement.
/// Across 16 draws at each of three
/// load levels on a 12-core host, the dispatch cost read 1300 to 1500 ns
/// idle and 3.2 to 7.0 million saturated, with no overlap, so the
/// cheaper record is the quieter draw and the ordering needs no
/// threshold.
///
/// Off restores the previous behavior, where the most recent draw wins
/// and a calibration taken while the host was busy stands until
/// something displaces it.
pub fn calibration_refusal() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read_defaulting_on("FLYNNEL_LEVER_CALIBRATION_REFUSAL"))
}

/// Every switch and its state, for a harness to print beside its result.
///
/// A row that does not say which arm produced it is a row that cannot be
/// checked, and an arm that failed to engage looks exactly like one that
/// engaged and did nothing.
///
/// The spin pair lives in [`crate::sched::jec_sleep`] rather than here,
/// and is reported anyway: it is a runtime switch that changes how the
/// pool parks, so a row without it does not say which arm produced it.
/// `spin_window` is reported beside `spin_adaptive` because the window
/// alone cannot say whether the controller ran - a rescue-dominated
/// workload grows it and is clamped back to the tuned default it
/// started from.
pub fn describe() -> String {
    format!(
        "oncore_spread={} batch_weight={} smt_window={} allowed_width={} \
         calibration_refusal={} latch_monitor={} spin_monitor={} join_park={} \
         join_park_oversubscribed={} slot_park_now={} spin_adaptive={} spin_window={} \
         serve_policy={:?} occupancy_floor={}",
        oncore_spread(),
        batch_weight(),
        smt_from_window(),
        allowed_width(),
        calibration_refusal(),
        latch_monitor(),
        spin_monitor(),
        join_park(),
        join_park_oversubscribed(),
        slot_park_now(),
        crate::sched::spin_adaptive(),
        crate::sched::spin_window(),
        serve_policy(),
        occupancy_floor_per_mille(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_values_that_mean_off_are_the_ones_a_script_writes() {
        // Read directly rather than through the cached accessors, which
        // would fix whatever the first test in the process observed.
        for off in ["", "0", "off", "false", "OFF", "False", " 0 "] {
            unsafe { std::env::set_var("FLYNNEL_LEVER_TEST", off) };
            assert!(!read("FLYNNEL_LEVER_TEST"), "{off:?} should read as off");
        }
        for on in ["1", "on", "true", "yes", "ON"] {
            unsafe { std::env::set_var("FLYNNEL_LEVER_TEST", on) };
            assert!(read("FLYNNEL_LEVER_TEST"), "{on:?} should read as on");
        }
        unsafe { std::env::remove_var("FLYNNEL_LEVER_TEST") };
        assert!(!read("FLYNNEL_LEVER_TEST"), "an unset switch is off");
    }

    #[test]
    fn an_unset_switch_takes_the_default_of_the_reader_it_uses() {
        // Both readers, not the accessors: each accessor caches in a
        // OnceLock, so calling one here fixes its value for every later
        // test in this binary.
        //
        // Named by reader rather than by lever because the levers do not
        // share a default. oncore_spread, calibration_refusal and
        // allowed_width take read_defaulting_on; batch_weight and
        // smt_from_window take read.
        unsafe { std::env::remove_var("FLYNNEL_LEVER_DEFAULT_TEST") };
        assert!(
            !read("FLYNNEL_LEVER_DEFAULT_TEST"),
            "read must answer off for a switch nobody set"
        );
        assert!(
            read_defaulting_on("FLYNNEL_LEVER_DEFAULT_TEST"),
            "read_defaulting_on must answer on for a switch nobody set"
        );

        // And that each reader still honours an explicit off, so a
        // default-on switch can be turned off by a caller.
        unsafe { std::env::set_var("FLYNNEL_LEVER_DEFAULT_TEST", "0") };
        assert!(!read("FLYNNEL_LEVER_DEFAULT_TEST"));
        assert!(!read_defaulting_on("FLYNNEL_LEVER_DEFAULT_TEST"));
        unsafe { std::env::remove_var("FLYNNEL_LEVER_DEFAULT_TEST") };
    }
}

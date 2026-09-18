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
//! One switch has earned it. [`calibration_refusal`] defaults on, and
//! the reading is on the function. Each of the others is off and says
//! what would have to be measured for that to change.
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
    /// [`crate::sched::calibration_store::CpuCalibration::is_trustworthy`]
    /// admits, which is a record some other draw has agreed with.
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

/// Decide SMT from the window the site's classifier last read rather
/// than from its lifetime cv^2, which never decays.
pub fn smt_from_window() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_SMT_WINDOW"))
}

/// Cap a plan's worker count by the CPUs the process may currently use,
/// re-read on a cadence, so a narrowed affinity mask or cgroup quota
/// reaches the sizing.
pub fn allowed_width() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_ALLOWED_WIDTH"))
}

/// Decline to displace a stored calibration whose dispatch cost is
/// cheaper than the one being offered.
///
/// The one switch here that defaults on. The others default off because
/// off is the behavior that shipped and a measurement has to earn the
/// flip; this one has the measurement. Across 16 draws at each of three
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
pub fn describe() -> String {
    format!(
        "oncore_spread={} batch_weight={} smt_window={} allowed_width={} \
         calibration_refusal={} serve_policy={:?} occupancy_floor={}",
        oncore_spread(),
        batch_weight(),
        smt_from_window(),
        allowed_width(),
        calibration_refusal(),
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
    fn every_lever_is_off_unless_asked_for() {
        // The default is what ships, so it is worth an assertion rather
        // than a comment: a switch that defaulted on would change the
        // crate for every consumer who never set it.
        for name in [
            "FLYNNEL_LEVER_ONCORE_SPREAD",
            "FLYNNEL_LEVER_BATCH_WEIGHT",
            "FLYNNEL_LEVER_SMT_WINDOW",
            "FLYNNEL_LEVER_ALLOWED_WIDTH",
            "FLYNNEL_LEVER_CALIBRATION_REFUSAL",
        ] {
            assert!(
                std::env::var_os(name).is_some() || !read(name),
                "{name} must be off when unset"
            );
        }
    }
}

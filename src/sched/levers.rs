//! Runtime switches for the noisy-host mechanisms, each defaulting OFF.
//!
//! Every mechanism here has to earn its place against the same code
//! without it: never slower on a quiet host, faster on a contended one.
//! A switch is what makes that measurable. One binary runs both arms, so
//! the comparison carries no difference in commit, harness, build or
//! machine - only the mechanism.
//!
//! # Why the default is off
//!
//! Off is the behavior that shipped. A mechanism defaults on only when a
//! measurement says it should, and until then the crate behaves exactly
//! as it did, so carrying the code costs nothing but the branch it is on.
//! A switch whose default flips on the strength of an argument rather
//! than a reading is how the thing this campaign is fixing got in.
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

/// Take the classifier's per-item spread from the thread's own clock
/// rather than from wall time.
///
/// Wall time rises both because the work is irregular and because the
/// thread lost its core, and preemption lands on some leaves and not
/// others, so it reaches a wall-time spread as variance indistinguishable
/// from the work's own. Costs two thread-clock reads on each sampled
/// leaf, which is why it is a switch rather than simply the behavior.
///
/// Read only by `record_leaf_sampled`, which only the plain
/// steal-driven bisect calls. A dispatch through an indexed or triple
/// entry times every leaf with a recorder that takes no bracket, so
/// this switch changes nothing there;
/// [`crate::sched::call_site::CallSiteState::oncore_items`] stays at
/// zero when no leaf was bracketed.
pub fn oncore_spread() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_ONCORE_SPREAD"))
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

/// Decline to displace a stored calibration drawn on a quieter host.
pub fn calibration_refusal() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| read("FLYNNEL_LEVER_CALIBRATION_REFUSAL"))
}

/// Every switch and its state, for a harness to print beside its result.
///
/// A row that does not say which arm produced it is a row that cannot be
/// checked, and an arm that failed to engage looks exactly like one that
/// engaged and did nothing.
pub fn describe() -> String {
    format!(
        "oncore_spread={} batch_weight={} smt_window={} allowed_width={} \
         calibration_refusal={}",
        oncore_spread(),
        batch_weight(),
        smt_from_window(),
        allowed_width(),
        calibration_refusal(),
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

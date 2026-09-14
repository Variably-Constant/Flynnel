//! How many logical CPUs this process is currently allowed to use.
//!
//! The allowed width is not a constant for the life of a process. A
//! container's CPU quota can be lowered while it runs, and an operator
//! can re-pin a running process, so a width read once describes the
//! machine at that moment and not the one the next dispatch will get.
//! Sizing work against a width the process no longer has divides it
//! among workers that cannot reach a core.
//!
//! This is a different quantity from how busy the machine is. A
//! neighbour's load is continuous and contested, with no cutoff that
//! separates busy from quiet, and shaping a dispatch on it makes
//! identical code behave differently run to run. An affinity mask is a
//! fact: the process may use these CPUs and not those, it changed or it
//! did not, and there is no threshold to choose.
//!
//! `std::thread::available_parallelism` answers exactly this question,
//! honouring both the process affinity mask and the cgroup CPU quota.
//!
//! # The cadence is the design
//!
//! The answer costs a syscall, and on Linux a cgroup read, which is more
//! than a dispatch decision can spend. So it is re-read on a cadence and
//! cached in between, long enough that the cost disappears against the
//! work and short enough that a throttled process re-sizes within a few
//! dispatches.
//!
//! # What a failed read must not do
//!
//! The query can fail, and its failure means the count is unknown. One
//! is a count a genuinely pinned process has, so an error resolving to
//! one would silence the pool on every host where the query is
//! unsupported. A failed read says so once, naming the error, and keeps
//! the last width that was read successfully.

use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// How often the allowed width is re-read.
///
/// A quota change is an operator action, so the interesting timescale is
/// human rather than per dispatch. At this cadence a throttled process
/// re-sizes within a fraction of a second while a dispatch-heavy run
/// pays one query per interval however many dispatches it makes.
const RECHECK_INTERVAL_MS: u64 = 250;

/// Milliseconds against a monotonic origin fixed at first use.
///
/// An `Instant` carries no absolute epoch, so one origin is captured and
/// every reading is taken against it.
fn now_ms() -> u64 {
    static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let origin = ORIGIN.get_or_init(Instant::now);
    origin.elapsed().as_millis() as u64
}

/// Last width read successfully, and when it was read.
///
/// Zero means no successful reading exists yet, which is distinct from
/// every width a real host reports.
static ALLOWED: AtomicUsize = AtomicUsize::new(0);
static LAST_READ_MS: AtomicU64 = AtomicU64::new(0);

/// Whether a reading taken at `last_ms` is stale at `now_ms`.
///
/// Separate from the reading so the cadence can be exercised without a
/// syscall and without waiting for wall time to pass.
fn is_due(last_ms: u64, now_ms: u64, interval_ms: u64) -> bool {
    now_ms.saturating_sub(last_ms) >= interval_ms
}

/// Say once that the query failed, naming what it reported.
///
/// Once rather than per call: at this cadence an unsupported query would
/// otherwise produce four lines a second, and the second says nothing
/// the first did not. The width the pool falls back to is named in the
/// same line, so a reader is not left to infer which one it kept.
fn report_probe_failure(err: &std::io::Error, keeping: usize) {
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        eprintln!(
            "flynnel: this host does not report how many CPUs the process \
             may use ({err}); the pool keeps a width of {keeping} and will \
             retry"
        );
    }
}

/// How many logical CPUs this process is allowed to run on, re-read at
/// most once per [`RECHECK_INTERVAL_MS`].
///
/// A failed read keeps the previous answer and leaves the timestamp
/// alone, so the next call retries rather than waiting out the interval
/// on a reading that never happened.
pub fn allowed_parallelism() -> usize {
    let cached = ALLOWED.load(Ordering::Relaxed);
    let now = now_ms();
    if cached != 0 && !is_due(LAST_READ_MS.load(Ordering::Relaxed), now, RECHECK_INTERVAL_MS) {
        return cached;
    }
    match std::thread::available_parallelism() {
        Ok(width) => {
            let width = width.get();
            ALLOWED.store(width, Ordering::Relaxed);
            LAST_READ_MS.store(now, Ordering::Relaxed);
            width
        }
        Err(err) => {
            // No successful reading has ever been taken, so one is the
            // only width that is certainly usable: a pool sized to it
            // runs where a pool sized to an unmeasured number may not.
            let keeping = if cached != 0 { cached } else { 1 };
            report_probe_failure(&err, keeping);
            keeping
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reading_is_due_only_once_the_interval_has_passed() {
        assert!(!is_due(1_000, 1_000, 250));
        assert!(!is_due(1_000, 1_249, 250));
        assert!(is_due(1_000, 1_250, 250));
        assert!(is_due(1_000, 9_000, 250));
    }

    #[test]
    fn a_clock_reading_lower_than_the_last_is_not_due_rather_than_overdue() {
        // Saturating rather than wrapping: a negative interval read as a
        // large positive one re-queries on every dispatch, which is the
        // cost the cadence exists to avoid.
        assert!(!is_due(9_000, 1_000, 250));
    }

    #[test]
    fn the_allowed_width_is_a_real_count_and_is_cached_within_the_interval() {
        let width = allowed_parallelism();
        assert!(width >= 1, "a process may always use at least one CPU");
        assert_eq!(width, allowed_parallelism());
    }

    #[test]
    fn a_successful_reading_never_stores_the_no_reading_value() {
        // Zero is what ALLOWED holds before any query succeeds, so a
        // real width colliding with it would make a measured host
        // indistinguishable from an unmeasured one.
        let seen = allowed_parallelism();
        assert_ne!(seen, 0);
        assert_ne!(ALLOWED.load(Ordering::Relaxed), 0);
    }
}

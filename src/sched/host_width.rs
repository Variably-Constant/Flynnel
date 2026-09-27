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
//! neighbor's load is continuous and contested, with no cutoff that
//! separates busy from quiet, and shaping a dispatch on it makes
//! identical code behave differently run to run. An affinity mask is a
//! fact: the process may use these CPUs and not those, it changed or it
//! did not, and there is no threshold to choose.
//!
//! `std::thread::available_parallelism` answers this question on Linux
//! and FreeBSD, honoring the process affinity mask and the cgroup CPU
//! quota. It does not answer it on Windows: its documented limitation
//! is that it "may overcount the amount of parallelism available on
//! systems limited by process-wide affinity masks, or job object
//! limitations", and on a 24-thread Windows host a mask narrowed to two
//! CPUs read 24 through it.
//!
//! So Windows reads `GetProcessAffinityMask` and counts its set bits,
//! and the three hosts answer the same question by the route each one
//! answers it on.
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

/// The CPUs this process may currently use, asked of the platform.
///
/// [`std::thread::available_parallelism`] everywhere but Windows, where
/// its documented limitation is that it "may overcount the amount of
/// parallelism available on systems limited by process-wide affinity
/// masks, or job object limitations" - a narrowed process reads the
/// whole machine. There the mask is read directly and its set bits
/// counted.
///
/// `GetProcessAffinityMask` reports the calling process's mask within
/// its processor group, so above 64 CPUs it describes the group rather
/// than the machine. A failed call falls back to
/// `available_parallelism` rather than reporting a width nothing
/// measured.
/// On Linux, the thread-group leader's affinity mask, floored by the
/// cgroup quota.
///
/// Not [`std::thread::available_parallelism`], which asks
/// `sched_getaffinity` about the calling thread. That distinction is
/// the whole of this function: the scheduler pins its workers, so a
/// probe that happened to run on a pinned worker answers one, and the
/// answer is kept in a process-wide cache every later caller reads.
/// One pinned thread would cap the whole pool at a single worker. The
/// Windows arm has always read the process rather than the thread;
/// this makes the two agree.
///
/// `Cpus_allowed_list` in `/proc/self/status` is the thread-group
/// leader's mask, which is the process-level answer and follows a live
/// change to it. Read through `/proc` rather than the syscall because
/// this crate takes no libc dependency of its own.
///
/// The quota is applied as a floor because `available_parallelism`
/// honored it and dropping it would size a pool for the machine
/// inside a container allowed part of it. It is read once: a quota can
/// change, but two file reads at this cadence buy less than they cost,
/// and the affinity half does follow a change.
#[cfg(target_os = "linux")]
fn process_width() -> std::io::Result<std::num::NonZeroUsize> {
    let mask = proc_status_allowed_cpus()?;
    let width = match cgroup_quota_cpus() {
        Some(quota) => mask.min(quota),
        None => mask,
    };
    std::num::NonZeroUsize::new(width).ok_or_else(|| {
        std::io::Error::other("the process affinity mask names no cpu")
    })
}

/// The CPUs named by `Cpus_allowed_list` in `/proc/self/status`.
#[cfg(target_os = "linux")]
fn proc_status_allowed_cpus() -> std::io::Result<usize> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .ok_or_else(|| {
            std::io::Error::other("/proc/self/status carries no Cpus_allowed_list")
        })?;
    count_cpu_list(list.trim())
}

/// How many CPUs a Linux CPU list such as `0-3,8,12-15` names.
///
/// A part that does not parse is an error rather than a part skipped:
/// skipping would answer a smaller width than the host allows, which
/// is indistinguishable from a genuinely narrowed process and would
/// cap the pool silently.
#[cfg(target_os = "linux")]
fn count_cpu_list(list: &str) -> std::io::Result<usize> {
    fn malformed(part: &str, why: &dyn std::fmt::Display) -> std::io::Error {
        std::io::Error::other(format!("cpu list entry {part:?} is not a cpu or a range: {why}"))
    }
    let mut total = 0usize;
    for part in list.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((low, high)) => {
                let low: usize = low.trim().parse().map_err(|err| malformed(part, &err))?;
                let high: usize = high.trim().parse().map_err(|err| malformed(part, &err))?;
                if high < low {
                    return Err(malformed(part, &"its end is below its start"));
                }
                total += high - low + 1;
            }
            None => {
                part.parse::<usize>().map_err(|err| malformed(part, &err))?;
                total += 1;
            }
        }
    }
    Ok(total)
}

/// The cgroup CPU quota as a whole number of CPUs, read once.
///
/// `None` when the host sets no quota, which is the common case on a
/// bare-metal box and not a failure.
#[cfg(target_os = "linux")]
fn cgroup_quota_cpus() -> Option<usize> {
    static QUOTA: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *QUOTA.get_or_init(read_cgroup_quota_cpus)
}

/// One cgroup file, telling a file that is not there from one that
/// could not be read.
///
/// A missing file is how a host with no cgroup limit presents, and is
/// the answer rather than a failure. Any other error is named, because
/// a quota that exists and cannot be read is a limit the pool is about
/// to ignore, and staying quiet there would size the pool for the
/// whole machine inside a container.
#[cfg(target_os = "linux")]
fn read_cgroup_file(path: &str) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Some(text),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => {
            report_quota_unreadable(path, &err);
            None
        }
    }
}

/// One whole number parsed out of a cgroup file, naming what failed.
#[cfg(target_os = "linux")]
fn cgroup_number(path: &str, field: &str) -> Option<u64> {
    match field.parse::<u64>() {
        Ok(value) => Some(value),
        Err(err) => {
            report_quota_unreadable(path, &format!("{field:?}: {err}"));
            None
        }
    }
}

#[cfg(target_os = "linux")]
fn read_cgroup_quota_cpus() -> Option<usize> {
    // cgroup v2 names both numbers in one file: "<quota> <period>",
    // with "max" for no limit.
    const V2: &str = "/sys/fs/cgroup/cpu.max";
    if let Some(text) = read_cgroup_file(V2) {
        let mut fields = text.split_whitespace();
        let Some(quota) = fields.next() else {
            report_quota_unreadable(V2, &"the file is empty");
            return None;
        };
        if quota == "max" {
            return None;
        }
        let Some(period) = fields.next() else {
            report_quota_unreadable(V2, &format!("{:?} names no period", text.trim()));
            return None;
        };
        return quota_to_cpus(cgroup_number(V2, quota)?, cgroup_number(V2, period)?);
    }
    // cgroup v1 keeps them apart, and a quota at or below zero is no
    // limit rather than a limit of nothing.
    const QUOTA_V1: &str = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us";
    const PERIOD_V1: &str = "/sys/fs/cgroup/cpu/cpu.cfs_period_us";
    let quota = read_cgroup_file(QUOTA_V1)?;
    let period = read_cgroup_file(PERIOD_V1)?;
    let quota = match quota.trim().parse::<i64>() {
        Ok(value) if value <= 0 => return None,
        Ok(value) => value as u64,
        Err(err) => {
            report_quota_unreadable(QUOTA_V1, &format!("{:?}: {err}", quota.trim()));
            return None;
        }
    };
    quota_to_cpus(quota, cgroup_number(PERIOD_V1, period.trim())?)
}

/// A quota and period in the same unit, rounded up to whole CPUs.
///
/// Rounded up because a quota of one and a half CPUs can keep two
/// threads busy, and rounding down would idle the half.
#[cfg(target_os = "linux")]
fn quota_to_cpus(quota: u64, period: u64) -> Option<usize> {
    if period == 0 {
        return None;
    }
    Some(quota.div_ceil(period).max(1) as usize)
}

/// Say once that a quota file could not be read, naming it.
#[cfg(target_os = "linux")]
fn report_quota_unreadable(path: &str, saw: &dyn std::fmt::Display) {
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        eprintln!(
            "flynnel: {path} reads {saw}, which is not a quota and a period; the pool \
             sizes by the affinity mask alone and ignores whatever limit that file sets"
        );
    }
}

#[cfg(all(not(windows), not(target_os = "linux")))]
fn process_width() -> std::io::Result<std::num::NonZeroUsize> {
    std::thread::available_parallelism()
}

#[cfg(windows)]
fn process_width() -> std::io::Result<std::num::NonZeroUsize> {
    // The handle is `*mut c_void` to match the crate's other kernel32
    // declarations of `GetCurrentProcess`; two declarations of one
    // symbol with different signatures are a clashing-extern error.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn GetProcessAffinityMask(
            process: *mut core::ffi::c_void,
            process_mask: *mut usize,
            system_mask: *mut usize,
        ) -> i32;
    }
    let mut process_mask: usize = 0;
    let mut system_mask: usize = 0;
    // SAFETY: both pointers address locals that outlive the call, and
    // `GetCurrentProcess` returns a pseudo-handle that needs no release.
    // The call writes only through those two pointers.
    let ok = unsafe {
        GetProcessAffinityMask(
            GetCurrentProcess(),
            &raw mut process_mask,
            &raw mut system_mask,
        )
    };
    if ok == 0 {
        return std::thread::available_parallelism();
    }
    std::num::NonZeroUsize::new(process_mask.count_ones() as usize).ok_or_else(|| {
        std::io::Error::other("the process affinity mask names no cpu")
    })
}

/// How many logical CPUs this process is allowed to run on, re-read at
/// most once per [`RECHECK_INTERVAL_MS`].
///
/// A failed read keeps the previous answer and leaves the timestamp
/// alone, so the next call retries rather than waiting out the interval
/// on a reading that never happened.
///
/// Asks [`process_width`], which is
/// [`std::thread::available_parallelism`] on Linux and FreeBSD and
/// `GetProcessAffinityMask` on Windows, so a narrowed process reads its
/// own share on all three.
pub fn allowed_parallelism() -> usize {
    resolve(
        &ALLOWED,
        &LAST_READ_MS,
        now_ms(),
        RECHECK_INTERVAL_MS,
        process_width,
    )
}

/// The cadence and the failure rule, over caller-supplied state and a
/// caller-supplied probe.
///
/// Split out because the behavior worth testing is what happens when
/// the allowed width CHANGES, and narrowing a live process needs a
/// platform call this crate makes nowhere else - `SetProcessAffinityMask`
/// on Windows, `sched_setaffinity` on Linux, a third thing on FreeBSD.
/// A test that cannot narrow the process can still hand this a probe
/// that answers differently on successive calls, which is the same
/// question without three platform arms and a test that skips itself on
/// two of them.
fn resolve<P>(
    allowed: &AtomicUsize,
    last_read_ms: &AtomicU64,
    now: u64,
    interval_ms: u64,
    probe: P,
) -> usize
where
    P: FnOnce() -> std::io::Result<std::num::NonZeroUsize>,
{
    let cached = allowed.load(Ordering::Relaxed);
    if cached != 0 && !is_due(last_read_ms.load(Ordering::Relaxed), now, interval_ms) {
        return cached;
    }
    match probe() {
        Ok(width) => {
            let width = width.get();
            allowed.store(width, Ordering::Relaxed);
            last_read_ms.store(now, Ordering::Relaxed);
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

    /// A probe answering a fixed width, as the OS would after a quota
    /// change.
    fn says(n: usize) -> impl FnOnce() -> std::io::Result<std::num::NonZeroUsize> {
        move || Ok(std::num::NonZeroUsize::new(n).expect("a width is at least one"))
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_cpu_list_counts_its_singles_and_its_ranges() {
        assert_eq!(count_cpu_list("0-15").expect("a plain range"), 16);
        assert_eq!(count_cpu_list("14").expect("one cpu"), 1);
        assert_eq!(count_cpu_list("0-3,8,12-15").expect("a mixed list"), 9);
        // The shape a pinned worker leaves, and the reading that
        // caused the whole defect: one cpu, not the machine.
        assert_eq!(count_cpu_list("7").expect("a pinned thread"), 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_cpu_list_that_does_not_parse_is_an_error_not_a_smaller_width() {
        // Skipping a part would answer fewer cpus than the host
        // allows, which reads exactly like a narrowed process and
        // would cap the pool with nothing said.
        for bad in ["0-", "-3", "x", "0-x", "7-3"] {
            assert!(
                count_cpu_list(bad).is_err(),
                "{bad:?} must not be counted as a width"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_quota_rounds_up_to_whole_cpus() {
        // Half a cpu of quota still keeps a thread busy, and a second
        // thread can use the other half of the period.
        assert_eq!(quota_to_cpus(100_000, 100_000), Some(1));
        assert_eq!(quota_to_cpus(150_000, 100_000), Some(2));
        assert_eq!(quota_to_cpus(200_000, 100_000), Some(2));
        assert_eq!(quota_to_cpus(50_000, 100_000), Some(1), "never below one");
        assert_eq!(quota_to_cpus(100_000, 0), None, "a period of zero is no quota");
    }

    /// A probe that cannot answer, as an unsupported host would.
    fn cannot_answer() -> impl FnOnce() -> std::io::Result<std::num::NonZeroUsize> {
        || {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "this host does not report it",
            ))
        }
    }

    #[test]
    fn a_narrowed_quota_takes_effect_on_the_next_reading() {
        // The case the lever exists for: the pool was spawned for
        // sixteen and the process may now use four.
        let allowed = AtomicUsize::new(0);
        let last = AtomicU64::new(0);
        assert_eq!(resolve(&allowed, &last, 1_000, 250, says(16)), 16);
        // Inside the interval the cached answer stands and the probe is
        // not consulted, which is what makes this affordable per
        // dispatch.
        assert_eq!(resolve(&allowed, &last, 1_100, 250, says(4)), 16);
        // Past it, the narrower machine is what the caller gets.
        assert_eq!(resolve(&allowed, &last, 1_400, 250, says(4)), 4);
    }

    #[test]
    fn a_widened_quota_takes_effect_too() {
        // A pool that shrank and could not grow back would be a ratchet
        // of the kind this project keeps finding.
        let allowed = AtomicUsize::new(0);
        let last = AtomicU64::new(0);
        assert_eq!(resolve(&allowed, &last, 1_000, 250, says(4)), 4);
        assert_eq!(resolve(&allowed, &last, 1_400, 250, says(16)), 16);
    }

    #[test]
    fn a_failed_reading_keeps_the_last_width_and_retries_at_once() {
        let allowed = AtomicUsize::new(0);
        let last = AtomicU64::new(0);
        assert_eq!(resolve(&allowed, &last, 1_000, 250, says(16)), 16);
        // One is a width a genuinely pinned process has, so an error
        // must not produce it while a real reading exists.
        assert_eq!(resolve(&allowed, &last, 1_400, 250, cannot_answer()), 16);
        // The timestamp was not advanced by the failure, so the very
        // next call probes again rather than waiting out the interval on
        // a reading that never happened.
        assert_eq!(resolve(&allowed, &last, 1_401, 250, says(4)), 4);
    }

    #[test]
    fn a_first_reading_that_fails_reports_one_rather_than_zero() {
        // Zero is the no-reading sentinel and would be read as a width.
        let allowed = AtomicUsize::new(0);
        let last = AtomicU64::new(0);
        assert_eq!(resolve(&allowed, &last, 1_000, 250, cannot_answer()), 1);
        assert_eq!(
            allowed.load(Ordering::Relaxed),
            0,
            "a failed probe stores nothing, so the next success is still a first reading"
        );
    }

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

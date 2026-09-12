//! Whether this thread is getting the core it is running on.
//!
//! Every adaptive input in the scheduler is derived from wall time: the
//! calibrated dispatch cost and its two thresholds, a call site's
//! coefficient of variation, the policy-arm averages. Wall time rises
//! and spreads for two unrelated reasons - the work is expensive or
//! irregular, and the thread is not getting its core - and nothing in
//! those measurements separates them.
//!
//! That conflation is not merely noise. Preemption lands on some leaves
//! and not others, so it enters the classifier as variance: a site whose
//! leaves are uniform reads as irregular, and the class it migrates to
//! selects a different fan-out shape and a different SMT setting. A busy
//! machine therefore moves the scheduler's choice rather than only its
//! speed, and the choice persists after the load has gone.
//!
//! How much a wrong class costs is not measured. The gate rests on the
//! decision being wrong, not on a figure for what it is worth.
//!
//! The separation is available directly. A thread's own CPU time
//! advances only while it is on a core; wall time advances regardless.
//! Their ratio over the same interval is what fraction of the interval
//! this thread actually ran.
//!
//! # What this is for, and what it is not for
//!
//! It reports, and nothing consumes it. No learner refuses a window on
//! it and no dispatch is routed on it, so the scheduler behaves exactly
//! as it did without it. What a trustworthy figure is cannot be named
//! until the distribution across quiet and loaded hosts is known, and a
//! threshold chosen ahead of that describes the person who picked it.
//!
//! Routing on it is a further step and a different one: making a
//! dispatch's shape depend on what else is running would mean identical
//! code behaving differently run to run.

/// A thread's on-core ticks and the elapsed ticks they were read
/// against, both from the same clock so their ratio is a fraction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OccupancySample {
    /// Ticks this thread spent on a core.
    pub thread_ticks: u64,
    /// Ticks of elapsed time over the same interval.
    pub wall_ticks: u64,
}

impl OccupancySample {
    /// Occupancy in hundredths: 100 is a thread that ran for the whole
    /// interval, 30 is a thread that got under a third of it.
    ///
    /// Saturates at 100. A thread cannot be on a core for more of an
    /// interval than the interval, but the two counters advance for
    /// different reasons - one per executed cycle, one at a fixed rate -
    /// so a boosted core reads slightly over.
    pub fn percent(&self) -> u32 {
        if self.wall_ticks == 0 {
            return 100;
        }
        let pct = self.thread_ticks.saturating_mul(100) / self.wall_ticks;
        pct.min(100) as u32
    }
}

/// Spans a measured interval, reporting what fraction of it this thread
/// was actually on a core.
///
/// Both counters are read at construction and again at
/// [`sample`](Self::sample), so the two cover the same interval by
/// construction rather than by the caller pairing them correctly, and
/// both come from [`clock_pair`] so they carry the same unit.
#[derive(Debug)]
pub struct OccupancyWindow {
    thread_at_start: u64,
    wall_at_start: u64,
}

impl OccupancyWindow {
    /// Open a window over whatever the caller measures next.
    pub fn start() -> Self {
        let (thread, wall) = clock_pair();
        Self { thread_at_start: thread, wall_at_start: wall }
    }

    /// Close the window and report the interval.
    ///
    /// On a platform with no thread clock this reports full occupancy,
    /// so a consumer behaves exactly as it did before this existed
    /// rather than reading every interval as idle.
    pub fn sample(&self) -> OccupancySample {
        let (thread, wall) = clock_pair();
        let wall_ticks = wall.saturating_sub(self.wall_at_start);
        if !HAS_THREAD_CLOCK {
            // The fraction is unknowable here, and the two readings a
            // consumer could take from an unknowable one are not
            // symmetric: reading every interval as idle would move a
            // decision, and reading it as owned leaves the decision
            // where it sat before this module existed.
            return OccupancySample { thread_ticks: wall_ticks, wall_ticks };
        }
        OccupancySample {
            thread_ticks: thread.saturating_sub(self.thread_at_start),
            wall_ticks,
        }
    }
}

/// This thread's on-core count and an elapsed count in the same unit.
///
/// Windows reports thread time in cycles, so the elapsed side is the
/// timestamp counter rather than a nanosecond clock; dividing cycles by
/// nanoseconds yields achieved clock rate, not a fraction. The two
/// counters advance for different reasons - the thread counter per
/// cycle actually executed, the timestamp counter at a fixed rate - so
/// a core running above its base frequency reads over 1.0 and is
/// clamped. Linux reports thread time in nanoseconds, so both sides
/// there are nanoseconds and the pairing is exact.
#[cfg(all(windows, target_arch = "x86_64"))]
pub(crate) fn clock_pair() -> (u64, u64) {
    // SAFETY: `_rdtsc` reads a counter register and touches no memory.
    (thread_on_core_ticks(), unsafe { core::arch::x86_64::_rdtsc() })
}

#[cfg(not(all(windows, target_arch = "x86_64")))]
pub(crate) fn clock_pair() -> (u64, u64) {
    // Nanoseconds since a fixed point in this process, from the
    // monotonic clock. A wall clock is the wrong instrument here twice
    // over: it can step backwards, which turns a subtraction into a
    // zero-length interval, and it is the slower read on a path taken
    // once per batch of leaves.
    static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let origin = ORIGIN.get_or_init(std::time::Instant::now);
    (thread_on_core_ticks(), origin.elapsed().as_nanos() as u64)
}

/// Ticks this thread has spent on a core, or zero where the platform
/// offers no such clock.
///
/// The unit differs by platform - cycles on Windows, nanoseconds on
/// Linux - so it is only ever divided by the elapsed count
/// [`clock_pair`] returns beside it, which carries the same unit.
///
/// Zero is the no-clock sentinel and produces full occupancy through
/// the subtraction in [`OccupancyWindow::sample`], which is the
/// behavior that leaves an unsupported platform where it was.
#[cfg(windows)]
pub fn thread_on_core_ticks() -> u64 {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn QueryThreadCycleTime(thread: isize, cycles: *mut u64) -> i32;
    }
    let mut cycles: u64 = 0;
    // SAFETY: a pseudo-handle to the calling thread and an out
    // parameter this stack frame owns. The call writes only through
    // `cycles` and reports failure rather than writing on error.
    let ok = unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut cycles) };
    if ok == 0 {
        return 0;
    }
    // Cycles are not nanoseconds and the conversion needs a frequency
    // this does not have. Both sides of the ratio are scaled by the
    // same unknown constant, so the ratio is unaffected and the
    // absolute figure is never read on its own.
    cycles
}

/// Ticks this thread has spent on a core, in nanoseconds, read from the
/// per-thread CPU clock.
///
/// Zero when that clock cannot be read.
#[cfg(any(target_os = "linux", target_os = "freebsd"))]
pub fn thread_on_core_ticks() -> u64 {
    let mut ts = libc_timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: an out parameter this stack frame owns; the clock id is
    // the per-thread CPU clock, defined on every target this arm covers.
    let r = unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if r != 0 {
        return 0;
    }
    (ts.tv_sec as u64).saturating_mul(1_000_000_000) + ts.tv_nsec as u64
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
#[repr(C)]
struct libc_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

/// The per-thread CPU clock's id, which each kernel numbers for itself:
/// 3 on Linux, 14 on FreeBSD.
#[cfg(target_os = "linux")]
const CLOCK_THREAD_CPUTIME_ID: i32 = 3;

#[cfg(target_os = "freebsd")]
const CLOCK_THREAD_CPUTIME_ID: i32 = 14;

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
#[link(name = "c")]
unsafe extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut libc_timespec) -> i32;
}

/// Zero, on a platform that offers no per-thread CPU clock.
///
/// [`HAS_THREAD_CLOCK`] is what tells a reader that the zero means no
/// clock rather than no time on core.
#[cfg(not(any(windows, target_os = "linux", target_os = "freebsd")))]
pub fn thread_on_core_ticks() -> u64 {
    0
}

/// Whether this platform reports a thread's own CPU time.
///
/// [`thread_on_core_ticks`] returns zero both where no such clock exists
/// and where a thread genuinely spent no time on a core, so the count
/// alone cannot tell a reader which it is holding. This can.
#[cfg(any(windows, target_os = "linux", target_os = "freebsd"))]
pub const HAS_THREAD_CLOCK: bool = true;

#[cfg(not(any(windows, target_os = "linux", target_os = "freebsd")))]
pub const HAS_THREAD_CLOCK: bool = false;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_thread_that_owns_its_core_reads_high() {
        let w = OccupancyWindow::start();
        // Spin rather than sleep: the point is an interval this thread
        // spends running.
        let mut x: u64 = 0;
        let until = std::time::Instant::now() + std::time::Duration::from_millis(40);
        while std::time::Instant::now() < until {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        }
        std::hint::black_box(x);
        let s = w.sample();
        // On a platform with no thread clock this is 100 by
        // construction, which is the same assertion.
        assert!(
            s.percent() >= 50,
            "a spinning thread should hold most of its core, read {}%",
            s.percent()
        );
    }

    #[test]
    fn a_thread_that_sleeps_reads_low_where_the_clock_exists() {
        let w = OccupancyWindow::start();
        std::thread::sleep(std::time::Duration::from_millis(40));
        let s = w.sample();
        if !HAS_THREAD_CLOCK {
            // No thread clock: the figure is inert and reports full
            // occupancy, so there is nothing to assert about sleeping.
            assert_eq!(s.percent(), 100);
            return;
        }
        assert!(
            s.percent() < 50,
            "a sleeping thread used almost no core, yet read {}%",
            s.percent()
        );
    }

    #[test]
    fn a_zero_length_window_reads_full_rather_than_undefined() {
        let s = OccupancySample { thread_ticks: 0, wall_ticks: 0 };
        assert_eq!(s.percent(), 100, "no interval divides by no interval");
    }

    #[test]
    fn occupancy_saturates_rather_than_exceeding_the_interval() {
        let s = OccupancySample { thread_ticks: 2_000, wall_ticks: 1_000 };
        assert_eq!(
            s.percent(),
            100,
            "two clocks of different resolutions must not report 200 percent"
        );
    }
}

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
//!
//! # Where there is no clock
//!
//! Not every platform reports a thread's own CPU time, and a read that
//! should work can fail. Neither is reported as a number. Every count
//! this module produces carries whether it was measured, because the
//! figures it deals in - zero ticks on core, zero percent occupancy -
//! are all values a genuinely starved thread can have, so any of them
//! chosen to stand for "not measured" is a value a caller will read as
//! a measurement.
//!
//! That is not hypothetical here. A `0` returned from an unimplemented
//! platform arm is what left occupancy silently dead on FreeBSD while
//! the crate compiled, linted and passed its suite there.
//!
//! A consumer therefore decides for itself what an unmeasured interval
//! means to it, at the point where it knows. The one thing none of them
//! does is contribute an elapsed interval with no on-core time against
//! it, which would read as a thread that never got a core.

/// Why a thread's on-core count is absent.
///
/// The two are different findings and do not share a value. A platform
/// with no such clock is structural and permanent, so a caller that
/// meets it will meet it on every call and there is nothing to report.
/// A clock that exists here and failed to read is a fault, and a caller
/// may want to say so.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NoReading {
    /// This platform reports no per-thread CPU time.
    NoClock,
    /// The per-thread clock exists on this platform and the read failed.
    Unreadable,
}

/// A thread's on-core count, or why there is none.
///
/// The absence is carried in the type rather than encoded as a zero.
/// Zero is a count a thread can genuinely have - a thread that got no
/// core over the interval - so a zero standing for "no measurement" is
/// indistinguishable from a measurement, and a caller reading the number
/// alone cannot tell which it holds. That ambiguity let a whole platform
/// report plausible occupancy while measuring nothing at all.
///
/// The unit differs by platform - cycles on Windows, nanoseconds
/// elsewhere - so a count is only ever divided by the elapsed count
/// [`clock_pair`] returns beside it, which carries the same unit.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ThreadTicks {
    /// Ticks this thread spent on a core.
    Measured(u64),
    /// No count, and why.
    Absent(NoReading),
}

impl ThreadTicks {
    /// The count, or `None` where there is nothing to count.
    ///
    /// A caller that only needs the number still has to say what it does
    /// without one, which is the property this type exists to enforce.
    pub fn ticks(self) -> Option<u64> {
        match self {
            Self::Measured(ticks) => Some(ticks),
            Self::Absent(_) => None,
        }
    }
}

/// A thread's on-core ticks and the elapsed ticks they were read
/// against, both from the same clock so their ratio is a fraction, or an
/// elapsed interval with no on-core figure to pair with it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OccupancySample {
    /// The thread's own clock was read at both ends of the window.
    Measured {
        /// Ticks this thread spent on a core.
        thread_ticks: u64,
        /// Ticks of elapsed time over the same interval.
        wall_ticks: u64,
    },
    /// The interval elapsed and no on-core count exists for it.
    Unmeasured {
        /// Ticks of elapsed time over the interval.
        wall_ticks: u64,
        /// Why there is no on-core count.
        reason: NoReading,
    },
}

impl OccupancySample {
    /// Occupancy in hundredths: 100 is a thread that ran for the whole
    /// interval, 30 is a thread that got under a third of it, and `None`
    /// is an interval this platform could not measure.
    ///
    /// `None` rather than a number, because every number here is one a
    /// contended thread could genuinely report, so any value chosen to
    /// stand for "not measured" is a value a caller will mistake for a
    /// measurement.
    ///
    /// Saturates at 100. A thread cannot be on a core for more of an
    /// interval than the interval, but the two counters advance for
    /// different reasons - one per executed cycle, one at a fixed rate -
    /// so a boosted core reads slightly over.
    pub fn percent(&self) -> Option<u32> {
        let (thread_ticks, wall_ticks) = match *self {
            Self::Measured { thread_ticks, wall_ticks } => (thread_ticks, wall_ticks),
            Self::Unmeasured { .. } => return None,
        };
        if wall_ticks == 0 {
            return Some(100);
        }
        let pct = thread_ticks.saturating_mul(100) / wall_ticks;
        Some(pct.min(100) as u32)
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
    thread_at_start: ThreadTicks,
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
    /// The interval is measured only when both ends of it carried a
    /// count. One end without one leaves the fraction unknowable, and
    /// the sample says so rather than choosing a number for it: a
    /// consumer decides what an unmeasured interval means to it, at the
    /// point where it knows.
    pub fn sample(&self) -> OccupancySample {
        let (thread, wall) = clock_pair();
        let wall_ticks = wall.saturating_sub(self.wall_at_start);
        match (self.thread_at_start, thread) {
            (ThreadTicks::Measured(start), ThreadTicks::Measured(end)) => {
                OccupancySample::Measured {
                    thread_ticks: end.saturating_sub(start),
                    wall_ticks,
                }
            }
            // A read that failed at either end is reported as the fault
            // it is; only a platform that has no clock at all reports
            // the structural absence.
            (ThreadTicks::Absent(NoReading::Unreadable), _)
            | (_, ThreadTicks::Absent(NoReading::Unreadable)) => OccupancySample::Unmeasured {
                wall_ticks,
                reason: NoReading::Unreadable,
            },
            _ => OccupancySample::Unmeasured { wall_ticks, reason: NoReading::NoClock },
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
pub(crate) fn clock_pair() -> (ThreadTicks, u64) {
    // SAFETY: `_rdtsc` reads a counter register and touches no memory.
    (thread_on_core_ticks(), unsafe { core::arch::x86_64::_rdtsc() })
}

#[cfg(not(all(windows, target_arch = "x86_64")))]
pub(crate) fn clock_pair() -> (ThreadTicks, u64) {
    // Nanoseconds since a fixed point in this process, from the
    // monotonic clock. A wall clock is the wrong instrument here twice
    // over: it can step backwards, which turns a subtraction into a
    // zero-length interval, and it is the slower read on a path taken
    // once per batch of leaves.
    static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let origin = ORIGIN.get_or_init(std::time::Instant::now);
    (thread_on_core_ticks(), origin.elapsed().as_nanos() as u64)
}

/// Ticks this thread has spent on a core, in cycles, read from the
/// thread's cycle counter.
///
/// The unit differs by platform - cycles on Windows, nanoseconds
/// elsewhere - so a count is only ever divided by the elapsed count
/// [`clock_pair`] returns beside it, which carries the same unit.
#[cfg(windows)]
pub fn thread_on_core_ticks() -> ThreadTicks {
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
        return ThreadTicks::Absent(NoReading::Unreadable);
    }
    // Cycles are not nanoseconds and the conversion needs a frequency
    // this does not have. Both sides of the ratio are scaled by the
    // same unknown constant, so the ratio is unaffected and the
    // absolute figure is never read on its own.
    ThreadTicks::Measured(cycles)
}

/// Ticks this thread has spent on a core, in nanoseconds, read from the
/// per-thread CPU clock.
#[cfg(any(target_os = "linux", target_os = "freebsd"))]
pub fn thread_on_core_ticks() -> ThreadTicks {
    let mut ts = libc_timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: an out parameter this stack frame owns; the clock id is
    // the per-thread CPU clock, defined on every target this arm covers.
    let r = unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if r != 0 {
        return ThreadTicks::Absent(NoReading::Unreadable);
    }
    ThreadTicks::Measured(
        (ts.tv_sec as u64).saturating_mul(1_000_000_000) + ts.tv_nsec as u64,
    )
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

/// No count, on a platform that offers no per-thread CPU clock.
///
/// None of the gated hosts compiles this arm - they are Windows, Linux
/// and FreeBSD - so it is written to be correct by inspection rather
/// than by test, and the logic that consumes it is exercised on every
/// host through [`OccupancySample`] values built directly.
#[cfg(not(any(windows, target_os = "linux", target_os = "freebsd")))]
pub fn thread_on_core_ticks() -> ThreadTicks {
    ThreadTicks::Absent(NoReading::NoClock)
}

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
        let Some(pct) = w.sample().percent() else {
            // A platform with no clock has nothing to assert about a
            // spinning thread, and says so rather than reporting a
            // figure this test would then check.
            return;
        };
        assert!(pct >= 50, "a spinning thread should hold most of its core, read {pct}%");
    }

    #[test]
    fn a_thread_that_sleeps_reads_low_where_the_clock_exists() {
        let w = OccupancyWindow::start();
        std::thread::sleep(std::time::Duration::from_millis(40));
        let Some(pct) = w.sample().percent() else {
            return;
        };
        assert!(pct < 50, "a sleeping thread used almost no core, yet read {pct}%");
    }

    #[test]
    fn a_zero_length_window_reads_full_rather_than_undefined() {
        let s = OccupancySample::Measured { thread_ticks: 0, wall_ticks: 0 };
        assert_eq!(s.percent(), Some(100), "no interval divides by no interval");
    }

    #[test]
    fn occupancy_saturates_rather_than_exceeding_the_interval() {
        let s = OccupancySample::Measured { thread_ticks: 2_000, wall_ticks: 1_000 };
        assert_eq!(
            s.percent(),
            Some(100),
            "two clocks of different resolutions must not report 200 percent"
        );
    }

    // The arms below are the ones no gated host compiles. They are
    // reached here by building the values directly, so the behavior a
    // clockless platform would get is checked on every host.

    #[test]
    fn an_unmeasured_interval_reports_no_figure_rather_than_a_plausible_one() {
        for reason in [NoReading::NoClock, NoReading::Unreadable] {
            let s = OccupancySample::Unmeasured { wall_ticks: 1_000, reason };
            assert_eq!(
                s.percent(),
                None,
                "an interval with no on-core count has no occupancy to report"
            );
        }
    }

    #[test]
    fn a_measured_zero_is_not_the_same_finding_as_no_measurement() {
        let measured = OccupancySample::Measured { thread_ticks: 0, wall_ticks: 1_000 };
        let unmeasured =
            OccupancySample::Unmeasured { wall_ticks: 1_000, reason: NoReading::NoClock };
        assert_eq!(
            measured.percent(),
            Some(0),
            "a thread that genuinely got no core ran at zero percent"
        );
        assert_eq!(unmeasured.percent(), None);
        assert_ne!(
            measured.percent(),
            unmeasured.percent(),
            "total contention and no clock must not read the same"
        );
    }

    #[test]
    fn an_absent_count_yields_no_number_to_a_caller_that_wants_one() {
        assert_eq!(ThreadTicks::Measured(7).ticks(), Some(7));
        assert_eq!(ThreadTicks::Absent(NoReading::NoClock).ticks(), None);
        assert_eq!(ThreadTicks::Absent(NoReading::Unreadable).ticks(), None);
    }

    #[test]
    #[cfg(any(windows, target_os = "linux", target_os = "freebsd"))]
    fn this_host_reads_its_own_thread_clock() {
        // The gate runs on Windows, Linux and FreeBSD, each of which has
        // the clock, so an absent count here is a defect in that host's
        // arm rather than an unsupported platform. Naming it this way is
        // what turns a silently dead instrument into a failing test.
        match thread_on_core_ticks() {
            ThreadTicks::Measured(_) => {}
            ThreadTicks::Absent(reason) => panic!(
                "this platform is expected to report thread CPU time, got {reason:?}"
            ),
        }
    }
}

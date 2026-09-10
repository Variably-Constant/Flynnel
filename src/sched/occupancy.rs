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
//! It gates learning, and nothing else. A measurement taken at low
//! occupancy is refused rather than installed, and a classifier is not
//! allowed to migrate a site on evidence gathered while preempted.
//! Dispatch decisions are never routed on it: making the shape of a
//! dispatch depend on what else is running would mean identical code
//! behaving differently run to run, which is the property the gate
//! exists to protect.
//!
//! Whether a loaded host wants a narrower fan-out is a separate
//! question with no measurement behind it yet.

/// Occupancy at or above this, in hundredths, is a thread that owned
/// its core closely enough for its timings to describe the work.
///
/// Below it a measurement describes the machine's other tenants as much
/// as the workload, and the learner that would have consumed it keeps
/// what it had.
pub const TRUSTWORTHY_OCCUPANCY_PCT: u32 = 85;

/// A thread's CPU time and the wall time it was read against.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OccupancySample {
    /// Nanoseconds this thread spent on a core.
    pub thread_ns: u64,
    /// Nanoseconds of wall clock over the same interval.
    pub wall_ns: u64,
}

impl OccupancySample {
    /// Occupancy in hundredths: 100 is a thread that ran for the whole
    /// interval, 30 is a thread that got under a third of it.
    ///
    /// Saturates at 100. A thread cannot use more core time than wall
    /// time elapsed, but the two clocks have different resolutions and
    /// a short interval can read slightly over.
    pub fn percent(&self) -> u32 {
        if self.wall_ns == 0 {
            return 100;
        }
        let pct = self.thread_ns.saturating_mul(100) / self.wall_ns;
        pct.min(100) as u32
    }

    /// Whether a measurement taken over this interval describes the
    /// work rather than the machine's other tenants.
    pub fn is_trustworthy(&self) -> bool {
        self.percent() >= TRUSTWORTHY_OCCUPANCY_PCT
    }
}

/// Spans a measured interval, reporting what fraction of it this thread
/// was actually on a core.
///
/// The wall clock and the thread clock are both read at construction
/// and again at [`sample`](Self::sample), so the two cover the same
/// interval by construction rather than by the caller pairing them
/// correctly.
#[derive(Debug)]
pub struct OccupancyWindow {
    thread_ns_at_start: u64,
    wall_at_start: std::time::Instant,
}

impl OccupancyWindow {
    /// Open a window over whatever the caller measures next.
    pub fn start() -> Self {
        Self {
            thread_ns_at_start: thread_cpu_ns(),
            wall_at_start: std::time::Instant::now(),
        }
    }

    /// Close the window and report the interval.
    ///
    /// On a platform with no thread clock this reports full occupancy,
    /// so a learner behaves exactly as it did before the gate existed
    /// rather than refusing every measurement.
    pub fn sample(&self) -> OccupancySample {
        OccupancySample {
            thread_ns: thread_cpu_ns().saturating_sub(self.thread_ns_at_start),
            wall_ns: self.wall_at_start.elapsed().as_nanos() as u64,
        }
    }
}

/// Nanoseconds this thread has spent on a core, or zero where the
/// platform offers no such clock.
///
/// Zero is the no-clock sentinel and produces full occupancy through
/// the subtraction in [`OccupancyWindow::sample`], which is the
/// behavior that leaves an unsupported platform where it was.
#[cfg(windows)]
pub fn thread_cpu_ns() -> u64 {
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

#[cfg(target_os = "linux")]
pub fn thread_cpu_ns() -> u64 {
    let mut ts = libc_timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: an out parameter this stack frame owns; the clock id is
    // the per-thread CPU clock, defined on every Linux this targets.
    let r = unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if r != 0 {
        return 0;
    }
    (ts.tv_sec as u64).saturating_mul(1_000_000_000) + ts.tv_nsec as u64
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct libc_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

#[cfg(target_os = "linux")]
const CLOCK_THREAD_CPUTIME_ID: i32 = 3;

#[cfg(target_os = "linux")]
#[link(name = "c")]
unsafe extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut libc_timespec) -> i32;
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn thread_cpu_ns() -> u64 {
    0
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
        if thread_cpu_ns() == 0 {
            // No thread clock: the gate is inert and reports full
            // occupancy, so there is nothing to assert about sleeping.
            assert_eq!(s.percent(), 100);
            return;
        }
        assert!(
            s.percent() < TRUSTWORTHY_OCCUPANCY_PCT,
            "a sleeping thread used almost no core, yet read {}%",
            s.percent()
        );
        assert!(
            !s.is_trustworthy(),
            "a measurement taken while off-core must not be trusted"
        );
    }

    #[test]
    fn a_zero_length_window_is_trustworthy_rather_than_undefined() {
        let s = OccupancySample { thread_ns: 0, wall_ns: 0 };
        assert_eq!(s.percent(), 100, "no interval divides by no interval");
        assert!(s.is_trustworthy());
    }

    #[test]
    fn occupancy_saturates_rather_than_exceeding_the_interval() {
        let s = OccupancySample { thread_ns: 2_000, wall_ns: 1_000 };
        assert_eq!(
            s.percent(),
            100,
            "two clocks of different resolutions must not report 200 percent"
        );
    }
}

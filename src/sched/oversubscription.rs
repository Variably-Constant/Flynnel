//! Whether this process has lately held more runnable threads than
//! cores, read from how long a `yield_now` took to come back.
//!
//! A yield with nothing else ready returns in about a microsecond. A
//! yield that hands the core to a ready thread returns when that
//! thread's time slice ends, milliseconds later. So a long yield is a
//! direct reading that another thread wanted this core, and it needs no
//! count of threads or cores, which a process cannot see for the host
//! it shares. How long counts as long is [`LONG_YIELD`], and it is set
//! from measured lengths rather than from that argument alone, because
//! on a guest a yield is also stretched by its vCPU being descheduled.
//!
//! The pool's idle rounds take the reading. They yield with no latch
//! pending, so a yield that runs long there costs a worker nothing it
//! was waiting for. A join waiter yielding to take the same reading
//! would pay the stall this exists to avoid: the latch it waits on is
//! set while it is away.
//!
//! [`crate::sched::levers::join_park_oversubscribed`] reads `recently`
//! to decide whether a join waiter whose spin budget is spent parks in
//! the kernel or yields. A quiet process then never parks a join
//! waiter. That matters on a virtual machine, where a parked thread
//! halts its vCPU and the wake that follows goes through the
//! hypervisor: on a 16-vCPU guest an unconditional join park raised a
//! quiet dispatch's median from 1.3 to 3.1 ms, while on bare metal the
//! same park woke in 2 to 7 us and cost nothing measurable.
//!
//! Nothing here runs unless that switch is on.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// A yield that took at least this long lost its core to another thread
/// for a time slice.
///
/// Set from the lengths yields take, every one timed, in the
/// oversubscribed caller on a 16-vCPU Linux guest. Two processes with no
/// spinners saw 13 and 37 of about 7.9 million yields reach 2 ms; two
/// with 12 spinners beside their 16 workers saw 10,106 of 7.4 million
/// and 27,393 of 7.2 million.
/// Lower lines do not separate the two: the quiet processes also held
/// about 1,500 yields between 0.5 and 2 ms, and at 100 us they read about
/// 1,970 long yields each and parked 15,400 to 21,300 times, which cost
/// the quiet median about 6 percent there. A guest's vCPU is descheduled
/// mid-yield often enough to look like a lost core; a lost time slice is
/// longer.
pub const LONG_YIELD: Duration = Duration::from_millis(2);

/// How long after a long yield the process still counts as
/// oversubscribed.
///
/// Short, because idle rounds under load take a new reading every time
/// a worker goes idle, and a stale reading keeps parking join waiters in
/// a process that has gone quiet. A virtual machine pays for each of
/// those parks.
pub const WINDOW: Duration = Duration::from_millis(10);

/// The origin of the readings below, fixed on first use.
static EPOCH: OnceLock<Instant> = OnceLock::new();

/// When the last long yield ended, in nanoseconds after [`EPOCH`] plus
/// one, or zero for none yet. The offset keeps a reading taken at the
/// epoch itself from reading as none.
static LAST_LONG_YIELD_NS: AtomicU64 = AtomicU64::new(0);

/// Long yields seen, process-wide.
static LONG_YIELDS: AtomicU64 = AtomicU64::new(0);

/// How many buckets [`yield_histogram`] answers.
pub const YIELD_BUCKETS: usize = 16;

/// Every timed yield, by length: bucket 0 is under a microsecond, bucket
/// `i` from 2^(i-1) up to 2^i microseconds, and the last bucket takes
/// everything from 2^14 microseconds up.
static YIELD_HISTOGRAM: [AtomicU64; YIELD_BUCKETS] = [const { AtomicU64::new(0) }; YIELD_BUCKETS];

fn bucket_of(took: Duration) -> usize {
    let us = took.as_micros();
    let bits = (u128::BITS - us.leading_zeros()) as usize;
    bits.min(YIELD_BUCKETS - 1)
}

fn now_ns() -> u64 {
    let since = EPOCH.get_or_init(Instant::now).elapsed().as_nanos();
    // Clamped rather than converted: u64 nanoseconds last 584 years.
    since.min(u128::from(u64::MAX - 1)) as u64 + 1
}

/// Yields the core and answers how long the yield took, recording one
/// that took [`LONG_YIELD`] or more.
pub(crate) fn timed_yield() -> Duration {
    let started = Instant::now();
    std::thread::yield_now();
    let took = started.elapsed();
    YIELD_HISTOGRAM[bucket_of(took)].fetch_add(1, Relaxed);
    if took >= LONG_YIELD {
        LONG_YIELDS.fetch_add(1, Relaxed);
        LAST_LONG_YIELD_NS.store(now_ns(), Relaxed);
    }
    took
}

/// Whether a long yield ended within [`WINDOW`] of `now_ns`, given when
/// the last one ended. Zero for `last_ns` is none.
///
/// Split out from [`recently`] so the window can be checked as
/// arithmetic: the reading it takes is process-wide, and every yield in
/// the process under the switch writes it.
fn within_window(last_ns: u64, now_ns: u64) -> bool {
    last_ns != 0 && now_ns.saturating_sub(last_ns) < WINDOW.as_nanos() as u64
}

/// Whether a yield anywhere in the process gave its core away within
/// the last [`WINDOW`].
pub(crate) fn recently() -> bool {
    within_window(LAST_LONG_YIELD_NS.load(Relaxed), now_ns())
}

/// How many yields have given their core away since the process started,
/// counted only while `FLYNNEL_LEVER_JOIN_PARK_OVERSUBSCRIBED` is on,
/// since only then are yields timed. A run that sets the switch under
/// load and reads zero here never saw the process oversubscribed.
pub fn total_long_yields() -> u64 {
    LONG_YIELDS.load(Relaxed)
}

/// Every timed yield since the process started, counted by length in
/// log2 buckets of microseconds: bucket 0 is under a microsecond, bucket
/// `i` from 2^(i-1) up to 2^i, the last from 2^14 up. Filled only while
/// `FLYNNEL_LEVER_JOIN_PARK_OVERSUBSCRIBED` is on. What separates a yield
/// that lost its core from one that did not is read here rather than
/// assumed, and [`LONG_YIELD`] is set from it.
pub fn yield_histogram() -> [u64; YIELD_BUCKETS] {
    std::array::from_fn(|i| YIELD_HISTOGRAM[i].load(Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_long_yield_is_never_recent() {
        assert!(!within_window(0, 1));
        assert!(!within_window(0, u64::MAX));
    }

    #[test]
    fn a_long_yield_counts_for_the_window_and_then_stops() {
        let window = WINDOW.as_nanos() as u64;
        let at = 5_000_000_000u64;
        assert!(within_window(at, at));
        assert!(within_window(at, at + window - 1));
        assert!(!within_window(at, at + window));
        assert!(!within_window(at, at + 10 * window));
    }

    #[test]
    fn a_yield_lands_in_the_bucket_of_its_length() {
        assert_eq!(bucket_of(Duration::from_nanos(900)), 0);
        assert_eq!(bucket_of(Duration::from_micros(1)), 1);
        assert_eq!(bucket_of(Duration::from_micros(3)), 2);
        assert_eq!(bucket_of(Duration::from_micros(4)), 3);
        assert_eq!(bucket_of(Duration::from_micros(100)), 7);
        assert_eq!(bucket_of(Duration::from_millis(8)), 13);
        assert_eq!(bucket_of(Duration::from_millis(16)), 14);
        assert_eq!(bucket_of(Duration::from_millis(17)), YIELD_BUCKETS - 1);
        assert_eq!(bucket_of(Duration::from_secs(10)), YIELD_BUCKETS - 1);
    }

    #[test]
    fn a_reading_from_before_the_last_one_still_counts() {
        // Two workers store their readings in either order, so a reader
        // can see a last yield stamped after its own clock read.
        let at = 5_000_000_000u64;
        assert!(within_window(at, at - 1));
    }

    #[test]
    fn a_timed_yield_comes_back_and_says_how_long_it_took() {
        // Whether this yield counts as long depends on what else the
        // host is running, so the reading it leaves is not asserted.
        let took = timed_yield();
        assert!(took < Duration::from_secs(1), "a single yield took {took:?}");
    }
}

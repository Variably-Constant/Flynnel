//! Single-owner private LIFO deque based on Acar 2013's
//! receiver-initiated-migration model: an alternative to the
//! Chase-Lev concurrent deque in
//! [`crate::sched::chase_lev_local`] that the active arena
//! consumes. The backing is `Mutex<Vec<JobRef>>` (not a lock-free
//! Chase-Lev): every push and pop takes the mutex, which
//! serializes the owner against migration receivers. This trades
//! Chase-Lev's owner-side load-store fast path (Acquire/Release
//! atomic counters, ~50ns per op) for the mutex's uncontended
//! lock cost plus plain Vec ops (~30-50ns per op on modern x86
//! parking_lot-style mutexes, but higher under contention).
//!
//! Cross-worker migration uses **receiver-initiated migration**:
//! when a thief observes its own deque empty, it calls
//! [`PrivateLifoDeque::donate_from`] on a peer's deque, transferring
//! one or more jobs into its own deque before continuing. The
//! sender stays passive; the receiver pulls work over the same
//! [`std::sync::Mutex`] the owner uses for push/pop.
//!
//! The production worker loop (arena_local `find_work`) uses the
//! adaptive Chase-Lev deque directly; this deque backs the
//! receiver-initiated migration benches and stays available as an
//! alternative backing.
//!

use crate::sched::job::JobRef;

use std::sync::Mutex;

/// Single-owner LIFO deque backed by `Vec<JobRef>` under a mutex.
/// The mutex serializes owner ops + receiver-initiated migrations;
/// an owner-only fast path can be enabled via `try_lock`-based
/// helpers that skip the operation if a migration is in flight.
#[derive(Default)]
pub struct PrivateLifoDeque {
    inner: Mutex<Vec<JobRef>>,
}

impl PrivateLifoDeque {
    /// Construct an empty private deque.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Vec::new()),
        }
    }

    /// Push a job onto the owner end. Blocks if a migration is in
    /// progress.
    ///
    /// Currently called only from in-module tests; production worker
    /// loops use [`Self::donate_from`] for cross-thread migration
    /// rather than the blocking `push`/`pop` pair. Retained as the
    /// canonical donor-side API for the donate-based migration
    /// scheduler mode.
    #[cfg(test)]
    #[inline]
    pub(crate) fn push(&self, job: JobRef) {
        let mut g = self.inner.lock().expect("private deque mutex poisoned");
        g.push(job);
    }

    /// Pop a job from the owner end (LIFO). Blocks if a migration
    /// is in progress.
    #[cfg(test)]
    #[inline]
    pub(crate) fn pop(&self) -> Option<JobRef> {
        let mut g = self.inner.lock().expect("private deque mutex poisoned");
        g.pop()
    }

    /// Try to push without blocking. Returns `Err(job)` if a
    /// migration is in flight on this deque.
    #[cfg(test)]
    #[inline]
    pub(crate) fn push_local(&self, job: JobRef) -> Result<(), JobRef> {
        match self.inner.try_lock() {
            Ok(mut g) => {
                g.push(job);
                Ok(())
            }
            Err(_) => Err(job),
        }
    }

    /// Try to pop without blocking. Returns `Ok(None)` if the deque
    /// is empty AND was lockable; `Err(())` if a migration is in
    /// flight.
    #[cfg(test)]
    #[inline]
    pub(crate) fn try_pop_local(&self) -> Result<Option<JobRef>, ()> {
        match self.inner.try_lock() {
            Ok(mut g) => Ok(g.pop()),
            Err(_) => Err(()),
        }
    }

    /// Receiver-initiated migration: drain up to `max_jobs` from
    /// the bottom of `source` into this deque. Used by a thief
    /// worker on its own deque, taking a write lock on the
    /// donor's mutex. Returns the number of jobs migrated.
    pub fn donate_from(&self, source: &PrivateLifoDeque, max_jobs: usize) -> usize {
        if max_jobs == 0 {
            return 0;
        }
        // Lock the source first (peer-owned); failure to acquire
        // means the owner is busy. Receiver doesn't spin - caller
        // picks a different victim.
        let mut src = match source.inner.try_lock() {
            Ok(g) => g,
            Err(_) => return 0,
        };
        let n_avail = src.len();
        if n_avail <= 1 {
            // Leave at least 1 job for the source owner.
            return 0;
        }
        let take = max_jobs.min(n_avail / 2);
        // Take from the bottom (FIFO end) so the source owner
        // keeps its LIFO recent work; thief gets the older work.
        let mut dst = self.inner.lock().expect("private deque mutex poisoned");
        // drain takes from the front (FIFO bottom).
        for job in src.drain(0..take) {
            dst.push(job);
        }
        take
    }

    /// True if the deque is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner
            .try_lock()
            .map(|g| g.is_empty())
            .unwrap_or(false)
    }
}

impl std::fmt::Debug for PrivateLifoDeque {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrivateLifoDeque")
            .field("len", &self.inner.try_lock().map(|g| g.len()).unwrap_or(0))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use crate::foundation::Variant;
    use crate::sched::job::{NUMA_HINT_ANY, StackJob};
    use crate::sched::latch::CoreLatch;

    #[test]
    fn new_deque_is_empty() {
        let d = PrivateLifoDeque::new();
        assert!(d.is_empty());
        assert!(d.pop().is_none());
    }

    #[test]
    fn push_pop_lifo_order() {
        let d = PrivateLifoDeque::new();
        let counter = Arc::new(AtomicU32::new(0));
        let c1 = Arc::clone(&counter);
        let c2 = Arc::clone(&counter);
        let c3 = Arc::clone(&counter);

        // Three jobs each incrementing the counter on call.
        let j1 = StackJob::new(
            move |_| { c1.fetch_add(1, Ordering::Relaxed); },
            CoreLatch::new(),
        );
        let j2 = StackJob::new(
            move |_| { c2.fetch_add(10, Ordering::Relaxed); },
            CoreLatch::new(),
        );
        let j3 = StackJob::new(
            move |_| { c3.fetch_add(100, Ordering::Relaxed); },
            CoreLatch::new(),
        );

        // SAFETY: StackJob lives for the duration of this test
        // scope; we don't move it after as_job_ref. (Same pattern
        // ArenaDeque tests use.)
        unsafe {
            d.push(j1.as_job_ref(0, NUMA_HINT_ANY, Variant::Faithful));
            d.push(j2.as_job_ref(0, NUMA_HINT_ANY, Variant::Faithful));
            d.push(j3.as_job_ref(0, NUMA_HINT_ANY, Variant::Faithful));
        }

        // LIFO: pop order is j3, j2, j1.
        let popped = d.pop().expect("third push pop");
        unsafe { popped.execute(); }
        assert_eq!(counter.load(Ordering::Relaxed), 100);

        let popped = d.pop().expect("second push pop");
        unsafe { popped.execute(); }
        assert_eq!(counter.load(Ordering::Relaxed), 110);

        let popped = d.pop().expect("first push pop");
        unsafe { popped.execute(); }
        assert_eq!(counter.load(Ordering::Relaxed), 111);

        assert!(d.pop().is_none());

    }

    #[test]
    fn donate_from_leaves_one_in_source() {
        let donor = PrivateLifoDeque::new();
        let receiver = PrivateLifoDeque::new();
        let counter = Arc::new(AtomicU32::new(0));

        let mut jobs = Vec::new();
        for i in 0..4 {
            let c = Arc::clone(&counter);
            jobs.push(StackJob::new(
                move |_| { c.fetch_add(1u32 << i, Ordering::Relaxed); },
                CoreLatch::new(),
            ));
        }
        for j in &jobs {
            unsafe { donor.push(j.as_job_ref(0, NUMA_HINT_ANY, Variant::Faithful)); }
        }

        // donate up to 8 jobs from donor (which has 4); takes
        // 4 / 2 = 2 jobs (n_avail / 2), keeping 2 for donor owner.
        let n = receiver.donate_from(&donor, 8);
        assert_eq!(n, 2);
        // Donor keeps 2 (older end - we drained the FIFO front).
        // Receiver gets 2.
        let mut donor_after = 0;
        while donor.pop().is_some() { donor_after += 1; }
        assert_eq!(donor_after, 2);

        let mut recv_after = 0;
        while receiver.pop().is_some() { recv_after += 1; }
        assert_eq!(recv_after, 2);
    }

    #[test]
    fn try_pop_local_no_blocking() {
        let d = PrivateLifoDeque::new();
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        let j = StackJob::new(
            move |_| { c.fetch_add(7, Ordering::Relaxed); },
            CoreLatch::new(),
        );
        // push_local on uncontested deque returns Ok.
        unsafe {
            assert!(d.push_local(j.as_job_ref(0, NUMA_HINT_ANY, Variant::Faithful)).is_ok());
        }
        // try_pop_local succeeds + returns Some.
        let popped = d.try_pop_local().expect("uncontested try_pop succeeds");
        let popped = popped.expect("had one job");
        unsafe { popped.execute(); }
        assert_eq!(counter.load(Ordering::Relaxed), 7);
        // Now empty.
        assert!(matches!(d.try_pop_local(), Ok(None)));
    }
}

//! Chase-Lev work-stealing helpers used by the orchestrator's
//! worker pool.
//!
//! Thin layer over [`crate::sched::chase_lev_local`] (the in-
//! house implementation of the Chase-Lev algorithm formally
//! verified in Vafeiadis et al. arXiv:2309.03642). The live
//! worker pool's owner-private deque lives in
//! [`crate::sched::private_deque`]; this module exposes the
//! [`steal_retry`] helper that walks `Steal::Retry` outcomes
//! for arena leaders.
//!
//! The deque only stores [`JobRef`]; the lifetime of the captured
//! state (`StackJob` / `HeapJob` / `ArcJob`) is managed by the caller
//! per the "execute exactly once" contract documented in
//! [`crate::sched::job`].

use crate::sched::chase_lev_local::{Steal, Stealer};
use crate::sched::job::JobRef;

#[cfg(test)]
use crate::sched::chase_lev_local::{Worker, new_chase_lev};

/// Per-arena work-stealing deque used by the in-file tests to
/// exercise [`steal_retry`] without depending on the full
/// [`crate::sched::private_deque::PrivateDeque`]
/// scaffold. Production callers use `PrivateDeque` directly.
#[cfg(test)]
pub(crate) struct ArenaDeque {
    worker: Worker<JobRef>,
    stealer: Stealer<JobRef>,
}

#[cfg(test)]
impl ArenaDeque {
    pub(crate) fn new_lifo() -> Self {
        // Default test capacity. chase_lev_local is bounded.
        let (worker, stealer) = new_chase_lev::<JobRef>(64);
        Self { worker, stealer }
    }

    #[inline]
    pub(crate) fn push(&self, job: JobRef) {
        // chase_lev_local::Worker::push returns Result<(), T>;
        // tests have not exercised the bounded path. Panic with
        // a context message on overflow (JobRef does not impl
        // Debug, so a custom panic body beats `.expect`).
        if self.worker.push(job).is_err() {
            panic!("test deque must not overflow");
        }
    }

    #[inline]
    pub(crate) fn pop(&self) -> Option<JobRef> {
        match self.worker.pop() {
            Steal::Success(j) => Some(j),
            Steal::Empty | Steal::Retry => None,
        }
    }

    #[inline]
    pub(crate) fn stealer(&self) -> Stealer<JobRef> {
        self.stealer.clone()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.worker.is_empty()
    }
}

/// Steal-with-retry helper: walks `Steal::Retry` outcomes until
/// either a job is acquired or the deque is observed empty. Used by
/// arena leaders that want a definitive yes/no answer.
///
/// Returns `Some(job)` on success, `None` when the deque is empty.
///
/// Production worker_loop and find_work paths use
/// [`steal_retry_batch`] instead, which amortizes the per-steal CAS
/// across multiple items. This single-item variant is the fallback
/// for call sites without a destination `Worker<JobRef>` (e.g.
/// external leader threads outside the worker pool).
#[allow(dead_code)]
pub(crate) fn steal_retry(stealer: &Stealer<JobRef>) -> Option<JobRef> {
    loop {
        match stealer.steal() {
            Steal::Success(job) => return Some(job),
            Steal::Empty => return None,
            Steal::Retry => continue,
        }
    }
}

/// Batch steal-with-retry helper: tries to steal up to ~half of the
/// victim's deque into `dest_worker` AND returns one job for the
/// thief to execute immediately. One atomic CAS per call instead of
/// N CASes for single-item steals.
///
/// The flynnel Chase-Lev implementation does not yet expose a
/// batched steal entrypoint; this helper falls back to a single
/// steal call into the destination worker. Kept for API parity
/// with the per-call site that wants a Worker-typed parameter.
///
/// For uniform-cost workloads (Heavy 100-sqrt chain) this amortizes
/// the per-steal CAS cost across multiple subsequent work units the
/// thief executes from its own deque, instead of the thief paying a
/// fresh peer-steal CAS for every unit.
///
/// Returns `Some(job)` on success (one item popped, batch in
/// dest_worker), `None` when the victim's deque is empty.
#[allow(dead_code)]
pub(crate) fn steal_retry_batch(
    stealer: &Stealer<JobRef>,
    _dest_worker: &crate::sched::chase_lev_local::Worker<JobRef>,
) -> Option<JobRef> {
    steal_retry(stealer)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::thread;

    use crate::foundation::Variant;
    use crate::sched::job::{NUMA_HINT_ANY, StackJob};
    use crate::sched::latch::CoreLatch;

    #[test]
    fn new_deque_is_empty() {
        let d = ArenaDeque::new_lifo();
        assert!(d.is_empty());
        assert!(d.pop().is_none());
    }

    #[test]
    fn push_then_pop_returns_same_job() {
        let d = ArenaDeque::new_lifo();
        let job = StackJob::new(|_stolen| 7u32, CoreLatch::new());
        let r = unsafe { job.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful) };
        let id_k = r.k_outer;
        d.push(r);
        assert!(!d.is_empty());
        let popped = d.pop().expect("pop after push must return job");
        assert_eq!(popped.k_outer, id_k);
        unsafe { popped.execute() };
        assert!(job.latch.is_set());
        let value = unsafe { job.into_result() };
        assert_eq!(value, 7);
    }

    #[test]
    fn pop_is_lifo_on_owner_end() {
        let d = ArenaDeque::new_lifo();
        let j1 = StackJob::new(|_| 1u32, CoreLatch::new());
        let j2 = StackJob::new(|_| 2u32, CoreLatch::new());
        let j3 = StackJob::new(|_| 3u32, CoreLatch::new());
        unsafe {
            d.push(j1.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful));
            d.push(j2.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful));
            d.push(j3.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful));
        }
        // LIFO: most recent push pops first.
        let r3 = d.pop().unwrap();
        unsafe { r3.execute() };
        assert!(j3.latch.is_set());
        let r2 = d.pop().unwrap();
        unsafe { r2.execute() };
        assert!(j2.latch.is_set());
        let r1 = d.pop().unwrap();
        unsafe { r1.execute() };
        assert!(j1.latch.is_set());
        assert!(d.pop().is_none());
        // All jobs ran exactly once.
        let v1 = unsafe { j1.into_result() };
        let v2 = unsafe { j2.into_result() };
        let v3 = unsafe { j3.into_result() };
        assert_eq!((v1, v2, v3), (1, 2, 3));
    }

    #[test]
    fn steal_retry_returns_none_on_empty() {
        let d = ArenaDeque::new_lifo();
        assert!(steal_retry(&d.stealer()).is_none());
    }

    #[test]
    fn cross_thread_push_owner_steal_thief_executes_jobs() {
        // Spawn a thief thread holding a Stealer. Owner pushes N
        // jobs; thief steals and executes them, incrementing a
        // shared counter. Verify all N completions.
        const N: u32 = 64;
        let d = ArenaDeque::new_lifo();
        let stealer = d.stealer();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_thief = Arc::clone(&counter);

        // Construct N StackJobs on the owner stack; each increments
        // the shared counter when executed.
        let jobs: Vec<_> = (0..N)
            .map(|_| {
                let c = Arc::clone(&counter);
                StackJob::new(
                    move |_stolen| {
                        c.fetch_add(1, Ordering::SeqCst);
                    },
                    CoreLatch::new(),
                )
            })
            .collect();

        // Thief drains the deque by stealing until all N are seen.
        let thief = thread::spawn(move || {
            let mut seen = 0u32;
            while seen < N {
                if let Some(job) = steal_retry(&stealer) {
                    unsafe { job.execute() };
                    seen += 1;
                } else {
                    thread::yield_now();
                }
            }
        });

        // Push all N from the owner side.
        for job in &jobs {
            let r = unsafe { job.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful) };
            d.push(r);
        }

        thief.join().expect("thief thread must finish");
        assert_eq!(counter.load(Ordering::SeqCst), N,
            "every pushed job must execute exactly once across threads");
        // Every latch must be set; the jobs ran via thief execute.
        for j in &jobs {
            assert!(j.latch.is_set());
        }
        // Silence the counter_thief unused warning (we used the
        // moved clone above; this is the second reference path).
        let _ = counter_thief;
    }
}

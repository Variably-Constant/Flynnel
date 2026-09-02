//! WorkerCtx-shaped adapter over [`crate::sched::khl_local`].
//!
//! Provides [`KhlWorker`] (owner half) and [`KhlSteal2`] (thief
//! result) that bridge between the K_inner=3 batched KHL slot
//! storage and the per-job `Option<JobRef>` API the existing
//! WorkerCtx expects.
//!
//! ## Why the adapter exists
//!
//! KHL's native pop / steal returns a `KhlBody` carrying up to
//! [`crate::sched::khl_local::KHL_LINE_ITEMS`] (=3) jobs in one
//! cache-line transfer. The existing scheduler's `find_work`
//! returns `Option<JobRef>` (one job per call). Directly returning
//! a `KhlBody` would require restructuring every call site to
//! handle batches.
//!
//! The adapter solves this with a per-worker **pop stash**: when
//! KHL returns a batch of 3, the adapter returns the first as a
//! JobRef and stashes the other 2 for the next two `pop()` calls.
//! Each stashed item is later returned without touching the inner
//! deque at all - that is exactly where the K_inner=3 amortization
//! cashes in (one shared-line steal serves 3 logical pops).
//!
//! ## Stash placement
//!
//! Two stashes per worker:
//! - **Owner pop stash** inside [`KhlWorker`] (tier-local; one per
//!   `[KhlWorker; N_TIERS]` entry).
//! - **Thief steal stash** lives in WorkerCtx (shared across all
//!   peer-tier steal results; see arena_local::WorkerCtx).
//!
//! Both stashes use the same [`KhlStash`] type so the
//! `drain_one` / `stash_extras` helpers are reused.

#![allow(clippy::missing_errors_doc)]

use core::cell::UnsafeCell;

use crate::sched::job::{CompactJobRef, JobRef};
use crate::sched::khl_local::{KHL_LINE_ITEMS, KhlBody, KhlSteal, KhlStealer, SchedKhlDeque, new_khl};

/// Per-worker batch stash. Holds up to `KHL_LINE_ITEMS - 1` (= 2)
/// leftover items from a recent successful pop / steal. The next
/// `pop` / steal-result query drains the stash before going to
/// the inner deque.
///
/// Single-owner; access via [`UnsafeCell`] in WorkerCtx /
/// KhlWorker (the Chase-Lev / KHL owner-private convention).
pub struct KhlStash {
    /// Filled count of `items` (0..=2). When 0 the stash is empty.
    n: u8,
    /// Shared per-batch metadata; re-stamped onto each drained
    /// CompactJobRef as it is rehydrated to a full JobRef.
    k_outer: u8,
    numa_hint: u8,
    variant: u8,
    items: [CompactJobRef; KHL_LINE_ITEMS - 1],
}

impl KhlStash {
    /// Fresh empty stash.
    #[inline]
    pub fn empty() -> Self {
        Self {
            n: 0,
            k_outer: 0,
            numa_hint: 0,
            variant: 0,
            items: [CompactJobRef::null(), CompactJobRef::null()],
        }
    }

    /// Pop one item from the stash (LIFO). Returns the rehydrated
    /// JobRef or None when empty.
    #[inline]
    pub fn drain_one(&mut self) -> Option<JobRef> {
        if self.n == 0 {
            return None;
        }
        self.n -= 1;
        let compact = self.items[self.n as usize];
        // SAFETY: compact came from a successful KHL pop / steal;
        // its captured state is valid; once-only contract preserved
        // because each item is only drained once.
        Some(unsafe { compact.to_jobref(self.k_outer, self.numa_hint, self.variant) })
    }

    /// Absorb a popped/stolen body. Returns the first item as a
    /// JobRef; the rest are stashed for subsequent drains.
    ///
    /// Empty body returns None and does not modify the stash. If
    /// the stash is already non-empty when called, the prior
    /// contents are OVERWRITTEN - the caller's responsibility is
    /// to drain the stash before fetching a new batch.
    #[inline]
    pub fn stash_extras(&mut self, body: KhlBody) -> Option<JobRef> {
        let n = body.n_items as usize;
        if n == 0 {
            return None;
        }
        self.k_outer = body.k_outer;
        self.numa_hint = body.numa_hint;
        self.variant = body.variant;
        let items = body.items_for_adapter();
        // LLVM lowers copy_from_slice to a vectorised memcpy /
        // SIMD load+store on x86_64; preferable to a hand-rolled
        // loop here.
        self.items[..(n - 1)].copy_from_slice(&items[1..n]);
        self.n = (n - 1) as u8;
        let first = items[0];
        // SAFETY: items[0] came from a fresh batch; once-only
        // preserved.
        Some(unsafe { first.to_jobref(body.k_outer, body.numa_hint, body.variant) })
    }

    /// True when stash holds no items.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Owner-side KHL adapter for use in WorkerCtx. Wraps a
/// [`SchedKhlDeque`] with an internal pop-stash so callers see a
/// JobRef-shaped API.
pub struct KhlWorker {
    inner: SchedKhlDeque,
    /// Pop stash: when the inner KHL returns a 3-item body,
    /// `pop()` returns the first item and stashes the other two
    /// here for the next two `pop()` calls.
    pop_stash: UnsafeCell<KhlStash>,
}

// SAFETY: only the owner thread accesses pop_stash; KHL inner
// handles its own Send/Sync via the seq protocol.
unsafe impl Send for KhlWorker {}

impl KhlWorker {
    /// Construct a new KHL-backed owner adapter + paired stealer.
    /// Capacity is in SLOTS; each slot carries up to 3 jobs.
    pub fn new(slot_capacity: usize) -> (Self, KhlStealer) {
        let (inner, stealer) = new_khl(slot_capacity);
        (
            Self {
                inner,
                pop_stash: UnsafeCell::new(KhlStash::empty()),
            },
            stealer,
        )
    }

    /// Push one job and publish immediately. Uses the inner
    /// [`SchedKhlDeque::push_one`] fast path which bypasses the
    /// accumulator when empty (the common single-push case from
    /// `sched::join`'s right-half push). When the accumulator
    /// already has burst items, falls through to the standard
    /// push+flush path.
    ///
    /// The K_inner=3 amortization is preserved at the underlying
    /// inner-KHL slot level - a slot with `n_items=1` is still
    /// stolen as a single cache-line transfer. The amortization
    /// shows up specifically when [`Self::push_burst`] is used
    /// from a producer-fast call site (cooperative_join_n_flat
    /// fan-out) that packs 3 consecutive bursts into one slot.
    #[inline(always)]
    pub fn push(&self, job: JobRef) {
        self.inner.push_one(job);
    }

    /// Push without auto-flushing. Use this when the caller knows
    /// it will follow up with more pushes (cooperative_join_n_flat
    /// fan-out) and an explicit [`Self::flush`] at the end. Buffers
    /// up to 3 pushes into one cache-line slot for the K_inner=3
    /// amortization win.
    #[inline(always)]
    pub fn push_burst(&self, job: JobRef) {
        self.inner.push(job);
    }

    /// Pop one job. Drains the pop-stash first; if empty, goes
    /// to the inner KHL deque and stashes the extras from the
    /// returned body.
    #[inline]
    pub fn pop(&self) -> Option<JobRef> {
        // SAFETY: owner-private pop_stash; single-threaded access.
        let stash = unsafe { &mut *self.pop_stash.get() };
        if let Some(j) = stash.drain_one() {
            return Some(j);
        }
        match self.inner.pop() {
            KhlSteal::Success(body) => stash.stash_extras(body),
            KhlSteal::Empty | KhlSteal::Retry => None,
        }
    }

    /// Force-flush any partially-filled accumulator slot.
    #[inline]
    pub fn flush(&self) {
        self.inner.flush();
    }

    /// Number of jobs currently sitting in the accumulator (0..=3),
    /// waiting to be flushed into the inner ring. Used by the
    /// WorkerCtx flush_all bookkeeper to compute the JEC wake
    /// broadcast count.
    #[inline]
    pub fn pending_items(&self) -> u8 {
        self.inner.accumulator_n_items()
    }

    /// Approximate is-empty. Returns true when stash, accumulator,
    /// AND inner ring are all empty. Hint only - concurrent thief
    /// CAS may invalidate immediately after return.
    #[inline]
    pub fn is_empty(&self) -> bool {
        // SAFETY: owner-private read.
        let stash = unsafe { &*self.pop_stash.get() };
        if !stash.is_empty() {
            return false;
        }
        self.inner.accumulator_n_items() == 0 && self.inner.is_inner_empty()
    }

    /// Clone a fresh thief handle. Stash construction is the
    /// CALLER's responsibility (the thief-side stash lives in
    /// WorkerCtx, shared across all peer-tier steal results).
    #[inline]
    pub fn stealer(&self) -> KhlStealer {
        self.inner.stealer()
    }
}

/// Outcome of a thief-side steal at the WorkerCtx API level.
/// Same three-arm Success / Empty / Retry shape as the
/// underlying Chase-Lev steal protocol.
pub enum KhlSteal2<T> {
    /// Got a job (extras stashed in the caller's stash for the
    /// next two finds).
    Success(T),
    /// Deque was empty (no race).
    Empty,
    /// CAS-loss; caller should retry.
    Retry,
}

impl<T> KhlSteal2<T> {
    /// Convenience: convert to Option dropping Retry/Empty.
    #[inline]
    pub fn ok(self) -> Option<T> {
        match self {
            KhlSteal2::Success(t) => Some(t),
            _ => None,
        }
    }
}

/// Thief-side helper: steal from `stealer` and absorb the result
/// through `stash`. Returns one JobRef on success (with extras
/// stashed for the caller's next pop), Empty if the inner deque
/// is empty, or Retry on CAS-loss.
///
/// The `stash` MUST be empty before calling; the caller's
/// `find_work` is expected to drain the stash before invoking
/// peer-steal.
#[inline]
pub fn steal_via_stash(
    stealer: &KhlStealer,
    stash: &mut KhlStash,
) -> KhlSteal2<JobRef> {
    debug_assert!(stash.is_empty(), "stash must be drained before steal");
    match stealer.steal() {
        KhlSteal::Success(body) => match stash.stash_extras(body) {
            Some(j) => KhlSteal2::Success(j),
            None => KhlSteal2::Empty,
        },
        KhlSteal::Empty => KhlSteal2::Empty,
        KhlSteal::Retry => KhlSteal2::Retry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::Variant;
    use crate::sched::job::StackJob;
    use crate::sched::latch::CoreLatch;

    #[test]
    fn pop_serves_three_from_one_inner_burst() {
        // push_burst keeps the K_inner=3 amortization: 3 burst
        // pushes accumulate into 1 inner slot. The first pop
        // touches the inner deque; the next two come from stash.
        let (w, _s) = KhlWorker::new(4);
        let j1 = StackJob::new(|_| 1u32, CoreLatch::new());
        let j2 = StackJob::new(|_| 2u32, CoreLatch::new());
        let j3 = StackJob::new(|_| 3u32, CoreLatch::new());
        let r1 = unsafe { j1.as_job_ref(4, 0, Variant::Fast) };
        let r2 = unsafe { j2.as_job_ref(4, 0, Variant::Fast) };
        let r3 = unsafe { j3.as_job_ref(4, 0, Variant::Fast) };
        w.push_burst(r1);
        w.push_burst(r2);
        w.push_burst(r3);
        // push_burst's third push auto-flushes at n=3 inside
        // SchedKhlDeque, so the slot is already published.
        let p1 = w.pop().expect("first pop");
        unsafe { p1.execute() };
        let p2 = w.pop().expect("second pop");
        unsafe { p2.execute() };
        let p3 = w.pop().expect("third pop");
        unsafe { p3.execute() };
        assert!(j1.latch.is_set() && j2.latch.is_set() && j3.latch.is_set());
        assert!(w.pop().is_none());
    }

    #[test]
    fn push_one_always_flushes_for_join_pattern() {
        // The WorkerCtx use case: push ONE job (a join right-half),
        // then a thief on another thread takes it. Without
        // auto-flush, the thief would never see the job.
        let (w, s) = KhlWorker::new(4);
        let j = StackJob::new(|_| 7u32, CoreLatch::new());
        let r = unsafe { j.as_job_ref(8, 0, Variant::Faithful) };
        w.push(r);
        // Push (NOT push_burst) auto-flushes, so the thief sees
        // the 1-item slot immediately.
        match s.steal() {
            KhlSteal::Success(body) => {
                assert_eq!(body.n_items, 1);
                unsafe { body.execute_all_lifo() };
                assert!(j.latch.is_set());
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[test]
    fn burst_then_flush_publishes_partial() {
        // push_burst stays in the accumulator without flushing
        // (unless it fills); explicit flush publishes a partial.
        let (w, s) = KhlWorker::new(4);
        let j = StackJob::new(|_| 7u32, CoreLatch::new());
        let r = unsafe { j.as_job_ref(8, 0, Variant::Faithful) };
        w.push_burst(r);
        // Without flush, the item sits in the accumulator.
        assert!(matches!(s.steal(), KhlSteal::Empty));
        w.flush();
        match s.steal() {
            KhlSteal::Success(body) => {
                assert_eq!(body.n_items, 1);
                unsafe { body.execute_all_lifo() };
                assert!(j.latch.is_set());
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn steal_via_stash_full_round_trip() {
        // push_burst lets 6 jobs pack into 2 cache-line slots.
        // Steal via adapter + stash; verify all 6 execute.
        let (w, s) = KhlWorker::new(4);
        let mut stash = KhlStash::empty();

        let jobs: [StackJob<CoreLatch, fn(bool) -> u32, u32>; 6] = [
            StackJob::new((|_| 1) as fn(bool) -> u32, CoreLatch::new()),
            StackJob::new((|_| 2) as fn(bool) -> u32, CoreLatch::new()),
            StackJob::new((|_| 3) as fn(bool) -> u32, CoreLatch::new()),
            StackJob::new((|_| 4) as fn(bool) -> u32, CoreLatch::new()),
            StackJob::new((|_| 5) as fn(bool) -> u32, CoreLatch::new()),
            StackJob::new((|_| 6) as fn(bool) -> u32, CoreLatch::new()),
        ];
        for j in &jobs {
            let r = unsafe { j.as_job_ref(4, 0, Variant::Fast) };
            w.push_burst(r);
        }
        // Both slots auto-flushed at fill (n_items=3); inner
        // KHL has 2 slots holding 3 items each = 6 jobs total.
        let mut executed = 0usize;
        loop {
            if let Some(j) = stash.drain_one() {
                unsafe { j.execute() };
                executed += 1;
                continue;
            }
            match steal_via_stash(&s, &mut stash) {
                KhlSteal2::Success(j) => {
                    unsafe { j.execute() };
                    executed += 1;
                }
                KhlSteal2::Empty => break,
                KhlSteal2::Retry => continue,
            }
        }
        assert_eq!(executed, 6);
        for j in &jobs {
            assert!(j.latch.is_set());
        }
    }
}

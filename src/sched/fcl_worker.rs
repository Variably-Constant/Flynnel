//! WorkerCtx-shaped adapter over [`crate::sched::fcl_local`].
//!
//! Mirrors [`crate::sched::khl_worker::KhlWorker`] but wraps
//! [`crate::sched::fcl_local::SchedFclDeque`] (counter-only
//! Chase-Lev family at K_inner=3). Used by the adaptive WorkerCtx
//! when the host's K_gating calibration prefers CounterOnly
//! (smaller-store-buffer cores).
//!
//! The K_inner=3 batching benefit is identical between Fcl and
//! KHL; the K_gating axis chooses how the publication signal is
//! transported (one shared bottom counter for Fcl, distributed
//! per-slot seq atomics for KHL).

#![allow(clippy::missing_errors_doc)]

use core::cell::UnsafeCell;

use crate::sched::chase_lev_local;
use crate::sched::fcl_local::{FclStealer, PoppedSlot, SchedFclDeque, new_fcl};
use crate::sched::job::{CompactJobRef, JobRef};
use crate::sched::khl_local::KHL_LINE_ITEMS;

/// Per-worker batch stash for the Fcl adapter (mirror of KhlStash
/// in shape; 0..=2 leftover items after a PoppedSlot drain).
pub struct FclStash {
    n: u8,
    k_outer: u8,
    numa_hint: u8,
    variant: u8,
    items: [CompactJobRef; KHL_LINE_ITEMS - 1],
}

impl FclStash {
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

    /// Drain one item LIFO.
    #[inline]
    pub fn drain_one(&mut self) -> Option<JobRef> {
        if self.n == 0 {
            return None;
        }
        self.n -= 1;
        let compact = self.items[self.n as usize];
        // SAFETY: compact came from a fresh PoppedSlot; once-only
        // preserved across drain calls.
        Some(unsafe { compact.to_jobref(self.k_outer, self.numa_hint, self.variant) })
    }

    /// Absorb a popped/stolen PoppedSlot. Returns first item; rest
    /// stash for subsequent drains.
    #[inline]
    pub fn stash_extras(&mut self, slot: PoppedSlot) -> Option<JobRef> {
        let n = slot.n_items as usize;
        if n == 0 {
            return None;
        }
        self.k_outer = slot.k_outer;
        self.numa_hint = slot.numa_hint;
        self.variant = slot.variant;
        // The items in PoppedSlot are pub(crate); use the slot's
        // execute_all_lifo-compatible access path. Here we copy
        // items[0..n] out via a small temp because the items field
        // is private to fcl_local. The adapter is in flynnel so
        // pub(crate) is visible.
        let items_arr = slot.compact_items_for_adapter();
        self.items[..(n - 1)].copy_from_slice(&items_arr[1..n]);
        self.n = (n - 1) as u8;
        let first = items_arr[0];
        // SAFETY: items[0] from fresh slot; once-only preserved.
        Some(unsafe { first.to_jobref(slot.k_outer, slot.numa_hint, slot.variant) })
    }

    /// True when stash holds no items.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Owner-side Fcl adapter. Wraps a [`SchedFclDeque`] with an
/// internal pop-stash so callers see a JobRef-shaped API
/// (mirrors `KhlWorker` so the two are interchangeable at the
/// WorkerCtx tagged-enum dispatch site).
pub struct FclWorker {
    inner: SchedFclDeque,
    pop_stash: UnsafeCell<FclStash>,
}

// SAFETY: only the owner thread accesses pop_stash; inner handles
// its own Send/Sync via Chase-Lev.
unsafe impl Send for FclWorker {}

impl FclWorker {
    /// Construct a new Fcl-backed owner adapter + paired stealer.
    /// Capacity is in SLOTS; each slot carries up to 3 jobs.
    pub fn new(slot_capacity: usize) -> (Self, FclStealer) {
        let (inner, stealer) = new_fcl(slot_capacity);
        (
            Self {
                inner,
                pop_stash: UnsafeCell::new(FclStash::empty()),
            },
            stealer,
        )
    }

    /// Push one job with auto-flush (single-push API for join sites).
    #[inline(always)]
    pub fn push(&self, job: JobRef) {
        self.inner.push(job);
        self.inner.flush();
    }

    /// Non-blocking single push; `Err(job)` when the inner deque is
    /// full instead of spinning for a thief.
    #[inline]
    pub fn try_push(&self, job: JobRef) -> Result<(), JobRef> {
        self.inner.try_push_one(job)
    }

    /// Push without auto-flush (burst API for cooperative fan-out).
    #[inline(always)]
    pub fn push_burst(&self, job: JobRef) {
        self.inner.push(job);
    }

    /// Flush any partially-filled accumulator slot.
    #[inline]
    pub fn flush(&self) {
        self.inner.flush();
    }

    /// Pop one job. Drains pop-stash first; if empty, fetches a
    /// slot from the inner Fcl and stashes the extras.
    #[inline]
    pub fn pop(&self) -> Option<JobRef> {
        // SAFETY: owner-private pop_stash.
        let stash = unsafe { &mut *self.pop_stash.get() };
        if let Some(j) = stash.drain_one() {
            return Some(j);
        }
        match self.inner.pop() {
            Some(slot) => stash.stash_extras(slot),
            None => None,
        }
    }

    /// Approximate is-empty hint.
    #[inline]
    pub fn is_empty(&self) -> bool {
        // SAFETY: owner-private read.
        let stash = unsafe { &*self.pop_stash.get() };
        stash.is_empty()
        // Inner emptiness is not exposed; conservative hint.
    }

    /// Number of items currently buffered in the accumulator
    /// (0..=3). Used by WorkerCtx::flush_all for the JEC wake
    /// broadcast count.
    #[inline]
    pub fn pending_items(&self) -> u8 {
        // Accumulator size isn't exposed on SchedFclDeque; the
        // adapter conservatively reports 0. The flush_all path
        // uses the burst counter as the canonical pending-count
        // for JEC wake purposes; this method exists for API
        // parity with KhlWorker.
        0
    }

    /// Clone a fresh thief handle.
    #[inline]
    pub fn stealer(&self) -> FclStealer {
        self.inner.stealer()
    }
}

/// Outcome of a thief-side Fcl steal at the WorkerCtx API level.
/// Mirrors KhlSteal2.
pub enum FclSteal2<T> {
    /// Got a job; extras stashed.
    Success(T),
    /// Deque empty.
    Empty,
    /// CAS-loss; caller should retry.
    Retry,
}

/// Thief-side helper: steal one slot via Fcl + absorb extras into
/// caller's stash. Mirrors `khl_worker::steal_via_stash`.
#[inline]
pub fn steal_via_stash(
    stealer: &FclStealer,
    stash: &mut FclStash,
) -> FclSteal2<JobRef> {
    debug_assert!(stash.is_empty(), "stash must be drained before steal");
    match stealer.steal() {
        chase_lev_local::Steal::Success(slot) => match stash.stash_extras(slot) {
            Some(j) => FclSteal2::Success(j),
            None => FclSteal2::Empty,
        },
        chase_lev_local::Steal::Empty => FclSteal2::Empty,
        chase_lev_local::Steal::Retry => FclSteal2::Retry,
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
        let (w, _s) = FclWorker::new(4);
        let j1 = StackJob::new(|_| 1u32, CoreLatch::new());
        let j2 = StackJob::new(|_| 2u32, CoreLatch::new());
        let j3 = StackJob::new(|_| 3u32, CoreLatch::new());
        let r1 = unsafe { j1.as_job_ref(4, 0, Variant::Fast) };
        let r2 = unsafe { j2.as_job_ref(4, 0, Variant::Fast) };
        let r3 = unsafe { j3.as_job_ref(4, 0, Variant::Fast) };
        w.push_burst(r1);
        w.push_burst(r2);
        w.push_burst(r3);
        // push_burst auto-flushes at 3 via SchedFclDeque.
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
    fn push_always_flushes_for_join_pattern() {
        let (w, s) = FclWorker::new(4);
        let j = StackJob::new(|_| 7u32, CoreLatch::new());
        let r = unsafe { j.as_job_ref(8, 0, Variant::Faithful) };
        w.push(r);
        // Push auto-flushes; the stealer sees the 1-item slot.
        match s.steal() {
            chase_lev_local::Steal::Success(slot) => {
                assert_eq!(slot.n_items, 1);
                unsafe { slot.execute_all_lifo() };
                assert!(j.latch.is_set());
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }
}

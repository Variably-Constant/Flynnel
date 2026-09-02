//! Fcl (Fat Chase-Lev): K_inner=3 Chase-Lev with shared counter.
//!
//! Wraps [`crate::sched::chase_lev_local`] with a 3-job batched slot
//! type ([`FatSlot`]) so each push from the producer's perspective
//! accumulates into a 64-byte cache-line slot that ultimately costs
//! one Chase-Lev push + one cache-line steal-coherence-transfer to
//! deliver 3 jobs to a thief. The amortization win: per-item
//! Chase-Lev cost is divided by 3.
//!
//! ## When Fcl wins
//!
//! Producer-fast patterns where the owner emits many jobs rapidly:
//! - Recursive splits (parallel for, divide-and-conquer reductions)
//! - Batched task submission (loop-body parallelism)
//! - Burst-y push from a single-thread driver
//!
//! ## When Fcl does NOT win (and falls back gracefully)
//!
//! Single-push call sites (classic `join(a, b)` where the parent
//! pushes ONE right-half then runs the left half inline) get no
//! batching benefit. The single right-half sits unflushed in the
//! owner buffer where thieves cannot see it; we provide an explicit
//! [`SchedFclDeque::flush`] for sites that know to publish before
//! waiting.
//!
//! The adaptive WorkerCtx carries both Fcl and KHL backings and
//! swaps by tag.
//!
//! ## Slot layout (exactly 64 bytes)
//!
//! ```text
//! +----+----+----+----+--------+---------------------------------+
//! | n  | ko | nh | va | _pad 12| items: [CompactJobRef; 3] (48B) |
//! +----+----+----+----+--------+---------------------------------+
//! ```
//!
//! `n_items` counts how many of the 3 slots are filled (1..=3).
//! `k_outer / numa_hint / variant` are shared across the slot's 3
//! jobs: recursive splits emit children with identical metadata,
//! and the shared header is what lets 3 jobs fit one 64-byte line.
//!
//! Slot is `#[repr(C, align(64))]` so it always starts on a cache-
//! line boundary.

#![allow(clippy::missing_errors_doc)]

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

use crate::sched::chase_lev_local::{self, new_chase_lev};
use crate::sched::job::{CompactJobRef, JobRef};

/// Items per Fcl slot. 3 is the cache-line-fit number for K_inner=3
/// (16-byte CompactJobRef * 3 = 48 bytes + 16-byte header = 64 B).
pub const FCL_LINE_ITEMS: usize = 3;

/// Fat Chase-Lev slot: shared metadata header + 3 inline
/// [`CompactJobRef`]s. Exactly 64 bytes.
#[repr(C, align(64))]
pub(crate) struct FatSlot {
    /// Filled count, 1..=3. The owner-side accumulator increments
    /// this as items arrive; on flush it stamps the slot into the
    /// underlying Chase-Lev deque.
    pub(crate) n_items: u8,
    /// `K_outer = log2(n_limbs)` for all jobs in this slot. Shared
    /// per the handoff's shared-header design.
    pub(crate) k_outer: u8,
    /// NUMA hint for all jobs in this slot.
    pub(crate) numa_hint: u8,
    /// Variant for all jobs in this slot.
    pub(crate) variant: u8,
    /// Padding so the items array starts at offset 16.
    pub(crate) _pad: [u8; 12],
    /// Compact-ref payload. `items[..n_items]` are valid; the rest
    /// are uninitialized slack (the pop / steal path honors
    /// `n_items` to avoid touching them).
    pub(crate) items: [CompactJobRef; FCL_LINE_ITEMS],
}

impl FatSlot {
    /// Construct an empty slot with `n_items = 0` and the named
    /// shared metadata. Used to initialize the owner accumulator.
    #[inline]
    pub(crate) fn empty(k_outer: u8, numa_hint: u8, variant: u8) -> Self {
        Self {
            n_items: 0,
            k_outer,
            numa_hint,
            variant,
            _pad: [0u8; 12],
            // Initialize with null sentinels; the pop / steal path
            // never touches indices >= n_items.
            items: [
                CompactJobRef::null(),
                CompactJobRef::null(),
                CompactJobRef::null(),
            ],
        }
    }
}

/// Snapshot of a popped or stolen slot. The caller is responsible
/// for executing each `items[i]` for `i < n_items` exactly once.
pub struct PoppedSlot {
    /// Number of valid items in `items` (1..=3). Items beyond this
    /// index are unspecified.
    pub n_items: u8,
    /// Shared `k_outer` metadata for the slot's items.
    pub k_outer: u8,
    /// Shared NUMA hint metadata for the slot's items.
    pub numa_hint: u8,
    /// Shared variant metadata for the slot's items.
    pub variant: u8,
    pub(crate) items: [CompactJobRef; FCL_LINE_ITEMS],
}

impl core::fmt::Debug for PoppedSlot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PoppedSlot")
            .field("n_items", &self.n_items)
            .field("k_outer", &self.k_outer)
            .field("numa_hint", &self.numa_hint)
            .field("variant", &self.variant)
            .finish_non_exhaustive()
    }
}

impl PoppedSlot {
    #[inline]
    fn from_slot(s: FatSlot) -> Self {
        Self {
            n_items: s.n_items,
            k_outer: s.k_outer,
            numa_hint: s.numa_hint,
            variant: s.variant,
            items: s.items,
        }
    }

    /// Crate-internal accessor for the underlying compact-ref
    /// array. Used by the [`super::fcl_worker::FclStash`] adapter
    /// to drain stash items without going through the LIFO-execute
    /// path. Caller must respect `n_items` to avoid touching
    /// padding slots.
    #[inline]
    pub(crate) fn compact_items_for_adapter(&self) -> &[CompactJobRef; FCL_LINE_ITEMS] {
        &self.items
    }

    /// Execute every item in this slot in LIFO order
    /// (`items[n_items-1]` first). Useful in single-thief drain
    /// loops; multi-thief code may execute in any order.
    ///
    /// # Safety
    ///
    /// Each item's captured-state pointer must still be valid; the
    /// slot's underlying job handles must not have been executed
    /// already. PoppedSlot is single-use - call this exactly once
    /// per slot.
    #[inline]
    pub unsafe fn execute_all_lifo(self) {
        for i in (0..self.n_items as usize).rev() {
            // SAFETY: items[i] is in the active prefix; the
            // caller-side contract on PoppedSlot says each item is
            // executed exactly once by execute_all_lifo.
            unsafe { self.items[i].execute() }
        }
    }
}

/// Fcl deque: owner accumulator + underlying
/// [`chase_lev_local::Worker`].
pub struct SchedFclDeque {
    inner: chase_lev_local::Worker<FatSlot>,
    /// Owner-side accumulator. Single-owner so UnsafeCell suffices;
    /// access is single-threaded by the Chase-Lev owner invariant.
    accumulator: UnsafeCell<MaybeUninit<FatSlot>>,
}

// SAFETY: only the owner thread accesses `accumulator`; Chase-Lev's
// owner-private invariant makes the UnsafeCell access single-
// threaded. The underlying Worker handles its own thread-safety.
unsafe impl Send for SchedFclDeque {}

/// Thief handle for a [`SchedFclDeque`]. Clonable.
pub struct FclStealer {
    inner: chase_lev_local::Stealer<FatSlot>,
}

impl Clone for FclStealer {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Construct a new Fcl deque with `capacity` slots (each slot holds
/// up to [`FCL_LINE_ITEMS`] jobs). Returns owner + one thief
/// handle.
pub fn new_fcl(capacity: usize) -> (SchedFclDeque, FclStealer) {
    let (w, s) = new_chase_lev::<FatSlot>(capacity);
    let acc = UnsafeCell::new(MaybeUninit::new(FatSlot::empty(0, 0, 0)));
    (
        SchedFclDeque {
            inner: w,
            accumulator: acc,
        },
        FclStealer { inner: s },
    )
}

impl SchedFclDeque {
    /// Buffer a `JobRef` into the owner accumulator. When the
    /// accumulator fills (`n_items == 3`) the slot is flushed to
    /// the underlying Chase-Lev. Infallible: on inner-full, spins
    /// until a thief frees capacity.
    ///
    /// Back-pressure rationale: the inner deque is owner-private,
    /// sized at construction. The only way it fills is if thieves
    /// cannot keep up; spinning at the push site hands time
    /// naturally to thieves. The bench harness sizes inner
    /// capacity to absorb typical bursts.
    ///
    /// The first push of a fresh accumulator stamps the slot's
    /// shared metadata from the JobRef. Subsequent pushes inherit
    /// that metadata; if the caller pushes jobs with different
    /// metadata, the first push wins. (The metadata is dispatch
    /// hints; execute() never reads it.)
    #[inline(always)]
    pub fn push(&self, job: JobRef) {
        // SAFETY: owner-private accumulator; single-threaded access
        // by Chase-Lev convention.
        let acc = unsafe { (*self.accumulator.get()).assume_init_mut() };
        if acc.n_items == 0 {
            acc.k_outer = job.k_outer;
            acc.numa_hint = job.numa_hint;
            acc.variant = job.variant;
        }
        let idx = acc.n_items as usize;
        // SAFETY: idx == n_items (pre-increment); n_items < 3 by
        // the auto-flush below resetting to 0; bounds-check-free.
        unsafe { *acc.items.get_unchecked_mut(idx) = job.compact(); }
        acc.n_items += 1;
        // JobRef has no Drop; dropping `job` here is byte-discard
        // with no cleanup. Captured-state ownership has moved into
        // the accumulator's compact slot.
        if acc.n_items == FCL_LINE_ITEMS as u8 {
            let slot = core::mem::replace(acc, FatSlot::empty(0, 0, 0));
            self.flush_slot(slot);
        }
    }

    /// Cold path: flush a full slot to the inner Chase-Lev, spin
    /// on inner-full. Hoisted out of `push` so the hot path stays
    /// linear in the instruction stream.
    #[cold]
    #[inline(never)]
    fn flush_slot(&self, slot: FatSlot) {
        let mut s = slot;
        loop {
            match self.inner.push(s) {
                Ok(()) => return,
                Err(retry) => {
                    s = retry;
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Force-flush any buffered items to the underlying Chase-Lev.
    /// Infallible: spins on inner-full like `push` does (same
    /// back-pressure semantics).
    #[inline]
    pub fn flush(&self) {
        // SAFETY: owner-private accumulator.
        let acc = unsafe { (*self.accumulator.get()).assume_init_mut() };
        if acc.n_items == 0 {
            return;
        }
        let slot = core::mem::replace(acc, FatSlot::empty(0, 0, 0));
        self.flush_slot(slot);
    }

    /// Pop one slot's worth of work. Drains the owner accumulator
    /// first (LIFO ordering across slots: most-recent push first),
    /// then falls through to the underlying Chase-Lev.
    pub fn pop(&self) -> Option<PoppedSlot> {
        // SAFETY: owner-private accumulator.
        let acc = unsafe { (*self.accumulator.get()).assume_init_mut() };
        if acc.n_items > 0 {
            let slot = core::mem::replace(acc, FatSlot::empty(0, 0, 0));
            return Some(PoppedSlot::from_slot(slot));
        }
        match self.inner.pop() {
            chase_lev_local::Steal::Success(s) => Some(PoppedSlot::from_slot(s)),
            chase_lev_local::Steal::Empty | chase_lev_local::Steal::Retry => None,
        }
    }

    /// Clone a fresh thief handle.
    #[inline]
    pub fn stealer(&self) -> FclStealer {
        FclStealer {
            inner: self.inner.stealer(),
        }
    }
}

impl Drop for SchedFclDeque {
    fn drop(&mut self) {
        // Drop the accumulator's FatSlot in place so any held items
        // run their destructors (compact refs are pointer-only and
        // have no Drop; the slot itself is POD-shaped, so this is
        // mostly hygiene for the future where slots might carry
        // owned data).
        //
        // SAFETY: owner-private accumulator; we hold &mut self so no
        // concurrent access is possible.
        unsafe {
            (*self.accumulator.get()).assume_init_drop();
        }
    }
}

impl FclStealer {
    /// Steal one slot's worth of work. The slot may carry 1-3
    /// items; caller honors `n_items`.
    #[inline]
    pub fn steal(&self) -> chase_lev_local::Steal<PoppedSlot> {
        match self.inner.steal() {
            chase_lev_local::Steal::Success(s) => {
                chase_lev_local::Steal::Success(PoppedSlot::from_slot(s))
            }
            chase_lev_local::Steal::Empty => chase_lev_local::Steal::Empty,
            chase_lev_local::Steal::Retry => chase_lev_local::Steal::Retry,
        }
    }

    /// Architectural prefetch hint for the next steal target.
    #[inline]
    pub fn prefetch_for_steal(&self) {
        self.inner.prefetch_for_steal();
    }

    /// Approximate is-empty snapshot. Same semantics as the
    /// underlying [`chase_lev_local::Stealer::is_empty`]: a hint,
    /// not a guarantee, since concurrent activity may invalidate
    /// the snapshot immediately.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_is_exactly_one_cache_line() {
        assert_eq!(core::mem::size_of::<FatSlot>(), 64,
            "FatSlot must be exactly 64 bytes (one cache line)");
        assert_eq!(core::mem::align_of::<FatSlot>(), 64,
            "FatSlot must be 64-byte aligned");
    }

    #[test]
    fn compact_jobref_is_two_words() {
        assert_eq!(core::mem::size_of::<CompactJobRef>(), 16,
            "CompactJobRef must be 16 bytes");
    }

    // ---- Functional tests using a fake JobRef-shaped payload ---

    use crate::sched::job::StackJob;
    use crate::sched::latch::CoreLatch;
    use crate::foundation::Variant;

    #[test]
    fn push_single_then_flush_then_steal() {
        let (deque, stealer) = new_fcl(4);
        let job = StackJob::new(|_stolen| 7u32, CoreLatch::new());
        let r = unsafe { job.as_job_ref(8, 0, Variant::Faithful) };
        deque.push(r);
        // Without flush, the item sits in the accumulator; steal
        // sees nothing.
        assert!(matches!(stealer.steal(), chase_lev_local::Steal::Empty));
        deque.flush();
        // Now the slot is in the deque; steal should return it.
        match stealer.steal() {
            chase_lev_local::Steal::Success(s) => {
                assert_eq!(s.n_items, 1);
                // Execute and verify latch.
                unsafe { s.execute_all_lifo(); }
                assert!(job.latch.is_set());
                let result = unsafe { job.into_result() };
                assert_eq!(result, 7);
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[test]
    fn buffer_fills_to_3_then_auto_flushes() {
        let (deque, stealer) = new_fcl(4);
        let j1 = StackJob::new(|_| 1u32, CoreLatch::new());
        let j2 = StackJob::new(|_| 2u32, CoreLatch::new());
        let j3 = StackJob::new(|_| 3u32, CoreLatch::new());
        let r1 = unsafe { j1.as_job_ref(8, 0, Variant::Faithful) };
        let r2 = unsafe { j2.as_job_ref(8, 0, Variant::Faithful) };
        let r3 = unsafe { j3.as_job_ref(8, 0, Variant::Faithful) };
        deque.push(r1);
        deque.push(r2);
        // First two are buffered; steal still sees empty.
        assert!(matches!(stealer.steal(), chase_lev_local::Steal::Empty));
        deque.push(r3);
        // Third push triggers auto-flush; the slot is in the inner deque.
        match stealer.steal() {
            chase_lev_local::Steal::Success(s) => {
                assert_eq!(s.n_items, 3);
                unsafe { s.execute_all_lifo(); }
                assert!(j1.latch.is_set());
                assert!(j2.latch.is_set());
                assert!(j3.latch.is_set());
                assert_eq!(unsafe { j1.into_result() }, 1);
                assert_eq!(unsafe { j2.into_result() }, 2);
                assert_eq!(unsafe { j3.into_result() }, 3);
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[test]
    fn owner_pop_drains_accumulator_first_lifo() {
        let (deque, _stealer) = new_fcl(4);
        let j1 = StackJob::new(|_| 10u32, CoreLatch::new());
        let j2 = StackJob::new(|_| 20u32, CoreLatch::new());
        let r1 = unsafe { j1.as_job_ref(4, 0, Variant::Fast) };
        let r2 = unsafe { j2.as_job_ref(4, 0, Variant::Fast) };
        deque.push(r1);
        deque.push(r2);
        // Owner pop drains the accumulator (n_items=2).
        match deque.pop() {
            Some(slot) => {
                assert_eq!(slot.n_items, 2);
                // Execute in LIFO: item 1 = r2, item 0 = r1.
                unsafe { slot.execute_all_lifo(); }
                assert!(j1.latch.is_set());
                assert!(j2.latch.is_set());
            }
            None => panic!("expected Some(slot)"),
        }
        // Subsequent pop is empty.
        assert!(deque.pop().is_none());
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn batched_push_then_steal_round_trip() {
        // Producer-fast pattern: push 9 items (3 batches),
        // steal them as 3 slots.
        let (deque, stealer) = new_fcl(8);
        let jobs: Vec<StackJob<CoreLatch, fn(bool) -> u32, u32>> =
            (0..9).map(|_| StackJob::new(default_job_fn as fn(bool) -> u32, CoreLatch::new())).collect();
        for j in &jobs {
            let r = unsafe { j.as_job_ref(4, 0, Variant::Fast) };
            deque.push(r);
        }
        // After 9 pushes, accumulator empty (3 auto-flushes).
        // Three slots are in the inner deque.
        let mut executed = 0;
        for _ in 0..3 {
            match stealer.steal() {
                chase_lev_local::Steal::Success(s) => {
                    assert_eq!(s.n_items, 3);
                    executed += s.n_items as usize;
                    unsafe { s.execute_all_lifo(); }
                }
                other => panic!("expected Success, got {other:?}"),
            }
        }
        assert_eq!(executed, 9);
        // No more slots.
        assert!(matches!(stealer.steal(), chase_lev_local::Steal::Empty));
    }

    fn default_job_fn(_stolen: bool) -> u32 { 0 }
}

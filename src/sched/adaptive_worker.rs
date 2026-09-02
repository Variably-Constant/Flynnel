//! Adaptive WorkerCtx primitive: tagged-enum dispatch between
//! K_gating::PerSlot (KHL) and K_gating::CounterOnly (Fcl) at
//! runtime with **zero per-op overhead**.
//!
//! Per-op cost measured on Zen+ R7 2700: direct call 10.96 ns,
//! AtomicU32 tag + enum match 10.98 ns (+0.02 ns, noise),
//! arc_swap + dyn-trait 29.0 ns (+163%). On x86 TSO the
//! Acquire-load lowers to a plain MOV and LLVM collapses the
//! statically-known match into the direct-call sequence.
//!
//! `AdaptiveWorker` carries both backings; the tag picks the
//! active one for `push` / `push_burst`, and `pop` drains active
//! first then dormant so items from a recent migration are not
//! lost. [`AdaptiveWorker::migrate_to`] flushes the active
//! accumulator, Release-stores the new tag, and lets the old
//! backing drain over the next pops. Thieves Acquire-read the tag
//! per steal; a racing flip costs at most one wasted steal.

#![allow(clippy::missing_errors_doc)]

use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use crate::sched::fcl_worker::{FclStash, FclSteal2, FclWorker, steal_via_stash as fcl_steal_via_stash};
use crate::sched::job::JobRef;
use crate::sched::k_gating::KGating;
use crate::sched::khl_local::KhlStealer;
use crate::sched::khl_worker::{KhlStash, KhlSteal2, KhlWorker, steal_via_stash as khl_steal_via_stash};
use crate::sched::fcl_local::FclStealer;

/// Active-backing tag: 0 = KHL (PerSlot), 1 = Fcl (CounterOnly).
const ACTIVE_KHL: u32 = 0;
const ACTIVE_FCL: u32 = 1;

/// Linkage confirmation marker. When the binary links this
/// module, `nm <bin> | grep __flynnel_marker` returns this
/// symbol, confirming the adaptive worker push code path is
/// present in the build (not dead-code-eliminated).
#[unsafe(no_mangle)]
pub static __flynnel_marker_adaptive_worker_push: u8 = 0;
/// Linkage confirmation marker for the pop code path. See
/// [`__flynnel_marker_adaptive_worker_push`] for the pattern.
#[unsafe(no_mangle)]
pub static __flynnel_marker_adaptive_worker_pop: u8 = 0;

/// Owner-side adaptive worker. Always carries both KHL and Fcl
/// backings; the `active` tag picks which is used for pushes.
/// Pops drain both backings to handle orphan items from a recent
/// migration.
pub struct AdaptiveWorker {
    /// Active-backing tag shared with all paired stealers via Arc.
    /// Owner Release-stores on migration; thieves Acquire-load on
    /// every steal.
    active: Arc<AtomicU32>,
    khl: KhlWorker,
    fcl: FclWorker,
    /// `true` when the inactive backing may contain orphaned items
    /// from a recent migration. Set to `true` on every `migrate_to`
    /// call; cleared back to `false` when a `pop()` or thief steal
    /// fall-through confirms the inactive backing is empty.
    ///
    /// Shared with all paired AdaptiveStealers via Arc so thief
    /// steals also benefit from the skip-check, not just the
    /// owner-side pop path.
    ///
    /// Flamegraph evidence (2026-06-19): flynnel_ring::pop at
    /// 0.19% SELF + steal_via_stash at 0.40% SELF when the thief
    /// side pays an unconditional dormant fallback; sharing the
    /// flag with stealers reclaims that 0.4%.
    inactive_dirty: Arc<AtomicBool>,
}

// SAFETY: Both backings are individually Send; the AtomicU32 is
// inherently Send + Sync. The owner-private invariant remains
// owner-only via the same WorkerCtx discipline as before.
unsafe impl Send for AdaptiveWorker {}

/// Thief-side adaptive stealer. Carries stealer handles for both
/// backings + a shared reference to the active tag so each steal
/// knows which backing to probe.
pub struct AdaptiveStealer {
    active: Arc<AtomicU32>,
    khl: KhlStealer,
    fcl: FclStealer,
    /// Shared with the paired [`AdaptiveWorker`] -- see that field's
    /// doc for the migration-orphan-drain protocol. Thief steals
    /// check this flag before walking the dormant backing's stealer.
    inactive_dirty: Arc<AtomicBool>,
}

impl Clone for AdaptiveStealer {
    fn clone(&self) -> Self {
        Self {
            active: Arc::clone(&self.active),
            khl: self.khl.clone(),
            fcl: self.fcl.clone(),
            inactive_dirty: Arc::clone(&self.inactive_dirty),
        }
    }
}

impl AdaptiveStealer {
    /// Approximate is-empty across both backings (hint only).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.khl.is_empty() && self.fcl.is_empty()
    }

    /// Unclaimed bodies in the KHL backing (diagnostic hint).
    #[inline]
    pub fn khl_len(&self) -> usize {
        self.khl.len()
    }
}

/// Per-worker thief-side stash that absorbs K_inner=3 batches
/// from EITHER backing. Internally carries one stash per backing;
/// only one is non-empty at a time (the one matching the most
/// recent successful steal).
pub struct AdaptiveStash {
    khl: KhlStash,
    fcl: FclStash,
}

impl AdaptiveStash {
    /// Fresh empty stash.
    #[inline]
    pub fn empty() -> Self {
        Self {
            khl: KhlStash::empty(),
            fcl: FclStash::empty(),
        }
    }

    /// Drain one item from EITHER backing's stash (whichever is
    /// non-empty). Returns None if both are empty.
    #[inline]
    pub fn drain_one(&mut self) -> Option<JobRef> {
        if let Some(j) = self.khl.drain_one() {
            return Some(j);
        }
        self.fcl.drain_one()
    }

    /// True when both backings' stashes are empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.khl.is_empty() && self.fcl.is_empty()
    }
}

/// Construct a new adaptive worker + paired stealer prototype.
/// `slot_capacity` sizes each backing's ring (in slots, each
/// holding up to 3 jobs). `initial_gating` picks which backing
/// starts active: PerSlot/Auto → KHL, CounterOnly → Fcl.
pub fn new_adaptive(
    slot_capacity: usize,
    initial_gating: KGating,
) -> (AdaptiveWorker, AdaptiveStealer) {
    let (khl_w, khl_s) = KhlWorker::new(slot_capacity);
    let (fcl_w, fcl_s) = FclWorker::new(slot_capacity);
    let active = match initial_gating.resolved() {
        KGating::CounterOnly => ACTIVE_FCL,
        // PerSlot is the default for store-buffer-rich silicon
        // (Zen+, Sapphire Rapids+); Auto resolves to PerSlot on
        // those hosts.
        _ => ACTIVE_KHL,
    };
    let active_arc = Arc::new(AtomicU32::new(active));
    let dirty_arc = Arc::new(AtomicBool::new(false));
    (
        AdaptiveWorker {
            active: Arc::clone(&active_arc),
            khl: khl_w,
            fcl: fcl_w,
            inactive_dirty: Arc::clone(&dirty_arc),
        },
        AdaptiveStealer {
            active: active_arc,
            khl: khl_s,
            fcl: fcl_s,
            inactive_dirty: dirty_arc,
        },
    )
}

impl AdaptiveWorker {
    /// Push one job (auto-flush) to the active backing.
    ///
    /// On Zen+ / Sapphire Rapids+ the calibration default is
    /// `ACTIVE_KHL`; the `_` arm covers `ACTIVE_FCL` for hosts
    /// where the calibration prefers CounterOnly. `cold_path()`
    /// tells LLVM the non-KHL arm is the rare side so it lays
    /// out the hot KHL arm as fall-through, keeping the
    /// instruction-cache footprint of the hot path tight.
    #[inline(always)]
    pub fn push(&self, job: JobRef) {
        match self.active.load(Ordering::Acquire) {
            ACTIVE_KHL => self.khl.push(job),
            _ => {
                core::hint::cold_path();
                self.fcl.push(job);
            }
        }
    }

    /// Non-blocking single push to the active backing; `Err(job)`
    /// when that backing is full. The caller runs a refused job
    /// inline rather than waiting for a thief.
    #[inline]
    pub fn try_push(&self, job: JobRef) -> Result<(), JobRef> {
        match self.active.load(Ordering::Acquire) {
            ACTIVE_KHL => self.khl.try_push(job),
            _ => self.fcl.try_push(job),
        }
    }

    /// Push burst (no auto-flush) to the active backing.
    #[inline(always)]
    pub fn push_burst(&self, job: JobRef) {
        match self.active.load(Ordering::Acquire) {
            ACTIVE_KHL => self.khl.push_burst(job),
            _ => {
                core::hint::cold_path();
                self.fcl.push_burst(job);
            }
        }
    }

    /// Flush any partially-filled accumulator on the active
    /// backing.
    #[inline]
    pub fn flush(&self) {
        match self.active.load(Ordering::Acquire) {
            ACTIVE_KHL => self.khl.flush(),
            _ => {
                core::hint::cold_path();
                self.fcl.flush();
            }
        }
    }

    /// Pop one job. Drains the active backing first; consults the
    /// inactive backing only when `inactive_dirty` says it might
    /// have orphan items from a recent migration. Clears the dirty
    /// flag when the fallback pop confirms the inactive is empty.
    #[inline]
    pub fn pop(&self) -> Option<JobRef> {
        let active = self.active.load(Ordering::Acquire);
        let primary = if active == ACTIVE_KHL {
            self.khl.pop()
        } else {
            core::hint::cold_path();
            self.fcl.pop()
        };
        if primary.is_some() {
            return primary;
        }
        // Active backing was empty; only check the inactive backing
        // if migration recently flagged it as potentially dirty.
        if !self.inactive_dirty.load(Ordering::Acquire) {
            return None;
        }
        core::hint::cold_path();
        let fallback = if active == ACTIVE_KHL {
            self.fcl.pop()
        } else {
            self.khl.pop()
        };
        if fallback.is_none() {
            // Inactive backing has fully drained; clear the flag so
            // future steady-state pops skip the fallback poll entirely.
            self.inactive_dirty.store(false, Ordering::Release);
        }
        fallback
    }

    /// Approximate is-empty hint across both backings.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.khl.is_empty() && self.fcl.is_empty()
    }

    /// Pending items count in the active backing's accumulator.
    #[inline]
    pub fn pending_items(&self) -> u8 {
        match self.active.load(Ordering::Acquire) {
            ACTIVE_KHL => self.khl.pending_items(),
            _ => self.fcl.pending_items(),
        }
    }

    /// Migrate to the requested K_gating. Single Release-store
    /// on the active tag; new pushes route to the new backing
    /// starting immediately. Existing items in the old backing
    /// keep getting drained by pops until empty.
    ///
    /// This call is essentially free: one atomic store on a
    /// shared line, no allocations, no copying. The amortized
    /// per-op cost across all subsequent ops is zero.
    ///
    /// Sets `inactive_dirty = true` so subsequent pops will check
    /// the inactive backing for orphaned items (the post-migration
    /// fallback drain).
    #[inline]
    pub fn migrate_to(&self, gating: KGating) {
        let resolved = gating.resolved();
        let target = match resolved {
            KGating::CounterOnly => ACTIVE_FCL,
            _ => ACTIVE_KHL,
        };
        // Flush the currently-active backing before flipping so
        // any buffered burst items become visible (otherwise
        // they'd sit in the now-dormant backing's accumulator
        // until the next migration back).
        let current = self.active.load(Ordering::Acquire);
        if current == ACTIVE_KHL {
            self.khl.flush();
        } else {
            self.fcl.flush();
        }
        // Mark the soon-to-be inactive backing as potentially holding
        // orphaned items. Pops consult this flag and only walk the
        // inactive backing when set, clearing the flag on the first
        // empty observation.
        if current != target {
            self.inactive_dirty.store(true, Ordering::Release);
        }
        self.active.store(target, Ordering::Release);
    }

    /// Current active gating (observability).
    #[inline]
    pub fn active_gating(&self) -> KGating {
        match self.active.load(Ordering::Acquire) {
            ACTIVE_KHL => KGating::PerSlot,
            _ => KGating::CounterOnly,
        }
    }

    /// Expose the shared active-tag Arc. Used by `LocalArena` to
    /// remote-flip all workers' tags on a global K_gating
    /// migration without needing to reach the WorkerCtx inside
    /// each worker thread's stack frame.
    #[inline]
    pub fn active_tag(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.active)
    }

    /// Clone a fresh stealer paired with this adaptive worker.
    /// All stealers share the same active-tag Arc; a migration
    /// is visible to all of them on their next Acquire-load.
    #[inline]
    pub fn stealer(&self) -> AdaptiveStealer {
        AdaptiveStealer {
            active: Arc::clone(&self.active),
            khl: self.khl.stealer(),
            fcl: self.fcl.stealer(),
            inactive_dirty: Arc::clone(&self.inactive_dirty),
        }
    }
}

/// Outcome of an adaptive steal. Mirrors KhlSteal2 / FclSteal2.
pub enum AdaptiveSteal2<T> {
    /// Got a job; extras stashed in the appropriate per-backing
    /// stash.
    Success(T),
    /// Both backings empty.
    Empty,
    /// CAS-loss on the active backing's steal; caller should
    /// retry.
    Retry,
}

/// Thief-side adaptive steal: tries active backing first, falls
/// back to the dormant backing only when `inactive_dirty` says it
/// might hold orphans from a recent migration. Absorbs the
/// K_inner=3 batch into the appropriate per-backing stash inside
/// [`AdaptiveStash`]. Clears `inactive_dirty` when a fallback
/// steal observes Empty -- subsequent thief steals skip the
/// dormant probe entirely until the next migration.
#[inline]
pub fn steal_via_stash(
    stealer: &AdaptiveStealer,
    stash: &mut AdaptiveStash,
) -> AdaptiveSteal2<JobRef> {
    let active = stealer.active.load(Ordering::Acquire);
    if active == ACTIVE_KHL {
        match khl_steal_via_stash(&stealer.khl, &mut stash.khl) {
            KhlSteal2::Success(j) => AdaptiveSteal2::Success(j),
            KhlSteal2::Retry => AdaptiveSteal2::Retry,
            KhlSteal2::Empty => {
                if !stealer.inactive_dirty.load(Ordering::Acquire) {
                    return AdaptiveSteal2::Empty;
                }
                // Dormant Fcl may still have orphans from a
                // recent migration.
                match fcl_steal_via_stash(&stealer.fcl, &mut stash.fcl) {
                    FclSteal2::Success(j) => AdaptiveSteal2::Success(j),
                    FclSteal2::Retry => AdaptiveSteal2::Retry,
                    FclSteal2::Empty => {
                        stealer.inactive_dirty.store(false, Ordering::Release);
                        AdaptiveSteal2::Empty
                    }
                }
            }
        }
    } else {
        match fcl_steal_via_stash(&stealer.fcl, &mut stash.fcl) {
            FclSteal2::Success(j) => AdaptiveSteal2::Success(j),
            FclSteal2::Retry => AdaptiveSteal2::Retry,
            FclSteal2::Empty => {
                if !stealer.inactive_dirty.load(Ordering::Acquire) {
                    return AdaptiveSteal2::Empty;
                }
                match khl_steal_via_stash(&stealer.khl, &mut stash.khl) {
                    KhlSteal2::Success(j) => AdaptiveSteal2::Success(j),
                    KhlSteal2::Retry => AdaptiveSteal2::Retry,
                    KhlSteal2::Empty => {
                        stealer.inactive_dirty.store(false, Ordering::Release);
                        AdaptiveSteal2::Empty
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::Variant;
    use crate::sched::job::StackJob;
    use crate::sched::latch::CoreLatch;

    #[test]
    fn default_starts_on_khl() {
        let (w, _s) = new_adaptive(4, KGating::PerSlot);
        assert_eq!(w.active_gating(), KGating::PerSlot);
    }

    #[test]
    fn explicit_counter_only_starts_on_fcl() {
        let (w, _s) = new_adaptive(4, KGating::CounterOnly);
        assert_eq!(w.active_gating(), KGating::CounterOnly);
    }

    #[test]
    fn migrate_flips_active_tag() {
        let (w, _s) = new_adaptive(4, KGating::PerSlot);
        assert_eq!(w.active_gating(), KGating::PerSlot);
        w.migrate_to(KGating::CounterOnly);
        assert_eq!(w.active_gating(), KGating::CounterOnly);
        w.migrate_to(KGating::PerSlot);
        assert_eq!(w.active_gating(), KGating::PerSlot);
    }

    #[test]
    fn push_pop_round_trip_on_khl() {
        let (w, _s) = new_adaptive(4, KGating::PerSlot);
        let j = StackJob::new(|_| 42u32, CoreLatch::new());
        let r = unsafe { j.as_job_ref(4, 0, Variant::Faithful) };
        w.push(r);
        let popped = w.pop().expect("pop returns the pushed item");
        unsafe { popped.execute() };
        assert!(j.latch.is_set());
        assert_eq!(unsafe { j.into_result() }, 42);
    }

    #[test]
    fn push_pop_round_trip_on_fcl() {
        let (w, _s) = new_adaptive(4, KGating::CounterOnly);
        let j = StackJob::new(|_| 99u32, CoreLatch::new());
        let r = unsafe { j.as_job_ref(4, 0, Variant::Faithful) };
        w.push(r);
        let popped = w.pop().expect("pop returns the pushed item");
        unsafe { popped.execute() };
        assert!(j.latch.is_set());
    }

    #[test]
    fn migration_preserves_orphan_items() {
        // Push on KHL, migrate to Fcl, then push more on Fcl.
        // Pop should drain BOTH backings.
        let (w, _s) = new_adaptive(4, KGating::PerSlot);
        let j1 = StackJob::new(|_| 1u32, CoreLatch::new());
        let r1 = unsafe { j1.as_job_ref(4, 0, Variant::Fast) };
        w.push(r1);  // on KHL

        w.migrate_to(KGating::CounterOnly);
        // KHL now dormant but still has the j1 slot.

        let j2 = StackJob::new(|_| 2u32, CoreLatch::new());
        let r2 = unsafe { j2.as_job_ref(4, 0, Variant::Fast) };
        w.push(r2);  // on Fcl

        // Active is Fcl; pop drains Fcl first (gets j2), then KHL (gets j1).
        let p1 = w.pop().expect("first pop from active Fcl");
        unsafe { p1.execute() };
        let p2 = w.pop().expect("second pop from dormant KHL");
        unsafe { p2.execute() };
        assert!(j1.latch.is_set() && j2.latch.is_set());
        assert!(w.pop().is_none());
    }

    #[test]
    fn stealer_observes_migration() {
        let (w, s) = new_adaptive(4, KGating::PerSlot);
        let mut stash = AdaptiveStash::empty();
        let j = StackJob::new(|_| 5u32, CoreLatch::new());
        let r = unsafe { j.as_job_ref(4, 0, Variant::Fast) };
        w.push(r);
        // Steal sees the item on active KHL backing.
        match steal_via_stash(&s, &mut stash) {
            AdaptiveSteal2::Success(jr) => {
                unsafe { jr.execute() };
                assert!(j.latch.is_set());
            }
            _ => panic!("expected Success"),
        }
    }
}

//! Generic in-process Chase-Lev work-stealing deque.
//!
//! K_inner-agnostic primitive: the slot type `T` is a generic
//! parameter. For K_inner=1 use `T=JobRef`; for Fcl/KHL variants use
//! `T=FatSlot`/`KhlSlot` (defined alongside in `fcl_local.rs` /
//! `khl_local.rs`).
//!
//! Forked from the crossbeam deque design to expose what its
//! private API hides: `slot_ptr(idx)` so prefetches target the slot
//! line the steal CAS reads (not the stealer handle's stack
//! address), a generic slot type `T`, and swap points for the
//! MOVDIR64B publish and per-slot Vyukov K_gating paths.
//!
//! Protocol: same as `src/backend/shared_mem/chase_lev_mmf.rs`
//! (pseudocode there), with `MmapMut` replaced by
//! `Box<[UnsafeCell<MaybeUninit<T>>]>`. Per Vafeiadis et al.
//! (arXiv:2309.03642) the safety invariants are slot-type-agnostic:
//! Release-store of `bottom` after the slot write, Acquire-load
//! before the slot read, SeqCst fence + CAS-on-`top` for the b==t
//! single-item race.
//!
//! Double-drop hygiene: owner-pop's `t == b` branch and
//! thief-steal's CAS-loss branch byte-copy the slot
//! (`MaybeUninit::assume_init_read`) before the CAS resolves
//! ownership; the losing side `mem::forget`s its tentative copy so
//! a `T` with a real `Drop` is not dropped twice.

#![allow(clippy::missing_errors_doc)]

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicI64, Ordering, fence};
use std::sync::Arc;

/// Outcome of [`Worker::pop`] / [`Stealer::steal`]. Mirrors
/// `crossbeam::deque::Steal` exactly so call sites can be swapped
/// 1:1.
#[derive(Debug, PartialEq, Eq)]
pub enum Steal<T> {
    /// Got a slot.
    Success(T),
    /// Deque was empty (no race; nothing to do).
    Empty,
    /// Thief CAS lost to a competing thief; caller should retry.
    Retry,
}

/// Header + slot buffer for the Chase-Lev deque. `top` and `bottom`
/// land on their own cache lines (false-sharing prevention); the
/// capacity / mask / buffer follow.
///
/// SAFETY: every field of this struct is protected by the Chase-Lev
/// per-atomic ordering discipline documented in the module header.
/// `UnsafeCell<MaybeUninit<T>>` slots are only written by the owner
/// (between the bottom-load and the Release-store of bottom+1) and
/// only read by either the owner under the LIFO pop path or by a
/// thief whose Acquire-load of bottom synchronizes-with the owner's
/// Release-store; the CAS on top linearizes competing thieves.
#[repr(C, align(64))]
struct Header<T> {
    /// Chase-Lev `top` counter. Thieves CAS this to claim a slot.
    top: AtomicI64,
    _pad_top: [u8; 56],
    /// Chase-Lev `bottom` counter. Owner stores this with Release
    /// on push (no atomic on the hot path).
    bottom: AtomicI64,
    _pad_bottom: [u8; 56],
    /// Slot count; always a power of two.
    capacity: usize,
    /// `capacity - 1` precomputed for `idx & mask` slot indexing.
    capacity_mask: i64,
    /// Backing storage. Indexed via `(top|bottom) & capacity_mask`.
    /// Slots are initialized on push and consumed on pop / steal.
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
}

// SAFETY: the Chase-Lev protocol's per-atomic ordering pairs gate
// all access to `buffer` and the counters. T: Send is sufficient
// because the slot bytes only cross threads at the moment a thief
// successfully CASes top (or the owner pops without thief race);
// the synchronizes-with relationship the Acquire-load of bottom
// establishes makes those bytes safe to read on the thief side.
unsafe impl<T: Send> Send for Header<T> {}
// SAFETY: Sync is required because Arc<Header<T>> only auto-derives
// Send if Header<T>: Send + Sync. The protocol guarantees safe
// concurrent access from one owner + many thieves; sharing &Header
// across threads is the intended access pattern (each Worker /
// Stealer holds Arc<Header>).
unsafe impl<T: Send> Sync for Header<T> {}

impl<T> Drop for Header<T> {
    fn drop(&mut self) {
        // Drain any unclaimed slots so their destructors run. Slots
        // in [top, bottom) hold initialized T; everything else is
        // MaybeUninit::uninit.
        //
        // SAFETY: we hold &mut self, so no concurrent access is
        // possible. `top` and `bottom` reads with Relaxed are
        // sufficient because there are no other observers.
        let t = self.top.load(Ordering::Relaxed);
        let b = self.bottom.load(Ordering::Relaxed);
        let mut i = t;
        while i < b {
            let idx = (i & self.capacity_mask) as usize;
            // SAFETY: slots in [top, bottom) are initialized by the
            // protocol; we drop the held T to honor the
            // execute-exactly-once contract from the caller side.
            // For T = JobRef this is a no-op (no Drop), but for any
            // T with Drop this drains the unclaimed work.
            unsafe {
                (*self.buffer[idx].get()).assume_init_drop();
            }
            i += 1;
        }
    }
}

/// Owner half of the deque. Single-owner-writer: only ONE thread
/// may hold a `Worker<T>` at a time. The companion `Stealer<T>`
/// handles (one or more clones) provide thief-side steal access.
pub struct Worker<T> {
    inner: Arc<Header<T>>,
}

/// Thief handle. Clonable; any number of thieves can hold one.
/// Concurrent `steal()` calls from multiple threads race via the
/// per-CAS linearization on `top`.
pub struct Stealer<T> {
    inner: Arc<Header<T>>,
}

impl<T> Clone for Stealer<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Construct a new Chase-Lev deque with `capacity` slots (rounded
/// up to next power of two, minimum 2). Returns the owner half +
/// one thief handle; call `Worker::stealer()` to clone additional
/// thief handles.
pub fn new_chase_lev<T>(capacity: usize) -> (Worker<T>, Stealer<T>) {
    let capacity = capacity.max(2).next_power_of_two();
    let mut buf = Vec::with_capacity(capacity);
    for _ in 0..capacity {
        buf.push(UnsafeCell::new(MaybeUninit::uninit()));
    }
    let inner = Arc::new(Header {
        top: AtomicI64::new(0),
        _pad_top: [0u8; 56],
        bottom: AtomicI64::new(0),
        _pad_bottom: [0u8; 56],
        capacity,
        capacity_mask: (capacity as i64) - 1,
        buffer: buf.into_boxed_slice(),
    });
    let w = Worker {
        inner: Arc::clone(&inner),
    };
    let s = Stealer { inner };
    (w, s)
}

impl<T> Worker<T> {
    /// LIFO push. Returns `Err(item)` if at capacity. Single-owner
    /// invariant: only the owner thread may call this.
    #[inline(always)]
    pub fn push(&self, item: T) -> Result<(), T> {
        let h = &*self.inner;
        let b = h.bottom.load(Ordering::Relaxed);
        let t = h.top.load(Ordering::Acquire);
        let size = b - t;
        if size >= h.capacity as i64 {
            return push_full_cold(item);
        }
        let idx = (b & h.capacity_mask) as usize;
        // SAFETY: the slot at `b` is not in [top, bottom) yet (we
        // haven't bumped bottom), so no thief observes it. The
        // owner is single-writer; no other thread races us here.
        // `get_unchecked` skips the bounds check; idx is masked
        // by capacity_mask so it is always in [0, capacity).
        unsafe {
            (*h.buffer.get_unchecked(idx).get()).write(item);
        }
        // Release-store of bottom synchronizes-with the thief's
        // Acquire-load of bottom; the slot bytes the owner just
        // wrote are visible to whichever thief reads the slot next.
        h.bottom.store(b + 1, Ordering::Release);
        Ok(())
    }

    /// LIFO pop. Races with thieves at the b == t single-item case;
    /// the embedded SeqCst fence + CAS linearize the race.
    #[inline]
    pub fn pop(&self) -> Steal<T> {
        let h = &*self.inner;
        let b = h.bottom.load(Ordering::Relaxed) - 1;
        // Reserve our slot by writing bottom = b. Thieves see
        // top..bottom shrinking by one. Relaxed because the SeqCst
        // fence directly below orders this with the subsequent
        // top.load.
        h.bottom.store(b, Ordering::Relaxed);
        fence(Ordering::SeqCst);
        let t = h.top.load(Ordering::Relaxed);
        if t > b {
            // Deque was empty; restore bottom and report.
            h.bottom.store(b + 1, Ordering::Relaxed);
            return Steal::Empty;
        }
        let idx = (b & h.capacity_mask) as usize;
        // SAFETY: tentative byte-copy via get_unchecked - idx is
        // masked. In the t < b branch we own the slot outright; in
        // the t == b branch we race the thief and forget the alias
        // on CAS loss.
        let item = unsafe { (*h.buffer.get_unchecked(idx).get()).assume_init_read() };
        if t < b {
            return Steal::Success(item);
        }
        // t == b: single-item race.
        let won = h
            .top
            .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok();
        h.bottom.store(b + 1, Ordering::Relaxed);
        if won {
            Steal::Success(item)
        } else {
            // Thief took it; forget our alias to suppress
            // double-drop on the winning side's canonical copy.
            core::mem::forget(item);
            Steal::Empty
        }
    }

    /// Clone a fresh thief handle. Multiple thieves can hold
    /// clones; each steal() races via the per-CAS linearization
    /// on `top`.
    pub fn stealer(&self) -> Stealer<T> {
        Stealer {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Approximate is-empty snapshot. Uses Acquire on both top and
    /// bottom for a consistent point-in-time view; a concurrent
    /// thief push or another thief steal may invalidate the result
    /// immediately after return. Use as a hint only.
    #[inline]
    pub fn is_empty(&self) -> bool {
        let h = &*self.inner;
        let b = h.bottom.load(Ordering::Acquire);
        let t = h.top.load(Ordering::Acquire);
        b <= t
    }

    /// Number of slots in the underlying buffer (always power of
    /// two; rounded up from the constructor's request).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Approximate pending-item count snapshot (`bottom - top`
    /// clamped to >= 0). Both atomics are loaded with `Acquire`;
    /// a concurrent thief steal or owner push may invalidate
    /// the result immediately after return. Hint only.
    #[inline]
    pub fn len(&self) -> usize {
        let h = &*self.inner;
        let b = h.bottom.load(Ordering::Acquire);
        let t = h.top.load(Ordering::Acquire);
        (b - t).max(0) as usize
    }

    /// Raw slot pointer at the given (unmasked) index. Used by the
    /// prefetch primitive; production code goes through `push` /
    /// `pop` which mask the index internally.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for reads of an initialized T
    /// only when `idx` is in `[top, bottom)`; outside that range
    /// the slot may be `MaybeUninit::uninit`. Callers that issue
    /// architectural prefetch hints (which never fault) may ignore
    /// this constraint; semantic readers must not.
    #[inline(always)]
    pub unsafe fn slot_ptr(&self, idx: i64) -> *const T {
        let h = &*self.inner;
        let slot_idx = (idx & h.capacity_mask) as usize;
        // SAFETY: slot_idx is masked by capacity_mask so it is
        // always in [0, capacity); get_unchecked skips the bounds
        // check the compiler may not always elide.
        unsafe { h.buffer.get_unchecked(slot_idx).get() as *const T }
    }
}

/// Cold path for `Worker::push` when the deque is at capacity. The
/// hot path branches to this helper, allowing the optimizer to
/// keep the no-error branch linear in the instruction stream.
#[cold]
#[inline(never)]
fn push_full_cold<T>(item: T) -> Result<(), T> {
    Err(item)
}

impl<T> Stealer<T> {
    /// FIFO steal. Returns `Steal::Retry` on CAS-loss; caller's
    /// outer loop should retry.
    #[inline]
    pub fn steal(&self) -> Steal<T> {
        let h = &*self.inner;
        let t = h.top.load(Ordering::Acquire);
        fence(Ordering::SeqCst);
        let b = h.bottom.load(Ordering::Acquire);
        if t >= b {
            return Steal::Empty;
        }
        let idx = (t & h.capacity_mask) as usize;
        // SAFETY: t is in [top, bottom); get_unchecked skips the
        // bounds check (idx masked by capacity_mask). The Acquire
        // load of bottom synchronizes-with the owner's Release
        // store; CAS-loss path forgets the alias.
        let item = unsafe { (*h.buffer.get_unchecked(idx).get()).assume_init_read() };
        let won = h
            .top
            .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok();
        if won {
            Steal::Success(item)
        } else {
            // Another thief took the slot; forget our alias.
            core::mem::forget(item);
            Steal::Retry
        }
    }

    /// Hint that a steal at the current `top` is upcoming. Issues
    /// `_mm_prefetch(_MM_HINT_T0)` on the slot bytes the next
    /// `steal()` will read. Best-effort architectural hint; never
    /// faults regardless of the slot's initialization state.
    ///
    /// Pattern: call `prefetch_for_steal`, do K_inflight-worth of
    /// unrelated work (other-deque probes, last-victim update,
    /// counter increments), then call `steal()`. The unrelated work
    /// overlaps with the slot-line coherence transfer, hiding the
    /// 60-80 ns cross-CCX miss the steal would otherwise pay.
    ///
    /// No-op on non-x86_64 targets (no stable cross-platform
    /// prefetch intrinsic).
    #[inline]
    pub fn prefetch_for_steal(&self) {
        let h = &*self.inner;
        let t = h.top.load(Ordering::Relaxed);
        let idx = (t & h.capacity_mask) as usize;
        let slot = h.buffer[idx].get() as *const u8;
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: _mm_prefetch is a no-side-effect hint that
            // accepts any pointer value without architectural fault.
            // The slot pointer is valid mapped memory inside the
            // Box buffer regardless of whether the slot is
            // currently initialized.
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    slot as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::hint::black_box(slot);
        }
    }

    /// Approximate is-empty snapshot. Same semantics as
    /// [`Worker::is_empty`]; use as a hint only.
    #[inline]
    pub fn is_empty(&self) -> bool {
        let h = &*self.inner;
        let b = h.bottom.load(Ordering::Acquire);
        let t = h.top.load(Ordering::Acquire);
        b <= t
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as O};
    use std::thread;

    #[test]
    fn push_then_pop_is_lifo() {
        let (w, _s) = new_chase_lev::<u32>(8);
        w.push(1).unwrap();
        w.push(2).unwrap();
        w.push(3).unwrap();
        assert_eq!(w.pop(), Steal::Success(3));
        assert_eq!(w.pop(), Steal::Success(2));
        assert_eq!(w.pop(), Steal::Success(1));
        assert_eq!(w.pop(), Steal::Empty);
    }

    #[test]
    fn steal_drains_fifo() {
        let (w, s) = new_chase_lev::<u32>(8);
        w.push(1).unwrap();
        w.push(2).unwrap();
        w.push(3).unwrap();
        // Steal is FIFO from the opposite end.
        assert_eq!(s.steal(), Steal::Success(1));
        assert_eq!(s.steal(), Steal::Success(2));
        assert_eq!(s.steal(), Steal::Success(3));
        assert_eq!(s.steal(), Steal::Empty);
    }

    #[test]
    fn capacity_rounds_to_power_of_two() {
        let (w, _s) = new_chase_lev::<u32>(5);
        // 5 -> next pow2 = 8
        assert_eq!(w.capacity(), 8);
        let (w2, _s2) = new_chase_lev::<u32>(0);
        // 0 -> max(2) -> next pow2 = 2
        assert_eq!(w2.capacity(), 2);
    }

    #[test]
    fn push_full_returns_err() {
        let (w, _s) = new_chase_lev::<u32>(2);
        w.push(1).unwrap();
        w.push(2).unwrap();
        let err = w.push(3).unwrap_err();
        assert_eq!(err, 3, "push must return the rejected item back");
    }

    #[test]
    fn empty_pop_returns_empty() {
        let (w, _s) = new_chase_lev::<u32>(4);
        assert_eq!(w.pop(), Steal::Empty);
    }

    #[test]
    fn empty_steal_returns_empty() {
        let (_w, s) = new_chase_lev::<u32>(4);
        assert_eq!(s.steal(), Steal::Empty);
    }

    #[test]
    fn prefetch_for_steal_is_safe_on_empty() {
        let (_w, s) = new_chase_lev::<u32>(4);
        // No pushes; top.load reads 0; slot[0] is MaybeUninit. The
        // prefetch is a no-fault hint regardless.
        s.prefetch_for_steal();
        s.prefetch_for_steal();
    }

    #[test]
    fn prefetch_does_not_disturb_state() {
        let (w, s) = new_chase_lev::<u32>(8);
        w.push(1).unwrap();
        w.push(2).unwrap();
        let was_empty_before = w.is_empty();
        for _ in 0..16 {
            s.prefetch_for_steal();
        }
        assert_eq!(was_empty_before, w.is_empty());
        // Drain to verify nothing leaked into the deque.
        assert_eq!(s.steal(), Steal::Success(1));
        assert_eq!(s.steal(), Steal::Success(2));
        assert_eq!(s.steal(), Steal::Empty);
    }

    #[test]
    fn stealer_clone_shares_state() {
        let (w, s1) = new_chase_lev::<u32>(8);
        let s2 = s1.clone();
        w.push(10).unwrap();
        w.push(20).unwrap();
        assert_eq!(s1.steal(), Steal::Success(10));
        // s2 sees the same state as s1.
        assert_eq!(s2.steal(), Steal::Success(20));
        assert_eq!(s1.steal(), Steal::Empty);
        assert_eq!(s2.steal(), Steal::Empty);
    }

    #[test]
    fn stealer_from_worker_is_same_deque() {
        let (w, _s_orig) = new_chase_lev::<u32>(8);
        let s = w.stealer();
        w.push(42).unwrap();
        assert_eq!(s.steal(), Steal::Success(42));
    }

    #[test]
    fn drop_runs_destructors_for_unclaimed_slots() {
        // Use a type with a Drop side effect to verify the Drop impl
        // drains unclaimed slots. Without that, a Vec<u8> pushed
        // onto a deque that gets dropped would leak.
        use std::sync::atomic::{AtomicUsize, Ordering as O};
        #[derive(Debug)]
        struct DropCount<'a>(&'a AtomicUsize);
        impl Drop for DropCount<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, O::Relaxed);
            }
        }
        let count = AtomicUsize::new(0);
        {
            let (w, _s) = new_chase_lev::<DropCount<'_>>(4);
            w.push(DropCount(&count)).unwrap();
            w.push(DropCount(&count)).unwrap();
            w.push(DropCount(&count)).unwrap();
            // Drop the deque with 3 unclaimed slots; Drop should
            // run on each.
        }
        assert_eq!(count.load(O::Relaxed), 3,
            "unclaimed slots must run their destructors when the deque drops");
    }

    #[test]
    fn drop_does_not_double_drop_after_pop() {
        // After a successful pop, the consumed slot must NOT be
        // re-dropped by Header::drop.
        use std::sync::atomic::{AtomicUsize, Ordering as O};
        #[derive(Debug)]
        struct DropCount<'a>(&'a AtomicUsize);
        impl Drop for DropCount<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, O::Relaxed);
            }
        }
        let count = AtomicUsize::new(0);
        {
            let (w, _s) = new_chase_lev::<DropCount<'_>>(4);
            w.push(DropCount(&count)).unwrap();
            w.push(DropCount(&count)).unwrap();
            // Pop one; one Drop should run via pop's Success drop.
            match w.pop() {
                Steal::Success(d) => drop(d),
                other => panic!("expected Success, got {other:?}"),
            }
            assert_eq!(count.load(O::Relaxed), 1, "pop's drop should run");
            // Deque has 1 unclaimed slot; its Drop runs at deque drop.
        }
        assert_eq!(count.load(O::Relaxed), 2,
            "exactly one extra drop should run when deque is dropped with 1 unclaimed slot");
    }

    #[test]
    fn concurrent_owner_push_and_thieves_no_double_take() {
        // The hardest invariant: across N owner pushes and many
        // thief steals, every slot is consumed exactly once. Run
        // a stress loop and check the sum.
        //
        // Mirrors `chase_lev_mmf::tests::owner_push_with_concurrent_thieves_no_double_take`.
        let (w, s_proto) = new_chase_lev::<u32>(64);
        let s_proto = Arc::new(s_proto);
        let w = Arc::new(w);
        let n = 5_000u32;
        let consumed = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));

        let mut thieves = Vec::new();
        for _ in 0..8 {
            let s = (*s_proto).clone();
            let consumed = Arc::clone(&consumed);
            let sum = Arc::clone(&sum);
            thieves.push(thread::spawn(move || {
                while consumed.load(O::Relaxed) < n as usize {
                    match s.steal() {
                        Steal::Success(v) => {
                            consumed.fetch_add(1, O::Relaxed);
                            sum.fetch_add(v as usize, O::Relaxed);
                        }
                        Steal::Empty | Steal::Retry => std::thread::yield_now(),
                    }
                }
            }));
        }

        // Owner pushes; periodically pops a few to interleave.
        let w_owner = Arc::clone(&w);
        let consumed_owner = Arc::clone(&consumed);
        let sum_owner = Arc::clone(&sum);
        let owner = thread::spawn(move || {
            let mut i = 0u32;
            while i < n {
                match w_owner.push(i) {
                    Ok(()) => i += 1,
                    Err(_) => {
                        // Deque full; drain locally to keep moving.
                        if let Steal::Success(v) = w_owner.pop() {
                            consumed_owner.fetch_add(1, O::Relaxed);
                            sum_owner.fetch_add(v as usize, O::Relaxed);
                        } else {
                            std::thread::yield_now();
                        }
                    }
                }
            }
            // Drain the rest from the owner side until all consumed.
            while consumed_owner.load(O::Relaxed) < n as usize {
                match w_owner.pop() {
                    Steal::Success(v) => {
                        consumed_owner.fetch_add(1, O::Relaxed);
                        sum_owner.fetch_add(v as usize, O::Relaxed);
                    }
                    Steal::Empty | Steal::Retry => std::thread::yield_now(),
                }
            }
        });

        owner.join().expect("owner joined");
        for h in thieves {
            h.join().expect("thief joined");
        }

        let expected: usize = (0..n as usize).sum();
        assert_eq!(
            sum.load(O::Relaxed),
            expected,
            "sum invariant violated: each slot should be consumed exactly once"
        );
        assert_eq!(consumed.load(O::Relaxed), n as usize);
    }

    #[test]
    fn wraparound_across_capacity_boundary_preserves_ordering() {
        // Push past the capacity-aligned boundary so the masked
        // index wraps. Verify pop/steal still produce the right
        // values.
        let (w, s) = new_chase_lev::<u32>(4);
        // Fill, pop 2, refill - top is now at 2, bottom advances
        // past the capacity boundary on subsequent pushes.
        w.push(0).unwrap();
        w.push(1).unwrap();
        assert_eq!(w.pop(), Steal::Success(1));
        assert_eq!(s.steal(), Steal::Success(0));
        w.push(10).unwrap();
        w.push(11).unwrap();
        w.push(12).unwrap();
        w.push(13).unwrap();
        // Now top=2, bottom=6 -> physical slots [2,3,0,1] hold [10,11,12,13].
        assert_eq!(s.steal(), Steal::Success(10));
        assert_eq!(s.steal(), Steal::Success(11));
        assert_eq!(w.pop(), Steal::Success(13));
        assert_eq!(w.pop(), Steal::Success(12));
        assert_eq!(w.pop(), Steal::Empty);
    }
}

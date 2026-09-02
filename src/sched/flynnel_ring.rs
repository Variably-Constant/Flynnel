//! Non-blocking bounded MPMC ring (Vyukov per-slot sequence
//! protocol). The fast-path primitive for flynnel call sites
//! that don't need a blocking-notify wrapper.
//!
//! The queueing surface has three regimes: owner-side deques
//! ([`crate::sched::chase_lev_local`] /
//! [`crate::sched::adaptive_worker`]), this bounded MPMC ring and
//! its SPSC / MPSC / composed variants, and the blocking notify
//! wrapper built atop [`crate::sched::sleep::Parker`].
//!
//! ## Protocol
//!
//! Classical Vyukov bounded MPMC ring (Dmitry Vyukov 2010). Each
//! slot carries a sequence number that gates publication; producers
//! and consumers race independently via fetch-and-CAS on `head` /
//! `tail` counters.
//!
//! Producer:
//! ```text
//! loop:
//!   pos = tail.load(Relaxed)
//!   slot = buffer[pos & mask]
//!   seq = slot.seq.load(Acquire)
//!   diff = seq - pos
//!   if diff == 0:
//!     if tail.cas(pos, pos + 1, Relaxed, Relaxed) succeeds: break
//!   else if diff < 0: return Full
//!   else: continue   // another producer beat us; reload tail
//! write slot.data
//! slot.seq.store(pos + 1, Release)
//! ```
//!
//! Consumer (symmetric on `head`):
//! ```text
//! loop:
//!   pos = head.load(Relaxed)
//!   slot = buffer[pos & mask]
//!   seq = slot.seq.load(Acquire)
//!   diff = seq - (pos + 1)
//!   if diff == 0:
//!     if head.cas(pos, pos + 1, Relaxed, Relaxed) succeeds: break
//!   else if diff < 0: return Empty
//!   else: continue
//! read slot.data
//! slot.seq.store(pos + capacity, Release)
//! ```
//!
//! ## Cross-platform discipline
//!
//! Uses only AtomicU64 / Acquire / Release / Relaxed - universal
//! across every 64-bit-atomics-capable target (Linux/macOS/Windows
//! x86_64/aarch64/armv7).

#![allow(clippy::missing_errors_doc)]

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

/// Outcome of [`FlynnelRing::push`].
#[derive(Debug, PartialEq, Eq)]
pub enum PushResult<T> {
    /// Item enqueued.
    Ok,
    /// Ring is at capacity; caller's responsibility to retry or
    /// drop.
    Full(T),
}

/// Outcome of [`FlynnelRing::pop`].
#[derive(Debug, PartialEq, Eq)]
pub enum PopResult<T> {
    /// Got an item.
    Ok(T),
    /// Ring is empty.
    Empty,
}

/// Slot type. AtomicU64 seq + UnsafeCell<MaybeUninit<T>>.
#[repr(C, align(64))]
struct Slot<T> {
    seq: AtomicU64,
    data: UnsafeCell<MaybeUninit<T>>,
}

/// Shared header + slot buffer for the ring.
#[repr(C, align(64))]
struct Header<T> {
    /// Producer counter. Producers CAS this to claim a slot.
    tail: AtomicI64,
    _pad_tail: [u8; 56],
    /// Consumer counter. Consumers CAS this to claim a slot.
    head: AtomicI64,
    _pad_head: [u8; 56],
    /// Capacity (always a power of two).
    capacity: usize,
    capacity_mask: i64,
    /// Slot buffer.
    buffer: Box<[Slot<T>]>,
}

// SAFETY: Vyukov protocol's per-atomic ordering pairs gate all
// access to buffer. T: Send is sufficient because data crosses
// threads only after the producer's Release-store of seq
// synchronizes-with the consumer's Acquire-load of seq.
unsafe impl<T: Send> Send for Header<T> {}
unsafe impl<T: Send> Sync for Header<T> {}

/// Non-blocking bounded MPMC ring. Cloneable handle; multiple
/// producers and multiple consumers share clones of this handle.
pub struct FlynnelRing<T> {
    inner: Arc<Header<T>>,
}

impl<T> Clone for FlynnelRing<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: Send> FlynnelRing<T> {
    /// Construct a new ring with `capacity` slots (rounded up to
    /// next power of two; minimum 2).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(2).next_power_of_two();
        let mut buf = Vec::with_capacity(capacity);
        for i in 0..capacity {
            buf.push(Slot {
                seq: AtomicU64::new(i as u64),
                data: UnsafeCell::new(MaybeUninit::uninit()),
            });
        }
        let inner = Arc::new(Header {
            tail: AtomicI64::new(0),
            _pad_tail: [0u8; 56],
            head: AtomicI64::new(0),
            _pad_head: [0u8; 56],
            capacity,
            capacity_mask: (capacity as i64) - 1,
            buffer: buf.into_boxed_slice(),
        });
        Self { inner }
    }

    /// Capacity (always a power of two).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Approximate length. Tail and head loads use Acquire; result
    /// may be invalidated by concurrent activity immediately after
    /// return.
    #[inline]
    pub fn len(&self) -> usize {
        let h = &*self.inner;
        let t = h.tail.load(Ordering::Acquire);
        let h2 = h.head.load(Ordering::Acquire);
        (t - h2).max(0) as usize
    }

    /// Approximate is-empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push an item. Returns `Ok` on success, `Full(item)` when
    /// the ring is at capacity (caller decides whether to retry,
    /// back off, or drop).
    ///
    /// Uses Vyukov CAS-loop on tail with cold_path hints on the
    /// Full/Retry branches. Bounded-ring backpressure semantics:
    /// when the slot's seq is behind pos (ring full), returns
    /// Full(item) without claiming the slot. The fetch_add MPMC
    /// variant would shave a few ns on success but loses the
    /// non-blocking Full semantics that try_push consumers
    /// depend on.
    #[inline]
    pub fn push(&self, item: T) -> PushResult<T> {
        let h = &*self.inner;
        let mut pos = h.tail.load(Ordering::Relaxed);
        loop {
            // SAFETY: pos & capacity_mask is always in [0, capacity).
            let slot = unsafe { h.buffer.get_unchecked((pos & h.capacity_mask) as usize) };
            let seq = slot.seq.load(Ordering::Acquire);
            let diff = (seq as i64) - pos;
            if diff == 0 {
                // Slot ready for producer claim.
                match h.tail.compare_exchange_weak(
                    pos,
                    pos + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: we own this slot via the seq
                        // invariant; consumer reads no data until
                        // our Release-store of seq below.
                        unsafe {
                            (*slot.data.get()).write(item);
                        }
                        slot.seq.store((pos as u64) + 1, Ordering::Release);
                        return PushResult::Ok;
                    }
                    Err(new_pos) => {
                        pos = new_pos;
                        continue;
                    }
                }
            } else if diff < 0 {
                // Cold path: ring full.
                core::hint::cold_path();
                return PushResult::Full(item);
            } else {
                // Cold path: another producer raced past;
                // reload tail and retry.
                core::hint::cold_path();
                pos = h.tail.load(Ordering::Relaxed);
            }
        }
    }

    /// Push using fetch_add instead of CAS-loop on tail. Faster
    /// than [`Self::push`] under heavy MPMC contention (LOCK XADD
    /// is one instruction; CAS-loop retries on contention) BUT
    /// has different backpressure semantics: when the ring is
    /// full, this method BLOCKS (spins) waiting for a consumer
    /// to drain. Caller MUST ensure consumers exist; otherwise
    /// this call deadlocks.
    ///
    /// Use this for the flynnel call sites that have guaranteed
    /// consumers (io_pool worker pool, hybrid GPU-result deliver,
    /// pipeline stage hand-off). For try_push semantics with
    /// non-blocking Full return, use [`Self::push`].
    #[inline]
    pub fn push_blocking(&self, item: T) {
        let h = &*self.inner;
        // One-instruction slot claim via LOCK XADD.
        let pos = h.tail.fetch_add(1, Ordering::Relaxed);
        // SAFETY: pos & capacity_mask in [0, capacity).
        let slot = unsafe { h.buffer.get_unchecked((pos & h.capacity_mask) as usize) };
        // Spin-wait for slot to be ready (consumer released the
        // previous round's data).
        loop {
            let seq = slot.seq.load(Ordering::Acquire);
            if (seq as i64) == pos {
                break;
            }
            core::hint::spin_loop();
        }
        // SAFETY: slot is ready per seq invariant.
        unsafe {
            (*slot.data.get()).write(item);
        }
        slot.seq.store((pos as u64) + 1, Ordering::Release);
    }

    /// Pop using fetch_add instead of CAS-loop on head. Mirrors
    /// [`Self::push_blocking`] - BLOCKS waiting for producer to
    /// publish; caller MUST ensure producers exist. Use only
    /// when the consumer can wait.
    #[inline]
    pub fn pop_blocking(&self) -> T {
        let h = &*self.inner;
        let pos = h.head.fetch_add(1, Ordering::Relaxed);
        let slot = unsafe { h.buffer.get_unchecked((pos & h.capacity_mask) as usize) };
        loop {
            let seq = slot.seq.load(Ordering::Acquire);
            if (seq as i64) == pos + 1 {
                break;
            }
            core::hint::spin_loop();
        }
        // SAFETY: producer's Release-store of seq synchronized-
        // with our Acquire-load above; data is visible.
        let item = unsafe { (*slot.data.get()).assume_init_read() };
        slot.seq.store(
            (pos as u64) + (h.capacity as u64),
            Ordering::Release,
        );
        item
    }

    /// Pop an item. Returns `Ok(item)` on success, `Empty` when
    /// the ring is empty.
    #[inline]
    pub fn pop(&self) -> PopResult<T> {
        let h = &*self.inner;
        let mut pos = h.head.load(Ordering::Relaxed);
        loop {
            // SAFETY: pos & capacity_mask is always in [0, capacity).
            let slot = unsafe { h.buffer.get_unchecked((pos & h.capacity_mask) as usize) };
            let seq = slot.seq.load(Ordering::Acquire);
            let diff = (seq as i64) - (pos + 1);
            if diff == 0 {
                match h.head.compare_exchange_weak(
                    pos,
                    pos + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: producer's Release-store of seq
                        // synchronized-with our Acquire-load above.
                        let item = unsafe { (*slot.data.get()).assume_init_read() };
                        slot.seq.store(
                            (pos as u64) + (h.capacity as u64),
                            Ordering::Release,
                        );
                        return PopResult::Ok(item);
                    }
                    Err(new_pos) => {
                        pos = new_pos;
                        continue;
                    }
                }
            } else if diff < 0 {
                // Cold path: ring empty.
                core::hint::cold_path();
                return PopResult::Empty;
            } else {
                core::hint::cold_path();
                pos = h.head.load(Ordering::Relaxed);
            }
        }
    }
}

impl<T> Drop for Header<T> {
    fn drop(&mut self) {
        // Drain any unclaimed slots so destructors run. Slots
        // where seq == producer_pos + 1 hold initialized items;
        // walk head..tail.
        //
        // SAFETY: we hold &mut self via Arc::drop's exclusive
        // access pattern (this Drop runs only when refcount hits
        // 0).
        let t = self.tail.load(Ordering::Relaxed);
        let h = self.head.load(Ordering::Relaxed);
        let mut pos = h;
        while pos < t {
            let slot = &self.buffer[(pos & self.capacity_mask) as usize];
            let seq = slot.seq.load(Ordering::Relaxed);
            if seq as i64 == pos + 1 {
                // Initialized slot.
                // SAFETY: per the seq invariant.
                unsafe {
                    (*slot.data.get()).assume_init_drop();
                }
            }
            pos += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Linkage confirmation markers
// ---------------------------------------------------------------------------

/// Marker symbol confirming the FlynnelRing push code path linked.
#[unsafe(no_mangle)]
pub static __flynnel_marker_ring_push: u8 = 0;
/// Marker symbol confirming the FlynnelRing pop code path linked.
#[unsafe(no_mangle)]
pub static __flynnel_marker_ring_pop: u8 = 0;

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
    fn push_pop_single_threaded() {
        let r = FlynnelRing::<u32>::new(8);
        assert!(matches!(r.push(1), PushResult::Ok));
        assert!(matches!(r.push(2), PushResult::Ok));
        assert!(matches!(r.push(3), PushResult::Ok));
        assert!(matches!(r.pop(), PopResult::Ok(1)));
        assert!(matches!(r.pop(), PopResult::Ok(2)));
        assert!(matches!(r.pop(), PopResult::Ok(3)));
        assert!(matches!(r.pop(), PopResult::Empty));
    }

    #[test]
    fn capacity_rounds_to_power_of_two() {
        let r = FlynnelRing::<u32>::new(5);
        assert_eq!(r.capacity(), 8);
        let r0 = FlynnelRing::<u32>::new(0);
        assert_eq!(r0.capacity(), 2);
    }

    #[test]
    fn full_returns_item_back() {
        let r = FlynnelRing::<u32>::new(2);
        assert!(matches!(r.push(1), PushResult::Ok));
        assert!(matches!(r.push(2), PushResult::Ok));
        match r.push(3) {
            PushResult::Full(v) => assert_eq!(v, 3),
            _ => panic!("expected Full"),
        }
    }

    #[test]
    fn empty_pop_returns_empty() {
        let r = FlynnelRing::<u32>::new(4);
        assert!(matches!(r.pop(), PopResult::Empty));
    }

    #[test]
    fn drop_runs_destructors_for_unclaimed_slots() {
        struct DropCount<'a>(&'a AtomicUsize);
        impl Drop for DropCount<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, O::Relaxed);
            }
        }
        let count = AtomicUsize::new(0);
        {
            let r = FlynnelRing::<DropCount<'_>>::new(4);
            assert!(matches!(r.push(DropCount(&count)), PushResult::Ok));
            assert!(matches!(r.push(DropCount(&count)), PushResult::Ok));
            assert!(matches!(r.push(DropCount(&count)), PushResult::Ok));
        }
        assert_eq!(count.load(O::Relaxed), 3);
    }

    #[test]
    fn concurrent_mpmc_stress_no_item_loss() {
        let r = Arc::new(FlynnelRing::<u32>::new(64));
        let n = 5_000u32;
        let produced = Arc::new(AtomicUsize::new(0));
        let consumed = Arc::new(AtomicUsize::new(0));
        let sum_consumed = Arc::new(AtomicUsize::new(0));

        let mut producers = Vec::new();
        for p_id in 0..4u32 {
            let r = Arc::clone(&r);
            let produced = Arc::clone(&produced);
            let n_per = n / 4;
            producers.push(thread::spawn(move || {
                for i in 0..n_per {
                    let item = p_id * n_per + i;
                    loop {
                        match r.push(item) {
                            PushResult::Ok => {
                                produced.fetch_add(1, O::Relaxed);
                                break;
                            }
                            PushResult::Full(_) => {
                                std::thread::yield_now();
                            }
                        }
                    }
                }
            }));
        }

        let mut consumers = Vec::new();
        for _ in 0..4 {
            let r = Arc::clone(&r);
            let consumed = Arc::clone(&consumed);
            let sum_consumed = Arc::clone(&sum_consumed);
            consumers.push(thread::spawn(move || {
                while consumed.load(O::Relaxed) < n as usize {
                    match r.pop() {
                        PopResult::Ok(v) => {
                            consumed.fetch_add(1, O::Relaxed);
                            sum_consumed.fetch_add(v as usize, O::Relaxed);
                        }
                        PopResult::Empty => std::thread::yield_now(),
                    }
                }
            }));
        }

        for h in producers {
            h.join().expect("producer");
        }
        for h in consumers {
            h.join().expect("consumer");
        }

        let n_per = n / 4;
        let expected: usize = (0..4u32)
            .flat_map(|p| (0..n_per).map(move |i| (p * n_per + i) as usize))
            .sum();
        assert_eq!(consumed.load(O::Relaxed), n as usize);
        assert_eq!(sum_consumed.load(O::Relaxed), expected,
            "sum invariant: each pushed item must be consumed exactly once");
    }
}

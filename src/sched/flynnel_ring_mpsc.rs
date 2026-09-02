//! Multi-Producer Single-Consumer (MPSC) bounded ring.
//! Producers contend on `tail` via CAS-loop; consumer is sole
//! owner of `head` and reads via Relaxed load + Release store.
//! Removes the consumer-side CAS contention from the full MPMC
//! design, fitting flynnel's mailbox-style access pattern where
//! multiple peers push but only one worker pops.
//!
//! ## When to use this vs FlynnelRing (MPMC) vs FlynnelRingSpsc
//!
//! | Pattern | Primitive | Cost per push | Cost per pop |
//! |---------|-----------|---------------|--------------|
//! | 1 producer, 1 consumer | FlynnelRingSpsc | 1 store | 1 store |
//! | N producers, 1 consumer | **FlynnelRingMpsc** | 1 CAS | 1 store |
//! | N producers, M consumers | FlynnelRing | 1 CAS | 1 CAS |
//!
//! The MPSC primitive is the right fit for:
//! - `arena_local.rs::WorkerCtx::mailbox` (peers push; owner pops)
//! - per-worker reply rings (workers push reply; one collector pops)
//!
//! ## Protocol
//!
//! Producer: standard Vyukov CAS-loop on `tail` + per-slot seq
//! publication.
//!
//! Consumer (single-owner): no CAS needed. Owner-private Relaxed
//! read of `head`, Acquire load of slot.seq, read data, Release
//! store of slot.seq (slot release for next round), Relaxed store
//! of `head + 1`.
//!
//! Per-slot seq still publishes the data; consumer's Acquire-load
//! of seq synchronizes-with the producer's Release-store.
//!
//! ## Cross-platform discipline
//!
//! Pure AtomicU64 / Acquire / Release / Relaxed; no x86-specific
//! intrinsics. Linux/macOS/Windows on x86_64/aarch64/armv7.

#![allow(clippy::missing_errors_doc)]

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

/// Outcome of [`MpscProducer::push`].
#[derive(Debug, PartialEq, Eq)]
pub enum MpscPushResult<T> {
    /// Item enqueued.
    Ok,
    /// Ring is at capacity; caller decides whether to retry or drop.
    Full(T),
}

/// Outcome of [`MpscConsumer::pop`].
#[derive(Debug, PartialEq, Eq)]
pub enum MpscPopResult<T> {
    /// Got an item.
    Ok(T),
    /// Ring is empty.
    Empty,
}

#[repr(C, align(64))]
struct MpscSlot<T> {
    seq: AtomicU64,
    data: UnsafeCell<MaybeUninit<T>>,
}

#[repr(C, align(64))]
struct MpscHeader<T> {
    /// Shared producer counter (multiple producers CAS).
    tail: AtomicI64,
    _pad_tail: [u8; 56],
    /// Owner-private consumer counter (single consumer).
    head: AtomicI64,
    _pad_head: [u8; 56],
    capacity: usize,
    capacity_mask: i64,
    buffer: Box<[MpscSlot<T>]>,
}

// SAFETY: same Vyukov ordering pairs as the MPMC variant; only
// difference is the head-side discipline (single owner).
unsafe impl<T: Send> Send for MpscHeader<T> {}
unsafe impl<T: Send> Sync for MpscHeader<T> {}

/// Multi-producer handle. `&self` API; clone for additional
/// producers.
pub struct MpscProducer<T> {
    inner: Arc<MpscHeader<T>>,
}

impl<T> Clone for MpscProducer<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Single-consumer handle. NOT Clone (cloning would create a
/// second consumer and violate the single-owner invariant).
pub struct MpscConsumer<T> {
    inner: Arc<MpscHeader<T>>,
}

// Send so the consumer can be moved to a different thread.
// !Sync because sharing &MpscConsumer across threads would mean
// multiple consumers.
unsafe impl<T: Send> Send for MpscConsumer<T> {}

/// Construct a new MPSC ring with `capacity` slots (rounded up
/// to next power of two; minimum 2). Returns one consumer + one
/// producer prototype; clone the producer for additional
/// producers.
pub fn new_mpsc<T: Send>(capacity: usize) -> (MpscProducer<T>, MpscConsumer<T>) {
    let capacity = capacity.max(2).next_power_of_two();
    let mut buf = Vec::with_capacity(capacity);
    for i in 0..capacity {
        buf.push(MpscSlot {
            seq: AtomicU64::new(i as u64),
            data: UnsafeCell::new(MaybeUninit::uninit()),
        });
    }
    let inner = Arc::new(MpscHeader {
        tail: AtomicI64::new(0),
        _pad_tail: [0u8; 56],
        head: AtomicI64::new(0),
        _pad_head: [0u8; 56],
        capacity,
        capacity_mask: (capacity as i64) - 1,
        buffer: buf.into_boxed_slice(),
    });
    (
        MpscProducer {
            inner: Arc::clone(&inner),
        },
        MpscConsumer { inner },
    )
}

impl<T: Send> MpscProducer<T> {
    /// Push an item. Returns `Ok` on success, `Full(item)` when
    /// the ring is at capacity. Multiple producers race via the
    /// Vyukov CAS-loop on `tail`; per-slot seq publication.
    #[inline]
    pub fn push(&self, item: T) -> MpscPushResult<T> {
        let h = &*self.inner;
        let mut pos = h.tail.load(Ordering::Relaxed);
        loop {
            // SAFETY: pos & capacity_mask in [0, capacity).
            let slot = unsafe { h.buffer.get_unchecked((pos & h.capacity_mask) as usize) };
            let seq = slot.seq.load(Ordering::Acquire);
            let diff = (seq as i64) - pos;
            if diff == 0 {
                match h.tail.compare_exchange_weak(
                    pos,
                    pos + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: seq invariant grants ownership.
                        unsafe {
                            (*slot.data.get()).write(item);
                        }
                        slot.seq.store((pos as u64) + 1, Ordering::Release);
                        return MpscPushResult::Ok;
                    }
                    Err(new_pos) => {
                        pos = new_pos;
                        continue;
                    }
                }
            } else if diff < 0 {
                core::hint::cold_path();
                return MpscPushResult::Full(item);
            } else {
                core::hint::cold_path();
                pos = h.tail.load(Ordering::Relaxed);
            }
        }
    }
}

impl<T: Send> MpscConsumer<T> {
    /// Pop an item. Returns `Ok(item)` on success, `Empty` when
    /// the ring is empty. NO CAS on head - the consumer is sole
    /// owner. One Acquire-load on the slot's seq + one Release-
    /// store after read.
    #[inline]
    pub fn pop(&self) -> MpscPopResult<T> {
        let h = &*self.inner;
        // Owner-private read of head; we're the only writer.
        let pos = h.head.load(Ordering::Relaxed);
        // SAFETY: pos & capacity_mask in [0, capacity).
        let slot = unsafe { h.buffer.get_unchecked((pos & h.capacity_mask) as usize) };
        let seq = slot.seq.load(Ordering::Acquire);
        let diff = (seq as i64) - (pos + 1);
        if diff != 0 {
            // Ring empty.
            core::hint::cold_path();
            return MpscPopResult::Empty;
        }
        // SAFETY: producer's Release-store of seq synchronized-
        // with our Acquire-load above; data is visible.
        let item = unsafe { (*slot.data.get()).assume_init_read() };
        // Release slot for the next round.
        slot.seq.store(
            (pos as u64) + (h.capacity as u64),
            Ordering::Release,
        );
        // Advance head (Relaxed - we're the only writer).
        h.head.store(pos + 1, Ordering::Relaxed);
        MpscPopResult::Ok(item)
    }
}

impl<T> Drop for MpscHeader<T> {
    fn drop(&mut self) {
        let t = self.tail.load(Ordering::Relaxed);
        let h = self.head.load(Ordering::Relaxed);
        let mut pos = h;
        while pos < t {
            let slot = &self.buffer[(pos & self.capacity_mask) as usize];
            let seq = slot.seq.load(Ordering::Relaxed);
            if seq as i64 == pos + 1 {
                // SAFETY: producer wrote, consumer hasn't drained.
                unsafe {
                    (*slot.data.get()).assume_init_drop();
                }
            }
            pos += 1;
        }
    }
}

/// Marker symbol for linkage confirmation.
#[unsafe(no_mangle)]
pub static __flynnel_marker_ring_mpsc_push: u8 = 0;
/// Companion marker for the MPSC pop path.
#[unsafe(no_mangle)]
pub static __flynnel_marker_ring_mpsc_pop: u8 = 0;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as O};
    use std::thread;

    #[test]
    fn push_pop_single_thread() {
        let (p, c) = new_mpsc::<u32>(8);
        assert!(matches!(p.push(1), MpscPushResult::Ok));
        assert!(matches!(p.push(2), MpscPushResult::Ok));
        assert!(matches!(p.push(3), MpscPushResult::Ok));
        assert!(matches!(c.pop(), MpscPopResult::Ok(1)));
        assert!(matches!(c.pop(), MpscPopResult::Ok(2)));
        assert!(matches!(c.pop(), MpscPopResult::Ok(3)));
        assert!(matches!(c.pop(), MpscPopResult::Empty));
    }

    #[test]
    fn full_returns_item_back() {
        let (p, _c) = new_mpsc::<u32>(2);
        assert!(matches!(p.push(1), MpscPushResult::Ok));
        assert!(matches!(p.push(2), MpscPushResult::Ok));
        match p.push(3) {
            MpscPushResult::Full(v) => assert_eq!(v, 3),
            _ => panic!("expected Full"),
        }
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
            let (p, _c) = new_mpsc::<DropCount<'_>>(4);
            assert!(matches!(p.push(DropCount(&count)), MpscPushResult::Ok));
            assert!(matches!(p.push(DropCount(&count)), MpscPushResult::Ok));
            assert!(matches!(p.push(DropCount(&count)), MpscPushResult::Ok));
        }
        assert_eq!(count.load(O::Relaxed), 3);
    }

    #[test]
    fn concurrent_mpsc_stress_no_item_loss() {
        let (p_proto, c) = new_mpsc::<u32>(64);
        let n_per = 5_000u32;
        let n_producers = 4u32;
        let total = (n_per * n_producers) as usize;
        let consumed = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));

        let consumed_c = Arc::clone(&consumed);
        let sum_c = Arc::clone(&sum);
        let consumer = thread::spawn(move || {
            while consumed_c.load(O::Relaxed) < total {
                match c.pop() {
                    MpscPopResult::Ok(v) => {
                        consumed_c.fetch_add(1, O::Relaxed);
                        sum_c.fetch_add(v as usize, O::Relaxed);
                    }
                    MpscPopResult::Empty => std::thread::yield_now(),
                }
            }
        });

        let mut producers = Vec::new();
        for p_id in 0..n_producers {
            let p = p_proto.clone();
            producers.push(thread::spawn(move || {
                for i in 0..n_per {
                    let item = p_id * n_per + i;
                    loop {
                        match p.push(item) {
                            MpscPushResult::Ok => break,
                            MpscPushResult::Full(_) => std::thread::yield_now(),
                        }
                    }
                }
            }));
        }

        for h in producers {
            h.join().expect("p");
        }
        consumer.join().expect("c");

        let expected: usize = (0..n_producers)
            .flat_map(|p| (0..n_per).map(move |i| (p * n_per + i) as usize))
            .sum();
        assert_eq!(consumed.load(O::Relaxed), total);
        assert_eq!(sum.load(O::Relaxed), expected,
            "sum invariant: each pushed item consumed exactly once");
    }
}

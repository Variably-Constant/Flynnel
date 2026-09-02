//! Single-Producer Single-Consumer (SPSC) bounded ring - the
//! fastest possible bounded queue. No CAS on push or pop; just
//! an Acquire/Release pair on owner-private tail/head counters.
//!
//! ## Why a separate SPSC type
//!
//! The MPMC ring in [`crate::sched::flynnel_ring`] pays CAS-loop
//! overhead on `tail` (producer) and `head` (consumer) because
//! multiple producers / consumers race to claim positions. In
//! SPSC scenarios (one producer thread, one consumer thread), no
//! such race exists; the CAS is pure overhead.
//!
//! This module's primitive uses Lamport's 1983 SPSC ring + the
//! standard Acquire/Release ordering pattern:
//!
//! - Producer: Acquire-load `head` (to check capacity); write
//!   data into slot; Release-store `tail + 1`
//! - Consumer: Acquire-load `tail` (to check availability); read
//!   data from slot; Release-store `head + 1`
//!
//! The Release-store of tail synchronizes-with the Acquire-load
//! of tail on the consumer side; the slot's data write
//! happens-before the consumer's read. No CAS needed because
//! only one writer touches each counter.
//!
//! A dedicated SPSC type with no CAS path targets the
//! architectural throughput limit on this host.
//!
//! ## Cross-platform discipline
//!
//! Pure AtomicU64 + Acquire / Release / Relaxed; no x86-specific
//! intrinsics. Linux/macOS/Windows on x86_64/aarch64/armv7.

#![allow(clippy::missing_errors_doc)]

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Outcome of [`Producer::push`].
#[derive(Debug, PartialEq, Eq)]
pub enum SpscPushResult<T> {
    /// Item enqueued.
    Ok,
    /// Ring is at capacity; caller decides whether to retry or
    /// drop.
    Full(T),
}

/// Outcome of [`Consumer::pop`].
#[derive(Debug, PartialEq, Eq)]
pub enum SpscPopResult<T> {
    /// Got an item.
    Ok(T),
    /// Ring is empty.
    Empty,
}

/// Shared header for the SPSC ring. `tail` and `head` get
/// dedicated cache lines to prevent false sharing between
/// producer and consumer.
#[repr(C, align(64))]
struct SpscHeader<T> {
    /// Producer-written count. Producer Release-stores this on
    /// publish; consumer Acquire-loads to learn what's available.
    tail: AtomicU64,
    _pad_tail: [u8; 56],
    /// Consumer-written count. Consumer Release-stores this on
    /// consume; producer Acquire-loads to learn what's free.
    head: AtomicU64,
    _pad_head: [u8; 56],
    capacity: u64,
    capacity_mask: u64,
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
}

// SAFETY: Lamport SPSC ordering discipline. T: Send required
// because the data crosses the producer-consumer thread boundary.
unsafe impl<T: Send> Send for SpscHeader<T> {}
unsafe impl<T: Send> Sync for SpscHeader<T> {}

/// Single-producer handle. Only ONE thread may hold this; the
/// `&self` API does NOT make `Producer` safely shareable across
/// producer threads. Sharing violates the SPSC invariant.
pub struct Producer<T> {
    inner: Arc<SpscHeader<T>>,
}

/// Single-consumer handle. Only ONE thread may hold this.
pub struct Consumer<T> {
    inner: Arc<SpscHeader<T>>,
}

// Send so producer/consumer can be moved to a different thread
// at construction. !Sync (default) so they can't be shared
// between threads (would violate single-owner).
unsafe impl<T: Send> Send for Producer<T> {}
unsafe impl<T: Send> Send for Consumer<T> {}

/// Construct a new SPSC ring with `capacity` slots (rounded up to
/// next power of two; minimum 2). Returns a (producer, consumer)
/// handle pair.
pub fn new_spsc<T: Send>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    let capacity = capacity.max(2).next_power_of_two();
    let mut buf = Vec::with_capacity(capacity);
    for _ in 0..capacity {
        buf.push(UnsafeCell::new(MaybeUninit::uninit()));
    }
    let inner = Arc::new(SpscHeader {
        tail: AtomicU64::new(0),
        _pad_tail: [0u8; 56],
        head: AtomicU64::new(0),
        _pad_head: [0u8; 56],
        capacity: capacity as u64,
        capacity_mask: (capacity as u64) - 1,
        buffer: buf.into_boxed_slice(),
    });
    (
        Producer {
            inner: Arc::clone(&inner),
        },
        Consumer { inner },
    )
}

impl<T: Send> Producer<T> {
    /// Push an item. Returns `Ok` on success, `Full(item)` when
    /// the ring is at capacity. No CAS; one Acquire-load + one
    /// Release-store on hot path.
    #[inline(always)]
    pub fn push(&self, item: T) -> SpscPushResult<T> {
        let h = &*self.inner;
        // Owner-private read of tail (we're the only writer).
        // Relaxed is sufficient because our own writes have
        // program-order visibility to ourselves.
        let tail = h.tail.load(Ordering::Relaxed);
        // Acquire-load head: synchronizes-with the consumer's
        // Release-store of head; tells us what slots are free.
        let head = h.head.load(Ordering::Acquire);
        if tail.wrapping_sub(head) >= h.capacity {
            // Cold path: ring full.
            core::hint::cold_path();
            return SpscPushResult::Full(item);
        }
        let idx = (tail & h.capacity_mask) as usize;
        // SAFETY: idx is masked by capacity_mask so always in
        // [0, capacity). We're the only writer; no race on the
        // slot until our Release-store of tail+1 below
        // synchronizes-with the consumer's Acquire-load of tail.
        unsafe {
            (*h.buffer.get_unchecked(idx).get()).write(item);
        }
        // Release-store of tail+1 publishes the slot write; the
        // consumer's Acquire-load of tail will see this, and the
        // happens-before relation makes the data visible.
        h.tail.store(tail + 1, Ordering::Release);
        SpscPushResult::Ok
    }
}

impl<T: Send> Consumer<T> {
    /// Pop an item. Returns `Ok(item)` on success, `Empty` when
    /// the ring is empty. No CAS; one Acquire-load + one
    /// Release-store on hot path.
    #[inline(always)]
    pub fn pop(&self) -> SpscPopResult<T> {
        let h = &*self.inner;
        let head = h.head.load(Ordering::Relaxed);
        let tail = h.tail.load(Ordering::Acquire);
        if head >= tail {
            core::hint::cold_path();
            return SpscPopResult::Empty;
        }
        let idx = (head & h.capacity_mask) as usize;
        // SAFETY: idx masked; the producer's Release-store of
        // tail synchronized-with our Acquire-load above, so the
        // slot's data write is visible here.
        let item = unsafe { (*h.buffer.get_unchecked(idx).get()).assume_init_read() };
        h.head.store(head + 1, Ordering::Release);
        SpscPopResult::Ok(item)
    }
}

impl<T> Drop for SpscHeader<T> {
    fn drop(&mut self) {
        // Drain unclaimed items. Walks [head, tail) for live
        // slots that need their destructors run.
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        let mut pos = head;
        while pos < tail {
            let idx = (pos & self.capacity_mask) as usize;
            // SAFETY: pos < tail means the producer wrote this
            // slot and the consumer hasn't drained it; the slot
            // holds initialized T.
            unsafe {
                (*self.buffer[idx].get()).assume_init_drop();
            }
            pos += 1;
        }
    }
}

/// Marker symbol for linkage confirmation.
#[unsafe(no_mangle)]
pub static __flynnel_marker_ring_spsc_push: u8 = 0;
/// Companion marker for the SPSC pop hot path.
#[unsafe(no_mangle)]
pub static __flynnel_marker_ring_spsc_pop: u8 = 0;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as O};
    use std::thread;

    #[test]
    fn push_pop_single_thread() {
        let (p, c) = new_spsc::<u32>(8);
        assert!(matches!(p.push(1), SpscPushResult::Ok));
        assert!(matches!(p.push(2), SpscPushResult::Ok));
        assert!(matches!(p.push(3), SpscPushResult::Ok));
        assert!(matches!(c.pop(), SpscPopResult::Ok(1)));
        assert!(matches!(c.pop(), SpscPopResult::Ok(2)));
        assert!(matches!(c.pop(), SpscPopResult::Ok(3)));
        assert!(matches!(c.pop(), SpscPopResult::Empty));
    }

    #[test]
    fn full_returns_item_back() {
        let (p, _c) = new_spsc::<u32>(2);
        assert!(matches!(p.push(1), SpscPushResult::Ok));
        assert!(matches!(p.push(2), SpscPushResult::Ok));
        match p.push(3) {
            SpscPushResult::Full(v) => assert_eq!(v, 3),
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
            let (p, _c) = new_spsc::<DropCount<'_>>(4);
            assert!(matches!(p.push(DropCount(&count)), SpscPushResult::Ok));
            assert!(matches!(p.push(DropCount(&count)), SpscPushResult::Ok));
            assert!(matches!(p.push(DropCount(&count)), SpscPushResult::Ok));
        }
        assert_eq!(count.load(O::Relaxed), 3);
    }

    #[test]
    fn concurrent_spsc_stress_no_item_loss() {
        let (p, c) = new_spsc::<u32>(64);
        let n = 50_000u32;
        let consumed = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));

        let consumed_c = Arc::clone(&consumed);
        let sum_c = Arc::clone(&sum);
        let consumer = thread::spawn(move || {
            while consumed_c.load(O::Relaxed) < n as usize {
                match c.pop() {
                    SpscPopResult::Ok(v) => {
                        consumed_c.fetch_add(1, O::Relaxed);
                        sum_c.fetch_add(v as usize, O::Relaxed);
                    }
                    SpscPopResult::Empty => std::thread::yield_now(),
                }
            }
        });

        let producer = thread::spawn(move || {
            for i in 0..n {
                loop {
                    match p.push(i) {
                        SpscPushResult::Ok => break,
                        SpscPushResult::Full(_) => std::thread::yield_now(),
                    }
                }
            }
        });

        producer.join().expect("producer");
        consumer.join().expect("consumer");

        let expected: usize = (0..n as usize).sum();
        assert_eq!(consumed.load(O::Relaxed), n as usize);
        assert_eq!(sum.load(O::Relaxed), expected,
            "sum invariant: each pushed item consumed exactly once");
    }

    #[test]
    fn capacity_rounds_to_power_of_two() {
        let (p, _c) = new_spsc::<u32>(5);
        for i in 0..8 {
            assert!(matches!(p.push(i), SpscPushResult::Ok),
                "should fit 8 items (5 rounded to 8)");
        }
        match p.push(99) {
            SpscPushResult::Full(_) => {}
            _ => panic!("9th push should be Full"),
        }
    }
}

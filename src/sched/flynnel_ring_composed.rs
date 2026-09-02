//! Composed N-by-M ring: MPMC built from N*M Lamport SPSC rings
//! instead of one Vyukov MPMC ring. Per-producer FIFO preserved;
//! global FIFO traded for per-op throughput.
//!
//! ## The composition insight
//!
//! Vyukov per-slot-sequence MPMC (the standard textbook design)
//! pays CAS-loop overhead on EVERY push (producers contend on
//! `tail`) and EVERY pop (consumers contend on `head`). Lamport
//! SPSC pays NO CAS at all - just an Acquire-load + Release-store
//! pair per op.
//!
//! Composing N producers and M consumers as N*M SPSC rings gives
//! each (producer, consumer) pair its own dedicated SPSC ring.
//! Producer i pushes to ring `(i, target_consumer)`; consumer j
//! drains rings `(*, j)` (all rings targeting consumer j) via
//! round-robin walk.
//!
//! ## When this wins
//!
//! - **Mailbox MPSC**: 1 consumer, N producers - N rings owned
//!   by the consumer.
//! - **N-to-M scheduler**: each consumer round-robins / work-
//!   steals across its assigned ring set. Pattern used by
//!   LMAX Disruptor and Go's per-P runqueues.
//!
//! ## What you give up
//!
//! Global FIFO ordering across producers. Items from producer A
//! and producer B can interleave at the consumer in any drain
//! order. **Per-producer FIFO IS preserved**: if producer i
//! pushes a then b, the consumer always sees a before b from
//! that producer's stream.
//!
//! For flynnel's mailbox / Injector / io_pool reply rings,
//! per-producer FIFO is what callers actually need. Global FIFO
//! is over-engineered (no caller is comparing across producer
//! streams).
//!
//! ## Cross-platform discipline
//!
//! Built directly on [`crate::sched::flynnel_ring_spsc`]; inherits
//! its pure-AtomicU64 / Acquire / Release / Relaxed shape. No
//! x86-specific intrinsics. Linux/macOS/Windows on
//! x86_64/aarch64/armv7.

#![allow(clippy::missing_errors_doc)]

use core::cell::Cell;

use crate::sched::flynnel_ring_spsc::{
    Consumer as SpscConsumer, Producer as SpscProducer, SpscPopResult,
    SpscPushResult, new_spsc,
};

/// Outcome of a composed-MPSC push.
#[derive(Debug, PartialEq, Eq)]
pub enum ComposedPushResult<T> {
    /// Item enqueued on this producer's dedicated SPSC ring.
    Ok,
    /// This producer's dedicated ring is full; caller decides
    /// whether to retry, back off, or drop.
    Full(T),
}

impl<T> ComposedPushResult<T> {
    /// Fire-and-forget helper: collapses to `Option<T>` of the
    /// rejected item. `Ok` -> `None`; `Full(t)` -> `Some(t)`.
    #[inline(always)]
    pub fn ok(self) -> Option<T> {
        match self {
            ComposedPushResult::Ok => None,
            ComposedPushResult::Full(t) => Some(t),
        }
    }

    /// Returns true if the push succeeded.
    #[inline(always)]
    pub fn is_ok(&self) -> bool {
        matches!(self, ComposedPushResult::Ok)
    }
}

/// Outcome of a composed-MPSC pop.
#[derive(Debug, PartialEq, Eq)]
pub enum ComposedPopResult<T> {
    /// Got an item from one of the N rings. The cursor advances
    /// to the next ring on the next pop call.
    Ok(T),
    /// All N rings are empty.
    Empty,
}

/// Composed MPSC ring: N producers, 1 consumer. Each producer
/// owns its dedicated SPSC ring; the consumer round-robins
/// across all N rings.
///
/// Construction: caller specifies `n_producers` and `capacity`
/// (per-producer ring). Returns one consumer + a vector of N
/// producer handles. Move each handle to the producer thread
/// that will use it.
pub struct ComposedMpsc<T: Send> {
    /// Producer-side handles, one per producer. Move each out to
    /// the owning producer thread.
    pub producers: Vec<SpscProducer<T>>,
    /// Consumer handle. Hold this on the single consumer thread.
    pub consumer: ComposedMpscConsumer<T>,
}

/// Single-consumer side. Owns the consumer halves of all N
/// underlying SPSC rings + a round-robin cursor.
pub struct ComposedMpscConsumer<T: Send> {
    rings: Vec<SpscConsumer<T>>,
    /// Round-robin cursor for the next-ring-to-probe; Cell so a
    /// `&self` API can advance it.
    cursor: Cell<usize>,
}

// SAFETY: cursor is Cell (single-owner via the !Sync default for
// Cell); rings each enforce SPSC consumer single-owner discipline.
unsafe impl<T: Send> Send for ComposedMpscConsumer<T> {}

/// Construct a composed MPSC ring with `n_producers` producer
/// handles, each backed by a per-producer SPSC ring of
/// `capacity_per_producer` slots (rounded up to next pow2;
/// minimum 2). Returns the [`ComposedMpsc`] struct holding the
/// producer handle vector + the single consumer.
pub fn new_composed_mpsc<T: Send>(
    n_producers: usize,
    capacity_per_producer: usize,
) -> ComposedMpsc<T> {
    let n = n_producers.max(1);
    let mut producers = Vec::with_capacity(n);
    let mut consumers = Vec::with_capacity(n);
    for _ in 0..n {
        let (p, c) = new_spsc::<T>(capacity_per_producer);
        producers.push(p);
        consumers.push(c);
    }
    ComposedMpsc {
        producers,
        consumer: ComposedMpscConsumer {
            rings: consumers,
            cursor: Cell::new(0),
        },
    }
}

impl<T: Send> SpscProducer<T> {
    /// Convenience push for use through a producer handle in the
    /// composed primitive: returns the [`ComposedPushResult`] so
    /// composed-mpsc and direct-spsc callers can use the same
    /// match shape. Internally identical to the SPSC push.
    #[inline(always)]
    pub fn push_composed(&self, item: T) -> ComposedPushResult<T> {
        match self.push(item) {
            SpscPushResult::Ok => ComposedPushResult::Ok,
            SpscPushResult::Full(t) => ComposedPushResult::Full(t),
        }
    }
}

impl<T: Send> ComposedMpscConsumer<T> {
    /// Pop one item via round-robin walk across the N rings.
    /// Starts at the current cursor position and probes each
    /// ring once; returns the first item found and advances the
    /// cursor PAST the ring it took from (so subsequent pops
    /// don't starve other producers).
    ///
    /// Empty case: all N rings probed once with no item; returns
    /// Empty. The cursor advances by 1 even on empty so the next
    /// call starts at a different ring (prevents repeated probe
    /// pattern thrashing).
    #[inline]
    pub fn pop(&self) -> ComposedPopResult<T> {
        let n = self.rings.len();
        if n == 0 {
            return ComposedPopResult::Empty;
        }
        let start = self.cursor.get();
        for offset in 0..n {
            let idx = (start + offset) % n;
            // SAFETY: idx in [0, n) so the indexed access is safe.
            // The ring's pop is the standard SPSC pop.
            match self.rings[idx].pop() {
                SpscPopResult::Ok(item) => {
                    // Advance cursor past this ring; next pop
                    // starts from the next ring (fair round-robin).
                    self.cursor.set((idx + 1) % n);
                    return ComposedPopResult::Ok(item);
                }
                SpscPopResult::Empty => continue,
            }
        }
        // All N rings empty.
        core::hint::cold_path();
        self.cursor.set((start + 1) % n);
        ComposedPopResult::Empty
    }

    /// Number of underlying SPSC rings (= producer count).
    #[inline]
    pub fn ring_count(&self) -> usize {
        self.rings.len()
    }
}

/// Marker symbol for linkage confirmation.
#[unsafe(no_mangle)]
pub static __flynnel_marker_ring_composed_push: u8 = 0;
/// Companion marker for the composed-MPSC pop hot path.
#[unsafe(no_mangle)]
pub static __flynnel_marker_ring_composed_pop: u8 = 0;

// =====================================================================
// MPMC grid: N producers x M consumers = N*M SPSC rings.
// Every (producer i, consumer j) pair shares a dedicated SPSC ring;
// producer i picks a target column j via round-robin and pushes;
// consumer j round-robins across its row of N rings.
// =====================================================================

/// Outcome of an MPMC-grid push.
#[derive(Debug, PartialEq, Eq)]
pub enum GridPushResult<T> {
    /// Item enqueued on some (i, j) ring; the producer's column
    /// cursor advanced.
    Ok,
    /// Every consumer column the producer probed was full; caller
    /// decides whether to retry or drop.
    Full(T),
}

impl<T> GridPushResult<T> {
    /// Fire-and-forget helper: collapses to `Option<T>` of the
    /// rejected item.
    #[inline(always)]
    pub fn ok(self) -> Option<T> {
        match self {
            GridPushResult::Ok => None,
            GridPushResult::Full(t) => Some(t),
        }
    }

    /// Returns true if the push succeeded.
    #[inline(always)]
    pub fn is_ok(&self) -> bool {
        matches!(self, GridPushResult::Ok)
    }
}

/// Outcome of an MPMC-grid pop.
#[derive(Debug, PartialEq, Eq)]
pub enum GridPopResult<T> {
    /// Got an item from one of this consumer's N source rings.
    Ok(T),
    /// All N source rings are empty.
    Empty,
}

/// Producer-side handle for the MPMC grid. Holds M dedicated SPSC
/// producer handles - one per consumer column - and a round-robin
/// cursor that picks the next consumer to target.
pub struct GridProducer<T: Send> {
    handles: Vec<SpscProducer<T>>,
    cursor: Cell<usize>,
}

// SAFETY: cursor is a Cell (`!Sync`); each underlying SpscProducer
// enforces single-owner discipline. Sending the GridProducer to a
// thread does not violate SPSC because no other thread holds the
// same producer handles.
unsafe impl<T: Send> Send for GridProducer<T> {}

impl<T: Send> GridProducer<T> {
    /// Number of consumer columns this producer can target.
    #[inline]
    pub fn consumer_count(&self) -> usize {
        self.handles.len()
    }

    /// Push an item; tries each consumer column in round-robin
    /// order. On Ok, the cursor advances PAST the column it
    /// pushed to (next push starts at the next column). On Full,
    /// the cursor advances by 1 so the next call probes a
    /// different starting column.
    #[inline]
    pub fn push(&self, item: T) -> GridPushResult<T> {
        let m = self.handles.len();
        if m == 0 {
            return GridPushResult::Full(item);
        }
        let start = self.cursor.get();
        let mut carry = item;
        for offset in 0..m {
            let idx = (start + offset) % m;
            match self.handles[idx].push(carry) {
                SpscPushResult::Ok => {
                    self.cursor.set((idx + 1) % m);
                    return GridPushResult::Ok;
                }
                SpscPushResult::Full(t) => {
                    carry = t;
                    continue;
                }
            }
        }
        core::hint::cold_path();
        self.cursor.set((start + 1) % m);
        GridPushResult::Full(carry)
    }
}

/// Consumer-side handle for the MPMC grid. Owns N dedicated SPSC
/// consumer handles - one per producer row - and a round-robin
/// cursor that picks the next producer to drain.
pub struct GridConsumer<T: Send> {
    handles: Vec<SpscConsumer<T>>,
    cursor: Cell<usize>,
}

// SAFETY: cursor is a Cell (`!Sync`); each underlying SpscConsumer
// enforces single-owner discipline.
unsafe impl<T: Send> Send for GridConsumer<T> {}

impl<T: Send> GridConsumer<T> {
    /// Number of producer rows this consumer drains from.
    #[inline]
    pub fn producer_count(&self) -> usize {
        self.handles.len()
    }

    /// Pop one item via round-robin walk across the N producer
    /// rings; same semantics as [`ComposedMpscConsumer::pop`].
    #[inline]
    pub fn pop(&self) -> GridPopResult<T> {
        let n = self.handles.len();
        if n == 0 {
            return GridPopResult::Empty;
        }
        let start = self.cursor.get();
        for offset in 0..n {
            let idx = (start + offset) % n;
            match self.handles[idx].pop() {
                SpscPopResult::Ok(item) => {
                    self.cursor.set((idx + 1) % n);
                    return GridPopResult::Ok(item);
                }
                SpscPopResult::Empty => continue,
            }
        }
        core::hint::cold_path();
        self.cursor.set((start + 1) % n);
        GridPopResult::Empty
    }
}

/// MPMC grid: N producers x M consumers, N*M dedicated SPSC rings.
/// Returned by [`new_composed_mpmc`]; the caller moves each
/// `producers[i]` to producer thread `i` and each `consumers[j]`
/// to consumer thread `j`.
pub struct ComposedMpmc<T: Send> {
    /// One handle per producer thread; length = N.
    pub producers: Vec<GridProducer<T>>,
    /// One handle per consumer thread; length = M.
    pub consumers: Vec<GridConsumer<T>>,
}

/// Construct an N-by-M MPMC grid: `n_producers` producer handles
/// and `n_consumers` consumer handles, each (i, j) pair backed by
/// its own dedicated SPSC ring of `capacity_per_pair` slots
/// (rounded up to next pow2; minimum 2). Total ring count =
/// `n_producers * n_consumers`.
pub fn new_composed_mpmc<T: Send>(
    n_producers: usize,
    n_consumers: usize,
    capacity_per_pair: usize,
) -> ComposedMpmc<T> {
    let n = n_producers.max(1);
    let m = n_consumers.max(1);
    // Build N rows of M SPSC rings: rings[i][j] = ring from
    // producer i to consumer j. We iterate over the producer
    // rows directly (mutable) and use enumerate to address the
    // consumer columns by index in the inner step.
    let mut producer_handles: Vec<Vec<SpscProducer<T>>> =
        (0..n).map(|_| Vec::with_capacity(m)).collect();
    let mut consumer_handles: Vec<Vec<SpscConsumer<T>>> =
        (0..m).map(|_| Vec::with_capacity(n)).collect();
    for producer_row in producer_handles.iter_mut() {
        for consumer_col in consumer_handles.iter_mut() {
            let (p, c) = new_spsc::<T>(capacity_per_pair);
            producer_row.push(p);
            consumer_col.push(c);
        }
    }
    let producers = producer_handles
        .into_iter()
        .map(|handles| GridProducer { handles, cursor: Cell::new(0) })
        .collect();
    let consumers = consumer_handles
        .into_iter()
        .map(|handles| GridConsumer { handles, cursor: Cell::new(0) })
        .collect();
    ComposedMpmc { producers, consumers }
}

/// Marker symbol for the MPMC grid push path.
#[unsafe(no_mangle)]
pub static __flynnel_marker_ring_grid_push: u8 = 0;
/// Marker symbol for the MPMC grid pop path.
#[unsafe(no_mangle)]
pub static __flynnel_marker_ring_grid_pop: u8 = 0;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as O};
    use std::thread;

    #[test]
    fn push_pop_round_trip_single_producer() {
        let composed = new_composed_mpsc::<u32>(1, 8);
        let p = &composed.producers[0];
        let c = &composed.consumer;
        assert!(matches!(p.push_composed(1), ComposedPushResult::Ok));
        assert!(matches!(p.push_composed(2), ComposedPushResult::Ok));
        assert!(matches!(c.pop(), ComposedPopResult::Ok(1)));
        assert!(matches!(c.pop(), ComposedPopResult::Ok(2)));
        assert!(matches!(c.pop(), ComposedPopResult::Empty));
    }

    #[test]
    fn round_robin_drains_fairly() {
        let composed = new_composed_mpsc::<u32>(3, 8);
        let p0 = &composed.producers[0];
        let p1 = &composed.producers[1];
        let p2 = &composed.producers[2];
        let c = &composed.consumer;
        // Push 1 item to each ring.
        p0.push_composed(100).ok();
        p1.push_composed(200).ok();
        p2.push_composed(300).ok();
        // Round-robin: cursor starts at 0; gets ring 0's item.
        assert!(matches!(c.pop(), ComposedPopResult::Ok(100)));
        // Cursor now at 1; gets ring 1's item.
        assert!(matches!(c.pop(), ComposedPopResult::Ok(200)));
        // Cursor at 2; gets ring 2's item.
        assert!(matches!(c.pop(), ComposedPopResult::Ok(300)));
        // All empty.
        assert!(matches!(c.pop(), ComposedPopResult::Empty));
    }

    #[test]
    fn per_producer_fifo_preserved() {
        let composed = new_composed_mpsc::<(u32, u32)>(2, 8);
        let p0 = &composed.producers[0];
        let p1 = &composed.producers[1];
        let c = &composed.consumer;
        // Producer 0 pushes (0, 0), (0, 1), (0, 2)
        for i in 0..3 {
            p0.push_composed((0, i)).ok();
        }
        // Producer 1 pushes (1, 0), (1, 1), (1, 2)
        for i in 0..3 {
            p1.push_composed((1, i)).ok();
        }
        // Drain - collect items per producer.
        let mut from_p0 = Vec::new();
        let mut from_p1 = Vec::new();
        while let ComposedPopResult::Ok((pid, val)) = c.pop() {
            match pid {
                0 => from_p0.push(val),
                1 => from_p1.push(val),
                _ => panic!("unexpected pid {pid}"),
            }
        }
        // Per-producer FIFO: each producer's stream must be in
        // push order.
        assert_eq!(from_p0, vec![0, 1, 2], "producer 0 FIFO broken");
        assert_eq!(from_p1, vec![0, 1, 2], "producer 1 FIFO broken");
    }

    #[test]
    fn full_returns_item_back() {
        let composed = new_composed_mpsc::<u32>(2, 2);
        let p0 = &composed.producers[0];
        assert!(matches!(p0.push_composed(1), ComposedPushResult::Ok));
        assert!(matches!(p0.push_composed(2), ComposedPushResult::Ok));
        match p0.push_composed(3) {
            ComposedPushResult::Full(v) => assert_eq!(v, 3),
            _ => panic!("expected Full"),
        }
    }

    // =============================================================
    // ComposedMpmc (N-by-M grid) tests
    // =============================================================

    #[test]
    fn mpmc_grid_single_pair_round_trip() {
        let grid = new_composed_mpmc::<u32>(1, 1, 8);
        let p = &grid.producers[0];
        let c = &grid.consumers[0];
        assert!(matches!(p.push(42), GridPushResult::Ok));
        assert!(matches!(c.pop(), GridPopResult::Ok(42)));
        assert!(matches!(c.pop(), GridPopResult::Empty));
    }

    #[test]
    fn mpmc_grid_producer_round_robins_across_consumers() {
        let grid = new_composed_mpmc::<u32>(1, 3, 4);
        let p = &grid.producers[0];
        // Push 3 items; producer cursor visits consumers 0, 1, 2.
        p.push(10).ok();
        p.push(20).ok();
        p.push(30).ok();
        // Each consumer should have exactly 1 item.
        assert!(matches!(grid.consumers[0].pop(), GridPopResult::Ok(10)));
        assert!(matches!(grid.consumers[1].pop(), GridPopResult::Ok(20)));
        assert!(matches!(grid.consumers[2].pop(), GridPopResult::Ok(30)));
        assert!(matches!(grid.consumers[0].pop(), GridPopResult::Empty));
    }

    #[test]
    fn mpmc_grid_consumer_round_robins_across_producers() {
        let grid = new_composed_mpmc::<u32>(3, 1, 4);
        // Each producer pushes one item; producer i targets the
        // single consumer (no choice).
        grid.producers[0].push(100).ok();
        grid.producers[1].push(200).ok();
        grid.producers[2].push(300).ok();
        let c = &grid.consumers[0];
        // Consumer starts at cursor 0; visits producers 0, 1, 2 in
        // order. Round-robin.
        assert!(matches!(c.pop(), GridPopResult::Ok(100)));
        assert!(matches!(c.pop(), GridPopResult::Ok(200)));
        assert!(matches!(c.pop(), GridPopResult::Ok(300)));
        assert!(matches!(c.pop(), GridPopResult::Empty));
    }

    #[test]
    fn mpmc_grid_full_returns_item_back() {
        let grid = new_composed_mpmc::<u32>(1, 1, 2);
        let p = &grid.producers[0];
        assert!(p.push(1).is_ok());
        assert!(p.push(2).is_ok());
        match p.push(3) {
            GridPushResult::Full(v) => assert_eq!(v, 3),
            _ => panic!("expected Full"),
        }
    }

    #[test]
    fn mpmc_grid_4p_4c_concurrent_no_item_loss() {
        let n_producers = 4;
        let n_consumers = 4;
        let n_per_producer = 5_000u32;
        let mut grid = new_composed_mpmc::<u32>(n_producers, n_consumers, 64);
        let producers: Vec<_> = grid.producers.drain(..).collect();
        let consumers: Vec<_> = grid.consumers.drain(..).collect();

        let total = (n_per_producer as usize) * n_producers;
        let consumed = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));

        let mut cons_handles = Vec::new();
        for c in consumers.into_iter() {
            let consumed_c = Arc::clone(&consumed);
            let sum_c = Arc::clone(&sum);
            cons_handles.push(thread::spawn(move || {
                loop {
                    match c.pop() {
                        GridPopResult::Ok(v) => {
                            consumed_c.fetch_add(1, O::Relaxed);
                            sum_c.fetch_add(v as usize, O::Relaxed);
                        }
                        GridPopResult::Empty => {
                            if consumed_c.load(O::Relaxed) >= total {
                                break;
                            }
                            std::thread::yield_now();
                        }
                    }
                }
            }));
        }

        let mut prod_handles = Vec::new();
        for (p_id, p) in producers.into_iter().enumerate() {
            prod_handles.push(thread::spawn(move || {
                for i in 0..n_per_producer {
                    let item = (p_id as u32) * n_per_producer + i;
                    loop {
                        match p.push(item) {
                            GridPushResult::Ok => break,
                            GridPushResult::Full(_) => std::thread::yield_now(),
                        }
                    }
                }
            }));
        }

        for h in prod_handles {
            h.join().expect("p");
        }
        for h in cons_handles {
            h.join().expect("c");
        }

        let expected: usize = (0..n_producers as u32)
            .flat_map(|p| (0..n_per_producer).map(move |i| (p * n_per_producer + i) as usize))
            .sum();
        assert_eq!(consumed.load(O::Relaxed), total);
        assert_eq!(sum.load(O::Relaxed), expected,
            "sum invariant: each pushed item consumed exactly once across the grid");
    }

    #[test]
    fn concurrent_4_producers_1_consumer_no_item_loss() {
        let n_producers = 4;
        let n_per = 10_000u32;
        let mut composed = new_composed_mpsc::<u32>(n_producers, 64);
        let consumer = composed.consumer;
        let producers: Vec<_> = composed.producers.drain(..).collect();

        let total = (n_per as usize) * n_producers;
        let consumed = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));

        let consumed_c = Arc::clone(&consumed);
        let sum_c = Arc::clone(&sum);
        let cons = thread::spawn(move || {
            while consumed_c.load(O::Relaxed) < total {
                match consumer.pop() {
                    ComposedPopResult::Ok(v) => {
                        consumed_c.fetch_add(1, O::Relaxed);
                        sum_c.fetch_add(v as usize, O::Relaxed);
                    }
                    ComposedPopResult::Empty => std::thread::yield_now(),
                }
            }
        });

        let mut prod_handles = Vec::new();
        for (p_id, p) in producers.into_iter().enumerate() {
            prod_handles.push(thread::spawn(move || {
                for i in 0..n_per {
                    let item = (p_id as u32) * n_per + i;
                    loop {
                        match p.push_composed(item) {
                            ComposedPushResult::Ok => break,
                            ComposedPushResult::Full(_) => std::thread::yield_now(),
                        }
                    }
                }
            }));
        }

        for h in prod_handles {
            h.join().expect("p");
        }
        cons.join().expect("c");

        let expected: usize = (0..n_producers as u32)
            .flat_map(|p| (0..n_per).map(move |i| (p * n_per + i) as usize))
            .sum();
        assert_eq!(consumed.load(O::Relaxed), total);
        assert_eq!(sum.load(O::Relaxed), expected,
            "sum invariant: each pushed item consumed exactly once");
    }
}

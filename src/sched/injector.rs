//! Global injector queue: external submitters push jobs here;
//! arena workers steal from it when their local deque is empty.
//!
//! Replacement for `crossbeam::deque::Injector` built on top of
//! [`crate::sched::flynnel_ring::FlynnelRing`] (Vyukov MPMC). The
//! same producer / consumer protocol; same Empty / Success / Retry
//! steal semantics; bounded rather than linked-chunk unbounded.
//!
//! ## Capacity
//!
//! Bounded at construction. The default capacity of
//! [`DEFAULT_INJECTOR_CAPACITY`] (= 4096) absorbs typical
//! external-submission bursts; on overflow, the submitter spins
//! until a worker drains a slot (back-pressure).
//!
//! ## API surface
//!
//! - [`Injector::new`] / [`Injector::with_capacity`] - construct
//! - [`Injector::push`] - infallible push; spins on full
//! - [`Injector::try_push`] - non-blocking push; returns Err on full
//! - [`Injector::steal`] - returns Success / Empty / Retry
//! - [`Injector::is_empty`] - hint
//!
//! ## Cross-platform discipline
//!
//! Pure AtomicU64 + Acquire / Release / Relaxed; no x86-specific
//! intrinsics. Linux/macOS/Windows on x86_64/aarch64/armv7.

#![allow(clippy::missing_errors_doc)]

use crate::sched::flynnel_ring::{FlynnelRing, PopResult, PushResult};

/// Default slot count for [`Injector::new`]. Sized to absorb a
/// burst of external submissions without back-pressure on the
/// submitter side; on host systems with deeper backlog needs use
/// [`Injector::with_capacity`].
pub const DEFAULT_INJECTOR_CAPACITY: usize = 4096;

/// Outcome of [`Injector::steal`]. Same three-arm shape as the
/// owner-side Chase-Lev steal protocol so a single match-arm
/// pattern covers both the local deque and the global injector.
#[derive(Debug, PartialEq, Eq)]
pub enum InjectorSteal<T> {
    /// Got an item from the injector.
    Success(T),
    /// Injector observed empty.
    Empty,
    /// Kept for API symmetry with the Chase-Lev owner-deque steal
    /// protocol. NEVER RETURNED by [`Injector::steal`] because
    /// [`crate::sched::flynnel_ring::FlynnelRing::pop`] loops on
    /// CAS contention internally; callers that pattern-match
    /// against the full three-arm shape can still write the
    /// arm and it stays unreachable.
    Retry,
}

/// Global MPMC fork queue. One per [`crate::sched::arena_local::LocalArena`].
pub struct Injector<T: Send> {
    ring: FlynnelRing<T>,
}

impl<T: Send> Injector<T> {
    /// Construct an injector with [`DEFAULT_INJECTOR_CAPACITY`] slots.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_INJECTOR_CAPACITY)
    }

    /// Construct an injector with the requested capacity (rounded
    /// up to next power of two, minimum 2).
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self { ring: FlynnelRing::new(cap) }
    }

    /// Push an item. Spins via `spin_loop` if the ring is at
    /// capacity (back-pressure); always succeeds.
    #[inline]
    pub fn push(&self, mut item: T) {
        loop {
            match self.ring.push(item) {
                PushResult::Ok => return,
                PushResult::Full(t) => {
                    item = t;
                    core::hint::spin_loop();
                }
            }
        }
    }

    /// Try to push without spinning. Returns `Err(item)` if the
    /// ring is at capacity.
    #[inline]
    pub fn try_push(&self, item: T) -> Result<(), T> {
        match self.ring.push(item) {
            PushResult::Ok => Ok(()),
            PushResult::Full(t) => Err(t),
        }
    }

    /// Steal one item. Same three-arm semantics as the local
    /// Chase-Lev deque's steal: `Success(t)` on a successful pop,
    /// `Empty` when the ring was observed empty, `Retry` when a
    /// concurrent thief took the slot we were eyeing.
    ///
    /// FlynnelRing's pop loops internally on CAS contention, so
    /// the Retry arm is reserved for symmetry with the
    /// owner-deque protocol; callers can match it the same way.
    #[inline]
    pub fn steal(&self) -> InjectorSteal<T> {
        match self.ring.pop() {
            PopResult::Ok(t) => InjectorSteal::Success(t),
            PopResult::Empty => InjectorSteal::Empty,
        }
    }

    /// Approximate is-empty snapshot. Use as a hint only - a
    /// concurrent push or pop may invalidate the result
    /// immediately after return.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Approximate pending-item count. Use as a hint only; the
    /// snapshot may be stale by the time the caller reads it.
    #[inline]
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Capacity reported by the underlying ring (always power of
    /// two; rounded up from the constructor's request).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.ring.capacity()
    }
}

impl<T: Send> Default for Injector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send> std::fmt::Debug for Injector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Injector")
            .field("capacity", &self.capacity())
            .field("is_empty", &self.is_empty())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as O};
    use std::thread;

    #[test]
    fn push_then_steal_returns_item() {
        let inj = Injector::<u32>::with_capacity(16);
        inj.push(42);
        match inj.steal() {
            InjectorSteal::Success(v) => assert_eq!(v, 42),
            other => panic!("expected Success(42), got {other:?}"),
        }
    }

    #[test]
    fn steal_empty_returns_empty() {
        let inj = Injector::<u32>::with_capacity(8);
        assert!(matches!(inj.steal(), InjectorSteal::Empty));
    }

    #[test]
    fn try_push_returns_err_on_full() {
        let inj = Injector::<u32>::with_capacity(2);
        assert!(inj.try_push(1).is_ok());
        assert!(inj.try_push(2).is_ok());
        match inj.try_push(3) {
            Err(v) => assert_eq!(v, 3),
            Ok(()) => panic!("expected full"),
        }
    }

    #[test]
    fn mpmc_round_trip() {
        let inj = Arc::new(Injector::<u32>::with_capacity(64));
        let total = 10_000usize;
        let n_producers = 4;
        let n_consumers = 4;
        let per_producer = (total / n_producers) as u32;
        let consumed = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..n_consumers {
            let inj = Arc::clone(&inj);
            let consumed = Arc::clone(&consumed);
            handles.push(thread::spawn(move || {
                while consumed.load(O::Relaxed) < total {
                    match inj.steal() {
                        InjectorSteal::Success(_) => {
                            consumed.fetch_add(1, O::Relaxed);
                        }
                        _ => std::thread::yield_now(),
                    }
                }
            }));
        }
        for p in 0..n_producers {
            let inj = Arc::clone(&inj);
            handles.push(thread::spawn(move || {
                for i in 0..per_producer {
                    inj.push(p as u32 * per_producer + i);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker");
        }
        assert_eq!(consumed.load(O::Relaxed), total);
    }
}

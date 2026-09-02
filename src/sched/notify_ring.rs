//! Blocking notify-wrapper over the flynnel ring primitives.
//!
//! Couples a non-blocking [`FlynnelRing`] with one [`Parker`]
//! per registered consumer to give the standard channel surface
//! (`send` / `recv` / `close`) using flynnel primitives only.
//! No external channel crate; no `std::sync::Mutex` on the hot
//! path.
//!
//! ## Surface
//!
//! - [`NotifyHub::new`] - construct with `capacity` (ring slots,
//!   rounded up to pow2) and `n_consumers` (max registered
//!   consumers; sets the parker slot vector length).
//! - [`NotifyHub::sender`] - clone a producer handle.
//! - [`NotifyHub::register_consumer`] - call from the consumer
//!   thread to claim the next parker slot.
//! - [`NotifyReceiver::recv`] - blocking pop.
//! - [`NotifyHub::shutdown`] / [`NotifySender::shutdown`] -
//!   signal every consumer to exit.
//!
//! ## Hot-path design
//!
//! - **Push**: `FlynnelRing::push` (CAS-loop on slot sequence)
//!   then `wake_one`. The wake reads the next parker via a
//!   `Relaxed` load on the next-wake cursor + an `OnceLock::get`
//!   (an `AtomicPtr::load(Acquire)`). No `Mutex`, no spinlock.
//! - **Recv**: `FlynnelRing::pop` first; on `Empty` enter
//!   `Parker::park_until` with the predicate that re-checks
//!   `!ring.is_empty() || shutdown` during the spin floor.
//!
//! ## Cross-platform discipline
//!
//! All state behind `AtomicU64` / `AtomicBool` / `AtomicUsize`
//! with Acquire / Release / Relaxed ordering. Parker dispatches
//! between `std::thread::park` (universal) and WAITPKG when
//! available. No x86-specific intrinsics outside the Parker's
//! own WAITPKG path.

#![allow(clippy::missing_errors_doc)]

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use crate::sched::flynnel_ring::{FlynnelRing, PopResult, PushResult};
use crate::sched::sleep::Parker;

/// Outcome of [`NotifySender::send`].
#[derive(Debug, PartialEq, Eq)]
pub enum NotifySendResult<T> {
    /// Item enqueued + one consumer woken (round-robin).
    Ok,
    /// Hub is shut down; caller may not send any more items.
    Closed(T),
}

impl<T> NotifySendResult<T> {
    /// Returns true on a successful send.
    #[inline]
    pub fn is_ok(&self) -> bool {
        matches!(self, NotifySendResult::Ok)
    }

    /// Collapse to `Option<T>` of the rejected item.
    #[inline]
    pub fn err(self) -> Option<T> {
        match self {
            NotifySendResult::Ok => None,
            NotifySendResult::Closed(t) => Some(t),
        }
    }
}

/// Outcome of [`NotifySender::try_send`]. Distinguishes Full
/// (transient) from Closed (terminal).
#[derive(Debug, PartialEq, Eq)]
pub enum NotifyTrySendResult<T> {
    /// Item enqueued + one consumer woken.
    Ok,
    /// Ring at capacity; caller may retry later.
    Full(T),
    /// Hub is shut down; caller may not send any more items.
    Closed(T),
}

/// Shared backing for a notify hub. Holds the ring, the
/// registered consumer parkers, and the shutdown flag.
struct NotifyInner<T: Send> {
    ring: FlynnelRing<T>,
    /// Pre-allocated parker slots. Set ONCE by each consumer's
    /// `register_consumer` call via `OnceLock`. Fixed length =
    /// `n_consumers` at hub construction so the wake path can
    /// index directly with no Mutex.
    parker_slots: Box<[OnceLock<Arc<Parker>>]>,
    /// Atomic claim cursor for `register_consumer`: each call
    /// increments and uses the previous value as its slot index
    /// modulo `parker_slots.len()`.
    register_next: AtomicUsize,
    /// Round-robin wake cursor. Relaxed because the read-modify-
    /// write only needs to distribute load.
    next_wake: AtomicUsize,
    /// Shutdown latch. Producers stop sending; consumers drain
    /// the ring then exit.
    shutdown: AtomicBool,
}

/// Multi-producer multi-consumer notify hub. Construct one via
/// [`NotifyHub::new`]; senders + receivers share the inner Arc.
pub struct NotifyHub<T: Send> {
    inner: Arc<NotifyInner<T>>,
}

impl<T: Send> NotifyHub<T> {
    /// Construct a hub with `capacity` ring slots (rounded up to
    /// next power of two, minimum 2) and pre-allocated for up to
    /// `n_consumers` registered consumers.
    pub fn new(capacity: usize, n_consumers: usize) -> Self {
        let n = n_consumers.max(1);
        let slots: Vec<OnceLock<Arc<Parker>>> =
            (0..n).map(|_| OnceLock::new()).collect();
        Self {
            inner: Arc::new(NotifyInner {
                ring: FlynnelRing::new(capacity),
                parker_slots: slots.into_boxed_slice(),
                register_next: AtomicUsize::new(0),
                next_wake: AtomicUsize::new(0),
                shutdown: AtomicBool::new(false),
            }),
        }
    }

    /// Clone a producer handle. Cheap (Arc clone).
    pub fn sender(&self) -> NotifySender<T> {
        NotifySender { inner: Arc::clone(&self.inner) }
    }

    /// Register a consumer on the calling thread. Allocates a
    /// `Parker` capturing `thread::current()`, claims the next
    /// parker slot, and returns a [`NotifyReceiver`] bound to
    /// that parker.
    pub fn register_consumer(&self) -> NotifyReceiver<T> {
        const RECV_SPIN_ROUNDS: u32 = 8;
        let parker = Arc::new(Parker::new(RECV_SPIN_ROUNDS));
        let n = self.inner.parker_slots.len();
        let raw = self.inner.register_next.fetch_add(1, Ordering::Relaxed);
        let idx = raw % n;
        // Try to set this slot. If it was already set (caller
        // over-registered), the duplicate parker is used for the
        // returned NotifyReceiver but other producers wake the
        // original via the established slot. drop() the Result
        // because the failure-to-set case is benign.
        drop(self.inner.parker_slots[idx].set(Arc::clone(&parker)));
        NotifyReceiver {
            inner: Arc::clone(&self.inner),
            parker,
        }
    }

    /// Signal shutdown. All consumers wake; they drain any
    /// remaining items then their `recv` returns `None`.
    pub fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::Release);
        wake_all(&self.inner);
    }

    /// Approximate pending-item count. Hint only.
    pub fn len(&self) -> usize {
        self.inner.ring.len()
    }

    /// Approximate is-empty. Hint only.
    pub fn is_empty(&self) -> bool {
        self.inner.ring.is_empty()
    }
}

impl<T: Send> Clone for NotifyHub<T> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

/// RAII guard that calls [`NotifyHub::shutdown`] when dropped.
/// Use inside a stage closure to guarantee the hub is shut down
/// even if the stage body panics - crossbeam channels close on
/// the last Sender drop; this primitive doesn't because the hub
/// is shared via Arc, so a guard provides equivalent
/// panic-safety. Construct via [`NotifyHub::shutdown_on_drop`].
pub struct NotifyShutdownOnDrop<T: Send> {
    hub: NotifyHub<T>,
}

impl<T: Send> Drop for NotifyShutdownOnDrop<T> {
    fn drop(&mut self) {
        self.hub.inner.shutdown.store(true, Ordering::Release);
        wake_all(&self.hub.inner);
    }
}

impl<T: Send> NotifyHub<T> {
    /// Wrap this hub in a guard that calls [`Self::shutdown`] on
    /// drop. Idiomatic placement: hold the guard for the lifetime
    /// of a stage thread's closure so panic-unwind triggers
    /// shutdown automatically.
    ///
    /// ```ignore
    /// scope.spawn(move || {
    ///     let _shutdown = hub.shutdown_on_drop();
    ///     while let Some(item) = rx.recv() { stage(item); }
    /// });
    /// ```
    pub fn shutdown_on_drop(self) -> NotifyShutdownOnDrop<T> {
        NotifyShutdownOnDrop { hub: self }
    }
}

/// Producer handle. Cheaply cloneable; any thread may hold one.
pub struct NotifySender<T: Send> {
    inner: Arc<NotifyInner<T>>,
}

impl<T: Send> Clone for NotifySender<T> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<T: Send> NotifySender<T> {
    /// Push an item + wake one consumer (round-robin). Spins
    /// via `spin_loop` if the ring is at capacity (back-pressure).
    /// Returns `Closed(item)` if the hub is shut down.
    #[inline]
    pub fn send(&self, mut item: T) -> NotifySendResult<T> {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return NotifySendResult::Closed(item);
        }
        loop {
            match self.inner.ring.push(item) {
                PushResult::Ok => {
                    wake_one(&self.inner);
                    return NotifySendResult::Ok;
                }
                PushResult::Full(t) => {
                    if self.inner.shutdown.load(Ordering::Acquire) {
                        return NotifySendResult::Closed(t);
                    }
                    item = t;
                    core::hint::spin_loop();
                }
            }
        }
    }

    /// Try to send without spinning. Returns `Full(item)` if the
    /// ring is at capacity; `Closed(item)` if shut down; `Ok`
    /// otherwise.
    #[inline]
    pub fn try_send(&self, item: T) -> NotifyTrySendResult<T> {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return NotifyTrySendResult::Closed(item);
        }
        match self.inner.ring.push(item) {
            PushResult::Ok => {
                wake_one(&self.inner);
                NotifyTrySendResult::Ok
            }
            PushResult::Full(t) => NotifyTrySendResult::Full(t),
        }
    }

    /// Signal shutdown via this sender handle.
    pub fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::Release);
        wake_all(&self.inner);
    }

    /// Approximate is-empty hint.
    pub fn is_empty(&self) -> bool {
        self.inner.ring.is_empty()
    }
}

/// Single-owner consumer handle. Returned by
/// [`NotifyHub::register_consumer`]; bound to the thread that
/// registered it via the captured `Parker::thread`.
pub struct NotifyReceiver<T: Send> {
    inner: Arc<NotifyInner<T>>,
    parker: Arc<Parker>,
}

impl<T: Send> NotifyReceiver<T> {
    /// Blocking receive. Returns `Some(t)` on a successful pop;
    /// `None` when the hub is shut down AND the ring is drained.
    pub fn recv(&self) -> Option<T> {
        loop {
            match self.inner.ring.pop() {
                PopResult::Ok(t) => return Some(t),
                PopResult::Empty => {
                    if self.inner.shutdown.load(Ordering::Acquire) {
                        // One last drain attempt in case a push
                        // raced the shutdown store.
                        if let PopResult::Ok(t) = self.inner.ring.pop() {
                            return Some(t);
                        }
                        return None;
                    }
                    let inner = &self.inner;
                    let ready = self.parker.park_until(|| {
                        !inner.ring.is_empty() || inner.shutdown.load(Ordering::Acquire)
                    });
                    if !ready {
                        if let PopResult::Ok(t) = self.inner.ring.pop() {
                            return Some(t);
                        }
                        return None;
                    }
                }
            }
        }
    }

    /// Non-blocking try-receive. Returns `Some(t)` on a
    /// successful pop; `None` if the ring is empty.
    #[inline]
    pub fn try_recv(&self) -> Option<T> {
        match self.inner.ring.pop() {
            PopResult::Ok(t) => Some(t),
            PopResult::Empty => None,
        }
    }
}

/// Wake one consumer via the round-robin cursor. Walks at most
/// `n_consumers` slots to find a registered parker.
#[inline]
fn wake_one<T: Send>(inner: &Arc<NotifyInner<T>>) {
    let n = inner.parker_slots.len();
    if n == 0 {
        return;
    }
    let start = inner.next_wake.fetch_add(1, Ordering::Relaxed) % n;
    for offset in 0..n {
        let idx = (start + offset) % n;
        if let Some(p) = inner.parker_slots[idx].get() {
            p.unpark();
            return;
        }
    }
}

/// Wake every registered consumer. Used by shutdown.
#[inline]
fn wake_all<T: Send>(inner: &Arc<NotifyInner<T>>) {
    for slot in inner.parker_slots.iter() {
        if let Some(p) = slot.get() {
            p.unpark();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as O};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn single_send_single_recv_round_trip() {
        let hub = NotifyHub::<u32>::new(8, 1);
        let tx = hub.sender();
        let hub2 = hub.clone();
        let cons = thread::spawn(move || {
            let rx = hub2.register_consumer();
            rx.recv()
        });
        thread::sleep(Duration::from_millis(20));
        assert!(tx.send(42).is_ok());
        assert_eq!(cons.join().expect("consumer"), Some(42));
    }

    #[test]
    fn shutdown_returns_none_after_drain() {
        let hub = NotifyHub::<u32>::new(8, 1);
        let tx = hub.sender();
        let hub2 = hub.clone();
        let cons = thread::spawn(move || {
            let rx = hub2.register_consumer();
            let a = rx.recv();
            let b = rx.recv();
            let c = rx.recv();
            (a, b, c)
        });
        thread::sleep(Duration::from_millis(20));
        assert!(tx.send(1).is_ok());
        assert!(tx.send(2).is_ok());
        hub.shutdown();
        let (a, b, c) = cons.join().expect("consumer");
        assert_eq!(a, Some(1));
        assert_eq!(b, Some(2));
        assert_eq!(c, None, "third recv after shutdown returns None");
    }

    #[test]
    fn mpmc_round_trip_4p_4c() {
        let hub = NotifyHub::<u32>::new(64, 4);
        let total = 4000usize;
        let n_producers = 4;
        let n_consumers = 4;
        let per_producer = (total / n_producers) as u32;

        let consumed = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));

        let mut cons_handles = Vec::new();
        for _ in 0..n_consumers {
            let hub2 = hub.clone();
            let consumed = Arc::clone(&consumed);
            let sum = Arc::clone(&sum);
            cons_handles.push(thread::spawn(move || {
                let rx = hub2.register_consumer();
                while let Some(v) = rx.recv() {
                    consumed.fetch_add(1, O::Relaxed);
                    sum.fetch_add(v as usize, O::Relaxed);
                }
            }));
        }
        thread::sleep(Duration::from_millis(20));

        let mut prod_handles = Vec::new();
        for p in 0..n_producers {
            let tx = hub.sender();
            prod_handles.push(thread::spawn(move || {
                for i in 0..per_producer {
                    let v = (p as u32) * per_producer + i;
                    while !tx.send(v).is_ok() {}
                }
            }));
        }
        for h in prod_handles {
            h.join().expect("p");
        }
        for _ in 0..200 {
            if consumed.load(O::Relaxed) >= total {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        hub.shutdown();
        for h in cons_handles {
            h.join().expect("c");
        }
        let expected: usize = (0..total).map(|i| i as u32 as usize).sum();
        assert_eq!(consumed.load(O::Relaxed), total);
        assert_eq!(sum.load(O::Relaxed), expected,
            "sum invariant: every pushed value consumed exactly once");
    }
}

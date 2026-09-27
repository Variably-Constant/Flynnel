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
//!   then `wake_one`. With one consumer the wake unparks it
//!   whatever it is doing, through an `OnceLock::get` (an
//!   `AtomicPtr::load(Acquire)`). With several, the wake goes only
//!   to a consumer that has announced it is about to park: a
//!   `SeqCst` fence, one load of the announced count, and when that
//!   is not zero a scan from the round-robin cursor for a raised
//!   flag this producer can lower. No `Mutex`, no spinlock.
//! - **Recv**: `FlynnelRing::pop` first; on `Empty`, a consumer of
//!   a hub with several raises its flag, fences, and enters
//!   `Parker::park_until` with the predicate that re-checks
//!   `!ring.is_empty() || shutdown` during the spin floor. The flag
//!   comes down when the park returns, unless a producer lowered it
//!   first to wake this consumer.
//!
//! ## Why a wake goes only to an announced consumer
//!
//! Any consumer takes any item, so one in its spin floor, or one
//! just registered, can pop an item whose wake went to another.
//! That other consumer finds the ring empty and parks again. A wake
//! addressed by position alone can then land on the consumer that
//! took the item and is busy running it, while an item waits in the
//! ring and the other consumer sleeps with nothing to wake it: on a
//! pool of long-running tasks, a worker short for as long as they
//! run. A wake that claims a raised flag reaches a consumer that is
//! parked or about to park, which then pops. The announcement and
//! the push are each followed by a `SeqCst` fence before the other
//! side is read, so either the producer sees the flag or the
//! consumer's last look at the ring sees the item.
//!
//! ## Cross-platform discipline
//!
//! All state behind `AtomicU64` / `AtomicBool` / `AtomicUsize`
//! with Acquire / Release / Relaxed ordering. Parker dispatches
//! between `std::thread::park` (universal) and WAITPKG when
//! available. No x86-specific intrinsics outside the Parker's
//! own WAITPKG path.

#![allow(clippy::missing_errors_doc)]

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};
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
    /// One flag per parker slot, raised by that slot's consumer once
    /// it has found the ring empty and is about to park, and lowered
    /// by whichever comes first of a producer claiming it for a wake
    /// and the consumer's park returning. Only a hub with more than
    /// one slot uses them.
    idle: Box<[AtomicBool]>,
    /// How many `idle` flags are raised. A send that reads zero after
    /// its fence has no consumer to wake and skips the scan.
    idle_count: AtomicUsize,
    /// Atomic claim cursor for `register_consumer`: each call
    /// increments and uses the previous value as its slot index
    /// modulo `parker_slots.len()`.
    register_next: AtomicUsize,
    /// Where a send starts its scan for a raised flag. Relaxed
    /// because the read-modify-write only needs to spread the wakes.
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
        let idle: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
        Self {
            inner: Arc::new(NotifyInner {
                ring: FlynnelRing::new(capacity),
                parker_slots: slots.into_boxed_slice(),
                idle: idle.into_boxed_slice(),
                idle_count: AtomicUsize::new(0),
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
        // A slot keeps the parker set first. A caller that registers
        // more consumers than the hub has slots gets a receiver whose
        // parker no producer wakes, so that receiver holds no slot and
        // never announces itself.
        let slot = self.inner.parker_slots[idx]
            .set(Arc::clone(&parker))
            .is_ok()
            .then_some(idx);
        NotifyReceiver {
            inner: Arc::clone(&self.inner),
            parker,
            slot,
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
    /// ```
    /// use flynnel::sched::notify_ring::NotifyHub;
    ///
    /// let hub = NotifyHub::<u32>::new(4, 1);
    /// let rx = hub.register_consumer();
    ///
    /// // The stage that owns the hub hands it to the guard, so the
    /// // shutdown fires however that stage ends.
    /// std::thread::scope(|scope| {
    ///     scope.spawn(move || {
    ///         let _shutdown = hub.shutdown_on_drop();
    ///     });
    /// });
    ///
    /// assert!(rx.recv().is_none(), "a shut-down hub releases its consumers");
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
    /// The parker slot this receiver's parker holds, which is the
    /// flag it raises before parking; `None` for a receiver
    /// registered past the hub's slot count.
    slot: Option<usize>,
}

impl<T: Send> NotifyReceiver<T> {
    /// Blocking receive. Returns `Some(t)` on a successful pop;
    /// `None` when the hub is shut down and the ring is drained.
    pub fn recv(&self) -> Option<T> {
        // A hub with several consumers wakes only one that has
        // announced itself, so this receiver raises its flag before
        // it parks. A hub with one wakes its consumer on every send.
        let announced = self.slot.filter(|_| self.inner.idle.len() > 1);
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
                    if let Some(slot) = announced {
                        inner.announce_idle(slot);
                    }
                    let ready = self.parker.park_until(|| {
                        !inner.ring.is_empty() || inner.shutdown.load(Ordering::Acquire)
                    });
                    if let Some(slot) = announced {
                        inner.withdraw_idle(slot);
                    }
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

impl<T: Send> NotifyInner<T> {
    /// Count `slot` as idle and raise its flag, then fence, so either
    /// a producer that pushes after this sees the flag or the caller's
    /// next look at the ring sees the push. The count goes up before
    /// the flag, which is a release store, so a producer that lowers
    /// the flag lowers the count after this raised it.
    #[inline]
    fn announce_idle(&self, slot: usize) {
        self.idle_count.fetch_add(1, Ordering::Relaxed);
        self.idle[slot].store(true, Ordering::Release);
        fence(Ordering::SeqCst);
    }

    /// Lower `slot`'s flag unless a producer already has.
    #[inline]
    fn withdraw_idle(&self, slot: usize) {
        if self.idle[slot].swap(false, Ordering::AcqRel) {
            self.idle_count.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Wake one consumer. A hub with one slot unparks that slot's
/// consumer whatever it is doing, which its park's permit turns into
/// a prompt return. A hub with several wakes only a consumer whose
/// raised flag this call lowers, scanning from the round-robin cursor,
/// and wakes none when no flag is raised: every consumer is then busy
/// and pops when it returns, or has yet to announce and looks at the
/// ring after it does.
#[inline]
fn wake_one<T: Send>(inner: &Arc<NotifyInner<T>>) {
    let n = inner.parker_slots.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        if let Some(p) = inner.parker_slots[0].get() {
            p.unpark();
        }
        return;
    }
    // Pairs with the fence in `announce_idle`.
    fence(Ordering::SeqCst);
    if inner.idle_count.load(Ordering::Relaxed) == 0 {
        return;
    }
    let start = inner.next_wake.fetch_add(1, Ordering::Relaxed) % n;
    for offset in 0..n {
        let idx = (start + offset) % n;
        let flag = &inner.idle[idx];
        if flag.load(Ordering::Relaxed) && flag.swap(false, Ordering::AcqRel) {
            inner.idle_count.fetch_sub(1, Ordering::Relaxed);
            if let Some(p) = inner.parker_slots[idx].get() {
                p.unpark();
            }
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

    /// A consumer pops an item no wake was addressed to, as one in its
    /// spin floor does, and then stays busy, while the consumer that wake
    /// went to finds the ring empty and parks again. The next send's wake
    /// goes to the busy consumer's slot, and the item it carries has to
    /// reach the parked consumer anyway. The busy consumer stands in for a
    /// spinning one by polling `try_recv`; an attempt in which the woken
    /// consumer wins the first item instead is repeated:
    /// `cargo test --profile release-test --lib sched::notify_ring::tests::a_wake_on_a_busy_consumer_still_reaches_a_parked_one -- --ignored --nocapture`
    #[test]
    #[ignore = "stages a stranded item; run by hand"]
    fn a_wake_on_a_busy_consumer_still_reaches_a_parked_one() {
        use std::sync::atomic::AtomicBool;
        use std::time::Instant;
        const ATTEMPTS: usize = 20;
        let bound = Duration::from_secs(2);
        for attempt in 1..=ATTEMPTS {
            let hub = NotifyHub::<u32>::new(8, 2);
            let tx = hub.sender();
            let a_got = Arc::new(AtomicUsize::new(0));
            let b_got = Arc::new(AtomicUsize::new(0));
            let registered = Arc::new(AtomicUsize::new(0));
            let release = Arc::new(AtomicBool::new(false));
            // A claims slot 0 before B starts, so the first wake is A's.
            let a = {
                let hub = hub.clone();
                let (got, registered) = (Arc::clone(&a_got), Arc::clone(&registered));
                thread::spawn(move || {
                    let rx = hub.register_consumer();
                    registered.fetch_add(1, O::SeqCst);
                    if let Some(v) = rx.recv() {
                        got.store(v as usize, O::SeqCst);
                    }
                })
            };
            while registered.load(O::SeqCst) < 1 {
                thread::yield_now();
            }
            let b = {
                let hub = hub.clone();
                let (got, registered, release) = (
                    Arc::clone(&b_got),
                    Arc::clone(&registered),
                    Arc::clone(&release),
                );
                thread::spawn(move || {
                    let rx = hub.register_consumer();
                    registered.fetch_add(1, O::SeqCst);
                    loop {
                        if let Some(v) = rx.try_recv() {
                            got.store(v as usize, O::SeqCst);
                            break;
                        }
                        if release.load(O::SeqCst) {
                            return;
                        }
                        std::hint::spin_loop();
                    }
                    while !release.load(O::SeqCst) {
                        thread::sleep(Duration::from_millis(1));
                    }
                })
            };
            while registered.load(O::SeqCst) < 2 {
                thread::yield_now();
            }
            thread::sleep(Duration::from_millis(100));
            assert!(tx.send(1).is_ok());
            let start = Instant::now();
            while a_got.load(O::SeqCst) == 0
                && b_got.load(O::SeqCst) == 0
                && start.elapsed() < bound
            {
                thread::yield_now();
            }
            let staged = b_got.load(O::SeqCst) == 1 && a_got.load(O::SeqCst) == 0;
            if staged {
                // A woke to an empty ring; this gives it time to park again.
                thread::sleep(Duration::from_millis(100));
                assert!(tx.send(2).is_ok());
                let start = Instant::now();
                while a_got.load(O::SeqCst) == 0 && start.elapsed() < bound {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            let reached = a_got.load(O::SeqCst) == 2;
            release.store(true, O::SeqCst);
            hub.shutdown();
            a.join().expect("consumer a");
            b.join().expect("consumer b");
            if staged {
                println!(
                    "STRAND hub attempt={attempt} second_item_reached_parked_consumer={reached} bound_ms={}",
                    bound.as_millis()
                );
                assert!(
                    reached,
                    "the second item stayed queued while consumer A slept, its wake spent on busy consumer B"
                );
                return;
            }
        }
        panic!(
            "in {ATTEMPTS} attempts the woken consumer always won the first item, so the interleaving was never staged"
        );
    }
}

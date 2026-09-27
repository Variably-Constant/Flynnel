//! Flynnel's in-process rings, instantiated over a byte payload.
//!
//! Six ring shapes live in the crate and every one of them is generic
//! over `T: Send`. A script has no Rust type to offer, so each is bound
//! here at one concrete instantiation: an item is a `byte[]`. That is
//! the same answer the sibling SubEtha module reached, and it is what
//! lets a staged pipeline put arbitrary payloads between stages without
//! the ring knowing what they are.
//!
//! # A ring refuses; it does not park
//!
//! Every push here returns an outcome rather than waiting. A full ring
//! hands the item back and the caller decides whether to retry, back
//! off, or drop it. This is the crate's contract and it is the thing to
//! know before designing a staged pipeline on top: there is no
//! back-pressure that blocks a producer, so a slow stage does not slow
//! its upstream by itself.
//!
//! The crate does carry blocking forms - `FlynnelRing::push_blocking`,
//! `FlynnelRing::pop_blocking`, `Injector::push`, `NotifySender::send`
//! and `NotifyReceiver::recv`. None of them is bound, and the reason is
//! not taste. Each waits inside Rust with no way to observe the
//! pipeline's stopping flag, so a script that called one on an empty or
//! full ring would hang a PowerShell host that cannot be interrupted:
//! Ctrl-C sets a flag the blocked thread never reads. `push_blocking`
//! on a ring nobody is draining spins forever by construction. The
//! bound surface is the try-forms, and a script that wants to wait
//! writes its own loop, where its own `Start-Sleep` and Ctrl-C work.
//!
//! # Many items in one call
//!
//! One Push or Pop per item costs 4848 ns on PowerShell 7.6 and 2013 ns
//! on Windows PowerShell, where 256 items passed in one call cost 39 ns
//! and 22 ns each. So `PushMany` and `PopMany` are the primary forms
//! and the single-item ones exist for the case where a script genuinely
//! has one item.
//!
//! Those four figures were measured in the sibling SubEtha module's
//! `bench/CallShapes.ps1`, over the rings that module binds, and they
//! are quoted here because the boundary is the same one and the shapes
//! are the same shapes. They are not a measurement of these rings, and
//! nothing has yet measured these. `bench/KernelShapes.ps1` is where
//! that goes.
//!
//! # Where the Rust handle lives
//!
//! Every object here carries an `Id`; the Rust handle sits in one
//! process-wide table keyed by that id, and the object holds a guard
//! that removes the entry when the object is disposed or collected.
//!
//! The table is there because the SPSC, composed and grid handles are
//! single-owner by construction: they carry a plain `Cell` cursor and
//! no synchronization, because the crate expects exactly one thread on
//! each. They are therefore not `Clone`, so there is no second copy to
//! give a cmdlet - and a cmdlet cannot take one of these objects as a
//! typed parameter either, because a class holding Rust state is not
//! reconstructible by value and its `FromPs` says so rather than
//! guessing. One table holding one handle behind one lock answers both:
//! `Send-FlynnelItem` reads `Id` off whatever object it was handed and
//! finds the handle, and the lock keeps the single-owner discipline
//! true when a script drives the same object from two runspaces.
//!
//! The lock costs an uncontended acquire on a path whose cheapest
//! crossing is 1907 ns, so it is under a tenth of a percent of a
//! method call, and a batch pays it once rather than once an item.
//! Held for the batch, though: a `PushMany` of ten thousand items
//! keeps every other runspace's ring call waiting for the whole of it.
//! That is the same serialization the single-owner shapes need anyway,
//! and a script wanting two rings driven at once gives each its own
//! runspace and batches in the thousands rather than the millions.
//!
//! # Disposing
//!
//! Every class here inherits `Dispose` from the generated proxy and
//! none declares one, because a declared `Dispose` would hide the
//! inherited member and a script's `using` block would then free
//! nothing. Disposing drops the object, which drops its guard, which
//! removes the table entry and frees the ring - so the ordinary
//! PowerShell idiom is the one that works, and a handle a script
//! forgets is freed when the garbage collector reaches it.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use pwrs::prelude::*;

use flynnel::sched::flynnel_ring::{FlynnelRing, PopResult, PushResult};
use flynnel::sched::flynnel_ring_composed::{
    ComposedMpscConsumer, ComposedPopResult, GridConsumer, GridPopResult, GridProducer,
    GridPushResult, new_composed_mpmc, new_composed_mpsc,
};
use flynnel::sched::flynnel_ring_mpsc::{
    MpscConsumer, MpscPopResult, MpscProducer, MpscPushResult, new_mpsc,
};
use flynnel::sched::flynnel_ring_spsc::{
    Consumer as SpscConsumerInner, Producer as SpscProducerInner, SpscPopResult,
    SpscPushResult, new_spsc,
};
use flynnel::sched::injector::{
    DEFAULT_INJECTOR_CAPACITY, Injector as InjectorInner, InjectorSteal,
};
use flynnel::sched::notify_ring::{
    NotifyHub, NotifyReceiver as NotifyReceiverInner, NotifySender as NotifySenderInner,
    NotifyTrySendResult,
};

/// What every ring in this module carries.
///
/// Named Payload rather than Item because `pwrs::prelude` exports an
/// `Item` of its own, for the provider surface.
type Payload = Vec<u8>;

// ---------------------------------------------------------------------
// Outcomes
// ---------------------------------------------------------------------

/// Why a push ended as it did.
#[psenum(name = "Flynnel.PushKind")]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum PushKind {
    /// The ring took the item.
    #[default]
    Ok,
    /// The ring is at capacity and gave the item back. Transient: the
    /// same push may succeed once a consumer has drained something.
    Full,
    /// The hub has been shut down and gave the item back. Terminal: no
    /// later push on this handle can succeed. Only a notify sender
    /// answers this; the other shapes have nothing to close.
    Closed,
}

/// Why a pop ended as it did.
#[psenum(name = "Flynnel.PopKind")]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum PopKind {
    /// An item came out.
    #[default]
    Ok,
    /// Nothing was there. Transient.
    Empty,
    /// Contention refused the read and the caller may try again. The
    /// crate's own rings never answer this: their pop loops on
    /// contention internally. It exists because the Chase-Lev steal
    /// protocol the injector mirrors carries the arm, so a script
    /// written against the full three-way shape stays correct if a ring
    /// ever starts using it.
    Retry,
    /// The pop side is shut down and drained.
    ///
    /// No pop in this module answers this, and the reason is worth
    /// knowing before writing a poll loop against it. A notify hub's
    /// shutdown flag is private and has no reader; the one call that
    /// would distinguish a closed hub from an empty one is a send,
    /// which on a live hub enqueues the probe and delivers a phantom
    /// item to whichever consumer reaches it. So an exhausted hub reads
    /// as Empty here. A script learns of the shutdown from its own
    /// Shutdown call, or from a send answering Closed.
    Closed,
}

/// Which side of a ring an object is.
#[psenum(name = "Flynnel.RingRole")]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum RingRole {
    /// One object driving both ends.
    #[default]
    Ring,
    /// A push side.
    Producer,
    /// A pop side.
    Consumer,
}

/// The outcome of a push, with the item back when the ring refused it.
///
/// The item is returned rather than dropped, so a full ring costs a
/// retry and never costs data.
#[psclass(name = "Flynnel.PushOutcome")]
#[derive(Clone, Default)]
pub struct PushOutcome {
    /// Ok, Full or Closed.
    pub kind: PushKind,
    /// True only for Ok, so a script can branch without comparing an
    /// enum.
    pub accepted: bool,
    /// The item the ring refused, empty when it accepted.
    pub item: Vec<u8>,
}

impl PushOutcome {
    fn ok() -> Self {
        Self { kind: PushKind::Ok, accepted: true, item: Vec::new() }
    }

    fn refused(kind: PushKind, item: Payload) -> Self {
        Self { kind, accepted: false, item }
    }
}

/// The outcome of a pop.
///
/// A kind rather than a null, because an empty ring and an item that
/// happens to be zero bytes long are different states and `$null`
/// reports both the same way.
#[psclass(name = "Flynnel.PopOutcome")]
#[derive(Clone, Default)]
pub struct PopOutcome {
    /// Ok, Empty or Retry.
    pub kind: PopKind,
    /// True only for Ok.
    pub got_item: bool,
    /// The item, empty on every kind but Ok - and also empty on Ok when
    /// the item itself was zero bytes, which is why GotItem exists.
    pub item: Vec<u8>,
}

impl PopOutcome {
    fn got(item: Payload) -> Self {
        Self { kind: PopKind::Ok, got_item: true, item }
    }

    fn none(kind: PopKind) -> Self {
        Self { kind, got_item: false, item: Vec::new() }
    }
}

/// What a ring holds and what it has refused.
#[psclass(name = "Flynnel.RingStat")]
#[derive(Clone, Default)]
pub struct RingStat {
    /// The handle this row describes.
    pub id: u64,
    /// Which side it is.
    pub role: RingRole,
    /// Whether Depth, IsEmpty and IsFull mean anything for this shape.
    /// The SPSC, MPSC, composed and grid handles expose no reader for
    /// the ring behind them, so those columns are zero on them and a
    /// zero depth would otherwise read as an empty ring.
    pub depth_known: bool,
    /// Items in the ring now, when DepthKnown. A hint: a concurrent
    /// push or pop can invalidate it before the caller reads it.
    pub depth: u64,
    /// Slots. Rounded up to a power of two by the ring, so this is
    /// frequently larger than the constructor's request; on a shape
    /// with no reader it is the request itself.
    pub capacity: u64,
    /// Whether the ring read as empty, when DepthKnown.
    pub is_empty: bool,
    /// Whether it read as full, when DepthKnown.
    pub is_full: bool,
    /// Items this handle has pushed successfully.
    pub pushed: u64,
    /// Pushes this handle made that the ring refused. Against Pushed,
    /// this is the back-pressure the ring is applying.
    pub full_refusals: u64,
    /// Items this handle has popped.
    pub popped: u64,
    /// Pops that found nothing. Against Popped, this is how much of a
    /// consumer's time goes on asking rather than working.
    pub empty_pops: u64,
}

// ---------------------------------------------------------------------
// The handle table
// ---------------------------------------------------------------------

/// One live ring handle. Held only by the table.
enum Handle {
    Ring(FlynnelRing<Payload>),
    SpscProducer(SpscProducerInner<Payload>),
    SpscConsumer(SpscConsumerInner<Payload>),
    MpscProducer(MpscProducer<Payload>),
    MpscConsumer(MpscConsumer<Payload>),
    ComposedConsumer(ComposedMpscConsumer<Payload>),
    GridProducer(GridProducer<Payload>),
    GridConsumer(GridConsumer<Payload>),
    Injector(InjectorInner<Payload>),
    NotifySender(NotifySenderInner<Payload>, NotifyHub<Payload>),
    NotifyReceiver(NotifyReceiverInner<Payload>, NotifyHub<Payload>),
}

impl Handle {
    /// Pushes, or hands the item straight back when this handle has no
    /// push side. The item comes back rather than being dropped so the
    /// caller can say how much did not go in.
    fn push(&self, item: Payload) -> Result<PushOutcome, Payload> {
        Ok(match self {
            Handle::Ring(r) => match r.push(item) {
                PushResult::Ok => PushOutcome::ok(),
                PushResult::Full(t) => PushOutcome::refused(PushKind::Full, t),
            },
            Handle::SpscProducer(p) => match p.push(item) {
                SpscPushResult::Ok => PushOutcome::ok(),
                SpscPushResult::Full(t) => PushOutcome::refused(PushKind::Full, t),
            },
            Handle::MpscProducer(p) => match p.push(item) {
                MpscPushResult::Ok => PushOutcome::ok(),
                MpscPushResult::Full(t) => PushOutcome::refused(PushKind::Full, t),
            },
            Handle::GridProducer(p) => match p.push(item) {
                GridPushResult::Ok => PushOutcome::ok(),
                GridPushResult::Full(t) => PushOutcome::refused(PushKind::Full, t),
            },
            Handle::Injector(q) => match q.try_push(item) {
                Ok(()) => PushOutcome::ok(),
                Err(t) => PushOutcome::refused(PushKind::Full, t),
            },
            Handle::NotifySender(s, _) => match s.try_send(item) {
                NotifyTrySendResult::Ok => PushOutcome::ok(),
                NotifyTrySendResult::Full(t) => PushOutcome::refused(PushKind::Full, t),
                NotifyTrySendResult::Closed(t) => PushOutcome::refused(PushKind::Closed, t),
            },
            Handle::SpscConsumer(_)
            | Handle::MpscConsumer(_)
            | Handle::ComposedConsumer(_)
            | Handle::GridConsumer(_)
            | Handle::NotifyReceiver(..) => return Err(item),
        })
    }

    /// Pops, or answers None when this handle has no pop side.
    fn pop(&self) -> Option<PopOutcome> {
        Some(match self {
            Handle::Ring(r) => match r.pop() {
                PopResult::Ok(t) => PopOutcome::got(t),
                PopResult::Empty => PopOutcome::none(PopKind::Empty),
            },
            Handle::SpscConsumer(c) => match c.pop() {
                SpscPopResult::Ok(t) => PopOutcome::got(t),
                SpscPopResult::Empty => PopOutcome::none(PopKind::Empty),
            },
            Handle::MpscConsumer(c) => match c.pop() {
                MpscPopResult::Ok(t) => PopOutcome::got(t),
                MpscPopResult::Empty => PopOutcome::none(PopKind::Empty),
            },
            Handle::ComposedConsumer(c) => match c.pop() {
                ComposedPopResult::Ok(t) => PopOutcome::got(t),
                ComposedPopResult::Empty => PopOutcome::none(PopKind::Empty),
            },
            Handle::GridConsumer(c) => match c.pop() {
                GridPopResult::Ok(t) => PopOutcome::got(t),
                GridPopResult::Empty => PopOutcome::none(PopKind::Empty),
            },
            Handle::Injector(q) => match q.steal() {
                InjectorSteal::Success(t) => PopOutcome::got(t),
                InjectorSteal::Empty => PopOutcome::none(PopKind::Empty),
                InjectorSteal::Retry => PopOutcome::none(PopKind::Retry),
            },
            // A shut-down hub also reads Empty here; PopKind::Closed
            // says why nothing can tell the two apart from outside.
            Handle::NotifyReceiver(r, _) => match r.try_recv() {
                Some(t) => PopOutcome::got(t),
                None => PopOutcome::none(PopKind::Empty),
            },
            Handle::SpscProducer(_)
            | Handle::MpscProducer(_)
            | Handle::GridProducer(_)
            | Handle::NotifySender(..) => return None,
        })
    }

    /// The depth, and the capacity where the shape has a reader for it.
    ///
    /// Only three of the eleven shapes expose either. The rest answer
    /// `None`, which is what DepthKnown reports, because a zero from a
    /// shape that cannot count reads exactly like an empty ring. A
    /// capacity of zero means the shape counts depth and not slots, and
    /// the caller's own request stands in.
    fn depth_and_capacity(&self) -> Option<(u64, u64)> {
        match self {
            Handle::Ring(r) => Some((r.len() as u64, r.capacity() as u64)),
            Handle::Injector(q) => Some((q.len() as u64, q.capacity() as u64)),
            Handle::NotifySender(_, hub) | Handle::NotifyReceiver(_, hub) => {
                Some((hub.len() as u64, 0))
            }
            _ => None,
        }
    }
}

/// One entry: the handle and the counters for the object that owns it.
struct Entry {
    id: u64,
    handle: Handle,
    pushed: u64,
    full_refusals: u64,
    popped: u64,
    empty_pops: u64,
}

// SAFETY of the shape rather than of a cast: the Cell-carrying handles
// - GridProducer, GridConsumer and ComposedMpscConsumer - are `Send`
// and not `Sync`, which is the single-owner discipline the SPSC rings
// behind them need. The table holds each exactly once behind this
// mutex, so one thread touches a handle at a time and the discipline
// holds even when a script drives the object from two runspaces.
static HANDLES: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// Recovers a poisoned table rather than failing every later call.
///
/// A panic while the lock was held leaves the vector intact - nothing
/// here keeps an invariant across two steps - so the contents are sound
/// and refusing them would strand every ring in the session over one
/// unrelated failure.
fn handles() -> std::sync::MutexGuard<'static, Vec<Entry>> {
    match HANDLES.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Strictly increasing handle ids, so `Id` names one object for a
/// script's whole session and a stat row matches the handle that made
/// it.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn register(handle: Handle) -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    handles().push(Entry { id, handle, pushed: 0, full_refusals: 0, popped: 0, empty_pops: 0 });
    id
}

/// Removes a handle's table entry when the object holding it goes.
///
/// One of these sits in every ring object as a Rust-only field, which
/// does two jobs at once. It ties the handle's life to the object's, so
/// the inherited `Dispose` and the garbage collector both free the ring
/// through the ordinary path. And, being Rust-only, it makes the class
/// unreadable back by value, so no cmdlet can take one of these objects
/// as a typed parameter and drop a value copy that would take the live
/// handle's entry with it.
struct HandleGuard(u64);

impl Drop for HandleGuard {
    fn drop(&mut self) {
        let mut table = handles();
        if let Some(at) = table.iter().position(|e| e.id == self.0) {
            table.remove(at);
        }
    }
}

/// The error for a handle the table does not hold.
fn unknown_handle(id: u64) -> PsError {
    PsError::new(
        ErrorCategory::ObjectNotFound,
        "FlynnelUnknownRing",
        format!(
            "ring handle {id} is not one this module holds: it has been disposed, or the \
             object came from another process's copy of the module"
        ),
    )
}

/// The error for a push aimed at a pop-side handle.
///
/// The item's size is in the message rather than in a property, because
/// a wrong-side push is a script bug and not a transient refusal: no
/// retry makes that handle take it, so there is nothing for the caller
/// to do with the bytes except know how many did not go in.
fn wrong_side_push(id: u64, item: Payload) -> PsError {
    PsError::new(
        ErrorCategory::InvalidOperation,
        "FlynnelWrongSide",
        format!(
            "ring handle {id} is a pop side and has nothing to push into, so the {} byte \
             item was not enqueued",
            item.len()
        ),
    )
}

/// The error for a pop aimed at a push-side handle.
fn wrong_side_pop(id: u64) -> PsError {
    PsError::new(
        ErrorCategory::InvalidOperation,
        "FlynnelWrongSide",
        format!("ring handle {id} is a push side and has nothing to pop from"),
    )
}

/// The error for an argument a ring cannot take.
///
/// Terminating, as the module's other argument refusals are: a script
/// that passed a capacity, a count or a ring object this cannot take
/// holds nothing to carry on with, and a non-terminating error would let
/// it carry on holding nothing. A push refused at the wrong side is the
/// per-item error, and stays non-terminating.
fn arg_err(message: impl Into<String>) -> PsError {
    PsError::new(ErrorCategory::InvalidArgument, "FlynnelArgument", message.into()).terminating()
}

/// Runs `f` against one entry under one acquisition of the table.
fn with_entry<T>(id: u64, f: impl FnOnce(&mut Entry) -> PsResult<T>) -> PsResult<T> {
    let mut table = handles();
    let entry = table.iter_mut().find(|e| e.id == id).ok_or_else(|| unknown_handle(id))?;
    f(entry)
}

/// Counts one push outcome into the entry that produced it.
fn count_push(entry: &mut Entry, outcome: &PushOutcome) {
    if outcome.accepted {
        entry.pushed += 1;
    } else {
        entry.full_refusals += 1;
    }
}

/// Pushes one item and counts the outcome, in one acquisition.
fn push_one(id: u64, item: Payload) -> PsResult<PushOutcome> {
    with_entry(id, |entry| {
        let outcome = match entry.handle.push(item) {
            Ok(outcome) => outcome,
            Err(returned) => return Err(wrong_side_push(id, returned)),
        };
        count_push(entry, &outcome);
        Ok(outcome)
    })
}

/// Pushes a whole batch under one acquisition, stopping at the first
/// refusal.
///
/// It stops rather than skipping, because a ring is ordered and
/// carrying on past a refusal would deliver later items ahead of an
/// earlier one. The outcomes come back one per item attempted, so their
/// count says how far it got and the last one carries the refused item.
fn push_batch(id: u64, items: Vec<Payload>) -> PsResult<Vec<PushOutcome>> {
    with_entry(id, |entry| {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let outcome = match entry.handle.push(item) {
                Ok(outcome) => outcome,
                Err(returned) => return Err(wrong_side_push(id, returned)),
            };
            count_push(entry, &outcome);
            let refused = !outcome.accepted;
            out.push(outcome);
            if refused {
                break;
            }
        }
        Ok(out)
    })
}

/// Pops one item and counts the outcome.
fn pop_one(id: u64) -> PsResult<PopOutcome> {
    with_entry(id, |entry| {
        let outcome = entry.handle.pop().ok_or_else(|| wrong_side_pop(id))?;
        if outcome.got_item {
            entry.popped += 1;
        } else {
            entry.empty_pops += 1;
        }
        Ok(outcome)
    })
}

/// Pops up to `count` items under one acquisition, stopping when the
/// ring stops answering.
fn pop_batch(id: u64, count: i64) -> PsResult<Vec<Vec<u8>>> {
    let n = checked_count(count)?;
    with_entry(id, |entry| {
        // Bounded, because Count may be a script's ceiling rather than
        // its expectation and reserving it outright would allocate for
        // items the ring does not hold.
        let mut out = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            let outcome = entry.handle.pop().ok_or_else(|| wrong_side_pop(id))?;
            if outcome.got_item {
                entry.popped += 1;
                out.push(outcome.item);
            } else {
                entry.empty_pops += 1;
                break;
            }
        }
        Ok(out)
    })
}

/// The stat row for one handle.
fn stat_row(id: u64, role: RingRole, requested_capacity: u64) -> PsResult<RingStat> {
    with_entry(id, |entry| {
        let mut row = RingStat {
            id,
            role,
            capacity: requested_capacity,
            pushed: entry.pushed,
            full_refusals: entry.full_refusals,
            popped: entry.popped,
            empty_pops: entry.empty_pops,
            ..RingStat::default()
        };
        if let Some((depth, capacity)) = entry.handle.depth_and_capacity() {
            row.depth_known = true;
            row.depth = depth;
            if capacity > 0 {
                row.capacity = capacity;
            }
            row.is_empty = depth == 0;
            row.is_full = row.capacity > 0 && depth >= row.capacity;
        }
        Ok(row)
    })
}

/// Refuses a count a pop cannot honor.
///
/// Zero is refused rather than answered with an empty array, because a
/// script asking for zero items has a bug in the expression that
/// computed the count and an empty result hides it.
fn checked_count(count: i64) -> PsResult<usize> {
    if count <= 0 {
        return Err(arg_err(format!(
            "Count must be at least one; {count} asks the ring for nothing"
        )));
    }
    Ok(count as usize)
}

/// Refuses a capacity the ring cannot take.
///
/// The crate rounds up to a power of two and clamps to a floor of two,
/// so a request of one is silently a two. Saying so here is cheaper
/// than a script wondering why Capacity disagrees with the request.
fn checked_capacity(capacity: i64) -> PsResult<usize> {
    if capacity < 1 {
        return Err(arg_err(format!(
            "Capacity must be at least one; {capacity} describes no ring"
        )));
    }
    Ok(capacity as usize)
}

/// Refuses a handle count of zero.
///
/// The crate clamps it to one with `max(1)`, which turns a script's
/// arithmetic mistake into a working ring carrying one producer where
/// the script expected none.
fn checked_handles(n: i64, what: &str) -> PsResult<usize> {
    if n < 1 {
        return Err(arg_err(format!("{what} must be at least one; {n} was asked for")));
    }
    Ok(n as usize)
}

// ---------------------------------------------------------------------
// The general MPMC ring
// ---------------------------------------------------------------------

/// A multi-producer multi-consumer ring, both ends on one object.
///
/// The most general shape and the most expensive: every push and pop is
/// a compare-and-swap on a shared sequence, which the SPSC shape avoids
/// entirely. Reach for it when the producer and consumer counts are
/// unknown or change; reach for New-FlynnelSpscRing when they are one
/// and one.
#[psclass(name = "Flynnel.Ring", mode = proxy)]
pub struct Ring {
    /// Names this handle for its session.
    pub id: u64,
    /// Ring, because one object drives both ends.
    pub role: RingRole,
    /// Slots, rounded up to a power of two.
    pub capacity: u64,
    // Read by nothing. Its Drop is the whole of its job, and that is
    // what removes the table entry when the object is disposed or
    // collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: HandleGuard,
}

/// The operations of a `Flynnel.Ring`.
#[psmethods]
impl Ring {
    /// A ring of at least Capacity slots, rounded up to a power of two.
    ///
    /// A script reaches this as `[Flynnel.Ring]::new(1024)`.
    /// New-FlynnelRing builds its ring here as well, so the two routes
    /// make the same object.
    pub fn new(capacity: i64) -> PsResult<Ring> {
        let cap = checked_capacity(capacity)?;
        let inner = FlynnelRing::<Payload>::new(cap);
        let capacity = inner.capacity() as u64;
        let id = register(Handle::Ring(inner));
        Ok(Ring { id, role: RingRole::Ring, capacity, guard: HandleGuard(id) })
    }

    /// Pushes one item, answering whether the ring took it.
    ///
    /// Costs a boundary crossing per item. Use PushMany for more than a
    /// handful.
    pub fn push(&self, item: Vec<u8>) -> PsResult<PushOutcome> {
        push_one(self.id, item)
    }

    /// Pushes every item in one crossing, stopping at the first the
    /// ring refuses so the ring's order is kept.
    pub fn push_many(&self, items: Vec<Vec<u8>>) -> PsResult<Vec<PushOutcome>> {
        push_batch(self.id, items)
    }

    /// Pops one item.
    pub fn pop(&self) -> PsResult<PopOutcome> {
        pop_one(self.id)
    }

    /// Pops up to Count items in one crossing, stopping when the ring
    /// empties. A shorter array than Count means it ran out, not that
    /// anything failed.
    pub fn pop_many(&self, count: i64) -> PsResult<Vec<Vec<u8>>> {
        pop_batch(self.id, count)
    }

    /// Items in the ring now. A hint rather than a fact under
    /// concurrency.
    pub fn len(&self) -> PsResult<u64> {
        Ok(self.stat()?.depth)
    }

    /// Whether the ring read as empty.
    pub fn is_empty(&self) -> PsResult<bool> {
        Ok(self.stat()?.is_empty)
    }

    /// Whether the ring read as full.
    pub fn is_full(&self) -> PsResult<bool> {
        Ok(self.stat()?.is_full)
    }

    /// The depth, the capacity and this handle's four counters in one
    /// crossing.
    pub fn stat(&self) -> PsResult<RingStat> {
        stat_row(self.id, self.role, self.capacity)
    }
}

/// Makes a multi-producer multi-consumer ring over byte items.
///
/// The capacity is rounded up to a power of two, so Capacity on the
/// returned object is what the ring has and may exceed the request.
///
/// A full ring refuses and hands the item back; it never blocks.
///
/// # Examples
///
/// `$ring = New-FlynnelRing -Capacity 1024`
///
/// `$ring.PushMany(@($a, $b, $c)) | Where-Object { -not $_.Accepted }`
///
/// `$ring = [Flynnel.Ring]::new(1024)` builds the same ring without the
/// cmdlet.
#[cmdlet(
    verb = "New",
    noun = "FlynnelRing",
    alias = "New-FlyRing",
    output = ["Flynnel.Ring"]
)]
#[derive(Default)]
pub struct NewFlynnelRing {
    /// Slots to ask for, rounded up to a power of two.
    #[param(mandatory, position = 0)]
    pub capacity: i64,
}

impl Cmdlet for NewFlynnelRing {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(Ring::new(self.capacity)?)
    }
}

// ---------------------------------------------------------------------
// SPSC
// ---------------------------------------------------------------------

/// The push side of a single-producer single-consumer ring.
///
/// Cheapest shape in the crate: no compare-and-swap on either side,
/// because there is exactly one thread on each. That is a contract, not
/// a hint, and nothing here can detect a script that breaks it.
#[psclass(name = "Flynnel.SpscProducer", mode = proxy)]
pub struct SpscProducer {
    /// Names this handle for its session.
    pub id: u64,
    /// Producer.
    pub role: RingRole,
    /// Which producer this is: zero on a plain SPSC ring, the index
    /// within the set on a composed one.
    pub index: u64,
    /// Slots the ring was asked for.
    pub capacity: u64,
    // Read by nothing. Its Drop is the whole of its job, and that is
    // what removes the table entry when the object is disposed or
    // collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: HandleGuard,
}

/// The operations of a `Flynnel.SpscProducer`.
#[psmethods]
impl SpscProducer {
    /// Pushes one item.
    pub fn push(&self, item: Vec<u8>) -> PsResult<PushOutcome> {
        push_one(self.id, item)
    }

    /// Pushes every item in one crossing, stopping at the first refusal.
    pub fn push_many(&self, items: Vec<Vec<u8>>) -> PsResult<Vec<PushOutcome>> {
        push_batch(self.id, items)
    }

    /// This handle's counters. An SPSC producer exposes no reader for
    /// the ring behind it, so DepthKnown is false and the consumer's
    /// row carries no depth either.
    pub fn stat(&self) -> PsResult<RingStat> {
        stat_row(self.id, self.role, self.capacity)
    }
}

/// The pop side of a single-producer single-consumer ring.
#[psclass(name = "Flynnel.SpscConsumer", mode = proxy)]
pub struct SpscConsumer {
    /// Names this handle for its session.
    pub id: u64,
    /// Consumer.
    pub role: RingRole,
    /// Slots the ring was asked for.
    pub capacity: u64,
    // Read by nothing. Its Drop is the whole of its job, and that is
    // what removes the table entry when the object is disposed or
    // collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: HandleGuard,
}

/// The operations of a `Flynnel.SpscConsumer`.
#[psmethods]
impl SpscConsumer {
    /// Pops one item.
    pub fn pop(&self) -> PsResult<PopOutcome> {
        pop_one(self.id)
    }

    /// Pops up to Count items in one crossing, stopping when empty.
    pub fn pop_many(&self, count: i64) -> PsResult<Vec<Vec<u8>>> {
        pop_batch(self.id, count)
    }

    /// This handle's counters.
    pub fn stat(&self) -> PsResult<RingStat> {
        stat_row(self.id, self.role, self.capacity)
    }
}

/// Makes a single-producer single-consumer ring and writes its two
/// ends: the producer first, then the consumer.
///
/// Two objects rather than one, because the ring is only correct with
/// one thread on each end, and two objects put that in the script's
/// hands rather than in a doc comment.
///
/// # Examples
///
/// `$p, $c = New-FlynnelSpscRing -Capacity 1024`
///
/// `$p.PushMany($batch); $c.PopMany(256)`
///
/// `$p, $c = [Flynnel.Rings]::Spsc(1024)` builds the same pair without
/// the cmdlet.
#[cmdlet(
    verb = "New",
    noun = "FlynnelSpscRing",
    alias = "New-FlySpscRing",
    output = ["Flynnel.SpscProducer", "Flynnel.SpscConsumer"]
)]
#[derive(Default)]
pub struct NewFlynnelSpscRing {
    /// Slots to ask for, rounded up to a power of two with a floor of
    /// two.
    #[param(mandatory, position = 0)]
    pub capacity: i64,
}

impl Cmdlet for NewFlynnelSpscRing {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let (producer, consumer) = spsc_ends(self.capacity)?;
        ps.write(producer)?;
        ps.write(consumer)
    }
}

/// An SPSC ring's two ends, producer first. New-FlynnelSpscRing and
/// `[Flynnel.Rings]::Spsc` both build here.
fn spsc_ends(capacity: i64) -> PsResult<(SpscProducer, SpscConsumer)> {
    let cap = checked_capacity(capacity)?;
    let (p, c) = new_spsc::<Payload>(cap);
    let producer_id = register(Handle::SpscProducer(p));
    let producer = SpscProducer {
        id: producer_id,
        role: RingRole::Producer,
        index: 0,
        capacity: cap as u64,
        guard: HandleGuard(producer_id),
    };
    let consumer_id = register(Handle::SpscConsumer(c));
    let consumer = SpscConsumer {
        id: consumer_id,
        role: RingRole::Consumer,
        capacity: cap as u64,
        guard: HandleGuard(consumer_id),
    };
    Ok((producer, consumer))
}

// ---------------------------------------------------------------------
// MPSC
// ---------------------------------------------------------------------

/// A push side of a multi-producer single-consumer ring. Cloneable in
/// the crate; each object here is one clone.
#[psclass(name = "Flynnel.MpscProducer", mode = proxy)]
pub struct MpscProducerHandle {
    /// Names this handle for its session.
    pub id: u64,
    /// Producer.
    pub role: RingRole,
    /// Which producer this is within the set the cmdlet wrote.
    pub index: u64,
    /// Slots in the shared ring, as asked for.
    pub capacity: u64,
    // Read by nothing. Its Drop is the whole of its job, and that is
    // what removes the table entry when the object is disposed or
    // collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: HandleGuard,
}

/// The operations of a `Flynnel.MpscProducer`.
#[psmethods]
impl MpscProducerHandle {
    /// Pushes one item.
    pub fn push(&self, item: Vec<u8>) -> PsResult<PushOutcome> {
        push_one(self.id, item)
    }

    /// Pushes every item in one crossing, stopping at the first refusal.
    ///
    /// Only this producer's items are ordered against each other. Two
    /// producers interleave in whatever order they reach the ring,
    /// which is what multi-producer means.
    pub fn push_many(&self, items: Vec<Vec<u8>>) -> PsResult<Vec<PushOutcome>> {
        push_batch(self.id, items)
    }

    /// This handle's counters.
    pub fn stat(&self) -> PsResult<RingStat> {
        stat_row(self.id, self.role, self.capacity)
    }
}

/// The single pop side of a multi-producer single-consumer ring.
#[psclass(name = "Flynnel.MpscConsumer", mode = proxy)]
pub struct MpscConsumerHandle {
    /// Names this handle for its session.
    pub id: u64,
    /// Consumer.
    pub role: RingRole,
    /// Slots in the shared ring, as asked for.
    pub capacity: u64,
    // Read by nothing. Its Drop is the whole of its job, and that is
    // what removes the table entry when the object is disposed or
    // collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: HandleGuard,
}

/// The operations of a `Flynnel.MpscConsumer`.
#[psmethods]
impl MpscConsumerHandle {
    /// Pops one item.
    pub fn pop(&self) -> PsResult<PopOutcome> {
        pop_one(self.id)
    }

    /// Pops up to Count items in one crossing, stopping when empty.
    pub fn pop_many(&self, count: i64) -> PsResult<Vec<Vec<u8>>> {
        pop_batch(self.id, count)
    }

    /// This handle's counters.
    pub fn stat(&self) -> PsResult<RingStat> {
        stat_row(self.id, self.role, self.capacity)
    }
}

/// Makes a multi-producer single-consumer ring and writes the consumer
/// first, then one producer per requested producer.
///
/// One shared ring behind every producer, so a producer's push contends
/// with the others. New-FlynnelComposedMpsc gives each producer its own
/// ring and no contention at all, at the cost of a consumer that
/// round-robins.
///
/// # Examples
///
/// `$c, $producers = New-FlynnelMpscRing -Capacity 4096 -Producers 4`
///
/// `$producers | ForEach-Object { $_.Index }`
///
/// `$c, $producers = [Flynnel.Rings]::Mpsc(4096, 4)` builds the same
/// objects without the cmdlet.
#[cmdlet(
    verb = "New",
    noun = "FlynnelMpscRing",
    alias = "New-FlyMpscRing",
    output = ["Flynnel.MpscConsumer", "Flynnel.MpscProducer"]
)]
#[derive(Default)]
pub struct NewFlynnelMpscRing {
    /// Slots in the shared ring, rounded up to a power of two.
    #[param(mandatory, position = 0)]
    pub capacity: i64,
    /// How many producer handles to write.
    #[param(mandatory, position = 1)]
    pub producers: i64,
}

impl Cmdlet for NewFlynnelMpscRing {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let (consumer, producers) = mpsc_ends(self.capacity, self.producers)?;
        ps.write(consumer)?;
        for producer in producers {
            ps.write(producer)?;
        }
        Ok(())
    }
}

/// An MPSC ring's consumer and one producer per requested producer,
/// consumer first. New-FlynnelMpscRing and `[Flynnel.Rings]::Mpsc` both
/// build here.
fn mpsc_ends(
    capacity: i64,
    producers: i64,
) -> PsResult<(MpscConsumerHandle, Vec<MpscProducerHandle>)> {
    let cap = checked_capacity(capacity)?;
    let n = checked_handles(producers, "Producers")?;
    let (p, c) = new_mpsc::<Payload>(cap);
    let consumer_id = register(Handle::MpscConsumer(c));
    let consumer = MpscConsumerHandle {
        id: consumer_id,
        role: RingRole::Consumer,
        capacity: cap as u64,
        guard: HandleGuard(consumer_id),
    };
    let handles = (0..n)
        .map(|index| {
            let id = register(Handle::MpscProducer(p.clone()));
            MpscProducerHandle {
                id,
                role: RingRole::Producer,
                index: index as u64,
                capacity: cap as u64,
                guard: HandleGuard(id),
            }
        })
        .collect();
    Ok((consumer, handles))
}

// ---------------------------------------------------------------------
// Composed MPSC: one dedicated SPSC ring per producer
// ---------------------------------------------------------------------

/// The consumer of a composed MPSC: it owns one SPSC consumer per
/// producer and round-robins across them.
#[psclass(name = "Flynnel.ComposedConsumer", mode = proxy)]
pub struct ComposedConsumer {
    /// Names this handle for its session.
    pub id: u64,
    /// Consumer.
    pub role: RingRole,
    /// How many per-producer rings it reads.
    pub ring_count: u64,
    /// Slots in each of those rings, as asked for.
    pub capacity: u64,
    // Read by nothing. Its Drop is the whole of its job, and that is
    // what removes the table entry when the object is disposed or
    // collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: HandleGuard,
}

/// The operations of a `Flynnel.ComposedConsumer`.
#[psmethods]
impl ComposedConsumer {
    /// Pops one item from whichever ring the cursor reaches next.
    ///
    /// Empty means every one of the per-producer rings was empty, not
    /// that the one it looked at was.
    pub fn pop(&self) -> PsResult<PopOutcome> {
        pop_one(self.id)
    }

    /// Pops up to Count items in one crossing, stopping when every ring
    /// is empty.
    ///
    /// The items interleave across producers by the round-robin, so the
    /// order within one producer is kept and the order between them is
    /// not.
    pub fn pop_many(&self, count: i64) -> PsResult<Vec<Vec<u8>>> {
        pop_batch(self.id, count)
    }

    /// This handle's counters.
    pub fn stat(&self) -> PsResult<RingStat> {
        stat_row(self.id, self.role, self.capacity)
    }
}

/// Makes a composed multi-producer single-consumer ring: one dedicated
/// SPSC ring per producer, read round-robin by one consumer. Writes the
/// consumer first, then one producer per requested producer.
///
/// No producer contends with another, because no two share a ring. The
/// cost is that Capacity is per producer rather than shared, so a
/// lopsided workload fills one producer's ring while the others sit
/// empty.
///
/// The producers are the same `Flynnel.SpscProducer` objects a plain
/// SPSC ring writes, because in the crate that is what they are.
///
/// # Examples
///
/// `$c, $producers = New-FlynnelComposedMpsc -Capacity 256 -Producers 8`
///
/// `$c, $producers = [Flynnel.Rings]::ComposedMpsc(256, 8)` builds the
/// same objects without the cmdlet.
#[cmdlet(
    verb = "New",
    noun = "FlynnelComposedMpsc",
    alias = "New-FlyComposedMpsc",
    output = ["Flynnel.ComposedConsumer", "Flynnel.SpscProducer"]
)]
#[derive(Default)]
pub struct NewFlynnelComposedMpsc {
    /// Slots in each producer's own ring, rounded up to a power of two.
    #[param(mandatory, position = 0)]
    pub capacity: i64,
    /// How many producers, and so how many rings.
    #[param(mandatory, position = 1)]
    pub producers: i64,
}

impl Cmdlet for NewFlynnelComposedMpsc {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let (consumer, producers) = composed_mpsc_ends(self.capacity, self.producers)?;
        ps.write(consumer)?;
        for producer in producers {
            ps.write(producer)?;
        }
        Ok(())
    }
}

/// A composed MPSC's consumer and one producer per requested producer,
/// consumer first. New-FlynnelComposedMpsc and
/// `[Flynnel.Rings]::ComposedMpsc` both build here.
fn composed_mpsc_ends(
    capacity: i64,
    producers: i64,
) -> PsResult<(ComposedConsumer, Vec<SpscProducer>)> {
    let cap = checked_capacity(capacity)?;
    let n = checked_handles(producers, "Producers")?;
    let composed = new_composed_mpsc::<Payload>(n, cap);
    let ring_count = composed.consumer.ring_count() as u64;
    let consumer_id = register(Handle::ComposedConsumer(composed.consumer));
    let consumer = ComposedConsumer {
        id: consumer_id,
        role: RingRole::Consumer,
        ring_count,
        capacity: cap as u64,
        guard: HandleGuard(consumer_id),
    };
    let handles = composed
        .producers
        .into_iter()
        .enumerate()
        .map(|(index, p)| {
            let id = register(Handle::SpscProducer(p));
            SpscProducer {
                id,
                role: RingRole::Producer,
                index: index as u64,
                capacity: cap as u64,
                guard: HandleGuard(id),
            }
        })
        .collect();
    Ok((consumer, handles))
}

// ---------------------------------------------------------------------
// Composed MPMC: an N by M grid of dedicated SPSC rings
// ---------------------------------------------------------------------

/// One producer's side of an MPMC grid. Holds one dedicated ring per
/// consumer and round-robins across them.
#[psclass(name = "Flynnel.GridProducer", mode = proxy)]
pub struct GridProducerHandle {
    /// Names this handle for its session.
    pub id: u64,
    /// Producer.
    pub role: RingRole,
    /// Which producer row this is.
    pub index: u64,
    /// How many consumer columns it can push to.
    pub consumer_count: u64,
    /// Slots in each of its rings, as asked for.
    pub capacity: u64,
    // Read by nothing. Its Drop is the whole of its job, and that is
    // what removes the table entry when the object is disposed or
    // collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: HandleGuard,
}

/// The operations of a `Flynnel.GridProducer`.
#[psmethods]
impl GridProducerHandle {
    /// Pushes one item to whichever consumer column has room, starting
    /// at the cursor.
    ///
    /// Full means every column this producer probed was full, so the
    /// whole grid is backed up rather than one consumer being slow.
    pub fn push(&self, item: Vec<u8>) -> PsResult<PushOutcome> {
        push_one(self.id, item)
    }

    /// Pushes every item in one crossing, stopping at the first refusal.
    ///
    /// Successive items land on different consumers by the round-robin,
    /// so a batch is spread across the grid rather than delivered in
    /// order to one consumer.
    pub fn push_many(&self, items: Vec<Vec<u8>>) -> PsResult<Vec<PushOutcome>> {
        push_batch(self.id, items)
    }

    /// This handle's counters.
    pub fn stat(&self) -> PsResult<RingStat> {
        stat_row(self.id, self.role, self.capacity)
    }
}

/// One consumer's side of an MPMC grid. Holds one dedicated ring per
/// producer and round-robins across them.
#[psclass(name = "Flynnel.GridConsumer", mode = proxy)]
pub struct GridConsumerHandle {
    /// Names this handle for its session.
    pub id: u64,
    /// Consumer.
    pub role: RingRole,
    /// Which consumer column this is.
    pub index: u64,
    /// How many producer rows feed it.
    pub producer_count: u64,
    /// Slots in each of its rings, as asked for.
    pub capacity: u64,
    // Read by nothing. Its Drop is the whole of its job, and that is
    // what removes the table entry when the object is disposed or
    // collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: HandleGuard,
}

/// The operations of a `Flynnel.GridConsumer`.
#[psmethods]
impl GridConsumerHandle {
    /// Pops one item from whichever producer row the cursor reaches.
    pub fn pop(&self) -> PsResult<PopOutcome> {
        pop_one(self.id)
    }

    /// Pops up to Count items in one crossing, stopping when every row
    /// feeding this consumer is empty.
    pub fn pop_many(&self, count: i64) -> PsResult<Vec<Vec<u8>>> {
        pop_batch(self.id, count)
    }

    /// This handle's counters.
    pub fn stat(&self) -> PsResult<RingStat> {
        stat_row(self.id, self.role, self.capacity)
    }
}

/// Makes an N-by-M grid of dedicated SPSC rings, one per producer and
/// consumer pair. Writes the producers first, then the consumers.
///
/// Every object carries Role and Index, so a script splits the stream
/// by Role rather than by counting.
///
/// Rings are Producers times Consumers, each of Capacity slots, so the
/// memory is the product of the three. A 16-by-16 grid at 4096 slots is
/// a million slots.
///
/// # Examples
///
/// `$all = New-FlynnelComposedMpmc -Capacity 256 -Producers 4 -Consumers 2`
///
/// `$prod = $all | Where-Object Role -eq Producer`
///
/// `$all = [Flynnel.Rings]::ComposedMpmc(256, 4, 2)` builds the same
/// grid without the cmdlet.
#[cmdlet(
    verb = "New",
    noun = "FlynnelComposedMpmc",
    alias = "New-FlyComposedMpmc",
    output = ["Flynnel.GridProducer", "Flynnel.GridConsumer"]
)]
#[derive(Default)]
pub struct NewFlynnelComposedMpmc {
    /// Slots in each of the Producers times Consumers rings.
    #[param(mandatory, position = 0)]
    pub capacity: i64,
    /// Producer rows.
    #[param(mandatory, position = 1)]
    pub producers: i64,
    /// Consumer columns.
    #[param(mandatory, position = 2)]
    pub consumers: i64,
}

impl Cmdlet for NewFlynnelComposedMpmc {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let (producers, consumers) =
            composed_mpmc_ends(self.capacity, self.producers, self.consumers)?;
        for producer in producers {
            ps.write(producer)?;
        }
        for consumer in consumers {
            ps.write(consumer)?;
        }
        Ok(())
    }
}

/// An MPMC grid's producers and then its consumers, one per requested
/// row and column. New-FlynnelComposedMpmc and
/// `[Flynnel.Rings]::ComposedMpmc` both build here.
fn composed_mpmc_ends(
    capacity: i64,
    producers: i64,
    consumers: i64,
) -> PsResult<(Vec<GridProducerHandle>, Vec<GridConsumerHandle>)> {
    let cap = checked_capacity(capacity)?;
    let n = checked_handles(producers, "Producers")?;
    let m = checked_handles(consumers, "Consumers")?;
    let grid = new_composed_mpmc::<Payload>(n, m, cap);
    let producer_handles = grid
        .producers
        .into_iter()
        .enumerate()
        .map(|(index, p)| {
            let consumer_count = p.consumer_count() as u64;
            let id = register(Handle::GridProducer(p));
            GridProducerHandle {
                id,
                role: RingRole::Producer,
                index: index as u64,
                consumer_count,
                capacity: cap as u64,
                guard: HandleGuard(id),
            }
        })
        .collect();
    let consumer_handles = grid
        .consumers
        .into_iter()
        .enumerate()
        .map(|(index, c)| {
            let producer_count = c.producer_count() as u64;
            let id = register(Handle::GridConsumer(c));
            GridConsumerHandle {
                id,
                role: RingRole::Consumer,
                index: index as u64,
                producer_count,
                capacity: cap as u64,
                guard: HandleGuard(id),
            }
        })
        .collect();
    Ok((producer_handles, consumer_handles))
}

// ---------------------------------------------------------------------
// The injector
// ---------------------------------------------------------------------

/// The global fork queue, one per arena in the crate and free-standing
/// here.
///
/// A ring under a steal-shaped interface: the pop answers Success,
/// Empty or Retry, matching the Chase-Lev owner-deque protocol so one
/// match covers both. Retry never comes back, because the ring's pop
/// loops on contention internally.
#[psclass(name = "Flynnel.Injector", mode = proxy)]
pub struct Injector {
    /// Names this handle for its session.
    pub id: u64,
    /// Ring: one object drives both ends.
    pub role: RingRole,
    /// Slots, rounded up to a power of two.
    pub capacity: u64,
    // Read by nothing. Its Drop is the whole of its job, and that is
    // what removes the table entry when the object is disposed or
    // collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: HandleGuard,
}

/// The operations of a `Flynnel.Injector`.
#[psmethods]
impl Injector {
    /// A free-standing injector of at least Capacity slots, or of the
    /// crate's own 4096 when Capacity is omitted.
    ///
    /// A script reaches this as `[Flynnel.Injector]::new()` or
    /// `[Flynnel.Injector]::new(65536)`. New-FlynnelInjector builds its
    /// queue here as well, so the two routes make the same object.
    pub fn new(capacity: Option<i64>) -> PsResult<Injector> {
        let cap = match capacity {
            Some(c) => checked_capacity(c)?,
            None => DEFAULT_INJECTOR_CAPACITY,
        };
        let inner = InjectorInner::<Payload>::with_capacity(cap);
        let capacity = inner.capacity() as u64;
        let id = register(Handle::Injector(inner));
        Ok(Injector { id, role: RingRole::Ring, capacity, guard: HandleGuard(id) })
    }

    /// Pushes one item, answering whether the queue took it.
    ///
    /// This is the crate's `try_push`. Its `push` is not bound: that one
    /// spins until a slot frees, which on a queue nobody is draining
    /// never returns and cannot be interrupted from a script.
    pub fn push(&self, item: Vec<u8>) -> PsResult<PushOutcome> {
        push_one(self.id, item)
    }

    /// Pushes every item in one crossing, stopping at the first refusal.
    pub fn push_many(&self, items: Vec<Vec<u8>>) -> PsResult<Vec<PushOutcome>> {
        push_batch(self.id, items)
    }

    /// Takes one item, under the steal protocol's three-way answer.
    pub fn pop(&self) -> PsResult<PopOutcome> {
        pop_one(self.id)
    }

    /// Takes up to Count items in one crossing.
    pub fn pop_many(&self, count: i64) -> PsResult<Vec<Vec<u8>>> {
        pop_batch(self.id, count)
    }

    /// Items pending. A hint.
    pub fn len(&self) -> PsResult<u64> {
        Ok(self.stat()?.depth)
    }

    /// Whether it read as empty.
    pub fn is_empty(&self) -> PsResult<bool> {
        Ok(self.stat()?.is_empty)
    }

    /// Whether it read as full.
    pub fn is_full(&self) -> PsResult<bool> {
        Ok(self.stat()?.is_full)
    }

    /// The depth, the capacity and this handle's counters in one
    /// crossing.
    pub fn stat(&self) -> PsResult<RingStat> {
        stat_row(self.id, self.role, self.capacity)
    }
}

/// Makes a free-standing injector queue.
///
/// Free-standing: this is not the arena's injector, and work pushed here
/// does not reach the worker pool. It is the queue's data structure on
/// its own, for a script building a fan-in of its own.
///
/// # Examples
///
/// `$q = New-FlynnelInjector`
///
/// `$q = New-FlynnelInjector -Capacity 65536`
///
/// `$q = [Flynnel.Injector]::new(65536)` builds the same queue without
/// the cmdlet.
#[cmdlet(
    verb = "New",
    noun = "FlynnelInjector",
    alias = "New-FlyInjector",
    output = ["Flynnel.Injector"]
)]
#[derive(Default)]
pub struct NewFlynnelInjector {
    /// Slots to ask for. Defaults to the crate's own 4096.
    #[param(position = 0)]
    pub capacity: Option<i64>,
}

impl Cmdlet for NewFlynnelInjector {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(Injector::new(self.capacity)?)
    }
}

// ---------------------------------------------------------------------
// The notify hub
// ---------------------------------------------------------------------

/// The send side of a notify hub: a ring that wakes a parked consumer
/// on every send.
///
/// The waking is the point, and it is also why only the try-forms are
/// bound. A consumer here is a Rust thread that parks; a PowerShell
/// receiver polls instead, so it never parks and never needs waking.
/// The hub is bound so a script can be one end of a pipeline whose
/// other end is Rust.
#[psclass(name = "Flynnel.NotifySender", mode = proxy)]
pub struct NotifySender {
    /// Names this handle for its session.
    pub id: u64,
    /// Producer.
    pub role: RingRole,
    /// Consumer slots the hub was built with.
    pub consumer_count: u64,
    /// Slots in the shared ring, as asked for.
    pub capacity: u64,
    // Read by nothing. Its Drop is the whole of its job, and that is
    // what removes the table entry when the object is disposed or
    // collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: HandleGuard,
}

/// The operations of a `Flynnel.NotifySender`.
#[psmethods]
impl NotifySender {
    /// Sends one item and wakes one consumer, answering Ok, Full or
    /// Closed.
    ///
    /// This is the crate's `try_send`. Its `send` is not bound: that one
    /// spins while the ring is full and returns only when a consumer
    /// drains or the hub shuts down, neither of which a script waiting
    /// inside it could bring about.
    pub fn push(&self, item: Vec<u8>) -> PsResult<PushOutcome> {
        push_one(self.id, item)
    }

    /// Sends every item in one crossing, stopping at the first refusal.
    ///
    /// A Closed outcome means every later item would be refused too, so
    /// the stop is not merely about ordering: it is the terminal state.
    pub fn push_many(&self, items: Vec<Vec<u8>>) -> PsResult<Vec<PushOutcome>> {
        push_batch(self.id, items)
    }

    /// Items in the hub's ring now. A hint.
    pub fn len(&self) -> PsResult<u64> {
        Ok(self.stat()?.depth)
    }

    /// Whether the ring read as empty.
    pub fn is_empty(&self) -> PsResult<bool> {
        Ok(self.stat()?.is_empty)
    }

    /// Shuts the hub down: later sends answer Closed, and every parked
    /// consumer wakes to drain what is left. Safe to call twice.
    pub fn shutdown(&self) -> PsResult<()> {
        with_entry(self.id, |entry| match &entry.handle {
            Handle::NotifySender(s, _) => {
                s.shutdown();
                Ok(())
            }
            _ => Err(PsError::new(
                ErrorCategory::InvalidOperation,
                "FlynnelWrongSide",
                format!(
                    "ring handle {} is not a notify sender and has no hub to shut down",
                    self.id
                ),
            )),
        })
    }

    /// The depth and this handle's counters in one crossing.
    pub fn stat(&self) -> PsResult<RingStat> {
        stat_row(self.id, self.role, self.capacity)
    }
}

/// The receive side of a notify hub.
///
/// Registered against one of the hub's consumer slots when it was made.
/// A hub built for two consumers has two slots, and a third receiver
/// wraps onto the first, so ask for as many as the script will use.
#[psclass(name = "Flynnel.NotifyReceiver", mode = proxy)]
pub struct NotifyReceiver {
    /// Names this handle for its session.
    pub id: u64,
    /// Consumer.
    pub role: RingRole,
    /// Which consumer slot it registered against.
    pub index: u64,
    /// Slots in the shared ring, as asked for.
    pub capacity: u64,
    // Read by nothing. Its Drop is the whole of its job, and that is
    // what removes the table entry when the object is disposed or
    // collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: HandleGuard,
}

/// The operations of a `Flynnel.NotifyReceiver`.
#[psmethods]
impl NotifyReceiver {
    /// Takes one item without waiting.
    ///
    /// A drained hub that has been shut down answers Empty, the same as
    /// one that is merely idle: PopKind.Closed says why nothing can
    /// tell them apart from outside. A script that owns the sender
    /// knows from its own Shutdown call. The crate's blocking `recv` is
    /// not bound: it parks the calling thread, which from a script is a
    /// host that cannot be interrupted.
    pub fn pop(&self) -> PsResult<PopOutcome> {
        pop_one(self.id)
    }

    /// Takes up to Count items in one crossing, stopping when the ring
    /// empties.
    pub fn pop_many(&self, count: i64) -> PsResult<Vec<Vec<u8>>> {
        pop_batch(self.id, count)
    }

    /// Items in the hub's ring now. A hint.
    pub fn len(&self) -> PsResult<u64> {
        Ok(self.stat()?.depth)
    }

    /// Whether the ring read as empty.
    pub fn is_empty(&self) -> PsResult<bool> {
        Ok(self.stat()?.is_empty)
    }

    /// The depth and this handle's counters in one crossing.
    pub fn stat(&self) -> PsResult<RingStat> {
        stat_row(self.id, self.role, self.capacity)
    }
}

/// Makes a notify hub and writes its sender first, then one receiver per
/// requested consumer.
///
/// The hub wakes a parked consumer on every send. A PowerShell receiver
/// polls rather than parking, so the waking matters only when the other
/// end is Rust; from a script this is an MPMC ring with a shutdown.
///
/// # Examples
///
/// `$s, $receivers = New-FlynnelNotifyRing -Capacity 1024 -Consumers 2`
///
/// `try { ... } finally { $s.Shutdown() }`
///
/// `$s, $receivers = [Flynnel.Rings]::Notify(1024, 2)` builds the same
/// hub without the cmdlet.
#[cmdlet(
    verb = "New",
    noun = "FlynnelNotifyRing",
    alias = "New-FlyNotifyRing",
    output = ["Flynnel.NotifySender", "Flynnel.NotifyReceiver"]
)]
#[derive(Default)]
pub struct NewFlynnelNotifyRing {
    /// Slots in the shared ring, rounded up to a power of two.
    #[param(mandatory, position = 0)]
    pub capacity: i64,
    /// How many consumer slots, and so how many receivers to write.
    #[param(mandatory, position = 1)]
    pub consumers: i64,
}

impl Cmdlet for NewFlynnelNotifyRing {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let (sender, receivers) = notify_ends(self.capacity, self.consumers)?;
        ps.write(sender)?;
        for receiver in receivers {
            ps.write(receiver)?;
        }
        Ok(())
    }
}

/// A notify hub's sender and one receiver per requested consumer slot,
/// sender first. New-FlynnelNotifyRing and `[Flynnel.Rings]::Notify`
/// both build here.
fn notify_ends(capacity: i64, consumers: i64) -> PsResult<(NotifySender, Vec<NotifyReceiver>)> {
    let cap = checked_capacity(capacity)?;
    let m = checked_handles(consumers, "Consumers")?;
    let hub = NotifyHub::<Payload>::new(cap, m);
    let sender_id = register(Handle::NotifySender(hub.sender(), hub.clone()));
    let sender = NotifySender {
        id: sender_id,
        role: RingRole::Producer,
        consumer_count: m as u64,
        capacity: cap as u64,
        guard: HandleGuard(sender_id),
    };
    let receivers = (0..m)
        .map(|index| {
            let id = register(Handle::NotifyReceiver(hub.register_consumer(), hub.clone()));
            NotifyReceiver {
                id,
                role: RingRole::Consumer,
                index: index as u64,
                capacity: cap as u64,
                guard: HandleGuard(id),
            }
        })
        .collect();
    Ok((sender, receivers))
}

// ---------------------------------------------------------------------
// The factory for the shapes that come as several objects
// ---------------------------------------------------------------------

/// Builds the ring shapes that come as several objects: a ring's two
/// ends, or one end with the handles on the other side.
///
/// Each method returns the objects in the order its New- cmdlet writes
/// them, so `$p, $c = [Flynnel.Rings]::Spsc(1024)` unpacks the way
/// `$p, $c = New-FlynnelSpscRing -Capacity 1024` does, and the cmdlet
/// builds through the same function. A single ring and the injector are
/// one object each and have constructors of their own,
/// `[Flynnel.Ring]::new(...)` and `[Flynnel.Injector]::new(...)`.
///
/// Nothing makes a Rings object. The type carries only these statics.
#[psclass(name = "Flynnel.Rings", mode = proxy)]
pub struct Rings {}

/// The statics of `Flynnel.Rings`.
#[psmethods]
impl Rings {
    /// A single-producer single-consumer ring: the producer, then the
    /// consumer, as New-FlynnelSpscRing writes them.
    pub fn spsc(capacity: i64) -> PsResult<Vec<PsObject>> {
        let (producer, consumer) = spsc_ends(capacity)?;
        Ok(vec![producer.into_ps()?, consumer.into_ps()?])
    }

    /// A multi-producer single-consumer ring: the consumer, then one
    /// producer per requested producer, as New-FlynnelMpscRing writes
    /// them.
    pub fn mpsc(capacity: i64, producers: i64) -> PsResult<Vec<PsObject>> {
        let (consumer, handles) = mpsc_ends(capacity, producers)?;
        one_then_many(consumer, handles)
    }

    /// A composed MPSC, one dedicated ring per producer: the consumer,
    /// then the producers, as New-FlynnelComposedMpsc writes them.
    pub fn composed_mpsc(capacity: i64, producers: i64) -> PsResult<Vec<PsObject>> {
        let (consumer, handles) = composed_mpsc_ends(capacity, producers)?;
        one_then_many(consumer, handles)
    }

    /// An N-by-M grid of dedicated rings: the producers, then the
    /// consumers, as New-FlynnelComposedMpmc writes them.
    pub fn composed_mpmc(capacity: i64, producers: i64, consumers: i64) -> PsResult<Vec<PsObject>> {
        let (producer_handles, consumer_handles) = composed_mpmc_ends(capacity, producers, consumers)?;
        let mut out = Vec::with_capacity(producer_handles.len() + consumer_handles.len());
        for p in producer_handles {
            out.push(p.into_ps()?);
        }
        for c in consumer_handles {
            out.push(c.into_ps()?);
        }
        Ok(out)
    }

    /// A notify hub: the sender, then one receiver per consumer slot, as
    /// New-FlynnelNotifyRing writes them.
    pub fn notify(capacity: i64, consumers: i64) -> PsResult<Vec<PsObject>> {
        let (sender, receivers) = notify_ends(capacity, consumers)?;
        one_then_many(sender, receivers)
    }
}

/// One object followed by several, as the objects a script unpacks.
fn one_then_many<A: IntoPs, B: IntoPs>(first: A, rest: Vec<B>) -> PsResult<Vec<PsObject>> {
    let mut out = Vec::with_capacity(1 + rest.len());
    out.push(first.into_ps()?);
    for r in rest {
        out.push(r.into_ps()?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// The pipeline forms
// ---------------------------------------------------------------------

/// Reads the handle id off whichever ring object the caller passed.
///
/// Every class in this module carries `Id` and nothing else identifies
/// a handle, so one accessor serves all eleven and the cmdlets do not
/// have to know which shape they were handed.
fn handle_id(obj: &PsObject, parameter: &str) -> PsResult<u64> {
    if obj.is_null() {
        return Err(arg_err(format!(
            "-{parameter} was given $null rather than a ring object"
        )));
    }
    let id = obj.get("Id").map_err(|e| {
        arg_err(format!(
            "-{parameter} needs an object from one of the New-Flynnel*Ring cmdlets; \
             reading its Id property failed: {e}"
        ))
    })?;
    u64::from_ps(&id)
        .map_err(|e| arg_err(format!("-{parameter} has an Id that is not a number: {e}")))
}

/// Pushes pipeline items into a ring, writing back every item the ring
/// refused.
///
/// A refused item is written to the pipeline rather than dropped, so a
/// full ring costs a retry and never costs data: collect the output and
/// pass it in again.
///
/// One pipeline record costs 1712 ns on PowerShell 7.6 and 7955 ns on
/// Windows PowerShell before anything here runs, against about 40 ns
/// for the ring's own push. So this form is for interleaving with other
/// pipeline stages; PushMany on the object is the form for throughput,
/// at 39 ns an item against 4848.
///
/// # Examples
///
/// `$items | Send-FlynnelItem -To $ring`
///
/// `$rejected = $items | Send-FlynnelItem -To $producer`
#[cmdlet(
    verb = "Send",
    noun = "FlynnelItem",
    alias = "Send-FlyItem",
    output = ["System.Byte[]"]
)]
#[derive(Default)]
pub struct SendFlynnelItem {
    /// The ring, producer, injector or sender to push into.
    #[param(mandatory, position = 0)]
    pub to: PsObject,
    /// One item, from the pipeline.
    #[param(mandatory, value_from_pipeline)]
    pub input_object: Vec<u8>,
    /// The handle id, read from `-To` on the first record.
    ///
    /// Reading a property is a call into the managed side, and `-To`
    /// is bound once for the whole pipeline while `process` runs per
    /// record, so resolving it every time would add a crossing to the
    /// path this cmdlet exists to keep short. Cleared in `begin`, so a
    /// second invocation of the same instance resolves its own.
    ///
    /// It carries no `#[param]`, which is what keeps it out of the
    /// parameter block: the macro walks the fields and skips every one
    /// that is not annotated.
    resolved: Option<u64>,
}

impl Cmdlet for SendFlynnelItem {
    fn begin(&mut self, _ps: &Pipeline<'_>) -> PsResult<()> {
        self.resolved = None;
        Ok(())
    }

    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let id = match self.resolved {
            Some(id) => id,
            None => {
                let id = handle_id(&self.to, "To")?;
                self.resolved = Some(id);
                id
            }
        };
        let item = std::mem::take(&mut self.input_object);
        let outcome = push_one(id, item)?;
        if outcome.accepted {
            return Ok(());
        }
        // PsArray rather than the bare Vec, so one refused item is one
        // record holding a byte[] instead of one record per byte.
        ps.write(PsArray(outcome.item))
    }
}

/// Reads items out of a ring and writes them to the pipeline, stopping
/// when it empties or when Count have come out.
///
/// Stops on empty rather than waiting, because there is no bound wait
/// in this module: a script that wants to wait writes the loop, where
/// its own Start-Sleep and Ctrl-C both work.
///
/// One pipeline record costs 1712 ns on PowerShell 7.6 and 7955 ns on
/// Windows PowerShell. PopMany on the object returns the same items in
/// one crossing.
///
/// # Examples
///
/// `Receive-FlynnelItem -From $consumer`
///
/// `Receive-FlynnelItem -From $ring -Count 100 | ForEach-Object { ... }`
#[cmdlet(
    verb = "Receive",
    noun = "FlynnelItem",
    alias = "Receive-FlyItem",
    output = ["System.Byte[]"]
)]
#[derive(Default)]
pub struct ReceiveFlynnelItem {
    /// The ring, consumer, injector or receiver to read from.
    #[param(mandatory, position = 0)]
    pub from: PsObject,
    /// At most this many items. Unset reads until the ring is empty.
    #[param(position = 1)]
    pub count: Option<i64>,
}

impl Cmdlet for ReceiveFlynnelItem {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let id = handle_id(&self.from, "From")?;
        let limit = match self.count {
            Some(c) => checked_count(c)?,
            None => usize::MAX,
        };
        for _ in 0..limit {
            let outcome = pop_one(id)?;
            if !outcome.got_item {
                break;
            }
            ps.write(PsArray(outcome.item))?;
            // The pipeline thread runs this loop, so between records is
            // the only place a Ctrl-C can be seen. Without it, an
            // unbounded read of a ring another thread keeps filling
            // would never return.
            if ps.stopping() {
                break;
            }
        }
        Ok(())
    }
}

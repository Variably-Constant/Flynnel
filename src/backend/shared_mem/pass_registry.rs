//! Process-local closure registry. Each participating process
//! registers `closure_id -> handler` at startup; the wire-format
//! `Pass` record carries `(closure_id, args)` rather than the
//! closure code itself.
//!
//! Rust closures cannot be safely serialized across process
//! boundaries - function pointers are not position-stable across
//! address spaces and captured environment can hold non-portable
//! types. The pattern here mirrors how Ray, Akka, and similar
//! distributed actor frameworks dispatch user code: each peer
//! declares the closures it knows about, and the wire only
//! carries the id + serialized args.
//!
//! Identifiers are caller-chosen `u32` values. For deterministic
//! cross-process agreement, callers can hash a stable string name
//! and use the resulting value as both the registry key and the
//! wire id; [`hash_name`] provides an FNV-1a hash for that purpose.

use core::cell::Cell;
use core::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::Arc;

use crate::sched::hazard::HazardDomain;

/// One unit of work that can be sent across the ring: a
/// closure-identifier plus its already-serialized argument blob.
/// The handler the receiving process has registered under
/// `closure_id` is responsible for decoding `args`.
#[derive(Debug, Clone)]
pub struct Pass {
    /// Identifier the receiving process uses to look up its
    /// registered handler. Typically derived deterministically
    /// via [`hash_name`] from a stable kernel name.
    pub closure_id: u32,
    /// Already-serialized argument blob; the handler decodes it.
    pub args: Vec<u8>,
}

/// Result of executing a Pass. Bytes are caller-defined; the
/// originating process knows how to interpret them.
pub type PassResult = Result<Vec<u8>, PassError>;

/// Failure modes for [`execute`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PassError {
    /// No handler is registered under this `closure_id` in the
    /// current process.
    UnknownClosureId(u32),
    /// The handler itself returned an error; payload is its
    /// human-readable diagnostic.
    ExecutionError(String),
}

impl std::fmt::Display for PassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PassError::UnknownClosureId(id) => write!(f, "no handler for closure id {id}"),
            PassError::ExecutionError(msg) => write!(f, "handler failed: {msg}"),
        }
    }
}

impl std::error::Error for PassError {}

/// Handler shape: takes raw arg bytes, returns either raw response
/// bytes or a structured error.
/// Shared rather than owned so that a handler handed back by
/// [`register`] or [`unregister`] stays callable while a peer is
/// still executing through the copy the table held.
pub type PassHandler = Arc<dyn Fn(&[u8]) -> PassResult + Send + Sync + 'static>;

/// Slot count of the handler table.
///
/// Slots are RECYCLED, which is what makes a fixed size right here.
/// Handler ids churn: `dispatch_calibration` registers under a
/// nanosecond nonce and unregisters, a fresh id per measurement, so a
/// table that only ever claimed slots would fill and start refusing
/// registrations. What bounds the table is the number of handlers
/// live at once, which is a handful.
const SLOTS: usize = 256;
const MASK: usize = SLOTS - 1;

/// Threads that may execute a pass. One slot each, claimed on first
/// execute and never returned.
const READERS: usize = 256;

/// Replaced handlers awaiting a sweep. Only one per live handler can
/// be outstanding at a time, so this is far above what the churn
/// needs.
const RETIRED: usize = 256;

/// A slot's identity and whether a handler is installed, in one word
/// so that claiming, tombstoning and re-keying are each a single
/// compare-exchange.
///
/// Zero means never used. Otherwise the low bits are `id + 1` and
/// [`LIVE`] says whether a handler is installed. A removal keeps the
/// key and clears `LIVE`, so the probe chain is never broken, and a
/// slot in that state can be re-keyed for a different id, which is
/// what stops the table filling.
/// A handler is installed and reachable.
const LIVE: u64 = 1 << 40;
/// A registrar owns this slot and is about to install one. Distinct
/// from a tombstone because a tombstone may be re-keyed and a slot
/// someone is mid-registration on may not.
const RESERVED: u64 = 1 << 41;
/// The `id + 1` a slot is keyed to, with both flags removed.
const KEY_MASK: u64 = !(LIVE | RESERVED);

struct Slot {
    state: AtomicU64,
    handler: AtomicPtr<PassHandler>,
}

static TABLE: [Slot; SLOTS] = [const {
    Slot {
        state: AtomicU64::new(0),
        handler: AtomicPtr::new(core::ptr::null_mut()),
    }
}; SLOTS];

static DOMAIN: HazardDomain<PassHandler, READERS, RETIRED> = HazardDomain::new();

fn key_of(id: u32) -> u64 {
    id as u64 + 1
}

fn reader_slot() -> usize {
    thread_local! {
        static READER: Cell<usize> = const { Cell::new(usize::MAX) };
    }
    READER.with(|cell| {
        let held = cell.get();
        if held != usize::MAX {
            return held;
        }
        let fresh = DOMAIN.claim_reader();
        cell.set(fresh);
        fresh
    })
}

/// The slot currently holding `id` live, if any.
fn find_live(id: u32) -> Option<&'static Slot> {
    let key = key_of(id);
    let mut idx = (id as usize) & MASK;
    for _ in 0..SLOTS {
        let state = TABLE[idx].state.load(Ordering::Acquire);
        if state == 0 {
            return None;
        }
        if state == key | LIVE {
            return Some(&TABLE[idx]);
        }
        idx = (idx + 1) & MASK;
    }
    None
}

/// Install `fresh` under `id` and answer what it displaced.
///
/// A slot already keyed to `id` is reused whether or not it is live.
/// Otherwise the first slot that is untouched, or tombstoned under
/// some other id, is claimed by a compare-exchange on its state word,
/// so exactly one registrar wins it.
fn install(id: u32, fresh: *mut PassHandler) -> Option<PassHandler> {
    let key = key_of(id);
    let slot = claim(id, key);
    // The slot is ours before any handler is written, so nothing can
    // re-key it underneath this and no other registrar can be writing
    // the same slot for a different id.
    // SeqCst because this unlink is one of the four operations in the
    // hazard domain's Dekker pair; see HazardDomain::retire for what an
    // AcqRel here would permit.
    let previous = slot.handler.swap(fresh, Ordering::SeqCst);
    slot.state.store(key | LIVE, Ordering::Release);
    displace(previous)
}

/// The slot this id owns, claiming one if it does not own one yet.
///
/// A slot is takeable only when it has never been used or is a bare
/// tombstone, and the take is a compare-exchange on the whole state
/// word, so exactly one registrar wins it. A tombstone stays occupied
/// through the change, so re-keying never cuts a probe chain: a
/// lookup that walks past a re-keyed slot is looking for an id that
/// is genuinely absent from it.
fn claim(id: u32, key: u64) -> &'static Slot {
    let mut idx = (id as usize) & MASK;
    for _ in 0..(SLOTS * 2) {
        let slot = &TABLE[idx];
        let state = slot.state.load(Ordering::Acquire);
        if state & KEY_MASK == key && state & (LIVE | RESERVED) != 0 {
            // Already ours and not a bare tombstone, so it cannot be
            // taken from under us. A concurrent registrar of the SAME
            // id may be here too; the later handler wins, which is
            // what re-registration means.
            return slot;
        }
        if state == 0 || state & (LIVE | RESERVED) == 0 {
            if slot
                .state
                .compare_exchange(state, key | RESERVED, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return slot;
            }
            // Lost the race for this slot; read it again rather than
            // moving on, because it may now be ours.
            continue;
        }
        idx = (idx + 1) & MASK;
    }
    panic!("pass registry full at {SLOTS} live handlers; id {id} could not be registered");
}

/// Hand back a copy of what a slot held and retire the table's own.
fn displace(previous: *mut PassHandler) -> Option<PassHandler> {
    if previous.is_null() {
        return None;
    }
    // SAFETY: a non-null handler pointer in the table came from
    // Box::into_raw and is not freed until a sweep proves no reader
    // holds it, and this read happens before that retire.
    let handed_back = unsafe { &*previous }.clone();
    // SAFETY: the swap above removed it from the only source a reader
    // can reach, and it is retired exactly once.
    unsafe { DOMAIN.retire(previous) };
    Some(handed_back)
}

/// Register `handler` under `id`. Returns the previous handler if
/// one was registered, so callers that want unique-id semantics can
/// assert on `None`. Re-registration is intentional: hot-reload of
/// handler implementations works by re-registering the same id.
pub fn register<F>(id: u32, handler: F) -> Option<PassHandler>
where
    F: Fn(&[u8]) -> PassResult + Send + Sync + 'static,
{
    let shared: PassHandler = Arc::new(handler);
    install(id, Box::into_raw(Box::new(shared)))
}

/// Unregister `id`; returns the previously-registered handler if any.
pub fn unregister(id: u32) -> Option<PassHandler> {
    let key = key_of(id);
    let mut idx = (id as usize) & MASK;
    for _ in 0..SLOTS {
        let slot = &TABLE[idx];
        let state = slot.state.load(Ordering::Acquire);
        if state == 0 {
            return None;
        }
        if state == key | LIVE
            && slot
                .state
                .compare_exchange(key | LIVE, key, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            // SeqCst for the reason given on the swap in `install`.
            let removed = slot.handler.swap(core::ptr::null_mut(), Ordering::SeqCst);
            return displace(removed);
        }
        idx = (idx + 1) & MASK;
    }
    None
}

/// True when `id` has a handler registered in the current process.
pub fn is_registered(id: u32) -> bool {
    find_live(id).is_some()
}

/// Number of handlers registered in the current process.
///
/// A registration landing in a slot the walk has passed is not in the
/// count, so this is the number live at some point during the walk
/// rather than at an instant.
pub fn registered_count() -> usize {
    TABLE
        .iter()
        .filter(|slot| slot.state.load(Ordering::Acquire) & LIVE != 0)
        .count()
}

/// Execute `pass` against the locally-registered handler. Returns
/// [`PassError::UnknownClosureId`] when no handler is registered
/// under `pass.closure_id`.
pub fn execute(pass: &Pass) -> PassResult {
    let id = pass.closure_id;
    let key = key_of(id);
    let reader = reader_slot();
    let mut idx = (id as usize) & MASK;
    for _ in 0..SLOTS {
        let slot = &TABLE[idx];
        let state = slot.state.load(Ordering::Acquire);
        if state == 0 {
            break;
        }
        if state == key | LIVE {
            let guard = DOMAIN.protect(reader, &slot.handler);
            // The slot could have been removed or re-keyed between the
            // state read and the protect, so the state is confirmed
            // again now that the handler cannot be freed. Without this
            // a re-keyed slot would run another id's handler.
            if slot.state.load(Ordering::Acquire) != key | LIVE {
                break;
            }
            if let Some(handler) = guard.get() {
                return (**handler)(&pass.args);
            }
            break;
        }
        idx = (idx + 1) & MASK;
    }
    Err(PassError::UnknownClosureId(id))
}

/// Deterministic 32-bit hash of a stable string name. FNV-1a 32-bit
/// variant; same input always yields the same id across processes
/// and runs, so callers can use [`hash_name`] to derive `closure_id`
/// from a kernel name without coordinating numeric ids out-of-band.
pub fn hash_name(name: &str) -> u32 {
    let mut hash: u32 = 0x811C_9DC5;
    for byte in name.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_then_execute_round_trips() {
        let id = hash_name("flynnel_test_doubler");
        register(id, |args| {
            assert_eq!(args.len(), 8);
            let mut arr = [0u8; 8];
            arr.copy_from_slice(args);
            let n = u64::from_le_bytes(arr);
            Ok((n * 2).to_le_bytes().to_vec())
        });

        let pass = Pass {
            closure_id: id,
            args: 21u64.to_le_bytes().to_vec(),
        };
        let bytes = execute(&pass).expect("execute");
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes);
        assert_eq!(u64::from_le_bytes(arr), 42);
        unregister(id);
    }

    #[test]
    fn unknown_id_returns_specific_error() {
        let pass = Pass {
            closure_id: 0xDEAD_BEEF,
            args: vec![],
        };
        match execute(&pass) {
            Err(PassError::UnknownClosureId(id)) => assert_eq!(id, 0xDEAD_BEEF),
            other => panic!("expected UnknownClosureId, got {other:?}"),
        }
    }

    #[test]
    fn hash_name_is_deterministic() {
        let a = hash_name("flynnel.kernels.add_one");
        let b = hash_name("flynnel.kernels.add_one");
        assert_eq!(a, b);
        let c = hash_name("flynnel.kernels.add_two");
        assert_ne!(a, c, "different names must hash to different ids");
    }

    #[test]
    fn re_register_returns_previous_handler() {
        let id = hash_name("flynnel_test_rereg");
        assert!(register(id, |_| Ok(vec![1])).is_none());
        let prev = register(id, |_| Ok(vec![2]));
        assert!(prev.is_some(), "second register must surface previous");
        unregister(id);
    }
}

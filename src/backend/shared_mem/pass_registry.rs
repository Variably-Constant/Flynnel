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

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

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
pub type PassHandler = Box<dyn Fn(&[u8]) -> PassResult + Send + Sync + 'static>;

fn registry() -> &'static RwLock<HashMap<u32, PassHandler>> {
    static CACHE: OnceLock<RwLock<HashMap<u32, PassHandler>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register `handler` under `id`. Returns the previous handler if
/// one was registered, so callers that want unique-id semantics can
/// assert on `None`. Re-registration is intentional: hot-reload of
/// handler implementations works by re-registering the same id.
pub fn register<F>(id: u32, handler: F) -> Option<PassHandler>
where
    F: Fn(&[u8]) -> PassResult + Send + Sync + 'static,
{
    let mut g = registry().write().expect("pass_registry write lock poisoned");
    g.insert(id, Box::new(handler))
}

/// Unregister `id`; returns the previously-registered handler if any.
pub fn unregister(id: u32) -> Option<PassHandler> {
    let mut g = registry().write().expect("pass_registry write lock poisoned");
    g.remove(&id)
}

/// True when `id` has a handler registered in the current process.
pub fn is_registered(id: u32) -> bool {
    let g = registry().read().expect("pass_registry read lock poisoned");
    g.contains_key(&id)
}

/// Number of handlers registered in the current process.
pub fn registered_count() -> usize {
    let g = registry().read().expect("pass_registry read lock poisoned");
    g.len()
}

/// Execute `pass` against the locally-registered handler. Returns
/// [`PassError::UnknownClosureId`] when no handler is registered
/// under `pass.closure_id`.
pub fn execute(pass: &Pass) -> PassResult {
    let g = registry().read().expect("pass_registry read lock poisoned");
    match g.get(&pass.closure_id) {
        Some(handler) => handler(&pass.args),
        None => Err(PassError::UnknownClosureId(pass.closure_id)),
    }
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

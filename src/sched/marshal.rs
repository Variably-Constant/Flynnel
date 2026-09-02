//! `Marshal` trait: jobs whose captured state can cross a process
//! boundary.
//!
//! ## Why a new trait beside `Send`?
//!
//! `Send` says "moveable across threads in the same address space."
//! An environment-capturing closure can be `Send` (the heap pointers
//! it captures are valid in the spawning process) but not portable
//! across processes (those heap pointers are meaningless in any
//! other address space).
//!
//! `Marshal` is the stricter contract:
//!
//! 1. The work can be reduced to a `(closure_id: u32, args: [u8;
//!    N])` pair.
//! 2. Both processes have registered the same handler under the same
//!    deterministic `closure_id`.
//! 3. The args fit in [`super::super::backend::shared_mem::chase_lev_mmf::ARGS_INLINE_BYTES`].
//!
//! ## Shape
//!
//! ```rust,ignore
//! struct AddOp { a: u32, b: u32 }
//!
//! impl flynnel::sched::Marshal for AddOp {
//!     const HANDLER_NAME: &'static str = "flynnel.demo.add";
//!
//!     fn marshal_args(&self) -> Vec<u8> {
//!         let mut buf = Vec::with_capacity(8);
//!         buf.extend_from_slice(&self.a.to_le_bytes());
//!         buf.extend_from_slice(&self.b.to_le_bytes());
//!         buf
//!     }
//! }
//! ```
//!
//! The handler at the receiving end decodes `args` symmetrically and
//! returns a result `Vec<u8>` that the originator decodes after the
//! latch is published.
//!
//! ## Integration boundary
//!
//! `Marshal` is the data-shape contract. The transport for a marshal
//! job is the MMF Chase-Lev deque at
//! [`super::super::backend::shared_mem::chase_lev_mmf`]; the
//! result-publication transport is the MMF latch arena at
//! [`super::super::backend::shared_mem::latch_mmf`]. The high-level
//! dispatch API that glues these together lives in
//! [`super::super::backend::shared_mem::chase_lev_backend`].

#![allow(clippy::missing_errors_doc)]

use crate::backend::shared_mem::pass_registry;

/// A job whose captured state can be serialized + dispatched across
/// a process boundary.
///
/// Implementors declare a stable handler name and a `marshal_args`
/// serializer. Both processes must register a handler under
/// [`pass_registry::hash_name`] of the same `HANDLER_NAME` before the
/// originator can dispatch this work.
pub trait Marshal {
    /// Stable handler name. The `closure_id` derives deterministically
    /// from this via [`pass_registry::hash_name`].
    const HANDLER_NAME: &'static str;

    /// Serialize the captured state into a flat byte blob. The
    /// receiving handler decodes it symmetrically.
    fn marshal_args(&self) -> Vec<u8>;

    /// Deterministic closure id. Default: FNV-1a hash of
    /// `HANDLER_NAME`. Override only if you need a custom id (e.g.,
    /// to share an id with a non-string-named legacy handler).
    fn closure_id() -> u32
    where
        Self: Sized,
    {
        pass_registry::hash_name(Self::HANDLER_NAME)
    }
}

/// Helper: register a Marshal type's handler in this process.
///
/// The handler receives the bytes produced by
/// [`Marshal::marshal_args`] and returns either a result blob or an
/// error. Typical usage:
///
/// ```rust,ignore
/// flynnel::sched::register_marshal_handler::<AddOp>(|args| {
///     let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
///     let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
///     Ok((a + b).to_le_bytes().to_vec())
/// });
/// ```
///
/// Each peer process must call this for every Marshal type it needs
/// to execute. The originator does NOT need to register the handler
/// if it only dispatches and never drains.
pub fn register_marshal_handler<M: Marshal>(
    handler: impl Fn(&[u8]) -> pass_registry::PassResult + Send + Sync + 'static,
) -> Option<pass_registry::PassHandler> {
    pass_registry::register(M::closure_id(), handler)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DemoAdd {
        a: u32,
        b: u32,
    }

    impl Marshal for DemoAdd {
        const HANDLER_NAME: &'static str = "flynnel.marshal.tests.demo_add";
        fn marshal_args(&self) -> Vec<u8> {
            let mut buf = Vec::with_capacity(8);
            buf.extend_from_slice(&self.a.to_le_bytes());
            buf.extend_from_slice(&self.b.to_le_bytes());
            buf
        }
    }

    #[test]
    fn closure_id_is_deterministic_across_calls() {
        let a = DemoAdd::closure_id();
        let b = DemoAdd::closure_id();
        assert_eq!(a, b);
    }

    #[test]
    fn closure_id_matches_pass_registry_hash() {
        assert_eq!(DemoAdd::closure_id(), pass_registry::hash_name(DemoAdd::HANDLER_NAME));
    }

    #[test]
    fn marshal_args_round_trips_through_handler() {
        // Register the handler under DemoAdd's deterministic id.
        register_marshal_handler::<DemoAdd>(|args| {
            assert_eq!(args.len(), 8);
            let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
            Ok((a + b).to_le_bytes().to_vec())
        });

        let op = DemoAdd { a: 13, b: 29 };
        let args = op.marshal_args();
        let pass = pass_registry::Pass {
            closure_id: DemoAdd::closure_id(),
            args,
        };
        let out = pass_registry::execute(&pass).expect("execute");
        let mut arr = [0u8; 4];
        arr.copy_from_slice(&out[..4]);
        assert_eq!(u32::from_le_bytes(arr), 42);

        pass_registry::unregister(DemoAdd::closure_id());
    }
}

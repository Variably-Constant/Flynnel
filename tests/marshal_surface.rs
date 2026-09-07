//! The `Marshal` trait surface: the identity two processes agree on
//! without talking to each other.
//!
//! A job crosses a process boundary as `(closure_id, args)`, never as
//! code. The id is derived from a name, so the originator and the peer
//! reach the same number independently. Everything here is about that
//! derivation holding: two types must not collide, the documented
//! override must actually override, and re-registering must hand back
//! what it displaced so a hot reload can tell it replaced something
//! rather than adding a second entry nobody reaches.

#![cfg(feature = "shared-memory-worker-reference")]

use flynnel::backend::shared_mem::pass_registry::{hash_name, is_registered, unregister};
use flynnel::sched::marshal::{Marshal, register_marshal_handler};

struct AddOp {
    a: u32,
    b: u32,
}

impl Marshal for AddOp {
    const HANDLER_NAME: &'static str = "flynnel.tests.marshal.add";
    fn marshal_args(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&self.a.to_le_bytes());
        buf.extend_from_slice(&self.b.to_le_bytes());
        buf
    }
}

struct MulOp {
    a: u32,
    b: u32,
}

impl Marshal for MulOp {
    const HANDLER_NAME: &'static str = "flynnel.tests.marshal.mul";
    fn marshal_args(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&self.a.to_le_bytes());
        buf.extend_from_slice(&self.b.to_le_bytes());
        buf
    }
}

/// Uses the documented override to share an id with a differently
/// named legacy handler.
struct LegacyAliasOp;

impl Marshal for LegacyAliasOp {
    const HANDLER_NAME: &'static str = "flynnel.tests.marshal.alias.new_name";
    fn marshal_args(&self) -> Vec<u8> {
        Vec::new()
    }
    fn closure_id() -> u32 {
        hash_name("flynnel.tests.marshal.alias.legacy_name")
    }
}

#[test]
fn two_marshal_types_do_not_share_an_id() {
    // A collision would send one type's args to the other's handler,
    // which decodes them as its own and produces a plausible answer.
    assert_ne!(
        AddOp::closure_id(),
        MulOp::closure_id(),
        "distinct handler names must give distinct ids"
    );
}

#[test]
fn the_id_is_the_hash_of_the_declared_name() {
    // This is what lets a peer derive the id from the name alone,
    // with no shared table and no coordination.
    assert_eq!(AddOp::closure_id(), hash_name(AddOp::HANDLER_NAME));
    assert_eq!(MulOp::closure_id(), hash_name(MulOp::HANDLER_NAME));
}

#[test]
fn the_documented_override_actually_overrides() {
    // The escape hatch for sharing an id with a legacy handler whose
    // name is not the type's own. If the default won instead, the
    // legacy peer would never be reached and the dispatch would find
    // no handler at all.
    assert_eq!(
        LegacyAliasOp::closure_id(),
        hash_name("flynnel.tests.marshal.alias.legacy_name"),
        "the override picks the legacy name"
    );
    assert_ne!(
        LegacyAliasOp::closure_id(),
        hash_name(LegacyAliasOp::HANDLER_NAME),
        "and not the type's own declared name"
    );
}

#[test]
fn registering_a_handler_makes_its_id_reachable() {
    let id = AddOp::closure_id();
    let displaced = register_marshal_handler::<AddOp>(|args| {
        let a = u32::from_le_bytes(args[0..4].try_into().expect("four bytes"));
        let b = u32::from_le_bytes(args[4..8].try_into().expect("four bytes"));
        Ok((a + b).to_le_bytes().to_vec())
    });
    assert!(displaced.is_none(), "the first registration displaces nothing");
    assert!(is_registered(id), "and the id is now reachable by a peer");

    // The handler is the one registered, over the bytes the type
    // produced: the whole round trip a peer performs.
    let handler = unregister(id).expect("just registered");
    let args = AddOp { a: 20, b: 22 }.marshal_args();
    let out = handler(&args).expect("the handler runs");
    assert_eq!(
        u32::from_le_bytes(out[0..4].try_into().expect("four bytes")),
        42,
        "the peer's answer came from the originator's arguments"
    );
    assert!(!is_registered(id), "and the id is released again");
}

#[test]
fn re_registering_returns_what_it_displaced() {
    // Hot reload of a handler implementation. A registration that
    // silently added a second entry would leave the old body live and
    // the new one unreachable.
    let id = MulOp::closure_id();
    let first = register_marshal_handler::<MulOp>(|_args| Ok(vec![1u8]));
    assert!(first.is_none(), "nothing was registered under this id yet");

    let second = register_marshal_handler::<MulOp>(|_args| Ok(vec![2u8]));
    assert!(second.is_some(), "the replacement hands back the body it replaced");

    let live = unregister(id).expect("registered");
    assert_eq!(
        live(&[]).expect("runs"),
        vec![2u8],
        "and the live handler is the replacement, not the original"
    );
    assert!(!is_registered(id));
}

#[test]
fn an_overriding_type_registers_under_its_overridden_id() {
    // The override is only useful if registration follows it too.
    let legacy = hash_name("flynnel.tests.marshal.alias.legacy_name");
    let own = hash_name(LegacyAliasOp::HANDLER_NAME);
    let displaced = register_marshal_handler::<LegacyAliasOp>(|_args| Ok(vec![7u8]));
    assert!(displaced.is_none());
    assert!(is_registered(legacy), "registered under the legacy id");
    assert!(!is_registered(own), "and not under the type's own name");
    let removed = unregister(legacy).expect("registered under the legacy id");
    assert_eq!(
        removed(&[]).expect("runs"),
        vec![7u8],
        "and the body found there is the one this type registered"
    );
}

#[test]
fn marshal_args_is_the_bytes_the_peer_decodes() {
    // The serializer and the handler are written by different people
    // in different processes; the only contract between them is this
    // blob.
    let bytes = AddOp { a: 0xAABB_CCDD, b: 1 }.marshal_args();
    assert_eq!(bytes.len(), 8, "two u32 fields, little-endian");
    assert_eq!(
        u32::from_le_bytes(bytes[0..4].try_into().expect("four bytes")),
        0xAABB_CCDD
    );
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().expect("four bytes")), 1);
}

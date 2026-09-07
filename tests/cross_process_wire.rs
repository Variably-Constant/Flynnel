//! The cross-process wire and the handler registry.
//!
//! These carry a kernel's arguments and its identity between
//! processes, which is the one place in this crate where a silent
//! corruption cannot be caught downstream: the peer has no access to
//! the values the originator meant to send, so a mis-decoded argument
//! is simply the argument the kernel runs with. The registry is the
//! other half, since a `closure_id` that hashed differently in two
//! processes would dispatch to the wrong handler or to none.

use flynnel::backend::shared_mem::pass_registry::{
    hash_name, is_registered, register, registered_count, unregister,
};
use flynnel::backend::shared_mem::wire::{DecodedArg, decode_args, encode_args};
use flynnel::{BackendError, KernelArg};

#[test]
fn every_encodable_argument_survives_the_round_trip() {
    let args = [
        KernelArg::I32(i32::MIN),
        KernelArg::I32(-1),
        KernelArg::I64(i64::MAX),
        KernelArg::U32(u32::MAX),
        KernelArg::U64(u64::MAX),
        KernelArg::F32(-0.5),
        KernelArg::F64(std::f64::consts::PI),
        KernelArg::DevicePtr(0xDEAD_BEEF),
    ];
    let blob = encode_args(&args).expect("every one of these is encodable");
    let got = decode_args(&blob).expect("and the blob decodes");
    assert_eq!(
        got,
        vec![
            DecodedArg::I32(i32::MIN),
            DecodedArg::I32(-1),
            DecodedArg::I64(i64::MAX),
            DecodedArg::U32(u32::MAX),
            DecodedArg::U64(u64::MAX),
            DecodedArg::F32(-0.5),
            DecodedArg::F64(std::f64::consts::PI),
            DecodedArg::DevicePtr(0xDEAD_BEEF),
        ],
        "values and their order both survive"
    );
}

#[test]
fn floats_survive_bit_for_bit_rather_than_approximately() {
    // A wire that round-tripped through a decimal form would lose the
    // low bits and nothing downstream would notice.
    let awkward = [
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MAX,
        1.0 / 3.0,
        -0.0,
    ];
    for &v in &awkward {
        let blob = encode_args(&[KernelArg::F64(v)]).expect("encodable");
        match decode_args(&blob).expect("decodable").as_slice() {
            [DecodedArg::F64(got)] => assert_eq!(
                got.to_bits(),
                v.to_bits(),
                "{v} came back as {got}, which is not the same bits"
            ),
            other => panic!("expected one f64, got {other:?}"),
        }
    }
}

#[test]
fn a_host_slice_is_refused_rather_than_sent_as_a_pointer() {
    // The bytes live in the originator's address space, so a peer
    // handed the pointer would read its own memory. Refusing is the
    // only correct answer.
    let payload = [1u8, 2, 3, 4];
    match encode_args(&[KernelArg::HostSlice(&payload)]) {
        Err(BackendError::NotSupported) => {}
        other => panic!("a host slice cannot cross a process boundary, got {other:?}"),
    }
    // And it is refused even when it is one argument among encodable
    // ones, rather than the encodable prefix being sent alone.
    match encode_args(&[KernelArg::U32(1), KernelArg::HostSlice(&payload), KernelArg::U32(2)]) {
        Err(BackendError::NotSupported) => {}
        other => panic!("a mixed list containing a host slice must be refused, got {other:?}"),
    }
}

#[test]
fn an_empty_argument_list_round_trips_to_an_empty_one() {
    let blob = encode_args(&[]).expect("no arguments is encodable");
    assert!(blob.is_empty(), "and encodes to nothing");
    assert!(decode_args(&blob).expect("decodable").is_empty(), "and decodes back to nothing");
}

#[test]
fn a_truncated_payload_is_refused_rather_than_decoded_short() {
    // A peer that decoded what it could and ran the kernel with fewer
    // arguments would produce a plausible wrong answer.
    let blob = encode_args(&[KernelArg::U64(7), KernelArg::U64(9)]).expect("encodable");
    for cut in 1..blob.len() {
        let truncated = &blob[..cut];
        let decoded = decode_args(truncated);
        if cut == 9 {
            // Exactly one complete argument: that is a whole payload,
            // not a truncated one.
            assert_eq!(
                decoded.expect("one complete argument decodes"),
                vec![DecodedArg::U64(7)],
                "a payload cut on an argument boundary is complete"
            );
        } else {
            assert!(
                decoded.is_err(),
                "a payload cut mid-argument at {cut} bytes must be refused"
            );
        }
    }
}

#[test]
fn an_unknown_tag_is_refused_rather_than_skipped() {
    // A newer peer's argument kind must not be silently dropped by an
    // older one.
    let blob = [200u8, 0, 0, 0, 0];
    match decode_args(&blob) {
        Err(BackendError::Launch(msg)) => {
            assert!(msg.contains("200"), "the refusal names the tag it did not know: {msg}");
        }
        other => panic!("an unknown tag must be refused, got {other:?}"),
    }
}

#[test]
fn the_name_hash_is_stable_and_separates_names() {
    // Two processes derive the same id from the same kernel name
    // without coordinating, so the hash must be a pure function of the
    // name and must not collide across the names in one build.
    assert_eq!(hash_name("gemm_f32"), hash_name("gemm_f32"), "the same name is the same id");
    let names = [
        "", "a", "b", "gemm_f32", "gemm_f64", "gemm_f32 ", "GEMM_F32", "knn_search",
        "knn_search_v2",
    ];
    let mut ids: Vec<u32> = names.iter().map(|n| hash_name(n)).collect();
    let before = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), before, "no two of these names share an id");
}

#[test]
fn the_registry_reports_what_it_holds() {
    // A unique id per test run, so parallel tests in this binary do
    // not collide on the process-global table.
    let id = hash_name("flynnel_test_registry_lifecycle");
    assert!(!is_registered(id), "the id starts unregistered");

    let before = registered_count();
    let previous = register(id, |_bytes| Ok(Vec::new()));
    assert!(previous.is_none(), "the first registration displaces nothing");
    assert!(is_registered(id), "and is then visible");
    assert_eq!(registered_count(), before + 1, "and counted");

    // Re-registration is the documented hot-reload path, so it must
    // hand back the handler it replaced rather than refusing.
    let displaced = register(id, |_bytes| Ok(vec![1u8]));
    assert!(displaced.is_some(), "re-registering returns the handler it replaced");
    assert_eq!(registered_count(), before + 1, "and does not add a second entry");

    let removed = unregister(id);
    assert!(removed.is_some(), "unregistering returns the handler");
    assert!(!is_registered(id), "and it is gone");
    assert_eq!(registered_count(), before, "and the count is back");
    assert!(unregister(id).is_none(), "unregistering twice removes nothing the second time");
}

#[test]
fn a_registered_handler_is_the_one_that_runs() {
    let id = hash_name("flynnel_test_registry_identity");
    register(id, |bytes| Ok(bytes.iter().map(|b| b.wrapping_add(1)).collect()));
    let handler = unregister(id).expect("just registered");
    let out = handler(&[1u8, 2, 3]).expect("the handler runs");
    assert_eq!(out, vec![2u8, 3, 4], "and it is the body that was registered");
}

//! A slot that fails reaches the caller as an error, and one that
//! succeeds does not.
//!
//! `wait` converts a completed-but-failed slot into `Err`, so a caller
//! testing only for an error cannot read an unfilled payload as an
//! answer; `wait_status` hands back the raw word for a caller that
//! wants to tell a lost rank from a timeout itself. Both are exercised
//! here against a slot that genuinely fails on the device rather than
//! a constructed status, because a counter or a branch that has never
//! been seen to fire reads exactly like one that cannot.
//!
//! An opcode between the last built-in and `OP_USER_BASE` is unknown
//! to the kernel, which marks it `STATUS_ERR`. Requires a CUDA device.
#![cfg(feature = "gpu-peer")]

use std::time::Duration;

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, STATUS_DONE, STATUS_ERR};

mod common;

/// Unknown to the kernel: past every built-in, below `OP_USER_BASE`.
const OP_UNKNOWN: u32 = 42;

fn peer() -> GpuPeer {
    GpuPeer::init(GpuPeerConfig::default())
        .expect("a CUDA device is required for this test")
}

#[test]
fn a_failed_slot_reaches_the_caller_as_an_error() {
    let _device = common::device();
    let mut peer = peer();

    // The control first, so a run where nothing works cannot pass by
    // reporting an error for every slot.
    let ok = peer.submit(flynnel::gpu_peer::layout::OP_NOP, &[]).expect("submit nop");
    assert_eq!(
        peer.wait(ok, Duration::from_secs(5)).expect("a nop completes"),
        STATUS_DONE,
        "a slot that did its work answers Ok"
    );
    peer.reap(ok).expect("reap the nop");

    let bad = peer.submit(OP_UNKNOWN, &[]).expect("submit an unknown op");
    let err = peer
        .wait(bad, Duration::from_secs(5))
        .expect_err("an unknown opcode fails on the device and must not answer Ok");
    assert!(
        !matches!(err, flynnel::gpu_peer::GpuPeerError::Timeout),
        "the slot completed; the error must say it failed, not that it never finished: {err}"
    );
    peer.reap(bad).expect("reap the failed slot");
}

#[test]
fn wait_status_hands_back_the_failed_word_instead_of_an_error() {
    let _device = common::device();
    let mut peer = peer();

    let bad = peer.submit(OP_UNKNOWN, &[]).expect("submit an unknown op");
    assert_eq!(
        peer.wait_status(bad, Duration::from_secs(5))
            .expect("wait_status answers Ok for any slot that finished"),
        STATUS_ERR,
        "the raw word is what lets a caller tell a failed slot from a timeout"
    );
    peer.reap(bad).expect("reap the failed slot");

    let ok = peer.submit(flynnel::gpu_peer::layout::OP_NOP, &[]).expect("submit nop");
    assert_eq!(
        peer.wait_status(ok, Duration::from_secs(5)).expect("a nop completes"),
        STATUS_DONE,
        "the same call reports a good slot as done"
    );
    peer.reap(ok).expect("reap the nop");
}

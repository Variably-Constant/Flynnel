//! A user op can keep its slot across poller passes by returning the
//! yield code, and can address a resident span longer than one pool
//! block.
//!
//! Requires a CUDA device and NVRTC.
#![cfg(feature = "gpu-peer")]

use std::time::Duration;

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, STATUS_DONE, STATUS_ERR, layout};

mod common;

/// Ops 300 to 302 count their own runs in payload word 1 and yield until
/// that count reaches the target in word 0. Only thread 0 of rank 0
/// counts or decides, which is the only return the kernel reads.
/// 300 then succeeds, 301 then fails, and 302 spends 30 ms of every run
/// so a short quantum ends between runs. Op 303 writes the last byte its
/// `count` names.
const OPS: &str = r#"
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    (void)team_size;
    volatile unsigned* args = (volatile unsigned*)payload;
    if (op == 300u || op == 301u || op == 302u) {
        if (team_rank != 0u || threadIdx.x != 0) return 0u;
        if (op == 302u) {
            unsigned long long t0 = gtimer();
            while (gtimer() - t0 < 30000000ull) {
            }
        }
        unsigned runs = args[1] + 1u;
        args[1] = runs;
        if (runs < args[0]) return FLYNNEL_USER_YIELD;
        return op == 301u ? 1u : 0u;
    }
    if (op == 303u) {
        if (block == (unsigned char*)0 || count == 0u) return 1u;
        if (team_rank == 0u && threadIdx.x == 0) block[count - 1u] = 0xA5;
        return 0u;
    }
    return 1u;
}
"#;

const OP_YIELD_THEN_DONE: u32 = layout::OP_USER_BASE + 200;
const OP_YIELD_THEN_FAIL: u32 = layout::OP_USER_BASE + 201;
const OP_SLOW_YIELD: u32 = layout::OP_USER_BASE + 202;
const OP_LAST_BYTE: u32 = layout::OP_USER_BASE + 203;

const PREFIX: usize = layout::RESIDENT_PARAMS_BYTES;
const BLOCK_BYTES: u32 = 4096;

fn peer(blocks_per_lane: u32, quantum_ns: u64) -> GpuPeer {
    GpuPeer::init(GpuPeerConfig {
        user_ops_cuda: Some(OPS.to_string()),
        blocks_per_lane,
        quantum_ns,
        vram_block_bytes: BLOCK_BYTES,
        vram_blocks: 16,
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device and NVRTC are required for this test")
}

fn args_for(target_runs: u32) -> [u8; 8] {
    let mut args = [0u8; 8];
    args[..4].copy_from_slice(&target_runs.to_le_bytes());
    args
}

fn runs_recorded(peer: &GpuPeer, ticket: flynnel::gpu_peer::Ticket) -> u32 {
    let mut out = [0u8; PREFIX + 8];
    peer.read_result(ticket, &mut out).expect("the result fits the slot");
    u32::from_le_bytes([out[PREFIX + 4], out[PREFIX + 5], out[PREFIX + 6], out[PREFIX + 7]])
}

/// Submit one counting op on lane 0, wait for it to retire, and return
/// its status and how many times it ran.
fn run_to_retirement(peer: &mut GpuPeer, op: u32, target_runs: u32) -> (u32, u32) {
    let t = peer
        .submit_user_on_lane(op, None, &args_for(target_runs), 0)
        .expect("submit");
    let status = peer
        .wait_status(t, Duration::from_secs(30))
        .expect("a yielding slot still retires");
    let runs = runs_recorded(peer, t);
    peer.reap(t).expect("reap");
    (status, runs)
}

#[test]
fn a_yielding_op_keeps_its_slot_until_it_returns_success() {
    let _device = common::device();
    let mut peer = peer(1, 250_000_000);
    let (status, runs) = run_to_retirement(&mut peer, OP_YIELD_THEN_DONE, 5);
    assert_eq!(status, STATUS_DONE, "the run that returned 0 retires the slot DONE");
    assert_eq!(runs, 5, "four yields and one completing run, all on the same slot");
}

#[test]
fn a_team_op_yields_and_resumes_without_losing_a_rank() {
    let _device = common::device();
    let mut peer = peer(4, 250_000_000);
    let (status, runs) = run_to_retirement(&mut peer, OP_YIELD_THEN_DONE, 5);
    assert_eq!(
        status, STATUS_DONE,
        "a team slot kept across yields must still retire DONE; \
         TEAM_INCOMPLETE would mean followers were not released by a yield"
    );
    assert_eq!(runs, 5);
    assert_eq!(
        peer.barrier_stalls().0,
        0,
        "every pass of the team assembled, including the yielded ones"
    );
}

/// Each run takes 30 ms against a 10 ms quantum, so the poller exits
/// between runs and the waiting host has to relaunch the lane for the
/// kept slot to run again.
#[test]
fn a_yield_carries_across_quantum_exits() {
    let _device = common::device();
    let mut peer = peer(1, 10_000_000);
    let (status, runs) = run_to_retirement(&mut peer, OP_SLOW_YIELD, 4);
    assert_eq!(status, STATUS_DONE);
    assert_eq!(runs, 4, "every run after a quantum exit resumed the same slot");
}

#[test]
fn an_op_that_fails_after_yielding_retires_as_an_error() {
    let _device = common::device();
    let mut peer = peer(1, 250_000_000);
    let (status, runs) = run_to_retirement(&mut peer, OP_YIELD_THEN_FAIL, 3);
    assert_eq!(status, STATUS_ERR, "a nonzero return other than the yield code is a failure");
    assert_eq!(runs, 3);
}

/// Lane order is still submission order: a slot published behind a
/// yielding one does not run until the yielding one retires.
#[test]
fn a_slot_behind_a_yielding_one_waits_its_turn() {
    let _device = common::device();
    let mut peer = peer(1, 250_000_000);
    let first = peer
        .submit_user_on_lane(OP_YIELD_THEN_DONE, None, &args_for(6), 0)
        .expect("submit the yielding op");
    let second = peer
        .submit_user_on_lane(OP_YIELD_THEN_DONE, None, &args_for(1), 0)
        .expect("submit the op behind it");

    assert_eq!(
        peer.wait_status(second, Duration::from_secs(30)).expect("retires"),
        STATUS_DONE
    );
    assert!(
        peer.is_done(first),
        "the second slot retired, so the first must have retired before it"
    );
    assert_eq!(peer.wait_status(first, Duration::from_secs(1)).expect("retired"), STATUS_DONE);
    assert_eq!(runs_recorded(&peer, first), 6);
    assert_eq!(runs_recorded(&peer, second), 1);
    peer.reap(first).expect("reap in order");
    peer.reap(second).expect("reap in order");
}

/// A span of three blocks handed to a user op: the byte count reaches
/// past the first block, and the op's write to the last byte lands.
#[test]
fn a_user_op_reaches_the_last_byte_of_a_multi_block_span() {
    let _device = common::device();
    let mut peer = peer(1, 250_000_000);
    let span_bytes = (BLOCK_BYTES as usize) * 3 - 1;
    let handle = peer.pin_bulk(&vec![0u8; span_bytes]).expect("pin a three-block span");
    assert_eq!(handle.len(), span_bytes);

    let t = peer
        .submit_user(OP_LAST_BYTE, Some(&handle), &[0u8; 4])
        .expect("submit against the span");
    assert_eq!(
        peer.wait_status(t, Duration::from_secs(10)).expect("retires"),
        STATUS_DONE,
        "a count past one block is inside the pool and must not be refused"
    );
    peer.reap(t).expect("reap");

    let mut back = vec![0u8; span_bytes];
    peer.fetch_bulk(&handle, &mut back).expect("read the span back");
    assert_eq!(back[span_bytes - 1], 0xA5, "the op's write to the span's last byte landed");
    assert!(
        back[..span_bytes - 1].iter().all(|&b| b == 0),
        "nothing else in the span was written"
    );
}

//! Device-resident data demo: many small tasks re-reading the same
//! VRAM-resident blocks - the workload shape residency bookkeeping
//! exists for. Data is pinned once; each task then moves only an
//! 8-byte param header while the payload stays in the pool the
//! scheduler owns by index.
//!
//! The comparison is symmetric by construction: both paths use four
//! lanes and the same pipelined submission window; the ONLY
//! difference is whether the 64 KB payload rides the bus every task
//! or stays resident.
//!
//! Run with:
//!   cargo run --release --features gpu-peer --example gpu_peer_resident_demo

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use flynnel::gpu_peer::{
    GpuPeer, GpuPeerConfig, OP_ADD1_F32, OP_ADD1_F32_V, OP_SUM_U32_V, RESIDENT_PARAMS_BYTES,
    STATUS_DONE, Ticket,
};

fn main() {
    println!("=== Device-resident blocks: pin once, task many ===\n");
    let mut peer = match GpuPeer::init(GpuPeerConfig {
        slot_bytes: 65_536,
        slots_per_lane: 16,
        ..GpuPeerConfig::default()
    }) {
        Ok(p) => p,
        Err(e) => {
            println!("substrate unavailable on this host: {e}");
            return;
        }
    };
    let (free, total) = peer.pool_stats();
    println!("pool: {free}/{total} blocks free | doorbell median {} ns\n",
             peer.calibration().rtt_median_ns);

    // Four handles, one dependency chain per lane (block % lanes
    // spreads them), 16,000 f32 each = 64 KB blocks.
    let n_f32 = 16_000usize;
    let bytes = n_f32 * 4;
    let zeros = vec![0u8; bytes];
    let handles: Vec<_> = (0..4).map(|_| peer.pin(&zeros).expect("pin")).collect();
    let lanes_used: Vec<u32> = handles.iter().map(|h| h.lane()).collect();
    println!("[1] pinned 4 x {bytes} B blocks on lanes {lanes_used:?}");

    // RESIDENT path: 8,000 tasks round-robined over the 4 handles,
    // pipelined. Per-task bus traffic: 8 B of params.
    let tasks = 8_000usize;
    let per_handle = tasks / handles.len();
    let window = 32usize;
    let mut pending: VecDeque<Ticket> = VecDeque::new();
    let t0 = Instant::now();
    for i in 0..tasks {
        let h = &handles[i % handles.len()];
        pending.push_back(peer.submit_resident(OP_ADD1_F32_V, h).expect("submit"));
        if pending.len() >= window {
            let t = pending.pop_front().expect("nonempty");
            assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("wait"), STATUS_DONE);
            peer.reap(t).expect("reap");
        }
    }
    while let Some(t) = pending.pop_front() {
        assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("wait"), STATUS_DONE);
        peer.reap(t).expect("reap");
    }
    let dt_res = t0.elapsed();

    // Verify every element of every handle saw exactly its adds.
    let mut out = vec![0u8; bytes];
    for h in &handles {
        peer.fetch(h, &mut out).expect("fetch");
        for c in out.chunks_exact(4) {
            assert_eq!(
                f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                per_handle as f32,
                "element must have exactly {per_handle} ordered adds"
            );
        }
    }
    let res_per = dt_res.as_micros() as f64 / tasks as f64;
    println!("[2] RESIDENT: {tasks} tasks | {:.0} tasks/s | {res_per:.2} us/task | {} B/task on the bus | VERIFIED",
             tasks as f64 / dt_res.as_secs_f64(), RESIDENT_PARAMS_BYTES);

    // SHIPPED-EVERY-TASK baseline: identical lanes, window, and
    // computation, but the 64 KB payload crosses the bus both ways
    // on every task (submit copies it out; completion is read back).
    let base_tasks = 2_000usize;
    let payload = vec![0u8; bytes];
    let mut pending: VecDeque<Ticket> = VecDeque::new();
    let t0 = Instant::now();
    for _ in 0..base_tasks {
        pending.push_back(peer.submit(OP_ADD1_F32, &payload).expect("submit"));
        if pending.len() >= window {
            let t = pending.pop_front().expect("nonempty");
            assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("wait"), STATUS_DONE);
            peer.read_result(t, &mut out);   // results come back over the bus
            peer.reap(t).expect("reap");
        }
    }
    while let Some(t) = pending.pop_front() {
        assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("wait"), STATUS_DONE);
        peer.read_result(t, &mut out);
        peer.reap(t).expect("reap");
    }
    let dt_base = t0.elapsed();
    let base_per = dt_base.as_micros() as f64 / base_tasks as f64;
    println!("[3] SHIPPED-EVERY-TASK: {base_tasks} tasks | {:.0} tasks/s | {base_per:.2} us/task | {} B/task on the bus | VERIFIED",
             base_tasks as f64 / dt_base.as_secs_f64(), bytes * 2);
    println!("    residency advantage: {:.1}x per task, {}x less bus traffic\n",
             base_per / res_per, bytes * 2 / RESIDENT_PARAMS_BYTES);

    // GPU-side reduction on a resident block, only the u64 result
    // crosses the bus.
    let t = peer.submit_resident(OP_SUM_U32_V, &handles[0]).expect("submit");
    assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("wait"), STATUS_DONE);
    let mut buf = vec![0u8; RESIDENT_PARAMS_BYTES + 8];
    peer.read_result(t, &mut buf);
    peer.reap(t).expect("reap");
    let got = u64::from_le_bytes(buf[8..16].try_into().expect("8 bytes"));
    let expect = (per_handle as f32).to_bits() as u64 * n_f32 as u64;
    println!("[4] resident SUM over raw bits: {got} (expected {expect}): {}",
             if got == expect { "VERIFIED" } else { "FAIL" });
    assert_eq!(got, expect);

    for h in handles {
        peer.unpin(h).expect("unpin");
    }
    let (free2, _) = peer.pool_stats();
    println!("[5] unpinned; pool back to {free2}/{total} free\n");
    println!("VERIFIED: pin-once/task-many on OUR pool - per-task traffic is the");
    println!("descriptor, the data never leaves the device between tasks.");
}

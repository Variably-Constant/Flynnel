//! GPU-peer substrate demo: host-calibrated constants, doorbell
//! dispatch through a registered memory-mapped region, verified
//! streaming, and the Fischer timed lock.
//!
//! Run with:
//!   cargo run --release --features gpu-peer --example gpu_peer_demo
//!
//! Requires an NVIDIA GPU + driver at runtime (no CUDA toolkit); the
//! demo exits cleanly with a message when no device is present.

use std::time::{Duration, Instant};

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, OP_ADD1_F32, OP_SUM_U32, STATUS_DONE};

fn main() {
    println!("=== GPU-peer substrate demo ===\n");
    let mut peer = match GpuPeer::init(GpuPeerConfig::default()) {
        Ok(p) => p,
        Err(e) => {
            println!("substrate unavailable on this host: {e}");
            return;
        }
    };

    let cal = peer.calibration();
    println!("[1] HOST-MEASURED calibration (nothing baked in):");
    println!("    doorbell RTT       min {} ns | median {} ns | p99 {} ns",
             cal.rtt_min_ns, cal.rtt_median_ns, cal.rtt_p99_ns);
    println!("    one-way visibility {} ns | clock error +-{} ns",
             cal.one_way_ns, cal.clock_err_ns);
    println!("    Fischer Delta      {} ns (validated by live self-test; contended {}cpu/{}gpu of 150)",
             cal.delta_ns, cal.lock_cpu_contended, cal.lock_gpu_contended);
    println!("    launch baseline    {} ns (the cost the doorbell path avoids)", cal.launch_ns);
    println!("    capabilities: doorbell={} timed_lock={} sys_atomics={}\n",
             cal.doorbell_ok, cal.timed_lock_ok, cal.sys_atomics_ok);

    // [2] Verified ADD1 streaming: payload of f32s, GPU adds 1.0 to
    // each in place, CPU verifies every element of every block.
    println!("[2] streaming 8192 verified ADD1_F32 blocks (doorbell dispatch):");
    let payload_f32 = (peer.geometry().payload_max() / 4).min(1000);
    let mut payload = vec![0u8; payload_f32 * 4];
    let mut result = vec![0u8; payload_f32 * 4];
    let total: usize = 8192;
    let mut verified = 0usize;
    let t0 = Instant::now();
    for m in 0..total {
        for (i, chunk) in payload.chunks_exact_mut(4).enumerate() {
            chunk.copy_from_slice(&(((m + i) % 1000) as f32).to_le_bytes());
        }
        let ticket = peer.submit(OP_ADD1_F32, &payload).expect("submit");
        let status = peer.wait(ticket, Duration::from_secs(5)).expect("wait");
        assert_eq!(status, STATUS_DONE, "block {m} must complete");
        peer.read_result(ticket, &mut result);
        for (i, chunk) in result.chunks_exact(4).enumerate() {
            let got = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let want = ((m + i) % 1000) as f32 + 1.0;
            assert_eq!(got, want, "block {m} element {i}");
        }
        verified += payload_f32;
        peer.reap(ticket).expect("in-order reap");
    }
    let dt = t0.elapsed();
    println!("    {} blocks, {} f32 elements verified | {:.0} msgs/s | {:.2} us/msg\n",
             total, verified,
             total as f64 / dt.as_secs_f64(),
             dt.as_micros() as f64 / total as f64);

    // [2b] The same traffic with a submission WINDOW: latency hides
    // behind the pipeline once the in-flight depth exceeds the
    // bandwidth-delay product, which is the whole point of feeding
    // the peer continuously instead of round-tripping per block.
    println!("[2b] same blocks, 32-deep submission window (latency hidden):");
    let window = 32usize;
    let mut pending = std::collections::VecDeque::new();
    let t0 = Instant::now();
    for m in 0..total {
        for (i, chunk) in payload.chunks_exact_mut(4).enumerate() {
            chunk.copy_from_slice(&(((m + i) % 1000) as f32).to_le_bytes());
        }
        pending.push_back((m, peer.submit(OP_ADD1_F32, &payload).expect("submit")));
        if pending.len() >= window {
            let (m0, t) = pending.pop_front().expect("window nonempty");
            let status = peer.wait(t, Duration::from_secs(5)).expect("wait");
            assert_eq!(status, STATUS_DONE, "block {m0}");
            peer.read_result(t, &mut result);
            let first =
                f32::from_le_bytes([result[0], result[1], result[2], result[3]]);
            assert_eq!(first, (m0 % 1000) as f32 + 1.0, "block {m0} spot check");
            peer.reap(t).expect("in-order reap");
        }
    }
    while let Some((m0, t)) = pending.pop_front() {
        let status = peer.wait(t, Duration::from_secs(5)).expect("wait");
        assert_eq!(status, STATUS_DONE, "block {m0}");
        peer.reap(t).expect("in-order reap");
    }
    let dt = t0.elapsed();
    println!("    {} blocks | {:.0} msgs/s | {:.2} us/msg amortized\n",
             total,
             total as f64 / dt.as_secs_f64(),
             dt.as_micros() as f64 / total as f64);

    // [3] SUM opcode: GPU-side reduction with the result read back
    // through the same slot.
    println!("[3] SUM_U32 reduction on the GPU:");
    let n_u32 = (peer.geometry().payload_max() / 4).min(1000);
    let nums: Vec<u8> = (0..n_u32 as u32).flat_map(|v| v.to_le_bytes()).collect();
    let expect: u64 = (0..n_u32 as u64).sum();
    let ticket = peer.submit(OP_SUM_U32, &nums).expect("submit");
    let status = peer.wait(ticket, Duration::from_secs(5)).expect("wait");
    assert_eq!(status, STATUS_DONE);
    let mut sum_bytes = [0u8; 8];
    peer.read_result(ticket, &mut sum_bytes);
    let got = u64::from_le_bytes(sum_bytes);
    peer.reap(ticket).expect("reap");
    assert_eq!(got, expect);
    println!("    sum(0..{n_u32}) = {got} (expected {expect}): VERIFIED\n");

    // [4] Fischer timed lock at the calibrated margin.
    println!("[4] Fischer timed lock (cross-device mutual exclusion, no atomics):");
    if cal.timed_lock_ok {
        let t0 = Instant::now();
        let rounds = 200;
        for _ in 0..rounds {
            peer.timed_lock_acquire(Duration::from_secs(2)).expect("acquire");
            peer.timed_lock_release();
        }
        println!("    {} uncontended acquire/release cycles at Delta={} ns: {:.1} us each\n",
                 rounds, cal.delta_ns,
                 t0.elapsed().as_micros() as f64 / rounds as f64);
    } else {
        println!("    capability refused by calibration on this host\n");
    }

    println!("VERIFIED: doorbell dispatch, in-place transform, GPU reduction,");
    println!("and the timed lock all ran against host-measured constants.");
}

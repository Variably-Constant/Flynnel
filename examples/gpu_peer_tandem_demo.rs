//! MIMT tandem demo: CPU workers and the GPU peer drain ONE workload
//! together, with the split share learned from measured throughput.
//!
//! Run with:
//!   cargo run --release --features gpu-peer --example gpu_peer_tandem_demo
//!
//! Each round transforms the same buffer of f32 blocks (+1.0 per
//! element): the CPU half through `for_each_chunk` on the
//! work-stealing pool, the GPU half through doorbell-dispatched peer
//! lanes with a pipelined submission window. The share drifts by
//! measured per-side throughput (the same EWMA-style rule the hybrid
//! split uses), and the FULL buffer is verified every round.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, OP_ADD1_F32, STATUS_DONE};
use flynnel::{JobPlan, for_each_chunk};

fn main() {
    println!("=== MIMT tandem: CPU pool + GPU peer on one workload ===\n");
    let mut peer = match GpuPeer::init(GpuPeerConfig::default()) {
        Ok(p) => p,
        Err(e) => {
            println!("substrate unavailable on this host: {e}");
            return;
        }
    };
    let cal = peer.calibration();
    println!("calibrated: doorbell median {} ns, Delta {} ns, launch {} ns\n",
             cal.rtt_median_ns, cal.delta_ns, cal.launch_ns);

    // Workload: BLOCKS blocks of f32s; each round adds 1.0 everywhere.
    let block_f32 = (peer.geometry().payload_max() / 4).min(1000);
    let blocks = 4096usize;
    let mut data = vec![0f32; blocks * block_f32];
    let window = 32usize;

    // Learned CPU share (per-mille), seeded even and driven by the
    // measured per-side throughputs each round.
    let mut cpu_share_pm: u64 = 500;
    println!("round | cpu share | cpu blocks | gpu blocks | cpu ms | gpu ms | verified");
    for round in 0..8u32 {
        let cpu_blocks = ((blocks as u64 * cpu_share_pm) / 1000) as usize;
        let cpu_blocks = cpu_blocks.clamp(1, blocks - 1);
        let (cpu_half, gpu_half) = data.split_at_mut(cpu_blocks * block_f32);

        // GPU half first: submissions are asynchronous, so the CPU
        // half executes while the peer drains its lanes.
        let t_gpu = Instant::now();
        let mut pending: VecDeque<(usize, flynnel::gpu_peer::Ticket)> = VecDeque::new();
        let mut payload = vec![0u8; block_f32 * 4];
        let gpu_blocks = blocks - cpu_blocks;
        let mut gpu_done = 0usize;
        let reap_into = |peer: &mut GpuPeer,
                             pending: &mut VecDeque<(usize, flynnel::gpu_peer::Ticket)>,
                             gpu_half: &mut [f32],
                             done: &mut usize| {
            let (b, t) = pending.pop_front().expect("nonempty");
            let status = peer.wait(t, Duration::from_secs(10)).expect("gpu wait");
            assert_eq!(status, STATUS_DONE, "gpu block {b}");
            let dst = &mut gpu_half[b * block_f32..(b + 1) * block_f32];
            let mut bytes = vec![0u8; block_f32 * 4];
            peer.read_result(t, &mut bytes);
            for (i, c) in bytes.chunks_exact(4).enumerate() {
                dst[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            peer.reap(t).expect("reap");
            *done += 1;
        };
        for b in 0..gpu_blocks {
            let src = &gpu_half[b * block_f32..(b + 1) * block_f32];
            for (i, v) in src.iter().enumerate() {
                payload[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            let t = peer.submit(OP_ADD1_F32, &payload).expect("submit");
            pending.push_back((b, t));
            if pending.len() >= window {
                reap_into(&mut peer, &mut pending, gpu_half, &mut gpu_done);
            }
        }

        // CPU half on the work-stealing pool while the GPU drains.
        let t_cpu = Instant::now();
        let plan = JobPlan::new(6, cpu_half.len() as u32);
        for_each_chunk(&plan, cpu_half, |chunk| {
            for x in chunk.iter_mut() {
                *x += 1.0;
            }
        });
        let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1e3;

        while !pending.is_empty() {
            reap_into(&mut peer, &mut pending, gpu_half, &mut gpu_done);
        }
        let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        assert_eq!(gpu_done, gpu_blocks);

        // Verify the WHOLE buffer advanced by exactly one round.
        let expect = (round + 1) as f32;
        let bad = data.iter().filter(|&&v| v != expect).count();
        println!("{round:>5} | {:>8}% | {cpu_blocks:>10} | {gpu_blocks:>10} | {cpu_ms:>6.1} | {gpu_ms:>6.1} | {}",
                 cpu_share_pm / 10,
                 if bad == 0 { "OK" } else { "FAIL" });
        assert_eq!(bad, 0, "round {round}: {bad} elements wrong");

        // Learned share update from measured per-block throughput
        // (same drift rule as the hybrid split: slower side sheds).
        let cpu_ns_per_block = (cpu_ms * 1e6 / cpu_blocks as f64).max(1.0);
        let gpu_ns_per_block = (gpu_ms * 1e6 / gpu_blocks as f64).max(1.0);
        let ideal_cpu_pm =
            (gpu_ns_per_block * 1000.0 / (cpu_ns_per_block + gpu_ns_per_block)) as u64;
        // EWMA alpha = 1/2 for a fast-converging demo.
        cpu_share_pm = ((cpu_share_pm + ideal_cpu_pm) / 2).clamp(50, 950);
    }
    println!("\nVERIFIED: 8 rounds, full-buffer checks green; the share drifted to");
    println!("the measured throughput balance between the pool and the peer.");
}

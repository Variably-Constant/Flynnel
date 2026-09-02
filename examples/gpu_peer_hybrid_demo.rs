//! One model for execution side AND data residence.
//!
//! Each round, the call site's learned placement model picks CPU or
//! device for the same logical step over a [`MirrorBuf`]; any
//! transfer a flip requires is executed inside the timed section, so
//! the model prices residence and compute together and converges to
//! the side that wins END TO END on this host.
//!
//! Run with:
//!   cargo run --release --features gpu-peer --example gpu_peer_hybrid_demo

use flynnel::JobPlan;
use flynnel::gpu_peer::hybrid::{ADD1_F32_RESIDENT, MirrorBuf, add1_f32_cpu, hybrid_auto_resident};
use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig};

fn main() {
    println!("=== One-model placement: residence + execution ===\n");
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

    let n_f32 = 16_000usize;
    let data: Vec<u8> = (0..n_f32).flat_map(|_| 0f32.to_le_bytes()).collect();
    let mut mirror = MirrorBuf::new(data);
    let plan = JobPlan::new(6, n_f32 as u32);

    let rounds = 48usize;
    let (mut n_cpu, mut n_dev, mut n_race) = (0usize, 0usize, 0usize);
    let mut seq = String::new();
    for r in 0..rounds {
        let p = hybrid_auto_resident(&plan, &mut peer, &mut mirror, ADD1_F32_RESIDENT, add1_f32_cpu)
            .expect("round");
        match p {
            flynnel::Placement::Cpu => { n_cpu += 1; seq.push('C'); }
            flynnel::Placement::Backend => { n_dev += 1; seq.push('D'); }
            flynnel::Placement::Race => { n_race += 1; seq.push('R'); }
        }
        if (r + 1) % 12 == 0 {
            println!("  rounds {:>2}-{:>2}: {} (mirror state {:?})",
                     r.saturating_sub(10), r + 1, &seq[seq.len() - 12..], mirror.state());
        }
    }
    println!("\nplacements: {n_cpu} cpu | {n_dev} device | {n_race} race");
    println!("sequence:   {seq}");

    // THE correctness property: exactly one application per round,
    // through every placement flip and residence transfer.
    let out = mirror.host_bytes(&mut peer).expect("sync host");
    let bad = out
        .chunks_exact(4)
        .filter(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) != rounds as f32)
        .count();
    println!("\nall {n_f32} elements == {rounds}.0 after mixed placements: {}",
             if bad == 0 { "VERIFIED" } else { "FAIL" });
    assert_eq!(bad, 0);
    mirror.evict(&mut peer).expect("evict");
    let converged = if n_cpu > n_dev { "CPU" } else { "device" };
    println!("model converged toward {converged} on this host (transfers priced in).");
}

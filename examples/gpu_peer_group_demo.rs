//! Peer-group demo: N GPU peers under one handle namespace, with
//! cross-peer migration through the host bridge.
//!
//! Run with:
//!   cargo run --release --features gpu-peer --example gpu_peer_group_demo
//!
//! On a single-GPU host the group members share one device (both
//! configs name ordinal 0): every group code path - unified
//! placement, routing, migration - is the multi-device path; only
//! the silicon is shared. On a multi-GPU host the same code spreads
//! across ordinals.

use std::time::Duration;

use flynnel::gpu_peer::{GpuPeerConfig, OP_ADD1_F32_V, PeerGroup, STATUS_DONE};

fn main() {
    println!("=== Peer group: unified namespace + migration ===\n");
    let mk = |ordinal| GpuPeerConfig {
        device_ordinal: ordinal,
        vram_blocks: 64,
        ..GpuPeerConfig::default()
    };
    let mut group = match PeerGroup::init(vec![mk(0), mk(0)]) {
        Ok(g) => g,
        Err(e) => {
            println!("group unavailable on this host: {e}");
            return;
        }
    };
    println!("[1] group of {} peers; per-peer calibrations:", group.len());
    for (i, c) in group.calibrations().iter().enumerate() {
        println!("    peer {i}: doorbell {} ns | Delta {} ns | timed_lock={}",
                 c.rtt_median_ns, c.delta_ns, c.timed_lock_ok);
    }

    // Unified pin - the group places by pool pressure.
    let n_f32 = 1000usize;
    let data: Vec<u8> = (0..n_f32).flat_map(|_| 5.0f32.to_le_bytes()).collect();
    let h = group.pin(&data).expect("pin");
    println!("\n[2] unified pin landed on peer {} (lane {})", h.peer, h.handle.lane());

    // Work on the owning peer.
    let (p, t) = group.submit_resident(OP_ADD1_F32_V, &h).expect("submit");
    assert_eq!(group.wait_reap(p, t, Duration::from_secs(5)).expect("wait"), STATUS_DONE);

    // Migrate to the OTHER peer through the host bridge; data must
    // survive byte-exact (5.0 + 1 add = 6.0).
    let to = (h.peer + 1) % group.len();
    let h2 = group.migrate(h, to).expect("migrate");
    println!("[3] migrated to peer {} (lane {})", h2.peer, h2.handle.lane());
    assert_eq!(h2.peer, to);

    // Keep computing on the destination peer, then verify the whole
    // history: 5.0 +1 (peer A) +1 (peer B) = 7.0 for every element.
    let (p2, t2) = group.submit_resident(OP_ADD1_F32_V, &h2).expect("submit");
    assert_eq!(group.wait_reap(p2, t2, Duration::from_secs(5)).expect("wait"), STATUS_DONE);
    let mut out = vec![0u8; n_f32 * 4];
    group.fetch(&h2, &mut out).expect("fetch");
    let bad = out
        .chunks_exact(4)
        .filter(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) != 7.0)
        .count();
    println!("[4] all {n_f32} elements == 7.0 after cross-peer compute chain: {}",
             if bad == 0 { "VERIFIED" } else { "FAIL" });
    assert_eq!(bad, 0);
    group.unpin(h2).expect("unpin");

    println!("\nVERIFIED: one handle namespace over N peers - placement, routing,");
    println!("and host-bridge migration with byte-exact data survival.");
}

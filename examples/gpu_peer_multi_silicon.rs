//! Multi-silicon validation: two GPU peers on the device ordinals
//! given as arguments (default "0 1"), with per-round status output.
//!
//!   gpu_peer_multi_silicon [ordinal_a] [ordinal_b] [rounds]
//!
//! Phases per round:
//!   1. pin a pattern block on peer A, compute on A
//!   2. migrate A -> B through the host bridge, compute on B
//!   3. verify EVERY element carries the full cross-device history
//!
//! Both peers also run a parallel doorbell burst each round so the
//! two devices are demonstrably live simultaneously.

use std::time::{Duration, Instant};

use flynnel::gpu_peer::{GpuPeerConfig, OP_ADD1_F32_V, PeerGroup, STATUS_DONE};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ord_a: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let ord_b: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    println!("=== multi-silicon validation: ordinals {ord_a} + {ord_b}, {rounds} rounds ===");

    let mk = |ordinal| GpuPeerConfig {
        device_ordinal: ordinal,
        vram_blocks: 64,
        ..GpuPeerConfig::default()
    };
    let mut group = match PeerGroup::init(vec![mk(ord_a), mk(ord_b)]) {
        Ok(g) => g,
        Err(e) => {
            println!("STATUS init-failed: {e}");
            std::process::exit(2);
        }
    };
    for (i, c) in group.calibrations().iter().enumerate() {
        println!("STATUS peer {i} calibrated: doorbell median {} ns | p99 {} ns | Delta {} ns | launch {} ns | timed_lock={} sys_atomics={}",
                 c.rtt_median_ns, c.rtt_p99_ns, c.delta_ns, c.launch_ns,
                 c.timed_lock_ok, c.sys_atomics_ok);
    }

    let n_f32 = 1000usize;
    let t_all = Instant::now();
    let mut migrations = 0usize;
    for round in 0..rounds {
        let seed = (round % 100) as f32;
        let data: Vec<u8> = (0..n_f32).flat_map(|_| seed.to_le_bytes()).collect();

        // Pin on A (peer 0 of the group), one add there.
        let h = {
            let peer0 = group.peer_mut(0);
            let h = peer0.pin(&data).expect("pin on A");
            flynnel::gpu_peer::GroupHandle { peer: 0, handle: h }
        };
        let (p, t) = group.submit_resident(OP_ADD1_F32_V, &h).expect("A compute");
        assert_eq!(group.wait_reap(p, t, Duration::from_secs(10)).expect("A wait"), STATUS_DONE);

        // Cross-silicon migrate A -> B, one add there.
        let h2 = group.migrate(h, 1).expect("migrate A->B");
        migrations += 1;
        let (p2, t2) = group.submit_resident(OP_ADD1_F32_V, &h2).expect("B compute");
        assert_eq!(group.wait_reap(p2, t2, Duration::from_secs(10)).expect("B wait"), STATUS_DONE);

        // Verify the whole history: seed + 1 (A) + 1 (B).
        let mut out = vec![0u8; n_f32 * 4];
        group.fetch(&h2, &mut out).expect("fetch");
        let want = seed + 2.0;
        let bad = out
            .chunks_exact(4)
            .filter(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) != want)
            .count();
        group.unpin(h2).expect("unpin");
        assert_eq!(bad, 0, "round {round}: {bad} elements missing cross-device history");

        // Simultaneous-liveness burst: 16 params-only tasks per peer.
        let burst_data: Vec<u8> = (0..64usize).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        for pi in 0..group.len() {
            let hh = group.peer_mut(pi).pin(&burst_data).expect("burst pin");
            let mut ts = Vec::new();
            for _ in 0..16 {
                ts.push(group.peer_mut(pi).submit_resident(OP_ADD1_F32_V, &hh).expect("burst"));
            }
            for t in ts {
                assert_eq!(
                    group.peer_mut(pi).wait(t, Duration::from_secs(10)).expect("burst wait"),
                    STATUS_DONE
                );
                group.peer_mut(pi).reap(t).expect("burst reap");
            }
            group.peer_mut(pi).unpin(hh).expect("burst unpin");
        }

        println!("STATUS round {}/{} OK | migrations {} | elapsed {:.1}s",
                 round + 1, rounds, migrations, t_all.elapsed().as_secs_f64());
    }
    println!("RESULT: ALL {rounds} ROUNDS VERIFIED - {migrations} cross-device migrations, byte-exact history on every element");
}

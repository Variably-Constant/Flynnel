//! `collect_indexed` and the resident pool at the scale a scoring
//! step uses: 65,536 light items per call must complete promptly, and
//! a pin / unpin cycle over multi-block spans must not hang or
//! exhaust the pool. Each phase runs under a watchdog so a hang fails
//! with the phase's name instead of stalling the suite.

use std::sync::mpsc;
use std::time::Duration;

use flynnel::sched::JobPlan;
use flynnel::sched::par_iter::collect_indexed;

/// Runs `f` on a worker thread and fails if it has not finished
/// within `limit`. On a timeout the pool state is printed twice, a
/// second apart, so a spinning worker (counters moving) can be told
/// from a blocked one.
fn watchdog<F: FnOnce() + Send + 'static>(phase: &str, limit: Duration, f: F) {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        f();
        tx.send(()).expect("watchdog receiver alive");
    });
    match rx.recv_timeout(limit) {
        Ok(()) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let arena = flynnel::sched::arena::global_local_arena();
            eprintln!("{phase}: pool state at timeout:\n{}", arena.debug_snapshot());
            std::thread::sleep(Duration::from_secs(1));
            eprintln!("{phase}: pool state 1 s later:\n{}", arena.debug_snapshot());
            panic!("{phase}: no completion within {limit:?}");
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{phase}: worker panicked"),
    }
}

#[test]
fn collect_indexed_65536_light_items_completes() {
    let site = flynnel::caller_site();
    for round in 0..6 {
        let plan = JobPlan::new(0, 65_536).with_site(site).apply_site_class();
        eprintln!(
            "round {round}: learned_class={:?} use_smt={} oversub={:?} est_ns={:?} tier={:?}",
            site.get().learned_class(),
            plan.use_smt,
            plan.oversubscription_log2,
            plan.estimated_per_item_ns,
            flynnel::sched::pick_tier(&plan, flynnel::numa_topology()),
        );
        watchdog(&format!("collect_indexed round {round}"), Duration::from_secs(20), move || {
            let n = 65_536usize;
            let plan = JobPlan::new(0, n as u32).with_site(site);
            let out: Vec<Vec<f64>> = collect_indexed(&plan, n, 1, |i| {
                let mut v = vec![0f64; 256];
                for (j, x) in v.iter_mut().enumerate() {
                    *x = (i * 256 + j) as f64 * 0.5;
                }
                v
            });
            assert_eq!(out.len(), n);
            assert_eq!(out[n - 1][255], ((n - 1) * 256 + 255) as f64 * 0.5);
        });
    }
}

/// Runs `rounds` collect_indexed dispatches of 65,536 light items
/// under `plan`, each under a 20 s watchdog, and reports the slowest
/// round.
fn rounds_under(plan_name: &str, rounds: usize, make_plan: fn() -> JobPlan) {
    let mut worst = Duration::ZERO;
    for round in 0..rounds {
        let t0 = std::time::Instant::now();
        watchdog(&format!("{plan_name} round {round}"), Duration::from_secs(20), move || {
            let n = 65_536usize;
            let plan = make_plan();
            let out: Vec<Vec<f64>> = collect_indexed(&plan, n, 1, |i| {
                let mut v = vec![0f64; 256];
                for (j, x) in v.iter_mut().enumerate() {
                    *x = (i * 256 + j) as f64 * 0.5;
                }
                v
            });
            assert_eq!(out.len(), n);
        });
        worst = worst.max(t0.elapsed());
    }
    eprintln!("{plan_name}: {rounds} rounds, slowest {worst:?}");
}

#[test]
fn collect_indexed_65536_smt_off_200_rounds() {
    rounds_under("PortBound (SMT off)", 200, || {
        JobPlan::set_profile(0, 65_536, flynnel::DispatchProfile::PortBound)
    });
}

#[test]
fn collect_indexed_65536_smt_on_200_rounds() {
    rounds_under("LatencyBound (SMT on)", 200, || {
        JobPlan::set_profile(0, 65_536, flynnel::DispatchProfile::LatencyBound)
    });
}

#[cfg(feature = "gpu-peer")]
#[test]
fn pin_bulk_span_cycle_does_not_hang_or_leak() {
    use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig};

    watchdog("pin/unpin span cycle", Duration::from_secs(60), || {
        let mut peer = GpuPeer::init(GpuPeerConfig {
            slot_bytes: 64 * 1024,
            slots_per_lane: 4,
            vram_block_bytes: 16 * 1024 * 1024,
            vram_blocks: 32,
            ..GpuPeerConfig::default()
        })
        .expect("a CUDA device is required for this test");
        let (free0, total) = peer.pool_stats();
        assert_eq!(free0 as u32, total);
        // Three 8-block spans out of 32 blocks, released out of order,
        // then pinned again: succeeds only if spans are found by
        // index and released in full.
        for cycle in 0..4 {
            let data = vec![cycle as u8; 128 * 1024 * 1024];
            let a = peer.pin_bulk(&data).expect("pin a");
            let b = peer.pin_bulk(&data).expect("pin b");
            let c = peer.pin_bulk(&data).expect("pin c");
            let mut back = vec![0u8; 4096];
            peer.fetch_bulk(&c, &mut back).expect("fetch c head");
            assert!(back.iter().all(|&x| x == cycle as u8));
            peer.unpin(b).expect("unpin b");
            peer.unpin(a).expect("unpin a");
            peer.unpin(c).expect("unpin c");
            let (free, _) = peer.pool_stats();
            assert_eq!(free as u32, total, "cycle {cycle} leaked pool blocks");
        }
    });
}

//! Submit latency on an idle lane while another lane stays resident.
//!
//! The poller relaunches only once every block of the previous launch
//! has exited. A block with nothing to do parks after `idle_exit_ns`;
//! a block still running an op stays until that op returns, which a
//! long op carries past the end of its quantum. A lane whose block has
//! parked therefore has no server, and its next submit waits for the
//! last working block to finish before anything can relaunch.
//!
//! The wait is bounded by the straggler's work rather than by the
//! quantum: a feeder spinning longer than `quantum_ns` holds the
//! parked lane for the whole spin.
//!
//! Two arms, identical but for the feeder:
//!   feeder present - one lane is held resident by a long device-side
//!                    spin while the timed lane sits idle past its
//!                    park threshold, then takes a timed submit
//!   feeder absent  - the same timed submits with no lane held
//!
//! A latency near the quantum in the first arm and not the second is
//! the gate. Clean in both arms refutes the account.
//!
//! Run with:
//!   cargo run --release --features gpu-peer --example gpu_peer_lane_stall
//!
//! Requires the NVRTC runtime library for the user-op composition;
//! exits with a message otherwise.

use std::time::{Duration, Instant};

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, STATUS_DONE};

/// op 100: spin on the device until `globaltimer` has advanced by the
/// u32 nanoseconds at payload+8. A zero argument returns at once, so
/// the same op serves the feeder and the timed submit.
const USER_OPS: &str = r#"
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    (void)block; (void)count; (void)team_rank; (void)team_size;
    if (op != 100u) return 1u;
    unsigned spin_ns = *(volatile unsigned*)payload;
    if (spin_ns == 0u) return 0u;
    unsigned long long t0, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - t0 < (unsigned long long)spin_ns);
    return 0u;
}
"#;

/// Lane held resident by the feeder.
const FEEDER_LANE: u32 = 0;
/// Lane left idle long enough to park, then timed.
const TIMED_LANE: u32 = 3;
/// Device-side spin per feeder op when the caller names none. Long
/// enough to outlast the idle threshold of every other lane by a wide
/// margin, and short enough to land inside one quantum.
const DEFAULT_FEEDER_SPIN_MS: u32 = 150;
/// Host-side gap between arming the feeder and the timed submit. Above
/// the 2 ms park threshold, far below the 250 ms quantum.
const IDLE_GAP: Duration = Duration::from_millis(10);
/// Timed submits per arm.
const ROUNDS: usize = 30;

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn report(label: &str, mut ms: Vec<f64>) {
    ms.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    println!(
        "{label:<16} n={:<4} min {:>8.3}  p50 {:>8.3}  p90 {:>8.3}  p99 {:>8.3}  max {:>8.3}",
        ms.len(),
        ms[0],
        pct(&ms, 0.50),
        pct(&ms, 0.90),
        pct(&ms, 0.99),
        ms[ms.len() - 1]
    );
}

/// One arm. With `feeder`, every round holds [`FEEDER_LANE`] resident
/// for `spin_ns` across the idle gap and the timed submit.
fn arm(peer: &mut GpuPeer, feeder: bool, spin_ns: u32) -> Vec<f64> {
    let mut out = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let held = if feeder {
            Some(
                peer.submit_user_on_lane(100, None, &spin_ns.to_le_bytes(), FEEDER_LANE)
                    .expect("feeder submit"),
            )
        } else {
            None
        };

        std::thread::sleep(IDLE_GAP);

        let t0 = Instant::now();
        let timed = peer
            .submit_user_on_lane(100, None, &0u32.to_le_bytes(), TIMED_LANE)
            .expect("timed submit");
        let status = peer.wait(timed, Duration::from_secs(5)).expect("timed wait");
        let elapsed = t0.elapsed();
        assert_eq!(status, STATUS_DONE, "timed op status");
        peer.reap(timed).expect("reap timed");
        out.push(elapsed.as_secs_f64() * 1e3);

        if let Some(ft) = held {
            assert_eq!(
                peer.wait(ft, Duration::from_secs(5)).expect("feeder wait"),
                STATUS_DONE
            );
            peer.reap(ft).expect("reap feeder");
        }
    }
    out
}

fn main() {
    // Optional first argument: feeder spin in milliseconds. A value
    // above the quantum shows whether the wait is bounded by the
    // straggler's residency or by the quantum that ends it.
    let spin_ms: u32 = match std::env::args().nth(1) {
        None => DEFAULT_FEEDER_SPIN_MS,
        Some(a) => match a.parse() {
            Ok(ms) => ms,
            Err(e) => {
                eprintln!("feeder spin argument {a:?} is not a millisecond count: {e}");
                std::process::exit(2);
            }
        },
    };
    let spin_ns = spin_ms.saturating_mul(1_000_000);

    // Optional second argument: blocks per lane, so the same arm runs
    // against a lane served by a block team.
    let blocks_per_lane: u32 = match std::env::args().nth(2) {
        None => 1,
        Some(a) => match a.parse() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("blocks-per-lane argument {a:?} is not a count: {e}");
                std::process::exit(2);
            }
        },
    };

    println!("=== Submit latency on a parked lane, with and without a resident peer ===\n");
    let cfg = GpuPeerConfig {
        user_ops_cuda: Some(USER_OPS.to_string()),
        lanes: 4,
        blocks_per_lane,
        ..GpuPeerConfig::default()
    };
    println!(
        "lanes {}  blocks_per_lane {}  quantum {} ms  idle_exit {} ms",
        cfg.lanes,
        cfg.blocks_per_lane,
        cfg.quantum_ns / 1_000_000,
        cfg.idle_exit_ns / 1_000_000
    );
    println!(
        "feeder lane {FEEDER_LANE} spins {spin_ms} ms per op; timed lane {TIMED_LANE} submits after a {} ms gap\n",
        IDLE_GAP.as_millis()
    );

    let mut peer = match GpuPeer::init(cfg) {
        Ok(p) => p,
        Err(e) => {
            println!("substrate unavailable on this host: {e}");
            return;
        }
    };

    // A warm submit so the first timed round meets an already-launched
    // poller rather than a cold one.
    let w = peer
        .submit_user_on_lane(100, None, &0u32.to_le_bytes(), TIMED_LANE)
        .expect("warm submit");
    peer.wait(w, Duration::from_secs(5)).expect("warm wait");
    peer.reap(w).expect("reap warm");

    let control = arm(&mut peer, false, spin_ns);
    report("feeder absent", control.clone());
    let held = arm(&mut peer, true, spin_ns);
    report("feeder present", held.clone());

    let mut c = control;
    let mut h = held;
    c.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    h.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let c99 = pct(&c, 0.99);
    let h50 = pct(&h, 0.50);
    let h99 = pct(&h, 0.99);
    // A parked lane waited on the resident peer for the rest of that
    // peer's work, so the held arm sits near the feeder's spin less the
    // gap already spent, and the control does not.
    let predicted = f64::from(spin_ms) - IDLE_GAP.as_secs_f64() * 1e3;
    let separation = h50 / c99.max(f64::MIN_POSITIVE);
    println!();
    println!("p99 with feeder {h99:.3} ms against {c99:.3} ms without");
    println!(
        "held p50 {h50:.3} ms against a predicted {predicted:.3} ms, separation {separation:.0}x"
    );
    let near_predicted = (h50 - predicted).abs() <= predicted * 0.25;
    if separation >= 5.0 && near_predicted {
        println!("the parked lane waits out the resident peer for the whole of its work");
    } else if separation >= 5.0 {
        println!("the parked lane waits on the resident peer, but not for the whole of its work");
    } else {
        println!("the parked lane is served while the peer is resident");
    }
}

//! Wave calibration measures a device's wave costs, keeps them on the peer,
//! and stores them so the next peer on the device starts with them.
//!
//! Requires a CUDA device and NVRTC. The binary holds one test, which points
//! the calibration table at a directory of its own before any peer starts,
//! so it neither reads nor overwrites the host's table.
#![cfg(feature = "gpu-peer")]

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig};

mod common;

/// A user op that refuses every opcode; calibration runs Flynnel's own op,
/// which composing any user source brings into the module.
const OPS: &str = r#"
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    (void)op; (void)block; (void)count; (void)payload; (void)team_rank; (void)team_size;
    return 1u;
}
"#;

fn peer(with_user_ops: bool) -> GpuPeer {
    GpuPeer::init(GpuPeerConfig {
        user_ops_cuda: if with_user_ops { Some(OPS.to_string()) } else { None },
        blocks_per_lane: 4,
        vram_block_bytes: 65_536,
        vram_blocks: 512,
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device and NVRTC are required for this test")
}

#[test]
fn wave_calibration_measures_keeps_and_stores_the_costs() {
    let _device = common::device();
    let dir = std::env::temp_dir().join(format!("flynnel-wave-calibration-{}", std::process::id()));
    // SAFETY: this binary runs one test, and the variable is set before any
    // peer, and so any thread of this crate, reads the environment.
    unsafe { std::env::set_var("FLYNNEL_CALIBRATION_DIR", &dir) };

    let mut without = peer(false);
    assert!(
        without.calibrate_waves().is_err(),
        "without user ops the calibration op is not in the module"
    );
    drop(without);

    let mut first = peer(true);
    assert_eq!(first.wave_costs(), None, "a fresh table holds no wave costs");
    let costs = first.calibrate_waves().expect("calibrate");
    println!("wave costs at team size {}: {costs:?}", first.team_size());
    assert_eq!(costs.width, first.team_size());
    assert!(costs.fixed_ns > 0, "a slice round trip costs time: {costs:?}");
    assert!(costs.generation_ns > 0, "a generation takes time: {costs:?}");
    assert!(
        costs.barrier_ns < 50_000_000,
        "a barrier longer than the calibration deadline would have failed the run: {costs:?}"
    );
    assert_eq!(first.wave_costs(), Some(costs), "the peer keeps what it measured");
    drop(first);

    if cfg!(feature = "persisted-calibration") {
        let second = peer(true);
        assert_eq!(
            second.wave_costs(),
            Some(costs),
            "the next peer on the device starts with the stored costs"
        );
        drop(second);
    }

    match std::fs::remove_dir_all(&dir) {
        Ok(()) => {}
        Err(err) => eprintln!("wave calibration test left {} behind: {err}", dir.display()),
    }
}

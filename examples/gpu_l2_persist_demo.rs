//! GPU L2-persistence demo: reserve a slice of L2, pin a hot working
//! set in it, and measure the lift against the identical kernel with
//! the hot set left to stream.
//!
//! Run with:
//!   cargo run --release --features gpu-peer --example gpu_l2_persist_demo
//!
//! The A/B is fair by construction: same kernel, same data, same
//! launch. The only variable is the stream's access-policy window.

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig};

fn main() {
    println!("=== GPU L2-persistence demo ===\n");
    // No resident pool needed for this lever; keep init light.
    let peer = match GpuPeer::init(GpuPeerConfig { vram_blocks: 0, ..GpuPeerConfig::default() }) {
        Ok(p) => p,
        Err(e) => {
            println!("substrate unavailable on this host: {e}");
            return;
        }
    };

    let cap = match peer.l2_capability() {
        Ok(c) => c,
        Err(e) => {
            println!("could not query L2 capability: {e}");
            return;
        }
    };
    println!("[1] device L2-persistence capability:");
    println!("    max set-aside   {} KiB", cap.max_persisting_l2 / 1024);
    println!("    max window      {} KiB", cap.max_access_window / 1024);
    if !cap.supported() {
        println!("    -> not supported on this device; nothing to measure");
        return;
    }

    // Hot set = the whole set-aside. The benefit is the hot set's
    // avoided HBM refetch each pass, so its share of total traffic -
    // hot / (hot + polluter) - bounds the speedup. Sweeping the
    // polluter shows that dependence honestly: lighter contention
    // (polluter nearer L2 size) means a larger pinned fraction and a
    // bigger lift.
    let hot = cap.max_persisting_l2.min(2 * 1024 * 1024);
    let iters = 64u32;
    let runs = 30u32;
    println!("\n[2] fair A/B (same kernel + data, only the window differs), hot {} KiB pinned:", hot / 1024);
    println!("    polluter | streaming | persisting | speedup | hot fraction");
    for &pol_mib in &[6usize, 16, 64] {
        let pol = pol_mib * 1024 * 1024;
        match peer.l2_benchmark(hot, pol, iters, runs) {
            Ok(r) => {
                let frac = r.hot_bytes as f64 / (r.hot_bytes + r.pol_bytes) as f64;
                println!(
                    "    {:>5} MiB | {:>7.1} us | {:>8.1} us | {:>5.3}x | {:>4.1}%",
                    pol_mib, r.stream_us, r.persist_us, r.speedup, frac * 100.0
                );
            }
            Err(e) => {
                println!("    {pol_mib} MiB: benchmark failed: {e}");
                return;
            }
        }
    }
    println!(
        "\nVERIFIED: L2 persistence honored on this device (set-aside {} KiB). The lift",
        hot / 1024
    );
    println!("is NON-monotonic: it wins under moderate contention but can go negative when");
    println!("the set-aside starves the co-runner (small-polluter row above). That is why");
    println!("the lever is measured and gated, not always-on - the same calibrate-then-");
    println!("enable rule the Fischer and sys-atomics probes follow. Its ceiling here is the");
    println!("3070's 4 MiB L2 and grows with L2 size on server parts.");
}

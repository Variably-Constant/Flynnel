//! Idle workers spin `yield_now` before parking. For a bursty-idle
//! workload - a short burst, then idle - that spin is wasted CPU (the
//! `sched_yield` a flamegraph flags). This demo measures the idle-yield
//! count across three configurations of the spin window:
//!   A. the tuned default (500) - long spin, tuned for throughput,
//!   B. the explicit short-window lever (the CPU analog of pausing
//!      the GPU poller),
//!   C. the adaptive controller, which shrinks the window on its own
//!      when it sees workers parking instead of being rescued.
//!
//! Run with:
//!   cargo run --release --example adaptive_spin_demo

use std::time::Duration;

use flynnel::{
    JobPlan, for_each_chunk, reset_spin_stats, set_spin_adaptive, set_spin_window, spin_window,
    total_idle_yields,
};

fn bursty_workload(bursts: u32, buf: &mut [u32]) {
    for _ in 0..bursts {
        // A short burst of real work, then an idle gap where the pool
        // workers have nothing to do.
        let plan = JobPlan::new(6, buf.len() as u32);
        for_each_chunk(&plan, buf, |c| {
            for x in c.iter_mut() {
                *x = x.wrapping_add(1);
            }
        });
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn main() {
    println!("=== Adaptive idle-spin window ===\n");
    let mut buf = vec![0u32; 65_536];
    let bursts = 120u32;

    // Warm the pool so the first phase is not paying spawn cost.
    bursty_workload(4, &mut buf);

    // A. tuned default window (500).
    set_spin_window(500);
    reset_spin_stats();
    bursty_workload(bursts, &mut buf);
    let yields_default = total_idle_yields();
    println!("[A] default window 500  : {yields_default:>10} idle yields over {bursts} bursts");

    // B. explicit short window - the lever for a known bursty-idle,
    // latency-insensitive workload.
    set_spin_window(8);
    reset_spin_stats();
    bursty_workload(bursts, &mut buf);
    let yields_short = total_idle_yields();
    println!("[B] forced window 8     : {yields_short:>10} idle yields  ({:.1}x fewer)",
             yields_default as f64 / yields_short.max(1) as f64);

    // C. adaptive: start from the default, let the controller observe
    // the parks and shrink on its own.
    set_spin_window(500);
    set_spin_adaptive(true);
    reset_spin_stats();
    bursty_workload(bursts, &mut buf);
    let yields_adaptive = total_idle_yields();
    let final_window = spin_window();
    println!("[C] adaptive            : {yields_adaptive:>10} idle yields  ({:.1}x fewer), window -> {final_window}",
             yields_default as f64 / yields_adaptive.max(1) as f64);

    // Correctness: every burst incremented every element once; the
    // buffer must equal the total burst count (workload unaffected by
    // the spin policy).
    let total_bursts = 4 + bursts * 3;
    let bad = buf.iter().filter(|&&v| v != total_bursts).count();
    println!("\ncorrectness: all {} elements == {total_bursts} bursts: {}",
             buf.len(), if bad == 0 { "OK" } else { "FAIL" });
    assert_eq!(bad, 0);
    assert!(yields_short < yields_default, "short window must yield less");
    assert!(yields_adaptive < yields_default, "adaptive must yield less than default");
    assert!(final_window < 500, "adaptive must have shrunk the window");

    println!("\nVERIFIED: the short and adaptive windows reclaim the idle-spin CPU a");
    println!("bursty workload wastes, while the tuned default stays available for");
    println!("throughput work that a long spin keeps hot.");
}

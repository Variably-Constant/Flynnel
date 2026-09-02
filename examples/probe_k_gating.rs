//! Print the K_gating calibration result on this host.
//!
//! Run with:
//! ```sh
//! cargo run --example probe_k_gating --release
//! ```
//!
//! Reports per-primitive per-iter timings and the winner the
//! scheduler's `KGating::Auto.resolved()` will return on this
//! host class.

fn main() {
    let r = flynnel::sched::k_gating::calibrate_k_gating_verbose();
    println!("K_gating calibration on this host:");
    println!("  CounterOnly (Chase-Lev / Fcl)  per-iter: {:>8} ns", r.counter_only_ns);
    println!("  PerSlot     (KHL / KHPD)       per-iter: {:>8} ns", r.per_slot_ns);
    println!("  Winner: {:?}", r.winner);
    let ratio = if r.per_slot_ns < r.counter_only_ns {
        r.counter_only_ns as f64 / r.per_slot_ns as f64
    } else {
        r.per_slot_ns as f64 / r.counter_only_ns as f64
    };
    println!("  Ratio (winner / loser): {ratio:.2}x faster");
}

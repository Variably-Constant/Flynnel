//! Cold-dispatch measurement for the PROBE_SMALL_MIN_N gate sweep.
//!
//! Measures one cold `for_each_chunk_indexed_min_leaf` dispatch
//! (min_leaf = 1) under a hint-less SMT-off profile
//! (`set_profile(0, n, PortBound)`: use_smt off, cost estimate
//! non-explicit), the exact region where the small-batch probe gate
//! decides between probing and inline serial execution. Each
//! invocation of this binary measures exactly one cell in a fresh
//! process so per-call-site learning from one cell cannot leak into
//! another; the sweep driver runs it once per (n, weight, repeat).
//!
//! Env: `SWEEP_N` (item count), `SWEEP_ITERS` (splitmix64 steps per
//! item; ~1.5 ns per step). Prints `cell n=<n> iters=<iters>
//! wall_ns=<w> serial_ns=<s>`.

use std::time::Instant;

use flynnel::DispatchProfile;
use flynnel::sched::JobPlan;
use flynnel::sched::par_iter::for_each_chunk_indexed_min_leaf;

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn spin_work(seed: u64, iters: u64) -> u64 {
    let mut acc = seed;
    for _ in 0..iters {
        acc = splitmix64(acc);
    }
    std::hint::black_box(acc)
}

fn env_u64(name: &str) -> u64 {
    std::env::var(name)
        .unwrap_or_else(|e| panic!("{name} must be set: {e}"))
        .parse()
        .unwrap_or_else(|e| panic!("{name} must be a u64: {e}"))
}

fn main() {
    let n = env_u64("SWEEP_N") as usize;
    let iters = env_u64("SWEEP_ITERS");

    // Spin the worker pool up so pool startup is not in the cell.
    let warm = JobPlan::new(0, 4096);
    let mut v = vec![0u64; 4096];
    for_each_chunk_indexed_min_leaf(&warm, &mut v, 1, |start, slots| {
        for (i, s) in slots.iter_mut().enumerate() {
            *s = spin_work((start + i) as u64, 500);
        }
    });
    std::hint::black_box(&v);

    let t0 = Instant::now();
    let mut serial_out = vec![0u64; n];
    for (i, s) in serial_out.iter_mut().enumerate() {
        *s = spin_work(i as u64, iters);
    }
    let serial_ns = t0.elapsed().as_nanos() as u64;
    std::hint::black_box(&serial_out);

    let plan = JobPlan::set_profile(0, n as u32, DispatchProfile::PortBound);
    let mut out = vec![0u64; n];
    let t1 = Instant::now();
    for_each_chunk_indexed_min_leaf(&plan, &mut out, 1, |start, slots| {
        for (i, s) in slots.iter_mut().enumerate() {
            *s = spin_work((start + i) as u64, iters);
        }
    });
    let wall_ns = t1.elapsed().as_nanos() as u64;
    assert_eq!(out, serial_out, "dispatch result diverged from serial");

    println!("cell n={n} iters={iters} wall_ns={wall_ns} serial_ns={serial_ns}");
}

//! End-to-end runnable demo of the plan-free `flynnel::flat` surface.
//!
//! Shows that `flynnel::flat::{join, par_for_each_mut,
//! par_for_each_chunk_mut}` give plain-function-call ergonomics
//! without requiring a `JobPlan`.
//!
//! Run with:
//!   cargo run --release --example flat_api_demo

use std::time::Instant;

fn main() {
    println!("=== Flynnel flat (plan-free) API demo ===\n");

    println!("[1] flat::join - plain-function-call shape, no JobPlan");
    let (left, right) = flynnel::flat::join(
        || (0..1_000_u32).sum::<u32>(),
        || (1_000..2_000_u32).sum::<u32>(),
    );
    let expected = (0..2_000_u32).sum::<u32>();
    println!("    left  = {left}");
    println!("    right = {right}");
    println!("    sum   = {} (expected {expected})", left + right);
    assert_eq!(left + right, expected);
    println!("    VERIFIED.\n");

    println!("[2] flat::par_for_each_mut - per-element closure over &mut [T]");
    let n = 100_000usize;
    let mut data: Vec<u32> = (0..n as u32).collect();
    let t0 = Instant::now();
    flynnel::flat::par_for_each_mut(&mut data, |x| {
        *x = x.wrapping_mul(3).wrapping_add(7);
    });
    let elapsed = t0.elapsed();
    println!("    n             = {n}");
    println!("    elapsed       = {} us", elapsed.as_micros());
    println!("    first 5 vals  = {:?}", &data[..5]);
    println!("    last 5 vals   = {:?}", &data[n - 5..]);
    for (i, &val) in data.iter().enumerate() {
        assert_eq!(val, (i as u32).wrapping_mul(3).wrapping_add(7));
    }
    println!("    VERIFIED: every element transformed correctly.\n");

    println!("[3] flat::par_for_each_chunk_mut - slice-chunk closure (SIMD-friendly)");
    let mut data: Vec<f64> = (1..=n as u64).map(|i| i as f64).collect();
    let t0 = Instant::now();
    flynnel::flat::par_for_each_chunk_mut(&mut data, |slice| {
        for x in slice.iter_mut() {
            *x = x.sqrt();
        }
    });
    let elapsed = t0.elapsed();
    println!("    n             = {n}");
    println!("    elapsed       = {} us", elapsed.as_micros());
    println!("    data[0]       = {} (expected {})", data[0], 1.0_f64.sqrt());
    println!("    data[100]     = {} (expected {})", data[100], 101.0_f64.sqrt());
    println!("    data[n-1]     = {} (expected {})", data[n - 1], (n as f64).sqrt());
    assert!((data[0] - 1.0_f64.sqrt()).abs() < 1e-12);
    assert!((data[n - 1] - (n as f64).sqrt()).abs() < 1e-6);
    println!("    VERIFIED: every element is sqrt(i+1) within tolerance.\n");

    println!("=== All three flat-API entry points VERIFIED end-to-end. ===");
    println!(
        "    The three plan-free entry points:\n      \
        flynnel::flat::join(a, b)                              // two-way fork-join\n      \
        flynnel::flat::par_for_each_mut(&mut v, op)            // per-element data-parallel\n      \
        flynnel::flat::par_for_each_chunk_mut(&mut v, op)      // per-chunk data-parallel"
    );
}

//! What a thread-clock read costs on this host.
//!
//! The on-core bracket in the sampled leaf path is two clock pairs per
//! sampled leaf, and a clock pair is one thread-clock read plus one
//! elapsed-count read. The per-leaf instrumentation budget the sampled
//! path is built against is a few nanoseconds amortized, so whether
//! that bracket fits depends on a number nobody here has measured.
//!
//! Neither half is the same call on every target, and this times what
//! the platform actually runs:
//!
//! - thread clock: `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` on Linux
//!   and FreeBSD, `QueryThreadCycleTime` on Windows. The vDSO fast path
//!   serves the realtime and monotonic clocks and refuses the
//!   per-thread CPU clock, so the first form enters the kernel and
//!   walks the task's own CPU-time accounting.
//! - elapsed count: `_rdtsc` on Windows x86_64, `Instant::elapsed`
//!   against a fixed origin elsewhere.
//!
//! The loop baseline is subtracted from both, so what is reported is
//! the call and not the loop around it.
//!
//! ```sh
//! clock_cost <calls_per_batch> <batches> <leaf_sample_stride>
//! ```

use std::env;
use std::hint::black_box;
use std::time::Instant;

use flynnel::sched::occupancy::thread_on_core_ticks;

fn arg<T>(n: usize, default: T) -> T
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match env::args().nth(n) {
        None => default,
        Some(text) => match text.parse() {
            Ok(value) => value,
            Err(err) => {
                eprintln!("argument {n} is not a valid value: {text:?} ({err})");
                std::process::exit(2);
            }
        },
    }
}

/// The elapsed half of a clock pair, as this platform reads it.
#[cfg(all(windows, target_arch = "x86_64"))]
fn elapsed_count(_origin: Instant) -> u64 {
    // SAFETY: `_rdtsc` reads a counter register and touches no memory.
    unsafe { core::arch::x86_64::_rdtsc() }
}

#[cfg(not(all(windows, target_arch = "x86_64")))]
fn elapsed_count(origin: Instant) -> u64 {
    origin.elapsed().as_nanos() as u64
}

#[cfg(all(windows, target_arch = "x86_64"))]
const ELAPSED_CALL: &str = "rdtsc";

#[cfg(not(all(windows, target_arch = "x86_64")))]
const ELAPSED_CALL: &str = "instant_elapsed";

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no timing is NaN"));
    v[v.len() / 2]
}

/// Nanoseconds per iteration of `body`, median over `batches` runs of
/// `calls` iterations.
///
/// The median rather than the minimum: the minimum of several runs is
/// whichever run's noise fell furthest in one direction, and what a
/// dispatch pays is the typical call.
fn per_call<F: FnMut() -> u64>(calls: u64, batches: usize, mut body: F) -> f64 {
    let mut samples = Vec::with_capacity(batches);
    for _ in 0..batches {
        let t0 = Instant::now();
        let mut sink = 0u64;
        for _ in 0..calls {
            sink = sink.wrapping_add(body());
        }
        let elapsed = t0.elapsed().as_nanos() as f64;
        black_box(sink);
        samples.push(elapsed / calls as f64);
    }
    median(&mut samples)
}

fn main() {
    let calls: u64 = arg(1, 200_000);
    let batches: usize = arg(2, 9);
    let stride: u64 = arg(3, 8);

    if calls == 0 || batches == 0 || stride == 0 {
        eprintln!("calls, batches and stride must each be at least one");
        std::process::exit(2);
    }

    // A platform with no thread clock has nothing to time here, and a
    // figure for it would be the cost of reporting an absence.
    if thread_on_core_ticks().ticks().is_none() {
        eprintln!(
            "this platform reports no thread clock, so the bracket it would pay for \
             does not exist and there is nothing to time"
        );
        std::process::exit(2);
    }

    let origin = Instant::now();

    // The loop, a counter increment and a black_box, with no clock in
    // it. Subtracted from the two figures below.
    let mut counter = 0u64;
    let baseline = per_call(calls, batches, || {
        counter = counter.wrapping_add(1);
        black_box(counter)
    });

    let thread_ns = per_call(calls, batches, || {
        thread_on_core_ticks().ticks().unwrap_or(0)
    });
    let elapsed_ns = per_call(calls, batches, || elapsed_count(origin));

    let thread_net = (thread_ns - baseline).max(0.0);
    let elapsed_net = (elapsed_ns - baseline).max(0.0);
    let pair = thread_net + elapsed_net;
    let bracket = 2.0 * pair;

    println!("clock loop_baseline        ns_per_call {baseline:.1}");
    println!("clock thread_on_core_ticks ns_per_call {thread_net:.1}");
    println!("clock {:<20} ns_per_call {elapsed_net:.1}", ELAPSED_CALL);
    println!("clock pair                 ns_per_call {pair:.1}");
    println!(
        "leaf_bracket two_pairs_ns {bracket:.1} stride {stride} amortized_per_leaf_ns {:.1}",
        bracket / stride as f64
    );
}

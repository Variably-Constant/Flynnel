//! How a dispatch from outside the pool fares when the process holds
//! more runnable threads than the host has cores.
//!
//! An outside caller runs none of its join's work. It hands the join to
//! a worker and waits until every leaf has finished, and a leaf whose
//! worker the OS has preempted cannot be stolen, so the call can wait out
//! a scheduler time slice. This times the same dispatch from the main
//! thread with and without spinner threads in this process, beside the
//! same work run serially on the main thread, and reports the whole
//! distribution, because the stall shows as a tail rather than in the
//! median.
//!
//! The phases alternate within one process, spinners off then on, for
//! several rounds, and the parallel and serial calls alternate within
//! each phase, so the box's drift lands on both arms alike.
//!
//! ```text
//! cargo run --release --example oversubscribed_caller -- [spinners] [rounds] [calls] [items] [flops]
//! ```
//!
//! Defaults: three quarters of the logical processors as spinners, 4
//! rounds, 200 calls per arm per phase, 1,000,000 items, 2 square roots
//! per item.
//!
//! With `FLYNNEL_TRACE=1` in the environment it times nothing of the
//! above. It runs `calls` parallel calls with the spinners on, prints
//! one `CALL index nanoseconds` line each and the trace clock's rate,
//! and dumps every thread's trace buffer to stderr as `TRACE,` rows, so
//! a slow call can be split into its pickup and its join's longest leaf.

use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use flynnel::sched::trace;
use flynnel::{JobPlan, for_each_chunk};

/// `flops` chained square roots on one item.
#[inline(never)]
fn work(x: &mut f64, flops: u32) {
    let mut v = *x;
    for _ in 0..flops {
        v = v.sqrt() * 1.000_000_1_f64;
    }
    *x = v;
}

/// The `index`th argument parsed as `T`, `default` when absent; an
/// argument that does not parse ends the run with its error.
fn argument<T: FromStr>(args: &[String], index: usize, default: T, name: &str) -> T
where
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        None => default,
        Some(text) => match text.parse() {
            Ok(value) => value,
            Err(e) => {
                eprintln!("argument {name} is {text:?}: {e}");
                std::process::exit(2);
            }
        },
    }
}

/// Threads that spin until told to stop, the in-process load.
struct Spinners {
    stop: Arc<AtomicBool>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl Spinners {
    fn start(count: usize) -> Spinners {
        let stop = Arc::new(AtomicBool::new(false));
        let handles = (0..count)
            .map(|_| {
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut acc = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        acc = acc.wrapping_mul(6364136223846793005).wrapping_add(1);
                        std::hint::black_box(acc);
                    }
                })
            })
            .collect();
        Spinners { stop, handles }
    }

    fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        for handle in self.handles {
            if let Err(payload) = handle.join() {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

/// Median, 90th and 99th percentiles, and the largest, of samples in
/// nanoseconds, returned in milliseconds.
fn summary(samples: &mut [u64]) -> (f64, f64, f64, f64) {
    samples.sort_unstable();
    let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize] as f64 / 1e6;
    (at(0.5), at(0.9), at(0.99), samples[samples.len() - 1] as f64 / 1e6)
}

/// How many samples fall in each of the millisecond buckets the stall
/// shows up in.
fn buckets(samples: &[u64]) -> String {
    let edges_ms = [1.0, 2.0, 5.0, 10.0, 20.0];
    let mut counts = [0usize; 6];
    for &s in samples {
        let ms = s as f64 / 1e6;
        let slot = edges_ms.iter().position(|&e| ms < e).unwrap_or(edges_ms.len());
        counts[slot] += 1;
    }
    format!(
        "<1ms {} 1-2 {} 2-5 {} 5-10 {} 10-20 {} >=20 {}",
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]
    )
}

/// Ticks of the trace clock per nanosecond. The trace stamps events with
/// the time-stamp counter on x86_64, measured here against the monotonic
/// clock over a fifth of a second, and with nanoseconds elsewhere.
fn trace_clock_per_ns() -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: RDTSC has no preconditions on x86_64.
        let c0 = unsafe { core::arch::x86_64::_rdtsc() };
        let t0 = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(200));
        // SAFETY: as above.
        let c1 = unsafe { core::arch::x86_64::_rdtsc() };
        let elapsed = t0.elapsed().as_nanos() as f64;
        (c1.wrapping_sub(c0)) as f64 / elapsed
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1.0
    }
}

/// The traced run: parallel calls under the spinners only, then every
/// thread's buffer dumped. Workers dump at the top of their next loop
/// pass, so wide dispatches are driven until each has, as
/// `examples/trace_dispatch.rs` does.
fn traced(spinners: usize, calls: usize, plan: &JobPlan, data: &mut [f64], flops: u32) {
    println!("TRACE_CLOCK_PER_NS {:.6}", trace_clock_per_ns());
    let load = Spinners::start(spinners);
    std::thread::sleep(std::time::Duration::from_millis(200));
    trace::reset_current_thread();
    for i in 0..calls {
        let t0 = Instant::now();
        for_each_chunk(plan, &mut *data, |s| s.iter_mut().for_each(|x| work(x, flops)));
        println!("CALL {i} {}", t0.elapsed().as_nanos());
    }
    load.stop();
    trace::dump_to_stderr("caller");
    trace::request_worker_flush();
    let mut wide = vec![1.5f64; 1 << 20];
    let wide_plan = JobPlan::new(6, wide.len() as u32).with_estimated_per_item_ns(50).with_smt();
    let expected = std::thread::available_parallelism().map_or(1, |n| n.get()) as u64;
    let deadline = Instant::now() + std::time::Duration::from_secs(30);
    let done = loop {
        for_each_chunk(&wide_plan, &mut wide, |s| s.iter_mut().for_each(|x| work(x, 16)));
        std::thread::sleep(std::time::Duration::from_millis(5));
        let done = trace::worker_flushes_done();
        if done >= expected || Instant::now() > deadline {
            break done;
        }
    };
    std::thread::sleep(std::time::Duration::from_millis(500));
    println!("WORKER_DUMPS {done} of {expected}");
    trace::clear_worker_flush_request();
    std::hint::black_box(&wide);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    let spinners: usize = argument(&args, 1, logical * 3 / 4, "spinners");
    let rounds: usize = argument(&args, 2, 4, "rounds");
    let calls: usize = argument(&args, 3, 200, "calls");
    let items: usize = argument(&args, 4, 1_000_000, "items");
    let flops: u32 = argument(&args, 5, 2, "flops");

    let workers = flynnel::sched::arena::global_local_arena().local_worker_count();
    println!(
        "logical processors {logical}, pool workers {workers}, spinners {spinners}, \
         rounds {rounds}, calls {calls} per arm per phase, items {items}, flops {flops}"
    );

    let mut data = vec![1.5f64; items];
    let plan = JobPlan::new(10, items.min(u32::MAX as usize) as u32);
    // Warm the pool and the clock before anything is recorded.
    for _ in 0..50 {
        for_each_chunk(&plan, &mut data, |s| s.iter_mut().for_each(|x| work(x, flops)));
    }
    if trace::is_enabled() {
        traced(spinners, calls, &plan, &mut data, flops);
        return;
    }

    // Per phase: parallel samples and serial samples, in nanoseconds.
    let mut quiet_parallel = Vec::with_capacity(rounds * calls);
    let mut quiet_serial = Vec::with_capacity(rounds * calls);
    let mut loaded_parallel = Vec::with_capacity(rounds * calls);
    let mut loaded_serial = Vec::with_capacity(rounds * calls);

    for round in 0..rounds {
        for loaded in [false, true] {
            let load = if loaded { Some(Spinners::start(spinners)) } else { None };
            // Let the spinners take their cores before the first call.
            std::thread::sleep(std::time::Duration::from_millis(200));
            let (parallel, serial) = if loaded {
                (&mut loaded_parallel, &mut loaded_serial)
            } else {
                (&mut quiet_parallel, &mut quiet_serial)
            };
            for call in 0..calls {
                // The arm order alternates by call so neither always
                // follows the other.
                let parallel_first = (call + round) % 2 == 0;
                for arm in 0..2 {
                    let run_parallel = (arm == 0) == parallel_first;
                    let t0 = Instant::now();
                    if run_parallel {
                        for_each_chunk(&plan, &mut data, |s| s.iter_mut().for_each(|x| work(x, flops)));
                    } else {
                        data.iter_mut().for_each(|x| work(x, flops));
                    }
                    let ns = t0.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
                    if run_parallel {
                        parallel.push(ns);
                    } else {
                        serial.push(ns);
                    }
                }
            }
            if let Some(load) = load {
                load.stop();
            }
        }
        println!("round {} of {rounds} done", round + 1);
    }

    for (label, parallel, serial) in [
        ("quiet", &mut quiet_parallel, &mut quiet_serial),
        ("loaded", &mut loaded_parallel, &mut loaded_serial),
    ] {
        let spread = buckets(parallel);
        let (pm, p90, p99, pmax) = summary(parallel);
        let (sm, s90, s99, smax) = summary(serial);
        println!(
            "{label:<6} parallel median {pm:8.3} p90 {p90:8.3} p99 {p99:8.3} max {pmax:8.3} ms  [{spread}]"
        );
        println!("{label:<6} serial   median {sm:8.3} p90 {s90:8.3} p99 {s99:8.3} max {smax:8.3} ms");
        println!(
            "{label:<6} parallel over serial: median {:.3}, p99 {:.3}",
            pm / sm,
            p99 / s99
        );
    }
    println!(
        "IDLE_EXITS parks={} spin_rescues={} self_rescues={} sleepless_backoffs={}",
        flynnel::total_park_events(),
        flynnel::total_rescue_events(),
        flynnel::total_self_rescues(),
        flynnel::total_sleepless_backoffs()
    );
    std::hint::black_box(&data);
}

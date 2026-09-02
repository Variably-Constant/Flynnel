//! Cold-workload bench harness.
//!
//! Standalone binary (not criterion). Measures real one-off dispatch
//! latency by forcing primaries to PARK between samples. Criterion's
//! iter-back-to-back pattern keeps the worker pool HOT, which masks
//! the wake-on-push + steal latency that dominates real one-off
//! workloads (a CLI tool processing one input, a service handling
//! one request, a notebook cell running an NMFD batch).
//!
//! Shapes are chosen to span the realistic axis:
//!   - small N + heavy per-item (NMFD-like)
//!   - balanced medium
//!   - deep-recursion bisect
//!   - streaming many-light-items
//!
//! For each shape, runs serial / rayon / flynnel in turn with a
//! mandatory COOLDOWN sleep between each sample so the JEC sleep
//! coordinator parks workers between calls.
//!
//! Output is a markdown-style table with median + p10/p90 wall
//! clock per contender and the flynnel/rayon ratio.

use std::time::{Duration, Instant};

use flynnel::JobPlan;
use flynnel::sched::par_iter::for_each_chunk;
use rayon::prelude::*;

/// sqrt(x+1) chain. Compiler can not elide because the result
/// feeds back as the input of the next iteration. Roughly the
/// target nanoseconds on x86_64 Zen3 ~4 GHz (~12ns per sqrt).
#[inline(never)]
fn sqrt_chain(seed: f64, iters: u32) -> f64 {
    let mut x = seed;
    for _ in 0..iters {
        x = (x + 1.0).sqrt();
    }
    x
}

/// Workload shape. n_items items, each costing roughly
/// sqrt_iters * 12 ns of FP-dependency-chained work.
struct Shape {
    label: &'static str,
    n_items: usize,
    sqrt_iters: u32,
    /// Description shown in the table.
    desc: &'static str,
}

const SHAPES: &[Shape] = &[
    Shape {
        label: "nmfd_5x100ms",
        n_items: 5,
        sqrt_iters: 8_333_333,
        desc: "NMFD-like: 5 items x 100ms",
    },
    Shape {
        label: "shallow_4x10ms",
        n_items: 4,
        sqrt_iters: 833_333,
        desc: "4 items x 10ms",
    },
    Shape {
        label: "shallow_8x10ms",
        n_items: 8,
        sqrt_iters: 833_333,
        desc: "8 items x 10ms",
    },
    Shape {
        label: "shallow_16x10ms",
        n_items: 16,
        sqrt_iters: 833_333,
        desc: "16 items x 10ms",
    },
    Shape {
        label: "medium_32x1ms",
        n_items: 32,
        sqrt_iters: 83_333,
        desc: "32 items x 1ms",
    },
    Shape {
        label: "medium_128x500us",
        n_items: 128,
        sqrt_iters: 41_666,
        desc: "128 items x 500us",
    },
    Shape {
        label: "deep_1024x100us",
        n_items: 1024,
        sqrt_iters: 8_333,
        desc: "1024 items x 100us",
    },
    Shape {
        label: "stream_16k_10us",
        n_items: 16_384,
        sqrt_iters: 833,
        desc: "16384 items x 10us",
    },
];

/// Samples per (shape, contender) cell. Median + p10/p90 reported.
const SAMPLES: usize = 10;
/// Mandatory sleep between samples so the JEC sleep coordinator
/// parks workers. Without this, criterion-style hot-loop bias creeps
/// back in. 100ms is well above the ROUNDS_UNTIL_SLEEPING threshold.
const COOLDOWN: Duration = Duration::from_millis(100);

fn run_serial(items: &mut [f64], iters: u32) {
    for x in items.iter_mut() {
        *x = sqrt_chain(*x, iters);
    }
}

fn run_rayon(items: &mut [f64], iters: u32) {
    items.par_iter_mut().for_each(|x| {
        *x = sqrt_chain(*x, iters);
    });
}

fn run_flynnel(items: &mut [f64], iters: u32) {
    let plan = JobPlan::new(6, items.len() as u32);
    for_each_chunk(&plan, items, |slice: &mut [f64]| {
        for x in slice.iter_mut() {
            *x = sqrt_chain(*x, iters);
        }
    });
}

fn run_flynnel_hinted(items: &mut [f64], iters: u32) {
    let plan = JobPlan::new(6, items.len() as u32)
        .with_estimated_per_item_ns(((iters as u64) * 12).min(u32::MAX as u64) as u32);
    for_each_chunk(&plan, items, |slice: &mut [f64]| {
        for x in slice.iter_mut() {
            *x = sqrt_chain(*x, iters);
        }
    });
}

/// Run `run` on a fresh items slice SAMPLES times, sleeping
/// COOLDOWN between samples so primaries park (the cold-cache
/// scenario we are trying to measure). Returns wall-clock durations
/// sorted ascending.
fn measure(label: &str, shape: &Shape, run: impl Fn(&mut [f64], u32)) -> Vec<Duration> {
    let mut samples = Vec::with_capacity(SAMPLES);
    for s in 0..SAMPLES {
        std::thread::sleep(COOLDOWN);
        let mut items: Vec<f64> = (0..shape.n_items).map(|i| (i as f64) + 1.0).collect();
        let t0 = Instant::now();
        run(&mut items, shape.sqrt_iters);
        let elapsed = t0.elapsed();
        std::hint::black_box(items);
        samples.push(elapsed);
        // Stream progress so a long run does not look stuck.
        if s == 0 {
            eprintln!("  [{label} first sample = {:?}]", elapsed);
        }
    }
    samples.sort();
    samples
}

fn pct(samples: &[Duration], p: usize) -> Duration {
    let idx = (samples.len() * p / 100).min(samples.len() - 1);
    samples[idx]
}

fn median(samples: &[Duration]) -> Duration {
    samples[samples.len() / 2]
}

fn fmt_dur(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{} ns", ns)
    } else if ns < 1_000_000 {
        format!("{:.2} us", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2} ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", d.as_secs_f64())
    }
}

fn main() {
    println!("# Cold-workload bench results");
    println!();
    println!("Force-parking between samples so the JEC sleep coordinator parks ");
    println!("primaries -- matches one-off dispatch latency that real workloads ");
    println!("(NMFD batches, CLI tools, services) see. Each cell = median of ");
    println!("{SAMPLES} samples with {}ms cooldown.", COOLDOWN.as_millis());
    println!();
    println!("Lower is faster. ratio = flynnel_hinted / rayon (< 1.0 means ");
    println!("flynnel wins).");
    println!();
    println!("| Shape                       | serial      | rayon       | flynnel_def | flynnel_hint | ratio f_h/r |");
    println!("|-----------------------------|-------------|-------------|-------------|--------------|-------------|");
    for shape in SHAPES {
        eprintln!("=== {} ({}) ===", shape.label, shape.desc);
        let s = measure("serial",  shape, run_serial);
        let r = measure("rayon",   shape, run_rayon);
        let fd = measure("flynnel_def", shape, run_flynnel);
        let fh = measure("flynnel_hint", shape, run_flynnel_hinted);
        let s_med = median(&s);
        let r_med = median(&r);
        let fd_med = median(&fd);
        let fh_med = median(&fh);
        let ratio = fh_med.as_secs_f64() / r_med.as_secs_f64();
        println!(
            "| {:<27} | {:>11} | {:>11} | {:>11} | {:>12} | {:>10.3}x |",
            shape.desc,
            fmt_dur(s_med),
            fmt_dur(r_med),
            fmt_dur(fd_med),
            fmt_dur(fh_med),
            ratio,
        );
        eprintln!(
            "  serial   p10={} med={} p90={}",
            fmt_dur(pct(&s, 10)), fmt_dur(s_med), fmt_dur(pct(&s, 90)),
        );
        eprintln!(
            "  rayon    p10={} med={} p90={}",
            fmt_dur(pct(&r, 10)), fmt_dur(r_med), fmt_dur(pct(&r, 90)),
        );
        eprintln!(
            "  flynnel_def  p10={} med={} p90={}",
            fmt_dur(pct(&fd, 10)), fmt_dur(fd_med), fmt_dur(pct(&fd, 90)),
        );
        eprintln!(
            "  flynnel_hint p10={} med={} p90={}",
            fmt_dur(pct(&fh, 10)), fmt_dur(fh_med), fmt_dur(pct(&fh, 90)),
        );
    }
    println!();
    println!("Done.");
}

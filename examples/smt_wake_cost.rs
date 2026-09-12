//! What a dispatch that wants the SMT siblings pays for their absence.
//!
//! An SMT extension worker that finds no request parks in stages: a
//! yield spin, then a sleep loop, then the parker. Where in that
//! sequence a sibling sits when a request arrives decides how long the
//! dispatch runs without it - a sibling mid-sleep notices at the end of
//! its sleep, a parked one is woken by the request itself. So the wall
//! of one `with_smt` dispatch, taken after the pool has idled for a
//! chosen interval, is the sibling's reaction time made visible, on top
//! of the work.
//!
//! One arm per process: an idle interval, a window. Each iteration
//! idles for the interval, then times a fixed dispatch of latency-bound
//! work the siblings are meant to help with. The row reports the
//! distribution of that wall over the window.
//!
//! Prints one line per window, so a caller watching the log can see
//! which arm it is on.

use std::hint::black_box;
use std::time::{Duration, Instant};

use flynnel::for_each_chunk;
use flynnel::JobPlan;

/// Items per dispatch. Enough to fan out across every worker of a
/// 16-thread pool with several leaves each, so a late sibling is a
/// leaf that waited.
const ITEMS: usize = 512;

/// Dependent multiplies per item: a chain the core cannot overlap, so
/// the work is latency-bound and a sibling on the same core genuinely
/// helps rather than contesting a saturated port.
const CHAIN: u32 = 4_000;

fn work(seed: u64) -> u64 {
    let mut x = seed;
    for _ in 0..CHAIN {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        x ^= x >> 29;
    }
    x
}

fn usage() -> ! {
    eprintln!("usage: smt_wake_cost <idle_us> <window_s>");
    eprintln!("  idle_us:  how long the pool sits idle before each dispatch");
    eprintln!("  window_s: how long to keep dispatching");
    std::process::exit(2)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let idle_us: u64 = match args.next().map(|a| a.parse()) {
        Some(Ok(v)) => v,
        _ => usage(),
    };
    let window_s: u64 = match args.next().map(|a| a.parse()) {
        Some(Ok(v)) => v,
        _ => usage(),
    };
    let idle = Duration::from_micros(idle_us);
    let window = Duration::from_secs(window_s);

    let mut items: Vec<u64> = (0..ITEMS as u64).collect();
    let plan = JobPlan::new(0, ITEMS as u32).with_smt();

    // One dispatch outside the window so the pool is spawned and every
    // worker has run a leaf; the first dispatch of a process measures
    // thread creation, which is not the question.
    for_each_chunk(&plan, &mut items, |slice| {
        for v in slice.iter_mut() {
            *v = work(*v);
        }
    });
    black_box(&items);

    let mut walls_us: Vec<f64> = Vec::new();
    let started = Instant::now();
    while started.elapsed() < window {
        // Idle with the caller asleep, not spinning, so the pool sees
        // no requests and the siblings walk their park sequence.
        std::thread::sleep(idle);
        let t0 = Instant::now();
        for_each_chunk(&plan, &mut items, |slice| {
            for v in slice.iter_mut() {
                *v = work(*v);
            }
        });
        walls_us.push(t0.elapsed().as_secs_f64() * 1e6);
        black_box(&items);
    }

    walls_us.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a measured wall"));
    let n = walls_us.len();
    if n == 0 {
        eprintln!("the window ran no dispatches; raise the window");
        std::process::exit(3);
    }
    let at = |q: f64| walls_us[((n as f64 - 1.0) * q).round() as usize];
    println!(
        "idle_us={idle_us} window_s={window_s} dispatches={n} min_us={:.1} median_us={:.1} \
         p90_us={:.1} max_us={:.1} items={ITEMS} chain={CHAIN}",
        walls_us[0],
        at(0.5),
        at(0.9),
        walls_us[n - 1],
    );
}

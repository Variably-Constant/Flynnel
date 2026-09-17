//! Dispatch throughput with foreign load running, and with none.
//!
//! The acceptance criterion for every adaptive mechanism in the
//! scheduler: it must NEVER be slower than the same code without it,
//! and it must be FASTER when the host is contended. Decision quality -
//! a steadier class, a spread a neighbor cannot move, an SMT answer
//! that tracks the work - is the means. This is the end, and nothing
//! ships on the means alone.
//!
//! # What is different from the other harnesses here
//!
//! `class_loop_cost` and `smt_recovery` time a quiet window, apply load
//! between windows, and time another quiet window. That shape answers
//! what a load did to a DECISION after it passed. It cannot answer what
//! throughput was while the load was on, because no window overlaps it.
//!
//! Here the burners run DURING the measured window. The figure is
//! dispatches completed per second while that much of the machine
//! belongs to someone else.
//!
//! # Why it uses only common API
//!
//! One file, copied into both trees, so the two arms differ in the
//! library and in nothing else. A harness that reads accessors only one
//! side carries cannot span the comparison it exists to make.
//!
//! ```sh
//! throughput_under_load <window_s> <load_threads> <trials>
//! ```

use std::env;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use flynnel::sched::par_iter::for_each_chunk_min_leaf;
use flynnel::{CallSiteState, JobPlan, SiteRef};

static SITE: CallSiteState = CallSiteState::new();

const ITEMS: usize = 1 << 16;
const MIN_LEAF: usize = 256;

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

/// Dispatches completed in `measured`, with whatever else is running.
fn window(buf: &mut [u64], measured: Duration) -> u64 {
    let start = Instant::now();
    let mut dispatches = 0u64;
    while start.elapsed() < measured {
        let plan = JobPlan::new(0, ITEMS as u32).with_site(SiteRef::new(&SITE));
        for_each_chunk_min_leaf(&plan, buf, MIN_LEAF, |chunk| {
            for slot in chunk.iter_mut() {
                let mut acc = *slot;
                for _ in 0..16 {
                    acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                }
                *slot = black_box(acc);
            }
        });
        dispatches += 1;
    }
    dispatches
}

/// Burners held for the lifetime of the returned guard.
struct Load {
    stop: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Load {
    fn start(n: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let mut threads = Vec::with_capacity(n);
        for _ in 0..n {
            let stop = Arc::clone(&stop);
            threads.push(std::thread::spawn(move || {
                let mut acc = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..4_096 {
                        acc = acc.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(3);
                    }
                    black_box(acc);
                }
            }));
        }
        Self { stop, threads }
    }

    /// Stop the burners and report any that panicked, rather than
    /// dropping the join result: a burner that died early means the
    /// window was less loaded than the row claims.
    fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        for (t, handle) in self.threads.into_iter().enumerate() {
            if let Err(panic) = handle.join() {
                eprintln!("load thread {t} panicked, so the window carried less load than stated: {panic:?}");
            }
        }
    }
}

fn main() {
    let window_s: u64 = arg(1, 5);
    let load_threads: usize = arg(2, 12);
    let trials: usize = arg(3, 5);

    // Which arm this process is. Printed rather than assumed, because a
    // switch that failed to engage produces rows indistinguishable from
    // the other arm's, and the driver's label would be the only record
    // of an intent that did not take effect.
    eprintln!("levers: {}", flynnel::sched::levers::describe());

    let measured = Duration::from_secs(window_s);
    let mut buf: Vec<u64> = (0..ITEMS as u64).collect();

    // One warm window, discarded: the first dispatches of a process pay
    // pool startup and the site's first classifier ticks, which belong
    // to no arm.
    let warm = window(&mut buf, Duration::from_secs(1));
    if warm == 0 {
        eprintln!("the warm window ran no dispatches; raise the window length");
        std::process::exit(2);
    }

    for t in 1..=trials {
        let load = (load_threads > 0).then(|| Load::start(load_threads));
        // Let the burners actually reach the cores before timing, or the
        // early part of the window is measured with less load than the
        // row claims.
        if load.is_some() {
            std::thread::sleep(Duration::from_millis(250));
        }
        let n = window(&mut buf, measured);
        if let Some(load) = load {
            load.stop();
        }
        let per_s = n as f64 / window_s as f64;
        println!("throughput {load_threads} {t} {n} {per_s:.2}");
    }
}

//! Dispatch throughput while the CPUs the process may use are taken
//! away from under it.
//!
//! The allowed-width lever caps the worker count by the CPUs the
//! process may use, read from the process affinity mask and the cgroup
//! CPU quota. Neither of those moves when
//! a neighbor gets busy, so a harness that applies LOAD can never
//! reach this mechanism however contended it makes the host: the pool
//! is sized against a width that is still correct. What reaches it is
//! an actual narrowing.
//!
//! The process prints its pid before it starts timing. A driver
//! narrows it partway through - `taskset -acp <cpus> <pid>` on Linux,
//! `cpuset` on FreeBSD, `SetProcessAffinityMask` on Windows - and every
//! window reports the width it was allowed and the worker count the
//! plan resolved. With the lever off the pool keeps chunking for the
//! machine it started on; with it on the count follows the narrowing
//! within the recheck interval.
//!
//! The last line says how many distinct widths were seen. One width is
//! a run where the narrowing never reached the process, which produces
//! windows indistinguishable from a lever that did not help.
//!
//! ```sh
//! width_narrowing <window_s> <windows>
//! ```

use std::env;
use std::hint::black_box;
use std::io::Write;
use std::time::{Duration, Instant};

use flynnel::sched::host_width::allowed_parallelism;
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

fn plan() -> JobPlan {
    JobPlan::new(0, ITEMS as u32).with_site(SiteRef::new(&SITE))
}

/// Dispatches completed in `measured`.
fn window(buf: &mut [u64], measured: Duration) -> u64 {
    let start = Instant::now();
    let mut dispatches = 0u64;
    while start.elapsed() < measured {
        let plan = plan();
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

fn main() {
    let window_s: u64 = arg(1, 5);
    let windows: usize = arg(2, 6).max(2);

    // Before anything else, and flushed: the driver cannot narrow a
    // process it has not been told the pid of, and a pid printed after
    // the first window arrives too late to be narrowed partway.
    println!("pid {}", std::process::id());
    if let Err(err) = std::io::stdout().flush() {
        eprintln!("the pid could not be flushed, so no driver can narrow this run: {err}");
        std::process::exit(2);
    }
    eprintln!("levers: {}", flynnel::sched::levers::describe());

    let measured = Duration::from_secs(window_s);
    let mut buf: Vec<u64> = (0..ITEMS as u64).collect();

    // One warm window, discarded: the first dispatches of a process pay
    // pool startup and the site's first classifier ticks.
    let warm = window(&mut buf, Duration::from_secs(1));
    if warm == 0 {
        eprintln!("the warm window ran no dispatches; raise the window length");
        std::process::exit(2);
    }

    let mut widths = Vec::with_capacity(windows);
    let mut empty = 0usize;
    for i in 1..=windows {
        let n = window(&mut buf, measured);
        if n == 0 {
            empty += 1;
        }
        let allowed = allowed_parallelism();
        widths.push(allowed);
        let per_s = n as f64 / window_s as f64;
        println!(
            "window {i} {n} {per_s:.2} allowed={allowed} workers={}",
            plan().resolved_workers()
        );
    }

    if empty > 0 {
        eprintln!("{empty} of the {windows} windows ran no dispatches; raise the window length");
    }

    widths.sort_unstable();
    widths.dedup();
    println!("widths_seen {}", widths.len());
    if widths.len() < 2 {
        eprintln!(
            "the allowed width held at {} for every window, so the narrowing never \
             reached this process and these rows measure nothing",
            widths.first().copied().unwrap_or(0)
        );
        std::process::exit(3);
    }
}

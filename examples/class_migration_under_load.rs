//! Report what class a uniform-leaf site is classified as, continuously,
//! so a neighbour's load can be started and stopped underneath it.
//!
//! The classifier takes two inputs, a leaf-time mean and a squared
//! coefficient of variation, and preemption moves the second while
//! barely moving the first. A site whose leaves are uniform therefore
//! reads as one class on a quiet host and can read as another on a busy
//! one, without its own work having changed.
//!
//! Both the process-global class and the site's own learned class are
//! printed, because they are separate state reached by separate paths,
//! and the inputs are printed beside them so a classification can be
//! reproduced from what produced it rather than trusted.
//!
//! A migration needs at least 64 leaves in one delta window to take the
//! fast path, and otherwise needs the same observation repeated across
//! consecutive windows, so the leaf count is reported and is what to
//! raise if nothing ever moves.
//!
//! Run it on the build host, announce it, and drive the load separately:
//!
//! ```sh
//! cargo run --release --example class_migration_under_load -- 120 4
//! # in another shell, part way through:
//! powershell -File C:\Temp\loadgen.ps1 -Cores 12 -Seconds 60
//! ```

use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use flynnel::sched::adaptive_profile::{WorkloadClass, active_workload_class};
use flynnel::sched::par_iter::for_each_chunk_min_leaf;
use flynnel::{CallSiteState, JobPlan, SiteRef};

/// One site, owned here rather than resolved from the call location, so
/// the statistics reported are this dispatch's and nothing else's.
static SITE: CallSiteState = CallSiteState::new();

/// Per-item work: a dependent chain of fixed length, so every leaf costs
/// the same and any spread in the measured leaf times came from the host
/// rather than from the workload.
#[inline(never)]
fn item_work(seed: u64, rounds: u32) -> u64 {
    let mut x = seed | 1;
    for _ in 0..rounds {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x = x.wrapping_mul(0x100000001B3);
    }
    x
}

/// An argument that was not supplied takes the default. One that was
/// supplied and does not parse stops the run and says so, because the
/// alternative is a typo silently selecting the default and a reader
/// attributing the result to the value they thought they passed.
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

/// The three site readings taken together, so a row reports one moment
/// rather than three moments a few microseconds apart.
struct SiteView {
    leaves: u64,
    cv2: Option<u64>,
    learned: Option<WorkloadClass>,
}

fn read_site(state: &CallSiteState) -> SiteView {
    SiteView {
        leaves: state.leaf_count(),
        cv2: state.cv2_per_mille(),
        learned: state.learned_class(),
    }
}

fn main() {
    // Every knob is an argument with its default printed below. A run
    // that reports nothing is usually a run whose leaf count never
    // reached the window the classifier needs, and that is only visible
    // if the numbers are on the page.
    let seconds: u64 = arg(1, 120);
    let report_every: u64 = arg(2, 4);
    let items: usize = arg(3, 32_768);
    let rounds: u32 = arg(4, 64);
    // The recursion floor, passed rather than derived. Leaf size is what
    // the classifier's variance is computed over, and a caller who sets
    // only the per-item cost does not thereby set it: `adaptive_min_leaf`
    // is the dispatch cost divided by the per-item cost, so heavier items
    // give a SMALLER floor and a finer split. Two runs that raised the
    // item cost to make leaves coarse got 3.5 items per leaf and a cv2 of
    // 16000 per mille instead.
    let min_leaf: usize = arg(5, 1_024);

    let workers = match std::thread::available_parallelism() {
        Ok(n) => n.get().to_string(),
        Err(err) => format!("unavailable ({err})"),
    };

    println!(
        "seconds {seconds}  report_every {report_every}s  items {items}  \
         rounds {rounds}  min_leaf {min_leaf}  workers {workers}"
    );
    // leaves_per_dispatch and items_per_leaf are the precondition, not
    // decoration. This experiment needs leaves whose own variance is near
    // zero, and a run whose leaves came out at a few items each cannot
    // answer it however clean the class column looks. Reading that off
    // two cumulative counters after the fact is how two runs were taken
    // before anyone noticed.
    println!(
        "elapsed_s  dispatches  leaves  per_disp  items_per_leaf  \
         cv2_per_mille  site_class  global_class  last_ms"
    );

    let site = SiteRef::new(&SITE);
    let mut buf: Vec<u64> = (0..items as u64).collect();

    let start = Instant::now();
    let mut next_report = Duration::from_secs(0);
    let mut dispatches: u64 = 0;
    let mut last_ms;

    while start.elapsed() < Duration::from_secs(seconds) {
        let t0 = Instant::now();
        let plan = JobPlan::new(0, buf.len() as u32).with_site(site);
        for_each_chunk_min_leaf(&plan, &mut buf, min_leaf, |slice| {
            for x in slice {
                *x = item_work(*x, rounds);
            }
        });
        black_box(buf[0]);
        last_ms = t0.elapsed().as_secs_f64() * 1000.0;
        dispatches += 1;

        if start.elapsed() >= next_report {
            let view = read_site(&SITE);
            // `none` is printed as itself rather than as a number. A
            // site below the sample floor has not classified, which is a
            // different statement from having classified as anything.
            let cv2 = match view.cv2 {
                Some(v) => v.to_string(),
                None => "none".to_string(),
            };
            let learned = match view.learned {
                Some(c) => format!("{c:?}"),
                None => "none".to_string(),
            };
            let per_disp = if dispatches > 0 {
                view.leaves as f64 / dispatches as f64
            } else {
                0.0
            };
            let items_per_leaf = if per_disp > 0.0 {
                items as f64 / per_disp
            } else {
                0.0
            };
            println!(
                "{:9.1}  {:10}  {:8}  {:8.1}  {:14.1}  {:>13}  {:>12}  {:>12}  {:7.2}",
                start.elapsed().as_secs_f64(),
                dispatches,
                view.leaves,
                per_disp,
                items_per_leaf,
                cv2,
                learned,
                format!("{:?}", active_workload_class()),
                last_ms,
            );
            next_report = start.elapsed() + Duration::from_secs(report_every);
        }
    }

    println!("done after {dispatches} dispatches");
}

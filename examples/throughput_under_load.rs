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
//! # The engagement line
//!
//! A mechanism that never ran produces the same clean uniform rows as
//! one that ran and did not help. The last line of a run reports what
//! each switch had to work with: how many leaves reached the sampled
//! on-core path, whether the on-core spread differs from the wall
//! spread, whether a window has been classified, what the SMT answer
//! came out as, and how the resolved width compares to the width the
//! process is allowed. A switch whose figures are identical across its
//! own on and off arms did not engage, and its throughput row says
//! nothing about the mechanism.
//!
//! ```sh
//! throughput_under_load <window_s> <load_threads> <trials> [smt_prior]
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

/// The plan every dispatch here runs under.
///
/// `smt_prior` asks for the SMT prior that `effective_use_smt` needs
/// before it consults anything: with the prior false the method returns
/// on its first line and the variance switch is never read. The shape
/// `(0, 1 << 16)` classifies as `Streaming`, whose profile parks
/// siblings, so the prior is false unless a caller sets it.
fn plan(smt_prior: bool) -> JobPlan {
    let plan = JobPlan::new(0, ITEMS as u32).with_site(SiteRef::new(&SITE));
    if smt_prior { plan.with_smt() } else { plan }
}

/// Dispatches completed in `measured`, with whatever else is running.
fn window(buf: &mut [u64], measured: Duration, smt_prior: bool) -> u64 {
    let start = Instant::now();
    let mut dispatches = 0u64;
    while start.elapsed() < measured {
        let plan = plan(smt_prior);
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

/// A figure that may not have been measured, printed so the two cases
/// are distinguishable. A dash is no reading; zero is a reading of
/// zero, which for a spread means perfectly uniform work.
fn reading(value: Option<u64>) -> String {
    match value {
        Some(v) => v.to_string(),
        None => "-".to_string(),
    }
}

/// What each switch had to work with by the end of the run.
///
/// Read against the same line from the arm the switch was off in. Two
/// arms whose engagement figures match ran the same code whatever the
/// throughput rows did.
fn engagement(smt_prior: bool) {
    let plan = plan(smt_prior);
    let site = &SITE;
    println!(
        "engagement leaves={} oncore_items={} per_item_ns={} cv2_wall={} cv2_oncore={} \
         cv2_window={} window_ticks={} class={:?} workers={} allowed={} smt={}",
        site.leaf_count(),
        site.oncore_items(),
        reading(site.per_item_ns()),
        reading(site.per_item_cv2_per_mille()),
        reading(site.per_item_oncore_cv2_per_mille()),
        reading(site.window_cv2_per_mille()),
        site.window_ticks(),
        site.learned_class(),
        plan.resolved_workers(),
        flynnel::sched::host_width::allowed_parallelism(),
        plan.effective_use_smt(),
    );
}

fn main() {
    let window_s: u64 = arg(1, 5);
    let load_threads: usize = arg(2, 12);
    let trials: usize = arg(3, 5);
    let smt_prior: bool = arg::<u8>(4, 0) != 0;

    // Which arm this process is. Printed rather than assumed, because a
    // switch that failed to engage produces rows indistinguishable from
    // the other arm's, and the driver's label would be the only record
    // of an intent that did not take effect.
    eprintln!("levers: {}", flynnel::sched::levers::describe());

    // The SMT switch is read inside `effective_use_smt`, past a return
    // that fires when the plan's prior is false. Asking for the prior
    // and not getting it leaves a run that cannot reach the switch,
    // and whose rows would look exactly like a switch that did not
    // help.
    if smt_prior && !plan(true).use_smt {
        eprintln!(
            "the SMT prior was asked for and the plan does not carry it, so \
             effective_use_smt short-circuits and no arm here can reach the \
             window switch; nothing is measured"
        );
        std::process::exit(2);
    }

    let measured = Duration::from_secs(window_s);
    let mut buf: Vec<u64> = (0..ITEMS as u64).collect();

    // One warm window, discarded: the first dispatches of a process pay
    // pool startup and the site's first classifier ticks, which belong
    // to no arm.
    let warm = window(&mut buf, Duration::from_secs(1), smt_prior);
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
        let n = window(&mut buf, measured, smt_prior);
        if let Some(load) = load {
            load.stop();
        }
        let per_s = n as f64 / window_s as f64;
        println!("throughput {load_threads} {t} {n} {per_s:.2}");
    }

    engagement(smt_prior);
}

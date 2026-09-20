//! Whether an SMT decision that moved under load comes back when the
//! load is gone.
//!
//! `JobPlan::effective_use_smt` asks a call site whether its leaves are
//! uniform enough that SMT siblings would only contest the same
//! execution unit. The answer is derived from a measured spread, and
//! contention enters a spread as variance, so a busy stretch can flip
//! the answer. The question here is whether it flips back.
//!
//! # Why this is a separate harness
//!
//! It uses only API that BOTH commits under comparison carry, so one
//! file can be copied into either tree and the two arms differ in the
//! library and in nothing else. The larger `class_loop_cost` harness
//! reads on-core accessors that exist on only one side, which makes it
//! unusable for a comparison that spans them.
//!
//! # What it reports
//!
//! One row: the SMT answer before the load, the answer in the first
//! window after it, and the window the answer returned in, counting the
//! first post-load window as one. A dash means it had not returned by
//! the last window, which is a censored observation and not a duration.
//!
//! ```sh
//! cargo run --release --example smt_recovery -- 8 4 12 6
//! ```

use std::env;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use flynnel::sched::par_iter::for_each_chunk_min_leaf;
use flynnel::{CallSiteState, JobPlan, SiteRef};

/// The one site this process dispatches through, so the figures belong
/// to it rather than to the process.
static SITE: CallSiteState = CallSiteState::new();

/// Items per dispatch and the recursion floor, both fixed so a routing
/// that lowers the floor is doing so on its own account.
const ITEMS: usize = 1 << 20;
const MIN_LEAF: usize = 1_024;

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

/// A plan carrying the site, with the SMT prior ON.
///
/// The prior has to be true or there is nothing to measure:
/// `effective_use_smt` returns false immediately when the plan says
/// false, before it consults any variance, so a plan built with the
/// default prior never reaches the code under test and every row reads
/// `false false 1` on both commits. That is a mechanism that never
/// engaged, and it looks exactly like two commits that agree.
fn plan() -> JobPlan {
    JobPlan::new(0, ITEMS as u32).with_smt().with_site(SiteRef::new(&SITE))
}

/// Dispatch uniform work for `measured`, and report how many dispatches
/// ran. Uniform on purpose: the spread the site learns should then be
/// the host's contribution and nothing else.
fn window(buf: &mut [u64], measured: Duration) -> u64 {
    let start = Instant::now();
    let mut dispatches = 0u64;
    while start.elapsed() < measured {
        let p = plan();
        for_each_chunk_min_leaf(&p, buf, MIN_LEAF, |chunk| {
            for slot in chunk.iter_mut() {
                let mut acc = *slot;
                // A fixed number of operations per item, so every leaf
                // costs the same per item and any spread the site sees
                // came from the machine.
                for _ in 0..24 {
                    acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                }
                *slot = black_box(acc);
            }
        });
        dispatches += 1;
    }
    dispatches
}

/// Occupy `threads` cores for `held`, then join them.
fn load(threads: usize, held: Duration) {
    if threads == 0 {
        std::thread::sleep(held);
        return;
    }
    let stop = Arc::new(AtomicBool::new(false));
    let mut burners = Vec::with_capacity(threads);
    for _ in 0..threads {
        let stop = Arc::clone(&stop);
        burners.push(std::thread::spawn(move || {
            let mut acc = 0u64;
            while !stop.load(Ordering::Relaxed) {
                for _ in 0..4_096 {
                    acc = acc.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(3);
                }
                black_box(acc);
            }
        }));
    }
    std::thread::sleep(held);
    stop.store(true, Ordering::Relaxed);
    for (t, burner) in burners.into_iter().enumerate() {
        if let Err(panic) = burner.join() {
            eprintln!("load thread {t} panicked: {panic:?}");
        }
    }
}

fn main() {
    let window_s: u64 = arg(1, 8);
    let load_s: u64 = arg(2, 4);
    let load_threads: usize = arg(3, 12);
    let post_windows: usize = arg(4, 6).max(1);

    eprintln!(
        "window {window_s}s  load {load_s}s  load_threads {load_threads}  \
         post_windows {post_windows}  items {ITEMS}  min_leaf {MIN_LEAF}"
    );

    let measured = Duration::from_secs(window_s);
    let mut buf: Vec<u64> = (0..ITEMS as u64).collect();

    // Before anything is timed: can the answer move at all? A site with
    // too few samples, or a prior of false, pins effective_use_smt to
    // one value whatever the host does, and a row from such a run says
    // nothing while looking like a clean negative.
    if !plan().use_smt {
        eprintln!(
            "the plan's SMT prior is false, so effective_use_smt short-circuits and this \
             run cannot observe the decision moving; nothing is measured"
        );
        std::process::exit(2);
    }

    let pre_n = window(&mut buf, measured);
    let smt_pre = plan().effective_use_smt();

    load(load_threads, Duration::from_secs(load_s));
    // Let the pool park and the host settle, so the first post window
    // times the scheduler rather than the tail of the burners.
    std::thread::sleep(Duration::from_millis(250));

    let post_n = window(&mut buf, measured);
    let smt_post = plan().effective_use_smt();

    let mut returned_at = (smt_post == smt_pre).then_some(1usize);
    let mut empty = 0usize;
    for n in 2..=post_windows {
        if returned_at.is_some() {
            break;
        }
        let ran = window(&mut buf, measured);
        // A window that dispatched nothing could not have moved the
        // answer, so it is not evidence that the answer failed to
        // return. Counted and reported, because a censored reading
        // built out of empty windows looks exactly like one built out
        // of real ones.
        if ran == 0 {
            empty += 1;
            continue;
        }
        if plan().effective_use_smt() == smt_pre {
            returned_at = Some(n);
        }
    }
    if empty > 0 {
        eprintln!(
            "{empty} of the {post_windows} windows ran no dispatches; raise the window \
             length before reading the returned column"
        );
    }
    if pre_n == 0 || post_n == 0 {
        eprintln!("the windows ran {pre_n} dispatches before and {post_n} after; raise the window");
    }

    // smt_pre, smt_post, the window it returned in, and the dispatch
    // counts the answer rests on.
    println!(
        "smt_recovery {} {} {} {} {}",
        smt_pre,
        smt_post,
        returned_at.map_or_else(|| "-".to_string(), |n| n.to_string()),
        pre_n,
        post_n,
    );

    // What the switch had to work with. It chooses between the window
    // the classifier last read and the site's lifetime figure, and
    // falls back to the lifetime one until a window has been
    // classified - so with window_ticks at zero both arms read the
    // same number and the switch cannot have moved anything. A row of
    // agreeing arms means one thing in that case and another when a
    // window exists, and the rows alone do not say which.
    let site = &SITE;
    // What every switch resolved to, beside what the run measured. Two
    // of them default on, so an arm that leaves its variable unset runs
    // with the lever on whatever its label says.
    eprintln!("levers: {}", flynnel::sched::levers::describe());
    let reading = |v: Option<u64>| v.map_or_else(|| "-".to_string(), |n| n.to_string());
    println!(
        "smt_engagement window_ticks={} cv2_window={} cv2_window_min={} \
         cv2_window_max={} cv2_lifetime={} leaves={} \
         per_item_ns={} class={:?} smt_switch={}",
        site.window_ticks(),
        reading(site.window_cv2_per_mille()),
        reading(site.window_cv2_range_per_mille().map(|(lo, _)| lo)),
        reading(site.window_cv2_range_per_mille().map(|(_, hi)| hi)),
        reading(site.cv2_per_mille()),
        site.leaf_count(),
        reading(site.per_item_ns()),
        site.learned_class(),
        flynnel::sched::levers::smt_from_window(),
    );
}

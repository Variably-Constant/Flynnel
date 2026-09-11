//! What a class that cannot return costs: one site's dispatch time before
//! a load burst and after it, with the load gone in both measured windows.
//!
//! A site whose learned class moves under load keeps the routing that class
//! selects, and that routing splits later dispatches differently. The
//! quantity here is what the routing costs once the load is gone, so the
//! burst runs between the two measured windows and in neither of them.
//!
//! Three routings run the same sequence:
//! - `adaptive` builds each plan with `JobPlan::new`, so the site's learned
//!   class re-derives the routing of every later dispatch;
//! - `streaming` pins the Streaming profile, so the site still classifies
//!   but its class routes nothing;
//! - `latency` pins LatencyBound, the routing an adaptive site lands on,
//!   for every window including the first.
//!
//! Each row carries both windows' median dispatch time, their ratio, the
//! class at the end of each window, and the leaf shape the op was handed.
//! A ratio that the pinned routings do not show is what the class having
//! moved costs. Each shape runs its routings in one order on even trials
//! and the reverse on odd ones, so an effect that depends on the order is
//! visible as one.
//!
//! Every arm owns its own site, so nothing one arm learns is another's
//! starting condition.
//!
//! ```sh
//! cargo run --release --example class_loop_cost -- 8 6 3 12
//! ```

use std::env;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use flynnel::sched::adaptive_profile::WorkloadClass;
use flynnel::sched::par_iter::for_each_chunk_min_leaf;
use flynnel::{CallSiteState, DispatchProfile, JobPlan, SiteRef};

/// The recursion floor every arm passes, so a routing that lowers the
/// floor below it is doing so on its own account.
const MIN_LEAF: usize = 1_024;

/// Trials one run can take. The site array is sized for it.
const MAX_TRIALS: usize = 4;

/// One site per shape, routing and trial.
static SITES: [CallSiteState; SHAPES.len() * ROUTINGS.len() * MAX_TRIALS] =
    [const { CallSiteState::new() }; SHAPES.len() * ROUTINGS.len() * MAX_TRIALS];

/// Leaves the op ran in the current window: how many, their summed items,
/// and the smallest and largest it was handed.
static LEAF_CALLS: AtomicU64 = AtomicU64::new(0);
static LEAF_ITEMS: AtomicU64 = AtomicU64::new(0);
static LEAF_MIN: AtomicU64 = AtomicU64::new(u64::MAX);
static LEAF_MAX: AtomicU64 = AtomicU64::new(0);

fn leaf_stats_reset() {
    LEAF_CALLS.store(0, Ordering::Relaxed);
    LEAF_ITEMS.store(0, Ordering::Relaxed);
    LEAF_MIN.store(u64::MAX, Ordering::Relaxed);
    LEAF_MAX.store(0, Ordering::Relaxed);
}

fn leaf_record(items: usize) {
    let n = items as u64;
    LEAF_CALLS.fetch_add(1, Ordering::Relaxed);
    LEAF_ITEMS.fetch_add(n, Ordering::Relaxed);
    LEAF_MIN.fetch_min(n, Ordering::Relaxed);
    LEAF_MAX.fetch_max(n, Ordering::Relaxed);
}

/// The window's leaves per dispatch, and the smallest and largest leaf, as
/// `min-max`. A window with no leaf says so rather than reporting zero.
fn leaf_stats(dispatches: u64) -> (f64, String) {
    let calls = LEAF_CALLS.load(Ordering::Relaxed);
    if calls == 0 || dispatches == 0 {
        return (0.0, "none".to_string());
    }
    let smallest = LEAF_MIN.load(Ordering::Relaxed);
    let largest = LEAF_MAX.load(Ordering::Relaxed);
    (calls as f64 / dispatches as f64, format!("{smallest}-{largest}"))
}

/// An argument that was not supplied takes the default. One that was
/// supplied and does not parse stops the run and says so.
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

/// Dependent xorshift chain. The cost is the dependency chain rather than
/// anything the optimizer can vectorize away, and `rounds` is what moves an
/// item between the classifier's per-item bands.
#[inline(never)]
fn chain(seed: u64, rounds: u32) -> u64 {
    let mut x = seed | 1;
    for _ in 0..rounds {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x = x.wrapping_mul(0x100000001B3);
    }
    x
}

/// Dependent gather over a table past last-level cache, so each item pays a
/// miss a sibling thread could overlap.
#[inline(never)]
fn gather(seed: u64, table: &[u64]) -> u64 {
    let mask = table.len() - 1;
    let mut idx = (seed as usize) & mask;
    let mut acc = 0u64;
    for _ in 0..64 {
        let v = table[idx];
        acc = acc.wrapping_add(v);
        idx = (v as usize) & mask;
    }
    acc
}

/// 8 MiB of u64, past last-level cache on the hosts this runs on, and a
/// power of two so the index mask is exact.
const TABLE_LEN: usize = 1 << 20;

fn build_table() -> Vec<u64> {
    let mut t = vec![0u64; TABLE_LEN];
    let mut x = 0x243F_6A88_85A3_08D3u64;
    for (i, slot) in t.iter_mut().enumerate() {
        x = chain(x ^ i as u64, 1);
        *slot = x;
    }
    t
}

/// The work shapes, spanning the per-item bands the classifier splits on.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Shape {
    Fine,
    Medium,
    Heavy,
    Huge,
    Gather,
}

const SHAPES: [Shape; 5] = [Shape::Fine, Shape::Medium, Shape::Heavy, Shape::Huge, Shape::Gather];

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Shape::Fine => "fine",
            Shape::Medium => "medium",
            Shape::Heavy => "heavy",
            Shape::Huge => "huge",
            Shape::Gather => "gather",
        }
    }

    fn index(self) -> usize {
        match self {
            Shape::Fine => 0,
            Shape::Medium => 1,
            Shape::Heavy => 2,
            Shape::Huge => 3,
            Shape::Gather => 4,
        }
    }

    /// Items per dispatch. The gather shape takes fewer because each item
    /// costs a chain of misses.
    fn items(self) -> usize {
        match self {
            Shape::Fine => 262_144,
            Shape::Medium => 65_536,
            Shape::Heavy => 16_384,
            Shape::Huge => 24_576,
            Shape::Gather => 32_768,
        }
    }

    fn apply(self, slice: &mut [u64], table: &[u64]) {
        match self {
            Shape::Fine => {
                for x in slice {
                    *x = chain(*x, 2);
                }
            }
            Shape::Medium => {
                for x in slice {
                    *x = chain(*x, 64);
                }
            }
            Shape::Heavy => {
                for x in slice {
                    *x = chain(*x, 400);
                }
            }
            Shape::Huge => {
                for x in slice {
                    *x = chain(*x, 4_000);
                }
            }
            Shape::Gather => {
                for x in slice {
                    *x = gather(*x, table);
                }
            }
        }
    }
}

/// How each dispatch's plan is built.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Routing {
    Adaptive,
    PinnedStreaming,
    PinnedLatency,
}

const ROUTINGS: [Routing; 3] =
    [Routing::Adaptive, Routing::PinnedStreaming, Routing::PinnedLatency];

impl Routing {
    fn name(self) -> &'static str {
        match self {
            Routing::Adaptive => "adaptive",
            Routing::PinnedStreaming => "streaming",
            Routing::PinnedLatency => "latency",
        }
    }

    fn index(self) -> usize {
        match self {
            Routing::Adaptive => 0,
            Routing::PinnedStreaming => 1,
            Routing::PinnedLatency => 2,
        }
    }

    fn plan(self, items: usize, site: SiteRef) -> JobPlan {
        let batch = items as u32;
        match self {
            Routing::Adaptive => JobPlan::new(0, batch),
            Routing::PinnedStreaming => {
                JobPlan::set_profile(0, batch, DispatchProfile::Streaming)
            }
            Routing::PinnedLatency => {
                JobPlan::set_profile(0, batch, DispatchProfile::LatencyBound)
            }
        }
        .with_site(site)
    }
}

/// The class a site holds, or `none` before it has classified.
fn class_name(class: Option<WorkloadClass>) -> String {
    match class {
        Some(c) => format!("{c:?}"),
        None => "none".to_string(),
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

/// One measured window: dispatch until `duration` has passed, timing each
/// dispatch. Returns the median dispatch time in milliseconds, the
/// dispatch count, the leaves per dispatch and the leaf-size span.
fn window(
    shape: Shape,
    routing: Routing,
    site: SiteRef,
    buf: &mut [u64],
    table: &[u64],
    duration: Duration,
) -> (f64, u64, f64, String) {
    leaf_stats_reset();
    let items = buf.len();
    let start = Instant::now();
    let mut times = Vec::new();
    while start.elapsed() < duration {
        let t0 = Instant::now();
        let plan = routing.plan(items, site);
        for_each_chunk_min_leaf(&plan, buf, MIN_LEAF, |slice| {
            leaf_record(slice.len());
            shape.apply(slice, table);
        });
        black_box(buf[0]);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let dispatches = times.len() as u64;
    let (per_dispatch, span) = leaf_stats(dispatches);
    (median(&mut times), dispatches, per_dispatch, span)
}

/// Dispatch without timing for `duration`, with `threads` burners loading
/// the box alongside. The burners stop and are joined before it returns.
fn load_window(
    shape: Shape,
    routing: Routing,
    site: SiteRef,
    buf: &mut [u64],
    table: &[u64],
    duration: Duration,
    threads: usize,
) {
    let stop = Arc::new(AtomicBool::new(false));
    let burners: Vec<_> = (0..threads)
        .map(|t| {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut x = t as u64 + 1;
                while !stop.load(Ordering::Relaxed) {
                    x = black_box(chain(x, 4_096));
                }
            })
        })
        .collect();

    let items = buf.len();
    let start = Instant::now();
    while start.elapsed() < duration {
        let plan = routing.plan(items, site);
        for_each_chunk_min_leaf(&plan, buf, MIN_LEAF, |slice| {
            leaf_record(slice.len());
            shape.apply(slice, table);
        });
        black_box(buf[0]);
    }

    stop.store(true, Ordering::Relaxed);
    for (t, burner) in burners.into_iter().enumerate() {
        if let Err(panic) = burner.join() {
            eprintln!("load thread {t} panicked: {panic:?}");
        }
    }
}

/// One trial's outcome, kept for the closing table.
struct Row {
    shape: Shape,
    routing: Routing,
    ratio: f64,
    class_after: Option<WorkloadClass>,
}

fn main() {
    let window_s: u64 = arg(1, 8);
    let load_s: u64 = arg(2, 6);
    let trials: usize = arg(3, 3);
    let load_threads: usize = arg(4, 12);
    if trials == 0 || trials > MAX_TRIALS {
        eprintln!("argument 3 is the trial count, 1 to {MAX_TRIALS}, not {trials}");
        std::process::exit(2);
    }
    println!(
        "window {window_s}s  load {load_s}s  trials {trials}  load_threads {load_threads}  \
         min_leaf {MIN_LEAF}"
    );
    println!(
        "shape    routing    trial  pre_ms     post_ms    ratio   class_pre     class_post    \
         leaves_pre  leaves_post  sizes_pre  sizes_post"
    );

    let table = build_table();
    let measured = Duration::from_secs(window_s);
    let loaded = Duration::from_secs(load_s);
    let mut rows = Vec::new();

    for shape in SHAPES {
        let mut buf: Vec<u64> = (0..shape.items() as u64).collect();
        for trial in 0..trials {
            let mut order = ROUTINGS;
            if trial % 2 == 1 {
                order.reverse();
            }
            for routing in order {
                let index = (shape.index() * ROUTINGS.len() + routing.index()) * MAX_TRIALS + trial;
                let state = &SITES[index];
                let site = SiteRef::new(state);

                let (pre_ms, pre_n, pre_leaves, pre_sizes) =
                    window(shape, routing, site, &mut buf, &table, measured);
                let class_pre = state.learned_class();

                load_window(shape, routing, site, &mut buf, &table, loaded, load_threads);
                // The burners are joined; this lets the pool's workers park
                // and the host settle before the second window is timed.
                std::thread::sleep(Duration::from_millis(250));

                let (post_ms, post_n, post_leaves, post_sizes) =
                    window(shape, routing, site, &mut buf, &table, measured);
                let class_post = state.learned_class();

                let ratio = if pre_ms > 0.0 { post_ms / pre_ms } else { f64::NAN };
                println!(
                    "{:7}  {:9}  {:5}  {:9.3}  {:9.3}  {:6.3}  {:>12}  {:>12}  {:10.1}  {:11.1}  \
                     {:>9}  {:>10}",
                    shape.name(),
                    routing.name(),
                    trial,
                    pre_ms,
                    post_ms,
                    ratio,
                    class_name(class_pre),
                    class_name(class_post),
                    pre_leaves,
                    post_leaves,
                    pre_sizes,
                    post_sizes,
                );
                if pre_n == 0 || post_n == 0 {
                    eprintln!(
                        "{} {} trial {trial} ran {pre_n} dispatches before and {post_n} after; \
                         raise the window",
                        shape.name(),
                        routing.name()
                    );
                }
                rows.push(Row { shape, routing, ratio, class_after: class_post });
            }
        }
    }

    println!();
    println!("shape    routing    median_ratio  trials  class_after");
    for shape in SHAPES {
        for routing in ROUTINGS {
            let mut ratios: Vec<f64> = rows
                .iter()
                .filter(|r| r.shape == shape && r.routing == routing)
                .map(|r| r.ratio)
                .collect();
            let classes: Vec<String> = rows
                .iter()
                .filter(|r| r.shape == shape && r.routing == routing)
                .map(|r| class_name(r.class_after))
                .collect();
            let count = ratios.len();
            println!(
                "{:7}  {:9}  {:12.3}  {:6}  {}",
                shape.name(),
                routing.name(),
                median(&mut ratios),
                count,
                classes.join(","),
            );
        }
    }
}

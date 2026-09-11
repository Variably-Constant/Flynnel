//! Report what class a uniform-leaf site is classified as, continuously,
//! while the run loads the box itself in a scheduled window, with the
//! box's CPU on every row.
//!
//! The classifier takes two inputs, a leaf-time mean and a squared
//! coefficient of variation, and preemption moves the second while
//! barely moving the first. A site whose leaves are uniform therefore
//! reads as one class on a quiet host and can read as another on a busy
//! one, without its own work having changed.
//!
//! Both the process-global class and the site's own learned class are
//! printed, because they are separate state reached by separate paths.
//! The site's class is decided per classifier tick from the mean and
//! cv^2 of the delta window that tick classified, and those two are
//! printed beside it so a classification can be reproduced from what
//! produced it rather than trusted. The lifetime cv^2 is printed as well;
//! it drifts with every leaf the site has run and does not decide the
//! class.
//!
//! A migration needs at least 64 leaves in one delta window to take the
//! fast path, and otherwise needs the same observation repeated across
//! consecutive windows, so the leaf count is reported and is what to
//! raise if nothing ever moves. That count is the leaves the site's
//! recorder timed, which on this dispatch's lazy bisect is one leaf in
//! every `LEAF_SAMPLE_STRIDE` (`src/sched/par_iter.rs`), and the window
//! mean is in the recorder's unit, rdtsc ticks on x86_64.
//!
//! The site times whole leaves and records no item counts, so leaves of
//! different sizes read as variance even when every item costs the same.
//! Every row therefore lists the leaves each size ran over its interval,
//! with their count, mean time and cv^2. When those per-size values are low
//! and the site's window cv^2 is high, the variance came from the mix of
//! sizes rather than from the time any one size took.
//!
//! The ninth argument picks the routing. `adaptive` builds each plan with
//! `JobPlan::new`, so the site's learned class re-derives the routing of
//! every later dispatch; `pinned` builds it with `JobPlan::set_profile` on
//! Streaming, so the site still classifies but its class routes nothing.
//! Running both separates what the load did to the leaves from what the
//! class did to them once it moved.
//!
//! The load comes from burner threads inside this process, from `load_at`
//! seconds for `load_for` seconds. Every row reports, over its interval,
//! the cores the whole box was busy, the cores this process used, and the
//! difference: load that no part of this run produced. A row whose CPU
//! context cannot be read says so, and the reason is logged with the row's
//! time. The closing report takes every foreign-load level the run saw, in
//! half-core steps, and for the rows at or below each level says whether
//! the site left Streaming before the load, whether it left under the
//! load, and what class it was in at the end.
//!
//! ```sh
//! cargo run --release --example class_migration_under_load -- 220 4 24576 4000 1024 60 60 12 adaptive
//! ```

use std::collections::BTreeMap;
use std::env;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use flynnel::sched::adaptive_profile::{WorkloadClass, active_workload_class};
use flynnel::sched::par_iter::for_each_chunk_min_leaf;
use flynnel::{CallSiteState, DispatchProfile, JobPlan, SiteRef};

/// One site, owned here rather than resolved from the call location, so
/// the statistics reported are this dispatch's and nothing else's.
static SITE: CallSiteState = CallSiteState::new();

/// The leaves one size bucket ran: bucket 0 holds empty slices and bucket
/// `b + 1` holds slices whose item count has base-2 logarithm `b`. `items`
/// is the item count of the latest leaf the bucket took. Times are summed
/// as nanoseconds and their squares as `(ns >> 8)^2`, the scaled form the
/// site's own statistics use. A leaf's sums are written before its count and
/// a reading takes the count first, so a reading's sums cover at least the
/// leaves its count does.
struct LeafSize {
    items: AtomicU64,
    count: AtomicU64,
    sum_ns: AtomicU64,
    sumsq_scaled: AtomicU64,
    square_overflows: AtomicU64,
}

impl LeafSize {
    const fn new() -> Self {
        Self {
            items: AtomicU64::new(0),
            count: AtomicU64::new(0),
            sum_ns: AtomicU64::new(0),
            sumsq_scaled: AtomicU64::new(0),
            square_overflows: AtomicU64::new(0),
        }
    }
}

const LEAF_BUCKETS: usize = 65;

static LEAF_SIZES: [LeafSize; LEAF_BUCKETS] = [const { LeafSize::new() }; LEAF_BUCKETS];

/// Record one leaf of `items` items that took `nanos`. A scaled square that
/// does not fit in 64 bits is counted in `square_overflows` instead of
/// being added.
fn record_leaf_size(items: usize, nanos: u64) {
    let bucket = match items.checked_ilog2() {
        Some(log2) => &LEAF_SIZES[log2 as usize + 1],
        None => &LEAF_SIZES[0],
    };
    let scaled = nanos >> 8;
    bucket.items.store(items as u64, Ordering::Relaxed);
    bucket.sum_ns.fetch_add(nanos, Ordering::Relaxed);
    match scaled.checked_mul(scaled) {
        Some(square) => {
            bucket.sumsq_scaled.fetch_add(square, Ordering::Relaxed);
        }
        None => {
            bucket.square_overflows.fetch_add(1, Ordering::Relaxed);
        }
    }
    bucket.count.fetch_add(1, Ordering::Relaxed);
}

/// One bucket's counters as read at one moment.
#[derive(Clone, Copy)]
struct LeafSizeReading {
    items: u64,
    count: u64,
    sum_ns: u64,
    sumsq_scaled: u64,
    square_overflows: u64,
}

fn read_leaf_sizes() -> [LeafSizeReading; LEAF_BUCKETS] {
    std::array::from_fn(|b| {
        let bucket = &LEAF_SIZES[b];
        let count = bucket.count.load(Ordering::Relaxed);
        LeafSizeReading {
            items: bucket.items.load(Ordering::Relaxed),
            count,
            sum_ns: bucket.sum_ns.load(Ordering::Relaxed),
            sumsq_scaled: bucket.sumsq_scaled.load(Ordering::Relaxed),
            square_overflows: bucket.square_overflows.load(Ordering::Relaxed),
        }
    })
}

/// The leaves each size ran between two readings, smallest first, as
/// `items:count:mean_ms:cv2_per_mille` joined by commas, or `none` when no
/// leaf ran. A size whose window lost a square to overflow carries
/// `:square_overflows=N` and its cv^2 is printed as `unknown`.
fn leaf_sizes_between(
    earlier: &[LeafSizeReading; LEAF_BUCKETS],
    later: &[LeafSizeReading; LEAF_BUCKETS],
) -> Result<String, String> {
    let mut sizes = Vec::new();
    for (before, after) in earlier.iter().zip(later) {
        let count = advanced("a leaf-size count", before.count, after.count)?;
        if count == 0 {
            continue;
        }
        let sum = advanced("a leaf-size time sum", before.sum_ns, after.sum_ns)? as f64;
        let sumsq = advanced("a leaf-size squared time sum", before.sumsq_scaled, after.sumsq_scaled)? as f64;
        let overflows = advanced("a leaf-size square overflow count", before.square_overflows, after.square_overflows)?;
        let n = count as f64;
        let mean = sum / n;
        if overflows == 0 {
            let variance = sumsq * 65_536.0 / n - mean * mean;
            let cv2 = variance / (mean * mean) * 1_000.0;
            sizes.push(format!("{}:{count}:{:.2}:{cv2:.0}", after.items, mean / 1e6));
        } else {
            sizes.push(format!(
                "{}:{count}:{:.2}:unknown:square_overflows={overflows}",
                after.items,
                mean / 1e6
            ));
        }
    }
    if sizes.is_empty() {
        return Ok("none".to_string());
    }
    Ok(sizes.join(","))
}

/// How each dispatch's plan is built.
#[derive(Clone, Copy)]
enum Routing {
    /// `JobPlan::new`: on every dispatch the site's learned class re-derives
    /// the plan's SMT activation, oversubscription and per-item estimate.
    Adaptive,
    /// `JobPlan::set_profile` with Streaming: the caller named the profile,
    /// so the site still classifies and its class is printed, but it routes
    /// nothing.
    Pinned,
}

impl Routing {
    fn name(self) -> &'static str {
        match self {
            Routing::Adaptive => "adaptive",
            Routing::Pinned => "pinned",
        }
    }
}

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
    sampled_leaves: u64,
    cv2: Option<u64>,
    window_mean_ticks: Option<u64>,
    window_cv2: Option<u64>,
    learned: Option<WorkloadClass>,
}

fn read_site(state: &CallSiteState) -> SiteView {
    SiteView {
        sampled_leaves: state.leaf_count(),
        cv2: state.cv2_per_mille(),
        window_mean_ticks: state.window_mean_ticks(),
        window_cv2: state.window_cv2_per_mille(),
        learned: state.learned_class(),
    }
}

/// `later - earlier` for a cumulative counter, or an error naming both
/// readings when the counter went backwards.
fn advanced(what: &str, earlier: u64, later: u64) -> Result<u64, String> {
    match later.checked_sub(earlier) {
        Some(delta) => Ok(delta),
        None => Err(format!("{what} went backwards between samples: {earlier} then {later}")),
    }
}

/// CPU accounting on Windows: the whole box's busy time and this process's
/// time, both in 100 ns units summed over every logical CPU.
#[cfg(windows)]
mod cpu {
    use std::ffi::c_void;
    use std::time::Instant;

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    impl FileTime {
        fn units(self) -> u64 {
            (u64::from(self.high) << 32) | u64::from(self.low)
        }
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetSystemTimes(idle: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn GetProcessTimes(
            process: *mut c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }

    pub struct Sample {
        at: Instant,
        busy: u64,
        own: u64,
    }

    pub fn read() -> Result<Sample, String> {
        let mut idle = FileTime::default();
        let mut kernel = FileTime::default();
        let mut user = FileTime::default();
        let mut created = FileTime::default();
        let mut exited = FileTime::default();
        let mut own_kernel = FileTime::default();
        let mut own_user = FileTime::default();
        // SAFETY: every pointer is to a live local FileTime, and the handle
        // GetCurrentProcess returns needs no closing.
        let (system_ok, process_ok) = unsafe {
            (
                GetSystemTimes(&mut idle, &mut kernel, &mut user),
                GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut own_kernel, &mut own_user),
            )
        };
        let at = Instant::now();
        if system_ok == 0 {
            return Err(format!("GetSystemTimes failed: {}", std::io::Error::last_os_error()));
        }
        if process_ok == 0 {
            return Err(format!("GetProcessTimes failed: {}", std::io::Error::last_os_error()));
        }
        // The kernel time GetSystemTimes reports includes idle time.
        let total = kernel.units() + user.units();
        let busy = match total.checked_sub(idle.units()) {
            Some(busy) => busy,
            None => {
                return Err(format!(
                    "GetSystemTimes reported idle time {} above kernel plus user time {total}",
                    idle.units()
                ));
            }
        };
        Ok(Sample { at, busy, own: own_kernel.units() + own_user.units() })
    }

    /// Cores the box and this process were busy between two samples.
    pub fn cores_between(earlier: &Sample, later: &Sample) -> Result<(f64, f64), String> {
        let wall_units = later.at.duration_since(earlier.at).as_nanos() as f64 / 100.0;
        if wall_units <= 0.0 {
            return Err("the two CPU samples were taken at the same instant".to_string());
        }
        let busy = super::advanced("the box's busy time", earlier.busy, later.busy)? as f64;
        let own = super::advanced("this process's CPU time", earlier.own, later.own)? as f64;
        Ok((busy / wall_units, own / wall_units))
    }
}

/// CPU accounting on Linux: ticks from /proc/stat and /proc/self/stat,
/// turned into cores by the share of every CPU's summed ticks, so no
/// ticks-per-second constant is assumed.
#[cfg(target_os = "linux")]
mod cpu {
    use std::fs;

    pub struct Sample {
        busy: u64,
        own: u64,
        total: u64,
        cpus: u32,
    }

    fn parse(field: Option<&str>, what: &str) -> Result<u64, String> {
        match field {
            Some(text) => text.parse::<u64>().map_err(|err| format!("{what} is not a count: {text:?} ({err})")),
            None => Err(format!("{what} is missing")),
        }
    }

    pub fn read() -> Result<Sample, String> {
        let stat = fs::read_to_string("/proc/stat").map_err(|err| format!("reading /proc/stat: {err}"))?;
        let Some(line) = stat.lines().next() else {
            return Err("/proc/stat is empty".to_string());
        };
        // user nice system idle iowait irq softirq steal; guest time is
        // already inside user and nice.
        let mut fields = line.split_whitespace().skip(1);
        let mut ticks = [0u64; 8];
        for (i, name) in ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"].into_iter().enumerate() {
            ticks[i] = parse(fields.next(), name)?;
        }
        let total: u64 = ticks.iter().sum();
        let busy = total - ticks[3] - ticks[4];
        let cpus = stat
            .lines()
            .filter(|l| l.starts_with("cpu") && l.as_bytes().get(3).is_some_and(u8::is_ascii_digit))
            .count() as u32;

        let own_stat = fs::read_to_string("/proc/self/stat").map_err(|err| format!("reading /proc/self/stat: {err}"))?;
        // The command name is parenthesised and may hold spaces, so fields
        // are counted from its closing parenthesis: utime and stime are the
        // 14th and 15th fields of the line, the 12th and 13th after it.
        let Some(close) = own_stat.rfind(')') else {
            return Err("/proc/self/stat has no command name".to_string());
        };
        let mut rest = own_stat[close + 1..].split_whitespace().skip(11);
        let utime = parse(rest.next(), "utime")?;
        let stime = parse(rest.next(), "stime")?;
        Ok(Sample { busy, own: utime + stime, total, cpus })
    }

    /// Cores the box and this process were busy between two samples.
    pub fn cores_between(earlier: &Sample, later: &Sample) -> Result<(f64, f64), String> {
        let total = super::advanced("the summed CPU ticks", earlier.total, later.total)? as f64;
        if total <= 0.0 {
            return Err("no CPU ticks passed between the two samples".to_string());
        }
        if later.cpus == 0 {
            return Err("/proc/stat lists no CPUs".to_string());
        }
        let cpus = f64::from(later.cpus);
        let busy = super::advanced("the box's busy ticks", earlier.busy, later.busy)? as f64;
        let own = super::advanced("this process's ticks", earlier.own, later.own)? as f64;
        Ok((busy / total * cpus, own / total * cpus))
    }
}

/// No CPU accounting on other platforms: every row reports why.
#[cfg(not(any(windows, target_os = "linux")))]
mod cpu {
    pub struct Sample;

    pub fn read() -> Result<Sample, String> {
        Err("this example reads CPU accounting on Windows and Linux only".to_string())
    }

    pub fn cores_between(_earlier: &Sample, _later: &Sample) -> Result<(f64, f64), String> {
        Err("this example reads CPU accounting on Windows and Linux only".to_string())
    }
}

/// A row's CPU context: the cores the box and this process were busy over
/// the row's interval, or why they could not be read.
enum CpuContext {
    Measured { busy: f64, own: f64 },
    Unavailable(String),
}

/// The CPU context between a row's two samples, carrying the reason each
/// failed sample or failed difference gave.
fn row_context(earlier: &Result<cpu::Sample, String>, later: &Result<cpu::Sample, String>) -> CpuContext {
    match earlier {
        Ok(first) => match later {
            Ok(last) => match cpu::cores_between(first, last) {
                Ok((busy, own)) => CpuContext::Measured { busy, own },
                Err(reason) => CpuContext::Unavailable(reason),
            },
            Err(last_reason) => CpuContext::Unavailable(format!("sample at the row's end: {last_reason}")),
        },
        Err(first_reason) => match later {
            Ok(..) => CpuContext::Unavailable(format!("sample at the row's start: {first_reason}")),
            Err(last_reason) => CpuContext::Unavailable(format!(
                "sample at the row's start: {first_reason}; sample at the row's end: {last_reason}"
            )),
        },
    }
}

/// Where a row's interval lies against the load window.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Before,
    Load,
    After,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Phase::Before => "before",
            Phase::Load => "load",
            Phase::After => "after",
        }
    }
}

/// One printed row, kept for the closing report.
struct Row {
    phase: Phase,
    learned: Option<WorkloadClass>,
    cpu: CpuContext,
}

/// "-" when no row stands at a level, and otherwise whether any did.
fn yes_no(rows: usize, any: bool) -> &'static str {
    if rows == 0 {
        "-"
    } else if any {
        "yes"
    } else {
        "no"
    }
}

/// Print whether an answer holds at every level, at none, or at which.
fn verdict(name: &str, yes: &str, no: &str, answers: &[(f64, bool)]) {
    let holds: Vec<f64> = answers.iter().filter(|answer| answer.1).map(|answer| answer.0).collect();
    let fails: Vec<f64> = answers.iter().filter(|answer| !answer.1).map(|answer| answer.0).collect();
    if answers.is_empty() {
        println!("{name}: no level had the rows to answer it");
    } else if fails.is_empty() {
        println!(
            "{name}: {yes} at every level with rows, foreign <= {:.1} to {:.1} cores ({} levels)",
            answers[0].0,
            answers[answers.len() - 1].0,
            answers.len()
        );
    } else if holds.is_empty() {
        println!(
            "{name}: {no} at every level with rows, foreign <= {:.1} to {:.1} cores ({} levels)",
            answers[0].0,
            answers[answers.len() - 1].0,
            answers.len()
        );
    } else {
        println!("{name}: {yes} at foreign <= {holds:?}; {no} at foreign <= {fails:?}");
    }
}

/// For the rows at or below each foreign-load level, in half-core steps up
/// to the highest level seen: whether the site left Streaming before the
/// load, whether it left under the load, and its class at the last row
/// after the load. Every reason a row had no CPU context is listed with
/// its count.
fn report(rows: &[Row]) {
    let mut measured: Vec<(&Row, f64)> = Vec::new();
    let mut unavailable: BTreeMap<&str, usize> = BTreeMap::new();
    for row in rows {
        match &row.cpu {
            CpuContext::Measured { busy, own } => measured.push((row, busy - own)),
            CpuContext::Unavailable(reason) => *unavailable.entry(reason.as_str()).or_insert(0) += 1,
        }
    }
    println!();
    for (reason, count) in &unavailable {
        println!("{count} rows had no CPU context: {reason}");
    }
    if measured.is_empty() {
        println!("no row carried CPU context, so no level can be reported");
        return;
    }
    let mut highest = f64::MIN;
    let mut lowest = f64::MAX;
    for entry in &measured {
        highest = highest.max(entry.1);
        lowest = lowest.min(entry.1);
    }
    println!(
        "foreign load across the {} rows with CPU context: {lowest:.2} to {highest:.2} cores; levels below {lowest:.2} hold no rows",
        measured.len()
    );
    println!("foreign_le  before_rows  before_left  load_rows  load_left  after_rows  after_last_class");
    let left = |row: &Row| matches!(row.learned, Some(class) if class != WorkloadClass::Streaming);
    let mut attribution = Vec::new();
    let mut persistence = Vec::new();
    let steps = (highest / 0.5).ceil().max(1.0) as usize;
    for step in 1..=steps {
        let level = step as f64 * 0.5;
        let clean: Vec<&Row> = measured.iter().filter(|entry| entry.1 <= level).map(|entry| entry.0).collect();
        let before: Vec<&Row> = clean.iter().copied().filter(|row| row.phase == Phase::Before).collect();
        let load: Vec<&Row> = clean.iter().copied().filter(|row| row.phase == Phase::Load).collect();
        let after: Vec<&Row> = clean.iter().copied().filter(|row| row.phase == Phase::After).collect();
        let before_left = before.iter().any(|row| left(row));
        let load_left = load.iter().any(|row| left(row));
        let after_last = match after.last() {
            Some(row) => match row.learned {
                Some(class) => format!("{class:?}"),
                None => "none".to_string(),
            },
            None => "-".to_string(),
        };
        println!(
            "{level:10.1}  {:11}  {:>11}  {:9}  {:>9}  {:10}  {:>16}",
            before.len(),
            yes_no(before.len(), before_left),
            load.len(),
            yes_no(load.len(), load_left),
            after.len(),
            after_last,
        );
        if !before.is_empty() && !load.is_empty() {
            attribution.push((level, !before_left && load_left));
        }
        if let Some(row) = after.last() {
            persistence.push((level, row.learned == Some(WorkloadClass::Streaming)));
        }
    }
    verdict(
        "attribution",
        "the site left Streaming only under the run's own load",
        "the site's leaving Streaming is not attributable to the run's own load",
        &attribution,
    );
    verdict(
        "persistence",
        "the site was back in Streaming after the load",
        "the site was still migrated after the load",
        &persistence,
    );
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
    // give a smaller floor and a finer split.
    let min_leaf: usize = arg(5, 1_024);
    let load_at: u64 = arg(6, 40);
    let load_for: u64 = arg(7, 60);
    let load_threads: usize = if env::args().len() > 8 {
        arg(8, 0)
    } else {
        match std::thread::available_parallelism() {
            Ok(n) => n.get(),
            Err(err) => {
                eprintln!("available_parallelism is unavailable ({err}); pass the load thread count as argument 8");
                std::process::exit(2);
            }
        }
    };

    let routing = match env::args().nth(9).as_deref() {
        None | Some("adaptive") => Routing::Adaptive,
        Some("pinned") => Routing::Pinned,
        Some(other) => {
            eprintln!("argument 9 is the routing, adaptive or pinned, not {other:?}");
            std::process::exit(2);
        }
    };

    println!(
        "seconds {seconds}  report_every {report_every}s  items {items}  rounds {rounds}  \
         min_leaf {min_leaf}  load_at {load_at}s  load_for {load_for}s  load_threads {load_threads}  routing {}",
        routing.name()
    );
    // leaf_sizes is the precondition: this experiment needs leaves whose
    // own variance is near zero, and a run whose leaves came out at mixed
    // or few-item sizes cannot answer it however clean the class column
    // looks.
    println!(
        "elapsed_s  phase   dispatches  sampled_leaves  cv2_per_mille  window_mean_ticks  window_cv2  \
         site_class  global_class  last_ms  box_cores  own_cores  foreign_cores  leaf_sizes"
    );

    let site = SiteRef::new(&SITE);
    let mut buf: Vec<u64> = (0..items as u64).collect();

    let start = Instant::now();
    let load_begin = start + Duration::from_secs(load_at);
    let load_end = load_begin + Duration::from_secs(load_for);
    let stop = Arc::new(AtomicBool::new(false));
    let burners: Vec<_> = (0..load_threads)
        .map(|t| {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while Instant::now() < load_begin {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                let mut x = t as u64 + 1;
                while Instant::now() < load_end && !stop.load(Ordering::Relaxed) {
                    x = black_box(item_work(x, 4_096));
                }
            })
        })
        .collect();

    let mut previous_cpu = cpu::read();
    let mut previous_leaf_sizes = read_leaf_sizes();
    let mut rows = Vec::new();
    let mut row_start = start;
    let mut next_report = Duration::from_secs(0);
    let mut dispatches: u64 = 0;
    let mut last_ms;

    while start.elapsed() < Duration::from_secs(seconds) {
        let t0 = Instant::now();
        let plan = match routing {
            Routing::Adaptive => JobPlan::new(0, buf.len() as u32),
            Routing::Pinned => JobPlan::set_profile(0, buf.len() as u32, DispatchProfile::Streaming),
        }
        .with_site(site);
        for_each_chunk_min_leaf(&plan, &mut buf, min_leaf, |slice| {
            let leaf_items = slice.len();
            let leaf_start = Instant::now();
            for x in slice {
                *x = item_work(*x, rounds);
            }
            record_leaf_size(leaf_items, leaf_start.elapsed().as_nanos() as u64);
        });
        black_box(buf[0]);
        last_ms = t0.elapsed().as_secs_f64() * 1000.0;
        dispatches += 1;

        if start.elapsed() >= next_report {
            let now = Instant::now();
            let elapsed_s = start.elapsed().as_secs_f64();
            let view = read_site(&SITE);
            let later_cpu = cpu::read();
            let context = row_context(&previous_cpu, &later_cpu);
            previous_cpu = later_cpu;
            let later_leaf_sizes = read_leaf_sizes();
            let leaf_sizes = match leaf_sizes_between(&previous_leaf_sizes, &later_leaf_sizes) {
                Ok(text) => text,
                Err(reason) => {
                    eprintln!("row at {elapsed_s:.1}s has no leaf sizes: {reason}");
                    "unavailable".to_string()
                }
            };
            previous_leaf_sizes = later_leaf_sizes;
            let phase = if now <= load_begin {
                Phase::Before
            } else if row_start >= load_end {
                Phase::After
            } else {
                Phase::Load
            };
            // `none` is printed as itself rather than as a number. A
            // site below the sample floor has not classified, which is a
            // different statement from having classified as anything.
            let cv2 = match view.cv2 {
                Some(v) => v.to_string(),
                None => "none".to_string(),
            };
            let window_mean = match view.window_mean_ticks {
                Some(v) => v.to_string(),
                None => "none".to_string(),
            };
            let window_cv2 = match view.window_cv2 {
                Some(v) => v.to_string(),
                None => "none".to_string(),
            };
            let learned = match view.learned {
                Some(c) => format!("{c:?}"),
                None => "none".to_string(),
            };
            let (box_text, own_text, foreign_text) = match &context {
                CpuContext::Measured { busy, own } => {
                    (format!("{busy:.2}"), format!("{own:.2}"), format!("{:.2}", busy - own))
                }
                CpuContext::Unavailable(reason) => {
                    eprintln!("row at {elapsed_s:.1}s has no CPU context: {reason}");
                    ("unavailable".to_string(), "unavailable".to_string(), "unavailable".to_string())
                }
            };
            println!(
                "{:9.1}  {:6}  {:10}  {:>14}  {:>13}  {:>17}  {:>10}  {:>12}  {:>12}  {:7.2}  {:>11}  {:>11}  {:>13}  {}",
                elapsed_s,
                phase.name(),
                dispatches,
                view.sampled_leaves,
                cv2,
                window_mean,
                window_cv2,
                learned,
                format!("{:?}", active_workload_class()),
                last_ms,
                box_text,
                own_text,
                foreign_text,
                leaf_sizes,
            );
            rows.push(Row { phase, learned: view.learned, cpu: context });
            row_start = now;
            next_report = start.elapsed() + Duration::from_secs(report_every);
        }
    }

    stop.store(true, Ordering::Relaxed);
    for (t, burner) in burners.into_iter().enumerate() {
        if let Err(panic) = burner.join() {
            eprintln!("load thread {t} panicked: {panic:?}");
        }
    }
    println!("done after {dispatches} dispatches");
    report(&rows);
}

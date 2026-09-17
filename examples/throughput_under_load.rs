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
//! came out as, how the resolved width compares to the width the
//! process is allowed, and where the spin window ended up. A switch
//! whose figures are identical across its own on and off arms did not
//! engage, and its throughput row says nothing about the mechanism.
//!
//! # Why the per-item cost has to be large
//!
//! `classify_observed` returns on the mean alone below `port_heavy_ns`
//! and never reads the variance. So a workload cheaper than that per
//! item cannot show ANY spread-driven mechanism doing anything,
//! however irregular it is: the branch that would read the spread is
//! not taken. The run says so after its warm window rather than
//! producing rows that read as a mechanism with no effect.
//!
//! # Why the load can alternate
//!
//! `duty_ms` makes the burners spin and sleep in phase rather than burn
//! throughout. Steady contention gives every batch about the same
//! on-core share, and a statistic that weighs batches by that share
//! divides a weighted total by a weighted count, so a share common to
//! every batch cancels and the figure does not move. Alternating puts
//! contended and quiet batches in one window, which is the condition
//! such a weighting exists for.
//!
//! Each trial measures twice in this process: a control window with the
//! burners off and a loaded window with them on. It prints a
//! `throughput` row for the loaded window, a `control` row for the quiet
//! one, and a `retained` row giving loaded dispatches as a share of
//! quiet ones. A trial therefore takes twice `window_s` plus the
//! burners' settling time.
//!
//! `retained` is the figure a change has to hold or improve, and it is a
//! ratio taken inside one process, so pool startup, the calibration draw
//! and the machine's state at that moment are common to both arms
//! instead of dividing one process by another.
//!
//! The last argument picks the dispatch entry: `plain` (the default),
//! `indexed` or `triple`. They do not share a leaf recorder, so a figure
//! taken through one says nothing about the others. `plain` reaches
//! `record_leaf_sampled`; the other two reach
//! `record_leaf_bracket_sampled`, which is the only path where the
//! on-core bracket's cost can be weighed.
//!
//! ```sh
//! throughput_under_load <window_s> <load_threads> <trials> [smt_prior] \
//!     [duty_ms] [reps] [irregular] [entry]
//! ```

use std::env;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use flynnel::sched::par_iter::{
    for_each_chunk_indexed_min_leaf, for_each_chunk_min_leaf, for_each_chunk_triple_min_leaf,
};
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

/// One item: how many rounds it costs, and what it accumulates.
///
/// The cost travels alongside the item rather than being derived from
/// its index, so the leaf body needs no index and the dispatch can go
/// through `for_each_chunk_min_leaf`. That entry matters: the on-core
/// bracket is taken by `record_leaf_sampled`, which only the plain
/// steal-driven bisect calls. The indexed and triple bisects record
/// every leaf through `record_leaf`, which has no bracket, so a
/// harness dispatching through them cannot reach the on-core switch at
/// all.
///
/// Deriving it from the accumulator instead would drift: every
/// dispatch rewrites the buffer, so two arms that completed different
/// numbers of dispatches would be running different work.
#[derive(Clone, Copy)]
struct Item {
    reps: u32,
    acc: u64,
}

/// How much work the item at `index` costs.
///
/// Uniform gives every item `reps` rounds. Irregular varies them
/// deterministically with the index, in `1 ..= 2 * reps - 1`, so the
/// mean is `reps` and the shapes are comparable in total work while
/// differing in spread.
#[inline]
fn reps_at(index: usize, reps: u32, irregular: bool) -> u32 {
    if !irregular || reps < 2 {
        return reps;
    }
    let h = (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    // Saturating, so a reps above half of u32 yields a narrower span
    // rather than wrapping to a small one in release.
    let span = reps.saturating_mul(2).saturating_sub(1).max(1);
    1 + (h >> 33) as u32 % span
}

/// Which dispatch entry a run measures.
///
/// The entries do not share a leaf recorder, so a figure taken through
/// one says nothing about the others. `Plain` reaches
/// `record_leaf_sampled`; `Indexed` and `Triple` reach
/// `record_leaf_bracket_sampled`, which times every leaf and brackets
/// one in the stride. A run that means to weigh the bracket has to ask
/// for an entry that can reach it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Entry {
    Plain,
    Indexed,
    Triple,
}

impl Entry {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "plain" => Some(Self::Plain),
            "indexed" => Some(Self::Indexed),
            "triple" => Some(Self::Triple),
            _ => None,
        }
    }
}

#[inline]
fn grind(slot: &mut Item) {
    let mut acc = slot.acc;
    for _ in 0..slot.reps {
        acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    }
    slot.acc = black_box(acc);
}

/// Dispatches completed in `measured`, with whatever else is running.
///
/// Every entry runs the same per-item body over the same buffer, so a
/// difference between two entries is the dispatch path and not the work.
fn window(
    buf: &mut [Item],
    aux: &mut [Item],
    measured: Duration,
    smt_prior: bool,
    entry: Entry,
) -> u64 {
    let start = Instant::now();
    let mut dispatches = 0u64;
    while start.elapsed() < measured {
        let plan = plan(smt_prior);
        match entry {
            Entry::Plain => {
                for_each_chunk_min_leaf(&plan, buf, MIN_LEAF, |chunk| {
                    for slot in chunk.iter_mut() {
                        grind(slot);
                    }
                });
            }
            Entry::Indexed => {
                for_each_chunk_indexed_min_leaf(&plan, buf, MIN_LEAF, |_start, chunk| {
                    for slot in chunk.iter_mut() {
                        grind(slot);
                    }
                });
            }
            Entry::Triple => {
                // The triple entry writes `out` from two reads, so the
                // body differs in shape from the others by necessity.
                // The rounds per item are the same, which is what the
                // per-item cost is made of.
                for_each_chunk_triple_min_leaf(
                    &plan,
                    aux,
                    buf,
                    buf,
                    MIN_LEAF,
                    |out, a, _b| {
                        for i in 0..out.len() {
                            out[i].reps = a[i].reps;
                            out[i].acc = a[i].acc;
                            grind(&mut out[i]);
                        }
                    },
                );
            }
        }
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
    /// `duty_ms` of zero burns for the whole window. Any other value
    /// alternates: every burner spins for that many milliseconds and
    /// sleeps for the same, all of them off one shared origin so they
    /// stay in phase. Staggered burners would sum to a steady load,
    /// which is the one shape the batch weighting cannot read.
    fn start(n: usize, duty_ms: u64) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let origin = Instant::now();
        // The period as a type that cannot be zero, so the burner loop
        // divides without a guard beside it.
        let period = std::num::NonZeroU64::new(duty_ms);
        let mut threads = Vec::with_capacity(n);
        for _ in 0..n {
            let stop = Arc::clone(&stop);
            threads.push(std::thread::spawn(move || {
                let mut acc = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    if let Some(period) = period {
                        let phase = origin.elapsed().as_millis() as u64 / period.get();
                        if phase % 2 == 1 {
                            std::thread::sleep(Duration::from_millis(1));
                            continue;
                        }
                    }
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
fn engagement(smt_prior: bool, duty_ms: u64, reps: u32, irregular: bool) {
    let plan = plan(smt_prior);
    let site = &SITE;
    println!(
        "engagement duty_ms={duty_ms} reps={reps} irregular={irregular} \
         leaves={} oncore_items={} per_item_ns={} cv2_wall={} \
         cv2_oncore={} cv2_window={} window_ticks={} class={:?} workers={} allowed={} smt={} \
         spin_adaptive={} spin_window={} idle_yields={}",
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
        flynnel::sched::spin_adaptive(),
        flynnel::sched::spin_window(),
        flynnel::sched::total_idle_yields(),
    );
}

fn main() {
    let window_s: u64 = arg(1, 5);
    let load_threads: usize = arg(2, 12);
    let trials: usize = arg(3, 5);
    let smt_prior: bool = arg::<u8>(4, 0) != 0;
    let duty_ms: u64 = arg(5, 0);
    let reps: u32 = arg(6, 16);
    let irregular: bool = arg::<u8>(7, 0) != 0;
    let entry_name: String = arg(8, "plain".to_string());
    let Some(entry) = Entry::parse(&entry_name) else {
        eprintln!("entry must be plain, indexed or triple, and was {entry_name:?}");
        std::process::exit(2);
    };

    if reps == 0 {
        eprintln!("reps must be at least one, or the leaf body does nothing");
        std::process::exit(2);
    }

    // Which arm this process is. Printed rather than assumed, because a
    // switch that failed to engage produces rows indistinguishable from
    // the other arm's, and the driver's label would be the only record
    // of an intent that did not take effect.
    eprintln!("levers: {}", flynnel::sched::levers::describe());
    // The entries do not share a leaf recorder, so a row means nothing
    // without knowing which one produced it. Printed on both streams
    // because the rows go to stdout and the switches to stderr, and a
    // reader may have only one of them.
    eprintln!("entry {entry_name}");
    println!("entry {entry_name}");

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
    // The per-item cost is fixed here, once, so every dispatch of every
    // arm runs the same work whatever order the buffer reaches.
    let mut buf: Vec<Item> = (0..ITEMS)
        .map(|i| Item { reps: reps_at(i, reps, irregular), acc: i as u64 })
        .collect();
    // The triple entry needs somewhere to write. Allocated for every
    // entry so the process's memory is the same whichever one runs, and
    // a comparison between entries is not also a comparison between
    // allocations.
    let mut aux: Vec<Item> = vec![Item { reps: 1, acc: 0 }; ITEMS];

    // One warm window, discarded: the first dispatches of a process pay
    // pool startup and the site's first classifier ticks, which belong
    // to no arm.
    let warm = window(&mut buf, &mut aux, Duration::from_secs(1), smt_prior, entry);
    if warm == 0 {
        eprintln!("the warm window ran no dispatches; raise the window length");
        std::process::exit(2);
    }

    // classify_observed returns on the mean alone below port_heavy_ns
    // and never reads the variance. Every switch here that acts on a
    // spread is therefore unreachable at a per-item cost under that
    // threshold, whatever the spread is, and a run below it produces
    // rows that look exactly like a mechanism that did not help.
    let heavy = flynnel::sched::adaptive_profile::class_thresholds()
        .port_heavy_ns
        .load(std::sync::atomic::Ordering::Relaxed);
    match SITE.per_item_ns() {
        Some(ns) if ns < heavy => eprintln!(
            "this workload costs {ns} ns an item and classify_observed reads the \
             variance only at {heavy} and above, so no spread-driven switch can \
             change a class here; raise reps"
        ),
        Some(_) => {}
        None => eprintln!(
            "the site reported no per-item cost after the warm window, so whether \
             the variance branch is reachable is unknown"
        ),
    }

    // Each trial measures both arms in this process: a control window
    // with the burners off and a loaded window with them on. Two
    // processes would carry their own pool startup, their own
    // calibration draw and their own place in the machine's day, and a
    // ratio between them would divide by all of it.
    //
    // The arms alternate by trial so neither always runs first. A window
    // is worth a few per cent more in one position than the other, and
    // an order held fixed would put that difference in the ratio.
    for t in 1..=trials {
        let control_first = t % 2 == 1;

        let mut control = 0u64;
        let mut loaded = 0u64;

        for half in 0..2 {
            let run_control = (half == 0) == control_first;
            if run_control {
                control = window(&mut buf, &mut aux, measured, smt_prior, entry);
            } else {
                let load = (load_threads > 0).then(|| Load::start(load_threads, duty_ms));
                // Let the burners reach the cores before timing, or the
                // early part of the window carries less load than the
                // row claims.
                if load.is_some() {
                    std::thread::sleep(Duration::from_millis(250));
                }
                loaded = window(&mut buf, &mut aux, measured, smt_prior, entry);
                if let Some(load) = load {
                    load.stop();
                }
            }
        }

        let control_per_s = control as f64 / window_s as f64;
        let loaded_per_s = loaded as f64 / window_s as f64;
        // Dispatches under load as a share of dispatches quiet. This is
        // what a change has to hold or improve, and it is a ratio within
        // one process rather than across two.
        let retained = if control > 0 { loaded as f64 / control as f64 } else { 0.0 };

        println!("throughput {load_threads} {t} {loaded} {loaded_per_s:.2}");
        println!("control {load_threads} {t} {control} {control_per_s:.2}");
        println!(
            "retained {load_threads} {t} {retained:.4} control_first={control_first}"
        );
    }

    engagement(smt_prior, duty_ms, reps, irregular);

    // The on-core switch is read inside record_leaf_sampled, and only
    // the plain steal-driven bisect calls that. A dispatch entry that
    // records every leaf through record_leaf takes no bracket, so the
    // switch is on and nothing happens - and the rows look exactly like
    // a mechanism that did not help. An arm that asked for it and took
    // no reading says so rather than being read as a result.
    if flynnel::sched::levers::oncore_spread() && SITE.oncore_items() == 0 {
        eprintln!(
            "the on-core switch is on and no leaf carried an on-core reading, so the \
             {entry_name} entry never reached a recorder that brackets and the switch \
             did nothing here"
        );
        std::process::exit(4);
    }
}

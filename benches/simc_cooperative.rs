//! A/B microbench for the SIMC primitive
//! ([`cooperative_join_n_flat`]) against rayon's closest equivalent
//! (`rayon::scope::spawn` fan-out).
//!
//! ## Why this bench exists
//!
//! The Flynn-axis taxonomy in `crate::backend::mod`'s doc table
//! assigns `cooperative_join_n` as the SIMC (Single Instruction,
//! Multiple Cores) primitive. Its win zone is N independent uniform-
//! cost closures fanned out across N cores as one logical mega-
//! SIMD vector. The flat-shape variant
//! [`cooperative_join_n_flat`] pushes each closure directly to a
//! specific peer worker's mailbox (URD-style owner-directed
//! distribution); the target worker drains its mailbox ahead of any
//! deque in `find_work`, so each closure starts on its assigned core with
//! no CAS contention on a shared deque head.
//!
//! Rayon's `scope::spawn` fans out via the shared deque + random
//! peer-steal. Every closure pushed by the calling thread lands
//! on one shared deque; thieves race to grab them. Cross-CCX peers
//! can pull a closure that was pushed from a core that shares L1d
//! with a different sibling - the cache hit-rate is random.
//!
//! ## Bench-audit (HARD RULE 3)
//!
//! - **Same payload**: each closure does the same fixed amount of
//!   pure CPU work (a 1000-iteration u64 xorshift mixer) so the
//!   comparison measures dispatch + steal latency, not workload
//!   variance.
//! - **Same N within a group**: every arm of a group fans out the same
//!   number of closures, so a group compares shapes and nothing else.
//!   The sweep runs fourteen widths - 8 through 1024 - which on a
//!   24-worker host straddle the mailbox gate at 24 and reach 43x the
//!   pool. Widths below the gate are the control: both
//!   flynnel arms run the same code there, so a group that does not
//!   show them agreeing is not measuring what it claims.
//! - **Same result-collection**: both halves materialise a Vec<u64>
//!   in caller order so the bench measures equivalent total work
//!   including the result-gather phase.
//! - **The primitive's named feature is exercised**:
//!   cooperative_join_n_flat's mailbox-distribute path fires at
//!   N >= the worker count. The bench calls cooperative_join_n_flat
//!   directly; its
//!   internal fan_out_external path wraps in a parent StackJob
//!   submitted onto the global arena so the inner fan_out_in_worker
//!   call runs on a Flynnel worker thread (per current_worker_ctx).
//!
//! ## Every size is registered twice, in opposite arm order
//!
//! Criterion measures arms sequentially, so a load that arrives or
//! departs partway through a group shifts the means of the arms it
//! covers and not the others. Confidence intervals do not defend
//! against this: they describe spread within an arm, so two arms can
//! have tight disjoint intervals and still be separated by the machine
//! rather than by the code.
//!
//! A ratio that holds in both orders is the code. One that appears in
//! one order and is absent or reversed in the other is the host, and
//! the group is unreadable rather than merely noisy. This is what lets
//! the sweep be read on a shared host, which is the only kind
//! available here.
//!
//! ## The two stall instruments
//!
//! Both arm a watchdog that prints what a dispatch reached when it
//! stops advancing and then ends the process. They differ in what they
//! cost a closure, and that difference decides which one can see this
//! bench's stall at all.
//!
//! `FLYNNEL_BENCH_STALL_REPORT=1` records, in every closure, that it
//! started and finished. It answers a stall with sets rather than an
//! event log: the indices that never started, the ones that started and
//! never finished, and the threads that ran anything. Those are bounded
//! by the fan-out width whatever the iteration count, where an event
//! log is bounded by iterations times width, which at this bench's
//! counts is tens of millions of records across forty-nine threads.
//!
//! Reading it: indices that never started with workers missing from the
//! thread list is work nothing was woken to take; indices that never
//! started with every worker present is work routed where none of them
//! looks; started-and-unfinished is a closure that is running or a
//! thread that died inside one.
//!
//! `FLYNNEL_BENCH_STALL_CENSUS=1` records nothing inside a closure. It
//! stamps a generation once per dispatch, on the calling thread before
//! the fan-out, and on a stall reports which arm stopped, at what
//! width, and which mailboxes still hold work.
//!
//! The census exists because the report suppresses what it is pointed
//! at. Nine of twelve unarmed runs of the n1024 group stall, every one
//! in the mailbox or routed arm and none in the deque arm; none of
//! eleven runs armed with the report stalls at all. Two atomic stores
//! and a lock per closure, across 1024 closures and sixteen workers, is
//! enough to close whatever window this needs. The census writes once
//! per dispatch instead of once per closure, and touches no worker
//! thread, so a run carrying it is still a run that can stall.
//!
//! Neither is on by default and neither measures: a run with either one
//! diagnoses. Prefer the census first, because a stall it catches is a
//! stall the report would have prevented.

#![allow(clippy::missing_docs_in_private_items)]

use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::sched::cooperative::{
    cooperative_join_n, cooperative_join_n_flat, cooperative_join_n_flat_mailbox,
};
use flynnel::sched::plan::JobPlan;

/// Closure body: deterministic, fixed-cost CPU work. Each call
/// runs a 1000-iter xorshift mixer + returns the final value so
/// the optimizer can't elide the loop.
#[inline(never)]
fn fixed_cost_work(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..1000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x = x.wrapping_mul(0x100000001B3);
    }
    x
}

/// What the stall instrument records.
///
/// Ordered by what it costs a closure, which is the axis that decides
/// whether a run carrying it can still stall.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Instrument {
    /// Nothing recorded and no watchdog. What a measuring run uses.
    Off,
    /// A generation stamped once per dispatch, on the calling thread,
    /// and a watchdog. Nothing inside a closure and nothing written by
    /// a worker.
    Census,
    /// The census plus two atomic stores in every closure and a lock
    /// once per thread per dispatch, which is what names the indices
    /// that never ran, and what suppresses the stall.
    Report,
}

/// Whether `name` is set to an affirmative value.
///
/// An unset variable is off, which is the default and not a failure. A
/// variable set to bytes that are not unicode is neither: it was set
/// deliberately and cannot be read, so it says so rather than reading
/// as unset and leaving a diagnostic run silently unarmed.
fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => value == "1" || value == "true",
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(raw)) => {
            eprintln!("{name} is set to {raw:?}, which is not unicode; leaving it off");
            false
        }
    }
}

/// Which instrument the environment arms.
///
/// Read once. `FLYNNEL_BENCH_STALL_REPORT` wins over
/// `FLYNNEL_BENCH_STALL_CENSUS` when both are set, because the report
/// is the census plus the per-closure marking: a run asking for both
/// gets the one that records more, and takes the report's timings and
/// the report's odds of stalling with it.
fn instrument() -> Instrument {
    static ARMED: OnceLock<Instrument> = OnceLock::new();
    *ARMED.get_or_init(|| {
        if env_flag("FLYNNEL_BENCH_STALL_REPORT") {
            Instrument::Report
        } else if env_flag("FLYNNEL_BENCH_STALL_CENSUS") {
            Instrument::Census
        } else {
            Instrument::Off
        }
    })
}

/// Which closures of the dispatch in flight have started and finished,
/// and which threads ran one.
///
/// A stalled fan-out is diagnosed by a set rather than a sequence: the
/// indices that never started say whether work was routed somewhere
/// nothing drains, and the threads that never appear say which workers
/// never woke. Both are bounded by the fan-out width however many
/// iterations run, which an event log is not.
struct StallWatch {
    /// The dispatch in which each index last started, and last finished.
    ///
    /// A generation stamp rather than a flag so that opening a dispatch
    /// costs nothing. Clearing 1024 atomics and two locks per iteration
    /// would be paid thousands of times a second by the armed run, and
    /// a race that needs a particular interleaving is exactly what such
    /// a cost suppresses - an instrument that removes what it is
    /// pointed at reports the absence as a finding.
    started: Vec<AtomicU64>,
    finished: Vec<AtomicU64>,
    /// Each thread that has run a closure, against the last dispatch it
    /// ran one in.
    threads: Mutex<BTreeMap<String, u64>>,
    /// Incremented at every dispatch entry. The watchdog reports only
    /// when it reads the same value twice a stall apart, so a slow run
    /// that keeps dispatching is never mistaken for a stopped one.
    generation: AtomicU64,
    /// The width of the dispatch in flight, which is not the vector
    /// length: the vectors are sized once to the widest fan-out the
    /// sweep registers.
    width: AtomicUsize,
    arm: Mutex<String>,
}

thread_local! {
    /// The last dispatch this thread announced itself in.
    ///
    /// Keeps the threads lock to one acquisition per thread per
    /// dispatch instead of one per closure.
    static ANNOUNCED_IN: std::cell::Cell<u64> = const { std::cell::Cell::new(u64::MAX) };
}

/// The generation no dispatch has, so a stamp left by no run reads as
/// absent rather than as dispatch zero.
const NO_DISPATCH: u64 = 0;

/// The widest fan-out any group registers.
const WIDEST_FAN_OUT: usize = 1024;

/// How long without a new dispatch counts as stalled.
const STALL_AFTER: Duration = Duration::from_secs(25);

fn stall_watch() -> &'static StallWatch {
    static WATCH: OnceLock<StallWatch> = OnceLock::new();
    WATCH.get_or_init(|| StallWatch {
        started: (0..WIDEST_FAN_OUT)
            .map(|_| AtomicU64::new(NO_DISPATCH))
            .collect(),
        finished: (0..WIDEST_FAN_OUT)
            .map(|_| AtomicU64::new(NO_DISPATCH))
            .collect(),
        threads: Mutex::new(BTreeMap::new()),
        generation: AtomicU64::new(NO_DISPATCH),
        width: AtomicUsize::new(0),
        arm: Mutex::new(String::new()),
    })
}

impl StallWatch {
    /// Open a new dispatch.
    ///
    /// The arm name and width are written before the generation, so a
    /// closure that reads the new generation finds the two describing
    /// its own dispatch.
    fn begin(&self, arm: &str, n: usize) {
        {
            let mut held = self.arm.lock().expect("stall watch arm");
            if *held != arm {
                held.clear();
                held.push_str(arm);
            }
        }
        self.width.store(n, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Release);
    }

    fn record_start(&self, i: usize) {
        let generation = self.generation.load(Ordering::Acquire);
        if i < WIDEST_FAN_OUT {
            // Stamped before the thread name, so an index carrying this
            // dispatch with its runner absent from the thread list is a
            // closure that began and whose thread had already announced
            // itself, rather than a lost write.
            self.started[i].store(generation, Ordering::Relaxed);
        }
        let fresh = ANNOUNCED_IN.with(|seen| {
            if seen.get() == generation {
                return false;
            }
            seen.set(generation);
            true
        });
        if fresh {
            let name = std::thread::current()
                .name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("{:?}", std::thread::current().id()));
            self.threads
                .lock()
                .expect("stall watch threads")
                .insert(name, generation);
        }
    }

    fn record_finish(&self, i: usize) {
        if i < WIDEST_FAN_OUT {
            let generation = self.generation.load(Ordering::Acquire);
            self.finished[i].store(generation, Ordering::Relaxed);
        }
    }

    /// Print what the stalled dispatch reached.
    ///
    /// The index and thread sets are printed only when the report
    /// armed them. Under the census nothing writes them, so printing
    /// them would say every closure never started and no thread ran
    /// one, which is what a genuinely stranded fan-out looks like.
    fn report(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        let n = self.width.load(Ordering::Relaxed).min(WIDEST_FAN_OUT);
        let arm = self.arm.lock().expect("stall watch arm").clone();

        eprintln!("STALL in {arm} at fan-out {n}, dispatch {generation}");

        // Every node rather than this thread's: the watchdog is its own
        // thread and resolves its own node, which need not be the node
        // the stalled fan-out ran on, and a census of the wrong node
        // prints an empty list that reads as an answer.
        let arena = flynnel::sched::arena::global_local_arena();
        let by_node = arena.mailbox_census_by_node();
        let parked_by_node = arena.parked_census_by_node();
        let held: usize = by_node.iter().map(Vec::len).sum();
        if held == 0 {
            eprintln!(
                "  mailboxes: all empty on all {} node(s), so nothing is stranded \
                 where only its own worker could take it",
                by_node.len()
            );
        } else {
            for (node, holding) in by_node.iter().enumerate() {
                if holding.is_empty() {
                    continue;
                }
                let parked: &[usize] = parked_by_node
                    .get(node)
                    .map_or(&[], |p: &Vec<usize>| p.as_slice());
                // Split by whether the holder is parked, because the two
                // are different defects. Parked on a full mailbox is a
                // wake that did not arrive or did not stick. Awake on
                // one is a worker that is running and not looking at
                // the one queue nobody else can drain for it.
                let asleep: Vec<usize> = holding
                    .iter()
                    .copied()
                    .filter(|i| parked.contains(i))
                    .collect();
                let awake: Vec<usize> =
                    holding.iter().copied().filter(|i| !parked.contains(i)).collect();
                eprintln!(
                    "  node {node}: mailboxes still holding work, by worker \
                     index: {holding:?}"
                );
                eprintln!("    of those, parked: {asleep:?}");
                eprintln!("    of those, awake:  {awake:?}");
                eprintln!("    every parked worker on this node: {parked:?}");
            }
        }

        if instrument() != Instrument::Report {
            eprintln!(
                "  which closures ran is not recorded under the census; re-run \
                 with FLYNNEL_BENCH_STALL_REPORT=1 for that, and note it has so \
                 far prevented the stall"
            );
            return;
        }

        let never: Vec<usize> = (0..n)
            .filter(|&i| self.started[i].load(Ordering::Relaxed) != generation)
            .collect();
        let unfinished: Vec<usize> = (0..n)
            .filter(|&i| {
                self.started[i].load(Ordering::Relaxed) == generation
                    && self.finished[i].load(Ordering::Relaxed) != generation
            })
            .collect();
        let threads = self.threads.lock().expect("stall watch threads");
        let ran: Vec<&String> = threads
            .iter()
            .filter(|&(_, &g)| g == generation)
            .map(|(name, _)| name)
            .collect();
        let idle: Vec<&String> = threads
            .iter()
            .filter(|&(_, &g)| g != generation)
            .map(|(name, _)| name)
            .collect();

        eprintln!("  threads that ran a closure in it: {}", ran.len());
        for t in &ran {
            eprintln!("    {t}");
        }
        // Threads seen in an earlier dispatch and not this one. These
        // are the workers that exist and did not wake, which is the
        // distinction a count of running threads cannot make.
        eprintln!("  threads that ran in an earlier dispatch only: {}", idle.len());
        for t in &idle {
            eprintln!("    {t}");
        }
        eprintln!("  closures that never started: {}", never.len());
        eprintln!("    {}", summarise(&never));
        eprintln!("  closures started and not finished: {}", unfinished.len());
        eprintln!("    {}", summarise(&unfinished));
    }
}

/// Render a sorted index list as runs, so 1024 missing indices read as
/// one range rather than a thousand numbers.
fn summarise(indices: &[usize]) -> String {
    if indices.is_empty() {
        return "none".to_string();
    }
    let mut runs: Vec<String> = Vec::new();
    let mut start = indices[0];
    let mut prev = indices[0];
    for &i in &indices[1..] {
        if i != prev + 1 {
            runs.push(if start == prev {
                format!("{start}")
            } else {
                format!("{start}..={prev}")
            });
            start = i;
        }
        prev = i;
    }
    runs.push(if start == prev {
        format!("{start}")
    } else {
        format!("{start}..={prev}")
    });
    runs.join(", ")
}

/// Watch for a dispatch that stops advancing, report it, and end the
/// process.
///
/// It ends the process because a stalled criterion run otherwise sits
/// until an external watchdog kills it, and a kill arrives after the
/// report would have been useful and takes the report with it.
fn spawn_stall_watchdog() {
    std::thread::spawn(|| {
        let watch = stall_watch();
        let mut last = watch.generation.load(Ordering::Acquire);
        let mut still = Duration::ZERO;
        loop {
            std::thread::sleep(Duration::from_secs(1));
            let now = watch.generation.load(Ordering::Acquire);
            if now == last {
                still += Duration::from_secs(1);
                if still >= STALL_AFTER {
                    watch.report();
                    std::process::exit(9);
                }
            } else {
                last = now;
                still = Duration::ZERO;
            }
        }
    });
}

/// One fan-out shape.
///
/// `Deque` pushes N-1 closures onto the calling worker's local deque
/// and lets random-victim peer-steal distribute them. `Mailbox` pushes
/// each closure to one worker's mailbox, behind a gate comparing N
/// against the worker count: below it the fan-out demotes to the
/// shared deque, so `Deque` and `Mailbox` run the same code there.
/// `Routed` is the entry point a caller uses, whose `Auto` arm picks
/// between the tree and mailbox variants. `Rayon` is the closest
/// equivalent outside the crate.
#[derive(Copy, Clone)]
enum Arm {
    Deque,
    Mailbox,
    Routed,
    Rayon,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Deque => "flynnel_baseline_deque",
            Self::Mailbox => "flynnel_mailbox",
            Self::Routed => "flynnel_routed",
            Self::Rayon => "rayon_scope_spawn",
        }
    }
}

/// Build the closure set one iteration fans out.
///
/// Rebuilt per iteration because each variant consumes it, so the
/// allocation is inside every arm's timed region and not only some.
fn closures(n: usize) -> Vec<Box<dyn FnOnce() -> u64 + Send>> {
    let armed = instrument() == Instrument::Report;
    (0..n)
        .map(|i| {
            let b: Box<dyn FnOnce() -> u64 + Send> = if armed {
                Box::new(move || {
                    stall_watch().record_start(i);
                    let out = fixed_cost_work(i as u64);
                    stall_watch().record_finish(i);
                    out
                })
            } else {
                Box::new(move || fixed_cost_work(i as u64))
            };
            b
        })
        .collect()
}

/// Fan out `n` closures through one shape and materialize the results.
///
/// Every arm produces a `Vec<u64>` in caller order, so the timed region
/// covers the result-gather phase equally.
fn run_arm(arm: Arm, plan: &JobPlan, n: usize) {
    if instrument() != Instrument::Off {
        stall_watch().begin(arm.name(), n);
    }
    match arm {
        Arm::Deque => {
            black_box(cooperative_join_n_flat::<u64>(plan, closures(n)));
        }
        Arm::Mailbox => {
            black_box(cooperative_join_n_flat_mailbox::<u64>(plan, closures(n)));
        }
        Arm::Routed => {
            black_box(cooperative_join_n::<u64>(plan, closures(n)));
        }
        Arm::Rayon => {
            let results: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(vec![0u64; n]);
            rayon::scope(|s| {
                for i in 0..n {
                    let results_ref = &results;
                    s.spawn(move |_| {
                        let r = fixed_cost_work(i as u64);
                        results_ref.lock().unwrap()[i] = r;
                    });
                }
            });
            black_box(results.into_inner().unwrap());
        }
    }
}

/// Register one group's arms at `n_closures`, in the order given.
fn register(c: &mut Criterion, group_name: String, n_closures: usize, arms: &[Arm]) {
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    let plan = JobPlan::new(2, 1);

    for arm in arms {
        group.bench_function(arm.name(), |b| {
            b.iter(|| run_arm(*arm, &plan, n_closures));
        });
    }

    group.finish();
}

/// Run the 4-way A/B at a specific N, forward and reversed.
fn bench_n(c: &mut Criterion, n_closures: usize) {
    const FWD: [Arm; 4] = [Arm::Deque, Arm::Mailbox, Arm::Routed, Arm::Rayon];
    const REV: [Arm; 4] = [Arm::Rayon, Arm::Routed, Arm::Mailbox, Arm::Deque];
    register(c, format!("simc_cooperative_n{n_closures}"), n_closures, &FWD);
    register(c, format!("simc_cooperative_n{n_closures}_rev"), n_closures, &REV);
}

fn bench_simc_cooperative(c: &mut Criterion) {
    match instrument() {
        Instrument::Off => {}
        Instrument::Census => {
            eprintln!(
                "stall census armed: nothing is recorded inside a closure, and a \
                 dispatch that does not advance for {STALL_AFTER:?} reports which \
                 arm stopped and which mailboxes still hold work, then ends the \
                 process. Timings from this run are not comparable with a run \
                 without it."
            );
            spawn_stall_watchdog();
        }
        Instrument::Report => {
            eprintln!(
                "stall report armed: closures record which of them ran, and a \
                 dispatch that does not advance for {STALL_AFTER:?} is reported \
                 and the process ends. Timings from this run are not comparable \
                 with a run without it, and the per-closure recording has so far \
                 prevented every stall it was pointed at."
            );
            spawn_stall_watchdog();
        }
    }
    // The sweep straddles the gate, which sits at the worker count.
    // Below it both flynnel shape arms run the same deque path.
    bench_n(c, 8);
    bench_n(c, 12);
    bench_n(c, 16);
    // At or above the worker count on hosts up to twenty-four, where
    // the mailbox arm reaches owner-directed distribution. A wider
    // pool moves the crossing up and these sizes back below it.
    bench_n(c, 24);
    bench_n(c, 32);
    bench_n(c, 56);
    bench_n(c, 64);
    // Past the widths where mailbox has been measured losing to deque
    // by 10 to 22 percent. The design's own argument is that
    // owner-directed placement eventually beats random peer-steal, so
    // the gate belongs wherever that crossing is rather than at the
    // worker count; these are what say whether there is one.
    bench_n(c, 128);
    bench_n(c, 256);
    // 256 reached 10.7x the pool with the penalty no smaller than at
    // 64. These are 21x and 43x, where a fan-out is deep enough that
    // random peer-steal has to rediscover a distribution the mailbox
    // path was handed. If the crossing is anywhere it is here.
    bench_n(c, 512);
    bench_n(c, 1024);
    // Between the tie at 512 and the 15 percent mailbox win at 1024.
    // A gate wants the width where the two meet, and 512 and 1024 give
    // only the bracket it lies in.
    bench_n(c, 640);
    bench_n(c, 768);
    bench_n(c, 896);
}

criterion_group!(benches, bench_simc_cooperative);
criterion_main!(benches);

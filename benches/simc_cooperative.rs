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
//! ## The stall report
//!
//! `FLYNNEL_BENCH_STALL_REPORT=1` makes every closure record that it
//! started and finished, and makes a watchdog print what a dispatch
//! reached if it stops advancing.
//!
//! It answers a stall with sets rather than an event log: the indices
//! that never started, the ones that started and never finished, and
//! the threads that ran anything. Those are bounded by the fan-out
//! width whatever the iteration count, where an event log is bounded by
//! iterations times width, which at this bench's counts is tens of
//! millions of records across forty-nine threads.
//!
//! Reading it: indices that never started with workers missing from the
//! thread list is work nothing was woken to take; indices that never
//! started with every worker present is work routed where none of them
//! looks; started-and-unfinished is a closure that is running or a
//! thread that died inside one.
//!
//! It is off by default because it puts two atomic stores and one lock
//! acquisition in each closure, which moves the numbers this bench
//! exists to take. A run with it on diagnoses and does not measure.

#![allow(clippy::missing_docs_in_private_items)]

use std::collections::BTreeSet;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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

/// Whether the stall report is armed, from `FLYNNEL_BENCH_STALL_REPORT`.
///
/// Off by default. When it is on every closure writes two atomic bits
/// and, the first time a thread runs one in a dispatch, takes a lock,
/// which moves the timings this bench exists to take. A run with it on
/// diagnoses; a run with it off measures.
fn stall_report_armed() -> bool {
    static ARMED: OnceLock<bool> = OnceLock::new();
    *ARMED.get_or_init(|| {
        matches!(
            std::env::var("FLYNNEL_BENCH_STALL_REPORT").as_deref(),
            Ok("1") | Ok("true")
        )
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
    started: Vec<AtomicBool>,
    finished: Vec<AtomicBool>,
    threads: Mutex<BTreeSet<String>>,
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

/// The widest fan-out any group registers.
const WIDEST_FAN_OUT: usize = 1024;

/// How long without a new dispatch counts as stalled.
const STALL_AFTER: Duration = Duration::from_secs(25);

fn stall_watch() -> &'static StallWatch {
    static WATCH: OnceLock<StallWatch> = OnceLock::new();
    WATCH.get_or_init(|| StallWatch {
        started: (0..WIDEST_FAN_OUT).map(|_| AtomicBool::new(false)).collect(),
        finished: (0..WIDEST_FAN_OUT).map(|_| AtomicBool::new(false)).collect(),
        threads: Mutex::new(BTreeSet::new()),
        generation: AtomicU64::new(0),
        width: AtomicUsize::new(0),
        arm: Mutex::new(String::new()),
    })
}

impl StallWatch {
    /// Clear the maps and open a new dispatch.
    fn begin(&self, arm: &str, n: usize) {
        for i in 0..n.min(WIDEST_FAN_OUT) {
            self.started[i].store(false, Ordering::Relaxed);
            self.finished[i].store(false, Ordering::Relaxed);
        }
        self.threads.lock().expect("stall watch threads").clear();
        *self.arm.lock().expect("stall watch arm") = arm.to_string();
        self.width.store(n, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Release);
    }

    fn started(&self, i: usize) {
        if i < WIDEST_FAN_OUT {
            // Recorded before the thread name so an index that appears
            // with no thread is a closure that began and whose runner
            // never reached the lock, rather than a lost write.
            self.started[i].store(true, Ordering::Relaxed);
        }
        let name = std::thread::current()
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| format!("{:?}", std::thread::current().id()));
        self.threads.lock().expect("stall watch threads").insert(name);
    }

    fn finished(&self, i: usize) {
        if i < WIDEST_FAN_OUT {
            self.finished[i].store(true, Ordering::Relaxed);
        }
    }

    /// Print what the stalled dispatch reached.
    fn report(&self) {
        let n = self.width.load(Ordering::Relaxed).min(WIDEST_FAN_OUT);
        let arm = self.arm.lock().expect("stall watch arm").clone();
        let never: Vec<usize> = (0..n)
            .filter(|&i| !self.started[i].load(Ordering::Relaxed))
            .collect();
        let unfinished: Vec<usize> = (0..n)
            .filter(|&i| {
                self.started[i].load(Ordering::Relaxed)
                    && !self.finished[i].load(Ordering::Relaxed)
            })
            .collect();
        let threads = self.threads.lock().expect("stall watch threads").clone();

        eprintln!("STALL in {arm} at fan-out {n}");
        eprintln!("  threads that ran a closure: {}", threads.len());
        for t in &threads {
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
    let armed = stall_report_armed();
    (0..n)
        .map(|i| {
            let b: Box<dyn FnOnce() -> u64 + Send> = if armed {
                Box::new(move || {
                    stall_watch().started(i);
                    let out = fixed_cost_work(i as u64);
                    stall_watch().finished(i);
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
    if stall_report_armed() {
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
    if stall_report_armed() {
        eprintln!(
            "stall report armed: closures record which of them ran, and a \
             dispatch that does not advance for {STALL_AFTER:?} is reported \
             and the process ends. Timings from this run are not comparable \
             with a run without it."
        );
        spawn_stall_watchdog();
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

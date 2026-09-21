//! A/B microbench: Parker wake latency across the three wake paths,
//! std::thread::park, WAITPKG (UMONITOR + UMWAIT) and MONITORX
//! (MONITORX + MWAITX).
//!
//! Two threads:
//! - **Owner**: parks via `park_until` past the spin floor.
//! - **Producer**: sleeps a fixed inter-arrival, then unparks owner.
//!
//! The bench measures wall-clock time from `unpark` to the owner's
//! observable return from `park_until`. That captures BOTH the
//! syscall transition cost (StdPark) or the UMWAIT-return cost
//! (`WAITPKG`) and the inter-thread coherence transfer on the
//! wake_counter / permit cache line.
//!
//! ## Bench-audit (HARD RULE 3)
//!
//! - **Same payload across A/B**: identical predicate closure
//!   (`AtomicU32::load() == 1`), identical inter-arrival
//!   (50 us sleep before unpark), identical spin-rounds (0 - skip
//!   the spin floor entirely so the bench measures only the wait
//!   path).
//! - **Same scheduling shape**: two threads pinned to distinct cores;
//!   one parks, one unparks; we measure the unpark-to-return delta.
//! - **The primitive's named feature IS exercised**: WAITPKG path
//!   issues UMONITOR + UMWAIT against `wake_counter`; MONITORX path
//!   issues MONITORX + MWAITX against the same line; StdPark path
//!   issues `thread::park()` + permits.
//! - **Idle and loaded, every arm in both**: the load arm occupies
//!   half the host with busy threads for the length of a group. A
//!   wake is what a kernel park pays to get back onto a core, so an
//!   idle host is the condition under which that cost is smallest
//!   and the arms are hardest to tell apart. Neither arm alone
//!   settles anything: the loaded rows are the case being argued
//!   for, and the idle rows are where "never slower" has to hold
//!   anyway.
//!
//! ## Hardware availability
//!
//! Each path is detected at runtime, WAITPKG via
//! `crate::cpu_info::has_waitpkg` and MONITORX via
//! `crate::cpu_info::has_monitorx`, and a path the host does not
//! carry is skipped with an explanatory eprintln rather than run.
//!
//! Which rows a host produces is itself the reading. WAITPKG needs
//! Tiger Lake and later or Zen 5 and later, so on Zen 1 to 4 only the
//! MONITORX rows appear beside the baseline, and that pair is the
//! comparison this bench exists for on those parts. A guest whose
//! hypervisor masks both leaves produces the StdPark rows alone,
//! which is not a result about the silicon.

#![allow(clippy::missing_docs_in_private_items)]

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::sched::sleep::{Parker, WaitStrategy};

/// Wake-latency for one strategy. Spawns a parker-owner thread that
/// constructs a Parker (with the requested strategy) and calls
/// `park_until` repeatedly. Each iter: producer sleeps `gap_us`,
/// unparks, owner returns. We measure unpark -> return delta from
/// the producer side (rdtsc bracketing on the unpark + the owner's
/// observable return via a Release/Acquire flag).
/// Busy threads occupying half the host, joined when this is dropped.
///
/// A wake measured on an idle host is the case a scheduler is least
/// often in: there is a free core waiting to take the woken thread.
/// Under load a kernel park has to queue behind other runnable work
/// to get back on a core, and an in-core monitor wait does not, so
/// the arms are only distinguishable here. An idle-only reading would
/// report the smaller half of whatever difference exists.
struct Load {
    stop: Arc<AtomicU32>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Load {
    fn spawn() -> Self {
        let n = std::thread::available_parallelism()
            .map(|p| p.get() / 2)
            .unwrap_or(1)
            .max(1);
        let stop = Arc::new(AtomicU32::new(0));
        let threads = (0..n)
            .map(|_| {
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut x = 0u64;
                    while stop.load(Ordering::Relaxed) == 0 {
                        // A dependent chain, so the thread occupies a
                        // core rather than being elided.
                        x = black_box(x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1));
                    }
                })
            })
            .collect();
        Self { stop, threads }
    }
}

impl Drop for Load {
    fn drop(&mut self) {
        self.stop.store(1, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            // A load thread that panicked took its share of the load
            // with it, so the arm it was loading measured something
            // lighter than the one beside it. Said out loud rather
            // than dropped, because the run would otherwise look
            // like a clean A/B.
            if t.join().is_err() {
                eprintln!("parker_wait_strategy: a load thread panicked, so this run's load arm carried less load than it reports");
            }
        }
    }
}

fn bench_strategy(
    c: &mut Criterion,
    label: &str,
    strategy: WaitStrategy,
    gap_us: u64,
    loaded: bool,
) {
    // Held for the whole group so every sample in it sees the same
    // occupancy, and dropped with the group so the next arm does not
    // inherit it.
    let _load = if loaded { Some(Load::spawn()) } else { None };
    let arm = if loaded { "load" } else { "idle" };
    let name = format!("parker_wait_{label}_{arm}_gap_{gap_us}us");
    // The fallback flag is process-wide and never cleared, so one
    // group can turn every later group into the kernel park. Reported
    // at the boundary it crossed, because a run that only says the
    // monitor stopped holding leaves every row after it unreadable
    // and every row before it indistinguishable from those.
    let held_before = flynnel::sched::sleep::monitor_wait_held();
    let mut group = c.benchmark_group(&name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("unpark_to_return", |b| {
        // For each measurement, spawn fresh owner thread + parker.
        // park_until uses spin_rounds=0 so the bench measures ONLY
        // the wait path (no spin-floor contribution).
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let ready = Arc::new(AtomicU32::new(0));
                let returned = Arc::new(AtomicU32::new(0));
                let returned_clone = Arc::clone(&returned);
                let ready_clone = Arc::clone(&ready);
                let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();
                let owner = std::thread::spawn(move || {
                    let p = Arc::new(Parker::with_strategy(0, strategy));
                    tx.send(Arc::clone(&p)).expect("send parker");
                    let ok = p.park_until(|| ready.load(Ordering::Acquire) == 1);
                    returned_clone.store(1, Ordering::Release);
                    ok
                });
                let p_owner = rx.recv().expect("recv parker");
                // Inter-arrival sleep so the owner has actually
                // entered the wait path.
                std::thread::sleep(Duration::from_micros(gap_us));
                let t0 = Instant::now();
                ready_clone.store(1, Ordering::Release);
                p_owner.unpark();
                // Spin-wait the returned flag so the measurement
                // covers unpark -> owner-thread-observably-returned.
                while returned.load(Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }
                total += t0.elapsed();
                owner.join().expect("owner join");
            }
            black_box(total)
        });
    });
    group.finish();
    report_latch(&name, held_before);
}

/// Names the group a fallback fired in, and how fast the arms were
/// that triggered it.
///
/// The span is what separates the two causes the verdict cannot: a
/// few thousand cycles is a monitor that never armed, hundreds of
/// thousands is one that armed and was ended by something else.
fn report_latch(name: &str, held_before: bool) {
    if held_before && !flynnel::sched::sleep::monitor_wait_held() {
        let (doubts, span) = flynnel::sched::sleep::monitor_doubts();
        eprintln!(
            "parker_wait_strategy: the monitor stopped holding during {name} after {doubts} \
             doubts, the last spanning {span} cycles. Rows from here on are the kernel park, \
             whatever their label says."
        );
    }
}

/// Wake latency with other parkers alive and being unparked.
///
/// A monitor wait watches the cache line its `wake_counter` sits in,
/// so what a neighbouring parker does to its own counter matters if
/// the two share a line. One parker has no neighbour, which is what
/// every other group here measures, and a pool has twenty-odd: this
/// group is the difference between those two, and it is the shape a
/// live arena actually has.
///
/// The noise thread unparks every parker except the one being timed.
/// Nothing it does should reach that one, and any effect on the
/// timing is the layout rather than the protocol.
fn bench_neighbours(c: &mut Criterion, label: &str, strategy: WaitStrategy) {
    let name = format!("parker_neighbours_{label}");
    let held_before = flynnel::sched::sleep::monitor_wait_held();
    let mut group = c.benchmark_group(&name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("unpark_to_return", |b| {
        b.iter_custom(|iters| {
            let n = std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(4);
            // Allocated together, the way an arena allocates its
            // worker parkers, so they land near each other.
            let neighbours: Vec<Arc<Parker>> = (0..n)
                .map(|_| Arc::new(Parker::with_strategy(0, strategy)))
                .collect();

            let stop = Arc::new(AtomicU32::new(0));
            let noise_stop = Arc::clone(&stop);
            let noise_set: Vec<Arc<Parker>> = neighbours.clone();
            let noise = std::thread::spawn(move || {
                while noise_stop.load(Ordering::Relaxed) == 0 {
                    for p in &noise_set {
                        p.unpark();
                    }
                }
            });

            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let ready = Arc::new(AtomicU32::new(0));
                let returned = Arc::new(AtomicU32::new(0));
                let returned_clone = Arc::clone(&returned);
                let ready_clone = Arc::clone(&ready);
                let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();
                let owner = std::thread::spawn(move || {
                    let p = Arc::new(Parker::with_strategy(0, strategy));
                    tx.send(Arc::clone(&p)).expect("send parker");
                    let ok = p.park_until(|| ready.load(Ordering::Acquire) == 1);
                    returned_clone.store(1, Ordering::Release);
                    ok
                });
                let p_owner = rx.recv().expect("recv parker");
                std::thread::sleep(Duration::from_micros(50));
                let t0 = Instant::now();
                ready_clone.store(1, Ordering::Release);
                p_owner.unpark();
                while returned.load(Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }
                total += t0.elapsed();
                owner.join().expect("owner join");
            }

            stop.store(1, Ordering::Relaxed);
            noise.join().expect("noise join");
            black_box(total)
        });
    });
    group.finish();
    report_latch(&name, held_before);
}

/// What a pool of parkers that are merely waiting costs a thread that
/// is not one of them.
///
/// Every other group here times the parker. This one times its
/// neighbour, and the parkers do nothing at all: they park once and
/// stay parked for the whole measurement. The question is what a
/// waiting worker takes from the rest of the machine while it waits.
///
/// It is worth asking because the arms are not alike in that respect.
/// `thread::park` leaves the run queue, so the logical CPU goes back
/// to the scheduler and something else can have it. `MWAITX` halts
/// the logical processor without a syscall, so the thread never
/// leaves the CPU from the kernel's point of view, though the halt
/// does hand a physical core's issue width to its SMT sibling. Which
/// of those dominates is not something the instruction set answers.
///
/// A pool holds one worker per logical CPU, so on a busy host this is
/// the difference between a neighbour having the box and sharing it
/// with two dozen residents.
///
/// The zero-parker row is the control and the rows only mean
/// something against it: it is the same workload with nothing else
/// alive, so it says what the measurement costs when there is nothing
/// to take.
fn bench_idle_neighbours(c: &mut Criterion, label: &str, parkers: Option<WaitStrategy>) {
    let name = format!("parker_idle_neighbours_{label}");
    let held_before = flynnel::sched::sleep::monitor_wait_held();
    let mut group = c.benchmark_group(&name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("neighbour_work", |b| {
        b.iter_custom(|iters| {
            let n = match parkers {
                None => 0,
                Some(_) => std::thread::available_parallelism()
                    .map(std::num::NonZeroUsize::get)
                    .unwrap_or(4),
            };
            let stop = Arc::new(AtomicU32::new(0));
            let mut waiting = Vec::with_capacity(n);
            let mut handles = Vec::with_capacity(n);
            for _ in 0..n {
                let strategy = parkers.expect("n is zero when no strategy is given");
                let stop_c = Arc::clone(&stop);
                let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();
                handles.push(std::thread::spawn(move || {
                    let p = Arc::new(Parker::with_strategy(0, strategy));
                    tx.send(Arc::clone(&p)).expect("send parker");
                    // Parks and stays parked. park_until may return
                    // without the predicate holding, so this re-enters
                    // until it does rather than spinning out here and
                    // becoming the very load it is meant not to be.
                    while stop_c.load(Ordering::Acquire) == 0 {
                        p.park_until(|| stop_c.load(Ordering::Acquire) == 1);
                    }
                }));
                waiting.push(rx.recv().expect("recv parker"));
            }
            // Let them all reach their wait before timing anything,
            // or the first iterations measure threads still starting.
            if n > 0 {
                std::thread::sleep(Duration::from_millis(50));
            }

            let t0 = Instant::now();
            let mut acc = 0u64;
            for i in 0..iters {
                // Deterministic, CPU-bound, and nothing to do with
                // the scheduler: this is the neighbour's own work.
                let mut x = i | 1;
                for _ in 0..512 {
                    x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    acc = acc.wrapping_add(x >> 33);
                }
            }
            let elapsed = t0.elapsed();
            black_box(acc);

            stop.store(1, Ordering::Release);
            for p in &waiting {
                p.unpark();
            }
            for h in handles {
                h.join().expect("waiting parker join");
            }
            elapsed
        });
    });
    group.finish();
    report_latch(&name, held_before);
}

/// Wake latency on one parker woken repeatedly on one thread, which
/// is the shape a pool worker has.
///
/// `pin` of `None` leaves the parker on the controller and is the
/// adaptive arm; `Some` holds one strategy still, the way the groups
/// above do. Both go through this same function because the arms
/// only compare if they are measured the same way, and the adaptive
/// arm cannot use the groups above: those build a parker per
/// iteration, and the controller's cadence is per thread, so a
/// thread that parks once never reaches a probe.
///
/// Two things this answers. On a host with no second strategy the
/// adaptive arm is the overhead arm, differing from the pinned
/// baseline beside it only by the controller being consulted, so the
/// gap between them is what the controller costs a host that can
/// never use it. On a host with both it has to end up at least as
/// fast as the better pinned arm, because otherwise the choosing is
/// worse than either choice.
///
/// These are steady-state numbers and do not replace the groups
/// above, which time a first park on a cold thread.
fn bench_repeat(
    c: &mut Criterion,
    label: &str,
    pin: Option<WaitStrategy>,
    gap_us: u64,
    loaded: bool,
) {
    let _load = if loaded { Some(Load::spawn()) } else { None };
    let arm = if loaded { "load" } else { "idle" };
    let held_before = flynnel::sched::sleep::monitor_wait_held();
    let name = format!("parker_repeat_{label}_{arm}_gap_{gap_us}us");
    let mut group = c.benchmark_group(&name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("unpark_to_return", |b| {
        b.iter_custom(|iters| {
            // One parker on one thread, parked and woken `iters`
            // times.
            //
            // The thread is long-lived because the controller's
            // cadence is per thread: a thread that parks once takes
            // one step toward a probe it needs sixty-four to reach,
            // so a thread per iteration leaves every park unsampled
            // and the controller at zero samples however long the
            // bench runs. It would report the baseline in use and
            // nothing measured, which reads like a verdict and is an
            // instrument that never engaged.
            let ready = Arc::new(AtomicU32::new(0));
            let done = Arc::new(AtomicU32::new(0));
            let stop = Arc::new(AtomicU32::new(0));
            let (ready_c, done_c, stop_c) =
                (Arc::clone(&ready), Arc::clone(&done), Arc::clone(&stop));
            let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();
            let owner = std::thread::spawn(move || {
                let p = Arc::new(match pin {
                    Some(s) => Parker::with_strategy(0, s),
                    None => Parker::new(0),
                });
                tx.send(Arc::clone(&p)).expect("send parker");
                let mut seen = 0u32;
                loop {
                    p.park_until(|| {
                        stop_c.load(Ordering::Acquire) == 1
                            || ready_c.load(Ordering::Acquire) != seen
                    });
                    if stop_c.load(Ordering::Acquire) == 1 {
                        return;
                    }
                    seen = ready_c.load(Ordering::Acquire);
                    done_c.store(seen, Ordering::Release);
                }
            });
            let p = rx.recv().expect("recv parker");

            // A generation rather than a flag, so the owner tells
            // this wake from the one it just served without the
            // harness clearing anything underneath it. Consecutive
            // wrapping values always differ, which is all it needs.
            let mut total = Duration::ZERO;
            let mut generation = 0u32;
            for _ in 0..iters {
                generation = generation.wrapping_add(1);
                std::thread::sleep(Duration::from_micros(gap_us));
                let t0 = Instant::now();
                ready.store(generation, Ordering::Release);
                p.unpark();
                while done.load(Ordering::Acquire) != generation {
                    std::hint::spin_loop();
                }
                total += t0.elapsed();
            }

            stop.store(1, Ordering::Release);
            p.unpark();
            owner.join().expect("owner join");
            black_box(total)
        });
    });
    group.finish();

    report_latch(&name, held_before);

    // Only the unpinned arm runs the controller, so only it has
    // anything to say about what the controller did.
    if pin.is_none() {
        let r = flynnel::sched::sleep::wait_controller().report();
        eprintln!(
            "parker_wait_strategy: after {name}, controller has baseline {} ns over {} samples, \
             challenger {} ns over {} samples, in use {:?}, {} switch(es), probing every {} \
             parks, paired score {}",
            r.baseline_ns,
            r.baseline_samples,
            r.challenger_ns,
            r.challenger_samples,
            r.in_use,
            r.switches,
            r.probe_every,
            r.score
        );
    }
}

fn bench_all(c: &mut Criterion) {
    // Which tree built this. Two hosts here carry a directory called
    // Flynnel-verify and they are different checkouts, so a run can
    // be against source from the wrong host with nothing in the
    // output to say so. Baked in at compile time, so it describes the
    // binary rather than wherever it ran.
    //
    // It names the tree and not the commit, and the difference bites:
    // a binary keeps running the source it linked while the tree
    // moves underneath, so this line stays identical across a sync
    // and cannot date the build. Getting the commit needs a build
    // script, which would then run for every consumer of the crate.
    eprintln!(
        "parker_wait_strategy: built from {} v{}",
        env!("CARGO_MANIFEST_DIR"),
        env!("CARGO_PKG_VERSION")
    );

    let waitpkg = flynnel::cpu_info::has_waitpkg();
    let monitorx = flynnel::cpu_info::has_monitorx();

    if !waitpkg {
        eprintln!(
            "parker_wait_strategy: WAITPKG arm skipped - host has no \
             WAITPKG (cpuid leaf 7 ECX bit 5 = 0)."
        );
    }
    if !monitorx {
        eprintln!(
            "parker_wait_strategy: MONITORX arm skipped - host has no \
             MONITORX (cpuid Fn8000_0001 ECX bit 29 = 0)."
        );
    }
    if !waitpkg && !monitorx {
        eprintln!(
            "parker_wait_strategy: this host has neither monitor-wait, \
             so the StdPark rows are the only path its Parker can take \
             and there is no A/B in this run."
        );
    }

    // Idle before load, and every arm present in both. The idle rows
    // alone cannot carry a claim, because the case being argued for
    // is a busy host; the load rows alone cannot either, because
    // "never slower" has to hold when the host is quiet too.
    for loaded in [false, true] {
        // StdPark is the control and runs whatever the host is: it is
        // what the Parker does today on every part that reaches this
        // code, so a run without it compares two treatments.
        for gap_us in [50, 500] {
            bench_strategy(c, "stdpark", WaitStrategy::StdPark, gap_us, loaded);

            // Each monitor wait is gated on its own bit rather than
            // one being the other's else-arm. A host carrying both
            // would otherwise report only the arm `pick` chooses,
            // and on such a host the comparison worth having is the
            // three side by side.
            if waitpkg {
                bench_strategy(c, "waitpkg", WaitStrategy::Waitpkg, gap_us, loaded);
            }
            if monitorx {
                bench_strategy(c, "monitorx", WaitStrategy::Monitorx, gap_us, loaded);
            }
        }
    }

    // The steady-state set: one parker woken repeatedly on one
    // thread, pinned and unpinned, so the adaptive arm has arms
    // beside it measured the same way. The groups above build a
    // parker per iteration and time a cold first park, which is a
    // different quantity and not this arm's comparison.
    //
    // The adaptive arm leads each cell. A pinned parker ignores the
    // controller, so nothing the controller decides can reach the
    // arms beside it; the coupling runs the other way. The fallback
    // flag is process-wide and never cleared, so a monitor arm that
    // gives up on the monitor leaves every later group with no
    // challenger to race, and the adaptive arm would then report no
    // samples for a reason that has nothing to do with it.
    for loaded in [false, true] {
        for gap_us in [50, 500] {
            bench_repeat(c, "adaptive", None, gap_us, loaded);
            bench_repeat(c, "stdpark", Some(WaitStrategy::StdPark), gap_us, loaded);
            if waitpkg {
                bench_repeat(c, "waitpkg", Some(WaitStrategy::Waitpkg), gap_us, loaded);
            }
            if monitorx {
                bench_repeat(c, "monitorx", Some(WaitStrategy::Monitorx), gap_us, loaded);
            }
        }
    }

    // What waiting parkers cost a thread that is not one of them. The
    // none row runs first because it is the control the others are
    // read against, and StdPark is included here unlike the group
    // below: this asks what a waiting thread does to the machine, and
    // leaving the run queue is as much an answer as halting on it.
    bench_idle_neighbours(c, "none", None);
    bench_idle_neighbours(c, "stdpark", Some(WaitStrategy::StdPark));
    if waitpkg {
        bench_idle_neighbours(c, "waitpkg", Some(WaitStrategy::Waitpkg));
    }
    if monitorx {
        bench_idle_neighbours(c, "monitorx", Some(WaitStrategy::Monitorx));
    }

    // Neighbours last, and only the two strategies a monitor wait can
    // take: StdPark has no monitor, so a neighbouring store cannot
    // reach it and the row would only restate the idle one.
    if waitpkg {
        bench_neighbours(c, "waitpkg", WaitStrategy::Waitpkg);
    }
    if monitorx {
        bench_neighbours(c, "stdpark", WaitStrategy::StdPark);
        bench_neighbours(c, "monitorx", WaitStrategy::Monitorx);
    }

    // Printed after the arms, because it is only knowable once a wait
    // has run. A MONITORX row that matches the StdPark row beside it
    // has usually fallen back to exactly that, and without this line
    // that is indistinguishable from the monitor wait being no
    // faster.
    if monitorx && !flynnel::sched::sleep::monitor_wait_held() {
        eprintln!(
            "parker_wait_strategy: the monitor did not hold on this host, so the MONITORX \
             rows above are partly or wholly the kernel park it falls back to."
        );
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);

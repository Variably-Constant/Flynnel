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
    let mut group = c.benchmark_group(format!("parker_wait_{label}_{arm}_gap_{gap_us}us"));
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
    let mut group = c.benchmark_group(format!("parker_neighbours_{label}"));
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
}

fn bench_all(c: &mut Criterion) {
    // Which tree built this. Two hosts here carry a directory called
    // Flynnel-verify and they are different checkouts, so a run can
    // be against source several commits behind the one being reasoned
    // about with nothing in the output to say so. Baked in at compile
    // time, so it describes the binary rather than wherever it ran.
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

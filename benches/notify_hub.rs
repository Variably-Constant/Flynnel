//! The notify hub's hand-offs and the IO pool built on it, quiet and beside
//! threads spinning on every logical processor.
//!
//! - `hub_round_trip_1`: one item through a one-consumer hub and back
//!   through a second, the single-consumer wake on both legs.
//! - `hub_drain_4`: a burst of 256 items through a hub with four consumer
//!   threads, timed until every item is taken, the consumers going idle
//!   between bursts.
//! - `io_pool_start_n`: n tasks submitted to a pool of n workers built just
//!   before, timed until all n have started. Each task works for 20 ms and
//!   returns, so a task left queued starts when a worker frees and the
//!   round reads that wait.
//! - `io_pool_drain_n`: 1000 short tasks through a pool of n workers, timed
//!   until all have run.
//! - `io_pool_idle_n`, quiet only: the processor time a pool of n idle
//!   workers spends in 50 ms of wall time. On Windows the figure is the
//!   process's cycle count, read in nanosecond units, so only its ratio
//!   between builds means anything. It follows the wait the process-wide
//!   controller has settled on by then: a worker in a monitor wait stays
//!   on its processor and counts as running, one in a kernel park does
//!   not. The controller's report is printed to stderr before and after
//!   the cell so each run shows which wait it read.
//!
//! n is the logical processor count. Built from two trees, the same cells
//! compare two builds of the hub: run the base build, the tip build and the
//! base again from a second file as the copy, one process a run, and read
//! each cell's tip against the base beside the copy's.
//!
//!   cargo bench --bench notify_hub

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::sched::IoPool;
use flynnel::sched::notify_ring::NotifyHub;
use flynnel::sched::sleep::wait_controller;

const BURST: usize = 256;
const POOL_TASKS: usize = 1000;
const TASK_WORK: Duration = Duration::from_millis(20);
const IDLE_WINDOW: Duration = Duration::from_millis(50);

/// Threads spinning until dropped.
struct Spinners {
    stop: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<u64>>,
}

impl Spinners {
    fn start(count: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let threads = (0..count)
            .map(|_| {
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut spins = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        spins = black_box(spins.wrapping_add(1));
                    }
                    spins
                })
            })
            .collect();
        Self { stop, threads }
    }
}

impl Drop for Spinners {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for thread in self.threads.drain(..) {
            if let Err(panicked) = thread.join() {
                panic!("a spinner panicked: {panicked:?}");
            }
        }
    }
}

/// The processor time this process has used, in nanoseconds; on Windows its
/// cycle count instead.
#[cfg(windows)]
fn process_cpu() -> u64 {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn QueryProcessCycleTime(process: *mut std::ffi::c_void, cycles: *mut u64) -> i32;
    }
    let mut cycles = 0u64;
    // SAFETY: the pointer is to a live local, and the handle
    // GetCurrentProcess returns needs no closing.
    let ok = unsafe { QueryProcessCycleTime(GetCurrentProcess(), &mut cycles) };
    assert!(
        ok != 0,
        "QueryProcessCycleTime failed: {}",
        std::io::Error::last_os_error()
    );
    cycles
}

#[cfg(unix)]
fn process_cpu() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: the pointer is to a live local timespec.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    assert_eq!(
        rc,
        0,
        "clock_gettime failed: {}",
        std::io::Error::last_os_error()
    );
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn bench_all(c: &mut Criterion) {
    let logical = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .expect("logical processor count");
    eprintln!("notify_hub logical_processors={logical}");

    for (label, spinners) in [("quiet", 0), ("all", logical)] {
        let load = Spinners::start(spinners);
        let mut group = c.benchmark_group(format!("notify_hub_{label}"));
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(2));

        // One item there and back, each leg a one-consumer hub.
        {
            let ping = NotifyHub::<u64>::new(64, 1);
            let pong = NotifyHub::<u64>::new(64, 1);
            let pong_rx = pong.register_consumer();
            let responder = {
                let ping = ping.clone();
                let pong_tx = pong.sender();
                std::thread::spawn(move || {
                    let rx = ping.register_consumer();
                    while let Some(v) = rx.recv() {
                        assert!(pong_tx.send(v).is_ok(), "the answer hub is open");
                    }
                })
            };
            let ping_tx = ping.sender();
            group.bench_function("hub_round_trip_1", |b| {
                let mut next = 0u64;
                b.iter(|| {
                    next += 1;
                    assert!(ping_tx.send(next).is_ok(), "the request hub is open");
                    black_box(pong_rx.recv().expect("an answer"))
                })
            });
            ping.shutdown();
            responder.join().expect("the responder thread");
        }

        // A burst taken by four consumers that go idle between bursts.
        {
            let hub = NotifyHub::<u64>::new(1024, 4);
            let taken = Arc::new(AtomicUsize::new(0));
            let consumers: Vec<_> = (0..4)
                .map(|_| {
                    let hub = hub.clone();
                    let taken = Arc::clone(&taken);
                    std::thread::spawn(move || {
                        let rx = hub.register_consumer();
                        while let Some(v) = rx.recv() {
                            black_box(v);
                            taken.fetch_add(1, Ordering::Release);
                        }
                    })
                })
                .collect();
            let tx = hub.sender();
            group.bench_function("hub_drain_4", |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let target = taken.load(Ordering::Acquire) + BURST;
                        let start = Instant::now();
                        for i in 0..BURST {
                            assert!(tx.send(i as u64).is_ok(), "the hub is open");
                        }
                        while taken.load(Ordering::Acquire) < target {
                            std::hint::spin_loop();
                        }
                        total += start.elapsed();
                    }
                    total
                })
            });
            hub.shutdown();
            for consumer in consumers {
                consumer.join().expect("a consumer thread");
            }
        }

        // n tasks on a fresh pool of n workers, until all have started.
        group.sample_size(10);
        group.bench_function("io_pool_start_n", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let pool = IoPool::new(logical);
                    let started = Arc::new(AtomicUsize::new(0));
                    let start = Instant::now();
                    for _ in 0..logical {
                        let started = Arc::clone(&started);
                        pool.submit(move || {
                            started.fetch_add(1, Ordering::Release);
                            std::thread::sleep(TASK_WORK);
                        });
                    }
                    while started.load(Ordering::Acquire) < logical {
                        std::thread::yield_now();
                    }
                    total += start.elapsed();
                    drop(pool);
                }
                total
            })
        });
        group.sample_size(100);

        // Short tasks through a standing pool.
        {
            let pool = IoPool::new(logical);
            let done = Arc::new(AtomicUsize::new(0));
            group.bench_function("io_pool_drain_n", |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let target = done.load(Ordering::Acquire) + POOL_TASKS;
                        let start = Instant::now();
                        for _ in 0..POOL_TASKS {
                            let done = Arc::clone(&done);
                            pool.submit(move || {
                                done.fetch_add(1, Ordering::Release);
                            });
                        }
                        while done.load(Ordering::Acquire) < target {
                            std::hint::spin_loop();
                        }
                        total += start.elapsed();
                    }
                    total
                })
            });
            drop(pool);
        }

        // The processor time an idle pool spends; only meaningful with no
        // spinners in the process.
        if spinners == 0 {
            let pool = IoPool::new(logical);
            std::thread::sleep(Duration::from_millis(100));
            eprintln!(
                "notify_hub idle_wait before {:?}",
                wait_controller().report()
            );
            group.sample_size(20);
            group.bench_function("io_pool_idle_n", |b| {
                b.iter_custom(|iters| {
                    let mut total = 0u64;
                    for _ in 0..iters {
                        let before = process_cpu();
                        std::thread::sleep(IDLE_WINDOW);
                        total += process_cpu() - before;
                    }
                    Duration::from_nanos(total)
                })
            });
            eprintln!(
                "notify_hub idle_wait after {:?}",
                wait_controller().report()
            );
            group.sample_size(100);
            drop(pool);
        }

        group.finish();
        drop(load);
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);

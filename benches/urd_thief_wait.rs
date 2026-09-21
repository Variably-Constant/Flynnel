//! A/B microbench: URD thief wake latency across the wait strategies
//! its mailbox can idle on, `PAUSE`-spin, WAITPKG and MONITORX.
//!
//! Two threads:
//! - **Thief**: sits in `wait_and_drain` on one mailbox.
//! - **Owner**: sleeps a fixed inter-arrival, then publishes to it.
//!
//! The measurement is wall-clock from the publish to the thief's
//! observable return, so it covers the wait primitive's exit plus the
//! coherence transfer on the mailbox's state line.
//!
//! ## What this bench is for, and it is not the same question the
//! parker bench asks
//!
//! The parker traded a kernel park for a monitor wait, and a syscall
//! is slower than an in-core wait on any host, so that arm only had to
//! be confirmed. The thief traded a `PAUSE`-spin, which is the fastest
//! wake there is: it never stopped looking, so it sees the store on
//! the next poll. A monitor wait cannot beat that on latency alone.
//!
//! What it buys is the core. A spinning thief occupies a logical CPU
//! and issues into the pipeline the whole time it waits, which on an
//! SMT sibling is taken directly out of the thread beside it. So the
//! two arms are not ordered by the idle rows, and the load rows are
//! the ones that decide it: they are where a spinning thief's cost is
//! paid by somebody.
//!
//! Reported rather than concluded. If the monitor wait is slower on
//! both arms on a host, that is the reading, and the thief's default
//! should follow the reading rather than this comment.
//!
//! ## Bench-audit
//!
//! - **Same payload across A/B**: same mailbox, same three-item
//!   publish, same inter-arrival, same thread pair. Only the strategy
//!   the thief idles on differs, set through
//!   `UrdDeque::set_wait_strategy`.
//! - **Every arm in both conditions**: idle, and against busy threads
//!   occupying half the host.
//! - **The named primitive is exercised**: the MONITORX arm reaches
//!   MONITORX + MWAITX against the mailbox's state line, the WAITPKG
//!   arm reaches UMONITOR + UMWAIT against the same line, and the
//!   PauseSpin arm reaches `std::hint::spin_loop`.

#![allow(clippy::missing_docs_in_private_items)]

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::backend::shared_mem::khpd::LineItem;
use flynnel::backend::shared_mem::urd::{UrdDeque, WaitStrategy};

/// Busy threads occupying half the host, joined when this is dropped.
///
/// Carries more weight here than in the parker bench. The difference
/// between a spinning thief and a parked one is what the rest of the
/// machine gets back, and on an idle host there is nothing waiting to
/// take it, so the idle rows cannot see the effect at all.
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
            // than dropped, because the run would otherwise look like
            // a clean A/B.
            if t.join().is_err() {
                eprintln!(
                    "urd_thief_wait: a load thread panicked, so this run's load arm carried less load than it reports"
                );
            }
        }
    }
}

fn temp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    p.push(format!("flynnel_urdbench_{pid}_{nonce}_{name}.bin"));
    p
}

fn item(v: u64) -> LineItem {
    LineItem { a: v, b: v }
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
    let mut group = c.benchmark_group(format!("urd_thief_{label}_{arm}_gap_{gap_us}us"));
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("publish_to_drained", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let path = temp_path(label);
                let mut owner = UrdDeque::create(&path, 1).expect("create urd");
                owner.set_wait_strategy(strategy);
                let thief_side = Arc::new(owner);

                let drained = Arc::new(AtomicU32::new(0));
                let drained_thread = Arc::clone(&drained);
                let urd = Arc::clone(&thief_side);
                let thief = std::thread::spawn(move || {
                    // A deadline far past the inter-arrival, so what
                    // ends the wait is the publish rather than the
                    // timer. The PauseSpin arm ignores it entirely.
                    let deadline = u64::MAX;
                    let got = urd.wait_and_drain(0, deadline);
                    drained_thread.store(1, Ordering::Release);
                    got
                });

                // Let the thief reach the wait before publishing, so
                // the measurement covers a wake and not a mailbox
                // that was already ready.
                std::thread::sleep(Duration::from_micros(gap_us));
                let items = [item(1), item(2), item(3)];
                let t0 = Instant::now();
                thief_side.publish_to(0, &items).expect("publish");
                while drained.load(Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }
                total += t0.elapsed();
                thief.join().expect("thief joins");

                if let Err(e) = std::fs::remove_file(&path) {
                    eprintln!("urd_thief_wait: could not remove {}: {e}", path.display());
                }
            }
            black_box(total)
        });
    });
    group.finish();
}

fn bench_all(c: &mut Criterion) {
    let waitpkg = flynnel::cpu_info::has_waitpkg();
    let monitorx = flynnel::cpu_info::has_monitorx();

    if !waitpkg {
        eprintln!(
            "urd_thief_wait: WAITPKG arm skipped - host has no WAITPKG \
             (cpuid leaf 7 ECX bit 5 = 0)."
        );
    }
    if !monitorx {
        eprintln!(
            "urd_thief_wait: MONITORX arm skipped - host has no MONITORX \
             (cpuid Fn8000_0001 ECX bit 29 = 0)."
        );
    }
    if !waitpkg && !monitorx {
        eprintln!(
            "urd_thief_wait: this host has neither monitor-wait, so the \
             PauseSpin rows are the only path its thief can take and \
             there is no A/B in this run."
        );
    }

    for loaded in [false, true] {
        for gap_us in [50, 500] {
            // PauseSpin is the control: it is what the thief does
            // today on every host that reaches this code.
            bench_strategy(c, "pausespin", WaitStrategy::PauseSpin, gap_us, loaded);

            if waitpkg {
                bench_strategy(c, "waitpkg", WaitStrategy::Waitpkg, gap_us, loaded);
            }
            if monitorx {
                bench_strategy(c, "monitorx", WaitStrategy::Monitorx, gap_us, loaded);
            }
        }
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);

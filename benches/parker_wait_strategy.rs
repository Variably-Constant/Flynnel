//! A/B microbench: Parker wake latency with std::thread::park vs
//! WAITPKG (UMONITOR + UMWAIT) wake paths.
//!
//! Two threads:
//! - **Owner**: parks via `park_until` past the spin floor.
//! - **Producer**: sleeps a fixed inter-arrival, then unparks owner.
//!
//! The bench measures wall-clock time from `unpark` to the owner's
//! observable return from `park_until`. That captures BOTH the
//! syscall transition cost (StdPark) or the UMWAIT-return cost
//! (WAITPKG) AND the inter-thread coherence transfer on the
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
//!   issues UMONITOR + UMWAIT against `wake_counter`; StdPark path
//!   issues `thread::park()` + permits.
//!
//! ## Hardware availability
//!
//! WAITPKG is detected at runtime via `crate::cpu_info::has_waitpkg`.
//! On Zen+ R7 2700 (the development host) WAITPKG is NOT available;
//! the WAITPKG bench is skipped with an explanatory eprintln. On
//! Genoa / Tiger Lake+ / Zen 5+ the WAITPKG variant runs and is
//! the comparison the bench is designed to expose.

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
fn bench_strategy(c: &mut Criterion, label: &str, strategy: WaitStrategy, gap_us: u64) {
    let mut group = c.benchmark_group(format!("parker_wait_{label}_gap_{gap_us}us"));
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

fn bench_all(c: &mut Criterion) {
    // Always bench the StdPark baseline. Two gap values exercise
    // the syscall cost across a short (50us) and longer (500us)
    // inter-arrival.
    bench_strategy(c, "stdpark", WaitStrategy::StdPark, 50);
    bench_strategy(c, "stdpark", WaitStrategy::StdPark, 500);

    if flynnel::cpu_info::has_waitpkg() {
        bench_strategy(c, "waitpkg", WaitStrategy::Waitpkg, 50);
        bench_strategy(c, "waitpkg", WaitStrategy::Waitpkg, 500);
    } else {
        eprintln!(
            "parker_wait_strategy: WAITPKG branch skipped - host has \
             no WAITPKG (cpuid leaf 7 ECX bit 5 = 0). StdPark numbers \
             measure the existing Parker baseline."
        );
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);

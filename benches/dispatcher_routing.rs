//! Micro-bench for the [`CrossProcessDispatcher`] routing layer.
//!
//! Three measurements, all single-process so the dispatcher's routing
//! overhead is isolated from any cross-process coherence cost:
//!
//! 1. **dispatcher_pick_only**: pure `pick_with_fallback` cost per
//!    call - no actual dispatch. Measures the routing-table lookup +
//!    payload-validation overhead. Should be < 50 ns/call.
//! 2. **dispatcher_through_chase_lev**: full `dispatch_marshal` +
//!    `wait_handle` round-trip routed through the Chase-Lev backend.
//!    The dispatcher adds one HashMap lookup + one Option-unwrap to
//!    the bare Chase-Lev call - the bench measures whether that
//!    overhead is observable against the ~500 ns round-trip baseline.
//! 3. **dispatcher_through_khpd_batched**: `dispatch_marshal_batch`
//!    of 64 items routed through KHPD. Runs the producer-fast
//!    shape KHPD is optimized for; the dispatcher must not
//!    materially regress KHPD's measured win zone.
//!
//! ## Bench-audit notes
//!
//! - All three sites construct exactly the backends they need; no
//!   surplus locks, no extra allocations, no JSON-style overhead.
//! - The payload is the same `(I32, I32) -> u32` adder used by every
//!   per-variant bench so numbers are directly comparable.
//! - `dispatcher_pick_only` does NOT install any backend - it stresses
//!   only the routing table itself, so the `pick_with_fallback` fallback
//!   path returns `Err(BackendError::NotSupported)` for shapes whose
//!   primary variant has no slot; the bench accepts that as the
//!   measurement.

#![allow(clippy::missing_docs_in_private_items)]

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::backend::shared_mem::{
    CrossProcessDispatcher, DispatcherRoutingTable, SharedMemoryChaseLevBackend,
    SharedMemoryKhpdBackend, WorkloadShape, hash_name, register, unregister,
};

const PRODUCER_K: usize = 64;

fn temp_path(label: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    p.push(format!("flynnel_bench_dr_{pid}_{nonce}_{label}.bin"));
    p
}

fn raw_adder(a: u32, b: u32) -> [u8; 8] {
    let mut p = [0u8; 8];
    p[..4].copy_from_slice(&a.to_le_bytes());
    p[4..].copy_from_slice(&b.to_le_bytes());
    p
}

fn register_adder(id: u32) {
    register(id, |args| {
        let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
        let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
        Ok(a.wrapping_add(b).to_le_bytes().to_vec())
    });
}

/// (1) Pure routing overhead. No backends, just the table lookup.
fn bench_pick_only(c: &mut Criterion) {
    let dispatcher = CrossProcessDispatcher::builder()
        .with_table(DispatcherRoutingTable::default_heuristic())
        .build();

    let shape_rr = WorkloadShape::request_reply(8);
    let shape_pf = WorkloadShape::producer_fast(8, 64);
    let shape_mt = WorkloadShape::multi_thief(8, 4, 64);

    let mut group = c.benchmark_group("dispatcher_pick_only");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("request_reply", |b| {
        b.iter(|| black_box(dispatcher.pick_with_fallback(black_box(&shape_rr))));
    });
    group.bench_function("producer_fast", |b| {
        b.iter(|| black_box(dispatcher.pick_with_fallback(black_box(&shape_pf))));
    });
    group.bench_function("multi_thief", |b| {
        b.iter(|| black_box(dispatcher.pick_with_fallback(black_box(&shape_mt))));
    });
    group.finish();
}

/// (2) Dispatcher overhead on a Chase-Lev round-trip.
fn bench_through_chase_lev(c: &mut Criterion) {
    let dp = temp_path("cl_d");
    let lp = temp_path("cl_l");
    let cl = Arc::new(
        SharedMemoryChaseLevBackend::create(0, &dp, &lp, 256, 1024).expect("create cl"),
    );
    let id = hash_name("flynnel.bench.dr.cl_adder");
    register_adder(id);

    // Spawn a drain thread so latches actually get published.
    let stop = Arc::new(AtomicBool::new(false));
    let wbe = Arc::clone(&cl);
    let wstop = Arc::clone(&stop);
    let w = std::thread::spawn(move || {
        while !wstop.load(Ordering::Relaxed) {
            match wbe.drain_one() {
                Ok(Some(())) => {}
                Ok(None) => std::hint::spin_loop(),
                Err(_) => return,
            }
        }
    });

    let dispatcher = CrossProcessDispatcher::builder()
        .with_table(DispatcherRoutingTable::default_heuristic())
        .with_chase_lev(Arc::clone(&cl))
        .build();
    let shape = WorkloadShape::request_reply(8);
    let args = raw_adder(3, 4);

    let mut group = c.benchmark_group("dispatcher_through_chase_lev");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("dispatch_wait_round_trip", |b| {
        b.iter(|| {
            let h = dispatcher
                .dispatch_marshal(&shape, id, &args)
                .expect("dispatch");
            dispatcher.wait_handle(h, 1024).expect("wait").expect("ok");
            black_box(h);
        });
    });
    group.finish();

    stop.store(true, Ordering::Relaxed);
    w.join().expect("worker join");
    unregister(id);
    std::fs::remove_file(&dp).ok();
    std::fs::remove_file(&lp).ok();
}

/// (3) Dispatcher overhead on a KHPD batched producer-fast burst.
fn bench_through_khpd_batched(c: &mut Criterion) {
    let dp = temp_path("khpd_d");
    let lp = temp_path("khpd_l");
    let khpd = Arc::new(
        SharedMemoryKhpdBackend::create(0, &dp, &lp, 1024, 4096).expect("create khpd"),
    );
    let id = hash_name("flynnel.bench.dr.khpd_adder");
    register_adder(id);

    let stop = Arc::new(AtomicBool::new(false));
    let wbe = Arc::clone(&khpd);
    let wstop = Arc::clone(&stop);
    let w = std::thread::spawn(move || {
        while !wstop.load(Ordering::Relaxed) {
            match wbe.drain_one_line() {
                Ok(Some(_)) => {}
                Ok(None) => std::hint::spin_loop(),
                Err(_) => return,
            }
        }
    });

    let dispatcher = CrossProcessDispatcher::builder()
        .with_table(DispatcherRoutingTable::default_heuristic())
        .with_khpd(Arc::clone(&khpd))
        .build();
    let shape = WorkloadShape::producer_fast(8, PRODUCER_K as u32);
    let args = raw_adder(3, 4);
    let items: Vec<(u32, &[u8])> = (0..PRODUCER_K).map(|_| (id, args.as_slice())).collect();

    let mut group = c.benchmark_group("dispatcher_through_khpd_batched");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(criterion::Throughput::Elements(PRODUCER_K as u64));
    group.bench_function("dispatch_batch_k64_no_wait", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let start = std::time::Instant::now();
                let handles = dispatcher
                    .dispatch_marshal_batch(&shape, &items)
                    .expect("dispatch");
                total += start.elapsed();
                if let Some(last) = handles.last() {
                    dispatcher
                        .wait_handle(*last, 1024)
                        .expect("wait")
                        .expect("ok");
                }
            }
            black_box(total)
        });
    });
    group.finish();

    stop.store(true, Ordering::Relaxed);
    w.join().expect("worker join");
    unregister(id);
    std::fs::remove_file(&dp).ok();
    std::fs::remove_file(&lp).ok();
}

fn bench_all(c: &mut Criterion) {
    bench_pick_only(c);
    bench_through_chase_lev(c);
    bench_through_khpd_batched(c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);

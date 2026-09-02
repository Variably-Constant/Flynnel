//! End-to-end adaptive-routing tests replaying the workload shapes
//! real Flynnel consumers dispatch, with real compute in every leaf.
//! Each test asserts the observable routing outcome: wall-clock
//! parallel speedup against a serial baseline run of the identical
//! closure, plus tier / plan-field / site observables where the
//! surface exposes them.
//!
//! Shapes covered (each named for the consumer pattern it replays):
//! - lexer fan-out: `JobPlan::new(0, n).with_leaf_shape(PortCompute)`
//!   + `for_each_chunk_indexed_min_leaf(_, _, 1, ..)` over heavy parts
//! - model-training fan-out: same shape at n = 8 (the batch >= 8
//!   explicit-shape floor)
//! - hinted heavy rows: `with_estimated_per_item_ns` on a small batch
//! - hint-less probe rescue: `JobPlan::new(0, 64)` heavy items, no hint
//! - hint-less indexed collect: `collect_indexed(_, n, 1, ..)` heavy
//! - SMT opt-in small fan-out: `.with_smt()` on a small class count
//! - hint-less small fan-out: `JobPlan::new(0, 12)` heavy items
//! - streaming byte scan: `set_profile(_, _, Streaming)` over a big
//!   byte buffer
//! - latency-bound profile small batch: `set_profile(_, _,
//!   LatencyBound)` at n = 64
//! - plan-free two-way join: `flat::join` with two heavy halves
//! - cooperative fan-out: `cooperative_join_n` with per-worker roles
//! - variant racing: `race_variants` fast/faithful/correct
//! - chunked reduce: `reduce_chunks` histogram with path observable
//! - site learning stability: repeat dispatches stay parallel and the
//!   site converges to a learned class
//! - hybrid pair: `join_hybrid` through the default CPU backend
//!
//! Timing tests take a shared mutex so no two of them overlap on the
//! worker pool; run cargo test on this binary normally.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use flynnel::sched::par_iter::{
    collect_indexed, for_each_chunk, for_each_chunk_indexed_min_leaf, last_reduce_chunks_path,
    reduce_chunks,
};
use flynnel::sched::{JobPlan, SchedTier, pick_tier};
use flynnel::{DispatchProfile, LeafShape, Variant};

/// Serializes the timing tests so speedup measurements never share
/// the worker pool.
fn pool_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// splitmix64 step; the compute kernel every heavy leaf spins on.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// CPU-bound work unit: `iters` dependent splitmix64 steps folded
/// into a checksum the assertions compare, so the loop cannot be
/// optimized away. ~1-2 ns per iteration on 2020s x86.
fn spin_work(seed: u64, iters: u64) -> u64 {
    let mut acc = seed;
    for _ in 0..iters {
        acc = splitmix64(acc);
    }
    std::hint::black_box(acc)
}

/// Median of three timed runs of `f`.
fn median3<F: FnMut()>(mut f: F) -> Duration {
    let mut samples: Vec<Duration> = (0..3)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed()
        })
        .collect();
    samples.sort();
    samples[1]
}

/// Spins the worker pool up (arena init, JEC wake) so the first
/// timed dispatch does not pay one-time startup.
fn warm_pool() {
    let plan = JobPlan::new(0, 4096);
    let mut v = vec![0u64; 4096];
    for_each_chunk_indexed_min_leaf(&plan, &mut v, 1, |start, slots| {
        for (i, s) in slots.iter_mut().enumerate() {
            *s = spin_work((start + i) as u64, 1_000);
        }
    });
    std::hint::black_box(v);
}

fn require_parallel_host() {
    let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1);
    assert!(
        cores >= 4,
        "these tests measure parallel speedup and need >= 4 logical cores; host has {cores}"
    );
}

/// Asserts the parallel run beat the serial baseline by at least
/// 1/ratio. `ratio` is deliberately loose against CI noise; real
/// speedups on an 8-core host are 4x+ for the heavy fan-outs.
fn assert_speedup(name: &str, serial: Duration, parallel: Duration, ratio: f64) {
    assert!(
        parallel.as_secs_f64() < serial.as_secs_f64() * ratio,
        "{name}: expected parallel < {ratio} x serial; serial={serial:?} parallel={parallel:?}"
    );
}

/// Heavy-item iteration count. ~1.5-3 ms per item depending on host.
const HEAVY_ITERS: u64 = 1_500_000;

/// Serial baseline for an n-item fan-out of `spin_work(seed, iters)`.
fn serial_fanout(n: usize, iters: u64) -> (Duration, u64) {
    let mut checksum = 0u64;
    let wall = median3(|| {
        checksum = 0;
        for i in 0..n {
            checksum ^= spin_work(i as u64, iters);
        }
    });
    (wall, checksum)
}

#[test]
fn lexer_fanout_portcompute_n16() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let n = 16usize;
    let (serial, expect) = serial_fanout(n, HEAVY_ITERS);

    let plan = JobPlan::new(0, n as u32).with_leaf_shape(LeafShape::PortCompute);
    assert_eq!(
        pick_tier(&plan, flynnel::numa_topology()),
        SchedTier::Local,
        "explicit leaf shape at batch >= 8 routes to the pool"
    );

    let mut out = vec![0u64; n];
    let parallel = median3(|| {
        out.iter_mut().for_each(|s| *s = 0);
        for_each_chunk_indexed_min_leaf(&plan, &mut out, 1, |start, slots| {
            for (i, s) in slots.iter_mut().enumerate() {
                *s = spin_work((start + i) as u64, HEAVY_ITERS);
            }
        });
    });
    let got = out.iter().fold(0u64, |a, b| a ^ b);
    assert_eq!(got, expect);
    assert_speedup("lexer_fanout_portcompute_n16", serial, parallel, 0.7);
}

#[test]
fn model_training_fanout_portcompute_n8() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let n = 8usize;
    let (serial, expect) = serial_fanout(n, 2 * HEAVY_ITERS);

    let plan = JobPlan::new(0, n as u32).with_leaf_shape(LeafShape::PortCompute);
    assert_eq!(pick_tier(&plan, flynnel::numa_topology()), SchedTier::Local);

    let mut out = vec![0u64; n];
    let parallel = median3(|| {
        for_each_chunk_indexed_min_leaf(&plan, &mut out, 1, |start, slots| {
            for (i, s) in slots.iter_mut().enumerate() {
                *s = spin_work((start + i) as u64, 2 * HEAVY_ITERS);
            }
        });
    });
    let got = out.iter().fold(0u64, |a, b| a ^ b);
    assert_eq!(got, expect);
    assert_speedup("model_training_fanout_portcompute_n8", serial, parallel, 0.7);
}

#[test]
fn hinted_heavy_rows_n8() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let n = 8usize;
    let iters = 2 * HEAVY_ITERS;
    let (serial, expect) = serial_fanout(n, iters);

    // Per-item hint well above the 50 us heavy-override threshold.
    let plan = JobPlan::new(0, n as u32).with_estimated_per_item_ns(3_000_000);
    assert_eq!(
        pick_tier(&plan, flynnel::numa_topology()),
        SchedTier::Local,
        "explicit heavy per-item hint promotes a small batch out of Inline"
    );

    let mut out = vec![0u64; n];
    let parallel = median3(|| {
        for_each_chunk_indexed_min_leaf(&plan, &mut out, 1, |start, slots| {
            for (i, s) in slots.iter_mut().enumerate() {
                *s = spin_work((start + i) as u64, iters);
            }
        });
    });
    let got = out.iter().fold(0u64, |a, b| a ^ b);
    assert_eq!(got, expect);
    assert_speedup("hinted_heavy_rows_n8", serial, parallel, 0.7);
}

#[test]
fn hintless_probe_rescue_n64() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let n = 64usize;
    let (serial, expect) = serial_fanout(n, HEAVY_ITERS);

    // No hint, no shape: the entry probe must measure the per-item
    // cost and promote the bulk dispatch to the pool.
    let plan = JobPlan::new(0, n as u32);
    let mut out = vec![0u64; n];
    let parallel = median3(|| {
        for_each_chunk_indexed_min_leaf(&plan, &mut out, 1, |start, slots| {
            for (i, s) in slots.iter_mut().enumerate() {
                *s = spin_work((start + i) as u64, HEAVY_ITERS);
            }
        });
    });
    let got = out.iter().fold(0u64, |a, b| a ^ b);
    assert_eq!(got, expect);
    assert_speedup("hintless_probe_rescue_n64", serial, parallel, 0.7);
}

#[test]
fn hintless_collect_indexed_n48() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let n = 48usize;
    let (serial, expect) = serial_fanout(n, HEAVY_ITERS);

    let plan = JobPlan::new(0, n as u32);
    let mut got = 0u64;
    let parallel = median3(|| {
        let v: Vec<u64> = collect_indexed(&plan, n, 1, |i| spin_work(i as u64, HEAVY_ITERS));
        got = v.iter().fold(0u64, |a, b| a ^ b);
    });
    assert_eq!(got, expect);
    assert_speedup("hintless_collect_indexed_n48", serial, parallel, 0.7);
}

#[test]
fn with_smt_small_fanout_n24() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let n = 24usize;
    let (serial, expect) = serial_fanout(n, HEAVY_ITERS);

    let plan = JobPlan::new(0, n as u32).with_smt();
    assert_eq!(
        pick_tier(&plan, flynnel::numa_topology()),
        SchedTier::Local,
        "caller SMT opt-in routes a small heavy batch to the pool"
    );

    let mut out = vec![0u64; n];
    let parallel = median3(|| {
        for_each_chunk_indexed_min_leaf(&plan, &mut out, 1, |start, slots| {
            for (i, s) in slots.iter_mut().enumerate() {
                *s = spin_work((start + i) as u64, HEAVY_ITERS);
            }
        });
    });
    let got = out.iter().fold(0u64, |a, b| a ^ b);
    assert_eq!(got, expect);
    assert_speedup("with_smt_small_fanout_n24", serial, parallel, 0.7);
}

#[test]
fn hintless_small_fanout_n12() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let n = 12usize;
    let iters = 2 * HEAVY_ITERS;
    let (serial, expect) = serial_fanout(n, iters);

    // The hint-less small-batch classifier picks LatencyBound
    // (SMT on) for batch <= 32; the tier picker honors it.
    let plan = JobPlan::new(0, n as u32);
    assert!(plan.use_smt, "hint-less batch <= 32 classifies LatencyBound");
    assert_eq!(
        pick_tier(&plan, flynnel::numa_topology()),
        SchedTier::Local,
        "classifier SMT signal promotes the small hint-less batch"
    );

    let mut out = vec![0u64; n];
    let parallel = median3(|| {
        for_each_chunk_indexed_min_leaf(&plan, &mut out, 1, |start, slots| {
            for (i, s) in slots.iter_mut().enumerate() {
                *s = spin_work((start + i) as u64, iters);
            }
        });
    });
    let got = out.iter().fold(0u64, |a, b| a ^ b);
    assert_eq!(got, expect);
    assert_speedup("hintless_small_fanout_n12", serial, parallel, 0.7);
}

#[test]
fn tiny_light_batches_stay_inline() {
    // Guard in the opposite direction: genuinely tiny light work
    // must not be promoted to the pool.
    let topo = flynnel::numa_topology();
    let light = JobPlan::new(0, 4).with_estimated_per_item_ns(10);
    assert_eq!(pick_tier(&light, topo), SchedTier::Inline);
    let mid = JobPlan::new(0, 100).with_estimated_per_item_ns(20);
    assert_eq!(pick_tier(&mid, topo), SchedTier::Inline);
}

#[test]
fn streaming_byte_scan_32mb() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let len = 32 * 1024 * 1024usize;
    let data: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();

    let count_serial = |buf: &[u8]| buf.iter().filter(|&&b| b == 42).count() as u64;
    let mut expect = 0u64;
    let serial = median3(|| {
        expect = count_serial(&data);
    });

    let plan = JobPlan::set_profile(0, len as u32, DispatchProfile::Streaming);
    assert!(!plan.use_smt, "streaming profile keeps SMT siblings parked");
    assert_eq!(pick_tier(&plan, flynnel::numa_topology()), SchedTier::Local);

    let mut owned = data.clone();
    let total = std::sync::atomic::AtomicU64::new(0);
    let parallel = median3(|| {
        total.store(0, std::sync::atomic::Ordering::Relaxed);
        for_each_chunk(&plan, &mut owned, |slice: &mut [u8]| {
            let c = slice.iter().filter(|&&b| b == 42).count() as u64;
            total.fetch_add(c, std::sync::atomic::Ordering::Relaxed);
        });
    });
    assert_eq!(total.load(std::sync::atomic::Ordering::Relaxed), expect);
    assert_speedup("streaming_byte_scan_32mb", serial, parallel, 0.8);
}

#[test]
fn latencybound_profile_small_batch_n64() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let n = 64usize;
    let iters = 200_000u64;
    let (serial, expect) = serial_fanout(n, iters);

    let plan = JobPlan::set_profile(0, n as u32, DispatchProfile::LatencyBound);
    assert!(plan.use_smt);
    assert_eq!(
        pick_tier(&plan, flynnel::numa_topology()),
        SchedTier::Local,
        "explicit LatencyBound profile routes a small batch to the pool"
    );

    let mut out = vec![0u64; n];
    let parallel = median3(|| {
        for_each_chunk_indexed_min_leaf(&plan, &mut out, 1, |start, slots| {
            for (i, s) in slots.iter_mut().enumerate() {
                *s = spin_work((start + i) as u64, iters);
            }
        });
    });
    let got = out.iter().fold(0u64, |a, b| a ^ b);
    assert_eq!(got, expect);
    assert_speedup("latencybound_profile_small_batch_n64", serial, parallel, 0.7);
}

#[test]
fn flat_join_two_heavy_halves() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let iters = 4 * HEAVY_ITERS;
    let mut expect = (0u64, 0u64);
    let serial = median3(|| {
        expect = (spin_work(1, iters), spin_work(2, iters));
    });

    let mut got = (0u64, 0u64);
    let parallel = median3(|| {
        got = flynnel::flat::join(|| spin_work(1, iters), || spin_work(2, iters));
    });
    assert_eq!(got, expect);
    assert_speedup("flat_join_two_heavy_halves", serial, parallel, 0.85);
}

#[test]
fn cooperative_fanout_n4() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let iters = 2 * HEAVY_ITERS;
    let mut expect = [0u64; 4];
    let serial = median3(|| {
        for (i, e) in expect.iter_mut().enumerate() {
            *e = spin_work(100 + i as u64, iters);
        }
    });

    let plan = JobPlan::new(0, 4);
    let mut results: Vec<u64> = Vec::new();
    let parallel = median3(|| {
        let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..4u64)
            .map(|i| {
                Box::new(move || spin_work(100 + i, iters)) as Box<dyn FnOnce() -> u64 + Send>
            })
            .collect();
        results = flynnel::cooperative_join_n(&plan, closures);
    });
    assert_eq!(results.as_slice(), expect.as_slice(), "results keep closure order");
    assert_speedup("cooperative_fanout_n4", serial, parallel, 0.7);
}

#[test]
fn race_variants_correct_always_wins_something() {
    let _g = pool_guard();
    warm_pool();
    let plan = JobPlan::new(0, 1024);
    // Fast declines (returns None); faithful and correct compute the
    // same answer. The winner must carry the right value and a
    // variant tag from a closure that can actually produce one.
    let (value, variant) = flynnel::race_variants(
        &plan,
        |_cancel| -> Option<u64> { None },
        |_cancel| Some(spin_work(7, 200_000)),
        |_cancel| spin_work(7, 200_000),
    );
    assert_eq!(value, spin_work(7, 200_000));
    assert!(
        matches!(variant, Variant::Faithful | Variant::Correct),
        "fast declined, so the winner is faithful or correct; got {variant:?}"
    );
}

#[test]
fn reduce_chunks_histogram_1m() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let data: Vec<u8> = (0..1_000_000usize).map(|i| (i % 256) as u8).collect();
    let mut expect = [0u64; 256];
    for &b in &data {
        expect[b as usize] += 1;
    }

    let plan = JobPlan::new(0, data.len() as u32);
    let hist = reduce_chunks(
        &plan,
        &data,
        || [0u64; 256],
        |mut acc, slice| {
            for &b in slice {
                acc[b as usize] += 1;
            }
            acc
        },
        |mut a, b| {
            for i in 0..256 {
                a[i] += b[i];
            }
            a
        },
    );
    assert_eq!(hist, expect);
    assert!(
        last_reduce_chunks_path().is_some(),
        "reduce_chunks records which routing path served the call"
    );
}

#[test]
fn site_learning_keeps_hintless_heavy_parallel() {
    let _g = pool_guard();
    require_parallel_host();
    warm_pool();
    let n = 64usize;
    let (serial, expect) = serial_fanout(n, HEAVY_ITERS);

    let site = flynnel::caller_site();
    let mut out = vec![0u64; n];
    let mut worst = Duration::ZERO;
    for round in 0..10 {
        let plan = JobPlan::new(0, n as u32).with_site(site);
        let t0 = Instant::now();
        for_each_chunk_indexed_min_leaf(&plan, &mut out, 1, |start, slots| {
            for (i, s) in slots.iter_mut().enumerate() {
                *s = spin_work((start + i) as u64, HEAVY_ITERS);
            }
        });
        let wall = t0.elapsed();
        let got = out.iter().fold(0u64, |a, b| a ^ b);
        assert_eq!(got, expect, "round {round}");
        if wall > worst {
            worst = wall;
        }
    }
    // Every round, including the ones after the site has learned a
    // class, must stay parallel: learning may never demote a
    // measured-heavy site back to serial.
    assert_speedup(
        "site_learning_keeps_hintless_heavy_parallel(worst round)",
        serial,
        worst,
        0.7,
    );
    assert!(
        site.get().learned_class().is_some(),
        "10 rounds x 64 recorded leaves converge the site classifier"
    );
}

#[test]
fn hybrid_pair_cpu_backend() {
    let _g = pool_guard();
    warm_pool();
    let plan = JobPlan::new(0, 2048);
    let (cpu_half, other_half) = flynnel::join_hybrid(
        &plan,
        || spin_work(11, 400_000),
        || spin_work(22, 400_000),
    );
    assert_eq!(cpu_half, spin_work(11, 400_000));
    assert_eq!(other_half, spin_work(22, 400_000));
}

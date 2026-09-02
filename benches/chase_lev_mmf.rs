//! Per-call latency bench for the MMF Chase-Lev cross-process dispatch
//! backend.
//!
//! Four bench groups, all single-process so both halves of the
//! cross-thread coherence pair can be pinned to specific cores.
//!
//! 1. **chase_lev_mmf** - end-to-end round-trip (originator dispatches,
//!    a single worker thread drains + publishes the reply through the
//!    MMF latch, originator polls the latch). Unpinned; the OS
//!    scheduler picks core placement.
//! 2. **substrate_only** - storage-layer-only measurements: Chase-Lev
//!    push+pop same-thread, Chase-Lev push+steal same-thread (CAS-on-top
//!    path that cross-thread / cross-process thieves use), and latch
//!    alloc+publish+read same-thread.
//! 3. **chase_lev_pinned_smt_siblings / intra_ccx / cross_ccx** - the
//!    end-to-end round-trip with originator + drain pinned to a
//!    specific pair of `CoreId`s, one pair per coherence tier so the
//!    bounce-latency floor is isolated from OS scheduling jitter.
//!
//! ## Bench-audit notes (per the project hard rule)
//!
//! - **Same payload**: every round-trip carries `(I32, I32)` =
//!   `2 + 4 + 1 + 4 = 11` encoded bytes after `wire::encode_args`.
//! - **Same handler shape**: a registered `adder` closure returns a
//!   4-byte u32 reply.
//! - **Same worker shape across tiers**: one drain thread, polling
//!   with `std::hint::spin_loop()` between pops. Only the pinning
//!   varies across the tiered variants.
//! - **No surplus locks or allocs in the originator hot path**: the
//!   benched closure inside `iter` only does dispatch + wait; reusable
//!   buffers live outside the iter closure scope.

#![allow(clippy::missing_docs_in_private_items)]

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use core_affinity::CoreId;
use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::backend::shared_mem::{
    MmfChaseLevDeque, MmfLatchArena, SharedMemoryChaseLevBackend, hash_name, register, unregister,
};
use flynnel::backend::shared_mem::chase_lev_mmf::{ARGS_INLINE_BYTES, RemoteJobSlot, Steal};
use flynnel::backend::shared_mem::wire;
use flynnel::backend::KernelArg;
use flynnel::numa_topology;

fn temp_path(label: &str, n: u32) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    p.push(format!("flynnel_bench_clmmf_{pid}_{nonce}_{label}_{n}.bin"));
    p
}

/// Pairs of pinned cores picked by [`pick_pinned_pairs`], one per
/// coherence tier.
struct PinnedPairs {
    /// SMT-sibling pair: cores that share an L1 cache (two logical
    /// threads on one physical core). Cheapest cross-thread tier.
    /// Round-trip cost here is dominated by store-to-load forwarding
    /// in shared L1d rather than any cross-core coherence.
    smt: (CoreId, CoreId),
    /// Intra-CCX, different physical cores. Shared L3 / L2 cluster,
    /// no L1d sharing. The architecturally "intra-CCX" case the
    /// scheduler treats as the same `SyncCostTier::IntraCcx` bucket.
    intra_ccx: Option<(CoreId, CoreId)>,
    /// Cross-CCX pair when the host has at least two clusters with
    /// the same `ccx_size`; `None` on single-cluster hosts.
    cross_ccx: Option<(CoreId, CoreId)>,
}

/// Pick representative core ids for three coherence tiers: SMT
/// siblings (shared L1d), intra-CCX non-siblings (shared L3 within
/// the same Zen CCX or Sapphire Rapids module), and cross-CCX.
///
/// On AMD Zen the cluster size comes from CPUID 0x8000_001D (the
/// `cluster_size_log2` detector in `numa_topology`); on Intel
/// Sapphire Rapids+ it comes from CPUID 1Fh Module domain.
///
/// Assumes the common x86 logical-core enumeration on both Windows
/// and Linux: SMT siblings are adjacent (0,1), (2,3), ... and each
/// CCX/cluster holds `ccx_size` adjacent logical cores.
fn pick_pinned_pairs() -> Option<PinnedPairs> {
    let core_ids = core_affinity::get_core_ids()?;
    if core_ids.len() < 2 {
        return None;
    }
    let topo = numa_topology();
    let ccx_size = 1usize << topo.cluster_size_log2 as usize;

    // SMT-sibling pair: cores 0 and 1 under the common logical
    // enumeration on both Linux and Windows for SMT-enabled x86.
    // Same-physical-core, two-SMT-thread case.
    let smt = (core_ids[0], core_ids[1]);

    // Intra-CCX, different physical cores: stride past one SMT pair.
    // Cores 0 and 2 land on the second physical core of the same
    // CCX so long as `ccx_size >= 4`. Requires `>= 3` logical cores
    // total.
    let intra_ccx = if core_ids.len() >= 3 && ccx_size >= 4 {
        Some((core_ids[0], core_ids[2]))
    } else {
        None
    };

    // Cross-CCX pair: cores `0` and `ccx_size` land in different
    // clusters when at least two clusters exist.
    let cross_ccx = if core_ids.len() > ccx_size && ccx_size >= 2 {
        Some((core_ids[0], core_ids[ccx_size]))
    } else {
        None
    };
    Some(PinnedPairs {
        smt,
        intra_ccx,
        cross_ccx,
    })
}

fn bench_chase_lev(c: &mut Criterion) {
    let deque_path = temp_path("clmmf_deque", 0);
    let latches_path = temp_path("clmmf_latches", 0);

    let backend = Arc::new(
        SharedMemoryChaseLevBackend::create(0, &deque_path, &latches_path, 64, 128)
            .expect("create chase-lev backend"),
    );

    let adder_name = "flynnel.bench.clmmf.adder.cl";
    let adder_id = hash_name(adder_name);
    register(adder_id, |args| {
        // Wire shape (from encode_args): [I32_tag][i32 LE][I32_tag][i32 LE].
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        a.copy_from_slice(&args[1..5]);
        b.copy_from_slice(&args[6..10]);
        let sum = i32::from_le_bytes(a) + i32::from_le_bytes(b);
        Ok(sum.to_le_bytes().to_vec())
    });

    let stop = Arc::new(AtomicBool::new(false));
    let worker_backend = Arc::clone(&backend);
    let worker_stop = Arc::clone(&stop);
    let worker = std::thread::spawn(move || {
        while !worker_stop.load(Ordering::Relaxed) {
            match worker_backend.drain_one() {
                Ok(Some(())) => {}
                Ok(None) => std::hint::spin_loop(),
                Err(e) => {
                    eprintln!("clmmf drain err: {e}");
                    return;
                }
            }
        }
    });

    let args_blob = wire::encode_args(&[KernelArg::I32(3), KernelArg::I32(4)])
        .expect("encode_args");
    assert!(
        args_blob.len() <= ARGS_INLINE_BYTES,
        "args don't fit in chase-lev slot"
    );

    let mut group = c.benchmark_group("chase_lev_mmf");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("round_trip_add_i32", |b| {
        b.iter(|| {
            let handle = backend
                .dispatch_marshal(adder_id, &args_blob)
                .expect("dispatch marshal");
            loop {
                match backend.poll_handle(handle).expect("poll") {
                    Some(r) => {
                        // drop(black_box(...)) consumes the Result so
                        // the must_use lint is satisfied; black_box
                        // still prevents the optimizer from eliding
                        // the result read.
                        drop(black_box(r));
                        break;
                    }
                    None => std::hint::spin_loop(),
                }
            }
        });
    });
    group.finish();

    stop.store(true, Ordering::Relaxed);
    worker.join().expect("worker join");
    unregister(adder_id);
    std::fs::remove_file(&deque_path).ok();
    std::fs::remove_file(&latches_path).ok();
}

/// Pure-substrate bench: Chase-Lev deque push+pop / push+steal, plus
/// the latch arena alloc+publish+read cycle. No handler, no latch
/// integration - isolates the storage-layer cost from the end-to-end
/// dispatch path so regressions land where the change was made.
fn bench_substrate_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("substrate_only");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(3));

    let payload = [0u8; 16];

    let deque_path = temp_path("substrate_deque", 0);
    let deque = MmfChaseLevDeque::create(&deque_path, 64).expect("deque");
    let slot = RemoteJobSlot::new(0, u32::MAX, &payload).expect("slot");

    group.bench_function("chase_lev_push_pop_same_thread", |b| {
        b.iter(|| {
            // Single-thread round-trip: push then LIFO-pop on the
            // Chase-Lev deque. Measures the per-call store-on-bottom
            // + LIFO-pop cost on the owner-side hot path.
            deque.push(slot).expect("push");
            match deque.pop() {
                Steal::Success(_) => {}
                Steal::Empty => panic!("deque empty"),
                Steal::Retry => panic!("deque retry on same-thread pop"),
            }
        });
    });
    group.bench_function("chase_lev_push_steal_same_thread", |b| {
        b.iter(|| {
            // Single-thread round-trip via STEAL path (CAS on top).
            // This is the cost a cross-thread / cross-process thief
            // pays; the gap vs the LIFO-pop bench is the cost of the
            // owner-side asymmetric pop optimization.
            deque.push(slot).expect("push");
            loop {
                match deque.steal() {
                    Steal::Success(_) => break,
                    Steal::Empty => panic!("deque empty"),
                    Steal::Retry => continue,
                }
            }
        });
    });

    let latches_path = temp_path("substrate_latches", 0);
    let latches = MmfLatchArena::create(&latches_path, 256).expect("latches");
    let mut out = Vec::new();
    group.bench_function("latch_alloc_publish_read_same_thread", |b| {
        b.iter(|| {
            let off = latches.alloc();
            latches.publish(off, &[1u8, 2, 3, 4]).expect("publish");
            latches.read_result(off, &mut out).expect("read");
            latches.reset(off).expect("reset");
        });
    });
    group.finish();
    drop(deque);
    drop(latches);
    std::fs::remove_file(&deque_path).ok();
    std::fs::remove_file(&latches_path).ok();
}

/// Pinned variant: originator + drain threads bound to two specific
/// `CoreId`s. Isolates the cross-thread coherence cost from the OS
/// scheduler's placement jitter so the per-call latency number
/// reflects the protocol's actual cost at that coherence tier.
fn bench_chase_lev_pinned_at(c: &mut Criterion, label: &str, originator: CoreId, drain: CoreId) {
    let deque_path = temp_path(&format!("clmmf_pinned_deque_{label}"), 0);
    let latches_path = temp_path(&format!("clmmf_pinned_latches_{label}"), 0);

    let backend = Arc::new(
        SharedMemoryChaseLevBackend::create(0, &deque_path, &latches_path, 64, 128)
            .expect("create chase-lev backend"),
    );

    let adder_name = format!("flynnel.bench.clmmf.adder.cl_pinned_{label}");
    let adder_id = hash_name(&adder_name);
    register(adder_id, |args| {
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        a.copy_from_slice(&args[1..5]);
        b.copy_from_slice(&args[6..10]);
        let sum = i32::from_le_bytes(a) + i32::from_le_bytes(b);
        Ok(sum.to_le_bytes().to_vec())
    });

    let stop = Arc::new(AtomicBool::new(false));
    let worker_backend = Arc::clone(&backend);
    let worker_stop = Arc::clone(&stop);
    let worker = std::thread::spawn(move || {
        let ok = core_affinity::set_for_current(drain);
        if !ok {
            eprintln!("warning: failed to pin drain to {drain:?}");
        }
        while !worker_stop.load(Ordering::Relaxed) {
            match worker_backend.drain_one() {
                Ok(Some(())) => {}
                Ok(None) => std::hint::spin_loop(),
                Err(e) => {
                    eprintln!("clmmf drain err: {e}");
                    return;
                }
            }
        }
    });

    let ok = core_affinity::set_for_current(originator);
    if !ok {
        eprintln!("warning: failed to pin originator to {originator:?}");
    }

    let args_blob = wire::encode_args(&[KernelArg::I32(3), KernelArg::I32(4)])
        .expect("encode_args");
    assert!(
        args_blob.len() <= ARGS_INLINE_BYTES,
        "args don't fit in chase-lev slot"
    );

    let mut group = c.benchmark_group(format!("chase_lev_pinned_{label}"));
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("round_trip_add_i32", |b| {
        b.iter(|| {
            let handle = backend
                .dispatch_marshal(adder_id, &args_blob)
                .expect("dispatch marshal");
            loop {
                match backend.poll_handle(handle).expect("poll") {
                    Some(r) => {
                        drop(black_box(r));
                        break;
                    }
                    None => std::hint::spin_loop(),
                }
            }
        });
    });
    group.finish();

    stop.store(true, Ordering::Relaxed);
    worker.join().expect("worker join");
    unregister(adder_id);
    std::fs::remove_file(&deque_path).ok();
    std::fs::remove_file(&latches_path).ok();
}

/// Entry point for the pinned-pair bench. Picks one core pair per
/// coherence tier via [`pick_pinned_pairs`] then runs the round-trip
/// bench against each tier.
fn bench_pinned(c: &mut Criterion) {
    let pairs = match pick_pinned_pairs() {
        Some(p) => p,
        None => {
            eprintln!("pinned benches: core_affinity::get_core_ids() unavailable; skipping");
            return;
        }
    };
    let topo = numa_topology();
    eprintln!(
        "pinned benches: cluster_size_log2={} ({} logical cores per CCX) [{:?}]",
        topo.cluster_size_log2,
        1usize << topo.cluster_size_log2 as usize,
        topo.cluster_source,
    );
    eprintln!(
        "  smt_siblings=({:?},{:?}), intra_ccx={}, cross_ccx={}",
        pairs.smt.0,
        pairs.smt.1,
        match pairs.intra_ccx {
            Some((a, b)) => format!("({a:?},{b:?})"),
            None => "<insufficient cores or unknown CCX size>".to_string(),
        },
        match pairs.cross_ccx {
            Some((a, b)) => format!("({a:?},{b:?})"),
            None => "<single-CCX host>".to_string(),
        }
    );

    bench_chase_lev_pinned_at(c, "smt_siblings", pairs.smt.0, pairs.smt.1);
    if let Some((a, b)) = pairs.intra_ccx {
        bench_chase_lev_pinned_at(c, "intra_ccx", a, b);
    }
    if let Some((a, b)) = pairs.cross_ccx {
        bench_chase_lev_pinned_at(c, "cross_ccx", a, b);
    }
}

criterion_group!(benches, bench_chase_lev, bench_substrate_only, bench_pinned);
criterion_main!(benches);

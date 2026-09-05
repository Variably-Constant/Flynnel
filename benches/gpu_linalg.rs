//! gpu_peer linalg: device kernels vs the Flynnel-parallel CPU
//! reference vs serial CPU, per op, matrix size and batch size, plus
//! the block-per-matrix vs thread-per-matrix Jacobi shape comparison
//! that sets `JACOBI_THREAD_SHAPE_BATCH_PER_N`.
//!
//! Every contender computes the same result from the same inputs.
//! GPU times are kernel wall (launch + sync) with the data already
//! resident, which is the steady state of a scoring loop that pins
//! once per step; the one-time pin + fetch cost is printed separately.
//! Run with:
//!   cargo bench --features gpu-peer --bench gpu_linalg
//! `FLYNNEL_BENCH_SECTIONS=syev,gesvd` (any of gemm, einsum, syev,
//! gesvd, bisect, tandem, lu, ozaki) limits the run to those sections.

use std::time::{Duration, Instant};

use flynnel::gpu_peer::linalg::{
    EinsumSpec, JacobiShape, LinalgKernels, cpu, default_sweeps, gemm_batched,
    gemm_tandem_batched, gesvd_auto_batched, gesvd_bisect_scratch_bytes, gesvd_tandem_batched,
    getrf_batched, getri_batched, getrs_batched, launch_einsum, launch_gemm, launch_gesvd,
    launch_gesvd_bisect, launch_getrf, launch_getrs, launch_syev, launch_syev_bisect,
    syev_auto_batched, syev_tandem_batched,
};
use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, ResidentHandle};
use flynnel::sched::JobPlan;
use flynnel::sched::par_iter::collect_indexed;

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn uniform(seed: u64, len: usize) -> Vec<f64> {
    let mut s = seed;
    (0..len).map(|_| (splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0).collect()
}

fn symmetric_batch(seed: u64, batch: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0f64; batch * n * n];
    for bi in 0..batch {
        let r = uniform(seed + bi as u64, n * n);
        for i in 0..n {
            for j in 0..n {
                out[bi * n * n + i * n + j] = 0.5 * (r[i * n + j] + r[j * n + i]);
            }
        }
    }
    out
}

fn bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Median of `runs` timings of one `launch` followed by a stream
/// sync, after a 300 ms ramp of back-to-back launches. The ramp is
/// bursts of 32 enqueues per sync: a launch-and-sync loop leaves the
/// GPU idle between calls and an RTX 3070 then never leaves idle
/// clocks (12.8 ms for an outer product that takes 0.5 ms at boost,
/// even after 150 ms of such calls); saturating the queue does.
fn gpu_median_ns<F: FnMut(&mut GpuPeer)>(peer: &mut GpuPeer, runs: usize, mut launch: F) -> f64 {
    let ramp = Instant::now();
    while ramp.elapsed() < Duration::from_millis(300) {
        for _ in 0..32 {
            launch(peer);
        }
        peer.sync_wide().expect("ramp sync");
    }
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let t0 = Instant::now();
            launch(peer);
            peer.sync_wide().expect("sync");
            t0.elapsed().as_nanos() as f64
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    t[t.len() / 2]
}

/// Median of `runs` timings of `f` after warming up: `f` runs until
/// at least 150 ms have elapsed (at least once, at most 4000 calls).
/// CPU contenders only; GPU timings go through [`gpu_median_ns`].
fn median_ns<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    let warm = Instant::now();
    let mut calls = 0;
    while calls == 0 || (calls < 4000 && warm.elapsed() < Duration::from_millis(150)) {
        f();
        calls += 1;
    }
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_nanos() as f64
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    t[t.len() / 2]
}

fn ms(ns: f64) -> f64 {
    ns / 1e6
}

/// Warm calls before a tandem cell is timed: the split model moves
/// one eighth of the way to each measured ratio per call, so a dozen
/// calls bring the share within a few percent of the balance the
/// cell measures.
const TANDEM_WARM_CALLS: usize = 12;

/// One tandem cell: `TANDEM_WARM_CALLS` warm calls, then the median
/// of three timed calls, then one more call whose report is
/// returned. Every call goes through the one closure, so the helper's
/// `#[track_caller]` site is the same for all of them and the timed
/// calls run at the share the warm calls learned; each warm call is
/// traced with its share and per-side cost.
fn tandem_cell(label: &str, call: &mut dyn FnMut() -> flynnel::SplitReport) -> (f64, u32, flynnel::SplitReport) {
    let mut share = 0u32;
    for _ in 0..TANDEM_WARM_CALLS {
        let r = call();
        trace(&format!(
            "{label} warm: share {} cpu {} items {:.3} ms/item dev {} items {:.3} ms/item",
            r.cpu_share_per_mille, r.cpu_items, ms(r.cpu_ns as f64) / r.cpu_items.max(1) as f64,
            r.backend_items, ms(r.backend_ns as f64) / r.backend_items.max(1) as f64
        ));
        share = r.cpu_share_per_mille;
    }
    let mut t: Vec<f64> = (0..3)
        .map(|_| {
            let t0 = Instant::now();
            std::hint::black_box(call());
            t0.elapsed().as_nanos() as f64
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let last = call();
    (t[1], share, last)
}

/// Matrices per CPU-parallel work item: enough work per item that
/// the adaptive plan dispatches it rather than running the probe's
/// light-item verdict inline (a 64-matrix chunk of n = 8 GEMMs is
/// ~40 us).
const CHUNK: usize = 64;

/// Phase trace to stderr when `FLYNNEL_BENCH_TRACE` is set, so a
/// stalled cell names the phase it stalled in.
fn trace(phase: &str) {
    if std::env::var_os("FLYNNEL_BENCH_TRACE").is_some() {
        eprintln!("[trace] {phase}");
    }
}

struct Dev {
    handle: ResidentHandle,
    ptr: u64,
}

fn pin(peer: &mut GpuPeer, data: &[u8]) -> Dev {
    let handle = peer.pin_bulk(data).expect("pin");
    let (ptr, _) = peer.resident_ptr(&handle).expect("ptr");
    Dev { handle, ptr }
}

/// Resident pool capacity in bytes for the config below.
const POOL_BYTES: usize = 96 * 16 * 1024 * 1024;

/// True when a cell's buffers fit the pool; prints a skip line
/// otherwise so the table never silently omits a cell.
fn fits(label: &str, bytes_needed: usize) -> bool {
    if bytes_needed <= POOL_BYTES {
        true
    } else {
        println!(
            "{label}: skipped, needs {:.1} GiB resident vs a {:.1} GiB pool",
            bytes_needed as f64 / (1u64 << 30) as f64,
            POOL_BYTES as f64 / (1u64 << 30) as f64
        );
        false
    }
}

use flynnel::gpu_peer::ozaki::{launch_ozaki_gemm, OzakiKernels, OzakiWorkspace};

/// Whether a section runs: `FLYNNEL_BENCH_SECTIONS` names a comma
/// list of gemm, einsum, syev, gesvd, bisect, tandem, lu, ozaki; unset
/// runs all eight.
fn wants(section: &str) -> bool {
    match std::env::var("FLYNNEL_BENCH_SECTIONS") {
        Ok(list) => list.split(',').any(|s| s.trim() == section),
        Err(_) => true,
    }
}

fn main() {
    let mut peer = match GpuPeer::init(GpuPeerConfig {
        slot_bytes: 64 * 1024,
        slots_per_lane: 4,
        vram_block_bytes: 16 * 1024 * 1024,
        vram_blocks: 96,
        ..GpuPeerConfig::default()
    }) {
        Ok(p) => p,
        Err(e) => {
            println!("no CUDA device: {e}; nothing measured");
            std::process::exit(1);
        }
    };
    let k = LinalgKernels::load(&peer).expect("linalg PTX");
    let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1);
    println!("=== gpu_peer linalg: GPU vs Flynnel-parallel CPU ({cores} threads) vs serial ===");
    println!("GPU column = kernel wall with resident data; pin+fetch printed separately.\n");
    let runs = 5usize;

    // ---------------------------------------------------------- gemm
    println!("--- batched GEMM (m = n = k), f64 ---");
    println!("{:>4} {:>7} | {:>10} {:>10} {:>10} | {:>9} {:>9} | {:>10}",
        "n", "batch", "gpu ms", "cpu-par ms", "serial ms", "gpu/par", "gpu/ser", "pin+fetch");
    for &n in &[8usize, 16, 32, 64] {
        if !wants("gemm") {
            break;
        }
        for &batch in &[1024usize, 8192, 65536] {
            if !fits(&format!("gemm n={n} batch={batch}"), 3 * batch * n * n * 8) {
                continue;
            }
            trace(&format!("gemm n={n} batch={batch}: generate"));
            let a = uniform(1, batch * n * n);
            let b = uniform(2, batch * n * n);
            let t_pin = Instant::now();
            trace("pin");
            let pa = pin(&mut peer, &bytes(&a));
            let pb = pin(&mut peer, &bytes(&b));
            let pc = pin(&mut peer, &vec![0u8; batch * n * n * 8]);
            let pin_ns = t_pin.elapsed().as_nanos() as f64;
            trace("gpu launches");
            let gpu = gpu_median_ns(&mut peer, runs, |p| {
                launch_gemm(p, &k, pa.ptr, pb.ptr, pc.ptr, batch as u32, n as u32, n as u32, n as u32)
                    .expect("launch");
            });
            trace("fetch");
            let mut out = vec![0u8; batch * n * n * 8];
            let t_fetch = Instant::now();
            peer.fetch_bulk(&pc.handle, &mut out).expect("fetch");
            let pin_fetch = pin_ns + t_fetch.elapsed().as_nanos() as f64;
            trace("serial");
            let serial = median_ns(1, || {
                std::hint::black_box(cpu::gemm_batched(&a, &b, batch, n, n, n));
            });
            trace("cpu-parallel");
            // The consumer shape: CHUNK matrices per item through the
            // adaptive default plan, so the CPU side has enough work
            // per item to be dispatched instead of probed inline.
            let par = median_ns(3, || {
                let items = batch.div_ceil(CHUNK);
                let plan = JobPlan::new(0, items as u32);
                let c: Vec<Vec<f64>> = collect_indexed(&plan, items, 1, |ci| {
                    let lo = ci * CHUNK;
                    let hi = (lo + CHUNK).min(batch);
                    cpu::gemm_batched(
                        &a[lo * n * n..hi * n * n],
                        &b[lo * n * n..hi * n * n],
                        hi - lo, n, n, n,
                    )
                });
                std::hint::black_box(c);
            });
            println!("{n:>4} {batch:>7} | {:>10.3} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x | {:>8.2} ms",
                ms(gpu), ms(par), ms(serial), par / gpu, serial / gpu, ms(pin_fetch));
            peer.unpin(pc.handle).expect("unpin");
            peer.unpin(pb.handle).expect("unpin");
            peer.unpin(pa.handle).expect("unpin");
        }
    }

    // ---------------------------------------------------------- einsum (outer + axissum)
    println!("\n--- einsum outer product \"i,j->ij\" (n x n) and row sum \"ij->i\", f64 ---");
    println!("{:>8} {:>4} {:>7} | {:>10} {:>10} | {:>9}", "op", "n", "batch", "gpu ms", "serial ms", "gpu/ser");
    for &n in &[16usize, 64] {
        if !wants("einsum") {
            break;
        }
        for &batch in &[8192usize, 65536] {
            if !fits(
                &format!("einsum n={n} batch={batch}"),
                batch * (n * n + 2 * n) * 8 + batch * (n * n + n) * 8,
            ) {
                continue;
            }
            let x = uniform(3, batch * n);
            let y = uniform(4, batch * n);
            let spec = EinsumSpec::parse("i,j->ij", &[n], Some(&[n])).expect("spec");
            let tables = pin(&mut peer, &spec.tables().iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            let px = pin(&mut peer, &bytes(&x));
            let py = pin(&mut peer, &bytes(&y));
            let po = pin(&mut peer, &vec![0u8; batch * n * n * 8]);
            let gpu = gpu_median_ns(&mut peer, runs, |p| {
                launch_einsum(p, &k, &spec, tables.ptr, px.ptr, py.ptr, po.ptr, batch as u32).expect("launch");
            });
            let serial = median_ns(1, || {
                std::hint::black_box(cpu::einsum(&spec, &x, Some(&y), batch));
            });
            println!("{:>8} {n:>4} {batch:>7} | {:>10.3} {:>10.3} | {:>8.2}x", "outer", ms(gpu), ms(serial), serial / gpu);
            let a = uniform(5, batch * n * n);
            let rspec = EinsumSpec::parse("ij->i", &[n, n], None).expect("spec");
            let rtables = pin(&mut peer, &rspec.tables().iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            let pa = pin(&mut peer, &bytes(&a));
            let pr = pin(&mut peer, &vec![0u8; batch * n * 8]);
            let gpu = gpu_median_ns(&mut peer, runs, |p| {
                launch_einsum(p, &k, &rspec, rtables.ptr, pa.ptr, pa.ptr, pr.ptr, batch as u32).expect("launch");
            });
            let serial = median_ns(1, || {
                std::hint::black_box(cpu::einsum(&rspec, &a, None, batch));
            });
            println!("{:>8} {n:>4} {batch:>7} | {:>10.3} {:>10.3} | {:>8.2}x", "rowsum", ms(gpu), ms(serial), serial / gpu);
            for d in [pr, pa, rtables, po, py, px, tables] {
                peer.unpin(d.handle).expect("unpin");
            }
        }
    }

    // ---------------------------------------------------------- jacobi eig
    println!("\n--- batched symmetric eigenvalues (Jacobi), f64; blk = block/matrix, thr = thread/matrix ---");
    println!("{:>4} {:>7} | {:>10} {:>10} {:>10} {:>10} | {:>9} {:>9}",
        "n", "batch", "blk ms", "thr ms", "cpu-par ms", "serial ms", "best/par", "best/ser");
    for &n in &[4usize, 8, 16, 32, 64] {
        if !wants("syev") {
            break;
        }
        for &batch in &[1024usize, 2048, 4096, 8192, 65536] {
            if !fits(&format!("syev n={n} batch={batch}"), batch * n * n * 8 + batch * n * 8) {
                continue;
            }
            let a = symmetric_batch(10, batch, n);
            let sweeps = default_sweeps(n);
            let pa = pin(&mut peer, &bytes(&a));
            let pw = pin(&mut peer, &vec![0u8; batch * n * 8]);
            let blk = gpu_median_ns(&mut peer, runs, |p| {
                launch_syev(p, &k, pa.ptr, pw.ptr, None, batch as u32, n as u32, sweeps, JacobiShape::BlockPerMatrix).expect("launch");
            });
            let thr = if n <= 16 {
                Some(gpu_median_ns(&mut peer, runs, |p| {
                    launch_syev(p, &k, pa.ptr, pw.ptr, None, batch as u32, n as u32, sweeps, JacobiShape::ThreadPerMatrix).expect("launch");
                }))
            } else {
                None
            };
            let serial = median_ns(1, || {
                std::hint::black_box(cpu::syev_jacobi_batched(&a, batch, n, sweeps, false));
            });
            let par = median_ns(3, || {
                let items = batch.div_ceil(CHUNK);
                let plan = JobPlan::new(0, items as u32);
                let w: Vec<Vec<f64>> = collect_indexed(&plan, items, 1, |ci| {
                    let lo = ci * CHUNK;
                    let hi = (lo + CHUNK).min(batch);
                    cpu::syev_jacobi_batched(&a[lo * n * n..hi * n * n], hi - lo, n, sweeps, false).0
                });
                std::hint::black_box(w);
            });
            let best = thr.map_or(blk, |t| t.min(blk));
            println!("{n:>4} {batch:>7} | {:>10.3} {:>10} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x",
                ms(blk),
                thr.map_or("-".to_string(), |t| format!("{:.3}", ms(t))),
                ms(par), ms(serial), par / best, serial / best);
            peer.unpin(pw.handle).expect("unpin");
            peer.unpin(pa.handle).expect("unpin");
        }
    }

    // ---------------------------------------------------------- jacobi svd
    println!("\n--- batched singular values (one-sided Jacobi), square m = n, f64 ---");
    println!("{:>4} {:>7} | {:>10} {:>10} {:>10} {:>10} | {:>9} {:>9}",
        "n", "batch", "blk ms", "thr ms", "cpu-par ms", "serial ms", "best/par", "best/ser");
    for &n in &[4usize, 8, 16, 32, 64] {
        if !wants("gesvd") {
            break;
        }
        for &batch in &[1024usize, 2048, 4096, 8192, 65536] {
            if !fits(&format!("gesvd n={n} batch={batch}"), batch * n * n * 8 + batch * n * 8) {
                continue;
            }
            let a = uniform(20, batch * n * n);
            let sweeps = default_sweeps(n);
            let pa = pin(&mut peer, &bytes(&a));
            let ps = pin(&mut peer, &vec![0u8; batch * n * 8]);
            // The kernel overwrites A with U; re-upload before each run.
            let a_bytes = bytes(&a);
            let blk = gpu_median_ns(&mut peer, runs, |p| {
                p.write_resident_bulk(&pa.handle, &a_bytes).expect("reload");
                launch_gesvd(p, &k, pa.ptr, ps.ptr, None, batch as u32, n as u32, n as u32, sweeps, JacobiShape::BlockPerMatrix).expect("launch");
            });
            let thr = if n <= 16 {
                Some(gpu_median_ns(&mut peer, runs, |p| {
                    p.write_resident_bulk(&pa.handle, &a_bytes).expect("reload");
                    launch_gesvd(p, &k, pa.ptr, ps.ptr, None, batch as u32, n as u32, n as u32, sweeps, JacobiShape::ThreadPerMatrix).expect("launch");
                }))
            } else {
                None
            };
            let serial = median_ns(1, || {
                std::hint::black_box(cpu::gesvd_jacobi_batched(&a, batch, n, n, sweeps, false));
            });
            let par = median_ns(3, || {
                let items = batch.div_ceil(CHUNK);
                let plan = JobPlan::new(0, items as u32);
                let s: Vec<Vec<f64>> = collect_indexed(&plan, items, 1, |ci| {
                    let lo = ci * CHUNK;
                    let hi = (lo + CHUNK).min(batch);
                    cpu::gesvd_jacobi_batched(&a[lo * n * n..hi * n * n], hi - lo, n, n, sweeps, false).1
                });
                std::hint::black_box(s);
            });
            let best = thr.map_or(blk, |t| t.min(blk));
            println!("{n:>4} {batch:>7} | {:>10.3} {:>10} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x",
                ms(blk),
                thr.map_or("-".to_string(), |t| format!("{:.3}", ms(t))),
                ms(par), ms(serial), par / best, serial / best);
            peer.unpin(ps.handle).expect("unpin");
            peer.unpin(pa.handle).expect("unpin");
        }
    }
    // ---------------------------------------------------------- bisect vs jacobi
    println!("\n--- eigenvalues and singular values, f64: Jacobi (blk) vs tridiagonalization / bidiagonalization + bisection ---");
    println!("{:>5} {:>4} {:>6} | {:>10} {:>10} {:>10} {:>10} | {:>9} {:>9}",
        "op", "n", "batch", "jacobi ms", "bisect ms", "cpu-par ms", "serial ms", "jac/bis", "par/bis");
    for &n in &[32usize, 64] {
        if !wants("bisect") {
            break;
        }
        for &batch in &[1024usize, 8192] {
            if !fits(&format!("bisect n={n} batch={batch}"), 6 * batch * n * n * 8) {
                continue;
            }
            let sweeps = default_sweeps(n);
            // Symmetric operands for the eigenvalue rows.
            let mut a = uniform(21, batch * n * n);
            for item in 0..batch {
                for i in 0..n {
                    for j in 0..i {
                        let v = 0.5 * (a[item * n * n + i * n + j] + a[item * n * n + j * n + i]);
                        a[item * n * n + i * n + j] = v;
                        a[item * n * n + j * n + i] = v;
                    }
                }
            }
            let pa = pin(&mut peer, &bytes(&a));
            let pw = pin(&mut peer, &vec![0u8; batch * n * 8]);
            let jac = gpu_median_ns(&mut peer, runs, |p| {
                launch_syev(p, &k, pa.ptr, pw.ptr, None, batch as u32, n as u32, sweeps, JacobiShape::BlockPerMatrix).expect("launch");
            });
            let qr = gpu_median_ns(&mut peer, runs, |p| {
                launch_syev_bisect(p, &k, pa.ptr, pw.ptr, None, 0, batch as u32, n as u32).expect("launch");
            });
            let serial = median_ns(1, || {
                std::hint::black_box(cpu::syev_jacobi_batched(&a, batch, n, sweeps, false));
            });
            let par = median_ns(3, || {
                let items = batch.div_ceil(CHUNK);
                let plan = JobPlan::new(0, items as u32);
                let w: Vec<Vec<f64>> = collect_indexed(&plan, items, 1, |ci| {
                    let lo = ci * CHUNK;
                    let hi = (lo + CHUNK).min(batch);
                    cpu::syev_jacobi_batched(&a[lo * n * n..hi * n * n], hi - lo, n, sweeps, false).0
                });
                std::hint::black_box(w);
            });
            println!("{:>5} {n:>4} {batch:>6} | {:>10.3} {:>10.3} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x",
                "syev", ms(jac), ms(qr), ms(par), ms(serial), jac / qr, par / qr);
            peer.unpin(pw.handle).expect("unpin");
            peer.unpin(pa.handle).expect("unpin");
            // Square SVD rows; A is overwritten by both kernels, so it is
            // reloaded before every launch as the Jacobi section does.
            let a = uniform(22, batch * n * n);
            let a_bytes = bytes(&a);
            let pa = pin(&mut peer, &a_bytes);
            let ps = pin(&mut peer, &vec![0u8; batch * n * 8]);
            let scratch = pin(&mut peer, &vec![0u8; gesvd_bisect_scratch_bytes(batch as u32, n as u32, n as u32)]);
            let jac = gpu_median_ns(&mut peer, runs, |p| {
                p.write_resident_bulk(&pa.handle, &a_bytes).expect("reload");
                launch_gesvd(p, &k, pa.ptr, ps.ptr, None, batch as u32, n as u32, n as u32, sweeps, JacobiShape::BlockPerMatrix).expect("launch");
            });
            let qr = gpu_median_ns(&mut peer, runs, |p| {
                p.write_resident_bulk(&pa.handle, &a_bytes).expect("reload");
                launch_gesvd_bisect(p, &k, pa.ptr, ps.ptr, None, scratch.ptr, batch as u32, n as u32, n as u32).expect("launch");
            });
            let serial = median_ns(1, || {
                std::hint::black_box(cpu::gesvd_jacobi_batched(&a, batch, n, n, sweeps, false));
            });
            let par = median_ns(3, || {
                let items = batch.div_ceil(CHUNK);
                let plan = JobPlan::new(0, items as u32);
                let s: Vec<Vec<f64>> = collect_indexed(&plan, items, 1, |ci| {
                    let lo = ci * CHUNK;
                    let hi = (lo + CHUNK).min(batch);
                    cpu::gesvd_jacobi_batched(&a[lo * n * n..hi * n * n], hi - lo, n, n, sweeps, false).1
                });
                std::hint::black_box(s);
            });
            println!("{:>5} {n:>4} {batch:>6} | {:>10.3} {:>10.3} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x",
                "gesvd", ms(jac), ms(qr), ms(par), ms(serial), jac / qr, par / qr);
            peer.unpin(scratch.handle).expect("unpin");
            peer.unpin(ps.handle).expect("unpin");
            peer.unpin(pa.handle).expect("unpin");
        }
    }

    // ---------------------------------------------------------- tandem (device + CPU pool)
    // Host-buffer helpers end to end, so every column carries its
    // pin and fetch: the device alone, the CPU pool alone, and the
    // two in tandem after three warm calls have taught the site its
    // share.
    println!("\n--- tandem over host buffers: device alone vs CPU pool alone vs both, f64 ---");
    println!("{:>6} {:>4} {:>6} | {:>10} {:>10} {:>10} | {:>9} {:>9} | {:>5} {:>9} {:>9}",
        "op", "n", "batch", "gpu ms", "cpu-par ms", "tandem ms", "gpu/tan", "par/tan", "share", "cpu-side", "dev-side");
    for &n in &[16usize, 32, 64] {
        if !wants("tandem") {
            break;
        }
        for &batch in &[1024usize, 8192] {
            // The host-buffer helpers pin inputs, outputs and the
            // bisection scratch per call; a cell whose pins exceed
            // half the pool is skipped, since repeated pin and unpin
            // cycles exhaust the pool before its nominal capacity.
            let need_gesvd = 3 * batch * n * n * 8 + gesvd_bisect_scratch_bytes(batch as u32, n as u32, n as u32);
            if !fits(&format!("tandem n={n} batch={batch}"), 2 * need_gesvd) {
                continue;
            }
            let a = uniform(21, batch * n * n);
            let b = uniform(22, batch * n * n);
            let sym = {
                let mut s = uniform(23, batch * n * n);
                for item in 0..batch {
                    for i in 0..n {
                        for j in 0..i {
                            let v = 0.5 * (s[item * n * n + i * n + j] + s[item * n * n + j * n + i]);
                            s[item * n * n + i * n + j] = v;
                            s[item * n * n + j * n + i] = v;
                        }
                    }
                }
                s
            };
            let (bu, nu) = (batch as u32, n as u32);
            let cpu_par = |f: &(dyn Fn(usize, usize) + Sync)| {
                let items = batch.div_ceil(CHUNK);
                let plan = JobPlan::new(0, items as u32);
                let c: Vec<()> = collect_indexed(&plan, items, 1, |ci| {
                    let lo = ci * CHUNK;
                    f(lo, (lo + CHUNK).min(batch));
                });
                std::hint::black_box(c);
            };
            // gemm
            trace(&format!("tandem gemm n={n} batch={batch}"));
            let gpu = median_ns(3, || {
                std::hint::black_box(gemm_batched(&mut peer, &k, &a, &b, bu, nu, nu, nu).expect("gemm"));
            });
            let par = median_ns(3, || {
                cpu_par(&|lo, hi| {
                    std::hint::black_box(cpu::gemm_batched(&a[lo * n * n..hi * n * n], &b[lo * n * n..hi * n * n], hi - lo, n, n, n));
                });
            });
            let plan = JobPlan::new(0, bu);
            let (tan, share, last) = tandem_cell(&format!("gemm n={n} batch={batch}"), &mut || {
                gemm_tandem_batched(&mut peer, &k, &plan, &a, &b, bu, nu, nu, nu).expect("tandem").1
            });
            println!("{:>6} {n:>4} {batch:>6} | {:>10.3} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x | {share:>5} {:>9.3} {:>9.3}",
                "gemm", ms(gpu), ms(par), ms(tan), gpu / tan, par / tan, ms(last.cpu_ns as f64), ms(last.backend_ns as f64));
            // syev with vectors
            trace(&format!("tandem syev n={n} batch={batch}"));
            let sweeps = default_sweeps(n);
            let gpu = median_ns(3, || {
                std::hint::black_box(syev_auto_batched(&mut peer, &k, &sym, bu, nu, true).expect("syev"));
            });
            let par = median_ns(3, || {
                cpu_par(&|lo, hi| {
                    std::hint::black_box(cpu::syev_jacobi_batched(&sym[lo * n * n..hi * n * n], hi - lo, n, sweeps, true));
                });
            });
            let plan = JobPlan::new(0, bu);
            let (tan, share, last) = tandem_cell(&format!("syev n={n} batch={batch}"), &mut || {
                syev_tandem_batched(&mut peer, &k, &plan, &sym, bu, nu, true).expect("tandem").1
            });
            println!("{:>6} {n:>4} {batch:>6} | {:>10.3} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x | {share:>5} {:>9.3} {:>9.3}",
                "syev", ms(gpu), ms(par), ms(tan), gpu / tan, par / tan, ms(last.cpu_ns as f64), ms(last.backend_ns as f64));
            // gesvd with V
            trace(&format!("tandem gesvd n={n} batch={batch}"));
            let gpu = median_ns(3, || {
                std::hint::black_box(gesvd_auto_batched(&mut peer, &k, &a, bu, nu, nu, true).expect("gesvd"));
            });
            let par = median_ns(3, || {
                cpu_par(&|lo, hi| {
                    std::hint::black_box(cpu::gesvd_jacobi_batched(&a[lo * n * n..hi * n * n], hi - lo, n, n, sweeps, true));
                });
            });
            let plan = JobPlan::new(0, bu);
            let (tan, share, last) = tandem_cell(&format!("gesvd n={n} batch={batch}"), &mut || {
                gesvd_tandem_batched(&mut peer, &k, &plan, &a, bu, nu, nu, true).expect("tandem").1
            });
            println!("{:>6} {n:>4} {batch:>6} | {:>10.3} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x | {share:>5} {:>9.3} {:>9.3}",
                "gesvd", ms(gpu), ms(par), ms(tan), gpu / tan, par / tan, ms(last.cpu_ns as f64), ms(last.backend_ns as f64));
        }
    }

    // ---------------------------------------------------------- LU
    println!("\n--- batched LU with partial pivoting, f64: factor, solve (nrhs = 1), inverse ---");
    println!("{:>7} {:>4} {:>6} | {:>10} {:>10} {:>10} | {:>9} {:>9} | {:>10}",
        "op", "n", "batch", "gpu ms", "cpu-par ms", "serial ms", "gpu/par", "gpu/ser", "pin+fetch");
    for &n in &[8usize, 16, 32, 64] {
        if !wants("lu") {
            break;
        }
        for &batch in &[1024usize, 8192] {
            let a = uniform(31, batch * n * n);
            let rhs = uniform(32, batch * n);
            let (bu, nu) = (batch as u32, n as u32);
            let cpu_par = |f: &(dyn Fn(usize, usize) + Sync)| {
                let items = batch.div_ceil(CHUNK);
                let plan = JobPlan::new(0, items as u32);
                let c: Vec<()> = collect_indexed(&plan, items, 1, |ci| {
                    let lo = ci * CHUNK;
                    f(lo, (lo + CHUNK).min(batch));
                });
                std::hint::black_box(c);
            };
            // factor: the resident copy is refactored in place each
            // launch; the work per launch does not depend on the data.
            trace(&format!("lu getrf n={n} batch={batch}"));
            let t_pin = Instant::now();
            let pa = pin(&mut peer, &bytes(&a));
            let ppiv = pin(&mut peer, &vec![0u8; batch * n * 4]);
            let pinfo = pin(&mut peer, &vec![0u8; batch * 4]);
            let pin_ns = t_pin.elapsed().as_nanos() as f64;
            let gpu = gpu_median_ns(&mut peer, runs, |p| {
                launch_getrf(p, &k, pa.ptr, ppiv.ptr, pinfo.ptr, bu, nu).expect("launch");
            });
            let t_fetch = Instant::now();
            let f = getrf_batched(&mut peer, &k, &a, bu, nu).expect("getrf");
            let pin_fetch = pin_ns + t_fetch.elapsed().as_nanos() as f64;
            let serial = median_ns(1, || {
                std::hint::black_box(cpu::getrf_batched(&a, batch, n));
            });
            let par = median_ns(3, || {
                cpu_par(&|lo, hi| {
                    std::hint::black_box(cpu::getrf_batched(&a[lo * n * n..hi * n * n], hi - lo, n));
                });
            });
            println!("{:>7} {n:>4} {batch:>6} | {:>10.3} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x | {:>8.2} ms",
                "getrf", ms(gpu), ms(par), ms(serial), par / gpu, serial / gpu, ms(pin_fetch));
            peer.unpin(pinfo.handle).expect("unpin");
            peer.unpin(ppiv.handle).expect("unpin");
            peer.unpin(pa.handle).expect("unpin");
            // solve, one right-hand side
            trace(&format!("lu getrs n={n} batch={batch}"));
            let plu = pin(&mut peer, &bytes(&f.lu));
            let ppiv = pin(&mut peer, &f.piv.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
            let pb = pin(&mut peer, &bytes(&rhs));
            let gpu = gpu_median_ns(&mut peer, runs, |p| {
                launch_getrs(p, &k, plu.ptr, ppiv.ptr, pb.ptr, bu, nu, 1, false).expect("launch");
            });
            let serial = median_ns(1, || {
                std::hint::black_box(cpu::getrs_batched(&f.lu, &f.piv, &rhs, batch, n, 1));
            });
            let par = median_ns(3, || {
                cpu_par(&|lo, hi| {
                    std::hint::black_box(cpu::getrs_batched(&f.lu[lo * n * n..hi * n * n], &f.piv[lo * n..hi * n], &rhs[lo * n..hi * n], hi - lo, n, 1));
                });
            });
            let t_all = Instant::now();
            std::hint::black_box(getrs_batched(&mut peer, &k, &f.lu, &f.piv, &rhs, bu, nu, 1).expect("getrs"));
            let pin_fetch = t_all.elapsed().as_nanos() as f64 - gpu;
            println!("{:>7} {n:>4} {batch:>6} | {:>10.3} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x | {:>8.2} ms",
                "getrs", ms(gpu), ms(par), ms(serial), par / gpu, serial / gpu, ms(pin_fetch.max(0.0)));
            peer.unpin(pb.handle).expect("unpin");
            // inverse
            trace(&format!("lu getri n={n} batch={batch}"));
            let pinv = pin(&mut peer, &vec![0u8; batch * n * n * 8]);
            let gpu = gpu_median_ns(&mut peer, runs, |p| {
                launch_getrs(p, &k, plu.ptr, ppiv.ptr, pinv.ptr, bu, nu, nu, true).expect("launch");
            });
            let serial = median_ns(1, || {
                std::hint::black_box(cpu::getri_batched(&f.lu, &f.piv, batch, n));
            });
            let par = median_ns(3, || {
                cpu_par(&|lo, hi| {
                    std::hint::black_box(cpu::getri_batched(&f.lu[lo * n * n..hi * n * n], &f.piv[lo * n..hi * n], hi - lo, n));
                });
            });
            let t_all = Instant::now();
            std::hint::black_box(getri_batched(&mut peer, &k, &f.lu, &f.piv, bu, nu).expect("getri"));
            let pin_fetch = t_all.elapsed().as_nanos() as f64 - gpu;
            println!("{:>7} {n:>4} {batch:>6} | {:>10.3} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x | {:>8.2} ms",
                "getri", ms(gpu), ms(par), ms(serial), par / gpu, serial / gpu, ms(pin_fetch.max(0.0)));
            peer.unpin(pinv.handle).expect("unpin");
            peer.unpin(ppiv.handle).expect("unpin");
            peer.unpin(plu.handle).expect("unpin");
        }
    }

    // ---------------------------------------------------------- ozaki gemm
    println!("\n--- GEMM, f64: native kernel vs Ozaki scheme on the int8 tensor cores ---");
    println!("{:>5} {:>6} | {:>10} {:>10} {:>10} | {:>9} {:>9}",
        "n", "batch", "f64 ms", "ozaki ms", "serial ms", "f64/oz", "ser/oz");
    let oz = OzakiKernels::load(&peer).expect("ozaki PTX");
    for &(n, batch) in &[(64usize, 1024usize), (64, 8192), (256, 1), (512, 1), (1024, 1), (2048, 1)] {
        if !wants("ozaki") {
            break;
        }
        let need = 3 * batch * n * n * 8
            + OzakiWorkspace::bytes(batch as u32, n as u32, n as u32, n as u32);
        if !fits(&format!("ozaki n={n} batch={batch}"), need) {
            continue;
        }
        let a = uniform(11, batch * n * n);
        let b = uniform(12, batch * n * n);
        let pa = pin(&mut peer, &bytes(&a));
        let pb = pin(&mut peer, &bytes(&b));
        let pc = pin(&mut peer, &vec![0u8; batch * n * n * 8]);
        let ws = OzakiWorkspace::new(&mut peer, batch as u32, n as u32, n as u32, n as u32)
            .expect("ozaki workspace");
        let native = gpu_median_ns(&mut peer, runs, |p| {
            launch_gemm(p, &k, pa.ptr, pb.ptr, pc.ptr, batch as u32, n as u32, n as u32, n as u32)
                .expect("launch");
        });
        let ozaki = gpu_median_ns(&mut peer, runs, |p| {
            launch_ozaki_gemm(p, &oz, &ws, pa.ptr, pb.ptr, pc.ptr).expect("launch");
        });
        let serial = if n <= 512 {
            Some(median_ns(1, || {
                std::hint::black_box(cpu::gemm_batched(&a, &b, batch, n, n, n));
            }))
        } else {
            None
        };
        let (serial_ms, ser_ratio) = match serial {
            Some(s) => (format!("{:.3}", ms(s)), format!("{:.2}x", s / ozaki)),
            None => ("-".to_string(), "-".to_string()),
        };
        println!("{n:>5} {batch:>6} | {:>10.3} {:>10.3} {:>10} | {:>8.2}x {:>9}",
            ms(native), ms(ozaki), serial_ms, native / ozaki, ser_ratio);
        ws.release(&mut peer).expect("release");
        peer.unpin(pc.handle).expect("unpin");
        peer.unpin(pb.handle).expect("unpin");
        peer.unpin(pa.handle).expect("unpin");
    }

    println!("\nDone.");
}

//! gpu_peer linalg: device kernels vs the Flynnel-parallel CPU
//! reference vs serial CPU, per op, matrix size and batch size, plus
//! the block-per-matrix vs thread-per-matrix Jacobi shape comparison
//! that sets `JACOBI_THREAD_SHAPE_MAX_N`.
//!
//! Every contender computes the same result from the same inputs.
//! GPU times are kernel wall (launch + sync) with the data already
//! resident, which is the steady state of a scoring loop that pins
//! once per step; the one-time pin + fetch cost is printed separately.
//! Run with:
//!   cargo bench --profile=release-test --bench gpu_linalg

use std::time::{Duration, Instant};

use flynnel::gpu_peer::linalg::{
    EinsumSpec, JacobiShape, LinalgKernels, cpu, default_sweeps, launch_einsum, launch_gemm,
    launch_gesvd, launch_syev,
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

/// Median of `runs` timings of `f` after warming up: `f` runs until
/// at least 150 ms have elapsed (at most 20 calls), long enough for
/// a GPU that idled through a multi-second CPU phase to raise its
/// clocks before the timed calls.
fn median_ns<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    let warm = Instant::now();
    let mut calls = 0;
    while calls < 20 && (calls == 0 || warm.elapsed() < Duration::from_millis(150)) {
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
            trace("gpu launches");
            let gpu = median_ns(runs, || {
                launch_gemm(&peer, &k, pa.ptr, pb.ptr, pc.ptr, batch as u32, n as u32, n as u32, n as u32)
                    .expect("launch");
                peer.sync_wide().expect("sync");
            });
            trace("fetch");
            let mut out = vec![0u8; batch * n * n * 8];
            peer.fetch_bulk(&pc.handle, &mut out).expect("fetch");
            let pin_fetch = t_pin.elapsed().as_nanos() as f64 - gpu * (runs as f64 + 1.0);
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
        for &batch in &[8192usize, 65536] {
            let x = uniform(3, batch * n);
            let y = uniform(4, batch * n);
            let spec = EinsumSpec::parse("i,j->ij", &[n], Some(&[n])).expect("spec");
            let tables = pin(&mut peer, &spec.tables().iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            let px = pin(&mut peer, &bytes(&x));
            let py = pin(&mut peer, &bytes(&y));
            let po = pin(&mut peer, &vec![0u8; batch * n * n * 8]);
            let gpu = median_ns(runs, || {
                launch_einsum(&peer, &k, &spec, tables.ptr, px.ptr, py.ptr, po.ptr, batch as u32).expect("launch");
                peer.sync_wide().expect("sync");
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
            let gpu = median_ns(runs, || {
                launch_einsum(&peer, &k, &rspec, rtables.ptr, pa.ptr, pa.ptr, pr.ptr, batch as u32).expect("launch");
                peer.sync_wide().expect("sync");
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
        for &batch in &[1024usize, 8192, 65536] {
            if !fits(&format!("syev n={n} batch={batch}"), batch * n * n * 8 + batch * n * 8) {
                continue;
            }
            let a = symmetric_batch(10, batch, n);
            let sweeps = default_sweeps(n);
            let pa = pin(&mut peer, &bytes(&a));
            let pw = pin(&mut peer, &vec![0u8; batch * n * 8]);
            let blk = median_ns(runs, || {
                launch_syev(&peer, &k, pa.ptr, pw.ptr, None, batch as u32, n as u32, sweeps, JacobiShape::BlockPerMatrix).expect("launch");
                peer.sync_wide().expect("sync");
            });
            let thr = if n <= 16 {
                Some(median_ns(runs, || {
                    launch_syev(&peer, &k, pa.ptr, pw.ptr, None, batch as u32, n as u32, sweeps, JacobiShape::ThreadPerMatrix).expect("launch");
                    peer.sync_wide().expect("sync");
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
        for &batch in &[1024usize, 8192, 65536] {
            if !fits(&format!("gesvd n={n} batch={batch}"), batch * n * n * 8 + batch * n * 8) {
                continue;
            }
            let a = uniform(20, batch * n * n);
            let sweeps = default_sweeps(n);
            let pa = pin(&mut peer, &bytes(&a));
            let ps = pin(&mut peer, &vec![0u8; batch * n * 8]);
            // The kernel overwrites A with U; re-upload before each run.
            let a_bytes = bytes(&a);
            let blk = median_ns(runs, || {
                peer.write_resident_bulk(&pa.handle, &a_bytes).expect("reload");
                launch_gesvd(&peer, &k, pa.ptr, ps.ptr, None, batch as u32, n as u32, n as u32, sweeps, JacobiShape::BlockPerMatrix).expect("launch");
                peer.sync_wide().expect("sync");
            });
            let thr = if n <= 16 {
                Some(median_ns(runs, || {
                    peer.write_resident_bulk(&pa.handle, &a_bytes).expect("reload");
                    launch_gesvd(&peer, &k, pa.ptr, ps.ptr, None, batch as u32, n as u32, n as u32, sweeps, JacobiShape::ThreadPerMatrix).expect("launch");
                    peer.sync_wide().expect("sync");
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
    println!("\nDone.");
}

//! Parity of the gpu_peer linalg kernels against their CPU references
//! on the real device: einsum and gemm bit for bit, the Jacobi ops to
//! rounding, both kernel shapes, plus the accel_op route. Requires a
//! CUDA device; the tests fail loudly without one so a skipped run is
//! never reported as a pass.

#![cfg(feature = "gpu-peer")]

use std::sync::{Mutex, OnceLock};

use flynnel::gpu_peer::linalg::{
    self, EinsumSpec, JacobiShape, LinalgKernels, cpu, default_sweeps, einsum_batched,
    gemm_batched, gesvd_batched, syev_batched,
};
use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig};

/// One peer per test binary; tests serialize on it.
fn peer() -> std::sync::MutexGuard<'static, (GpuPeer, LinalgKernels)> {
    static PEER: OnceLock<Mutex<(GpuPeer, LinalgKernels)>> = OnceLock::new();
    PEER.get_or_init(|| {
        let peer = GpuPeer::init(GpuPeerConfig {
            slot_bytes: 64 * 1024,
            slots_per_lane: 4,
            vram_block_bytes: 8 * 1024 * 1024,
            vram_blocks: 64,
            ..GpuPeerConfig::default()
        })
        .expect("a CUDA device is required for the gpu linalg parity tests");
        let kernels = LinalgKernels::load(&peer).expect("linalg PTX loads on this driver");
        Mutex::new((peer, kernels))
    })
    .lock()
    .unwrap_or_else(std::sync::PoisonError::into_inner)
}

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

fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    s
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0f64, |m, x| m.max(x.abs()))
}

#[test]
fn einsum_matmul_bit_exact() {
    let mut g = peer();
    let (peer, k) = &mut *g;
    let (batch, m, kd, n) = (3usize, 5usize, 7usize, 4usize);
    let a = uniform(1, batch * m * kd);
    let b = uniform(2, batch * kd * n);
    let spec = EinsumSpec::parse("ij,jk->ik", &[m, kd], Some(&[kd, n])).expect("spec");
    let gpu = einsum_batched(peer, k, &spec, &a, Some(&b), batch as u32).expect("einsum");
    let want = cpu::einsum(&spec, &a, Some(&b), batch);
    assert_eq!(gpu.len(), want.len());
    for (i, (g, w)) in gpu.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}: gpu {g} cpu {w}");
    }
}

#[test]
fn einsum_outer_axissum_trace_bit_exact() {
    let mut g = peer();
    let (peer, k) = &mut *g;
    let batch = 4usize;
    let (p, q) = (6usize, 9usize);
    let x = uniform(3, batch * p);
    let y = uniform(4, batch * q);
    let outer = EinsumSpec::parse("i,j->ij", &[p], Some(&[q])).expect("spec");
    let gpu = einsum_batched(peer, k, &outer, &x, Some(&y), batch as u32).expect("outer");
    assert_eq!(gpu, cpu::einsum(&outer, &x, Some(&y), batch));

    let n = 8usize;
    let a = uniform(5, batch * n * n);
    let rows = EinsumSpec::parse("ij->i", &[n, n], None).expect("spec");
    let gpu = einsum_batched(peer, k, &rows, &a, None, batch as u32).expect("axissum");
    assert_eq!(gpu, cpu::einsum(&rows, &a, None, batch));

    let tr = EinsumSpec::parse("ii->", &[n, n], None).expect("spec");
    let gpu = einsum_batched(peer, k, &tr, &a, None, batch as u32).expect("trace");
    assert_eq!(gpu, cpu::einsum(&tr, &a, None, batch));

    let nd = EinsumSpec::parse("ij,kl->ijkl", &[3, 2], Some(&[2, 3])).expect("spec");
    let a2 = uniform(6, batch * 6);
    let b2 = uniform(7, batch * 6);
    let gpu = einsum_batched(peer, k, &nd, &a2, Some(&b2), batch as u32).expect("nd outer");
    assert_eq!(gpu, cpu::einsum(&nd, &a2, Some(&b2), batch));
}

#[test]
fn gemm_ragged_dims_bit_exact() {
    let mut g = peer();
    let (peer, k) = &mut *g;
    let (batch, m, n, kd) = (16usize, 33usize, 17usize, 45usize);
    let a = uniform(8, batch * m * kd);
    let b = uniform(9, batch * kd * n);
    let gpu = gemm_batched(peer, k, &a, &b, batch as u32, m as u32, n as u32, kd as u32).expect("gemm");
    let want = cpu::gemm_batched(&a, &b, batch, m, n, kd);
    for (i, (g, w)) in gpu.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}: gpu {g} cpu {w}");
    }
}

fn check_syev(peer: &mut GpuPeer, k: &LinalgKernels, n: usize, batch: usize, shape: JacobiShape) {
    let a = symmetric_batch(100 + n as u64, batch, n);
    let sweeps = default_sweeps(n);
    let (w, v) = syev_batched(peer, k, &a, batch as u32, n as u32, sweeps, true, shape).expect("syev");
    let (wc, _) = cpu::syev_jacobi_batched(&a, batch, n, sweeps, false);
    let v = v.expect("vectors requested");
    for bi in 0..batch {
        let ab = &a[bi * n * n..(bi + 1) * n * n];
        let scale = max_abs(&wc[bi * n..(bi + 1) * n]).max(1e-300);
        let gs = sorted(&w[bi * n..(bi + 1) * n]);
        let cs = sorted(&wc[bi * n..(bi + 1) * n]);
        for i in 0..n {
            assert!(
                (gs[i] - cs[i]).abs() <= 1e-10 * scale,
                "{shape:?} n={n} item {bi} eig {i}: gpu {} cpu {}",
                gs[i],
                cs[i]
            );
        }
        // A v = lambda v for every returned column.
        let vb = &v[bi * n * n..(bi + 1) * n * n];
        let anorm = max_abs(ab).max(1e-300);
        for col in 0..n {
            let lambda = w[bi * n + col];
            for i in 0..n {
                let mut av = 0.0;
                for j in 0..n {
                    av += ab[i * n + j] * vb[j * n + col];
                }
                assert!(
                    (av - lambda * vb[i * n + col]).abs() <= 1e-9 * anorm,
                    "{shape:?} n={n} item {bi} col {col} row {i}: residual {}",
                    (av - lambda * vb[i * n + col]).abs()
                );
            }
        }
    }
}

#[test]
fn syev_block_shape_matches_cpu() {
    let mut g = peer();
    let (peer, k) = &mut *g;
    for &(n, batch) in &[(2usize, 5usize), (4, 8), (16, 8), (33, 6), (64, 4)] {
        check_syev(peer, k, n, batch, JacobiShape::BlockPerMatrix);
    }
}

#[test]
fn syev_thread_shape_matches_cpu() {
    let mut g = peer();
    let (peer, k) = &mut *g;
    for &(n, batch) in &[(2usize, 5usize), (4, 300), (9, 64), (16, 64)] {
        check_syev(peer, k, n, batch, JacobiShape::ThreadPerMatrix);
    }
}

fn check_gesvd(peer: &mut GpuPeer, k: &LinalgKernels, m: usize, n: usize, batch: usize, shape: JacobiShape) {
    let a = uniform(200 + (m * 64 + n) as u64, batch * m * n);
    let sweeps = default_sweeps(n);
    let r = gesvd_batched(peer, k, &a, batch as u32, m as u32, n as u32, sweeps, true, shape).expect("gesvd");
    let (_, sc, _) = cpu::gesvd_jacobi_batched(&a, batch, m, n, sweeps, false);
    let v = r.v.expect("v requested");
    for bi in 0..batch {
        let gs = sorted(&r.sigma[bi * n..(bi + 1) * n]);
        let cs = sorted(&sc[bi * n..(bi + 1) * n]);
        let scale = cs[n - 1].max(1e-300);
        for i in 0..n {
            assert!(
                (gs[i] - cs[i]).abs() <= 1e-10 * scale,
                "{shape:?} {m}x{n} item {bi} sigma {i}: gpu {} cpu {}",
                gs[i],
                cs[i]
            );
        }
        // A = U diag(sigma) V^T reconstruction and U^T U = I.
        let ub = &r.u[bi * m * n..(bi + 1) * m * n];
        let vb = &v[bi * n * n..(bi + 1) * n * n];
        let sb = &r.sigma[bi * n..(bi + 1) * n];
        let ab = &a[bi * m * n..(bi + 1) * m * n];
        let anorm = max_abs(ab).max(1e-300);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for kk in 0..n {
                    acc += ub[i * n + kk] * sb[kk] * vb[j * n + kk];
                }
                assert!(
                    (acc - ab[i * n + j]).abs() <= 1e-9 * anorm,
                    "{shape:?} {m}x{n} item {bi} ({i},{j}): reconstruction error {}",
                    (acc - ab[i * n + j]).abs()
                );
            }
        }
        for p in 0..n {
            for q in 0..n {
                let mut dot = 0.0;
                for i in 0..m {
                    dot += ub[i * n + p] * ub[i * n + q];
                }
                let want = if p == q { 1.0 } else { 0.0 };
                assert!((dot - want).abs() <= 1e-9, "{shape:?} {m}x{n} item {bi}: U^T U ({p},{q}) = {dot}");
            }
        }
    }
}

#[test]
fn gesvd_block_shape_matches_cpu() {
    let mut g = peer();
    let (peer, k) = &mut *g;
    for &(m, n, batch) in &[(2usize, 2usize, 5usize), (8, 8, 8), (16, 12, 8), (40, 33, 4), (64, 64, 3)] {
        check_gesvd(peer, k, m, n, batch, JacobiShape::BlockPerMatrix);
    }
}

#[test]
fn gesvd_thread_shape_matches_cpu() {
    let mut g = peer();
    let (peer, k) = &mut *g;
    for &(m, n, batch) in &[(2usize, 2usize, 5usize), (8, 8, 300), (16, 12, 64), (16, 16, 64)] {
        check_gesvd(peer, k, m, n, batch, JacobiShape::ThreadPerMatrix);
    }
}

#[test]
fn accel_route_runs_gemm_on_device_and_matches() {
    use flynnel::backend::Backend;
    use flynnel::backend::cuda::CudaBackend;
    use flynnel::backend::registry::register_backend;
    use flynnel::sched::JobPlan;
    use std::sync::Arc;

    let mut g = peer();
    let (peer, _) = &mut *g;
    let backend = CudaBackend::new().expect("cuda backend");
    let id = Backend::Cuda { device_id: 0 };
    register_backend(Arc::new(backend));
    let ops = linalg::register_linalg_accel_ops();
    linalg::bind_linalg_kernels(&ops, id).expect("bind linalg kernels");

    let (batch, m, n, kd) = (8usize, 24usize, 20usize, 28usize);
    let a = uniform(31, batch * m * kd);
    let b = uniform(32, batch * kd * n);
    let want = cpu::gemm_batched(&a, &b, batch, m, n, kd);

    let to_bytes = |v: &[f64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let pa = peer.pin_bulk(&to_bytes(&a)).expect("pin a");
    let pb = peer.pin_bulk(&to_bytes(&b)).expect("pin b");
    let pc = peer.pin_bulk(&vec![0u8; batch * m * n * 8]).expect("pin c");
    let dev = (
        peer.resident_ptr(&pa).expect("a ptr").0,
        peer.resident_ptr(&pb).expect("b ptr").0,
        peer.resident_ptr(&pc).expect("c ptr").0,
    );

    let plan = JobPlan::new(0, batch as u32).with_backend(id);
    let mut c_host = vec![0f64; batch * m * n];
    let mut saw_backend = false;
    for _ in 0..3 {
        let report = linalg::gemm_accel(
            &plan, &ops, &a, &b, &mut c_host, dev, batch as u32, m as u32, n as u32, kd as u32,
        );
        assert!(!report.fell_back, "kernel launch failed: {report:?}");
        if report.backend_ns.is_some() {
            saw_backend = true;
        }
    }
    assert!(saw_backend, "the device side never ran across three dispatches");
    let mut out = vec![0u8; batch * m * n * 8];
    peer.fetch_bulk(&pc, &mut out).expect("fetch c");
    let got: Vec<f64> = out
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "device element {i}: {g} vs cpu {w}");
    }
    peer.unpin(pc).expect("unpin");
    peer.unpin(pb).expect("unpin");
    peer.unpin(pa).expect("unpin");
}

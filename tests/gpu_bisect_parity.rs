//! The tridiagonalization / bidiagonalization plus bisection kernels
//! against the Jacobi CPU references on the real device: eigenvalues
//! and singular values to 1e-10 relative, eigenvectors by
//! `A v = lambda v`, `A = U diag(sigma) V^T`, and orthonormal columns,
//! including repeated and vanishing spectra. Requires a CUDA device.
#![cfg(feature = "gpu-peer")]

use std::sync::{Mutex, MutexGuard};

use flynnel::gpu_peer::linalg::{
    cpu, default_sweeps, gesvd_auto_batched, gesvd_bisect_batched, gesvd_method_for,
    syev_auto_batched, syev_bisect_batched, syev_method_for, LinalgKernels, LinalgMethod,
};
use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig};

static GPU: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn peer() -> GpuPeer {
    GpuPeer::init(GpuPeerConfig {
        slot_bytes: 64 * 1024,
        slots_per_lane: 4,
        vram_block_bytes: 16 * 1024 * 1024,
        vram_blocks: 32,
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device is required for this test")
}

/// splitmix64 stream in [-1, 1).
fn uniform(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

fn symmetric_batch(seed: u64, batch: usize, n: usize) -> Vec<f64> {
    let mut a = uniform(seed, batch * n * n);
    for item in 0..batch {
        for i in 0..n {
            for j in 0..i {
                let v = 0.5 * (a[item * n * n + i * n + j] + a[item * n * n + j * n + i]);
                a[item * n * n + i * n + j] = v;
                a[item * n * n + j * n + i] = v;
            }
        }
    }
    a
}

fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    s
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0f64, |m, x| m.max(x.abs()))
}

fn check_syev(batch: usize, n: usize, seed: u64) {
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    let a = symmetric_batch(seed, batch, n);
    let (w, v) = syev_bisect_batched(&mut p, &k, &a, batch as u32, n as u32, true).expect("syev_bisect");
    let v = v.expect("eigenvectors requested");
    let (w_ref, _) = cpu::syev_jacobi_batched(&a, batch, n, default_sweeps(n), false);
    for item in 0..batch {
        let got = &w[item * n..(item + 1) * n];
        let want = sorted(&w_ref[item * n..(item + 1) * n]);
        assert!(got.windows(2).all(|p| p[0] <= p[1]), "item {item}: eigenvalues not ascending");
        let scale = max_abs(&want).max(1e-300);
        for j in 0..n {
            assert!(
                (got[j] - want[j]).abs() <= 1e-10 * scale,
                "item {item} eigenvalue {j}: {} vs {}",
                got[j], want[j]
            );
        }
        let am = &a[item * n * n..(item + 1) * n * n];
        let vm = &v[item * n * n..(item + 1) * n * n];
        let anorm = max_abs(am).max(1e-300);
        for j in 0..n {
            for i in 0..n {
                let av: f64 = (0..n).map(|kk| am[i * n + kk] * vm[kk * n + j]).sum();
                let resid = av - got[j] * vm[i * n + j];
                assert!(resid.abs() <= 1e-9 * anorm * n as f64, "item {item} A v = lambda v at ({i},{j}): {resid:e}");
            }
            for j2 in 0..n {
                let dot: f64 = (0..n).map(|i| vm[i * n + j] * vm[i * n + j2]).sum();
                let want = if j == j2 { 1.0 } else { 0.0 };
                assert!((dot - want).abs() <= 1e-9, "item {item} V^T V at ({j},{j2}): {dot}");
            }
        }
    }
}

fn check_gesvd(batch: usize, m: usize, n: usize, seed: u64) {
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    let a = uniform(seed, batch * m * n);
    let r = gesvd_bisect_batched(&mut p, &k, &a, batch as u32, m as u32, n as u32, true).expect("gesvd_bisect");
    let v = r.v.expect("V requested");
    let (_, s_ref, _) = cpu::gesvd_jacobi_batched(&a, batch, m, n, default_sweeps(n), false);
    for item in 0..batch {
        let got = &r.sigma[item * n..(item + 1) * n];
        let mut want = sorted(&s_ref[item * n..(item + 1) * n]);
        want.reverse();
        assert!(got.windows(2).all(|p| p[0] >= p[1]), "item {item}: singular values not descending");
        let scale = want[0].max(1e-300);
        for j in 0..n {
            assert!(got[j] >= 0.0, "item {item} sigma {j} negative: {}", got[j]);
            assert!(
                (got[j] - want[j]).abs() <= 1e-10 * scale,
                "item {item} singular value {j}: {} vs {}",
                got[j], want[j]
            );
        }
        let am = &a[item * m * n..(item + 1) * m * n];
        let um = &r.u[item * m * n..(item + 1) * m * n];
        let vm = &v[item * n * n..(item + 1) * n * n];
        let anorm = max_abs(am).max(1e-300);
        for i in 0..m {
            for j in 0..n {
                let rec: f64 = (0..n).map(|kk| um[i * n + kk] * got[kk] * vm[j * n + kk]).sum();
                assert!((rec - am[i * n + j]).abs() <= 1e-9 * anorm * n as f64, "item {item} reconstruction at ({i},{j})");
            }
        }
        for j in 0..n {
            for j2 in 0..n {
                let du: f64 = (0..m).map(|i| um[i * n + j] * um[i * n + j2]).sum();
                let dv: f64 = (0..n).map(|i| vm[i * n + j] * vm[i * n + j2]).sum();
                let want = if j == j2 { 1.0 } else { 0.0 };
                assert!((du - want).abs() <= 1e-9, "item {item} U^T U at ({j},{j2}): {du}");
                assert!((dv - want).abs() <= 1e-9, "item {item} V^T V at ({j},{j2}): {dv}");
            }
        }
    }
}

#[test]
fn syev_bisect_matches_jacobi_n4() {
    check_syev(16, 4, 31);
}

#[test]
fn syev_bisect_matches_jacobi_n32() {
    check_syev(4, 32, 32);
}

#[test]
fn syev_bisect_matches_jacobi_n64() {
    check_syev(3, 64, 33);
}

#[test]
fn syev_bisect_handles_diagonal_and_repeated_eigenvalues() {
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    let n = 8usize;
    let mut a = vec![0f64; n * n];
    let diag = [3.0, -1.0, 3.0, 0.0, 2.5, 3.0, -1.0, 7.0];
    for i in 0..n {
        a[i * n + i] = diag[i];
    }
    let (w, v) = syev_bisect_batched(&mut p, &k, &a, 1, n as u32, true).expect("syev_bisect");
    let want = sorted(&diag);
    for j in 0..n {
        assert!((w[j] - want[j]).abs() <= 1e-12, "eigenvalue {j}: {} vs {}", w[j], want[j]);
    }
    let v = v.expect("V");
    for j in 0..n {
        for i in 0..n {
            let av: f64 = (0..n).map(|kk| a[i * n + kk] * v[kk * n + j]).sum();
            assert!((av - w[j] * v[i * n + j]).abs() <= 1e-10, "A v = lambda v at ({i},{j})");
        }
        for j2 in 0..n {
            let dot: f64 = (0..n).map(|i| v[i * n + j] * v[i * n + j2]).sum();
            let want = if j == j2 { 1.0 } else { 0.0 };
            assert!((dot - want).abs() <= 1e-10, "V^T V at ({j},{j2}): {dot}");
        }
    }
}

/// The automatic helpers route by the measured rule and agree with
/// the references whichever kernel they pick.
#[test]
fn auto_helpers_route_by_the_measured_rule() {
    assert_eq!(syev_method_for(16), LinalgMethod::Jacobi);
    assert_eq!(syev_method_for(32), LinalgMethod::Bisection);
    assert_eq!(gesvd_method_for(32), LinalgMethod::Jacobi);
    assert_eq!(gesvd_method_for(64), LinalgMethod::Bisection);
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    for &n in &[16usize, 64] {
        let batch = 2usize;
        let a = symmetric_batch(50 + n as u64, batch, n);
        let (w, _) = syev_auto_batched(&mut p, &k, &a, batch as u32, n as u32, false).expect("syev_auto");
        let (w_ref, _) = cpu::syev_jacobi_batched(&a, batch, n, default_sweeps(n), false);
        for item in 0..batch {
            let got = sorted(&w[item * n..(item + 1) * n]);
            let want = sorted(&w_ref[item * n..(item + 1) * n]);
            let scale = max_abs(&want).max(1e-300);
            for j in 0..n {
                assert!((got[j] - want[j]).abs() <= 1e-10 * scale, "n={n} item {item} eigenvalue {j}");
            }
        }
        let b = uniform(60 + n as u64, batch * n * n);
        let r = gesvd_auto_batched(&mut p, &k, &b, batch as u32, n as u32, n as u32, false).expect("gesvd_auto");
        let (_, s_ref, _) = cpu::gesvd_jacobi_batched(&b, batch, n, n, default_sweeps(n), false);
        for item in 0..batch {
            let got = sorted(&r.sigma[item * n..(item + 1) * n]);
            let want = sorted(&s_ref[item * n..(item + 1) * n]);
            let scale = max_abs(&want).max(1e-300);
            for j in 0..n {
                assert!((got[j] - want[j]).abs() <= 1e-10 * scale, "n={n} item {item} singular value {j}");
            }
        }
    }
}

#[test]
fn gesvd_bisect_matches_jacobi_square_n32() {
    check_gesvd(4, 32, 32, 41);
}

#[test]
fn gesvd_bisect_matches_jacobi_square_n64() {
    check_gesvd(2, 64, 64, 42);
}

#[test]
fn gesvd_bisect_matches_jacobi_rectangular_48x32() {
    check_gesvd(3, 48, 32, 43);
}

#[test]
fn gesvd_bisect_small_and_rank_deficient() {
    check_gesvd(8, 6, 4, 44);
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    // Rank-2 matrix: two identical column pairs. Singular values 2
    // and 3 vanish; U and V must still be orthonormal and reconstruct.
    let (m, n) = (8usize, 4usize);
    let base = uniform(45, m * 2);
    let mut a = vec![0f64; m * n];
    for i in 0..m {
        a[i * n] = base[i * 2];
        a[i * n + 1] = base[i * 2 + 1];
        a[i * n + 2] = base[i * 2];
        a[i * n + 3] = base[i * 2 + 1];
    }
    let r = gesvd_bisect_batched(&mut p, &k, &a, 1, m as u32, n as u32, true).expect("gesvd_bisect");
    let v = r.v.expect("V");
    assert!(r.sigma[2].abs() <= 1e-12 * r.sigma[0] && r.sigma[3].abs() <= 1e-12 * r.sigma[0],
        "rank-2 matrix must have two vanishing singular values: {:?}", r.sigma);
    let anorm = max_abs(&a);
    for i in 0..m {
        for j in 0..n {
            let rec: f64 = (0..n).map(|kk| r.u[i * n + kk] * r.sigma[kk] * v[j * n + kk]).sum();
            assert!((rec - a[i * n + j]).abs() <= 1e-9 * anorm * n as f64, "reconstruction at ({i},{j})");
        }
    }
    for j in 0..n {
        for j2 in 0..n {
            let du: f64 = (0..m).map(|i| r.u[i * n + j] * r.u[i * n + j2]).sum();
            let dv: f64 = (0..n).map(|i| v[i * n + j] * v[i * n + j2]).sum();
            let want = if j == j2 { 1.0 } else { 0.0 };
            assert!((du - want).abs() <= 1e-9, "U^T U at ({j},{j2}): {du}");
            assert!((dv - want).abs() <= 1e-9, "V^T V at ({j},{j2}): {dv}");
        }
    }
}

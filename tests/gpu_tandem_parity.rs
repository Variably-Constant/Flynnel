//! The tandem helpers split a batch between the device and the CPU
//! pool by the call site's learned share; every item must come back
//! the same as the reference whichever side computed it, in the
//! uniform order (eigenvalues ascending, singular values descending),
//! and the share must move toward the measured balance over repeated
//! calls. Requires a CUDA device.
#![cfg(feature = "gpu-peer")]

use std::sync::{Mutex, MutexGuard};

use flynnel::JobPlan;
use flynnel::gpu_peer::linalg::{
    LinalgKernels, cpu, default_sweeps, gemm_tandem_batched, gesvd_tandem_batched,
    sort_eigenpairs_ascending, sort_singular_descending, syev_tandem_batched,
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

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0f64, |m, x| m.max(x.abs()))
}

#[test]
fn gemm_tandem_is_bit_exact_and_covers_every_item() {
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    let (batch, n) = (1024usize, 32usize);
    let a = uniform(71, batch * n * n);
    let b = uniform(72, batch * n * n);
    let reference = cpu::gemm_batched(&a, &b, batch, n, n, n);
    let plan = JobPlan::new(0, batch as u32);
    let mut last = None;
    for round in 0..6 {
        let (got, report) =
            gemm_tandem_batched(&mut p, &k, &plan, &a, &b, batch as u32, n as u32, n as u32, n as u32)
                .expect("gemm tandem");
        assert_eq!(report.cpu_items + report.backend_items, batch, "round {round}: items covered");
        assert!(report.cpu_items >= 1 && report.backend_items >= 1, "round {round}: both sides run");
        assert!((50..=950).contains(&report.cpu_share_per_mille), "round {round}: share {}", report.cpu_share_per_mille);
        for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
            assert_eq!(g.to_bits(), r.to_bits(), "round {round} element {i}: {g} vs {r}");
        }
        eprintln!(
            "gemm round {round}: share {} cpu {} items {:.2} ms, device {} items {:.2} ms",
            report.cpu_share_per_mille, report.cpu_items, report.cpu_ns as f64 / 1e6,
            report.backend_items, report.backend_ns as f64 / 1e6
        );
        last = Some(report);
    }
    // After warm rounds the share reflects the measured balance: the
    // slower side per item holds the smaller share.
    let r = last.expect("six rounds");
    let cpu_per = r.cpu_ns as f64 / r.cpu_items as f64;
    let dev_per = r.backend_ns as f64 / r.backend_items as f64;
    if cpu_per > 2.0 * dev_per {
        assert!(r.cpu_share_per_mille < 500, "CPU slower per item yet holds {} per mille", r.cpu_share_per_mille);
    } else if dev_per > 2.0 * cpu_per {
        assert!(r.cpu_share_per_mille > 500, "device slower per item yet CPU holds {} per mille", r.cpu_share_per_mille);
    }
}

#[test]
fn syev_tandem_orders_every_item_ascending_and_matches_reference() {
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    for &n in &[16usize, 64] {
        let batch = 512usize;
        let a = symmetric_batch(80 + n as u64, batch, n);
        let (mut w_ref, mut v_ref) = cpu::syev_jacobi_batched(&a, batch, n, default_sweeps(n), true);
        sort_eigenpairs_ascending(&mut w_ref, v_ref.as_deref_mut(), n);
        let plan = JobPlan::new(0, batch as u32);
        for round in 0..3 {
            let ((w, v), report) =
                syev_tandem_batched(&mut p, &k, &plan, &a, batch as u32, n as u32, true).expect("syev tandem");
            let v = v.expect("eigenvectors requested");
            assert_eq!(report.cpu_items + report.backend_items, batch);
            for item in 0..batch {
                let wi = &w[item * n..(item + 1) * n];
                assert!(wi.windows(2).all(|p| p[0] <= p[1]), "n={n} round {round} item {item}: not ascending");
                let scale = max_abs(&w_ref[item * n..(item + 1) * n]).max(1e-300);
                for j in 0..n {
                    assert!((wi[j] - w_ref[item * n + j]).abs() <= 1e-10 * scale, "n={n} item {item} eigenvalue {j}");
                }
                let am = &a[item * n * n..(item + 1) * n * n];
                let vm = &v[item * n * n..(item + 1) * n * n];
                let anorm = max_abs(am).max(1e-300);
                for j in 0..n {
                    for i in 0..n {
                        let av: f64 = (0..n).map(|kk| am[i * n + kk] * vm[kk * n + j]).sum();
                        assert!((av - wi[j] * vm[i * n + j]).abs() <= 1e-9 * anorm * n as f64, "n={n} item {item} A v = lambda v ({i},{j})");
                    }
                }
            }
        }
    }
}

#[test]
fn gesvd_tandem_orders_every_item_descending_and_reconstructs() {
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    for &n in &[16usize, 64] {
        let batch = 256usize;
        let a = uniform(90 + n as u64, batch * n * n);
        let (mut u_ref, mut s_ref, mut v_ref) = cpu::gesvd_jacobi_batched(&a, batch, n, n, default_sweeps(n), true);
        sort_singular_descending(&mut u_ref, &mut s_ref, v_ref.as_deref_mut(), n, n);
        let plan = JobPlan::new(0, batch as u32);
        for round in 0..3 {
            let (r, report) = gesvd_tandem_batched(&mut p, &k, &plan, &a, batch as u32, n as u32, n as u32, true)
                .expect("gesvd tandem");
            let v = r.v.expect("V requested");
            assert_eq!(report.cpu_items + report.backend_items, batch);
            for item in 0..batch {
                let si = &r.sigma[item * n..(item + 1) * n];
                assert!(si.windows(2).all(|p| p[0] >= p[1]), "n={n} round {round} item {item}: not descending");
                let scale = s_ref[item * n].max(1e-300);
                for j in 0..n {
                    assert!((si[j] - s_ref[item * n + j]).abs() <= 1e-10 * scale, "n={n} item {item} singular value {j}");
                }
                let am = &a[item * n * n..(item + 1) * n * n];
                let um = &r.u[item * n * n..(item + 1) * n * n];
                let vm = &v[item * n * n..(item + 1) * n * n];
                let anorm = max_abs(am).max(1e-300);
                for i in 0..n {
                    for j in 0..n {
                        let rec: f64 = (0..n).map(|kk| um[i * n + kk] * si[kk] * vm[j * n + kk]).sum();
                        assert!((rec - am[i * n + j]).abs() <= 1e-9 * anorm * n as f64, "n={n} item {item} reconstruction ({i},{j})");
                    }
                }
            }
        }
    }
}

#[test]
fn sort_helpers_permute_columns_with_their_values() {
    let n = 3usize;
    let mut w = vec![3.0, 1.0, 2.0];
    // Columns: v[:,0] = (1,0,0) for 3, v[:,1] = (0,1,0) for 1, v[:,2] = (0,0,1) for 2.
    let mut v = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    sort_eigenpairs_ascending(&mut w, Some(&mut v), n);
    assert_eq!(w, vec![1.0, 2.0, 3.0]);
    assert_eq!(v, vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let mut s = vec![1.0, 3.0, 2.0];
    let mut u = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut vv = u.clone();
    sort_singular_descending(&mut u, &mut s, Some(&mut vv), n, n);
    assert_eq!(s, vec![3.0, 2.0, 1.0]);
    assert_eq!(u, vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    assert_eq!(vv, u);
}

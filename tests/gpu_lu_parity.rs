//! Parity of the batched LU kernels against the CPU reference on the
//! real device: factors, pivots and flags bit for bit, solves and
//! inverses bit for bit, determinants and residuals to rounding.
//! Requires a CUDA device.
#![cfg(feature = "gpu-peer")]

use std::sync::{Mutex, MutexGuard};

use flynnel::gpu_peer::linalg::{
    LinalgKernels, cpu, getrf_batched, getri_batched, getrs_batched, lu_det_batched,
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

fn assert_bits_equal(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what} element {i}: gpu {g} cpu {w}");
    }
}

#[test]
fn getrf_matches_reference_bit_for_bit_at_every_size() {
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    for &n in &[1usize, 2, 3, 8, 16, 33, 64] {
        let batch = 64usize;
        let a = uniform(100 + n as u64, batch * n * n);
        let got = getrf_batched(&mut p, &k, &a, batch as u32, n as u32).expect("getrf");
        let (lu, piv, info) = cpu::getrf_batched(&a, batch, n);
        assert_bits_equal(&format!("n={n} lu"), &got.lu, &lu);
        assert_eq!(got.piv, piv, "n={n} pivots");
        assert_eq!(got.info, info, "n={n} flags");
        assert!(info.iter().all(|&f| f == 0), "n={n}: random matrices are nonsingular");
    }
}

#[test]
fn getrs_and_getri_match_reference_and_solve_the_system() {
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    for &(n, nrhs) in &[(4usize, 1usize), (16, 3), (64, 1), (64, 64), (33, 17)] {
        let batch = 32usize;
        let a = uniform(200 + n as u64, batch * n * n);
        let b = uniform(300 + n as u64, batch * n * nrhs);
        let f = getrf_batched(&mut p, &k, &a, batch as u32, n as u32).expect("getrf");
        let x = getrs_batched(&mut p, &k, &f.lu, &f.piv, &b, batch as u32, n as u32, nrhs as u32).expect("getrs");
        let want = cpu::getrs_batched(&f.lu, &f.piv, &b, batch, n, nrhs);
        assert_bits_equal(&format!("n={n} nrhs={nrhs} solve"), &x, &want);
        // Residual A x - b relative to |A| |x| + |b|.
        for item in 0..batch {
            let am = &a[item * n * n..(item + 1) * n * n];
            let xm = &x[item * n * nrhs..(item + 1) * n * nrhs];
            let bm = &b[item * n * nrhs..(item + 1) * n * nrhs];
            for i in 0..n {
                for j in 0..nrhs {
                    let mut ax = 0.0;
                    let mut scale = bm[i * nrhs + j].abs();
                    for kk in 0..n {
                        ax += am[i * n + kk] * xm[kk * nrhs + j];
                        scale += (am[i * n + kk] * xm[kk * nrhs + j]).abs();
                    }
                    assert!((ax - bm[i * nrhs + j]).abs() <= 1e-12 * scale.max(1e-300) * n as f64, "n={n} item {item} residual ({i},{j})");
                }
            }
        }
        if nrhs == n {
            let inv = getri_batched(&mut p, &k, &f.lu, &f.piv, batch as u32, n as u32).expect("getri");
            let want = cpu::getri_batched(&f.lu, &f.piv, batch, n);
            assert_bits_equal(&format!("n={n} inverse"), &inv, &want);
            for item in 0..batch {
                let am = &a[item * n * n..(item + 1) * n * n];
                let im = &inv[item * n * n..(item + 1) * n * n];
                for i in 0..n {
                    for j in 0..n {
                        let mut s = 0.0;
                        let mut scale = 0.0;
                        for kk in 0..n {
                            s += am[i * n + kk] * im[kk * n + j];
                            scale += (am[i * n + kk] * im[kk * n + j]).abs();
                        }
                        let want = if i == j { 1.0 } else { 0.0 };
                        assert!((s - want).abs() <= 1e-12 * scale.max(1.0) * n as f64, "n={n} item {item} A inv(A) ({i},{j})");
                    }
                }
            }
        }
    }
}

#[test]
fn determinants_follow_the_pivot_signs() {
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    // 2 x 2 with known determinants, one needing a row swap.
    let a = vec![
        1.0, 2.0, 3.0, 4.0, // det -2, swap (3 > 1)
        4.0, 3.0, 2.0, 1.0, // det -2, no swap
        2.0, 0.0, 0.0, 5.0, // det 10
    ];
    let f = getrf_batched(&mut p, &k, &a, 3, 2).expect("getrf");
    let det = lu_det_batched(&f.lu, &f.piv, 3, 2);
    for (got, want) in det.iter().zip(&[-2.0, -2.0, 10.0]) {
        assert!((got - want).abs() <= 1e-14, "det {got} vs {want}");
    }
    assert_eq!(f.piv, vec![1, 1, 0, 1, 0, 1]);
}

#[test]
fn singular_matrix_sets_info_on_both_sides() {
    let _serial = serial();
    let mut p = peer();
    let k = LinalgKernels::load(&p).expect("linalg PTX");
    let n = 4usize;
    let mut a = uniform(400, 2 * n * n);
    // Second item: a zero column, so step 2 finds no pivot.
    for i in 0..n {
        a[n * n + i * n + 2] = 0.0;
    }
    let got = getrf_batched(&mut p, &k, &a, 2, n as u32).expect("getrf");
    let (lu, piv, info) = cpu::getrf_batched(&a, 2, n);
    assert_eq!(got.info, info);
    assert_eq!(info, vec![0, 3]);
    assert_eq!(got.piv, piv);
    assert_bits_equal("singular lu", &got.lu, &lu);
}

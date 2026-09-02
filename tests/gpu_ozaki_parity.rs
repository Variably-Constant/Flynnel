//! The Ozaki-scheme f64 GEMM against the CPU reference: exact where
//! the reference is exact (small integers), and within the scheme's
//! error bound `2^-53 * k * max|A row| * max|B column|` per element
//! elsewhere, including ill-scaled rows and columns. Requires a CUDA
//! device with int8 tensor cores.
#![cfg(feature = "gpu-peer")]

use std::sync::{Mutex, MutexGuard};

use flynnel::gpu_peer::linalg::cpu;
use flynnel::gpu_peer::ozaki::{error_bound, ozaki_gemm_batched, OzakiKernels};
use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig};

/// One device at a time: each test builds its own peer and pool.
static GPU: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn peer() -> GpuPeer {
    GpuPeer::init(GpuPeerConfig {
        slot_bytes: 64 * 1024,
        slots_per_lane: 4,
        vram_block_bytes: 16 * 1024 * 1024,
        vram_blocks: 48,
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

/// Checks every element of `got` against `reference` within the
/// scheme's bound for its row of `a` and column of `b`, over `batch`
/// products of shape `(m, n, k)`; returns the largest ratio of error
/// to bound.
fn check_within_bound(
    a: &[f64],
    b: &[f64],
    got: &[f64],
    reference: &[f64],
    batch: usize,
    shape: (usize, usize, usize),
) -> f64 {
    let (m, n, k) = shape;
    let mut worst = 0f64;
    for item in 0..batch {
        for r in 0..m {
            let a_row = &a[(item * m + r) * k..(item * m + r + 1) * k];
            for c in 0..n {
                let b_col: Vec<f64> = (0..k).map(|kk| b[(item * k + kk) * n + c]).collect();
                let bound = error_bound(a_row, &b_col);
                let idx = (item * m + r) * n + c;
                let err = (got[idx] - reference[idx]).abs();
                // The reference carries its own rounding; allow it the
                // same budget.
                let ratio = err / (2.0 * bound).max(f64::MIN_POSITIVE);
                worst = worst.max(ratio);
                assert!(
                    ratio <= 1.0,
                    "item {item} ({r},{c}): got {} reference {} error {err:e} bound {bound:e}",
                    got[idx], reference[idx]
                );
            }
        }
    }
    worst
}

#[test]
fn ozaki_is_exact_on_small_integers() {
    let _serial = serial();
    let mut p = peer();
    let k = OzakiKernels::load(&p).expect("ozaki PTX");
    let (batch, n) = (4usize, 64usize);
    let a: Vec<f64> = uniform(1, batch * n * n).iter().map(|x| (x * 50.0).round()).collect();
    let b: Vec<f64> = uniform(2, batch * n * n).iter().map(|x| (x * 50.0).round()).collect();
    let got = ozaki_gemm_batched(&mut p, &k, &a, &b, batch as u32, n as u32, n as u32, n as u32)
        .expect("ozaki gemm");
    let reference = cpu::gemm_batched(&a, &b, batch, n, n, n);
    assert_eq!(got.len(), reference.len());
    for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
        assert_eq!(g.to_bits(), r.to_bits(), "element {i}: {g} vs {r}");
    }
}

#[test]
fn ozaki_uniform_operands_within_bound() {
    let _serial = serial();
    let mut p = peer();
    let k = OzakiKernels::load(&p).expect("ozaki PTX");
    let (batch, n) = (8usize, 64usize);
    let a = uniform(3, batch * n * n);
    let b = uniform(4, batch * n * n);
    let got = ozaki_gemm_batched(&mut p, &k, &a, &b, batch as u32, n as u32, n as u32, n as u32)
        .expect("ozaki gemm");
    let reference = cpu::gemm_batched(&a, &b, batch, n, n, n);
    let worst = check_within_bound(&a, &b, &got, &reference, batch, (n, n, n));
    eprintln!("uniform 64x64x64 x8: worst error / bound = {worst:.3}");
}

#[test]
fn ozaki_ill_scaled_rows_and_columns_within_bound() {
    let _serial = serial();
    let mut p = peer();
    let k = OzakiKernels::load(&p).expect("ozaki PTX");
    let (batch, n) = (2usize, 64usize);
    // Row r of A scaled by 2^(-r), column c of B by 2^(c - 32), plus
    // a few exact zeros.
    let mut a = uniform(5, batch * n * n);
    let mut b = uniform(6, batch * n * n);
    for item in 0..batch {
        for r in 0..n {
            for c in 0..n {
                a[(item * n + r) * n + c] *= 2f64.powi(-(r as i32));
                b[(item * n + r) * n + c] *= 2f64.powi(c as i32 - 32);
            }
        }
        a[item * n * n + 7] = 0.0;
        b[item * n * n + 9 * n + 3] = 0.0;
    }
    let got = ozaki_gemm_batched(&mut p, &k, &a, &b, batch as u32, n as u32, n as u32, n as u32)
        .expect("ozaki gemm");
    let reference = cpu::gemm_batched(&a, &b, batch, n, n, n);
    let worst = check_within_bound(&a, &b, &got, &reference, batch, (n, n, n));
    eprintln!("ill-scaled 64x64x64 x2: worst error / bound = {worst:.3}");
}

#[test]
fn ozaki_single_512_within_bound() {
    let _serial = serial();
    let mut p = peer();
    let k = OzakiKernels::load(&p).expect("ozaki PTX");
    let n = 512usize;
    let a = uniform(7, n * n);
    let b = uniform(8, n * n);
    let got = ozaki_gemm_batched(&mut p, &k, &a, &b, 1, n as u32, n as u32, n as u32)
        .expect("ozaki gemm");
    let reference = cpu::gemm_batched(&a, &b, 1, n, n, n);
    let worst = check_within_bound(&a, &b, &got, &reference, 1, (n, n, n));
    eprintln!("single 512x512x512: worst error / bound = {worst:.3}");
}

#[test]
fn ozaki_rejects_unaligned_shapes() {
    let _serial = serial();
    let mut p = peer();
    let k = OzakiKernels::load(&p).expect("ozaki PTX");
    let a = vec![1.0; 48 * 48];
    assert!(ozaki_gemm_batched(&mut p, &k, &a, &a, 1, 48, 48, 48).is_err());
    let a = vec![1.0; 32 * 24];
    let b = vec![1.0; 24 * 32];
    assert!(ozaki_gemm_batched(&mut p, &k, &a, &b, 1, 32, 32, 24).is_err());
}

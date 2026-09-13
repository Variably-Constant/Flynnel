//! What the K_inner width buys on this host, per matmul shape.
//!
//! `JobPlan::with_k_inner_log2` asks the batched CPU matmul to carry
//! `2^log2` output cells through each k-iteration instead of one. The
//! arithmetic is identical either way - every cell accumulates over `k`
//! in the same order against the same operands - so the only thing to
//! measure is time, and the width that pays is a property of the host's
//! register file and the matrix shape rather than of this crate.
//!
//! Prints one row per (shape, width): the median wall of a fixed batch,
//! and its ratio against the one-cell loop. A ratio below 1 is a win.
//! Widths past a matrix's column count are reported too, because that
//! is where the blocked loop takes no whole group and every column runs
//! through the tail - the row should read 1.00 there rather than a
//! speedup, and a row that does not is the bench disagreeing with the
//! implementation.
//!
//! Correctness is not this bench's job and is not checked here: the
//! suite asserts bit-equality against the one-cell loop in
//! `a_lane_blocked_matmul_is_bit_identical_to_the_one_cell_loop`.

use std::hint::black_box;
use std::time::Instant;

use flynnel::gpu_peer::linalg::cpu;

/// Passes per cell. The reported figure is the median of these, so a
/// single scheduling excursion moves the row by nothing.
const PASSES: usize = 7;

fn median_ns(passes: usize, mut f: impl FnMut()) -> f64 {
    let mut runs: Vec<f64> = (0..passes)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_nanos() as f64
        })
        .collect();
    runs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a measured wall"));
    runs[runs.len() / 2]
}

fn operands(batch: usize, m: usize, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let a = (0..batch * m * k).map(|i| (i as f64 * 0.37).sin()).collect();
    let b = (0..batch * k * n).map(|i| (i as f64 * 0.11).cos()).collect();
    (a, b)
}

fn main() {
    // Square shapes across the range a tandem split actually hands the
    // CPU half, plus one tall-thin and one short-wide, because the
    // width blocks the column loop and a matrix with few columns is
    // the case where it cannot.
    let shapes: &[(usize, usize, usize, usize)] = &[
        (64, 16, 16, 16),
        (16, 32, 32, 32),
        (4, 64, 64, 64),
        (1, 128, 128, 128),
        (16, 64, 4, 64),
        (16, 4, 64, 64),
    ];
    println!("shape(batch,m,n,k)  lanes  median_ns  ratio_vs_1");
    for &(batch, m, n, k) in shapes {
        let (a, b) = operands(batch, m, n, k);
        let base = median_ns(PASSES, || {
            black_box(cpu::gemm_batched(&a, &b, batch, m, n, k));
        });
        println!("{batch},{m},{n},{k}  1  {base:.0}  1.000");
        for log2 in 1u8..=5 {
            let lanes = 1usize << log2;
            let t = median_ns(PASSES, || {
                black_box(cpu::gemm_batched_lanes(&a, &b, batch, m, n, k, lanes));
            });
            println!("{batch},{m},{n},{k}  {lanes}  {t:.0}  {:.3}", t / base);
        }
    }
}

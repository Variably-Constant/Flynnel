//! Trace the closing-loop observer's decision flow across multiple
//! workload shapes. Prints `(mean_ns, cv2_per_mille, classified,
//! active)` for each iteration so we can see why the auto-classifier
//! picks one DispatchProfile over another -- and where it gets the
//! classification wrong for genuinely-streaming workloads.

#![allow(clippy::needless_range_loop)]

use flynnel::sched::adaptive_profile::{active_workload_class, classify_observed};
use flynnel::sched::par_iter::for_each_chunk_indexed_min_leaf;
use flynnel::sched::split_observer::{leaf_cv_squared_per_mille, snapshot_leaf_stats};
use flynnel::JobPlan;

const RGB_W: usize = 2048;
const RGB_H: usize = 2048;

fn make_rgb(seed: u64) -> Vec<u8> {
    let mut s = seed.wrapping_add(0xDEAD_BEEF_FEED_1234);
    (0..RGB_W * RGB_H * 3)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 32) as u8
        })
        .collect()
}

#[inline]
fn gray(r: u8, g: u8, b: u8) -> u8 {
    let y = (r as u32 * 76 + g as u32 * 150 + b as u32 * 30) >> 8;
    y as u8
}

fn rgb_to_gray_one_iter(input: &[u8], output: &mut [u8]) {
    let plan = JobPlan::new(6, (RGB_W * RGB_H) as u32);
    for_each_chunk_indexed_min_leaf(&plan, output, 1024, |start_idx, slab| {
        for (i, slot) in slab.iter_mut().enumerate() {
            let pi = start_idx + i;
            *slot = gray(input[pi * 3], input[pi * 3 + 1], input[pi * 3 + 2]);
        }
    });
}

fn print_header() {
    println!("{:>4} | {:>10} | {:>8} | {:>16} | {:>16}",
        "iter", "stats.cnt", "cv2_pm", "classified", "active");
}

fn print_row(iter: u32, prev_count: u64) -> u64 {
    let s = snapshot_leaf_stats();
    let delta_count = s.count.saturating_sub(prev_count);
    let delta_mean = if delta_count > 0 {
        s.sum_ns / s.count.max(1)
    } else {
        0
    };
    let cv2 = leaf_cv_squared_per_mille(s).unwrap_or(0);
    let classified = classify_observed(delta_mean, cv2);
    let active = active_workload_class();
    println!("{:>4} | {:>10} | {:>8} | {:>16} | {:>16}",
        iter, s.count, cv2, format!("{classified:?}"), format!("{active:?}"));
    s.count
}

fn main() {
    println!("== Observer-trace: rgb_to_gray (streaming per-pixel) ==");
    println!();
    let input = make_rgb(0xFE_FE_FE_FE);
    let mut output = vec![0u8; RGB_W * RGB_H];
    print_header();
    let mut prev = 0u64;
    for i in 0..15 {
        rgb_to_gray_one_iter(&input, &mut output);
        prev = print_row(i, prev);
    }
    println!();
    println!("Active class at the end: {:?}", active_workload_class());
    println!();
    println!("If the active class settles on something OTHER than Streaming");
    println!("for this purely-sequential per-pixel workload, the classifier");
    println!("thresholds (cv2 < 50 for Streaming) need to widen to absorb");
    println!("measurement noise.");
}

//! Where dispatching a tile-shaped batch starts beating a serial loop.
//!
//! `pick_tier` promotes a declared matrix-extension class out of Inline
//! at any batch size above one. A consumer measured that as a loss at
//! four tiles, 0.94x, and a win at sixteen, 2.71x, so the crossover
//! lies between and is not located. Their sweep steps 1, 4, 16, and the
//! strip it runs on is not Flynnel's code.
//!
//! This locates it in Flynnel's own terms. A tile is a per-item cost
//! and a grid is a batch size, so the question is at what batch size a
//! dispatch of items costing about a tile each overtakes running them
//! in a loop. It needs no matrix extension: the class only selects the
//! branch, and the branch's worth is the dispatch against the work.
//!
//! Both arms run in one process and alternate which goes first, because
//! a batch measured second on a cold pool reads worse for reasons that
//! belong to neither arm.
//!
//! The reported figure is the MEDIAN over repeats of the ratio serial
//! over dispatched. Above one, dispatching wins. The crossover is the
//! smallest batch where that holds, and it is printed rather than
//! turned into a constant here: a threshold belongs where the host
//! measures it, not where a run happens to see it.
//!
//! ```sh
//! tile_crossover [tile_ns] [repeats]
//! ```

use std::env;
use std::hint::black_box;
use std::time::Instant;

use flynnel::sched::par_iter::for_each_chunk_min_leaf;
use flynnel::{CallSiteState, JobPlan, SiteRef};

static SITE: CallSiteState = CallSiteState::new();

/// Batch sizes swept. Dense between 2 and 16 because that is where the
/// consumer's sweep leaves a gap, and carried past it so the shape
/// after the crossing is visible rather than assumed.
const SIZES: &[usize] = &[1, 2, 3, 4, 6, 8, 9, 12, 16, 25, 36, 64, 128];

fn arg<T: std::str::FromStr>(n: usize, default: T) -> T {
    match env::args().nth(n) {
        None => default,
        Some(text) => text.parse().unwrap_or(default),
    }
}

/// Rounds that cost about `target_ns`, measured rather than assumed.
///
/// A rep count picked from an instruction count would be wrong by
/// whatever the host's clock rate is, and the whole point of this probe
/// is that the per-item cost is the tile's.
fn reps_for(target_ns: u64) -> u32 {
    let mut reps = 16u32;
    loop {
        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..reps {
            acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        }
        black_box(acc);
        let ns = t0.elapsed().as_nanos() as u64;
        if ns >= target_ns || reps > 1 << 22 {
            return reps;
        }
        reps = reps.saturating_mul(2);
    }
}

#[inline]
fn grind(slot: &mut u64, reps: u32) {
    let mut acc = *slot;
    for _ in 0..reps {
        acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    }
    *slot = black_box(acc);
}

fn main() {
    let tile_ns: u64 = arg(1, 150);
    let repeats: usize = arg(2, 40);

    let reps = reps_for(tile_ns);
    println!("tile_crossover tile_ns={tile_ns} reps={reps} repeats={repeats}");

    let mut best_win: Option<usize> = None;

    for &n in SIZES {
        let mut buf: Vec<u64> = (0..n as u64).collect();
        let mut ratios: Vec<f64> = Vec::with_capacity(repeats);

        for r in 0..repeats {
            let serial_first = r % 2 == 0;
            let mut serial_ns = 0u64;
            let mut par_ns = 0u64;

            for half in 0..2 {
                if (half == 0) == serial_first {
                    let t0 = Instant::now();
                    for slot in buf.iter_mut() {
                        grind(slot, reps);
                    }
                    serial_ns = t0.elapsed().as_nanos() as u64;
                } else {
                    let plan = JobPlan::new(0, n as u32).with_site(SiteRef::new(&SITE));
                    let t0 = Instant::now();
                    for_each_chunk_min_leaf(&plan, &mut buf, 1, |chunk| {
                        for slot in chunk.iter_mut() {
                            grind(slot, reps);
                        }
                    });
                    par_ns = t0.elapsed().as_nanos() as u64;
                }
            }

            if par_ns > 0 {
                ratios.push(serial_ns as f64 / par_ns as f64);
            }
        }

        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = if ratios.is_empty() {
            0.0
        } else if ratios.len() % 2 == 1 {
            ratios[ratios.len() / 2]
        } else {
            (ratios[ratios.len() / 2 - 1] + ratios[ratios.len() / 2]) / 2.0
        };

        // The spread over repeats, so a median close to one can be read
        // against how much it moved rather than on its own.
        let lo = ratios.first().copied().unwrap_or(0.0);
        let hi = ratios.last().copied().unwrap_or(0.0);
        println!("size {n} ratio_med {med:.3} ratio_min {lo:.3} ratio_max {hi:.3} n {}", ratios.len());

        if med > 1.0 && best_win.is_none() {
            best_win = Some(n);
        }
    }

    match best_win {
        Some(n) => println!(
            "crossover at batch {n}: the smallest size whose median ratio exceeds one"
        ),
        None => println!(
            "no crossover in this sweep: dispatching did not beat the loop at any size up to \
             {}, so either the per-item cost is too small or the sweep is too short",
            SIZES.last().copied().unwrap_or(0)
        ),
    }
}

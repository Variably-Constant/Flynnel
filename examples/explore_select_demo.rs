//! Episode racing with `explore_select`: N independent explorers each
//! run a full trajectory to completion, and the FEWEST-ACTIONS one is
//! kept - regardless of which finished first.
//!
//! The winning explorer is deliberately made the SLOWEST to finish
//! (it sleeps longest), so this demonstrates the semantic that
//! distinguishes `explore_select` from `race_variants`: a
//! first-past-the-post race would cancel and discard the best-scoring
//! explorer here. `explore_select` keeps it.
//!
//! Run with:
//!   cargo run --release --example explore_select_demo

use std::time::{Duration, Instant};

use flynnel::{JobPlan, explore_select};

/// Tiny reproducible RNG (SplitMix64) so each clone's trajectory is
/// deterministic in its seed - episode racing must be reproducible.
struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One explorer's outcome.
#[derive(Debug, Clone, Copy)]
struct Episode {
    clone_id: usize,
    actions: u32,
    reached: bool,
}

/// Run clone `i`'s episode: a noisy walk to a goal on a line. Each
/// clone has its own RNG stream, so action counts vary. The
/// fewest-actions clone found the best roll-out.
fn run_episode(i: usize) -> Episode {
    let mut rng = SplitMix64::new(0x1234_5678 ^ (i as u64).wrapping_mul(0x0010_0000_01B3));
    let goal = 60i32;
    let mut pos = 0i32;
    let mut actions = 0u32;
    while pos < goal && actions < 100_000 {
        actions += 1;
        if rng.next_f64() < 0.70 {
            pos += 1; // toward the goal
        } else {
            pos -= 1; // noise
        }
        if pos < 0 {
            pos = 0;
        }
    }
    // Make the BEST (fewest-actions) explorer the SLOWEST to finish:
    // sleep inversely to action count. First-past-the-post would
    // therefore discard exactly the explorer we want to keep.
    let sleep_us = (240_000u64 / actions.max(1) as u64).min(20_000);
    std::thread::sleep(Duration::from_micros(sleep_us));
    Episode { clone_id: i, actions, reached: pos >= goal }
}

fn main() {
    println!("=== Episode racing: explore_select (keep fewest actions) ===\n");
    let n = 32usize;
    let plan = JobPlan::new(6, n as u32);

    // The primitive: all N explorers run to completion; comparator
    // picks fewest actions.
    let t0 = Instant::now();
    let (idx, best) =
        explore_select(&plan, n, run_episode, |a, b| a.actions < b.actions).expect("n > 0");
    let parallel_ms = t0.elapsed().as_secs_f64() * 1e3;

    // Ground truth: brute-force serial argmin over the same episodes.
    let all: Vec<Episode> = (0..n).map(run_episode).collect();
    let truth = all.iter().min_by_key(|e| e.actions).expect("nonempty");
    let worst = all.iter().max_by_key(|e| e.actions).expect("nonempty");

    println!("[1] {n} explorers, all run to completion");
    println!("    winner: clone {} with {} actions (reached goal: {})",
             best.clone_id, best.actions, best.reached);
    println!("    brute-force argmin: clone {} with {} actions",
             truth.clone_id, truth.actions);
    assert_eq!(best.actions, truth.actions, "must select the global fewest-actions episode");
    assert_eq!(idx, truth.clone_id, "index must point at the winner");
    println!("    -> MATCHES brute force: VERIFIED\n");

    // The distinguishing property: the winner slept the LONGEST (it
    // had the fewest actions), so it finished last. A first-past-the-
    // post race would have cancelled it in favor of clone {worst}.
    let winner_sleep = 240_000u64 / best.actions.max(1) as u64;
    let worst_sleep = 240_000u64 / worst.actions.max(1) as u64;
    println!("[2] semantic contrast:");
    println!("    winner (clone {}) sleeps ~{} us -> finishes LAST",
             best.clone_id, winner_sleep.min(20_000));
    println!("    fastest-finisher (clone {}, {} actions) sleeps ~{} us",
             worst.clone_id, worst.actions, worst_sleep.min(20_000));
    println!("    race_variants would have kept the fast-but-worse clone;");
    println!("    explore_select kept the slow-but-best one.\n");

    // Reproducibility: same explorers, same winner, every run.
    let (idx2, best2) =
        explore_select(&plan, n, run_episode, |a, b| a.actions < b.actions).expect("n > 0");
    assert_eq!((idx2, best2.actions), (idx, best.actions), "deterministic winner");
    println!("[3] re-ran the race: same winner (clone {idx2}, {} actions): VERIFIED",
             best2.actions);

    let serial_ms: f64 = all.iter()
        .map(|e| (240_000u64 / e.actions.max(1) as u64).min(20_000) as f64 / 1e3)
        .sum();
    println!("\n[4] wall-clock: parallel {parallel_ms:.1} ms vs serial sleep budget {serial_ms:.1} ms");
    println!("\nVERIFIED: MIMD explore-all + argmin select - the best explorer is");
    println!("kept even though it finished last.");
}

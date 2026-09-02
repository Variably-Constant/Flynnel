//! The racing zoo: one runnable tour of every racing primitive, each
//! section self-verifying against a known-correct answer.
//!
//! Run with:
//!   cargo run --release --example racing_zoo_demo

use std::time::Duration;

use flynnel::{
    Agreement, JobPlan, Settled, StatOpts, explore_select, race_agree, race_any, race_deadline,
    race_quorum, race_refute, race_statistical, race_tournament, race_variants,
};

/// Small reproducible RNG so the statistical section is honestly noisy
/// yet deterministic across runs.
struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() {
    println!("=== The racing zoo: nine ways to race ===\n");
    let plan = JobPlan::new(6, 32);

    // --- 0. race_variants (first tolerable wins, MISD) --------------
    // Fast meets tolerance instantly; the correct-tier safety net is
    // slow, so the fast tolerable answer wins and correct is cancelled.
    let (r, v) = race_variants::<u32, _, _, _>(
        &plan,
        |_t| Some(1u32),
        |_t| None,
        |t| {
            for _ in 0..200 {
                if t.is_cancelled() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            2u32
        },
    );
    println!("[race_variants] first tolerable = {r} (tier {v:?})");
    assert_eq!(r, 1);

    // --- 1. explore_select (all finish, best by score, MIMD) --------
    let counts = [7u32, 3, 9, 1, 8, 4, 6, 2, 5, 10];
    let (idx, best) =
        explore_select(&plan, counts.len(), |i| counts[i], |a, b| a < b).expect("n>0");
    println!("[explore_select] fewest actions = {best} at index {idx}");
    assert_eq!((idx, best), (3, 1));

    // --- 2. race_any (hedged, first to finish wins) -----------------
    let (win, _p) = race_any(&plan, 8, |i, token| {
        if i == 0 {
            return 0u32; // the instant replica
        }
        for _ in 0..100 {
            if token.is_cancelled() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        i as u32
    })
    .expect("n>0");
    println!("[race_any] fastest replica = {win}");
    assert_eq!(win, 0);

    // --- 3. race_quorum (first k of n) ------------------------------
    let winners = race_quorum(&plan, 8, 3, |i, _t| {
        std::thread::sleep(Duration::from_millis((i as u64) * 3));
        i
    });
    let mut ids: Vec<usize> = winners.iter().map(|(i, _)| *i).collect();
    ids.sort_unstable();
    println!("[race_quorum] first 3 of 8 to answer = {ids:?}");
    assert_eq!(winners.len(), 3);
    assert_eq!(ids, vec![0, 1, 2], "the three fastest replicas");

    // --- 4. race_refute (prover vs refuter, portfolio) --------------
    // Toy: is n composite? prover finds a factor, refuter certifies
    // primality by trial. n = 91 = 7 x 13.
    let n = 91u64;
    let verdict = race_refute::<(u64, u64), (), _, _>(
        &plan,
        |_t| (2..n).find(|d| n.is_multiple_of(*d)).map(|d| (d, n / d)),
        |t| {
            // Only certify prime after a FULL clean scan. Cancelled or
            // factor-found means no verdict from this side.
            for d in 2..n {
                if t.is_cancelled() || n.is_multiple_of(d) {
                    return None;
                }
            }
            Some(())
        },
    );
    match verdict {
        Settled::Proved((a, b)) => println!("[race_refute] {n} is composite: {a} x {b}"),
        Settled::Refuted(()) => println!("[race_refute] {n} is prime"),
        Settled::Unsettled => println!("[race_refute] unsettled"),
    }
    assert!(matches!(verdict, Settled::Proved((7, 13)) | Settled::Proved((13, 7))));

    // --- 5. race_agree (consensus verification) ---------------------
    // Three "implementations" of the same function; one has a bug.
    let good = |x: u64| x * x;
    let buggy = |x: u64| x * x + 1; // silent divergence
    let out = race_agree(&plan, 3, 3, |i| match i {
        2 => buggy(9),
        _ => good(9),
    });
    match out {
        Agreement::Consensus { value, agree, total } => {
            println!("[race_agree] consensus {value} ({agree}/{total})")
        }
        Agreement::Split { plurality, total } => {
            println!("[race_agree] DISAGREEMENT caught: best bloc {plurality}/{total}")
        }
    }
    assert_eq!(out, Agreement::Split { plurality: 2, total: 3 }, "the bug is caught");

    // --- 6. race_deadline (anytime, best within a budget) -----------
    // Each explorer polishes toward its ceiling; take the best at the
    // deadline. Explorer i can reach score i.
    let out = race_deadline(&plan, Duration::from_millis(50), 8, |i, ctx| {
        let mut score = 0.0;
        while !ctx.is_expired() {
            if score < i as f64 {
                score += 0.5;
                ctx.submit(score, i as u32);
            }
            std::hint::spin_loop();
        }
    });
    let (score, who) = out.expect("someone published");
    println!("[race_deadline] best-at-deadline = clone {who} scored {score}");
    assert_eq!(who, 7);

    // --- 7. race_tournament (successive halving) --------------------
    // Candidate quality rises with id; low budget is noisy, high
    // budget reveals the truth. Winner must be the best candidate.
    let (tid, tscore) = race_tournament(
        &plan,
        16,
        2,
        4,
        |id, budget| {
            // "Train" for `budget` steps; score converges to id.
            let mut s = 0.0f64;
            for _ in 0..budget {
                s += id as f64;
            }
            s / budget as f64
        },
        |a, b| a > b,
    )
    .expect("n>0");
    println!("[race_tournament] survivor = candidate {tid} (score {tscore})");
    assert_eq!(tid, 15);

    // --- 8. race_statistical (Hoeffding races) ----------------------
    // Five noisy arms with true means 0,10,20,30,40. The race must
    // pick arm 4 while cutting the rest early, under real noise.
    let opts = StatOpts {
        value_range: 60.0,
        delta: 0.05,
        batch: 48,
        max_samples: 8192,
        maximize: true,
    };
    // A per-arm draw counter makes every sample a distinct, genuinely
    // noisy observation (not a re-seeded constant), while staying
    // reproducible in aggregate.
    let draws: Vec<std::sync::atomic::AtomicU64> =
        (0..5).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    let stat = race_statistical(&plan, 5, opts, |id| {
        let d = draws[id].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut rng = SplitMix64::new((id as u64).wrapping_mul(0x9E37).wrapping_add(d ^ 0xABCD));
        let noise = (rng.next_f64() - 0.5) * 20.0; // uniform +-10
        (id as f64) * 10.0 + noise
    })
    .expect("n>0");
    println!(
        "[race_statistical] winner arm {} (mean {:.1}) after {} samples each, {} survivor(s)",
        stat.winner, stat.mean, stat.samples_each, stat.survivors
    );
    assert_eq!(stat.winner, 4, "highest true mean wins under noise");

    println!("\nVERIFIED: all nine racing primitives produced the correct winner.");
}

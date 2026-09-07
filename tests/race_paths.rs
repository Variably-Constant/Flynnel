//! The racing entries, including what each does with nothing to race.
//!
//! Racing is non-deterministic by construction, so these assert the
//! contracts rather than a particular winner: that the value handed
//! back is the one the reported variant produced, that a quorum is the
//! size it promised, that every explorer's vote was counted, and that
//! an empty race answers rather than panicking. Where a test needs a
//! deterministic winner it removes the race instead of hoping, by
//! making every other arm decline.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use flynnel::{
    Agreement, Anytime, CancelToken, JobPlan, Settled, StatOpts, Variant, explore_select,
    race_agree, race_any, race_deadline, race_quorum, race_refute, race_statistical,
    race_tournament, race_variants,
};

fn plan(n: usize) -> JobPlan {
    JobPlan::new(6, n as u32)
}

#[test]
fn race_variants_labels_the_answer_with_the_arm_that_produced_it() {
    // Three distinguishable answers: whichever wins, the tag and the
    // value must agree. A mislabelled winner is a silently wrong
    // precision claim.
    let (value, variant) = race_variants(&plan(3), |_| Some(1u32), |_| Some(2u32), |_| 3u32);
    let expected = match variant {
        Variant::Fast => 1,
        Variant::Faithful => 2,
        Variant::Correct => 3,
    };
    assert_eq!(value, expected, "the value must be the one {variant:?} computed");
}

#[test]
fn race_variants_falls_through_to_correct_when_no_tier_is_tolerable() {
    // Both tolerable tiers decline, so there is no race left and the
    // answer is determined.
    let (value, variant) = race_variants(&plan(3), |_| None::<u32>, |_| None::<u32>, |_| 42u32);
    assert_eq!(value, 42);
    assert_eq!(variant, Variant::Correct, "only the correct tier answered");
}

#[test]
fn race_variants_runs_the_correct_tier_to_completion_even_when_it_loses() {
    // The safety net is not cancellable: it must finish whatever else
    // wins, or a later caller relying on it has nothing.
    let correct_finished = AtomicUsize::new(0);
    let (value, variant) = race_variants(
        &plan(3),
        |_| Some(1u32),
        |_| Some(2u32),
        |_| {
            correct_finished.fetch_add(1, Ordering::Relaxed);
            3u32
        },
    );
    assert_eq!(
        correct_finished.load(Ordering::Relaxed),
        1,
        "the correct tier ran to completion regardless of who won"
    );
    let expected = match variant {
        Variant::Fast => 1,
        Variant::Faithful => 2,
        Variant::Correct => 3,
    };
    assert_eq!(value, expected);
}

#[test]
fn explore_select_keeps_the_best_and_every_explorer_finishes() {
    let n = 8usize;
    let ran = AtomicUsize::new(0);
    let got = explore_select(
        &plan(n),
        n,
        |i| {
            ran.fetch_add(1, Ordering::Relaxed);
            i
        },
        |a, b| a < b,
    );
    assert_eq!(got, Some((0, 0)), "the smallest wins under this ordering");
    assert_eq!(ran.load(Ordering::Relaxed), n, "no explorer was cancelled");
    assert_eq!(explore_select(&plan(0), 0, |i| i, |a, b| a < b), None, "nothing to select");
}

#[test]
fn race_any_returns_a_real_attempt_and_nothing_for_an_empty_race() {
    let n = 8usize;
    let got = race_any(&plan(n), n, |i, _t| i * 10);
    let (idx, value) = got.expect("some attempt finished first");
    assert!(idx < n, "the winning index is one of the attempts");
    assert_eq!(value, idx * 10, "and the value is that attempt's own result");
    assert_eq!(race_any(&plan(0), 0, |i, _t| i), None, "an empty race has no winner");
}

#[test]
fn race_any_cancels_the_losers() {
    // The point of hedging: the slow attempts stop once a peer wins.
    let n = 4usize;
    let saw_cancel = AtomicUsize::new(0);
    let winner = race_any(&plan(n), n, |i, token| {
        if i > 0 {
            // Poll for a bounded time rather than forever, so a failure
            // to cancel fails the assertion instead of hanging.
            for _ in 0..2000 {
                if token.is_cancelled() {
                    saw_cancel.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(Duration::from_micros(100));
            }
        }
        i
    });
    assert!(winner.is_some(), "the race produced a winner");
    assert!(
        saw_cancel.load(Ordering::Relaxed) > 0,
        "at least one loser observed the cancel a winner fired"
    );
}

#[test]
fn race_quorum_returns_exactly_the_quorum_it_was_asked_for() {
    let n = 8usize;
    let got = race_quorum(&plan(n), n, 3, |i, _t| i * 2);
    assert_eq!(got.len(), 3, "a quorum of three is three results");
    let mut idxs: Vec<usize> = got.iter().map(|&(i, _)| i).collect();
    idxs.sort_unstable();
    idxs.dedup();
    assert_eq!(idxs.len(), 3, "three distinct attempts, not one counted thrice");
    for (i, v) in got {
        assert_eq!(v, i * 2, "each result belongs to its own attempt");
    }
}

#[test]
fn race_quorum_clamps_a_quorum_larger_than_the_field_and_answers_an_empty_one() {
    let n = 4usize;
    let all = race_quorum(&plan(n), n, 99, |i, _t| i);
    assert_eq!(all.len(), n, "a quorum past the field size is the whole field");
    assert!(race_quorum(&plan(n), n, 0, |i, _t| i).is_empty(), "a quorum of none is none");
}

#[test]
fn race_refute_reports_which_side_settled_it() {
    let proved: Settled<u32, &str> = race_refute(&plan(2), |_| Some(7u32), |_| None);
    assert_eq!(proved, Settled::Proved(7));

    let refuted: Settled<u32, &str> = race_refute(&plan(2), |_| None, |_| Some("counterexample"));
    assert_eq!(refuted, Settled::Refuted("counterexample"));

    let neither: Settled<u32, &str> = race_refute(&plan(2), |_| None, |_| None);
    assert_eq!(neither, Settled::Unsettled, "neither side settling is its own verdict");
}

#[test]
fn race_agree_reaches_consensus_only_when_the_votes_line_up() {
    let n = 5usize;
    match race_agree(&plan(n), n, n, |_| 99u32) {
        Agreement::Consensus { value, agree, total } => {
            assert_eq!(value, 99);
            assert_eq!(agree, n, "every explorer agreed");
            assert_eq!(total, n);
        }
        other => panic!("unanimous explorers must reach consensus, got {other:?}"),
    }

    // Every explorer disagrees, so the largest bloc is one.
    match race_agree(&plan(n), n, 2, |i| i as u32) {
        Agreement::Split { plurality, total } => {
            assert_eq!(plurality, 1, "no two explorers agreed");
            assert_eq!(total, n);
        }
        other => panic!("total disagreement must be reported as a split, got {other:?}"),
    }
}

#[test]
fn race_agree_counts_a_majority_that_clears_the_threshold() {
    // Three of five agree on 1; a threshold of three is met, four is
    // not, and the same votes decide both.
    let n = 5usize;
    let vote = |i: usize| if i < 3 { 1u32 } else { i as u32 };
    match race_agree(&plan(n), n, 3, vote) {
        Agreement::Consensus { value, agree, total } => {
            assert_eq!(value, 1);
            assert_eq!(agree, 3);
            assert_eq!(total, n);
        }
        other => panic!("a bloc of three must clear a threshold of three, got {other:?}"),
    }
    match race_agree(&plan(n), n, 4, vote) {
        Agreement::Split { plurality, total } => {
            assert_eq!(plurality, 3, "the largest bloc is still reported");
            assert_eq!(total, n);
        }
        other => panic!("a bloc of three must not clear a threshold of four, got {other:?}"),
    }
}

#[test]
fn race_agree_answers_an_empty_field() {
    match race_agree(&plan(0), 0, 1, |i| i as u32) {
        Agreement::Split { plurality, total } => {
            assert_eq!(plurality, 0);
            assert_eq!(total, 0);
        }
        other => panic!("an empty field is a split of nothing, got {other:?}"),
    }
}

#[test]
fn race_deadline_returns_the_best_published_and_nothing_when_none_was() {
    let n = 4usize;
    let got = race_deadline(&plan(n), Duration::from_millis(60), n, |i, any| {
        // Publish once, then idle out the budget.
        any.submit(i as f64, i);
        while !any.is_expired() {
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    let (score, value) = got.expect("someone published");
    assert_eq!(value, n - 1, "the highest score wins");
    assert_eq!(score, (n - 1) as f64);

    let nobody =
        race_deadline(&plan(n), Duration::from_millis(30), n, |_, any: &Anytime<u32>| {
            while !any.is_expired() {
                std::thread::sleep(Duration::from_millis(1));
            }
        });
    assert!(nobody.is_none(), "a race where nobody published has no result");

    let empty = race_deadline(&plan(0), Duration::from_millis(10), 0, |_, _: &Anytime<u32>| {});
    assert!(empty.is_none(), "an empty field has no result");
}

#[test]
fn race_deadline_keeps_the_incumbent_on_a_tie() {
    let got = race_deadline(&plan(1), Duration::from_millis(40), 1, |_, any| {
        any.submit(1.0, "first");
        any.submit(1.0, "second");
        while !any.is_expired() {
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    assert_eq!(got.map(|(_, v)| v), Some("first"), "an equal score does not displace");
}

#[test]
fn race_tournament_keeps_promoting_until_one_candidate_is_left() {
    let n = 16usize;
    let got = race_tournament(&plan(n), n, 2, 1, |id, budget| id as u32 * budget, |a, b| a > b);
    let (idx, score) = got.expect("a tournament with candidates has a winner");
    assert_eq!(idx, n - 1, "the candidate that scores highest at every budget wins");
    assert!(score > 0, "the winner carries the score from its final round");
    assert!(
        race_tournament(&plan(0), 0, 2, 1, |id, _b| id, |a, b| a > b).is_none(),
        "an empty tournament has no winner"
    );
}

#[test]
fn race_tournament_treats_an_eta_below_two_as_two() {
    // eta 0 would keep every candidate forever; the documented clamp is
    // what stops the loop.
    let n = 8usize;
    let got = race_tournament(&plan(n), n, 0, 1, |id, _b| id as u32, |a, b| a > b);
    assert_eq!(got.map(|(i, _)| i), Some(n - 1), "the race terminated and picked the best");
}

#[test]
fn race_statistical_finds_the_dominant_candidate_and_answers_an_empty_field() {
    let n = 4usize;
    let opts = StatOpts {
        value_range: 1.0,
        delta: 0.05,
        batch: 64,
        max_samples: 4096,
        maximize: true,
    };
    // Candidate 3 is separated far enough that the bound resolves it.
    let outcome = race_statistical(&plan(n), n, opts, |id| if id == 3 { 1.0 } else { 0.0 })
        .expect("a field with candidates has an outcome");
    assert_eq!(outcome.winner, 3, "the dominant candidate wins");
    assert!(outcome.mean > 0.5, "and its observed mean reflects its samples");
    assert!(outcome.samples_each > 0, "samples were actually drawn");

    assert!(
        race_statistical(&plan(0), 0, opts, |_| 0.0).is_none(),
        "an empty field has no outcome"
    );
}

#[test]
fn race_statistical_minimises_when_asked_to() {
    let n = 4usize;
    let opts = StatOpts {
        value_range: 1.0,
        delta: 0.05,
        batch: 64,
        max_samples: 4096,
        maximize: false,
    };
    let outcome =
        race_statistical(&plan(n), n, opts, |id| if id == 1 { 0.0 } else { 1.0 }).expect("outcome");
    assert_eq!(outcome.winner, 1, "the lowest mean wins when maximize is false");
}

#[test]
fn a_cancel_token_composed_by_hand_carries_the_signal() {
    // The documented use for a caller building its own race on join.
    let token = CancelToken::new();
    assert!(!token.is_cancelled(), "a fresh token is not cancelled");
    let clone = token.clone();
    token.cancel();
    assert!(clone.is_cancelled(), "a clone observes the cancel");
    token.cancel();
    assert!(clone.is_cancelled(), "and cancelling twice changes nothing");
}

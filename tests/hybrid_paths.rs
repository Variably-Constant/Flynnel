//! The hybrid entries: two implementations of one computation, and a
//! boundary between them that is a performance decision only.
//!
//! The split entries are where a fault would be silent. If the CPU and
//! backend sub-ranges overlapped, items would be processed twice; if
//! they left a gap, items would be skipped and the answer would simply
//! be missing part of its input. Neither shows up as an error, so
//! these check the tiling directly rather than checking that the call
//! returned.
//!
//! No accelerator backend is registered here, so the backend side
//! resolves to the CPU fallback. That is the configuration every
//! consumer runs in until it registers one, and the contracts under
//! test are about division and ordering rather than about hardware.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use flynnel::{
    JobPlan, Placement, hybrid_auto, hybrid_auto_split, hybrid_auto_split_ranges, hybrid_pipeline,
    join_hybrid,
};

#[test]
fn join_hybrid_returns_both_halves_in_order() {
    let plan = JobPlan::new(8, 1024);
    let (cpu, gpu) = join_hybrid(&plan, || (0..512u64).sum::<u64>(), || (512..1024u64).sum::<u64>());
    assert_eq!(cpu, (0..512u64).sum::<u64>(), "the CPU half comes back first");
    assert_eq!(gpu, (512..1024u64).sum::<u64>());
    assert_eq!(cpu + gpu, (0..1024u64).sum::<u64>(), "and together they are the whole");
}

#[test]
fn join_hybrid_propagates_a_panic_from_either_half() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let plan = JobPlan::new(8, 1024);

    let cpu_side = std::panic::catch_unwind(|| {
        join_hybrid(&plan, || panic!("cpu half"), || 1u32);
    });
    assert!(cpu_side.is_err(), "a panic in the CPU half reaches the caller");

    let backend_side = std::panic::catch_unwind(|| {
        join_hybrid(&plan, || 1u32, || panic!("backend half"));
    });
    assert!(
        backend_side.is_err(),
        "and a panic on the backend side is re-raised rather than swallowed"
    );

    std::panic::set_hook(previous);
}

#[test]
fn hybrid_auto_answers_the_same_whichever_side_it_placed() {
    // Both implementations must compute the same thing, so the answer
    // is fixed and only the placement varies.
    let plan = JobPlan::new(8, 4096);
    for _ in 0..32 {
        let (value, placement) = hybrid_auto(&plan, || 6u64 * 7, || 6u64 * 7);
        assert_eq!(value, 42, "the answer does not depend on the placement");
        assert!(
            matches!(placement, Placement::Cpu | Placement::Backend | Placement::Race),
            "the placement is one of the three documented decisions"
        );
    }
}

#[test]
fn hybrid_auto_stops_racing_once_it_has_learned() {
    // Racing is the exploration mode. A site that never leaves it is
    // paying double for every call forever.
    let plan = JobPlan::new(8, 4096);
    let mut settled = 0usize;
    for _ in 0..64 {
        let (_value, placement) = hybrid_auto(&plan, || 1u64, || 1u64);
        if placement != Placement::Race {
            settled += 1;
        }
    }
    assert!(
        settled > 0,
        "after 64 calls at one site the model should have committed at least once"
    );
}

#[test]
fn hybrid_auto_split_touches_every_item_exactly_once() {
    // The failure this guards is silent: an overlap double-applies the
    // transformation, a gap leaves items untouched, and both return
    // normally.
    let n = 4096usize;
    let plan = JobPlan::new(8, n as u32);
    for _ in 0..8 {
        let mut items: Vec<u64> = vec![0; n];
        let report = hybrid_auto_split(
            &plan,
            &mut items,
            |c| {
                for x in c {
                    *x += 1;
                }
            },
            |c| {
                for x in c {
                    *x += 1;
                }
            },
        );
        assert!(
            items.iter().all(|&x| x == 1),
            "every item incremented exactly once, not zero or twice"
        );
        assert_eq!(
            report.cpu_items + report.backend_items,
            n,
            "the two sides account for the whole slice"
        );
        assert!(
            report.cpu_share_per_mille <= 1000,
            "a share is parts per thousand, got {}",
            report.cpu_share_per_mille
        );
    }
}

#[test]
fn hybrid_auto_split_handles_the_sizes_a_split_can_degenerate_on() {
    let plan = JobPlan::new(8, 4);
    for n in [0usize, 1, 2, 3] {
        let mut items: Vec<u64> = vec![0; n];
        let report = hybrid_auto_split(
            &plan,
            &mut items,
            |c| {
                for x in c {
                    *x += 1;
                }
            },
            |c| {
                for x in c {
                    *x += 1;
                }
            },
        );
        assert!(items.iter().all(|&x| x == 1), "every item touched once at n = {n}");
        assert_eq!(report.cpu_items + report.backend_items, n, "accounted for at n = {n}");
    }
}

#[test]
fn hybrid_auto_split_ranges_tiles_the_index_space_exactly() {
    let n = 1000usize;
    let plan = JobPlan::new(8, n as u32);
    let marks: Arc<Vec<AtomicUsize>> = Arc::new((0..n).map(|_| AtomicUsize::new(0)).collect());

    let cpu_marks = Arc::clone(&marks);
    let gpu_marks = Arc::clone(&marks);
    let report = hybrid_auto_split_ranges(
        &plan,
        n,
        move |r| {
            for i in r {
                cpu_marks[i].fetch_add(1, Ordering::Relaxed);
            }
        },
        move |r| {
            for i in r {
                gpu_marks[i].fetch_add(1, Ordering::Relaxed);
            }
        },
    );

    for (i, m) in marks.iter().enumerate() {
        assert_eq!(
            m.load(Ordering::Relaxed),
            1,
            "index {i} was visited once; the two ranges neither overlap nor leave a gap"
        );
    }
    assert_eq!(report.cpu_items + report.backend_items, n);
}

#[test]
fn hybrid_pipeline_keeps_input_order_through_three_stages() {
    // The stages run concurrently, so a pipeline that returned
    // completion order would reorder a caller's results silently.
    let plan = JobPlan::new(8, 16);
    let results: Vec<i64> = hybrid_pipeline(
        &plan,
        0..16i64,
        |seed: i64| seed * 2,
        |doubled: i64| doubled + 1,
        |t: i64| t * 10,
    );
    let expect: Vec<i64> = (0..16i64).map(|s| (s * 2 + 1) * 10).collect();
    assert_eq!(results, expect, "results follow input order, not completion order");
}

#[test]
fn hybrid_pipeline_handles_an_empty_input() {
    let plan = JobPlan::new(8, 1);
    let results: Vec<i64> = hybrid_pipeline(&plan, 0..0i64, |s: i64| s, |a: i64| a, |b: i64| b);
    assert!(results.is_empty(), "no inputs is no outputs, not a hang");
}

#[test]
fn hybrid_pipeline_runs_each_stage_once_per_item() {
    let n = 32usize;
    let plan = JobPlan::new(8, n as u32);
    let pre = Arc::new(AtomicUsize::new(0));
    let mid = Arc::new(AtomicUsize::new(0));
    let post = Arc::new(AtomicUsize::new(0));
    let (p, m, q) = (Arc::clone(&pre), Arc::clone(&mid), Arc::clone(&post));

    let results: Vec<usize> = hybrid_pipeline(
        &plan,
        0..n,
        move |i: usize| {
            p.fetch_add(1, Ordering::Relaxed);
            i
        },
        move |i: usize| {
            m.fetch_add(1, Ordering::Relaxed);
            i
        },
        move |i: usize| {
            q.fetch_add(1, Ordering::Relaxed);
            i
        },
    );

    assert_eq!(results, (0..n).collect::<Vec<_>>());
    assert_eq!(pre.load(Ordering::Relaxed), n, "the first stage ran once per item");
    assert_eq!(mid.load(Ordering::Relaxed), n, "and so did the second");
    assert_eq!(post.load(Ordering::Relaxed), n, "and the third");
}

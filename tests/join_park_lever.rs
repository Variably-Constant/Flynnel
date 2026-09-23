//! The two park levers, `FLYNNEL_LEVER_JOIN_PARK` and
//! `FLYNNEL_LEVER_SLOT_PARK_NOW`, both on: dispatches from outside the
//! pool compute the same answers, and a join whose stolen half outlasts
//! the waiter's spin budget parks that waiter in the kernel.
//!
//! A test binary of its own, because each lever is read once per process.

use std::time::{Duration, Instant};

use flynnel::JobPlan;

/// Spins for about `d`, answering how many rounds that took.
fn busy_for(d: Duration) -> u64 {
    let t0 = Instant::now();
    let mut n = 0u64;
    while t0.elapsed() < d {
        n = n.wrapping_add(1);
        std::hint::black_box(n);
    }
    n
}

#[test]
fn with_both_parks_on_joins_stay_correct_and_a_long_half_parks_its_waiter() {
    // SAFETY: set before this process's first dispatch, which is what
    // resolves both switches, while no other thread of the test runs.
    unsafe {
        std::env::set_var("FLYNNEL_LEVER_JOIN_PARK", "1");
        std::env::set_var("FLYNNEL_LEVER_SLOT_PARK_NOW", "1");
    }
    let workers = flynnel::sched::arena::global_local_arena().local_worker_count();
    let plan = JobPlan::new(10, 1 << 20);

    // A short left half and a long right half: the worker that forks
    // them finishes the left and then waits on the right, which a woken
    // peer has stolen, far past any spin budget.
    let before = flynnel::total_join_parks();
    for _ in 0..20 {
        let halves = flynnel::join(
            &plan,
            || {
                busy_for(Duration::from_micros(200));
                1u32
            },
            || {
                busy_for(Duration::from_millis(20));
                2u32
            },
        );
        assert_eq!(
            halves,
            (1, 2),
            "a join returned its halves out of order or wrong"
        );
    }
    let parks = flynnel::total_join_parks() - before;
    println!("JOIN_PARKS {parks} over 20 joins on {workers} workers");

    let mut data = vec![0u64; 1 << 16];
    for round in 1..=50u64 {
        flynnel::for_each_chunk(&plan, &mut data, |s| s.iter_mut().for_each(|x| *x += 1));
        assert!(
            data.iter().all(|&x| x == round),
            "round {round}: an element was skipped or run twice"
        );
    }

    if workers < 2 {
        println!(
            "JOIN_PARK_UNREACHED the pool has {workers} worker, so no half is stolen and no join waits"
        );
        return;
    }
    assert!(
        parks > 0,
        "twenty joins whose stolen half runs 20 ms parked no waiter across {workers} workers"
    );
}

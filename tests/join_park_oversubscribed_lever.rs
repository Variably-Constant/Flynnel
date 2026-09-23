//! `FLYNNEL_LEVER_JOIN_PARK_OVERSUBSCRIBED` on, with the unconditional
//! `FLYNNEL_LEVER_JOIN_PARK` held off: in a process holding more
//! runnable threads than cores, a yield comes back late, the process
//! reads that as oversubscription, and a join waiter whose stolen half
//! outlasts its spin budget parks in the kernel. The joins answer as
//! they would without the switch.
//!
//! What a quiet host does with the switch on is not asserted here. A
//! quiet run parks nothing only while nothing else on the host is
//! runnable, and a test cannot know that of the machine it runs on.
//!
//! A test binary of its own, because each switch is read once per
//! process.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
fn an_oversubscribed_process_parks_its_join_waiters_and_the_joins_stay_correct() {
    // SAFETY: set before this process's first dispatch, which is what
    // resolves the switches, while no other thread of the test runs.
    unsafe {
        std::env::set_var("FLYNNEL_LEVER_JOIN_PARK_OVERSUBSCRIBED", "1");
        std::env::set_var("FLYNNEL_LEVER_JOIN_PARK", "0");
    }
    let workers = flynnel::sched::arena::global_local_arena().local_worker_count();
    let plan = JobPlan::new(10, 1 << 20);
    assert!(
        flynnel::sched::levers::join_park_oversubscribed(),
        "the switch was set and reads off"
    );
    assert!(!flynnel::sched::levers::join_park(), "the unconditional park reads on");

    // Twice as many spinners as the host has logical processors, so a
    // worker that yields hands its core to one of them.
    let cores = std::thread::available_parallelism()
        .expect("the host reports how many logical processors it has")
        .get();
    let stop = Arc::new(AtomicBool::new(false));
    let spinners: Vec<_> = (0..cores * 2)
        .map(|_| {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut n = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    n = n.wrapping_add(1);
                    std::hint::black_box(n);
                }
            })
        })
        .collect();

    let parks_before = flynnel::total_join_parks();
    let long_before = flynnel::total_long_yields();
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
        assert_eq!(halves, (1, 2), "a join returned its halves out of order or wrong");
    }
    let mut data = vec![0u64; 1 << 16];
    for round in 1..=50u64 {
        flynnel::for_each_chunk(&plan, &mut data, |s| s.iter_mut().for_each(|x| *x += 1));
        assert!(
            data.iter().all(|&x| x == round),
            "round {round}: an element was skipped or run twice"
        );
    }

    stop.store(true, Ordering::Relaxed);
    for s in spinners {
        s.join().expect("a spinner thread panicked");
    }
    let parks = flynnel::total_join_parks() - parks_before;
    let long = flynnel::total_long_yields() - long_before;
    println!(
        "JOIN_PARKS {parks} LONG_YIELDS {long} over 20 joins on {workers} workers beside {} spinners",
        cores * 2
    );

    if workers < 2 {
        println!(
            "JOIN_PARK_UNREACHED the pool has {workers} worker, so no half is stolen and no join waits"
        );
        return;
    }
    assert!(
        long > 0,
        "{} spinners on {cores} logical processors, and no yield in the process came back late",
        cores * 2
    );
    assert!(
        parks > 0,
        "the process read {long} long yields and parked no join waiter across twenty joins"
    );
}

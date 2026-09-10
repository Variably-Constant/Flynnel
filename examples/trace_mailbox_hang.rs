//! One traced mailbox fan-out, with the closures reporting which of
//! them ran, so a fan-out that does not complete says which closures
//! were never executed rather than only that it stopped.
//!
//! Run with the trace switch on:
//!
//! ```text
//! cargo run --release --example trace_mailbox_hang -- 8 rayon 2> hang.csv
//! cargo run --release --example trace_mailbox_hang -- 8 alone 2> alone.csv
//! ```
//!
//! Arguments: how far above the local worker count to fan out
//! (default 8), and whether a `rayon` scope fan-out runs first or the
//! mailbox fan-out runs `alone`. The two arms differ in nothing else,
//! so a completion in one and a stall in the other attributes the
//! difference to the foreign pool.
//!
//! # What the rows say
//!
//! The caller emits 5 (join push) and 6 (join wait begin) with the
//! closure count as payload, and 7 (join wait end) when the fan-out
//! completes. Push and wait-begin without a wait-end is a fan-out that
//! never finished.
//!
//! Each closure emits 3 on entry and 4 on exit, with its own index as
//! the payload, on whichever thread ran it. Indices present in no
//! thread's rows are the closures that never executed - which is the
//! difference between a fan-out that is slow and one that is waiting
//! on work nothing will run.
//!
//! # Why the wake loop after a stall
//!
//! A worker dumps its buffer at the top of its next loop pass, and a
//! parked worker reaches that only when woken. After a stall the
//! workers holding the answer are exactly the ones that are not
//! running, so the dump is preceded by wide dispatches to wake them.

use std::str::FromStr;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use flynnel::sched::arena::global_local_arena;
use flynnel::sched::cooperative::cooperative_join_n_flat_mailbox;
use flynnel::sched::trace::{self, TraceEvent};
use flynnel::{JobPlan, for_each_chunk};

/// How long to let the fan-out run before calling it stalled.
const STALL_AFTER: Duration = Duration::from_secs(20);

/// How long to keep waking workers so they can dump.
const DUMP_WINDOW: Duration = Duration::from_secs(30);

/// The `index`th argument parsed as `T`, `default` when absent.
fn argument<T: FromStr>(args: &[String], index: usize, default: T, name: &str) -> T
where
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        None => default,
        Some(text) => match text.parse() {
            Ok(value) => value,
            Err(e) => {
                eprintln!("argument {name} is {text:?}: {e}");
                std::process::exit(2);
            }
        },
    }
}

/// Wake every worker repeatedly so each reaches the top of its loop and
/// dumps, returning how many dumps completed.
fn drain_worker_dumps(expected: u64) -> u64 {
    trace::request_worker_flush();
    let deadline = Instant::now() + DUMP_WINDOW;
    let plan = JobPlan::new(6, 1 << 16).with_estimated_per_item_ns(200);
    let mut wide = vec![1.5f64; 1 << 16];
    loop {
        for_each_chunk(&plan, &mut wide, |s| {
            for x in s.iter_mut() {
                let mut y = *x;
                for _ in 0..64 {
                    y = y.sqrt() * 1.0000001;
                }
                *x = y;
            }
        });
        std::thread::sleep(Duration::from_millis(5));
        let done = trace::worker_flushes_done();
        if done >= expected || Instant::now() > deadline {
            return done;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let over: usize = argument(&args, 1, 8, "over");
    let arm = args.get(2).map(String::as_str).unwrap_or("rayon");
    if !trace::is_enabled() {
        eprintln!("the trace switch is not set; the run will report completion but record nothing");
    }

    let workers = global_local_arena().local_worker_count();
    let n = workers + over;
    println!("workers {workers}, fan-out {n}, arm {arm}");

    if arm == "rayon" {
        let mut warm: Vec<u64> = (0..n as u64).collect();
        rayon::scope(|s| {
            for slot in warm.iter_mut() {
                s.spawn(move |_| {
                    *slot = slot.wrapping_mul(0x100000001B3);
                });
            }
        });
        std::hint::black_box(&warm);
        println!("rayon scope of {n} completed, its pool is now up");
    }

    trace::reset_current_thread();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let plan = JobPlan::new(8, 1);
        let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> = (0..n)
            .map(|i| {
                Box::new(move || {
                    trace::emit(TraceEvent::LeafStart, i as u32);
                    let out = i as u32;
                    trace::emit(TraceEvent::LeafEnd, i as u32);
                    out
                }) as _
            })
            .collect();
        let started = Instant::now();
        let results = cooperative_join_n_flat_mailbox(&plan, closures);
        if let Err(e) = tx.send((results, started.elapsed())) {
            // Finished, but after the main thread had already reported
            // a stall. A slow fan-out and a stuck one must not read the
            // same, so this says which it was.
            eprintln!("the fan-out completed after the stall deadline: {e}");
        }
    });

    match rx.recv_timeout(STALL_AFTER) {
        Ok((results, wall)) => {
            println!("completed {} closures in {wall:?}", results.len());
        }
        Err(RecvTimeoutError::Timeout) => {
            println!("STALLED: no completion in {STALL_AFTER:?} at fan-out {n}");
        }
        Err(RecvTimeoutError::Disconnected) => {
            // The fan-out thread ended without sending, so a closure or
            // the join itself panicked. That is a different failure from
            // a stall and the trace below means something different too.
            println!("the fan-out thread ended without sending: it panicked rather than stalled");
        }
    }

    trace::dump_to_stderr("caller");
    let done = drain_worker_dumps(workers as u64);
    println!("worker dumps completed: {done} of {workers}");
    println!(
        "closures whose index appears in no row never ran; \
         a push and a wait-begin with no wait-end is a fan-out that did not finish"
    );
}

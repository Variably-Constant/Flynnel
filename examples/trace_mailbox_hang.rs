//! One traced mailbox fan-out, with the closures reporting which of
//! them ran, so a fan-out that does not complete says which closures
//! were never executed rather than only that it stopped.
//!
//! Run with the trace switch on:
//!
//! ```text
//! cargo run --release --example trace_mailbox_hang -- 8 rayon 1 2> hang.csv
//! cargo run --release --example trace_mailbox_hang -- 1000 alone 300 mailbox,deque
//! ```
//!
//! Arguments: how far above the local worker count to fan out
//! (default 8); whether a `rayon` scope fan-out runs first or the
//! fan-out runs `alone`; how many times to repeat each shape (default
//! 1); and the shapes to run, comma-separated, in order (default
//! `mailbox`). The first two arms differ in nothing else, so a
//! completion in one and a stall in the other attributes the difference
//! to the foreign pool.
//!
//! The shape argument is a sequence rather than a single choice because
//! neither shape alone reproduces the stall: 300 iterations of 1024
//! closures complete in under a tenth of a second in both, while the
//! criterion sweep stalls on a deque arm that ran directly after a
//! mailbox arm in the same process. A single shape cannot express a run
//! whose earlier arm leaves state behind, so the sequence is the
//! experiment and one shape on its own is the control for it.
//!
//! The repeat count exists because a single call is the one thing this
//! does that a criterion sweep does not: criterion calls the same
//! fan-out thousands of times inside a warm-up and a sampling loop, and
//! a stall seen there and not here may need the repetition rather than
//! the shape. A run that stalls at some iteration reports which one, so
//! a rare event is distinguishable from one that needs a particular
//! predecessor.
//!
//! Every closure emits two rows, so a large repeat count with the trace
//! switch on buffers `2 * n * repeats` of them before anything is
//! written. Find whether it stalls with the switch off, then re-run
//! with tracing at a count near the iteration it reached.
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
use flynnel::sched::cooperative::{cooperative_join_n_flat, cooperative_join_n_flat_mailbox};
use flynnel::sched::trace::{self, TraceEvent};
use flynnel::{JobPlan, for_each_chunk};

/// What the fan-out thread reports back.
///
/// `Entering` is sent before each call rather than after, so a silence
/// names the iteration that stalled instead of the last that finished.
enum Progress {
    Entering(&'static str, usize),
    Finished(usize, Duration),
}

/// Which fan-out entry point one block of iterations calls.
///
/// Both reach the same wait path and differ in how a pushed job is made
/// visible to a worker, so a stall in one and completion in the other
/// separates the wait from the push.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Mailbox,
    Deque,
}

impl Shape {
    /// The shapes named by a comma-separated list, in order, or the
    /// first word that names none.
    ///
    /// A sequence rather than a single shape because the stall was seen
    /// on a deque arm that ran after a mailbox arm in the same process,
    /// and one arm alone reproduces neither.
    fn sequence(text: &str) -> Result<Vec<Shape>, String> {
        text.split(',')
            .map(|word| match word.trim() {
                "mailbox" => Ok(Shape::Mailbox),
                "deque" => Ok(Shape::Deque),
                other => Err(other.to_string()),
            })
            .collect()
    }

    fn name(self) -> &'static str {
        match self {
            Shape::Mailbox => "mailbox",
            Shape::Deque => "deque",
        }
    }
}

/// How long to wait for the next report before calling it stalled.
///
/// This bounds one iteration, not the whole run: a repeat count that
/// makes progress keeps resetting it.
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
    let repeats: usize = argument(&args, 3, 1, "repeats").max(1);
    let shapes = match args.get(4) {
        None => vec![Shape::Mailbox],
        Some(text) => match Shape::sequence(text) {
            Ok(shapes) => shapes,
            Err(word) => {
                eprintln!("shape {word:?} is not one of mailbox, deque");
                std::process::exit(2);
            }
        },
    };
    if !trace::is_enabled() {
        eprintln!("the trace switch is not set; the run will report completion but record nothing");
    }

    let workers = global_local_arena().local_worker_count();
    let n = workers + over;
    let written: Vec<&str> = shapes.iter().map(|s| s.name()).collect();
    let total = shapes.len() * repeats;
    println!(
        "workers {workers}, fan-out {n}, arm {arm}, repeats {repeats} per shape, shapes {}",
        written.join(",")
    );

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
        let started = Instant::now();
        let mut last = 0usize;
        for shape in shapes {
            for iteration in 0..repeats {
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
                // Reported before the call rather than after, so a stall
                // says which iteration it stalled on rather than which one
                // last finished.
                if let Err(e) = tx.send(Progress::Entering(shape.name(), iteration)) {
                    eprintln!(
                        "the harness stopped listening in {} at iteration {iteration}: {e}",
                        shape.name()
                    );
                    return;
                }
                last = match shape {
                    Shape::Deque => cooperative_join_n_flat(&plan, closures).len(),
                    Shape::Mailbox => cooperative_join_n_flat_mailbox(&plan, closures).len(),
                };
            }
        }
        if let Err(e) = tx.send(Progress::Finished(last, started.elapsed())) {
            // Finished, but after the main thread had already reported
            // a stall. A slow fan-out and a stuck one must not read the
            // same, so this says which it was.
            eprintln!("the fan-out completed after the stall deadline: {e}");
        }
    });

    let mut reached = ("none", 0usize);
    let outcome = loop {
        match rx.recv_timeout(STALL_AFTER) {
            Ok(Progress::Entering(shape, i)) => reached = (shape, i),
            other => break other,
        }
    };

    let (shape, iteration) = reached;
    match outcome {
        Ok(Progress::Entering(..)) => unreachable!("the loop breaks on anything else"),
        Ok(Progress::Finished(count, wall)) => {
            println!("completed {steps} x {count} closures in {wall:?}", steps = total);
        }
        Err(RecvTimeoutError::Timeout) => {
            // Which shape it stalled in is the whole answer when a
            // sequence runs: the arm that stops is not necessarily the
            // one that caused it to.
            println!(
                "STALLED: no progress in {STALL_AFTER:?} at fan-out {n}, \
                 in shape {shape} on iteration {iteration} of {repeats}"
            );
        }
        Err(RecvTimeoutError::Disconnected) => {
            // The fan-out thread ended without sending, so a closure or
            // the join itself panicked. That is a different failure from
            // a stall and the trace below means something different too.
            println!(
                "the fan-out thread ended without sending in shape {shape} \
                 on iteration {iteration}: it panicked rather than stalled"
            );
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

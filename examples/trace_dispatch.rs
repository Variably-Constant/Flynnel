//! One traced dispatch of a chosen shape, for attributing the fixed
//! cost of a dispatch at small total work. Runs 200 warm calls, then
//! resets the caller's trace buffer, runs one more call, and dumps
//! every thread's buffer as comma-separated rows on stderr.
//!
//! Run with the trace switch on (`FLYNNEL_TRACE=1` in the
//! environment):
//!
//! ```text
//! cargo run --release --example trace_dispatch -- 10000 6 light 2> trace_light.csv
//! cargo run --release --example trace_dispatch -- 1000 30 add 2> trace_add.csv
//! ```
//!
//! Arguments: item count, the per-item estimate in nanoseconds
//! passed to the plan (which decides collapse against dispatch), and
//! the kind: `light` (a few flops per f64 through `for_each_chunk`)
//! or `add` (`out = a + b` over three slices through
//! `for_each_chunk_triple_min_leaf` with a 64-item floor). Each row
//! is `thread, event, payload, tsc` after a fixed first field; the
//! events are 1 enter, 2 exit, 3 leaf start, 4 leaf end, 5 join
//! push, 6 join wait begin, 7 join wait end (payload 1 when a thief
//! ran the half), 10 slot push, 11 slot wait end (payload 1 when the
//! caller was still spinning), 12 slot job start, 13 slot job end;
//! 8 and 9 are defined for a worker wake and a steal hit but no hook
//! emits them. The wall time of the traced call is printed on
//! stdout.

use std::str::FromStr;
use std::time::Instant;

use flynnel::sched::par_iter::for_each_chunk_triple_min_leaf;
use flynnel::sched::trace;
use flynnel::{JobPlan, for_each_chunk};

/// Three chained square roots per item, a few tens of nanoseconds.
#[inline(never)]
fn light(x: &mut f64) {
    let mut v = *x;
    for _ in 0..3 {
        v = v.sqrt() * 1.0000001_f64;
    }
    *x = v;
}

/// The `index`th argument parsed as `T`, `default` when absent; an
/// argument that does not parse ends the run with its error.
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = argument(&args, 1, 10_000, "items");
    let est: u32 = argument(&args, 2, 6, "estimate");
    let kind = args.get(3).map(String::as_str).unwrap_or("light");
    if !trace::is_enabled() {
        eprintln!("the trace switch is not set; the run will time the call but record nothing");
    }
    let p = flynnel::sched::par_iter::calibrate_host_dispatch();
    println!(
        "host profile: dispatch cost {} ns, collapse {} ns, wake {} ns",
        p.dispatch_cost_ns, p.collapse_threshold_ns, p.jec_wake_threshold_ns
    );
    let plan = JobPlan::new(6, n as u32).with_estimated_per_item_ns(est);
    let mut v = vec![1.5f64; n];
    let a = vec![1.0f64; n];
    let b = vec![2.0f64; n];
    let mut out = vec![0f64; n];
    let mut call = || match kind {
        "add" => for_each_chunk_triple_min_leaf(&plan, &mut out, &a, &b, 64, |o, x, y| {
            for ((o, x), y) in o.iter_mut().zip(x).zip(y) {
                *o = x + y;
            }
        }),
        _ => for_each_chunk(&plan, &mut v, |s| s.iter_mut().for_each(light)),
    };
    for _ in 0..200 {
        call();
    }
    trace::reset_current_thread();
    let t0 = Instant::now();
    call();
    let wall = t0.elapsed();
    println!("traced call: {kind} n={n} est={est} ns/item, wall {wall:?}");
    trace::dump_to_stderr("caller");
    trace::request_worker_flush();
    // Workers dump at the top of their next loop pass, and a parked
    // worker reaches it only when woken: wide heavy dispatches wake
    // every worker, repeated with pauses so each gets a pass.
    // A dump writes thousands of rows through the stderr lock and
    // takes milliseconds; the process must not exit while one is in
    // progress, or the newest rows, which are the traced call's, are
    // the ones lost. The pool holds one worker per hardware thread,
    // so that is the count of dumps to wait for.
    let mut wide = vec![1.5f64; 1 << 20];
    let wide_plan = JobPlan::new(6, wide.len() as u32).with_estimated_per_item_ns(50).with_smt();
    let expected = std::thread::available_parallelism().map_or(1, |n| n.get()) as u64;
    let deadline = Instant::now() + std::time::Duration::from_secs(30);
    let done = loop {
        for_each_chunk(&wide_plan, &mut wide, |s| {
            for x in s.iter_mut() {
                let mut y = *x;
                for _ in 0..16 {
                    y = y.sqrt() * 1.0000001;
                }
                *x = y;
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(5));
        let done = trace::worker_flushes_done();
        if done >= expected {
            break done;
        }
        if Instant::now() > deadline {
            eprintln!("only {done} of {expected} worker dumps completed after 30 s");
            break done;
        }
    };
    // The last dumps may still be writing: give the stderr lock time
    // to drain before exit.
    std::thread::sleep(std::time::Duration::from_millis(500));
    println!("worker dumps completed: {done} of {expected}");
    trace::clear_worker_flush_request();
    std::hint::black_box(&wide);
    std::hint::black_box((&v, &out));
}

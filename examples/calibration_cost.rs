//! What a calibration costs: the draw, and the read that follows it.
//!
//! The age bound makes a record expire, and a process whose record has
//! expired draws again. How much that costs decides whether a day is a
//! reasonable bound or an expensive one, and the figure was asserted in
//! a doc comment before anyone measured it.
//!
//! Two timings from one process. The first `calibrate_host_dispatch`
//! finds nothing stored and measures; the second finds what the first
//! published and reads it back. The difference between them is what a
//! process pays when its record has aged out.
//!
//! The directory must be empty. A directory holding a record makes the
//! first call a read, and the run would report two reads while looking
//! exactly like a draw and a read.
//!
//! One draw per process, so a spread over draws needs the process run
//! repeatedly against a fresh directory each time.
//!
//! ```sh
//! FLYNNEL_CALIBRATION_DIR=/fresh/dir calibration_cost
//! ```

use std::time::Instant;

use flynnel::sched::par_iter::calibrate_host_dispatch;

/// Files directly under `dir`, or a reason it could not be counted.
///
/// An entry that will not read is reported rather than skipped: the
/// count decides whether the first calibration call draws or reads, so
/// a directory that looks emptier than it is would make this run report
/// a read as a draw.
fn count_entries(dir: &std::path::Path) -> Result<usize, String> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(err) => return Err(format!("cannot read {}: {err}", dir.display())),
    };
    let mut n = 0usize;
    for entry in entries {
        match entry {
            Ok(_) => n += 1,
            Err(err) => {
                return Err(format!("cannot read an entry under {}: {err}", dir.display()));
            }
        }
    }
    Ok(n)
}

fn main() {
    let Some(dir) = flynnel::sched::calibration_store::calibration_dir() else {
        eprintln!(
            "no calibration directory: FLYNNEL_CALIBRATION_DIR is unset and the \
             per-user cache location could not be resolved"
        );
        std::process::exit(2);
    };

    if std::env::var_os("FLYNNEL_HOST_PROFILE_NS").is_some() {
        eprintln!(
            "FLYNNEL_HOST_PROFILE_NS is set, so calibrate_host_dispatch installs that \
             pin and measures nothing; there is no draw here to time"
        );
        std::process::exit(2);
    }

    let existing = match count_entries(&dir) {
        Ok(n) => n,
        Err(reason) => {
            eprintln!("{reason}");
            std::process::exit(3);
        }
    };
    if existing > 0 {
        eprintln!(
            "{} already holds {existing} file(s), so the first call would read rather \
             than draw; point FLYNNEL_CALIBRATION_DIR at a fresh path",
            dir.display()
        );
        std::process::exit(2);
    }

    let t0 = Instant::now();
    let drawn = calibrate_host_dispatch();
    let draw_ns = t0.elapsed().as_nanos() as u64;

    let t1 = Instant::now();
    let served = calibrate_host_dispatch();
    let read_ns = t1.elapsed().as_nanos() as u64;

    // The two calls must agree. A second call returning different
    // figures measured again rather than reading, and read_ns would be a
    // second draw wearing the wrong label.
    let agree = drawn.dispatch_cost_ns == served.dispatch_cost_ns
        && drawn.collapse_threshold_ns == served.collapse_threshold_ns
        && drawn.jec_wake_threshold_ns == served.jec_wake_threshold_ns;

    println!(
        "calibration draw_ns={draw_ns} read_ns={read_ns} agree={agree} \
         dispatch={} collapse={} wake={}",
        drawn.dispatch_cost_ns, drawn.collapse_threshold_ns, drawn.jec_wake_threshold_ns
    );

    if !agree {
        eprintln!(
            "the second call returned different figures, so it measured rather than \
             reading and read_ns is a draw; nothing here times a read"
        );
        std::process::exit(4);
    }
}

//! What this host's calibration store holds, without measuring anything.
//!
//! The store decides how every dispatch on the machine routes, and
//! until now nothing could show what was in it. Answering "is the
//! stored record any good" meant decoding the file by hand at known
//! byte offsets, which is how a record here was called contended for a
//! day on the strength of when it was written rather than what it said.
//!
//! Reads and never writes. `CalibrationStore::open_or_create` would lay
//! out a fresh table where none exists, so the path is tested first: a
//! diagnostic that creates the thing it reports on answers a question
//! about itself.
//!
//! Other stamps' files are listed but not decoded. A different stamp is
//! a different table, drawn by a different core count or layout, and
//! the numbers in it describe something else.
//!
//! ```sh
//! FLYNNEL_CALIBRATION_DIR=/some/dir calibration_dump
//! ```

use flynnel::sched::calibration_store::{CalibrationStore, HostStamp, calibration_dir, table_path};

fn main() {
    let Some(dir) = calibration_dir() else {
        eprintln!(
            "no calibration directory: FLYNNEL_CALIBRATION_DIR is unset and the per-user \
             cache location could not be resolved from the environment"
        );
        std::process::exit(2);
    };
    println!("dir {}", dir.display());

    let stamp = HostStamp::detect();
    let path = table_path(&dir, &stamp);
    println!(
        "stamp {:016x} vendor={} cpuid={:#010x} arch={} os={} primary={} total={} layout={}",
        stamp.hash(),
        stamp.vendor,
        stamp.cpuid_signature,
        stamp.arch,
        stamp.os,
        stamp.primary_workers,
        stamp.total_workers,
        stamp.layout_version,
    );

    // Every table in the directory, so a reader can see that this host's
    // stamp is one of several and which one answers for it.
    match std::fs::read_dir(&dir) {
        Ok(entries) => {
            for entry in entries {
                match entry {
                    Ok(entry) => {
                        let name = entry.file_name();
                        let mine = entry.path() == path;
                        let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
                        println!(
                            "table {} bytes={len}{}",
                            name.to_string_lossy(),
                            if mine { "  <- this host" } else { "" }
                        );
                    }
                    Err(err) => println!("table unreadable: {err}"),
                }
            }
        }
        Err(err) => println!("directory unreadable: {err}"),
    }

    if !path.exists() {
        println!("record absent: nothing has published for this stamp");
        return;
    }

    let store = match CalibrationStore::open_or_create(&dir, &stamp) {
        Ok(store) => store,
        Err(err) => {
            eprintln!("the table for this stamp exists and would not open: {err:?}");
            std::process::exit(3);
        }
    };

    let Some((cpu, accel)) = store.read() else {
        // Distinct from an absent record: the reader lost every race
        // against a writer, which means something is publishing now.
        println!("record unreadable: a writer held the payload for every retry");
        std::process::exit(4);
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let age = now.saturating_sub(cpu.measured_unix_s);

    println!(
        "record dispatch={} collapse={} wake={} measured_unix={} age_s={age} \
         spread={} samples={} occupancy={:?} trustworthy={} devices={}",
        cpu.dispatch_cost_ns,
        cpu.collapse_threshold_ns,
        cpu.jec_wake_threshold_ns,
        cpu.measured_unix_s,
        cpu.spread_per_mille,
        cpu.samples,
        cpu.occupancy(),
        cpu.is_trustworthy(),
        accel.len(),
    );

    // The age is printed beside the record rather than judged here. What
    // counts as too old is a bound the caller configures, and a dump
    // that applied one would be reporting its own default rather than
    // what the store holds.
    println!(
        "age_s={age} is what FLYNNEL_CALIBRATION_MAX_AGE_S is compared against; \
         a record older than the bound is drawn again at the next start"
    );
}

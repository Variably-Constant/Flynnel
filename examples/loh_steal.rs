//! Cross-process LOH steal end-to-end.
//!
//! Same two-role shape as `examples/chase_lev_mmf_steal.rs` but
//! routed through the LOH backend: originator dispatches a burst of
//! 100 add-jobs via `dispatch_marshal_batch`, child process steals
//! the slots via the LCRQ ring's CAS-on-head, and the parent reads
//! results back through the MMF latch arena.
//!
//! Run:
//! ```text
//! cargo run --release --features shared-memory-worker-reference \
//!     --example loh_steal
//! ```

#![allow(clippy::missing_docs_in_private_items)]

use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use flynnel::backend::shared_mem::{
    LATCH_SET, LohDeque, LohSteal, MmfLatchArena, hash_name, register,
};
use flynnel::backend::shared_mem::lcrq_lifo::LohLifoEntry;

const ADD_HANDLER_NAME: &str = "flynnel.example.loh_steal.add";
const EXIT_HANDLER_NAME: &str = "flynnel.example.loh_steal.exit";
const FLUSH_THRESHOLD: usize = 8;
const RING_CAPACITY: usize = 128;
const LATCH_CAPACITY: usize = 256;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 4 && args[1] == "--worker" {
        return run_worker(&args[2], &args[3]);
    }
    run_originator()
}

fn run_originator() -> std::io::Result<()> {
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let deque_path = std::env::temp_dir()
        .join(format!("flynnel_loh_xproc_deque_{pid}_{nonce}.bin"));
    let latches_path = std::env::temp_dir()
        .join(format!("flynnel_loh_xproc_latches_{pid}_{nonce}.bin"));

    let deque = LohDeque::create(&deque_path, RING_CAPACITY, FLUSH_THRESHOLD)
        .expect("create deque");
    let latches = MmfLatchArena::create(&latches_path, LATCH_CAPACITY)
        .expect("create latch arena");

    println!("originator: deque   = {}", deque_path.display());
    println!("originator: latches = {}", latches_path.display());
    println!("originator: spawning worker child process");

    let me = std::env::current_exe().expect("current exe");
    let mut child = Command::new(me)
        .arg("--worker")
        .arg(&deque_path)
        .arg(&latches_path)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("spawn worker child");

    std::thread::sleep(std::time::Duration::from_millis(250));

    let add_id = hash_name(ADD_HANDLER_NAME);
    let exit_id = hash_name(EXIT_HANDLER_NAME);

    let n = 100usize;
    let mut latch_offsets = Vec::with_capacity(n);
    let mut expected = Vec::with_capacity(n);

    let t_dispatch_start = Instant::now();
    // Dispatch in fully-batched form: every push goes into the local
    // LIFO; auto-flush at FLUSH_THRESHOLD fires every 8 pushes; an
    // explicit final flush below sweeps the tail.
    for i in 0..n {
        let a = i as u32;
        let b = (i as u32) * 7 + 1;
        expected.push(a + b);
        let latch_off = latches.alloc();
        latch_offsets.push(latch_off);
        let mut args = [0u8; 8];
        args[..4].copy_from_slice(&a.to_le_bytes());
        args[4..].copy_from_slice(&b.to_le_bytes());
        let entry = LohLifoEntry::new(add_id, latch_off, &args)
            .expect("build entry");
        loop {
            match deque.push(entry) {
                Ok(()) => break,
                Err(e) => {
                    eprintln!("originator: push failed: {e:?}, flushing + retrying");
                    deque.flush().ok();
                    std::thread::yield_now();
                }
            }
        }
    }
    // Sweep any tail items left in the LIFO below the threshold.
    let n_migrated = deque.flush().expect("final flush");
    println!("originator: final flush migrated {n_migrated} tail items");
    let dispatch_elapsed = t_dispatch_start.elapsed();

    println!(
        "originator: dispatched {n} jobs in {:.3?} ({:.1} ns/job)",
        dispatch_elapsed,
        dispatch_elapsed.as_nanos() as f64 / n as f64
    );

    let t_wait_start = Instant::now();
    let mut results = Vec::with_capacity(n);
    let mut buf = Vec::new();
    for (i, off) in latch_offsets.iter().enumerate() {
        loop {
            if latches.is_set(*off).expect("is_set") {
                break;
            }
            std::hint::spin_loop();
        }
        let state = latches.read_result(*off, &mut buf).expect("read");
        assert_eq!(state, LATCH_SET, "result {i} not SET");
        let mut arr = [0u8; 4];
        arr.copy_from_slice(&buf[..4]);
        results.push(u32::from_le_bytes(arr));
        latches.reset(*off).expect("reset");
    }
    let wait_elapsed = t_wait_start.elapsed();
    let total_elapsed = t_dispatch_start.elapsed();

    println!(
        "originator: collected {n} results in {:.3?} (round-trip {:.1} ns/job total)",
        wait_elapsed,
        total_elapsed.as_nanos() as f64 / n as f64
    );

    let mut all_good = true;
    for (i, (got, want)) in results.iter().zip(expected.iter()).enumerate() {
        if got != want {
            eprintln!("MISMATCH at i={i}: got={got}, want={want}");
            all_good = false;
        }
    }
    if all_good {
        println!("originator: all {n} results match expected sums (cross-process LOH steal verified)");
    } else {
        println!("originator: FAILED - results do not match expected sums");
    }

    // Tell the worker to exit. One more push + flush.
    let exit_latch = latches.alloc();
    let exit_entry = LohLifoEntry::new(exit_id, exit_latch, &[]).expect("build exit");
    deque.push(exit_entry).expect("push exit");
    deque.flush().expect("flush exit");
    loop {
        if latches.is_set(exit_latch).expect("is_set exit") {
            break;
        }
        std::hint::spin_loop();
    }
    println!("originator: worker acked exit");
    let status = child.wait().expect("wait child");
    println!("originator: child exit status = {status:?}");

    std::fs::remove_file(&deque_path).ok();
    std::fs::remove_file(&latches_path).ok();
    println!("originator: cleaned up MMF files");

    if !all_good || !status.success() {
        std::process::exit(1);
    }
    Ok(())
}

fn run_worker(deque_path: &str, latches_path: &str) -> std::io::Result<()> {
    let deque = LohDeque::open(Path::new(deque_path), FLUSH_THRESHOLD)
        .expect("worker: open deque");
    let latches = MmfLatchArena::open(Path::new(latches_path))
        .expect("worker: open latch arena");

    let add_id = hash_name(ADD_HANDLER_NAME);
    let exit_id = hash_name(EXIT_HANDLER_NAME);

    register(add_id, |args| {
        let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
        let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
        Ok(a.wrapping_add(b).to_le_bytes().to_vec())
    });
    register(exit_id, |_| Ok(b"bye".to_vec()));

    println!("worker: attached, draining LOH deque (pid={})", std::process::id());
    std::io::stdout().flush().ok();

    let mut drained = 0usize;
    loop {
        match deque.steal() {
            LohSteal::Success(slot) => {
                if slot.closure_id == exit_id {
                    latches.publish(slot.latch_offset, b"bye").expect("publish bye");
                    drained += 1;
                    println!("worker: drained {drained} total; exit received");
                    break;
                }
                let pass = flynnel::backend::shared_mem::Pass {
                    closure_id: slot.closure_id,
                    args: slot.args().to_vec(),
                };
                match flynnel::backend::shared_mem::pass_registry::execute(&pass) {
                    Ok(reply) => latches
                        .publish(slot.latch_offset, &reply)
                        .expect("publish reply"),
                    Err(err) => latches
                        .publish_err(slot.latch_offset, err.to_string().as_bytes())
                        .expect("publish err"),
                };
                drained += 1;
            }
            LohSteal::Empty | LohSteal::Retry => std::hint::spin_loop(),
        }
    }
    Ok(())
}

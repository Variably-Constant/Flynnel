//! Cross-process MMF Chase-Lev steal end-to-end.
//!
//! Two roles, one binary:
//!
//! - **Originator** (default): creates the deque + latch arena files,
//!   spawns the **worker** as a child process pointing at the same
//!   files, then dispatches N add-jobs and waits for the result of
//!   each through the MMF latch. Finally posts an `exit` job that
//!   tells the worker to drain and quit.
//! - **Worker** (when invoked with `--worker <deque_path>
//!   <latch_path>`): opens the two MMF files, registers the
//!   "add" + "exit" handlers in its local pass_registry, drains the
//!   deque in a loop, and publishes results until it receives the
//!   exit signal.
//!
//! Run:
//! ```text
//! cargo run --example chase_lev_mmf_steal \
//!     --features shared-memory-worker-reference --release
//! ```
//!
//! The example demonstrates: (a) two processes sharing one
//! MmfChaseLevDeque file, (b) a thief in the child process stealing
//! jobs the parent pushed via the same byte-layout Chase-Lev
//! protocol that cross-thread thieves use, (c) results published
//! through the MMF latch arena back to the originator with no
//! response ring at all.
//!
//! What this example does NOT do: replace the in-process scheduler
//! or wire up the dual-deque worker pool. Those integrations build
//! on this same substrate; the substrate's correctness is the
//! demonstration here.

#![allow(clippy::missing_docs_in_private_items)]

use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use flynnel::backend::shared_mem::{
    LATCH_SET, MmfChaseLevDeque, MmfLatchArena, hash_name, register,
};
use flynnel::backend::shared_mem::chase_lev_mmf::{RemoteJobSlot, Steal};

const ADD_HANDLER_NAME: &str = "flynnel.example.chase_lev_mmf_steal.add";
const EXIT_HANDLER_NAME: &str = "flynnel.example.chase_lev_mmf_steal.exit";

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // Worker-mode invocation:
    //   chase_lev_mmf_steal --worker <deque_path> <latch_path>
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
        .join(format!("flynnel_xproc_deque_{pid}_{nonce}.bin"));
    let latches_path = std::env::temp_dir()
        .join(format!("flynnel_xproc_latches_{pid}_{nonce}.bin"));

    let deque = MmfChaseLevDeque::create(&deque_path, 64)
        .expect("create deque");
    let latches = MmfLatchArena::create(&latches_path, 128)
        .expect("create latch arena");

    println!("originator: deque   = {}", deque_path.display());
    println!("originator: latches = {}", latches_path.display());
    println!("originator: spawning worker child process");

    // Spawn this same binary in worker mode pointing at the files.
    let me = std::env::current_exe().expect("current exe");
    let mut child = Command::new(me)
        .arg("--worker")
        .arg(&deque_path)
        .arg(&latches_path)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("spawn worker child");

    // Wait briefly for the worker to attach. A short sleep is fine
    // here because the worker prints "attached" once ready and we
    // wouldn't dispatch before that; for the example we rely on a
    // small fixed delay rather than IPC handshake plumbing.
    std::thread::sleep(std::time::Duration::from_millis(250));

    let add_id = hash_name(ADD_HANDLER_NAME);
    let exit_id = hash_name(EXIT_HANDLER_NAME);

    // Dispatch N add jobs (a + b). Each pushes a slot with the
    // pre-allocated latch_offset; the worker publishes the sum.
    let n = 100usize;
    let mut latch_offsets = Vec::with_capacity(n);
    let mut expected = Vec::with_capacity(n);

    let t_dispatch_start = Instant::now();
    for i in 0..n {
        let a = i as u32;
        let b = (i as u32) * 7 + 1;
        expected.push(a + b);
        let latch_off = latches.alloc();
        latch_offsets.push(latch_off);
        let mut args = [0u8; 8];
        args[..4].copy_from_slice(&a.to_le_bytes());
        args[4..].copy_from_slice(&b.to_le_bytes());
        let slot = RemoteJobSlot::new(add_id, latch_off, &args)
            .expect("build slot");
        loop {
            match deque.push(slot) {
                Ok(()) => break,
                Err(e) => {
                    eprintln!("originator: push failed: {e:?}, retrying");
                    std::thread::yield_now();
                }
            }
        }
    }
    let dispatch_elapsed = t_dispatch_start.elapsed();

    println!(
        "originator: dispatched {n} jobs in {:.3?} ({:.1} ns/job)",
        dispatch_elapsed,
        dispatch_elapsed.as_nanos() as f64 / n as f64
    );

    // Wait for every latch to set, in order. Steady-state spinwait;
    // for a real workload the caller would use a backoff or park.
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

    // Verify correctness.
    let mut all_good = true;
    for (i, (got, want)) in results.iter().zip(expected.iter()).enumerate() {
        if got != want {
            eprintln!("MISMATCH at i={i}: got={got}, want={want}");
            all_good = false;
        }
    }
    if all_good {
        println!("originator: all {n} results match expected sums (cross-process steal verified)");
    } else {
        println!("originator: FAILED - results do not match expected sums");
    }

    // Tell the worker to drain remaining + exit.
    let exit_latch = latches.alloc();
    let exit_slot = RemoteJobSlot::new(exit_id, exit_latch, &[]).expect("build exit");
    loop {
        match deque.push(exit_slot) {
            Ok(()) => break,
            Err(_) => std::thread::yield_now(),
        }
    }
    // Wait for the worker to ack the exit (publishes "bye" into the
    // latch, then exits).
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
    let deque = MmfChaseLevDeque::open(Path::new(deque_path))
        .expect("worker: open deque");
    let latches = MmfLatchArena::open(Path::new(latches_path))
        .expect("worker: open latch arena");

    let add_id = hash_name(ADD_HANDLER_NAME);
    let exit_id = hash_name(EXIT_HANDLER_NAME);

    // Register handlers in this child's local registry.
    register(add_id, |args| {
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        a.copy_from_slice(&args[0..4]);
        b.copy_from_slice(&args[4..8]);
        let sum = u32::from_le_bytes(a).wrapping_add(u32::from_le_bytes(b));
        Ok(sum.to_le_bytes().to_vec())
    });
    register(exit_id, |_args| {
        // The exit handler publishes a "bye" payload; the worker
        // loop below sees the exit_id, publishes "bye", and breaks.
        Ok(b"bye".to_vec())
    });

    println!("worker: attached, draining deque (pid={})", std::process::id());
    std::io::stdout().flush().ok();

    let mut drained = 0usize;
    loop {
        match deque.steal() {
            Steal::Success(slot) => {
                if slot.closure_id == exit_id {
                    // Publish the exit ack into the latch and stop.
                    latches.publish(slot.latch_offset, b"bye").expect("publish bye");
                    drained += 1;
                    println!("worker: drained {drained} total; exit received, shutting down");
                    break;
                }
                // Add path: hand the args to the registered handler
                // and publish the reply.
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
            Steal::Empty => {
                std::hint::spin_loop();
            }
            Steal::Retry => {
                std::hint::spin_loop();
            }
        }
    }
    Ok(())
}

//! Cross-process [`CrossProcessDispatcher`] demo.
//!
//! Same two-role shape as `examples/chase_lev_mmf_steal.rs` and the
//! three sibling steal demos, but the originator routes through the
//! dispatcher so different workload shapes pick different backends.
//!
//! Two backends are wired into the dispatcher in this demo:
//!
//! - `SharedMemoryChaseLevBackend` for `WorkloadShape::request_reply`
//!   (single-drain, small payload) - the dispatcher's routing table
//!   sends those calls here.
//! - `SharedMemoryKhpdBackend` for `WorkloadShape::producer_fast`
//!   batched bursts - the table sends those there.
//!
//! Originator: dispatches 30 request-reply jobs + 1 producer-fast
//! batch of 60 jobs; the child process drains both backends in
//! parallel via two worker threads.
//!
//! Run:
//! ```text
//! cargo run --example dispatcher_demo \
//!     --features shared-memory-worker-reference --release
//! ```

#![allow(clippy::missing_docs_in_private_items)]

use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use flynnel::backend::shared_mem::{
    CrossProcessDispatcher, DequeVariant, DispatcherRoutingTable, SharedMemoryChaseLevBackend,
    SharedMemoryKhpdBackend, WorkloadShape, hash_name, register,
};

const ADD_HANDLER: &str = "flynnel.example.dispatcher_demo.add";
const EXIT_HANDLER: &str = "flynnel.example.dispatcher_demo.exit";
const N_REQUEST_REPLY: usize = 30;
const N_PRODUCER_FAST: usize = 60;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 6 && args[1] == "--worker" {
        return run_worker(&args[2], &args[3], &args[4], &args[5]);
    }
    run_originator()
}

fn raw_pair(a: u32, b: u32) -> [u8; 8] {
    let mut p = [0u8; 8];
    p[..4].copy_from_slice(&a.to_le_bytes());
    p[4..].copy_from_slice(&b.to_le_bytes());
    p
}

fn run_originator() -> std::io::Result<()> {
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let cl_deque = std::env::temp_dir()
        .join(format!("flynnel_dd_cl_d_{pid}_{nonce}.bin"));
    let cl_latch = std::env::temp_dir()
        .join(format!("flynnel_dd_cl_l_{pid}_{nonce}.bin"));
    let khpd_deque = std::env::temp_dir()
        .join(format!("flynnel_dd_khpd_d_{pid}_{nonce}.bin"));
    let khpd_latch = std::env::temp_dir()
        .join(format!("flynnel_dd_khpd_l_{pid}_{nonce}.bin"));

    let cl = Arc::new(
        SharedMemoryChaseLevBackend::create(0, &cl_deque, &cl_latch, 256, 1024)
            .expect("create chase-lev"),
    );
    let khpd = Arc::new(
        SharedMemoryKhpdBackend::create(0, &khpd_deque, &khpd_latch, 256, 2048)
            .expect("create khpd"),
    );

    let dispatcher = CrossProcessDispatcher::builder()
        .with_table(DispatcherRoutingTable::default_heuristic())
        .with_chase_lev(Arc::clone(&cl))
        .with_khpd(Arc::clone(&khpd))
        .build();

    println!("originator: ChaseLev deque   = {}", cl_deque.display());
    println!("originator: ChaseLev latches = {}", cl_latch.display());
    println!("originator: KHPD     deque   = {}", khpd_deque.display());
    println!("originator: KHPD     latches = {}", khpd_latch.display());
    println!("originator: spawning child worker");

    let me = std::env::current_exe().expect("current exe");
    let mut child = Command::new(me)
        .arg("--worker")
        .arg(&cl_deque)
        .arg(&cl_latch)
        .arg(&khpd_deque)
        .arg(&khpd_latch)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("spawn worker child");

    // Give the child time to attach to both MMF files + register
    // handlers before we start dispatching.
    std::thread::sleep(std::time::Duration::from_millis(250));

    let add_id = hash_name(ADD_HANDLER);
    let exit_id = hash_name(EXIT_HANDLER);

    // ----- (A) request-reply jobs ---------------------------------
    let shape_rr = WorkloadShape::request_reply(8);
    let picked_a = dispatcher
        .pick_with_fallback(&shape_rr)
        .expect("pick rr");
    println!(
        "originator: request_reply shape routes to {:?}",
        picked_a
    );
    assert_eq!(picked_a, DequeVariant::ChaseLev,
        "default heuristic must route request_reply to Chase-Lev");

    let mut rr_handles = Vec::with_capacity(N_REQUEST_REPLY);
    let t_rr_start = Instant::now();
    for i in 0..N_REQUEST_REPLY as u32 {
        let payload = raw_pair(i, i * 7 + 1);
        let h = dispatcher
            .dispatch_marshal(&shape_rr, add_id, &payload)
            .expect("dispatch rr");
        rr_handles.push((h, i.wrapping_add(i * 7 + 1)));
    }
    let mut rr_correct = 0usize;
    for (h, expected) in &rr_handles {
        let r = dispatcher.wait_handle(*h, 1024).expect("wait").expect("ok");
        let v = u32::from_le_bytes(r[..4].try_into().unwrap());
        if v == *expected {
            rr_correct += 1;
        }
    }
    let rr_elapsed = t_rr_start.elapsed();
    println!(
        "originator: {N_REQUEST_REPLY} request_reply jobs: {rr_correct}/{N_REQUEST_REPLY} correct in {rr_elapsed:.3?}"
    );

    // ----- (B) producer-fast batch --------------------------------
    let shape_pf = WorkloadShape::producer_fast(8, N_PRODUCER_FAST as u32);
    let picked_b = dispatcher
        .pick_with_fallback(&shape_pf)
        .expect("pick pf");
    println!(
        "originator: producer_fast shape routes to {:?}",
        picked_b
    );
    assert_eq!(picked_b, DequeVariant::Khpd,
        "default heuristic must route producer_fast(8 B, burst 60) to KHPD");

    let pf_payloads: Vec<[u8; 8]> = (0..N_PRODUCER_FAST)
        .map(|i| raw_pair(i as u32 * 100, i as u32 + 1))
        .collect();
    let pf_items: Vec<(u32, &[u8])> =
        pf_payloads.iter().map(|p| (add_id, p.as_slice())).collect();
    let t_pf_start = Instant::now();
    let pf_handles = dispatcher
        .dispatch_marshal_batch(&shape_pf, &pf_items)
        .expect("dispatch pf");
    let mut pf_correct = 0usize;
    for (i, h) in pf_handles.iter().enumerate() {
        let r = dispatcher.wait_handle(*h, 1024).expect("wait").expect("ok");
        let v = u32::from_le_bytes(r[..4].try_into().unwrap());
        let want = (i as u32 * 100).wrapping_add(i as u32 + 1);
        if v == want {
            pf_correct += 1;
        }
    }
    let pf_elapsed = t_pf_start.elapsed();
    println!(
        "originator: {N_PRODUCER_FAST} producer_fast jobs: {pf_correct}/{N_PRODUCER_FAST} correct in {pf_elapsed:.3?}"
    );

    let total = N_REQUEST_REPLY + N_PRODUCER_FAST;
    let total_correct = rr_correct + pf_correct;
    println!(
        "originator: GRAND TOTAL: {total_correct}/{total} correct across both variants"
    );

    // Send exit signal through Chase-Lev (so the child's CL worker
    // sees it; KHPD worker exits on the same flag).
    let exit_handle = dispatcher
        .dispatch_marshal(&shape_rr, exit_id, &[])
        .expect("dispatch exit");
    dispatcher
        .wait_handle(exit_handle, 4096)
        .expect("wait exit")
        .expect("ok exit");
    println!("originator: worker acked exit");
    let status = child.wait().expect("wait child");
    println!("originator: child exit status = {status:?}");

    std::fs::remove_file(&cl_deque).ok();
    std::fs::remove_file(&cl_latch).ok();
    std::fs::remove_file(&khpd_deque).ok();
    std::fs::remove_file(&khpd_latch).ok();
    println!("originator: cleaned up MMF files");

    if total_correct != total || !status.success() {
        std::process::exit(1);
    }
    Ok(())
}

fn run_worker(
    cl_deque: &str,
    cl_latch: &str,
    khpd_deque: &str,
    khpd_latch: &str,
) -> std::io::Result<()> {
    let cl = Arc::new(
        SharedMemoryChaseLevBackend::open(0, Path::new(cl_deque), Path::new(cl_latch))
            .expect("open chase-lev"),
    );
    let khpd = Arc::new(
        SharedMemoryKhpdBackend::open(0, Path::new(khpd_deque), Path::new(khpd_latch))
            .expect("open khpd"),
    );

    let add_id = hash_name(ADD_HANDLER);
    let exit_id = hash_name(EXIT_HANDLER);
    register(add_id, |args| {
        let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
        let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
        Ok(a.wrapping_add(b).to_le_bytes().to_vec())
    });
    register(exit_id, |_| Ok(b"bye".to_vec()));

    println!("worker (pid={}): attached to both backends; draining...", std::process::id());
    std::io::stdout().flush().ok();

    // Stop flag flipped by the CL drain when it sees the exit
    // closure_id - that flips the KHPD drain off too.
    let stop = Arc::new(AtomicBool::new(false));

    // CL drain thread: catches add + exit slots.
    let cl_be = Arc::clone(&cl);
    let cl_stop = Arc::clone(&stop);
    let cl_w = std::thread::spawn(move || {
        let mut drained = 0usize;
        while !cl_stop.load(Ordering::Relaxed) {
            // drain_one already executes the handler and publishes
            // the latch. We just need to detect when an exit was
            // drained so we can flip the stop flag for the KHPD
            // thread; the latch on the exit's handle has been
            // published by drain_one already.
            //
            // We can't directly tell from drain_one whether the
            // slot was exit or add. Instead the originator waits
            // on the exit handle and only signals child-exit AFTER
            // the latch is set - so by the time we drain the exit
            // slot, the originator's wait_handle returns and the
            // originator sends SIGCHILD by closing stdin / exiting.
            //
            // Simpler: count drained slots; the exit job is the
            // LAST one the originator sends, so the count maxing
            // out is our signal. The dispatcher_demo schedule sends
            // N_REQUEST_REPLY add jobs + 1 exit on Chase-Lev.
            match cl_be.drain_one() {
                Ok(Some(())) => {
                    drained += 1;
                    if drained > N_REQUEST_REPLY {
                        cl_stop.store(true, Ordering::Relaxed);
                        break;
                    }
                }
                Ok(None) => std::hint::spin_loop(),
                Err(e) => {
                    eprintln!("worker: cl drain err: {e}");
                    cl_stop.store(true, Ordering::Relaxed);
                    return;
                }
            }
        }
        println!("worker: cl drained {drained} total items");
    });

    // KHPD drain thread: catches the producer-fast batched lines.
    let khpd_be = Arc::clone(&khpd);
    let khpd_stop = Arc::clone(&stop);
    let khpd_w = std::thread::spawn(move || {
        let mut drained = 0usize;
        while !khpd_stop.load(Ordering::Relaxed) {
            match khpd_be.drain_one_line() {
                Ok(Some(n)) => drained += n,
                Ok(None) => std::hint::spin_loop(),
                Err(e) => {
                    eprintln!("worker: khpd drain err: {e}");
                    return;
                }
            }
        }
        println!("worker: khpd drained {drained} total items");
    });

    cl_w.join().expect("cl join");
    khpd_w.join().expect("khpd join");
    Ok(())
}

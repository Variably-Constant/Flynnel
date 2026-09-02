//! Cross-process URD steal end-to-end.
//!
//! Same two-role shape as `examples/chase_lev_mmf_steal.rs`,
//! `examples/loh_steal.rs`, and `examples/khpd_steal.rs`, routed
//! through the URD backend: the originator publishes 100 add-jobs
//! in MAILBOX_ITEMS-sized batches via round-robin (here, 1 mailbox
//! so always mailbox 0); the child process drains mailbox 0 and
//! publishes results through the MMF latch arena.
//!
//! Reports which wait strategy URD picked at startup
//! (PauseSpin / WAITPKG) per CPUID.

#![allow(clippy::missing_docs_in_private_items)]

use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use flynnel::backend::shared_mem::{
    KHPD_ARGS_INLINE_BYTES, LATCH_SET, LineItem, MAILBOX_ITEMS, MmfLatchArena, UrdDeque, UrdDrain,
    UrdWaitStrategy, hash_name, register,
};
use flynnel::cpu_info::has_waitpkg;

const ADD_HANDLER_NAME: &str = "flynnel.example.urd_steal.add";
const EXIT_HANDLER_NAME: &str = "flynnel.example.urd_steal.exit";
const N_MAILBOXES: usize = 1;
const LATCH_CAPACITY: usize = 256;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 4 && args[1] == "--worker" {
        return run_worker(&args[2], &args[3]);
    }
    run_originator()
}

fn raw_pair(a: u32, b: u32) -> [u8; KHPD_ARGS_INLINE_BYTES] {
    let mut p = [0u8; KHPD_ARGS_INLINE_BYTES];
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
    let deque_path = std::env::temp_dir()
        .join(format!("flynnel_urd_xproc_deque_{pid}_{nonce}.bin"));
    let latches_path = std::env::temp_dir()
        .join(format!("flynnel_urd_xproc_latches_{pid}_{nonce}.bin"));

    let deque = UrdDeque::create(&deque_path, N_MAILBOXES).expect("create deque");
    let latches = MmfLatchArena::create(&latches_path, LATCH_CAPACITY)
        .expect("create latch arena");

    println!(
        "originator: WAITPKG available = {} (chosen strategy: {:?})",
        has_waitpkg(),
        UrdWaitStrategy::pick(),
    );
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
    // Collect items first, then publish in batches of MAILBOX_ITEMS.
    let mut items_buf = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as u32;
        let b = (i as u32) * 7 + 1;
        expected.push(a.wrapping_add(b));
        let latch_off = latches.alloc();
        latch_offsets.push(latch_off);
        let args = raw_pair(a, b);
        items_buf.push(LineItem::new(add_id, latch_off, &args).expect("build item"));
    }

    let t_dispatch_start = Instant::now();
    for chunk in items_buf.chunks(MAILBOX_ITEMS) {
        // round-robin publish (with n=1 mailbox, always mailbox 0)
        loop {
            match deque.publish_round_robin(chunk) {
                Ok(_) => break,
                Err(e) => {
                    eprintln!("originator: publish failed: {e:?}, retrying");
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
        println!("originator: all {n} results match expected sums (cross-process URD steal verified)");
    } else {
        println!("originator: FAILED - results do not match expected sums");
    }

    // Tell the worker to exit: one item published.
    let exit_latch = latches.alloc();
    let exit_item = LineItem::new(exit_id, exit_latch, &[]).expect("build exit");
    deque
        .publish_round_robin(&[exit_item])
        .expect("publish exit");
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
    let deque = UrdDeque::open(Path::new(deque_path)).expect("worker: open deque");
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

    println!(
        "worker: attached, wait_strategy={:?}, draining mailbox 0 (pid={})",
        deque.wait_strategy(),
        std::process::id(),
    );
    std::io::stdout().flush().ok();

    let mut drained_items = 0usize;
    let mut saw_exit = false;
    loop {
        match deque.drain_mailbox(0) {
            UrdDrain::Success(r) => {
                for i in 0..r.n_items {
                    let item = r.items[i];
                    if item.closure_id == exit_id {
                        latches
                            .publish(item.latch_offset, b"bye")
                            .expect("publish bye");
                        saw_exit = true;
                        drained_items += 1;
                        continue;
                    }
                    let pass = flynnel::backend::shared_mem::Pass {
                        closure_id: item.closure_id,
                        args: item.args_inline.to_vec(),
                    };
                    match flynnel::backend::shared_mem::pass_registry::execute(&pass) {
                        Ok(reply) => latches
                            .publish(item.latch_offset, &reply)
                            .expect("publish reply"),
                        Err(err) => latches
                            .publish_err(item.latch_offset, err.to_string().as_bytes())
                            .expect("publish err"),
                    };
                    drained_items += 1;
                }
                if saw_exit {
                    println!(
                        "worker: drained {drained_items} total items; exit received"
                    );
                    break;
                }
            }
            UrdDrain::Empty => std::hint::spin_loop(),
        }
    }
    Ok(())
}

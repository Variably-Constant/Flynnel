//! Learned hybrid placement demo: `hybrid_auto` races CPU vs
//! backend once per size bucket (racing IS the calibration), then
//! routes subsequent calls to whichever side measured faster.
//!
//! Run with:
//!   cargo run --release --example hybrid_auto_demo
//!
//! A deliberately-slow stub backend (3 ms sleep per dispatch,
//! standing in for a device whose launch latency dwarfs this
//! workload) is registered under `Backend::Custom(9001)`. The first
//! call races both sides; every later call in the same size bucket
//! goes straight to the CPU because the model learned the backend
//! loses at this batch size.

use std::sync::Arc;

use flynnel::{
    Backend, BackendCapabilities, DispatchBackend, JobPlan, Placement,
    hybrid_auto, hybrid_auto_split, register_backend,
};

/// Stub device: runs dispatched closures on a fresh thread with a
/// fixed 3 ms launch penalty.
struct SlowStubBackend;

impl DispatchBackend for SlowStubBackend {
    fn id(&self) -> Backend {
        Backend::Custom(9001)
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            simt_width: 32,
            max_threads_in_flight: 4096,
            launch_latency_ns: 3_000_000,
            h2d_bw_bytes_per_sec: 10_000_000_000,
        }
    }
    fn dispatch_parallel_for(&self, _count: u32, _work: &(dyn Fn(u32) + Send + Sync)) {}
    fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(3));
            work();
        });
    }
}

fn main() {
    println!("=== Learned hybrid placement demo ===\n");
    register_backend(Arc::new(SlowStubBackend));

    let data: Vec<u64> = (0..4096u64).collect();
    let expected: u64 = data.iter().sum();
    let plan = JobPlan::new(6, data.len() as u32).with_backend(Backend::Custom(9001));

    println!("[1] hybrid_auto: 6 calls at one batch size");
    let mut placements = Vec::new();
    for call in 1..=6u32 {
        let d = data.clone();
        let d2 = data.clone();
        let (sum, placement) = hybrid_auto(
            &plan,
            move || d.iter().sum::<u64>(),
            move || d2.iter().sum::<u64>(),
        );
        assert_eq!(sum, expected, "both implementations must agree");
        println!("    call {call}: placement = {placement:?}");
        placements.push(placement);
    }
    assert_eq!(
        placements[0],
        Placement::Race,
        "cold bucket must race both sides"
    );
    assert_eq!(
        *placements.last().expect("6 calls recorded"),
        Placement::Cpu,
        "model must learn the CPU wins against a 3ms-launch stub"
    );
    println!("    VERIFIED: raced once, then exploited the faster side.\n");

    println!("[2] hybrid_auto_split: learned split ratio drifts toward the CPU");
    for call in 1..=4u32 {
        let mut items = vec![1u64; 64 * 1024];
        let report = hybrid_auto_split(
            &plan,
            &mut items,
            |cpu_half| {
                for x in cpu_half.iter_mut() {
                    *x *= 3;
                }
            },
            |backend_half| {
                for x in backend_half.iter_mut() {
                    *x *= 3;
                }
            },
        );
        assert!(items.iter().all(|&x| x == 3), "every item transformed");
        println!(
            "    call {call}: cpu {} items in {} us | backend {} items in {} us | cpu share {}%",
            report.cpu_items,
            report.cpu_ns / 1_000,
            report.backend_items,
            report.backend_ns / 1_000,
            report.cpu_share_per_mille / 10,
        );
    }
    println!("    VERIFIED: split sizing follows the measured per-item throughputs.");
}

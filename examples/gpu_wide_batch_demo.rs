//! Two WDDM-shaped fixes for resident wide ops:
//!   A. async batch launch - queue N dependent kernels, sync once,
//!      instead of one command-buffer flush per kernel.
//!   B. a quiescable poller - pause the busy-polling doorbell poller
//!      so a device-filling wide op is not contended by it.
//!
//! Wide ops run on their own stream, so they no longer serialize
//! behind a resident poller quantum; these two levers remove the
//! remaining costs.
//!
//! Run with:
//!   cargo run --release --features gpu-peer --example gpu_wide_batch_demo

use std::time::{Duration, Instant};

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, OP_NOP, STATUS_DONE};

// Tiny dependent step: increment the first `n` elements. Chained in
// stream order, N of these add N. Small enough that the per-call sync
// dominates its cost.
const INC: &str = r#"
extern "C" __global__ void inc(float* d, unsigned n) {
    for (unsigned i = blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += gridDim.x*blockDim.x)
        d[i] += 1.0f;
}
"#;

// Memory-bandwidth-bound streaming kernel over a buffer larger than L2,
// re-read `iters` times. This is the RBM-shaped workload: it needs
// enough warps in flight across the SMs to saturate HBM, so blocks the
// poller holds are blocks this op cannot use to hide memory latency.
const STREAM: &str = r#"
extern "C" __global__ void stream_read(float* d, unsigned n, unsigned iters) {
    float acc = 0.0f;
    for (unsigned it = 0; it < iters; ++it)
        for (unsigned i = blockIdx.x*blockDim.x + threadIdx.x; i < n;
             i += gridDim.x*blockDim.x)
            acc += d[i];
    unsigned tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid == 0) d[0] = acc; // anti-DCE
}
"#;

fn main() {
    println!("=== Wide-op batch + quiescable poller ===\n");
    // 16 MiB resident block: bigger than the 3070's 4 MiB L2, so the
    // streaming op below hits HBM every pass.
    let block_elems = 4 * 1024 * 1024usize; // 4M f32 = 16 MiB
    let block_bytes = block_elems * 4;
    let mut peer = match GpuPeer::init(GpuPeerConfig {
        slot_bytes: 64 * 1024,
        slots_per_lane: 4,
        vram_block_bytes: block_bytes as u32,
        vram_blocks: 4,
        idle_exit_ns: 40_000_000, // a live poller stays resident ~40 ms
        ..GpuPeerConfig::default()
    }) {
        Ok(p) => p,
        Err(e) => {
            println!("substrate unavailable on this host: {e}");
            return;
        }
    };

    let inc_k = match peer.compile_wide_kernel(INC, "inc") {
        Ok(k) => k,
        Err(e) => {
            println!("NVRTC needed for wide kernels: {e}");
            return;
        }
    };
    let stream_k = peer.compile_wide_kernel(STREAM, "stream_read").expect("stream compile");

    // Pin a small header; the block is the full 16 MiB (alloc-zeroed),
    // which is what the streaming op reads. The small buffer keeps the
    // doorbell reset/fetch cheap.
    let hdr = 4096usize;
    let buf = peer.pin(&vec![0u8; hdr]).expect("pin");
    let (ptr, _) = peer.resident_ptr(&buf).expect("ptr");

    let reset = |peer: &mut GpuPeer| peer.write_resident(&buf, &vec![0u8; hdr]).expect("reset");
    let read_first = |peer: &mut GpuPeer| -> f32 {
        let mut o = vec![0u8; hdr];
        peer.fetch(&buf, &mut o).expect("fetch");
        f32::from_le_bytes([o[0], o[1], o[2], o[3]])
    };

    // --- A. async batch vs per-call sync (tiny kernels) ---
    let rounds = 200u32;
    let n_small = 256u32; // one block
    println!("[A] {rounds} tiny dependent inc kernels (one block each):");
    reset(&mut peer);
    let t0 = Instant::now();
    for _ in 0..rounds {
        peer.launch_wide(&inc_k, 1, 256, &[ptr], &[n_small]).expect("sync launch");
    }
    let per_call_ms = t0.elapsed().as_secs_f64() * 1e3;
    let v_per = read_first(&mut peer);

    reset(&mut peer);
    let t0 = Instant::now();
    for _ in 0..rounds {
        peer.launch_wide_async(&inc_k, 1, 256, &[ptr], &[n_small]).expect("async launch");
    }
    peer.sync_wide().expect("one sync");
    let batch_ms = t0.elapsed().as_secs_f64() * 1e3;
    let v_batch = read_first(&mut peer);

    println!("    per-call sync ({rounds} flushes) : {per_call_ms:>7.2} ms  (result {v_per})");
    println!("    async batch   (1 flush)          : {batch_ms:>7.2} ms  (result {v_batch})");
    println!("    batch speedup: {:.1}x", per_call_ms / batch_ms);
    assert_eq!(v_per, rounds as f32, "per-call chain correct");
    assert_eq!(v_batch, rounds as f32, "batch chain correct (same result)");

    // --- B. a wide op does not stall behind a resident poller ---
    // Kick the poller so a quantum is resident and busy-polling for
    // idle_exit_ns (40 ms). On the OLD shared-stream design a wide op
    // queued behind it and waited that long; on the dedicated wide
    // stream it runs immediately.
    println!("\n[B] wide op launched with a 40 ms-idle poller resident:");
    reset(&mut peer);
    let t = peer.submit(OP_NOP, &[]).expect("nop");
    assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("wait"), STATUS_DONE);
    peer.reap(t).expect("reap");
    let t0 = Instant::now();
    peer.launch_wide(&inc_k, 1, 256, &[ptr], &[n_small]).expect("small wide op");
    let no_stall_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("    small wide op ran in {no_stall_ms:.2} ms (idle_exit is 40 ms)");
    assert!(no_stall_ms < 10.0, "wide op must NOT wait for the poller quantum");
    println!("    -> dedicated wide stream: NO serialization behind the poller");

    // --- C. pause_poller frees SMs for a memory-bound wide op ---
    // A memory-bound op needs warps across every SM to saturate HBM.
    // The poller's resident blocks hold SMs it could use, so pausing
    // the poller returns them. This is the RBM-shaped case.
    let grid = 512u32;
    let iters = 120u32;
    println!("\n[C] memory-bound streaming op ({} MiB x {iters} passes, {grid} blocks),",
             block_bytes / (1024 * 1024));
    println!("    poller alive vs paused:");
    let run_stream = |peer: &mut GpuPeer, pause: bool| -> (f64, f32) {
        reset(peer);
        let t = peer.submit(OP_NOP, &[]).expect("nop");
        assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("wait"), STATUS_DONE);
        peer.reap(t).expect("reap");
        if pause {
            peer.pause_poller().expect("pause");
        }
        let t0 = Instant::now();
        peer.launch_wide(&stream_k, grid, 256, &[ptr], &[block_elems as u32, iters]).expect("stream");
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if pause {
            peer.resume_poller(); // resume before the doorbell fetch
        }
        (ms, read_first(peer))
    };
    let mut alive = f64::INFINITY;
    let mut paused = f64::INFINITY;
    let (mut va, mut vp) = (0f32, 0f32);
    for _ in 0..7 {
        let (a, x) = run_stream(&mut peer, false);
        let (p, y) = run_stream(&mut peer, true);
        alive = alive.min(a);
        paused = paused.min(p);
        va = x;
        vp = y;
    }
    println!("    poller alive  : {alive:>7.2} ms  (result {va:.1})");
    println!("    poller paused : {paused:>7.2} ms  (result {vp:.1})");
    let delta_pct = (alive - paused) / alive * 100.0;
    println!("    reclaimed by pausing: {:.2} ms ({delta_pct:+.1}%)", alive - paused);
    assert!((va - vp).abs() < 1.0, "result unchanged by pausing");

    peer.unpin(buf).expect("unpin");
    println!("\nVERIFIED: async batch pays one flush for a dependent chain; the dedicated");
    println!("wide stream keeps a wide op from stalling behind the poller; pausing the");
    println!("poller returns its SMs to a memory-bound wide op.");
}

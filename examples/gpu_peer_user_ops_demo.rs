//! User opcodes + zero-synchronization prefetch.
//!
//! (1) A user-written CUDA device function is NVRTC-composed into
//!     the poller at init and dispatched by doorbell like any
//!     built-in op - the GPU is a PROGRAMMABLE peer.
//! (2) `pin_prefetch` front-loads data with NO wait anywhere: lane
//!     FIFO order is the dependency order, so compute submitted
//!     immediately after the un-awaited upload is ordered after it
//!     for free.
//!
//! Run with:
//!   cargo run --release --features gpu-peer --example gpu_peer_user_ops_demo
//!
//! Requires the NVRTC runtime library (ships with the CUDA toolkit)
//! for the user-op composition; exits with a message otherwise.

use std::time::Duration;

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, RESIDENT_PARAMS_BYTES, STATUS_DONE};

/// op 100: scale every f32 of the resident block by the f32 arg at
/// payload+8, then write the block's new element[0] back as a
/// result at payload+12. Block-cooperative: all 256 threads call in.
const USER_OPS: &str = r#"
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    (void)team_rank; (void)team_size;
    if (op != 100u || block == (unsigned char*)0) return 1u;
    float k = *(volatile float*)payload;
    float* p = (float*)block;
    unsigned n = count / 4u;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x)
        p[i] = p[i] * k;
    __syncthreads();
    if (threadIdx.x == 0u)
        *(volatile float*)(payload + 4) = p[0];
    return 0u;
}
"#;

fn main() {
    println!("=== User opcodes + zero-sync prefetch ===\n");
    let mut peer = match GpuPeer::init(GpuPeerConfig {
        user_ops_cuda: Some(USER_OPS.to_string()),
        ..GpuPeerConfig::default()
    }) {
        Ok(p) => p,
        Err(e) => {
            println!("substrate unavailable on this host: {e}");
            return;
        }
    };
    println!("[1] user op NVRTC-composed into the poller at init: OK");

    // Zero-sync prefetch: upload is submitted and NOT awaited; the
    // user-op tasks queued right behind it on the same lane are
    // ordered after it by lane FIFO alone.
    let n_f32 = 500usize;
    let data: Vec<u8> = (0..n_f32).flat_map(|_| 2.0f32.to_le_bytes()).collect();
    let (handle, upload_t) = peer.pin_prefetch(&data).expect("prefetch");
    let mut tickets = Vec::new();
    for _ in 0..3 {
        // x2 three times: 2.0 -> 4 -> 8 -> 16, no wait between any of
        // these submissions or the upload.
        tickets.push(
            peer.submit_user(100, Some(&handle), &2.0f32.to_le_bytes()).expect("submit"),
        );
    }
    println!("[2] prefetch + 3 user tasks submitted back-to-back with ZERO waits");

    // Reap in lane order: upload first, then the three tasks.
    assert_eq!(peer.wait(upload_t, Duration::from_secs(5)).expect("up"), STATUS_DONE);
    peer.reap(upload_t).expect("reap upload first");
    let mut last_elem0 = 0f32;
    for (i, t) in tickets.into_iter().enumerate() {
        assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("wait"), STATUS_DONE,
                   "user task {i}");
        let mut buf = vec![0u8; RESIDENT_PARAMS_BYTES + 8];
        peer.read_result(t, &mut buf);
        last_elem0 = f32::from_le_bytes(buf[12..16].try_into().expect("4B"));
        peer.reap(t).expect("in order");
    }
    assert_eq!(last_elem0, 16.0, "2.0 x2 x2 x2 through the ordered chain");
    println!("[3] user-op result chain: element[0] = {last_elem0} (expected 16): VERIFIED");

    // Full verification through fetch.
    let mut out = vec![0u8; n_f32 * 4];
    peer.fetch(&handle, &mut out).expect("fetch");
    let bad = out
        .chunks_exact(4)
        .filter(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) != 16.0)
        .count();
    println!("[4] all {n_f32} resident elements == 16.0: {}",
             if bad == 0 { "VERIFIED" } else { "FAIL" });
    assert_eq!(bad, 0);
    peer.unpin(handle).expect("unpin");

    println!("\nVERIFIED: a caller-authored kernel ran as a doorbell opcode against");
    println!("prefetched resident data, ordered by the transport with no waits.");
}

//! The composed user-op module compiles with the caller's NVRTC options, and
//! a module already compiled in this process is loaded again rather than
//! recompiled.
//!
//! `0.1 * 10.0 - 1.0` is 0 when the product is rounded before the
//! subtraction, as the host computes it, and 2^-54 when a fused multiply-add
//! keeps the product exact. With `--fmad=false` the device must give the
//! host's answer.
//!
//! Requires a CUDA device and NVRTC. Every test holds the device lock, so
//! the process-wide compile count moves only for the test reading it.
#![cfg(feature = "gpu-peer")]

use std::time::Duration;

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, STATUS_DONE, layout};

mod common;

/// Payload: a, b, c as f64 at bytes 0, 8 and 16; the op writes a * b + c at
/// byte 24.
const MUL_ADD: &str = r#"
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    if (op != 700u) return 1u;
    if (threadIdx.x != 0) return 0u;
    volatile double* v = (volatile double*)payload;
    double a = v[0];
    double b = v[1];
    double c = v[2];
    v[3] = a * b + c;
    return 0u;
}
"#;

const OP_MUL_ADD: u32 = layout::OP_USER_BASE + 600;
const PREFIX: usize = layout::RESIDENT_PARAMS_BYTES;

fn peer(user_src: &str, options: &[&str]) -> GpuPeer {
    GpuPeer::init(GpuPeerConfig {
        user_ops_cuda: Some(user_src.to_string()),
        user_ops_nvrtc_options: options.iter().map(|o| o.to_string()).collect(),
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device and NVRTC are required for this test")
}

fn mul_add_on_device(peer: &mut GpuPeer, a: f64, b: f64, c: f64) -> f64 {
    let mut args = [0u8; 32];
    args[0..8].copy_from_slice(&a.to_le_bytes());
    args[8..16].copy_from_slice(&b.to_le_bytes());
    args[16..24].copy_from_slice(&c.to_le_bytes());
    let t = peer.submit_user(OP_MUL_ADD, None, &args).expect("submit");
    assert_eq!(peer.wait_status(t, Duration::from_secs(10)).expect("retires"), STATUS_DONE);
    let mut out = [0u8; PREFIX + 32];
    peer.read_result(t, &mut out).expect("the result fits the slot");
    peer.reap(t).expect("reap");
    let mut word = [0u8; 8];
    word.copy_from_slice(&out[PREFIX + 24..PREFIX + 32]);
    f64::from_le_bytes(word)
}

#[test]
fn a_module_built_without_fused_multiply_add_matches_the_host() {
    let _device = common::device();
    let (a, b, c) = (0.1f64, 10.0f64, -1.0f64);
    let host = a * b + c;
    let fused = a.mul_add(b, c);
    assert_ne!(host.to_bits(), fused.to_bits(), "the inputs must separate the two rounding orders");

    let mut unfused = peer(MUL_ADD, &["--fmad=false"]);
    let device = mul_add_on_device(&mut unfused, a, b, c);
    assert_eq!(
        device.to_bits(),
        host.to_bits(),
        "--fmad=false gave {device:e}; the host gives {host:e} and a fused multiply-add {fused:e}"
    );
}

#[test]
fn a_second_init_with_the_same_source_and_options_reuses_the_compiled_module() {
    let _device = common::device();
    let source = format!("{MUL_ADD}\n// compiled once per process by the cache test\n");

    let before = GpuPeer::user_ops_compiles();
    let first = peer(&source, &["--fmad=false"]);
    assert_eq!(GpuPeer::user_ops_compiles(), before + 1, "a first init compiles the module");
    drop(first);

    let mut second = peer(&source, &["--fmad=false"]);
    assert_eq!(GpuPeer::user_ops_compiles(), before + 1, "the same source and options reuse it");
    let (a, b, c) = (0.1f64, 10.0f64, -1.0f64);
    assert_eq!(mul_add_on_device(&mut second, a, b, c).to_bits(), (a * b + c).to_bits(), "the reused module runs");
    drop(second);

    let _other = peer(&source, &["--fmad=true"]);
    assert_eq!(GpuPeer::user_ops_compiles(), before + 2, "different options compile afresh");
}

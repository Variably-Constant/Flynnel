//! Every documented failure of the peer, exercised against a real
//! device rather than asserted from the source.
//!
//! The happy paths are covered by the parity suites and the demos.
//! These are the guards and the edges: a payload that does not fit, a
//! reap out of order, a wait that expires, a user op that reports an
//! error, an opcode below the user base, a lane count the header
//! cannot address, a pool too small for its buffer, a group with no
//! members, source that will not compile, a lane past the count, a
//! buffer longer or shorter than its block, and a ticket asked
//! whether it is done while it is not. A guard that has never been
//! seen to fire is indistinguishable from one that cannot.
//! Requires a CUDA device.
#![cfg(feature = "gpu-peer")]

use std::time::Duration;

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, GpuPeerError, STATUS_DONE, STATUS_ERR, layout};

mod common;

/// One device at a time, across test binaries as well as within this
/// one. See [`common`]: cargo runs the binaries concurrently, so a
/// per-binary mutex leaves the parity suites and the peer tests driving
/// the device together.
fn serial() -> common::DeviceLock {
    common::device()
}

fn peer() -> GpuPeer {
    GpuPeer::init(GpuPeerConfig::default()).expect("a CUDA device is required for this test")
}

/// op 101 reports an error; op 102 spins for the microseconds at
/// payload+8 so a wait can be made to expire on demand.
const USER_OPS: &str = r#"
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    (void)block; (void)count; (void)team_rank; (void)team_size;
    if (op == 101u) return 1u;
    if (op == 102u) {
        unsigned us = *(volatile unsigned*)payload;
        unsigned long long t0, now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        do {
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        } while (now - t0 < (unsigned long long)us * 1000ull);
        return 0u;
    }
    return 1u;
}
"#;

fn user_peer() -> GpuPeer {
    GpuPeer::init(GpuPeerConfig {
        user_ops_cuda: Some(USER_OPS.to_string()),
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device and NVRTC are required for this test")
}

#[test]
fn a_payload_larger_than_the_slot_is_refused_at_submit() {
    let _g = serial();
    let mut peer = peer();
    let max = peer.region().geometry().payload_max();
    let too_big = vec![0u8; max + 1];
    match peer.submit(layout::OP_SUM_U32, &too_big) {
        Err(GpuPeerError::PayloadTooLarge { len, max: reported }) => {
            assert_eq!(len, max + 1, "the rejection names the length it refused");
            assert_eq!(reported, max, "and the capacity it measured against");
        }
        other => panic!("an oversized payload must be refused, got {other:?}"),
    }
    // A payload of exactly the capacity is accepted, so the guard is
    // not simply refusing everything near the boundary.
    let exact = peer.submit(layout::OP_SUM_U32, &vec![0u8; max]).expect("the capacity fits");
    assert_eq!(peer.wait(exact, Duration::from_secs(5)).expect("completes"), STATUS_DONE);
    peer.reap(exact).expect("reap");
}

#[test]
fn reaping_out_of_order_names_both_sequences() {
    let _g = serial();
    let mut peer = user_peer();
    // Both on one lane, since the ordering contract is per lane.
    let first = peer.submit_user_on_lane(102, None, &0u32.to_le_bytes(), 0).expect("first");
    let second = peer.submit_user_on_lane(102, None, &0u32.to_le_bytes(), 0).expect("second");
    peer.wait(second, Duration::from_secs(5)).expect("both complete");
    match peer.reap(second) {
        Err(GpuPeerError::ReapOutOfOrder { lane, expected, got }) => {
            assert_eq!(lane, 0);
            assert_eq!(expected, first.seq, "the lane's oldest unreaped");
            assert_eq!(got, second.seq, "the one the caller tried");
        }
        other => panic!("reaping the second first must be refused, got {other:?}"),
    }
    peer.reap(first).expect("in order");
    peer.reap(second).expect("now in order");
}

#[test]
fn a_wait_that_expires_reports_a_timeout_and_the_slot_still_completes() {
    let _g = serial();
    let mut peer = user_peer();
    // 200 ms of device-side spin against a 5 ms wait.
    let t = peer.submit_user(102, None, &200_000u32.to_le_bytes()).expect("submit spin");
    match peer.wait(t, Duration::from_millis(5)) {
        Err(GpuPeerError::Timeout) => {}
        other => panic!("a wait shorter than the op must expire, got {other:?}"),
    }
    // The slot is still in flight, not lost: waiting properly completes
    // it, which is what makes a timeout recoverable rather than fatal.
    assert_eq!(
        peer.wait(t, Duration::from_secs(5)).expect("the op finishes"),
        STATUS_DONE
    );
    peer.reap(t).expect("reap");
}

#[test]
fn a_user_op_reporting_an_error_reaches_the_caller_as_one() {
    let _g = serial();
    let mut peer = user_peer();
    let t = peer.submit_user(101, None, &[]).expect("submit");
    let err = peer
        .wait(t, Duration::from_secs(5))
        .expect_err("a user op returning non-zero must not answer Ok");
    assert!(!matches!(err, GpuPeerError::Timeout), "it completed, it did not hang: {err}");
    assert_eq!(
        peer.wait_status(t, Duration::from_secs(5)).expect("status is readable"),
        STATUS_ERR,
        "and the raw word says which failure it was"
    );
    peer.reap(t).expect("reap");
}

#[test]
fn a_user_opcode_below_the_user_base_is_refused_at_submit() {
    let _g = serial();
    let mut peer = peer();
    match peer.submit_user(layout::OP_USER_BASE - 1, None, &[]) {
        Err(GpuPeerError::Unavailable(_)) => {}
        other => panic!("a user op below OP_USER_BASE must be refused, got {other:?}"),
    }
}

#[test]
fn more_lanes_than_the_header_addresses_is_refused_at_init() {
    let _g = serial();
    let cfg = GpuPeerConfig {
        lanes: (layout::MAX_POLLER_LANES + 1) as u32,
        ..GpuPeerConfig::default()
    };
    match GpuPeer::init(cfg) {
        Err(GpuPeerError::Unavailable(_)) => {}
        Err(e) => panic!("a lane count past MAX_POLLER_LANES must be refused as unavailable: {e}"),
        Ok(_) => panic!("a lane count past MAX_POLLER_LANES must be refused"),
    }
    // The boundary itself is accepted, so the guard is off by nothing.
    assert!(
        GpuPeer::init(GpuPeerConfig {
            lanes: layout::MAX_POLLER_LANES as u32,
            ..GpuPeerConfig::default()
        })
        .is_ok(),
        "the documented maximum must itself be usable"
    );
}

#[test]
fn fetching_clamps_to_the_shorter_of_the_buffer_and_the_block() {
    let _g = serial();
    let mut peer = peer();
    let data = vec![7u8; 2048];
    let handle = peer.pin(&data).expect("pin");

    // The contract is min(out.len(), handle.len()): a longer buffer is
    // filled to the block's length and its tail is left as the caller
    // had it, rather than being refused or filled with device memory
    // past the pin.
    let mut longer = vec![0xABu8; data.len() + 64];
    peer.fetch(&handle, &mut longer).expect("a longer buffer is accepted");
    assert_eq!(&longer[..data.len()], &data[..], "the block's bytes land at the front");
    assert!(
        longer[data.len()..].iter().all(|&b| b == 0xAB),
        "and the tail past the block is untouched, not zeroed or filled from the device"
    );

    // A shorter buffer takes a prefix, so the clamp works from both
    // sides rather than only guarding the long case.
    let mut shorter = vec![0u8; 512];
    peer.fetch(&handle, &mut shorter).expect("a shorter buffer is accepted");
    assert_eq!(shorter, data[..512], "the prefix is what a short buffer receives");

    let mut exact = vec![0u8; data.len()];
    peer.fetch(&handle, &mut exact).expect("the matching size works");
    assert_eq!(exact, data, "and the matching size returns what was pinned");
    peer.unpin(handle).expect("unpin");
}

#[test]
fn pin_is_bounded_by_the_slot_and_pin_bulk_is_not() {
    let _g = serial();
    let mut peer = peer();
    // `pin` ships the bytes through one slot, so its ceiling is the
    // slot's payload less the resident-op parameter header.
    let over = vec![3u8; peer.region().geometry().payload_max()];
    match peer.pin(&over) {
        Err(GpuPeerError::PayloadTooLarge { .. }) => {}
        other => panic!("pin past the slot capacity must be refused, got {other:?}"),
    }
    // The same buffer through the bulk path, which does not go slot by
    // slot, and back again byte for byte.
    let handle = peer.pin_bulk(&over).expect("pin_bulk takes what pin refused");
    let mut back = vec![0u8; over.len()];
    peer.fetch_bulk(&handle, &mut back).expect("fetch_bulk");
    assert_eq!(back, over, "the bulk round trip is byte-exact");
    peer.unpin(handle).expect("unpin");
}

#[test]
fn a_resident_op_mutates_the_block_and_the_handle_addresses_it() {
    let _g = serial();
    let mut peer = peer();
    let floats: Vec<u8> = (0..256).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    let handle = peer.pin(&floats).expect("pin");

    let (ptr, len) = peer.resident_ptr(&handle).expect("resident_ptr");
    assert_ne!(ptr, 0, "a pinned block has a device address");
    assert!(len >= floats.len(), "and covers at least what was pinned");

    let t = peer.submit_resident(layout::OP_ADD1_F32_V, &handle).expect("submit");
    assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("completes"), STATUS_DONE);
    assert!(peer.is_done(t), "a completed ticket reports done");
    peer.reap(t).expect("reap");

    let mut back = vec![0u8; floats.len()];
    peer.fetch(&handle, &mut back).expect("fetch");
    let got: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(
        got.iter().all(|&x| x == 2.0),
        "every f32 of the resident block advanced by one"
    );
    peer.unpin(handle).expect("unpin");
}

#[test]
fn write_resident_bulk_replaces_the_block_in_place() {
    let _g = serial();
    let mut peer = peer();
    let first = vec![1u8; 8192];
    let handle = peer.pin_bulk(&first).expect("pin_bulk");
    let second = vec![9u8; 8192];
    peer.write_resident_bulk(&handle, &second).expect("write_resident_bulk");
    let mut back = vec![0u8; second.len()];
    peer.fetch_bulk(&handle, &mut back).expect("fetch_bulk");
    assert_eq!(back, second, "the second write is what comes back");
    peer.unpin(handle).expect("unpin");
}

#[test]
fn a_prefetched_handle_orders_work_behind_the_upload_without_a_wait() {
    let _g = serial();
    let mut peer = peer();
    let data: Vec<u8> = (0..256).flat_map(|_| 2.0f32.to_le_bytes()).collect();
    let (handle, upload) = peer.pin_prefetch(&data).expect("pin_prefetch");
    // Submitted with no wait on the upload: lane order is the
    // dependency order.
    let t = peer.submit_resident(layout::OP_ADD1_F32_V, &handle).expect("submit");
    assert_eq!(peer.wait(upload, Duration::from_secs(5)).expect("upload"), STATUS_DONE);
    peer.reap(upload).expect("reap upload first");
    assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("op"), STATUS_DONE);
    peer.reap(t).expect("reap in order");

    let mut back = vec![0u8; data.len()];
    peer.fetch(&handle, &mut back).expect("fetch");
    let got: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(
        got.iter().all(|&x| x == 3.0),
        "the op ran after the upload, so 2.0 became 3.0 rather than 1.0"
    );
    peer.unpin(handle).expect("unpin");
}

#[test]
fn a_paused_poller_resumes_and_serves_again() {
    let _g = serial();
    let mut peer = peer();
    let before = peer.submit(layout::OP_NOP, &[]).expect("submit");
    assert_eq!(peer.wait(before, Duration::from_secs(5)).expect("completes"), STATUS_DONE);
    peer.reap(before).expect("reap");

    peer.pause_poller().expect("pause");
    peer.resume_poller();

    let after = peer.submit(layout::OP_NOP, &[]).expect("submit after resume");
    assert_eq!(
        peer.wait(after, Duration::from_secs(5)).expect("a resumed poller serves"),
        STATUS_DONE
    );
    peer.reap(after).expect("reap");
}

#[test]
fn a_pool_too_small_for_the_buffer_refuses_the_pin() {
    let _g = serial();
    let mut peer = GpuPeer::init(GpuPeerConfig {
        vram_block_bytes: 4096,
        vram_blocks: 4,
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device is required for this test");
    // Four blocks of 4 KiB is a 16 KiB pool.
    assert!(
        peer.pin_bulk(&vec![5u8; 64 * 1024]).is_err(),
        "a buffer past the whole pool must be refused rather than truncated"
    );
    // The pool is still usable afterwards, so the refusal did not
    // leave blocks claimed.
    let handle = peer.pin_bulk(&vec![5u8; 8192]).expect("a fitting buffer still pins");
    peer.unpin(handle).expect("unpin");
}

#[test]
fn a_group_with_no_peers_is_refused() {
    let _g = serial();
    match flynnel::gpu_peer::PeerGroup::init(Vec::new()) {
        Err(GpuPeerError::Unavailable(_)) => {}
        Err(e) => panic!("an empty group must be refused as unavailable: {e}"),
        Ok(_) => panic!("an empty group must be refused: a group with no members skews placement"),
    }
    // One peer is a group, so the refusal is about emptiness and not
    // about a minimum size the caller cannot meet.
    let group = flynnel::gpu_peer::PeerGroup::init(vec![GpuPeerConfig::default()])
        .expect("a single-peer group is valid");
    assert_eq!(group.len(), 1);
    assert!(!group.is_empty());
    assert_eq!(group.calibrations().len(), 1, "one calibration per member");
}

#[test]
fn a_wide_kernel_that_does_not_compile_is_refused() {
    let _g = serial();
    let peer = peer();
    match peer.compile_wide_kernel("this is not CUDA", "nope") {
        Err(_) => {}
        Ok(_) => panic!("source that cannot compile must not yield a kernel"),
    }
    // And a kernel whose entry point is absent from valid source.
    let valid = r#"extern "C" __global__ void present(float* p) { (void)p; }"#;
    match peer.compile_wide_kernel(valid, "absent") {
        Err(_) => {}
        Ok(_) => panic!("an entry point that is not in the source must be refused"),
    }
}

#[test]
fn a_wide_kernel_runs_across_the_grid_over_a_resident_block() {
    let _g = serial();
    let mut peer = peer();
    let n = 1024usize;
    let data: Vec<u8> = (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    let handle = peer.pin_bulk(&data).expect("pin_bulk");
    let (ptr, _len) = peer.resident_ptr(&handle).expect("resident_ptr");

    let src = r#"
extern "C" __global__ void addk(float* p, unsigned n, unsigned k) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        p[i] += (float)k;
    }
}
"#;
    let kernel = peer.compile_wide_kernel(src, "addk").expect("compile");
    peer.launch_wide(&kernel, 32, 256, &[ptr], &[n as u32, 4]).expect("launch_wide");

    let mut back = vec![0u8; data.len()];
    peer.fetch_bulk(&handle, &mut back).expect("fetch_bulk");
    let got: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got.len(), n);
    assert!(
        got.iter().all(|&x| x == 5.0),
        "every element of the resident block saw the wide kernel exactly once"
    );
    peer.unpin(handle).expect("unpin");
}

#[test]
fn the_timed_lock_round_trips_when_the_self_test_granted_it() {
    let _g = serial();
    let peer = peer();
    let cal = peer.calibration();
    if !cal.timed_lock_ok {
        // The grant is a property of the host, not of this test: the
        // self-test must have observed contention on both sides.
        return;
    }
    peer.timed_lock_acquire(Duration::from_secs(2)).expect("acquire on a granted host");
    peer.timed_lock_release();
    peer.timed_lock_acquire(Duration::from_secs(2))
        .expect("a released lock can be taken again");
    peer.timed_lock_release();
}

#[test]
fn calibration_reports_measured_values_rather_than_zeros() {
    let _g = serial();
    let peer = peer();
    let cal = peer.calibration();
    assert!(cal.launch_ns > 0, "a measured launch baseline is positive");
    assert!(cal.rtt_min_ns > 0, "a measured doorbell round trip is positive");
    assert!(
        cal.rtt_median_ns >= cal.rtt_min_ns,
        "the median cannot sit below the minimum: {} < {}",
        cal.rtt_median_ns,
        cal.rtt_min_ns
    );
    assert!(
        cal.rtt_p99_ns >= cal.rtt_median_ns,
        "nor the p99 below the median: {} < {}",
        cal.rtt_p99_ns,
        cal.rtt_median_ns
    );
    if cal.timed_lock_ok {
        assert!(
            cal.lock_cpu_contended > 0 && cal.lock_gpu_contended > 0,
            "a granted timed lock must carry contention evidence on both sides"
        );
    }
}

#[test]
fn a_lane_past_the_count_wraps_rather_than_failing() {
    let _g = serial();
    let mut peer = user_peer();
    let lanes = peer.region().geometry().lanes;
    let t = peer
        .submit_user_on_lane(102, None, &0u32.to_le_bytes(), lanes + 3)
        .expect("a lane past the count is accepted");
    assert_eq!(t.lane, (lanes + 3) % lanes, "and lands on the wrapped lane");
    assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("completes"), STATUS_DONE);
    peer.reap(t).expect("reap");
}

#[test]
fn a_ticket_is_not_done_while_its_op_is_still_running() {
    let _g = serial();
    let mut peer = user_peer();
    // 300 ms of device-side spin: long enough that the check below is
    // not racing the completion.
    let t = peer.submit_user(102, None, &300_000u32.to_le_bytes()).expect("submit");
    assert!(!peer.is_done(t), "an op still spinning is not done");
    assert_eq!(peer.wait(t, Duration::from_secs(5)).expect("completes"), STATUS_DONE);
    assert!(peer.is_done(t), "and is done once it has finished");
    peer.reap(t).expect("reap");
}

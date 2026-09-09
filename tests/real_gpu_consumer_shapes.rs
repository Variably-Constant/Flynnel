//! Peer configurations taken from consumers that exist, at the settings
//! they ship.
//!
//! The GPU-peer suite covers the surface a feature at a time: teams,
//! wide kernels, resident blocks, the poller's pause and resume. What
//! it did not cover is any one consumer's whole arrangement of them,
//! and a consumer does not use one feature at a time. Each test below
//! stands up the configuration a real call site passes, names the file
//! it came from, and exercises the combination that site depends on.
//!
//! The three shapes surveyed are deliberately unalike, and are
//! described by what they do rather than named, because the
//! arrangement is the reusable part:
//!
//! - A desktop compositor's force-field op runs a 64-block team per
//!   lane over quarter-megabyte slots and reads its answer back through
//!   the same doorbell payload it submitted.
//! - A full-text index scanner pins a corpus in bulk and sweeps it with
//!   a wide kernel, never touching the doorbell.
//! - A vision filter bank also uses only wide launches, and parks the
//!   poller at 100 us on purpose: a resident poller stalls its device
//!   under WDDM and hangs the run.
//!
//! The third is the one worth stating plainly, because it is a
//! dependency on something not happening. If a wide launch ever came to
//! need a live poller, every other test here would still pass and that
//! consumer would hang.
//!
//! Requires a CUDA device and NVRTC.
#![cfg(feature = "gpu-peer")]

use std::time::Duration;

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, STATUS_DONE, layout};

mod common;

/// One device at a time, across test binaries as well as within this
/// one; cargo runs the binaries concurrently, and these shapes are the
/// ones whose timings a neighbour distorts most.
fn serial() -> common::DeviceLock {
    common::device()
}

/// The field op's shape: read the point count out of the payload
/// header, then have every rank of the team stride over the points and
/// write one output plane, so a hole in the answer means a rank was
/// missing when the slot retired.
///
/// The layout mirrors the consumer's: a 20-byte header, then `n`
/// input floats, then room for `n` outputs.
const FIELD_OP: &str = r#"
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    (void)block; (void)count;
    if (op != 200u) return 1u;
    volatile unsigned* head = (volatile unsigned*)payload;
    unsigned n = head[0];
    volatile float* xin = (volatile float*)(payload + 20);
    volatile float* xout = (volatile float*)(payload + 20 + n * 4u);
    unsigned lane = team_rank * blockDim.x + threadIdx.x;
    unsigned stride = team_size * blockDim.x;
    for (unsigned i = lane; i < n; i += stride) {
        xout[i] = xin[i] * 2.0f + 1.0f;
    }
    return 0u;
}
"#;

/// A compositor's force-field op: a 64-block team per lane, two lanes,
/// slots sized to hold a whole coordinate plane, and the answer read
/// back out of the submitted payload.
///
/// The team size is the part that matters: it is eight times the
/// largest any other test configures, and rank 0's barrier deadline is
/// what decides whether a team that wide retires whole.
#[test]
fn a_wide_team_reads_its_answer_back_from_the_submitted_payload() {
    let _g = serial();
    const N: usize = 4096;
    const HEADER: usize = 20;
    // The submitted args are the device's payload exactly, but a read
    // hands back the whole slot payload, which opens with the
    // resident-parameter block the kernel skipped. So a host offset is
    // a device offset on the way in and this much more on the way back.
    const RESIDENT_PREFIX: usize = 8;
    // The header, the input plane, the output plane, the prefix a read
    // adds, and the slack the consumer leaves above them.
    let slot = (RESIDENT_PREFIX + HEADER + N * 4 * 2 + 4096) as u32;

    let mut peer = GpuPeer::init(GpuPeerConfig {
        lanes: 2,
        slot_bytes: slot,
        slots_per_lane: 4,
        blocks_per_lane: 64,
        user_ops_cuda: Some(FIELD_OP.to_string()),
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device and NVRTC are required for this test");

    let mut args = vec![0u8; HEADER + N * 4 * 2];
    args[..4].copy_from_slice(&(N as u32).to_le_bytes());
    for (i, chunk) in args[HEADER..HEADER + N * 4].chunks_exact_mut(4).enumerate() {
        chunk.copy_from_slice(&(i as f32).to_le_bytes());
    }

    let t = peer
        .submit_user(layout::OP_USER_BASE + 100, None, &args)
        .expect("submit the field op");
    assert_eq!(
        peer.wait_status(t, Duration::from_secs(10)).expect("the op completes"),
        STATUS_DONE,
        "a 64-block team did not retire whole. STATUS_TEAM_INCOMPLETE here \
         means the team did not assemble inside its barrier deadline, which \
         on a device another process is loading is a condition of the host \
         rather than a defect; the measured cost of assembling 64 blocks is \
         53 us against a 5 ms deadline, so check what else holds the device \
         before reading this as a regression"
    );

    // The consumer reads the result back over the region it submitted,
    // sized to hold the prefix as well.
    let mut res = vec![0u8; RESIDENT_PREFIX + args.len()];
    peer.read_result(t, &mut res).expect("the result fits the slot");
    peer.reap(t).expect("reap");

    let out = &res[RESIDENT_PREFIX + HEADER + N * 4..];
    for i in 0..N {
        let o = i * 4;
        let got = f32::from_le_bytes([out[o], out[o + 1], out[o + 2], out[o + 3]]);
        let want = i as f32 * 2.0 + 1.0;
        assert_eq!(
            got, want,
            "point {i} of {N} was not written. A 64-block team that retires \
             early leaves exactly this: a plane correct where some ranks \
             ran and stale where others did not"
        );
    }

}

/// A full-text scanner's corpus: two lanes of 64 KB slots, an arena
/// pinned with `pin_bulk` in three pieces, and a wide kernel sweeping
/// all of them.
///
/// `pin_bulk` is the part under test: the arena is far larger than a
/// slot, so the bounded `pin` would refuse it.
#[test]
fn a_corpus_pinned_in_bulk_is_swept_by_a_wide_kernel() {
    let _g = serial();
    let mut peer = GpuPeer::init(GpuPeerConfig {
        lanes: 2,
        slot_bytes: 64 * 1024,
        slots_per_lane: 8,
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device is required for this test");

    // An arena several slots wide, plus the two index arrays that ride
    // alongside it, matching the scanner's three-handle arrangement.
    let entries = 4096usize;
    let arena: Vec<u8> = (0..entries * 64).map(|i| (i % 251) as u8).collect();
    let offsets: Vec<u8> =
        (0..entries).flat_map(|i| ((i * 64) as u32).to_le_bytes()).collect();
    let lens: Vec<u8> = (0..entries).flat_map(|_| 64u32.to_le_bytes()).collect();
    assert!(
        arena.len() > 64 * 1024,
        "the arena must exceed a slot, or this is not testing the bulk path"
    );

    let h_arena = peer.pin_bulk(&arena).expect("the arena pins in bulk");
    let h_offsets = peer.pin_bulk(&offsets).expect("offsets pin");
    let h_lens = peer.pin_bulk(&lens).expect("lens pin");
    let (p_arena, _) = peer.resident_ptr(&h_arena).expect("arena ptr");
    let (p_offsets, _) = peer.resident_ptr(&h_offsets).expect("offsets ptr");
    let (p_lens, _) = peer.resident_ptr(&h_lens).expect("lens ptr");

    // One thread per entry, reading through the offset and length
    // arrays the way the scanner's kernel does, and writing a per-entry
    // answer back over the lens array.
    let src = r#"
extern "C" __global__ void sum_entries(
    unsigned char* arena, unsigned* offsets, unsigned* lens, unsigned n)
{
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        unsigned off = offsets[i], len = lens[i], acc = 0;
        for (unsigned j = 0; j < len; ++j) acc += arena[off + j];
        lens[i] = acc;
    }
}
"#;
    let kernel = peer.compile_wide_kernel(src, "sum_entries").expect("compile");
    peer.launch_wide(&kernel, 64, 256, &[p_arena, p_offsets, p_lens], &[entries as u32])
        .expect("launch_wide over the pinned corpus");

    let mut back = vec![0u8; lens.len()];
    peer.fetch_bulk(&h_lens, &mut back).expect("fetch_bulk");
    for i in 0..entries {
        let o = i * 4;
        let got = u32::from_le_bytes([back[o], back[o + 1], back[o + 2], back[o + 3]]);
        let want: u32 = (0..64).map(|j| ((i * 64 + j) % 251) as u32).sum();
        assert_eq!(got, want, "entry {i} of {entries} did not read its own slice of the arena");
    }

    peer.unpin(h_arena).expect("unpin arena");
    peer.unpin(h_offsets).expect("unpin offsets");
    peer.unpin(h_lens).expect("unpin lens");
}

/// A vision filter bank: a short `idle_exit_ns` so the poller parks
/// almost at once, and then only wide launches.
///
/// That consumer's source is explicit that a resident poller starves
/// its device under WDDM and hangs the run, so parking is not a tuning
/// preference there - it is the reason the path works at all.
///
/// The poller is left to park on its own rather than paused, because
/// that is what the consumer does - it sets the idle window and never
/// calls `pause_poller`. Both launches are separated by a wait far
/// longer than that window, so a wide launch that quietly depended on a
/// live poller cannot pass by catching one still warm.
#[test]
fn a_wide_launch_runs_with_the_poller_parked() {
    let _g = serial();
    let n = 2048usize;
    let mut peer = GpuPeer::init(GpuPeerConfig {
        slots_per_lane: 4,
        idle_exit_ns: 100_000,
        vram_block_bytes: (n * 4) as u32,
        vram_blocks: 16,
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device is required for this test");

    let data: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let handle = peer.pin_bulk(&data).expect("pin_bulk");
    let (ptr, _) = peer.resident_ptr(&handle).expect("resident_ptr");

    let src = r#"
extern "C" __global__ void scale(float* p, unsigned n) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        p[i] = p[i] * 2.0f;
    }
}
"#;
    let kernel = peer.compile_wide_kernel(src, "scale").expect("compile");

    // Five hundred times the 100 us idle window, so whatever the pin
    // woke has long since parked.
    std::thread::sleep(Duration::from_millis(50));
    peer.launch_wide(&kernel, 16, 256, &[ptr], &[n as u32])
        .expect("a wide launch must not need a live poller");
    std::thread::sleep(Duration::from_millis(50));
    peer.launch_wide(&kernel, 16, 256, &[ptr], &[n as u32])
        .expect("a second wide launch after the idle window must still run");

    let mut back = vec![0u8; data.len()];
    peer.fetch_bulk(&handle, &mut back).expect("fetch_bulk");
    for i in 0..n {
        let o = i * 4;
        let got = f32::from_le_bytes([back[o], back[o + 1], back[o + 2], back[o + 3]]);
        assert_eq!(
            got,
            i as f32 * 4.0,
            "element {i} did not see both launches; a wide launch that \
             needs a live poller hangs the consumer that parks one"
        );
    }
    peer.unpin(handle).expect("unpin");
}

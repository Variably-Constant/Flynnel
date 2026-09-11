//! Waves built from the `flw_` helpers run every segment exactly once,
//! whichever way the frontier is kept and however the slices are paced.
//!
//! The op walks a binary tree of segments. A segment's id carries its depth
//! in the top byte and a serial in the low 24 bits, and a segment above the
//! depth limit pushes two children. Every segment counts itself and adds
//! its id to a checksum, both in the slot payload, so a segment run twice
//! or missed shows up against the count and sum the host computes by
//! walking the same tree.
//!
//! Requires a CUDA device and NVRTC.
#![cfg(feature = "gpu-peer")]

use std::num::NonZeroU32;
use std::time::Duration;

use flynnel::gpu_peer::wave::{
    FAIL_ARENA, FAIL_IDS, Frontier, Resume, SliceBudget, SliceState, Wave, WaveSpec,
};
use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, STATUS_DONE, STATUS_ERR, layout};

mod common;

/// Payload words: 0 depth limit, 1 id that fails with code 7 (or
/// 0xFFFFFFFF), 2 ns of work block 0 spends per generation, 3 segments
/// counted, 4 checksum of ids. Op 401 also reserves 12 arena bytes per
/// segment and writes the id there.
const OPS: &str = r#"
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    if (op != 400u && op != 401u) return 1u;
    volatile unsigned* args = (volatile unsigned*)payload;
    u32 limit = args[0];
    u32 fail_at = args[1];
    u32 spin_ns = args[2];
    u32* tally = (u32*)(payload + 12);

    flw_slice s;
    u32 bad = flw_slice_begin(&s, block, count, team_rank, team_size);
    if (bad != 0u) return bad;
    while (s.running) {
        for (u32 k = flw_first(&s); k < s.end; k += flw_stride(&s)) {
            u32 id = flw_id(&s, k);
            u32 depth = id >> 24;
            atomicAdd(tally, 1u);
            atomicAdd(tally + 1, id);
            if (id == fail_at) flw_fail(&s, id, 7u);
            if (op == 401u) {
                u32 at = flw_alloc(&s, 12u);
                if (at != 0xFFFFFFFFu) *(u32*)(block + at) = id;
            }
            if (depth < limit) {
                u32 serial = (id & 0xFFFFFFu) << 1;
                flw_push(&s, ((depth + 1u) << 24) | (serial & 0xFFFFFFu));
                flw_push(&s, ((depth + 1u) << 24) | ((serial | 1u) & 0xFFFFFFu));
            }
        }
        if (spin_ns != 0u && team_rank == 0u && threadIdx.x == 0) {
            u64 t0 = gtimer();
            while (gtimer() - t0 < (u64)spin_ns) {
            }
        }
        flw_generation_end(&s);
    }
    return flw_slice_end(&s);
}
"#;

const OP_TREE: u32 = layout::OP_USER_BASE + 300;
const OP_TREE_ARENA: u32 = layout::OP_USER_BASE + 301;
const PREFIX: usize = layout::RESIDENT_PARAMS_BYTES;
const NO_FAIL: u32 = u32::MAX;

fn peer(blocks_per_lane: u32) -> GpuPeer {
    GpuPeer::init(GpuPeerConfig {
        user_ops_cuda: Some(OPS.to_string()),
        blocks_per_lane,
        vram_block_bytes: 65_536,
        vram_blocks: 256,
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device and NVRTC are required for this test")
}

fn spec(frontier: Frontier, resume: Resume, budget: SliceBudget, roots: Vec<u32>, id_capacity: u32) -> WaveSpec {
    WaveSpec {
        roots,
        id_capacity,
        arena_bytes: 0,
        frontier,
        resume,
        slice_budget: budget,
        barrier_deadline: Duration::from_millis(50),
        done_deadline: Some(Duration::from_millis(500)),
        longest_generation_seed: Duration::ZERO,
    }
}

/// Every id of the trees under `roots` down to `limit`: how many, and their
/// wrapping sum.
fn expected(roots: &[u32], limit: u32) -> (u32, u32) {
    let mut count = 0u32;
    let mut sum = 0u32;
    let mut frontier: Vec<u32> = roots.to_vec();
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for id in frontier {
            count += 1;
            sum = sum.wrapping_add(id);
            let depth = id >> 24;
            if depth < limit {
                let serial = (id & 0xFF_FFFF) << 1;
                next.push(((depth + 1) << 24) | (serial & 0xFF_FFFF));
                next.push(((depth + 1) << 24) | ((serial | 1) & 0xFF_FFFF));
            }
        }
        frontier = next;
    }
    (count, sum)
}

fn args(limit: u32, fail_at: u32, spin_ns: u32) -> [u8; 20] {
    let mut a = [0u8; 20];
    a[0..4].copy_from_slice(&limit.to_le_bytes());
    a[4..8].copy_from_slice(&fail_at.to_le_bytes());
    a[8..12].copy_from_slice(&spin_ns.to_le_bytes());
    a
}

/// Submit one slice, wait for it to retire, and return its status with the
/// count and checksum it recorded.
fn slice(peer: &mut GpuPeer, wave: &Wave, op: u32, a: &[u8; 20]) -> (u32, u32, u32) {
    let t = peer.submit_wave(wave, op, a).expect("submit a slice");
    let status = peer.wait_status(t, Duration::from_secs(60)).expect("the slice retires");
    let mut out = [0u8; PREFIX + 20];
    peer.read_result(t, &mut out).expect("the result fits the slot");
    peer.reap(t).expect("reap");
    let word = |i: usize| u32::from_le_bytes([out[PREFIX + i], out[PREFIX + i + 1], out[PREFIX + i + 2], out[PREFIX + i + 3]]);
    (status, word(12), word(16))
}

fn roots(n: u32) -> Vec<u32> {
    (0..n).collect()
}

/// A wave run to completion on `blocks` blocks visits every segment once.
fn runs_every_segment_once(blocks: u32, frontier: Frontier, root_count: u32, limit: u32, id_capacity: u32) {
    let _device = common::device();
    let mut peer = peer(blocks);
    let r = roots(root_count);
    let wave = peer
        .create_wave(&spec(frontier, Resume::Device, SliceBudget::Unbounded, r.clone(), id_capacity))
        .expect("create the wave");
    let (status, count, sum) = slice(&mut peer, &wave, OP_TREE, &args(limit, NO_FAIL, 0));
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert_eq!(status, STATUS_DONE, "{frontier:?} on {} blocks: {stats:?}", peer.team_size());
    assert_eq!((count, sum), expected(&r, limit), "{frontier:?}: every segment exactly once");
    assert_eq!(stats.slice_state, SliceState::Finished);
    assert_eq!(stats.failure, None);
    assert_eq!(stats.barrier_timeouts, 0);
    peer.release_wave(wave).expect("release the span");
}

#[test]
fn a_global_wave_on_one_block_runs_every_segment_once() {
    runs_every_segment_once(1, Frontier::Global, 3, 8, 4096);
}

#[test]
fn a_global_wave_on_a_team_runs_every_segment_once() {
    runs_every_segment_once(4, Frontier::Global, 3, 8, 4096);
}

#[test]
fn a_partition_that_never_rebalances_runs_every_segment_once() {
    runs_every_segment_once(4, Frontier::Partition { rebalance_every: None }, 8, 7, 16_384);
}

/// Every root dealt to the last block, so the other blocks idle until a
/// rebalance deals them work, which the recorded imbalance shows.
#[test]
fn a_partition_that_rebalances_runs_every_segment_once_and_records_imbalance() {
    let _device = common::device();
    let mut peer = peer(4);
    let r = roots(1);
    let every = NonZeroU32::new(2).expect("nonzero");
    let wave = peer
        .create_wave(&spec(
            Frontier::Partition { rebalance_every: Some(every) },
            Resume::Device,
            SliceBudget::Unbounded,
            r.clone(),
            65_536,
        ))
        .expect("create the wave");
    let (status, count, sum) = slice(&mut peer, &wave, OP_TREE, &args(10, NO_FAIL, 0));
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert_eq!(status, STATUS_DONE, "{stats:?}");
    assert_eq!((count, sum), expected(&r, 10));
    assert!(stats.rebalances >= 1, "{stats:?}");
    if peer.team_size() > 1 {
        assert!(
            stats.imbalance_per_mille > 1000,
            "one block held every pending segment before the first rebalance: {stats:?}"
        );
    }
    peer.release_wave(wave).expect("release the span");
}

/// A budget smaller than the wave makes slices stop early and yield, and
/// the wave still runs every segment exactly once across them.
#[test]
fn a_wave_that_yields_across_slices_runs_every_segment_once() {
    let _device = common::device();
    let mut peer = peer(1);
    let r = roots(1);
    let wave = peer
        .create_wave(&spec(
            Frontier::Global,
            Resume::Device,
            SliceBudget::Fixed(Duration::from_millis(20)),
            r.clone(),
            65_536,
        ))
        .expect("create the wave");
    let (status, count, sum) = slice(&mut peer, &wave, OP_TREE, &args(14, NO_FAIL, 2_000_000));
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert_eq!(status, STATUS_DONE, "{stats:?}");
    assert_eq!((count, sum), expected(&r, 14));
    assert!(stats.yields >= 1, "15 generations of 2 ms each cannot fit a 20 ms slice: {stats:?}");
    assert_eq!(stats.slice_state, SliceState::Finished);
    peer.release_wave(wave).expect("release the span");
}

/// With host resume, each slice retires and reports Continue until the wave
/// finishes, and the slices together run every segment once.
#[test]
fn a_host_continued_wave_runs_every_segment_once_across_submissions() {
    let _device = common::device();
    let mut peer = peer(4);
    let r = roots(1);
    let wave = peer
        .create_wave(&spec(
            Frontier::Global,
            Resume::Host,
            SliceBudget::Fixed(Duration::from_millis(20)),
            r.clone(),
            65_536,
        ))
        .expect("create the wave");
    let mut count = 0u32;
    let mut sum = 0u32;
    let mut slices = 0;
    loop {
        let (status, c, s) = slice(&mut peer, &wave, OP_TREE, &args(14, NO_FAIL, 2_000_000));
        assert_eq!(status, STATUS_DONE, "a continued slice retires DONE");
        count += c;
        sum = sum.wrapping_add(s);
        slices += 1;
        let stats = peer.wave_stats(&wave).expect("read the wave");
        match stats.slice_state {
            SliceState::Finished => break,
            SliceState::Continue => {}
            other => panic!("a host-continued slice ended {other:?}: {stats:?}"),
        }
        assert!(slices < 100, "the wave did not finish in 100 slices: {stats:?}");
    }
    assert!(slices >= 2, "the budget must have split the wave");
    assert_eq!((count, sum), expected(&r, 14));
    peer.release_wave(wave).expect("release the span");
}

/// A segment that fails on a team retires the slot as an error, and the
/// wave names the failing segment and its code.
#[test]
fn a_failing_segment_is_carried_to_the_slot_status_and_named() {
    let _device = common::device();
    let mut peer = peer(4);
    let fail_at = (2u32 << 24) | 3;
    let wave = peer
        .create_wave(&spec(Frontier::Global, Resume::Device, SliceBudget::Unbounded, roots(2), 4096))
        .expect("create the wave");
    let (status, _count, _sum) = slice(&mut peer, &wave, OP_TREE, &args(6, fail_at, 0));
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert_eq!(status, STATUS_ERR, "{stats:?}");
    assert_eq!(stats.slice_state, SliceState::Failed);
    let failure = stats.failure.expect("the wave records its failure");
    assert_eq!(failure.segment, Some(fail_at));
    assert_eq!(failure.code, 7);
    peer.release_wave(wave).expect("release the span");
}

#[test]
fn a_wave_that_outgrows_its_ids_fails_with_the_ids_code() {
    let _device = common::device();
    let mut peer = peer(1);
    let wave = peer
        .create_wave(&spec(Frontier::Global, Resume::Device, SliceBudget::Unbounded, roots(1), 16))
        .expect("create the wave");
    let (status, _count, _sum) = slice(&mut peer, &wave, OP_TREE, &args(8, NO_FAIL, 0));
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert_eq!(status, STATUS_ERR, "{stats:?}");
    assert_eq!(stats.failure.map(|f| f.code), Some(FAIL_IDS));
    peer.release_wave(wave).expect("release the span");
}

/// Each segment reserves 12 bytes, rounded to 16: an arena of exactly that
/// much is used in full, and one segment short of it fails.
#[test]
fn the_arena_reserves_per_segment_and_fails_when_exhausted() {
    let _device = common::device();
    let mut peer = peer(1);
    let r = roots(1);
    let (segments, _sum) = expected(&r, 5);

    let mut exact = spec(Frontier::Global, Resume::Device, SliceBudget::Unbounded, r.clone(), 4096);
    exact.arena_bytes = segments * 16;
    let wave = peer.create_wave(&exact).expect("create the wave");
    let (status, count, _sum) = slice(&mut peer, &wave, OP_TREE_ARENA, &args(5, NO_FAIL, 0));
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert_eq!(status, STATUS_DONE, "{stats:?}");
    assert_eq!(count, segments);
    assert_eq!(stats.arena_used_bytes, segments * 16);
    peer.release_wave(wave).expect("release the span");

    let mut short = exact.clone();
    short.arena_bytes = (segments - 1) * 16;
    let wave = peer.create_wave(&short).expect("create the wave");
    let (status, _count, _sum) = slice(&mut peer, &wave, OP_TREE_ARENA, &args(5, NO_FAIL, 0));
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert_eq!(status, STATUS_ERR, "{stats:?}");
    assert_eq!(stats.failure.map(|f| f.code), Some(FAIL_ARENA));
    peer.release_wave(wave).expect("release the span");
}

/// A span that does not hold a wave is refused by the op, not run.
#[test]
fn a_span_that_is_not_a_wave_is_refused() {
    let _device = common::device();
    let mut peer = peer(1);
    let handle = peer.pin_bulk(&[0u8; 1024]).expect("pin a plain span");
    let t = peer.submit_user(OP_TREE, Some(&handle), &args(4, NO_FAIL, 0)).expect("submit");
    assert_eq!(peer.wait_status(t, Duration::from_secs(10)).expect("retires"), STATUS_ERR);
    peer.reap(t).expect("reap");
    peer.unpin(handle).expect("release the span");
}

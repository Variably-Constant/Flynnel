//! Every position of a wave's frontier runs exactly once, on wide teams as
//! well as narrow ones: no block's generation range may run past another's.
//!
//! The op walks a binary tree per root with dense ids. A root is its index,
//! a child takes the next number from a counter in the wave's arena, and
//! each segment adds one to its own run count there. The host then reads
//! every id's count, so a segment run twice and a segment never run are
//! each named rather than cancelling out in a total.
//!
//! Requires a CUDA device and NVRTC.
#![cfg(feature = "gpu-peer")]

use std::num::NonZeroU32;
use std::time::Duration;

use flynnel::gpu_peer::wave::{Frontier, Resume, SliceBudget, SliceState, WaveSpec};
use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, STATUS_DONE, layout};

mod common;

/// Payload words: 0 depth limit, 1 root count. Arena word 0 counts
/// children, word 1 + id holds id's depth, and word 1 + CAPACITY + id counts
/// id's runs.
const OPS: &str = r#"
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    if (op != 600u) return 1u;
    volatile unsigned* args = (volatile unsigned*)payload;
    u32 limit = args[0];
    u32 roots = args[1];
    u32 capacity = args[2];

    flw_slice s;
    u32 bad = flw_slice_begin(&s, block, count, team_rank, team_size);
    if (bad != 0u) return bad;
    u32* arena = (u32*)(block + flw_get(block, FLW_ARENA_OFF_OFF));
    while (s.running) {
        for (u32 k = flw_first(&s); k < s.end; k += flw_stride(&s)) {
            u32 id = flw_id(&s, k);
            atomicAdd(arena + 1u + capacity + id, 1u);
            u32 depth = flw_ld(arena + 1u + id);
            if (depth < limit) {
                for (u32 c = 0u; c < 2u; c++) {
                    u32 child = roots + atomicAdd(arena, 1u);
                    flw_st(arena + 1u + child, depth + 1u);
                    flw_push(&s, child);
                }
            }
        }
        flw_generation_end(&s);
    }
    return flw_slice_end(&s);
}
"#;

const OP_COVER: u32 = layout::OP_USER_BASE + 500;
const ROOTS: u32 = 64;
const LIMIT: u32 = 9;
/// Every tree's ids.
const CAPACITY: u32 = ROOTS * ((1 << (LIMIT + 1)) - 1);

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

/// Run one tree wave on a team of `blocks` and check every id ran once.
fn every_position_runs_once(blocks: u32, frontier: Frontier) {
    let _device = common::device();
    let mut peer = peer(blocks);
    let team = peer.team_size();
    let wave = peer
        .create_wave(&WaveSpec {
            roots: (0..ROOTS).collect(),
            id_capacity: 2 * CAPACITY,
            arena_bytes: 4 * (1 + 2 * CAPACITY),
            frontier,
            resume: Resume::Device,
            slice_budget: SliceBudget::Unbounded,
            barrier_deadline: Duration::from_millis(50),
            done_deadline: Some(Duration::from_millis(500)),
            longest_generation_seed: Duration::ZERO,
            rob: None,
        })
        .expect("create the wave");

    let mut args = [0u8; 12];
    for (i, word) in [LIMIT, ROOTS, CAPACITY].into_iter().enumerate() {
        args[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    let t = peer.submit_wave(&wave, OP_COVER, &args).expect("submit the wave");
    let status = peer.wait_status(t, Duration::from_secs(60)).expect("the slice retires");
    peer.reap(t).expect("reap");
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert_eq!(status, STATUS_DONE, "{frontier:?} on a team of {team}: {stats:?}");
    assert_eq!(stats.slice_state, SliceState::Finished, "{stats:?}");

    let mut bytes = vec![0u8; 4 * (1 + 2 * CAPACITY as usize)];
    peer.fetch_bulk_at(wave.handle(), wave.layout().arena_off as usize, &mut bytes)
        .expect("read the arena");
    let words: Vec<u32> = bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
    assert_eq!(words[0], CAPACITY - ROOTS, "{frontier:?} on {team}: every child took an id");
    let runs = &words[1 + CAPACITY as usize..];
    let never: Vec<usize> = (0..CAPACITY as usize).filter(|&id| runs[id] == 0).collect();
    let repeated: Vec<usize> = (0..CAPACITY as usize).filter(|&id| runs[id] > 1).collect();
    assert!(
        never.is_empty() && repeated.is_empty(),
        "{frontier:?} on a team of {team}: {} of {CAPACITY} ids never ran (first {:?}) and {} ran more than once (first {:?}): {stats:?}",
        never.len(),
        &never[..never.len().min(8)],
        repeated.len(),
        &repeated[..repeated.len().min(8)],
    );
    peer.release_wave(wave).expect("release the span");
}

#[test]
fn every_position_of_a_global_frontier_runs_once_on_a_wide_team() {
    every_position_runs_once(24, Frontier::Global);
}

#[test]
fn every_position_of_a_global_frontier_runs_once_on_a_narrow_team() {
    every_position_runs_once(4, Frontier::Global);
}

#[test]
fn every_position_of_a_rebalancing_partition_runs_once_on_a_wide_team() {
    every_position_runs_once(24, Frontier::Partition { rebalance_every: NonZeroU32::new(2) });
}

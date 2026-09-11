//! A wave's ROB commits every row in pre-order over (parent, ordinal),
//! whatever order its segments finish in, stops a row at its first pending
//! or refused segment, and carries its rows across device yields and host
//! continuation.
//!
//! The op walks a binary tree per root. Ids are dense: a root is its row
//! index, and every child takes the next number from a counter in the
//! wave's arena, where each id's depth is also kept. Generations expand the
//! tree breadth-first, so a node's later siblings finish before its own
//! descendants do, which is the order pre-order commit has to see past.
//!
//! Requires a CUDA device and NVRTC.
#![cfg(feature = "gpu-peer")]

use std::num::NonZeroU32;
use std::time::Duration;

use flynnel::gpu_peer::wave::{
    Frontier, R_ROW, ROB_STRIDE, Resume, RobSpec, SegmentReport, SliceBudget, SliceState, Wave, WaveSpec,
};
use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, STATUS_DONE, layout};

mod common;

/// Payload words: 0 depth limit, 1 row that refuses (or 0xFFFFFFFF), 2 depth
/// at which that row refuses, 3 ns block 0 spins per generation, 4 nonzero
/// when segments at the depth limit and below are left to the host, 5 root
/// count, where child ids start. Arena word 0 counts children, and word
/// 1 + id holds id's depth.
const OPS: &str = r#"
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    if (op != 500u) return 1u;
    volatile unsigned* args = (volatile unsigned*)payload;
    u32 limit = args[0];
    u32 refuse_row = args[1];
    u32 refuse_depth = args[2];
    u32 spin_ns = args[3];
    u32 host_leaves = args[4];
    u32 roots = args[5];

    flw_slice s;
    u32 bad = flw_slice_begin(&s, block, count, team_rank, team_size);
    if (bad != 0u) return bad;
    u32* arena = (u32*)(block + flw_get(block, FLW_ARENA_OFF_OFF));
    while (s.running) {
        for (u32 k = flw_first(&s); k < s.end; k += flw_stride(&s)) {
            u32 id = flw_id(&s, k);
            u32 depth = flw_ld(arena + 1u + id);
            if (flw_ld(flw_rec(&s, id, FLW_R_ROW)) == refuse_row && depth == refuse_depth) {
                flw_rob_refuse(&s, id);
                continue;
            }
            if (host_leaves != 0u && depth >= limit) continue;
            if (depth < limit) {
                for (u32 c = 0u; c < 2u; c++) {
                    u32 child = roots + atomicAdd(arena, 1u);
                    flw_st(arena + 1u + child, depth + 1u);
                    flw_push_child(&s, id, child);
                }
            }
            flw_rob_expanded(&s, id);
            flw_rob_retired(&s, id);
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

const OP_ROB: u32 = layout::OP_USER_BASE + 400;
const NONE: u32 = u32::MAX;
const ROWS: u32 = 4;
const LIMIT: u32 = 7;
/// Segments in one row's full tree.
const PER_ROW: u32 = (1 << (LIMIT + 1)) - 1;
/// Every row's tree, and room for the host's extra children.
const CAPACITY: u32 = ROWS * PER_ROW + 8;

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

fn spec(frontier: Frontier, resume: Resume, budget: SliceBudget) -> WaveSpec {
    WaveSpec {
        roots: (0..ROWS).collect(),
        id_capacity: 4 * CAPACITY,
        arena_bytes: 4 * (CAPACITY + 1),
        frontier,
        resume,
        slice_budget: budget,
        barrier_deadline: Duration::from_millis(50),
        done_deadline: Some(Duration::from_millis(500)),
        longest_generation_seed: Duration::ZERO,
        rob: Some(RobSpec { capacity: CAPACITY }),
    }
}

fn args(limit: u32, refuse_row: u32, refuse_depth: u32, spin_ns: u32, host_leaves: u32) -> [u8; 24] {
    let mut a = [0u8; 24];
    for (i, word) in [limit, refuse_row, refuse_depth, spin_ns, host_leaves, ROWS].into_iter().enumerate() {
        a[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    a
}

/// Submit one slice, wait for it to retire, and return its status.
fn run_slice(peer: &mut GpuPeer, wave: &Wave, a: &[u8; 24]) -> u32 {
    let t = peer.submit_wave(wave, OP_ROB, a).expect("submit a slice");
    let status = peer.wait_status(t, Duration::from_secs(60)).expect("the slice retires");
    peer.reap(t).expect("reap");
    status
}

/// The arena's words: the child counter, then every id's depth.
fn arena(peer: &mut GpuPeer, wave: &Wave) -> Vec<u32> {
    let mut bytes = vec![0u8; 4 * (CAPACITY as usize + 1)];
    peer.fetch_bulk_at(wave.handle(), wave.layout().arena_off as usize, &mut bytes)
        .expect("read the arena");
    bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect()
}

fn depth_of(peer: &mut GpuPeer, wave: &Wave, id: u32) -> u32 {
    arena(peer, wave)[1 + id as usize]
}

fn row_of(peer: &mut GpuPeer, wave: &Wave, id: u32) -> u32 {
    let mut word = [0u8; 4];
    peer.fetch_bulk_at(
        wave.handle(),
        wave.layout().rob_off as usize + id as usize * ROB_STRIDE + R_ROW,
        &mut word,
    )
    .expect("read a record's row");
    u32::from_le_bytes(word)
}

fn every_row_commits_in_full(frontier: Frontier) {
    let _device = common::device();
    let mut peer = peer(4);
    let wave = peer
        .create_wave(&spec(frontier, Resume::Device, SliceBudget::Unbounded))
        .expect("create the wave");
    let status = run_slice(&mut peer, &wave, &args(LIMIT, NONE, NONE, 0, 0));
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert_eq!(status, STATUS_DONE, "{frontier:?}: {stats:?}");
    assert_eq!(stats.slice_state, SliceState::Finished);
    let rows = peer.wave_rows(&wave).expect("read the rows");
    for (r, row) in rows.iter().enumerate() {
        assert_eq!(row.root, r as u32);
        assert!(row.complete, "{frontier:?} row {r}: {row:?}");
        assert_eq!(row.committed_through, None);
        assert_eq!(row.committed, PER_ROW, "{frontier:?} row {r}: {row:?}");
        assert_eq!((row.first_refused, row.lowest_refused), (None, None));
        assert!(row.root_retired);
    }
    assert_eq!(stats.rob_committed, ROWS * PER_ROW);
    peer.release_wave(wave).expect("release the span");
}

#[test]
fn every_row_commits_in_full_on_a_global_frontier() {
    every_row_commits_in_full(Frontier::Global);
}

#[test]
fn every_row_commits_in_full_on_a_partition_that_rebalances() {
    every_row_commits_in_full(Frontier::Partition { rebalance_every: NonZeroU32::new(2) });
}

#[test]
fn every_row_commits_in_full_on_a_partition_that_never_rebalances() {
    every_row_commits_in_full(Frontier::Partition { rebalance_every: None });
}

/// With the leaves left unexpanded, each row commits the chain from its
/// root down to the last depth above the leaves, and holds at the leftmost
/// leaf, though every other internal node finished before that leaf's
/// ancestors' later subtrees did.
#[test]
fn a_segment_never_expanded_holds_its_row_at_the_leftmost_leaf() {
    let _device = common::device();
    let mut peer = peer(4);
    let wave = peer
        .create_wave(&spec(Frontier::Global, Resume::Host, SliceBudget::Unbounded))
        .expect("create the wave");
    assert_eq!(run_slice(&mut peer, &wave, &args(LIMIT, NONE, NONE, 0, 1)), STATUS_DONE);
    let rows = peer.wave_rows(&wave).expect("read the rows");
    for (r, row) in rows.iter().enumerate() {
        assert!(!row.complete, "{row:?}");
        assert_eq!(row.committed, LIMIT, "row {r}: {row:?}");
        let cursor = row.committed_through.expect("the row holds at a leaf");
        assert_eq!(depth_of(&mut peer, &wave, cursor), LIMIT);
        assert_eq!(row_of(&mut peer, &wave, cursor), r as u32);
    }
    peer.release_wave(wave).expect("release the span");
}

/// Every segment at one depth of one row refuses: that row stops at the
/// first of them in pre-order and names the lowest, and the others finish.
#[test]
fn a_refusal_stops_only_its_row_and_names_the_segment() {
    let _device = common::device();
    let mut peer = peer(4);
    let wave = peer
        .create_wave(&spec(Frontier::Global, Resume::Device, SliceBudget::Unbounded))
        .expect("create the wave");
    let refuse_row = 1;
    let refuse_depth = 3;
    assert_eq!(
        run_slice(&mut peer, &wave, &args(LIMIT, refuse_row, refuse_depth, 0, 0)),
        STATUS_DONE,
        "a refusal is the row's verdict, not a wave failure"
    );
    let rows = peer.wave_rows(&wave).expect("read the rows");
    for (r, row) in rows.iter().enumerate() {
        if r as u32 == refuse_row {
            assert!(!row.complete, "{row:?}");
            let first = row.first_refused.expect("the frontier stopped at a refusal");
            assert_eq!(row.committed_through, Some(first));
            assert_eq!(row.committed, refuse_depth, "{row:?}");
            assert_eq!(depth_of(&mut peer, &wave, first), refuse_depth);
            let lowest = row.lowest_refused.expect("the row keeps its lowest refusal");
            assert!(lowest <= first, "{row:?}");
            assert_eq!(depth_of(&mut peer, &wave, lowest), refuse_depth);
            assert_eq!(row_of(&mut peer, &wave, lowest), refuse_row);
        } else {
            assert!(row.complete, "row {r}: {row:?}");
            assert_eq!(row.committed, PER_ROW);
            assert_eq!(row.lowest_refused, None);
        }
    }
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert_eq!(stats.failure, None);
    peer.release_wave(wave).expect("release the span");
}

#[test]
fn rows_commit_in_full_across_device_yields() {
    let _device = common::device();
    let mut peer = peer(4);
    let wave = peer
        .create_wave(&spec(Frontier::Global, Resume::Device, SliceBudget::Fixed(Duration::from_millis(10))))
        .expect("create the wave");
    assert_eq!(run_slice(&mut peer, &wave, &args(LIMIT, NONE, NONE, 2_000_000, 0)), STATUS_DONE);
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert!(stats.yields >= 1, "8 generations of 2 ms each cannot fit a 10 ms slice: {stats:?}");
    for row in peer.wave_rows(&wave).expect("read the rows") {
        assert!(row.complete && row.committed == PER_ROW, "{row:?}");
    }
    peer.release_wave(wave).expect("release the span");
}

/// The device leaves every leaf to the host. Between slices the host gives
/// row 0's held leaf two children and reports every leaf expanded; a slice
/// then carries row 0 to the host's first child and finishes the other rows,
/// and once the host reports its children a slice with nothing pending
/// finishes row 0.
#[test]
fn host_run_segments_join_their_rows_between_slices() {
    let _device = common::device();
    let mut peer = peer(4);
    let wave = peer
        .create_wave(&spec(Frontier::Global, Resume::Host, SliceBudget::Unbounded))
        .expect("create the wave");
    let leaves_to_host = args(LIMIT, NONE, NONE, 0, 1);

    assert_eq!(run_slice(&mut peer, &wave, &leaves_to_host), STATUS_DONE);
    let held = peer.wave_rows(&wave).expect("read the rows")[0]
        .committed_through
        .expect("row 0 holds at its leftmost leaf");
    let words = arena(&mut peer, &wave);
    let pushed = words[0];
    assert_eq!(pushed, ROWS * (PER_ROW - 1), "every child the device pushed took an id");
    let leaves: Vec<u32> = (ROWS..ROWS + pushed).filter(|&id| words[1 + id as usize] == LIMIT).collect();
    assert_eq!(leaves.len() as u32, ROWS << LIMIT);
    assert!(leaves.contains(&held));

    // The held leaf's children lie past the depth limit, so the op leaves
    // them to the host as well.
    let extra = [ROWS + pushed, ROWS + pushed + 1];
    let arena_off = wave.layout().arena_off as usize;
    peer.write_resident_bulk_at(wave.handle(), arena_off, &(pushed + 2).to_le_bytes())
        .expect("advance the id counter");
    for id in extra {
        peer.write_resident_bulk_at(wave.handle(), arena_off + 4 + id as usize * 4, &(LIMIT + 1).to_le_bytes())
            .expect("record a depth");
    }
    peer.push_wave_segments(&wave, &[(held, extra[0]), (held, extra[1])])
        .expect("push the host's children");
    let reports: Vec<(u32, SegmentReport)> = leaves.iter().map(|&id| (id, SegmentReport::Expanded)).collect();
    peer.report_wave_segments(&wave, &reports).expect("report every leaf expanded");

    assert_eq!(run_slice(&mut peer, &wave, &leaves_to_host), STATUS_DONE);
    let rows = peer.wave_rows(&wave).expect("read the rows");
    assert_eq!(rows[0].committed_through, Some(extra[0]), "{:?}", rows[0]);
    assert_eq!(rows[0].committed, LIMIT + 1, "{:?}", rows[0]);
    for row in &rows[1..] {
        assert!(row.complete && row.committed == PER_ROW, "{row:?}");
    }

    peer.report_wave_segments(&wave, &[(extra[0], SegmentReport::Expanded), (extra[1], SegmentReport::Expanded)])
        .expect("report the host's children");
    assert_eq!(run_slice(&mut peer, &wave, &leaves_to_host), STATUS_DONE);
    let rows = peer.wave_rows(&wave).expect("read the rows");
    assert!(rows[0].complete, "{:?}", rows[0]);
    assert_eq!(rows[0].committed, PER_ROW + 2);
    let stats = peer.wave_stats(&wave).expect("read the wave");
    assert_eq!(stats.rob_committed, ROWS * PER_ROW + 2);
    peer.release_wave(wave).expect("release the span");
}

/// Host pushes and reports wait for the slice in flight, a host refusal
/// stops its row at the next walk, and a wave without a ROB takes neither.
#[test]
fn host_pushes_and_reports_wait_for_the_slice_and_need_a_rob() {
    let _device = common::device();
    let mut peer = peer(4);
    let wave = peer
        .create_wave(&spec(Frontier::Global, Resume::Host, SliceBudget::Unbounded))
        .expect("create the wave");
    let t = peer.submit_wave(&wave, OP_ROB, &args(LIMIT, NONE, NONE, 20_000_000, 1)).expect("submit");
    assert!(
        peer.report_wave_segments(&wave, &[(0, SegmentReport::Expanded)]).is_err(),
        "a report waits for the slice in flight"
    );
    assert!(
        peer.push_wave_segments(&wave, &[(0, CAPACITY - 1)]).is_err(),
        "a push waits for the slice in flight"
    );
    assert_eq!(peer.wait_status(t, Duration::from_secs(60)).expect("the slice retires"), STATUS_DONE);
    peer.reap(t).expect("reap");

    let held = peer.wave_rows(&wave).expect("read the rows")[0]
        .committed_through
        .expect("row 0 holds at its leftmost leaf");
    peer.report_wave_segments(&wave, &[(held, SegmentReport::Refused)]).expect("report a refusal");
    assert_eq!(run_slice(&mut peer, &wave, &args(LIMIT, NONE, NONE, 0, 1)), STATUS_DONE);
    let row = peer.wave_rows(&wave).expect("read the rows")[0];
    assert_eq!(row.first_refused, Some(held), "{row:?}");
    assert_eq!(row.lowest_refused, Some(held), "{row:?}");
    assert!(!row.complete);
    peer.release_wave(wave).expect("release the span");

    let mut plain = spec(Frontier::Global, Resume::Host, SliceBudget::Unbounded);
    plain.rob = None;
    let plain = peer.create_wave(&plain).expect("create a wave without a ROB");
    assert!(peer.wave_rows(&plain).is_err());
    assert!(peer.report_wave_segments(&plain, &[(0, SegmentReport::Expanded)]).is_err());
    assert!(peer.push_wave_segments(&plain, &[(0, 1)]).is_err());
    peer.release_wave(plain).expect("release the span");
}

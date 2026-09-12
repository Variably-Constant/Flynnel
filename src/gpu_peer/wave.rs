//! Segmented waves: the host side of the `flw_` helpers in
//! `kernels/gpu_peer_wave.cu`.
//!
//! A wave runs a set of segments in generations on every block of a lane's
//! team. Its state lives in one resident span: a header, a table with one
//! entry per block, an id array, a staging array that a partition
//! rebalance deals through, and an arena for segment state created on the
//! device. [`WaveSpec`] describes a wave, [`GpuPeer::create_wave`] lays out
//! and pins its span, [`GpuPeer::submit_wave`] runs a slice of it through
//! the consumer's user op, and [`GpuPeer::wave_stats`] reads its state back.
//! [`GpuPeer::calibrate_waves`] measures what waves cost on the device, and
//! [`plan`] turns those costs and a wave's stats into a frontier choice.
//!
//! A wave can keep a reorder buffer ([`RobSpec`]): a record per segment id
//! and a row per root. Segments link their children and report themselves
//! expanded, refused or retired, and block 0 commits each row in pre-order.
//! [`GpuPeer::wave_rows`] reads the rows, and between slices
//! [`GpuPeer::push_wave_segments`] and [`GpuPeer::report_wave_segments`] add
//! what the host ran.
//!
//! The consumer's op follows the loop shown at the top of the kernel file.
//! Every offset here mirrors a define there, and a test compares the two.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use super::{GpuPeer, GpuPeerError, ResidentHandle, STATUS_DONE, Ticket, layout};

pub mod plan;

/// The helper source, composed between the poller and the user source.
pub(crate) const WAVE_CU: &str = include_str!("../../kernels/gpu_peer_wave.cu");

/// First word of a wave span ("FLWV" little-endian).
pub const MAGIC: u32 = 0x5657_4C46;
/// Wave span format version.
pub const VERSION: u32 = 1;

/// Magic word.
pub const MAGIC_OFF: usize = 0x00;
/// Format version.
pub const VERSION_OFF: usize = 0x04;
/// Blocks in the wave, which is the lane's team size.
pub const WIDTH_OFF: usize = 0x08;
/// [`MODE_GLOBAL`] or [`MODE_PARTITION`].
pub const MODE_OFF: usize = 0x0C;
/// Partition rebalance interval in generations; 0 is never.
pub const REBALANCE_OFF: usize = 0x10;
/// [`RESUME_DEVICE`] or [`RESUME_HOST`].
pub const RESUME_MODE_OFF: usize = 0x14;
/// Barrier arrivals, never reset.
pub const ARRIVE_OFF: usize = 0x18;
/// Longest barrier wait, ns.
pub const WAIT_MAX_OFF: usize = 0x1C;
/// Summed barrier waits, in units of 1024 ns.
pub const WAIT_SUM_KNS_OFF: usize = 0x20;
/// Barriers a block left on its deadline.
pub const TIMEOUTS_OFF: usize = 0x24;
/// Longest first-barrier wait of a slice, ns.
pub const SKEW_MAX_OFF: usize = 0x28;
/// Slice-end arrivals, never reset.
pub const DONE_OFF: usize = 0x2C;
/// Complement of the lowest failing segment id; 0 when nothing failed.
pub const FAIL_COMP_OFF: usize = 0x30;
/// Largest failure code reported.
pub const FAIL_CODE_OFF: usize = 0x34;
/// Global frontier: ids pushed.
pub const PUSH_OFF: usize = 0x38;
/// Global frontier: current generation start.
pub const START_OFF: usize = 0x3C;
/// Global frontier: current generation end.
pub const END_OFF: usize = 0x40;
/// Ids the id array holds.
pub const ID_CAPACITY_OFF: usize = 0x44;
/// Arena bytes reserved.
pub const ARENA_BUMP_OFF: usize = 0x48;
/// Arena bytes available.
pub const ARENA_CAPACITY_OFF: usize = 0x4C;
/// Block 0's stop decision for the coming barrier.
pub const STOP_OFF: usize = 0x50;
/// The last slice's `SLICE_*` state.
pub const SLICE_STATE_OFF: usize = 0x54;
/// Slice budget from the watchdog, ns (u64); 0 is no budget.
pub const BUDGET_NS_OFF: usize = 0x58;
/// Longest generation measured, ns.
pub const LONGEST_GEN_OFF: usize = 0x60;
/// Slices ended by a device yield.
pub const YIELDS_OFF: usize = 0x64;
/// Slices run.
pub const SLICES_OFF: usize = 0x68;
/// Generations completed by block 0.
pub const GENERATIONS_OFF: usize = 0x6C;
/// Byte offset of the id array.
pub const IDS_OFF_OFF: usize = 0x70;
/// Byte offset of the rebalance staging array.
pub const STAGING_OFF_OFF: usize = 0x74;
/// Byte offset of the arena.
pub const ARENA_OFF_OFF: usize = 0x78;
/// Byte offset of the per-block table.
pub const TABLE_OFF_OFF: usize = 0x7C;
/// How long a block waits at a barrier, ns; never 0.
pub const BARRIER_DEADLINE_OFF: usize = 0x80;
/// The largest block's share over the mean, per mille: of the pending ids
/// at a partition rebalance, or of the children pushed in one generation on
/// a global frontier.
pub const IMBALANCE_OFF: usize = 0x84;
/// How long block 0 waits at slice end, ns (u64); 0 is no limit.
pub const DONE_DEADLINE_NS_OFF: usize = 0x88;
/// Rebalances run.
pub const REBALANCES_OFF: usize = 0x90;
/// Pending ids moved through staging by rebalances.
pub const MOVED_OFF: usize = 0x94;
/// Slice time summed on block 0, ns (u64).
pub const ELAPSED_NS_OFF: usize = 0x98;
/// Block 0's time in rebalances that moved ids, ns (u64).
pub const REBALANCE_NS_OFF: usize = 0xA0;
/// Byte offset of the ROB table; 0 when the wave keeps no ROB.
pub const ROB_OFF_OFF: usize = 0xA8;
/// Records in the ROB table; every segment id is below it.
pub const ROB_CAPACITY_OFF: usize = 0xAC;
/// Byte offset of the row table.
pub const ROWS_OFF_OFF: usize = 0xB0;
/// Rows, one per root.
pub const ROW_COUNT_OFF: usize = 0xB4;
/// Segments committed over every row.
pub const ROB_COMMITTED_OFF: usize = 0xB8;
/// Header size.
pub const HEADER_BYTES: usize = 0x100;

/// Per-block table entry size.
pub const TABLE_STRIDE: usize = 0x40;
/// Ids the block pushed: into its region for a partition, into the shared
/// array for a global frontier.
pub const T_PUSH: usize = 0x00;
/// Current generation start.
pub const T_START: usize = 0x04;
/// Current generation end.
pub const T_END: usize = 0x08;
/// 1 while the block's threads run another generation.
pub const T_DECISION: usize = 0x0C;
/// 1 when the block's current range is empty.
pub const T_EMPTY: usize = 0x10;
/// Rebalance: pending ids.
pub const T_PENDING: usize = 0x14;
/// Rebalance: staging offset of the block's run. Global frontier: the
/// block's pushes counted at the last barrier.
pub const T_PREFIX: usize = 0x18;
/// Generations the block completed.
pub const T_GENERATION: usize = 0x1C;
/// Rebalance: pending ids over every block.
pub const T_TOTAL: usize = 0x20;
/// Rebalance: staging start dealt to the block.
pub const T_DEAL_LO: usize = 0x24;
/// Rebalance: staging end dealt to the block.
pub const T_DEAL_HI: usize = 0x28;

/// ROB record size, one per segment id.
pub const ROB_STRIDE: usize = 0x28;
/// Parent id; [`ROB_NONE`] for a root or an id never linked.
pub const R_PARENT: usize = 0x00;
/// Index among the parent's children.
pub const R_ORDINAL: usize = 0x04;
/// The row the segment belongs to.
pub const R_ROW: usize = 0x08;
/// Child with ordinal 0; [`ROB_NONE`] for none.
pub const R_FIRST_CHILD: usize = 0x0C;
/// The parent's next child; [`ROB_NONE`] after the last.
pub const R_NEXT_SIBLING: usize = 0x10;
/// Most recently linked child.
pub const R_LAST_CHILD: usize = 0x14;
/// 1 once the segment's own step is done and its children are linked.
pub const R_EXPANDED: usize = 0x18;
/// 1 once the segment refused.
pub const R_REFUSED: usize = 0x1C;
/// 1 once the segment's value retired.
pub const R_RETIRED: usize = 0x20;
/// Children linked.
pub const R_CHILDREN: usize = 0x24;

/// Row size, one per root.
pub const ROW_STRIDE: usize = 0x18;
/// Root id.
pub const W_ROOT: usize = 0x00;
/// First segment in pre-order not committed; [`ROB_NONE`] once complete.
pub const W_CURSOR: usize = 0x04;
/// [`ROW_COMPLETE`] and [`ROW_REFUSED`] bits, written by block 0's walk.
pub const W_FLAGS: usize = 0x08;
/// Complement of the lowest id that refused; 0 when none did.
pub const W_REFUSED_COMP: usize = 0x0C;
/// 1 once the root's value retired.
pub const W_ROOT_RETIRED: usize = 0x10;
/// Segments of the row committed.
pub const W_COMMITTED: usize = 0x14;
/// Row flag: every segment of the row committed.
pub const ROW_COMPLETE: u32 = 1;
/// Row flag: the row's frontier stopped at a refused segment.
pub const ROW_REFUSED: u32 = 2;
/// An absent segment in a ROB link.
pub const ROB_NONE: u32 = 0xFFFF_FFFF;

/// Frontier mode: one frontier across the team.
pub const MODE_GLOBAL: u32 = 0;
/// Frontier mode: one frontier per block.
pub const MODE_PARTITION: u32 = 1;
/// Resume mode: a slice that stops early yields and runs again on the device.
pub const RESUME_DEVICE: u32 = 0;
/// Resume mode: a slice that stops early retires, and the host continues it.
pub const RESUME_HOST: u32 = 1;

/// Slice state: running.
pub const SLICE_RUNNING: u32 = 0;
/// Slice state: stopped early and yielded to run again on the device.
pub const SLICE_YIELDED: u32 = 1;
/// Slice state: stopped early for the host to continue.
pub const SLICE_CONTINUE: u32 = 2;
/// Slice state: every frontier is empty.
pub const SLICE_FINISHED: u32 = 3;
/// Slice state: the wave failed.
pub const SLICE_FAILED: u32 = 4;

/// Segment id recorded for a failure that belongs to no segment.
pub const NO_SEGMENT: u32 = 0xFFFF_FFFE;
/// Failure code: the id array is full.
pub const FAIL_IDS: u32 = 0xF001;
/// Failure code: the arena is exhausted.
pub const FAIL_ARENA: u32 = 0xF002;
/// Failure code: a barrier expired.
pub const FAIL_BARRIER: u32 = 0xF003;
/// Failure code: block 0's slice-end wait expired.
pub const FAIL_DONE: u32 = 0xF004;
/// Failure code: the longest generation leaves no room in the budget.
pub const FAIL_BUDGET: u32 = 0xF005;
/// Failure code: a segment id lies outside the ROB, or the ROB's links hold
/// a cycle.
pub const FAIL_ROB: u32 = 0xF006;

/// Op return for a failed wave; the slot retires as an error.
pub const STATUS_FAILED: u32 = 2;
/// Op return when the span is not a wave for this team.
pub const STATUS_BAD_SPAN: u32 = 3;

/// How a wave's frontier is kept across the blocks of the team.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frontier {
    /// One frontier across the team, fixed by a barrier at every
    /// generation. Load stays even, and each generation waits for its
    /// slowest block.
    Global,
    /// One frontier per block, with children kept in their parent's block.
    /// Every `rebalance_every` generations the blocks meet and their pending
    /// segments are dealt out evenly; with `None` they never meet.
    Partition {
        /// Generations between rebalances.
        rebalance_every: Option<NonZeroU32>,
    },
}

/// What a slice that stops before the wave finishes does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resume {
    /// Yield its slot, so the poller runs the next slice with no host
    /// round trip.
    Device,
    /// Retire its slot, so the host submits the next slice.
    Host,
}

/// Where a wave's slice budget comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceBudget {
    /// From the watchdog detected on the peer's device, less the poller
    /// quantum and the kernel's barrier deadline.
    Detected,
    /// No budget: a slice runs until the wave finishes or fails.
    Unbounded,
    /// A budget the caller chose.
    Fixed(Duration),
}

/// A wave's reorder buffer: a record per segment id and a row per root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RobSpec {
    /// Records the ROB holds. Every segment id, roots included, is below
    /// it, and it is below [`ROB_NONE`].
    pub capacity: u32,
}

/// A wave to create.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveSpec {
    /// Segment ids of the first generation.
    pub roots: Vec<u32>,
    /// Segment ids the wave can hold over its whole run, roots included. A
    /// partition rounds this up to a multiple of the team size.
    pub id_capacity: u32,
    /// Bytes of arena for segment state created on the device.
    pub arena_bytes: u32,
    /// How the frontier is kept across blocks.
    pub frontier: Frontier,
    /// What a slice that stops early does.
    pub resume: Resume,
    /// Where the slice budget comes from.
    pub slice_budget: SliceBudget,
    /// How long a block waits at a barrier before the wave fails.
    pub barrier_deadline: Duration,
    /// How long block 0 waits for every block at the end of a slice, or
    /// `None` for no limit.
    pub done_deadline: Option<Duration>,
    /// Longest generation to assume before one has been measured.
    pub longest_generation_seed: Duration,
    /// The wave's reorder buffer, or `None` for a wave that keeps none.
    /// Each root starts one row.
    pub rob: Option<RobSpec>,
}

/// Where each part of a wave span sits, in bytes from the span start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveLayout {
    /// Blocks in the wave.
    pub width: u32,
    /// Ids the id array holds, after rounding for a partition.
    pub id_capacity: u32,
    /// Start of the per-block table.
    pub table_off: u32,
    /// Start of the row table, right after the per-block table.
    pub rows_off: u32,
    /// Rows: one per root when the wave keeps a ROB, and 0 otherwise.
    pub row_count: u32,
    /// Start of the id array.
    pub ids_off: u32,
    /// Start of the staging array.
    pub staging_off: u32,
    /// Ids the staging array holds: the id capacity for a partition that
    /// rebalances, and 0 otherwise.
    pub staging_ids: u32,
    /// Start of the ROB table.
    pub rob_off: u32,
    /// Records the ROB table holds; 0 when the wave keeps none.
    pub rob_capacity: u32,
    /// Start of the arena.
    pub arena_off: u32,
    /// Arena bytes.
    pub arena_bytes: u32,
    /// Bytes the whole span occupies.
    pub total_bytes: u32,
}

/// `v` as a u32 byte offset, or the error saying the span cannot hold it.
fn span_offset(v: u64) -> Result<u32, GpuPeerError> {
    if v > u64::from(u32::MAX) {
        return Err(GpuPeerError::Unavailable(
            "the wave span exceeds the 4 GiB a user op's byte count can address",
        ));
    }
    Ok(v as u32)
}

impl WaveLayout {
    /// The layout of a wave with `width` blocks.
    pub fn new(width: u32, spec: &WaveSpec) -> Result<Self, GpuPeerError> {
        if width == 0 {
            return Err(GpuPeerError::Unavailable("a wave needs at least one block"));
        }
        if spec.id_capacity == 0 {
            return Err(GpuPeerError::Unavailable("a wave needs room for at least one segment id"));
        }
        let id_capacity = match spec.frontier {
            Frontier::Global => u64::from(spec.id_capacity),
            Frontier::Partition { .. } => {
                u64::from(spec.id_capacity).div_ceil(u64::from(width)) * u64::from(width)
            }
        };
        let staging_ids = match spec.frontier {
            Frontier::Partition { rebalance_every: Some(_) } => id_capacity,
            Frontier::Global | Frontier::Partition { rebalance_every: None } => 0,
        };
        let (row_count, rob_capacity) = match spec.rob {
            Some(rob) => {
                if rob.capacity == 0 || rob.capacity == ROB_NONE {
                    return Err(GpuPeerError::Unavailable(
                        "a ROB's capacity must be at least 1 and below ROB_NONE",
                    ));
                }
                (spec.roots.len() as u64, u64::from(rob.capacity))
            }
            None => (0, 0),
        };
        let table_off = HEADER_BYTES as u64;
        let rows_off = table_off + u64::from(width) * TABLE_STRIDE as u64;
        let ids_off = rows_off + row_count * ROW_STRIDE as u64;
        let staging_off = ids_off + id_capacity * 4;
        let rob_off = (staging_off + staging_ids * 4).div_ceil(8) * 8;
        let arena_off = (rob_off + rob_capacity * ROB_STRIDE as u64).div_ceil(8) * 8;
        let total = arena_off + u64::from(spec.arena_bytes);
        Ok(Self {
            width,
            id_capacity: span_offset(id_capacity)?,
            table_off: span_offset(table_off)?,
            rows_off: span_offset(rows_off)?,
            row_count: span_offset(row_count)?,
            ids_off: span_offset(ids_off)?,
            staging_off: span_offset(staging_off)?,
            staging_ids: span_offset(staging_ids)?,
            rob_off: span_offset(rob_off)?,
            rob_capacity: span_offset(rob_capacity)?,
            arena_off: span_offset(arena_off)?,
            arena_bytes: spec.arena_bytes,
            total_bytes: span_offset(total)?,
        })
    }
}

/// The slice budget a device's watchdog leaves a wave, in nanoseconds, or
/// 0 when no watchdog applies.
///
/// A poller launch can run for up to its quantum before it takes a slot,
/// and the kernel's barrier after the op can add its deadline, so both are
/// taken off the watchdog's delay. The wave's stop rule then compares the
/// time a slice has run plus its longest measured generation against what
/// remains.
pub fn budget_from_watchdog(
    delay_ns: Option<u64>,
    quantum_ns: u64,
    kernel_barrier_ns: u64,
) -> Result<u64, GpuPeerError> {
    match delay_ns {
        None => Ok(0),
        Some(delay) => {
            let reserved = quantum_ns.saturating_add(kernel_barrier_ns);
            if delay <= reserved {
                Err(GpuPeerError::Unavailable(
                    "the watchdog delay leaves no time for a wave slice after the \
                     poller quantum and the kernel barrier deadline",
                ))
            } else {
                Ok(delay - reserved)
            }
        }
    }
}

fn put_u32(span: &mut [u8], at: usize, v: u32) {
    span[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(span: &mut [u8], at: usize, v: u64) {
    span[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn get_u32(span: &[u8], at: usize) -> u32 {
    let mut w = [0u8; 4];
    w.copy_from_slice(&span[at..at + 4]);
    u32::from_le_bytes(w)
}

fn get_u64(span: &[u8], at: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&span[at..at + 8]);
    u64::from_le_bytes(w)
}

/// `d` in nanoseconds, or `too_long` when it does not fit in `limit`.
fn nanos_within(d: Duration, limit: u64, too_long: &'static str) -> Result<u64, GpuPeerError> {
    let ns = d.as_nanos();
    if ns > u128::from(limit) {
        return Err(GpuPeerError::Unavailable(too_long));
    }
    Ok(ns as u64)
}

/// The bytes of a new wave span: header, table, first generation and, with
/// a ROB, its rows and records.
pub fn initial_span(spec: &WaveSpec, layout: &WaveLayout, budget_ns: u64) -> Result<Vec<u8>, GpuPeerError> {
    let width = layout.width;
    let barrier_ns = nanos_within(
        spec.barrier_deadline,
        u64::from(u32::MAX),
        "the barrier deadline exceeds 4.29 s",
    )?;
    if barrier_ns == 0 {
        return Err(GpuPeerError::Unavailable("the barrier deadline must be longer than zero"));
    }
    let done_ns = match spec.done_deadline {
        Some(d) => nanos_within(d, u64::MAX, "the slice-end deadline does not fit in nanoseconds")?.max(1),
        None => 0,
    };
    let seed_ns = nanos_within(
        spec.longest_generation_seed,
        u64::from(u32::MAX),
        "the generation seed exceeds 4.29 s",
    )?;
    let rebalance = match spec.frontier {
        Frontier::Global => 0,
        Frontier::Partition { rebalance_every: Some(n) } => n.get(),
        Frontier::Partition { rebalance_every: None } => 0,
    };
    let mode = match spec.frontier {
        Frontier::Global => MODE_GLOBAL,
        Frontier::Partition { .. } => MODE_PARTITION,
    };
    let resume = match spec.resume {
        Resume::Device => RESUME_DEVICE,
        Resume::Host => RESUME_HOST,
    };

    let mut span = vec![0u8; layout.total_bytes as usize];
    put_u32(&mut span, MAGIC_OFF, MAGIC);
    put_u32(&mut span, VERSION_OFF, VERSION);
    put_u32(&mut span, WIDTH_OFF, width);
    put_u32(&mut span, MODE_OFF, mode);
    put_u32(&mut span, REBALANCE_OFF, rebalance);
    put_u32(&mut span, RESUME_MODE_OFF, resume);
    put_u32(&mut span, ID_CAPACITY_OFF, layout.id_capacity);
    put_u32(&mut span, ARENA_CAPACITY_OFF, layout.arena_bytes);
    put_u64(&mut span, BUDGET_NS_OFF, budget_ns);
    put_u32(&mut span, LONGEST_GEN_OFF, seed_ns as u32);
    put_u32(&mut span, IDS_OFF_OFF, layout.ids_off);
    put_u32(&mut span, STAGING_OFF_OFF, layout.staging_off);
    put_u32(&mut span, ARENA_OFF_OFF, layout.arena_off);
    put_u32(&mut span, TABLE_OFF_OFF, layout.table_off);
    put_u32(&mut span, BARRIER_DEADLINE_OFF, barrier_ns as u32);
    put_u64(&mut span, DONE_DEADLINE_NS_OFF, done_ns);

    if let Some(rob) = spec.rob {
        let mut sorted = spec.roots.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(GpuPeerError::Unavailable("a ROB's roots must be distinct, one per row"));
        }
        if sorted.last().is_some_and(|&top| top >= rob.capacity) {
            return Err(GpuPeerError::Unavailable("every root id must be below the ROB's capacity"));
        }
        put_u32(&mut span, ROB_OFF_OFF, layout.rob_off);
        put_u32(&mut span, ROB_CAPACITY_OFF, rob.capacity);
        put_u32(&mut span, ROWS_OFF_OFF, layout.rows_off);
        put_u32(&mut span, ROW_COUNT_OFF, layout.row_count);
        let records = layout.rob_off as usize;
        for id in 0..rob.capacity as usize {
            let r = records + id * ROB_STRIDE;
            put_u32(&mut span, r + R_PARENT, ROB_NONE);
            put_u32(&mut span, r + R_FIRST_CHILD, ROB_NONE);
            put_u32(&mut span, r + R_NEXT_SIBLING, ROB_NONE);
            put_u32(&mut span, r + R_LAST_CHILD, ROB_NONE);
        }
        for (row, &root) in spec.roots.iter().enumerate() {
            put_u32(&mut span, records + root as usize * ROB_STRIDE + R_ROW, row as u32);
            let w = layout.rows_off as usize + row * ROW_STRIDE;
            put_u32(&mut span, w + W_ROOT, root);
            put_u32(&mut span, w + W_CURSOR, root);
        }
    }

    let roots = spec.roots.len();
    let ids = layout.ids_off as usize;
    match spec.frontier {
        Frontier::Global => {
            if roots > layout.id_capacity as usize {
                return Err(GpuPeerError::Unavailable("the roots exceed the wave's id capacity"));
            }
            for (i, id) in spec.roots.iter().enumerate() {
                put_u32(&mut span, ids + i * 4, *id);
            }
            put_u32(&mut span, PUSH_OFF, roots as u32);
            put_u32(&mut span, END_OFF, roots as u32);
        }
        Frontier::Partition { .. } => {
            let per_block = (layout.id_capacity / width) as usize;
            let w = width as usize;
            for b in 0..w {
                let lo = b * roots / w;
                let hi = (b + 1) * roots / w;
                if hi - lo > per_block {
                    return Err(GpuPeerError::Unavailable(
                        "the roots dealt to one block exceed that block's share of the id capacity",
                    ));
                }
                let region = ids + b * per_block * 4;
                for (i, id) in spec.roots[lo..hi].iter().enumerate() {
                    put_u32(&mut span, region + i * 4, *id);
                }
                let entry = layout.table_off as usize + b * TABLE_STRIDE;
                put_u32(&mut span, entry + T_PUSH, (hi - lo) as u32);
                put_u32(&mut span, entry + T_END, (hi - lo) as u32);
            }
        }
    }
    Ok(span)
}

/// How the last slice of a wave ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceState {
    /// No slice has ended yet.
    Running,
    /// The slice stopped early and yielded to run again on the device.
    Yielded,
    /// The slice stopped early for the host to continue.
    Continue,
    /// Every frontier is empty.
    Finished,
    /// The wave failed.
    Failed,
}

/// A wave failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveFailure {
    /// The lowest failing segment id, or `None` for a failure that belongs
    /// to no segment.
    pub segment: Option<u32>,
    /// The largest failure code reported. Codes below 0xF000 are the op's
    /// own; a `FAIL_*` code reported beside them wins.
    pub code: u32,
}

/// One block's frontier as the table holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockState {
    /// Current generation start.
    pub start: u32,
    /// Current generation end.
    pub end: u32,
    /// Ids the block pushed: into its region for a partition, into the
    /// shared array for a global frontier.
    pub pushed: u32,
    /// Generations the block completed.
    pub generations: u32,
}

/// A wave's state and measurements, read from its span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveStats {
    /// How the last slice ended.
    pub slice_state: SliceState,
    /// Slices run.
    pub slices: u32,
    /// Slices ended by a device yield.
    pub yields: u32,
    /// Generations completed by block 0.
    pub generations: u32,
    /// Longest barrier wait, ns.
    pub barrier_wait_max_ns: u32,
    /// Summed barrier waits, ns.
    pub barrier_wait_sum_ns: u64,
    /// Barriers a block left on its deadline.
    pub barrier_timeouts: u32,
    /// Longest first-barrier wait of a slice, ns.
    pub start_skew_max_ns: u32,
    /// Longest generation, ns.
    pub longest_generation_ns: u32,
    /// The largest block's share over the mean, per mille: of the pending
    /// ids at a partition rebalance, or of the children pushed in one
    /// generation on a global frontier. 1000 is an even share, and the
    /// largest reading of the wave is the one kept. 0 when nothing was
    /// measured.
    ///
    /// The mean is taken over the blocks that could have held work, not
    /// over the team. Every block of a partition has a region of its own,
    /// so there the two are the same. A global frontier is dealt in runs
    /// of one block's threads from rank 0, so a generation of n ids
    /// reaches only its first `ceil(n / threads per block)` blocks, and a
    /// block holding everything when it was the only block dealt anything
    /// is an even share rather than the worst imbalance there is. The
    /// kernel divides by the block dimension it was launched with, which
    /// the poller sets; it is not a figure this type holds.
    pub imbalance_per_mille: u32,
    /// Rebalances run.
    pub rebalances: u32,
    /// Pending ids moved through staging by rebalances.
    pub moved_ids: u32,
    /// Slice time summed on block 0, ns.
    pub elapsed_ns: u64,
    /// Block 0's time in rebalances that moved ids, ns.
    pub rebalance_ns: u64,
    /// Arena bytes reserved.
    pub arena_used_bytes: u32,
    /// Global frontier: ids pushed.
    pub pushed: u32,
    /// Segments committed over every row of the wave's ROB.
    pub rob_committed: u32,
    /// The failure, when the wave failed.
    pub failure: Option<WaveFailure>,
    /// Every block's frontier.
    pub blocks: Vec<BlockState>,
}

/// The slice state a header word names.
fn slice_state(word: u32) -> Result<SliceState, GpuPeerError> {
    if word == SLICE_RUNNING {
        Ok(SliceState::Running)
    } else if word == SLICE_YIELDED {
        Ok(SliceState::Yielded)
    } else if word == SLICE_CONTINUE {
        Ok(SliceState::Continue)
    } else if word == SLICE_FINISHED {
        Ok(SliceState::Finished)
    } else if word == SLICE_FAILED {
        Ok(SliceState::Failed)
    } else {
        Err(GpuPeerError::Unavailable("the wave holds a slice state this build does not know"))
    }
}

impl WaveStats {
    /// Decode a wave's header and table from the first bytes of its span.
    pub fn decode(bytes: &[u8], width: u32) -> Result<Self, GpuPeerError> {
        let table_end = HEADER_BYTES + width as usize * TABLE_STRIDE;
        if bytes.len() < table_end {
            return Err(GpuPeerError::Unavailable("too few bytes for the wave header and table"));
        }
        if get_u32(bytes, MAGIC_OFF) != MAGIC || get_u32(bytes, VERSION_OFF) != VERSION {
            return Err(GpuPeerError::Unavailable("the span does not hold a wave of this version"));
        }
        let comp = get_u32(bytes, FAIL_COMP_OFF);
        let failure = if comp == 0 {
            None
        } else {
            let segment = u32::MAX - comp;
            Some(WaveFailure {
                segment: if segment == NO_SEGMENT { None } else { Some(segment) },
                code: get_u32(bytes, FAIL_CODE_OFF),
            })
        };
        let table = get_u32(bytes, TABLE_OFF_OFF) as usize;
        if table + width as usize * TABLE_STRIDE > bytes.len() {
            return Err(GpuPeerError::Unavailable("the wave's table lies past the bytes read"));
        }
        let blocks = (0..width as usize)
            .map(|b| {
                let e = table + b * TABLE_STRIDE;
                BlockState {
                    start: get_u32(bytes, e + T_START),
                    end: get_u32(bytes, e + T_END),
                    pushed: get_u32(bytes, e + T_PUSH),
                    generations: get_u32(bytes, e + T_GENERATION),
                }
            })
            .collect();
        Ok(Self {
            slice_state: slice_state(get_u32(bytes, SLICE_STATE_OFF))?,
            slices: get_u32(bytes, SLICES_OFF),
            yields: get_u32(bytes, YIELDS_OFF),
            generations: get_u32(bytes, GENERATIONS_OFF),
            barrier_wait_max_ns: get_u32(bytes, WAIT_MAX_OFF),
            barrier_wait_sum_ns: u64::from(get_u32(bytes, WAIT_SUM_KNS_OFF)) << 10,
            barrier_timeouts: get_u32(bytes, TIMEOUTS_OFF),
            start_skew_max_ns: get_u32(bytes, SKEW_MAX_OFF),
            longest_generation_ns: get_u32(bytes, LONGEST_GEN_OFF),
            imbalance_per_mille: get_u32(bytes, IMBALANCE_OFF),
            rebalances: get_u32(bytes, REBALANCES_OFF),
            moved_ids: get_u32(bytes, MOVED_OFF),
            elapsed_ns: get_u64(bytes, ELAPSED_NS_OFF),
            rebalance_ns: get_u64(bytes, REBALANCE_NS_OFF),
            arena_used_bytes: get_u32(bytes, ARENA_BUMP_OFF),
            pushed: get_u32(bytes, PUSH_OFF),
            rob_committed: get_u32(bytes, ROB_COMMITTED_OFF),
            failure,
            blocks,
        })
    }
}

/// One row of a wave's ROB: a root and every segment descended from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowState {
    /// The row's root id.
    pub root: u32,
    /// The first segment in the row's pre-order that has not committed;
    /// every segment before it has. `None` once the row is complete.
    pub committed_through: Option<u32>,
    /// Every segment of the row has committed.
    pub complete: bool,
    /// The refused segment the row's commit frontier stopped at.
    pub first_refused: Option<u32>,
    /// The lowest id that refused anywhere in the row.
    pub lowest_refused: Option<u32>,
    /// The root's value has retired.
    pub root_retired: bool,
    /// Segments of the row committed.
    pub committed: u32,
}

impl RowState {
    /// Decode the row whose entry starts `at` bytes into `bytes`.
    fn decode(bytes: &[u8], at: usize) -> Self {
        let cursor = get_u32(bytes, at + W_CURSOR);
        let flags = get_u32(bytes, at + W_FLAGS);
        let comp = get_u32(bytes, at + W_REFUSED_COMP);
        let refused = flags & ROW_REFUSED != 0;
        Self {
            root: get_u32(bytes, at + W_ROOT),
            committed_through: if cursor == ROB_NONE { None } else { Some(cursor) },
            complete: flags & ROW_COMPLETE != 0,
            first_refused: if refused { Some(cursor) } else { None },
            lowest_refused: if comp == 0 { None } else { Some(u32::MAX - comp) },
            root_retired: get_u32(bytes, at + W_ROOT_RETIRED) != 0,
            committed: get_u32(bytes, at + W_COMMITTED),
        }
    }
}

/// What a host-run segment reports into its wave's ROB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentReport {
    /// Its own step is done, and every child it pushed is linked.
    Expanded,
    /// It refused, so its row commits nothing past it.
    Refused,
    /// Its value has retired.
    Retired,
}

/// A wave pinned on a peer.
#[derive(Debug, Clone)]
pub struct Wave {
    handle: ResidentHandle,
    layout: WaveLayout,
    budget_ns: u64,
    watchdog_basis: Option<String>,
    /// The last slice submitted against the wave.
    in_flight: Cell<Option<Ticket>>,
}

impl Wave {
    /// The resident span holding the wave.
    pub fn handle(&self) -> &ResidentHandle {
        &self.handle
    }

    /// Where each part of the span sits.
    pub fn layout(&self) -> WaveLayout {
        self.layout
    }

    /// The slice budget written to the span, ns; 0 is no budget.
    pub fn budget_ns(&self) -> u64 {
        self.budget_ns
    }

    /// What was read to derive the budget, when it came from the watchdog.
    pub fn watchdog_basis(&self) -> Option<&str> {
        self.watchdog_basis.as_deref()
    }
}

/// Segmented-wave costs measured on one device at one team size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveCosts {
    /// Team size the costs were measured at.
    pub width: u32,
    /// Cost of one cross-block generation barrier, ns.
    pub barrier_ns: u64,
    /// Host round trip of a wave slice apart from its segments, ns.
    pub fixed_ns: u64,
    /// Substrate cost of one segment, ps.
    pub segment_ps: u64,
    /// Fixed cost of one rebalance apart from the ids it moves, ns: its
    /// two barriers and its synchronization.
    pub rebalance_fixed_ns: u64,
    /// Rebalance cost of moving one pending id, ps.
    pub copy_ps_per_id: u64,
    /// First-barrier wait of a coupled slice as its blocks pick up the
    /// slot, ns; paid once per slice.
    pub start_skew_ns: u64,
    /// Longest generation of the calibration waves, ns.
    pub generation_ns: u64,
}

impl WaveCosts {
    /// Planner inputs for a program whose last wave ran with `frontier` and
    /// produced `stats`.
    ///
    /// The imbalance is the one the wave recorded, over one generation for a
    /// global frontier and over the rebalance interval for a partition; a
    /// wave that recorded none gives none. Pending ids are the ids moved per
    /// rebalance when the wave rebalanced, and otherwise the ids pushed per
    /// generation.
    pub fn plan_inputs(&self, stats: &WaveStats, frontier: Frontier) -> plan::PlanInputs {
        let block_generations = stats.blocks.iter().map(|b| b.generations).fold(0, u32::max);
        let generations = f64::from(stats.generations.max(block_generations).max(1));
        let over_generations = match frontier {
            Frontier::Global => 1,
            Frontier::Partition { rebalance_every: Some(n) } => n.get(),
            Frontier::Partition { rebalance_every: None } => generations as u32,
        };
        let pending_ids = if stats.rebalances > 0 {
            f64::from(stats.moved_ids) / f64::from(stats.rebalances)
        } else {
            let pushed: u64 = match frontier {
                Frontier::Global => u64::from(stats.pushed),
                Frontier::Partition { .. } => stats.blocks.iter().map(|b| u64::from(b.pushed)).sum(),
            };
            pushed as f64 / generations
        };
        let imbalance = if stats.imbalance_per_mille == 0 {
            None
        } else {
            Some(plan::Imbalance { per_mille: stats.imbalance_per_mille, over_generations })
        };
        plan::PlanInputs {
            width: self.width,
            barrier_ns: self.barrier_ns as f64,
            rebalance_fixed_ns: self.rebalance_fixed_ns as f64,
            copy_ns_per_id: self.copy_ps_per_id as f64 / 1000.0,
            pending_ids,
            generation_ns: f64::from(stats.longest_generation_ns),
            imbalance,
        }
    }
}

/// Depths of the calibration tree. Each is run as three frontiers.
const CALIBRATION_DEPTHS: [u32; 4] = [2, 4, 6, 8];
/// Roots of the calibration tree.
const CALIBRATION_ROOTS: u32 = 64;
/// Runs of each depth and frontier; the medians are kept.
const CALIBRATION_REPEATS: usize = 5;

/// The medians of one calibration configuration's runs.
struct CalibrationRun {
    wall_ns: u64,
    elapsed_ns: u64,
    moved_ids: u64,
    rebalances: u64,
    rebalance_ns: u64,
    skew_ns: u64,
    longest_ns: u64,
}

fn median(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

/// The intercept and slope of the least-squares line through `points`.
fn least_squares(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    match points {
        [] => return (0.0, 0.0),
        [only] => return (only.1, 0.0),
        _ => {}
    }
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxy: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
    let sxx: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.0 - mean_x)).sum();
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (mean_y - slope * mean_x, slope)
}

/// The non-negative least-squares fit of `y = fixed * r + per_id * m`
/// through `(r, m, y)` points, returned as `(fixed, per_id)`. When the joint
/// fit would make a term negative, that term is zero and the other is fit
/// alone; of the two single-term fits, the one with the smaller residual is
/// returned.
fn rebalance_fit(points: &[(f64, f64, f64)]) -> (f64, f64) {
    let srr: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let srm: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let smm: f64 = points.iter().map(|p| p.1 * p.1).sum();
    let sry: f64 = points.iter().map(|p| p.0 * p.2).sum();
    let smy: f64 = points.iter().map(|p| p.1 * p.2).sum();
    let det = srr * smm - srm * srm;
    if det > f64::EPSILON * srr * smm {
        let fixed = (sry * smm - smy * srm) / det;
        let per_id = (smy * srr - sry * srm) / det;
        if fixed >= 0.0 && per_id >= 0.0 {
            return (fixed, per_id);
        }
    }
    let fixed_only = if srr > 0.0 { (sry / srr).max(0.0) } else { 0.0 };
    let per_id_only = if smm > 0.0 { (smy / smm).max(0.0) } else { 0.0 };
    let residual = |fixed: f64, per_id: f64| -> f64 {
        points
            .iter()
            .map(|p| {
                let e = p.2 - fixed * p.0 - per_id * p.1;
                e * e
            })
            .sum()
    };
    if residual(fixed_only, 0.0) <= residual(0.0, per_id_only) {
        (fixed_only, 0.0)
    } else {
        (0.0, per_id_only)
    }
}

impl GpuPeer {
    /// Lay out and pin a wave for this peer's team.
    ///
    /// The wave's width is [`GpuPeer::team_size`], which every submission
    /// runs on: the device refuses a span whose width differs from the team
    /// running it.
    pub fn create_wave(&mut self, spec: &WaveSpec) -> Result<Wave, GpuPeerError> {
        let width = self.team_size();
        let (budget_ns, watchdog_basis) = match spec.slice_budget {
            SliceBudget::Detected => {
                let state = super::watchdog::detect(self.device_ordinal);
                let budget = budget_from_watchdog(state.delay_ns, self.quantum_ns, self.barrier_deadline_ns)?;
                (budget, Some(state.basis))
            }
            SliceBudget::Unbounded => (0, None),
            SliceBudget::Fixed(d) => (
                nanos_within(d, u64::MAX, "the slice budget does not fit in nanoseconds")?.max(1),
                None,
            ),
        };
        let layout = WaveLayout::new(width, spec)?;
        let bytes = initial_span(spec, &layout, budget_ns)?;
        let handle = self.pin_bulk(&bytes)?;
        Ok(Wave { handle, layout, budget_ns, watchdog_basis, in_flight: Cell::new(None) })
    }

    /// Run a slice of `wave` through the user opcode `op`, whose source
    /// follows the `flw_` loop. `args` reach the op at its payload. The wave
    /// keeps the ticket, so host pushes and reports wait until it retires.
    pub fn submit_wave(&mut self, wave: &Wave, op: u32, args: &[u8]) -> Result<Ticket, GpuPeerError> {
        let ticket = self.submit_user(op, Some(&wave.handle), args)?;
        wave.in_flight.set(Some(ticket));
        Ok(ticket)
    }

    /// Read a wave's state from its span. Call it between slices: a slice in
    /// flight is still writing what this reads.
    pub fn wave_stats(&mut self, wave: &Wave) -> Result<WaveStats, GpuPeerError> {
        let mut bytes = vec![0u8; HEADER_BYTES + wave.layout.width as usize * TABLE_STRIDE];
        self.fetch_bulk(&wave.handle, &mut bytes)?;
        WaveStats::decode(&bytes, wave.layout.width)
    }

    /// Return a wave's span to the resident pool once no slice of it is in
    /// flight.
    pub fn release_wave(&mut self, wave: Wave) -> Result<(), GpuPeerError> {
        self.unpin(wave.handle)
    }

    /// Every row of `wave`'s ROB, in root order. Call it between slices.
    pub fn wave_rows(&mut self, wave: &Wave) -> Result<Vec<RowState>, GpuPeerError> {
        let layout = wave.layout;
        if layout.rob_capacity == 0 {
            return Err(GpuPeerError::Unavailable("the wave keeps no ROB"));
        }
        let mut bytes = vec![0u8; layout.row_count as usize * ROW_STRIDE];
        self.fetch_bulk_at(&wave.handle, layout.rows_off as usize, &mut bytes)?;
        Ok((0..layout.row_count as usize)
            .map(|row| RowState::decode(&bytes, row * ROW_STRIDE))
            .collect())
    }

    /// Link host-run children into `wave`'s ROB and append them to its
    /// pending frontier, as `flw_push_child` does on the device.
    ///
    /// Each `(parent, id)` links `id` as the next child of `parent`, so a
    /// parent's children go in ordinal order, before the parent is reported
    /// [`SegmentReport::Expanded`]. A global frontier takes the ids after its
    /// pending range; a partition gives each to the block with the fewest
    /// pending ids that has room. Refused while the wave's last submitted
    /// slice has not retired.
    pub fn push_wave_segments(&mut self, wave: &Wave, children: &[(u32, u32)]) -> Result<(), GpuPeerError> {
        self.wave_idle(wave)?;
        let layout = wave.layout;
        if layout.rob_capacity == 0 {
            return Err(GpuPeerError::Unavailable("the wave keeps no ROB"));
        }
        let mut prefix = vec![0u8; layout.rows_off as usize];
        self.fetch_bulk(&wave.handle, &mut prefix)?;

        let mut records = BTreeMap::new();
        for &(parent, id) in children {
            if parent == id {
                return Err(GpuPeerError::Unavailable("a segment cannot be its own child"));
            }
            let (ordinal, row, last) = {
                let p = self.rob_record(wave, &mut records, parent)?;
                (get_u32(&p[..], R_CHILDREN), get_u32(&p[..], R_ROW), get_u32(&p[..], R_LAST_CHILD))
            };
            let child = self.rob_record(wave, &mut records, id)?;
            put_u32(&mut child[..], R_PARENT, parent);
            put_u32(&mut child[..], R_ORDINAL, ordinal);
            put_u32(&mut child[..], R_ROW, row);
            if ordinal == 0 {
                let p = self.rob_record(wave, &mut records, parent)?;
                put_u32(&mut p[..], R_FIRST_CHILD, id);
            } else {
                let previous = self.rob_record(wave, &mut records, last)?;
                put_u32(&mut previous[..], R_NEXT_SIBLING, id);
            }
            let p = self.rob_record(wave, &mut records, parent)?;
            put_u32(&mut p[..], R_LAST_CHILD, id);
            put_u32(&mut p[..], R_CHILDREN, ordinal + 1);
        }

        // Runs of ids to write into the id array, each from its first index.
        let mut runs: Vec<(usize, Vec<u32>)> = Vec::new();
        if get_u32(&prefix, MODE_OFF) == MODE_GLOBAL {
            let push = get_u32(&prefix, PUSH_OFF);
            let end = get_u32(&prefix, END_OFF);
            if push != end {
                return Err(GpuPeerError::Unavailable(
                    "the wave's frontier holds pushes past its range, so it is not between slices",
                ));
            }
            if push as usize + children.len() > layout.id_capacity as usize {
                return Err(GpuPeerError::Unavailable("the host push exceeds the wave's id capacity"));
            }
            let added = children.len() as u32;
            put_u32(&mut prefix, PUSH_OFF, push + added);
            put_u32(&mut prefix, END_OFF, end + added);
            runs.push((push as usize, children.iter().map(|&(_, id)| id).collect()));
        } else {
            let width = layout.width as usize;
            let per_block = (layout.id_capacity / layout.width) as usize;
            let entry = |b: usize| layout.table_off as usize + b * TABLE_STRIDE;
            if (0..width).any(|b| get_u32(&prefix, entry(b) + T_PUSH) != get_u32(&prefix, entry(b) + T_END)) {
                return Err(GpuPeerError::Unavailable(
                    "a block's frontier holds pushes past its range, so the wave is not between slices",
                ));
            }
            let starts: Vec<usize> = (0..width).map(|b| get_u32(&prefix, entry(b) + T_END) as usize).collect();
            let mut added: Vec<Vec<u32>> = vec![Vec::new(); width];
            for &(_, id) in children {
                let mut best: Option<(usize, u32)> = None;
                for b in 0..width {
                    let end = get_u32(&prefix, entry(b) + T_END);
                    if end as usize >= per_block {
                        continue;
                    }
                    let pending = end.saturating_sub(get_u32(&prefix, entry(b) + T_START));
                    if best.is_none_or(|(_, least)| pending < least) {
                        best = Some((b, pending));
                    }
                }
                let Some((b, _)) = best else {
                    return Err(GpuPeerError::Unavailable("every block's share of the wave's ids is full"));
                };
                let end = get_u32(&prefix, entry(b) + T_END);
                put_u32(&mut prefix, entry(b) + T_END, end + 1);
                put_u32(&mut prefix, entry(b) + T_PUSH, end + 1);
                put_u32(&mut prefix, entry(b) + T_EMPTY, 0);
                added[b].push(id);
            }
            for (b, ids) in added.into_iter().enumerate() {
                if !ids.is_empty() {
                    runs.push((b * per_block + starts[b], ids));
                }
            }
        }

        self.write_rob_records(wave, &records)?;
        for (first, ids) in &runs {
            let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            self.write_resident_bulk_at(&wave.handle, layout.ids_off as usize + first * 4, &bytes)?;
        }
        self.write_resident_bulk(&wave.handle, &prefix)
    }

    /// Record what host-run segments of `wave` did, as `flw_rob_expanded`,
    /// `flw_rob_refuse` and `flw_rob_retired` do on the device. The rows move
    /// at block 0's next walk, so a slice submitted with nothing pending
    /// walks them and returns. Refused while the wave's last submitted slice
    /// has not retired.
    pub fn report_wave_segments(
        &mut self,
        wave: &Wave,
        reports: &[(u32, SegmentReport)],
    ) -> Result<(), GpuPeerError> {
        self.wave_idle(wave)?;
        let layout = wave.layout;
        if layout.rob_capacity == 0 {
            return Err(GpuPeerError::Unavailable("the wave keeps no ROB"));
        }
        let mut rows = vec![0u8; layout.row_count as usize * ROW_STRIDE];
        self.fetch_bulk_at(&wave.handle, layout.rows_off as usize, &mut rows)?;
        let mut records = BTreeMap::new();
        for &(id, report) in reports {
            let record = self.rob_record(wave, &mut records, id)?;
            let row = get_u32(&record[..], R_ROW);
            if row >= layout.row_count {
                return Err(GpuPeerError::Unavailable("a segment's record names no row of the wave"));
            }
            let at = row as usize * ROW_STRIDE;
            let root = get_u32(&rows, at + W_ROOT) == id;
            if get_u32(&record[..], R_PARENT) == ROB_NONE && !root {
                return Err(GpuPeerError::Unavailable("a reported segment was never linked into the wave's ROB"));
            }
            match report {
                SegmentReport::Expanded => put_u32(&mut record[..], R_EXPANDED, 1),
                SegmentReport::Refused => {
                    put_u32(&mut record[..], R_REFUSED, 1);
                    let comp = u32::MAX - id;
                    if comp > get_u32(&rows, at + W_REFUSED_COMP) {
                        put_u32(&mut rows, at + W_REFUSED_COMP, comp);
                    }
                }
                SegmentReport::Retired => {
                    put_u32(&mut record[..], R_RETIRED, 1);
                    if root {
                        put_u32(&mut rows, at + W_ROOT_RETIRED, 1);
                    }
                }
            }
        }
        self.write_rob_records(wave, &records)?;
        self.write_resident_bulk_at(&wave.handle, layout.rows_off as usize, &rows)
    }

    /// An error while the wave's last submitted slice has not retired.
    fn wave_idle(&self, wave: &Wave) -> Result<(), GpuPeerError> {
        match wave.in_flight.get() {
            Some(ticket) if !self.is_done(ticket) => {
                Err(GpuPeerError::Unavailable("a slice of this wave has not retired"))
            }
            Some(_) | None => Ok(()),
        }
    }

    /// The ROB record for `id`, read from the span on first use and kept in
    /// `records` for the host's edits.
    fn rob_record<'a>(
        &mut self,
        wave: &Wave,
        records: &'a mut BTreeMap<u32, [u8; ROB_STRIDE]>,
        id: u32,
    ) -> Result<&'a mut [u8; ROB_STRIDE], GpuPeerError> {
        if id >= wave.layout.rob_capacity {
            return Err(GpuPeerError::Unavailable("a segment id lies outside the wave's ROB"));
        }
        match records.entry(id) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let mut record = [0u8; ROB_STRIDE];
                self.fetch_bulk_at(
                    &wave.handle,
                    wave.layout.rob_off as usize + id as usize * ROB_STRIDE,
                    &mut record,
                )?;
                Ok(entry.insert(record))
            }
        }
    }

    /// Write every record in `records` back to `wave`'s span.
    fn write_rob_records(
        &mut self,
        wave: &Wave,
        records: &BTreeMap<u32, [u8; ROB_STRIDE]>,
    ) -> Result<(), GpuPeerError> {
        for (&id, record) in records {
            self.write_resident_bulk_at(
                &wave.handle,
                wave.layout.rob_off as usize + id as usize * ROB_STRIDE,
                record,
            )?;
        }
        Ok(())
    }

    /// The wave costs for this device at this team size, from
    /// [`GpuPeer::calibrate_waves`] or the device's stored record.
    pub fn wave_costs(&self) -> Option<WaveCosts> {
        self.wave_costs
    }

    /// Measure this device's wave costs at [`GpuPeer::team_size`] and keep
    /// them.
    ///
    /// Flynnel's calibration op ([`layout::OP_WAVE_CALIBRATE`]) walks a
    /// binary segment tree whose segments only push their children, from 64
    /// roots to depths 2, 4, 6 and 8. Each depth runs as a global frontier,
    /// a partition that never rebalances and a partition that rebalances
    /// every generation, five times each, and the medians are kept:
    ///
    /// - `fixed_ns` and `segment_ps`: the least-squares line through the
    ///   global frontier's host round trip against its segment count;
    /// - `barrier_ns`: the global frontier's device time above the
    ///   non-rebalancing partition's, summed over every depth, per
    ///   generation;
    /// - `rebalance_fixed_ns` and `copy_ps_per_id`: the non-negative
    ///   least-squares fit of block 0's time in rebalances against the
    ///   rebalances run and the ids they moved, over every depth of the
    ///   partition that rebalances every generation;
    /// - `start_skew_ns`: the global frontier's first-barrier wait;
    /// - `generation_ns`: the longest generation any run measured.
    ///
    /// A difference that comes out negative is kept as zero. The costs stay
    /// on this peer, and with the `persisted-calibration` feature they are
    /// written to the device's stored record, where the next peer on the
    /// device reads them.
    ///
    /// Needs user ops composed through [`super::GpuPeerConfig::user_ops_cuda`],
    /// which is what brings the helpers and the calibration op into the
    /// poller module.
    pub fn calibrate_waves(&mut self) -> Result<WaveCosts, GpuPeerError> {
        if !self.user_ops {
            return Err(GpuPeerError::Unavailable(
                "wave calibration needs user ops composed through GpuPeerConfig::user_ops_cuda",
            ));
        }
        let width = self.team_size();
        let mut wall_points = Vec::with_capacity(CALIBRATION_DEPTHS.len());
        let mut skew_per_depth = Vec::with_capacity(CALIBRATION_DEPTHS.len());
        let mut global_extra_ns = 0u64;
        let mut generations = 0u64;
        let mut rebalance_points = Vec::with_capacity(CALIBRATION_DEPTHS.len());
        let mut longest = 0u64;
        for depth in CALIBRATION_DEPTHS {
            let segments = u64::from(CALIBRATION_ROOTS) * ((1u64 << (depth + 1)) - 1);
            let depth_generations = u64::from(depth + 1);
            let global = self.calibration_run(Frontier::Global, depth, segments)?;
            let partitioned = segments * u64::from(width);
            let never = self.calibration_run(Frontier::Partition { rebalance_every: None }, depth, partitioned)?;
            let every =
                self.calibration_run(Frontier::Partition { rebalance_every: NonZeroU32::new(1) }, depth, partitioned)?;
            wall_points.push((segments as f64, global.wall_ns as f64));
            skew_per_depth.push(global.skew_ns);
            global_extra_ns = global_extra_ns.saturating_add(global.elapsed_ns.saturating_sub(never.elapsed_ns));
            generations += depth_generations;
            rebalance_points.push((every.rebalances as f64, every.moved_ids as f64, every.rebalance_ns as f64));
            longest = longest.max(global.longest_ns).max(never.longest_ns).max(every.longest_ns);
        }
        let barrier_ns = global_extra_ns / generations.max(1);
        let (rebalance_fixed, copy_per_id) = rebalance_fit(&rebalance_points);
        let (fixed, slope) = least_squares(&wall_points);
        let costs = WaveCosts {
            width,
            barrier_ns,
            fixed_ns: fixed.max(0.0) as u64,
            segment_ps: (slope * 1000.0).max(0.0) as u64,
            rebalance_fixed_ns: rebalance_fixed as u64,
            copy_ps_per_id: (copy_per_id * 1000.0) as u64,
            start_skew_ns: median(&skew_per_depth),
            generation_ns: longest,
        };
        self.wave_costs = Some(costs);
        persist_wave_costs(self.device_ordinal, costs);
        Ok(costs)
    }

    /// Run one calibration configuration [`CALIBRATION_REPEATS`] times and
    /// keep the medians.
    fn calibration_run(&mut self, frontier: Frontier, depth: u32, id_capacity: u64) -> Result<CalibrationRun, GpuPeerError> {
        let spec = WaveSpec {
            roots: (0..CALIBRATION_ROOTS).collect(),
            id_capacity: span_offset(id_capacity)?,
            arena_bytes: 0,
            frontier,
            resume: Resume::Host,
            slice_budget: SliceBudget::Unbounded,
            barrier_deadline: Duration::from_millis(50),
            done_deadline: Some(Duration::from_secs(2)),
            longest_generation_seed: Duration::ZERO,
            rob: None,
        };
        let mut walls = Vec::with_capacity(CALIBRATION_REPEATS);
        let mut elapsed = Vec::with_capacity(CALIBRATION_REPEATS);
        let mut moved = Vec::with_capacity(CALIBRATION_REPEATS);
        let mut skews = Vec::with_capacity(CALIBRATION_REPEATS);
        let mut rebalances = Vec::with_capacity(CALIBRATION_REPEATS);
        let mut rebalance_times = Vec::with_capacity(CALIBRATION_REPEATS);
        let mut longest = 0u64;
        for _ in 0..CALIBRATION_REPEATS {
            let wave = self.create_wave(&spec)?;
            let started = Instant::now();
            let ticket = self.submit_wave(&wave, layout::OP_WAVE_CALIBRATE, &depth.to_le_bytes())?;
            let status = self.wait_status(ticket, Duration::from_secs(30))?;
            let wall = started.elapsed();
            self.reap(ticket)?;
            let stats = self.wave_stats(&wave)?;
            self.release_wave(wave)?;
            if status != STATUS_DONE || stats.slice_state != SliceState::Finished {
                eprintln!(
                    "flynnel gpu_peer: a wave calibration run ({frontier:?}, depth {depth}) \
                     retired with status {status}: {stats:?}"
                );
                return Err(GpuPeerError::Unavailable("a wave calibration run did not finish"));
            }
            let wall_ns = wall.as_nanos();
            walls.push(if wall_ns > u128::from(u64::MAX) { u64::MAX } else { wall_ns as u64 });
            elapsed.push(stats.elapsed_ns);
            moved.push(u64::from(stats.moved_ids));
            skews.push(u64::from(stats.start_skew_max_ns));
            rebalances.push(u64::from(stats.rebalances));
            rebalance_times.push(stats.rebalance_ns);
            longest = longest.max(u64::from(stats.longest_generation_ns));
        }
        Ok(CalibrationRun {
            wall_ns: median(&walls),
            elapsed_ns: median(&elapsed),
            moved_ids: median(&moved),
            rebalances: median(&rebalances),
            rebalance_ns: median(&rebalance_times),
            skew_ns: median(&skews),
            longest_ns: longest,
        })
    }
}

/// The wave costs stored for device `ordinal`, when they were measured at
/// `width`.
#[cfg(feature = "persisted-calibration")]
pub(crate) fn stored_wave_costs(ordinal: usize, width: u32) -> Option<WaveCosts> {
    use crate::sched::calibration_store::{CalibrationStore, HostStamp, calibration_dir};

    let dir = calibration_dir()?;
    let store = match CalibrationStore::open_or_create(&dir, &HostStamp::detect()) {
        Ok(store) => store,
        Err(err) => {
            eprintln!(
                "flynnel gpu_peer: the calibration table under {} is unusable ({err:?}), \
                 so no stored wave costs are read",
                dir.display()
            );
            return None;
        }
    };
    let (_cpu, devices) = store.read()?;
    let record = devices.into_iter().find(|d| d.ordinal as usize == ordinal)?.wave()?;
    if record.width != width {
        return None;
    }
    Some(WaveCosts {
        width: record.width,
        barrier_ns: record.barrier_ns,
        fixed_ns: record.fixed_ns,
        segment_ps: record.segment_ps,
        rebalance_fixed_ns: record.rebalance_fixed_ns,
        copy_ps_per_id: record.copy_ps_per_id,
        start_skew_ns: u64::from(record.skew_ns),
        generation_ns: record.generation_ns,
    })
}

#[cfg(not(feature = "persisted-calibration"))]
pub(crate) fn stored_wave_costs(_ordinal: usize, _width: u32) -> Option<WaveCosts> {
    None
}

/// Write `costs` to device `ordinal`'s stored record. A table that cannot be
/// written leaves the costs on the peer only, and says so.
#[cfg(feature = "persisted-calibration")]
fn persist_wave_costs(ordinal: usize, costs: WaveCosts) {
    use crate::sched::calibration_store::{
        CalibrationStore, HostStamp, StoreError, WaveCostRecord, calibration_dir,
    };

    let Some(dir) = calibration_dir() else {
        eprintln!("flynnel gpu_peer: no calibration directory, so the wave costs stay on this peer");
        return;
    };
    let store = match CalibrationStore::open_or_create(&dir, &HostStamp::detect()) {
        Ok(store) => store,
        Err(err) => {
            eprintln!(
                "flynnel gpu_peer: the calibration table under {} is unusable ({err:?}), \
                 so the wave costs stay on this peer",
                dir.display()
            );
            return;
        }
    };
    let writer = match store.try_acquire_writer() {
        Ok(writer) => writer,
        Err(StoreError::WriterActive) => {
            eprintln!(
                "flynnel gpu_peer: another process is writing the calibration table, \
                 so the wave costs stay on this peer"
            );
            return;
        }
        Err(err) => {
            eprintln!("flynnel gpu_peer: the calibration lease failed ({err:?}), so the wave costs stay on this peer");
            return;
        }
    };
    writer.beat();
    let Some((cpu, mut devices)) = store.read() else {
        eprintln!(
            "flynnel gpu_peer: the calibration table could not be read between writers, \
             so the wave costs stay on this peer"
        );
        return;
    };
    let Some(i) = devices.iter().position(|d| d.ordinal as usize == ordinal) else {
        eprintln!(
            "flynnel gpu_peer: device {ordinal} has no stored record to carry the wave costs, \
             so they stay on this peer"
        );
        return;
    };
    devices[i] = devices[i].with_wave(WaveCostRecord {
        width: costs.width,
        barrier_ns: costs.barrier_ns,
        fixed_ns: costs.fixed_ns,
        segment_ps: costs.segment_ps,
        rebalance_fixed_ns: costs.rebalance_fixed_ns,
        copy_ps_per_id: costs.copy_ps_per_id,
        skew_ns: if costs.start_skew_ns > u64::from(u32::MAX) {
            u32::MAX
        } else {
            costs.start_skew_ns as u32
        },
        generation_ns: costs.generation_ns,
    });
    writer.publish(&cpu, &devices);
}

#[cfg(not(feature = "persisted-calibration"))]
fn persist_wave_costs(_ordinal: usize, _costs: WaveCosts) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value of `#define name value` in the wave kernel source.
    fn define(name: &str) -> u64 {
        for line in WAVE_CU.lines() {
            let mut words = line.split_whitespace();
            if words.next() != Some("#define") || words.next() != Some(name) {
                continue;
            }
            let Some(value) = words.next() else {
                panic!("#define {name} has no value");
            };
            let digits = value.trim_end_matches('u');
            let parsed = match digits.strip_prefix("0x") {
                Some(hex) => u64::from_str_radix(hex, 16),
                None => digits.parse::<u64>(),
            };
            return match parsed {
                Ok(v) => v,
                Err(err) => panic!("#define {name} {value} does not parse ({err})"),
            };
        }
        panic!("the wave kernel source defines no {name}");
    }

    #[test]
    fn offsets_match_the_wave_kernel_defines() {
        let pairs: &[(&str, u64)] = &[
            ("FLW_MAGIC", u64::from(MAGIC)),
            ("FLW_VERSION", u64::from(VERSION)),
            ("FLW_MAGIC_OFF", MAGIC_OFF as u64),
            ("FLW_VERSION_OFF", VERSION_OFF as u64),
            ("FLW_WIDTH_OFF", WIDTH_OFF as u64),
            ("FLW_MODE_OFF", MODE_OFF as u64),
            ("FLW_REBALANCE_OFF", REBALANCE_OFF as u64),
            ("FLW_RESUME_MODE_OFF", RESUME_MODE_OFF as u64),
            ("FLW_ARRIVE_OFF", ARRIVE_OFF as u64),
            ("FLW_WAIT_MAX_OFF", WAIT_MAX_OFF as u64),
            ("FLW_WAIT_SUM_KNS_OFF", WAIT_SUM_KNS_OFF as u64),
            ("FLW_TIMEOUTS_OFF", TIMEOUTS_OFF as u64),
            ("FLW_SKEW_MAX_OFF", SKEW_MAX_OFF as u64),
            ("FLW_DONE_OFF", DONE_OFF as u64),
            ("FLW_FAIL_COMP_OFF", FAIL_COMP_OFF as u64),
            ("FLW_FAIL_CODE_OFF", FAIL_CODE_OFF as u64),
            ("FLW_PUSH_OFF", PUSH_OFF as u64),
            ("FLW_START_OFF", START_OFF as u64),
            ("FLW_END_OFF", END_OFF as u64),
            ("FLW_ID_CAPACITY_OFF", ID_CAPACITY_OFF as u64),
            ("FLW_ARENA_BUMP_OFF", ARENA_BUMP_OFF as u64),
            ("FLW_ARENA_CAPACITY_OFF", ARENA_CAPACITY_OFF as u64),
            ("FLW_STOP_OFF", STOP_OFF as u64),
            ("FLW_SLICE_STATE_OFF", SLICE_STATE_OFF as u64),
            ("FLW_BUDGET_NS_OFF", BUDGET_NS_OFF as u64),
            ("FLW_LONGEST_GEN_OFF", LONGEST_GEN_OFF as u64),
            ("FLW_YIELDS_OFF", YIELDS_OFF as u64),
            ("FLW_SLICES_OFF", SLICES_OFF as u64),
            ("FLW_GENERATIONS_OFF", GENERATIONS_OFF as u64),
            ("FLW_IDS_OFF_OFF", IDS_OFF_OFF as u64),
            ("FLW_STAGING_OFF_OFF", STAGING_OFF_OFF as u64),
            ("FLW_ARENA_OFF_OFF", ARENA_OFF_OFF as u64),
            ("FLW_TABLE_OFF_OFF", TABLE_OFF_OFF as u64),
            ("FLW_BARRIER_DEADLINE_OFF", BARRIER_DEADLINE_OFF as u64),
            ("FLW_IMBALANCE_OFF", IMBALANCE_OFF as u64),
            ("FLW_DONE_DEADLINE_NS_OFF", DONE_DEADLINE_NS_OFF as u64),
            ("FLW_REBALANCES_OFF", REBALANCES_OFF as u64),
            ("FLW_MOVED_OFF", MOVED_OFF as u64),
            ("FLW_ELAPSED_NS_OFF", ELAPSED_NS_OFF as u64),
            ("FLW_REBALANCE_NS_OFF", REBALANCE_NS_OFF as u64),
            ("FLW_ROB_OFF_OFF", ROB_OFF_OFF as u64),
            ("FLW_ROB_CAPACITY_OFF", ROB_CAPACITY_OFF as u64),
            ("FLW_ROWS_OFF_OFF", ROWS_OFF_OFF as u64),
            ("FLW_ROW_COUNT_OFF", ROW_COUNT_OFF as u64),
            ("FLW_ROB_COMMITTED_OFF", ROB_COMMITTED_OFF as u64),
            ("FLW_HEADER_BYTES", HEADER_BYTES as u64),
            ("FLW_TABLE_STRIDE", TABLE_STRIDE as u64),
            ("FLW_T_PUSH", T_PUSH as u64),
            ("FLW_T_START", T_START as u64),
            ("FLW_T_END", T_END as u64),
            ("FLW_T_DECISION", T_DECISION as u64),
            ("FLW_T_EMPTY", T_EMPTY as u64),
            ("FLW_T_PENDING", T_PENDING as u64),
            ("FLW_T_PREFIX", T_PREFIX as u64),
            ("FLW_T_GENERATION", T_GENERATION as u64),
            ("FLW_T_TOTAL", T_TOTAL as u64),
            ("FLW_T_DEAL_LO", T_DEAL_LO as u64),
            ("FLW_T_DEAL_HI", T_DEAL_HI as u64),
            ("FLW_ROB_STRIDE", ROB_STRIDE as u64),
            ("FLW_R_PARENT", R_PARENT as u64),
            ("FLW_R_ORDINAL", R_ORDINAL as u64),
            ("FLW_R_ROW", R_ROW as u64),
            ("FLW_R_FIRST_CHILD", R_FIRST_CHILD as u64),
            ("FLW_R_NEXT_SIBLING", R_NEXT_SIBLING as u64),
            ("FLW_R_LAST_CHILD", R_LAST_CHILD as u64),
            ("FLW_R_EXPANDED", R_EXPANDED as u64),
            ("FLW_R_REFUSED", R_REFUSED as u64),
            ("FLW_R_RETIRED", R_RETIRED as u64),
            ("FLW_R_CHILDREN", R_CHILDREN as u64),
            ("FLW_ROW_STRIDE", ROW_STRIDE as u64),
            ("FLW_W_ROOT", W_ROOT as u64),
            ("FLW_W_CURSOR", W_CURSOR as u64),
            ("FLW_W_FLAGS", W_FLAGS as u64),
            ("FLW_W_REFUSED_COMP", W_REFUSED_COMP as u64),
            ("FLW_W_ROOT_RETIRED", W_ROOT_RETIRED as u64),
            ("FLW_W_COMMITTED", W_COMMITTED as u64),
            ("FLW_ROW_COMPLETE", u64::from(ROW_COMPLETE)),
            ("FLW_ROW_REFUSED", u64::from(ROW_REFUSED)),
            ("FLW_ROB_NONE", u64::from(ROB_NONE)),
            ("FLW_MODE_GLOBAL", u64::from(MODE_GLOBAL)),
            ("FLW_MODE_PARTITION", u64::from(MODE_PARTITION)),
            ("FLW_RESUME_DEVICE", u64::from(RESUME_DEVICE)),
            ("FLW_RESUME_HOST", u64::from(RESUME_HOST)),
            ("FLW_SLICE_RUNNING", u64::from(SLICE_RUNNING)),
            ("FLW_SLICE_YIELDED", u64::from(SLICE_YIELDED)),
            ("FLW_SLICE_CONTINUE", u64::from(SLICE_CONTINUE)),
            ("FLW_SLICE_FINISHED", u64::from(SLICE_FINISHED)),
            ("FLW_SLICE_FAILED", u64::from(SLICE_FAILED)),
            ("FLW_NO_SEGMENT", u64::from(NO_SEGMENT)),
            ("FLW_FAIL_IDS", u64::from(FAIL_IDS)),
            ("FLW_FAIL_ARENA", u64::from(FAIL_ARENA)),
            ("FLW_FAIL_BARRIER", u64::from(FAIL_BARRIER)),
            ("FLW_FAIL_DONE", u64::from(FAIL_DONE)),
            ("FLW_FAIL_BUDGET", u64::from(FAIL_BUDGET)),
            ("FLW_FAIL_ROB", u64::from(FAIL_ROB)),
            ("FLW_STATUS_FAILED", u64::from(STATUS_FAILED)),
            ("FLW_STATUS_BAD_SPAN", u64::from(STATUS_BAD_SPAN)),
        ];
        for (name, rust) in pairs {
            assert_eq!(define(name), *rust, "{name} differs between the kernel and wave.rs");
        }
        // The calibration opcode is defined in the poller kernel.
        let kernel = super::super::PEER_CU;
        assert!(
            kernel.contains(&format!("#define FLW_OP_CALIBRATE 0x{:08X}u", layout::OP_WAVE_CALIBRATE)),
            "FLW_OP_CALIBRATE in gpu_peer.cu differs from layout::OP_WAVE_CALIBRATE"
        );
    }

    fn spec(frontier: Frontier, roots: usize, id_capacity: u32) -> WaveSpec {
        WaveSpec {
            roots: (0..roots as u32).map(|i| 1000 + i).collect(),
            id_capacity,
            arena_bytes: 100,
            frontier,
            resume: Resume::Device,
            slice_budget: SliceBudget::Unbounded,
            barrier_deadline: Duration::from_millis(2),
            done_deadline: None,
            longest_generation_seed: Duration::ZERO,
            rob: None,
        }
    }

    #[test]
    fn a_global_layout_places_each_part_after_the_last() {
        let layout = WaveLayout::new(4, &spec(Frontier::Global, 3, 10)).expect("fits");
        assert_eq!(layout.table_off as usize, HEADER_BYTES);
        assert_eq!(layout.ids_off as usize, HEADER_BYTES + 4 * TABLE_STRIDE);
        assert_eq!(layout.staging_off, layout.ids_off + 10 * 4);
        assert_eq!(layout.staging_ids, 0, "a global frontier never rebalances");
        assert_eq!(layout.arena_off % 8, 0);
        assert!(layout.arena_off >= layout.staging_off);
        assert_eq!(layout.total_bytes, layout.arena_off + 100);
    }

    #[test]
    fn a_partition_rounds_its_capacity_up_and_stages_only_when_it_rebalances() {
        let never = WaveLayout::new(4, &spec(Frontier::Partition { rebalance_every: None }, 3, 10))
            .expect("fits");
        assert_eq!(never.id_capacity, 12, "rounded up to a multiple of the team size, never down");
        assert_eq!(never.staging_ids, 0);
        let rebalancing = WaveLayout::new(
            4,
            &spec(Frontier::Partition { rebalance_every: NonZeroU32::new(3) }, 3, 10),
        )
        .expect("fits");
        assert_eq!(rebalancing.staging_ids, 12);
        assert!(rebalancing.arena_off >= rebalancing.staging_off + 12 * 4);
    }

    #[test]
    fn a_global_span_holds_the_roots_as_the_first_generation() {
        let s = spec(Frontier::Global, 3, 10);
        let layout = WaveLayout::new(2, &s).expect("fits");
        let span = initial_span(&s, &layout, 7).expect("encodes");
        assert_eq!(get_u32(&span, MAGIC_OFF), MAGIC);
        assert_eq!(get_u32(&span, WIDTH_OFF), 2);
        assert_eq!(get_u32(&span, START_OFF), 0);
        assert_eq!(get_u32(&span, END_OFF), 3);
        assert_eq!(get_u32(&span, PUSH_OFF), 3);
        let ids = layout.ids_off as usize;
        assert_eq!((0..3).map(|i| get_u32(&span, ids + i * 4)).collect::<Vec<_>>(), vec![1000, 1001, 1002]);
        assert_eq!(get_u64(&span, BUDGET_NS_OFF), 7);
    }

    #[test]
    fn a_partition_span_deals_the_roots_evenly_across_block_regions() {
        let s = spec(Frontier::Partition { rebalance_every: None }, 7, 12);
        let layout = WaveLayout::new(3, &s).expect("fits");
        let span = initial_span(&s, &layout, 0).expect("encodes");
        let per_block = (layout.id_capacity / 3) as usize;
        let mut seen = Vec::new();
        for b in 0..3usize {
            let entry = layout.table_off as usize + b * TABLE_STRIDE;
            let len = get_u32(&span, entry + T_END) as usize;
            assert_eq!(get_u32(&span, entry + T_PUSH) as usize, len);
            assert!((2..=3).contains(&len), "block {b} got {len} of 7 roots");
            let region = layout.ids_off as usize + b * per_block * 4;
            seen.extend((0..len).map(|i| get_u32(&span, region + i * 4)));
        }
        assert_eq!(seen, (1000..1007).collect::<Vec<_>>(), "every root once, in order");
    }

    #[test]
    fn roots_beyond_the_capacity_are_refused() {
        let s = spec(Frontier::Global, 11, 10);
        let layout = WaveLayout::new(1, &s).expect("fits");
        assert!(initial_span(&s, &layout, 0).is_err());
    }

    #[test]
    fn the_budget_is_the_watchdog_delay_less_the_quantum_and_the_kernel_barrier() {
        assert_eq!(budget_from_watchdog(None, 250_000_000, 5_000_000).expect("no watchdog"), 0);
        assert_eq!(
            budget_from_watchdog(Some(2_000_000_000), 250_000_000, 5_000_000).expect("room left"),
            1_745_000_000
        );
        assert!(budget_from_watchdog(Some(200_000_000), 250_000_000, 5_000_000).is_err());
    }

    #[test]
    fn stats_decode_the_lowest_failing_segment_and_each_block() {
        let s = spec(Frontier::Global, 2, 10);
        let layout = WaveLayout::new(2, &s).expect("fits");
        let mut span = initial_span(&s, &layout, 0).expect("encodes");
        put_u32(&mut span, SLICE_STATE_OFF, SLICE_FAILED);
        put_u32(&mut span, FAIL_COMP_OFF, u32::MAX - 7);
        put_u32(&mut span, FAIL_CODE_OFF, 42);
        put_u32(&mut span, WAIT_SUM_KNS_OFF, 3);
        put_u32(&mut span, MOVED_OFF, 11);
        put_u64(&mut span, ELAPSED_NS_OFF, 123_456);
        put_u64(&mut span, REBALANCE_NS_OFF, 7_890);
        let entry = layout.table_off as usize + TABLE_STRIDE;
        put_u32(&mut span, entry + T_GENERATION, 5);
        let stats = WaveStats::decode(&span, 2).expect("decodes");
        assert_eq!(stats.slice_state, SliceState::Failed);
        assert_eq!(stats.failure, Some(WaveFailure { segment: Some(7), code: 42 }));
        assert_eq!(stats.barrier_wait_sum_ns, 3 << 10);
        assert_eq!(stats.moved_ids, 11);
        assert_eq!(stats.elapsed_ns, 123_456);
        assert_eq!(stats.rebalance_ns, 7_890);
        assert_eq!(stats.blocks[1].generations, 5);

        put_u32(&mut span, FAIL_COMP_OFF, u32::MAX - NO_SEGMENT);
        let stats = WaveStats::decode(&span, 2).expect("decodes");
        assert_eq!(stats.failure, Some(WaveFailure { segment: None, code: 42 }));

        put_u32(&mut span, SLICE_STATE_OFF, 99);
        assert!(WaveStats::decode(&span, 2).is_err(), "an unknown slice state is refused");
    }

    #[test]
    fn the_least_squares_line_recovers_a_fixed_and_a_per_segment_cost() {
        let points: Vec<(f64, f64)> = [100.0, 400.0, 1600.0, 6400.0]
            .iter()
            .map(|&x| (x, 5_000.0 + 2.5 * x))
            .collect();
        let (intercept, slope) = least_squares(&points);
        assert!((intercept - 5_000.0).abs() < 1e-6, "{intercept}");
        assert!((slope - 2.5).abs() < 1e-9, "{slope}");
        assert_eq!(median(&[5, 1, 9, 3, 7]), 5);
        assert_eq!(median(&[]), 0);
    }

    #[test]
    fn the_rebalance_fit_recovers_a_fixed_and_a_per_id_cost_and_never_goes_negative() {
        let points: Vec<(f64, f64, f64)> = [(3.0, 20.0), (5.0, 300.0), (7.0, 2_000.0), (9.0, 9_000.0)]
            .iter()
            .map(|&(r, m)| (r, m, 500.0 * r + 3.0 * m))
            .collect();
        let (fixed, per_id) = rebalance_fit(&points);
        assert!((fixed - 500.0).abs() < 1e-6, "{fixed}");
        assert!((per_id - 3.0).abs() < 1e-9, "{per_id}");

        // Time that falls as more ids move cannot be a per-id cost.
        let falling: Vec<(f64, f64, f64)> = [(3.0, 20.0, 9_000.0), (5.0, 300.0, 8_000.0), (7.0, 2_000.0, 7_000.0)].to_vec();
        let (fixed, per_id) = rebalance_fit(&falling);
        assert!(fixed >= 0.0 && per_id >= 0.0, "{fixed} {per_id}");
        assert_eq!(rebalance_fit(&[]), (0.0, 0.0));
    }

    #[test]
    fn plan_inputs_take_the_imbalance_over_the_interval_the_wave_ran() {
        let costs = WaveCosts {
            width: 4,
            barrier_ns: 2_000,
            fixed_ns: 100_000,
            segment_ps: 500,
            rebalance_fixed_ns: 4_000,
            copy_ps_per_id: 3_000,
            start_skew_ns: 0,
            generation_ns: 1_000,
        };
        let s = spec(Frontier::Global, 2, 10);
        let layout = WaveLayout::new(4, &s).expect("fits");
        let mut span = initial_span(&s, &layout, 0).expect("encodes");
        put_u32(&mut span, GENERATIONS_OFF, 10);
        put_u32(&mut span, PUSH_OFF, 1_000);
        put_u32(&mut span, IMBALANCE_OFF, 1_500);
        put_u32(&mut span, LONGEST_GEN_OFF, 9_000);
        let stats = WaveStats::decode(&span, 4).expect("decodes");

        let global = costs.plan_inputs(&stats, Frontier::Global);
        assert_eq!(global.imbalance, Some(plan::Imbalance { per_mille: 1_500, over_generations: 1 }));
        assert_eq!(global.pending_ids, 100.0, "1000 pushed over 10 generations");
        assert_eq!(global.copy_ns_per_id, 3.0);
        assert_eq!(global.rebalance_fixed_ns, 4_000.0);
        assert_eq!(global.generation_ns, 9_000.0);

        let every = costs.plan_inputs(&stats, Frontier::Partition { rebalance_every: NonZeroU32::new(4) });
        assert_eq!(every.imbalance.map(|i| i.over_generations), Some(4));
    }

    fn rob_spec(roots: Vec<u32>, capacity: u32) -> WaveSpec {
        let mut s = spec(Frontier::Global, 0, 64);
        s.roots = roots;
        s.rob = Some(RobSpec { capacity });
        s
    }

    #[test]
    fn a_rob_layout_places_rows_after_the_table_and_records_before_the_arena() {
        let layout = WaveLayout::new(2, &rob_spec(vec![0, 5, 9], 16)).expect("fits");
        assert_eq!(layout.rows_off as usize, HEADER_BYTES + 2 * TABLE_STRIDE);
        assert_eq!(layout.row_count, 3);
        assert_eq!(layout.ids_off, layout.rows_off + 3 * ROW_STRIDE as u32);
        assert_eq!(layout.rob_capacity, 16);
        assert!(layout.rob_off >= layout.staging_off + layout.staging_ids * 4);
        assert_eq!(layout.rob_off % 8, 0);
        assert_eq!(layout.arena_off, (layout.rob_off + 16 * ROB_STRIDE as u32).div_ceil(8) * 8);

        let plain = WaveLayout::new(2, &spec(Frontier::Global, 3, 64)).expect("fits");
        assert_eq!((plain.row_count, plain.rob_capacity), (0, 0));
        assert_eq!(plain.ids_off, plain.rows_off, "a wave without a ROB has no rows");
    }

    #[test]
    fn a_rob_span_starts_each_row_at_its_root_with_every_link_empty() {
        let s = rob_spec(vec![4, 1], 8);
        let layout = WaveLayout::new(1, &s).expect("fits");
        let span = initial_span(&s, &layout, 0).expect("encodes");
        assert_eq!(get_u32(&span, ROB_OFF_OFF), layout.rob_off);
        assert_eq!(get_u32(&span, ROB_CAPACITY_OFF), 8);
        assert_eq!(get_u32(&span, ROWS_OFF_OFF), layout.rows_off);
        assert_eq!(get_u32(&span, ROW_COUNT_OFF), 2);
        for (row, root) in [(0usize, 4u32), (1, 1)] {
            let state = RowState::decode(&span, layout.rows_off as usize + row * ROW_STRIDE);
            assert_eq!(state.root, root);
            assert_eq!(state.committed_through, Some(root));
            assert!(!state.complete && !state.root_retired);
            assert_eq!((state.first_refused, state.lowest_refused, state.committed), (None, None, 0));
            let r = layout.rob_off as usize + root as usize * ROB_STRIDE;
            assert_eq!(get_u32(&span, r + R_ROW), row as u32);
        }
        for id in 0..8usize {
            let r = layout.rob_off as usize + id * ROB_STRIDE;
            for field in [R_PARENT, R_FIRST_CHILD, R_NEXT_SIBLING, R_LAST_CHILD] {
                assert_eq!(get_u32(&span, r + field), ROB_NONE, "record {id} field {field:#x}");
            }
            assert_eq!(get_u32(&span, r + R_EXPANDED), 0);
        }
    }

    #[test]
    fn a_rob_refuses_repeated_roots_roots_past_its_capacity_and_an_unusable_capacity() {
        let repeated = rob_spec(vec![2, 2], 8);
        let layout = WaveLayout::new(1, &repeated).expect("fits");
        assert!(initial_span(&repeated, &layout, 0).is_err());
        let past = rob_spec(vec![8], 8);
        let layout = WaveLayout::new(1, &past).expect("fits");
        assert!(initial_span(&past, &layout, 0).is_err());
        assert!(WaveLayout::new(1, &rob_spec(vec![0], 0)).is_err());
        assert!(WaveLayout::new(1, &rob_spec(vec![0], ROB_NONE)).is_err());
    }

    #[test]
    fn a_row_decodes_its_frontier_refusals_and_retired_root() {
        let mut bytes = vec![0u8; ROW_STRIDE];
        put_u32(&mut bytes, W_ROOT, 3);
        put_u32(&mut bytes, W_CURSOR, 11);
        put_u32(&mut bytes, W_FLAGS, ROW_REFUSED);
        put_u32(&mut bytes, W_REFUSED_COMP, u32::MAX - 9);
        put_u32(&mut bytes, W_ROOT_RETIRED, 1);
        put_u32(&mut bytes, W_COMMITTED, 4);
        let row = RowState::decode(&bytes, 0);
        assert_eq!(row.committed_through, Some(11));
        assert_eq!(row.first_refused, Some(11));
        assert_eq!(row.lowest_refused, Some(9));
        assert!(row.root_retired && !row.complete);
        assert_eq!(row.committed, 4);

        put_u32(&mut bytes, W_CURSOR, ROB_NONE);
        put_u32(&mut bytes, W_FLAGS, ROW_COMPLETE);
        put_u32(&mut bytes, W_REFUSED_COMP, 0);
        let row = RowState::decode(&bytes, 0);
        assert_eq!((row.committed_through, row.first_refused, row.lowest_refused), (None, None, None));
        assert!(row.complete);
    }
}

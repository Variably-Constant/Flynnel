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
//!
//! The consumer's op follows the loop shown at the top of the kernel file.
//! Every offset here mirrors a define there, and a test compares the two.

use std::num::NonZeroU32;
use std::time::Duration;

use super::{GpuPeer, GpuPeerError, ResidentHandle, Ticket};

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
/// Largest pending run over the mean at a rebalance, per mille.
pub const IMBALANCE_OFF: usize = 0x84;
/// How long block 0 waits at slice end, ns (u64); 0 is no limit.
pub const DONE_DEADLINE_NS_OFF: usize = 0x88;
/// Rebalances run.
pub const REBALANCES_OFF: usize = 0x90;
/// Header size.
pub const HEADER_BYTES: usize = 0x100;

/// Per-block table entry size.
pub const TABLE_STRIDE: usize = 0x40;
/// Partition: ids pushed into the block's region.
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
/// Rebalance: staging offset of the block's run.
pub const T_PREFIX: usize = 0x18;
/// Generations the block completed.
pub const T_GENERATION: usize = 0x1C;
/// Rebalance: pending ids over every block.
pub const T_TOTAL: usize = 0x20;
/// Rebalance: staging start dealt to the block.
pub const T_DEAL_LO: usize = 0x24;
/// Rebalance: staging end dealt to the block.
pub const T_DEAL_HI: usize = 0x28;

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
    /// Start of the id array.
    pub ids_off: u32,
    /// Start of the staging array.
    pub staging_off: u32,
    /// Ids the staging array holds: the id capacity for a partition that
    /// rebalances, and 0 otherwise.
    pub staging_ids: u32,
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
        let table_off = HEADER_BYTES as u64;
        let ids_off = table_off + u64::from(width) * TABLE_STRIDE as u64;
        let staging_off = ids_off + id_capacity * 4;
        let arena_off = (staging_off + staging_ids * 4).div_ceil(8) * 8;
        let total = arena_off + u64::from(spec.arena_bytes);
        Ok(Self {
            width,
            id_capacity: span_offset(id_capacity)?,
            table_off: span_offset(table_off)?,
            ids_off: span_offset(ids_off)?,
            staging_off: span_offset(staging_off)?,
            staging_ids: span_offset(staging_ids)?,
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

/// `d` in nanoseconds, or `too_long` when it does not fit in `limit`.
fn nanos_within(d: Duration, limit: u64, too_long: &'static str) -> Result<u64, GpuPeerError> {
    let ns = d.as_nanos();
    if ns > u128::from(limit) {
        return Err(GpuPeerError::Unavailable(too_long));
    }
    Ok(ns as u64)
}

/// The bytes of a new wave span: header, table and first generation.
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
    /// The largest failure code reported.
    pub code: u32,
}

/// One block's frontier as the table holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockState {
    /// Current generation start.
    pub start: u32,
    /// Current generation end.
    pub end: u32,
    /// Partition: ids pushed into the block's region.
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
    /// Largest pending run over the mean at a rebalance, per mille.
    pub imbalance_per_mille: u32,
    /// Rebalances run.
    pub rebalances: u32,
    /// Arena bytes reserved.
    pub arena_used_bytes: u32,
    /// Global frontier: ids pushed.
    pub pushed: u32,
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
            arena_used_bytes: get_u32(bytes, ARENA_BUMP_OFF),
            pushed: get_u32(bytes, PUSH_OFF),
            failure,
            blocks,
        })
    }
}

/// A wave pinned on a peer.
#[derive(Debug, Clone)]
pub struct Wave {
    handle: ResidentHandle,
    layout: WaveLayout,
    budget_ns: u64,
    watchdog_basis: Option<String>,
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
        Ok(Wave { handle, layout, budget_ns, watchdog_basis })
    }

    /// Run a slice of `wave` through the user opcode `op`, whose source
    /// follows the `flw_` loop. `args` reach the op at its payload.
    pub fn submit_wave(&mut self, wave: &Wave, op: u32, args: &[u8]) -> Result<Ticket, GpuPeerError> {
        self.submit_user(op, Some(&wave.handle), args)
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
}

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
            ("FLW_STATUS_FAILED", u64::from(STATUS_FAILED)),
            ("FLW_STATUS_BAD_SPAN", u64::from(STATUS_BAD_SPAN)),
        ];
        for (name, rust) in pairs {
            assert_eq!(define(name), *rust, "{name} differs between the kernel and wave.rs");
        }
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
        let mut budget = [0u8; 8];
        budget.copy_from_slice(&span[BUDGET_NS_OFF..BUDGET_NS_OFF + 8]);
        assert_eq!(u64::from_le_bytes(budget), 7);
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
        let entry = layout.table_off as usize + TABLE_STRIDE;
        put_u32(&mut span, entry + T_GENERATION, 5);
        let stats = WaveStats::decode(&span, 2).expect("decodes");
        assert_eq!(stats.slice_state, SliceState::Failed);
        assert_eq!(stats.failure, Some(WaveFailure { segment: Some(7), code: 42 }));
        assert_eq!(stats.barrier_wait_sum_ns, 3 << 10);
        assert_eq!(stats.blocks[1].generations, 5);

        put_u32(&mut span, FAIL_COMP_OFF, u32::MAX - NO_SEGMENT);
        let stats = WaveStats::decode(&span, 2).expect("decodes");
        assert_eq!(stats.failure, Some(WaveFailure { segment: None, code: 42 }));

        put_u32(&mut span, SLICE_STATE_OFF, 99);
        assert!(WaveStats::decode(&span, 2).is_err(), "an unknown slice state is refused");
    }
}

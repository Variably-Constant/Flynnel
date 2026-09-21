//! Byte layout of a wave span, and the codes written into it.
//!
//! The device side of this contract lives in `kernels/gpu_peer_wave.cu`
//! as `#define` offsets; the values here have to stay in lockstep with
//! that file. All cross-device wave state is addressed by byte offset
//! from the span base, because the two sides share pages rather than a
//! type.
//!
//! Kept in a module of its own for the same reason
//! [`crate::gpu_peer::layout`] is: a contract with one `.cu` file is
//! one thing, and ninety-seven constants loose among the operations
//! that read them make both harder to see. Re-exported at
//! [`super`], so every existing path still reaches them.

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

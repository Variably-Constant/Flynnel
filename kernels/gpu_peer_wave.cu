// Segmented-wave helpers for user ops composed with the gpu_peer poller.
//
// Composed after gpu_peer.cu and before the user source whenever user ops
// are registered, so a user op can call these, and these use the kernel's
// typedefs, gtimer() and FLYNNEL_USER_YIELD. Every name starts with flw_
// or FLW_.
//
// A wave lives in one resident span (GpuPeer::pin_bulk) that the user op
// receives as `block` and `count`. The span starts with the header laid
// out below, which src/gpu_peer/wave.rs mirrors; a test there reads these
// defines and compares them. Every counter is a VRAM word touched only by
// GPU threads, where atomicAdd and atomicMax are exact. atomicAdd returns
// the word's value from before the addition, so a push takes the position
// it returns and an arrival counts itself by adding one to it.
//
// Every other word that more than one thread touches is read and written
// through flw_ld / flw_st, which go through volatile pointers, and a thread
// fences its stores before an arrival or a sync that lets another thread
// read them. Without that a compiler may move a load of such a word ahead
// of a barrier's wait and read it before the block that writes it arrives.
//
// Synchronization contract, which every helper keeps:
//  - Every thread of a block reaches the same number of __syncthreads.
//    Segment work runs between syncs, and decisions are read after them.
//  - Blocks meet only at the barriers here. An arrival count is never
//    reset: a block waits for the round its own arrival fell in, so rounds
//    stay aligned from one slice to the next.
//  - A thread acts only on words its block's thread 0 wrote before the
//    last sync, so no thread reads a word another block is still writing.
//
// A wave op has this shape, and every thread of every block runs it:
//
//   flw_slice s;
//   u32 bad = flw_slice_begin(&s, block, count, team_rank, team_size);
//   if (bad != 0u) return bad;
//   while (s.running) {
//       for (u32 k = flw_first(&s); k < s.end; k += flw_stride(&s)) {
//           u32 id = flw_id(&s, k);
//           // run segment `id`; flw_push(&s, child) for each child it
//           // emits, flw_alloc(&s, bytes) for device-born state, and
//           // flw_fail(&s, id, code) when it fails
//       }
//       flw_generation_end(&s);
//   }
//   return flw_slice_end(&s);
//
// A wave may keep a reorder buffer: a record per segment id below
// FLW_ROB_CAPACITY_OFF and a row per root. The thread running a segment
// links each child with flw_push_child, in ordinal order, then calls
// flw_rob_expanded; a segment that refuses calls flw_rob_refuse, and one
// whose value has retired calls flw_rob_retired. Block 0 commits every row
// in pre-order over (parent, ordinal), up to its first segment still
// pending or refused.

// ----------------------------------------------------------------- header

#define FLW_MAGIC                 0x56574C46u   // "FLWV"
#define FLW_VERSION               1u

#define FLW_MAGIC_OFF             0x00u
#define FLW_VERSION_OFF           0x04u
#define FLW_WIDTH_OFF             0x08u   // blocks in the wave: the lane's team size
#define FLW_MODE_OFF              0x0Cu   // FLW_MODE_*
#define FLW_REBALANCE_OFF         0x10u   // partition rebalance interval N; 0 = never
#define FLW_RESUME_MODE_OFF       0x14u   // FLW_RESUME_*
#define FLW_ARRIVE_OFF            0x18u   // barrier arrivals, never reset
#define FLW_WAIT_MAX_OFF          0x1Cu   // longest barrier wait, ns
#define FLW_WAIT_SUM_KNS_OFF      0x20u   // summed barrier waits, units of 1024 ns
#define FLW_TIMEOUTS_OFF          0x24u   // barriers left on their deadline
#define FLW_SKEW_MAX_OFF          0x28u   // longest first-barrier wait of a slice, ns
#define FLW_DONE_OFF              0x2Cu   // slice-end arrivals, never reset
#define FLW_FAIL_COMP_OFF         0x30u   // complement of the lowest failing segment; 0 = none
#define FLW_FAIL_CODE_OFF         0x34u   // largest failure code reported
#define FLW_PUSH_OFF              0x38u   // global frontier: ids pushed
#define FLW_START_OFF             0x3Cu   // global frontier: current generation start
#define FLW_END_OFF               0x40u   // global frontier: current generation end
#define FLW_ID_CAPACITY_OFF       0x44u   // ids the id array holds
#define FLW_ARENA_BUMP_OFF        0x48u   // arena bytes reserved
#define FLW_ARENA_CAPACITY_OFF    0x4Cu   // arena bytes available
#define FLW_STOP_OFF              0x50u   // block 0's stop decision for the coming barrier
#define FLW_SLICE_STATE_OFF       0x54u   // FLW_SLICE_* of the last slice
#define FLW_BUDGET_NS_OFF         0x58u   // u64: slice budget from the watchdog; 0 = none
#define FLW_LONGEST_GEN_OFF       0x60u   // longest generation, ns
#define FLW_YIELDS_OFF            0x64u   // slices ended by a device yield
#define FLW_SLICES_OFF            0x68u   // slices run
#define FLW_GENERATIONS_OFF       0x6Cu   // generations completed by block 0
#define FLW_IDS_OFF_OFF           0x70u   // byte offset of the id array
#define FLW_STAGING_OFF_OFF       0x74u   // byte offset of the rebalance staging array
#define FLW_ARENA_OFF_OFF         0x78u   // byte offset of the arena
#define FLW_TABLE_OFF_OFF         0x7Cu   // byte offset of the per-block table
#define FLW_BARRIER_DEADLINE_OFF  0x80u   // ns a block waits at a barrier; must be nonzero
#define FLW_IMBALANCE_OFF         0x84u   // largest block's share over the mean, per mille (see wave.rs)
#define FLW_DONE_DEADLINE_NS_OFF  0x88u   // u64: ns block 0 waits at slice end; 0 = no limit
#define FLW_REBALANCES_OFF        0x90u   // rebalances run
#define FLW_MOVED_OFF             0x94u   // pending ids moved through staging by rebalances
#define FLW_ELAPSED_NS_OFF        0x98u   // u64: slice time summed on block 0, ns
#define FLW_REBALANCE_NS_OFF      0xA0u   // u64: block 0's time in rebalances that moved ids, ns
#define FLW_ROB_OFF_OFF           0xA8u   // byte offset of the ROB table; 0 = no ROB
#define FLW_ROB_CAPACITY_OFF      0xACu   // ROB records; every segment id is below it
#define FLW_ROWS_OFF_OFF          0xB0u   // byte offset of the row table
#define FLW_ROW_COUNT_OFF         0xB4u   // rows, one per root
#define FLW_ROB_COMMITTED_OFF     0xB8u   // segments committed over every row
#define FLW_HEADER_BYTES          0x100u

// Per-block table, one FLW_TABLE_STRIDE entry per block.
#define FLW_TABLE_STRIDE          0x40u
#define FLW_T_PUSH                0x00u   // ids the block pushed: into its region, or to the global array
#define FLW_T_START               0x04u   // current generation start
#define FLW_T_END                 0x08u   // current generation end
#define FLW_T_DECISION            0x0Cu   // 1 while the block's threads run another generation
#define FLW_T_EMPTY               0x10u   // 1 when the block's current range is empty
#define FLW_T_PENDING             0x14u   // rebalance: pending ids
#define FLW_T_PREFIX              0x18u   // rebalance: staging offset; global: pushes seen at the last barrier
#define FLW_T_GENERATION          0x1Cu   // generations the block completed
#define FLW_T_TOTAL               0x20u   // rebalance: pending ids over every block
#define FLW_T_DEAL_LO             0x24u   // rebalance: staging start dealt to the block
#define FLW_T_DEAL_HI             0x28u   // rebalance: staging end dealt to the block

// ROB record, one FLW_ROB_STRIDE entry per segment id.
#define FLW_ROB_STRIDE            0x28u
#define FLW_R_PARENT              0x00u   // parent id; FLW_ROB_NONE for a root or an id never linked
#define FLW_R_ORDINAL             0x04u   // index among the parent's children
#define FLW_R_ROW                 0x08u   // the row the segment belongs to
#define FLW_R_FIRST_CHILD         0x0Cu   // child with ordinal 0; FLW_ROB_NONE for none
#define FLW_R_NEXT_SIBLING        0x10u   // the parent's next child; FLW_ROB_NONE after the last
#define FLW_R_LAST_CHILD          0x14u   // most recently linked child
#define FLW_R_EXPANDED            0x18u   // 1 once the segment's own step is done and its children linked
#define FLW_R_REFUSED             0x1Cu   // 1 once the segment refused
#define FLW_R_RETIRED             0x20u   // 1 once the segment's value retired
#define FLW_R_CHILDREN            0x24u   // children linked

// Row, one FLW_ROW_STRIDE entry per root.
#define FLW_ROW_STRIDE            0x18u
#define FLW_W_ROOT                0x00u   // root id
#define FLW_W_CURSOR              0x04u   // first segment in pre-order not committed; FLW_ROB_NONE once complete
#define FLW_W_FLAGS               0x08u   // FLW_ROW_* bits, written by block 0's walk only
#define FLW_W_REFUSED_COMP        0x0Cu   // complement of the lowest id that refused; 0 = none
#define FLW_W_ROOT_RETIRED        0x10u   // 1 once the root's value retired
#define FLW_W_COMMITTED           0x14u   // segments of the row committed
#define FLW_ROW_COMPLETE          1u
#define FLW_ROW_REFUSED           2u
#define FLW_ROB_NONE              0xFFFFFFFFu

#define FLW_MODE_GLOBAL           0u
#define FLW_MODE_PARTITION        1u
#define FLW_RESUME_DEVICE         0u
#define FLW_RESUME_HOST           1u

#define FLW_SLICE_RUNNING         0u
#define FLW_SLICE_YIELDED         1u
#define FLW_SLICE_CONTINUE        2u
#define FLW_SLICE_FINISHED        3u
#define FLW_SLICE_FAILED          4u

// Failure codes. flw_fail keeps the largest code reported, so codes below
// 0xF000 are the op's own, and an FLW_FAIL_* code reported beside them wins.
// FLW_FAIL_ROB marks a broken ROB invariant (an id outside the ROB, or a
// cycle in its links), never a program's verdict: a program refuses through
// flw_rob_refuse.
#define FLW_NO_SEGMENT            0xFFFFFFFEu
#define FLW_FAIL_IDS              0xF001u
#define FLW_FAIL_ARENA            0xF002u
#define FLW_FAIL_BARRIER          0xF003u
#define FLW_FAIL_DONE             0xF004u
#define FLW_FAIL_BUDGET           0xF005u
#define FLW_FAIL_ROB              0xF006u

#define FLW_STATUS_FAILED         2u
#define FLW_STATUS_BAD_SPAN       3u

// ------------------------------------------------------------------ state

// One thread's view of the slice it is running.
typedef struct {
    unsigned char* base;
    u32 rank;
    u32 width;
    u32 mode;
    u32 rebalance;
    u32 coupled;     // 1 when blocks meet at barriers: global, or partition with N
    u32 cap;         // ids per region: the whole array, or one block's share
    u32 start;
    u32 end;
    u32 running;
    u32 local_gen;   // generations this thread ran in the slice
    u64 slice_t0;    // thread 0 only
    u64 gen_t0;      // thread 0 only
} flw_slice;

// Load and store of a word more than one thread touches.
__device__ __forceinline__ u32 flw_ld(u32* p)
{
    return *(volatile u32*)p;
}

__device__ __forceinline__ void flw_st(u32* p, u32 v)
{
    *(volatile u32*)p = v;
}

__device__ __forceinline__ u32* flw_u32(unsigned char* base, u32 off)
{
    return (u32*)(base + off);
}

__device__ __forceinline__ u32 flw_get(unsigned char* base, u32 off)
{
    return flw_ld(flw_u32(base, off));
}

__device__ __forceinline__ void flw_set(unsigned char* base, u32 off, u32 v)
{
    flw_st(flw_u32(base, off), v);
}

__device__ __forceinline__ u64 flw_get64(unsigned char* base, u32 off)
{
    return *(volatile u64*)(base + off);
}

__device__ __forceinline__ void flw_set64(unsigned char* base, u32 off, u64 v)
{
    *(volatile u64*)(base + off) = v;
}

__device__ __forceinline__ u32* flw_table(flw_slice* s, u32 blk, u32 field)
{
    return (u32*)(s->base + flw_get(s->base, FLW_TABLE_OFF_OFF) + blk * FLW_TABLE_STRIDE + field);
}

__device__ __forceinline__ u32 flw_tget(flw_slice* s, u32 blk, u32 field)
{
    return flw_ld(flw_table(s, blk, field));
}

__device__ __forceinline__ void flw_tset(flw_slice* s, u32 blk, u32 field, u32 v)
{
    flw_st(flw_table(s, blk, field), v);
}

__device__ __forceinline__ u32* flw_ids(flw_slice* s)
{
    return (u32*)(s->base + flw_get(s->base, FLW_IDS_OFF_OFF));
}

__device__ __forceinline__ u32* flw_staging(flw_slice* s)
{
    return (u32*)(s->base + flw_get(s->base, FLW_STAGING_OFF_OFF));
}

__device__ __forceinline__ u32 flw_min(u32 a, u32 b)
{
    return a < b ? a : b;
}

__device__ __forceinline__ u32 flw_sat(u64 v)
{
    return v > 0xFFFFFFFFull ? 0xFFFFFFFFu : (u32)v;
}

// Where a block's ids start in the id array: 0 for a global frontier, the
// block's own region for a partition.
__device__ __forceinline__ u32 flw_region(flw_slice* s, u32 blk)
{
    return s->mode == FLW_MODE_GLOBAL ? 0u : blk * s->cap;
}

// ---------------------------------------------------------- segment calls

// Records a failure. The lowest failing segment id is kept, as its
// complement under atomicMax, together with the largest code reported. A
// failure that belongs to no segment passes FLW_NO_SEGMENT.
__device__ __forceinline__ void flw_fail(flw_slice* s, u32 segment, u32 code)
{
    atomicMax(flw_u32(s->base, FLW_FAIL_COMP_OFF), 0xFFFFFFFFu - segment);
    atomicMax(flw_u32(s->base, FLW_FAIL_CODE_OFF), code);
}

// Appends a child segment id to the next generation and returns its
// position, or 0xFFFFFFFF with the wave failed when the ids are full. A
// global frontier appends to the wave's array; a partition appends to the
// calling block's region.
__device__ __forceinline__ u32 flw_push(flw_slice* s, u32 id)
{
    u32* counter = s->mode == FLW_MODE_GLOBAL
        ? flw_u32(s->base, FLW_PUSH_OFF)
        : flw_table(s, s->rank, FLW_T_PUSH);
    u32 p = atomicAdd(counter, 1u);
    if (s->mode == FLW_MODE_GLOBAL) {
        // Children counted per block, which block 0 turns into the
        // frontier's per-generation imbalance at each barrier.
        atomicAdd(flw_table(s, s->rank, FLW_T_PUSH), 1u);
    }
    if (p >= s->cap) {
        flw_fail(s, id, FLW_FAIL_IDS);
        return 0xFFFFFFFFu;
    }
    flw_st(flw_ids(s) + flw_region(s, s->rank) + p, id);
    return p;
}

// Reserves `bytes` of the wave's arena, rounded up to a multiple of eight,
// and returns its byte offset from the span start, or 0xFFFFFFFF with the
// wave failed when the arena is exhausted. The arena is released only with
// the span.
__device__ __forceinline__ u32 flw_alloc(flw_slice* s, u32 bytes)
{
    u32 size = (bytes + 7u) & ~7u;
    u32 at = atomicAdd(flw_u32(s->base, FLW_ARENA_BUMP_OFF), size);
    u32 top = at + size;
    if (size < bytes || top < at || top > flw_get(s->base, FLW_ARENA_CAPACITY_OFF)) {
        flw_fail(s, FLW_NO_SEGMENT, FLW_FAIL_ARENA);
        return 0xFFFFFFFFu;
    }
    return flw_get(s->base, FLW_ARENA_OFF_OFF) + at;
}

// The segment id at position `k` of the current generation.
__device__ __forceinline__ u32 flw_id(flw_slice* s, u32 k)
{
    return flw_ld(flw_ids(s) + flw_region(s, s->rank) + k);
}

// The first position this thread takes in the current generation. A global
// frontier is divided across every thread of the team, a partition across
// the threads of one block.
__device__ __forceinline__ u32 flw_first(flw_slice* s)
{
    return s->start + (s->mode == FLW_MODE_GLOBAL
        ? s->rank * blockDim.x + threadIdx.x
        : threadIdx.x);
}

// The step from one position this thread takes to its next.
__device__ __forceinline__ u32 flw_stride(flw_slice* s)
{
    return s->mode == FLW_MODE_GLOBAL ? s->width * blockDim.x : blockDim.x;
}

// -------------------------------------------------------------------- ROB

// Records the wave's ROB holds, or 0 when it keeps none.
__device__ __forceinline__ u32 flw_rob_capacity(flw_slice* s)
{
    return flw_get(s->base, FLW_ROB_OFF_OFF) == 0u ? 0u : flw_get(s->base, FLW_ROB_CAPACITY_OFF);
}

__device__ __forceinline__ u32* flw_rec(flw_slice* s, u32 id, u32 field)
{
    return (u32*)(s->base + flw_get(s->base, FLW_ROB_OFF_OFF) + (u64)id * FLW_ROB_STRIDE + field);
}

__device__ __forceinline__ u32* flw_row(flw_slice* s, u32 row, u32 field)
{
    return (u32*)(s->base + flw_get(s->base, FLW_ROWS_OFF_OFF) + (u64)row * FLW_ROW_STRIDE + field);
}

// Links `id` into the ROB as the next child of `parent`, in the parent's
// row, and appends it to the next generation with flw_push. The one thread
// running `parent` links all of its children, in ordinal order, before
// flw_rob_expanded(parent). Returns flw_push's position, or 0xFFFFFFFF with
// the wave failed when either id lies outside the ROB.
__device__ __forceinline__ u32 flw_push_child(flw_slice* s, u32 parent, u32 id)
{
    u32 cap = flw_rob_capacity(s);
    if (parent >= cap || id >= cap || parent == id) {
        flw_fail(s, FLW_NO_SEGMENT, FLW_FAIL_ROB);
        return 0xFFFFFFFFu;
    }
    u32 ordinal = flw_ld(flw_rec(s, parent, FLW_R_CHILDREN));
    flw_st(flw_rec(s, id, FLW_R_PARENT), parent);
    flw_st(flw_rec(s, id, FLW_R_ORDINAL), ordinal);
    flw_st(flw_rec(s, id, FLW_R_ROW), flw_ld(flw_rec(s, parent, FLW_R_ROW)));
    if (ordinal == 0u) {
        flw_st(flw_rec(s, parent, FLW_R_FIRST_CHILD), id);
    } else {
        flw_st(flw_rec(s, flw_ld(flw_rec(s, parent, FLW_R_LAST_CHILD)), FLW_R_NEXT_SIBLING), id);
    }
    flw_st(flw_rec(s, parent, FLW_R_LAST_CHILD), id);
    flw_st(flw_rec(s, parent, FLW_R_CHILDREN), ordinal + 1u);
    return flw_push(s, id);
}

// The row of a ROB record, or FLW_ROB_NONE with the wave failed when `id`
// lies outside the ROB or its record names no row.
__device__ __forceinline__ u32 flw_rob_row(flw_slice* s, u32 id)
{
    if (id >= flw_rob_capacity(s)) {
        flw_fail(s, FLW_NO_SEGMENT, FLW_FAIL_ROB);
        return FLW_ROB_NONE;
    }
    u32 row = flw_ld(flw_rec(s, id, FLW_R_ROW));
    if (row >= flw_get(s->base, FLW_ROW_COUNT_OFF)) {
        flw_fail(s, id, FLW_FAIL_ROB);
        return FLW_ROB_NONE;
    }
    return row;
}

// Marks `id`'s own step done, with every child it has linked.
__device__ __forceinline__ void flw_rob_expanded(flw_slice* s, u32 id)
{
    if (flw_rob_row(s, id) == FLW_ROB_NONE) return;
    flw_st(flw_rec(s, id, FLW_R_EXPANDED), 1u);
}

// Marks `id` refused: its row commits nothing past it in pre-order, and the
// row keeps the lowest id that refused.
__device__ __forceinline__ void flw_rob_refuse(flw_slice* s, u32 id)
{
    u32 row = flw_rob_row(s, id);
    if (row == FLW_ROB_NONE) return;
    flw_st(flw_rec(s, id, FLW_R_REFUSED), 1u);
    atomicMax(flw_row(s, row, FLW_W_REFUSED_COMP), 0xFFFFFFFFu - id);
}

// Marks `id`'s value retired. For a row's root, the row's value is ready.
__device__ __forceinline__ void flw_rob_retired(flw_slice* s, u32 id)
{
    u32 row = flw_rob_row(s, id);
    if (row == FLW_ROB_NONE) return;
    flw_st(flw_rec(s, id, FLW_R_RETIRED), 1u);
    if (flw_ld(flw_row(s, row, FLW_W_ROOT)) == id) {
        flw_st(flw_row(s, row, FLW_W_ROOT_RETIRED), 1u);
    }
}

// On block 0's thread 0, while no other block writes the ROB: commits each
// open row's expanded segments in pre-order, from its cursor up to the first
// segment still pending or refused. A row whose pre-order runs out is
// complete; one stopped at a refused segment is refused there. A walk that
// takes more steps than a tree of the ROB's size allows has met a cycle or
// a link outside the ROB, and fails the wave.
__device__ __forceinline__ void flw_rob_walk(flw_slice* s)
{
    u32 cap = flw_rob_capacity(s);
    if (cap == 0u) return;
    u32 rows = flw_get(s->base, FLW_ROW_COUNT_OFF);
    u64 steps = 0ull;
    u64 limit = 2ull * (u64)cap + (u64)rows;
    u32 all = 0u;
    for (u32 r = 0u; r < rows; r++) {
        u32 flags = flw_ld(flw_row(s, r, FLW_W_FLAGS));
        if (flags != 0u) continue;
        u32 node = flw_ld(flw_row(s, r, FLW_W_CURSOR));
        u32 committed = 0u;
        while (node != FLW_ROB_NONE) {
            steps++;
            if (node >= cap || steps > limit) {
                flw_fail(s, FLW_NO_SEGMENT, FLW_FAIL_ROB);
                return;
            }
            if (flw_ld(flw_rec(s, node, FLW_R_REFUSED)) != 0u) {
                flags = FLW_ROW_REFUSED;
                break;
            }
            if (flw_ld(flw_rec(s, node, FLW_R_EXPANDED)) == 0u) break;
            committed++;
            u32 next = flw_ld(flw_rec(s, node, FLW_R_FIRST_CHILD));
            u32 up = node;
            while (next == FLW_ROB_NONE && up != FLW_ROB_NONE) {
                steps++;
                if (up >= cap || steps > limit) {
                    flw_fail(s, FLW_NO_SEGMENT, FLW_FAIL_ROB);
                    return;
                }
                next = flw_ld(flw_rec(s, up, FLW_R_NEXT_SIBLING));
                if (next == FLW_ROB_NONE) up = flw_ld(flw_rec(s, up, FLW_R_PARENT));
            }
            node = next;
        }
        if (node == FLW_ROB_NONE) flags = FLW_ROW_COMPLETE;
        flw_st(flw_row(s, r, FLW_W_CURSOR), node);
        flw_st(flw_row(s, r, FLW_W_COMMITTED), flw_ld(flw_row(s, r, FLW_W_COMMITTED)) + committed);
        flw_st(flw_row(s, r, FLW_W_FLAGS), flags);
        all += committed;
    }
    flw_set(s->base, FLW_ROB_COMMITTED_OFF, flw_get(s->base, FLW_ROB_COMMITTED_OFF) + all);
}

// ------------------------------------------------------------ coordination

// Ids pushed for block `blk`, bounded by the region's capacity.
__device__ __forceinline__ u32 flw_pushed(flw_slice* s, u32 blk)
{
    u32 pushed = s->mode == FLW_MODE_GLOBAL
        ? atomicAdd(flw_u32(s->base, FLW_PUSH_OFF), 0u)
        : atomicAdd(flw_table(s, blk, FLW_T_PUSH), 0u);
    return flw_min(pushed, s->cap);
}

// On block 0's thread 0, before it arrives at a barrier: waits until every
// other block has arrived in the coming round, when their threads are synced
// and their stores fenced, or until the barrier deadline passes. Returns 1
// when every other block has arrived. The other blocks' deadlines cover
// what block 0 does before it arrives.
__device__ __forceinline__ u32 flw_await_others(flw_slice* s)
{
    unsigned char* b = s->base;
    __threadfence_system();
    u64 t0 = gtimer();
    u64 deadline = (u64)flw_get(b, FLW_BARRIER_DEADLINE_OFF);
    u32 seen = atomicAdd(flw_u32(b, FLW_ARRIVE_OFF), 0u);
    u32 last = (seen / s->width + 1u) * s->width - 1u;
    while (atomicAdd(flw_u32(b, FLW_ARRIVE_OFF), 0u) < last && gtimer() - t0 <= deadline) {
    }
    return atomicAdd(flw_u32(b, FLW_ARRIVE_OFF), 0u) == last ? 1u : 0u;
}

// On a block's thread 0: arrive at a barrier and wait until every block of
// the wave has arrived in the same round, or the deadline passes. Records
// the wait. Returns 1 when the round completed; on the deadline the wave is
// marked failed and 0 is returned. The block's stores are fenced first, so
// a block that reads them after the round reads what this one wrote.
__device__ __forceinline__ u32 flw_barrier(flw_slice* s, u32 first_of_slice)
{
    unsigned char* b = s->base;
    __threadfence_system();
    u64 t0 = gtimer();
    u64 deadline = (u64)flw_get(b, FLW_BARRIER_DEADLINE_OFF);
    u32 mine = atomicAdd(flw_u32(b, FLW_ARRIVE_OFF), 1u) + 1u;
    u32 goal = ((mine + s->width - 1u) / s->width) * s->width;
    u32 whole = 1u;
    while (atomicAdd(flw_u32(b, FLW_ARRIVE_OFF), 0u) < goal) {
        if (gtimer() - t0 > deadline) {
            whole = 0u;
            break;
        }
    }
    u32 waited = flw_sat(gtimer() - t0);
    atomicMax(flw_u32(b, FLW_WAIT_MAX_OFF), waited);
    atomicAdd(flw_u32(b, FLW_WAIT_SUM_KNS_OFF), waited >> 10);
    if (first_of_slice != 0u) {
        atomicMax(flw_u32(b, FLW_SKEW_MAX_OFF), waited);
    }
    if (whole == 0u) {
        atomicAdd(flw_u32(b, FLW_TIMEOUTS_OFF), 1u);
        flw_fail(s, FLW_NO_SEGMENT, FLW_FAIL_BARRIER);
    }
    return whole;
}

// On thread 0: whether the slice must end before `ahead` more generations,
// from the time the slice has run and the longest generation measured,
// against the watchdog budget. A zero budget never ends a slice.
__device__ __forceinline__ u32 flw_should_stop(flw_slice* s, u64 now, u32 ahead)
{
    u64 budget = flw_get64(s->base, FLW_BUDGET_NS_OFF);
    if (budget == 0ull) return 0u;
    u64 longest = (u64)flw_get(s->base, FLW_LONGEST_GEN_OFF);
    return (now - s->slice_t0) + longest * (u64)ahead >= budget ? 1u : 0u;
}

// Starts a slice. Every thread of every block calls it once, before its
// first generation. Returns 0, or a status the op returns at once, without
// flw_slice_end, when `base` is not a wave span for this team.
__device__ __forceinline__ u32 flw_slice_begin(flw_slice* s, unsigned char* base, u32 count,
                                               u32 team_rank, u32 team_size)
{
    s->base = base;
    s->rank = team_rank;
    s->width = team_size;
    s->running = 0u;
    s->local_gen = 0u;
    s->start = 0u;
    s->end = 0u;
    if (base == (unsigned char*)0 || count < FLW_HEADER_BYTES
        || flw_get(base, FLW_MAGIC_OFF) != FLW_MAGIC
        || flw_get(base, FLW_VERSION_OFF) != FLW_VERSION
        || flw_get(base, FLW_WIDTH_OFF) != team_size) {
        return FLW_STATUS_BAD_SPAN;
    }
    s->mode = flw_get(base, FLW_MODE_OFF);
    s->rebalance = flw_get(base, FLW_REBALANCE_OFF);
    s->coupled = (s->mode == FLW_MODE_GLOBAL || s->rebalance != 0u) ? 1u : 0u;
    s->cap = s->mode == FLW_MODE_GLOBAL
        ? flw_get(base, FLW_ID_CAPACITY_OFF)
        : flw_get(base, FLW_ID_CAPACITY_OFF) / team_size;

    if (threadIdx.x == 0) {
        s->slice_t0 = gtimer();
        u32 start = s->mode == FLW_MODE_GLOBAL
            ? flw_get(base, FLW_START_OFF)
            : flw_tget(s, team_rank, FLW_T_START);
        u32 end = s->mode == FLW_MODE_GLOBAL
            ? flw_get(base, FLW_END_OFF)
            : flw_tget(s, team_rank, FLW_T_END);
        flw_tset(s, team_rank, FLW_T_START, start);
        flw_tset(s, team_rank, FLW_T_END, end);
        flw_tset(s, team_rank, FLW_T_EMPTY, start < end ? 0u : 1u);

        // A slice that cannot fit the generations before its first
        // decision point would yield forever without progress.
        u32 ahead = s->mode == FLW_MODE_PARTITION && s->rebalance != 0u ? s->rebalance : 1u;
        u64 budget = flw_get64(base, FLW_BUDGET_NS_OFF);
        if (budget != 0ull
            && (u64)flw_get(base, FLW_LONGEST_GEN_OFF) * (u64)ahead >= budget) {
            flw_fail(s, FLW_NO_SEGMENT, FLW_FAIL_BUDGET);
        }

        u32 go;
        if (s->coupled != 0u) {
            u32 whole = flw_barrier(s, 1u);
            u32 any = 0u;
            for (u32 k = 0u; k < team_size; k++) {
                if (flw_tget(s, k, FLW_T_EMPTY) == 0u) any = 1u;
            }
            go = (whole != 0u && any != 0u && flw_get(base, FLW_FAIL_COMP_OFF) == 0u) ? 1u : 0u;
        } else {
            go = (start < end && flw_get(base, FLW_FAIL_COMP_OFF) == 0u) ? 1u : 0u;
        }
        flw_tset(s, team_rank, FLW_T_DECISION, go);
        s->gen_t0 = gtimer();
        __threadfence_system();
    }
    __syncthreads();
    s->start = flw_tget(s, team_rank, FLW_T_START);
    s->end = flw_tget(s, team_rank, FLW_T_END);
    s->running = flw_tget(s, team_rank, FLW_T_DECISION);
    return 0u;
}

// A partition's rebalance, run by every thread of every block at the same
// generation. Every block's pending ids are gathered to staging in block
// order, then dealt out evenly, one contiguous run per block starting at
// the beginning of its region, so a region's space is reclaimed. Blocks
// meet at two barriers: before the gather, and between gather and deal.
__device__ __forceinline__ void flw_rebalance(flw_slice* s)
{
    unsigned char* b = s->base;
    u32 me = s->rank;
    u32 w = s->width;
    u64 rebalance_t0 = 0ull;

    if (threadIdx.x == 0) {
        u64 now = gtimer();
        rebalance_t0 = now;
        atomicMax(flw_u32(b, FLW_LONGEST_GEN_OFF), flw_sat(now - s->gen_t0));
        u32 pend_start = s->end;
        u32 pend_end = flw_pushed(s, me);
        flw_tset(s, me, FLW_T_START, pend_start);
        flw_tset(s, me, FLW_T_END, pend_end);
        flw_tset(s, me, FLW_T_PENDING, pend_end > pend_start ? pend_end - pend_start : 0u);
        if (me == 0u) {
            flw_set(b, FLW_STOP_OFF, flw_should_stop(s, now, s->rebalance));
            if (flw_await_others(s) != 0u) flw_rob_walk(s);
        }
        u32 whole = flw_barrier(s, 0u);
        u32 prefix = 0u;
        u32 total = 0u;
        u32 largest = 0u;
        for (u32 k = 0u; k < w; k++) {
            u32 len = flw_tget(s, k, FLW_T_PENDING);
            if (k < me) prefix += len;
            total += len;
            if (len > largest) largest = len;
        }
        flw_tset(s, me, FLW_T_PREFIX, prefix);
        flw_tset(s, me, FLW_T_TOTAL, total);
        flw_tset(s, me, FLW_T_DECISION, whole);
        if (me == 0u && total > 0u) {
            atomicMax(flw_u32(b, FLW_IMBALANCE_OFF),
                      flw_sat((u64)largest * 1000ull * (u64)w / (u64)total));
            atomicAdd(flw_u32(b, FLW_REBALANCES_OFF), 1u);
            atomicAdd(flw_u32(b, FLW_MOVED_OFF), total);
        }
        __threadfence_system();
    }
    __syncthreads();

    {
        u32 src0 = flw_tget(s, me, FLW_T_START);
        u32 src1 = flw_tget(s, me, FLW_T_END);
        u32 prefix = flw_tget(s, me, FLW_T_PREFIX);
        u32* ids = flw_ids(s);
        u32* staging = flw_staging(s);
        u32 region = flw_region(s, me);
        for (u32 k = src0 + threadIdx.x; k < src1; k += blockDim.x) {
            flw_st(staging + prefix + (k - src0), flw_ld(ids + region + k));
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        u32 whole = flw_barrier(s, 0u);
        u32 total = flw_tget(s, me, FLW_T_TOTAL);
        flw_tset(s, me, FLW_T_DEAL_LO, (u32)((u64)me * (u64)total / (u64)w));
        flw_tset(s, me, FLW_T_DEAL_HI, (u32)((u64)(me + 1u) * (u64)total / (u64)w));
        if (whole == 0u) flw_tset(s, me, FLW_T_DECISION, 0u);
        __threadfence_system();
    }
    __syncthreads();

    {
        u32 lo = flw_tget(s, me, FLW_T_DEAL_LO);
        u32 hi = flw_tget(s, me, FLW_T_DEAL_HI);
        u32* ids = flw_ids(s);
        u32* staging = flw_staging(s);
        u32 region = flw_region(s, me);
        for (u32 k = threadIdx.x; k < hi - lo; k += blockDim.x) {
            flw_st(ids + region + k, flw_ld(staging + lo + k));
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        u32 len = flw_tget(s, me, FLW_T_DEAL_HI) - flw_tget(s, me, FLW_T_DEAL_LO);
        flw_tset(s, me, FLW_T_PUSH, len);
        flw_tset(s, me, FLW_T_START, 0u);
        flw_tset(s, me, FLW_T_END, len);
        flw_tset(s, me, FLW_T_EMPTY, len == 0u ? 1u : 0u);
        flw_tset(s, me, FLW_T_GENERATION, flw_tget(s, me, FLW_T_GENERATION) + 1u);
        u32 go = (flw_tget(s, me, FLW_T_DECISION) != 0u
                  && flw_tget(s, me, FLW_T_TOTAL) > 0u
                  && flw_get(b, FLW_STOP_OFF) == 0u
                  && flw_get(b, FLW_FAIL_COMP_OFF) == 0u) ? 1u : 0u;
        flw_tset(s, me, FLW_T_DECISION, go);
        s->gen_t0 = gtimer();
        if (me == 0u) {
            flw_set(b, FLW_GENERATIONS_OFF, flw_get(b, FLW_GENERATIONS_OFF) + 1u);
            if (flw_tget(s, me, FLW_T_TOTAL) > 0u) {
                flw_set64(b, FLW_REBALANCE_NS_OFF,
                          flw_get64(b, FLW_REBALANCE_NS_OFF) + (s->gen_t0 - rebalance_t0));
            }
        }
        __threadfence_system();
    }
    __syncthreads();
}

// Ends a generation. Every thread of every block calls it once per
// generation, after its share of segments. Afterwards every thread holds
// the next generation's range and whether the slice runs it.
__device__ __forceinline__ void flw_generation_end(flw_slice* s)
{
    unsigned char* b = s->base;
    u32 me = s->rank;
    u32 next_gen = s->local_gen + 1u;
    u32 rebalancing = (s->mode == FLW_MODE_PARTITION && s->rebalance != 0u
                       && next_gen % s->rebalance == 0u) ? 1u : 0u;
    __threadfence_system();
    __syncthreads();

    if (rebalancing != 0u) {
        flw_rebalance(s);
    } else {
        if (threadIdx.x == 0) {
            u64 now = gtimer();
            atomicMax(flw_u32(b, FLW_LONGEST_GEN_OFF), flw_sat(now - s->gen_t0));
            u32 next_start = s->end;
            u32 next_end;
            u32 go;
            if (s->mode == FLW_MODE_GLOBAL) {
                if (me == 0u) {
                    flw_set(b, FLW_STOP_OFF, flw_should_stop(s, now, 1u));
                    if (flw_await_others(s) != 0u) {
                        // Every other block has arrived with its threads
                        // synced, so this generation's push counts are
                        // final: take each block's share of the children,
                        // walk the ROB, and publish the next range.
                        u32 largest = 0u;
                        u32 sum = 0u;
                        for (u32 k = 0u; k < s->width; k++) {
                            u32 pushed = atomicAdd(flw_table(s, k, FLW_T_PUSH), 0u);
                            u32 delta = pushed - flw_tget(s, k, FLW_T_PREFIX);
                            flw_tset(s, k, FLW_T_PREFIX, pushed);
                            sum += delta;
                            if (delta > largest) largest = delta;
                        }
                        if (sum > 0u) {
                            atomicMax(flw_u32(b, FLW_IMBALANCE_OFF),
                                      flw_sat((u64)largest * 1000ull * (u64)s->width / (u64)sum));
                        }
                        flw_rob_walk(s);
                        flw_set(b, FLW_START_OFF, next_start);
                        flw_set(b, FLW_END_OFF, flw_pushed(s, 0u));
                    } else {
                        flw_fail(s, FLW_NO_SEGMENT, FLW_FAIL_BARRIER);
                    }
                }
                u32 whole = flw_barrier(s, 0u);
                next_end = flw_get(b, FLW_END_OFF);
                go = (whole != 0u && next_start < next_end
                      && flw_get(b, FLW_STOP_OFF) == 0u
                      && flw_get(b, FLW_FAIL_COMP_OFF) == 0u) ? 1u : 0u;
                if (me == 0u) {
                    flw_set(b, FLW_GENERATIONS_OFF, flw_get(b, FLW_GENERATIONS_OFF) + 1u);
                }
            } else {
                next_end = flw_pushed(s, me);
                if (s->coupled != 0u) {
                    go = 1u;
                } else {
                    go = (next_start < next_end
                          && flw_should_stop(s, now, 1u) == 0u
                          && flw_get(b, FLW_FAIL_COMP_OFF) == 0u) ? 1u : 0u;
                }
                flw_tset(s, me, FLW_T_GENERATION, flw_tget(s, me, FLW_T_GENERATION) + 1u);
            }
            flw_tset(s, me, FLW_T_START, next_start);
            flw_tset(s, me, FLW_T_END, next_end);
            flw_tset(s, me, FLW_T_EMPTY, next_start < next_end ? 0u : 1u);
            flw_tset(s, me, FLW_T_DECISION, go);
            s->gen_t0 = gtimer();
            __threadfence_system();
        }
        __syncthreads();
    }
    s->start = flw_tget(s, me, FLW_T_START);
    s->end = flw_tget(s, me, FLW_T_END);
    s->running = flw_tget(s, me, FLW_T_DECISION);
    s->local_gen = next_gen;
}

// Ends a slice. Every thread of every block calls it once, after its loop.
// Block 0's thread 0 waits until every block has ended its slice, then
// decides what the op returns: 0 when the wave finished or the host is to
// continue it, FLYNNEL_USER_YIELD to run again on the device, and
// FLW_STATUS_FAILED when the wave failed. The kernel reads only that
// thread's return.
__device__ __forceinline__ u32 flw_slice_end(flw_slice* s)
{
    unsigned char* b = s->base;
    u32 result = 0u;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        u64 t0 = gtimer();
        u32 mine = atomicAdd(flw_u32(b, FLW_DONE_OFF), 1u) + 1u;
        if (s->rank == 0u) {
            u32 goal = ((mine + s->width - 1u) / s->width) * s->width;
            u64 deadline = flw_get64(b, FLW_DONE_DEADLINE_NS_OFF);
            while (atomicAdd(flw_u32(b, FLW_DONE_OFF), 0u) < goal) {
                if (deadline != 0ull && gtimer() - t0 > deadline) {
                    flw_fail(s, FLW_NO_SEGMENT, FLW_FAIL_DONE);
                    break;
                }
            }
            if (atomicAdd(flw_u32(b, FLW_DONE_OFF), 0u) >= goal) {
                flw_rob_walk(s);
            }
            u32 finished = 1u;
            for (u32 k = 0u; k < s->width; k++) {
                if (flw_tget(s, k, FLW_T_EMPTY) == 0u) finished = 0u;
            }
            flw_set(b, FLW_SLICES_OFF, flw_get(b, FLW_SLICES_OFF) + 1u);
            flw_set64(b, FLW_ELAPSED_NS_OFF,
                      flw_get64(b, FLW_ELAPSED_NS_OFF) + (gtimer() - s->slice_t0));
            flw_set(b, FLW_STOP_OFF, 0u);
            u32 state;
            if (flw_get(b, FLW_FAIL_COMP_OFF) != 0u) {
                state = FLW_SLICE_FAILED;
                result = FLW_STATUS_FAILED;
            } else if (finished != 0u) {
                state = FLW_SLICE_FINISHED;
            } else if (flw_get(b, FLW_RESUME_MODE_OFF) == FLW_RESUME_DEVICE) {
                state = FLW_SLICE_YIELDED;
                flw_set(b, FLW_YIELDS_OFF, flw_get(b, FLW_YIELDS_OFF) + 1u);
                result = FLYNNEL_USER_YIELD;
            } else {
                state = FLW_SLICE_CONTINUE;
            }
            flw_set(b, FLW_SLICE_STATE_OFF, state);
            __threadfence_system();
        }
    }
    __syncthreads();
    return result;
}

// ------------------------------------------------------------ calibration

// Flynnel's wave calibration op, which the poller dispatches for
// FLW_OP_CALIBRATE. It walks a binary segment tree to the depth in payload
// word 0. Its segments do nothing but push their children, so a slice's
// time is the helpers' own.
__device__ unsigned flw_calibration_op(unsigned char* block, unsigned count,
                                       volatile unsigned char* payload,
                                       unsigned team_rank, unsigned team_size)
{
    u32 limit = *(volatile u32*)payload;
    flw_slice s;
    u32 bad = flw_slice_begin(&s, block, count, team_rank, team_size);
    if (bad != 0u) return bad;
    while (s.running) {
        for (u32 k = flw_first(&s); k < s.end; k += flw_stride(&s)) {
            u32 id = flw_id(&s, k);
            u32 depth = id >> 24;
            if (depth < limit) {
                u32 serial = (id & 0xFFFFFFu) << 1;
                flw_push(&s, ((depth + 1u) << 24) | (serial & 0xFFFFFFu));
                flw_push(&s, ((depth + 1u) << 24) | ((serial | 1u) & 0xFFFFFFu));
            }
        }
        flw_generation_end(&s);
    }
    return flw_slice_end(&s);
}

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
#define FLW_IMBALANCE_OFF         0x84u   // largest pending run over the mean at a rebalance, per mille
#define FLW_DONE_DEADLINE_NS_OFF  0x88u   // u64: ns block 0 waits at slice end; 0 = no limit
#define FLW_REBALANCES_OFF        0x90u   // rebalances run
#define FLW_HEADER_BYTES          0x100u

// Per-block table, one FLW_TABLE_STRIDE entry per block.
#define FLW_TABLE_STRIDE          0x40u
#define FLW_T_PUSH                0x00u   // partition: ids pushed into the block's region
#define FLW_T_START               0x04u   // current generation start
#define FLW_T_END                 0x08u   // current generation end
#define FLW_T_DECISION            0x0Cu   // 1 while the block's threads run another generation
#define FLW_T_EMPTY               0x10u   // 1 when the block's current range is empty
#define FLW_T_PENDING             0x14u   // rebalance: pending ids
#define FLW_T_PREFIX              0x18u   // rebalance: staging offset of the block's run
#define FLW_T_GENERATION          0x1Cu   // generations the block completed
#define FLW_T_TOTAL               0x20u   // rebalance: pending ids over every block
#define FLW_T_DEAL_LO             0x24u   // rebalance: staging start dealt to the block
#define FLW_T_DEAL_HI             0x28u   // rebalance: staging end dealt to the block

#define FLW_MODE_GLOBAL           0u
#define FLW_MODE_PARTITION        1u
#define FLW_RESUME_DEVICE         0u
#define FLW_RESUME_HOST           1u

#define FLW_SLICE_RUNNING         0u
#define FLW_SLICE_YIELDED         1u
#define FLW_SLICE_CONTINUE        2u
#define FLW_SLICE_FINISHED        3u
#define FLW_SLICE_FAILED          4u

#define FLW_NO_SEGMENT            0xFFFFFFFEu
#define FLW_FAIL_IDS              0xF001u
#define FLW_FAIL_ARENA            0xF002u
#define FLW_FAIL_BARRIER          0xF003u
#define FLW_FAIL_DONE             0xF004u
#define FLW_FAIL_BUDGET           0xF005u

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

// ------------------------------------------------------------ coordination

// Ids pushed for block `blk`, bounded by the region's capacity.
__device__ __forceinline__ u32 flw_pushed(flw_slice* s, u32 blk)
{
    u32 pushed = s->mode == FLW_MODE_GLOBAL
        ? atomicAdd(flw_u32(s->base, FLW_PUSH_OFF), 0u)
        : atomicAdd(flw_table(s, blk, FLW_T_PUSH), 0u);
    return flw_min(pushed, s->cap);
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
    u32 mine = atomicAdd(flw_u32(b, FLW_ARRIVE_OFF), 1u) + 1u;
    u32 goal = ((mine + s->width - 1u) / s->width) * s->width;
    u64 deadline = (u64)flw_get(b, FLW_BARRIER_DEADLINE_OFF);
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

    if (threadIdx.x == 0) {
        u64 now = gtimer();
        atomicMax(flw_u32(b, FLW_LONGEST_GEN_OFF), flw_sat(now - s->gen_t0));
        u32 pend_start = s->end;
        u32 pend_end = flw_pushed(s, me);
        flw_tset(s, me, FLW_T_START, pend_start);
        flw_tset(s, me, FLW_T_END, pend_end);
        flw_tset(s, me, FLW_T_PENDING, pend_end > pend_start ? pend_end - pend_start : 0u);
        if (me == 0u) {
            flw_set(b, FLW_STOP_OFF, flw_should_stop(s, now, s->rebalance));
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
        if (me == 0u) {
            flw_set(b, FLW_GENERATIONS_OFF, flw_get(b, FLW_GENERATIONS_OFF) + 1u);
        }
        s->gen_t0 = gtimer();
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
                }
                u32 whole = flw_barrier(s, 0u);
                next_end = flw_pushed(s, me);
                go = (whole != 0u && next_start < next_end
                      && flw_get(b, FLW_STOP_OFF) == 0u
                      && flw_get(b, FLW_FAIL_COMP_OFF) == 0u) ? 1u : 0u;
                if (me == 0u) {
                    flw_set(b, FLW_START_OFF, next_start);
                    flw_set(b, FLW_END_OFF, next_end);
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
            u32 finished = 1u;
            for (u32 k = 0u; k < s->width; k++) {
                if (flw_tget(s, k, FLW_T_EMPTY) == 0u) finished = 0u;
            }
            flw_set(b, FLW_SLICES_OFF, flw_get(b, FLW_SLICES_OFF) + 1u);
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

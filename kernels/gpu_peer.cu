// GPU-peer substrate kernels.
//
// Source of record for kernels/gpu_peer.ptx, which is the artifact the
// crate embeds (include_str!) and the driver JIT-compiles at runtime.
// Consumers of the crate never need the CUDA toolkit; regenerating the
// PTX after editing this file requires nvcc:
//
//   nvcc -ptx -arch=compute_75 gpu_peer.cu -o gpu_peer.ptx
//
// compute_75 (Turing) is the oldest target the supported CUDA driver
// line JIT-compiles; every newer GPU accepts the same PTX.
//
// Shared-layout contract: byte offsets below MUST stay in lockstep with
// src/gpu_peer/layout.rs (LAYOUT_* constants). The layout unit tests on
// the Rust side pin these values.
//
// Concurrency contract (mirrors the Rust side):
//   - Lane indices are single-writer: `head` is written ONLY by the CPU
//     producer, `tail` ONLY by the consuming GPU block. Release order
//     is store-payload, __threadfence_system(), store-index.
//   - NO system-scope atomics anywhere: measured unsafe over PCIe hosts
//     without native atomics (double-claims observed). Cross-device
//     arbitration that cannot be single-writer uses the Fischer timed
//     protocol with the host-calibrated Delta.
//   - DEVICE-scope atomics on the mapped region are exact among GPU
//     threads and are used only off the per-message hot path (the
//     kernel-exit counter).

// No standard-header includes: this source is ALSO compiled by NVRTC
// at runtime (header-less) when user opcodes are registered.
typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;

// ---------------------------------------------------------------- layout

#define HDR_MAGIC_OFF        0x000ULL
#define HDR_VERSION_OFF      0x008ULL
#define HDR_LANES_OFF        0x00CULL
#define HDR_SLOT_BYTES_OFF   0x010ULL
#define HDR_SLOTS_PER_OFF    0x014ULL
#define HDR_STOP_OFF         0x058ULL
#define HDR_EXITS_OFF        0x05CULL
#define HDR_ACTIVE_GEN_OFF   0x060ULL
#define HDR_CALIB_PING_OFF   0x064ULL
#define HDR_CALIB_PONG_OFF   0x068ULL
#define HDR_FISCHER_X_OFF    0x080ULL
#define HDR_FISCHER_CS_OFF   0x0C0ULL
#define HDR_FISCHER_VIOL_OFF 0x100ULL
#define HDR_FISCHER_ACQS_OFF 0x140ULL
#define HDR_FISCHER_STARTED_OFF 0x148ULL
#define HDR_FISCHER_GPU_CONT_OFF 0x14CULL
#define HDR_GTS_OFF          0x180ULL   // u64[400]
#define HDR_BYTES            0x1000ULL

#define LANE_STRIDE          0x100ULL
#define LANE_HEAD_OFF        0x00ULL
#define LANE_TAIL_OFF        0x40ULL
// Team barrier for lanes served by more than one block: arrivals for
// the slot in flight, and the generation that distinguishes one slot's
// barrier from the next. Their own cache lines, clear of head/tail.
#define LANE_TEAM_ARRIVE_OFF 0x80ULL
#define LANE_TEAM_GEN_OFF    0xC0ULL

#define SLOT_OP_OFF          0x00
#define SLOT_LEN_OFF         0x04
#define SLOT_SEQ_OFF         0x08
#define SLOT_STATUS_OFF      0x0C
#define SLOT_PAYLOAD_OFF     0x10

#define OP_NOP       0u
#define OP_ADD1_F32  1u
#define OP_SUM_U32   2u
// Resident-block ops: the slot payload starts with an 8-byte param
// header (u32 vram block index, u32 byte count); bulk data - when the
// op moves any - follows at payload+8. Compute ops touch ONLY the
// 8-byte params per task; the data stays in the VRAM pool.
#define OP_H2V         3u
#define OP_V2H         4u
#define OP_ADD1_F32_V  5u
#define OP_SUM_U32_V   6u

#define STATUS_DONE  1u
#define STATUS_ERR   2u

// User opcodes start here. When the host registers user CUDA source,
// this file is NVRTC-compiled together with it (-DFLYNNEL_USER_OPS)
// and the poller routes op >= OP_USER_BASE through the hook below.
// The hook is called by ALL threads of the lane's block
// (cooperative; __syncthreads is allowed inside). `block` is the
// resident VRAM block (0 when the descriptor named no valid block),
// `count` the param-declared byte count, `payload` the slot payload
// AFTER the 8-byte param header (user argument/result space).
// Return 0 for success, nonzero to mark the slot STATUS_ERR.
#define OP_USER_BASE 100u

#ifdef FLYNNEL_USER_OPS
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size);
#else
// Precompiled-PTX default: user ops are not linked; reject them.
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    (void)op; (void)block; (void)count; (void)payload;
    (void)team_rank; (void)team_size;
    return 1u;
}
#endif

__device__ __forceinline__ u64 gtimer() {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ u32 ld_vol(const volatile u32* p) { return *p; }
__device__ __forceinline__ void st_vol(volatile u32* p, u32 v) { *p = v; }

// ---------------------------------------------------------------- poller
//
// Bounded-quantum persistent consumer. One block per lane; thread 0
// owns the lane's `tail`, the whole block cooperates on payloads.
// Exits when: stop flag set, generation superseded, quantum elapsed,
// or idle for longer than idle_exit_ns. The host relaunches while a
// backlog exists; an exit is counted so the host can distinguish
// "parked" from "running" without stream queries.
extern "C" __global__ void flynnel_peer_poller(
    unsigned char* base,
    u32 lanes,
    u32 slot_bytes,
    u32 slots_per_lane,
    u64 quantum_ns,
    u64 idle_exit_ns,
    u32 my_gen,
    unsigned char* vram_base,      // 0 when no resident pool
    u32 vram_block_bytes,
    u32 vram_blocks,
    u32 blocks_per_lane)           // >1 gives each lane a block team
{
    if (blocks_per_lane == 0u) blocks_per_lane = 1u;
    // Consecutive blocks form one lane's team: rank 0 owns the
    // descriptor and the ring, every rank runs the user op.
    const u32 my_lane = blockIdx.x / blocks_per_lane;
    const u32 team_rank = blockIdx.x % blocks_per_lane;
    if (my_lane >= lanes) return;
    volatile u32* stop = (volatile u32*)(base + HDR_STOP_OFF);
    volatile u32* active_gen = (volatile u32*)(base + HDR_ACTIVE_GEN_OFF);
    volatile u32* head = (volatile u32*)(base + HDR_BYTES + my_lane * LANE_STRIDE + LANE_HEAD_OFF);
    volatile u32* tail = (volatile u32*)(base + HDR_BYTES + my_lane * LANE_STRIDE + LANE_TAIL_OFF);
    unsigned char* slab = base + HDR_BYTES + (u64)lanes * LANE_STRIDE
                        + (u64)my_lane * slots_per_lane * slot_bytes;

    volatile u32* team_arrive = (volatile u32*)(base + HDR_BYTES
                              + my_lane * LANE_STRIDE + LANE_TEAM_ARRIVE_OFF);
    volatile u32* team_gen = (volatile u32*)(base + HDR_BYTES
                           + my_lane * LANE_STRIDE + LANE_TEAM_GEN_OFF);
    __shared__ u32 s_run;      // 0 keep running, 1 exit
    __shared__ u32 s_slot;     // slot index to process this round, ~0u = none
    u32 my_gen_local = (blocks_per_lane > 1u) ? ld_vol(team_gen) : 0u;
    u64 t_start = gtimer();
    u64 t_last_work = t_start;

    for (;;) {
        if (threadIdx.x == 0) {
            s_run = 0;
            s_slot = ~0u;
            u64 now = gtimer();
            if (ld_vol(stop) != 0u || ld_vol(active_gen) != my_gen
                || now - t_start > quantum_ns
                || now - t_last_work > idle_exit_ns) {
                s_run = 1;
            } else {
                u32 h = ld_vol(head);
                u32 t = ld_vol(tail);
                if (h != t) {
                    s_slot = t % slots_per_lane;
                    t_last_work = now;
                }
            }
        }
        __syncthreads();
        if (s_run) break;
        if (s_slot == ~0u) continue;

        unsigned char* slot = slab + (u64)s_slot * slot_bytes;
        volatile u32* d_op = (volatile u32*)(slot + SLOT_OP_OFF);
        volatile u32* d_len = (volatile u32*)(slot + SLOT_LEN_OFF);
        volatile u32* d_status = (volatile u32*)(slot + SLOT_STATUS_OFF);
        u32 op = ld_vol(d_op);
        u32 len = ld_vol(d_len);
        u32 payload_max = slot_bytes - (u32)SLOT_PAYLOAD_OFF;
        if (len > payload_max) { op = ~0u; }

        // The builtin ops are single-block bodies; only user ops are
        // written to spread across a team, so ranks above 0 skip
        // straight to the barrier for everything else.
        if (blocks_per_lane > 1u && team_rank != 0u && op < OP_USER_BASE) {
            // fall through to the barrier with no work
        } else if (op == OP_ADD1_F32) {
            volatile float* p = (volatile float*)(slot + SLOT_PAYLOAD_OFF);
            u32 n = len / 4;
            for (u32 i = threadIdx.x; i < n; i += blockDim.x)
                p[i] = p[i] + 1.0f;
        } else if (op == OP_SUM_U32) {
            volatile u32* p = (volatile u32*)(slot + SLOT_PAYLOAD_OFF);
            u32 n = len / 4;
            __shared__ u64 partial[256];
            u64 acc = 0;
            for (u32 i = threadIdx.x; i < n; i += blockDim.x)
                acc += p[i];
            partial[threadIdx.x] = acc;
            __syncthreads();
            for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                volatile u64* out = (volatile u64*)(slot + SLOT_PAYLOAD_OFF);
                *out = partial[0];
            }
        }
        else if (op >= OP_H2V && op <= OP_SUM_U32_V) {
            // Resident-block ops: params at payload+0, data at +8.
            volatile u32* params = (volatile u32*)(slot + SLOT_PAYLOAD_OFF);
            u32 bidx = params[0];
            u32 count = params[1];
            if (vram_base == 0 || bidx >= vram_blocks || count > vram_block_bytes
                || (op == OP_H2V && count + 8u > payload_max)
                || (op == OP_V2H && count + 8u > payload_max)) {
                op = ~0u;
            } else {
                unsigned char* blk = vram_base + (u64)bidx * vram_block_bytes;
                if (op == OP_H2V) {
                    volatile u32* src = (volatile u32*)(slot + SLOT_PAYLOAD_OFF + 8);
                    u32* dst = (u32*)blk;
                    for (u32 i = threadIdx.x; i < count / 4; i += blockDim.x)
                        dst[i] = src[i];
                } else if (op == OP_V2H) {
                    const u32* src = (const u32*)blk;
                    volatile u32* dst = (volatile u32*)(slot + SLOT_PAYLOAD_OFF + 8);
                    for (u32 i = threadIdx.x; i < count / 4; i += blockDim.x)
                        dst[i] = src[i];
                } else if (op == OP_ADD1_F32_V) {
                    float* p = (float*)blk;
                    for (u32 i = threadIdx.x; i < count / 4; i += blockDim.x)
                        p[i] = p[i] + 1.0f;
                } else { // OP_SUM_U32_V
                    const u32* p = (const u32*)blk;
                    __shared__ u64 vpartial[256];
                    u64 acc = 0;
                    for (u32 i = threadIdx.x; i < count / 4; i += blockDim.x)
                        acc += p[i];
                    vpartial[threadIdx.x] = acc;
                    __syncthreads();
                    for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
                        if (threadIdx.x < s) vpartial[threadIdx.x] += vpartial[threadIdx.x + s];
                        __syncthreads();
                    }
                    if (threadIdx.x == 0) {
                        volatile u64* out = (volatile u64*)(slot + SLOT_PAYLOAD_OFF + 8);
                        *out = vpartial[0];
                    }
                }
            }
        }
        else if (op >= OP_USER_BASE && op != ~0u) {
            // User opcode: params name an OPTIONAL resident block
            // (index ~0u = none) plus a byte count; payload+8 is the
            // user's argument/result space.
            volatile u32* params = (volatile u32*)(slot + SLOT_PAYLOAD_OFF);
            u32 bidx = params[0];
            u32 count = params[1];
            unsigned char* blk = (unsigned char*)0;
            if (bidx != ~0u) {
                if (vram_base == 0 || bidx >= vram_blocks || count > vram_block_bytes) {
                    op = ~0u;
                } else {
                    blk = vram_base + (u64)bidx * vram_block_bytes;
                }
            }
            if (op != ~0u) {
                __shared__ u32 s_user_err;
                if (threadIdx.x == 0) s_user_err = 0u;
                __syncthreads();
                u32 e = flynnel_user_op(op, blk, count,
                                        (volatile unsigned char*)(slot + SLOT_PAYLOAD_OFF + 8),
                                        team_rank, blocks_per_lane);
                if (threadIdx.x == 0 && e != 0u) s_user_err = 1u;
                __syncthreads();
                if (s_user_err) op = ~0u;
            }
        }
        // OP_NOP and unknown ops fall through; unknown marks STATUS_ERR.

        __syncthreads();
        __threadfence_system();
        if (blocks_per_lane > 1u) {
            // The whole team worked this slot; rank 0 may only retire
            // it once every rank has finished writing.
            if (threadIdx.x == 0) atomicAdd((u32*)team_arrive, 1u);
            __threadfence_system();
            if (team_rank == 0u) {
                if (threadIdx.x == 0) {
                    while (atomicAdd((u32*)team_arrive, 0u) < blocks_per_lane) { }
                    st_vol(team_arrive, 0u);
                    __threadfence_system();
                    st_vol(d_status, op == ~0u ? STATUS_ERR : STATUS_DONE);
                    __threadfence_system();
                    st_vol(tail, ld_vol(tail) + 1u);
                    __threadfence_system();
                    // The ring moved: the team may take the next slot.
                    st_vol(team_gen, ld_vol(team_gen) + 1u);
                    __threadfence_system();
                }
            } else if (threadIdx.x == 0) {
                // Wait for rank 0 to publish the retirement, so this
                // block does not read the same slot twice.
                u32 want = my_gen_local + 1u;
                while (ld_vol(team_gen) != want) { }
            }
            if (threadIdx.x == 0) my_gen_local += 1u;
            __syncthreads();
        } else {
            if (threadIdx.x == 0) {
                st_vol(d_status, op == ~0u ? STATUS_ERR : STATUS_DONE);
                __threadfence_system();
                st_vol(tail, ld_vol(tail) + 1u);
                __threadfence_system();
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        // Device-scope atomic on mapped memory: exact among GPU threads
        // (one increment per exiting block; the host sums per launch).
        atomicAdd((u32*)(base + HDR_EXITS_OFF), 1u);
        __threadfence_system();
    }
}

// ------------------------------------------------------------- calibration

// Doorbell ping-pong + globaltimer sampling. Round i: wait ping == i,
// record globaltimer into gts[i], answer pong = i. Spin-capped so a
// broken handshake exits long before any watchdog fires.
extern "C" __global__ void flynnel_peer_calib_pong(unsigned char* base, u32 rounds)
{
    volatile u32* ping = (volatile u32*)(base + HDR_CALIB_PING_OFF);
    volatile u32* pong = (volatile u32*)(base + HDR_CALIB_PONG_OFF);
    volatile u64* gts = (volatile u64*)(base + HDR_GTS_OFF);
    u64 spins = 0;
    for (u32 i = 1; i <= rounds; ++i) {
        while (ld_vol(ping) != i)
            if (++spins > 4000000ULL) return;
        gts[i] = gtimer();
        __threadfence_system();
        st_vol(pong, i);
        __threadfence_system();
    }
}

// Fischer timed-lock contender (GPU side of the calibration self-test).
// Identical protocol to the CPU side: claim, wait Delta, verify claim,
// enter. The critical-section occupancy word doubles as the violation
// detector; under a correct mutex it is never observed != 1.
//
// Contention EVIDENCE is part of the contract: `started` is raised so
// the CPU contender begins only once this kernel is resident, and
// every round that observed the lock held (or lost the claim recheck)
// bumps the contended counter. A zero-violation run without contended
// rounds on both sides is inconclusive, not a pass.
extern "C" __global__ void flynnel_peer_fischer(
    unsigned char* base, u32 acqs, u64 delta_ns, u64 cs_ns)
{
    volatile u32* x = (volatile u32*)(base + HDR_FISCHER_X_OFF);
    volatile i32* cs = (volatile i32*)(base + HDR_FISCHER_CS_OFF);
    volatile u32* viol = (volatile u32*)(base + HDR_FISCHER_VIOL_OFF);
    volatile u32* done = (volatile u32*)(base + HDR_FISCHER_ACQS_OFF);
    volatile u32* started = (volatile u32*)(base + HDR_FISCHER_STARTED_OFF);
    volatile u32* cont = (volatile u32*)(base + HDR_FISCHER_GPU_CONT_OFF);
    st_vol(started, 1u);
    __threadfence_system();
    u64 wall0 = gtimer();
    for (u32 k = 0; k < acqs; ++k) {
        u32 contended = 0;
        for (;;) {
            if (gtimer() - wall0 > 3000000000ULL) { __threadfence_system(); return; }
            while (ld_vol(x) != 0u) {
                contended = 1;
                if (gtimer() - wall0 > 3000000000ULL) { __threadfence_system(); return; }
            }
            st_vol(x, 2u);
            __threadfence_system();
            u64 t0 = gtimer();
            while (gtimer() - t0 < delta_ns) {}
            if (ld_vol(x) == 2u) break;
            contended = 1;
        }
        if (contended) {
            st_vol(cont, ld_vol(cont) + 1u);
        }
        *cs = *cs + 1;
        __threadfence_system();
        if (*cs != 1) st_vol(viol, ld_vol(viol) + 1u);
        u64 t0 = gtimer();
        while (gtimer() - t0 < cs_ns) {}
        *cs = *cs - 1;
        __threadfence_system();
        st_vol(x, 0u);
        __threadfence_system();
        st_vol(done, k + 1u);
    }
}

// L2-persistence demonstrator. Every outer pass re-reads the whole HOT
// working set (the same addresses each time, which a persisting
// access-policy window can pin in the set-aside L2) and then streams a
// POLLUTER buffer sized at or above L2, which evicts HOT unless it is
// pinned. The kernel is identical whether or not the stream carries an
// access-policy window; the window is the only variable, so the two
// timings are a fair A/B on L2 persistence. atomicAdd keeps the reads
// from being optimized away.
extern "C" __global__ void flynnel_l2_hammer(
    const unsigned* hot, unsigned hot_len,
    const unsigned* pol, unsigned pol_len,
    unsigned iters, unsigned long long* out)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
    unsigned long long acc = 0;
    for (unsigned it = 0; it < iters; ++it) {
        for (unsigned i = tid; i < hot_len; i += stride) {
            acc += hot[i];
        }
        for (unsigned j = tid; j < pol_len; j += stride) {
            acc += pol[j];
        }
    }
    atomicAdd(out, acc);
}

// System-atomics safety probe: GPU claims slots with atomicCAS_system
// while the CPU claims with its own interlocked CAS. Conservation of
// claims decides the `sys_atomics_ok` capability flag; on PCIe hosts
// without native atomics this LOSES claims and the flag stays off.
//
// Overlap is FORCED, not hoped for: the kernel raises `started`
// before claiming and BOTH sides then walk the array forward from
// slot 0, so the claim fronts track each other and the same words
// race densely for the whole run (a single meeting-point front is
// not enough contention to expose lost claims reliably). The host
// side additionally requires both sides to have won slots before
// trusting the result.
extern "C" __global__ void flynnel_peer_cas_probe(u32* slots, u32 n,
                                                  u32* gpu_won, u32* started)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *(volatile u32*)started = 1u;
        __threadfence_system();
    }
    u32 won = 0;
    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (atomicCAS_system(&slots[i], 0u, 2u) == 0u) won++;
    }
    atomicAdd(gpu_won, won);   // device scope: exact among GPU threads
}

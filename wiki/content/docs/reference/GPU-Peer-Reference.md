---
title: GPU-Peer Reference
weight: 8
---

The `gpu-peer` feature makes the GPU a shared-memory PEER of the
scheduler: one memory-mapped file is simultaneously a plain mapped
region (this process and any process opening the same file) and
device-visible memory (registered with the CUDA driver) that a
resident kernel polls. Work moves through Lamport single-producer /
single-consumer lanes with doorbell signalling - no kernel launch on
the per-message path, no atomics across the CPU/GPU boundary, no
copies beyond the payload writes.

Enable with `--features gpu-peer` (pulls cudarc with dynamic driver
loading + memmap2). The kernels ship as pre-generated PTX embedded in
the crate and JIT-compiled by the driver at runtime: consumers never
need the CUDA toolkit. `kernels/gpu_peer.cu` is the source of record;
its header documents the regeneration command.

## Host calibration - the constants are measured, never baked

`GpuPeer::init` measures, on the RUNNING host, everything the
substrate depends on, then stores the results in the region header so
attaching processes inherit them (`PeerCalibration`):

| Constant | How it is measured |
|---|---|
| `rtt_min/median/p99_ns` | doorbell ping-pong against a resident kernel |
| `clock_err_ns` | paired QPC/globaltimer sampling, integer-domain differencing, best-RTT-quartile spread |
| `one_way_ns` | `rtt_p99 / 2 + clock_err` (the visibility bound) |
| `delta_ns` | `clamp(10 x one_way, 5us..=100us)`, then VALIDATED by a live CPU-vs-GPU Fischer self-test with CONTENTION EVIDENCE: the CPU contends only after the GPU contender is resident, both sides must log contended rounds above a floor (a zero-violation run without interleaving is inconclusive, never a pass), a violation escalates Delta x2, and the grant needs two consecutive contended clean runs. The granting run's contended counts are exposed as `lock_cpu_contended` / `lock_gpu_contended` |
| `launch_ns` | kernel launch + synchronize median (the wake-from-idle cost) |
| `sys_atomics_ok` | forced-contention `atomicCAS_system` vs CPU CAS conservation probe; granted only on two consecutive fully-contended conserving runs |

Capability flags (`doorbell_ok`, `timed_lock_ok`, `sys_atomics_ok`)
gate every protocol at runtime: a host with a coherent CPU-GPU link
measures tighter constants and unlocks more, a PCIe host without
native cross-device atomics keeps the timed protocols. No
compile-time platform assumptions exist anywhere in the module.

## API surface

```rust
pub struct GpuPeerConfig { region_path, lanes, slot_bytes, slots_per_lane,
                           quantum_ns, idle_exit_ns, device_ordinal,
                           blocks_per_lane }
pub struct GpuPeer;
impl GpuPeer {
    pub fn init(config: GpuPeerConfig) -> Result<Self, GpuPeerError>;
    pub fn calibration(&self) -> PeerCalibration;
    pub fn submit(&mut self, op: u32, payload: &[u8]) -> Result<Ticket, GpuPeerError>;
    pub fn is_done(&self, ticket: Ticket) -> bool;
    pub fn wait(&mut self, ticket: Ticket, timeout: Duration) -> Result<u32, GpuPeerError>;
    pub fn read_result(&self, ticket: Ticket, dst: &mut [u8]);
    pub fn reap(&mut self, ticket: Ticket) -> Result<(), GpuPeerError>;
    pub fn timed_lock_acquire(&self, timeout: Duration) -> Result<(), GpuPeerError>;
    pub fn timed_lock_release(&self);
    // Resident blocks and wide ops (detailed in the sections below):
    pub fn pin(&mut self, data: &[u8]) -> Result<ResidentHandle, GpuPeerError>;
    pub fn pin_bulk(&mut self, data: &[u8]) -> Result<ResidentHandle, GpuPeerError>;
    pub fn pin_prefetch(&mut self, data: &[u8]) -> Result<(ResidentHandle, Ticket), GpuPeerError>;
    pub fn write_resident_bulk(&mut self, handle: &ResidentHandle, data: &[u8]) -> Result<(), GpuPeerError>;
    pub fn fetch(&mut self, handle: &ResidentHandle, out: &mut [u8]) -> Result<(), GpuPeerError>;
    pub fn fetch_bulk(&mut self, handle: &ResidentHandle, out: &mut [u8]) -> Result<(), GpuPeerError>;
    pub fn unpin(&mut self, handle: ResidentHandle) -> Result<(), GpuPeerError>;
    pub fn resident_ptr(&self, handle: &ResidentHandle) -> Result<(u64, usize), GpuPeerError>;
    pub fn submit_resident(&mut self, op: u32, handle: &ResidentHandle) -> Result<Ticket, GpuPeerError>;
    pub fn submit_user(&mut self, op: u32, handle: Option<&ResidentHandle>, args: &[u8]) -> Result<Ticket, GpuPeerError>;
    pub fn compile_wide_kernel(&self, src: &str, entry: &str) -> Result<WideKernel, GpuPeerError>;
    pub fn load_wide_kernel_ptx(&self, ptx: &str, entry: &str) -> Result<WideKernel, GpuPeerError>;
    pub fn launch_wide(&self, kernel: &WideKernel, grid_blocks: u32, block_threads: u32, ptrs: &[u64], scalars: &[u32]) -> Result<(), GpuPeerError>;
    pub fn launch_wide_async(&self, ..same..) -> Result<(), GpuPeerError>;
    pub fn sync_wide(&self) -> Result<(), GpuPeerError>;
    pub fn pause_poller(&mut self) -> Result<(), GpuPeerError>;
    pub fn resume_poller(&mut self) -> Result<(), GpuPeerError>;
    pub fn context(&self) -> &Arc<CudaContext>;
    pub fn wide_stream(&self) -> &Arc<CudaStream>;
}
```

Built-in opcodes: `OP_NOP`, `OP_ADD1_F32` (in-place +1.0 per f32),
`OP_SUM_U32` (u64 sum replaces payload start). Slots are reusable
only after `reap` (results are written in place); reaping is in-order
per lane.

## Execution model

The consumer is a bounded-quantum persistent kernel (one block per
lane): it runs at most `quantum_ns` per launch (watchdog-safe on
display GPUs), parks after `idle_exit_ns` without work, and is
relaunched on demand - wake-from-idle costs one launch
(`launch_ns`), which a continuously fed queue never pays. Running
vs parked is tracked by a device-scope-atomic exit counter plus a
generation word (no stream queries; stragglers from superseded
launches exit at their next poll).

The Fischer timed lock provides cross-device mutual exclusion from
plain stores + the calibrated `delta_ns` - the mechanism that
replaces the cross-device CAS the hardware does not reliably provide
over PCIe. It is available only when the init self-test passed
(`timed_lock_ok`).

## Device-resident blocks (the pool the scheduler owns by index)

`GpuPeerConfig::vram_blocks x vram_block_bytes` of device memory back
a block pool the CPU addresses by INDEX only - it never dereferences
device memory, exactly the discipline the host-side region already
follows. `pin(data)` uploads once into a claimed block; every
subsequent `submit_resident(OP_ADD1_F32_V | OP_SUM_U32_V, &handle)`
moves only an 8-byte param header across the bus while the payload
stays in VRAM; `fetch` downloads on demand; `unpin` returns the
block. The pool never evicts silently - placement is the scheduler's
decision.

Dependencies come from lane affinity: every task touching one handle
rides the handle's lane, so same-handle ordering (read-after-write,
write-after-write) is the lane's FIFO order with zero extra
synchronization; independent handles ride independent lanes in
parallel. This is data-driven dependency ordering expressed directly
in the transport instead of a separate DAG engine.

Measured on the symmetric in-repo comparison (both paths pipelined
over four lanes, 64 KB payloads, RTX 3070): resident tasks 2.1x
faster per task than shipping the payload every task, with 16,000x
less bus traffic (8 B vs 128 KB per task).

## Two execution shapes: doorbell ops and wide ops

A resident op can run two ways, and the choice is about parallelism,
not correctness. Which one fits depends on a single question: is this
one small op among many, or one large data-parallel op?

The **doorbell op** (`submit_user`, `submit_resident`) rides a lane
and runs on ONE block of the persistent poller - 256 threads, one SM.
That is exactly right for many small ops: the doorbell dispatches in
~2.9 us and the poller consumes op after op without a launch. But a
single large op - a 12k-pixel convolution, a full-image stencil -
stays pinned to that one SM while the rest of the device sits idle.

The **wide op** (`launch_wide`) is the answer for that case: a
caller-authored `__global__` launched across a full grid over a
resident block, so every SM works the op while the data stays
resident. `compile_wide_kernel(src, entry)` NVRTC-compiles the
kernel; `resident_ptr(&handle)` hands back the block's device
address; `launch_wide(kernel, grid, block, ptrs, scalars)` launches
it (pointer arguments first, then u32 scalars, matching the kernel
signature) and blocks until done. Measured on the RTX 3070 with a
16,384-pixel 3x3 blob blur, the identical grid-stride kernel: grid=1
(one SM) 41.0 us, grid=full (64 blocks, all SMs) 19.2 us - a 2.14x
lift from spreading the op across the device, same result both ways.
So keep small ops on the doorbell and send large ones through
`launch_wide`. E2E: [`examples/gpu_wide_op_demo.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/gpu_wide_op_demo.rs).

Wide ops run on their OWN stream, separate from the poller's. That
matters more than it sounds. A wide op on the poller's stream would
queue BEHIND a resident poller quantum and wait up to `idle_exit_ns`
before it even starts - a wide op launched with a 40 ms-idle poller
resident measured 0.10 ms on the dedicated stream, where the shared
stream would have made it wait the full 40 ms. So a workload can
interleave doorbell ops and wide ops without one stalling the other.

### Block teams per lane

`GpuPeerConfig::blocks_per_lane` sits between the two shapes. At 1 a
lane is served by one poller block on one SM, which is the shape the
doorbell op description above assumes. Above 1 a lane is worked by a
team of consecutive blocks: rank 0 owns the ring and retires the slot
once the whole team has finished, and the user op receives its rank
and the team size and strides its work over them. One doorbell op
then spreads across the device without the caller authoring a
separate `__global__`.

The team size that covers the device once is the multiprocessor count
divided by the lane count.
[`backend::detect::cuda_sm_count`](Backend-System.md#device-properties)
returns that count, and returns it without a CUDA context, so the
config is sized to the host before `GpuPeer::init` reads it:

```rust
use flynnel::backend::detect::cuda_sm_count;
use flynnel::gpu_peer::GpuPeerConfig;

let lanes = 4;
let config = GpuPeerConfig {
    lanes,
    blocks_per_lane: match cuda_sm_count(0) {
        Some(sm) => (sm / lanes).max(1),
        None => 1,
    },
    ..Default::default()
};
```

The RTX 3070 reports 46 multiprocessors, so four lanes take 11 blocks
each. `None` leaves the caller with the single-block default, which is
the correct shape on a host with no device to spread across. Clamp the
quotient to whatever team size the user op supports: a hook that opens
with `if (team_rank != 0u) return 0u;` serves one rank and stays on one
SM whatever the config says.

### Batching wide ops and quiescing the poller

Two knobs matter for a many-small-kernel workload on WDDM, where
every stream sync flushes the command buffer.

`launch_wide` is synchronous - it flushes per call, which is fine for
one-off work but costly for a chain. `launch_wide_async` enqueues on
the wide stream and returns; the stream is FIFO, so a dependent chain
stays correct, and one `sync_wide` at the end pays a single flush for
the whole batch. Measured on a 200-kernel dependent chain (RTX 3070):
per-call 3.71 ms, async batch 2.16 ms, a 1.7x lift from paying one
flush instead of 200.

`pause_poller` / `resume_poller` quiesce the busy-polling doorbell
poller. The poller spins its lanes while resident, holding SM
occupancy; a co-running memory-bound wide op wants those SMs to keep
enough warps in flight to saturate memory bandwidth. Pause the poller
around a heavy wide batch to hand it the whole device, then resume;
small doorbell ops submitted while paused simply queue and are
consumed after resume. Measured on a 16 MiB streaming op: pausing
reclaimed ~10% on this hardware, more where the co-runner is more
occupancy-starved. E2E:
[`examples/gpu_wide_batch_demo.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/gpu_wide_batch_demo.rs).

### Loading wide kernels from checked-in PTX; the context and stream

`load_wide_kernel_ptx(ptx, entry)` builds a `WideKernel` from PTX
text the driver JIT-compiles, so a consumer can ship its kernels the
same way Flynnel ships its own, with no NVRTC at run time.
`context()` and `wide_stream()` expose the CUDA context and the wide
stream: work a consumer enqueues on that stream (its own kernels,
library calls) is FIFO-ordered with `launch_wide_async` launches and
fenced by `sync_wide`. cudarc's `CudaContext::new` retains the
device's PRIMARY context, so device pointers from the peer's resident
pool are valid for any other cudarc user on the same device in the
process (verified by the accel-route parity test, which launches on
peer-pinned buffers through a separate `CudaBackend`).

`pin_bulk` claims the lowest free run of consecutive pool blocks that
covers the data, whatever order earlier blocks were released in, and
`unpin` returns every block of the span.

## Batched linear algebra (`gpu_peer::linalg`)

Four op families as house-owned f64 kernels in
`kernels/linalg_f64.ptx` (driver-JIT'd; NVRTC fallback from the
embedded `.cu` when a driver rejects the PTX; no cuBLAS / cuSOLVER at
build or run time), over resident blocks:

| Op | Kernel | Contract |
|---|---|---|
| einsum | `flynnel_einsum_f64` | Any `"ij,jk->ik"`-style contraction over one or two operands (matmul, outer product, n-d outer, axis sums, trace, transposed contractions), batched; one thread per output element, fma-accumulated in ascending contracted-index order |
| GEMM | `flynnel_gemm_batched_f64` | Batched row-major `C = A x B`, 16x16 shared-memory tiles, fma with `k` ascending |
| symmetric eigen | `flynnel_syev_jacobi_f64_{blk,thr}` | Batched Jacobi eigendecomposition, `n <= 64`; eigenvalues in diagonal order plus optional eigenvectors |
| SVD | `flynnel_gesvd_jacobi_f64_{blk,thr}` | Batched one-sided Jacobi SVD, `m >= n`, `m <= 64`; `A` is overwritten with `U`, singular values in column order, optional `V` |
| symmetric eigen (bisection) | `flynnel_syev_bisect_f64_blk` (`linalg_bisect_f64.ptx`) | Batched Householder tridiagonalization (EISPACK tred2) then bisection with Sturm counts, one thread per eigenvalue; eigenvectors by inverse iteration with cluster orthonormalization; `n <= 64`, one block per matrix; eigenvalues ascending |
| SVD (bisection) | `flynnel_gesvd_bisect_f64_blk` (`linalg_bisect_f64.ptx`) | Batched Householder bidiagonalization (LINPACK dsvdc) then bisection on the Golub-Kahan tridiagonal for the singular values, inverse iteration for the vectors; `m >= n`, `m <= 64`; `A` is overwritten with `U`, singular values descending, optional `V`, needs a scratch buffer |
| LU | `flynnel_getrf_f64_blk`, `flynnel_getrs_f64_blk` (`linalg_lu_f64.ptx`) | Batched LU with partial pivoting, one block per matrix, `n <= 64`: the factor packs `U` and the unit-lower multipliers in place with the pivot row per step and a flag one past the first zero pivot; the solve takes `n x nrhs` right-hand sides (`nrhs <= 64`) or, with the identity flag, writes the inverse; every update is an explicit fused multiply-add in the host reference's order |
| GEMM (Ozaki) | `flynnel_ozaki_{rowexp,colexp,split_a,split_bt,gemm}_f64` (`ozaki_f64.ptx`) | `C = A x B` on the int8 tensor cores: operands split into eight 7-bit slices aligned to their row (A) or column (B) maximum exponent, 36 slice pairs multiplied exactly by int8 mma into int32, recombined in f64 with two-sum compensation; `m`, `n`, `k` multiples of 32, `k <= 16384`, finite operands |

Two Jacobi kernel shapes exist because the best one depends on `n`
and on the batch: `blk` runs one 256-thread block per matrix with the
matrix in 32 KB of static shared memory and applies `n / 2` disjoint
rotations per tournament round; `thr` runs one thread per matrix in
local memory (`n <= 16`) and needs enough matrices to fill the
device. `jacobi_shape_for(n, batch)` picks `thr` when `n <= 16` and
`batch >= JACOBI_THREAD_SHAPE_BATCH_PER_N * n` (256 per unit of `n`),
`blk` otherwise. The constant is measured, not chosen:
`benches/gpu_linalg.rs` on an RTX 3070 and an RTX 5070 puts the
crossover at batch 1024 for n=4, 2048 for n=8 and 4096 for n=16, for
syev and gesvd alike, with `blk` ahead by up to 3x below it (n=16,
batch 1024) and `thr` ahead by up to 24x above it (n=4, batch 65536).
The measured tables are at the end of this section.

Three surfaces over the same kernels:

```rust
pub struct LinalgKernels { einsum, gemm, syev_blk, syev_thr, gesvd_blk, gesvd_thr, syev_bisect, gesvd_bisect, getrf, getrs }
impl LinalgKernels { pub fn load(peer: &GpuPeer) -> Result<Self, GpuPeerError> }
pub struct EinsumSpec;               // EinsumSpec::parse("ij,jk->ik", &a_shape, Some(&b_shape))
// Async, over device addresses; queue a step, then peer.sync_wide() once:
pub fn launch_einsum(peer, k, spec, tables_dev, a, b, out, batch)
pub fn launch_gemm(peer, k, a, b, c, batch, m, n, kdim)
pub fn launch_syev(peer, k, a, w, v: Option<u64>, batch, n, max_sweeps, shape)
pub fn launch_gesvd(peer, k, a, sigma, v: Option<u64>, batch, m, n, max_sweeps, shape)
pub fn launch_syev_bisect(peer, k, a, w, v: Option<u64>, scratch, batch, n)        // eigenvalues ascending
pub fn launch_gesvd_bisect(peer, k, a, sigma, v: Option<u64>, scratch, batch, m, n) // singular values descending
pub fn syev_bisect_scratch_bytes(batch, n) / gesvd_bisect_scratch_bytes(batch, m, n) // workspace sizes
// Synchronous over host buffers (pin, launch, sync, fetch, unpin):
pub fn einsum_batched(..) / gemm_batched(..) / syev_batched(..) / gesvd_batched(..)
pub fn syev_bisect_batched(peer, k, a, batch, n, want_v) / gesvd_bisect_batched(peer, k, a, batch, m, n, want_v)
// The measured routing: bisection from SYEV_BISECT_MIN_N = 32 (eigen) and GESVD_BISECT_MIN_N = 64 (SVD), Jacobi below
pub fn syev_method_for(n) / gesvd_method_for(n) -> LinalgMethod
pub fn syev_auto_batched(peer, k, a, batch, n, want_v) / gesvd_auto_batched(peer, k, a, batch, m, n, want_v)
// LU (linalg_lu_f64.ptx), async over device addresses and synchronous over host buffers:
pub fn launch_getrf(peer, k, a, piv, info, batch, n)                      // in place; piv and info are i32
pub fn launch_getrs(peer, k, lu, piv, b, batch, n, nrhs, identity_rhs)    // solves in place, or the inverse
pub struct LuResult { lu, piv, info }
pub fn getrf_batched(peer, k, a, batch, n) -> LuResult
pub fn getrs_batched(peer, k, lu, piv, b, batch, n, nrhs) / getri_batched(peer, k, lu, piv, batch, n)
pub fn lu_det_batched(lu, piv, batch, n) -> Vec<f64>                     // host fold over the factors
// Tandem: the batch split between the device and the CPU pool by the call site's learned share
// (sched::hybrid_auto_split_ranges). The helpers are #[track_caller]: the share is learned per
// calling source location and per log2 batch bucket, starting at 500 per mille, so call from one
// place. The CPU share runs the cpu:: reference on the pool in runs of at most TANDEM_CPU_CHUNK = 64
// matrices, two runs per worker. Eigenvalues come back ascending and singular values descending
// for every item whichever side computed it.
pub fn gemm_tandem_batched(peer, k, plan, a, b, batch, m, n, kdim) -> (Vec<f64>, SplitReport)
pub fn syev_tandem_batched(peer, k, plan, a, batch, n, want_v) -> ((w, Option<v>), SplitReport)
pub fn gesvd_tandem_batched(peer, k, plan, a, batch, m, n, want_v) -> (GesvdResult, SplitReport)
pub fn sort_eigenpairs_ascending(w, v: Option<&mut [f64]>, n) / sort_singular_descending(u, sigma, v, m, n)
// Ragged inputs: one uniform batch per shape (every kernel takes one shape per call)
pub fn group_by_shape<S: Ord + Copy>(shapes: &[S]) -> Vec<(S, Vec<usize>)>   // ascending shape, each index once
pub fn gather_items(src, offsets, item_len, indices) -> Vec<f64> / scatter_items(batch, offsets, item_len, indices, dst)
// gpu_peer::ozaki - the tensor-core GEMM, with its workspace pinned in the resident pool:
pub struct OzakiKernels;  impl OzakiKernels { pub fn load(peer) -> Result<Self, GpuPeerError> }
pub struct OzakiWorkspace; // OzakiWorkspace::new(peer, batch, m, n, k), ::bytes(..), ::release(peer)
pub fn launch_ozaki_gemm(peer, kern, ws, a, b, c)                        // async, device addresses
pub fn ozaki_gemm_batched(peer, kern, a, b, batch, m, n, k) -> Vec<f64>
pub fn error_bound(a_row, b_col) -> f64                                  // 2^-53 * k * max|row| * max|col|
// The Ozaki path is explicit: gemm_batched and the accel route stay on the native f64 kernel.
// CPU references with the kernels' semantics:
pub mod cpu { einsum, gemm_batched, syev_jacobi[_batched], gesvd_jacobi[_batched], getrf_batched, getrs_batched, getri_batched }
// accel_op registrations (see Backend System): CPU side = cpu::*, kernel side = the blk kernels
pub fn register_linalg_accel_ops() -> LinalgAccelOps
pub fn bind_linalg_kernels(&ops, backend) -> Result<(), BackendError>
pub fn gemm_accel(..) / syev_accel(..) / gesvd_accel(..) -> AccelReport
```

Parity contract, verified on the device by
`tests/gpu_linalg_parity.rs` (RTX 3070 and RTX 5070): einsum and gemm
match the CPU references bit for bit (the same fma order); the Jacobi
ops rotate pairs in tournament order while the CPU references sweep
cyclically, so eigenvalues and singular values agree to `1e-10`
relative, with eigenvectors checked by `A v = lambda v`, and the SVD
by `A = U diag(sigma) V^T` and `U^T U = I` at `1e-9` of the operand
norm. Reducers a consumer wants (trace, Frobenius norm, spectral
radius, nuclear norm, condition number, determinant) are host-side
folds over the einsum, eigenvalue and singular-value outputs.

The bisection kernels are checked by `tests/gpu_bisect_parity.rs`
against the same Jacobi CPU references: eigenvalues and singular
values to `1e-10` relative, eigenvectors by `A v = lambda v`, the
SVD by reconstruction and orthonormal `U` and `V`, square and
rectangular, with repeated and vanishing spectra. Their outputs are
sorted (eigenvalues ascending, singular values descending), unlike
the Jacobi kernels' diagonal order. Eigenvalues closer than
`1e3 * eps * ||T||` are treated as one cluster: their inverse
iterations use distinct shifts and the cluster is orthonormalized
by its first thread.

The LU kernels are checked by `tests/gpu_lu_parity.rs`: factors,
pivots and flags match `cpu::getrf_batched` bit for bit at n = 1, 2,
3, 8, 16, 33 and 64, solves and inverses match `cpu::getrs_batched`
and `cpu::getri_batched` bit for bit, residuals `A x - b` and `A
inv(A) - I` hold at `1e-12` of the operand scale, determinants follow
the pivot signs, and a zero column sets the same flag on both sides.
The Frobenius inner product `"ij,ij->"` is covered in
`tests/gpu_linalg_parity.rs`, bit for bit against `cpu::einsum`.

The tandem helpers are checked by `tests/gpu_tandem_parity.rs`: the
GEMM result is bit-identical to the CPU reference whatever the
split, both sides always receive items, the share stays within
50..=950 per mille and after six calls sits on the side of the
measured per-item balance; eigenvalues and singular values match the
sorted references to `1e-10` relative with `A v = lambda v` and the
SVD reconstruction at `1e-9`, ascending and descending per item.

The Ozaki GEMM is not bit-identical to `flynnel_gemm_batched_f64`:
its summation order differs from the fma-ordered kernel, and
elements far below their row or column maximum lose the bits that
fall outside the eight slices. `tests/gpu_ozaki_parity.rs` holds it
to the bound `error_bound` states, `2^-53 * k * max|A row| *
max|B column|` per element (plus the same allowance for the
reference's own rounding), on uniform and ill-scaled operands, and
to bit-identical results where the reference is exact (small
integers).

### Measured: kernels against the CPU

`benches/gpu_linalg.rs`, 2026-09-02. Two hosts: an RTX 3070 with a
Ryzen 7 2700 (16 threads) and an RTX 5070 with a Ryzen 9 7900X (24
threads). Every contender computes the same result from the same
inputs. GPU columns are kernel wall per call (one launch, one stream
sync) with the operands already resident, taken after a 300 ms ramp
of back-to-back launches so the device is at boost clocks; the 3070
otherwise measures at idle clocks for kernels under a millisecond.
CPU-par is the CPU reference run through Flynnel `collect_indexed`
with 64 matrices per item under the default adaptive plan; serial is
the same reference on one thread. Batches of 65536 at n=64 exceed the
bench's 1.5 GiB resident pool and are not measured. The 3070 columns
come from one run of the whole bench at commit `f068879` plus the
ramp; the 5070 columns from two runs on an idle machine at `a3fa53f`
(Jacobi sections, then GEMM and einsum), whose sub-millisecond cells
were already at boost.

#### Tandem: device alone, CPU pool alone, both

`benches/gpu_linalg.rs` section `tandem`. Every column is a
host-buffer helper end to end, so each carries its pin and fetch:
`gemm_batched` / `syev_auto_batched` / `gesvd_auto_batched` for the
device alone, the CPU reference through `collect_indexed` with 64
matrices per item for the pool alone, and the `*_tandem_batched`
helper after twelve warm calls have taught the call site its share
(the model moves one eighth per call; medians of three further calls).
Every call of a cell is made from the same source line: the helpers
are `#[track_caller]` and learn the share per calling location, so a
call from another line starts at 500 per mille. `share` is the CPU share in per
mille the site had learned, `cpu-side` and `dev-side` the two halves'
own wall times on one more call. Eigen and SVD rows request vectors.
Cells whose pins would exceed half the 1.5 GiB pool are skipped
(n = 64 at batch 8192). 2026-09-04, RTX 3070 with the Ryzen 7 2700
(16 threads), idle machine.

| op | n | batch | 3070 GPU ms | 3070 CPU-par ms | 3070 tandem ms | 3070 GPU/tan | 3070 par/tan | 3070 share | 3070 cpu-side ms | 3070 dev-side ms |
|---|---|---|---|---|---|---|---|---|---|---|
| gemm | 16 | 1024 | 6.751 | 1.379 | 2.081 | 3.24x | 0.66x | 789 | 0.975 | 1.035 |
| syev | 16 | 1024 | 8.880 | 22.253 | 8.643 | 1.03x | 2.57x | 460 | 6.148 | 6.797 |
| gesvd | 16 | 1024 | 11.228 | 19.726 | 8.315 | 1.35x | 2.37x | 621 | 7.597 | 7.304 |
| gemm | 16 | 8192 | 31.644 | 6.465 | 12.979 | 2.44x | 0.50x | 881 | 7.828 | 6.317 |
| syev | 16 | 8192 | 57.370 | 85.703 | 64.155 | 0.89x | 1.34x | 480 | 55.861 | 54.977 |
| gesvd | 16 | 8192 | 60.649 | 72.960 | 57.520 | 1.05x | 1.27x | 576 | 49.616 | 50.851 |
| gemm | 32 | 1024 | 15.059 | 10.578 | 7.868 | 1.91x | 1.34x | 818 | 5.919 | 9.286 |
| syev | 32 | 1024 | 31.110 | 152.024 | 31.906 | 0.98x | 4.76x | 314 | 29.785 | 30.434 |
| gesvd | 32 | 1024 | 45.124 | 139.589 | 60.463 | 0.75x | 2.31x | 455 | 53.221 | 52.673 |
| gemm | 32 | 8192 | 221.141 | 43.403 | 70.690 | 3.13x | 0.61x | 855 | 49.830 | 45.817 |
| syev | 32 | 8192 | 293.604 | 638.386 | 278.310 | 1.05x | 2.29x | 323 | 235.625 | 203.108 |
| gesvd | 32 | 8192 | 401.058 | 740.072 | 400.764 | 1.00x | 1.85x | 417 | 330.714 | 272.997 |
| gemm | 64 | 1024 | 98.197 | 88.010 | 56.013 | 1.75x | 1.57x | 742 | 37.126 | 31.553 |
| syev | 64 | 1024 | 144.415 | 1380.074 | 191.191 | 0.76x | 7.22x | 166 | 180.995 | 189.477 |
| gesvd | 64 | 1024 | 234.260 | 1321.027 | 305.610 | 0.77x | 4.32x | 281 | 287.879 | 292.405 |

The same section on the RTX 5070 with the Ryzen 9 7900X (24 threads),
2026-09-05 at 02:15, at commit 13ebee5; another process's accuracy
pass was running on one thread, the host's load sampled 1 percent
before the section and 0 percent after it.

| op | n | batch | 5070 GPU ms | 5070 CPU-par ms | 5070 tandem ms | 5070 GPU/tan | 5070 par/tan | 5070 share | 5070 cpu-side ms | 5070 dev-side ms |
|---|---|---|---|---|---|---|---|---|---|---|
| gemm | 16 | 1024 | 1.758 | 0.694 | 0.854 | 2.06x | 0.81x | 826 | 0.590 | 0.444 |
| syev | 16 | 1024 | 4.985 | 7.565 | 3.196 | 1.56x | 2.37x | 570 | 2.700 | 2.756 |
| gesvd | 16 | 1024 | 6.648 | 6.615 | 3.398 | 1.96x | 1.95x | 723 | 2.782 | 2.666 |
| gemm | 16 | 8192 | 11.871 | 1.606 | 3.411 | 3.48x | 0.47x | 929 | 1.988 | 2.780 |
| syev | 16 | 8192 | 23.604 | 31.222 | 22.149 | 1.07x | 1.41x | 618 | 19.264 | 17.939 |
| gesvd | 16 | 8192 | 24.898 | 24.602 | 20.264 | 1.23x | 1.21x | 731 | 18.364 | 17.957 |
| gemm | 32 | 1024 | 5.987 | 3.187 | 2.737 | 2.19x | 1.16x | 873 | 1.930 | 1.452 |
| syev | 32 | 1024 | 17.029 | 32.142 | 14.760 | 1.15x | 2.18x | 452 | 10.360 | 12.176 |
| gesvd | 32 | 1024 | 27.264 | 47.893 | 18.102 | 1.51x | 2.65x | 603 | 17.917 | 15.307 |
| gemm | 32 | 8192 | 46.786 | 9.731 | 13.407 | 3.49x | 0.73x | 903 | 9.944 | 17.581 |
| syev | 32 | 8192 | 131.182 | 211.761 | 102.601 | 1.28x | 2.06x | 491 | 100.501 | 110.317 |
| gesvd | 32 | 8192 | 216.579 | 197.723 | 127.142 | 1.70x | 1.56x | 606 | 120.441 | 113.048 |
| gemm | 64 | 1024 | 23.998 | 24.077 | 13.551 | 1.77x | 1.78x | 793 | 10.505 | 14.889 |
| syev | 64 | 1024 | 48.131 | 408.421 | 58.838 | 0.82x | 6.94x | 229 | 59.678 | 53.080 |
| gesvd | 64 | 1024 | 111.143 | 422.921 | 104.577 | 1.06x | 4.04x | 399 | 102.547 | 90.087 |


#### Batched LU: factor, solve (nrhs = 1), inverse

`benches/gpu_linalg.rs` section `lu`, 2026-09-04, RTX 3070 with the
Ryzen 7 2700, same method as the tables below: GPU is kernel wall with
the operands resident (the factor refactors its resident copy in
place each launch; the work per launch does not depend on the data),
CPU-par the CPU reference through `collect_indexed` with 64 matrices
per item, serial the reference on one thread. Pin+fetch is the
host-buffer helper's transfer cost per call.

| op | n | batch | GPU ms | CPU-par ms | serial ms | GPU/par | GPU/ser | pin+fetch ms |
|---|---|---|---|---|---|---|---|---|
| getrf | 8 | 1024 | 0.229 | 0.195 | 0.694 | 0.85x | 3.0x | 1.57 |
| getrs | 8 | 1024 | 0.102 | 0.204 | 0.360 | 2.0x | 3.5x | 0.43 |
| getri | 8 | 1024 | 0.086 | 0.325 | 1.444 | 3.8x | 16.8x | 0.66 |
| getrf | 8 | 8192 | 1.597 | 0.706 | 5.867 | 0.44x | 3.7x | 10.54 |
| getrs | 8 | 8192 | 0.653 | 0.568 | 3.086 | 0.87x | 4.7x | 3.24 |
| getri | 8 | 8192 | 0.561 | 1.254 | 10.724 | 2.2x | 19.1x | 9.72 |
| getrf | 16 | 1024 | 0.445 | 1.021 | 5.865 | 2.3x | 13.2x | 6.60 |
| getrs | 16 | 1024 | 0.200 | 0.665 | 3.304 | 3.3x | 16.6x | 2.90 |
| getri | 16 | 1024 | 0.166 | 2.880 | 12.077 | 17.4x | 72.8x | 10.28 |
| getrf | 16 | 8192 | 3.375 | 4.450 | 50.842 | 1.3x | 15.1x | 108.45 |
| getrs | 16 | 8192 | 1.320 | 3.699 | 31.514 | 2.8x | 23.9x | 22.94 |
| getri | 16 | 8192 | 1.201 | 7.496 | 89.055 | 6.2x | 74.1x | 65.87 |
| getrf | 32 | 1024 | 0.977 | 5.249 | 25.090 | 5.4x | 25.7x | 29.35 |
| getrs | 32 | 1024 | 0.372 | 2.655 | 13.135 | 7.1x | 35.3x | 10.66 |
| getri | 32 | 1024 | 0.494 | 11.150 | 65.554 | 22.6x | 132.7x | 50.47 |
| getrf | 32 | 8192 | 7.070 | 14.787 | 169.476 | 2.1x | 24.0x | 265.21 |
| getrs | 32 | 8192 | 2.277 | 12.318 | 91.214 | 5.4x | 40.1x | 67.40 |
| getri | 32 | 8192 | 3.578 | 38.783 | 463.215 | 10.8x | 129.5x | 271.97 |
| getrf | 64 | 1024 | 2.400 | 42.430 | 119.154 | 17.7x | 49.7x | 129.74 |
| getrs | 64 | 1024 | 0.713 | 22.075 | 83.828 | 31.0x | 117.6x | 53.58 |
| getri | 64 | 1024 | 2.362 | 38.649 | 266.501 | 16.4x | 112.8x | 111.90 |
| getrf | 64 | 8192 | 17.343 | 102.590 | 910.304 | 5.9x | 52.5x | 867.07 |
| getrs | 64 | 8192 | 5.231 | 44.250 | 446.476 | 8.5x | 85.4x | 253.29 |
| getri | 64 | 8192 | 17.319 | 189.269 | 1978.660 | 10.9x | 114.3x | 815.91 |

The same section on the RTX 5070 with the Ryzen 9 7900X (24 threads),
idle machine, same day:

| op | n | batch | GPU ms | CPU-par ms | serial ms | GPU/par | GPU/ser | pin+fetch ms |
|---|---|---|---|---|---|---|---|---|
| getrf | 8 | 1024 | 0.188 | 0.095 | 0.307 | 0.50x | 1.6x | 0.99 |
| getrs | 8 | 1024 | 0.080 | 0.073 | 0.186 | 0.91x | 2.3x | 0.33 |
| getri | 8 | 1024 | 0.058 | 0.176 | 0.610 | 3.0x | 10.4x | 0.46 |
| getrf | 8 | 8192 | 1.076 | 0.373 | 3.057 | 0.35x | 2.8x | 4.37 |
| getrs | 8 | 8192 | 0.506 | 0.253 | 2.697 | 0.50x | 5.3x | 1.41 |
| getri | 8 | 8192 | 0.380 | 0.595 | 6.038 | 1.6x | 15.9x | 4.15 |
| getrf | 16 | 1024 | 0.290 | 0.370 | 1.992 | 1.3x | 6.9x | 2.33 |
| getrs | 16 | 1024 | 0.146 | 0.205 | 1.276 | 1.4x | 8.7x | 0.94 |
| getri | 16 | 1024 | 0.148 | 0.607 | 4.402 | 4.1x | 29.7x | 2.00 |
| getrf | 16 | 8192 | 2.192 | 1.229 | 16.230 | 0.56x | 7.4x | 15.04 |
| getrs | 16 | 8192 | 0.844 | 0.814 | 9.688 | 0.97x | 11.5x | 4.67 |
| getri | 16 | 8192 | 0.824 | 2.306 | 32.130 | 2.8x | 39.0x | 14.88 |
| getrf | 32 | 1024 | 0.620 | 2.264 | 11.722 | 3.7x | 18.9x | 7.68 |
| getrs | 32 | 1024 | 0.278 | 0.664 | 4.534 | 2.4x | 16.3x | 2.31 |
| getri | 32 | 1024 | 0.321 | 4.989 | 27.752 | 15.6x | 86.5x | 6.82 |
| getrf | 32 | 8192 | 4.775 | 6.933 | 100.772 | 1.5x | 21.1x | 53.03 |
| getrs | 32 | 8192 | 1.831 | 2.920 | 40.296 | 1.6x | 22.0x | 16.14 |
| getri | 32 | 8192 | 2.609 | 14.173 | 205.939 | 5.4x | 78.9x | 50.06 |
| getrf | 64 | 1024 | 1.536 | 7.715 | 50.397 | 5.0x | 32.8x | 27.47 |
| getrs | 64 | 1024 | 0.524 | 3.378 | 21.215 | 6.5x | 40.5x | 7.54 |
| getri | 64 | 1024 | 1.573 | 16.235 | 118.151 | 10.3x | 75.1x | 24.51 |
| getrf | 64 | 8192 | 12.260 | 29.600 | 389.781 | 2.4x | 31.8x | 192.37 |
| getrs | 64 | 8192 | 3.700 | 12.119 | 134.918 | 3.3x | 36.5x | 60.77 |
| getri | 64 | 8192 | 11.568 | 70.609 | 881.799 | 6.1x | 76.2x | 189.53 |

#### Batched GEMM (m = n = k)

| n | batch | 3070 GPU ms | 3070 CPU-par ms | 3070 serial ms | 3070 GPU/par | 3070 GPU/ser | 5070 GPU ms | 5070 CPU-par ms | 5070 serial ms | 5070 GPU/par | 5070 GPU/ser |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1024 | 0.037 | 0.192 | 0.466 | 5.2x | 12.6x | 0.018 | 0.089 | 0.363 | 4.9x | 20.2x |
| 8 | 8192 | 0.118 | 3.132 | 4.778 | 26.5x | 40.5x | 0.092 | 1.855 | 3.276 | 20.2x | 35.6x |
| 8 | 65536 | 0.828 | 23.304 | 39.653 | 28.1x | 47.9x | 0.513 | 17.197 | 25.019 | 33.5x | 48.8x |
| 16 | 1024 | 0.041 | 1.036 | 4.677 | 25.3x | 114.1x | 0.026 | 0.452 | 3.099 | 17.4x | 119.2x |
| 16 | 8192 | 0.214 | 7.197 | 38.368 | 33.6x | 179.3x | 0.133 | 4.009 | 24.555 | 30.1x | 184.6x |
| 16 | 65536 | 1.587 | 59.883 | 317.968 | 37.7x | 200.4x | 0.997 | 34.084 | 188.773 | 34.2x | 189.3x |
| 32 | 1024 | 0.216 | 11.323 | 61.322 | 52.4x | 283.9x | 0.134 | 2.958 | 22.215 | 22.1x | 165.8x |
| 32 | 8192 | 1.603 | 45.385 | 322.299 | 28.3x | 201.1x | 0.992 | 15.867 | 188.569 | 16.0x | 190.1x |
| 32 | 65536 | 12.588 | 364.272 | 2606.454 | 28.9x | 207.1x | 7.971 | 106.250 | 1314.482 | 13.3x | 164.9x |
| 64 | 1024 | 1.576 | 55.813 | 355.628 | 35.4x | 225.7x | 0.997 | 16.633 | 188.629 | 16.7x | 189.2x |
| 64 | 8192 | 12.542 | 300.222 | 2917.730 | 23.9x | 232.6x | 8.059 | 105.231 | 1538.680 | 13.1x | 190.9x |

#### einsum outer product `"i,j->ij"` and row sum `"ij->i"`

| op | n | batch | 3070 GPU ms | 3070 serial ms | 3070 GPU/ser | 5070 GPU ms | 5070 serial ms | 5070 GPU/ser |
|---|---|---|---|---|---|---|---|---|
| outer | 16 | 8192 | 0.591 | 23.893 | 40.4x | 0.195 | 20.283 | 104.0x |
| outer | 16 | 65536 | 4.620 | 196.956 | 42.6x | 1.152 | 166.092 | 144.2x |
| outer | 64 | 8192 | 8.897 | 422.079 | 47.4x | 2.130 | 329.642 | 154.8x |
| rowsum | 16 | 8192 | 0.201 | 12.049 | 59.9x | 0.065 | 10.102 | 155.4x |
| rowsum | 16 | 65536 | 1.294 | 108.068 | 83.5x | 0.362 | 81.475 | 225.1x |
| rowsum | 64 | 8192 | 2.844 | 156.359 | 55.0x | 0.645 | 158.184 | 245.2x |

#### Symmetric eigenvalues (Jacobi): kernel wall per shape

`thr` is thread-per-matrix and exists for `n <= 16`; the shape rule
above follows from these two tables.

| n | batch | 3070 blk ms | 3070 thr ms | 3070 CPU-par ms | 3070 serial ms | 5070 blk ms | 5070 thr ms | 5070 CPU-par ms | 5070 serial ms |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1024 | 0.580 | 0.254 | 0.318 | 1.661 | 0.447 | 0.169 | 0.211 | 1.280 |
| 4 | 2048 | 1.097 | 0.255 | 0.622 | 3.354 | 0.874 | 0.172 | 0.307 | 3.240 |
| 4 | 4096 | 2.192 | 0.260 | 0.917 | 6.665 | 1.714 | 0.174 | 0.466 | 5.123 |
| 4 | 8192 | 4.255 | 0.265 | 1.750 | 13.518 | 3.390 | 0.177 | 0.780 | 9.597 |
| 4 | 65536 | 33.789 | 1.508 | 11.105 | 107.614 | 27.514 | 0.974 | 5.130 | 83.492 |
| 8 | 1024 | 1.522 | 1.817 | 2.126 | 10.765 | 1.187 | 1.227 | 1.077 | 8.707 |
| 8 | 2048 | 2.844 | 1.818 | 3.901 | 21.297 | 2.354 | 1.234 | 1.596 | 15.574 |
| 8 | 4096 | 5.578 | 1.818 | 5.489 | 42.266 | 4.591 | 1.236 | 2.733 | 29.856 |
| 8 | 8192 | 11.096 | 1.873 | 9.233 | 86.540 | 9.171 | 1.278 | 4.564 | 60.196 |
| 8 | 65536 | 88.205 | 21.876 | 68.520 | 700.034 | 73.413 | 7.255 | 33.739 | 518.821 |
| 16 | 1024 | 4.207 | 13.088 | 11.858 | 63.216 | 3.107 | 9.607 | 7.300 | 60.091 |
| 16 | 2048 | 8.095 | 13.216 | 20.969 | 124.859 | 6.194 | 9.559 | 10.852 | 118.699 |
| 16 | 4096 | 15.881 | 14.783 | 33.957 | 247.766 | 12.466 | 9.629 | 18.208 | 215.571 |
| 16 | 8192 | 31.637 | 18.230 | 70.985 | 536.388 | 24.172 | 9.716 | 30.440 | 465.845 |
| 16 | 65536 | 253.246 | 200.575 | 440.614 | 4119.796 | 195.561 | 87.195 | 207.875 | 3603.960 |
| 32 | 1024 | 17.363 | - | 69.621 | 393.813 | 12.199 | - | 24.166 | 368.755 |
| 32 | 2048 | 33.796 | - | 140.201 | 802.135 | 23.935 | - | 66.348 | 709.098 |
| 32 | 4096 | 66.901 | - | 245.432 | 1705.866 | 48.235 | - | 110.128 | 1361.831 |
| 32 | 8192 | 132.476 | - | 417.251 | 3253.364 | 95.223 | - | 167.224 | 2703.140 |
| 32 | 65536 | 1067.243 | - | 3208.659 | 26405.830 | 747.293 | - | 1233.424 | 21315.795 |
| 64 | 1024 | 114.629 | - | 519.050 | 3031.548 | 78.695 | - | 328.658 | 2177.959 |
| 64 | 2048 | 222.952 | - | 1161.497 | 7842.398 | 152.554 | - | 385.959 | 4473.775 |
| 64 | 4096 | 438.150 | - | 2077.091 | 15733.000 | 297.719 | - | 831.598 | 8892.447 |
| 64 | 8192 | 873.231 | - | 3982.621 | 23943.736 | 592.702 | - | 1434.014 | 17663.988 |

#### Symmetric eigenvalues (Jacobi): best shape against the CPU

| n | batch | 3070 best/CPU-par | 3070 best/serial | 5070 best/CPU-par | 5070 best/serial |
|---|---|---|---|---|---|
| 4 | 1024 | 1.3x | 6.5x | 1.2x | 7.6x |
| 4 | 2048 | 2.4x | 13.2x | 1.8x | 18.8x |
| 4 | 4096 | 3.5x | 25.6x | 2.7x | 29.4x |
| 4 | 8192 | 6.6x | 51.0x | 4.4x | 54.2x |
| 4 | 65536 | 7.4x | 71.4x | 5.3x | 85.7x |
| 8 | 1024 | 1.4x | 7.1x | 0.9x | 7.3x |
| 8 | 2048 | 2.1x | 11.7x | 1.3x | 12.6x |
| 8 | 4096 | 3.0x | 23.2x | 2.2x | 24.2x |
| 8 | 8192 | 4.9x | 46.2x | 3.6x | 47.1x |
| 8 | 65536 | 3.1x | 32.0x | 4.7x | 71.5x |
| 16 | 1024 | 2.8x | 15.0x | 2.3x | 19.3x |
| 16 | 2048 | 2.6x | 15.4x | 1.8x | 19.2x |
| 16 | 4096 | 2.3x | 16.8x | 1.9x | 22.4x |
| 16 | 8192 | 3.9x | 29.4x | 3.1x | 47.9x |
| 16 | 65536 | 2.2x | 20.5x | 2.4x | 41.3x |
| 32 | 1024 | 4.0x | 22.7x | 2.0x | 30.2x |
| 32 | 2048 | 4.1x | 23.7x | 2.8x | 29.6x |
| 32 | 4096 | 3.7x | 25.5x | 2.3x | 28.2x |
| 32 | 8192 | 3.1x | 24.6x | 1.8x | 28.4x |
| 32 | 65536 | 3.0x | 24.7x | 1.7x | 28.5x |
| 64 | 1024 | 4.5x | 26.4x | 4.2x | 27.7x |
| 64 | 2048 | 5.2x | 35.2x | 2.5x | 29.3x |
| 64 | 4096 | 4.7x | 35.9x | 2.8x | 29.9x |
| 64 | 8192 | 4.6x | 27.4x | 2.4x | 29.8x |

#### Singular values (one-sided Jacobi): kernel wall per shape

| n | batch | 3070 blk ms | 3070 thr ms | 3070 CPU-par ms | 3070 serial ms | 5070 blk ms | 5070 thr ms | 5070 CPU-par ms | 5070 serial ms |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1024 | 0.484 | 0.284 | 0.363 | 1.225 | 0.436 | 0.237 | 0.156 | 0.798 |
| 4 | 2048 | 0.831 | 0.281 | 0.488 | 2.785 | 0.915 | 0.268 | 0.252 | 1.537 |
| 4 | 4096 | 1.660 | 0.322 | 0.747 | 4.889 | 1.638 | 0.441 | 0.396 | 3.093 |
| 4 | 8192 | 3.181 | 0.414 | 1.268 | 9.946 | 2.813 | 0.302 | 0.600 | 8.037 |
| 4 | 65536 | 25.505 | 2.613 | 7.660 | 79.856 | 22.174 | 1.287 | 3.439 | 55.685 |
| 8 | 1024 | 1.460 | 1.775 | 1.494 | 8.057 | 1.319 | 1.193 | 0.826 | 6.094 |
| 8 | 2048 | 2.802 | 1.852 | 3.622 | 20.127 | 2.641 | 1.247 | 1.234 | 11.413 |
| 8 | 4096 | 5.438 | 2.077 | 4.515 | 32.887 | 5.038 | 1.267 | 1.992 | 24.055 |
| 8 | 8192 | 10.976 | 2.464 | 7.498 | 68.278 | 10.123 | 1.409 | 3.374 | 46.795 |
| 8 | 65536 | 86.557 | 19.096 | 59.530 | 531.857 | 79.913 | 10.005 | 22.321 | 382.778 |
| 16 | 1024 | 4.726 | 12.408 | 14.275 | 72.461 | 4.427 | 8.139 | 5.857 | 42.755 |
| 16 | 2048 | 9.328 | 12.868 | 19.686 | 121.088 | 8.684 | 8.547 | 9.180 | 87.569 |
| 16 | 4096 | 18.385 | 13.459 | 31.095 | 245.489 | 17.373 | 8.868 | 14.724 | 161.409 |
| 16 | 8192 | 35.871 | 15.060 | 55.793 | 509.471 | 34.695 | 9.311 | 23.862 | 322.220 |
| 16 | 65536 | 282.562 | 167.832 | 397.171 | 3911.153 | 271.411 | 63.695 | 166.895 | 2652.303 |
| 32 | 1024 | 19.582 | - | 64.942 | 551.482 | 17.546 | - | 30.628 | 322.124 |
| 32 | 2048 | 37.260 | - | 130.764 | 1041.295 | 34.782 | - | 64.407 | 624.924 |
| 32 | 4096 | 72.285 | - | 260.454 | 2118.753 | 68.028 | - | 106.918 | 1263.087 |
| 32 | 8192 | 143.887 | - | 462.471 | 4022.909 | 135.139 | - | 155.750 | 2591.528 |
| 32 | 65536 | 1139.699 | - | 3254.142 | 31962.391 | 1077.113 | - | 1192.306 | 20684.323 |
| 64 | 1024 | 94.936 | - | 627.888 | 4162.211 | 84.652 | - | 255.002 | 2852.299 |
| 64 | 2048 | 183.646 | - | 1378.476 | 8213.834 | 165.113 | - | 610.039 | 5716.008 |
| 64 | 4096 | 363.226 | - | 2478.876 | 16751.929 | 325.720 | - | 990.749 | 11398.338 |
| 64 | 8192 | 722.890 | - | 4652.798 | 34358.128 | 642.530 | - | 1686.235 | 22817.286 |

#### Singular values (one-sided Jacobi): best shape against the CPU

| n | batch | 3070 best/CPU-par | 3070 best/serial | 5070 best/CPU-par | 5070 best/serial |
|---|---|---|---|---|---|
| 4 | 1024 | 1.3x | 4.3x | 0.7x | 3.4x |
| 4 | 2048 | 1.7x | 9.9x | 0.9x | 5.7x |
| 4 | 4096 | 2.3x | 15.2x | 0.9x | 7.0x |
| 4 | 8192 | 3.1x | 24.0x | 2.0x | 26.6x |
| 4 | 65536 | 2.9x | 30.6x | 2.7x | 43.3x |
| 8 | 1024 | 1.0x | 5.5x | 0.7x | 5.1x |
| 8 | 2048 | 2.0x | 10.9x | 1.0x | 9.2x |
| 8 | 4096 | 2.2x | 15.8x | 1.6x | 19.0x |
| 8 | 8192 | 3.0x | 27.7x | 2.4x | 33.2x |
| 8 | 65536 | 3.1x | 27.9x | 2.2x | 38.3x |
| 16 | 1024 | 3.0x | 15.3x | 1.3x | 9.7x |
| 16 | 2048 | 2.1x | 13.0x | 1.1x | 10.2x |
| 16 | 4096 | 2.3x | 18.2x | 1.7x | 18.2x |
| 16 | 8192 | 3.7x | 33.8x | 2.6x | 34.6x |
| 16 | 65536 | 2.4x | 23.3x | 2.6x | 41.6x |
| 32 | 1024 | 3.3x | 28.2x | 1.7x | 18.4x |
| 32 | 2048 | 3.5x | 27.9x | 1.9x | 18.0x |
| 32 | 4096 | 3.6x | 29.3x | 1.6x | 18.6x |
| 32 | 8192 | 3.2x | 28.0x | 1.2x | 19.2x |
| 32 | 65536 | 2.9x | 28.0x | 1.1x | 19.2x |
| 64 | 1024 | 6.6x | 43.8x | 3.0x | 33.7x |
| 64 | 2048 | 7.5x | 44.7x | 3.7x | 34.6x |
| 64 | 4096 | 6.8x | 46.1x | 3.0x | 35.0x |
| 64 | 8192 | 6.4x | 47.5x | 2.6x | 35.5x |

#### GEMM: native f64 kernel vs the Ozaki scheme

Commit `03ed261`, both hosts idle on the GPU; kernel wall per call
after the same ramp, operands resident, the Ozaki workspace pinned
once per shape.

| n | batch | 3070 native ms | 3070 Ozaki ms | 3070 native/Ozaki | 5070 native ms | 5070 Ozaki ms | 5070 native/Ozaki |
|---|---|---|---|---|---|---|---|
| 64 | 1024 | 1.723 | 3.081 | 0.56x | 1.056 | 2.126 | 0.50x |
| 64 | 8192 | 12.616 | 23.785 | 0.53x | 8.122 | 16.328 | 0.50x |
| 256 | 1 | 0.133 | 0.119 | 1.12x | 0.085 | 0.170 | 0.50x |
| 512 | 1 | 0.828 | 0.346 | 2.39x | 0.911 | 0.407 | 2.24x |
| 1024 | 1 | 6.316 | 1.624 | 3.89x | 4.123 | 1.108 | 3.72x |
| 2048 | 1 | 49.594 | 14.751 | 3.36x | 31.813 | 7.120 | 4.47x |

#### Symmetric eigenvalues: Jacobi (blk) vs tridiagonalization + bisection

Commit `03ed261`, eigenvalues only. The 5070's CPU column is omitted:
that host's CPU was saturated by other work during the run, so only
its GPU-to-GPU ratio is reported.

| n | batch | 3070 Jacobi ms | 3070 bisection ms | 3070 CPU-par ms | 3070 Jacobi/bisection | 5070 Jacobi ms | 5070 bisection ms | 5070 Jacobi/bisection |
|---|---|---|---|---|---|---|---|---|
| 32 | 1024 | 17.658 | 10.143 | 77.978 | 1.74x | 12.009 | 8.472 | 1.42x |
| 32 | 8192 | 134.069 | 75.448 | 469.958 | 1.78x | 93.507 | 65.924 | 1.42x |
| 64 | 1024 | 115.381 | 28.828 | 550.605 | 4.00x | 76.411 | 19.114 | 4.00x |
| 64 | 8192 | 882.849 | 216.487 | 4303.191 | 4.08x | 591.616 | 148.709 | 3.98x |

#### Singular values: Jacobi (blk) vs bidiagonalization + bisection

Same run, singular values only, square `m = n`.

| n | batch | 3070 Jacobi ms | 3070 bisection ms | 3070 CPU-par ms | 3070 Jacobi/bisection | 5070 Jacobi ms | 5070 bisection ms | 5070 Jacobi/bisection |
|---|---|---|---|---|---|---|---|---|
| 32 | 1024 | 19.496 | 27.960 | 79.311 | 0.70x | 17.990 | 22.969 | 0.78x |
| 32 | 8192 | 148.754 | 209.385 | 506.507 | 0.71x | 133.232 | 180.310 | 0.74x |
| 64 | 1024 | 97.521 | 92.549 | 905.330 | 1.05x | 83.814 | 60.748 | 1.38x |
| 64 | 8192 | 733.331 | 670.885 | 4577.038 | 1.09x | 634.704 | 471.169 | 1.35x |

## User opcodes (a programmable doorbell op)

`GpuPeerConfig::user_ops_cuda` carries a caller-authored CUDA device
function - `flynnel_user_op(op, block, count, payload, team_rank,
team_size)` - that NVRTC
composes with the poller at init. Ops at or above `OP_USER_BASE`
dispatch through it by doorbell, so the GPU is a programmable peer,
not a fixed-function one. `submit_user(op, handle, args)` sends one;
with a handle it rides the handle's lane (ordered with its other
tasks) and the hook receives the resident block, without one it
round-robins and the hook receives a null block. The precompiled-PTX
build (no NVRTC needed) rejects user ops cleanly. These are
block-cooperative like every doorbell op; a large user op belongs on
`launch_wide` instead.

## Zero-synchronization prefetch

`pin_prefetch(data)` uploads a block WITHOUT waiting and returns the
handle plus the upload ticket. Lane FIFO order is the dependency
order, so any task submitted on that handle afterwards is ordered
after the upload with no fence, no event, no wait. The scheduler
front-loads a working set the way a CPU prefetcher front-loads a
cache line - fire and forget, correctness by construction. Reap the
upload ticket first among the lane's tickets.

## Multiple GPUs (peer groups and migration)

`PeerGroup::init(configs)` runs one peer per device under a single
handle namespace, or the members are used individually - the same
type does both. Each peer keeps its OWN region, calibration, and
resident pool, so a group spanning a PCIe card and a coherent-link
card carries both calibrations. `pin` places on the peer with the
most free blocks; `submit_resident` routes to the owning peer;
`migrate(handle, to_peer)` moves a resident block between peers
through the host bridge (fetch from the source, pin into the
destination). Validated on the local single GPU (every group code
path, ordinal 0 twice) and packaged for a true two-silicon run on
Kaggle T4x2.

## One model for residence and execution

`hybrid_auto_resident(plan, peer, mirror, op, cpu_impl)` routes a
repeated step through the SAME per-call-site placement EWMAs the
CPU/GPU hybrid uses, over a `MirrorBuf` whose bytes live on the host,
on the device, or both. Any transfer a placement flip needs runs
INSIDE the timed section, so the one model prices data movement and
execution together and sticky residence emerges from measurement
rather than from a residency planner. E2E:
[`examples/gpu_peer_hybrid_demo.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/gpu_peer_hybrid_demo.rs).

## Examples

- `examples/gpu_peer_demo.rs` - calibration printout, 8192 verified
  ADD1 blocks serial and through a 32-deep submission window
  (latency-hiding demonstrated by the amortized per-message cost),
  GPU-side reduction, timed-lock cycles.
- `examples/gpu_peer_tandem_demo.rs` - MIMT tandem: CPU
  `for_each_chunk` half + GPU peer half of one buffer, split share
  learned from measured per-side throughput, full-buffer verification
  every round.

- `examples/gpu_peer_resident_demo.rs` - pin-once/task-many: four
  resident 64 KB blocks, 8,000 params-only tasks with full ordered
  verification, the shipped-every-task baseline under identical
  lanes/window, and a resident GPU reduction.
- `examples/gpu_peer_user_ops_demo.rs` - a caller-authored kernel run
  as a doorbell opcode against prefetched resident data, ordered by
  the transport with zero waits.
- `examples/gpu_peer_group_demo.rs` - one handle namespace over N
  peers: unified placement, routing, and host-bridge migration with
  byte-exact data survival.
- `examples/gpu_wide_op_demo.rs` - a 16,384-pixel resident blur
  launched grid=1 (one SM) vs full grid (all SMs), verified against a
  CPU blur, showing the wide-launch lift.
- `examples/gpu_l2_persist_demo.rs` - L2 persistence: reserve L2, pin
  a hot set, and measure the non-monotonic lift (see the
  [Cache-Residency reference](Cache-Residency-Reference.md)).

All exit cleanly with a message when no NVIDIA device is present.

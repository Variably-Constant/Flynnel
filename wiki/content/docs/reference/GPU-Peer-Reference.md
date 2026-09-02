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
`launch_wide`. E2E: [`examples/gpu_wide_op_demo.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/gpu_wide_op_demo.rs).

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
[`examples/gpu_wide_batch_demo.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/gpu_wide_batch_demo.rs).

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
[`examples/gpu_peer_hybrid_demo.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/gpu_peer_hybrid_demo.rs).

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

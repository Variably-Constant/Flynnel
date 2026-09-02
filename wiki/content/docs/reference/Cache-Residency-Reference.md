---
title: Cache-Residency Reference
weight: 9
---

One thesis, three silicons: reserve fast on-chip memory for a hot working set and keep it resident, so a noisy co-runner cannot evict it. The GPU-peer VRAM pool already does this at the HBM level by owning blocks by index. These levers push residency one level deeper - into L2 on the GPU, into L3 ways on the CPU, into VMEM on the TPU - and each is capability-gated and measured on the running host, never assumed. Where a device does not honor the lever, the capability reads unsupported and the lever is refused. No panics.

## The honest split across silicon

Can you `mmap` a cache? Almost never - a cache is a content-addressed mirror of DRAM with no address of its own, so there is nothing for a page table to point at. The one true exception is Intel Cache Pseudo-Locking, which exposes a pinned region as an mmap-able character device; it is Intel-only and shrinking on newer parts. AMD never shipped it - through Zen 5 and Turin, AMD's answer is L3 CAT partitioning (reserve ways) plus transparent 3D V-Cache (make the cache huge, keep it automatic). GPUs give a middle path: reserve part of L2 and mark an address window persisting. TPUs invert the question entirely - the fast on-chip memory *is* a software-managed scratchpad, so you never map a cache because there is no cache.

Each lever below matches one of those realities.

## GPU: L2 persistence (`gpu_peer::l2_persist`)

On compute capability 8.0 and up, the driver sets aside part of L2 and lets a stream mark a device-address window as persisting, so its lines are preferentially retained while streaming traffic cannot evict them.

- `L2Capability::query(ctx)` reports the set-aside ceiling and the access-window ceiling; both read zero where the feature is absent.
- `L2Persist::reserve(ctx, bytes)` claims the set-aside (clamped to the ceiling); `pin_window(stream, dev_ptr, num_bytes, hit_ratio)` marks the window; `clear_window` releases it.
- `benchmark(ctx, hot, pol, iters, runs)` runs a fair A/B - the identical hammer kernel over identical data, timed with the hot set pinned versus streaming, min-of-runs.

The result on an RTX 3070 is instructive and non-monotonic. Pinning wins under moderate contention (1.12x with a 16 MiB polluter) but goes *negative* when the set-aside starves the co-runner (0.89x with a 6 MiB polluter). So the lever is measured and gated, not always-on - the same rule the Fischer and sys-atomics probes follow. Its ceiling is the device's L2 size; it grows on server parts. E2E: [`examples/gpu_l2_persist_demo.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/gpu_l2_persist_demo.rs).

## CPU: L3 way reservation (`sched::cat`)

Linux `resctrl` exposes L3 cache-way allocation. You carve a class of service with a capacity bitmask - one bit per way - and bind a process to it, reserving those ways. AMD supports this as Platform QoS from Zen2 on; Intel as RDT. It reserves ways, it does not map bytes.

- `CatCapability::detect()` reads `/sys/fs/resctrl/info/L3` for way count, class count, and domains; it reports unsupported on any non-Linux host or where resctrl is not mounted.
- `L3Reservation::reserve_ways(name, first_way, num_ways)` builds a contiguous mask, creates a resctrl group, applies the mask to every L3 domain, and binds the current process. Dropping it removes the group, re-homing the process to the default. `schemata()` reads the reservation back.

The reservation path needs a mounted resctrl and root; where either is missing the lever returns `Unsupported` cleanly. E2E: [`examples/cat_demo.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/cat_demo.rs), which reserves half the L3 ways and reads the schemata back on a Zen2+/RDT Linux host, and reports the capability without acting anywhere else.

## TPU: VMEM/HBM residency (`tpu/tpu_residency.ipynb`)

A TPU has no cache to reserve. Its on-chip SRAM is a scratchpad the XLA compiler stages into ahead of time, so residency here means keeping arrays in device HBM across `jit` calls and letting XLA place them in VMEM - the same pin-once, reuse-many discipline the VRAM pool uses, expressed through the compiler instead of a raw pointer.

The notebook uses the authoritative JAX placement API: `jax.device_put` with `jax.sharding.SingleDeviceSharding(dev, memory_kind='device')`, `array.sharding.memory_kind` to confirm placement, `jit` for the fused step, and `compiled.memory_analysis()` for the VMEM-staging footprint. It measures device-resident reuse against re-transferring the working set from host every call, with a placement-neutral correctness check (both paths run the identical step; only residence differs). The harness was verified on CPU-JAX before packaging; the v6e run on Colab is the real measurement.

## Why they are gated, not default

Every one of these can backfire. L2 persistence starves a co-runner when the set-aside is too large. L3 reservation shrinks the pool the rest of the process draws on. VMEM staging competes with the fusion's own scratch. So the substrate measures each on the host it lands on and enables it only where it wins - residency is a decision the scheduler makes from evidence, not a switch left on.

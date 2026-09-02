---
title: Shared-Memory Worker Backend
weight: 6
---

A `DispatchBackend` that puts the work into a different OS process. The transport is a memory-mapped Chase-Lev work-stealing deque paired with a memory-mapped latch arena; the wire format carries `(closure_id, args)`, not closure code. Per-call cost sits between 342 ns (pinned to SMT siblings) and 881 ns (pinned cross-CCX) on Zen+, roughly 25-60x faster than pipe IPC and faster than `std::sync::mpsc` in every measured pinning tier.

Gated by the `shared-memory-worker-reference` Cargo feature.

## When to reach for it

The in-process [CPU backend](../reference/Backend-System.md) is always the first choice. Its atomics-only dispatch costs about 17 ns on Zen+; nothing kernel-mediated will get close. Reach for the shared-memory backend only when crossing the process boundary is what you actually need.

Three cases the in-process pool cannot serve:

1. **Sandboxed worker farms.** Each peer runs in its own address space, so a kernel that segfaults, leaks, or loops cannot corrupt the originator.
2. **Cross-language interop.** Anything that can `mmap` a file (Python, Go, C, Zig, Node-via-FFI) can attach to the same deque + arena and serve registered handlers.
3. **Process-isolated runtimes.** Kernels that depend on incompatible runtime state (allocator configs, signal handlers, license-isolated dynamic libraries) each live in their own process while still sharing one dispatch surface.

What does NOT belong here: the in-process scheduler hot path. The MMF backend's per-call cost is one to two orders of magnitude above the work-stealing pool's own dispatch cost. Use this backend when cross-process is a hard requirement, not as a substitute.

## How it works

Three primitives compose:

- **`shared_mem::chase_lev_mmf::MmfChaseLevDeque`** is a fixed-capacity Chase-Lev work-stealing deque over an mmap. The originator (the process that created the file) owner-pushes one end with a single Release-store on `bottom`; any number of thieves (in any process that has the MMF open) CAS the other end on `top` to claim a slot. One 64-byte cache-line slot per deque position. The same byte layout serves cross-thread, cross-process, and disk-persistent deployments unchanged because the OS page cache aliases the MMF onto the same physical pages across address spaces, and CPU atomic operations apply to the physical line regardless of which page table mapped it.
- **`shared_mem::latch_mmf::MmfLatchArena`** is a bump-allocated arena of 64-byte latch cells. Each cell carries one `AtomicU8` state byte plus 56 bytes of inline result payload. The originator allocates a fresh cell, stamps its byte offset into the deque slot's `latch_offset` field, then polls the cell's `state` byte with `Acquire` ordering until the peer publishes; the peer Release-stores `SET` after copying its reply bytes into the cell.
- **`shared_mem::pass_registry`** is a process-local `closure_id -> handler` table. Each peer calls `pass_registry::register(id, handler)` at startup. The wire format carries `Pass { closure_id, args }` because Rust closures cannot safely cross address spaces (function pointers are not position-stable, and captured environment can hold non-portable types). Identifiers are deterministic via `hash_name(name)` so peers agree on the wire id without coordination.
- **`shared_mem::SharedMemoryChaseLevBackend`** wires the three together into the `DispatchBackend` impl. `register_kernel(name, _)` returns a handle whose `u64` is the deterministic `hash_name(name)`. `dispatch_kernel(handle, _, args)` encodes the args slice, allocates a latch cell, and pushes a `RemoteJobSlot { closure_id, args_inline, latch_offset }` onto the Chase-Lev deque. Peers drain via `drain_one()` (which calls `steal()` on the deque), execute the resolved handler, and publish the reply bytes into the slot's latch cell.

The trait surface stays uniform:

| `DispatchBackend` method     | Shared-memory worker semantics                                                          |
|------------------------------|-----------------------------------------------------------------------------------------|
| `register_kernel(name, _)`   | Returns `KernelHandle(hash_name(name) as u64)`. Peer must have already registered the same id |
| `dispatch_kernel(...)`       | Encodes args, allocates a latch cell, pushes a slot onto the deque. Non-blocking         |
| `dispatch_one(...)`          | **Panics.** Rust closures cannot safely cross processes; use `register_kernel` instead  |
| `dispatch_parallel_for(...)` | **No-op.** Pool fan-out happens by attaching N peer processes to the same deque         |

The asymmetric handling of the two unsupported methods is deliberate. `dispatch_one` panics because silently dropping the work would be the worse failure mode. `dispatch_parallel_for` is a no-op because the router walks every backend in turn and a panic there would crash unrelated paths.

## Per-call cost

Measured on Ryzen 7 2700 Zen+ (16 logical cores, 2 CCX of 8 logical each via the AmdCpuidCcx detector) via `cargo bench --features shared-memory-worker-reference --bench chase_lev_mmf`:

| Dispatch mechanism                                | Per-call latency (median) | Ratio vs in-process |
|---------------------------------------------------|---------------------------|---------------------|
| `flynnel::flat::join` (in-process scheduler)      | **16.9 ns**               | 1x                  |
| Chase-Lev MMF backend, SMT-siblings pinned        | **342 ns**                | 20x slower          |
| Chase-Lev MMF backend, intra-CCX pinned           | **424 ns**                | 25x slower          |
| Chase-Lev MMF backend, unpinned                   | **533 ns**                | 32x slower          |
| Chase-Lev MMF backend, cross-CCX pinned           | **881 ns**                | 52x slower          |
| `std::sync::mpsc::sync_channel` round-trip        | **909.5 ns**              | 54x slower          |

Two things to read out of that table.

First, **Chase-Lev wins in every tier vs `std::sync::mpsc`**. The shared-memory path pays for `KernelArg`-blob encode/decode and a 48-byte payload memcpy through the mapped page on every call; the mpsc path moves a single `u64` and nothing else. The asymmetric Chase-Lev protocol still wins because the owner's hot path is one Release-store on `bottom` (no atomic, no contention) while `std::sync::mpsc` parks on a `Mutex<Condvar>`. Pipe IPC sits at roughly 20-50 us on the same host, so the shared-memory backend is 25-60x faster than pipes for the same per-call shape and 100-500x faster than the JSON-over-pipe transport the TPU JAX bridge uses.

Second, **the coherence tier the dispatcher + drainer land on matters**. SMT-siblings pinning (two threads on the same physical core, shared L1d) hits 342 ns because no cache-line bouncing happens. Intra-CCX pinning (different physical cores in the same CCX, shared L3) hits 424 ns. Cross-CCX pinning (different dies) hits 881 ns. The protocol's critical path is one `bottom` → `top` coherence bounce per round-trip; the bounce-latency floor at each tier sets the per-call number. The substrate atomics themselves are 16 ns same-thread.

## Round-trip example

```rust
use flynnel::backend::shared_mem::{
    SharedMemoryChaseLevBackend, hash_name, register,
};
use flynnel::backend::{DispatchBackend, KernelArg};

// Originator creates the deque + latch arena.
let backend = SharedMemoryChaseLevBackend::create(
    /* backend_id */ 0,
    "/tmp/flynnel-deque.bin",
    "/tmp/flynnel-latches.bin",
    /* deque_capacity */ 64,
    /* latch_capacity */ 128,
)?;

// Peer-side registration. In production this runs in the peer's own
// binary at startup; shown here in one process for brevity.
let id = hash_name("flynnel.demo.doubler");
register(id, |args| {
    // args is the encoded KernelArg slice: tag byte + LE bytes per scalar.
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&args[1..9]);
    let n = u64::from_le_bytes(buf);
    Ok((n * 2).to_le_bytes().to_vec())
});

// Dispatch.
let handle = backend.register_kernel("flynnel.demo.doubler", &[])?;
backend.dispatch_kernel(handle, 1, &[KernelArg::U64(21)])?;

// Peer side runs in a loop, stealing slots and publishing results.
backend.drain_one()?;
```

A runnable cross-process example is at [`examples/chase_lev_mmf_steal.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/chase_lev_mmf_steal.rs). The originator spawns the same binary as a child process pointing at the same deque + arena files, dispatches 100 add-jobs, and verifies all 100 results round-trip bit-exact through cross-process Chase-Lev steal.

## Crossing the process boundary

Both processes call `SharedMemoryChaseLevBackend::create` (originator) or `::open` (peer) against the same two files. The OS page cache aliases the mappings onto the same physical memory, and the per-atomic ordering pairs synchronize as if both sides were threads of one process. The kernel is not in the path; the MMU just translates the same physical pages through each process's own page table.

The one thing peers DO have to coordinate is the `pass_registry::register(id, handler)` calls at startup. The originator's `register_kernel(name, _)` only mints the handle; the peer is responsible for installing the actual handler under `hash_name(name)`. If they do not match, the peer's `drain_one()` writes a `PassError::UnknownClosureId(id)` diagnostic into the latch cell's `ERR` state. That is a clean named failure, not a deadlock.

## What this backend is NOT

**Not a network transport.** All peers must share a kernel page cache, which means same-host. Cross-host dispatch needs a different transport; the same `(closure_id, args)` wire shape ports cleanly to QUIC, but the backend module here is host-local.

**Not a replacement for `CpuBackend`.** Per-call latency is 20-52x higher than the in-process pool. The trade you get for that cost is process isolation, not throughput.

**Not a closure transport.** `dispatch_one(Box<dyn FnOnce>)` panics because Rust closures' function pointers are not position-stable across address spaces. Use `register_kernel` + `dispatch_kernel` for every cross-process path.

## References

- Chase, David and Lev, Yossi. [Dynamic Circular Work-Stealing Deque](https://dl.acm.org/doi/10.1145/1073970.1073974). 17th Annual ACM Symposium on Parallelism in Algorithms and Architectures (SPAA '05), pp. 21-28. The asymmetric one-owner-many-thieves protocol the MMF deque ports to a cross-process memory map: owner push / pop is one Release-store on `bottom`; only thieves CAS `top`.
- IEEE and The Open Group. [POSIX.1-2024: `mmap`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/mmap.html). IEEE Std 1003.1-2024 / The Open Group Base Specifications Issue 8. Defines `MAP_SHARED` semantics that make CPU atomic operations on the mapped region visible across processes sharing the same backing object.
- Moritz, Philipp et al. [Ray: A Distributed Framework for Emerging AI Applications](https://www.usenix.org/conference/osdi18/presentation/moritz). 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI '18), pp. 561-577. The actor-id wire pattern (closure registered locally, identifier travels over the wire) that `pass_registry` recreates for cross-process dispatch.
- [memmap2 crate](https://crates.io/crates/memmap2) (v0.9.10). Pure-Rust portable wrapper over POSIX `mmap` and Windows `CreateFileMapping`; the dep that backs `MmfChaseLevDeque` and `MmfLatchArena`.

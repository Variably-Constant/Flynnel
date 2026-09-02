---
title: How To Use The Shared-Memory Worker Backend
weight: 6
---

End-to-end recipe for dispatching work from an originator process into one or more peer worker processes over a memory-mapped Chase-Lev work-stealing deque + memory-mapped latch arena. Per-call cost lands in the **342-881 ns** range on Zen+ R7 2700 depending on which coherence tier the dispatcher + drainer pair land on.

This page is the setup-and-run guide; the architecture, protocol, and cost model live at [Shared-Memory Worker Backend](../explanation/Shared-Memory-Worker-Backend.md).

## Prerequisites

- Cargo feature `shared-memory-worker-reference` enabled.
- The originator and every peer process must be the same crate version (they share an MMF byte layout).
- Same host. The kernel page cache is what aliases the MMF onto the same physical pages across address spaces.

```toml
[dependencies]
flynnel = { version = "0.1", features = ["shared-memory-worker-reference"] }
```

## Step 1 - decide your handler protocol

Each cross-process call carries `(closure_id: u32, args: [u8; 48])`. The peer resolves `closure_id` to a registered handler that takes the raw arg bytes and returns a raw reply blob.

Pick stable string names for every handler. The `closure_id` derives deterministically via FNV-1a hash so peers do not need to coordinate numeric ids:

```rust
use flynnel::backend::shared_mem::hash_name;

const ADD_ID: u32 = 0; // populated at runtime; const slot for readability
let add_id = hash_name("flynnel.demo.add");
```

The wire shape is up to you; this example uses two `u32` operands serialized little-endian:

```rust
fn encode_add(a: u32, b: u32) -> [u8; 8] {
    let mut buf = [0u8; 8];
    buf[..4].copy_from_slice(&a.to_le_bytes());
    buf[4..].copy_from_slice(&b.to_le_bytes());
    buf
}

fn decode_add(args: &[u8]) -> (u32, u32) {
    let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
    let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
    (a, b)
}
```

## Step 2 - originator creates the MMF files

The originator process calls `SharedMemoryChaseLevBackend::create` against two paths it owns. The first file is the work deque; the second is the latch arena where peer-published results land.

```rust
use flynnel::backend::shared_mem::SharedMemoryChaseLevBackend;

let backend = SharedMemoryChaseLevBackend::create(
    /* backend_id     */ 0,
    /* deque path     */ "/tmp/flynnel-demo-deque.bin",
    /* latches path   */ "/tmp/flynnel-demo-latches.bin",
    /* deque capacity */ 64,
    /* latch capacity */ 128,
)?;
```

Capacities round up to the next power of two. Deque capacity is the maximum in-flight dispatches before the originator must back off or drain locally; latch capacity is the wrap-around horizon for the bump allocator.

## Step 3 - peer processes attach

Each peer process opens the same two files. The originator owns the deque (only the originating process pushes / pops `bottom`); peers are thieves (any number of processes can `steal` from `top`).

```rust
use flynnel::backend::shared_mem::{
    SharedMemoryChaseLevBackend, hash_name, register,
};

let backend = SharedMemoryChaseLevBackend::open(
    /* backend_id   */ 0,
    /* deque path   */ "/tmp/flynnel-demo-deque.bin",
    /* latches path */ "/tmp/flynnel-demo-latches.bin",
)?;
```

Before the peer's drain loop starts, it MUST register every handler the originator will dispatch:

```rust
register(hash_name("flynnel.demo.add"), |args| {
    let (a, b) = decode_add(args);
    Ok((a + b).to_le_bytes().to_vec())
});
```

## Step 4 - peer drain loop

The peer steals slots, executes the handler, and publishes the reply. `drain_one()` does all three in one call:

```rust
loop {
    match backend.drain_one()? {
        Some(()) => { /* one slot processed */ }
        None => std::hint::spin_loop(),
    }
    // graceful shutdown: break on a signal / sentinel slot / etc.
}
```

The peer never sees the originator's calling code. It only ever resolves `closure_id` via its local `pass_registry` and decodes the args bytes the handler expects.

## Step 5 - originator dispatch + wait

The originator builds the args blob, dispatches, and waits on the returned handle:

```rust
use flynnel::backend::KernelArg;
use flynnel::backend::DispatchBackend;

let handle = backend.register_kernel("flynnel.demo.add", &[])?;

backend.dispatch_kernel(handle, 1, &[KernelArg::U32(3), KernelArg::U32(4)])?;
```

For replies, use the typed Marshal path:

```rust
let args = encode_add(3, 4);
let dispatch = backend.dispatch_marshal(hash_name("flynnel.demo.add"), &args)?;
let result = backend.wait_handle(dispatch, 1024)?;
let sum = u32::from_le_bytes(result.expect("ok")[..4].try_into().unwrap());
assert_eq!(sum, 7);
```

`wait_handle` spins on the latch cell's `state` byte with `Acquire` ordering up to the iter budget, then yields. `poll_handle` is the non-blocking equivalent for callers that need to poll many in-flight dispatches.

## What can go wrong

- **Unregistered handler.** Peer drains a slot whose `closure_id` is not in its local `pass_registry`. The peer publishes a `PassError::UnknownClosureId(id)` diagnostic into the latch cell (state = `ERR`); the originator's `wait_handle` returns `Err(diagnostic)` instead of `Ok(bytes)`. No deadlock; the failure is named.
- **Args don't fit.** The deque slot's inline-args payload is 48 bytes. Larger payloads need a separate transport (typed allocator + device pointer).
- **Deque full.** The deque has fixed capacity. `dispatch_kernel` returns a `BackendError::Launch("chase-lev push failed: Full")` when the originator is faster than every peer combined. The originator can spin, back off, or drain locally via the owner-side pop.
- **Peer dies.** The originator's `wait_handle` will spin forever on an unpublished latch cell. Production code wires a watchdog (the deque header carries an `epoch` field the owner advances on `close_owner()`); peers and originators can check it.

## Runnable example

[`examples/chase_lev_mmf_steal.rs`](https://github.com/markusmcnugen/flynnel/blob/main/examples/chase_lev_mmf_steal.rs) ships an end-to-end cross-process demo: the parent process creates the deque + arena, spawns the same binary as a child process pointing at the same files, dispatches 100 add-jobs, and verifies all 100 results round-trip bit-exact through the cross-process steal path.

```sh
cargo run --release --features shared-memory-worker-reference --example chase_lev_mmf_steal
```

## See also

- [Shared-Memory Worker Backend](../explanation/Shared-Memory-Worker-Backend.md) - architecture, per-tier latency matrix, protocol details.
- [Backend System](../reference/Backend-System.md) - the `DispatchBackend` trait the shared-memory backend implements.

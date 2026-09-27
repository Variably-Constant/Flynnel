//! Reference WebAssembly backend built on wasmtime. Compiled only
//! when the `wasm-reference` Cargo feature is enabled.
//!
//! One wasmtime `Engine` is shared across kernels; each
//! `register_kernel` compiles the `.wasm` bytes, instantiates a
//! fresh per-kernel `Store<()>` with empty host imports, and looks
//! up the named typed export. `dispatch_kernel` translates the
//! `KernelArg` slice to `wasmtime::Val` slots and calls it; the
//! `count` parameter is ignored (WASM kernels are single-threaded
//! scalar). `dispatch_one` uses the same persistent-worker-thread
//! shape as the CUDA backend. `dispatch_parallel_for` is host-side
//! fan-out through [`crate::sched::par_iter::for_each_chunk`] over
//! a Rust closure, not a WASM kernel.
//!
//! Use for sandboxed portable kernels (plugins, user-supplied
//! transforms, browser / WASI targets). Compilation (cranelift
//! JIT) is paid once at `register_kernel`; per-launch cost is a
//! sandboxed function call.
//!
//! Wire types: `I32` / `I64` / `F32` / `F64` pass as scalars,
//! `U32` / `U64` as their signed bit patterns, `DevicePtr` as an
//! unvalidated `i32` linear-memory offset. `HostSlice` returns
//! [`BackendError::NotSupported`]; consumers needing host-to-WASM
//! transfer ship their own backend with an allocator API.

#![allow(clippy::missing_errors_doc)]

use std::cell::RefCell;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use wasmtime::{Engine, Instance, Module, Store, ValRaw, ValType};

use crate::backend::{
    Backend, BackendCapabilities, BackendError, DispatchBackend, KernelArg, KernelHandle,
};
use crate::sched::notify_ring::{NotifyHub, NotifySender};

/// Boxed closure shape the persistent worker thread consumes from
/// the `dispatch_one` channel.
type WorkItem = Box<dyn FnOnce() + Send + 'static>;

/// Stored per registered kernel.
///
/// `shared` is the store every thread dispatches through, and the lock
/// on it is what a wasmtime store requires rather than a choice: a
/// store is not `Sync` and the call takes a mutable one, so two
/// dispatches of one kernel take turns. `module` and `export` are what
/// a thread needs to build a store of its own instead, which is what
/// [`crate::sched::levers::wasm_local_store`] turns on.
struct KernelEntry {
    shared: Mutex<SharedInstance>,
    module: Module,
    export: String,
}

/// A store and the function resolved in it, with the function's
/// signature as a dispatch checks it. The instance keeps the module
/// memory alive while the wrapper holds the function.
struct SharedInstance {
    store: Store<()>,
    func: wasmtime::Func,
    /// Each parameter's type, read once from the function's type; a
    /// parameter no [`KernelArg`] can carry reads `None`.
    params: Box<[Option<Scalar>]>,
    /// How many results the function returns.
    results: usize,
}

impl SharedInstance {
    /// `func` in `store`, with its signature read from its type once.
    fn new(store: Store<()>, func: wasmtime::Func) -> Self {
        let ty = func.ty(&store);
        let params = ty.params().map(|p| Scalar::of(&p)).collect();
        let results = ty.results().len();
        Self {
            store,
            func,
            params,
            results,
        }
    }
}

/// The scalar types a [`KernelArg`] carries into a kernel, as a
/// dispatch checks them against the function's parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scalar {
    I32,
    I64,
    F32,
    F64,
}

impl Scalar {
    /// The parameter type `ty` as a scalar, or `None` for one no
    /// argument can carry, such as a vector or a reference.
    fn of(ty: &ValType) -> Option<Self> {
        match ty {
            ValType::I32 => Some(Self::I32),
            ValType::I64 => Some(Self::I64),
            ValType::F32 => Some(Self::F32),
            ValType::F64 => Some(Self::F64),
            _ => None,
        }
    }

    /// `arg` as the scalar it carries and its raw slot, or `None` for an
    /// argument a wasm kernel cannot take.
    fn raw(arg: &KernelArg<'_>) -> Option<(Self, ValRaw)> {
        Some(match arg {
            KernelArg::I32(v) => (Self::I32, ValRaw::i32(*v)),
            KernelArg::I64(v) => (Self::I64, ValRaw::i64(*v)),
            KernelArg::U32(v) => (Self::I32, ValRaw::i32(*v as i32)),
            KernelArg::U64(v) => (Self::I64, ValRaw::i64(*v as i64)),
            KernelArg::F32(v) => (Self::F32, ValRaw::f32(v.to_bits())),
            KernelArg::F64(v) => (Self::F64, ValRaw::f64(v.to_bits())),
            KernelArg::DevicePtr(p) => (Self::I32, ValRaw::i32(*p as i32)),
            KernelArg::HostSlice(_) | KernelArg::Buffer(_) => return None,
        })
    }
}

/// Slots a dispatch holds on the stack, one per parameter or result,
/// whichever is more; a function needing more takes them from the heap.
const INLINE_SLOTS: usize = 16;

/// Distinguishes one backend's kernels from another's in the
/// thread-local table below. Handles come from a counter per backend
/// and every backend's first kernel is handle one, so the handle alone
/// does not say whose kernel it is.
static NEXT_BACKEND_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// This thread's own store per kernel, indexed by handle, built on
    /// its first dispatch of that kernel. Nothing here is shared, so
    /// nothing here needs excluding; a `RefCell` is the whole of the
    /// discipline because only this thread can reach it.
    ///
    /// A slot carries the id of the backend whose kernel built it. The
    /// table belongs to the thread rather than to a backend, so two
    /// backends in one process both start at handle one and would
    /// otherwise share the slot: a dispatch through the second would
    /// run the first one's function, with no error anywhere because
    /// both are valid. Where the id does not match, the slot belongs to
    /// someone else and is rebuilt.
    static LOCAL_INSTANCES: RefCell<Vec<Option<(u64, SharedInstance)>>> =
        const { RefCell::new(Vec::new()) };
}

/// Run `call` on this thread's own instance for `handle`, building one
/// where the slot is empty or holds another backend's.
///
/// The slot is keyed by handle alone because that is what a dispatch
/// has, and carries the owning backend's id because a handle does not
/// say whose it is. A mismatch means another backend reached this
/// index first, and its instance is replaced rather than called.
fn with_local_instance<R>(
    owner: u64,
    handle: u64,
    build: impl FnOnce() -> Result<SharedInstance, BackendError>,
    call: impl FnOnce(&mut SharedInstance) -> Result<R, BackendError>,
) -> Result<R, BackendError> {
    let index = (handle - 1) as usize;
    LOCAL_INSTANCES.with(|cell| {
        let mut mine = cell.borrow_mut();
        if mine.len() <= index {
            mine.resize_with(index + 1, || None);
        }
        let mismatched = match &mine[index] {
            Some((held, _)) => *held != owner,
            None => true,
        };
        if mismatched {
            // This thread's first dispatch of this kernel pays the
            // instantiation. Registration paid it once for the shared
            // store; here every thread that runs the kernel pays it
            // once, which is the cost this arm is measured on.
            mine[index] = Some((owner, build()?));
        }
        let (_, instance) = mine[index]
            .as_mut()
            .expect("the slot was filled above or the call returned");
        call(instance)
    })
}

/// Which backend owns this thread's slot for `handle`, or `None` where
/// the slot is empty.
///
/// The collision this guards against is not observable from a
/// dispatch: two backends holding the same module at the same handle
/// return the same answer either way, and a store with no memory has
/// no state to tell them apart. Building a second module to make it
/// observable would mean a second `.wasm` asset and a `wat` feature
/// this crate does not take, so the bookkeeping is what is checked.
#[cfg(test)]
fn local_slot_owner(handle: u64) -> Option<u64> {
    let index = (handle - 1) as usize;
    LOCAL_INSTANCES.with(|cell| {
        let mine = cell.borrow();
        mine.get(index)
            .and_then(|slot| slot.as_ref())
            .map(|(owner, _)| *owner)
    })
}

/// Entries one block of [`KernelTable`] holds.
const KERNEL_BLOCK: usize = 64;

/// Registered kernels, indexed by handle.
///
/// Handles are handed out by one counter from one, so they are dense
/// and a table indexed by them needs no hashing and no probing. A
/// block is published once and never replaced or removed, so a reader
/// walks to its block and loads one pointer, and nothing has to be
/// reclaimed. Blocks chain rather than sitting in a fixed array, so
/// there is no count of kernels past which registering fails.
struct KernelTable {
    head: Block,
}

struct Block {
    entries: [AtomicPtr<Arc<KernelEntry>>; KERNEL_BLOCK],
    next: AtomicPtr<Block>,
}

impl Block {
    fn new() -> Self {
        Self {
            entries: [const { AtomicPtr::new(core::ptr::null_mut()) }; KERNEL_BLOCK],
            next: AtomicPtr::new(core::ptr::null_mut()),
        }
    }
}

impl KernelTable {
    fn new() -> Self {
        Self { head: Block::new() }
    }

    /// The block holding `index`, appending blocks until it exists.
    ///
    /// Two registrations racing on the same missing block both build
    /// one and one wins the exchange; the loser's is dropped before it
    /// is ever published, so no reader can have seen it.
    fn block_for(&self, index: usize) -> &Block {
        let mut block = &self.head;
        for _ in 0..(index / KERNEL_BLOCK) {
            let mut next = block.next.load(Ordering::Acquire);
            if next.is_null() {
                let fresh = Box::into_raw(Box::new(Block::new()));
                match block.next.compare_exchange(
                    core::ptr::null_mut(),
                    fresh,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => next = fresh,
                    Err(theirs) => {
                        // SAFETY: `fresh` was never published, so this
                        // thread holds the only pointer to it.
                        drop(unsafe { Box::from_raw(fresh) });
                        next = theirs;
                    }
                }
            }
            // SAFETY: a published block is never freed or replaced.
            block = unsafe { &*next };
        }
        block
    }

    /// Publish `entry` at `handle`. Handles are unique, so no slot is
    /// written twice.
    fn insert(&self, handle: u64, entry: Arc<KernelEntry>) {
        let index = (handle - 1) as usize;
        let block = self.block_for(index);
        let slot = &block.entries[index % KERNEL_BLOCK];
        slot.store(Box::into_raw(Box::new(entry)), Ordering::Release);
    }

    /// The entry at `handle`, or `None` where nothing was registered
    /// under it.
    fn get(&self, handle: u64) -> Option<Arc<KernelEntry>> {
        if handle == 0 {
            return None;
        }
        let index = (handle - 1) as usize;
        let mut block = &self.head;
        for _ in 0..(index / KERNEL_BLOCK) {
            let next = block.next.load(Ordering::Acquire);
            if next.is_null() {
                return None;
            }
            // SAFETY: a published block is never freed or replaced.
            block = unsafe { &*next };
        }
        let held = block.entries[index % KERNEL_BLOCK].load(Ordering::Acquire);
        if held.is_null() {
            return None;
        }
        // SAFETY: a published entry is never freed or replaced, and
        // the clone is of the Arc rather than of what it points at.
        Some(unsafe { (*held).clone() })
    }
}

/// wasmtime-backed reference WebAssembly backend.
pub struct WasmBackend {
    device_id: u32,
    engine: Engine,
    caps: BackendCapabilities,
    /// Tells this backend's kernels from another's in the per-thread
    /// store table, where handles alone do not, since every backend's
    /// first kernel is handle one.
    backend_id: u64,
    next_handle: AtomicU64,
    /// Registered kernels, indexed by handle. Each entry owns its
    /// own `Store`, so dispatches on different handles do not wait on
    /// each other; two on the same handle still do, because a
    /// wasmtime store is not Sync and its API takes `&mut Store`.
    kernels: KernelTable,
    /// Persistent worker thread for `dispatch_one`. Routed
    /// through a flynnel notify hub (FlynnelRing + Parker).
    worker_hub: NotifyHub<WorkItem>,
    /// Cached sender so `dispatch_one` avoids `Arc::clone` per call.
    worker_tx: NotifySender<WorkItem>,
    /// The dispatch worker, taken by `Drop`, which holds this
    /// exclusively and so needs nothing to guard it.
    worker_handle: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for WasmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmBackend")
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl WasmBackend {
    /// Initialize a new WASM backend on the primary device (id 0).
    /// Pure-Rust path: cannot fail for runtime-resolution reasons
    /// the way CUDA can. Returns `BackendError::DeviceUnavailable`
    /// only if the OS rejects the worker-thread spawn (out of
    /// thread-id resources, etc.).
    pub fn new() -> Result<Self, BackendError> {
        Self::with_device(0)
    }

    /// Initialize a new WASM backend with a specific `device_id`.
    /// The id is purely informational for WASM (no hardware-device
    /// notion in wasmtime); kept for symmetry with the GPU
    /// backends so [`crate::backend::Backend::Wasm`] can carry it
    /// through dispatcher routing.
    pub fn with_device(device_id: u32) -> Result<Self, BackendError> {
        // Default Engine uses the cranelift compiler. wasmtime's
        // pure-Rust engine cannot fail to construct on supported
        // platforms; if a host lacks cranelift support the build
        // would not have linked.
        let engine = Engine::default();

        // Spawn the persistent dispatch_one worker. Same shape as
        // the CUDA backend's worker (notify-hub-based, exits when
        // the hub is shut down).
        const WASM_WORKER_RING_CAPACITY: usize = 1024;
        let worker_hub = NotifyHub::<WorkItem>::new(WASM_WORKER_RING_CAPACITY, 1);
        let worker_tx = worker_hub.sender();
        let hub_for_worker = worker_hub.clone();
        let worker_handle = std::thread::Builder::new()
            .name(format!("flynnel-wasm-{device_id}"))
            .spawn(move || {
                let rx = hub_for_worker.register_consumer();
                while let Some(work) = rx.recv() {
                    work();
                }
            })
            .map_err(|_| BackendError::DeviceUnavailable(Backend::Wasm { device_id }))?;

        Ok(Self {
            device_id,
            engine,
            caps: probe_capabilities(),
            backend_id: NEXT_BACKEND_ID.fetch_add(1, Ordering::Relaxed),
            next_handle: AtomicU64::new(1),
            kernels: KernelTable::new(),
            worker_hub,
            worker_tx,
            worker_handle: Some(worker_handle),
        })
    }
}

/// Build a store of this thread's own for `entry`, resolving the same
/// export the shared one holds.
fn instantiate(entry: &KernelEntry) -> Result<SharedInstance, BackendError> {
    let mut store: Store<()> = Store::new(entry.module.engine(), ());
    let instance = Instance::new(&mut store, &entry.module, &[]).map_err(|e| {
        BackendError::Launch(format!(
            "wasm instance create failed for kernel `{}`: {e}",
            entry.export
        ))
    })?;
    let func = instance
        .get_func(&mut store, &entry.export)
        .ok_or_else(|| {
            BackendError::Launch(format!(
                "wasm export `{}` not found in module",
                entry.export
            ))
        })?;
    Ok(SharedInstance::new(store, func))
}

/// Call `instance` with `args` through wasmtime's unchecked call, after
/// checking them against the signature the instance read when it was
/// made.
///
/// wasmtime's checked call rebuilds the function's type from the
/// engine's shared type registry on every call. The count and each
/// argument's type are checked here instead, against a signature read
/// once, and answer the errors the checked call would; the slots are on
/// the stack for a function of up to [`INLINE_SLOTS`] parameters and
/// results.
fn call_instance(
    instance: &mut SharedInstance,
    args: &[KernelArg<'_>],
) -> Result<(), BackendError> {
    let expected_params = instance.params.len();
    if expected_params != args.len() {
        return Err(BackendError::Launch(format!(
            "wasm kernel arg count mismatch: function expects {} params, caller provided {}",
            expected_params,
            args.len(),
        )));
    }
    let slots = args.len().max(instance.results);
    let mut inline = [ValRaw::i32(0); INLINE_SLOTS];
    let mut spilled: Vec<ValRaw> = Vec::new();
    let buf: &mut [ValRaw] = if slots <= INLINE_SLOTS {
        &mut inline[..slots]
    } else {
        spilled.resize(slots, ValRaw::i32(0));
        &mut spilled[..]
    };
    for (i, (arg, want)) in args.iter().zip(instance.params.iter()).enumerate() {
        let (got, raw) = Scalar::raw(arg).ok_or(BackendError::NotSupported)?;
        if Some(got) != *want {
            return Err(BackendError::Launch(match want {
                Some(want) => format!(
                    "wasm kernel call failed: argument type mismatch: argument {i} is {got:?}, parameter {i} is {want:?}"
                ),
                None => format!(
                    "wasm kernel call failed: argument type mismatch: parameter {i} is a type no argument can carry"
                ),
            }));
        }
        buf[i] = raw;
    }
    let SharedInstance {
        ref mut store,
        ref func,
        ..
    } = *instance;
    // SAFETY: `buf` holds a slot for every parameter and every result;
    // the arguments' count matched the function's and each argument's
    // type matched its parameter's in the loop above, against the type
    // read from this function when the instance was made; every argument
    // is a scalar, so none is a reference needing a root; and `func`
    // belongs to `store`.
    unsafe { func.call_unchecked(store, buf as *mut [ValRaw]) }
        .map_err(|e| BackendError::Launch(format!("wasm kernel call failed: {e}")))?;
    Ok(())
}

fn probe_capabilities() -> BackendCapabilities {
    // WASM execution is scalar single-threaded. Conservative
    // numbers: 1-wide, host-thread-count for max in-flight (host-
    // side fan-out via dispatch_parallel_for), ~5us per launch
    // (cranelift call setup + sandbox entry).
    let threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1) as u32;
    BackendCapabilities {
        simt_width: 1,
        max_threads_in_flight: threads,
        launch_latency_ns: 5_000,
        h2d_bw_bytes_per_sec: 0,
    }
}

impl DispatchBackend for WasmBackend {
    fn id(&self) -> Backend {
        Backend::Wasm {
            device_id: self.device_id,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.caps
    }

    fn dispatch_parallel_for(&self, count: u32, work: &(dyn Fn(u32) + Send + Sync)) {
        // Host-side fan-out via the global flynnel scheduler arena.
        // The closure body is a CPU-runnable Rust closure (not a
        // WASM kernel); for WASM kernel parallel fan-out, call
        // dispatch_kernel inside a host-side parallel loop.
        let plan = crate::sched::JobPlan::new(0, count);
        let mut indices: Vec<u32> = (0..count).collect();
        crate::sched::par_iter::for_each_chunk(
            &plan,
            indices.as_mut_slice(),
            move |slice: &mut [u32]| {
                for i in slice.iter() {
                    work(*i);
                }
            },
        );
    }

    fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
        // Best-effort send. If the worker has already exited
        // (during Drop) the hub is shut down and `send` returns
        // Closed; drop the work item silently to match the CUDA
        // backend's behavior.
        drop(self.worker_tx.send(work));
    }

    fn register_kernel(&self, name: &str, source: &[u8]) -> Result<KernelHandle, BackendError> {
        // Compile the .wasm module via the engine's cranelift JIT.
        let module = Module::new(&self.engine, source)
            .map_err(|e| BackendError::KernelCompile(format!("wasm module compile failed: {e}")))?;
        // Each kernel owns its own Store. Empty host imports (no
        // host functions exported to the kernel; the kernel is
        // pure compute over its arguments and linear memory).
        let mut store: Store<()> = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, &module, &[]).map_err(|e| {
            BackendError::KernelCompile(format!(
                "wasm instance create failed for kernel `{name}`: {e}"
            ))
        })?;
        // Look up the named export and confirm it is a function.
        let func = instance.get_func(&mut store, name).ok_or_else(|| {
            BackendError::KernelCompile(format!("wasm export `{name}` not found in module"))
        })?;
        let handle_id = self.next_handle.fetch_add(1, Ordering::SeqCst);
        // The module and the export name are kept beside the shared
        // store, because a thread building a store of its own needs
        // both and neither can be recovered from the store.
        let entry = KernelEntry {
            shared: Mutex::new(SharedInstance::new(store, func)),
            module,
            export: name.to_string(),
        };
        self.kernels.insert(handle_id, Arc::new(entry));
        Ok(KernelHandle(handle_id))
    }

    fn dispatch_kernel(
        &self,
        handle: KernelHandle,
        _count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        // A wasm kernel takes scalars only; a slice or a buffer is refused
        // before the kernel is looked up.
        if args.iter().any(|a| Scalar::raw(a).is_none()) {
            return Err(BackendError::NotSupported);
        }
        // Locate the kernel entry and call its function.
        let entry_arc = self.kernels.get(handle.0).ok_or_else(|| {
            BackendError::Launch(format!("wasm kernel handle {} not registered", handle.0))
        })?;
        if crate::sched::levers::wasm_local_store() {
            return with_local_instance(
                self.backend_id,
                handle.0,
                || instantiate(&entry_arc),
                |instance| call_instance(instance, args),
            );
        }
        let mut shared = entry_arc
            .shared
            .lock()
            .map_err(|_| BackendError::Launch("wasm kernel store mutex poisoned".into()))?;
        call_instance(&mut shared, args)
    }
}

impl Drop for WasmBackend {
    fn drop(&mut self) {
        // Shut down the notify hub so the worker's recv() returns
        // None and it exits cleanly after draining queued work.
        self.worker_hub.shutdown();
        if let Some(handle) = self.worker_handle.take() {
            match handle.join() {
                Ok(()) => {}
                // The worker panicked. Drop cannot recover, and
                // reporting it here would replace whatever is already
                // unwinding.
                Err(panicked) => drop(panicked),
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in entry, so the table's own behavior is tested
    /// without building a wasmtime store for every slot.
    fn table_entry(engine: &Engine) -> Arc<KernelEntry> {
        let module = Module::new(engine, ADD_WASM).expect("the add module builds");
        let mut store: Store<()> = Store::new(engine, ());
        let instance = Instance::new(&mut store, &module, &[]).expect("it instantiates");
        let func = instance
            .get_func(&mut store, "add")
            .expect("add is exported");
        Arc::new(KernelEntry {
            shared: Mutex::new(SharedInstance::new(store, func)),
            module,
            export: "add".to_string(),
        })
    }

    #[test]
    fn a_second_backend_does_not_inherit_the_first_ones_slot() {
        let engine = Engine::default();
        // Two backends both start their handles at one, and the
        // per-thread table is keyed by handle, so without the owner id
        // the second one's dispatch would run the first one's
        // function. Both hold the same module here, so the answer
        // would be right either way and only the bookkeeping shows it.
        let entry = table_entry(&engine);
        let build = || instantiate(&entry);

        with_local_instance(101, 1, build, |_| Ok(())).expect("the first backend builds its slot");
        assert_eq!(
            local_slot_owner(1),
            Some(101),
            "the slot belongs to whichever backend built it"
        );

        with_local_instance(202, 1, build, |_| Ok(())).expect("the second backend builds its own");
        assert_eq!(
            local_slot_owner(1),
            Some(202),
            "a slot another backend holds is rebuilt rather than reused"
        );

        with_local_instance(101, 1, build, |_| Ok(())).expect("the first backend builds again");
        assert_eq!(
            local_slot_owner(1),
            Some(101),
            "and back again, because the slot follows whoever asked last"
        );
    }

    #[test]
    fn the_kernel_table_holds_every_handle_across_blocks() {
        let engine = Engine::default();
        let table = KernelTable::new();
        // Past two block boundaries, so the walk and the appending are
        // both exercised rather than only the head block.
        let count = (KERNEL_BLOCK * 2 + 3) as u64;
        for handle in 1..=count {
            table.insert(handle, table_entry(&engine));
        }
        for handle in 1..=count {
            assert!(
                table.get(handle).is_some(),
                "handle {handle} was registered and must be found"
            );
        }
        assert!(table.get(0).is_none(), "zero is never a handle");
        assert!(
            table.get(count + 1).is_none(),
            "a handle nothing registered under reads as absent"
        );
        assert!(
            table.get(count + KERNEL_BLOCK as u64 * 4).is_none(),
            "so does one whose block was never appended"
        );
    }

    #[test]
    fn two_registrations_racing_on_one_missing_block_both_land() {
        let engine = Engine::default();
        let table = Arc::new(KernelTable::new());
        // Handles chosen so both need the same block appended, which
        // is the case the exchange in block_for exists for.
        let first = KERNEL_BLOCK as u64 * 3 + 1;
        let second = first + 1;
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let mut threads = Vec::new();
        for handle in [first, second] {
            let table = Arc::clone(&table);
            let barrier = Arc::clone(&barrier);
            let engine = engine.clone();
            threads.push(std::thread::spawn(move || {
                let entry = table_entry(&engine);
                barrier.wait();
                table.insert(handle, entry);
            }));
        }
        for t in threads {
            t.join().expect("a registering thread must not panic");
        }
        assert!(table.get(first).is_some(), "the first handle is present");
        assert!(table.get(second).is_some(), "and so is the second");
    }

    /// Tiny WASM module exporting a function `add` that takes two
    /// i32 parameters and returns their sum. Hand-assembled binary
    /// from the WAT source:
    ///   (module
    ///     (func (export "add") (param i32 i32) (result i32)
    ///       local.get 0
    ///       local.get 1
    ///       i32.add))
    /// Self-contained byte literal so the test does not pull in
    /// `wat` or `wabt` as a dev-dependency.
    const ADD_WASM: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, // magic
        0x01, 0x00, 0x00, 0x00, // version
        // type section: 1 type, (i32, i32) -> (i32)
        0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
        // function section: 1 function, type 0
        0x03, 0x02, 0x01, 0x00, // export section: 1 export, "add" func 0
        0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00, // code section: 1 body
        0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
    ];

    #[test]
    fn wasm_backend_constructs() {
        WasmBackend::new().expect("wasm backend should init");
    }

    #[test]
    fn wasm_backend_reports_correct_id_and_caps() {
        let b = WasmBackend::with_device(7).expect("init");
        assert_eq!(b.id(), Backend::Wasm { device_id: 7 });
        let caps = b.capabilities();
        assert_eq!(caps.simt_width, 1);
        assert!(caps.max_threads_in_flight >= 1);
    }

    #[test]
    fn wasm_backend_dispatches_kernel_with_add_module() {
        let b = WasmBackend::new().expect("init");
        let handle = b
            .register_kernel("add", ADD_WASM)
            .expect("register `add` export");
        // (i32, i32) -> i32 with values (3, 4). dispatch_kernel
        // does not surface the return value through the trait
        // surface; the call succeeding (no error) is the test
        // contract. The function executes inside the wasmtime
        // sandbox and produces 7, which wasmtime discards.
        b.dispatch_kernel(handle, 1, &[KernelArg::I32(3), KernelArg::I32(4)])
            .expect("dispatch_kernel should succeed on valid args");
    }

    #[test]
    fn wasm_backend_rejects_arity_mismatch() {
        let b = WasmBackend::new().expect("init");
        let handle = b.register_kernel("add", ADD_WASM).expect("register");
        // `add` expects 2 args; pass 1. dispatch_kernel must
        // surface a Launch error rather than panicking inside
        // wasmtime.
        let err = b
            .dispatch_kernel(handle, 1, &[KernelArg::I32(3)])
            .expect_err("arity mismatch must error");
        assert!(matches!(err, BackendError::Launch(_)));
    }

    /// A module exporting `add` over two f64 parameters, returning their
    /// sum, from the WAT source:
    ///   (module
    ///     (func (export "add") (param f64 f64) (result f64)
    ///       local.get 0
    ///       local.get 1
    ///       f64.add))
    const F64_ADD_WASM: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, // magic
        0x01, 0x00, 0x00, 0x00, // version
        0x01, 0x07, 0x01, 0x60, 0x02, 0x7c, 0x7c, 0x01, 0x7c, // type: (f64, f64) -> (f64)
        0x03, 0x02, 0x01, 0x00, // function: type 0
        0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00, // export: "add", function 0
        0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0xa0, 0x0b, // code: f64.add
    ];

    /// A module exporting `seven`, which takes nothing and returns 7: a
    /// function with more results than parameters, from the WAT source:
    ///   (module (func (export "seven") (result i32) i32.const 7))
    const SEVEN_WASM: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, // magic
        0x01, 0x00, 0x00, 0x00, // version
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f, // type: () -> (i32)
        0x03, 0x02, 0x01, 0x00, // function: type 0
        0x07, 0x09, 0x01, 0x05, 0x73, 0x65, 0x76, 0x65, 0x6e, 0x00, 0x00, // export: "seven"
        0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x07, 0x0b, // code: i32.const 7
    ];

    #[test]
    fn wasm_backend_rejects_an_argument_of_the_wrong_type() {
        let b = WasmBackend::new().expect("init");
        let handle = b.register_kernel("add", ADD_WASM).expect("register");
        // `add` takes two i32; an i64 in the first place is refused as a
        // Launch error before the unchecked call, never passed to it.
        let err = b
            .dispatch_kernel(handle, 1, &[KernelArg::I64(3), KernelArg::I32(4)])
            .expect_err("a mistyped argument must error");
        assert!(matches!(err, BackendError::Launch(_)), "{err:?}");
    }

    #[test]
    fn wasm_backend_runs_a_float_kernel_and_refuses_the_other_float() {
        let b = WasmBackend::new().expect("init");
        let handle = b.register_kernel("add", F64_ADD_WASM).expect("register");
        b.dispatch_kernel(handle, 1, &[KernelArg::F64(1.5), KernelArg::F64(2.25)])
            .expect("two f64 arguments reach an (f64, f64) kernel");
        let err = b
            .dispatch_kernel(handle, 1, &[KernelArg::F32(1.5), KernelArg::F64(2.25)])
            .expect_err("an f32 where the kernel takes an f64 must error");
        assert!(matches!(err, BackendError::Launch(_)), "{err:?}");
    }

    #[test]
    fn wasm_backend_runs_a_kernel_returning_more_than_it_takes() {
        let b = WasmBackend::new().expect("init");
        let handle = b.register_kernel("seven", SEVEN_WASM).expect("register");
        // No arguments and one result, so the call's slots are sized by
        // the result rather than by the arguments.
        b.dispatch_kernel(handle, 1, &[])
            .expect("a kernel taking nothing runs");
    }

    #[test]
    fn wasm_backend_register_kernel_with_missing_export_errors() {
        let b = WasmBackend::new().expect("init");
        let err = b
            .register_kernel("does_not_exist", ADD_WASM)
            .expect_err("missing export must error");
        assert!(matches!(err, BackendError::KernelCompile(_)));
    }

    #[test]
    fn wasm_backend_register_invalid_module_errors() {
        let b = WasmBackend::new().expect("init");
        let err = b
            .register_kernel("x", b"not a wasm module")
            .expect_err("invalid bytes must error");
        assert!(matches!(err, BackendError::KernelCompile(_)));
    }

    #[test]
    fn wasm_backend_host_slice_arg_returns_not_supported() {
        let b = WasmBackend::new().expect("init");
        let handle = b.register_kernel("add", ADD_WASM).expect("register");
        let buf: [u8; 4] = [1, 2, 3, 4];
        let err = b
            .dispatch_kernel(handle, 1, &[KernelArg::HostSlice(&buf)])
            .expect_err("HostSlice must be unsupported in reference impl");
        assert!(matches!(err, BackendError::NotSupported));
    }
}

//! A/B microbench: where a wasm kernel's store lives.
//!
//! A `wasmtime::Store` is not `Sync` and calling into it takes a
//! mutable borrow, so a store shared between threads has to be behind
//! a lock and two dispatches of one kernel take turns. The lever
//! `FLYNNEL_LEVER_WASM_LOCAL_STORE` gives each thread a store of its
//! own per kernel instead, built on that thread's first dispatch of
//! it, which removes the lock and the taking of turns with it.
//!
//! The lever is read once per process, so the two arms are two runs
//! of this binary and not two groups inside one:
//!
//!   FLYNNEL_LEVER_WASM_LOCAL_STORE=0 cargo bench --bench wasm_store_locality
//!   FLYNNEL_LEVER_WASM_LOCAL_STORE=1 cargo bench --bench wasm_store_locality
//!
//! ## Bench-audit
//!
//! - **Same payload across A/B**: the same `(i32, i32) -> i32` add
//!   module, the same argument slice, the same thread count, the same
//!   dispatch count. Only where the store came from differs.
//! - **The instantiation is inside the measured span.** Per-thread
//!   stores do not remove that cost, they move it from registration
//!   to first dispatch on each thread. A steady-state-only reading
//!   would credit the lever with a win no short-lived dispatch ever
//!   sees, so `first_dispatch` starts its threads inside the iter and
//!   pays instantiation where a caller would.
//! - **Both the short-lived and the settled shape, every arm in
//!   both.** `first_dispatch` is one dispatch per freshly started
//!   thread, where the shared arm instantiates once and the local arm
//!   instantiates per thread, and it is the shape the lever can only
//!   lose in. `settled` hands work to a crew that outlives the
//!   iterations, so a thread's store is built during warm-up and the
//!   lock is the only thing left between dispatches, and it is the
//!   shape the lever exists for. A crew respawned per iteration would
//!   make `settled` a second copy of `first_dispatch` wearing the
//!   other name. A lever adopted on `settled` alone would be adopted
//!   on the half of the evidence that favors it.
//! - **One thread and many.** An uncontended lock is a handful of
//!   cycles, so the single-thread rows are where "never slower" has to
//!   hold, and the many-thread rows are where the turns being taken
//!   are visible at all.
//! - **Idle and loaded.** Under load a thread waiting its turn on the
//!   store loses its core to other runnable work, which an idle host
//!   hides.
//!
//! The crew hands out work through a generation counter and counts
//! completions back, both atomics, and waits by spinning. A park would
//! put a wake inside the span, and the wake is not what is being
//! compared.
//!
//! The backend needs the `wasm-reference` feature, which the bench
//! declares in Cargo.toml; a host without it builds no binary here
//! rather than reporting an empty pass.

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::backend::wasm::WasmBackend;
use flynnel::backend::{DispatchBackend, KernelArg, KernelHandle};

/// The `(i32, i32) -> i32` add module the backend's own tests use.
const ADD_WASM: &[u8] = include_bytes!("../kernels/add_i32.wasm");

/// The module's export. `register_kernel` looks up the export by the
/// name it is given, so this is the name inside the module and not the
/// name of the file holding it, which is add_i32.
const ADD_EXPORT: &str = "add";

/// Spins before a courtesy yield, so a waiter on an oversubscribed
/// host gives up its core instead of holding it against the thread it
/// is waiting for.
const SPINS_PER_YIELD: u64 = 1024;

/// Dispatches that did not complete, counted rather than asserted so a
/// failure is reported beside the timing it was taken with instead of
/// aborting the run that found it.
static UNFINISHED: AtomicU64 = AtomicU64::new(0);

/// The first dispatch failure, said once.
///
/// A backend that has stopped answering fails every later call the
/// same way, and a line each would bury the run's own output under
/// thousands of copies of one fact, in a log a chain then greps. The
/// count in `UNFINISHED` is what says how many there were.
static SAID: std::sync::Once = std::sync::Once::new();

fn say_once(what: impl FnOnce() -> String) {
    SAID.call_once(|| eprintln!("{}", what()));
}

/// Busy threads occupying half the host, joined when this is dropped.
struct Load {
    stop: Arc<AtomicU32>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Load {
    fn spawn() -> Self {
        let n = std::thread::available_parallelism()
            .map(|p| p.get() / 2)
            .unwrap_or(1)
            .max(1);
        let stop = Arc::new(AtomicU32::new(0));
        let threads = (0..n)
            .map(|_| {
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut x = 0u64;
                    while stop.load(Ordering::Relaxed) == 0 {
                        x = black_box(x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1));
                    }
                })
            })
            .collect();
        Self { stop, threads }
    }
}

impl Drop for Load {
    fn drop(&mut self) {
        self.stop.store(1, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            if t.join().is_err() {
                eprintln!(
                    "wasm_store_locality: a load thread panicked, so this run's loaded rows carried less load than they report"
                );
            }
        }
    }
}

/// One thread's share: `count` dispatches of the add kernel.
///
/// The backend is shared by `Arc` rather than built per thread,
/// because a backend per thread would give the shared arm a store per
/// thread by the back door and there would be nothing left to measure.
fn dispatch_n(backend: &Arc<WasmBackend>, handle: KernelHandle, count: u32) {
    for i in 0..count {
        let a = (i % 7) as i32;
        let b = (i % 5) as i32;
        match backend.dispatch_kernel(handle, 1, &[KernelArg::I32(a), KernelArg::I32(b)]) {
            Ok(()) => {}
            Err(e) => {
                say_once(|| format!("wasm_store_locality: dispatch failed: {e}"));
                UNFINISHED.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        black_box(a + b);
    }
}

/// Spin until `ready` returns true, yielding periodically.
fn spin_until<F: Fn() -> bool>(ready: F) {
    let mut spins = 0u64;
    while !ready() {
        std::hint::spin_loop();
        spins += 1;
        if spins.is_multiple_of(SPINS_PER_YIELD) {
            std::thread::yield_now();
        }
    }
}

/// Threads that outlive the iterations, so a thread-local store built
/// on the first iteration is still there on the rest.
///
/// Work is handed out by bumping `generation` and taken back by
/// counting `done` up to the crew size.
struct Crew {
    generation: Arc<AtomicU64>,
    done: Arc<AtomicUsize>,
    stop: Arc<AtomicU32>,
    threads: Vec<std::thread::JoinHandle<()>>,
    size: usize,
}

impl Crew {
    fn spawn(
        backend: &Arc<WasmBackend>,
        handle: KernelHandle,
        size: usize,
        per_thread: u32,
    ) -> Self {
        let generation = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicU32::new(0));
        let threads = (0..size)
            .map(|_| {
                let backend = Arc::clone(backend);
                let generation = Arc::clone(&generation);
                let done = Arc::clone(&done);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut seen = 0u64;
                    loop {
                        let mut spins = 0u64;
                        loop {
                            if stop.load(Ordering::Relaxed) != 0 {
                                return;
                            }
                            let g = generation.load(Ordering::Acquire);
                            if g > seen {
                                seen = g;
                                break;
                            }
                            std::hint::spin_loop();
                            spins += 1;
                            if spins.is_multiple_of(SPINS_PER_YIELD) {
                                std::thread::yield_now();
                            }
                        }
                        dispatch_n(&backend, handle, per_thread);
                        done.fetch_add(1, Ordering::Release);
                    }
                })
            })
            .collect();
        Self {
            generation,
            done,
            stop,
            threads,
            size,
        }
    }

    /// One round: every thread does its share, and this returns when
    /// all of them have. The reset of `done` is safe to do first
    /// because the previous round did not return until every thread
    /// had counted itself in.
    fn round(&self) {
        self.done.store(0, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Release);
        spin_until(|| self.done.load(Ordering::Acquire) >= self.size);
    }
}

impl Drop for Crew {
    fn drop(&mut self) {
        self.stop.store(1, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            if t.join().is_err() {
                eprintln!(
                    "wasm_store_locality: a crew thread panicked, so the settled rows are short a thread's share"
                );
                UNFINISHED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Start `threads` threads, each doing `per_thread` dispatches, and
/// join them. The start is inside the timed span on purpose: it is
/// what makes a thread's first dispatch its first, which is the only
/// place the local arm's instantiation is paid.
fn fan_out(backend: &Arc<WasmBackend>, handle: KernelHandle, threads: usize, per_thread: u32) {
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let backend = Arc::clone(backend);
        handles.push(std::thread::spawn(move || {
            dispatch_n(&backend, handle, per_thread);
        }));
    }
    for h in handles {
        if h.join().is_err() {
            eprintln!(
                "wasm_store_locality: a dispatch thread panicked, so this sample is short its share of the work"
            );
            UNFINISHED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn arm_label() -> &'static str {
    if flynnel::sched::levers::wasm_local_store() {
        "local"
    } else {
        "shared"
    }
}

/// Returns whether any row was timed, so the closing line can tell a
/// clean run from one that measured nothing. Every dispatch completing
/// is true of a run with no dispatches in it, and on its own it reads
/// like a pass.
fn bench_shapes(c: &mut Criterion, loaded: bool) -> bool {
    let backend = match WasmBackend::new() {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("wasm_store_locality: no wasm backend on this host ({e}), nothing measured");
            return false;
        }
    };
    let handle = match backend.register_kernel(ADD_EXPORT, ADD_WASM) {
        Ok(h) => h,
        Err(e) => {
            eprintln!(
                "wasm_store_locality: the add module did not register ({e}), nothing measured"
            );
            return false;
        }
    };

    let wide = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .max(2);
    let suffix = if loaded { "loaded" } else { "idle" };
    let _load = if loaded { Some(Load::spawn()) } else { None };

    // A cell's name does not carry the arm. The lever is read once per
    // process, so the two arms are two runs of this binary, and a
    // report pairs a cell in one run with the cell of the same name in
    // the other. Naming the arm there gives the two runs disjoint
    // names, nothing pairs, and the comparison reports no rows at all.
    // Which arm a run was is on its first line and in the file it was
    // written to.
    let mut group = c.benchmark_group(format!("wasm_store/{suffix}"));
    group.measurement_time(Duration::from_secs(8));
    // Deduplicated, because criterion refuses two benchmarks with one
    // id and a host of one core would otherwise ask for t1 twice. The
    // floor of two on `wide` covers that already; this holds if the
    // floor ever moves.
    let mut widths = vec![1usize, wide];
    widths.dedup();

    // One dispatch per freshly started thread. The local arm
    // instantiates once per thread here and the shared arm does not,
    // so this is where the lever pays before it can earn anything.
    for threads in widths.iter().copied() {
        group.bench_function(format!("first_dispatch/t{threads}"), |b| {
            b.iter(|| fan_out(&backend, handle, threads, 1));
        });
    }

    // Many dispatches from a crew that is already running. Past its
    // first round the local arm has its stores and the shared arm
    // still has the lock.
    for threads in widths.iter().copied() {
        let crew = Crew::spawn(&backend, handle, threads, 64);
        group.bench_function(format!("settled/t{threads}"), |b| {
            b.iter(|| crew.round());
        });
    }

    group.finish();
    true
}

fn bench_all(c: &mut Criterion) {
    eprintln!(
        "wasm_store_locality: arm={} (FLYNNEL_LEVER_WASM_LOCAL_STORE), cores={}",
        arm_label(),
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(0)
    );
    let idle = bench_shapes(c, false);
    let loaded = bench_shapes(c, true);
    let unfinished = UNFINISHED.load(Ordering::Relaxed);
    if !idle && !loaded {
        eprintln!(
            "wasm_store_locality: nothing measured in either shape, so there is no result here to read"
        );
    } else if unfinished > 0 {
        eprintln!(
            "wasm_store_locality: {unfinished} shares did not complete, so these timings are of a run that was not doing all of its work"
        );
    } else {
        eprintln!("wasm_store_locality: every dispatch completed");
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);

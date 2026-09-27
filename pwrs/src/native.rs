//! Native entry points: work another native library in this process can
//! run on this module's worker pool without linking the flynnel crate a
//! second time.
//!
//! Two cdylibs that each link flynnel get two arenas, each sizing itself
//! against the same cores. A library that wants to share this one calls
//! into it here instead. PWRS loads every native library from a renamed
//! copy under a per-process temp folder, so an entry cannot be found by
//! name; `Get-FlynnelNativeEntry` hands out its address from the copy
//! that is loaded.
//!
//! Every symbol carries its ABI version in its name. Once a released
//! library calls an entry, a change to it is a new symbol beside the old
//! one, never an edit under that caller. Before then an entry changes in
//! place, and [`NativeEntry`] says which form of it the loaded module
//! has, so a caller built for the other form refuses rather than calls.

use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use flynnel::sched::par_iter::for_each_chunk_indexed_min_leaf;
use pwrs::prelude::*;

use crate::kernels::band_for;
use crate::plan::Plan;

/// The ABI version `Get-FlynnelNativeEntry` reports for the entries it
/// hands out.
pub const NATIVE_ABI_VERSION: u32 = 1;

/// Returned by [`flynnel_run_chunks_v1`] when a panic in this module's
/// own dispatch code was caught.
///
/// A panic in the caller's body cannot reach this: unwinding out of an
/// `extern "C"` function aborts the process at that function's own
/// boundary, before control returns here.
pub const RUN_CHUNKS_PANIC: i32 = -1;

/// Returned by [`flynnel_run_chunks_plan_v1`] when its plan handle names
/// no live `Flynnel.NativePlan`: one never handed out, or one whose object
/// has been disposed or collected. No body runs.
pub const RUN_CHUNKS_NO_PLAN: i32 = -2;

/// The body a caller runs over one chunk: `ctx` as the caller passed it
/// and the half-open index range `[start, end)`. Zero continues; any
/// other value stops the run.
pub type ChunkBodyV1 = extern "C" fn(ctx: *const c_void, start: usize, end: usize) -> i32;

/// [`flynnel_run_chunks_v1`] as a function pointer: the type a caller
/// holding its address casts it back to.
pub type RunChunksV1 = unsafe extern "C" fn(
    n: usize,
    min_leaf: usize,
    site: u64,
    body: ChunkBodyV1,
    ctx: *const c_void,
) -> i32;

/// [`flynnel_run_chunks_plan_v1`] as a function pointer.
pub type RunChunksPlanV1 = unsafe extern "C" fn(
    n: usize,
    min_leaf: usize,
    site: u64,
    plan: u64,
    body: ChunkBodyV1,
    ctx: *const c_void,
) -> i32;

/// Run `body` over `[0, n)` in chunks on this module's worker pool, and
/// return once every chunk has finished or the run has stopped.
///
/// Returns 0 when every chunk ran and every body returned 0; otherwise
/// the first nonzero value a body returned, after which no further
/// chunk starts and chunks already running finish; or
/// [`RUN_CHUNKS_PANIC`].
///
/// `min_leaf` is the smallest chunk worth dispatching, and zero is read
/// as one. Light per-item work wants a large floor so dispatch cost
/// does not dominate it; heavy per-item work wants one, so a short job
/// is still split across workers rather than run on one.
///
/// `site` names the kind of work this job is. The pool learns how to
/// dispatch per site - the class of the work, what an item costs, how
/// deep to seed the split - so jobs passed the same key learn together
/// and jobs passed different keys learn apart. A caller running several
/// kernels passes one key per kernel; a key shared by work of different
/// costs trains one classifier on a mixture that describes none of it.
/// Any value is a key, and no key names a site of this module's own
/// cmdlets, whose sites are source locations.
///
/// Synchronous. A caller that has to stay responsive calls it once per
/// batch and checks for cancellation between calls. The calling thread
/// blocks for the call.
///
/// No body runs on the calling thread. The dispatch is started from one
/// of this module's workers, so the probe that measures per-item cost
/// and any job small enough to run inline both land on a worker, whose
/// 8 MiB stack this module sets, rather than on the caller's, whose
/// stack it neither sets nor knows.
///
/// The dispatch keeps SMT siblings parked, as this crate does for any
/// call site that is not one of its own classified kernel operations: a
/// foreign body's latency class is not known here.
///
/// # Safety
///
/// `ctx` is read by every chunk at once, from several threads, for the
/// whole call, so what it points at must be safe to share that way and
/// must outlive the call. Chunks cover disjoint index ranges, so a body
/// that writes only what its own range owns does not race another. A
/// body must touch no CLR object, because a worker that calls into the
/// managed runtime stays attached to it for the rest of its life, and
/// must not unwind, because that aborts the process.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flynnel_run_chunks_v1(
    n: usize,
    min_leaf: usize,
    site: u64,
    body: ChunkBodyV1,
    ctx: *const c_void,
) -> i32 {
    run_chunks(n, min_leaf, site, None, body, ctx, "flynnel_run_chunks_v1")
}

/// [`flynnel_run_chunks_v1`] under a plan the caller chose rather than the
/// one this module sizes to `n`.
///
/// `plan` is the `Handle` of a live `Flynnel.NativePlan`, which
/// `New-FlynnelNativePlan` makes from a `Flynnel.JobPlan`. The dispatch runs
/// under that plan as the kernel cmdlets run under their `-Plan`, and
/// learns at the site `site` keys. A handle that names no live plan
/// returns [`RUN_CHUNKS_NO_PLAN`] and runs no body. Every other argument,
/// the other returns and the contract are [`flynnel_run_chunks_v1`]'s.
///
/// # Safety
///
/// As [`flynnel_run_chunks_v1`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flynnel_run_chunks_plan_v1(
    n: usize,
    min_leaf: usize,
    site: u64,
    plan: u64,
    body: ChunkBodyV1,
    ctx: *const c_void,
) -> i32 {
    match plan_for(plan) {
        Some(chosen) => run_chunks(
            n,
            min_leaf,
            site,
            Some(chosen),
            body,
            ctx,
            "flynnel_run_chunks_plan_v1",
        ),
        None => RUN_CHUNKS_NO_PLAN,
    }
}

/// Both entries' shared body: inject the dispatch onto a worker, wait for
/// it, and answer the first nonzero code or [`RUN_CHUNKS_PANIC`]. `plan`
/// is the caller's, or `None` for the one sized to `n`. `entry` names the
/// entry in the message a caught panic prints.
fn run_chunks(
    n: usize,
    min_leaf: usize,
    site: u64,
    plan: Option<flynnel::JobPlan>,
    body: ChunkBodyV1,
    ctx: *const c_void,
    entry: &str,
) -> i32 {
    // A raw pointer is neither Send nor Sync, so the address travels as
    // an integer and is rebuilt in each chunk. The caller's contract is
    // what makes sharing it sound.
    let ctx_addr = ctx as usize;
    let stop = AtomicI32::new(0);
    let dispatched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if n == 0 {
            return;
        }
        // An explicit leaf shape with a batch of at least eight is the
        // one plan pick_tier never routes Inline, so the pool always
        // takes this join. Called from outside the pool the whole join
        // is injected onto a worker and the caller blocks; called from a
        // worker it runs where it already is.
        let onto_a_worker =
            flynnel::JobPlan::new(0, 8).with_leaf_shape(flynnel::LeafShape::PortCompute);
        flynnel::join(
            &onto_a_worker,
            || dispatch_chunks(n, min_leaf, site, plan, body, ctx_addr, &stop),
            || (),
        );
    }));
    match dispatched {
        Ok(()) => stop.load(Ordering::Acquire),
        Err(panic) => {
            let what = panic
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("a panic that carried no message");
            eprintln!("{entry} caught a panic in its dispatch: {what}");
            RUN_CHUNKS_PANIC
        }
    }
}

/// The chunked dispatch itself, which [`run_chunks`] runs from a worker.
///
/// Records into `stop` the first nonzero code a body returns and skips
/// every chunk that has not started by then. Runs under `plan`, or one
/// sized to `n` when that is `None`, and learns at the site `site` keys,
/// not at this function's own source location.
fn dispatch_chunks(
    n: usize,
    min_leaf: usize,
    site: u64,
    plan: Option<flynnel::JobPlan>,
    body: ChunkBodyV1,
    ctx_addr: usize,
    stop: &AtomicI32,
) {
    let plan = match plan {
        Some(chosen) => chosen,
        None => flynnel::JobPlan::new(band_for(n), n.min(u32::MAX as usize) as u32),
    }
    .with_site(flynnel::site_for_key(site));
    // One zero-sized slot per index. The indexed helper splits a slice,
    // and a slice of unit values carries the range and allocates nothing.
    let mut slots = vec![(); n];
    for_each_chunk_indexed_min_leaf(&plan, &mut slots, min_leaf.max(1), |start, chunk| {
        if stop.load(Ordering::Acquire) != 0 {
            return;
        }
        let code = body(ctx_addr as *const c_void, start, start + chunk.len());
        if code != 0 {
            match stop.compare_exchange(0, code, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {}
                // An earlier chunk stopped the run first, and the first
                // code is the one reported.
                Err(_earlier) => {}
            }
        }
    });
}

/// The plans `New-FlynnelNativePlan` has handed out, each under the handle
/// its object carries. An entry lives exactly as long as that object.
static PLANS: Mutex<Vec<(u64, flynnel::JobPlan)>> = Mutex::new(Vec::new());

/// Strictly increasing handles from 1, so 0 names no plan and a handle
/// is never reused within a process.
static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

/// The plan table, recovered from poisoning: nothing here keeps an
/// invariant across two steps, so a panic while the lock was held leaves
/// the entries sound.
fn plans() -> MutexGuard<'static, Vec<(u64, flynnel::JobPlan)>> {
    match PLANS.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// The plan `handle` names, if its object still lives.
fn plan_for(handle: u64) -> Option<flynnel::JobPlan> {
    plans()
        .iter()
        .find(|(h, _)| *h == handle)
        .map(|(_, plan)| *plan)
}

/// Removes a plan's entry when the object holding its handle goes, by
/// `Dispose` or collection. Rust-only, so the class cannot be read back
/// by value and no copy can take the live entry with it.
struct PlanGuard(u64);

impl Drop for PlanGuard {
    fn drop(&mut self) {
        let mut table = plans();
        if let Some(at) = table.iter().position(|(h, _)| *h == self.0) {
            table.remove(at);
        }
    }
}

/// A plan a native caller passes to `flynnel_run_chunks_plan_v1` by its
/// handle. The handle names the plan until this object is disposed or
/// collected; after that the entry refuses it.
#[psclass(name = "Flynnel.NativePlan", mode = proxy)]
pub struct NativePlan {
    /// The value a native caller passes as the entry's `plan` argument.
    pub handle: u64,
    /// The plan's `k_outer`, as resolved when the handle was made.
    pub k_outer: u8,
    /// The plan's batch size, as resolved when the handle was made.
    pub batch_size: u32,
    // Read by nothing. Its Drop is the whole of its job, and that is what
    // removes the table entry when the object is disposed or collected.
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: PlanGuard,
}

/// Turn a plan into a handle a native caller can pass to
/// `flynnel_run_chunks_plan_v1`.
///
/// The plan is resolved now, as the kernel cmdlets resolve their -Plan,
/// so a plan whose numbers are missing is refused here rather than at the
/// call. The handle stays valid while the object this writes lives;
/// dispose it, or let it be collected, and the entry refuses the handle.
///
/// # Examples
///
/// `$native = New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Workers 4 | New-FlynnelNativePlan`
#[cmdlet(
    verb = "New",
    noun = "FlynnelNativePlan",
    alias = "New-FlyNativePlan",
    output = ["Flynnel.NativePlan"]
)]
#[derive(Default)]
pub struct NewFlynnelNativePlan {
    /// The plan the native caller's dispatch runs under.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub plan: Plan,
}

impl Cmdlet for NewFlynnelNativePlan {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(native_plan(&self.plan)?)
    }
}

/// Registers `plan` and answers the object that keeps its handle live.
fn native_plan(plan: &Plan) -> PsResult<NativePlan> {
    let job = plan.to_job_plan()?;
    let handle = NEXT_PLAN.fetch_add(1, Ordering::Relaxed);
    plans().push((handle, job));
    Ok(NativePlan {
        handle,
        k_outer: job.k_outer,
        batch_size: job.batch_size,
        guard: PlanGuard(handle),
    })
}

/// Where the native entry points are in the library that is loaded now,
/// and the ABI version they speak.
#[psclass(name = "Flynnel.NativeEntry")]
#[derive(Clone, Default)]
pub struct NativeEntry {
    /// The address of `flynnel_run_chunks_v1`.
    pub run_chunks_v1: u64,
    /// The ABI version of the entries in this object.
    pub abi_version: u32,
    /// Whether `flynnel_run_chunks_v1` takes a site key, its third
    /// argument: `(n, min_leaf, site, body, ctx)`. A module without this
    /// property, or with it false, has the form without one,
    /// `(n, min_leaf, body, ctx)`, under the same name and ABI version,
    /// so a caller checks this before calling rather than the version.
    pub site_key: bool,
    /// The revision of the declared kernels this module's cmdlets run,
    /// `flynnel::kernels::REVISION` in the build that made it. A library
    /// driving the same kernels from the crate it links answers what
    /// these cmdlets answer when the two revisions agree.
    pub kernels_revision: u32,
    /// Whether this module's worker pool has started, read without
    /// starting it. The first dispatch starts it, here or through
    /// `flynnel_run_chunks_v1`.
    pub pool_started: bool,
    /// The address of `flynnel_run_chunks_plan_v1`, which takes a plan
    /// handle from `New-FlynnelNativePlan` as its fourth argument:
    /// `(n, min_leaf, site, plan, body, ctx)`. Zero in a module without
    /// the entry.
    pub run_chunks_plan_v1: u64,
    /// Whether this module has `flynnel_run_chunks_plan_v1` and
    /// `New-FlynnelNativePlan`. A module without this property, or with it
    /// false, has neither, so a caller checks this before calling.
    pub plan_handle: bool,
}

/// Get the addresses of the module's native entry points.
///
/// For a native library in this process that wants to run work on this
/// module's worker pool rather than start a pool of its own. The library
/// PWRS loads is a renamed copy in a per-process temp folder, so an entry
/// cannot be found by name; this answers from the copy that is loaded.
///
/// An address stays callable for the life of the process, because PWRS
/// never frees a library it has loaded. It can go stale: a reload after a
/// rebuild loads a new copy at a new address, with a pool of its own, and
/// a cached address keeps running work on the old copy's pool. So a
/// caller may cache the answer, and asks again whenever the process has
/// loaded a native library since the answer was taken.
///
/// A rebuild that changes the module's cmdlets gets a load context of its
/// own, so runspaces that imported the module before it and after it run
/// different copies. A caller serving several runspaces keeps one answer
/// per runspace, taken in that runspace.
///
/// # Examples
///
/// `Get-FlynnelNativeEntry`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelNativeEntry",
    alias = "Get-FlyNativeEntry",
    output = ["Flynnel.NativeEntry"]
)]
#[derive(Default)]
pub struct GetFlynnelNativeEntry {}

impl Cmdlet for GetFlynnelNativeEntry {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(NativeEntry {
            run_chunks_v1: flynnel_run_chunks_v1 as RunChunksV1 as usize as u64,
            abi_version: NATIVE_ABI_VERSION,
            site_key: true,
            kernels_revision: flynnel::kernels::REVISION,
            pool_started: flynnel::sched::arena::global_local_arena_started(),
            run_chunks_plan_v1: flynnel_run_chunks_plan_v1 as RunChunksPlanV1 as usize as u64,
            plan_handle: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize};

    /// The site key the tests that are not about keys pass.
    const TEST_SITE: u64 = 0x7E57;

    /// Marks every index in its range once. The slice it marks is what
    /// `ctx` points at.
    extern "C" fn mark(ctx: *const c_void, start: usize, end: usize) -> i32 {
        // SAFETY: every test passes a pointer to a live Vec<AtomicU8>
        // at least `end` long, and atomics make the shared marking sound.
        let seen = unsafe { &*(ctx as *const Vec<AtomicU8>) };
        for slot in &seen[start..end] {
            slot.fetch_add(1, Ordering::Relaxed);
        }
        0
    }

    /// Returns 7 from the chunk holding index zero and 0 from every other,
    /// so exactly one chunk asks the run to stop.
    extern "C" fn stop_at_zero(_ctx: *const c_void, start: usize, _end: usize) -> i32 {
        if start == 0 { 7 } else { 0 }
    }

    /// Counts how many times it was called.
    extern "C" fn count(ctx: *const c_void, _start: usize, _end: usize) -> i32 {
        // SAFETY: the test passes a pointer to a live AtomicUsize.
        let calls = unsafe { &*(ctx as *const AtomicUsize) };
        calls.fetch_add(1, Ordering::Relaxed);
        0
    }

    fn marks(n: usize) -> Vec<AtomicU8> {
        (0..n).map(|_| AtomicU8::new(0)).collect()
    }

    #[test]
    fn every_index_runs_exactly_once() {
        let n = 10_000;
        let seen = marks(n);
        // SAFETY: `seen` outlives the call and `mark` only reads it
        // through atomics.
        let code = unsafe {
            flynnel_run_chunks_v1(
                n,
                1,
                TEST_SITE,
                mark,
                &seen as *const Vec<AtomicU8> as *const c_void,
            )
        };
        assert_eq!(code, 0);
        let wrong: Vec<usize> = (0..n)
            .filter(|&i| seen[i].load(Ordering::Relaxed) != 1)
            .collect();
        assert!(wrong.is_empty(), "indices not run exactly once: {:?}", &wrong[..wrong.len().min(20)]);
    }

    #[test]
    fn a_nonzero_return_is_the_code_reported() {
        // SAFETY: `stop_at_zero` does not read its context.
        let code =
            unsafe { flynnel_run_chunks_v1(10_000, 1, TEST_SITE, stop_at_zero, std::ptr::null()) };
        assert_eq!(code, 7);
    }

    #[test]
    fn an_empty_range_never_calls_the_body() {
        let calls = AtomicUsize::new(0);
        // SAFETY: `calls` outlives the call.
        let code = unsafe {
            flynnel_run_chunks_v1(
                0,
                1,
                TEST_SITE,
                count,
                &calls as *const AtomicUsize as *const c_void,
            )
        };
        assert_eq!(code, 0);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_floor_of_zero_is_read_as_one() {
        let n = 5;
        let seen = marks(n);
        // SAFETY: as in every_index_runs_exactly_once.
        let code = unsafe {
            flynnel_run_chunks_v1(
                n,
                0,
                TEST_SITE,
                mark,
                &seen as *const Vec<AtomicU8> as *const c_void,
            )
        };
        assert_eq!(code, 0);
        assert!((0..n).all(|i| seen[i].load(Ordering::Relaxed) == 1));
    }

    /// What a body needs to tell whether it ran on the thread that
    /// called the export.
    struct CallerCheck {
        caller: std::thread::ThreadId,
        ran_on_caller: AtomicBool,
        covered: AtomicUsize,
    }

    extern "C" fn note_thread(ctx: *const c_void, start: usize, end: usize) -> i32 {
        // SAFETY: the test passes a pointer to a live CallerCheck whose
        // fields are atomics or read-only.
        let check = unsafe { &*(ctx as *const CallerCheck) };
        if std::thread::current().id() == check.caller {
            check.ran_on_caller.store(true, Ordering::Relaxed);
        }
        check.covered.fetch_add(end - start, Ordering::Relaxed);
        0
    }

    #[test]
    fn no_body_runs_on_the_calling_thread() {
        // The test harness's thread is not one of the pool's, so it
        // stands where a foreign caller does. The sizes reach both ways a
        // dispatch run from here could put a body on this thread: the
        // probe that measures cost on a prefix, and a job small enough
        // to run inline where it started.
        for n in [1, 2, 7, 64, 1_000, 100_000] {
            let check = CallerCheck {
                caller: std::thread::current().id(),
                ran_on_caller: AtomicBool::new(false),
                covered: AtomicUsize::new(0),
            };
            // SAFETY: `check` outlives the call and note_thread reads it
            // only through atomics.
            let code = unsafe {
                flynnel_run_chunks_v1(
                    n,
                    1,
                    TEST_SITE,
                    note_thread,
                    &check as *const CallerCheck as *const c_void,
                )
            };
            assert_eq!(code, 0, "n = {n}");
            assert_eq!(check.covered.load(Ordering::Relaxed), n, "coverage at n = {n}");
            assert!(
                !check.ran_on_caller.load(Ordering::Relaxed),
                "a body ran on the calling thread at n = {n}"
            );
        }
    }

    #[test]
    fn two_keys_learn_at_two_sites() {
        let n = 100_000;
        let first = 0x7E57_0A11_u64;
        let second = 0x7E57_0B22_u64;
        for key in [first, second] {
            let seen = marks(n);
            // SAFETY: as in every_index_runs_exactly_once.
            let code = unsafe {
                flynnel_run_chunks_v1(
                    n,
                    1,
                    key,
                    mark,
                    &seen as *const Vec<AtomicU8> as *const c_void,
                )
            };
            assert_eq!(code, 0, "key {key:#x}");
        }
        let a = flynnel::site_for_key(first);
        let b = flynnel::site_for_key(second);
        assert_ne!(a, b, "two keys learn apart");
        assert!(a.get().leaf_count() > 0, "the job passed the first key learned at its site");
        assert!(b.get().leaf_count() > 0, "the job passed the second key learned at its site");
    }

    #[test]
    fn the_entry_reports_this_symbol_and_version_one() {
        assert_eq!(NATIVE_ABI_VERSION, 1);
        assert_ne!(flynnel_run_chunks_v1 as RunChunksV1 as usize, 0);
        assert_ne!(flynnel_run_chunks_plan_v1 as RunChunksPlanV1 as usize, 0);
    }

    /// A plan with the two numbers every plan needs and no hints.
    fn test_plan(k_outer: u8, batch_size: u32) -> Plan {
        Plan {
            k_outer,
            batch_size,
            ..Plan::default()
        }
    }

    #[test]
    fn a_plan_handle_runs_every_index_exactly_once() {
        let n = 100_000;
        let native = native_plan(&test_plan(10, n as u32)).expect("the plan resolves");
        let seen = marks(n);
        // SAFETY: as in every_index_runs_exactly_once.
        let code = unsafe {
            flynnel_run_chunks_plan_v1(
                n,
                1,
                TEST_SITE,
                native.handle,
                mark,
                &seen as *const Vec<AtomicU8> as *const c_void,
            )
        };
        assert_eq!(code, 0);
        assert!((0..n).all(|i| seen[i].load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn the_callers_plan_is_the_one_the_dispatch_runs() {
        // Two plans over one range whose stated costs ask for opposite
        // splits: a task priced far above the work wants one chunk, and a
        // heavy item with a free task wants about one chunk an item. The
        // chunk counts part only if each handle's plan is the one the
        // dispatch ran.
        let n = 100_000;
        let few = native_plan(&Plan {
            per_item_ns: Some(1),
            task_overhead_ns: Some(1_000_000_000),
            ..test_plan(10, n as u32)
        })
        .expect("the plan resolves");
        let many = native_plan(&Plan {
            per_item_ns: Some(100_000),
            task_overhead_ns: Some(1),
            ..test_plan(10, n as u32)
        })
        .expect("the plan resolves");
        let few_calls = AtomicUsize::new(0);
        let many_calls = AtomicUsize::new(0);
        // SAFETY: both counters outlive their calls.
        let codes = unsafe {
            (
                flynnel_run_chunks_plan_v1(
                    n,
                    1,
                    TEST_SITE,
                    few.handle,
                    count,
                    &few_calls as *const AtomicUsize as *const c_void,
                ),
                flynnel_run_chunks_plan_v1(
                    n,
                    1,
                    TEST_SITE,
                    many.handle,
                    count,
                    &many_calls as *const AtomicUsize as *const c_void,
                ),
            )
        };
        assert_eq!(codes, (0, 0));
        let (few_calls, many_calls) = (
            few_calls.load(Ordering::Relaxed),
            many_calls.load(Ordering::Relaxed),
        );
        assert!(
            many_calls > 10 * few_calls,
            "the plans split alike: {few_calls} chunks against {many_calls}"
        );
    }

    /// Adds each chunk's length to the counter `ctx` points at: the least
    /// a body can do and still touch every chunk.
    extern "C" fn light(ctx: *const c_void, start: usize, end: usize) -> i32 {
        // SAFETY: the timings pass a pointer to a live AtomicUsize.
        let total = unsafe { &*(ctx as *const AtomicUsize) };
        total.fetch_add(end - start, Ordering::Relaxed);
        0
    }

    /// Threads spinning until dropped.
    struct Spinners {
        stop: std::sync::Arc<AtomicBool>,
        threads: Vec<std::thread::JoinHandle<()>>,
    }

    impl Spinners {
        fn start(count: usize) -> Self {
            let stop = std::sync::Arc::new(AtomicBool::new(false));
            let threads = (0..count)
                .map(|_| {
                    let stop = std::sync::Arc::clone(&stop);
                    std::thread::spawn(move || {
                        while !stop.load(Ordering::Relaxed) {
                            std::hint::spin_loop();
                        }
                    })
                })
                .collect();
            Self { stop, threads }
        }
    }

    impl Drop for Spinners {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            for thread in self.threads.drain(..) {
                if let Err(panicked) = thread.join() {
                    panic!("a spinner panicked: {panicked:?}");
                }
            }
        }
    }

    /// Per-call nanoseconds of `call` at each of `rounds` rounds of
    /// `calls` calls, sorted.
    fn per_call_ns(rounds: usize, calls: usize, mut call: impl FnMut()) -> Vec<f64> {
        let mut out: Vec<f64> = (0..rounds)
            .map(|_| {
                let start = std::time::Instant::now();
                for _ in 0..calls {
                    call();
                }
                start.elapsed().as_nanos() as f64 / calls as f64
            })
            .collect();
        out.sort_by(|a, b| a.total_cmp(b));
        out
    }

    /// Prints one timing the way criterion does, an id line and then
    /// `time: [low median high]`, so the reader that reads criterion's
    /// logs reads these.
    fn print_criterion(id: &str, sorted_ns: &[f64]) {
        let low = sorted_ns[0];
        let median = sorted_ns[sorted_ns.len() / 2];
        let high = sorted_ns[sorted_ns.len() - 1];
        println!("{id}");
        println!("                        time:   [{low:.1} ns {median:.1} ns {high:.1} ns]");
    }

    /// Per-call time of `flynnel_run_chunks_v1` over a light body at three
    /// sizes, quiet and beside a spinning thread on every logical
    /// processor. Built in two trees, the rows compare two builds of the
    /// entry: run the base, the tip and the base again as the copy, one
    /// process a run:
    /// `cargo test --release --manifest-path pwrs/Cargo.toml native::tests::v1_entry_timing -- --ignored --exact --nocapture`
    #[test]
    #[ignore = "a timing, read by hand"]
    fn v1_entry_timing() {
        let logical = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .expect("logical processor count");
        for (label, spinners) in [("quiet", 0), ("all", logical)] {
            let load = Spinners::start(spinners);
            for n in [1_000usize, 100_000, 1_000_000] {
                let total = AtomicUsize::new(0);
                let ctx = &total as *const AtomicUsize as *const c_void;
                // SAFETY: `total` outlives every call and `light` reads it
                // only through an atomic.
                let mut call = || {
                    let code = unsafe { flynnel_run_chunks_v1(n, 1, TEST_SITE, light, ctx) };
                    assert_eq!(code, 0);
                };
                for _ in 0..20 {
                    call();
                }
                let ns = per_call_ns(15, 50, &mut call);
                print_criterion(&format!("chunks_{label}/v1/n{n}"), &ns);
            }
            drop(load);
        }
    }

    /// Per-call time of `flynnel_run_chunks_plan_v1` under a plan equal to
    /// the one `flynnel_run_chunks_v1` sizes to each n, beside v1 itself,
    /// the two alternating round by round in one process, quiet and beside
    /// a spinning thread on every logical processor:
    /// `cargo test --release --manifest-path pwrs/Cargo.toml native::tests::plan_entry_timing -- --ignored --exact --nocapture`
    #[test]
    #[ignore = "a timing, read by hand"]
    fn plan_entry_timing() {
        let logical = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .expect("logical processor count");
        for (label, spinners) in [("quiet", 0), ("all", logical)] {
            let load = Spinners::start(spinners);
            for n in [1_000usize, 100_000, 1_000_000] {
                let native =
                    native_plan(&test_plan(band_for(n), n as u32)).expect("the plan resolves");
                let total = AtomicUsize::new(0);
                let ctx = &total as *const AtomicUsize as *const c_void;
                // SAFETY: `total` outlives every call and `light` reads it
                // only through an atomic.
                let mut v1 = || {
                    let code = unsafe { flynnel_run_chunks_v1(n, 1, TEST_SITE, light, ctx) };
                    assert_eq!(code, 0);
                };
                // SAFETY: as for `v1`; the handle's object lives past the calls.
                let mut planned = || {
                    let code = unsafe {
                        flynnel_run_chunks_plan_v1(n, 1, TEST_SITE, native.handle, light, ctx)
                    };
                    assert_eq!(code, 0);
                };
                for _ in 0..20 {
                    v1();
                    planned();
                }
                let (mut v1_ns, mut plan_ns) = (Vec::new(), Vec::new());
                for round in 0..15 {
                    if round % 2 == 0 {
                        v1_ns.extend(per_call_ns(1, 50, &mut v1));
                        plan_ns.extend(per_call_ns(1, 50, &mut planned));
                    } else {
                        plan_ns.extend(per_call_ns(1, 50, &mut planned));
                        v1_ns.extend(per_call_ns(1, 50, &mut v1));
                    }
                }
                v1_ns.sort_by(|a, b| a.total_cmp(b));
                plan_ns.sort_by(|a, b| a.total_cmp(b));
                print_criterion(&format!("chunks_{label}/v1/n{n}"), &v1_ns);
                print_criterion(&format!("chunks_{label}/plan/n{n}"), &plan_ns);
            }
            drop(load);
        }
    }

    #[test]
    fn a_released_or_unknown_handle_is_refused_and_runs_no_body() {
        let calls = AtomicUsize::new(0);
        let native = native_plan(&test_plan(10, 1_000)).expect("the plan resolves");
        let released = native.handle;
        drop(native);
        for handle in [released, 0] {
            // SAFETY: `calls` outlives the call.
            let code = unsafe {
                flynnel_run_chunks_plan_v1(
                    1_000,
                    1,
                    TEST_SITE,
                    handle,
                    count,
                    &calls as *const AtomicUsize as *const c_void,
                )
            };
            assert_eq!(code, RUN_CHUNKS_NO_PLAN, "handle {handle}");
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}

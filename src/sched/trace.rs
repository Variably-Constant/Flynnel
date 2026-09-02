//! Per-worker TSC-event trace facility for one-shot scheduler debugging.
//!
//! Each Flynnel worker (and the external caller thread) accumulates
//! `(event_kind, payload, tsc)` tuples into a thread-local buffer at
//! pre-defined hook points. After a dispatch completes,
//! [`dump_to_stderr`] serializes every recorded buffer to stderr as
//! a CSV stream (one row per event) so the timing of the dispatch
//! can be reconstructed offline.
//!
//! # Enabling
//!
//! Off by default. Set environment variable `FLYNNEL_TRACE=1` at
//! process start. The first [`is_enabled`] call latches the decision
//! via `OnceLock<bool>`; subsequent calls are a single cached load.
//!
//! # Cost when off
//!
//! Single Relaxed atomic load + branch (`if !is_enabled() { return; }`).
//! Per recorded event when on: one `read_tsc()` (~20 cycles) + one
//! `Vec::push` on a thread-local buffer (amortized O(1)). The
//! thread-local sits behind `RefCell` so the borrow is cheap and
//! single-threaded; no atomics on the recording path.
//!
//! # Why a binary (not the bench harness)
//!
//! Criterion runs hundreds of iterations per measurement; tracing
//! every iteration produces gigabytes of CSV. The companion
//! `examples/trace_heavy_dispatch.rs` binary runs ONE dispatch with
//! `FLYNNEL_TRACE=1`, dumps the trace, then exits.

use core::cell::RefCell;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Event kinds emitted at the instrumented hook points. Compact `u8`
/// so the per-event memory cost stays low.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TraceEvent {
    /// `for_each_chunk` entry. Payload = total item count.
    DispatchEnter = 1,
    /// `for_each_chunk` return. Payload = total item count.
    DispatchExit = 2,
    /// A leaf body is about to execute. Payload = leaf item count.
    LeafStart = 3,
    /// The leaf body completed. Payload = leaf item count.
    LeafEnd = 4,
    /// `join_in_worker` pushed the right half to the deque. Payload = 0.
    JoinPush = 5,
    /// `join_in_worker` began its drain loop waiting for the right
    /// half to finish. Payload = 0.
    JoinWaitBegin = 6,
    /// `join_in_worker` returned (right half done). Payload = 0.
    JoinWaitEnd = 7,
    /// A worker thread woke up from park to find work. Payload =
    /// worker_id.
    WorkerWake = 8,
    /// A worker successfully stole from a peer. Payload = victim
    /// worker_id.
    StealHit = 9,
}

/// One trace event row recorded into the per-thread buffer.
#[derive(Copy, Clone, Debug)]
pub struct TraceRecord {
    /// Which instrumentation point emitted this row.
    pub event: TraceEvent,
    /// Per-event integer payload (item count, worker id, etc.).
    pub payload: u32,
    /// Raw TSC value captured at emission time.
    pub tsc: u64,
}

thread_local! {
    /// Per-thread trace buffer. Pre-grown to 16k slots so the first
    /// few thousand events do not pay alloc cost. Buffer is fully
    /// owned by this thread; no atomics on the hot path.
    static THREAD_TRACE: RefCell<Vec<TraceRecord>> =
        RefCell::new(Vec::with_capacity(16_384));
}

/// Process-wide enable flag. Latched on first call to [`is_enabled`]
/// from `FLYNNEL_TRACE` env var.
static TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

/// Strictly-monotonic id for `register_thread`, so the dump can
/// associate each recorded buffer with a thread name.
static THREAD_ID_GEN: AtomicU64 = AtomicU64::new(0);

/// Process-wide flag that the worker_loop reads at the top of each
/// iteration. When set, the worker flushes its trace buffer to
/// stderr (tagged with `flynnel-worker-{idx}`) and clears its
/// thread-local. Use [`request_worker_flush`] to set; the workers
/// auto-clear after dumping so consecutive flushes work.
static WORKER_FLUSH_FLAG: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Signal all worker_loop iterations to dump their trace buffers
/// on the next loop pass. Workers check the flag at the top of
/// each iteration and dump+clear if set. The flag is consumed
/// per-thread via a thread-local last-seen counter so all workers
/// observe each flush request exactly once.
pub fn request_worker_flush() {
    WORKER_FLUSH_FLAG.store(true, core::sync::atomic::Ordering::Release);
}

thread_local! {
    /// Whether THIS worker thread has handled the current flush
    /// request. Reset whenever a fresh request arrives (i.e. when
    /// the global flag transitions to true while ours is also
    /// true, we re-flush).
    static FLUSH_OBSERVED: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Worker_loop hook: check whether a flush has been requested. If
/// yes AND this thread hasn't already handled this request, dump
/// the trace buffer (tagged with `label`) and mark this thread
/// done. Returns true if a dump happened.
pub fn worker_loop_maybe_flush(label: &str) -> bool {
    if !is_enabled() {
        return false;
    }
    let requested = WORKER_FLUSH_FLAG.load(core::sync::atomic::Ordering::Acquire);
    let already_done = FLUSH_OBSERVED.with(|c| c.get());
    if requested && !already_done {
        dump_to_stderr(label);
        FLUSH_OBSERVED.with(|c| c.set(true));
        return true;
    }
    if !requested && already_done {
        // Caller cleared the request; reset our per-thread state
        // so the next request can fire.
        FLUSH_OBSERVED.with(|c| c.set(false));
    }
    false
}

/// Caller-side: reset the request after a dump cycle so subsequent
/// requests can fire.
pub fn clear_worker_flush_request() {
    WORKER_FLUSH_FLAG.store(false, core::sync::atomic::Ordering::Release);
}

/// Returns true if the `FLYNNEL_TRACE` env var was set to a truthy
/// value at process startup. Latched on first call via `OnceLock`.
#[inline]
pub fn is_enabled() -> bool {
    *TRACE_ENABLED.get_or_init(|| {
        std::env::var("FLYNNEL_TRACE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[inline]
fn read_tsc_local() -> u64 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_rdtsc` is part of the base x86_64 ISA (Pentium and
    // newer); the intrinsic has no CPU-feature preconditions and
    // no architectural preconditions on operand state. The
    // `#[cfg(target_arch = "x86_64")]` guard above is the only
    // gate the call site needs to be sound.
    unsafe {
        std::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::time::Instant::now().elapsed().as_nanos() as u64
    }
}

/// Record one event on the calling thread's trace buffer. Cheap
/// fast-exit when tracing is disabled.
#[inline]
pub fn emit(event: TraceEvent, payload: u32) {
    if !is_enabled() {
        return;
    }
    let tsc = read_tsc_local();
    let rec = TraceRecord { event, payload, tsc };
    THREAD_TRACE.with(|buf| {
        let mut v = buf.borrow_mut();
        v.push(rec);
    });
}

/// Reset all thread buffers known to this thread (and our own).
/// Call before a dispatch you want to trace cleanly.
pub fn reset_current_thread() {
    THREAD_TRACE.with(|buf| {
        let mut v = buf.borrow_mut();
        v.clear();
    });
}

/// Per-thread registration. Worker threads call this once at startup
/// so the dump knows who they are. Returns an integer id used in
/// the dumped CSV.
pub fn register_thread() -> u64 {
    THREAD_ID_GEN.fetch_add(1, Ordering::Relaxed)
}

/// Dump THIS thread's trace buffer to stderr as CSV. Other workers
/// have to flush themselves before the dump can include them; the
/// worker loop has a "trace dump" path that writes its buffer on
/// demand.
///
/// CSV columns: thread_name,event_kind_num,payload,tsc
pub fn dump_to_stderr(thread_name: &str) {
    if !is_enabled() {
        return;
    }
    THREAD_TRACE.with(|buf| {
        let v = buf.borrow();
        for rec in v.iter() {
            eprintln!(
                "TRACE,{thread},{ev},{payload},{tsc}",
                thread = thread_name,
                ev = rec.event as u8,
                payload = rec.payload,
                tsc = rec.tsc,
            );
        }
    });
}

/// Worker-loop hook: each worker's loop calls this after it runs a
/// job so the buffer is flushed at a known quiescent point. The
/// `name` is `flynnel-worker-{idx}` typically.
///
/// In practice the example binary triggers a process-wide flush by
/// calling this on every worker via a barrier - see
/// `examples/trace_heavy_dispatch.rs`.
pub fn flush_with_label(label: &str) {
    dump_to_stderr(label);
    reset_current_thread();
}

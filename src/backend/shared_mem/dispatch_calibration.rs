//! Per-host calibration of the [`super::variant_dispatch`] routing
//! table. Re-runs each variant's micro-bench on this host and writes
//! the per-shape winning variant into the table in place.
//!
//! ## Mechanism
//!
//! Each `WorkloadShape` cell to calibrate has up to four candidate
//! variants (whichever backends the dispatcher has installed). For
//! each candidate, [`measure_variant_ns_per_call`] runs a small
//! producer-fast micro-bench and records the average per-call cost.
//! The variant with the lowest cost wins the cell.
//!
//! Calibration is bounded: each measurement runs ~1k iterations
//! after a 100-iter warm-up, total ~milliseconds per variant. The
//! whole calibration sweep across the default heuristic cell set
//! takes well under a second on Zen+.
//!
//! ## Composition with `bg_calibration`
//!
//! The scheduler's [`crate::sched::bg_calibration`] module exposes
//! [`crate::sched::bg_calibration::spawn_calibration`], which
//! schedules caller-supplied closures on the SMT-sibling
//! [`crate::sched::io_pool::IoPool`]. Production setups can wrap
//! [`calibrate_routing_table`] in a `spawn_calibration` closure so
//! the calibration runs once at startup on the IO pool, leaving the
//! compute pool free.
//!
//! The bare entrypoint [`calibrate_routing_table`] is synchronous +
//! takes a `&mut DispatcherRoutingTable`; it makes no assumption
//! about which pool it runs on.

#![allow(clippy::missing_errors_doc)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::variant_dispatch::{DequeVariant, DispatcherRoutingTable, WorkloadShape};
use super::{
    SharedMemoryChaseLevBackend, SharedMemoryKhpdBackend, SharedMemoryLohBackend,
    SharedMemoryUrdBackend, hash_name, register, unregister,
};
use crate::backend::BackendError;

/// Convert the `wait_handle` result shape into `Option<()>` so the
/// calibration helpers can use `?` for early exit on failure. Both
/// failure modes (backend-level `BackendError`, handler-reported
/// `String` error) are logged before they are collapsed - silently
/// swallowing them was the bug pattern this helper replaces. Returns
/// `Some(())` only when the handler completed without error.
fn calib_wait(r: Result<Result<Vec<u8>, String>, BackendError>) -> Option<()> {
    match r {
        Ok(Ok(_reply)) => Some(()),
        Ok(Err(handler_err)) => {
            eprintln!("calib: handler reported error: {handler_err}");
            None
        }
        Err(backend_err) => {
            eprintln!("calib: backend error: {backend_err}");
            None
        }
    }
}

/// Outcome of a single variant measurement for one workload shape.
#[derive(Debug, Clone, Copy)]
pub struct VariantMeasurement {
    /// Which variant was measured.
    pub variant: DequeVariant,
    /// Average per-call cost in nanoseconds. `None` when the
    /// measurement could not run (e.g., the variant's backend file
    /// could not be created on this host's tmp dir).
    pub ns_per_call: Option<f64>,
}

/// Full calibration report for one `WorkloadShape`: the per-variant
/// measurements + the variant the table is set to after calibration.
#[derive(Debug, Clone)]
pub struct CellCalibration {
    /// The shape calibrated.
    pub shape: WorkloadShape,
    /// One entry per measured variant; variants with `None` ns were
    /// skipped (could not build a backend on this host).
    pub measurements: Vec<VariantMeasurement>,
    /// The variant the routing table was updated to. `None` when no
    /// variant could be measured (the cell is left at the heuristic
    /// default).
    pub winner: Option<DequeVariant>,
}

/// Identifier under which the calibration micro-bench registers its
/// `(u32, u32) -> u32` adder handler. Distinct per-(pid, variant,
/// nonce) so concurrent calibrations across different variants do
/// not collide on `register`/`unregister`. The earlier per-pid-only
/// name caused a race under parallel-test load: measure_chase_lev's
/// terminal `unregister(id)` stripped the shared handler while
/// measure_loh's worker was still draining slots referencing it,
/// and the LOH worker reported "no handler for closure id" mid-burst.
fn calib_handler_name(variant: DequeVariant, nonce: u64) -> String {
    format!(
        "flynnel.calib.dispatch.adder.{}.{}.{nonce}",
        std::process::id(),
        variant.label(),
    )
}

/// Build the temp file paths for one calibration backend. Each
/// backend needs its own pair of MMF files.
fn temp_pair(variant: DequeVariant, suffix: u64) -> (PathBuf, PathBuf) {
    let mut d = std::env::temp_dir();
    let mut l = std::env::temp_dir();
    let pid = std::process::id();
    d.push(format!("flynnel_calib_{pid}_{suffix}_{}_d.bin", variant.label()));
    l.push(format!("flynnel_calib_{pid}_{suffix}_{}_l.bin", variant.label()));
    (d, l)
}

/// Make a `(u32, u32) -> u32` payload for the adder handler.
fn adder_payload(a: u32, b: u32) -> [u8; 8] {
    let mut p = [0u8; 8];
    p[..4].copy_from_slice(&a.to_le_bytes());
    p[4..].copy_from_slice(&b.to_le_bytes());
    p
}

/// Measure one variant on one workload shape. Spawns the variant's
/// backend on temp files, registers an adder handler, runs the
/// micro-bench, then tears down.
///
/// Returns `None` for a variant that cannot be measured on this host
/// (e.g., its backend file create call failed) or for a variant that
/// cannot serve the shape (e.g., args don't fit).
///
/// `iters` is the number of measured iterations (after a small
/// warm-up). 1024 iters typical; calibrate-callers can lower it for
/// quick sanity checks.
pub fn measure_variant_ns_per_call(
    variant: DequeVariant,
    shape: &WorkloadShape,
    iters: u32,
) -> Option<f64> {
    if (shape.args_inline_bytes as usize) > variant.inline_args_bytes() {
        return None;
    }
    let payload = adder_payload(3, 4);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    // Per-(variant, nonce) handler isolates this measurement from any
    // other calibration measurement running in parallel - critical
    // under cargo-test default parallelism where multiple measure_*
    // calls execute concurrently and would otherwise race on a
    // pid-shared handler id.
    let handler = calib_handler_name(variant, nonce);
    let id = hash_name(&handler);
    register(id, |args| {
        let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
        let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
        Ok(a.wrapping_add(b).to_le_bytes().to_vec())
    });

    // Per-variant measurement closure: builds the backend, runs the
    // bench, returns ns/call.
    let ns = match variant {
        DequeVariant::ChaseLev => measure_chase_lev(id, &payload, shape, iters, nonce),
        DequeVariant::Loh => measure_loh(id, &payload, shape, iters, nonce),
        DequeVariant::Khpd => measure_khpd(id, &payload, shape, iters, nonce),
        DequeVariant::Urd => measure_urd(id, &payload, shape, iters, nonce),
    };

    unregister(id);
    ns
}

fn measure_chase_lev(
    id: u32,
    payload: &[u8; 8],
    shape: &WorkloadShape,
    iters: u32,
    nonce: u64,
) -> Option<f64> {
    let (dp, lp) = temp_pair(DequeVariant::ChaseLev, nonce);
    let be = Arc::new(
        SharedMemoryChaseLevBackend::create(0, &dp, &lp, 256, 1024).ok()?,
    );
    let stop = Arc::new(AtomicBool::new(false));
    let wbe = Arc::clone(&be);
    let wstop = Arc::clone(&stop);
    let w = std::thread::spawn(move || {
        while !wstop.load(Ordering::Relaxed) {
            match wbe.drain_one() {
                Ok(Some(())) => {}
                // yield_now (not spin_loop) so calibration workers
                // cooperate with the test runner's many parallel
                // threads; a spin_loop here starves the drain
                // under a loaded runner.
                Ok(None) => std::thread::yield_now(),
                Err(_) => return,
            }
        }
    });

    // Warm-up.
    for _ in 0..64 {
        let h = be.dispatch_marshal(id, payload).ok()?;
        calib_wait(be.wait_handle(h, 1024))?;
    }
    // Measure: for batch shapes, dispatch `expected_burst_size` items
    // per outer iter and only wait on the last one.
    let burst = shape.expected_burst_size.max(1) as usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut last = None;
        for _ in 0..burst {
            last = Some(be.dispatch_marshal(id, payload).ok()?);
        }
        if let Some(h) = last {
            calib_wait(be.wait_handle(h, 1024))?;
        }
    }
    let elapsed = t0.elapsed();
    let total_calls = (iters as u64) * (burst as u64);
    stop.store(true, Ordering::Relaxed);
    w.join().ok();
    std::fs::remove_file(&dp).ok();
    std::fs::remove_file(&lp).ok();
    Some(elapsed.as_nanos() as f64 / total_calls as f64)
}

fn measure_loh(
    id: u32,
    payload: &[u8; 8],
    shape: &WorkloadShape,
    iters: u32,
    nonce: u64,
) -> Option<f64> {
    let burst = shape.expected_burst_size.max(1) as usize;
    let (dp, lp) = temp_pair(DequeVariant::Loh, nonce);
    let be = Arc::new(
        SharedMemoryLohBackend::create(0, &dp, &lp, 1024, 2048, burst).ok()?,
    );
    let stop = Arc::new(AtomicBool::new(false));
    let wbe = Arc::clone(&be);
    let wstop = Arc::clone(&stop);
    let w = std::thread::spawn(move || {
        while !wstop.load(Ordering::Relaxed) {
            match wbe.drain_one() {
                Ok(Some(())) => {}
                // yield_now (not spin_loop) so calibration workers
                // cooperate with the test runner's many parallel
                // threads; a spin_loop here starves the drain
                // under a loaded runner.
                Ok(None) => std::thread::yield_now(),
                Err(_) => return,
            }
        }
    });
    let items: Vec<(u32, &[u8])> = (0..burst).map(|_| (id, payload.as_slice())).collect();
    for _ in 0..8 {
        let handles = be.dispatch_marshal_batch(&items).ok()?;
        if let Some(last) = handles.last() {
            calib_wait(be.wait_handle(*last, 1024))?;
        }
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let handles = be.dispatch_marshal_batch(&items).ok()?;
        if let Some(last) = handles.last() {
            calib_wait(be.wait_handle(*last, 1024))?;
        }
    }
    let elapsed = t0.elapsed();
    let total_calls = (iters as u64) * (burst as u64);
    stop.store(true, Ordering::Relaxed);
    w.join().ok();
    std::fs::remove_file(&dp).ok();
    std::fs::remove_file(&lp).ok();
    Some(elapsed.as_nanos() as f64 / total_calls as f64)
}

fn measure_khpd(
    id: u32,
    payload: &[u8; 8],
    shape: &WorkloadShape,
    iters: u32,
    nonce: u64,
) -> Option<f64> {
    let burst = shape.expected_burst_size.max(1) as usize;
    let (dp, lp) = temp_pair(DequeVariant::Khpd, nonce);
    let be = Arc::new(
        SharedMemoryKhpdBackend::create(0, &dp, &lp, 256, 2048).ok()?,
    );
    let stop = Arc::new(AtomicBool::new(false));
    let wbe = Arc::clone(&be);
    let wstop = Arc::clone(&stop);
    let w = std::thread::spawn(move || {
        while !wstop.load(Ordering::Relaxed) {
            match wbe.drain_one_line() {
                Ok(Some(_)) => {}
                // yield_now (not spin_loop) so calibration workers
                // cooperate with the test runner's many parallel
                // threads; a spin_loop here starves the drain
                // under a loaded runner.
                Ok(None) => std::thread::yield_now(),
                Err(_) => return,
            }
        }
    });
    let items: Vec<(u32, &[u8])> = (0..burst).map(|_| (id, payload.as_slice())).collect();
    for _ in 0..8 {
        let handles = be.dispatch_marshal_batch(&items).ok()?;
        if let Some(last) = handles.last() {
            calib_wait(be.wait_handle(*last, 1024))?;
        }
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let handles = be.dispatch_marshal_batch(&items).ok()?;
        if let Some(last) = handles.last() {
            calib_wait(be.wait_handle(*last, 1024))?;
        }
    }
    let elapsed = t0.elapsed();
    let total_calls = (iters as u64) * (burst as u64);
    stop.store(true, Ordering::Relaxed);
    w.join().ok();
    std::fs::remove_file(&dp).ok();
    std::fs::remove_file(&lp).ok();
    Some(elapsed.as_nanos() as f64 / total_calls as f64)
}

fn measure_urd(
    id: u32,
    payload: &[u8; 8],
    shape: &WorkloadShape,
    iters: u32,
    nonce: u64,
) -> Option<f64> {
    let n_thieves = shape.n_drain_threads.max(1) as usize;
    let burst = shape.expected_burst_size.max(1) as usize;
    let (dp, lp) = temp_pair(DequeVariant::Urd, nonce);
    let be = Arc::new(
        SharedMemoryUrdBackend::create(0, &dp, &lp, n_thieves, 2048).ok()?,
    );
    let stop = Arc::new(AtomicBool::new(false));
    let mut workers = Vec::with_capacity(n_thieves);
    for thief in 0..n_thieves {
        let wbe = Arc::clone(&be);
        let wstop = Arc::clone(&stop);
        workers.push(std::thread::spawn(move || {
            while !wstop.load(Ordering::Relaxed) {
                match wbe.drain_mailbox(thief) {
                    Ok(Some(_)) => {}
                    // yield_now (not spin_loop) so calibration workers
                    // cooperate with the test runner's many parallel
                    // threads.
                    Ok(None) => std::thread::yield_now(),
                    Err(_) => return,
                }
            }
        }));
    }
    let items: Vec<(u32, &[u8])> = (0..burst).map(|_| (id, payload.as_slice())).collect();
    for _ in 0..8 {
        let handles = be.dispatch_marshal_batch(&items).ok()?;
        if let Some(last) = handles.last() {
            calib_wait(be.wait_handle(*last, 1024))?;
        }
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let handles = be.dispatch_marshal_batch(&items).ok()?;
        if let Some(last) = handles.last() {
            calib_wait(be.wait_handle(*last, 1024))?;
        }
    }
    let elapsed = t0.elapsed();
    let total_calls = (iters as u64) * (burst as u64);
    stop.store(true, Ordering::Relaxed);
    for w in workers {
        w.join().ok();
    }
    std::fs::remove_file(&dp).ok();
    std::fs::remove_file(&lp).ok();
    Some(elapsed.as_nanos() as f64 / total_calls as f64)
}

/// Calibrate one cell: measure each candidate variant on `shape` and
/// write the winner into `table`. Returns the per-variant measurements
/// + the chosen winner.
pub fn calibrate_cell(
    table: &mut DispatcherRoutingTable,
    shape: WorkloadShape,
    candidates: &[DequeVariant],
    iters: u32,
) -> CellCalibration {
    let mut measurements = Vec::with_capacity(candidates.len());
    let mut best: Option<(DequeVariant, f64)> = None;
    for &variant in candidates {
        let ns = measure_variant_ns_per_call(variant, &shape, iters);
        if let Some(ns) = ns
            && (best.is_none() || ns < best.unwrap().1)
        {
            best = Some((variant, ns));
        }
        measurements.push(VariantMeasurement {
            variant,
            ns_per_call: ns,
        });
    }
    let winner = best.map(|(v, _)| v);
    if let Some(v) = winner {
        table.update_cell(shape, v);
    }
    CellCalibration {
        shape,
        measurements,
        winner,
    }
}

/// Calibrate the default heuristic cell set: the three shapes the
/// per-variant benches measured. Returns a per-cell report.
///
/// Time budget: ~3 cells x 4 variants x 1024 iters x ~1us per call
/// = ~12 ms on a representative host (smaller benches; the iters
/// parameter is tunable per call).
pub fn calibrate_routing_table(
    table: &mut DispatcherRoutingTable,
    iters: u32,
) -> Vec<CellCalibration> {
    let shapes = [
        WorkloadShape::request_reply(8),
        WorkloadShape::producer_fast(8, 64),
        WorkloadShape::multi_thief(8, 4, 64),
    ];
    let all_candidates = [
        DequeVariant::ChaseLev,
        DequeVariant::Loh,
        DequeVariant::Khpd,
        DequeVariant::Urd,
    ];
    let mut reports = Vec::with_capacity(shapes.len());
    for shape in shapes {
        reports.push(calibrate_cell(table, shape, &all_candidates, iters));
    }
    reports
}

/// Best-effort time budget for one full `calibrate_routing_table`
/// sweep at the given iter count. Returned as a `Duration` so
/// callers can reason about start-up cost.
pub fn estimate_calibration_budget(iters: u32) -> Duration {
    // Empirical floor: ~1us per call x iters x 3 shapes x 4 variants
    // + ~5ms per backend spin-up.
    let per_variant_ns = (iters as u64).saturating_mul(1_000);
    let total_ns = per_variant_ns.saturating_mul(3 * 4);
    Duration::from_nanos(total_ns) + Duration::from_millis(60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measure_chase_lev_request_reply_returns_finite_ns() {
        // Small iter count to keep the test fast; we just need to
        // confirm the path runs end-to-end and produces a finite
        // measurement.
        let shape = WorkloadShape::request_reply(8);
        let ns = measure_variant_ns_per_call(DequeVariant::ChaseLev, &shape, 64);
        let ns = ns.expect("chase-lev measurement must succeed");
        assert!(ns.is_finite() && ns > 0.0, "ns must be finite and positive, got {ns}");
    }

    #[test]
    fn measure_loh_producer_fast_returns_finite_ns() {
        let shape = WorkloadShape::producer_fast(8, 8);
        let ns = measure_variant_ns_per_call(DequeVariant::Loh, &shape, 32);
        let ns = ns.expect("loh measurement must succeed");
        assert!(ns.is_finite() && ns > 0.0);
    }

    #[test]
    fn measure_oversize_args_returns_none() {
        // 16 B args don't fit KHPD's 8 B ceiling; measurement must
        // skip cleanly.
        let shape = WorkloadShape {
            n_drain_threads: 1,
            args_inline_bytes: 16,
            expected_burst_size: 1,
            k_unified: 0,
            k_hardware_class: 0,
        };
        let ns = measure_variant_ns_per_call(DequeVariant::Khpd, &shape, 32);
        assert!(ns.is_none(), "oversize-args measurement must return None");
    }

    #[test]
    fn calibrate_cell_picks_some_winner_when_chase_lev_available() {
        let mut table = DispatcherRoutingTable::empty();
        let shape = WorkloadShape::request_reply(8);
        let report = calibrate_cell(
            &mut table,
            shape,
            &[DequeVariant::ChaseLev],
            32,
        );
        assert!(report.winner.is_some(), "must pick a winner");
        // After calibration the table cell is set to the winner.
        assert_eq!(
            table.pick(&shape),
            report.winner.expect("winner set"),
            "table cell must agree with reported winner"
        );
    }

    #[test]
    fn calibration_budget_is_reasonable_for_small_iters() {
        let budget = estimate_calibration_budget(64);
        // At iters=64 the projected time is below 1 second on any
        // representative host.
        assert!(budget < Duration::from_secs(1),
            "calibration budget for iters=64 should be < 1s, got {budget:?}");
    }
}

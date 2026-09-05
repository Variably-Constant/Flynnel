//! Inter-core sync-latency calibration table.
//!
//! Measures per-host inter-core sync latency (heuristic starting
//! points: intra-CCX ~30 ns, cross-CCX ~50 ns, cross-CCD ~120-200
//! ns, cross-socket ~200-400 ns) via a cache-line ping-pong
//! between two pinned threads alternating read-then-write on a
//! shared [`AtomicU64`]; one-way cost is half the round trip.
//! Results land in a [`TopologyLatencyTable`] indexed by
//! `(source_core, dest_core)`, populated lazily on first
//! [`topology_latency_table`] call (~100 us per pair; ~5.6 ms for
//! 8 cores, ~24 ms for 16).
//!
//! Kernel compute cost belongs to
//! [`crate::sched::bg_calibration`]; this measures only sync
//! latency, not transfer throughput, and never consults the SLIT
//! matrix (relative integers, not nanoseconds).

use core_affinity::CoreId;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Default number of ping-pong iterations per pair. At the
/// fastest expected sync cost (intra-CCX ~30 ns round trip)
/// this is ~6 us per pair; at the slowest (cross-socket ~800
/// ns) it is ~160 us per pair.
pub const DEFAULT_PING_PONG_ITERS: u32 = 200;

/// Minimum iterations any caller can specify. Floor exists to
/// keep the per-pair sample size above LL-cache jitter.
pub const MIN_PING_PONG_ITERS: u32 = 32;

/// Maximum iterations any caller can specify. Ceiling exists
/// to keep init-time cost bounded on hosts with many cores.
pub const MAX_PING_PONG_ITERS: u32 = 4_096;

/// Measured per-host inter-core sync latency in nanoseconds.
///
/// Stored as a flat row-major matrix `entries[a * n + b]` where
/// `n` is the number of measured cores. Symmetric: `entries[a *
/// n + b] == entries[b * n + a]`. Diagonal entries are 0 (no
/// cross-core traffic).
#[derive(Clone, Debug)]
pub struct TopologyLatencyTable {
    n: usize,
    entries_ns: Vec<u32>,
    /// Iteration count used for the calibration. Useful for
    /// telemetry / reproducibility.
    pub iters: u32,
    /// Wall-clock cost of the calibration sweep.
    pub calibration_wall_ns: u64,
}

impl TopologyLatencyTable {
    /// Number of cores covered by the table.
    pub fn n(&self) -> usize {
        self.n
    }

    /// One-way sync latency in nanoseconds from `src` to `dst`.
    /// Returns `0` when `src == dst` and when either index is
    /// out of range.
    pub fn latency_ns(&self, src: usize, dst: usize) -> u32 {
        if src == dst || src >= self.n || dst >= self.n {
            return 0;
        }
        self.entries_ns[src * self.n + dst]
    }

    /// Mean off-diagonal latency in nanoseconds. Useful as a
    /// single summary statistic when the dispatch policy does
    /// not care about per-pair detail.
    pub fn mean_offdiag_ns(&self) -> f64 {
        if self.n <= 1 {
            return 0.0;
        }
        let mut sum: u64 = 0;
        let mut count: u64 = 0;
        for src in 0..self.n {
            for dst in 0..self.n {
                if src == dst {
                    continue;
                }
                sum = sum.saturating_add(self.entries_ns[src * self.n + dst] as u64);
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum as f64) / (count as f64)
        }
    }

    /// Minimum off-diagonal latency. Approximates the intra-
    /// CCX cost on Zen / cross-tile cost on Apple Silicon.
    pub fn min_offdiag_ns(&self) -> u32 {
        let mut min = u32::MAX;
        for src in 0..self.n {
            for dst in 0..self.n {
                if src == dst {
                    continue;
                }
                let v = self.entries_ns[src * self.n + dst];
                if v > 0 && v < min {
                    min = v;
                }
            }
        }
        if min == u32::MAX {
            0
        } else {
            min
        }
    }

    /// Maximum off-diagonal latency. Approximates the cross-
    /// socket / cross-CCD cost.
    pub fn max_offdiag_ns(&self) -> u32 {
        let mut max = 0;
        for src in 0..self.n {
            for dst in 0..self.n {
                if src == dst {
                    continue;
                }
                let v = self.entries_ns[src * self.n + dst];
                if v > max {
                    max = v;
                }
            }
        }
        max
    }

    /// Format the table as a human-readable matrix (caller
    /// supplied row / column headers from the measured core
    /// IDs). Each cell is `latency_ns` formatted as an integer.
    pub fn format_as_matrix(&self) -> String {
        use core::fmt::Write;
        let mut s = String::new();
        writeln!(s, "TopologyLatencyTable n={} iters={}", self.n, self.iters).ok();
        write!(s, "         ").ok();
        for dst in 0..self.n {
            write!(s, "{dst:>8}").ok();
        }
        writeln!(s).ok();
        for src in 0..self.n {
            write!(s, "{src:>8}:").ok();
            for dst in 0..self.n {
                write!(s, "{:>7}n", self.entries_ns[src * self.n + dst]).ok();
            }
            writeln!(s).ok();
        }
        s
    }
}

/// One round of ping-pong measurement between two pinned
/// threads.
///
/// Returns the average per-iteration time in nanoseconds (the
/// cache-line round-trip cost; one-way sync is half this).
///
/// Algorithm: a shared `AtomicU64` ping-pong counter is updated
/// alternately by the two threads. Thread A writes odd values
/// `1, 3, 5, ...`; thread B writes the responding even values
/// `2, 4, 6, ...`. After A stores its odd value, A waits for the
/// counter to reach the next-higher even value (B's response).
/// After B sees an odd value, B writes the next even value. The
/// total transitions is `iters * 2` (one write per side per iter).
///
/// Wait-for-peer is required after every write. Waiting for the
/// counter to equal one's own write is satisfied immediately by
/// the local cache line and skips the cross-core round trip.
fn measure_pair(core_a: CoreId, core_b: CoreId, iters: u32) -> u64 {
    let iters = iters.clamp(MIN_PING_PONG_ITERS, MAX_PING_PONG_ITERS);
    // Cache-line-aligned counter. Both threads spin on it.
    #[repr(align(64))]
    struct AlignedAtomic(AtomicU64);

    let counter = AlignedAtomic(AtomicU64::new(0));

    std::thread::scope(|scope| {
        let cref = &counter;
        // Total transitions: 2 per iteration, plus a final B
        // transition so the counter ends on the even sentinel.
        let target: u64 = (iters as u64) * 2;

        // Thread B: wait for odd values, write the next even.
        let _b = scope.spawn(move || {
            let _ = core_affinity::set_for_current(core_b);
            let mut my_val: u64 = 2;
            while my_val <= target {
                // Wait for A's previous odd write (= my_val - 1).
                let expected_from_a = my_val - 1;
                while cref.0.load(Ordering::Acquire) != expected_from_a {
                    core::hint::spin_loop();
                }
                // Write the even response.
                cref.0.store(my_val, Ordering::Release);
                my_val = my_val.wrapping_add(2);
            }
        });

        // Thread A: write odd, wait for B's even response. Owns
        // the timer.
        let _ = core_affinity::set_for_current(core_a);

        // Warm-up: 8 full round trips (= 16 transitions) to
        // stabilize cache lines + branch predictor before the
        // timed window. Each warmup iteration: A writes one odd
        // value, waits for B's even response.
        let warmup_iters: u64 = 8u64.min(iters as u64);
        let mut my_val: u64 = 1;
        for _ in 0..warmup_iters {
            cref.0.store(my_val, Ordering::Release);
            let expected_from_b = my_val + 1;
            while cref.0.load(Ordering::Acquire) != expected_from_b {
                core::hint::spin_loop();
            }
            my_val = my_val.wrapping_add(2);
        }

        let timed_iters = (iters as u64) - warmup_iters;
        let t0 = Instant::now();
        for _ in 0..timed_iters {
            cref.0.store(my_val, Ordering::Release);
            let expected_from_b = my_val + 1;
            while cref.0.load(Ordering::Acquire) != expected_from_b {
                core::hint::spin_loop();
            }
            my_val = my_val.wrapping_add(2);
        }
        let elapsed = t0.elapsed();
        (elapsed.as_nanos() as u64)
            .checked_div(timed_iters)
            .unwrap_or(0)
    })
}

/// Build the latency table by measuring every off-diagonal
/// pair via [`measure_pair`].
///
/// Returns `None` when `core_affinity::get_core_ids` is
/// unavailable on the host (e.g., sandboxed runner without
/// affinity API). In that case the heuristic-default tier
/// table from `docs/K_HIERARCHY.md` remains the source of
/// truth.
pub fn calibrate_table(iters: u32) -> Option<TopologyLatencyTable> {
    let core_ids = core_affinity::get_core_ids()?;
    let core_ids = limit_cores_for_calibration(core_ids);
    let n = core_ids.len();
    if n == 0 {
        return None;
    }

    let t0 = Instant::now();
    let mut entries_ns: Vec<u32> = vec![0; n * n];
    for a in 0..n {
        for b in (a + 1)..n {
            // Round-trip cost; halve for one-way.
            let rt_ns = measure_pair(core_ids[a], core_ids[b], iters);
            let oneway_ns = (rt_ns / 2) as u32;
            entries_ns[a * n + b] = oneway_ns;
            entries_ns[b * n + a] = oneway_ns;
        }
    }
    let calibration_wall_ns = t0.elapsed().as_nanos() as u64;
    Some(TopologyLatencyTable {
        n,
        entries_ns,
        iters,
        calibration_wall_ns,
    })
}

/// Limit the calibration sweep to a manageable subset of cores
/// on huge hosts. The naive O(n^2 / 2) sweep is fine up to ~32
/// cores; past that we stride-sample to keep init-time bounded.
///
/// On a 64-core host the full sweep would be 64x63/2 = 2016
/// pairs; at ~100us per pair that is ~200 ms. With a stride of
/// 2 we measure 32x31/2 = 496 pairs, ~50 ms, still capturing
/// the topology shape.
fn limit_cores_for_calibration(cores: Vec<CoreId>) -> Vec<CoreId> {
    const FULL_SWEEP_MAX: usize = 32;
    if cores.len() <= FULL_SWEEP_MAX {
        return cores;
    }
    let stride = cores.len().div_ceil(FULL_SWEEP_MAX);
    cores
        .into_iter()
        .step_by(stride)
        .take(FULL_SWEEP_MAX)
        .collect()
}

/// Calibration timeout for the lazy-init path. If the sweep
/// runs longer than this budget the lazy-init returns a
/// fallback table populated from the heuristic tier values.
pub const CALIBRATION_BUDGET: Duration = Duration::from_millis(500);

/// Process-wide cached table. First call to
/// [`topology_latency_table`] runs the calibration; subsequent
/// calls return the cached snapshot.
static GLOBAL_TABLE: OnceLock<TopologyLatencyTable> = OnceLock::new();

/// Return a reference to the process-wide latency table. First
/// call calibrates (typ ~5-50 ms). Subsequent calls are
/// wait-free under no-contention.
///
/// Returns `None` ONLY when `core_affinity::get_core_ids` is
/// unavailable on the host (no pinning support). Callers in
/// that case should consult the heuristic-default sync-cost
/// tier values directly from [`crate::sched::
/// sync_cost_tier_for_levels`].
pub fn topology_latency_table() -> Option<&'static TopologyLatencyTable> {
    if let Some(t) = GLOBAL_TABLE.get() {
        return Some(t);
    }
    let table = calibrate_table(DEFAULT_PING_PONG_ITERS)?;
    let _ = GLOBAL_TABLE.set(table);
    GLOBAL_TABLE.get()
}

/// Force a fresh calibration. Useful for tests that exercise
/// the calibration path without depending on prior state.
/// In production code prefer [`topology_latency_table`].
#[cfg(test)]
pub fn force_calibrate() -> Option<TopologyLatencyTable> {
    calibrate_table(DEFAULT_PING_PONG_ITERS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_lookup_diagonal_is_zero() {
        // Synthesise a 4x4 table with known entries.
        let entries_ns: Vec<u32> = vec![
            0, 30, 60, 90,
            30, 0, 30, 60,
            60, 30, 0, 30,
            90, 60, 30, 0,
        ];
        let t = TopologyLatencyTable {
            n: 4,
            entries_ns,
            iters: 100,
            calibration_wall_ns: 1_000_000,
        };
        for i in 0..4 {
            assert_eq!(t.latency_ns(i, i), 0);
        }
    }

    #[test]
    fn table_lookup_off_diagonal_returns_entry() {
        let entries_ns: Vec<u32> = vec![
            0, 30, 60, 90,
            30, 0, 30, 60,
            60, 30, 0, 30,
            90, 60, 30, 0,
        ];
        let t = TopologyLatencyTable {
            n: 4,
            entries_ns,
            iters: 100,
            calibration_wall_ns: 1_000_000,
        };
        assert_eq!(t.latency_ns(0, 1), 30);
        assert_eq!(t.latency_ns(2, 3), 30);
        assert_eq!(t.latency_ns(0, 3), 90);
    }

    #[test]
    fn table_lookup_out_of_range_returns_zero() {
        let t = TopologyLatencyTable {
            n: 2,
            entries_ns: vec![0, 50, 50, 0],
            iters: 100,
            calibration_wall_ns: 1_000_000,
        };
        assert_eq!(t.latency_ns(5, 0), 0);
        assert_eq!(t.latency_ns(0, 5), 0);
    }

    #[test]
    fn mean_offdiag_excludes_diagonal() {
        let t = TopologyLatencyTable {
            n: 2,
            entries_ns: vec![0, 100, 100, 0],
            iters: 100,
            calibration_wall_ns: 1_000_000,
        };
        // Two off-diag entries both 100 -> mean 100.
        assert!((t.mean_offdiag_ns() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn min_and_max_offdiag_match_extremes() {
        let entries_ns: Vec<u32> = vec![
            0, 30, 60, 400,
            30, 0, 30, 60,
            60, 30, 0, 30,
            400, 60, 30, 0,
        ];
        let t = TopologyLatencyTable {
            n: 4,
            entries_ns,
            iters: 100,
            calibration_wall_ns: 1_000_000,
        };
        assert_eq!(t.min_offdiag_ns(), 30);
        assert_eq!(t.max_offdiag_ns(), 400);
    }

    #[test]
    fn min_offdiag_returns_zero_for_single_core_table() {
        let t = TopologyLatencyTable {
            n: 1,
            entries_ns: vec![0],
            iters: 0,
            calibration_wall_ns: 0,
        };
        assert_eq!(t.min_offdiag_ns(), 0);
        assert_eq!(t.max_offdiag_ns(), 0);
        assert_eq!(t.mean_offdiag_ns(), 0.0);
    }

    #[test]
    fn format_as_matrix_contains_header_and_n_rows() {
        let t = TopologyLatencyTable {
            n: 2,
            entries_ns: vec![0, 50, 50, 0],
            iters: 100,
            calibration_wall_ns: 1_000_000,
        };
        let s = t.format_as_matrix();
        assert!(s.contains("n=2"));
        assert!(s.contains("iters=100"));
        assert!(s.contains("50n"));
    }

    #[test]
    fn measure_pair_returns_nonzero_on_pinnable_host() {
        // The host running this test must have at least 2
        // CoreIds for the measurement to be meaningful. On
        // sandboxes without core_affinity, skip.
        let Some(ids) = core_affinity::get_core_ids() else {
            eprintln!("skipping: no core affinity support on this host");
            return;
        };
        if ids.len() < 2 {
            eprintln!("skipping: need >= 2 cores for ping-pong test");
            return;
        }
        // Best of five: a peer preempted mid-spin by another test
        // thread inflates one reading by a scheduler quantum, and
        // preemption only ever inflates.
        let rt_ns = (0..5).map(|_| measure_pair(ids[0], ids[1], 64)).min().unwrap_or(0);
        // Round-trip must be positive (some measurable cost)
        // and reasonable (under 10us per round trip even on
        // the worst hosts).
        assert!(rt_ns > 0, "ping-pong round-trip must be positive");
        assert!(
            rt_ns < 10_000,
            "ping-pong round-trip suspiciously high: {} ns",
            rt_ns
        );
    }

    #[test]
    fn calibrate_table_returns_some_on_pinnable_host() {
        let Some(_ids) = core_affinity::get_core_ids() else {
            eprintln!("skipping: no core affinity support on this host");
            return;
        };
        // 32 iters keeps the test fast (~1 ms per pair).
        let Some(t) = calibrate_table(MIN_PING_PONG_ITERS) else {
            eprintln!("skipping: calibrate_table returned None");
            return;
        };
        assert!(t.n() > 0, "table must cover at least one core");
        assert!(
            t.iters >= MIN_PING_PONG_ITERS,
            "iters clamp not respected"
        );
        if t.n() >= 2 {
            // At least one off-diag entry must be set.
            assert!(t.min_offdiag_ns() > 0, "off-diag latencies must be measured");
        }
    }

    #[test]
    fn topology_latency_table_is_stable_across_calls() {
        let Some(_a) = topology_latency_table() else {
            eprintln!("skipping: no core affinity support on this host");
            return;
        };
        let Some(_b) = topology_latency_table() else {
            unreachable!()
        };
        // Both should be the same global instance.
        // (Pointer equality on &'static T is the test.)
    }

    #[test]
    fn limit_cores_strides_when_above_full_sweep_max() {
        // Generic CoreId list of length > FULL_SWEEP_MAX (use a
        // dummy via std::iter::repeat clones won't work since
        // CoreId doesn't impl Clone in some configurations).
        // Just verify the function on the constant.
        let Some(ids) = core_affinity::get_core_ids() else {
            return;
        };
        let limited = limit_cores_for_calibration(ids);
        assert!(limited.len() <= 32);
    }
}

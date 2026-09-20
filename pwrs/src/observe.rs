//! What the scheduler saw: the dispatch trace, the leaf statistics,
//! occupancy, and which shape the last reduce took.
//!
//! Every item already crosses a Rust-owned boundary, so the counters
//! below cost the scheduler nothing extra to keep. A script has had no
//! way to read them.
//!
//! # Nothing here reports a zero it did not measure
//!
//! Every figure the crate can leave unmeasured is an `Option` there and
//! is written as null here. A thread clock the platform does not have,
//! a leaf spread with fewer than four samples, a reduce path on a
//! thread that has not run one: each is null with a reason beside it,
//! never a zero a reader would take for an answer.
//!
//! # The trace counters are destructive at the source
//!
//! `dispatch_trace_snapshot` and `dispatch_trace_wait_snapshot` swap
//! their counters to zero as they read. Two callers would therefore
//! each see part of the truth and neither would see the total. This
//! module adds every snapshot into its own running totals, so
//! `Get-FlynnelTrace` reports both the delta it just took and the total
//! since the module loaded, and calling it twice is not a way to lose
//! counts.

use std::sync::atomic::{AtomicU64, Ordering};

use pwrs::prelude::*;

use flynnel::sched::occupancy::{NoReading, OccupancySample, OccupancyWindow, ThreadTicks};
use flynnel::sched::par_iter::{
    ReduceChunksPath, last_reduce_chunks_path, sample_iqr_per_mille, sample_spread_per_mille,
};
use flynnel::sched::split_observer;
use flynnel::sched::trace;
use flynnel::sched::{dispatch_trace_snapshot, dispatch_trace_wait_snapshot};

/// The error for an observation argument the module cannot take.
fn arg_err(message: impl Into<String>) -> PsError {
    PsError::new(
        ErrorCategory::InvalidArgument,
        "FlynnelArgument",
        message.into(),
    )
}

// ---------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------

/// Why a figure has no reading.
#[psenum(name = "Flynnel.NoReading")]
#[derive(Clone, Copy, Default)]
pub enum NoReadingReason {
    /// The platform has no per-thread CPU clock.
    #[default]
    NoClock,
    /// The platform has one and the call to it failed.
    Unreadable,
}

impl From<NoReading> for NoReadingReason {
    fn from(r: NoReading) -> Self {
        match r {
            NoReading::NoClock => NoReadingReason::NoClock,
            NoReading::Unreadable => NoReadingReason::Unreadable,
        }
    }
}

/// Which shape a reduce took.
#[psenum(name = "Flynnel.ReducePath")]
#[derive(Clone, Copy, Default)]
pub enum ReducePath {
    /// One flat pass over the chunks.
    #[default]
    Flat,
    /// A bisecting tree.
    Bisect,
}

impl From<ReduceChunksPath> for ReducePath {
    fn from(p: ReduceChunksPath) -> Self {
        match p {
            ReduceChunksPath::Flat => ReducePath::Flat,
            ReduceChunksPath::Bisect => ReducePath::Bisect,
        }
    }
}

// ---------------------------------------------------------------------
// The trace
// ---------------------------------------------------------------------

/// Running totals over every snapshot this process has taken, so a
/// destructive read at the source is not a destructive read here.
static TOTAL_CALLS: AtomicU64 = AtomicU64::new(0);
static TOTAL_BODY: AtomicU64 = AtomicU64::new(0);
static TOTAL_WAIT: AtomicU64 = AtomicU64::new(0);
static TOTAL_STEAL: AtomicU64 = AtomicU64::new(0);
static TOTAL_IDLE: AtomicU64 = AtomicU64::new(0);

/// Whether the trace is armed, and how it is armed.
#[psclass(name = "Flynnel.TraceState")]
#[derive(Clone, Default)]
pub struct TraceState {
    /// Whether the per-event ring is recording.
    pub is_enabled: bool,
    /// The variable that arms the per-event ring. It is read once, at
    /// the first call that asks, and latched, so setting it after the
    /// process starts has no effect.
    pub enabled_by: String,
    /// Whether the dispatch counters are accumulating. A separate
    /// variable from the one above, so a process can have one without
    /// the other.
    pub dispatch_counters_armed: bool,
    /// The variable that arms the dispatch counters.
    pub counters_armed_by: String,
    /// How many worker flushes have completed.
    pub worker_flushes_done: u64,
}

/// The dispatch counters: what this read took, and the total since the
/// module loaded.
#[psclass(name = "Flynnel.TraceCounters")]
#[derive(Clone, Default)]
pub struct TraceCounters {
    /// Dispatches counted since the last read.
    pub calls: u64,
    /// Cycles inside dispatch bodies since the last read.
    pub body_cycles: u64,
    /// Cycles waiting since the last read.
    pub wait_cycles: u64,
    /// Cycles spent stealing since the last read.
    pub steal_cycles: u64,
    /// Cycles spent idle since the last read.
    pub idle_cycles: u64,
    /// Dispatches counted since the module loaded.
    pub total_calls: u64,
    /// Cycles inside dispatch bodies since the module loaded.
    pub total_body_cycles: u64,
    /// Cycles waiting since the module loaded.
    pub total_wait_cycles: u64,
    /// Cycles spent stealing since the module loaded.
    pub total_steal_cycles: u64,
    /// Cycles spent idle since the module loaded.
    pub total_idle_cycles: u64,
    /// Whether the counters were armed when this was read. False means
    /// every figure above is a zero the scheduler never counted, rather
    /// than a quiet run.
    pub armed: bool,
}

/// Set once Set-FlynnelTraceState has moved the flag in this process.
///
/// Without it EnabledBy would keep naming the environment variable
/// after a cmdlet had overridden it, which is a row asserting a
/// provenance that is no longer true.
static TRACE_SET_BY_CMDLET: AtomicU64 = AtomicU64::new(0);

/// The trace row, shared by the cmdlet that reads it and the one that
/// sets it, so the two cannot describe the state differently.
fn trace_state_row() -> TraceState {
    // The dispatch counters have no public predicate. Whether they
    // are armed is read from the variable that arms them, which is
    // the same thing the crate latches.
    let counters = matches!(
        std::env::var("FLYNNEL_TRACE_DISPATCH").as_deref(),
        Ok("1") | Ok("on") | Ok("true") | Ok("ON") | Ok("TRUE")
    );
    let by = if TRACE_SET_BY_CMDLET.load(Ordering::Relaxed) == 0 {
        "FLYNNEL_TRACE"
    } else {
        "Set-FlynnelTraceState"
    };
    TraceState {
        is_enabled: trace::is_enabled(),
        enabled_by: by.to_string(),
        dispatch_counters_armed: counters,
        counters_armed_by: "FLYNNEL_TRACE_DISPATCH".to_string(),
        worker_flushes_done: trace::worker_flushes_done(),
    }
}

/// Reads whether the dispatch trace is recording and how it was armed.
///
/// The ring is seeded from an environment variable read once, and
/// Set-FlynnelTraceState can move it afterwards. EnabledBy says which
/// of those last decided it.
///
/// The dispatch counters are a separate switch and are still latched
/// from their own variable, so CountersArmed false means every figure
/// on Get-FlynnelTrace is a zero the scheduler never counted.
///
/// # Examples
///
/// `Get-FlynnelTraceState`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelTraceState",
    alias = "Get-FlyTraceState",
    output = ["Flynnel.TraceState"]
)]
#[derive(Default)]
pub struct GetFlynnelTraceState {}

impl Cmdlet for GetFlynnelTraceState {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(trace_state_row())
    }
}

/// Reads the dispatch counters.
///
/// The crate's snapshot swaps its counters to zero as it reads, so the
/// figures are a delta since the last read rather than a running total.
/// This cmdlet adds each delta into its own totals and reports both, so
/// reading twice does not lose counts and a script does not have to
/// keep them itself.
///
/// Armed false means the scheduler never counted: the zeros are an
/// unarmed instrument, not a quiet run.
///
/// # Examples
///
/// `Get-FlynnelTrace`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelTrace",
    alias = "Get-FlyTrace",
    output = ["Flynnel.TraceCounters"]
)]
#[derive(Default)]
pub struct GetFlynnelTrace {}

impl Cmdlet for GetFlynnelTrace {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let (calls, body, wait) = dispatch_trace_snapshot();
        let (steal, idle) = dispatch_trace_wait_snapshot();
        let total_calls = TOTAL_CALLS.fetch_add(calls, Ordering::Relaxed) + calls;
        let total_body = TOTAL_BODY.fetch_add(body, Ordering::Relaxed) + body;
        let total_wait = TOTAL_WAIT.fetch_add(wait, Ordering::Relaxed) + wait;
        let total_steal = TOTAL_STEAL.fetch_add(steal, Ordering::Relaxed) + steal;
        let total_idle = TOTAL_IDLE.fetch_add(idle, Ordering::Relaxed) + idle;
        let armed = matches!(
            std::env::var("FLYNNEL_TRACE_DISPATCH").as_deref(),
            Ok("1") | Ok("on") | Ok("true") | Ok("ON") | Ok("TRUE")
        );
        if !armed {
            pwrs::warning!(
                ps,
                "the dispatch counters are not armed; every figure is a zero the scheduler \
                 never counted. Set FLYNNEL_TRACE_DISPATCH before the process starts."
            )?;
        }
        ps.write(TraceCounters {
            calls,
            body_cycles: body,
            wait_cycles: wait,
            steal_cycles: steal,
            idle_cycles: idle,
            total_calls,
            total_body_cycles: total_body,
            total_wait_cycles: total_wait,
            total_steal_cycles: total_steal,
            total_idle_cycles: total_idle,
            armed,
        })
    }
}

/// Clears the calling thread's trace ring and any pending worker-flush
/// request.
///
/// This is the pipeline thread's ring. A worker's ring is its own and
/// is not reachable from here.
///
/// # Examples
///
/// `Clear-FlynnelTrace`
#[cmdlet(
    verb = "Clear",
    noun = "FlynnelTrace",
    alias = "Clear-FlyTrace"
)]
#[derive(Default)]
pub struct ClearFlynnelTrace {}

impl Cmdlet for ClearFlynnelTrace {
    fn process(&mut self, _ps: &Pipeline<'_>) -> PsResult<()> {
        trace::reset_current_thread();
        trace::clear_worker_flush_request();
        Ok(())
    }
}

/// Asks every worker to flush its trace ring, and answers how many
/// flushes had completed at the moment of the request.
///
/// A worker flushes when it next reaches the top of its loop, so the
/// count rises after this returns. Read it again with
/// `Get-FlynnelTraceState` to see the request land.
///
/// # Examples
///
/// `Request-FlynnelTraceFlush`
#[cmdlet(
    verb = "Request",
    noun = "FlynnelTraceFlush",
    alias = "Request-FlyTraceFlush",
    output = ["System.UInt64"]
)]
#[derive(Default)]
pub struct RequestFlynnelTraceFlush {}

impl Cmdlet for RequestFlynnelTraceFlush {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let before = trace::worker_flushes_done();
        trace::request_worker_flush();
        ps.write(before)
    }
}

// ---------------------------------------------------------------------
// Leaf statistics
// ---------------------------------------------------------------------

/// What the pool's leaves have cost, process-wide.
#[psclass(name = "Flynnel.LeafStats")]
#[derive(Clone, Default)]
pub struct LeafStatRow {
    /// How many leaves were timed.
    pub count: u64,
    /// Their total wall time in nanoseconds.
    pub sum_ns: u64,
    /// How many items those leaves covered.
    pub items: u64,
    /// The mean leaf in nanoseconds. Null below the sample floor the
    /// crate needs before a mean means anything.
    pub mean_leaf_ns: Option<u64>,
    /// The mean nanoseconds per item. Null on the same terms.
    pub per_item_ns: Option<u64>,
    /// The leaves' squared coefficient of variation in parts per
    /// thousand, which is the spread statistic the classifier reads.
    /// Null below the sample floor.
    pub leaf_cv2_per_mille: Option<u64>,
    /// The same statistic per item. Null below the sample floor.
    pub per_item_cv2_per_mille: Option<u64>,
}

/// Reads what the pool's leaves have cost since the statistics were
/// last reset.
///
/// These are process-wide rather than per call site. Flynnel keeps
/// per-site statistics too, but its registry has no public walk, so a
/// binding cannot enumerate the sites. The census records that.
///
/// A spread below the crate's sample floor is null rather than zero: a
/// spread computed from three leaves is not a spread.
///
/// # Examples
///
/// `Get-FlynnelLeafStat`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelLeafStat",
    alias = "Get-FlyLeafStat",
    output = ["Flynnel.LeafStats"]
)]
#[derive(Default)]
pub struct GetFlynnelLeafStat {}

impl Cmdlet for GetFlynnelLeafStat {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let stats = split_observer::snapshot_leaf_stats();
        ps.write(LeafStatRow {
            count: stats.count,
            sum_ns: stats.sum_ns,
            items: stats.items,
            mean_leaf_ns: split_observer::observed_mean_leaf_ns(),
            per_item_ns: split_observer::observed_per_item_ns(stats),
            leaf_cv2_per_mille: split_observer::leaf_cv_squared_per_mille(stats),
            per_item_cv2_per_mille: split_observer::per_item_cv_squared_per_mille(stats),
        })
    }
}

/// Discards the pool's accumulated leaf statistics, so the next
/// reading covers only what happens after it.
///
/// # Examples
///
/// `Reset-FlynnelLeafStat`
#[cmdlet(
    verb = "Reset",
    noun = "FlynnelLeafStat",
    alias = "Reset-FlyLeafStat"
)]
#[derive(Default)]
pub struct ResetFlynnelLeafStat {}

impl Cmdlet for ResetFlynnelLeafStat {
    fn process(&mut self, _ps: &Pipeline<'_>) -> PsResult<()> {
        split_observer::reset_leaf_stats();
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Occupancy
// ---------------------------------------------------------------------

/// The share of an interval a thread held a core.
#[psclass(name = "Flynnel.Occupancy")]
#[derive(Clone, Default)]
pub struct Occupancy {
    /// The share as a percentage, 0 to 100. Null where the platform
    /// could not read a thread clock, which is not the same as a thread
    /// that held no core.
    pub percent: Option<u32>,
    /// Ticks the thread was on a core. Null on the same terms as
    /// Percent.
    pub thread_ticks: Option<u64>,
    /// Ticks of wall time the window covered.
    pub wall_ticks: u64,
    /// Why there is no reading, where there is none.
    pub no_reading: Option<NoReadingReason>,
    /// What a thread tick counts on this platform. Windows answers CPU
    /// cycles and the others answer nanoseconds, so the two are not
    /// comparable and only the ratio is.
    pub thread_tick_unit: String,
}

/// What this platform's thread ticks count.
fn tick_unit() -> String {
    if cfg!(windows) {
        "cycles".to_string()
    } else if cfg!(any(target_os = "linux", target_os = "freebsd")) {
        "nanoseconds".to_string()
    } else {
        "none".to_string()
    }
}

/// Turn a sample into the row, keeping an absent reading absent.
fn occupancy_of(sample: OccupancySample) -> Occupancy {
    match sample {
        OccupancySample::Measured {
            thread_ticks,
            wall_ticks,
        } => Occupancy {
            percent: sample.percent(),
            thread_ticks: Some(thread_ticks),
            wall_ticks,
            no_reading: None,
            thread_tick_unit: tick_unit(),
        },
        OccupancySample::Unmeasured { wall_ticks, reason } => Occupancy {
            percent: None,
            thread_ticks: None,
            wall_ticks,
            no_reading: Some(reason.into()),
            thread_tick_unit: tick_unit(),
        },
    }
}

/// Measures what share of a window the calling thread holds a core.
///
/// This times a window on the pipeline thread, so it measures the
/// shell, not the pool. It is the instrument the scheduler judges its
/// own draws with, and a caller wanting to know whether this box is
/// quiet enough to measure on can ask it the same question.
///
/// A percentage outside 0 to 100 would be the instrument rather than
/// the box; the crate saturates at 100.
///
/// # Examples
///
/// `Measure-FlynnelOccupancy -Seconds 1`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelOccupancy",
    alias = "Measure-FlyOccupancy",
    output = ["Flynnel.Occupancy"]
)]
#[derive(Default)]
pub struct MeasureFlynnelOccupancy {
    /// How long a window to measure, in seconds.
    #[param(position = 0)]
    pub seconds: Option<f64>,
}

impl Cmdlet for MeasureFlynnelOccupancy {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let seconds = match self.seconds {
            None => 1.0,
            Some(s) if s > 0.0 && s <= 600.0 => s,
            Some(_) => {
                return Err(
                    arg_err("Seconds must be above zero and no more than 600").terminating()
                );
            }
        };
        let window = OccupancyWindow::start();
        std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
        ps.write(occupancy_of(window.sample()))
    }
}

/// Reads the calling thread's on-core tick counter.
///
/// One reading alone says nothing: the counter is meaningful only as a
/// difference across an interval, which is what Measure-FlynnelOccupancy
/// does. This is here so a caller can bracket their own interval.
///
/// Ticks is null where the platform has no per-thread clock or the call
/// to it failed, and NoReading says which.
///
/// # Examples
///
/// `Get-FlynnelThreadTick`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelThreadTick",
    alias = "Get-FlyThreadTick",
    output = ["Flynnel.Occupancy"]
)]
#[derive(Default)]
pub struct GetFlynnelThreadTick {}

impl Cmdlet for GetFlynnelThreadTick {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let row = match flynnel::sched::occupancy::thread_on_core_ticks() {
            ThreadTicks::Measured(ticks) => Occupancy {
                percent: None,
                thread_ticks: Some(ticks),
                wall_ticks: 0,
                no_reading: None,
                thread_tick_unit: tick_unit(),
            },
            ThreadTicks::Absent(reason) => Occupancy {
                percent: None,
                thread_ticks: None,
                wall_ticks: 0,
                no_reading: Some(reason.into()),
                thread_tick_unit: tick_unit(),
            },
        };
        ps.write(row)
    }
}

// ---------------------------------------------------------------------
// Paths and spreads
// ---------------------------------------------------------------------

/// Reads which shape the last reduce on this thread took.
///
/// The crate records this per thread, not per process. A cmdlet runs on
/// the pipeline thread, and so do the kernels here, so a reduce run by
/// this module is visible to this cmdlet. A reduce run on a worker is
/// not.
///
/// Null means no reduce has run on this thread, which is not the same
/// as a reduce that took neither path.
///
/// # Examples
///
/// `Measure-FlynnelReduce -InputObject $x -Operation Sum; Get-FlynnelReducePath`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelReducePath",
    alias = "Get-FlyReducePath",
    output = ["Flynnel.ReducePath"]
)]
#[derive(Default)]
pub struct GetFlynnelReducePath {}

impl Cmdlet for GetFlynnelReducePath {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        // Nothing written where no reduce has run, rather than a
        // value standing in for one. A caller distinguishes the two by
        // whether anything came back.
        match last_reduce_chunks_path() {
            Some(path) => ps.write(ReducePath::from(path)),
            None => {
                pwrs::verbose!(
                    ps,
                    "no reduce has run on this thread, so there is no path to report"
                )
            }
        }
    }
}

/// The spread of a set of samples in parts per thousand, by the same
/// statistic the scheduler's own classifier reads.
///
/// The crate's function requires its input already sorted. This sorts
/// for the caller, so the answer does not silently depend on the order
/// the samples arrived in.
///
/// Fewer than two samples have no spread and answer zero, which is what
/// the crate answers; the Count column is what distinguishes that from
/// a measured zero.
///
/// # Examples
///
/// `Get-FlynnelSpread -Sample 100,102,99,101`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelSpread",
    alias = "Get-FlySpread",
    output = ["Flynnel.Spread"]
)]
#[derive(Default)]
pub struct GetFlynnelSpread {
    /// The samples, in any order.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub sample: Vec<u64>,
}

/// A spread reading over a caller's own samples.
#[psclass(name = "Flynnel.Spread")]
#[derive(Clone, Default)]
pub struct Spread {
    /// How many samples it read.
    pub count: u64,
    /// The spread in parts per thousand.
    pub spread_per_mille: u32,
    /// The interquartile range, also in parts per thousand.
    ///
    /// Beside the spread rather than instead of it, because the two
    /// disagree exactly when it matters. The spread reads the extremes
    /// and a single stalled sample moves it; this reads the middle
    /// half and does not. A run whose spread is wide and whose
    /// interquartile range is narrow was steady with an interruption
    /// in it, and one where both are wide was not steady.
    pub iqr_per_mille: u32,
    /// The smallest sample. Null over no samples.
    pub minimum: Option<u64>,
    /// The median sample. Null over no samples.
    pub median: Option<u64>,
    /// The largest sample. Null over no samples.
    pub maximum: Option<u64>,
}

impl Cmdlet for GetFlynnelSpread {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let mut samples = std::mem::take(&mut self.sample);
        samples.sort_unstable();
        let n = samples.len();
        let (minimum, median, maximum) = if n == 0 {
            (None, None, None)
        } else {
            (
                Some(samples[0]),
                Some(samples[n / 2]),
                Some(samples[n - 1]),
            )
        };
        ps.write(Spread {
            count: n as u64,
            spread_per_mille: sample_spread_per_mille(&samples),
            iqr_per_mille: sample_iqr_per_mille(&samples),
            minimum,
            median,
            maximum,
        })
    }
}

/// Turns the dispatch trace ring on or off, and writes back the state
/// that is now in force.
///
/// Until this existed the ring could only be armed by setting
/// FLYNNEL_TRACE before the process started, which a module cannot do
/// from inside the process it is already running in.
///
/// Recording is per thread and a ring is drained by a worker flush,
/// so turning it on records from the next event on each thread and
/// says nothing about what happened before. Turning it off stops the
/// recording and leaves whatever each ring already holds.
///
/// On is a switch, so turning the ring off takes the colon form
/// PowerShell uses for a switch given a value.
///
/// This is process-wide.
///
/// # Examples
///
/// `Set-FlynnelTraceState -On`
///
/// `Set-FlynnelTraceState -On:$false`
#[cmdlet(
    verb = "Set",
    noun = "FlynnelTraceState",
    alias = "Set-FlyTraceState",
    output = ["Flynnel.TraceState"]
)]
#[derive(Default)]
pub struct SetFlynnelTraceState {
    /// Record events from now on.
    #[param]
    pub on: bool,
}

impl Cmdlet for SetFlynnelTraceState {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let was = trace::set_enabled(self.on);
        TRACE_SET_BY_CMDLET.store(1, Ordering::Relaxed);
        if was == self.on {
            pwrs::warning!(
                ps,
                "the trace ring was already {}, so nothing changed",
                if was { "on" } else { "off" }
            )?;
        }
        ps.write(trace_state_row())
    }
}

// ---------------------------------------------------------------------
// Call sites
// ---------------------------------------------------------------------

/// What one dispatch call site has learned about the work that reaches
/// it.
///
/// The scheduler keeps this per source location, so two callers of the
/// same kernel with different workloads each get their own classifier
/// rather than averaging into one.
#[psclass(name = "Flynnel.CallSite")]
#[derive(Clone, Default)]
pub struct CallSite {
    /// The source file the site is in.
    pub file: String,
    /// Its line.
    pub line: u32,
    /// Its column, which is what tells two sites on one line apart.
    pub column: u32,
    /// The class the site settled on, null before it has classified a
    /// window.
    pub learned_class: Option<crate::types::WorkloadClass>,
    /// Leaves the site has sampled. Sampled, not dispatched: the
    /// on-core pair below is filled on a strided path and this is the
    /// count that path produced.
    pub leaf_count: u64,
    /// What one item cost, in nanoseconds, null below the sample floor.
    pub per_item_ns: Option<u64>,
    /// Spread of the per-item cost over the site's whole life, in parts
    /// per thousand of the mean squared.
    ///
    /// Lifetime, not the window the class came from. A site that
    /// changed regime once carries both regimes in this figure forever,
    /// so read WindowCv2MinPerMille and WindowCv2MaxPerMille to see
    /// what the classifier actually acted on.
    pub cv2_per_mille: Option<u64>,
    /// The same spread measured on the threads' own clocks rather than
    /// the wall, which advance only while a thread is on a core. Also
    /// lifetime.
    ///
    /// There is no window-scoped form of this figure, so the spread the
    /// lever acted on at the window it acted cannot be read. That is a
    /// hole in the crate's surface rather than in this row.
    pub per_item_oncore_cv2_per_mille: Option<u64>,
    /// Items the on-core figures were measured over. Zero says the
    /// strided path never sampled here, which is why the spread beside
    /// it is null rather than a reading of nothing.
    pub oncore_items: u64,
    /// Mean leaf time of the delta window the latest tick classified.
    pub window_mean_ns: Option<u64>,
    /// Spread of that one window.
    ///
    /// One classifier tick out of thousands. It spans the whole range
    /// within a single run, so a reading of it says almost nothing on
    /// its own; the two extremes below are the figure to judge a run
    /// by.
    pub window_cv2_per_mille: Option<u64>,
    /// The lowest per-window spread across every tick this site has
    /// classified.
    pub window_cv2_min_per_mille: Option<u64>,
    /// The highest, which with the lowest gives the range the
    /// classifier acted over.
    pub window_cv2_max_per_mille: Option<u64>,
    /// Windows the site has classified.
    pub window_ticks: u64,
    /// What fraction of its interval the most recent dispatch here
    /// spent on a core, in hundredths. Null before any dispatch has
    /// reported, which is a different state from a pool that held none
    /// of its cores.
    pub recent_occupancy_pct: Option<u32>,
    /// The seed depth in force, null before one is established.
    pub seeded_depth: Option<u32>,
    /// Dispatches that seeded a different leaf count from the dispatch
    /// before them. The figure the seed-depth stabilizers are measured
    /// against.
    ///
    /// Counts only dispatches that reach the adaptive depth, which a
    /// plan naming an explicit variant does not. Zero at a site whose
    /// callers all name one means the question was never asked, not
    /// that the answer was steady.
    pub seed_depth_flips: u32,
    /// Whether a body that ran inline here overran the threshold that
    /// admitted it. While true, this site dispatches whatever the
    /// caller estimates.
    pub collapse_overran: bool,
    /// Wall time of the execution-policy arms, in nanoseconds, as an
    /// exponential moving average. Zero means the arm has no samples.
    pub arm_ewma_default_ns: u64,
    /// The alternative execution-policy arm, on the same terms.
    pub arm_ewma_alternative_ns: u64,
    /// The routing arms, kept apart from the execution-policy pair
    /// because one EWMA over both would let neither consumer read its
    /// own effect.
    pub routing_ewma_default_ns: u64,
    /// The alternative routing arm.
    pub routing_ewma_alternative_ns: u64,
    /// The share of a split this site sends to the CPU, in parts per
    /// thousand.
    pub split_cpu_share_per_mille: u32,
    /// Average cost of one reduce merge here, in cycles, null until the
    /// observer has timed any.
    pub reduce_cost_avg_cycles: Option<u64>,
}

fn call_site_row(entry: &flynnel::RegisteredSite) -> CallSite {
    let s = entry.site.get();
    let (arm_default, arm_alternative) = s.arm_ewmas();
    let (route_default, route_alternative) = s.routing_ewmas();
    let (cv2_min, cv2_max) = match s.window_cv2_range_per_mille() {
        Some((lo, hi)) => (Some(lo), Some(hi)),
        None => (None, None),
    };
    CallSite {
        file: entry.location.file().to_string(),
        line: entry.location.line(),
        column: entry.location.column(),
        learned_class: s.learned_class().map(|c| c.into()),
        leaf_count: s.leaf_count(),
        per_item_ns: s.per_item_ns(),
        cv2_per_mille: s.per_item_cv2_per_mille(),
        per_item_oncore_cv2_per_mille: s.per_item_oncore_cv2_per_mille(),
        oncore_items: s.oncore_items(),
        window_mean_ns: s.window_mean_ns(),
        window_cv2_per_mille: s.window_cv2_per_mille(),
        window_cv2_min_per_mille: cv2_min,
        window_cv2_max_per_mille: cv2_max,
        window_ticks: s.window_ticks(),
        recent_occupancy_pct: s.recent_occupancy(),
        seeded_depth: s.seeded_depth(),
        seed_depth_flips: s.seed_depth_flips(),
        collapse_overran: s.collapse_overran(),
        arm_ewma_default_ns: arm_default,
        arm_ewma_alternative_ns: arm_alternative,
        routing_ewma_default_ns: route_default,
        routing_ewma_alternative_ns: route_alternative,
        split_cpu_share_per_mille: s.split_cpu_share_per_mille(),
        reduce_cost_avg_cycles: s.reduce_cost_avg_cycles(),
    }
}

/// Reads every dispatch call site the scheduler has materialised in
/// this process, and what each has learned.
///
/// A site appears once a dispatch has reached that source location, so
/// a process that has run no work through Flynnel answers nothing. The
/// locations are inside the scheduler and inside this module, because
/// those are the callers: a cmdlet's own line is not a call site.
///
/// Every site in one call. The registry sits behind the lock a dispatch
/// meeting a new location has to take to write, so reading it a site at
/// a time would hold that lock repeatedly against the pool.
///
/// # Examples
///
/// `Get-FlynnelCallSite`
///
/// `Get-FlynnelCallSite | Sort-Object LeafCount -Descending`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelCallSite",
    alias = "Get-FlyCallSite",
    output = ["Flynnel.CallSite"]
)]
#[derive(Default)]
pub struct GetFlynnelCallSite {}

impl Cmdlet for GetFlynnelCallSite {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let sites = flynnel::registered_sites();
        if sites.is_empty() {
            pwrs::warning!(
                ps,
                "no dispatch has reached a call site in this process yet, so there is nothing \
                 to report; run a kernel first"
            )?;
            return Ok(());
        }
        for entry in &sites {
            ps.write(call_site_row(entry))?;
        }
        Ok(())
    }
}

/// Returns every call site to the state it starts a process in, and
/// writes how many it reset.
///
/// For measuring two arms in one process. Site state is kept for the
/// life of the process, so without this the second arm inherits the
/// class, the per-arm averages and the seed depth the first one taught
/// the classifier, and its numbers describe both. Running the arms in
/// separate processes instead carries whatever else differed between
/// those processes, which for a decision driven by a measured estimate
/// is the thing being measured.
///
/// This throws measurement away and cannot be undone, so it asks.
/// Between arms, never during one: a dispatch running while the reset
/// lands sees some counters cleared and some not.
///
/// # Examples
///
/// `Reset-FlynnelCallSite -Confirm:$false`
#[cmdlet(
    verb = "Reset",
    noun = "FlynnelCallSite",
    alias = "Reset-FlyCallSite",
    supports_should_process,
    confirm_impact = "High",
    output = ["System.UInt64"]
)]
#[derive(Default)]
pub struct ResetFlynnelCallSite {}

impl Cmdlet for ResetFlynnelCallSite {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let held = flynnel::registered_sites().len();
        if !ps.should_process(
            &format!("{held} call site(s) in this process"),
            "discard everything they have learned",
        )? {
            return Ok(());
        }
        ps.write(flynnel::reset_all_sites() as u64)
    }
}

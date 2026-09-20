//! What the scheduler measured about this host, what it decided from
//! it, and how a caller makes it measure again.
//!
//! # Every figure says where it came from
//!
//! A calibrated number and a shipped constant look identical once they
//! are in an atomic, and a figure whose provenance is invisible gets
//! quoted as a measurement. So each one is written with a Source:
//!
//! - `Measured`, and only when this module ran the calibration in this
//!   process, with the time it was taken.
//! - `Stored`, read back from the persisted table, carrying the time
//!   the record itself records.
//! - `Default`, equal to the constant the crate ships and with no
//!   calibration run through this module.
//! - `Unattributed`, different from the shipped constant with no
//!   calibration run through this module. Something set it and this
//!   module cannot say what.
//!
//! The last one exists because the crate keeps no provenance beside
//! most of these values. Calling a figure Default when it merely
//! matches the default would be the same lie in the other direction.
//!
//! # Several of these cmdlets measure
//!
//! A getter that takes 10 to 40 milliseconds of every core is not what
//! a reader expects of `Get-`, so the ones that can are named
//! `Measure-` and say so. The one exception is forced by the crate:
//! `host_dispatch_profile()` measures on its first call in a process,
//! whoever makes it. `Get-FlynnelHostDispatch` reports whether the call
//! it just made was the one that paid.

use std::sync::atomic::{AtomicU64, Ordering};

use pwrs::prelude::*;

use flynnel::sched::adaptive_profile::{
    self, ClassThresholds as CrateThresholds, ThresholdCalibration as CrateThresholdCal,
};
use flynnel::sched::k_gating::{self, KGating as CrateKGating};
use flynnel::sched::par_iter;

/// The error for a calibration argument the module cannot take.
fn arg_err(message: impl Into<String>) -> PsError {
    PsError::new(
        ErrorCategory::InvalidArgument,
        "FlynnelArgument",
        message.into(),
    )
}

/// The error for a persisted store that could not be reached.
fn store_err(detail: impl std::fmt::Display) -> PsError {
    PsError::new(
        ErrorCategory::ResourceUnavailable,
        "FlynnelStore",
        format!("the calibration store could not be read: {detail}"),
    )
}

// ---------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------

/// Where a calibrated figure came from.
#[psenum(name = "Flynnel.Source")]
#[derive(Clone, Copy, Default)]
pub enum Source {
    /// The constant the crate ships, with no calibration run through
    /// this module in this process.
    #[default]
    Default,
    /// This module ran the calibration in this process.
    Measured,
    /// Read back from the persisted table for this host.
    Stored,
    /// Not the shipped constant, and no calibration ran through this
    /// module. Something else set it and this module cannot say what.
    Unattributed,
}

/// Unix seconds at which this module last ran each calibration, or
/// zero for never. The crate keeps no such record for the live
/// thresholds, so a binding that wants to say `Measured` honestly has
/// to keep its own.
static HOST_MEASURED_AT: AtomicU64 = AtomicU64::new(0);
static CLASS_MEASURED_AT: AtomicU64 = AtomicU64::new(0);
static KGATING_MEASURED_AT: AtomicU64 = AtomicU64::new(0);

/// Seconds since the Unix epoch.
///
/// A clock reading before the epoch is refused rather than stamped
/// zero: zero is the value this module uses for never-measured, so a
/// broken clock would make a measurement that did happen read as one
/// that did not.
fn now_unix_s() -> PsResult<u64> {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => Ok(d.as_secs()),
        Err(e) => Err(PsError::new(
            ErrorCategory::InvalidResult,
            "FlynnelClock",
            format!(
                "this host's clock reads before the Unix epoch ({e}), so a measurement taken \
                 now cannot be stamped and would be indistinguishable from one never taken"
            ),
        )
        .terminating()),
    }
}

/// The provenance of a live figure, given whether this module measured
/// it and whether it still equals the shipped constant.
fn source_of(measured_at: u64, is_default_value: bool) -> Source {
    if measured_at != 0 {
        Source::Measured
    } else if is_default_value {
        Source::Default
    } else {
        Source::Unattributed
    }
}

/// A timestamp as an option, so never-measured is null rather than the
/// epoch.
fn measured_at_or_none(at: u64) -> Option<u64> {
    if at == 0 { None } else { Some(at) }
}

/// The host profile's provenance, preferring the crate's own answer
/// over this module's bookkeeping where it has one.
fn host_source(measured_at: u64, crate_measured: Option<u64>) -> Source {
    match (measured_at, crate_measured) {
        (0, None) => Source::Default,
        (0, Some(_)) => Source::Unattributed,
        _ => Source::Measured,
    }
}

// ---------------------------------------------------------------------
// The class thresholds
// ---------------------------------------------------------------------

/// The shipped defaults, from `ClassThresholds::new_defaults`. Held
/// here so a live value can be told from the constant it started as.
const DEFAULT_FINE_GRAIN_NS: u64 = 50;
const DEFAULT_PORT_HEAVY_NS: u64 = 500;
const DEFAULT_MEMORY_LATENCY_NS: u64 = 2000;
const DEFAULT_CV2_LOW_PER_MILLE: u64 = 50;
const DEFAULT_CV2_HIGH_PER_MILLE: u64 = 500;
const DEFAULT_TRIVIAL_REDUCE_CYCLES: u64 = 30_000;

/// The boundaries that decide which class a workload is learned as.
#[psclass(name = "Flynnel.ClassThresholds")]
#[derive(Clone, Default)]
pub struct ClassThresholds {
    /// At or below this mean leaf, in nanoseconds, work is FineGrain.
    pub fine_grain_ns: u64,
    /// Above this mean leaf, work is port-heavy rather than streaming.
    pub port_heavy_ns: u64,
    /// Above this mean leaf, work is treated as memory-latency bound.
    pub memory_latency_ns: u64,
    /// At or below this squared coefficient of variation, in parts per
    /// thousand, leaves are even.
    pub cv2_low_per_mille: u64,
    /// Above this, leaves are ragged.
    pub cv2_high_per_mille: u64,
    /// The cycle ceiling that separates a trivial reduce from a real
    /// one.
    pub trivial_reduce_cycles: u64,
    /// Where these came from.
    pub source: Source,
    /// When this module measured them, in Unix seconds. Null where it
    /// did not.
    pub measured_at: Option<u64>,
}

/// Read the live thresholds with their provenance.
fn read_thresholds(live: &CrateThresholds) -> ClassThresholds {
    let fine = live.fine_grain_ns.load(Ordering::Relaxed);
    let port = live.port_heavy_ns.load(Ordering::Relaxed);
    let memory = live.memory_latency_ns.load(Ordering::Relaxed);
    let low = live.cv2_low_per_mille.load(Ordering::Relaxed);
    let high = live.cv2_high_per_mille.load(Ordering::Relaxed);
    let trivial = live.trivial_reduce_cycles.load(Ordering::Relaxed);
    let at = CLASS_MEASURED_AT.load(Ordering::Relaxed);
    let all_default = fine == DEFAULT_FINE_GRAIN_NS
        && port == DEFAULT_PORT_HEAVY_NS
        && memory == DEFAULT_MEMORY_LATENCY_NS
        && low == DEFAULT_CV2_LOW_PER_MILLE
        && high == DEFAULT_CV2_HIGH_PER_MILLE
        && trivial == DEFAULT_TRIVIAL_REDUCE_CYCLES;
    ClassThresholds {
        fine_grain_ns: fine,
        port_heavy_ns: port,
        memory_latency_ns: memory,
        cv2_low_per_mille: low,
        cv2_high_per_mille: high,
        trivial_reduce_cycles: trivial,
        source: source_of(at, all_default),
        measured_at: measured_at_or_none(at),
    }
}

/// Reads the boundaries that decide which class a workload is learned
/// as.
///
/// Source says whether this module measured them. The crate keeps no
/// provenance beside these values, so Default means they equal the
/// shipped constants and Unattributed means they do not and something
/// other than this module set them.
///
/// # Examples
///
/// `Get-FlynnelClassThreshold`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelClassThreshold",
    alias = "Get-FlyClassThreshold",
    output = ["Flynnel.ClassThresholds"]
)]
#[derive(Default)]
pub struct GetFlynnelClassThreshold {}

impl Cmdlet for GetFlynnelClassThreshold {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(read_thresholds(adaptive_profile::class_thresholds()))
    }
}

/// What a threshold calibration measured.
#[psclass(name = "Flynnel.ThresholdCalibration")]
#[derive(Clone, Default)]
pub struct ThresholdCalibration {
    /// What one join cost, in nanoseconds.
    pub join_ns: u64,
    /// The fine-grain boundary it derived, in nanoseconds.
    pub fine_grain_ns: u64,
    /// What the reference reduce merge measured, in cycles.
    pub trivial_reduce_measured_cycles: u64,
    /// The trivial-reduce ceiling it installed, in cycles.
    pub trivial_reduce_cycles: u64,
    /// What an SMT sibling was worth, in parts per thousand.
    pub smt_ratio_per_mille: u64,
    /// The memory-latency boundary it derived, in nanoseconds.
    pub memory_latency_ns: u64,
    /// When it was taken, in Unix seconds.
    pub measured_at: u64,
}

impl ThresholdCalibration {
    fn of(c: CrateThresholdCal, at: u64) -> Self {
        Self {
            join_ns: c.join_ns,
            fine_grain_ns: c.fine_grain_ns,
            trivial_reduce_measured_cycles: c.trivial_reduce_measured_cycles,
            trivial_reduce_cycles: c.trivial_reduce_cycles,
            smt_ratio_per_mille: c.smt_ratio_per_mille,
            memory_latency_ns: c.memory_latency_ns,
            measured_at: at,
        }
    }
}

/// Measures the class boundaries on this host and installs them.
///
/// This takes the cores while it runs. It is a measurement, so it obeys
/// the same discipline as any other: run it on a box quiet enough that
/// its cores are not deciding the answer, or the thresholds it installs
/// describe the load rather than the host.
///
/// # Examples
///
/// `Measure-FlynnelClassThreshold`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelClassThreshold",
    alias = "Measure-FlyClassThreshold",
    output = ["Flynnel.ThresholdCalibration"]
)]
#[derive(Default)]
pub struct MeasureFlynnelClassThreshold {
    /// Run it on the background pool and return at once, rather than
    /// waiting. Nothing is written in that case: the crate's spawned
    /// form answers no handle and no result.
    #[param]
    pub background: bool,
}

impl Cmdlet for MeasureFlynnelClassThreshold {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        if self.background {
            adaptive_profile::spawn_class_threshold_calibration();
            pwrs::warning!(
                ps,
                "started in the background. The crate's spawned form answers no handle and no \
                 result, so there is nothing to wait on and nothing to write; read \
                 Get-FlynnelClassThreshold later to see whether the values moved. It is also a \
                 no-op when the IO pool is disabled."
            )?;
            return Ok(());
        }
        // Stamped before the atomic is written, so a clock this module
        // refuses leaves the record saying never-measured rather than
        // saying measured at the epoch.
        let at = now_unix_s()?;
        let result = adaptive_profile::calibrate_class_thresholds();
        CLASS_MEASURED_AT.store(at, Ordering::Relaxed);
        ps.write(ThresholdCalibration::of(result, at))
    }
}

// ---------------------------------------------------------------------
// The host dispatch profile
// ---------------------------------------------------------------------

/// What a dispatch costs on this host.
#[psclass(name = "Flynnel.HostDispatch")]
#[derive(Clone, Default)]
pub struct HostDispatch {
    /// What one pool dispatch costs, in nanoseconds.
    pub dispatch_cost_ns: u64,
    /// Below this total, in nanoseconds, work collapses inline rather
    /// than dispatching.
    pub collapse_threshold_ns: u64,
    /// The wake threshold for the join-and-continue path, in
    /// nanoseconds.
    pub jec_wake_threshold_ns: u64,
    /// The collapse threshold as a measurement, null where none has
    /// been taken. This is the crate's own answer to "was this
    /// measured", and it is the only figure here that carries one.
    pub measured_collapse_threshold_ns: Option<u64>,
    /// Where these came from.
    pub source: Source,
    /// When this module measured them, in Unix seconds. Null where it
    /// did not.
    pub measured_at: Option<u64>,
    /// Whether the call that produced this row is the one that paid for
    /// the measurement. The crate measures on the first call in a
    /// process, whoever makes it, so a caller timing this cmdlet needs
    /// to know which call they got.
    pub this_call_measured: bool,
}

/// Reads what a dispatch costs on this host.
///
/// The crate measures on the first call in a process, whoever makes it,
/// so this getter can cost 10 to 40 milliseconds of every core once.
/// ThisCallMeasured says whether the call that produced the row is the
/// one that paid.
///
/// # Examples
///
/// `Get-FlynnelHostDispatch`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelHostDispatch",
    alias = "Get-FlyHostDispatch",
    output = ["Flynnel.HostDispatch"]
)]
#[derive(Default)]
pub struct GetFlynnelHostDispatch {}

impl Cmdlet for GetFlynnelHostDispatch {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        // Asked before the profile, because reading the profile is what
        // triggers the measurement. Asked after as well, and the two
        // together say whether this call is the one that paid.
        let before = par_iter::measured_collapse_threshold_ns();
        let profile = par_iter::host_dispatch_profile();
        let after = par_iter::measured_collapse_threshold_ns();
        let at = HOST_MEASURED_AT.load(Ordering::Relaxed);
        ps.write(HostDispatch {
            dispatch_cost_ns: profile.dispatch_cost_ns,
            collapse_threshold_ns: profile.collapse_threshold_ns,
            jec_wake_threshold_ns: profile.jec_wake_threshold_ns,
            measured_collapse_threshold_ns: after,
            source: host_source(at, after),
            measured_at: measured_at_or_none(at),
            this_call_measured: before.is_none() && after.is_some(),
        })
    }
}

/// Measures what a dispatch costs on this host and installs it.
///
/// This takes the cores while it runs, and the numbers it installs
/// decide leaf widths for the rest of the process, so a draw taken
/// under load makes every later dispatch decision from a loaded
/// measurement.
///
/// # Examples
///
/// `Measure-FlynnelHostDispatch`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelHostDispatch",
    alias = "Measure-FlyHostDispatch",
    output = ["Flynnel.HostDispatch"]
)]
#[derive(Default)]
pub struct MeasureFlynnelHostDispatch {}

impl Cmdlet for MeasureFlynnelHostDispatch {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let at = now_unix_s()?;
        let profile = par_iter::calibrate_host_dispatch();
        HOST_MEASURED_AT.store(at, Ordering::Relaxed);
        ps.write(HostDispatch {
            dispatch_cost_ns: profile.dispatch_cost_ns,
            collapse_threshold_ns: profile.collapse_threshold_ns,
            jec_wake_threshold_ns: profile.jec_wake_threshold_ns,
            measured_collapse_threshold_ns: par_iter::measured_collapse_threshold_ns(),
            source: Source::Measured,
            measured_at: Some(at),
            this_call_measured: true,
        })
    }
}

// ---------------------------------------------------------------------
// K-gating
// ---------------------------------------------------------------------

/// Which counting scheme the wave barrier uses.
#[psenum(name = "Flynnel.KGating")]
#[derive(Clone, Copy, Default)]
pub enum KGating {
    /// One shared counter.
    #[default]
    CounterOnly,
    /// A slot per participant.
    PerSlot,
    /// Whichever the calibration picked.
    Auto,
}

impl From<CrateKGating> for KGating {
    fn from(g: CrateKGating) -> Self {
        match g {
            CrateKGating::CounterOnly => KGating::CounterOnly,
            CrateKGating::PerSlot => KGating::PerSlot,
            CrateKGating::Auto => KGating::Auto,
        }
    }
}

/// What a K-gating calibration measured.
#[psclass(name = "Flynnel.KGatingResult")]
#[derive(Clone, Default)]
pub struct KGatingResult {
    /// What the per-slot scheme cost, in nanoseconds.
    pub per_slot_ns: u64,
    /// What the single-counter scheme cost, in nanoseconds.
    pub counter_only_ns: u64,
    /// Which one won.
    pub winner: KGating,
    /// Where this came from.
    pub source: Source,
    /// When this module measured it, in Unix seconds. Null where it did
    /// not.
    pub measured_at: Option<u64>,
}

/// Reads which counting scheme the wave barrier resolves to.
///
/// This reads the cached winner rather than measuring.
///
/// # Examples
///
/// `Get-FlynnelKGating`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelKGating",
    alias = "Get-FlyKGating",
    output = ["Flynnel.KGating"]
)]
#[derive(Default)]
pub struct GetFlynnelKGating {}

impl Cmdlet for GetFlynnelKGating {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let resolved: KGating = CrateKGating::Auto.resolved().into();
        ps.write(resolved)
    }
}

/// Measures both counting schemes and answers what each cost.
///
/// The crate re-runs the microbenchmark on every call rather than
/// consulting its cache, so this measures every time it is asked.
///
/// # Examples
///
/// `Measure-FlynnelKGating`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelKGating",
    alias = "Measure-FlyKGating",
    output = ["Flynnel.KGatingResult"]
)]
#[derive(Default)]
pub struct MeasureFlynnelKGating {}

impl Cmdlet for MeasureFlynnelKGating {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let at = now_unix_s()?;
        let result = k_gating::calibrate_k_gating_verbose();
        KGATING_MEASURED_AT.store(at, Ordering::Relaxed);
        ps.write(KGatingResult {
            per_slot_ns: result.per_slot_ns,
            counter_only_ns: result.counter_only_ns,
            winner: result.winner.into(),
            source: Source::Measured,
            measured_at: Some(at),
        })
    }
}

// ---------------------------------------------------------------------
// Seed hysteresis
// ---------------------------------------------------------------------

/// Reads whether the bisect's seed-depth hysteresis is on.
///
/// One atomic load. The switch sits on a line the bisect reads on every
/// dispatch, so a script polling this in a loop costs the caller and
/// leaves the scheduler's own path alone.
///
/// # Examples
///
/// `Get-FlynnelSeedHysteresis`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelSeedHysteresis",
    alias = "Get-FlySeedHysteresis",
    output = ["System.Boolean"]
)]
#[derive(Default)]
pub struct GetFlynnelSeedHysteresis {}

impl Cmdlet for GetFlynnelSeedHysteresis {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(par_iter::seed_hysteresis())
    }
}

/// Turns the bisect's seed-depth hysteresis on or off, and answers the
/// value it had before.
///
/// On is a switch, so turning the hysteresis off takes the colon form
/// PowerShell uses for a switch given a value.
///
/// # Examples
///
/// `Set-FlynnelSeedHysteresis -On`
///
/// `Set-FlynnelSeedHysteresis -On:$false`
#[cmdlet(
    verb = "Set",
    noun = "FlynnelSeedHysteresis",
    alias = "Set-FlySeedHysteresis",
    output = ["System.Boolean"]
)]
#[derive(Default)]
pub struct SetFlynnelSeedHysteresis {
    /// What to set it to.
    #[param(mandatory, position = 0)]
    pub on: bool,
}

impl Cmdlet for SetFlynnelSeedHysteresis {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let was = par_iter::set_seed_hysteresis(self.on);
        // Read back rather than trust the write, as every other setter
        // in this module does.
        let now = par_iter::seed_hysteresis();
        if now != self.on {
            pwrs::warning!(
                ps,
                "seed hysteresis reads back as {now} after being set to {}; something else is \
                 writing it",
                self.on
            )?;
        }
        ps.write(was)
    }
}

// ---------------------------------------------------------------------
// The persisted store
// ---------------------------------------------------------------------

/// The persisted table.
///
/// Not gated. `persisted-calibration` is a feature of the scheduler,
/// not of this crate, so a cfg naming it here is a condition that is
/// always false and silently compiles nothing. This crate takes the
/// scheduler with its default features, which include that one, by the
/// decision that a PowerShell module ships as one artifact with no way
/// to ask for an extra. If that ever changes, this module stops
/// compiling, which is the assurance a cfg would have removed.
mod store {
    use super::{Source, store_err};
    use pwrs::prelude::*;

    use flynnel::sched::calibration_store::{
        AccelKind as CrateAccelKind, CalibrationStore, HostStamp, LAYOUT_VERSION, calibration_dir,
        table_path,
    };

    /// What kind of device a stored accelerator record describes.
    #[psenum(name = "Flynnel.AccelKind")]
    #[derive(Clone, Copy, Default)]
    pub enum AccelKind {
        /// No device.
        #[default]
        None,
        /// A CUDA device reached through the driver.
        Cuda,
        /// A device joined to the pool as a shared-memory peer.
        GpuPeer,
        /// A discriminant this build does not know. A record written by
        /// a newer build reads as this rather than as None, because a
        /// device this build cannot name is not the same as no device.
        Unknown,
    }

    /// The kind a stored discriminant names.
    fn kind_of(raw: u32) -> AccelKind {
        if raw == CrateAccelKind::None as u32 {
            AccelKind::None
        } else if raw == CrateAccelKind::Cuda as u32 {
            AccelKind::Cuda
        } else if raw == CrateAccelKind::GpuPeer as u32 {
            AccelKind::GpuPeer
        } else {
            AccelKind::Unknown
        }
    }

    /// What identifies this host to the calibration table.
    #[psclass(name = "Flynnel.HostStamp")]
    #[derive(Clone, Default)]
    pub struct Stamp {
        /// The CPU vendor string.
        pub vendor: String,
        /// The CPUID signature.
        pub cpuid_signature: u32,
        /// The target architecture.
        pub arch: String,
        /// The operating system.
        pub os: String,
        /// Primary workers the pool sizes to.
        pub primary_workers: u32,
        /// Every worker including the SMT extensions.
        pub total_workers: u32,
        /// The table layout this build understands.
        pub layout_version: u32,
        /// The canonical text the hash is taken over.
        pub canonical: String,
        /// The hash the table is keyed by.
        pub hash: u64,
    }

    /// The persisted table for this host.
    #[psclass(name = "Flynnel.CalibrationStore")]
    #[derive(Clone, Default)]
    pub struct StoreInfo {
        /// Whether a table exists for this host. False is not an error:
        /// a host that has never calibrated has no table, and that is
        /// the normal first state.
        pub exists: bool,
        /// Where it is, or would be.
        pub path: String,
        /// The layout version this build writes and reads.
        pub layout_version: u32,
        /// The stamp hash the table is keyed by. Null where no table
        /// exists.
        pub stamp_hash: Option<u64>,
        /// The stamp hash this process computes. A mismatch is why a
        /// process measures instead of reading, and it is written out
        /// rather than left to be inferred.
        pub process_stamp_hash: u64,
        /// Whether the two agree. Null where no table exists.
        pub stamp_matches: Option<bool>,
        /// Whether the read caught a stable snapshot. False means a
        /// writer held the table through the whole retry window, so
        /// every figure below is absent because the read failed, not
        /// because the table is empty.
        pub read_settled: bool,
        /// How many accelerator records it holds.
        pub accel_records: Option<u32>,
        /// How many samples the stored CPU record took. Zero means
        /// nothing was ever published.
        pub cpu_samples: Option<u32>,
        /// When the stored CPU record was measured, in Unix seconds.
        pub cpu_measured_at: Option<u64>,
        /// Whether the stored CPU record passes the crate's own trust
        /// check.
        pub cpu_trustworthy: Option<bool>,
    }

    /// Reads what identifies this host to the calibration table.
    ///
    /// A process whose stamp does not match the table's measures
    /// instead of reading, so this is the figure to compare when a host
    /// keeps re-measuring.
    ///
    /// # Examples
    ///
    /// `Get-FlynnelHostStamp`
    #[cmdlet(
        verb = "Get",
        noun = "FlynnelHostStamp",
        alias = "Get-FlyHostStamp",
        output = ["Flynnel.HostStamp"]
    )]
    #[derive(Default)]
    pub struct GetFlynnelHostStamp {}

    impl Cmdlet for GetFlynnelHostStamp {
        fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
            let stamp = HostStamp::detect();
            ps.write(Stamp {
                vendor: stamp.vendor.clone(),
                cpuid_signature: stamp.cpuid_signature,
                arch: stamp.arch.to_string(),
                os: stamp.os.to_string(),
                primary_workers: stamp.primary_workers,
                total_workers: stamp.total_workers,
                layout_version: stamp.layout_version,
                canonical: stamp.canonical(),
                hash: stamp.hash(),
            })
        }
    }

    /// Reads the persisted calibration table for this host.
    ///
    /// A host that has never calibrated has no table. That is the
    /// normal first state, not an error, so it is a row with Exists
    /// false rather than a failure.
    ///
    /// # Examples
    ///
    /// `Get-FlynnelCalibrationStore`
    #[cmdlet(
        verb = "Get",
        noun = "FlynnelCalibrationStore",
        alias = "Get-FlyCalibrationStore",
        output = ["Flynnel.CalibrationStore"]
    )]
    #[derive(Default)]
    pub struct GetFlynnelCalibrationStore {}

    impl Cmdlet for GetFlynnelCalibrationStore {
        fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
            let stamp = HostStamp::detect();
            let process_hash = stamp.hash();
            let Some(dir) = calibration_dir() else {
                return Err(store_err(
                    "no calibration directory is configured on this host; set \
                     FLYNNEL_CALIBRATION_DIR",
                )
                .terminating());
            };
            let path = table_path(&dir, &stamp);
            if !path.exists() {
                return ps.write(StoreInfo {
                    exists: false,
                    path: path.display().to_string(),
                    layout_version: LAYOUT_VERSION,
                    stamp_hash: None,
                    process_stamp_hash: process_hash,
                    stamp_matches: None,
                    read_settled: false,
                    accel_records: None,
                    cpu_samples: None,
                    cpu_measured_at: None,
                    cpu_trustworthy: None,
                });
            }
            let opened = CalibrationStore::open_or_create(&dir, &stamp)
                .map_err(|e| store_err(format!("{e:?}")))?;
            let stamp_hash = opened.stamp_hash();
            let mut row = StoreInfo {
                exists: true,
                path: path.display().to_string(),
                layout_version: LAYOUT_VERSION,
                stamp_hash: Some(stamp_hash),
                process_stamp_hash: process_hash,
                stamp_matches: Some(stamp_hash == process_hash),
                read_settled: false,
                accel_records: None,
                cpu_samples: None,
                cpu_measured_at: None,
                cpu_trustworthy: None,
            };
            match opened.read() {
                Some((cpu, accels)) => {
                    row.read_settled = true;
                    row.accel_records = Some(accels.len() as u32);
                    row.cpu_samples = Some(cpu.samples);
                    row.cpu_measured_at = if cpu.samples == 0 {
                        None
                    } else {
                        Some(cpu.measured_unix_s)
                    };
                    row.cpu_trustworthy = Some(cpu.is_trustworthy());
                }
                None => {
                    // A writer held the table through the whole retry
                    // window. The row still goes out, because the path
                    // and the stamps are real, but the failure is named
                    // rather than left to look like an empty table.
                    ps.write_error(&store_err(
                        "the table did not settle within its retry window, so a writer is \
                         holding it; the record columns are absent because the read failed, \
                         not because the table is empty",
                    ))?;
                }
            }
            ps.write(row)
        }
    }

    /// A stored CPU calibration record.
    #[psclass(name = "Flynnel.CpuCalibration")]
    #[derive(Clone, Default)]
    pub struct CpuRecord {
        /// What one dispatch cost when this was taken, in nanoseconds.
        pub dispatch_cost_ns: u64,
        /// The collapse threshold it carries, in nanoseconds.
        pub collapse_threshold_ns: u64,
        /// The join-and-continue wake threshold, in nanoseconds.
        pub jec_wake_threshold_ns: u64,
        /// When it was measured, in Unix seconds. Null where nothing
        /// was ever published, which reads back as a zeroed record.
        pub measured_at: Option<u64>,
        /// The spread of the draw it came from, in parts per thousand.
        /// Null on an unpublished record.
        pub spread_per_mille: Option<u32>,
        /// How many samples it took. Zero means nothing was published.
        pub samples: u32,
        /// Independent draws of this host whose dispatch median agreed
        /// with this record's. Zero makes the record provisional:
        /// stored so the next draw has something to agree with, and
        /// not served, because one draw cannot say whether its own
        /// median reproduces.
        pub confirmations: u32,
        /// The share of a core the measuring thread actually held while
        /// this was drawn, in parts per thousand. Null where the
        /// platform reports no thread clock.
        ///
        /// SpreadPerMille says whether the samples agreed with each
        /// other. It cannot say whether they agreed on the wrong
        /// number, which is what a draw taken while a neighbour held
        /// half the machine produces: every sample slow, and slow by
        /// about the same amount. This is the figure that separates
        /// those, and it gates nothing.
        pub occupancy_per_mille: Option<u32>,
        /// Whether it passes the crate's own trust check, which is
        /// Samples and Confirmations both above zero.
        ///
        /// Not a spread test. Reproducibility is a property of two
        /// draws, and no statistic over one draw's samples substitutes
        /// for a second draw agreeing.
        pub is_trustworthy: bool,
        /// Where it came from, which for this row is always the table.
        pub source: Source,
    }

    /// Reads the stored CPU calibration for this host.
    ///
    /// A table that exists but was never published reads back zeroed.
    /// Samples is what tells that from a measurement, and the
    /// timestamp is null rather than the epoch.
    ///
    /// # Examples
    ///
    /// `Get-FlynnelCpuCalibration`
    #[cmdlet(
        verb = "Get",
        noun = "FlynnelCpuCalibration",
        alias = "Get-FlyCpuCalibration",
        output = ["Flynnel.CpuCalibration"]
    )]
    #[derive(Default)]
    pub struct GetFlynnelCpuCalibration {}

    impl Cmdlet for GetFlynnelCpuCalibration {
        fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
            let stamp = HostStamp::detect();
            let Some(dir) = calibration_dir() else {
                return Err(store_err(
                    "no calibration directory is configured on this host; set \
                     FLYNNEL_CALIBRATION_DIR",
                )
                .terminating());
            };
            if !table_path(&dir, &stamp).exists() {
                pwrs::warning!(
                    ps,
                    "no calibration table exists for this host yet, so there is no stored \
                     record to read"
                )?;
                return Ok(());
            }
            let opened = CalibrationStore::open_or_create(&dir, &stamp)
                .map_err(|e| store_err(format!("{e:?}")))?;
            let Some((cpu, _)) = opened.read() else {
                return Err(store_err(
                    "the table did not settle within its retry window, which means a writer is \
                     holding it rather than that it is empty",
                )
                .terminating());
            };
            ps.write(CpuRecord {
                dispatch_cost_ns: cpu.dispatch_cost_ns,
                collapse_threshold_ns: cpu.collapse_threshold_ns,
                jec_wake_threshold_ns: cpu.jec_wake_threshold_ns,
                measured_at: if cpu.samples == 0 {
                    None
                } else {
                    Some(cpu.measured_unix_s)
                },
                spread_per_mille: if cpu.samples == 0 {
                    None
                } else {
                    Some(cpu.spread_per_mille)
                },
                samples: cpu.samples,
                confirmations: cpu.confirmations,
                occupancy_per_mille: cpu.occupancy(),
                is_trustworthy: cpu.is_trustworthy(),
                source: Source::Stored,
            })
        }
    }

    /// A stored accelerator calibration record.
    #[psclass(name = "Flynnel.AccelCalibration")]
    #[derive(Clone, Default)]
    pub struct AccelRecord {
        /// What kind of device it describes.
        pub kind: AccelKind,
        /// The raw discriminant, so a record this build cannot name is
        /// still identifiable.
        pub kind_raw: u32,
        /// Which device of that kind.
        pub ordinal: u32,
        /// Its compute capability.
        pub capability: u32,
        /// How many multiprocessors it has.
        pub multiprocessors: u32,
        /// Its clock in kilohertz.
        pub clock_khz: u32,
        /// Its memory in mebibytes.
        pub memory_mib: u32,
        /// The smallest doorbell round trip measured, in nanoseconds.
        pub rtt_min_ns: u64,
        /// The median doorbell round trip, in nanoseconds.
        pub rtt_median_ns: u64,
        /// The 99th percentile doorbell round trip, in nanoseconds.
        pub rtt_p99_ns: u64,
        /// What a launch cost, in nanoseconds.
        pub launch_ns: u64,
        /// The spread of the draw, in parts per thousand.
        pub spread_per_mille: u32,
        /// Whether it passes the crate's own trust check.
        pub is_trustworthy: bool,
        /// The team size the wave costs were measured at. Null where no
        /// wave costs were taken, which the crate marks with a zero
        /// width, and then every Wave field below is null with it.
        pub wave_width: Option<u32>,
        /// One cross-block generation barrier, in nanoseconds.
        pub wave_barrier_ns: Option<u64>,
        /// The host round trip of a wave slice apart from its segments,
        /// in nanoseconds.
        pub wave_fixed_ns: Option<u64>,
        /// One segment on the substrate, in picoseconds.
        pub wave_segment_ps: Option<u64>,
        /// One rebalance apart from the ids it moves, in nanoseconds.
        pub wave_rebalance_fixed_ns: Option<u64>,
        /// Moving one pending id in a rebalance, in picoseconds.
        pub wave_copy_ps_per_id: Option<u64>,
        /// A coupled slice's wait at the first barrier, in nanoseconds.
        pub wave_skew_ns: Option<u32>,
        /// The longest generation of the calibration waves, in
        /// nanoseconds.
        pub wave_generation_ns: Option<u64>,
        /// Where it came from, which for this row is always the table.
        pub source: Source,
    }

    /// Reads the stored accelerator calibrations for this host, all of
    /// them in one call.
    ///
    /// # Examples
    ///
    /// `Get-FlynnelAccelCalibration`
    #[cmdlet(
        verb = "Get",
        noun = "FlynnelAccelCalibration",
        alias = "Get-FlyAccelCalibration",
        output = ["Flynnel.AccelCalibration"]
    )]
    #[derive(Default)]
    pub struct GetFlynnelAccelCalibration {}

    impl Cmdlet for GetFlynnelAccelCalibration {
        fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
            let stamp = HostStamp::detect();
            let Some(dir) = calibration_dir() else {
                return Err(store_err(
                    "no calibration directory is configured on this host; set \
                     FLYNNEL_CALIBRATION_DIR",
                )
                .terminating());
            };
            if !table_path(&dir, &stamp).exists() {
                pwrs::warning!(
                    ps,
                    "no calibration table exists for this host yet, so there is no stored \
                     record to read"
                )?;
                return Ok(());
            }
            let opened = CalibrationStore::open_or_create(&dir, &stamp)
                .map_err(|e| store_err(format!("{e:?}")))?;
            let Some((_, accels)) = opened.read() else {
                return Err(store_err(
                    "the table did not settle within its retry window, which means a writer is \
                     holding it rather than that it is empty",
                )
                .terminating());
            };
            for a in accels {
                let wave = a.wave();
                ps.write(AccelRecord {
                    kind: kind_of(a.kind),
                    kind_raw: a.kind,
                    ordinal: a.ordinal,
                    capability: a.capability,
                    multiprocessors: a.multiprocessors,
                    clock_khz: a.clock_khz,
                    memory_mib: a.memory_mib,
                    rtt_min_ns: a.rtt_min_ns,
                    rtt_median_ns: a.rtt_median_ns,
                    rtt_p99_ns: a.rtt_p99_ns,
                    launch_ns: a.launch_ns,
                    spread_per_mille: a.spread_per_mille,
                    is_trustworthy: a.is_trustworthy(),
                    wave_width: wave.map(|w| w.width),
                    wave_barrier_ns: wave.map(|w| w.barrier_ns),
                    wave_fixed_ns: wave.map(|w| w.fixed_ns),
                    wave_segment_ps: wave.map(|w| w.segment_ps),
                    wave_rebalance_fixed_ns: wave.map(|w| w.rebalance_fixed_ns),
                    wave_copy_ps_per_id: wave.map(|w| w.copy_ps_per_id),
                    wave_skew_ns: wave.map(|w| w.skew_ns),
                    wave_generation_ns: wave.map(|w| w.generation_ns),
                    source: Source::Stored,
                })?;
            }
            Ok(())
        }
    }
}

pub use store::{
    AccelKind, AccelRecord, CpuRecord, GetFlynnelAccelCalibration, GetFlynnelCalibrationStore,
    GetFlynnelCpuCalibration, GetFlynnelHostStamp, Stamp, StoreInfo,
};

// ---------------------------------------------------------------------
// Everything in one call
// ---------------------------------------------------------------------

/// Everything this process currently believes about the host.
#[psclass(name = "Flynnel.Calibration")]
#[derive(Clone, Default)]
pub struct Calibration {
    /// What one pool dispatch costs, in nanoseconds.
    pub dispatch_cost_ns: u64,
    /// Below this total, work collapses inline, in nanoseconds.
    pub collapse_threshold_ns: u64,
    /// The same figure as a measurement, null where none was taken.
    pub measured_collapse_threshold_ns: Option<u64>,
    /// The join-and-continue wake threshold, in nanoseconds.
    pub jec_wake_threshold_ns: u64,
    /// Where the three above came from.
    pub host_source: Source,
    /// The fine-grain class boundary, in nanoseconds.
    pub fine_grain_ns: u64,
    /// The port-heavy class boundary, in nanoseconds.
    pub port_heavy_ns: u64,
    /// The memory-latency class boundary, in nanoseconds.
    pub memory_latency_ns: u64,
    /// The even-leaf spread boundary, in parts per thousand.
    pub cv2_low_per_mille: u64,
    /// The ragged-leaf spread boundary, in parts per thousand.
    pub cv2_high_per_mille: u64,
    /// The trivial-reduce ceiling, in cycles.
    pub trivial_reduce_cycles: u64,
    /// Where the six above came from.
    pub class_source: Source,
    /// Which counting scheme the wave barrier resolves to.
    pub k_gating: KGating,
    /// Whether the bisect's seed-depth hysteresis is on.
    pub seed_hysteresis: bool,
    /// The dispatch profile the process is currently running under.
    pub active_profile: String,
    /// The workload class the process is currently learning.
    pub active_class: crate::types::WorkloadClass,
}

/// Reads everything this process currently believes about the host in
/// one call.
///
/// Reading these one cmdlet at a time is one crossing each. This is one
/// crossing for all of them.
///
/// It reads the host dispatch profile, which the crate measures on the
/// first call in a process, so the first call in a session can cost 10
/// to 40 milliseconds. Get-FlynnelHostDispatch is the one that says
/// whether a given call paid.
///
/// It also writes the seed-hysteresis switch twice to read it, because
/// the crate exposes no getter, and that switch is one the bisect
/// reads on every dispatch. Polling this cmdlet in a tight loop
/// therefore costs the scheduler and not only the caller. Once a
/// second is nothing; once a millisecond is not.
///
/// # Examples
///
/// `Get-FlynnelCalibration`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelCalibration",
    alias = "Get-FlyCalibration",
    output = ["Flynnel.Calibration"]
)]
#[derive(Default)]
pub struct GetFlynnelCalibration {}

impl Cmdlet for GetFlynnelCalibration {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let profile = par_iter::host_dispatch_profile();
        let measured = par_iter::measured_collapse_threshold_ns();
        let host_at = HOST_MEASURED_AT.load(Ordering::Relaxed);
        let thresholds = read_thresholds(adaptive_profile::class_thresholds());
        ps.write(Calibration {
            dispatch_cost_ns: profile.dispatch_cost_ns,
            collapse_threshold_ns: profile.collapse_threshold_ns,
            measured_collapse_threshold_ns: measured,
            jec_wake_threshold_ns: profile.jec_wake_threshold_ns,
            host_source: host_source(host_at, measured),
            fine_grain_ns: thresholds.fine_grain_ns,
            port_heavy_ns: thresholds.port_heavy_ns,
            memory_latency_ns: thresholds.memory_latency_ns,
            cv2_low_per_mille: thresholds.cv2_low_per_mille,
            cv2_high_per_mille: thresholds.cv2_high_per_mille,
            trivial_reduce_cycles: thresholds.trivial_reduce_cycles,
            class_source: thresholds.source,
            k_gating: CrateKGating::Auto.resolved().into(),
            seed_hysteresis: par_iter::seed_hysteresis(),
            active_profile: format!("{:?}", adaptive_profile::active_dispatch_profile()),
            active_class: adaptive_profile::active_workload_class().into(),
        })
    }
}

/// Classifies a mean leaf time and a spread against the live
/// boundaries, without running anything.
///
/// This is the same decision the scheduler makes about its own
/// measurements, so a caller holding leaf figures of their own can ask
/// what class they would be learned as.
///
/// # Examples
///
/// `Get-FlynnelWorkloadClass -MeanNs 300 -Cv2PerMille 40`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelWorkloadClass",
    alias = "Get-FlyWorkloadClass",
    output = ["Flynnel.WorkloadClass"]
)]
#[derive(Default)]
pub struct GetFlynnelWorkloadClass {
    /// The mean leaf time in nanoseconds.
    #[param(mandatory, position = 0)]
    pub mean_ns: u64,
    /// The squared coefficient of variation in parts per thousand.
    #[param(mandatory, position = 1)]
    pub cv2_per_mille: u64,
}

impl Cmdlet for GetFlynnelWorkloadClass {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        if self.mean_ns == 0 {
            return Err(arg_err("MeanNs must be above zero").terminating());
        }
        let class: crate::types::WorkloadClass =
            adaptive_profile::classify_observed(self.mean_ns, self.cv2_per_mille).into();
        ps.write(class)
    }
}

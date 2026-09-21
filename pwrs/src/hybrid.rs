//! The CPU half and the device half of one call, and what the
//! scheduler learns from running them.
//!
//! # These cmdlets measure a placement; they do not transform your
//! data
//!
//! Every one of them is a `Measure-`, and that is the honest verb. A
//! caller who wants an array transformed has `Invoke-FlynnelMap`,
//! which puts the whole pool on it. The hybrid shapes split one call
//! two ways - the calling thread and one backend thread - so on a host
//! with no device they are strictly slower than the kernel family at
//! the same work. What they produce that nothing else does is the
//! measurement: how long each half took, which half the call site has
//! learned to prefer at this size, and what share of a divisible range
//! it now gives the CPU.
//!
//! So the work is declared and synthetic: `-Count` items of the
//! module's own making, put through a declared operation `-Repetitions`
//! times. Nothing crosses the PowerShell boundary but the report, and
//! the numbers describe the hybrid plumbing rather than the marshalling
//! in front of it.
//!
//! # With no device registered the backend half is the CPU backend
//!
//! `JobPlan::pick_backend` falls through to the CPU backend when no
//! hint is set or the hinted backend is not registered, and the CPU
//! backend runs the half on a spawned thread. So both halves really do
//! run concurrently and the reports are real; what they are not is a
//! device measurement. Every report carries `BackendIsCpu` so a reading
//! taken on a deviceless host can never be mistaken for one taken
//! against a card.
//!
//! # Each half runs its own range serially
//!
//! A half could dispatch its range across the pool, and a real caller's
//! CPU half usually would. These do not: both halves would then be
//! drawing on the same workers, and the split's numbers would describe
//! that contention rather than the split. The crate says the same thing
//! from the other side - `join_hybrid` reads only the plan's backend
//! hint, and the leaf shape and worker knobs belong to whatever each
//! half dispatches internally, which is a separate call with a plan of
//! its own.
//!
//! # One call site for the whole module
//!
//! The learned state hangs off the caller's source location, and from
//! the crate's point of view that location is this file. Every script
//! in a session shares one site per cmdlet, bucketed by `log2(count)`.
//! `Reset-FlynnelCallSite` clears it, and a script that wants an
//! unlearned measurement resets first.
//!
//! `Get-FlynnelCallSite` reads the site-wide half of it: the two arm
//! ewmas and the overall split share. The per-bucket state that
//! actually decides a placement at one size is not on that row, and
//! the only reading of it is the report each of these cmdlets
//! returns. A caller following one bucket watches the reports rather
//! than the site.

use pwrs::prelude::*;

use flynnel::backend::registry::backend_by_id;
use flynnel::backend::Backend as CrateBackend;
use flynnel::sched::call_site::Placement as CratePlacement;
use flynnel::{
    JobPlan, SplitReport, hybrid_auto, hybrid_auto_split_ranges, hybrid_pipeline, join_hybrid,
};

use crate::backends::BackendKind;
use crate::host::arg_err;
use crate::kernels::{MapOp, MapOperands, map_each};

/// Which side of a hybrid call ran.
#[psenum(name = "Flynnel.Placement")]
#[derive(Clone, Copy, Default)]
pub enum PlacementKind {
    /// Only the CPU implementation ran.
    #[default]
    Cpu,
    /// Only the backend implementation ran.
    Backend,
    /// Both ran concurrently and both were timed, which is what a cold
    /// size bucket and a scheduled re-probe do.
    Race,
}

impl From<CratePlacement> for PlacementKind {
    fn from(p: CratePlacement) -> Self {
        match p {
            CratePlacement::Cpu => PlacementKind::Cpu,
            CratePlacement::Backend => PlacementKind::Backend,
            CratePlacement::Race => PlacementKind::Race,
        }
    }
}

/// What both halves call. Shared rather than cloned because the two
/// run at the same time and a `Box` cannot be in two places.
type Body = std::sync::Arc<dyn Fn(&mut f64) + Send + Sync>;

/// Builds the declared body once and refuses a missing operand before
/// any half starts, so a bad argument is an error rather than a half
/// that panics on a backend thread.
fn body(op: MapOp, operands: MapOperands) -> PsResult<Body> {
    Ok(map_each(op, operands)?.into())
}

/// One half's work: build its own range of items, put each through the
/// declared body `reps` times, and answer a checksum.
///
/// The checksum is returned rather than discarded so the optimizer
/// cannot delete the loop, and so the two halves of a shape whose
/// contract says they compute the same thing can be compared.
fn run_range(
    lo: usize,
    hi: usize,
    reps: u32,
    each: &(dyn Fn(&mut f64) + Send + Sync),
) -> f64 {
    let mut acc = 0.0f64;
    for i in lo..hi {
        // A value that varies with the index and stays in a range every
        // declared operation answers finitely for: above zero, so Log
        // and Sqrt are defined, and small, so Exp does not overflow.
        let mut x = 1.0 + (i % 64) as f64 * 0.125;
        for _ in 0..reps {
            each(&mut x);
        }
        acc += x;
    }
    acc
}

/// The plan a hybrid call runs under: the caller's, or a bare plan at
/// this count when none is given.
///
/// The batch size is overwritten either way. The learned model keys on
/// its base-2 logarithm, so a plan the caller built for another size
/// would record this call's timings against the wrong bucket and every
/// later placement decision at both sizes would read them.
fn hybrid_plan(plan: Option<&crate::plan::Plan>, count: u32) -> PsResult<JobPlan> {
    let mut p = match plan {
        Some(p) => p.to_job_plan()?,
        None => JobPlan::bare(0, count),
    };
    p.batch_size = count;
    Ok(p)
}

/// Which backend the plan resolves to, as a row column rather than a
/// word in the help.
fn resolved_backend(plan: &JobPlan) -> (BackendKind, bool) {
    match plan.backend_hint {
        Some(b) if backend_by_id(&b).is_some() => {
            (BackendKind::from(b), matches!(b, CrateBackend::Cpu))
        }
        // pick_backend falls through to the CPU backend both when no
        // hint is set and when the hinted one is not registered, and
        // the two are different facts about the host. The column says
        // Cpu either way because that is what ran; the help says why.
        _ => (BackendKind::Cpu, true),
    }
}

// ---------------------------------------------------------------------
// join_hybrid
// ---------------------------------------------------------------------

/// Both halves of one hybrid call, and what each cost.
#[psclass(name = "Flynnel.HybridJoin")]
#[derive(Clone, Default)]
pub struct HybridJoin {
    /// Items the CPU half processed.
    pub cpu_items: u32,
    /// Items the backend half processed.
    pub backend_items: u32,
    /// Wall time of the CPU half.
    pub cpu_ns: u64,
    /// Wall time of the backend half, from the calling thread's point
    /// of view, so it includes the hand-off both ways.
    pub backend_ns: u64,
    /// Wall time of the whole call. Below the sum of the two halves
    /// when they really overlapped, which is the property the shape
    /// exists for.
    pub total_ns: u64,
    /// The backend the plan resolved to.
    pub backend: BackendKind,
    /// Whether that backend is the CPU backend, in which case both
    /// halves ran on host threads and this is not a device reading.
    pub backend_is_cpu: bool,
    /// The CPU half's checksum.
    pub cpu_checksum: f64,
    /// The backend half's checksum.
    pub backend_checksum: f64,
}

/// Runs two halves of one declared operation concurrently, the first on
/// the calling thread and the second on the plan's backend, and reports
/// what each cost.
///
/// This is the plain MIMT shape with no learning in it: the split is
/// fixed at -CpuShare and nothing is recorded against the call site.
/// Measure-FlynnelHybridPlacement and Measure-FlynnelHybridSplit are
/// the learning forms.
///
/// The two halves overlap, so TotalNs below CpuNs + BackendNs is the
/// answer that says the shape did its job. On a host with no registered
/// device BackendIsCpu is true and both halves ran on host threads:
/// still two threads, still concurrent, but not a device measurement.
///
/// # Examples
///
/// `Measure-FlynnelHybridJoin -Count 100000 -Operation Sqrt`
///
/// `Measure-FlynnelHybridJoin -Count 1000000 -Operation Exp -Repetitions 4 -CpuShare 250`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelHybridJoin",
    alias = "Measure-FlyHybridJoin",
    output = ["Flynnel.HybridJoin"]
)]
#[derive(Default)]
pub struct MeasureFlynnelHybridJoin {
    /// How many items the whole call covers.
    #[param(mandatory, position = 0)]
    pub count: u32,
    /// The declared operation each item goes through.
    #[param(position = 1)]
    pub operation: MapOp,
    /// How many times to apply it per item. The lever for per-item
    /// weight: at one repetition the call is dominated by the hand-off,
    /// and a shape's breakeven is where it stops being.
    #[param(position = 2)]
    pub repetitions: Option<u32>,
    /// The CPU half's share in parts per thousand. Half and half when
    /// unset.
    #[param]
    pub cpu_share: Option<u32>,
    /// The plan whose backend hint chooses the device half. A bare plan
    /// at this count when unset, which resolves to the CPU backend.
    #[param]
    pub plan: Option<crate::plan::Plan>,
    /// Clamp's lower bound.
    #[param]
    pub min: Option<f64>,
    /// Clamp's upper bound.
    #[param]
    pub max: Option<f64>,
    /// Scale's multiplier.
    #[param]
    pub factor: Option<f64>,
    /// Offset's addend.
    #[param]
    pub addend: Option<f64>,
}

impl MeasureFlynnelHybridJoin {
    fn operands(&self) -> MapOperands {
        MapOperands {
            min: self.min,
            max: self.max,
            factor: self.factor,
            addend: self.addend,
        }
    }
}

impl Cmdlet for MeasureFlynnelHybridJoin {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let n = self.count as usize;
        if n == 0 {
            return Err(arg_err("Count must be above zero").terminating());
        }
        let share = self.cpu_share.unwrap_or(500);
        if share > 1000 {
            return Err(arg_err("CpuShare is parts per thousand, so at most 1000").terminating());
        }
        let reps = self.repetitions.unwrap_or(1).max(1);
        let each = body(self.operation, self.operands())?;
        let plan = hybrid_plan(self.plan.as_ref(), self.count)?;
        let (kind, is_cpu) = resolved_backend(&plan);

        let mid = ((n as u64 * share as u64) / 1000) as usize;
        let for_backend = std::sync::Arc::clone(&each);
        let t0 = std::time::Instant::now();
        let ((cpu_sum, cpu_ns), (dev_sum, dev_ns)) = join_hybrid(
            &plan,
            || {
                let t = std::time::Instant::now();
                let s = run_range(0, mid, reps, each.as_ref());
                (s, t.elapsed().as_nanos() as u64)
            },
            move || {
                let t = std::time::Instant::now();
                let s = run_range(mid, n, reps, for_backend.as_ref());
                (s, t.elapsed().as_nanos() as u64)
            },
        );
        let total_ns = t0.elapsed().as_nanos() as u64;

        ps.write(HybridJoin {
            cpu_items: mid as u32,
            backend_items: (n - mid) as u32,
            cpu_ns,
            backend_ns: dev_ns,
            total_ns,
            backend: kind,
            backend_is_cpu: is_cpu,
            cpu_checksum: cpu_sum,
            backend_checksum: dev_sum,
        })
    }
}

// ---------------------------------------------------------------------
// hybrid_auto
// ---------------------------------------------------------------------

/// Which side a learned call site chose, and what it cost.
#[psclass(name = "Flynnel.HybridPlacement")]
#[derive(Clone, Default)]
pub struct HybridPlacement {
    /// The side that ran. Race means the bucket was cold or due a
    /// re-probe, so both ran and both were timed.
    pub placement: PlacementKind,
    /// Items the call covered.
    pub count: u32,
    /// Wall time of the whole call.
    pub total_ns: u64,
    /// The size bucket the model keys on, which is log2 of the count.
    pub bucket: u32,
    /// The backend the plan resolved to.
    pub backend: BackendKind,
    /// Whether that backend is the CPU backend, in which case both
    /// sides of a race ran on host threads.
    pub backend_is_cpu: bool,
    /// The checksum, which is the same whichever side ran because the
    /// two implementations are the same declared operation.
    pub checksum: f64,
}

/// Runs one declared operation through the placement the call site has
/// learned for this size, and reports which side that was.
///
/// The model: a cold size bucket races both sides and times each, a
/// warm bucket runs only the cheaper side, and every thirty-second call
/// in a bucket re-races so the model tracks drift. Racing is the
/// calibration - the first call in a bucket pays double work once,
/// instead of an offline calibration pass.
///
/// The learned state belongs to this cmdlet, not to your script: the
/// site is this module's source location, so every caller in the
/// session shares one per size bucket. Reset-FlynnelCallSite clears it
/// and Get-FlynnelCallSite reads it.
///
/// A run of this on a host with no device will settle on Cpu, because
/// the backend side is the CPU backend reached through a thread
/// hand-off and is therefore the slower of two identical bodies. That
/// is the model working, not the model failing.
///
/// # Examples
///
/// `1..40 | ForEach-Object { Measure-FlynnelHybridPlacement -Count 65536 -Operation Sqrt } |
///     Group-Object Placement | Select-Object Name, Count`
///
/// `Reset-FlynnelCallSite; Measure-FlynnelHybridPlacement -Count 1024 -Operation Exp`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelHybridPlacement",
    alias = "Measure-FlyHybridPlacement",
    output = ["Flynnel.HybridPlacement"]
)]
#[derive(Default)]
pub struct MeasureFlynnelHybridPlacement {
    /// How many items the call covers. Also the size bucket, through
    /// its base-2 logarithm.
    #[param(mandatory, position = 0)]
    pub count: u32,
    /// The declared operation each item goes through.
    #[param(position = 1)]
    pub operation: MapOp,
    /// How many times to apply it per item.
    #[param(position = 2)]
    pub repetitions: Option<u32>,
    /// The plan whose backend hint chooses the device side.
    #[param]
    pub plan: Option<crate::plan::Plan>,
    /// Clamp's lower bound.
    #[param]
    pub min: Option<f64>,
    /// Clamp's upper bound.
    #[param]
    pub max: Option<f64>,
    /// Scale's multiplier.
    #[param]
    pub factor: Option<f64>,
    /// Offset's addend.
    #[param]
    pub addend: Option<f64>,
}

impl Cmdlet for MeasureFlynnelHybridPlacement {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let n = self.count as usize;
        if n == 0 {
            return Err(arg_err("Count must be above zero").terminating());
        }
        let reps = self.repetitions.unwrap_or(1).max(1);
        let operands = MapOperands {
            min: self.min,
            max: self.max,
            factor: self.factor,
            addend: self.addend,
        };
        let each = body(self.operation, operands)?;
        let plan = hybrid_plan(self.plan.as_ref(), self.count)?;
        let (kind, is_cpu) = resolved_backend(&plan);

        let for_backend = std::sync::Arc::clone(&each);
        let t0 = std::time::Instant::now();
        let (checksum, placement) = hybrid_auto(
            &plan,
            || run_range(0, n, reps, each.as_ref()),
            move || run_range(0, n, reps, for_backend.as_ref()),
        );
        let total_ns = t0.elapsed().as_nanos() as u64;

        ps.write(HybridPlacement {
            placement: PlacementKind::from(placement),
            count: self.count,
            total_ns,
            bucket: usize::BITS - 1 - n.leading_zeros(),
            backend: kind,
            backend_is_cpu: is_cpu,
            checksum,
        })
    }
}

// ---------------------------------------------------------------------
// hybrid_auto_split_ranges
// ---------------------------------------------------------------------

/// How a learned split divided one range, and what each side cost.
#[psclass(name = "Flynnel.HybridSplit")]
#[derive(Clone, Default)]
pub struct HybridSplit {
    /// Items the CPU side took.
    pub cpu_items: u32,
    /// Items the backend side took.
    pub backend_items: u32,
    /// Wall time of the CPU side.
    pub cpu_ns: u64,
    /// Wall time of the backend side.
    pub backend_ns: u64,
    /// The share this call used, in parts per thousand. An even split
    /// until both sides have been measured at this size.
    pub cpu_share_per_mille: u32,
    /// Wall time of the whole call.
    pub total_ns: u64,
    /// The size bucket the model keys on.
    pub bucket: u32,
    /// The backend the plan resolved to.
    pub backend: BackendKind,
    /// Whether that backend is the CPU backend.
    pub backend_is_cpu: bool,
}

/// Divides one range between the CPU and the backend by the
/// per-item throughputs this call site has measured, runs both sides
/// concurrently, and reports the division.
///
/// A size the model has no data for does not start even: it reads the
/// site's overall ratio, which is whatever the last calls at other
/// sizes established. An even first share means the site as a whole is
/// even, not that this size is unmeasured.
///
/// The share is the ratio of the two measured per-item costs, so it
/// stays even while both sides cost the same per item. On a host with
/// no device that is the ordinary case, because the backend side is
/// the CPU backend running the same body, and each side's clock starts
/// inside its own half so the thread hand-off is outside both
/// readings. An even split holding there is the model working.
///
/// -BackendRepetitions is how to see it move. It applies the body more
/// times on the backend side than on the CPU side, which is a backend
/// with a different per-item cost and is the thing the split model
/// exists to track. It deliberately breaks the rule that both sides
/// compute the same result, so the two sides' outputs are no longer
/// comparable; nothing here compares them.
///
/// The model records per-item cost as a whole number of nanoseconds,
/// so at a per-item cost of a few nanoseconds both sides truncate to
/// the same integer and the share cannot move whatever the real
/// difference is. -Repetitions is the lever that lifts a per-item cost
/// far enough above that floor to be resolved.
///
/// The learned state is this module's, shared by every caller in the
/// session and keyed by size bucket, the same as
/// Measure-FlynnelHybridPlacement. Reset-FlynnelCallSite clears it.
///
/// # Examples
///
/// `1..10 | ForEach-Object { Measure-FlynnelHybridSplit -Count 200000 -Operation Sqrt } |
///     Select-Object CpuSharePerMille, CpuNs, BackendNs`
///
/// `1..10 | ForEach-Object { Measure-FlynnelHybridSplit -Count 65536 -Operation Exp
///     -Repetitions 8 -BackendRepetitions 32 } | Select-Object CpuSharePerMille`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelHybridSplit",
    alias = "Measure-FlyHybridSplit",
    output = ["Flynnel.HybridSplit"]
)]
#[derive(Default)]
pub struct MeasureFlynnelHybridSplit {
    /// How many items the call covers.
    #[param(mandatory, position = 0)]
    pub count: u32,
    /// The declared operation each item goes through.
    #[param(position = 1)]
    pub operation: MapOp,
    /// How many times to apply it per item on the CPU side, and on
    /// the backend side unless BackendRepetitions says otherwise.
    #[param(position = 2)]
    pub repetitions: Option<u32>,
    /// How many times to apply it per item on the backend side, which
    /// is how a backend with a different per-item cost is modelled.
    /// Same as Repetitions when unset, which is the case where the
    /// share has no reason to move.
    #[param]
    pub backend_repetitions: Option<u32>,
    /// The plan whose backend hint chooses the device side.
    #[param]
    pub plan: Option<crate::plan::Plan>,
    /// Clamp's lower bound.
    #[param]
    pub min: Option<f64>,
    /// Clamp's upper bound.
    #[param]
    pub max: Option<f64>,
    /// Scale's multiplier.
    #[param]
    pub factor: Option<f64>,
    /// Offset's addend.
    #[param]
    pub addend: Option<f64>,
}

impl Cmdlet for MeasureFlynnelHybridSplit {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let n = self.count as usize;
        if n == 0 {
            return Err(arg_err("Count must be above zero").terminating());
        }
        let reps = self.repetitions.unwrap_or(1).max(1);
        let operands = MapOperands {
            min: self.min,
            max: self.max,
            factor: self.factor,
            addend: self.addend,
        };
        let each = body(self.operation, operands)?;
        let plan = hybrid_plan(self.plan.as_ref(), self.count)?;
        let (kind, is_cpu) = resolved_backend(&plan);

        let for_backend = std::sync::Arc::clone(&each);
        let t0 = std::time::Instant::now();
        let backend_reps = self.backend_repetitions.unwrap_or(reps).max(1);
        let report: SplitReport = hybrid_auto_split_ranges(
            &plan,
            n,
            |r| {
                std::hint::black_box(run_range(r.start, r.end, reps, each.as_ref()));
            },
            move |r| {
                std::hint::black_box(run_range(
                    r.start,
                    r.end,
                    backend_reps,
                    for_backend.as_ref(),
                ));
            },
        );
        let total_ns = t0.elapsed().as_nanos() as u64;

        ps.write(HybridSplit {
            cpu_items: report.cpu_items as u32,
            backend_items: report.backend_items as u32,
            cpu_ns: report.cpu_ns,
            backend_ns: report.backend_ns,
            cpu_share_per_mille: report.cpu_share_per_mille,
            total_ns,
            bucket: usize::BITS - 1 - n.leading_zeros(),
            backend: kind,
            backend_is_cpu: is_cpu,
        })
    }
}

// ---------------------------------------------------------------------
// hybrid_pipeline
// ---------------------------------------------------------------------

/// What a three-stage pipeline produced and what it cost.
#[psclass(name = "Flynnel.HybridPipeline")]
#[derive(Clone, Default)]
pub struct HybridPipelineRun {
    /// Inputs put through the pipeline.
    pub inputs: u32,
    /// Results it produced. Below Inputs would mean a stage dropped
    /// work, which is why the column is here rather than assumed.
    pub outputs: u32,
    /// Items each input carries through the three stages.
    pub width: u32,
    /// Wall time of the whole run, from the first input to the last
    /// result.
    pub total_ns: u64,
    /// Wall time divided by the number of inputs. In steady state this
    /// approaches the slowest stage rather than the sum of the three,
    /// which is what pipelining buys.
    pub ns_per_input: u64,
    /// The sum of every result, so the run cannot be optimized away and
    /// two runs over the same arguments can be compared.
    pub checksum: f64,
}

/// Runs a three-stage pipeline - a CPU stage, then the device stage,
/// then a CPU stage - over a sequence of inputs, with the stages on
/// their own threads so each input's later stage overlaps the next
/// input's earlier one.
///
/// This is the coupled shape: propose on the CPU, evaluate on the
/// device, accept on the CPU, one iteration feeding the next.
/// Measure-FlynnelHybridJoin handles a single pair with no pipelining
/// across iterations.
///
/// After the pipeline fills, throughput is set by the slowest stage
/// alone, so NsPerInput at a large Count is the reading that matters
/// and a single input measures the fill instead.
///
/// The device stage runs on an ordinary host thread here, as every
/// stage does; the plan's backend hint is not read by the crate's
/// pipeline. The stage is called the device stage because that is its
/// place in the shape, not because a card ran it.
///
/// # Examples
///
/// `Measure-FlynnelHybridPipeline -Count 200 -Width 4096`
///
/// `Measure-FlynnelHybridPipeline -Count 500 -Width 1024 -PreOperation Sqrt -PostOperation Abs`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelHybridPipeline",
    alias = "Measure-FlyHybridPipeline",
    output = ["Flynnel.HybridPipeline"]
)]
#[derive(Default)]
pub struct MeasureFlynnelHybridPipeline {
    /// How many inputs to push through.
    #[param(mandatory, position = 0)]
    pub count: u32,
    /// How many items each input carries.
    #[param(position = 1)]
    pub width: Option<u32>,
    /// The declared operation the first CPU stage applies.
    #[param]
    pub pre_operation: MapOp,
    /// The declared operation the device stage applies before reducing
    /// the input to one number.
    #[param]
    pub device_operation: MapOp,
    /// The declared operation the last CPU stage applies to that
    /// number.
    #[param]
    pub post_operation: MapOp,
}

impl Cmdlet for MeasureFlynnelHybridPipeline {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let n = self.count as usize;
        if n == 0 {
            return Err(arg_err("Count must be above zero").terminating());
        }
        let width = self.width.unwrap_or(1024).max(1) as usize;
        // Every stage's body is built here, before a thread exists, so
        // an operand a stage needs and has not got is an error record
        // rather than a panic three threads deep.
        //
        // Boxed rather than shared, because each body goes to exactly
        // one stage and a Box is callable where an Arc is not.
        let pre = map_each(self.pre_operation, MapOperands::default())?;
        let dev = map_each(self.device_operation, MapOperands::default())?;
        let post = map_each(self.post_operation, MapOperands::default())?;
        let plan = JobPlan::bare(0, self.count);

        let t0 = std::time::Instant::now();
        let results: Vec<f64> = hybrid_pipeline(
            &plan,
            0..n,
            move |seed: usize| -> Vec<f64> {
                let mut v: Vec<f64> = (0..width)
                    .map(|i| 1.0 + ((seed + i) % 64) as f64 * 0.125)
                    .collect();
                for x in &mut v {
                    pre(x);
                }
                v
            },
            move |mut v: Vec<f64>| -> f64 {
                for x in &mut v {
                    dev(x);
                }
                v.iter().sum()
            },
            move |mut s: f64| -> f64 {
                post(&mut s);
                s
            },
        );
        let total_ns = t0.elapsed().as_nanos() as u64;

        ps.write(HybridPipelineRun {
            inputs: self.count,
            outputs: results.len() as u32,
            width: width as u32,
            total_ns,
            ns_per_input: total_ns / self.count.max(1) as u64,
            checksum: results.iter().sum(),
        })
    }
}

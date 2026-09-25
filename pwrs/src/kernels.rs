//! The declared kernels: the work this module owns and Flynnel's own
//! workers run.
//!
//! Flynnel's parallel primitives take Rust closures. A script has none
//! to give, and this module never runs a script block on a worker, so
//! the way a shell reaches the scheduler's workers is to name a body the
//! module already owns. Each cmdlet here is one such body over one
//! substrate: arrays and numbers, files, or text.
//!
//! # One implementation, in the crate
//!
//! Every kernel's partition, per-block work and combine is
//! `flynnel::kernels`, plain Rust with no pool behind it. A cmdlet here
//! resolves its managed input, builds the crate's job for it, runs the
//! job's blocks on this module's pool, and writes the answer as this
//! module's own classes. A library that links the crate runs the same
//! jobs through `flynnel_run_chunks_v1` on this module's pool, so the two
//! answer alike to the bit: a job cuts its input from the input alone and
//! folds its blocks in block order, so neither the worker count nor the
//! plan reaches an answer.
//!
//! # A kernel body calls nothing managed, and the compiler only
//! catches half of that
//!
//! Flynnel's workers are threads .NET has never heard of: not garbage
//! collection roots, never suspended at a safepoint, not competing
//! with the engine's thread pool. A reverse call into the managed
//! vtable attaches the calling thread to the runtime and it loses all
//! of that for the life of the process.
//!
//! Half of this the type system enforces. `Pipeline` is `!Send` and
//! `!Sync` by construction, so no closure handed to a parallel
//! primitive can capture it: the stream calls, the stopping check, the
//! session state and a script block are all refused at compile time.
//!
//! The other half it does not. `PsObject` is `unsafe impl Send` and
//! `unsafe impl Sync` on purpose, because a GCHandle may be held from
//! any thread, and both `PsObject::get` and `PsObject::pin` take
//! `&self` and call the vtable. A `Send + Sync` closure may therefore
//! capture one and call either, and that compiles.
//!
//! So the rule is a rule rather than a guarantee: **resolve every
//! managed value before the parallel section and hand the job plain Rust
//! data.** Pinning is where this is easy to get wrong, because `pin` is
//! exactly what a kernel wants when its input is a shell `double[]`, and
//! calling it inside a block instead of before the job reads naturally
//! and attaches every worker that runs the block.
//!
//! Every body below keeps to it: the jobs see `&mut [f64]`, `&[u8]`,
//! strings and indices, and every `ps.write` sits outside the parallel
//! section.
//!
//! # The shape every kernel shares
//!
//! The whole input crosses in one call, and the whole answer crosses
//! back in one. A per-item crossing costs 1907 nanoseconds in
//! PowerShell 7.6 against 4.5 amortized, so an array arrives as an
//! array and a file list arrives as a list.
//!
//! # Both directions, and the return is the one that is easy to lose
//!
//! A `Vec` handed to the pipeline is enumerated by default: one
//! record per element, at 1712 ns each in PowerShell 7.6 and 7955 in
//! Windows PowerShell. Every bulk answer here is therefore wrapped in
//! `PsArray`, which writes it as one object.
//!
//! Measured on pc2 before the wrap: Invoke-FlynnelMap over 200,000
//! elements cost 114.62 ms while Measure-FlynnelReduce over the same
//! input cost 38.87 ms. Both pay the same input crossing and the same
//! trivial arithmetic; the 75 ms between them was the return, one
//! record at a time.
//!
//! What this changes for a caller: `$y = Invoke-FlynnelMap ...` gives
//! the array, as before. `Invoke-FlynnelMap ... | ForEach-Object` now
//! receives the array as one item rather than a stream of elements.
//! That is the trade the module is built around, and piping 200,000
//! doubles one at a time is the cost it exists to avoid.
//!
//! # The input has a fast path too, and it is silent
//!
//! A typed array crosses as one pinned copy. Anything else, including
//! the `Object[]` that `1..$n | ForEach-Object { ... }` produces, is
//! read element by element. Pass `[double[]]$x` rather than `$x` where
//! the array was built in the shell.
//!
//! `-Plan` is optional everywhere. Without one the kernel builds a plan
//! whose band comes from the item count, capped at the Hierarchical
//! band so a single-NUMA host collapses it to Local rather than
//! federating work it has one node for. With one, the caller's plan
//! governs how the job's blocks are dispatched. It does not change the
//! blocks, so it does not change the answer.
//!
//! `-Verbose` reports the plan that ran the work, the workers it
//! resolved to and the leaves it asked for.
//!
//! An input that cannot be read is a non-terminating error record
//! naming it, and the count of those refusals is written at the end, so
//! a run over a thousand files reports the nine it could not open
//! rather than stopping at the first or silently returning 991 rows.

use std::sync::Mutex;

use pwrs::prelude::*;

use flynnel::kernels::{self, BlockError, Job, Refusal, RefusalId};
use flynnel::sched::par_iter::for_each_chunk_indexed_min_leaf;

use crate::plan::Plan;

/// The error for a kernel argument the module cannot take.
fn arg_err(message: impl Into<String>) -> PsError {
    PsError::new(
        ErrorCategory::InvalidArgument,
        "FlynnelArgument",
        message.into(),
    )
}

/// A kernel's refusal as the error record the module writes: an argument
/// or an internal refusal stops the cmdlet, an unreadable input is a
/// record beside the rows that did read.
fn refusal_err(r: Refusal) -> PsError {
    match r.id {
        RefusalId::Argument => {
            PsError::new(ErrorCategory::InvalidArgument, r.id.as_str(), r.message).terminating()
        }
        RefusalId::Unreadable => PsError::new(ErrorCategory::ReadError, r.id.as_str(), r.message),
        RefusalId::Internal => {
            PsError::new(ErrorCategory::InvalidResult, r.id.as_str(), r.message).terminating()
        }
    }
}

// ---------------------------------------------------------------------
// The plan a kernel runs under
// ---------------------------------------------------------------------

/// The band for an item count, capped at the top of the Hierarchical
/// range.
///
/// Flynnel reads `k_outer` as log2 of the data size and bands it:
/// 0 to 4 inline, 5 to 7 local, 8 to 10 hierarchical, 11 and over
/// federated. The cap is at 10 because the tier pick collapses
/// Hierarchical to Local on a single-NUMA host and has no such
/// collapse for Federated, so a box with one node would otherwise be
/// asked for per-node arenas it does not have.
pub(crate) fn band_for(n: usize) -> u8 {
    if n <= 1 {
        return 0;
    }
    let bits = usize::BITS - (n - 1).leading_zeros();
    bits.min(10) as u8
}

/// The plan this dispatch runs under: the caller's if they gave one,
/// otherwise one sized to the input.
///
/// A caller's plan is built into the crate's own here rather than held
/// as one, so a shape whose numbers are missing is refused at the
/// kernel that would have run it.
fn kernel_plan(caller: Option<&Plan>, n: usize) -> PsResult<flynnel::JobPlan> {
    match caller {
        Some(p) => p.to_job_plan(),
        None => Ok(flynnel::JobPlan::new(
            band_for(n),
            n.min(u32::MAX as usize) as u32,
        )),
    }
}

/// Report the plan that ran the work on the verbose stream.
fn say_plan(ps: &Pipeline<'_>, plan: &flynnel::JobPlan, n: usize, what: &str) -> PsResult<()> {
    pwrs::verbose!(
        ps,
        "{what}: {n} items, k_outer {}, batch {}, {} worker(s), {} leaf/worker, tier {:?}",
        plan.k_outer,
        plan.batch_size,
        plan.resolved_workers(),
        plan.effective_leaves_per_worker(),
        flynnel::sched::plan::pick_tier(plan, flynnel::numa_topology())
    )
}

/// Write the refusal tally after a run that produced error records.
fn say_refusals(ps: &Pipeline<'_>, refused: usize, total: usize) -> PsResult<()> {
    match kernels::refusal_tally(refused, total) {
        Some(tally) => pwrs::warning!(ps, "{tally}"),
        None => Ok(()),
    }
}

/// Run a job's blocks on this module's pool, phase by phase, and answer
/// what it answers.
///
/// The job fixes its own blocks, so the plan decides only how they are
/// dispatched: one leaf holds one or more whole blocks, and which worker
/// ran which block cannot reach the answer.
fn run_on_pool<J: Job>(plan: &flynnel::JobPlan, job: J) -> PsResult<J::Answer> {
    kernels::drive(job, |n, body| {
        let first: Mutex<Option<BlockError>> = Mutex::new(None);
        let mut blocks = vec![(); n];
        for_each_chunk_indexed_min_leaf(plan, &mut blocks, 1, |start, chunk| {
            for b in start..start + chunk.len() {
                if let Err(e) = body(b) {
                    // Nothing can panic while this lock is held, so a
                    // poisoned one still holds a whole value.
                    let mut held = match first.lock() {
                        Ok(guard) => guard,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    if held.is_none() {
                        *held = Some(e);
                    }
                }
            }
        });
        let held = match first.into_inner() {
            Ok(held) => held,
            Err(poisoned) => poisoned.into_inner(),
        };
        match held {
            Some(e) => Err(e),
            None => Ok(()),
        }
    })
    .map_err(refusal_err)
}

/// Refuse unless every column has the first one's length.
fn same_length(columns: &[(&str, usize)]) -> PsResult<()> {
    let Some(&(first, n)) = columns.first() else {
        return Ok(());
    };
    for &(name, len) in &columns[1..] {
        if len != n {
            return Err(arg_err(format!(
                "{first} has {n} element(s) and {name} has {len}; every column needs the same \
                 length"
            ))
            .terminating());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Arrays and numbers
// ---------------------------------------------------------------------

/// An element-wise operation over one array.
#[psenum(name = "Flynnel.MapOp")]
#[derive(Clone, Copy, Debug, Default)]
pub enum MapOp {
    /// `x * x`.
    #[default]
    Square,
    /// The magnitude.
    Abs,
    /// `-x`.
    Negate,
    /// `1 / x`, which answers an infinity at zero.
    Reciprocal,
    /// The square root, which answers NaN below zero.
    Sqrt,
    /// The natural logarithm, NaN below zero and negative infinity at
    /// zero.
    Log,
    /// The base-2 logarithm, on the same terms as Log.
    Log2,
    /// `e` to the power x.
    Exp,
    /// To the nearest integer, halves away from zero.
    Round,
    /// To the largest integer at or below x.
    Floor,
    /// To the smallest integer at or above x.
    Ceiling,
    /// Held within Min and Max, both of which are required.
    Clamp,
    /// Multiplied by Factor.
    Scale,
    /// Added to Addend.
    Offset,
}

impl MapOp {
    /// The crate's operation of the same name.
    pub(crate) fn to_kernel(self) -> kernels::MapOp {
        match self {
            Self::Square => kernels::MapOp::Square,
            Self::Abs => kernels::MapOp::Abs,
            Self::Negate => kernels::MapOp::Negate,
            Self::Reciprocal => kernels::MapOp::Reciprocal,
            Self::Sqrt => kernels::MapOp::Sqrt,
            Self::Log => kernels::MapOp::Log,
            Self::Log2 => kernels::MapOp::Log2,
            Self::Exp => kernels::MapOp::Exp,
            Self::Round => kernels::MapOp::Round,
            Self::Floor => kernels::MapOp::Floor,
            Self::Ceiling => kernels::MapOp::Ceiling,
            Self::Clamp => kernels::MapOp::Clamp,
            Self::Scale => kernels::MapOp::Scale,
            Self::Offset => kernels::MapOp::Offset,
        }
    }
}

/// A pairwise operation over two arrays of the same length.
#[psenum(name = "Flynnel.ZipOp")]
#[derive(Clone, Copy, Debug, Default)]
pub enum ZipOp {
    /// `left + right`.
    #[default]
    Add,
    /// `left - right`.
    Subtract,
    /// `left * right`.
    Multiply,
    /// `left / right`, which answers an infinity where right is zero.
    Divide,
    /// The smaller of the two.
    Min,
    /// The larger of the two.
    Max,
}

impl ZipOp {
    /// The crate's operation of the same name.
    fn to_kernel(self) -> kernels::ZipOp {
        match self {
            Self::Add => kernels::ZipOp::Add,
            Self::Subtract => kernels::ZipOp::Subtract,
            Self::Multiply => kernels::ZipOp::Multiply,
            Self::Divide => kernels::ZipOp::Divide,
            Self::Min => kernels::ZipOp::Min,
            Self::Max => kernels::ZipOp::Max,
        }
    }
}

/// A reduction over one array to a single number.
#[psenum(name = "Flynnel.ReduceOp")]
#[derive(Clone, Copy, Debug, Default)]
pub enum ReduceOp {
    /// The total.
    #[default]
    Sum,
    /// The smallest element.
    Min,
    /// The largest element.
    Max,
    /// The arithmetic mean.
    Mean,
    /// The population variance, by the pairwise-combining form rather
    /// than a sum of squares, so a large mean does not cancel the
    /// spread away.
    Variance,
    /// The product of every element.
    Product,
    /// How many elements sit within Min and Max inclusive.
    CountMatching,
}

impl ReduceOp {
    /// The crate's reduction of the same name.
    fn to_kernel(self) -> kernels::ReduceOp {
        match self {
            Self::Sum => kernels::ReduceOp::Sum,
            Self::Min => kernels::ReduceOp::Min,
            Self::Max => kernels::ReduceOp::Max,
            Self::Mean => kernels::ReduceOp::Mean,
            Self::Variance => kernels::ReduceOp::Variance,
            Self::Product => kernels::ReduceOp::Product,
            Self::CountMatching => kernels::ReduceOp::CountMatching,
        }
    }
}

/// What a reduction answered.
#[psclass(name = "Flynnel.Reduction")]
#[derive(Clone, Default)]
pub struct Reduction {
    /// The operation that produced it.
    pub operation: ReduceOp,
    /// How many elements it read.
    pub count: u64,
    /// The answer. Null for a reduction with no defined value over an
    /// empty input, which is every one of them except Sum, Product and
    /// CountMatching.
    pub value: Option<f64>,
}

/// One bin of a histogram.
#[psclass(name = "Flynnel.HistogramBin")]
#[derive(Clone, Default)]
pub struct HistogramBin {
    /// The bin's position, counting from zero.
    pub index: u32,
    /// The lowest value the bin takes.
    pub low: f64,
    /// The lowest value the next bin takes. The last bin takes this
    /// value itself, so the whole range is covered.
    pub high: f64,
    /// How many elements landed here.
    pub count: u64,
}

impl From<kernels::HistogramBin> for HistogramBin {
    fn from(b: kernels::HistogramBin) -> Self {
        Self {
            index: b.index,
            low: b.low,
            high: b.high,
            count: b.count,
        }
    }
}

#[psmethods]
impl HistogramBin {
    /// One HistogramBin per position of the columns, which must all have
    /// the same length: a whole histogram handed over in one call and
    /// built by this module.
    pub fn from_columns(
        index: Vec<u32>,
        low: Vec<f64>,
        high: Vec<f64>,
        count: Vec<u64>,
    ) -> PsResult<Vec<HistogramBin>> {
        same_length(&[
            ("Index", index.len()),
            ("Low", low.len()),
            ("High", high.len()),
            ("Count", count.len()),
        ])?;
        Ok(index
            .into_iter()
            .zip(low)
            .zip(high)
            .zip(count)
            .map(|(((index, low), high), count)| HistogramBin {
                index,
                low,
                high,
                count,
            })
            .collect())
    }
}

/// A whole histogram as one object, with the counts as an array.
///
/// The bins are recoverable from the range: bin `i` runs from
/// `Low + Width * i` to `Low + Width * (i + 1)`.
#[psclass(name = "Flynnel.Histogram")]
#[derive(Clone, Default)]
pub struct Histogram {
    /// The lowest value the first bin takes.
    pub low: f64,
    /// The highest value the last bin takes. The last bin takes it
    /// itself, so the whole range is covered.
    pub high: f64,
    /// How wide one bin is. Zero when every element has one value,
    /// which puts all of them in the first bin.
    pub width: f64,
    /// How many elements landed in each bin, in bin order. As long as
    /// Bins asked for.
    ///
    /// Plural because every PowerShell object answers an intrinsic
    /// Count of one, so a property named Count that failed to resolve
    /// would read as a histogram of a single bin rather than as an
    /// error.
    pub counts: Vec<u64>,
}

/// Applies one operation to every element of an array on Flynnel's
/// workers.
///
/// The whole array crosses once and the whole answer crosses back
/// once.
///
/// # Examples
///
/// `Invoke-FlynnelMap -InputObject $x -Operation Sqrt`
///
/// `Invoke-FlynnelMap -InputObject $x -Operation Clamp -Min 0 -Max 1`
///
/// `Invoke-FlynnelMap -InputObject $x -Operation Scale -Factor 2.5`
#[cmdlet(
    verb = "Invoke",
    noun = "FlynnelMap",
    alias = "Invoke-FlyMap",
    output = ["System.Double[]"]
)]
#[derive(Default)]
pub struct InvokeFlynnelMap {
    /// The numbers to transform.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub input_object: Vec<f64>,
    /// Which transform to apply.
    #[param(mandatory, position = 1)]
    pub operation: MapOp,
    /// The lower bound, for Clamp.
    #[param]
    pub min: Option<f64>,
    /// The upper bound, for Clamp.
    #[param]
    pub max: Option<f64>,
    /// The multiplier, for Scale.
    #[param]
    pub factor: Option<f64>,
    /// The addend, for Offset.
    #[param]
    pub addend: Option<f64>,
    /// The plan to run under. Without one the kernel sizes a plan to
    /// the input.
    #[param]
    pub plan: Option<Plan>,
}

/// The operands a map operation needs, read once rather than per
/// element, so a closure carries plain numbers and a missing operand is
/// refused before any work is dispatched. The crate's own type, so the
/// hybrid and racing families and the kernels read operands alike.
pub(crate) type MapOperands = kernels::MapOperands;

/// One declared operation's per-element body.
///
/// `Send` as well as `Sync` because the hybrid family moves the body
/// to a backend thread; the kernels family only shares it across
/// workers.
pub(crate) type ElementBody = Box<dyn Fn(&mut f64) + Send + Sync>;

/// The per-element body of one declared operation, with its operands
/// already read. Every family that runs a declared map goes through
/// this, and it applies the crate's own element operation, so no two of
/// them can answer differently for the same operation.
pub(crate) fn map_each(op: MapOp, operands: MapOperands) -> PsResult<ElementBody> {
    let element = kernels::Element::new(op.to_kernel(), operands).map_err(refusal_err)?;
    Ok(Box::new(move |x: &mut f64| *x = element.apply(*x)))
}

impl Cmdlet for InvokeFlynnelMap {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let mut items = std::mem::take(&mut self.input_object);
        let n = items.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Invoke-FlynnelMap")?;
        let operands = MapOperands {
            min: self.min,
            max: self.max,
            factor: self.factor,
            addend: self.addend,
        };
        let job =
            kernels::map(&mut items, self.operation.to_kernel(), operands).map_err(refusal_err)?;
        run_on_pool(&plan, job)?;
        // PsArray and not a PsMemory view. The view hands the buffer
        // over without a managed copy and measured the same: 5.19 ms
        // against 5.18 over 200,000 elements, inside a control that
        // drifted 4.55 per cent. So the return's remaining 25 ns an
        // element is not the copy, and the view only adds a Rust-side
        // one. Do not re-try this without a different reason.
        //
        // The second reason not to: a view hands the managed side a
        // free callback, which is a reverse call into this library
        // and attaches whichever thread runs it. Nothing here
        // registers one, so no worker is attached by a kernel's
        // answer any more than by its body.
        ps.write(PsArray(items))
    }
}

/// Applies one operation to every element of an array in place, on
/// Flynnel's workers, and writes nothing.
///
/// This is the same work Invoke-FlynnelMap does with the return taken
/// out. Measured at 200,000 elements on pc2: the input crossing is
/// 0.17 ms and the input plus the return is 4.97 ms, so for a kernel
/// whose arithmetic is a multiply the return is nearly all of the
/// cost. A loop that transforms one buffer repeatedly pays it once
/// per pass and does not need to.
///
/// The array is changed in place. It must be a typed double array,
/// because an
/// in-place update writes through a pin of the caller's own buffer
/// and there is nothing to pin in a boxed collection. Cast once:
///
///     $x = [double[]]$x
///     Update-FlynnelArray -InputObject $x -Operation Square
///
/// # Examples
///
/// `Update-FlynnelArray -InputObject $x -Operation Sqrt`
///
/// `Update-FlynnelArray -InputObject $x -Operation Scale -Factor 2.5`
#[cmdlet(
    verb = "Update",
    noun = "FlynnelArray",
    alias = "Update-FlyArray"
)]
#[derive(Default)]
pub struct UpdateFlynnelArray {
    /// The typed double array to transform in place.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub input_object: PsObject,
    /// Which transform to apply.
    #[param(mandatory, position = 1)]
    pub operation: MapOp,
    /// The lower bound, for Clamp.
    #[param]
    pub min: Option<f64>,
    /// The upper bound, for Clamp.
    #[param]
    pub max: Option<f64>,
    /// The multiplier, for Scale.
    #[param]
    pub factor: Option<f64>,
    /// The addend, for Offset.
    #[param]
    pub addend: Option<f64>,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for UpdateFlynnelArray {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        // Pinned rather than copied. Silently copying instead would
        // answer correctly and cost exactly what this cmdlet exists to
        // avoid.
        //
        // The pin checks the element type as well as its width, so an
        // array of another eight-byte value is refused here rather than
        // read as doubles and written back over the caller's storage.
        let mut pinned = self.input_object.pin::<f64>().map_err(|e| {
            arg_err(format!(
                "InputObject must be a typed double array to be updated in place: {e}. \
                 Cast it once with [double[]]$x, or use Invoke-FlynnelMap, which takes any \
                 collection and answers a new array."
            ))
            .terminating()
        })?;
        let n = pinned.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Update-FlynnelArray")?;
        let operands = MapOperands {
            min: self.min,
            max: self.max,
            factor: self.factor,
            addend: self.addend,
        };
        let job =
            kernels::map(&mut pinned, self.operation.to_kernel(), operands).map_err(refusal_err)?;
        run_on_pool(&plan, job)
    }
}

/// Applies one pairwise operation across two arrays on Flynnel's
/// workers, in one crossing each way.
///
/// # Examples
///
/// `Invoke-FlynnelZip -Left $a -Right $b -Operation Multiply`
#[cmdlet(
    verb = "Invoke",
    noun = "FlynnelZip",
    alias = "Invoke-FlyZip",
    output = ["System.Double[]"]
)]
#[derive(Default)]
pub struct InvokeFlynnelZip {
    /// The left operand, and the shape of the answer.
    #[param(mandatory, position = 0)]
    pub left: Vec<f64>,
    /// The right operand, which must be the same length as Left.
    #[param(mandatory, position = 1)]
    pub right: Vec<f64>,
    /// Which pairwise operation to apply.
    #[param(mandatory, position = 2)]
    pub operation: ZipOp,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for InvokeFlynnelZip {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let mut lhs = std::mem::take(&mut self.left);
        let rhs = std::mem::take(&mut self.right);
        let n = lhs.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        let job = kernels::zip(&mut lhs, &rhs, self.operation.to_kernel()).map_err(refusal_err)?;
        say_plan(ps, &plan, n, "Invoke-FlynnelZip")?;
        run_on_pool(&plan, job)?;
        ps.write(PsArray(lhs))
    }
}

/// Reduces an array to one number on Flynnel's workers.
///
/// # Examples
///
/// `Measure-FlynnelReduce -InputObject $x -Operation Sum`
///
/// `Measure-FlynnelReduce -InputObject $x -Operation CountMatching -Min 0 -Max 1`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelReduce",
    alias = "Measure-FlyReduce",
    output = ["Flynnel.Reduction"]
)]
#[derive(Default)]
pub struct MeasureFlynnelReduce {
    /// The numbers to reduce.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub input_object: Vec<f64>,
    /// Which reduction to run.
    #[param(mandatory, position = 1)]
    pub operation: ReduceOp,
    /// The lower bound, for CountMatching.
    #[param]
    pub min: Option<f64>,
    /// The upper bound, for CountMatching.
    #[param]
    pub max: Option<f64>,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for MeasureFlynnelReduce {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let items = std::mem::take(&mut self.input_object);
        let n = items.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Measure-FlynnelReduce")?;
        let job = kernels::reduce(&items, self.operation.to_kernel(), self.min, self.max)
            .map_err(refusal_err)?;
        let answer = run_on_pool(&plan, job)?;
        ps.write(Reduction {
            operation: self.operation,
            count: answer.count,
            value: answer.value,
        })
    }
}

/// The inclusive running total of an array, computed as a two-phase
/// parallel scan on Flynnel's workers.
///
/// Phase one sums each block, phase two scans each block from its
/// block's offset. The offsets between them are a scan over one value
/// per block.
///
/// # Examples
///
/// `Get-FlynnelPrefixSum -InputObject $x`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelPrefixSum",
    alias = "Get-FlyPrefixSum",
    output = ["System.Double[]"]
)]
#[derive(Default)]
pub struct GetFlynnelPrefixSum {
    /// The numbers to scan.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub input_object: Vec<f64>,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for GetFlynnelPrefixSum {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let mut items = std::mem::take(&mut self.input_object);
        let n = items.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Get-FlynnelPrefixSum")?;
        run_on_pool(&plan, kernels::prefix_sum(&mut items))?;
        ps.write(PsArray(items))
    }
}

/// Bins an array into equal-width buckets on Flynnel's workers.
///
/// Each block fills its own bin vector and the vectors are added, so
/// no two workers touch the same counter.
///
/// Without Min and Max the range comes from the data, in one parallel
/// pass before the binning pass.
///
/// Answers one record per bin, which suits filtering and formatting,
/// or with AsArray one record for the whole histogram, which suits a
/// bin count high enough that a record each would cost more than the
/// binning.
///
/// # Examples
///
/// `Get-FlynnelHistogram -InputObject $x -Bins 16`
///
/// `Get-FlynnelHistogram -InputObject $x -Bins 10 -Min 0 -Max 1`
///
/// `Get-FlynnelHistogram -InputObject $x -Bins 1000000 -AsArray`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelHistogram",
    alias = "Get-FlyHistogram",
    output = ["Flynnel.HistogramBin", "Flynnel.Histogram"]
)]
#[derive(Default)]
pub struct GetFlynnelHistogram {
    /// The numbers to bin.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub input_object: Vec<f64>,
    /// How many bins to cut the range into.
    #[param(mandatory, position = 1)]
    pub bins: u32,
    /// The bottom of the range. Absent, the smallest element.
    #[param]
    pub min: Option<f64>,
    /// The top of the range. Absent, the largest element.
    #[param]
    pub max: Option<f64>,
    /// Answer one Flynnel.Histogram, whose Count is the whole array of
    /// counts and which carries the range beside it, instead of one
    /// Flynnel.HistogramBin record per bin.
    ///
    /// One record crosses the boundary once. A bin per record crosses
    /// it once per bin, which is what Bins asks for and nothing else
    /// bounds.
    #[param]
    pub as_array: bool,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for GetFlynnelHistogram {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let items = std::mem::take(&mut self.input_object);
        let n = items.len();
        let job = kernels::histogram(&items, self.bins, self.min, self.max).map_err(refusal_err)?;
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Get-FlynnelHistogram")?;
        let Some(histogram) = run_on_pool(&plan, job)? else {
            return Ok(());
        };
        if self.as_array {
            return ps.write(Histogram {
                low: histogram.low,
                high: histogram.high,
                width: histogram.width,
                counts: histogram.counts,
            });
        }
        for bin in histogram.bins() {
            ps.write(HistogramBin::from(bin))?;
        }
        Ok(())
    }
}

/// The dot product of two arrays, summed per block on Flynnel's
/// workers and combined once.
///
/// # Examples
///
/// `Get-FlynnelDotProduct -Left $a -Right $b`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelDotProduct",
    alias = "Get-FlyDotProduct",
    output = ["System.Double"]
)]
#[derive(Default)]
pub struct GetFlynnelDotProduct {
    /// The left operand.
    #[param(mandatory, position = 0)]
    pub left: Vec<f64>,
    /// The right operand, the same length as Left.
    #[param(mandatory, position = 1)]
    pub right: Vec<f64>,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for GetFlynnelDotProduct {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let lhs = std::mem::take(&mut self.left);
        let rhs = std::mem::take(&mut self.right);
        let job = kernels::dot_product(&lhs, &rhs).map_err(refusal_err)?;
        let n = lhs.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Get-FlynnelDotProduct")?;
        ps.write(run_on_pool(&plan, job)?)
    }
}

/// Sorts an array on Flynnel's workers: each run is sorted in
/// parallel, then the runs are merged in parallel rounds.
///
/// NaN sorts above every number, which is the total order
/// `f64::total_cmp` gives and the only one a comparison sort can use.
///
/// Sort is not one of PowerShell's approved verbs, so the cmdlet takes
/// Invoke, as Invoke-FlynnelMap and Invoke-FlynnelZip do, and answers
/// to Sort-FlynnelArray and Sort-FlyArray as aliases as well.
///
/// # Examples
///
/// `Invoke-FlynnelSort -InputObject $x`
///
/// `Invoke-FlynnelSort -InputObject $x -Descending`
#[cmdlet(
    verb = "Invoke",
    noun = "FlynnelSort",
    alias = ["Invoke-FlySort", "Sort-FlynnelArray", "Sort-FlyArray"],
    output = ["System.Double[]"]
)]
#[derive(Default)]
pub struct InvokeFlynnelSort {
    /// The numbers to sort.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub input_object: Vec<f64>,
    /// Largest first.
    #[param]
    pub descending: bool,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for InvokeFlynnelSort {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let items = std::mem::take(&mut self.input_object);
        let n = items.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Invoke-FlynnelSort")?;
        let sorted = run_on_pool(&plan, kernels::sort(&items, self.descending))?;
        ps.write(PsArray(sorted))
    }
}

// ---------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------

/// A file's BLAKE3 root.
#[psclass(name = "Flynnel.FileHash")]
#[derive(Clone, Default)]
pub struct FileHash {
    /// The file this describes.
    pub path: String,
    /// The BLAKE3 root, lower-case hex.
    pub hash: String,
    /// How many bytes were read.
    pub bytes: u64,
}

impl From<kernels::FileHash> for FileHash {
    fn from(h: kernels::FileHash) -> Self {
        Self {
            path: h.path,
            hash: h.hash,
            bytes: h.bytes,
        }
    }
}

#[psmethods]
impl FileHash {
    /// One FileHash per position of the columns, which must all have the
    /// same length: a whole answer handed over in one call and built by
    /// this module.
    pub fn from_columns(
        path: Vec<String>,
        hash: Vec<String>,
        bytes: Vec<u64>,
    ) -> PsResult<Vec<FileHash>> {
        same_length(&[
            ("Path", path.len()),
            ("Hash", hash.len()),
            ("Bytes", bytes.len()),
        ])?;
        Ok(path
            .into_iter()
            .zip(hash)
            .zip(bytes)
            .map(|((path, hash), bytes)| FileHash { path, hash, bytes })
            .collect())
    }
}

/// One file checked against a manifest.
#[psclass(name = "Flynnel.HashCheck")]
#[derive(Clone, Default)]
pub struct HashCheck {
    /// The file this describes.
    pub path: String,
    /// What the manifest said it should be, lower-case hex.
    pub expected: String,
    /// What it actually hashed to. Null when the file could not be
    /// read, which is why a row exists at all in that case.
    pub actual: Option<String>,
    /// Whether the two agree. False for a file that could not be read,
    /// never null, so a script filtering on this cannot silently pass
    /// an unreadable file.
    pub is_match: bool,
}

impl From<kernels::HashCheck> for HashCheck {
    fn from(c: kernels::HashCheck) -> Self {
        Self {
            path: c.path,
            expected: c.expected,
            actual: c.actual,
            is_match: c.is_match,
        }
    }
}

#[psmethods]
impl HashCheck {
    /// One HashCheck per position of the columns, which must all have the
    /// same length. An empty Actual stands for a file that could not be
    /// read and becomes null: a real root is never empty.
    pub fn from_columns(
        path: Vec<String>,
        expected: Vec<String>,
        actual: Vec<String>,
        is_match: Vec<bool>,
    ) -> PsResult<Vec<HashCheck>> {
        same_length(&[
            ("Path", path.len()),
            ("Expected", expected.len()),
            ("Actual", actual.len()),
            ("IsMatch", is_match.len()),
        ])?;
        Ok(path
            .into_iter()
            .zip(expected)
            .zip(actual)
            .zip(is_match)
            .map(|(((path, expected), actual), is_match)| HashCheck {
                path,
                expected,
                actual: (!actual.is_empty()).then_some(actual),
                is_match,
            })
            .collect())
    }
}

/// A line of a file that matched.
#[psclass(name = "Flynnel.FileMatch")]
#[derive(Clone, Default)]
pub struct FileMatch {
    /// The file the line came from.
    pub path: String,
    /// The line's position, counting from one.
    pub line_number: u64,
    /// The line, without its terminator.
    pub line: String,
}

impl From<kernels::FileMatch> for FileMatch {
    fn from(m: kernels::FileMatch) -> Self {
        Self {
            path: m.path,
            line_number: m.line_number,
            line: m.line,
        }
    }
}

#[psmethods]
impl FileMatch {
    /// One FileMatch per position of the columns, which must all have the
    /// same length: a whole answer handed over in one call and built by
    /// this module.
    pub fn from_columns(
        path: Vec<String>,
        line_number: Vec<u64>,
        line: Vec<String>,
    ) -> PsResult<Vec<FileMatch>> {
        same_length(&[
            ("Path", path.len()),
            ("LineNumber", line_number.len()),
            ("Line", line.len()),
        ])?;
        Ok(path
            .into_iter()
            .zip(line_number)
            .zip(line)
            .map(|((path, line_number), line)| FileMatch {
                path,
                line_number,
                line,
            })
            .collect())
    }
}

/// What one file measured.
#[psclass(name = "Flynnel.FileMeasure")]
#[derive(Clone, Default)]
pub struct FileMeasure {
    /// The file this describes.
    pub path: String,
    /// How many lines it holds. A final line with no terminator counts.
    pub lines: u64,
    /// How many bytes it holds.
    pub bytes: u64,
}

impl From<kernels::FileMeasure> for FileMeasure {
    fn from(m: kernels::FileMeasure) -> Self {
        Self {
            path: m.path,
            lines: m.lines,
            bytes: m.bytes,
        }
    }
}

#[psmethods]
impl FileMeasure {
    /// One FileMeasure per position of the columns, which must all have
    /// the same length: a whole answer handed over in one call and built
    /// by this module.
    pub fn from_columns(
        path: Vec<String>,
        lines: Vec<u64>,
        bytes: Vec<u64>,
    ) -> PsResult<Vec<FileMeasure>> {
        same_length(&[
            ("Path", path.len()),
            ("Lines", lines.len()),
            ("Bytes", bytes.len()),
        ])?;
        Ok(path
            .into_iter()
            .zip(lines)
            .zip(bytes)
            .map(|((path, lines), bytes)| FileMeasure { path, lines, bytes })
            .collect())
    }
}

/// Write each row that read and an error record for each that did not,
/// then the tally of the refusals.
fn write_rows<R, W: IntoPs>(
    ps: &Pipeline<'_>,
    rows: Vec<Result<R, Refusal>>,
    mut each: impl FnMut(R) -> Vec<W>,
) -> PsResult<()> {
    let total = rows.len();
    let mut refused = 0usize;
    for row in rows {
        match row {
            Ok(r) => {
                for w in each(r) {
                    ps.write(w)?;
                }
            }
            Err(refusal) => {
                refused += 1;
                ps.write_error(&refusal_err(refusal))?;
            }
        }
    }
    say_refusals(ps, refused, total)
}

/// Hashes each file on the process-wide IO pool instead of the arena,
/// returning `None` when no such pool exists.
///
/// `None` rather than a silent fall-through to the arena: the caller
/// has asked for a route and is entitled to know it was not taken.
/// The crate's own submit helpers run the task inline when the pool is
/// absent, which is correct for them and would make this cmdlet report
/// a route it did not use.
///
/// The paths are cloned because `IoPool::submit` takes a `'static`
/// closure. That is one allocation per file, on top of the read, and
/// it is the cost this route has to earn back. Each file streams
/// through the crate's one-file hash, so this route answers the root
/// the kernel does.
fn hash_files_on_io_pool(paths: &[String]) -> Option<Vec<Result<kernels::FileHash, Refusal>>> {
    let pool = flynnel::sched::io_pool::global_io_pool()?;
    let (tx, rx) = std::sync::mpsc::channel();
    for (i, path) in paths.iter().enumerate() {
        let tx = tx.clone();
        let path = path.clone();
        pool.submit(move || {
            let row = kernels::file_hash_streamed(&path);
            if tx.send((i, row)).is_err() {
                eprintln!(
                    "flynnel: the result channel for {path} closed before its hash was reported"
                );
            }
        });
    }
    drop(tx);

    let mut out: Vec<Option<Result<kernels::FileHash, Refusal>>> =
        (0..paths.len()).map(|_unfilled| None).collect();
    for (i, row) in rx {
        out[i] = Some(row);
    }
    // Every slot is filled unless a task panicked, which the pool does
    // not catch. Reported as a refusal for that path rather than
    // silently standing in a value.
    Some(
        out.into_iter()
            .enumerate()
            .map(|(i, slot)| match slot {
                Some(row) => row,
                None => Err(Refusal {
                    id: RefusalId::Unreadable,
                    message: format!(
                        "{} could not be read: the IO pool task for it did not return a result",
                        paths[i]
                    ),
                }),
            })
            .collect(),
    )
}

/// Hashes files with BLAKE3 on Flynnel's workers, one task per file.
///
/// BLAKE3 comes from the crate's own verify-chain hasher, so the root
/// is the same one Flynnel's attestation path produces and the module
/// takes no hashing dependency of its own.
///
/// One file is split across workers as well. BLAKE3 is a tree, so a
/// span hashed at its true input offset yields the chaining value the
/// specification defines for it, and merging those up the tree gives
/// the same root a single pass would. The answer is standard BLAKE3,
/// not a scheme of this module's, and the suite checks it against a
/// published vector and against the one-thread path.
///
/// A file above a gigabyte is streamed on one thread instead, because
/// the split needs the bytes in memory. Both paths give the same root,
/// so the bound is on memory and speed and never on the answer.
///
/// # Examples
///
/// `Measure-FlynnelFileHash -Path (Get-ChildItem *.dll).FullName`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelFileHash",
    alias = "Measure-FlyFileHash",
    output = ["Flynnel.FileHash"]
)]
#[derive(Default)]
pub struct MeasureFlynnelFileHash {
    /// The files to hash. All of them cross in one call.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub path: Vec<String>,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
    /// Read the files on the process-wide IO pool rather than on the
    /// scheduler's own workers, so a blocking read does not hold a
    /// worker that compute could be using.
    ///
    /// Warns and uses the workers when no such pool exists, which is
    /// the default: one is created only when FLYNNEL_SCHED_SMT_AS_IO
    /// is set.
    #[param]
    pub use_io_pool: bool,
}

impl Cmdlet for MeasureFlynnelFileHash {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let paths = std::mem::take(&mut self.path);
        let n = paths.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Measure-FlynnelFileHash")?;
        // Asked for and available, asked for and absent, or not asked
        // for. The middle one warns rather than falling through
        // quietly, because a pool that does not exist and a pool that
        // was never wanted produce identical output otherwise.
        let rows = match (self.use_io_pool, n) {
            (_, 0) => Vec::new(),
            (true, _) => match hash_files_on_io_pool(&paths) {
                Some(rows) => rows,
                None => {
                    pwrs::warning!(
                        ps,
                        "this process has no IO pool, so the reads ran on the scheduler's \
                         workers. Set FLYNNEL_SCHED_SMT_AS_IO before the first dispatch to \
                         create one"
                    )?;
                    run_on_pool(&plan, kernels::file_hash(&paths))?
                }
            },
            (false, _) => run_on_pool(&plan, kernels::file_hash(&paths))?,
        };
        write_rows(ps, rows, |h| vec![FileHash::from(h)])
    }
}

/// Checks files against a manifest of expected BLAKE3 roots, hashing
/// on Flynnel's workers.
///
/// A file that could not be read is a row with a null Actual and a
/// false IsMatch, and an error record beside it. It is never an absent
/// row: a manifest check whose failure mode is silence cannot be used
/// to decide anything.
///
/// # Examples
///
/// `Test-FlynnelFileHash -Path $files -Manifest $hashes`
#[cmdlet(
    verb = "Test",
    noun = "FlynnelFileHash",
    alias = "Test-FlyFileHash",
    output = ["Flynnel.HashCheck"]
)]
#[derive(Default)]
pub struct TestFlynnelFileHash {
    /// The files to check.
    #[param(mandatory, position = 0)]
    pub path: Vec<String>,
    /// The expected roots as hex, in the same order as Path and the
    /// same length. Case does not matter.
    #[param(mandatory, position = 1)]
    pub manifest: Vec<String>,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for TestFlynnelFileHash {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let paths = std::mem::take(&mut self.path);
        let manifest = std::mem::take(&mut self.manifest);
        let job = kernels::file_hash_check(&paths, &manifest).map_err(refusal_err)?;
        let n = paths.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Test-FlynnelFileHash")?;
        let mut refused = 0usize;
        for checked in run_on_pool(&plan, job)? {
            if let Some(refusal) = checked.refusal {
                refused += 1;
                ps.write_error(&refusal_err(refusal))?;
            }
            ps.write(HashCheck::from(checked.check))?;
        }
        say_refusals(ps, refused, n)
    }
}

/// Searches files for a literal string on Flynnel's workers, one task
/// per file.
///
/// The pattern is a literal, not a regular expression: a regex engine
/// is a dependency this module does not take, and a literal search is
/// what the parallel shape is for.
///
/// Matching is ordinal, over bytes, and `-IgnoreCase` folds the ASCII
/// range and nothing else. The PowerShell ways to get the same answer
/// do not match that way: `-like` and `Select-String` fold case
/// through .NET's culture-aware casing, which comes from ICU under
/// PowerShell 7 and from the older NLS tables under Windows
/// PowerShell 5.1, and falls back to NLS where ICU cannot be loaded.
/// Casing is one of the areas those two libraries are documented to
/// differ in, so the same script can fold a pair outside ASCII in one
/// shell and not in the other. This cmdlet gives the same answer on
/// every host, and differs from both of them on such a pair.
///
/// # Examples
///
/// `Search-FlynnelFile -Pattern 'panic' -Path $files`
#[cmdlet(
    verb = "Search",
    noun = "FlynnelFile",
    alias = "Search-FlyFile",
    output = ["Flynnel.FileMatch"]
)]
#[derive(Default)]
pub struct SearchFlynnelFile {
    /// The literal string to look for.
    #[param(mandatory, position = 0)]
    pub pattern: String,
    /// The files to search.
    #[param(mandatory, position = 1, value_from_pipeline)]
    pub path: Vec<String>,
    /// Match without regard to ASCII case.
    #[param]
    pub ignore_case: bool,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for SearchFlynnelFile {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let paths = std::mem::take(&mut self.path);
        let pattern = std::mem::take(&mut self.pattern);
        let job = kernels::search_file(&pattern, &paths, self.ignore_case).map_err(refusal_err)?;
        let n = paths.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Search-FlynnelFile")?;
        let rows = run_on_pool(&plan, job)?;
        write_rows(ps, rows, |matches| {
            matches.into_iter().map(FileMatch::from).collect()
        })
    }
}

/// Counts the lines in files on Flynnel's workers, one task per file.
///
/// A final line with no terminator counts, so a one-line file with no
/// newline reports one rather than zero.
///
/// # Examples
///
/// `Measure-FlynnelFileLine -Path $files`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelFileLine",
    alias = "Measure-FlyFileLine",
    output = ["Flynnel.FileMeasure"]
)]
#[derive(Default)]
pub struct MeasureFlynnelFileLine {
    /// The files to measure.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub path: Vec<String>,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for MeasureFlynnelFileLine {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let paths = std::mem::take(&mut self.path);
        let n = paths.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Measure-FlynnelFileLine")?;
        let rows = run_on_pool(&plan, kernels::file_line(&paths))?;
        write_rows(ps, rows, |m| vec![FileMeasure::from(m)])
    }
}

/// Reads the byte length of files on Flynnel's workers, one task per
/// file.
///
/// This is a metadata read rather than a content read, so it answers
/// for a file too large to hold in memory. The Lines column is zero
/// because no line was counted; Measure-FlynnelFileLine is the cmdlet
/// that counts them.
///
/// # Examples
///
/// `Measure-FlynnelFileByte -Path $files`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelFileByte",
    alias = "Measure-FlyFileByte",
    output = ["Flynnel.FileMeasure"]
)]
#[derive(Default)]
pub struct MeasureFlynnelFileByte {
    /// The files to measure.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub path: Vec<String>,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for MeasureFlynnelFileByte {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let paths = std::mem::take(&mut self.path);
        let n = paths.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Measure-FlynnelFileByte")?;
        let rows = run_on_pool(&plan, kernels::file_byte(&paths))?;
        write_rows(ps, rows, |m| vec![FileMeasure::from(m)])
    }
}

// ---------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------

/// A place in a string where a pattern was found.
#[psclass(name = "Flynnel.TextMatch")]
#[derive(Clone, Default)]
pub struct TextMatch {
    /// The byte offset of the match, counting from zero.
    pub index: u64,
    /// The line the match is on, counting from one.
    pub line_number: u64,
    /// The whole line the match is on, without its terminator.
    pub line: String,
}

impl From<kernels::TextMatch> for TextMatch {
    fn from(m: kernels::TextMatch) -> Self {
        Self {
            index: m.index,
            line_number: m.line_number,
            line: m.line,
        }
    }
}

#[psmethods]
impl TextMatch {
    /// One TextMatch per position of the columns, which must all have the
    /// same length: a whole answer handed over in one call and built by
    /// this module.
    pub fn from_columns(
        index: Vec<u64>,
        line_number: Vec<u64>,
        line: Vec<String>,
    ) -> PsResult<Vec<TextMatch>> {
        same_length(&[
            ("Index", index.len()),
            ("LineNumber", line_number.len()),
            ("Line", line.len()),
        ])?;
        Ok(index
            .into_iter()
            .zip(line_number)
            .zip(line)
            .map(|((index, line_number), line)| TextMatch {
                index,
                line_number,
                line,
            })
            .collect())
    }
}

/// What a text measurement answered.
#[psclass(name = "Flynnel.TextMeasure")]
#[derive(Clone, Default)]
pub struct TextMeasure {
    /// How many bytes the text holds.
    pub bytes: u64,
    /// How many lines it holds.
    pub lines: u64,
    /// How many whitespace-separated words it holds.
    pub words: u64,
    /// How many times the pattern occurs, counting non-overlapping
    /// occurrences. Null when no pattern was given.
    pub matches: Option<u64>,
}

/// Finds every occurrence of a literal in one large string, searched
/// in parallel on Flynnel's workers.
///
/// The whole string crosses once. Blocks overlap by the pattern's
/// length less one so a match on a boundary is still found, and each
/// match is owned by the block its first byte falls in so none is
/// reported twice.
///
/// # Examples
///
/// `Search-FlynnelText -Text $log -Pattern 'ERROR'`
#[cmdlet(
    verb = "Search",
    noun = "FlynnelText",
    alias = "Search-FlyText",
    output = ["Flynnel.TextMatch"]
)]
#[derive(Default)]
pub struct SearchFlynnelText {
    /// The text to search.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub text: String,
    /// The literal string to look for.
    #[param(mandatory, position = 1)]
    pub pattern: String,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for SearchFlynnelText {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let text = std::mem::take(&mut self.text);
        let pattern = std::mem::take(&mut self.pattern);
        let job = kernels::search_text(&text, &pattern).map_err(refusal_err)?;
        let plan = kernel_plan(self.plan.as_ref(), text.len())?;
        say_plan(ps, &plan, text.len(), "Search-FlynnelText")?;
        for m in run_on_pool(&plan, job)? {
            ps.write(TextMatch::from(m))?;
        }
        Ok(())
    }
}

/// Counts bytes, lines, words and pattern occurrences in one large
/// string, all on Flynnel's workers.
///
/// # Examples
///
/// `Measure-FlynnelTextCount -Text $log`
///
/// `Measure-FlynnelTextCount -Text $log -Pattern 'WARN'`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelTextCount",
    alias = "Measure-FlyTextCount",
    output = ["Flynnel.TextMeasure"]
)]
#[derive(Default)]
pub struct MeasureFlynnelTextCount {
    /// The text to measure.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub text: String,
    /// A literal whose non-overlapping occurrences to count. Without
    /// one the Matches column is null rather than zero.
    #[param]
    pub pattern: Option<String>,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for MeasureFlynnelTextCount {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let text = std::mem::take(&mut self.text);
        let job = kernels::text_count(&text, self.pattern.as_deref()).map_err(refusal_err)?;
        let plan = kernel_plan(self.plan.as_ref(), text.len())?;
        say_plan(ps, &plan, text.len(), "Measure-FlynnelTextCount")?;
        let m = run_on_pool(&plan, job)?;
        ps.write(TextMeasure {
            bytes: m.bytes,
            lines: m.lines,
            words: m.words,
            matches: m.matches,
        })
    }
}

/// Splits one large string on a literal separator, found in parallel
/// on Flynnel's workers.
///
/// The whole answer crosses back in one call.
///
/// # Examples
///
/// `Split-FlynnelText -Text $csv -Separator ','`
#[cmdlet(
    verb = "Split",
    noun = "FlynnelText",
    alias = "Split-FlyText",
    output = ["System.String[]"]
)]
#[derive(Default)]
pub struct SplitFlynnelText {
    /// The text to split.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub text: String,
    /// The literal to split on.
    #[param(mandatory, position = 1)]
    pub separator: String,
    /// Drop the empty pieces two adjacent separators produce.
    #[param]
    pub no_empty: bool,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for SplitFlynnelText {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let text = std::mem::take(&mut self.text);
        let separator = std::mem::take(&mut self.separator);
        let job = kernels::split_text(&text, &separator, self.no_empty).map_err(refusal_err)?;
        let plan = kernel_plan(self.plan.as_ref(), text.len())?;
        say_plan(ps, &plan, text.len(), "Split-FlynnelText")?;
        ps.write(PsArray(run_on_pool(&plan, job)?))
    }
}

/// What Update-FlynnelText does to the text.
#[psenum(name = "Flynnel.TextTransform")]
#[derive(Clone, Copy, Debug, Default)]
pub enum TextTransform {
    /// Replace every non-overlapping occurrence of Pattern with
    /// Replacement.
    #[default]
    Replace,
    /// Upper-case, by the Unicode mapping.
    ToUpper,
    /// Lower-case, by the Unicode mapping.
    ToLower,
}

impl TextTransform {
    /// The crate's transform of the same name.
    fn to_kernel(self) -> kernels::TextTransform {
        match self {
            Self::Replace => kernels::TextTransform::Replace,
            Self::ToUpper => kernels::TextTransform::ToUpper,
            Self::ToLower => kernels::TextTransform::ToLower,
        }
    }
}

/// Rewrites one large string on Flynnel's workers and answers the
/// whole result in one call.
///
/// Replace finds every occurrence in parallel and then builds the
/// answer once, so the search scales and the copy is a single pass.
/// The case transforms run per piece. ToUpper splits on character
/// boundaries; ToLower splits only after an ASCII whitespace
/// character, because lower-casing a capital sigma depends on the
/// letters either side of it, and no piece boundary may stand between
/// them.
///
/// # Examples
///
/// `Update-FlynnelText -Text $s -Operation Replace -Pattern 'a' -Replacement 'b'`
///
/// `Update-FlynnelText -Text $s -Operation ToUpper`
#[cmdlet(
    verb = "Update",
    noun = "FlynnelText",
    alias = "Update-FlyText",
    output = ["System.String"]
)]
#[derive(Default)]
pub struct UpdateFlynnelText {
    /// The text to rewrite.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub text: String,
    /// What to do to it.
    #[param(mandatory, position = 1)]
    pub operation: TextTransform,
    /// The literal to replace, for Replace.
    #[param]
    pub pattern: Option<String>,
    /// What to put in its place, for Replace. Leaving it out deletes
    /// the pattern, which is a request rather than an omission.
    #[param]
    pub replacement: Option<String>,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for UpdateFlynnelText {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let text = std::mem::take(&mut self.text);
        let job = kernels::update_text(
            &text,
            self.operation.to_kernel(),
            self.pattern.as_deref(),
            self.replacement.as_deref(),
        )
        .map_err(refusal_err)?;
        let plan = kernel_plan(self.plan.as_ref(), text.len())?;
        say_plan(ps, &plan, text.len(), "Update-FlynnelText")?;
        ps.write(run_on_pool(&plan, job)?)
    }
}

// ---------------------------------------------------------------------
// The pool's own primitives, run over a declared body
// ---------------------------------------------------------------------

/// A pool primitive Measure-FlynnelPrimitive runs a declared operation
/// through.
#[psenum(name = "Flynnel.Primitive")]
#[derive(Clone, Copy, Debug, Default)]
pub enum Primitive {
    /// `reduce_chunks`: per-chunk sums folded into one, split as the
    /// pool chooses. It records the path it took on the calling thread,
    /// which Get-FlynnelReducePath reads.
    #[default]
    ReduceChunks,
    /// `for_each_chunk`: the operation applied in place, chunk by chunk.
    ForEachChunk,
    /// `for_each_chunk_indexed`: the same, each chunk given its start
    /// index.
    ForEachChunkIndexed,
    /// `collect_indexed`: one sum per chunk, four chunks a worker,
    /// collected in order.
    CollectIndexed,
}

/// What one run of a pool primitive answered.
#[psclass(name = "Flynnel.PrimitiveRun")]
#[derive(Clone, Default)]
pub struct PrimitiveRun {
    /// The primitive that ran.
    pub primitive: Primitive,
    /// The operation applied to each element.
    pub operation: MapOp,
    /// How many elements it read.
    pub count: u64,
    /// The operation's results added together. The primitive chooses its
    /// own split, so the last bits of this sum can differ between
    /// primitives and between hosts, which the declared kernels' answers
    /// do not.
    pub sum: f64,
    /// Nanoseconds the primitive's own call took, the serial sum after an
    /// in-place run not included.
    pub elapsed_ns: u64,
}

/// Runs one declared element operation through one of the pool's own
/// primitives and answers the sum of the results, for measuring the
/// primitives against each other on the same body.
///
/// The declared kernels run on blocks they cut themselves, dispatched
/// through the entry `flynnel_run_chunks_v1` uses, so none of them
/// reaches these primitives; this cmdlet is how a script does. After a
/// ReduceChunks run, Get-FlynnelReducePath reports the path the fold
/// took.
///
/// An operation that needs an operand (Clamp, Scale, Offset) is refused,
/// because the cmdlet takes none.
///
/// # Examples
///
/// `Measure-FlynnelPrimitive -InputObject $x -Primitive ReduceChunks; Get-FlynnelReducePath`
///
/// `Measure-FlynnelPrimitive -InputObject $x -Primitive ForEachChunk -Operation Sqrt`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelPrimitive",
    alias = "Measure-FlyPrimitive",
    output = ["Flynnel.PrimitiveRun"]
)]
#[derive(Default)]
pub struct MeasureFlynnelPrimitive {
    /// The numbers to run the operation over.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub input_object: Vec<f64>,
    /// Which primitive to run it through.
    #[param(mandatory, position = 1)]
    pub primitive: Primitive,
    /// The operation applied to each element. Square when not given.
    #[param]
    pub operation: MapOp,
    /// The plan to run under.
    #[param]
    pub plan: Option<Plan>,
}

impl Cmdlet for MeasureFlynnelPrimitive {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        use flynnel::sched::par_iter::{
            collect_indexed, for_each_chunk, for_each_chunk_indexed, reduce_chunks,
        };

        let mut items = std::mem::take(&mut self.input_object);
        let n = items.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Measure-FlynnelPrimitive")?;
        let element = kernels::Element::new(self.operation.to_kernel(), MapOperands::default())
            .map_err(refusal_err)?;
        let started = std::time::Instant::now();
        let folded = match self.primitive {
            Primitive::ReduceChunks => Some(reduce_chunks(
                &plan,
                &items,
                || 0.0f64,
                |acc, s| acc + s.iter().map(|&x| element.apply(x)).sum::<f64>(),
                |a, b| a + b,
            )),
            Primitive::ForEachChunk => {
                for_each_chunk(&plan, &mut items, |s| element.apply_slice(s));
                None
            }
            Primitive::ForEachChunkIndexed => {
                for_each_chunk_indexed(&plan, &mut items, |_start, s| element.apply_slice(s));
                None
            }
            Primitive::CollectIndexed => {
                // Four chunks a worker, so a chunk that finishes early
                // leaves its worker something to steal.
                let chunks = (plan.resolved_workers().max(1) * 4).min(n).max(1);
                let len = n.div_ceil(chunks).max(1);
                let sums: Vec<f64> = collect_indexed(&plan, chunks, 1, |c| {
                    let lo = (c * len).min(n);
                    let hi = (lo + len).min(n);
                    items[lo..hi].iter().map(|&x| element.apply(x)).sum::<f64>()
                });
                Some(sums.iter().sum::<f64>())
            }
        };
        let elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let sum = match folded {
            Some(sum) => sum,
            None => items.iter().sum::<f64>(),
        };
        ps.write(PrimitiveRun {
            primitive: self.primitive,
            operation: self.operation,
            count: n as u64,
            sum,
            elapsed_ns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parameters a cmdlet's own metadata declares, in its order:
    /// name, CLR type, mandatory, position, pipeline.
    fn declared(descriptor_json: &str) -> Vec<(String, String, bool, Option<u64>, bool)> {
        let parsed: serde_json::Value =
            serde_json::from_str(descriptor_json).expect("the cmdlet descriptor is JSON");
        let params = parsed["params"]
            .as_array()
            .expect("the descriptor lists its params");
        params
            .iter()
            .map(|p| {
                (
                    p["name"].as_str().expect("a name").to_string(),
                    p["clr"].as_str().expect("a CLR type").to_string(),
                    p["mandatory"].as_bool().expect("mandatory is a bool"),
                    p["position"].as_u64(),
                    p["pipeline"].as_bool().expect("pipeline is a bool"),
                )
            })
            .collect()
    }

    /// Fail naming every difference between a cmdlet and its kernel's
    /// descriptor.
    fn check<C: CmdletMeta>(kernel: &str) {
        let d = kernels::descriptor_for(kernel).expect("the crate describes this kernel");
        assert_eq!(d.cmdlet, C::NAME, "{kernel} names its cmdlet");
        let have = declared(&C::descriptor());
        let want: Vec<(String, String, bool, Option<u64>, bool)> = d
            .params
            .iter()
            .map(|p| {
                (
                    p.name.to_string(),
                    p.ty.clr().to_string(),
                    p.mandatory,
                    p.position.map(u64::from),
                    p.from_pipeline,
                )
            })
            .collect();
        assert_eq!(
            have,
            want,
            "{} and the {kernel} descriptor disagree",
            C::NAME
        );
    }

    #[test]
    fn every_kernel_cmdlet_matches_its_descriptor() {
        check::<InvokeFlynnelMap>("Map");
        check::<UpdateFlynnelArray>("MapInPlace");
        check::<InvokeFlynnelZip>("Zip");
        check::<MeasureFlynnelReduce>("Reduce");
        check::<GetFlynnelPrefixSum>("PrefixSum");
        check::<GetFlynnelHistogram>("Histogram");
        check::<GetFlynnelDotProduct>("DotProduct");
        check::<InvokeFlynnelSort>("Sort");
        check::<MeasureFlynnelFileHash>("FileHash");
        check::<TestFlynnelFileHash>("FileHashCheck");
        check::<SearchFlynnelFile>("SearchFile");
        check::<MeasureFlynnelFileLine>("FileLine");
        check::<MeasureFlynnelFileByte>("FileByte");
        check::<SearchFlynnelText>("SearchText");
        check::<MeasureFlynnelTextCount>("TextCount");
        check::<SplitFlynnelText>("SplitText");
        check::<UpdateFlynnelText>("UpdateText");
        assert_eq!(
            kernels::DESCRIPTORS.len(),
            17,
            "every descriptor was checked"
        );
    }

    #[test]
    fn the_module_enums_name_the_crate_operations_alike() {
        for (op, name) in kernels::MapOp::ALL.iter().zip(kernels::MapOp::NAMES) {
            let ours = match op {
                kernels::MapOp::Square => MapOp::Square,
                kernels::MapOp::Abs => MapOp::Abs,
                kernels::MapOp::Negate => MapOp::Negate,
                kernels::MapOp::Reciprocal => MapOp::Reciprocal,
                kernels::MapOp::Sqrt => MapOp::Sqrt,
                kernels::MapOp::Log => MapOp::Log,
                kernels::MapOp::Log2 => MapOp::Log2,
                kernels::MapOp::Exp => MapOp::Exp,
                kernels::MapOp::Round => MapOp::Round,
                kernels::MapOp::Floor => MapOp::Floor,
                kernels::MapOp::Ceiling => MapOp::Ceiling,
                kernels::MapOp::Clamp => MapOp::Clamp,
                kernels::MapOp::Scale => MapOp::Scale,
                kernels::MapOp::Offset => MapOp::Offset,
            };
            assert_eq!(format!("{ours:?}"), name);
            assert_eq!(ours.to_kernel(), *op);
        }
        for (op, name) in kernels::ZipOp::ALL.iter().zip(kernels::ZipOp::NAMES) {
            let ours = match op {
                kernels::ZipOp::Add => ZipOp::Add,
                kernels::ZipOp::Subtract => ZipOp::Subtract,
                kernels::ZipOp::Multiply => ZipOp::Multiply,
                kernels::ZipOp::Divide => ZipOp::Divide,
                kernels::ZipOp::Min => ZipOp::Min,
                kernels::ZipOp::Max => ZipOp::Max,
            };
            assert_eq!(format!("{ours:?}"), name);
            assert_eq!(ours.to_kernel(), *op);
        }
        for (op, name) in kernels::ReduceOp::ALL.iter().zip(kernels::ReduceOp::NAMES) {
            let ours = match op {
                kernels::ReduceOp::Sum => ReduceOp::Sum,
                kernels::ReduceOp::Min => ReduceOp::Min,
                kernels::ReduceOp::Max => ReduceOp::Max,
                kernels::ReduceOp::Mean => ReduceOp::Mean,
                kernels::ReduceOp::Variance => ReduceOp::Variance,
                kernels::ReduceOp::Product => ReduceOp::Product,
                kernels::ReduceOp::CountMatching => ReduceOp::CountMatching,
            };
            assert_eq!(format!("{ours:?}"), name);
            assert_eq!(ours.to_kernel(), *op);
        }
        for (op, name) in kernels::TextTransform::ALL
            .iter()
            .zip(kernels::TextTransform::NAMES)
        {
            let ours = match op {
                kernels::TextTransform::Replace => TextTransform::Replace,
                kernels::TextTransform::ToUpper => TextTransform::ToUpper,
                kernels::TextTransform::ToLower => TextTransform::ToLower,
            };
            assert_eq!(format!("{ours:?}"), name);
            assert_eq!(ours.to_kernel(), *op);
        }
    }

    #[test]
    fn unequal_columns_are_refused_naming_both_lengths() {
        let refused = TextMatch::from_columns(vec![1, 2], vec![1], vec!["a".to_string()])
            .err()
            .map(|e| e.to_string());
        assert!(
            refused
                .as_deref()
                .is_some_and(|m| m.contains("Index has 2 element(s) and LineNumber has 1")),
            "{refused:?}"
        );
        let rows = HashCheck::from_columns(
            vec!["a".to_string(), "b".to_string()],
            vec!["x".to_string(), "y".to_string()],
            vec!["x".to_string(), String::new()],
            vec![true, false],
        )
        .expect("equal columns");
        assert_eq!(rows[0].actual.as_deref(), Some("x"));
        assert_eq!(rows[1].actual, None);
    }
}

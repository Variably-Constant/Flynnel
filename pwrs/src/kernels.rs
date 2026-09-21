//! The declared kernels: the work this module owns and Flynnel's own
//! workers run.
//!
//! Flynnel's parallel primitives take Rust closures. A script has none
//! to give, and this module never runs a script block on a worker, so
//! the way a shell reaches `par_map_in_place`, `par_zip_apply`,
//! `reduce_chunks` and `collect_indexed` is to name a body the module
//! already owns. Each cmdlet here is one such body over one substrate:
//! arrays and numbers, files, or text.
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
//! managed value before the parallel section and hand the closure
//! plain Rust data.** Pinning is where this is easy to get wrong,
//! because `pin` is exactly what a kernel wants when its input is a
//! shell `double[]`, and calling it inside `collect_indexed` instead
//! of before it reads naturally and attaches every worker that runs
//! the chunk.
//!
//! Every body below keeps to it: the closures see `&mut [f64]`,
//! `&[u8]` and indices, and every `ps.write` sits outside the
//! parallel section.
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
//! governs and nothing here overrides it.
//!
//! `-Verbose` reports the plan that ran the work, the workers it
//! resolved to and the leaves it asked for.
//!
//! An input that cannot be read is a non-terminating error record
//! naming it, and the count of those refusals is written at the end, so
//! a run over a thousand files reports the nine it could not open
//! rather than stopping at the first or silently returning 991 rows.

use pwrs::prelude::*;

use blake3::hazmat::{
    ChainingValue, HasherExt, Mode, left_subtree_len, merge_subtrees_non_root,
    merge_subtrees_root,
};
use flynnel::sched::par_iter::{
    collect_indexed, for_each_chunk, for_each_chunk_indexed, reduce_chunks,
};

use crate::plan::Plan;

/// The error for a kernel argument the module cannot take.
fn arg_err(message: impl Into<String>) -> PsError {
    PsError::new(
        ErrorCategory::InvalidArgument,
        "FlynnelArgument",
        message.into(),
    )
}

/// The error for an input the kernel could not read, carrying the path
/// so a caller collecting error records can tell which one it was.
fn read_err(path: &str, detail: impl std::fmt::Display) -> PsError {
    PsError::new(
        ErrorCategory::ReadError,
        "FlynnelUnreadable",
        format!("{path} could not be read: {detail}"),
    )
}

/// The error for a state the kernel's own arithmetic says cannot
/// happen. It is an error rather than a fallback value because a
/// fallback would read as an answer.
fn internal_err(what: &str) -> PsError {
    PsError::new(
        ErrorCategory::InvalidResult,
        "FlynnelInternal",
        format!("{what}; this is a defect in the module, not in the input"),
    )
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
fn band_for(n: usize) -> u8 {
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
    if refused == 0 {
        return Ok(());
    }
    pwrs::warning!(
        ps,
        "{refused} of {total} input(s) could not be read; each one has an error record above"
    )
}

/// How many chunks to cut an input of `n` into for a plan.
///
/// Four leaves per worker rather than one: a chunk that finishes early
/// leaves its worker something to steal, and the per-chunk cost here is
/// a closure call over a slice rather than a dispatch.
fn chunking(plan: &flynnel::JobPlan, n: usize) -> (usize, usize) {
    if n == 0 {
        return (0, 0);
    }
    let workers = plan.resolved_workers().max(1) as usize;
    let n_chunks = (workers * 4).max(1).min(n);
    let chunk_len = n.div_ceil(n_chunks);
    (n_chunks, chunk_len)
}

/// The half-open range of chunk `c`.
fn span(c: usize, chunk_len: usize, n: usize) -> (usize, usize) {
    let lo = (c * chunk_len).min(n);
    let hi = (lo + chunk_len).min(n);
    (lo, hi)
}

// ---------------------------------------------------------------------
// Arrays and numbers
// ---------------------------------------------------------------------

/// An element-wise operation over one array.
#[psenum(name = "Flynnel.MapOp")]
#[derive(Clone, Copy, Default)]
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

/// A pairwise operation over two arrays of the same length.
#[psenum(name = "Flynnel.ZipOp")]
#[derive(Clone, Copy, Default)]
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

/// A reduction over one array to a single number.
#[psenum(name = "Flynnel.ReduceOp")]
#[derive(Clone, Copy, Default)]
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
/// element, so the closure carries plain numbers and a missing operand
/// is refused before any work is dispatched.
#[derive(Clone, Copy, Default)]
pub(crate) struct MapOperands {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub factor: Option<f64>,
    pub addend: Option<f64>,
}

/// The per-element body of one declared operation, with its operands
/// already read. Every family that runs a declared map goes through
/// this, so two of them cannot answer differently for the same
/// operation.
///
/// `Send` as well as `Sync` because the hybrid family moves the body
/// to a backend thread; the kernels family only shares it across
/// workers.
pub(crate) fn map_each(
    op: MapOp,
    operands: MapOperands,
) -> PsResult<Box<dyn Fn(&mut f64) + Send + Sync>> {
    Ok(match op {
        MapOp::Clamp => {
            let (Some(lo), Some(hi)) = (operands.min, operands.max) else {
                return Err(arg_err("Clamp needs both Min and Max").terminating());
            };
            if lo > hi {
                return Err(arg_err("Min must not be above Max").terminating());
            }
            Box::new(move |x: &mut f64| *x = x.clamp(lo, hi))
        }
        MapOp::Scale => {
            let Some(k) = operands.factor else {
                return Err(arg_err("Scale needs Factor").terminating());
            };
            Box::new(move |x: &mut f64| *x *= k)
        }
        MapOp::Offset => {
            let Some(k) = operands.addend else {
                return Err(arg_err("Offset needs Addend").terminating());
            };
            Box::new(move |x: &mut f64| *x += k)
        }
        MapOp::Square => Box::new(|x: &mut f64| *x *= *x),
        MapOp::Abs => Box::new(|x: &mut f64| *x = x.abs()),
        MapOp::Negate => Box::new(|x: &mut f64| *x = -*x),
        MapOp::Reciprocal => Box::new(|x: &mut f64| *x = 1.0 / *x),
        MapOp::Sqrt => Box::new(|x: &mut f64| *x = x.sqrt()),
        MapOp::Log => Box::new(|x: &mut f64| *x = x.ln()),
        MapOp::Log2 => Box::new(|x: &mut f64| *x = x.log2()),
        MapOp::Exp => Box::new(|x: &mut f64| *x = x.exp()),
        MapOp::Round => Box::new(|x: &mut f64| *x = x.round()),
        MapOp::Floor => Box::new(|x: &mut f64| *x = x.floor()),
        MapOp::Ceiling => Box::new(|x: &mut f64| *x = x.ceil()),
    })
}

/// Apply one element-wise operation across a slice on Flynnel's
/// workers. Shared by the copying and the in-place cmdlets so the two
/// cannot answer differently.
///
/// Dispatched with `for_each_chunk`, whose recursion floor is 256
/// items, and not with `par_map_in_place`, which is one task per
/// element. The crate says so plainly: par_map_in_place is for "few
/// large units", the shape of per-row matrix work, and these
/// operations are a multiply. Measured at 200,000 elements on pc2,
/// one task an element cost 5.21 ms against a 0.16 ms input crossing,
/// so the scheduling was 25 ns an element and the arithmetic was
/// nothing.
fn apply_map(
    plan: &flynnel::JobPlan,
    items: &mut [f64],
    op: MapOp,
    operands: MapOperands,
) -> PsResult<()> {
    let each = map_each(op, operands)?;
    for_each_chunk(plan, items, |slice| {
        for x in slice {
            each(x);
        }
    });
    Ok(())
}

impl Cmdlet for InvokeFlynnelMap {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let mut items = std::mem::take(&mut self.input_object);
        let n = items.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Invoke-FlynnelMap")?;
        if n == 0 {
            return ps.write(PsArray(Vec::<f64>::new()));
        }
        apply_map(
            &plan,
            &mut items,
            self.operation,
            MapOperands {
                min: self.min,
                max: self.max,
                factor: self.factor,
                addend: self.addend,
            },
        )?;
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
        // Pinned rather than copied. A pin fails on anything that is
        // not a double array, and that refusal is the whole contract:
        // silently copying instead would answer correctly and cost
        // exactly what this cmdlet exists to avoid.
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
        if n == 0 {
            return Ok(());
        }
        apply_map(
            &plan,
            &mut pinned,
            self.operation,
            MapOperands {
                min: self.min,
                max: self.max,
                factor: self.factor,
                addend: self.addend,
            },
        )
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
        if lhs.len() != rhs.len() {
            return Err(arg_err(format!(
                "Left has {} element(s) and Right has {}; a pairwise operation needs the same \
                 length on both",
                lhs.len(),
                rhs.len()
            ))
            .terminating());
        }
        let n = lhs.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Invoke-FlynnelZip")?;
        if n == 0 {
            return ps.write(PsArray(Vec::<f64>::new()));
        }
        // Chunked, for the same reason as the map: par_zip_apply is
        // one task per index, and these are a single instruction.
        // The index the chunk starts at is what reaches the right
        // operand, which no chunked helper pairs for us.
        let each: fn(&mut f64, f64) = match self.operation {
            ZipOp::Add => |a, b| *a += b,
            ZipOp::Subtract => |a, b| *a -= b,
            ZipOp::Multiply => |a, b| *a *= b,
            ZipOp::Divide => |a, b| *a /= b,
            ZipOp::Min => |a, b| *a = a.min(b),
            ZipOp::Max => |a, b| *a = a.max(b),
        };
        for_each_chunk_indexed(&plan, &mut lhs, |start, slice| {
            for (offset, a) in slice.iter_mut().enumerate() {
                each(a, rhs[start + offset]);
            }
        });
        ps.write(PsArray(lhs))
    }
}

/// The running state of a pairwise-combining mean and variance.
///
/// Carried rather than a sum of squares because the sum-of-squares
/// form subtracts two large nearly equal numbers, and over a long
/// array with a large mean that cancellation is the whole answer.
#[derive(Clone, Copy, Default)]
struct Moments {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Moments {
    fn push(mut self, x: f64) -> Self {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
        self
    }

    fn merge(self, other: Self) -> Self {
        if self.n == 0 {
            return other;
        }
        if other.n == 0 {
            return self;
        }
        let n = self.n + other.n;
        let delta = other.mean - self.mean;
        let mean = self.mean + delta * (other.n as f64 / n as f64);
        let m2 =
            self.m2 + other.m2 + delta * delta * (self.n as f64 * other.n as f64 / n as f64);
        Self { n, mean, m2 }
    }
}

/// Reduces an array to one number on Flynnel's workers, through
/// `reduce_chunks`.
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

        let op = self.operation;
        let value: Option<f64> = match op {
            ReduceOp::Sum => Some(reduce_chunks(
                &plan,
                &items,
                || 0.0f64,
                |acc, s| acc + s.iter().sum::<f64>(),
                |a, b| a + b,
            )),
            ReduceOp::Product => Some(reduce_chunks(
                &plan,
                &items,
                || 1.0f64,
                |acc, s| acc * s.iter().product::<f64>(),
                |a, b| a * b,
            )),
            ReduceOp::CountMatching => {
                let (Some(lo), Some(hi)) = (self.min, self.max) else {
                    return Err(arg_err("CountMatching needs both Min and Max").terminating());
                };
                if lo > hi {
                    return Err(arg_err("Min must not be above Max").terminating());
                }
                let c = reduce_chunks(
                    &plan,
                    &items,
                    || 0u64,
                    |acc, s| acc + s.iter().filter(|&&x| x >= lo && x <= hi).count() as u64,
                    |a, b| a + b,
                );
                Some(c as f64)
            }
            // The four below have no value over an empty input. A zero
            // would read as a measured answer, so the column is null
            // and Count says why.
            ReduceOp::Min | ReduceOp::Max | ReduceOp::Mean | ReduceOp::Variance if n == 0 => None,
            ReduceOp::Min => Some(reduce_chunks(
                &plan,
                &items,
                || f64::INFINITY,
                |acc, s| s.iter().fold(acc, |a, &x| a.min(x)),
                |a, b| a.min(b),
            )),
            ReduceOp::Max => Some(reduce_chunks(
                &plan,
                &items,
                || f64::NEG_INFINITY,
                |acc, s| s.iter().fold(acc, |a, &x| a.max(x)),
                |a, b| a.max(b),
            )),
            ReduceOp::Mean => {
                let m = reduce_chunks(
                    &plan,
                    &items,
                    Moments::default,
                    |acc, s| s.iter().fold(acc, |a, &x| a.push(x)),
                    Moments::merge,
                );
                Some(m.mean)
            }
            ReduceOp::Variance => {
                let m = reduce_chunks(
                    &plan,
                    &items,
                    Moments::default,
                    |acc, s| s.iter().fold(acc, |a, &x| a.push(x)),
                    Moments::merge,
                );
                Some(m.m2 / m.n as f64)
            }
        };

        ps.write(Reduction {
            operation: op,
            count: n as u64,
            value,
        })
    }
}

/// The inclusive running total of an array, computed as a two-phase
/// parallel scan on Flynnel's workers.
///
/// Phase one sums each chunk, phase two scans each chunk from its
/// chunk's offset. Both phases run through `collect_indexed`; the
/// offsets between them are a scan over one value per chunk.
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
        let items = std::mem::take(&mut self.input_object);
        let n = items.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Get-FlynnelPrefixSum")?;
        if n == 0 {
            return ps.write(PsArray(Vec::<f64>::new()));
        }
        let (n_chunks, chunk_len) = chunking(&plan, n);

        let sums: Vec<f64> = collect_indexed(&plan, n_chunks, 1, |c| {
            let (lo, hi) = span(c, chunk_len, n);
            items[lo..hi].iter().sum()
        });

        let mut offsets = Vec::with_capacity(n_chunks);
        let mut running = 0.0f64;
        for s in &sums {
            offsets.push(running);
            running += *s;
        }

        let parts: Vec<Vec<f64>> = collect_indexed(&plan, n_chunks, 1, |c| {
            let (lo, hi) = span(c, chunk_len, n);
            let mut acc = offsets[c];
            let mut out = Vec::with_capacity(hi - lo);
            for &x in &items[lo..hi] {
                acc += x;
                out.push(acc);
            }
            out
        });

        let mut out = Vec::with_capacity(n);
        for part in parts {
            out.extend_from_slice(&part);
        }
        ps.write(PsArray(out))
    }
}

/// Bins an array into equal-width buckets on Flynnel's workers.
///
/// Each chunk fills its own bin vector and the vectors are added, so
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
        if self.bins == 0 {
            return Err(arg_err("Bins must be at least one").terminating());
        }
        let bins = self.bins as usize;
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Get-FlynnelHistogram")?;
        if n == 0 {
            return Ok(());
        }
        let (n_chunks, chunk_len) = chunking(&plan, n);

        let lo = match self.min {
            Some(v) => v,
            None => reduce_chunks(
                &plan,
                &items,
                || f64::INFINITY,
                |acc, s| s.iter().fold(acc, |a, &x| a.min(x)),
                |a, b| a.min(b),
            ),
        };
        let hi = match self.max {
            Some(v) => v,
            None => reduce_chunks(
                &plan,
                &items,
                || f64::NEG_INFINITY,
                |acc, s| s.iter().fold(acc, |a, &x| a.max(x)),
                |a, b| a.max(b),
            ),
        };
        if !(lo.is_finite() && hi.is_finite()) {
            return Err(arg_err(
                "the range is not finite; pass Min and Max when the data holds an infinity or \
                 a NaN",
            )
            .terminating());
        }
        if lo > hi {
            return Err(arg_err("Min must not be above Max").terminating());
        }
        // A range of zero width would divide by zero. One bin holding
        // everything is what the data says.
        let width = if hi > lo {
            (hi - lo) / bins as f64
        } else {
            0.0
        };

        let parts: Vec<Vec<u64>> = collect_indexed(&plan, n_chunks, 1, |c| {
            let (a, b) = span(c, chunk_len, n);
            let mut local = vec![0u64; bins];
            for &x in &items[a..b] {
                if x.is_nan() || x < lo || x > hi {
                    continue;
                }
                let slot = if width > 0.0 {
                    // The top of the range belongs to the last bin
                    // rather than to a bin past the end.
                    (((x - lo) / width) as usize).min(bins - 1)
                } else {
                    0
                };
                local[slot] += 1;
            }
            local
        });

        let mut total = vec![0u64; bins];
        for part in &parts {
            for (slot, count) in part.iter().enumerate() {
                total[slot] += *count;
            }
        }

        if self.as_array {
            return ps.write(Histogram {
                low: lo,
                high: hi,
                width,
                counts: total,
            });
        }

        for (index, count) in total.iter().enumerate() {
            ps.write(HistogramBin {
                index: index as u32,
                low: lo + width * index as f64,
                high: lo + width * (index + 1) as f64,
                count: *count,
            })?;
        }
        Ok(())
    }
}

/// The dot product of two arrays, summed per chunk on Flynnel's
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
        if lhs.len() != rhs.len() {
            return Err(arg_err(format!(
                "Left has {} element(s) and Right has {}; a dot product needs the same length \
                 on both",
                lhs.len(),
                rhs.len()
            ))
            .terminating());
        }
        let n = lhs.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Get-FlynnelDotProduct")?;
        if n == 0 {
            return ps.write(0.0f64);
        }
        let (n_chunks, chunk_len) = chunking(&plan, n);
        let parts: Vec<f64> = collect_indexed(&plan, n_chunks, 1, |c| {
            let (a, b) = span(c, chunk_len, n);
            lhs[a..b]
                .iter()
                .zip(&rhs[a..b])
                .map(|(x, y)| x * y)
                .sum::<f64>()
        });
        ps.write(parts.iter().sum::<f64>())
    }
}

/// Sorts an array on Flynnel's workers: each chunk is sorted in
/// parallel, then the runs are merged in parallel rounds.
///
/// NaN sorts above every number, which is the total order
/// `f64::total_cmp` gives and the only one a comparison sort can use.
///
/// # Examples
///
/// `Sort-FlynnelArray -InputObject $x`
///
/// `Sort-FlynnelArray -InputObject $x -Descending`
#[cmdlet(
    verb = "Sort",
    noun = "FlynnelArray",
    alias = "Sort-FlyArray",
    output = ["System.Double[]"]
)]
#[derive(Default)]
pub struct SortFlynnelArray {
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

impl Cmdlet for SortFlynnelArray {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let items = std::mem::take(&mut self.input_object);
        let n = items.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Sort-FlynnelArray")?;
        if n <= 1 {
            return ps.write(PsArray(items));
        }
        let (n_chunks, chunk_len) = chunking(&plan, n);

        let mut runs: Vec<Vec<f64>> = collect_indexed(&plan, n_chunks, 1, |c| {
            let (a, b) = span(c, chunk_len, n);
            let mut part = items[a..b].to_vec();
            part.sort_by(f64::total_cmp);
            part
        });

        // Each round halves the run count, and every pair in a round
        // merges independently, so the round is one dispatch.
        while runs.len() > 1 {
            let pairs = runs.len().div_ceil(2);
            let taken = std::mem::take(&mut runs);
            runs = collect_indexed(&plan, pairs, 1, |p| {
                let left = &taken[p * 2];
                // An odd run count leaves the last run unpaired, which
                // carries forward to the next round unchanged.
                match taken.get(p * 2 + 1) {
                    None => left.clone(),
                    Some(right) => merge_runs(left, right),
                }
            });
        }

        let Some(mut out) = runs.into_iter().next() else {
            return Err(internal_err("the merge rounds consumed every run").terminating());
        };
        if self.descending {
            out.reverse();
        }
        ps.write(PsArray(out))
    }
}

/// One merge of two ascending runs.
fn merge_runs(left: &[f64], right: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < left.len() && j < right.len() {
        if left[i].total_cmp(&right[j]).is_le() {
            out.push(left[i]);
            i += 1;
        } else {
            out.push(right[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    out
}

// ---------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------

/// How much of a file one read takes. Large enough that the syscall is
/// amortized over real work, small enough that a thousand files in
/// flight do not each hold a large buffer.
const FILE_CHUNK: usize = 1 << 20;

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

/// Read a file whole, answering the failure text when it cannot be
/// read so the caller can put it in an error record.
fn slurp(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| e.to_string())
}

/// Lower-case hex for a 32-byte root.
fn hex32(bytes: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 0x0F) as usize] as char);
    }
    s
}

/// The largest file this module reads whole in order to hash its
/// subtrees in parallel. Above it the file is streamed and hashed on
/// one thread.
///
/// Both paths produce the same BLAKE3 root, so this bounds memory and
/// speed and never what the answer is.
const MAX_IN_MEMORY: u64 = 1 << 30;

/// The largest span one worker hashes as a single subtree. Below this
/// the split stops and one hasher runs the span, which is where
/// BLAKE3's own SIMD parallelism does the work; above it the span is
/// divided and the halves go to different workers.
const SUBTREE_LEAF: usize = 1 << 20;

/// The subtree spans of a buffer, left to right, by BLAKE3's own split
/// rule.
///
/// `left_subtree_len` is the only split that produces valid subtrees;
/// any other either panics or yields a root that is not BLAKE3's. The
/// recursion stops at a span small enough to be worth one worker, and
/// a span at or below one chunk cannot be split at all.
fn plan_subtrees(start: usize, len: usize, out: &mut Vec<(usize, usize)>) {
    if len <= SUBTREE_LEAF || len <= blake3::CHUNK_LEN {
        out.push((start, len));
        return;
    }
    let left = left_subtree_len(len as u64) as usize;
    plan_subtrees(start, left, out);
    plan_subtrees(start + left, len - left, out);
}

/// Combine the subtree chaining values back up the tree, walking the
/// same split that produced them so each one lands where it belongs.
fn fold_subtrees(
    start: usize,
    len: usize,
    cvs: &[ChainingValue],
    next: &mut usize,
) -> ChainingValue {
    if len <= SUBTREE_LEAF || len <= blake3::CHUNK_LEN {
        let cv = cvs[*next];
        *next += 1;
        return cv;
    }
    let left = left_subtree_len(len as u64) as usize;
    let l = fold_subtrees(start, left, cvs, next);
    let r = fold_subtrees(start + left, len - left, cvs, next);
    merge_subtrees_non_root(&l, &r, Mode::Hash)
}

/// The BLAKE3 root of a buffer, its subtrees hashed on Flynnel's
/// workers and merged into the standard root.
///
/// This is BLAKE3's own tree, not a scheme of this module's: each
/// worker hashes a span at its true input offset and answers the
/// chaining value the specification defines for it, and the merges are
/// the specification's. The answer equals `blake3::hash` over the same
/// bytes, which Kernels.Tests.ps1 checks against a published vector
/// and against the sequential path.
fn hash_bytes_parallel(plan: &flynnel::JobPlan, bytes: &[u8]) -> String {
    let n = bytes.len();
    if n <= SUBTREE_LEAF {
        return hex32(blake3::hash(bytes).as_bytes());
    }
    // The root split is the one merge that is root-flagged, so it is
    // taken here and the two halves are folded as ordinary subtrees.
    let left = left_subtree_len(n as u64) as usize;
    let mut spans = Vec::new();
    plan_subtrees(0, left, &mut spans);
    let left_count = spans.len();
    plan_subtrees(left, n - left, &mut spans);

    let cvs: Vec<ChainingValue> = collect_indexed(plan, spans.len(), 1, |i| {
        let (start, len) = spans[i];
        blake3::Hasher::new()
            .set_input_offset(start as u64)
            .update(&bytes[start..start + len])
            .finalize_non_root()
    });

    let mut from_left = 0usize;
    let lcv = fold_subtrees(0, left, &cvs[..left_count], &mut from_left);
    let mut from_right = 0usize;
    let rcv = fold_subtrees(left, n - left, &cvs[left_count..], &mut from_right);
    hex32(merge_subtrees_root(&lcv, &rcv, Mode::Hash).as_bytes())
}

/// The BLAKE3 root of a file read a chunk at a time, for a file too
/// large to hold.
fn hash_file_streaming(path: &str) -> Result<(String, u64), String> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; FILE_CHUNK];
    let mut total = 0u64;
    loop {
        let got = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if got == 0 {
            break;
        }
        hasher.update(&buf[..got]);
        total += got as u64;
    }
    Ok((hex32(hasher.finalize().as_bytes()), total))
}

/// The BLAKE3 root of one file, hashed across workers when it fits in
/// memory and streamed on one thread when it does not.
fn hash_file_parallel(plan: &flynnel::JobPlan, path: &str) -> Result<(String, u64), String> {
    let size = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    if size > MAX_IN_MEMORY {
        return hash_file_streaming(path);
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    Ok((hash_bytes_parallel(plan, &bytes), bytes.len() as u64))
}

/// The BLAKE3 root of each of several files, one file per worker.
///
/// Splitting within a file as well would nest a dispatch inside a
/// worker for no gain: with more files than workers every worker is
/// already busy, and the subtree split would only add merges.
fn hash_files(plan: &flynnel::JobPlan, paths: &[String]) -> Vec<Result<(String, u64), String>> {
    if paths.len() == 1 {
        return vec![hash_file_parallel(plan, &paths[0])];
    }
    collect_indexed(plan, paths.len(), 1, |i| hash_file_streaming(&paths[i]))
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
}

impl Cmdlet for MeasureFlynnelFileHash {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let paths = std::mem::take(&mut self.path);
        let n = paths.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Measure-FlynnelFileHash")?;
        if n == 0 {
            return Ok(());
        }
        let rows = hash_files(&plan, &paths);

        let mut refused = 0usize;
        for (i, row) in rows.into_iter().enumerate() {
            match row {
                Ok((hash, bytes)) => ps.write(FileHash {
                    path: paths[i].clone(),
                    hash,
                    bytes,
                })?,
                Err(why) => {
                    refused += 1;
                    ps.write_error(&read_err(&paths[i], why))?;
                }
            }
        }
        say_refusals(ps, refused, n)
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
        if paths.len() != manifest.len() {
            return Err(arg_err(format!(
                "Path has {} entr(ies) and Manifest has {}; each file needs exactly one \
                 expected root",
                paths.len(),
                manifest.len()
            ))
            .terminating());
        }
        let n = paths.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Test-FlynnelFileHash")?;
        if n == 0 {
            return Ok(());
        }
        let rows = hash_files(&plan, &paths);

        let mut refused = 0usize;
        for (i, row) in rows.into_iter().enumerate() {
            let expected = manifest[i].to_ascii_lowercase();
            match row {
                Ok((hash, _)) => {
                    let is_match = hash == expected;
                    ps.write(HashCheck {
                        path: paths[i].clone(),
                        expected,
                        actual: Some(hash),
                        is_match,
                    })?;
                }
                Err(why) => {
                    refused += 1;
                    ps.write_error(&read_err(&paths[i], why))?;
                    ps.write(HashCheck {
                        path: paths[i].clone(),
                        expected,
                        actual: None,
                        is_match: false,
                    })?;
                }
            }
        }
        say_refusals(ps, refused, n)
    }
}

/// A line with its terminating carriage return removed, when it has
/// one. A line with none is already the whole line.
fn without_cr(line: &[u8]) -> &[u8] {
    match line.strip_suffix(b"\r") {
        Some(trimmed) => trimmed,
        None => line,
    }
}

/// Whether `hay` holds `needle`.
fn contains(hay: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > hay.len() {
        return false;
    }
    hay.windows(needle.len()).any(|w| w == needle)
}

/// Every line of a byte buffer holding `needle`, with its one-based
/// line number.
fn matching_lines(hay: &[u8], needle: &[u8], ignore_case: bool) -> Vec<(u64, String)> {
    let mut out = Vec::new();
    if needle.is_empty() {
        return out;
    }
    let folded_needle = if ignore_case {
        needle.to_ascii_lowercase()
    } else {
        needle.to_vec()
    };
    for (idx, raw) in hay.split(|&b| b == b'\n').enumerate() {
        let line = without_cr(raw);
        let hit = if ignore_case {
            contains(&line.to_ascii_lowercase(), &folded_needle)
        } else {
            contains(line, &folded_needle)
        };
        if hit {
            out.push((idx as u64 + 1, String::from_utf8_lossy(line).into_owned()));
        }
    }
    out
}

/// Searches files for a literal string on Flynnel's workers, one task
/// per file.
///
/// The pattern is a literal, not a regular expression: a regex engine
/// is a dependency this module does not take, and a literal search is
/// what the parallel shape is for.
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
        if pattern.is_empty() {
            return Err(arg_err("Pattern must not be empty").terminating());
        }
        let n = paths.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Search-FlynnelFile")?;
        if n == 0 {
            return Ok(());
        }
        let needle = pattern.as_bytes();
        let fold = self.ignore_case;
        let rows: Vec<Result<Vec<(u64, String)>, String>> = collect_indexed(&plan, n, 1, |i| {
            slurp(&paths[i]).map(|bytes| matching_lines(&bytes, needle, fold))
        });

        let mut refused = 0usize;
        for (i, row) in rows.into_iter().enumerate() {
            match row {
                Ok(hits) => {
                    for (line_number, line) in hits {
                        ps.write(FileMatch {
                            path: paths[i].clone(),
                            line_number,
                            line,
                        })?;
                    }
                }
                Err(why) => {
                    refused += 1;
                    ps.write_error(&read_err(&paths[i], why))?;
                }
            }
        }
        say_refusals(ps, refused, n)
    }
}

/// Whether a buffer's last byte leaves a line open, which makes that
/// final unterminated line one more line.
fn unterminated_tail(bytes: &[u8]) -> u64 {
    match bytes.last() {
        None => 0,
        Some(&b'\n') => 0,
        Some(_) => 1,
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
        if n == 0 {
            return Ok(());
        }
        let rows: Vec<Result<(u64, u64), String>> = collect_indexed(&plan, n, 1, |i| {
            slurp(&paths[i]).map(|bytes| {
                let newlines = bytes.iter().filter(|&&b| b == b'\n').count() as u64;
                (newlines + unterminated_tail(&bytes), bytes.len() as u64)
            })
        });

        let mut refused = 0usize;
        for (i, row) in rows.into_iter().enumerate() {
            match row {
                Ok((lines, bytes)) => ps.write(FileMeasure {
                    path: paths[i].clone(),
                    lines,
                    bytes,
                })?,
                Err(why) => {
                    refused += 1;
                    ps.write_error(&read_err(&paths[i], why))?;
                }
            }
        }
        say_refusals(ps, refused, n)
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
        if n == 0 {
            return Ok(());
        }
        let rows: Vec<Result<u64, String>> = collect_indexed(&plan, n, 1, |i| {
            std::fs::metadata(&paths[i])
                .map(|m| m.len())
                .map_err(|e| e.to_string())
        });

        let mut refused = 0usize;
        for (i, row) in rows.into_iter().enumerate() {
            match row {
                Ok(bytes) => ps.write(FileMeasure {
                    path: paths[i].clone(),
                    lines: 0,
                    bytes,
                })?,
                Err(why) => {
                    refused += 1;
                    ps.write_error(&read_err(&paths[i], why))?;
                }
            }
        }
        say_refusals(ps, refused, n)
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

/// Every byte offset at which `needle` occurs in `hay`, found in
/// parallel and answered in ascending order.
///
/// A chunk searches its own span extended by the pattern's length less
/// one, so a match straddling a boundary is found by the chunk to its
/// left. A match belongs to the chunk its first byte falls in, which
/// is what keeps the overlap from reporting one twice.
fn find_all(plan: &flynnel::JobPlan, hay: &[u8], needle: &[u8]) -> Vec<usize> {
    let n = hay.len();
    if needle.is_empty() || needle.len() > n {
        return Vec::new();
    }
    let (n_chunks, chunk_len) = chunking(plan, n);
    if n_chunks == 0 {
        return Vec::new();
    }
    let reach = needle.len() - 1;
    let parts: Vec<Vec<usize>> = collect_indexed(plan, n_chunks, 1, |c| {
        let (lo, hi) = span(c, chunk_len, n);
        let stop = (hi + reach).min(n);
        if lo >= stop || stop - lo < needle.len() {
            return Vec::new();
        }
        let mut hits = Vec::new();
        for (offset, w) in hay[lo..stop].windows(needle.len()).enumerate() {
            if w == needle && lo + offset < hi {
                hits.push(lo + offset);
            }
        }
        hits
    });
    let mut all: Vec<usize> = parts.into_iter().flatten().collect();
    all.sort_unstable();
    all
}

/// The one-based line number of a byte offset, given the newline
/// positions in ascending order.
fn line_of(newlines: &[usize], at: usize) -> u64 {
    newlines.partition_point(|&p| p < at) as u64 + 1
}

/// The whole line containing a byte offset, without its terminator.
fn line_at(hay: &[u8], newlines: &[usize], at: usize) -> String {
    let idx = newlines.partition_point(|&p| p < at);
    let start = if idx == 0 { 0 } else { newlines[idx - 1] + 1 };
    // No newline at or after the offset means the line runs to the end
    // of the text.
    let end = match newlines.get(idx) {
        Some(&p) => p,
        None => hay.len(),
    };
    String::from_utf8_lossy(without_cr(&hay[start..end])).into_owned()
}

/// How many of the found offsets survive a left-to-right
/// non-overlapping walk.
fn non_overlapping(hits: &[usize], width: usize) -> usize {
    let mut kept = 0usize;
    let mut next_free = 0usize;
    for &at in hits {
        if at >= next_free {
            kept += 1;
            next_free = at + width;
        }
    }
    kept
}

/// Finds every occurrence of a literal in one large string, searched
/// in parallel on Flynnel's workers.
///
/// The whole string crosses once. Chunks overlap by the pattern's
/// length less one so a match on a boundary is still found, and each
/// match is owned by the chunk its first byte falls in so none is
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
        if pattern.is_empty() {
            return Err(arg_err("Pattern must not be empty").terminating());
        }
        let hay = text.as_bytes();
        let plan = kernel_plan(self.plan.as_ref(), hay.len())?;
        say_plan(ps, &plan, hay.len(), "Search-FlynnelText")?;
        if hay.is_empty() {
            return Ok(());
        }
        let hits = find_all(&plan, hay, pattern.as_bytes());
        let newlines = find_all(&plan, hay, b"\n");
        for at in hits {
            ps.write(TextMatch {
                index: at as u64,
                line_number: line_of(&newlines, at),
                line: line_at(hay, &newlines, at),
            })?;
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
        let hay = text.as_bytes();
        let n = hay.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Measure-FlynnelTextCount")?;

        let matches = match self.pattern.as_deref() {
            None => None,
            Some("") => return Err(arg_err("Pattern must not be empty").terminating()),
            Some(p) => {
                let hits = find_all(&plan, hay, p.as_bytes());
                Some(non_overlapping(&hits, p.len()) as u64)
            }
        };

        if n == 0 {
            return ps.write(TextMeasure {
                bytes: 0,
                lines: 0,
                words: 0,
                matches,
            });
        }
        let (n_chunks, chunk_len) = chunking(&plan, n);

        // A word is counted where it starts, so a chunk needs to know
        // whether the byte before its span was whitespace. Reading one
        // byte to the left is what makes the per-chunk counts add up to
        // the serial answer.
        let counts: Vec<(u64, u64)> = collect_indexed(&plan, n_chunks, 1, |c| {
            let (lo, hi) = span(c, chunk_len, n);
            let mut lines = 0u64;
            let mut words = 0u64;
            let mut prev_space = lo == 0 || hay[lo - 1].is_ascii_whitespace();
            for &b in &hay[lo..hi] {
                if b == b'\n' {
                    lines += 1;
                }
                let space = b.is_ascii_whitespace();
                if prev_space && !space {
                    words += 1;
                }
                prev_space = space;
            }
            (lines, words)
        });

        ps.write(TextMeasure {
            bytes: n as u64,
            lines: counts.iter().map(|(l, _)| l).sum::<u64>() + unterminated_tail(hay),
            words: counts.iter().map(|(_, w)| w).sum(),
            matches,
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
        if separator.is_empty() {
            return Err(arg_err("Separator must not be empty").terminating());
        }
        let hay = text.as_bytes();
        let plan = kernel_plan(self.plan.as_ref(), hay.len())?;
        say_plan(ps, &plan, hay.len(), "Split-FlynnelText")?;

        let hits = find_all(&plan, hay, separator.as_bytes());
        let width = separator.len();
        let mut out: Vec<String> = Vec::with_capacity(hits.len() + 1);
        let mut cursor = 0usize;
        for &at in &hits {
            if at < cursor {
                continue;
            }
            out.push(String::from_utf8_lossy(&hay[cursor..at]).into_owned());
            cursor = at + width;
        }
        out.push(String::from_utf8_lossy(&hay[cursor..]).into_owned());
        if self.no_empty {
            out.retain(|s| !s.is_empty());
        }
        ps.write(PsArray(out))
    }
}

/// What Update-FlynnelText does to the text.
#[psenum(name = "Flynnel.TextTransform")]
#[derive(Clone, Copy, Default)]
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

/// Split points for a string, every one of them on a character
/// boundary.
///
/// A byte-count chunking can land inside a multi-byte character, and
/// slicing there panics. Each nominal split walks forward until the
/// string agrees a boundary is there.
fn char_bounds(text: &str, plan: &flynnel::JobPlan) -> Vec<usize> {
    let n = text.len();
    let (n_chunks, chunk_len) = chunking(plan, n);
    let mut bounds = vec![0usize];
    let mut last = 0usize;
    for c in 1..n_chunks {
        let mut at = (c * chunk_len).min(n);
        while at < n && !text.is_char_boundary(at) {
            at += 1;
        }
        if at > last {
            bounds.push(at);
            last = at;
        }
    }
    if n > last {
        bounds.push(n);
    }
    bounds
}

/// Rewrites one large string on Flynnel's workers and answers the
/// whole result in one call.
///
/// Replace finds every occurrence in parallel and then builds the
/// answer once, so the search scales and the copy is a single pass.
/// The case transforms run per chunk, split on character boundaries so
/// no multi-byte character is cut.
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
        let n = text.len();
        let plan = kernel_plan(self.plan.as_ref(), n)?;
        say_plan(ps, &plan, n, "Update-FlynnelText")?;

        match self.operation {
            TextTransform::Replace => {
                let Some(pattern) = self.pattern.clone() else {
                    return Err(arg_err("Replace needs Pattern").terminating());
                };
                if pattern.is_empty() {
                    return Err(arg_err("Pattern must not be empty").terminating());
                }
                let replacement = match self.replacement.as_deref() {
                    Some(r) => r,
                    // Deleting the pattern is what an absent
                    // Replacement asks for.
                    None => "",
                };
                let hay = text.as_bytes();
                let hits = find_all(&plan, hay, pattern.as_bytes());
                let width = pattern.len();
                let mut out = String::with_capacity(n);
                let mut cursor = 0usize;
                for &at in &hits {
                    if at < cursor {
                        continue;
                    }
                    out.push_str(&String::from_utf8_lossy(&hay[cursor..at]));
                    out.push_str(replacement);
                    cursor = at + width;
                }
                out.push_str(&String::from_utf8_lossy(&hay[cursor..]));
                ps.write(out)
            }
            TextTransform::ToUpper | TextTransform::ToLower => {
                if n == 0 {
                    return ps.write(String::new());
                }
                let upper = matches!(self.operation, TextTransform::ToUpper);
                let bounds = char_bounds(&text, &plan);
                let n_parts = bounds.len() - 1;
                let parts: Vec<String> = collect_indexed(&plan, n_parts, 1, |c| {
                    let piece = &text[bounds[c]..bounds[c + 1]];
                    if upper {
                        piece.to_uppercase()
                    } else {
                        piece.to_lowercase()
                    }
                });
                ps.write(parts.concat())
            }
        }
    }
}

//! The array kernels: element-wise maps, pairwise zips, reductions, the
//! running total, histograms, the dot product and the sort.

use super::{
    ARRAY_BLOCK_MIN, BlockError, Blocks, Job, MAX_BLOCKS, Parts, Refusal, Slots, Tracker, tracked,
};

/// Blocks in the first phase of a job cut on `blocks`, or `None` when
/// the input is empty and there is nothing to run.
fn opening(blocks: &Blocks) -> Option<usize> {
    (blocks.count() > 0).then_some(blocks.count())
}

// ---------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------

/// An element-wise operation over one array.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
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
    /// Every operation, in declaration order.
    pub const ALL: [Self; 14] = [
        Self::Square,
        Self::Abs,
        Self::Negate,
        Self::Reciprocal,
        Self::Sqrt,
        Self::Log,
        Self::Log2,
        Self::Exp,
        Self::Round,
        Self::Floor,
        Self::Ceiling,
        Self::Clamp,
        Self::Scale,
        Self::Offset,
    ];

    /// Every operation's name, in the order of [`Self::ALL`].
    pub const NAMES: [&'static str; 14] = [
        "Square",
        "Abs",
        "Negate",
        "Reciprocal",
        "Sqrt",
        "Log",
        "Log2",
        "Exp",
        "Round",
        "Floor",
        "Ceiling",
        "Clamp",
        "Scale",
        "Offset",
    ];
}

/// A pairwise operation over two arrays of the same length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
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
    /// Every operation, in declaration order.
    pub const ALL: [Self; 6] = [
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Divide,
        Self::Min,
        Self::Max,
    ];

    /// Every operation's name, in the order of [`Self::ALL`].
    pub const NAMES: [&'static str; 6] = ["Add", "Subtract", "Multiply", "Divide", "Min", "Max"];
}

/// A reduction over one array to a single number.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
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
    /// than a sum of squares, so a large mean does not cancel the spread
    /// away.
    Variance,
    /// The product of every element.
    Product,
    /// How many elements sit within Min and Max inclusive.
    CountMatching,
}

impl ReduceOp {
    /// Every reduction, in declaration order.
    pub const ALL: [Self; 7] = [
        Self::Sum,
        Self::Min,
        Self::Max,
        Self::Mean,
        Self::Variance,
        Self::Product,
        Self::CountMatching,
    ];

    /// Every reduction's name, in the order of [`Self::ALL`].
    pub const NAMES: [&'static str; 7] = [
        "Sum",
        "Min",
        "Max",
        "Mean",
        "Variance",
        "Product",
        "CountMatching",
    ];
}

/// The operands a map operation may take. Each operation reads the ones
/// it needs and ignores the rest.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MapOperands {
    /// The lower bound, for Clamp.
    pub min: Option<f64>,
    /// The upper bound, for Clamp.
    pub max: Option<f64>,
    /// The multiplier, for Scale.
    pub factor: Option<f64>,
    /// The addend, for Offset.
    pub addend: Option<f64>,
}

/// One map operation with its operands read and checked, ready to apply
/// to one element or to a whole slice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Element {
    /// `x * x`.
    Square,
    /// The magnitude.
    Abs,
    /// `-x`.
    Negate,
    /// `1 / x`.
    Reciprocal,
    /// The square root.
    Sqrt,
    /// The natural logarithm.
    Log,
    /// The base-2 logarithm.
    Log2,
    /// `e` to the power x.
    Exp,
    /// To the nearest integer, halves away from zero.
    Round,
    /// To the largest integer at or below x.
    Floor,
    /// To the smallest integer at or above x.
    Ceiling,
    /// Held within the two bounds.
    Clamp {
        /// The lower bound.
        lo: f64,
        /// The upper bound, not below `lo`.
        hi: f64,
    },
    /// Multiplied by the factor.
    Scale(f64),
    /// Added to the addend.
    Offset(f64),
}

/// `f` applied to every element of `xs` in place. Generic, so each
/// operation's loop is compiled on its own and the compiler sees the
/// arithmetic rather than a call per element.
fn each(xs: &mut [f64], f: impl Fn(f64) -> f64) {
    for x in xs.iter_mut() {
        *x = f(*x);
    }
}

impl Element {
    /// `op` with the operands it needs, refused in the module's words when
    /// one is missing or the bounds cannot hold a value.
    pub fn new(op: MapOp, operands: MapOperands) -> Result<Self, Refusal> {
        Ok(match op {
            MapOp::Clamp => {
                let (Some(lo), Some(hi)) = (operands.min, operands.max) else {
                    return Err(Refusal::argument("Clamp needs both Min and Max"));
                };
                // f64::clamp panics on a NaN bound, which on a worker would
                // take the whole dispatch down.
                if lo.is_nan() || hi.is_nan() {
                    return Err(Refusal::argument("Min and Max must be numbers, not NaN"));
                }
                if lo > hi {
                    return Err(Refusal::argument("Min must not be above Max"));
                }
                Self::Clamp { lo, hi }
            }
            MapOp::Scale => match operands.factor {
                Some(k) => Self::Scale(k),
                None => return Err(Refusal::argument("Scale needs Factor")),
            },
            MapOp::Offset => match operands.addend {
                Some(k) => Self::Offset(k),
                None => return Err(Refusal::argument("Offset needs Addend")),
            },
            MapOp::Square => Self::Square,
            MapOp::Abs => Self::Abs,
            MapOp::Negate => Self::Negate,
            MapOp::Reciprocal => Self::Reciprocal,
            MapOp::Sqrt => Self::Sqrt,
            MapOp::Log => Self::Log,
            MapOp::Log2 => Self::Log2,
            MapOp::Exp => Self::Exp,
            MapOp::Round => Self::Round,
            MapOp::Floor => Self::Floor,
            MapOp::Ceiling => Self::Ceiling,
        })
    }

    /// The operation applied to one value.
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::Square => x * x,
            Self::Abs => x.abs(),
            Self::Negate => -x,
            Self::Reciprocal => 1.0 / x,
            Self::Sqrt => x.sqrt(),
            Self::Log => x.ln(),
            Self::Log2 => x.log2(),
            Self::Exp => x.exp(),
            Self::Round => x.round(),
            Self::Floor => x.floor(),
            Self::Ceiling => x.ceil(),
            Self::Clamp { lo, hi } => x.clamp(lo, hi),
            Self::Scale(k) => x * k,
            Self::Offset(k) => x + k,
        }
    }

    /// The operation applied to every element of `xs` in place, the
    /// choice of operation made once for the slice.
    pub fn apply_slice(self, xs: &mut [f64]) {
        match self {
            Self::Square => each(xs, |x| x * x),
            Self::Abs => each(xs, f64::abs),
            Self::Negate => each(xs, |x| -x),
            Self::Reciprocal => each(xs, |x| 1.0 / x),
            Self::Sqrt => each(xs, f64::sqrt),
            Self::Log => each(xs, f64::ln),
            Self::Log2 => each(xs, f64::log2),
            Self::Exp => each(xs, f64::exp),
            Self::Round => each(xs, f64::round),
            Self::Floor => each(xs, f64::floor),
            Self::Ceiling => each(xs, f64::ceil),
            Self::Clamp { lo, hi } => each(xs, |x| x.clamp(lo, hi)),
            Self::Scale(k) => each(xs, |x| x * k),
            Self::Offset(k) => each(xs, |x| x + k),
        }
    }
}

// ---------------------------------------------------------------------
// Map and zip
// ---------------------------------------------------------------------

/// [`map`]'s job: one phase, each block transforming its own range in
/// place.
pub struct MapJob<'a> {
    parts: Parts<'a, f64>,
    element: Element,
    tracker: Tracker,
}

/// Apply `op` to every element of `data` in place.
///
/// The module's Invoke-FlynnelMap runs it over its own copy of the input
/// and answers that copy; Update-FlynnelArray runs it over the caller's
/// pinned buffer. A caller that must keep its input passes a copy.
pub fn map(data: &mut [f64], op: MapOp, operands: MapOperands) -> Result<MapJob<'_>, Refusal> {
    let element = Element::new(op, operands)?;
    let blocks = Blocks::new(data.len(), ARRAY_BLOCK_MIN, MAX_BLOCKS);
    Ok(MapJob {
        tracker: Tracker::new(opening(&blocks)),
        parts: Parts::new(data, &blocks),
        element,
    })
}

impl Job for MapJob<'_> {
    type Answer = ();

    tracked!();

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.tracker.run(phase, block, || {
            let mut part = self.parts.lock(block);
            self.element.apply_slice(&mut part);
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.tracker.check_ended(phase)?;
        self.tracker.advance(None);
        Ok(())
    }

    fn finish(self) -> Result<(), Refusal> {
        self.tracker.check_finished()
    }
}

/// [`zip`]'s job: one phase, each block combining its own range of the
/// left operand with the right one, in place.
pub struct ZipJob<'a> {
    parts: Parts<'a, f64>,
    right: &'a [f64],
    blocks: Blocks,
    op: ZipOp,
    tracker: Tracker,
}

/// Combine `left` with `right` element by element, writing into `left`.
pub fn zip<'a>(left: &'a mut [f64], right: &'a [f64], op: ZipOp) -> Result<ZipJob<'a>, Refusal> {
    if left.len() != right.len() {
        return Err(Refusal::argument(format!(
            "Left has {} element(s) and Right has {}; a pairwise operation needs the same length \
             on both",
            left.len(),
            right.len()
        )));
    }
    let blocks = Blocks::new(left.len(), ARRAY_BLOCK_MIN, MAX_BLOCKS);
    Ok(ZipJob {
        tracker: Tracker::new(opening(&blocks)),
        parts: Parts::new(left, &blocks),
        right,
        blocks,
        op,
    })
}

/// `f` applied to each pair, writing into the left one.
fn pairwise(left: &mut [f64], right: &[f64], f: impl Fn(f64, f64) -> f64) {
    for (a, &b) in left.iter_mut().zip(right) {
        *a = f(*a, b);
    }
}

impl Job for ZipJob<'_> {
    type Answer = ();

    tracked!();

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.tracker.run(phase, block, || {
            let mut part = self.parts.lock(block);
            let right = &self.right[self.blocks.range(block)];
            match self.op {
                ZipOp::Add => pairwise(&mut part, right, |a, b| a + b),
                ZipOp::Subtract => pairwise(&mut part, right, |a, b| a - b),
                ZipOp::Multiply => pairwise(&mut part, right, |a, b| a * b),
                ZipOp::Divide => pairwise(&mut part, right, |a, b| a / b),
                ZipOp::Min => pairwise(&mut part, right, f64::min),
                ZipOp::Max => pairwise(&mut part, right, f64::max),
            }
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.tracker.check_ended(phase)?;
        self.tracker.advance(None);
        Ok(())
    }

    fn finish(self) -> Result<(), Refusal> {
        self.tracker.check_finished()
    }
}

// ---------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------

/// What a reduction answered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reduction {
    /// The operation that produced it.
    pub operation: ReduceOp,
    /// How many elements it read.
    pub count: u64,
    /// The answer. `None` for a reduction with no defined value over an
    /// empty input, which is every one but Sum, Product and
    /// CountMatching.
    pub value: Option<f64>,
}

/// The running state of a pairwise-combining mean and variance.
///
/// Carried rather than a sum of squares because the sum-of-squares form
/// subtracts two large nearly equal numbers, and over a long array with a
/// large mean that cancellation is the whole answer.
#[derive(Clone, Copy, Debug, Default)]
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
        let m2 = self.m2 + other.m2 + delta * delta * (self.n as f64 * other.n as f64 / n as f64);
        Self { n, mean, m2 }
    }
}

/// One block's part of a reduction.
#[derive(Clone, Copy, Debug)]
enum Partial {
    Value(f64),
    Count(u64),
    Moments(Moments),
}

/// [`reduce`]'s job: one phase of per-block partials, folded in block
/// order.
pub struct ReduceJob<'a> {
    input: &'a [f64],
    op: ReduceOp,
    bounds: (f64, f64),
    blocks: Blocks,
    partials: Slots<Partial>,
    folded: Vec<Partial>,
    tracker: Tracker,
}

/// Reduce `input` to one number. `min` and `max` are CountMatching's
/// bounds and are read by no other operation.
pub fn reduce(
    input: &[f64],
    op: ReduceOp,
    min: Option<f64>,
    max: Option<f64>,
) -> Result<ReduceJob<'_>, Refusal> {
    let bounds = if op == ReduceOp::CountMatching {
        let (Some(lo), Some(hi)) = (min, max) else {
            return Err(Refusal::argument("CountMatching needs both Min and Max"));
        };
        if lo > hi {
            return Err(Refusal::argument("Min must not be above Max"));
        }
        (lo, hi)
    } else {
        (f64::NEG_INFINITY, f64::INFINITY)
    };
    let blocks = Blocks::new(input.len(), ARRAY_BLOCK_MIN, MAX_BLOCKS);
    Ok(ReduceJob {
        input,
        op,
        bounds,
        partials: Slots::new(blocks.count()),
        folded: Vec::new(),
        tracker: Tracker::new(opening(&blocks)),
        blocks,
    })
}

impl ReduceJob<'_> {
    fn partial(&self, s: &[f64]) -> Partial {
        match self.op {
            ReduceOp::Sum => Partial::Value(s.iter().sum::<f64>()),
            ReduceOp::Product => Partial::Value(s.iter().product::<f64>()),
            ReduceOp::Min => Partial::Value(s.iter().fold(f64::INFINITY, |a, &x| a.min(x))),
            ReduceOp::Max => Partial::Value(s.iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x))),
            ReduceOp::Mean | ReduceOp::Variance => {
                Partial::Moments(s.iter().fold(Moments::default(), |a, &x| a.push(x)))
            }
            ReduceOp::CountMatching => {
                let (lo, hi) = self.bounds;
                Partial::Count(s.iter().filter(|&&x| x >= lo && x <= hi).count() as u64)
            }
        }
    }
}

/// The value inside a partial that must be one, or the refusal naming
/// the defect when a block produced the wrong kind.
fn value_of(p: &Partial) -> Result<f64, Refusal> {
    match p {
        Partial::Value(v) => Ok(*v),
        other => Err(Refusal::internal(format!(
            "a reduction block produced {other:?} where a value was due"
        ))),
    }
}

impl Job for ReduceJob<'_> {
    type Answer = Reduction;

    tracked!();

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.tracker.run(phase, block, || {
            let p = self.partial(&self.input[self.blocks.range(block)]);
            self.partials.put(block, p);
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.tracker.check_ended(phase)?;
        self.folded = self.partials.take_all()?;
        self.tracker.advance(None);
        Ok(())
    }

    fn finish(self) -> Result<Reduction, Refusal> {
        self.tracker.check_finished()?;
        let n = self.input.len();
        let value = match self.op {
            ReduceOp::Sum => Some(
                self.folded
                    .iter()
                    .map(value_of)
                    .try_fold(0.0f64, |a, v| v.map(|v| a + v))?,
            ),
            ReduceOp::Product => Some(
                self.folded
                    .iter()
                    .map(value_of)
                    .try_fold(1.0f64, |a, v| v.map(|v| a * v))?,
            ),
            ReduceOp::CountMatching => {
                let mut total = 0u64;
                for p in &self.folded {
                    match p {
                        Partial::Count(c) => total += c,
                        other => {
                            return Err(Refusal::internal(format!(
                                "a CountMatching block produced {other:?}"
                            )));
                        }
                    }
                }
                Some(total as f64)
            }
            // No value over an empty input: a zero would read as a
            // measured answer, so the value is absent and the count says
            // why.
            ReduceOp::Min | ReduceOp::Max | ReduceOp::Mean | ReduceOp::Variance if n == 0 => None,
            ReduceOp::Min => Some(
                self.folded
                    .iter()
                    .map(value_of)
                    .try_fold(f64::INFINITY, |a, v| v.map(|v| a.min(v)))?,
            ),
            ReduceOp::Max => Some(
                self.folded
                    .iter()
                    .map(value_of)
                    .try_fold(f64::NEG_INFINITY, |a, v| v.map(|v| a.max(v)))?,
            ),
            ReduceOp::Mean | ReduceOp::Variance => {
                let mut m = Moments::default();
                for p in &self.folded {
                    match p {
                        Partial::Moments(b) => m = m.merge(*b),
                        other => {
                            return Err(Refusal::internal(format!(
                                "a {:?} block produced {other:?}",
                                self.op
                            )));
                        }
                    }
                }
                Some(if self.op == ReduceOp::Mean {
                    m.mean
                } else {
                    m.m2 / m.n as f64
                })
            }
        };
        Ok(Reduction {
            operation: self.op,
            count: n as u64,
            value,
        })
    }
}

// ---------------------------------------------------------------------
// The running total
// ---------------------------------------------------------------------

/// [`prefix_sum`]'s job: block sums, a serial scan of them, then each
/// block's own scan from its offset, in place.
pub struct PrefixSumJob<'a> {
    parts: Parts<'a, f64>,
    sums: Slots<f64>,
    offsets: Vec<f64>,
    blocks: Blocks,
    tracker: Tracker,
}

/// Replace every element of `data` with the inclusive running total up
/// to it. A caller that must keep its input passes a copy.
pub fn prefix_sum(data: &mut [f64]) -> PrefixSumJob<'_> {
    let blocks = Blocks::new(data.len(), ARRAY_BLOCK_MIN, MAX_BLOCKS);
    PrefixSumJob {
        tracker: Tracker::new(opening(&blocks)),
        sums: Slots::new(blocks.count()),
        offsets: Vec::new(),
        parts: Parts::new(data, &blocks),
        blocks,
    }
}

impl Job for PrefixSumJob<'_> {
    type Answer = ();

    tracked!();

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.tracker.run(phase, block, || {
            let mut part = self.parts.lock(block);
            if phase == 0 {
                self.sums.put(block, part.iter().sum::<f64>());
            } else {
                let mut acc = self.offsets[block];
                for x in part.iter_mut() {
                    acc += *x;
                    *x = acc;
                }
            }
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.tracker.check_ended(phase)?;
        if phase == 0 {
            let sums = self.sums.take_all()?;
            let mut running = 0.0f64;
            self.offsets = sums
                .iter()
                .map(|s| {
                    let at = running;
                    running += *s;
                    at
                })
                .collect();
            self.tracker.advance(Some(self.blocks.count()));
        } else {
            self.tracker.advance(None);
        }
        Ok(())
    }

    fn finish(self) -> Result<(), Refusal> {
        self.tracker.check_finished()
    }
}

// ---------------------------------------------------------------------
// Histograms
// ---------------------------------------------------------------------

/// One bin of a histogram.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistogramBin {
    /// The bin's position, counting from zero.
    pub index: u32,
    /// The lowest value the bin takes.
    pub low: f64,
    /// The lowest value the next bin takes. The last bin takes this value
    /// itself, so the whole range is covered.
    pub high: f64,
    /// How many elements landed here.
    pub count: u64,
}

/// A whole histogram, with the counts in bin order.
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    /// The lowest value the first bin takes.
    pub low: f64,
    /// The highest value the last bin takes, which that bin takes itself.
    pub high: f64,
    /// How wide one bin is. Zero when every element has one value, which
    /// puts all of them in the first bin.
    pub width: f64,
    /// How many elements landed in each bin, as long as the bins asked
    /// for.
    pub counts: Vec<u64>,
}

impl Histogram {
    /// One row per bin: bin `i` runs from `low + width * i` to
    /// `low + width * (i + 1)`.
    pub fn bins(&self) -> Vec<HistogramBin> {
        self.counts
            .iter()
            .enumerate()
            .map(|(i, &count)| HistogramBin {
                index: i as u32,
                low: self.low + self.width * i as f64,
                high: self.low + self.width * (i + 1) as f64,
                count,
            })
            .collect()
    }
}

/// [`histogram`]'s job: the data's range when a bound is missing, then
/// per-block bin counts added in block order.
pub struct HistogramJob<'a> {
    input: &'a [f64],
    bins: usize,
    given: (Option<f64>, Option<f64>),
    extremes: Slots<(f64, f64)>,
    extreme_blocks: Blocks,
    counts: Slots<Vec<u64>>,
    bin_blocks: Blocks,
    /// The range and width, once known.
    scale: Option<(f64, f64, f64)>,
    binning_phase: usize,
    total: Option<Vec<u64>>,
    tracker: Tracker,
}

/// Bin `input` into `bins` equal-width buckets over `[min, max]`, each
/// bound taken from the data where it is not given.
///
/// A block holds at least four elements for every bin, so the per-block
/// counts it adds together total at most a quarter of the input's own
/// size and the fold stays small beside the binning.
pub fn histogram(
    input: &[f64],
    bins: u32,
    min: Option<f64>,
    max: Option<f64>,
) -> Result<HistogramJob<'_>, Refusal> {
    if bins == 0 {
        return Err(Refusal::argument("Bins must be at least one"));
    }
    let n = input.len();
    let bins = bins as usize;
    let extreme_blocks = Blocks::new(n, ARRAY_BLOCK_MIN, MAX_BLOCKS);
    let bin_blocks = Blocks::new(n, ARRAY_BLOCK_MIN.max(bins.saturating_mul(4)), MAX_BLOCKS);
    let mut job = HistogramJob {
        input,
        bins,
        given: (min, max),
        extremes: Slots::new(0),
        extreme_blocks,
        counts: Slots::new(0),
        bin_blocks,
        scale: None,
        binning_phase: 0,
        total: None,
        tracker: Tracker::new(None),
    };
    if n == 0 {
        return Ok(job);
    }
    match (min, max) {
        (Some(lo), Some(hi)) => {
            job.scale = Some(scale_for(lo, hi, bins)?);
            job.counts = Slots::new(bin_blocks.count());
            job.tracker = Tracker::new(Some(bin_blocks.count()));
        }
        _one_missing => {
            job.binning_phase = 1;
            job.extremes = Slots::new(extreme_blocks.count());
            job.tracker = Tracker::new(Some(extreme_blocks.count()));
        }
    }
    Ok(job)
}

/// The range and bin width for `[lo, hi]`, refused in the module's words
/// when the range cannot be binned.
fn scale_for(lo: f64, hi: f64, bins: usize) -> Result<(f64, f64, f64), Refusal> {
    if !(lo.is_finite() && hi.is_finite()) {
        return Err(Refusal::argument(
            "the range is not finite; pass Min and Max when the data holds an infinity or a NaN",
        ));
    }
    if lo > hi {
        return Err(Refusal::argument("Min must not be above Max"));
    }
    // A range of zero width would divide by zero. One bin holding
    // everything is what the data says.
    let width = if hi > lo {
        (hi - lo) / bins as f64
    } else {
        0.0
    };
    Ok((lo, hi, width))
}

impl Job for HistogramJob<'_> {
    type Answer = Option<Histogram>;

    tracked!();

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.tracker.run(phase, block, || {
            if phase < self.binning_phase {
                let s = &self.input[self.extreme_blocks.range(block)];
                let lo = s.iter().fold(f64::INFINITY, |a, &x| a.min(x));
                let hi = s.iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x));
                self.extremes.put(block, (lo, hi));
                return;
            }
            let Some((lo, hi, width)) = self.scale else {
                panic!("block {block} of the binning phase ran before the range was known");
            };
            let mut local = vec![0u64; self.bins];
            for &x in &self.input[self.bin_blocks.range(block)] {
                if x.is_nan() || x < lo || x > hi {
                    continue;
                }
                let slot = if width > 0.0 {
                    // The top of the range belongs to the last bin rather
                    // than to a bin past the end.
                    (((x - lo) / width) as usize).min(self.bins - 1)
                } else {
                    0
                };
                local[slot] += 1;
            }
            self.counts.put(block, local);
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.tracker.check_ended(phase)?;
        if phase < self.binning_phase {
            let found = self.extremes.take_all()?;
            let data_lo = found.iter().fold(f64::INFINITY, |a, e| a.min(e.0));
            let data_hi = found.iter().fold(f64::NEG_INFINITY, |a, e| a.max(e.1));
            let lo = self.given.0.unwrap_or(data_lo);
            let hi = self.given.1.unwrap_or(data_hi);
            self.scale = Some(scale_for(lo, hi, self.bins)?);
            self.counts = Slots::new(self.bin_blocks.count());
            self.tracker.advance(Some(self.bin_blocks.count()));
            return Ok(());
        }
        let mut total = vec![0u64; self.bins];
        for part in self.counts.take_all()? {
            for (slot, count) in part.iter().enumerate() {
                total[slot] += *count;
            }
        }
        self.total = Some(total);
        self.tracker.advance(None);
        Ok(())
    }

    fn finish(self) -> Result<Option<Histogram>, Refusal> {
        self.tracker.check_finished()?;
        if self.input.is_empty() {
            return Ok(None);
        }
        let (Some((low, high, width)), Some(counts)) = (self.scale, self.total) else {
            return Err(Refusal::internal(
                "the histogram ended with no range or no counts",
            ));
        };
        Ok(Some(Histogram {
            low,
            high,
            width,
            counts,
        }))
    }
}

// ---------------------------------------------------------------------
// The dot product
// ---------------------------------------------------------------------

/// [`dot_product`]'s job: per-block sums of products, added in block
/// order.
pub struct DotProductJob<'a> {
    left: &'a [f64],
    right: &'a [f64],
    blocks: Blocks,
    partials: Slots<f64>,
    folded: Vec<f64>,
    tracker: Tracker,
}

/// The dot product of `left` and `right`.
pub fn dot_product<'a>(left: &'a [f64], right: &'a [f64]) -> Result<DotProductJob<'a>, Refusal> {
    if left.len() != right.len() {
        return Err(Refusal::argument(format!(
            "Left has {} element(s) and Right has {}; a dot product needs the same length on both",
            left.len(),
            right.len()
        )));
    }
    let blocks = Blocks::new(left.len(), ARRAY_BLOCK_MIN, MAX_BLOCKS);
    Ok(DotProductJob {
        left,
        right,
        partials: Slots::new(blocks.count()),
        folded: Vec::new(),
        tracker: Tracker::new(opening(&blocks)),
        blocks,
    })
}

impl Job for DotProductJob<'_> {
    type Answer = f64;

    tracked!();

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.tracker.run(phase, block, || {
            let r = self.blocks.range(block);
            let s = self.left[r.clone()]
                .iter()
                .zip(&self.right[r])
                .map(|(x, y)| x * y)
                .sum::<f64>();
            self.partials.put(block, s);
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.tracker.check_ended(phase)?;
        self.folded = self.partials.take_all()?;
        self.tracker.advance(None);
        Ok(())
    }

    fn finish(self) -> Result<f64, Refusal> {
        self.tracker.check_finished()?;
        if self.folded.is_empty() {
            return Ok(0.0);
        }
        Ok(self.folded.iter().sum::<f64>())
    }
}

// ---------------------------------------------------------------------
// The sort
// ---------------------------------------------------------------------

/// The least number of elements a sort run holds, where the input has
/// that many.
const SORT_RUN_MIN: usize = 4096;

/// The most runs a sort is cut into, which bounds its merge rounds at
/// six.
const SORT_RUNS_MAX: usize = 64;

/// [`sort`]'s job: each run sorted, then merge rounds, each a phase whose
/// blocks merge one pair of runs.
pub struct SortJob<'a> {
    input: &'a [f64],
    descending: bool,
    blocks: Blocks,
    runs: Vec<Vec<f64>>,
    next: Slots<Vec<f64>>,
    /// The unpaired run of an odd round, carried to the next round's end.
    carry: Option<Vec<f64>>,
    done: Option<Vec<f64>>,
    tracker: Tracker,
}

/// Sort `input`, NaN above every number: the total order
/// `f64::total_cmp` gives, the only one a comparison sort can use. The
/// sorted order under it is unique, so the answer is the same for any
/// cut of the input.
pub fn sort(input: &[f64], descending: bool) -> SortJob<'_> {
    let blocks = Blocks::new(input.len(), SORT_RUN_MIN, SORT_RUNS_MAX);
    let trivial = input.len() <= 1;
    SortJob {
        input,
        descending,
        runs: Vec::new(),
        next: Slots::new(if trivial { 0 } else { blocks.count() }),
        carry: None,
        done: trivial.then(|| input.to_vec()),
        tracker: Tracker::new(if trivial { None } else { opening(&blocks) }),
        blocks,
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

impl Job for SortJob<'_> {
    type Answer = Vec<f64>;

    tracked!();

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.tracker.run(phase, block, || {
            if phase == 0 {
                let mut run = self.input[self.blocks.range(block)].to_vec();
                run.sort_unstable_by(f64::total_cmp);
                self.next.put(block, run);
            } else {
                let merged = merge_runs(&self.runs[2 * block], &self.runs[2 * block + 1]);
                self.next.put(block, merged);
            }
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.tracker.check_ended(phase)?;
        let mut produced = self.next.take_all()?;
        if let Some(carried) = self.carry.take() {
            produced.push(carried);
        }
        if produced.len() <= 1 {
            self.done = produced.pop();
            self.runs = Vec::new();
            self.tracker.advance(None);
            return Ok(());
        }
        if produced.len() % 2 == 1 {
            self.carry = produced.pop();
        }
        let pairs = produced.len() / 2;
        self.runs = produced;
        self.next = Slots::new(pairs);
        self.tracker.advance(Some(pairs));
        Ok(())
    }

    fn finish(self) -> Result<Vec<f64>, Refusal> {
        self.tracker.check_finished()?;
        let Some(mut out) = self.done else {
            return Err(Refusal::internal("the merge rounds left no run"));
        };
        if self.descending {
            out.reverse();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BlockState, drive, drive_serial};
    use super::*;

    /// Whether two slices hold the same values bit for bit, which tells
    /// -0.0 from 0.0 and one NaN from another where `==` cannot.
    fn same_bits(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits())
    }

    /// Deterministic values with a wide spread and some repeats.
    fn sample(n: usize) -> Vec<f64> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        (0..n)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
                if i % 97 == 0 {
                    0.5
                } else {
                    (unit - 0.5) * 1.0e6
                }
            })
            .collect()
    }

    /// Run every phase's blocks in reverse order on the calling thread.
    fn drive_reversed<J: Job>(job: J) -> Result<J::Answer, Refusal> {
        drive(job, |n, body| {
            for b in (0..n).rev() {
                body(b)?;
            }
            Ok(())
        })
    }

    /// Run every phase's blocks on four threads at once.
    fn drive_threaded<J: Job>(job: J) -> Result<J::Answer, Refusal> {
        drive(job, |n, body| {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let first_error = std::sync::Mutex::new(None);
            std::thread::scope(|scope| {
                for _worker in 0..4 {
                    scope.spawn(|| {
                        loop {
                            let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if b >= n {
                                break;
                            }
                            if let Err(e) = body(b) {
                                let mut held = first_error.lock().expect("the error lock");
                                if held.is_none() {
                                    *held = Some(e);
                                }
                            }
                        }
                    });
                }
            });
            match first_error.into_inner().expect("the error lock") {
                Some(e) => Err(e),
                None => Ok(()),
            }
        })
    }

    #[test]
    fn every_map_operation_matches_its_scalar_form() {
        let operands = MapOperands {
            min: Some(-10.0),
            max: Some(10.0),
            factor: Some(2.5),
            addend: Some(-3.0),
        };
        let input = sample(10_000);
        for op in MapOp::ALL {
            let element = Element::new(op, operands).expect("every operand is present");
            let mut data = input.clone();
            drive_threaded(map(&mut data, op, operands).expect("the operands hold"))
                .expect("the map finishes");
            let want: Vec<f64> = input.iter().map(|x| element.apply(*x)).collect();
            assert!(same_bits(&data, &want), "{op:?}");
        }
    }

    #[test]
    fn a_map_refuses_in_the_module_words() {
        let refusal = |op, operands| {
            let mut data = vec![1.0];
            map(&mut data, op, operands).err().map(|r| r.message)
        };
        assert_eq!(
            refusal(
                MapOp::Clamp,
                MapOperands {
                    min: Some(0.0),
                    ..MapOperands::default()
                }
            )
            .as_deref(),
            Some("Clamp needs both Min and Max")
        );
        assert_eq!(
            refusal(
                MapOp::Clamp,
                MapOperands {
                    min: Some(2.0),
                    max: Some(1.0),
                    ..MapOperands::default()
                }
            )
            .as_deref(),
            Some("Min must not be above Max")
        );
        assert_eq!(
            refusal(
                MapOp::Clamp,
                MapOperands {
                    min: Some(f64::NAN),
                    max: Some(1.0),
                    ..MapOperands::default()
                }
            )
            .as_deref(),
            Some("Min and Max must be numbers, not NaN")
        );
        assert_eq!(
            refusal(MapOp::Scale, MapOperands::default()).as_deref(),
            Some("Scale needs Factor")
        );
        assert_eq!(
            refusal(MapOp::Offset, MapOperands::default()).as_deref(),
            Some("Offset needs Addend")
        );
        let mut empty: Vec<f64> = Vec::new();
        drive_serial(
            map(&mut empty, MapOp::Sqrt, MapOperands::default()).expect("Sqrt takes no operand"),
        )
        .expect("an empty map finishes");
    }

    #[test]
    fn a_zip_pairs_every_element_and_refuses_unequal_lengths() {
        let a = sample(9000);
        let b: Vec<f64> = sample(9001)[1..].to_vec();
        for op in ZipOp::ALL {
            let mut left = a.clone();
            drive_reversed(zip(&mut left, &b, op).expect("equal lengths"))
                .expect("the zip finishes");
            let want: Vec<f64> = a
                .iter()
                .zip(&b)
                .map(|(x, y)| match op {
                    ZipOp::Add => x + y,
                    ZipOp::Subtract => x - y,
                    ZipOp::Multiply => x * y,
                    ZipOp::Divide => x / y,
                    ZipOp::Min => x.min(*y),
                    ZipOp::Max => x.max(*y),
                })
                .collect();
            assert!(same_bits(&left, &want), "{op:?}");
        }
        let mut short = vec![1.0];
        let refused = zip(&mut short, &b, ZipOp::Add).err().map(|r| r.message);
        assert_eq!(
            refused.as_deref(),
            Some(
                "Left has 1 element(s) and Right has 9000; a pairwise operation needs the same \
                 length on both"
            )
        );
    }

    #[test]
    fn a_reduction_answers_the_same_bits_whatever_order_its_blocks_ran_in() {
        let x = sample(300_001);
        for op in ReduceOp::ALL {
            let job = || reduce(&x, op, Some(-1.0e5), Some(1.0e5)).expect("bounds given");
            let serial = drive_serial(job()).expect("serial");
            let reversed = drive_reversed(job()).expect("reversed");
            let threaded = drive_threaded(job()).expect("threaded");
            let bits = |r: &Reduction| r.value.map(f64::to_bits);
            assert_eq!(bits(&serial), bits(&reversed), "{op:?}");
            assert_eq!(bits(&serial), bits(&threaded), "{op:?}");
            assert_eq!(serial.count, 300_001);
        }
    }

    #[test]
    fn a_reduction_agrees_with_the_plain_fold() {
        let x = sample(50_000);
        let value = |op| {
            drive_serial(reduce(&x, op, Some(-2.0e5), Some(2.0e5)).expect("bounds given"))
                .expect("finishes")
                .value
                .expect("a value over a non-empty input")
        };
        let sum: f64 = x.iter().sum();
        assert!((value(ReduceOp::Sum) - sum).abs() <= 1.0e-6 * x.len() as f64);
        let smallest = x.iter().fold(f64::INFINITY, |a, &v| a.min(v));
        assert_eq!(value(ReduceOp::Min), smallest);
        let largest = x.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v));
        assert_eq!(value(ReduceOp::Max), largest);
        let mean = sum / x.len() as f64;
        assert!((value(ReduceOp::Mean) - mean).abs() < 1.0e-6);
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / x.len() as f64;
        assert!((value(ReduceOp::Variance) - var).abs() / var < 1.0e-9);
        let inside = x.iter().filter(|&&v| (-2.0e5..=2.0e5).contains(&v)).count();
        assert_eq!(value(ReduceOp::CountMatching), inside as f64);
    }

    #[test]
    fn an_empty_reduction_answers_what_each_operation_means_there() {
        let answer = |op| {
            drive_serial(reduce(&[], op, Some(0.0), Some(1.0)).expect("bounds given"))
                .expect("finishes")
        };
        assert_eq!(answer(ReduceOp::Sum).value, Some(0.0));
        assert_eq!(answer(ReduceOp::Product).value, Some(1.0));
        assert_eq!(answer(ReduceOp::CountMatching).value, Some(0.0));
        for op in [
            ReduceOp::Min,
            ReduceOp::Max,
            ReduceOp::Mean,
            ReduceOp::Variance,
        ] {
            assert_eq!(answer(op).value, None, "{op:?}");
            assert_eq!(answer(op).count, 0);
        }
        let refused = reduce(&[], ReduceOp::CountMatching, Some(1.0), None)
            .err()
            .map(|r| r.message);
        assert_eq!(
            refused.as_deref(),
            Some("CountMatching needs both Min and Max")
        );
    }

    #[test]
    fn a_running_total_is_exact_on_integers_and_invariant_to_block_order() {
        let ints: Vec<f64> = (0..20_000).map(|i| (i % 13) as f64).collect();
        let mut scanned = ints.clone();
        drive_threaded(prefix_sum(&mut scanned)).expect("finishes");
        let mut running = 0.0;
        let want: Vec<f64> = ints
            .iter()
            .map(|x| {
                running += x;
                running
            })
            .collect();
        assert!(same_bits(&scanned, &want));
        let x = sample(70_000);
        let mut a = x.clone();
        let mut b = x;
        drive_serial(prefix_sum(&mut a)).expect("finishes");
        drive_reversed(prefix_sum(&mut b)).expect("finishes");
        assert!(same_bits(&a, &b));
    }

    #[test]
    fn a_histogram_counts_what_a_serial_pass_counts() {
        let x = sample(100_000);
        let got = drive_threaded(histogram(&x, 16, None, None).expect("bins"))
            .expect("finishes")
            .expect("a non-empty input");
        let lo = x.iter().fold(f64::INFINITY, |a, &v| a.min(v));
        let hi = x.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v));
        let width = (hi - lo) / 16.0;
        let mut want = vec![0u64; 16];
        for &v in &x {
            want[(((v - lo) / width) as usize).min(15)] += 1;
        }
        assert_eq!(got.counts, want);
        assert_eq!((got.low, got.high, got.width), (lo, hi, width));
        assert_eq!(got.bins()[3].count, want[3]);
        assert_eq!(got.counts.iter().sum::<u64>(), 100_000);
    }

    #[test]
    fn a_histogram_refuses_in_the_module_words() {
        let message = |x: &[f64], bins, min, max| match histogram(x, bins, min, max) {
            Err(r) => Some(r.message),
            Ok(job) => drive_serial(job).err().map(|r| r.message),
        };
        assert_eq!(
            message(&[1.0], 0, None, None).as_deref(),
            Some("Bins must be at least one")
        );
        assert_eq!(
            message(&[1.0, f64::INFINITY], 4, None, None).as_deref(),
            Some(
                "the range is not finite; pass Min and Max when the data holds an infinity or a NaN"
            )
        );
        assert_eq!(
            message(&[1.0], 4, Some(2.0), Some(1.0)).as_deref(),
            Some("Min must not be above Max")
        );
        assert_eq!(
            drive_serial(histogram(&[], 4, None, None).expect("bins")).expect("finishes"),
            None
        );
        let flat = drive_serial(histogram(&[3.0, 3.0], 4, None, None).expect("bins"))
            .expect("finishes")
            .expect("non-empty");
        assert_eq!(flat.counts, vec![2, 0, 0, 0]);
        assert_eq!(flat.width, 0.0);
    }

    #[test]
    fn a_dot_product_is_invariant_to_block_order() {
        let a = sample(123_457);
        let b: Vec<f64> = a.iter().map(|v| v * 0.5 + 1.0).collect();
        let serial = drive_serial(dot_product(&a, &b).expect("lengths")).expect("finishes");
        let threaded = drive_threaded(dot_product(&a, &b).expect("lengths")).expect("finishes");
        assert!(same_bits(&[serial], &[threaded]));
        assert_eq!(
            drive_serial(dot_product(&[], &[]).expect("lengths")).expect("finishes"),
            0.0
        );
    }

    #[test]
    fn a_sort_answers_the_standard_sort_in_either_direction() {
        let mut x = sample(300_000);
        x.push(f64::NAN);
        x.push(-0.0);
        x.push(0.0);
        let mut want = x.clone();
        want.sort_by(f64::total_cmp);

        let got = drive_threaded(sort(&x, false)).expect("finishes");
        assert!(same_bits(&got, &want));

        let down = drive_serial(sort(&x, true)).expect("finishes");
        want.reverse();
        assert!(same_bits(&down, &want));

        assert_eq!(
            drive_serial(sort(&[2.0], false)).expect("finishes"),
            vec![2.0]
        );
        assert!(drive_serial(sort(&[], false)).expect("finishes").is_empty());
    }

    #[test]
    fn a_stopped_phase_resumes_with_the_blocks_it_did_not_run() {
        let mut data = vec![3.0; 50_000];
        let mut job = map(&mut data, MapOp::Square, MapOperands::default()).expect("no operand");
        let n = job.blocks(0).expect("one phase");
        assert!(n > 2);
        job.run(0, 0).expect("block 0 runs");
        job.run(0, 2).expect("block 2 runs");
        assert_eq!(
            job.run(0, 0),
            Err(BlockError::AlreadyRan {
                phase: 0,
                block: 0,
                found: BlockState::Done
            })
        );
        let refused = job.end_phase(0).expect_err("block 1 has not run");
        assert_eq!(refused.message, "block 1 of phase 0 never ran");
        for b in 0..n {
            if job.state(0, b) == Some(BlockState::NotRun) {
                job.run(0, b).expect("a block not yet run runs");
            }
        }
        job.end_phase(0).expect("every block finished");
        job.finish().expect("every phase ended");
        assert!(data.iter().all(|&v| v == 9.0), "each element squared once");
    }

    #[test]
    fn the_names_follow_the_declaration_order() {
        for (op, name) in MapOp::ALL.iter().zip(MapOp::NAMES) {
            assert_eq!(format!("{op:?}"), name);
        }
        for (op, name) in ZipOp::ALL.iter().zip(ZipOp::NAMES) {
            assert_eq!(format!("{op:?}"), name);
        }
        for (op, name) in ReduceOp::ALL.iter().zip(ReduceOp::NAMES) {
            assert_eq!(format!("{op:?}"), name);
        }
    }
}

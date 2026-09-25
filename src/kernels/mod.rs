//! The declared kernels as plain Rust: each kernel's partition, its
//! per-block work and its combine, with no pool behind any of it.
//!
//! A kernel runs as a [`Job`]: one or more phases, each a set of blocks
//! that do not depend on one another, with a serial combine between
//! phases. A job decides every block from its input alone and never from
//! the pool that runs it, and folds per-block results in block order, so
//! which worker ran which block cannot reach an answer. Two callers that
//! drive the same job over the same input answer alike to the bit, on any
//! host and with any number of workers.
//!
//! Nothing here starts a pool, reads a [`crate::JobPlan`] or touches the
//! scheduler's global state. A caller runs the blocks wherever it likes:
//! on its own workers, through another library's chunk runner, or in
//! order on the calling thread with [`drive_serial`].
//!
//! # Driving a job
//!
//! ```
//! use flynnel::kernels::{self, Job, ReduceOp};
//!
//! let x = [1.0, 2.0, 3.0, 4.0];
//! let mut job = kernels::reduce(&x, ReduceOp::Sum, None, None)?;
//! let mut phase = 0;
//! while let Some(blocks) = job.blocks(phase) {
//!     for b in 0..blocks {
//!         job.run(phase, b).expect("each block of the open phase runs once");
//!     }
//!     job.end_phase(phase)?;
//!     phase += 1;
//! }
//! assert_eq!(job.finish()?.value, Some(10.0));
//! # Ok::<(), kernels::Refusal>(())
//! ```
//!
//! The blocks of one phase may run in any order and on any threads. A run
//! that stopped part way resumes by running the blocks whose
//! [`Job::state`] is still [`BlockState::NotRun`]; a block left
//! [`BlockState::Started`] by a panic cannot resume, because an in-place
//! kernel may have written part of it, and the job then refuses to end
//! its phase.
//!
//! # Refusals
//!
//! A job refuses with the words and identifiers the module's cmdlets
//! write: an argument it cannot take from its constructor, a range the
//! data does not have from [`Job::end_phase`], and an input it could not
//! read as a row of its answer, so a run over a thousand files reports
//! the nine it could not open rather than stopping at the first.

use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

mod arrays;
mod descriptor;
mod files;
mod text;

pub use arrays::{
    DotProductJob, Element, Histogram, HistogramBin, HistogramJob, MapJob, MapOp, MapOperands,
    PrefixSumJob, ReduceJob, ReduceOp, Reduction, SortJob, ZipJob, ZipOp, dot_product, histogram,
    map, prefix_sum, reduce, sort, zip,
};
pub use descriptor::{
    AnswerKind, DESCRIPTORS, InputKind, KernelDescriptor, ParamDescriptor, ParamType,
    descriptor_for,
};
pub use files::{
    FileByteJob, FileLineJob, FileMatch, FileMeasure, SearchFileJob, file_byte, file_line,
    search_file,
};
#[cfg(feature = "verify-chain")]
pub use files::{
    FileHash, FileHashCheckJob, FileHashJob, HashCheck, HashChecked, file_hash, file_hash_check,
    file_hash_streamed,
};
pub use text::{
    SearchTextJob, SplitTextJob, TextCountJob, TextMatch, TextMeasure, TextTransform,
    UpdateTextJob, search_text, split_text, text_count, update_text,
};

/// The revision of the kernels' answers.
///
/// Raised whenever a change here can change what any kernel answers for
/// some input, so two builds that report the same revision answer alike.
/// The module reports it beside its native entry points, which is how a
/// library driving these jobs on the module's pool can tell that the
/// module's cmdlets were built from the same kernels it links.
pub const REVISION: u32 = 1;

// ---------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------

/// Which kind of refusal a [`Refusal`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefusalId {
    /// An argument the kernel cannot take.
    Argument,
    /// An input the kernel could not read.
    Unreadable,
    /// A state the kernel's own logic says cannot arise, or a job driven
    /// out of its protocol.
    Internal,
}

impl RefusalId {
    /// The identifier the module's error records carry:
    /// `FlynnelArgument`, `FlynnelUnreadable` or `FlynnelInternal`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Argument => "FlynnelArgument",
            Self::Unreadable => "FlynnelUnreadable",
            Self::Internal => "FlynnelInternal",
        }
    }
}

/// Why a kernel refused, in the words the module's cmdlets use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// Which kind of refusal.
    pub id: RefusalId,
    /// The text, as the module writes it.
    pub message: String,
}

impl Refusal {
    /// An argument the kernel cannot take.
    pub(crate) fn argument(message: impl Into<String>) -> Self {
        Self {
            id: RefusalId::Argument,
            message: message.into(),
        }
    }

    /// An input that could not be read, naming it.
    pub(crate) fn unreadable(path: &str, detail: impl fmt::Display) -> Self {
        Self {
            id: RefusalId::Unreadable,
            message: format!("{path} could not be read: {detail}"),
        }
    }

    /// A state the kernel's own arithmetic says cannot arise. A refusal
    /// rather than a fallback value, because a fallback would read as an
    /// answer.
    pub(crate) fn internal(what: impl fmt::Display) -> Self {
        Self {
            id: RefusalId::Internal,
            message: format!("{what}; this is a defect in the kernel, not in the input"),
        }
    }

    /// A job driven out of its protocol: a phase ended before its blocks
    /// finished, or an answer asked for before its phases ended.
    pub(crate) fn misdriven(what: impl fmt::Display) -> Self {
        Self {
            id: RefusalId::Internal,
            message: what.to_string(),
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Refusal {}

/// The warning the module writes after a run that could not read some
/// of its inputs, or `None` when it read them all.
pub fn refusal_tally(refused: usize, total: usize) -> Option<String> {
    (refused > 0).then(|| {
        format!(
            "{refused} of {total} input(s) could not be read; each one has an error record above"
        )
    })
}

// ---------------------------------------------------------------------
// The job protocol
// ---------------------------------------------------------------------

/// Where one block of the open phase stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockState {
    /// Not yet claimed; running it now runs it.
    NotRun,
    /// Claimed and not finished: running now, or stopped part way by a
    /// panic, which leaves its output partly written.
    Started,
    /// Finished, its output complete.
    Done,
}

/// A block run outside the protocol. Never a property of the data: a
/// kernel's own refusals come from its constructor and
/// [`Job::end_phase`], and an input it could not read is a row of its
/// answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockError {
    /// The block was already claimed in this phase, so this call did
    /// nothing.
    AlreadyRan {
        /// The phase asked for.
        phase: usize,
        /// The block asked for.
        block: usize,
        /// Where the claim found it: started by another call and not
        /// finished, or finished.
        found: BlockState,
    },
    /// The phase asked for is not the open one.
    NotThisPhase {
        /// The phase asked for.
        phase: usize,
        /// The phase open now, or `None` once every phase has ended.
        open: Option<usize>,
    },
    /// The phase has fewer blocks than the one asked for.
    OutOfRange {
        /// The phase asked for.
        phase: usize,
        /// The block asked for.
        block: usize,
        /// How many blocks the phase has.
        blocks: usize,
    },
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::AlreadyRan {
                phase,
                block,
                found,
            } => write!(
                f,
                "block {block} of phase {phase} was already claimed and is {found:?}"
            ),
            Self::NotThisPhase {
                phase,
                open: Some(open),
            } => write!(f, "phase {phase} was asked for while phase {open} is open"),
            Self::NotThisPhase { phase, open: None } => {
                write!(f, "phase {phase} was asked for after every phase ended")
            }
            Self::OutOfRange {
                phase,
                block,
                blocks,
            } => write!(
                f,
                "block {block} of phase {phase} was asked for and the phase has {blocks}"
            ),
        }
    }
}

impl std::error::Error for BlockError {}

/// One kernel run over one input: phases of blocks, a serial combine
/// between phases, and an answer at the end.
///
/// The protocol, which every job checks rather than assumes:
///
/// 1. For phase 0, 1, 2 and on, while [`Job::blocks`] answers `Some(n)`,
///    run each block `0..n` once with [`Job::run`], in any order and on
///    any threads, then close the phase with [`Job::end_phase`].
/// 2. Once `blocks` answers `None`, take the answer with
///    [`Job::finish`].
///
/// `Sync`, because `run` takes `&self` from several threads at once.
pub trait Job: Sync {
    /// What the kernel answers.
    type Answer: Send;

    /// How many blocks `phase` has, or `None` when `phase` is not the
    /// open one, which after the last phase ends is every phase.
    fn blocks(&self, phase: usize) -> Option<usize>;

    /// Where block `block` of `phase` stands, or `None` when `phase` is
    /// not open or has no such block.
    fn state(&self, phase: usize, block: usize) -> Option<BlockState>;

    /// Run block `block` of `phase`. The block is claimed before any of
    /// its work, so a second call for it answers
    /// [`BlockError::AlreadyRan`] and does nothing.
    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError>;

    /// Close `phase`: refuse, naming the block, unless every block of it
    /// finished, then fold the phase's results in block order.
    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal>;

    /// The answer, once every phase has ended.
    fn finish(self) -> Result<Self::Answer, Refusal>
    where
        Self: Sized;
}

/// Run `job` to its answer, handing each phase's blocks to `run_blocks`.
///
/// `run_blocks(n, body)` must call `body(b)` for every `b` in `0..n`
/// before it returns, on whatever threads it likes, and answer the first
/// error a body answered. A block it failed to run is caught by
/// [`Job::end_phase`], which names it.
pub fn drive<J, R>(mut job: J, mut run_blocks: R) -> Result<J::Answer, Refusal>
where
    J: Job,
    R: FnMut(usize, &(dyn Fn(usize) -> Result<(), BlockError> + Sync)) -> Result<(), BlockError>,
{
    let mut phase = 0;
    while let Some(n) = job.blocks(phase) {
        let ran = {
            let body = |b: usize| job.run(phase, b);
            run_blocks(n, &body)
        };
        if let Err(e) = ran {
            return Err(Refusal::misdriven(e));
        }
        job.end_phase(phase)?;
        phase += 1;
    }
    job.finish()
}

/// Run `job` to its answer on the calling thread, each phase's blocks in
/// order.
pub fn drive_serial<J: Job>(job: J) -> Result<J::Answer, Refusal> {
    drive(job, |n, body| {
        for b in 0..n {
            body(b)?;
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------
// What every job is built from
// ---------------------------------------------------------------------

/// How an input of `n` items is cut into blocks: at least `min_len`
/// items a block where the input has that many, at most `max_blocks`
/// blocks, every block the same length but the last. It depends on
/// nothing else, which is what keeps an answer the same on every host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Blocks {
    n: usize,
    len: usize,
    count: usize,
}

impl Blocks {
    /// The cut for `n` items. Zero items is zero blocks.
    pub(crate) fn new(n: usize, min_len: usize, max_blocks: usize) -> Self {
        if n == 0 {
            return Self {
                n,
                len: 0,
                count: 0,
            };
        }
        let wanted = (n / min_len.max(1)).clamp(1, max_blocks.max(1));
        let len = n.div_ceil(wanted);
        Self {
            n,
            len,
            count: n.div_ceil(len),
        }
    }

    /// How many blocks.
    pub(crate) fn count(&self) -> usize {
        self.count
    }

    /// The length every block but the last has.
    pub(crate) fn block_len(&self) -> usize {
        self.len
    }

    /// The items block `b` covers.
    pub(crate) fn range(&self, b: usize) -> Range<usize> {
        let start = b.saturating_mul(self.len).min(self.n);
        start..start.saturating_add(self.len).min(self.n)
    }
}

/// The least number of items an array block holds, where the array has
/// that many. A block's fixed cost is a claim, a lock and a store, tens
/// of nanoseconds, against about a nanosecond an item for the lightest
/// kernel.
pub(crate) const ARRAY_BLOCK_MIN: usize = 2048;

/// The most blocks one phase of an array or text kernel is cut into:
/// enough for four blocks a worker at 64 workers.
pub(crate) const MAX_BLOCKS: usize = 256;

const NOT_RUN: u8 = 0;
const STARTED: u8 = 1;
const DONE: u8 = 2;

/// The state a block's word holds.
fn decode(word: u8) -> BlockState {
    if word == NOT_RUN {
        BlockState::NotRun
    } else if word == STARTED {
        BlockState::Started
    } else {
        BlockState::Done
    }
}

/// `n` blocks, none of them run.
fn fresh(n: usize) -> Vec<AtomicU8> {
    std::iter::repeat_with(|| AtomicU8::new(NOT_RUN))
        .take(n)
        .collect()
}

/// The open phase of a job and where each of its blocks stands.
pub(crate) struct Tracker {
    phase: usize,
    states: Vec<AtomicU8>,
    finished: bool,
}

impl Tracker {
    /// Phase 0 open with `blocks` blocks, or every phase already ended
    /// when `None`, which is a job with nothing to run.
    pub(crate) fn new(blocks: Option<usize>) -> Self {
        match blocks {
            Some(n) => Self {
                phase: 0,
                states: fresh(n),
                finished: false,
            },
            None => Self {
                phase: 0,
                states: Vec::new(),
                finished: true,
            },
        }
    }

    fn open(&self) -> Option<usize> {
        (!self.finished).then_some(self.phase)
    }

    pub(crate) fn blocks(&self, phase: usize) -> Option<usize> {
        (self.open() == Some(phase)).then_some(self.states.len())
    }

    pub(crate) fn state(&self, phase: usize, block: usize) -> Option<BlockState> {
        if self.open() != Some(phase) {
            return None;
        }
        self.states
            .get(block)
            .map(|s| decode(s.load(Ordering::Acquire)))
    }

    /// Claim block `block` of `phase`, run `work` on it, and mark it
    /// finished. A panic in `work` leaves the block claimed and not
    /// finished, which [`Tracker::check_ended`] then names.
    pub(crate) fn run(
        &self,
        phase: usize,
        block: usize,
        work: impl FnOnce(),
    ) -> Result<(), BlockError> {
        if self.open() != Some(phase) {
            return Err(BlockError::NotThisPhase {
                phase,
                open: self.open(),
            });
        }
        let Some(state) = self.states.get(block) else {
            return Err(BlockError::OutOfRange {
                phase,
                block,
                blocks: self.states.len(),
            });
        };
        if let Err(held) =
            state.compare_exchange(NOT_RUN, STARTED, Ordering::AcqRel, Ordering::Acquire)
        {
            return Err(BlockError::AlreadyRan {
                phase,
                block,
                found: decode(held),
            });
        }
        work();
        state.store(DONE, Ordering::Release);
        Ok(())
    }

    /// Refuse unless `phase` is open and every block of it finished.
    pub(crate) fn check_ended(&self, phase: usize) -> Result<(), Refusal> {
        match self.open() {
            Some(open) if open == phase => {}
            Some(open) => {
                return Err(Refusal::misdriven(format!(
                    "phase {phase} was ended while phase {open} is open"
                )));
            }
            None => {
                return Err(Refusal::misdriven(format!(
                    "phase {phase} was ended after every phase ended"
                )));
            }
        }
        for (block, state) in self.states.iter().enumerate() {
            match decode(state.load(Ordering::Acquire)) {
                BlockState::Done => {}
                BlockState::NotRun => {
                    return Err(Refusal::misdriven(format!(
                        "block {block} of phase {phase} never ran"
                    )));
                }
                BlockState::Started => {
                    return Err(Refusal::misdriven(format!(
                        "block {block} of phase {phase} started and did not finish, so its \
                         output is partly written and the job cannot resume"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Open the next phase with `blocks` blocks, or end the last one
    /// when `None`.
    pub(crate) fn advance(&mut self, blocks: Option<usize>) {
        self.phase += 1;
        match blocks {
            Some(n) => self.states = fresh(n),
            None => {
                self.states = Vec::new();
                self.finished = true;
            }
        }
    }

    /// Refuse unless every phase has ended.
    pub(crate) fn check_finished(&self) -> Result<(), Refusal> {
        match self.open() {
            None => Ok(()),
            Some(open) => Err(Refusal::misdriven(format!(
                "the answer was asked for while phase {open} is open"
            ))),
        }
    }
}

/// One result per block, each stored once by the block that owns it.
pub(crate) struct Slots<T> {
    cells: Vec<OnceLock<T>>,
}

impl<T> Slots<T> {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            cells: std::iter::repeat_with(OnceLock::new).take(n).collect(),
        }
    }

    /// Store block `b`'s result.
    ///
    /// The block's claim makes it the only writer of its cell, so a
    /// filled cell here is a defect in the job. It panics, which leaves
    /// the block started and not finished, so its phase refuses to end
    /// and names it.
    pub(crate) fn put(&self, b: usize, value: T) {
        if let Err(second) = self.cells[b].set(value) {
            drop(second);
            panic!("block {b}'s result was stored twice, which its claim rules out");
        }
    }

    /// Every result in block order, once every block has finished.
    pub(crate) fn take_all(&mut self) -> Result<Vec<T>, Refusal> {
        self.cells
            .iter_mut()
            .enumerate()
            .map(|(b, cell)| {
                cell.take()
                    .ok_or_else(|| Refusal::internal(format!("block {b} finished with no result")))
            })
            .collect()
    }
}

/// A slice cut into its blocks' ranges up front, each behind a lock of
/// its own, so blocks on several threads write their own ranges in place
/// without sharing one.
///
/// A block locks only its own part, and a block runs at most once a
/// phase, so no lock here is ever contended.
pub(crate) struct Parts<'a, T> {
    parts: Vec<Mutex<&'a mut [T]>>,
}

impl<'a, T> Parts<'a, T> {
    /// `data` cut on `blocks`, which must have been made for its length.
    pub(crate) fn new(data: &'a mut [T], blocks: &Blocks) -> Self {
        let len = blocks.block_len().max(1);
        Self {
            parts: data.chunks_mut(len).map(Mutex::new).collect(),
        }
    }

    /// Block `b`'s range.
    ///
    /// A part's only holder is its own block, one phase at a time, and a
    /// block that panicked stops its phase from ending, so a later phase
    /// never reaches a part a panic poisoned. A poisoned lock here is a
    /// defect in the job and panics, naming the block.
    pub(crate) fn lock(&self, b: usize) -> MutexGuard<'_, &'a mut [T]> {
        match self.parts[b].lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                panic!("block {b}'s range was reached after a panic in it: {poisoned}")
            }
        }
    }
}

/// The two [`Job`] methods every job answers from its tracker.
macro_rules! tracked {
    () => {
        fn blocks(&self, phase: usize) -> Option<usize> {
            self.tracker.blocks(phase)
        }

        fn state(&self, phase: usize, block: usize) -> Option<$crate::kernels::BlockState> {
            self.tracker.state(phase, block)
        }
    };
}
pub(crate) use tracked;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cut_depends_on_the_length_alone() {
        let b = Blocks::new(200_000, ARRAY_BLOCK_MIN, MAX_BLOCKS);
        assert_eq!(b.count(), 97);
        assert_eq!(b.range(0), 0..2062);
        assert_eq!(b.range(96).end, 200_000);
        let covered: usize = (0..b.count()).map(|i| b.range(i).len()).sum();
        assert_eq!(covered, 200_000);
        assert_eq!(Blocks::new(0, ARRAY_BLOCK_MIN, MAX_BLOCKS).count(), 0);
        assert_eq!(Blocks::new(5, ARRAY_BLOCK_MIN, MAX_BLOCKS).count(), 1);
        assert_eq!(
            Blocks::new(100_000_000, ARRAY_BLOCK_MIN, MAX_BLOCKS).count(),
            256
        );
    }

    #[test]
    fn no_cut_leaves_an_empty_block() {
        for n in 1..5000 {
            let b = Blocks::new(n, 7, 13);
            for i in 0..b.count() {
                assert!(!b.range(i).is_empty(), "n {n} block {i}");
            }
            assert_eq!(b.range(b.count() - 1).end, n);
        }
    }

    #[test]
    fn a_block_runs_once_and_a_phase_ends_only_when_every_block_finished() {
        let mut t = Tracker::new(Some(3));
        assert_eq!(t.blocks(0), Some(3));
        assert_eq!(t.blocks(1), None);
        t.run(0, 1, || ()).expect("block 1 runs");
        assert_eq!(
            t.run(0, 1, || ()),
            Err(BlockError::AlreadyRan {
                phase: 0,
                block: 1,
                found: BlockState::Done
            })
        );
        assert_eq!(
            t.run(0, 3, || ()),
            Err(BlockError::OutOfRange {
                phase: 0,
                block: 3,
                blocks: 3
            })
        );
        assert_eq!(
            t.run(1, 0, || ()),
            Err(BlockError::NotThisPhase {
                phase: 1,
                open: Some(0)
            })
        );
        let refused = t.check_ended(0).expect_err("blocks 0 and 2 never ran");
        assert_eq!(refused.message, "block 0 of phase 0 never ran");
        assert_eq!(t.state(0, 2), Some(BlockState::NotRun));
        t.run(0, 0, || ()).expect("block 0 runs");
        t.run(0, 2, || ()).expect("block 2 runs");
        t.check_ended(0).expect("every block of phase 0 finished");
        t.advance(None);
        assert_eq!(t.blocks(1), None);
        t.check_finished().expect("every phase ended");
    }

    #[test]
    fn a_block_that_panicked_stays_started_and_its_phase_cannot_end() {
        let t = Tracker::new(Some(1));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            t.run(0, 0, || panic!("mid-block"))
        }));
        match outcome {
            Ok(returned) => panic!("the block's panic did not reach the caller: {returned:?}"),
            Err(payload) => assert_eq!(payload.downcast_ref::<&str>(), Some(&"mid-block")),
        }
        assert_eq!(t.state(0, 0), Some(BlockState::Started));
        assert_eq!(
            t.run(0, 0, || ()),
            Err(BlockError::AlreadyRan {
                phase: 0,
                block: 0,
                found: BlockState::Started
            })
        );
        let refused = t.check_ended(0).expect_err("the block did not finish");
        assert!(refused.message.contains("started and did not finish"));
        assert_eq!(refused.id, RefusalId::Internal);
    }

    #[test]
    fn the_tally_speaks_only_when_something_was_refused() {
        assert_eq!(refusal_tally(0, 10), None);
        assert_eq!(
            refusal_tally(1, 2).as_deref(),
            Some("1 of 2 input(s) could not be read; each one has an error record above")
        );
    }
}

//! Racing: several attempts at one piece of work, and what taking the
//! first of them buys.
//!
//! # The two shapes answer opposite questions
//!
//! `Measure-FlynnelRaceAny` fires several attempts and keeps whichever
//! finishes first, signalling the rest to stop. That is the
//! tail-latency move: where an attempt's latency varies, firing a few
//! and taking the earliest trims the slow tail. What it reports is
//! what that trim is worth on this host, as `TailRatio`: the slowest
//! arm's time divided by the winner's, so 1.0 is a race that saved
//! nothing and larger is more trimmed.
//!
//! `Measure-FlynnelExploreSelect` runs every attempt to completion and
//! picks the best by a comparator. Nothing is cancelled, because a
//! slow explorer that finds the best answer is the entire point. What
//! it reports is what exploring costs: every arm is paid for.
//!
//! Run both over the same shape and the pair answers whether hedging
//! is worth it here, which is a property of the host and the workload
//! rather than of the scheduler.
//!
//! # The call returns when every arm has returned
//!
//! Both of them. Cancelling a loser does not hand the call back early;
//! it stops the loser from spending more. That is the crate's join
//! contract and the rows say so through `SlowestArmNs`, which is what
//! the call actually waited for.
//!
//! # Cancellation is observed, not imposed
//!
//! A losing arm stops at its next check, not the instant a peer wins,
//! so the body here polls the token between blocks of work rather than
//! per item: a token read per item would cost more than the work it
//! guards. `CancelledEarly` counts the arms that saw it and stopped,
//! and a race where none did is a race whose arms all finished before
//! the winner's signal reached them, which is a real answer about how
//! even this host is.
//!
//! # What is not bound here, and why
//!
//! The other seven racing entry points need an arm that can decline,
//! refute or disagree with another arm. Every body this module can
//! offer is a declared deterministic kernel: it cannot answer None to
//! a contract it failed, it has nothing to refute, and two arms of it
//! agree by construction. A cmdlet over `race_agree` would always
//! answer unanimous, which would be a property of the binding rather
//! than of the work. They carry census entries saying so.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use pwrs::prelude::*;

use flynnel::{CancelToken, JobPlan, explore_select, race_any};

use crate::host::arg_err;
use crate::kernels::{MapOp, MapOperands, map_each};

/// How often an arm looks at the cancel token, in items.
///
/// A read per item costs more than the arithmetic it guards; a read
/// per arm would never see a signal at all. Between blocks is where a
/// poll is cheap and still timely.
const POLL_EVERY: usize = 4096;

/// What one arm of a race did.
#[psclass(name = "Flynnel.RaceArm")]
#[derive(Clone, Default)]
pub struct RaceArm {
    /// Which attempt this was.
    pub index: u32,
    /// How long it ran.
    pub elapsed_ns: u64,
    /// Items it got through before it stopped or finished.
    pub items_done: u32,
    /// Whether it saw the cancel signal and stopped short. False on
    /// the winner, and false on a loser that finished before the
    /// signal reached it.
    pub cancelled_early: bool,
    /// Whether this arm won.
    pub won: bool,
    /// Its checksum, so the arms can be compared and the optimizer
    /// cannot delete the loop. Arms that ran to completion agree.
    pub checksum: f64,
}

/// What a race cost and what it saved.
#[psclass(name = "Flynnel.RaceOutcome")]
#[derive(Clone, Default)]
pub struct RaceOutcome {
    /// Attempts fired.
    pub attempts: u32,
    /// The attempt that won.
    pub winner_index: u32,
    /// How long the winner took.
    pub winner_ns: u64,
    /// How long the slowest arm took. The call waited for this,
    /// because cancelling a loser stops it spending more rather than
    /// handing the call back.
    pub slowest_arm_ns: u64,
    /// Wall time of the whole call.
    pub total_ns: u64,
    /// Arms that saw the cancel signal and stopped short. Zero means
    /// every loser finished before the winner's signal reached it,
    /// which says this host's arms are even rather than that
    /// cancellation is broken.
    pub cancelled_early: u32,
    /// The slowest arm's time over the winner's. What hedging trimmed
    /// on this host for this shape: 1.0 is a race that saved nothing.
    pub tail_ratio: f64,
    /// Items each arm was given.
    pub count: u32,
}

/// One arm's work: the declared operation over its own range, looking
/// at the cancel token between blocks.
fn run_arm(
    index: usize,
    count: usize,
    reps: u32,
    each: &(dyn Fn(&mut f64) + Send + Sync),
    token: Option<&CancelToken>,
) -> RaceArm {
    let t0 = std::time::Instant::now();
    let mut acc = 0.0f64;
    let mut done = 0usize;
    let mut cancelled = false;
    while done < count {
        if let Some(t) = token
            && t.is_cancelled()
        {
            cancelled = true;
            break;
        }
        let end = (done + POLL_EVERY).min(count);
        for i in done..end {
            // Above zero and small, so every declared operation
            // answers finitely: Log and Sqrt are defined and Exp does
            // not overflow.
            let mut x = 1.0 + (i % 64) as f64 * 0.125;
            for _ in 0..reps {
                each(&mut x);
            }
            acc += x;
        }
        done = end;
    }
    RaceArm {
        index: index as u32,
        elapsed_ns: t0.elapsed().as_nanos() as u64,
        items_done: done as u32,
        cancelled_early: cancelled,
        won: false,
        checksum: acc,
    }
}

/// Fires several attempts at one declared operation and keeps
/// whichever finishes first, signalling the rest to stop.
///
/// The tail-latency move. Where an attempt's latency varies, firing a
/// few and taking the earliest trims the slow tail, and TailRatio is
/// what that trim was worth here: the slowest arm's time over the
/// winner's, where 1.0 is a race that saved nothing.
///
/// The call returns once every arm has returned. Cancelling a loser
/// stops it spending more; it does not hand the call back early, and
/// SlowestArmNs is what the call actually waited for. That is the
/// crate's join contract and this cmdlet does not pretend otherwise.
///
/// CancelledEarly counts the arms that saw the signal and stopped.
/// Zero is a real answer and the common one at small sizes: it means
/// every loser finished before the winner's signal reached it, which
/// says the arms on this host are even rather than that cancellation
/// failed. Raise Count or Repetitions to give the signal time to
/// arrive.
///
/// Every arm runs the same body, so a difference between them is the
/// host and not the work. That is the point: what is being measured is
/// the spread this machine puts on identical attempts.
///
/// # Examples
///
/// `Measure-FlynnelRaceAny -Count 200000 -Attempts 8 -Operation Sqrt`
///
/// `(Measure-FlynnelRaceAny -Count 500000 -Attempts 16 -Operation Exp).TailRatio`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelRaceAny",
    alias = "Measure-FlyRaceAny",
    output = ["Flynnel.RaceOutcome"]
)]
#[derive(Default)]
pub struct MeasureFlynnelRaceAny {
    /// Items each arm is given.
    #[param(mandatory, position = 0)]
    pub count: u32,
    /// How many attempts to fire.
    #[param(position = 1)]
    pub attempts: Option<u32>,
    /// The declared operation each item goes through.
    #[param(position = 2)]
    pub operation: MapOp,
    /// How many times to apply it per item, which is the lever for
    /// per-item weight.
    #[param]
    pub repetitions: Option<u32>,
    /// Write a row per arm as well as the outcome, for a caller
    /// looking at the spread rather than the summary.
    #[param]
    pub include_arms: bool,
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

/// The operands shared by both cmdlets here.
fn operands(
    min: Option<f64>,
    max: Option<f64>,
    factor: Option<f64>,
    addend: Option<f64>,
) -> MapOperands {
    MapOperands {
        min,
        max,
        factor,
        addend,
    }
}

/// Attempts, defaulted and refused at zero: a race of none has no
/// winner, and the crate answers None rather than erroring, which a
/// cmdlet must not turn into an empty row.
fn attempt_count(attempts: Option<u32>) -> PsResult<usize> {
    let n = attempts.unwrap_or(4);
    if n == 0 {
        return Err(arg_err("Attempts must be above zero; a race of none has no winner")
            .terminating());
    }
    Ok(n as usize)
}

impl Cmdlet for MeasureFlynnelRaceAny {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let count = self.count as usize;
        if count == 0 {
            return Err(arg_err("Count must be above zero").terminating());
        }
        let n = attempt_count(self.attempts)?;
        let reps = self.repetitions.unwrap_or(1).max(1);
        let each: Arc<dyn Fn(&mut f64) + Send + Sync> = map_each(
            self.operation,
            operands(self.min, self.max, self.factor, self.addend),
        )?
        .into();
        let plan = JobPlan::bare(0, n as u32);

        // The arms write their own rows here rather than through the
        // race's return, because race_any hands back only the winner
        // and the spread across the losers is most of the answer.
        let arms: Arc<Vec<Mutex<Option<RaceArm>>>> =
            Arc::new((0..n).map(|_| Mutex::new(None)).collect());
        let cancelled = Arc::new(AtomicU32::new(0));

        let body = Arc::clone(&each);
        let slots = Arc::clone(&arms);
        let counter = Arc::clone(&cancelled);
        let t0 = std::time::Instant::now();
        let won = race_any(&plan, n, move |i, token| {
            let arm = run_arm(i, count, reps, body.as_ref(), Some(token));
            if arm.cancelled_early {
                counter.fetch_add(1, Ordering::Relaxed);
            }
            let elapsed = arm.elapsed_ns;
            // A poisoned slot still holds nothing anyone read, and
            // refusing here would lose an arm's row for a panic that
            // happened beside it.
            match slots[i].lock() {
                Ok(mut slot) => *slot = Some(arm),
                Err(poisoned) => *poisoned.into_inner() = Some(arm),
            }
            elapsed
        });
        let total_ns = t0.elapsed().as_nanos() as u64;

        let Some((winner_index, winner_ns)) = won else {
            return Err(arg_err("the race returned no winner, which needs at least one attempt")
                .terminating());
        };

        let mut rows: Vec<RaceArm> = Vec::with_capacity(n);
        for (i, slot) in arms.iter().enumerate() {
            let taken = match slot.lock() {
                Ok(mut g) => g.take(),
                Err(poisoned) => poisoned.into_inner().take(),
            };
            match taken {
                Some(mut arm) => {
                    arm.won = i == winner_index;
                    rows.push(arm);
                }
                // An arm with no row never ran its body, which is a
                // finding rather than a gap to skip over.
                None => {
                    return Err(arg_err(format!(
                        "attempt {i} left no row, so it never ran and the outcome below \
                         would describe fewer arms than were fired"
                    ))
                    .terminating());
                }
            }
        }
        let slowest = rows.iter().map(|a| a.elapsed_ns).max().unwrap_or(winner_ns);

        ps.write(RaceOutcome {
            attempts: n as u32,
            winner_index: winner_index as u32,
            winner_ns,
            slowest_arm_ns: slowest,
            total_ns,
            cancelled_early: cancelled.load(Ordering::Relaxed),
            tail_ratio: if winner_ns == 0 {
                1.0
            } else {
                slowest as f64 / winner_ns as f64
            },
            count: self.count,
        })?;
        if self.include_arms {
            for arm in rows {
                ps.write(arm)?;
            }
        }
        Ok(())
    }
}

/// Runs every attempt at one declared operation to completion and
/// reports the fastest, cancelling nothing.
///
/// The complement of Measure-FlynnelRaceAny. Nothing is cancelled
/// because a slow explorer that finds the best answer is the entire
/// point of this shape, so what this reports is what exploring costs:
/// every arm is paid for, and TotalNs covers all of them.
///
/// The comparator here is wall time, with the earlier index kept on a
/// tie, so the winner is the fastest arm and is deterministic given
/// deterministic arms.
///
/// Run this and Measure-FlynnelRaceAny over the same shape and the
/// pair says whether hedging is worth it on this host: the race's
/// TailRatio is what cancelling saved, and this cmdlet's spread is
/// what it would have cost to keep every arm.
///
/// # Examples
///
/// `Measure-FlynnelExploreSelect -Count 200000 -Attempts 8 -Operation Sqrt`
///
/// `Measure-FlynnelExploreSelect -Count 200000 -Attempts 8 -Operation Sqrt -IncludeArms`
#[cmdlet(
    verb = "Measure",
    noun = "FlynnelExploreSelect",
    alias = "Measure-FlyExploreSelect",
    output = ["Flynnel.RaceOutcome"]
)]
#[derive(Default)]
pub struct MeasureFlynnelExploreSelect {
    /// Items each arm is given.
    #[param(mandatory, position = 0)]
    pub count: u32,
    /// How many attempts to run.
    #[param(position = 1)]
    pub attempts: Option<u32>,
    /// The declared operation each item goes through.
    #[param(position = 2)]
    pub operation: MapOp,
    /// How many times to apply it per item.
    #[param]
    pub repetitions: Option<u32>,
    /// Write a row per arm as well as the outcome.
    #[param]
    pub include_arms: bool,
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

impl Cmdlet for MeasureFlynnelExploreSelect {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let count = self.count as usize;
        if count == 0 {
            return Err(arg_err("Count must be above zero").terminating());
        }
        let n = attempt_count(self.attempts)?;
        let reps = self.repetitions.unwrap_or(1).max(1);
        let each: Arc<dyn Fn(&mut f64) + Send + Sync> = map_each(
            self.operation,
            operands(self.min, self.max, self.factor, self.addend),
        )?
        .into();
        let plan = JobPlan::bare(0, n as u32);

        // Every arm's row is collected here, because explore_select
        // hands back only the winner and the spread is the answer.
        let slots: Arc<Vec<Mutex<Option<RaceArm>>>> =
            Arc::new((0..n).map(|_| Mutex::new(None)).collect());
        let body = Arc::clone(&each);
        let keep = Arc::clone(&slots);

        let t0 = std::time::Instant::now();
        let picked = explore_select(
            &plan,
            n,
            move |i| {
                // No token: this shape cancels nothing.
                let arm = run_arm(i, count, reps, body.as_ref(), None);
                let elapsed = arm.elapsed_ns;
                match keep[i].lock() {
                    Ok(mut slot) => *slot = Some(arm),
                    Err(poisoned) => *poisoned.into_inner() = Some(arm),
                }
                elapsed
            },
            // Fastest wins; the crate keeps the earlier index on a tie.
            |a, b| a < b,
        );
        let total_ns = t0.elapsed().as_nanos() as u64;

        let Some((winner_index, winner_ns)) = picked else {
            return Err(arg_err("the exploration returned nothing, which needs at least one attempt")
                .terminating());
        };

        let mut rows: Vec<RaceArm> = Vec::with_capacity(n);
        for (i, slot) in slots.iter().enumerate() {
            let taken = match slot.lock() {
                Ok(mut g) => g.take(),
                Err(poisoned) => poisoned.into_inner().take(),
            };
            match taken {
                Some(mut arm) => {
                    arm.won = i == winner_index;
                    rows.push(arm);
                }
                None => {
                    return Err(arg_err(format!(
                        "attempt {i} left no row, so it never ran and the outcome below \
                         would describe fewer arms than were run"
                    ))
                    .terminating());
                }
            }
        }
        let slowest = rows.iter().map(|a| a.elapsed_ns).max().unwrap_or(winner_ns);

        ps.write(RaceOutcome {
            attempts: n as u32,
            winner_index: winner_index as u32,
            winner_ns,
            slowest_arm_ns: slowest,
            total_ns,
            // Nothing is cancelled in this shape, and a zero here is
            // the shape rather than a host that was even.
            cancelled_early: 0,
            tail_ratio: if winner_ns == 0 {
                1.0
            } else {
                slowest as f64 / winner_ns as f64
            },
            count: self.count,
        })?;
        if self.include_arms {
            for arm in rows {
                ps.write(arm)?;
            }
        }
        Ok(())
    }
}

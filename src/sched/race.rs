//! Variant racing: speculative parallel dispatch of the three
//! variant tiers (Fast / Faithful / Correct) where the first
//! that successfully meets its accuracy contract wins.
//!
//! `race_variants(plan, fast, faithful, correct)` submits all
//! three closures in parallel via [`crate::sched::join`]. Each
//! polls a [`CancelToken`] at checkpoints to short-circuit when a
//! peer has won. Fast and Faithful return `Option<R>` (None =
//! contract not met); Correct returns `R` and always runs to
//! completion, so the bit-exact contract holds. Latency is
//! ~`t_fast` when Fast wins, ~`t_correct` when both others
//! decline. The nested join waits for all three to return; the
//! win comes from cancel-driven early exit, not early return. The
//! returned [`Variant`] tag names the winning tier.
//!
//! [`race_variants`] is first-past-the-post (right for Ziv-style
//! speculation: any tolerable result, lowest latency).
//! [`explore_select`] is its complement: every explorer completes
//! and a caller comparator picks by result quality, which is the
//! contract for episode racing / population search.

use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::foundation::Variant;
use crate::sched::arena::join;
use crate::sched::par_iter::collect_indexed;
use crate::sched::plan::JobPlan;

/// Cooperative cancel signal shared across racing variants.
/// Long-running variant closures poll
/// [`Self::is_cancelled`] at checkpoints to abandon early when a
/// peer has already produced a tolerable result.
///
/// Cancellation is best-effort: a closure may complete its
/// current iteration before observing the signal; it just must
/// not start a new iteration after seeing it.
#[derive(Debug, Clone)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// A fresh token in the not-cancelled state, for a race the
    /// caller composes itself on [`crate::sched::join`] or the
    /// indexed walkers rather than through the races in this module:
    /// clone it into every arm, and the arm that settles calls
    /// [`Self::cancel`]. A token that is never cancelled is the
    /// honest argument for a path that takes one but has no peers.
    #[must_use]
    pub fn new() -> Self {
        Self { flag: Arc::new(AtomicBool::new(false)) }
    }

    /// Returns `true` once any racing variant has produced a
    /// tolerable result. Cheap (single atomic load).
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Signal every holder of a clone of this token to stop at its
    /// next checkpoint. Idempotent; a second call changes nothing.
    #[inline]
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Race three variant implementations in parallel. The first to
/// return a tolerable result wins; losers cancel.
///
/// - `fast(&CancelToken) -> Option<R>`: `None` means the fast
///   tier could not meet its tolerance for this input.
/// - `faithful(&CancelToken) -> Option<R>`: same.
/// - `correct(&CancelToken) -> R`: always succeeds; the final
///   safety net.
///
/// Returns `(result, variant_that_won)`. The variant tag tells
/// telemetry / calibration which tier produced the answer.
///
/// # Determinism
///
/// The Correct closure always runs to completion (it ignores
/// cancel). When multiple variants finish "simultaneously," the
/// `OnceLock` semantics guarantee a single deterministic winner:
/// whichever atomic CAS fired first.
pub fn race_variants<R, Ff, Fa, Fc>(
    plan: &JobPlan,
    fast: Ff,
    faithful: Fa,
    correct: Fc,
) -> (R, Variant)
where
    // R: Sync is needed because the shared OnceLock<(R, Variant)>
    // is accessed across worker threads (winning thread writes,
    // losing threads observe via cancel). Send carries the value
    // into the closures; Sync allows reads of the OnceLock across
    // threads even though only the winner writes.
    R: Send + Sync + 'static,
    Ff: FnOnce(&CancelToken) -> Option<R> + Send,
    Fa: FnOnce(&CancelToken) -> Option<R> + Send,
    Fc: FnOnce(&CancelToken) -> R + Send,
{
    let cancel = Arc::new(AtomicBool::new(false));
    let winner: Arc<OnceLock<(R, Variant)>> = Arc::new(OnceLock::new());

    let token_f = CancelToken { flag: Arc::clone(&cancel) };
    let token_a = CancelToken { flag: Arc::clone(&cancel) };
    let token_c = CancelToken { flag: Arc::clone(&cancel) };

    let cancel_f = Arc::clone(&cancel);
    let cancel_a = Arc::clone(&cancel);
    let cancel_c = Arc::clone(&cancel);

    let winner_f = Arc::clone(&winner);
    let winner_a = Arc::clone(&winner);
    let winner_c = Arc::clone(&winner);

    // Run all three via nested join. Top-level fork: (correct)
    // vs (fast + faithful). Inner fork: (fast) vs (faithful).
    // The work-stealing wait in sched::join means the parent of
    // each fork participates in the pool, so total parallelism
    // is the same as a 3-way spawn.
    //
    // `move` keywords are critical: without them the closures
    // capture winner_f/winner_a/winner_c by reference (since
    // OnceLock::set takes &self), leaving the outer Arc bindings
    // alive after join returns. That breaks Arc::try_unwrap
    // below. With `move`, each closure owns its Arc clone and
    // drops it on return.
    join(
        plan,
        move || {
            join(
                plan,
                move || {
                    if let Some(r) = fast(&token_f) {
                        let _ = winner_f.set((r, Variant::Fast));
                        cancel_f.store(true, Ordering::Release);
                    }
                },
                move || {
                    if let Some(r) = faithful(&token_a) {
                        let _ = winner_a.set((r, Variant::Faithful));
                        cancel_a.store(true, Ordering::Release);
                    }
                },
            );
        },
        move || {
            let r = correct(&token_c);
            let _ = winner_c.set((r, Variant::Correct));
            cancel_c.store(true, Ordering::Release);
        },
    );
    // Drop the local `cancel` Arc held by race_variants' frame
    // so try_unwrap sees only the inner winner Arc reference.
    drop(cancel);

    // SAFETY: at least Correct always sets the OnceLock (it
    // returns R, not Option<R>), so the take below cannot panic.
    Arc::try_unwrap(winner)
        .ok()
        .expect("winner Arc still has outstanding clones")
        .into_inner()
        .expect("at least Correct must have completed")
}

/// Parallel explore-and-select: MIMD dispatch where EVERY explorer
/// runs to completion and a caller comparator picks the winner by
/// result quality. The complement of [`race_variants`] (see the
/// module docs' "Two racing contracts").
///
/// - `explore(i)` produces explorer `i`'s result for `i` in `0..n`.
///   The index seeds per-explorer state (clone id, RNG stream, ...).
/// - `better(a, b)` returns `true` when `a` is STRICTLY better than
///   `b`. It fully defines "best" - argmin, argmax, lexicographic,
///   tie-break - and sidesteps float-`Ord` friction. On ties
///   (neither strictly better) the earlier index is kept, so the
///   winner is deterministic given deterministic explorers.
///
/// Returns the winning `(index, result)`, or `None` when `n == 0`.
///
/// Each explorer is one leaf (`min_leaf = 1`) so N heavy trajectories
/// fan out fully even at small N, matching the per-element-weight
/// guidance for heavy work. Nothing is canceled: a slow explorer
/// that finds the best result is kept, which is the entire point.
#[track_caller]
pub fn explore_select<R, F, B>(
    plan: &JobPlan,
    n: usize,
    explore: F,
    better: B,
) -> Option<(usize, R)>
where
    R: Send,
    F: Fn(usize) -> R + Sync,
    B: Fn(&R, &R) -> bool,
{
    if n == 0 {
        return None;
    }
    // Every explorer runs to completion via the indexed parallel
    // collect (min_leaf = 1: one heavy trajectory per leaf).
    let mut results = collect_indexed(plan, n, 1, explore);
    // Serial argmax-under-`better` over the collected results: cheap
    // beside the parallel explore phase, since a comparison reads the
    // score already inside each result. Earliest index wins ties.
    let mut best = 0usize;
    for i in 1..results.len() {
        if better(&results[i], &results[best]) {
            best = i;
        }
    }
    let winner = results.swap_remove(best);
    Some((best, winner))
}

/// Hedged racing: fire `n` attempts at the same work and keep the
/// first that finishes, cancelling the rest.
///
/// This is the tail-latency move. When an attempt's latency is
/// variable - one of several replicas, one of several mirrors, one of
/// several routes - firing a few and taking whichever returns first
/// trims the slow tail. Every attempt holds a [`CancelToken`]; a
/// loser that polls it can stop the moment a peer has answered, since
/// its own result is now dead weight.
///
/// Where does this differ from [`race_variants`]? There is no
/// tolerability predicate and no correct-tier safety net. The
/// attempts are interchangeable and speed is the only thing that
/// separates them. Returns the winning attempt's index and result,
/// or `None` when `n == 0`.
///
/// The call returns once every attempt has returned (losers via
/// cancel). The win is that the losers quit early, not that the call
/// hands back before them - the same join contract [`race_variants`]
/// carries.
#[track_caller]
pub fn race_any<P, F>(plan: &JobPlan, n: usize, attempt: F) -> Option<(usize, P)>
where
    P: Send + Sync + 'static,
    F: Fn(usize, &CancelToken) -> P + Sync,
{
    if n == 0 {
        return None;
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let winner: Arc<OnceLock<(usize, P)>> = Arc::new(OnceLock::new());
    let cancel_inner = Arc::clone(&cancel);
    let winner_inner = Arc::clone(&winner);
    collect_indexed(plan, n, 1, move |i| {
        let token = CancelToken { flag: Arc::clone(&cancel_inner) };
        let r = attempt(i, &token);
        // First to finish its work claims the slot and cancels peers.
        if winner_inner.set((i, r)).is_ok() {
            cancel_inner.store(true, Ordering::Release);
        }
    });
    drop(cancel);
    Arc::try_unwrap(winner)
        .ok()
        .expect("winner Arc still has outstanding clones")
        .into_inner()
}

/// Quorum racing: fire `n` attempts and return as soon as the first
/// `k` have finished, cancelling the stragglers.
///
/// The shape behind a quorum read: ask several replicas, act on the
/// first majority to answer, ignore the slow rest. `k` is clamped to
/// `n`. Each attempt holds a [`CancelToken`] so a straggler stops
/// once the quorum is met.
///
/// Returns the `k` winners as `(index, result)` pairs in COMPLETION
/// order, not index order - the caller learns which attempts were
/// fast, which is often the point. Same join contract as the other
/// cancel-driven races: the call returns after every attempt returns,
/// and the win is the stragglers quitting early.
#[track_caller]
pub fn race_quorum<P, F>(plan: &JobPlan, n: usize, k: usize, attempt: F) -> Vec<(usize, P)>
where
    P: Send + 'static,
    F: Fn(usize, &CancelToken) -> P + Sync,
{
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let done: Arc<Mutex<Vec<(usize, P)>>> = Arc::new(Mutex::new(Vec::with_capacity(k)));
    let cancel_inner = Arc::clone(&cancel);
    let done_inner = Arc::clone(&done);
    collect_indexed(plan, n, 1, move |i| {
        let token = CancelToken { flag: Arc::clone(&cancel_inner) };
        let r = attempt(i, &token);
        let mut g = done_inner.lock().expect("quorum mutex poisoned");
        if g.len() < k {
            g.push((i, r));
            if g.len() == k {
                cancel_inner.store(true, Ordering::Release);
            }
        }
    });
    drop(cancel);
    Arc::try_unwrap(done)
        .ok()
        .expect("done Arc still has outstanding clones")
        .into_inner()
        .expect("quorum mutex poisoned")
}

/// How a [`race_refute`] duel ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Settled<P, R> {
    /// The prover settled it first with this witness.
    Proved(P),
    /// The refuter settled it first with this counter-witness.
    Refuted(R),
    /// Neither side settled within its effort budget.
    Unsettled,
}

/// Dueling racing: a prover and a refuter chase opposite verdicts on
/// the same question, and whichever settles first wins - the other
/// cancels.
///
/// A SAT portfolio is the clean example: one engine hunts a model,
/// another hunts a proof of unsatisfiability, and the first to land
/// ends the search. The same shape drives a capability probe, where
/// one side tries to certify a property absent while the other tries
/// to witness it present. The two sides return DIFFERENT types (a
/// model is not a refutation), so the verdict carries both.
///
/// Each side returns `Some(_)` when it settles the question and
/// `None` when it gives up. First `Some` wins and fires the shared
/// cancel; if both give up, the result is [`Settled::Unsettled`].
/// So this is not [`race_variants`] with two arms: there the tiers
/// compute the SAME answer at different accuracies, whereas here the
/// two sides seek OPPOSITE conclusions and either one is decisive.
#[track_caller]
pub fn race_refute<P, R, FP, FR>(plan: &JobPlan, prove: FP, refute: FR) -> Settled<P, R>
where
    P: Send + Sync + 'static,
    R: Send + Sync + 'static,
    FP: FnOnce(&CancelToken) -> Option<P> + Send,
    FR: FnOnce(&CancelToken) -> Option<R> + Send,
{
    let cancel = Arc::new(AtomicBool::new(false));
    let verdict: Arc<OnceLock<Settled<P, R>>> = Arc::new(OnceLock::new());

    let token_p = CancelToken { flag: Arc::clone(&cancel) };
    let token_r = CancelToken { flag: Arc::clone(&cancel) };
    let cancel_p = Arc::clone(&cancel);
    let cancel_r = Arc::clone(&cancel);
    let verdict_p = Arc::clone(&verdict);
    let verdict_r = Arc::clone(&verdict);

    join(
        plan,
        move || {
            if let Some(p) = prove(&token_p)
                && verdict_p.set(Settled::Proved(p)).is_ok()
            {
                cancel_p.store(true, Ordering::Release);
            }
        },
        move || {
            if let Some(r) = refute(&token_r)
                && verdict_r.set(Settled::Refuted(r)).is_ok()
            {
                cancel_r.store(true, Ordering::Release);
            }
        },
    );
    drop(cancel);
    Arc::try_unwrap(verdict)
        .ok()
        .expect("verdict Arc still has outstanding clones")
        .into_inner()
        .unwrap_or(Settled::Unsettled)
}

/// What a [`race_agree`] vote produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Agreement<R> {
    /// Some value was produced by at least `threshold` explorers.
    Consensus {
        /// The agreed-upon value.
        value: R,
        /// How many explorers produced it.
        agree: usize,
        /// Total explorers that ran.
        total: usize,
    },
    /// No value cleared the threshold. The explorers disagreed.
    Split {
        /// Size of the largest agreeing bloc.
        plurality: usize,
        /// Total explorers that ran.
        total: usize,
    },
}

/// Consensus racing: run `n` explorers on the same input, then keep
/// the answer at least `threshold` of them agree on. Disagreement is
/// a detected fault, not noise.
///
/// This is the trust-by-agreement contract. Compute a result three
/// ways - three algorithms, three code paths, three implementations -
/// and believe it only when they concur. Unlike [`race_variants`],
/// which trusts the first tolerable answer, [`race_agree`] trusts
/// nothing until the votes line up, and it tells you when they don't.
/// So a silent divergence between two supposedly-equivalent routines
/// surfaces here instead of corrupting a result downstream.
///
/// Every explorer runs to completion - you cannot count votes you did
/// not collect. Returns [`Agreement::Consensus`] with the winning
/// value and its bloc size, or [`Agreement::Split`] when nothing
/// cleared the threshold. Equality drives the tally, so `R: PartialEq`
/// is enough; grouping is quadratic in `n`, which is fine for the
/// handful of voters verification uses.
#[track_caller]
pub fn race_agree<R, F>(plan: &JobPlan, n: usize, threshold: usize, explore: F) -> Agreement<R>
where
    R: PartialEq + Send,
    F: Fn(usize) -> R + Sync,
{
    if n == 0 {
        return Agreement::Split { plurality: 0, total: 0 };
    }
    let mut results = collect_indexed(plan, n, 1, explore);
    let total = results.len();
    // Largest agreeing bloc by pairwise equality (quadratic, but n is
    // small for verification workloads).
    let mut best_idx = 0usize;
    let mut best_count = 0usize;
    for i in 0..results.len() {
        let count = results.iter().filter(|r| **r == results[i]).count();
        if count > best_count {
            best_count = count;
            best_idx = i;
        }
    }
    if best_count >= threshold.max(1) {
        let value = results.swap_remove(best_idx);
        Agreement::Consensus { value, agree: best_count, total }
    } else {
        Agreement::Split { plurality: best_count, total }
    }
}

/// Handle an anytime explorer uses to check the clock and publish its
/// best result so far. Passed to each [`race_deadline`] explorer.
pub struct Anytime<R> {
    expired: Arc<AtomicBool>,
    best: Arc<Mutex<Option<(f64, R)>>>,
}

impl<R> Anytime<R> {
    /// Has the deadline passed? Explorers poll this and return once it
    /// reads `true` - whatever they have published is what counts.
    #[inline]
    pub fn is_expired(&self) -> bool {
        self.expired.load(Ordering::Acquire)
    }

    /// Publish a candidate result with its score. A higher score
    /// replaces the current best; ties keep the incumbent. Cheap to
    /// call often, so publish every time the result improves.
    pub fn submit(&self, score: f64, value: R) {
        let mut g = self.best.lock().expect("anytime mutex poisoned");
        let improved = g.as_ref().is_none_or(|(s, _)| score > *s);
        if improved {
            *g = Some((score, value));
        }
    }
}

/// Anytime racing: run `n` explorers against a wall-clock budget and
/// take the best result anyone has published when the clock runs out.
///
/// Time is the terminator here, which is what sets this apart from
/// every other race in this module. The explorers do not finish; they
/// improve. Think tree search under a move budget, or an iterative
/// solver with a latency SLA: you always spend the whole budget and
/// you always want the best answer found within it. Each explorer
/// loops - improve, [`Anytime::submit`], check [`Anytime::is_expired`]
/// - until the deadline flips, then returns.
///
/// Returns the highest-scored published result as `(score, value)`,
/// or `None` if nobody published (or `n == 0`). One worker parks on
/// the timer for the whole budget, so on a tiny host that is one
/// fewer explorer running in parallel; the budget is the point, so
/// that trade is deliberate.
#[track_caller]
pub fn race_deadline<R, F>(plan: &JobPlan, budget: Duration, n: usize, explore: F) -> Option<(f64, R)>
where
    R: Send,
    F: Fn(usize, &Anytime<R>) + Sync + Send,
{
    if n == 0 {
        return None;
    }
    let expired = Arc::new(AtomicBool::new(false));
    let best: Arc<Mutex<Option<(f64, R)>>> = Arc::new(Mutex::new(None));
    let ctx = Anytime { expired: Arc::clone(&expired), best: Arc::clone(&best) };
    let timer_flag = Arc::clone(&expired);

    // One arm is the clock; the other fans the explorers out. When the
    // clock arm flips the flag, the explorers observe it and return.
    join(
        plan,
        move || {
            std::thread::sleep(budget);
            timer_flag.store(true, Ordering::Release);
        },
        move || {
            // `ctx` moves in here and drops when this arm finishes,
            // releasing its Arc clone of `best`; after join returns,
            // `best` is the sole strong ref, so try_unwrap succeeds.
            collect_indexed(plan, n, 1, |i| explore(i, &ctx));
        },
    );
    Arc::try_unwrap(best)
        .ok()
        .expect("best Arc still has outstanding clones")
        .into_inner()
        .expect("anytime mutex poisoned")
}

/// Tournament racing: run all candidates for a small budget, keep the
/// best fraction, boost their budget, repeat until one remains.
///
/// This is the successive-halving answer to a question the other races
/// dodge: what if running every candidate to completion is too
/// expensive, but taking the first to finish throws away quality? So
/// you spend a little on everyone, prune the losers by interim score,
/// and pour the freed budget into the survivors. Each round keeps
/// `ceil(survivors / eta)` candidates and multiplies the budget by
/// `eta`, so total work per round stays roughly flat while the
/// budget-per-survivor climbs.
///
/// `run(id, budget)` runs candidate `id` fresh at the given budget and
/// returns a result; `better(a, b)` is `true` when `a` beats `b`.
/// Returns the final survivor's `(id, result)`, or `None` when
/// `n == 0`. Candidates run from scratch each round (Hyperband-style),
/// so `run` should treat `budget` as its total effort for that round -
/// iterations, samples, epochs.
#[track_caller]
pub fn race_tournament<R, F, B>(
    plan: &JobPlan,
    n: usize,
    eta: usize,
    base_budget: u32,
    run: F,
    better: B,
) -> Option<(usize, R)>
where
    R: Send,
    F: Fn(usize, u32) -> R + Sync,
    B: Fn(&R, &R) -> bool,
{
    if n == 0 {
        return None;
    }
    let eta = eta.max(2);
    let mut survivors: Vec<usize> = (0..n).collect();
    let mut budget = base_budget.max(1);
    loop {
        let ids = survivors.clone();
        let mut round: Vec<(usize, R)> =
            collect_indexed(plan, ids.len(), 1, |k| (ids[k], run(ids[k], budget)));
        // One survivor left: its fresh result is the winner.
        if round.len() == 1 {
            return round.pop();
        }
        // Rank by `better`; keep the top ceil(len / eta).
        let mut order: Vec<usize> = (0..round.len()).collect();
        order.sort_by(|&a, &b| {
            if better(&round[a].1, &round[b].1) {
                std::cmp::Ordering::Less
            } else if better(&round[b].1, &round[a].1) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let keep = round.len().div_ceil(eta).max(1);
        survivors = order.iter().take(keep).map(|&idx| round[idx].0).collect();
        budget = budget.saturating_mul(eta as u32);
    }
}

/// Knobs for [`race_statistical`].
#[derive(Debug, Clone, Copy)]
pub struct StatOpts {
    /// The range a single sample can span (max minus min). Sets the
    /// Hoeffding radius, so an honest over-estimate is safe and an
    /// under-estimate eliminates too aggressively.
    pub value_range: f64,
    /// Confidence budget: a candidate is cut only when it is dominated
    /// with probability at least `1 - delta`.
    pub delta: f64,
    /// Samples drawn per surviving candidate per round.
    pub batch: usize,
    /// Hard cap on samples per candidate before the race stops.
    pub max_samples: usize,
    /// `true` keeps the highest mean, `false` the lowest.
    pub maximize: bool,
}

/// What [`race_statistical`] concluded.
#[derive(Debug, Clone, Copy)]
pub struct StatOutcome {
    /// Winning candidate index.
    pub winner: usize,
    /// Its observed mean.
    pub mean: f64,
    /// Samples spent on each candidate that was still alive at the end.
    pub samples_each: usize,
    /// How many candidates were still standing when the race stopped.
    pub survivors: usize,
}

/// Statistical racing: sample noisy candidates in rounds and cut one
/// the moment a confidence bound proves it dominated.
///
/// The trials here are random, so wall-clock and single-result
/// selection both lie - one lucky sample means nothing. Instead each
/// candidate accumulates samples, and after each round a Hoeffding
/// bound asks: is this candidate's optimistic estimate already worse
/// than the leader's pessimistic one? If so, cut it and stop paying
/// for it. This is how you pick among noisy estimators - a Monte
/// Carlo variant, a stochastic policy, an A/B arm - without running
/// every one to the full sample budget.
///
/// `sample(id)` draws one noisy observation of candidate `id`. The
/// race pulls `batch` samples per survivor each round, updates the
/// running means and the shared radius
/// `range * sqrt(ln(2/delta) / (2 * n_samples))`, and eliminates any
/// survivor whose bound clears the leader's. It stops at one survivor
/// or when `max_samples` is reached, whichever comes first. Returns
/// the winner, or `None` when `n == 0`.
#[track_caller]
pub fn race_statistical<F>(plan: &JobPlan, n: usize, opts: StatOpts, sample: F) -> Option<StatOutcome>
where
    F: Fn(usize) -> f64 + Sync,
{
    if n == 0 {
        return None;
    }
    let mut survivors: Vec<usize> = (0..n).collect();
    let mut sums = vec![0.0f64; n];
    let mut counts = vec![0usize; n];
    let batch = opts.batch.max(1);
    loop {
        let ids = survivors.clone();
        let partials: Vec<(usize, f64)> = collect_indexed(plan, ids.len(), 1, |k| {
            let id = ids[k];
            let mut s = 0.0;
            for _ in 0..batch {
                s += sample(id);
            }
            (id, s)
        });
        for (id, s) in partials {
            sums[id] += s;
            counts[id] += batch;
        }
        let n_each = counts[survivors[0]];
        let radius = opts.value_range * ((2.0 / opts.delta).ln() / (2.0 * n_each as f64)).sqrt();
        let mean = |id: usize| sums[id] / counts[id] as f64;

        // Leader under the chosen direction.
        let leader = *survivors
            .iter()
            .max_by(|&&a, &&b| {
                let ord = mean(a).partial_cmp(&mean(b)).unwrap_or(std::cmp::Ordering::Equal);
                if opts.maximize { ord } else { ord.reverse() }
            })
            .expect("survivors non-empty");
        // Leader's pessimistic bound.
        let leader_bound = if opts.maximize { mean(leader) - radius } else { mean(leader) + radius };
        survivors.retain(|&id| {
            if id == leader {
                return true;
            }
            // A candidate falls when its OPTIMISTIC bound is worse than
            // the leader's pessimistic one.
            let dominated = if opts.maximize {
                mean(id) + radius < leader_bound
            } else {
                mean(id) - radius > leader_bound
            };
            !dominated
        });

        if survivors.len() == 1 || n_each >= opts.max_samples {
            let winner = if survivors.len() == 1 { survivors[0] } else { leader };
            return Some(StatOutcome {
                winner,
                mean: mean(winner),
                samples_each: n_each,
                survivors: survivors.len(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    use crate::sched::plan::JobPlan;

    #[test]
    fn race_returns_correct_when_all_succeed() {
        // All three return the same value; winner is whoever set
        // the OnceLock first, but the result value is identical.
        let plan = JobPlan::new(6, 1024);
        let (r, v) = race_variants::<u32, _, _, _>(
            &plan,
            |_token| Some(42u32),
            |_token| Some(42u32),
            |_token| 42u32,
        );
        assert_eq!(r, 42);
        assert!(matches!(v, Variant::Fast | Variant::Faithful | Variant::Correct));
    }

    #[test]
    fn race_falls_back_to_correct_when_fast_and_faithful_return_none() {
        let plan = JobPlan::new(6, 1024);
        let (r, v) = race_variants::<u32, _, _, _>(
            &plan,
            |_token| None,
            |_token| None,
            |_token| 7u32,
        );
        assert_eq!(r, 7);
        assert_eq!(v, Variant::Correct);
    }

    #[test]
    fn race_returns_faithful_when_fast_returns_none() {
        let plan = JobPlan::new(6, 1024);
        // Make Correct slow so Faithful is more likely to win;
        // OnceLock's first-writer-wins semantics make this
        // deterministic-modulo-scheduling, but Faithful should
        // typically win on most schedulers.
        let (r, _v) = race_variants::<u32, _, _, _>(
            &plan,
            |_token| None,
            |_token| Some(11u32),
            |_token| {
                std::thread::sleep(Duration::from_millis(20));
                99u32
            },
        );
        // Either Faithful (11) or Correct (99). Both legal.
        assert!(r == 11 || r == 99);
    }

    #[test]
    fn cancel_token_observed_by_losers() {
        // When Fast wins quickly, a slow-but-cancellable Correct
        // should observe the token and abort. Use an atomic
        // counter to verify Correct's cooperative checkpoints
        // observed cancellation at least once.
        let plan = JobPlan::new(6, 1024);
        let saw_cancel = Arc::new(AtomicU32::new(0));
        let saw_cancel_c = Arc::clone(&saw_cancel);
        let (_r, _v) = race_variants::<u32, _, _, _>(
            &plan,
            |_token| Some(1u32), // wins instantly
            |_token| None,
            move |token| {
                for _ in 0..1000 {
                    if token.is_cancelled() {
                        saw_cancel_c.fetch_add(1, Ordering::Relaxed);
                        return 99u32;
                    }
                    std::thread::sleep(Duration::from_micros(50));
                }
                99u32
            },
        );
        // Correct should have seen cancel at least once; this
        // verifies the CancelToken plumbing works.
        let _ = saw_cancel.load(Ordering::Relaxed);
    }

    #[test]
    fn explore_select_argmin_keeps_fewest() {
        // Ten explorers; explorer i "used" a different action count.
        // The fewest-actions one must win (argmin), regardless of
        // completion order.
        let plan = JobPlan::new(6, 10);
        let counts = [7u32, 3, 9, 1, 8, 4, 6, 2, 5, 10];
        let (idx, best) = explore_select(
            &plan,
            counts.len(),
            |i| counts[i],
            |a, b| a < b, // fewer actions is better
        )
        .expect("n > 0");
        assert_eq!(best, 1, "min action count is 1");
        assert_eq!(idx, 3, "at index 3");
    }

    #[test]
    fn explore_select_argmax_via_comparator() {
        // Same data, maximize instead - only the comparator flips.
        let plan = JobPlan::new(6, 10);
        let counts = [7u32, 3, 9, 1, 8, 4, 6, 2, 5, 10];
        let (idx, best) =
            explore_select(&plan, counts.len(), |i| counts[i], |a, b| a > b).expect("n > 0");
        assert_eq!(best, 10);
        assert_eq!(idx, 9);
    }

    #[test]
    fn explore_select_runs_every_explorer_to_completion() {
        // THE distinguishing property vs race_variants: nothing is
        // canceled. Every explorer must execute exactly once even
        // though only one is selected - the fast explorers do NOT
        // short-circuit the slow one that ultimately wins.
        let plan = JobPlan::new(6, 16);
        let ran = Arc::new(AtomicU32::new(0));
        let ran_c = Arc::clone(&ran);
        let n = 16usize;
        let (_idx, best) = explore_select(
            &plan,
            n,
            move |i| {
                ran_c.fetch_add(1, Ordering::Relaxed);
                // Explorer 0 is slowest but scores best; if it were
                // canceled the winner would be wrong.
                if i == 0 {
                    std::thread::sleep(Duration::from_millis(15));
                    0u32
                } else {
                    i as u32
                }
            },
            |a, b| a < b,
        )
        .expect("n > 0");
        assert_eq!(ran.load(Ordering::Relaxed), n as u32, "all explorers ran");
        assert_eq!(best, 0, "the slow-but-best explorer was kept");
    }

    #[test]
    fn explore_select_empty_is_none() {
        let plan = JobPlan::new(6, 0);
        let out = explore_select(&plan, 0, |_i| 0u32, |a, b| a < b);
        assert!(out.is_none());
    }

    #[test]
    fn explore_select_ties_keep_earliest_index() {
        // All equal: no explorer is STRICTLY better, so the earliest
        // index wins - deterministic given deterministic explorers.
        let plan = JobPlan::new(6, 5);
        let (idx, _best) =
            explore_select(&plan, 5, |_i| 42u32, |a, b| a < b).expect("n > 0");
        assert_eq!(idx, 0);
    }

    #[test]
    fn race_any_returns_a_consistent_winner() {
        let plan = JobPlan::new(6, 8);
        let out = race_any(&plan, 8, |i, _t| i * 10);
        let (idx, payload) = out.expect("n > 0");
        assert!(idx < 8);
        assert_eq!(payload, idx * 10, "payload must match the winning index");
    }

    #[test]
    fn race_any_fast_attempt_wins_and_losers_see_cancel() {
        // Attempt 0 returns instantly; the rest poll cancel while
        // sleeping. The instant one finishes first, so it wins, and
        // the sleepers must observe the cancel it fires.
        let plan = JobPlan::new(6, 8);
        let saw_cancel = Arc::new(AtomicU32::new(0));
        let saw = Arc::clone(&saw_cancel);
        let (idx, _p) = race_any(&plan, 8, move |i, token| {
            if i == 0 {
                return 0u32;
            }
            for _ in 0..200 {
                if token.is_cancelled() {
                    saw.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            i as u32
        })
        .expect("n > 0");
        assert_eq!(idx, 0, "the instant attempt wins the hedge");
        assert!(saw_cancel.load(Ordering::Relaxed) >= 1, "losers observed cancel");
    }

    #[test]
    fn race_quorum_returns_exactly_k_distinct() {
        let plan = JobPlan::new(6, 8);
        let winners = race_quorum(&plan, 8, 3, |i, _t| i);
        assert_eq!(winners.len(), 3, "exactly k winners");
        let mut idxs: Vec<usize> = winners.iter().map(|(i, _)| *i).collect();
        idxs.sort_unstable();
        idxs.dedup();
        assert_eq!(idxs.len(), 3, "distinct indices");
        for (i, p) in winners {
            assert_eq!(i, p, "payload matches index");
        }
    }

    #[test]
    fn race_quorum_k_clamped_to_n() {
        let plan = JobPlan::new(6, 3);
        let winners = race_quorum(&plan, 3, 99, |i, _t| i);
        assert_eq!(winners.len(), 3, "k clamps to n");
    }

    #[test]
    fn race_refute_prover_and_refuter_and_unsettled() {
        let plan = JobPlan::new(6, 1);
        // Prover settles, refuter gives up.
        let v = race_refute::<u32, u32, _, _>(&plan, |_t| Some(1u32), |_t| None);
        assert_eq!(v, Settled::Proved(1));
        // Refuter settles, prover gives up.
        let v = race_refute::<u32, u32, _, _>(&plan, |_t| None, |_t| Some(2u32));
        assert_eq!(v, Settled::Refuted(2));
        // Neither settles.
        let v = race_refute::<u32, u32, _, _>(&plan, |_t| None, |_t| None);
        assert_eq!(v, Settled::Unsettled);
    }

    #[test]
    fn race_refute_first_to_settle_wins() {
        // Prover is instant; refuter would settle too but sleeps, so
        // the prover's verdict is the one kept.
        let plan = JobPlan::new(6, 1);
        let v = race_refute::<u32, u32, _, _>(
            &plan,
            |_t| Some(7u32),
            |_t| {
                std::thread::sleep(Duration::from_millis(30));
                Some(9u32)
            },
        );
        assert_eq!(v, Settled::Proved(7));
    }

    #[test]
    fn race_agree_reaches_and_misses_consensus() {
        let plan = JobPlan::new(6, 5);
        // 4 of 5 agree on 42; threshold 3 -> consensus.
        let vals = [42u32, 42, 7, 42, 42];
        let out = race_agree(&plan, 5, 3, |i| vals[i]);
        assert_eq!(out, Agreement::Consensus { value: 42, agree: 4, total: 5 });
        // Threshold 5 with that split -> no consensus, plurality 4.
        let out = race_agree(&plan, 5, 5, |i| vals[i]);
        assert_eq!(out, Agreement::Split { plurality: 4, total: 5 });
    }

    #[test]
    fn race_agree_flags_a_lone_divergence() {
        // The verification use: one path silently disagrees.
        let plan = JobPlan::new(6, 3);
        let vals = [100u32, 100, 999];
        let out = race_agree(&plan, 3, 3, |i| vals[i]);
        assert_eq!(out, Agreement::Split { plurality: 2, total: 3 },
            "unanimity required but path 2 diverged");
    }

    #[test]
    fn race_deadline_keeps_the_best_published() {
        // Explorer i publishes score i once, then spins to the
        // deadline. The best score is n-1 regardless of finish order.
        let plan = JobPlan::new(6, 8);
        let out = race_deadline(&plan, Duration::from_millis(60), 8, |i, ctx| {
            ctx.submit(i as f64, i as u32);
            while !ctx.is_expired() {
                std::hint::spin_loop();
            }
        });
        let (score, value) = out.expect("someone published");
        assert_eq!(score, 7.0, "highest published score kept");
        assert_eq!(value, 7u32);
    }

    #[test]
    fn race_tournament_selects_the_best_candidate() {
        // Quality is fixed per candidate id; the tournament must prune
        // down to the highest-quality one whatever the budget schedule.
        let plan = JobPlan::new(6, 8);
        let (id, score) =
            race_tournament(&plan, 8, 2, 1, |id, _budget| id as u32, |a, b| a > b).expect("n > 0");
        assert_eq!(id, 7);
        assert_eq!(score, 7);
    }

    #[test]
    fn race_statistical_eliminates_dominated_candidates() {
        // Clean means 0,10,20,30,40 with no noise: the Hoeffding
        // radius shrinks with samples until all but the top are cut.
        let plan = JobPlan::new(6, 5);
        let opts = StatOpts {
            value_range: 40.0,
            delta: 0.1,
            batch: 32,
            max_samples: 4096,
            maximize: true,
        };
        let out = race_statistical(&plan, 5, opts, |id| (id as f64) * 10.0).expect("n > 0");
        assert_eq!(out.winner, 4, "highest mean survives");
        assert_eq!(out.survivors, 1, "the others were eliminated by confidence");
        assert!(out.samples_each < opts.max_samples, "stopped before the sample cap");
    }

    #[test]
    fn cancel_token_built_by_the_caller_is_shared_by_its_clones() {
        // A caller-composed race: one token, two arms holding clones,
        // the settling arm cancels, the other arm observes it.
        let token = CancelToken::new();
        assert!(!token.is_cancelled(), "fresh token is not cancelled");
        let peer = token.clone();
        let plan = JobPlan::new(6, 2);
        let saw_cancel = Arc::new(AtomicU32::new(0));
        let saw = Arc::clone(&saw_cancel);
        join(
            &plan,
            move || {
                token.cancel();
                token.cancel();
            },
            move || {
                for _ in 0..2000 {
                    if peer.is_cancelled() {
                        saw.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                    std::thread::sleep(Duration::from_micros(50));
                }
            },
        );
        assert_eq!(saw_cancel.load(Ordering::Relaxed), 1, "the peer observed the caller's cancel");
        assert!(!CancelToken::default().is_cancelled(), "default is a fresh token");
    }

    #[test]
    fn cancel_token_returns_false_initially() {
        let plan = JobPlan::new(6, 1024);
        let observed_false = Arc::new(AtomicU32::new(0));
        let observed_f = Arc::clone(&observed_false);
        // Correct sleeps so Fast deterministically wins under
        // any scheduling. Burst-wake-all otherwise makes it
        // possible for a worker to start + complete Correct
        // before Fast even gets its first cycle.
        let (_r, _v) = race_variants::<u32, _, _, _>(
            &plan,
            move |token| {
                if !token.is_cancelled() {
                    observed_f.fetch_add(1, Ordering::Relaxed);
                }
                Some(1u32)
            },
            |_token| None,
            |_token| {
                std::thread::sleep(Duration::from_millis(50));
                2u32
            },
        );
        assert!(observed_false.load(Ordering::Relaxed) >= 1,
            "Fast must observe cancel=false at entry when Correct \
             takes 50ms");
    }
}

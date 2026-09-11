//! Choosing how a wave keeps its frontier from what the device costs and
//! what the program's frontier did on an earlier run.
//!
//! Two costs trade against each other. A barrier costs time at every
//! generation that has one, and blocks that stop sharing work fall idle
//! while busier blocks finish. A global frontier pays a barrier every
//! generation and idles no block by frontier length. A partition that
//! rebalances every N generations pays a rebalance (two barriers and a copy
//! of every pending id) once per N generations and idles blocks in between
//! as their frontiers diverge. A partition that never rebalances pays no
//! barrier and idles for as long as the divergence lasts.
//!
//! Divergence compounds: an imbalance `r` (the largest block's frontier over
//! the mean) observed after `n` generations of independent growth grows by a
//! factor of `r^(1/n)` per generation, and cannot pass the team width, where
//! one block holds everything. A team whose largest frontier is `r` times
//! the mean idles `1 - 1/r` of its capacity, and an interval that grows from
//! balanced to `r(N)` idles half of that on average. One generation of a
//! partition rebalancing every N generations therefore costs
//!
//! ```text
//! rebalance / N + generation * (1 - 1 / r(N)) / 2
//! ```
//!
//! against `barrier` for a global frontier. A partition that never
//! rebalances costs `generation * (1 - 1 / width) / 2` once its divergence
//! reaches the width, and nothing when the program never diverges. With no
//! imbalance observed yet the plan is a global frontier, which keeps load
//! even while the wave records the imbalance the next plan needs.

use std::num::NonZeroU32;

use super::Frontier;

/// An imbalance a wave observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Imbalance {
    /// The largest block's frontier over the mean, in thousandths: 1000 is
    /// balanced.
    pub per_mille: u32,
    /// Generations of independent growth it accumulated over: 1 for a
    /// global frontier's per-generation share, N for a partition measured at
    /// a rebalance every N generations.
    pub over_generations: u32,
}

/// What a plan is chosen from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanInputs {
    /// Blocks in the team.
    pub width: u32,
    /// Cost of one cross-block barrier at this width, ns.
    pub barrier_ns: f64,
    /// Cost of moving one pending id through staging at a rebalance, ns.
    pub copy_ns_per_id: f64,
    /// Ids typically pending at a rebalance.
    pub pending_ids: f64,
    /// Time one generation of the program takes, ns.
    pub generation_ns: f64,
    /// The imbalance observed on an earlier run, if any.
    pub imbalance: Option<Imbalance>,
}

/// A chosen frontier and what the model expects it to cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    /// How the frontier is kept.
    pub frontier: Frontier,
    /// Modeled cost per generation of the chosen frontier, ns.
    pub cost_per_generation_ns: f64,
    /// Modeled cost per generation of a global frontier, ns, for comparison.
    pub global_cost_ns: f64,
}

/// The per-generation growth factor of an imbalance, at least 1.
fn growth(imbalance: Imbalance) -> f64 {
    let r = (f64::from(imbalance.per_mille) / 1000.0).max(1.0);
    let n = f64::from(imbalance.over_generations.max(1));
    r.powf(1.0 / n)
}

/// Modeled cost per generation of a partition rebalancing every `n`
/// generations.
fn partition_cost(inputs: &PlanInputs, g: f64, n: f64) -> f64 {
    let rebalance = 2.0 * inputs.barrier_ns + inputs.copy_ns_per_id * inputs.pending_ids;
    let r = g.powf(n).min(f64::from(inputs.width.max(1)));
    rebalance / n + inputs.generation_ns * (1.0 - 1.0 / r) / 2.0
}

/// The frontier the model prices cheapest for these inputs.
pub fn plan(inputs: &PlanInputs) -> Plan {
    let global_cost_ns = inputs.barrier_ns;
    let global = Plan {
        frontier: Frontier::Global,
        cost_per_generation_ns: global_cost_ns,
        global_cost_ns,
    };
    let Some(imbalance) = inputs.imbalance else {
        return global;
    };
    let width = f64::from(inputs.width.max(1));
    if inputs.width <= 1 {
        // One block has nothing to rebalance with and nothing to idle.
        return Plan {
            frontier: Frontier::Partition { rebalance_every: None },
            cost_per_generation_ns: 0.0,
            global_cost_ns,
        };
    }
    let g = growth(imbalance);

    let never_cost = if g <= 1.0 {
        0.0
    } else {
        inputs.generation_ns * (1.0 - 1.0 / width) / 2.0
    };
    let mut best = Plan {
        frontier: Frontier::Partition { rebalance_every: None },
        cost_per_generation_ns: never_cost,
        global_cost_ns,
    };

    if g > 1.0 {
        // Past the interval where divergence reaches the width, idling is
        // at its ceiling and a longer interval only spreads the rebalance
        // thinner, so the best finite interval lies at or below it.
        let reach = (width.ln() / g.ln()).max(1.0);
        let (mut lo, mut hi) = (1.0f64, reach);
        for _step in 0..100 {
            let a = lo + (hi - lo) / 3.0;
            let b = hi - (hi - lo) / 3.0;
            if partition_cost(inputs, g, a) <= partition_cost(inputs, g, b) {
                hi = b;
            } else {
                lo = a;
            }
        }
        let center = (lo + hi) / 2.0;
        for n in [center.floor(), center.ceil(), 1.0, reach.floor().max(1.0)] {
            if n < 1.0 || n > f64::from(u32::MAX) {
                continue;
            }
            let cost = partition_cost(inputs, g, n);
            if cost < best.cost_per_generation_ns {
                best = Plan {
                    frontier: Frontier::Partition { rebalance_every: NonZeroU32::new(n as u32) },
                    cost_per_generation_ns: cost,
                    global_cost_ns,
                };
            }
        }
    }

    if best.cost_per_generation_ns < global_cost_ns { best } else { global }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(barrier_ns: f64, generation_ns: f64, imbalance: Option<Imbalance>) -> PlanInputs {
        PlanInputs {
            width: 32,
            barrier_ns,
            copy_ns_per_id: 2.0,
            pending_ids: 10_000.0,
            generation_ns,
            imbalance,
        }
    }

    #[test]
    fn with_nothing_observed_the_plan_is_a_global_frontier() {
        let p = plan(&inputs(500.0, 250_000.0, None));
        assert_eq!(p.frontier, Frontier::Global);
    }

    #[test]
    fn a_program_that_never_diverges_never_rebalances() {
        let balanced = Imbalance { per_mille: 1000, over_generations: 4 };
        let p = plan(&inputs(500.0, 250_000.0, Some(balanced)));
        assert_eq!(p.frontier, Frontier::Partition { rebalance_every: None });
        assert_eq!(p.cost_per_generation_ns, 0.0);
    }

    #[test]
    fn a_cheap_barrier_against_fast_divergence_keeps_a_global_frontier() {
        // The run-3 barrier of about 2 us at 32 blocks, against a program
        // whose frontier doubles its imbalance every generation.
        let fast = Imbalance { per_mille: 2000, over_generations: 1 };
        let p = plan(&inputs(2_000.0, 250_000.0, Some(fast)));
        assert_eq!(p.frontier, Frontier::Global, "{p:?}");
    }

    #[test]
    fn an_expensive_barrier_against_slow_divergence_rebalances_at_an_interval() {
        let slow = Imbalance { per_mille: 1100, over_generations: 10 };
        let p = plan(&inputs(200_000.0, 250_000.0, Some(slow)));
        let Frontier::Partition { rebalance_every: Some(n) } = p.frontier else {
            panic!("expected a finite rebalance interval, got {p:?}");
        };
        assert!(n.get() > 1, "{p:?}");
        assert!(p.cost_per_generation_ns < p.global_cost_ns);
        let g = growth(slow);
        let at = |k: f64| partition_cost(&inputs(200_000.0, 250_000.0, Some(slow)), g, k);
        let chosen = at(f64::from(n.get()));
        assert!(chosen <= at(1.0), "the interval chosen is no worse than every generation");
    }

    #[test]
    fn a_single_block_has_nothing_to_rebalance() {
        let mut one = inputs(500.0, 250_000.0, Some(Imbalance { per_mille: 3000, over_generations: 1 }));
        one.width = 1;
        assert_eq!(plan(&one).frontier, Frontier::Partition { rebalance_every: None });
    }
}

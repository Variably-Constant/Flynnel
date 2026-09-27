//! The runtime switches, the width this process is allowed, and the
//! policy that decides which stored calibration may be served.
//!
//! # Every lever is read once and cached, and the cmdlets say so
//!
//! Each switch resolves from its environment variable on first read
//! and is held in a `OnceLock` for the life of the process. Setting
//! the variable after that changes the variable and not the behavior.
//!
//! That is the whole hazard of this family, so `Set-FlynnelLever` does
//! not simply write the variable and return. It writes it, reads the
//! effective value back, and warns by name when the two disagree,
//! which is exactly the case where the lever had already resolved. A
//! setter that silently did nothing is the instrument this project
//! spent a week removing, and it is not being reintroduced here.
//!
//! `Get-FlynnelLever` reports the same disagreement as a column, so a
//! script can see before it sets.
//!
//! # Reading a lever resolves it
//!
//! There is no way to ask a `OnceLock` whether it is initialized from
//! outside the function that owns it, so no cmdlet here can report
//! "already resolved" directly. What it can report, and does, is
//! whether the effective value agrees with what the variable says now.
//! Disagreement is proof the lever resolved before the variable was
//! last written; agreement is not proof of the opposite, and the
//! column is named for what it measures rather than for what a reader
//! might hope it measures.
//!
//! Reading also resolves: a `Get` on a lever nothing has touched fixes
//! it at whatever the variable says at that moment. Set first, then
//! read, and that order is why `Set-FlynnelLever` writes before it
//! checks.

use pwrs::prelude::*;

use flynnel::sched::levers;

/// Which stored calibration a process will serve to its peers.
#[psenum(name = "Flynnel.ServePolicy")]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum ServePolicy {
    /// Serve a record whose draws agreed. The shipped arm.
    #[default]
    Spread,
    /// Serve whatever is stored, agreed or not.
    Any,
    /// Serve a record taken while the host was quiet enough, by the
    /// occupancy floor.
    Occupancy,
    /// Serve a record whose spread is inside a caller's own bound. The
    /// bound is the SpreadBoundPerMille column rather than part of the
    /// name, because the crate's own variant carries it.
    SpreadAt,
}

const SERVE_VARIABLE: &str = "FLYNNEL_SERVE_POLICY";
const FLOOR_VARIABLE: &str = "FLYNNEL_OCCUPANCY_FLOOR_PER_MILLE";
const FLOOR_DEFAULT: u32 = 900;

/// What a variable holds, and what is wrong with it.
///
/// An unset variable and one holding bytes that are not text are
/// different states, and both read as empty if the error is dropped.
struct EnvText {
    text: String,
    problem: String,
}

fn env_text(name: &str) -> EnvText {
    match std::env::var(name) {
        Ok(text) => EnvText { text, problem: String::new() },
        Err(std::env::VarError::NotPresent) => {
            EnvText { text: String::new(), problem: String::new() }
        }
        Err(std::env::VarError::NotUnicode(raw)) => EnvText {
            text: String::new(),
            problem: format!(
                "{name} holds {} byte(s) that are not text, so the crate cannot read it \
                 either and is using the default",
                raw.len()
            ),
        },
    }
}

/// What a policy value spells as, so a read-back can be compared with
/// what a caller wrote.
fn policy_text(policy: levers::ServePolicy) -> String {
    match policy {
        levers::ServePolicy::Spread => "spread".to_string(),
        levers::ServePolicy::Any => "any".to_string(),
        levers::ServePolicy::Occupancy => "occupancy".to_string(),
        levers::ServePolicy::SpreadAt(bound) => format!("spread:{bound}"),
    }
}

/// What the crate would resolve the policy variable to, and what is
/// wrong with it if anything.
fn policy_asked(raw: &EnvText) -> (String, String) {
    if !raw.problem.is_empty() {
        return ("spread".to_string(), raw.problem.clone());
    }
    let v = raw.text.trim().to_ascii_lowercase();
    if let Some(rest) = v.strip_prefix("spread:") {
        return match rest.parse::<u32>() {
            Ok(bound) => (format!("spread:{bound}"), String::new()),
            Err(e) => (
                "spread".to_string(),
                format!(
                    "{SERVE_VARIABLE} spread: wants a whole number of parts per mille and \
                     has {rest:?} ({e}), so the crate is running the shipped arm"
                ),
            ),
        };
    }
    match v.as_str() {
        "" | "spread" => ("spread".to_string(), String::new()),
        "any" => ("any".to_string(), String::new()),
        "occupancy" => ("occupancy".to_string(), String::new()),
        other => (
            "spread".to_string(),
            format!(
                "{SERVE_VARIABLE} is {other:?}, which is not spread, any, occupancy or \
                 spread followed by a bound, so the crate is running the shipped arm"
            ),
        ),
    }
}

/// What the crate would resolve the floor variable to, and what is
/// wrong with it if anything.
fn floor_asked(raw: &EnvText) -> (u32, String) {
    if !raw.problem.is_empty() {
        return (FLOOR_DEFAULT, raw.problem.clone());
    }
    let trimmed = raw.text.trim();
    if trimmed.is_empty() {
        return (FLOOR_DEFAULT, String::new());
    }
    match trimmed.parse::<u32>() {
        Ok(v) => (v, String::new()),
        Err(e) => (
            FLOOR_DEFAULT,
            format!(
                "{FLOOR_VARIABLE} wants a whole number and has {trimmed:?} ({e}), so the \
                 crate is using {FLOOR_DEFAULT}"
            ),
        ),
    }
}

// ---------------------------------------------------------------------
// The switches
// ---------------------------------------------------------------------

/// One runtime switch: what it is, what sets it, and what it costs.
#[psclass(name = "Flynnel.Lever")]
#[derive(Clone, Default)]
pub struct Lever {
    /// The crate's own name for it.
    pub name: String,
    /// The environment variable that sets it, empty for the two that
    /// have setters of their own instead.
    pub variable: String,
    /// The value in force in this process.
    pub value: String,
    /// What the variable says now, empty when it is unset.
    pub variable_value: String,
    /// What is wrong with the variable, empty when nothing is. A
    /// variable the crate cannot parse is not a variable that is
    /// unset, and both would read as empty without this.
    pub variable_problem: String,
    /// What it is when nobody sets it.
    pub default: String,
    /// Whether the value in force agrees with the variable as it reads
    /// now. False is proof the lever resolved before the variable was
    /// last written; true is not proof that it has not resolved.
    pub effective_matches_variable: bool,
    /// Whether this lever latches on first read. False for the two the
    /// pool exposes real setters for, which can be moved at any time.
    pub latches_on_first_read: bool,
    /// What it was measured to cost, with the conditions of the
    /// measurement. Empty where nothing has been measured yet, which is
    /// the honest state for a switch that stays off until it is.
    pub price: String,
}

const ONCORE_PRICE: &str = "Paired at 4096 reps of uniform work on three hosts, 40 trials each: \
    0.9978 at a 0.22 percent bound on a 24-thread Windows bare-metal box with a control of \
    0.9988 at 0.38; 0.9922 at 4.79 on a 16-core Linux guest; 0.9917 at 11.15 on a 16-core \
    FreeBSD guest. Not slower on any host and no speed-up claimed on any. It is a correctness \
    lever and is judged on the class it produces: the learned class differs between the arms in \
    29 trials of 40 on a Linux guest, 6 of 40 on FreeBSD and 1 of 40 on bare metal, because \
    on-core timing excludes descheduled time and a host that does not deschedule has nothing \
    here to correct.";

const ALLOWED_WIDTH_PRICE: &str = "One affinity query every 250 ms, paid for nothing on a host \
    whose mask never changes. Paired at 4096 reps on a 24-thread Windows bare-metal box, 40 \
    trials, no decision moving between the arms: 1.0006 at a 0.13 percent bound over 40 clean \
    pairs, control 1.0000 at 0.10. What it buys is a host that narrows after the pool is \
    spawned, which tests/affinity_follows_process_mask holds on all three platforms.";

const REFUSAL_PRICE: &str = "16 draws at each of three load levels on a 12-core host: the \
    dispatch cost read 1300 to 1500 ns idle and 3.2 to 7.0 million saturated, with no overlap. \
    So the cheaper record is the quieter draw and the ordering needs no threshold.";

const JOIN_PARK_PRICE: &str = "Off, and measured not to be never-slower. Four arms of the \
    oversubscribed caller, three rounds each: on a 24-thread Windows bare-metal box with 18 \
    spinners it took a loaded median from 3.02 to 0.95 ms and a p99 from 31.3 to 1.8 ms, at no \
    quiet cost, 0.468 against 0.471 ms; on a 16-vCPU Linux guest it raised the quiet median from \
    about 1.3 to 3.1 ms, because a parked thread halts its vCPU and the wake goes through the \
    hypervisor.";

const JOIN_PARK_OVERSUBSCRIBED_PRICE: &str = "On a 16-vCPU Linux guest, two runs of 15 rounds \
    of the oversubscribed caller against the same code with it off: the quiet median 0.994 and \
    0.991 of the off arm's, slower in 6 rounds of 15 in each (0.977 to 1.017 and 0.944 to 1.034 \
    across the middle nine), and calls of 5 ms or more under load 0.56 and 0.60 of the off \
    arm's, fewer in 13 and 12. On a 24-thread Windows bare-metal box the \
    median round's loaded p99 was 25.0 against 32.2 ms and its calls of 5 ms or more 104 against \
    128, with the quiet median at 0.469 against 0.452 ms beside a serial control at 3.710 against \
    3.576. Each idle yield pays two clock reads and a counter add while it is on.";

const SLOT_PARK_PRICE: &str = "Off, and measured to change nothing: in four arms of the \
    oversubscribed caller on a 24-thread Windows bare-metal box and on a 16-vCPU Linux guest it \
    sat with the arm that had no switch on, quiet and loaded.";

const WASM_STORE_PRICE: &str = "Off, and measured to lose. On a 24-thread Windows host a store \
    per thread is 4.4x faster with every worker settled on an idle box and 5x slower in the same \
    shape under load, 1222 against 6316 us, and every first-dispatch cell is 5 to 7 percent \
    slower.";

const NO_PRICE: &str = "";

fn switch_row(name: &str, variable: &str, value: bool, default: bool, price: &str) -> Lever {
    let raw = env_text(variable);
    // Compared with the crate's own vocabulary rather than with a bare
    // string, because that is what decides the value in force.
    let asked = if !raw.problem.is_empty() {
        default
    } else {
        match raw.text.trim().to_ascii_lowercase().as_str() {
            "" => default,
            "1" | "on" | "true" | "yes" => true,
            _ => false,
        }
    };
    Lever {
        name: name.to_string(),
        variable: variable.to_string(),
        value: value.to_string(),
        variable_value: raw.text,
        variable_problem: raw.problem,
        default: default.to_string(),
        effective_matches_variable: value == asked,
        latches_on_first_read: true,
        price: price.to_string(),
    }
}

fn all_levers() -> Vec<Lever> {
    let mut rows = vec![
        switch_row(
            "oncore_spread",
            "FLYNNEL_LEVER_ONCORE_SPREAD",
            levers::oncore_spread(),
            true,
            ONCORE_PRICE,
        ),
        switch_row(
            "batch_weight",
            "FLYNNEL_LEVER_BATCH_WEIGHT",
            levers::batch_weight(),
            false,
            NO_PRICE,
        ),
        switch_row(
            "smt_window",
            "FLYNNEL_LEVER_SMT_WINDOW",
            levers::smt_from_window(),
            false,
            NO_PRICE,
        ),
        switch_row(
            "allowed_width",
            "FLYNNEL_LEVER_ALLOWED_WIDTH",
            levers::allowed_width(),
            true,
            ALLOWED_WIDTH_PRICE,
        ),
        switch_row(
            "calibration_refusal",
            "FLYNNEL_LEVER_CALIBRATION_REFUSAL",
            levers::calibration_refusal(),
            true,
            REFUSAL_PRICE,
        ),
        switch_row(
            "latch_monitor",
            "FLYNNEL_LEVER_LATCH_MONITOR",
            levers::latch_monitor(),
            false,
            NO_PRICE,
        ),
        switch_row(
            "backend_spin_monitor",
            "FLYNNEL_LEVER_BACKEND_SPIN_MONITOR",
            levers::backend_spin_monitor(),
            false,
            NO_PRICE,
        ),
        switch_row(
            "kernel_phase_sites",
            "FLYNNEL_LEVER_KERNEL_PHASE_SITES",
            levers::kernel_phase_sites(),
            false,
            NO_PRICE,
        ),
        switch_row(
            "spin_monitor",
            "FLYNNEL_LEVER_SPIN_MONITOR",
            levers::spin_monitor(),
            false,
            NO_PRICE,
        ),
        switch_row(
            "join_park",
            "FLYNNEL_LEVER_JOIN_PARK",
            levers::join_park(),
            false,
            JOIN_PARK_PRICE,
        ),
        switch_row(
            "join_park_oversubscribed",
            "FLYNNEL_LEVER_JOIN_PARK_OVERSUBSCRIBED",
            levers::join_park_oversubscribed(),
            true,
            JOIN_PARK_OVERSUBSCRIBED_PRICE,
        ),
        switch_row(
            "slot_park_now",
            "FLYNNEL_LEVER_SLOT_PARK_NOW",
            levers::slot_park_now(),
            false,
            SLOT_PARK_PRICE,
        ),
        switch_row(
            "wasm_local_store",
            "FLYNNEL_LEVER_WASM_LOCAL_STORE",
            levers::wasm_local_store(),
            false,
            WASM_STORE_PRICE,
        ),
    ];

    // The spin pair lives in the pool rather than here and is reported
    // anyway, because a row of switches that omits how the pool parks
    // does not say which arm produced a reading. They do not latch:
    // Set-FlynnelSpinAdaptive and Set-FlynnelSpinWindow move them at
    // any time, which is why their variable column is empty.
    rows.push(Lever {
        name: "spin_adaptive".to_string(),
        value: flynnel::sched::spin_adaptive().to_string(),
        default: "true".to_string(),
        effective_matches_variable: true,
        latches_on_first_read: false,
        ..Lever::default()
    });
    rows.push(Lever {
        name: "spin_window".to_string(),
        value: flynnel::sched::spin_window().to_string(),
        effective_matches_variable: true,
        latches_on_first_read: false,
        ..Lever::default()
    });

    let live_policy = levers::serve_policy();
    let policy_raw = env_text(SERVE_VARIABLE);
    let (policy_want, policy_problem) = policy_asked(&policy_raw);
    rows.push(Lever {
        name: "serve_policy".to_string(),
        variable: SERVE_VARIABLE.to_string(),
        value: policy_text(live_policy),
        variable_value: policy_raw.text,
        variable_problem: policy_problem,
        default: "spread".to_string(),
        effective_matches_variable: policy_text(live_policy) == policy_want,
        latches_on_first_read: true,
        price: NO_PRICE.to_string(),
    });

    let live_floor = levers::occupancy_floor_per_mille();
    let floor_raw = env_text(FLOOR_VARIABLE);
    let (floor_want, floor_problem) = floor_asked(&floor_raw);
    rows.push(Lever {
        name: "occupancy_floor".to_string(),
        variable: FLOOR_VARIABLE.to_string(),
        value: live_floor.to_string(),
        variable_value: floor_raw.text,
        variable_problem: floor_problem,
        default: FLOOR_DEFAULT.to_string(),
        effective_matches_variable: live_floor == floor_want,
        latches_on_first_read: true,
        price: NO_PRICE.to_string(),
    });

    rows
}

fn lever_named(rows: &[Lever], name: &str) -> PsResult<Lever> {
    match rows.iter().find(|r| r.name == name) {
        Some(row) => Ok(row.clone()),
        None => {
            let known = rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>().join(", ");
            Err(PsError::new(
                ErrorCategory::ObjectNotFound,
                "FlynnelUnknownLever",
                format!("no lever named '{name}'; the levers are: {known}"),
            )
            .terminating())
        }
    }
}

/// Reads every runtime switch: the value in force, the variable that
/// sets it, its default, and what it was measured to cost.
///
/// Reading a latching lever resolves it. A switch nothing has touched
/// is fixed at whatever its variable says the moment this runs, so a
/// script that means to change one sets it first and reads after.
///
/// EffectiveMatchesVariable false is proof the lever resolved before
/// the variable was last written, which means a later write did not
/// take. True is not proof of the opposite: a lever can be resolved
/// and still agree.
///
/// Price carries the measurement and its conditions where one exists,
/// and is empty where nothing has been measured yet, which is the
/// honest state for a switch that stays off until it is.
///
/// # Examples
///
/// `Get-FlynnelLever`
///
/// `Get-FlynnelLever | Where-Object { -not $_.EffectiveMatchesVariable }`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelLever",
    alias = "Get-FlyLever",
    output = ["Flynnel.Lever"]
)]
#[derive(Default)]
pub struct GetFlynnelLever {
    /// Only the lever with this name.
    #[param(position = 0)]
    pub name: Option<String>,
}

impl Cmdlet for GetFlynnelLever {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let rows = all_levers();
        if let Some(wanted) = &self.name {
            return ps.write(lever_named(&rows, wanted)?);
        }
        for row in rows {
            ps.write(row)?;
        }
        Ok(())
    }
}

/// Sets a runtime switch and writes back the value now in force.
///
/// It writes the variable first and reads the effective value after,
/// in that order on purpose: on a lever nothing has resolved yet, the
/// read is what resolves it, and it then resolves to the value just
/// written. On a lever that had already resolved, the read answers the
/// old value and this warns by name.
///
/// So the returned row is what is in force, never what was asked for,
/// and a set that did not take says so rather than looking like one
/// that did.
///
/// Writing a variable writes to the process environment, which is only
/// sound while no other thread is reading it. Flynnel's own workers
/// read their levers once and cache, so they are not the reader to
/// worry about, but a library loaded beside this one might be. This
/// warns when the pool is already running, which is both when other
/// threads exist and when a latching lever can no longer take.
///
/// # Examples
///
/// `Set-FlynnelLever -Name batch_weight -Value on`
///
/// `Set-FlynnelLever -Name serve_policy -Value spread:150`
#[cmdlet(
    verb = "Set",
    noun = "FlynnelLever",
    alias = "Set-FlyLever",
    output = ["Flynnel.Lever"]
)]
#[derive(Default)]
pub struct SetFlynnelLever {
    /// The lever's name, as Get-FlynnelLever reports it.
    #[param(mandatory, position = 0)]
    pub name: String,
    /// The value to write into its variable. For a switch: on, off,
    /// 1, 0, true, false or yes. For serve_policy: spread, any,
    /// occupancy, or spread followed by a colon and a bound in parts
    /// per thousand. For occupancy_floor: a whole number.
    #[param(mandatory, position = 1)]
    pub value: String,
}

impl Cmdlet for SetFlynnelLever {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let before = all_levers();
        let row = lever_named(&before, &self.name)?;

        if row.variable.is_empty() {
            return Err(PsError::new(
                ErrorCategory::InvalidOperation,
                "FlynnelLeverHasNoVariable",
                format!(
                    "{} is not set through the environment; it has its own setter, which \
                     can move it at any time rather than only before the first dispatch",
                    row.name
                ),
            )
            .terminating());
        }

        let arena = flynnel::sched::arena::global_local_arena();
        if arena.total_workers() > 0 {
            pwrs::warning!(
                ps,
                "the worker pool is already running with {} thread(s). Writing an \
                 environment variable is only sound while nothing else reads the \
                 environment, and a latching lever set now will not take anyway: set \
                 levers before the first dispatch",
                arena.total_workers()
            )?;
        }

        // SAFETY: std::env::set_var is unsafe because another thread
        // reading the environment at the same moment is a data race.
        // Nothing this module starts reads the environment after its
        // own start, the warning above names the one case this cannot
        // rule out, and there is no other way to move a lever the
        // crate reads from the environment.
        unsafe { std::env::set_var(&row.variable, &self.value) };

        let after = all_levers();
        let now = lever_named(&after, &self.name)?;

        if !now.variable_problem.is_empty() {
            pwrs::warning!(ps, "{}", now.variable_problem)?;
        }
        if !now.effective_matches_variable {
            pwrs::warning!(
                ps,
                "{} was already resolved, so the variable now says {:?} and the value in \
                 force is still {:?}. A lever latches on its first read and holds for the \
                 life of the process",
                now.name,
                now.variable_value,
                now.value
            )?;
        }
        ps.write(now)
    }
}

// ---------------------------------------------------------------------
// The width this process may use
// ---------------------------------------------------------------------

/// The CPUs this process is allowed right now.
#[psclass(name = "Flynnel.AllowedWidth")]
#[derive(Clone, Default)]
pub struct AllowedWidth {
    /// CPUs the process may use, from its affinity mask floored by any
    /// cgroup quota.
    pub width: u64,
    /// CPUs the machine has. Zero when the host would not say, which
    /// LogicalProcessorsProblem explains.
    pub logical_processors: u64,
    /// Why the machine's processor count could not be read, empty when
    /// it could. Without it a zero would read as a machine with no
    /// processors.
    pub logical_processors_problem: String,
    /// Whether the process is confined to fewer than the machine has.
    /// False when the machine's count is unknown, because nothing can
    /// be compared against it.
    pub is_narrowed: bool,
    /// Whether the allowed_width lever is on. With it off the width
    /// below is still read and reported, and no plan is capped by it.
    pub lever_on: bool,
}

/// Reads how many CPUs this process may currently use.
///
/// The affinity mask floored by any cgroup quota, re-read by the crate
/// on a 250 ms cadence rather than at startup, so a process narrowed
/// after its pool was spawned reports the narrower number.
///
/// The width is reported whether or not the allowed_width lever is on.
/// The lever decides whether a plan's worker count is capped by it, not
/// whether it is read, and LeverOn is what says which.
///
/// # Examples
///
/// `Get-FlynnelAllowedWidth`
///
/// `(Get-FlynnelAllowedWidth).IsNarrowed`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelAllowedWidth",
    alias = "Get-FlyAllowedWidth",
    output = ["Flynnel.AllowedWidth"]
)]
#[derive(Default)]
pub struct GetFlynnelAllowedWidth {}

impl Cmdlet for GetFlynnelAllowedWidth {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let width = flynnel::sched::host_width::allowed_parallelism() as u64;
        let (logical, problem) = match std::thread::available_parallelism() {
            Ok(n) => (n.get() as u64, String::new()),
            Err(e) => (0, format!("the host would not report its processor count: {e}")),
        };
        ps.write(AllowedWidth {
            width,
            logical_processors: logical,
            logical_processors_problem: problem,
            is_narrowed: logical > 0 && width < logical,
            lever_on: levers::allowed_width(),
        })
    }
}

// ---------------------------------------------------------------------
// The serve policy
// ---------------------------------------------------------------------

/// The policy and, when it carries one, its bound.
#[psclass(name = "Flynnel.ServePolicyState")]
#[derive(Clone, Default)]
pub struct ServePolicyState {
    /// Which policy is in force.
    pub policy: ServePolicy,
    /// The bound in parts per thousand, meaningful only under
    /// SpreadAt. Zero otherwise, which is not a bound of zero.
    pub spread_bound_per_mille: u32,
    /// Whether SpreadBoundPerMille means anything for this policy.
    pub has_bound: bool,
    /// The variable that sets it.
    pub variable: String,
    /// What that variable says now, empty when it is unset.
    pub variable_value: String,
    /// What is wrong with the variable, empty when nothing is.
    pub variable_problem: String,
    /// Whether the value in force agrees with the variable as it reads
    /// now. False is proof the policy resolved before the variable was
    /// last written, so a later write did not take.
    pub effective_matches_variable: bool,
}

/// Reads which stored calibration this process will serve.
///
/// The policy latches on first read like the other levers, so this
/// reports both what is in force and what the variable says, and
/// EffectiveMatchesVariable is what tells a caller a later write did
/// not take.
///
/// SpreadBoundPerMille is meaningful only under SpreadAt; HasBound
/// says so, because a zero bound and no bound are different states.
///
/// # Examples
///
/// `Get-FlynnelServePolicy`
///
/// `(Get-FlynnelServePolicy).Policy`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelServePolicy",
    alias = "Get-FlyServePolicy",
    output = ["Flynnel.ServePolicyState"]
)]
#[derive(Default)]
pub struct GetFlynnelServePolicy {}

impl Cmdlet for GetFlynnelServePolicy {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let live = levers::serve_policy();
        let raw = env_text(SERVE_VARIABLE);
        let (want, problem) = policy_asked(&raw);
        let (policy, bound, has_bound) = match live {
            levers::ServePolicy::Spread => (ServePolicy::Spread, 0, false),
            levers::ServePolicy::Any => (ServePolicy::Any, 0, false),
            levers::ServePolicy::Occupancy => (ServePolicy::Occupancy, 0, false),
            levers::ServePolicy::SpreadAt(b) => (ServePolicy::SpreadAt, b, true),
        };
        ps.write(ServePolicyState {
            policy,
            spread_bound_per_mille: bound,
            has_bound,
            variable: SERVE_VARIABLE.to_string(),
            variable_value: raw.text,
            variable_problem: problem,
            effective_matches_variable: policy_text(live) == want,
        })
    }
}

/// Reads the occupancy floor a served calibration must have been taken
/// above, in parts per thousand.
///
/// It latches like the other levers, and reads 900 when nothing sets
/// it. Set-FlynnelLever -Name occupancy_floor is how it moves, and
/// only before the first read.
///
/// # Examples
///
/// `Get-FlynnelOccupancyFloor`
///
/// `(Get-FlynnelOccupancyFloor).Value`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelOccupancyFloor",
    alias = "Get-FlyOccupancyFloor",
    output = ["Flynnel.Lever"]
)]
#[derive(Default)]
pub struct GetFlynnelOccupancyFloor {}

impl Cmdlet for GetFlynnelOccupancyFloor {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let rows = all_levers();
        ps.write(lever_named(&rows, "occupancy_floor")?)
    }
}

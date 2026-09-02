//! Domain-agnostic operation classification.
//!
//! [`OpClass`] is the trait every kernel-op enum implements so that
//! [`crate::sched::JobPlan::for_op_generic`] can accept any domain's
//! op enum (numerical, string, signal-processing, …) and read the one
//! scheduling-relevant property the work-stealing pool needs at
//! dispatch time: [`OpClass::is_latency_bound`].
//!
//! Each domain decides per-op whether SMT siblings help (latency-
//! bound, branch-divergent, memory-stall-heavy ops where siblings
//! hide stalls) or hurt (IMUL/FMA/vector-issue-port saturated ops
//! where siblings contest the same port).
//!
//! The trait surface is deliberately minimal because the call-site
//! constructor ([`crate::sched::JobPlan::for_op_generic`]) consumes
//! the op only to read this one boolean and store it on the plan.
//! Domain enums that want the full per-call tuning (cost estimate +
//! oversubscription factor in addition to SMT) map their variants to
//! [`crate::dispatch_profile::DispatchProfile`] and call
//! [`crate::sched::JobPlan::set_profile`] instead. Other op-specific
//! reasoning (cost models, kernel selection) lives in domain-specific
//! dispatch tables that callers can layer on top.

use core::fmt::Debug;
use core::hash::Hash;

/// Marker trait that every domain-specific kernel-op enum implements.
/// Lets the call-site constructor on [`crate::sched::JobPlan`] accept
/// any future domain enum via a single generic constructor.
pub trait OpClass: Copy + Clone + Eq + Hash + Debug + 'static {
    /// `true` when SMT-2 siblings on the same physical core help
    /// throughput (long-latency dependency chains, branch divergence,
    /// frequent cache stalls). `false` when siblings contest the same
    /// issue port (IMUL/FMA saturated kernels).
    fn is_latency_bound(&self) -> bool;
}

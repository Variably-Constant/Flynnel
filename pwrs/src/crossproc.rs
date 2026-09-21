//! Work that crosses a process boundary, and how it is routed there.
//!
//! # The wire carries an id, never code
//!
//! A cross-process job cannot carry a closure: the peer has no way to
//! dereference a pointer into this process's heap. So the wire carries
//! `(closure_id, args)` and the peer looks the id up in its own pass
//! registry. That is the same id-crosses-not-code shape the
//! accelerator ops use at the device boundary, and it is what makes
//! this family reachable from a script at all: a script names a pass
//! that the peer already holds.
//!
//! # What is bound here so far
//!
//! The routing question and the registry reading, both of which answer
//! without a peer process existing. Which of the four deque variants a
//! dispatch of a given shape would use is decided from the shape and
//! the host, and a script sizing a cross-process dispatch wants it
//! before it starts one. How many passes this process has registered,
//! and whether a named one is among them, is a reading of this
//! process.
//!
//! Submitting work to a peer, and the calibration that re-measures the
//! routing table, wait on a second process and land in later slices.

use pwrs::prelude::*;

use flynnel::backend::shared_mem::pass_registry;
use flynnel::backend::shared_mem::variant_dispatch::{
    DequeVariant as CrateVariant, DispatcherRoutingTable, WorkloadShape,
};

use crate::host::arg_err;

/// Which cross-process deque a dispatch runs over.
#[psenum(name = "Flynnel.DequeVariant")]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum DequeVariantKind {
    /// Mapped-file Chase-Lev, the production default. The largest
    /// inline-argument slot and the smallest constant on a single
    /// request and reply.
    #[default]
    ChaseLev,
    /// An LCRQ-on-LIFO hybrid. A batched producer amortizes the ring
    /// tail update across a burst.
    Loh,
    /// Cache-line publication, three items to a line, which spreads
    /// one release store over a small batch.
    Khpd,
    /// A mailbox per thief, so several drainers never contend on one
    /// shared head.
    Urd,
}

impl From<CrateVariant> for DequeVariantKind {
    fn from(v: CrateVariant) -> Self {
        match v {
            CrateVariant::ChaseLev => DequeVariantKind::ChaseLev,
            CrateVariant::Loh => DequeVariantKind::Loh,
            CrateVariant::Khpd => DequeVariantKind::Khpd,
            CrateVariant::Urd => DequeVariantKind::Urd,
        }
    }
}

/// One cross-process deque variant and what it can carry.
#[psclass(name = "Flynnel.DequeVariantInfo")]
#[derive(Clone, Default)]
pub struct DequeVariantInfo {
    /// Which variant.
    pub variant: DequeVariantKind,
    /// The label the crate's own diagnostics and benches use, so a
    /// figure read from one of those can be matched to a row here.
    pub label: String,
    /// The most argument bytes one item can carry inline. A dispatch
    /// whose payload is larger than this cannot use the variant, which
    /// is why the routing gate reads it.
    pub inline_args_bytes: u32,
    /// Whether this is the production default.
    pub is_default: bool,
}

/// Lists the cross-process deque variants and what each can carry
/// inline.
///
/// The inline ceiling is the column that decides a routing: a dispatch
/// whose argument payload is larger than a variant's slot cannot use
/// that variant whatever else recommends it, which is why
/// Get-FlynnelCrossProcessRoute takes the payload size rather than
/// inferring one.
///
/// # Examples
///
/// `Get-FlynnelCrossProcessVariant`
///
/// `Get-FlynnelCrossProcessVariant | Sort-Object InlineArgsBytes -Descending`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelCrossProcessVariant",
    alias = "Get-FlyCrossProcessVariant",
    output = ["Flynnel.DequeVariantInfo"]
)]
#[derive(Default)]
pub struct GetFlynnelCrossProcessVariant {}

impl Cmdlet for GetFlynnelCrossProcessVariant {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        for v in [
            CrateVariant::ChaseLev,
            CrateVariant::Loh,
            CrateVariant::Khpd,
            CrateVariant::Urd,
        ] {
            ps.write(DequeVariantInfo {
                variant: DequeVariantKind::from(v),
                label: v.label().to_string(),
                inline_args_bytes: v.inline_args_bytes() as u32,
                is_default: matches!(v, CrateVariant::ChaseLev),
            })?;
        }
        Ok(())
    }
}

/// Where a cross-process dispatch of one shape would go.
#[psclass(name = "Flynnel.CrossProcessRoute")]
#[derive(Clone, Default)]
pub struct CrossProcessRoute {
    /// The variant the table picks.
    pub variant: DequeVariantKind,
    /// Its label.
    pub label: String,
    /// The variant the fixed heuristic alone would pick. Different
    /// from Variant means an explicit cell decided this shape.
    pub heuristic_variant: DequeVariantKind,
    /// Whether the routing table holds an explicit cell for this exact
    /// shape rather than falling through to the heuristic. A
    /// calibration pins cells, so this says whether a measurement or a
    /// rule answered.
    pub from_explicit_cell: bool,
    /// The chosen variant's inline-argument ceiling.
    pub inline_args_bytes: u32,
    /// Whether the payload fits that ceiling. False is a real answer
    /// and means the dispatch would have to carry its arguments out of
    /// line, which this routing does not account for.
    pub payload_fits: bool,
    /// Drain threads the shape declared.
    pub n_drain_threads: u32,
    /// Inline argument bytes the shape declared.
    pub args_inline_bytes: u32,
    /// Items in the burst the shape declared.
    pub expected_burst_size: u32,
    /// Cells the routing table holds explicitly. The table is layered:
    /// a small map of overrides over a fixed rule, so this is the
    /// number of shapes a measurement has pinned.
    pub explicit_cells: u32,
}

/// One shape field, refused rather than truncated when it will not fit
/// the byte the shape carries it in.
///
/// A silently wrapped 256 routes as 0 and reads as a correct answer
/// for a payload that does not exist, so the range is checked and the
/// value is named.
fn shape_byte(name: &str, value: u32) -> PsResult<u8> {
    if value > u32::from(u8::MAX) {
        return Err(arg_err(format!(
            "{name} is {value} and the workload shape carries it in one byte, so it must be \
             at most 255"
        ))
        .terminating());
    }
    Ok(value as u8)
}

/// Reads which cross-process deque variant a dispatch of one shape
/// would route to, without dispatching anything.
///
/// The routing table is layered: a small map of explicitly pinned
/// cells over a fixed heuristic. FromExplicitCell says which of the
/// two answered, and HeuristicVariant is what the rule alone would
/// have said, so a cell a calibration pinned is visible as a
/// disagreement rather than being silently the same answer.
///
/// The shape's fields are the call-site parameters the per-variant win
/// zones depend on: how many peers drain, how many argument bytes each
/// item carries inline, and how many items a burst holds. A payload
/// larger than the chosen variant's slot is reported through
/// PayloadFits rather than changing the routing, because a routing
/// that quietly moved to a wider variant would hide the reason.
///
/// This needs no peer process. It is the decision alone.
///
/// # Examples
///
/// `Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8`
///
/// `Get-FlynnelCrossProcessRoute -NDrainThreads 8 -ArgsInlineBytes 8 -ExpectedBurstSize 64`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelCrossProcessRoute",
    alias = "Get-FlyCrossProcessRoute",
    output = ["Flynnel.CrossProcessRoute"]
)]
#[derive(Default)]
pub struct GetFlynnelCrossProcessRoute {
    /// Argument bytes each item carries inline.
    #[param(mandatory, position = 0)]
    pub args_inline_bytes: u32,
    /// Peer processes or threads draining from this site. One when
    /// unset. Zero means nobody is draining, which routes as one does.
    #[param(position = 1)]
    pub n_drain_threads: Option<u32>,
    /// Items the caller means to dispatch in one burst. One when
    /// unset, which is request and reply.
    #[param(position = 2)]
    pub expected_burst_size: Option<u32>,
    /// The base-2 logarithm of the cores cooperating as one logical
    /// vector. Part of a shape's identity for an explicitly pinned
    /// cell, and not read by the heuristic.
    #[param]
    pub k_unified: Option<u32>,
    /// The hardware-class tier the dispatch targets, where zero is
    /// scalar or SMT-shared cores and higher values are further-away
    /// coherence tiers.
    #[param]
    pub k_hardware_class: Option<u32>,
}

impl Cmdlet for GetFlynnelCrossProcessRoute {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let args_inline_bytes = shape_byte("ArgsInlineBytes", self.args_inline_bytes)?;
        let k_unified = shape_byte("KUnified", self.k_unified.unwrap_or(0))?;
        let k_hardware_class = shape_byte("KHardwareClass", self.k_hardware_class.unwrap_or(0))?;

        let shape = WorkloadShape {
            n_drain_threads: self.n_drain_threads.unwrap_or(1),
            args_inline_bytes,
            expected_burst_size: self.expected_burst_size.unwrap_or(1),
            k_unified,
            k_hardware_class,
        };
        let table = DispatcherRoutingTable::default_heuristic();
        let picked = table.pick(&shape);
        let heuristic = DispatcherRoutingTable::pick_heuristic(&shape);
        let ceiling = picked.inline_args_bytes();
        ps.write(CrossProcessRoute {
            variant: DequeVariantKind::from(picked),
            label: picked.label().to_string(),
            heuristic_variant: DequeVariantKind::from(heuristic),
            from_explicit_cell: table.cells().contains_key(&shape),
            inline_args_bytes: ceiling as u32,
            payload_fits: usize::from(args_inline_bytes) <= ceiling,
            n_drain_threads: shape.n_drain_threads,
            args_inline_bytes: u32::from(args_inline_bytes),
            expected_burst_size: shape.expected_burst_size,
            explicit_cells: table.cells().len() as u32,
        })
    }
}

/// What this process will run on behalf of a peer.
#[psclass(name = "Flynnel.PassRegistry")]
#[derive(Clone, Default)]
pub struct PassRegistryRow {
    /// Passes registered in this process.
    pub count: u32,
    /// The name asked about, when one was.
    pub name: Option<String>,
    /// The id that name hashes to, which is what travels on the wire.
    pub id: Option<u32>,
    /// Whether that id has a handler here. Null when no name or id was
    /// asked about, which is a different answer from false.
    pub registered: Option<bool>,
}

/// Reads the pass registry: how many passes this process will run for
/// a peer, and whether a named one is among them.
///
/// A cross-process job carries an id and its arguments, never code,
/// because the peer cannot dereference a pointer into this process's
/// heap. The registry is the table that turns an id back into
/// something to run, and it is per process: a peer's registry is its
/// own, and an id registered here says nothing about what the peer
/// will accept.
///
/// There is no Register cmdlet. A pass is registered with its handler
/// as a Rust closure, which a script has none of; a process that means
/// to serve passes registers them in its own code before it starts
/// draining.
///
/// Asking by Name hashes it to the id that travels on the wire, and
/// the row carries both, so two processes that disagree can be
/// compared by number rather than by spelling.
///
/// # Examples
///
/// `Get-FlynnelPassRegistry`
///
/// `Get-FlynnelPassRegistry -Name 'gemm.f64'`
///
/// `Get-FlynnelPassRegistry -Id 12345`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelPassRegistry",
    alias = "Get-FlyPassRegistry",
    output = ["Flynnel.PassRegistry"]
)]
#[derive(Default)]
pub struct GetFlynnelPassRegistry {
    /// A pass name to ask about, hashed to its wire id.
    #[param(position = 0)]
    pub name: Option<String>,
    /// A wire id to ask about directly, for a pass whose name this
    /// caller does not have.
    #[param]
    pub id: Option<u32>,
}

impl Cmdlet for GetFlynnelPassRegistry {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        if self.name.is_some() && self.id.is_some() {
            return Err(arg_err(
                "give Name or Id, not both; a name hashes to an id and the two could name \
                 different passes",
            )
            .terminating());
        }
        let id = match (&self.name, self.id) {
            (Some(name), None) => Some(pass_registry::hash_name(name)),
            (None, Some(id)) => Some(id),
            (None, None) => None,
            (Some(_), Some(_)) => unreachable!("both were refused above"),
        };
        ps.write(PassRegistryRow {
            count: pass_registry::registered_count() as u32,
            name: self.name.clone(),
            id,
            // Null rather than false when nothing was asked about: a
            // caller reading only the count must not see a false here
            // and take it for an answer about some pass.
            registered: id.map(pass_registry::is_registered),
        })
    }
}

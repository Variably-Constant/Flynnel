//! Per-call dispatcher composition over the four cross-process deque
//! backends. Given a workload-shape descriptor, picks the deque
//! variant that wins on that shape, then delegates to that backend's
//! native `dispatch_marshal*` entry point.
//!
//! ## What this does
//!
//! Four backends now live in [`super`]:
//!
//! | Variant   | Win zone (measured on Zen+ R7 2700)                  |
//! |-----------|------------------------------------------------------|
//! | ChaseLev  | single-drain request-reply; 504 ns single round-trip |
//! | LOH       | producer-fast batched dispatch (~1.13 x vs Chase-Lev)|
//! | KHPD      | producer-fast batched dispatch (~1.16 x vs Chase-Lev)|
//! | URD       | multi-thief 2+ drain threads (~2.33 x vs Chase-Lev)  |
//!
//! Each variant's per-slot inline-args ceiling differs:
//!
//! | Variant   | Inline args ceiling |
//! |-----------|---------------------|
//! | ChaseLev  | 48 B                |
//! | LOH       | 40 B                |
//! | KHPD      |  8 B                |
//! | URD       |  8 B                |
//!
//! ## Routing primitives
//!
//! - [`DequeVariant`]: enumerates the four backends.
//! - [`WorkloadShape`]: the routing key, capturing the call-site
//!   parameters that map a workload onto a backend win zone.
//! - [`DispatcherRoutingTable`]: a `WorkloadShape -> DequeVariant`
//!   table, seeded from the measured-win table above and updatable
//!   in place by [`super::dispatch_calibration`].
//! - [`CrossProcessDispatcher`]: the facade. Holds one optional
//!   backend per variant + the routing table; `dispatch_marshal` /
//!   `dispatch_marshal_batch` pick the variant by shape, dispatch
//!   through the matching backend, and return a tagged
//!   [`DispatchedHandle`] the caller waits on through the same facade.

#![allow(clippy::missing_errors_doc)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::chase_lev_backend::{DispatchHandle, SharedMemoryChaseLevBackend};
use super::chase_lev_mmf::ARGS_INLINE_BYTES as CHASE_LEV_INLINE_BYTES;
use super::khpd::KHPD_ARGS_INLINE_BYTES;
use super::khpd_backend::SharedMemoryKhpdBackend;
use super::lcrq_lifo::LOH_ARGS_INLINE_BYTES;
use super::loh_backend::SharedMemoryLohBackend;
use super::urd_backend::SharedMemoryUrdBackend;

use crate::backend::BackendError;
use crate::cpu_info::has_waitpkg;

/// The four cross-process deque backends the dispatcher routes among.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DequeVariant {
    /// MMF Chase-Lev work-stealing deque. Production default; 48 B
    /// inline args; smallest constant on single-drain request-reply.
    ChaseLev,
    /// LCRQ-on-LIFO hybrid. 40 B inline args; producer-fast batched
    /// dispatch amortizes ring-tail updates across a burst.
    Loh,
    /// Cache-line publication ring. 8 B inline args; 3 items per
    /// publication line amortizes Release-store cost across a small
    /// batch.
    Khpd,
    /// Per-thief mailbox with optional WAITPKG hardware wait. 8 B
    /// inline args; multi-thief workloads see no shared-head CAS
    /// contention.
    Urd,
}

impl DequeVariant {
    /// Per-variant inline-args ceiling in bytes. Used by the
    /// dispatcher to reject routings whose payload would overflow
    /// the chosen variant's slot.
    pub const fn inline_args_bytes(self) -> usize {
        match self {
            Self::ChaseLev => CHASE_LEV_INLINE_BYTES,
            Self::Loh => LOH_ARGS_INLINE_BYTES,
            Self::Khpd => KHPD_ARGS_INLINE_BYTES,
            Self::Urd => KHPD_ARGS_INLINE_BYTES,
        }
    }

    /// Short string label, used by bench harnesses + diagnostics.
    pub const fn label(self) -> &'static str {
        match self {
            Self::ChaseLev => "chase_lev",
            Self::Loh => "loh",
            Self::Khpd => "khpd",
            Self::Urd => "urd",
        }
    }
}

/// Workload-shape descriptor the dispatcher consumes per call. The
/// fields are the call-site parameters the per-variant win zones
/// depend on.
///
/// Hashed by value into the routing table; small fixed structure so
/// `derive(Hash)` is cheap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkloadShape {
    /// Number of drain threads (peer processes / threads draining
    /// from this dispatch site). `0` is "fire-and-forget; nobody is
    /// draining" and routes the same as `1` for variant selection;
    /// the dispatcher's wait path may still report no completion.
    pub n_drain_threads: u32,
    /// Bytes of inline argument payload per item. Used to gate
    /// variants whose per-slot inline-args ceiling is smaller than
    /// this number.
    pub args_inline_bytes: u8,
    /// Items the caller intends to dispatch in this burst. `1` is
    /// request-reply; >= 8 is the producer-fast burst the LOH / KHPD
    /// variants amortize across.
    pub expected_burst_size: u32,
    /// `log2(cores cooperating as one logical mega-vector)`.
    /// Corresponds to the K_unified axis in the K-hierarchy design.
    /// This field is part of the [`WorkloadShape`] hash + eq so it
    /// discriminates explicit routing-table entries, but
    /// [`DispatcherRoutingTable::pick_heuristic`] does not consult
    /// it (the heuristic uses `n_drain_threads`, `args_inline_bytes`,
    /// and `expected_burst_size`).
    pub k_unified: u8,
    /// Hardware-class tier the dispatch targets. `0` = scalar /
    /// SMT-shared cores; higher values denote progressively
    /// further-away coherence tiers (intra-CCX, cross-CCX,
    /// cross-socket, cross-class accelerator).
    pub k_hardware_class: u8,
}

impl WorkloadShape {
    /// Convenience constructor for the default request-reply shape:
    /// 1 drain thread, 8 B args, burst of 1, scalar tier.
    pub const fn request_reply(args_inline_bytes: u8) -> Self {
        Self {
            n_drain_threads: 1,
            args_inline_bytes,
            expected_burst_size: 1,
            k_unified: 0,
            k_hardware_class: 0,
        }
    }

    /// Convenience constructor for the producer-fast batched shape:
    /// 1 drain thread, caller-supplied burst, scalar tier.
    pub const fn producer_fast(args_inline_bytes: u8, burst: u32) -> Self {
        Self {
            n_drain_threads: 1,
            args_inline_bytes,
            expected_burst_size: burst,
            k_unified: 0,
            k_hardware_class: 0,
        }
    }

    /// Convenience constructor for the multi-thief shape:
    /// caller-supplied drain-thread count + burst, scalar tier.
    pub const fn multi_thief(args_inline_bytes: u8, n_drain_threads: u32, burst: u32) -> Self {
        Self {
            n_drain_threads,
            args_inline_bytes,
            expected_burst_size: burst,
            k_unified: n_drain_threads.ilog2() as u8,
            k_hardware_class: 0,
        }
    }
}

/// Mutable per-host routing table. `pick(shape)` returns the variant
/// chosen for that shape; `update_cell(shape, variant)` overrides one
/// cell (used by [`super::dispatch_calibration`] when a measured cell
/// disagrees with the heuristic seed).
///
/// The table is layered: a small `cells` HashMap holds explicit
/// overrides; cells not present in the map fall back to the fixed
/// heuristic logic in [`Self::pick_heuristic`]. This keeps the
/// initial table compact (~5 cells) while still letting calibration
/// pin specific cells.
#[derive(Debug, Clone, Default)]
pub struct DispatcherRoutingTable {
    cells: HashMap<WorkloadShape, DequeVariant>,
}

impl DispatcherRoutingTable {
    /// Construct an empty table. All `pick` calls fall through to
    /// the heuristic. Useful for tests that want the bare heuristic
    /// path without any explicit overrides.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Construct a routing table seeded with the measured-win
    /// overrides for the three shapes the per-variant benches
    /// established: single-drain request-reply -> ChaseLev,
    /// single-drain producer-fast (burst=64) -> KHPD, and
    /// multi-thief (4 thieves, burst=64) -> URD. The dispatcher
    /// consults this table on every call; shapes not present here
    /// fall through to [`Self::pick_heuristic`].
    pub fn default_heuristic() -> Self {
        let mut cells: HashMap<WorkloadShape, DequeVariant> = HashMap::new();
        // Single-drain, 8 B args, request-reply: Chase-Lev wins (504 ns).
        cells.insert(WorkloadShape::request_reply(8), DequeVariant::ChaseLev);
        // Single-drain, 8 B args, producer-fast (burst=64): KHPD wins
        // (1.16 x vs Chase-Lev). LOH ties closely with 1.13 x; pick
        // KHPD as the tiebreaker because its per-line publication
        // amortizes one Release-store across 3 items.
        cells.insert(
            WorkloadShape::producer_fast(8, 64),
            DequeVariant::Khpd,
        );
        // Multi-thief (4 thieves), 8 B args, burst=64: URD wins
        // (2.33 x vs Chase-Lev) because per-thief mailboxes
        // eliminate the shared-head CAS contention.
        cells.insert(
            WorkloadShape::multi_thief(8, 4, 64),
            DequeVariant::Urd,
        );
        Self { cells }
    }

    /// Look up a cell, returning the variant the dispatcher should
    /// pick. Falls through to [`Self::pick_heuristic`] when the cell
    /// is not explicitly stored.
    pub fn pick(&self, shape: &WorkloadShape) -> DequeVariant {
        if let Some(&v) = self.cells.get(shape) {
            return v;
        }
        Self::pick_heuristic(shape)
    }

    /// Default heuristic for shapes not in the explicit cells map.
    /// The decision flow mirrors the description in
    /// `docs/handoff_designs_d_and_e.md`.
    pub fn pick_heuristic(shape: &WorkloadShape) -> DequeVariant {
        // Multi-thief with WAITPKG available: URD WAITPKG path wakes
        // each thief on store without burning a SMT sibling.
        if shape.n_drain_threads >= 2 && has_waitpkg() {
            // URD's 8 B inline-args ceiling is the constraint; if
            // payload exceeds it, fall through to LOH / Chase-Lev.
            if (shape.args_inline_bytes as usize) <= KHPD_ARGS_INLINE_BYTES {
                return DequeVariant::Urd;
            }
        }
        // Multi-thief without WAITPKG: URD still wins on shared-head
        // contention via PauseSpin, as long as args fit.
        if shape.n_drain_threads >= 2
            && (shape.args_inline_bytes as usize) <= KHPD_ARGS_INLINE_BYTES
        {
            return DequeVariant::Urd;
        }
        // Producer-fast burst on a single drain thread: KHPD (tightest
        // inline amortization) when args fit; otherwise LOH at 40 B;
        // otherwise Chase-Lev at 48 B.
        if shape.expected_burst_size >= 8 {
            if (shape.args_inline_bytes as usize) <= KHPD_ARGS_INLINE_BYTES {
                return DequeVariant::Khpd;
            }
            if (shape.args_inline_bytes as usize) <= LOH_ARGS_INLINE_BYTES {
                return DequeVariant::Loh;
            }
        }
        // Request-reply or large-args: Chase-Lev is the production
        // default; 48 B inline ceiling covers everything we ship today.
        DequeVariant::ChaseLev
    }

    /// Override one cell. Used by [`super::dispatch_calibration`]
    /// after a host-specific measurement disagrees with the seeded
    /// heuristic for that exact shape.
    pub fn update_cell(&mut self, shape: WorkloadShape, variant: DequeVariant) {
        self.cells.insert(shape, variant);
    }

    /// Number of explicit cells currently in the table.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the table has any explicit cells (when `false`, every
    /// `pick` call hits the heuristic branch).
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Read-only view of the explicit cells. Used by diagnostics + the
    /// bench harness to enumerate the routing decisions.
    pub fn cells(&self) -> &HashMap<WorkloadShape, DequeVariant> {
        &self.cells
    }
}

/// Tagged dispatch handle returned by [`CrossProcessDispatcher`]. The
/// `variant` tag tells the caller which backend's `wait_handle` /
/// `poll_handle` to route through.
#[derive(Debug, Clone, Copy)]
pub struct DispatchedHandle {
    /// Which backend's latch arena holds the result cell.
    pub variant: DequeVariant,
    /// The variant-native handle; carries the latch offset.
    pub handle: DispatchHandle,
}

/// Top-level dispatcher facade. Owns one optional backend per variant
/// plus the routing table; routes per-call to the chosen variant +
/// returns a tagged handle the caller waits on through the facade.
///
/// The four backend slots are independent. A caller that only ships
/// Chase-Lev workloads can omit the other three; the dispatcher
/// returns [`BackendError::NotSupported`] when a routing decision
/// names a variant whose backend slot is `None`.
pub struct CrossProcessDispatcher {
    table: DispatcherRoutingTable,
    chase_lev: Option<Arc<SharedMemoryChaseLevBackend>>,
    loh: Option<Arc<SharedMemoryLohBackend>>,
    khpd: Option<Arc<SharedMemoryKhpdBackend>>,
    urd: Option<Arc<SharedMemoryUrdBackend>>,
    dispatched: AtomicU64,
}

impl CrossProcessDispatcher {
    /// Builder entry point. Start from an empty dispatcher; chain
    /// `with_<variant>` to install the backends the caller has
    /// configured + `with_table` to override the default routing
    /// table.
    pub fn builder() -> CrossProcessDispatcherBuilder {
        CrossProcessDispatcherBuilder::default()
    }

    /// Borrow the routing table. Diagnostics + tests.
    pub fn table(&self) -> &DispatcherRoutingTable {
        &self.table
    }

    /// Mutable borrow of the routing table. Used by
    /// [`super::dispatch_calibration`] to write measured cells in
    /// place.
    pub fn table_mut(&mut self) -> &mut DispatcherRoutingTable {
        &mut self.table
    }

    /// True when the named variant has an installed backend.
    pub fn has(&self, variant: DequeVariant) -> bool {
        match variant {
            DequeVariant::ChaseLev => self.chase_lev.is_some(),
            DequeVariant::Loh => self.loh.is_some(),
            DequeVariant::Khpd => self.khpd.is_some(),
            DequeVariant::Urd => self.urd.is_some(),
        }
    }

    /// Total items dispatched through the facade since construction.
    pub fn dispatched(&self) -> u64 {
        self.dispatched.load(Ordering::Relaxed)
    }

    /// Pick the variant the table chooses for `shape`, falling back
    /// to ChaseLev when the chosen variant's backend slot is `None`
    /// (and ChaseLev is installed) so callers can configure a strict
    /// subset of the four backends without crashing on cross-shape
    /// requests.
    pub fn pick_with_fallback(&self, shape: &WorkloadShape) -> Result<DequeVariant, BackendError> {
        let primary = self.table.pick(shape);
        if self.has(primary) {
            return Ok(primary);
        }
        // ChaseLev is the production default; if the primary slot is
        // empty but ChaseLev is installed, fall back to it (with the
        // payload-size invariant honored).
        if let Some(_be) = &self.chase_lev
            && (shape.args_inline_bytes as usize) <= CHASE_LEV_INLINE_BYTES
        {
            return Ok(DequeVariant::ChaseLev);
        }
        Err(BackendError::NotSupported)
    }

    /// Per-call dispatch. Picks the variant, validates the payload
    /// against that variant's inline-args ceiling, and delegates to
    /// the matching backend.
    pub fn dispatch_marshal(
        &self,
        shape: &WorkloadShape,
        closure_id: u32,
        args: &[u8],
    ) -> Result<DispatchedHandle, BackendError> {
        let variant = self.pick_with_fallback(shape)?;
        if args.len() > variant.inline_args_bytes() {
            return Err(BackendError::Launch(format!(
                "dispatcher: args length {} exceeds {} inline ceiling {}",
                args.len(),
                variant.label(),
                variant.inline_args_bytes(),
            )));
        }
        let handle = match variant {
            DequeVariant::ChaseLev => self
                .chase_lev
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .dispatch_marshal(closure_id, args)?,
            DequeVariant::Loh => self
                .loh
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .dispatch_marshal(closure_id, args)?,
            DequeVariant::Khpd => self
                .khpd
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .dispatch_marshal(closure_id, args)?,
            DequeVariant::Urd => self
                .urd
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .dispatch_marshal(closure_id, args)?,
        };
        self.dispatched.fetch_add(1, Ordering::Relaxed);
        Ok(DispatchedHandle { variant, handle })
    }

    /// Batched dispatch. The variant is picked once for the whole
    /// batch (the burst size + thief count are batch-level
    /// properties); every item ships through the chosen backend's
    /// `dispatch_marshal_batch`. ChaseLev falls back to per-item
    /// `dispatch_marshal` because the Chase-Lev backend has no
    /// native batch entry point.
    pub fn dispatch_marshal_batch(
        &self,
        shape: &WorkloadShape,
        items: &[(u32, &[u8])],
    ) -> Result<Vec<DispatchedHandle>, BackendError> {
        let variant = self.pick_with_fallback(shape)?;
        for (_, args) in items {
            if args.len() > variant.inline_args_bytes() {
                return Err(BackendError::Launch(format!(
                    "dispatcher: batch item args length {} exceeds {} inline ceiling {}",
                    args.len(),
                    variant.label(),
                    variant.inline_args_bytes(),
                )));
            }
        }
        let handles = match variant {
            DequeVariant::ChaseLev => {
                let be = self.chase_lev.as_ref().ok_or(BackendError::NotSupported)?;
                let mut out = Vec::with_capacity(items.len());
                for (cid, args) in items {
                    out.push(be.dispatch_marshal(*cid, args)?);
                }
                out
            }
            DequeVariant::Loh => self
                .loh
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .dispatch_marshal_batch(items)?,
            DequeVariant::Khpd => self
                .khpd
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .dispatch_marshal_batch(items)?,
            DequeVariant::Urd => self
                .urd
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .dispatch_marshal_batch(items)?,
        };
        self.dispatched
            .fetch_add(items.len() as u64, Ordering::Relaxed);
        Ok(handles.into_iter().map(|h| DispatchedHandle { variant, handle: h }).collect())
    }

    /// Wait on a tagged handle. Routes through the right backend's
    /// `wait_handle` based on the handle's variant tag.
    pub fn wait_handle(
        &self,
        handle: DispatchedHandle,
        iter_budget: u32,
    ) -> Result<Result<Vec<u8>, String>, BackendError> {
        match handle.variant {
            DequeVariant::ChaseLev => self
                .chase_lev
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .wait_handle(handle.handle, iter_budget),
            DequeVariant::Loh => self
                .loh
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .wait_handle(handle.handle, iter_budget),
            DequeVariant::Khpd => self
                .khpd
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .wait_handle(handle.handle, iter_budget),
            DequeVariant::Urd => self
                .urd
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .wait_handle(handle.handle, iter_budget),
        }
    }

    /// Non-blocking poll on a tagged handle.
    pub fn poll_handle(
        &self,
        handle: DispatchedHandle,
    ) -> Result<Option<Result<Vec<u8>, String>>, BackendError> {
        match handle.variant {
            DequeVariant::ChaseLev => self
                .chase_lev
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .poll_handle(handle.handle),
            DequeVariant::Loh => self
                .loh
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .poll_handle(handle.handle),
            DequeVariant::Khpd => self
                .khpd
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .poll_handle(handle.handle),
            DequeVariant::Urd => self
                .urd
                .as_ref()
                .ok_or(BackendError::NotSupported)?
                .poll_handle(handle.handle),
        }
    }

    /// Borrow the installed ChaseLev backend (if any). Used by peer
    /// processes that need to call `drain_one` against the same
    /// backend.
    pub fn chase_lev_backend(&self) -> Option<&SharedMemoryChaseLevBackend> {
        self.chase_lev.as_deref()
    }
    /// Borrow the installed LOH backend (if any).
    pub fn loh_backend(&self) -> Option<&SharedMemoryLohBackend> {
        self.loh.as_deref()
    }
    /// Borrow the installed KHPD backend (if any).
    pub fn khpd_backend(&self) -> Option<&SharedMemoryKhpdBackend> {
        self.khpd.as_deref()
    }
    /// Borrow the installed URD backend (if any).
    pub fn urd_backend(&self) -> Option<&SharedMemoryUrdBackend> {
        self.urd.as_deref()
    }
}

/// Builder for [`CrossProcessDispatcher`]. Default state has no
/// backends installed and an empty routing table; call the matching
/// `with_*` setter for each backend the caller has configured.
#[derive(Default)]
pub struct CrossProcessDispatcherBuilder {
    table: DispatcherRoutingTable,
    chase_lev: Option<Arc<SharedMemoryChaseLevBackend>>,
    loh: Option<Arc<SharedMemoryLohBackend>>,
    khpd: Option<Arc<SharedMemoryKhpdBackend>>,
    urd: Option<Arc<SharedMemoryUrdBackend>>,
}

impl CrossProcessDispatcherBuilder {
    /// Replace the routing table.
    pub fn with_table(mut self, table: DispatcherRoutingTable) -> Self {
        self.table = table;
        self
    }
    /// Install the ChaseLev backend slot.
    pub fn with_chase_lev(mut self, be: Arc<SharedMemoryChaseLevBackend>) -> Self {
        self.chase_lev = Some(be);
        self
    }
    /// Install the LOH backend slot.
    pub fn with_loh(mut self, be: Arc<SharedMemoryLohBackend>) -> Self {
        self.loh = Some(be);
        self
    }
    /// Install the KHPD backend slot.
    pub fn with_khpd(mut self, be: Arc<SharedMemoryKhpdBackend>) -> Self {
        self.khpd = Some(be);
        self
    }
    /// Install the URD backend slot.
    pub fn with_urd(mut self, be: Arc<SharedMemoryUrdBackend>) -> Self {
        self.urd = Some(be);
        self
    }
    /// Finalize the dispatcher. The caller is responsible for
    /// having installed at least one backend whose payload ceiling
    /// satisfies the workload-shape requirements; the dispatcher
    /// itself does not panic on empty-slot routings (it returns
    /// [`BackendError::NotSupported`]).
    pub fn build(self) -> CrossProcessDispatcher {
        CrossProcessDispatcher {
            table: self.table,
            chase_lev: self.chase_lev,
            loh: self.loh,
            khpd: self.khpd,
            urd: self.urd,
            dispatched: AtomicU64::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::shared_mem::pass_registry::{hash_name, register, unregister};
    use std::path::PathBuf;

    fn temp_pair(label: &str) -> (PathBuf, PathBuf) {
        let mut d = std::env::temp_dir();
        let mut l = std::env::temp_dir();
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        d.push(format!("flynnel_vd_{pid}_{nonce}_{label}_d.bin"));
        l.push(format!("flynnel_vd_{pid}_{nonce}_{label}_l.bin"));
        (d, l)
    }

    #[test]
    fn heuristic_picks_chase_lev_for_request_reply() {
        let table = DispatcherRoutingTable::default_heuristic();
        let shape = WorkloadShape::request_reply(8);
        assert_eq!(table.pick(&shape), DequeVariant::ChaseLev);
    }

    #[test]
    fn heuristic_picks_khpd_for_producer_fast_small_args() {
        let table = DispatcherRoutingTable::default_heuristic();
        let shape = WorkloadShape::producer_fast(8, 64);
        assert_eq!(table.pick(&shape), DequeVariant::Khpd);
    }

    #[test]
    fn heuristic_picks_loh_for_producer_fast_args_over_khpd() {
        // 16 B args won't fit KHPD/URD's 8 B ceiling; LOH at 40 B wins.
        let shape = WorkloadShape::producer_fast(16, 64);
        let table = DispatcherRoutingTable::empty();
        assert_eq!(table.pick(&shape), DequeVariant::Loh);
    }

    #[test]
    fn heuristic_picks_chase_lev_when_args_exceed_loh() {
        // 45 B args > LOH's 40; Chase-Lev's 48 still fits.
        let shape = WorkloadShape::producer_fast(45, 64);
        let table = DispatcherRoutingTable::empty();
        assert_eq!(table.pick(&shape), DequeVariant::ChaseLev);
    }

    #[test]
    fn heuristic_picks_urd_for_multi_thief_small_args() {
        let table = DispatcherRoutingTable::empty();
        let shape = WorkloadShape::multi_thief(8, 4, 64);
        assert_eq!(table.pick(&shape), DequeVariant::Urd);
    }

    #[test]
    fn update_cell_overrides_heuristic() {
        let mut table = DispatcherRoutingTable::empty();
        let shape = WorkloadShape::request_reply(8);
        assert_eq!(table.pick(&shape), DequeVariant::ChaseLev);
        table.update_cell(shape, DequeVariant::Urd);
        assert_eq!(table.pick(&shape), DequeVariant::Urd);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn inline_args_bytes_matches_per_backend_ceilings() {
        assert_eq!(DequeVariant::ChaseLev.inline_args_bytes(), CHASE_LEV_INLINE_BYTES);
        assert_eq!(DequeVariant::Loh.inline_args_bytes(), LOH_ARGS_INLINE_BYTES);
        assert_eq!(DequeVariant::Khpd.inline_args_bytes(), KHPD_ARGS_INLINE_BYTES);
        assert_eq!(DequeVariant::Urd.inline_args_bytes(), KHPD_ARGS_INLINE_BYTES);
    }

    #[test]
    fn dispatcher_with_no_backends_rejects_dispatch() {
        let d = CrossProcessDispatcher::builder()
            .with_table(DispatcherRoutingTable::default_heuristic())
            .build();
        let shape = WorkloadShape::request_reply(8);
        let r = d.dispatch_marshal(&shape, 0, &[]);
        assert!(matches!(r, Err(BackendError::NotSupported)));
    }

    #[test]
    fn dispatcher_round_trips_through_chase_lev() {
        let (d, l) = temp_pair("cl_route");
        let cl =
            Arc::new(SharedMemoryChaseLevBackend::create(0, &d, &l, 4, 8).expect("create"));
        let dispatcher = CrossProcessDispatcher::builder()
            .with_table(DispatcherRoutingTable::default_heuristic())
            .with_chase_lev(Arc::clone(&cl))
            .build();

        let id = hash_name("flynnel.test.vd.cl_adder");
        register(id, |args| {
            let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
            Ok((a + b).to_le_bytes().to_vec())
        });

        let mut payload = [0u8; 8];
        payload[..4].copy_from_slice(&13u32.to_le_bytes());
        payload[4..].copy_from_slice(&29u32.to_le_bytes());
        let shape = WorkloadShape::request_reply(8);
        let handle = dispatcher
            .dispatch_marshal(&shape, id, &payload)
            .expect("dispatch");
        assert_eq!(handle.variant, DequeVariant::ChaseLev);

        cl.drain_owner().expect("drain").expect("had work");
        let r = dispatcher
            .wait_handle(handle, 1024)
            .expect("wait")
            .expect("ok branch");
        let v = u32::from_le_bytes(r[..4].try_into().unwrap());
        assert_eq!(v, 42);

        unregister(id);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn dispatcher_rejects_oversize_args() {
        let (d, l) = temp_pair("oversize");
        let cl =
            Arc::new(SharedMemoryChaseLevBackend::create(0, &d, &l, 4, 8).expect("create"));
        let dispatcher = CrossProcessDispatcher::builder()
            .with_chase_lev(Arc::clone(&cl))
            .build();
        let shape = WorkloadShape::request_reply(8);
        // 64 B exceeds Chase-Lev's 48 B ceiling.
        let big = vec![0u8; 64];
        let r = dispatcher.dispatch_marshal(&shape, 0, &big);
        assert!(matches!(r, Err(BackendError::Launch(_))));
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn fallback_to_chase_lev_when_primary_variant_absent() {
        // Routing table picks URD; URD slot empty; falls back to ChaseLev.
        let (d, l) = temp_pair("fallback");
        let cl =
            Arc::new(SharedMemoryChaseLevBackend::create(0, &d, &l, 4, 8).expect("create"));
        let dispatcher = CrossProcessDispatcher::builder()
            .with_chase_lev(Arc::clone(&cl))
            .build();
        let shape = WorkloadShape::multi_thief(8, 4, 64);
        let picked = dispatcher.pick_with_fallback(&shape).expect("pick");
        assert_eq!(picked, DequeVariant::ChaseLev);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }
}

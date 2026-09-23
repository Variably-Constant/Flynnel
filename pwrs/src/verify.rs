//! The verify chain, and the mode region the CGRA backends run in.
//!
//! # What a chain is for
//!
//! A chain absorbs a sequence of chunks and answers a 32-byte root.
//! Two chains fed the same bytes in the same order root the same, and
//! that is the whole claim: it is how a CPU trace and a device trace
//! are checked for being bit-exact without holding both in memory.
//!
//! The order is the load-bearing part. `Add` and `AddMany` submit in
//! the order given, and the crate folds them in that order however the
//! work is scheduled.
//!
//! # Why Compare can name the index
//!
//! The crate's chain answers a root and nothing else, so a root that
//! disagrees says only that something differs. This module keeps a
//! digest of each chunk beside the chain, so `Compare-FlynnelVerifyChain`
//! can say WHERE: the first index whose chunks differ, which is the
//! thing a caller then goes and looks at.
//!
//! The digests are BLAKE3 over each chunk, so the index is exact
//! rather than a fingerprint that could collide and name the wrong
//! one. They cost 32 bytes a chunk and nothing else; the chunks
//! themselves are not kept.
//!
//! # The mode region
//!
//! `run_in_region` enters a tile mode, runs a bounded body, and exits,
//! with the exit paired to the enter by a guard that fires on an
//! unwinding panic as well as a normal return. That pairing is the
//! whole guarantee, and `Test-FlynnelModeRegion` is here to check it
//! rather than to assume it - including through a body that panics on
//! purpose, because a guard that has never seen a panic is untested.

use std::panic::AssertUnwindSafe;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use pwrs::prelude::*;

use flynnel::sched::mode_region::{MatrixModeBackend, ScalarConfig, ScalarFallback, run_in_region};
use flynnel::sched::verify_chain::{FxFallbackHasher, VerifyChain as CrateChain};

/// Which hasher a chain roots with.
#[psenum(name = "Flynnel.VerifyHasher")]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum VerifyHasherKind {
    /// The BLAKE3 root, which is what an attestation uses.
    #[default]
    Blake3,
    /// A cheap non-cryptographic fallback, for a build without the
    /// verify-chain feature or a caller who wants speed over a root
    /// anyone else will trust.
    FxFallback,
}

// ---------------------------------------------------------------------
// The chain table
// ---------------------------------------------------------------------

/// One live chain: the crate's, the per-chunk digests this module
/// keeps so a comparison can name an index, and the root once taken.
struct ChainEntry {
    id: u64,
    hasher: VerifyHasherKind,
    /// Taken by the first Root call, because the crate's finalize
    /// consumes the chain.
    chain: Option<CrateChain>,
    digests: Vec<[u8; 32]>,
    /// The root as hex once finalized. A chain answers its root once
    /// and holds it: the crate's finalize consumes the hasher and a
    /// second call would answer thirty-two zero bytes, which reads as
    /// a root and is not one.
    root: Option<String>,
}

static CHAINS: Mutex<Vec<ChainEntry>> = Mutex::new(Vec::new());
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// The chain table, recovering a lock a panicking caller poisoned.
///
/// The state is a vector of entries and nothing here spans two
/// operations, so a panic leaves it consistent and refusing would
/// strand every chain in the session over one unrelated failure.
fn chains() -> std::sync::MutexGuard<'static, Vec<ChainEntry>> {
    match CHAINS.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn unknown_chain(id: u64) -> PsError {
    PsError::new(
        ErrorCategory::ObjectNotFound,
        "FlynnelUnknownChain",
        format!(
            "verify chain {id} is not one this module holds: it has been disposed, or the \
             object came from another process's copy of the module"
        ),
    )
}

fn arg_err(message: impl Into<String>) -> PsError {
    PsError::new(ErrorCategory::InvalidArgument, "FlynnelArgument", message.into())
}

fn with_chain<T>(id: u64, f: impl FnOnce(&mut ChainEntry) -> PsResult<T>) -> PsResult<T> {
    let mut table = chains();
    let entry = table.iter_mut().find(|c| c.id == id).ok_or_else(|| unknown_chain(id))?;
    f(entry)
}

/// Adds chunks to a chain in the order given, recording a digest of
/// each so a later comparison can name the index it diverges at.
fn add_chunks(id: u64, chunks: Vec<Vec<u8>>) -> PsResult<u64> {
    with_chain(id, |entry| {
        let Some(chain) = entry.chain.as_ref() else {
            return Err(PsError::new(
                ErrorCategory::InvalidOperation,
                "FlynnelChainClosed",
                format!(
                    "verify chain {id} has answered its root and takes no more chunks; a \
                     root is the end of a chain"
                ),
            ));
        };
        for chunk in chunks {
            entry.digests.push(*blake3::hash(&chunk).as_bytes());
            chain.submit_chunk(chunk);
        }
        Ok(entry.digests.len() as u64)
    })
}

fn hex32(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// A chain that absorbs chunks and answers one root.
#[psclass(name = "Flynnel.VerifyChain", mode = proxy)]
pub struct VerifyChainHandle {
    /// Names this chain for its session, and is how a cmdlet reaches
    /// it.
    pub id: u64,
    /// Which hasher it roots with.
    pub hasher: VerifyHasherKind,
    #[allow(dead_code)]
    #[psfield(skip)]
    guard: ChainGuard,
}

/// Frees a chain's table entry when the object is disposed or
/// collected.
///
/// A skipped field, which also makes the class unreadable by value, so
/// no cmdlet takes one as a typed parameter and drops a copy that
/// would take the live chain's entry with it. The cmdlets that need a
/// chain read its Id off the object instead.
struct ChainGuard(u64);

impl Drop for ChainGuard {
    fn drop(&mut self) {
        let mut table = chains();
        if let Some(at) = table.iter().position(|c| c.id == self.0) {
            table.remove(at);
        }
    }
}

/// The operations of a `Flynnel.VerifyChain`.
#[psmethods]
impl VerifyChainHandle {
    /// A chain that roots with Hasher, or with BLAKE3 when Hasher is
    /// omitted.
    ///
    /// A script reaches this as `[Flynnel.VerifyChain]::new()` or
    /// `[Flynnel.VerifyChain]::new('FxFallback')`. New-FlynnelVerifyChain
    /// builds its chain here as well, so the two routes make the same
    /// object.
    pub fn new(hasher: Option<VerifyHasherKind>) -> PsResult<VerifyChainHandle> {
        let kind = hasher.unwrap_or(VerifyHasherKind::Blake3);
        let chain = match kind {
            VerifyHasherKind::Blake3 => CrateChain::new(),
            VerifyHasherKind::FxFallback => {
                CrateChain::with_hasher(Box::new(FxFallbackHasher::new()))
            }
        };
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        chains().push(ChainEntry {
            id,
            hasher: kind,
            chain: Some(chain),
            digests: Vec::new(),
            root: None,
        });
        Ok(VerifyChainHandle { id, hasher: kind, guard: ChainGuard(id) })
    }

    /// Adds one chunk, answering how many the chain now holds.
    ///
    /// Costs a crossing per chunk. AddMany is the form for more than
    /// a handful.
    pub fn add(&self, chunk: Vec<u8>) -> PsResult<u64> {
        add_chunks(self.id, vec![chunk])
    }

    /// Adds every chunk in one crossing, in the order given.
    ///
    /// Order is what a root means, so the order here is the order the
    /// chain folds them in.
    pub fn add_many(&self, chunks: Vec<Vec<u8>>) -> PsResult<u64> {
        add_chunks(self.id, chunks)
    }

    /// Chunks submitted to this chain.
    pub fn count(&self) -> PsResult<u64> {
        with_chain(self.id, |entry| Ok(entry.digests.len() as u64))
    }

    /// Chunks submitted and not yet folded in.
    ///
    /// Zero on a process with no IO pool, where a submission is
    /// folded in on the calling thread before Add returns.
    pub fn pending(&self) -> PsResult<u64> {
        with_chain(self.id, |entry| match entry.chain.as_ref() {
            Some(chain) => Ok(chain.pending_count() as u64),
            None => Ok(0),
        })
    }

    /// The root as hex, waiting for every submitted chunk first.
    ///
    /// A chain answers its root once and then holds it: the crate's
    /// finalize consumes the hasher, and asking twice would answer
    /// thirty-two zero bytes, which reads as a root and is not one.
    /// Calling this again returns the same string, and Add after it is
    /// refused.
    pub fn root(&self) -> PsResult<String> {
        with_chain(self.id, |entry| {
            if let Some(root) = &entry.root {
                return Ok(root.clone());
            }
            let Some(chain) = entry.chain.take() else {
                return Err(PsError::new(
                    ErrorCategory::InvalidOperation,
                    "FlynnelChainClosed",
                    format!("verify chain {} has no hasher left to finalize", self.id),
                ));
            };
            let root = hex32(&chain.finalize());
            entry.root = Some(root.clone());
            Ok(root)
        })
    }
}

/// Makes a verify chain.
///
/// A chain absorbs chunks and answers one 32-byte root. Two chains fed
/// the same bytes in the same order root the same, which is how a CPU
/// trace and a device trace are checked for being bit-exact.
///
/// Order is what a root means. Add and AddMany submit in the order
/// given and the crate folds them in that order however the work is
/// scheduled.
///
/// # Examples
///
/// `$chain = New-FlynnelVerifyChain`
///
/// `$chain = New-FlynnelVerifyChain -Hasher FxFallback`
#[cmdlet(
    verb = "New",
    noun = "FlynnelVerifyChain",
    alias = "New-FlyVerifyChain",
    output = ["Flynnel.VerifyChain"]
)]
#[derive(Default)]
pub struct NewFlynnelVerifyChain {
    /// Which hasher to root with. BLAKE3 by default, which is what an
    /// attestation uses.
    #[param(position = 0)]
    pub hasher: Option<VerifyHasherKind>,
}

impl Cmdlet for NewFlynnelVerifyChain {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(VerifyChainHandle::new(self.hasher)?)
    }
}

/// Reads the chain id off whichever chain object was passed.
fn chain_id(obj: &PsObject, parameter: &str) -> PsResult<u64> {
    if obj.is_null() {
        return Err(arg_err(format!(
            "-{parameter} was given $null rather than a verify chain"
        )));
    }
    let id = obj.get("Id").map_err(|e| {
        arg_err(format!(
            "-{parameter} needs a chain from New-FlynnelVerifyChain; reading its Id \
             property failed: {e}"
        ))
    })?;
    u64::from_ps(&id)
        .map_err(|e| arg_err(format!("-{parameter} has an Id that is not a number: {e}")))
}

/// What two chains agree and disagree about.
#[psclass(name = "Flynnel.VerifyComparison")]
#[derive(Clone, Default)]
pub struct VerifyComparison {
    /// Whether the two roots are the same string.
    pub roots_agree: bool,
    /// The reference chain's root.
    pub reference_root: String,
    /// The other chain's root.
    pub difference_root: String,
    /// Chunks in the reference chain.
    pub reference_count: u64,
    /// Chunks in the other chain.
    pub difference_count: u64,
    /// Whether an index could be named. False when the roots agree,
    /// and also when they disagree only in length, which
    /// FirstExtraIndex covers instead.
    pub has_diverging_index: bool,
    /// The first index whose chunks differ. Meaningful only when
    /// HasDivergingIndex.
    pub first_diverging_index: u64,
    /// The index where the shorter chain ran out, when one is a
    /// prefix of the other. Meaningful only when LengthsDiffer.
    pub first_extra_index: u64,
    /// Whether the chains hold different numbers of chunks. Two
    /// chains of different lengths can still share every chunk they
    /// both have, and that is a different finding from a chunk that
    /// differs.
    pub lengths_differ: bool,
    /// Whether the two chains root with different hashers. When they
    /// do, the roots are not comparable and RootsAgree is false for
    /// that reason rather than because the traces differ - but the
    /// index columns still hold, because the per-chunk digests this
    /// module keeps are the same whatever the chain roots with.
    pub hashers_differ: bool,
}

/// Compares two verify chains and says where they first differ.
///
/// A root that disagrees says only that something differs. This says
/// which chunk: the first index whose contents differ, found from a
/// BLAKE3 digest of each chunk kept beside the chain, so the index is
/// exact rather than a fingerprint that could name the wrong one.
///
/// Two chains of different lengths that agree on every chunk they both
/// hold is a different finding from a chunk that differs, and the row
/// separates them: LengthsDiffer with no diverging index means one is
/// a prefix of the other, and FirstExtraIndex is where the shorter one
/// ran out.
///
/// Both chains are finalized by this if they have not been already,
/// because a root is what the comparison is about.
///
/// # Examples
///
/// `Compare-FlynnelVerifyChain -Reference $cpu -Difference $gpu`
///
/// `(Compare-FlynnelVerifyChain -Reference $a -Difference $b).FirstDivergingIndex`
#[cmdlet(
    verb = "Compare",
    noun = "FlynnelVerifyChain",
    alias = "Compare-FlyVerifyChain",
    output = ["Flynnel.VerifyComparison"]
)]
#[derive(Default)]
pub struct CompareFlynnelVerifyChain {
    /// The chain taken as correct.
    #[param(mandatory, position = 0)]
    pub reference: PsObject,
    /// The chain being checked against it.
    #[param(mandatory, position = 1)]
    pub difference: PsObject,
}

impl Cmdlet for CompareFlynnelVerifyChain {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let left = chain_id(&self.reference, "Reference")?;
        let right = chain_id(&self.difference, "Difference")?;
        if left == right {
            return Err(arg_err(
                "Reference and Difference are the same chain, which can only agree with \
                 itself and says nothing",
            )
            .terminating());
        }

        // Finalize each before reading the digests, so the roots
        // reported are the roots of everything submitted.
        let reference_root = finalize_root(left)?;
        let difference_root = finalize_root(right)?;

        let (a, b, hashers_differ) = {
            let table = chains();
            let left_entry =
                table.iter().find(|c| c.id == left).ok_or_else(|| unknown_chain(left))?;
            let right_entry =
                table.iter().find(|c| c.id == right).ok_or_else(|| unknown_chain(right))?;
            (
                left_entry.digests.clone(),
                right_entry.digests.clone(),
                left_entry.hasher != right_entry.hasher,
            )
        };

        if hashers_differ {
            // Two hashers over identical bytes root differently, so a
            // disagreement here says nothing about the traces. Warned
            // rather than refused, because the index columns are still
            // sound: the per-chunk digests are BLAKE3 whatever the
            // chain itself roots with.
            pwrs::warning!(
                ps,
                "these chains root with different hashers, so their roots are not \
                 comparable and RootsAgree is false for that reason rather than because \
                 the traces differ. The index columns still hold"
            )?;
        }

        let shared = a.len().min(b.len());
        let diverging = (0..shared).find(|&i| a[i] != b[i]);

        ps.write(VerifyComparison {
            hashers_differ,
            roots_agree: reference_root == difference_root,
            reference_root,
            difference_root,
            reference_count: a.len() as u64,
            difference_count: b.len() as u64,
            has_diverging_index: diverging.is_some(),
            first_diverging_index: diverging.unwrap_or(0) as u64,
            first_extra_index: shared as u64,
            lengths_differ: a.len() != b.len(),
        })
    }
}

/// Takes a chain's root, finalizing it if it has not been.
fn finalize_root(id: u64) -> PsResult<String> {
    with_chain(id, |entry| {
        if let Some(root) = &entry.root {
            return Ok(root.clone());
        }
        let Some(chain) = entry.chain.take() else {
            return Err(PsError::new(
                ErrorCategory::InvalidOperation,
                "FlynnelChainClosed",
                format!("verify chain {id} has no hasher left to finalize"),
            ));
        };
        let root = hex32(&chain.finalize());
        entry.root = Some(root.clone());
        Ok(root)
    })
}

// ---------------------------------------------------------------------
// The mode region
// ---------------------------------------------------------------------

/// A matrix-mode backend this host could run a region in.
#[psclass(name = "Flynnel.MatrixBackend")]
#[derive(Clone, Default)]
pub struct MatrixBackend {
    /// The backend's name.
    pub name: String,
    /// Whether a region can be entered on it here.
    pub available: bool,
    /// Whether this is the fallback the substrate always has rather
    /// than a tile backend.
    pub is_fallback: bool,
    /// What it does on enter and exit.
    pub note: String,
}

/// Reads every matrix-mode backend a region can be entered in.
///
/// There is exactly one today and it is the scalar fallback. The crate
/// carries the substrate - the trait, the guard that pairs exit to
/// enter, and `run_in_region` - and no tile backend implements it yet,
/// so a host with AMX or SME has nothing here to select and the row
/// says so.
///
/// That is a row rather than an empty listing on purpose. An empty
/// listing reads as a family that failed to enumerate; a row saying
/// the only backend is the fallback says what is actually true.
///
/// # Examples
///
/// `Get-FlynnelMatrixBackend`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelMatrixBackend",
    alias = "Get-FlyMatrixBackend",
    output = ["Flynnel.MatrixBackend"]
)]
#[derive(Default)]
pub struct GetFlynnelMatrixBackend {}

impl Cmdlet for GetFlynnelMatrixBackend {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(MatrixBackend {
            name: "ScalarFallback".to_string(),
            available: true,
            is_fallback: true,
            note: "enter and exit are no-ops and the body runs as plain scalar or vector \
                   code, so a region compiles and runs on a host with no matrix extension"
                .to_string(),
        })
    }
}

/// Counts of enters and exits, so the pairing can be checked rather
/// than assumed.
static ENTERS: AtomicUsize = AtomicUsize::new(0);
static EXITS: AtomicUsize = AtomicUsize::new(0);

/// A backend that does nothing but count, so a test can see whether
/// the guard paired the exit to the enter.
///
/// The scalar fallback's enter and exit are no-ops, which makes them
/// unobservable: a run through it cannot tell a paired exit from no
/// exit at all. This counts, which is the only way the guarantee can
/// be checked from outside the crate.
struct CountingBackend;

impl MatrixModeBackend for CountingBackend {
    type Config = ScalarConfig;
    type Context = ();

    unsafe fn enter(_config: &Self::Config) -> Self::Context {
        ENTERS.fetch_add(1, Ordering::SeqCst);
    }

    unsafe fn exit(_ctx: Self::Context) {
        EXITS.fetch_add(1, Ordering::SeqCst);
    }
}

/// What a region check found.
#[psclass(name = "Flynnel.ModeRegionCheck")]
#[derive(Clone, Default)]
pub struct ModeRegionCheck {
    /// Whether a region that returned normally exited exactly once.
    pub normal_return_paired: bool,
    /// Whether a region whose body panicked still exited exactly once.
    pub panicking_body_paired: bool,
    /// Whether the panic was caught rather than crossing the boundary.
    pub panic_caught: bool,
    /// Enters counted across both arms.
    pub enters: u64,
    /// Exits counted across both arms. Equal to Enters is the whole
    /// guarantee.
    pub exits: u64,
    /// Whether the scalar fallback also entered and exited cleanly,
    /// which is the backend a host without a matrix extension uses.
    pub fallback_ran: bool,
}

/// Enters and exits a mode region and reports whether the exit was
/// paired to the enter, including through a body that panics.
///
/// The pairing is the substrate's whole guarantee: a host enters a
/// tile mode, runs a bounded body, and must leave that mode however
/// the body ends. A guard that has never seen a panic is untested, so
/// this runs one on purpose and catches it.
///
/// It counts through a backend of this module's own rather than
/// through the scalar fallback, whose enter and exit are no-ops and
/// therefore unobservable: a run through the fallback cannot tell a
/// paired exit from no exit at all. The fallback is exercised too, and
/// reported separately, because it is what a host with no matrix
/// extension actually uses.
///
/// # Examples
///
/// `Test-FlynnelModeRegion`
///
/// `(Test-FlynnelModeRegion).PanickingBodyPaired`
#[cmdlet(
    verb = "Test",
    noun = "FlynnelModeRegion",
    alias = "Test-FlyModeRegion",
    output = ["Flynnel.ModeRegionCheck"]
)]
#[derive(Default)]
pub struct TestFlynnelModeRegion {}

impl Cmdlet for TestFlynnelModeRegion {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let enters_before = ENTERS.load(Ordering::SeqCst);
        let exits_before = EXITS.load(Ordering::SeqCst);

        // A body that returns normally.
        run_in_region::<CountingBackend, _, _>(&ScalarConfig, |_ctx| {});
        let normal_return_paired = ENTERS.load(Ordering::SeqCst) - enters_before == 1
            && EXITS.load(Ordering::SeqCst) - exits_before == 1;

        // A body that panics. The guard's Drop runs while the panic
        // unwinds, so the exit happens before the catch sees it.
        let enters_mid = ENTERS.load(Ordering::SeqCst);
        let exits_mid = EXITS.load(Ordering::SeqCst);
        let caught = std::panic::catch_unwind(AssertUnwindSafe(|| {
            run_in_region::<CountingBackend, _, _>(&ScalarConfig, |_ctx| {
                panic!("deliberate panic, to check the region still exits");
            })
        }));
        let panicking_body_paired = ENTERS.load(Ordering::SeqCst) - enters_mid == 1
            && EXITS.load(Ordering::SeqCst) - exits_mid == 1;

        // And the fallback itself, which is what a host with no matrix
        // extension runs. Its no-op enter and exit cannot be counted,
        // so all this says is that a region through it completes.
        let fallback_ran = run_in_region::<ScalarFallback, _, _>(&ScalarConfig, |_ctx| true);

        ps.write(ModeRegionCheck {
            normal_return_paired,
            panicking_body_paired,
            panic_caught: caught.is_err(),
            enters: (ENTERS.load(Ordering::SeqCst) - enters_before) as u64,
            exits: (EXITS.load(Ordering::SeqCst) - exits_before) as u64,
            fallback_ran,
        })
    }
}

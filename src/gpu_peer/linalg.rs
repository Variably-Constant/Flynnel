//! House-owned f64 linear algebra over resident VRAM blocks: generic
//! einsum, batched small GEMM, batched Jacobi symmetric eigen, and
//! batched one-sided Jacobi SVD, all from `kernels/linalg_f64.ptx`
//! (driver-JIT'd, no toolkit or vendor library at build or run time).
//!
//! Three surfaces over the same kernels:
//! - Async launches over device addresses ([`launch_einsum`],
//!   [`launch_gemm`], [`launch_syev`], [`launch_gesvd`]) on the wide
//!   stream; the caller queues a whole step and syncs once with
//!   [`GpuPeer::sync_wide`].
//! - Synchronous host-buffer wrappers ([`einsum_batched`],
//!   [`gemm_batched`], [`syev_batched`], [`gesvd_batched`]) that pin,
//!   launch, sync, fetch and unpin.
//! - [`register_linalg_accel_ops`] + [`bind_linalg_kernels`]: the
//!   same kernels as [`crate::backend::accel_op`] ops with the CPU
//!   references in [`cpu`] as their CPU side, so
//!   [`crate::backend::accel_op::dispatch_accel`] learns placement.
//!
//! einsum and gemm match [`cpu`] bit for bit (fma in the same order);
//! the Jacobi ops agree to rounding (tournament vs cyclic pair order).
//! Matrices are row-major and batched contiguously.

use std::sync::{Arc, Mutex};

use cudarc::nvrtc::Ptx;

use crate::sched::hybrid::{SplitReport, hybrid_auto_split_ranges};

use super::{GpuPeer, GpuPeerError, WideKernel};
use crate::backend::accel_op::{AccelOpId, AccelReport, bind_accel_kernel, dispatch_accel, register_accel_op};
use crate::backend::{Backend, BackendError, KernelArg};
use crate::sched::plan::JobPlan;

/// PTX for every kernel in this module, generated from
/// `kernels/linalg_f64.cu` by `kernels/build_ptx.bat`.
pub const LINALG_PTX: &str = include_str!("../../kernels/linalg_f64.ptx");

/// The kernel source, NVRTC-compiled at load when the driver rejects
/// [`LINALG_PTX`] (a toolchain newer than the driver).
pub const LINALG_CU: &str = include_str!("../../kernels/linalg_f64.cu");
/// PTX of the tridiagonalization / bidiagonalization plus bisection
/// kernels (`kernels/linalg_bisect_f64.cu`), the second module of
/// [`LinalgKernels`].
pub const LINALG_BISECT_PTX: &str = include_str!("../../kernels/linalg_bisect_f64.ptx");
/// Source of [`LINALG_BISECT_PTX`] for the NVRTC fallback.
pub const LINALG_BISECT_CU: &str = include_str!("../../kernels/linalg_bisect_f64.cu");
/// PTX of the LU factorization and solve kernels
/// (`kernels/linalg_lu_f64.cu`), the third module of [`LinalgKernels`].
pub const LINALG_LU_PTX: &str = include_str!("../../kernels/linalg_lu_f64.ptx");
/// Source of [`LINALG_LU_PTX`] for the NVRTC fallback.
pub const LINALG_LU_CU: &str = include_str!("../../kernels/linalg_lu_f64.cu");

/// Largest square dimension the block-per-matrix Jacobi kernels take.
pub const LINALG_MAX_N: usize = 64;
/// Largest dimension the thread-per-matrix Jacobi kernels take.
pub const LINALG_THR_MAX_N: usize = 16;
/// Largest einsum rank per operand and contraction set.
pub const EINSUM_MAX_RANK: usize = 12;
/// Threads per block every kernel here is launched with.
pub const LINALG_BLOCK: u32 = 256;
/// Matrices per unit of `n` a batch needs before the thread-per-
/// matrix Jacobi kernels beat block-per-matrix. Measured by
/// `benches/gpu_linalg.rs` on RTX 3070 and RTX 5070 for syev and
/// gesvd alike: thr wins at n=4 from batch 1024, n=8 from 2048, n=16
/// from 4096; blk wins below (3x at n=16, batch 1024).
pub const JACOBI_THREAD_SHAPE_BATCH_PER_N: usize = 256;

/// Device-side parallelization of one matrix in the batched Jacobi
/// kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JacobiShape {
    /// One 256-thread block per matrix, matrix in shared memory,
    /// n/2 disjoint rotations per round.
    BlockPerMatrix,
    /// One thread per matrix, matrix in local memory, cyclic sweep;
    /// `n <= LINALG_THR_MAX_N` only.
    ThreadPerMatrix,
}

/// The faster shape for `batch` matrices of dimension `n` when the
/// caller does not pin one: thread-per-matrix needs enough matrices
/// to fill the device (`batch >= JACOBI_THREAD_SHAPE_BATCH_PER_N * n`)
/// and `n <= LINALG_THR_MAX_N`; block-per-matrix otherwise.
pub fn jacobi_shape_for(n: usize, batch: usize) -> JacobiShape {
    if n <= LINALG_THR_MAX_N && batch >= JACOBI_THREAD_SHAPE_BATCH_PER_N * n {
        JacobiShape::ThreadPerMatrix
    } else {
        JacobiShape::BlockPerMatrix
    }
}

/// Which batched kernel family the automatic host helpers use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinalgMethod {
    /// The Jacobi kernels (`syev_batched`, `gesvd_batched`), shape by
    /// [`jacobi_shape_for`].
    Jacobi,
    /// Householder reduction plus bisection (`syev_bisect_batched`,
    /// `gesvd_bisect_batched`).
    Bisection,
}

/// Smallest `n` at which bisection beats the Jacobi kernels for
/// symmetric eigenvalues: measured 1.4x to 1.8x at n = 32 and 4.0x at
/// n = 64 over the block Jacobi kernel on RTX 3070 and RTX 5070.
pub const SYEV_BISECT_MIN_N: usize = 32;
/// Smallest `n` at which bisection beats the Jacobi kernels for
/// singular values: 1.05x to 1.4x at n = 64 on the same hosts, behind
/// Jacobi at n = 32.
pub const GESVD_BISECT_MIN_N: usize = 64;

/// The measured choice for symmetric eigenvalues of dimension `n`.
pub fn syev_method_for(n: usize) -> LinalgMethod {
    if n >= SYEV_BISECT_MIN_N { LinalgMethod::Bisection } else { LinalgMethod::Jacobi }
}

/// The measured choice for singular values of `m x n` with `m >= n`.
pub fn gesvd_method_for(n: usize) -> LinalgMethod {
    if n >= GESVD_BISECT_MIN_N { LinalgMethod::Bisection } else { LinalgMethod::Jacobi }
}

/// Sweep cap the reference SVD tier uses: `4 * ceil(log2 n) + 8`.
pub fn default_sweeps(n: usize) -> u32 {
    let mut sweeps = 8u32;
    let mut nn = n.max(1);
    while nn > 1 {
        nn = nn.div_ceil(2);
        sweeps += 4;
    }
    sweeps
}

/// The six kernels of `linalg_f64.ptx`, loaded from one module.
pub struct LinalgKernels {
    /// `flynnel_einsum_f64`.
    pub einsum: WideKernel,
    /// `flynnel_gemm_batched_f64`.
    pub gemm: WideKernel,
    /// `flynnel_syev_jacobi_f64_blk`.
    pub syev_blk: WideKernel,
    /// `flynnel_syev_jacobi_f64_thr`.
    pub syev_thr: WideKernel,
    /// `flynnel_gesvd_jacobi_f64_blk`.
    pub gesvd_blk: WideKernel,
    /// `flynnel_gesvd_jacobi_f64_thr`.
    pub gesvd_thr: WideKernel,
    /// `flynnel_syev_bisect_f64_blk`.
    pub syev_bisect: WideKernel,
    /// `flynnel_gesvd_bisect_f64_blk`.
    pub gesvd_bisect: WideKernel,
    /// `flynnel_getrf_f64_blk`.
    pub getrf: WideKernel,
    /// `flynnel_getrs_f64_blk`.
    pub getrs: WideKernel,
}

/// Load one PTX module, compiling `cu` with NVRTC when the driver
/// rejects the checked-in text.
fn load_module(
    peer: &GpuPeer,
    what: &str,
    ptx: &str,
    cu: &str,
) -> Result<std::sync::Arc<cudarc::driver::CudaModule>, GpuPeerError> {
    match peer.context().load_module(Ptx::from_src(ptx)) {
        Ok(m) => Ok(m),
        Err(ptx_err) => {
            eprintln!(
                "flynnel gpu_peer {what}: checked-in PTX rejected ({ptx_err:?}); \
                 compiling the {what} kernels with NVRTC instead"
            );
            let ptx = cudarc::nvrtc::compile_ptx(cu).map_err(|e| {
                GpuPeerError::Driver(format!(
                    "{what} PTX load: {ptx_err:?}; NVRTC fallback compile: {e:?}"
                ))
            })?;
            peer.context()
                .load_module(ptx)
                .map_err(|e| GpuPeerError::Driver(format!("{what} NVRTC-fallback load: {e:?}")))
        }
    }
}

impl LinalgKernels {
    /// Load the three PTX modules into the peer's context once and
    /// resolve every entry point.
    pub fn load(peer: &GpuPeer) -> Result<Self, GpuPeerError> {
        let module = load_module(peer, "linalg", LINALG_PTX, LINALG_CU)?;
        let bisect = load_module(peer, "linalg_bisect", LINALG_BISECT_PTX, LINALG_BISECT_CU)?;
        let lu = load_module(peer, "linalg_lu", LINALG_LU_PTX, LINALG_LU_CU)?;
        let load = |module: &std::sync::Arc<cudarc::driver::CudaModule>,
                    entry: &str|
         -> Result<WideKernel, GpuPeerError> {
            let func = module
                .load_function(entry)
                .map_err(|e| GpuPeerError::Driver(format!("linalg entry `{entry}`: {e:?}")))?;
            Ok(WideKernel::new(module.clone(), func))
        };
        Ok(Self {
            einsum: load(&module, "flynnel_einsum_f64")?,
            gemm: load(&module, "flynnel_gemm_batched_f64")?,
            syev_blk: load(&module, "flynnel_syev_jacobi_f64_blk")?,
            syev_thr: load(&module, "flynnel_syev_jacobi_f64_thr")?,
            gesvd_blk: load(&module, "flynnel_gesvd_jacobi_f64_blk")?,
            gesvd_thr: load(&module, "flynnel_gesvd_jacobi_f64_thr")?,
            syev_bisect: load(&bisect, "flynnel_syev_bisect_f64_blk")?,
            gesvd_bisect: load(&bisect, "flynnel_gesvd_bisect_f64_blk")?,
            getrf: load(&lu, "flynnel_getrf_f64_blk")?,
            getrs: load(&lu, "flynnel_getrs_f64_blk")?,
        })
    }
}

// ------------------------------------------------------------ einsum spec

/// A malformed or inconsistent einsum subscript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EinsumError {
    /// Missing `->`, wrong operand count, or a non-letter subscript.
    BadSpec(String),
    /// An operand's subscript length differs from its shape's rank.
    RankMismatch {
        /// Operand index (0 = `a`, 1 = `b`).
        operand: usize,
        /// Subscript letters given for it.
        subscripts: usize,
        /// Rank of the shape given for it.
        rank: usize,
    },
    /// One letter carries two different extents.
    ExtentMismatch {
        /// The subscript letter.
        letter: char,
        /// Extent at its first appearance.
        first: usize,
        /// Conflicting extent at a later appearance.
        second: usize,
    },
    /// An output letter that no operand carries.
    UnknownOutputAxis(char),
    /// Rank or contraction count above [`EINSUM_MAX_RANK`].
    RankTooLarge(usize),
}

impl std::fmt::Display for EinsumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadSpec(s) => write!(f, "einsum: bad spec: {s}"),
            Self::RankMismatch { operand, subscripts, rank } => write!(
                f,
                "einsum: operand {operand} has {subscripts} subscripts but rank {rank}"
            ),
            Self::ExtentMismatch { letter, first, second } => write!(
                f,
                "einsum: axis `{letter}` has extent {first} and {second}"
            ),
            Self::UnknownOutputAxis(c) => write!(f, "einsum: output axis `{c}` not in any operand"),
            Self::RankTooLarge(r) => write!(f, "einsum: rank {r} exceeds {EINSUM_MAX_RANK}"),
        }
    }
}

impl std::error::Error for EinsumError {}

/// A parsed contraction: shapes plus the stride / kind tables the
/// kernel and the CPU reference walk. The tables live in one `i32`
/// buffer laid out `[a_strides | b_strides | o_strides | c_extents |
/// a_kind | b_kind]`; [`Self::table_offsets`] gives the six starts.
#[derive(Debug, Clone)]
pub struct EinsumSpec {
    a_shape: Vec<usize>,
    b_shape: Option<Vec<usize>>,
    out_shape: Vec<usize>,
    tables: Vec<i32>,
    offsets: [usize; 6],
    n_contract: usize,
}

impl EinsumSpec {
    /// Parse `"ij,jk->ik"`-style subscripts against the operand
    /// shapes. Letters absent from the output are contracted in
    /// order of first appearance; a letter repeated within one
    /// operand (`"ii->"`) is a diagonal.
    pub fn parse(
        spec: &str,
        a_shape: &[usize],
        b_shape: Option<&[usize]>,
    ) -> Result<Self, EinsumError> {
        let (lhs, rhs) = spec
            .split_once("->")
            .ok_or_else(|| EinsumError::BadSpec(format!("`{spec}` has no `->`")))?;
        let inputs: Vec<&str> = lhs.split(',').map(str::trim).collect();
        let rhs = rhs.trim();
        if inputs.len() != if b_shape.is_some() { 2 } else { 1 } {
            return Err(EinsumError::BadSpec(format!(
                "`{spec}` names {} operands, {} given",
                inputs.len(),
                if b_shape.is_some() { 2 } else { 1 }
            )));
        }
        for s in inputs.iter().chain(std::iter::once(&rhs)) {
            if let Some(c) = s.chars().find(|c| !c.is_ascii_alphabetic()) {
                return Err(EinsumError::BadSpec(format!("non-letter subscript `{c}`")));
            }
        }
        let shapes: Vec<&[usize]> = match b_shape {
            Some(b) => vec![a_shape, b],
            None => vec![a_shape],
        };
        let mut extent_of: Vec<(char, usize)> = Vec::new();
        for (oi, (sub, shape)) in inputs.iter().zip(&shapes).enumerate() {
            let subs: Vec<char> = sub.chars().collect();
            if subs.len() != shape.len() {
                return Err(EinsumError::RankMismatch {
                    operand: oi,
                    subscripts: subs.len(),
                    rank: shape.len(),
                });
            }
            if subs.len() > EINSUM_MAX_RANK {
                return Err(EinsumError::RankTooLarge(subs.len()));
            }
            for (c, &e) in subs.iter().zip(shape.iter()) {
                match extent_of.iter().find(|(l, _)| l == c) {
                    Some(&(_, first)) if first != e => {
                        return Err(EinsumError::ExtentMismatch { letter: *c, first, second: e });
                    }
                    Some(_) => {}
                    None => extent_of.push((*c, e)),
                }
            }
        }
        let out: Vec<char> = rhs.chars().collect();
        if out.len() > EINSUM_MAX_RANK {
            return Err(EinsumError::RankTooLarge(out.len()));
        }
        let mut out_shape = Vec::with_capacity(out.len());
        for c in &out {
            let (_, e) = extent_of
                .iter()
                .find(|(l, _)| l == c)
                .ok_or(EinsumError::UnknownOutputAxis(*c))?;
            out_shape.push(*e);
        }
        let contracted: Vec<(char, usize)> = extent_of
            .iter()
            .filter(|(l, _)| !out.contains(l))
            .copied()
            .collect();
        if contracted.len() > EINSUM_MAX_RANK {
            return Err(EinsumError::RankTooLarge(contracted.len()));
        }

        let row_major = |shape: &[usize]| -> Vec<i32> {
            let mut strides = vec![0i32; shape.len()];
            let mut acc = 1i32;
            for i in (0..shape.len()).rev() {
                strides[i] = acc;
                acc *= shape[i] as i32;
            }
            strides
        };
        let kinds = |sub: &str| -> Vec<i32> {
            sub.chars()
                .map(|c| match out.iter().position(|o| *o == c) {
                    Some(pos) => pos as i32,
                    None => {
                        let pos = contracted.iter().position(|(l, _)| *l == c).expect("contracted");
                        (1 << 16) | pos as i32
                    }
                })
                .collect()
        };

        let a_strides = row_major(a_shape);
        let b_strides = b_shape.map(row_major).unwrap_or_default();
        let o_strides = row_major(&out_shape);
        let c_extents: Vec<i32> = contracted.iter().map(|(_, e)| *e as i32).collect();
        let a_kind = kinds(inputs[0]);
        let b_kind = if b_shape.is_some() { kinds(inputs[1]) } else { Vec::new() };

        let mut tables = Vec::new();
        let mut offsets = [0usize; 6];
        for (i, part) in [&a_strides, &b_strides, &o_strides, &c_extents, &a_kind, &b_kind]
            .into_iter()
            .enumerate()
        {
            offsets[i] = tables.len();
            tables.extend_from_slice(part);
        }
        Ok(Self {
            a_shape: a_shape.to_vec(),
            b_shape: b_shape.map(<[usize]>::to_vec),
            out_shape,
            tables,
            offsets,
            n_contract: contracted.len(),
        })
    }

    /// Elements per output item.
    pub fn out_size(&self) -> usize {
        self.out_shape.iter().product()
    }
    /// Elements per `a` item.
    pub fn a_size(&self) -> usize {
        self.a_shape.iter().product()
    }
    /// Elements per `b` item; zero without a second operand.
    pub fn b_size(&self) -> usize {
        self.b_shape.as_ref().map_or(0, |s| s.iter().product())
    }
    /// Output shape.
    pub fn out_shape(&self) -> &[usize] {
        &self.out_shape
    }
    /// The packed `i32` table buffer.
    pub fn tables(&self) -> &[i32] {
        &self.tables
    }
    /// Element offsets of `[a_strides, b_strides, o_strides,
    /// c_extents, a_kind, b_kind]` within [`Self::tables`].
    pub fn table_offsets(&self) -> [usize; 6] {
        self.offsets
    }
    fn a_rank(&self) -> usize {
        self.a_shape.len()
    }
    fn b_rank(&self) -> usize {
        self.b_shape.as_ref().map_or(0, Vec::len)
    }
    fn has_b(&self) -> bool {
        self.b_shape.is_some()
    }
    fn part(&self, i: usize) -> &[i32] {
        let start = self.offsets[i];
        let end = if i + 1 < 6 { self.offsets[i + 1] } else { self.tables.len() };
        &self.tables[start..end]
    }
}

// ------------------------------------------------------------ launches
//
// The launch and host-buffer wrappers below take one parameter per
// kernel argument; clippy::too_many_arguments is allowed on each
// because the signature IS the kernel ABI and a parameter struct would
// only re-spell it.

pub(crate) fn check_dim(cond: bool, what: &'static str) -> Result<(), GpuPeerError> {
    if cond { Ok(()) } else { Err(GpuPeerError::Unavailable(what)) }
}

/// Enqueue a batched einsum on the wide stream. `tables_dev` is the
/// device copy of [`EinsumSpec::tables`]; `a`, `b`, `out` are device
/// addresses of `batch` contiguous items each (`b` is ignored for a
/// single-operand spec). No sync.
#[allow(clippy::too_many_arguments)]
pub fn launch_einsum(
    peer: &GpuPeer,
    k: &LinalgKernels,
    spec: &EinsumSpec,
    tables_dev: u64,
    a: u64,
    b: u64,
    out: u64,
    batch: u32,
) -> Result<(), GpuPeerError> {
    check_dim(batch > 0, "einsum: empty batch")?;
    let off = spec.table_offsets();
    let tp = |i: usize| tables_dev + (off[i] * 4) as u64;
    let ptrs = [out, a, if spec.has_b() { b } else { a }, tp(0), tp(1), tp(2), tp(3), tp(4), tp(5)];
    let o_size = spec.out_size() as u32;
    let scalars = [
        o_size,
        spec.a_rank() as u32,
        spec.b_rank() as u32,
        spec.out_shape.len() as u32,
        spec.n_contract as u32,
        u32::from(spec.has_b()),
        batch,
        spec.a_size() as u32,
        spec.b_size() as u32,
        o_size,
    ];
    let grid = (o_size as u64 * batch as u64).div_ceil(LINALG_BLOCK as u64).max(1) as u32;
    peer.launch_wide_async(&k.einsum, grid, LINALG_BLOCK, &ptrs, &scalars)
}

/// Enqueue batched row-major `C = A * B` (`m x k` times `k x n`,
/// contiguous items) on the wide stream. No sync.
#[allow(clippy::too_many_arguments)]
pub fn launch_gemm(
    peer: &GpuPeer,
    k: &LinalgKernels,
    a: u64,
    b: u64,
    c: u64,
    batch: u32,
    m: u32,
    n: u32,
    kdim: u32,
) -> Result<(), GpuPeerError> {
    check_dim(batch > 0 && m > 0 && n > 0 && kdim > 0, "gemm: empty dimension")?;
    let tiles = m.div_ceil(16) * n.div_ceil(16);
    let grid = tiles * batch;
    peer.launch_wide_async(
        &k.gemm,
        grid,
        LINALG_BLOCK,
        &[a, b, c],
        &[batch, m, n, kdim, kdim, n, n],
    )
}

/// Enqueue batched symmetric eigendecomposition on the wide stream:
/// `a` holds `batch` row-major `n x n` matrices (read only), `w`
/// receives `batch * n` eigenvalues in diagonal order, `v` (when
/// `Some`) receives eigenvectors as columns. No sync.
#[allow(clippy::too_many_arguments)]
pub fn launch_syev(
    peer: &GpuPeer,
    k: &LinalgKernels,
    a: u64,
    w: u64,
    v: Option<u64>,
    batch: u32,
    n: u32,
    max_sweeps: u32,
    shape: JacobiShape,
) -> Result<(), GpuPeerError> {
    check_dim(batch > 0 && n > 0, "syev: empty batch or n")?;
    check_dim(n as usize <= LINALG_MAX_N, "syev: n exceeds LINALG_MAX_N")?;
    let scalars = [batch, n, max_sweeps, u32::from(v.is_some())];
    let ptrs = [a, w, v.unwrap_or(0)];
    match shape {
        JacobiShape::BlockPerMatrix => {
            peer.launch_wide_async(&k.syev_blk, batch, LINALG_BLOCK, &ptrs, &scalars)
        }
        JacobiShape::ThreadPerMatrix => {
            check_dim(n as usize <= LINALG_THR_MAX_N, "syev: n exceeds LINALG_THR_MAX_N")?;
            let grid = batch.div_ceil(LINALG_BLOCK);
            peer.launch_wide_async(&k.syev_thr, grid, LINALG_BLOCK, &ptrs, &scalars)
        }
    }
}

/// Enqueue batched one-sided Jacobi SVD on the wide stream: `a`
/// holds `batch` row-major `m x n` matrices (`m >= n`) and is
/// overwritten with `U`; `sigma` receives `batch * n` singular values
/// in column order; `v` (when `Some`) receives `V`. No sync.
#[allow(clippy::too_many_arguments)]
pub fn launch_gesvd(
    peer: &GpuPeer,
    k: &LinalgKernels,
    a: u64,
    sigma: u64,
    v: Option<u64>,
    batch: u32,
    m: u32,
    n: u32,
    max_sweeps: u32,
    shape: JacobiShape,
) -> Result<(), GpuPeerError> {
    check_dim(batch > 0 && n > 0 && m >= n, "gesvd: needs batch > 0 and m >= n > 0")?;
    check_dim(m as usize <= LINALG_MAX_N, "gesvd: m exceeds LINALG_MAX_N")?;
    let scalars = [batch, m, n, max_sweeps, u32::from(v.is_some())];
    let ptrs = [a, sigma, v.unwrap_or(0)];
    match shape {
        JacobiShape::BlockPerMatrix => {
            peer.launch_wide_async(&k.gesvd_blk, batch, LINALG_BLOCK, &ptrs, &scalars)
        }
        JacobiShape::ThreadPerMatrix => {
            check_dim(m as usize <= LINALG_THR_MAX_N, "gesvd: m exceeds LINALG_THR_MAX_N")?;
            let grid = batch.div_ceil(LINALG_BLOCK);
            peer.launch_wide_async(&k.gesvd_thr, grid, LINALG_BLOCK, &ptrs, &scalars)
        }
    }
}

// ------------------------------------------------------------ host-buffer wrappers

pub(crate) fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn bytes_f64(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A pinned span plus its device address; unpinned on drop-by-hand
/// through [`Self::release`].
pub(crate) struct Pinned {
    pub(crate) handle: super::ResidentHandle,
    pub(crate) ptr: u64,
}

pub(crate) fn pin(peer: &mut GpuPeer, bytes: &[u8]) -> Result<Pinned, GpuPeerError> {
    let handle = peer.pin_bulk(bytes)?;
    let (ptr, _) = peer.resident_ptr(&handle)?;
    Ok(Pinned { handle, ptr })
}

impl Pinned {
    pub(crate) fn release(self, peer: &mut GpuPeer) -> Result<(), GpuPeerError> {
        peer.unpin(self.handle)
    }
}

fn fetch_f64(peer: &mut GpuPeer, p: &Pinned, elems: usize) -> Result<Vec<f64>, GpuPeerError> {
    let mut out = vec![0u8; elems * 8];
    peer.fetch_bulk(&p.handle, &mut out)?;
    Ok(bytes_f64(&out))
}

/// Batched einsum over host buffers: `a` holds `batch * a_size`
/// elements, `b` `batch * b_size`; returns `batch * out_size`.
pub fn einsum_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    spec: &EinsumSpec,
    a: &[f64],
    b: Option<&[f64]>,
    batch: u32,
) -> Result<Vec<f64>, GpuPeerError> {
    let batch_us = batch as usize;
    check_dim(a.len() == batch_us * spec.a_size(), "einsum: a length")?;
    check_dim(
        b.map_or(!spec.has_b(), |b| spec.has_b() && b.len() == batch_us * spec.b_size()),
        "einsum: b length",
    )?;
    let out_elems = batch_us * spec.out_size();
    let tables = pin(peer, &i32_bytes(spec.tables()))?;
    let pa = pin(peer, &f64_bytes(a))?;
    let pb = match b {
        Some(b) => Some(pin(peer, &f64_bytes(b))?),
        None => None,
    };
    let po = pin(peer, &vec![0u8; out_elems * 8])?;
    launch_einsum(
        peer,
        k,
        spec,
        tables.ptr,
        pa.ptr,
        pb.as_ref().map_or(pa.ptr, |p| p.ptr),
        po.ptr,
        batch,
    )?;
    peer.sync_wide()?;
    let out = fetch_f64(peer, &po, out_elems)?;
    po.release(peer)?;
    if let Some(pb) = pb {
        pb.release(peer)?;
    }
    pa.release(peer)?;
    tables.release(peer)?;
    Ok(out)
}

/// Batched GEMM over host buffers: `a` holds `batch` `m x k` items,
/// `b` `batch` `k x n` items; returns `batch` `m x n` items.
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    a: &[f64],
    b: &[f64],
    batch: u32,
    m: u32,
    n: u32,
    kdim: u32,
) -> Result<Vec<f64>, GpuPeerError> {
    let (bu, mu, nu, ku) = (batch as usize, m as usize, n as usize, kdim as usize);
    check_dim(a.len() == bu * mu * ku && b.len() == bu * ku * nu, "gemm: operand length")?;
    let pa = pin(peer, &f64_bytes(a))?;
    let pb = pin(peer, &f64_bytes(b))?;
    let pc = pin(peer, &vec![0u8; bu * mu * nu * 8])?;
    launch_gemm(peer, k, pa.ptr, pb.ptr, pc.ptr, batch, m, n, kdim)?;
    peer.sync_wide()?;
    let out = fetch_f64(peer, &pc, bu * mu * nu)?;
    pc.release(peer)?;
    pb.release(peer)?;
    pa.release(peer)?;
    Ok(out)
}

/// Workspace bytes [`launch_syev_bisect`] needs for `batch` matrices
/// of dimension `n` when eigenvectors are wanted (`n x n` per matrix).
pub fn syev_bisect_scratch_bytes(batch: u32, n: u32) -> usize {
    batch as usize * n as usize * n as usize * 8
}

/// Workspace bytes [`launch_gesvd_bisect`] needs for `batch` `m x n`
/// matrices (`3 n x n + m x n` per matrix, always).
pub fn gesvd_bisect_scratch_bytes(batch: u32, m: u32, n: u32) -> usize {
    let (bu, mu, nu) = (batch as usize, m as usize, n as usize);
    bu * (3 * nu * nu + mu * nu) * 8
}

/// Enqueue batched symmetric eigendecomposition by tridiagonalization
/// and bisection on the wide stream: `a` holds `batch` row-major
/// `n x n` matrices (read only), `w` receives `batch * n` eigenvalues
/// in ascending order, `v` (when `Some`) receives eigenvectors as
/// columns and needs `scratch` of [`syev_bisect_scratch_bytes`]. One
/// block per matrix, `n <= LINALG_MAX_N`. No sync.
#[allow(clippy::too_many_arguments)]
pub fn launch_syev_bisect(
    peer: &GpuPeer,
    k: &LinalgKernels,
    a: u64,
    w: u64,
    v: Option<u64>,
    scratch: u64,
    batch: u32,
    n: u32,
) -> Result<(), GpuPeerError> {
    check_dim(batch > 0 && n > 0, "syev_bisect: empty batch or n")?;
    check_dim(n as usize <= LINALG_MAX_N, "syev_bisect: n exceeds LINALG_MAX_N")?;
    check_dim(v.is_none() || scratch != 0, "syev_bisect: eigenvectors need a scratch buffer")?;
    let scalars = [batch, n, u32::from(v.is_some())];
    let ptrs = [a, w, v.unwrap_or(0), scratch];
    peer.launch_wide_async(&k.syev_bisect, batch, LINALG_BLOCK, &ptrs, &scalars)
}

/// [`launch_syev_bisect`] over host buffers: returns `(eigenvalues
/// ascending, eigenvectors as columns when want_v)`.
pub fn syev_bisect_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    a: &[f64],
    batch: u32,
    n: u32,
    want_v: bool,
) -> Result<(Vec<f64>, Option<Vec<f64>>), GpuPeerError> {
    let (bu, nu) = (batch as usize, n as usize);
    check_dim(a.len() == bu * nu * nu, "syev_bisect: a length")?;
    let pa = pin(peer, &f64_bytes(a))?;
    let pw = pin(peer, &vec![0u8; bu * nu * 8])?;
    let pv = if want_v { Some(pin(peer, &vec![0u8; bu * nu * nu * 8])?) } else { None };
    let scratch = if want_v {
        Some(pin(peer, &vec![0u8; syev_bisect_scratch_bytes(batch, n)])?)
    } else {
        None
    };
    launch_syev_bisect(
        peer,
        k,
        pa.ptr,
        pw.ptr,
        pv.as_ref().map(|p| p.ptr),
        scratch.as_ref().map_or(0, |p| p.ptr),
        batch,
        n,
    )?;
    peer.sync_wide()?;
    let w = fetch_f64(peer, &pw, bu * nu)?;
    let v = match &pv {
        Some(p) => Some(fetch_f64(peer, p, bu * nu * nu)?),
        None => None,
    };
    if let Some(p) = scratch {
        p.release(peer)?;
    }
    if let Some(p) = pv {
        p.release(peer)?;
    }
    pw.release(peer)?;
    pa.release(peer)?;
    Ok((w, v))
}

/// Enqueue batched SVD by bidiagonalization and bisection on the
/// Golub-Kahan tridiagonal on the wide stream: `a` holds `batch`
/// row-major `m x n` matrices with `m >= n` and is overwritten with
/// `U`, `sigma` receives `batch * n` singular values descending, `v`
/// (when `Some`) the right singular vectors; `scratch` is
/// [`gesvd_bisect_scratch_bytes`] and always required. One block per
/// matrix, `m <= LINALG_MAX_N`. No sync.
#[allow(clippy::too_many_arguments)]
pub fn launch_gesvd_bisect(
    peer: &GpuPeer,
    k: &LinalgKernels,
    a: u64,
    sigma: u64,
    v: Option<u64>,
    scratch: u64,
    batch: u32,
    m: u32,
    n: u32,
) -> Result<(), GpuPeerError> {
    check_dim(batch > 0 && n > 0 && m >= n, "gesvd_bisect: empty batch, empty n, or m < n")?;
    check_dim(m as usize <= LINALG_MAX_N, "gesvd_bisect: m exceeds LINALG_MAX_N")?;
    check_dim(scratch != 0, "gesvd_bisect: scratch buffer required")?;
    let scalars = [batch, m, n, u32::from(v.is_some())];
    let ptrs = [a, sigma, v.unwrap_or(0), scratch];
    peer.launch_wide_async(&k.gesvd_bisect, batch, LINALG_BLOCK, &ptrs, &scalars)
}

/// [`launch_gesvd_bisect`] over host buffers; singular values
/// descending.
#[allow(clippy::too_many_arguments)]
pub fn gesvd_bisect_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    a: &[f64],
    batch: u32,
    m: u32,
    n: u32,
    want_v: bool,
) -> Result<GesvdResult, GpuPeerError> {
    let (bu, mu, nu) = (batch as usize, m as usize, n as usize);
    check_dim(a.len() == bu * mu * nu, "gesvd_bisect: a length")?;
    let pa = pin(peer, &f64_bytes(a))?;
    let ps = pin(peer, &vec![0u8; bu * nu * 8])?;
    let pv = if want_v { Some(pin(peer, &vec![0u8; bu * nu * nu * 8])?) } else { None };
    let scratch = pin(peer, &vec![0u8; gesvd_bisect_scratch_bytes(batch, m, n)])?;
    launch_gesvd_bisect(
        peer,
        k,
        pa.ptr,
        ps.ptr,
        pv.as_ref().map(|p| p.ptr),
        scratch.ptr,
        batch,
        m,
        n,
    )?;
    peer.sync_wide()?;
    let u = fetch_f64(peer, &pa, bu * mu * nu)?;
    let sigma = fetch_f64(peer, &ps, bu * nu)?;
    let v = match &pv {
        Some(p) => Some(fetch_f64(peer, p, bu * nu * nu)?),
        None => None,
    };
    scratch.release(peer)?;
    if let Some(p) = pv {
        p.release(peer)?;
    }
    ps.release(peer)?;
    pa.release(peer)?;
    Ok(GesvdResult { u, sigma, v })
}

/// Batched symmetric eigendecomposition over host buffers by the
/// measured method for `n` ([`syev_method_for`]): bisection returns
/// eigenvalues ascending, Jacobi in diagonal order.
pub fn syev_auto_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    a: &[f64],
    batch: u32,
    n: u32,
    want_v: bool,
) -> Result<(Vec<f64>, Option<Vec<f64>>), GpuPeerError> {
    match syev_method_for(n as usize) {
        LinalgMethod::Bisection => syev_bisect_batched(peer, k, a, batch, n, want_v),
        LinalgMethod::Jacobi => syev_batched(
            peer,
            k,
            a,
            batch,
            n,
            default_sweeps(n as usize),
            want_v,
            jacobi_shape_for(n as usize, batch as usize),
        ),
    }
}

/// Batched SVD over host buffers by the measured method for `n`
/// ([`gesvd_method_for`]): bisection returns singular values
/// descending, Jacobi in column order.
#[allow(clippy::too_many_arguments)]
pub fn gesvd_auto_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    a: &[f64],
    batch: u32,
    m: u32,
    n: u32,
    want_v: bool,
) -> Result<GesvdResult, GpuPeerError> {
    match gesvd_method_for(n as usize) {
        LinalgMethod::Bisection => gesvd_bisect_batched(peer, k, a, batch, m, n, want_v),
        LinalgMethod::Jacobi => gesvd_batched(
            peer,
            k,
            a,
            batch,
            m,
            n,
            default_sweeps(n as usize),
            want_v,
            jacobi_shape_for(m as usize, batch as usize),
        ),
    }
}

// ------------------------------------------------------------ LU: factor, solve, inverse

/// A batched LU factorization: `lu` holds `U` on and above the
/// diagonal and the unit-lower multipliers below it, `piv[item * n +
/// k]` the row swapped with row `k` at step `k`, and `info[item]` zero
/// or one past the first step whose pivot was exactly zero.
#[derive(Clone, Debug, PartialEq)]
pub struct LuResult {
    /// Packed factors, `batch * n * n`.
    pub lu: Vec<f64>,
    /// Row interchanges, `batch * n`.
    pub piv: Vec<i32>,
    /// Singularity flags, `batch`.
    pub info: Vec<i32>,
}

fn fetch_i32(peer: &mut GpuPeer, p: &Pinned, elems: usize) -> Result<Vec<i32>, GpuPeerError> {
    let mut out = vec![0u8; elems * 4];
    peer.fetch_bulk(&p.handle, &mut out)?;
    Ok(out.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Enqueue batched in-place LU with partial pivoting on the wide
/// stream: `a` holds `batch` row-major `n x n` matrices and receives
/// the packed factors, `piv` `batch * n` `i32` interchanges, `info`
/// `batch` `i32` flags (see [`LuResult`]). One block per matrix,
/// `n <= LINALG_MAX_N`. No sync.
pub fn launch_getrf(
    peer: &GpuPeer,
    k: &LinalgKernels,
    a: u64,
    piv: u64,
    info: u64,
    batch: u32,
    n: u32,
) -> Result<(), GpuPeerError> {
    check_dim(batch > 0 && n > 0, "getrf: empty batch or n")?;
    check_dim(n as usize <= LINALG_MAX_N, "getrf: n exceeds LINALG_MAX_N")?;
    peer.launch_wide_async(&k.getrf, batch, LINALG_BLOCK, &[a, piv, info], &[batch, n])
}

/// Enqueue batched solves with packed factors from [`launch_getrf`]
/// on the wide stream: `b` holds `batch` row-major `n x nrhs`
/// right-hand sides and receives the solutions. With `identity_rhs`,
/// `b` is not read, `nrhs` must equal `n`, and `b` receives the
/// inverse. `nrhs <= LINALG_MAX_N`. No sync.
#[allow(clippy::too_many_arguments)]
pub fn launch_getrs(
    peer: &GpuPeer,
    k: &LinalgKernels,
    lu: u64,
    piv: u64,
    b: u64,
    batch: u32,
    n: u32,
    nrhs: u32,
    identity_rhs: bool,
) -> Result<(), GpuPeerError> {
    check_dim(batch > 0 && n > 0 && nrhs > 0, "getrs: empty batch, n, or nrhs")?;
    check_dim(n as usize <= LINALG_MAX_N, "getrs: n exceeds LINALG_MAX_N")?;
    check_dim(nrhs as usize <= LINALG_MAX_N, "getrs: nrhs exceeds LINALG_MAX_N")?;
    check_dim(!identity_rhs || nrhs == n, "getrs: the identity right-hand side needs nrhs == n")?;
    let scalars = [batch, n, nrhs, u32::from(identity_rhs)];
    peer.launch_wide_async(&k.getrs, batch, LINALG_BLOCK, &[lu, piv, b], &scalars)
}

/// [`launch_getrf`] over host buffers.
pub fn getrf_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    a: &[f64],
    batch: u32,
    n: u32,
) -> Result<LuResult, GpuPeerError> {
    let (bu, nu) = (batch as usize, n as usize);
    check_dim(a.len() == bu * nu * nu, "getrf: a length")?;
    let pa = pin(peer, &f64_bytes(a))?;
    let ppiv = pin(peer, &vec![0u8; bu * nu * 4])?;
    let pinfo = pin(peer, &vec![0u8; bu * 4])?;
    launch_getrf(peer, k, pa.ptr, ppiv.ptr, pinfo.ptr, batch, n)?;
    peer.sync_wide()?;
    let lu = fetch_f64(peer, &pa, bu * nu * nu)?;
    let piv = fetch_i32(peer, &ppiv, bu * nu)?;
    let info = fetch_i32(peer, &pinfo, bu)?;
    pinfo.release(peer)?;
    ppiv.release(peer)?;
    pa.release(peer)?;
    Ok(LuResult { lu, piv, info })
}

/// [`launch_getrs`] over host buffers: the solutions of `batch`
/// systems with `nrhs` right-hand sides each.
#[allow(clippy::too_many_arguments)]
pub fn getrs_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    lu: &[f64],
    piv: &[i32],
    b: &[f64],
    batch: u32,
    n: u32,
    nrhs: u32,
) -> Result<Vec<f64>, GpuPeerError> {
    let (bu, nu, ru) = (batch as usize, n as usize, nrhs as usize);
    check_dim(lu.len() == bu * nu * nu, "getrs: lu length")?;
    check_dim(piv.len() == bu * nu, "getrs: piv length")?;
    check_dim(b.len() == bu * nu * ru, "getrs: b length")?;
    let plu = pin(peer, &f64_bytes(lu))?;
    let ppiv = pin(peer, &i32_bytes(piv))?;
    let pb = pin(peer, &f64_bytes(b))?;
    launch_getrs(peer, k, plu.ptr, ppiv.ptr, pb.ptr, batch, n, nrhs, false)?;
    peer.sync_wide()?;
    let x = fetch_f64(peer, &pb, bu * nu * ru)?;
    pb.release(peer)?;
    ppiv.release(peer)?;
    plu.release(peer)?;
    Ok(x)
}

/// [`launch_getrs`] with the identity right-hand side over host
/// buffers: the inverses, `batch * n * n`.
pub fn getri_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    lu: &[f64],
    piv: &[i32],
    batch: u32,
    n: u32,
) -> Result<Vec<f64>, GpuPeerError> {
    let (bu, nu) = (batch as usize, n as usize);
    check_dim(lu.len() == bu * nu * nu, "getri: lu length")?;
    check_dim(piv.len() == bu * nu, "getri: piv length")?;
    let plu = pin(peer, &f64_bytes(lu))?;
    let ppiv = pin(peer, &i32_bytes(piv))?;
    let pinv = pin(peer, &vec![0u8; bu * nu * nu * 8])?;
    launch_getrs(peer, k, plu.ptr, ppiv.ptr, pinv.ptr, batch, n, n, true)?;
    peer.sync_wide()?;
    let inv = fetch_f64(peer, &pinv, bu * nu * nu)?;
    pinv.release(peer)?;
    ppiv.release(peer)?;
    plu.release(peer)?;
    Ok(inv)
}

/// Determinants from packed factors: the product of the diagonal of
/// `U`, negated once per row interchange. `batch` values.
pub fn lu_det_batched(lu: &[f64], piv: &[i32], batch: usize, n: usize) -> Vec<f64> {
    (0..batch)
        .map(|item| {
            let f = &lu[item * n * n..(item + 1) * n * n];
            let pv = &piv[item * n..(item + 1) * n];
            let mut det = 1.0;
            for k in 0..n {
                det *= f[k * n + k];
                if pv[k] as usize != k {
                    det = -det;
                }
            }
            det
        })
        .collect()
}

// ------------------------------------------------------------ tandem: device + CPU pool

/// Most matrices per CPU work item in the tandem helpers. The CPU
/// share is walked in runs sized to give the pool two runs per
/// worker, capped here so an item stays a few microseconds of work
/// or more and is dispatched rather than probed inline.
pub const TANDEM_CPU_CHUNK: usize = 64;

/// Sort each item's eigenpairs ascending by eigenvalue in place:
/// `w` is `batch * n`, `v` (when given) holds eigenvectors as
/// columns of `n x n` items.
pub fn sort_eigenpairs_ascending(w: &mut [f64], v: Option<&mut [f64]>, n: usize) {
    if n == 0 {
        return;
    }
    let batch = w.len() / n;
    let mut perm: Vec<usize> = Vec::with_capacity(n);
    let mut row = vec![0f64; n];
    let mut v = v;
    for item in 0..batch {
        let wi = &mut w[item * n..(item + 1) * n];
        perm.clear();
        perm.extend(0..n);
        perm.sort_by(|&x, &y| wi[x].partial_cmp(&wi[y]).expect("finite eigenvalue"));
        let sorted: Vec<f64> = perm.iter().map(|&j| wi[j]).collect();
        wi.copy_from_slice(&sorted);
        if let Some(v) = v.as_deref_mut() {
            let vi = &mut v[item * n * n..(item + 1) * n * n];
            for r in 0..n {
                row.copy_from_slice(&vi[r * n..(r + 1) * n]);
                for (j, &p) in perm.iter().enumerate() {
                    vi[r * n + j] = row[p];
                }
            }
        }
    }
}

/// Sort each item's singular triplets descending by singular value
/// in place: `sigma` is `batch * n`, `u` holds `m x n` items with
/// left vectors as columns, `v` (when given) `n x n` items with
/// right vectors as columns.
pub fn sort_singular_descending(u: &mut [f64], sigma: &mut [f64], v: Option<&mut [f64]>, m: usize, n: usize) {
    if n == 0 {
        return;
    }
    let batch = sigma.len() / n;
    let mut perm: Vec<usize> = Vec::with_capacity(n);
    let mut row = vec![0f64; n];
    let mut v = v;
    for item in 0..batch {
        let si = &mut sigma[item * n..(item + 1) * n];
        perm.clear();
        perm.extend(0..n);
        perm.sort_by(|&x, &y| si[y].partial_cmp(&si[x]).expect("finite singular value"));
        let sorted: Vec<f64> = perm.iter().map(|&j| si[j]).collect();
        si.copy_from_slice(&sorted);
        let ui = &mut u[item * m * n..(item + 1) * m * n];
        for r in 0..m {
            row.copy_from_slice(&ui[r * n..(r + 1) * n]);
            for (j, &p) in perm.iter().enumerate() {
                ui[r * n + j] = row[p];
            }
        }
        if let Some(v) = v.as_deref_mut() {
            let vi = &mut v[item * n * n..(item + 1) * n * n];
            for r in 0..n {
                row.copy_from_slice(&vi[r * n..(r + 1) * n]);
                for (j, &p) in perm.iter().enumerate() {
                    vi[r * n + j] = row[p];
                }
            }
        }
    }
}

/// Walk `range` on the pool in runs of at most [`TANDEM_CPU_CHUNK`]
/// items, two runs per worker when the range allows, calling
/// `f(lo, hi)` once per run with disjoint item ranges.
fn tandem_cpu_runs<F>(range: std::ops::Range<usize>, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    let workers = crate::sched::arena::global_local_arena().total_workers().max(1);
    let width = range.len().div_ceil(2 * workers).clamp(1, TANDEM_CPU_CHUNK);
    let runs = range.len().div_ceil(width);
    let (start, end) = (range.start, range.end);
    // Every run is a batch of dense kernels, tens of microseconds to
    // hundreds of milliseconds: a latency-bound plan dispatches all
    // runs at once instead of timing one inline first.
    let plan = JobPlan::set_profile(0, runs as u32, crate::DispatchProfile::LatencyBound);
    crate::sched::par_iter::for_each_indexed(&plan, runs, 1, |ri| {
        let lo = start + ri * width;
        let hi = (lo + width).min(end);
        f(lo, hi);
    });
}

/// Copy `src` into `dst` at element offset `at`, where `dst` is the
/// address of a buffer the caller keeps alive and only ever writes
/// through disjoint offsets from concurrent runs.
///
/// # Safety
///
/// `dst..dst + (at + src.len()) * 8` must be inside a live `[f64]`
/// no other run writes at these offsets while this runs.
unsafe fn scatter(dst: usize, at: usize, src: &[f64]) {
    // SAFETY: the caller's contract above.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), (dst as *mut f64).add(at), src.len()) };
}

/// Batched GEMM over host buffers with the batch split between the
/// device and the CPU pool by the call site's learned share
/// ([`hybrid_auto_split_ranges`]): the device part runs
/// [`gemm_batched`], the CPU part [`cpu::gemm_batched`] in runs of
/// [`TANDEM_CPU_CHUNK`] matrices on the pool. Every item is the same
/// bit for bit whichever side computed it. Returns the result and
/// the split report.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn gemm_tandem_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    plan: &JobPlan,
    a: &[f64],
    b: &[f64],
    batch: u32,
    m: u32,
    n: u32,
    kdim: u32,
) -> Result<(Vec<f64>, SplitReport), GpuPeerError> {
    let (bu, mu, nu, ku) = (batch as usize, m as usize, n as usize, kdim as usize);
    check_dim(batch > 0 && m > 0 && n > 0 && kdim > 0, "gemm: empty dimension")?;
    check_dim(a.len() == bu * mu * ku, "gemm: a length")?;
    check_dim(b.len() == bu * ku * nu, "gemm: b length")?;
    let (per_a, per_b, per_c) = (mu * ku, ku * nu, mu * nu);
    let mut out = vec![0f64; bu * per_c];
    let out_addr = out.as_mut_ptr() as usize;
    let device_err: Arc<Mutex<Option<GpuPeerError>>> = Arc::new(Mutex::new(None));
    let err_slot = Arc::clone(&device_err);
    let (peer_addr, k_addr) = (peer as *mut GpuPeer as usize, k as *const LinalgKernels as usize);
    let (a_addr, a_len, b_addr, b_len) = (a.as_ptr() as usize, a.len(), b.as_ptr() as usize, b.len());
    let report = hybrid_auto_split_ranges(
        plan,
        bu,
        |r| {
            tandem_cpu_runs(r, |lo, hi| {
                let c = cpu::gemm_batched(&a[lo * per_a..hi * per_a], &b[lo * per_b..hi * per_b], hi - lo, mu, nu, ku);
                // SAFETY: `out` outlives the split; runs write disjoint item ranges.
                unsafe { scatter(out_addr, lo * per_c, &c) };
            });
        },
        move |r| {
            // SAFETY: the caller frame blocks inside the split until this
            // closure returns, so the peer, the kernels, the inputs and
            // `out` outlive every access here; the device item range is
            // disjoint from the CPU range.
            let peer = unsafe { &mut *(peer_addr as *mut GpuPeer) };
            let k = unsafe { &*(k_addr as *const LinalgKernels) };
            let a = unsafe { std::slice::from_raw_parts(a_addr as *const f64, a_len) };
            let b = unsafe { std::slice::from_raw_parts(b_addr as *const f64, b_len) };
            let (lo, hi) = (r.start, r.end);
            match gemm_batched(peer, k, &a[lo * per_a..hi * per_a], &b[lo * per_b..hi * per_b], (hi - lo) as u32, m, n, kdim) {
                Ok(c) => unsafe { scatter(out_addr, lo * per_c, &c) },
                Err(e) => *err_slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(e),
            }
        },
    );
    if let Some(e) = device_err.lock().unwrap_or_else(|p| p.into_inner()).take() {
        return Err(e);
    }
    Ok((out, report))
}

/// Batched symmetric eigendecomposition over host buffers with the
/// batch split between the device ([`syev_auto_batched`]) and the
/// CPU pool ([`cpu::syev_jacobi_batched`]) by the call site's learned
/// share. Eigenvalues come back ascending for every item, with
/// eigenvectors as columns when `want_v`, whichever side computed it.
#[track_caller]
pub fn syev_tandem_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    plan: &JobPlan,
    a: &[f64],
    batch: u32,
    n: u32,
    want_v: bool,
) -> Result<((Vec<f64>, Option<Vec<f64>>), SplitReport), GpuPeerError> {
    let (bu, nu) = (batch as usize, n as usize);
    check_dim(batch > 0 && n > 0, "syev: empty batch or n")?;
    check_dim(a.len() == bu * nu * nu, "syev: a length")?;
    let mut w = vec![0f64; bu * nu];
    let mut v = if want_v { vec![0f64; bu * nu * nu] } else { Vec::new() };
    let (w_addr, v_addr) = (w.as_mut_ptr() as usize, v.as_mut_ptr() as usize);
    let device_err: Arc<Mutex<Option<GpuPeerError>>> = Arc::new(Mutex::new(None));
    let err_slot = Arc::clone(&device_err);
    let (peer_addr, k_addr) = (peer as *mut GpuPeer as usize, k as *const LinalgKernels as usize);
    let (a_addr, a_len) = (a.as_ptr() as usize, a.len());
    let sweeps = default_sweeps(nu);
    let report = hybrid_auto_split_ranges(
        plan,
        bu,
        |r| {
            tandem_cpu_runs(r, |lo, hi| {
                let (mut wc, mut vc) = cpu::syev_jacobi_batched(&a[lo * nu * nu..hi * nu * nu], hi - lo, nu, sweeps, want_v);
                sort_eigenpairs_ascending(&mut wc, vc.as_deref_mut(), nu);
                // SAFETY: outputs outlive the split; runs write disjoint item ranges.
                unsafe { scatter(w_addr, lo * nu, &wc) };
                if let Some(vc) = vc {
                    unsafe { scatter(v_addr, lo * nu * nu, &vc) };
                }
            });
        },
        move |r| {
            // SAFETY: as in gemm_tandem_batched; the caller blocks until
            // this returns and the device range is disjoint from the CPU's.
            let peer = unsafe { &mut *(peer_addr as *mut GpuPeer) };
            let k = unsafe { &*(k_addr as *const LinalgKernels) };
            let a = unsafe { std::slice::from_raw_parts(a_addr as *const f64, a_len) };
            let (lo, hi) = (r.start, r.end);
            match syev_auto_batched(peer, k, &a[lo * nu * nu..hi * nu * nu], (hi - lo) as u32, n, want_v) {
                Ok((mut wd, mut vd)) => {
                    sort_eigenpairs_ascending(&mut wd, vd.as_deref_mut(), nu);
                    unsafe { scatter(w_addr, lo * nu, &wd) };
                    if let Some(vd) = vd {
                        unsafe { scatter(v_addr, lo * nu * nu, &vd) };
                    }
                }
                Err(e) => *err_slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(e),
            }
        },
    );
    if let Some(e) = device_err.lock().unwrap_or_else(|p| p.into_inner()).take() {
        return Err(e);
    }
    Ok(((w, if want_v { Some(v) } else { None }), report))
}

/// Batched SVD over host buffers with the batch split between the
/// device ([`gesvd_auto_batched`]) and the CPU pool
/// ([`cpu::gesvd_jacobi_batched`]) by the call site's learned share.
/// Singular values come back descending for every item, with the
/// matching columns of `U` and `V`, whichever side computed it.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn gesvd_tandem_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    plan: &JobPlan,
    a: &[f64],
    batch: u32,
    m: u32,
    n: u32,
    want_v: bool,
) -> Result<(GesvdResult, SplitReport), GpuPeerError> {
    let (bu, mu, nu) = (batch as usize, m as usize, n as usize);
    check_dim(batch > 0 && n > 0 && m >= n, "gesvd: empty batch, empty n, or m < n")?;
    check_dim(a.len() == bu * mu * nu, "gesvd: a length")?;
    let mut u = vec![0f64; bu * mu * nu];
    let mut sigma = vec![0f64; bu * nu];
    let mut v = if want_v { vec![0f64; bu * nu * nu] } else { Vec::new() };
    let (u_addr, s_addr, v_addr) = (u.as_mut_ptr() as usize, sigma.as_mut_ptr() as usize, v.as_mut_ptr() as usize);
    let device_err: Arc<Mutex<Option<GpuPeerError>>> = Arc::new(Mutex::new(None));
    let err_slot = Arc::clone(&device_err);
    let (peer_addr, k_addr) = (peer as *mut GpuPeer as usize, k as *const LinalgKernels as usize);
    let (a_addr, a_len) = (a.as_ptr() as usize, a.len());
    let sweeps = default_sweeps(nu);
    let report = hybrid_auto_split_ranges(
        plan,
        bu,
        |r| {
            tandem_cpu_runs(r, |lo, hi| {
                let (mut uc, mut sc, mut vc) =
                    cpu::gesvd_jacobi_batched(&a[lo * mu * nu..hi * mu * nu], hi - lo, mu, nu, sweeps, want_v);
                sort_singular_descending(&mut uc, &mut sc, vc.as_deref_mut(), mu, nu);
                // SAFETY: outputs outlive the split; runs write disjoint item ranges.
                unsafe {
                    scatter(u_addr, lo * mu * nu, &uc);
                    scatter(s_addr, lo * nu, &sc);
                }
                if let Some(vc) = vc {
                    unsafe { scatter(v_addr, lo * nu * nu, &vc) };
                }
            });
        },
        move |r| {
            // SAFETY: as in gemm_tandem_batched.
            let peer = unsafe { &mut *(peer_addr as *mut GpuPeer) };
            let k = unsafe { &*(k_addr as *const LinalgKernels) };
            let a = unsafe { std::slice::from_raw_parts(a_addr as *const f64, a_len) };
            let (lo, hi) = (r.start, r.end);
            match gesvd_auto_batched(peer, k, &a[lo * mu * nu..hi * mu * nu], (hi - lo) as u32, m, n, want_v) {
                Ok(mut res) => {
                    sort_singular_descending(&mut res.u, &mut res.sigma, res.v.as_deref_mut(), mu, nu);
                    unsafe {
                        scatter(u_addr, lo * mu * nu, &res.u);
                        scatter(s_addr, lo * nu, &res.sigma);
                    }
                    if let Some(vd) = res.v {
                        unsafe { scatter(v_addr, lo * nu * nu, &vd) };
                    }
                }
                Err(e) => *err_slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(e),
            }
        },
    );
    if let Some(e) = device_err.lock().unwrap_or_else(|p| p.into_inner()).take() {
        return Err(e);
    }
    Ok((GesvdResult { u, sigma, v: if want_v { Some(v) } else { None } }, report))
}

/// Batched symmetric eigendecomposition over host buffers: `a` holds
/// `batch` `n x n` matrices; returns `(eigenvalues, eigenvectors)`
/// with eigenvalues in diagonal (unsorted) order, `batch * n` long,
/// and eigenvectors as columns (`batch * n * n`) when `want_v`.
#[allow(clippy::too_many_arguments)]
pub fn syev_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    a: &[f64],
    batch: u32,
    n: u32,
    max_sweeps: u32,
    want_v: bool,
    shape: JacobiShape,
) -> Result<(Vec<f64>, Option<Vec<f64>>), GpuPeerError> {
    let (bu, nu) = (batch as usize, n as usize);
    check_dim(a.len() == bu * nu * nu, "syev: a length")?;
    let pa = pin(peer, &f64_bytes(a))?;
    let pw = pin(peer, &vec![0u8; bu * nu * 8])?;
    let pv = if want_v { Some(pin(peer, &vec![0u8; bu * nu * nu * 8])?) } else { None };
    launch_syev(peer, k, pa.ptr, pw.ptr, pv.as_ref().map(|p| p.ptr), batch, n, max_sweeps, shape)?;
    peer.sync_wide()?;
    let w = fetch_f64(peer, &pw, bu * nu)?;
    let v = match &pv {
        Some(p) => Some(fetch_f64(peer, p, bu * nu * nu)?),
        None => None,
    };
    if let Some(p) = pv {
        p.release(peer)?;
    }
    pw.release(peer)?;
    pa.release(peer)?;
    Ok((w, v))
}

/// Result of [`gesvd_batched`].
#[derive(Debug, Clone)]
pub struct GesvdResult {
    /// `batch` `m x n` matrices with orthonormal columns.
    pub u: Vec<f64>,
    /// `batch * n` singular values in column (unsorted) order.
    pub sigma: Vec<f64>,
    /// `batch` `n x n` right singular vector matrices when requested.
    pub v: Option<Vec<f64>>,
}

/// Batched one-sided Jacobi SVD over host buffers: `a` holds `batch`
/// `m x n` matrices with `m >= n`.
#[allow(clippy::too_many_arguments)]
pub fn gesvd_batched(
    peer: &mut GpuPeer,
    k: &LinalgKernels,
    a: &[f64],
    batch: u32,
    m: u32,
    n: u32,
    max_sweeps: u32,
    want_v: bool,
    shape: JacobiShape,
) -> Result<GesvdResult, GpuPeerError> {
    let (bu, mu, nu) = (batch as usize, m as usize, n as usize);
    check_dim(a.len() == bu * mu * nu, "gesvd: a length")?;
    let pa = pin(peer, &f64_bytes(a))?;
    let ps = pin(peer, &vec![0u8; bu * nu * 8])?;
    let pv = if want_v { Some(pin(peer, &vec![0u8; bu * nu * nu * 8])?) } else { None };
    launch_gesvd(
        peer, k, pa.ptr, ps.ptr, pv.as_ref().map(|p| p.ptr), batch, m, n, max_sweeps, shape,
    )?;
    peer.sync_wide()?;
    let u = fetch_f64(peer, &pa, bu * mu * nu)?;
    let sigma = fetch_f64(peer, &ps, bu * nu)?;
    let v = match &pv {
        Some(p) => Some(fetch_f64(peer, p, bu * nu * nu)?),
        None => None,
    };
    if let Some(p) = pv {
        p.release(peer)?;
    }
    ps.release(peer)?;
    pa.release(peer)?;
    Ok(GesvdResult { u, sigma, v })
}

// ------------------------------------------------------------ ragged inputs by shape

/// Group item indices by shape, so a ragged input becomes one
/// uniform batch per shape: `(shape, indices)` pairs in ascending
/// shape order, every index exactly once, indices ascending within a
/// group. The batched kernels take one shape per call; a consumer
/// with mixed shapes calls once per group.
pub fn group_by_shape<S: Ord + Copy>(shapes: &[S]) -> Vec<(S, Vec<usize>)> {
    let mut order: Vec<usize> = (0..shapes.len()).collect();
    order.sort_by_key(|&i| (shapes[i], i));
    let mut groups: Vec<(S, Vec<usize>)> = Vec::new();
    for i in order {
        match groups.last_mut() {
            Some((s, idx)) if *s == shapes[i] => idx.push(i),
            _ => groups.push((shapes[i], vec![i])),
        }
    }
    groups
}

/// Gather the items named by `indices` into a contiguous batch:
/// item `i` is `src[offsets[i]..offsets[i] + item_len]`, and the
/// result holds them in `indices` order, `indices.len() * item_len`
/// long.
pub fn gather_items(src: &[f64], offsets: &[usize], item_len: usize, indices: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(indices.len() * item_len);
    for &i in indices {
        out.extend_from_slice(&src[offsets[i]..offsets[i] + item_len]);
    }
    out
}

/// Scatter a contiguous batch back: item `j` of `batch` (`item_len`
/// elements) is written to `dst[offsets[indices[j]]..]`. The inverse
/// of [`gather_items`] for outputs of the same item length.
pub fn scatter_items(batch: &[f64], offsets: &[usize], item_len: usize, indices: &[usize], dst: &mut [f64]) {
    for (j, &i) in indices.iter().enumerate() {
        dst[offsets[i]..offsets[i] + item_len].copy_from_slice(&batch[j * item_len..(j + 1) * item_len]);
    }
}

// ------------------------------------------------------------ CPU references

/// CPU implementations with the kernels' semantics. einsum and gemm
/// accumulate with `mul_add` in the kernels' index order and match
/// them bit for bit; the Jacobi routines sweep cyclically and match
/// the kernels to rounding.
pub mod cpu {
    use super::EinsumSpec;

    /// Batched LU with partial pivoting, the kernel's semantics:
    /// `(packed factors, interchanges, flags)` as in
    /// [`super::LuResult`]. Pivot by largest magnitude, ties to the
    /// lowest row; every update a fused multiply-add in the kernel's
    /// order, so the two match bit for bit.
    pub fn getrf_batched(a: &[f64], batch: usize, n: usize) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
        let mut lu = a.to_vec();
        let mut piv = vec![0i32; batch * n];
        let mut info = vec![0i32; batch];
        for item in 0..batch {
            let s = &mut lu[item * n * n..(item + 1) * n * n];
            for k in 0..n {
                let mut p = k;
                let mut best = s[k * n + k].abs();
                for i in k + 1..n {
                    let v = s[i * n + k].abs();
                    if v > best {
                        best = v;
                        p = i;
                    }
                }
                piv[item * n + k] = p as i32;
                if s[p * n + k] == 0.0 && info[item] == 0 {
                    info[item] = k as i32 + 1;
                }
                if p != k {
                    for j in 0..n {
                        s.swap(k * n + j, p * n + j);
                    }
                }
                let pivot = s[k * n + k];
                if pivot != 0.0 {
                    for i in k + 1..n {
                        s[i * n + k] /= pivot;
                    }
                    for i in k + 1..n {
                        let l = s[i * n + k];
                        for j in k + 1..n {
                            s[i * n + j] = (-l).mul_add(s[k * n + j], s[i * n + j]);
                        }
                    }
                }
            }
        }
        (lu, piv, info)
    }

    /// Batched solve with packed factors from [`getrf_batched`]: `b`
    /// holds `batch` row-major `n x nrhs` right-hand sides.
    pub fn getrs_batched(lu: &[f64], piv: &[i32], b: &[f64], batch: usize, n: usize, nrhs: usize) -> Vec<f64> {
        let mut x = b.to_vec();
        for item in 0..batch {
            let f = &lu[item * n * n..(item + 1) * n * n];
            let pv = &piv[item * n..(item + 1) * n];
            let xs = &mut x[item * n * nrhs..(item + 1) * n * nrhs];
            for k in 0..n {
                let p = pv[k] as usize;
                if p != k {
                    for j in 0..nrhs {
                        xs.swap(k * nrhs + j, p * nrhs + j);
                    }
                }
            }
            for k in 0..n.saturating_sub(1) {
                for i in k + 1..n {
                    let l = f[i * n + k];
                    for j in 0..nrhs {
                        xs[i * nrhs + j] = (-l).mul_add(xs[k * nrhs + j], xs[i * nrhs + j]);
                    }
                }
            }
            for k in (0..n).rev() {
                let d = f[k * n + k];
                for j in 0..nrhs {
                    xs[k * nrhs + j] /= d;
                }
                for i in 0..k {
                    let u = f[i * n + k];
                    for j in 0..nrhs {
                        xs[i * nrhs + j] = (-u).mul_add(xs[k * nrhs + j], xs[i * nrhs + j]);
                    }
                }
            }
        }
        x
    }

    /// Batched inverse from packed factors: [`getrs_batched`] with
    /// the identity right-hand side.
    pub fn getri_batched(lu: &[f64], piv: &[i32], batch: usize, n: usize) -> Vec<f64> {
        let mut eye = vec![0f64; batch * n * n];
        for item in 0..batch {
            for i in 0..n {
                eye[item * n * n + i * n + i] = 1.0;
            }
        }
        getrs_batched(lu, piv, &eye, batch, n, n)
    }

    /// Batched einsum.
    pub fn einsum(spec: &EinsumSpec, a: &[f64], b: Option<&[f64]>, batch: usize) -> Vec<f64> {
        let o_size = spec.out_size();
        let a_size = spec.a_size();
        let b_size = spec.b_size();
        let (a_strides, b_strides, o_strides, c_extents, a_kind, b_kind) =
            (spec.part(0), spec.part(1), spec.part(2), spec.part(3), spec.part(4), spec.part(5));
        let c_total: usize = c_extents.iter().map(|e| *e as usize).product::<usize>().max(1);
        let mut out = vec![0f64; batch * o_size];
        let mut p_idx = vec![0i32; o_strides.len()];
        let mut c_idx = vec![0i32; c_extents.len()];
        let offset = |strides: &[i32], kinds: &[i32], p_idx: &[i32], c_idx: &[i32]| -> usize {
            let mut off = 0i32;
            for (axis, k) in kinds.iter().enumerate() {
                let idx = if (k >> 16) & 0xFFFF == 0 {
                    p_idx[(k & 0xFFFF) as usize]
                } else {
                    c_idx[(k & 0xFFFF) as usize]
                };
                off += idx * strides[axis];
            }
            off as usize
        };
        for bi in 0..batch {
            let ab = &a[bi * a_size..(bi + 1) * a_size];
            let bb = b.map(|b| &b[bi * b_size..(bi + 1) * b_size]);
            for tid in 0..o_size {
                let mut rem = tid as i32;
                for (axis, s) in o_strides.iter().enumerate() {
                    let v = if *s == 0 { 0 } else { rem / s };
                    p_idx[axis] = v;
                    rem -= v * s;
                }
                let mut acc = 0f64;
                for c_flat in 0..c_total {
                    let mut r = c_flat as i32;
                    for j in (0..c_extents.len()).rev() {
                        let e = c_extents[j];
                        let v = if e <= 1 { 0 } else { r % e };
                        c_idx[j] = v;
                        if e > 1 {
                            r /= e;
                        }
                    }
                    let a_off = offset(a_strides, a_kind, &p_idx, &c_idx);
                    match bb {
                        Some(bb) => {
                            let b_off = offset(b_strides, b_kind, &p_idx, &c_idx);
                            acc = ab[a_off].mul_add(bb[b_off], acc);
                        }
                        None => acc += ab[a_off],
                    }
                }
                out[bi * o_size + tid] = acc;
            }
        }
        out
    }

    /// Batched row-major `C = A * B`.
    pub fn gemm_batched(a: &[f64], b: &[f64], batch: usize, m: usize, n: usize, k: usize) -> Vec<f64> {
        let mut c = vec![0f64; batch * m * n];
        for bi in 0..batch {
            let ab = &a[bi * m * k..(bi + 1) * m * k];
            let bb = &b[bi * k * n..(bi + 1) * k * n];
            let cb = &mut c[bi * m * n..(bi + 1) * m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f64;
                    for kk in 0..k {
                        acc = ab[i * k + kk].mul_add(bb[kk * n + j], acc);
                    }
                    cb[i * n + j] = acc;
                }
            }
        }
        c
    }

    /// Jacobi rotation annihilating `s_pq` of the symmetric block
    /// `(s_pp, s_pq; s_pq, s_qq)`; identity when `s_pq` is negligible.
    fn jacobi_rotation(s_pp: f64, s_qq: f64, s_pq: f64) -> (f64, f64) {
        if s_pq.abs() < 1e-40 {
            return (1.0, 0.0);
        }
        let theta_num = s_qq - s_pp;
        let theta_den = 2.0 * s_pq;
        let t = if theta_den.abs() < 1e-40 * theta_num.abs() {
            0.0
        } else {
            let theta = theta_num / theta_den;
            let disc = (1.0 + theta * theta).sqrt();
            if theta >= 0.0 { 1.0 / (theta + disc) } else { 1.0 / (theta - disc) }
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        (c, c * t)
    }

    /// Symmetric eigendecomposition of one `n x n` row-major matrix by
    /// cyclic Jacobi; returns eigenvalues in diagonal order and, when
    /// `want_v`, eigenvectors as columns.
    pub fn syev_jacobi(a: &[f64], n: usize, max_sweeps: u32, want_v: bool) -> (Vec<f64>, Option<Vec<f64>>) {
        let mut s = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                s[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
            }
        }
        let mut v = if want_v {
            let mut v = vec![0f64; n * n];
            for i in 0..n {
                v[i * n + i] = 1.0;
            }
            Some(v)
        } else {
            None
        };
        let mut prev_off: Option<f64> = None;
        for _ in 0..max_sweeps {
            let mut off = 0f64;
            for p in 0..n {
                for q in (p + 1)..n {
                    off = s[p * n + q].mul_add(s[p * n + q], off);
                }
            }
            if off == 0.0 {
                break;
            }
            if let Some(prev) = prev_off
                && off >= prev
            {
                break;
            }
            prev_off = Some(off);
            for p in 0..n.saturating_sub(1) {
                for q in (p + 1)..n {
                    let (c, sn) = jacobi_rotation(s[p * n + p], s[q * n + q], s[p * n + q]);
                    if sn == 0.0 && c == 1.0 {
                        continue;
                    }
                    for k in 0..n {
                        let s_kp = s[k * n + p];
                        let s_kq = s[k * n + q];
                        s[k * n + p] = c * s_kp - sn * s_kq;
                        s[k * n + q] = sn * s_kp + c * s_kq;
                    }
                    for k in 0..n {
                        let s_pk = s[p * n + k];
                        let s_qk = s[q * n + k];
                        s[p * n + k] = c * s_pk - sn * s_qk;
                        s[q * n + k] = sn * s_pk + c * s_qk;
                    }
                    let avg = 0.5 * (s[p * n + q] + s[q * n + p]);
                    s[p * n + q] = avg;
                    s[q * n + p] = avg;
                    if let Some(v) = v.as_mut() {
                        for k in 0..n {
                            let v_kp = v[k * n + p];
                            let v_kq = v[k * n + q];
                            v[k * n + p] = c * v_kp - sn * v_kq;
                            v[k * n + q] = sn * v_kp + c * v_kq;
                        }
                    }
                }
            }
        }
        let w: Vec<f64> = (0..n).map(|i| s[i * n + i]).collect();
        (w, v)
    }

    /// [`syev_jacobi`] over `batch` contiguous matrices.
    pub fn syev_jacobi_batched(
        a: &[f64],
        batch: usize,
        n: usize,
        max_sweeps: u32,
        want_v: bool,
    ) -> (Vec<f64>, Option<Vec<f64>>) {
        let mut w = Vec::with_capacity(batch * n);
        let mut v = if want_v { Some(Vec::with_capacity(batch * n * n)) } else { None };
        for bi in 0..batch {
            let (wi, vi) = syev_jacobi(&a[bi * n * n..(bi + 1) * n * n], n, max_sweeps, want_v);
            w.extend_from_slice(&wi);
            if let (Some(v), Some(vi)) = (v.as_mut(), vi) {
                v.extend_from_slice(&vi);
            }
        }
        (w, v)
    }

    /// One-sided Jacobi SVD of one `m x n` row-major matrix (`m >=
    /// n`): returns `(U, sigma, V)` with `U` `m x n` orthonormal
    /// columns, `sigma` in column order, `V` when `want_v`. A pair
    /// rotates when its cosine exceeds the f64 unit roundoff.
    pub fn gesvd_jacobi(
        a: &[f64],
        m: usize,
        n: usize,
        max_sweeps: u32,
        want_v: bool,
    ) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
        let tol = f64::EPSILON;
        let mut u = a.to_vec();
        let mut v = if want_v {
            let mut v = vec![0f64; n * n];
            for i in 0..n {
                v[i * n + i] = 1.0;
            }
            Some(v)
        } else {
            None
        };
        for _ in 0..max_sweeps {
            let mut converged = true;
            for p in 0..n.saturating_sub(1) {
                for q in (p + 1)..n {
                    let (mut aa, mut bb, mut cc) = (0f64, 0f64, 0f64);
                    for i in 0..m {
                        let up = u[i * n + p];
                        let uq = u[i * n + q];
                        aa = up.mul_add(up, aa);
                        bb = uq.mul_add(uq, bb);
                        cc = up.mul_add(uq, cc);
                    }
                    let denom = (aa * bb).sqrt();
                    if denom == 0.0 || cc.abs() / denom < tol {
                        continue;
                    }
                    converged = false;
                    let zeta = (bb - aa) / (2.0 * cc);
                    let mut t = 1.0 / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                    if zeta < 0.0 {
                        t = -t;
                    }
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let sn = t * c;
                    for i in 0..m {
                        let up = u[i * n + p];
                        let uq = u[i * n + q];
                        u[i * n + p] = c * up - sn * uq;
                        u[i * n + q] = sn * up + c * uq;
                    }
                    if let Some(v) = v.as_mut() {
                        for i in 0..n {
                            let vp = v[i * n + p];
                            let vq = v[i * n + q];
                            v[i * n + p] = c * vp - sn * vq;
                            v[i * n + q] = sn * vp + c * vq;
                        }
                    }
                }
            }
            if converged {
                break;
            }
        }
        let mut sigma = vec![0f64; n];
        for j in 0..n {
            let mut ss = 0f64;
            for i in 0..m {
                ss = u[i * n + j].mul_add(u[i * n + j], ss);
            }
            let sj = ss.sqrt();
            sigma[j] = sj;
            if sj > 0.0 {
                for i in 0..m {
                    u[i * n + j] /= sj;
                }
            }
        }
        (u, sigma, v)
    }

    /// [`gesvd_jacobi`] over `batch` contiguous matrices; returns
    /// `(U, sigma, V)` batched the same way.
    pub fn gesvd_jacobi_batched(
        a: &[f64],
        batch: usize,
        m: usize,
        n: usize,
        max_sweeps: u32,
        want_v: bool,
    ) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
        let mut u = Vec::with_capacity(batch * m * n);
        let mut sigma = Vec::with_capacity(batch * n);
        let mut v = if want_v { Some(Vec::with_capacity(batch * n * n)) } else { None };
        for bi in 0..batch {
            let (ui, si, vi) = gesvd_jacobi(&a[bi * m * n..(bi + 1) * m * n], m, n, max_sweeps, want_v);
            u.extend_from_slice(&ui);
            sigma.extend_from_slice(&si);
            if let (Some(v), Some(vi)) = (v.as_mut(), vi) {
                v.extend_from_slice(&vi);
            }
        }
        (u, sigma, v)
    }
}

// ------------------------------------------------------------ accel_op layer

/// The linalg ops registered with [`crate::backend::accel_op`].
#[derive(Clone, Copy, Debug)]
pub struct LinalgAccelOps {
    /// Batched GEMM.
    pub gemm: AccelOpId,
    /// Batched symmetric eigen (block-per-matrix kernel).
    pub syev: AccelOpId,
    /// Batched SVD (block-per-matrix kernel).
    pub gesvd: AccelOpId,
}

/// Argument layout shared by the CPU side and the kernel side of the
/// registered ops. Host pointers travel as `U64`, device addresses as
/// `DevicePtr`; the scalar tail is identical on both sides.
///
/// - gemm: `[a, b, c, batch, m, n, k, lda, ldb, ldc]` (the CPU side
///   reads the first seven; items are contiguous)
/// - syev: `[a, w, v, batch, n, max_sweeps, want_v]`
/// - gesvd: `[a, sigma, v, batch, m, n, max_sweeps, want_v]`
///
/// The CPU implementations dereference the host pointers, so a
/// dispatch's `cpu_args` must name buffers that are live, correctly
/// sized, and not aliased for the whole call.
pub fn register_linalg_accel_ops() -> LinalgAccelOps {
    fn ptr(a: &KernelArg<'_>) -> u64 {
        match a {
            KernelArg::U64(p) => *p,
            KernelArg::DevicePtr(p) => *p as u64,
            other => panic!("linalg accel op: expected a pointer argument, got {other:?}"),
        }
    }
    fn u32_of(a: &KernelArg<'_>) -> u32 {
        match a {
            KernelArg::U32(v) => *v,
            other => panic!("linalg accel op: expected a u32 argument, got {other:?}"),
        }
    }
    let gemm = register_accel_op("flynnel.linalg.gemm_batched_f64", 0, |_count, args| {
        let (a, b, c) = (ptr(&args[0]), ptr(&args[1]), ptr(&args[2]));
        let (batch, m, n, k) = (
            u32_of(&args[3]) as usize,
            u32_of(&args[4]) as usize,
            u32_of(&args[5]) as usize,
            u32_of(&args[6]) as usize,
        );
        // SAFETY: the registration contract above - live, sized,
        // non-aliased host buffers for the duration of the dispatch.
        let (a, b, c) = unsafe {
            (
                std::slice::from_raw_parts(a as *const f64, batch * m * k),
                std::slice::from_raw_parts(b as *const f64, batch * k * n),
                std::slice::from_raw_parts_mut(c as *mut f64, batch * m * n),
            )
        };
        c.copy_from_slice(&cpu::gemm_batched(a, b, batch, m, n, k));
    });
    let syev = register_accel_op("flynnel.linalg.syev_jacobi_f64", 0, |_count, args| {
        let (a, w, v) = (ptr(&args[0]), ptr(&args[1]), ptr(&args[2]));
        let (batch, n, sweeps, want_v) = (
            u32_of(&args[3]) as usize,
            u32_of(&args[4]) as usize,
            u32_of(&args[5]),
            u32_of(&args[6]) != 0,
        );
        // SAFETY: registration contract (live, sized, non-aliased).
        let a = unsafe { std::slice::from_raw_parts(a as *const f64, batch * n * n) };
        let (wv, vv) = cpu::syev_jacobi_batched(a, batch, n, sweeps, want_v);
        // SAFETY: as above; `w` holds batch * n f64.
        unsafe { std::slice::from_raw_parts_mut(w as *mut f64, batch * n) }.copy_from_slice(&wv);
        if let Some(vv) = vv {
            // SAFETY: as above; `v` holds batch * n * n f64 when want_v.
            unsafe { std::slice::from_raw_parts_mut(v as *mut f64, batch * n * n) }
                .copy_from_slice(&vv);
        }
    });
    let gesvd = register_accel_op("flynnel.linalg.gesvd_jacobi_f64", 0, |_count, args| {
        let (a, s, v) = (ptr(&args[0]), ptr(&args[1]), ptr(&args[2]));
        let (batch, m, n, sweeps, want_v) = (
            u32_of(&args[3]) as usize,
            u32_of(&args[4]) as usize,
            u32_of(&args[5]) as usize,
            u32_of(&args[6]),
            u32_of(&args[7]) != 0,
        );
        // SAFETY: registration contract (live, sized, non-aliased).
        let a_mut = unsafe { std::slice::from_raw_parts_mut(a as *mut f64, batch * m * n) };
        let (u, sv, vv) = cpu::gesvd_jacobi_batched(a_mut, batch, m, n, sweeps, want_v);
        a_mut.copy_from_slice(&u);
        // SAFETY: as above; `s` holds batch * n f64.
        unsafe { std::slice::from_raw_parts_mut(s as *mut f64, batch * n) }.copy_from_slice(&sv);
        if let Some(vv) = vv {
            // SAFETY: as above; `v` holds batch * n * n f64 when want_v.
            unsafe { std::slice::from_raw_parts_mut(v as *mut f64, batch * n * n) }
                .copy_from_slice(&vv);
        }
    });
    LinalgAccelOps { gemm, syev, gesvd }
}

/// Bind the block-per-matrix kernels of `linalg_f64.ptx` to `ops`
/// on `backend` (a registered CUDA backend). After this,
/// [`dispatch_accel`] can route each op to the device.
pub fn bind_linalg_kernels(ops: &LinalgAccelOps, backend: Backend) -> Result<(), BackendError> {
    let ptx = LINALG_PTX.as_bytes();
    bind_accel_kernel(ops.gemm, backend, "flynnel_gemm_batched_f64", ptx)?;
    bind_accel_kernel(ops.syev, backend, "flynnel_syev_jacobi_f64_blk", ptx)?;
    bind_accel_kernel(ops.gesvd, backend, "flynnel_gesvd_jacobi_f64_blk", ptx)?;
    Ok(())
}

/// Work-item count that makes a backend's 256-thread launch heuristic
/// produce exactly `blocks` blocks.
fn blocks_as_count(blocks: u32) -> u32 {
    blocks.max(1) * LINALG_BLOCK
}

/// Route a batched GEMM through [`dispatch_accel`]: host buffers for
/// the CPU side, device addresses (`dev` = `(a, b, c)`) for the
/// kernel side.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn gemm_accel(
    plan: &JobPlan,
    ops: &LinalgAccelOps,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    dev: (u64, u64, u64),
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
) -> AccelReport {
    // Kernel ABI: (a, b, c, batch, m, n, k, lda, ldb, ldc); the CPU
    // side reads the first seven and ignores the leading dimensions
    // (contiguous items).
    let tail = [
        KernelArg::U32(batch),
        KernelArg::U32(m),
        KernelArg::U32(n),
        KernelArg::U32(k),
        KernelArg::U32(k),
        KernelArg::U32(n),
        KernelArg::U32(n),
    ];
    let cpu_args = [
        KernelArg::U64(a.as_ptr() as u64),
        KernelArg::U64(b.as_ptr() as u64),
        KernelArg::U64(c.as_mut_ptr() as u64),
        tail[0], tail[1], tail[2], tail[3], tail[4], tail[5], tail[6],
    ];
    let kernel_args = [
        KernelArg::DevicePtr(dev.0 as usize),
        KernelArg::DevicePtr(dev.1 as usize),
        KernelArg::DevicePtr(dev.2 as usize),
        tail[0], tail[1], tail[2], tail[3], tail[4], tail[5], tail[6],
    ];
    let count = blocks_as_count(m.div_ceil(16) * n.div_ceil(16) * batch);
    dispatch_accel(plan, ops.gemm, count, &cpu_args, &kernel_args)
}

/// Route a batched symmetric eigen through [`dispatch_accel`]; `dev`
/// = `(a, w, v)` device addresses (`v` may be 0 without `want_v`).
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn syev_accel(
    plan: &JobPlan,
    ops: &LinalgAccelOps,
    a: &[f64],
    w: &mut [f64],
    v: Option<&mut [f64]>,
    dev: (u64, u64, u64),
    batch: u32,
    n: u32,
    max_sweeps: u32,
) -> AccelReport {
    let want_v = v.is_some();
    let v_ptr = v.map_or(0u64, |v| v.as_mut_ptr() as u64);
    let tail = [
        KernelArg::U32(batch),
        KernelArg::U32(n),
        KernelArg::U32(max_sweeps),
        KernelArg::U32(u32::from(want_v)),
    ];
    let cpu_args = [
        KernelArg::U64(a.as_ptr() as u64),
        KernelArg::U64(w.as_mut_ptr() as u64),
        KernelArg::U64(v_ptr),
        tail[0], tail[1], tail[2], tail[3],
    ];
    let kernel_args = [
        KernelArg::DevicePtr(dev.0 as usize),
        KernelArg::DevicePtr(dev.1 as usize),
        KernelArg::DevicePtr(dev.2 as usize),
        tail[0], tail[1], tail[2], tail[3],
    ];
    dispatch_accel(plan, ops.syev, blocks_as_count(batch), &cpu_args, &kernel_args)
}

/// Route a batched SVD through [`dispatch_accel`]; `a` is overwritten
/// with `U`; `dev` = `(a, sigma, v)` device addresses.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn gesvd_accel(
    plan: &JobPlan,
    ops: &LinalgAccelOps,
    a: &mut [f64],
    sigma: &mut [f64],
    v: Option<&mut [f64]>,
    dev: (u64, u64, u64),
    batch: u32,
    m: u32,
    n: u32,
    max_sweeps: u32,
) -> AccelReport {
    let want_v = v.is_some();
    let v_ptr = v.map_or(0u64, |v| v.as_mut_ptr() as u64);
    let tail = [
        KernelArg::U32(batch),
        KernelArg::U32(m),
        KernelArg::U32(n),
        KernelArg::U32(max_sweeps),
        KernelArg::U32(u32::from(want_v)),
    ];
    let cpu_args = [
        KernelArg::U64(a.as_mut_ptr() as u64),
        KernelArg::U64(sigma.as_mut_ptr() as u64),
        KernelArg::U64(v_ptr),
        tail[0], tail[1], tail[2], tail[3], tail[4],
    ];
    let kernel_args = [
        KernelArg::DevicePtr(dev.0 as usize),
        KernelArg::DevicePtr(dev.1 as usize),
        KernelArg::DevicePtr(dev.2 as usize),
        tail[0], tail[1], tail[2], tail[3], tail[4],
    ];
    dispatch_accel(plan, ops.gesvd, blocks_as_count(batch), &cpu_args, &kernel_args)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measured crossovers: thr from batch 1024 at n=4, 2048 at
    /// n=8, 4096 at n=16; blk below those and for every n > 16.
    #[test]
    fn jacobi_shape_follows_measured_crossovers() {
        use JacobiShape::{BlockPerMatrix as Blk, ThreadPerMatrix as Thr};
        assert_eq!(jacobi_shape_for(4, 1024), Thr);
        assert_eq!(jacobi_shape_for(4, 512), Blk);
        assert_eq!(jacobi_shape_for(8, 1024), Blk);
        assert_eq!(jacobi_shape_for(8, 2048), Thr);
        assert_eq!(jacobi_shape_for(16, 2048), Blk);
        assert_eq!(jacobi_shape_for(16, 4096), Thr);
        assert_eq!(jacobi_shape_for(32, 65536), Blk);
        assert_eq!(jacobi_shape_for(64, 65536), Blk);
    }

    #[test]
    fn einsum_spec_matmul_tables() {
        let s = EinsumSpec::parse("ij,jk->ik", &[2, 3], Some(&[3, 4])).expect("spec");
        assert_eq!(s.out_shape(), &[2, 4]);
        assert_eq!(s.out_size(), 8);
        assert_eq!(s.n_contract, 1);
        // a: i preserved slot 0, j contracted slot 0; b: j contracted, k preserved slot 1.
        assert_eq!(s.part(4), &[0, 1 << 16]);
        assert_eq!(s.part(5), &[1 << 16, 1]);
        assert_eq!(s.part(3), &[3]);
    }

    #[test]
    fn einsum_spec_rejects_mismatch() {
        assert!(matches!(
            EinsumSpec::parse("ij,jk->ik", &[2, 3], Some(&[5, 4])),
            Err(EinsumError::ExtentMismatch { letter: 'j', .. })
        ));
        assert!(matches!(
            EinsumSpec::parse("ij->ik", &[2, 3], None),
            Err(EinsumError::UnknownOutputAxis('k'))
        ));
        assert!(matches!(
            EinsumSpec::parse("ijk->i", &[2, 3], None),
            Err(EinsumError::RankMismatch { .. })
        ));
    }

    #[test]
    fn cpu_einsum_matches_direct_matmul() {
        let (m, k, n) = (3usize, 4usize, 5usize);
        let a: Vec<f64> = (0..m * k).map(|i| i as f64 * 0.5 + 1.0).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64).sin()).collect();
        let spec = EinsumSpec::parse("ij,jk->ik", &[m, k], Some(&[k, n])).expect("spec");
        let e = cpu::einsum(&spec, &a, Some(&b), 1);
        let g = cpu::gemm_batched(&a, &b, 1, m, n, k);
        assert_eq!(e, g);
    }

    #[test]
    fn cpu_trace_and_axissum() {
        let n = 4usize;
        let a: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let tr = cpu::einsum(&EinsumSpec::parse("ii->", &[n, n], None).expect("spec"), &a, None, 1);
        let expect_tr: f64 = (0..n).map(|i| a[i * n + i]).sum();
        assert_eq!(tr, vec![expect_tr]);
        let rows = cpu::einsum(&EinsumSpec::parse("ij->i", &[n, n], None).expect("spec"), &a, None, 1);
        for i in 0..n {
            let s: f64 = a[i * n..(i + 1) * n].iter().sum();
            assert_eq!(rows[i], s);
        }
    }

    #[test]
    fn cpu_syev_diagonalizes_known_matrix() {
        // Eigenvalues of [[2,1],[1,2]] are 1 and 3.
        let (w, v) = cpu::syev_jacobi(&[2.0, 1.0, 1.0, 2.0], 2, default_sweeps(2), true);
        let mut sorted = w.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        assert!((sorted[0] - 1.0).abs() < 1e-14 && (sorted[1] - 3.0).abs() < 1e-14);
        let v = v.expect("vectors");
        for (col, lambda) in w.iter().enumerate() {
            let x = [v[col], v[2 + col]];
            let ax = [2.0 * x[0] + x[1], x[0] + 2.0 * x[1]];
            assert!((ax[0] - lambda * x[0]).abs() < 1e-13 && (ax[1] - lambda * x[1]).abs() < 1e-13);
        }
    }

    #[test]
    fn cpu_gesvd_recovers_singular_values() {
        // diag(3, 2) with m = n = 2: sigmas 3 and 2.
        let (u, s, v) = cpu::gesvd_jacobi(&[3.0, 0.0, 0.0, 2.0], 2, 2, default_sweeps(2), true);
        let mut ss = s.clone();
        ss.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
        assert!((ss[0] - 3.0).abs() < 1e-14 && (ss[1] - 2.0).abs() < 1e-14);
        let v = v.expect("v");
        // Reconstruct A = U diag(s) V^T.
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += u[i * 2 + k] * s[k] * v[j * 2 + k];
                }
                let want = [[3.0, 0.0], [0.0, 2.0]][i][j];
                assert!((acc - want).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn group_by_shape_covers_every_index_once_in_shape_order() {
        let shapes = [(4usize, 4usize), (2, 3), (4, 4), (2, 3), (8, 8), (2, 3)];
        let groups = group_by_shape(&shapes);
        assert_eq!(groups, vec![((2, 3), vec![1, 3, 5]), ((4, 4), vec![0, 2]), ((8, 8), vec![4])]);
        assert!(group_by_shape::<(usize, usize)>(&[]).is_empty());
    }

    #[test]
    fn gather_then_scatter_round_trips_ragged_items() {
        // Three items of lengths 4, 9, 4 laid out back to back.
        let src: Vec<f64> = (0..17).map(|x| x as f64).collect();
        let offsets = [0usize, 4, 13];
        let small = [0usize, 2];
        let batch = gather_items(&src, &offsets, 4, &small);
        assert_eq!(batch, vec![0.0, 1.0, 2.0, 3.0, 13.0, 14.0, 15.0, 16.0]);
        let doubled: Vec<f64> = batch.iter().map(|x| 2.0 * x).collect();
        let mut dst = vec![0f64; 17];
        scatter_items(&doubled, &offsets, 4, &small, &mut dst);
        assert_eq!(&dst[0..4], &[0.0, 2.0, 4.0, 6.0]);
        assert_eq!(&dst[13..17], &[26.0, 28.0, 30.0, 32.0]);
        assert!(dst[4..13].iter().all(|&x| x == 0.0), "the untouched item stays zero");
    }
}

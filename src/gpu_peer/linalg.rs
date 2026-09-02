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

use cudarc::nvrtc::Ptx;

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

/// Largest square dimension the block-per-matrix Jacobi kernels take.
pub const LINALG_MAX_N: usize = 64;
/// Largest dimension the thread-per-matrix Jacobi kernels take.
pub const LINALG_THR_MAX_N: usize = 16;
/// Largest einsum rank per operand and contraction set.
pub const EINSUM_MAX_RANK: usize = 12;
/// Threads per block every kernel here is launched with.
pub const LINALG_BLOCK: u32 = 256;
/// Largest `n` routed to the thread-per-matrix Jacobi kernels by
/// [`jacobi_shape_for`]. Both shapes are valid up to
/// [`LINALG_THR_MAX_N`]; `benches/gpu_linalg.rs` measures the
/// crossover on each bench host.
pub const JACOBI_THREAD_SHAPE_MAX_N: usize = 16;

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

/// The shape used for dimension `n` when the caller does not pin one.
pub fn jacobi_shape_for(n: usize) -> JacobiShape {
    if n <= JACOBI_THREAD_SHAPE_MAX_N {
        JacobiShape::ThreadPerMatrix
    } else {
        JacobiShape::BlockPerMatrix
    }
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
}

impl LinalgKernels {
    /// Load the PTX into the peer's context once and resolve every
    /// entry point.
    pub fn load(peer: &GpuPeer) -> Result<Self, GpuPeerError> {
        let module = match peer.context().load_module(Ptx::from_src(LINALG_PTX)) {
            Ok(m) => m,
            Err(ptx_err) => {
                eprintln!(
                    "flynnel gpu_peer linalg: checked-in PTX rejected ({ptx_err:?}); \
                     compiling the linalg kernels with NVRTC instead"
                );
                let ptx = cudarc::nvrtc::compile_ptx(LINALG_CU).map_err(|e| {
                    GpuPeerError::Driver(format!(
                        "linalg PTX load: {ptx_err:?}; NVRTC fallback compile: {e:?}"
                    ))
                })?;
                peer.context()
                    .load_module(ptx)
                    .map_err(|e| GpuPeerError::Driver(format!("linalg NVRTC-fallback load: {e:?}")))?
            }
        };
        let load = |entry: &str| -> Result<WideKernel, GpuPeerError> {
            let func = module
                .load_function(entry)
                .map_err(|e| GpuPeerError::Driver(format!("linalg entry `{entry}`: {e:?}")))?;
            Ok(WideKernel { _module: module.clone(), func })
        };
        Ok(Self {
            einsum: load("flynnel_einsum_f64")?,
            gemm: load("flynnel_gemm_batched_f64")?,
            syev_blk: load("flynnel_syev_jacobi_f64_blk")?,
            syev_thr: load("flynnel_syev_jacobi_f64_thr")?,
            gesvd_blk: load("flynnel_gesvd_jacobi_f64_blk")?,
            gesvd_thr: load("flynnel_gesvd_jacobi_f64_thr")?,
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

fn check_dim(cond: bool, what: &'static str) -> Result<(), GpuPeerError> {
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

fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f64(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A pinned span plus its device address; unpinned on drop-by-hand
/// through [`Self::release`].
struct Pinned {
    handle: super::ResidentHandle,
    ptr: u64,
}

fn pin(peer: &mut GpuPeer, bytes: &[u8]) -> Result<Pinned, GpuPeerError> {
    let handle = peer.pin_bulk(bytes)?;
    let (ptr, _) = peer.resident_ptr(&handle)?;
    Ok(Pinned { handle, ptr })
}

impl Pinned {
    fn release(self, peer: &mut GpuPeer) -> Result<(), GpuPeerError> {
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

// ------------------------------------------------------------ CPU references

/// CPU implementations with the kernels' semantics. einsum and gemm
/// accumulate with `mul_add` in the kernels' index order and match
/// them bit for bit; the Jacobi routines sweep cyclically and match
/// the kernels to rounding.
pub mod cpu {
    use super::EinsumSpec;

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
}

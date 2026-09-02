//! Ozaki-scheme f64 GEMM on the int8 tensor cores: the kernels in
//! `kernels/ozaki_f64.cu` split each operand into 7-bit slices aligned
//! to its row (A) or column (B) maximum exponent, multiply the slice
//! pairs exactly with int8 mma into int32, and recombine in f64 with
//! two-sum compensation. The result is not bit-identical to the
//! fma-ordered f64 kernel; its error is bounded by
//! `2^-53 * k * max|A row| * max|B column|` per element, before the
//! final rounding.

use cudarc::nvrtc::Ptx;

use super::linalg::{bytes_f64, check_dim, f64_bytes, pin, Pinned};
use super::{GpuPeer, GpuPeerError, WideKernel};

const OZAKI_PTX: &str = include_str!("../../kernels/ozaki_f64.ptx");
const OZAKI_CU: &str = include_str!("../../kernels/ozaki_f64.cu");

/// 7-bit slices per operand element, covering 56 bits.
pub const OZAKI_SLICES: usize = 8;
/// Output tile edge; `m` and `n` must be multiples of it.
pub const OZAKI_TILE: u32 = 32;
/// Staged k step (two mma halves); `k` must be a multiple of it.
pub const OZAKI_K_STEP: u32 = 32;
/// Largest `k`: the eight pair products sharing a diagonal's int32
/// accumulator stay exact below it.
pub const OZAKI_MAX_K: u32 = 16_384;
const BLOCK: u32 = 256;
const GEMM_THREADS: u32 = 128;

/// The five kernels of `ozaki_f64.ptx`, loaded from one module.
pub struct OzakiKernels {
    rowexp: WideKernel,
    colexp: WideKernel,
    split_a: WideKernel,
    split_bt: WideKernel,
    gemm: WideKernel,
}

impl OzakiKernels {
    /// Load the PTX into the peer's context once and resolve every
    /// entry point; compiles the `.cu` with NVRTC when the driver
    /// rejects the checked-in PTX.
    pub fn load(peer: &GpuPeer) -> Result<Self, GpuPeerError> {
        let module = match peer.context().load_module(Ptx::from_src(OZAKI_PTX)) {
            Ok(m) => m,
            Err(ptx_err) => {
                eprintln!(
                    "flynnel gpu_peer ozaki: checked-in PTX rejected ({ptx_err:?}); \
                     compiling the ozaki kernels with NVRTC instead"
                );
                let ptx = cudarc::nvrtc::compile_ptx(OZAKI_CU).map_err(|e| {
                    GpuPeerError::Driver(format!(
                        "ozaki PTX load: {ptx_err:?}; NVRTC fallback compile: {e:?}"
                    ))
                })?;
                peer.context()
                    .load_module(ptx)
                    .map_err(|e| GpuPeerError::Driver(format!("ozaki NVRTC-fallback load: {e:?}")))?
            }
        };
        let load = |entry: &str| -> Result<WideKernel, GpuPeerError> {
            let func = module
                .load_function(entry)
                .map_err(|e| GpuPeerError::Driver(format!("ozaki entry `{entry}`: {e:?}")))?;
            Ok(WideKernel::new(module.clone(), func))
        };
        Ok(Self {
            rowexp: load("flynnel_ozaki_rowexp_f64")?,
            colexp: load("flynnel_ozaki_colexp_f64")?,
            split_a: load("flynnel_ozaki_split_a_f64")?,
            split_bt: load("flynnel_ozaki_split_bt_f64")?,
            gemm: load("flynnel_ozaki_gemm_f64")?,
        })
    }
}

/// Shape constraints of the kernels.
pub fn check_shape(batch: u32, m: u32, n: u32, k: u32) -> Result<(), GpuPeerError> {
    check_dim(batch > 0 && m > 0 && n > 0 && k > 0, "ozaki: empty dimension")?;
    check_dim(
        m.is_multiple_of(OZAKI_TILE) && n.is_multiple_of(OZAKI_TILE),
        "ozaki: m and n must be multiples of 32",
    )?;
    check_dim(k.is_multiple_of(OZAKI_K_STEP), "ozaki: k must be a multiple of 32")?;
    check_dim(k <= OZAKI_MAX_K, "ozaki: k exceeds OZAKI_MAX_K")
}

/// Device workspace for one GEMM shape: the exponent vectors and the
/// slice matrices, pinned in the resident pool.
pub struct OzakiWorkspace {
    rowexp: Pinned,
    colexp: Pinned,
    a_slices: Pinned,
    bt_slices: Pinned,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
}

impl OzakiWorkspace {
    /// Pin the workspace for `batch` products of `m x k` by `k x n`.
    pub fn new(peer: &mut GpuPeer, batch: u32, m: u32, n: u32, k: u32) -> Result<Self, GpuPeerError> {
        check_shape(batch, m, n, k)?;
        let (bu, mu, nu, ku) = (batch as usize, m as usize, n as usize, k as usize);
        let rowexp = pin(peer, &vec![0u8; bu * mu * 4])?;
        let colexp = pin(peer, &vec![0u8; bu * nu * 4])?;
        let a_slices = pin(peer, &vec![0u8; OZAKI_SLICES * bu * mu * ku])?;
        let bt_slices = pin(peer, &vec![0u8; OZAKI_SLICES * bu * ku * nu])?;
        Ok(Self { rowexp, colexp, a_slices, bt_slices, batch, m, n, k })
    }

    /// Bytes the workspace pins for a shape.
    pub fn bytes(batch: u32, m: u32, n: u32, k: u32) -> usize {
        let (bu, mu, nu, ku) = (batch as usize, m as usize, n as usize, k as usize);
        bu * mu * 4 + bu * nu * 4 + OZAKI_SLICES * bu * ku * (mu + nu)
    }

    /// Unpin the workspace.
    pub fn release(self, peer: &mut GpuPeer) -> Result<(), GpuPeerError> {
        self.bt_slices.release(peer)?;
        self.a_slices.release(peer)?;
        self.colexp.release(peer)?;
        self.rowexp.release(peer)
    }
}

/// Enqueue `C = A * B` for the workspace's shape on the wide stream:
/// `a`, `b`, `c` are device addresses of contiguous row-major batches.
/// No sync.
pub fn launch_ozaki_gemm(
    peer: &GpuPeer,
    kern: &OzakiKernels,
    ws: &OzakiWorkspace,
    a: u64,
    b: u64,
    c: u64,
) -> Result<(), GpuPeerError> {
    let (batch, m, n, k) = (ws.batch, ws.m, ws.n, ws.k);
    let rows = batch as u64 * m as u64;
    let grid_rowexp = rows.div_ceil(u64::from(BLOCK / 32)) as u32;
    peer.launch_wide_async(&kern.rowexp, grid_rowexp, BLOCK, &[a, ws.rowexp.ptr], &[batch, m, k])?;
    let grid_colexp = batch * n.div_ceil(32);
    peer.launch_wide_async(&kern.colexp, grid_colexp, BLOCK, &[b, ws.colexp.ptr], &[batch, k, n])?;
    let a_elems = rows * k as u64;
    peer.launch_wide_async(
        &kern.split_a,
        a_elems.div_ceil(u64::from(BLOCK)) as u32,
        BLOCK,
        &[a, ws.rowexp.ptr, ws.a_slices.ptr],
        &[batch, m, k],
    )?;
    let grid_split_bt = batch * k.div_ceil(32) * n.div_ceil(32);
    peer.launch_wide_async(
        &kern.split_bt,
        grid_split_bt,
        BLOCK,
        &[b, ws.colexp.ptr, ws.bt_slices.ptr],
        &[batch, k, n],
    )?;
    let grid_gemm = batch * (m / OZAKI_TILE) * (n / OZAKI_TILE);
    peer.launch_wide_async(
        &kern.gemm,
        grid_gemm,
        GEMM_THREADS,
        &[ws.a_slices.ptr, ws.bt_slices.ptr, ws.rowexp.ptr, ws.colexp.ptr, c],
        &[batch, m, n, k],
    )
}

/// Batched GEMM over host buffers through the Ozaki path: `a` holds
/// `batch` `m x k` items, `b` `batch` `k x n` items; returns `batch`
/// `m x n` items.
#[allow(clippy::too_many_arguments)]
pub fn ozaki_gemm_batched(
    peer: &mut GpuPeer,
    kern: &OzakiKernels,
    a: &[f64],
    b: &[f64],
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<Vec<f64>, GpuPeerError> {
    check_shape(batch, m, n, k)?;
    let (bu, mu, nu, ku) = (batch as usize, m as usize, n as usize, k as usize);
    check_dim(a.len() == bu * mu * ku, "ozaki: a length")?;
    check_dim(b.len() == bu * ku * nu, "ozaki: b length")?;
    let ws = OzakiWorkspace::new(peer, batch, m, n, k)?;
    let pa = pin(peer, &f64_bytes(a))?;
    let pb = pin(peer, &f64_bytes(b))?;
    let pc = pin(peer, &vec![0u8; bu * mu * nu * 8])?;
    launch_ozaki_gemm(peer, kern, &ws, pa.ptr, pb.ptr, pc.ptr)?;
    peer.sync_wide()?;
    let mut out = vec![0u8; bu * mu * nu * 8];
    peer.fetch_bulk(&pc.handle, &mut out)?;
    pc.release(peer)?;
    pb.release(peer)?;
    pa.release(peer)?;
    ws.release(peer)?;
    Ok(bytes_f64(&out))
}

/// Per-element error bound of the scheme for one product, the
/// quantity the parity tests check against:
/// `2^-53 * k * max|A row| * max|B column|`.
pub fn error_bound(a_row: &[f64], b_col: &[f64]) -> f64 {
    let amax = a_row.iter().fold(0f64, |m, x| m.max(x.abs()));
    let bmax = b_col.iter().fold(0f64, |m, x| m.max(x.abs()));
    (a_row.len() as f64) * amax * bmax * 2f64.powi(-53)
}

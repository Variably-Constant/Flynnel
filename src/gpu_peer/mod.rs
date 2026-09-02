//! GPU-peer substrate: the GPU joins the scheduler as a shared-memory
//! peer over a CUDA-registered memory-mapped file.
//!
//! One physical region is simultaneously (a) a plain mapped file this
//! process and any other process can open, and (b) device-visible
//! memory a resident GPU kernel polls and writes. Work flows through
//! Lamport single-producer/single-consumer lanes with doorbell
//! signalling - no kernel launch on the per-message path, no atomics
//! across the CPU/GPU boundary (measured unsafe over PCIe hosts
//! without native atomics), and no data copies besides the payload
//! writes themselves.
//!
//! Every timing constant is HOST-CALIBRATED at [`GpuPeer::init`]:
//! doorbell round-trip, cross-device clock error, the Fischer
//! timed-lock margin (validated by a live contention self-test), the
//! launch baseline, and the system-atomics capability flag. Nothing
//! is baked from a reference machine; a host with a coherent CPU-GPU
//! link measures tighter constants and unlocks more capability
//! automatically. See [`PeerCalibration`].
//!
//! The consumer runs as a bounded-quantum persistent kernel
//! (watchdog-safe on display GPUs) that parks when idle and costs one
//! launch to wake; a continuously fed queue never pays the wake cost
//! (see the poller module docs).

pub mod calibration;
pub mod group;
pub mod hybrid;
pub mod l2_persist;
pub mod lanes;
pub mod linalg;
pub mod layout;
pub mod region;
pub mod timed_lock;
pub mod vram;

mod poller;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub use calibration::PeerCalibration;
pub use lanes::{LaneSet, RegionWords, Ticket};
pub use layout::{
    Geometry, OP_ADD1_F32, OP_ADD1_F32_V, OP_H2V, OP_NOP, OP_SUM_U32, OP_SUM_U32_V, OP_V2H,
    RESIDENT_PARAMS_BYTES, STATUS_DONE, STATUS_ERR,
};
pub use group::{GroupHandle, PeerGroup};
pub use l2_persist::{L2BenchReport, L2Capability, L2Persist};
// WideKernel is defined in this module; re-exported at the crate root
// alongside the other gpu_peer types via lib.rs.
pub use region::PeerRegion;
pub use vram::VramPool;

/// Pre-generated PTX for the peer kernels (driver-JIT'd at runtime;
/// regenerating after a kernels/gpu_peer.cu edit requires nvcc - see
/// that file's header).
pub(crate) const PEER_PTX: &str = include_str!("../../kernels/gpu_peer.ptx");

/// The kernel SOURCE, embedded so user opcodes can be NVRTC-composed
/// with the poller at init into one module (device-function linkage
/// requires a single compilation unit).
const PEER_CU: &str = include_str!("../../kernels/gpu_peer.cu");

/// Errors from the GPU-peer substrate.
#[derive(Debug)]
pub enum GpuPeerError {
    /// No usable CUDA device / driver.
    NoDevice(String),
    /// A CUDA driver call failed.
    Driver(String),
    /// Region file I/O failed.
    Io(std::io::Error),
    /// Payload exceeds the slot capacity for this geometry.
    PayloadTooLarge {
        /// Rejected payload length.
        len: usize,
        /// Slot payload capacity.
        max: usize,
    },
    /// Tickets must be reaped in submission order per lane.
    ReapOutOfOrder {
        /// Lane of the offending reap.
        lane: u32,
        /// The lane's oldest unreaped sequence.
        expected: u32,
        /// The sequence the caller tried to reap.
        got: u32,
    },
    /// A bounded wait expired.
    Timeout,
    /// The substrate cannot operate on this host (capability refused
    /// by calibration).
    Unavailable(&'static str),
}

impl core::fmt::Display for GpuPeerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoDevice(s) => write!(f, "no CUDA device: {s}"),
            Self::Driver(s) => write!(f, "CUDA driver error: {s}"),
            Self::Io(e) => write!(f, "region I/O error: {e}"),
            Self::PayloadTooLarge { len, max } => {
                write!(f, "payload {len} bytes exceeds slot capacity {max}")
            }
            Self::ReapOutOfOrder { lane, expected, got } => write!(
                f,
                "lane {lane} reap out of order: expected seq {expected}, got {got}"
            ),
            Self::Timeout => write!(f, "bounded wait expired"),
            Self::Unavailable(s) => write!(f, "substrate unavailable: {s}"),
        }
    }
}

impl std::error::Error for GpuPeerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Construction parameters. The defaults suit control-plane traffic
/// (4 KB slots); bulk streaming raises `slot_bytes`.
#[derive(Debug, Clone)]
pub struct GpuPeerConfig {
    /// Backing file for the shared region. Default: a per-process
    /// file in the OS temp directory, removed on drop. A caller-fixed
    /// path makes the region attachable by other processes.
    pub region_path: Option<PathBuf>,
    /// SPSC lane count (one consumer block each).
    pub lanes: u32,
    /// Slot size including the 16-byte descriptor.
    pub slot_bytes: u32,
    /// Ring depth per lane.
    pub slots_per_lane: u32,
    /// Poller quantum (bounded residency per launch).
    pub quantum_ns: u64,
    /// Idle time after which a resident quantum parks.
    pub idle_exit_ns: u64,
    /// CUDA device ordinal.
    pub device_ordinal: usize,
    /// Resident-pool block size (bytes of VRAM per block).
    pub vram_block_bytes: u32,
    /// Resident-pool block count (0 disables the pool).
    pub vram_blocks: u32,
    /// Blocks serving each lane. 1 keeps a lane on one SM, which suits
    /// many small ops; above 1 a lane is worked by a team of
    /// consecutive blocks so a single doorbell op spreads across the
    /// device. Rank 0 owns the ring and retires the slot once the
    /// whole team has finished; the user op receives its rank and the
    /// team size and strides its work over them.
    pub blocks_per_lane: u32,
    /// User opcode implementations as CUDA C source defining
    /// `extern "C" __device__ unsigned flynnel_user_op(unsigned op,
    /// unsigned char* block, unsigned count, volatile unsigned char*
    /// payload, unsigned team_rank, unsigned team_size)` - the
    /// six-argument team-aware hook the kernel forward-declares; a
    /// source defining any other arity fails the NVRTC compose with a
    /// duplicate-C-linkage error. When set, the poller is NVRTC-compiled at init
    /// together with this source and ops >= [`layout::OP_USER_BASE`]
    /// dispatch through it (called block-cooperatively by all 256
    /// threads). Requires the NVRTC runtime library on the host;
    /// `None` uses the pre-generated PTX and needs only the driver.
    pub user_ops_cuda: Option<String>,
}

impl Default for GpuPeerConfig {
    fn default() -> Self {
        Self {
            region_path: None,
            lanes: 4,
            slot_bytes: 4096,
            slots_per_lane: 64,
            quantum_ns: 250_000_000,
            idle_exit_ns: 2_000_000,
            device_ordinal: 0,
            vram_block_bytes: 65_536,
            vram_blocks: 1024,
            blocks_per_lane: 1,
            user_ops_cuda: None,
        }
    }
}

/// A full user kernel compiled for the wide-launch path.
///
/// The doorbell user-op ([`GpuPeer::submit_user`]) runs on ONE block
/// of 256 threads - one SM - which is right for many small
/// latency-sensitive ops but caps a single large data-parallel op
/// (a big convolution, a full-image stencil) at one SM. A
/// `WideKernel` is the complement: a caller-authored `__global__`
/// launched across a full grid over a resident block, so every SM
/// works the op while the data stays resident. The kernel keeps its
/// own grid-stride loop; [`GpuPeer::launch_wide`] sets the grid.
pub struct WideKernel {
    // Keeps the module alive for the function's lifetime.
    _module: Arc<CudaModule>,
    func: CudaFunction,
}

/// A block of DEVICE-RESIDENT data the scheduler owns by index. Data
/// pinned through [`GpuPeer::pin`] stays in the VRAM pool across any
/// number of tasks; each resident task moves only an 8-byte param
/// header over the bus. All tasks touching one handle ride the
/// handle's lane, so same-handle ordering (read-after-write,
/// write-after-write) is the lane's FIFO order - no extra
/// synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentHandle {
    block: u32,
    lane: u32,
    bytes: u32,
}

impl ResidentHandle {
    /// Bytes pinned in the block.
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes as usize
    }
    /// True when zero bytes are pinned.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
    /// The lane all of this handle's tasks ride (its dependency
    /// chain).
    #[inline]
    pub fn lane(&self) -> u32 {
        self.lane
    }
}

/// The GPU as a scheduler peer. Single-submitter surface (methods
/// take `&mut self`), matching the single-producer lane protocol.
pub struct GpuPeer {
    // Field order = drop order: poller shutdown flag is set in Drop
    // before region unregisters and the pool frees; module/stream/
    // context outlive all of them.
    region: PeerRegion,
    lane_set: LaneSet,
    poller: poller::Poller,
    pool: Option<VramPool>,
    calibration: PeerCalibration,
    _module: Arc<CudaModule>,
    _stream: Arc<CudaStream>,
    // Wide ops run on their OWN stream so they neither serialize behind
    // a resident poller quantum nor block doorbell traffic; the two run
    // concurrently, which is exactly when pause_poller matters.
    wide_stream: Arc<CudaStream>,
    _ctx: Arc<CudaContext>,
}

impl GpuPeer {
    /// Initialize the substrate: CUDA context, region creation +
    /// registration, kernel load, and the full host calibration
    /// (doorbell, clocks, Fischer self-test, atomics probe).
    ///
    /// Returns `Err` - never panics - when no device is present, so
    /// callers can fall back to CPU-only dispatch.
    pub fn init(config: GpuPeerConfig) -> Result<Self, GpuPeerError> {
        let ctx = CudaContext::new(config.device_ordinal)
            .map_err(|e| GpuPeerError::NoDevice(format!("{e:?}")))?;
        let stream = ctx.default_stream();
        let module = match &config.user_ops_cuda {
            None => match ctx.load_module(Ptx::from_src(PEER_PTX)) {
                Ok(m) => m,
                Err(ptx_err) => {
                    // A driver older than the toolchain that produced
                    // the checked-in PTX rejects it
                    // (CUDA_ERROR_UNSUPPORTED_PTX_VERSION); the host's
                    // own NVRTC emits PTX its driver accepts.
                    eprintln!(
                        "flynnel gpu_peer: checked-in PTX rejected ({ptx_err:?}); \
                         compiling the peer kernels with NVRTC instead"
                    );
                    let ptx = cudarc::nvrtc::compile_ptx(PEER_CU).map_err(|e| {
                        GpuPeerError::Driver(format!(
                            "PTX load: {ptx_err:?}; NVRTC fallback compile: {e:?}"
                        ))
                    })?;
                    ctx.load_module(ptx).map_err(|e| {
                        GpuPeerError::Driver(format!("NVRTC-fallback PTX load: {e:?}"))
                    })?
                }
            },
            Some(user_src) => {
                // Compose poller + user ops into ONE compilation unit
                // so the device-function call links, then JIT.
                let src = format!("#define FLYNNEL_USER_OPS 1\n{PEER_CU}\n{user_src}\n");
                let ptx = cudarc::nvrtc::compile_ptx(src).map_err(|e| {
                    GpuPeerError::Driver(format!("user-ops NVRTC compile: {e:?}"))
                })?;
                ctx.load_module(ptx)
                    .map_err(|e| GpuPeerError::Driver(format!("user-ops PTX load: {e:?}")))?
            }
        };
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| GpuPeerError::Driver(format!("kernel `{name}`: {e:?}")))
        };
        let f_poller = load("flynnel_peer_poller")?;
        let kernels = calibration::CalibKernels {
            calib_pong: load("flynnel_peer_calib_pong")?,
            fischer: load("flynnel_peer_fischer")?,
            cas_probe: load("flynnel_peer_cas_probe")?,
        };

        let geometry = Geometry {
            lanes: config.lanes.max(1),
            slot_bytes: config.slot_bytes.max(64),
            slots_per_lane: config.slots_per_lane.max(2),
        };
        let (path, remove_on_drop) = match &config.region_path {
            Some(p) => (p.clone(), false),
            None => {
                // Unique per instance, not just per process: a peer
                // GROUP creates several regions in one process.
                static INSTANCE: core::sync::atomic::AtomicU32 =
                    core::sync::atomic::AtomicU32::new(0);
                let n = INSTANCE.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                (
                    std::env::temp_dir().join(format!(
                        "flynnel_gpu_peer_{}_{n}.bin",
                        std::process::id()
                    )),
                    true,
                )
            }
        };
        let region = PeerRegion::create(&ctx, &path, geometry, remove_on_drop)?;
        let calibration = calibration::calibrate(&region, &stream, &kernels)?;
        if !calibration.doorbell_ok {
            return Err(GpuPeerError::Unavailable(
                "doorbell handshake failed during calibration",
            ));
        }

        let pool = if config.vram_blocks > 0 {
            Some(VramPool::new(
                &ctx,
                &stream,
                config.vram_block_bytes.max(64),
                config.vram_blocks,
            )?)
        } else {
            None
        };
        let (vbase, vbytes, vblocks) = match &pool {
            Some(p) => (p.base(), p.block_bytes(), p.blocks()),
            None => (0, 0, 0),
        };
        let lane_set = LaneSet::new(geometry);
        let poller = poller::Poller::new(
            Arc::clone(&stream),
            f_poller,
            config.quantum_ns,
            config.idle_exit_ns,
            vbase,
            vbytes,
            vblocks,
            config.blocks_per_lane,
        );
        let wide_stream = ctx
            .new_stream()
            .map_err(|e| GpuPeerError::Driver(format!("wide stream: {e:?}")))?;
        Ok(Self {
            region,
            lane_set,
            poller,
            pool,
            calibration,
            _module: module,
            _stream: stream,
            wide_stream,
            _ctx: ctx,
        })
    }

    /// The host-measured constants and capability flags.
    #[inline]
    pub fn calibration(&self) -> PeerCalibration {
        self.calibration
    }

    /// Region geometry.
    #[inline]
    pub fn geometry(&self) -> Geometry {
        self.region.geometry()
    }

    /// The shared region (attachment path, offset accessors).
    #[inline]
    pub fn region(&self) -> &PeerRegion {
        &self.region
    }

    /// Submit `payload` under a built-in opcode. Blocks (bounded) on
    /// backpressure when every lane is full.
    pub fn submit(&mut self, op: u32, payload: &[u8]) -> Result<Ticket, GpuPeerError> {
        let t0 = Instant::now();
        loop {
            if let Some(t) = self.lane_set.try_submit(&self.region, op, payload)? {
                self.poller.ensure_running(&self.region)?;
                return Ok(t);
            }
            self.poller.ensure_running(&self.region)?;
            if t0.elapsed() > Duration::from_secs(10) {
                return Err(GpuPeerError::Timeout);
            }
            std::thread::yield_now();
        }
    }

    /// True when the consumer has completed `ticket`.
    #[inline]
    pub fn is_done(&self, ticket: Ticket) -> bool {
        self.lane_set.is_done(&self.region, ticket)
    }

    /// Wait (bounded) for completion; returns the slot status word.
    pub fn wait(&mut self, ticket: Ticket, timeout: Duration) -> Result<u32, GpuPeerError> {
        let t0 = Instant::now();
        let mut spins = 0u32;
        while !self.lane_set.is_done(&self.region, ticket) {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(4096) {
                // Self-heal the exit-vs-new-work race: if the quantum
                // idled out between our submit and its poll, relaunch.
                self.poller.ensure_running(&self.region)?;
                if t0.elapsed() > timeout {
                    return Err(GpuPeerError::Timeout);
                }
            }
            core::hint::spin_loop();
        }
        Ok(self.lane_set.status(&self.region, ticket))
    }

    /// Copy a completed ticket's result payload into `dst`.
    pub fn read_result(&self, ticket: Ticket, dst: &mut [u8]) {
        self.lane_set.read_result(&self.region, ticket, dst);
    }

    /// Release the ticket's slot for reuse (in submission order per
    /// lane).
    pub fn reap(&mut self, ticket: Ticket) -> Result<(), GpuPeerError> {
        self.lane_set.reap(ticket)
    }

    /// Submit on a SPECIFIC lane (bounded backpressure wait).
    fn submit_on_lane(
        &mut self,
        lane: u32,
        op: u32,
        payload: &[u8],
    ) -> Result<Ticket, GpuPeerError> {
        let t0 = Instant::now();
        loop {
            if let Some(t) = self.lane_set.try_submit_on(&self.region, lane, op, payload)? {
                self.poller.ensure_running(&self.region)?;
                return Ok(t);
            }
            self.poller.ensure_running(&self.region)?;
            if t0.elapsed() > Duration::from_secs(10) {
                return Err(GpuPeerError::Timeout);
            }
            std::thread::yield_now();
        }
    }

    /// Pin `data` into a device-resident block. Synchronous (waits
    /// for the upload); requires the assigned lane to have no
    /// unreaped tickets outstanding.
    pub fn pin(&mut self, data: &[u8]) -> Result<ResidentHandle, GpuPeerError> {
        let pool = self
            .pool
            .as_mut()
            .ok_or(GpuPeerError::Unavailable("resident pool disabled"))?;
        let block_bytes = pool.block_bytes() as usize;
        if data.len() > block_bytes
            || data.len() + RESIDENT_PARAMS_BYTES > self.region.geometry().payload_max()
        {
            return Err(GpuPeerError::PayloadTooLarge {
                len: data.len(),
                max: block_bytes.min(
                    self.region.geometry().payload_max() - RESIDENT_PARAMS_BYTES,
                ),
            });
        }
        let block = pool
            .alloc()
            .ok_or(GpuPeerError::Unavailable("resident pool exhausted"))?;
        let lane = block % self.region.geometry().lanes;
        let mut payload = Vec::with_capacity(RESIDENT_PARAMS_BYTES + data.len());
        payload.extend_from_slice(&block.to_le_bytes());
        payload.extend_from_slice(&(data.len() as u32).to_le_bytes());
        payload.extend_from_slice(data);
        let t = self.submit_on_lane(lane, OP_H2V, &payload)?;
        let status = self.wait(t, Duration::from_secs(10))?;
        self.reap(t)?;
        if status != STATUS_DONE {
            if let Some(p) = self.pool.as_mut() {
                p.release(block);
            }
            return Err(GpuPeerError::Unavailable("resident upload rejected"));
        }
        Ok(ResidentHandle { block, lane, bytes: data.len() as u32 })
    }

    /// Re-upload `data` into an EXISTING resident block (synchronous
    /// H2V on the handle's lane). The residence-flip primitive: a
    /// host-modified mirror pushes its bytes back to the device
    /// without re-allocating. `data` must not exceed the handle's
    /// pinned length.
    pub fn write_resident(
        &mut self,
        handle: &ResidentHandle,
        data: &[u8],
    ) -> Result<(), GpuPeerError> {
        if data.len() > handle.len() {
            return Err(GpuPeerError::PayloadTooLarge { len: data.len(), max: handle.len() });
        }
        let mut payload = Vec::with_capacity(RESIDENT_PARAMS_BYTES + data.len());
        payload.extend_from_slice(&handle.block.to_le_bytes());
        payload.extend_from_slice(&(data.len() as u32).to_le_bytes());
        payload.extend_from_slice(data);
        let t = self.submit_on_lane(handle.lane, layout::OP_H2V, &payload)?;
        let status = self.wait(t, Duration::from_secs(10))?;
        self.reap(t)?;
        if status != STATUS_DONE {
            return Err(GpuPeerError::Unavailable("resident re-upload rejected"));
        }
        Ok(())
    }

    /// Pin a buffer of ANY size straight into VRAM, bypassing the
    /// doorbell.
    ///
    /// [`Self::pin`] carries its data in a slot payload, so it is
    /// capped by `slot_bytes` - fine for the kilobyte operands a
    /// doorbell op takes, useless for a corpus. This copies host to
    /// device directly and spans as many consecutive pool blocks as
    /// the data needs, which is what a workload that must stay
    /// RESIDENT across many calls requires: upload once, query
    /// forever, and only the query's own arguments ever cross again.
    ///
    /// The returned handle names the FIRST block; `resident_ptr` gives
    /// its device address and the span is contiguous by construction.
    pub fn pin_bulk(&mut self, data: &[u8]) -> Result<ResidentHandle, GpuPeerError> {
        let pool = self
            .pool
            .as_mut()
            .ok_or(GpuPeerError::Unavailable("resident pool disabled"))?;
        let block_bytes = pool.block_bytes() as usize;
        let need = data.len().div_ceil(block_bytes.max(1)).max(1);
        let first = pool
            .alloc_span(need as u32)
            .ok_or(GpuPeerError::Unavailable("no contiguous resident span"))?;
        let dst = pool.block_ptr(first);
        // SAFETY: `dst` is the pool's own device allocation and the
        // span was just claimed, so it covers `data.len()` bytes.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                dst,
                data,
                self.wide_stream.cu_stream() as _,
            )
            .map_err(|e| GpuPeerError::Driver(format!("pin_bulk htod: {e:?}")))?;
        }
        self.wide_stream
            .synchronize()
            .map_err(|e| GpuPeerError::Driver(format!("pin_bulk sync: {e:?}")))?;
        Ok(ResidentHandle {
            block: first,
            lane: first % self.region.geometry().lanes,
            bytes: data.len() as u32,
        })
    }

    /// Overwrite a [`Self::pin_bulk`] span host-to-device directly,
    /// for the small per-call operands a resident workload still
    /// changes (a query's needle, a result counter's reset).
    pub fn write_resident_bulk(
        &mut self,
        handle: &ResidentHandle,
        data: &[u8],
    ) -> Result<(), GpuPeerError> {
        let (dst, _) = self.resident_ptr(handle)?;
        // SAFETY: `dst` is the pool's own device span for this handle.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                dst,
                data,
                self.wide_stream.cu_stream() as _,
            )
            .map_err(|e| GpuPeerError::Driver(format!("write_resident_bulk: {e:?}")))?;
        }
        self.wide_stream
            .synchronize()
            .map_err(|e| GpuPeerError::Driver(format!("write_resident_bulk sync: {e:?}")))
    }

    /// Read a [`Self::pin_bulk`] span device-to-host directly.
    pub fn fetch_bulk(
        &mut self,
        handle: &ResidentHandle,
        out: &mut [u8],
    ) -> Result<(), GpuPeerError> {
        let (src, _) = self.resident_ptr(handle)?;
        // SAFETY: `src` is the pool's own device span for this handle.
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(
                out,
                src,
                self.wide_stream.cu_stream() as _,
            )
            .map_err(|e| GpuPeerError::Driver(format!("fetch_bulk: {e:?}")))?;
        }
        self.wide_stream
            .synchronize()
            .map_err(|e| GpuPeerError::Driver(format!("fetch_bulk sync: {e:?}")))
    }

    /// [`Self::pin`] WITHOUT waiting: zero-synchronization prefetch.
    /// The upload rides the handle's lane, and lane FIFO order IS the
    /// dependency order - any task submitted on this handle
    /// afterwards executes after the data has landed, with no fence,
    /// no event, no wait anywhere. The returned upload ticket must be
    /// reaped first among the lane's tickets (in-order reap rule).
    pub fn pin_prefetch(
        &mut self,
        data: &[u8],
    ) -> Result<(ResidentHandle, Ticket), GpuPeerError> {
        let pool = self
            .pool
            .as_mut()
            .ok_or(GpuPeerError::Unavailable("resident pool disabled"))?;
        let block_bytes = pool.block_bytes() as usize;
        if data.len() > block_bytes
            || data.len() + RESIDENT_PARAMS_BYTES > self.region.geometry().payload_max()
        {
            return Err(GpuPeerError::PayloadTooLarge {
                len: data.len(),
                max: block_bytes
                    .min(self.region.geometry().payload_max() - RESIDENT_PARAMS_BYTES),
            });
        }
        let block = pool
            .alloc()
            .ok_or(GpuPeerError::Unavailable("resident pool exhausted"))?;
        let lane = block % self.region.geometry().lanes;
        let mut payload = Vec::with_capacity(RESIDENT_PARAMS_BYTES + data.len());
        payload.extend_from_slice(&block.to_le_bytes());
        payload.extend_from_slice(&(data.len() as u32).to_le_bytes());
        payload.extend_from_slice(data);
        let t = self.submit_on_lane(lane, layout::OP_H2V, &payload)?;
        Ok((ResidentHandle { block, lane, bytes: data.len() as u32 }, t))
    }

    /// Submit a USER opcode (>= [`layout::OP_USER_BASE`], implemented
    /// by the CUDA source registered via
    /// [`GpuPeerConfig::user_ops_cuda`]). With a handle, the task
    /// rides the handle's lane (ordered with its other tasks) and the
    /// hook receives the resident block; without one it round-robins
    /// and the hook receives a null block. `args` land at payload+8
    /// (the hook's argument/result space).
    pub fn submit_user(
        &mut self,
        op: u32,
        handle: Option<&ResidentHandle>,
        args: &[u8],
    ) -> Result<Ticket, GpuPeerError> {
        if op < layout::OP_USER_BASE {
            return Err(GpuPeerError::Unavailable("op below OP_USER_BASE"));
        }
        let (block, count, lane) = match handle {
            Some(h) => (h.block, h.bytes, Some(h.lane)),
            None => (layout::NO_BLOCK, 0u32, None),
        };
        let mut payload = Vec::with_capacity(RESIDENT_PARAMS_BYTES + args.len());
        payload.extend_from_slice(&block.to_le_bytes());
        payload.extend_from_slice(&count.to_le_bytes());
        payload.extend_from_slice(args);
        match lane {
            Some(l) => self.submit_on_lane(l, op, &payload),
            None => self.submit(op, &payload),
        }
    }

    /// Submit a resident-block task (`OP_ADD1_F32_V` / `OP_SUM_U32_V`).
    /// Only the 8-byte param header crosses the bus; the data stays
    /// in VRAM. Tasks on one handle execute in submission order (lane
    /// FIFO).
    pub fn submit_resident(
        &mut self,
        op: u32,
        handle: &ResidentHandle,
    ) -> Result<Ticket, GpuPeerError> {
        let mut params = [0u8; RESIDENT_PARAMS_BYTES];
        params[..4].copy_from_slice(&handle.block.to_le_bytes());
        params[4..].copy_from_slice(&handle.bytes.to_le_bytes());
        self.submit_on_lane(handle.lane, op, &params)
    }

    /// Download a resident block into `out` (synchronous; same lane
    /// discipline as [`Self::pin`]). `out` receives
    /// `min(out.len(), handle.len())` bytes.
    pub fn fetch(
        &mut self,
        handle: &ResidentHandle,
        out: &mut [u8],
    ) -> Result<(), GpuPeerError> {
        let mut params = [0u8; RESIDENT_PARAMS_BYTES];
        params[..4].copy_from_slice(&handle.block.to_le_bytes());
        params[4..].copy_from_slice(&handle.bytes.to_le_bytes());
        let t = self.submit_on_lane(handle.lane, OP_V2H, &params)?;
        let status = self.wait(t, Duration::from_secs(10))?;
        if status != STATUS_DONE {
            self.reap(t)?;
            return Err(GpuPeerError::Unavailable("resident download rejected"));
        }
        let n = out.len().min(handle.len());
        let mut buf = vec![0u8; RESIDENT_PARAMS_BYTES + n];
        self.read_result(t, &mut buf);
        out[..n].copy_from_slice(&buf[RESIDENT_PARAMS_BYTES..]);
        self.reap(t)?;
        Ok(())
    }

    /// Return the handle's block to the pool. The caller is done
    /// with the data (any still-queued tasks on the lane complete
    /// first by lane order before a new pin can reuse the block's
    /// lane slot).
    pub fn unpin(&mut self, handle: ResidentHandle) -> Result<(), GpuPeerError> {
        let pool = self
            .pool
            .as_mut()
            .ok_or(GpuPeerError::Unavailable("resident pool disabled"))?;
        // A pin_bulk handle spans every block its bytes cover.
        let span = (handle.bytes as usize)
            .div_ceil(pool.block_bytes().max(1) as usize)
            .max(1) as u32;
        for b in handle.block..handle.block + span {
            pool.release(b);
        }
        Ok(())
    }

    /// Resident-pool stats: (free blocks, total blocks); zeros when
    /// the pool is disabled.
    pub fn pool_stats(&self) -> (usize, u32) {
        match &self.pool {
            Some(p) => (p.free_blocks(), p.blocks()),
            None => (0, 0),
        }
    }

    /// Raw device address and byte length of a resident handle's
    /// block. A wide-launch kernel targets this base directly.
    pub fn resident_ptr(&self, handle: &ResidentHandle) -> Result<(u64, usize), GpuPeerError> {
        let pool = self
            .pool
            .as_ref()
            .ok_or(GpuPeerError::Unavailable("resident pool disabled"))?;
        Ok((pool.block_ptr(handle.block), handle.len()))
    }

    /// NVRTC-compile a caller-authored full `__global__` kernel for
    /// the wide-launch path. The source stands alone (it is not
    /// composed with the poller); `entry` names the `extern "C"`
    /// entry point. Requires the NVRTC runtime; returns a driver
    /// error where it is absent.
    ///
    /// The kernel's signature is `(T0* p0, .., Tn* pn, u32 s0, ..)` -
    /// pointer arguments first (device addresses from
    /// [`Self::resident_ptr`]), then u32 scalars - matching the
    /// `ptrs` and `scalars` passed to [`Self::launch_wide`]. Write a
    /// grid-stride loop so any grid size is correct.
    pub fn compile_wide_kernel(&self, src: &str, entry: &str) -> Result<WideKernel, GpuPeerError> {
        let ptx = cudarc::nvrtc::compile_ptx(src)
            .map_err(|e| GpuPeerError::Driver(format!("wide-kernel NVRTC compile: {e:?}")))?;
        self.load_wide_kernel(ptx, entry)
    }

    /// Load a wide-launch kernel from PTX text (pre-generated, driver-
    /// JIT'd; no NVRTC needed). Same signature contract as
    /// [`Self::compile_wide_kernel`].
    pub fn load_wide_kernel_ptx(&self, ptx: &str, entry: &str) -> Result<WideKernel, GpuPeerError> {
        self.load_wide_kernel(Ptx::from_src(ptx), entry)
    }

    fn load_wide_kernel(&self, ptx: Ptx, entry: &str) -> Result<WideKernel, GpuPeerError> {
        let module = self
            ._ctx
            .load_module(ptx)
            .map_err(|e| GpuPeerError::Driver(format!("wide-kernel PTX load: {e:?}")))?;
        let func = module
            .load_function(entry)
            .map_err(|e| GpuPeerError::Driver(format!("wide-kernel entry `{entry}`: {e:?}")))?;
        Ok(WideKernel { _module: module, func })
    }

    /// The CUDA context this peer runs on, for consumers that bind
    /// their own library handles or allocate device memory the wide
    /// kernels then read.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self._ctx
    }

    /// The stream wide launches run on. Work a consumer enqueues here
    /// (its own kernels, library calls) is FIFO-ordered with
    /// [`Self::launch_wide_async`] launches and fenced by
    /// [`Self::sync_wide`].
    pub fn wide_stream(&self) -> &Arc<CudaStream> {
        &self.wide_stream
    }

    /// Enqueue a [`WideKernel`] on the resident stream WITHOUT
    /// synchronizing. Pointer arguments come first, then u32 scalars,
    /// matching the kernel signature; `grid_blocks` = 0 auto-sizes the
    /// grid from `scalars[0]` (an element count).
    fn launch_wide_inner(
        &self,
        kernel: &WideKernel,
        grid_blocks: u32,
        block_threads: u32,
        ptrs: &[u64],
        scalars: &[u32],
    ) -> Result<(), GpuPeerError> {
        let block = block_threads.clamp(1, 1024);
        let grid = if grid_blocks == 0 {
            scalars.first().copied().unwrap_or(1).div_ceil(block).max(1)
        } else {
            grid_blocks
        };
        let mut b = self.wide_stream.launch_builder(&kernel.func);
        for p in ptrs {
            b.arg(p);
        }
        for s in scalars {
            b.arg(s);
        }
        // SAFETY: the caller's kernel signature matches (ptrs as
        // device-pointer args, then u32 scalars); the arg values live
        // in the caller's slices for the duration of the launch.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| GpuPeerError::Driver(format!("wide launch: {e:?}")))?;
        Ok(())
    }

    /// Launch a [`WideKernel`] across `grid_blocks x block_threads`
    /// threads and block until it finishes. This is the full-device
    /// path a large resident op takes instead of the one-block
    /// doorbell op. One-off convenience: it pays a stream sync (a
    /// WDDM command-buffer flush) per call. For a chain of many small
    /// dependent kernels use [`Self::launch_wide_async`] +
    /// [`Self::sync_wide`], which pays ONE flush for the whole batch.
    pub fn launch_wide(
        &self,
        kernel: &WideKernel,
        grid_blocks: u32,
        block_threads: u32,
        ptrs: &[u64],
        scalars: &[u32],
    ) -> Result<(), GpuPeerError> {
        self.launch_wide_inner(kernel, grid_blocks, block_threads, ptrs, scalars)?;
        self.sync_wide()
    }

    /// Enqueue a [`WideKernel`] on the resident stream and return
    /// immediately - no sync. The resident stream is FIFO, so kernels
    /// queued back to back run in order and a dependent chain is
    /// correct without a per-kernel sync. Queue the whole batch, then
    /// call [`Self::sync_wide`] once. This is the WDDM-friendly path
    /// for a many-small-dependent-kernel workload, where a sync per
    /// call would flush the command buffer N times.
    pub fn launch_wide_async(
        &self,
        kernel: &WideKernel,
        grid_blocks: u32,
        block_threads: u32,
        ptrs: &[u64],
        scalars: &[u32],
    ) -> Result<(), GpuPeerError> {
        self.launch_wide_inner(kernel, grid_blocks, block_threads, ptrs, scalars)
    }

    /// Block until every enqueued wide launch on the resident stream
    /// has finished. One flush for a whole [`Self::launch_wide_async`]
    /// batch.
    pub fn sync_wide(&self) -> Result<(), GpuPeerError> {
        self.wide_stream
            .synchronize()
            .map_err(|e| GpuPeerError::Driver(format!("wide sync: {e:?}")))
    }

    /// Quiesce the doorbell poller: force its resident quantum to exit
    /// and hold off relaunches until [`Self::resume_poller`]. The
    /// poller busy-polls its lanes while resident, so a live poller
    /// steals SM occupancy and L2 bandwidth from a concurrent wide op.
    /// Pause it around a heavy wide batch to hand the whole device to
    /// that batch, then resume. Small doorbell ops submitted while
    /// paused simply queue and are consumed after resume.
    pub fn pause_poller(&mut self) -> Result<(), GpuPeerError> {
        self.poller.pause(&self.region)
    }

    /// Undo [`Self::pause_poller`]; the next submit relaunches the
    /// poller.
    pub fn resume_poller(&mut self) {
        self.poller.resume();
    }

    /// Whether the doorbell poller is currently paused.
    #[inline]
    pub fn poller_paused(&self) -> bool {
        self.poller.is_paused()
    }

    /// The device's L2-persistence ceilings (set-aside + window max).
    pub fn l2_capability(&self) -> Result<L2Capability, GpuPeerError> {
        L2Capability::query(self._ctx_ref())
    }

    /// Measure L2 persistence on this device: the same hammer kernel
    /// timed with the hot working set pinned in L2 versus streaming.
    /// See [`l2_persist::benchmark`].
    pub fn l2_benchmark(
        &self,
        hot_bytes: usize,
        pol_bytes: usize,
        iters: u32,
        runs: u32,
    ) -> Result<L2BenchReport, GpuPeerError> {
        l2_persist::benchmark(self._ctx_ref(), hot_bytes, pol_bytes, iters, runs)
    }

    #[inline]
    fn _ctx_ref(&self) -> &Arc<CudaContext> {
        &self._ctx
    }

    /// Acquire the region's Fischer timed lock (cross-device mutual
    /// exclusion without atomics) at the CALIBRATED margin. Only
    /// available when the calibration self-test granted the
    /// capability.
    pub fn timed_lock_acquire(&self, timeout: Duration) -> Result<(), GpuPeerError> {
        if !self.calibration.timed_lock_ok {
            return Err(GpuPeerError::Unavailable("timed lock not validated on this host"));
        }
        if timed_lock::acquire(
            &self.region,
            Duration::from_nanos(self.calibration.delta_ns),
            timeout,
        ) {
            Ok(())
        } else {
            Err(GpuPeerError::Timeout)
        }
    }

    /// Release the region's Fischer timed lock.
    pub fn timed_lock_release(&self) {
        timed_lock::release(&self.region);
    }
}

impl Drop for GpuPeer {
    fn drop(&mut self) {
        self.poller.shutdown(&self.region);
    }
}

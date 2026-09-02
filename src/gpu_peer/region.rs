//! The shared region: a memory-mapped file registered with the CUDA
//! driver so the SAME physical pages are addressable by this process,
//! by other processes mapping the same file, and by GPU kernels.
//!
//! Registration uses `cuMemHostRegister_v2(.., DEVICEMAP)` through
//! cudarc's dynamically loaded driver bindings - no CUDA toolkit is
//! involved at build time. The device observes the region at a
//! DIFFERENT virtual address (`cuMemHostGetDevicePointer_v2`), so all
//! cross-device state is exchanged as byte offsets and translated on
//! each side ([`PeerRegion::dev_base`] + offset for kernel arguments).
//!
//! Drop order is load-bearing: unregister BEFORE the mapping unmaps
//! and the file closes (mirrors the mmap-then-file teardown rule the
//! storage tier uses on Windows).

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;
use memmap2::MmapMut;

use super::GpuPeerError;
use super::layout::{self, Geometry};

/// `CU_MEMHOSTREGISTER_DEVICEMAP`: map the registered range into the
/// device address space (cuda.h flag value).
const REGISTER_DEVICEMAP: u32 = 0x02;

/// A file-backed, page-locked, GPU-visible shared region.
pub struct PeerRegion {
    ctx: Arc<CudaContext>,
    map: MmapMut,
    dev_base: u64,
    geometry: Geometry,
    path: PathBuf,
    remove_on_drop: bool,
    registered: bool,
}

impl PeerRegion {
    /// Create (or truncate) the backing file at `path`, size it for
    /// `geometry`, pre-fault every page (registration pins resident
    /// pages), zero the control state, and register the mapping with
    /// the CUDA driver.
    pub fn create(
        ctx: &Arc<CudaContext>,
        path: &Path,
        geometry: Geometry,
        remove_on_drop: bool,
    ) -> Result<Self, GpuPeerError> {
        let bytes = geometry.region_bytes();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(GpuPeerError::Io)?;
        file.set_len(bytes as u64).map_err(GpuPeerError::Io)?;
        // SAFETY: the file was just created with the exact length; the
        // mapping is private to this struct and outlives all raw-offset
        // accessors, which never form references over shared words.
        let mut map = unsafe { MmapMut::map_mut(&file).map_err(GpuPeerError::Io)? };

        // Pre-fault: touch every page so the whole range is resident
        // when the driver pins it.
        for off in (0..bytes).step_by(4096) {
            map[off] = 0;
        }

        let mut region = Self {
            ctx: Arc::clone(ctx),
            map,
            dev_base: 0,
            geometry,
            path: path.to_path_buf(),
            remove_on_drop,
            registered: false,
        };
        region.write_header();
        region.register()?;
        Ok(region)
    }

    fn write_header(&mut self) {
        self.store_u64(layout::HDR_MAGIC_OFF, layout::MAGIC);
        self.store_u32(layout::HDR_VERSION_OFF, layout::VERSION);
        self.store_u32(layout::HDR_LANES_OFF, self.geometry.lanes);
        self.store_u32(layout::HDR_SLOT_BYTES_OFF, self.geometry.slot_bytes);
        self.store_u32(layout::HDR_SLOTS_PER_OFF, self.geometry.slots_per_lane);
    }

    fn register(&mut self) -> Result<(), GpuPeerError> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| GpuPeerError::Driver(format!("bind_to_thread: {e:?}")))?;
        let base = self.map.as_mut_ptr() as *mut core::ffi::c_void;
        let bytes = self.geometry.region_bytes();
        // SAFETY: `base..base+bytes` is a valid private mapping owned by
        // `self.map`, which outlives the registration (drop unregisters
        // first). DEVICEMAP is the documented flag for device-visible
        // registration.
        let r = unsafe { cu::cuMemHostRegister_v2(base, bytes, REGISTER_DEVICEMAP) };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(GpuPeerError::Driver(format!("cuMemHostRegister_v2: {r:?}")));
        }
        self.registered = true;
        let mut dptr: cu::CUdeviceptr = 0;
        // SAFETY: `base` was successfully registered above.
        let r = unsafe { cu::cuMemHostGetDevicePointer_v2(&mut dptr, base, 0) };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(GpuPeerError::Driver(format!(
                "cuMemHostGetDevicePointer_v2: {r:?}"
            )));
        }
        self.dev_base = dptr as u64;
        Ok(())
    }

    /// Device-side base address for kernel arguments.
    #[inline]
    pub fn dev_base(&self) -> u64 {
        self.dev_base
    }

    /// Region geometry.
    #[inline]
    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    /// Backing file path (other processes open this to attach).
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The owning CUDA context.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    #[inline]
    fn base_ptr(&self) -> *mut u8 {
        self.map.as_ptr() as *mut u8
    }

    /// Host-side base address of the mapping. Crate-internal: the
    /// calibration CAS probe forms `AtomicU32` views over region
    /// words; everything else addresses the region by offset through
    /// the typed accessors.
    #[inline]
    pub(crate) fn base_addr(&self) -> *mut u8 {
        self.base_ptr()
    }

    /// Volatile u32 load at a byte offset.
    #[inline]
    pub fn load_u32(&self, off: usize) -> u32 {
        debug_assert!(off + 4 <= self.geometry.region_bytes());
        // SAFETY: offset bounds asserted; shared words are only ever
        // accessed volatilely (no references formed), matching the
        // GPU's concurrent volatile accesses.
        unsafe { (self.base_ptr().add(off) as *const u32).read_volatile() }
    }

    /// Volatile u32 store at a byte offset.
    #[inline]
    pub fn store_u32(&self, off: usize, v: u32) {
        debug_assert!(off + 4 <= self.geometry.region_bytes());
        // SAFETY: as in `load_u32`.
        unsafe { (self.base_ptr().add(off) as *mut u32).write_volatile(v) }
    }

    /// Volatile u64 load at a byte offset.
    #[inline]
    pub fn load_u64(&self, off: usize) -> u64 {
        debug_assert!(off + 8 <= self.geometry.region_bytes());
        // SAFETY: as in `load_u32`.
        unsafe { (self.base_ptr().add(off) as *const u64).read_volatile() }
    }

    /// Volatile u64 store at a byte offset.
    #[inline]
    pub fn store_u64(&self, off: usize, v: u64) {
        debug_assert!(off + 8 <= self.geometry.region_bytes());
        // SAFETY: as in `load_u32`.
        unsafe { (self.base_ptr().add(off) as *mut u64).write_volatile(v) }
    }

    /// Volatile i32 load at a byte offset (Fischer occupancy word).
    #[inline]
    pub fn load_i32(&self, off: usize) -> i32 {
        self.load_u32(off) as i32
    }

    /// Volatile i32 store at a byte offset (Fischer occupancy word).
    #[inline]
    pub fn store_i32(&self, off: usize, v: i32) {
        self.store_u32(off, v as u32);
    }

    /// Copy `src` into the region at `off` (payload writes; the
    /// release fence + index store come AFTER this returns).
    #[inline]
    pub fn write_bytes(&self, off: usize, src: &[u8]) {
        debug_assert!(off + src.len() <= self.geometry.region_bytes());
        // SAFETY: bounds asserted; the destination slot is owned by the
        // producer until the index store publishes it.
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), self.base_ptr().add(off), src.len());
        }
    }

    /// Copy `len` bytes from the region at `off` into `dst`.
    #[inline]
    pub fn read_bytes(&self, off: usize, dst: &mut [u8]) {
        debug_assert!(off + dst.len() <= self.geometry.region_bytes());
        // SAFETY: bounds asserted; the source slot was released to the
        // producer by the consumer's index store (acquire observed).
        unsafe {
            core::ptr::copy_nonoverlapping(self.base_ptr().add(off), dst.as_mut_ptr(), dst.len());
        }
    }

    /// Full store fence: payload writes drain before the index store
    /// that publishes them (x86: compiler barrier + sfence).
    #[inline]
    pub fn release_fence(&self) {
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for PeerRegion {
    fn drop(&mut self) {
        if self.registered {
            let base = self.map.as_mut_ptr() as *mut core::ffi::c_void;
            // SAFETY: registered in `register`; unregister must precede
            // the MmapMut drop (unmap) below. Failure here is
            // unactionable during teardown (context may already be
            // gone); the unmap still proceeds.
            let _rc: cu::CUresult = unsafe { cu::cuMemHostUnregister(base) };
            self.registered = false;
        }
        if self.remove_on_drop {
            // Mapping drops with `self.map` after this body; file
            // removal is best-effort (another attached process keeps
            // the contents alive on its own handle).
            drop(std::fs::remove_file(&self.path));
        }
    }
}

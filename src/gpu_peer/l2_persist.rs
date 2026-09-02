//! GPU L2-persistence: reserve a slice of L2 and pin a hot address
//! window in it.
//!
//! This is the GPU analog of reserving cache residency for a working
//! set. On compute capability 8.0 and up, the driver lets you set
//! aside part of L2 and mark a contiguous device-address window as
//! persisting, so its lines are preferentially retained while
//! streaming traffic from a noisy co-runner cannot evict them. The
//! VRAM pool already keeps a block device-resident; this keeps the
//! HOTTEST block resident one level up, in L2.
//!
//! Every knob is measured, not assumed: [`L2Capability::query`]
//! reports the device's set-aside ceiling and access-window ceiling,
//! and callers clamp to them. Where the device does not support L2
//! persistence the capability reads zero and the levers are refused.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;

use super::GpuPeerError;

/// `CU_LIMIT_PERSISTING_L2_CACHE_SIZE`.
const LIMIT_PERSISTING_L2: cu::CUlimit = cu::CUlimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE;

/// What this device allows for L2 persistence, in bytes. Both fields
/// read zero on hardware without the feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2Capability {
    /// Largest L2 slice that can be set aside for persisting lines.
    pub max_persisting_l2: usize,
    /// Largest access-policy window the device honors.
    pub max_access_window: usize,
}

impl L2Capability {
    /// Whether the device supports L2 persistence at all.
    #[inline]
    pub fn supported(&self) -> bool {
        self.max_persisting_l2 > 0 && self.max_access_window > 0
    }

    /// Query the running device (through the given context).
    pub fn query(ctx: &Arc<CudaContext>) -> Result<Self, GpuPeerError> {
        ctx.bind_to_thread()
            .map_err(|e| GpuPeerError::Driver(format!("bind_to_thread: {e:?}")))?;
        let mut dev: cu::CUdevice = 0;
        // SAFETY: a context is current after bind_to_thread.
        let r = unsafe { cu::cuCtxGetDevice(&mut dev) };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(GpuPeerError::Driver(format!("cuCtxGetDevice: {r:?}")));
        }
        let attr = |a: cu::CUdevice_attribute| -> usize {
            let mut v: i32 = 0;
            // SAFETY: valid device handle from cuCtxGetDevice.
            let r = unsafe { cu::cuDeviceGetAttribute(&mut v, a, dev) };
            if r == cu::CUresult::CUDA_SUCCESS && v > 0 { v as usize } else { 0 }
        };
        Ok(Self {
            max_persisting_l2: attr(
                cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
            ),
            max_access_window: attr(
                cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
            ),
        })
    }
}

/// A reserved L2 set-aside plus the ability to pin address windows in
/// it. Dropping it releases the set-aside back to general L2.
pub struct L2Persist {
    ctx: Arc<CudaContext>,
    cap: L2Capability,
    reserved: usize,
}

impl L2Persist {
    /// Reserve `bytes` of L2 for persisting lines (clamped to the
    /// device ceiling). Returns `Err(Unavailable)` when the device
    /// lacks the feature.
    pub fn reserve(ctx: &Arc<CudaContext>, bytes: usize) -> Result<Self, GpuPeerError> {
        let cap = L2Capability::query(ctx)?;
        if !cap.supported() {
            return Err(GpuPeerError::Unavailable("L2 persistence unsupported on this device"));
        }
        let want = bytes.min(cap.max_persisting_l2);
        // SAFETY: context is current (query bound the thread); the
        // limit enum and value are valid.
        let r = unsafe { cu::cuCtxSetLimit(LIMIT_PERSISTING_L2, want) };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(GpuPeerError::Driver(format!("cuCtxSetLimit(persisting L2): {r:?}")));
        }
        let mut got: usize = 0;
        // SAFETY: valid limit enum; pvalue is a live local.
        let _rc: cu::CUresult = unsafe { cu::cuCtxGetLimit(&mut got, LIMIT_PERSISTING_L2) };
        Ok(Self { ctx: Arc::clone(ctx), cap, reserved: got })
    }

    /// The device's L2-persistence ceilings.
    #[inline]
    pub fn capability(&self) -> L2Capability {
        self.cap
    }

    /// Bytes actually set aside (driver may round the request).
    #[inline]
    pub fn reserved(&self) -> usize {
        self.reserved
    }

    /// Mark `[dev_ptr, dev_ptr + num_bytes)` as persisting on `stream`,
    /// so `hit_ratio` (0.0..=1.0) of its lines are preferentially
    /// retained in the set-aside L2. Kernels launched on that stream
    /// afterwards read the window from L2 instead of HBM once it is
    /// primed. `num_bytes` clamps to the access-window ceiling.
    pub fn pin_window(
        &self,
        stream: &cudarc::driver::CudaStream,
        dev_ptr: u64,
        num_bytes: usize,
        hit_ratio: f32,
    ) -> Result<(), GpuPeerError> {
        let n = num_bytes.min(self.cap.max_access_window);
        let window = cu::CUaccessPolicyWindow_st {
            base_ptr: dev_ptr as *mut core::ffi::c_void,
            num_bytes: n,
            hitRatio: hit_ratio.clamp(0.0, 1.0),
            hitProp: cu::CUaccessProperty::CU_ACCESS_PROPERTY_PERSISTING,
            missProp: cu::CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
        };
        // The stream-attribute value is a union; zero it, then set the
        // access-policy-window arm.
        // SAFETY: the union is plain-old-data; zeroing is a valid init.
        let mut value: cu::CUstreamAttrValue = unsafe { core::mem::zeroed() };
        value.accessPolicyWindow = window;
        // SAFETY: valid stream handle; attr id and value type match
        // (CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW <-> accessPolicyWindow).
        let r = unsafe {
            cu::cuStreamSetAttribute(
                stream.cu_stream(),
                cu::CUstreamAttrID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &value,
            )
        };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(GpuPeerError::Driver(format!("cuStreamSetAttribute: {r:?}")));
        }
        Ok(())
    }

    /// Clear the persisting window on `stream` (num_bytes = 0) and
    /// reset any persisting lines back to normal. Call this between
    /// phases so a stale window does not pin the wrong range.
    pub fn clear_window(&self, stream: &cudarc::driver::CudaStream) -> Result<(), GpuPeerError> {
        // SAFETY: the union is plain-old-data; zeroing is a valid init.
        let mut value: cu::CUstreamAttrValue = unsafe { core::mem::zeroed() };
        value.accessPolicyWindow = cu::CUaccessPolicyWindow_st {
            base_ptr: core::ptr::null_mut(),
            num_bytes: 0,
            hitRatio: 0.0,
            hitProp: cu::CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
            missProp: cu::CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
        };
        // SAFETY: as in pin_window.
        let r = unsafe {
            cu::cuStreamSetAttribute(
                stream.cu_stream(),
                cu::CUstreamAttrID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &value,
            )
        };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(GpuPeerError::Driver(format!("cuStreamSetAttribute(clear): {r:?}")));
        }
        self.ctx
            .bind_to_thread()
            .map_err(|e| GpuPeerError::Driver(format!("bind_to_thread: {e:?}")))?;
        // SAFETY: context current; no arguments.
        let _rc: cu::CUresult = unsafe { cu::cuCtxResetPersistingL2Cache() };
        Ok(())
    }
}

impl Drop for L2Persist {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_ok() {
            // Release the set-aside back to general L2. Teardown
            // failure is unactionable.
            // SAFETY: context current; valid limit enum.
            let _rc: cu::CUresult = unsafe { cu::cuCtxSetLimit(LIMIT_PERSISTING_L2, 0) };
            let _rc2: cu::CUresult = unsafe { cu::cuCtxResetPersistingL2Cache() };
        }
    }
}

/// Measured outcome of [`benchmark`]: the same L2-hammer kernel timed
/// with the hot working set pinned in L2 versus left streaming.
#[derive(Debug, Clone, Copy)]
pub struct L2BenchReport {
    /// L2 actually set aside (bytes).
    pub reserved: usize,
    /// Hot working-set size (bytes) - the pinned window.
    pub hot_bytes: usize,
    /// Polluter size (bytes) - the eviction pressure per pass.
    pub pol_bytes: usize,
    /// Best kernel time with the hot set pinned in L2 (microseconds).
    pub persist_us: f64,
    /// Best kernel time with no window, hot set streaming (us).
    pub stream_us: f64,
    /// `stream_us / persist_us`: how much pinning bought on this host.
    pub speedup: f64,
}

/// Measure L2 persistence on the running device with a fair A/B: the
/// identical `flynnel_l2_hammer` kernel over identical data, timed once
/// with the hot buffer pinned in the set-aside L2 and once left to
/// stream. The only variable is the access-policy window. Returns the
/// best-of-`runs` timings (min rejects scheduler noise).
///
/// `hot_bytes` clamps to the device's persisting-L2 ceiling; make
/// `pol_bytes` at least L2-sized so an unpinned hot set is evicted
/// every pass.
pub fn benchmark(
    ctx: &Arc<CudaContext>,
    hot_bytes: usize,
    pol_bytes: usize,
    iters: u32,
    runs: u32,
) -> Result<L2BenchReport, GpuPeerError> {
    use std::time::Instant;

    let l2 = L2Persist::reserve(ctx, hot_bytes)?;
    let hot_bytes = (hot_bytes.min(l2.reserved()).max(4096)) & !3usize;
    let hot_n = (hot_bytes / 4) as u32;
    let pol_n = (pol_bytes / 4) as u32;

    ctx.bind_to_thread()
        .map_err(|e| GpuPeerError::Driver(format!("bind_to_thread: {e:?}")))?;

    // Load the l2_hammer kernel from the embedded PTX.
    let ptx = std::ffi::CString::new(super::PEER_PTX)
        .map_err(|e| GpuPeerError::Driver(format!("PTX CString: {e}")))?;
    let mut module: cu::CUmodule = core::ptr::null_mut();
    // SAFETY: valid NUL-terminated PTX image; context is current.
    let r = unsafe { cu::cuModuleLoadData(&mut module, ptx.as_ptr() as *const core::ffi::c_void) };
    if r != cu::CUresult::CUDA_SUCCESS {
        return Err(GpuPeerError::Driver(format!("cuModuleLoadData: {r:?}")));
    }
    let fname = std::ffi::CString::new("flynnel_l2_hammer").expect("no NUL");
    let mut func: cu::CUfunction = core::ptr::null_mut();
    // SAFETY: module loaded above; name matches a .entry.
    let r = unsafe { cu::cuModuleGetFunction(&mut func, module, fname.as_ptr()) };
    if r != cu::CUresult::CUDA_SUCCESS {
        // SAFETY: module handle valid.
        let _u: cu::CUresult = unsafe { cu::cuModuleUnload(module) };
        return Err(GpuPeerError::Driver(format!("cuModuleGetFunction: {r:?}")));
    }

    // Device buffers.
    let mut hot: cu::CUdeviceptr = 0;
    let mut pol: cu::CUdeviceptr = 0;
    let mut out: cu::CUdeviceptr = 0;
    // SAFETY: sizes are non-zero; pointers are live locals. Primed with
    // 1s so the accumulator is non-zero and reads cannot be elided.
    unsafe {
        let a = cu::cuMemAlloc_v2(&mut hot, hot_bytes);
        let b = cu::cuMemAlloc_v2(&mut pol, pol_bytes);
        let c = cu::cuMemAlloc_v2(&mut out, 8);
        if a != cu::CUresult::CUDA_SUCCESS
            || b != cu::CUresult::CUDA_SUCCESS
            || c != cu::CUresult::CUDA_SUCCESS
        {
            let _u: cu::CUresult = cu::cuModuleUnload(module);
            return Err(GpuPeerError::Driver("cuMemAlloc failed".into()));
        }
        let _s1: cu::CUresult = cu::cuMemsetD32_v2(hot, 1, hot_n as usize);
        let _s2: cu::CUresult = cu::cuMemsetD32_v2(pol, 1, pol_n as usize);
    }

    let stream = ctx.default_stream();
    let s = stream.cu_stream();

    let launch = |func: cu::CUfunction| -> cu::CUresult {
        let mut d_hot = hot;
        let mut d_pol = pol;
        let mut d_out = out;
        let mut hn = hot_n;
        let mut pn = pol_n;
        let mut it = iters;
        let mut params: [*mut core::ffi::c_void; 6] = [
            (&mut d_hot) as *mut _ as *mut core::ffi::c_void,
            (&mut hn) as *mut _ as *mut core::ffi::c_void,
            (&mut d_pol) as *mut _ as *mut core::ffi::c_void,
            (&mut pn) as *mut _ as *mut core::ffi::c_void,
            (&mut it) as *mut _ as *mut core::ffi::c_void,
            (&mut d_out) as *mut _ as *mut core::ffi::c_void,
        ];
        // SAFETY: params match flynnel_l2_hammer(const u32*, u32,
        // const u32*, u32, u32, u64*); grid/block valid; s is the
        // context default stream.
        unsafe {
            cu::cuLaunchKernel(
                func, 256, 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), core::ptr::null_mut(),
            )
        }
    };

    let measure = |pin: bool| -> Result<f64, GpuPeerError> {
        if pin {
            l2.pin_window(&stream, hot, hot_bytes, 1.0)?;
        } else {
            l2.clear_window(&stream)?;
        }
        // Warm up (prime the window / caches).
        let _warm: cu::CUresult = launch(func);
        // SAFETY: valid stream.
        let _sy: cu::CUresult = unsafe { cu::cuStreamSynchronize(s) };
        let mut best = f64::INFINITY;
        for _ in 0..runs.max(1) {
            let t0 = Instant::now();
            let lr = launch(func);
            // SAFETY: valid stream.
            let sy = unsafe { cu::cuStreamSynchronize(s) };
            if lr != cu::CUresult::CUDA_SUCCESS || sy != cu::CUresult::CUDA_SUCCESS {
                return Err(GpuPeerError::Driver(format!("l2 launch: {lr:?} / sync {sy:?}")));
            }
            best = best.min(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok(best)
    };

    let stream_us = measure(false)?;
    let persist_us = measure(true)?;

    let mut host_out: u64 = 0;
    // SAFETY: out is 8 live device bytes; host_out is a live local.
    let _cp: cu::CUresult = unsafe {
        cu::cuMemcpyDtoH_v2((&mut host_out) as *mut u64 as *mut core::ffi::c_void, out, 8)
    };

    // Clear the window and free everything.
    let _w: Result<(), GpuPeerError> = l2.clear_window(&stream);
    // SAFETY: all handles allocated/loaded above, unused hereafter.
    unsafe {
        let _f1: cu::CUresult = cu::cuMemFree_v2(hot);
        let _f2: cu::CUresult = cu::cuMemFree_v2(pol);
        let _f3: cu::CUresult = cu::cuMemFree_v2(out);
        let _u: cu::CUresult = cu::cuModuleUnload(module);
    }

    if host_out == 0 {
        return Err(GpuPeerError::Driver("l2 hammer produced zero (reads elided?)".into()));
    }

    Ok(L2BenchReport {
        reserved: l2.reserved(),
        hot_bytes,
        pol_bytes,
        persist_us,
        stream_us,
        speedup: stream_us / persist_us.max(f64::MIN_POSITIVE),
    })
}

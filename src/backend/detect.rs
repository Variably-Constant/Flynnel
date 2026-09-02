//! Runtime backend detection: probe whether GPU / TPU runtimes are
//! loadable on the host *without* linking the SDK at build time.
//!
//! Every probe is a single `libloading::Library::new` call on the
//! platform-specific shared library name, or a filesystem / env-var
//! check for runtimes that ship as device files or driver shims.
//! Results are cached per process via `OnceLock` because the
//! probes do not change between calls within a single run.
//!
//! ## Why dlopen-probe instead of link-and-probe
//!
//! - **No SDK at build time**: Flynnel builds and tests on hosts
//!   without any GPU toolchain (Windows MSVC, macOS arm64,
//!   ARM Linux SBCs). Linking `libcuda` at build time would
//!   require the CUDA driver SDK on every build host; dlopen-ing
//!   it at probe time requires only the runtime driver on the
//!   *execution* host.
//! - **Graceful degradation**: a binary built with `--features
//!   cuda-reference` on a workstation runs unchanged on a server
//!   without CUDA. The detector returns `false` and the
//!   [`crate::sched::JobPlan::pick_backend`] router falls through
//!   to the CPU backend.
//! - **Multi-version tolerance**: probes try several library names
//!   (`libcuda.so.1`, `libcuda.so`, `libcuda.dylib`) so the same
//!   build works across driver minor versions and across distros.
//!
//! ## Device properties
//!
//! `cuda_sm_count` goes one step past availability and reads a
//! device attribute off the CUDA driver. It answers `None` wherever
//! the probes answer `false`, so the same graceful-degradation
//! contract holds: a caller sizing a launch geometry to the device
//! gets a number on a CUDA host and nothing to size by elsewhere.

use std::sync::OnceLock;

use crate::backend::Backend;

/// True when the NVIDIA CUDA driver is loadable on this host.
/// Probes `nvcuda.dll` (Windows), `libcuda.so.1` / `libcuda.so`
/// (Linux), `libcuda.dylib` (macOS). Cached per process.
pub fn cuda_available() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        probe_lib(&["nvcuda.dll", "libcuda.so.1", "libcuda.so", "libcuda.dylib"])
    })
}

/// True when the AMD ROCm / HIP runtime is loadable on this host.
/// Probes `amdhip64.dll` (Windows) and several `libamdhip64.so*`
/// variants on Linux. Cached per process.
pub fn rocm_available() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        probe_lib(&[
            "amdhip64.dll",
            "libamdhip64.so",
            "libamdhip64.so.6",
            "libamdhip64.so.5",
        ])
    })
}

/// True when the Apple Metal compute runtime is present. macOS-only
/// (any macOS host since 10.11 has Metal).
pub fn metal_available() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    })
}

/// True when a Google TPU or Coral Edge TPU device is visible.
/// Checks `TPU_NAME` env (set by Google Cloud TPU VM images) AND
/// device-file presence (`/dev/accel0` for vfio-bound TPUs;
/// `/dev/apex_0` for Coral Edge TPUs). Cached per process.
pub fn tpu_available() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        if std::env::var("TPU_NAME").is_ok() {
            return true;
        }
        #[cfg(target_os = "linux")]
        {
            let paths = [
                "/dev/accel0",
                "/dev/apex_0",
                "/dev/vfio/vfio",
            ];
            for p in paths {
                if std::path::Path::new(p).exists() {
                    return true;
                }
            }
        }
        false
    })
}

/// True when Apple Neural Engine is available. Always true on
/// macOS arm64 (M-series), false elsewhere. Cached per process.
pub fn ane_available() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        cfg!(all(target_os = "macos", target_arch = "aarch64"))
    })
}

/// True when a WebAssembly runtime is compiled into this build.
/// Unlike the GPU probes, WASM does not require a host runtime
/// library - wasmtime ships as a pure-Rust crate, so availability
/// reduces to "was the `wasm-reference` feature enabled at build
/// time." Hosts without the feature still get `false`.
pub fn wasm_available() -> bool {
    cfg!(feature = "wasm-reference")
}

/// True when the shared-memory worker backend is compiled in. Like
/// the WASM probe, this reduces to a feature-flag check because the
/// dependency (memmap2) ships as a pure-Rust crate that always
/// resolves at runtime. Whether any peer process is actually
/// listening on a given deque is a separate question the caller
/// answers by attempting to attach via
/// `SharedMemoryChaseLevBackend::open`.
pub fn shared_memory_worker_available() -> bool {
    cfg!(feature = "shared-memory-worker-reference")
}

/// The streaming-multiprocessor count of CUDA device `device_ordinal`,
/// or `None` when the host has no loadable CUDA driver, the ordinal
/// names no device, the driver refuses the query, or this build has
/// neither the `cuda-reference` nor the `gpu-peer` feature. A `Some`
/// value is always at least 1.
///
/// This is the device-side width available to a grid: a
/// `gpu_peer::GpuPeerConfig::blocks_per_lane` of `sm_count / lanes`
/// gives each lane a team of blocks that covers the device once,
/// instead of the single SM a `blocks_per_lane` of 1 occupies.
/// Callers clamp the quotient to at least 1 and to whatever team size
/// their user op supports.
///
/// ```
/// use flynnel::backend::detect::cuda_sm_count;
///
/// let lanes = 4u32;
/// let sm = cuda_sm_count(0);
/// let blocks_per_lane = match sm {
///     Some(sm) => (sm / lanes).max(1),
///     None => 1,
/// };
/// println!("device 0 SM count {sm:?}, blocks_per_lane {blocks_per_lane}");
/// assert!(blocks_per_lane >= 1);
/// ```
///
/// The query is a `cuDeviceGetAttribute` read, so it needs no CUDA
/// context and can run before `gpu_peer::GpuPeer::init` to build the
/// config that call consumes. Unlike the availability probes above
/// the result is not cached: it is one driver call, and a caller
/// walking several ordinals wants each one answered.
#[cfg(any(feature = "cuda-reference", feature = "gpu-peer"))]
pub fn cuda_sm_count(device_ordinal: usize) -> Option<u32> {
    use cudarc::driver::sys as cu;

    // Two gates on every `cu` symbol below. cudarc resolves those
    // lazily and panics when it cannot load libcuda, so the driver
    // must be known loadable before the first one is touched.
    // `cuda_available` is the cached probe and answers the common
    // no-driver case without a dlopen; `is_culib_present` then
    // answers it against cudarc's own library-name candidates, which
    // is the list its loader will use.
    if !cuda_available() {
        return None;
    }
    // SAFETY: the call only attempts a `libloading::Library::new` on
    // each candidate name and reports whether one resolved.
    if !unsafe { cu::is_culib_present() } {
        return None;
    }
    // SAFETY: the driver is loadable. cuInit is idempotent and is the
    // required prelude to any other driver call.
    let r = unsafe { cu::cuInit(0) };
    if r != cu::CUresult::CUDA_SUCCESS {
        return None;
    }
    let ordinal = i32::try_from(device_ordinal).ok()?;
    let mut dev: cu::CUdevice = 0;
    // SAFETY: the driver is initialized; `dev` is a live local. An
    // out-of-range ordinal returns CUDA_ERROR_INVALID_DEVICE.
    let r = unsafe { cu::cuDeviceGet(&mut dev, ordinal) };
    if r != cu::CUresult::CUDA_SUCCESS {
        return None;
    }
    let attr = |a: cu::CUdevice_attribute| -> Option<u32> {
        let mut v: i32 = 0;
        // SAFETY: valid device handle from cuDeviceGet.
        let r = unsafe { cu::cuDeviceGetAttribute(&mut v, a, dev) };
        if r == cu::CUresult::CUDA_SUCCESS && v > 0 { Some(v as u32) } else { None }
    };
    attr(cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
}

/// The streaming-multiprocessor count of CUDA device `device_ordinal`.
/// Always `None` in a build without the `cuda-reference` or
/// `gpu-peer` feature, which is what carries the CUDA driver bindings.
#[cfg(not(any(feature = "cuda-reference", feature = "gpu-peer")))]
pub fn cuda_sm_count(_device_ordinal: usize) -> Option<u32> {
    None
}

/// Returns every backend variant detection finds on this host.
/// Always includes [`Backend::Cpu`] (Flynnel always has the CPU
/// backend available). Order: CPU first, then the detected
/// hardware variants in fixed order (Cuda, Rocm, Metal, Tpu, Ane),
/// then the software-runtime variants when their feature flags
/// are on (Wasm under `wasm-reference`, SharedMemoryWorker under
/// `shared-memory-worker-reference`).
pub fn detect_all() -> Vec<Backend> {
    let mut out = vec![Backend::Cpu];
    if cuda_available() {
        out.push(Backend::Cuda { device_id: 0 });
    }
    if rocm_available() {
        out.push(Backend::Rocm { device_id: 0 });
    }
    if metal_available() {
        out.push(Backend::Metal { device_id: 0 });
    }
    if tpu_available() {
        out.push(Backend::Tpu { device_id: 0 });
    }
    if ane_available() {
        out.push(Backend::Ane);
    }
    if wasm_available() {
        out.push(Backend::Wasm { device_id: 0 });
    }
    if shared_memory_worker_available() {
        out.push(Backend::SharedMemoryWorker { backend_id: 0 });
    }
    out
}

/// Try to `dlopen` each candidate library name; return true on the
/// first one that loads. Loaded libraries are immediately dropped
/// (the dlopen handle is taken only long enough to confirm the
/// runtime is resolvable on the dynamic loader's search path).
fn probe_lib(candidates: &[&str]) -> bool {
    for name in candidates {
        // SAFETY: `Library::new` is unsafe because loading a shared
        // library may run initialization code under that library's
        // control. The libraries we probe (libcuda, libamdhip64,
        // libmetal) are first-party platform runtimes whose
        // initializers are part of the host's driver stack. Loading
        // them is the same operation any GPU-using program performs
        // at startup; doing it here is no riskier than that.
        if let Ok(lib) = unsafe { libloading::Library::new(*name) } {
            drop(lib);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_all_always_includes_cpu() {
        let detected = detect_all();
        assert!(detected.contains(&Backend::Cpu));
    }

    #[test]
    fn cuda_probe_returns_a_bool() {
        let a = cuda_available();
        let b = cuda_available();
        assert_eq!(a, b, "cached probe must be stable");
    }

    #[test]
    fn rocm_probe_returns_a_bool() {
        let a = rocm_available();
        let b = rocm_available();
        assert_eq!(a, b);
    }

    #[test]
    fn metal_probe_is_false_off_macos() {
        #[cfg(not(target_os = "macos"))]
        assert!(!metal_available());
    }

    #[test]
    fn ane_probe_matches_target_classifier() {
        let expected = cfg!(all(target_os = "macos", target_arch = "aarch64"));
        assert_eq!(ane_available(), expected);
    }

    #[test]
    fn tpu_probe_returns_a_bool() {
        let a = tpu_available();
        let b = tpu_available();
        assert_eq!(a, b);
    }

    /// Ties the accessor to an oracle that does not share its code
    /// path: device 0 opens as a CUDA context exactly when device 0
    /// exists, so `cuda_sm_count(0)` must be `Some` exactly then. The
    /// test therefore asserts on a CUDA host and on one without.
    #[cfg(any(feature = "cuda-reference", feature = "gpu-peer"))]
    #[test]
    fn sm_count_answers_whenever_device_zero_opens() {
        let device_opens = cuda_available() && cudarc::driver::CudaContext::new(0).is_ok();
        match cuda_sm_count(0) {
            Some(sm) => {
                assert!(device_opens, "SM count {sm} reported where no device 0 opens");
                // No shipping CUDA part has fewer than 1 SM or more
                // than 1024. The ceiling is several times the widest
                // announced so far and catches a garbage read rather
                // than bounding the hardware.
                assert!(
                    (1..=1024).contains(&sm),
                    "SM count {sm} outside the plausible range for a CUDA device"
                );
            }
            None => assert!(!device_opens, "device 0 opens but cuda_sm_count(0) answered None"),
        }
    }

    #[cfg(not(any(feature = "cuda-reference", feature = "gpu-peer")))]
    #[test]
    fn sm_count_is_none_without_the_cuda_features() {
        assert_eq!(cuda_sm_count(0), None);
    }

    #[test]
    fn sm_count_is_none_for_an_ordinal_no_host_has() {
        assert_eq!(cuda_sm_count(4096), None);
    }

    #[test]
    fn sm_count_is_stable_across_calls() {
        assert_eq!(cuda_sm_count(0), cuda_sm_count(0));
    }

    #[test]
    fn probe_lib_returns_false_for_nonexistent_names() {
        assert!(!probe_lib(&[
            "this_library_does_not_exist_anywhere_42.so",
            "nor_does_this_one_pleasestop.dll",
        ]));
    }
}

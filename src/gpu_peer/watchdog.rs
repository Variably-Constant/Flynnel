//! Which GPU watchdog, if any, resets a device whose work runs too long,
//! read from the host rather than assumed.
//!
//! Windows' timeout detection and recovery (TDR) covers devices under the
//! WDDM and MCDM driver models, does not cover TCC devices, and is off
//! entirely at TDR level 0. Its level and delay are registry values with
//! documented defaults. No other platform has an equivalent read here.
//!
//! Reading is separate from deciding: [`detect`] reads the driver model
//! and the TDR settings, and [`decide`] turns what was read into a
//! [`WatchdogState`], so every rule is testable without a device.

use std::ffi::{CStr, CString, c_char, c_void};
use std::fmt;

/// Windows' TDR level when the registry carries no `TdrLevel` value:
/// recover the adapter.
pub const TDR_DEFAULT_LEVEL: u32 = 3;

/// Windows' TDR delay in seconds when the registry carries no `TdrDelay`
/// value.
pub const TDR_DEFAULT_DELAY_S: u32 = 2;

/// A device's driver model as NVML reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverModel {
    /// Windows Display Driver Model, which TDR covers.
    Wddm,
    /// Tesla Compute Cluster, NVML's `WDM`, which TDR does not cover.
    Tcc,
    /// Microsoft Compute Driver Model, which TDR covers.
    Mcdm,
}

/// The watchdog that applies to one device, and what was read to decide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchdogState {
    /// How long one piece of device work may run before the watchdog
    /// resets the device, or `None` when no watchdog applies.
    pub delay_ns: Option<u64>,
    /// What was read and what each read returned, including every read
    /// that failed.
    pub basis: String,
}

impl WatchdogState {
    /// Whether a watchdog applies to the device.
    pub fn applies(&self) -> bool {
        self.delay_ns.is_some()
    }
}

impl fmt::Display for WatchdogState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.basis)
    }
}

/// The TDR level and delay in seconds.
pub type TdrSettings = (u32, u32);

/// The watchdog state for what was read.
///
/// `tdr` is `None` on a platform without TDR. A TDR read that failed
/// takes the documented default delay, and the basis says so: a
/// watchdog that is probably present and treated as absent ends in a
/// device reset, while one treated as present only shortens slices.
/// An unreadable driver model is treated as covered, for the same
/// reason, since only a TCC reading rules TDR out.
pub fn decide(
    model: &Result<DriverModel, String>,
    tdr: Option<&Result<TdrSettings, String>>,
) -> WatchdogState {
    let (model_text, outside_tdr) = match model {
        Ok(DriverModel::Tcc) => (format!("driver model {:?}", DriverModel::Tcc), true),
        Ok(covered) => (format!("driver model {covered:?}"), false),
        Err(err) => (format!("driver model unreadable ({err})"), false),
    };
    let Some(tdr) = tdr else {
        return WatchdogState {
            delay_ns: None,
            basis: format!("{model_text}; no watchdog is known on this platform"),
        };
    };
    if outside_tdr {
        return WatchdogState {
            delay_ns: None,
            basis: format!("{model_text}; TCC devices are outside TDR"),
        };
    }
    match tdr {
        Ok((0, delay_s)) => WatchdogState {
            delay_ns: None,
            basis: format!("{model_text}; TdrLevel 0 disables detection (TdrDelay {delay_s} s)"),
        },
        Ok((level, delay_s)) => WatchdogState {
            delay_ns: Some(u64::from(*delay_s) * 1_000_000_000),
            basis: format!("{model_text}; TdrLevel {level}, TdrDelay {delay_s} s"),
        },
        Err(err) => WatchdogState {
            delay_ns: Some(u64::from(TDR_DEFAULT_DELAY_S) * 1_000_000_000),
            basis: format!(
                "{model_text}; TDR settings unreadable ({err}), so the documented \
                 {TDR_DEFAULT_DELAY_S} s delay is taken"
            ),
        },
    }
}

/// Read the driver model and the TDR settings for CUDA device `ordinal`
/// and decide which watchdog applies.
///
/// Read once per device per process. Both reads describe hardware and a
/// driver configuration that a running process cannot change, and the
/// driver-model read loads NVML, initializes it and shuts it down, which
/// costs tens of milliseconds whatever the span being sized.
pub fn detect(ordinal: usize) -> WatchdogState {
    static CACHE: std::sync::Mutex<Vec<(usize, WatchdogState)>> =
        std::sync::Mutex::new(Vec::new());

    // A poisoned cache still holds readings, and every entry in it is a
    // value some earlier call already returned.
    let mut cache = CACHE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((_, state)) = cache.iter().find(|(known, _)| *known == ordinal) {
        return state.clone();
    }
    let state = read_watchdog(ordinal);
    cache.push((ordinal, state.clone()));
    state
}

/// One reading of the driver model and the TDR settings for `ordinal`.
fn read_watchdog(ordinal: usize) -> WatchdogState {
    let model = pci_bus_id(ordinal).and_then(|bus_id| nvml_driver_model(&bus_id));
    #[cfg(windows)]
    let tdr = Some(tdr_settings());
    #[cfg(not(windows))]
    let tdr: Option<Result<TdrSettings, String>> = None;
    decide(&model, tdr.as_ref())
}

/// The PCI bus id the CUDA driver reports for `ordinal`, which is how the
/// CUDA ordinal is matched to NVML's own enumeration.
fn pci_bus_id(ordinal: usize) -> Result<String, String> {
    use cudarc::driver::sys as cu;

    if !crate::backend::detect::cuda_available() {
        return Err("the CUDA driver is not loadable".to_string());
    }
    // SAFETY: only attempts a library load on cudarc's candidate names
    // and reports whether one resolved.
    if !unsafe { cu::is_culib_present() } {
        return Err("cudarc found no CUDA driver library".to_string());
    }
    // SAFETY: the driver is loadable, and cuInit is idempotent.
    let rc = unsafe { cu::cuInit(0) };
    if rc != cu::CUresult::CUDA_SUCCESS {
        return Err(format!("cuInit returned {rc:?}"));
    }
    let driver_ordinal = i32::try_from(ordinal)
        .map_err(|err| format!("device ordinal {ordinal} does not fit the driver's type ({err})"))?;
    let mut dev: cu::CUdevice = 0;
    // SAFETY: the driver is initialized and `dev` is a live out-parameter.
    let rc = unsafe { cu::cuDeviceGet(&mut dev, driver_ordinal) };
    if rc != cu::CUresult::CUDA_SUCCESS {
        return Err(format!("cuDeviceGet({ordinal}) returned {rc:?}"));
    }
    let mut buf = [0 as c_char; 32];
    // SAFETY: the buffer is 32 bytes and that length is passed, so the
    // driver writes within it.
    let rc = unsafe { cu::cuDeviceGetPCIBusId(buf.as_mut_ptr(), buf.len() as i32, dev) };
    if rc != cu::CUresult::CUDA_SUCCESS {
        return Err(format!("cuDeviceGetPCIBusId({ordinal}) returned {rc:?}"));
    }
    // SAFETY: on success the driver leaves a NUL-terminated string inside
    // the buffer.
    let id = unsafe { CStr::from_ptr(buf.as_ptr()) };
    id.to_str()
        .map(str::to_string)
        .map_err(|err| format!("the PCI bus id is not UTF-8 ({err})"))
}

#[cfg(windows)]
const NVML_LIBRARY: &str = "nvml.dll";
#[cfg(not(windows))]
const NVML_LIBRARY: &str = "libnvidia-ml.so.1";

const NVML_SUCCESS: i32 = 0;
const NVML_DRIVER_WDDM: i32 = 0;
const NVML_DRIVER_WDM: i32 = 1;
const NVML_DRIVER_MCDM: i32 = 2;

type NvmlDevice = *mut c_void;
type NvmlInit = unsafe extern "C" fn() -> i32;
type NvmlByBusId = unsafe extern "C" fn(*const c_char, *mut NvmlDevice) -> i32;
type NvmlDriverModel = unsafe extern "C" fn(NvmlDevice, *mut i32, *mut i32) -> i32;

/// The driver model NVML reports for the device at `bus_id`.
///
/// `nvmlDeviceGetDriverModel_v2` is the entry point that knows MCDM, and
/// the older `nvmlDeviceGetDriverModel` is used when a driver predates it.
fn nvml_driver_model(bus_id: &str) -> Result<DriverModel, String> {
    // SAFETY: loading NVML runs the vendor library's initializers, which
    // every NVML consumer runs.
    let lib = unsafe { libloading::Library::new(NVML_LIBRARY) }
        .map_err(|err| format!("{NVML_LIBRARY} did not load ({err})"))?;
    // SAFETY: each signature follows nvml.h for the named symbol.
    let init: libloading::Symbol<NvmlInit> = unsafe { lib.get(b"nvmlInit_v2") }
        .map_err(|err| format!("nvmlInit_v2 is missing ({err})"))?;
    // SAFETY: as above.
    let shutdown: libloading::Symbol<NvmlInit> = unsafe { lib.get(b"nvmlShutdown") }
        .map_err(|err| format!("nvmlShutdown is missing ({err})"))?;
    // SAFETY: as above.
    let by_bus: libloading::Symbol<NvmlByBusId> =
        unsafe { lib.get(b"nvmlDeviceGetHandleByPciBusId_v2") }
            .map_err(|err| format!("nvmlDeviceGetHandleByPciBusId_v2 is missing ({err})"))?;
    // SAFETY: as above; the two entry points share a signature.
    let model_fn: libloading::Symbol<NvmlDriverModel> =
        match unsafe { lib.get(b"nvmlDeviceGetDriverModel_v2") } {
            Ok(f) => f,
            Err(v2_err) => unsafe { lib.get(b"nvmlDeviceGetDriverModel") }.map_err(|err| {
                format!(
                    "neither nvmlDeviceGetDriverModel_v2 ({v2_err}) nor \
                     nvmlDeviceGetDriverModel ({err}) is exported"
                )
            })?,
        };

    let bus = CString::new(bus_id)
        .map_err(|err| format!("the PCI bus id {bus_id:?} holds a NUL ({err})"))?;
    // SAFETY: NVML is loaded; init takes no arguments.
    let rc = unsafe { init() };
    if rc != NVML_SUCCESS {
        return Err(format!("nvmlInit_v2 returned {rc}"));
    }
    let mut device: NvmlDevice = std::ptr::null_mut();
    // SAFETY: NVML is initialized, `bus` is NUL-terminated and outlives the
    // call, and `device` is a live out-parameter.
    let rc = unsafe { by_bus(bus.as_ptr(), &mut device) };
    let result = if rc != NVML_SUCCESS {
        Err(format!("nvmlDeviceGetHandleByPciBusId_v2({bus_id}) returned {rc}"))
    } else {
        let mut current = -1i32;
        let mut pending = -1i32;
        // SAFETY: `device` is the handle NVML just returned, and both
        // out-parameters are live.
        let rc = unsafe { model_fn(device, &mut current, &mut pending) };
        match (rc, current) {
            (NVML_SUCCESS, NVML_DRIVER_WDDM) => Ok(DriverModel::Wddm),
            (NVML_SUCCESS, NVML_DRIVER_WDM) => Ok(DriverModel::Tcc),
            (NVML_SUCCESS, NVML_DRIVER_MCDM) => Ok(DriverModel::Mcdm),
            (NVML_SUCCESS, other) => Err(format!("NVML reported driver model {other}")),
            (failed, word) => Err(format!(
                "the driver model query returned {failed}, leaving the model word at {word}"
            )),
        }
    };
    // SAFETY: balances the successful init above.
    let rc = unsafe { shutdown() };
    if rc != NVML_SUCCESS {
        eprintln!("flynnel gpu_peer watchdog: nvmlShutdown returned {rc}");
    }
    result
}

/// The TDR level and delay from the registry, each taking its documented
/// default when the value is absent.
#[cfg(windows)]
fn tdr_settings() -> Result<TdrSettings, String> {
    let level = match registry_dword("TdrLevel")? {
        Some(value) => value,
        None => TDR_DEFAULT_LEVEL,
    };
    let delay_s = match registry_dword("TdrDelay")? {
        Some(value) => value,
        None => TDR_DEFAULT_DELAY_S,
    };
    Ok((level, delay_s))
}

/// A DWORD under `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`,
/// or `None` when the value is not present.
#[cfg(windows)]
fn registry_dword(name: &str) -> Result<Option<u32>, String> {
    use std::os::windows::ffi::OsStrExt;

    // The predefined key is the 32-bit value 0x80000002 sign-extended to
    // pointer width, which is how the Windows headers define it.
    const HKEY_LOCAL_MACHINE: isize = 0x8000_0002_u32 as i32 as isize;
    const RRF_RT_REG_DWORD: u32 = 0x0000_0010;
    const ERROR_SUCCESS: i32 = 0;
    const ERROR_FILE_NOT_FOUND: i32 = 2;

    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn RegGetValueW(
            hkey: isize,
            sub_key: *const u16,
            value: *const u16,
            flags: u32,
            kind: *mut u32,
            data: *mut c_void,
            data_len: *mut u32,
        ) -> i32;
    }

    let wide = |s: &str| -> Vec<u16> { std::ffi::OsStr::new(s).encode_wide().chain(Some(0)).collect() };
    let key = wide(r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers");
    let value_name = wide(name);
    let mut data: u32 = 0;
    let mut len: u32 = 4;
    // SAFETY: both names are NUL-terminated UTF-16 that outlive the call,
    // `data` is a four-byte out-parameter whose size `len` states, and
    // RRF_RT_REG_DWORD makes the call refuse a value of any other type.
    let status = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            key.as_ptr(),
            value_name.as_ptr(),
            RRF_RT_REG_DWORD,
            std::ptr::null_mut(),
            (&mut data as *mut u32).cast(),
            &mut len,
        )
    };
    match status {
        ERROR_SUCCESS => Ok(Some(data)),
        ERROR_FILE_NOT_FOUND => Ok(None),
        other => Err(format!("RegGetValueW(GraphicsDrivers\\{name}) returned {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_platform_without_tdr_has_no_watchdog() {
        let state = decide(&Ok(DriverModel::Wddm), None);
        assert!(!state.applies());
        assert!(state.basis.contains("no watchdog is known"), "{}", state.basis);
    }

    #[test]
    fn a_tcc_device_is_outside_tdr_whatever_the_settings() {
        let state = decide(&Ok(DriverModel::Tcc), Some(&Ok((3, 2))));
        assert!(!state.applies());
        assert!(state.basis.contains("TCC"), "{}", state.basis);
    }

    #[test]
    fn tdr_level_zero_disables_the_watchdog() {
        let state = decide(&Ok(DriverModel::Wddm), Some(&Ok((0, 2))));
        assert!(!state.applies());
        assert!(state.basis.contains("TdrLevel 0"), "{}", state.basis);
    }

    #[test]
    fn a_covered_device_takes_the_registry_delay() {
        let wddm = decide(&Ok(DriverModel::Wddm), Some(&Ok((3, 8))));
        assert_eq!(wddm.delay_ns, Some(8_000_000_000));
        let mcdm = decide(&Ok(DriverModel::Mcdm), Some(&Ok((3, 2))));
        assert_eq!(mcdm.delay_ns, Some(2_000_000_000));
    }

    #[test]
    fn unreadable_settings_take_the_documented_delay_and_say_so() {
        let state = decide(&Ok(DriverModel::Wddm), Some(&Err("denied".to_string())));
        assert_eq!(state.delay_ns, Some(u64::from(TDR_DEFAULT_DELAY_S) * 1_000_000_000));
        assert!(state.basis.contains("unreadable (denied)"), "{}", state.basis);
    }

    #[test]
    fn an_unreadable_driver_model_is_treated_as_covered() {
        let state = decide(&Err("no NVML".to_string()), Some(&Ok((3, 2))));
        assert_eq!(state.delay_ns, Some(2_000_000_000));
        assert!(state.basis.contains("driver model unreadable (no NVML)"), "{}", state.basis);
    }
}

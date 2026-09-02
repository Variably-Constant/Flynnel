//! Reference TPU backend driven by a Python-JAX child process.
//! Compiled only when the `tpu-jax-reference` Cargo feature is
//! enabled.
//!
//! ## How it works
//!
//! At construction the backend locates a Python interpreter (tries
//! `python3` first, then `python`), spawns it with the embedded
//! `tpu_jax_bridge.py` script, and exchanges a `ping` handshake.
//! The handshake verifies that:
//!
//! 1. The interpreter starts.
//! 2. `import jax` succeeds (JAX is installed).
//! 3. `jax.devices()` reports at least one device.
//!
//! Any of these failing returns
//! [`crate::backend::BackendError::DeviceUnavailable`], so a binary
//! built with `--features tpu-jax-reference` runs unchanged on hosts
//! without Python / without JAX / without TPU - the routing helper
//! falls back to the CPU backend.
//!
//! The bridge script is `include_str!`-baked into the Rust binary at
//! compile time. At construction it is written to a temp file under
//! the platform temp directory; the path is passed to the child as
//! its script argument. The temp file is cleaned up when the
//! backend drops.
//!
//! ## Wire protocol
//!
//! See `src/backend/tpu_jax_bridge.py` for the protocol contract.
//! Every Rust call (`register_kernel`, `dispatch_kernel`, the
//! `ping` handshake, `shutdown`) serializes a one-line JSON
//! request, writes it to the child's stdin, reads one line from the
//! child's stdout, and parses the JSON response.
//!
//! Concurrency: the bridge is request-response over a single
//! channel; the backend serializes calls through a [`Mutex`] so
//! concurrent `dispatch_kernel` callers cannot interleave traffic.
//! Throughput-wise the bridge is single-flight, which matches
//! JAX's actual TPU launch semantics (per-device).

#![allow(clippy::missing_errors_doc)]

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::backend::{
    Backend, BackendCapabilities, BackendError, DispatchBackend, KernelArg, KernelHandle,
};

/// Embedded Python bridge script. Compile-time `include_str!` so the
/// crate ships as a single artifact (no external file dependency at
/// runtime).
const BRIDGE_PY: &str = include_str!("tpu_jax_bridge.py");

/// Reference TPU backend that drives a Python-JAX child process.
pub struct TpuJaxBackend {
    device_id: u32,
    caps: BackendCapabilities,
    bridge: Mutex<BridgeState>,
}

/// State the bridge wrapper holds. Kept in a Mutex so only one
/// request-response transaction is in flight at a time.
struct BridgeState {
    child: Option<Child>,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    script_path: PathBuf,
    devices: Vec<String>,
}

impl std::fmt::Debug for TpuJaxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TpuJaxBackend")
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl TpuJaxBackend {
    /// Spawn the JAX bridge on the primary TPU device.
    pub fn new() -> Result<Self, BackendError> {
        Self::with_device(0)
    }

    /// Spawn the JAX bridge targeting `device_id` (passed through as
    /// the identity tag on the resulting [`Backend::Tpu`] id only;
    /// JAX itself manages device selection per its own
    /// `jax.devices()` order).
    pub fn with_device(device_id: u32) -> Result<Self, BackendError> {
        let script_path = write_bridge_script()?;
        let (child, stdin, stdout) = spawn_python(&script_path)?;
        let mut state = BridgeState {
            child: Some(child),
            stdin,
            stdout: BufReader::new(stdout),
            script_path,
            devices: Vec::new(),
        };
        let pong = ping_handshake(&mut state)?;
        state.devices = pong.devices;
        let caps = probe_capabilities();
        Ok(Self {
            device_id,
            caps,
            bridge: Mutex::new(state),
        })
    }

    /// Devices the JAX runtime reported during the handshake (e.g.
    /// `["TpuDevice(id=0, ...)"]`). Useful for telemetry.
    pub fn devices(&self) -> Vec<String> {
        self.bridge
            .lock()
            .map(|g| g.devices.clone())
            .unwrap_or_default()
    }
}

impl Drop for TpuJaxBackend {
    fn drop(&mut self) {
        if let Ok(mut state) = self.bridge.lock() {
            // Best-effort polite shutdown; ignore errors because
            // Drop must not panic.
            let req = serde_json::json!({"op": "shutdown"});
            let _ignored = writeln!(state.stdin, "{req}");
            if let Some(mut child) = state.child.take() {
                let _ignored = child.wait();
            }
            // Best-effort cleanup of the temp script.
            let _ignored = std::fs::remove_file(&state.script_path);
        }
    }
}

impl DispatchBackend for TpuJaxBackend {
    fn id(&self) -> Backend {
        Backend::Tpu {
            device_id: self.device_id,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.caps
    }

    fn dispatch_parallel_for(&self, count: u32, work: &(dyn Fn(u32) + Send + Sync)) {
        // The closure body is a Rust closure, not a JAX function;
        // there is no codegen path from Rust source to TPU XLA. To
        // keep parity with the other backends' parallel-for shape
        // we run the closure host-side, fanned out across worker
        // threads. For real TPU compute, consumers use the
        // `dispatch_kernel` handle path with a Python source body.
        if count == 0 {
            return;
        }
        std::thread::scope(|scope| {
            let threads = (count as usize).min(
                std::thread::available_parallelism()
                    .map(std::num::NonZeroUsize::get)
                    .unwrap_or(1),
            );
            let chunks = count.div_ceil(threads as u32);
            for t in 0..threads as u32 {
                let lo = t.saturating_mul(chunks);
                let hi = (lo + chunks).min(count);
                if lo >= hi {
                    continue;
                }
                scope.spawn(move || {
                    for i in lo..hi {
                        work(i);
                    }
                });
            }
        });
    }

    fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
        std::thread::spawn(work);
    }

    fn register_kernel(&self, name: &str, source: &[u8]) -> Result<KernelHandle, BackendError> {
        let source_str = std::str::from_utf8(source).map_err(|e| {
            BackendError::KernelCompile(format!("source must be UTF-8 Python: {e}"))
        })?;
        let req = RegisterRequest {
            op: "register",
            name,
            source: source_str,
        };
        let resp: RegisterResponse = transact(&self.bridge, &req)?;
        if !resp.ok {
            return Err(BackendError::KernelCompile(
                resp.error.unwrap_or_else(|| "register failed".into()),
            ));
        }
        let handle =
            resp.handle.ok_or_else(|| BackendError::KernelCompile("missing handle".into()))?;
        Ok(KernelHandle(handle))
    }

    fn dispatch_kernel(
        &self,
        handle: KernelHandle,
        count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        let json_args = args
            .iter()
            .map(arg_to_json)
            .collect::<Result<Vec<_>, _>>()?;
        let req = DispatchRequest {
            op: "dispatch",
            handle: handle.0,
            count,
            args: json_args,
        };
        let resp: PlainResponse = transact(&self.bridge, &req)?;
        if !resp.ok {
            return Err(BackendError::Launch(
                resp.error.unwrap_or_else(|| "dispatch failed".into()),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct PingRequest<'a> {
    op: &'a str,
}

#[derive(Deserialize)]
struct PingResponse {
    ok: bool,
    #[serde(default)]
    devices: Vec<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    jax_version: Option<String>,
}

#[derive(Serialize)]
struct RegisterRequest<'a> {
    op: &'a str,
    name: &'a str,
    source: &'a str,
}

#[derive(Deserialize)]
struct RegisterResponse {
    ok: bool,
    #[serde(default)]
    handle: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Serialize)]
struct DispatchRequest<'a> {
    op: &'a str,
    handle: u64,
    count: u32,
    args: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct PlainResponse {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_bridge_script() -> Result<PathBuf, BackendError> {
    let mut path = std::env::temp_dir();
    let pid = std::process::id();
    path.push(format!("flynnel_tpu_jax_bridge_{pid}.py"));
    std::fs::write(&path, BRIDGE_PY).map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("write bridge script: {e}"))
    })?;
    Ok(path)
}

fn spawn_python(script: &PathBuf) -> Result<(Child, ChildStdin, ChildStdout), BackendError> {
    for interpreter in ["python3", "python"] {
        let res = Command::new(interpreter)
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();
        if let Ok(mut child) = res {
            let stdin = child.stdin.take().ok_or_else(|| {
                BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
                    .map_io_context("child stdin missing".into())
            })?;
            let stdout = child.stdout.take().ok_or_else(|| {
                BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
                    .map_io_context("child stdout missing".into())
            })?;
            return Ok((child, stdin, stdout));
        }
    }
    Err(BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
        .map_io_context("no python3 / python interpreter on PATH".into()))
}

fn ping_handshake(state: &mut BridgeState) -> Result<PingResponse, BackendError> {
    let req = PingRequest { op: "ping" };
    let req_json = serde_json::to_string(&req).map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("ping serialize: {e}"))
    })?;
    writeln!(state.stdin, "{req_json}").map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("ping stdin write: {e}"))
    })?;
    state.stdin.flush().map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("ping stdin flush: {e}"))
    })?;
    let mut line = String::new();
    state.stdout.read_line(&mut line).map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("ping stdout read: {e}"))
    })?;
    if line.trim().is_empty() {
        return Err(BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context("ping: bridge sent empty response".into()));
    }
    let pong: PingResponse = serde_json::from_str(line.trim()).map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("ping parse `{}`: {e}", line.trim()))
    })?;
    if !pong.ok {
        return Err(BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(
                pong.error.unwrap_or_else(|| "ping reported not-ok".into()),
            ));
    }
    if pong.devices.is_empty() {
        return Err(BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context("ping: jax reported zero devices".into()));
    }
    Ok(pong)
}

fn transact<Req, Resp>(
    bridge: &Mutex<BridgeState>,
    req: &Req,
) -> Result<Resp, BackendError>
where
    Req: Serialize,
    Resp: for<'de> Deserialize<'de>,
{
    let mut guard = bridge.lock().map_err(|_| {
        BackendError::Launch("bridge mutex poisoned".into())
    })?;
    let body = serde_json::to_string(req).map_err(|e| {
        BackendError::Launch(format!("request serialize: {e}"))
    })?;
    writeln!(guard.stdin, "{body}")
        .map_err(|e| BackendError::Launch(format!("stdin write: {e}")))?;
    guard.stdin
        .flush()
        .map_err(|e| BackendError::Launch(format!("stdin flush: {e}")))?;
    let mut line = String::new();
    guard.stdout
        .read_line(&mut line)
        .map_err(|e| BackendError::Launch(format!("stdout read: {e}")))?;
    serde_json::from_str(line.trim())
        .map_err(|e| BackendError::Launch(format!("response parse `{}`: {e}", line.trim())))
}

fn arg_to_json(arg: &KernelArg<'_>) -> Result<serde_json::Value, BackendError> {
    let v = match *arg {
        KernelArg::I32(v) => serde_json::json!({"i32": v}),
        KernelArg::I64(v) => serde_json::json!({"i64": v}),
        KernelArg::U32(v) => serde_json::json!({"u32": v}),
        KernelArg::U64(v) => serde_json::json!({"u64": v}),
        KernelArg::F32(v) => serde_json::json!({"f32": v}),
        KernelArg::F64(v) => serde_json::json!({"f64": v}),
        KernelArg::DevicePtr(p) => serde_json::json!({"device_ptr": p as u64}),
        KernelArg::HostSlice(_) => return Err(BackendError::NotSupported),
    };
    Ok(v)
}

fn probe_capabilities() -> BackendCapabilities {
    BackendCapabilities {
        // TPU "MXU lane" sized at the 128-wide systolic array.
        simt_width: 128,
        // TPU v4 + v5 are good for hundreds of thousands of in-
        // flight tiles; nominal upper bound.
        max_threads_in_flight: 200_000,
        // Python-JAX dispatch round-trip dominated by JSON encode +
        // subprocess pipe latency: ~100us is realistic.
        launch_latency_ns: 100_000,
        // PCIe class on standalone TPU edge; cloud TPU is hosted
        // through high-bandwidth interconnect that's not bottle-
        // necked by host transfer.
        h2d_bw_bytes_per_sec: 25_000_000_000,
    }
}

/// Internal helper trait so the BackendError construction sites
/// above can attach an I/O context message without exposing the
/// detail in the public error variant. Keeps DeviceUnavailable as a
/// single-variant tag while still surfacing the cause via Display.
trait WithIoContext: Sized {
    fn map_io_context(self, msg: String) -> Self;
}

impl WithIoContext for BackendError {
    fn map_io_context(self, msg: String) -> Self {
        match self {
            BackendError::DeviceUnavailable(b) => {
                eprintln!("[flynnel::tpu_jax] {}: {msg}", b.name());
                BackendError::DeviceUnavailable(b)
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_matches_python_jax_availability() {
        let res = TpuJaxBackend::new();
        // No assertion about Ok vs Err: hosts vary. We only assert
        // that construction returns a typed Result, not a panic /
        // process abort.
        match res {
            Ok(b) => {
                assert_eq!(b.id(), Backend::Tpu { device_id: 0 });
                assert_eq!(b.capabilities().simt_width, 128);
                let devices = b.devices();
                assert!(
                    !devices.is_empty(),
                    "successful ping must report at least one device"
                );
            }
            Err(BackendError::DeviceUnavailable(Backend::Tpu { device_id })) => {
                assert_eq!(device_id, 0);
            }
            Err(other) => panic!("unexpected construction error: {other}"),
        }
    }

    #[test]
    fn capabilities_have_expected_shape() {
        let caps = probe_capabilities();
        assert_eq!(caps.simt_width, 128);
        assert_eq!(caps.h2d_bw_bytes_per_sec, 25_000_000_000);
        assert!(caps.launch_latency_ns >= 10_000);
    }
}

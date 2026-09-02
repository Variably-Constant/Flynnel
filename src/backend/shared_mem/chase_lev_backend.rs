//! `DispatchBackend` impl over an MMF-backed Chase-Lev deque + MMF
//! latch arena.
//!
//! Wires three substrate pieces into one dispatch surface:
//!
//! - [`super::chase_lev_mmf::MmfChaseLevDeque`] for the work queue.
//!   The originator owner-pushes a slot with one Release-store on
//!   `bottom`; cross-process peer thieves steal via CAS-on-top.
//!   Owner push pays no contended atomic on the hot path.
//! - [`super::latch_mmf::MmfLatchArena`] for the result transport.
//!   The originator allocates a latch cell, stamps its offset into
//!   the deque slot, then polls the cell's `state` byte until the
//!   peer publishes. No second ring required for results.
//! - [`super::pass_registry`] for handler resolution. Each peer
//!   registers `closure_id -> handler` at startup under the
//!   deterministic [`super::pass_registry::hash_name`] of the same
//!   string; the slot carries `(closure_id, args_inline[..])`.
//!
//! ## Per-call round-trip cost
//!
//! Measured on Zen+ R7 2700 (16 logical cores, 2 CCX x 8 logical via
//! `AmdCpuidCcx`) by `benches/chase_lev_mmf.rs`. Round-trip is
//! dispatch + drain + result-wait per call; payload is the
//! `(I32, I32) -> u32` adder.
//!
//! | Pinning tier              | Round-trip |
//! |---------------------------|------------|
//! | SMT siblings (shared L1d) |    342 ns  |
//! | Intra-CCX (shared L3)     |    424 ns  |
//! | Cross-CCX (cross-die)     |    881 ns  |
//! | Unpinned (OS-scheduled)   |    533 ns  |
//!
//! Substrate-only same-thread push+pop is 16 ns; latch
//! alloc+publish+read same-thread is 13 ns. The tier-to-tier scaling
//! tracks one bottom->top coherence bounce per call: the bounce-
//! latency floor at each tier is what the end-to-end number measures.

#![allow(clippy::missing_errors_doc)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::chase_lev_mmf::{
    self, ARGS_INLINE_BYTES, MmfChaseLevDeque, RemoteJobSlot, Steal,
};
use super::latch_mmf::{ERR, MmfLatchArena, SET, UNSET};
use super::pass_registry::{self, Pass};
use super::wire;

use crate::backend::{
    Backend, BackendCapabilities, BackendError, DispatchBackend, KernelArg, KernelHandle,
};

/// `DispatchBackend` impl over an MMF Chase-Lev deque + MMF latch
/// arena.
pub struct SharedMemoryChaseLevBackend {
    deque: MmfChaseLevDeque,
    latches: MmfLatchArena,
    backend_id: u32,
    deque_path: PathBuf,
    latches_path: PathBuf,
    caps: BackendCapabilities,
    dispatched: AtomicU64,
}

/// Handle to an in-flight dispatch. The originator passes this to
/// [`SharedMemoryChaseLevBackend::wait_handle`] (or
/// [`SharedMemoryChaseLevBackend::poll_handle`]) to retrieve the
/// peer-published result.
#[derive(Debug, Clone, Copy)]
pub struct DispatchHandle {
    /// Byte offset of the latch cell within the latch arena.
    pub latch_offset: u32,
}

impl SharedMemoryChaseLevBackend {
    /// Create the deque + latch arena files at the given paths.
    /// Truncates if they exist. `deque_capacity` and
    /// `latch_capacity` are rounded up to powers of two; minimum 2
    /// each. The current process becomes the deque owner.
    pub fn create(
        backend_id: u32,
        deque_path: impl Into<PathBuf>,
        latches_path: impl Into<PathBuf>,
        deque_capacity: usize,
        latch_capacity: usize,
    ) -> Result<Self, BackendError> {
        let deque_path = deque_path.into();
        let latches_path = latches_path.into();
        let deque = MmfChaseLevDeque::create(&deque_path, deque_capacity)
            .map_err(|e| BackendError::Memory(format!("create chase-lev deque: {e}")))?;
        let latches = MmfLatchArena::create(&latches_path, latch_capacity)
            .map_err(|e| BackendError::Memory(format!("create latch arena: {e}")))?;
        let caps = Self::caps_for(deque.capacity());
        Ok(Self {
            deque,
            latches,
            backend_id,
            deque_path,
            latches_path,
            caps,
            dispatched: AtomicU64::new(0),
        })
    }

    /// Attach to an existing deque + latch arena. The opening
    /// process becomes a peer (drains; never owns the deque-side
    /// push/pop).
    pub fn open(
        backend_id: u32,
        deque_path: impl Into<PathBuf>,
        latches_path: impl Into<PathBuf>,
    ) -> Result<Self, BackendError> {
        let deque_path = deque_path.into();
        let latches_path = latches_path.into();
        let deque = MmfChaseLevDeque::open(&deque_path)
            .map_err(|e| BackendError::Memory(format!("open chase-lev deque: {e}")))?;
        let latches = MmfLatchArena::open(&latches_path)
            .map_err(|e| BackendError::Memory(format!("open latch arena: {e}")))?;
        let caps = Self::caps_for(deque.capacity());
        Ok(Self {
            deque,
            latches,
            backend_id,
            deque_path,
            latches_path,
            caps,
            dispatched: AtomicU64::new(0),
        })
    }

    fn caps_for(capacity: usize) -> BackendCapabilities {
        BackendCapabilities {
            simt_width: 1,
            max_threads_in_flight: capacity.max(1) as u32,
            launch_latency_ns: 150,
            h2d_bw_bytes_per_sec: 0,
        }
    }

    /// Path of the Chase-Lev deque file.
    pub fn deque_path(&self) -> &std::path::Path {
        &self.deque_path
    }

    /// Path of the latch arena file.
    pub fn latches_path(&self) -> &std::path::Path {
        &self.latches_path
    }

    /// Total dispatches issued through this backend since creation.
    pub fn dispatched(&self) -> u64 {
        self.dispatched.load(Ordering::Relaxed)
    }

    /// Owner-side: allocate a latch cell + push a Marshal-shaped
    /// slot onto the Chase-Lev deque. Returns a [`DispatchHandle`]
    /// the caller passes to [`Self::wait_handle`] /
    /// [`Self::poll_handle`].
    pub fn dispatch_marshal(
        &self,
        closure_id: u32,
        args: &[u8],
    ) -> Result<DispatchHandle, BackendError> {
        if args.len() > ARGS_INLINE_BYTES {
            return Err(BackendError::Launch(format!(
                "marshal args length {} exceeds slot capacity {}",
                args.len(),
                ARGS_INLINE_BYTES
            )));
        }
        let latch_offset = self.latches.alloc();
        let slot = RemoteJobSlot::new(closure_id, latch_offset, args).map_err(|e| {
            BackendError::Launch(format!("build slot for closure_id={closure_id}: {e:?}"))
        })?;
        self.deque.push(slot).map_err(|e| {
            BackendError::Launch(format!("chase-lev push failed: {e:?}"))
        })?;
        self.dispatched.fetch_add(1, Ordering::Relaxed);
        Ok(DispatchHandle { latch_offset })
    }

    /// Peer-side: steal a slot, execute its registered handler,
    /// publish the handler's reply (or error) into the slot's latch
    /// cell. Returns:
    /// - `Ok(Some(()))` on a successful steal + execute;
    /// - `Ok(None)` when the deque is currently empty;
    /// - `Err(_)` on decode / lookup / publish failure.
    ///
    /// The `Steal::Retry` case is handled transparently: this method
    /// re-loops until either `Empty` or `Success`.
    pub fn drain_one(&self) -> Result<Option<()>, BackendError> {
        let slot = loop {
            match self.deque.steal() {
                Steal::Success(s) => break s,
                Steal::Empty => return Ok(None),
                Steal::Retry => continue,
            }
        };
        let pass = Pass {
            closure_id: slot.closure_id,
            args: slot.args().to_vec(),
        };
        // Execute via the local pass_registry. The success branch
        // publishes the reply bytes; the error branch publishes a
        // utf-8 diagnostic.
        match pass_registry::execute(&pass) {
            Ok(reply) => self.latches.publish(slot.latch_offset, &reply).map_err(|e| {
                BackendError::Launch(format!("latch publish failed: {e:?}"))
            })?,
            Err(err) => {
                let msg = err.to_string();
                let bytes = msg.as_bytes();
                let truncated_len = bytes.len().min(chase_lev_mmf::SLOT_SIZE);
                self.latches
                    .publish_err(slot.latch_offset, &bytes[..truncated_len])
                    .map_err(|e| {
                        BackendError::Launch(format!("latch publish_err failed: {e:?}"))
                    })?
            }
        };
        Ok(Some(()))
    }

    /// Same as [`Self::drain_one`] but issues a `prefetch_for_steal`
    /// at the END of the call so the NEXT call's steal CAS hits a
    /// warm slot line. Each iteration of a tight drain loop
    /// (`while running { be.drain_one_prefetched(); }`) sees the
    /// slot bytes already in flight by the time it issues the steal,
    /// hiding the cross-CCX slot-line coherence transfer behind the
    /// prior iteration's `pass_registry::execute` + `latch.publish`
    /// work.
    ///
    /// First-iteration behavior: no prior call to warm the line, so
    /// the first call pays the full cold-slot miss. Subsequent calls
    /// see the win zone (~10-20 % improvement at cross-CCX pinning,
    /// measured on Zen+ R7 2700 2026-06-06).
    ///
    /// Use this over [`Self::drain_one`] when the worker thread is
    /// dedicated to draining from a single victim deque and you can
    /// arrange a steady stream of slots so the trailing prefetch has
    /// time to land before the next steal. For request-reply patterns
    /// where the worker only drains once and then waits, the cold
    /// first-call cost dominates and the plain `drain_one` is
    /// preferable.
    pub fn drain_one_prefetched(&self) -> Result<Option<()>, BackendError> {
        let r = self.drain_one()?;
        // Trailing hint: warm the slot the NEXT drain call will
        // claim. The CPU starts a coherence fill for this line now;
        // by the time the caller loops back into drain_one and the
        // steal() does its slot load, the fill has had several
        // hundred cycles to make progress.
        //
        // ONLY issue the prefetch when we just successfully stole.
        // If the prior drain returned None (empty), there is no
        // "next slot" to warm yet - prefetching slot[top] against
        // an empty deque burns a Line Fill Buffer entry on memory
        // the originator has not written, evicting useful lines
        // from the prefetcher's queue. Empirically (cross-CCX
        // Zen+ bench, single-item dispatch loop), unconditional
        // prefetch slowed the round-trip by ~5 %; the conditional
        // restores parity in the empty-deque case and gains in the
        // producer-fast (deque-rarely-empty) case.
        if r.is_some() {
            self.deque.prefetch_for_steal();
        }
        Ok(r)
    }

    /// Owner-side: drain whatever the owner's local deque pops
    /// directly without going through a steal. Used by the dual-
    /// deque worker that runs the owner-side pop loop concurrently
    /// with peer thieves.
    pub fn drain_owner(&self) -> Result<Option<()>, BackendError> {
        let slot = match self.deque.pop() {
            Steal::Success(s) => s,
            Steal::Empty => return Ok(None),
            Steal::Retry => {
                // pop never returns Retry; treat defensively as Empty.
                return Ok(None);
            }
        };
        let pass = Pass {
            closure_id: slot.closure_id,
            args: slot.args().to_vec(),
        };
        match pass_registry::execute(&pass) {
            Ok(reply) => self.latches.publish(slot.latch_offset, &reply).map_err(|e| {
                BackendError::Launch(format!("latch publish failed: {e:?}"))
            })?,
            Err(err) => {
                let msg = err.to_string();
                let bytes = msg.as_bytes();
                let truncated_len = bytes.len().min(chase_lev_mmf::SLOT_SIZE);
                self.latches
                    .publish_err(slot.latch_offset, &bytes[..truncated_len])
                    .map_err(|e| {
                        BackendError::Launch(format!("latch publish_err failed: {e:?}"))
                    })?
            }
        };
        Ok(Some(()))
    }

    /// Originator-side: non-blocking check for handle completion.
    /// Returns `Ok(None)` if the latch is still UNSET; returns
    /// `Ok(Some(Ok(bytes)))` on success or `Ok(Some(Err(msg)))` on
    /// publisher-reported error.
    pub fn poll_handle(
        &self,
        handle: DispatchHandle,
    ) -> Result<Option<Result<Vec<u8>, String>>, BackendError> {
        let is_set = self.latches.is_set(handle.latch_offset).map_err(|e| {
            BackendError::Launch(format!("latch poll: {e:?}"))
        })?;
        if !is_set {
            return Ok(None);
        }
        let mut buf = Vec::new();
        let state = self.latches.read_result(handle.latch_offset, &mut buf).map_err(|e| {
            BackendError::Launch(format!("latch read: {e:?}"))
        })?;
        // Reset the cell so the bump-allocator can reuse it without
        // a wrap-around hazard.
        self.latches.reset(handle.latch_offset).map_err(|e| {
            BackendError::Launch(format!("latch reset: {e:?}"))
        })?;
        match state {
            SET => Ok(Some(Ok(buf))),
            ERR => Ok(Some(Err(String::from_utf8_lossy(&buf).into_owned()))),
            UNSET => Ok(None), // raced with is_set; treat as not-yet-set
            other => Err(BackendError::Launch(format!(
                "latch read returned unexpected state: {other}"
            ))),
        }
    }

    /// Originator-side: blocking wait. Spins (with `spin_loop` hint)
    /// up to `iter_budget` times; if the latch is still unset, yields
    /// the thread and tries again until set. Returns the same shape as
    /// [`Self::poll_handle`].
    pub fn wait_handle(
        &self,
        handle: DispatchHandle,
        iter_budget: u32,
    ) -> Result<Result<Vec<u8>, String>, BackendError> {
        for _ in 0..iter_budget {
            if let Some(r) = self.poll_handle(handle)? {
                return Ok(r);
            }
            std::hint::spin_loop();
        }
        // Past the spin budget: yield until set. No timeout - caller
        // controls outer dead-man timeout if they need one.
        loop {
            if let Some(r) = self.poll_handle(handle)? {
                return Ok(r);
            }
            std::thread::yield_now();
        }
    }
}

impl DispatchBackend for SharedMemoryChaseLevBackend {
    fn id(&self) -> Backend {
        Backend::SharedMemoryWorker {
            backend_id: self.backend_id,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.caps
    }

    fn dispatch_parallel_for(&self, _count: u32, _work: &(dyn Fn(u32) + Send + Sync)) {
        // Closures don't cross processes (function pointers + captured
        // environment are not portable). Callers needing cross-process
        // fan-out attach multiple peer processes to the same deque /
        // arena and rely on Chase-Lev steal for distribution.
    }

    fn dispatch_one(&self, _work: Box<dyn FnOnce() + Send>) {
        panic!(
            "SharedMemoryChaseLevBackend does not support dispatch_one; \
             use register_kernel + dispatch_kernel (Marshal path)"
        );
    }

    fn register_kernel(&self, name: &str, _source: &[u8]) -> Result<KernelHandle, BackendError> {
        // The handle is the deterministic `hash_name(name)` so every
        // peer can resolve it without coordinating numeric ids.
        let id = pass_registry::hash_name(name);
        Ok(KernelHandle(id as u64))
    }

    fn dispatch_kernel(
        &self,
        handle: KernelHandle,
        _count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        let closure_id = handle.0 as u32;
        let args_blob = wire::encode_args(args)?;
        self.dispatch_marshal(closure_id, &args_blob).map(|_handle| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::shared_mem::pass_registry::{hash_name, register, unregister};

    fn temp_paths(label: &str) -> (PathBuf, PathBuf) {
        let mut d = std::env::temp_dir();
        let mut l = std::env::temp_dir();
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        d.push(format!("flynnel_cl_d_{pid}_{nonce}_{label}.bin"));
        l.push(format!("flynnel_cl_l_{pid}_{nonce}_{label}.bin"));
        (d, l)
    }

    #[test]
    fn id_carries_backend_id() {
        let (d, l) = temp_paths("id");
        let be = SharedMemoryChaseLevBackend::create(11, &d, &l, 4, 4).expect("create");
        match be.id() {
            Backend::SharedMemoryWorker { backend_id } => assert_eq!(backend_id, 11),
            other => panic!("wrong id: {other:?}"),
        }
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn open_attaches_to_existing_files() {
        let (d, l) = temp_paths("open");
        let _creator = SharedMemoryChaseLevBackend::create(0, &d, &l, 8, 8).expect("create");
        let attached = SharedMemoryChaseLevBackend::open(0, &d, &l).expect("open");
        assert_eq!(attached.deque.capacity(), 8);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn dispatch_drain_wait_round_trips_in_one_process() {
        let (d, l) = temp_paths("e2e_one_proc");
        let be = SharedMemoryChaseLevBackend::create(0, &d, &l, 4, 4).expect("create");

        let id = hash_name("flynnel.test.chase_lev.adder");
        register(id, |args| {
            let mut a = [0u8; 4];
            let mut b = [0u8; 4];
            // Wire shape: 1-byte tag + 4-byte LE int, twice.
            a.copy_from_slice(&args[1..5]);
            b.copy_from_slice(&args[6..10]);
            let sum = i32::from_le_bytes(a) + i32::from_le_bytes(b);
            Ok(sum.to_le_bytes().to_vec())
        });

        let h = be.register_kernel("flynnel.test.chase_lev.adder", &[]).expect("register");
        assert_eq!(h.0, id as u64);

        be.dispatch_kernel(h, 1, &[KernelArg::I32(3), KernelArg::I32(4)])
            .expect("dispatch");
        assert_eq!(be.dispatched(), 1);

        // Same-process drain: pull the slot and execute the handler.
        be.drain_owner().expect("drain owner").expect("had work");

        unregister(id);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn dispatch_marshal_drain_wait_round_trips() {
        let (d, l) = temp_paths("marshal_round_trip");
        let be = SharedMemoryChaseLevBackend::create(0, &d, &l, 4, 4).expect("create");

        let id = hash_name("flynnel.test.chase_lev.marshal_adder");
        register(id, |args| {
            assert_eq!(args.len(), 8);
            let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
            Ok((a + b).to_le_bytes().to_vec())
        });

        let mut payload = Vec::with_capacity(8);
        payload.extend_from_slice(&13u32.to_le_bytes());
        payload.extend_from_slice(&29u32.to_le_bytes());

        let handle = be.dispatch_marshal(id, &payload).expect("dispatch");
        be.drain_owner().expect("drain").expect("had work");
        let r = be.wait_handle(handle, 1024).expect("wait").expect("ok branch");
        let result = u32::from_le_bytes(r[..4].try_into().unwrap());
        assert_eq!(result, 42);

        unregister(id);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn unknown_closure_id_publishes_err() {
        let (d, l) = temp_paths("unknown_id_err");
        let be = SharedMemoryChaseLevBackend::create(0, &d, &l, 2, 2).expect("create");

        // Dispatch under an unregistered id.
        let handle = be.dispatch_marshal(0xDEAD_BEEF, &[1, 2, 3]).expect("dispatch");
        be.drain_owner().expect("drain").expect("had work");
        let r = be.wait_handle(handle, 1024).expect("wait");
        match r {
            Err(msg) => assert!(msg.contains("no handler"), "got: {msg}"),
            Ok(_) => panic!("expected Err branch for unknown closure_id"),
        }

        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn drain_one_returns_none_when_empty() {
        let (d, l) = temp_paths("drain_empty");
        let be = SharedMemoryChaseLevBackend::create(0, &d, &l, 4, 4).expect("create");
        assert!(be.drain_one().expect("drain").is_none());
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn drain_one_prefetched_round_trips_same_as_drain_one() {
        let (d, l) = temp_paths("prefetched_round_trip");
        let be = SharedMemoryChaseLevBackend::create(0, &d, &l, 4, 4).expect("create");

        let id = hash_name("flynnel.test.chase_lev.prefetched_adder");
        register(id, |args| {
            let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
            Ok((a + b).to_le_bytes().to_vec())
        });

        let mut payload = Vec::with_capacity(8);
        payload.extend_from_slice(&7u32.to_le_bytes());
        payload.extend_from_slice(&35u32.to_le_bytes());

        let handle = be.dispatch_marshal(id, &payload).expect("dispatch");
        // First call pays the cold-slot miss; trailing prefetch
        // warms the NEXT slot (which is empty here, but the call
        // itself MUST succeed and produce the same Some(()) result
        // shape as plain drain_one.
        let r = be.drain_one_prefetched().expect("drain prefetched");
        assert!(r.is_some(), "drain_one_prefetched returned None on populated deque");

        let result = be
            .wait_handle(handle, 1024)
            .expect("wait")
            .expect("ok branch");
        let v = u32::from_le_bytes(result[..4].try_into().unwrap());
        assert_eq!(v, 42);

        // Empty drain also returns None and does not fault on the
        // trailing prefetch.
        let r = be.drain_one_prefetched().expect("drain empty prefetched");
        assert!(r.is_none());

        unregister(id);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn dispatch_one_panics_with_clear_message() {
        let (d, l) = temp_paths("dispatch_one_panic");
        let be = SharedMemoryChaseLevBackend::create(0, &d, &l, 4, 4).expect("create");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            be.dispatch_one(Box::new(|| ()));
        }));
        assert!(r.is_err(), "dispatch_one must panic");
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn payload_too_large_is_rejected() {
        let (d, l) = temp_paths("oversize");
        let be = SharedMemoryChaseLevBackend::create(0, &d, &l, 2, 2).expect("create");
        let big = vec![0u8; ARGS_INLINE_BYTES + 1];
        let err = be.dispatch_marshal(0, &big).expect_err("expected oversize");
        match err {
            BackendError::Launch(msg) => assert!(msg.contains("exceeds slot capacity")),
            other => panic!("expected Launch, got {other:?}"),
        }
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }
}

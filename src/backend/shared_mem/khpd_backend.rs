//! `DispatchBackend` impl over the [`super::khpd::KhpdDeque`] +
//! [`super::latch_mmf::MmfLatchArena`].
//!
//! Companion to [`super::chase_lev_backend::SharedMemoryChaseLevBackend`]
//! and [`super::loh_backend::SharedMemoryLohBackend`] that targets
//! the same producer-fast batched-dispatch workload as LOH but with
//! a different amortization mechanism: K items per cache-line
//! publication, one CAS-on-head per K items, instead of N items
//! migrating through N independent ring slots.

#![allow(clippy::missing_errors_doc)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::chase_lev_backend::DispatchHandle;
use super::khpd::{KHPD_ARGS_INLINE_BYTES, KhpdDeque, LINE_ITEMS, LineItem, Steal};
use super::latch_mmf::{ERR, MmfLatchArena, SET, UNSET};
use super::pass_registry::{self, Pass};

use crate::backend::{Backend, BackendCapabilities, BackendError, DispatchBackend, KernelHandle};

/// `DispatchBackend` impl over the KHPD deque + MMF latch arena.
pub struct SharedMemoryKhpdBackend {
    deque: KhpdDeque,
    latches: MmfLatchArena,
    backend_id: u32,
    deque_path: PathBuf,
    latches_path: PathBuf,
    caps: BackendCapabilities,
    dispatched: AtomicU64,
}

impl SharedMemoryKhpdBackend {
    /// Create the deque + latch arena files.
    pub fn create(
        backend_id: u32,
        deque_path: impl Into<PathBuf>,
        latches_path: impl Into<PathBuf>,
        deque_capacity: usize,
        latch_capacity: usize,
    ) -> Result<Self, BackendError> {
        let deque_path = deque_path.into();
        let latches_path = latches_path.into();
        let deque = KhpdDeque::create(&deque_path, deque_capacity)
            .map_err(|e| BackendError::Memory(format!("create KHPD deque: {e}")))?;
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

    /// Attach to an existing KHPD deque + latch arena.
    pub fn open(
        backend_id: u32,
        deque_path: impl Into<PathBuf>,
        latches_path: impl Into<PathBuf>,
    ) -> Result<Self, BackendError> {
        let deque_path = deque_path.into();
        let latches_path = latches_path.into();
        let deque = KhpdDeque::open(&deque_path)
            .map_err(|e| BackendError::Memory(format!("open KHPD deque: {e}")))?;
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
            max_threads_in_flight: (capacity * LINE_ITEMS).max(1) as u32,
            launch_latency_ns: 200,
            h2d_bw_bytes_per_sec: 0,
        }
    }

    /// Path of the KHPD deque file.
    pub fn deque_path(&self) -> &std::path::Path {
        &self.deque_path
    }

    /// Path of the latch arena file.
    pub fn latches_path(&self) -> &std::path::Path {
        &self.latches_path
    }

    /// Total items dispatched.
    pub fn dispatched(&self) -> u64 {
        self.dispatched.load(Ordering::Relaxed)
    }

    /// Owner-side: stage one item + auto-flush at LINE_ITEMS.
    /// Returns a [`DispatchHandle`] the caller waits on.
    pub fn dispatch_marshal(
        &self,
        closure_id: u32,
        args: &[u8],
    ) -> Result<DispatchHandle, BackendError> {
        if args.len() > KHPD_ARGS_INLINE_BYTES {
            return Err(BackendError::Launch(format!(
                "KHPD args length {} exceeds slot capacity {}",
                args.len(),
                KHPD_ARGS_INLINE_BYTES
            )));
        }
        let latch_offset = self.latches.alloc();
        let item = LineItem::new(closure_id, latch_offset, args)
            .map_err(|e| BackendError::Launch(format!("KHPD build item: {e:?}")))?;
        let pending_count = self
            .deque
            .stage(item)
            .map_err(|e| BackendError::Launch(format!("KHPD stage: {e:?}")))?;
        if pending_count >= LINE_ITEMS {
            self.deque
                .publish()
                .map_err(|e| BackendError::Launch(format!("KHPD publish: {e:?}")))?;
        }
        self.dispatched.fetch_add(1, Ordering::Relaxed);
        Ok(DispatchHandle { latch_offset })
    }

    /// Owner-side: stage N items, then publish in one call. This is
    /// the producer-fast entry point.
    pub fn dispatch_marshal_batch(
        &self,
        items: &[(u32, &[u8])],
    ) -> Result<Vec<DispatchHandle>, BackendError> {
        let mut handles = Vec::with_capacity(items.len());
        for (closure_id, args) in items {
            if args.len() > KHPD_ARGS_INLINE_BYTES {
                return Err(BackendError::Launch(format!(
                    "KHPD args length {} exceeds slot capacity {}",
                    args.len(),
                    KHPD_ARGS_INLINE_BYTES
                )));
            }
            let latch_offset = self.latches.alloc();
            let item = LineItem::new(*closure_id, latch_offset, args)
                .map_err(|e| BackendError::Launch(format!("KHPD build item: {e:?}")))?;
            self.deque
                .stage(item)
                .map_err(|e| BackendError::Launch(format!("KHPD stage: {e:?}")))?;
            handles.push(DispatchHandle { latch_offset });
        }
        // One publish call drains the pending buffer into multiple
        // lines (ceil(N / LINE_ITEMS)) with one tail.fetch_add each.
        self.deque
            .publish()
            .map_err(|e| BackendError::Launch(format!("KHPD publish: {e:?}")))?;
        self.dispatched
            .fetch_add(items.len() as u64, Ordering::Relaxed);
        Ok(handles)
    }

    /// Owner-side: explicit flush of any pending items.
    pub fn flush(&self) -> Result<usize, BackendError> {
        self.deque
            .publish()
            .map_err(|e| BackendError::Launch(format!("KHPD flush: {e:?}")))
    }

    /// Peer-side: steal one publication line and execute every item
    /// in it via the local pass_registry. Returns:
    /// - `Ok(Some(n))` on a successful claim of `n` items
    /// - `Ok(None)` when ring empty
    pub fn drain_one_line(&self) -> Result<Option<usize>, BackendError> {
        let result = loop {
            match self.deque.steal_line() {
                Steal::Success(r) => break r,
                Steal::Empty => return Ok(None),
                Steal::Retry => continue,
            }
        };
        for i in 0..result.n_items {
            let item = result.items[i];
            let pass = Pass {
                closure_id: item.closure_id,
                args: item.args_inline.to_vec(),
            };
            match pass_registry::execute(&pass) {
                Ok(reply) => self
                    .latches
                    .publish(item.latch_offset, &reply)
                    .map_err(|e| {
                        BackendError::Launch(format!("KHPD latch publish: {e:?}"))
                    })?,
                Err(err) => {
                    let msg = err.to_string();
                    let bytes = msg.as_bytes();
                    let n = bytes.len().min(64);
                    self.latches
                        .publish_err(item.latch_offset, &bytes[..n])
                        .map_err(|e| {
                            BackendError::Launch(format!("KHPD latch publish_err: {e:?}"))
                        })?;
                }
            }
        }
        Ok(Some(result.n_items))
    }

    /// Originator-side non-blocking poll on a single dispatch handle.
    pub fn poll_handle(
        &self,
        handle: DispatchHandle,
    ) -> Result<Option<Result<Vec<u8>, String>>, BackendError> {
        let is_set = self
            .latches
            .is_set(handle.latch_offset)
            .map_err(|e| BackendError::Launch(format!("latch poll: {e:?}")))?;
        if !is_set {
            return Ok(None);
        }
        let mut buf = Vec::new();
        let state = self
            .latches
            .read_result(handle.latch_offset, &mut buf)
            .map_err(|e| BackendError::Launch(format!("latch read: {e:?}")))?;
        self.latches
            .reset(handle.latch_offset)
            .map_err(|e| BackendError::Launch(format!("latch reset: {e:?}")))?;
        match state {
            SET => Ok(Some(Ok(buf))),
            ERR => Ok(Some(Err(String::from_utf8_lossy(&buf).into_owned()))),
            UNSET => Ok(None),
            other => Err(BackendError::Launch(format!(
                "latch read returned unexpected state: {other}"
            ))),
        }
    }

    /// Originator-side blocking wait.
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
        loop {
            if let Some(r) = self.poll_handle(handle)? {
                return Ok(r);
            }
            std::thread::yield_now();
        }
    }
}

impl DispatchBackend for SharedMemoryKhpdBackend {
    fn id(&self) -> Backend {
        Backend::SharedMemoryWorker {
            backend_id: self.backend_id,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.caps
    }

    fn dispatch_parallel_for(&self, _count: u32, _work: &(dyn Fn(u32) + Send + Sync)) {}

    fn dispatch_one(&self, _work: Box<dyn FnOnce() + Send>) {
        panic!(
            "SharedMemoryKhpdBackend does not support dispatch_one; \
             use dispatch_marshal / dispatch_marshal_batch"
        );
    }

    fn register_kernel(&self, name: &str, _source: &[u8]) -> Result<KernelHandle, BackendError> {
        let id = pass_registry::hash_name(name);
        Ok(KernelHandle(id as u64))
    }

    fn dispatch_kernel(
        &self,
        _handle: KernelHandle,
        _count: u32,
        _args: &[crate::backend::KernelArg<'_>],
    ) -> Result<(), BackendError> {
        // KHPD's slot has 8-byte inline args; the generic
        // `KernelArg` wire format needs at least 10 bytes for the
        // adder case. Callers route through `dispatch_marshal` /
        // `dispatch_marshal_batch` with raw payloads instead.
        Err(BackendError::NotSupported)
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
        d.push(format!("flynnel_khpd_be_d_{pid}_{nonce}_{label}.bin"));
        l.push(format!("flynnel_khpd_be_l_{pid}_{nonce}_{label}.bin"));
        (d, l)
    }

    #[test]
    fn dispatch_marshal_batch_round_trips() {
        let (d, l) = temp_paths("batch_round_trip");
        let be = SharedMemoryKhpdBackend::create(0, &d, &l, 16, 64).expect("create");

        let id = hash_name("flynnel.test.khpd.adder");
        register(id, |args| {
            let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
            Ok((a + b).to_le_bytes().to_vec())
        });

        let mut payloads = Vec::with_capacity(6);
        for i in 0..6u32 {
            let mut p = [0u8; 8];
            p[..4].copy_from_slice(&i.to_le_bytes());
            p[4..].copy_from_slice(&(i * 2).to_le_bytes());
            payloads.push(p);
        }
        let items: Vec<(u32, &[u8])> = payloads.iter().map(|p| (id, p.as_slice())).collect();
        let handles = be.dispatch_marshal_batch(&items).expect("dispatch");
        assert_eq!(handles.len(), 6);

        // Drain (2 publication lines for 6 items).
        let mut drained = 0;
        while drained < 6 {
            match be.drain_one_line().expect("drain") {
                Some(n) => drained += n,
                None => std::hint::spin_loop(),
            }
        }
        for (i, h) in handles.iter().enumerate() {
            let r = be.wait_handle(*h, 1024).expect("wait").expect("ok");
            let v = u32::from_le_bytes(r[..4].try_into().unwrap());
            assert_eq!(v, i as u32 + i as u32 * 2);
        }

        unregister(id);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn dispatch_marshal_auto_flushes_at_line_items() {
        let (d, l) = temp_paths("auto_flush");
        let be = SharedMemoryKhpdBackend::create(0, &d, &l, 8, 32).expect("create");

        let id = hash_name("flynnel.test.khpd.add2");
        register(id, |args| {
            let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
            Ok((a + b).to_le_bytes().to_vec())
        });

        // Stage LINE_ITEMS items; the LINE_ITEMS-th call auto-flushes.
        let mut handles = Vec::new();
        for i in 0..LINE_ITEMS as u32 {
            let mut p = [0u8; 8];
            p[..4].copy_from_slice(&i.to_le_bytes());
            p[4..].copy_from_slice(&(10 + i).to_le_bytes());
            handles.push(be.dispatch_marshal(id, &p).expect("dispatch"));
        }
        // Drain the published line.
        match be.drain_one_line().expect("drain") {
            Some(n) => assert_eq!(n, LINE_ITEMS),
            None => panic!("expected drained line"),
        }
        for (i, h) in handles.iter().enumerate() {
            let r = be.wait_handle(*h, 1024).expect("wait").expect("ok");
            let v = u32::from_le_bytes(r[..4].try_into().unwrap());
            assert_eq!(v, (i as u32) + (10 + i as u32));
        }
        unregister(id);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn dispatch_one_panics() {
        let (d, l) = temp_paths("panic");
        let be = SharedMemoryKhpdBackend::create(0, &d, &l, 2, 8).expect("create");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            be.dispatch_one(Box::new(|| ()));
        }));
        assert!(r.is_err());
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn oversize_args_rejected() {
        let (d, l) = temp_paths("oversize");
        let be = SharedMemoryKhpdBackend::create(0, &d, &l, 2, 8).expect("create");
        let big = vec![0u8; KHPD_ARGS_INLINE_BYTES + 1];
        let err = be.dispatch_marshal(0, &big).expect_err("oversize");
        match err {
            BackendError::Launch(msg) => assert!(msg.contains("exceeds slot capacity")),
            other => panic!("expected Launch, got {other:?}"),
        }
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }
}

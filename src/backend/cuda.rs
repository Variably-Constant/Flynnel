//! Reference CUDA backend built on the cudarc crate's dynamic-
//! loading feature. Compiled only when the `cuda-reference`
//! Cargo feature is enabled.
//!
//! ## What this provides
//!
//! - [`CudaBackend::new`] / [`CudaBackend::with_device`]:
//!   constructors that initialize the CUDA driver via cudarc's
//!   dynamic-loading path, claim a device, and create a default
//!   stream.
//! - A full [`crate::backend::DispatchBackend`] implementation
//!   that:
//!   - implements `dispatch_parallel_for` as a synchronous host-
//!     side loop fan-out across worker OS threads (the closure
//!     body is CPU-runnable, not a GPU kernel);
//!   - implements `dispatch_one` by sending the work item to a
//!     single persistent worker thread the constructor spawns
//!     (routed via a flynnel `NotifyHub` MPMC ring); `Drop` shuts
//!     the hub down and joins the worker;
//!   - implements `register_kernel` as [`CudaBackend::register_ptx`]
//!     over the UTF-8 text of the source bytes;
//!   - implements `dispatch_kernel` by launching the registered
//!     function through cudarc's `launch_builder` with the supplied
//!     `count` work-items and [`KernelArg`] list. Argument count and
//!     type correctness is a contract with the kernel's author, which
//!     cudarc's launch cannot verify.
//! - Device memory: a [`DeviceBuffer`] is made by
//!   [`CudaBackend::alloc_zeroed`] or [`CudaBackend::upload`], written
//!   by [`CudaBackend::copy_in`], read by [`CudaBackend::copy_out`] or
//!   [`CudaBackend::copy_out_range`], handed to a kernel by
//!   [`DeviceBuffer::arg`], and freed when dropped.
//! - Mapped memory: a [`MappedBuffer`] is page-locked host memory the
//!   device reads and writes directly, made by [`CudaBackend::map_host`],
//!   written and read through its host slices with no copy, handed to a
//!   kernel by [`MappedBuffer::arg`], and freed when dropped.
//! - [`CudaBackend::device_name`] and [`CudaBackend::mem_info`], so a
//!   caller can refuse or cut a block before allocating it.
//! - [`CudaBackend::device_info`] and [`DeviceInfo::of_ordinal`], the
//!   device's capability, size and limits, and
//!   [`CudaBackend::device_count`]; neither of the last two makes a
//!   context.
//!
//! ## Nothing beyond the driver
//!
//! A kernel loads from PTX text through the driver's own JIT, and every
//! allocation, copy and launch is a driver call. Nothing here calls NVRTC
//! or needs the CUDA toolkit, so PTX written once for a target the driver
//! supports loads on any machine with an NVIDIA driver and no other CUDA
//! software.
//!
//! ## Order
//!
//! Copies and launches on one stream run in the order they were called;
//! the copies, and [`DispatchBackend::dispatch_kernel`], use the default
//! stream. [`CudaBackend::copy_out`] and [`CudaBackend::copy_out_range`]
//! return only once the host slice holds the data. A [`DeviceBuffer`] is
//! freed when dropped, and the free waits for every copy and launch that
//! used it, on whichever stream. A launch through
//! [`CudaBackend::dispatch_kernel_on_stream`] on another stream is ordered
//! against the default stream's copies only by the caller: synchronize or
//! join that stream before reading back what the kernel wrote.
//!
//! A [`MappedBuffer`]'s host slices wait for every launch that passed the
//! buffer, on whichever stream, and so does its free.
//!
//! ## When to use
//!
//! For consumers that have pre-compiled PTX they want to launch
//! through a uniform Flynnel surface. Consumers that need richer
//! CUDA semantics (per-launch configuration, CUDA graphs) typically
//! ship their own
//! [`crate::backend::DispatchBackend`] impl backed by their
//! preferred CUDA wrapper.

#![allow(clippy::missing_errors_doc)]

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::thread::JoinHandle;

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError,
    LaunchArgs, LaunchConfig, PushKernelArg, result, sys,
};

use crate::sched::notify_ring::{NotifyHub, NotifySendResult, NotifySender};

use crate::backend::{
    Backend, BackendCapabilities, BackendError, DeviceArg, DispatchBackend, KernelArg, KernelHandle,
};

/// Boxed closure shape the persistent worker thread consumes from
/// the dispatch_one channel.
type WorkItem = Box<dyn FnOnce() + Send + 'static>;

mod sealed {
    /// What a [`super::DeviceElement`] needs from cudarc, kept off the
    /// public trait so no cudarc name reaches a caller's signature.
    pub trait Sealed:
        cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy + Send + Sync + 'static
    {
    }

    impl Sealed for u8 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for i64 {}
    impl Sealed for u64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// An element type a [`DeviceBuffer`] holds: `u8`, `i32`, `u32`, `i64`,
/// `u64`, `f32` or `f64`. Sealed, so the set is exactly the one a launch
/// knows how to pass as a [`KernelArg::Buffer`].
pub trait DeviceElement: sealed::Sealed {}

impl DeviceElement for u8 {}
impl DeviceElement for i32 {}
impl DeviceElement for u32 {}
impl DeviceElement for i64 {}
impl DeviceElement for u64 {}
impl DeviceElement for f32 {}
impl DeviceElement for f64 {}

/// Device memory holding a run of `T` on one CUDA device, made by
/// [`CudaBackend::alloc_zeroed`] or [`CudaBackend::upload`].
///
/// Freed when dropped. The free waits for every copy and launch that
/// used the buffer, on whichever of the backend's streams, so dropping it
/// while a kernel that reads it is still queued is safe. Only the backend
/// that allocated it accepts it for a copy or a launch.
pub struct DeviceBuffer<T: DeviceElement> {
    slice: CudaSlice<T>,
}

impl<T: DeviceElement> DeviceBuffer<T> {
    /// Elements the buffer holds.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Whether it holds none, which an allocation never makes.
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// The buffer as a kernel argument. It reaches the kernel as a 64-bit
    /// device pointer to its first element.
    pub fn arg(&self) -> KernelArg<'_> {
        KernelArg::Buffer(self)
    }
}

impl<T: DeviceElement> DeviceArg for DeviceBuffer<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T: DeviceElement> std::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("element", &std::any::type_name::<T>())
            .field("len", &self.slice.len())
            .field("device", &self.slice.ordinal())
            .finish()
    }
}

/// Page-locked host memory holding a run of `T` that one CUDA device
/// reads and writes directly, made by [`CudaBackend::map_host`].
///
/// A kernel reaches it over the bus through [`Self::arg`], so what the
/// host writes into [`Self::as_mut_slice`] needs no copy in and what a
/// kernel writes is read from [`Self::as_slice`] with no copy out. Every
/// device access crosses the bus, so a buffer a kernel reads many times,
/// or updates atomically, belongs in a [`DeviceBuffer`].
///
/// A host slice first waits for every launch that passed the buffer:
/// for the last of them when all of them since the previous host slice
/// ran on one stream, and for all work on the device when more than one
/// stream did. Launches on two streams are not ordered against each
/// other. Freed when dropped, after the same wait. Only the backend that
/// made it accepts it for a launch.
pub struct MappedBuffer<T: DeviceElement> {
    memory: MappedMemory,
    len: usize,
    element: PhantomData<T>,
}

impl<T: DeviceElement> MappedBuffer<T> {
    /// Elements the buffer holds.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds none, which an allocation never makes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer as a kernel argument. It reaches the kernel as the
    /// 64-bit device address of its first element.
    pub fn arg(&self) -> KernelArg<'_> {
        KernelArg::Buffer(&self.memory)
    }

    /// The elements, once every launch that passed the buffer has
    /// finished.
    pub fn as_slice(&mut self) -> Result<&[T], BackendError> {
        self.memory.settle()?;
        // SAFETY: the allocation holds `len` elements of `T` at a page
        // boundary and every bit pattern is a `T`; `settle` has waited
        // for every launch that passed the buffer, and `&mut self` keeps
        // any other from being issued while the slice lives.
        Ok(unsafe { std::slice::from_raw_parts(self.memory.host.as_ptr().cast::<T>(), self.len) })
    }

    /// The elements to write, once every launch that passed the buffer
    /// has finished. A launch issued after the slice is dropped reads what
    /// was written.
    pub fn as_mut_slice(&mut self) -> Result<&mut [T], BackendError> {
        self.memory.settle()?;
        // SAFETY: as for `as_slice`, and `&mut self` makes this the only
        // slice of the allocation.
        Ok(unsafe {
            std::slice::from_raw_parts_mut(self.memory.host.as_ptr().cast::<T>(), self.len)
        })
    }
}

impl<T: DeviceElement> std::fmt::Debug for MappedBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedBuffer")
            .field("element", &std::any::type_name::<T>())
            .field("len", &self.len)
            .field("device", &self.memory.event.context().ordinal())
            .finish()
    }
}

/// `CU_STREAM_PER_THREAD`: one handle that names a different stream on
/// each thread that launches on it.
const PER_THREAD_STREAM: u64 = 0x2;

/// The allocation behind a [`MappedBuffer`], untyped, and what a launch
/// takes from it: the device address it passes and the event it records.
struct MappedMemory {
    /// Page-locked host memory from `cuMemHostAlloc`, `bytes` long.
    host: NonNull<u8>,
    /// The same memory's address in the device's address space.
    device: u64,
    bytes: usize,
    /// Recorded on a launch's stream after every launch that passes the
    /// memory. Holds the context the memory was allocated in.
    event: CudaEvent,
    /// The stream every launch since the last settle ran on, as its
    /// handle plus one, so that zero means none has; the legacy default
    /// stream's handle is zero.
    stream: AtomicU64,
    /// Set when a launch since the last settle ran on a second stream or
    /// on the per-thread stream, or when recording the event failed.
    mixed: AtomicBool,
}

// SAFETY: `host` is page-locked memory this value owns and frees once. A
// shared reference reads `device` and `bytes`, records the event, which
// cudarc documents as thread safe, and updates two atomics; the host
// slices of the memory need `&mut` to the buffer that owns it.
unsafe impl Send for MappedMemory {}
// SAFETY: as for `Send`.
unsafe impl Sync for MappedMemory {}

impl MappedMemory {
    /// Notes a launch on `stream` that passed this memory: the stream
    /// joins the ones the next settle waits for, and the event moves past
    /// the launch. A record the driver refuses marks the memory mixed, so
    /// the next settle waits for the whole device, which covers the
    /// launch the event missed.
    fn note_launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        // Relaxed throughout: a settle holds `&mut`, which it can have only
        // once every launch holding `&self` has returned.
        let handle = stream.cu_stream() as usize as u64;
        if handle == PER_THREAD_STREAM {
            self.mixed.store(true, Ordering::Relaxed);
        } else {
            let tag = handle.wrapping_add(1);
            if let Err(held) =
                self.stream
                    .compare_exchange(0, tag, Ordering::Relaxed, Ordering::Relaxed)
                && held != tag
            {
                self.mixed.store(true, Ordering::Relaxed);
            }
        }
        let recorded = self.event.record(stream);
        if recorded.is_err() {
            self.mixed.store(true, Ordering::Relaxed);
        }
        recorded
    }

    /// Waits until no launch that passed this memory is in flight: for
    /// the event when every launch since the last settle ran on one
    /// stream, since each recorded it after itself and a stream runs in
    /// order, and for all work on the device otherwise. The next launch
    /// starts the account again.
    fn settle(&mut self) -> Result<(), BackendError> {
        if *self.mixed.get_mut() {
            self.event.context().synchronize().map_err(|e| {
                memory_error(
                    "waiting for the device, which more than one stream used, before a \
                     mapped buffer's host slice"
                        .to_string(),
                    e,
                )
            })?;
        } else {
            self.event.synchronize().map_err(|e| {
                memory_error("waiting for a mapped buffer's last launch".to_string(), e)
            })?;
        }
        *self.mixed.get_mut() = false;
        *self.stream.get_mut() = 0;
        Ok(())
    }
}

impl DeviceArg for MappedMemory {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl std::fmt::Debug for MappedMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedMemory")
            .field("bytes", &self.bytes)
            .field("device", &self.event.context().ordinal())
            .finish()
    }
}

impl Drop for MappedMemory {
    fn drop(&mut self) {
        if let Err(e) = self.settle() {
            notice!(
                "[flynnel::cuda] freeing {} bytes of mapped memory after its launches \
                 could not be waited for: {e}",
                self.bytes
            );
        }
        if let Err(e) = self.event.context().bind_to_thread() {
            notice!("[flynnel::cuda] binding the context to free mapped memory: {e:?}");
        }
        // SAFETY: `host` came from cuMemHostAlloc and is freed only here.
        if let Err(e) = unsafe { result::free_host(self.host.as_ptr().cast()) } {
            notice!(
                "[flynnel::cuda] the driver refused to free {} bytes of mapped memory: {e:?}",
                self.bytes
            );
        }
    }
}

/// A launch's view of one [`KernelArg::Buffer`]: the typed slice, handed
/// to cudarc as it is, so cudarc orders the launch after the buffer's
/// last write and its free after the launch.
enum BufferRef<'a> {
    U8(&'a CudaSlice<u8>),
    I32(&'a CudaSlice<i32>),
    U32(&'a CudaSlice<u32>),
    I64(&'a CudaSlice<i64>),
    U64(&'a CudaSlice<u64>),
    F32(&'a CudaSlice<f32>),
    F64(&'a CudaSlice<f64>),
    /// A [`MappedBuffer`]'s memory, passed as its device address; the
    /// launch notes itself on it once queued.
    Mapped(&'a MappedMemory),
}

impl<'a> BufferRef<'a> {
    /// The [`DeviceBuffer`] or [`MappedBuffer`] behind `arg`, or `None`
    /// when it is some other backend's memory.
    fn of(arg: &'a dyn DeviceArg) -> Option<Self> {
        let any = arg.as_any();
        macro_rules! find {
            ($($t:ty => $variant:ident),*) => {
                $(
                    if let Some(buffer) = any.downcast_ref::<DeviceBuffer<$t>>() {
                        return Some(BufferRef::$variant(&buffer.slice));
                    }
                )*
            };
        }
        find!(u8 => U8, i32 => I32, u32 => U32, i64 => I64, u64 => U64, f32 => F32, f64 => F64);
        any.downcast_ref::<MappedMemory>().map(BufferRef::Mapped)
    }

    fn context(&self) -> &Arc<CudaContext> {
        match self {
            BufferRef::U8(s) => s.context(),
            BufferRef::I32(s) => s.context(),
            BufferRef::U32(s) => s.context(),
            BufferRef::I64(s) => s.context(),
            BufferRef::U64(s) => s.context(),
            BufferRef::F32(s) => s.context(),
            BufferRef::F64(s) => s.context(),
            BufferRef::Mapped(m) => m.event.context(),
        }
    }

    fn push_to<'b>(&self, builder: &mut LaunchArgs<'b>)
    where
        'a: 'b,
    {
        match *self {
            BufferRef::U8(s) => builder.arg(s),
            BufferRef::I32(s) => builder.arg(s),
            BufferRef::U32(s) => builder.arg(s),
            BufferRef::I64(s) => builder.arg(s),
            BufferRef::U64(s) => builder.arg(s),
            BufferRef::F32(s) => builder.arg(s),
            BufferRef::F64(s) => builder.arg(s),
            BufferRef::Mapped(m) => builder.arg(&m.device),
        };
    }
}

/// cudarc-backed reference CUDA backend.
pub struct CudaBackend {
    device_id: u32,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Secondary stream for ping-pong / double-buffered dispatch.
    /// Independent of the default `stream` so two operations on
    /// different streams overlap on the GPU. Consumers that
    /// alternate device buffers across iterations use this stream
    /// for the "pong" iteration while `stream` carries the "ping".
    secondary_stream: Arc<CudaStream>,
    caps: BackendCapabilities,
    next_handle: AtomicU64,
    /// Loaded kernels indexed by the handle the registration handed
    /// out. `next_handle` counts up from zero, so the handle IS the
    /// index and there is nothing to hash. Nothing is ever removed:
    /// entries live until the backend drops, which is what keeps a
    /// module alive for as long as any function pointer taken from it.
    kernels: Box<[AtomicPtr<LoadedKernel>; MAX_KERNELS]>,
    /// Persistent worker thread that processes `dispatch_one`
    /// work items. Routed through a flynnel notify hub
    /// (FlynnelRing + Parker); `Drop` calls `hub.shutdown()` to
    /// signal the worker to exit cleanly.
    worker_hub: NotifyHub<WorkItem>,
    /// Cached sender handle so `dispatch_one` does not pay the
    /// `Arc::clone` per call.
    worker_tx: NotifySender<WorkItem>,
    /// Join handle for the persistent worker; taken in `Drop`, which
    /// holds `&mut self` and so needs nothing to guard it.
    worker_handle: Option<JoinHandle<()>>,
}

/// A loaded module and the entry point taken from it, kept together
/// because the function borrows the module's lifetime.
struct LoadedKernel {
    /// Held so the module outlives the function pointer taken from it.
    _module: Arc<CudaModule>,
    function: CudaFunction,
}

/// Kernels one backend can register. Registration is a startup act
/// per distinct kernel source, and the table costs one pointer per
/// slot whether or not it is used.
const MAX_KERNELS: usize = 1024;

/// Bytes kept for each of the driver JIT's two logs when a registration
/// is refused.
const JIT_LOG_BYTES: usize = 16 * 1024;

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device_id", &self.device_id)
            .finish()
    }
}

/// What the driver reports about one CUDA device: its name, its
/// compute capability and size, and the limits a kernel is chosen and
/// sized by. A later fact is a new field.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceInfo {
    /// The device's name.
    pub name: String,
    /// Compute capability, the major version times ten plus the minor:
    /// 86 for 8.6, 120 for 12.0.
    pub capability: u32,
    /// Streaming multiprocessors.
    pub multiprocessors: u32,
    /// Peak multiprocessor clock, in kHz.
    pub clock_khz: u32,
    /// Single-precision over double-precision throughput: 2 where
    /// doubles run at half the rate of singles.
    pub fp64_ratio: u32,
    /// Device memory, in bytes.
    pub memory_bytes: u64,
    /// Whether the device shares the host's memory instead of having
    /// its own.
    pub integrated: bool,
    /// Whether the device can map page-locked host memory, which
    /// [`CudaBackend::map_host`] needs.
    pub maps_host_memory: bool,
    /// Whether the device and the host share one virtual address space.
    pub unified_addressing: bool,
    /// Whether atomic operations between the device and the host are
    /// native to the link between them.
    pub host_native_atomics: bool,
    /// Whether a kernel's run time is limited, as it is on a device that
    /// drives a display under the operating system's watchdog.
    pub watchdog: bool,
    /// L2 cache, in bytes.
    pub l2_cache_bytes: u32,
    /// Peak memory clock, in kHz.
    pub memory_clock_khz: u32,
    /// Width of the global memory bus, in bits.
    pub memory_bus_bits: u32,
    /// Threads in a warp.
    pub warp_size: u32,
    /// Most threads one block can hold.
    pub max_threads_per_block: u32,
    /// Most threads resident on one multiprocessor.
    pub max_threads_per_multiprocessor: u32,
    /// Shared memory one block can use without opting in to more, in
    /// bytes.
    pub shared_memory_per_block: u32,
    /// Asynchronous engines, each able to run a copy beside a kernel.
    pub async_engines: u32,
}

impl DeviceInfo {
    /// What the driver reports about device `ordinal`, read through
    /// `cuDeviceGetAttribute`, `cuDeviceGetName` and `cuDeviceTotalMem`
    /// without making a context, so reading it takes no device memory.
    ///
    /// [`BackendError::DeviceUnavailable`] when the driver is not loadable
    /// or has no device `ordinal`, and when it refuses any one fact, which
    /// is named in a note beside the driver's answer rather than read as
    /// zero.
    pub fn of_ordinal(ordinal: u32) -> Result<Self, BackendError> {
        use sys::CUdevice_attribute as A;
        driver_loadable(ordinal)?;
        let unavailable = |what: String| {
            BackendError::DeviceUnavailable(Backend::Cuda { device_id: ordinal })
                .map_io_context(what)
        };
        result::init().map_err(|e| unavailable(format!("initializing the driver: {e:?}")))?;
        let index = i32::try_from(ordinal).map_err(|e| {
            unavailable(format!("device {ordinal} is past the driver's range: {e}"))
        })?;
        let device = result::device::get(index)
            .map_err(|e| unavailable(format!("no device {ordinal}: {e:?}")))?;
        let fact = |attribute: A| -> Result<u32, BackendError> {
            // SAFETY: `device` came from cuDeviceGet on the initialized
            // driver.
            let value = unsafe { result::device::get_attribute(device, attribute) }
                .map_err(|e| unavailable(format!("reading {attribute:?}: {e:?}")))?;
            u32::try_from(value)
                .map_err(|e| unavailable(format!("{attribute:?} answered {value}: {e}")))
        };
        let flag = |attribute: A| fact(attribute).map(|value| value != 0);
        let name = result::device::get_name(device)
            .map_err(|e| unavailable(format!("reading the name: {e:?}")))?;
        // SAFETY: as for the attributes.
        let memory_bytes = unsafe { result::device::total_mem(device) }
            .map_err(|e| unavailable(format!("reading the total memory: {e:?}")))?;
        Ok(Self {
            name,
            capability: fact(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? * 10
                + fact(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            multiprocessors: fact(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            clock_khz: fact(A::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)?,
            fp64_ratio: fact(A::CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)?,
            memory_bytes: memory_bytes as u64,
            integrated: flag(A::CU_DEVICE_ATTRIBUTE_INTEGRATED)?,
            maps_host_memory: flag(A::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY)?,
            unified_addressing: flag(A::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)?,
            host_native_atomics: flag(A::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED)?,
            watchdog: flag(A::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)?,
            l2_cache_bytes: fact(A::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?,
            memory_clock_khz: fact(A::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?,
            memory_bus_bits: fact(A::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?,
            warp_size: fact(A::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?,
            max_threads_per_block: fact(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            max_threads_per_multiprocessor: fact(
                A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            )?,
            shared_memory_per_block: fact(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
            async_engines: fact(A::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)?,
        })
    }
}

impl CudaBackend {
    /// The kernel registered under `handle`.
    ///
    /// The borrow is the backend's: entries are never replaced or
    /// removed, and they are freed in `Drop`, which cannot run while
    /// a caller holds `&self`.
    fn loaded(&self, handle: KernelHandle) -> Result<&LoadedKernel, BackendError> {
        let unknown = || BackendError::Launch(format!("unknown kernel handle {handle:?}"));
        let slot = self.kernels.get(handle.0 as usize).ok_or_else(unknown)?;
        let published = slot.load(Ordering::Acquire);
        if published.is_null() {
            return Err(unknown());
        }
        // SAFETY: a published slot holds a Box this backend owns and
        // does not free before Drop.
        Ok(unsafe { &*published })
    }

    /// Initialize the CUDA driver on the primary device (id 0).
    /// Returns [`BackendError::DeviceUnavailable`] when the runtime
    /// is not loadable or the device cannot be opened.
    pub fn new() -> Result<Self, BackendError> {
        Self::with_device(0)
    }

    /// Initialize the CUDA driver on a specific device. `device_id`
    /// indexes into the platform's enumerated GPUs (0 for the
    /// first NVIDIA GPU).
    ///
    /// A driver refusal is [`BackendError::DeviceUnavailable`], which
    /// carries no text, so the driver's own error is written as a note
    /// beside it.
    pub fn with_device(device_id: u32) -> Result<Self, BackendError> {
        driver_loadable(device_id)?;
        let context =
            CudaContext::new(device_id as usize).map_err(|e| map_driver_error(device_id, e))?;
        let stream = context.default_stream();
        // Secondary stream for ping-pong dispatch. cudarc's
        // `new_stream` creates a non-default stream that runs
        // concurrently with the default stream on the device.
        let secondary_stream = context
            .new_stream()
            .map_err(|e| map_driver_error(device_id, e))?;
        let caps = probe_capabilities();

        // Spawn a persistent worker thread that consumes work
        // items from a flynnel notify hub. dispatch_one sends to
        // this hub instead of spawning a fresh OS thread per call.
        const CUDA_WORKER_RING_CAPACITY: usize = 1024;
        let worker_hub = NotifyHub::<WorkItem>::new(CUDA_WORKER_RING_CAPACITY, 1);
        let worker_tx = worker_hub.sender();
        let hub_for_worker = worker_hub.clone();
        let worker_handle = std::thread::Builder::new()
            .name(format!("flynnel-cuda-{device_id}"))
            .spawn(move || {
                let rx = hub_for_worker.register_consumer();
                while let Some(work) = rx.recv() {
                    work();
                }
            })
            .map_err(|e| {
                BackendError::DeviceUnavailable(Backend::Cuda { device_id })
                    .map_io_context(format!("worker thread spawn: {e}"))
            })?;

        Ok(Self {
            device_id,
            context,
            stream,
            secondary_stream,
            caps,
            next_handle: AtomicU64::new(1),
            kernels: Box::new([const { AtomicPtr::new(core::ptr::null_mut()) }; MAX_KERNELS]),
            worker_hub,
            worker_tx,
            worker_handle: Some(worker_handle),
        })
    }

    /// Convenience accessor exposing the underlying cudarc context
    /// for consumers that want to mix Flynnel-routed launches with
    /// direct cudarc usage (e.g. async stream synchronization
    /// outside the trait surface).
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Convenience accessor exposing the default stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Secondary stream for ping-pong / double-buffered dispatch.
    /// Independent of [`Self::stream`]; operations queued on the
    /// two streams overlap on the GPU. Consumers alternating
    /// device-buffer pairs across iterations use this stream for
    /// the "pong" iteration while [`Self::stream`] carries the
    /// "ping".
    pub fn secondary_stream(&self) -> &Arc<CudaStream> {
        &self.secondary_stream
    }

    /// Pick `stream` or `secondary_stream` by parity of `slot`.
    /// Use this from a pipelined dispatch loop where each
    /// iteration N owns a device-buffer pair indexed by
    /// `N & 1`.
    pub fn stream_for_slot(&self, slot: usize) -> &Arc<CudaStream> {
        if slot & 1 == 0 {
            &self.stream
        } else {
            &self.secondary_stream
        }
    }

    /// The device's name, as the driver reports it.
    pub fn device_name(&self) -> Result<String, BackendError> {
        self.context
            .name()
            .map_err(|e| map_driver_error(self.device_id, e))
    }

    /// Free and total memory on the device, in bytes, as the driver
    /// reports them at the moment of the call, so a caller can refuse or
    /// cut a block before allocating it.
    pub fn mem_info(&self) -> Result<(usize, usize), BackendError> {
        self.context
            .mem_get_info()
            .map_err(|e| memory_error(format!("reading device {}'s memory", self.device_id), e))
    }

    /// The CUDA devices the driver enumerates, counted without making a
    /// context. [`BackendError::DeviceUnavailable`] when the driver is not
    /// loadable, so a host without one reads differently from a driver
    /// that enumerates no device.
    pub fn device_count() -> Result<u32, BackendError> {
        driver_loadable(0)?;
        result::init().map_err(|e| map_driver_error(0, e))?;
        let count = result::device::get_count().map_err(|e| map_driver_error(0, e))?;
        u32::try_from(count).map_err(|e| {
            BackendError::DeviceUnavailable(Backend::Cuda { device_id: 0 })
                .map_io_context(format!("the driver counted {count} devices: {e}"))
        })
    }

    /// What the driver reports about this backend's device; see
    /// [`DeviceInfo::of_ordinal`], which this reads through.
    pub fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        DeviceInfo::of_ordinal(self.device_id)
    }

    /// Page-locked host memory for `len` elements of `T` that this device
    /// reads and writes directly, set to zero. Refused for zero elements,
    /// and by a device that cannot map host memory, which
    /// [`DeviceInfo::maps_host_memory`] reports.
    pub fn map_host<T: DeviceElement>(&self, len: usize) -> Result<MappedBuffer<T>, BackendError> {
        if len == 0 {
            return Err(BackendError::Memory(
                "mapped memory needs at least one element".to_string(),
            ));
        }
        let what = || {
            format!(
                "mapping {len} {} for device {}",
                std::any::type_name::<T>(),
                self.device_id
            )
        };
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| BackendError::Memory(format!("{}: past the address space", what())))?;
        let event = self
            .context
            .new_event(None)
            .map_err(|e| memory_error(what(), e))?;
        self.context
            .bind_to_thread()
            .map_err(|e| memory_error(what(), e))?;
        // SAFETY: this device's context is current on this thread, and the
        // driver writes the address of what it allocates.
        let raw = unsafe { result::malloc_host(bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) }
            .map_err(|e| memory_error(what(), e))?;
        let Some(host) = NonNull::new(raw.cast::<u8>()) else {
            return Err(BackendError::Memory(format!(
                "{}: the driver answered a null address",
                what()
            )));
        };
        let mut device: sys::CUdeviceptr = 0;
        // SAFETY: `host` starts an allocation cuMemHostAlloc just made with
        // DEVICEMAP in the current context, and the flags must be zero.
        let mapped =
            unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device, host.as_ptr().cast(), 0) }
                .result();
        if let Err(e) = mapped {
            // SAFETY: the allocation above, which nothing else holds.
            if let Err(freed) = unsafe { result::free_host(host.as_ptr().cast()) } {
                notice!(
                    "[flynnel::cuda] freeing {bytes} bytes the device could not map: {freed:?}"
                );
            }
            return Err(memory_error(format!("{}: no device address", what()), e));
        }
        // SAFETY: the allocation is `bytes` long and writable, and no launch
        // can reach it before this returns.
        unsafe { host.as_ptr().write_bytes(0, bytes) };
        Ok(MappedBuffer {
            memory: MappedMemory {
                host,
                device,
                bytes,
                event,
                stream: AtomicU64::new(0),
                mixed: AtomicBool::new(false),
            },
            len,
            element: PhantomData,
        })
    }

    /// Device memory for `len` elements of `T`, set to zero. Refused for
    /// zero elements, which the driver cannot allocate.
    pub fn alloc_zeroed<T: DeviceElement>(
        &self,
        len: usize,
    ) -> Result<DeviceBuffer<T>, BackendError> {
        if len == 0 {
            return Err(BackendError::Memory(
                "a device buffer needs at least one element".to_string(),
            ));
        }
        let slice = self.stream.alloc_zeros::<T>(len).map_err(|e| {
            memory_error(
                format!(
                    "allocating {len} {} on device {}",
                    std::any::type_name::<T>(),
                    self.device_id
                ),
                e,
            )
        })?;
        Ok(DeviceBuffer { slice })
    }

    /// Device memory holding a copy of `data`. Refused for an empty slice.
    pub fn upload<T: DeviceElement>(&self, data: &[T]) -> Result<DeviceBuffer<T>, BackendError> {
        if data.is_empty() {
            return Err(BackendError::Memory(
                "a device buffer needs at least one element".to_string(),
            ));
        }
        let slice = self.stream.clone_htod(data).map_err(|e| {
            memory_error(
                format!(
                    "uploading {} {} to device {}",
                    data.len(),
                    std::any::type_name::<T>(),
                    self.device_id
                ),
                e,
            )
        })?;
        Ok(DeviceBuffer { slice })
    }

    /// Copies `data` into the start of `buf`, leaving the rest as it was.
    /// Refused when `data` is longer than `buf`, or `buf` belongs to
    /// another backend.
    pub fn copy_in<T: DeviceElement>(
        &self,
        data: &[T],
        buf: &mut DeviceBuffer<T>,
    ) -> Result<(), BackendError> {
        self.check_owner(&buf.slice)?;
        if data.len() > buf.len() {
            return Err(BackendError::Memory(format!(
                "{} elements do not fit a buffer of {}",
                data.len(),
                buf.len()
            )));
        }
        if data.is_empty() {
            return Ok(());
        }
        self.stream
            .memcpy_htod(data, &mut buf.slice)
            .map_err(|e| memory_error(format!("copying {} elements in", data.len()), e))
    }

    /// Copies every element of `buf` into `out`, whose length must be the
    /// buffer's, and returns once `out` holds them.
    pub fn copy_out<T: DeviceElement>(
        &self,
        buf: &DeviceBuffer<T>,
        out: &mut [T],
    ) -> Result<(), BackendError> {
        if out.len() != buf.len() {
            return Err(BackendError::Memory(format!(
                "a buffer of {} elements read into a slice of {}",
                buf.len(),
                out.len()
            )));
        }
        self.copy_out_range(buf, 0, out)
    }

    /// Copies `out.len()` elements of `buf`, starting at element
    /// `offset`, into `out`, and returns once `out` holds them. A buffer
    /// whose use is a prefix of it reads back only that prefix.
    pub fn copy_out_range<T: DeviceElement>(
        &self,
        buf: &DeviceBuffer<T>,
        offset: usize,
        out: &mut [T],
    ) -> Result<(), BackendError> {
        self.check_owner(&buf.slice)?;
        let end = offset
            .checked_add(out.len())
            .filter(|&end| end <= buf.len())
            .ok_or_else(|| {
                BackendError::Memory(format!(
                    "{} elements from element {offset} of a buffer of {}",
                    out.len(),
                    buf.len()
                ))
            })?;
        if out.is_empty() {
            return Ok(());
        }
        let view = buf.slice.slice(offset..end);
        self.stream
            .memcpy_dtoh(&view, out)
            .map_err(|e| memory_error(format!("copying elements {offset}..{end} out"), e))?;
        // The copy is queued on the stream; the data is the caller's only
        // once the stream has run it.
        self.stream
            .synchronize()
            .map_err(|e| memory_error(format!("waiting for elements {offset}..{end}"), e))
    }

    /// Refuses a slice this backend did not allocate: memory another
    /// context owns is not addressable from this one.
    fn check_owner<T>(&self, slice: &CudaSlice<T>) -> Result<(), BackendError> {
        if Arc::ptr_eq(slice.context(), &self.context) {
            Ok(())
        } else {
            Err(BackendError::Memory(format!(
                "a buffer allocated by another CUDA backend was handed to device {}'s",
                self.device_id
            )))
        }
    }

    /// Loads kernel `name` from PTX text and returns the handle
    /// [`DispatchBackend::dispatch_kernel`] launches it by.
    ///
    /// The PTX is compiled for this device by the driver's own JIT, so
    /// loading needs no CUDA toolkit and never calls NVRTC: PTX written
    /// once for a target the driver supports loads on any machine with an
    /// NVIDIA driver. A refusal is [`BackendError::KernelCompile`] carrying
    /// the driver's error and the JIT's error and information logs.
    pub fn register_ptx(&self, name: &str, ptx: &str) -> Result<KernelHandle, BackendError> {
        if ptx.contains('\0') {
            return Err(BackendError::KernelCompile(format!(
                "the PTX for `{name}` holds a NUL byte, which ends the text the driver reads"
            )));
        }
        let module = match self.context.load_module(cudarc::nvrtc::Ptx::from_src(ptx)) {
            Ok(module) => module,
            Err(e) => {
                return Err(BackendError::KernelCompile(format!(
                    "the driver refused the PTX for `{name}`: {e:?}; {}",
                    jit_logs(&self.context, ptx)
                )));
            }
        };
        let function = module
            .load_function(name)
            .map_err(|e| BackendError::KernelCompile(format!("function lookup `{name}`: {e:?}")))?;
        let handle_id = self.next_handle.fetch_add(1, Ordering::Relaxed);
        let index = handle_id as usize;
        if index >= MAX_KERNELS {
            return Err(BackendError::KernelCompile(format!(
                "this backend holds {MAX_KERNELS} kernels and `{name}` would be the {index}th"
            )));
        }
        let entry = Box::into_raw(Box::new(LoadedKernel {
            _module: module,
            function,
        }));
        // The counter handed this index to this call alone, so nothing
        // else writes this slot, and the entry is complete before the
        // pointer that publishes it.
        self.kernels[index].store(entry, Ordering::Release);
        Ok(KernelHandle(handle_id))
    }

    /// Launch a registered kernel on a caller-chosen stream.
    /// Same semantics as [`DispatchBackend::dispatch_kernel`] but
    /// targets `stream` (typically [`Self::stream`] or
    /// [`Self::secondary_stream`]) instead of the default.
    /// Consumers driving a ping-pong pipeline call this with
    /// `stream_for_slot(iter & 1)` so adjacent iterations queue
    /// on independent streams and overlap on the GPU.
    ///
    /// A count of zero launches nothing and answers `Ok`. A
    /// [`KernelArg::HostSlice`] is copied to device memory of its own on
    /// `stream` before the launch and freed after it. A
    /// [`KernelArg::Buffer`] must be one this backend allocated.
    pub fn dispatch_kernel_on_stream(
        &self,
        stream: &Arc<CudaStream>,
        handle: KernelHandle,
        count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        let function = self.loaded(handle)?.function.clone();
        if count == 0 {
            return Ok(());
        }
        // Owned storage for every value the builder points into: it holds
        // raw pointers until `launch` returns, so all of it is declared
        // before the builder and outlives it.
        let mut i32s: Vec<i32> = Vec::new();
        let mut i64s: Vec<i64> = Vec::new();
        let mut u32s: Vec<u32> = Vec::new();
        let mut u64s: Vec<u64> = Vec::new();
        let mut f32s: Vec<f32> = Vec::new();
        let mut f64s: Vec<f64> = Vec::new();
        let mut staged: Vec<CudaSlice<u8>> = Vec::new();
        let mut buffers: Vec<BufferRef<'_>> = Vec::new();
        for arg in args {
            match arg {
                KernelArg::I32(v) => i32s.push(*v),
                KernelArg::I64(v) => i64s.push(*v),
                KernelArg::U32(v) => u32s.push(*v),
                KernelArg::U64(v) => u64s.push(*v),
                KernelArg::F32(v) => f32s.push(*v),
                KernelArg::F64(v) => f64s.push(*v),
                KernelArg::DevicePtr(p) => u64s.push(*p as u64),
                KernelArg::HostSlice(bytes) => staged.push(self.stage(stream, bytes)?),
                KernelArg::Buffer(buffer) => buffers.push(self.own_buffer(*buffer)?),
            }
        }
        let block = 256u32.min(count);
        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&function);
        // Second pass: push references into the storage in the order the
        // caller gave the arguments.
        let (mut ii32, mut ii64, mut iu32, mut iu64, mut if32, mut if64) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let (mut istaged, mut ibuffer) = (0usize, 0usize);
        for arg in args {
            match arg {
                KernelArg::I32(_) => {
                    builder.arg(&i32s[ii32]);
                    ii32 += 1;
                }
                KernelArg::I64(_) => {
                    builder.arg(&i64s[ii64]);
                    ii64 += 1;
                }
                KernelArg::U32(_) => {
                    builder.arg(&u32s[iu32]);
                    iu32 += 1;
                }
                KernelArg::U64(_) | KernelArg::DevicePtr(_) => {
                    builder.arg(&u64s[iu64]);
                    iu64 += 1;
                }
                KernelArg::F32(_) => {
                    builder.arg(&f32s[if32]);
                    if32 += 1;
                }
                KernelArg::F64(_) => {
                    builder.arg(&f64s[if64]);
                    if64 += 1;
                }
                KernelArg::HostSlice(_) => {
                    builder.arg(&staged[istaged]);
                    istaged += 1;
                }
                KernelArg::Buffer(_) => {
                    buffers[ibuffer].push_to(&mut builder);
                    ibuffer += 1;
                }
            }
        }
        // SAFETY: every argument reference points into the storage above,
        // which is declared before the builder and lives until this
        // function returns, after the launch returns. The function comes
        // from a module the kernel table keeps alive. Argument count and
        // type correctness is a contract with the kernel's author, the
        // safety hole cudarc documents on launch().
        unsafe { builder.launch(cfg) }.map_err(|e| BackendError::Launch(format!("{e:?}")))?;
        note_mapped(stream, &buffers)
    }

    /// A host slice's bytes in device memory of their own for one launch
    /// on `stream`. Dropped after the launch, its free waits for the
    /// kernel.
    fn stage(&self, stream: &Arc<CudaStream>, bytes: &[u8]) -> Result<CudaSlice<u8>, BackendError> {
        if bytes.is_empty() {
            return Err(BackendError::Memory(
                "an empty host slice has no device copy to pass".to_string(),
            ));
        }
        stream.clone_htod(bytes).map_err(|e| {
            memory_error(
                format!(
                    "copying a {}-byte host slice to device {}",
                    bytes.len(),
                    self.device_id
                ),
                e,
            )
        })
    }

    /// The typed buffer behind a [`KernelArg::Buffer`]. Another
    /// backend's kind of memory is [`BackendError::NotSupported`]; a CUDA
    /// buffer another backend allocated is refused on the terms of
    /// [`Self::check_owner`].
    fn own_buffer<'a>(&self, arg: &'a dyn DeviceArg) -> Result<BufferRef<'a>, BackendError> {
        let buffer = BufferRef::of(arg).ok_or(BackendError::NotSupported)?;
        if Arc::ptr_eq(buffer.context(), &self.context) {
            Ok(buffer)
        } else {
            Err(BackendError::Memory(format!(
                "{arg:?} was allocated by another CUDA backend than device {}'s",
                self.device_id
            )))
        }
    }
}

impl DispatchBackend for CudaBackend {
    fn id(&self) -> Backend {
        Backend::Cuda {
            device_id: self.device_id,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.caps
    }

    fn dispatch_parallel_for(&self, count: u32, work: &(dyn Fn(u32) + Send + Sync)) {
        // The closure body is CPU-runnable (an arbitrary Rust
        // closure cannot codegen to PTX). For GPU codegen, callers
        // use the `dispatch_kernel` handle path; this method runs
        // the CPU-shaped body fan-out so a `DispatchBackend` user
        // sees consistent dispatch_parallel_for semantics even on
        // GPU-class backends.
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
        // Send to the persistent worker thread (no per-call OS
        // thread spawn). The notify hub is MPMC and lock-free on
        // the hot path. It closes only in Drop, which cannot run while
        // a caller holds `&self`, and a closed hub is said rather than
        // passed over.
        if let NotifySendResult::Closed(_) = self.worker_tx.send(work) {
            notice!(
                "[flynnel::cuda] device {}: the worker has shut down, so the work item did not run",
                self.device_id
            );
        }
    }

    fn register_kernel(&self, name: &str, source: &[u8]) -> Result<KernelHandle, BackendError> {
        // The `source` is expected to be PTX text (UTF-8 bytes).
        let ptx = std::str::from_utf8(source)
            .map_err(|e| BackendError::KernelCompile(format!("PTX must be UTF-8: {e}")))?;
        self.register_ptx(name, ptx)
    }

    fn dispatch_kernel(
        &self,
        handle: KernelHandle,
        count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        // Launch geometry: blocks of 256 work-items and as many as cover
        // `count`. Consumers that need precise launch configuration ship
        // their own backend impl; the reference impl provides a
        // one-size-fits-most heuristic.
        self.dispatch_kernel_on_stream(&self.stream, handle, count, args)
    }

    fn dispatch_kernel_sync(
        &self,
        handle: KernelHandle,
        count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        // CUDA launches queue asynchronously on the stream; the
        // completion contract the auto-routing layer times against
        // needs the launch AND a stream synchronize.
        self.dispatch_kernel(handle, count, args)?;
        self.stream
            .synchronize()
            .map_err(|e| BackendError::Launch(format!("stream synchronize: {e:?}")))
    }
}

/// Capability descriptor for an NVIDIA SIMT backend. Conservative
/// nominals derived from the cudarc 0.19 / CUDA 12.6 ABI surface;
/// consumers that need exact device characteristics query cudarc
/// directly via [`CudaBackend::context`].
fn probe_capabilities() -> BackendCapabilities {
    BackendCapabilities {
        // NVIDIA warp is 32 threads.
        simt_width: 32,
        // Coarse upper bound: most modern NVIDIA GPUs hold 50k-
        // 200k threads in flight. 100k is a safe nominal.
        max_threads_in_flight: 100_000,
        // Driver launch overhead ~10us on PCIe.
        launch_latency_ns: 10_000,
        // PCIe 4.0 x16 sustained ~25 GB/s.
        h2d_bw_bytes_per_sec: 25_000_000_000,
    }
}

/// Notes a queued launch on `stream` on each [`MappedBuffer`] it passed.
/// A record the driver refuses is reported with the kernel already
/// queued, and that buffer's next host slice waits for the whole device.
fn note_mapped(stream: &CudaStream, buffers: &[BufferRef<'_>]) -> Result<(), BackendError> {
    let mut refused = Vec::new();
    for buffer in buffers {
        if let BufferRef::Mapped(memory) = buffer
            && let Err(e) = memory.note_launch(stream)
        {
            refused.push(format!("{e:?}"));
        }
    }
    if refused.is_empty() {
        return Ok(());
    }
    Err(BackendError::Launch(format!(
        "the kernel was queued, and recording its launch on {} mapped buffer(s) failed: {}; \
         each one's next host slice waits for the whole device",
        refused.len(),
        refused.join("; ")
    )))
}

/// The driver loadable, or [`BackendError::DeviceUnavailable`] for
/// `device_id`. Two gates before the first driver call: cudarc resolves
/// its symbols lazily and panics when it cannot load libcuda, so the
/// driver must be known loadable first. `cuda_available` is the cached
/// probe, and `is_culib_present` answers against cudarc's own
/// library-name candidates.
fn driver_loadable(device_id: u32) -> Result<(), BackendError> {
    if !crate::backend::detect::cuda_available() {
        return Err(BackendError::DeviceUnavailable(Backend::Cuda { device_id }));
    }
    // SAFETY: the call only attempts a `libloading::Library::new` on
    // each candidate name and reports whether one resolved.
    if !unsafe { sys::is_culib_present() } {
        return Err(BackendError::DeviceUnavailable(Backend::Cuda { device_id }));
    }
    Ok(())
}

/// `DeviceUnavailable` for a driver refusal. The variant carries no text,
/// so the driver's own error is written as a note beside it rather than
/// dropped.
fn map_driver_error(device_id: u32, e: DriverError) -> BackendError {
    BackendError::DeviceUnavailable(Backend::Cuda { device_id }).map_io_context(format!("{e:?}"))
}

/// A memory call's failure with the driver's error beside what was asked.
fn memory_error(what: String, e: DriverError) -> BackendError {
    BackendError::Memory(format!("{what}: {e:?}"))
}

/// The driver JIT's own account of why it refused `ptx`. The text is
/// loaded a second time through `cuModuleLoadDataEx` with error and
/// information log buffers, and the logs are what that answers; a second
/// attempt that loads is unloaded at once and says so.
fn jit_logs(context: &CudaContext, ptx: &str) -> String {
    let source = match std::ffi::CString::new(ptx) {
        Ok(source) => source,
        Err(nul) => return format!("the PTX holds a NUL byte at {}", nul.nul_position()),
    };
    if let Err(e) = context.bind_to_thread() {
        return format!("no JIT log, the context would not bind: {e:?}");
    }
    let mut error_log = vec![0u8; JIT_LOG_BYTES];
    let mut info_log = vec![0u8; JIT_LOG_BYTES];
    let mut options = [
        sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
        sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        sys::CUjit_option::CU_JIT_INFO_LOG_BUFFER,
        sys::CUjit_option::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        sys::CUjit_option::CU_JIT_LOG_VERBOSE,
    ];
    // The two sizes and the verbosity flag travel in their pointer slots
    // by value, as the driver API defines for these options.
    let mut values: [*mut std::ffi::c_void; 5] = [
        error_log.as_mut_ptr().cast(),
        std::ptr::without_provenance_mut(JIT_LOG_BYTES),
        info_log.as_mut_ptr().cast(),
        std::ptr::without_provenance_mut(JIT_LOG_BYTES),
        std::ptr::without_provenance_mut(1),
    ];
    let mut module: sys::CUmodule = std::ptr::null_mut();
    // SAFETY: `source` is a NUL-terminated image that outlives the call;
    // `options` and `values` hold five entries each; both log buffers are
    // JIT_LOG_BYTES long, as their size entries say, and outlive the call.
    let loaded = unsafe {
        sys::cuModuleLoadDataEx(
            &mut module,
            source.as_ptr().cast(),
            options.len() as u32,
            options.as_mut_ptr(),
            values.as_mut_ptr(),
        )
    };
    let mut text = format!(
        "JIT {loaded:?}; error log: {}; information log: {}",
        log_text(&error_log),
        log_text(&info_log)
    );
    if !module.is_null() {
        // SAFETY: the call above loaded this module, and nothing else
        // holds it.
        let unloaded = unsafe { cudarc::driver::result::module::unload(module) };
        text.push_str(&format!(
            "; that load succeeded, and unloading it answered {unloaded:?}"
        ));
    }
    text
}

/// A NUL-terminated log buffer as text, or `empty` when the driver wrote
/// nothing into it.
fn log_text(buffer: &[u8]) -> String {
    let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
    let text = String::from_utf8_lossy(&buffer[..end]).trim().to_string();
    if text.is_empty() {
        "empty".to_string()
    } else {
        text
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Shut down the notify hub: the worker thread's recv()
        // returns None and it exits cleanly.
        self.worker_hub.shutdown();
        if let Some(handle) = self.worker_handle.take()
            && let Err(panicked) = handle.join()
        {
            // A worker that died of a panic took its work with it, and
            // a drop cannot return that to anyone. Saying so is the
            // only thing left; dropping it reports a clean shutdown.
            notice!(
                "flynnel: CUDA backend worker thread panicked: {}",
                panic_message(&panicked)
            );
        }
        for slot in self.kernels.iter() {
            let published = slot.swap(core::ptr::null_mut(), Ordering::AcqRel);
            if !published.is_null() {
                // SAFETY: this is the last owner of the backend, so no
                // caller can be holding a reference into the entry.
                drop(unsafe { Box::from_raw(published) });
            }
        }
    }
}

/// Whatever a panicking thread carried, as text.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    match payload.downcast_ref::<String>() {
        Some(s) => s.clone(),
        None => "a payload of an unknown type".to_string(),
    }
}

/// Helper used in the constructor to attach a context note
/// to a `DeviceUnavailable` error before returning. The
/// `BackendError::DeviceUnavailable` variant carries no message
/// field; the trait-side context is preserved by writing it as a note.
trait WithIoContext: Sized {
    fn map_io_context(self, msg: String) -> Self;
}

impl WithIoContext for BackendError {
    fn map_io_context(self, msg: String) -> Self {
        match self {
            BackendError::DeviceUnavailable(b) => {
                notice!("[flynnel::cuda] {}: {msg}", b.name());
                BackendError::DeviceUnavailable(b)
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::detect::cuda_available;

    const ADD_ONE_PTX: &str = include_str!("../../kernels/add_one.ptx");

    /// The backend on device 0, or `None` on a host with no usable CUDA
    /// device, which is what these tests skip on. Any other refusal
    /// fails the test that asked, rather than reading as no device.
    fn backend() -> Option<CudaBackend> {
        match CudaBackend::new() {
            Ok(backend) => Some(backend),
            Err(BackendError::DeviceUnavailable(_)) => None,
            Err(other) => panic!("the backend refused for a reason other than no device: {other}"),
        }
    }

    #[test]
    fn cuda_backend_construction_matches_availability() {
        let res = CudaBackend::new();
        if cuda_available() {
            match res {
                Ok(b) => assert_eq!(b.id(), Backend::Cuda { device_id: 0 }),
                Err(BackendError::DeviceUnavailable(_)) => {}
                Err(e) => panic!("unexpected CUDA construction error: {e}"),
            }
        } else {
            assert!(matches!(res, Err(BackendError::DeviceUnavailable(_))));
        }
    }

    #[test]
    fn capabilities_report_warp_width_32() {
        if let Some(b) = backend() {
            assert_eq!(b.capabilities().simt_width, 32);
        }
    }

    #[test]
    fn dispatch_parallel_for_invokes_each_index_on_host_fanout() {
        let Some(backend) = backend() else {
            return;
        };
        use std::sync::atomic::AtomicU32;
        let counters: Vec<AtomicU32> = (0..256).map(|_| AtomicU32::new(0)).collect();
        let cref = &counters;
        backend.dispatch_parallel_for(256, &|i| {
            cref[i as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        for c in &counters {
            assert_eq!(c.load(std::sync::atomic::Ordering::Relaxed), 1);
        }
    }

    /// Smoke-test the register_kernel + dispatch_kernel path with
    /// a trivial PTX kernel (mul3.ptx is a no-op entry that takes
    /// one u32 dummy arg and returns). Validates that cudarc can
    /// parse the PTX, load the module, look up the function by
    /// name, and successfully launch it - all without depending
    /// on any meaningful kernel math. Skips when CUDA is not
    /// available on the host.
    #[test]
    fn register_and_dispatch_trivial_kernel() {
        let Some(backend) = backend() else {
            return;
        };
        const TRIVIAL_PTX: &str = include_str!("../../kernels/mul3.ptx");
        let handle = backend
            .register_kernel("mul3", TRIVIAL_PTX.as_bytes())
            .expect("registering trivial PTX kernel");
        // Launch with 32 work-items and the one u32 dummy arg the
        // kernel signature declares. The kernel body is just `ret;`
        // so success here means: PTX parsed, module loaded, function
        // resolved, launch geometry accepted, kernel ran to completion.
        backend
            .dispatch_kernel(handle, 32, &[KernelArg::U32(7)])
            .expect("dispatching trivial PTX kernel");
        backend
            .stream()
            .synchronize()
            .expect("sync after trivial kernel dispatch");
    }

    #[test]
    fn a_buffer_round_trips_through_a_kernel() {
        let Some(backend) = backend() else {
            return;
        };
        let handle = backend
            .register_ptx("add_one", ADD_ONE_PTX)
            .expect("add_one loads");
        // More than one block, and not a multiple of the block size, so
        // the tail threads' bound check is exercised.
        let data: Vec<u32> = (0..1000).map(|i| i * 3).collect();
        let buf = backend.upload(&data).expect("upload");
        let n = data.len() as u32;
        backend
            .dispatch_kernel(handle, n, &[buf.arg(), KernelArg::U32(n)])
            .expect("launch");
        let mut out = vec![0u32; data.len()];
        backend.copy_out(&buf, &mut out).expect("copy out");
        let want: Vec<u32> = data.iter().map(|v| v + 1).collect();
        assert_eq!(out, want);
    }

    #[test]
    fn a_ranged_readback_reads_only_its_range() {
        let Some(backend) = backend() else {
            return;
        };
        let data: Vec<f64> = (0..300).map(f64::from).collect();
        let buf = backend.upload(&data).expect("upload");
        let mut part = vec![0.0f64; 10];
        backend
            .copy_out_range(&buf, 100, &mut part)
            .expect("ranged copy");
        assert_eq!(part, data[100..110]);
        let mut past = vec![0.0f64; 10];
        assert!(matches!(
            backend.copy_out_range(&buf, 295, &mut past),
            Err(BackendError::Memory(_))
        ));
    }

    #[test]
    fn a_zero_count_launches_nothing() {
        let Some(backend) = backend() else {
            return;
        };
        let handle = backend
            .register_ptx("add_one", ADD_ONE_PTX)
            .expect("add_one loads");
        let data = vec![7u32; 64];
        let buf = backend.upload(&data).expect("upload");
        backend
            .dispatch_kernel(handle, 0, &[buf.arg(), KernelArg::U32(64)])
            .expect("a zero count is not an error");
        let mut out = vec![0u32; 64];
        backend.copy_out(&buf, &mut out).expect("copy out");
        assert_eq!(out, data, "nothing ran, so nothing changed");
    }

    #[test]
    fn a_host_slice_reaches_the_kernel_as_device_memory() {
        let Some(backend) = backend() else {
            return;
        };
        let handle = backend
            .register_ptx("copy_add_one", ADD_ONE_PTX)
            .expect("copy_add_one loads");
        let src: Vec<u32> = (0..257).collect();
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dst = backend.alloc_zeroed::<u32>(src.len()).expect("alloc");
        let n = src.len() as u32;
        backend
            .dispatch_kernel(
                handle,
                n,
                &[KernelArg::HostSlice(&bytes), dst.arg(), KernelArg::U32(n)],
            )
            .expect("launch");
        let mut out = vec![0u32; src.len()];
        backend.copy_out(&dst, &mut out).expect("copy out");
        let want: Vec<u32> = src.iter().map(|v| v + 1).collect();
        assert_eq!(out, want);
    }

    #[test]
    fn copy_in_fills_a_prefix_and_leaves_the_rest() {
        let Some(backend) = backend() else {
            return;
        };
        let mut buf = backend.alloc_zeroed::<i64>(8).expect("alloc");
        backend.copy_in(&[1, 2, 3], &mut buf).expect("copy in");
        let mut out = vec![9i64; 8];
        backend.copy_out(&buf, &mut out).expect("copy out");
        assert_eq!(out, [1, 2, 3, 0, 0, 0, 0, 0]);
        assert!(matches!(
            backend.copy_in(&[0i64; 9], &mut buf),
            Err(BackendError::Memory(_))
        ));
    }

    #[test]
    fn bad_ptx_is_refused_with_the_jit_log() {
        let Some(backend) = backend() else {
            return;
        };
        let bad = ".version 7.0\n.target sm_70\n.address_size 64\n\
                   .visible .entry broken() { this is not an instruction; }\n";
        match backend.register_ptx("broken", bad) {
            Err(BackendError::KernelCompile(text)) => {
                assert!(text.contains("error log"), "{text}");
                assert!(text.contains("broken"), "{text}");
            }
            other => panic!("malformed PTX must be refused, got {other:?}"),
        }
        assert!(matches!(
            backend.register_ptx("nul", "a\0b"),
            Err(BackendError::KernelCompile(_))
        ));
    }

    #[test]
    fn the_device_answers_its_name_and_memory() {
        let Some(backend) = backend() else {
            return;
        };
        let name = backend.device_name().expect("name");
        assert!(!name.trim().is_empty());
        let (free, total) = backend.mem_info().expect("mem_info");
        assert!(total > 0);
        assert!(free <= total, "{free} free of {total}");
        eprintln!("device 0: {name}, {free} of {total} bytes free");
    }

    #[test]
    fn another_backends_buffer_is_refused() {
        let (Some(one), Some(two)) = (backend(), backend()) else {
            return;
        };
        let handle = one
            .register_ptx("add_one", ADD_ONE_PTX)
            .expect("add_one loads");
        let theirs = two.upload(&[1u32, 2, 3]).expect("upload");
        assert!(matches!(
            one.dispatch_kernel(handle, 3, &[theirs.arg(), KernelArg::U32(3)]),
            Err(BackendError::Memory(_))
        ));
        let mut out = [0u32; 3];
        assert!(matches!(
            one.copy_out(&theirs, &mut out),
            Err(BackendError::Memory(_))
        ));
    }

    #[test]
    fn a_kernel_reads_and_writes_mapped_memory_with_no_copy() {
        let Some(backend) = backend() else {
            return;
        };
        let handle = backend
            .register_ptx("add_one", ADD_ONE_PTX)
            .expect("add_one loads");
        let mut mapped = backend.map_host::<u32>(1000).expect("map");
        for (i, v) in mapped
            .as_mut_slice()
            .expect("host slice")
            .iter_mut()
            .enumerate()
        {
            *v = i as u32 * 3;
        }
        let n = mapped.len() as u32;
        backend
            .dispatch_kernel(handle, n, &[mapped.arg(), KernelArg::U32(n)])
            .expect("launch");
        let want: Vec<u32> = (0..n).map(|i| i * 3 + 1).collect();
        assert_eq!(mapped.as_slice().expect("host slice"), &want[..]);
    }

    #[test]
    fn mapped_memory_starts_zeroed_and_refuses_zero_elements() {
        let Some(backend) = backend() else {
            return;
        };
        let mut mapped = backend.map_host::<u64>(17).expect("map");
        assert!(
            mapped
                .as_slice()
                .expect("host slice")
                .iter()
                .all(|&v| v == 0)
        );
        assert!(matches!(
            backend.map_host::<u8>(0),
            Err(BackendError::Memory(_))
        ));
    }

    #[test]
    fn a_host_slice_waits_for_the_last_launch_on_one_stream() {
        let Some(backend) = backend() else {
            return;
        };
        let handle = backend
            .register_ptx("add_one", ADD_ONE_PTX)
            .expect("add_one loads");
        let mut mapped = backend.map_host::<u32>(4096).expect("map");
        let n = mapped.len() as u32;
        for _ in 0..3 {
            backend
                .dispatch_kernel(handle, n, &[mapped.arg(), KernelArg::U32(n)])
                .expect("launch");
        }
        assert!(
            !*mapped.memory.mixed.get_mut(),
            "three launches on one stream wait on the event"
        );
        assert!(
            mapped
                .as_slice()
                .expect("host slice")
                .iter()
                .all(|&v| v == 3)
        );
        assert_eq!(
            *mapped.memory.stream.get_mut(),
            0,
            "a settle starts the account again"
        );
    }

    #[test]
    fn a_host_slice_waits_for_the_device_once_two_streams_used_the_buffer() {
        let Some(backend) = backend() else {
            return;
        };
        let handle = backend
            .register_ptx("add_one", ADD_ONE_PTX)
            .expect("add_one loads");
        let mut mapped = backend.map_host::<u32>(4096).expect("map");
        let n = mapped.len() as u32;
        backend
            .dispatch_kernel_on_stream(
                backend.stream(),
                handle,
                n,
                &[mapped.arg(), KernelArg::U32(n)],
            )
            .expect("first launch");
        backend.stream().synchronize().expect("first launch done");
        backend
            .dispatch_kernel_on_stream(
                backend.secondary_stream(),
                handle,
                n,
                &[mapped.arg(), KernelArg::U32(n)],
            )
            .expect("second launch");
        assert!(
            *mapped.memory.mixed.get_mut(),
            "a second stream marks the buffer mixed"
        );
        assert!(
            mapped
                .as_slice()
                .expect("host slice")
                .iter()
                .all(|&v| v == 2)
        );
        assert!(!*mapped.memory.mixed.get_mut(), "a settle clears the mark");
    }

    #[test]
    fn a_launch_on_the_per_thread_stream_counts_as_mixed() {
        let Some(backend) = backend() else {
            return;
        };
        let handle = backend
            .register_ptx("add_one", ADD_ONE_PTX)
            .expect("add_one loads");
        let mut mapped = backend.map_host::<u32>(64).expect("map");
        let per_thread = backend.context().per_thread_stream();
        backend
            .dispatch_kernel_on_stream(&per_thread, handle, 64, &[mapped.arg(), KernelArg::U32(64)])
            .expect("launch");
        assert!(
            *mapped.memory.mixed.get_mut(),
            "one handle names a stream per thread"
        );
        assert!(
            mapped
                .as_slice()
                .expect("host slice")
                .iter()
                .all(|&v| v == 1)
        );
    }

    #[test]
    fn another_backends_mapped_buffer_is_refused() {
        let (Some(one), Some(two)) = (backend(), backend()) else {
            return;
        };
        let handle = one
            .register_ptx("add_one", ADD_ONE_PTX)
            .expect("add_one loads");
        let theirs = two.map_host::<u32>(3).expect("map");
        assert!(matches!(
            one.dispatch_kernel(handle, 3, &[theirs.arg(), KernelArg::U32(3)]),
            Err(BackendError::Memory(_))
        ));
    }

    #[test]
    fn the_device_answers_every_fact_and_agrees_with_the_context() {
        let Some(backend) = backend() else {
            return;
        };
        let info = backend.device_info().expect("device info");
        eprintln!("device 0: {info:?}");
        assert_eq!(info.name, backend.device_name().expect("name"));
        let (major, minor) = backend.context().compute_capability().expect("capability");
        assert_eq!(
            i64::from(info.capability),
            i64::from(major) * 10 + i64::from(minor)
        );
        assert_eq!(
            Some(info.multiprocessors),
            crate::backend::detect::cuda_sm_count(0)
        );
        assert_eq!(
            info.memory_bytes,
            backend.mem_info().expect("mem_info").1 as u64
        );
        assert_eq!(info.warp_size, backend.capabilities().simt_width);
        assert!(info.maps_host_memory, "map_host's tests ran on this device");
        assert!(info.max_threads_per_block <= info.max_threads_per_multiprocessor);
        assert!(info.clock_khz > 0 && info.memory_clock_khz > 0);
        assert!(info.memory_bus_bits > 0 && info.l2_cache_bytes > 0);
        assert!(info.fp64_ratio >= 1 && info.shared_memory_per_block > 0);
        assert_eq!(
            DeviceInfo::of_ordinal(0).expect("device 0 by ordinal"),
            info
        );
        let count = CudaBackend::device_count().expect("count");
        assert!(count >= 1);
        assert!(
            matches!(
                DeviceInfo::of_ordinal(count),
                Err(BackendError::DeviceUnavailable(_))
            ),
            "no device past the count"
        );
    }

    #[test]
    fn zero_elements_are_refused_before_the_driver() {
        let Some(backend) = backend() else {
            return;
        };
        assert!(matches!(
            backend.alloc_zeroed::<u8>(0),
            Err(BackendError::Memory(_))
        ));
        assert!(matches!(
            backend.upload::<u8>(&[]),
            Err(BackendError::Memory(_))
        ));
    }
}

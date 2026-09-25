//! The CUDA backend's device memory used the way a library linking the
//! crate beside another module's pool uses it: allocate, upload, launch,
//! copy out and free, without the scheduler's arena ever starting.
//!
//! A test binary of its own, because the arena is process-wide: in the
//! library's test binary some other test has started it long before this
//! one would look. A host with no usable CUDA device checks the arena
//! alone, since a backend that could not be made must not start it
//! either.

#![cfg(feature = "cuda-reference")]

use flynnel::backend::cuda::CudaBackend;
use flynnel::backend::{BackendError, DispatchBackend, KernelArg};

const ADD_ONE_PTX: &str = include_str!("../kernels/add_one.ptx");

#[test]
fn device_memory_round_trips_without_starting_the_arena() {
    match CudaBackend::new() {
        Ok(backend) => {
            let handle = backend
                .register_ptx("add_one", ADD_ONE_PTX)
                .expect("add_one loads through the driver");
            let data: Vec<u32> = (0..4096).collect();
            let buf = backend.upload(&data).expect("upload");
            let n = data.len() as u32;
            backend
                .dispatch_kernel(handle, n, &[buf.arg(), KernelArg::U32(n)])
                .expect("launch");
            let mut out = vec![0u32; data.len()];
            backend.copy_out(&buf, &mut out).expect("copy out");
            assert!(out.iter().zip(&data).all(|(o, d)| *o == d + 1));
            let mut tail = vec![0u32; 16];
            backend
                .copy_out_range(&buf, data.len() - 16, &mut tail)
                .expect("ranged copy out");
            assert_eq!(tail, out[data.len() - 16..]);
            drop(buf);
            drop(backend);
        }
        Err(BackendError::DeviceUnavailable(device)) => {
            eprintln!(
                "no usable CUDA device ({}): checking the arena alone",
                device.name()
            );
        }
        Err(other) => panic!("the CUDA backend refused for a reason other than no device: {other}"),
    }
    assert!(
        !flynnel::sched::arena::global_local_arena_started(),
        "the CUDA path started the scheduler's arena"
    );
}

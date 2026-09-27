//! The CUDA backend's device memory used the way a library linking the
//! crate beside another module's pool uses it: allocate, upload, launch,
//! copy out and free, and the same work through mapped host memory with
//! no copy, without the scheduler's arena ever starting.
//!
//! A test binary of its own, because the arena is process-wide: in the
//! library's test binary some other test has started it long before this
//! one would look. A host with no usable CUDA device checks the arena
//! alone, since a backend that could not be made must not start it
//! either.

#![cfg(feature = "cuda-reference")]

use flynnel::backend::cuda::{CudaBackend, DeviceInfo};
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
            let mut mapped = backend
                .map_host::<u32>(data.len())
                .expect("map host memory");
            mapped
                .as_mut_slice()
                .expect("host slice")
                .copy_from_slice(&data);
            backend
                .dispatch_kernel(handle, n, &[mapped.arg(), KernelArg::U32(n)])
                .expect("launch on mapped memory");
            let answers = mapped.as_slice().expect("host slice");
            assert!(answers.iter().zip(&data).all(|(m, d)| *m == d + 1));
            drop(mapped);
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

/// nvidia-smi's rows, one per device: name, compute capability, total
/// memory in MiB, and the maximum multiprocessor and memory clocks in
/// MHz. An error saying why when nvidia-smi could not be run.
fn nvidia_smi_rows() -> Result<Vec<Vec<String>>, String> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| format!("nvidia-smi did not run: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "nvidia-smi exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split(',')
                .map(|field| field.trim().to_string())
                .collect()
        })
        .collect())
}

/// Each device's name and capability match its nvidia-smi row, its memory
/// clock equals nvidia-smi's maximum memory clock, and its total memory
/// and clock are no more than nvidia-smi's total and maximum SM clock.
#[test]
fn device_info_agrees_with_nvidia_smi() {
    let count = match CudaBackend::device_count() {
        Ok(count) => count,
        Err(BackendError::DeviceUnavailable(device)) => {
            eprintln!("no CUDA driver ({}): nothing to compare", device.name());
            return;
        }
        Err(other) => panic!("counting devices refused for a reason other than no driver: {other}"),
    };
    let rows = match nvidia_smi_rows() {
        Ok(rows) => rows,
        Err(why) => panic!("the driver counts {count} device(s) and {why}"),
    };
    for ordinal in 0..count {
        let info = DeviceInfo::of_ordinal(ordinal).expect("device info");
        eprintln!("device {ordinal}: {info:?}");
        let matching: Vec<&Vec<String>> = rows.iter().filter(|row| row[0] == info.name).collect();
        assert!(
            !matching.is_empty(),
            "nvidia-smi lists no device named {:?}: {rows:?}",
            info.name
        );
        for row in matching {
            eprintln!("nvidia-smi: {row:?}");
            let capability = row[1].replace('.', "");
            assert_eq!(info.capability.to_string(), capability, "{row:?}");
            match row[2].parse::<u64>() {
                Ok(mib) => assert!(
                    info.memory_bytes <= mib << 20,
                    "the device reports {} bytes, more than nvidia-smi's {mib} MiB: {row:?}",
                    info.memory_bytes
                ),
                Err(e) => eprintln!(
                    "nvidia-smi's memory.total {:?} is not a number ({e}); not compared",
                    row[2]
                ),
            }
            match row[3].parse::<u64>() {
                Ok(mhz) => assert!(
                    u64::from(info.clock_khz) <= mhz * 1000,
                    "the device's clock of {} kHz is above nvidia-smi's maximum of {mhz} MHz: {row:?}",
                    info.clock_khz
                ),
                Err(e) => eprintln!(
                    "nvidia-smi's clocks.max.sm {:?} is not a number ({e}); not compared",
                    row[3]
                ),
            }
            match row[4].parse::<u64>() {
                Ok(mhz) => assert_eq!(u64::from(info.memory_clock_khz), mhz * 1000, "{row:?}"),
                Err(e) => eprintln!(
                    "nvidia-smi's clocks.max.memory {:?} is not a number ({e}); not compared",
                    row[4]
                ),
            }
        }
    }
}

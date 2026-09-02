//! Wide-launch resident op: a full-grid kernel over a resident block,
//! spanning every SM, versus the one-block (one-SM) limit of the
//! doorbell user-op.
//!
//! Run with:
//!   cargo run --release --features gpu-peer --example gpu_wide_op_demo
//!
//! A 128x128 (16,384-pixel) image is pinned resident and 3x3
//! box-blurred. The identical grid-stride kernel is launched twice:
//! grid=1 (256 threads, one SM - what a doorbell user-op is capped
//! at) and grid=full (one thread per pixel, the whole device). Both
//! must produce the same result as a CPU blur; the wide launch is the
//! answer for a large data-parallel resident op.

use std::time::Instant;

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig};

const BLUR: &str = r#"
extern "C" __global__ void box_blur3(const float* src, float* dst,
                                     unsigned w, unsigned h) {
    unsigned n = w * h;
    for (unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += gridDim.x * blockDim.x) {
        int x = idx % w, y = idx / w;
        float sum = 0.0f; int cnt = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < (int)w && ny >= 0 && ny < (int)h) {
                    sum += src[ny * w + nx]; cnt++;
                }
            }
        dst[idx] = sum / (float)cnt;
    }
}
"#;

fn cpu_blur(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut sum, mut cnt) = (0f32, 0);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        sum += src[ny as usize * w + nx as usize];
                        cnt += 1;
                    }
                }
            }
            out[y * w + x] = sum / cnt as f32;
        }
    }
    out
}

fn main() {
    println!("=== Wide-launch resident op (full grid vs one SM) ===\n");
    let (w, h) = (128usize, 128usize);
    let n = w * h;
    let bytes = n * 4;
    println!("image {w}x{h} = {n} pixels ({} KiB), 3x3 box blur\n", bytes / 1024);

    // Slots must hold the whole image for the H2V upload; the pool
    // block must hold it too.
    let mut peer = match GpuPeer::init(GpuPeerConfig {
        slot_bytes: 256 * 1024,
        slots_per_lane: 4,
        vram_block_bytes: bytes as u32,
        vram_blocks: 8,
        ..GpuPeerConfig::default()
    }) {
        Ok(p) => p,
        Err(e) => {
            println!("substrate unavailable on this host: {e}");
            return;
        }
    };

    // A gradient with a spike so the blur is visibly non-trivial.
    let mut img = vec![0f32; n];
    for y in 0..h {
        for x in 0..w {
            img[y * w + x] = (x as f32) + (y as f32) * 0.5;
        }
    }
    img[(h / 2) * w + w / 2] = 9999.0;
    let img_bytes: Vec<u8> = img.iter().flat_map(|v| v.to_le_bytes()).collect();
    let want = cpu_blur(&img, w, h);

    let src = peer.pin(&img_bytes).expect("pin src");
    let dst = peer.pin(&vec![0u8; bytes]).expect("pin dst");
    let (src_ptr, _) = peer.resident_ptr(&src).expect("src ptr");
    let (dst_ptr, _) = peer.resident_ptr(&dst).expect("dst ptr");
    println!("[1] image pinned resident; src+dst blocks live in VRAM");

    let kernel = match peer.compile_wide_kernel(BLUR, "box_blur3") {
        Ok(k) => k,
        Err(e) => {
            println!("wide-kernel compile failed (NVRTC needed): {e}");
            return;
        }
    };
    println!("[2] user blur kernel NVRTC-compiled for the wide path");

    let full_grid = (n as u32).div_ceil(256);
    let runs = 50u32;

    let bench = |peer: &mut GpuPeer, grid: u32| -> (f64, Vec<f32>) {
        let mut best = f64::INFINITY;
        for _ in 0..runs {
            let t0 = Instant::now();
            peer.launch_wide(&kernel, grid, 256, &[src_ptr, dst_ptr], &[w as u32, h as u32])
                .expect("wide launch");
            best = best.min(t0.elapsed().as_secs_f64() * 1e6);
        }
        let mut out_bytes = vec![0u8; bytes];
        peer.fetch(&dst, &mut out_bytes).expect("fetch");
        let out: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (best, out)
    };

    // grid = 1: 256 threads on ONE SM - the doorbell user-op limit.
    let (g1_us, out1) = bench(&mut peer, 1);
    // grid = full: one thread per pixel, the whole device.
    let (gf_us, outf) = bench(&mut peer, full_grid);

    let bad1 = out1.iter().zip(&want).filter(|(a, b)| (**a - **b).abs() > 1e-2).count();
    let badf = outf.iter().zip(&want).filter(|(a, b)| (**a - **b).abs() > 1e-2).count();
    println!("\n[3] correctness vs CPU blur:");
    println!("    grid=1    : {}", if bad1 == 0 { "OK" } else { "FAIL" });
    println!("    grid=full : {}", if badf == 0 { "OK" } else { "FAIL" });
    assert_eq!(bad1, 0);
    assert_eq!(badf, 0);

    println!("\n[4] timing (min of {runs} runs):");
    println!("    grid=1    (  1 block , 256 thr, 1 SM)   : {g1_us:>8.1} us");
    println!("    grid=full ({full_grid:>3} blocks, 256 thr, all SM) : {gf_us:>8.1} us");
    println!("    wide-launch speedup: {:.2}x", g1_us / gf_us);

    peer.unpin(src).expect("unpin src");
    peer.unpin(dst).expect("unpin dst");

    println!("\nVERIFIED: the wide launch spreads a large resident op across every SM.");
    println!("A doorbell user-op stays one block (256 threads) - right for many small ops,");
    println!("wrong for a 12k+ pixel convolution; launch_wide is the full-device path.");
}

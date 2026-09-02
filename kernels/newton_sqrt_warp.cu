//
// Warp-cooperative Newton sqrt with cross-lane early exit on convergence.
//
// Each thread computes Newton sqrt iteration v = 0.5f * (v + x/v) on
// its assigned element. After each iteration, the per-thread residual
// |v_new - v| is max-reduced across the 32-lane warp via butterfly
// shuffle (`__shfl_xor_sync`). When the warp-max falls below the
// epsilon threshold, all 32 lanes take the same early-exit branch.
//
// This is genuine warp-cooperative SIMT: threads exchange register
// values across lanes without going through shared memory, and the
// convergence decision is a warp-wide ballot rather than per-thread.
// Compare against newton_sqrt.ptx (per-thread, no warp primitives)
// in the SIMT bench group.
//
// Compile to PTX with nvcc (one-shot, regenerate when this source
// changes). The CUDA 13.1 nvcc no longer supports sm_70 (Volta), so
// the lowest viable arch is sm_75 (Turing); the driver JIT
// retargets the resulting PTX to whatever SM the live GPU is
// (sm_86 on RTX 3070, sm_89 on RTX 40 / L4, sm_90 on Hopper, etc.).
//
// On Windows with the BuildTools-only MSVC install, nvcc needs
// vcvars64.bat sourced first so cl.exe is on PATH for preprocessing:
//
//   cmd /c "call \"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && nvcc -ptx -arch=sm_75 -o newton_sqrt_warp.ptx newton_sqrt_warp.cu"
//
// The PTX header `.version` line MUST be edited down to `.version 7.0`
// after nvcc 13.1 emits it (nvcc 13.1 defaults to `.version 9.1`, which
// requires a driver from the R581+ series to JIT). The kernel body uses
// only PTX-6.0-era instructions (`shfl.sync.bfly.b32`, `ld.global.f32`,
// `setp.geu.f32`, etc.), so the version downgrade is a header-only edit
// that produces byte-identical SASS on every modern target arch
// (validated via `ptxas -arch=sm_86 -o cubin && cuobjdump --dump-sass`).
// PTX 7.0 ships with CUDA 11.0 and is loadable by every driver from
// R450 onward (mid-2020+); R580 datacenter drivers on Google Colab L4
// hosts will then JIT it cleanly.
//
// The resulting .ptx file is consumed by benches/flynn_axes.rs via
// include_str! and registered through `CudaBackend::register_kernel`,
// which feeds the driver's PTX JIT. No runtime NVRTC dependency.
//
extern "C" __global__
void newton_sqrt_warp(float* data, int n, int max_iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float x = data[tid];
    float v = x;
    const float eps = 1e-6f;

    for (int i = 0; i < max_iters; i++) {
        float v_new = 0.5f * (v + x / v);
        float residual = fabsf(v_new - v);
        v = v_new;

        // Warp-cooperative max-reduce via butterfly shuffle.
        // After this sequence, every lane holds the warp-max.
        residual = fmaxf(residual, __shfl_xor_sync(0xffffffff, residual, 16));
        residual = fmaxf(residual, __shfl_xor_sync(0xffffffff, residual, 8));
        residual = fmaxf(residual, __shfl_xor_sync(0xffffffff, residual, 4));
        residual = fmaxf(residual, __shfl_xor_sync(0xffffffff, residual, 2));
        residual = fmaxf(residual, __shfl_xor_sync(0xffffffff, residual, 1));

        // Every lane sees the same warp-max and takes the same branch
        // (no warp divergence at the early exit point).
        if (residual < eps) break;
    }

    data[tid] = v;
}

// House-owned f64 linear algebra for gpu_peer resident blocks.
//
// Source of record for kernels/linalg_f64.ptx, which the crate embeds
// (include_str!) and the driver JIT-compiles at runtime; consumers
// never need the CUDA toolkit. Regenerate after editing:
//
//   nvcc -ptx -arch=compute_75 linalg_f64.cu -o linalg_f64.ptx
//
// Argument convention matches GpuPeer::launch_wide: pointer arguments
// first, then u32 scalars. Every kernel grid-strides over its work
// items, so any grid size is correct. No standard headers: the file
// is also NVRTC-compilable header-less, like gpu_peer.cu.
//
// Accumulation contract: einsum and gemm accumulate with fma in
// ascending index order, and the CPU reference implementations in
// src/gpu_peer/linalg.rs use mul_add in the same order, so those two
// ops match the CPU bit for bit. The Jacobi eigen and SVD kernels
// rotate pairs in Brent-Luk tournament order (n/2 disjoint pairs per
// round) while the CPU reference rotates cyclically, so their results
// agree to rounding, not bit for bit.

typedef unsigned int u32;
typedef int i32;

// Largest square dimension the block-per-matrix kernels take: the
// working matrix lives in static shared memory (64 x 64 x 8 B = 32 KB,
// under the 48 KB default so no launch attribute is needed).
#define LINALG_MAX_N 64
#define LINALG_MAX_PAIRS (LINALG_MAX_N / 2)
// Largest dimension the one-thread-per-matrix kernels take: the matrix
// lives in per-thread local memory (16 x 16 x 8 B = 2 KB).
#define LINALG_THR_MAX_N 16
#define EINSUM_MAX_RANK 12
#define GEMM_TILE 16
// Block size every block-per-matrix kernel is launched with.
#define LINALG_BLOCK 256

// ------------------------------------------------------------ einsum

__device__ __forceinline__ int kind_which(int kind) { return (kind >> 16) & 0xFFFF; }
__device__ __forceinline__ int kind_slot(int kind) { return kind & 0xFFFF; }

// Flat offset into an operand from the preserved-index vector and the
// contracted-index vector, per the operand's per-axis kind table
// (which << 16 | slot; which 0 = preserved, 1 = contracted).
__device__ __forceinline__ int operand_offset(
    int rank, const int* __restrict__ strides, const int* __restrict__ kinds,
    const int* __restrict__ p_idx, const int* __restrict__ c_idx)
{
    int off = 0;
    for (int axis = 0; axis < rank; ++axis) {
        int k = kinds[axis];
        int idx = (kind_which(k) == 0) ? p_idx[kind_slot(k)] : c_idx[kind_slot(k)];
        off += idx * strides[axis];
    }
    return off;
}

// Generic per-element contraction, batched. One thread per output
// element per batch item: decode the preserved multi-index from the
// flat id, iterate the contracted multi-index, fma-accumulate.
// Covers matmul, outer product, n-d outer, axis sums, trace and
// transposed contractions from one kernel; the host builds the
// stride / kind tables. Batch item i reads a + i * a_batch_stride,
// b + i * b_batch_stride and writes out + i * o_batch_stride.
extern "C" __global__ void flynnel_einsum_f64(
    double* __restrict__ out,
    const double* __restrict__ a,
    const double* __restrict__ b,
    const int* __restrict__ a_strides,
    const int* __restrict__ b_strides,
    const int* __restrict__ o_strides,
    const int* __restrict__ c_extents,
    const int* __restrict__ a_kind,
    const int* __restrict__ b_kind,
    u32 o_size, u32 a_rank, u32 b_rank, u32 o_rank, u32 n_contract, u32 has_b,
    u32 batch, u32 a_batch_stride, u32 b_batch_stride, u32 o_batch_stride)
{
    u32 total = o_size * batch;
    for (u32 gid = blockIdx.x * blockDim.x + threadIdx.x; gid < total;
         gid += gridDim.x * blockDim.x) {
        u32 bi = gid / o_size;
        u32 tid = gid - bi * o_size;
        const double* ab = a + (unsigned long long)bi * a_batch_stride;
        const double* bb = b + (unsigned long long)bi * b_batch_stride;
        double* ob = out + (unsigned long long)bi * o_batch_stride;

        int p_idx[EINSUM_MAX_RANK];
        int c_idx[EINSUM_MAX_RANK];
        for (int z = 0; z < EINSUM_MAX_RANK; ++z) { p_idx[z] = 0; c_idx[z] = 0; }

        int rem = (int)tid;
        for (u32 axis = 0; axis < o_rank; ++axis) {
            int s = o_strides[axis];
            int v = (s == 0) ? 0 : (rem / s);
            p_idx[axis] = v;
            rem -= v * s;
        }

        int c_total = 1;
        for (u32 i = 0; i < n_contract; ++i) c_total *= c_extents[i];

        double acc = 0.0;
        for (int c_flat = 0; c_flat < c_total; ++c_flat) {
            int r = c_flat;
            for (int j = (int)n_contract - 1; j >= 0; --j) {
                int e = c_extents[j];
                int v = (e <= 1) ? 0 : (r % e);
                c_idx[j] = v;
                if (e > 1) r /= e;
            }
            int a_off = operand_offset((int)a_rank, a_strides, a_kind, p_idx, c_idx);
            if (has_b != 0u) {
                int b_off = operand_offset((int)b_rank, b_strides, b_kind, p_idx, c_idx);
                acc = fma(ab[a_off], bb[b_off], acc);
            } else {
                acc += ab[a_off];
            }
        }
        ob[tid] = acc;
    }
}

// ------------------------------------------------------------ gemm

// Batched row-major C = A * B with 16 x 16 shared-memory tiles.
// Launch with LINALG_BLOCK threads: thread t owns tile cell
// (t / 16, t % 16). Batch item i is at a + i * m * lda,
// b + i * k * ldb, c + i * m * ldc. Each tile iterates k ascending
// with fma, the CPU reference's order.
extern "C" __global__ void flynnel_gemm_batched_f64(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ c,
    u32 batch, u32 m, u32 n, u32 k, u32 lda, u32 ldb, u32 ldc)
{
    __shared__ double as[GEMM_TILE][GEMM_TILE + 1];
    __shared__ double bs[GEMM_TILE][GEMM_TILE + 1];
    u32 tiles_n = (n + GEMM_TILE - 1) / GEMM_TILE;
    u32 tiles_m = (m + GEMM_TILE - 1) / GEMM_TILE;
    u32 tiles_per = tiles_m * tiles_n;
    u32 total_blocks = tiles_per * batch;
    u32 tx = threadIdx.x % GEMM_TILE;
    u32 ty = threadIdx.x / GEMM_TILE;
    for (u32 blk = blockIdx.x; blk < total_blocks; blk += gridDim.x) {
        u32 bi = blk / tiles_per;
        u32 t = blk - bi * tiles_per;
        u32 tm = t / tiles_n;
        u32 tn = t - tm * tiles_n;
        const double* ab = a + (unsigned long long)bi * m * lda;
        const double* bb = b + (unsigned long long)bi * k * ldb;
        double* cb = c + (unsigned long long)bi * m * ldc;
        u32 row = tm * GEMM_TILE + ty;
        u32 col = tn * GEMM_TILE + tx;
        double acc = 0.0;
        for (u32 k0 = 0; k0 < k; k0 += GEMM_TILE) {
            u32 ka = k0 + tx;
            u32 kb = k0 + ty;
            as[ty][tx] = (row < m && ka < k) ? ab[row * lda + ka] : 0.0;
            bs[ty][tx] = (kb < k && col < n) ? bb[kb * ldb + col] : 0.0;
            __syncthreads();
            u32 kk_end = (k - k0 < GEMM_TILE) ? (k - k0) : GEMM_TILE;
            for (u32 kk = 0; kk < kk_end; ++kk) {
                acc = fma(as[ty][kk], bs[kk][tx], acc);
            }
            __syncthreads();
        }
        if (row < m && col < n) cb[row * ldc + col] = acc;
    }
}

// ------------------------------------------------------------ jacobi helpers

// Two-sided Jacobi rotation for the symmetric 2 x 2 block
// (s_pp, s_pq; s_pq, s_qq): the CPU reference's formula. Writes
// c and s of the rotation that annihilates s_pq; identity when s_pq
// is negligible.
__device__ __forceinline__ void jacobi_rotation(
    double s_pp, double s_qq, double s_pq, double* c_out, double* s_out)
{
    if (fabs(s_pq) < 1e-40) { *c_out = 1.0; *s_out = 0.0; return; }
    double theta_num = s_qq - s_pp;
    double theta_den = 2.0 * s_pq;
    double t;
    if (fabs(theta_den) < 1e-40 * fabs(theta_num)) {
        t = 0.0;
    } else {
        double theta = theta_num / theta_den;
        double disc = sqrt(1.0 + theta * theta);
        t = (theta >= 0.0) ? 1.0 / (theta + disc) : 1.0 / (theta - disc);
    }
    double c = 1.0 / sqrt(1.0 + t * t);
    *c_out = c;
    *s_out = c * t;
}

// One-sided Jacobi rotation for a column pair with Gram entries
// aa = |u_p|^2, bb = |u_q|^2, cc = u_p . u_q (LAPACK xJACOBI form).
__device__ __forceinline__ void onesided_rotation(
    double aa, double bb, double cc, double* c_out, double* s_out)
{
    double zeta = (bb - aa) / (2.0 * cc);
    double t = 1.0 / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
    if (zeta < 0.0) t = -t;
    double c = 1.0 / sqrt(1.0 + t * t);
    *c_out = c;
    *s_out = t * c;
}

// Block-wide sum of `v` into `red[0]`; every thread contributes,
// LINALG_BLOCK threads. Leaves red[0] valid after the trailing
// barrier.
__device__ __forceinline__ void block_sum(double v, double* red)
{
    red[threadIdx.x] = v;
    __syncthreads();
    for (u32 stride = LINALG_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
}

// Advance the Brent-Luk tournament: top/bot hold `half` pairs; after
// the step every index has met a new partner, and nn - 1 steps visit
// every pair once. Cooperative: called by all threads, barrier
// inside. `half` must be at least 2.
__device__ __forceinline__ void tournament_step(int* top, int* bot, int half)
{
    int t_prev = 0, b_next = 0;
    int i = (int)threadIdx.x;
    bool active = i < half;
    if (active) {
        t_prev = (i == 0) ? top[0] : ((i == 1) ? bot[0] : top[i - 1]);
        b_next = (i == half - 1) ? top[half - 1] : bot[i + 1];
    }
    __syncthreads();
    if (active) { top[i] = t_prev; bot[i] = b_next; }
    __syncthreads();
}

// ------------------------------------------------------------ syev (block per matrix)

// Batched symmetric eigendecomposition, one block of LINALG_BLOCK
// threads per matrix, n <= LINALG_MAX_N. Matrix i is n x n row-major
// at a_batch + i * n * n; eigenvalues (unsorted, diagonal order)
// land at w_batch + i * n; with want_v != 0 the eigenvectors (as
// columns) land at v_batch + i * n * n. Each sweep applies n - 1
// tournament rounds of n / 2 disjoint rotations: rotations are
// computed from the current matrix, columns are rotated, then rows,
// then the pair entries are re-symmetrized, matching the CPU
// reference's per-rotation update. Sweeps stop when the off-diagonal
// Frobenius mass reaches zero or stops decreasing, or at max_sweeps.
extern "C" __global__ void flynnel_syev_jacobi_f64_blk(
    const double* __restrict__ a_batch,
    double* __restrict__ w_batch,
    double* __restrict__ v_batch,
    u32 batch, u32 n, u32 max_sweeps, u32 want_v)
{
    __shared__ double s[LINALG_MAX_N * LINALG_MAX_N];
    __shared__ double red[LINALG_BLOCK];
    __shared__ int top[LINALG_MAX_PAIRS];
    __shared__ int bot[LINALG_MAX_PAIRS];
    __shared__ int pp[LINALG_MAX_PAIRS];
    __shared__ int qq[LINALG_MAX_PAIRS];
    __shared__ double rc[LINALG_MAX_PAIRS];
    __shared__ double rs[LINALG_MAX_PAIRS];
    __shared__ double prev_off;
    __shared__ int stop;

    if (n == 0u || n > LINALG_MAX_N) return;
    u32 tid = threadIdx.x;
    u32 nn = (n + 1u) & ~1u;
    int half = (int)(nn / 2u);
    u32 n2 = n * n;

    for (u32 bi = blockIdx.x; bi < batch; bi += gridDim.x) {
        const double* a = a_batch + (unsigned long long)bi * n2;
        double* w = w_batch + (unsigned long long)bi * n;
        double* v = v_batch + (unsigned long long)bi * n2;

        for (u32 idx = tid; idx < n2; idx += LINALG_BLOCK) {
            u32 i = idx / n, j = idx - i * n;
            s[idx] = 0.5 * (a[i * n + j] + a[j * n + i]);
            if (want_v != 0u) v[idx] = (i == j) ? 1.0 : 0.0;
        }
        if ((int)tid < half) { top[tid] = (int)tid; bot[tid] = (int)nn - 1 - (int)tid; }
        if (tid == 0u) { prev_off = -1.0; stop = 0; }
        __syncthreads();

        if (n == 1u) {
            if (tid == 0u) w[0] = s[0];
            __syncthreads();
            continue;
        }

        for (u32 sweep = 0; sweep < max_sweeps; ++sweep) {
            double local = 0.0;
            for (u32 idx = tid; idx < n2; idx += LINALG_BLOCK) {
                u32 i = idx / n, j = idx - i * n;
                if (i < j) local = fma(s[idx], s[idx], local);
            }
            block_sum(local, red);
            if (tid == 0u) {
                double off = red[0];
                stop = (off == 0.0 || (prev_off >= 0.0 && off >= prev_off)) ? 1 : 0;
                prev_off = off;
            }
            __syncthreads();
            if (stop) break;

            for (u32 round = 0; round + 1u < nn; ++round) {
                if ((int)tid < half) {
                    int p = top[tid], q = bot[tid];
                    if (p > q) { int t = p; p = q; q = t; }
                    pp[tid] = p; qq[tid] = q;
                    if (q >= (int)n) {
                        rc[tid] = 1.0; rs[tid] = 0.0;
                    } else {
                        jacobi_rotation(s[p * n + p], s[q * n + q], s[p * n + q],
                                        &rc[tid], &rs[tid]);
                    }
                }
                __syncthreads();
                // Columns p, q of every row k.
                for (u32 item = tid; item < (u32)half * n; item += LINALG_BLOCK) {
                    u32 pr = item / n, k = item - pr * n;
                    int p = pp[pr], q = qq[pr];
                    if (q >= (int)n) continue;
                    double c = rc[pr], sn = rs[pr];
                    double s_kp = s[k * n + p], s_kq = s[k * n + q];
                    s[k * n + p] = c * s_kp - sn * s_kq;
                    s[k * n + q] = sn * s_kp + c * s_kq;
                }
                __syncthreads();
                // Rows p, q of every column k.
                for (u32 item = tid; item < (u32)half * n; item += LINALG_BLOCK) {
                    u32 pr = item / n, k = item - pr * n;
                    int p = pp[pr], q = qq[pr];
                    if (q >= (int)n) continue;
                    double c = rc[pr], sn = rs[pr];
                    double s_pk = s[p * n + k], s_qk = s[q * n + k];
                    s[p * n + k] = c * s_pk - sn * s_qk;
                    s[q * n + k] = sn * s_pk + c * s_qk;
                }
                __syncthreads();
                if ((int)tid < half) {
                    int p = pp[tid], q = qq[tid];
                    if (q < (int)n) {
                        double avg = 0.5 * (s[p * n + q] + s[q * n + p]);
                        s[p * n + q] = avg;
                        s[q * n + p] = avg;
                    }
                }
                if (want_v != 0u) {
                    for (u32 item = tid; item < (u32)half * n; item += LINALG_BLOCK) {
                        u32 pr = item / n, k = item - pr * n;
                        int p = pp[pr], q = qq[pr];
                        if (q >= (int)n) continue;
                        double c = rc[pr], sn = rs[pr];
                        double v_kp = v[k * n + p], v_kq = v[k * n + q];
                        v[k * n + p] = c * v_kp - sn * v_kq;
                        v[k * n + q] = sn * v_kp + c * v_kq;
                    }
                }
                __syncthreads();
                if (half >= 2) tournament_step(top, bot, half);
            }
        }
        for (u32 i = tid; i < n; i += LINALG_BLOCK) w[i] = s[i * n + i];
        __syncthreads();
    }
}

// ------------------------------------------------------------ syev (thread per matrix)

// Batched symmetric eigendecomposition, one THREAD per matrix,
// n <= LINALG_THR_MAX_N: the CPU reference's cyclic sweep
// transliterated, matrix in local memory. Same layout and outputs
// as the block variant. Matrices with n above the limit are left
// untouched; the host routes those to the block kernel.
extern "C" __global__ void flynnel_syev_jacobi_f64_thr(
    const double* __restrict__ a_batch,
    double* __restrict__ w_batch,
    double* __restrict__ v_batch,
    u32 batch, u32 n, u32 max_sweeps, u32 want_v)
{
    if (n == 0u || n > LINALG_THR_MAX_N) return;
    u32 n2 = n * n;
    for (u32 bi = blockIdx.x * blockDim.x + threadIdx.x; bi < batch;
         bi += gridDim.x * blockDim.x) {
        double s[LINALG_THR_MAX_N * LINALG_THR_MAX_N];
        double vv[LINALG_THR_MAX_N * LINALG_THR_MAX_N];
        const double* a = a_batch + (unsigned long long)bi * n2;
        double* w = w_batch + (unsigned long long)bi * n;
        for (u32 i = 0; i < n; ++i) {
            for (u32 j = 0; j < n; ++j) {
                s[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
                vv[i * n + j] = (i == j) ? 1.0 : 0.0;
            }
        }
        double prev_off = -1.0;
        for (u32 sweep = 0; sweep < max_sweeps; ++sweep) {
            double off = 0.0;
            for (u32 p = 0; p < n; ++p)
                for (u32 q = p + 1; q < n; ++q)
                    off = fma(s[p * n + q], s[p * n + q], off);
            if (off == 0.0) break;
            if (prev_off >= 0.0 && off >= prev_off) break;
            prev_off = off;
            for (u32 p = 0; p + 1 < n; ++p) {
                for (u32 q = p + 1; q < n; ++q) {
                    double c, sn;
                    jacobi_rotation(s[p * n + p], s[q * n + q], s[p * n + q], &c, &sn);
                    if (sn == 0.0 && c == 1.0) continue;
                    for (u32 k = 0; k < n; ++k) {
                        double s_kp = s[k * n + p], s_kq = s[k * n + q];
                        s[k * n + p] = c * s_kp - sn * s_kq;
                        s[k * n + q] = sn * s_kp + c * s_kq;
                    }
                    for (u32 k = 0; k < n; ++k) {
                        double s_pk = s[p * n + k], s_qk = s[q * n + k];
                        s[p * n + k] = c * s_pk - sn * s_qk;
                        s[q * n + k] = sn * s_pk + c * s_qk;
                    }
                    double avg = 0.5 * (s[p * n + q] + s[q * n + p]);
                    s[p * n + q] = avg;
                    s[q * n + p] = avg;
                    if (want_v != 0u) {
                        for (u32 k = 0; k < n; ++k) {
                            double v_kp = vv[k * n + p], v_kq = vv[k * n + q];
                            vv[k * n + p] = c * v_kp - sn * v_kq;
                            vv[k * n + q] = sn * v_kp + c * v_kq;
                        }
                    }
                }
            }
        }
        for (u32 i = 0; i < n; ++i) w[i] = s[i * n + i];
        if (want_v != 0u) {
            double* v = v_batch + (unsigned long long)bi * n2;
            for (u32 idx = 0; idx < n2; ++idx) v[idx] = vv[idx];
        }
    }
}

// ------------------------------------------------------------ gesvd (block per matrix)

// Batched one-sided Jacobi SVD (Drmac-Veselic style), one block of
// LINALG_BLOCK threads per matrix, m >= n, m and n <= LINALG_MAX_N.
// Matrix i is m x n row-major at a_batch + i * m * n and is
// OVERWRITTEN with U (orthonormal columns); singular values (unsorted,
// column order) land at sigma_batch + i * n; with want_v != 0, V
// (n x n) lands at v_batch + i * n * n. Column pairs rotate in
// tournament order; a sweep with no rotation applied ends the loop.
// A pair rotates when |u_p . u_q| / (|u_p| |u_q|) >= 2^-52, the f64
// unit roundoff.
extern "C" __global__ void flynnel_gesvd_jacobi_f64_blk(
    double* __restrict__ a_batch,
    double* __restrict__ sigma_batch,
    double* __restrict__ v_batch,
    u32 batch, u32 m, u32 n, u32 max_sweeps, u32 want_v)
{
    __shared__ double u[LINALG_MAX_N * LINALG_MAX_N];
    __shared__ int top[LINALG_MAX_PAIRS];
    __shared__ int bot[LINALG_MAX_PAIRS];
    __shared__ int pp[LINALG_MAX_PAIRS];
    __shared__ int qq[LINALG_MAX_PAIRS];
    __shared__ double rc[LINALG_MAX_PAIRS];
    __shared__ double rs[LINALG_MAX_PAIRS];
    __shared__ int rotated[LINALG_MAX_PAIRS];
    __shared__ int any_rotated;

    if (n == 0u || m == 0u || n > LINALG_MAX_N || m > LINALG_MAX_N || m < n) return;
    const double tol = 2.220446049250313e-16;
    u32 tid = threadIdx.x;
    u32 nn = (n + 1u) & ~1u;
    int half = (int)(nn / 2u);
    u32 mn = m * n;

    for (u32 bi = blockIdx.x; bi < batch; bi += gridDim.x) {
        double* a = a_batch + (unsigned long long)bi * mn;
        double* sigma = sigma_batch + (unsigned long long)bi * n;
        double* v = v_batch + (unsigned long long)bi * n * n;

        for (u32 idx = tid; idx < mn; idx += LINALG_BLOCK) u[idx] = a[idx];
        if (want_v != 0u) {
            for (u32 idx = tid; idx < n * n; idx += LINALG_BLOCK) {
                u32 i = idx / n, j = idx - i * n;
                v[idx] = (i == j) ? 1.0 : 0.0;
            }
        }
        if ((int)tid < half) { top[tid] = (int)tid; bot[tid] = (int)nn - 1 - (int)tid; }
        __syncthreads();

        if (n >= 2u) {
            for (u32 sweep = 0; sweep < max_sweeps; ++sweep) {
                if (tid == 0u) any_rotated = 0;
                __syncthreads();
                for (u32 round = 0; round + 1u < nn; ++round) {
                    if ((int)tid < half) {
                        int p = top[tid], q = bot[tid];
                        if (p > q) { int t = p; p = q; q = t; }
                        pp[tid] = p; qq[tid] = q;
                        int rot = 0;
                        double c = 1.0, sn = 0.0;
                        if (q < (int)n) {
                            double aa = 0.0, bb = 0.0, cc = 0.0;
                            for (u32 i = 0; i < m; ++i) {
                                double up = u[i * n + p], uq = u[i * n + q];
                                aa = fma(up, up, aa);
                                bb = fma(uq, uq, bb);
                                cc = fma(up, uq, cc);
                            }
                            double denom = sqrt(aa * bb);
                            if (denom != 0.0 && fabs(cc) / denom >= tol) {
                                onesided_rotation(aa, bb, cc, &c, &sn);
                                rot = 1;
                            }
                        }
                        rc[tid] = c; rs[tid] = sn; rotated[tid] = rot;
                        if (rot) any_rotated = 1;
                    }
                    __syncthreads();
                    for (u32 item = tid; item < (u32)half * m; item += LINALG_BLOCK) {
                        u32 pr = item / m, i = item - pr * m;
                        if (!rotated[pr]) continue;
                        int p = pp[pr], q = qq[pr];
                        double c = rc[pr], sn = rs[pr];
                        double up = u[i * n + p], uq = u[i * n + q];
                        u[i * n + p] = c * up - sn * uq;
                        u[i * n + q] = sn * up + c * uq;
                    }
                    if (want_v != 0u) {
                        for (u32 item = tid; item < (u32)half * n; item += LINALG_BLOCK) {
                            u32 pr = item / n, i = item - pr * n;
                            if (!rotated[pr]) continue;
                            int p = pp[pr], q = qq[pr];
                            double c = rc[pr], sn = rs[pr];
                            double vp = v[i * n + p], vq = v[i * n + q];
                            v[i * n + p] = c * vp - sn * vq;
                            v[i * n + q] = sn * vp + c * vq;
                        }
                    }
                    __syncthreads();
                    if (half >= 2) tournament_step(top, bot, half);
                }
                if (!any_rotated) break;
                __syncthreads();
            }
        }
        // Column norms are the singular values; normalized columns are U.
        for (u32 j = tid; j < n; j += LINALG_BLOCK) {
            double ss = 0.0;
            for (u32 i = 0; i < m; ++i) ss = fma(u[i * n + j], u[i * n + j], ss);
            double sj = sqrt(ss);
            sigma[j] = sj;
            if (sj > 0.0) {
                for (u32 i = 0; i < m; ++i) u[i * n + j] = u[i * n + j] / sj;
            }
        }
        __syncthreads();
        for (u32 idx = tid; idx < mn; idx += LINALG_BLOCK) a[idx] = u[idx];
        __syncthreads();
    }
}

// ------------------------------------------------------------ gesvd (thread per matrix)

// Batched one-sided Jacobi SVD, one THREAD per matrix, m and n <=
// LINALG_THR_MAX_N: the CPU reference's cyclic sweep transliterated,
// matrix in local memory. Same layout and outputs as the block
// variant. Larger matrices are left untouched; the host routes those
// to the block kernel.
extern "C" __global__ void flynnel_gesvd_jacobi_f64_thr(
    double* __restrict__ a_batch,
    double* __restrict__ sigma_batch,
    double* __restrict__ v_batch,
    u32 batch, u32 m, u32 n, u32 max_sweeps, u32 want_v)
{
    if (n == 0u || m == 0u || n > LINALG_THR_MAX_N || m > LINALG_THR_MAX_N || m < n) return;
    const double tol = 2.220446049250313e-16;
    u32 mn = m * n;
    for (u32 bi = blockIdx.x * blockDim.x + threadIdx.x; bi < batch;
         bi += gridDim.x * blockDim.x) {
        double u[LINALG_THR_MAX_N * LINALG_THR_MAX_N];
        double vv[LINALG_THR_MAX_N * LINALG_THR_MAX_N];
        double* a = a_batch + (unsigned long long)bi * mn;
        double* sigma = sigma_batch + (unsigned long long)bi * n;
        for (u32 idx = 0; idx < mn; ++idx) u[idx] = a[idx];
        for (u32 i = 0; i < n; ++i)
            for (u32 j = 0; j < n; ++j) vv[i * n + j] = (i == j) ? 1.0 : 0.0;
        for (u32 sweep = 0; sweep < max_sweeps; ++sweep) {
            int converged = 1;
            for (u32 p = 0; p + 1 < n; ++p) {
                for (u32 q = p + 1; q < n; ++q) {
                    double aa = 0.0, bb = 0.0, cc = 0.0;
                    for (u32 i = 0; i < m; ++i) {
                        double up = u[i * n + p], uq = u[i * n + q];
                        aa = fma(up, up, aa);
                        bb = fma(uq, uq, bb);
                        cc = fma(up, uq, cc);
                    }
                    double denom = sqrt(aa * bb);
                    if (denom == 0.0) continue;
                    if (fabs(cc) / denom < tol) continue;
                    converged = 0;
                    double c, sn;
                    onesided_rotation(aa, bb, cc, &c, &sn);
                    for (u32 i = 0; i < m; ++i) {
                        double up = u[i * n + p], uq = u[i * n + q];
                        u[i * n + p] = c * up - sn * uq;
                        u[i * n + q] = sn * up + c * uq;
                    }
                    if (want_v != 0u) {
                        for (u32 i = 0; i < n; ++i) {
                            double vp = vv[i * n + p], vq = vv[i * n + q];
                            vv[i * n + p] = c * vp - sn * vq;
                            vv[i * n + q] = sn * vp + c * vq;
                        }
                    }
                }
            }
            if (converged) break;
        }
        for (u32 j = 0; j < n; ++j) {
            double ss = 0.0;
            for (u32 i = 0; i < m; ++i) ss = fma(u[i * n + j], u[i * n + j], ss);
            double sj = sqrt(ss);
            sigma[j] = sj;
            if (sj > 0.0)
                for (u32 i = 0; i < m; ++i) u[i * n + j] = u[i * n + j] / sj;
        }
        for (u32 idx = 0; idx < mn; ++idx) a[idx] = u[idx];
        if (want_v != 0u) {
            double* v = v_batch + (unsigned long long)bi * n * n;
            for (u32 idx = 0; idx < n * n; ++idx) v[idx] = vv[idx];
        }
    }
}

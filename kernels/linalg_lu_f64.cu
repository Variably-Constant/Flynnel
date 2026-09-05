// Batched LU factorization with partial pivoting, solve and inverse
// for n <= LINALG_MAX_N, one block per matrix. The factorization
// keeps the matrix in shared memory: each step finds the pivot by a
// block argmax, swaps rows, forms the multipliers and applies the
// rank-1 update with every thread owning a strip of the trailing
// block. The solve keeps the right-hand sides in shared memory and
// reads the packed factor from global memory, one column of L or one
// row of U per step. Every update is an explicit fused multiply-add
// in the same order as the host reference, so the two agree bit for
// bit. Same launch contract as linalg_f64.cu: pointers first, then
// u32 scalars.

typedef unsigned int u32;

#define LINALG_MAX_N 64
#define LINALG_BLOCK 256

// Argmax of |val| over the block, ties to the lowest index; `red`
// and `idx` are LINALG_BLOCK entries each, reusable after return.
__device__ __forceinline__ int block_argmax(double val, int index, double* red, int* idx)
{
    red[threadIdx.x] = val;
    idx[threadIdx.x] = index;
    __syncthreads();
    for (u32 stride = LINALG_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const double o = red[threadIdx.x + stride];
            const int oi = idx[threadIdx.x + stride];
            const double m = red[threadIdx.x];
            const int mi = idx[threadIdx.x];
            if (o > m || (o == m && oi < mi)) {
                red[threadIdx.x] = o;
                idx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    const int r = idx[0];
    __syncthreads();
    return r;
}

// In-place LU with partial pivoting of `batch` row-major n x n
// matrices: on return `a` holds U on and above the diagonal and the
// unit-lower multipliers below it, `piv[item * n + k]` the row
// swapped with row k at step k, and `info[item]` zero or one past the
// first step whose pivot was exactly zero (the factorization
// continues with that column left as is).
extern "C" __global__ void flynnel_getrf_f64_blk(
    double* __restrict__ a, int* __restrict__ piv, int* __restrict__ info, u32 batch, u32 n)
{
    __shared__ double s[LINALG_MAX_N * LINALG_MAX_N];
    __shared__ double red[LINALG_BLOCK];
    __shared__ int idx[LINALG_BLOCK];
    __shared__ int first_zero;
    const u32 item = blockIdx.x;
    if (item >= batch) return;
    const u32 t = threadIdx.x;
    const u32 nn = n * n;
    double* g = a + (size_t)item * nn;
    for (u32 i = t; i < nn; i += LINALG_BLOCK) s[i] = g[i];
    if (t == 0) first_zero = 0;
    __syncthreads();

    for (u32 k = 0; k < n; ++k) {
        const double cand = (t + k < n) ? fabs(s[(t + k) * n + k]) : -1.0;
        const int p = block_argmax(cand, (int)(t + k), red, idx);
        if (t == 0) {
            piv[item * n + k] = p;
            if (s[(u32)p * n + k] == 0.0 && first_zero == 0) first_zero = (int)k + 1;
        }
        if ((u32)p != k) {
            for (u32 j = t; j < n; j += LINALG_BLOCK) {
                const double tmp = s[k * n + j];
                s[k * n + j] = s[(u32)p * n + j];
                s[(u32)p * n + j] = tmp;
            }
        }
        __syncthreads();
        const double pivot = s[k * n + k];
        if (pivot != 0.0) {
            for (u32 i = k + 1 + t; i < n; i += LINALG_BLOCK) s[i * n + k] /= pivot;
        }
        __syncthreads();
        if (pivot != 0.0) {
            const u32 rem = n - k - 1;
            for (u32 e = t; e < rem * rem; e += LINALG_BLOCK) {
                const u32 i = k + 1 + e / rem;
                const u32 j = k + 1 + e % rem;
                s[i * n + j] = fma(-s[i * n + k], s[k * n + j], s[i * n + j]);
            }
        }
        __syncthreads();
    }

    for (u32 i = t; i < nn; i += LINALG_BLOCK) g[i] = s[i];
    if (t == 0) info[item] = first_zero;
}

// Solve with the packed factor of flynnel_getrf_f64_blk: `b` holds
// `batch` row-major n x nrhs right-hand sides and receives the
// solutions. With `identity_rhs` set, `b` is not read: the
// right-hand side is the identity (nrhs must be n) and `b` receives
// the inverse. A zero pivot leaves infinities or NaNs in the
// affected rows, as the host reference does.
extern "C" __global__ void flynnel_getrs_f64_blk(
    const double* __restrict__ lu, const int* __restrict__ piv, double* __restrict__ b,
    u32 batch, u32 n, u32 nrhs, u32 identity_rhs)
{
    __shared__ double x[LINALG_MAX_N * LINALG_MAX_N];
    const u32 item = blockIdx.x;
    if (item >= batch) return;
    const u32 t = threadIdx.x;
    const u32 nr = n * nrhs;
    const double* f = lu + (size_t)item * n * n;
    const int* pv = piv + (size_t)item * n;
    double* g = b + (size_t)item * nr;
    if (identity_rhs) {
        for (u32 e = t; e < nr; e += LINALG_BLOCK) x[e] = (e / nrhs == e % nrhs) ? 1.0 : 0.0;
    } else {
        for (u32 e = t; e < nr; e += LINALG_BLOCK) x[e] = g[e];
    }
    __syncthreads();

    // Row interchanges in factorization order.
    for (u32 k = 0; k < n; ++k) {
        const u32 p = (u32)pv[k];
        if (p != k) {
            for (u32 j = t; j < nrhs; j += LINALG_BLOCK) {
                const double tmp = x[k * nrhs + j];
                x[k * nrhs + j] = x[p * nrhs + j];
                x[p * nrhs + j] = tmp;
            }
        }
        __syncthreads();
    }
    // Forward substitution with unit L.
    for (u32 k = 0; k + 1 < n; ++k) {
        const u32 rows = n - k - 1;
        for (u32 e = t; e < rows * nrhs; e += LINALG_BLOCK) {
            const u32 i = k + 1 + e / nrhs;
            const u32 j = e % nrhs;
            x[i * nrhs + j] = fma(-f[i * n + k], x[k * nrhs + j], x[i * nrhs + j]);
        }
        __syncthreads();
    }
    // Back substitution with U.
    for (u32 kk = 0; kk < n; ++kk) {
        const u32 k = n - 1 - kk;
        const double d = f[k * n + k];
        for (u32 j = t; j < nrhs; j += LINALG_BLOCK) x[k * nrhs + j] /= d;
        __syncthreads();
        for (u32 e = t; e < k * nrhs; e += LINALG_BLOCK) {
            const u32 i = e / nrhs;
            const u32 j = e % nrhs;
            x[i * nrhs + j] = fma(-f[i * n + k], x[k * nrhs + j], x[i * nrhs + j]);
        }
        __syncthreads();
    }
    for (u32 e = t; e < nr; e += LINALG_BLOCK) g[e] = x[e];
}

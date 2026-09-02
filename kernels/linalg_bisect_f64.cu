// Batched symmetric eigendecomposition and SVD for n <= LINALG_MAX_N,
// one block per matrix: Householder reduction to tridiagonal
// (EISPACK tred2) or bidiagonal (LINPACK dsvdc) form runs over the
// block; the eigenvalues of the reduced matrix are then found by
// bisection with Sturm counts, one thread per eigenvalue, and the
// vectors by inverse iteration with cluster orthogonalization, one
// thread per vector. Singular values are the positive eigenvalues of
// the Golub-Kahan tridiagonal of the bidiagonal. No phase runs a long
// dependent f64 recurrence on a single thread, which a GeForce part
// issues at 1/64 rate. Companion module to linalg_f64.cu with the
// same launch contract: pointers first, then u32 scalars.

#include <float.h>
#include <math.h>

typedef unsigned int u32;

#define LINALG_MAX_N 64
#define LINALG_BLOCK 256
#define GK_MAX (2 * LINALG_MAX_N)

// Block-wide sum of `v`, returned to every thread; LINALG_BLOCK
// threads, `red` reusable after return.
__device__ __forceinline__ double block_total(double v, double* red)
{
    red[threadIdx.x] = v;
    __syncthreads();
    for (u32 stride = LINALG_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const double r = red[0];
    __syncthreads();
    return r;
}

// ------------------------------------------------------------ tridiagonal eigen tools
// A symmetric tridiagonal is (diag[0..n), off[1..n)); off[i] couples
// i - 1 and i.

// Eigenvalues below x, by the Sturm sequence with the pivot floor
// `pivmin`.
__device__ __forceinline__ int sturm_count(
    const double* diag, const double* off, int n, double x, double pivmin)
{
    int c = 0;
    double q = diag[0] - x;
    if (q < 0.0) ++c;
    if (fabs(q) < pivmin) q = -pivmin;
    for (int i = 1; i < n; ++i) {
        q = diag[i] - x - off[i] * off[i] / q;
        if (q < 0.0) ++c;
        if (fabs(q) < pivmin) q = -pivmin;
    }
    return c;
}

// The k-th smallest eigenvalue (k from 0) by bisection of [lo, hi].
__device__ __forceinline__ double bisect_kth(
    const double* diag, const double* off, int n, int k, double lo, double hi, double pivmin)
{
    for (int it = 0; it < 256; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi) break;
        if (sturm_count(diag, off, n, mid, pivmin) > k) hi = mid; else lo = mid;
        if (hi - lo <= 2.0 * DBL_EPSILON * fmax(fabs(lo), fabs(hi)) + pivmin) break;
    }
    return 0.5 * (lo + hi);
}

// Gershgorin bounds of the tridiagonal.
__device__ __forceinline__ void gershgorin(
    const double* diag, const double* off, int n, double* lo, double* hi)
{
    double l = DBL_MAX, h = -DBL_MAX;
    for (int i = 0; i < n; ++i) {
        const double r = (i > 0 ? fabs(off[i]) : 0.0) + (i + 1 < n ? fabs(off[i + 1]) : 0.0);
        l = fmin(l, diag[i] - r);
        h = fmax(h, diag[i] + r);
    }
    const double pad = DBL_EPSILON * fmax(fabs(l), fabs(h)) * (double)n + DBL_MIN;
    *lo = l - pad;
    *hi = h + pad;
}

// Inverse iteration on (T - lam I) x = b for the tridiagonal, three
// passes through an LU factorization with partial pivoting (LAPACK
// dgttrf / dgttrs). `x` receives the unit vector; `w` is 6 n doubles
// of thread-private scratch. Returns the norm of (T - lam I) x after
// the last pass, for diagnostics.
__device__ void tridiag_inverse_iteration(
    const double* diag, const double* off, int n, double lam, double pivmin,
    u32 salt, double* x, double* w)
{
    double* dl = w;               // multipliers / sub-diagonal
    double* dd = w + n;           // pivots
    double* du = w + 2 * n;       // first super-diagonal
    double* du2 = w + 3 * n;      // second super-diagonal (fill-in)
    double* b = w + 4 * n;
    double* swapped = w + 5 * n;  // 1.0 where rows i and i + 1 were swapped
    for (int i = 0; i < n; ++i) {
        dd[i] = diag[i] - lam;
        dl[i] = (i > 0) ? off[i] : 0.0;
        du[i] = (i + 1 < n) ? off[i + 1] : 0.0;
        du2[i] = 0.0;
        swapped[i] = 0.0;
    }
    for (int i = 0; i + 1 < n; ++i) {
        if (fabs(dd[i]) >= fabs(dl[i + 1])) {
            if (fabs(dd[i]) < pivmin) dd[i] = pivmin;
            const double fact = dl[i + 1] / dd[i];
            dl[i + 1] = fact;
            dd[i + 1] -= fact * du[i];
        } else {
            const double fact = dd[i] / dl[i + 1];
            dd[i] = dl[i + 1];
            dl[i + 1] = fact;
            const double temp = du[i];
            du[i] = dd[i + 1];
            dd[i + 1] = temp - fact * dd[i + 1];
            if (i + 2 < n) {
                du2[i] = du[i + 1];
                du[i + 1] = -fact * du[i + 1];
            }
            swapped[i] = 1.0;
        }
    }
    if (fabs(dd[n - 1]) < pivmin) dd[n - 1] = pivmin;
    // Starting vector: ones with a deterministic per-vector ripple so
    // no eigenvector is orthogonal to it.
    for (int i = 0; i < n; ++i) {
        u32 h = (u32)i * 2654435761u ^ (salt * 40503u + 12345u);
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        x[i] = 1.0 + 0.25 * ((double)(h & 0xFFFFu) / 65535.0 - 0.5);
    }
    for (int pass = 0; pass < 3; ++pass) {
        for (int i = 0; i < n; ++i) b[i] = x[i];
        for (int i = 0; i + 1 < n; ++i) {
            if (swapped[i] == 0.0) {
                b[i + 1] -= dl[i + 1] * b[i];
            } else {
                const double temp = b[i];
                b[i] = b[i + 1];
                b[i + 1] = temp - dl[i + 1] * b[i];
            }
        }
        x[n - 1] = b[n - 1] / dd[n - 1];
        if (n > 1) x[n - 2] = (b[n - 2] - du[n - 2] * x[n - 1]) / dd[n - 2];
        for (int i = n - 3; i >= 0; --i) {
            x[i] = (b[i] - du[i] * x[i + 1] - du2[i] * x[i + 2]) / dd[i];
        }
        double nrm = 0.0;
        for (int i = 0; i < n; ++i) nrm += x[i] * x[i];
        nrm = sqrt(nrm);
        if (nrm == 0.0 || !isfinite(nrm)) {
            for (int i = 0; i < n; ++i) x[i] = (i == (int)(salt % (u32)n)) ? 1.0 : 0.0;
        } else {
            for (int i = 0; i < n; ++i) x[i] /= nrm;
        }
    }
}

// Orthonormalize column j of the row-major `cols` (`len` rows, `n`
// columns) against columns c0..j-1 by modified Gram-Schmidt; a
// vanished column is replaced by the unit vector least represented.
__device__ void orthonormalize_column(double* cols, int len, int n, int c0, int j)
{
    for (int i = c0; i < j; ++i) {
        double dot = 0.0;
        for (int r = 0; r < len; ++r) dot += cols[r * n + i] * cols[r * n + j];
        for (int r = 0; r < len; ++r) cols[r * n + j] -= dot * cols[r * n + i];
    }
    double nrm = 0.0;
    for (int r = 0; r < len; ++r) nrm += cols[r * n + j] * cols[r * n + j];
    nrm = sqrt(nrm);
    if (nrm > 1e-8) {
        for (int r = 0; r < len; ++r) cols[r * n + j] /= nrm;
        return;
    }
    // Degenerate: restart from a unit vector and orthogonalize against
    // every earlier column of the cluster.
    for (int cand = 0; cand < len; ++cand) {
        for (int r = 0; r < len; ++r) cols[r * n + j] = (r == cand) ? 1.0 : 0.0;
        for (int i = c0; i < j; ++i) {
            double dot = 0.0;
            for (int r = 0; r < len; ++r) dot += cols[r * n + i] * cols[r * n + j];
            for (int r = 0; r < len; ++r) cols[r * n + j] -= dot * cols[r * n + i];
        }
        nrm = 0.0;
        for (int r = 0; r < len; ++r) nrm += cols[r * n + j] * cols[r * n + j];
        nrm = sqrt(nrm);
        if (nrm > 0.5) {
            for (int r = 0; r < len; ++r) cols[r * n + j] /= nrm;
            return;
        }
    }
}

// ------------------------------------------------------------ syev

// Matrix i is n x n row-major at a_batch + i * n * n; eigenvalues
// ascending at w_batch + i * n; with want_v != 0 the eigenvectors as
// columns at v_batch + i * n * n, using scratch + i * n * n as a
// per-matrix workspace. Per Householder step the row scaling, the
// symmetric matrix-vector product, the rank-2 update and the
// reflector accumulation run over the block; then thread j bisects
// for eigenvalue j and, when vectors are wanted, inverse-iterates for
// its eigenvector of the tridiagonal (clustered eigenvalues get
// distinct shifts and are orthonormalized by the cluster's first
// thread) and multiplies it through the accumulated reflectors.
extern "C" __global__ void flynnel_syev_bisect_f64_blk(
    const double* __restrict__ a_batch,
    double* __restrict__ w_batch,
    double* __restrict__ v_batch,
    double* __restrict__ scratch,
    u32 batch, u32 n, u32 want_v)
{
    __shared__ double V[LINALG_MAX_N * LINALG_MAX_N];
    __shared__ double d[LINALG_MAX_N];
    __shared__ double e[LINALG_MAX_N];
    __shared__ double lam[LINALG_MAX_N];
    __shared__ double red[LINALG_BLOCK];
    __shared__ double sc[4];

    const u32 tid = threadIdx.x;
    const u32 b = blockIdx.x;
    if (b >= batch || n == 0u || n > LINALG_MAX_N) return;
    const int N = (int)n;
    const double* A = a_batch + (size_t)b * n * n;
    double* w = w_batch + (size_t)b * n;
    double* vout = v_batch + (size_t)b * n * n;
    double* X = scratch + (size_t)b * n * n;      // eigenvectors of the tridiagonal, as columns
#define VV(r, c) V[(r) * N + (c)]

    for (u32 idx = tid; idx < n * n; idx += LINALG_BLOCK) V[idx] = A[idx];
    __syncthreads();
    if (N == 1) {
        if (tid == 0) { w[0] = V[0]; if (want_v) vout[0] = 1.0; }
        return;
    }

    // ---- tred2
    for (int j = (int)tid; j < N; j += LINALG_BLOCK) d[j] = VV(N - 1, j);
    __syncthreads();
    for (int i = N - 1; i > 0; --i) {
        double part = 0.0;
        for (int k = (int)tid; k < i; k += LINALG_BLOCK) part += fabs(d[k]);
        const double scale = block_total(part, red);
        double h = 0.0;
        if (scale == 0.0) {
            if (tid == 0) e[i] = d[i - 1];
            __syncthreads();
            for (int j = (int)tid; j < i; j += LINALG_BLOCK) {
                d[j] = VV(i - 1, j);
                VV(i, j) = 0.0;
                VV(j, i) = 0.0;
            }
        } else {
            double hp = 0.0;
            for (int k = (int)tid; k < i; k += LINALG_BLOCK) { d[k] /= scale; hp += d[k] * d[k]; }
            h = block_total(hp, red);
            if (tid == 0) {
                double f = d[i - 1];
                double g = sqrt(h);
                if (f > 0.0) g = -g;
                e[i] = scale * g;
                h -= f * g;
                d[i - 1] = f - g;
                sc[0] = h;
            }
            __syncthreads();
            h = sc[0];
            for (int j = (int)tid; j < i; j += LINALG_BLOCK) {
                VV(j, i) = d[j];
                double g = 0.0;
                for (int k = 0; k <= j; ++k) g += VV(j, k) * d[k];
                for (int k = j + 1; k < i; ++k) g += VV(k, j) * d[k];
                e[j] = g;
            }
            __syncthreads();
            double fp = 0.0;
            for (int j = (int)tid; j < i; j += LINALG_BLOCK) { e[j] /= h; fp += e[j] * d[j]; }
            const double f = block_total(fp, red);
            const double hh = f / (h + h);
            for (int j = (int)tid; j < i; j += LINALG_BLOCK) e[j] -= hh * d[j];
            __syncthreads();
            for (int idx = (int)tid; idx < i * i; idx += LINALG_BLOCK) {
                const int k = idx / i, j = idx - k * i;
                if (k >= j) VV(k, j) -= (d[j] * e[k] + e[j] * d[k]);
            }
            __syncthreads();
            for (int j = (int)tid; j < i; j += LINALG_BLOCK) { d[j] = VV(i - 1, j); VV(i, j) = 0.0; }
        }
        __syncthreads();
        if (tid == 0) d[i] = h;
        __syncthreads();
    }
    if (want_v) {
        for (int i = 0; i < N - 1; ++i) {
            if (tid == 0) { VV(N - 1, i) = VV(i, i); VV(i, i) = 1.0; }
            __syncthreads();
            const double h = d[i + 1];
            if (h != 0.0) {
                for (int k = (int)tid; k <= i; k += LINALG_BLOCK) d[k] = VV(k, i + 1) / h;
                __syncthreads();
                for (int j = (int)tid; j <= i; j += LINALG_BLOCK) {
                    double g = 0.0;
                    for (int k = 0; k <= i; ++k) g += VV(k, i + 1) * VV(k, j);
                    red[j] = g;
                }
                __syncthreads();
                for (int idx = (int)tid; idx < (i + 1) * (i + 1); idx += LINALG_BLOCK) {
                    const int k = idx / (i + 1), j = idx - k * (i + 1);
                    VV(k, j) -= red[j] * d[k];
                }
                __syncthreads();
            }
            for (int k = (int)tid; k <= i; k += LINALG_BLOCK) VV(k, i + 1) = 0.0;
            __syncthreads();
        }
        for (int j = (int)tid; j < N; j += LINALG_BLOCK) { d[j] = VV(N - 1, j); VV(N - 1, j) = 0.0; }
        __syncthreads();
        if (tid == 0) VV(N - 1, N - 1) = 1.0;
    } else {
        for (int j = (int)tid; j < N; j += LINALG_BLOCK) d[j] = VV(j, j);
    }
    if (tid == 0) e[0] = 0.0;
    __syncthreads();

    // ---- eigenvalues by bisection, one thread each.
    if (tid == 0) {
        double lo, hi;
        gershgorin(d, e, N, &lo, &hi);
        double tn = 0.0;
        for (int i = 0; i < N; ++i) tn = fmax(tn, fabs(d[i]) + (i > 0 ? fabs(e[i]) : 0.0) + (i + 1 < N ? fabs(e[i + 1]) : 0.0));
        sc[0] = lo;
        sc[1] = hi;
        sc[2] = fmax(DBL_EPSILON * tn * (double)N, DBL_MIN);   // pivot floor
        sc[3] = tn;
    }
    __syncthreads();
    const double pivmin = sc[2];
    const double tnorm = sc[3];
    if ((int)tid < N) lam[tid] = bisect_kth(d, e, N, (int)tid, sc[0], sc[1], pivmin);
    __syncthreads();
    for (int j = (int)tid; j < N; j += LINALG_BLOCK) w[j] = lam[j];
    if (!want_v) return;

    // ---- eigenvectors: inverse iteration on the tridiagonal, clustered
    // eigenvalues shifted apart, then Q * x through V.
    if ((int)tid < N) {
        const int j = (int)tid;
        const double ctol = 1e3 * DBL_EPSILON * tnorm;
        int c0 = j;
        while (c0 > 0 && lam[c0] - lam[c0 - 1] <= ctol) --c0;
        const double shift = lam[j] + (double)(j - c0) * 4.0 * DBL_EPSILON * tnorm;
        double x[LINALG_MAX_N];
        double wk[6 * LINALG_MAX_N];
        tridiag_inverse_iteration(d, e, N, shift, pivmin, (u32)j + 1u, x, wk);
        for (int r = 0; r < N; ++r) X[r * N + j] = x[r];
    }
    __syncthreads();
    // Cluster leaders orthonormalize their clusters in place.
    if ((int)tid < N) {
        const int j = (int)tid;
        const double ctol = 1e3 * DBL_EPSILON * tnorm;
        const bool leader = (j == 0) || (lam[j] - lam[j - 1] > ctol);
        if (leader) {
            int c1 = j;
            while (c1 + 1 < N && lam[c1 + 1] - lam[c1] <= ctol) ++c1;
            for (int c = j + 1; c <= c1; ++c) orthonormalize_column(X, N, N, j, c);
        }
    }
    __syncthreads();
    if ((int)tid < N) {
        const int j = (int)tid;
        double x[LINALG_MAX_N];
        for (int r = 0; r < N; ++r) x[r] = X[r * N + j];
        for (int r = 0; r < N; ++r) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k) acc += VV(r, k) * x[k];
            vout[r * N + j] = acc;
        }
    }
#undef VV
}

// ------------------------------------------------------------ gesvd

// Matrix i is m x n row-major at a_batch + i * m * n and is
// overwritten with U (m x n, orthonormal columns); singular values
// descending at sigma_batch + i * n; with want_v != 0 the right
// singular vectors (n x n, columns) at v_batch + i * n * n, using
// scratch + i * (3 n * n + m * n) as a per-matrix workspace
// (Q_V, the Golub-Kahan eigenvectors, then U). Householder
// bidiagonalization (LINPACK dsvdc) runs over the block with U
// sharing A's shared-memory tile and Q_V generated in the workspace;
// the singular values are the largest n eigenvalues of the
// Golub-Kahan tridiagonal of the bidiagonal, by bisection, one thread
// each, and their vectors come from inverse iteration on that
// tridiagonal (v the even and u the odd components), clusters
// orthonormalized by their first thread, then multiplied through
// the accumulated reflectors.
extern "C" __global__ void flynnel_gesvd_bisect_f64_blk(
    double* __restrict__ a_batch,
    double* __restrict__ sigma_batch,
    double* __restrict__ v_batch,
    double* __restrict__ scratch,
    u32 batch, u32 m, u32 n, u32 want_v)
{
    __shared__ double U[LINALG_MAX_N * LINALG_MAX_N];
    __shared__ double s[LINALG_MAX_N];
    __shared__ double e[LINALG_MAX_N];
    __shared__ double work[LINALG_MAX_N];
    __shared__ double gk_diag[GK_MAX];
    __shared__ double gk_off[GK_MAX];
    __shared__ double sig[LINALG_MAX_N];
    __shared__ double red[LINALG_BLOCK];
    __shared__ double sc[4];

    const u32 tid = threadIdx.x;
    const u32 b = blockIdx.x;
    if (b >= batch || n == 0u || m < n || m > LINALG_MAX_N) return;
    const int M = (int)m, N = (int)n;
    double* A = a_batch + (size_t)b * m * n;
    double* sigma_out = sigma_batch + (size_t)b * n;
    double* vout = v_batch + (size_t)b * n * n;
    double* QV = scratch + (size_t)b * (3 * n * n + m * n);   // n x n, right reflectors
    double* Z = QV + (size_t)n * n;                 // 2n x n, Golub-Kahan eigenvectors as columns
    double* UO = Z + (size_t)2 * n * n;             // m x n, U before it overwrites A
#define UU(i, j) U[(i) * N + (j)]
#define QVV(i, j) QV[(i) * N + (j)]

    for (u32 idx = tid; idx < m * n; idx += LINALG_BLOCK) U[idx] = A[idx];
    __syncthreads();

    // ---- bidiagonalization
    const int nct = min(M - 1, N);
    const int nrt = max(0, min(N - 2, M));
    for (int k = 0; k < max(nct, nrt); ++k) {
        if (k < nct) {
            double sq = 0.0;
            for (int i = k + (int)tid; i < M; i += LINALG_BLOCK) sq += UU(i, k) * UU(i, k);
            double sk = sqrt(block_total(sq, red));
            if (sk != 0.0) {
                if (UU(k, k) < 0.0) sk = -sk;
                for (int i = k + (int)tid; i < M; i += LINALG_BLOCK) UU(i, k) /= sk;
                __syncthreads();
                if (tid == 0) UU(k, k) += 1.0;
                __syncthreads();
            }
            if (tid == 0) s[k] = -sk;
        }
        __syncthreads();
        for (int j = k + 1 + (int)tid; j < N; j += LINALG_BLOCK) {
            if (k < nct && s[k] != 0.0) {
                double t = 0.0;
                for (int i = k; i < M; ++i) t += UU(i, k) * UU(i, j);
                t = -t / UU(k, k);
                for (int i = k; i < M; ++i) UU(i, j) += t * UU(i, k);
            }
            e[j] = UU(k, j);
        }
        __syncthreads();
        if (k < nrt) {
            if (tid == 0) {
                double ek = 0.0;
                for (int i = k + 1; i < N; ++i) ek = hypot(ek, e[i]);
                if (ek != 0.0) {
                    if (e[k + 1] < 0.0) ek = -ek;
                    for (int i = k + 1; i < N; ++i) e[i] /= ek;
                    e[k + 1] += 1.0;
                }
                e[k] = -ek;
            }
            __syncthreads();
            if (k + 1 < M && e[k] != 0.0) {
                for (int i = k + 1 + (int)tid; i < M; i += LINALG_BLOCK) {
                    double acc = 0.0;
                    for (int j = k + 1; j < N; ++j) acc += e[j] * UU(i, j);
                    work[i] = acc;
                }
                __syncthreads();
                for (int idx = (int)tid; idx < (M - k - 1) * (N - k - 1); idx += LINALG_BLOCK) {
                    const int i = k + 1 + idx / (N - k - 1);
                    const int j = k + 1 + idx - (i - k - 1) * (N - k - 1);
                    UU(i, j) += (-e[j] / e[k + 1]) * work[i];
                }
                __syncthreads();
            }
            if (want_v) {
                for (int i = k + 1 + (int)tid; i < N; i += LINALG_BLOCK) QVV(i, k) = e[i];
            }
        }
        __syncthreads();
    }
    const int p = min(N, M + 1);
    if (tid == 0) {
        if (nct < N) s[nct] = UU(nct, nct);
        if (M < p) s[p - 1] = 0.0;
        if (nrt + 1 < p) e[nrt] = UU(nrt, p - 1);
        e[p - 1] = 0.0;
    }
    __syncthreads();

    // ---- generate U in place (columns >= nct become the identity).
    for (int idx = (int)tid; idx < M * (N - nct); idx += LINALG_BLOCK) {
        const int i = idx / (N - nct), j = nct + idx - i * (N - nct);
        UU(i, j) = (i == j) ? 1.0 : 0.0;
    }
    __syncthreads();
    for (int k = nct - 1; k >= 0; --k) {
        if (s[k] != 0.0) {
            for (int j = k + 1 + (int)tid; j < N; j += LINALG_BLOCK) {
                double t = 0.0;
                for (int i = k; i < M; ++i) t += UU(i, k) * UU(i, j);
                t = -t / UU(k, k);
                for (int i = k; i < M; ++i) UU(i, j) += t * UU(i, k);
            }
            __syncthreads();
            for (int i = k + (int)tid; i < M; i += LINALG_BLOCK) UU(i, k) = -UU(i, k);
            __syncthreads();
            if (tid == 0) UU(k, k) += 1.0;
            // U shares A's storage, so row k - 1 still holds the
            // superdiagonal: every row above the diagonal is zeroed.
            for (int i = (int)tid; i < k; i += LINALG_BLOCK) UU(i, k) = 0.0;
        } else {
            for (int i = (int)tid; i < M; i += LINALG_BLOCK) UU(i, k) = (i == k) ? 1.0 : 0.0;
        }
        __syncthreads();
    }
    // ---- generate Q_V in the workspace.
    if (want_v) {
        for (int k = N - 1; k >= 0; --k) {
            if (k < nrt && e[k] != 0.0) {
                for (int j = k + 1 + (int)tid; j < N; j += LINALG_BLOCK) {
                    double t = 0.0;
                    for (int i = k + 1; i < N; ++i) t += QVV(i, k) * QVV(i, j);
                    t = -t / QVV(k + 1, k);
                    for (int i = k + 1; i < N; ++i) QVV(i, j) += t * QVV(i, k);
                }
                __syncthreads();
            }
            for (int i = (int)tid; i < N; i += LINALG_BLOCK) QVV(i, k) = (i == k) ? 1.0 : 0.0;
            __syncthreads();
        }
    }

    // ---- singular values: the largest n eigenvalues of the Golub-Kahan
    // tridiagonal (zero diagonal; off-diagonals s0, e0, s1, e1, ...).
    const int G = 2 * N;
    for (int i = (int)tid; i < G; i += LINALG_BLOCK) {
        gk_diag[i] = 0.0;
        if (i == 0) gk_off[i] = 0.0;
        else if ((i & 1) == 1) gk_off[i] = s[i >> 1];
        else gk_off[i] = e[(i >> 1) - 1];
    }
    __syncthreads();
    if (tid == 0) {
        double lo, hi;
        gershgorin(gk_diag, gk_off, G, &lo, &hi);
        double tn = 0.0;
        for (int i = 0; i < G; ++i) tn = fmax(tn, (i > 0 ? fabs(gk_off[i]) : 0.0) + (i + 1 < G ? fabs(gk_off[i + 1]) : 0.0));
        sc[0] = lo;
        sc[1] = hi;
        sc[2] = fmax(DBL_EPSILON * tn * (double)G, DBL_MIN);
        sc[3] = tn;
    }
    __syncthreads();
    const double pivmin = sc[2];
    const double tnorm = sc[3];
    if ((int)tid < N) {
        // Thread j takes eigenvalue index 2n - 1 - j: descending order.
        const double v = bisect_kth(gk_diag, gk_off, G, G - 1 - (int)tid, sc[0], sc[1], pivmin);
        sig[tid] = fmax(v, 0.0);
    }
    __syncthreads();
    for (int j = (int)tid; j < N; j += LINALG_BLOCK) sigma_out[j] = sig[j];

    // ---- vectors: inverse iteration on the Golub-Kahan tridiagonal.
    if ((int)tid < N) {
        const int j = (int)tid;
        const double ctol = 1e3 * DBL_EPSILON * tnorm;
        int c0 = j;
        while (c0 > 0 && sig[c0 - 1] - sig[c0] <= ctol) --c0;
        const double shift = sig[j] - (double)(j - c0) * 4.0 * DBL_EPSILON * tnorm;
        double z[GK_MAX];
        double wk[6 * GK_MAX];
        tridiag_inverse_iteration(gk_diag, gk_off, G, shift, pivmin, (u32)j + 1u, z, wk);
        for (int r = 0; r < G; ++r) Z[r * N + j] = z[r];
    }
    __syncthreads();
    if ((int)tid < N) {
        const int j = (int)tid;
        const double ctol = 1e3 * DBL_EPSILON * tnorm;
        const bool leader = (j == 0) || (sig[j - 1] - sig[j] > ctol);
        if (leader) {
            int c1 = j;
            while (c1 + 1 < N && sig[c1] - sig[c1 + 1] <= ctol) ++c1;
            for (int c = j + 1; c <= c1; ++c) orthonormalize_column(Z, G, N, j, c);
        }
    }
    __syncthreads();
    // u = odd components, v = even components, each renormalized;
    // U_out = Q_U u, V_out = Q_V v. Q_U is the shared tile, which is
    // overwritten with U_out only after every thread has read it.
    double uh[LINALG_MAX_N];
    double vh[LINALG_MAX_N];
    if ((int)tid < N) {
        const int j = (int)tid;
        double nu = 0.0, nv = 0.0;
        for (int r = 0; r < N; ++r) {
            vh[r] = Z[(2 * r) * N + j];
            uh[r] = Z[(2 * r + 1) * N + j];
            nu += uh[r] * uh[r];
            nv += vh[r] * vh[r];
        }
        nu = sqrt(nu);
        nv = sqrt(nv);
        for (int r = 0; r < N; ++r) {
            uh[r] = (nu > 0.0) ? uh[r] / nu : 0.0;
            vh[r] = (nv > 0.0) ? vh[r] / nv : 0.0;
        }
        if (want_v) {
            for (int r = 0; r < N; ++r) {
                double acc = 0.0;
                for (int k = 0; k < N; ++k) acc += QVV(r, k) * vh[k];
                vout[r * N + j] = acc;
            }
        }
        for (int r = 0; r < M; ++r) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k) acc += UU(r, k) * uh[k];
            UO[r * N + j] = acc;
        }
    }
    __syncthreads();
    // Vanished singular values leave undefined directions; complete
    // U and V to orthonormal column sets (thread 0, rare).
    if (tid == 0) {
        for (int j = 0; j < N; ++j) {
            if (sig[j] <= 64.0 * DBL_EPSILON * tnorm) {
                orthonormalize_column(UO, M, N, 0, j);
                if (want_v) orthonormalize_column(vout, N, N, 0, j);
            }
        }
    }
    __syncthreads();
    for (u32 idx = tid; idx < m * n; idx += LINALG_BLOCK) A[idx] = UO[idx];
#undef UU
#undef QVV
}

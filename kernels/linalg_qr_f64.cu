// Batched symmetric eigendecomposition by Householder
// tridiagonalisation and implicit QL with shifts (the EISPACK tred2 /
// tql2 pair as in JAMA), one block per matrix, n <= LINALG_MAX_N.
// Companion module to linalg_f64.cu with the same launch contract:
// pointers first, then u32 scalars.

#include <math.h>

typedef unsigned int u32;

#define LINALG_MAX_N 64
#define LINALG_BLOCK 256

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

// Matrix i is n x n row-major at a_batch + i * n * n; eigenvalues
// ascending at w_batch + i * n; with want_v != 0 the eigenvectors as
// columns at v_batch + i * n * n. Per Householder step the row
// scaling, the symmetric matrix-vector product, the rank-2 update and
// the reflector accumulation run over the block; each QL iteration's
// recurrence runs on thread 0 with its rotations recorded, then the
// block applies them to every row of V.
extern "C" __global__ void flynnel_syev_qr_f64_blk(
    const double* __restrict__ a_batch,
    double* __restrict__ w_batch,
    double* __restrict__ v_batch,
    u32 batch, u32 n, u32 want_v)
{
    __shared__ double V[LINALG_MAX_N * LINALG_MAX_N];
    __shared__ double d[LINALG_MAX_N];
    __shared__ double e[LINALG_MAX_N];
    __shared__ double rc[LINALG_MAX_N];
    __shared__ double rs[LINALG_MAX_N];
    __shared__ double red[LINALG_BLOCK];
    __shared__ double sc[2];
    __shared__ int perm[LINALG_MAX_N];
    __shared__ int ctl[2];

    const u32 tid = threadIdx.x;
    const u32 b = blockIdx.x;
    if (b >= batch || n == 0u || n > LINALG_MAX_N) return;
    const int N = (int)n;
    const double* A = a_batch + (size_t)b * n * n;
    double* w = w_batch + (size_t)b * n;
    double* vout = v_batch + (size_t)b * n * n;
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
            // Similarity transformation of the leading i x i block:
            // e[j] = sum_k Vlow(j, k) d[k] over the stored lower triangle.
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
        // Accumulate the reflectors into V.
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

    // ---- tql2
    if (tid == 0) {
        for (int i = 1; i < N; ++i) e[i - 1] = e[i];
        e[N - 1] = 0.0;
    }
    __syncthreads();
    const double eps = ldexp(1.0, -52);
    double f = 0.0, tst1 = 0.0;          // meaningful on thread 0 only
    for (int l = 0; l < N; ++l) {
        if (tid == 0) {
            tst1 = fmax(tst1, fabs(d[l]) + fabs(e[l]));
            int m = l;
            while (m < N - 1 && fabs(e[m]) > eps * tst1) ++m;
            ctl[0] = m;
        }
        __syncthreads();
        const int m = ctl[0];
        __syncthreads();
        if (m > l) {
            int more;
            do {
                if (tid == 0) {
                    double g = d[l];
                    double p = (d[l + 1] - g) / (2.0 * e[l]);
                    double r = hypot(p, 1.0);
                    if (p < 0.0) r = -r;
                    d[l] = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);
                    const double dl1 = d[l + 1];
                    double h = g - d[l];
                    for (int i = l + 2; i < N; ++i) d[i] -= h;
                    f += h;
                    p = d[m];
                    double c = 1.0, c2 = 1.0, c3 = 1.0, s = 0.0, s2 = 0.0;
                    const double el1 = e[l + 1];
                    for (int i = m - 1; i >= l; --i) {
                        c3 = c2; c2 = c; s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = hypot(p, e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);
                        rc[i] = c;
                        rs[i] = s;
                    }
                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;
                    ctl[1] = fabs(e[l]) > eps * tst1;
                }
                __syncthreads();
                more = ctl[1];
                if (want_v) {
                    for (int k = (int)tid; k < N; k += LINALG_BLOCK) {
                        for (int i = m - 1; i >= l; --i) {
                            const double h = VV(k, i + 1);
                            VV(k, i + 1) = rs[i] * VV(k, i) + rc[i] * h;
                            VV(k, i) = rc[i] * VV(k, i) - rs[i] * h;
                        }
                    }
                }
                __syncthreads();
            } while (more);
        }
        if (tid == 0) { d[l] += f; e[l] = 0.0; }
        __syncthreads();
    }

    // ---- ascending order through a permutation, written to global.
    if (tid == 0) {
        for (int i = 0; i < N; ++i) perm[i] = i;
        for (int i = 1; i < N; ++i) {
            const int pi = perm[i];
            int j = i - 1;
            while (j >= 0 && d[perm[j]] > d[pi]) { perm[j + 1] = perm[j]; --j; }
            perm[j + 1] = pi;
        }
    }
    __syncthreads();
    for (int j = (int)tid; j < N; j += LINALG_BLOCK) w[j] = d[perm[j]];
    if (want_v) {
        for (u32 idx = tid; idx < n * n; idx += LINALG_BLOCK) {
            const int r = (int)(idx / n), j = (int)(idx - (u32)r * n);
            vout[idx] = VV(r, perm[j]);
        }
    }
#undef VV
}

// ------------------------------------------------------------ gesvd by bidiagonalisation + QR

// Batched SVD by Householder bidiagonalisation and implicit-shift QR
// on the bidiagonal (LINPACK dsvdc as in JAMA), one block per matrix,
// m >= n, m <= LINALG_MAX_N. Matrix i is m x n row-major at a_batch +
// i * m * n and is overwritten with U (m x n, orthonormal columns);
// singular values descending at sigma_batch + i * n; with want_v != 0
// the right singular vectors (n x n, columns) at v_batch + i * n * n.
// U shares A's shared-memory tile; V is worked in the output buffer
// itself. Column reductions and updates run over the block; the QR
// control loop runs on thread 0 with each rotation sequence recorded
// and then applied to every row by the block.
extern "C" __global__ void flynnel_gesvd_qr_f64_blk(
    double* __restrict__ a_batch,
    double* __restrict__ sigma_batch,
    double* __restrict__ v_batch,
    u32 batch, u32 m, u32 n, u32 want_v)
{
    __shared__ double U[LINALG_MAX_N * LINALG_MAX_N];
    __shared__ double s[LINALG_MAX_N];
    __shared__ double e[LINALG_MAX_N];
    __shared__ double work[LINALG_MAX_N];
    __shared__ double red[LINALG_BLOCK];
    __shared__ double rot_u[3 * LINALG_MAX_N];     // (cs, sn, other column) per step
    __shared__ double rot_v[3 * LINALG_MAX_N];
    __shared__ int ctl[6];

    const u32 tid = threadIdx.x;
    const u32 b = blockIdx.x;
    if (b >= batch || n == 0u || m < n || m > LINALG_MAX_N) return;
    const int M = (int)m, N = (int)n;
    double* A = a_batch + (size_t)b * m * n;
    double* sig = sigma_batch + (size_t)b * n;
    double* vout = v_batch + (size_t)b * n * n;
#define UU(i, j) U[(i) * N + (j)]
#define VG(i, j) vout[(i) * N + (j)]

    for (u32 idx = tid; idx < m * n; idx += LINALG_BLOCK) U[idx] = A[idx];
    __syncthreads();

    // ---- bidiagonalisation
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
        // Columns j > k: reflect by column k, then e[j] = A[k][j].
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
                for (int i = k + 1 + (int)tid; i < N; i += LINALG_BLOCK) VG(i, k) = e[i];
            }
        }
        __syncthreads();
    }
    // The final bidiagonal of order p.
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
    // ---- generate V in the output buffer.
    if (want_v) {
        for (int k = N - 1; k >= 0; --k) {
            if (k < nrt && e[k] != 0.0) {
                for (int j = k + 1 + (int)tid; j < N; j += LINALG_BLOCK) {
                    double t = 0.0;
                    for (int i = k + 1; i < N; ++i) t += VG(i, k) * VG(i, j);
                    t = -t / VG(k + 1, k);
                    for (int i = k + 1; i < N; ++i) VG(i, j) += t * VG(i, k);
                }
                __syncthreads();
            }
            for (int i = (int)tid; i < N; i += LINALG_BLOCK) VG(i, k) = (i == k) ? 1.0 : 0.0;
            __syncthreads();
        }
    }

    // ---- QR iteration on the bidiagonal. Thread 0 steers; ctl[0] =
    // kase, ctl[1] = k, ctl[2] = p, ctl[3] = rotation count, ctl[4] =
    // continue flag.
    const double eps = ldexp(1.0, -52);
    const double tiny = ldexp(1.0, -966);
    const int pfinal = p - 1;
    if (tid == 0) ctl[2] = p;
    __syncthreads();
    while (true) {
        if (tid == 0) {
            int kase = 0, k;
            int cur_p = ctl[2];
            if (cur_p <= 0) {
                ctl[4] = 0;
            } else {
                for (k = cur_p - 2; k >= -1; --k) {
                    if (k == -1) break;
                    if (fabs(e[k]) <= tiny + eps * (fabs(s[k]) + fabs(s[k + 1]))) { e[k] = 0.0; break; }
                }
                if (k == cur_p - 2) {
                    kase = 4;
                } else {
                    int ks;
                    for (ks = cur_p - 1; ks >= k; --ks) {
                        if (ks == k) break;
                        const double t = (ks != cur_p ? fabs(e[ks]) : 0.0) + (ks != k + 1 ? fabs(e[ks - 1]) : 0.0);
                        if (fabs(s[ks]) <= tiny + eps * t) { s[ks] = 0.0; break; }
                    }
                    if (ks == k) kase = 3;
                    else if (ks == cur_p - 1) kase = 1;
                    else { kase = 2; k = ks; }
                }
                ++k;
                int nrot = 0;
                if (kase == 1) {
                    double f = e[cur_p - 2];
                    e[cur_p - 2] = 0.0;
                    for (int j = cur_p - 2; j >= k; --j) {
                        double t = hypot(s[j], f);
                        const double cs = s[j] / t, sn = f / t;
                        s[j] = t;
                        if (j != k) { f = -sn * e[j - 1]; e[j - 1] = cs * e[j - 1]; }
                        rot_v[3 * nrot] = cs; rot_v[3 * nrot + 1] = sn; rot_v[3 * nrot + 2] = (double)j;
                        ++nrot;
                    }
                } else if (kase == 2) {
                    double f = e[k - 1];
                    e[k - 1] = 0.0;
                    for (int j = k; j < cur_p; ++j) {
                        double t = hypot(s[j], f);
                        const double cs = s[j] / t, sn = f / t;
                        s[j] = t;
                        f = -sn * e[j];
                        e[j] = cs * e[j];
                        rot_u[3 * nrot] = cs; rot_u[3 * nrot + 1] = sn; rot_u[3 * nrot + 2] = (double)j;
                        ++nrot;
                    }
                } else if (kase == 3) {
                    const double scale = fmax(fmax(fmax(fmax(fabs(s[cur_p - 1]), fabs(s[cur_p - 2])),
                        fabs(e[cur_p - 2])), fabs(s[k])), fabs(e[k]));
                    const double sp = s[cur_p - 1] / scale, spm1 = s[cur_p - 2] / scale;
                    const double epm1 = e[cur_p - 2] / scale, sk = s[k] / scale, ek = e[k] / scale;
                    const double bb = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
                    const double cc = (sp * epm1) * (sp * epm1);
                    double shift = 0.0;
                    if (bb != 0.0 || cc != 0.0) {
                        shift = sqrt(bb * bb + cc);
                        if (bb < 0.0) shift = -shift;
                        shift = cc / (bb + shift);
                    }
                    double f = (sk + sp) * (sk - sp) + shift;
                    double g = sk * ek;
                    for (int j = k; j < cur_p - 1; ++j) {
                        double t = hypot(f, g);
                        double cs = f / t, sn = g / t;
                        if (j != k) e[j - 1] = t;
                        f = cs * s[j] + sn * e[j];
                        e[j] = cs * e[j] - sn * s[j];
                        g = sn * s[j + 1];
                        s[j + 1] = cs * s[j + 1];
                        rot_v[3 * nrot] = cs; rot_v[3 * nrot + 1] = sn; rot_v[3 * nrot + 2] = (double)j;
                        t = hypot(f, g);
                        cs = f / t; sn = g / t;
                        s[j] = t;
                        f = cs * e[j] + sn * s[j + 1];
                        s[j + 1] = -sn * e[j] + cs * s[j + 1];
                        g = sn * e[j + 1];
                        e[j + 1] = cs * e[j + 1];
                        rot_u[3 * nrot] = cs; rot_u[3 * nrot + 1] = sn; rot_u[3 * nrot + 2] = (double)j;
                        ++nrot;
                    }
                    e[cur_p - 2] = f;
                } else {
                    // kase 4: a singular value converged; make it non-negative
                    // and sink it into descending order.
                    ctl[5] = 0;
                    if (s[k] <= 0.0) {
                        ctl[5] = (s[k] < 0.0) ? 1 : 0;   // column k of V flips sign
                        s[k] = (s[k] < 0.0) ? -s[k] : 0.0;
                    }
                    int kk = k;
                    while (kk < pfinal) {
                        if (s[kk] >= s[kk + 1]) break;
                        const double t = s[kk]; s[kk] = s[kk + 1]; s[kk + 1] = t;
                        ++kk;
                    }
                    rot_u[0] = (double)k;              // swap range for the vectors
                    rot_u[1] = (double)kk;
                    --cur_p;
                }
                ctl[0] = kase; ctl[1] = k; ctl[2] = cur_p; ctl[3] = nrot; ctl[4] = 1;
            }
        }
        __syncthreads();
        const int go = ctl[4];
        if (!go) break;
        const int kase = ctl[0], k = ctl[1], cur_p_after = ctl[2], nrot = ctl[3];
        if (kase == 1) {
            if (want_v) {
                for (int i = (int)tid; i < N; i += LINALG_BLOCK) {
                    for (int r = 0; r < nrot; ++r) {
                        const int j = (int)rot_v[3 * r + 2];
                        const double cs = rot_v[3 * r], sn = rot_v[3 * r + 1];
                        const double t = cs * VG(i, j) + sn * VG(i, cur_p_after - 1);
                        VG(i, cur_p_after - 1) = -sn * VG(i, j) + cs * VG(i, cur_p_after - 1);
                        VG(i, j) = t;
                    }
                }
            }
        } else if (kase == 2) {
            for (int i = (int)tid; i < M; i += LINALG_BLOCK) {
                for (int r = 0; r < nrot; ++r) {
                    const int j = (int)rot_u[3 * r + 2];
                    const double cs = rot_u[3 * r], sn = rot_u[3 * r + 1];
                    const double t = cs * UU(i, j) + sn * UU(i, k - 1);
                    UU(i, k - 1) = -sn * UU(i, j) + cs * UU(i, k - 1);
                    UU(i, j) = t;
                }
            }
        } else if (kase == 3) {
            if (want_v) {
                for (int i = (int)tid; i < N; i += LINALG_BLOCK) {
                    for (int r = 0; r < nrot; ++r) {
                        const int j = (int)rot_v[3 * r + 2];
                        const double cs = rot_v[3 * r], sn = rot_v[3 * r + 1];
                        const double t = cs * VG(i, j) + sn * VG(i, j + 1);
                        VG(i, j + 1) = -sn * VG(i, j) + cs * VG(i, j + 1);
                        VG(i, j) = t;
                    }
                }
            }
            for (int i = (int)tid; i < M; i += LINALG_BLOCK) {
                for (int r = 0; r < nrot; ++r) {
                    const int j = (int)rot_u[3 * r + 2];
                    if (j < M - 1) {
                        const double cs = rot_u[3 * r], sn = rot_u[3 * r + 1];
                        const double t = cs * UU(i, j) + sn * UU(i, j + 1);
                        UU(i, j + 1) = -sn * UU(i, j) + cs * UU(i, j + 1);
                        UU(i, j) = t;
                    }
                }
            }
        } else {
            const int k0 = (int)rot_u[0], k1 = (int)rot_u[1];
            if (want_v && ctl[5]) {
                for (int i = (int)tid; i < N; i += LINALG_BLOCK) VG(i, k0) = -VG(i, k0);
            }
            __syncthreads();
            // Column k0 sinks to k1: rotate the columns [k0, k1] left by one.
            if (k1 > k0) {
                for (int i = (int)tid; i < M; i += LINALG_BLOCK) {
                    const double t = UU(i, k0);
                    for (int j = k0; j < k1; ++j) UU(i, j) = UU(i, j + 1);
                    UU(i, k1) = t;
                }
                if (want_v) {
                    for (int i = (int)tid; i < N; i += LINALG_BLOCK) {
                        const double t = VG(i, k0);
                        for (int j = k0; j < k1; ++j) VG(i, j) = VG(i, j + 1);
                        VG(i, k1) = t;
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int j = (int)tid; j < N; j += LINALG_BLOCK) sig[j] = s[j];
    for (u32 idx = tid; idx < m * n; idx += LINALG_BLOCK) A[idx] = U[idx];
#undef UU
#undef VG
}

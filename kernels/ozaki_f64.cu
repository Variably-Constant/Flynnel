// Ozaki-scheme f64 GEMM on the int8 tensor cores. Each f64 operand is
// aligned to its row (A) or column (B) maximum exponent, scaled to a
// 56-bit integer and cut into SLICES 7-bit signed slices; slice pairs
// (i, j) with i + j < SLICES are multiplied exactly by int8 mma with
// int32 accumulation and recombined into f64 with two-sum
// compensation, smallest weight first. Truncation error is below
// 2^-53 of (max |A row| * max |B column| * k); operands must be
// finite; m, n and k multiples of 32, k <= 16384 (eight pair products
// share one int32 accumulator).
//
// Kernel argument convention: pointers first, then u32 scalars. Grids
// are one-dimensional; each kernel decodes batch and tile from
// blockIdx.x.

#include <cstdint>
#include <math.h>
#include <mma.h>

using namespace nvcuda;

namespace {
constexpr int SLICES = 8;
constexpr int BITS_PER_SLICE = 7;
constexpr int TOP_BIT = 55;             // implicit 1 of the aligned integer
constexpr int TILE = 32;                // block tile, 2 x 2 warps of 16 x 16
constexpr int KSTEP = 32;               // k per staged step: two 16-wide mma halves
constexpr int NO_EXP = -1000000;        // exponent of an all-zero row / column
constexpr int THREADS = 128;
}  // namespace

// Binary exponent of |x| for finite non-zero x; NO_EXP otherwise.
__device__ __forceinline__ int exp_of(double x) {
    if (x == 0.0 || !isfinite(x)) return NO_EXP;
    return ilogb(x);
}

// Per-row maximum exponent of A (batch x m x k): one warp per row,
// rows numbered batch-major across the grid.
extern "C" __global__ void flynnel_ozaki_rowexp_f64(
    const double* __restrict__ a, int* __restrict__ rowexp,
    unsigned batch, unsigned m, unsigned k)
{
    const unsigned lane = threadIdx.x & 31;
    const size_t row = (size_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= (size_t)batch * m) return;
    const double* src = a + row * k;
    int e = NO_EXP;
    for (unsigned c = lane; c < k; c += 32) e = max(e, exp_of(src[c]));
    for (int off = 16; off > 0; off >>= 1) e = max(e, __shfl_xor_sync(0xffffffffu, e, off));
    if (lane == 0) rowexp[row] = e;
}

// Per-column maximum exponent of B (batch x k x n): a block covers 32
// columns of one matrix, its 8 warps stride over the rows, coalesced.
extern "C" __global__ void flynnel_ozaki_colexp_f64(
    const double* __restrict__ bmat, int* __restrict__ colexp,
    unsigned batch, unsigned k, unsigned n)
{
    __shared__ int part[8][32];
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned ntiles = (n + 31) / 32;
    const unsigned b = blockIdx.x / ntiles;
    const unsigned col = (blockIdx.x - b * ntiles) * 32 + lane;
    if (b >= batch) return;
    const double* src = bmat + (size_t)b * k * n;
    int e = NO_EXP;
    if (col < n) {
        for (unsigned r = warp; r < k; r += 8) e = max(e, exp_of(src[(size_t)r * n + col]));
    }
    part[warp][lane] = e;
    __syncthreads();
    if (warp == 0 && col < n) {
        for (int w = 1; w < 8; ++w) e = max(e, part[w][lane]);
        colexp[(size_t)b * n + col] = e;
    }
}

// The 56-bit integer of |x| aligned to exponent `ref`: bit TOP_BIT is
// the implicit 1 when x sits at the row / column maximum.
__device__ __forceinline__ unsigned long long aligned_int(double x, int ref) {
    const int ex = exp_of(x);
    if (ex == NO_EXP) return 0ull;
    const int delta = ref - ex;
    if (delta >= 64) return 0ull;
    const double scaled = scalbn(fabs(x), TOP_BIT - ex);   // in [2^55, 2^56), exact
    const unsigned long long u = (unsigned long long)scaled;
    return delta > 0 ? (u >> delta) : u;
}

__device__ __forceinline__ void write_slices(
    unsigned long long u, bool negative, int8_t* out, size_t elem, size_t slice_stride)
{
#pragma unroll
    for (int s = 0; s < SLICES; ++s) {
        const int v = (int)((u >> (TOP_BIT - (BITS_PER_SLICE - 1) - BITS_PER_SLICE * s)) & 0x7Fu);
        out[s * slice_stride + elem] = (int8_t)(negative ? -v : v);
    }
}

// A (batch x m x k) into SLICES slice matrices of the same layout.
extern "C" __global__ void flynnel_ozaki_split_a_f64(
    const double* __restrict__ a, const int* __restrict__ rowexp, int8_t* __restrict__ out,
    unsigned batch, unsigned m, unsigned k)
{
    const size_t total = (size_t)batch * m * k;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t row = idx / k;                       // batch-major row index
    const double x = a[idx];
    const unsigned long long u = aligned_int(x, rowexp[row]);
    write_slices(u, signbit(x) != 0, out, idx, total);
}

// B (batch x k x n) into SLICES slice matrices stored transposed
// (batch x n x k), so a warp's 16-column mma operand is a 32-byte
// aligned column-major tile. A block transposes one 32 x 32 tile
// through shared memory so the reads run along n and the byte writes
// along k, both coalesced; batch * ceil(k / 32) * ceil(n / 32) blocks
// of 256 threads.
extern "C" __global__ void __launch_bounds__(THREADS * 2)
flynnel_ozaki_split_bt_f64(
    const double* __restrict__ bmat, const int* __restrict__ colexp, int8_t* __restrict__ out_t,
    unsigned batch, unsigned k, unsigned n)
{
    __shared__ int8_t tile[SLICES][32][33];
    const unsigned kt = (k + 31) / 32, nt = (n + 31) / 32;
    const unsigned per_b = kt * nt;
    const unsigned b = blockIdx.x / per_b;
    if (b >= batch) return;
    const unsigned rem = blockIdx.x - b * per_b;
    const unsigned r0 = (rem / nt) * 32;
    const unsigned c0 = (rem - (rem / nt) * nt) * 32;
    const size_t total = (size_t)batch * k * n;
    const double* src = bmat + (size_t)b * k * n;
    int8_t* dst = out_t + (size_t)b * k * n;
    for (unsigned e = threadIdx.x; e < 32 * 32; e += blockDim.x) {
        const unsigned rr = e >> 5, cc = e & 31;
        const unsigned r = r0 + rr, c = c0 + cc;
        unsigned long long u = 0ull;
        bool neg = false;
        if (r < k && c < n) {
            const double x = src[(size_t)r * n + c];
            u = aligned_int(x, colexp[(size_t)b * n + c]);
            neg = signbit(x) != 0;
        }
#pragma unroll
        for (int s = 0; s < SLICES; ++s) {
            const int v = (int)((u >> (TOP_BIT - (BITS_PER_SLICE - 1) - BITS_PER_SLICE * s)) & 0x7Fu);
            tile[s][cc][rr] = (int8_t)(neg ? -v : v);
        }
    }
    __syncthreads();
    for (unsigned e = threadIdx.x; e < SLICES * 32 * 32; e += blockDim.x) {
        const unsigned s = e >> 10, rem2 = e & 1023, cc = rem2 >> 5, rr = rem2 & 31;
        const unsigned r = r0 + rr, c = c0 + cc;
        if (r < k && c < n) dst[(size_t)s * total + (size_t)c * k + r] = tile[s][cc][rr];
    }
}

// Two-sum accumulate of v into (hi, lo).
__device__ __forceinline__ void two_sum_add(double& hi, double& lo, double v) {
    const double t = hi + v;
    const double bb = t - hi;
    lo += (hi - (t - bb)) + (v - bb);
    hi = t;
}

// C = A * B over the slice pairs: batch * (m / 32) * (n / 32) blocks
// of 128 threads, one 32 x 32 output tile each. Per 32 of k the
// threads stage every slice's A and B tiles once, in 256-byte units
// (a fragment origin must be 32-byte aligned, which a 16-column int8
// sub-tile in place is not); each warp then runs the 36 pair
// products, accumulating pair (i, j) into the int32 fragment of its
// diagonal d = i + j, which shares the weight 2^(-7 d) and stays
// exact for k <= 16384. The eight diagonal tiles are folded into the
// f64 tile smallest weight first.
extern "C" __global__ void __launch_bounds__(THREADS)
flynnel_ozaki_gemm_f64(
    const int8_t* __restrict__ a_s, const int8_t* __restrict__ bt_s,
    const int* __restrict__ rowexp, const int* __restrict__ colexp, double* __restrict__ c,
    unsigned batch, unsigned m, unsigned n, unsigned k)
{
    __shared__ __align__(32) int8_t sa[SLICES][2][2][16][16];  // [slice][warp row][k half][row][k]
    __shared__ __align__(32) int8_t sb[SLICES][2][2][16][16];  // [slice][warp col][k half][col][k]
    __shared__ __align__(32) int32_t tile[TILE * TILE];
    __shared__ double acc_hi[TILE * TILE];
    __shared__ double acc_lo[TILE * TILE];

    const unsigned mt = m / TILE;
    const unsigned nt = n / TILE;
    const unsigned b = blockIdx.x / (mt * nt);
    const unsigned rem = blockIdx.x - b * (mt * nt);
    const unsigned row0 = (rem / nt) * TILE;
    const unsigned col0 = (rem - (rem / nt) * nt) * TILE;
    if (b >= batch) return;
    const unsigned t = threadIdx.x;
    const unsigned warp = t >> 5;
    const unsigned wr = warp >> 1;                    // warp's 16-row half of the tile
    const unsigned wc = warp & 1;                     // warp's 16-column half
    const size_t a_stride = (size_t)batch * m * k;    // one slice matrix
    const size_t b_stride = (size_t)batch * k * n;
    const int8_t* a_tile = a_s + ((size_t)b * m + row0) * k;
    const int8_t* bt_tile = bt_s + ((size_t)b * n + col0) * k;

    for (unsigned e = t; e < TILE * TILE; e += THREADS) {
        acc_hi[e] = 0.0;
        acc_lo[e] = 0.0;
    }

    using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, int>;
    FragC cd[SLICES];
#pragma unroll
    for (int d = 0; d < SLICES; ++d) wmma::fill_fragment(cd[d], 0);

    for (unsigned k0 = 0; k0 < k; k0 += KSTEP) {
        __syncthreads();                              // previous step's tiles consumed
        // 1024 16-byte vectors per step: 512 for the A slices, 512 for
        // the B slices; index -> (slice, 16-row half, k half, line).
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const unsigned idx = p * THREADS + t;
            const bool is_a = idx < 512;
            const unsigned u = is_a ? idx : idx - 512;
            const unsigned s = u >> 6;
            const unsigned half_sel = (u >> 5) & 1;
            const unsigned kh = (u >> 4) & 1;
            const unsigned line = u & 15;
            const size_t off = (size_t)(16 * half_sel + line) * k + k0 + 16 * kh;
            const int4 v = is_a
                ? *(const int4*)(a_tile + (size_t)s * a_stride + off)
                : *(const int4*)(bt_tile + (size_t)s * b_stride + off);
            int4* dst = is_a ? (int4*)sa[s][half_sel][kh][line] : (int4*)sb[s][half_sel][kh][line];
            *dst = v;
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            FragA af[SLICES];
#pragma unroll
            for (int i = 0; i < SLICES; ++i) wmma::load_matrix_sync(af[i], &sa[i][wr][h][0][0], 16);
#pragma unroll
            for (int j = 0; j < SLICES; ++j) {
                FragB bf;
                wmma::load_matrix_sync(bf, &sb[j][wc][h][0][0], 16);
#pragma unroll
                for (int i = 0; i + j < SLICES; ++i) wmma::mma_sync(cd[i + j], af[i], bf, cd[i + j]);
            }
        }
    }

#pragma unroll
    for (int d = SLICES - 1; d >= 0; --d) {
        const double weight = ldexp(1.0, -BITS_PER_SLICE * d);
        __syncthreads();                              // previous diagonal folded
        wmma::store_matrix_sync(tile + wr * 16 * TILE + wc * 16, cd[d], TILE, wmma::mem_row_major);
        __syncthreads();
        for (unsigned e = t; e < TILE * TILE; e += THREADS) {
            two_sum_add(acc_hi[e], acc_lo[e], (double)tile[e] * weight);
        }
    }
    __syncthreads();

    const int* re = rowexp + (size_t)b * m + row0;
    const int* ce = colexp + (size_t)b * n + col0;
    double* out = c + ((size_t)b * m + row0) * n + col0;
    for (unsigned e = threadIdx.x; e < TILE * TILE; e += THREADS) {
        const unsigned r = e / TILE;
        const unsigned cc = e - r * TILE;
        const double v = acc_hi[e] + acc_lo[e];
        const int scale = re[r] + ce[cc] - 2 * TOP_BIT + 98;
        out[(size_t)r * n + cc] = (v == 0.0) ? 0.0 : ldexp(v, scale);
    }
}

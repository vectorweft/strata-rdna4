// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 bf16 packed in a uint4 against 8 floats.
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = fmaf(__uint_as_float(v[j] << 16), x[2 * j], acc);
        acc = fmaf(__uint_as_float(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}

__global__ void __launch_bounds__(THREADS) gr_down_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float xn[D];
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        if (blockIdx.x == 0) a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    __syncthreads();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc = 0.0f;
#pragma unroll 4
    for (int j = lane; j < D / 8; j += 32) acc += dot8(__ldg(w4 + j), xn + j * 8);
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + __expf(-x));
    }
}

__global__ void __launch_bounds__(THREADS) gr_up_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float lo[LR];
    __shared__ float g[HC][UP_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    __syncthreads();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(a.w_up + (size_t) i * LR);
        float acc = dot8(__ldg(w4 + lane), lo + lane * 8);
        if (lane < LR / 8 - 32) acc += dot8(__ldg(w4 + 32 + lane), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    __syncthreads();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};

// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to global.
__global__ void __launch_bounds__(THREADS) gr_norm_multi_kernel(GrMulti m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}

#if defined(__HIPCC__)
constexpr int TILE = 1280;             // eight-token tile fits gfx1100's 64 KiB LDS limit
#else
constexpr int TILE = 2560;             // xn floats per token staged at a time: 320 chunks of 8, 10 per lane
#endif
constexpr int TQ = TILE / 8 / 32;      // uint4 weight chunks per lane per tile

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's 10 weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
__global__ void __launch_bounds__(THREADS) gr_down_multi_kernel(GrMulti m) {
    extern __shared__ __align__(16) float tile[];   // [T][TILE]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
    for (int base = 0; base < D; base += TILE) {
        uint4 wv[TQ];
        if (active) {
#pragma unroll
            for (int q = 0; q < TQ; ++q) wv[q] = __ldg(w4 + base / 8 + lane + 32 * q);
        }
        __syncthreads();                                   // the previous tile is consumed
        const float4* src4 = reinterpret_cast<const float4*>(m.xn);
        float4* tile4 = reinterpret_cast<float4*>(tile);
        for (int i = t; i < T * (TILE / 4); i += THREADS) {
            const int k = i / (TILE / 4), off = i - k * (TILE / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) acc[k] += dot8(wv[q], tile + k * TILE + j * 8);
        }
    }
    if (!active) return;
    float s[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// Split-K down projection (R9700 / RDNA4: 32 WGPs, 41 blocks of the kernel above leave most of them idle and each
// lane with 5 loads in flight).  Block (g, s) computes 16 rows (2 per warp) over K slice s of 1280: the 10 weight
// chunks per lane are issued before the activation tile is staged, the T x 1280 tile is shared by the 16 rows, and
// each (split, token, row) partial is written to `part`; the up kernel sums the 8 splits in a fixed order.
constexpr int SK_KC = 1280;
constexpr int SK_S = D / SK_KC;                    // 8
constexpr int SK_ROWS = 16;
constexpr int SK_RPW = SK_ROWS / WARPS;            // 2 rows per warp
constexpr int SK_ALL = LR + HC;                    // 320 down rows, then the 4 inject rows
constexpr int SK_GROUPS = (SK_ALL + SK_ROWS - 1) / SK_ROWS;   // 21
constexpr int SK_Q = SK_KC / 8 / 32;               // 5 uint4 chunks per lane per row
static_assert(SK_S * kFusedGrMaxT * SK_ALL + SK_S * kFusedGrMaxT == kFusedGrScratchExtra, "scratch layout");
static_assert(N % SK_KC == 0, "a K slice lies in one stream");

__global__ void __launch_bounds__(THREADS) gr_down_splitk_kernel(GrMulti m, float* __restrict__ part,
                                                                   float* __restrict__ part_ss) {
    extern __shared__ __align__(16) float tile[];   // [T][SK_KC]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T, g = blockIdx.x, s = blockIdx.y, base = s * SK_KC;
    const bool has_inject = m.a[0].w_inject != nullptr;
    uint4 wv[SK_RPW][SK_Q];
    int rows[SK_RPW];
    bool live[SK_RPW];
#pragma unroll
    for (int rr = 0; rr < SK_RPW; ++rr) {
        const int r = g * SK_ROWS + warp * SK_RPW + rr;
        rows[rr] = r;
        live[rr] = r < LR || (r < SK_ALL && has_inject);
        const uint16_t* wrow = live[rr] ? (r < LR ? m.a[0].w_down + (size_t) r * D : m.a[0].w_inject + (size_t) (r - LR) * D)
                                        : m.a[0].w_down;
        const uint4* w4 = reinterpret_cast<const uint4*>(wrow) + base / 8;
#pragma unroll
        for (int q = 0; q < SK_Q; ++q) wv[rr][q] = live[rr] ? __ldg(w4 + lane + 32 * q) : make_uint4(0, 0, 0, 0);
    }
    // the tile: R' * w_norm for this slice of every token (R' = R + the pending write), and in the first row group
    // each token's sum of squares of R' over the slice (the RMS of stream base / N comes from two slices)
    __shared__ float ss_part[WARPS][kFusedGrMaxT];
    const int c = base / N, d0 = base - c * N;
    float4* tile4 = reinterpret_cast<float4*>(tile);
    float ssq[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) ssq[k] = 0.0f;
    for (int i = t; i < T * (SK_KC / 4); i += THREADS) {
        const int k = i / (SK_KC / 4), off = i - k * (SK_KC / 4);
        const FusedGrArgs& a = m.a[k];
        float4 r = reinterpret_cast<const float4*>(a.R + base)[off];
        if (a.apply) {
            const float gw = 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC);
            const float4 b = reinterpret_cast<const float4*>(a.bo_prev + d0)[off];
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float4 g = reinterpret_cast<const float4*>(a.w_norm + base)[off];
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int kk = 0; kk < kFusedGrMaxT; ++kk) if (kk == k) ssq[kk] += sq;
        tile4[i] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    if (g == 0) {
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            const float v = warp_sum(ssq[k]);
            if (lane == 0) ss_part[warp][k] = v;
        }
    }
    __syncthreads();
    if (g == 0 && t < T) {
        float v = 0.0f;
        for (int w = 0; w < WARPS; ++w) v += ss_part[w][t];
        part_ss[s * kFusedGrMaxT + t] = v;
    }
#pragma unroll
    for (int rr = 0; rr < SK_RPW; ++rr) {
        if (!live[rr]) continue;
        float acc[kFusedGrMaxT];
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int q = 0; q < SK_Q; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) acc[k] += dot8(wv[rr][q], tile + k * SK_KC + j * 8);
        }
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            const float v = warp_sum(acc[k]);
            if (lane == k) part[((size_t) s * kFusedGrMaxT + k) * SK_ALL + rows[rr]] = v;
        }
    }
}

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
__global__ void __launch_bounds__(THREADS) gr_up_multi_kernel(GrMulti m, const float* __restrict__ part,
                                                              const float* __restrict__ part_ss) {
    __shared__ __align__(16) float lo[kFusedGrMaxT][LR];
    __shared__ float g[kFusedGrMaxT][HC][UPM_COLS];
    __shared__ float s_rs[kFusedGrMaxT][HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    if (part == nullptr) {
        for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    } else {
        // split-K: each stream's RMS scale from its two slices' sums of squares
        if (t < T * HC) {
            const int k = t / HC, c = t - k * HC;
            const int per = N / SK_KC;
            float ss = 0.0f;
            for (int j = 0; j < per; ++j) ss += part_ss[(c * per + j) * kFusedGrMaxT + k];
            const float v = rsqrtf(ss / (float) N + m.a[k].eps);
            s_rs[k][c] = v;
            if (blockIdx.x == 0) m.a[k].rs[c] = v;
        }
        __syncthreads();
        // the 8 partials of each row scaled by their stream's rs, in split order; then silu of the mean
        for (int i = t; i < T * SK_ALL; i += THREADS) {
            const int k = i / SK_ALL, r = i - k * SK_ALL;
            float sum = 0.0f;
#pragma unroll
            for (int sp = 0; sp < SK_S; ++sp)
                sum += s_rs[k][sp * SK_KC / N] * part[((size_t) sp * kFusedGrMaxT + k) * SK_ALL + r];
            if (r < LR) {
                const float x = sum / (float) HC;
                const float v = x / (1.0f + __expf(-x));
                lo[k][r] = v;
                if (blockIdx.x == 0) m.a[k].lo[r] = v;
            } else if (blockIdx.x == 0 && m.a[0].w_inject != nullptr) {
                m.a[k].inject_out[r - LR] = sum;
            }
        }
    }
    __syncthreads();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const uint4 wa = __ldg(w4 + lane);
        const uint4 wb = lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0);
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = part ? s_rs[lane][c] : a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float acc = dot8(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}

// ================================ variant 2: split-K with register blocking ================================
// Variant 1 is LDS-bound once T > 1: every warp reads the T x 1280 activation tile from shared memory once per row
// it owns (2) and per token, and the up kernel reads lo[T][320] from shared memory once per row (8 per warp), so
// both kernels fall from ~600 GB/s at T = 1 to ~330 at T = 4.  Here the down kernel gives each warp RPW2 rows and
// reads each activation chunk once for all of them, and the up kernel holds lo in registers.  A lane's T (padded to
// TT, a power of two) partial dots are reduce-scattered: each xor step hands half of the values still held to the
// partner lane, so TT = 4 costs 6 shuffles instead of 20, and token k ends in lanes [k * 32 / TT, (k + 1) * 32 / TT).
template <int TT>
__device__ __forceinline__ float reduce_scatter(float (&v)[TT], int lane) {
#pragma unroll
    for (int h = TT / 2, o = 16; h >= 1; h >>= 1, o >>= 1) {
        const bool hi = (lane & o) != 0;
#pragma unroll
        for (int i = 0; i < h; ++i) {
            const float send = hi ? v[i] : v[i + h];
            const float keep = hi ? v[i + h] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
    float s = v[0];
#pragma unroll
    for (int o = 16 / TT; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;   // the sum of token lane / (32 / TT)
}

// dot8 with the 8 activations in registers
__device__ __forceinline__ float dot8r(const uint4 w, const float4 a, const float4 b) {
    float acc = 0.0f;
    acc = fmaf(__uint_as_float(w.x << 16), a.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xffff0000u), a.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), a.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xffff0000u), a.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), b.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xffff0000u), b.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), b.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xffff0000u), b.w, acc);
    return acc;
}

constexpr int RPW2 = 4;                               // down rows per warp
constexpr int SK2_ROWS = WARPS * RPW2;                // 32
constexpr int SK2_GROUPS = (SK_ALL + SK2_ROWS - 1) / SK2_ROWS;   // 11

template <int TT>
__global__ void __launch_bounds__(THREADS) gr_down_splitk2_kernel(GrMulti m, float* __restrict__ part,
                                                                    float* __restrict__ part_ss) {
    extern __shared__ __align__(16) float tile[];   // [T][SK_KC]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T, g = blockIdx.x, s = blockIdx.y, base = s * SK_KC;
    const bool has_inject = m.a[0].w_inject != nullptr;
    uint4 wv[RPW2][SK_Q];
    int rows[RPW2];
    bool live[RPW2];
#pragma unroll
    for (int rr = 0; rr < RPW2; ++rr) {
        const int r = g * SK2_ROWS + warp * RPW2 + rr;
        rows[rr] = r;
        live[rr] = r < LR || (r < SK_ALL && has_inject);
        const uint16_t* wrow = live[rr] ? (r < LR ? m.a[0].w_down + (size_t) r * D : m.a[0].w_inject + (size_t) (r - LR) * D)
                                        : m.a[0].w_down;
        const uint4* w4 = reinterpret_cast<const uint4*>(wrow) + base / 8;
#pragma unroll
        for (int q = 0; q < SK_Q; ++q) wv[rr][q] = live[rr] ? __ldg(w4 + lane + 32 * q) : make_uint4(0, 0, 0, 0);
    }
    __shared__ float ss_part[WARPS][kFusedGrMaxT];
    const int c = base / N, d0 = base - c * N;
    float4* tile4 = reinterpret_cast<float4*>(tile);
    float ssq[TT];
#pragma unroll
    for (int k = 0; k < TT; ++k) {
        ssq[k] = 0.0f;
        if (k >= T) continue;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int off = t; off < SK_KC / 4; off += THREADS) {
            float4 r = reinterpret_cast<const float4*>(a.R + base)[off];
            if (a.apply) {
                const float4 b = reinterpret_cast<const float4*>(a.bo_prev + d0)[off];
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 w = reinterpret_cast<const float4*>(m.a[0].w_norm + base)[off];
            ssq[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            tile4[k * (SK_KC / 4) + off] = make_float4(r.x * w.x, r.y * w.y, r.z * w.z, r.w * w.w);
        }
    }
    if (g == 0) {
        const float v = reduce_scatter<TT>(ssq, lane);
        const int k = lane / (32 / TT);
        if (lane % (32 / TT) == 0 && k < T) ss_part[warp][k] = v;
    }
    __syncthreads();
    if (g == 0 && t < T) {
        float v = 0.0f;
        for (int w = 0; w < WARPS; ++w) v += ss_part[w][t];
        part_ss[s * kFusedGrMaxT + t] = v;
    }
    float acc[RPW2][TT];
#pragma unroll
    for (int rr = 0; rr < RPW2; ++rr)
#pragma unroll
        for (int k = 0; k < TT; ++k) acc[rr][k] = 0.0f;
#pragma unroll
    for (int q = 0; q < SK_Q; ++q) {
        const int j = lane + 32 * q;
#pragma unroll
        for (int k = 0; k < TT; ++k) {
            if (k >= T) continue;
            const float4 xa = tile4[(k * SK_KC + j * 8) / 4], xb = tile4[(k * SK_KC + j * 8) / 4 + 1];
#pragma unroll
            for (int rr = 0; rr < RPW2; ++rr) acc[rr][k] += dot8r(wv[rr][q], xa, xb);
        }
    }
#pragma unroll
    for (int rr = 0; rr < RPW2; ++rr) {
        const float v = reduce_scatter<TT>(acc[rr], lane);
        const int k = lane / (32 / TT);
        if (live[rr] && lane % (32 / TT) == 0 && k < T) part[((size_t) s * kFusedGrMaxT + k) * SK_ALL + rows[rr]] = v;
    }
}

// The T sums of an 8-lane group reduce-scattered the same way: token k ends in the group's lanes
// [k * 8 / TT, (k + 1) * 8 / TT) (TT <= 8).
template <int TT>
__device__ __forceinline__ float reduce_scatter8(float (&v)[TT], int lane) {
#pragma unroll
    for (int h = TT / 2, o = 4; h >= 1; h >>= 1, o >>= 1) {
        const bool hi = (lane & o) != 0;
#pragma unroll
        for (int i = 0; i < h; ++i) {
            const float send = hi ? v[i] : v[i + h];
            const float keep = hi ? v[i + h] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
    float s = v[0];
#pragma unroll
    for (int o = 4 / TT; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;
}

// The up projection: a row of w_up is 320 bf16 = 40 chunks of 8, so an 8-lane group owns a row (5 chunks per lane)
// and a warp works on 4 rows per pass; each lane keeps the weights of its IT rows in registers and applies every lo
// chunk it reads from shared memory to all of them (shared-memory reads are what bound variant 1 at T > 1).
constexpr int UP2_Q = LR / 8 / 8;                 // 5 chunks per lane
constexpr int UP2_IT = 4;                         // rows per lane: 128 rows per block, 80 blocks (IT = 1 / 2: T = 4 +40% / +4%)

// The T x 324 sums of the 8 split partials: every load of a thread issued before the first add.
template <int TT>
__device__ __forceinline__ void reduce_partials(const GrMulti& m, const float* __restrict__ part,
                                                const float (*s_rs)[HC], float (*lo)[LR], int t) {
    constexpr int U = (TT * SK_ALL + THREADS - 1) / THREADS;
    const int T = m.T;
    float v[U][SK_S];
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const int i = t + u * THREADS, k = i / SK_ALL, r = i - k * SK_ALL;
#pragma unroll
        for (int sp = 0; sp < SK_S; ++sp)
            v[u][sp] = i < T * SK_ALL ? part[((size_t) sp * kFusedGrMaxT + k) * SK_ALL + r] : 0.0f;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const int i = t + u * THREADS, k = i / SK_ALL, r = i - k * SK_ALL;
        if (i >= T * SK_ALL) break;
        float sum = 0.0f;
#pragma unroll
        for (int sp = 0; sp < SK_S; ++sp) sum += s_rs[k][sp * SK_KC / N] * v[u][sp];
        if (r < LR) {
            const float x = sum / (float) HC;
            const float y = x / (1.0f + __expf(-x));
            lo[k][r] = y;
            if (blockIdx.x == 0) m.a[k].lo[r] = y;
        } else if (blockIdx.x == 0 && m.a[0].w_inject != nullptr) {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
}

template <int TT, int IT>
__global__ void __launch_bounds__(THREADS) gr_up2_kernel(GrMulti m, const float* __restrict__ part,
                                                          const float* __restrict__ part_ss) {
    constexpr int COLS = IT * 4 * WARPS / HC;     // columns per block (x 4 streams = IT * 32 rows)
    __shared__ __align__(16) float lo[kFusedGrMaxT][LR];
    __shared__ float g[kFusedGrMaxT][HC][COLS];
    __shared__ float s_rs[kFusedGrMaxT][HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, sub = lane & 7;
    const int T = m.T;
    const int d0 = blockIdx.x * COLS;
    uint4 wv[IT][UP2_Q];
    int rowi[IT];
#pragma unroll
    for (int it = 0; it < IT; ++it) {
        const int r = warp * (4 * IT) + it * 4 + (lane >> 3), c = r / COLS, dd = r - c * COLS;
        rowi[it] = r;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) (c * N + d0 + dd) * LR);
#pragma unroll
        for (int q = 0; q < UP2_Q; ++q) wv[it][q] = __ldg(w4 + sub + 8 * q);
    }
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        const int per = N / SK_KC;
        float ss = 0.0f;
        for (int j = 0; j < per; ++j) ss += part_ss[(c * per + j) * kFusedGrMaxT + k];
        const float v = rsqrtf(ss / (float) N + m.a[k].eps);
        s_rs[k][c] = v;
        if (blockIdx.x == 0) m.a[k].rs[c] = v;
    }
    __syncthreads();
    reduce_partials<TT>(m, part, s_rs, lo, t);
    __syncthreads();
    const int mk = sub / (8 / TT);                               // the token this lane ends with
    const bool owner = sub % (8 / TT) == 0 && mk < T;
    float acc[IT][TT];
#pragma unroll
    for (int it = 0; it < IT; ++it)
#pragma unroll
        for (int k = 0; k < TT; ++k) acc[it][k] = 0.0f;
#pragma unroll
    for (int q = 0; q < UP2_Q; ++q) {
        const int j = sub + 8 * q;
#pragma unroll
        for (int k = 0; k < TT; ++k) {
            if (k >= T) continue;
            const float4* l4 = reinterpret_cast<const float4*>(lo[k]);
            const float4 xa = l4[2 * j], xb = l4[2 * j + 1];
#pragma unroll
            for (int it = 0; it < IT; ++it) acc[it][k] += dot8r(wv[it][q], xa, xb);
        }
    }
#pragma unroll
    for (int it = 0; it < IT; ++it) {
        const int r = rowi[it], c = r / COLS, dd = r - c * COLS, i = c * N + d0 + dd;
        const float mine = reduce_scatter8<TT>(acc[it], lane);
        if (owner) {
            const FusedGrArgs& a = m.a[mk];
            float rv = a.R[i];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), rv);
                a.R_out[i] = rv;
            }
            g[mk][c][dd] = rv * a.w_norm[i] * s_rs[mk][c] * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * COLS; i += THREADS) {
        const int k = i / COLS, col = i - k * COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}

template <int TT>
void launch_splitk2(const GrMulti& m, float* part, float* part_ss, cudaStream_t st, unsigned long long* stamp_buf,
                    int stamp_i0) {
    static bool attr[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !attr[dev]) {
        cudaFuncSetAttribute(gr_down_splitk2_kernel<TT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) (kFusedGrMaxT * SK_KC * sizeof(float)));
        cudaGetLastError();
        attr[dev] = true;
    }
    gr_down_splitk2_kernel<TT><<<dim3(SK2_GROUPS, SK_S), THREADS, (size_t) m.T * SK_KC * sizeof(float), st>>>(m, part, part_ss);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, st);
    gr_up2_kernel<TT, UP2_IT><<<N / (UP2_IT * 4 * WARPS / HC), THREADS, 0, st>>>(m, part, part_ss);
}

}  // namespace

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf,
                         int stamp_i0) {
    static const int variant = [] {
        const char* v = std::getenv("STRATA_GR_SPLITK");
        return v && v[0] == '0' ? 0 : v && v[0] == '1' ? 1 : 2;
    }();
    fused_gr_read_multi_variant(a, n_tok, xn_scratch, stream, variant, stamp_buf, stamp_i0);
}

void fused_gr_read_multi_variant(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, int variant,
                                 unsigned long long* stamp_buf, int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    m.xn = xn_scratch;
    m.T = n_tok;
    cudaStream_t st = (cudaStream_t) stream;
    if (variant == 2) {
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
        float* part = xn_scratch + (size_t) n_tok * D;
        float* part_ss = part + (size_t) SK_S * kFusedGrMaxT * SK_ALL;
        if (n_tok == 1) launch_splitk2<1>(m, part, part_ss, st, stamp_buf, stamp_i0);
        else if (n_tok == 2) launch_splitk2<2>(m, part, part_ss, st, stamp_buf, stamp_i0);
        else if (n_tok <= 4) launch_splitk2<4>(m, part, part_ss, st, stamp_buf, stamp_i0);
        else launch_splitk2<8>(m, part, part_ss, st, stamp_buf, stamp_i0);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi (split-K 2): %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        return;
    }
    if (variant == 1) {
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);   // (no norm kernel: the split keeps the stamp layout)
        float* part = xn_scratch + (size_t) n_tok * D;   // kFusedGrScratchExtra floats past the tokens' xn
        float* part_ss = part + (size_t) SK_S * kFusedGrMaxT * SK_ALL;
        static bool sk_attr[64] = {};
        int sdev = 0;
        cudaGetDevice(&sdev);
        if (sdev >= 0 && sdev < 64 && !sk_attr[sdev]) {
            cudaFuncSetAttribute(gr_down_splitk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int) (kFusedGrMaxT * SK_KC * sizeof(float)));
            cudaGetLastError();
            sk_attr[sdev] = true;
        }
        gr_down_splitk_kernel<<<dim3(SK_GROUPS, SK_S), THREADS, (size_t) n_tok * SK_KC * sizeof(float), st>>>(m, part, part_ss);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        gr_up_multi_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, part_ss);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi (split-K): %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        return;
    }
    gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
    // the shared-memory opt-in is a per-DEVICE setting: once per device, not once per process (a layer split
    // runs this kernel on two cards)
    static bool attr[64] = {};
    static int chunk[64] = {};   // Turing port: tokens the down kernel may carry in one launch on this card
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !attr[dev]) {
        // at most what the card allows (Turing: 64 KB - enough for windows of up to 6 tokens)
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        int want = (int) (kFusedGrMaxT * TILE * sizeof(float));
        if (optin > 0 && want > optin) want = optin;
        cudaFuncSetAttribute(gr_down_multi_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        cudaGetLastError();      // drop any error the attempt left behind
        // Turing port: the down kernel stages n_tok*TILE floats of dynamic shared memory - 80 KB at the full
        // 8 tokens.  A card whose opt-in is below that (Turing: 64 KB, so 7+ tokens fail to launch as
        // "invalid argument") processes the tokens in slices that fit; a card that reports no opt-in gets what
        // fits the 48 KB default (4 tokens of the CUDA tile; all 8 of HIP's smaller tile).  The down kernel's
        // outputs (lo, inject_out) are strictly per-token, so the chunk boundaries are safe, and the up kernel
        // below still sees every token of the batch in one launch.
        const int capacity = (optin > 0 ? optin : 48 * 1024) / (int) (TILE * sizeof(float));
        chunk[dev] = capacity < 1 ? 1 : (capacity > kFusedGrMaxT ? kFusedGrMaxT : capacity);
        attr[dev] = true;
    }
    const int chunk_tok = (dev >= 0 && dev < 64 && chunk[dev]) ? chunk[dev] : kFusedGrMaxT;
    if (chunk_tok >= n_tok) {
        gr_down_multi_kernel<<<DOWN_BLOCKS + 1, THREADS, (size_t) n_tok * TILE * sizeof(float), st>>>(m);
    } else {
        for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
            const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
            GrMulti c{};
            c.xn = xn_scratch + (size_t) c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k) c.a[k] = a[c0 + k];
            gr_down_multi_kernel<<<DOWN_BLOCKS + 1, THREADS, (size_t) ct * TILE * sizeof(float), st>>>(c);
        }
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
    gr_up_multi_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m, nullptr, nullptr);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    gr_down_kernel<<<DOWN_BLOCKS + 1, THREADS, 0, st>>>(a);
    gr_up_kernel<<<UP_BLOCKS, THREADS, 0, st>>>(a);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels

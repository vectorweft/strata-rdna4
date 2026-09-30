// include/strata/kernels/router_rows.cuh - the decode router's pieces as device functions, shared by their own
// kernels (native_bf16.cu: the fast multi-row BF16 GEMV; native_router.cu: softmax + top-10) and the verify window's
// fused router (verify_kernels.cu), so every path computes bitwise the same logits, ids and weights.
#pragma once

#include <cfloat>
#include <cstdint>

namespace strata::kernels::router_rows {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    return v;
}

constexpr int FG_TILE = 1024;
constexpr int FG_ROWS = 8;   // rows per block, one per warp (256 threads)
__device__ __forceinline__ float fg_dot8(const uint4 w, const float* x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = fmaf(__uint_as_float(v[j] << 16), x[2 * j], acc);
        acc = fmaf(__uint_as_float(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}
// rows [blk * FG_ROWS, +FG_ROWS) of y[k][row] = w[row] . x[k] for n_tok <= NT tokens; xs: n_tok * FG_TILE floats of LDS
template <int NT>
__device__ __forceinline__ void bf16_rows_multi(const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
                                                float* __restrict__ y, int64_t ldy, int n_in, int n_out, int n_tok,
                                                float* xs, int blk) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blk * FG_ROWS + warp;
    const bool active = row < n_out;
    const uint4* w4 = reinterpret_cast<const uint4*>(w + (size_t) (active ? row : 0) * n_in);
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int base = 0; base < n_in; base += FG_TILE) {
        const int chunk = min(FG_TILE, n_in - base);
        uint4 wv[FG_TILE / 8 / 32];
#pragma unroll
        for (int q = 0; q < FG_TILE / 8 / 32; ++q) {
            const int j = lane + 32 * q;
            wv[q] = (active && j * 8 < chunk) ? __ldg(w4 + base / 8 + j) : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < n_tok * (chunk / 4); i += blockDim.x) {
            const int k = i / (chunk / 4), off = i - k * (chunk / 4);
            reinterpret_cast<float4*>(xs + k * FG_TILE)[off] = reinterpret_cast<const float4*>(x + (size_t) k * ldx + base)[off];
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < FG_TILE / 8 / 32; ++q) {
            const int j = lane + 32 * q;
            if (j * 8 >= chunk) break;
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (k < n_tok) acc[k] += fg_dot8(wv[q], xs + k * FG_TILE + j * 8);
        }
    }
    if (!active) return;
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        if (k >= n_tok) break;
        const float v = warp_sum(acc[k]);
        if (lane == k) y[(size_t) k * ldy + row] = v;
    }
}

// one warp: softmax over the 512 logits, the top 10 (ties to the lower id) and their renormalized weights
__device__ __forceinline__ void route_top10(const float* logits, int32_t* ids, float* weights, int lane) {
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}

}  // namespace strata::kernels::router_rows

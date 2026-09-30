// include/strata/kernels/gdn_ab_row.cuh - one row of the GDN alpha / beta projection for up to kVerifyMaxT tokens,
// one warp per row (rows [0, h_v) alpha -> gate, [h_v, 2 h_v) beta), shared by gdn_ab_multi and the fused GDN
// front (native_mmvq.cu) so both compute bitwise the same values.
#pragma once

#include "strata/kernels/verify_kernels.hpp"

#include <cstdint>

namespace strata::kernels {

__device__ __forceinline__ void gdn_ab_row(const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                           const uint16_t* __restrict__ wb, const float* __restrict__ dt,
                                           const float* __restrict__ ssm_a, float* __restrict__ gate,
                                           float* __restrict__ beta, int n, int h_v, int T, int row, int lane) {
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    float acc[kVerifyMaxT];
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
#pragma unroll
        for (int t = 0; t < kVerifyMaxT; ++t) {
            if (t >= T) break;
            const float* xt = x + (size_t) t * n;
            const float4 xa = *reinterpret_cast<const float4*>(xt + j * 8);
            const float4 xb = *reinterpret_cast<const float4*>(xt + j * 8 + 4);
            float a = acc[t];
            a = fmaf(__uint_as_float(wv.x << 16), xa.x, a); a = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, a);
            a = fmaf(__uint_as_float(wv.y << 16), xa.z, a); a = fmaf(__uint_as_float(wv.y & 0xffff0000u), xa.w, a);
            a = fmaf(__uint_as_float(wv.z << 16), xb.x, a); a = fmaf(__uint_as_float(wv.z & 0xffff0000u), xb.y, a);
            a = fmaf(__uint_as_float(wv.w << 16), xb.z, a); a = fmaf(__uint_as_float(wv.w & 0xffff0000u), xb.w, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) {
        if (t >= T) break;
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t) t * h_v + r] = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

}  // namespace strata::kernels

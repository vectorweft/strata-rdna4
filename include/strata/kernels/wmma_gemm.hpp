// include/strata/kernels/wmma_gemm.hpp - the prompt path's dense Q8_0 projections on RDNA4 FP16 WMMA (gfx12 only).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// K a multiple of 32 (whole Q8_0 blocks); false on a non-HIP build.
bool wmma_q8_0_gemm_supported(int64_t K);

/// Y[t * ldy + n] = sum_k X[t * ldx + k] * W[n][k] for t < T, n < N, with W the GGUF Q8_0 rows (N rows of K / 32
/// blocks).  FP16 operands (the weights dequantized once per block, the activations rounded as staged), FP32
/// accumulation.  `layout` selects the WMMA result's lane mapping (0, the gfx12 one; 1 its transpose: a test knob).
void wmma_q8_0_gemm(const float* X, int64_t ldx, const void* W, float* Y, int64_t ldy, int64_t T, int64_t N, int64_t K,
                    void* stream, int layout = 0);

}  // namespace strata::kernels

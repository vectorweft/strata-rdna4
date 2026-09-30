// include/strata/kernels/cpu/kq_avx512.hpp - AVX-512 VNNI multi-token dot products for the K-quant GGUF expert
// formats: Q4_K against Q8_K activations, Q5_1 against Q8_1, Q8_0 against Q8_0 (ggml's blocks, unchanged).
// Callers must have checked the CPU for AVX-512 (cpu_avx512_ok); nt is 1..16.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Types kq512_gu_rows takes (gate/up rows).
bool kq512_gu_supported(int ggml_type) noexcept;
/// Types kq512_rows takes as down rows.
bool kq512_down_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void kq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).  Q4_K (n <= 2560), Q5_1 or Q8_0 (n a multiple of 64, <= 1024).
void kq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu

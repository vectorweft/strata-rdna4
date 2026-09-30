// src/kernels/cpu/kq_avx512.cpp - see include/strata/kernels/cpu/kq_avx512.hpp.
//
// Multi-token AVX-512 VNNI kernels for the K-quant GGUF expert formats (Unsloth UD-Q4_K_XL): Q4_K gate/up against
// Q8_K activations, Q5_1 down against Q8_1, Q8_0 down against Q8_0.  Each weight block is loaded and unpacked once
// and multiplied into every token of the window with vpdpbusd (u8 x s8 -> s32); ggml-cpu's vec_dot, which the pool
// used before, re-reads and re-unpacks the row for every token.  The arithmetic is ggml's: the integer sums are
// exact, only the float accumulation order differs (parity is checked against ggml's vec_dot by
// native_expert_parity).
#include "strata/kernels/cpu/kq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

constexpr int kMaxT = 16;   // tokens per call (the pool's windows are at most 8)

inline float h2f(ggml_half h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
// the two fp16 scales every block here starts with (d/dmin, d/m, d/s), read by offset: ggml's C++ declaration
// keeps them in an anonymous union
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}
constexpr int kQ5_1 = 7, kQ8_0 = 8, kQ4_K = 12;   // ggml_type values

// ------------------------------------------------------------------------------------------ Q4_K x Q8_K
// Q4_K superblock: d, dmin, 12 bytes of 6-bit scales/mins, 128 bytes of nibbles.  Bytes [32p, 32p+32) hold
// sub-block 2p in their low nibbles and sub-block 2p+1 in their high nibbles.  A 64-byte load therefore covers two
// pairs: its low nibbles are sub-blocks {2p, 2p+2}, its high nibbles {2p+1, 2p+3}.  The activation is permuted once
// per call into that order, so the inner loop is load, mask/shift, vpdpbusd.
struct ActQ4K {
    int nb = 0;
    alignas(64) int8_t q[kMaxT][10 * 256];      // per superblock: [s0 s2 | s1 s3 | s4 s6 | s5 s7], 32 bytes each
    float d[kMaxT][10];
    int32_t bs[kMaxT][10][8];                   // sub-block sums of the activation codes (for the mins)
};

inline void permute_q8k(const block_q8_K* y, int nb, int8_t* q, float* d, int32_t (*bs)[8]) {
    static const int order[8] = {0, 2, 1, 3, 4, 6, 5, 7};
    for (int b = 0; b < nb; ++b) {
        for (int k = 0; k < 8; ++k) std::memcpy(q + b * 256 + 32 * k, y[b].qs + 32 * order[k], 32);
        d[b] = y[b].d;
        for (int j = 0; j < 8; ++j) bs[b][j] = (int32_t) y[b].bsums[2 * j] + (int32_t) y[b].bsums[2 * j + 1];
    }
}

// ggml's scale/min unpacking (ggml_vec_dot_q4_K_q8_K): 8 six-bit scales then 8 six-bit mins
inline void q4k_scales(const uint8_t* s12, uint8_t sc[8], uint8_t mn[8]) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t u[4];
    std::memcpy(u, s12, 12);
    u[3] = ((u[2] >> 4) & kmask2) | (((u[1] >> 6) & kmask3) << 4);
    const uint32_t aux = u[1] & kmask1;
    u[1] = (u[2] & kmask2) | (((u[0] >> 6) & kmask3) << 4);
    u[2] = aux;
    u[0] &= kmask1;
    std::memcpy(sc, &u[0], 8);
    std::memcpy(mn, &u[2], 8);
}

// one Q4_K row against nt permuted activations: out[t] = row . act[t]
inline void q4k_row(const block_q4_K* x, int nb, const ActQ4K& A, int nt, float* out) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    __m512 facc[kMaxT];
    float mins[kMaxT];
    for (int t = 0; t < nt; ++t) { facc[t] = _mm512_setzero_ps(); mins[t] = 0.f; }
    for (int b = 0; b < nb; ++b) {
        uint8_t sc[8], mn[8];
        q4k_scales(x[b].scales, sc, mn);
        const float dx = half_at(&x[b], 0), dmx = half_at(&x[b], 1);
        const __m512i w0 = _mm512_loadu_si512((const void*) (x[b].qs));
        const __m512i w1 = _mm512_loadu_si512((const void*) (x[b].qs + 64));
        const __m512i l0 = _mm512_and_si512(w0, m4), h0 = _mm512_and_si512(_mm512_srli_epi16(w0, 4), m4);
        const __m512i l1 = _mm512_and_si512(w1, m4), h1 = _mm512_and_si512(_mm512_srli_epi16(w1, 4), m4);
        // per-lane scales: lanes 0-7 the first sub-block of the 64-byte half, lanes 8-15 the second
        const __m512i s_l0 = _mm512_inserti32x8(_mm512_set1_epi32(sc[0]), _mm256_set1_epi32(sc[2]), 1);
        const __m512i s_h0 = _mm512_inserti32x8(_mm512_set1_epi32(sc[1]), _mm256_set1_epi32(sc[3]), 1);
        const __m512i s_l1 = _mm512_inserti32x8(_mm512_set1_epi32(sc[4]), _mm256_set1_epi32(sc[6]), 1);
        const __m512i s_h1 = _mm512_inserti32x8(_mm512_set1_epi32(sc[5]), _mm256_set1_epi32(sc[7]), 1);
        for (int t = 0; t < nt; ++t) {
            const int8_t* q = A.q[t] + b * 256;
            const __m512i d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), l0, _mm512_load_si512((const void*) (q + 0)));
            const __m512i d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), h0, _mm512_load_si512((const void*) (q + 64)));
            const __m512i d2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), l1, _mm512_load_si512((const void*) (q + 128)));
            const __m512i d3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), h1, _mm512_load_si512((const void*) (q + 192)));
            __m512i s = _mm512_mullo_epi32(d0, s_l0);
            s = _mm512_add_epi32(s, _mm512_mullo_epi32(d1, s_h0));
            s = _mm512_add_epi32(s, _mm512_mullo_epi32(d2, s_l1));
            s = _mm512_add_epi32(s, _mm512_mullo_epi32(d3, s_h1));
            const float dy = A.d[t][b];
            facc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_set1_ps(dy * dx), facc[t]);
            const int32_t* bs = A.bs[t][b];
            int32_t ms = 0;
            for (int j = 0; j < 8; ++j) ms += (int32_t) mn[j] * bs[j];
            mins[t] += dy * dmx * (float) ms;
        }
    }
    for (int t = 0; t < nt; ++t) out[t] = _mm512_reduce_add_ps(facc[t]) - mins[t];
}

// ------------------------------------------------------------------------------------------ Q5_1 x Q8_1
// Q5_1 block (32 values): d, m, 32 high bits, 16 bytes of nibbles (low nibbles = values 0-15, high = 16-31).
// Q8_1 block: d, s = d * sum(q), 32 codes.  row . act = sum_b (d_x d_y sum_i x_i y_i + m_x s_y).
struct ActQ81 {
    int nb = 0;
    alignas(64) int8_t q[kMaxT][32 * 32];   // up to 1024 values
    float d[kMaxT][32], s[kMaxT][32];
};

inline void pack_q81(const block_q8_1* y, int nb, int8_t* q, float* d, float* s) {
    for (int b = 0; b < nb; ++b) {
        std::memcpy(q + 32 * b, y[b].qs, 32);
        d[b] = half_at(&y[b], 0);
        s[b] = half_at(&y[b], 1);
    }
}

// the 32 five-bit values of two consecutive Q5_1 blocks as 64 unsigned bytes
inline __m512i q5_1_pair(const block_q5_1* x) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    // [qs_a (16) | qs_a (16) | qs_b (16) | qs_b (16)], low nibbles in the first copy, high in the second
    const __m128i qa = _mm_loadu_si128((const __m128i*) x[0].qs), qb = _mm_loadu_si128((const __m128i*) x[1].qs);
    __m512i v = _mm512_castsi128_si512(qa);
    v = _mm512_inserti32x4(v, _mm_srli_epi16(qa, 4), 1);
    v = _mm512_inserti32x4(v, qb, 2);
    v = _mm512_inserti32x4(v, _mm_srli_epi16(qb, 4), 3);
    v = _mm512_and_si512(v, m4);
    uint32_t ha, hb;
    std::memcpy(&ha, x[0].qh, 4);
    std::memcpy(&hb, x[1].qh, 4);
    const __mmask64 hm = _cvtu64_mask64((uint64_t) ha | ((uint64_t) hb << 32));
    return _mm512_or_si512(v, _mm512_maskz_mov_epi8(hm, _mm512_set1_epi8(0x10)));
}

inline void q5_1_row(const block_q5_1* x, int nb, const ActQ81& A, int nt, float* out) {
    __m512 facc[kMaxT];
    float ms[kMaxT];
    for (int t = 0; t < nt; ++t) { facc[t] = _mm512_setzero_ps(); ms[t] = 0.f; }
    for (int b = 0; b < nb; b += 2) {
        const __m512i w = q5_1_pair(x + b);
        const float da = half_at(&x[b], 0), db = half_at(&x[b + 1], 0), ma = half_at(&x[b], 1), mb = half_at(&x[b + 1], 1);
        for (int t = 0; t < nt; ++t) {
            const __m512i p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w,
                                                  _mm512_load_si512((const void*) (A.q[t] + 32 * b)));
            const __m512 sc = _mm512_insertf32x8(_mm512_set1_ps(da * A.d[t][b]), _mm256_set1_ps(db * A.d[t][b + 1]), 1);
            facc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p), sc, facc[t]);
            ms[t] += ma * A.s[t][b] + mb * A.s[t][b + 1];
        }
    }
    for (int t = 0; t < nt; ++t) out[t] = _mm512_reduce_add_ps(facc[t]) + ms[t];
}

// ------------------------------------------------------------------------------------------ Q8_0 x Q8_0
struct ActQ80 {
    int nb = 0;
    alignas(64) int8_t q[kMaxT][32 * 32];
    float d[kMaxT][32];
};

inline void pack_q80(const block_q8_0* y, int nb, int8_t* q, float* d) {
    for (int b = 0; b < nb; ++b) {
        std::memcpy(q + 32 * b, y[b].qs, 32);
        d[b] = h2f(y[b].d);
    }
}

inline void q8_0_row(const block_q8_0* x, int nb, const ActQ80& A, int nt, float* out) {
    __m512 facc[kMaxT];
    for (int t = 0; t < nt; ++t) facc[t] = _mm512_setzero_ps();
    for (int b = 0; b < nb; b += 2) {
        const __m512i w = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) x[b].qs)),
                                             _mm256_loadu_si256((const __m256i*) x[b + 1].qs), 1);
        // signed x signed on vpdpbusd: |w| as the unsigned operand, the activation negated where w < 0
        const __mmask64 neg = _mm512_movepi8_mask(w);
        const __m512i aw = _mm512_abs_epi8(w);
        const float da = h2f(x[b].d), db = h2f(x[b + 1].d);
        for (int t = 0; t < nt; ++t) {
            const __m512i y = _mm512_load_si512((const void*) (A.q[t] + 32 * b));
            const __m512i ys = _mm512_mask_sub_epi8(y, neg, _mm512_setzero_si512(), y);
            const __m512i p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, ys);
            const __m512 sc = _mm512_insertf32x8(_mm512_set1_ps(da * A.d[t][b]), _mm256_set1_ps(db * A.d[t][b + 1]), 1);
            facc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p), sc, facc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) out[t] = _mm512_reduce_add_ps(facc[t]);
}

}  // namespace

bool kq512_gu_supported(int type) noexcept { return type == kQ4_K; }
bool kq512_down_supported(int type) noexcept { return type == kQ5_1 || type == kQ8_0; }

void kq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    if (type != kQ4_K || n % 256 || n > 2560 || nt < 1 || nt > kMaxT) return;
    thread_local ActQ4K A;
    const int nb = n / 256;
    for (int t = 0; t < nt; ++t) permute_q8k((const block_q8_K*) act[t], nb, A.q[t], A.d[t], A.bs[t]);
    float g[kMaxT], u[kMaxT];
    for (int r = r0; r < r1; ++r) {
        q4k_row((const block_q4_K*) (blob + (size_t) r * gu_row), nb, A, nt, g);
        q4k_row((const block_q4_K*) (blob + up_off + (size_t) r * gu_row), nb, A, nt, u);
        for (int t = 0; t < nt; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

void kq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    if (nt < 1 || nt > kMaxT) return;
    float o[kMaxT];
    if (type == kQ4_K) {
        if (n % 256 || n > 2560) return;
        thread_local ActQ4K A;
        const int nb = n / 256;
        for (int t = 0; t < nt; ++t) permute_q8k((const block_q8_K*) act[t], nb, A.q[t], A.d[t], A.bs[t]);
        for (int r = r0; r < r1; ++r) {
            q4k_row((const block_q4_K*) (w + (size_t) r * row_bytes), nb, A, nt, o);
            for (int t = 0; t < nt; ++t) out[t][r] = o[t];
        }
    } else if (type == kQ5_1) {
        if (n % 64 || n > 1024) return;   // whole block pairs
        thread_local ActQ81 A;
        const int nb = n / 32;
        for (int t = 0; t < nt; ++t) pack_q81((const block_q8_1*) act[t], nb, A.q[t], A.d[t], A.s[t]);
        for (int r = r0; r < r1; ++r) {
            q5_1_row((const block_q5_1*) (w + (size_t) r * row_bytes), nb, A, nt, o);
            for (int t = 0; t < nt; ++t) out[t][r] = o[t];
        }
    } else if (type == kQ8_0) {
        if (n % 64 || n > 1024) return;
        thread_local ActQ80 A;
        const int nb = n / 32;
        for (int t = 0; t < nt; ++t) pack_q80((const block_q8_0*) act[t], nb, A.q[t], A.d[t]);
        for (int r = r0; r < r1; ++r) {
            q8_0_row((const block_q8_0*) (w + (size_t) r * row_bytes), nb, A, nt, o);
            for (int t = 0; t < nt; ++t) out[t][r] = o[t];
        }
    }
}

}  // namespace strata::kernels::cpu

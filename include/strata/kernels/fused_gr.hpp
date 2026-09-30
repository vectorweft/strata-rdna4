// include/strata/kernels/fused_gr.hpp - plan v0.3 P3: the hyper-connection read in TWO kernels, with the
// previous half's write folded in.
//
// The native path spends six kernels per `gr_read` (norm, down MMVF, silu, up MMVF, gate+mean, inject MMVF) and
// one per `gr_write`, 96 + 96 times per token, and its up projection (10240 rows of 320) runs one 160-thread
// block per row: 20.6 us for 6.5 MB.  Here:
//
//   fused_gr_down : R' = R + bo_prev * 2 sigmoid(inj_prev / hc)  (only when `apply`, computed on the fly)
//                   rs[c] = rsqrt(mean(R'[c]^2) + eps),  xn = R' * w_norm * rs
//                   lo[k] = silu((w_down[k] . xn) / hc)          k < hc_lr
//                   inject[c] = w_inject[c] . xn                 when w_inject is given
//   fused_gr_up   : R <- R' in place for this block's columns (when `apply`)
//                   mixed[d] = mean_c  xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo)
//
// FP32 activations and BF16 weights, like the native MMVF contract; the summation order differs from it (G-C
// judges the result).  Geometry is the artifact's: n_embd 2560, hc 4, hc_lr 320.  `inj_prev` and `inject_out`
// must be different buffers (every block reads the former while one block writes the latter).
#pragma once

#include <cstdint>

namespace strata::kernels {

struct FusedGrArgs {
    const float* R = nullptr;          ///< (hc, n_embd), read by `down`; `up` updates it in place when apply
    float* R_out = nullptr;            ///< == R for the in-place update
    bool apply = false;                ///< fold the previous half's gr_write
    const float* bo_prev = nullptr;    ///< that half's block output, n_embd
    const float* inj_prev = nullptr;   ///< that half's injection, hc
    const float* w_norm = nullptr;     ///< (hc * n_embd) f32
    const uint16_t* w_down = nullptr;  ///< bf16 [hc_lr][hc*n_embd]
    const uint16_t* w_up = nullptr;    ///< bf16 [hc*n_embd][hc_lr]
    const uint16_t* w_inject = nullptr;///< bf16 [hc][hc*n_embd], or null (the final mixer)
    /// The Q8_0 source of w_down / w_up (core::WeightRef::hc_q8: [rows][K] int8 then [rows][K / 32] fp16 scales),
    /// read instead of the bf16 copies by the split-K variant when both are given; half the bytes.
    const int8_t* q_down = nullptr;
    const int8_t* q_up = nullptr;
    /// Variant 2 runs down and up as ONE kernel with a grid barrier when the device holds all its blocks at once;
    /// the barrier's counter slot (0-3) must differ between streams that may run this concurrently.  -1: never.
    int bar_slot = 0;
    float eps = 1e-6f;
    float* lo = nullptr;               ///< workspace, hc_lr floats
    float* rs = nullptr;               ///< workspace, hc floats
    float* inject_out = nullptr;       ///< hc floats (when w_inject)
    float* mixed = nullptr;            ///< n_embd
};

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr);
void fused_gr_read(const FusedGrArgs& a, void* stream);

/// Plan v0.3 P6: the same read for up to 8 tokens that share the weights (a verify window): the weights are read
/// once for all of them.  `a[t]` is token t's arguments (its own R, pending write, lo, rs, inject, mixed; the four
/// weight pointers and eps must be the same for every t); `xn_scratch` is n_tok * hc * n_embd floats.  Every
/// token's outputs are bitwise `fused_gr_read(a[t])`.
constexpr int kFusedGrMaxT = 8;
/// Floats `xn_scratch` must hold beyond n_tok * hc * n_embd: the split-K down projection's partial sums
/// (8 K-splits x 8 tokens x (hc_lr + hc) rows) and each split's sum of squares per token.
constexpr int kFusedGrScratchExtra = 8 * kFusedGrMaxT * (320 + 4) + 8 * kFusedGrMaxT;
/// Measures, on the current device, whether the one-kernel read (variant 2 with a grid barrier) has all its blocks
/// resident at once for each token count; until this has run on a device the two-kernel path is used there.  Call
/// outside stream capture (it synchronizes the device).
void fused_gr_prepare();
/// A bf16 [rows][K] view of a hyper-connection matrix held only as Q8_0 planes (core::WeightRef::hc_q8), for the
/// kernels that read bf16: dequantized on `stream` into this device's scratch `slot` (0 or 1; valid until the next
/// call with that slot on the device), bitwise the values the pack's bf16 copy holds.  hc_bf16_reserve() first,
/// outside stream capture.
bool hc_bf16_reserve();
const uint16_t* hc_bf16_from_q8(const void* planes, int64_t rows, int64_t K, int slot, void* stream);
/// Grid barriers of the one-kernel read that timed out on the current device (1000 each + the blocks that had
/// arrived); 0 unless something is wrong.
unsigned fused_gr_barrier_timeouts();
void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf = nullptr, int stamp_i0 = 0);
/// The same with the implementation chosen: 0 = the reference kernels (one warp per down row over all of K; every
/// token bitwise fused_gr_read), 1 = split-K without the norm kernel: the down projection over 168 blocks instead of
/// 41 reads R' * w_norm itself, and the per-stream RMS scale is applied when the up kernel reduces the partials (the
/// projection is linear and a K slice lies in one stream); outputs equal to rounding (1e-7).  2 = variant 1 with
/// register blocking (a down warp applies each activation chunk it reads to 4 rows, an up lane to 4 rows) and
/// T-way butterfly reductions: variant 1 is shared-memory bound at T > 1 (R9700, T = 4: 38 -> 24 us, T = 8: 67 -> 30
/// us); equal to rounding.  fused_gr_read_multi uses 2 unless STRATA_GR_SPLITK=0 or 1.
void fused_gr_read_multi_variant(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, int variant,
                                 unsigned long long* stamp_buf = nullptr, int stamp_i0 = 0);

}  // namespace strata::kernels

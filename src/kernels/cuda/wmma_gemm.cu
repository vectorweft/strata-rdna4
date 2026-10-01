// src/kernels/cuda/wmma_gemm.cu - the prompt path's dense Q8_0 projections on RDNA4's FP16 WMMA (gfx12).
//
// Y[T, N] = X[T, K] . W^T with X in FP32 rows and W as GGUF Q8_0 blocks (34 bytes: an fp16 scale and 32 int8).  The
// weight blocks are dequantized ONCE per tile into shared memory as FP16 (d * q in FP32, rounded once: |err| <= 2^-11
// relative), the activations rounded to FP16 as they are staged, and v_wmma_f32_16x16x16_f16 accumulates in FP32.
// llama.cpp's MMQ (int8 WMMA) instead applies two block scales per 16x16x32 product, which keeps it far from the
// card's peak on these shapes (~42 TOPS on the GDN projections).
//
// Block: 128 tokens x 128 output rows, 8 waves of 64 x 32 (4 x 2 WMMA tiles); K in steps of one Q8_0 block (32),
// shared memory double-buffered, the next step's global loads in flight during this step's WMMAs.
#include "strata/kernels/wmma_gemm.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::kernels {

#if defined(__HIP_PLATFORM_AMD__)
namespace {
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));

constexpr int BM = 128, BN = 128, BK = 32, PAD = 8, LDS_K = BK + PAD;   // 80-byte rows: 16-byte aligned fragments
constexpr int THREADS = 256;

__device__ __forceinline__ float h2f_bits(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }

// one thread's share of a step: A = 16 floats of one token row, B = 16 quants (+ the scale) of one weight row
struct Stage {
    float4 a[4];
    uint4 q;       // 16 int8 (2-byte aligned source: assembled from 8 ushorts)
    float d;
};

__device__ __forceinline__ void load_stage(Stage& s, const float* __restrict__ X, int64_t ldx, const uint8_t* __restrict__ W,
                                           int64_t w_row_bytes, int T, int N, int m0, int n0, int kb, int tid) {
    const int r = tid >> 1, half = tid & 1;
    const int m = m0 + r;
    if (m < T) {
        const float4* src = reinterpret_cast<const float4*>(X + (int64_t) m * ldx + (int64_t) kb * BK + half * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) s.a[i] = src[i];
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) s.a[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    const int n = n0 + r;
    if (n < N) {
        const uint8_t* blk = W + (int64_t) n * w_row_bytes + (int64_t) kb * 34;
        const uint16_t* p16 = reinterpret_cast<const uint16_t*>(blk);
        s.d = h2f_bits(p16[0]);
        const uint16_t* qs = p16 + 1 + half * 8;   // quants at byte 2: 2-byte aligned
        uint32_t w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = (uint32_t) qs[2 * i] | ((uint32_t) qs[2 * i + 1] << 16);
        s.q = make_uint4(w[0], w[1], w[2], w[3]);
    } else {
        s.d = 0.f;
        s.q = make_uint4(0, 0, 0, 0);
    }
}

__device__ __forceinline__ void store_stage(const Stage& s, _Float16 (*As)[LDS_K], _Float16 (*Bs)[LDS_K], int tid) {
    const int r = tid >> 1, half = tid & 1;
    h8 lo, hi;
    const float* af = reinterpret_cast<const float*>(s.a);
#pragma unroll
    for (int i = 0; i < 8; ++i) { lo[i] = (_Float16) af[i]; hi[i] = (_Float16) af[8 + i]; }
    *reinterpret_cast<h8*>(&As[r][half * 16]) = lo;
    *reinterpret_cast<h8*>(&As[r][half * 16 + 8]) = hi;
    const uint32_t qw[4] = {s.q.x, s.q.y, s.q.z, s.q.w};
    h8 b0, b1;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int8_t q0 = (int8_t) ((qw[i / 4] >> (8 * (i % 4))) & 0xff);
        const int8_t q1 = (int8_t) ((qw[2 + i / 4] >> (8 * (i % 4))) & 0xff);
        b0[i] = (_Float16) (s.d * (float) q0);
        b1[i] = (_Float16) (s.d * (float) q1);
    }
    *reinterpret_cast<h8*>(&Bs[r][half * 16]) = b0;
    *reinterpret_cast<h8*>(&Bs[r][half * 16 + 8]) = b1;
}

__global__ void __launch_bounds__(THREADS) wmma_q8_0_gemm_kernel(const float* __restrict__ X, int64_t ldx,
                                                                 const uint8_t* __restrict__ W, float* __restrict__ Y,
                                                                 int64_t ldy, int T, int N, int K, int a_major) {
    __shared__ __align__(16) _Float16 As[2][BM][LDS_K];
    __shared__ __align__(16) _Float16 Bs[2][BN][LDS_K];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int wm = wave & 1, wn = wave >> 1;                  // 2 x 4 waves: 64 tokens x 32 rows each
    const int nb = K / BK;
    const int64_t w_row_bytes = (int64_t) nb * 34;
    f8 acc[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.0f;
    Stage st;
    load_stage(st, X, ldx, W, w_row_bytes, T, N, m0, n0, 0, tid);
    store_stage(st, As[0], Bs[0], tid);
    __syncthreads();
    const int frow = lane & 15, fk = (lane >> 4) * 8;   // a lane's row (A) / column (B) and its 8-wide K half
    for (int kb = 0; kb < nb; ++kb) {
        const int cur = kb & 1;
        if (kb + 1 < nb) load_stage(st, X, ldx, W, w_row_bytes, T, N, m0, n0, kb + 1, tid);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            h8 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const h8*>(&As[cur][wm * 64 + i * 16 + frow][kk + fk]);
#pragma unroll
            for (int j = 0; j < 2; ++j) b[j] = *reinterpret_cast<const h8*>(&Bs[cur][wn * 32 + j * 16 + frow][kk + fk]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
        }
        if (kb + 1 < nb) store_stage(st, As[cur ^ 1], Bs[cur ^ 1], tid);
        __syncthreads();
    }
    // D: a lane holds column (lane & 15) of a 16 x 16 tile, rows (lane >> 4) * 8 + e (a_major = 0), or the transpose
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                int tr, tc;
                if (a_major == 0) { tr = (lane >> 4) * 8 + e; tc = lane & 15; }
                else { tr = lane & 15; tc = (lane >> 4) * 8 + e; }
                const int m = m0 + wm * 64 + i * 16 + tr, n = n0 + wn * 32 + j * 16 + tc;
                if (m < T && n < N) Y[(int64_t) m * ldy + n] = acc[i][j][e];
            }
}
}  // namespace

bool wmma_q8_0_gemm_supported(int64_t K) {
    return K > 0 && K % BK == 0;
}

void wmma_q8_0_gemm(const float* X, int64_t ldx, const void* W, float* Y, int64_t ldy, int64_t T, int64_t N, int64_t K,
                    void* stream, int layout) {
    if (!wmma_q8_0_gemm_supported(K) || T <= 0 || N <= 0 || ldx % 4 != 0)
        throw std::invalid_argument("wmma_q8_0_gemm: unsupported shape");
    const dim3 grid((unsigned) ((N + BN - 1) / BN), (unsigned) ((T + BM - 1) / BM));
    wmma_q8_0_gemm_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(X, ldx, (const uint8_t*) W, Y, ldy, (int) T,
                                                                       (int) N, (int) K, layout);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("wmma_q8_0_gemm: ") + cudaGetErrorString(e));
}
#else
bool wmma_q8_0_gemm_supported(int64_t) { return false; }
void wmma_q8_0_gemm(const float*, int64_t, const void*, float*, int64_t, int64_t, int64_t, int64_t, void*, int) {
    throw std::runtime_error("wmma_q8_0_gemm: HIP (gfx12) only");
}
#endif

}  // namespace strata::kernels

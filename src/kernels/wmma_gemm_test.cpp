// wmma_gemm_test: wmma_q8_0_gemm against an FP64 reference on random data (small shapes), then timed at the prompt
// path's shapes (8192 tokens).
//
//   wmma_gemm_test [layout]
#include "strata/kernels/wmma_gemm.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

static uint16_t f2h(float f) {   // round to nearest even, normal range
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t sgn = (u >> 16) & 0x8000;
    const int32_t e = (int32_t) ((u >> 23) & 0xff) - 127 + 15;
    if (e <= 0) return (uint16_t) sgn;
    if (e >= 31) return (uint16_t) (sgn | 0x7c00);
    uint32_t m = u & 0x7fffff, r = m >> 13, rem = m & 0x1fff;
    uint32_t h = sgn | ((uint32_t) e << 10) | r;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++h;
    return (uint16_t) h;
}
static float h2f(uint16_t h) {
    const uint32_t sgn = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0) { float f = (float) m * (1.0f / 16777216.0f); std::memcpy(&u, &f, 4); u |= sgn; }
    else u = sgn | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}

static std::vector<uint8_t> make_q8(int N, int K, std::mt19937& rng) {
    std::vector<uint8_t> w((size_t) N * (K / 32) * 34);
    std::uniform_real_distribution<float> sd(0.001f, 0.02f);
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < K / 32; ++b) {
            uint8_t* blk = w.data() + ((size_t) n * (K / 32) + b) * 34;
            const uint16_t d = f2h(sd(rng));
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
        }
    return w;
}

int main(int argc, char** argv) {
    const int layout = argc > 1 ? std::atoi(argv[1]) : 0;
    std::mt19937 rng(7);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    // ---- correctness: odd T and N to exercise the edges
    {
        const int T = 200, N = 300, K = 640;
        std::vector<float> X((size_t) T * K);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : X) v = nd(rng);
        const auto W = make_q8(N, K, rng);
        float *dX, *dY; void* dW;
        cudaMalloc((void**) &dX, X.size() * 4); cudaMalloc(&dW, W.size()); cudaMalloc((void**) &dY, (size_t) T * N * 4);
        cudaMemcpy(dX, X.data(), X.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
        k::wmma_q8_0_gemm(dX, K, dW, dY, N, T, N, K, s, layout);
        cudaStreamSynchronize(s);
        std::vector<float> Y((size_t) T * N);
        cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost);
        double max_rel = 0, max_abs = 0, ref_scale = 0;
        for (int t = 0; t < T; ++t)
            for (int n = 0; n < N; ++n) {
                double ref = 0;
                for (int b = 0; b < K / 32; ++b) {
                    const uint8_t* blk = W.data() + ((size_t) n * (K / 32) + b) * 34;
                    uint16_t d; std::memcpy(&d, blk, 2);
                    for (int i = 0; i < 32; ++i) ref += (double) X[(size_t) t * K + b * 32 + i] * h2f(d) * (int8_t) blk[2 + i];
                }
                const double err = std::fabs(Y[(size_t) t * N + n] - ref);
                max_abs = std::max(max_abs, err);
                ref_scale = std::max(ref_scale, std::fabs(ref));
            }
        max_rel = max_abs / (ref_scale + 1e-30);
        std::printf("correctness T=%d N=%d K=%d layout %d: max |err| %.3e (relative to max |ref| %.3e: %.2e) -> %s\n", T, N,
                    K, layout, max_abs, ref_scale, max_rel, max_rel < 5e-3 ? "OK" : "WRONG");
        cudaFree(dX); cudaFree(dW); cudaFree(dY);
        if (max_rel >= 5e-3) return 1;
    }
    // ---- timing at the prompt path's shapes
    struct Shape { const char* name; int N, K; };
    const Shape shapes[] = {{"gdn qkv", 10240, 2560}, {"gdn z", 6144, 2560}, {"gdn out", 2560, 6144},
                            {"qsa q", 12288, 2560}, {"qsa k", 512, 2560}, {"qsa out", 2560, 6144},
                            {"shexp gate", 640, 2560}, {"hc down", 320, 10240}, {"hc up", 10240, 320}};
    const int T = 8192;
    for (const Shape& sh : shapes) {
        float *dX, *dY; void* dW;
        cudaMalloc((void**) &dX, (size_t) T * sh.K * 4); cudaMalloc((void**) &dY, (size_t) T * sh.N * 4);
        cudaMemset(dX, 0, (size_t) T * sh.K * 4);
        const auto W = make_q8(sh.N, sh.K, rng);
        cudaMalloc(&dW, W.size());
        cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
        for (int i = 0; i < 3; ++i) k::wmma_q8_0_gemm(dX, sh.K, dW, dY, sh.N, T, sh.N, sh.K, s, layout);
        cudaStreamSynchronize(s);
        const int it = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i) k::wmma_q8_0_gemm(dX, sh.K, dW, dY, sh.N, T, sh.N, sh.K, s, layout);
        cudaStreamSynchronize(s);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
        const double tf = 2.0 * T * sh.N * sh.K / us / 1e6;
        std::printf("%-11s T=%d N=%5d K=%5d: %8.1f us  %6.1f TFLOPS\n", sh.name, T, sh.N, sh.K, us, tf);
        cudaFree(dX); cudaFree(dY); cudaFree(dW);
    }
    return 0;
}

// mmvq_bench: native_mmvq (Q8_0) at the decode window's real shapes, weights cold (a ring of copies larger than the
// last-level cache), T = 1..6 columns.  Prints us and GB/s of weight bytes.
//
//   mmvq_bench [iterations]
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

static int q4_head(cudaStream_t s) {   // the MTP draft heads: Q4_0, 2560 -> 248320 / 52195 (fits the MALL? no: 341 MB)
    for (int n_out : {248320, 52195}) {
        const int n_in = 2560;
        const size_t wb = (size_t) n_out * (n_in / 32) * 18;
        void* w = nullptr;
        cudaMalloc(&w, wb);
        cudaMemset(w, 0x11, wb);
        float *x = nullptr, *y = nullptr;
        void* xq = nullptr;
        cudaMalloc((void**) &x, 8 * n_in * 4);
        cudaMalloc((void**) &y, (size_t) 8 * n_out * 4);
        cudaMalloc(&xq, k::native_q8_1_bytes(n_in, 8));
        cudaMemset(x, 0, 8 * n_in * 4);
        for (int T : {1, 2, 4}) {
            k::native_quantize_q8_1(x, xq, n_in, T, s);
            for (int i = 0; i < 3; ++i) k::native_mmvq(2, w, xq, y, n_in, n_out, T, s);
            cudaStreamSynchronize(s);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 50; ++i) k::native_mmvq(2, w, xq, y, n_in, n_out, T, s);
            cudaStreamSynchronize(s);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 50;
            std::printf("Q4_0 2560->%d (%.0f MB) T%d: %.1f us %.0f GB/s\n", n_out, wb / 1e6, T, us, wb / us / 1e3);
        }
        cudaFree(w); cudaFree(x); cudaFree(y); cudaFree(xq);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--q4head") {
        cudaStream_t s0;
        cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
        return q4_head(s0);
    }
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    struct Shape { const char* name; int n_in, n_out; };
    const Shape shapes[] = {{"gdn ssm_out 6144->2560", 6144, 2560}, {"qsa attn_out 6144->2560", 6144, 2560},
                            {"gdn qkv 2560->10240", 2560, 10240}, {"qsa attn_q 2560->12288", 2560, 12288},
                            {"shared gate 2560->640", 2560, 640}};
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    std::mt19937 rng(1);
    for (const Shape& sh : shapes) {
        const size_t wbytes = (size_t) sh.n_out * (sh.n_in / 32) * 34;
        const int copies = (int) std::max<size_t>(4, (256ull << 20) / wbytes + 1);   // > the 64 MB MALL
        std::vector<uint8_t> h(wbytes);
        for (size_t i = 0; i < wbytes; ++i) h[i] = (uint8_t) rng();
        for (size_t b = 0; b < wbytes / 34; ++b) { h[b * 34] = 0x00; h[b * 34 + 1] = 0x3c; }   // fp16 scale 1.0
        std::vector<void*> w((size_t) copies);
        for (auto& p : w) { cudaMalloc(&p, wbytes); cudaMemcpy(p, h.data(), wbytes, cudaMemcpyHostToDevice); }
        float* x = nullptr;
        float* y = nullptr;
        void* xq = nullptr;
        cudaMalloc((void**) &x, (size_t) 8 * sh.n_in * 4);
        cudaMalloc((void**) &y, (size_t) 8 * sh.n_out * 4);
        cudaMalloc(&xq, k::native_q8_1_bytes(sh.n_in, 8));
        std::vector<float> hx((size_t) 8 * sh.n_in);
        for (auto& v : hx) v = (float) (rng() % 2000) / 1000.0f - 1.0f;
        cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
        std::printf("%s (%.1f MB):", sh.name, wbytes / 1e6);
        for (int T = 1; T <= 6; ++T) {
            k::native_quantize_q8_1(x, xq, sh.n_in, T, s);
            for (int i = 0; i < copies; ++i) k::native_mmvq(8, w[(size_t) i], xq, y, sh.n_in, sh.n_out, T, s);
            cudaStreamSynchronize(s);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) k::native_mmvq(8, w[(size_t) (i % copies)], xq, y, sh.n_in, sh.n_out, T, s);
            cudaStreamSynchronize(s);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
            std::printf("  T%d %.1f us %.0f GB/s", T, us, wbytes / us / 1e3);
        }
        std::printf("\n");
        const int cfgs[][2] = {{4, 1}, {8, 1}, {16, 1}, {4, 2}, {8, 2}};
        for (const auto& c : cfgs) {
            std::printf("    nw %2d rows %d:", c[0], c[1]);
            for (int T = 1; T <= 6; ++T) {
                k::native_quantize_q8_1(x, xq, sh.n_in, T, s);
                for (int i = 0; i < copies; ++i) k::native_q8_0_mmvq_cfg(w[(size_t) i], xq, y, sh.n_in, sh.n_out, T, c[0], c[1], s);
                cudaStreamSynchronize(s);
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i)
                    k::native_q8_0_mmvq_cfg(w[(size_t) (i % copies)], xq, y, sh.n_in, sh.n_out, T, c[0], c[1], s);
                cudaStreamSynchronize(s);
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
                std::printf("  T%d %.1f", T, us);
            }
            std::printf(" us\n");
        }
        for (auto& p : w) cudaFree(p);
        cudaFree(x); cudaFree(y); cudaFree(xq);
    }
    return 0;
}

// gr_bench: fused_gr_read_multi timed at the artifact's geometry for 1..8 tokens, and the split-K variant compared
// against the reference one (STRATA_GR_SPLITK selects the variant inside the library; this tool calls both through
// fused_gr_read_multi_variant).
//
//   gr_bench [iterations]
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;

uint16_t bf16(float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) ((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }

template <typename T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc((void**) &d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& v : wd) v = bf16(nd(rng) * 0.01f);
    for (auto& v : wu) v = bf16(nd(rng) * 0.05f);
    for (auto& v : wi) v = bf16(nd(rng) * 0.01f);
    std::vector<float> wn(D), bo(N), ip(HC);
    for (auto& v : wn) v = 1.f + 0.1f * nd(rng);
    for (auto& v : bo) v = nd(rng) * 0.1f;
    for (auto& v : ip) v = nd(rng);
    const uint16_t *d_wd = dev(wd), *d_wu = dev(wu), *d_wi = dev(wi);
    const float *d_wn = dev(wn), *d_bo = dev(bo), *d_ip = dev(ip);
    float* scratch = nullptr;
    cudaMalloc((void**) &scratch, (size_t) k::kFusedGrMaxT * D * sizeof(float) * 4);
    cudaStream_t s;
    cudaStreamCreate(&s);

    int failures = 0;
    for (int T : {1, 2, 4, 6, 8}) {
        std::vector<float> R((size_t) T * D);
        for (auto& v : R) v = nd(rng);
        // two independent output sets: variant 0 (reference) and variant 1 (split-K)
        std::vector<float> out[3][4];
        double us[3] = {0, 0, 0};
        for (int var = 0; var < 3; ++var) {
            float* d_R = dev(R);
            float *lo, *rs, *inj, *mixed;
            cudaMalloc((void**) &lo, (size_t) T * LR * 4);
            cudaMalloc((void**) &rs, (size_t) T * HC * 4);
            cudaMalloc((void**) &inj, (size_t) T * HC * 4);
            cudaMalloc((void**) &mixed, (size_t) T * N * 4);
            k::FusedGrArgs a[k::kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                a[t].R = d_R + (size_t) t * D; a[t].R_out = d_R + (size_t) t * D; a[t].apply = false;
                a[t].bo_prev = d_bo; a[t].inj_prev = d_ip;
                a[t].w_norm = d_wn; a[t].w_down = d_wd; a[t].w_up = d_wu; a[t].w_inject = d_wi; a[t].eps = 1e-6f;
                a[t].lo = lo + t * LR; a[t].rs = rs + t * HC; a[t].inject_out = inj + t * HC; a[t].mixed = mixed + t * N;
            }
            k::fused_gr_read_multi_variant(a, T, scratch, s, var);
            cudaStreamSynchronize(s);
            auto get = [&](float* p, size_t n) { std::vector<float> h(n); cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost); return h; };
            out[var][0] = get(lo, (size_t) T * LR);
            out[var][1] = get(inj, (size_t) T * HC);
            out[var][2] = get(mixed, (size_t) T * N);
            out[var][3] = get(rs, (size_t) T * HC);
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) k::fused_gr_read_multi_variant(a, T, scratch, s, var);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            us[var] = 1000.0 * ms / iters;
            cudaFree(d_R); cudaFree(lo); cudaFree(rs); cudaFree(inj); cudaFree(mixed);
        }
        {   // per-kernel split of one call (gpu stamps: before, after norm, after down, after up), both variants
            unsigned long long* st = nullptr;
            cudaMalloc((void**) &st, 8 * 8);
            float* d_R = dev(R);
            float *lo, *rs, *inj, *mixed;
            cudaMalloc((void**) &lo, (size_t) T * LR * 4); cudaMalloc((void**) &rs, (size_t) T * HC * 4);
            cudaMalloc((void**) &inj, (size_t) T * HC * 4); cudaMalloc((void**) &mixed, (size_t) T * N * 4);
            k::FusedGrArgs a[k::kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                a[t].R = d_R + (size_t) t * D; a[t].R_out = d_R + (size_t) t * D;
                a[t].w_norm = d_wn; a[t].w_down = d_wd; a[t].w_up = d_wu; a[t].w_inject = d_wi;
                a[t].lo = lo + t * LR; a[t].rs = rs + t * HC; a[t].inject_out = inj + t * HC; a[t].mixed = mixed + t * N;
            }
            for (int var = 0; var < 3; ++var) {
                double acc[3] = {0, 0, 0};
                const int reps = 200;
                for (int i = 0; i < reps; ++i) {
                    k::gpu_stamp(st, 0, s);
                    k::fused_gr_read_multi_variant(a, T, scratch, s, var, st, 1);
                    k::gpu_stamp(st, 3, s);
                    cudaStreamSynchronize(s);
                    unsigned long long h[4];
                    cudaMemcpy(h, st, sizeof h, cudaMemcpyDeviceToHost);
                    for (int j = 0; j < 3; ++j) acc[j] += (double) (h[j + 1] - h[j]) / 1e3;
                }
                std::printf("  T=%d %s: norm %.1f us  down %.1f us  up %.1f us\n", T, var == 2 ? "split-K 2" : var ? "split-K  " : "reference",
                            acc[0] / reps, acc[1] / reps, acc[2] / reps);
            }
            cudaFree(d_R); cudaFree(lo); cudaFree(rs); cudaFree(inj); cudaFree(mixed); cudaFree(st);
        }
        const char* names[4] = {"lo", "inject", "mixed", "rs"};
        double worst = 0;
        for (int v = 1; v < 3; ++v)
        for (int o = 0; o < 4; ++o) {
            double n = 0, d = 0;
            for (size_t i = 0; i < out[0][o].size(); ++i) { n += std::fabs(out[v][o][i] - out[0][o][i]); d += std::fabs(out[0][o][i]); }
            const double r = n / (d + 1e-30);
            if (r > worst) worst = r;
            if (r > 1e-5) { std::printf("  T=%d var %d %s rel %.2e MISMATCH\n", T, v, names[o], r); ++failures; }
        }
        std::printf("T=%d  reference %.1f us  split-K %.1f us  split-K 2 %.1f us  (13.1 MB of weights -> %.0f / %.0f / %.0f GB/s)  worst rel %.1e\n",
                    T, us[0], us[1], us[2], 13100.0 / us[0], 13100.0 / us[1], 13100.0 / us[2], worst);
    }
    return failures ? 1 : 0;
}

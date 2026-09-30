// gguf_kernel_check: the GPU dequantizers and MMVQ against ggml's CPU reference, on a real model's tensors.
//
//   gguf_kernel_check <any shard of a split model.gguf> [tensor ...]
//
// For each named tensor (default: the embedding, the head, the PLE table and a few projections of every format
// in the file set) the first rows are dequantized on the GPU (iq_dequant_f32) and by ggml's to_float, and a
// Q8_1 activation is pushed through native_mmvq and through ggml's vec_dot.  Prints the relative L1 difference.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <vector>

namespace {

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: gguf_kernel_check <shard.gguf> [tensor ...]\n"); return 2; }
    ggml_cpu_init();
    std::vector<std::unique_ptr<strata::GgufFile>> shards;
    const std::string first = argv[1];
    std::smatch m;
    const std::regex split("-(\\d{5})-of-(\\d{5})\\.gguf$");
    if (std::regex_search(first, m, split)) {
        const int total = std::stoi(m[2].str());
        const std::string stem = first.substr(0, (size_t) m.position(0));
        for (int i = 1; i <= total; ++i) {
            char tail[32];
            std::snprintf(tail, sizeof tail, "-%05d-of-%05d.gguf", i, total);
            shards.push_back(std::make_unique<strata::GgufFile>(stem + tail));
        }
    } else {
        shards.push_back(std::make_unique<strata::GgufFile>(first));
    }
    std::vector<std::string> names;
    for (int i = 2; i < argc; ++i) names.push_back(argv[i]);
    if (names.empty())
        names = {"token_embd.weight", "output.weight", "per_layer_token_embd.weight", "blk.0.attn_qkv.weight",
                 "blk.0.ssm_out.weight", "blk.3.attn_q.weight", "blk.0.ffn_gate_exps.weight",
                 "blk.0.ffn_down_exps.weight", "blk.2.ffn_gate_exps.weight", "blk.2.ffn_down_exps.weight"};
    cudaStream_t s;
    cudaStreamCreate(&s);
    int failures = 0;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const std::string& name : names) {
        const strata::TensorInfo* t = nullptr;
        const strata::GgufFile* owner = nullptr;
        for (const auto& g : shards)
            if (const auto* x = g->find(name)) { t = x; owner = g.get(); }
        if (!t) { std::printf("%-32s absent\n", name.c_str()); continue; }
        const int type = (int) t->type;
        const int64_t cols = (int64_t) t->shape[0];
        const int64_t rows = std::min<int64_t>(64, (int64_t) (t->shape.size() > 1 ? t->shape[1] : 1));
        const size_t row_bytes = ggml_row_size((ggml_type) type, cols);
        const uint8_t* src = owner->tensor_data(*t);
        // (a) dequantization
        std::vector<float> ref((size_t) (rows * cols)), got(ref.size());
        const auto* tr = ggml_get_type_traits((ggml_type) type);
        for (int64_t r = 0; r < rows; ++r) tr->to_float(src + r * row_bytes, ref.data() + r * cols, cols);
        double dq = -1;
        if (strata::kernels::iq_supported(type) && (rows * cols) % 256 == 0) {
            void* dw; float* dy;
            cudaMalloc(&dw, rows * row_bytes);
            cudaMalloc((void**) &dy, ref.size() * 4);
            cudaMemcpy(dw, src, rows * row_bytes, cudaMemcpyHostToDevice);
            strata::kernels::iq_dequant_f32(type, dw, rows * cols, dy, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost);
            dq = rel(got, ref);
            cudaFree(dw); cudaFree(dy);
        }
        // (b) MMVQ: y = W x with x random, GPU native_mmvq vs float reference over the dequantized rows
        double mv = -1;
        if (strata::kernels::native_mmvq_supported(type) && cols % 256 == 0) {
            std::vector<float> x((size_t) cols), yref((size_t) rows), y((size_t) rows);
            for (auto& v : x) v = nd(rng);
            for (int64_t r = 0; r < rows; ++r) {
                double acc = 0;
                for (int64_t c = 0; c < cols; ++c) acc += (double) ref[(size_t) (r * cols + c)] * x[(size_t) c];
                yref[(size_t) r] = (float) acc;
            }
            void *dw, *dq8; float *dx, *dy;
            cudaMalloc(&dw, rows * row_bytes);
            cudaMalloc((void**) &dx, cols * 4);
            cudaMalloc((void**) &dy, rows * 4);
            cudaMalloc(&dq8, strata::kernels::native_q8_1_bytes((int) cols, 1));
            cudaMemcpy(dw, src, rows * row_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(dx, x.data(), cols * 4, cudaMemcpyHostToDevice);
            strata::kernels::native_quantize_q8_1(dx, dq8, (int) cols, 1, s);
            strata::kernels::native_mmvq(type, dw, dq8, dy, (int) cols, (int) rows, 1, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(y.data(), dy, rows * 4, cudaMemcpyDeviceToHost);
            mv = rel(y, yref);
            cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dq8);
        }
        const bool bad = dq > 1e-3 || mv > 3e-2;
        failures += bad;
        std::printf("%-32s %-7s %6lld x %-6lld  dequant rel %s  mmvq rel %s  %s\n", name.c_str(), ggml_type_name((ggml_type) type),
                    (long long) rows, (long long) cols,
                    dq < 0 ? "   n/a  " : (std::to_string(dq).substr(0, 8)).c_str(),
                    mv < 0 ? "   n/a  " : (std::to_string(mv).substr(0, 8)).c_str(), bad ? "BAD" : "ok");
    }
    return failures ? 1 : 0;
}

#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <set>

namespace strata::core {
namespace {
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS and IQ4_XS. Other keys retain the packed BF16 fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// The hyper-connection projections: bf16 in the canonical pack (the prompt path's GEMMs), Q8_0 in the GGUF.  The
// decode window reads the Q8_0 bytes instead - half the traffic, and the source values exactly.
bool hc_q8_eligible(const strata::TensorInfo& tensor) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0 || tensor.type != 8 || tensor.shape.size() != 2 || tensor.shape[0] % 32 != 0)
        return false;
    static const char* suffixes[] = {".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_ffn_down.weight",
                                     ".hc_ffn_up.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
bool hc_q8_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_HC_Q8"); return !(v && v[0] == '0'); }();
    return on;
}
struct DeviceFree { void operator()(void* p) const { if (p) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
};
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if ((eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                     tensor.shape.size() == 2) || (hc_q8_enabled() && hc_q8_eligible(tensor)))
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

NativeDense::~NativeDense() {
    if (scratch_) cudaFree(scratch_);
    for (void* p : weights_) cudaFree(p);
}

bool NativeDense::in_range(const std::string& name, int64_t lb, int64_t le) {
    if (name.compare(0, 4, "blk.") != 0) return true;
    const int64_t layer = std::atoll(name.c_str() + 4);
    return layer >= lb && layer < le;
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key) {
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture")) {
                err = strata::check_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key)) continue;
                if (!in_range(tensor.name, lb_, le_)) continue;   // another layer-split stage's layer
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, bytes);
                DevicePtr data(allocation);
                if (status == cudaSuccess)
                    status = cudaMemcpy(data.get(), gguf.tensor_data(tensor), bytes, cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status); return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data)});
            }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        // the hyper-connection Q8_0 planes: [rows][K] int8, then [rows][K / 32] fp16 scales (16-byte aligned rows)
        std::vector<std::pair<WeightRef*, void*>> hc_pending;
        for (const auto& path : shards) {
            if (!hc_q8_enabled()) break;
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors()) {
                if (!hc_q8_eligible(tensor) || !in_range(tensor.name, lb_, le_)) continue;
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) continue;
                auto& ref = found->second;
                const uint64_t K = tensor.shape[0], rows = tensor.shape[1], nb = K / 32;
                if (ref.hc_q8 || (int64_t) K != ref.ne0 || (int64_t) rows != ref.ne1 || rows * K > 10240ull * 320ull) {
                    err = "native dense: incompatible hyper-connection matrix " + tensor.name; return false;
                }
                std::vector<uint8_t> host(rows * K + rows * nb * 2);
                const uint8_t* src = gguf.tensor_data(tensor);
                int8_t* qs = (int8_t*) host.data();
                uint8_t* ds = host.data() + rows * K;
                for (uint64_t r = 0; r < rows; ++r)
                    for (uint64_t b = 0; b < nb; ++b) {
                        const uint8_t* blk = src + (r * nb + b) * 34;
                        std::memcpy(ds + (r * nb + b) * 2, blk, 2);
                        std::memcpy(qs + r * K + b * 32, blk + 2, 32);
                    }
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, host.size());
                if (status == cudaSuccess)
                    status = cudaMemcpy(allocation, host.data(), host.size(), cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    if (allocation) cudaFree(allocation);
                    for (auto& h : hc_pending) cudaFree(h.second);
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status); return false;
                }
                total += host.size();
                hc_pending.emplace_back(&ref, allocation);
            }
        }
        void* allocation = nullptr;
        const auto status = cudaMalloc(&allocation, strata::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation);
        if (status != cudaSuccess) { err = std::string("native dense scratch: ") + cudaGetErrorString(status); return false; }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            weights_.push_back(item.data.release());
        }
        for (auto& h : hc_pending) {
            h.first->hc_q8 = h.second;
            weights_.push_back(h.second);
        }
        if (!hc_pending.empty() && !strata::kernels::hc_bf16_reserve()) {
            err = "native dense: the hyper-connection bf16 scratch"; return false;
        }
        scratch_ = scratch.release();
        bytes_ = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
} // namespace strata::core

#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::core {
class WeightTable;

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Owns one Q8_1 scratch vector shared by all
// these projections, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false);
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err);
    /// Only the layers [lb, le) are uploaded (a layer-split stage's own layers); tensors outside any layer
    /// (blk.* prefix absent) always are.  Call before load.
    void set_layer_range(int64_t lb, int64_t le) { lb_ = lb; le_ = le; }
    static bool in_range(const std::string& name, int64_t lb, int64_t le);
    uint64_t weight_bytes() const { return bytes_; }
    size_t tensor_count() const { return weights_.size(); }

private:
    std::vector<void*> weights_;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    int64_t lb_ = 0, le_ = INT64_MAX;
};
} // namespace strata::core

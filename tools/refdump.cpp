// refdump: run a prompt through llama.cpp once and write chosen intermediate tensors in full, as the reference
// the Strata port is bisected against.
//
//   refdump <model.gguf> <token ids file> <out dir> [regex ...]
//
// Every tensor whose name matches one of the regexes (default: l_last, ffn_moe_topk, hc_init, result_output,
// model.input_embed, ple_embd) is written as <out dir>/<name>.bin: a header of 4 int64 ne[] and a type code
// (0 = f32, 1 = i32), then the data in ggml order.  Extra llama.cpp arguments come from the environment:
// REFDUMP_OT="ffn_(up|gate|down)_exps=CPU" keeps the experts on the CPU.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Ctx {
    std::string out;
    FILE* trace = nullptr;   // REFDUMP_TRACE: ffn_moe_topk-<l> appended as Strata --dump-routing records
    std::vector<std::regex> keep;
    std::vector<uint8_t> buf;
    int written = 0;
};

bool cb(ggml_tensor * t, bool ask, void * user) {
    auto * c = (Ctx *) user;
    bool match = false;
    for (const auto & r : c->keep) match = match || std::regex_search(t->name, r);
    if (ask) return match;
    if (!match || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32)) return true;
    if (c->trace && std::strncmp(t->name, "ffn_moe_topk-", 13) == 0 && t->type == GGML_TYPE_I32) {
        const int layer = std::atoi(t->name + 13);
        const int k = (int) t->ne[0];
        std::vector<int32_t> ids((size_t) k);
        std::vector<float> zero((size_t) k, 0.f);
        // one read of the whole strided region (the view's rows are nb[1] apart), then pick the ids out
        const size_t span = (size_t) ((t->ne[1] - 1) * t->nb[1] + k * t->nb[0]);
        c->buf.resize(span);
        ggml_backend_tensor_get(t, c->buf.data(), 0, span);
        for (int64_t tok = 0; tok < t->ne[1]; ++tok) {
            for (int i = 0; i < k; ++i)
                std::memcpy(&ids[(size_t) i], c->buf.data() + tok * t->nb[1] + i * t->nb[0], 4);
            const int32_t rec[2] = {layer, k};
            std::fwrite(rec, sizeof rec, 1, c->trace);
            std::fwrite(ids.data(), 4, (size_t) k, c->trace);
            std::fwrite(zero.data(), 4, (size_t) k, c->trace);
        }
        return true;
    }
    const size_t n = ggml_nbytes(t);
    c->buf.resize(n);
    // views may be non-contiguous; copy row by row through the backend when needed
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, c->buf.data(), 0, n);
    } else {
        std::vector<uint8_t> row;
        const size_t rb = (size_t) t->ne[0] * ggml_type_size(t->type);
        c->buf.resize(rb * t->ne[1] * t->ne[2] * t->ne[3]);
        size_t o = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                    const size_t off = i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    if (t->nb[0] == ggml_type_size(t->type)) {
                        ggml_backend_tensor_get(t, c->buf.data() + o, off, rb);
                    } else {
                        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0)
                            ggml_backend_tensor_get(t, c->buf.data() + o + i0 * ggml_type_size(t->type),
                                                    off + i0 * t->nb[0], ggml_type_size(t->type));
                    }
                    o += rb;
                }
    }
    std::string name = t->name;
    for (char & ch : name) if (ch == ' ' || ch == '(' || ch == ')' || ch == '/') ch = '_';
    std::ofstream f(c->out + "/" + name + ".bin", std::ios::binary);
    const int64_t hdr[5] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->type == GGML_TYPE_I32 ? 1 : 0};
    f.write((const char *) hdr, sizeof hdr);
    f.write((const char *) c->buf.data(), (std::streamsize) c->buf.size());
    ++c->written;
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: refdump <model.gguf> <token ids file> <out dir> [regex ...]\n");
        return 2;
    }
    Ctx c;
    c.out = argv[3];
    std::vector<std::string> pats;
    for (int i = 4; i < argc; ++i) pats.push_back(argv[i]);
    if (pats.empty()) pats = {"^l_last-", "^ffn_moe_topk-", "^hc_init", "^result_output", "^model.input_embed", "^ple_embd"};
    for (const auto & p : pats) c.keep.emplace_back(p);

    std::vector<llama_token> toks;
    {
        std::ifstream f(argv[2]);
        std::string s((std::istreambuf_iterator<char>(f)), {});
        for (char & ch : s) if (ch == ',') ch = ' ';
        std::istringstream is(s);
        long long v;
        while (is >> v) toks.push_back((llama_token) v);
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    std::vector<llama_model_tensor_buft_override> ot;
    const char * ot_env = std::getenv("REFDUMP_OT");
    std::string ot_pat = ot_env ? ot_env : "";
    if (!ot_pat.empty()) {
        const size_t eq = ot_pat.find('=');
        static std::string pattern = ot_pat.substr(0, eq);
        ot.push_back({pattern.c_str(), ggml_backend_cpu_buffer_type()});
        ot.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = ot.data();
    }
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "refdump: cannot load %s\n", argv[1]); return 1; }
    if (const char* tr = std::getenv("REFDUMP_TRACE")) c.trace = std::fopen(tr, "wb");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) toks.size() + 64;
    cp.n_batch = 2048;
    cp.n_ubatch = 2048;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = &c;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "refdump: no context\n"); return 1; }
    for (size_t at = 0; at < toks.size(); at += 2048) {   // in batch-sized pieces
        const int32_t n = (int32_t) std::min<size_t>(2048, toks.size() - at);
        // REFDUMP_ALL_OUTPUTS=1: every token is an output, so llama.cpp computes the LAST layer for every row too
        // (with one output per batch it gathers the output rows before that layer's FFN, and a routing trace then
        // holds 1 record of layer n-1 per batch: a profile built from it leaves the last layer's experts uncached)
        static const bool all_out = std::getenv("REFDUMP_ALL_OUTPUTS") != nullptr;
        llama_batch b = llama_batch_get_one(toks.data() + at, n);
        llama_batch ba{};
        if (all_out) {
            ba = llama_batch_init(n, 0, 1);
            for (int32_t i = 0; i < n; ++i) {
                ba.token[i] = toks[at + i]; ba.pos[i] = (llama_pos) (at + i); ba.n_seq_id[i] = 1;
                ba.seq_id[i][0] = 0; ba.logits[i] = 1;
            }
            ba.n_tokens = n;
            b = ba;
        }
        const int rc = llama_decode(ctx, b);
        if (all_out) llama_batch_free(ba);
        if (rc != 0) { std::fprintf(stderr, "refdump: decode failed\n"); return 1; }
    }
    if (c.trace) std::fclose(c.trace);
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
    std::printf("refdump: %zu tokens, %d tensors written to %s, next token %d\n", toks.size(), c.written, c.out.c_str(), best);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

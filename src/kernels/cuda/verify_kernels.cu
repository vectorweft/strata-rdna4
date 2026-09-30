// src/kernels/cuda/verify_kernels.cu - see include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/gdn_ab_row.cuh"
#include "strata/kernels/router_rows.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void __launch_bounds__(S) gdn_conv_l2_multi_kernel(const float* __restrict__ hist,
                                                              const float* __restrict__ qkv,
                                                              const float* __restrict__ w, float* __restrict__ h,
                                                              int C, int qk_heads, float eps, int t_begin) {
    __shared__ float part[S / 32];
    const int t = t_begin + blockIdx.y;
    const int c = blockIdx.x * S + threadIdx.x;
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

__global__ void gdn_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                                       const int32_t* __restrict__ n_keep, size_t hist_stride, size_t qkv_stride) {
    hist += blockIdx.y * hist_stride;   // (blockIdx.y: the layer of a batched commit)
    qkv += blockIdx.y * qkv_stride;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

__global__ void __launch_bounds__(256) gdn_ab_multi_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                                           const uint16_t* __restrict__ wb,
                                                           const float* __restrict__ dt,
                                                           const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                           float* __restrict__ beta, int n, int h_v, int T) {
    gdn_ab_row(x, wa, wb, dt, ssm_a, gate, beta, n, h_v, T, blockIdx.x * 8 + (threadIdx.x >> 5), threadIdx.x & 31);
}

__global__ void __launch_bounds__(S * RG) gdn_step_norm_multi_kernel(float* __restrict__ state,
                                                                     const float* __restrict__ hbuf, int C,
                                                                     const float* __restrict__ gate,
                                                                     const float* __restrict__ beta,
                                                                     const float* __restrict__ z,
                                                                     const float* __restrict__ gamma, float eps,
                                                                     float* __restrict__ y, int h_k, int h_v, int T,
                                                                     const int32_t* __restrict__ n_keep, int t_out_begin,
                                                                     size_t state_stride, size_t h_stride, size_t gb_stride) {
    state += blockIdx.y * state_stride;   // (blockIdx.y: the layer of a batched commit)
    hbuf += blockIdx.y * h_stride;
    gate += blockIdx.y * gb_stride;
    beta += blockIdx.y * gb_stride;
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();                // the previous token is done with sk/sq/red/wsum
        if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        if (t < t_out_begin) continue;   // a replayed token: its state update is needed, its output is not
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            y[(size_t) t * value_dim + head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
        }
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

__global__ void embedding_gather_dev_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                            const float* __restrict__ offsets, const int32_t* __restrict__ tokens,
                                            int64_t n, int code_bits, int code_bias, int group_elems,
                                            unsigned long long row_codes, unsigned long long row_groups,
                                            float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = __fmul_rn((float) (code + code_bias), sc[group]);
    out[(size_t) t * n + i] = __fadd_rn(product, of ? of[group] : 0.0f);
}

__global__ void broadcast_streams_kernel(const float* __restrict__ x, float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

__global__ void copy_indexed_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t stride,
                                    const int32_t* __restrict__ index, int64_t n) {
    const int idx = *index;
    if (idx < 0) return;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[(size_t) idx * stride + i];
}

__global__ void fetch_blobs_kernel(const unsigned long long* __restrict__ src, const int32_t* __restrict__ n,
                                   uint4* __restrict__ dst, long long per) {
    const long long total = (long long) *n * per;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const uint4*) src[k])[off];
    }
}

__global__ void rebase_ptrs_kernel(unsigned long long* ptr, const int32_t* n, unsigned long long base, long long bytes) {
    const int k = threadIdx.x;
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

__global__ void add_streams_broadcast_kernel(const float* __restrict__ h, const float* __restrict__ e,
                                             float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

__global__ void ident_hits_kernel(const int32_t* __restrict__ ids, int n, int32_t* __restrict__ slot,
                                  int32_t* __restrict__ dst, int32_t* __restrict__ count) {
    const int i = threadIdx.x;
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template<typename E>
__global__ void gather_rows_kernel(const E* __restrict__ src, long long row_e, const int32_t* __restrict__ ids,
                                   long long n, E* __restrict__ dst) {
    const long long total = n * row_e;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__global__ void map_ids_kernel(int32_t* ids, const int32_t* __restrict__ table, int n) {
    const int i = threadIdx.x;
    if (i < n) ids[i] = table[ids[i]];
}

__global__ void row_top_prob_kernel(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ ids,
                                    float* __restrict__ probs) {
    __shared__ float part[32];
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_vocab; i += blockDim.x) s += __expf(l[i] - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) tot += part[w];
        probs[t] = 1.0f / tot;
    }
}

__global__ void mtp_select_kernel(const float* __restrict__ R_src, int64_t stride, const int32_t* __restrict__ ids,
                                  const int32_t* __restrict__ row_dev, float* __restrict__ R_dst,
                                  int32_t* __restrict__ tok_dst, int32_t* out, int j, const float* probs, float* out_p) {
    const int row = *row_dev;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += (int64_t) gridDim.x * blockDim.x)
        R_dst[i] = R_src[(size_t) row * stride + i];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
    }
}

__global__ void dense_steps_kernel(const int32_t* __restrict__ cells, int n, int32_t* __restrict__ steps) {
    const int i = threadIdx.x;
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    fetch_blobs_kernel<<<48 * 8, 256, 0, (cudaStream_t) stream>>>(src, n, (uint4*) dst, (long long) (blob_bytes / 16));
    check("fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    rebase_ptrs_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ptr, n, (unsigned long long) base, (long long) blob_bytes);
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    add_streams_broadcast_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                                   (cudaStream_t) stream>>>(h, e, R, n_embd, hc);
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    ident_hits_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(ids, n, slot, dst, count);
    check("ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    mtp_select_kernel<<<16, 256, 0, (cudaStream_t) stream>>>(R_src, R_stride, ids, row_dev, R_dst, tok_dst, out, j,
                                                             probs, out_p);
    check("mtp_select");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    if (row_bytes % 16 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint4*) src, row_bytes / 16, ids, n, (uint4*) dst);
    else if (row_bytes % 4 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint32_t*) src, row_bytes / 4, ids, n, (uint32_t*) dst);
    else
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>(src, row_bytes, ids, n, dst);
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    map_ids_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(ids, table, n);
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    row_top_prob_kernel<<<n_rows, 1024, 0, (cudaStream_t) stream>>>(logits, n_vocab, ids, probs);
    check("row_top_prob");
}

namespace {
__global__ void window_ids_kernel(int32_t* steps, int window, int32_t* ids, long long stride) {
    const int q = blockIdx.y;
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x)
        ids[q * stride + j] = start + j;
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) st[3] = width;
}
}  // namespace

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    window_ids_kernel<<<dim3(8, (unsigned) n), 256, 0, (cudaStream_t) stream>>>(steps, window, ids, (long long) ids_stride);
    check("window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    dense_steps_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(cells, n, steps);
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    gdn_conv_l2_multi_kernel<<<dim3((unsigned) (channels / S), (unsigned) n_tok), S, 0, (cudaStream_t) stream>>>(
        history, qkv, conv_w, h, channels, qk_heads, eps, t_begin);
    check("gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    gdn_conv_commit_kernel<<<(unsigned) ((channels + 255) / 256), 256, 0, (cudaStream_t) stream>>>(history, qkv,
                                                                                                 channels, n_keep, 0, 0);
    check("gdn_conv_commit");
}

void gdn_conv_commit_layers(float* history, size_t history_stride, const float* qkv, size_t qkv_stride, int channels,
                            int n_layers, const int32_t* n_keep, void* stream) {
    if (n_layers <= 0) return;
    gdn_conv_commit_kernel<<<dim3((unsigned) ((channels + 255) / 256), (unsigned) n_layers), 256, 0,
                             (cudaStream_t) stream>>>(history, qkv, channels, n_keep, history_stride, qkv_stride);
    check("gdn_conv_commit_layers");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    gdn_ab_multi_kernel<<<(unsigned) ((2 * h_v + 7) / 8), 256, 0, (cudaStream_t) stream>>>(
        x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    check("gdn_ab_multi");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    gdn_step_norm_multi_kernel<<<(unsigned) h_v, dim3(S, RG), 0, (cudaStream_t) stream>>>(
        state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, n_keep, t_out_begin, 0, 0, 0);
    check("gdn_step_norm_multi");
}

void gdn_step_commit_layers(float* state, size_t state_stride, const float* h, size_t h_stride, int conv_channels,
                            const float* gate, const float* beta, size_t gb_stride, int h_k, int h_v, int n_tok,
                            int n_layers, const int32_t* n_keep, void* stream) {
    if (n_layers <= 0) return;
    if (!state || !h || !gate || !beta || !n_keep || h_k <= 0 || h_v % h_k || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_commit_layers: invalid arguments\n");
        std::exit(1);
    }
    // t_out_begin = n_tok: every token is a replay, no outputs (z, gamma, y are never read or written)
    gdn_step_norm_multi_kernel<<<dim3((unsigned) h_v, (unsigned) n_layers), dim3(S, RG), 0, (cudaStream_t) stream>>>(
        state, h, conv_channels, gate, beta, h, gate, 1e-6f, nullptr, h_k, h_v, n_tok, n_keep, n_tok, state_stride,
        h_stride, gb_stride);
    check("gdn_step_commit_layers");
}

namespace {
__global__ void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t value) {
    while (*flag < value) __nanosleep(100);
    __threadfence_system();
}
}  // namespace

namespace {
__device__ __forceinline__ void resident_plan_one(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                                  int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                                  long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                                  uint32_t ring) {
    // one thread: at most kVerifyMaxT * 10 entries, the host's exact loop
    for (int i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert || res[e] < 0) { *skip = 0; return; }
    }
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;
    int groups = 0, entries = 0;
    for (int i0 = 0; i0 < n; ++i0) {
        bool first = true;
        for (int j = 0; j < i0; ++j) if (ids[j] == ids[i0]) { first = false; break; }
        if (!first) continue;
        const int32_t slot = res[ids[i0]];
        ptr[groups] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        start[groups] = entries;
        for (int i = i0; i < n; ++i)
            if (ids[i] == ids[i0]) {
                // an entry belongs to i0's group when its first occurrence is i0: the same expert id
                dst[entries] = i;
                tok[entries] = i / k;
                ++entries;
            }
        ++groups;
    }
    start[groups] = entries;
    start2[0] = entries;
    counts[0] = groups;
    counts[1] = entries;
    counts[2] = 0;
    __threadfence();
    *skip = ring;
}
__global__ void resident_plan_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                     int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                     long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                     uint32_t ring) {
    resident_plan_one(ids, n, k, res, n_expert, cache_base, slot_off, blob, pl, capx, skip, ring);
}

// ---- the verify window's router in ONE launch (router_fused): 64 blocks of 8 rows run the 512 x n_embd BF16 GEMV
// (router_rows::bf16_rows_multi, the fast multi-row GEMV's arithmetic) and copy a slice of the activation to the
// host's doorbell rows; the LAST block to finish (a counter, no spinning) runs the top-10 of every token (one warp
// each, router_rows::route_top10), the device plan and the doorbell - what the router, top-10, resident_plan and
// doorbell_publish kernels did as four launches, bitwise.
__device__ unsigned g_router_done;
__device__ __forceinline__ void doorbell_store_sys(uint32_t* seq, uint32_t v) {
#if defined(__HIP_PLATFORM_AMD__)
    __hip_atomic_store(seq, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    *(volatile uint32_t*) seq = v;
#endif
}
template <int NT>
__global__ void __launch_bounds__(256) router_fused_kernel(RouterFusedArgs a) {
    extern __shared__ __align__(16) float xs[];   // [n_tok][FG_TILE]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int n_x = a.n_tok * a.n_embd;
    {   // this block's slice of the doorbell activation rows (host-mapped)
        const int per = ((n_x / 4 + (int) gridDim.x - 1) / (int) gridDim.x) * 4;
        const int b0 = (int) blockIdx.x * per, b1 = min(n_x, b0 + per);
        for (int i = b0 + 4 * t; i < b1; i += 4 * (int) blockDim.x)
            *reinterpret_cast<float4*>(a.x_out + i) = *reinterpret_cast<const float4*>(a.x + i);
    }
    router_rows::bf16_rows_multi<NT>(a.x, a.n_embd, a.w, a.logits, 512, a.n_embd, 512, a.n_tok, xs, blockIdx.x);
    __shared__ bool last;
    __syncthreads();
    if (t == 0) {
        __threadfence_system();   // the logits (device) and the activation slice (host) before the arrival
        last = atomicAdd(&g_router_done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (warp < a.n_tok) router_rows::route_top10(a.logits + warp * 512, a.ids + warp * 10, a.weights + warp * 10, lane);
    __syncthreads();
    if (t == 0 && a.plan)
        resident_plan_one(a.ids, a.n_tok * 10, 10, a.res_layer, a.n_expert, a.cache_base, a.slot_off, a.blob, a.plan,
                          a.capx, a.skip, a.ring);
    for (int i = t; i < a.n_tok * 10; i += (int) blockDim.x) { a.ids_out[i] = a.ids[i]; a.w_out[i] = a.weights[i]; }
    __threadfence_system();
    __syncthreads();
    if (t == 0) {
        __threadfence_system();
        doorbell_store_sys(a.seq, *(volatile uint32_t*) a.seq + 1u);
        __hip_atomic_store(&g_router_done, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}
__global__ void wait_flag_ge_or_kernel(const volatile uint32_t* flag, uint32_t value, const volatile uint32_t* skip) {
    if (*skip == value) return;
    while (*flag < value) __nanosleep(100);
    __threadfence_system();
}
__global__ void copy_i32_unless_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n,
                                       const uint32_t* skip, uint32_t value) {
    if (*skip == value) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
__global__ void copy_or_zero_kernel(float4* __restrict__ dst, const volatile float4* src, long long n4,
                                    const uint32_t* skip, uint32_t value) {
    const bool zero = *skip == value;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long) gridDim.x * blockDim.x)
        dst[i] = zero ? make_float4(0.f, 0.f, 0.f, 0.f) : const_cast<const float4*>(src)[i];
}
}  // namespace

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream) {
    resident_plan_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ids, n_entries, k, res_layer, n_expert, cache_base, slot_off,
                                                             blob, plan, capx, skip, ring);
    check("resident_plan");
}
bool router_fused(const RouterFusedArgs& a, void* stream) {
    static const bool on = [] { const char* v = std::getenv("STRATA_ROUTER_FUSED"); return !(v && v[0] == '0'); }();
    auto al = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; };
    if (!on || a.n_tok < 2 || a.n_tok > kVerifyMaxT || a.n_embd % 8 != 0 || !al(a.x) || !al(a.w) || !al(a.x_out) ||
        !a.logits || !a.ids || !a.weights || !a.ids_out || !a.w_out || !a.seq)
        return false;
    const unsigned blocks = 512 / router_rows::FG_ROWS;   // 64
    const size_t lds = (size_t) a.n_tok * router_rows::FG_TILE * sizeof(float);
    if (a.n_tok <= 4) router_fused_kernel<4><<<blocks, 256, lds, (cudaStream_t) stream>>>(a);
    else router_fused_kernel<8><<<blocks, 256, lds, (cudaStream_t) stream>>>(a);
    check("router_fused");
    return true;
}

void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    wait_flag_ge_or_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value, skip);
    check("wait_flag_ge_or");
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    copy_i32_unless_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(dst, (const volatile int32_t*) src, (int) n, skip, value);
    check("copy_i32_from_mapped_unless");
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    copy_or_zero_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src, n4, skip,
                                                                    value);
    check("copy_or_zero_from_mapped");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    wait_flag_ge_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value);
    check("wait_flag_ge");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    embedding_gather_dev_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) n_tok), 256, 0,
                                  (cudaStream_t) stream>>>(codes, scales, offsets, tokens, n, code_bits, code_bias,
                                                           group_elems, row_codes, row_groups, out);
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    broadcast_streams_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                               (cudaStream_t) stream>>>(x, R, n_embd, hc);
    check("broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    copy_indexed_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(dst, src, stride, index, n);
    check("copy_indexed");
}

// a GPU timestamp (ns, %globaltimer) into buf[i] - the verify window's stage profiler
namespace { __global__ void gpu_stamp_kernel(unsigned long long* buf, int i) {
    unsigned long long t;
#if defined(__HIPCC__)
    t = wall_clock64() * 10ull;   // gfx11: a constant 100 MHz counter, in ns
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
    buf[i] = t;
} }
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    gpu_stamp_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(buf, i);
}

}  // namespace strata::kernels

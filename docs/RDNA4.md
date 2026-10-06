# Qwen3.8-Flash-Next UD-Q4_K_XL on AMD RDNA4 (strata-rdna4)

strata-rdna4, built on [Strata](https://github.com/Niko1221/Strata), runs Unsloth's 4-bit Qwen3.8-Flash-Next (`UD-Q4_K_XL`) on AMD Radeon RDNA4 cards (gfx1201: Radeon AI PRO
R9700, RX 9070 / 9070 XT; gfx1200: RX 9060 XT), on one card or two. Against upstream Strata 0.1.40.1 on the same PC
it reads prompts 2.1-2.8x faster on one card and 2.9-3.6x on two, and writes 1.06-1.21x / 1.53-1.92x faster
([measurements](../bench/results/2026-10-06-fork-vs-upstream/README.md)).

What it changes, briefly: RDNA4 WMMA kernels for the prompt path's dense projections and attention, K-quant expert
products through MMQ with a per-expert pointer table, a second GPU as a helper expert cache that also computes its
own experts while a prompt is read, fused per-layer decode kernels, MTP draft heads requantized to Q4_0 with a full
first head and a smaller later-steps head, an expert profile built from this model's own routing, and page-locked RAM
for the experts no GPU holds.

Measured on: 2x Radeon AI PRO R9700 (32 GB, PCIe 5.0 x8 each), Ryzen 9 9950X, 96 GB DDR5-6000, Ubuntu with ROCm 7.1.
Other RDNA4 cards, other RAM sizes and Windows are untested.

## What you need

| | |
| --- | --- |
| GPU | one or two RDNA4 cards; the configurations below fill a 32 GB card's VRAM with experts (smaller cards work, with fewer experts cached) |
| RAM | 96 GB measured. The engine page-locks ~21 GB of experts and keeps up to 16 GB of parked conversations (`--conversation-cache-mib`); the rest of the experts are read through the OS file cache. With 64 GB, lower `--conversation-cache-mib` (e.g. 4096) and expect slower output |
| Disk | ~192 GB: the four GGUF shards (111.3 GB), the pack with `experts.bin` (77 GB + 1.4 GB), the MTP head (2.8 GB) and its runtime files (0.8 GB). An NVMe SSD |
| Software | ROCm with a clang that targets gfx12 (measured: ROCm 7.1 from Ubuntu's archive, clang 21; older ROCm untested), CMake 3.24+, Ninja, git, Python 3.10+ |

## 1. llama.cpp (ggml and gguf-py) and Python

The engine takes ggml from llama.cpp revision `3cf0325`, and the packers use its gguf-py. One checkout serves both:

```sh
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
python3 -m venv .venv
.venv/bin/pip install numpy jinja2 regex pyyaml requests pillow psutil huggingface_hub
```

The build never modifies that checkout: the one llama.cpp file this fork changes ships patched in
`third_party/ggml-mmq` (see its README).

## 2. Build the engine

```sh
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=/usr/lib/llvm-21/bin/clang++ \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
cmake --build build-hip -j
```

- `gfx1200` for an RX 9060 XT. `CMAKE_HIP_COMPILER`: your ROCm's clang (`/opt/rocm/llvm/bin/clang++` for AMD's
  packages).
- Without `STRATA_GGML_DIR`, CMake clones the pinned revision itself (silent for several minutes).
- Optional: `ctest --test-dir build-hip -E '^(ple_parity|platform_memory_test)$'` (the two excluded tests need the
  model's files).

## 3. Download the model and the MTP head

```sh
.venv/bin/hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" --local-dir models
.venv/bin/hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" --local-dir models
```

Upstream's [docs/UNSLOTH_Q4.md](https://github.com/Niko1221/Strata/blob/main/docs/UNSLOTH_Q4.md#the-files) lists the
shards' sizes and SHA-256 sums (revision `38bb39e`).

## 4. Pack

```sh
.venv/bin/python tools/iq_pack.py --gguf models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --out packs/ud-q4_k_xl --compat-bf16 --experts-bin
```

`--experts-bin` writes the routed experts once more as `experts.bin` (77 GB, a few minutes): this engine reads them
from there. `--compat-bf16` converts the 195 small Q8_0 projections the engine reads as BF16.

## 5. The MTP draft layer

```sh
.venv/bin/python tools/mtp_rt_from_gguf.py --gguf models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --out mtp/rt
.venv/bin/python tools/mtp_rt_steps.py --rt mtp/rt --out mtp/rt-steps52
```

The second step adds a full-vocabulary head for the first draft and a 52K-token head for the later drafts
(`data/draft_vocab_steps52.bin`); the engine requantizes both to Q4_0 on the GPU.

## 6. Start the server

```sh
cp configs/rdna4-ud-q4_k_xl-2gpu.example.json strata-rdna4.json     # or ...-1gpu.example.json for one card
.venv/bin/python -m serve.server --engine strata --config strata-rdna4.json --port 8080
```

Run it from the repository root (the config's paths are relative to it). The first start takes a minute or two (it
copies ~21 GB of experts into page-locked RAM). Then: OpenAI `http://127.0.0.1:8080/v1/chat/completions`, Anthropic
`/v1/messages`, a web chat at `/`.

In the config:

- `HIP_VISIBLE_DEVICES` (in `env`): the cards to use, as `rocm-smi` numbers them. With two: the first runs every layer,
  the second is the helper. Use cards on CPU PCIe lanes: the helper exchanges data with the first card every layer.
- `--vram-reserve-mib 384`: VRAM left free on the first card. Raise it (e.g. 1024) if that card also drives a display.
- `--max-context 262144`: the context. A smaller one leaves more VRAM for experts.
- `STRATA_HELPER_RESERVE_MIB 1536`: the helper's room for the prompt path's 8192-token chunks (`--prefill 8192`).
- Listening on the network: `--host 0.0.0.0`, and set `"api_key"` in the config.

Server behaviour worth knowing:

- Requests without sampling fields (most agent frameworks send none) get the model file's recommended sampling
  (temperature 1.0, top-k 20, top-p 0.95), not greedy decoding. A `"sampling"` block in the config overrides it;
  `"sampling": {}` keeps greedy.
- If the model ends its turn inside its thinking (no `</think>`), the server closes the thinking and lets it write
  the answer, once per request, instead of returning an empty answer. `"close_thinking_on_eos": false` turns it off.
- One request runs at a time; up to four conversations stay parked in RAM and continue without reading their history
  again.
- Other programs that use a lot of RAM next to it (another model server) can push the engine's memory into swap; a
  request then stalls, and after 60 s without progress the engine stops itself and the server starts it again. Its log
  then holds a stall report.

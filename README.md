# strata-rdna4

**AMD RDNA4 work on top of [Strata](https://github.com/Niko1221/Strata) by Niko1221.** Strata runs Qwen3.8-Flash-Next,
a 125-billion-parameter mixture-of-experts model, on a desktop PC. This repository adds kernels, a second-GPU mode and
tuning for Radeon RDNA4 cards and for Unsloth's 4-bit `UD-Q4_K_XL` quantization.

The engine is Strata's: its expert tiers, CPU/GPU expert pool, MTP speculative decoding, conversation cache and the
OpenAI/Anthropic server are Niko1221's and the Strata contributors' work. Everything here is offered back to upstream
Strata. Until it is merged there (or if it is not), RDNA4 owners can use it from this repository.

## Results

Same PC, same model files, same prompts, against upstream Strata 0.1.40.1 configured by its own setup. 2x Radeon AI
PRO R9700 (32 GB each, PCIe 5.0 x8), Ryzen 9 9950X, 96 GB DDR5-6000, Ubuntu with ROCm 7.1. Upstream's speed method:
a fresh code-agent prompt per length, 256 generated tokens, greedy, MTP on, median of 3 runs.

| Tokens/s | Prompt 4K | Prompt 32K | Prompt 128K | Output at 4K | Output at 32K | Output at 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream Strata 0.1.40.1, one GPU | 454 | 515 | 493 | 52.9 | 51.5 | 51.1 |
| strata-rdna4, one GPU | 955 | 1,422 | 1,162 | 60.2 | 61.8 | 54.4 |
| strata-rdna4, two GPUs | 1,523 | 1,826 | 1,415 | 88.6 | 84.7 | 77.9 |

Upstream runs on one GPU on this PC (its two-GPU mode needs ~135 GB of RAM). Time to the first token at 128K: 260 s
upstream, 110 s on one GPU, 91 s on two. Every run, the prompts and the runner:
[bench/results/2026-10-06-fork-vs-upstream](bench/results/2026-10-06-fork-vs-upstream/README.md).

## What this adds to Strata

- **RDNA4 (gfx1201 / gfx1200) prompt path:** dense Q8_0 projections on the cards' FP16 WMMA units, prompt attention
  on WMMA (the tensor-core kernel's algorithm, which the HIP build had compiled out), K-quant expert products through
  llama.cpp's MMQ with a per-expert pointer table that reads weights in place.
- **A second GPU as a helper:** it holds the experts the first card has no room for, and during prompt reading it
  computes them itself (activations cross PCIe instead of weights) plus a share of the experts in page-locked RAM.
- **Decode:** fused per-layer kernels (router, shared expert, hyper-connection read, GDN front, batched commit),
  long-row Q8_0 GEMV, MTP draft heads requantized to Q4_0 with a full first head and a smaller later-steps head,
  an expert profile ranked from this model's own routing, the adaptive expert tier working with pinned RAM experts.
- **Unsloth UD-Q4_K_XL:** native Q4_K / Q5_K / Q5_1 / Q8_0 expert paths, the MTP head from Unsloth's GGUF.
- **Server behaviour for agents:** the model file's recommended sampling instead of greedy when a client sends none,
  and an answer instead of an empty turn when the model stops inside its thinking.

## Get started

**[docs/RDNA4.md](docs/RDNA4.md)**: requirements, build, model download, packing and the server, with example
configurations for one and two cards.

This repository is tested on the hardware above with `UD-Q4_K_XL`. For NVIDIA cards, other quantizations and
Strata's one-click setup, use [upstream Strata](https://github.com/Niko1221/Strata), whose
[README](https://github.com/Niko1221/Strata#readme) and [docs](docs/) cover the engine in depth.

## License and credits

MIT, as upstream ([LICENSE](LICENSE)).

- [Strata](https://github.com/Niko1221/Strata): Niko1221 and the Strata contributors, the engine this builds on.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT): ggml and the MMQ kernels
  ([third_party/ggml-mmq](third_party/ggml-mmq/README.md)).
- [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF): the `UD-Q4_K_XL` quantization and the MTP head.
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): the model.

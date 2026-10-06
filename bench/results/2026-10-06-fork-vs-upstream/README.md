# This fork against upstream Strata 0.1.40.1 (UD-Q4_K_XL, AMD R9700)

Measured 2026-10-06 on one machine, with the same model files, the same prompts and the same runner for every engine.
The method is upstream's published speed method (`bench/results/*-speed-*`): one fresh code-agent prompt per length,
256 generated tokens, greedy, MTP `--spec 4`, numbers from the engine's own timing lines (`PP` / `DONE`). Each cell is
the **median of 3 runs** (every run is in the JSON files).

## Machine

| | |
| --- | --- |
| GPUs | 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32 GB each), each on PCIe 5.0 x8 to the CPU. A third R9700 (chipset x4) ran another program's model server and was not used. |
| CPU / RAM | Ryzen 9 9950X, 96 GB DDR5-6000 (the OS reports 91.2 GiB) |
| Software | Ubuntu, Linux 7.2.6, ROCm 7.1 from the Ubuntu archive (hipcc 7.1.1), both engines built from source for gfx1201 against llama.cpp `3cf0325` (the ggml pin of both) |
| Model | [unsloth/Qwen3.8-Flash-Next-GGUF `UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), the four shards upstream's `docs/UNSLOTH_Q4.md` lists |
| Other load | a llama.cpp server with another model was running (idle) on the third GPU during all runs |

## Configurations

- **Upstream 1 GPU** - upstream Strata **v0.1.40.1** (`82f46a8`), with exactly the arguments its `setup.py` writes for
  this PC and a 135,168-token context: one GPU (its two-GPU layer split needs ~135 GB of RAM), `--kv int8`,
  `--kv-resident 32768` (KV streaming from 64K), `--resident-budget-gib 65` (= `resident_budget_gib(UD-Q4_K_XL,
  91.2, 1.86)`), its expert profile, `--prefill auto`, `--spec 4 --spec-min-p 0.5`, its MTP draft layer (`mtp/rt`).
  Pack made with its own `tools/iq_pack.py --compat-bf16`.
  - Upstream cannot use the second GPU on this PC: the RAM-budget mode refuses a helper expert cache
    (`--expert-cache-device1 auto` -> "`--resident-cpu-experts does not support remote expert caches`").
- **Fork 2 GPU** - this fork as `qwen-serve.sh chat` runs it (`configs/qwen-chat.json`): GPU0 runs every layer, GPU1
  is a helper expert cache that also computes its experts during prompt reading, 262,144-token context, `--kv int8`,
  pinned RAM for the rest of the experts, `--adapt-every 4`, `--spec 4 --spec-min-p 0.3`, MTP draft heads requantized
  to Q4_0 (`rt-steps52`), the expert profile built from this model's own routing.
- **Fork 1 GPU** - the same without the helper card (GPU0 + CPU only): the like-for-like counterpart of upstream.

Every request is read fresh (`--prompt-cache 0`; `reused` is 0 in every run).

## Prompt reading (tokens/s, median of 3)

| Prompt | Upstream 1 GPU | Fork 1 GPU | Fork 2 GPU | Fork 1 GPU / upstream | Fork 2 GPU / upstream |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 235 | 271 | 284 | 1.16x | 1.21x |
| 4K | 454 | 955 | 1,523 | 2.10x | 3.36x |
| 32K | 515 | 1,422 | 1,826 | 2.76x | 3.55x |
| 64K | 508 | 1,351 | 1,705 | 2.66x | 3.36x |
| 128K | 493 | 1,162 | 1,415 | 2.36x | 2.87x |

## Output (tokens/s, median of 3)

| Prompt | Upstream 1 GPU | Fork 1 GPU | Fork 2 GPU | Fork 1 GPU / upstream | Fork 2 GPU / upstream |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 53.6 | 61.9 | 93.1 | 1.15x | 1.74x |
| 4K | 52.9 | 60.2 | 88.6 | 1.14x | 1.67x |
| 32K | 51.5 | 61.8 | 84.7 | 1.20x | 1.65x |
| 64K | 52.0 | 63.1 | 100.0 | 1.21x | 1.92x |
| 128K | 51.1 | 54.4 | 77.9 | 1.06x | 1.53x |

## Time to first token (seconds, median of 3)

| Prompt | Upstream 1 GPU | Fork 1 GPU | Fork 2 GPU |
| --- | ---: | ---: | ---: |
| 1K | 4.3 | 3.7 | 3.5 |
| 4K | 8.8 | 4.2 | 2.6 |
| 32K | 62.2 | 22.5 | 17.6 |
| 64K | 126.1 | 47.4 | 37.6 |
| 128K | 259.5 | 110.2 | 90.5 |

## Reading the numbers

- **Prompt reading** is where most of the difference is, on one GPU as well: 2.1-2.8x from 4K up. The fork reads dense
  Q8_0 projections with its own RDNA4 FP16 WMMA GEMM, runs prompt attention on WMMA, and groups the K-quant experts
  through MMQ with a per-expert pointer table, and lets the second card compute its own experts while a prompt is
  read. Upstream's `docs/UNSLOTH_Q4.md` says its prompt kernels for this file's Q4_K / Q5_K experts are NVIDIA-only
  and that it has not been run on AMD cards.
- **Output on one GPU** is close: the fork is 6-21% faster. Most of the fork's output gain comes from the second card,
  which upstream cannot use on a 96 GB PC.
- Output speed moves with the text, through the share of accepted drafts (`spec_accept` in the JSON). One run can land
  several percent either way: the fork's 2-GPU 32K runs gave 48.7, 84.7 and 101.1 tokens/s.
- 1K prompts are dominated by fixed per-request costs (graph setup, the first window): neither engine reaches its
  bulk speed there.
- The two engines use different MTP settings (`--spec-min-p` 0.5 vs 0.3, different draft heads). Each runs its own
  recommended configuration, so the output column compares the engines as shipped, not one kernel against another.

## Failures

One upstream attempt failed and was run again (it is not in the medians; its partial runs are in
`upstream-0.1.40.1-1gpu-failed-attempt.json`). In its third repetition, the 4K prompt read at 75 tokens/s (453-456 in
the other runs), then the 32K prompt stopped with `ERR verify: timed out at layer 4; its GPU waits were released but the
GPU did not finish within 5 s (#267)` and the engine exited. The rerun of that repetition finished normally. The fork had
no failed run.

## Files

- `fork-2gpu.json`, `fork-1gpu.json`, `upstream-0.1.40.1-1gpu.json`: every run. Fields: `prompt_tokens`, `reused`,
  `prefill_tok_s` (prompt tokens / the engine's prompt time), `decoded`, `decode_tok_s` (generated tokens / the engine's
  decode time), `spec_accept` (accepted / offered drafts), `ttft_s` (from sending the request to the first token, at the
  engine's stdin/stdout), `wall_s`, and the raw `DONE` line.
- `make_bench_prompts.py`: builds the five prompts with the model's chat template and tokenizer: a coding-agent system
  prompt, two tools, upstream Strata's own C++ sources (v0.1.31's `src/`, MIT) as `read_file` results, then a request
  to explain a function and write a unit test without calling more tools. Thinking on (the template's default). No
  repeated filler. `prompts.sha256` holds the hashes of the token-id files used.
- `bench_matrix.py`: the runner (one `strata --serve` process per configuration; machine-specific paths and GPU ids).

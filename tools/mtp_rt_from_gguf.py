"""tools/mtp_rt_from_gguf.py - the MTP draft layer's runtime files from a llama.cpp-format MTP head GGUF.

    python tools/mtp_rt_from_gguf.py --gguf mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --out mtp/rt [--eh-order embedding-first|hidden-first]

Unsloth ships the Qwen3.8-Flash-Next MTP block as a GGUF in llama.cpp's naming (`blk.48.nextn.*`, `blk.48.hc_*`,
`blk.48.attn_*`, `blk.48.ffn_*`), not the BF16 checkpoint tools/mtp_fetch.py + tools/mtp_pack.py start from. This
writes the same three files tools/mtp_rt.py writes (experts.bin, dense.bin, dense.txt), with these differences:

  - The source tensors are dequantized from the GGUF (Q8_0 / BF16 / F32), not read as BF16.
  - RMSNorm weights are copied as stored: llama.cpp's converter already folds GemmaRMSNorm's 1 + w into them
    (the main model's output_hc_norm averages 3.75 in the same file family), which is the form dense.txt wants.
  - `nextn.eh_proj` [2560 x 5120] is llama.cpp's concatenation of fc_embedding and fc_hidden along the input axis;
    it is split at 2560 on a Q8_0 block boundary, so each half re-quantizes to the same Q8_0 bytes. --eh-order says
    which half is which (default: embedding first, Qwen3-Next's concat order).
  - The indexer tensors are not written: the drafter runs dense attention over its own cells.
  - The routed experts are re-quantized from Q8_0 to the engine's MTP blob format (Q2_0 by default, as
    tools/mtp_pack.py does); drafts only steer speed, the verifying model decides every token.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

from mtp_pack import q2_0  # noqa: E402
from mtp_rt import BLOB, FF, H, NE, Q8, blob_of, q8_0  # noqa: E402

# llama.cpp name (after the blk.<n>. prefix) -> dense.txt name
NAMES = {
    "nextn.enorm.weight": "pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "pre_fc_norm_hidden.weight",
    "nextn.hc_head_norm.weight": "hyper_connection_mixer.hc_norm.weight",
    "nextn.hc_head_down.weight": "hyper_connection_mixer.input_mix_weight_down.weight",
    "nextn.hc_head_up.weight": "hyper_connection_mixer.input_mix_weight_up.weight",
    "hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
    "hc_attn_down.weight": "attn_hyper_connection.input_mix_weight_down.weight",
    "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up.weight",
    "hc_attn_inject.weight": "attn_hyper_connection.block_inject_weight.weight",
    "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
    "hc_ffn_down.weight": "mlp_hyper_connection.input_mix_weight_down.weight",
    "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up.weight",
    "hc_ffn_inject.weight": "mlp_hyper_connection.block_inject_weight.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
}
SKIP = ("indexer.",)
EXPERTS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")


def to_f32(t) -> np.ndarray:
    data = np.asarray(t.data)
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        return data.astype(np.float32)
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        u16 = data.view(np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape)
    return quants.dequantize(data, t.tensor_type).astype(np.float32)


def bf16_bytes(x: np.ndarray) -> bytes:
    """float32 -> BF16 with round-to-nearest-even (ggml's conversion)."""
    u = x.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16).tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eh-order", choices=("embedding-first", "hidden-first"), default="embedding-first",
                    help="which half of nextn.eh_proj's input axis is fc_embedding")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = gguf.GGUFReader(a.gguf)
    n_layer = int(r.fields["qwen4exp.block_count"].contents())
    prefix = "blk.%d." % (n_layer - 1)
    tens = {t.name[len(prefix):]: t for t in r.tensors if t.name.startswith(prefix)}

    # ---- the routed experts: Q8_0 -> float -> the engine's Q2_0 blob, one expert at a time
    g, u, d = (np.asarray(tens[n].data) for n in EXPERTS)
    tg, tu, td = (tens[n].tensor_type for n in EXPERTS)
    with open(out / "experts.bin", "wb") as f:
        for e in range(NE):
            gate = quants.dequantize(g[e], tg).reshape(FF, H)
            up = quants.dequantize(u[e], tu).reshape(FF, H)
            down = quants.dequantize(d[e], td).reshape(H, FF)
            gu = q2_0(np.concatenate([gate, up], axis=0)).reshape(2 * FF, H // 64 * 18)
            dn = q2_0(down).reshape(H, FF // 64 * 18)
            f.write(blob_of(gu, dn))
            if e % 128 == 0:
                print("  experts %3d/%d" % (e, NE), flush=True)

    # ---- the dense tensors
    dense: list[tuple[str, np.ndarray, bool]] = []   # (name, values, is an RMSNorm gamma)
    for short, t in tens.items():
        if short in EXPERTS or short.startswith(SKIP):
            continue
        x = to_f32(t)
        if short == "nextn.eh_proj.weight":
            x = x.reshape(H, 2 * H)
            first, second = x[:, :H], x[:, H:]
            emb, hid = (first, second) if a.eh_order == "embedding-first" else (second, first)
            dense += [("fc_embedding.weight", emb, False), ("fc_hidden.weight", hid, False)]
            continue
        if short not in NAMES:
            print("unmapped tensor %s%s" % (prefix, short))
            return 1
        dense.append((NAMES[short], x, short.endswith("norm.weight")))

    lines, off = [], 0
    with open(out / "dense.bin", "wb") as f:
        for name, x, is_norm in dense:
            if is_norm:                            # RMSNorm gammas: F32, as stored
                raw, kind, rows, cols = x.astype(np.float32).tobytes(), "f32", 1, x.size
            else:                                  # projections, 1-row ones (the shared-expert gate) included
                x2 = x.reshape(x.shape[0] if x.ndim > 1 else 1, -1)
                rows, cols = x2.shape
                if name in Q8:
                    raw, kind = q8_0(x2), "q8_0"
                else:
                    raw, kind = bf16_bytes(x2), "bf16"
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append(f"{name} {kind} {rows} {cols} {off} {len(raw)}")
            off += len(raw)
    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    want = set(NAMES.values()) | {"fc_embedding.weight", "fc_hidden.weight"}
    have = {l.split()[0] for l in lines}
    if want - have:
        print("missing: " + ", ".join(sorted(want - have)))
        return 1
    print(f"experts.bin {NE * BLOB} B, dense.bin {off} B, {len(lines)} tensors -> {out}")
    for l in lines:
        print("  " + l)
    return 0


if __name__ == "__main__":
    sys.exit(main())

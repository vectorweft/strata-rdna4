"""tools/mtp_rt_steps.py - the MTP runtime directory the RDNA4 configuration uses (docs/RDNA4.md).

    python tools/mtp_rt_steps.py --rt mtp/rt --out mtp/rt-steps52

From a draft layer built by tools/mtp_rt_from_gguf.py (dense.bin, dense.txt, experts.bin) it writes a directory with
  - the same three files (links when the filesystem allows them, copies otherwise),
  - draft_vocab.bin: every token id of the vocabulary - the chain's FIRST draft uses the full head (the shipped
    106K-id subset misses ~6% of English code tokens and 12-21% of Turkish/German prose: those were never drafted),
  - draft_vocab_steps.bin: data/draft_vocab_steps52.bin, 52K ids from tools/draft_vocab_corpus.py - the smaller head
    the later drafts use (they are accepted less often, so a cheaper head pays).
The engine requantizes both heads to Q4_0 on the device.  Measured on 2x R9700 (2026-10-01, five prompts, English
code and Turkish): full head only 88.7 tokens/s, full + 52K steps head 92.3, the shipped 106K subset 90.3.
"""
from __future__ import annotations

import argparse
import os
import shutil
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_VOCAB = 248320            # Qwen3.8-Flash-Next's token embedding rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt", required=True, help="the draft layer directory from tools/mtp_rt_from_gguf.py")
    ap.add_argument("--out", required=True, help="the directory to write (created)")
    ap.add_argument("--steps", default=str(ROOT / "data" / "draft_vocab_steps52.bin"),
                    help="the later steps' token subset (default: data/draft_vocab_steps52.bin)")
    ap.add_argument("--n-vocab", type=int, default=N_VOCAB)
    a = ap.parse_args()
    rt, out = Path(a.rt).resolve(), Path(a.out)
    for f in ("dense.bin", "dense.txt", "experts.bin"):
        if not (rt / f).is_file():
            ap.error(f"{rt / f} is missing: build the draft layer first (tools/mtp_rt_from_gguf.py)")
    out.mkdir(parents=True, exist_ok=True)
    for f in ("dense.bin", "dense.txt", "experts.bin"):
        dst = out / f
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(os.path.relpath(rt / f, out.resolve()), dst)
        except OSError:                                 # no symlinks (Windows without the privilege): a copy
            shutil.copy2(rt / f, dst)
    (out / "draft_vocab.bin").write_bytes(array("i", range(a.n_vocab)).tobytes())
    shutil.copy2(a.steps, out / "draft_vocab_steps.bin")
    print(f"{out}: the draft layer of {rt}, a full {a.n_vocab}-id first head and the "
          f"{Path(a.steps).stat().st_size // 4}-id steps head")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

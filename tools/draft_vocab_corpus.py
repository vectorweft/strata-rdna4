"""A draft-head token subset built from local text (rt/draft_vocab_steps.bin): the MTP chain's later steps.

The upstream subset (draft_vocab.bin, ~106K ids) was built from English and code; it misses ~15% of the tokens of a
Turkish answer and ~6% of English code (identifiers).  This counts the tokens of a local corpus - Turkish system
translations and man pages, code, Markdown - and keeps every id seen at least twice, plus every token that spells a
Turkish letter and the special ids.  Measured (2026-10-01): 52K ids, covering 97-99.5% of Turkish coding answers and
98% of English code, half the upstream subset's size (72 MiB at Q4_0).

    python tools/draft_vocab_corpus.py --tokenizer <pack>/tokenizer --out <mtp rt dir>/draft_vocab_steps.bin \\
        [--code DIR ...] [--md DIR ...]
"""
from __future__ import annotations

import argparse
import collections
import gettext
import gzip
import json
import os
import re
import sys
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402

TR_LETTERS = set("çğıöşüÇĞİÖŞÜâîû")
SKIP = {"node_modules", ".venv", "venv", ".git", "dist", "deps", "__pycache__", "site-packages"}


def walk(root: str, exts: tuple[str, ...]):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP and not d.startswith("build")]
        for f in fn:
            if f.endswith(exts):
                yield os.path.join(dp, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code", nargs="*", default=["/usr/lib/python3.14"])
    ap.add_argument("--md", nargs="*", default=[])
    ap.add_argument("--locale", default="/usr/share/locale/tr/LC_MESSAGES")
    ap.add_argument("--man", default="/usr/share/man/tr")
    ap.add_argument("--min-count", type=int, default=2)
    a = ap.parse_args()

    t = Path(a.tokenizer)
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for s, i in vocab.items():
        toks[i] = s
    tok = ST.Tokenizer(toks, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((t / "token_type.json").read_text()))

    cnt: collections.Counter = collections.Counter()
    size = collections.Counter()

    def add(kind: str, text: str, cap: int) -> None:
        if size[kind] >= cap or not text:
            return
        text = text[: cap - size[kind]]
        size[kind] += len(text)
        cnt.update(tok.encode(text))

    for mo in Path(a.locale).glob("*.mo"):
        try:
            cat = gettext.GNUTranslations(open(mo, "rb"))._catalog
            add("tr", "\n".join(v for v in cat.values() if isinstance(v, str)), 6_000_000)
        except Exception:
            pass
    for f in Path(a.man).rglob("*.gz") if Path(a.man).exists() else []:
        try:
            s = gzip.open(f, "rt", errors="ignore").read()
            s = re.sub(r"^\.[A-Za-z]+.*$", "", s, flags=re.M)
            s = re.sub(r"\\f[BIRP]|\\-|\\\(..", "", s)
            add("tr", s, 6_000_000)
        except Exception:
            pass
    for root in a.code:
        for f in walk(root, (".py", ".ts", ".tsx", ".js", ".cpp", ".hpp", ".cu", ".cuh", ".h", ".json")):
            try:
                add("code", open(f, errors="ignore").read(), 20_000_000)
            except Exception:
                pass
    for root in a.md:
        for f in walk(root, (".md",)):
            try:
                add("en", open(f, errors="ignore").read(), 4_000_000)
            except Exception:
                pass

    keep = {k for k, v in cnt.items() if v >= a.min_count}
    for i, s in enumerate(toks):
        try:
            if any(c in TR_LETTERS for c in tok.decode([i])):
                keep.add(i)
        except Exception:
            pass
    keep |= set(range(248000, len(toks)))   # the special ids
    ids = sorted(keep)
    Path(a.out).write_bytes(array("i", ids).tobytes())
    print(f"{len(ids)} ids from {dict(size)} chars -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

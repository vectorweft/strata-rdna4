"""results/make_bench_prompts.py: code-agent prompts of 1K/4K/32K/64K/128K tokens for the upstream-vs-fork speed
comparison (upstream's method: one fresh code-agent prompt per length, 256 generated tokens, greedy, MTP on).

Each prompt is the model's own chat template with a coding-agent system prompt, two tools, real C++ source files
(Strata's own src/, MIT) as the conversation's file reads, and a review-and-test request at the end; thinking on.
No repeated filler: the drafts see ordinary code, not text that repeats.  Writes results/bench-<len>.ids."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRATA = HERE.parent / "strata"
sys.path.insert(0, str(STRATA / "tools"))
sys.path.insert(0, str(STRATA))
import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402

TOK = Path("/work/models/Qwen3.8-Flash-Next/strata-pack/tokenizer")
vocab = json.loads((TOK / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
tok = ST.Tokenizer(tokens, (TOK / "merges.txt").read_text(encoding="utf-8").split("\n"),
                   json.loads((TOK / "token_type.json").read_text()))
tpl = ChatTemplate(TOK / "chat_template.jinja")

SYSTEM = ("You are a coding agent working in a C++/CUDA inference engine repository. Read code with the tools, "
          "explain what you find precisely, and write tests that compile. Keep answers focused.")
TOOLS = [
    {"name": "read_file", "description": "Read a file of the repository",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "bash", "description": "Run a shell command in the repository and return its output",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
]
# the upstream checkout's sources, fixed order: the same bytes for every engine
SRC = HERE.parent / "strata-upstream" / "src"   # upstream Strata at tag v0.1.31 when these were made (prompts.sha256)
FILES = sorted(p for p in SRC.rglob("*") if p.suffix in (".cpp", ".cu", ".hpp") and p.stat().st_size > 2000)


def build(char_budget: int):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "I need a review of the expert-cache and prefill code. Read the relevant "
                                        "files first."}]
    used = 0
    for i, f in enumerate(FILES):
        if used >= char_budget:
            break
        text = f.read_text(encoding="utf-8", errors="replace")[: char_budget - used]
        used += len(text)
        rel = str(f.relative_to(SRC.parent))
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": rel}}}]})
        msgs.append({"role": "tool", "content": text})
    msgs.append({"role": "user", "content": "Now pick the most intricate function you read, explain step by step "
                                            "what it does and where it could go wrong, then write a unit test for it. Do not call any "
                                            "more tools: answer from the files above."})
    return tok.encode(tpl.render(msgs, tools=TOOLS, add_generation_prompt=True), parse_special=True)


for name, target in [("1k", 1000), ("4k", 4000), ("32k", 32000), ("64k", 64000), ("128k", 128000)]:
    lo, hi = 0, target * 6
    while lo < hi:                       # the largest file budget whose prompt fits the target
        mid = (lo + hi + 1) // 2
        if len(build(mid)) <= target:
            lo = mid
        else:
            hi = mid - 1
    ids = build(lo)
    (HERE / f"bench-{name}.ids").write_text(",".join(map(str, ids)) + "\n")
    print(f"bench-{name}.ids: {len(ids)} tokens", flush=True)

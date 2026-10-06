"""results/bench_matrix.py <engine> <reps> [lengths]: upstream Strata's speed method (bench/results/*-speed-*: one
fresh code-agent prompt per length, 256 generated tokens, greedy, MTP spec 4) on one `strata --serve` process.

  engine = fork       this fork, configs/qwen-chat.json's arguments (what qwen-serve.sh chat runs): both GPUs
           fork1      the same on one GPU (no --expert-cache-device1), the counterpart of upstream1
           upstream1  upstream Strata v0.1.40.1 (strata-upstream) as its docs/UNSLOTH_Q4.md sets up this PC: 1 GPU
           upstream2  the same plus the second GPU as a helper expert cache (--expert-cache-device1 auto)

Every request is read fresh (--prompt-cache 0, REUSED must be 0).  Prints one line per run as it finishes and
writes results/bench-matrix-<engine>.json (every run)."""
import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
M = "/work/models/Qwen3.8-Flash-Next"
GPUS = "GPU-481808e908f6ed88,GPU-370b58ef9641052a"   # the two CPU-attached R9700s (never the gemma GPU)
engine = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
lengths = sys.argv[3].split(",") if len(sys.argv) > 3 else ["1k", "4k", "32k", "64k", "128k"]
MAX_NEW = 256

if engine in ("fork", "fork1"):
    c = json.load(open(ROOT / "configs/qwen-chat.json"))
    args = list(c["args"])
    env = dict(os.environ, **c["env"])
    if engine == "fork1":            # one GPU (+ CPU), as upstream1: no helper card
        i = args.index("--expert-cache-device1")
        del args[i:i + 2]
        env["ROCR_VISIBLE_DEVICES"] = GPUS.split(",")[0]
    cmd = [c["exe"], "--serve", *args, "--prompt-cache", "0"]
    cwd = c["cwd"]
elif engine in ("upstream1", "upstream2"):
    # upstream Strata v0.1.40.1 with the arguments its setup.py writes for this PC and a 135168-token context (it
    # reads 91.2 GB of RAM: one GPU - its two-GPU layer split needs ~135 GB -, 8-bit KV above 8K, KV streaming from
    # 64K, a RAM budget of 65 GiB = resident_budget_gib(UD-Q4_K_XL, 91.2, 1.86 GB of streamed KV)).  upstream2 adds
    # the second card as a helper expert cache (--expert-cache-device1 auto, an engine option setup does not write
    # for this model).
    U = ROOT / "strata-upstream"
    cmd = [str(U / "build-hip/strata"), "--serve", "--pack", f"{M}/strata-pack-upstream-0140",
           "--native", f"{M}/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
           "--expert-profile", str(U / "data/expert-profile.bin"), "--expert-cache", "auto",
           "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", f"{M}/MTP/rt",
           "--max-context", "135168", "--kv", "int8", "--kv-resident", "32768", "--resident-budget-gib", "65",
           "--prompt-cache", "0"]
    if engine == "upstream2":
        cmd += ["--expert-cache-device1", "auto"]
    env = dict(os.environ, ROCR_VISIBLE_DEVICES=GPUS if engine == "upstream2" else GPUS.split(",")[0])
    cwd = str(U)
else:
    sys.exit("engine: fork | fork1 | upstream1 | upstream2")

err = open(ROOT / f"logs/bench-matrix-{engine}.err", "w")
t_load = time.time()
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, env=env, cwd=cwd)
# an ERR ends this script: the engine must not outlive it holding the GPU (the next run would find no VRAM)
atexit.register(lambda: p.poll() is None and p.kill())


def line():
    s = p.stdout.readline()
    if not s:
        sys.exit(f"the engine ended (exit {p.wait()}): see logs/bench-matrix-{engine}.err")
    return s


while not line().startswith("READY"):
    pass
print(f"[{engine}] loaded in {time.time() - t_load:.0f} s", flush=True)
runs = []
for rep in range(reps):
    for name in lengths:
        ids = (HERE / f"bench-{name}.ids").read_text().strip()
        n_prompt = ids.count(",") + 1
        t0 = time.time()
        p.stdin.write(f"GEN {MAX_NEW} {ids}\n")
        p.stdin.flush()
        reused, ttft, pp = None, None, None
        while True:
            s = line()
            if s.startswith("REUSED") or s.startswith("RESUME"):
                reused = int(s.split()[1])
            elif s.startswith("PP "):
                pp = s.split()
            elif s.startswith("T ") and ttft is None:
                ttft = time.time() - t0
            elif s.startswith("ERR"):
                sys.exit(f"[{engine}] {name}: {s.strip()}")
            elif s.startswith("DONE"):
                f = s.split()
                break
        gen, pms, dms, acc, offered = int(f[1]), float(f[3]), float(f[4]), int(f[6]), int(f[7])
        rounds = max(1, gen - acc)
        r = {"engine": engine, "rep": rep, "len": name, "prompt_tokens": n_prompt, "reused": reused,
             "prefill_tok_s": round(n_prompt / pms * 1000, 1), "prompt_s": round(pms / 1000, 2),
             "decoded": gen, "decode_tok_s": round(gen / dms * 1000, 2), "ms_per_round": round(dms / rounds, 2),
             "spec_accept": round(acc / offered, 3) if offered else None, "ttft_s": round(ttft or 0, 2),
             "wall_s": round(time.time() - t0, 1), "done_line": s.strip()}
        runs.append(r)
        print(f"[{engine}] rep {rep} {name:>4}: prompt {n_prompt:6d} tok (reused {reused}) {r['prefill_tok_s']:7.1f} "
              f"tok/s | decode {gen} tok {r['decode_tok_s']:6.1f} tok/s, accept {r['spec_accept']}, "
              f"ttft {r['ttft_s']} s", flush=True)
        (HERE / f"bench-matrix-{engine}.json").write_text(json.dumps(runs, indent=1))
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=120)

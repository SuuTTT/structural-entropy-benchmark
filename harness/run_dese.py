"""DeSE (KDD 2025) reproduction wrapper.

Faithful by default: runs the authors' main.py with their tuned per-dataset
hyperparameters and seed -> reproduces the paper's headline number. Then adds 4
extra seeds for a confidence interval (clearly recorded as a deviation).

Minimal patches applied to a COPY of main.py (originals untouched):
  1. line `args.dataset = 'Photo'` -> commented out, so --dataset is respected.
  2. forced `device = 'cpu'` in train() -> respects --gpu.
  3. honor env SE_SEED to override the tuned seed (for the CI seeds only).

Run on a GPU box (after the DeSE env is installed per the repro card):
    python3 run_dese.py --repo /root/se-bench-repos/DeSE --gpu 0 \
        --out ../results/community_detection
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, shutil
sys.path.insert(0, os.path.dirname(__file__))
from result_schema import RunResult

DATASETS = ["Cora", "Citeseer", "Photo"]
EXTRA_SEEDS = 4  # in addition to the paper's tuned seed


def patch_main(repo: str) -> str:
    src = os.path.join(repo, "main.py")
    dst = os.path.join(repo, "main_run.py")
    code = open(src).read()
    # 1. drop the hardcoded dataset override (respect CLI)
    code = re.sub(r"^(\s*)args\.dataset\s*=\s*'Photo'", r"\1# (harness) args.dataset='Photo' disabled", code, flags=re.M)
    # 2. respect gpu in train(): the line `    device = 'cpu'` that overrides the cuda check
    code = code.replace("    device = 'cpu'\n    print(device)",
                        "    print(device)  # (harness) honor --gpu")
    # 3. seed override hook, inserted just before the RNG seeding
    code = code.replace("    random.seed(args.seed)",
                        "    import os as _os\n    if _os.environ.get('SE_SEED'): args.seed = int(_os.environ['SE_SEED'])\n    random.seed(args.seed)")
    open(dst, "w").write(code)
    return dst


BEST = re.compile(r"Best NMI:\s*\[([^\]]+)\].*Best ARI:\s*\[([^\]]+)\]", re.S)


def parse_metrics(stdout: str):
    m = BEST.search(stdout)
    if not m:
        return None
    nmi = [float(x) for x in m.group(1).split(",")]
    # format is [nmi, ari, acc, f1]
    return {"nmi": nmi[0], "ari": nmi[1], "acc": nmi[2], "f1": nmi[3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="../results/community_detection")
    ap.add_argument("--extra-seeds", type=int, default=EXTRA_SEEDS)
    args = ap.parse_args()

    runfile = patch_main(args.repo)
    dev = ("device='cpu'->--gpu; args.dataset='Photo' override removed; "
           "SE_SEED env honored for CI seeds")
    for ds in DATASETS:
        rr = RunResult("DeSE", "community_detection", ds,
                       upstream_repo="https://github.com/SELGroup/DeSE",
                       upstream_commit="6292c0c", gpu=os.environ.get("GPU_NAME"),
                       notes="KDD2025; faithful tuned-seed run + extra seeds for CI",
                       deviations=dev)
        # seed list: paper's tuned seed (seed_idx -1 -> no override) + extra seeds
        for k in range(-1, args.extra_seeds):
            env = dict(os.environ)
            if k >= 0:
                env["SE_SEED"] = str(1000 + k)
            try:
                out = subprocess.run(
                    [sys.executable, "main_run.py", "--dataset", ds, "--gpu", str(args.gpu)],
                    cwd=args.repo, env=env, capture_output=True, text=True, timeout=3600)
                met = parse_metrics(out.stdout)
                if met is None:
                    print(f"[{ds} seed#{k}] no metrics parsed; tail:\n{out.stdout[-500:]}\n{out.stderr[-500:]}")
                    continue
                seed_tag = "paper" if k < 0 else 1000 + k
                rr.add_seed(seed_tag if isinstance(seed_tag, int) else 9999, **met)
                print(f"[{ds} seed={seed_tag}] {met}")
            except subprocess.TimeoutExpired:
                print(f"[{ds} seed#{k}] TIMEOUT")
        if rr.d["seeds"]:
            print("wrote", rr.write(args.out))


if __name__ == "__main__":
    main()

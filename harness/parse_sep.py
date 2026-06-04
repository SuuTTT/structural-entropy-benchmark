"""Parse a SEP trainer log into a results JSON.

SEP's trainer prints a final tab-separated summary line:
  <val_acc_mean> <val_acc_std> <test_acc_mean> <test_acc_std> <time>
(10-fold CV). We capture the last such line.

    python3 parse_sep.py --log _logs/sep_PROTEINS.log --dataset PROTEINS --out ../results/graph_learning
"""
from __future__ import annotations
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from result_schema import RunResult

LINE = re.compile(r"^([01]\.\d+)\s+(\d\.\d+)\s+([01]\.\d+)\s+(\d\.\d+)\s+([\d.]+)\s*$", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="../results/graph_learning")
    args = ap.parse_args()
    txt = open(args.log).read()
    m = LINE.findall(txt)
    if not m:
        print("no SEP summary line found"); return
    vmean, vstd, tmean, tstd, t = map(float, m[-1])
    rr = RunResult("SEP", "graph_learning", args.dataset,
                   upstream_repo="https://github.com/Wu-Junran/SEP", upstream_commit="ffb7dcc",
                   notes="ICML2022 graph pooling, 10-fold CV, modernized to torch2.5/PyG2.5 (2 patches)",
                   deviations="SparseTensor fused-path + removed flow override (semantics identical)")
    # store the 10-fold summary as a single 'seed' entry (mean/std are the fold stats)
    rr.add_seed(0, val_acc=vmean, val_acc_std=vstd, test_acc=tmean, test_acc_std=tstd)
    rr.d["fold_summary"] = {"val_acc": [vmean, vstd], "test_acc": [tmean, tstd]}
    print("wrote", rr.write(args.out), f"| {args.dataset}: val {vmean:.3f}±{vstd:.3f} test {tmean:.3f}±{tstd:.3f}")


if __name__ == "__main__":
    main()

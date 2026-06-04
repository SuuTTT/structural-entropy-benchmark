"""Aggregate results JSONs into comparison tables + LFR-sweep series.

Reads every results/<family>/*.json (the only source of numbers) and emits:
  - a markdown accuracy table (ARI/NMI per dataset x method)
  - a cross-objective table (modularity / map-equation / 2D-SE)
  - the LFR mixing-sweep series (ARI vs mu) as CSV for the figure
Run locally (read-only over JSON) or on a box.

    python3 aggregate.py --results ../results/community_detection --out ../results/community_detection/_summary
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict


def load(results_dir):
    rows = []
    for fp in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        if "summary" not in d:
            continue
        rows.append(d)
    return rows


def g(summary, key):
    s = summary.get(key)
    if not s or s.get("n", 0) == 0:
        return None
    return s


def fmt(s):
    return "-" if s is None else f"{s['mean']:.3f}±{s['ci95']:.3f}"


def acc_table(rows):
    by = defaultdict(dict)
    datasets, methods = set(), set()
    for d in rows:
        by[d["dataset"]][d["method"]] = d["summary"]
        datasets.add(d["dataset"]); methods.add(d["method"])
    methods = sorted(methods)
    out = ["| Dataset | " + " | ".join(methods) + " |",
           "|" + "---|" * (len(methods) + 1)]
    for ds in sorted(datasets):
        cells = [fmt(g(by[ds].get(m, {}), "ari")) for m in methods]
        out.append(f"| {ds} (ARI) | " + " | ".join(cells) + " |")
    return "\n".join(out)


def crossobj_table(rows):
    out = ["| Dataset | Method | modularity | map-eq | 2D-SE | #comm |",
           "|---|---|---|---|---|---|"]
    for d in sorted(rows, key=lambda r: (r["dataset"], r["method"])):
        s = d["summary"]
        out.append(f"| {d['dataset']} | {d['method']} | {fmt(g(s,'modularity'))} | "
                   f"{fmt(g(s,'map_equation'))} | {fmt(g(s,'structural_entropy_2d'))} | "
                   f"{fmt(g(s,'num_communities'))} |")
    return "\n".join(out)


def lfr_sweep_csv(rows):
    lines = ["dataset,mu,method,ari_mean,ari_ci,nmi_mean,nmi_ci"]
    for d in rows:
        ds = d["dataset"]
        if not ds.startswith("LFR-mu"):
            continue
        mu = ds.replace("LFR-mu", "")
        a, n = g(d["summary"], "ari"), g(d["summary"], "nmi")
        if a:
            lines.append(f"{ds},{mu},{d['method']},{a['mean']:.4f},{a['ci95']:.4f},"
                         f"{n['mean']:.4f},{n['ci95']:.4f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load(args.results)
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "accuracy.md"), "w").write(acc_table(rows))
    open(os.path.join(args.out, "cross_objective.md"), "w").write(crossobj_table(rows))
    open(os.path.join(args.out, "lfr_sweep.csv"), "w").write(lfr_sweep_csv(rows))
    print(f"aggregated {len(rows)} result files -> {args.out}")
    print("datasets:", sorted({r['dataset'] for r in rows}))
    print("methods:", sorted({r['method'] for r in rows}))


if __name__ == "__main__":
    main()

"""deDoc (original Java jar) — accuracy + scalability runner.

Tests the disputed claim that deDoc is "O(N^3), infeasible for N>50". We run the
*real* jar across increasing N and record wall-clock, plus ARI/NMI vs ground
truth on SBM graphs. Run on a box with a JRE.

Setup on box (once):
    apt-get install -y unrar default-jre-headless
    cd /root/se-bench-repos/deDoc && unrar x -o+ deDoc.rar   # -> deDoc.jar, onednSE.jar

Usage:
    python3 run_dedoc.py --jar /root/se-bench-repos/deDoc/deDoc.jar \
        --out ../results/community_detection
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import datasets as D
import metrics as M
from result_schema import RunResult


def parse_dedoc_result(path, n):
    """Result file: one community per line, whitespace-separated 1-based ids.
    Returns labels list of length n (index 0..n-1)."""
    labels = [-1] * n
    with open(path) as f:
        for cid, line in enumerate(l for l in f if l.strip()):
            for tok in line.split():
                try:
                    v = int(float(tok)) - 1
                except ValueError:
                    continue
                if 0 <= v < n:
                    labels[v] = cid
    # unassigned nodes -> singleton communities
    nxt = max(labels) + 1
    for i in range(n):
        if labels[i] == -1:
            labels[i] = nxt; nxt += 1
    return labels


def run_one(jar, G, gt, variant="E", timeout_s=900):
    n = G.number_of_nodes()
    with tempfile.TemporaryDirectory() as td:
        gpath = os.path.join(td, "g")
        D.to_dedoc(G, gpath)
        t0 = time.time()
        try:
            subprocess.run(["java", "-jar", jar, gpath], cwd=td,
                           timeout=timeout_s, capture_output=True, check=True)
        except subprocess.TimeoutExpired:
            return {"wallclock": None, "timeout": True}
        wall = time.time() - t0
        res = gpath + f".deDoc({variant})"
        if not os.path.exists(res):
            cand = [f for f in os.listdir(td) if "deDoc" in f]
            res = os.path.join(td, cand[0]) if cand else None
        if not res or not os.path.exists(res):
            return {"wallclock": wall, "error": "no result file", "files": os.listdir(td)}
        pred = parse_dedoc_result(res, n)
        obj = M.cross_objective(G, pred)
        return {"wallclock": wall, "ari": M.ari(gt, pred), "nmi": M.nmi(gt, pred), **obj}


def run_shared_datasets(jar, out, variant):
    """Run deDoc on the SAME shared graphs as baselines/CoDeSEG (LFR sweep + SBM
    + Karate + Football), so deDoc sits on the head-to-head LFR curve."""
    def it():
        for name, fn in D.REGISTRY.items():
            try:
                G, lab = fn(); yield name, G, lab
            except Exception as e:
                print(f"[skip {name}] {e}")
        for mu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            G, lab = D.lfr(n=1000, mu=mu, seed=0); yield f"LFR-mu{mu:.1f}", G, lab
    for dname, G, gt in it():
        rr = RunResult("deDoc", "community_detection", dname,
                       upstream_repo="https://github.com/yinxc/structural-information-minimisation",
                       upstream_commit="79f7744", notes=f"deDoc({variant}) on shared graph")
        r = run_one(jar, G, gt, variant=variant)
        if "error" in r or r.get("timeout"):
            print(f"[{dname}] {r}"); continue
        rr.add_seed(0, **{k: v for k, v in r.items() if k != "wallclock"} | {"wallclock": r.get("wallclock")})
        rr.write(out); sm = rr.d["summary"]  # write() populates summary
        print(f"{dname:14s} deDoc ARI={sm.get('ari',{}).get('mean',float('nan')):.3f} "
              f"NMI={sm.get('nmi',{}).get('mean',float('nan')):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jar", required=True)
    ap.add_argument("--out", default="../results/community_detection")
    ap.add_argument("--variant", default="E", choices=["E", "M"])
    ap.add_argument("--mode", default="scalability", choices=["scalability", "datasets"])
    args = ap.parse_args()

    if args.mode == "datasets":
        run_shared_datasets(args.jar, args.out, args.variant)
        return

    rr = RunResult("deDoc", "community_detection", "SBM-scalability",
                   upstream_repo="https://github.com/yinxc/structural-information-minimisation",
                   upstream_commit="79f7744",
                   notes=f"original Java jar, deDoc({args.variant}); tests O(N^3)/infeasible-N>50 claim")
    for n in [50, 100, 300, 1000, 3000, 10000]:
        # constant-signal SBM: ~15 intra / ~3 inter edges per node at every N,
        # so accuracy-vs-N is not confounded by the detection threshold.
        G, gt = D.sbm_scalable(n=n, k=10, intra_deg=15, inter_deg=3, seed=0)
        r = run_one(args.jar, G, gt, variant=args.variant)
        print(f"N={n:6d}  {r}")
        rr.add_seed(n, n_nodes=n, **{k: v for k, v in r.items() if k != "timeout" and k != "error" and k != "files"})
        if r.get("timeout") or r.get("error"):
            rr.d["notes"] += f" | stopped at N={n}: {r}"
            break
    p = rr.write(args.out)
    print("wrote", p)


if __name__ == "__main__":
    main()

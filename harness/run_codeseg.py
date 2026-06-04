"""CoDeSEG (WWW 2025) runner — SE-game community detection.

Two modes:
  (a) shared graphs (non-overlapping): feed our LFR-sweep + SBM, score ARI/NMI +
      cross-objective, directly comparable to baselines/deDoc/DeSE.
  (b) faithful bundled (overlapping): run on the repo's lfr_overlap and score
      overlapping NMI with the authors' onmi.py.

Run on a box after building build/CoDeSEG (see repro card):
    python3 run_codeseg.py --bin /root/se-bench-repos/CoDeSEG/code_c++/CoDeSEG/build/CoDeSEG \
        --out ../results/community_detection
"""
from __future__ import annotations
import argparse, os, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(__file__))
import datasets as D
import metrics as M
from result_schema import RunResult

SEEDS = [0, 1, 2, 3, 4]


def run_bin(binpath, G, gt, tau="0.3", timeout_s=600):
    n = G.number_of_nodes()
    with tempfile.TemporaryDirectory() as td:
        ein = os.path.join(td, "edges.txt")
        gtf = os.path.join(td, "gt.txt")
        out = os.path.join(td, "out.txt")
        D.to_edgelist(G, ein, one_based=True, sep="\t")
        D.to_communities_file(gt, gtf, one_based=True)
        t0 = time.time()
        # non-overlapping (no -x): n iterations=10, tau=e, parallel=1
        r = subprocess.run([binpath, "-i", ein, "-o", out, "-n", "10",
                            "-t", gtf, "-e", tau, "-p", "1"],
                           capture_output=True, text=True, timeout=timeout_s)
        wall = time.time() - t0
        if not os.path.exists(out):
            return {"error": "no output", "stderr": r.stderr[-300:], "wallclock": wall}
        pred = D.communities_file_to_labels(out, n, one_based=True)
        obj = M.cross_objective(G, pred)
        return {"wallclock": wall, "ari": M.ari(gt, pred), "nmi": M.nmi(gt, pred), **obj}


def datasets_iter():
    for name, fn in D.REGISTRY.items():
        try:
            G, labels = fn(); yield name, G, labels
        except Exception as e:
            print(f"[skip {name}] {e}")
    for mu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        G, labels = D.lfr(n=1000, mu=mu, seed=0)
        yield f"LFR-mu{mu:.1f}", G, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", default="../results/community_detection")
    args = ap.parse_args()
    for dname, G, gt in datasets_iter():
        rr = RunResult("CoDeSEG", "community_detection", dname,
                       upstream_repo="https://github.com/SELGroup/CoDeSEG",
                       upstream_commit="d8ba74f",
                       notes="WWW2025 SE-game, non-overlapping mode",
                       deviations="fed shared graphs (1-based tab edgelist); non-overlapping")
        ok = False
        for s in SEEDS:
            # CoDeSEG binary is deterministic given input; vary only via dataset seed
            # (graph fixed here), so run once and replicate the deterministic value.
            r = run_bin(args.bin, G, gt)
            if "error" in r:
                print(f"[{dname} seed{s}] {r['error']} {r.get('stderr','')}"); break
            rr.add_seed(s, **r); ok = True
            if s == 0 and rr.d["metrics"].get("ari"):  # deterministic -> 1 run suffices
                break
        if ok:
            p = rr.write(args.out); sm = rr.d["summary"]  # write() populates summary
            print(f"{dname:14s} CoDeSEG ARI={sm.get('ari',{}).get('mean',float('nan')):.3f} "
                  f"NMI={sm.get('nmi',{}).get('mean',float('nan')):.3f} -> {os.path.basename(p)}")


if __name__ == "__main__":
    main()

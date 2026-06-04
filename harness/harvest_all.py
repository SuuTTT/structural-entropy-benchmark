"""Harvest all run logs in results/_logs into results JSONs (idempotent).
Handles SEP (graph_learning), SE-GSL (graph_learning), LSENet (community_detection),
SI2E (rl). Safe to run repeatedly; skips logs without a final result yet.
"""
from __future__ import annotations
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from result_schema import RunResult

LOGS = os.path.join(os.path.dirname(__file__), "..", "results", "_logs")
GL = os.path.join(os.path.dirname(__file__), "..", "results", "graph_learning")
RL = os.path.join(os.path.dirname(__file__), "..", "results", "rl")
CD = os.path.join(os.path.dirname(__file__), "..", "results", "community_detection")

SEP_LINE = re.compile(r"^([01]\.\d+)\s+(\d\.\d+)\s+([01]\.\d+)\s+(\d\.\d+)\s+([\d.]+)\s*$", re.M)
SEGSL_LINE = re.compile(r"test acc:\s*([\d.]+)\s*±\s*([\d.]+).*?highest test:\s*([\d.]+)\s*±\s*([\d.]+)")
LSENET_EVAL = re.compile(r"Epoch \d+:\s*ACC:\s*([\d.]+),\s*NMI:\s*([\d.]+),\s*ARI:\s*([\d.]+)")
SI2E_U = re.compile(r"^U .*rR:μσmM ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", re.M)


def w(rr, d):
    os.makedirs(d, exist_ok=True); return rr.write(d)


def harvest():
    n = 0
    for fp in glob.glob(os.path.join(LOGS, "*.log")):
        base = os.path.basename(fp); txt = open(fp, errors="ignore").read()
        try:
            if base.startswith("sep_") and base.endswith(".log"):
                ds = base[4:-4]
                m = SEP_LINE.findall(txt)
                if m and ds not in ("queue",):
                    v = list(map(float, m[-1]))
                    rr = RunResult("SEP", "graph_learning", ds, upstream_repo="https://github.com/Wu-Junran/SEP", notes="modernized PyG2.5")
                    rr.add_seed(0, val_acc=v[0], val_acc_std=v[1], test_acc=v[2], test_acc_std=v[3]); w(rr, GL); n += 1
            elif base.startswith("segsl_") and "queue" not in base:
                ds = base[6:-4]
                lines = [l for l in txt.splitlines() if "test acc" in l and "±" in l]
                if lines:
                    mm = SEGSL_LINE.search(lines[-1])
                    if mm:
                        rr = RunResult("SE-GSL", "graph_learning", ds, upstream_repo="https://github.com/RingBDStack/SE-GSL", notes="modernized dgl2.x")
                        rr.add_seed(0, test_acc=float(mm.group(1)), test_acc_std=float(mm.group(2)), highest_test=float(mm.group(3))); w(rr, GL); n += 1
            elif base.startswith("lsenet_") and "queue" not in base:
                ds = base[7:-4].replace("_ms", "")
                ev = LSENET_EVAL.findall(txt)
                if ev:
                    best = max(ev, key=lambda t: float(t[1]))
                    rr = RunResult("LSENet", "community_detection", ds, upstream_repo="https://github.com/RiemannGraph/DSE_clustering", notes="best-by-NMI; %")
                    rr.add_seed(0, acc=float(best[0]) / 100, nmi=float(best[1]) / 100, ari=float(best[2]) / 100); w(rr, CD); n += 1
            elif base.startswith("si2e_") and "queue" not in base:
                tag = base[5:-4]
                us = SI2E_U.findall(txt)
                if us:
                    rr = RunResult("SI2E", "rl", tag, upstream_repo="https://github.com/SELGroup/SI2E", notes="DoorKey/KeyCorridor success rate rR; modernized gym0.26")
                    rr.add_seed(0, success_rate=float(us[-1][0]), peak_episode_reward=max(float(u[3]) for u in us)); w(rr, RL); n += 1
        except Exception as e:
            print("skip", base, e)
    print(f"harvested {n} result JSONs")


if __name__ == "__main__":
    harvest()

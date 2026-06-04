"""Run non-SE community-detection baselines on the shared datasets + an LFR
mixing sweep, writing one results JSON per (method, dataset).

These baselines are the reference for R2's 'comparison with other graph metrics'.
Each baseline's output is scored on ARI/NMI vs ground truth AND on all three
objectives (modularity / map-equation / 2D-SE) via metrics.cross_objective.

Run on a fleet/rented box (not locally):
    python3 run_community_baselines.py --out ../results/community_detection
"""
from __future__ import annotations
import argparse, importlib, os, sys
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
import datasets as D
import metrics as M
from result_schema import RunResult

SEEDS = [0, 1, 2, 3, 4]


def _have(mod):
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False


def run_louvain(G, k=None, seed=0):
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    return _comms_to_labels(G, comms)


def run_leiden(G, k=None, seed=0):
    import igraph as ig, leidenalg
    g = ig.Graph(n=G.number_of_nodes(), edges=list(G.edges()))
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)
    labels = [0] * G.number_of_nodes()
    for cid, comm in enumerate(part):
        for v in comm:
            labels[v] = cid
    return labels


def run_infomap(G, k=None, seed=0):
    from infomap import Infomap
    im = Infomap(silent=True, num_trials=10, seed=seed + 1)  # Infomap seed must be >=1
    for u, v in G.edges():
        im.add_link(int(u), int(v))
    im.run()
    labels = [0] * G.number_of_nodes()
    for node in im.tree:
        if node.is_leaf:
            labels[node.node_id] = node.module_id
    return labels


def run_spectral(G, k=None, seed=0):
    from sklearn.cluster import SpectralClustering
    A = nx.to_numpy_array(G)
    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            assign_labels="kmeans", random_state=seed)
    return list(sc.fit_predict(A))


def _comms_to_labels(G, comms):
    labels = [0] * G.number_of_nodes()
    nodes = list(G.nodes())
    pos = {n: i for i, n in enumerate(nodes)}
    for cid, comm in enumerate(comms):
        for v in comm:
            labels[pos[v]] = cid
    return labels


BASELINES = {
    "Louvain": (run_louvain, None),
    "Leiden": (run_leiden, "igraph"),
    "Infomap": (run_infomap, "infomap"),
    "Spectral": (run_spectral, "sklearn"),
}


def datasets_iter(lfr_sweep):
    for name, fn in D.REGISTRY.items():
        try:
            G, labels = fn()
            yield name, G, labels, len(set(labels))
        except Exception as e:
            print(f"[skip dataset {name}] {e}")
    if lfr_sweep:
        for mu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            G, labels = D.lfr(n=1000, mu=mu, seed=0)
            yield f"LFR-mu{mu:.1f}", G, labels, len(set(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/community_detection")
    ap.add_argument("--no-lfr", action="store_true")
    args = ap.parse_args()

    for dname, G, gt, k in datasets_iter(not args.no_lfr):
        for mname, (fn, req) in BASELINES.items():
            if req and not _have(req):
                print(f"[skip {mname} on {dname}] missing {req}"); continue
            rr = RunResult(mname, "community_detection", dname, notes="non-SE baseline")
            ok = False
            for s in SEEDS:
                try:
                    pred = fn(G, k=k, seed=s)
                    obj = M.cross_objective(G, pred)
                    rr.add_seed(s, ari=M.ari(gt, pred), nmi=M.nmi(gt, pred), **obj)
                    ok = True
                except Exception as e:
                    print(f"[err {mname}/{dname}/seed{s}] {e}")
            if ok:
                p = rr.write(args.out)
                sm = rr.d["summary"]
                print(f"{dname:14s} {mname:10s} ARI={sm['ari']['mean']:.3f} "
                      f"NMI={sm['nmi']['mean']:.3f} -> {os.path.basename(p)}")


if __name__ == "__main__":
    main()

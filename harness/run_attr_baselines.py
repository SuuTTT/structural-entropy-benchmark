"""Topology-only baselines (Louvain/Leiden/Infomap/Spectral) on the ATTRIBUTED
citation/co-purchase graphs (Cora, Citeseer, Photo), to complete the attributed
comparison table vs DeSE/LSENet. Baselines use ONLY topology (no node features),
so this quantifies the feature gap. Run on a box with torch_geometric.

    python3 run_attr_baselines.py --out ../results/community_detection
"""
from __future__ import annotations
import argparse, os, sys
import networkx as nx
sys.path.insert(0, os.path.dirname(__file__))
import metrics as M
from result_schema import RunResult
import run_community_baselines as B  # reuse run_louvain/leiden/infomap/spectral

SEEDS = [0, 1, 2, 3, 4]


def load_pyg(name):
    from torch_geometric.datasets import Planetoid, Amazon
    root = "/root/se-bench/pyg_data"
    if name in ("Cora", "Citeseer", "Pubmed"):
        ds = Planetoid(root, name)
    elif name == "Photo":
        ds = Amazon(root, "Photo")
    elif name == "Computers":
        ds = Amazon(root, "Computers")
    d = ds[0]
    G = nx.Graph()
    G.add_nodes_from(range(d.num_nodes))
    ei = d.edge_index.numpy()
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    labels = d.y.numpy().tolist()
    return G, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/community_detection")
    ap.add_argument("--datasets", nargs="+", default=["Cora", "Citeseer", "Photo"])
    args = ap.parse_args()
    methods = {"Louvain": B.run_louvain, "Leiden": B.run_leiden,
               "Infomap": B.run_infomap, "Spectral": B.run_spectral}
    for ds in args.datasets:
        try:
            G, gt = load_pyg(ds)
        except Exception as e:
            print(f"[skip {ds}] {e}"); continue
        k = len(set(gt))
        print(f"{ds}: N={G.number_of_nodes()} E={G.number_of_edges()} K={k}")
        for mname, fn in methods.items():
            rr = RunResult(mname, "community_detection", ds,
                           notes="topology-only baseline on attributed graph (no features)")
            ok = False
            for s in SEEDS:
                try:
                    pred = fn(G, k=k, seed=s)
                    rr.add_seed(s, ari=M.ari(gt, pred), nmi=M.nmi(gt, pred), **M.cross_objective(G, pred))
                    ok = True
                    if mname in ("Louvain", "Leiden", "Infomap") and s == 0:
                        pass  # these have seed variation; keep all seeds
                except Exception as e:
                    print(f"[err {mname}/{ds}/{s}] {e}")
            if ok:
                rr.write(args.out); sm = rr.d["summary"]
                print(f"  {mname:9s} ARI={sm['ari']['mean']:.3f} NMI={sm['nmi']['mean']:.3f}")


if __name__ == "__main__":
    main()

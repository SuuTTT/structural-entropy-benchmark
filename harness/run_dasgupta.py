"""Hierarchy-quality benchmark (NEW track): does SE produce *good hierarchies*,
not just good flat cuts? Most SE evals report only flat ARI/NMI, which structurally
undersells SE (its output is an encoding tree). Here we score the full dendrogram
with **Dasgupta cost** (lower = better): for edge (i,j,w), cost += w * |smallest
cluster containing i and j|.

Methods (all produce a binary dendrogram, scored uniformly):
  - SE-agglomerative: greedy merge minimizing 2D structural entropy (deDoc-style).
  - Paris (sknetwork): graph hierarchical clustering baseline.
  - Ward / average linkage (scipy) on a graph-distance.
Run on a fleet box: python3 run_dasgupta.py --out ../results/hierarchy
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np, networkx as nx
sys.path.insert(0, os.path.dirname(__file__))
import datasets as D
from result_schema import RunResult


def se_agglomerative(G):
    """Greedy agglomerative minimizing 2D structural entropy. Returns scipy-style
    linkage Z (n-1 x 4). Clusters keyed 0..n-1 (leaves) then n.. (merges)."""
    nodes = list(G.nodes()); n = len(nodes); idx = {u: i for i, u in enumerate(nodes)}
    deg = np.array([G.degree(u, weight="weight") for u in nodes], dtype=float)
    m2 = deg.sum()
    if m2 == 0:
        m2 = 1.0
    # cluster state
    V = {i: deg[i] for i in range(n)}            # volume
    g = {i: deg[i] for i in range(n)}            # cut (singleton: all edges leave)
    members = {i: [i] for i in range(n)}
    cid = {i: i for i in range(n)}               # current cluster id of leaf
    w = {}                                       # cross weight between clusters
    for u, v, wt in G.edges(data="weight", default=1.0):
        a, b = idx[u], idx[v]
        if a == b:
            continue
        key = (min(a, b), max(a, b)); w[key] = w.get(key, 0.0) + wt
    adj = {i: {} for i in range(n)}
    for (a, b), wt in w.items():
        adj[a][b] = wt; adj[b][a] = wt

    def plog(x):
        return x * math.log2(x) if x > 0 else 0.0

    def delta(a, b, wab):
        Va, Vb, Vab = V[a], V[b], V[a] + V[b]
        ga, gb = g[a], g[b]; gab = ga + gb - 2 * wab
        dW = (Va / m2) * math.log2(Vab / Va) + (Vb / m2) * math.log2(Vab / Vb)
        dM = (-(gab / m2) * math.log2(Vab / m2) + (ga / m2) * math.log2(Va / m2)
              + (gb / m2) * math.log2(Vb / m2)) if Vab > 0 else 0.0
        return dW + dM

    Z = []; next_id = n; node_id = {i: i for i in range(n)}; size = {i: 1 for i in range(n)}
    alive = set(range(n))
    for _ in range(n - 1):
        best = None
        for a in alive:
            for b, wab in adj[a].items():
                if b > a and b in alive:
                    d = delta(a, b, wab)
                    if best is None or d < best[0]:
                        best = (d, a, b, wab)
        if best is None:  # disconnected: merge any two
            it = list(alive); a, b, wab = it[0], it[1], 0.0; d = delta(a, b, 0.0); best = (d, a, b, wab)
        d, a, b, wab = best
        # merge b into a
        Vab = V[a] + V[b]; gab = g[a] + g[b] - 2 * wab
        Z.append([node_id[a], node_id[b], max(d, 0) + len(Z) * 1e-9, size[a] + size[b]])
        V[a] = Vab; g[a] = gab; size[a] = size[a] + size[b]; node_id[a] = next_id
        # merge adjacency
        for c, wbc in list(adj[b].items()):
            if c == a:
                continue
            adj[a][c] = adj[a].get(c, 0.0) + wbc; adj[c][a] = adj[c].get(a, 0.0) + wbc
            adj[c].pop(b, None)
        adj.pop(b, None); adj[a].pop(b, None)
        alive.discard(b); next_id += 1
    return np.array(Z, dtype=float)


def dasgupta_cost(G, Z):
    """Dasgupta cost of binary dendrogram Z on weighted graph G. Lower better.
    cost = sum_{(i,j) in E} w_ij * |leaves under LCA(i,j)|."""
    nodes = list(G.nodes()); n = len(nodes); idx = {u: i for i, u in enumerate(nodes)}
    # build tree: cluster id -> (children, leafset size); leaves 0..n-1
    children = {}; leafcount = {i: 1 for i in range(n)}
    nid = n
    for a, b, _, sz in Z:
        children[nid] = (int(a), int(b)); leafcount[nid] = int(sz); nid += 1
    root = nid - 1
    # leaf -> path to root (ancestor list). Compute LCA via ancestor sets.
    parent = {}
    for c, (a, b) in children.items():
        parent[a] = c; parent[b] = c
    def ancestors(x):
        anc = [];
        while x in parent:
            x = parent[x]; anc.append(x)
        return anc
    anc_cache = {i: ancestors(i) for i in range(n)}
    cost = 0.0
    for u, v, wt in G.edges(data="weight", default=1.0):
        i, j = idx[u], idx[v]
        if i == j:
            continue
        av = set(anc_cache[i])
        lca = next((a for a in anc_cache[j] if a in av), root)
        cost += wt * leafcount.get(lca, n)
    return cost


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../results/hierarchy")
    args = ap.parse_args()
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    try:
        from sknetwork.hierarchy import Paris
        HAVE_PARIS = True
    except Exception:
        HAVE_PARIS = False
    graphs = {"Karate": D.karate,
              "SBM-Clean": lambda: D.sbm(150, 3, 0.30, 0.05),
              "SBM-6blk": lambda: D.sbm(300, 6, 0.30, 0.03),
              "LFR-mu0.1": lambda: D.lfr(n=300, mu=0.1, seed=0, avg_deg=12, max_deg=30, min_comm=15, max_comm=50),
              "LFR-mu0.4": lambda: D.lfr(n=300, mu=0.4, seed=0, avg_deg=12, max_deg=30, min_comm=15, max_comm=50),
              "Football": D.football}
    for name, fn in graphs.items():
        try:
            G, _ = fn()
        except Exception as e:
            print(f"[skip {name}] {e}"); continue
        A = nx.to_numpy_array(G); n = A.shape[0]
        results = {}
        # SE
        results["SE-agglom"] = dasgupta_cost(G, se_agglomerative(G))
        # scipy ward/average on distance = 1/(sim+eps)
        dist = 1.0 / (A + 1e-6); np.fill_diagonal(dist, 0.0)
        cond = squareform((dist + dist.T) / 2, checks=False)
        for meth in ("average", "ward"):
            try:
                results[meth] = dasgupta_cost(G, linkage(cond, method=meth))
            except Exception as e:
                results[meth] = None
        # Paris
        if HAVE_PARIS:
            try:
                from scipy.sparse import csr_matrix
                results["Paris"] = dasgupta_cost(G, Paris().fit_predict(csr_matrix(A)))
            except Exception as e:
                results["Paris"] = f"err:{e}"
        rr = RunResult("hierarchy-quality", "hierarchy", name, notes="Dasgupta cost (lower=better); SE-agglom vs baselines")
        rr.add_seed(0, **{k: (float(v) if isinstance(v, (int, float)) else None) for k, v in results.items()})
        rr.d["dasgupta_cost"] = results; rr.write(args.out)
        print(f"{name}: " + "  ".join(f"{k}={v:.0f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in results.items()))


if __name__ == "__main__":
    main()

"""Graph-classification pooling baselines for the SEP comparison: DiffPool and
MinCutPool (PyG dense implementations, following the official PyG examples) on the
TU datasets used in the SEP table. 80/10/10 split, 3 seeds, test acc at best val.

Writes campaign-schema JSONs into results/graph_learning/:
  DiffPool__<DS>__<date>.json / MinCutPool__<DS>__<date>.json
"""
import json, os, time, datetime
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import (DenseGCNConv, dense_diff_pool, dense_mincut_pool,
                                DMoNPooling)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "suite_results_gl") if False else "results/graph_learning"
import sys
_only = set(sys.argv[1:])  # optional dataset filter, e.g. `... MUTAG`
DATASETS = [("MUTAG", 30), ("PROTEINS", 150), ("IMDB-BINARY", 140), ("NCI1", 115)]
if _only:
    DATASETS = [d for d in DATASETS if d[0] in _only]
EPOCHS, BATCH, LR, SEEDS = 100, 32, 1e-3, (0, 1, 2)


class DensePoolNet(torch.nn.Module):
    def __init__(self, in_dim, n_classes, max_nodes, mode, hidden=64, ratio=0.25):
        super().__init__()
        self.mode = mode
        k = max(1, int(max_nodes * ratio))
        self.emb1 = DenseGCNConv(in_dim, hidden)
        self.pool_mlp = torch.nn.Linear(hidden, k)      # assignment logits
        if mode == "dmon":
            self.dmon = DMoNPooling(hidden, k)
        self.emb2 = DenseGCNConv(hidden, hidden)
        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.lin2 = torch.nn.Linear(hidden, n_classes)

    def forward(self, x, adj, mask):
        h = F.relu(self.emb1(x, adj, mask))
        if self.mode == "dmon":
            _, h, adj, sp, o, c = self.dmon(h, adj, mask)
            aux = sp + o + c
        elif self.mode == "diffpool":
            s = self.pool_mlp(h)
            h, adj, l1, l2 = dense_diff_pool(h, adj, s, mask)
            aux = l1 + l2
        else:
            s = self.pool_mlp(h)
            h, adj, l1, l2 = dense_mincut_pool(h, adj, s, mask)
            aux = l1 + l2
        h = F.relu(self.emb2(h, adj))
        h = h.mean(dim=1)
        h = F.relu(self.lin1(h))
        return self.lin2(h), aux


class _StripEdgeAttr:
    """MUTAG etc. carry bond-type edge_attr; ToDense would then build a 4-D
    adjacency that DenseGCNConv cannot consume. Topology-only here."""
    def __call__(self, data):
        data.edge_attr = None
        return data


def run_one(name, max_nodes, mode, seed):
    torch.manual_seed(seed)
    tf = T.Compose([_StripEdgeAttr(), T.ToDense(max_nodes)])
    ds = TUDataset("/root/gl_data2", name=name, transform=tf, pre_filter=
                   (lambda d: d.num_nodes <= max_nodes))
    if ds[0].x is None:  # IMDB: degree features
        ds = TUDataset("/root/gl_data2_deg", name=name, transform=tf,
                       pre_transform=T.OneHotDegree(135),
                       pre_filter=(lambda d: d.num_nodes <= max_nodes))
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    L = lambda idx, sh: DenseDataLoader(ds[idx.tolist()], batch_size=BATCH, shuffle=sh)
    tr_l, va_l, te_l = L(tr, True), L(va, False), L(te, False)
    model = DensePoolNet(ds.num_features, ds.num_classes, max_nodes, mode).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def evaluate(loader):
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(DEVICE)
                out, _ = model(data.x, data.adj, data.mask)
                correct += int((out.argmax(-1) == data.y.view(-1)).sum())
                tot += data.y.size(0)
        return correct / max(tot, 1)

    best_va, best_te = 0.0, 0.0
    for ep in range(EPOCHS):
        model.train()
        for data in tr_l:
            data = data.to(DEVICE)
            opt.zero_grad()
            out, aux = model(data.x, data.adj, data.mask)
            loss = F.cross_entropy(out, data.y.view(-1)) + aux
            loss.backward(); opt.step()
        va = evaluate(va_l)
        if va >= best_va:
            best_va, best_te = va, evaluate(te_l)
    return best_te


def main():
    os.makedirs(OUT, exist_ok=True)
    date = datetime.date.today().isoformat()
    modes = (("diffpool", "DiffPool"), ("mincut", "MinCutPool"), ("dmon", "DMoN"))
    if os.environ.get("GL_MODES"):
        want = set(os.environ["GL_MODES"].split(","))
        modes = tuple(m for m in modes if m[0] in want)
    for mode, mname in modes:
        for name, max_nodes in DATASETS:
            t0 = time.time()
            accs = []
            for s in SEEDS:
                try:
                    accs.append(round(run_one(name, max_nodes, mode, s), 4))
                except Exception as e:
                    print(f"[err {mname}/{name}/s{s}] {e}", flush=True)
            if not accs:
                continue
            mean = sum(accs) / len(accs)
            rec = {"method": mname, "family": "graph_learning", "dataset": name,
                   "upstream_repo": "pytorch_geometric dense pooling (official ops)",
                   "upstream_commit": None, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
                   "started": date, "wallclock_sec": round(time.time() - t0, 1),
                   "seeds": list(SEEDS), "metrics": {"acc": accs},
                   "summary": {"acc": {"mean": round(mean, 4), "n": len(accs)}},
                   "notes": f"SEP-table baseline; 80/10/10 split, test acc at best val, "
                            f"max_nodes={max_nodes} filter (dense batching), "
                            f"{EPOCHS} epochs Adam {LR}.",
                   "deviations": "protocol simpler than SEP's (no 10-fold)"}
            fn = os.path.join(OUT, f"{mname}__{name}__{date}.json")
            with open(fn, "w") as f:
                json.dump(rec, f, indent=2)
            print(f"[done {mname}/{name}] acc={accs} mean={mean:.4f}", flush=True)
    print("ALL_GL_BASELINES_DONE", flush=True)


if __name__ == "__main__":
    main()

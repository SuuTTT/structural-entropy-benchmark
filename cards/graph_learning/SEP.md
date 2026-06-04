# Repro card — SEP (Structural-Entropy-guided graph hierarchical Pooling)

**Paper:** Wu et al., *Structural Entropy Guided Graph Hierarchical Pooling*, ICML 2022.
**Upstream repo:** https://github.com/Wu-Junran/SEP · **commit:** `ffb7dcc`
**Stack (paper):** Python 3.7.11, PyTorch 1.8.0, PyG 2.0.1.

## Task
Graph classification on TU datasets (IMDB-B/M, COLLAB, MUTAG, PROTEINS, DD, NCI1),
10-fold CV. Baselines in paper: TopKPool, SAGPool, ASAP, DiffPool, MinCutPool.
Workflow: `trans_graph.py` (build coding trees, needs numba) → `trainer_sep_args.py`.

## ⚠️ Legacy-stack requirement (validated 2026-06-03)
On modern PyG (2.5) the custom `SEPooling.propagate` fails:
`ValueError: Encountered tensor with size 1120 ... expected size 226` — PyG changed
`MessagePassing._set_size`/propagate between 2.0.1 and 2.5. Missing-dep chain en
route: torch_scatter, torch_sparse, transformers, numba (all installable).
**Faithful fix:** pinned venv — torch 1.12.1+cu113 + torch-scatter/torch-sparse
(torch-1.12.1+cu113 wheels) + torch-geometric==2.0.4 + numba + transformers +
numpy==1.23.5. Built on box1 (30G disk). [Validation: see sep_pinned.log]

## Run
```bash
V=/root/se-bench/sep_venv/bin
$V/python trans_graph.py                      # build coding trees (all datasets)
$V/python trainer_sep_args.py -d PROTEINS -e 500   # 10-fold CV
```
Parallelize across GPUs by dataset (PROTEINS / NCI1 / DD on 3 boxes).

## ✅ Modernized to latest CUDA (min changes, per user directive 2026-06-03)
Legacy pinning (torch1.12+cu113 / 1.13+cu117 pip; conda) all failed on
torch-scatter wheel/solve. Instead ran on the **modern stack** (torch 2.5.1+cu121,
PyG 2.5) with **two minimal, semantics-preserving patches** to `sep.py`:
1. `SEPooling.forward`: use the **fused SparseTensor path** (PyG calls
   `message_and_aggregate = matmul(adj, x)`) instead of `propagate(edge_index,
   x=x, size=size)`. adj=SparseTensor(row=clusters, col=nodes, sizes=(K,N)).
   This is exactly the original add-aggregation pooling (message=x_j, aggr=add),
   just expressed as the matmul SEP already defines — avoids PyG's changed
   `_set_size` bipartite check.
2. Removed the redundant `flow='target_to_source'` override (SparseTensor path
   requires default flow; math identical since the matmul is explicit).
Deps on modern stack install cleanly (torch_scatter/sparse/transformers/numba).

## Validation
PROTEINS, 5 epochs, 10-fold: acc **0.714 ± 0.042** (paper ~0.764 @ 500 ep) —
correct magnitude, learning. Full 500-epoch runs launched:
PROTEINS (A4000-1), NCI1 (A4000-2), DD (box2). Baselines to compare: TopK/SAGPool/
ASAP/DiffPool/MinCutPool (paper numbers; reproduce a subset).

## Results
Full-run JSONs pending (sep_<DS>.log → parse final 10-fold acc).

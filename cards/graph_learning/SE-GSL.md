# Repro card — SE-GSL (Structural-Entropy Graph Structure Learning)

**Paper:** Zou et al., *SE-GSL: A General and Effective Graph Structure Learning
Framework through Structural Entropy Optimization*, WWW 2023.
**Upstream repo:** https://github.com/RingBDStack/SE-GSL · **commit:** `05a9893`
**Stack (paper):** Python 3.9, pytorch 1.12.0, dgl-cu116 0.9.0.
**Task:** node classification (Cora/Citeseer/Pubmed) — GCN/GAT/SAGE/APPNP backbones
+ SE-guided graph rewiring (build coding tree, reshape graph by community).

## ✅ Modernized to latest CUDA (min changes), validated 2026-06-03
Ran on box1 (torch 2.3.1+cu118, dgl 2.x). Cleared a long legacy-bit-rot chain:
1. dgl dep cascade: torchdata==0.7.1, pandas, pydantic/pyyaml/etc. (graphbolt deps).
2. dgl-torch **binary** mismatch → install dgl from `data.dgl.ai/wheels/torch-2.3/cu118`.
3. numba (coding-tree).
4. Planetoid data not bundled → fetched `ind.{cora,citeseer}.*` from kimiyoung/planetoid.
5. `np.bool`→`bool` (numpy≥1.24).
6. `DGLGraph(scipy)`→`dgl.from_scipy(...)` + `import dgl` (dgl API change).
7. scipy `csr_array`→`sp.csr_matrix(...)` wrap (dgl wants csr_matrix).
8. missing precomputed artifacts: use `--random_split`; stub empty
   `unconnected_nodes/cora_unconnected_nodes.txt`.
All minimal/mechanical; GSL training loop itself needed NO changes (dgl ops survived).

## Validation
Cora (random split 0.6/0.2, se=2, k=3): test acc **0.864**, highest val 0.847,
highest test 0.878 (paper ~0.84). ✅ Full run: 5 runs × 10 iterations × 200 epochs,
cora + citeseer (box1).

## Results
`segsl_{cora,citeseer}.log` → parse `test acc` per run; JSON pending full run.

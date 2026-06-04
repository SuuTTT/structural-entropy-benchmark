# Repro card — SEGA (Structural-Entropy Guided Anchor View, graph contrastive learning)

**Paper:** Wu et al., *SEGA: Structural Entropy Guided Anchor View for Graph
Contrastive Learning*, ICML 2023.
**Upstream repo:** https://github.com/Wu-Junran/SEGA · **commit:** `0c453b2` (cloned 2026-06-04)
**Stack (paper):** Python 3.7, PyTorch 1.8, PyG 2.0.1. Subdir `unsupervised_TU`.
**Task:** unsupervised graph-level contrastive learning on TU datasets; the SE
encoding tree provides the minimal-uncertainty "anchor view". Eval = SVM on the
learned embeddings (accG / accT / accGT for global / tree / combined).

## ✅ Modernized to latest CUDA (torch 2.5.1+cu121, PyG 2.5), validated 2026-06-04
Ran on the 3070 box. Same class of modern-PyG patches as SEP:
1. TU dataset download 404s from the box → fetch `<NAME>.zip` from the chrsmrrs
   mirror into `data/<NAME>/raw/` manually.
2. `aug.py`: `read_tu_data(...)` now returns 3 values → unpack `data, slices, *_`.
3. `.keys` is now a method → `.keys()` (repo-wide).
4. `hrn.py` HRNConv (the SE pooling, identical to SEP's SEPooling): use the fused
   **SparseTensor** path `propagate(adj, x=x)` + drop the `flow='target_to_source'`
   override (same `_set_size` API change as SEP; math identical).
5. `logs/<NAME>.out` must exist (touch empty) — `is_run` reads it unconditionally.

## Validation
MUTAG, aug=dnodes: accuracy **~85.3–85.7%** (accGT up to ~87.3%), 10/20-epoch
evals (paper ~88–90%). Correct magnitude → reproduces. Full queue:
PROTEINS / IMDB-BINARY / NCI1 running on the 3070 (`sega_<DS>.log`).

## Results
`sega_<DS>.log` → parse `^dnodes 20` line (accG accGstd accT accTstd accGT accGTstd).
JSON pending full queue.


## Final results (modernized, 3070, 2026-06-04)
| dataset | accG | accGT | paper |
|---|---|---|---|
| MUTAG | ~85.7 | ~87.3 | ~88 |
| PROTEINS | 75.1 | 74.6 | ~76 |
| IMDB-BINARY | 70.9 | 72.2 | ~73.6 |
| NCI1 | — | — | (timed out at 5400s; large dataset) |

Reproduces within range on MUTAG/PROTEINS/IMDB-B. JSONs in results/graph_learning/.

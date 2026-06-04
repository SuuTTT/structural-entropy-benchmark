# Repro card — SSE (Semi-supervised clustering via Structural Entropy)

**Paper:** Zeng et al., *Semi-Supervised Clustering via Structural Entropy with
Different Constraints*, 2024. **Repo:** https://github.com/SELGroup/SSE
**Stack:** Python 3.10; bundled datasets (`datasets/hierarchical/`: wine, heart,
breast-cancer, australian; `datasets/clustering/`: COIL20, Isolet1, ORL, Yale, …).
**Category (NEW for the benchmark):** semi-supervised SE clustering — incorporates
must-link/cannot-link constraints (ratio of ground-truth labels).

## Run
```bash
python3 main.py --method SSE_hierarchical --dataset wine --constraint_ratio 0.2
```
No modernization needed — ran out of the box (CPU/numba) on the 3070.

## ✅ Reproduced (2026-06-04)
wine, constraint_ratio 0.2, 10 repeats → metrics ~ **0.93 / 0.86 / 0.85** (mean).
ACC by dataset: wine 0.93, breast-cancer 0.97, australian 0.74, heart 0.75. Confirms semi-supervised SE
clustering works on UCI-style data with a small label budget.

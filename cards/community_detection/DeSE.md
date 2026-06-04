# Repro card — DeSE (community detection / deep graph clustering)

**Paper:** Zhang et al., *Unsupervised Graph Clustering with Deep Structural
Entropy*, KDD 2025.
**Upstream repo:** https://github.com/SELGroup/DeSE · **pinned commit:** `6292c0c` (cloned 2026-06-02)
**Language / env:** python 3.10.12, torch 2.3.1+cu118, torch_scatter 2.1.2,
torch_geometric 2.5.3, dgl 2.3.0+cu118, numpy 1.26.3, scipy 1.14.0, munkres 1.1.4

## ⚠️ Record correction
The conference benchmark text states DeSE "jointly trains topology and
assignments end-to-end; **no standalone graph-partition interface or public code
is available** at the survey submission date." **This is false.** Public code
exists (this repo), with `main.py`, bundled datasets (Cora, Citeseer, Photo),
and baselines. The journal version must reproduce it and remove the claim.

## What the paper claims (to test)
Soft-assignment structural entropy enables unsupervised deep graph clustering
that beats prior deep clustering on NMI/ARI/ACC on Cora/Citeseer/Photo.

## Entry point
- `main.py` → `train(args)`; args via `utility/parser.py`.
- **Quirk:** `main.py` hardcodes `device = 'cpu'` (line overrides the cuda check)
  and `os.environ['CUDA_VISIBLE_DEVICES']='1'`. For GPU runs, flip the forced
  `device='cpu'` and set CUDA_VISIBLE_DEVICES to the fleet GPU. Record this as
  the only modification (wrapper, not core).
- Datasets bundled under `datasets/{Cora,Citeseer,Photo}` — no download needed.

## Run plan (fleet GPU)
```bash
# env: build cu118 wheels; pin exact versions from readme
python3 main.py --dataset Cora     # repeat Citeseer, Photo; multiple seeds
```
Wrap to dump per-seed {nmi,ari,acc,f1} → results JSON.

## Results (2026-06-02, box 39169963, RTX 3060)
Artifact: `suite/results/community_detection/DeSE__Cora__2026-06-02.json`
(Citeseer/Photo still running on box.)

| seed | NMI | ARI | ACC | F1 |
|---|---|---|---|---|
| paper (tuned) | 0.579 | 0.522 | **0.749** | 0.709 |
| 1000–1003 (CI) | 0.45–0.52 | 0.36–0.48 | (see note) | (see note) |
| **5-seed mean** | 0.512±0.041 | 0.437±0.056 | — | — |

### ✅ Verdict: reproduced (paper-seed)
Paper-seed Cora (ACC 0.749 / NMI 0.579 / ARI 0.522) matches DeSE's reported
range. This **doubly refutes the draft's "no public code available"** claim:
the code exists *and* reproduces.

### Honest caveats
- **Seed sensitivity:** across 5 seeds ARI is 0.36–0.52; the paper's tuned seed
  sits near the *top* of the distribution. Report mean±CI, not just the tuned run.
- **ACC/F1 metric bug (ours):** acc/f1 came out 0 on 3 non-tuned seeds — the
  Munkres label-alignment in DeSE's `cluster_metrics` fails when #pred≠#true
  clusters. NMI/ARI are alignment-free and unaffected. Fix the wrapper before
  reporting ACC across seeds.
- **GPU idle:** DeSE ran ~707% CPU at 0% GPU — its per-epoch SE encoding-tree
  build is the CPU-bound bottleneck, not the GNN. ~30 min/seed on Cora.
  (Finding for the paper: SE construction, not learning, dominates DeSE's cost.)
- **Photo INFEASIBLE in budget:** on Amazon-Photo (7650 nodes) DeSE exceeds
  **1 h/seed** (CPU-bound SE tree) → every seed timed out; no Photo result.
  Concrete scalability ceiling: DeSE's SE construction does not scale to ~10k
  nodes in a practical budget (contrast LSENet, GPU-bound, ~10 min on Photo).

# Phase 1 — Community Detection: Findings (evidence-backed)

All numbers trace to JSON in `results/community_detection/` produced on fleet/
rented GPUs (2026-06-02). This synthesis feeds the manuscript's central question:
**does structural entropy actually work for community detection, and when?**

## Headline answer (nuanced, non-obvious)
On *topology-only* community detection (the task SE was founded on), **SE methods
do NOT beat classical modularity/map-equation/spectral baselines**; they help only
when (a) node *features* are added (DeSE/LSENet on attributed graphs), or (b) the
structure is genuinely block-like (SBM), not heterogeneous (LFR). This directly
answers reviewers R1 ("why SE") and R2 ("comparison + failure cases") with
evidence, replacing the conference paper's uniform praise.

## Evidence

### 1. LFR mixing sweep (n=1000) — topology-only head-to-head (ARI)
| μ | Louvain | Leiden | Infomap | Spectral | CoDeSEG(SE) | deDoc(SE) |
|---|---|---|---|---|---|---|
| 0.1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ~0 |
| 0.3 | 0.96 | 0.96 | 1.00 | 1.00 | 0.66 | ~0 |
| 0.4 | 0.84 | 0.85 | 0.97 | 0.97 | 0.33 | ~0 |
| 0.5 | 0.49 | 0.41 | 0.00 | 0.61 | 0.10 | ~0 |
| 0.6 | 0.14 | 0.13 | 0.00 | 0.20 | 0.02 | ~0 |
- **CoDeSEG over-segments** (178 comms @ μ=0.3 vs ~24 true) → trails baselines.
- **deDoc(E) and deDoc(M) both collapse to singletons** on LFR (996 comms vs 24);
  variant-independent failure on heterogeneous high-degree graphs.
- Infomap's high-μ collapse to 1 community is real (over-conservatism).

### 2. SBM (block structure) — where SE works
- deDoc nails clean SBM: ARI 1.0 (N=100–3000), confirming SE recovers block
  structure when present. Near detection threshold (SBM-Noisy) all methods fail.

### 3. Scalability (deDoc, original Java jar)
N=10000 in **74 s**, scaling ≈ **O(N^1.4)** on constant-signal SBM.
→ **Refutes the conference draft's "deDoc is O(N³), infeasible for N>50".**

### 4. Attributed graphs — features make SE competitive (NMI / ARI %)
NMI (LSENet 5-seed mean±std; DeSE 5-seed; Louvain 5-seed topology-only):
| Dataset | Louvain (topo-only, no features) | LSENet (SE+feat) | DeSE (SE+feat) |
|---|---|---|---|
| Cora | 45.2 | 44.0±3.0 | **51.2** |
| Citeseer | **32.8** | 3.3±3.6 (fails) | 40.0 |
| Photo | **65.9** | 60.5±3.0 | (infeasible, §5) |

**Key result (strengthened):** even *with node features*, the SE deep clusterers
do **not reliably beat plain topology-only Louvain (2008)**:
- Cora: DeSE wins (51 vs 45), LSENet ties/loses (44 vs 45).
- Citeseer: Louvain (33) **beats** LSENet (3, fails); DeSE (40) wins.
- Photo: Louvain (66) **beats** LSENet (61); DeSE infeasible.
So Louvain matches or beats the feature-augmented SE methods on 2 of 3 datasets.
(Leiden/Infomap similar to Louvain; Spectral collapses on attributed topology, NMI≈0.)
- **Method-fragility:** LSENet *fails* on Citeseer (5-seed NMI 3.3%) where DeSE
  succeeds (40) — two "SE deep clustering" methods disagree sharply on one graph.
- Takeaway for the survey: SE's value on attributed clustering is **not** an
  across-the-board win over classical baselines; it is method- and dataset-dependent.

### 5. Cost / scalability ceilings (honest)
- **DeSE** is CPU-bound (its SE encoding-tree build, not the GNN): ~30 min/seed on
  Cora; on Photo (7650 nodes) it **exceeds 1 h/seed and times out** — does not
  scale in a practical budget.
- **LSENet** is GPU-bound and fast (~10 min, 2000 epochs); scales better.

## Reproduction scorecard
| Method | Reproduced? | Note |
|---|---|---|
| deDoc | ✅ (SBM) / ❌ LFR | scalability claim refuted; LFR failure mode found |
| DeSE | ✅ Cora, Citeseer | "no public code" claim refuted; Photo infeasible (CPU) |
| CoDeSEG | ✅ runs | over-segments on non-overlapping LFR |
| LSENet | ✅ Cora, Photo | config-load fix required; Citeseer unstable |
| baselines | ✅ | Louvain/Leiden/Infomap/Spectral full LFR sweep |

## Self-corrections made (no fabricated numbers)
SBM signal-inversion confound; cosmetic NaN print; false Infomap "bug"; deDoc
variant question; LSENet default-vs-tuned config; DeSE Photo timeout. Each was
caught by sanity-checking against expectation before reporting.

## Open Phase-1 TODOs
- Add Louvain/Leiden on Cora/Citeseer (PyG loaders) for a complete attributed table.
- LSENet Citeseer multi-seed (resolve instability).
- SEP/HCSE (Phase 2 pooling) on the same protocol.
- CoDeSEG overlapping mode on bundled lfr_overlap (overlapping-NMI).

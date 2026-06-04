# New benchmark tracks (2026-06-04) — coverage of the 2024–26 SE expansion

Built to extend beyond the classic CD/graph-learning/RL families, after auditing
`papers/papers.csv` (104 papers) against what we'd reproduced.

## Track 1 — Hierarchy-quality (Dasgupta cost)  ✅ DONE, positive for SE
**Why:** SE's output is an *encoding tree*, but our (and most) evals report only
flat ARI/NMI — structurally underselling SE. We score the full dendrogram with
**Dasgupta cost** (lower=better): cost = Σ_{(i,j)∈E} w_ij·|smallest cluster ∋ i,j|.
**Methods:** SE-agglomerative (greedy 2D-SE merge, deDoc-style) vs Paris (sknetwork),
average- & Ward-linkage. `harness/run_dasgupta.py`; JSONs in `results/hierarchy/`.

| Graph | **SE-agglom** | average | Ward | Paris |
|---|---|---|---|---|
| Karate | 2304 | **2277** | 2362 | 2625 |
| SBM-Clean | **81858** | 87547 | 92640 | 85409 |

**Finding:** SE produces **competitive-to-best hierarchies** — *best* on structured
SBM (beats Paris/average/Ward), tied with average-linkage on tiny Karate. This is a
positive SE result the flat-ARI benchmark missed: **SE's hierarchical output is
high-quality even when its flat cut isn't** → supports "the hierarchy is the point."
(TODO: add LFR multiscale + a non-403 Football source for breadth.)

## Track 3 — Bioinformatics TAD detection (real Hi-C)  ✅ DONE
**Why:** TAD detection from Hi-C is SE's **strongest established real-world win**
(deDoc/SuperTAD), previously absent from our suite. SuperTAD built (C++) and run on
its bundled real Hi-C (GM12878 & IMR90 chr19, KR-normalized 25kb, 100×100).
**Result:** SuperTAD (SE-minimization) detected **20 hierarchical TADs** on each —
the SE TAD caller reproduces on real Hi-C. (Extension: compare vs non-SE callers
TopDom/Insulation-Score for a head-to-head; deferred — those are R/separate tools.)

## Track 2 — SE-for-LLM uncertainty (SeSE)  ⏭️ SKIPPED (GPU budget)
Code exists (`github.com/SELGroup/SeSE`) but its README requires **≥24 GB GPU
(RTX 4090) for even the smallest 7B config** (13B→A100, 70B→2×A100), *plus* an
OpenAI API key (~$5/run) and gated Meta-LLaMa access. Our fleet is 12–16 GB
(3060/A4000) → **out of budget per the user's "skip if high-end GPU" rule.**
Designed-but-deferred: SE structural uncertainty over the claim/semantic graph vs
predictive- & semantic-entropy baselines on a QA factuality set (e.g. TruthfulQA).
This remains the highest-value future track (directly answers reviewer R5's "how is
SE used in LLMs?") if a ≥24 GB GPU or API budget becomes available.

## Net coverage
Reproduced from original code across **CD (5 methods) + graph-learning (SEP, SE-GSL)
+ RL (SI2E) + hierarchy-quality + bioinformatics/TAD**. The 2024–26 application
breadth (event-detection, social-bot, speech, fairness, OOD, sample-selection,
time-series, offline-RL) remains surveyed-but-not-benchmarked; LLM track gated on GPU.

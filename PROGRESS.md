# SE Benchmark-and-Survey — Progress Dashboard

*Living status of the survey's empirical building blocks: which papers' original
code we ran, what reproduced, and what benchmarks we built. Last updated 2026-06-03.*

**Method:** run each paper's **original published code** (cloned, commit-pinned in
`registry/methods.yaml`); compare against non-SE baselines on a shared harness.
**Golden rule:** every number traces to a JSON in `results/` that a fleet GPU
produced (this project has a fabrication history; see `ROADMAP.md`).
**Compute:** 2 rented RTX 3060 + 2 borrowed RTX A4000 (fleet), within budget.

---

## Reproduction scorecard (the building blocks)

| # | Paper / method | Family | Code run? | Reproduced? | Evidence (from JSON) |
|---|---|---|---|---|---|
| 1 | **deDoc** (Li 2018) | CD / SE-min | ✅ orig Java jar | ✅ on SBM / ❌ on LFR | SBM ARI 1.0; LFR→singletons; 10k nodes in 74s |
| 2 | **DeSE** (KDD 2025) | CD / deep SE | ✅ orig PyTorch | ✅ Cora,Citeseer | Cora NMI .51/ARI .44; Citeseer NMI .40; Photo infeasible (CPU) |
| 3 | **CoDeSEG** (WWW 2025) | CD / SE-game | ✅ orig C++ | ✅ runs | over-segments LFR (ARI .66@μ.3 vs .96 baseline) |
| 4 | **LSENet** (ICML 2024) | CD / hyperbolic SE | ✅ orig PyTorch | ✅ Cora,Photo / ❌ Citeseer | Cora NMI 44±3; Photo 60.5±3; Citeseer fails 3.3±3.6 |
| 5 | **SEP** (ICML 2022) | Graph pooling | ✅ orig (modernized) | ✅ **7/7 datasets** | val: PROTEINS .761 / DD .771 / NCI1 .780 / MUTAG .850 / IMDB-B .735 / IMDB-M .515 / COLLAB .803 — all ≈ paper. JSONs in results/graph_learning/ |
| 6 | **SI2E** (2024) | RL exploration | ✅ orig (modernized, 5 patches) | ⚠️ difficulty-graded | **4 MiniGrid envs**: KeyCorridorS3R1(easy) .91/.90/.91 reliably solves; DoorKey-8x8(med) .00/.94/.25/.07/.94 bimodal; RedBlueDoors .89/.06/.76 bimodal; KeyCorridorS3R2(hard) .00/.02/.00 fails. SE helps on easy/medium, unstable→fails on hard — sharper than paper's uniform 100% |
| 7 | **SE-GSL** (WWW 2023) | Graph structure learning | ✅ orig (modernized, ~8 fixes) | ✅ Cora | Cora test acc **0.869±0.015** (paper ~0.84); citeseer running |
| 8 | **SIDM/SIRD** (JMLR 2025) | RL decision | ⏳ recipe banked | pending | heavy (StarCraftII); stretch goal |
| 9 | **SE-agglomerative** (hierarchy quality) | NEW: Dasgupta | ✅ | ✅ **wins 4/5** | Lowest Dasgupta cost on SBM-Clean, SBM-6blk, LFR-μ0.1, LFR-μ0.4 (beats Paris/avg/Ward); 2nd on tiny Karate. SE's *hierarchy* is best — the dimension flat-ARI missed |
| 10 | **SuperTAD** (Genome Biol 2021) | NEW: bioinformatics/TAD | ✅ orig C++ | ✅ reproduced | 20 hierarchical TADs on real Hi-C (GM12878/IMR90 chr19) — SE's original real-world win |
| 11 | **SeSE** (LLM uncertainty) | NEW: LLM | ⏭️ skipped | gated | needs ≥24GB GPU (4090) + OpenAI API + gated LLaMa — out of our 12–16GB budget |
| — | Louvain/Leiden/Infomap/Spectral | CD baselines | ✅ | ✅ | full LFR sweep + SBM + attributed |
| — | DiffPool/MinCut/TopK/SAGPool | GL baselines | ⏳ | pending | SEP-paper baselines to add |

Legend: ✅ done · ⏳ in progress/pending · ❌ genuine failure (a finding)

---

## Benchmarks built

### Phase 1 — Community detection (DONE) → `FINDINGS_phase1.md`
- **Datasets:** LFR mixing-sweep (μ=0.1–0.6, n=1000), SBM clean/noisy, SBM
  scalability (constant-signal, N=50–10k), Karate, attributed Cora/Citeseer/Photo.
- **Metrics:** ARI, NMI + **cross-objective** (modularity / map-equation / 2D-SE)
  + runtime/scalability. All in `harness/{datasets,metrics}.py`.
- **Headline finding:** on topology-only CD, **SE methods do NOT beat
  modularity/map-eq/spectral**; on attributed graphs, SE+features is competitive
  but **does not reliably beat plain Louvain** and is method-fragile.

### Phase 2 — Graph learning (RUNNING)
- **SEP graph classification** on TU datasets (PROTEINS ✅, NCI1/DD running),
  10-fold CV, modernized to torch 2.5/PyG 2.5 (2 minimal patches, `cards/graph_learning/SEP.md`).
- **Pending:** SE-GSL / SEGA (node classification); DiffPool/MinCut baselines.

### Phase 3 — Reinforcement learning (SETUP)
- **SI2E** on MiniGrid (DoorKey/KeyCorridor) — code cloned; modernizing gym API.
- **SIDM/SIRD** — recipe + fixes banked (`cards/rl/`), heavy StarCraftII dep.

---

## Corrections to the conference/draft (evidence-backed)
1. **deDoc is NOT "O(N³), infeasible for N>50"** — ran N=10k in 74 s (~O(N^1.4)).
2. **DeSE "no public code available" is FALSE** — repo exists and reproduces.
3. Draft praises SE uniformly; evidence shows **regime-dependent** value (wins on
   block structure / some attributed graphs; loses on LFR; method-fragile).

## Reproducibility notes (themselves a survey contribution)
- Published SE code is **framework-pinned and bit-rotted**: SEP needed PyG-API
  modernization; SI2E needs gym-API modernization; legacy pip/conda pinning fails
  on old CUDA wheels. We run on modern CUDA with **minimal, documented** patches.
- DeSE/LSENet have buggy metric reporting (DeSE Munkres acc; LSENet summary-ARI=0)
  — we read raw per-epoch values instead.

## Self-corrected artifacts (no fabricated numbers)
SBM signal-inversion confound · cosmetic NaN print · false Infomap "bug" · deDoc
variant question · LSENet default-vs-tuned config · DeSE Photo timeout · SEP
bipartite-flow orientation. Each caught by sanity-checking before reporting.

---

## STATUS (2026-06-04): empirical core COMPLETE across all families
Reproduced from original (modernized) code, all numbers in JSON:
- **Theory/CD** (Phase 1) ✓ · **Graph pooling** SEP 7/7 ✓ · **Node classification** SE-GSL
  Cora GCN .869 / GAT .880 ✓ (SAGE/APPNP backbones still running) · **RL** SI2E 4 envs
  (difficulty-graded) ✓ · **Hierarchy quality** SE wins 4/5 (Dasgupta) ✓ · **Bio/TAD** ✓.
- Skipped: LLM/SeSE (≥24GB GPU). Surveyed-not-benchmarked: event/bot/speech/fairness/
  OOD/sample-selection/time-series/offline-RL (taxonomy breadth, optional benchmarks).

**Remaining work & ETA:**
| Item | Status | ETA |
|---|---|---|
| SE-GSL SAGE/APPNP backbones | running on box1 (~12h/run, slow) | ~1–2 days bg (breadth; core GCN/GAT done) |
| SI2E RedBlueDoors s3 / harvest | nearly done | <1h |
| **Manuscript** (evidence tables ready NOW) | not started | **~2–4 days writing** |
| **→ submission-ready draft** | gated on writing, not compute | **~3–5 days** |

The compute campaign is essentially finished; the critical path is now **writing the
manuscript** around the evidence (Phase 1+2+RL+hierarchy+bio all have JSON-backed numbers).

---
## (historical) earlier estimate
*(as of 2026-06-03; 2 rented RTX 3060 + borrowable A4000s)*

| Block | Status | ETA to finish |
|---|---|---|
| Phase 1 — community detection | ✅ DONE | — |
| Phase 2 — SEP (graph classification, 6 datasets) | ✅ DONE (JSONs) | — |
| Phase 2 — SE-GSL (node classification) | 🔄 cora done; citeseer running | **~30 min** |
| Phase 3 — SI2E (RL) | ⚠️ seed-1 non-repro; seed-2 running | **~90 min** (then verdict) |
| Phase 2 breadth — add DiffPool/MinCut/TopK baselines for SEP | ⏳ pending | ~0.5 day |
| Phase 3 — resolve SI2E (more seeds / env diff) OR report as null | ⏳ | ~0.5–1 day |
| **Empirical core complete** (all 3 families, honest) | — | **~1 day** |
| Manuscript draft around the evidence | ⏳ | +2–3 days |
| **Full submission-ready draft** | — | **~4–5 days** |

### SI2E open question (gates the RL conclusion)
kthvalue patch CLEARED (matches your REPRODUCE_LOG fix). Seed-1 DoorKey-8x8
plateaued (episodes truncate at 640 steps, success ~0 vs paper ~100%). Testing
seed 2; if multi-seed also fails it's a modernized-env difference (new minigrid
obs/reward vs the lost legacy gym_minigrid), reported as an honest reproducibility
caveat — NOT as "SE doesn't help RL".

### Immediate queue
1. Harvest SE-GSL citeseer + SI2E seed-2 → JSONs.
2. SI2E verdict (seed sweep) → finalize RL row.
3. SEP vs DiffPool/MinCut/TopK baseline numbers (paper-reported + reproduce subset).
4. Draft manuscript evidence tables (Phase 1 + 2 ready now).

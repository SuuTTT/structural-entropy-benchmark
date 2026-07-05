---
layout: page
title: Results Ledger
description: "A durable registry of every structural-entropy experiment, grid-cell, method, and lever we have run across the SE research program — with its status, honest result, and source artifact. Single source of truth for what we already know, so we never redo an experiment."
---

# SE Program Results Ledger

This is a **durable registry**, not marketing. It logs every experiment, grid-cell,
method, and lever tried across the structural-entropy (SE) research program, with its
outcome and the exact artifact the number lives in. The purpose is to **avoid redundant
re-runs**: before starting any SE experiment, check here for what we already know.

**Anti-fabrication rule:** every result is quoted from an artifact (a result JSON/JSONL,
a paper `.tex`, or a FINDINGS log). Where a source has **no extractable number**, the row
says "not run / no artifact" rather than guessing.

**Status legend:** ✅ settled &nbsp;·&nbsp; ⚠️ low-n / preliminary &nbsp;·&nbsp;
❌ null / inconclusive / retracted &nbsp;·&nbsp; 🔄 running &nbsp;·&nbsp; ⬜ not run (no code / not executed).

Four threads: [Benchmark reproductions](#thread-1--benchmark-reproductions) ·
[Graph-SE](#thread-2--graph-se-attention-prior) ·
[DM-SISA](#thread-3--dm-sisa-rl-levers) ·
[SeSE calibration](#thread-4--sese-llm-calibration).

Row counts: Benchmark **30**, Graph-SE **21**, DM-SISA **15 levers + 5 washed-out**, SeSE **17**.
All paths are absolute on the research box; JSON paths are stable, `.tex`/JSONL cite commit where relevant.

---

## Thread 1 — Benchmark reproductions

Repo root: `/home/ubuntu/structural-entropy-benchmark/`. Registry of methods ×
repos: `registry/methods.yaml`. Result JSONs: `results/<family>/<Method>__<Dataset>__<date>.json`
(arrays = one entry per seed). Verdicts: `FINDINGS_phase1.md`, `FINDINGS_new_tracks.md`, `PROGRESS.md`.

**Headline finding for the registry:** on **topology-only** community detection, SE methods
(deDoc, CoDeSEG) do **not** beat modularity / map-equation / spectral. SE helps only with node
**features** (DeSE) or genuine **block** structure (SBM-Clean). Even with features, SE does not
reliably beat plain Louvain. Do not re-chase "SE beats classical CD on plain graphs" — it is settled negative.

### Community detection — `results/community_detection/`

| What was tried | Status | Result (honest) | Source |
|---|---|---|---|
| deDoc (SE-min, orig Java) × LFR μ0.1–0.6 | ✅ | **Fails**: ARI ≈ 0 at all μ; over-segments to ~993–998 comms (24 true), NMI ~0.62 | `deDoc__LFR-mu0.{1..6}__2026-06-02.json` |
| deDoc × SBM-Clean | ✅ | **Recovers blocks**: ARI 0.990, NMI 0.985, 4 comms | `deDoc__SBM-Clean__2026-06-02.json` |
| deDoc × SBM-Noisy | ✅ | Fails at threshold: ARI 0.0015, 146 comms | `deDoc__SBM-Noisy__2026-06-02.json` |
| deDoc × Karate | ✅ | Over-segments: ARI 0.0151, NMI 0.340, 31 comms | `deDoc__Karate__2026-06-02.json` |
| deDoc scalability N=50–10000 | ✅ | ~O(N^1.4), 74.2s @ N=10k; **refutes "O(N³) infeasible"** claim | `deDoc__SBM-scalability__2026-06-02.json` |
| CoDeSEG (SE-game, orig C++) × LFR μ0.1–0.6 | ✅ | ARI 1.00→0.02; over-segments (140→425 comms) | `CoDeSEG__LFR-mu0.{1..6}__2026-06-02.json` |
| CoDeSEG × SBM-Clean / Noisy / Karate | ✅ | Fails SBM (ARI 0.162 / 0.008); Karate ARI 0.185 | `CoDeSEG__SBM-{Clean,Noisy}__…`, `CoDeSEG__Karate__2026-06-02.json` |
| DeSE (deep SE + features, PyTorch, KDD25) × Cora | ✅ | 5-seed NMI mean ≈ 0.512, ARI ≈ 0.437 (acc/f1 have Munkres bug — use NMI/ARI) | `DeSE__Cora__2026-06-02.json` |
| DeSE × Citeseer | ✅ | 5-seed NMI ≈ 0.400, ARI ≈ 0.399 | `DeSE__Citeseer__2026-06-02.json` |
| DeSE × Photo | ⬜ | Not run — CPU-infeasible (>1h/seed), per FINDINGS_phase1 §5 | no JSON |
| LSENet (hyperbolic SE, ICML24) × Cora | ⚠️ | acc 0.631 / NMI 0.495 / ARI 0.418 (single best-by-NMI point, not 5-seed) | `LSENet__Cora__2026-06-0{2,3,4}.json` |
| LSENet × Citeseer | ⚠️ | **Fails**: NMI 0.067→0.094, ARI 0.070→0.113 (method-fragile) | `LSENet__Citeseer__2026-06-0{2,3,4}.json` |
| LSENet × Photo | ⚠️ | NMI 0.594→0.667, ARI 0.456→0.574 (config fix bumped it) | `LSENet__Photo__2026-06-0{2,3,4}.json` |
| Louvain / Leiden / Infomap / Spectral × LFR/SBM/Karate | ✅ | Baselines; Infomap collapses μ≥0.5 (ARI 0); Spectral best on Karate (ARI 0.882) | `{Louvain,Leiden,Infomap,Spectral}__*__2026-06-02.json` |
| Same 4 baselines × Cora/Citeseer/Photo (topology-only) | ✅ | Louvain Cora ARI 0.246/NMI 0.450; Photo 0.564/0.658; Spectral collapses (NMI ~0.04) | `{…}__{Cora,Citeseer,Photo}__2026-06-03.json` |

> Note: `FINDINGS_phase1.md` quotes LSENet as "5-seed mean±std"; the JSON artifacts hold single best-by-NMI points — the ± is **not** reconstructible from these files. `_summary/accuracy.md` + `_summary/cross_objective.md` are derived tables, not primary sources.

### Graph learning — `results/graph_learning/`

| What was tried | Status | Result (honest) | Source |
|---|---|---|---|
| SEP (SE graph pooling, ICML22, PyG2.5) × 7 TU datasets | ✅ | Reproduced 7/7. test_acc: PROTEINS 0.802, DD 0.810, NCI1 0.804, MUTAG 0.928, IMDB-B 0.781, IMDB-M 0.547, COLLAB 0.829 | `SEP__{…}__2026-06-0{3,4}.json` |
| SEGA (SE contrastive, ICML23) × PROTEINS, IMDB-B | ⚠️ | Single-run unsup SVM: PROTEINS accG 0.751, IMDB-B 0.709. MUTAG (~85-87% in PROGRESS) has **no JSON** | `SEGA__{PROTEINS,IMDB-BINARY}__2026-06-04.json` |
| SE-GSL (SE graph-structure learning, WWW23) × Cora | ✅ | test_acc 0.869±0.015 (GCN); GAT 0.880; SAGE 0.885. Citeseer/APPNP not present | `SE-GSL__{Cora,gat_cora,sage_cora}__2026-06-0{3,4}.json` |
| Baselines DiffPool / MinCutPool / DMoN (SEP comparison, 3 seeds) | ✅ | SEP > DiffPool/MinCut on PROTEINS/NCI1, ~tie MUTAG/IMDB-B | `{DiffPool,MinCutPool,DMoN}__*__2026-06-0{5,6}.json` |

### Other task families

| What was tried | Status | Result (honest) | Source |
|---|---|---|---|
| SI2E (SE intrinsic reward, MiniGrid, PPO) × KeyCorridorS3R1 (easy) | ✅ | Reliably solves: success 0.90–0.91 (3 seeds) | `SI2E__kc1-s{1,2,3}__2026-06-04.json` |
| SI2E × DoorKey-8x8 (medium) | ❌ | **Bimodal**: success {0.94,0.94,0.94,0.25,0.07,0.00} — unstable | `SI2E__{DoorKey-8x8,dk8-s*,si2e}__2026-06-0{3,4}.json` |
| SI2E × RedBlueDoors | ❌ | Bimodal: {0.89, 0.06, 0.82}. (PROGRESS text says 0.76; JSON says 0.82 — trust JSON) | `SI2E__rbd-s{1,2,3}__2026-06-04.json` |
| SI2E × KeyCorridorS3R2 (hard) | ❌ | **Fails**: success 0.00 / 0.02 / 0.00 | `SI2E__kc-s{1,2,3}__2026-06-04.json` |
| Hierarchy quality (Dasgupta cost) — SE-agglom vs average/ward/Paris × 5 graphs | ✅ | SE wins 4/5: best on SBM-Clean/6blk, LFR-μ0.1/0.4; 2nd on Karate (avg wins) | `hierarchy-quality__{Karate,SBM-Clean,SBM-6blk,LFR-mu0.1,LFR-mu0.4}__2026-06-04.json` |
| SSE (semi-supervised SE, UCI, ratio 0.2, 10 reps) × 4 datasets | ✅ | acc: wine 0.930, breast-cancer 0.967, australian 0.739, heart 0.749 | `SSE__{wine,breast-cancer,australian,heart}__2026-06-04.json` |
| UnDBot (SE social-bot detection) × botwiki-2019 | ⚠️ | Single dataset/run: AUC 0.883, F1 0.513 (large sets OOM) | `UnDBot__botwiki-2019__2026-06-04.json` |
| SuperTAD (SE TAD-calling, Genome Biol 21, C++) × Hi-C GM12878/IMR90 | ⚠️ | "20 hierarchical TADs" reported **prose-only** — `results/bioinformatics/` is **EMPTY**, no JSON backs it | `FINDINGS_new_tracks.md` (Track 3), `PROGRESS.md` #10 |

### Registry methods with no results (not run / no code)

| Method (family) | Status | Note | Source |
|---|---|---|---|
| deDoc2 (CD) | ⬜ | Repo verified, not run | `registry/methods.yaml` |
| SEP_HCSE / HCSE builder (GL) | ⬜ | Same repo as SEP; no separate JSON | `registry/methods.yaml` |
| SEAT, SSSE, HiTIN | ⬜ | Repos verified, not run | `registry/methods.yaml` |
| SIDM / SIRD (JMLR25, RL) | ⬜ | Recipe banked; heavy StarCraftII dep (PROGRESS #8 pending) | `registry/methods.yaml`, `PROGRESS.md` |
| COLLAB (SDM25) | ⬜ | Verified, not run | `registry/methods.yaml` |
| SeSE (LLM-UQ) | ⬜ | GPU-gated (≥24GB / gated LLaMa); see Thread 4 for the separate calibration study | `FINDINGS_new_tracks.md` (Track 2) |
| USER, HiPART (GL); SISA, SIRD (RL) | ⬜ | Repo `to-locate` (verified:false) — no code located | `registry/methods.yaml` |

---

## Thread 2 — Graph-SE (attention prior)

Repo: `/home/ubuntu/SE-graph-learning/`. Stage sweeps: `experiments/d6_diffse/results/*.jsonl`.
Master log: `experiments/d6_diffse/FINDINGS.md`. Paper: `/home/ubuntu/se-attention-prior-paper/main.tex`
("The Multi-Level Edge: Structural Entropy as a Hierarchical Prior for Graph Attention", compiled 2026-07-03).

**Headline:** differentiable **multi-level** soft-SE added as an attention-graph regularizer improves
**low-label** node classification; **flat** SE only ties modularity/MinCut. The SE-specific lever is the
**multi-level encoding tree**, confirmed against a capacity control and per-method-λ control. Not SOTA
(below tuned GCNII/APPNP). Program is experimentally **complete**.

| Stage — what was tried | Status | Result (honest) | Source |
|---|---|---|---|
| S9 — k-robustness: reg{none,se,mod,mincut} × k{0..128} × 8 datasets × 7 seeds | ✅ | SE never collapses (worst-case-over-k ≥ −0.004 on all 8); mod collapses Citeseer (−0.020), MinCut Citeseer (−0.038). "SE beats both" fails (5/8); "bounded downside" holds | `stage9_krobust_full.jsonl` (1232 rows) |
| S10 — mega: {none,se,dropedge,selftrain,se+st} × label-rate × 8 × 7 | ✅ | SE > DropEdge 8/8; SE > self-train only **5/8** (self-train wins CoraFull/Photo/CS) | `stage10_mega.jsonl` (728) |
| S11 — λ sweep {0,0.01,0.03,0.1,0.3} | ✅ | Inverted-U, peak **λ ≈ 0.03** | `stage11_lambda.jsonl` (480) |
| S12 — heterophily k-robustness × 5 datasets | ✅❌ | Settled NULL: SE gain Texas +0.012, Cornell −0.018, Wisconsin −0.050, cham −0.010, squirrel −0.004 → inert/harmful 4/5. Confirms homophily-only scope | `stage12_hetero.jsonl` (770) |
| **S13 — multi-level vs flat SE vs mod vs mincut × 8 × 15 seeds** | ✅ | **se_ml − se_flat +2.0pp Wilcoxon p<1e-5 (n=120), 6/8**; se_ml > mod +3.7pp p<1e-6 (7/8); > MinCut +1.4pp p=3e-6 (6/8). MinCut beats se_ml on CoraFull (honest exception) | `stage13_ml.jsonl` (600) |
| S14 — depth L{2,3,4} × 5 datasets × 10 seeds | ✅ | Inverted-U, **L3 peak 4/5**; pooled L3>L2 +2.6pp p=2e-4, L3>L4 +2.3pp p=2e-5 | `stage14_depth.jsonl` (150) |
| S15 — strong backbone (3-layer residual+JK GAT) × 4 × 10 | ✅ | se_ml > se_flat > none 4/4; se_ml>none +11.5pp p<1e-6; se_ml>se_flat +3.3pp p=3e-4 | `stage15_strongbb.jsonl` (120) |
| **S16 — capacity control: ml_16_4 vs flat_{4,16,64} × 4 × 10** | ✅ | ml wins 4/4 (Cora 0.713, Citeseer 0.584, CoraFull 0.278, Photo 0.808); flat_64 worst → **hierarchy, not capacity**, is the lever. Pooled +6.8pp p<1e-6 | `stage16_capacity.jsonl` (160) |
| S17 — planted-hierarchy depth{2,3} × L{2,3,4} × 30 seeds | ✅ | Weak signal: planted-3 peaks at L3 (0.457), L3>L2 p=0.0014, L3>L4 p=0.042; planted-2 ~flat (tie) | `stage17_planted.jsonl` (180) |
| S17b — planted HARD variant, same grid | ❌ | **NULL**: L irrelevant within each depth (all within ~0.001). Unreported negative control; paper uses S17 (has headroom). No committed `.py` | `stage17b_hard.jsonl` (180) |
| S18 — per-method optimal-λ recheck × 4 × λ{4 vals} × 8 | ✅ | Even at each method's own best-λ: se_ml wins Cora/Citeseer/Photo; MinCut wins CoraFull. Reg-strength confound does not explain the win. No committed `.py` | `stage18_lambda.jsonl` (384) |
| D5 — SE-agglomerative hierarchical clustering × 34 graphs | ✅ | SE beats all baselines 33/34; SE vs Paris Wilcoxon p=1.16e-10, median +10.3%. Caveat: Dasgupta cost gameable → switched to NMI (SE 0.897 syn / 0.700 real) | `experiments/d5_hierarchy/FINDINGS.md` |
| D1 — SEP pooling ablation: SE tree vs shape-matched random | ✅❌ | **NULL/negative**: "structurally meaningful, practically irrelevant" — SE trees ≈ random-tree accuracy (PROTEINS .755 ≈ .757) despite 3-4× more cohesive clusters; cohesion-gate doesn't rescue | `experiments/d1_ablation/FINDINGS.md`, `README.md` |
| BH-FDR correction across 24 se_ml-vs-baseline tests | ✅ | 15/24 survive p<0.05; failures = known nulls (DBLP, Pubmed/Computers vs se_flat, CoraFull vs MinCut) | `HANDOFF_TO_FABLE5_2026-07-02.md` |
| Transformer-transfer of the prior | ❌ | Negative (kept in paper Limitations) | `se-attention-prior-paper/main.tex` |
| SOTA comparison (vs tuned GCNII/APPNP, GRAND, C&S) | ❌ | **Not SOTA** — below tuned baselines in absolute accuracy (honest, in Limitations) | `se-attention-prior-paper/main.tex` L349 |

> Provenance flags: S16/S17/S18 cell means were recomputed directly from the JSONL and match FINDINGS/paper to rounding. S17b and S18 JSONL exist without a committed script (backed up in commit `48a2f8b`).

---

## Thread 3 — DM-SISA (RL levers)

Repo: `/home/ubuntu/dm-sisa/` (HEAD `5b85963`). Levers dashboard: `docs/LEVERS.md`.
Paper: `/home/ubuntu/dmsisa-paper/main.tex` (HEAD `ef06b95` = the correction commit).
Baseline to beat: **SISA**.

**Overall verdict (corrected):** the 4-way comparison (RAD-SAC / SISA / DM-SISA λ0.01 / DM-SISA λ0)
is **statistically inseparable** at n=5 (same hardware+seeds). The contribution is **formulation +
efficiency, not a return beat**. The earlier "over-regularization" claim is **RETRACTED** (2-seed artifact).
The real, surviving result is **~16,000× cheaper abstraction** (1.3 ms vs 21.4 s per call).

4-way late-phase return, n=5, RTX 3060 (`main.tex` Table `tab:main`, L166–186):
walker/walk RAD-SAC 652±94 · SISA 552±190 · DM-SISA λ0.01 774±63 · λ0 676±39;
cheetah/run 475±25 · 412±73 · 407±104 · 349±101. Only raw-significant contrast (λ0.01 vs SISA p=0.016)
fails 4-test correction (p=0.064) and leans on one SISA seed collapsing to 191.

### Levers L1–L10 — `docs/LEVERS.md`

| Lever | What it is | Status | Result (honest) | Source |
|---|---|---|---|---|
| L1 | λ_SE = 0 (no aux penalty) | ✅ | Parity + more stable: walker 676±39 vs SISA 552±190 (p=0.22); ~5× lower variance; 16000× cheaper | `docs/LEVERS.md` |
| L2 | λ_SE = 0.01 (default) — does SE over-regularize? | ❌ RETRACTED | **No effect**: walker 774±63 ≥ λ0's 676 at n=5; the 562 (n=2) was an artifact (p=0.10/0.55) | `docs/LEVERS.md`, paper L188–194 |
| L3 | λ_SE = 0.003 | ⚠️ | Small-λ hint: 735.8 (n=1) > λ0's 676, but never reached rigorous n — qualitative only | `docs/LEVERS.md` |
| L4 | λ_SE = 0.001/0.002/0.005 (bracket optimum) | ⬜ | Planned, pending | `docs/LEVERS.md` |
| L5 | Policy-input injection (`--dmsisa_inject`) | ❌ (infra) | Built + trains clean, but every run crashed on flaky box — 0 completions. Infra, not method | `docs/LEVERS.md` |
| L6 | Depth: multi-level vs flat | ⚠️ | Running on single slow GPU; insufficient n | `docs/LEVERS.md` |
| L7 | Deeper 3-level hierarchy | ⬜ | Not run | `docs/LEVERS.md` |
| L8 | Community count k1/k2 sweep | ⬜ | Not run | `docs/LEVERS.md` |
| L9 | Aux-task coefficient tuning | ⬜ | Not run | `docs/LEVERS.md` |
| L10 | Distractor / hard-obs regime | ⬜ | Not run | `docs/LEVERS.md` |
| — | Efficiency (soft-SE vs discrete tree) | ✅ | **~16,000× faster** per abstraction call: 1.32 ms vs 2.14×10⁴ ms | paper Abstract L34–36, `tab:validation` L140–142 |
| — | Correctness gate: one-hot soft-SE = discrete SE | ✅ | \|Δ\| = 1.15×10⁻⁷ (pass) | paper `tab:validation` |
| — | 4-way n=5 return comparison | ✅ | **Statistically inseparable** (see above) | paper `tab:main` L166–186 |
| — | Over-regularization claim | ❌ | **WITHDRAWN** | paper L188–194, commit `ef06b95` |
| — | Stop condition | ✅ | "No lever produced a significant beat" (LEVERS.md closing note, 2026-07-05) | `docs/LEVERS.md` |

### ⚠️ Washed-out signals — DO NOT RE-CHASE

Five single-seed / low-n "wins" that **regressed to parity** under more seeds. These are settled dead
ends; re-running them wastes GPU. (No single enumerated list existed in the artifacts — reconstructed and
traced below.)

| # | Signal (as originally claimed) | Original | Washed out to | Source |
|---|---|---|---|---|
| S1 | Walker "beat" over SISA | Δ+71, 1-sided p=0.036 (n=3) | Δ+35, p=0.41 n.s. (n=4) → parity | `LEVERS.md`; commits `42fa252`→`c97f266` |
| S2 | λ0.01 over-regularizes | 562±41 (n=2), −22% vs λ0 | 774±63 (n=5) ≥ 676 — RETRACTED | `LEVERS.md` L2; paper `ef06b95` L188–194 |
| S3 | DM-SISA(λ0) single-seed win over SISA | 711.4 (n=1) > 691.4 (n=1) | 676±39 vs 725±107 (n=5), p=0.55 n.s. | paper commits `7250357`→`bb926b0` |
| S4 | Naive DM-SISA return cost "−19%" | 558.8 (n=1) | Subsumed into retracted over-reg claim | paper commit `7250357` |
| S5 | λ0.01 vs SISA significance | raw p=0.016 | Fails 4-test correction p=0.064; leans on one collapsed seed (191) | paper `ef06b95` L177–179 |

---

## Thread 4 — SeSE (LLM calibration)

Repo: `/home/ubuntu/sese-calibration-paper/`. **All numbers live in `main.tex`** (Table `tab:cells`
L88–119; prose L73–142; abstract L19–38) — the repo has **no result JSON/CSV**; raw run data is not
committed, so `main.tex` is the durable source. Verdict history in commits `1dcbe9c`, `dd589c3`, `b17ff79`.

**Headline:** the sign of the chain−answer AUROC gap (does chain-level SE beat answer-level SE?) is set by
the model's **family / calibration**, not competence. Only the badly-overconfident **gemma-3-12b** is a
chain-SE winner. **CORR = 0.75 (n=17, gemma-anchored)** between overconfidence and the gap. The cross-family
**Mistral** probe is a clean **negative**. Columns below: Acc / Overconf (peak−acc) / Gap (chain−answer AUROC;
positive = chain-SE wins).

| Model × dataset | Status | Result (Acc / Overconf / Gap) | Source |
|---|---|---|---|
| Qwen2.5-0.5B × GSM8K | ✅ | 0.397 / +0.113 / **−0.007** (answer-SE wins) | `main.tex` L94 |
| Qwen2.5-0.5B × SVAMP | ✅ | 0.508 / +0.124 / −0.066 | `main.tex` L95 |
| Qwen2.5-0.5B × AQuA | ✅ | 0.300 / +0.171 / −0.123 | `main.tex` L96 |
| Qwen2.5-1.5B × GSM8K | ✅ | 0.612 / +0.091 / −0.023 | `main.tex` L97 |
| Qwen2.5-1.5B × SVAMP | ✅ | 0.760 / +0.068 / −0.055 | `main.tex` L98 |
| Qwen2.5-3B × GSM8K | ✅ | 0.827 / +0.021 / −0.069 | `main.tex` L99 |
| Qwen2.5-3B × SVAMP | ✅ | 0.842 / +0.050 / −0.089 | `main.tex` L100 |
| Qwen2.5-7B × GSM8K | ✅ | 0.787 / +0.038 / −0.221 | `main.tex` L101 |
| Phi-3.5-mini × GSM8K | ✅ | 0.875 / +0.038 / −0.080 (answer-SE wins) | `main.tex` L102 |
| gemma-3-12b × GSM8K (s1) | ✅ | 0.410 / +0.400 / **+0.122** (chain-SE wins) | `main.tex` L103 |
| gemma-3-12b × GSM8K (s2) | ✅ | 0.385 / +0.409 / +0.157 | `main.tex` L104 |
| gemma-3-12b × GSM8K (s3) | ✅ | 0.430 / +0.358 / +0.183 | `main.tex` L105 |
| gemma-3-12b × SVAMP | ✅ | 0.340 / +0.460 / +0.083 (chain-win robust across task) | `main.tex` L106 |
| Mistral-7B × AQuA (cross-family probe) | ⚠️ | overconf +0.19, gap −0.09 (answer-SE wins), n=2 → clean **NEGATIVE**; Mistral not in chain-winning regime | `main.tex` L132–139, commit `1dcbe9c` |
| Mistral-7B × GSM8K | ❌ | **EXCLUDED** (parse artifact): 73% empty extracted answers → spurious; removed by >25%-empty validity filter | `main.tex` L108–109, L121–125 |
| CORR(overconfidence, gap) headline | ✅ | **ρ = 0.75 (n=17)**, gemma-anchored, validity-filtered set | `main.tex` L31–32, L83, L139; commit `1dcbe9c` |
| Degenerate-answer-graph mechanism | ❌ | **Falsified**: outlier's answer diversity ≈ strong model's (qualitative, no numeric artifact) | `main.tex` L78–81 |

Not run / open (no numbers — do not invent): full **AQuA** column for Qwen/Phi/gemma is 🔄 running
(`main.tex` L117); **ECE / selective-generation curves** are ⬜ not run (`\TODO` L141–142) — no ECE value
exists anywhere in the repo; additional overconfident models (gemma-2, Llama-3.2) ⬜ gated/inaccessible.

> Internal-consistency flag: the initial 13-cell correlation is reported as **0.825** (L83, commit `dd589c3`) but rounded to **0.83** in the abstract (L33, commit `1dcbe9c`); the current headline is the n=17 value 0.75.

---

## What is already settled (do not redo)

- **Topology-only community detection:** SE (deDoc, CoDeSEG) does **not** beat modularity/map-eq/spectral. SE needs features or block structure. (Thread 1)
- **DM-SISA return:** no lever beats SISA; 4-way is statistically inseparable at n=5. The contribution is the 16,000× efficiency, not returns. The 5 washed-out signals above are **dead ends**. (Thread 3)
- **DM-SISA over-regularization:** retracted — do not report it. (Thread 3)
- **Graph-SE flat objective:** flat SE only ties modularity/MinCut; only **multi-level** SE is the real lever, and it is **not SOTA**. Heterophily is out of scope (inert/harmful). D1 pooling ablation is a settled NULL. (Thread 2)
- **Graph-SE planted-depth (hard regime, S17b):** NULL — the depth-peaks-at-L result does not survive; don't cite it. (Thread 2)
- **SeSE:** chain-SE only wins for badly-overconfident models (gemma-3); cross-family Mistral is negative. (Thread 4)

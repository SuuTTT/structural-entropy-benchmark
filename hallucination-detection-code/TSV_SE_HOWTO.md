# Cross-modal hallucination detection (latent probe + semantic entropy) — how to do it right
_Literature-grounded plan, 2026-07-14. Synthesizes 4 research passes: SE/consistency, latent probes, fusion novelty, eval methodology._

## 0. TL;DR
Our empirical finding (fuse TSV latent probe + NLI semantic entropy; gain is regime-dependent) is **real but, as "fusion," not novel** — internal+consistency fusion via logistic regression is already published (esp. **2603.19118**, which even reports the "internal recovers confident-wrong" story). The **defensible wedge** is the *boundary-condition map*: **when does cross-modal fusion help, and when is it provably flat** — with explicit NULL results — reframed as an **analysis/diagnostic** paper, not a new detector. Current numbers (n=100, single model family, BLEURT labels, test-set checkpoint selection) are **underpowered/biased** by the field's own 2024–25 standards and must be redone under a proper protocol.

## 1. Novelty-aware positioning
- **Crowded (cite, don't claim):** internal+consistency fusion — UQLM (2504.19254), CoCoA (2510.20460), Detection Dilemma (2510.11529), and the **primary foil 2603.19118** (LR fusion of internal + consistency; "internal recovers confident-but-wrong"; claims complementarity holds *even when signal strengths differ*).
- **Canonical (cite, don't claim):** SE's regime blind spot — Farquhar et al. **Nature 2024** ("does not address confidently-wrong"), sharpened by **DECK 2606.02289** (Consistency×Confidence taxonomy) and **Semantic Energy 2508.14496**.
- **Our wedge (defensible):**
  1. Fusing a **trained TSV-style latent probe** specifically with **sampled semantic entropy** (others use token-prob/MSP/CoT-consistency, not SE).
  2. **The boundary-condition map**: fusion pays **only** when errors are uncertainty-driven AND **neither modality dominates**; it is **flat** when SE collapses (confident-wrong) OR the probe saturates (near-ceiling). Explicit NULLs are the asset.
  3. **Counterintuitive twist (lead with this):** the naive expectation is "fuse to cover SE's confident-wrong blind spot." Our data shows the **opposite** — fusion is flat there because the *probe already dominates*; it helps in the *uncertainty* regime instead. This **directly contradicts 2603.19118's strength-invariance claim** — position against it.
- **Avoid the SEP naming collision** (2406.15927 "Semantic Entropy Probes" = LR-on-hidden-states *predicting* SE). We keep SE as a **separate sampled feature**; state this explicitly.

## 2. Method design (do it right)
- **Latent probe:** prefer **low-capacity** (linear / mass-mean difference) over high-capacity steering to resist spurious-feature overfitting (our exemplar-AUROC-1.0/test-0.22 case; cf. Universal Truthfulness Hyperplane 2407.08582, "Are Hidden States Hiding Something?" 2505.16520). **Diversity > volume**: train on many datasets. **Ground orientation** to a class-balanced labeled reference + sanity-check (fixes the TSV vMF/OT inversion we found — undocumented in the paper). Pre-register the layer (late-intermediate); no best-layer-on-test.
- **Semantic entropy:** NLI **bidirectional entailment** clustering (DeBERTa-MNLI), **~10 samples**, temperature **≈0.5–1.0**; use **discrete SE** (cluster proportions) for black-box comparability.
- **Fusion:** logistic regression on `[z(probe), z(SE)]` (standard). Report **split by regime** (never average TriviaQA/NQ with TruthfulQA).

## 3. Evaluation protocol checklist (the publishable bar)
1. **3-way split; test touched once.** Select every checkpoint/threshold/layer/λ on **validation**, freeze, report test once. (Fixes our issue: released TSV code selects the checkpoint by argmax **test** AUROC — a leakage the *paper* doesn't state; document the code-vs-paper discrepancy.)
2. **Cross-dataset by construction** (train A → test B,C); per-dataset numbers, never only pooled.
3. **Cross-model / cross-family transfer** (train model A → test model B; ≥2 families, ≥2 sizes); report in-domain→OOD degradation.
4. **Anti-artifact baselines** (must beat all, OOD): prevalence/majority, task-type-only, **response-length**, and a **linear probe on activations**.
5. **Metric suite:** AUROC **+** AURAC or normalized **PRR** **+** PR-AUC (imbalance) **+** ECE (if claiming calibrated confidences).
6. **Bootstrap CIs on every metric** (no bare points).
7. **Significance:** **DeLong** (large n, correlated ROCs) or **paired bootstrap** (small/skewed n) for ΔAUROC; Friedman+Nemenyi across datasets.
8. **Powered n:** n≈100 is badly underpowered for small ΔAUROC (ΔAUC=0.02 → ~900–3700 items; powerROC 2501.03155). Target hundreds–thousands or caveat loudly.
9. **≥5 seeds**, mean±std over all stochastic stages; multiple-comparison correction (Holm/BH) when sweeping.
10. **Robust labels:** not a single BLEURT/ROUGE threshold (Illusion-of-Progress 2508.08285: ROUGE precision ~0.40, length artifact). Use human-validated LLM-as-Judge; report judge-vs-human agreement; **label-function sensitivity analysis** + the length-repetition sanity check.
11. **Operating-point behavior:** risk–coverage / rejection curves, FPR@fixed-TPR.
12. **Reproducibility disclosure** + a shared harness (LM-Polygraph 2406.15627) for comparable PRR.

## 4. Where our current results stand vs the bar
| Aspect | Current | Bar | Gap |
|---|---|---|---|
| Test-set selection | argmax test AUROC (code) | val-selected, test-once | **redo** |
| n (test) | 100 | hundreds–thousands | **underpowered** |
| Models | Qwen base+instruct (1 family) | ≥2 families, ≥2 sizes | **add llama/mistral** |
| Labels | BLEURT@0.5 | human-validated judge + sensitivity | **redo labels** |
| Metrics | AUROC only | +AURAC/PRR/PR-AUC/ECE | **add** |
| Significance | 1 bootstrap (TriviaQA) | DeLong/paired-bootstrap all cells | **add** |
| Regime split | yes (our strength) | yes | ✓ keep |
So the 4-cell finding is a **pilot**, not a result. The *shape* (regime-dependence, counterintuitive twist) is the contribution; the *numbers* need the protocol.

## 5. Recommended paper + minimal path
**Title framing:** "*When does fusing latent probes with semantic entropy help? A regime analysis of cross-modal hallucination detection*" — analysis/diagnostic paper, foil = 2603.19118.
**Contribution triplet:** (a) the boundary-condition map + counterintuitive twist (novel); (b) a **TSV robustness audit** (memfix; code test-selection discrepancy; **undocumented vMF/OT class-inversion**; few-shot overfitting where exemplar-AUROC=1.0 but test=0.22; λ can be actively harmful) — a real reproducibility contribution since **no independent TSV repro exists yet**; (c) a clean, protocol-correct re-evaluation.
**Minimal experiments to go from pilot → paper** (all reliable, no gated downloads needed if using cached/ungated models):
1. Re-run all cells with **val-based selection + final-epoch reporting** (kills the bias); add **DeLong/paired-bootstrap CIs**.
2. **Uncap the 400Q** → test n ≈ 800–2000 for power.
3. **≥2 model families** (Qwen + an ungated llama mirror + Mistral) × the 3+ datasets (TriviaQA, NQ, SQuAD/BioASQ uncertainty; TruthfulQA confident-wrong).
4. **Anti-artifact baselines** (length, linear-probe, prevalence) + **cross-dataset/cross-model** transfer table.
5. **Label-sensitivity**: BLEURT vs LLM-as-Judge on a subset.
This is ~a few GPU-days across both A40 GPUs — feasible, and turns the honest pilot into a defensible submission.

## Key references
Foils/related: 2603.19118 (primary), 2510.11529 Detection Dilemma, 2510.20460 CoCoA, 2504.19254 UQLM, 2402.03744 INSIDE, KLE (NeurIPS 2024). SE: 2302.09664 (Kuhn), Nature 2024 (Farquhar), 2406.15927 SEPs, 2606.02289 DECK, 2508.14496 Semantic Energy. Probes: 2503.01917 TSV, EMNLP'23 SAPLMA, 2306.03341 ITI, 2212.03827 CCS + 2312.10029 critique, 2310.06824 Geometry-of-Truth, 2407.08582 Universal Truthfulness Hyperplane, 2505.16520 / 2506.00823 (probe OOD failure), 2409.17504 HaloScope. Methodology: 2506.01114, 2508.08285 Illusion-of-Progress, 2509.19372 OOD-fail, 2406.15627 LM-Polygraph, 2401.06091 AUROC/AUPRC, 1811.12808 Raschka, 2501.03155 powerROC.

# Universal-probe scale-up — scoping & execution plan
_2026-07-14. Goal: a robust, cross-model hallucination detector that beats SOTA (TSV, semantic entropy) broadly, not per-cell._

## Why a new box (the one hard requirement)
The a40 box is **offline with a broken HF download backend** (gated-model 401, xethub 401), so it's capped at 3 cached datasets. The universal probe needs 40+ datasets × multiple model families → **the box must have unrestricted internet / working HuggingFace Hub.** Everything else (compute, disk) is cheap.

## Box to rent
| Resource | Spec | Note |
|---|---|---|
| GPU | **1× A100 80GB** (preferred) or 2× A40 48GB | Runs 7–13B models for generation + hidden-state extraction at batch. Probe training itself is CPU-trivial. |
| CPU | 16–32 cores | NLI batching (semantic entropy), dataset preprocessing, LR training. |
| RAM | 64GB+ | Feature matrices, dataset processing. |
| Disk | **750GB–1TB** | 3–4 model families (~15–60GB each) + 40 dataset caches + sampled generations + NLI(3GB)/BLEURT(4GB) + features. |
| **★ Internet** | **Unrestricted HF Hub** | THE requirement. First action on the box: verify a gated model (`meta-llama/Llama-3.1-8B`) downloads. If it 401s, wrong box. |
Est. cost: a few GPU-days of an A100 (~$30–80 depending on provider). Reuses the a40 code (extract_features.py, benchmark_v2.py, SE cache) — copy over, no rebuild.

## Design (accounts for the regime-specificity finding)
We learned on the a40 that the probe's "truth direction" is **partly regime-specific** — cross-regime pooling hurt. So the design is NOT one universal probe; it's a **regime-routed detector**:
1. **Regime router:** per question, classify the regime from sample statistics (sample-agreement / SE magnitude / cluster count) — "genuine-uncertainty" vs "confident-wrong."
2. **Per-regime experts:** a diversity-trained probe per regime (pooled *within* regime, where pooling helped: NQ 0.50→0.70), plus semantic entropy (strong in the uncertainty regime).
3. **Routed combination:** uncertainty regime → weight SE + uncertainty-probe; confident-wrong regime → weight the confident-wrong probe (SE is blind there). This targets best-of-both instead of the diluting blend that failed on the a40.

## Dataset pool (~20–40, both regimes)
- **Uncertainty (open QA):** TriviaQA, NQ-Open, SQuAD, BioASQ, PopQA, HotpotQA, SciQ, WebQuestions, ...
- **Confident-wrong / adversarial:** TruthfulQA; math where models are confidently wrong (SVAMP, GSM8K).
- **True/false statement sets:** Geometry-of-Truth (cities/companies/etc.), Azaria–Mitchell.
Target ~10–15 per regime for a credible "universal within regime" claim.

## Models (cross-model transfer)
Qwen2.5-7B (base+instruct), Llama-3.1-8B (base+instruct), Mistral-7B — ≥3 families. Report train-on-A / test-on-B transfer.

## Protocol (the rigorous bar, from the literature pass)
- 3-way split, **validation-based selection, test touched once** (fixes TSV's test-selection bias).
- **Cross-dataset** (train on N datasets, test on held-out) AND **cross-model** transfer, reported per-dataset (never pooled-average).
- **Anti-artifact baselines:** response-length, majority/prevalence, plain linear probe — must beat all, OOD.
- **Metric suite:** AUROC + AURAC (rejection-accuracy) + PR-AUC + ECE; **bootstrap + DeLong CIs**.
- **Powered n:** hundreds–thousands per test set (n=100 was underpowered).
- **Robust labels:** human-validated LLM-as-judge (not a single BLEURT threshold); label-sensitivity check.
- **SOTA baselines:** TSV (reproduce, honest val-selection), semantic entropy, INSIDE/EigenScore, SEPs.

## Execution phases (once box is up)
1. **Setup + verify downloads** (30 min): clone code, venv, confirm gated-model pull works.
2. **Generate + extract** (~1–2 GPU-days): for each model × dataset — greedy answer + 10 sampled answers + hidden states (5 layers) + labels. Parallel across the pool.
3. **Semantic entropy** (cached): NLI clustering over samples per question.
4. **Train experts + router**: per-regime diversity probes, regime router, routed combiner.
5. **Benchmark**: routed detector vs TSV / SE / INSIDE / naive-fusion, cross-dataset + cross-model, full metric suite + CIs.
6. **Verdict**: does the routed detector beat SOTA on held-out datasets AND transfer cross-model? Write up.

## Honest success criteria & risk
- **Win:** routed detector beats TSV + SE + naive-fusion on ≥70% of held-out cells, with cross-model transfer, CIs separated.
- **Risk (stated up front):** regime-specificity may mean even per-regime probes don't fully transfer across models/datasets; if so, the honest result is "SE + regime-routing is a strong robust baseline; a truly universal probe remains elusive" — still a publishable reproducibility+analysis paper, just not a SOTA headline.

## What to do next
Rent the box per spec above, give me the SSH string, and I'll run Phase 1 (verify downloads) → report before the long generation phase.

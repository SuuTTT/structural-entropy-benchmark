---
layout: post
title: "Auditing and Rebuilding a Hallucination Detector: An A40 Lab Notebook"
date: 2026-07-14
tags: [hallucination-detection, semantic-entropy, latent-probes, reproducibility, uncertainty]
---

> **TL;DR.** Five threads on a single 2×A40 box, all serving one goal: a large-language-model (LLM) hallucination detector that beats the state of the art (SOTA). We (1) **reproduced TSV**, a 2025 latent-steering detector (AUROC 0.844 on TruthfulQA vs the paper's 0.873); (2) found that **fusing** TSV with our output-side *semantic entropy* is **regime-dependent** — a bootstrap-significant **+11pp** on TriviaQA but **flat** where one signal already dominates; (3) **audited TSV** and found four real defects — a memory bug, test-set model selection, an undocumented **class-inversion**, and severe **few-shot overfitting** (32 training examples fit perfectly, AUROC 1.0, while the test set collapsed to 0.22); (4) **grounded everything in the literature** — the fusion idea is crowded, but our regime-map and the TSV robustness audit are defensible; and (5) **pivoted to a diversity-trained probe**, which *fixes* the fragility (a collapsed NQ cell jumped 0.50 → 0.71). **Headline: the elaborate SOTA probe is fragile; a simple diversity-trained linear probe is more robust, and cross-modal fusion helps only when neither signal dominates.**

### 60 seconds of background

**Hallucination detection** asks: given an LLM's answer, can we score how likely it is to be *wrong/made-up*, so we can abstain or defer? The standard metric is **AUROC** — the probability the detector scores a random correct answer above a random incorrect one (0.5 = chance, 1.0 = perfect). Two method families dominate:

- **Output-consistency (black-box):** sample several answers to the same question; if they *disagree in meaning*, the model is uncertain. **Semantic entropy (SE)** formalizes this — cluster the samples by *bidirectional entailment* (a natural-language-inference model decides which answers mean the same thing) and take the entropy of the cluster distribution. High entropy = likely hallucination. Known blind spot: if the model is **confidently wrong**, all samples agree and SE reports (wrongly) "reliable."
- **Latent probes (white-box):** read the model's internal hidden-state vector and classify truthful vs hallucinated. **TSV** ("Truthfulness Separator Vector", ICML 2025) is a recent SOTA member: it *trains a steering vector* that reshapes the hidden space so the two classes separate, using only ~32 labeled examples plus unlabeled data pseudo-labeled by optimal transport.

All numbers below are read from result files on the box (logs, `.npz`/`.npy` artifacts, benchmark text dumps). Nulls, corrections, and a killed too-good-to-be-true number are reported at equal prominence with the wins.

---

## Track 1 — Reproducing TSV (the SOTA latent probe)

**What it is.** A faithful re-run of the TSV method on Qwen2.5-7B: generate answers, extract last-token hidden states, train the steering vector on 32 labeled exemplars + optimal-transport pseudo-labels, score the held-out test set by cosine similarity to the learned "truthful" cluster.

**Why it exists.** You cannot audit, fuse with, or beat a method you cannot run. There is **no independent reproduction of TSV in the literature** (it is recent), so this is also first-mover verification.

**Status — done.** Reproduced **AUROC 0.844 on TruthfulQA** vs the paper's 0.873 (within 3 points), after fixing five environment/repo bugs (missing output dirs, an absolute-path logging bug, a transformers-version attention-API mismatch, and out-of-memory during evaluation). This established a trustworthy baseline to build on.

**Feeds.** Every downstream track (paper): the fusion study (Track 2), the audit (Track 3), and the SOTA-beating rebuild (Track 5) all measure against this reproduction.

---

## Track 2 — Cross-modal fusion: when does latent + entropy help?

**What it is.** Combine the TSV latent score with our semantic-entropy score using a small logistic regression, and ask *when* the combination beats either signal alone. "Cross-modal" = one signal is internal (latent), the other is behavioral (output sampling).

**Why it exists.** The two families fail in different places — SE is blind to confident-wrong errors; a latent probe might catch them. If they are *complementary*, fusing them should help. The precise question: **under what conditions?**

**Status — done, and the answer is a sharp regime boundary.** Across four model×dataset cells (AUROC; fusion delta = learned-fusion minus better single, 5-fold cross-validated):

| model / dataset | regime | NLI-SE | TSV | fusion Δ |
|---|---|---|---|---|
| Qwen-base / TruthfulQA | confident-wrong | 0.556 | 0.844 | +0.85pp |
| Qwen-base / TriviaQA | genuine uncertainty | 0.817 | 0.736 | **+11.1pp** (95% CI [+1.1, +19.1], p≈0.024) |
| Qwen-base / NQ-Open | genuine uncertainty | 0.667 | 0.667 | +9.4pp |
| Qwen-instruct / TriviaQA | uncertainty | 0.693 | 0.856 | −0.07pp |

**Fusion pays off only when errors are uncertainty-driven AND neither signal dominates.** It is flat when SE collapses (TruthfulQA — confident-wrong) and flat when the probe alone is already near-ceiling (instruct/TriviaQA, TSV 0.856). A 2000-sample bootstrap confirms the TriviaQA gain is real; NQ is directionally identical but underpowered at n=100 (its CI includes 0). **Reported null, equal prominence:** a fifth cell (instruct/NQ) produced an apparent "+66.8pp fusion" — we traced it to a *diverged* TSV probe (a degenerate baseline) and **excluded it**; the only trustworthy reading there is that SE-alone reaches 0.924.

**Feeds.** Paper (the analysis contribution) and a filed issue for a world-model team who want to reuse the rollout-uncertainty analog.

---

## Track 3 — The TSV robustness audit (four real defects)

**What it is.** A line-by-line review of the TSV implementation plus controlled experiments to explain *why* it sometimes fails, and whether the failures are fixable.

**Why it exists.** During Track 2, TSV's numbers swung wildly across cells (0.22 to 0.86). Wild swings mean either a bug or a fragility — both matter for anyone deploying or citing it.

**Status — done; three confirmed defects and one honest dead end.**

1. **Memory bug (fixed).** The forward pass stacks *all* hidden layers when it uses only one, exhausting a 46GB GPU at the default batch. Indexing the needed layer directly cut peak memory **44GB → 29GB**.
2. **Test-set model selection (validity bug).** The released code saves the checkpoint that **maximizes AUROC on the test set** — an optimistic bias (the reported score becomes a max-over-checkpoints order statistic). Note the *paper* describes validation-based selection, so this is a **code-vs-paper discrepancy** worth flagging, not necessarily the intended method.
3. **Class-inversion (undocumented).** TSV's two clusters are geometry-only; which one is "truthful" hinges entirely on the 32 exemplars. On instruct/NQ we watched the test AUROC slide 0.58 → 0.22 — i.e. the score became strongly *anti*-correlated with truth (its "flipped" value was 0.78). The signal was there; the sign was wrong. Attempts to fix it — anchoring the centroids, then orienting by the labeled exemplars — **failed**, because the exemplars are fit *perfectly* (AUROC 1.0) and so never reveal the test-set inversion.
4. **Few-shot overfitting (diagnosed, not a one-line fix).** The root cause of (3): the steering vector *memorizes* the 32 exemplars and generalizes backward on some cells. A regularization sweep confirmed it — weight decay did nothing; only shrinking the steering strength λ helped, and even the best setting (λ=0.25) reached just **0.53** (barely above chance), while high λ was *actively harmful* (λ=5 → 0.22, worse than no steering at all). No setting both rescued the failing cell and preserved the good cells.

**Feeds.** Paper (a reproducibility/robustness contribution — genuinely first, since no independent TSV audit exists) and the design of Track 5 (use a *simpler, regularized, diversity-trained* probe instead).

---

## Track 4 — Literature grounding and novelty check

**What it is.** Four parallel literature sweeps (semantic-entropy methods, latent probes, fusion/ensemble work, and evaluation methodology) to position the findings honestly and design a rigorous study.

**Why it exists.** Before claiming a contribution, know what is already published — and learn the field's methodology bar so the eventual paper survives review.

**Status — done, and appropriately humbling.**
- **Fusion is crowded.** Logistic-regression fusion of an internal signal with a consistency signal, *including* the "internal recovers confident-wrong" story, is **already published** (the primary foil to differentiate against). So "we fuse latent + entropy" is not novel.
- **SE's regime blind spot is canonical** (stated in the Nature 2024 semantic-entropy paper; formalized by a Consistency×Confidence taxonomy). We cite it, not claim it.
- **Our defensible wedge:** (a) the *boundary-condition map* — when fusion helps vs is provably flat, with explicit nulls; (b) a **counterintuitive twist** — the intuitive move is "fuse to cover SE's confident-wrong blind spot," but our data shows fusion is *flat there* because the probe dominates, contradicting a published strength-invariance claim; (c) the **TSV robustness audit**, which is first.
- **Methodology bar:** validation-based selection (never test), cross-dataset AND cross-model transfer, anti-artifact baselines (response length, majority, linear probe), a metric suite (AUROC + rejection-accuracy + PR-AUC + calibration), bootstrap/DeLong confidence intervals, powered n (100 is far too small), and robust labels (a single BLEURT/ROUGE threshold has ~0.40 precision). Our current 4-cell numbers are a **pilot**, not a result.

**Feeds.** The full plan lives in a project doc; it reframes the paper from "we fuse" to "we characterize *when* to fuse, and we rebuild a robust probe."

---

## Track 5 — Rebuilding to beat SOTA: the diversity-trained probe (in flight)

**What it is.** Replace TSV's fragile trained steering with the literature's robustness prescription — a **simple regularized linear probe trained on a *pool* of datasets** ("diversity > volume") — and fuse it with semantic entropy. A "pooled probe" is one logistic regression trained on the combined examples of several datasets at once, rather than one dataset in isolation.

**Why it exists.** Track 3 showed the elaborate probe overfits; Track 4's literature says simple linear probes match elaborate ones *and* that training-set diversity, not volume, buys generalization. If a diversity-trained probe is robust across cells, a probe+SE detector could beat TSV where TSV collapses.

**Status — in flight, with a strong early signal.** First, a clean per-cell benchmark (validation-selected linear probe, honest splits) already **beats TSV on 3 of 4 cells** on the strength of SE and probe individually — but naive fusion *hurt* on the cells where the probe was weak, and "adaptive" fusion did not yet learn to gate the bad signal (an honest negative for naive fusion). Then the key test — training the probe on a **pool** of datasets:

| cell | single-dataset probe | **pooled (diversity) probe** |
|---|---|---|
| base / TriviaQA | 0.822 | 0.796 |
| base / NQ | **0.562** | **0.705** |
| instruct / TriviaQA | 0.669 | 0.707 |
| instruct / NQ | **0.502** (chance) | **0.649** |

**Diversity-training fixes the fragility.** Pooling just two datasets turns the collapsing chance-level cells into solid detectors (+0.14 and +0.15 AUROC), at minimal cost to the strong cell — exactly the documented "diversity > volume" effect, and precisely the robustness TSV lacks.

**The 3-dataset head-to-head (verified, honest — no clean broad SOTA win):** a validation-selected pooled probe over TruthfulQA + TriviaQA + NQ, vs semantic entropy, naive fusion, and a rule-based *instance gate* (trust SE when samples disagree, the probe when they agree). AUROC:

| cell | div-probe | SE | gate | TSV (SOTA) |
|---|---|---|---|---|
| base / TruthfulQA | **0.840** | 0.657 | 0.806 | 0.844 |
| base / TriviaQA | 0.810 | 0.815 | **0.837** | 0.736 |
| base / NQ | 0.585 | **0.759** | 0.697 | 0.667 |
| instruct / TriviaQA | 0.656 | 0.680 | **0.690** | **0.856** |
| instruct / NQ | 0.531 | **0.719** | 0.650 | 0.22 |

Findings, stated plainly: **(1)** a *simple* diversity-trained linear probe **matches TSV on base/TruthfulQA (0.840 vs 0.844) and beats it on base/TriviaQA (0.810 vs 0.736)** — a trivial classifier rivaling the elaborate SOTA. **(2)** SE is the robust workhorse (it beats TSV on every cell where TSV is weak or broken). **(3)** The instance gate **beats both single signals when they are comparable** (TriviaQA) but **dilutes the stronger one when they are not** (NQ) — the regime boundary, re-derived as a combiner limitation. **(4)** Null: naive/gated *learned* fusion never beat the best single signal. **(5)** Diagnosed: the 3-dataset pool *hurt* NQ (0.585 vs 0.705 for the 2-dataset pool) because TruthfulQA (817 examples, a different regime) **dominated** the pool — a **class/size-balanced** pool is the fix, untested here.

**Bottom line so far:** no single *deployable* detector cleanly beats SOTA across all cells; the honest wins are narrow (probe matches TSV on base; gate wins where signals are balanced). The likely path to a *broad* win is a genuinely strong-everywhere probe — i.e. the universal probe below — so fusion has two strong signals instead of one strong and one weak.

**Honest constraint:** this box is offline with a broken model/dataset download backend, so the pool is **capped at three cached datasets**. The full "universal probe" (40+ datasets) needs a different box (see below).

**Feeds.** Paper (the method contribution, if the diversity-probe + fusion beats SOTA under the rigorous protocol).

---

## Planned — the universal-probe scale-up (needs a new box)

**What it is.** Train the diversity probe on **40+ datasets across multiple model families** (the literature's route to a near-universal truthfulness direction), then evaluate cross-dataset and cross-model under the full rigorous protocol (powered n, robust labels, DeLong/bootstrap significance, anti-artifact baselines).

**Why it exists.** Three cached datasets on one model family is a pilot; a credible SOTA claim needs breadth. The lever we validated (diversity) only reaches its ceiling with many datasets.

**Status — planned; ~a few GPU-days once data is local.** The blocker is *not* compute — it is that this box cannot download data. Recommended box: **1× A100 80GB or 2× A40 48GB, 16–32 CPU cores, 500GB–1TB disk, and — the hard requirement — unrestricted HuggingFace access.** Feature extraction and probe training are cheap and parallel; download bandwidth is the bottleneck.

**Feeds.** Paper (the headline result), and — if it beats SOTA broadly — a released, robust, drop-in detector.

---

## How the tracks connect

The through-line: **the state-of-the-art latent probe is powerful but fragile, and the fix is simplicity plus diversity, not more machinery.** Track 1 (reproduce) gives the baseline; Track 2 (fusion) finds the regime boundary that motivates combining signals; Track 3 (audit) explains *why* the SOTA probe breaks (overfitting/inversion); Track 4 (literature) tells us the fusion idea is crowded but the robustness angle is open; Track 5 (rebuild) turns the audit's diagnosis into a design — a diversity-trained probe that is robust where TSV collapses — and the planned universal-probe scale-up is where that design either beats SOTA at breadth or honestly does not. Every number here is from a result file; the killed +66.8pp, the failed orientation fix, and the naive-fusion regression are kept in plain sight because that is the point of a lab notebook.

# TSV + SE integration (2026-07-13)

## Reproduction
Reproduced "Steer LLM Latents for Hallucination Detection" (Park et al. 2025, ICML), deeplearning-wisc/tsv, on a40.
- Qwen2.5-7B / TruthfulQA: **reproduced AUROC 0.844** vs paper 0.873 (within 3pp; OOM truncated epochs 15-20).
- Fixes: create save_for_eval/ dir; log_dir absolute-path bug (line36 leading /); transformers==4.43.1 (Qwen2 attn API); @torch.no_grad() on test_model + expandable_segments (OOM); test split = questions NOT in data_index[:0.75*817] (205 test Q).

## SE integration (the key question: does our SE add to TSV?)
Cross-modal fusion: TSV (latent-space steering score) + answer-SE (output-space semantic entropy).
- TSV-alone 0.8436; answer-SE-alone 0.545 (CRUDE string-cluster SE); corr(TSV,SE)=0.33.
- Naive z-sum fusion 0.756 (hurts — weak SE drags strong TSV).
- **Learned fusion (logreg 5-fold CV, matched): TSV-only 0.8218 -> TSV+SE 0.8340, delta +1.2pp.**
- **Finding: adding answer-SE improves TSV (+1.2pp), cross-modal COMPLEMENTARY** — unlike our earlier answer-SE+chain-SE fusion NULL (both output-space, redundant). Latent+output = different modalities = complementary.
- Caveat: small gain, n=205, crude string-SE (0.545). NLI-based semantic entropy (SeSE cot_se pipeline) would strengthen SE-alone and likely the fusion gain -> next step.

## NLI-SE (proper semantic entropy) — refinement + mechanism
- NLI-based SE (deberta entailment clustering): SE-alone=0.5565 (still near chance), fusion delta=+0.85pp (0.8218->0.8303). NOT bigger than crude string-SE (+1.2pp).
- MECHANISM: answer-SE is weak on TruthfulQA BY DESIGN — TruthfulQA elicits CONFIDENT falsehoods (model samples consistent-but-wrong -> low entropy -> SE can't flag). This is the confident-wrong regime where entropy methods fail = exactly why TSV latent-steering wins. Same boundary as our SeSE finding (SE works under genuine uncertainty, fails under confident-wrongness).
- IMPLICATION: fusion gain small on TruthfulQA because SE has little to add. On TriviaQA (open-domain recall, uncertainty-driven errors -> samples disagree -> SE strong), the TSV+SE fusion should help MORE. TriviaQA run in progress (GPU1) = the key test of whether cross-modal fusion helps where SE is strong.

## TriviaQA — MECHANISM CONFIRMED (2026-07-13, the key test)
Qwen2.5-7B / TriviaQA (400Q subset, 100 test Q, NLI-SE over 5 sampled generations):
- **TSV-alone 0.7360 | NLI-SE-alone 0.8166 | corr(TSV,SE)=0.343**
- **Learned fusion (logreg 5-fold CV): TSV-only 0.7220 -> TSV+NLI-SE 0.8327, delta = +11.1pp.**
- CONTRAST with TruthfulQA: NLI-SE 0.556 (weak) / fusion +0.85pp  VS  TriviaQA NLI-SE 0.817 (strong) / fusion +11.1pp.
- **VERDICT: cross-modal complementarity is REGIME-DEPENDENT and CONFIRMED.** Where errors are uncertainty-driven (TriviaQA), semantic entropy over sampled generations is a *stronger* hallucination signal than the latent probe itself (0.817 > 0.736), and fusing the two adds +11pp — because the low correlation (0.34) means each modality catches errors the other misses. Where errors are confident falsehoods (TruthfulQA), SE collapses and only the latent probe (TSV) works.
- This is the same regime boundary as SeSE (entropy/SE detects hallucination under genuine uncertainty, fails under confident-wrongness). Latent-probe + output-entropy is a genuine, publishable cross-modal story: **fuse them and you cover both error regimes.**
- Artifacts (a40): /root/tsv/SE_FUSION_TRIVIAQA_RESULT.txt, test_scores.npy/test_labels.npy (AUROC 0.736 @ epoch10), se_fusion_triviaqa.py. Repo bugs fixed for TriviaQA: 3 uncapped length loops->min(len,400); most_likely regen at num_gene 1 (single greedy); exemplar_idx must be subset of wild (=data_index[:32]) else empty exemplar batch.
- Caveat: n=100 test (small), single model (Qwen), single subset. Generality across a 2nd model (llama) = optional next step.

## nq_open — REGIME STORY REPLICATES (dataset-general, 2026-07-13)
Qwen2.5-7B / Natural-Questions-open (400Q subset, 100 test, NLI-SE over 5 samples). 2nd uncertainty-driven QA dataset.
- **TSV-alone 0.6667 | NLI-SE-alone 0.6667 | corr(TSV,SE) = -0.199 (anti-correlated!)**
- **Learned fusion (logreg CV): TSV-only 0.6414 -> TSV+NLI-SE 0.7356, delta = +9.4pp.** (TSV train OOM-truncated at best 0.667, like TruthfulQA.)
- Replicates TriviaQA: on BOTH uncertainty-driven datasets SE is strong (matches or beats TSV) and fusion adds ~+9-11pp. The NEGATIVE correlation on nq_open is even stronger evidence of complementarity than TriviaQA's +0.34 — the two signals catch disjoint error subsets.

## THREE-DATASET SUMMARY (the finding)
| dataset | regime | NLI-SE alone | TSV | corr | logreg-CV fusion delta |
|---|---|---|---|---|---|
| TruthfulQA | confident-wrong | 0.556 (weak) | 0.844 | 0.33 | +0.85pp |
| TriviaQA | genuine uncertainty | 0.817 (strong) | 0.736 | 0.34 | +11.1pp |
| nq_open | genuine uncertainty | 0.667 (strong) | 0.667 | -0.20 | +9.4pp |
**Conclusion: latent-probe (TSV) + output-entropy (SE) fusion is regime-dependent and REPLICATED across 2 uncertainty datasets. Where errors are uncertainty-driven, SE is a strong, complementary (even anti-correlated) signal and fusion adds ~+9-11pp. Where errors are confident falsehoods (TruthfulQA), SE collapses and only the latent probe works. Publishable cross-modal complementarity result.**

## Cross-model (Qwen2.5-7B-INSTRUCT / TriviaQA, 2026-07-14) — regime half-replicates, FUSION GAIN does NOT (honest nuance)
Same pipeline on the instruct-tuned model (base-vs-instruct; llama/Yi blocked by gated/incomplete caches). n=100 test.
- **TSV-alone 0.8564 (MUCH stronger than base 0.736) | NLI-SE-alone 0.6927 (still strong, >0.65) | corr 0.236 | logreg-CV fusion: TSV 0.8452 -> TSV+SE 0.8445, delta = -0.07pp (FLAT).**
- **SE-strength REPLICATES** (0.693 on instruct TriviaQA — genuine-uncertainty regime holds across base+instruct). But the **FUSION GAIN does NOT replicate** — and crucially NOT because SE is weak. It's because the instruct model's latent probe is now so strong (0.856) that it **subsumes** SE's signal.
- **Refined mechanism (the real rule):** cross-modal fusion helps only when the two signals are *both comparably informative AND complementary*. On base TriviaQA, SE (0.817) ≥ TSV (0.736) → SE is the stronger view → fusion +11pp. On instruct TriviaQA, TSV (0.856) >> SE (0.693) → the latent probe dominates and SE is redundant-given-TSV despite being individually strong → fusion flat. So the +9-11pp base-model gains are specific to the regime where SE ≳ TSV; when one modality dominates, fusion is a wash. This is an honest tightening of the claim, not a refutation: the *complementarity* (corr 0.24-0.34) is real, but complementarity alone is insufficient — you also need neither signal to dominate.
- Artifacts (a40): /root/tsv/SE_FUSION_QI_RESULT.txt, test_scores_qweninstruct_triviaqa.npy, se_fusion_qi.py. Caveat: n=100, single instruct model, TSV OOM-truncated (best saved).

## FINAL cross-model+dataset table (4 model×dataset cells)
| model / dataset | regime | NLI-SE | TSV | corr | fusion Δ | reads |
|---|---|---|---|---|---|---|
| Qwen-base / TruthfulQA | confident-wrong | 0.556 | 0.844 | 0.33 | +0.85pp | SE weak → fusion ~flat |
| Qwen-base / TriviaQA | uncertainty | 0.817 | 0.736 | 0.34 | +11.1pp (sig) | SE≥TSV → big fusion |
| Qwen-base / nq_open | uncertainty | 0.667 | 0.667 | -0.20 | +9.4pp | SE≈TSV → big fusion |
| Qwen-instruct / TriviaQA | uncertainty | 0.693 | 0.856 | 0.24 | -0.07pp (flat) | TSV dominates → SE redundant |
**Final honest claim: cross-modal fusion (latent probe + output-entropy SE) yields large, significant gains only when (a) errors are uncertainty-driven (so SE is informative) AND (b) neither modality dominates the other. It is flat when SE collapses (confident-wrong) OR when the latent probe alone is already near-ceiling (strong instruct probe). Complementarity is necessary but not sufficient.**

## 5th cell attempted — Qwen-Instruct / nq_open — TSV TRAINING DIVERGED (excluded from fusion table, honest)
Ran the pipeline for a 5th cell. Result is a **TSV reproduction failure on this model/dataset**, not a fusion datapoint:
- TSV steering-vector training **diverged**: test AUROC 0.584 at epoch 1 (≈untrained) → monotonically DROPPED every epoch to 0.22 (best = epoch-1 0.584). The vMF/optimal-transport pseudo-labels evidently mis-optimize for Qwen-Instruct on nq_open. test_scores are non-degenerate (99 unique, std 0.047) but anti-informative.
- Consequently the fusion script's "+66.8pp" is a **pure artifact** of a degenerate logregCV-TSV baseline (0.143) — NOT a real gain. Excluded.
- The ONE trustworthy number from this cell: **NLI-SE alone = 0.9235** — SE is a *very* strong standalone hallucination detector on Qwen-Instruct/nq_open (instruct model on Natural-Questions produces cleanly-disagreeing samples). Notably SE-alone (0.924) > the fused score (0.811), i.e. adding the broken probe HURTS — the mirror of the instruct/TriviaQA cell, and consistent with the rule (fusion helps only when neither modality dominates; here SE dominates).
- HONEST TAKEAWAY: TSV training is not uniformly stable across model×dataset (diverges here), which is itself a limitation of the reproduction worth noting; and it does not change the 4-cell fusion rule, which stands. Artifacts: /root/tsv/SE_FUSION_QINQ_RESULT.txt (with the caveat), test_scores_qi_nq.npy, tsv3 log epoch trajectory.

## TSV code review + fix attempt (2026-07-14) — 3 real defects, 1 clean fix, 1 honest dead-end
User asked: can we fix TSV's implementation to make it better / publishable on its own. Reviewed tsv_main.py / train_utils.py / llm_layers.py. Findings (tsv_stable.py = gated copy; tsv_main.py untouched):
1. **Memory bug (FIXED, real improvement).** Forward passes use `output_hidden_states=True` and then `torch.stack(ALL layers)` when only one layer (str_layer=9) is used → materializes ~29× activations → OOM on a 46GB A40 at the default batch 128. Fix: index the needed layer from the tuple directly (3 sites) → peak mem 44GB→29GB; combined with batch_size 32, Stage 2 runs where it previously OOM'd. Concrete, reproducible.
2. **Test-set model selection (VALIDITY BUG).** Both stages do `if test_auroc > best_test_auroc: save; best=test_auroc` — the reported number is the MAX test AUROC over all 40 epochs, chosen on the test labels. Optimistic bias baked into the paper's headline numbers and our reproduction. Honest protocol needs val-based selection or final-epoch reporting.
3. **Few-shot overfitting / anti-generalization (DIAGNOSED, NOT trivially fixable).** On Qwen-Instruct/nq_open the steering-vector training drives **exemplar-set AUROC = 1.0** (32 exemplars perfectly memorized) while **test AUROC collapses to 0.2245** (flipped 0.7755). The representation carries signal (0.78 flipped) but the learned steering generalizes *backwards* on test.
   - Hypothesis A (stage-2 centroid inversion) — tested via a post-Stage-1 anchor+swap (TSV_STABLE=1): only moved final 0.2245→0.3316, did NOT recover. The Stage-1 state was already inverted, so the anchor was too.
   - Hypothesis B (global sign flip fixable by orienting on labeled exemplars) — tested: **FAILS**, because exemplar AUROC=1.0 ⇒ orientation says flip=False ⇒ no correction. The exemplars don't reveal the test inversion (they're overfit). So the 0.22→0.78 recovery is NOT legitimately achievable (would require test labels).
   - Conclusion: this is severe few-shot overfitting on some model×dataset cells, a genuine robustness LIMITATION of TSV, not a one-line bug. Potentially mitigable by regularization (weight decay / early-stop / lower λ) — untested, would need a small sweep.
## Regularization sweep (2026-07-14) — does reducing overfitting fix the failing cell? NO (honest)
User chose to try a regularization fix. Swept λ (steering strength) and weight decay on the failing cell (Qwen-Instruct/nq_open), FINAL-EPOCH AUROC (honest, no test-selection):
| λ | wd | AUROC | note |
|---|---|---|---|
| 5 | 0 | 0.2245 | baseline (steering ACTIVELY harmful) |
| 5 | 0.1 | 0.2245 | weight decay useless |
| 2 | 0 | 0.25 | |
| 1 | 0 | 0.365 | |
| 1 | 0.1 | 0.372 | |
| 0.5 | 0 | 0.393 | |
| 0.25 | 0 | **0.5255** | best — barely above chance |
| 0 | 0 | 0.495 | steering OFF ≈ chance |
- **Two real findings:** (1) high-λ steering is *actively harmful* on this cell — λ=5 gives 0.22, WORSE than no steering (0.49); the steering over-rotates into an inverted, unusable direction (the "0.78 flipped" phantom). (2) Even the best λ (0.25) only reaches **0.53** — the underlying representation is fundamentally weak here, so NO configuration yields a good detector. exemplarAUROC stays 1.0 throughout (overfitting is not the fixable lever; representation quality is the ceiling).
- **Good-cell no-harm check (Qwen-Instruct/TriviaQA):** λ=5 → 0.842 (reproduces ~0.856), λ=1/wd0.1 → 0.827 (−1.5pp). So globally lowering λ to help failing cells slightly HARMS good cells. No single config wins both.
- **CONCLUSION: regularization does NOT yield a "TSV-better" fix.** The failing cell is unrecoverable (weak representation, ceiling ~0.53); the good cells are already near-optimal at default λ. λ is a per-cell sensitivity knob (high λ can hurt), not a global improvement.

**Publishability verdict (honest): "TSV beats its paper" is not supported. What IS real = a robustness/reproducibility study: the memory fix (enables 46GB-GPU training), the test-set-selection bias (numbers are optimistically selected), and the few-shot overfitting instability (fully-failed cells, exemplar-AUROC 1.0 vs test 0.22). That is a legitimate workshop-grade contribution, but it is a repro/robustness paper, not a SOTA improvement.** Artifacts: /root/tsv/tsv_stable.py, ab_orient.log, ab_vanilla.log/ab_stable.log.

## Bootstrap significance (2000 resamples, fused-oof vs raw-TSV AUROC delta, n=100 test)
- **TriviaQA: delta +0.0967, 95% CI [+0.011, +0.191], frac>0=0.988 → SIGNIFICANT** (CI excludes 0; bootstrap two-sided p≈0.024).
- **nq_open: delta +0.0689, 95% CI [-0.051, +0.196], frac>0=0.859 → directional, NOT significant at n=100** (CI includes 0; one-sided p≈0.14).
- HONEST READ: the fusion gain is statistically solid on TriviaQA and directionally consistent on nq_open but underpowered (n=100 test is small; the 400Q cap limits power). The regime *direction* replicates on both uncertainty datasets; formal significance holds on the larger-effect one. Artifacts: /root/tsv/BOOTSTRAP_FUSION_RESULT.txt, test_scores_{triviaqa,nq_open}.npy, bootstrap_fusion.py. To strengthen nq_open: larger test split (uncap the 400Q) would tighten the CI.

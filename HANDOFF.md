# Research Handoff — Structural-Entropy / Hallucination-Detection Program
_Written 2026-07-15 for a 2-day pause. Everything needed to audit the program and resume._

## 0. TL;DR — where we are
Seven research tracks under one measure (structural/semantic entropy). The **live, freshest result** is Track 7: a **learned meta-fusion that robustly beats semantic entropy (SE) by ~7pp on the Qwen model family** (significant, 8/8 random splits), with an **honest, diagnosed limit on Llama**. The rest of the program (graph attention, RL, LLM-confidence, TSV audit, cross-modal fusion) is documented below with each track's publication gap. **The single most publishable unit right now = Track 7 (meta-fusion beats SE) + Track 5 (TSV audit) as one "hallucination detection" paper** — its gap is cross-model breadth (a scale-up was running and is now paused) + head-to-head baselines.

## 1. Infrastructure & data state (post-pause)
- **All boxes DESTROYED at pause** (per user): the RTX-4090 "uprobe" box (vast 44915573) and the dm-sisa 4×3060 "newbox". The a40 and a broken A100 were already destroyed earlier.
- **g4z2 (vast 43745495, 3.92.24.194)** — ⚠️ NOT destroyed. It holds the **user's** `codex-orchestra` / btc-sim project (not ours), and was on hold pending the user's explicit "codex-orchestra saved" all-clear, which never came. **Decision needed from user:** confirm codex-orchestra is saved → then it can be destroyed; otherwise keep.
- **Local backups on the control machine (`/home/ubuntu/`)**:
  - `uprobe_backup/` (1.1GB) — the Track-7 campaign: 26 feature cells `raw2/{model}_{dataset}.npz` (hidden states at 4 layers + greedy/sample/paraphrase answer texts + gold), `feats2/judge_*.npy` (LLM-judge labels) + `feats2/se_*.npz` (SE_temp, SE_para), all pipeline scripts, `para_{dataset}.npz` (shared faithful paraphrases), `BENCH2.txt`.
  - `a40_tsv_backup/` (194MB) — Tracks 5/6: TSV reproduction + audit code, feature caches, all result `.txt`.
  - `dm-sisa/results/newbox_sisa_backup/` — Track-2 §9c eval logs.
- **GitHub (durable):** `SuuTTT/structural-entropy-benchmark` (labnote + this handoff + `hallucination-detection-code/`), `SuuTTT/dm-sisa` (DM_SISA_STUDY.md), `SuuTTT/sese-harness` (SeSE), issue filed `SuuTTT/tdmpc-glass#1`.
- **Models cached were re-downloadable; the pipeline reruns from scratch on any box with working HuggingFace access.** (The a40 had *broken* HF downloads — that is why we migrated; always test a *sustained* multi-GB download, `curl -L`, before trusting a rented box.)

## 2. Per-track status & the gap to publish

### Track 1 — Multi-level SE for graph attention (STRONGEST, likely paper-ready)
- **Result:** multi-level SE prior beats flat SE +2.0pp and modularity +3.7pp at **p<10⁻⁶** (8 datasets × 15 seeds, Benjamini–Hochberg corrected); parameter-control + planted-hierarchy control both hold; nulls (depth-2 washed at 30 seeds) reported.
- **Gap to publish:** essentially none scientifically — **gap is the writeup/submission** (target: a graph-learning venue). Verify the final numbers against artifacts before submission.

### Track 2 — DM-SISA: differentiable SE for RL state abstraction
- **Result:** engineering win (soft-SE matches discrete to 1e-7, ~16,000× faster). One real return beat: **walker-walk +14%, p=0.0073, 10 seeds**. §9c re-test (2026-07-14) showed it is **narrow**: walker-*run* −4% (n.s.), hopper-hop **−62%** (reward *hurts*) → verdict **"walker-walk-specific, not a locomotion-family property."** Committed `dm-sisa` eb965e9.
- **Gap to publish:** the honest story is a *scoped positive with a sharp boundary* — publishable as a systems/workshop contribution, **not** a "beats baseline everywhere" claim. Gap = decide framing (efficiency+scoped-beat) and write it; quadruped vanilla arm never finished (box destroyed) but cannot change the verdict.

### Track 3 — SeSE: reasoning-chain SE as LLM self-doubt
- **Result:** chain-SE beats answer-SE at error detection **only under overconfidence, and only in the gemma family** (corr(overconfidence, chain-advantage)≈0.75). ~10 non-gemma families are answer-SE-wins/calibrated. Within-SE fusion (answer+chain) = **NULL** (redundant, both output-space). `sese-harness` ae072f4/25b16ee.
- **Gap to publish:** gemma-anchored scope is the ceiling. Publishable as an honest calibration-findings paper ("gemma-anchored"). The cross-family generalization hope moved to Tracks 6/7 (cross-*modal* fusion), which is where the action is.

### Track 4 — Benchmark & provenance audit (infrastructure)
- **Result:** reproducible harness + ledger; equation-provenance audit fixed 2 real transcription errors. Underwrites all tracks.
- **Gap:** none — it is the honesty infrastructure, not a standalone paper.

### Track 5 — TSV reproduction + robustness audit (FIRST independent audit)
- **Result:** reproduced TSV (Park et al., ICML 2025) at AUROC 0.844 vs paper 0.873. Found **4 real defects**: (1) memory OOM bug (stacks all layers, 44→29GB fixed); (2) **test-set model selection** in the released code (optimistic bias; the *paper* claims val-selection → a code-vs-paper discrepancy); (3) undocumented **class-inversion** (vMF/OT self-labeling can flip the truthful/hallucinated assignment → AUROC collapses to 0.22, "flipped"=0.78); (4) severe **few-shot overfitting** (32 exemplars fit to AUROC 1.0 while test collapses). Fixes to (3)/(4) failed (orientation-by-exemplars, regularization sweep) — the failures are *fundamental*, not bugs. **Simple SE beats TSV on uncertainty datasets** (0.85 vs 0.74, 0.77 vs 0.67). `~/TSV_SE_INTEGRATION.md`, `a40_tsv_backup/`.
- **Gap to publish:** strong reproducibility/audit content; **no independent TSV audit exists in the literature** = first-mover. Gap = fold into the Track-7 paper (audit motivates "don't use a fragile trained probe; use SE + our fusion").

### Track 6 — Cross-modal fusion (regime-dependence)
- **Result:** fuse TSV latent + SE. **Regime-dependent:** TriviaQA +11.1pp (bootstrap-sig 95%CI[+1.1,+19.1]), nq_open +9.4pp; TruthfulQA +0.85pp (flat). Refined rule: fusion helps only when errors are uncertainty-driven **and** neither modality dominates. Cross-family (llama/Yi) blocked by gated/incomplete caches at the time.
- **Gap to publish:** the "fuse latent+consistency" idea is **crowded** (primary foil **arXiv 2603.19118** already does LR fusion + "internal recovers confident-wrong"; also CoCoA, UQLM, Detection-Dilemma). Our defensible wedge = the *boundary-condition map* + the counterintuitive twist (fusion is *flat* in the confident-wrong regime because the probe dominates — contradicts 2603.19118's strength-invariance claim). Superseded by Track 7's stronger, significance-tested version.

### Track 7 — Universal probe / META-FUSION (THE LIVE HEADLINE RESULT)
- **Method:** a small logistic-regression **meta-fusion** of four features predicting answer-correctness:
  `[z(SE_temp), z(SE_para), z(residual_probe), conf, SE_temp·conf]` where
  - **SE_temp** = standard temperature-sampling semantic entropy (the SOTA baseline).
  - **SE_para** = *paraphrase-SE* — SE over answers to 3 faithfully-rephrased questions (a model-agnostic consistency axis; targets confident-wrong = brittle-to-rephrasing).
  - **residual_probe** = a linear probe trained **only on the confident (low-SE_temp) examples** to separate confident-wrong from confident-right — i.e. specialised to SE's *blind spot*.
  - **conf** = confidence-regime indicator (SE_temp < median).
  - Labels are **clean LLM-judge** (Qwen-Instruct grading vs gold), replacing noisy string-match.
- **Result (8 random splits/model, mean±sd, SE_temp = baseline to beat):**
  | model | SE_temp | META Δ over SE | splits+ |
  |---|---|---|---|
  | Qwen2.5-7B | 0.708 | **+7.7pp ± 1.9** | **8/8** |
  | Qwen2.5-7B-Instruct | 0.656 | **+6.8pp ± 1.7** | **8/8** |
  | Llama-3.1-8B | 0.611 | +3.7pp ± 4.2 (noisy) | 6/8 |
  - **Robust, significant beat over SE on the Qwen family.** Mechanism: each signal is weak/negative alone; the *learned fusion* wins because they catch different hallucinations. Clean labels sharpened qwen-7B from borderline +2.5pp → +6.9pp.
  - **Honest limit:** Llama is directional but not robust (sd>mean, 2/8 splits negative) — its hidden states carry a weaker decodable hallucination signal, so the probe component overfits. The **model-agnostic pieces (the two SE signals) transfer; the latent probe is Qwen-specific.**
- **Scale-up (PAUSED mid-run):** was generating 5 more models — **Qwen-14B base+instruct (size), Mistral-7B base+instruct + Yi-9B (new families)** × 5 datasets, reusing paraphrases. **Qwen-14B base = fully generated (in `uprobe_backup/raw2/`); 14B-instruct partial; Mistral/Yi not started.** Judge+SE for the new models had not run.
- **Gap to publish (this is the main gap for the headline paper):**
  1. **Cross-model breadth** — finish the scale-up (Mistral/Yi/Gemma families + 14B/larger sizes). Does the beat generalize beyond Qwen, or is it Qwen-specific (like Llama hints)?
  2. **★ The key untested thesis:** is **residual-probe validation-AUROC (latent "decodability") predictive of the META gain across models?** If they correlate, the Llama weakness becomes a *characterized law* — "the fusion beats SE iff the model's hidden states are decodable, predictable per-model from a cheap probe-val check." That turns a limitation into the paper's contribution. **This is the highest-value next experiment** (needs the multi-model data + a correlation).
  3. **Head-to-head baselines** — run TSV, INSIDE/EigenScore, SelfCheckGPT on the same cells (currently only SE_temp is the baseline).
  4. **Larger n** — n≈400/model is noisy (that is why Llama swung ±). Bump N per dataset.
  5. **More datasets** (PopQA, HotpotQA, GSM8K/math for more confident-wrong).
  6. **Rigor** — DeLong or paired-bootstrap already used; add per-dataset reporting, anti-artifact baselines (length, majority), label-sensitivity (judge vs string-match), multiple-comparison note (several fusion designs tried).

## 3. How to resume (fastest path to the paper)
1. **Rent a box with VERIFIED sustained downloads** (test `curl -L` on a full model shard first; A100-80GB or 2×A40 if adding >13B models; unrestricted HF is the hard requirement).
2. `scp -r /home/ubuntu/uprobe_backup /root/uprobe` and reinstall deps (`transformers==4.44.2 datasets accelerate scikit-learn sentencepiece hf_transfer`). All code + the 26 existing feature cells are there — **no need to regenerate Qwen/Llama/Qwen-14B-base.**
3. Finish the scale-up matrix (`gen2.py` HF map already has all 9 models): run remaining `mistral7bi, yi9b, mistral7b, qwen14bi` × 5 datasets → `judge.py` → `se2.py`.
4. Run the **8-split multi-model bench** (script in labnote Track 7 / uprobe_backup) → per-model META-Δ + **residual_probe-val vs META-Δ correlation** (the thesis test).
5. Add TSV/INSIDE baselines; write the paper.
- **Pipeline gotchas learned:** instruct/verbose cells are slow for NLI-SE (~6min; 14B-instruct ~50min/cell — run families before slow big-instruct models); paraphrases must come from an *instruct* paraphraser (base models drift the meaning); judge+SE are idempotent (skip-done) so safe to relaunch, but don't spawn duplicates; SSH to cheap boxes is flaky (short commands, verify via re-check).

## 4. Key artifacts index
- **Labnote (public):** https://suuttt.github.io/structural-entropy-benchmark/2026/07/14/hallucination-detection-a40-labnote.html (Tracks 1–7).
- **Code archived to git:** `SuuTTT/structural-entropy-benchmark/hallucination-detection-code/` (TSV audit, benchmarks, prep pipeline) + this `HANDOFF.md`.
- **Plans:** `/home/ubuntu/TSV_SE_HOWTO.md` (rigorous eval protocol + must-cite related work), `/home/ubuntu/UNIVERSAL_PROBE_PLAN.md`, `/home/ubuntu/TSV_SE_INTEGRATION.md` (full TSV+fusion log).
- **Local data:** `uprobe_backup/` (Track 7), `a40_tsv_backup/` (Tracks 5/6), `dm-sisa/results/newbox_sisa_backup/` (Track 2).

## 5. Open decisions for the user (to audit over the 2 days)
1. **Headline paper unit:** confirm it's Track 7 (meta-fusion > SE) + Track 5 (TSV audit) combined, framed as "when does fusing beat semantic entropy — and can we predict it from decodability."
2. **Scale-up scope:** how many model families / sizes for the cross-model claim; whether to add a bigger box for 70B.
3. **g4z2:** confirm codex-orchestra is saved → authorize its destroy, or keep it.
4. **Standalone papers:** Track 1 (graph, strongest) and Track 2 (DM-SISA scoped-beat) can be written independently — decide priority.

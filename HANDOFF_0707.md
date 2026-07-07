# HANDOFF — 2026-07-07 — Structural-Entropy Research Program

**Read this first.** Written for a fresh session (model switch + restart). It has the full state of three
research lines, the box fleet, every repo, what is running right now, how to resume, and the honest
verdicts + open questions. Everything is version-controlled and self-healing; nothing depends on this
session staying alive.

---

## 0. TL;DR — three honest results

| line | verdict | publishability | repo |
|---|---|---|---|
| **Graph — "The Multi-Level Edge"** | **Clean significant win** — multi-level SE beats flat SE (+2pp) & modularity (+3.7pp), **p<10⁻⁶**, BH-corrected, capacity + planted-ground-truth controls; ties MinCut (tuning-free robustness) | **STRONGEST — lead with this** | se-attention-prior-paper / SE-graph-learning |
| **DM-SISA (RL)** | **Parity + efficiency, no return beat** — matches SISA *and* RAD-SAC at 16,000× less cost; over-reg claim retracted; injection (L5) tested to n=5 = no beat | efficiency/systems paper (workshop-tier) | dm-sisa / dmsisa-paper |
| **SeSE (LLM calibration)** | **Real finding, gemma-anchored** — chain-SE beats answer-SE iff overconfident (corr(oc,gap)≈0.75); 8 non-gemma models all calibrated/artifact → cross-family eludes | real finding + honest scope | sese-harness / sese-calibration-paper |

Guiding discipline all session: **numbers only from artifacts; retract on contradiction; multiple-comparison-aware.**
Five DM-SISA "wins" and the one SeSE cross-family lead (Mistral) all washed out / were artifacts — each caught before it reached a paper.

---

## 1. Repos (single source of truth) — all on github.com/SuuTTT

| repo | contents |
|---|---|
| **dm-sisa** | DM-SISA code (`src/`: dm_sisa.py, curl_sac.py w/ `--dmsisa_inject`, train.py), **harness/** (forever_v5.sh self-healing, launch_campaign.sh tmux, harvest_v2.py, pull_seeds.py), `results/` (g4x/g4z/a4 eval.logs), **DM_SISA_STUDY.md**, **HANDOFF.md** (env recipe) |
| **sese-harness** | SeSE harness (`cot_se_run.py` w/ Moonshot/Kimi + DeepSeek API routing), calib_analyze.py, run scripts, all `results/` summary.json cells, **SeSE_CALIBRATION_STUDY.md** |
| **SE-graph-learning** | graph code, `experiments/d6_diffse/results/*.jsonl` (stage9-18), **MULTI_LEVEL_EDGE_STUDY.md** |
| **dmsisa-paper** | DM-SISA paper (LaTeX). Latest: corrected n=5, over-reg retracted, injection no-beat. |
| **sese-calibration-paper** | SeSE paper. CORR 0.75 (n=17), Mistral debunk recorded, gemma-anchored scope. |
| **se-attention-prior-paper** | Graph paper "The Multi-Level Edge" (8pp, figs committed). ⚠️ TODO: add stage17b hard-planted NULL caveat before submission. |
| **structural-entropy-benchmark** | Benchmark repros + **results ledger** (GitHub Pages: suuttt.github.io/structural-entropy-benchmark/ledger/) — registry of every experiment/lever tried (avoid redo) |
| **structural-entropy-survey-paper** | The TGINA survey (empirical reappraisal). |

Local clones under `/home/ubuntu/`. Public blog: suuttt.github.io (Hugo, /tmp/sio_fresh working clone; commit user suuttt@icloud.com/sudingli, push origin master).

---

## 2. Box fleet (ssh + what runs where)

ssh key: `~/.ssh/vastai_id_ed25519`. Get IPs via `vastai ssh-url <ID>`. Aliases in `~/.ssh/config`.

| alias | hardware | ID / addr | role | now running |
|---|---|---|---|---|
| **a40box** | 2×A40 | (SeSE box) | SeSE / LLM-UQ | mid-difficulty cross-family search (both GPUs) |
| **g4z** | 4×RTX3060 | 43745495 / 1.193.138.249:38867 | DM-SISA | broad-task 4-way (forever_v5 tmux `camp`) |
| **sisab** | A4000 | 39109169 / 167.179.138.57:41030 | DM-SISA ablations | λ-sweep + depth (forever_v5 tmux `camp`) |
| **a4** | A4000 | ssh4.vast.ai:29168 | markov A/B | low priority |

**a40box specifics:** harness `/root/sese/sentence_structural_entropy/`, venv `/root/sese_venv` (py3.12), NLI = deberta-v2-xlarge-mnli (cached /workspace/.hf_home). **API keys (paths, NOT values):** `/root/.moonshot_key` (Moonshot/Kimi, base_url api.moonshot.cn/v1), `/root/.deepseek_key` (DeepSeek, api.deepseek.com). API routing in cot_se_run.py (`kimi`/`moonshot`/`deepseek` model prefixes). Cached local LLMs: gemma-2-9b, Yi-1.5-9B, zephyr-7b, Mistral-7B.

**DM-SISA boxes (g4z, sisab):** env `/venv/sisa` (py3.10.20, gym0.21.0/mujoco3.8.1/torch2.5.1+cu121, `MUJOCO_GL=egl`), code `/root/SIDM/SISA/source code/dmcontrol/`. Launch: `MAXJOBS=N [DISK_MAX=94] bash /root/launch_campaign.sh` (tmux). Note: sisab disk shared w/ other projects (sold/sold_venv — DON'T touch); helios-rl + mujoco_playground deleted (user-authorized).

**vastai policy:** NEVER destroy instances; recommend a destroy list, user destroys manually.

---

## 3. The self-healing harness (USE THIS on every box)

`forever_v5.sh` (in dm-sisa/harness, deployed on g4z+sisab) fixes the flaky-box failures that plagued this
session:
- **Self-healing:** a job is "done" only when its `eval.log` reaches MINSTEP=85000 (NOT on launch). Crashed/preempted jobs auto-retry. (v4's done-on-launch was the root bug that made boxes go "empty".)
- **Disk guard** (DISK_MAX%, default 85): cleans regenerables + pauses.
- **Interrupt log** `/root/interrupts.log`: WHY each job died (exit code, OOM/CUDA, SIGKILL=oom, SIGTERM=preempt). Monitor it → notify on interrupt. Empty log = healthy host.
- **tmux persistence** (`launch_campaign.sh`): survives ssh drops. NEVER use `setsid`-over-ssh — it kept failing (ssh timeout kills the launch before detach).
Resume any campaign: `MAXJOBS=6 bash /root/launch_campaign.sh` (idempotent — skips completed jobs). Deep queue in `/root/aaai_jobs.txt` (lines: `domain task mode seed steps`; modes in forever_v5 flags_for: baseline=SISA, vanilla, dmsisa=λ0.01, dmsisa_sc0=λ0, dmsisa_sc003/sc03, dmsisa_flat, dmsisa_inject).

SeSE launches: file-based setsid scripts (mirror `/root/kimi_run.sh` / `/root/middiff.sh`) — NOT ssh heredocs (escaping breaks them). One API model at a time (rate limits). GPU-idle during API runs is NORMAL (sampling is remote; GPU only for NLI bursts).

---

## 4. WHAT IS RUNNING RIGHT NOW (in-flight, ~hours to complete)

- **a40 SeSE mid-difficulty search** (`/root/middiff.sh` GPU0 = gemma-2/zephyr/Yi on gsm8k; `/root/middiff_g1.sh` GPU1 = same on svamp; 200q×8chains×2seeds). Goal: find an overconfident non-gemma model at ~40-70% acc (the regime where AUROC is meaningful — AIME too easy=100%, HMMT too hard=0-23% both broke it). **So far: zephyr-gsm8k calibrated (negative). gemma-2 = the key pending cell** (does overconfidence extend beyond gemma-3 → gemma-family-wide?).
- **g4z broad-task 4-way** (cartpole/reacher/ball_in_cup × baseline/vanilla/dmsisa/dmsisa_sc0 ×5) → completes the 6-task same-box DM-SISA table. First cell in (cartpole baseline 870.7). Expect parity/neutral on saturated tasks.
- **sisab λ-ablations** (sc003=λ0.003, sc03=λ0.03, flat × walker/cheetah ×5) → the λ U-curve + depth. Have λ0=676, λ0.003=735.8(n1), λ0.01=774; **return trends UP with λ (consistent with the over-reg retraction — the penalty does not hurt).** Need sc03 + n≥3 for the figure.

**Next-session actions when these land:** (i) harvest cells by artifact (calib_analyze / pull_seeds); (ii) if any non-gemma cell is acc~40-70% + high-oc + chain-wins(gap>0.05) + valid(empty<0.25) → CROSS-FAMILY CONFIRMED (rare, report+update paper); (iii) build λ U-curve fig (/tmp/hv matplotlib → figs/lambda_ucurve.pdf) + 6-task table + gemma-family note → commit into docs/papers; (iv) after ~4-6 non-gemma cells all calibrated → update SeSE paper scope: "broad mid-difficulty sweep confirms gemma-anchored."

---

## 5. Per-thread state + open questions

**Graph (strongest):** paper 8pp, figs committed, p<10⁻⁶ result solid. **OPEN:** add the `stage17b` hard-planted NULL caveat (it does NOT reproduce the depth result; the paper's claim rests on stage17_planted which held). Then submit-ready.

**DM-SISA:** verdict FINAL — parity + 16,000× efficiency, no beat via any lever (λ-sweep, depth, injection all n.s.). Paper corrected. **OPEN (optional):** the λ U-curve + 6-task table (running now) round out the ablations; nothing changes the verdict. Regimes where abstraction *should* help (distractor / long-horizon) untested — future work, not a gap.

**SeSE:** the calibration finding (chain-SE wins under overconfidence) is real and gemma-anchored. Mistral cross-family lead = parse artifact (debunked: fixed extractor → acc 0.06→0.67, oc +0.69→-0.04, gap +0.196→0.00). HMMT frontier test = honest negative + the **difficulty-calibration lesson** (need model at ~40-70% acc). **OPEN:** (a) is there ANY overconfident non-gemma model at mid-difficulty? (running); (b) does the effect hold for a SOTA model in its *genuine* error regime on a mid-difficulty-for-it benchmark (e.g. MATH L3/4 for DeepSeek)? — the real "alive for frontier?" test, still unanswered because AIME/HMMT were the wrong difficulty.

---

## 6. Hard-won lessons (don't relearn these)

1. **Pixel-RL return is dominated by seed variance** — a 2-seed ablation is not a mechanism; require n≥5 + multiple-comparison correction. (5 DM-SISA signals washed out.)
2. **Answer-parsing artifacts masquerade as overconfidence** — a model with 73% unparseable answers looks confidently-wrong but isn't. Always check empty-fraction; the Mistral "lead" was entirely this.
3. **UQ test-beds need mid-difficulty** — 100% acc → undefined AUROC; ~0% acc → degenerate AUROC. Target ~40-70%.
4. **Flaky Vast hosts**: prefer on-demand (not spot); forever_v5 + tmux + interrupt-log; SIGTERM/SIGKILL rows = host preempting → switch hosts. Full disk (even from other projects) wedges the box.
5. **ssh-background launches fail** (timeout kills them); use tmux sessions; verify by artifact, not stdout.

---

## 7. Papers — what's left before submission

- **Graph** (se-attention-prior-paper): add stage17b caveat → submit. **Highest priority / best bet.**
- **DM-SISA** (dmsisa-paper): fold in λ U-curve + 6-task table when they land; already honest. Reframe as efficiency/systems for the right venue.
- **SeSE** (sese-calibration-paper): fold in the mid-difficulty sweep (gemma-anchored confirmation) + any frontier-mid-difficulty result. Scope is the key discussion.

*Anything a future session should verify: recall a `results/` artifact before trusting a remembered number; the study docs and this handoff cite where each number lives.*

# Repro card — SI2E (RL exploration via value-conditional structural entropy)

**Paper:** Zeng et al., SI2E (value-conditional SE intrinsic motivation), 2024.
**Upstream repo:** https://github.com/SELGroup/SI2E
**User's working reproduction (USE THIS):** https://github.com/SuuTTT/learn-si2e
(has REPRODUCE_LOG.md, RESULTS_SUMMARY.md, batch scripts, and all fixes)

## Why this is the Phase-3 entry point
Far lighter than SIRD/StarCraftII: SI2E runs on **MiniGrid** (A2C/PPO) and
optionally DrQv2/DMControl. MiniGrid SI2E ≈ **97 min/run** (3M frames, 1 GPU).

## Env (from the user's REPRODUCE_LOG)
Python 3.10, torch 2.7.1+cu118, numpy≥2.0, gym 0.26.2, minigrid 3.1.0,
SI2E's torch-ac fork (`pip install -e SI2E/SI2E_A2C/torch-ac`), hydra, dm_control 1.0.41.

## Methods compared (A2C/PPO, 16 workers, 3M frames)
Baseline (no intrinsic) · SE (kNN state entropy) · VCSE (value-conditional) ·
**SI2E** (PartitionTree H₂ = structural info + value conditioning).

## Representative run (MiniGrid DoorKey-8x8)
```bash
cd SI2E/SI2E_A2C/rl-starter-files/rl-starter-files
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 scripts/train.py --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model si2e \
  --use_entropy_reward --use_value_condition --use_batch \
  --frames 3000000 --save-interval 100 --log-dir ./logs/a2c_si2e
```

## Critical fixes (must apply, from user's log)
1. `train.py:101` device `cuda:1`→`cuda:0`; DrQv2 `cfgs/config.yaml:24` same.
2. MiniGrid numpy2: `np_random.randint`→`.integers` (minigrid.py ~822); gym `np.bool8`→`np.bool_`.
3. A2C `utils/env.py:6`: add `disable_env_checker=True`.
4. **kthvalue guard** (crashes ~100k frames): use `dists.shape[1]` not `[0]` in torch-ac base.py entropy fns.
5. DrQv2 only: `MUJOCO_GL=egl`; `dmc.py:201` wrap actions in `np.float32`.

## Headline results to reproduce (user's, A2C, MiniGrid)
| Task | VCSE | SI2E |
|---|---|---|
| DoorKey-8x8 (5 seeds) | 97.8%±3.1 | **100%±0.0** |
| KeyCorridorS3R2 (5) | 54.0%±50.3 | **67.5%±31.2** |
| RedBlueDoors-6x6 (3) | 55.4%±47.9 | 55.7%±47.3 |
Main finding: SI2E ↓variance, ↑mean on hard tasks. SI2E ~10–12× slower (O(n²) tree/update).

## Plan
Reproduce DoorKey-8x8 (cleanest: SI2E 100% vs VCSE 97.8%) with ≥3 seeds on a
borrowed A4000 (GPU-bound, ~1.6h/run). Compare SI2E vs VCSE vs SE vs Baseline.
This is the SE-helps-here positive case to balance the SE-doesn't-help CD results.

## Scope (per user, 2026-06-03)
Survey reports **only the original published SI2E** — NOT the baseline/SE/VCSE/
fast-si2e variants in learn-si2e (those are the user's unpublished work).

## Status: BLOCKED on exact frozen env (2026-06-03)
Cloned SELGroup/SI2E (full clone, not --depth 1 — shallow clone loses the nested
SI2E_A2C content; local copy also at /home/ubuntu/SI2E). Hit a **cascade of
gym-API version walls** — the original code targets the OLD gym API; modern
gym/gym-minigrid break sequentially:
1. `import gym` missing → installed gym==0.26.2 (+ np.bool8→np.bool_ patch). ✓
2. `skimage` missing → scikit-image. ✓
3. `import gym_minigrid` missing → installed gym-minigrid==1.2.2. ✓
4. `env.seed()` gone in gym 0.26 → patched to `reset(seed=)`. ✓
5. obs space is Dict(direction,image,mission) → enabled `ImgObsWrapper`. ✓
6. **WALL:** preprocessor gets `(16,2)` inhomogeneous array — gym-minigrid 1.2.2
   returns new-API `(obs,info)` / 5-tuple step that the old rl-starter-files
   preprocessor/penv can't unpack.
Root cause: SI2E's rl-starter-files + torch-ac expect a **pre-0.26 gym +
gym-minigrid 1.0.x** API. Reconstructing the exact pins by trial-and-error is
wasteful of GPU/time.
**Fastest unblock:** use the user's WORKING env (learn-si2e on /workspace ran it
to SI2E 100% on DoorKey). Need their `pip freeze` (esp. gym / gym-minigrid
versions) OR a copy of their patched rl-starter-files/torch-ac. Then DoorKey-8x8
reproduction is ~97 min/run; parallelize seeds across the 4 GPUs.

## ✅ UNBLOCKED — modernized to latest CUDA (gym 0.26.2), min changes (2026-06-03)
User lost the legacy env → per directive, modernized the original code with 5
minimal, documented patches (training confirmed on box2 RTX 3060, gym 0.26.2):
1. `train.py:101` device `cuda:1`→`cuda:0`.
2. `utils/env.py`: guard `env.seed` (try/except → `reset(seed=)`); enable `ImgObsWrapper`.
3. `torch-ac/.../base.py`: kthvalue clamp `dists.shape[0]`→`[1]`.
4. `torch-ac/utils/penv.py`: **gym→gymnasium** (4 spots) — `step` 4-tuple→5-tuple
   (`done = terminated or truncated`); `reset()`→`obs,_ = reset()`; ParallelEnv
   reset/step likewise.
5. `torch-ac/algos/ppo.py:105`: grad_norm skips params with `p.grad is None`.
Deps (modern): torch 2.3.1+cu118, gym==0.26.2 (+np.bool8 patch), gym-minigrid==1.2.2
(+randint→integers), scikit-image, transformers, the SI2E torch-ac (editable).

## Validation (live)
DoorKey-8x8 SI2E (PPO, entropy+value-condition+batch H₂): training at ~700 FPS,
reward rR climbing from 0 (U23: 0.04). Target: ~100% @ 3M frames (~97 min).
Paper headline to reproduce: SI2E 100%±0 vs VCSE 97.8%±3.1 on DoorKey-8x8.

## Results (2026-06-03, box2 RTX 3060, gym 0.26.2)
Artifact: `results/_logs/si2e_si2e.log` (DoorKey-8x8, PPO, 3M frames, 16 workers).

### Partially reproduces — HIGH seed variance (DoorKey-8x8, 3M frames)
Multi-seed success rate (rR mean at 3M frames):
| seed | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| success | 0.00 | **0.94** | 0.25 | 0.07 | (running) |
- SE-based intrinsic motivation **can** solve DoorKey-8x8 (seed 2 = 0.94, peak 0.99)
  → the mechanism works. But across seeds it is **highly unstable** (0.00–0.94,
  mean so far ~0.3), **contradicting the paper's reported 100%±0 (zero variance).**
- The kthvalue patch is NOT the cause (matches your REPRODUCE_LOG fix). Likely a
  **modernized-env difference** (new minigrid/gym 0.26 obs/reward vs the lost legacy
  gym_minigrid the original used) and/or genuine PPO+intrinsic instability.
- **Honest claim for the survey:** SE intrinsic motivation helps exploration (best
  seeds solve hard sparse-reward tasks), but our modernized reproduction does NOT
  show the paper's claimed stability — a reproducibility caveat worth reporting.
**Overnight:** finishing seed 5 + KeyCorridorS3R2 seeds 1–3 for the full sweep.

### Diagnosis plan (do NOT report SI2E as working until resolved)
The agent *can* occasionally solve it (max 0.97), so the env is fine, but the
exploration/intrinsic-reward signal isn't driving stable success. Prime suspects:
1. **The `kthvalue` patch** (`dists.shape[0]`→`[1]`) sits in the SE/kNN intrinsic-
   reward estimator — if it changed the entropy estimate, the exploration bonus is
   wrong. **Verify this patch preserves the kNN-entropy semantics** (compare to a
   tiny reference); it was made to stop a crash, correctness unverified.
2. Single seed + PPO instability — paper averages multiple seeds; rerun ≥3 seeds.
3. Confirm the intrinsic reward is actually added (`--use_entropy_reward` path).
Next: re-examine base.py entropy fns; if the patch is wrong, fix it faithfully;
then rerun DoorKey-8x8 (≥3 seeds). Honesty: report only after it converges.

# SE Benchmark-and-Survey — Campaign Roadmap

**Goal.** Turn the IJCAI'25 structural-entropy (SE) survey — which reviewers
correctly flagged as a broad, shallow, name-listing "LLM survey" — into a
NeurIPS Datasets-&-Benchmarks-style paper: **a reproducible SE benchmark suite
across all three SE task families, plus a survey written around that evidence.**

## Decisions (locked 2026-06-02, with the user)

1. **Scope:** all three SE task families, in depth —
   (1) community detection / SE-minimization, (2) graph learning
   (pooling / structure learning / augmentation), (3) reinforcement learning.
2. **Reproduction:** run each paper's **original published repo as-is**
   (Java/C++/Python/JAX); wrap only I/O into a shared harness. We report what
   the authors' code actually does, not a reimplementation.
3. **Framing:** reproducible **benchmark suite + survey** (D&B style). Central
   empirical question: *does SE actually work, and when/why — vs. modularity,
   the map equation, spectral methods, and learned baselines?*

## Verification discipline (non-negotiable)

This project has a documented history of fabricated numbers. Therefore:

- **No number appears in the paper unless it traces to a JSON artifact** under
  `suite/results/<family>/<method>__<dataset>__<date>.json` that a **fleet GPU
  actually produced.** The harness writes raw per-seed values, not just means.
- Repro cards record the **exact upstream commit, env, command, and dataset
  hash**. "It should give X" is not a result; only the JSON is.
- Claims we *cannot* reproduce are reported as non-reproductions — not dropped,
  not faked. Negative/null results are first-class benchmark findings.

## Compute: the fleet

- `cd /home/ubuntu/gpu-fleet && ./gpufleet free` to find schedulable GPUs.
  Fleet rotates; `mahjong-league` boxes run CPU self-play (GPU borrowable).
- Launch pattern: rsync code+data to `root@sshN.vast.ai:/root/se-bench/...`,
  then `nohup python3 -u run.py >log 2>&1 </dev/null &`. Verify with
  `nvidia-smi`, not `pgrep`.
- Register on the dashboard as project `se-bench` so runs are tracked.
- Original repos are cloned to a **staging area** (`~/se-bench-repos/`, not in
  git) and pinned by commit in `registry/methods.yaml`; rsynced to GPUs.

## Phases

| Phase | Family | Tractability | Primary repos | Status |
|---|---|---|---|---|
| 1 | Community detection | High (mostly CPU + 1 GPU method) | SEP/HCSE, deDoc(2), LSENet, CoDeSEG, DeSE, SuperTAD, SEAT | not started |
| 2 | Graph learning | Medium (GPU training) | SEP, SE-GSL, SEGA, HiTIN, USER, Hi-PART | not started |
| 3 | Reinforcement learning | Low (GPU-heavy, hard repro) | SIDM, SI2E, COLLAB | not started |
| 4 | Manuscript | — | writes around Phases 1–3 | blocked on 1–3 |

### Phase 1 — Community detection (start here)
Fixes the weakest part of the existing benchmark (which ran a *reimplementation*
on tiny graphs with known artifacts: deDoc "O(N^3)", HCSE crippled to binary).
- **Datasets:** LFR benchmark sweeps (vary mixing μ — the canonical
  community-detection difficulty axis), larger SBM, real graphs (Cora/Citeseer/
  Amazon/email-Eu-core/football), and overlapping benchmarks for CoDeSEG.
- **Methods:** original SEP/HCSE, deDoc, deDoc2, LSENet, CoDeSEG, DeSE +
  baselines Louvain/Leiden/Infomap/spectral/DMoN.
- **Metrics:** ARI, NMI (disjoint); overlapping-NMI/F1 (CoDeSEG); **runtime &
  scalability**; **failure modes** (R2). Cross-objective table (SE vs modularity
  vs map-equation value of each partition).
- **Key questions:** Does SE beat modularity/map-equation near the LFR detection
  threshold? Where does it fail? How does it scale vs the claimed complexity?

### Phase 2 — Graph learning
- **Tasks:** graph classification (TU: PROTEINS, NCI1, DD, IMDB), node
  classification (Planetoid, possibly ogbn-arxiv).
- **Question:** does the SE-derived hierarchy *actually* improve downstream
  accuracy over strong non-SE pooling/GSL baselines, and on which graph types?

### Phase 3 — RL
- RL reproduction is notoriously fragile. Scope to the **single cleanest
  reproducible claim per repo** (e.g., SI2E exploration sample-efficiency;
  COLLAB emergent-coalition metric; SIDM skill discovery), with multiple seeds
  and the paper's own baselines. Report non-reproductions honestly.

## Repository layout

```
benchmark/suite/
  ROADMAP.md            ← this file
  README.md             ← how to run the suite
  registry/methods.yaml ← verified upstream repos + pinned commits (source of truth)
  cards/<family>/<method>.md   ← per-method repro card (commit, env, cmd, datasets, results-ptr)
  harness/              ← thin I/O wrappers, dataset loaders, metric code, fleet launchers
  results/<family>/     ← raw per-seed JSON artifacts (the ONLY source of reported numbers)
```

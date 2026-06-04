# SE Benchmark Suite

A reproducible benchmark of structural-entropy (SE) methods across the three SE
task families — community detection, graph learning, and reinforcement learning
— run from the **original authors' code** and compared against non-SE baselines.

See [`ROADMAP.md`](ROADMAP.md) for the campaign plan and decisions, and
[`registry/methods.yaml`](registry/methods.yaml) for the verified upstream repos.

## Golden rule

**Every number in the paper traces to a JSON artifact in `results/` that a GPU
produced.** The harness writes raw per-seed values. If there is no JSON, there
is no number. (This project has a history of fabricated numbers; the suite
exists partly to make that impossible.)

## Workflow per method

1. Add/confirm the method in `registry/methods.yaml` (verified repo + commit).
2. Clone upstream to `~/se-bench-repos/<method>` and pin the commit.
3. Write a thin harness wrapper (`harness/`) that feeds shared datasets in and
   writes a standard results JSON out — without modifying the method's core.
4. Launch on a fleet GPU (`harness/launch.py` / `gpufleet run`).
5. Fill the repro card (`cards/<family>/<method>.md`) from the JSON.

## Status

Bootstrapping. Registry + roadmap + cards template in place; Phase 1
(community detection) is the first reproduction target.

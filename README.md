# Structural Entropy Benchmark

A **reproducible benchmark** of structural-entropy (SE) methods, run from the
**original authors' code** across five task families and compared against non-SE
baselines under a shared harness. Companion to our SE survey.

> **Golden rule:** every reported number traces to a results JSON in `results/`
> that a GPU actually produced. No JSON, no number.

📝 Blog: *"Does Structural Entropy Actually Work?"* — https://suuttt.github.io
📄 Survey paper: companion `structural-entropy-survey` repo.

## Headline finding: SE's value is **regime-dependent**

| Task family | Method(s) | Verdict |
|---|---|---|
| Community detection (LFR/SBM) | deDoc, DeSE, CoDeSEG, LSENet | ⚠️ loses to Louvain/spectral on LFR; method-fragile on attributed |
| Graph pooling | **SEP** | ✅ reproduces 7/7 TU datasets |
| Node classification | **SE-GSL** | ✅ reproduces (Cora GCN 0.869 / GAT 0.880) |
| **Hierarchy quality (Dasgupta)** | SE-agglomerative | ✅ **best on 4/5 graphs** (the dimension flat-ARI hides) |
| RL exploration | **SI2E** | ⚠️ difficulty-graded: solves easy, bimodal medium, fails hard |
| Bioinformatics (Hi-C TAD) | **SuperTAD** | ✅ reproduces on real Hi-C |
| LLM uncertainty | SeSE | ⏭️ skipped (needs ≥24 GB GPU) |

→ **SE reliably helps in supervised graph learning and produces the best hierarchies,
but is fragile in unsupervised community detection and hard RL.** Full numbers in
[`PROGRESS.md`](PROGRESS.md), [`FINDINGS_phase1.md`](FINDINGS_phase1.md),
[`FINDINGS_new_tracks.md`](FINDINGS_new_tracks.md).

## Two corrected literature claims
- deDoc is **not** "O(N³), infeasible for N>50" — the real jar does 10k nodes in 74 s.
- DeSE's "no public code available" is **false** — the repo exists and reproduces.

## Repository layout
```
ROADMAP.md            campaign plan + decisions + verification discipline
PROGRESS.md           live scorecard (every method, reproduced?, evidence)
FINDINGS_*.md         synthesized findings per phase / new tracks
registry/methods.yaml verified upstream repos + pinned commits (source of truth)
cards/<family>/*.md   per-method repro cards (commit, env, patches, result ptr)
harness/              shared dataset loaders, metrics, runners, parsers
results/<family>/     raw per-seed result JSONs — the ONLY source of reported numbers
materials/            awesome-structural-entropy reading list, papers.csv, review response
```

## Reproduce
Each method runs its own original repo; the harness wraps I/O. See the per-method
repro card in `cards/` for the exact upstream commit, environment, and minimal
modernization patches. Common harness deps: `harness/requirements.txt`.

```bash
# community-detection baselines on the shared graphs
python3 harness/run_community_baselines.py --out results/community_detection
# hierarchy-quality (Dasgupta) benchmark
python3 harness/run_dasgupta.py --out results/hierarchy
# aggregate JSONs -> tables
python3 harness/aggregate.py --results results/community_detection --out results/community_detection/_summary
```

Most SE repos are **framework bit-rotted** (old PyTorch/PyG/gym/dgl); we modernize to
current CUDA with minimal, documented patches (see each card) — itself a finding.

## Honesty / verification
We caught and corrected seven of our own artifacts mid-campaign (signal-inverting
SBM, cosmetic NaN print, a false "Infomap bug", a mis-tuned config, etc.) before they
became reported results. Negative and null results are first-class.

## License
MIT (benchmark harness + results). Upstream method code is each project's own license.

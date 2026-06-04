# Repro card — <METHOD> (<family>)

**Paper:** <author year, title, venue>
**Upstream repo:** <url>  · **pinned commit:** `<sha>` (cloned <date>)
**Language / env:** <python3.x + requirements / java / c++ build>

## What the paper claims (the specific claim we test)
<one or two falsifiable sentences — e.g. "deDoc runs in O(n log^2 n)"; "SEP
pooling beats DiffPool/TopK on PROTEINS graph classification by X%">

## Build / install (on a fleet GPU)
```bash
# exact commands that produced a working environment
```

## Datasets
| dataset | source | sha256 / N,E,K | notes |
|---|---|---|---|

## Run command
```bash
# exact command, seeds, hyperparameters (record any deviation from defaults)
```

## Results
- Artifact: `suite/results/<family>/<method>__<dataset>__<date>.json`
- GPU/instance: `<id>`  · wall-clock: `<t>`
- Headline (read from JSON, never typed by hand):

| dataset | metric | paper-reported | ours (mean±CI, seeds) |
|---|---|---|---|

## Reproduction verdict
- [ ] Reproduced within noise
- [ ] Reproduced qualitatively (ranking holds, magnitude differs)
- [ ] Partial (some datasets/settings)
- [ ] Not reproduced — reason: <...>

## Failure modes / caveats observed
<honest notes: crashes, scaling walls, metric mismatches, undocumented deps>

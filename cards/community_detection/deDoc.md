# Repro card — deDoc (community detection / SE minimization, founding method)

**Paper:** Li et al., *Deciphering the genomic decoding code via structural
information minimisation* (deDoc), 2018.
**Upstream repo:** https://github.com/yinxc/structural-information-minimisation · **pinned commit:** `79f7744` (cloned 2026-06-02)
**Language / env:** compiled Java — `deDoc.jar`, `onednSE.jar` (source inside the
jars / `deDoc.rar`). Runs anywhere with a JRE — **CPU only, no GPU needed.**

## ⚠️ Claim under test (R2 + factual dispute)
The conference benchmark marks deDoc as **"O(N³) per run, infeasible for N>50"**
and reports it only on Karate. But that was the *glass-jax reimplementation*, not
this jar; the deDoc paper itself claims near-linear `O(n log² n)`. **Resolve
empirically:** run the real jar across increasing N and fit the runtime curve.
This single experiment fixes both the complexity-depth (R2) and the
reproducibility concerns for the founding method.

## Input format
```
<num_nodes>
<u> <v> <weight>     # node ids 1-based, symmetric adjacency (Hi-C matrix)
...
```
Output: `graph.deDoc(E)` / `graph.deDoc(M)` — communities (one TAD per line) +
the 2D normalized SE printed. `onednSE.jar` gives 1D-nSE.

## Run plan (fleet CPU box)
```bash
java -jar deDoc.jar  <graphfile>          # deDoc(E) and deDoc(M) partitions + 2D-nSE
java -jar onednSE.jar <graphfile>         # 1D-nSE
```
Harness: emit our shared graphs (LFR, SBM, real) in deDoc's edge format; sweep
N ∈ {50, 100, 500, 1k, 5k, 10k, 50k}; record wall-clock + partition; compute
ARI/NMI against ground truth in the harness (not in Java).

## Results (2026-06-02, box 39169948, RTX 3060 — CPU/Java)
Artifact: `suite/results/community_detection/deDoc__SBM-scalability__2026-06-02.json`

**Scalability (the headline):** deDoc(E) ran the full sweep; wall-clock:

| N | 50 | 100 | 300 | 1000 | 3000 | 10000 |
|---|---|---|---|---|---|---|
| sec | 0.44 | 0.83 | 2.6 | 8.0 | 66 | 655 |

10× nodes (1000→10000) → 82× time ⇒ empirical scaling ≈ **O(N^1.9)** (quadratic-ish),
NOT cubic and NOT the paper's claimed O(n log²n) for this Java build.

### ✅ Verdict: claim REFUTED
The conference draft's "deDoc is O(N³), infeasible for N>50" is **false**: the
real jar processed **N=10,000 in ~11 min** on one CPU box. The draft's mark was
an artifact of the *glass-jax reimplementation*, not deDoc itself.

### Corrected accuracy + timing (constant-signal `sbm_scalable`, k=10, ~15 intra/3 inter)
Rerun 2026-06-02 (same artifact, fixed generator):

| N | 50 | 100 | 300 | 1000 | 3000 | 10000 |
|---|---|---|---|---|---|---|
| wall (s) | 0.3 | 0.4 | 0.9 | 3.2 | 10.8 | 74 |
| ARI | 0.961 | 1.000 | 1.000 | 1.000 | 1.000 | 0.730 |
| #comm (true=10) | 12 | 10 | 10 | 10 | 10 | 4006 |

With a non-degenerate graph deDoc is **both accurate and fast**: ARI=1.0 from
N=100–3000, N=10000 in 74 s (scaling ≈ O(N^1.4)). Mild over-segmentation at
N=10000 (4006 comms, ARI 0.73) is an honest large-N caveat. The earlier
all-singleton ARI=0 at large N was OUR confound (fixed-p_out SBM inverts the
signal past the detection threshold), not a deDoc failure.

### deDoc on the LFR sweep (shared graphs) — a GENUINE failure mode
On LFR (n=1000, avg-deg≈27, power-law degrees), deDoc(E) returns ~all-singletons
at every μ: LFR-μ0.1 → 996 communities (true=24), ARI≈0.0003, NMI=0.62. Contrast
SBM-Clean → 4 comms (true=3), ARI=0.99. So deDoc(E) recovers block structure but
**collapses to singletons on heterogeneous high-degree LFR graphs** — a real
limitation (failure-case section). This is NOT the earlier dataset confound: the
LFR graphs have strong community signal at low μ; deDoc(E) simply doesn't merge.
**Resolved:** deDoc(M) was tested on LFR-μ0.1 too → 987 communities, ARI=0.001
(same collapse). So **both deDoc variants** fail on LFR-type graphs — a robust,
variant-independent failure mode, not a parameterization artifact.

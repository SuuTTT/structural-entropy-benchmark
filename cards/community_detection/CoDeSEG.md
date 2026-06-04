# Repro card — CoDeSEG (community detection as a structural-entropy game)

**Paper:** Xian et al., *Community Detection in Large-Scale Networks via
Structural Entropy Game*, WWW 2025.
**Upstream repo:** https://github.com/SELGroup/CoDeSEG · **commit:** `d8ba74f`
**Language:** C++ core + Python wrappers.

## Layout (learned 2026-06-02)
- C++ source: `code_c++/CoDeSEG/` (cmake) — build the `CoDeSEG` binary.
- Driver: `code_py/CoDeSEG.py` → calls binary with
  `-i <edgelist> -o <out> -n 10 -t <ground_truth> -e <tau=0.3> -p <parallel>`;
  flags: `-x` overlapping, `-w` weighted, `-d` directed, `-v` verbose.
- **Supports BOTH** non-overlapping (use ARI/NMI) and overlapping (`-x`, score
  with bundled `onmi.py` / `xmeasures.py`).
- Bundled datasets: `dataset/{lfr_overlap, tweet12, tweet18}`; bundled baselines
  (louvain/leiden/bigclam/SLPA/NCG/fox).

## Build (on box, next cycle)
```bash
cd /root/se-bench-repos/CoDeSEG/code_c++/CoDeSEG && \
  cmake -S . -B build && cmake --build build -j   # -> build/CoDeSEG
```
NOTE the driver hardcodes `Game_se="../code/build/CoDeSEG"`; point it at the real
built path (a wrapper, not a core edit).

## Plan
- Non-overlapping: feed our shared LFR-μ-sweep + SBM-scalable graphs, score
  ARI/NMI + cross-objective; compare to baselines + deDoc + DeSE.
- Overlapping: run on `lfr_overlap`, score overlapping-NMI (onmi).
- Scalability: time vs N (CoDeSEG claims near-linear O(nt)).

## Results
Pending build+run (box1, after deDoc-clean finishes).

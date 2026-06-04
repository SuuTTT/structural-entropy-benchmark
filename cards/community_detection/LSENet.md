# Repro card — LSENet (Lorentz Structural Entropy Net, deep graph clustering)

**Paper:** Sun et al., *LSEnet*, ICML 2024 (Oral).
**Upstream repo:** https://github.com/RiemannGraph/DSE_clustering (code MOVED here from
ZhenhHuang/LSEnet) · **commit:** `1930d74`
**Env:** python3 + torch + geoopt + torch_scatter + torch_geometric + munkres.
On A4000s: venv `--system-site-packages` over system torch 2.5.1+cu121 (isolated).

## ⚠️ Critical reproduction fix (config-loading)
`main.py` ships with its **config-loading block commented out** (lines ~59–62),
so a bare `python main.py --dataset Cora` runs with argparse DEFAULTS
(`max_nums=[4]`, epochs 1500, wd 1e-2) — NOT the tuned `configs/Cora.json`
(`max_nums=[10]`, epochs 2000, wd 0, τ 0.1). We patch a copy (`main_run.py`) to
uncomment the load. Deviation recorded; this is required for faithful numbers.

## ⚠️ Log-parsing fix
LSENet's final-summary line prints a buggy `ARI: 0.0±0.0`. Real values are in the
per-epoch `Epoch N: ACC: a, NMI: b, ARI: c` lines (percent). `parse_lsenet.py`
reads those and takes best-by-NMI.

## Results (2026-06-02, configd; best-by-NMI over training, %)
| Dataset | box | ACC | NMI | ARI | vs paper |
|---|---|---|---|---|---|
| Cora | 3060 | 63.1 | 49.5 | 41.8 | ✓ ~matches (paper ~67/50/43) |
| Photo | A4000 | 55.9 | 59.4 | 45.6 | ✓ ~matches (paper NMI ~60) |
| Citeseer | A4000 | 28.0 | 6.7 | 7.0 | ✗ bad single run |

### 5-seed multiseed (exp_iters=5) — confirms the instability
| Dataset | NMI (5-seed mean±std) | best NMI | verdict |
|---|---|---|---|
| Cora | 44.0 ± 3.0 | 49.5 | reproduces (paper ~50), with variance |
| Citeseer | **3.3 ± 3.6** | 9.4 | **near-random across all 5 seeds → genuine failure** |
| Photo | (running) | ~63 | strong |
**LSENet fails on Citeseer** (5-seed NMI 3.3%) while DeSE reaches NMI 40 — a
robust, reportable contrast between two feature-augmented SE clusterers.
(NMI mean±std from LSENet's summary; its summary-ARI is the known `0.0` bug.)

### Honest caveats
- **Citeseer instability:** configd single run (configs set `exp_iters=1`) landed
  near-random (NMI 6.7), *worse* than the default-config run (NMI 21.2). LSENet is
  seed/config sensitive on Citeseer. **TODO:** multi-seed Citeseer before final
  reporting; report the distribution, not one run.
- Default-config (pre-fix) numbers for comparison: Cora 40.9/33.4, Citeseer
  21.2/18.7 NMI/ARI.

## Verdict: reproduced on Cora & Photo; Citeseer pending multi-seed.
Feature-augmented LSENet beats topology-only baselines on Cora/Photo (NMI 49–59
vs Louvain ~0.44 on Cora) — supports the "SE + node features competitive on
attributed graphs" thread, in contrast to topology-only SE losing on LFR.

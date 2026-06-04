# Repro card — UnDBot (Unsupervised social-Bot detection via Structural Information)

**Paper:** Peng et al., *Unsupervised Social Bot Detection via Structural
Information Theory*, ACM TOIS 2024. **Repo:** https://github.com/SELGroup/UnDBot
**Stack:** Python 3.10, numpy/numba/ijson. **Bundled datasets:** botwiki-2019 (699),
cresci-2015, cresci-2017, pronbots-2019 (17.9k) — with `*_f.csv` features + `label.csv`.
**Category (NEW):** social-bot detection — builds a multi-relational user graph and
labels bot communities via heterogeneous SE minimization (multirank + SE tree).

## Run
```bash
python3 main.py   # test(<dataset>) — edit the dataset arg in __main__
```
ijson required. CPU/numba.

## Status (2026-06-04)
- **pronbots-2019 (17.9k accounts): OOM** on the 3070-laptop RAM (multirank builds
  dense matrices) — needs a higher-RAM CPU box.
- **botwiki-2019 (699): AUC 0.883** (acc 0.41/F1 0.51 — modest on this small bot-skewed set; 125 comms, 2.9s). Pipeline reproduces.
Honest note: large social graphs need more RAM than the laptop box; small datasets
reproduce the unsupervised bot-detection pipeline.

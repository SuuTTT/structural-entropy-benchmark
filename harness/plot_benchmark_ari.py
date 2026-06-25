"""Generate the LFR community-detection summary figure for the survey.

Reads results/community_detection/_summary/lfr_sweep.csv (the only source of
numbers; produced by aggregate.py over the per-run result JSONs) and renders a
two-panel ARI / NMI vs. mixing-parameter figure in the paper's serif font.

    python3 plot_benchmark_ari.py \
        --csv ../results/community_detection/_summary/lfr_sweep.csv \
        --out <survey>/figures/benchmark_ari_summary.pdf
"""
from __future__ import annotations
import argparse, csv, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt

# Display label, line style, marker, color, is-SE-method
STYLE = {
    "Louvain":  dict(label="Louvain",      ls="-",  marker="o", color="#1f77b4"),
    "Leiden":   dict(label="Leiden",       ls="-",  marker="s", color="#ff7f0e"),
    "Infomap":  dict(label="Infomap",      ls="-",  marker="^", color="#2ca02c"),
    "Spectral": dict(label="Spectral",     ls="-",  marker="D", color="#d62728"),
    "CoDeSEG":  dict(label="CoDeSEG (SE)",  ls="--", marker="x", color="#9467bd"),
    "deDoc":    dict(label="deDoc (SE)",    ls="--", marker="+", color="#8c564b"),
}
ORDER = ["Louvain", "Leiden", "Infomap", "Spectral", "CoDeSEG", "deDoc"]


def load(csv_path):
    # series[method] -> list of (mu, ari, ari_ci, nmi, nmi_ci)
    series = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            series[row["method"]].append((
                float(row["mu"]),
                float(row["ari_mean"]), float(row["ari_ci"]),
                float(row["nmi_mean"]), float(row["nmi_ci"]),
            ))
    for m in series:
        series[m].sort(key=lambda t: t[0])
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    series = load(args.csv)
    fig, (axA, axN) = plt.subplots(1, 2, figsize=(9.4, 3.9), sharex=True)

    for m in ORDER:
        if m not in series:
            continue
        st = STYLE[m]
        mus = [t[0] for t in series[m]]
        ari = [t[1] for t in series[m]]
        ari_ci = [t[2] for t in series[m]]
        nmi = [t[3] for t in series[m]]
        nmi_ci = [t[4] for t in series[m]]
        axA.errorbar(mus, ari, yerr=ari_ci, label=st["label"], ls=st["ls"],
                     marker=st["marker"], color=st["color"], capsize=2, lw=1.6, ms=5)
        axN.errorbar(mus, nmi, yerr=nmi_ci, label=st["label"], ls=st["ls"],
                     marker=st["marker"], color=st["color"], capsize=2, lw=1.6, ms=5)

    for ax, ylab, title in ((axA, "ARI", "(a) Adjusted Rand Index"),
                            (axN, "NMI", "(b) Normalized Mutual Information")):
        ax.set_xlabel(r"LFR mixing parameter $\mu$ (higher = harder)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axA.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    fig.suptitle("Community detection on LFR: SE methods vs. classical baselines",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

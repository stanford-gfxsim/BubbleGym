"""Supplement figure: residual vs direct head, one per-bin panel per input feature.

The compact paper version of ``feature_axis_panels.png``: the same eight
stratified test-split panels (``alt_bin_sets/<feature>/``), laid out two rows of
four under one legend, without the render strips or per-panel totals (the
supplement's table carries those).

Usage:
    python python/visualization/plot_heads_feature_panels.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

FIG04_DIR = Path("results/experiments/fig04_per_bin_model_error")
TEAL, ORANGE, GREY, DARK = "#1f9e89", "#e0792b", "#888888", "#444444"

SERIES = [
    ("minnaert", "Minnaert baseline", GREY, "--", "v"),
    ("strasberg", "Ellipsoid proxy", DARK, "--", "^"),
    ("residual", "Residual-learning surrogate", TEAL, "-", "o"),
    ("direct", "Direct-learning surrogate", ORANGE, "-", "s"),
]

# Trainer column order, with the symbols the paper uses.
FEATURES = [
    ("i11_over_i00", r"$\tilde I_{11}/\tilde I_{00}$"),
    ("i22_over_i00", r"$\tilde I_{22}/\tilde I_{00}$"),
    ("non_sph_va", r"$\bar{\Phi}_{VA}$"),
    ("non_sph_vm", r"$\bar{\Phi}_{VM}$"),
    ("non_sph_w", r"$\bar{\Phi}_{W}$"),
    ("eta_V", r"$\eta_V$"),
    ("eta_A", r"$\eta_A$"),
    ("eta_M", r"$\eta_M$"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fig04-dir", type=Path, default=FIG04_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fontsize", type=float, default=12.0)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    if args.out is None:
        args.out = args.fig04_dir / "heads_feature_panels.png"
    fs = float(args.fontsize)

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.0), dpi=args.dpi, sharey=True)
    for i, (ax, (name, symbol)) in enumerate(zip(axes.ravel(), FEATURES)):
        pb = pd.read_csv(args.fig04_dir / "alt_bin_sets" / name / "per_bin_mape.csv")
        for key, label, colour, ls, marker in SERIES:
            y = pb[pb.series == key].sort_values("bin")["mape_pct"].to_numpy()
            ax.plot(np.arange(len(y)), y, color=colour, label=label, linestyle=ls,
                    marker=marker, markersize=4, linewidth=1.6 if ls == "-" else 1.2)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0, 10.0]))
        ax.yaxis.set_major_formatter(FixedFormatter(["0.01%", "0.1%", "1%", "10%"]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0.01, 40.0)
        ax.set_xticks(np.arange(len(y)))
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=fs - 1.0)
        ax.grid(alpha=0.25, axis="y", which="both")
        ax.set_title(f"({'abcdefgh'[i]}) binned on {symbol}", fontsize=fs, loc="left", pad=5)
        ax.set_xlabel(f"low {symbol}  →  high {symbol}", fontsize=fs - 1.0, labelpad=3)
    for ax in axes[:, 0]:
        ax.set_ylabel("MAPE (%)", fontsize=fs)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=fs,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.0, w_pad=0.6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

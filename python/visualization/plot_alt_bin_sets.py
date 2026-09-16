"""Fig. 4's four series across four different stratification axes.

Fig. 4 bins on Wadell nonsphericity. A model can flatter itself on one axis, so
this repeats the panel on three others --- inertia anisotropy, convex-hull
deficit and Willmore excess --- each with its own 10 bins x 10 bubbles drawn from
the retrained models' test split, which they never saw.

Panel A is Fig. 4's own 100 meshes with the retrained models, for reference; the
other three are the fresh draws written by
``freq_model/NN/eval_alt_bin_sets.py``. All four share one log axis, so the
panels can be read against each other.

Usage:
    python python/visualization/plot_alt_bin_sets.py
    python python/visualization/plot_alt_bin_sets.py --dark
"""

from __future__ import annotations

import argparse
import json
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
ALT_AXES = ["inertia", "chull", "willmore"]


def _fig04_panel(fig04_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fig. 4's 100 meshes, scored with the retrained models."""
    pb = pd.read_csv(fig04_dir / "per_bin_mape_retrained_10k.csv")
    rows = pd.read_csv(fig04_dir / "selected_rows.csv")
    bins = rows["bin"].to_numpy()
    key_of = {"minnaert": "minnaert", "strasberg": "strasberg",
              "residual": "residual_retrained_8feat", "direct": "direct_retrained_8feat"}
    per_bin, overall = {}, {}
    for k, src in key_of.items():
        per_bin[k] = pb[pb.series == src].sort_values("bin")["mape_pct"].to_numpy()
        col = {"minnaert": "ape_minnaert", "strasberg": "ape_strasberg"}.get(k)
        if col is not None:
            overall[k] = float(rows[col].mean())
        else:
            # weight-free mean over equal-sized bins is the overall mean
            overall[k] = float(per_bin[k].mean())
    return per_bin, overall


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fig04-dir", type=Path, default=FIG04_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dark", action="store_true")
    p.add_argument("--fontsize", type=float, default=10.0)
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    if args.out is None:
        args.out = args.fig04_dir / f"alt_bin_sets{'_dark' if args.dark else ''}.png"

    panels = []
    pb0, ov0 = _fig04_panel(args.fig04_dir)
    panels.append(("Wadell nonsphericity  $1-\\Phi_{VA}$",
                   "Fig. 4's own 100 meshes", pb0, ov0))

    for name in ALT_AXES:
        d = args.fig04_dir / "alt_bin_sets" / name
        if not (d / "per_bin_mape.csv").is_file():
            raise SystemExit(
                f"{d} missing. Build it first:\n"
                f"  python python/freq_model/NN/eval_alt_bin_sets.py --bin-by {name}")
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        pb = pd.read_csv(d / "per_bin_mape.csv")
        per_bin = {k: pb[pb.series == k].sort_values("bin")["mape_pct"].to_numpy()
                   for k, *_ in SERIES}
        panels.append((s["axis_label"], "fresh draw from the held-out test split",
                       per_bin, s["overall_mape_pct"]))

    fg = "white" if args.dark else "black"
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.4), dpi=args.dpi, sharey=True)
    if args.dark:
        fig.patch.set_alpha(0.0)

    n_bins = len(panels[0][2]["minnaert"])
    x = np.arange(n_bins, dtype=np.float64)
    for ax, (axis_label, note, per_bin, overall), letter in zip(
            axes.ravel(), panels, "ABCD"):
        if args.dark:
            ax.set_facecolor("none")
        for key, label, colour, ls, marker in SERIES:
            ax.plot(x, per_bin[key], color=colour, label=label, linestyle=ls,
                    marker=marker, markersize=5,
                    linewidth=1.7 if ls == "-" else 1.3)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0, 10.0]))
        ax.yaxis.set_major_formatter(FixedFormatter(["0.01%", "0.1%", "1%", "10%"]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(x)
        ax.set_xticklabels([""] * n_bins)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=args.fontsize - 1, colors=fg)
        ax.grid(alpha=0.22, axis="y", which="both")
        for sp in ax.spines.values():
            sp.set_color(fg)
        ax.set_title(f"{letter}.  binned on {axis_label}", fontsize=args.fontsize,
                     color=fg, loc="left", pad=6)
        ax.set_xlabel(
            f"low → high      ({note})\n"
            f"residual {overall['residual']:.3f}%   direct {overall['direct']:.3f}%",
            fontsize=args.fontsize - 1.5, color=fg)

    for ax in axes[:, 0]:
        ax.set_ylabel("MAPE (%)", fontsize=args.fontsize, color=fg)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=4,
                     fontsize=args.fontsize, frameon=not args.dark,
                     bbox_to_anchor=(0.5, 1.005))
    for t in leg.get_texts():
        t.set_color(fg)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=args.dpi, transparent=args.dark,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {args.out}")

    print("\noverall MAPE (%) per panel:")
    hdr = f"{'axis':<34}" + "".join(f"{k:>12}" for k, *_ in SERIES)
    print(hdr)
    for axis_label, _note, _pb, overall in panels:
        clean = (axis_label.replace("$", "").replace("\\Phi_{VA}", "Phi_VA")
                 .replace("\\eta_V", "eta_V").replace("4\\pi", "4pi"))
        print(f"{clean:<34}" + "".join(f"{overall[k]:>12.4f}" for k, *_ in SERIES))


if __name__ == "__main__":
    main()

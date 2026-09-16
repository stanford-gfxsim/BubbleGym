"""Redraw Fig. 4's panel with both learned surrogates retrained on the 10k benchmark.

Fig. 4's published models were trained on the 8,998-row predecessor of the
released benchmark and their checkpoints were never archived. This plots the same
100 bubbles, in the same ten nonsphericity bins, adding the retrained pair from
``freq_model/NN/fit_shape_freq_model_variants.py``.

Six series, all scored on the identical mesh set:

  Minnaert baseline                analytic  \\  read from the archived run
  Ellipsoid proxy                  analytic   > shipped alongside the figure
  Residual surrogate, published    learned   /   in fig04_per_bin_model_error/
  Direct surrogate, published      learned  /

  Residual, retrained on 10k       learned, this repository's trainer
  Direct, retrained on 10k         learned, this repository's trainer

Colour identifies the method, linestyle the vintage: dashed with a hollow marker
is the published model, solid and filled is the retrain. Both retrained models
held all 100 of these bubbles out of training and validation, so every point is
an out-of-sample prediction.

The CSV this writes carries every variant, both feature sets included, whichever
pair the figure draws.

Usage:
    python python/visualization/plot_fig04_retrained_comparison.py
    python python/visualization/plot_fig04_retrained_comparison.py --retrain-features 6
    python python/visualization/plot_fig04_retrained_comparison.py --dark
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from bin_thumbnail_strip import (  # noqa: E402
    FIG4_FIGSIZE,
    add_thumbnail_strip,
    bin_representatives,
)

FIG04_DIR = Path("results/experiments/fig04_per_bin_model_error")
RETRAIN_DIR = Path("python/freq_model/output/fig04_retrain")

TEAL, ORANGE, GREY, DARK = "#1f9e89", "#e0792b", "#888888", "#444444"

# key -> (label, colour, linestyle, marker, filled)
STYLE = {
    "minnaert": ("Minnaert baseline", GREY, "--", "v", True),
    "strasberg": ("Ellipsoid proxy", DARK, "--", "^", True),
    "residual_published": ("Residual surrogate (published)", TEAL, "--", "o", False),
    "direct_published": ("Direct surrogate (published)", ORANGE, "--", "s", False),
    "residual_retrained": ("Residual, retrained on 10k", TEAL, "-", "o", True),
    "direct_retrained": ("Direct, retrained on 10k", ORANGE, "-", "s", True),
}
PLOT_ORDER = ["minnaert", "strasberg", "residual_published", "residual_retrained",
              "direct_published", "direct_retrained"]

# --only-retrained: the published pair drops out and the retrains carry the plain
# names, giving the published figure's four-series layout with the new models.
PLOT_ORDER_RETRAINED_ONLY = ["minnaert", "strasberg", "residual_retrained", "direct_retrained"]
LABELS_RETRAINED_ONLY = {
    "residual_retrained": "Residual-learning surrogate",
    "direct_retrained": "Direct-learning surrogate",
}


def _load_retrained(model_dir: Path, rows: pd.DataFrame) -> np.ndarray:
    """Per-row APE (%) for a retrained variant, aligned to ``rows`` by mesh id."""
    pred_csv = model_dir / "holdout_predictions.csv"
    if not pred_csv.is_file():
        raise SystemExit(
            f"{pred_csv} not found. Train it first, e.g.\n"
            f"  python python/freq_model/NN/fit_shape_freq_model_variants.py \\\n"
            f"      --features 8 --baseline strasberg \\\n"
            f"      --holdout-mesh-ids {FIG04_DIR.as_posix()}/selected_rows.csv")
    pred = pd.read_csv(pred_csv).set_index("mesh_id")
    missing = set(rows["mesh_id_10k"]) - set(pred.index)
    if missing:
        raise SystemExit(f"{pred_csv} is missing {len(missing)} of the figure's meshes")
    aligned = pred.loc[rows["mesh_id_10k"]]

    gt_gap = float(np.max(np.abs(aligned["f_gt"].to_numpy(float) - rows["f_gt"].to_numpy(float))))
    if gt_gap > 1e-5:
        raise SystemExit(f"{pred_csv} ground truth differs from the figure's by {gt_gap:.2e} Hz")
    return aligned["ape_pct"].to_numpy(dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fig04-dir", type=Path, default=FIG04_DIR)
    p.add_argument("--retrain-dir", type=Path, default=RETRAIN_DIR)
    p.add_argument("--retrain-features", type=int, choices=(6, 8), default=8,
                   help="Which retrained pair the figure draws (default 8, the "
                        "production feature set). The CSV always carries both.")
    p.add_argument("--only-retrained", action="store_true",
                   help="Drop the published pair: baselines plus the two retrained "
                        "models alone, in the published figure's four-series layout, "
                        "with its thumbnail strip.")
    p.add_argument("--no-thumbnails", action="store_true",
                   help="Skip the strip under the axis (--only-retrained only).")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--dark", action="store_true", help="White lettering on a transparent ground.")
    p.add_argument("--fontsize", type=float, default=11.0)
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    tag = "" if args.retrain_features == 8 else f"_{args.retrain_features}feat"
    if args.only_retrained:
        tag += "_only"
    if args.out is None:
        args.out = args.fig04_dir / (
            f"per_bin_mape_retrained_10k{tag}{'_dark' if args.dark else ''}.png")
    if args.csv_out is None:
        args.csv_out = args.fig04_dir / "per_bin_mape_retrained_10k.csv"

    rows = pd.read_csv(args.fig04_dir / "selected_rows.csv")
    n_bins = int(rows["bin"].max()) + 1
    if sorted(rows.groupby("bin").size().unique()) != [10]:
        raise SystemExit("expected 10 bubbles in every bin")
    bins = rows["bin"].to_numpy()

    # Everything, both feature sets, for the CSV.
    ape_all: dict[str, np.ndarray] = {
        "minnaert": rows["ape_minnaert"].to_numpy(dtype=np.float64),
        "strasberg": rows["ape_strasberg"].to_numpy(dtype=np.float64),
        "residual_published": rows["ape_residual"].to_numpy(dtype=np.float64),
        "direct_published": rows["ape_direct"].to_numpy(dtype=np.float64),
    }
    label_all = {k: STYLE[k][0] for k in ape_all}
    for n_feat in (6, 8):
        for objective in ("residual", "direct"):
            key = f"{objective}_retrained_{n_feat}feat"
            ape_all[key] = _load_retrained(
                args.retrain_dir / f"{n_feat}feature_{objective}", rows)
            label_all[key] = f"{objective.capitalize()}, retrained on 10k ({n_feat} feat)"

    per_bin_all = {k: np.array([v[bins == b].mean() for b in range(n_bins)])
                   for k, v in ape_all.items()}

    drawn = {f"{o}_retrained_{args.retrain_features}feat": f"{o}_retrained"
             for o in ("residual", "direct")}
    records = []
    for key, vals in per_bin_all.items():
        for b in range(n_bins):
            sub = ape_all[key][bins == b]
            records.append({
                "series": key, "label": label_all[key],
                "in_default_figure": key in ape_all and (
                    key in ("minnaert", "strasberg", "residual_published", "direct_published")
                    or key in drawn),
                "bin": b, "n": int(sub.size),
                "mape_pct": float(vals[b]),
                "sem_ape_pct": float(sub.std(ddof=1) / np.sqrt(sub.size)),
                "max_ape_pct": float(sub.max()),
            })
    pd.DataFrame(records).to_csv(args.csv_out, index=False)

    # Map the chosen feature set onto the two "retrained" plot slots.
    ape = {k: ape_all[k] for k in ("minnaert", "strasberg",
                                   "residual_published", "direct_published")}
    for src, dst in drawn.items():
        ape[dst] = ape_all[src]
    per_bin = {k: np.array([v[bins == b].mean() for b in range(n_bins)]) for k, v in ape.items()}

    fg = "white" if args.dark else "black"
    # The clean panel reproduces Fig. 4's layout, thumbnail strip included.
    want_strip = args.only_retrained and not args.no_thumbnails
    fig, ax = plt.subplots(figsize=FIG4_FIGSIZE if want_strip else (11.0, 5.4),
                           dpi=args.dpi)
    if args.dark:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

    order = PLOT_ORDER_RETRAINED_ONLY if args.only_retrained else PLOT_ORDER
    x = np.arange(n_bins, dtype=np.float64)
    for key in order:
        label, colour, ls, marker, filled = STYLE[key]
        if key.endswith("_retrained"):
            label = (LABELS_RETRAINED_ONLY[key] if args.only_retrained
                     else f"{label} ({args.retrain_features} feat)")
        ax.plot(x, per_bin[key], color=colour, label=label, linestyle=ls,
                marker=marker, markersize=6,
                markerfacecolor=colour if filled else ("none"),
                markeredgecolor=colour,
                linewidth=1.8 if ls == "-" else 1.4)

    fs = float(args.fontsize)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0, 10.0]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.01%", "0.1%", "1%", "10%"]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("MAPE (%)", fontsize=fs, color=fg)
    ax.set_xlabel("low nonsphericity  →  high nonsphericity", fontsize=fs, color=fg)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * n_bins)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=fs - 1.0, colors=fg)
    ax.grid(alpha=0.25, axis="y", which="both")
    for spine in ax.spines.values():
        spine.set_color(fg)
    leg = ax.legend(fontsize=fs - 1.0 if args.only_retrained else fs - 1.5,
                    loc="upper left", ncol=1 if args.only_retrained else 2,
                    framealpha=0.0 if args.dark else 0.85)
    for t in leg.get_texts():
        t.set_color(fg)

    fig.tight_layout()
    if want_strip:
        n_drawn = add_thumbnail_strip(ax, bin_representatives(rows, n_bins))
        print(f"thumbnail strip: {n_drawn}/{n_bins} renders")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, transparent=args.dark,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"wrote {args.out}")
    print(f"wrote {args.csv_out}")
    print("\noverall MAPE (%) over the 100 bubbles:")
    for key, vals in ape_all.items():
        mark = "*" if key in drawn or key in (
            "minnaert", "strasberg", "residual_published", "direct_published") else " "
        print(f" {mark} {label_all[key]:<44} {vals.mean():7.4f}   "
              f"worst bin {per_bin_all[key].max():7.4f}")
    print("\n* drawn in this figure")


if __name__ == "__main__":
    main()

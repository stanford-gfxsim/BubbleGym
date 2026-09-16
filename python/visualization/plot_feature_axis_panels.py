"""One per-bin error panel per model feature: eight plots, plus a combined grid.

Fig. 4 bins on a single shape measure. The surrogates take eight, so this
stratifies along each of them in turn --- 10 bins of 10 bubbles drawn from the
retrained models' held-out test split --- and draws the same four series on each.

The eight are binned on their **raw** value, ascending, so the panels are not all
oriented the same way. ``non_sph_*`` and the two inertia ratios grow as a bubble
departs from a sphere, so those panels climb left to right; the convex-hull
ratios ``eta_*`` approach 1 for a convex bubble, so theirs fall. Reading the
slope is the point: it says whether a feature actually orders the difficulty.

Writes ``per_bin_mape.png`` inside each axis folder plus a combined
``feature_axis_panels.png``, all on one shared log scale so the eight are
directly comparable. Every panel carries Fig. 4's render strip, which is why the
combined grid is 4x2 rather than 2x4: ten thumbnails need a panel wide enough
that they do not collide. ``--no-thumbnails`` drops the strips and falls back to
the compact 2x4.

Usage:
    python python/visualization/plot_feature_axis_panels.py
    python python/visualization/plot_feature_axis_panels.py --dark
"""

from __future__ import annotations

import argparse
import json
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
TEAL, ORANGE, GREY, DARK = "#1f9e89", "#e0792b", "#888888", "#444444"

SERIES = [
    ("minnaert", "Minnaert baseline", GREY, "--", "v"),
    ("strasberg", "Ellipsoid proxy", DARK, "--", "^"),
    ("residual", "Residual-learning surrogate", TEAL, "-", "o"),
    ("direct", "Direct-learning surrogate", ORANGE, "-", "s"),
]

# The model's eight features, in the trainer's column order.
FEATURES = ["i11_over_i00", "i22_over_i00",
            "non_sph_va", "non_sph_vm", "non_sph_w",
            "eta_V", "eta_A", "eta_M"]

# Only the two convex-hull ratios that approach 1 for a convex bubble run the
# other way; every other axis, composite ones included, grows with nonsphericity.
SHRINKS_WITH_NONSPHERICITY = {"eta_V", "eta_A"}


def grows_with_nonsphericity(scalar_name: str) -> bool:
    return scalar_name not in SHRINKS_WITH_NONSPHERICITY

YTICKS = [0.01, 0.1, 1.0, 10.0]
YLABELS = ["0.01%", "0.1%", "1%", "10%"]


def _load(axis_dir: Path, name: str):
    if not (axis_dir / "per_bin_mape.csv").is_file():
        raise SystemExit(
            f"{axis_dir} missing. Build it first:\n"
            f"  python python/freq_model/NN/eval_alt_bin_sets.py --bin-by {name}")
    pb = pd.read_csv(axis_dir / "per_bin_mape.csv")
    rows = pd.read_csv(axis_dir / "selected_rows.csv")
    s = json.loads((axis_dir / "summary.json").read_text(encoding="utf-8"))
    per_bin = {k: pb[pb.series == k].sort_values("bin")["mape_pct"].to_numpy()
               for k, *_ in SERIES}
    return per_bin, s["overall_mape_pct"], s, rows


def _draw(ax, per_bin, overall, s, fg, fs, *, title_prefix="") -> None:
    n_bins = len(per_bin["minnaert"])
    x = np.arange(n_bins, dtype=np.float64)
    for key, label, colour, ls, marker in SERIES:
        ax.plot(x, per_bin[key], color=colour, label=label, linestyle=ls,
                marker=marker, markersize=5, linewidth=1.7 if ls == "-" else 1.3)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(YTICKS))
    ax.yaxis.set_major_formatter(FixedFormatter(YLABELS))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.008, 40.0)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * n_bins)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=fs - 1, colors=fg)
    ax.grid(alpha=0.22, axis="y", which="both")
    for sp in ax.spines.values():
        sp.set_color(fg)

    lo, hi = s["bin_edges"][0], s["bin_edges"][-1]
    arrow = ("more spherical → less" if grows_with_nonsphericity(s["scalar"])
             else "less spherical → more")
    ax.set_title(f"{title_prefix}binned on {s['axis_label']}",
                 fontsize=fs, color=fg, loc="left", pad=6)
    ax.set_xlabel(f"{lo:.3g} → {hi:.3g}   ({arrow})\n"
                  f"residual {overall['residual']:.3f}%   direct {overall['direct']:.3f}%",
                  fontsize=fs - 1.5, color=fg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fig04-dir", type=Path, default=FIG04_DIR)
    p.add_argument("--dark", action="store_true")
    p.add_argument("--no-thumbnails", action="store_true",
                   help="Skip the render strip under each standalone panel.")
    p.add_argument("--axes", nargs="*", default=None,
                   help="Which axis folders get a standalone panel "
                        "(default: every one built under alt_bin_sets/).")
    p.add_argument("--panel-height", type=float, default=6.0,
                   help="Inches per row of the combined grid, panel plus its "
                        "strip (default 6.0).")
    p.add_argument("--col-gap", type=float, default=0.13,
                   help="Horizontal gap between the grid's columns, as a "
                        "fraction of panel width (matplotlib wspace).")
    p.add_argument("--fontsize", type=float, default=13.0)
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    suffix = "_dark" if args.dark else ""
    fg = "white" if args.dark else "black"
    base = args.fig04_dir / "alt_bin_sets"

    # Every axis folder gets its own panel; the combined grid shows the eight
    # model features. `--axes` narrows the standalone set.
    present = sorted(d.name for d in base.iterdir()
                     if d.is_dir() and (d / "per_bin_mape.csv").is_file())
    standalone = args.axes if args.axes else present
    unknown = [a for a in standalone if a not in present]
    if unknown:
        raise SystemExit(f"no bin set built for {unknown}; found {present}")

    loaded = {f: _load(base / f, f) for f in sorted(set(standalone) | set(FEATURES))}

    # ---- standalone panels, each in Fig. 4's layout ------------------------
    for name in standalone:
        per_bin, overall, s, rows = loaded[name]
        fig, ax = plt.subplots(figsize=FIG4_FIGSIZE, dpi=args.dpi)
        if args.dark:
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
        _draw(ax, per_bin, overall, s, fg, args.fontsize)
        ax.set_ylabel("MAPE (%)", fontsize=args.fontsize, color=fg)
        leg = ax.legend(fontsize=args.fontsize - 1.5, loc="upper left",
                        framealpha=0.0 if args.dark else 0.85)
        for t in leg.get_texts():
            t.set_color(fg)
        fig.tight_layout()
        if not args.no_thumbnails:
            n_drawn = add_thumbnail_strip(ax, bin_representatives(rows, len(per_bin["minnaert"])))
            if n_drawn != len(per_bin["minnaert"]):
                print(f"[{name}] only {n_drawn} thumbnails drawn")
        out = base / name / f"per_bin_mape{suffix}.png"
        fig.savefig(out, dpi=args.dpi, transparent=args.dark,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")

    # ---- combined grid ----------------------------------------------------
    # With strips the panels must stay wide enough for ten renders not to
    # collide, so the grid turns 4x2 and each panel keeps Fig. 4's proportions.
    # Without them the compact 2x4 is fine.
    strips = not args.no_thumbnails
    if strips:
        # Each panel gets Fig. 4's own proportions: ~11in wide over a ~4.2in
        # panel, plus room beneath for the render strip.
        nrows, ncols = 4, 2
        figsize, hspace = (22.0, args.panel_height * nrows), 0.46
        wspace = args.col_gap
    else:
        nrows, ncols, figsize, hspace = 2, 4, (21.0, 8.0), 0.30
        wspace = args.col_gap
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=args.dpi, sharey=True)
    if args.dark:
        fig.patch.set_alpha(0.0)
    for ax, name, letter in zip(axes.ravel(), FEATURES, "ABCDEFGH"):
        if args.dark:
            ax.set_facecolor("none")
        per_bin, overall, s, rows = loaded[name]
        _draw(ax, per_bin, overall, s, fg, args.fontsize, title_prefix=f"{letter}.  ")
    for ax in axes[:, 0]:
        ax.set_ylabel("MAPE (%)", fontsize=args.fontsize, color=fg)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=4,
                     fontsize=args.fontsize + 1, frameon=not args.dark,
                     bbox_to_anchor=(0.5, 1.005))
    for t in leg.get_texts():
        t.set_color(fg)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    # The strips are annotations, which tight_layout cannot see, so the row gap
    # is opened by hand afterwards and the renders drawn into it.
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    if strips:
        for ax, name in zip(axes.ravel(), FEATURES):
            _pb, _ov, _s, rows = loaded[name]
            add_thumbnail_strip(ax, bin_representatives(rows, len(_pb["minnaert"])))
    out = args.fig04_dir / f"feature_axis_panels{suffix}.png"
    fig.savefig(out, dpi=args.dpi, transparent=args.dark,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")

    # ---- one table over the eight ---------------------------------------
    recs = []
    for name in FEATURES:
        _pb, overall, s, _rows = loaded[name]
        recs.append({"binned_on": name,
                     "grows_with_nonsphericity": grows_with_nonsphericity(s["scalar"]),
                     **{f"mape_{k}": overall[k] for k, *_ in SERIES}})
    tab = pd.DataFrame(recs)
    csv_out = args.fig04_dir / "feature_axis_panels.csv"
    tab.to_csv(csv_out, index=False)
    print(f"wrote {csv_out}\n")
    print(tab.to_string(index=False,
                        formatters={c: "{:.4f}".format for c in tab.columns
                                    if c.startswith("mape_")}))


if __name__ == "__main__":
    main()

"""Fig. 4 as it now appears in the paper: the Wadell-binned panel of retrained models.

Draws ``alt_bin_sets/non_sph_va`` -- 100 bubbles from the 8-feature retrains'
test split, ten bins of Wadell nonsphericity ``1 - Phi_VA`` -- in the published
figure's layout, with the published figure's own ten renders beneath it.

Those renders are not members of this panel: they come from the original Fig. 4
set (``selected_rows.csv``), which the retrains held out entirely. The script
re-derives them with the published picker's rule -- in each of that set's bins,
the VOF bubble nearest the bin's median nonsphericity, ties going to the one
earlier in the 10k dataset CSV -- and checks the result against the ten ids
recorded in ``paper_thumbnails.json``. They illustrate each bin's shape range,
so the script also refuses to draw one that falls outside its bin here.

Usage:
    python python/visualization/plot_fig04_wadell_panel.py
    python python/visualization/plot_fig04_wadell_panel.py --dark
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
from bin_thumbnail_strip import FIG4_FIGSIZE, add_thumbnail_strip  # noqa: E402

FIG04_DIR = Path("results/experiments/fig04_per_bin_model_error")
DATASET_CSV = Path("dataset/bubble_gym/dataset_bubblegym_10k.csv")
TEAL, ORANGE, GREY, DARK = "#1f9e89", "#e0792b", "#888888", "#444444"

SERIES = [
    ("minnaert", "Minnaert baseline", GREY, "--", "v"),
    ("strasberg", "Ellipsoid proxy", DARK, "--", "^"),
    ("residual", "Residual-learning surrogate", TEAL, "-", "o"),
    ("direct", "Direct-learning surrogate", ORANGE, "-", "s"),
]


def wadell_nonsphericity(volume: np.ndarray, area: np.ndarray) -> np.ndarray:
    """``1 - Phi_VA``, Phi_VA = pi^(1/3) (6V)^(2/3) / A: 0 for a sphere."""
    return 1.0 - np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area


def pick_published_thumbnails(rows: pd.DataFrame) -> list[str]:
    """The published Fig. 4 renders, one per bin of the original 100-bubble set.

    ``plot_per_bin_mape_bars.py`` took, per bin, the VOF bubble whose
    ``non_sphericity`` is nearest the bin median. Bins hold an even count more
    often than not, so the two middle bubbles tie exactly; the tie goes to the
    one read first, i.e. the lower ``row_10k``.
    """
    vof = rows[rows["source_10k"] == "VOF"].sort_values("row_10k", kind="stable")
    out = []
    for b in range(int(rows["bin"].max()) + 1):
        sub = vof[vof["bin"] == b]
        v = sub["non_sphericity"].to_numpy(dtype=np.float64)
        k = int(np.argsort(np.abs(v - np.median(v)), kind="stable")[0])
        out.append(str(sub["mesh_id_10k"].iloc[k]))
    return out


def check_thumbnails_in_bins(mesh_ids: list[str], edges: list[float], dataset_csv: Path) -> None:
    df = pd.read_csv(dataset_csv)
    df["mesh_id"] = df["source"].astype(str) + "/" + df["mesh_filename"].astype(str)
    df = df.set_index("mesh_id")
    nva = wadell_nonsphericity(df.loc[mesh_ids, "volume"].to_numpy(float),
                               df.loc[mesh_ids, "surface_area"].to_numpy(float))
    for b, (mid, v) in enumerate(zip(mesh_ids, nva)):
        lo, hi = edges[b], edges[b + 1]
        if not lo <= v <= hi:
            raise SystemExit(f"thumbnail {mid} has 1-Phi_VA {v:.4f}, outside bin {b} [{lo:.4f}, {hi:.4f}]")
        print(f"bin {b}: {mid:28s} 1-Phi_VA {v:.4f} in [{lo:.4f}, {hi:.4f}]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fig04-dir", type=Path, default=FIG04_DIR)
    p.add_argument("--dataset", type=Path, default=DATASET_CSV)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dark", action="store_true", help="White lettering on a transparent ground.")
    p.add_argument("--fontsize", type=float, default=17.0)
    p.add_argument("--thumb-zoom", type=float, default=0.85,
                   help="Scale of each 100 px render (default 0.85).")
    p.add_argument("--thumb-y-offset", type=float, default=-0.08,
                   help="Axes-fraction y of the top of the render strip (default -0.08).")
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    panel = args.fig04_dir / "alt_bin_sets" / "non_sph_va"
    if args.out is None:
        args.out = args.fig04_dir / f"per_bin_mape_wadell_10k{'_dark' if args.dark else ''}.png"

    summary = json.loads((panel / "summary.json").read_text(encoding="utf-8"))
    pb = pd.read_csv(panel / "per_bin_mape.csv")
    per_bin = {k: pb[pb.series == k].sort_values("bin")["mape_pct"].to_numpy() for k, *_ in SERIES}
    n_bins = len(per_bin["minnaert"])

    thumbs = pick_published_thumbnails(pd.read_csv(args.fig04_dir / "selected_rows.csv"))
    recorded = json.loads((args.fig04_dir / "paper_thumbnails.json").read_text(encoding="utf-8"))["mesh_ids"]
    if thumbs != recorded:
        raise SystemExit(f"picker gives {thumbs},\nbut the published figure shows {recorded}")
    if len(thumbs) != n_bins:
        raise SystemExit(f"{len(thumbs)} thumbnails for {n_bins} bins")
    check_thumbnails_in_bins(thumbs, summary["bin_edges"], args.dataset)

    fg = "white" if args.dark else "black"
    fs = float(args.fontsize)
    fig, ax = plt.subplots(figsize=FIG4_FIGSIZE, dpi=args.dpi)
    if args.dark:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

    x = np.arange(n_bins, dtype=np.float64)
    for key, label, colour, ls, marker in SERIES:
        ax.plot(x, per_bin[key], color=colour, label=label, linestyle=ls, marker=marker,
                markersize=6, linewidth=1.8 if ls == "-" else 1.4)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0, 10.0]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.01%", "0.1%", "1%", "10%"]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.01, 60.0)
    ax.set_ylabel("MAPE (%)", fontsize=fs, color=fg)
    ax.set_xlabel("low nonsphericity  →  high nonsphericity", fontsize=fs, color=fg)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * n_bins)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=fs - 1.0, colors=fg)
    ax.grid(alpha=0.25, axis="y", which="both")
    for spine in ax.spines.values():
        spine.set_color(fg)
    leg = ax.legend(fontsize=fs - 1.0, loc="upper left", ncol=2, columnspacing=1.2,
                    framealpha=0.0 if args.dark else 0.85)
    for t in leg.get_texts():
        t.set_color(fg)

    fig.tight_layout()
    n_drawn = add_thumbnail_strip(ax, thumbs, zoom=args.thumb_zoom,
                                  y_offset=args.thumb_y_offset)
    if n_drawn != n_bins:
        raise SystemExit(f"only {n_drawn}/{n_bins} renders found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, transparent=args.dark,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    ov = summary["overall_mape_pct"]
    print(f"wrote {args.out}")
    print("overall MAPE (%): " + "  ".join(f"{k} {ov[k]:.4f}" for k, *_ in SERIES))


if __name__ == "__main__":
    main()

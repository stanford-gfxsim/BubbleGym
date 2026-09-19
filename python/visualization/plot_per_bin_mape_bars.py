"""Per-bin MAPE comparison plot (Fig. 3) for 3 series:

  1. Minnaert baseline
  2. Ellipsoid / Strasberg baseline
  3. The learned 8-feature surrogate

All three come from a single ``stratified_eval_model.py --model-kind nn8`` run,
so they share one selected set by construction (same seed / per-bin count / bin
metric / split.json).

Supports grouped bar charts (``--style bars``) and line/curve overlays
(``--style curves``, the default).

Usage:

    python python/freq_model/NN/stratified_eval_model.py --model-kind nn8 \\
        --output-dir results/stratified_eval_nn8
    python python/visualization/plot_per_bin_mape_bars.py \\
        --eval-dir results/stratified_eval_nn8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


_THIS_DIR = Path(__file__).resolve().parent
_PYTHON_ROOT = _THIS_DIR.parent
for _p in (_PYTHON_ROOT, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from visualization.plot_dataset_distribution import (  # noqa: E402
    DEFAULT_MESH_ROOT,
    get_or_render_transparent_thumbnail,
)


_SERIES_COLORS = {
    "minnaert": "#888888",
    "strasberg": "#444444",
    "model": "#e0792b",
}
_SERIES_LABELS = {
    "minnaert": "Minnaert baseline",
    "strasberg": "Ellipsoid baseline",
    "model": "Learned model (8 features)",
}


def _select_tim_thumbnail_per_bin(
    rows_csv: Path,
    bin_edges: np.ndarray,
    n_bins: int,
    *,
    mesh_root: Path,
) -> list[str | None]:
    """For each bin index, pick one VOF mesh_filename whose non_sphericity
    is closest to the bin median and whose .obj exists under ``mesh_root``.
    Returns a list of length ``n_bins`` (None where no candidate is found)."""
    if not rows_csv.is_file():
        return [None] * n_bins
    df = pd.read_csv(rows_csv)
    needed = {"mesh_filename", "non_sphericity", "source"}
    if not needed.issubset(df.columns):
        return [None] * n_bins
    sub = df[df["source"].astype(str) == "VOF"].copy()
    sub["non_sphericity"] = pd.to_numeric(sub["non_sphericity"], errors="coerce")
    sub = sub.dropna(subset=["non_sphericity", "mesh_filename"])
    if sub.empty:
        return [None] * n_bins

    out: list[str | None] = []
    ns = sub["non_sphericity"].to_numpy(dtype=np.float64)
    names = sub["mesh_filename"].astype(str).to_numpy()
    for b in range(n_bins):
        lo, hi = float(bin_edges[b]), float(bin_edges[b + 1])
        mask = (ns >= lo) & (ns <= hi if b == n_bins - 1 else ns < hi)
        if not np.any(mask):
            out.append(None)
            continue
        bin_ns = ns[mask]
        bin_names = names[mask]
        med = float(np.median(bin_ns))
        order = np.argsort(np.abs(bin_ns - med))
        chosen: str | None = None
        for k in order:
            cand = str(bin_names[int(k)])
            if (mesh_root / cand).is_file():
                chosen = cand
                break
        out.append(chosen)
    return out


def _render_thumbnail_paths(
    mesh_filenames: list[str | None],
    *,
    mesh_root: Path,
    cache_dir: Path,
    mesh_color: str,
    edge_color: str,
    force: bool,
) -> list[Path | None]:
    """Render-or-fetch a transparent thumbnail per mesh_filename. None entries
    pass through. Failed renders also become None."""
    out: list[Path | None] = []
    for name in mesh_filenames:
        if not name:
            out.append(None)
            continue
        try:
            p = get_or_render_transparent_thumbnail(
                name,
                mesh_root,
                cache_dir,
                mesh_color=mesh_color,
                edge_color=edge_color,
                force=force,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[thumb] {name}: {e}")
            p = None
        out.append(p)
    return out


def _load_per_bin(summary_path: Path) -> dict[str, list[dict]]:
    with summary_path.open(encoding="utf-8") as f:
        s = json.load(f)
    if "metrics_per_bin_selected" not in s:
        raise ValueError(f"{summary_path} missing 'metrics_per_bin_selected'")
    return s["metrics_per_bin_selected"]


def _vals_and_counts(per_bin: list[dict], n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    if len(per_bin) != n_bins:
        raise ValueError(f"per-bin length {len(per_bin)} != expected {n_bins}")
    vals = np.array([float(d.get("mape", float("nan"))) for d in per_bin], dtype=np.float64)
    counts = np.array([int(d.get("count", 0)) for d in per_bin], dtype=np.int64)
    return vals, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("results/stratified_eval_nn8"),
        help="stratified_eval_model.py --model-kind nn8 output dir (summary.json + selected_rows.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to per_bin_mape_<style>.png inside --eval-dir.",
    )
    parser.add_argument(
        "--style",
        choices=["curves", "bars"],
        default="curves",
        help="Plot style: 'curves' (line plot, default) or 'bars' (grouped bar chart).",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional fixed upper y-limit (MAPE in %%). Default: auto, with a "
             "log-y fallback if the largest series is much larger than the smallest.",
    )
    parser.add_argument("--log-y", action="store_true",
                        help="Force log-scale y axis (useful when baselines "
                             "dominate over learned models).")
    parser.add_argument(
        "--fontsize",
        type=float,
        default=11.0,
        help="Base font size (pt). Axis labels = fontsize, "
             "tick labels = fontsize-1, legend = fontsize-1.",
    )
    parser.add_argument(
        "--thumbnails",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a row of one VOF bubble thumbnail per bin below the x-axis.",
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=DEFAULT_MESH_ROOT,
        help="Directory containing per-bubble .obj meshes for the VOF source.",
    )
    parser.add_argument(
        "--thumbnail-cache-dir",
        type=Path,
        default=Path("dataset/langlois2016/thumbnail_transparent"),
        help="Cache directory for transparent-background langlois2016 thumbnails "
             "(reuses cached PNGs; renders missing ones via PyVista).",
    )
    parser.add_argument(
        "--thumb-zoom",
        type=float,
        default=0.20,
        help="Zoom factor for the thumbnail OffsetImage (smaller = smaller bubbles).",
    )
    parser.add_argument(
        "--thumb-y-offset",
        type=float,
        default=-0.09,
        help="Axes-fraction y position for the TOP of the thumbnail row "
             "(more-negative = farther below the x-axis).",
    )
    parser.add_argument(
        "--caption-gap",
        type=float,
        default=-0.02,
        help="Axes-fraction gap between the top of the thumbnail row and the "
             "'low non-sphericity -> high non-sphericity' caption above it.",
    )
    parser.add_argument(
        "--thumb-mesh-color",
        type=str,
        default="#bcd4f5",
        help="Light-blue mesh face color for the rendered langlois2016 thumbnails.",
    )
    parser.add_argument(
        "--thumb-edge-color",
        type=str,
        default="#2f4a6d",
        help="Edge color for the rendered langlois2016 thumbnails.",
    )
    parser.add_argument(
        "--force-rerender",
        action="store_true",
        help="Re-render cached thumbnails (PyVista) even when they already exist.",
    )
    args = parser.parse_args()

    if args.output is None:
        suffix = "curves" if args.style == "curves" else "bars"
        args.output = args.eval_dir / f"per_bin_mape_{suffix}.png"

    res_pb = _load_per_bin(args.eval_dir / "summary.json")

    missing = [k for k in ("minnaert", "strasberg", "model") if k not in res_pb]
    if missing:
        raise ValueError(
            f"{args.eval_dir}/summary.json missing per-bin entries: {missing}"
        )

    n_bins = len(res_pb["model"])

    minn_vals, counts = _vals_and_counts(res_pb["minnaert"], n_bins)
    str_vals, _ = _vals_and_counts(res_pb["strasberg"], n_bins)
    model_vals, _ = _vals_and_counts(res_pb["model"], n_bins)

    series = [
        ("minnaert", minn_vals),
        ("strasberg", str_vals),
        ("model", model_vals),
    ]

    n_series = len(series)
    x = np.arange(n_bins, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=160)
    use_log = args.log_y
    if not use_log:
        all_pos = np.concatenate([v[v > 0] for _, v in series])
        if all_pos.size > 0 and (np.nanmax(all_pos) / max(np.nanmin(all_pos), 1e-12)) > 200.0:
            use_log = True

    if args.style == "bars":
        bar_w = 0.8 / n_series
        offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_w
        for off, (key, vals) in zip(offsets, series):
            ax.bar(
                x + off,
                np.where(np.isfinite(vals), vals, 0.0),
                width=bar_w,
                color=_SERIES_COLORS[key],
                label=_SERIES_LABELS[key],
                edgecolor="white",
                linewidth=0.4,
            )
    else:
        line_styles = {
            "minnaert": {"linestyle": "--", "marker": "v", "linewidth": 1.4},
            "strasberg": {"linestyle": "--", "marker": "^", "linewidth": 1.4},
            "model": {"linestyle": "-", "marker": "o", "linewidth": 1.8},
        }
        for key, vals in series:
            style = line_styles[key]
            ax.plot(
                x,
                vals,
                color=_SERIES_COLORS[key],
                label=_SERIES_LABELS[key],
                markersize=6,
                **style,
            )

    fs = float(args.fontsize)
    fs_tick = max(6.0, fs - 1.0)
    fs_legend = max(6.0, fs - 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([""] * n_bins)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("MAPE (%)", fontsize=fs)
    ax.grid(alpha=0.25, axis="y", which="both")
    if use_log:
        ax.set_yscale("log")
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

        major_ticks = [0.1, 1.0, 10.0]
        major_labels = ["0.1%", "1%", "10%"]
        ax.yaxis.set_major_locator(FixedLocator(major_ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(major_labels))
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif args.ymax is not None:
        ax.set_ylim(0.0, float(args.ymax))
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.tick_params(axis="x", labelsize=fs_tick)
    ax.legend(fontsize=fs_legend, loc="upper left", framealpha=0.85)

    n_thumbs_drawn = 0
    if args.thumbnails:
        try:
            with (args.eval_dir / "summary.json").open(encoding="utf-8") as f:
                summary = json.load(f)
            bin_edges = np.array(summary.get("bin_edges_selected", []), dtype=np.float64)
            if bin_edges.size != n_bins + 1:
                raise ValueError(
                    f"bin_edges_selected length {bin_edges.size} != n_bins+1 {n_bins + 1}"
                )
            picked = _select_tim_thumbnail_per_bin(
                args.eval_dir / "selected_rows.csv",
                bin_edges,
                n_bins,
                mesh_root=args.mesh_root,
            )
            thumbs = _render_thumbnail_paths(
                picked,
                mesh_root=args.mesh_root,
                cache_dir=args.thumbnail_cache_dir,
                mesh_color=args.thumb_mesh_color,
                edge_color=args.thumb_edge_color,
                force=args.force_rerender,
            )
            for i, p in enumerate(thumbs):
                if p is None or not p.is_file():
                    continue
                try:
                    img = plt.imread(p)
                except Exception as e:  # noqa: BLE001
                    print(f"[thumb] read failed for {p}: {e}")
                    continue
                ab = AnnotationBbox(
                    OffsetImage(img, zoom=float(args.thumb_zoom)),
                    xy=(float(i), float(args.thumb_y_offset)),
                    xycoords=("data", "axes fraction"),
                    frameon=False,
                    box_alignment=(0.5, 1.0),
                    pad=0.0,
                    annotation_clip=False,
                )
                ax.add_artist(ab)
                n_thumbs_drawn += 1
        except Exception as e:  # noqa: BLE001
            print(f"[thumb] disabled: {e}")

    # Single-line direction caption sitting just above the thumbnail row
    # (or just below the axis when there are no thumbnails).
    direction_text = r"low non-sphericity $\rightarrow$ high non-sphericity"
    if n_thumbs_drawn > 0:
        # Reserve room at the bottom for the thumbnail row instead of relying on
        # tight_layout, which clips artists drawn at negative axes-fraction y.
        fig.subplots_adjust(bottom=0.30)
        caption_y = float(args.thumb_y_offset) + float(args.caption_gap)
        ax.text(
            0.5, caption_y,
            direction_text,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fs,
        )
    else:
        ax.text(
            0.5, -0.06, direction_text,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=fs,
        )
        fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {args.output}")
    print(f"  bins={n_bins}, n_per_bin={counts.tolist()}")
    if args.thumbnails:
        print(f"  thumbnails drawn: {n_thumbs_drawn} / {n_bins}")
    print(f"  Minnaert  MAPE per bin (%): {[round(v, 3) for v in minn_vals.tolist()]}")
    print(f"  Ellipsoid MAPE per bin (%): {[round(v, 3) for v in str_vals.tolist()]}")
    print(f"  Learned   MAPE per bin (%): {[round(v, 3) for v in model_vals.tolist()]}")


if __name__ == "__main__":
    main()

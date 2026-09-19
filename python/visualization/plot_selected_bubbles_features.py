"""Render a grid figure of selected bubbles with their 6 shape features.

For each bubble listed in ``--selected``, look up the matching row in the
combined dataset CSV, render a thumbnail with mesh color = rainbow(f/f_M),
and place a small horizontal bar plot under each thumbnail showing where
the bubble sits inside the dataset distribution for each of the six shape
features.

The six features are:
    1 - Phi_VA   (Wadell non-sphericity)
    1 - Phi_VM   (mean-curvature reach)
    1 - Phi_W    (Willmore curvature concentration; vertex estimator)
    eta_V        = V(bubble) / V(hull)
    eta_A        = A(bubble) / A(hull)
    eta_M        = M(bubble) / M(hull)
"""

from __future__ import annotations

import argparse
import math
import shlex
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_hex

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from plot_dataset_distribution import (  # noqa: E402
    DEFAULT_MESH_ROOT,
    get_or_render_transparent_thumbnail,
)

PYTHON_ROOT = _THIS_DIR.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from freq_model.analytical.minnaert_freq import minnaert_frequency_unit_volume  # noqa: E402


FEATURE_LATEX_NAMES: tuple[tuple[str, str], ...] = (
    ("non_sph_va", r"$1 - \Phi_{VA}$"),
    ("non_sph_vm", r"$1 - \Phi_{VM}$"),
    ("non_sph_w",  r"$1 - \Phi_{W}$"),
    ("eta_V",      r"$\eta_V$"),
    ("eta_A",      r"$\eta_A$"),
    ("eta_M",      r"$\eta_M$"),
)
FEATURE_KEYS: tuple[str, ...] = tuple(k for k, _ in FEATURE_LATEX_NAMES)


def parse_selected_file(path: Path) -> list[str]:
    """Return a list of mesh stems from selected_bub.txt, preserving order."""
    stems: list[str] = []
    seen: set[str] = set()
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = shlex.split(line, posix=False)
        except ValueError:
            tokens = [line]
        if not tokens:
            continue
        candidate = tokens[0].strip().strip('"').strip("'")
        if not candidate:
            continue
        stem = Path(candidate.replace("\\", "/")).stem
        if stem and stem not in seen:
            seen.add(stem)
            stems.append(stem)
    return stems


def find_csv_row(df: pd.DataFrame, stem: str) -> pd.Series | None:
    obj_name = f"{stem}.obj"
    matches = df[df["mesh_filename"] == obj_name]
    if matches.empty:
        stems = df["mesh_filename"].astype(str).map(lambda s: Path(s).stem)
        matches = df[stems == stem]
    if matches.empty:
        return None
    return matches.iloc[0]


def compute_features(row: pd.Series) -> dict[str, float]:
    surface_area = float(row["surface_area"])
    volume = float(row["volume"])
    phi_vm = float(row["Phi_VM"])
    w_vertex = float(row["W_vertex"])
    eta_v = float(row["eta_V"])
    eta_a = float(row["eta_A"])
    eta_m = float(row["eta_M"])

    if not (math.isfinite(surface_area) and surface_area > 0.0 and volume > 0.0):
        phi_va = float("nan")
    else:
        phi_va = (math.pi ** (1.0 / 3.0)) * (6.0 * volume) ** (2.0 / 3.0) / surface_area
    # Not clipped: Phi_VA outside (0, 1] indicates a defective mesh.
    if math.isfinite(phi_va) and not (0.0 < phi_va <= 1.0 + 1e-9):
        phi_va = float("nan")

    phi_w = (4.0 * math.pi / w_vertex) if (math.isfinite(w_vertex) and w_vertex > 0.0) else float("nan")

    return {
        "non_sph_va": 1.0 - phi_va if math.isfinite(phi_va) else float("nan"),
        "non_sph_vm": 1.0 - phi_vm if math.isfinite(phi_vm) else float("nan"),
        "non_sph_w": 1.0 - phi_w if math.isfinite(phi_w) else float("nan"),
        "eta_V": eta_v,
        "eta_A": eta_a,
        "eta_M": eta_m,
    }


def compute_dataset_distributions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return sorted arrays of valid feature values across the whole CSV.

    Used to compute percentile ranks for each selected bubble.
    """
    surface_area = pd.to_numeric(df["surface_area"], errors="coerce").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=np.float64)
    phi_vm = pd.to_numeric(df["Phi_VM"], errors="coerce").to_numpy(dtype=np.float64)
    w_vertex = pd.to_numeric(df["W_vertex"], errors="coerce").to_numpy(dtype=np.float64)
    eta_v = pd.to_numeric(df["eta_V"], errors="coerce").to_numpy(dtype=np.float64)
    eta_a = pd.to_numeric(df["eta_A"], errors="coerce").to_numpy(dtype=np.float64)
    eta_m = pd.to_numeric(df["eta_M"], errors="coerce").to_numpy(dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        phi_va = np.where(
            (surface_area > 0.0) & (volume > 0.0),
            (np.pi ** (1.0 / 3.0)) * (6.0 * volume) ** (2.0 / 3.0) / surface_area,
            np.nan,
        )
        # Not clipped: Phi_VA outside (0, 1] indicates a defective mesh.
        phi_va = np.where(
            np.isfinite(phi_va) & (phi_va > 0.0) & (phi_va <= 1.0 + 1e-9),
            phi_va,
            np.nan,
        )
        phi_w = np.where(w_vertex > 0.0, 4.0 * np.pi / w_vertex, np.nan)

    raw = {
        "non_sph_va": 1.0 - phi_va,
        "non_sph_vm": 1.0 - phi_vm,
        "non_sph_w": 1.0 - phi_w,
        "eta_V": eta_v,
        "eta_A": eta_a,
        "eta_M": eta_m,
    }
    out: dict[str, np.ndarray] = {}
    for key, arr in raw.items():
        finite = arr[np.isfinite(arr)]
        out[key] = np.sort(finite)
    return out


def percentile_rank(value: float, sorted_values: np.ndarray) -> float:
    """Return the percentile (0..1) of ``value`` inside the sorted distribution."""
    if not math.isfinite(value) or sorted_values.size == 0:
        return float("nan")
    n = sorted_values.size
    idx_left = int(np.searchsorted(sorted_values, value, side="left"))
    idx_right = int(np.searchsorted(sorted_values, value, side="right"))
    midpoint = 0.5 * (idx_left + idx_right)
    return float(np.clip(midpoint / n, 0.0, 1.0))


def crop_thumbnail_to_content(img: np.ndarray, *, pad_frac: float = 0.02) -> np.ndarray:
    """Crop a thumbnail to its non-transparent (or non-white) bounding box."""
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[0] < 4 or arr.shape[1] < 4:
        return arr

    if arr.shape[2] >= 4:
        mask = arr[..., 3] > 0.02
    else:
        mask = np.any(arr[..., :3] < 0.97, axis=-1)

    if not mask.any():
        return arr

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1

    h, w = arr.shape[:2]
    pad_r = max(1, int(round(pad_frac * (r1 - r0))))
    pad_c = max(1, int(round(pad_frac * (c1 - c0))))
    r0 = max(0, r0 - pad_r)
    r1 = min(h, r1 + pad_r)
    c0 = max(0, c0 - pad_c)
    c1 = min(w, c1 + pad_c)
    return arr[r0:r1, c0:c1]


def grid_dimensions(n: int, cols: int | None) -> tuple[int, int]:
    if n <= 0:
        return 1, 1
    if cols is not None and cols > 0:
        c = int(cols)
    elif n <= 3:
        c = n
    elif n <= 4:
        c = 2
    elif n <= 9:
        c = 3
    else:
        c = 4
    r = int(math.ceil(n / c))
    return r, c


def _draw_image_cell(
    ax,
    *,
    thumb_path: str | None,
    label_fontsize: float,
) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)

    if thumb_path and Path(thumb_path).is_file():
        img = plt.imread(str(thumb_path))
        img = crop_thumbnail_to_content(img)
        h, w = img.shape[:2]
        scale = max(h, w)
        half_x = w / scale
        half_y = h / scale
        ax.imshow(
            img,
            extent=(-half_x, half_x, -half_y, half_y),
            interpolation="bilinear",
            zorder=2,
        )
    else:
        ax.text(
            0.0, 0.0,
            "thumbnail\nmissing",
            ha="center", va="center",
            fontsize=label_fontsize, color="#888888",
            zorder=2,
        )


def _draw_bar_cell(
    ax,
    *,
    percentiles: list[float],
    bar_color: str,
    label_fontsize: float,
    show_labels: bool,
    track_color: str = "#dcdcdc",
) -> None:
    n = len(FEATURE_LATEX_NAMES)
    y_positions = np.arange(n)[::-1]  # top-to-bottom = first-to-last feature

    bar_height = 0.62
    ax.barh(
        y_positions,
        [1.0] * n,
        height=bar_height,
        color=track_color,
        edgecolor="none",
        zorder=1,
    )
    valid = [p if math.isfinite(p) else 0.0 for p in percentiles]
    ax.barh(
        y_positions,
        valid,
        height=bar_height,
        color=bar_color,
        edgecolor="none",
        zorder=2,
    )

    for y, p in zip(y_positions, percentiles):
        if math.isfinite(p):
            ax.plot(
                [p], [y],
                marker="o",
                markersize=max(2.5, label_fontsize * 0.32),
                markerfacecolor=bar_color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(y_positions)
    if show_labels:
        ax.set_yticklabels(
            [latex for _, latex in FEATURE_LATEX_NAMES],
            fontsize=label_fontsize,
            color="#1f2937",
        )
    else:
        ax.set_yticklabels([])
    ax.set_xticks([])
    ax.tick_params(axis="y", which="both", length=0, pad=2)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def render_grid_figure(
    *,
    items: list[dict],
    output: Path,
    cols: int | None,
    title: str,
    dpi: int,
    cell_aspect: float,
    title_fontsize: float,
    label_fontsize: float,
    image_text_gap: float,
) -> None:
    n = len(items)
    n_rows, n_cols = grid_dimensions(n, cols)

    cell_w = 2.6
    cell_h = cell_w * cell_aspect
    fig_w = n_cols * cell_w + 0.3
    fig_h = n_rows * cell_h + (0.55 if title else 0.15)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    if title:
        fig.suptitle(title, y=0.99, fontsize=title_fontsize)

    gs_outer = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        hspace=0.22,
        wspace=0.10,
        left=0.01,
        right=0.99,
        top=0.94 if title else 0.99,
        bottom=0.02,
    )

    for idx, item in enumerate(items):
        r = idx // n_cols
        c = idx % n_cols
        gs_inner = gs_outer[r, c].subgridspec(
            nrows=2, ncols=1,
            height_ratios=[3.4, 1.7],
            hspace=image_text_gap,
        )
        ax_img = fig.add_subplot(gs_inner[0, 0])
        ax_bar = fig.add_subplot(gs_inner[1, 0])

        _draw_image_cell(
            ax_img,
            thumb_path=item.get("thumbnail"),
            label_fontsize=label_fontsize,
        )
        _draw_bar_cell(
            ax_bar,
            percentiles=item["percentiles"],
            bar_color=item["color"],
            label_fontsize=label_fontsize,
            show_labels=(c == 0),
        )

    total_cells = n_rows * n_cols
    for empty_idx in range(n, total_cells):
        r = empty_idx // n_cols
        c = empty_idx % n_cols
        ax_blank = fig.add_subplot(gs_outer[r, c])
        ax_blank.set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected",
        type=Path,
        default=Path("results/dataset_distribution_3d_png/selected_bub.txt"),
        help="Plain text file with one bubble PNG/OBJ path per line.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset/bubble_gym/dataset_bubblegym_10k.csv"),
        help="Combined dataset CSV with shape features and convex-hull etas.",
    )
    parser.add_argument("--freq-col", default="frequency")
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=DEFAULT_MESH_ROOT,
        help="Directory containing per-bubble OBJ meshes.",
    )
    parser.add_argument(
        "--thumbnail-dir",
        type=Path,
        default=Path("dataset/langlois2016/thumbnail_transparent"),
        help="Cache directory for re-rendered transparent thumbnails (color-keyed).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dataset_distribution_3d_png/selected_bubbles_features.png"),
    )
    parser.add_argument("--cols", type=int, default=None, help="Number of grid columns; default auto.")
    parser.add_argument("--cmap", type=str, default="rainbow")
    parser.add_argument(
        "--color-min",
        type=float,
        default=None,
        help="Lower bound for the f/f_M -> color mapping. Defaults to the dataset min.",
    )
    parser.add_argument(
        "--color-max",
        type=float,
        default=None,
        help="Upper bound for the f/f_M -> color mapping. Defaults to the dataset max.",
    )
    parser.add_argument("--mesh-edge-color", type=str, default="#2f4a6d")
    parser.add_argument("--force-rerender", action="store_true", help="Re-render thumbnails even if cached.")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--cell-aspect", type=float, default=1.55)
    parser.add_argument("--image-text-gap", type=float, default=0.05)
    parser.add_argument(
        "--font-size",
        type=float,
        default=10.5,
        help="Base font size (pt) for axis labels. Title scales above it.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Multiplier applied on top of --font-size.",
    )
    args = parser.parse_args()

    stems = parse_selected_file(args.selected)
    if not stems:
        raise RuntimeError(f"No bubble entries parsed from {args.selected}")

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.lstrip("#").str.strip()

    distributions = compute_dataset_distributions(df)
    f_minnaert = float(minnaert_frequency_unit_volume())
    freq_hz_all = pd.to_numeric(df[args.freq_col], errors="coerce").to_numpy(dtype=np.float64)
    freq_ratio_all = freq_hz_all / f_minnaert
    valid_freq = freq_ratio_all[np.isfinite(freq_ratio_all) & (freq_ratio_all > 0.0)]
    color_min = float(args.color_min) if args.color_min is not None else float(np.nanmin(valid_freq))
    color_max = float(args.color_max) if args.color_max is not None else float(np.nanmax(valid_freq))
    if not color_max > color_min:
        raise ValueError("--color-max must be greater than --color-min")
    cmap = plt.get_cmap(args.cmap)
    norm = Normalize(vmin=color_min, vmax=color_max)

    items: list[dict] = []
    missing_csv: list[str] = []
    missing_thumb: list[str] = []

    for stem in stems:
        row = find_csv_row(df, stem)
        if row is None:
            missing_csv.append(stem)
            continue

        features = compute_features(row)
        freq_hz = float(row[args.freq_col])
        freq_ratio = freq_hz / f_minnaert if math.isfinite(freq_hz) else float("nan")
        bar_color = to_hex(cmap(norm(freq_ratio))) if math.isfinite(freq_ratio) else "#888888"

        percentiles = [percentile_rank(features[k], distributions[k]) for k in FEATURE_KEYS]

        mesh_filename = str(row["mesh_filename"])
        thumb_path: Path | None = None
        if (args.mesh_root / mesh_filename).is_file():
            thumb_path = get_or_render_transparent_thumbnail(
                mesh_filename,
                mesh_root=args.mesh_root,
                transparent_dir=args.thumbnail_dir,
                mesh_color=bar_color,
                edge_color=args.mesh_edge_color,
                force=args.force_rerender,
            )
        if thumb_path is None:
            missing_thumb.append(stem)

        items.append({
            "stem": stem,
            "features": features,
            "freq_ratio": freq_ratio,
            "color": bar_color,
            "percentiles": percentiles,
            "thumbnail": str(thumb_path) if thumb_path else None,
        })

    base = float(args.font_size) * float(args.font_scale)
    label_fontsize = base
    title_fontsize = base * 1.30

    render_grid_figure(
        items=items,
        output=args.output,
        cols=args.cols,
        title=args.title,
        dpi=args.dpi,
        cell_aspect=args.cell_aspect,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        image_text_gap=args.image_text_gap,
    )

    print(f"Plotted {len(items)} bubbles to {args.output}")
    print(f"Color range: f/f_M in [{color_min:.4g}, {color_max:.4g}]")
    if missing_csv:
        print(f"Missing in CSV: {missing_csv}")
    if missing_thumb:
        print(f"Could not render thumbnails for: {missing_thumb}")


if __name__ == "__main__":
    main()

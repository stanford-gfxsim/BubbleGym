"""Fig. 4's layout: a per-bin error panel with a row of bubble renders beneath it.

The published figure sets the axes over a strip of ten meshes, one per bin, so a
reader can see what "high nonsphericity" actually looks like. This reproduces
that arrangement for any panel built on the same ten-bin convention.

The renders come from the shipped thumbnail set,
``dataset/bubble_gym/bubble_mesh_thumbnails_100x100/<source>/<mesh>.png`` --
10,000 transparent PNGs, one per benchmark bubble -- so no mesh archive is
needed. The 400x400 folder holds only a handful of samples and is not a
substitute.

Placement follows the published ``plot_per_bin_mape_bars.py``: each render is an
``AnnotationBbox`` anchored to the *data* x of its bin and an *axes-fraction* y
below the frame. That keeps every thumbnail at its true pixel aspect and locks it
to its bin's column, whatever the figure size. ``bbox_inches="tight"`` on save
grows the canvas to include the strip, so no bottom margin needs reserving.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

THUMB_ROOT = Path("dataset/bubble_gym/bubble_mesh_thumbnails_100x100")

# Matched to the published figure: renders just under the axis label, each a
# little narrower than its bin so neighbours do not touch.
DEFAULT_ZOOM = 0.62
DEFAULT_Y_OFFSET = -0.12

# Fig. 4's panel is wide and short -- roughly 3:1 -- which is what puts the strip
# close under the axis label. A taller panel pushes the same axes-fraction offset
# further away in absolute terms, so the proportions and the offset go together.
FIG4_FIGSIZE = (11.0, 4.2)


def thumbnail_path(mesh_id: str, thumb_root: Path = THUMB_ROOT) -> Path | None:
    """``VOF/bubble.0044.15.obj`` -> the shipped PNG for that bubble, or None."""
    mesh_id = str(mesh_id)
    if "/" not in mesh_id:
        return None
    source, name = mesh_id.split("/", 1)
    p = thumb_root / source / (Path(name).stem + ".png")
    return p if p.is_file() else None


def add_thumbnail_strip(
    ax: plt.Axes,
    mesh_ids: list[str | None],
    *,
    thumb_root: Path = THUMB_ROOT,
    zoom: float = DEFAULT_ZOOM,
    y_offset: float = DEFAULT_Y_OFFSET,
) -> int:
    """Draw one render per bin under ``ax``, centred on its bin's x position.

    ``mesh_ids`` is one mesh id per bin, in bin order. Returns how many were
    drawn; a missing render leaves its slot empty rather than shifting the rest.
    """
    drawn = 0
    for i, mesh_id in enumerate(mesh_ids):
        p = thumbnail_path(mesh_id, thumb_root) if mesh_id else None
        if p is None:
            print(f"[thumb] no render for bin {i}: {mesh_id}")
            continue
        try:
            img = plt.imread(p)
        except OSError as e:
            print(f"[thumb] unreadable {p}: {e}")
            continue
        ax.add_artist(AnnotationBbox(
            OffsetImage(img, zoom=zoom),
            xy=(float(i), y_offset),
            xycoords=("data", "axes fraction"),
            frameon=False,
            box_alignment=(0.5, 1.0),
            pad=0.0,
            annotation_clip=False,
        ))
        drawn += 1
    return drawn


def bin_representatives(rows, n_bins: int) -> list[str | None]:
    """Pull the per-bin thumbnail mesh ids out of a selected_rows-style frame."""
    if "is_bin_thumbnail" not in rows.columns:
        return [None] * n_bins
    reps = rows[rows["is_bin_thumbnail"].astype(bool)]
    out: list[str | None] = []
    for b in range(n_bins):
        hit = reps[reps["bin"] == b]
        out.append(str(hit.iloc[0]["mesh_id_10k"]) if len(hit) else None)
    return out


def median_representatives(rows, n_bins: int, scalar_col: str) -> list[str | None]:
    """Fallback: the bubble nearest each bin's median of ``scalar_col``."""
    out: list[str | None] = []
    for b in range(n_bins):
        sub = rows[rows["bin"] == b]
        if not len(sub):
            out.append(None)
            continue
        v = sub[scalar_col].to_numpy(dtype=np.float64)
        out.append(str(sub.iloc[int(np.argmin(np.abs(v - np.median(v))))]["mesh_id_10k"]))
    return out

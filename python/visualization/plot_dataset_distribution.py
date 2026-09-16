"""
Plot the distribution of bubbles in a dataset CSV as non-sphericity vs. BEMPP frequency.

For each row in the dataset we:
  * Compute Wadell sphericity  Phi = pi^(1/3) * (6*V)^(2/3) / A
    and non-sphericity         non_sph = 1 - clip(Phi, 0, 1)
  * Plot (non_sphericity, frequency) as a scatter point, where `frequency`
    is the BEMPP P1-DP1 Dirichlet capacitance-based Minnaert frequency
    stored in the dataset.

On top of the scatter we overlay thumbnail callouts for a handful of bubbles
that are approximately uniformly spaced along the non-sphericity axis (in log
space), each shown inside a small circular sub-window connected to its scatter
point by a leader line.

The callouts are laid out as a "ladder" that climbs from the lower-left to the
upper-right, following the control points measured off the published figure
(see REFERENCE_LADDER), and they annotate the eight bubbles that figure calls
out (REFERENCE_CALLOUT_MESHES). Pass --thumb-layout near-curve for the older
layout that pins each callout above its own scatter point, or
--callout-meshes auto to let the sampler choose the bubbles again.

The defaults reproduce the published figure, so this runs with no arguments:

    python python/visualization/plot_dataset_distribution.py

which is equivalent to

    python python/visualization/plot_dataset_distribution.py \
        --csv dataset/bubble_gym/dataset_bubblegym_10k.csv \
        --prerendered-thumbnail-dir dataset/bubble_gym/bubble_mesh_thumbnails_100x100 \
        --callout-meshes reference --thumb-layout ladder \
        --output results/dataset_bubblegym_10k_distribution.jpg --dpi 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_hex
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Tim2016 pouring meshes, flat. Kept for the scripts that import this module;
# this figure plots the BubbleGym dataset, whose meshes live under
# DEFAULT_BUBBLE_MESH_ROOT in per-source folders.
DEFAULT_MESH_ROOT = Path(os.environ.get("TIM2016_MESH_ROOT", "dataset/tim2016/individual_bubbles"))

# The 10k benchmark's .obj archive, split into VOF/ and LBM/.
# Only needed to render new thumbnails: the .obj archive is distributed
# separately from this repo, so point BUBBLEGYM_MESH_ROOT (or --bubble-mesh-root)
# at it. Reproducing the published figure does not need it -- the callout
# thumbnails it uses ship under dataset/bubble_gym/bubble_mesh_thumbnails_400x400.
DEFAULT_BUBBLE_MESH_ROOT = Path(
    os.environ.get("BUBBLEGYM_MESH_ROOT", "dataset/bubble_gym/meshes10k")
)

# Minnaert frequency f0 for a 1 mm radius spherical air bubble in water (in Hz):
#   f0 = 1 / (2 pi) * sqrt(3 * gamma * P0 / (rho * R^2))
# with gamma=1.4, P0=101325 Pa, rho=1000 kg/m^3, R=1e-3 m.
# Using the equivalent form 4*pi*gamma*P0*(3/(4*pi))^(1/3)/rho, the factor
# 0.62035 = (3/(4 pi))^(1/3) appears to express R^2 via volume V=1 mm^3.
MINNAERT_F0_HZ = (1.0 / (2.0 * math.pi)) * math.sqrt(
    4.0 * math.pi * 1.4 * 101325.0 * 0.62035 / 1000.0 / 1.0
)


# Thumbnail-callout ladder, in axes-fraction coordinates, measured off the
# published distribution figure: eight equal-radius circles climbing from the
# lower-left to the upper-right, each leading its scatter point to the left.
# The lead shrinks toward the right (0.136 -> 0.022 of the axes width) so the
# callouts hug the rising tail instead of drifting off it, and the rungs
# accelerate upward the same way the frequency curve does.
# The eight bubbles called out in the published figure, ordered left to right
# along the non-sphericity axis. Pinning them by name keeps a rerun on exactly
# these meshes: the automatic picker targets log-uniform non-sphericities, and
# near the mode of the distribution dozens of bubbles sit within a fraction of
# a percent of each target, so a slightly different valid-row set is enough to
# swap one out.
# Stored as the dataset's own ``mesh_id`` (``<source>/<mesh_filename>``) so the
# source is never in doubt: 56 mesh filenames of the 10k dataset occur under
# both VOF/ and LBM/. Bare filenames are accepted on the command line too.
REFERENCE_CALLOUT_MESHES: tuple[str, ...] = (
    "VOF/bubble.0166.68.obj",   # 1-Phi = 0.00581, f = 5.2979 Hz
    "VOF/bubble.0270.74.obj",   # 1-Phi = 0.01035, f = 5.3002 Hz
    "VOF/bubble.0257.61.obj",   # 1-Phi = 0.01846, f = 5.3140 Hz
    "VOF/bubble.0308.42.obj",   # 1-Phi = 0.03290, f = 5.3275 Hz
    "VOF/bubble.0301.34.obj",   # 1-Phi = 0.05868, f = 5.3734 Hz
    "VOF/bubble.0280.52.obj",   # 1-Phi = 0.10449, f = 5.3980 Hz
    "VOF/bubble.0190.33.obj",   # 1-Phi = 0.18645, f = 5.6538 Hz
    "VOF/bubble.0182.28.obj",   # 1-Phi = 0.33207, f = 6.0581 Hz
)

# Lower anchor of the log y-stretch, in the plotted f/f_M units. The published
# figure's y-axis fits log(f - 5.0 Hz), i.e. an anchor of 5.0/f_M = 0.9447; 0.9
# sits a little further below the data, which compresses the flat low end
# slightly and gives the rising tail more of the axis.
DEFAULT_Y_ANCHOR = 0.9

# Defaults that reproduce the published figure without any arguments.
DEFAULT_CSV = Path("dataset/bubble_gym/dataset_bubblegym_10k.csv")
# Searched in order, first hit wins: the 400 px masters exist only for the
# callout bubbles, so everything else falls through to the 100 px library that
# ships with the repo. --thumb-ref-px rescales whichever one is found, so the
# callout circles come out the same size either way.
DEFAULT_PRERENDERED_THUMBNAIL_DIRS = [
    Path("dataset/bubble_gym/bubble_mesh_thumbnails_400x400"),
    Path("dataset/bubble_gym/bubble_mesh_thumbnails_100x100"),
]
DEFAULT_OUTPUT = Path("results/dataset_bubblegym_10k_distribution.jpg")

# Identity of this object marks "the user did not pass --callout-meshes", which
# is what lets --seed fall back to sampling without overriding an explicit pin.
# Type sizes. This figure is drawn 11 in wide and included at \columnwidth
# (3.38 in), so everything is reduced ~0.31x on the page: at the old 12 pt an
# axis label printed at under 4 pt. These land near the 8-9 pt of the caption.
FIG_LABEL_FS = 20
FIG_TICK_FS = 17
FIG_CBAR_FS = 18
FIG_NOTE_FS = 16

DEFAULT_CALLOUT_MESHES = ["selected"]

# The callout set in use, drawn with --seed 1 and then frozen. Unlike the
# reference set it spans both solvers and reaches further into the tail, because
# the sampler now runs over all 10000 rows rather than the VOF half.
SELECTED_CALLOUT_MESHES: tuple[str, ...] = (
    "VOF/bubble.0218.6.obj",     # 1-Phi = 0.00749, f = 5.2969 Hz
    "VOF/bubble.0225.71.obj",    # 1-Phi = 0.01344, f = 5.3037 Hz
    "VOF/bubble.0278.20.obj",    # 1-Phi = 0.02408, f = 5.3216 Hz
    "VOF/bubble.0222.38.obj",    # 1-Phi = 0.04380, f = 5.3433 Hz
    "LBM/bubble.0101.33.obj",    # 1-Phi = 0.07877, f = 5.4036 Hz
    "LBM/bubble.0064.115.obj",   # 1-Phi = 0.14222, f = 5.5422 Hz
    "VOF/bubble.0244.12.obj",    # 1-Phi = 0.25787, f = 5.8136 Hz
    "LBM/bubble.0009.2.obj",     # 1-Phi = 0.45803, f = 6.0591 Hz
)

REFERENCE_LADDER: tuple[tuple[float, float], ...] = (
    (0.0845, 0.2206),
    (0.2067, 0.2694),
    (0.3253, 0.3258),
    (0.4446, 0.3935),
    (0.5604, 0.4624),
    (0.6733, 0.5576),
    (0.7699, 0.7068),
    (0.8601, 0.8672),
)


def sphericity_phi_wadell(surface_area: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Wadell sphericity Phi = pi^(1/3) * (6V)^(2/3) / A. Returns NaN where invalid."""
    a = np.asarray(surface_area, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(v) & (a > 0.0) & (v > 0.0)
    phi = np.full_like(a, np.nan)
    phi[ok] = (np.pi ** (1.0 / 3.0)) * (6.0 * v[ok]) ** (2.0 / 3.0) / a[ok]
    return phi


def non_sphericity_from_phi(phi: np.ndarray) -> np.ndarray:
    """``1 - Phi``, not clipped: a sphericity outside (0, 1] means the
    underlying mesh is defective, so it stays NaN instead of being squashed.
    """
    phi = np.asarray(phi, dtype=np.float64)
    valid = np.isfinite(phi) & (phi > 0.0) & (phi <= 1.0 + 1e-9)
    return np.where(valid, 1.0 - phi, np.nan)


def find_thumbnail(thumb_dir: Path, mesh_filename: str) -> Path | None:
    stem = Path(str(mesh_filename).strip()).stem
    candidates = [
        thumb_dir / f"{stem}.png",
        thumb_dir / str(mesh_filename).replace(".obj", ".png"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_prerendered_thumbnail_index(
    thumb_dirs: Path | Sequence[Path] | None,
) -> dict[str, Path]:
    """
    Index directories of pre-rendered thumbnails as ``{key: png}``, searched
    recursively so per-source subfolders (``VOF/``, ``LBM/``) are picked up.

    Several directories may be given; they are searched in order and the first
    hit wins, which is how a small set of high-resolution masters overrides a
    full lower-resolution library without having to re-render the library.

    Every PNG is registered under both ``<parent>/<stem>`` and the bare
    ``<stem>``. The qualified key disambiguates mesh names that occur under
    more than one source (56 stems of the 10k dataset live under both VOF/ and
    LBM/); the bare key keeps flat thumbnail directories working.
    """
    if thumb_dirs is None:
        dirs: list[Path] = []
    elif isinstance(thumb_dirs, (str, Path)):
        dirs = [Path(thumb_dirs)]
    else:
        dirs = [Path(d) for d in thumb_dirs]

    index: dict[str, Path] = {}
    for thumb_dir in dirs:
        if not thumb_dir.is_dir():
            continue
        for png in sorted(thumb_dir.rglob("*.png")):
            index.setdefault(f"{png.parent.name}/{png.stem}", png)
            index.setdefault(png.stem, png)
    return index


def resolve_mesh_path(
    mesh_root: Path,
    mesh_filename: str,
    source: str | None = None,
) -> Path:
    """
    Locate a mesh under ``mesh_root``, which may hold the .obj files flat or
    split them into per-source folders (``VOF/``, ``LBM/``).

    Returns the nested path when it exists, otherwise the flat one -- so a
    caller that finds nothing still has a sensible path to name in its error.
    """
    name = str(mesh_filename).strip()
    if source:
        nested = Path(mesh_root) / str(source) / name
        if nested.is_file():
            return nested
    return Path(mesh_root) / name


def render_transparent_thumbnail(
    mesh_path: Path,
    out_png: Path,
    *,
    mesh_color: str = "#7eb6ff",
    edge_color: str = "#333333",
    window_size: tuple[int, int] = (400, 400),
    largest_component_only: bool = False,
    show_edges: bool = True,
    ambient: float = 0.0,
    diffuse: float = 1.0,
    specular: float = 0.0,
    specular_power: float = 30.0,
    smooth_shading: bool = False,
    extra_lighting: bool = False,
) -> bool:
    """Render a bubble mesh to PNG with a transparent background using PyVista.

    When ``largest_component_only`` is True the mesh is loaded via the
    plain numpy parser and reduced to its largest connected component
    before being handed to PyVista. This is the right behaviour for
    Tim2016 OBJs that occasionally store satellite shells under one
    physical_tag.

    ``show_edges`` draws the triangle wireframe, which reads as surface texture
    at the ~400 px used by the paper figures. Turn it off below roughly 200 px:
    the meshes carry thousands of triangles, so at small sizes the edges cover
    the whole surface and the bubble renders as a near-black silhouette.
    """
    try:
        import pyvista as pv
    except ImportError as e:
        print(f"[fail] pyvista not available: {e}")
        return False

    if not mesh_path.is_file():
        print(f"[skip] missing mesh: {mesh_path}")
        return False
    try:
        if largest_component_only:
            try:
                import numpy as _np
                from shape_feature.mesh_utils import (
                    largest_connected_component_mesh,
                    load_obj_mesh,
                )

                v, f = load_obj_mesh(mesh_path)
                v, f, _n_cc, _ = largest_connected_component_mesh(v, f)
                faces_pv = (
                    _np.hstack(
                        [_np.full((f.shape[0], 1), 3, dtype=_np.int64), f]
                    )
                    .astype(_np.int64)
                    .ravel()
                )
                mesh = pv.PolyData(_np.asarray(v, dtype=_np.float64), faces_pv)
            except Exception:
                # Fall back to PyVista's reader if the LCC path fails for
                # any reason (missing helper, parse error, etc.).
                mesh = pv.read(str(mesh_path))
        else:
            mesh = pv.read(str(mesh_path))
        pl = pv.Plotter(off_screen=True, window_size=list(window_size))
        pl.set_background("white")
        pl.add_mesh(
            mesh,
            color=mesh_color,
            show_edges=show_edges,
            edge_color=edge_color,
            lighting=True,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specular_power=specular_power,
            smooth_shading=smooth_shading,
        )
        if extra_lighting:
            # The default single headlight leaves concave bubbles muddy. A
            # three-point kit fills the shadowed side so shape stays readable
            # at thumbnail size.
            pl.enable_3_lights()
        pl.camera_position = "iso"
        pl.reset_camera()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(out_png), transparent_background=True)
        pl.close()
        return True
    except Exception as e:
        print(f"[fail] {mesh_path} -> {e}")
        return False


def _color_tag(hex_color: str) -> str:
    """File-safe tag derived from a '#rrggbb' color string."""
    return hex_color.lstrip("#").lower()


def get_or_render_transparent_thumbnail(
    mesh_filename: str,
    mesh_root: Path,
    transparent_dir: Path,
    *,
    mesh_color: str = "#7eb6ff",
    edge_color: str = "#333333",
    force: bool = False,
    window_size: tuple[int, int] = (400, 400),
    source: str | None = None,
) -> Path | None:
    """Cache path is keyed by (source, mesh stem, mesh color) so neither a colour
    change nor a filename shared between VOF/ and LBM/ collides."""
    stem = split_mesh_id(mesh_filename)[1]
    tag = _color_tag(mesh_color)
    out_dir = (transparent_dir / str(source)) if source else transparent_dir
    out_png = out_dir / f"{stem}__c_{tag}.png"
    if out_png.exists() and not force:
        return out_png
    mesh_path = resolve_mesh_path(mesh_root, mesh_filename, source)
    if render_transparent_thumbnail(
        mesh_path, out_png,
        mesh_color=mesh_color, edge_color=edge_color,
        window_size=window_size,
    ):
        return out_png
    return None


# Bubble names are dotted -- ``bubble.0141.41`` -- so ``Path.stem`` is wrong
# here: it would strip the ``.41``. Only a real file extension comes off.
_MESH_SUFFIXES = (".obj", ".png")


def split_mesh_id(name: str) -> tuple[str | None, str]:
    """Split a mesh id into ``(source, stem)``; source is None if unqualified."""
    text = str(name).strip().replace("\\", "/")
    head, _, tail = text.rpartition("/")
    for suffix in _MESH_SUFFIXES:
        if tail.lower().endswith(suffix):
            tail = tail[: -len(suffix)]
            break
    return (head or None), tail


def resolve_named_samples(
    mesh_filenames: np.ndarray,
    wanted: list[str],
    qualified_keys: np.ndarray | None = None,
) -> list[int]:
    """
    Map an explicit list of mesh ids onto row indices, preserving the order
    given.

    Ids may be fully qualified (``VOF/bubble.0166.68.obj``, the dataset's own
    ``mesh_id``) or bare (``bubble.0166.68.obj``); a missing or extra ``.obj``
    suffix is ignored either way. A qualified id is matched against
    ``qualified_keys`` first, so a filename that occurs under more than one
    source stays unambiguous.

    Raises if an id is absent, rather than silently dropping a callout: a
    figure quietly missing one of its labelled bubbles is worse than a crash.
    """
    by_stem: dict[str, int] = {}
    for i, name in enumerate(mesh_filenames):
        by_stem.setdefault(split_mesh_id(name)[1], i)
    by_qual: dict[str, int] = {}
    if qualified_keys is not None:
        for i, key in enumerate(qualified_keys):
            by_qual.setdefault(str(key), i)

    chosen: list[int] = []
    missing: list[str] = []
    for name in wanted:
        source, stem = split_mesh_id(name)
        i = by_qual.get(f"{source}/{stem}") if source else None
        if i is None and source is None:
            i = by_stem.get(stem)
        if i is None:
            missing.append(name)
        else:
            chosen.append(i)
    if missing:
        raise ValueError(
            "--callout-meshes named bubbles that are not in the plotted rows: "
            + ", ".join(missing)
        )
    return chosen


def pick_uniform_samples(
    non_sph: np.ndarray,
    has_thumbnail: Callable[[int], bool],
    n: int,
    *,
    rng: np.random.Generator | None = None,
    pool: int = 24,
) -> list[int]:
    """
    Pick N row indices whose non_sphericity values are approximately uniformly
    spaced on a log scale between the data's min and max. A candidate must pass
    ``has_thumbnail(idx)``, i.e. we must be able to draw a picture of it -- a
    pre-rendered PNG, or an .obj mesh we can render on demand.

    Without ``rng`` each target takes its single nearest candidate, so the
    selection is a pure function of the data. With ``rng`` the target instead
    draws uniformly from its ``pool`` nearest candidates, which yields a
    different set of bubbles per seed while keeping them at the same points
    along the axis -- the callout ladder is spaced for log-uniform targets, so
    the choices have to stay near them.

    Taking the N *nearest* rather than everything inside a fixed window is what
    makes this behave at both ends: near the mode hundreds of bubbles sit within
    a percent of a target, while the last target may have only a handful.
    """
    ns = np.asarray(non_sph, dtype=np.float64)
    valid = np.isfinite(ns) & (ns > 0.0)
    if not np.any(valid):
        return []

    lo = np.quantile(ns[valid], 0.02)
    hi = np.quantile(ns[valid], 0.98)
    lo = max(lo, np.nanmin(ns[valid]) * 1.001)
    hi = max(hi, lo * 1.1)
    targets = np.exp(np.linspace(math.log(lo), math.log(hi), n))

    chosen: list[int] = []
    used = set()
    all_idx = np.where(valid)[0]
    pool_size = max(1, pool) if rng is not None else 1
    for t in targets:
        order = np.argsort(np.abs(np.log(ns[all_idx]) - math.log(t)))
        eligible: list[int] = []
        for rank in order:
            cand = int(all_idx[rank])
            if cand in used or not has_thumbnail(cand):
                continue
            eligible.append(cand)
            if len(eligible) >= pool_size:
                break
        if eligible:
            picked = int(rng.choice(eligible)) if rng is not None else eligible[0]
            used.add(picked)
            chosen.append(picked)
    return chosen


def data_to_axes_fraction(ax: plt.Axes, x: float, y: float) -> tuple[float, float]:
    """Convert a data-space point to axes-fraction coordinates."""
    disp = ax.transData.transform((x, y))
    axf = ax.transAxes.inverted().transform(disp)
    return float(axf[0]), float(axf[1])


def apply_dark_theme(fig) -> None:
    """Re-key a finished figure for a dark slide: transparent, white chrome.

    Applied after every artist exists, so nothing has to be threaded through
    the drawing code. Everything that carries no data becomes white -- frame,
    ticks, tick labels, axis and colorbar labels, the callout circles and their
    leader lines, the legend, and the dashed f/f_M = 1 reference. The scatter
    colormap and the bubble thumbnails keep their colours: they ARE the data,
    and whitening them would erase the figure.

    Only dashed lines are recoloured, which is what distinguishes the reference
    line from the eight callout point markers -- those are Line2D too, and
    their face colour is a data colour.
    """
    import matplotlib.text as mtext
    from matplotlib.offsetbox import AnnotationBbox

    WHITE = "#ffffff"
    fig.patch.set_alpha(0.0)

    for ax in fig.axes:
        ax.patch.set_alpha(0.0)
        for spine in ax.spines.values():
            spine.set_color(WHITE)
        ax.tick_params(colors=WHITE, which="both")
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        for line in ax.lines:
            if line.get_linestyle() not in ("None", "none", "-"):
                line.set_color(WHITE)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor("none")
            leg.get_frame().set_edgecolor(WHITE)

    for artist in fig.findobj(mtext.Text):
        artist.set_color(WHITE)
        # Labels drawn on an opaque white plate (the f/f_M reference) would put
        # white lettering on a white box; the plate exists to lift the label off
        # the scatter, which the dark page already does.
        bbox = artist.get_bbox_patch()
        if bbox is not None:
            bbox.set_facecolor("none")
            bbox.set_edgecolor("none")

    for artist in fig.findobj(AnnotationBbox):
        if artist.patch is not None:
            artist.patch.set_edgecolor(WHITE)
            artist.patch.set_facecolor("none")
        if artist.arrow_patch is not None:
            artist.arrow_patch.set_color(WHITE)


def export_kwargs(path: Path, jpeg_quality: int) -> dict:
    """Extra ``savefig`` kwargs for the requested container format.

    JPEG is the deliverable format, so it gets an explicit quality; the DPI
    tag is filled in by matplotlib from the ``dpi`` passed to ``savefig``.
    """
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return dict(pil_kwargs={"quality": jpeg_quality, "optimize": True})
    return {}


def report_export(path: Path) -> None:
    """Print the written file's pixel size so the resolution floor is visible."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            dpi = im.info.get("dpi", ("?", "?"))
        print(f"Wrote {path} ({w}x{h} px, {dpi[0]} dpi)")
    except Exception:
        print(f"Wrote {path}")


class DraggableAnnotations:
    """
    Interactive drag support for a list of AnnotationBbox callouts.

    Each bbox keeps its anchored `xy` (the data point being called out) and only
    its `xybox` (placement of the thumbnail circle) moves. The connecting line
    is part of the AnnotationBbox arrow and updates automatically.

    Hotkeys:
        s  - save PNG + JSON layout
        r  - reset to auto-computed positions
        q  - close window
    """

    def __init__(self, fig, ax, annotations, mesh_names,
                 auto_positions, save_png, save_json,
                 dpi=200, jpeg_quality=95):
        self.fig = fig
        self.ax = ax
        self.annotations = annotations
        self.mesh_names = mesh_names
        self.auto_positions = [tuple(p) for p in auto_positions]
        self.save_png = Path(save_png)
        self.save_json = Path(save_json) if save_json else None
        self.dpi = int(dpi)
        self.jpeg_quality = int(jpeg_quality)
        self.active_idx: int | None = None
        self.active_offset_disp: tuple[float, float] = (0.0, 0.0)
        self._connect()
        self._print_help()

    def _print_help(self) -> None:
        print("Interactive mode:")
        print("  - drag any thumbnail circle with the mouse")
        print("  - press 's' to save PNG + JSON layout")
        print("  - press 'r' to reset to auto-computed positions")
        print("  - press 'q' (or close the window) to quit")

    def _connect(self) -> None:
        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _axbox_to_display(self, fx: float, fy: float) -> tuple[float, float]:
        xp, yp = self.ax.transAxes.transform((fx, fy))
        return float(xp), float(yp)

    def _display_to_axbox(self, xp: float, yp: float) -> tuple[float, float]:
        fx, fy = self.ax.transAxes.inverted().transform((xp, yp))
        return float(fx), float(fy)

    def _on_press(self, event) -> None:
        if event.button != 1 or event.x is None or event.y is None:
            return
        for i in range(len(self.annotations) - 1, -1, -1):
            ab = self.annotations[i]
            try:
                bbox = ab.get_window_extent(self.fig.canvas.get_renderer())
            except Exception:
                continue
            if bbox.contains(event.x, event.y):
                self.active_idx = i
                curr_disp = self._axbox_to_display(*ab.xybox)
                self.active_offset_disp = (event.x - curr_disp[0], event.y - curr_disp[1])
                return

    def _on_motion(self, event) -> None:
        if self.active_idx is None or event.x is None or event.y is None:
            return
        new_disp = (event.x - self.active_offset_disp[0],
                    event.y - self.active_offset_disp[1])
        fx, fy = self._display_to_axbox(*new_disp)
        fx = float(np.clip(fx, 0.0, 1.0))
        fy = float(np.clip(fy, 0.0, 1.05))
        self.annotations[self.active_idx].xybox = (fx, fy)
        self.fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        self.active_idx = None

    def _on_key(self, event) -> None:
        if event.key in ("s", "S"):
            self.save()
        elif event.key in ("r", "R"):
            self.reset()
        elif event.key in ("q", "Q"):
            plt.close(self.fig)

    def reset(self) -> None:
        for ab, pos in zip(self.annotations, self.auto_positions):
            ab.xybox = tuple(pos)
        self.fig.canvas.draw_idle()
        print("Reset to auto-computed positions.")

    def save(self) -> None:
        self.save_png.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.save_png, dpi=self.dpi, bbox_inches="tight",
                         **export_kwargs(self.save_png, self.jpeg_quality))
        report_export(self.save_png)
        if self.save_json is not None:
            layout = {
                "callouts": [
                    {"mesh_filename": name, "xybox": [float(ab.xybox[0]), float(ab.xybox[1])]}
                    for name, ab in zip(self.mesh_names, self.annotations)
                ],
            }
            self.save_json.parent.mkdir(parents=True, exist_ok=True)
            with self.save_json.open("w", encoding="utf-8") as f:
                json.dump(layout, f, indent=2)
            print(f"Saved layout JSON -> {self.save_json}")


def load_layout_json(path: Path) -> dict[str, tuple[float, float]]:
    """Return {mesh_filename: (fx, fy)} from a saved layout file."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, tuple[float, float]] = {}
    for row in data.get("callouts", []):
        name = str(row.get("mesh_filename", "")).strip()
        xybox = row.get("xybox")
        if name and isinstance(xybox, (list, tuple)) and len(xybox) == 2:
            out[name] = (float(xybox[0]), float(xybox[1]))
    return out


def compute_near_curve_anchors(
    ax: plt.Axes,
    xs_sel: np.ndarray,
    ys_sel: np.ndarray,
    *,
    min_offset_frac: float = 0.10,
    target_top: float = 0.93,
    target_right: float = 0.83,
    max_y_frac: float = 0.95,
) -> list[tuple[float, float]]:
    """
    For each selected data point, place its thumbnail in axes-fraction space so
    that:
      * callouts on the left (low non-sph) are pushed toward the top so they
        fill the empty upper-left region of the figure;
      * callouts near the rising curve stay closer to their own scatter point
        (so they don't collide with the data cloud on the upper-right).

    Each callout y is at least `min_offset_frac` above its data point; the
    target ceiling decreases linearly from `target_top` at x-fraction 0 to
    `target_right` at x-fraction 1.
    """
    anchors: list[tuple[float, float]] = []
    for xv, yv in zip(xs_sel, ys_sel):
        fx, fy_point = data_to_axes_fraction(ax, xv, yv)
        fx = float(np.clip(fx, 0.04, 0.96))
        target = target_top + (target_right - target_top) * float(np.clip(fx, 0.0, 1.0))
        fy = max(fy_point + min_offset_frac, target)
        fy = min(max_y_frac, fy)
        anchors.append((fx, fy))
    return anchors


def compute_ladder_anchors(
    ax: plt.Axes,
    xs_sel: np.ndarray,
    ys_sel: np.ndarray,
    *,
    control_points: tuple[tuple[float, float], ...] = REFERENCE_LADDER,
    min_offset_frac: float = 0.10,
    max_y_frac: float = 0.99,
) -> list[tuple[float, float]]:
    """
    Place the callouts on the published figure's ladder (``control_points``, in
    axes-fraction coordinates), so a run reproduces that layout instead of
    re-deriving one from whatever the data happens to look like.

    With as many callouts as control points the positions are used verbatim;
    otherwise the ladder is resampled by linear interpolation along its own
    index, which keeps the endpoints and the overall sweep for any
    ``--num-thumbnails``. A callout is still pushed up if the ladder would put
    it within ``min_offset_frac`` of the point it annotates, so a thumbnail
    never lands on top of its own scatter point.
    """
    n = len(xs_sel)
    if n == 0:
        return []
    ctrl = np.asarray(control_points, dtype=np.float64)
    if n == len(ctrl):
        rungs = ctrl
    else:
        src = np.linspace(0.0, 1.0, len(ctrl))
        dst = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])
        rungs = np.column_stack([np.interp(dst, src, ctrl[:, k]) for k in (0, 1)])

    anchors: list[tuple[float, float]] = []
    for (fx, fy), xv, yv in zip(rungs, xs_sel, ys_sel):
        _fx_point, fy_point = data_to_axes_fraction(ax, xv, yv)
        fy = max(float(fy), fy_point + min_offset_frac)
        anchors.append((float(np.clip(fx, 0.0, 1.0)), float(min(fy, max_y_frac))))
    return anchors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Dataset CSV (expects columns: mesh_filename, surface_area, volume, frequency).")
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_BUBBLE_MESH_ROOT,
                        help=("Root holding the per-bubble .obj meshes, either flat or split into "
                              "VOF/ and LBM/. Only consulted for bubbles the pre-rendered "
                              "directories do not cover; those get rendered on demand and cached "
                              "in --thumbnail-dir."))
    parser.add_argument("--thumbnail-dir", type=Path,
                        default=Path("dataset/bubble_gym/thumbnail_render_cache"),
                        help="Cache directory for thumbnails rendered on demand from --mesh-root.")
    parser.add_argument("--prerendered-thumbnail-dir", type=Path, nargs="+",
                        default=DEFAULT_PRERENDERED_THUMBNAIL_DIRS, metavar="DIR",
                        help=("Directories of pre-rendered bubble PNGs, searched recursively (so a "
                              "parent holding VOF/ and LBM/ works) and in order, first hit wins. "
                              "Used in preference to rendering from --mesh-root, which is what lets "
                              "the figure be reproduced without the .obj mesh archive."))
    parser.add_argument("--source", type=str, default=None,
                        help="Keep only rows whose 'source' column matches, e.g. VOF or LBM (default: all).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output image path; a .jpg/.jpeg suffix writes JPEG, .png writes PNG.")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality when --output has a .jpg/.jpeg suffix.")
    parser.add_argument("--num-thumbnails", type=int, default=8,
                        help="Number of thumbnail callouts to draw (ignored when --callout-meshes is given).")
    parser.add_argument("--callout-meshes", nargs="*", default=DEFAULT_CALLOUT_MESHES, metavar="MESH",
                        help=("Mesh ids to call out, in left-to-right order, either qualified "
                              "('VOF/bubble.0166.68.obj') or bare. Defaults to the token "
                              "'selected', i.e. SELECTED_CALLOUT_MESHES. Use 'reference' for "
                              "REFERENCE_CALLOUT_MESHES -- the eight bubbles of "
                              "the published figure, or 'selected' (the default) for "
                              "SELECTED_CALLOUT_MESHES. Pass 'auto' to sample log-uniformly "
                              "instead, which is what --num-thumbnails and --seed control."))
    parser.add_argument("--seed", type=int, default=None,
                        help=("Draw each callout at random from the --sample-pool candidates "
                              "nearest its target, instead of taking the nearest one. Implies "
                              "--callout-meshes auto unless meshes are named explicitly."))
    parser.add_argument("--sample-pool", type=int, default=24,
                        help="How many nearest candidates per target --seed draws from.")
    parser.add_argument("--freq-col", type=str, default="frequency",
                        help="Column to use for frequency (BEMPP P1-DP1 is stored in 'frequency').")
    parser.add_argument("--thumb-zoom", type=float, default=0.17,
                        help=("Zoom factor for thumbnail images inside callouts (smaller = smaller "
                              "windows), expressed for a --thumb-ref-px-wide source image."))
    parser.add_argument("--thumb-ref-px", type=int, default=400,
                        help=("Source-image width --thumb-zoom is calibrated for. Thumbnails of a "
                              "different resolution are rescaled to match, so the callout circles "
                              "keep the same on-page size whatever renders they come from."))
    parser.add_argument("--thumb-layout", type=str, default="ladder",
                        choices=["ladder", "near-curve"],
                        help=("'ladder' reproduces the published figure's callout arrangement "
                              "(REFERENCE_LADDER); 'near-curve' pins each callout above its own "
                              "scatter point using --thumb-target-top/--thumb-target-right."))
    parser.add_argument("--thumb-min-offset", type=float, default=0.10,
                        help="Minimum vertical offset (axes fraction) between a scatter point and its thumbnail.")
    parser.add_argument("--thumb-target-top", type=float, default=0.93,
                        help="Thumbnail y-fraction target at the left edge (fills upper-left space).")
    parser.add_argument("--thumb-target-right", type=float, default=0.83,
                        help="Thumbnail y-fraction target at the right edge (stays above the rising curve).")
    parser.add_argument("--cmap", type=str, default="magma",
                        help="Colormap for the scatter points (and colorbar).")
    parser.add_argument("--color-by", type=str, default="frequency",
                        choices=["frequency", "non_sphericity"],
                        help="Which axis to use as the color variable.")
    parser.add_argument("--mesh-color", type=str, default="#bcd4f5",
                        help="Flat light-blue face color used for all rendered mesh thumbnails.")
    parser.add_argument("--mesh-edge-color", type=str, default="#2f4a6d",
                        help="Edge color for rendered bubble meshes.")
    parser.add_argument("--y-scale", type=str, default="log", choices=["log", "power", "linear"],
                        help=("Y-axis transform: 'log' uses log(y - y0) with original-value tick labels, "
                              "'power' uses (y - y0)**p, 'linear' disables the stretch."))
    parser.add_argument("--y-stretch-low", type=float, default=0.35,
                        help=("Power exponent used when --y-scale=power (y_plot ~ (y-y0)**p). "
                              "Values <1 stretch the low end; p=1 disables the stretch."))
    parser.add_argument("--y-stretch-anchor", type=float, default=DEFAULT_Y_ANCHOR,
                        help=(f"Lower anchor y0 for the y-axis transform, in f/f_M units "
                              f"(default {DEFAULT_Y_ANCHOR:g}). Use --y-scale linear to opt out "
                              "of the stretch entirely."))
    parser.add_argument("--force-rerender", action="store_true",
                        help="Rerender transparent thumbnails even if cached copies exist.")
    parser.add_argument("--interactive", action="store_true",
                        help="Open an interactive window so thumbnail circles can be dragged; 's' saves, 'r' resets.")
    parser.add_argument("--layout-json", type=Path, default=None,
                        help=("JSON file of thumbnail positions. If it exists it is loaded as the initial "
                              "layout; 's' in interactive mode writes to it. "
                              "Defaults to <output>.layout.json."))
    parser.add_argument("--ref-y", type=float, default=1.0,
                        help=("Y value to highlight with a dashed reference line, in units of the "
                              "1 mm-radius Minnaert frequency f0. Set to NaN to disable."))
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output resolution; 300 with the default 11x6.2 in figure gives ~3300x1860 px.")
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Transparent background with white lettering, frame, callouts and "
        "reference line, for dark slides. The scatter colormap and the bubble "
        "thumbnails keep their colours. Forces PNG: JPEG has no alpha channel "
        "and would bake in a white page.",
    )
    args = parser.parse_args()

    if args.dark and args.output.suffix.lower() in (".jpg", ".jpeg"):
        args.output = args.output.with_suffix(".png")
        print(f"--dark needs an alpha channel; writing PNG instead: {args.output}")

    if args.layout_json is None:
        args.layout_json = args.output.with_suffix(args.output.suffix + ".layout.json")

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.lstrip("#").str.strip()
    if args.source is not None:
        if "source" not in df.columns:
            raise ValueError(f"{args.csv} has no 'source' column to filter with --source")
        df = df[df["source"].astype(str).str.upper() == args.source.upper()].reset_index(drop=True)
        print(f"Kept {len(df)} rows with source == {args.source!r}")
        if df.empty:
            raise RuntimeError(f"--source {args.source!r} matched no rows")
    for col in ["mesh_filename", "surface_area", "volume", args.freq_col]:
        if col not in df.columns:
            raise ValueError(f"{args.csv} missing required column: {col!r}")

    surface_area = pd.to_numeric(df["surface_area"], errors="coerce").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=np.float64)
    freq = pd.to_numeric(df[args.freq_col], errors="coerce").to_numpy(dtype=np.float64)
    mesh = df["mesh_filename"].astype(str).to_numpy()

    # Express all frequencies in units of the 1 mm-radius Minnaert frequency f0
    # so the most spherical bubble sits near y = 1.
    freq = freq / MINNAERT_F0_HZ
    print(f"Normalizing frequencies by Minnaert f0 = {MINNAERT_F0_HZ:.4f} Hz")

    phi = sphericity_phi_wadell(surface_area, volume)
    non_sph = non_sphericity_from_phi(phi)

    ok = np.isfinite(non_sph) & np.isfinite(freq) & (non_sph > 0.0) & (freq > 0.0)
    if not np.any(ok):
        raise RuntimeError("No valid rows to plot.")

    # Lookup key into the pre-rendered thumbnail index. Mesh filenames repeat
    # across sources in the 10k dataset, so qualify them with the source when
    # the CSV carries one.
    if "source" in df.columns:
        src = df["source"].astype(str).to_numpy()
        thumb_keys = np.array([f"{s}/{Path(str(m)).stem}" for m, s in zip(mesh, src)])
    else:
        thumb_keys = np.array([Path(str(m)).stem for m in mesh])

    x = non_sph[ok]
    y = freq[ok]
    mesh_ok = mesh[ok]
    thumb_keys_ok = thumb_keys[ok]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xscale("log")

    color_values = y if args.color_by == "frequency" else x
    norm = Normalize(vmin=float(np.nanmin(color_values)), vmax=float(np.nanmax(color_values)))
    cmap = plt.get_cmap(args.cmap)
    sc = ax.scatter(
        x, y,
        s=14,
        c=color_values,
        cmap=cmap,
        norm=norm,
        alpha=0.80,
        edgecolors="none",
        label=f"{x.size} bubbles",
    )

    ax.set_xlabel("Non-sphericity  $1 - \\Phi$  (Wadell)", fontsize=FIG_LABEL_FS)
    ax.set_ylabel(f"Frequency ratio $f / f_M$", fontsize=FIG_LABEL_FS)
    ax.tick_params(axis="both", labelsize=FIG_TICK_FS)
    # fig.suptitle(f"Bubble distribution in {args.csv.name}", fontsize=13, y=0.995)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    cbar_label = {
        "frequency": "Frequency ratio $f / f_M$",
        "non_sphericity": "Non-sphericity  $1 - \\Phi$",
    }[args.color_by]
    cbar = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.035)
    cbar.set_label(cbar_label, fontsize=FIG_CBAR_FS)
    cbar.ax.tick_params(labelsize=FIG_TICK_FS)

    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    yr = ymax - ymin

    y0 = float(args.y_stretch_anchor) if args.y_stretch_anchor is not None else (ymin - 0.30 * yr)
    # Guard: the log/power y transforms require y0 < min(y). If the user (or
    # default) gave a value that isn't strictly below the data, clamp it just
    # below ymin so matplotlib's affine inverse doesn't divide-by-zero. This
    # does not change y-scaling behavior when the anchor is already valid.
    _y0_max = ymin - max(1e-6, 0.01 * yr)
    if y0 > _y0_max:
        if args.y_stretch_anchor is not None:
            print(f"[warn] --y-stretch-anchor={args.y_stretch_anchor:g} >= data min f/f0={ymin:g}; "
                  f"clamping to {_y0_max:g} to keep the y transform well-defined.")
        y0 = _y0_max
    if args.y_scale == "log":
        _eps = max(1e-6, 1e-4 * yr)
        def _fwd(yv, _y0=y0, _e=_eps):
            arr = np.asarray(yv, dtype=np.float64)
            return np.log(np.maximum(arr - _y0, _e))
        def _inv(tv, _y0=y0):
            arr = np.asarray(tv, dtype=np.float64)
            return _y0 + np.exp(arr)
        ax.set_yscale("function", functions=(_fwd, _inv))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    elif args.y_scale == "power":
        p = float(np.clip(args.y_stretch_low, 0.15, 1.0))
        if p < 0.999:
            def _fwd(yv, _y0=y0, _p=p):
                arr = np.asarray(yv, dtype=np.float64)
                return np.sign(arr - _y0) * np.power(np.abs(arr - _y0), _p)
            def _inv(tv, _y0=y0, _p=p):
                arr = np.asarray(tv, dtype=np.float64)
                return _y0 + np.sign(arr) * np.power(np.abs(arr), 1.0 / _p)
            ax.set_yscale("function", functions=(_fwd, _inv))

    y_lower = ymin - 0.01 * yr
    if np.isfinite(args.ref_y):
        y_lower = min(y_lower, float(args.ref_y) - 0.012 * yr)
    ax.set_ylim(y_lower, ymax + 0.50 * yr)

    if np.isfinite(args.ref_y):
        ref_y = float(args.ref_y)
        ax.axhline(ref_y, linestyle="--", color="#b03a2e", lw=1.2, alpha=0.9, zorder=3)
        ax.annotate(
            f"$f/f_0 = {ref_y:g}$",
            xy=(0.01, ref_y),
            xycoords=("axes fraction", "data"),
            xytext=(4, 3),
            textcoords="offset points",
            ha="left", va="bottom",
            color="#b03a2e", fontsize=FIG_NOTE_FS,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        )

    prerendered = build_prerendered_thumbnail_index(args.prerendered_thumbnail_dir)
    if prerendered:
        found = [str(d) for d in args.prerendered_thumbnail_dir if Path(d).is_dir()]
        print(f"Indexed {len(prerendered)} pre-rendered thumbnails under {', '.join(found)}")

    def _prerendered_thumb(i: int) -> Path | None:
        key = str(thumb_keys_ok[i])
        return prerendered.get(key) or prerendered.get(key.rsplit("/", 1)[-1])

    def _mesh_source(i: int) -> str | None:
        return split_mesh_id(str(thumb_keys_ok[i]))[0]

    def _has_thumbnail(i: int) -> bool:
        if _prerendered_thumb(i) is not None:
            return True
        return resolve_mesh_path(args.mesh_root, mesh_ok[i], _mesh_source(i)).is_file()

    pinned_by_default = args.callout_meshes is DEFAULT_CALLOUT_MESHES
    wanted = list(args.callout_meshes or [])
    named_sets = {"selected": SELECTED_CALLOUT_MESHES, "reference": REFERENCE_CALLOUT_MESHES}
    if len(wanted) == 1 and wanted[0].lower() in named_sets:
        # --seed on its own means "show me other options", so it releases the
        # default pin -- but never one the user asked for by name.
        wanted = ([] if (args.seed is not None and pinned_by_default)
                  else list(named_sets[wanted[0].lower()]))
    elif len(wanted) == 1 and wanted[0].lower() == "auto":
        wanted = []
    if wanted:
        idx_in_ok = resolve_named_samples(mesh_ok, wanted, qualified_keys=thumb_keys_ok)
        unavailable = [str(mesh_ok[i]) for i in idx_in_ok if not _has_thumbnail(i)]
        if unavailable:
            raise ValueError("no thumbnail available for: " + ", ".join(unavailable))
    else:
        idx_in_ok = pick_uniform_samples(
            x, _has_thumbnail, args.num_thumbnails,
            rng=np.random.default_rng(args.seed) if args.seed is not None else None,
            pool=args.sample_pool,
        )
    if not idx_in_ok:
        print("[warn] No thumbnails available from --prerendered-thumbnail-dir or "
              "--mesh-root; scatter-only figure written.")

    xs_sel = np.array([x[i] for i in idx_in_ok], dtype=np.float64)
    ys_sel = np.array([y[i] for i in idx_in_ok], dtype=np.float64)
    cs_sel = np.array([color_values[i] for i in idx_in_ok], dtype=np.float64)
    selected_mesh_names = [str(mesh_ok[i]) for i in idx_in_ok]
    if selected_mesh_names:
        how = ("pinned" if wanted
               else (f"seed {args.seed}, pool {args.sample_pool}" if args.seed is not None
                     else "auto-picked"))
        print(f"Callout bubbles ({how}), left to right:")
        for k, i in enumerate(idx_in_ok):
            print(f"  {k}  {mesh_ok[i]:<22} 1-Phi={x[i]:.5f}  "
                  f"f={y[i] * MINNAERT_F0_HZ:.4f} Hz")

    fig.canvas.draw()
    if args.thumb_layout == "ladder":
        auto_anchors = compute_ladder_anchors(
            ax, xs_sel, ys_sel,
            min_offset_frac=args.thumb_min_offset,
        )
    else:
        auto_anchors = compute_near_curve_anchors(
            ax, xs_sel, ys_sel,
            min_offset_frac=args.thumb_min_offset,
            target_top=args.thumb_target_top,
            target_right=args.thumb_target_right,
        )

    saved_layout = load_layout_json(args.layout_json)
    callout_anchors = [
        saved_layout.get(name, auto_anchors[k])
        for k, name in enumerate(selected_mesh_names)
    ]
    if saved_layout:
        print(f"Loaded {len(saved_layout)} thumbnail positions from {args.layout_json}")

    annotations: list[AnnotationBbox] = []
    for (i, (fx, fy), cv) in zip(idx_in_ok, callout_anchors, cs_sel):
        mesh_name = str(mesh_ok[i])
        scatter_color = to_hex(cmap(norm(cv)))
        thumb = _prerendered_thumb(i)
        if thumb is None:
            thumb = get_or_render_transparent_thumbnail(
                mesh_name,
                mesh_root=args.mesh_root,
                transparent_dir=args.thumbnail_dir,
                mesh_color=args.mesh_color,
                edge_color=args.mesh_edge_color,
                force=args.force_rerender,
                source=_mesh_source(i),
            )
        if thumb is None:
            continue
        img = plt.imread(thumb)
        # Keep the callout circles the same on-page size no matter what the
        # source thumbnails were rendered at.
        zoom = args.thumb_zoom * (args.thumb_ref_px / float(img.shape[1]))
        imagebox = OffsetImage(img, zoom=zoom)
        imagebox.image.axes = ax
        ab = AnnotationBbox(
            imagebox,
            xy=(x[i], y[i]),
            xybox=(fx, fy),
            xycoords="data",
            boxcoords="axes fraction",
            frameon=True,
            pad=0.35,
            bboxprops=dict(
                boxstyle="circle,pad=0.30",
                edgecolor="#4a6a8f",
                facecolor="none",
                linewidth=1.1,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#4a6a8f",
                lw=0.9,
                shrinkA=2.0,
                shrinkB=2.0,
                connectionstyle="arc3,rad=0.0",
            ),
        )
        ax.add_artist(ab)
        ax.plot([x[i]], [y[i]], marker="o", ms=6.0,
                markerfacecolor=scatter_color, markeredgecolor="white", mew=1.2, zorder=5)
        annotations.append(ab)

    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    if args.interactive:
        _drag = DraggableAnnotations(
            fig, ax, annotations, selected_mesh_names,
            auto_positions=auto_anchors,
            save_png=args.output,
            save_json=args.layout_json,
            dpi=args.dpi,
            jpeg_quality=args.jpeg_quality,
        )
        plt.show()
        _drag.save()
        print(f"Done. Final layout in {args.layout_json}")
    else:
        if args.dark:
            apply_dark_theme(fig)
            fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight",
                        facecolor="none", transparent=True,
                        **export_kwargs(args.output, args.jpeg_quality))
        else:
            fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight",
                        facecolor="white",
                        **export_kwargs(args.output, args.jpeg_quality))
        plt.close(fig)
        report_export(args.output)
        print(f"  {len(annotations)} thumbnail callouts, layout={args.thumb_layout}")


if __name__ == "__main__":
    main()

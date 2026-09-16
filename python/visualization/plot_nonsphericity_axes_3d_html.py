"""Interactive 3D scatter of bubble non-sphericity axes.

Axes:
    x = 1 - Phi_VA       (Wadell non-sphericity)
    y = 1 - Phi_VM       (elongation / reach)
    z = 1 - 4*pi/W_vertex (curvature concentration, vertex Willmore estimator)

Point colour is the frequency ratio f/f_M using the rainbow colour scale.

Usage:
    python python/visualization/plot_nonsphericity_axes_3d_html.py \
        --csv dataset/bubble_gym/dataset_bubblegym_10k.csv \
        --output results/dataset_bubblegym_10k_nonsphericity_axes_3d.html
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


MINNAERT_F0_HZ = (1.0 / (2.0 * math.pi)) * math.sqrt(
    4.0 * math.pi * 1.4 * 101325.0 * 0.62035 / 1000.0 / 1.0
)


def wadell_phi(surface_area: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Wadell sphericity Phi_VA from surface area and volume."""
    a = np.asarray(surface_area, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(v) & (a > 0.0) & (v > 0.0)
    phi = np.full_like(a, np.nan)
    phi[ok] = (np.pi ** (1.0 / 3.0)) * (6.0 * v[ok]) ** (2.0 / 3.0) / a[ok]
    return phi


def numeric_column(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)


def natural_sort_key(path: Path) -> list[tuple[int, int | str]]:
    return [
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", path.name)
    ]


def make_hover_text(
    mesh_name: np.ndarray,
    freq_hz: np.ndarray,
    freq_ratio: np.ndarray,
    phi_va: np.ndarray,
    phi_vm: np.ndarray,
    phi_w_vertex: np.ndarray,
) -> list[str]:
    # Phi_VA / Phi_VM / Phi_W are omitted on purpose: the card draws them (and
    # the other five descriptors) as bars underneath, so repeating them as text
    # is redundant.
    return [
        "<br>".join(
            [
                f"<b>{mesh}</b>",
                f"f (unit volume) = {f_hz:.6g} Hz",
                f"f/f<sub>M</sub> = {fr:.6g}",
            ]
        )
        for mesh, f_hz, fr in zip(mesh_name, freq_hz, freq_ratio)
    ]


def make_lbm_mesh_hover_text(
    mesh_name: np.ndarray,
    phi_va: np.ndarray,
    phi_vm: np.ndarray,
    phi_w_vertex: np.ndarray,
    surface_area: np.ndarray,
    volume: np.ndarray,
    n_components: np.ndarray,
    freq_hz: np.ndarray,
    freq_ratio: np.ndarray,
) -> list[str]:
    hover_text: list[str] = []
    for mesh, pva, pvm, pw, area, vol, n_cc, f_hz, fr in zip(
        mesh_name,
        phi_va,
        phi_vm,
        phi_w_vertex,
        surface_area,
        volume,
        n_components,
        freq_hz,
        freq_ratio,
    ):
        lines = [
            "<b>LBM mesh</b>",
            f"{mesh}",
            f"Phi_VA = {pva:.6g}",
            f"Phi_VM = {pvm:.6g}",
            f"Phi_W(vertex) = {pw:.6g}",
            f"area(unit V) = {area:.6g}",
            f"volume = {vol:.6g}",
            f"components = {int(n_cc)}",
        ]
        if np.isfinite(f_hz) and f_hz > 0.0 and np.isfinite(fr) and fr > 0.0:
            lines.extend(
                [
                    f"BEM frequency = {f_hz:.6g} Hz",
                    f"f/f_M = {fr:.6g}",
                ]
            )
        else:
            lines.append("BEM frequency = pending")
        hover_text.append("<br>".join(lines))
    return hover_text


def make_lbm_hover_text(
    mesh_name: np.ndarray,
    freq_hz: np.ndarray,
    freq_ratio: np.ndarray,
    phi_va: np.ndarray,
    phi_vm: np.ndarray,
    phi_w_vertex: np.ndarray,
    frame_index: np.ndarray,
) -> list[str]:
    return [
        "<br>".join(
            [
                f"<b>LBM hero frame {frame}</b>",
                f"{mesh}",
                f"frequency = {f_hz:.6g} Hz",
                f"f/f_M = {fr:.6g}",
                f"Phi_VA = {pva:.6g}",
                f"Phi_VM = {pvm:.6g}",
                f"Phi_W(vertex) = {pw:.6g}",
            ]
        )
        for mesh, f_hz, fr, pva, pvm, pw, frame in zip(
            mesh_name, freq_hz, freq_ratio, phi_va, phi_vm, phi_w_vertex, frame_index
        )
    ]


def load_lbm_hero_overlay(path: Path) -> dict[str, np.ndarray | list[str]]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lstrip("#").str.strip()
    required = ["mesh_file", "f_bem_hz", "Phi_VA", "Phi_VM", "Phi_W_vertex"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Run compare_models_vs_bem_lbm_hero_bub1.py first.")

    phi_va = numeric_column(df, "Phi_VA")
    phi_vm = numeric_column(df, "Phi_VM")
    phi_w = numeric_column(df, "Phi_W_vertex")
    freq_hz = numeric_column(df, "f_bem_hz")
    freq_ratio = freq_hz / MINNAERT_F0_HZ
    frame_index = np.arange(len(df), dtype=np.int64)
    mesh_name = df["mesh_file"].astype(str).to_numpy()

    x = 1.0 - phi_va
    y = 1.0 - phi_vm
    z = 1.0 - phi_w
    ok = (
        np.isfinite(x) & (x > 0.0)
        & np.isfinite(y) & (y > 0.0)
        & np.isfinite(z) & (z > 0.0)
        & np.isfinite(freq_ratio) & (freq_ratio > 0.0)
    )
    hover_text = make_lbm_hover_text(
        mesh_name[ok],
        freq_hz[ok],
        freq_ratio[ok],
        phi_va[ok],
        phi_vm[ok],
        phi_w[ok],
        frame_index[ok],
    )
    return {
        "x": x[ok],
        "y": y[ok],
        "z": z[ok],
        "freq_hz": freq_hz[ok],
        "freq_ratio": freq_ratio[ok],
        "hover_text": hover_text,
        "frame_index": frame_index[ok],
        "mesh_name": mesh_name[ok],
    }


_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from shape_feature.nonspherical_features import (  # noqa: E402
    mesh_feature_row,
)


def load_lbm_mesh_overlay(
    mesh_dir: Path,
    *,
    cache_path: Path | None,
    recompute: bool,
    compute_missing: bool,
    limit: int | None,
) -> dict[str, np.ndarray | list[str]]:
    mesh_paths = sorted(mesh_dir.glob("*.obj"), key=natural_sort_key)
    if limit is not None:
        mesh_paths = mesh_paths[:limit]
    if not mesh_paths:
        raise FileNotFoundError(f"No OBJ meshes found in {mesh_dir}")

    names = [p.name for p in mesh_paths]
    cache_df = pd.DataFrame()
    if cache_path is not None and cache_path.exists() and not recompute:
        cache_df = pd.read_csv(cache_path)
        cache_df.columns = cache_df.columns.str.lstrip("#").str.strip()

    cached_names: set[str] = set()
    if "mesh_file" in cache_df.columns:
        cache_df = cache_df.drop_duplicates("mesh_file", keep="last")
        cached_names = set(cache_df["mesh_file"].astype(str))

    missing_paths = [p for p in mesh_paths if p.name not in cached_names]
    computed_rows: list[dict[str, float | int | str]] = []
    should_compute = recompute or cache_df.empty or (compute_missing and missing_paths)
    if should_compute:
        if recompute or cache_df.empty:
            missing_paths = mesh_paths
            cache_df = pd.DataFrame()
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            tqdm = None  # type: ignore

        iterator = missing_paths
        if tqdm is not None:
            iterator = tqdm(missing_paths, desc="LBM mesh non-sphericity", unit="mesh")
        failed = 0
        for obj_path in iterator:
            try:
                computed_rows.append(mesh_feature_row(obj_path))
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[fail] {obj_path.name}: {exc}")
        if failed:
            print(f"LBM mesh feature failures: {failed}")
    elif missing_paths:
        print(
            f"Using cached LBM mesh features only; "
            f"{len(missing_paths)} OBJ file(s) are not in {cache_path}."
        )

    if computed_rows:
        computed_df = pd.DataFrame(computed_rows)
        cache_df = pd.concat([cache_df, computed_df], ignore_index=True)
        cache_df = cache_df.drop_duplicates("mesh_file", keep="last")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_df.to_csv(cache_path, index=False)
            print(f"Wrote LBM mesh feature cache: {cache_path}")

    if cache_df.empty:
        raise RuntimeError(f"No LBM mesh features available for {mesh_dir}")

    available = cache_df[cache_df["mesh_file"].astype(str).isin(names)].copy()
    order = {name: i for i, name in enumerate(names)}
    available["_plot_order"] = available["mesh_file"].astype(str).map(order)
    available = available.sort_values("_plot_order")

    phi_va = numeric_column(available, "Phi_VA")
    phi_vm = numeric_column(available, "Phi_VM")
    phi_w = numeric_column(available, "Phi_W_vertex")
    surface_area = numeric_column(available, "surface_area")
    volume = numeric_column(available, "volume")
    n_components = numeric_column(available, "n_components")
    if "frequency_bem_galerkin_hz" in available.columns:
        freq_hz = numeric_column(available, "frequency_bem_galerkin_hz")
    else:
        freq_hz = np.full(len(available), np.nan, dtype=np.float64)
    freq_ratio = freq_hz / MINNAERT_F0_HZ
    mesh_name = available["mesh_file"].astype(str).to_numpy()
    if "mesh_path" in available.columns:
        mesh_path = available["mesh_path"].astype(str).to_numpy()
    else:
        mesh_path = np.array([str(mesh_dir / name) for name in mesh_name], dtype=object)

    x = 1.0 - phi_va
    y = 1.0 - phi_vm
    z = 1.0 - phi_w
    ok = (
        np.isfinite(x) & (x > 0.0)
        & np.isfinite(y) & (y > 0.0)
        & np.isfinite(z) & (z > 0.0)
        & np.isfinite(n_components)
    )
    invalid = ~ok
    if np.any(invalid):
        print(f"Skipping {int(invalid.sum())} invalid LBM mesh row(s):")
        for path, xv, yv, zv in zip(mesh_path[invalid], x[invalid], y[invalid], z[invalid]):
            print(f"  {path}  x={xv:.6g}, y={yv:.6g}, z={zv:.6g}")

    hover_text = make_lbm_mesh_hover_text(
        mesh_name[ok],
        phi_va[ok],
        phi_vm[ok],
        phi_w[ok],
        surface_area[ok],
        volume[ok],
        n_components[ok],
        freq_hz[ok],
        freq_ratio[ok],
    )
    return {
        "x": x[ok],
        "y": y[ok],
        "z": z[ok],
        "freq_hz": freq_hz[ok],
        "freq_ratio": freq_ratio[ok],
        "hover_text": hover_text,
        "mesh_name": mesh_name[ok],
        "mesh_path": mesh_path[ok],
        "computed_count": np.array([len(computed_rows)], dtype=np.int64),
    }


def paper_colorscale(n: int = 64) -> list[list]:
    """Matplotlib's ``rainbow`` sampled as a Plotly colorscale.

    The paper's figures were produced with matplotlib, whose ``rainbow`` runs
    violet -> blue -> cyan -> green -> yellow -> red. Plotly's own "Rainbow" is a
    different ramp (it starts magenta), so the two do not match visually; this
    samples the real thing to keep the HTML consistent with the paper.
    """
    try:
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        cmap = colormaps["rainbow"]
    except Exception:  # pragma: no cover - matplotlib always present here
        return "Rainbow"  # type: ignore[return-value]
    return [
        [i / (n - 1), to_hex(cmap(i / (n - 1)))] for i in range(n)
    ]


def transform_color_values(
    values: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    transform: str,
    gamma: float,
) -> np.ndarray:
    """Map original f/f_M values to color coordinates.

    The power transform uses gamma < 1 to stretch the lower part of the
    frequency-ratio range while keeping a monotone colorbar.
    """
    values = np.asarray(values, dtype=np.float64)
    if transform == "linear":
        return values

    span = max(vmax - vmin, np.finfo(float).eps)
    t = np.clip((values - vmin) / span, 0.0, 1.0)
    if transform == "power":
        t = t ** gamma
    else:
        raise ValueError(f"unknown color transform: {transform}")
    return vmin + span * t


def colorbar_ticks(vmin: float, vmax: float, *, linear: bool = False) -> np.ndarray:
    """Tick positions for the f/f_M colourbar.

    On a linear scale the paper uses an evenly spaced 0.05 ladder (1.05, 1.10,
    ... 1.35); the uneven candidate set below only makes sense under the power
    transform, where it compensates for the compressed low end.
    """
    if linear:
        step = 0.05
        lo = math.ceil(vmin / step) * step
        ticks = np.arange(lo, vmax + 1e-9, step)
        if ticks.size >= 3:
            return np.round(ticks, 2)
        return np.round(np.linspace(vmin, vmax, 6), 3)

    candidates = np.array([1.00, 1.02, 1.05, 1.08, 1.10, 1.15, 1.20, 1.25, 1.30])
    ticks = candidates[(candidates >= vmin) & (candidates <= vmax)]
    if ticks.size < 3:
        ticks = np.linspace(vmin, vmax, 6)
    return ticks


def thumbnail_sources(
    mesh_names: np.ndarray,
    *,
    thumbnail_dir: Path,
    html_output: Path,
    thumbnail_base_url: str | None,
) -> list[str]:
    """Return browser-loadable relative thumbnail paths for each mesh.

    ``mesh_names`` may be bare filenames (``bubble.0019.0.obj``) or dataset
    ``mesh_id`` values carrying a group folder (``VOF/bubble.0019.0.obj``). Any
    parent component is preserved, because 56 stems occur in both the VOF and
    LBM groups -- a stem-only lookup would silently pair those points with the
    wrong bubble's thumbnail.
    """
    def _rel_png(mesh: object) -> str:
        return Path(str(mesh)).with_suffix(".png").as_posix()

    if thumbnail_base_url:
        base = thumbnail_base_url.rstrip("/")
        return [f"{base}/{_rel_png(mesh)}" for mesh in mesh_names]

    out_dir = html_output.resolve().parent
    thumb_dir = thumbnail_dir.resolve()
    sources: list[str] = []
    for mesh in mesh_names:
        thumb = thumb_dir / _rel_png(mesh)
        if thumb.exists():
            rel = os.path.relpath(thumb, out_dir)
            sources.append(Path(rel).as_posix())
        else:
            sources.append("")
    return sources


def image_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    base = base_url.strip().rstrip("/")
    if not base:
        return None
    github_tree = "https://github.com/"
    if base.startswith(github_tree) and "/tree/" in base:
        rest = base[len(github_tree) :]
        owner, repo, marker_and_path = rest.split("/", 2)
        branch_and_path = marker_and_path[len("tree/") :]
        branch, path = branch_and_path.split("/", 1)
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    return base


def render_lbm_thumbnail(
    mesh_path: Path,
    out_png: Path,
    *,
    window_size: tuple[int, int],
) -> bool:
    """Render one low-resolution mesh thumbnail for the HTML hover card."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("pyvista is required for --generate-lbm-mesh-thumbnails") from exc

    if not mesh_path.is_file():
        print(f"[thumbnail skip] missing mesh: {mesh_path}")
        return False
    try:
        mesh = pv.read(mesh_path)
        plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
        plotter.set_background("white")
        plotter.add_mesh(
            mesh,
            color="#9fdcff",
            show_edges=False,
            lighting=True,
            smooth_shading=True,
        )
        plotter.camera_position = "iso"
        plotter.reset_camera()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(out_png, transparent_background=False)
        plotter.close()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[thumbnail fail] {mesh_path}: {exc}")
        return False


def lbm_mesh_thumbnail_sources(
    mesh_names: np.ndarray,
    mesh_paths: np.ndarray,
    *,
    thumbnail_dir: Path,
    html_output: Path,
    generate: bool,
    force: bool,
    size: int,
    thumbnail_base_url: str | None,
) -> list[str]:
    base = image_base_url(thumbnail_base_url)
    out_dir = html_output.resolve().parent
    thumb_dir = thumbnail_dir.resolve()
    size = max(32, int(size))
    window_size = (size, size)

    sources: list[str] = []
    written = 0
    skipped = 0
    iterator = list(zip(mesh_names, mesh_paths))
    if generate:
        try:
            from tqdm import tqdm  # type: ignore

            iterator = tqdm(iterator, desc="LBM thumbnails", unit="mesh")
        except Exception:
            pass
    for mesh_name, mesh_path in iterator:
        thumb = thumb_dir / f"{Path(str(mesh_name)).stem}.png"
        if generate and (force or not thumb.exists()):
            if render_lbm_thumbnail(Path(str(mesh_path)), thumb, window_size=window_size):
                written += 1
            else:
                skipped += 1
        elif not thumb.exists():
            skipped += 1

        if base:
            sources.append(f"{base}/{Path(str(mesh_name)).stem}.png")
        elif thumb.exists():
            rel = os.path.relpath(thumb, out_dir)
            sources.append(Path(rel).as_posix())
        else:
            sources.append("")

    if generate:
        print(f"LBM thumbnails: wrote {written}, skipped/missing {skipped}, dir={thumbnail_dir}")
    return sources


GROUP_ACCENTS = {
    "VOF": "#2f6fb5",
    "LBM": "#c2542f",
}
_DEFAULT_ACCENT = "#4a5568"


def control_panel_script(
    groups: list[dict],
    feature_labels: list[str],
    feature_titles: list[str],
    n_main_traces: int,
    default_xyz: tuple[int, int, int] = (2, 3, 4),
) -> str:
    """One draggable window holding every control.

    Replaces four separate floating panels (group toggles, axis picker, axis
    definitions, theme switch). They competed for the same corners and each had
    to re-implement its own chrome; a single window can be dragged out of the
    way as a unit and collapsed when the plot needs the room.

    Theming is CSS custom properties on :root plus a Plotly relayout, so the
    scene text, gridlines, and every panel change together.
    """
    cfg = json.dumps(
        {
            "groups": [
                {
                    "label": f"Show {g['name']} bubbles",
                    "count": int(g.get("count", 0)),
                    "traces": [int(i) for i in (g.get("extra_indices") or [g["index"]])],
                    "accent": GROUP_ACCENTS.get(str(g["name"]), _DEFAULT_ACCENT),
                }
                for g in groups
            ],
            "labels": feature_labels,
            "titles": feature_titles,
            "traces": list(range(n_main_traces)),
            "default": list(default_xyz),
            "presets": [
                {"name": "Wadell  (\u03a6)", "xyz": [2, 3, 4]},
                {"name": "Convex hull  (\u03b7)", "xyz": [5, 6, 7]},
                {"name": "Inertia + Wadell", "xyz": [0, 1, 2]},
            ],
        }
    )
    return "\nconst PANEL_CFG = " + cfg + ";\n" + r"""
(function () {
  const plot = document.getElementById('{plot_id}');
  if (!plot) return;
  const host = plot.parentNode;
  if (getComputedStyle(host).position === 'static') host.style.position = 'relative';

  /* ---------------------------------------------------------------- theme */
  const THEMES = {
    light: {
      '--bg-page': '#ffffff', '--bg-panel': 'rgba(255,255,255,0.95)',
      '--bg-border': 'rgba(60,70,90,0.16)', '--bg-shadow': 'rgba(15,23,42,0.12)',
      '--bg-text': '#334155', '--bg-strong': '#0f172a', '--bg-muted': '#64748b',
      '--bg-track': 'rgba(148,163,184,.30)', '--bg-input': '#ffffff',
      '--bg-hover': 'rgba(148,163,184,.12)', '--bg-head': 'rgba(241,245,249,.85)'
    },
    dark: {
      '--bg-page': '#0b1220', '--bg-panel': 'rgba(23,33,51,0.94)',
      '--bg-border': 'rgba(148,163,184,0.24)', '--bg-shadow': 'rgba(0,0,0,0.5)',
      '--bg-text': '#cbd5e1', '--bg-strong': '#f1f5f9', '--bg-muted': '#94a3b8',
      '--bg-track': 'rgba(148,163,184,.22)', '--bg-input': '#1e293b',
      '--bg-hover': 'rgba(148,163,184,.18)', '--bg-head': 'rgba(15,23,42,.75)'
    }
  };
  const SCENE = {
    light: { paper: '#ffffff', font: '#1f2937', grid: '#d7dde5',
             line: '#b8c2ce', back: 'rgba(0,0,0,0)' },
    dark:  { paper: '#0b1220', font: '#e2e8f0', grid: '#2c3a52',
             line: '#3b4a63', back: 'rgba(0,0,0,0)' }
  };
  let dark = false;

  function applyTheme() {
    const t = THEMES[dark ? 'dark' : 'light'];
    for (const k in t) document.documentElement.style.setProperty(k, t[k]);
    document.body.style.background = t['--bg-page'];
    const s = SCENE[dark ? 'dark' : 'light'];
    const up = { paper_bgcolor: s.paper, plot_bgcolor: s.paper,
                 'font.color': s.font, 'title.font.color': s.font };
    ['xaxis', 'yaxis', 'zaxis'].forEach(function (ax) {
      up['scene.' + ax + '.gridcolor'] = s.grid;
      up['scene.' + ax + '.zerolinecolor'] = s.grid;
      up['scene.' + ax + '.linecolor'] = s.line;
      up['scene.' + ax + '.backgroundcolor'] = s.back;
      up['scene.' + ax + '.showbackground'] = false;
      up['scene.' + ax + '.color'] = s.font;
      up['scene.' + ax + '.tickfont.color'] = s.font;
      up['scene.' + ax + '.title.font.color'] = s.font;
    });
    up['coloraxis.colorbar.tickfont.color'] = s.font;
    Plotly.relayout(plot, up);
    Plotly.restyle(plot, { 'marker.colorbar.tickfont.color': s.font,
                           'marker.colorbar.title.font.color': s.font }, [0]);
    themeBtn.textContent = dark ? '\u2600  Light mode' : '\u263e  Dark mode';
  }

  /* ---------------------------------------------------------------- window */
  const win = document.createElement('div');
  win.setAttribute('data-bubble-panel', '1');
  win.style.cssText = [
    'position:absolute', 'top:14px', 'left:14px', 'z-index:1000', 'width:250px',
    'border-radius:12px', 'overflow:hidden', 'background:var(--bg-panel)',
    'border:1px solid var(--bg-border)', 'box-shadow:0 10px 28px var(--bg-shadow)',
    'font:13px/1.35 -apple-system,Segoe UI,Roboto,Arial,sans-serif',
    'color:var(--bg-text)', 'backdrop-filter:blur(8px)', 'user-select:none'
  ].join(';');

  const head = document.createElement('div');
  head.style.cssText = [
    'display:flex', 'align-items:center', 'gap:8px', 'padding:9px 12px',
    'cursor:move', 'background:var(--bg-head)',
    'border-bottom:1px solid var(--bg-border)'
  ].join(';');
  const grip = document.createElement('span');
  grip.textContent = '\u2630';
  grip.style.cssText = 'color:var(--bg-muted);font-size:12px;';
  const title = document.createElement('span');
  title.textContent = 'Controls';
  title.style.cssText =
    'flex:1;font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--bg-muted);';
  const fold = document.createElement('button');
  fold.type = 'button';
  fold.textContent = '\u2212';
  fold.style.cssText =
    'border:0;background:transparent;cursor:pointer;color:var(--bg-muted);font:15px inherit;padding:0 4px;';
  head.appendChild(grip); head.appendChild(title); head.appendChild(fold);

  const bodyEl = document.createElement('div');
  bodyEl.style.cssText = 'padding:11px 12px 12px 12px;';
  win.appendChild(head); win.appendChild(bodyEl);

  let folded = false;
  fold.addEventListener('click', function (e) {
    e.stopPropagation();
    folded = !folded;
    bodyEl.style.display = folded ? 'none' : 'block';
    fold.textContent = folded ? '+' : '\u2212';
  });

  // Drag by the header. Pointer events + capture keep the drag alive even when
  // the cursor crosses the WebGL canvas, which swallows plain mousemove.
  let dragging = false, ox = 0, oy = 0;
  head.addEventListener('pointerdown', function (e) {
    if (e.target === fold) return;
    dragging = true;
    const r = win.getBoundingClientRect();
    const h = host.getBoundingClientRect();
    ox = e.clientX - r.left; oy = e.clientY - r.top;
    win.style.left = (r.left - h.left) + 'px';
    win.style.top = (r.top - h.top) + 'px';
    head.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  head.addEventListener('pointermove', function (e) {
    if (!dragging) return;
    const h = host.getBoundingClientRect();
    let nx = e.clientX - h.left - ox;
    let ny = e.clientY - h.top - oy;
    nx = Math.max(0, Math.min(nx, h.width - win.offsetWidth));
    ny = Math.max(0, Math.min(ny, h.height - win.offsetHeight));
    win.style.left = nx + 'px'; win.style.top = ny + 'px';
  });
  head.addEventListener('pointerup', function (e) {
    dragging = false;
    try { head.releasePointerCapture(e.pointerId); } catch (_) {}
  });

  function section(label) {
    const s = document.createElement('div');
    s.style.cssText =
      'font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--bg-muted);margin:2px 0 6px 0;';
    s.textContent = label;
    return s;
  }

  /* ------------------------------------------------------- group toggles */
  bodyEl.appendChild(section('Dataset groups'));
  PANEL_CFG.groups.forEach(function (g) {
    let on = true;
    const btn = document.createElement('button');
    btn.type = 'button'; btn.setAttribute('aria-pressed', 'true');
    const dot = document.createElement('span');
    dot.style.cssText = 'width:9px;height:9px;border-radius:50%;flex:0 0 auto;';
    const lab = document.createElement('span');
    lab.textContent = g.label; lab.style.cssText = 'flex:1;text-align:left;';
    const cnt = document.createElement('span');
    cnt.textContent = g.count.toLocaleString();
    cnt.style.cssText = 'font-variant-numeric:tabular-nums;font-size:10.5px;opacity:.65;';
    btn.appendChild(dot); btn.appendChild(lab); btn.appendChild(cnt);
    function paint() {
      btn.style.cssText = [
        'display:flex', 'align-items:center', 'gap:8px', 'width:100%',
        'padding:6px 9px', 'margin-bottom:5px', 'cursor:pointer',
        'border-radius:8px', 'font:12px inherit', 'transition:all .15s',
        'border:1px solid ' + (on ? g.accent : 'var(--bg-border)'),
        'background:' + (on ? g.accent + '18' : 'transparent'),
        'color:' + (on ? g.accent : 'var(--bg-muted)')
      ].join(';');
      dot.style.background = on ? g.accent : 'transparent';
      dot.style.boxShadow = on ? 'none' : 'inset 0 0 0 1.5px var(--bg-muted)';
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    }
    paint();
    btn.addEventListener('click', function () {
      on = !on; paint();
      Plotly.restyle(plot, { visible: on ? true : 'legendonly' }, g.traces);
    });
    bodyEl.appendChild(btn);
  });

  /* -------------------------------------------------------------- axes */
  const sel = PANEL_CFG.default.slice();
  bodyEl.appendChild(section('Axes'));
  const presets = document.createElement('div');
  presets.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px;margin-bottom:7px;';
  bodyEl.appendChild(presets);

  const selects = [];
  ['X', 'Y', 'Z'].forEach(function (axName, k) {
    const row = document.createElement('label');
    row.style.cssText = 'display:flex;align-items:center;gap:7px;margin:3px 0;font-size:12px;';
    const tag = document.createElement('span');
    tag.textContent = axName;
    tag.style.cssText = 'width:11px;font-weight:600;color:var(--bg-strong);';
    const s = document.createElement('select');
    s.style.cssText = [
      'flex:1', 'padding:3px 5px', 'font:11.5px inherit', 'cursor:pointer',
      'border:1px solid var(--bg-border)', 'border-radius:6px',
      'background:var(--bg-input)', 'color:var(--bg-text)'
    ].join(';');
    PANEL_CFG.labels.forEach(function (lab, i) {
      const o = document.createElement('option');
      o.value = String(i); o.textContent = lab; s.appendChild(o);
    });
    s.value = String(sel[k]);
    s.addEventListener('change', function () { sel[k] = parseInt(s.value, 10); applyAxes(); });
    row.appendChild(tag); row.appendChild(s);
    bodyEl.appendChild(row);
    selects.push(s);
  });

  function applyAxes() {
    const upd = { x: [], y: [], z: [] };
    PANEL_CFG.traces.forEach(function (ti) {
      const cd = plot.data[ti].customdata || [];
      const cols = [[], [], []];
      for (let r = 0; r < cd.length; r++)
        for (let k = 0; k < 3; k++) cols[k].push(Number(cd[r][2 + sel[k]]));
      upd.x.push(cols[0]); upd.y.push(cols[1]); upd.z.push(cols[2]);
    });
    Plotly.restyle(plot, upd, PANEL_CFG.traces);
    Plotly.relayout(plot, {
      'scene.xaxis.title.text': PANEL_CFG.titles[sel[0]],
      'scene.yaxis.title.text': PANEL_CFG.titles[sel[1]],
      'scene.zaxis.title.text': PANEL_CFG.titles[sel[2]]
    });
    selects.forEach(function (s, k) { s.value = String(sel[k]); });
  }

  PANEL_CFG.presets.forEach(function (pz) {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = pz.name;
    b.style.cssText = [
      'padding:3px 8px', 'font:10.5px inherit', 'cursor:pointer',
      'border:1px solid var(--bg-border)', 'border-radius:999px',
      'background:var(--bg-input)', 'color:var(--bg-muted)', 'transition:all .15s'
    ].join(';');
    b.addEventListener('mouseenter', function () { b.style.background = 'var(--bg-hover)'; });
    b.addEventListener('mouseleave', function () { b.style.background = 'var(--bg-input)'; });
    b.addEventListener('click', function () {
      sel[0] = pz.xyz[0]; sel[1] = pz.xyz[1]; sel[2] = pz.xyz[2]; applyAxes();
    });
    presets.appendChild(b);
  });

  /* ------------------------------------------------------------- display */
  bodyEl.appendChild(section('Display'));
  const themeBtn = document.createElement('button');
  themeBtn.type = 'button';
  themeBtn.style.cssText = [
    'display:block', 'width:100%', 'padding:6px 9px', 'margin-bottom:6px',
    'cursor:pointer', 'border-radius:8px', 'font:12px inherit',
    'border:1px solid var(--bg-border)', 'background:var(--bg-input)',
    'color:var(--bg-text)', 'transition:all .15s'
  ].join(';');
  themeBtn.addEventListener('click', function () { dark = !dark; applyTheme(); });
  bodyEl.appendChild(themeBtn);

  const defsBtn = document.createElement('button');
  defsBtn.type = 'button';
  defsBtn.textContent = '\u25b8  Axis definitions';
  defsBtn.style.cssText = themeBtn.style.cssText;
  const defs = document.createElement('div');
  defs.style.cssText = 'display:none;font-size:11px;line-height:1.5;color:var(--bg-text);padding:2px 2px 0 2px;';
  defs.innerHTML = `
    <div style="display:grid;grid-template-columns:auto 1fr;gap:4px 8px;align-items:baseline;">
      <b style="color:var(--bg-strong)">\u03a6<sub>VA</sub></b><span>(R<sub>V</sub>/R<sub>A</sub>)&sup2;</span>
      <b style="color:var(--bg-strong)">\u03a6<sub>VM</sub></b><span>R<sub>V</sub>/R<sub>M</sub>, R<sub>M</sub>=M/4&pi;</span>
      <b style="color:var(--bg-strong)">\u03a6<sub>W</sub></b><span>4&pi;/W<sub>vertex</sub></span>
      <b style="color:var(--bg-strong)">\u03b7</b><span>ratio to the convex hull</span>
      <b style="color:var(--bg-strong)">colour</b><span>f/f<sub>M</sub>; 1.0 = equal-volume sphere</span>
    </div>
    <div style="margin-top:7px;padding-top:6px;border-top:1px solid var(--bg-border);color:var(--bg-muted);">
      Hovering a point pops up its card, which follows the mouse.
      <b style="color:var(--bg-strong)">Click</b> to freeze the card in place so you
      can reach the <b style="color:var(--bg-strong)">Play tone</b> button and the
      radius selector. Click anywhere on empty space to un-freeze and dismiss it.
      Drag this window by its header if it is in the way.
    </div>`;
  let defsOpen = false;
  defsBtn.addEventListener('click', function () {
    defsOpen = !defsOpen;
    defs.style.display = defsOpen ? 'block' : 'none';
    defsBtn.textContent = (defsOpen ? '\u25be' : '\u25b8') + '  Axis definitions';
  });
  bodyEl.appendChild(defsBtn);
  bodyEl.appendChild(defs);

  host.appendChild(win);
  applyTheme();
})();
"""


def thumbnail_hover_script(
    feature_labels: list[str] | None = None,
    feat_lo: list[float] | None = None,
    feat_hi: list[float] | None = None,
) -> str:
    """Extra JS overlay: hover card with thumbnail, details, and feature bars.

    ``feat_lo`` / ``feat_hi`` are the 1st/99th percentiles of each descriptor
    across the plotted set; bars show where this bubble sits in that range,
    mirroring the readout beside each mesh in the paper's Fig. 2.
    """
    cfg = json.dumps(
        {
            "labels": feature_labels or [],
            "lo": [float(v) for v in (feat_lo or [])],
            "hi": [float(v) for v in (feat_hi or [])],
        }
    )
    return "\nconst BUBBLE_FEATURES = " + cfg + ";\n" + r"""
const plot = document.getElementById('{plot_id}');
const card = document.createElement('div');
card.id = 'bubble-hover-card';
card.style.position = 'fixed';
card.style.display = 'none';
card.style.zIndex = '9999';
card.style.pointerEvents = 'auto';
card.style.background = 'var(--bg-panel)';
card.style.border = '1px solid var(--bg-border)';
card.style.borderRadius = '8px';
card.style.boxShadow = '0 8px 26px var(--bg-shadow)';
card.style.padding = '8px';
card.style.maxWidth = '430px';
card.style.font = '12px -apple-system, Segoe UI, Roboto, Arial, sans-serif';
card.style.color = 'var(--bg-text)';
card.innerHTML = `
  <div style="display:flex; gap:10px; align-items:flex-start;">
    <img style="display:block; width:130px; height:130px; object-fit:contain; flex:0 0 auto;" />
    <div class="bubble-hover-text" style="line-height:1.35; padding-top:2px;"></div>
  </div>
  <div class="bubble-hover-bars" style="margin-top:8px; padding-top:8px;
       border-top:1px solid var(--bg-border);"></div>
  <span class="bubble-freeze-tag" style="display:none; align-items:center; gap:4px;
        margin-top:6px; padding:2px 8px; border-radius:999px; font-size:10px;
        background:#2f6fb518; color:#2f6fb5; border:1px solid #2f6fb5;
        cursor:pointer;">
    frozen &mdash; click this or empty space to release
  </span>
  <div class="bubble-hover-audio" style="display:flex; margin-top:8px; padding-top:8px;
       border-top:1px solid var(--bg-border); align-items:center; gap:8px;">
    <button class="bubble-play" type="button"></button>
    <label style="display:flex;align-items:center;gap:5px;font-size:11px;color:var(--bg-muted);">
      r
      <select class="bubble-radius" style="font:inherit;padding:2px 4px;border-radius:5px;
              border:1px solid var(--bg-border);background:var(--bg-input);color:var(--bg-text);">
        <option value="0.5">0.5 mm</option>
        <option value="1">1 mm</option>
        <option value="2" selected>2 mm</option>
        <option value="5">5 mm</option>
      </select>
    </label>
    <span class="bubble-pitch" style="font-size:11px;color:var(--bg-muted);
          font-variant-numeric:tabular-nums;"></span>
  </div>

`;
document.body.appendChild(card);
const img = card.querySelector('img');
const text = card.querySelector('.bubble-hover-text');
const bars = card.querySelector('.bubble-hover-bars');
const audioRow = card.querySelector('.bubble-hover-audio');
const playBtn = card.querySelector('.bubble-play');
const radiusSel = card.querySelector('.bubble-radius');
const pitchOut = card.querySelector('.bubble-pitch');
const freezeTag = card.querySelector('.bubble-freeze-tag');

let currentFreqUnit = NaN;

function stylePlay() {
    playBtn.style.cssText = [
        'display:inline-flex', 'align-items:center', 'gap:6px',
        'padding:5px 12px', 'cursor:pointer', 'border-radius:999px',
        'font:12px inherit', 'border:1px solid #2f6fb5',
        'background:#2f6fb514', 'color:#2f6fb5', 'transition:all .15s'
    ].join(';');
    playBtn.textContent = '▶  Play tone';
}
stylePlay();

// Rescale the unit-volume frequency to the pitch a bubble of the chosen
// equivalent radius would sound at: f = f_unit * V^(-1/3). The dataset's
// ~5.3 Hz unit-volume values are far below hearing; 2 mm lands the set in
// roughly 1.6-2.2 kHz.
function playbackHz() {
    const r = parseFloat(radiusSel.value) * 1e-3;
    const V = (4 / 3) * Math.PI * r * r * r;
    return currentFreqUnit * Math.pow(V, -1 / 3);
}

function refreshPitch() {
    const f = playbackHz();
    pitchOut.textContent = isFinite(f) ? f.toFixed(0) + ' Hz' : '';
}
radiusSel.addEventListener('change', refreshPitch);

// Damped harmonic oscillator driven by a Gaussian pressure pulse, integrated
// with the same Verlet scheme as the forcing demo (Q = 12).
let audioCtx = null;
async function playTone() {
    const f0 = playbackHz();
    if (!isFinite(f0) || f0 <= 0) return;
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') await audioCtx.resume();

    const sr = audioCtx.sampleRate;
    const dur = 0.45;
    const n = Math.floor(sr * dur);
    const buf = audioCtx.createBuffer(1, n, sr);
    const data = buf.getChannelData(0);

    const omega = 2 * Math.PI * f0;
    const Q = 12;
    const beta = omega / (2 * Q);
    const dt = 1 / sr;
    const tau = 1 / (8 * f0);      // pulse short enough to excite f0 broadly
    const t0 = Math.max(2 * tau, 0.001);

    let xPrev = 0, xCur = 0;
    for (let i = 0; i < n; i++) {
        const t = i * dt;
        const dx = (t - t0) / tau;
        const F = Math.exp(-dx * dx) * 1000;
        const a = F - 2 * beta * (xCur - xPrev) / dt - omega * omega * xCur;
        const xNext = 2 * xCur - xPrev + a * dt * dt;
        xPrev = xCur; xCur = xNext;
        data[i] = xNext;
    }
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(data[i]));
    if (peak > 0) for (let i = 0; i < n; i++) data[i] = data[i] / peak * 0.5;

    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    const gain = audioCtx.createGain();
    gain.gain.value = 1.0;
    src.connect(gain).connect(audioCtx.destination);
    src.start();
}
playBtn.addEventListener('click', function (e) { e.stopPropagation(); playTone(); });

// One row per descriptor: label, grey track, filled portion, numeric value.
// Fill fraction is the value's position between the 1st and 99th percentile of
// that descriptor across the plotted set (clamped), matching Fig. 2's readout.
function renderBars(values, accent) {
    if (!BUBBLE_FEATURES.labels.length || !values || !values.length) {
        bars.innerHTML = '';
        bars.style.display = 'none';
        return;
    }
    bars.style.display = 'block';
    let html = '<div style="display:grid;grid-template-columns:auto 1fr auto;'
             + 'gap:3px 7px;align-items:center;font-size:10.5px;">';
    for (let i = 0; i < BUBBLE_FEATURES.labels.length; i++) {
        const v = Number(values[i]);
        const lo = BUBBLE_FEATURES.lo[i], hi = BUBBLE_FEATURES.hi[i];
        let frac = (hi > lo) ? (v - lo) / (hi - lo) : 0;
        if (!isFinite(frac)) frac = 0;
        frac = Math.max(0, Math.min(1, frac));
        html += '<span style="color:var(--bg-text);white-space:nowrap;">'
              + BUBBLE_FEATURES.labels[i] + '</span>'
              + '<span style="display:block;height:7px;border-radius:4px;'
              + 'background:var(--bg-track);overflow:hidden;">'
              + '<span style="display:block;height:100%;width:' + (frac * 100).toFixed(1)
              + '%;background:' + accent + ';border-radius:4px;"></span></span>'
              + '<span style="color:var(--bg-muted);font-variant-numeric:tabular-nums;'
              + 'white-space:nowrap;">' + (isFinite(v) ? v.toFixed(3) : '--') + '</span>';
    }
    bars.innerHTML = html + '</div>';
}

// Plotly's plotly_hover does NOT carry a DOM event for gl3d traces, so the
// cursor position has to be tracked separately -- without this the card keeps
// its static position and lands below the fold.
let lastMouse = { x: 0, y: 0 };
document.addEventListener('mousemove', function (e) {
    lastMouse.x = e.clientX;
    lastMouse.y = e.clientY;
}, { passive: true });

function placeCard(evt) {
    const padX = 18;
    const padY = 24;
    const cx = (evt && evt.clientX != null) ? evt.clientX : lastMouse.x;
    const cy = (evt && evt.clientY != null) ? evt.clientY : lastMouse.y;
    const rect = card.getBoundingClientRect();
    const width = rect.width || 430;
    const height = rect.height || 300;
    let left = cx + padX;
    let top = cy + padY;
    if (left + width > window.innerWidth) left = cx - width - padX;
    if (top + height > window.innerHeight) top = cy - height - padY;
    // Clamp into the viewport so a tall card never runs off an edge.
    left = Math.min(Math.max(4, left), Math.max(4, window.innerWidth - width - 4));
    top = Math.min(Math.max(4, top), Math.max(4, window.innerHeight - height - 4));
    card.style.left = left + 'px';
    card.style.top = top + 'px';
}

function fillCard(point) {
    const cd = point.customdata;
    const thumbSrc = cd[0] || '';
    if (thumbSrc) { img.src = thumbSrc; img.style.display = 'block'; }
    else { img.removeAttribute('src'); img.style.display = 'none'; }
    text.innerHTML = cd[1] || '';
    const accent = (point.data && point.data.marker && point.data.marker.line
                    && point.data.marker.line.color) || '#2f6fb5';
    renderBars(cd.slice(2, 2 + BUBBLE_FEATURES.labels.length), accent);
    currentFreqUnit = Number(cd[2 + BUBBLE_FEATURES.labels.length]);
    refreshPitch();
}

// The card has to be reachable by the mouse for its Play button to work, but a
// card that grabs pointer events the moment it appears would sit under the
// cursor and fire plotly_unhover immediately. Standard hoverable-tooltip
// pattern: hide on a short delay after unhover, and cancel that timer if the
// pointer lands on the card. (plotly_click is not usable here -- gl3d treats a
// click as the start of an orbit drag and never emits the event.)
let hideTimer = null;
let overCard = false;
let frozen = false;          // click-to-freeze, so Play is reachable
let lastPoint = null;        // most recent hovered point, for the freeze click

function cancelHide() { if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; } }
function scheduleHide(ms) {
    cancelHide();
    // The pointer entering the card fires mouseenter BEFORE Plotly's
    // unhover, so a bare cancel would be undone a moment later; the flag is
    // what actually keeps the card alive while it is being used.
    hideTimer = setTimeout(function () {
        if (!overCard && !frozen) card.style.display = 'none';
    }, ms);
}

card.addEventListener('mouseenter', function () { overCard = true; cancelHide(); });
card.addEventListener('mouseleave', function () {
    overCard = false;
    if (!frozen) scheduleHide(120);
});

freezeTag.addEventListener('click', function (e) {
    e.stopPropagation();
    setFrozen(false);
    card.style.display = 'none';
});

function setFrozen(on) {
    frozen = on;
    freezeTag.style.display = on ? 'inline-flex' : 'none';
    card.style.outline = on ? '2px solid #2f6fb5' : 'none';
    if (on) cancelHide();
}

// Plotly never emits plotly_click for gl3d traces -- it treats the press as the
// start of an orbit drag -- so freezing is driven off a plain DOM click plus the
// last point reported by plotly_hover. Listening on the document (not just the
// plot) means a click anywhere outside the card releases it.
document.addEventListener('click', function (e) {
    // Clicks inside the card (Play, radius) or the control window are never
    // freeze/unfreeze gestures.
    if (card.contains(e.target)) return;
    if (e.target.closest && e.target.closest('[data-bubble-panel]')) return;

    // Release takes priority: while frozen the card is still displayed, so
    // testing "is a card showing?" first would re-freeze on every click and
    // there would be no way out.
    if (frozen) {
        setFrozen(false);
        card.style.display = 'none';
        return;
    }
    if (lastPoint && card.style.display === 'block') setFrozen(true);
});

plot.on('plotly_hover', function(data) {
    const point = data.points && data.points[0];
    if (!point || !point.customdata) return;
    lastPoint = point;
    if (frozen) return;
    cancelHide();
    fillCard(point);
    card.style.display = 'block';
    // Place after the card is visible so getBoundingClientRect sees real
    // dimensions; data.event is absent on gl3d, hence the tracked-cursor fallback.
    placeCard(data.event);
});

plot.on('plotly_unhover', function() {
    if (frozen) return;
    // Long enough to walk the pointer from the marker onto the card.
    scheduleHide(450);
});

plot.addEventListener('mousemove', function(evt) {
    // A frozen card must stay put: this handler is what makes the card trail
    // the cursor, and while it runs the Play button slides away from any click.
    if (frozen) return;
    if (card.style.display !== 'none') {
        placeCard(evt);
    }
});
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("dataset/bubble_gym/dataset_bubblegym_10k.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dataset_bubblegym_10k_nonsphericity_axes_3d.html"),
    )
    parser.add_argument("--freq-col", default="frequency")
    parser.add_argument(
        "--axis-scale",
        choices=["log", "linear"],
        default="linear",
        help=(
            "Linear by default, matching plot_dataset_distribution_3d_png.py "
            "(the paper's figure generator), which plots the raw descriptor "
            "values with no transform. Log compresses the dense near-spherical "
            "cluster into a spike and requires every coordinate to be strictly "
            "positive, which silently drops rows."
        ),
    )
    parser.add_argument("--marker-size", type=float, default=3.0)
    parser.add_argument("--opacity", type=float, default=0.82)
    parser.add_argument(
        "--color-transform",
        choices=["linear", "power"],
        default="linear",
        help="Nonlinear color mapping. 'power' stretches the lower f/f_M range.",
    )
    parser.add_argument(
        "--color-gamma",
        type=float,
        default=0.45,
        help="Power color exponent. Values < 1 stretch colours near the low end.",
    )
    parser.add_argument("--color-min", type=float, default=None, help="Lower colorbar f/f_M bound.")
    parser.add_argument("--color-max", type=float, default=None, help="Upper colorbar f/f_M bound.")
    parser.add_argument(
        "--thumbnail-dir",
        type=Path,
        default=Path("dataset/bubble_gym/bubble_mesh_thumbnails_400x400/VOF"),
        help="Directory containing thumbnails named like the mesh stem plus .png.",
    )
    parser.add_argument(
        "--thumbnail-base-url",
        default="",
        help=(
            "Remote base URL for thumbnails. If set, each image is loaded as "
            "<base>/<mesh-stem>.png. Use an empty string to fall back to --thumbnail-dir."
        ),
    )
    parser.add_argument(
        "--lbm-hero-csv",
        type=Path,
        default=None,
        help=(
            "Optional lbm_hero_bub1 model_vs_bem_freq.csv to overlay in the same "
            "3D feature space using larger diamond markers."
        ),
    )
    parser.add_argument(
        "--lbm-thumbnail-base-url",
        default="",
        help=(
            "Remote base URL for LBM hero thumbnails. Each image is loaded as "
            "<base>/<mesh-stem>.png. Use an empty string to disable LBM thumbnails."
        ),
    )
    parser.add_argument("--lbm-marker-size", type=float, default=2.0)
    parser.add_argument(
        "--lbm-mesh-dir",
        type=Path,
        default=None,
        help="Optional directory of LBM OBJ bubble meshes to compute and overlay as diamonds.",
    )
    parser.add_argument(
        "--lbm-mesh-feature-cache",
        type=Path,
        default=Path("results/lbm_exhale_nonsphericity_features.csv"),
        help="CSV cache for non-sphericity features computed from --lbm-mesh-dir.",
    )
    parser.add_argument(
        "--lbm-mesh-recompute",
        action="store_true",
        help="Ignore the LBM mesh feature cache and recompute all selected meshes.",
    )
    parser.add_argument(
        "--lbm-mesh-compute-missing",
        action="store_true",
        help="Compute OBJ meshes missing from --lbm-mesh-feature-cache.",
    )
    parser.add_argument(
        "--lbm-mesh-limit",
        type=int,
        default=None,
        help="Process only the first N LBM meshes after natural filename sorting.",
    )
    parser.add_argument("--lbm-mesh-color", default="#19a0ff")
    parser.add_argument("--lbm-mesh-name", default="lbm_exhale")
    parser.add_argument(
        "--lbm-mesh-thumbnail-dir",
        type=Path,
        default=Path("results/lbm_exhale_thumbnails"),
        help="Directory for local LBM mesh thumbnails named <mesh-stem>.png.",
    )
    parser.add_argument(
        "--lbm-mesh-thumbnail-base-url",
        default="",
        help=(
            "Remote base URL for LBM mesh thumbnails. GitHub tree URLs are converted "
            "to raw image URLs for browser display. Use an empty string for local paths."
        ),
    )
    parser.add_argument(
        "--generate-lbm-mesh-thumbnails",
        action="store_true",
        help="Generate missing low-resolution thumbnails for plotted LBM mesh points.",
    )
    parser.add_argument(
        "--lbm-mesh-thumbnail-force",
        action="store_true",
        help="Regenerate LBM mesh thumbnails even if PNG files already exist.",
    )
    parser.add_argument(
        "--lbm-mesh-thumbnail-size",
        type=int,
        default=128,
        help="Square thumbnail resolution in pixels for generated LBM mesh PNGs.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.lstrip("#").str.strip()

    required = ["mesh_filename", "surface_area", "volume", "Phi_VM", "W_vertex", args.freq_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{args.csv} missing columns: {missing}")

    surface_area = numeric_column(df, "surface_area")
    volume = numeric_column(df, "volume")
    phi_va = wadell_phi(surface_area, volume)
    phi_vm = numeric_column(df, "Phi_VM")
    w_vertex = numeric_column(df, "W_vertex")
    phi_w_vertex = 4.0 * np.pi / w_vertex

    x = 1.0 - phi_va
    y = 1.0 - phi_vm
    z = 1.0 - phi_w_vertex

    freq_hz = numeric_column(df, args.freq_col)
    freq_ratio = freq_hz / MINNAERT_F0_HZ

    # A log axis cannot show a non-positive coordinate, so those rows have to go
    # when --axis-scale log. On a linear axis they are perfectly plottable, and
    # the paper's generator (plot_dataset_distribution_3d_png.py) keeps them --
    # it masks on finiteness alone. Requiring positivity unconditionally would
    # silently drop bubbles from a figure that claims to show the whole set.
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    ok &= np.isfinite(freq_ratio) & (freq_ratio > 0.0)
    if args.axis_scale == "log":
        ok &= (x > 0.0) & (y > 0.0) & (z > 0.0)
    if not np.any(ok):
        raise RuntimeError("No finite points to plot.")

    mesh_name = df["mesh_filename"].astype(str).to_numpy()[ok]
    # Thumbnails are keyed by mesh_id ("VOF/bubble.0019.0.obj") when the dataset
    # provides it, so the two groups' colliding filenames stay distinct. Hover
    # labels keep the bare filename.
    thumb_key = (
        df["mesh_id"].astype(str).to_numpy()[ok]
        if "mesh_id" in df.columns
        else mesh_name
    )
    thumb_src = thumbnail_sources(
        thumb_key,
        thumbnail_dir=args.thumbnail_dir,
        html_output=args.output,
        thumbnail_base_url=args.thumbnail_base_url.strip() or None,
    )
    hover_text = make_hover_text(
        mesh_name,
        freq_hz[ok],
        freq_ratio[ok],
        phi_va[ok],
        phi_vm[ok],
        phi_w_vertex[ok],
    )

    lbm_overlay = load_lbm_hero_overlay(args.lbm_hero_csv) if args.lbm_hero_csv is not None else None
    lbm_mesh_overlay = (
        load_lbm_mesh_overlay(
            args.lbm_mesh_dir,
            cache_path=args.lbm_mesh_feature_cache,
            recompute=args.lbm_mesh_recompute,
            compute_missing=args.lbm_mesh_compute_missing,
            limit=args.lbm_mesh_limit,
        )
        if args.lbm_mesh_dir is not None
        else None
    )
    color_source = [freq_ratio[ok]]
    lbm_mesh_trace_indices: list[int] = []
    if lbm_overlay is not None:
        color_source.append(np.asarray(lbm_overlay["freq_ratio"], dtype=np.float64))
    if lbm_mesh_overlay is not None:
        lbm_mesh_freq_ratio_all = np.asarray(lbm_mesh_overlay["freq_ratio"], dtype=np.float64)
        lbm_mesh_freq_ok_all = np.isfinite(lbm_mesh_freq_ratio_all) & (lbm_mesh_freq_ratio_all > 0.0)
        if np.any(lbm_mesh_freq_ok_all):
            color_source.append(lbm_mesh_freq_ratio_all[lbm_mesh_freq_ok_all])
    color_source_arr = np.concatenate(color_source)
    color_min = float(np.nanmin(color_source_arr)) if args.color_min is None else float(args.color_min)
    color_max = float(np.nanmax(color_source_arr)) if args.color_max is None else float(args.color_max)
    if not color_max > color_min:
        raise ValueError("--color-max must be greater than --color-min")
    gamma = float(np.clip(args.color_gamma, 0.05, 5.0))
    color_values = transform_color_values(
        freq_ratio[ok],
        vmin=color_min,
        vmax=color_max,
        transform=args.color_transform,
        gamma=gamma,
    )
    tick_values = colorbar_ticks(color_min, color_max, linear=(args.color_transform == "linear"))
    tick_values_transformed = transform_color_values(
        tick_values,
        vmin=color_min,
        vmax=color_max,
        transform=args.color_transform,
        gamma=gamma,
    )
    colorbar_title = "Frequency ratio f/f<sub>M</sub>"
    if args.color_transform != "linear":
        colorbar_title += f"<br>({args.color_transform}, gamma={gamma:g})"

    cmin_t = transform_color_values(
        np.array([color_min]), vmin=color_min, vmax=color_max,
        transform=args.color_transform, gamma=gamma,
    )[0]
    cmax_t = transform_color_values(
        np.array([color_max]), vmin=color_min, vmax=color_max,
        transform=args.color_transform, gamma=gamma,
    )[0]

    # Split the benchmark into its solver groups so each can be toggled on its
    # own. Both share one colour scale (frequency is comparable across solvers),
    # so only the first trace carries the colourbar.
    thumb_arr = np.asarray(thumb_src, dtype=object)
    hover_arr = np.asarray(hover_text, dtype=object)

    # The eight descriptor components of Eq. 11, in the order the model
    # consumes them, so the hover card can draw the same bar readout as the
    # paper's Fig. 2 inset.
    def _col(name: str) -> np.ndarray:
        return numeric_column(df, name)[ok] if name in df.columns else np.full(int(ok.sum()), np.nan)

    _i00, _i11, _i22 = _col("i00"), _col("i11"), _col("i22")
    with np.errstate(divide="ignore", invalid="ignore"):
        i11_r = np.where(_i00 > 0, _i11 / _i00, np.nan)
        i22_r = np.where(_i00 > 0, _i22 / _i00, np.nan)
    feature_specs = [
        ("I₁₁/I₀₀", i11_r),
        ("I₂₂/I₀₀", i22_r),
        ("1−Φ_VA", x[ok]),
        ("1−Φ_VM", y[ok]),
        ("1−Φ_W", z[ok]),
        ("η_V", _col("eta_V")),
        ("η_A", _col("eta_A")),
        ("η_M", _col("eta_M")),
    ]
    feature_labels = [n for n, _ in feature_specs]
    feature_titles = [
        "I<sub>11</sub>/I<sub>00</sub>", "I<sub>22</sub>/I<sub>00</sub>",
        "1 − Φ<sub>VA</sub>", "1 − Φ<sub>VM</sub>", "1 − Φ<sub>W</sub>",
        "η<sub>V</sub>", "η<sub>A</sub>", "η<sub>M</sub>",
    ]
    feature_matrix = np.column_stack([v for _, v in feature_specs]).astype(np.float64)
    # Bars are drawn as a fraction of each feature's observed range across the
    # whole dataset, so a bubble's profile is readable in isolation.
    feat_lo = np.nanpercentile(feature_matrix, 1.0, axis=0)
    feat_hi = np.nanpercentile(feature_matrix, 99.0, axis=0)
    feat_hi = np.where(feat_hi > feat_lo, feat_hi, feat_lo + 1e-12)
    if "source" in df.columns:
        src_arr = df["source"].astype(str).to_numpy()[ok]
        order = [g for g in ("VOF", "LBM") if (src_arr == g).any()]
        order += [g for g in pd.unique(src_arr) if g not in order]
        group_masks = [(g, src_arr == g) for g in order]
    else:
        group_masks = [("bubbles", np.ones(int(ok.sum()), dtype=bool))]

    group_symbols = {"VOF": "circle", "LBM": "diamond"}
    fig = go.Figure()
    main_group_traces: list[dict] = []
    for gi, (gname, gmask) in enumerate(group_masks):
        marker = dict(
            size=args.marker_size,
            color=color_values[gmask],
            colorscale=paper_colorscale(),
            opacity=args.opacity,
            cmin=cmin_t,
            cmax=cmax_t,
            symbol=group_symbols.get(gname, "circle"),
            showscale=(gi == 0),
        )
        if gi == 0:
            marker["colorbar"] = dict(
                title=colorbar_title,
                tickmode="array",
                tickvals=tick_values_transformed,
                ticktext=[f"{v:.2f}" for v in tick_values],
                len=0.72,
                thickness=16,
                outlinewidth=0,
            )
        fig.add_trace(
            go.Scatter3d(
                x=x[ok][gmask],
                y=y[ok][gmask],
                z=z[ok][gmask],
                mode="markers",
                name=f"{gname} ({int(gmask.sum())})",
                text=hover_arr[gmask],
                customdata=np.column_stack(
                    [
                        thumb_arr[gmask],
                        hover_arr[gmask],
                        np.round(feature_matrix[gmask], 5).astype(object),
                        # trailing column: unit-volume frequency, which the
                        # card's audio preview rescales to an audible pitch.
                        np.round(freq_hz[ok][gmask], 6).astype(object),
                    ]
                ),
                hoverinfo="none",
                showlegend=False,
                marker=marker,
            )
        )
        main_group_traces.append(
            {"name": gname, "index": len(fig.data) - 1, "count": int(gmask.sum())}
        )
    if lbm_overlay is not None:
        lbm_freq_ratio = np.asarray(lbm_overlay["freq_ratio"], dtype=np.float64)
        lbm_color_values = transform_color_values(
            lbm_freq_ratio,
            vmin=color_min,
            vmax=color_max,
            transform=args.color_transform,
            gamma=gamma,
        )
        lbm_hover_text = list(lbm_overlay["hover_text"])
        lbm_thumb_base = args.lbm_thumbnail_base_url.strip().rstrip("/")
        if lbm_thumb_base:
            lbm_thumb_src = [
                f"{lbm_thumb_base}/{Path(str(name)).stem}.png"
                for name in np.asarray(lbm_overlay["mesh_name"], dtype=object)
            ]
        else:
            lbm_thumb_src = [""] * len(lbm_hover_text)
        fig.add_trace(
            go.Scatter3d(
                x=np.asarray(lbm_overlay["x"], dtype=np.float64),
                y=np.asarray(lbm_overlay["y"], dtype=np.float64),
                z=np.asarray(lbm_overlay["z"], dtype=np.float64),
                mode="markers",
                name="lbm_hero_bub1",
                text=lbm_hover_text,
                customdata=np.column_stack(
                    [
                        np.asarray(lbm_thumb_src, dtype=object),
                        np.asarray(lbm_hover_text, dtype=object),
                    ]
                ),
                hoverinfo="none",
                marker=dict(
                    size=args.lbm_marker_size,
                    symbol="diamond",
                    color=lbm_color_values,
                    colorscale=paper_colorscale(),
                    opacity=0.98,
                    cmin=transform_color_values(
                        np.array([color_min]),
                        vmin=color_min,
                        vmax=color_max,
                        transform=args.color_transform,
                        gamma=gamma,
                    )[0],
                    cmax=transform_color_values(
                        np.array([color_max]),
                        vmin=color_min,
                        vmax=color_max,
                        transform=args.color_transform,
                        gamma=gamma,
                    )[0],
                    showscale=False,
                    line=dict(width=0),
                ),
            )
        )
    if lbm_mesh_overlay is not None:
        lbm_mesh_hover_text = list(lbm_mesh_overlay["hover_text"])
        lbm_mesh_thumb_src = lbm_mesh_thumbnail_sources(
            np.asarray(lbm_mesh_overlay["mesh_name"], dtype=object),
            np.asarray(lbm_mesh_overlay["mesh_path"], dtype=object),
            thumbnail_dir=args.lbm_mesh_thumbnail_dir,
            html_output=args.output,
            generate=args.generate_lbm_mesh_thumbnails,
            force=args.lbm_mesh_thumbnail_force,
            size=args.lbm_mesh_thumbnail_size,
            thumbnail_base_url=args.lbm_mesh_thumbnail_base_url,
        )
        lbm_mesh_x = np.asarray(lbm_mesh_overlay["x"], dtype=np.float64)
        lbm_mesh_y = np.asarray(lbm_mesh_overlay["y"], dtype=np.float64)
        lbm_mesh_z = np.asarray(lbm_mesh_overlay["z"], dtype=np.float64)
        lbm_mesh_freq_ratio = np.asarray(lbm_mesh_overlay["freq_ratio"], dtype=np.float64)
        lbm_mesh_freq_ok = np.isfinite(lbm_mesh_freq_ratio) & (lbm_mesh_freq_ratio > 0.0)
        lbm_mesh_hover_arr = np.asarray(lbm_mesh_hover_text, dtype=object)
        lbm_mesh_thumb_arr = np.asarray(lbm_mesh_thumb_src, dtype=object)

        if np.any(lbm_mesh_freq_ok):
            lbm_mesh_color_values = transform_color_values(
                lbm_mesh_freq_ratio[lbm_mesh_freq_ok],
                vmin=color_min,
                vmax=color_max,
                transform=args.color_transform,
                gamma=gamma,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=lbm_mesh_x[lbm_mesh_freq_ok],
                    y=lbm_mesh_y[lbm_mesh_freq_ok],
                    z=lbm_mesh_z[lbm_mesh_freq_ok],
                    mode="markers",
                    name=f"{args.lbm_mesh_name} BEM solved",
                    text=lbm_mesh_hover_arr[lbm_mesh_freq_ok].tolist(),
                    customdata=np.column_stack(
                        [
                            lbm_mesh_thumb_arr[lbm_mesh_freq_ok],
                            lbm_mesh_hover_arr[lbm_mesh_freq_ok],
                        ]
                    ),
                    hoverinfo="none",
                    marker=dict(
                        size=args.lbm_marker_size,
                        symbol="diamond",
                        color=lbm_mesh_color_values,
                        colorscale=paper_colorscale(),
                        opacity=0.98,
                        cmin=transform_color_values(
                            np.array([color_min]),
                            vmin=color_min,
                            vmax=color_max,
                            transform=args.color_transform,
                            gamma=gamma,
                        )[0],
                        cmax=transform_color_values(
                            np.array([color_max]),
                            vmin=color_min,
                            vmax=color_max,
                            transform=args.color_transform,
                            gamma=gamma,
                        )[0],
                        showscale=False,
                        line=dict(width=0),
                    ),
                )
            )
            lbm_mesh_trace_indices.append(len(fig.data) - 1)

        lbm_mesh_pending = ~lbm_mesh_freq_ok
        if np.any(lbm_mesh_pending):
            fig.add_trace(
                go.Scatter3d(
                    x=lbm_mesh_x[lbm_mesh_pending],
                    y=lbm_mesh_y[lbm_mesh_pending],
                    z=lbm_mesh_z[lbm_mesh_pending],
                    mode="markers",
                    name=f"{args.lbm_mesh_name} BEM pending",
                    text=lbm_mesh_hover_arr[lbm_mesh_pending].tolist(),
                    customdata=np.column_stack(
                        [
                            lbm_mesh_thumb_arr[lbm_mesh_pending],
                            lbm_mesh_hover_arr[lbm_mesh_pending],
                        ]
                    ),
                    hoverinfo="none",
                    marker=dict(
                        size=args.lbm_marker_size,
                        symbol="diamond",
                        color=args.lbm_mesh_color,
                        opacity=0.42,
                        line=dict(width=0),
                    ),
                )
            )
            lbm_mesh_trace_indices.append(len(fig.data) - 1)

    axis_common = dict(
        type=args.axis_scale,
        showgrid=True,
        zeroline=False,
        showspikes=False,
    )
    # Group visibility is driven by the HTML toggle panel injected in
    # toggle_panel_script(), not by Plotly's updatemenus -- a restyle button
    # cannot show its own on/off state, which is the whole point here.
    toggle_groups = list(main_group_traces)
    if lbm_mesh_trace_indices:
        toggle_groups.append(
            {
                "name": args.lbm_mesh_name,
                "index": lbm_mesh_trace_indices[0],
                "count": len(lbm_mesh_trace_indices),
                "extra_indices": lbm_mesh_trace_indices,
            }
        )
    updatemenus: list[dict] = []
    fig.update_layout(
        title=dict(
            text=(
                "<b>BubbleGym shape space</b>"
                "<br><span style='font-size:13px;color:#64748b'>"
                "10,000 bubbles · colour = f/f<sub>M</sub> · "
                "pick any three of the eight shape descriptors"
                "</span>"
            ),
            x=0.5,
            xanchor="center",
            y=0.975,
            yanchor="top",
            font=dict(size=20, color="#1f2937"),
        ),
        updatemenus=updatemenus,
        scene=dict(
            xaxis=dict(axis_common, title="1 − Φ<sub>VA</sub>"),
            yaxis=dict(axis_common, title="1 − Φ<sub>VM</sub>"),
            zaxis=dict(axis_common, title="1 − Φ<sub>W</sub>"),
            camera=dict(eye=dict(x=1.65, y=1.65, z=1.1)),
            aspectmode="cube",
        ),
        # The axis definitions used to sit in a MathJax block above the plot,
        # which cost ~135 px of headroom on every view. They now live in the
        # collapsible panel injected by legend_note_script(), so the scene gets
        # the full window height.
        margin=dict(l=0, r=0, t=64, b=0),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        hovermode=False,
        autosize=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        args.output,
        include_plotlyjs="cdn",
        full_html=True,
        default_height="100vh",
        post_script=(
            # control_panel_script first: it seeds the CSS variables the
            # hover card styles itself from.
            control_panel_script(
                toggle_groups, feature_labels, feature_titles,
                len(main_group_traces),
            )
            + thumbnail_hover_script(
                feature_labels, feat_lo.tolist(), feat_hi.tolist()
            )
        ),
    )

    # Plotly writes a bare <head> and sizes the plot div at 100vh. The
    # browser's default 8px body margin then pushes the document 16px past the
    # viewport, which the browser answers with a scrollbar -- glaring once this
    # page is embedded in an iframe. Drop the margin so the plot owns the frame.
    page_style = "<style>html,body{margin:0;padding:0;overflow:hidden}</style>\n"
    written = args.output.read_text(encoding="utf-8")
    if page_style not in written:
        args.output.write_text(
            written.replace("</head>", page_style + "</head>", 1), encoding="utf-8"
        )

    print(f"Normalizing frequencies by Minnaert f0 = {MINNAERT_F0_HZ:.4f} Hz")
    print(f"Plotted {int(ok.sum())} of {len(df)} bubbles")
    if args.thumbnail_base_url.strip():
        print(f"Thumbnail base URL: {args.thumbnail_base_url.rstrip('/')}")
    else:
        print(f"Thumbnails found: {sum(bool(s) for s in thumb_src)} of {len(thumb_src)}")
    if lbm_overlay is not None and args.lbm_thumbnail_base_url.strip():
        print(f"LBM thumbnail base URL: {args.lbm_thumbnail_base_url.rstrip('/')}")
    if lbm_mesh_overlay is not None:
        print(f"Plotted {len(lbm_mesh_overlay['hover_text'])} LBM mesh bubbles from {args.lbm_mesh_dir}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

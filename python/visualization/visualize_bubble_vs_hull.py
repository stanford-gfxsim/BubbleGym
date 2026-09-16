"""Visualize bubbles overlaid with their convex hull.

Picks N random bubbles from a dataset CSV (default: dataset_bubblegym_10k.csv,
joined with the LBM `mesh_path` column from the source LBM CSV), computes the
convex hull for each, and writes a single multi-panel interactive HTML where
each panel shows:

  * the bubble surface as a semi-transparent Mesh3d
  * the hull as a wireframe (Scatter3d lines, one segment per hull edge)

Also reports global hull-resolution statistics over the whole dataset
(n_v_hull / n_v_bubble, n_f_hull / n_f_bubble, hull edge count) so you can see
how compressive the hull is.

Usage:
    python python/visualization/visualize_bubble_vs_hull.py \
        --dataset dataset/bubble_gym/dataset_bubblegym_10k.csv \
        --lbm-csv results/lbm_exhale_bem_galerkin_3k_selected.csv \
        --n 10 \
        --output results/bubble_vs_hull_lbm_3k_random10.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_THIS_FILE = Path(__file__).resolve()
_PYTHON_ROOT = _THIS_FILE.parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from shape_feature.nonspherical_features import (  # noqa: E402
    convex_hull_vf,
    convex_hull_features_from_vf,
)
from shape_feature.mesh_utils import (
    largest_connected_component_mesh,
    load_obj_mesh,
)
from shape_feature.mesh_utils import mesh_signed_volume


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lstrip("#").str.strip()
    return df


def _resolve_mesh_paths(
    dataset_df: pd.DataFrame,
    lbm_csv: Path | None,
    mesh_folder: Path | None,
) -> pd.DataFrame:
    """Return a copy of `dataset_df` with a `mesh_path` column for each row."""
    out = dataset_df.copy()
    if "mesh_path" in out.columns and out["mesh_path"].notna().all():
        return out

    if lbm_csv is not None and lbm_csv.is_file():
        lbm = _clean_columns(pd.read_csv(lbm_csv))
        if "mesh_file" in lbm.columns and "mesh_path" in lbm.columns:
            mapping = dict(zip(lbm["mesh_file"].astype(str), lbm["mesh_path"].astype(str)))
            out["mesh_path"] = out["mesh_filename"].astype(str).map(mapping)

    if mesh_folder is not None and "mesh_path" not in out.columns:
        out["mesh_path"] = out["mesh_filename"].astype(str).map(
            lambda n: str((mesh_folder / n).resolve())
        )

    if "mesh_path" not in out.columns or out["mesh_path"].isna().all():
        raise FileNotFoundError(
            "Could not resolve mesh paths. Pass --lbm-csv (with a mesh_path column) "
            "or --mesh-folder."
        )
    return out


def _load_clean_mesh(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    v_raw, f_raw = load_obj_mesh(obj_path)
    v, f, _, _ = largest_connected_component_mesh(v_raw, f_raw)
    if mesh_signed_volume(v, f) < 0.0:
        f = f[:, [0, 2, 1]]
    return v, f


def _hull_edges(simplices: np.ndarray) -> np.ndarray:
    """Return unique undirected edges (E, 2) of a triangle mesh."""
    e = np.vstack(
        [
            simplices[:, [0, 1]],
            simplices[:, [1, 2]],
            simplices[:, [2, 0]],
        ]
    )
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    return e


def _bubble_trace(
    v: np.ndarray,
    f: np.ndarray,
    *,
    color: str,
    opacity: float,
    name: str,
) -> go.Mesh3d:
    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color=color,
        opacity=opacity,
        flatshading=False,
        name=name,
        showlegend=False,
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.05),
        lightposition=dict(x=1.0, y=1.0, z=2.0),
        hoverinfo="skip",
    )


def _hull_wireframe_trace(
    v_hull: np.ndarray,
    f_hull: np.ndarray,
    *,
    color: str,
    width: float,
    name: str,
) -> go.Scatter3d:
    edges = _hull_edges(f_hull)
    seg_x: list[float] = []
    seg_y: list[float] = []
    seg_z: list[float] = []
    for a, b in edges:
        seg_x.extend([v_hull[a, 0], v_hull[b, 0], None])
        seg_y.extend([v_hull[a, 1], v_hull[b, 1], None])
        seg_z.extend([v_hull[a, 2], v_hull[b, 2], None])
    return go.Scatter3d(
        x=seg_x,
        y=seg_y,
        z=seg_z,
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        showlegend=False,
        hoverinfo="skip",
    )


def _resolution_stats(rows: list[dict]) -> dict:
    arr = lambda key: np.asarray([r[key] for r in rows], dtype=np.float64)  # noqa: E731
    n_v_b = arr("n_v_bubble")
    n_v_h = arr("n_v_hull")
    n_f_b = arr("n_f_bubble")
    n_f_h = arr("n_f_hull")
    n_e_h = arr("n_e_hull")
    return {
        "n_bubbles": len(rows),
        "bubble_vertices": dict(min=int(n_v_b.min()), median=float(np.median(n_v_b)), max=int(n_v_b.max()), mean=float(n_v_b.mean())),
        "bubble_faces": dict(min=int(n_f_b.min()), median=float(np.median(n_f_b)), max=int(n_f_b.max()), mean=float(n_f_b.mean())),
        "hull_vertices": dict(min=int(n_v_h.min()), median=float(np.median(n_v_h)), max=int(n_v_h.max()), mean=float(n_v_h.mean())),
        "hull_faces": dict(min=int(n_f_h.min()), median=float(np.median(n_f_h)), max=int(n_f_h.max()), mean=float(n_f_h.mean())),
        "hull_edges": dict(min=int(n_e_h.min()), median=float(np.median(n_e_h)), max=int(n_e_h.max()), mean=float(n_e_h.mean())),
        "vertex_compression_hull_over_bubble": dict(
            min=float((n_v_h / n_v_b).min()),
            median=float(np.median(n_v_h / n_v_b)),
            max=float((n_v_h / n_v_b).max()),
            mean=float((n_v_h / n_v_b).mean()),
        ),
        "face_compression_hull_over_bubble": dict(
            min=float((n_f_h / n_f_b).min()),
            median=float(np.median(n_f_h / n_f_b)),
            max=float((n_f_h / n_f_b).max()),
            mean=float((n_f_h / n_f_b).mean()),
        ),
    }


def _print_block(title: str, lines: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for line in lines:
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/bubble_gym/dataset_bubblegym_10k.csv"),
        help="CSV listing bubbles (must have mesh_filename, ideally eta_*).",
    )
    parser.add_argument(
        "--lbm-csv",
        type=Path,
        default=Path("results/lbm_exhale_bem_galerkin_3k_selected.csv"),
        help="LBM CSV providing the mesh_path column (joined on mesh_filename).",
    )
    parser.add_argument(
        "--mesh-folder",
        type=Path,
        default=None,
        help="Optional fallback: directory containing OBJ files named like mesh_filename.",
    )
    parser.add_argument("--n", type=int, default=10, help="Number of random bubbles to render.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bubble_vs_hull_lbm_3k_random10.html"),
    )
    parser.add_argument(
        "--stratify",
        choices=["random", "dendritic"],
        default="random",
        help=(
            "random: uniform sample of N. dendritic: stratify on the chull "
            "dendritic axis (max(eta_A-1,0)+max(1-eta_V,0)-max(eta_M-1,0)) so the "
            "selected bubbles span low to high dendritic signal."
        ),
    )
    parser.add_argument(
        "--global-resolution-sample",
        type=int,
        default=200,
        help=(
            "Compute hull-resolution statistics on this many randomly sampled "
            "bubbles (in addition to the N rendered). Set to 0 to skip."
        ),
    )
    parser.add_argument("--bubble-color", default="#a8c4ff")
    parser.add_argument("--bubble-opacity", type=float, default=0.45)
    parser.add_argument("--hull-color", default="#cc1f3a")
    parser.add_argument("--hull-line-width", type=float, default=2.5)
    args = parser.parse_args()

    df = _clean_columns(pd.read_csv(args.dataset))
    if "mesh_filename" not in df.columns:
        raise ValueError(f"{args.dataset} missing mesh_filename column")

    df = _resolve_mesh_paths(df, args.lbm_csv if args.lbm_csv else None, args.mesh_folder)

    df = df[df["mesh_path"].notna()].reset_index(drop=True)
    df = df[df["mesh_path"].astype(str).map(lambda p: Path(p).is_file())].reset_index(drop=True)
    if df.empty:
        raise FileNotFoundError("No mesh files found on disk after path resolution.")

    rng = np.random.default_rng(args.seed)

    if args.stratify == "random" or not {"eta_V", "eta_A", "eta_M"}.issubset(df.columns):
        if args.stratify == "dendritic":
            print("[warn] eta_V/A/M not present, falling back to random selection.")
        idx_render = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
    else:
        eta_v = pd.to_numeric(df["eta_V"], errors="coerce").to_numpy()
        eta_a = pd.to_numeric(df["eta_A"], errors="coerce").to_numpy()
        eta_m = pd.to_numeric(df["eta_M"], errors="coerce").to_numpy()
        dendritic = (
            np.maximum(eta_a - 1.0, 0.0)
            + np.maximum(1.0 - eta_v, 0.0)
            - np.maximum(eta_m - 1.0, 0.0)
        )
        ok = np.isfinite(dendritic)
        order = np.argsort(dendritic[ok])
        avail = np.where(ok)[0][order]
        n = min(args.n, len(avail))
        positions = np.linspace(0, len(avail) - 1, n).astype(int)
        idx_render = avail[positions]

    rendered_rows: list[dict] = []
    bubble_v_list: list[np.ndarray] = []
    bubble_f_list: list[np.ndarray] = []
    hull_v_list: list[np.ndarray] = []
    hull_f_list: list[np.ndarray] = []

    for i in idx_render:
        row = df.iloc[int(i)]
        obj_path = Path(str(row["mesh_path"]))
        v, f = _load_clean_mesh(obj_path)
        feats = convex_hull_features_from_vf(v, f, target_volume=1.0)
        v_unit, _ = convex_hull_vf(v)  # not used; keep for symmetry; cheap
        # Recompute the actual hull at unit volume so the visualization is in the same scale.
        from shape_feature.nonspherical_features import vertices_scaled_to_target_volume

        v_scaled = vertices_scaled_to_target_volume(v.astype(np.float64), f.astype(np.int64), 1.0)
        v_hull, f_hull = convex_hull_vf(v_scaled)

        bubble_v_list.append(v_scaled)
        bubble_f_list.append(f.astype(np.int64))
        hull_v_list.append(v_hull)
        hull_f_list.append(f_hull)

        rendered_rows.append(
            dict(
                mesh_filename=str(row["mesh_filename"]),
                mesh_path=str(obj_path),
                n_v_bubble=int(v.shape[0]),
                n_f_bubble=int(f.shape[0]),
                n_v_hull=int(v_hull.shape[0]),
                n_f_hull=int(f_hull.shape[0]),
                n_e_hull=int(_hull_edges(f_hull).shape[0]),
                eta_V=float(feats["eta_V"]),
                eta_A=float(feats["eta_A"]),
                eta_M=float(feats["eta_M"]),
            )
        )

    _print_block(
        f"Per-bubble hull resolution (rendered N={len(rendered_rows)})",
        [
            f"{r['mesh_filename']:<24s}  "
            f"bub V/F = {r['n_v_bubble']:>5d}/{r['n_f_bubble']:>5d}  "
            f"hull V/F/E = {r['n_v_hull']:>4d}/{r['n_f_hull']:>4d}/{r['n_e_hull']:>4d}  "
            f"V_hull/V_bub = {r['n_v_hull'] / max(r['n_v_bubble'], 1):.3f}  "
            f"eta_V={r['eta_V']:.3f}  eta_A={r['eta_A']:.3f}  eta_M={r['eta_M']:.3f}"
            for r in rendered_rows
        ],
    )

    if args.global_resolution_sample > 0:
        sample_size = min(args.global_resolution_sample, len(df))
        sample_idx = rng.choice(len(df), size=sample_size, replace=False)
        global_rows: list[dict] = []
        from tqdm import tqdm  # type: ignore

        for i in tqdm(sample_idx, desc=f"hull stats (N={sample_size})", unit="mesh"):
            row = df.iloc[int(i)]
            obj_path = Path(str(row["mesh_path"]))
            try:
                v, f = _load_clean_mesh(obj_path)
                from shape_feature.nonspherical_features import vertices_scaled_to_target_volume

                v_scaled = vertices_scaled_to_target_volume(v.astype(np.float64), f.astype(np.int64), 1.0)
                v_hull, f_hull = convex_hull_vf(v_scaled)
                global_rows.append(
                    dict(
                        n_v_bubble=int(v.shape[0]),
                        n_f_bubble=int(f.shape[0]),
                        n_v_hull=int(v_hull.shape[0]),
                        n_f_hull=int(f_hull.shape[0]),
                        n_e_hull=int(_hull_edges(f_hull).shape[0]),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {obj_path.name}: {exc}")

        stats = _resolution_stats(global_rows)
        _print_block(
            f"Global hull-resolution stats over {stats['n_bubbles']} random bubbles",
            [
                "                  min   median       max     mean",
                f"  bubble V    {stats['bubble_vertices']['min']:>6d}   {stats['bubble_vertices']['median']:>6.0f}   {stats['bubble_vertices']['max']:>7d}   {stats['bubble_vertices']['mean']:>7.1f}",
                f"  bubble F    {stats['bubble_faces']['min']:>6d}   {stats['bubble_faces']['median']:>6.0f}   {stats['bubble_faces']['max']:>7d}   {stats['bubble_faces']['mean']:>7.1f}",
                f"  hull   V    {stats['hull_vertices']['min']:>6d}   {stats['hull_vertices']['median']:>6.0f}   {stats['hull_vertices']['max']:>7d}   {stats['hull_vertices']['mean']:>7.1f}",
                f"  hull   F    {stats['hull_faces']['min']:>6d}   {stats['hull_faces']['median']:>6.0f}   {stats['hull_faces']['max']:>7d}   {stats['hull_faces']['mean']:>7.1f}",
                f"  hull   E    {stats['hull_edges']['min']:>6d}   {stats['hull_edges']['median']:>6.0f}   {stats['hull_edges']['max']:>7d}   {stats['hull_edges']['mean']:>7.1f}",
                "",
                f"  V_hull/V_bub   min={stats['vertex_compression_hull_over_bubble']['min']:.3f}  median={stats['vertex_compression_hull_over_bubble']['median']:.3f}  max={stats['vertex_compression_hull_over_bubble']['max']:.3f}  mean={stats['vertex_compression_hull_over_bubble']['mean']:.3f}",
                f"  F_hull/F_bub   min={stats['face_compression_hull_over_bubble']['min']:.3f}  median={stats['face_compression_hull_over_bubble']['median']:.3f}  max={stats['face_compression_hull_over_bubble']['max']:.3f}  mean={stats['face_compression_hull_over_bubble']['mean']:.3f}",
            ],
        )

    n = len(rendered_rows)
    cols = 5 if n >= 5 else n
    rows_grid = (n + cols - 1) // cols
    specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows_grid)]
    titles = [
        (
            f"{r['mesh_filename']}<br>"
            f"bub V/F={r['n_v_bubble']}/{r['n_f_bubble']}, hull V/F={r['n_v_hull']}/{r['n_f_hull']}<br>"
            f"η_V={r['eta_V']:.3f}, η_A={r['eta_A']:.3f}, η_M={r['eta_M']:.3f}"
        )
        for r in rendered_rows
    ]
    fig = make_subplots(
        rows=rows_grid,
        cols=cols,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.005,
        vertical_spacing=0.06,
    )

    for k, r in enumerate(rendered_rows):
        rr = k // cols + 1
        cc = k % cols + 1
        fig.add_trace(
            _bubble_trace(
                bubble_v_list[k],
                bubble_f_list[k],
                color=args.bubble_color,
                opacity=args.bubble_opacity,
                name="bubble",
            ),
            row=rr,
            col=cc,
        )
        fig.add_trace(
            _hull_wireframe_trace(
                hull_v_list[k],
                hull_f_list[k],
                color=args.hull_color,
                width=args.hull_line_width,
                name="hull",
            ),
            row=rr,
            col=cc,
        )

    scene_kwargs = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
    )
    layout_updates: dict[str, dict] = {}
    for k in range(n):
        scene_id = "scene" if k == 0 else f"scene{k+1}"
        layout_updates[scene_id] = scene_kwargs
    fig.update_layout(
        title=dict(
            text=(
                f"Bubble (light blue surface) vs convex hull (red wireframe) — "
                f"{args.stratify} sample of {n} from {args.dataset.name}"
            ),
            x=0.5,
            xanchor="center",
        ),
        margin=dict(l=10, r=10, t=110, b=10),
        template="plotly_white",
        height=320 * rows_grid + 120,
        **layout_updates,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn", full_html=True)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

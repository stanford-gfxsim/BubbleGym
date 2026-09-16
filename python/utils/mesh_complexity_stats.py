"""Vertex / face / topology statistics for a dataset's bubble meshes.

Reports per-source and overall vertex and triangle counts for every mesh
referenced by a dataset CSV, plus the genus distribution derived from the Euler
characteristic. Useful for describing the benchmark ("meshes average ~4k
vertices") and for spotting topology the shape features have to cope with.

Counting is done with a single line scan per OBJ rather than a full parse, so a
10k-mesh dataset takes seconds. Polygonal faces are fan-triangulated the same
way :func:`shape_feature.mesh_utils.load_obj_mesh` does, so the triangle counts
match what the feature pipeline actually integrates over.

Genus is exact only for closed manifold meshes. For a closed triangle mesh every
edge has two incident faces, so ``E = 3F/2`` and::

    chi = V - E + F = V - F/2,      chi = 2 - 2g

Meshes with holes make that identity meaningless, so ``--check-watertight``
validates topology first (slower: it parses faces and builds the edge table).

Examples
--------
::

    python python/utils/mesh_complexity_stats.py \\
        --mesh-folder C:/path/to/meshes10k

    python python/utils/mesh_complexity_stats.py \\
        --mesh-folder C:/path/to/meshes10k \\
        --check-watertight --per-mesh-csv results/mesh_complexity.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
PYTHON_ROOT = _THIS.parents[1]
REPO_ROOT = PYTHON_ROOT.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


__all__ = ["count_obj_elements", "collect_mesh_stats", "summarize"]


def count_obj_elements(obj_path: Path) -> tuple[int, int, int]:
    """Return ``(n_vertices, n_face_lines, n_triangles)`` for one OBJ.

    ``n_triangles`` fan-triangulates polygonal faces (an ``n``-gon contributes
    ``n - 2`` triangles), matching the loader used by the feature pipeline. For
    an already-triangulated mesh it equals ``n_face_lines``.
    """
    n_v = n_f = n_tri = 0
    with Path(obj_path).open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                n_v += 1
            elif line.startswith("f "):
                n_f += 1
                n_tri += max(0, len(line.split()) - 3)
    return n_v, n_f, n_tri


def _resolve(mesh_folder: Path, key: str, filename: str) -> Path | None:
    """Meshes ship either flat or grouped as ``<source>/<filename>``."""
    for candidate in (mesh_folder / key, mesh_folder / filename):
        if candidate.is_file():
            return candidate
    return None


def collect_mesh_stats(
    df: pd.DataFrame, mesh_folder: Path, *, check_watertight: bool = False
) -> pd.DataFrame:
    """Scan every mesh referenced by ``df``; returns one row per mesh found.

    Adds ``chi`` and ``genus`` columns. When ``check_watertight`` is set, each
    mesh is fully parsed and its edge incidence verified, and ``watertight``
    records the result -- genus is only meaningful where that is True.
    """
    keys = (
        df["mesh_id"].astype(str).str.strip().tolist()
        if "mesh_id" in df.columns
        else df["mesh_filename"].astype(str).str.strip().tolist()
    )
    names = df["mesh_filename"].astype(str).str.strip().tolist()
    sources = (
        df["source"].astype(str).tolist() if "source" in df.columns else ["?"] * len(df)
    )

    if check_watertight:
        from shape_feature.mesh_utils import edge_incidence_report, load_obj_mesh

    rows, missing = [], 0
    for key, name, src in zip(keys, names, sources):
        path = _resolve(mesh_folder, key, name)
        if path is None:
            missing += 1
            continue
        n_v, n_f, n_tri = count_obj_elements(path)
        rec = {
            "mesh_id": key, "source": src,
            "n_vertices": n_v, "n_face_lines": n_f, "n_triangles": n_tri,
        }
        if check_watertight:
            _v, f = load_obj_mesh(path, require_watertight=False)
            rep = edge_incidence_report(f)
            rec["watertight"] = (
                rep["n_boundary_edges"] == 0 and rep["n_nonmanifold_edges"] == 0
            )
            rec["n_boundary_edges"] = rep["n_boundary_edges"]
        rows.append(rec)

    if missing:
        print(f"WARNING: {missing} mesh file(s) referenced by the CSV were not found "
              f"under {mesh_folder}; they are excluded from the statistics.")
    if not rows:
        raise SystemExit(f"no meshes found under {mesh_folder}")

    out = pd.DataFrame(rows)
    # Closed triangle mesh: E = 3F/2, so chi = V - F/2 and chi = 2 - 2g.
    out["chi"] = out["n_vertices"] - out["n_triangles"] // 2
    out["genus"] = (2 - out["chi"]) // 2
    return out


def _describe(label: str, d: pd.DataFrame) -> None:
    print(f"--- {label}  (n={len(d)}) ---")
    for col, name in (("n_vertices", "vertices"), ("n_triangles", "faces")):
        v = d[col].to_numpy()
        std = v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"  {name:<9} mean {v.mean():9.1f}  median {np.median(v):8.1f}  "
              f"std {std:8.1f}  min {v.min():6d}  max {v.max():7d}")
    if "watertight" in d.columns:
        n_bad = int((~d["watertight"]).sum())
        print(f"  watertight: {len(d) - n_bad}/{len(d)}"
              + (f"   ({n_bad} NOT closed -- genus below is meaningless for those)"
                 if n_bad else ""))
    g = Counter(d["genus"].tolist())
    top = ", ".join(f"g={k}: {v} ({100 * v / len(d):.1f}%)" for k, v in sorted(g.items())[:5])
    print(f"  genus     min {d['genus'].min()}  median {int(d['genus'].median())}  "
          f"max {d['genus'].max()}")
    print(f"            {top}")
    print()


def summarize(stats: pd.DataFrame) -> None:
    non_tri = int((stats["n_face_lines"] != stats["n_triangles"]).sum())
    print(f"meshes scanned: {len(stats)}")
    print(f"meshes with non-triangular faces: {non_tri}\n")
    _describe("ALL", stats)
    for s in sorted(stats["source"].unique()):
        _describe(str(s), stats[stats["source"] == s])
    print(f"total vertices : {int(stats['n_vertices'].sum()):,}")
    print(f"total triangles: {int(stats['n_triangles'].sum()):,}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "dataset" / "bubble_gym" / "dataset_bubblegym_10k.csv",
        help="CSV with 'mesh_filename' (and ideally 'mesh_id' / 'source').",
    )
    parser.add_argument("--mesh-folder", type=Path, required=True,
                        help="Folder of OBJs, flat or grouped as <source>/<filename>.")
    parser.add_argument("--check-watertight", action="store_true",
                        help="Also verify each mesh is closed (slower: full parse).")
    parser.add_argument("--per-mesh-csv", type=Path, default=None,
                        help="Write the per-mesh table here as well.")
    args = parser.parse_args()

    csv_path = args.dataset.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"dataset not found: {csv_path}")
    mesh_folder = args.mesh_folder.resolve()
    if not mesh_folder.is_dir():
        raise SystemExit(f"mesh folder not found: {mesh_folder}")

    df = pd.read_csv(csv_path, float_precision="round_trip")
    if "mesh_filename" not in df.columns:
        raise SystemExit(f"{csv_path} missing 'mesh_filename'")

    stats = collect_mesh_stats(df, mesh_folder, check_watertight=args.check_watertight)
    summarize(stats)

    if args.per_mesh_csv is not None:
        args.per_mesh_csv.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.per_mesh_csv, index=False)
        print(f"\nwrote per-mesh table: {args.per_mesh_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

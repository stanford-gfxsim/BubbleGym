"""Render one small PNG thumbnail per bubble in the BubbleGym benchmark.

Walks ``dataset_bubblegym_10k.csv`` and renders every mesh it references to a
transparent-background PNG, mirroring the CSV's ``mesh_id`` layout::

    dataset/bubble_gym/bubble_mesh_thumbnails/
        VOF/bubble.0019.0.png
        LBM/bubble.0007.1.png

The VOF/LBM split is not cosmetic: 56 filenames occur in both groups (distinct
bubbles from the two solvers that happen to share a ``bubble.NNNN.M.obj`` name),
so a flat output directory would silently overwrite them.

Rendering reuses :func:`visualization.plot_dataset_distribution.render_transparent_thumbnail`,
which reduces each mesh to its largest connected component before drawing --
the same path the paper's Fig. 3 / Fig. 4 insets use, so thumbnails here match
the figures (light blue ``#bcd4f5`` fill, dark slate ``#2f4a6d`` wireframe,
transparent background).

Each bubble is rendered at ``--size`` (400 px, the paper resolution) and then
Lanczos-downsampled to ``--out-size`` (100 px). Rendering directly at 100 px
does *not* work: the meshes carry thousands of triangles, so at that scale the
wireframe covers the whole surface and every bubble comes out a dark blob.
Supersampling averages those edges into shading instead.

Usage::

    python python/visualization/render_dataset_thumbnails.py
    python python/visualization/render_dataset_thumbnails.py --out-size 0    # keep 400 px
    python python/visualization/render_dataset_thumbnails.py --limit 50      # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

_REPO_ROOT = _PYTHON_ROOT.parent

DEFAULT_CSV = _REPO_ROOT / "dataset" / "bubble_gym" / "dataset_bubblegym_10k.csv"
# The 10k mesh archive is distributed separately from this repo (see
# dataset/README.md). Point BUBBLEGYM_MESH_ROOT at it, or pass --mesh-root.
DEFAULT_MESH_ROOT = Path(
    os.environ.get(
        "BUBBLEGYM_MESH_ROOT",
        str(_REPO_ROOT / "dataset" / "bubble_gym" / "meshes10k"),
    )
)
# Output dir is named for the saved resolution, so the 400x400 masters and the
# 100x100 distribution copies live side by side.
DEFAULT_OUT_PARENT = _REPO_ROOT / "dataset" / "bubble_gym"


def downsample_png(path: Path, out_size: int) -> None:
    """Lanczos-resize an RGBA PNG in place to ``out_size`` square.

    Rendering large and shrinking is what keeps the paper's wireframe legible at
    small sizes: PyVista drawing directly at 100 px puts one screen pixel on many
    triangle edges, so the surface floods dark. Supersampling averages those edges
    into shading instead.
    """
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGBA")
        if im.size != (out_size, out_size):
            im = im.resize((out_size, out_size), Image.LANCZOS)
        im.save(path, optimize=True)


def _render_one(job: tuple[str, str, dict, int]) -> tuple[str, bool, str]:
    """Worker: render one mesh. Returns ``(mesh_id, ok, message)``."""
    mesh_path_str, out_png_str, kwargs, out_size = job
    out_png = Path(out_png_str)
    if out_png.is_file() and out_png.stat().st_size > 0:
        return mesh_path_str, True, "cached"
    try:
        from visualization.plot_dataset_distribution import render_transparent_thumbnail

        out_png.parent.mkdir(parents=True, exist_ok=True)
        ok = render_transparent_thumbnail(Path(mesh_path_str), out_png, **kwargs)
        if ok and out_size > 0:
            downsample_png(out_png, out_size)
        return mesh_path_str, bool(ok), "" if ok else "renderer returned False"
    except Exception as exc:  # noqa: BLE001
        return mesh_path_str, False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT,
                    help="Root holding the VOF/ and LBM/ mesh folders.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Defaults to dataset/bubble_gym/bubble_mesh_thumbnails_<N>x<N>, "
                         "where N is the saved resolution.")
    ap.add_argument("--size", type=int, default=400,
                    help="Render (supersample) resolution. 400 reproduces the "
                         "paper figures. Do not lower this to shrink the output "
                         "-- use --out-size, which downsamples instead.")
    ap.add_argument("--out-size", type=int, default=100,
                    help="Final saved resolution; the render at --size is "
                         "Lanczos-downsampled to it. 0 keeps the full --size "
                         "render. Default 100.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel render processes. 1 = in-process (easier to debug).")
    # Defaults below are the paper's Fig. 4 / Fig. 7 inset style.
    ap.add_argument("--mesh-color", type=str, default="#bcd4f5")
    ap.add_argument("--edge-color", type=str, default="#2f4a6d")
    ap.add_argument("--show-edges", action=argparse.BooleanOptionalAction, default=True,
                    help="Draw the triangle wireframe (on, as in the paper).")
    ap.add_argument("--ambient", type=float, default=0.0,
                    help="Ambient light term; raise to lift the shadowed side.")
    ap.add_argument("--diffuse", type=float, default=1.0)
    ap.add_argument("--specular", type=float, default=0.0,
                    help="Specular highlight strength.")
    ap.add_argument("--specular-power", type=float, default=30.0)
    ap.add_argument("--smooth-shading", action=argparse.BooleanOptionalAction, default=False,
                    help="Phong shading. Off in the paper style.")
    ap.add_argument("--extra-lighting", action=argparse.BooleanOptionalAction, default=False,
                    help="Add a three-point light kit on top of the default headlight.")
    ap.add_argument("--force", action="store_true",
                    help="Re-render even when a non-empty PNG already exists.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Render only the first N rows (smoke test).")
    args = ap.parse_args()

    render_kwargs = dict(
        mesh_color=args.mesh_color,
        edge_color=args.edge_color,
        window_size=(int(args.size), int(args.size)),
        show_edges=bool(args.show_edges),
        ambient=float(args.ambient),
        diffuse=float(args.diffuse),
        specular=float(args.specular),
        specular_power=float(args.specular_power),
        smooth_shading=bool(args.smooth_shading),
        extra_lighting=bool(args.extra_lighting),
    )

    df = pd.read_csv(args.csv)
    for col in ("mesh_id", "mesh_filename", "source"):
        if col not in df.columns:
            raise SystemExit(f"{args.csv} missing column {col!r}")
    if args.limit > 0:
        df = df.head(args.limit)

    mesh_root = args.mesh_root.resolve()
    saved_px = int(args.out_size) or int(args.size)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else DEFAULT_OUT_PARENT / f"bubble_mesh_thumbnails_{saved_px}x{saved_px}"
    )

    jobs: list[tuple[str, str, dict, int]] = []
    missing: list[str] = []
    for mesh_id in df["mesh_id"].astype(str):
        src = mesh_root / mesh_id
        if not src.is_file():
            missing.append(mesh_id)
            continue
        dst = out_dir / Path(mesh_id).with_suffix(".png")
        if args.force and dst.is_file():
            dst.unlink()
        jobs.append((str(src), str(dst), render_kwargs, int(args.out_size)))

    if missing:
        print(f"[warn] {len(missing)} mesh_id(s) not found under {mesh_root}; "
              f"e.g. {missing[:3]}")
    tgt = int(args.out_size) or int(args.size)
    print(f"Rendering {len(jobs)} thumbnails at {args.size}x{args.size}"
          f"{f' -> downsampled to {tgt}x{tgt}' if args.out_size else ''} "
          f"-> {out_dir}  (workers={args.workers})", flush=True)

    t0 = time.perf_counter()
    n_ok = n_cached = 0
    failures: list[tuple[str, str]] = []

    def _consume(it) -> None:
        nonlocal n_ok, n_cached
        for i, (mid, ok, msg) in enumerate(it, 1):
            if ok:
                n_ok += 1
                if msg == "cached":
                    n_cached += 1
            else:
                failures.append((mid, msg))
            if i % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  {i}/{len(jobs)} | {el/60:.1f} min | "
                      f"eta {(len(jobs)-i)*el/i/60:.1f} min", flush=True)

    if args.workers <= 1:
        _consume(_render_one(j) for j in jobs)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            _consume(ex.map(_render_one, jobs, chunksize=16))

    el = time.perf_counter() - t0
    print(f"\ndone in {el/60:.1f} min: {n_ok} ok ({n_cached} already cached), "
          f"{len(failures)} failed")
    for mid, msg in failures[:10]:
        print(f"  [fail] {mid}: {msg}")

    for group in sorted({str(s) for s in df["source"]}):
        d = out_dir / group
        if d.is_dir():
            pngs = list(d.glob("*.png"))
            mb = sum(p.stat().st_size for p in pngs) / 1024**2
            print(f"  {group}: {len(pngs)} PNGs, {mb:.1f} MB")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

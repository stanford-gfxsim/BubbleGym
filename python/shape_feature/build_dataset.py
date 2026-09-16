"""Compute every shape feature for a list of meshes, in one pass per mesh.

One pass yields every column::

    surface_area, volume                        (at unit volume)
    M, R_M, Phi_VM, eta_VM                      (mean-curvature integral)
    W_vertex, W, Phi_W, eta_W                   (Willmore)
    V_hull, A_hull, M_hull, eta_V, eta_A, eta_M (convex hull)
    i00, i11, i22                               (principal moments, ascending)

The same worker also backs inference-time extraction: :func:`extract_features_one`
optionally Laplacian-smooths and caches the smoothed OBJ first, which is what
the scene pipelines need. ``run_feature_pool`` is the shared
``ProcessPoolExecutor`` driver.

Usage::

    python python/shape_feature/build_dataset.py \\
        --dataset dataset/bubble_gym/dataset_bubblegym_10k.csv \\
        --mesh-folder <folder of OBJs named by mesh_filename>

Meshes that are not closed are reported and counted as failures rather than
silently contributing NaN columns -- see ``mesh_utils.check_watertight``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_PYTHON_ROOT = _THIS_DIR.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from shape_feature.mesh_utils import (  # noqa: E402
    compute_mirtich_moments,
    ensure_outward_orientation,
    largest_connected_component_mesh,
    load_obj_mesh,
    load_or_smooth_obj,
    principal_inertia,
    vertices_scaled_to_target_volume,
)
from shape_feature.nonspherical_features import (  # noqa: E402
    convex_hull_features_from_vf,
    curvature_integrals_from_vf,
    sphericities_from_integrals,
)


__all__ = ["extract_features_one", "run_feature_pool", "main"]


# Columns written back to the dataset CSV, in order.
CURVATURE_COLUMNS = (
    "M", "R_M", "Phi_VM", "eta_VM",
    "W_vertex", "W", "Phi_W", "eta_W",
)
HULL_COLUMNS = ("V_hull", "A_hull", "M_hull", "eta_V", "eta_A", "eta_M")
INERTIA_COLUMNS = ("i00", "i11", "i22")
AREA_COLUMNS = ("surface_area", "volume")

# Everything the worker caches per mesh, so a rerun costs nothing.
_CACHE_FIELDS = (
    "surface_area", "volume",
    "M", "W_vertex",
    "V_hull", "A_hull", "M_hull",
    "i00", "i11", "i22",
)
# Keyed by mesh_key (see _mesh_keys), NOT by mesh_filename: filenames are not
# unique across sources -- in dataset_bubblegym_10k.csv, 56 names collide
# between VOF and LBM -- so a filename-keyed cache silently hands one source's
# features to the other's mesh.
CACHE_COLUMNS = ("mesh_key", *_CACHE_FIELDS)


def _mesh_keys(df: pd.DataFrame) -> list[str]:
    """Per-row unique mesh identifier, preferring ``mesh_id`` over the filename.

    ``mesh_id`` is ``<source>/<filename>`` and is unique; ``mesh_filename``
    alone is not. Falls back to the filename for datasets that lack mesh_id.
    """
    if "mesh_id" in df.columns:
        return df["mesh_id"].astype(str).str.strip().tolist()
    return df["mesh_filename"].astype(str).str.strip().tolist()


def _resolve_mesh_path(mesh_folder: Path, key: str, filename: str) -> Path | None:
    """Locate a mesh under ``mesh_folder``, trying the layouts we ship.

    Meshes may sit flat (``<folder>/<filename>``) or be grouped by source
    (``<folder>/<source>/<filename>``, which is what ``mesh_id`` encodes).
    """
    for candidate in (mesh_folder / key, mesh_folder / filename):
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Per-mesh worker
# ---------------------------------------------------------------------------


def extract_features_one(job: tuple) -> tuple[int, dict[str, Any]]:
    """Compute every shape integral for one mesh.

    ``job`` is ``(job_idx, raw_obj_path)`` or, for the smoothing path,
    ``(job_idx, raw_obj_path, cached_obj_path, smooth_iters)``. When a cache
    path is given the mesh is Laplacian-smoothed (and the smoothed OBJ cached)
    before features are computed -- the convention the audio / scene pipelines
    use. Otherwise the raw mesh is used, matching the dataset CSV columns.

    Returns ``(job_idx, feats)``. ``feats["status"]`` is ``"ok"`` or
    ``"failed: <reason>"``; a single malformed frame never aborts the pool.
    Non-closed meshes surface here as ``failed: NonWatertightMeshError: ...``.
    """
    job_idx = int(job[0])
    raw_path = Path(job[1])
    cached_path = Path(job[2]) if len(job) > 2 and job[2] else None
    smooth_iters = int(job[3]) if len(job) > 3 else 0

    feats: dict[str, Any] = {"mesh_path": str(raw_path), "obj_path": str(raw_path)}
    try:
        if cached_path is not None and smooth_iters > 0:
            v, f = load_or_smooth_obj(raw_path, cached_path, iters=smooth_iters)
        else:
            v, f = load_obj_mesh(raw_path)
            v, f, _n_cc, _kept = largest_connected_component_mesh(v, f)
            f = ensure_outward_orientation(v, f)

        v_unit = vertices_scaled_to_target_volume(v, f, 1.0)

        curv = curvature_integrals_from_vf(v_unit, f, target_volume=None)
        phi = sphericities_from_integrals(
            volume=float(curv["volume"]),
            area=float(curv["area"]),
            mean_curvature=float(curv["M"]),
            willmore_vertex=float(curv["W_vertex"]),
        )
        hull = convex_hull_features_from_vf(v_unit, f, target_volume=None)

        _mass, _com, inertia = compute_mirtich_moments(v_unit, f)
        moments = principal_inertia(inertia)

        feats.update(
            {
                "surface_area": float(curv["area"]),
                "volume": float(curv["volume"]),
                "M": float(curv["M"]),
                "W_vertex": float(curv["W_vertex"]),
                "V_hull": float(hull["V_hull"]),
                "A_hull": float(hull["A_hull"]),
                "M_hull": float(hull["M_hull"]),
                "eta_V": float(hull["eta_V"]),
                "eta_A": float(hull["eta_A"]),
                "eta_M": float(hull["eta_M"]),
                "i00": float(moments[0]),
                "i11": float(moments[1]),
                "i22": float(moments[2]),
                "Phi_VA": phi["Phi_VA"],
                "Phi_VM": phi["Phi_VM"],
                "Phi_W": phi["Phi_W"],
                "non_sph_va": 1.0 - phi["Phi_VA"],
                "non_sph_vm": 1.0 - phi["Phi_VM"],
                "non_sph_w": 1.0 - phi["Phi_W"],
                "inertia_principal_unit": [float(x) for x in moments],
                "status": "ok",
            }
        )
    except Exception as exc:  # noqa: BLE001
        feats["status"] = f"failed: {type(exc).__name__}: {exc}"
    return job_idx, feats


# ---------------------------------------------------------------------------
# Pool driver
# ---------------------------------------------------------------------------


def run_feature_pool(
    worker: Callable[[Any], tuple[int, dict[str, Any]]],
    jobs: Sequence[Any],
    workers: int,
    chunksize: int | None = None,
    milestone_every: int = 0,
) -> list[dict[str, Any]]:
    """Map ``worker`` over ``jobs``, returning results ordered by ``job_idx``.

    Each job's first element is its index into the output list, so results
    reassemble in order regardless of completion sequence. ``chunksize``
    defaults to roughly 16 batches per worker. ``workers <= 1`` runs in-process,
    which keeps tracebacks readable when debugging one bad mesh.

    Progress appears both as a ``tqdm`` bar and -- when ``milestone_every > 0``
    -- as flushed ``[features] N/M`` lines that survive output redirection.
    """
    n = len(jobs)
    if n == 0:
        return []
    if chunksize is None or chunksize <= 0:
        chunksize = max(1, min(256, n // (max(1, workers) * 16) or 1))
    print(f"  [features] {n} OBJs | workers={workers} | chunksize={chunksize}", flush=True)

    out: list[dict[str, Any] | None] = [None] * n
    try:
        from tqdm import tqdm

        pbar = tqdm(total=n, desc="features", unit="obj", mininterval=2.0)
    except ImportError:  # pragma: no cover
        pbar = None

    t0 = time.perf_counter()
    milestones_on = milestone_every is not None and milestone_every > 0
    next_milestone = milestone_every if milestones_on else 10**18
    n_done = 0

    def _consume(it) -> None:
        nonlocal n_done, next_milestone
        for job_idx, feats in it:
            out[job_idx] = feats
            n_done += 1
            if pbar is not None:
                pbar.update(1)
            if milestones_on and n_done >= next_milestone:
                elapsed = time.perf_counter() - t0
                rate = n_done / max(1e-9, elapsed)
                print(
                    f"  [features] {n_done}/{n} ({100.0 * n_done / n:.1f}%) | "
                    f"{rate:.0f} obj/s | elapsed {elapsed / 60.0:.1f} min | "
                    f"eta {(n - n_done) / max(1e-9, rate) / 60.0:.1f} min",
                    flush=True,
                )
                next_milestone += milestone_every

    if workers <= 1:
        _consume(worker(j) for j in jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            _consume(ex.map(worker, jobs, chunksize=chunksize))

    if pbar is not None:
        pbar.close()
    elapsed = time.perf_counter() - t0
    print(
        f"  [features] done: {n_done}/{n} in {elapsed / 60.0:.2f} min "
        f"({n_done / max(1e-9, elapsed):.0f} obj/s)",
        flush=True,
    )

    if any(v is None for v in out):
        missing = [i for i, v in enumerate(out) if v is None]
        raise RuntimeError(f"Feature extraction missing for jobs {missing[:10]}...")
    return [v for v in out if v is not None]  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _load_cache(cache_path: Path) -> dict[str, dict[str, float]]:
    """Read the per-mesh cache, dropping any row with a non-finite entry.

    Never trust a cache blindly: a run in a broken environment (e.g. without
    ``igl``) could have written non-finite values.
    """
    if not cache_path.exists():
        return {}
    try:
        df = pd.read_csv(cache_path, float_precision="round_trip")
    except Exception:
        return {}
    df.columns = df.columns.str.strip()
    if not all(c in df.columns for c in CACHE_COLUMNS):
        return {}
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        key = str(row["mesh_key"]).strip()
        if not key:
            continue
        try:
            vals = {k: float(row[k]) for k in _CACHE_FIELDS}
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(x) for x in vals.values()):
            continue
        out[key] = vals
    return out


def _lossless(value) -> str:
    """Shortest decimal string that round-trips a float64 exactly.

    ``repr`` of a Python float is round-trip exact; the ``float()`` call is
    required because ``repr`` of a *numpy* scalar renders as ``np.float64(...)``
    under NumPy 2, which would land in the CSV verbatim. pandas' default
    formatting can drop the 17th significant digit, which would make a rebuild
    differ from its own cache in the last bit.
    """
    return repr(float(value))


def _save_cache(cache_path: Path, cache: dict[str, dict[str, float]]) -> None:
    rows = [{"mesh_key": k, **v} for k, v in cache.items()]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(CACHE_COLUMNS)).to_csv(
        cache_path, index=False, float_format=_lossless
    )


def _backup_csv(csv_path: Path) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = csv_path.with_suffix(csv_path.suffix + f".bak-{stamp}")
    shutil.copy2(csv_path, backup)
    return backup


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------


def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ratios that follow algebraically from the cached integrals."""
    out = df.copy()
    g = lambda c: pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=np.float64)  # noqa: E731

    volume, M = g("volume"), g("M")
    W_vertex = g("W_vertex")
    surface_area = g("surface_area")
    V_hull, A_hull, M_hull = g("V_hull"), g("A_hull"), g("M_hull")

    R_V = np.where(
        (volume > 0.0) & np.isfinite(volume),
        (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0),
        np.nan,
    )
    R_M = np.where(np.isfinite(M), M / (4.0 * np.pi), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["R_M"] = R_M
        out["Phi_VM"] = np.where((R_M > 0) & np.isfinite(R_M), R_V / R_M, np.nan)
        out["eta_VM"] = np.where(
            (R_V > 0) & np.isfinite(R_V) & np.isfinite(R_M), R_M / R_V - 1.0, np.nan
        )
        out["W"] = W_vertex
        out["Phi_W"] = np.where(
            (W_vertex > 0) & np.isfinite(W_vertex), 4.0 * np.pi / W_vertex, np.nan
        )
        out["eta_W"] = np.where(np.isfinite(W_vertex), W_vertex / (4.0 * np.pi) - 1.0, np.nan)
        out["eta_V"] = np.where(np.isfinite(V_hull) & (V_hull > 0.0), volume / V_hull, np.nan)
        out["eta_A"] = np.where(np.isfinite(A_hull) & (A_hull > 0.0), surface_area / A_hull, np.nan)
        out["eta_M"] = np.where(np.isfinite(M_hull) & (M_hull > 0.0), M / M_hull, np.nan)
    return out


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Group the computed columns together, anchored after ``mesh_filename``."""
    new_cols = [*AREA_COLUMNS, *CURVATURE_COLUMNS, *HULL_COLUMNS, *INERTIA_COLUMNS]
    present = [c for c in new_cols if c in df.columns]
    if not present:
        return df
    rest = [c for c in df.columns if c not in present]
    anchors = [c for c in ("source", "mesh_filename", "mesh_id") if c in rest]
    if not anchors:
        return df[[*present, *rest]]
    idx = rest.index(anchors[0])
    return df[[*rest[: idx + 1], *present, *rest[idx + 1 :]]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute all shape features (area/volume, curvature integrals, "
            "convex-hull ratios, principal moments) for every mesh referenced "
            "by a dataset CSV, in a single pass per mesh."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/bubble_gym/dataset_bubblegym_10k.csv"),
        help="CSV with a 'mesh_filename' column. Updated in place unless --output.",
    )
    parser.add_argument(
        "--mesh-folder",
        type=Path,
        required=True,
        help="Folder containing the OBJ meshes named by 'mesh_filename'.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("dataset/bubble_gym/shape_feature_cache.csv"),
        help="Per-mesh cache of the raw integrals.",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Write here instead of overwriting --dataset.")
    parser.add_argument("--target-volume", type=float, default=1.0,
                        help="Rescale each mesh to this |signed volume| before integrating.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Process-pool size; 1 runs in-process (readable tracebacks).")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore the cache and recompute every mesh.")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip the timestamped backup of the dataset CSV.")
    parser.add_argument("--flush-every", type=int, default=200,
                        help="Flush the cache to disk every N newly computed meshes.")
    args = parser.parse_args()

    csv_path = args.dataset.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    mesh_folder = args.mesh_folder.resolve()
    if not mesh_folder.exists():
        raise FileNotFoundError(f"mesh folder not found: {mesh_folder}")

    df = pd.read_csv(csv_path, float_precision="round_trip")
    df.columns = df.columns.str.lstrip("#").str.strip()
    if "mesh_filename" not in df.columns:
        raise ValueError(f"{csv_path} missing 'mesh_filename'")

    names = df["mesh_filename"].astype(str).str.strip().tolist()
    keys = _mesh_keys(df)
    if len(set(keys)) != len(keys):
        raise ValueError(
            "mesh keys are not unique; cannot cache per mesh. Ensure the CSV "
            "has a unique 'mesh_id' column."
        )
    cache = {} if args.recompute else _load_cache(args.cache)
    print(f"Cache entries loaded: {len(cache)} from {args.cache}")

    todo = []
    missing = []
    for i, (key, name) in enumerate(zip(keys, names)):
        if not args.recompute and key in cache:
            continue
        resolved = _resolve_mesh_path(mesh_folder, key, name)
        if resolved is None:
            missing.append(i)
        else:
            todo.append((i, str(resolved)))
    # Re-index so job_idx is contiguous over the jobs actually submitted.
    submitted = [(k, p) for k, (_i, p) in enumerate(todo)]
    row_of_job = [i for i, _p in todo]

    failures: list[tuple[str, str]] = []
    if submitted:
        results = run_feature_pool(
            extract_features_one,
            submitted,
            workers=args.workers,
            milestone_every=args.flush_every,
        )
        for k, feats in enumerate(results):
            key = keys[row_of_job[k]]
            if feats.get("status") != "ok":
                failures.append((key, str(feats.get("status"))))
                continue
            cache[key] = {fld: float(feats[fld]) for fld in _CACHE_FIELDS}
        _save_cache(args.cache, cache)

    # Materialize columns from the cache.
    cols = {fld: np.full(len(names), np.nan, dtype=np.float64) for fld in _CACHE_FIELDS}
    for i, key in enumerate(keys):
        entry = cache.get(key)
        if entry is None:
            continue
        for fld in _CACHE_FIELDS:
            cols[fld][i] = entry[fld]
    for key, arr in cols.items():
        df[key] = arr

    df = _derive_columns(df)
    df = _reorder_columns(df)

    n_bad = int(np.count_nonzero(~np.isfinite(cols["volume"])))
    print(f"\nComputed: {len(submitted) - len(failures)} new | cached total: {len(cache)}")
    if missing:
        print(f"Missing mesh files: {len(missing)}")
    if failures:
        print(f"FAILED meshes: {len(failures)} (first 10 shown)")
        for name, why in failures[:10]:
            print(f"  [fail] {name}: {why}")
        n_holes = sum(1 for _n, w in failures if "NonWatertight" in w)
        if n_holes:
            print(
                f"  -> {n_holes} of those are not closed surfaces (holes or "
                "non-manifold edges). Shape integrals are undefined on an open "
                "surface; repair or exclude these meshes."
            )
    if n_bad:
        print(
            f"WARNING: {n_bad} of {len(names)} rows have no finite features and "
            "will be dropped by downstream trainers. Investigate before use."
        )

    target = args.output.resolve() if args.output is not None else csv_path
    if target == csv_path and not args.no_backup:
        print(f"Backup: {_backup_csv(csv_path)}")
    df.to_csv(target, index=False, float_format=_lossless)
    print(f"Wrote: {target}")


if __name__ == "__main__":
    main()

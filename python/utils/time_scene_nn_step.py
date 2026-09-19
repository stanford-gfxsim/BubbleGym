"""Time the surrogate's whole-scene frequency step, for Fig. 8.

This is the "Frequency estimation" slice of the surrogate pie: every mesh the
BEM sweep solved goes through mesh load + the eight shape features (a process
pool of ``--workers``) and then one batched call of the production model --
the same path ``scene/_common/write_trackedbubinfo_nn.py`` runs. Using the
sweep's own mesh set keeps the two frequency slices of the figure covering
identical work.

The mesh set comes from a sweep rows CSV (``<scene>_all_rows.csv``, written by
``nn8_vs_bem_all_bubbles_chunked.py``): rows with a finite, positive
``f_hz_bem``. Meshes are the pre-smoothed per-bubble OBJs those rows point at,
so no smoothing is charged, as for the BEM slice.

Protocol:
- Worker processes pin their BLAS/OpenMP pools to one thread, so ``--workers``
  is the parallelism actually used.
- The OS file cache is warmed with one untimed read of every mesh first, so a
  scene measured second does not look faster merely because it was read second.
- The model's predictions are compared with the sweep's recorded
  ``f_hz_nn8_unit``; a mismatch means the timed path is not the production one.

Run from the repository root, e.g.::

    python python/utils/time_scene_nn_step.py --rows <fruit08_all_rows.csv> \\
        --label fruit08 --workers 16 --json-out fruit08_nn_step_timing.json
"""

from __future__ import annotations

# torch before numpy/scipy/igl: on Windows with the conda stack the other order
# makes torch's shm.dll fail to load.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch as _torch  # noqa: E402,F401

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

DEFAULT_ARTIFACTS = (
    PYTHON_ROOT / "freq_model" / "output" / "output_8feature_direct_bubblegym_10k"
)
_THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
)


def _pin_worker_threads() -> None:
    for var in _THREAD_VARS:
        os.environ[var] = "1"


def _features(job):
    from shape_feature.build_dataset import extract_features_one

    return extract_features_one(job)


def _read_mesh_set(rows_csv: Path) -> tuple[list[str], np.ndarray]:
    paths, nn_ref = [], []
    with rows_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                f_bem = float(row["f_hz_bem"])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(f_bem) and f_bem > 0.0):
                continue
            paths.append(row["path"])
            try:
                nn_ref.append(float(row["f_hz_nn8_unit"]))
            except (TypeError, ValueError):
                nn_ref.append(float("nan"))
    return paths, np.asarray(nn_ref, dtype=np.float64)


def _cpu_name() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
            capture_output=True, text=True, timeout=30,
        )
        name = out.stdout.strip()
        if name:
            return name
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True, help="sweep <scene>_all_rows.csv")
    ap.add_argument("--label", required=True, help="scene label for the report")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunksize", type=int, default=64)
    ap.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    ap.add_argument("--limit", type=int, default=0, help="time only the first N meshes (0 = all)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    from freq_model.NN.nn_inference import predict_nn_unit_for_frames_inertia_8feat_direct

    paths, nn_ref = _read_mesh_set(args.rows)
    if args.limit > 0:
        paths, nn_ref = paths[: args.limit], nn_ref[: args.limit]
    missing = [p for p in paths[:50] if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"meshes not found, e.g. {missing[0]}")
    n = len(paths)
    print(f"[{args.label}] {n:,} BEM-solved meshes from {args.rows.name}", flush=True)

    t0 = time.perf_counter()
    for p in paths:
        with open(p, "rb") as fh:
            fh.read()
    print(f"[{args.label}] warmed file cache in {time.perf_counter() - t0:.1f} s (untimed)",
          flush=True)

    jobs = [(i, p) for i, p in enumerate(paths)]

    # ---- timed: load + features in the pool ------------------------------
    t_start = time.perf_counter()
    feats: list[dict] = [dict() for _ in range(n)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_pin_worker_threads) as ex:
        for idx, fe in ex.map(_features, jobs, chunksize=args.chunksize):
            feats[idx] = fe
    t_features = time.perf_counter() - t_start

    # ---- timed: one batched model call ----------------------------------
    t1 = time.perf_counter()
    inertia = [
        np.asarray(fe.get("inertia_principal_unit", [float("nan")] * 3), dtype=np.float64)
        for fe in feats
    ]
    f_unit, valid = predict_nn_unit_for_frames_inertia_8feat_direct(
        feats, inertia, args.artifacts
    )
    t_inference = time.perf_counter() - t1
    t_total = time.perf_counter() - t_start

    n_failed = sum(1 for fe in feats if fe.get("status") != "ok")
    both = valid & np.isfinite(nn_ref)
    rel = np.abs(f_unit[both] - nn_ref[both]) / nn_ref[both] if both.any() else np.array([])
    max_rel = float(rel.max()) if rel.size else float("nan")

    report = {
        "label": args.label,
        "rows_csv": args.rows.name,
        "n_meshes": n,
        "n_feature_failures": int(n_failed),
        "n_scored": int(valid.sum()),
        "workers": args.workers,
        "chunksize": args.chunksize,
        "t_load_and_features_s": round(t_features, 3),
        "t_batched_inference_s": round(t_inference, 3),
        "t_total_s": round(t_total, 3),
        "ms_per_mesh": round(1e3 * t_total / max(n, 1), 4),
        "check_max_rel_diff_vs_sweep_nn": max_rel,
        "cpu": _cpu_name(),
        "gpu": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "none",
        "inference_device": "cpu (production path)",
        "file_cache": "warmed before timing",
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

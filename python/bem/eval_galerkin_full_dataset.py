"""
Parallel P1-DP0 Galerkin Minnaert-frequency sweep over a whole dataset CSV.

Three resumable modes:

1. ``--benchmark-n N`` : run ``N`` random rows (fixed seed) and report per-row
   and projected full-dataset wallclock, RAM and scaling efficiency. Benchmark
   results land in a separate CSV so they never leak into the production
   partial file.
2. (default) : run every dataset row without a ``status == "ok"`` entry in
   ``<out-dir>/p1_dp0_partial.csv``, scheduled in descending order of a cheap
   mesh-size proxy ("f " line count in the OBJ) so the tail stays short.
3. ``--merge`` : back up the dataset CSV, add the float column
   ``bempp_p1_dp0``, and emit a coverage / error summary JSON.

Work runs in a ``ProcessPoolExecutor`` with ``mp_context="spawn"`` (mandatory
on Windows, and needed to give each worker a clean bempp-cl / OpenCL context).
Each worker's initializer caps its thread env vars, imports bempp, and runs a
tiny warm-up solve so the OpenCL JIT cost is paid once per worker.

Usage::

    python -m python.bem.eval_galerkin_full_dataset --benchmark-n 100
    python -m python.bem.eval_galerkin_full_dataset                     # full sweep
    python -m python.bem.eval_galerkin_full_dataset --retry-errors
    python -m python.bem.eval_galerkin_full_dataset --merge
"""

from __future__ import annotations

# IMPORTANT: Only stdlib / pandas / numpy / tqdm at module top level.
# DO NOT import bempp here: it must be imported inside worker processes
# AFTER their thread-cap env vars are set by the initializer.

import argparse
import csv
import json
import os
import random
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    _REPO_ROOT / "dataset" / "bubble_gym" / "dataset_bubblegym_10k.csv"
)
# The source meshes are distributed separately from this repo (see
# dataset/README.md). Point BUBBLEGYM_MESH_ROOT at them, or pass --mesh-root.
DEFAULT_MESH_ROOT = Path(
    os.environ.get(
        "BUBBLEGYM_MESH_ROOT",
        str(_REPO_ROOT / "dataset" / "bubble_gym" / "meshes10k"),
    )
)
DEFAULT_OUT_DIR = _REPO_ROOT / "results" / "bem_galerkin_full"

PARTIAL_FIELDS = [
    "mesh_filename",
    "bempp_p1_dp0",
    "capacitance",
    "capacitance_raw",
    "n_triangles",
    "n_dirichlet_dofs",
    "n_neumann_dofs",
    "gmres_iterations",
    "t_total_s",
    "status",
    "error",
]


# ---------------------------------------------------------------------------
# Cheap mesh-size proxy (no bempp needed) for descending-cost scheduling
# ---------------------------------------------------------------------------
def _count_f_lines(obj_path: Path) -> int:
    """Count 'f '-prefixed lines in an OBJ. Returns 0 if the file is missing.

    This is an order-of-magnitude proxy for the Galerkin solve cost; we do not
    need the exact triangle count for scheduling.
    """
    try:
        n = 0
        with obj_path.open("rb") as fh:
            for line in fh:
                if line.startswith(b"f "):
                    n += 1
        return n
    except FileNotFoundError:
        return 0
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Worker-side functions (run inside each spawned process)
# ---------------------------------------------------------------------------
def _worker_init(bempp_threads: int) -> None:
    """Runs once per worker process on startup.

    Sets thread-cap env vars *before* bempp / numpy kernels decide how many
    threads to use, then imports bempp and runs a tiny sphere solve to pay the
    OpenCL JIT cost once.
    """
    n = str(int(bempp_threads))
    for var in ("BEMPP_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = n
    # Discourage pocl from spawning a thread per logical core.
    os.environ.setdefault("POCL_MAX_PTHREAD_COUNT", n)
    os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "0")
    # Silence the harmless "splu converted its input to CSC format" warning.
    import warnings
    warnings.filterwarnings(
        "ignore", message=".*splu converted its input to CSC format.*"
    )

    # Warm up bempp so the first real task doesn't pay JIT latency. Any closed
    # manifold mesh serves -- the result is discarded -- so a unit octahedron is
    # built inline rather than depending on a mesh-generation module.
    try:
        import numpy as _np
        from python.bem.compute_freq_bempp_galerkin import (
            solve_minnaert_frequency_galerkin,
        )
        V = _np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                       [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])
        F = _np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                       [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
        _ = solve_minnaert_frequency_galerkin(
            (V, F),
            trial_pair="P1-DP0",
            precond="mass",
            solver="gmres",
            gmres_tol=1e-12,
            quadrature_regular=6,
            quadrature_singular=6,
            verbose=False,
        )
    except Exception:  # noqa: BLE001
        # Warm-up is best-effort: if it fails the real solves will re-raise.
        pass


def _worker_task(args: Tuple[str, str]) -> dict:
    """Run one P1-DP0 solve; never raises.

    Returns a dict with the same keys as ``PARTIAL_FIELDS`` plus diagnostic
    fields. On failure, ``status="error"`` and ``bempp_p1_dp0`` is NaN.
    """
    mesh_filename, mesh_root_str = args
    mesh_root = Path(mesh_root_str)
    mesh_path = mesh_root / mesh_filename

    t0 = time.perf_counter()
    rec = {
        "mesh_filename": mesh_filename,
        "bempp_p1_dp0": float("nan"),
        "capacitance": float("nan"),
        "capacitance_raw": float("nan"),
        "n_triangles": 0,
        "n_dirichlet_dofs": 0,
        "n_neumann_dofs": 0,
        "gmres_iterations": 0,
        "t_total_s": 0.0,
        "status": "error",
        "error": "",
    }

    try:
        if not mesh_path.is_file():
            raise FileNotFoundError(f"mesh not found: {mesh_path}")

        from python.bem.geometry_utils import load_bubble_unit_volume
        from python.bem.compute_freq_bempp_galerkin import (
            solve_minnaert_frequency_galerkin,
        )

        V, F, _scale = load_bubble_unit_volume(str(mesh_path))
        res = solve_minnaert_frequency_galerkin(
            (V, F),
            trial_pair="P1-DP0",
            precond="mass",
            solver="gmres",
            gmres_tol=1e-12,
            gmres_maxiter=4000,
            gmres_restart=300,
            quadrature_regular=6,
            quadrature_singular=6,
            verbose=False,
        )
        rec.update({
            "bempp_p1_dp0": float(res["frequency"]),
            "capacitance": float(res["capacitance"]),
            "capacitance_raw": float(res["capacitance_raw"]),
            "n_triangles": int(res["n_triangles"]),
            "n_dirichlet_dofs": int(res["n_dirichlet_dofs"]),
            "n_neumann_dofs": int(res["n_neumann_dofs"]),
            "gmres_iterations": int(res["gmres_iterations"]),
            "t_total_s": float(
                res.get("wall_time_assemble_s", 0.0)
                + res.get("wall_time_solve_s", 0.0)
            ),
            "status": "ok",
            "error": "",
        })
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["t_total_s"] = time.perf_counter() - t0

    if rec["t_total_s"] == 0.0:
        rec["t_total_s"] = time.perf_counter() - t0
    return rec


# ---------------------------------------------------------------------------
# Driver-side: CSV / resume / scheduling helpers (main process only)
# ---------------------------------------------------------------------------
def _read_partial(partial_csv: Path) -> Tuple[dict, List[str]]:
    """Return ({mesh: status}, errored_mesh_list) for resume + retry logic."""
    status_by_mesh: dict = {}
    errors: List[str] = []
    if not partial_csv.is_file():
        return status_by_mesh, errors
    with partial_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            mf = row.get("mesh_filename", "")
            st = row.get("status", "")
            if not mf:
                continue
            # Last write wins (keeps later retries over earlier errors).
            status_by_mesh[mf] = st
            if st != "ok":
                errors.append(mf)
    errors = [m for m in errors if status_by_mesh.get(m) != "ok"]
    return status_by_mesh, errors


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.is_file()
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PARTIAL_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in PARTIAL_FIELDS})


def _append_error_log(
    jsonl_path: Path, mesh_filename: str, record: dict, tb: Optional[str] = None
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mesh_filename": mesh_filename,
        "error": record.get("error", ""),
        "t_total_s": record.get("t_total_s", 0.0),
    }
    if tb:
        entry["traceback"] = tb
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


@dataclass
class _Task:
    mesh_filename: str
    size_proxy: int = 0


def _build_work_queue(
    mesh_filenames: Sequence[str],
    mesh_root: Path,
    *,
    skip: set,
    probe_size: bool = True,
    verbose: bool = True,
) -> List[_Task]:
    """Filter already-done rows, compute mesh-size proxy, sort desc."""
    todo = [m for m in mesh_filenames if m not in skip]
    if verbose:
        print(f"[plan] {len(todo)} meshes to process "
              f"(skipped {len(mesh_filenames) - len(todo)} already done)")
    if not probe_size:
        return [_Task(m, 0) for m in todo]

    tasks: List[_Task] = []
    iterator = todo
    if verbose and len(todo) > 500:
        iterator = tqdm(todo, desc="probing mesh sizes", unit="mesh")
    for m in iterator:
        sz = _count_f_lines(mesh_root / m)
        tasks.append(_Task(m, sz))
    tasks.sort(key=lambda t: t.size_proxy, reverse=True)
    return tasks


# ---------------------------------------------------------------------------
# Main sweep driver (shared between benchmark + full modes)
# ---------------------------------------------------------------------------
@dataclass
class SweepConfig:
    mesh_root: Path
    workers: int
    bempp_threads: int
    partial_csv: Path
    errors_csv: Path
    log_jsonl: Path
    progress_desc: str = "p1_dp0"
    record_errors: bool = True


def _process_pool(workers: int, bempp_threads: int) -> ProcessPoolExecutor:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(bempp_threads,),
    )


def _run_sweep(tasks: Sequence[_Task], cfg: SweepConfig) -> List[dict]:
    """Dispatch ``tasks`` to the process pool; write results as they complete."""
    if not tasks:
        print(f"[{cfg.progress_desc}] nothing to do.")
        return []

    print(
        f"[{cfg.progress_desc}] starting: "
        f"{len(tasks)} tasks, {cfg.workers} workers x "
        f"{cfg.bempp_threads} bempp-threads"
    )
    print(f"[{cfg.progress_desc}] partial -> {cfg.partial_csv}")

    results: List[dict] = []
    pool = _process_pool(cfg.workers, cfg.bempp_threads)
    t_start = time.perf_counter()

    try:
        fut_to_mesh = {
            pool.submit(
                _worker_task, (t.mesh_filename, str(cfg.mesh_root))
            ): t.mesh_filename
            for t in tasks
        }
        bar = tqdm(
            total=len(tasks), desc=cfg.progress_desc, unit="bubble", smoothing=0.1
        )
        ok_times: List[float] = []
        n_err = 0
        for fut in as_completed(fut_to_mesh):
            mf = fut_to_mesh[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 (worker died)
                tb = traceback.format_exc()
                rec = {
                    "mesh_filename": mf,
                    "bempp_p1_dp0": float("nan"),
                    "capacitance": float("nan"),
                    "capacitance_raw": float("nan"),
                    "n_triangles": 0,
                    "n_dirichlet_dofs": 0,
                    "n_neumann_dofs": 0,
                    "gmres_iterations": 0,
                    "t_total_s": 0.0,
                    "status": "error",
                    "error": f"worker crashed: {type(exc).__name__}: {exc}",
                }
                if cfg.record_errors:
                    _append_error_log(cfg.log_jsonl, mf, rec, tb=tb)

            _append_row(cfg.partial_csv, rec)
            if rec["status"] == "ok":
                ok_times.append(float(rec["t_total_s"]))
            else:
                n_err += 1
                if cfg.record_errors:
                    _append_row(cfg.errors_csv, rec)
                    _append_error_log(cfg.log_jsonl, mf, rec)
            results.append(rec)
            if ok_times:
                p50 = float(np.median(ok_times))
                p95 = float(np.percentile(ok_times, 95))
                bar.set_postfix(
                    ok=len(ok_times), err=n_err, p50=f"{p50:.1f}s", p95=f"{p95:.1f}s"
                )
            bar.update(1)
        bar.close()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    dt = time.perf_counter() - t_start
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = sum(1 for r in results if r["status"] != "ok")
    print(
        f"[{cfg.progress_desc}] done in {dt/60.0:.1f} min  "
        f"ok={n_ok}  err={n_err}  total={len(results)}"
    )
    return results


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def _run_benchmark(
    df: pd.DataFrame,
    *,
    n: int,
    workers: int,
    bempp_threads: int,
    mesh_root: Path,
    out_dir: Path,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    all_meshes = df["mesh_filename"].astype(str).tolist()
    n_eff = min(n, len(all_meshes))
    sample = rng.sample(all_meshes, n_eff)

    tasks = _build_work_queue(
        sample, mesh_root, skip=set(), probe_size=True, verbose=True
    )

    bench_dir = out_dir / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    cfg = SweepConfig(
        mesh_root=mesh_root,
        workers=workers,
        bempp_threads=bempp_threads,
        partial_csv=bench_dir / "benchmark_partial.csv",
        errors_csv=bench_dir / "benchmark_errors.csv",
        log_jsonl=bench_dir / "benchmark_log.jsonl",
        progress_desc="bench",
    )
    # Always start from scratch for benchmarks.
    if cfg.partial_csv.exists():
        cfg.partial_csv.unlink()
    if cfg.errors_csv.exists():
        cfg.errors_csv.unlink()

    t_wall0 = time.perf_counter()
    try:
        import psutil

        proc = psutil.Process()
        rss0 = proc.memory_info().rss
    except Exception:
        proc = None
        rss0 = 0

    results = _run_sweep(tasks, cfg)

    wall = time.perf_counter() - t_wall0
    ok_results = [r for r in results if r["status"] == "ok"]
    err_results = [r for r in results if r["status"] != "ok"]
    ok_times = np.array([r["t_total_s"] for r in ok_results], dtype=float)

    if ok_times.size == 0:
        print("[bench] no successful solves; aborting benchmark summary.")
        return

    p50 = float(np.median(ok_times))
    p95 = float(np.percentile(ok_times, 95))
    pmax = float(np.max(ok_times))
    cpu_s = float(np.sum(ok_times))
    n_total_rows = len(df)

    print()
    print(f"[bench] ran {len(ok_results)} ok / {len(err_results)} err / "
          f"{len(results)} total in {wall/60.0:.1f} min wallclock.")
    print(f"[bench] per-bubble: p50={p50:.2f}s  p95={p95:.2f}s  max={pmax:.2f}s")
    print(f"[bench] cumulative CPU time in successful solves = {cpu_s/3600.0:.2f} h")
    print(f"[bench] observed parallel efficiency = "
          f"{cpu_s/max(wall,1e-9):.2f}  (ideal = {workers})")
    proj_full_cpu = cpu_s * (n_total_rows / max(len(ok_results), 1))
    proj_full_wall = wall * (n_total_rows / max(len(ok_results), 1))
    print(f"[bench] projected full-dataset ({n_total_rows} rows) "
          f"cpu_time = {proj_full_cpu/3600.0:.1f} h, "
          f"wallclock @ current fan-out = {proj_full_wall/3600.0:.1f} h")
    if proc is not None:
        rss = proc.memory_info().rss
        print(f"[bench] main-proc RSS = {rss/1e9:.2f} GB (delta {(rss-rss0)/1e9:+.2f} GB)")

    summary = {
        "n_requested": n,
        "n_ok": len(ok_results),
        "n_err": len(err_results),
        "workers": workers,
        "bempp_threads": bempp_threads,
        "wallclock_s": wall,
        "per_bubble_s": {"p50": p50, "p95": p95, "max": pmax, "mean": float(np.mean(ok_times))},
        "cumulative_cpu_s": cpu_s,
        "projected_full_dataset": {
            "n_rows_total": n_total_rows,
            "cpu_hours": proj_full_cpu / 3600.0,
            "wallclock_hours": proj_full_wall / 3600.0,
        },
    }
    with (bench_dir / "benchmark_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[bench] wrote {bench_dir / 'benchmark_summary.json'}")


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------
def _run_full(
    df: pd.DataFrame,
    *,
    workers: int,
    bempp_threads: int,
    mesh_root: Path,
    out_dir: Path,
    retry_errors: bool = False,
    limit: Optional[int] = None,
) -> None:
    partial_csv = out_dir / "p1_dp0_partial.csv"
    errors_csv = out_dir / "p1_dp0_errors.csv"
    log_jsonl = out_dir / "p1_dp0_log.jsonl"

    status_by_mesh, errored = _read_partial(partial_csv)
    n_ok = sum(1 for v in status_by_mesh.values() if v == "ok")
    n_err_cached = len(errored)
    print(f"[full] checkpoint: {n_ok} ok + {n_err_cached} err "
          f"already in {partial_csv.name}")

    skip = {m for m, st in status_by_mesh.items() if st == "ok"}
    if retry_errors and errored:
        print(f"[full] --retry-errors: will retry {len(errored)} previously-failed rows.")
    else:
        skip.update(errored)  # don't retry errors unless asked

    mesh_list = df["mesh_filename"].astype(str).tolist()
    if limit is not None:
        mesh_list = mesh_list[: int(limit)]
    tasks = _build_work_queue(
        mesh_list, mesh_root, skip=skip, probe_size=True, verbose=True
    )

    if not tasks:
        print("[full] all rows already resolved. Run --merge to deposit column.")
        return

    cfg = SweepConfig(
        mesh_root=mesh_root,
        workers=workers,
        bempp_threads=bempp_threads,
        partial_csv=partial_csv,
        errors_csv=errors_csv,
        log_jsonl=log_jsonl,
        progress_desc="full",
    )
    _run_sweep(tasks, cfg)


# ---------------------------------------------------------------------------
# Merge into the dataset CSV
# ---------------------------------------------------------------------------
def _merge_into_dataset(
    dataset_csv: Path, out_dir: Path, *, in_place: bool = True
) -> None:
    partial_csv = out_dir / "p1_dp0_partial.csv"
    if not partial_csv.is_file():
        raise FileNotFoundError(f"{partial_csv} not found; nothing to merge.")

    partial = pd.read_csv(
        partial_csv,
        usecols=["mesh_filename", "bempp_p1_dp0", "status", "t_total_s"],
    )
    # Keep only latest-seen row per mesh (drop dupes that can occur on resume).
    partial = partial.drop_duplicates(subset="mesh_filename", keep="last")
    ok = partial[partial["status"] == "ok"]
    print(f"[merge] partial rows total = {len(partial)}, ok = {len(ok)}")

    dataset = pd.read_csv(dataset_csv)
    n_rows = len(dataset)
    if "bempp_p1_dp0" in dataset.columns:
        print("[merge] WARNING: column bempp_p1_dp0 already present; "
              "will be overwritten.")
        dataset = dataset.drop(columns=["bempp_p1_dp0"])

    merged = dataset.merge(
        ok[["mesh_filename", "bempp_p1_dp0"]], on="mesh_filename", how="left"
    )
    n_filled = int(merged["bempp_p1_dp0"].notna().sum())
    n_missing = n_rows - n_filled
    print(f"[merge] dataset rows = {n_rows},  "
          f"filled = {n_filled},  missing = {n_missing}")

    target = dataset_csv if in_place else dataset_csv.with_name(
        dataset_csv.stem + "_with_p1_dp0.csv"
    )
    if in_place:
        bak = dataset_csv.with_name(dataset_csv.stem + "_pre_p1dp1.csv.bak")
        if not bak.exists():
            shutil.copy2(dataset_csv, bak)
            print(f"[merge] backup -> {bak}")
        else:
            print(f"[merge] backup already exists at {bak}; not overwriting it.")

    tmp = target.with_suffix(target.suffix + ".tmp")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, target)
    print(f"[merge] wrote {target}")

    missing_rows = merged.loc[merged["bempp_p1_dp0"].isna(), "mesh_filename"].tolist()
    summary = {
        "dataset_csv": str(target),
        "n_rows_total": n_rows,
        "n_filled": n_filled,
        "n_missing": n_missing,
        "missing_mesh_filenames": missing_rows[:200],  # cap for readability
    }
    with (out_dir / "p1_dp0_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[merge] wrote {out_dir / 'p1_dp0_summary.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--bempp-threads", type=int, default=2)
    ap.add_argument(
        "--benchmark-n", type=int, default=0,
        help="If >0, run this many random rows in benchmark mode and exit.",
    )
    ap.add_argument("--benchmark-seed", type=int, default=42)
    ap.add_argument(
        "--retry-errors", action="store_true",
        help="Also re-run rows currently marked 'error' in the partial CSV.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of dataset rows (debug).",
    )
    ap.add_argument(
        "--merge", action="store_true",
        help="Merge p1_dp0_partial.csv into --dataset and exit.",
    )
    ap.add_argument(
        "--merge-copy", action="store_true",
        help="With --merge: write a new <dataset>_with_p1_dp0.csv instead of "
             "overwriting --dataset.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        _merge_into_dataset(
            args.dataset.resolve(), args.out_dir.resolve(), in_place=not args.merge_copy
        )
        return

    import sys as _sys

    if str(_REPO_ROOT / "python") not in _sys.path:
        _sys.path.insert(0, str(_REPO_ROOT / "python"))
    from utils.mesh_archive import require_mesh_root

    require_mesh_root(args.mesh_root)

    print(f"[main] dataset     = {args.dataset}")
    print(f"[main] mesh_root   = {args.mesh_root}")
    print(f"[main] out_dir     = {args.out_dir}")
    print(f"[main] workers     = {args.workers}")
    print(f"[main] bempp_threads = {args.bempp_threads}")

    df = pd.read_csv(args.dataset, usecols=["mesh_filename"])

    if args.benchmark_n and args.benchmark_n > 0:
        _run_benchmark(
            df,
            n=args.benchmark_n,
            workers=args.workers,
            bempp_threads=args.bempp_threads,
            mesh_root=args.mesh_root.resolve(),
            out_dir=args.out_dir.resolve(),
            seed=args.benchmark_seed,
        )
        return

    _run_full(
        df,
        workers=args.workers,
        bempp_threads=args.bempp_threads,
        mesh_root=args.mesh_root.resolve(),
        out_dir=args.out_dir.resolve(),
        retry_errors=args.retry_errors,
        limit=args.limit,
    )


if __name__ == "__main__":
    # Mandatory on Windows + spawn; also keeps imports clean for test harnesses.
    _main()

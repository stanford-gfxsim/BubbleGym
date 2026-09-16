"""Resumable Galerkin BEM sweep over a dataset CSV — the dataset build driver.

This is what produced the ground-truth frequencies in the shipped benchmark. It
walks the rows of a CSV of meshes, solves each one, and writes the frequency
back into the same CSV in place, alongside per-row capacitance and timing.

Resumable by design, because a full sweep runs for hours: rows that already
carry a finite positive ``frequency_bem_galerkin_hz`` are skipped, so an
interrupted run continues where it stopped. ``--start-row`` overrides the
starting point.

Its defaults are the dataset profile the paper's supplement states — P1-DP0,
mass-preconditioned GMRES to 1e-12 with maxiter 4000 and restart 300,
quadrature 6/6 — deliberately tighter than the solver's own defaults, since
these numbers are ground truth.

Point ``--csv`` at the mesh list you want solved; there is nothing scene-specific
about the driver.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.bem.compute_freq_bempp_galerkin import solve_minnaert_frequency_galerkin  # noqa: E402


# No default mesh list: the driver is scene-agnostic, and silently sweeping the
# wrong CSV in place is worse than being asked for one.
DEFAULT_CSV = None
FREQ_COLUMN = "frequency_bem_galerkin_hz"
BEM_TIME_COLUMNS = [
    "row_index",
    "mesh_file",
    "mesh_path",
    "elapsed_s",
    "frequency_hz",
    "status",
]
DIAGNOSTIC_COLUMNS = [
    "bem_galerkin_capacitance",
    "bem_galerkin_capacitance_raw",
    "bem_galerkin_gmres_info",
    "bem_galerkin_gmres_iterations",
    "bem_galerkin_n_triangles",
    "bem_galerkin_wall_time_s",
    "bem_galerkin_status",
]


def _finite_positive(value: object) -> bool:
    try:
        val = float(str(value).strip())
    except Exception:
        return False
    return math.isfinite(val) and val > 0.0


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if FREQ_COLUMN not in df.columns:
        df[FREQ_COLUMN] = ""
    df[FREQ_COLUMN] = df[FREQ_COLUMN].astype(object)
    for col in DIAGNOSTIC_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)
    return df


def _backup_csv(path: Path, backup_dir: Path | None) -> Path:
    if backup_dir is None:
        backup_dir = path.parent
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = backup_dir / f"{path.stem}.backup_{stamp}{path.suffix}"
    shutil.copy2(path, backup)
    return backup


def _write_checkpoint(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _load_bem_time_df(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if path.is_file():
        out = pd.read_csv(path)
        out.columns = out.columns.str.lstrip("#").str.strip()
        for col in BEM_TIME_COLUMNS:
            if col in out.columns:
                out[col] = out[col].astype(object)
        return out
    out = pd.DataFrame(columns=BEM_TIME_COLUMNS)
    for col in BEM_TIME_COLUMNS:
        out[col] = out[col].astype(object)
    return out


def _write_bem_time_row(
    bem_time_df: pd.DataFrame | None,
    path: Path | None,
    *,
    row_index: int,
    mesh_file: str,
    mesh_path: Path,
    elapsed_s: float,
    frequency_hz: float | None,
    status: str,
) -> pd.DataFrame | None:
    if bem_time_df is None or path is None:
        return bem_time_df
    if path.is_file():
        on_disk = _load_bem_time_df(path)
        if on_disk is not None and len(on_disk):
            bem_time_df = on_disk
    row = {
        "row_index": int(row_index),
        "mesh_file": mesh_file,
        "mesh_path": str(mesh_path),
        "elapsed_s": float(elapsed_s),
        "frequency_hz": np.nan if frequency_hz is None else float(frequency_hz),
        "status": status,
    }
    if len(bem_time_df) and "row_index" in bem_time_df.columns:
        mask = pd.to_numeric(bem_time_df["row_index"], errors="coerce") == int(row_index)
        if bool(mask.any()):
            for key, val in row.items():
                bem_time_df.loc[mask, key] = val
        else:
            bem_time_df.loc[len(bem_time_df)] = row
    else:
        bem_time_df.loc[len(bem_time_df)] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(bem_time_df, path)
    return bem_time_df


def _apply_success(df: pd.DataFrame, row_idx: int, out: dict) -> None:
    df.at[row_idx, FREQ_COLUMN] = f"{float(out['frequency']):.10g}"
    df.at[row_idx, "bem_galerkin_capacitance"] = f"{float(out['capacitance']):.10g}"
    df.at[row_idx, "bem_galerkin_capacitance_raw"] = f"{float(out['capacitance_raw']):.10g}"
    df.at[row_idx, "bem_galerkin_gmres_info"] = str(int(out["gmres_info"]))
    df.at[row_idx, "bem_galerkin_gmres_iterations"] = str(int(out["gmres_iterations"]))
    df.at[row_idx, "bem_galerkin_n_triangles"] = str(int(out["n_triangles"]))
    wall_time = float(out.get("wall_time_assemble_s", 0.0)) + float(out.get("wall_time_solve_s", 0.0))
    df.at[row_idx, "bem_galerkin_wall_time_s"] = f"{wall_time:.6g}"
    df.at[row_idx, "bem_galerkin_status"] = "ok"


def _apply_failure(df: pd.DataFrame, row_idx: int, exc: Exception) -> None:
    df.at[row_idx, "bem_galerkin_status"] = f"failed: {type(exc).__name__}: {exc}"


def run(args: argparse.Namespace) -> None:
    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    backup = None
    if not args.no_backup:
        backup = _backup_csv(csv_path, args.backup_dir)
        print(f"Backed up current CSV to: {backup}", flush=True)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lstrip("#").str.strip()
    if "mesh_path" not in df.columns:
        raise ValueError(f"{csv_path} missing required column 'mesh_path'")
    df = _ensure_columns(df)

    solved = df[FREQ_COLUMN].map(_finite_positive)
    first_remaining = int(np.flatnonzero(~solved)[0]) if np.any(~solved) else len(df)
    start_row = first_remaining if args.start_row is None else int(args.start_row)
    if start_row < 0 or start_row > len(df):
        raise ValueError(f"--start-row {start_row} out of range for {len(df)} rows")

    pending_indices = [i for i in range(start_row, len(df)) if not _finite_positive(df.at[i, FREQ_COLUMN])]
    if args.limit is not None:
        pending_indices = pending_indices[: max(0, int(args.limit))]

    print(f"Input/output CSV: {csv_path}", flush=True)
    print(f"Rows total: {len(df)}", flush=True)
    print(f"Already solved: {int(solved.sum())}", flush=True)
    print(f"First remaining row: {first_remaining}", flush=True)
    print(f"Starting row: {start_row}", flush=True)
    print(f"Scheduled solves: {len(pending_indices)}", flush=True)
    print(
        "Solver: "
        f"trial_pair={args.trial_pair}, precond={args.precond}, solver={args.solver}, "
        f"gmres_tol={args.gmres_tol:g}, gmres_maxiter={args.gmres_maxiter}, "
        f"quad=({args.quad_regular},{args.quad_singular})",
        flush=True,
    )

    bem_time_path = args.bem_time_csv.resolve() if args.bem_time_csv is not None else None
    bem_time_df = _load_bem_time_df(bem_time_path)
    if bem_time_path is not None:
        print(f"BEM timing log: {bem_time_path}", flush=True)

    if not pending_indices:
        print("Nothing to solve.", flush=True)
        return

    n_ok = 0
    n_failed = 0
    t_all = time.perf_counter()
    for task_i, row_idx in enumerate(pending_indices, start=1):
        mesh_path = Path(str(df.at[row_idx, "mesh_path"]))
        mesh_file = str(df.at[row_idx, "mesh_file"]) if "mesh_file" in df.columns else mesh_path.name
        print(
            f"[{task_i}/{len(pending_indices)}] row={row_idx} mesh={mesh_file} path={mesh_path}",
            flush=True,
        )
        t0 = time.perf_counter()
        solve_ok = False
        freq: float | None = None
        fail_status = ""
        try:
            if not mesh_path.is_file():
                raise FileNotFoundError(mesh_path)
            out = solve_minnaert_frequency_galerkin(
                mesh_path,
                trial_pair=args.trial_pair,
                precond=args.precond,
                solver=args.solver,
                gmres_tol=args.gmres_tol,
                gmres_maxiter=args.gmres_maxiter,
                gmres_restart=args.gmres_restart,
                quadrature_regular=args.quad_regular,
                quadrature_singular=args.quad_singular,
                gamma=args.gamma,
                p0=args.p0,
                rho=args.rho,
                verbose=args.verbose_solver,
            )
            freq = float(out["frequency"])
            if not (math.isfinite(freq) and freq > 0.0):
                raise RuntimeError(f"non-finite/non-positive frequency: {freq}")
            _apply_success(df, row_idx, out)
            n_ok += 1
            solve_ok = True
        except Exception as exc:  # noqa: BLE001
            _apply_failure(df, row_idx, exc)
            n_failed += 1
            fail_status = f"failed: {type(exc).__name__}: {exc}"
            print(f"  failed after {time.perf_counter() - t0:.2f}s: {fail_status}", flush=True)

        elapsed = time.perf_counter() - t0
        if solve_ok:
            print(f"  ok: f={freq:.9g} Hz in {elapsed:.2f}s", flush=True)
        if bem_time_path is not None:
            try:
                bem_time_df = _write_bem_time_row(
                    bem_time_df,
                    bem_time_path,
                    row_index=row_idx,
                    mesh_file=mesh_file,
                    mesh_path=mesh_path,
                    elapsed_s=elapsed,
                    frequency_hz=freq if solve_ok else None,
                    status="ok" if solve_ok else fail_status,
                )
            except Exception as log_exc:  # noqa: BLE001
                print(f"  warn: could not update BEM_time.csv: {log_exc}", flush=True)

        if task_i % max(1, args.checkpoint_every) == 0:
            _write_checkpoint(df, csv_path)
            print(f"  checkpoint written after {task_i} scheduled rows", flush=True)

    _write_checkpoint(df, csv_path)
    solved_after = int(df[FREQ_COLUMN].map(_finite_positive).sum())
    print(
        f"Done. ok={n_ok}, failed={n_failed}, solved_after={solved_after}/{len(df)}, "
        f"elapsed={(time.perf_counter() - t_all) / 60.0:.2f} min",
        flush=True,
    )
    if backup is not None:
        print(f"Backup preserved at: {backup}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV, required=DEFAULT_CSV is None,
        help="CSV of meshes to solve; the frequency column is written back in place.",
    )
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--start-row", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--bem-time-csv",
        type=Path,
        default=None,
        help="Append per-bubble wall times (matches 'ok: ... in X.XXs' log lines).",
    )
    parser.add_argument("--trial-pair", default="P1-DP0", choices=["DP0-DP0", "P1-DP0", "P1-DP1"])
    parser.add_argument("--precond", default="mass", choices=["none", "mass", "calderon"])
    parser.add_argument("--solver", default="gmres", choices=["gmres", "dense"])
    parser.add_argument("--gmres-tol", type=float, default=1e-12)
    parser.add_argument("--gmres-maxiter", type=int, default=4000)
    parser.add_argument("--gmres-restart", type=int, default=300)
    parser.add_argument("--quad-regular", type=int, default=6)
    parser.add_argument("--quad-singular", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--p0", type=float, default=101325.0)
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--verbose-solver", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

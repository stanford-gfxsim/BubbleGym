"""
Precompute and cache Minnaert and Strasberg baseline frequencies as new columns
in a bubble dataset CSV, once per bubble. Downstream trainers and the eval /
viewer stack read these columns directly at training/inference time instead of
recomputing the elliptic-integral every iteration.

Design:

- Reads volume + principal inertia moments (i00, i11, i22) per row.
- Computes
    f_minnaert = 3.283 (MINNAERT_CONSTANT = 3.283243423687599) / R  with R = (3 V / 4 pi)^(1/3) on unit-volume data,
                 or equivalently  f_min_unit * V^(-1/3)  (see helpers below).
    f_strasberg = strasberg_frequency_unit_volume(I) * V^(-1/3)
- Writes columns `f_minnaert` and `f_strasberg` in place into the CSV.
- Idempotent: re-running with both columns already present is a no-op.
- Emits a short report to stdout with the residual stddev under each baseline
  so you can confirm the Strasberg residual is smaller BEFORE retraining.

Typical usage:
    python python/freq_model/NN/add_baseline_frequencies_to_dataset.py \
        --input dataset/bubble_gym/dataset_bubblegym_10k.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
_PYTHON_ROOT = _ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from freq_model.analytical.minnaert_freq import minnaert_frequency_unit_volume  # noqa: E402
from freq_model.analytical.strasberg_freq import strasberg_frequency_unit_volume  # noqa: E402


REQUIRED_COLS = ["volume", "i00", "i11", "i22", "frequency"]
F_MINNAERT_COL = "f_minnaert"
F_STRASBERG_COL = "f_strasberg"


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.lstrip("#").str.strip()
    return out


def _has_finite_column(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    return bool(np.all(np.isfinite(v)) and np.all(v > 0.0))


def compute_baseline_columns(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return (f_minnaert_arr, f_strasberg_arr, strasberg_seconds) for every row.

    Minnaert is closed-form; Strasberg uses the unit-volume rescale workaround
    from strasberg_frequency_unit_volume, then scales back to the real volume
    via f ~ V^(-1/3). Any row where Strasberg raises is written as NaN; the
    caller reports the count but we don't drop rows here.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Have: {sorted(df.columns.tolist())}"
        )

    volume = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=np.float64)
    i00 = pd.to_numeric(df["i00"], errors="coerce").to_numpy(dtype=np.float64)
    i11 = pd.to_numeric(df["i11"], errors="coerce").to_numpy(dtype=np.float64)
    i22 = pd.to_numeric(df["i22"], errors="coerce").to_numpy(dtype=np.float64)

    n = len(df)
    f_minnaert = np.full(n, np.nan, dtype=np.float64)
    f_strasberg = np.full(n, np.nan, dtype=np.float64)

    # Closed-form Minnaert: f(V) = f_unit * V^(-1/3).
    f_min_unit = minnaert_frequency_unit_volume()
    valid_v = np.isfinite(volume) & (volume > 0.0)
    f_minnaert[valid_v] = f_min_unit * np.power(volume[valid_v], -1.0 / 3.0)

    # Strasberg: one elliptic-integral eval per row.
    t0 = time.perf_counter()
    strasberg_failures = 0
    for k in range(n):
        I = np.array([i00[k], i11[k], i22[k]], dtype=np.float64)
        V = float(volume[k])
        if not (np.all(np.isfinite(I)) and np.all(I > 0.0) and np.isfinite(V) and V > 0.0):
            strasberg_failures += 1
            continue
        try:
            f_unit = strasberg_frequency_unit_volume(I)
            f_strasberg[k] = f_unit * (V ** (-1.0 / 3.0))
        except Exception:
            strasberg_failures += 1
            f_strasberg[k] = np.nan
    t_strasberg = time.perf_counter() - t0

    if strasberg_failures > 0:
        print(
            f"  [warn] Strasberg failed on {strasberg_failures}/{n} rows "
            f"(inertia tensor not ellipsoid-consistent or non-finite inputs)."
        )

    return f_minnaert, f_strasberg, t_strasberg


def add_baseline_columns_in_place(
    csv_path: Path,
    *,
    force: bool = False,
) -> dict[str, float]:
    """
    Add f_minnaert and f_strasberg columns to `csv_path`, writing in place.

    Returns a dict with counts and residual stddev diagnostics so callers
    (including the CLI main) can print / log the same summary.
    """
    csv_path = csv_path.resolve()
    df = pd.read_csv(csv_path)
    df = _canonicalize_columns(df)

    already_have_both = (
        not force
        and _has_finite_column(df, F_MINNAERT_COL)
        and _has_finite_column(df, F_STRASBERG_COL)
    )
    if already_have_both:
        print(
            f"[skip] '{F_MINNAERT_COL}' and '{F_STRASBERG_COL}' already present "
            f"with finite values in {csv_path.name}; no work to do."
        )
        t_strasberg = 0.0
    else:
        f_minnaert, f_strasberg, t_strasberg = compute_baseline_columns(df)
        df[F_MINNAERT_COL] = f_minnaert
        df[F_STRASBERG_COL] = f_strasberg
        df.to_csv(csv_path, index=False)
        print(f"[ok] wrote {F_MINNAERT_COL} and {F_STRASBERG_COL} to {csv_path.name}")

    # Residual-stddev report (always computed, even when no work was done).
    f_bem = pd.to_numeric(df["frequency"], errors="coerce").to_numpy(dtype=np.float64)
    f_m = pd.to_numeric(df[F_MINNAERT_COL], errors="coerce").to_numpy(dtype=np.float64)
    f_s = pd.to_numeric(df[F_STRASBERG_COL], errors="coerce").to_numpy(dtype=np.float64)
    mask = (
        np.isfinite(f_bem) & (f_bem > 0.0)
        & np.isfinite(f_m) & (f_m > 0.0)
        & np.isfinite(f_s) & (f_s > 0.0)
    )
    r_m = np.log(f_bem[mask]) - np.log(f_m[mask])
    r_s = np.log(f_bem[mask]) - np.log(f_s[mask])
    sigma_m = float(np.std(r_m, ddof=0)) if r_m.size > 1 else float("nan")
    sigma_s = float(np.std(r_s, ddof=0)) if r_s.size > 1 else float("nan")

    return {
        "n_rows_total": int(len(df)),
        "n_rows_valid": int(mask.sum()),
        "strasberg_seconds": float(t_strasberg),
        "sigma_minnaert": sigma_m,
        "sigma_strasberg": sigma_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cache f_minnaert and f_strasberg per row into a dataset CSV."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/bubble_gym/dataset_bubblegym_10k.csv"),
        help="CSV to augment in place (default: the shipped "
             "dataset/bubble_gym/dataset_bubblegym_10k.csv, relative to the repo root).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute columns even if they already exist.",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.input}")

    print(f"=== Caching baseline frequencies into {args.input} ===")
    stats = add_baseline_columns_in_place(args.input, force=args.force)

    n_total = stats["n_rows_total"]
    n_valid = stats["n_rows_valid"]
    t_s = stats["strasberg_seconds"]
    sigma_m = stats["sigma_minnaert"]
    sigma_s = stats["sigma_strasberg"]
    ratio = (sigma_s / sigma_m) if (sigma_m and np.isfinite(sigma_m) and sigma_m > 0) else float("nan")

    print("")
    print(f"Processed: {n_total} bubbles ({n_valid} with finite f_bem / f_min / f_stras)")
    print(f"Strasberg preprocessing wall time: {t_s:.3f} s")
    print("Residual stddev (log-space):")
    print(f"  Minnaert baseline:  sigma = {sigma_m:.6f}")
    print(f"  Strasberg baseline: sigma = {sigma_s:.6f}")
    print(f"  ratio sigma_strasberg / sigma_minnaert = {ratio:.4f}")
    if np.isfinite(ratio) and ratio < 1.0:
        print("  -> Strasberg residual is smaller; safe to retrain with Strasberg baseline.")
    elif np.isfinite(ratio):
        print("  -> WARNING: Strasberg residual is NOT smaller than Minnaert. Investigate before retraining.")


if __name__ == "__main__":
    main()

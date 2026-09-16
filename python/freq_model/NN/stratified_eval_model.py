"""Stratified per-bin evaluation of the shipped NN8 frequency surrogate.

Bins the dataset into quantiles of a non-sphericity scalar, samples a balanced
set per bin (JND thinning in frequency, then farthest-point sampling), and
reports MAPE / max APE / RMSE(log f) per bin for three predictors: Minnaert,
Strasberg, and the network. Feeds visualization/plot_per_bin_mape_bars.py.

Verified provenance of the shipped 10x10 ablation panel: it was built with
``--non-sphericity-metric inertia_aniso --per-bin 10``, NOT with the Wadell
``phi`` metric the paper's Fig. 4 text calls for. Confirmed by matching
``bin_edges_selected`` in the shipped ``curated_eval.json`` against the ten
quantile edges of inertia anisotropy over the 1500-row test split: min 0.0329
and max 43.676 reproduce exactly. The defaults below are deliberately left at
the values that produced the shipped panel -- do not "fix" them to match the
paper text without regenerating every downstream artifact.
"""

from __future__ import annotations

# Import torch BEFORE joblib/matplotlib to avoid a Windows DLL conflict where
# joblib's loky/cloudpickle deps lock an OpenMP runtime that prevents torch's
# shm.dll from loading later. See OSError "Error loading ... torch/lib/shm.dll".
import torch

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_PYTHON_ROOT = _THIS_DIR.parents[1]
for _p in (_PYTHON_ROOT, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from freq_model.NN.bub_freq_net import BubbleFreqNet
from freq_model.NN.nn_inference import EXPECTED_FEATURE_COLS_8FEAT
from freq_model.analytical.strasberg_freq import strasberg_frequency_unit_volume
from utils.worst_case_utils import export_worst_cases, find_thumbnail_path, sanitize_filename


# The shipped production checkpoint (Eq. 16), trained on the 10k benchmark.
NN8_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "output_8feature_direct_bubblegym_10k"
)

INERTIA_ONLY_REQUIRED_COLS = ["i00", "i11", "i22"]
TARGET_COL = "frequency"


@dataclass(frozen=True)
class EvalRow:
    idx: int
    non_sphericity: float
    freq_hz: float


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.lstrip("#").str.strip()
    return out


def _require_cols(df: pd.DataFrame, cols: list[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx}: missing required columns: {missing}")


def _filter_by_split(df: pd.DataFrame, split_json: Path, split: str) -> pd.DataFrame:
    """
    Filter dataset to a saved train/val/test split.

    Prefers `mesh_id` (e.g. "VOF/bubble.0019.0.obj"), which is unique
    across the combined Tim2016 + LBM datasets, and falls back to
    `mesh_filename` for legacy splits.

    This ensures stratified eval samples only from the held-out test set
    (recommended), instead of the full dataset (which would mix training and
    test samples).
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("--split must be one of: train, val, test")
    with split_json.open(encoding="utf-8") as f:
        sj = json.load(f)

    mesh_id_key = f"mesh_id_{split}"
    if mesh_id_key in sj and "mesh_id" in df.columns:
        allowed = {str(x) for x in sj[mesh_id_key]}
        out = df[df["mesh_id"].astype(str).isin(allowed)].reset_index(drop=True)
        return out

    mesh_name_key = f"mesh_filename_{split}"
    if mesh_name_key not in sj:
        raise ValueError(
            f"split.json missing keys '{mesh_id_key}' and '{mesh_name_key}'. "
            "Re-train with updated trainer to generate them."
        )
    _require_cols(df, ["mesh_filename"], "dataset for split filter")
    allowed = {str(x) for x in sj[mesh_name_key]}
    out = df[df["mesh_filename"].astype(str).isin(allowed)].reset_index(drop=True)
    return out


def compute_radius_from_volume(df: pd.DataFrame, volume_col: str = "volume", default_volume: float = 1.0) -> np.ndarray:
    """
    If volume is normalized to 1 (your dataset), radius can be computed directly:
      R = (3V / (4π))^(1/3)

    - If `volume_col` exists, uses it (numeric-coerced).
    - Otherwise falls back to `default_volume` (default: 1.0).
    """
    if volume_col in df.columns:
        v = pd.to_numeric(df[volume_col], errors="coerce").to_numpy(dtype=np.float64)
    else:
        v = np.full(len(df), float(default_volume), dtype=np.float64)
    v = np.where(np.isfinite(v) & (v > 0.0), v, np.nan)
    r = np.cbrt((3.0 * v) / (4.0 * math.pi))
    r = np.where(np.isfinite(r) & (r > 1e-12), r, np.nan)
    return r


def minnaert_frequency_from_radius(radius: np.ndarray) -> np.ndarray:
    # 3.26 rather than the 3.2832 of analytical/minnaert_freq.py: this is the
    # historical constant the shipped stratified-eval artifacts were produced
    # with, kept so the Minnaert baseline column stays reproducible.
    return 3.26 / np.maximum(radius, 1e-8)


# The production head targets log(f_BEM) directly, so its log-baseline is 0.
_DIRECT_MODEL_KINDS = {"nn8"}


def _baseline_log_for_rows(model_kind: str, df_rows: pd.DataFrame) -> np.ndarray:
    """Return per-row log-baseline array used to reconstruct log(f) from the network's
    z-scored output: pred_log = pred_log_residual + log_baseline.

    The shipped head is direct, so this returns zeros for "nn8". The Strasberg
    fallback is kept for the log-residual convention that ``training.py`` still
    supports; it is unreachable while ``--model-kind`` only offers "nn8".
    """
    if model_kind in _DIRECT_MODEL_KINDS:
        return np.zeros(len(df_rows), dtype=np.float64)
    f_str = strasberg_frequency_from_inertia_unit_volume(df_rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(f_str)


def strasberg_frequency_from_inertia_unit_volume(df: pd.DataFrame) -> np.ndarray:
    """
    Strasberg baseline at unit volume from volume-normalized inertia.

    Thin wrapper around `strasberg_frequency_unit_volume` (the canonical
    implementation of the rescale-to-ellipsoid-consistent-volume workaround).
    This function exists only to batch-apply the helper over a dataframe row
    by row and swallow per-row failures to NaN so callers can `.isfinite()`.
    """
    _require_cols(df, ["i00", "i11", "i22"], "strasberg baseline")
    i00 = pd.to_numeric(df["i00"], errors="coerce").to_numpy(dtype=np.float64)
    i11 = pd.to_numeric(df["i11"], errors="coerce").to_numpy(dtype=np.float64)
    i22 = pd.to_numeric(df["i22"], errors="coerce").to_numpy(dtype=np.float64)
    out = np.full(len(df), np.nan, dtype=np.float64)
    for k in range(len(df)):
        I = np.array([i00[k], i11[k], i22[k]], dtype=np.float64)
        if not np.all(np.isfinite(I)) or np.any(I <= 0):
            continue
        try:
            out[k] = strasberg_frequency_unit_volume(I)
        except Exception:
            continue
    return out


def sphericity_phi_wadell(
    df: pd.DataFrame,
    volume_col: str = "volume",
) -> np.ndarray:
    """
    Wadell sphericity Phi in (0, 1]; Phi = 1 for a sphere:

      Phi = pi^(1/3) * (6*V)^(2/3) / S

    V and S must come from the same geometry (consistent units): the benchmark CSV
    measures both after the mesh has been scaled to unit volume.
    """
    _require_cols(df, ["surface_area"], "Wadell sphericity Phi")
    if volume_col in df.columns:
        v = pd.to_numeric(df[volume_col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.any(np.isfinite(v) & (v > 0.0)):
            raise ValueError(
                f"phi metric: column '{volume_col}' is missing valid positive values. "
                "Regenerate the dataset with python/shape_feature/build_dataset.py."
            )
    else:
        raise ValueError(
            "phi metric requires a numeric `volume` column alongside `surface_area`."
        )
    s = pd.to_numeric(df["surface_area"], errors="coerce").to_numpy(dtype=np.float64)
    v = np.where(np.isfinite(v) & (v > 0.0), v, np.nan)
    s = np.where(np.isfinite(s) & (s > 0.0), s, np.nan)
    pi_cbrt = math.pi ** (1.0 / 3.0)
    phi = pi_cbrt * np.power(6.0 * v, 2.0 / 3.0) / np.maximum(s, 1e-24)
    phi = np.where(np.isfinite(phi) & (phi > 0.0), phi, np.nan)
    return phi


def non_sphericity_from_phi(phi: np.ndarray) -> np.ndarray:
    """``1 - Phi``: larger means less spherical.

    Not clipped. A sphericity outside (0, 1] is a defect in the underlying
    mesh, not something to squash into range, so it propagates as NaN and is
    excluded from the bins.
    """
    phi = np.asarray(phi, dtype=np.float64)
    valid = np.isfinite(phi) & (phi > 0.0) & (phi <= 1.0 + 1e-9)
    return np.where(valid, 1.0 - phi, np.nan)


def non_sphericity_inertia_anisotropy(df: pd.DataFrame) -> np.ndarray:
    """
    Simple inertia-based non-sphericity proxy:
      a = sqrt((i11/i00 - 1)² + (i22/i00 - 1)²)
    For a sphere in principal frame, ratios are ~1.
    """
    _require_cols(df, INERTIA_ONLY_REQUIRED_COLS, "inertia non-sphericity")
    i00 = pd.to_numeric(df["i00"], errors="coerce").to_numpy(dtype=np.float64)
    i11 = pd.to_numeric(df["i11"], errors="coerce").to_numpy(dtype=np.float64)
    i22 = pd.to_numeric(df["i22"], errors="coerce").to_numpy(dtype=np.float64)
    denom = np.maximum(i00, 1e-18)
    i11_hat = i11 / denom
    i22_hat = i22 / denom
    a = np.sqrt((i11_hat - 1.0) ** 2 + (i22_hat - 1.0) ** 2)
    a = np.where(np.isfinite(a) & (a >= 0.0), a, np.nan)
    return a


def add_nonsphericity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add only the three Wadell-style non-sphericity axes
    (``non_sph_va``, ``non_sph_vm``, ``non_sph_w``).
    """
    _require_cols(
        df,
        ["surface_area", "volume", "Phi_VM", "W_vertex"],
        "nonsphericity features",
    )
    out = df.copy()
    phi_va = sphericity_phi_wadell(out, volume_col="volume")
    phi_vm = pd.to_numeric(out["Phi_VM"], errors="coerce").to_numpy(dtype=np.float64)
    w_vertex = pd.to_numeric(out["W_vertex"], errors="coerce").to_numpy(dtype=np.float64)
    out["non_sph_va"] = non_sphericity_from_phi(phi_va)
    out["non_sph_vm"] = 1.0 - phi_vm
    out["non_sph_w"] = 1.0 - (4.0 * math.pi / w_vertex)
    return out


def add_nn8_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the eight features of the production model (Eq. 11).

    The six non-sphericity + convex-hull axes from
    :func:`add_nonsphericity_chull_features`, plus the two volume-normalized
    inertia ratios of Eq. 6 derived from the dataset's ``i00,i11,i22`` columns.
    Column order is fixed by ``EXPECTED_FEATURE_COLS_8FEAT`` -- the same order
    the trainer used and the scalers were fit on.
    """
    out = add_nonsphericity_chull_features(df)
    _require_cols(out, ["i00", "i11", "i22"], "nn8 inertia ratios")
    i00 = pd.to_numeric(out["i00"], errors="coerce").to_numpy(dtype=np.float64)
    i11 = pd.to_numeric(out["i11"], errors="coerce").to_numpy(dtype=np.float64)
    i22 = pd.to_numeric(out["i22"], errors="coerce").to_numpy(dtype=np.float64)
    denom = np.maximum(i00, 1e-18)
    out = out.copy()
    out["i11_over_i00"] = i11 / denom
    out["i22_over_i00"] = i22 / denom
    return out


def add_nonsphericity_chull_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add six features: the three non-sphericity axes plus the three
    convex-hull ratios. Used as the base of :func:`add_nn8_features`.

    Requires the convex-hull eta columns to already exist on the row (add them
    with ``python/shape_feature/build_dataset.py``).
    """
    _require_cols(
        df,
        ["surface_area", "volume", "Phi_VM", "W_vertex", "eta_V", "eta_A", "eta_M"],
        "nonsphericity_chull features",
    )
    out = add_nonsphericity_features(df)
    out["eta_V"] = pd.to_numeric(out["eta_V"], errors="coerce").to_numpy(dtype=np.float64)
    out["eta_A"] = pd.to_numeric(out["eta_A"], errors="coerce").to_numpy(dtype=np.float64)
    out["eta_M"] = pd.to_numeric(out["eta_M"], errors="coerce").to_numpy(dtype=np.float64)
    return out


def non_sphericity_chull_dendritic(df: pd.DataFrame) -> np.ndarray:
    """Dendritic-signal scalar built from convex-hull eta features.

    Per the user's analysis: filaments push eta_A way above 1 while keeping
    eta_M near 1, so we use

        dendritic = max(eta_A - 1, 0) + max(1 - eta_V, 0) - max(eta_M - 1, 0)

    Higher values mean "more dendritic" (high surface-area excess and low
    convexity ratio relative to the mean-curvature excess). Pure convex
    shapes give 0; smooth concave dents (eta_V < 1 but eta_A ~ eta_M ~ 1) give
    a small positive value; long thin filament structures push it up.
    """
    _require_cols(df, ["eta_V", "eta_A", "eta_M"], "chull_dendritic non-sphericity")
    eta_V = pd.to_numeric(df["eta_V"], errors="coerce").to_numpy(dtype=np.float64)
    eta_A = pd.to_numeric(df["eta_A"], errors="coerce").to_numpy(dtype=np.float64)
    eta_M = pd.to_numeric(df["eta_M"], errors="coerce").to_numpy(dtype=np.float64)
    val = np.maximum(eta_A - 1.0, 0.0) + np.maximum(1.0 - eta_V, 0.0) - np.maximum(eta_M - 1.0, 0.0)
    val = np.where(np.isfinite(val), val, np.nan)
    return val


def make_quantile_bins(values: np.ndarray, k: int) -> np.ndarray:
    """
    Returns bin edges (length k+1). Uses quantiles; duplicates are jittered slightly to remain monotone.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    v = values[np.isfinite(values)]
    if v.size == 0:
        raise ValueError("No finite values for binning.")
    qs = np.linspace(0.0, 1.0, k + 1)
    edges = np.quantile(v, qs).astype(np.float64)
    # Ensure strict monotonicity to avoid empty/ambiguous bins when values are highly repeated.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    return edges


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # bins: 0..k-1
    k = len(edges) - 1
    b = np.digitize(values, edges[1:-1], right=False)
    b = np.clip(b, 0, k - 1)
    return b.astype(np.int32)


def jnd_thin_by_frequency(
    rows: list[EvalRow],
    jnd_rel: float,
) -> list[EvalRow]:
    """
    JND-style thinning in frequency space.

    Greedy 1D clustering on sorted frequency:
    - Start new cluster when |f - f_cluster_center| / f_cluster_center > jnd_rel
    - Represent cluster by its median-frequency member (robust).
    """
    if not rows:
        return []
    if jnd_rel <= 0.0:
        return rows
    rows_sorted = sorted(rows, key=lambda r: r.freq_hz)
    clusters: list[list[EvalRow]] = []
    cur: list[EvalRow] = []
    cur_center = None
    for r in rows_sorted:
        if cur_center is None:
            cur = [r]
            cur_center = r.freq_hz
            continue
        rel = abs(r.freq_hz - cur_center) / max(cur_center, 1e-12)
        if rel <= jnd_rel:
            cur.append(r)
            cur_center = float(np.median([x.freq_hz for x in cur]))
        else:
            clusters.append(cur)
            cur = [r]
            cur_center = r.freq_hz
    if cur:
        clusters.append(cur)

    reps: list[EvalRow] = []
    for c in clusters:
        if len(c) == 1:
            reps.append(c[0])
            continue
        med = float(np.median([x.freq_hz for x in c]))
        reps.append(min(c, key=lambda x: abs(x.freq_hz - med)))
    return reps


def farthest_point_sample(
    rows: list[EvalRow],
    n: int,
    seed: int,
) -> list[EvalRow]:
    """
    Small, deterministic farthest-point sampler in 2D (non_sphericity, log(freq)).
    Good enough for representative diversity after thinning.
    """
    if n <= 0 or not rows:
        return []
    if len(rows) <= n:
        return rows

    rng = np.random.default_rng(seed)
    ns = np.array([r.non_sphericity for r in rows], dtype=np.float64)
    lf = np.log(np.maximum(1e-12, np.array([r.freq_hz for r in rows], dtype=np.float64)))
    # Normalize to comparable scales.
    x0 = (ns - np.nanmin(ns)) / max(1e-12, (np.nanmax(ns) - np.nanmin(ns)))
    x1 = (lf - np.nanmin(lf)) / max(1e-12, (np.nanmax(lf) - np.nanmin(lf)))
    X = np.stack([x0, x1], axis=1)

    start = int(rng.integers(0, len(rows)))
    chosen = [start]
    d2 = np.sum((X - X[start]) ** 2, axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(d2))
        chosen.append(nxt)
        d2 = np.minimum(d2, np.sum((X - X[nxt]) ** 2, axis=1))
    chosen_rows = [rows[i] for i in chosen]
    return chosen_rows


class BatchPredictor:
    def __init__(
        self,
        model_path: Path | None,
        feature_scaler_path: Path | None,
        target_scaler_path: Path | None,
        feature_cols: list[str],
        input_dim: int,
        device: str,
    ):
        self.feature_cols = feature_cols
        self.device = device
        self.ready = False
        self.load_error = ""
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None

        if model_path is None or feature_scaler_path is None or target_scaler_path is None:
            self.load_error = "model/scaler paths not set"
            return
        if not model_path.exists() or not feature_scaler_path.exists() or not target_scaler_path.exists():
            self.load_error = "model/scaler files not found"
            return
        try:
            self.feature_scaler = joblib.load(feature_scaler_path)
            self.target_scaler = joblib.load(target_scaler_path)
            self.model = BubbleFreqNet(input_dim=input_dim, hidden_dim=64, hidden_dim2=32, dropout=0.1).to(device)
            state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.ready = True
        except Exception as exc:
            self.load_error = f"{exc}"

    def predict_freq(
        self,
        df: pd.DataFrame,
        idxs: np.ndarray,
        log_f_strasberg: np.ndarray,
    ) -> np.ndarray:
        """
        Predict absolute frequency for rows `df.iloc[idxs]`.

        Models trained under the new Strasberg log-residual target emit
            y = z-scored( log(f_BEM) - log(f_strasberg) )
        so reconstruction requires adding log(f_strasberg) back before exp().
        `log_f_strasberg` must be a 1-D array with len == len(idxs), aligned to
        the SAME rows as idxs (i.e. element k corresponds to df.iloc[idxs[k]]).
        """
        if not self.ready or self.model is None or self.feature_scaler is None or self.target_scaler is None:
            return np.full(len(idxs), np.nan, dtype=np.float64)
        log_f_strasberg = np.asarray(log_f_strasberg, dtype=np.float64)
        if log_f_strasberg.shape[0] != len(idxs):
            raise ValueError(
                f"log_f_strasberg length {log_f_strasberg.shape[0]} does not match "
                f"idxs length {len(idxs)}"
            )
        sub = df.iloc[idxs]
        x = sub[self.feature_cols].to_numpy(dtype=np.float32)
        x_norm = self.feature_scaler.transform(x).astype(np.float32)
        xb = torch.from_numpy(x_norm).to(self.device)
        with torch.no_grad():
            y_norm = self.model(xb).detach().cpu().numpy().reshape(-1, 1)
        pred_log_resid = self.target_scaler.inverse_transform(y_norm).flatten()
        # Rows where Strasberg failed -> NaN propagates and predictions get masked by
        # compute_metrics() downstream. This is safer than silently falling back.
        pred_log = pred_log_resid + log_f_strasberg
        pred_f = np.exp(pred_log)
        return pred_f.astype(np.float64)


def compute_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    m = np.isfinite(pred) & np.isfinite(tgt) & (tgt > 0.0) & (pred > 0.0)
    if not np.any(m):
        return {"count": 0, "mape": float("nan"), "max_ape": float("nan"), "rmse_log": float("nan")}
    pred = pred[m]
    tgt = tgt[m]
    ape = np.abs(pred - tgt) / np.maximum(tgt, 1e-12) * 100.0
    rmse_log = float(np.sqrt(np.mean((np.log(pred) - np.log(tgt)) ** 2)))
    return {
        "count": int(m.sum()),
        "mape": float(np.mean(ape)),
        "max_ape": float(np.max(ape)),
        "rmse_log": rmse_log,
    }


def clean_output_dir(out_dir: Path, *, required_prefix: str = "stratified_eval") -> None:
    """
    Remove all contents of `out_dir` (files and subdirs), then recreate it empty.

    Safety: refuses to touch a directory whose name doesn't start with `required_prefix`.
    This prevents accidental deletion of unrelated folders if `--output-dir` is mis-typed.
    """
    out_dir = out_dir.resolve()
    if out_dir.exists() and not out_dir.name.startswith(required_prefix):
        raise ValueError(
            f"Refusing to clean '{out_dir}': folder name must start with '{required_prefix}'. "
            "Pass --no-clean to disable the guard, or rename your --output-dir."
        )
    if out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                try:
                    child.unlink()
                except FileNotFoundError:
                    pass
    out_dir.mkdir(parents=True, exist_ok=True)


def select_representatives_by_bin(
    df_sel: pd.DataFrame,
    non_sph_sel: np.ndarray,
    bin_sel: np.ndarray,
    n_bins: int,
) -> list[dict]:
    """
    Pick 1 representative bubble per bin: the row whose non_sphericity is closest to the
    bin's median. Returns a list of {bin, row_pos, index, mesh_filename, non_sphericity,
    frequency}. `row_pos` is the position within df_sel.
    """
    reps: list[dict] = []
    mesh_col = "mesh_filename" if "mesh_filename" in df_sel.columns else None
    idx_col = "index" if "index" in df_sel.columns else None
    freq_col = "frequency" if "frequency" in df_sel.columns else None
    for b in range(n_bins):
        positions = np.where(bin_sel == b)[0]
        if positions.size == 0:
            continue
        vals = non_sph_sel[positions]
        med = float(np.median(vals))
        winner = int(positions[int(np.argmin(np.abs(vals - med)))])
        row = df_sel.iloc[winner]
        reps.append(
            {
                "bin": int(b),
                "row_pos": int(winner),
                "index": (int(row[idx_col]) if idx_col and pd.notna(row.get(idx_col, np.nan)) else None),
                "mesh_filename": (str(row[mesh_col]) if mesh_col else ""),
                "non_sphericity": float(non_sph_sel[winner]),
                "frequency": (float(row[freq_col]) if freq_col else None),
                "bin_median": med,
            }
        )
    return reps


def emit_bin_thumbnails(
    df_sel: pd.DataFrame,
    bin_sel: np.ndarray,
    n_bins: int,
    thumbnail_dir: Path | None,
    bins_root: Path,
    *,
    thumbnail_dir_by_source: dict[str, Path] | None = None,
) -> int:
    """
    Copy thumbnails for every row in df_sel into `bins_root/bin_XX/`.

    ``thumbnail_dir_by_source`` lets callers route per-row thumbnail lookups based on
    the row's ``source`` column (e.g. ``{"VOF": Path(...), "LBM":
    Path(...)}``). Rows whose source matches a known key use that directory; rows
    with an unknown / missing source fall back to ``thumbnail_dir``.

    Returns the number of thumbnails copied.
    """
    if "mesh_filename" not in df_sel.columns:
        return 0

    by_source = {k: Path(v) for k, v in (thumbnail_dir_by_source or {}).items()}
    fallback_dir = Path(thumbnail_dir) if thumbnail_dir is not None else None
    if fallback_dir is not None and not fallback_dir.exists():
        fallback_dir = None
    if not by_source and fallback_dir is None:
        return 0

    has_source = "source" in df_sel.columns

    copied = 0
    for b in range(n_bins):
        positions = np.where(bin_sel == b)[0]
        if positions.size == 0:
            continue
        bin_dir = bins_root / f"bin_{b:02d}"
        bin_dir.mkdir(parents=True, exist_ok=True)
        for pos in positions:
            row = df_sel.iloc[int(pos)]
            mesh_name = str(row.get("mesh_filename", ""))
            src = str(row["source"]) if has_source and pd.notna(row.get("source", None)) else ""
            # Strict per-source routing: if `src` matches a registered key, only
            # look in that directory. Cross-source fallback (e.g. lbm row -> tim2016
            # dir) is forbidden because the two datasets share `bubble.NNNN.M.png`
            # filename conventions and would silently produce wrong thumbnails.
            if src and src in by_source:
                chosen_dir = by_source[src]
                if chosen_dir is None or not chosen_dir.exists():
                    continue
            else:
                chosen_dir = fallback_dir
            if chosen_dir is None:
                continue
            thumb = find_thumbnail_path(chosen_dir, mesh_name)
            if thumb is None:
                continue
            if "index" in df_sel.columns and pd.notna(row.get("index", None)):
                ds_idx: object = row["index"]
            elif "mesh_id" in df_sel.columns and pd.notna(row.get("mesh_id", None)):
                ds_idx = Path(str(row["mesh_id"])).stem
            else:
                ds_idx = Path(mesh_name).stem or "na"
            freq = row.get("frequency", float("nan"))
            try:
                freq_str = f"{float(freq):.6f}"
            except Exception:
                freq_str = "nan"
            src_tag = f"_{src}" if src else ""
            name = sanitize_filename(f"{ds_idx}{src_tag}_{freq_str}.png")
            try:
                shutil.copy2(thumb, bin_dir / name)
                copied += 1
            except Exception:
                pass
    return copied


def _load_test_filenames(split_json: Path) -> set[str] | None:
    try:
        with split_json.open(encoding="utf-8") as f:
            sj = json.load(f)
    except Exception:
        return None
    key = "mesh_filename_test"
    if key not in sj:
        return None
    return {str(x) for x in sj[key]}


def emit_worst_case_thumbnails(
    df_full: pd.DataFrame,
    predictor: "BatchPredictor",
    model_kind: str,
    out_dir: Path,
    thumbnail_dir: Path | None,
    top_k: int,
    split_json: Path | None,
    *,
    thumbnail_dir_by_source: dict[str, Path] | None = None,
) -> None:
    """
    Predict frequencies on the full test split and copy the top-k APE cases into
    `out_dir/worst_case/`. df_full is expected to already be cleaned and (for inertia
    model-kind) have `i11_hat` / `i22_hat` columns. Needs `mesh_filename` and `frequency`.
    """
    if not predictor.ready:
        print(f"[worst-case] predictor not ready: {predictor.load_error}")
        return
    if "mesh_filename" not in df_full.columns or "frequency" not in df_full.columns:
        print("[worst-case] dataset missing mesh_filename / frequency; skipping")
        return

    if split_json is None:
        print(f"[worst-case] no --split-json given for model={model_kind}; skipping")
        return
    test_names = _load_test_filenames(split_json)
    if test_names is None:
        print(f"[worst-case] split.json missing mesh_filename_test: {split_json}; skipping")
        return

    mask = df_full["mesh_filename"].astype(str).isin(test_names)
    idx_test = np.where(mask.to_numpy())[0]
    if idx_test.size == 0:
        print("[worst-case] no rows from df intersect the test split; skipping")
        return

    # Reconstruct absolute frequencies from the network output. The required
    # log-baseline depends on the model-kind: Strasberg residual / Minnaert residual /
    # zero (direct log-f). _baseline_log_for_rows handles all three.
    log_fs_test = _baseline_log_for_rows(model_kind, df_full.iloc[idx_test])
    pred_all = predictor.predict_freq(df_full, idx_test, log_f_strasberg=log_fs_test)
    tgt = pd.to_numeric(df_full["frequency"], errors="coerce").to_numpy(dtype=np.float64)[idx_test]
    wc_dir = out_dir / "worst_case"
    wc_dir.mkdir(parents=True, exist_ok=True)

    export_worst_cases(
        df_used=df_full,
        test_indices=idx_test,
        pred_f=pred_all,
        tgt_f=tgt,
        out_dir=wc_dir,
        top_k=top_k,
        thumbnail_dir=thumbnail_dir,
        thumbnail_dir_by_source=thumbnail_dir_by_source,
    )
    print(f"[worst-case] exported top-{top_k} test APE thumbnails to {wc_dir}")


def _non_sphericity_for(
    df: pd.DataFrame, metric: str, volume_col: str
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(non_sphericity, phi)``; ``phi`` is None unless metric == "phi"."""
    if metric == "phi":
        phi = sphericity_phi_wadell(df, volume_col=volume_col)
        return non_sphericity_from_phi(phi), phi
    if metric == "chull_dendritic":
        return non_sphericity_chull_dendritic(df), None
    return non_sphericity_inertia_anisotropy(df), None


def _setup_predictor(args, df: pd.DataFrame) -> tuple[pd.DataFrame, BatchPredictor, dict]:
    """Add the model's feature columns to ``df`` and build its BatchPredictor.

    Returns ``(df_with_features, predictor, paths)`` where ``paths`` carries the
    resolved checkpoint / scaler locations and feature order for summary.json.
    """
    if args.model_kind != "nn8":
        raise ValueError(f"unsupported --model-kind: {args.model_kind}")
    feature_cols = list(EXPECTED_FEATURE_COLS_8FEAT)
    df = add_nn8_features(df)
    model_path = args.model_path or (
        NN8_ARTIFACT_DIR / "bubble_freq_net_best.pt"
    )
    feat_scaler = args.feature_scaler or (NN8_ARTIFACT_DIR / "feature_scaler.joblib")
    tgt_scaler = args.target_scaler or (NN8_ARTIFACT_DIR / "target_log_scaler.joblib")
    predictor = BatchPredictor(
        model_path=model_path,
        feature_scaler_path=feat_scaler,
        target_scaler_path=tgt_scaler,
        feature_cols=feature_cols,
        input_dim=len(feature_cols),
        device=args.device,
    )
    paths = {
        "kind": args.model_kind,
        "model_path": str(Path(model_path).as_posix()),
        "feature_scaler": str(Path(feat_scaler).as_posix()),
        "target_scaler": str(Path(tgt_scaler).as_posix()),
        "predictor_ready": bool(predictor.ready),
        "predictor_error": predictor.load_error,
        "feature_cols": feature_cols,
    }
    return df, predictor, paths


def _metrics_by_bin(
    preds: dict[str, np.ndarray],
    tgt: np.ndarray,
    bin_ids: np.ndarray,
    n_bins: int,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, dict[str, float]]]:
    """Per-bin and overall metrics for each named predictor."""
    per_bin: dict[str, list[dict[str, float]]] = {name: [] for name in preds}
    for b in range(n_bins):
        m = bin_ids == b
        for name, p in preds.items():
            per_bin[name].append(compute_metrics(p[m], tgt[m]))
    overall = {name: compute_metrics(p, tgt) for name, p in preds.items()}
    return per_bin, overall


def _auto_thresholds(non_sph: np.ndarray) -> list[float]:
    """21 evenly spaced quantiles of the non-sphericity scalar, deduplicated."""
    qs = np.linspace(0.0, 1.0, 21)
    thrs = np.quantile(non_sph[np.isfinite(non_sph)], qs).tolist()
    return sorted(set(float(t) for t in thrs))


def _hybrid_sweep(
    non_sph: np.ndarray,
    pred_model: np.ndarray,
    f_minnaert: np.ndarray,
    tgt: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float]]:
    """Metrics of 'Minnaert at or below the threshold, network above' per threshold."""
    rows: list[dict[str, float]] = []
    for t in thresholds:
        use_model = non_sph > float(t)
        met = compute_metrics(np.where(use_model, pred_model, f_minnaert), tgt)
        rows.append(
            {"threshold": float(t), "use_model_frac": float(np.mean(use_model)), **met}
        )
    return rows


def _emit_eval_plots(
    out_dir: Path,
    per_bin: dict[str, list[dict[str, float]]],
    hybrid_rows: list[dict[str, float]],
    model_kind: str,
    hybrid_title: str,
) -> None:
    """Write the three per-bin MAPE bar charts plus the hybrid-sweep curve."""
    save_bin_barplot(
        per_bin["model"],
        out_dir / "per_bin_model_mape.png",
        title=f"Network MAPE per non-sphericity bin ({model_kind})",
        key="mape",
    )
    save_bin_barplot(
        per_bin["minnaert"],
        out_dir / "per_bin_minnaert_mape.png",
        title="Minnaert MAPE per non-sphericity bin",
        key="mape",
    )
    save_bin_barplot(
        per_bin["strasberg"],
        out_dir / "per_bin_strasberg_mape.png",
        title="Strasberg MAPE per non-sphericity bin",
        key="mape",
    )
    plt.figure(figsize=(7.2, 4.0), dpi=160)
    plt.plot(
        [r["threshold"] for r in hybrid_rows],
        [r["mape"] for r in hybrid_rows],
        marker="o",
        label="hybrid MAPE",
    )
    plt.xlabel("non-sphericity threshold (use Minnaert below, network above)")
    plt.ylabel("MAPE (%)")
    plt.title(hybrid_title)
    plt.ylim(0.0, 20.0)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "hybrid_sweep_mape.png")
    plt.close()


def _print_overall(
    overall: dict[str, dict[str, float]], hybrid_rows: list[dict[str, float]]
) -> None:
    print(f"Overall Minnaert MAPE:              {overall['minnaert']['mape']:.2f}%")
    print(f"Overall Strasberg MAPE:             {overall['strasberg']['mape']:.2f}%")
    print(f"Overall network (model) MAPE:       {overall['model']['mape']:.2f}%")
    if hybrid_rows:
        best = min(
            hybrid_rows,
            key=lambda r: float(r["mape"]) if np.isfinite(r["mape"]) else float("inf"),
        )
        print(
            f"Best hybrid sweep: thr={best['threshold']:.4g}, "
            f"MAPE={best['mape']:.2f}%, use_model_frac={best['use_model_frac']:.2f}"
        )


def replot_selected_metrics(
    out_dir: Path,
    *,
    sync_f_gt: bool = True,
    ground_truth_col: str = "frequency",
) -> None:
    """
    After `selected_rows.csv` is updated (e.g. BEM++ `frequency`), recompute per-bin / hybrid MAPE
    using the same bin edges as stored in `summary.json`, and regenerate the PNGs + metric sections
    in `summary.json` without resampling or rerunning the neural model.
    """
    out_dir = out_dir.resolve()
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "selected_rows.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")

    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)

    edges = np.array(summary["bin_edges_selected"], dtype=np.float64)
    k = len(edges) - 1
    if k < 2:
        raise ValueError("summary.json bin_edges_selected must define at least 2 bins")

    df = _clean_columns(pd.read_csv(csv_path))
    _require_cols(
        df,
        ["non_sphericity", ground_truth_col, "f_pred_model", "f_minnaert", "i00", "i11", "i22"],
        "selected_rows.csv (replot)",
    )

    non_sph = pd.to_numeric(df["non_sphericity"], errors="coerce").to_numpy(dtype=np.float64)
    freq_gt = pd.to_numeric(df[ground_truth_col], errors="coerce").to_numpy(dtype=np.float64)
    pred_model = pd.to_numeric(df["f_pred_model"], errors="coerce").to_numpy(dtype=np.float64)
    f_minnaert = pd.to_numeric(df["f_minnaert"], errors="coerce").to_numpy(dtype=np.float64)
    f_strasberg = strasberg_frequency_from_inertia_unit_volume(df)

    bin_sel = assign_bins(non_sph, edges)

    per_bin, overall = _metrics_by_bin(
        {"minnaert": f_minnaert, "strasberg": f_strasberg, "model": pred_model},
        freq_gt,
        bin_sel,
        k,
    )

    # Reuse the recorded thresholds when present so the replotted sweep is
    # directly comparable to the original run.
    if summary.get("hybrid_sweep"):
        thrs = [float(row["threshold"]) for row in summary["hybrid_sweep"]]
    else:
        thrs = _auto_thresholds(non_sph)
    hybrid_rows = _hybrid_sweep(non_sph, pred_model, f_minnaert, freq_gt, thrs)

    model_kind = summary.get("model", {}).get("kind", "model")
    _emit_eval_plots(
        out_dir,
        per_bin,
        hybrid_rows,
        model_kind,
        "Hybrid go/no-go sweep (balanced stratified set)",
    )

    summary["metrics_overall_selected"] = overall
    summary["metrics_per_bin_selected"] = per_bin
    summary["hybrid_sweep"] = hybrid_rows
    summary["replot_note"] = (
        f"MAPE plots and metrics_* sections recomputed using `{ground_truth_col}` as ground truth; "
        f"bin edges unchanged from bin_edges_selected."
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if sync_f_gt and "f_gt" in df.columns:
        df_out = df.copy()
        df_out["f_gt"] = pd.to_numeric(df_out[ground_truth_col], errors="coerce")
        df_out.to_csv(csv_path, index=False)

    print("=== Replot from selected_rows (GT = %s) ===" % ground_truth_col)
    _print_overall(overall, hybrid_rows)
    print(f"Updated PNGs + summary.json under: {out_dir}")
    if sync_f_gt and "f_gt" in df.columns:
        print(f"Synced column f_gt <- {ground_truth_col} in {csv_path.name}")


def save_bin_barplot(
    per_bin: list[dict[str, float]],
    path: Path,
    title: str,
    key: str,
) -> None:
    vals = [d.get(key, float("nan")) for d in per_bin]
    counts = [d.get("count", 0) for d in per_bin]
    x = np.arange(len(vals))
    plt.figure(figsize=(8.4, 4.2), dpi=160)
    plt.bar(x, vals)
    plt.xticks(x, [f"bin {i}\n(n={counts[i]})" for i in range(len(vals))])
    plt.ylabel(key)
    plt.title(title)
    if key.lower() == "mape":
        plt.ylim(0.0, 30.0)
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified evaluation for bubble frequency models")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/bubble_gym/dataset_bubblegym_10k.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/stratified_eval"))
    parser.add_argument(
        "--eval-selected-rows",
        type=Path,
        default=None,
        help=(
            "Evaluate a fixed selected set CSV (e.g. results/.../selected_rows.csv) instead of "
            "sampling from --dataset. This recomputes f_pred_model for the chosen model-kind and "
            "uses `frequency` as ground truth."
        ),
    )
    parser.add_argument(
        "--model-kind",
        type=str,
        default="nn8",
        choices=["nn8"],
        help=(
            "Which learned model to evaluate. Only the shipped 8-feature direct "
            "log-frequency head is released, so this has a single value; it is "
            "kept as a flag because it selects the checkpoint / scaler defaults "
            "and is recorded in summary.json."
        ),
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--feature-scaler", type=Path, default=None)
    parser.add_argument("--target-scaler", type=Path, default=None)
    parser.add_argument(
        "--non-sphericity-metric",
        type=str,
        default="inertia_aniso",
        choices=["inertia_aniso", "phi", "chull_dendritic"],
        help=(
            "inertia_aniso (default): sqrt((i11/i00-1)^2 + (i22/i00-1)^2). "
            "phi: Wadell Phi from V,S; stratification uses (1-Phi) so higher = less spherical. "
            "chull_dendritic: max(eta_A-1,0) + max(1-eta_V,0) - max(eta_M-1,0), positive on "
            "filament/dendritic shapes. "
            "VERIFIED: the shipped 10x10 ablation panel was produced with "
            "inertia_aniso and --per-bin 10 (bin_edges_selected in the shipped "
            "curated_eval.json reproduces the ten quantile edges of inertia "
            "anisotropy over the 1500-row test split exactly, min 0.0329 / max "
            "43.676), whereas the paper's Fig. 4 text calls for phi (Wadell). "
            "The default is left at inertia_aniso to match the shipped artifact."
        ),
    )
    parser.add_argument("--bins", type=int, default=10, help="Quantile bins for non-sphericity.")
    parser.add_argument("--per-bin", type=int, default=120, help="Target selected examples per bin.")
    parser.add_argument(
        "--thin-jnd-pct",
        type=float,
        default=0.5,
        help="Within-bin thinning: relative JND in frequency (percent). 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--volume-col",
        type=str,
        default="volume",
        help="Volume column used for the Minnaert radius R = (3V/4pi)^(1/3).",
    )
    parser.add_argument(
        "--default-volume",
        type=float,
        default=1.0,
        help="Fallback volume if --volume-col is missing from the dataset.",
    )
    parser.add_argument(
        "--go-thresholds",
        type=str,
        default="",
        help="Comma-separated non-sphericity thresholds for hybrid gating sweep (empty -> auto).",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Optional path to training split.json; when set, only evaluate on the chosen split.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate when --split-json is provided (default: test).",
    )
    parser.add_argument(
        "--replot-selected-only",
        action="store_true",
        help=(
            "Only read output-dir/selected_rows.csv + summary.json, regenerate MAPE PNGs and metric "
            "sections using column `frequency` as ground truth (e.g. after BEM++ correction)."
        ),
    )
    parser.add_argument(
        "--no-sync-f-gt",
        action="store_true",
        help="With --replot-selected-only, do not overwrite f_gt in selected_rows.csv to match frequency.",
    )
    parser.add_argument(
        "--thumbnail-dir",
        type=Path,
        default=Path("dataset/bubble_gym/bubble_mesh_thumbnails_400x400/VOF"),
        help=(
            "Directory with per-bubble PNG thumbnails for the Tim2016 6k source, used as a "
            "fallback for rows without a recognized 'source' tag."
        ),
    )
    parser.add_argument(
        "--lbm-thumbnail-dir",
        type=Path,
        default=Path(
            "dataset/bubble_gym/bubble_mesh_thumbnails_400x400/LBM/"
            "lbm_exhale_bem_galerkin_3k_thumbnails"
        ),
        help=(
            "Directory with per-bubble PNG thumbnails for the LBM source. "
            "Used to populate bins/ and worst_case/ for LBM rows in the combined / "
            "LBM-only datasets."
        ),
    )
    parser.add_argument(
        "--worst-top-k",
        type=int,
        default=10,
        help="How many worst-APE test cases to export into out_dir/worst_case/.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do NOT wipe output_dir before writing new results (useful for incremental debugging).",
    )
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    if not args.no_clean and not args.replot_selected_only:
        clean_output_dir(out_dir, required_prefix="stratified_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_selected_only:
        replot_selected_metrics(
            out_dir,
            sync_f_gt=not args.no_sync_f_gt,
            ground_truth_col="frequency",
        )
        return

    # =============================================================================================
    # Evaluate a pre-selected set (no resampling)
    # =============================================================================================
    if args.eval_selected_rows is not None:
        df_sel = _clean_columns(pd.read_csv(args.eval_selected_rows))
        # We require ground truth in `frequency` (already BEM++ corrected per your workflow).
        _require_cols(df_sel, [TARGET_COL] + INERTIA_ONLY_REQUIRED_COLS, "eval-selected-rows")
        df_sel[TARGET_COL] = pd.to_numeric(df_sel[TARGET_COL], errors="coerce")
        for c in INERTIA_ONLY_REQUIRED_COLS:
            df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")
        if "surface_area" in df_sel.columns:
            df_sel["surface_area"] = pd.to_numeric(df_sel["surface_area"], errors="coerce")
        if args.volume_col in df_sel.columns:
            df_sel[args.volume_col] = pd.to_numeric(df_sel[args.volume_col], errors="coerce")
        df_sel = df_sel.replace([np.inf, -np.inf], np.nan)
        _drop_eval = [TARGET_COL, "i00", "i11", "i22"]
        df_sel = df_sel.dropna(subset=_drop_eval)
        df_sel = df_sel[df_sel[TARGET_COL] > 0.0].reset_index(drop=True)

        freq_sel = pd.to_numeric(df_sel[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)

        # Non-sphericity: prefer column if present (so we "share" the exact same selected set scalar).
        if "non_sphericity" in df_sel.columns:
            non_sph_sel = pd.to_numeric(df_sel["non_sphericity"], errors="coerce").to_numpy(dtype=np.float64)
        else:
            non_sph_sel, phi_arr = _non_sphericity_for(
                df_sel, args.non_sphericity_metric, args.volume_col
            )
            if phi_arr is not None:
                df_sel = df_sel.copy()
                df_sel["phi"] = phi_arr

        # Minnaert baseline
        radius = compute_radius_from_volume(
            df_sel, volume_col=args.volume_col, default_volume=args.default_volume
        )
        f_minnaert_sel = minnaert_frequency_from_radius(radius)
        f_strasberg_sel = strasberg_frequency_from_inertia_unit_volume(df_sel)

        df_sel, predictor, model_info = _setup_predictor(args, df_sel)
        idxs = np.arange(len(df_sel), dtype=np.int64)
        log_fs_sel = _baseline_log_for_rows(args.model_kind, df_sel.iloc[idxs])
        pred_model = predictor.predict_freq(df_sel, idxs, log_f_strasberg=log_fs_sel)

        # Bin + metrics (based on selected set only)
        edges_sel = make_quantile_bins(non_sph_sel, k=args.bins)
        bin_sel = assign_bins(non_sph_sel, edges_sel)

        per_bin, overall = _metrics_by_bin(
            {
                "minnaert": f_minnaert_sel,
                "strasberg": f_strasberg_sel,
                "model": pred_model,
            },
            freq_sel,
            bin_sel,
            args.bins,
        )
        hybrid_rows = _hybrid_sweep(
            non_sph_sel,
            pred_model,
            f_minnaert_sel,
            freq_sel,
            _auto_thresholds(non_sph_sel),
        )

        # Save artifacts
        assign_kw: dict[str, np.ndarray] = {
            "non_sphericity": non_sph_sel,
            "f_minnaert": f_minnaert_sel,
            "f_strasberg": f_strasberg_sel,
            "f_pred_model": pred_model,
            "f_gt": freq_sel,
        }
        with (out_dir / "selected_rows.csv").open("w", encoding="utf-8", newline="") as f:
            df_sel.assign(**assign_kw).to_csv(f, index=False)

        _emit_eval_plots(
            out_dir,
            per_bin,
            hybrid_rows,
            args.model_kind,
            "Hybrid go/no-go sweep (fixed selected set)",
        )

        summary = {
            "dataset": str(Path(args.dataset).as_posix()),
            "eval_mode": "eval_selected_rows",
            "eval_selected_rows": str(Path(args.eval_selected_rows).as_posix()),
            "eval_split": (args.split if args.split_json is not None else "all"),
            "eval_split_json": (str(Path(args.split_json).as_posix()) if args.split_json is not None else ""),
            "n_total_clean": int(len(df_sel)),
            "non_sphericity_metric": args.non_sphericity_metric,
            "bin_edges_selected": edges_sel.tolist(),
            "sampling": {
                "bins": int(args.bins),
                "per_bin_target": int(args.per_bin),
                "thin_jnd_pct": float(args.thin_jnd_pct),
                "selected_unique": int(len(df_sel)),
            },
            "model": model_info,
            "metrics_overall_selected": overall,
            "metrics_per_bin_selected": per_bin,
            "hybrid_sweep": hybrid_rows,
        }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("=== Stratified evaluation complete ===")
        print(f"Selected set size: {len(df_sel)} (fixed from --eval-selected-rows)")
        _print_overall(overall, hybrid_rows)
        print(f"Saved artifacts to: {out_dir}")
        return

    df = _clean_columns(pd.read_csv(args.dataset))
    if args.split_json is not None:
        df = _filter_by_split(df, args.split_json.resolve(), args.split)
    # The only released model-kind is nn8, which needs all eight feature columns.
    _require_cols(
        df,
        ["surface_area"],
        "dataset (surface_area must be measured after unit-volume scaling)",
    )
    df["surface_area"] = pd.to_numeric(df["surface_area"], errors="coerce")
    _require_cols(
        df,
        ["volume", "Phi_VM", "W_vertex", "eta_V", "eta_A", "eta_M"],
        f"dataset ({args.model_kind})",
    )
    if args.non_sphericity_metric == "chull_dendritic":
        _require_cols(df, ["eta_V", "eta_A", "eta_M"], "dataset (chull_dendritic metric)")
    _require_cols(df, [TARGET_COL] + INERTIA_ONLY_REQUIRED_COLS, "dataset")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    for c in INERTIA_ONLY_REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if args.volume_col in df.columns:
        df[args.volume_col] = pd.to_numeric(df[args.volume_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    _drop_main = [
        TARGET_COL, "i00", "i11", "i22",
        "volume", "Phi_VM", "W_vertex", "eta_V", "eta_A", "eta_M",
    ]
    df = df.dropna(subset=_drop_main)
    df = df[df[TARGET_COL] > 0.0].reset_index(drop=True)

    non_sph, phi_arr = _non_sphericity_for(
        df, args.non_sphericity_metric, args.volume_col
    )

    freq = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(non_sph) & np.isfinite(freq) & (freq > 0.0)
    df = df.iloc[np.where(valid)[0]].reset_index(drop=True)
    non_sph = non_sph[valid]
    freq = freq[valid]
    if phi_arr is not None:
        phi_arr = phi_arr[valid]

    # Baseline: Minnaert (needs radius estimate).
    radius = compute_radius_from_volume(
        df, volume_col=args.volume_col, default_volume=args.default_volume
    )
    f_minnaert = minnaert_frequency_from_radius(radius)
    f_strasberg = strasberg_frequency_from_inertia_unit_volume(df)

    df, predictor, model_info = _setup_predictor(args, df)

    # Stratify + thin + sample.
    edges = make_quantile_bins(non_sph, k=args.bins)
    bin_id = assign_bins(non_sph, edges)
    jnd_rel = max(0.0, args.thin_jnd_pct / 100.0)

    chosen_idxs: list[int] = []
    per_bin_summary: list[dict[str, float]] = []
    for b in range(args.bins):
        idxs = np.where(bin_id == b)[0]
        rows = [EvalRow(int(i), float(non_sph[i]), float(freq[i])) for i in idxs]
        rows_thin = jnd_thin_by_frequency(rows, jnd_rel=jnd_rel)
        rows_sel = farthest_point_sample(rows_thin, n=args.per_bin, seed=args.seed + 1000 * b)
        chosen_idxs.extend([r.idx for r in rows_sel])
        per_bin_summary.append(
            {
                "bin": b,
                "bin_count_raw": int(len(rows)),
                "bin_count_thin": int(len(rows_thin)),
                "bin_count_selected": int(len(rows_sel)),
                "non_sph_min": float(np.min(non_sph[idxs])) if idxs.size else None,
                "non_sph_max": float(np.max(non_sph[idxs])) if idxs.size else None,
            }
        )

    chosen_idxs = np.array(sorted(set(chosen_idxs)), dtype=np.int64)
    df_sel = df.iloc[chosen_idxs].reset_index(drop=True)
    non_sph_sel = non_sph[chosen_idxs]
    freq_sel = freq[chosen_idxs]
    f_minnaert_sel = f_minnaert[chosen_idxs]
    f_strasberg_sel = f_strasberg[chosen_idxs]

    log_fs_chosen = _baseline_log_for_rows(args.model_kind, df.iloc[chosen_idxs])
    pred_model = predictor.predict_freq(df, chosen_idxs, log_f_strasberg=log_fs_chosen)

    # Per-bin metrics (on the selected balanced set).
    edges_sel = make_quantile_bins(non_sph_sel, k=args.bins)
    bin_sel = assign_bins(non_sph_sel, edges_sel)

    per_bin, overall = _metrics_by_bin(
        {"minnaert": f_minnaert_sel, "strasberg": f_strasberg_sel, "model": pred_model},
        freq_sel,
        bin_sel,
        args.bins,
    )

    # Hybrid gating sweep: choose Minnaert if non-sphericity <= thr else model.
    if args.go_thresholds.strip():
        thrs = [float(x.strip()) for x in args.go_thresholds.split(",") if x.strip()]
    else:
        thrs = _auto_thresholds(non_sph_sel)
    hybrid_rows = _hybrid_sweep(
        non_sph_sel, pred_model, f_minnaert_sel, freq_sel, thrs
    )

    # Save artifacts.
    assign_kw: dict[str, np.ndarray] = {
        "non_sphericity": non_sph_sel,
        "f_minnaert": f_minnaert_sel,
        "f_strasberg": f_strasberg_sel,
        "f_pred_model": pred_model,
        "f_gt": freq_sel,
    }
    if phi_arr is not None:
        assign_kw["phi"] = phi_arr[chosen_idxs]
    df_sel_out = df_sel.assign(**assign_kw)
    with (out_dir / "selected_rows.csv").open("w", encoding="utf-8", newline="") as f:
        df_sel_out.to_csv(f, index=False)

    representatives = select_representatives_by_bin(
        df_sel_out, non_sph_sel, bin_sel, n_bins=args.bins
    )

    thumb_dir = args.thumbnail_dir.resolve() if args.thumbnail_dir is not None else None
    lbm_thumb_dir = (
        args.lbm_thumbnail_dir.resolve() if args.lbm_thumbnail_dir is not None else None
    )
    thumb_by_source: dict[str, Path] = {}
    if thumb_dir is not None and thumb_dir.exists():
        thumb_by_source["VOF"] = thumb_dir
    if lbm_thumb_dir is not None and lbm_thumb_dir.exists():
        thumb_by_source["LBM"] = lbm_thumb_dir

    if thumb_by_source or (thumb_dir is not None and thumb_dir.exists()):
        copied = emit_bin_thumbnails(
            df_sel_out,
            bin_sel,
            n_bins=args.bins,
            thumbnail_dir=thumb_dir,
            bins_root=out_dir / "bins",
            thumbnail_dir_by_source=thumb_by_source,
        )
        sources_msg = ", ".join(f"{k}={v}" for k, v in thumb_by_source.items()) or str(thumb_dir)
        print(f"[bins] copied {copied} thumbnails into {out_dir / 'bins'} (sources: {sources_msg})")
    else:
        print(
            f"[bins] no usable thumbnail dirs (tim2016='{thumb_dir}', "
            f"lbm='{lbm_thumb_dir}'); skipping bin thumbnails"
        )

    emit_worst_case_thumbnails(
        df_full=df,
        predictor=predictor,
        model_kind=args.model_kind,
        out_dir=out_dir,
        thumbnail_dir=thumb_dir,
        top_k=int(args.worst_top_k),
        split_json=(args.split_json.resolve() if args.split_json is not None else None),
        thumbnail_dir_by_source=thumb_by_source,
    )

    _emit_eval_plots(
        out_dir,
        per_bin,
        hybrid_rows,
        args.model_kind,
        "Hybrid go/no-go sweep (balanced stratified set)",
    )

    summary = {
        "dataset": str(Path(args.dataset).as_posix()),
        "eval_split": (args.split if args.split_json is not None else "all"),
        "eval_split_json": (str(Path(args.split_json).as_posix()) if args.split_json is not None else ""),
        "n_total_clean": int(len(df)),
        "non_sphericity_metric": args.non_sphericity_metric,
        "stratify_scalar_note": (
            "Bins/hybrid use (1 - clip(Phi,0,1)); column non_sphericity in selected_rows matches that scalar."
            if args.non_sphericity_metric == "phi"
            else "non_sphericity is the chosen metric directly."
        ),
        "bin_edges_full": edges.tolist(),
        "bin_edges_selected": edges_sel.tolist(),
        "sampling": {
            "bins": int(args.bins),
            "per_bin_target": int(args.per_bin),
            "thin_jnd_pct": float(args.thin_jnd_pct),
            "selected_unique": int(len(chosen_idxs)),
        },
        "model": model_info,
        "bin_population_full": per_bin_summary,
        "metrics_overall_selected": overall,
        "metrics_per_bin_selected": per_bin,
        "hybrid_sweep": hybrid_rows,
        "representatives": representatives,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Stratified evaluation complete ===")
    print(f"Selected set size: {len(df_sel)} (target ~{args.bins * args.per_bin}, after thinning/dedup)")
    _print_overall(overall, hybrid_rows)
    print(f"Saved artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

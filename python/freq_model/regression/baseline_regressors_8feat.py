"""Non-neural regression baselines on the 8-feature bubble-frequency task.

Is the small MLP necessary at all -- could a linear model, polynomial
regression, or an RBF regressor reach comparable accuracy on the same 8D shape
descriptors? Answered under the EXACT training contract of the production
8-feature model (``fit_shape_freq_model.py``):

* identical features: [i11_over_i00, i22_over_i00, non_sph_va, non_sph_vm,
  non_sph_w, eta_V, eta_A, eta_M], built by the trainer's own ``load_xy``;
* identical 70/15/15 split (seed 42) -- mechanically asserted against the
  ``split.json`` saved by the trained model, so "same test set as the paper"
  is verified, not assumed;
* identical target (log f_BEM) and identical metrics (MAPE / RMSE(log) /
  max APE from ``freq_model.NN.training.evaluate_metrics``'s formula).

Models (all fit on train only; hyperparameters selected by VALIDATION MAPE,
the test set is never touched during tuning):

* linear : least squares on the 8 standardized features
* poly2  : degree-2 polynomial features -> ridge (alpha tuned on val)
* poly3  : degree-3 polynomial features -> ridge (alpha tuned on val)
* rbf    : kernel ridge with RBF kernel (alpha, gamma tuned on val)

``--clamp-to-minnaert`` optionally clamps predictions to
``[f_Minnaert, 1.35 * f_Minnaert]`` at unit volume. It is OFF by default so the
recorded numbers stay reproducible; it exists because poly2 extrapolates
catastrophically (~1e237) on out-of-distribution scene meshes.

RUN (repo root; needs $env:KMP_DUPLICATE_LIB_OK = "TRUE" on Windows):
    python python/freq_model/regression/baseline_regressors_8feat.py

Writes results/baseline_regressors_8feat_bubblegym_10k/{baseline_metrics.json,
baseline_metrics.md}.
"""

from __future__ import annotations

# Import torch first: on Windows this avoids a joblib/torch OpenMP DLL clash
# (same defensive ordering as near_duplicate_controlled_mape.py). The trainer
# module imported below pulls in torch anyway.
import torch  # noqa: F401

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

# The trainer module imports SummaryWriter, but tensorboard is optional in this
# environment. This script never logs to tensorboard, so stub the module instead
# of adding the dependency.
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
except ImportError:
    _tb_stub = types.ModuleType("torch.utils.tensorboard")

    class _SummaryWriterStub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("tensorboard is stubbed out in baseline_regressors_8feat.py")

    _tb_stub.SummaryWriter = _SummaryWriterStub
    sys.modules["torch.utils.tensorboard"] = _tb_stub

_THIS = Path(__file__).resolve()
_FREQ_MODEL_DIR = _THIS.parents[1]
_PYTHON_ROOT = _FREQ_MODEL_DIR.parent
_REPO_ROOT = _PYTHON_ROOT.parent
for _p in (_PYTHON_ROOT, _FREQ_MODEL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sklearn.kernel_ridge import KernelRidge  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402

from freq_model.NN.bubble_dataset import split_dataset  # noqa: E402
from freq_model.NN.fit_shape_freq_model import (  # noqa: E402
    FEATURE_COLS,
    load_xy,
)
from freq_model.analytical.minnaert_freq import (  # noqa: E402
    minnaert_frequency_unit_volume,
)

MODEL_DIR = _FREQ_MODEL_DIR / "output" / "output_8feature_direct_bubblegym_10k"
SPLIT_JSON = MODEL_DIR / "split.json"
METRICS_JSON = MODEL_DIR / "metrics.json"

OUT_DIR = _REPO_ROOT / "results" / "baseline_regressors_8feat_bubblegym_10k"

DEFAULT_DATASET = _REPO_ROOT / "dataset" / "bubble_gym" / "dataset_bubblegym_10k.csv"

RIDGE_ALPHAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
KRR_ALPHAS = np.logspace(-10, 0, 11)
KRR_GAMMAS = np.logspace(-3, 2, 12)

# Physical admissibility window for a unit-volume bubble: a nonspherical bubble
# can never resonate below its equal-volume Minnaert frequency, and the shape
# factor over this dataset stays under ~1.35. Used only when a caller opts in
# via --clamp-to-minnaert; see minnaert_log_clamp_bounds().
MINNAERT_CLAMP_FACTOR = 1.35

CLAMP_FLAG_HELP = (
    "Clamp predictions to [f_Minnaert, %.2f * f_Minnaert] at unit volume. OFF by "
    "default: every recorded number in results/ was produced unclamped, and the "
    "clamp is inert on in-distribution data. Turn it on for out-of-distribution "
    "scene meshes, where poly2 extrapolates to ~1e237." % MINNAERT_CLAMP_FACTOR
)


def minnaert_log_clamp_bounds(factor: float = MINNAERT_CLAMP_FACTOR) -> tuple[float, float]:
    """Return ``(lo, hi)`` bounds in natural-log space for a unit-volume bubble.

    The regressors predict ``log f`` at V = 1, so the clamp is a plain interval
    in log space; rescaling to a real volume (``* V**(-1/3)``) is monotone and
    commutes with it, so clamping here is equivalent to clamping f_real against
    the same-volume Minnaert frequency.
    """
    f_min = float(minnaert_frequency_unit_volume())
    return float(np.log(f_min)), float(np.log(factor * f_min))


def clamp_log_predictions(
    pred_log: np.ndarray, bounds: tuple[float, float] | None
) -> np.ndarray:
    """Clip ``pred_log`` into ``bounds``; a ``None`` bounds is a no-op."""
    if bounds is None:
        return pred_log
    return np.clip(pred_log, bounds[0], bounds[1])


def metrics_from_log(pred_log: np.ndarray, tgt_log: np.ndarray) -> dict[str, float]:
    """Same formula as ``freq_model.NN.training.evaluate_metrics``."""
    pred_log = np.asarray(pred_log, dtype=np.float64).flatten()
    tgt_log = np.asarray(tgt_log, dtype=np.float64).flatten()
    rmse_log = float(np.sqrt(np.mean((pred_log - tgt_log) ** 2)))
    f_pred = np.exp(pred_log)
    f_tgt = np.exp(tgt_log)
    ape = np.abs(f_pred - f_tgt) / (f_tgt + 1e-12) * 100.0
    return {
        "rmse_log": rmse_log,
        "mape": float(np.mean(ape)),
        "max_ape": float(np.max(ape)),
    }


def check_split_against_paper(df_used, split, split_json: Path = SPLIT_JSON) -> int:
    """Hard-fail unless the regenerated test partition equals split.json's."""
    with split_json.open("r", encoding="utf-8") as f:
        saved = json.load(f)
    ids = df_used["mesh_id"].astype(str).tolist()
    regenerated = [ids[i] for i in split.idx_test]
    saved_test = [str(s) for s in saved["mesh_id_test"]]
    if regenerated != saved_test:
        n_match = sum(a == b for a, b in zip(regenerated, saved_test))
        raise SystemExit(
            "FATAL: regenerated test split does not match the paper's split.json "
            f"({n_match}/{len(saved_test)} ids agree in order; "
            f"regenerated n={len(regenerated)}, saved n={len(saved_test)}). "
            "The baseline numbers would not be comparable -- aborting."
        )
    return len(saved_test)


def fit_eval(
    model,
    x_train,
    y_train,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    clamp_bounds: tuple[float, float] | None = None,
):
    """Fit on train, return {split_name: metrics} using the shared MAPE formula."""
    model.fit(x_train, y_train)
    out = {}
    for name, (x, y) in splits.items():
        out[name] = metrics_from_log(
            clamp_log_predictions(model.predict(x), clamp_bounds), y
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--model-dir", type=Path, default=MODEL_DIR,
        help="Trained MLP output dir providing split.json (split assertion) and "
             "metrics.json (reference MLP row). Point at a retrained model's dir "
             "when the dataset differs from the paper's.",
    )
    parser.add_argument(
        "--clamp-to-minnaert", action="store_true", help=CLAMP_FLAG_HELP,
    )
    args = parser.parse_args()
    clamp_bounds = minnaert_log_clamp_bounds() if args.clamp_to_minnaert else None
    if clamp_bounds is not None:
        print(
            f"clamping predictions to log f in [{clamp_bounds[0]:.6f}, {clamp_bounds[1]:.6f}] "
            f"(f_Minnaert .. {MINNAERT_CLAMP_FACTOR} x f_Minnaert at unit volume)"
        )
    model_dir = args.model_dir.resolve()
    split_json = model_dir / "split.json"
    metrics_json = model_dir / "metrics.json"

    x_raw, y_raw, log_fs_raw, df_used = load_xy(args.dataset.resolve())
    split = split_dataset(
        x_raw, y_raw, log_fs_raw,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
    )
    n_test = check_split_against_paper(df_used, split, split_json)
    print(
        f"split matches split.json (n_test={n_test}) | "
        f"train={len(split.x_train)} val={len(split.x_val)} test={len(split.x_test)}"
    )

    # Same preprocessing as the NN trainer: StandardScaler fit on train only.
    # Target stays raw log(f) -- linear-in-parameters models need no y-scaling.
    x_scaler = StandardScaler().fit(split.x_train)
    x_train = x_scaler.transform(split.x_train)
    x_val = x_scaler.transform(split.x_val)
    x_test = x_scaler.transform(split.x_test)
    y_train = split.y_train.astype(np.float64)
    y_val = split.y_val.astype(np.float64)
    y_test = split.y_test.astype(np.float64)

    eval_splits = {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }

    results: dict[str, dict] = {}

    # --- linear -----------------------------------------------------------
    t0 = time.perf_counter()
    results["linear"] = {
        "hparams": {},
        "metrics": fit_eval(
            LinearRegression(), x_train, y_train, eval_splits, clamp_bounds
        ),
        "fit_seconds": time.perf_counter() - t0,
    }
    print(f"[linear] test MAPE = {results['linear']['metrics']['test']['mape']:.4f}%")

    # --- poly2 / poly3: PolynomialFeatures -> ridge, alpha tuned on val ----
    for degree in (2, 3):
        name = f"poly{degree}"
        t0 = time.perf_counter()
        best = None
        for alpha in RIDGE_ALPHAS:
            reg = LinearRegression() if alpha == 0.0 else Ridge(alpha=alpha)
            model = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False), reg
            )
            model.fit(x_train, y_train)
            val_mape = metrics_from_log(
                clamp_log_predictions(model.predict(x_val), clamp_bounds), y_val
            )["mape"]
            if best is None or val_mape < best[0]:
                best = (val_mape, alpha, model)
        val_mape, alpha, model = best
        results[name] = {
            "hparams": {"degree": degree, "ridge_alpha": alpha},
            "metrics": {
                k: metrics_from_log(
                    clamp_log_predictions(model.predict(x), clamp_bounds), y
                )
                for k, (x, y) in eval_splits.items()
            },
            "fit_seconds": time.perf_counter() - t0,
        }
        print(
            f"[{name}] best alpha={alpha:g} (val MAPE {val_mape:.4f}%) | "
            f"test MAPE = {results[name]['metrics']['test']['mape']:.4f}%"
        )

    # --- rbf: kernel ridge, (alpha, gamma) tuned on val --------------------
    t0 = time.perf_counter()
    best = None
    for gamma in KRR_GAMMAS:
        for alpha in KRR_ALPHAS:
            model = KernelRidge(kernel="rbf", alpha=float(alpha), gamma=float(gamma))
            model.fit(x_train, y_train)
            val_mape = metrics_from_log(
                clamp_log_predictions(model.predict(x_val), clamp_bounds), y_val
            )["mape"]
            if best is None or val_mape < best[0]:
                best = (val_mape, float(alpha), float(gamma), model)
    val_mape, alpha, gamma, model = best
    results["rbf"] = {
        "hparams": {"kernel": "rbf", "alpha": alpha, "gamma": gamma},
        "metrics": {
            k: metrics_from_log(
                clamp_log_predictions(model.predict(x), clamp_bounds), y
            )
            for k, (x, y) in eval_splits.items()
        },
        "fit_seconds": time.perf_counter() - t0,
    }
    print(
        f"[rbf] best alpha={alpha:g} gamma={gamma:g} (val MAPE {val_mape:.4f}%) | "
        f"test MAPE = {results['rbf']['metrics']['test']['mape']:.4f}%"
    )

    # --- reference NN row from the paper's metrics.json --------------------
    with metrics_json.open("r", encoding="utf-8") as f:
        nn_metrics = json.load(f)["test_metrics"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(args.dataset),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "feature_cols": FEATURE_COLS,
        "split_check": f"test partition identical to {split_json} (n_test={n_test})",
        "n_train": int(len(split.x_train)),
        "n_val": int(len(split.x_val)),
        "n_test": int(len(split.x_test)),
        "tuning": "hyperparameters selected by validation MAPE; test untouched",
        "minnaert_clamp": (
            {
                "factor": MINNAERT_CLAMP_FACTOR,
                "log_lo": clamp_bounds[0],
                "log_hi": clamp_bounds[1],
            }
            if clamp_bounds is not None
            else None
        ),
        "models": results,
        "reference_mlp_test_metrics": nn_metrics,
    }
    json_path = args.out_dir / "baseline_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    display = {
        "linear": "Linear regression",
        "poly2": "Polynomial (deg 2) + ridge",
        "poly3": "Polynomial (deg 3) + ridge",
        "rbf": "RBF kernel ridge",
    }
    lines = [
        "# Non-neural baselines on the 8-feature frequency task",
        "",
        f"Same features, split (seed {args.seed}, 70/15/15, verified against "
        f"`{split_json}`), target log(f_BEM), and MAPE formula as the paper's MLP. "
        "Hyperparameters tuned on the validation split only."
        + (
            f" Predictions clamped to [f_Minnaert, {MINNAERT_CLAMP_FACTOR} x f_Minnaert]."
            if clamp_bounds is not None
            else ""
        ),
        "",
        "| Model | Test MAPE (%) | RMSE(log f) | Max APE (%) |",
        "|---|---|---|---|",
    ]
    for key, label in display.items():
        m = results[key]["metrics"]["test"]
        lines.append(
            f"| {label} | {m['mape']:.3f} | {m['rmse_log']:.5f} | {m['max_ape']:.2f} |"
        )
    lines.append(
        f"| **MLP (paper)** | **{nn_metrics['mape']:.3f}** | "
        f"{nn_metrics['rmse_log']:.5f} | {nn_metrics['max_ape']:.2f} |"
    )
    md_path = args.out_dir / "baseline_metrics.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nSaved: {json_path}\n       {md_path}")


if __name__ == "__main__":
    main()

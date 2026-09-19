"""Fit and serialize the four baseline regressors as joblib artifacts.

Companion to ``baseline_regressors_8feat.py``: re-fits linear / poly2 / poly3 /
RBF-kernel-ridge on the paper's train split (seed 42, asserted against
``split.json``) using the validation-selected hyperparameters recorded in
``results/experiments/table02_regressor_comparison/baseline_metrics.json`` -- no
grid search is repeated, so this runs in well under a minute.

Artifacts land in ``python/freq_model/output/output_baseline_regressors_8feature/``
(mirroring the ``output_*`` convention of the NN trainers):

* ``feature_scaler.joblib``  -- StandardScaler fit on the train split
* ``linear.joblib`` / ``poly2.joblib`` / ``poly3.joblib`` / ``rbf.joblib``
* ``export_config.json``     -- hyperparameters + provenance

Inference contract (identical to the NN8 surrogate): standardize the 8
features with ``feature_scaler``, ``model.predict`` returns ``log f`` at unit
volume, then ``f_real = exp(log_f) * V_target ** (-1/3)``.

``--clamp-to-minnaert`` (OFF by default) records a
``[f_Minnaert, 1.35 * f_Minnaert]`` window in ``export_config.json`` for callers
to apply; the exported .joblib models never clamp themselves, and the
recorded-MAPE assertion always runs on unclamped predictions.

RUN (repo root; $env:KMP_DUPLICATE_LIB_OK = "TRUE" on Windows):
    python -u python/freq_model/regression/export_baseline_models_8feat.py
"""

from __future__ import annotations

import torch  # noqa: F401  (torch first: Windows OpenMP DLL ordering)

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

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

from freq_model.regression.baseline_regressors_8feat import (  # noqa: E402
    CLAMP_FLAG_HELP,
    DEFAULT_DATASET,
    MINNAERT_CLAMP_FACTOR,
    check_split_against_paper,
    clamp_log_predictions,
    load_xy,
    metrics_from_log,
    minnaert_log_clamp_bounds,
)
from freq_model.NN.bubble_dataset import split_dataset  # noqa: E402
from freq_model.NN.fit_shape_freq_model import (  # noqa: E402
    FEATURE_COLS,
)

METRICS_JSON = (
    _REPO_ROOT / "results" / "experiments" / "table02_regressor_comparison" / "baseline_metrics.json"
)
MODEL_DIR = (
    _FREQ_MODEL_DIR / "output" / "output_8feature_direct_bubblegym_10k"
)
OUT_DIR = _FREQ_MODEL_DIR / "output" / "output_baseline_regressors_8feature"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metrics-json", type=Path, default=METRICS_JSON,
                        help="baseline_metrics.json supplying the val-selected hparams.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                        help="Trained MLP output dir providing split.json for the split assertion.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--clamp-to-minnaert", action="store_true", help=CLAMP_FLAG_HELP)
    args = parser.parse_args()
    clamp_bounds = minnaert_log_clamp_bounds() if args.clamp_to_minnaert else None
    dataset = args.dataset.resolve()
    metrics_json = args.metrics_json.resolve()
    out_dir = args.out_dir.resolve()

    with metrics_json.open("r", encoding="utf-8") as f:
        chosen = json.load(f)["models"]

    x_raw, y_raw, log_fs_raw, df_used = load_xy(dataset)
    split = split_dataset(
        x_raw, y_raw, log_fs_raw, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    n_test = check_split_against_paper(df_used, split, args.model_dir.resolve() / "split.json")
    print(f"split matches split.json (n_test={n_test})")

    x_scaler = StandardScaler().fit(split.x_train)
    x_train = x_scaler.transform(split.x_train)
    x_test = x_scaler.transform(split.x_test)
    y_train = split.y_train.astype(np.float64)
    y_test = split.y_test.astype(np.float64)

    def _poly(degree: int):
        alpha = float(chosen[f"poly{degree}"]["hparams"]["ridge_alpha"])
        reg = LinearRegression() if alpha == 0.0 else Ridge(alpha=alpha)
        return make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False), reg
        )

    models = {
        "linear": LinearRegression(),
        "poly2": _poly(2),
        "poly3": _poly(3),
        "rbf": KernelRidge(
            kernel="rbf",
            alpha=float(chosen["rbf"]["hparams"]["alpha"]),
            gamma=float(chosen["rbf"]["hparams"]["gamma"]),
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(x_scaler, out_dir / "feature_scaler.joblib")
    exported: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        # Reproduce the recorded test MAPE as a serialization sanity check.
        test_mape = metrics_from_log(model.predict(x_test), y_test)["mape"]
        recorded = float(chosen[name]["metrics"]["test"]["mape"])
        if abs(test_mape - recorded) > 1e-6:
            raise SystemExit(
                f"{name}: refit test MAPE {test_mape:.6f}% != recorded "
                f"{recorded:.6f}% -- hyperparameters or data drifted, aborting."
            )
        joblib.dump(model, out_dir / f"{name}.joblib")
        exported[name] = {
            "hparams": chosen[name]["hparams"],
            "test_mape": test_mape,
        }
        clamp_note = ""
        if clamp_bounds is not None:
            # The recorded-MAPE assertion above deliberately uses the UNCLAMPED
            # predictions, so the artifacts stay verifiable against the shipped
            # metrics regardless of this flag.
            clamped_mape = metrics_from_log(
                clamp_log_predictions(model.predict(x_test), clamp_bounds), y_test
            )["mape"]
            exported[name]["test_mape_clamped"] = clamped_mape
            clamp_note = f" | clamped {clamped_mape:.4f}%"
        print(
            f"  {name}: test MAPE {test_mape:.4f}% (matches recorded){clamp_note} "
            f"-> {name}.joblib"
        )

    with (out_dir / "export_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(dataset),
                "seed": 42,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "feature_cols": FEATURE_COLS,
                "target": "log(f_BEM) at unit volume",
                "inference": "f_real = exp(model.predict(scaler.transform(X))) * V_target**(-1/3)",
                "minnaert_clamp": (
                    {
                        "factor": MINNAERT_CLAMP_FACTOR,
                        "log_lo": clamp_bounds[0],
                        "log_hi": clamp_bounds[1],
                        "note": "clip the predicted log f into [log_lo, log_hi] before "
                                "the V**(-1/3) rescale; the exported .joblib models do "
                                "NOT apply it themselves",
                    }
                    if clamp_bounds is not None
                    else None
                ),
                "models": exported,
                "source_metrics": str(metrics_json.relative_to(_REPO_ROOT)),
            },
            f,
            indent=2,
        )
    print(f"saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()

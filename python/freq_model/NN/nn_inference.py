"""Load the trained BubbleFreqNet artifact and run it over per-frame features.

An *artifact directory* is what ``fit_shape_freq_model.py``
writes:

    train_config.json        feature_cols, baseline_kind, model_checkpoint
    <checkpoint>.pt          BubbleFreqNet state dict
    feature_scaler.joblib    z-score over the input features
    target_log_scaler.joblib z-score over the (log) target

The model is the paper's production surrogate (Sec. 5.2, Eq. 16): the
8-dimensional shape descriptor ``q_8`` of Eq. 11 mapped to ``log f`` directly,
with no analytical baseline to add back. :func:`load_nn_artifacts` validates
``feature_cols`` and ``baseline_kind`` against what is expected, so pointing a
driver at some other artifact directory fails loudly rather than silently
mispredicting.

Predictions are **at unit volume**; callers recover the physical frequency with
``f = f_unit * V^(-1/3)`` (Sec. 4.1). Frames whose features were missing or
non-finite come back as NaN with ``valid[i] = False``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "EXPECTED_FEATURE_COLS_8FEAT",
    "load_nn_artifacts",
    "load_nn_chull_inertia_8feat_direct",
    "predict_nn_unit_for_frames_inertia_8feat_direct",
]


# Eq. 11: two inertia ratios (Eq. 6), three Wadell-style sphericity
# complements (Eq. 8), three convex-hull ratios (Eq. 10). Order is the exact
# column order the trainer used and the scalers were fit on.
EXPECTED_FEATURE_COLS_8FEAT = [
    "i11_over_i00",
    "i22_over_i00",
    "non_sph_va",
    "non_sph_vm",
    "non_sph_w",
    "eta_V",
    "eta_A",
    "eta_M",
]


def _bubble_freq_net_cls():
    """Import ``BubbleFreqNet`` whether or not ``freq_model/`` is on sys.path."""
    try:
        from freq_model.NN.bub_freq_net import BubbleFreqNet  # type: ignore
    except ImportError:  # pragma: no cover - legacy sys.path layout
        from freq_model.NN.bub_freq_net import BubbleFreqNet  # type: ignore
    return BubbleFreqNet


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_nn_artifacts(
    artifact_dir: Path,
    expected_feature_cols: list[str],
    expected_baseline_kind: str | None = None,
):
    """Load ``(model, feature_scaler, target_scaler, feature_cols)``.

    Validates that ``train_config.json`` reports exactly
    ``expected_feature_cols`` and, when ``expected_baseline_kind`` is given,
    that ``baseline_kind`` matches. Architecture is the paper's
    ``BubbleFreqNet`` (hidden 64 -> 32, dropout 0.1, Sec. 5.2.2).
    """
    artifact_dir = Path(artifact_dir)
    cfg_path = artifact_dir / "train_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    feature_cols = list(cfg.get("feature_cols", []))
    if feature_cols != expected_feature_cols:
        raise ValueError(
            f"Expected NN feature columns {expected_feature_cols}, got "
            f"{feature_cols} ({cfg_path})"
        )
    if expected_baseline_kind is not None:
        got = str(cfg.get("baseline_kind", "")).strip()
        if got != expected_baseline_kind:
            raise ValueError(
                f"Expected baseline_kind={expected_baseline_kind!r} in "
                f"{cfg_path}, got {got!r}"
            )

    ckpt = artifact_dir / str(cfg.get("model_checkpoint", "")).strip()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    import joblib  # type: ignore
    import torch  # type: ignore

    feature_scaler = joblib.load(artifact_dir / "feature_scaler.joblib")
    target_scaler = joblib.load(artifact_dir / "target_log_scaler.joblib")

    model = _bubble_freq_net_cls()(
        input_dim=len(feature_cols), hidden_dim=64, hidden_dim2=32, dropout=0.1
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, feature_scaler, target_scaler, feature_cols


def load_nn_chull_inertia_8feat_direct(artifact_dir: Path):
    """Load the production model: 8 features, ``log(f)`` direct target (Eq. 16)."""
    return load_nn_artifacts(
        artifact_dir, EXPECTED_FEATURE_COLS_8FEAT, "none_log_direct"
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_nn_unit_for_frames_inertia_8feat_direct(
    feats_by_frame: list[dict[str, Any]],
    inertia_principal_unit_per_frame: list[np.ndarray],
    artifact_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the 8-feature model over all valid frames (Eq. 16).

    ``feats_by_frame`` is what
    :func:`shape_feature.build_dataset.extract_features_one` produces. The
    two inertia features ``i11_over_i00`` / ``i22_over_i00`` are derived here
    from each frame's volume-normalized principal moments, sorted ascending so
    ``I00 <= I11 <= I22`` -- matching the dataset's ``i00, i11, i22`` columns
    (Eq. 6). Frames with non-finite convex-hull features or a non-positive I00
    are marked invalid.

    Returns ``(f_unit, valid_mask)``.
    """
    import torch  # type: ignore

    n = len(feats_by_frame)
    f_unit = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return f_unit, valid
    if len(inertia_principal_unit_per_frame) != n:
        raise ValueError(
            f"len(inertia_principal_unit_per_frame)="
            f"{len(inertia_principal_unit_per_frame)} != len(feats_by_frame)={n}"
        )

    model, feature_scaler, target_scaler, feature_cols = (
        load_nn_chull_inertia_8feat_direct(artifact_dir)
    )

    chull_cols = [c for c in feature_cols if c not in ("i11_over_i00", "i22_over_i00")]
    cols = {c: np.full(n, np.nan, dtype=np.float64) for c in feature_cols}

    for i, feats in enumerate(feats_by_frame):
        if feats.get("status") != "ok":
            continue
        try:
            chull_row = [float(feats[c]) for c in chull_cols]
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(x) for x in chull_row):
            continue

        I_principal = np.asarray(inertia_principal_unit_per_frame[i], dtype=np.float64)
        if (
            I_principal.shape != (3,)
            or not np.all(np.isfinite(I_principal))
            or float(I_principal[0]) <= 0.0
        ):
            continue
        i00 = float(I_principal[0])
        i11_over_i00 = float(I_principal[1]) / i00
        i22_over_i00 = float(I_principal[2]) / i00
        if not (math.isfinite(i11_over_i00) and math.isfinite(i22_over_i00)):
            continue

        cols["i11_over_i00"][i] = i11_over_i00
        cols["i22_over_i00"][i] = i22_over_i00
        for c, x in zip(chull_cols, chull_row):
            cols[c][i] = x
        valid[i] = True

    if not np.any(valid):
        return f_unit, valid

    X = np.column_stack([cols[c][valid] for c in feature_cols])
    Xn = feature_scaler.transform(X.astype(np.float32)).astype(np.float32)
    with torch.no_grad():
        y_norm = model(torch.from_numpy(Xn)).numpy().reshape(-1, 1)
    log_pred = target_scaler.inverse_transform(y_norm).flatten().astype(np.float64)
    f_unit[valid] = np.exp(log_pred)
    return f_unit, valid

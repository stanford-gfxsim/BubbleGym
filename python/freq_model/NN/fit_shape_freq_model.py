"""Train BubbleFreqNet directly on log(f_BEM) over an 8-feature mix.

Eight features per sample (in this exact column order; the inertia ratios
come first so they are easy to spot in the scaler diagnostics):

  2 inertia-ratio features (volume-normalized principal moments, sorted
  ascending so I00 <= I11 <= I22):
      i11_over_i00 = I11 / I00
      i22_over_i00 = I22 / I00

  3 Wadell-style (sphere-perturbation) axes:
      non_sph_va = 1 - Phi_VA
      non_sph_vm = 1 - Phi_VM
      non_sph_w  = 1 - 4*pi / W_vertex

  3 convex-hull (envelope-deviation) axes:
      eta_V = V(bubble) / V(hull)
      eta_A = A(bubble) / A(hull)
      eta_M = M(bubble) / M(hull)

The target is log(f_BEM) directly (z-scored), with NO log-baseline subtraction
(``baseline_kind = "none_log_direct"``). The shared training loop in
``training.py`` is written around a log-baseline reconstruction, so this
trainer passes an all-zeros ``log_fs`` to make that step a no-op.

This is the trainer for the shipped model. It produces the artifact directory
consumed by ``nn_inference.py`` and the audio pipeline; ``ablation_study/``
reuses its ``load_xy`` and ``FEATURE_COLS`` so the ablations share this exact
feature pipeline.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys

_THIS_DIR = Path(__file__).resolve().parent
_PYTHON_ROOT = _THIS_DIR.parents[1]
for _p in (_PYTHON_ROOT, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from freq_model.NN.bub_freq_net import BubbleFreqNet  # noqa: E402
from freq_model.NN.bubble_dataset import (  # noqa: E402
    BubbleDataset,
    canonicalize_columns as _canonicalize_columns,
    set_seed,
    split_dataset,
)
from shape_feature.nonspherical_features import (  # noqa: E402
    non_sph_va_from_area_volume as _non_sph_va,
)
from freq_model.NN.training import (  # noqa: E402
    collect_log_predictions,
    evaluate_metrics,
    train,
)
from utils.worst_case_utils import export_worst_cases  # noqa: E402


TARGET_COL = "frequency"
SURFACE_AREA_CANON_COL = "surface_area"
PHI_VM_COL = "Phi_VM"
WILLMORE_COL = "W_vertex"

# Strasberg baseline column is required ONLY so we apply the same row filter as the
# Strasberg-residual trainer. This guarantees (with seed=42 and the same hparams)
# that split_dataset() produces an identical train/val/test partition across the
# direct/residual/8-feature variants, so the comparisons share one test set.
STRASBERG_FILTER_COL = "f_strasberg"

INERTIA_FEATURE_COLS = ["i11_over_i00", "i22_over_i00"]
WADELL_FEATURE_COLS = ["non_sph_va", "non_sph_vm", "non_sph_w"]
CHULL_FEATURE_COLS = ["eta_V", "eta_A", "eta_M"]
FEATURE_COLS = INERTIA_FEATURE_COLS + WADELL_FEATURE_COLS + CHULL_FEATURE_COLS

INERTIA_RAW_COLS = ["i00", "i11", "i22"]

REQUIRED_RAW_COLS = [
    SURFACE_AREA_CANON_COL,
    "volume",
    PHI_VM_COL,
    WILLMORE_COL,
    "eta_V", "eta_A", "eta_M",
    *INERTIA_RAW_COLS,
    TARGET_COL,
]

BASELINE_KIND = "none_log_direct"
MODEL_FILENAME = "bubble_freq_net_best.pt"

THUMBNAIL_DIR_BY_SOURCE = {
    "VOF": Path("dataset/bubble_gym/bubble_mesh_thumbnails_400x400/VOF"),
    "LBM": Path("results/lbm_exhale_thumbnails"),
}


def load_xy(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(dataset_path)
    df = _canonicalize_columns(df)

    missing = [c for c in REQUIRED_RAW_COLS + [STRASBERG_FILTER_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Run python/shape_feature/build_dataset.py, "
            "add_baseline_frequencies_to_dataset.py, and "
            "the shipped dataset/bubble_gym/dataset_bubblegym_10k.csv."
        )

    cols = [*REQUIRED_RAW_COLS, STRASBERG_FILTER_COL]
    for opt in ("index", "mesh_filename", "mesh_id", "source"):
        if opt in df.columns and opt not in cols:
            cols = [opt, *cols]

    data = df[cols].copy()
    str_cols = {c for c in ("mesh_filename", "mesh_id", "source") if c in data.columns}
    numeric_cols = [c for c in data.columns if c not in str_cols]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols)
    data = data[data[TARGET_COL] > 0.0]
    data = data[data[STRASBERG_FILTER_COL] > 0.0]
    data = data[data[SURFACE_AREA_CANON_COL] > 0.0]
    data = data[data["volume"] > 0.0]
    data = data[data[PHI_VM_COL] > 0.0]
    data = data[data[WILLMORE_COL] > 0.0]
    data = data[data["eta_V"] > 0.0]
    data = data[data["eta_A"] > 0.0]
    data = data[data["eta_M"] > 0.0]
    # Inertia ratios require strictly positive I00; reject any borderline /
    # degenerate rows so the divide is well-defined and the resulting ratios
    # are finite.
    data = data[data["i00"] > 0.0]
    data = data[data["i11"] > 0.0]
    data = data[data["i22"] > 0.0]

    area = data[SURFACE_AREA_CANON_COL].to_numpy(dtype=np.float64)
    volume = data["volume"].to_numpy(dtype=np.float64)
    phi_vm = data[PHI_VM_COL].to_numpy(dtype=np.float64)
    w_vertex = data[WILLMORE_COL].to_numpy(dtype=np.float64)
    eta_V = data["eta_V"].to_numpy(dtype=np.float64)
    eta_A = data["eta_A"].to_numpy(dtype=np.float64)
    eta_M = data["eta_M"].to_numpy(dtype=np.float64)
    i00 = data["i00"].to_numpy(dtype=np.float64)
    i11 = data["i11"].to_numpy(dtype=np.float64)
    i22 = data["i22"].to_numpy(dtype=np.float64)

    non_sph_va = _non_sph_va(area, volume, context=str(dataset_path))
    non_sph_vm = 1.0 - phi_vm
    non_sph_w = 1.0 - (4.0 * np.pi / w_vertex)

    i11_over_i00 = i11 / i00
    i22_over_i00 = i22 / i00

    feature_matrix = np.column_stack(
        [
            i11_over_i00, i22_over_i00,
            non_sph_va, non_sph_vm, non_sph_w,
            eta_V, eta_A, eta_M,
        ]
    )
    finite = np.all(np.isfinite(feature_matrix), axis=1)
    finite &= non_sph_va >= 0.0
    finite &= non_sph_vm >= 0.0
    finite &= non_sph_w >= 0.0
    finite &= eta_V > 0.0
    finite &= eta_A > 0.0
    finite &= eta_M > 0.0
    # Sphere has I11/I00 = I22/I00 = 1; any volume-normalized inertia ratio
    # below one indicates a column-order mismatch upstream, so we reject it
    # rather than feeding negative-stretch garbage to the network.
    finite &= i11_over_i00 >= 1.0 - 1e-6
    finite &= i22_over_i00 >= 1.0 - 1e-6
    data = data.loc[finite].reset_index(drop=True)
    x = feature_matrix[finite].astype(np.float32)

    f_bem = data[TARGET_COL].to_numpy(dtype=np.float64)
    # No baseline subtraction: y is log(f_BEM); log_fs is zero so reconstruction
    # via pred_log = pred_log_resid + log_fs is just pred_log = pred_log_resid.
    log_fs = np.zeros_like(f_bem, dtype=np.float32)
    y_direct = np.log(f_bem).astype(np.float32)
    return x, y_direct, log_fs, data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train BubbleFreqNet directly on log(f_BEM) over EIGHT features: "
            "2 inertia-ratio (i11/i00, i22/i00) + 3 Wadell-style "
            "(non_sph_va, non_sph_vm, non_sph_w) + 3 convex-hull "
            "(eta_V, eta_A, eta_M). No Strasberg / Minnaert residual."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/bubble_gym/dataset_bubblegym_10k.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "python/freq_model/output/output_8feature_direct_bubblegym_10k"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-run", action="store_true", help="Quick end-to-end check using a small subset.")
    parser.add_argument("--test-run-samples", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    if args.test_run:
        output_dir = output_dir / f"test_run_{args.test_run_samples}"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_raw, y_raw, log_fs_raw, df_used = load_xy(dataset_path)

    if args.test_run:
        n_use = min(args.test_run_samples, len(x_raw))
        if n_use < 10:
            raise ValueError(f"Not enough samples for --test-run: {n_use}")
        rng = np.random.default_rng(args.seed)
        subset_idx = rng.permutation(len(x_raw))[:n_use]
        x_raw = x_raw[subset_idx]
        y_raw = y_raw[subset_idx]
        log_fs_raw = log_fs_raw[subset_idx]
        df_used = df_used.iloc[subset_idx].reset_index(drop=True)

    split = split_dataset(
        x_raw,
        y_raw,
        log_fs_raw,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    x_scaler = StandardScaler().fit(split.x_train)
    y_scaler = StandardScaler().fit(split.y_train.reshape(-1, 1))

    x_train = x_scaler.transform(split.x_train).astype(np.float32)
    x_val = x_scaler.transform(split.x_val).astype(np.float32)
    x_test = x_scaler.transform(split.x_test).astype(np.float32)
    y_train = y_scaler.transform(split.y_train.reshape(-1, 1)).flatten().astype(np.float32)
    y_val = y_scaler.transform(split.y_val.reshape(-1, 1)).flatten().astype(np.float32)
    y_test = y_scaler.transform(split.y_test.reshape(-1, 1)).flatten().astype(np.float32)

    train_loader = DataLoader(
        BubbleDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        BubbleDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        BubbleDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = BubbleFreqNet(input_dim=len(FEATURE_COLS), hidden_dim=64, hidden_dim2=32, dropout=0.1)
    model_save_path = output_dir / MODEL_FILENAME

    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("dataset/path", str(dataset_path), 0)
    writer.add_text("features", ", ".join(FEATURE_COLS), 0)
    writer.add_text("baseline_kind", BASELINE_KIND, 0)
    if "source" in df_used.columns:
        src_counts = df_used["source"].astype(str).value_counts().to_dict()
        writer.add_text(
            "dataset/source_counts",
            ", ".join(f"{k}={v}" for k, v in src_counts.items()),
            0,
        )

    print(f"Loaded samples: {len(df_used)}")
    if "source" in df_used.columns:
        print("Per-source sample counts:")
        print(df_used["source"].astype(str).value_counts().to_string())
    if args.test_run:
        print(f"Test run mode enabled: using {len(df_used)} samples")
    print(
        f"Split -> train: {len(split.x_train)}, val: {len(split.x_val)}, "
        f"test: {len(split.x_test)} | device: {args.device}"
    )
    print(f"Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")
    print(f"Baseline kind: {BASELINE_KIND} (target = log(f_BEM); no residual)")

    history, best_epoch = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_scaler=y_scaler,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        writer=writer,
        model_save_path=model_save_path,
        val_log_fs=split.log_fs_val,
    )

    test_metrics = evaluate_metrics(
        model,
        test_loader,
        y_scaler=y_scaler,
        device=args.device,
        log_fs=split.log_fs_test,
    )
    writer.add_hparams(
        {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        },
        {
            "hparam/rmse_log": test_metrics["rmse_log"],
            "hparam/mape": test_metrics["mape"],
            "hparam/max_ape": test_metrics["max_ape"],
        },
    )
    writer.flush()
    writer.close()

    joblib.dump(x_scaler, output_dir / "feature_scaler.joblib")
    joblib.dump(y_scaler, output_dir / "target_log_scaler.joblib")
    cfg = vars(args)
    cfg["baseline_kind"] = BASELINE_KIND
    cfg["baseline_col"] = ""
    cfg["feature_cols"] = FEATURE_COLS
    cfg["model_checkpoint"] = MODEL_FILENAME
    cfg["willmore_source_col"] = WILLMORE_COL
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)

    split_out = {
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "n_total_used": int(len(df_used)),
        "idx_train": split.idx_train.astype(int).tolist(),
        "idx_val": split.idx_val.astype(int).tolist(),
        "idx_test": split.idx_test.astype(int).tolist(),
    }
    if "index" in df_used.columns:
        ds_idx = df_used["index"].astype(int).to_numpy()
        split_out["idx_train"] = ds_idx[split.idx_train].astype(int).tolist()
        split_out["idx_val"] = ds_idx[split.idx_val].astype(int).tolist()
        split_out["idx_test"] = ds_idx[split.idx_test].astype(int).tolist()
    if "mesh_filename" in df_used.columns:
        names = df_used["mesh_filename"].astype(str).tolist()
        split_out["mesh_filename_train"] = [names[i] for i in split.idx_train]
        split_out["mesh_filename_val"] = [names[i] for i in split.idx_val]
        split_out["mesh_filename_test"] = [names[i] for i in split.idx_test]
    if "mesh_id" in df_used.columns:
        ids = df_used["mesh_id"].astype(str).tolist()
        split_out["mesh_id_train"] = [ids[i] for i in split.idx_train]
        split_out["mesh_id_val"] = [ids[i] for i in split.idx_val]
        split_out["mesh_id_test"] = [ids[i] for i in split.idx_test]
    if "source" in df_used.columns:
        srcs = df_used["source"].astype(str).tolist()
        split_out["source_train"] = [srcs[i] for i in split.idx_train]
        split_out["source_val"] = [srcs[i] for i in split.idx_val]
        split_out["source_test"] = [srcs[i] for i in split.idx_test]
    with (output_dir / "split.json").open("w", encoding="utf-8") as f:
        json.dump(split_out, f, indent=2)

    pred_log, tgt_log = collect_log_predictions(
        model,
        test_loader,
        y_scaler=y_scaler,
        device=args.device,
        log_fs=split.log_fs_test,
    )
    pred_freq = np.exp(pred_log)
    tgt_freq = np.exp(tgt_log)
    worst_dir = Path(
        "results/worst_cases_8feature_direct"
    )
    worst_rows = export_worst_cases(
        df_used,
        split.idx_test,
        pred_freq,
        tgt_freq,
        worst_dir,
        top_k=10,
        thumbnail_dir=Path("dataset/bubble_gym/bubble_mesh_thumbnails_400x400/VOF"),
        thumbnail_dir_by_source=THUMBNAIL_DIR_BY_SOURCE,
    )
    test_per_source: dict[str, dict[str, float]] = {}
    if "source" in df_used.columns:
        df_test = df_used.iloc[split.idx_test].reset_index(drop=True)
        df_test = df_test.assign(
            pred_freq_hz=pred_freq,
            gt_freq_hz=tgt_freq,
            ape_pct=np.abs(pred_freq - tgt_freq) / np.maximum(tgt_freq, 1e-12) * 100.0,
        )
        for src, grp in df_test.groupby("source"):
            ape = grp["ape_pct"].to_numpy(dtype=np.float64)
            log_diff = np.log(grp["pred_freq_hz"].to_numpy(dtype=np.float64)) - np.log(
                grp["gt_freq_hz"].to_numpy(dtype=np.float64)
            )
            test_per_source[str(src)] = {
                "count": int(len(grp)),
                "rmse_log": float(np.sqrt(np.mean(log_diff ** 2))),
                "mape": float(np.mean(ape)),
                "max_ape": float(np.max(ape)) if len(ape) else float("nan"),
            }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "test_metrics": test_metrics,
                "test_metrics_per_source": test_per_source,
                "history_len": len(history["train_loss"]),
                "feature_cols": FEATURE_COLS,
                "model_checkpoint": MODEL_FILENAME,
                "baseline_kind": BASELINE_KIND,
                "baseline_col": "",
                "willmore_source_col": WILLMORE_COL,
                "worst_cases_dir": str(worst_dir.as_posix()),
                "worst_cases_count": int(len(worst_rows)),
            },
            f,
            indent=2,
        )

    print("\n=== Test Metrics ===")
    print(f"RMSE(log f): {test_metrics['rmse_log']:.5f}")
    print(f"MAPE:        {test_metrics['mape']:.2f}%")
    print(f"Max APE:     {test_metrics['max_ape']:.2f}%")
    if test_per_source:
        print("\n=== Per-source Test Metrics ===")
        for src, m in test_per_source.items():
            print(
                f"{src:>16s}  n={m['count']:>5d}  RMSE(log)={m['rmse_log']:.5f}  "
                f"MAPE={m['mape']:.2f}%  MaxAPE={m['max_ape']:.2f}%"
            )
    print("\nSaved artifacts:")
    print(f"- model:      {model_save_path}")
    print(f"- scalers:    {output_dir / 'feature_scaler.joblib'}")
    print(f"              {output_dir / 'target_log_scaler.joblib'}")
    print(f"- tensorboard:{tb_dir}")
    print(f"- metrics:    {output_dir / 'metrics.json'}")
    print(f"- worst cases:{worst_dir}")


if __name__ == "__main__":
    main()

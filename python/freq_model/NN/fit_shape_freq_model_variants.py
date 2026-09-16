"""Train the BubbleFreqNet variants the shipped production trainer cannot: the
ellipsoid-proxy log-residual objective, and either objective over a split with a
named set of bubbles held out.

Both learned series of the paper's Fig. 4 are trained here. The objectives
differ only in what the network is asked to predict:

``--baseline strasberg``  (default) the residual objective ---
    y = log(f_BEM) - log(f_strasberg),  f_pred = f_strasberg * exp(y_pred)
``--baseline none``       the direct objective, matching the shipped
    ``fit_shape_freq_model.py`` --- y = log(f_BEM), f_pred = exp(y_pred)

``training.py`` was written around the first and defaults to it (``BASELINE_COL``
/ ``BASELINE_KIND`` there); the direct case is the one that passes an all-zeros
``log_fs``. Row filtering and feature construction are imported wholesale from
the production trainer, so every variant sees exactly the same rows in the same
order.

Two feature sets:

``--features 8``  (default) the production set --- 2 inertia ratios, 3
                  Wadell-style axes, 3 convex-hull axes.
``--features 6``  the set the published Fig. 4 models used --- the same list
                  without the two inertia ratios.

``--holdout-mesh-ids`` pins a named set of bubbles outside train and val, so a
fixed evaluation set stays genuinely held out. That is what makes an honest
Fig. 4 retrain possible: 51 of that figure's 100 bubbles fall in the reference
split's *training* set, so scoring them against a model trained on that split
would report training error for half the panel. The held-out rows are written to
``holdout_predictions.csv`` with per-row APE. Without the flag the split is the
plain seeded 70/15/15.

``--baseline none`` here is NOT the shipped production model: the holdout changes
the split, so its checkpoint is for controlled comparison, not for release.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_THIS_DIR = Path(__file__).resolve().parent
_PYTHON_ROOT = _THIS_DIR.parents[1]
for _p in (_PYTHON_ROOT, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from freq_model.NN.bub_freq_net import BubbleFreqNet  # noqa: E402
from freq_model.NN.bubble_dataset import (  # noqa: E402
    BubbleDataset,
    SplitData,
    set_seed,
    split_dataset,
)
from freq_model.NN.fit_shape_freq_model import (  # noqa: E402
    FEATURE_COLS as FEATURE_COLS_8,
    load_xy,
)
from freq_model.NN.training import (  # noqa: E402
    collect_log_predictions,
    evaluate_metrics,
    train,
)

BASELINE_COL = "f_strasberg"
BASELINE_KIND = {"strasberg": "strasberg_log_residual", "none": "none_log_direct"}

# FEATURE_COLS_8 is [i11_over_i00, i22_over_i00, <3 Wadell>, <3 convex hull>],
# so the published six-feature set is exactly its tail.
FEATURE_COLS_6 = FEATURE_COLS_8[2:]
assert FEATURE_COLS_6 == ["non_sph_va", "non_sph_vm", "non_sph_w", "eta_V", "eta_A", "eta_M"]


MODEL_FILENAME = "bubble_freq_net_best.pt"


def _split_with_holdout(
    x: np.ndarray,
    y: np.ndarray,
    log_fs: np.ndarray,
    mesh_ids: np.ndarray,
    holdout: set[str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[SplitData, np.ndarray]:
    """Seeded split over the rows NOT in ``holdout``; returns the holdout rows too.

    The holdout rows never reach train or val, so a model trained here can be
    evaluated on them honestly. They are not folded into the test split either,
    which keeps the reported test metric comparable to a run without a holdout.
    """
    is_held = np.array([m in holdout for m in mesh_ids], dtype=bool)
    held_pos = np.flatnonzero(is_held)
    free_pos = np.flatnonzero(~is_held)

    sub = split_dataset(
        x[free_pos], y[free_pos], log_fs[free_pos],
        val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
    )
    # Re-express the split's indices in the ORIGINAL row space.
    sub.idx_train = free_pos[sub.idx_train]
    sub.idx_val = free_pos[sub.idx_val]
    sub.idx_test = free_pos[sub.idx_test]
    return sub, held_pos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path,
                        default=Path("dataset/bubble_gym/dataset_bubblegym_10k.csv"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: python/freq_model/output/fig04_retrain/"
                             "<N>feature_<objective>")
    parser.add_argument("--features", type=int, choices=(6, 8), default=8)
    parser.add_argument("--baseline", choices=("strasberg", "none"), default="strasberg",
                        help="'strasberg' learns the ellipsoid-proxy log-residual; "
                             "'none' learns log(f_BEM) directly.")
    parser.add_argument("--holdout-mesh-ids", type=Path, default=None,
                        help="CSV with a mesh_id_10k (or mesh_id) column; those rows are "
                             "kept out of train and val and scored separately.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    feature_cols = FEATURE_COLS_8 if args.features == 8 else FEATURE_COLS_6
    objective = "residual" if args.baseline == "strasberg" else "direct"
    baseline_kind = BASELINE_KIND[args.baseline]
    if args.output_dir is None:
        args.output_dir = Path(
            f"python/freq_model/output/fig04_retrain/{args.features}feature_{objective}")

    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared row filter + feature construction with the production trainer.
    x8, y_direct, zero_log_fs, df_used = load_xy(args.dataset.resolve())
    x_raw = x8 if args.features == 8 else x8[:, 2:]

    f_base = df_used[BASELINE_COL].to_numpy(dtype=np.float64)
    if not np.all(f_base > 0.0):
        raise ValueError(f"{BASELINE_COL} must be strictly positive after load_xy's filter")
    if args.baseline == "strasberg":
        log_fs_raw = np.log(f_base).astype(np.float32)
        y_raw = (y_direct.astype(np.float64) - log_fs_raw).astype(np.float32)
    else:
        # Direct objective: no baseline to add back, so log_fs is the zero vector
        # load_xy already built and y is log(f_BEM) unchanged.
        log_fs_raw = zero_log_fs
        y_raw = y_direct

    mesh_ids = df_used["mesh_id"].astype(str).to_numpy()
    holdout: set[str] = set()
    if args.holdout_mesh_ids is not None:
        import pandas as pd
        hdf = pd.read_csv(args.holdout_mesh_ids)
        col = "mesh_id_10k" if "mesh_id_10k" in hdf.columns else "mesh_id"
        holdout = set(hdf[col].astype(str))
        found = sum(m in holdout for m in mesh_ids)
        if found != len(holdout):
            raise ValueError(
                f"{found} of {len(holdout)} holdout mesh_ids survive load_xy's row filter; "
                "the evaluation set must be fully present in the dataset")
        print(f"Holdout: {len(holdout)} meshes pinned out of train/val")

    split, held_pos = _split_with_holdout(
        x_raw, y_raw, log_fs_raw, mesh_ids, holdout,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
    )

    x_scaler = StandardScaler().fit(split.x_train)
    y_scaler = StandardScaler().fit(split.y_train.reshape(-1, 1))

    def loader(xx: np.ndarray, yy: np.ndarray, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            BubbleDataset(x_scaler.transform(xx).astype(np.float32),
                          y_scaler.transform(yy.reshape(-1, 1)).flatten().astype(np.float32)),
            batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)

    train_loader = loader(split.x_train, split.y_train, shuffle=True)
    val_loader = loader(split.x_val, split.y_val, shuffle=False)
    test_loader = loader(split.x_test, split.y_test, shuffle=False)

    model = BubbleFreqNet(input_dim=len(feature_cols), hidden_dim=64, hidden_dim2=32, dropout=0.1)
    model_save_path = output_dir / MODEL_FILENAME

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    writer.add_text("baseline_kind", baseline_kind, 0)
    writer.add_text("features", ", ".join(feature_cols), 0)

    print(f"Loaded samples: {len(df_used)}")
    print(f"Split -> train: {len(split.x_train)}, val: {len(split.x_val)}, "
          f"test: {len(split.x_test)}, holdout: {len(held_pos)} | device: {args.device}")
    print(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    target_desc = (f"log f_BEM - log {BASELINE_COL}" if args.baseline == "strasberg"
                   else "log f_BEM")
    print(f"Baseline kind: {baseline_kind} (target = {target_desc})")

    history, best_epoch = train(
        model=model, train_loader=train_loader, val_loader=val_loader,
        y_scaler=y_scaler, device=args.device, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience, writer=writer,
        model_save_path=model_save_path, val_log_fs=split.log_fs_val,
    )

    test_metrics = evaluate_metrics(model, test_loader, y_scaler=y_scaler,
                                    device=args.device, log_fs=split.log_fs_test)
    writer.flush()
    writer.close()

    joblib.dump(x_scaler, output_dir / "feature_scaler.joblib")
    joblib.dump(y_scaler, output_dir / "target_log_scaler.joblib")

    # ---- holdout scoring -------------------------------------------------
    holdout_metrics: dict[str, float] = {}
    if len(held_pos):
        import pandas as pd
        h_loader = loader(x_raw[held_pos], y_raw[held_pos], shuffle=False)
        pred_log, tgt_log = collect_log_predictions(
            model, h_loader, y_scaler=y_scaler, device=args.device,
            log_fs=log_fs_raw[held_pos])
        f_pred, f_gt = np.exp(pred_log), np.exp(tgt_log)
        gt_csv = df_used[  # ground truth straight from the CSV, as a cross-check
            "frequency"].to_numpy(dtype=np.float64)[held_pos]
        # The targets round-trip through float32 and the y-scaler, so agreement is
        # to ~1e-6 relative. A genuine row misalignment would be percent-level.
        if not np.allclose(f_gt, gt_csv, rtol=1e-5, atol=0):
            worst = float(np.max(np.abs(f_gt - gt_csv) / gt_csv))
            raise ValueError("reconstructed ground truth disagrees with the dataset column "
                             f"(worst relative difference {worst:.2e}); holdout rows are "
                             "probably misaligned against log_fs")
        ape = np.abs(f_pred - f_gt) / f_gt * 100.0
        holdout_metrics = {
            "count": int(len(held_pos)),
            "mape": float(np.mean(ape)),
            "max_ape": float(np.max(ape)),
            "rmse_log": float(np.sqrt(np.mean((pred_log - tgt_log) ** 2))),
        }
        pd.DataFrame({
            "mesh_id": mesh_ids[held_pos],
            "source": df_used["source"].astype(str).to_numpy()[held_pos],
            "f_gt": f_gt,
            "f_strasberg": f_base[held_pos],
            "f_pred": f_pred,
            "ape_pct": ape,
        }).to_csv(output_dir / "holdout_predictions.csv", index=False)
        print(f"\n=== Holdout ({holdout_metrics['count']} meshes) ===")
        print(f"MAPE {holdout_metrics['mape']:.4f}%   max APE {holdout_metrics['max_ape']:.4f}%")

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg.update({"baseline_kind": baseline_kind,
                "baseline_col": BASELINE_COL if args.baseline == "strasberg" else "",
                "feature_cols": feature_cols, "model_checkpoint": MODEL_FILENAME})
    (output_dir / "train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    (output_dir / "metrics.json").write_text(json.dumps({
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "holdout_metrics": holdout_metrics,
        "history_len": len(history["train_loss"]),
        "feature_cols": feature_cols,
        "model_checkpoint": MODEL_FILENAME,
        "baseline_kind": baseline_kind,
        "baseline_col": BASELINE_COL if args.baseline == "strasberg" else "",
        "n_train": int(len(split.x_train)),
        "n_val": int(len(split.x_val)),
        "n_test": int(len(split.x_test)),
        "n_holdout": int(len(held_pos)),
    }, indent=2), encoding="utf-8")

    names = df_used["mesh_id"].astype(str).tolist()
    (output_dir / "split.json").write_text(json.dumps({
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "n_total_used": int(len(df_used)),
        "mesh_id_train": [names[i] for i in split.idx_train],
        "mesh_id_val": [names[i] for i in split.idx_val],
        "mesh_id_test": [names[i] for i in split.idx_test],
        "mesh_id_holdout": [names[i] for i in held_pos],
    }, indent=2), encoding="utf-8")

    print("\n=== Test Metrics ===")
    print(f"RMSE(log f): {test_metrics['rmse_log']:.5f}")
    print(f"MAPE:        {test_metrics['mape']:.4f}%")
    print(f"Max APE:     {test_metrics['max_ape']:.4f}%")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()

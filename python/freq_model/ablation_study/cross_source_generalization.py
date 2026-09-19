"""Cross-solver generalization of the 8-feature bubble frequency model.

The shipped NN8 model trains on a *random mixed* split of
``dataset_bubblegym_10k.csv``, so its per-source LBM test MAPE does not measure
transfer -- LBM bubbles were in its training set. This trains on one solver and
tests on the other (``--train-source VOF --test-source LBM`` by default, and the
reverse). Everything except the split matches the mixed run; ``load_xy`` and
``FEATURE_COLS`` are imported from ``fit_shape_freq_model`` verbatim so the
feature pipeline cannot drift.

Two test sets are reported: **primary**, the test-source rows of the reference
mixed run's test partition (matched by ``mesh_id`` via its ``split.json``, so the
number is directly comparable to the mixed per-source metric), and
**secondary**, every row of the test source. Scalers are fit on train-source
training rows only, and two assertions enforce that no test-source ``mesh_id``
reaches training.

    python python/freq_model/ablation_study/cross_source_generalization.py \\
        [--train-source LBM --test-source VOF]
"""

from __future__ import annotations

# Import torch first: on Windows this avoids a joblib/torch OpenMP DLL clash
# (same defensive ordering as stratified_eval_model.py).
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_THIS = Path(__file__).resolve()
_FREQ_MODEL_DIR = _THIS.parents[1]
_PYTHON_ROOT = _FREQ_MODEL_DIR.parent
_REPO_ROOT = _PYTHON_ROOT.parent
for _p in (_PYTHON_ROOT, _FREQ_MODEL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Must follow the sys.path setup above: running this file as a script puts only
# its own directory on sys.path, so ``freq_model`` is not importable until
# _PYTHON_ROOT is inserted.
from freq_model.NN.bub_freq_net import BubbleFreqNet  # noqa: E402
from freq_model.NN.bubble_dataset import BubbleDataset, set_seed  # noqa: E402
from freq_model.NN.fit_shape_freq_model import (  # noqa: E402
    FEATURE_COLS,
    THUMBNAIL_DIR_BY_SOURCE,
    load_xy,
)
from freq_model.NN.training import (  # noqa: E402
    collect_log_predictions,
    evaluate_metrics,
    train,
)
from utils.worst_case_utils import export_worst_cases  # noqa: E402


BASELINE_KIND = "none_log_direct"

# The mixed-training run whose test partition defines the primary test rows.
REFERENCE_RUN_DIR = (
    _FREQ_MODEL_DIR
    / "output"
    / "output_8feature_direct_bubblegym_10k"
)
REFERENCE_SPLIT_JSON = REFERENCE_RUN_DIR / "split.json"

# Short names used in output paths, matching the write-up's directory names.
SOURCE_SLUG = {"VOF": "tim", "LBM": "lbm"}


def _slug(source: str) -> str:
    return SOURCE_SLUG.get(source, source.lower())


def run_name_for(train_source: str, test_source: str) -> str:
    return f"8feat_direct_train_{_slug(train_source)}_test_{_slug(test_source)}"


# ---------------------------------------------------------------------------
# Source-based split (the only new logic vs. the mixed trainer)
# ---------------------------------------------------------------------------


def build_source_split(
    df_used: pd.DataFrame,
    *,
    train_source: str,
    test_source: str,
    reference_split: dict,
    val_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Partition ``df_used`` by solver instead of at random.

    Train and val come from ``train_source`` only. The primary test set is the
    ``test_source`` rows of the reference mixed run's test partition, matched by
    ``mesh_id``; the secondary set is every ``test_source`` row.

    Returns index arrays into ``df_used``: ``idx_train``, ``idx_val``,
    ``idx_test`` (primary) and ``idx_test_all`` (secondary).
    """
    sources = df_used["source"].astype(str).to_numpy()
    mesh_ids = df_used["mesh_id"].astype(str).to_numpy()

    for flag, src in (("train", train_source), ("test", test_source)):
        if not np.any(sources == src):
            available = sorted(set(sources.tolist()))
            raise ValueError(
                f"--{flag}-source {src!r} matches no rows; dataset has {available}"
            )

    # Test-source mesh_ids of the reference (mixed-run) test partition.
    ref_test_ids = {
        mid
        for mid, src in zip(
            reference_split["mesh_id_test"], reference_split["source_test"]
        )
        if src == test_source
    }
    if not ref_test_ids:
        raise ValueError(
            f"the reference split has no test rows with source == {test_source!r}"
        )

    train_src_idx = np.where(sources == train_source)[0]
    test_all_idx = np.where(sources == test_source)[0]
    idx_test = np.array(
        [i for i in test_all_idx if mesh_ids[i] in ref_test_ids], dtype=int
    )
    if idx_test.size == 0:
        raise ValueError(
            "no test-source row survives the reference-split match; the "
            "reference split.json and this dataset disagree on mesh_id"
        )

    rng = np.random.default_rng(seed)
    train_perm = rng.permutation(train_src_idx)
    n_val = max(1, int(len(train_perm) * val_ratio))
    idx_val = np.sort(train_perm[:n_val])
    idx_train = np.sort(train_perm[n_val:])
    if idx_train.size == 0:
        raise ValueError(f"val_ratio={val_ratio} leaves no training rows")

    # No-leakage guarantees: the model must never see the test solver.
    train_ids = set(mesh_ids[idx_train]) | set(mesh_ids[idx_val])
    overlap = train_ids & set(mesh_ids[idx_test])
    if overlap:
        raise AssertionError(
            f"train/test mesh_id overlap ({len(overlap)} ids), "
            f"e.g. {sorted(overlap)[:3]}"
        )
    leaked = train_ids & set(mesh_ids[test_all_idx])
    if leaked:
        raise AssertionError(
            f"test-source rows leaked into training ({len(leaked)} ids), "
            f"e.g. {sorted(leaked)[:3]}"
        )

    return {
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "idx_test_all": test_all_idx,
    }


def _loader(x, y, batch_size: int, num_workers: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        BubbleDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the 8-feature BubbleFreqNet on bubbles from ONE solver and "
            "test on the other, to measure cross-solver generalization."
        )
    )
    parser.add_argument("--train-source", type=str, default="VOF",
                        help="Solver whose bubbles are trained on (default: VOF / Langlois2016).")
    parser.add_argument("--test-source", type=str, default="LBM",
                        help="Solver held out entirely (default: LBM).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_REPO_ROOT / "dataset" / "bubble_gym" / "dataset_bubblegym_10k.csv",
    )
    parser.add_argument(
        "--reference-split",
        type=Path,
        default=REFERENCE_SPLIT_JSON,
        help="split.json of the mixed run that defines the primary test rows.",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to python/freq_model/output/output_<run name>.")
    # Hyperparameters below are identical to the mixed-training run.
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.train_source == args.test_source:
        raise SystemExit("--train-source and --test-source must differ")

    set_seed(args.seed)
    run_name = run_name_for(args.train_source, args.test_source)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _FREQ_MODEL_DIR / "output" / f"output_{run_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.dataset.resolve()
    ref_path = args.reference_split.resolve()
    if not ref_path.is_file():
        raise SystemExit(
            f"reference split not found: {ref_path}\n"
            "It is produced by the mixed-training run (fit_shape_freq_model.py) "
            "and defines the primary test rows."
        )
    with ref_path.open("r", encoding="utf-8") as fh:
        reference_split = json.load(fh)

    # Feature pipeline imported verbatim from the mixed trainer.
    x_raw, y_raw, log_fs_raw, df_used = load_xy(dataset_path)
    for col in ("source", "mesh_id"):
        if col not in df_used.columns:
            raise SystemExit(
                f"dataset is missing the {col!r} column, which the source split needs"
            )

    split = build_source_split(
        df_used,
        train_source=args.train_source,
        test_source=args.test_source,
        reference_split=reference_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    idx_train = split["idx_train"]
    idx_val = split["idx_val"]
    idx_test = split["idx_test"]
    idx_test_all = split["idx_test_all"]

    # Scalers see the training solver only -- no statistics leak from the test one.
    x_scaler = StandardScaler().fit(x_raw[idx_train])
    y_scaler = StandardScaler().fit(y_raw[idx_train].reshape(-1, 1))

    def _x(idx):
        return x_scaler.transform(x_raw[idx]).astype(np.float32)

    def _y(idx):
        return y_scaler.transform(y_raw[idx].reshape(-1, 1)).flatten().astype(np.float32)

    train_loader = _loader(_x(idx_train), _y(idx_train), args.batch_size,
                           args.num_workers, shuffle=True)
    # Eval loaders must not shuffle: log_fs is matched positionally.
    val_loader = _loader(_x(idx_val), _y(idx_val), args.batch_size,
                         args.num_workers, shuffle=False)
    test_loader = _loader(_x(idx_test), _y(idx_test), args.batch_size,
                          args.num_workers, shuffle=False)
    test_all_loader = _loader(_x(idx_test_all), _y(idx_test_all), args.batch_size,
                              args.num_workers, shuffle=False)

    model = BubbleFreqNet(
        input_dim=len(FEATURE_COLS), hidden_dim=64, hidden_dim2=32, dropout=0.1
    )
    model_save_path = output_dir / f"bubble_freq_net_{run_name}_best.pt"

    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("dataset/path", str(dataset_path), 0)
    writer.add_text("features", ", ".join(FEATURE_COLS), 0)
    writer.add_text("baseline_kind", BASELINE_KIND, 0)
    writer.add_text("split/train_source", args.train_source, 0)
    writer.add_text("split/test_source", args.test_source, 0)

    print(f"Loaded samples: {len(df_used)}")
    print(f"Direction: train {args.train_source} -> test {args.test_source}")
    print(
        f"Split -> train: {len(idx_train)}, val: {len(idx_val)}, "
        f"test({args.test_source} ref-split): {len(idx_test)}, "
        f"test({args.test_source} all): {len(idx_test_all)} | device: {args.device}"
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
        val_log_fs=log_fs_raw[idx_val],
    )

    test_metrics = evaluate_metrics(
        model, test_loader, y_scaler=y_scaler, device=args.device,
        log_fs=log_fs_raw[idx_test],
    )
    test_all_metrics = evaluate_metrics(
        model, test_all_loader, y_scaler=y_scaler, device=args.device,
        log_fs=log_fs_raw[idx_test_all],
    )

    writer.add_hparams(
        {
            "train_source": args.train_source,
            "test_source": args.test_source,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "val_ratio": args.val_ratio,
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

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg.update(
        {
            "baseline_kind": BASELINE_KIND,
            "baseline_col": "",
            "feature_cols": FEATURE_COLS,
            "model_checkpoint": model_save_path.name,
            "reference_split": str(ref_path),
        }
    )
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str)

    mesh_ids = df_used["mesh_id"].astype(str).to_numpy()
    split_out = {
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "train_source": args.train_source,
        "test_source": args.test_source,
        "n_total_used": int(len(df_used)),
        "reference_split": str(ref_path),
        "idx_train": idx_train.astype(int).tolist(),
        "idx_val": idx_val.astype(int).tolist(),
        "idx_test": idx_test.astype(int).tolist(),
        "idx_test_all": idx_test_all.astype(int).tolist(),
        "mesh_id_train": mesh_ids[idx_train].tolist(),
        "mesh_id_val": mesh_ids[idx_val].tolist(),
        "mesh_id_test": mesh_ids[idx_test].tolist(),
        "mesh_id_test_all": mesh_ids[idx_test_all].tolist(),
    }
    with (output_dir / "split.json").open("w", encoding="utf-8") as fh:
        json.dump(split_out, fh, indent=2)

    pred_log, tgt_log = collect_log_predictions(
        model, test_loader, y_scaler=y_scaler, device=args.device,
        log_fs=log_fs_raw[idx_test],
    )
    worst_dir = _REPO_ROOT / "results" / f"worst_cases_{run_name}"
    worst_rows = export_worst_cases(
        df_used,
        idx_test,
        np.exp(pred_log),
        np.exp(tgt_log),
        worst_dir,
        top_k=10,
        thumbnail_dir=THUMBNAIL_DIR_BY_SOURCE.get(args.test_source),
        thumbnail_dir_by_source=THUMBNAIL_DIR_BY_SOURCE,
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "best_epoch": best_epoch,
                "train_source": args.train_source,
                "test_source": args.test_source,
                "n_train": int(len(idx_train)),
                "n_val": int(len(idx_val)),
                "n_test_ref_split": int(len(idx_test)),
                "n_test_all": int(len(idx_test_all)),
                "test_metrics": test_metrics,
                "test_metrics_all_test_source": test_all_metrics,
                "history_len": len(history["train_loss"]),
                "feature_cols": FEATURE_COLS,
                "model_checkpoint": model_save_path.name,
                "baseline_kind": BASELINE_KIND,
                "reference_split": str(ref_path),
                "worst_cases_dir": str(worst_dir.as_posix()),
                "worst_cases_count": int(len(worst_rows)),
            },
            fh,
            indent=2,
        )

    print(f"\n=== {args.train_source} -> {args.test_source} ===")
    for label, n, m in (
        (f"{args.test_source} ref-test", len(idx_test), test_metrics),
        (f"{args.test_source} all", len(idx_test_all), test_all_metrics),
    ):
        print(
            f"{label:>22s}  n={n:>5d}  MAPE={m['mape']:.3f}%  "
            f"RMSE(log)={m['rmse_log']:.4f}  MaxAPE={m['max_ape']:.2f}%"
        )
    print("\nSaved artifacts:")
    print(f"- model:      {model_save_path}")
    print(f"- scalers:    {output_dir / 'feature_scaler.joblib'}")
    print(f"              {output_dir / 'target_log_scaler.joblib'}")
    print(f"- tensorboard:{tb_dir}")
    print(f"- metrics:    {output_dir / 'metrics.json'}")
    print(f"- worst cases:{worst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

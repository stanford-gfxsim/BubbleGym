"""Shared helpers for feature-subset and width ablations (8feat direct contract)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_THIS = Path(__file__).resolve().parent
_FREQ_MODEL = _THIS.parent
_PYTHON_ROOT = _FREQ_MODEL.parent
for _p in (_PYTHON_ROOT, _FREQ_MODEL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from freq_model.NN.bub_freq_net import BubbleFreqNet  # noqa: E402
from freq_model.NN.bubble_dataset import BubbleDataset, set_seed, split_dataset  # noqa: E402
from shape_feature.nonspherical_features import (  # noqa: E402
    non_sph_va_from_area_volume as _non_sph_va,
)
from freq_model.NN.training import collect_log_predictions, evaluate_metrics, train  # noqa: E402
from freq_model.NN.fit_shape_freq_model import (  # noqa: E402
    FEATURE_COLS as ALL_FEATURE_COLS,
    load_xy,
)
from freq_model.NN.stratified_eval_model import assign_bins, compute_metrics  # noqa: E402

def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for ancestor in [p, *p.parents]:
        if (ancestor / "dataset").is_dir() and (ancestor / "python" / "freq_model").is_dir():
            return ancestor
    return p.parents[3]


_REPO_ROOT = _find_repo_root(_THIS)

BASELINE_KIND = "none_log_direct"
MODEL_CHECKPOINT_NAME = "bubble_freq_net_best.pt"
TARGET_COL = "frequency"


def feature_indices(feature_keys: list[str]) -> list[int]:
    idxs: list[int] = []
    for k in feature_keys:
        if k not in ALL_FEATURE_COLS:
            raise ValueError(f"Unknown feature key {k!r}; allowed: {ALL_FEATURE_COLS}")
        idxs.append(ALL_FEATURE_COLS.index(k))
    return idxs


def load_xy_with_keys(
    dataset_path: Path,
    feature_keys: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Same cleaning/split population as 8feat trainer; X has only ``feature_keys`` columns (in that order)."""
    x8, y, log_fs, df = load_xy(dataset_path)
    idxs = feature_indices(feature_keys)
    x = x8[:, idxs].astype(np.float32)
    return x, y, log_fs, df


def features_matrix_from_curated_df(df: pd.DataFrame) -> np.ndarray:
    """Build the full 8-column matrix in ``ALL_FEATURE_COLS`` order (same formulas as ``load_xy``)."""
    _require_cols(
        df,
        [
            "surface_area",
            "volume",
            "Phi_VM",
            "W_vertex",
            "eta_V",
            "eta_A",
            "eta_M",
            "i00",
            "i11",
            "i22",
        ],
        "curated dataframe",
    )
    area = pd.to_numeric(df["surface_area"], errors="coerce").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=np.float64)
    phi_vm = pd.to_numeric(df["Phi_VM"], errors="coerce").to_numpy(dtype=np.float64)
    w_vertex = pd.to_numeric(df["W_vertex"], errors="coerce").to_numpy(dtype=np.float64)
    eta_V = pd.to_numeric(df["eta_V"], errors="coerce").to_numpy(dtype=np.float64)
    eta_A = pd.to_numeric(df["eta_A"], errors="coerce").to_numpy(dtype=np.float64)
    eta_M = pd.to_numeric(df["eta_M"], errors="coerce").to_numpy(dtype=np.float64)
    i00 = pd.to_numeric(df["i00"], errors="coerce").to_numpy(dtype=np.float64)
    i11 = pd.to_numeric(df["i11"], errors="coerce").to_numpy(dtype=np.float64)
    i22 = pd.to_numeric(df["i22"], errors="coerce").to_numpy(dtype=np.float64)

    non_sph_va = _non_sph_va(area, volume)
    non_sph_vm = 1.0 - phi_vm
    non_sph_w = 1.0 - (4.0 * np.pi / w_vertex)
    i11_over_i00 = i11 / i00
    i22_over_i00 = i22 / i00

    return np.column_stack(
        [
            i11_over_i00,
            i22_over_i00,
            non_sph_va,
            non_sph_vm,
            non_sph_w,
            eta_V,
            eta_A,
            eta_M,
        ]
    ).astype(np.float32)


def _require_cols(df: pd.DataFrame, cols: list[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx}: missing columns {missing}")


def _sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmark_inference_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: str,
    *,
    warmup: int = 30,
    repeats: int = 150,
) -> dict[str, float | int | str]:
    """Time ``model(x)`` forward only (no scaler). ``x`` must already be on ``device``.

    Uses CUDA events on GPU and ``perf_counter`` on CPU/MPS. Reports wall time for one
    full forward on the given batch (all curated rows in one batch by default).
    """
    model.eval()
    n = int(x.shape[0])
    if n <= 0:
        raise ValueError("benchmark_inference_forward: empty batch")

    with torch.no_grad():
        if device.startswith("cuda") and torch.cuda.is_available():
            _sync_device(device)
            for _ in range(warmup):
                model(x)
            _sync_device(device)
            times_ms: list[float] = []
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(repeats):
                starter.record()
                model(x)
                ender.record()
                _sync_device(device)
                times_ms.append(float(starter.elapsed_time(ender)))
        else:
            for _ in range(warmup):
                model(x)
            _sync_device(device)
            times_ms = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                model(x)
                _sync_device(device)
                times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    return {
        "device": device,
        "batch_size": n,
        "warmup": int(warmup),
        "repeats": int(repeats),
        "forward_mean_ms": mean_ms,
        "forward_std_ms": float(np.std(arr)),
        "forward_median_ms": float(np.median(arr)),
        "per_sample_mean_us": float(mean_ms * 1000.0 / n),
        "throughput_samples_per_s": float(n / (mean_ms / 1000.0)) if mean_ms > 0 else float("nan"),
    }


def merge_inference_timing_into_metrics(variant_dir: Path, timing: dict[str, float | int | str]) -> None:
    """Attach ``inference_timing`` to ``metrics.json`` if present."""
    mp = variant_dir / "metrics.json"
    if not mp.is_file():
        return
    with mp.open(encoding="utf-8") as f:
        m = json.load(f)
    m["inference_timing"] = timing
    with mp.open("w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train_one(
    *,
    variant_name: str,
    feature_keys: list[str],
    hidden_dim: int,
    hidden_dim2: int,
    dropout: float,
    output_dir: Path,
    dataset_path: Path,
    seed: int = 42,
    epochs: int = 400,
    batch_size: int = 64,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    patience: int = 40,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    device: str = "cpu",
) -> None:
    """Train one model; write checkpoints, scalers, split.json, metrics.json."""
    set_seed(seed)
    dataset_path = dataset_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    x_raw, y_raw, log_fs_raw, df_used = load_xy_with_keys(dataset_path, feature_keys)

    split = split_dataset(
        x_raw,
        y_raw,
        log_fs_raw,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        BubbleDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        BubbleDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    input_dim = len(feature_keys)
    model = BubbleFreqNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hidden_dim2=hidden_dim2,
        dropout=dropout,
    )
    model_save_path = output_dir / MODEL_CHECKPOINT_NAME
    n_params = count_params(model)

    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("dataset/path", str(dataset_path), 0)
    writer.add_text("features", ", ".join(feature_keys), 0)
    writer.add_text("baseline_kind", BASELINE_KIND, 0)
    writer.add_text("variant", variant_name, 0)
    writer.add_text("hidden", f"{hidden_dim},{hidden_dim2}", 0)

    history, best_epoch = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_scaler=y_scaler,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        writer=writer,
        model_save_path=model_save_path,
        val_log_fs=split.log_fs_val,
    )

    test_metrics = evaluate_metrics(
        model,
        test_loader,
        y_scaler=y_scaler,
        device=device,
        log_fs=split.log_fs_test,
    )

    writer.add_hparams(
        {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "hidden_dim": hidden_dim,
            "hidden_dim2": hidden_dim2,
            "input_dim": input_dim,
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

    cfg = {
        "variant_name": variant_name,
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "num_workers": num_workers,
        "device": device,
        "baseline_kind": BASELINE_KIND,
        "feature_cols": feature_keys,
        "hidden_dim": hidden_dim,
        "hidden_dim2": hidden_dim2,
        "dropout": dropout,
        "model_checkpoint": MODEL_CHECKPOINT_NAME,
        "n_params": n_params,
    }
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)

    split_out = {
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
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
        device=device,
        log_fs=split.log_fs_test,
    )
    pred_freq = np.exp(pred_log)
    tgt_freq = np.exp(tgt_log)

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
                "rmse_log": float(np.sqrt(np.mean(log_diff**2))),
                "mape": float(np.mean(ape)),
                "max_ape": float(np.max(ape)) if len(ape) else float("nan"),
            }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "variant_name": variant_name,
                "best_epoch": best_epoch,
                "test_metrics": test_metrics,
                "test_metrics_per_source": test_per_source,
                "history_len": len(history["train_loss"]),
                "feature_cols": feature_keys,
                "hidden_dim": hidden_dim,
                "hidden_dim2": hidden_dim2,
                "n_params": n_params,
                "model_checkpoint": MODEL_CHECKPOINT_NAME,
                "baseline_kind": BASELINE_KIND,
            },
            f,
            indent=2,
        )

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[{variant_name}] Saved to {output_dir} | test MAPE={test_metrics['mape']:.4f}%")


def evaluate_on_curated(
    *,
    variant_dir: Path,
    feature_keys: list[str],
    hidden_dim: int,
    hidden_dim2: int,
    dropout: float = 0.1,
    curated_csv: Path | None = None,
    summary_json: Path | None = None,
    device: str = "cpu",
    model_checkpoint: str | None = None,
) -> dict:
    """Evaluate one checkpoint on the fixed 10x10 curated CSV; write ``curated_eval.json``."""
    variant_dir = variant_dir.resolve()
    if curated_csv is None:
        curated_csv = _REPO_ROOT / "results" / "stratified_eval_chull_direct" / "selected_rows.csv"
    if summary_json is None:
        summary_json = _REPO_ROOT / "results" / "stratified_eval_chull_direct" / "summary.json"
    curated_csv = curated_csv.resolve()
    summary_json = summary_json.resolve()
    ckpt_name = model_checkpoint or MODEL_CHECKPOINT_NAME
    model_path = variant_dir / ckpt_name
    feat_path = variant_dir / "feature_scaler.joblib"
    tgt_path = variant_dir / "target_log_scaler.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not feat_path.is_file() or not tgt_path.is_file():
        raise FileNotFoundError(f"Missing scaler in {variant_dir}")

    df = pd.read_csv(curated_csv)
    df.columns = df.columns.str.lstrip("#").str.strip()
    _require_cols(df, [TARGET_COL, "non_sphericity"], "curated CSV")

    x8 = features_matrix_from_curated_df(df)
    idxs = feature_indices(feature_keys)
    x = x8[:, idxs]

    x_scaler = joblib.load(feat_path)
    y_scaler = joblib.load(tgt_path)
    x_norm = x_scaler.transform(x.astype(np.float64)).astype(np.float32)

    model = BubbleFreqNet(
        input_dim=len(feature_keys),
        hidden_dim=int(hidden_dim),
        hidden_dim2=int(hidden_dim2),
        dropout=dropout,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    x_t = torch.from_numpy(x_norm).to(device)
    inference_timing = benchmark_inference_forward(model, x_t, device)

    with torch.no_grad():
        y_norm = model(x_t).cpu().numpy().reshape(-1, 1)
    pred_log = y_scaler.inverse_transform(y_norm).flatten()
    pred_hz = np.exp(pred_log.astype(np.float64))

    tgt = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)

    with summary_json.open(encoding="utf-8") as f:
        summ = json.load(f)
    edges = np.array(summ["bin_edges_selected"], dtype=np.float64)
    non_sph = pd.to_numeric(df["non_sphericity"], errors="coerce").to_numpy(dtype=np.float64)
    bin_id = assign_bins(non_sph, edges)
    n_bins = int(summ.get("sampling", {}).get("bins", len(edges) - 1))

    per_bin: list[dict] = []
    for b in range(n_bins):
        m = bin_id == b
        per_bin.append(compute_metrics(pred_hz[m], tgt[m]))

    overall = compute_metrics(pred_hz, tgt)

    out = {
        "curated_csv": str(curated_csv.as_posix()),
        "summary_json": str(summary_json.as_posix()),
        "feature_cols": feature_keys,
        "hidden_dim": hidden_dim,
        "hidden_dim2": hidden_dim2,
        "n_bins": n_bins,
        "metrics_overall": overall,
        "metrics_per_bin": per_bin,
        "bin_edges_selected": edges.tolist(),
        "inference_timing": inference_timing,
    }
    out_path = variant_dir / "curated_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    merge_inference_timing_into_metrics(variant_dir, inference_timing)
    print(
        f"  curated eval -> {out_path} | overall MAPE={overall['mape']:.4f}% | "
        f"infer mean={inference_timing['forward_mean_ms']:.4f} ms/batch "
        f"(n={inference_timing['batch_size']})"
    )
    return out


def ensure_curated_eval(
    variant_dir: Path,
    feature_keys: list[str],
    hidden_dim: int,
    hidden_dim2: int,
    *,
    device: str,
    force: bool = False,
    model_checkpoint: str | None = None,
) -> None:
    path = variant_dir / "curated_eval.json"
    if path.is_file() and not force:
        return
    evaluate_on_curated(
        variant_dir=variant_dir,
        feature_keys=feature_keys,
        hidden_dim=hidden_dim,
        hidden_dim2=hidden_dim2,
        device=device,
        model_checkpoint=model_checkpoint,
    )

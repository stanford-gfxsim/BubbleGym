"""Training loop, optimizer factory, and metric helpers shared by the
BubbleFreq trainers. Model- and feature-agnostic: callers own the feature
engineering, the CLI and the artifact export.

Target convention: ``y`` is a log-space quantity plus a per-row log-baseline
``log_fs``, and eval reconstructs ``pred_log = pred_log_resid + log_fs``.
The shipped direct model passes ``log_fs = 0`` so the add is a no-op; the
residual convention passes ``log f_strasberg`` (see ``BASELINE_COL``).

Trap: ``log_fs`` is indexed positionally against the eval loader's output, so
eval loaders MUST be built with ``shuffle=False`` or the baseline is added to
the wrong rows. The length check in ``_reconstruct_log_predictions`` catches a
size mismatch but cannot catch a permutation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


__all__ = [
    "BASELINE_COL",
    "BASELINE_KIND",
    "make_optimizer",
    "evaluate_metrics",
    "collect_log_predictions",
    "train",
]


# Default Strasberg-residual convention. Trainers that use a different
# baseline (e.g. Minnaert) override these in their CLI module.
BASELINE_COL = "f_strasberg"
BASELINE_KIND = "strasberg_log_residual"


def make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """AdamW with weight-decay applied to weights only (not biases / norms).

    Known, deliberately preserved discrepancy: the split is by parameter NAME,
    so it only catches the standalone ``block1_norm``. The two LayerNorms nested
    inside ``nn.Sequential`` are auto-named ``input_proj.1.weight`` and
    ``block2.1.weight`` -- no "norm" substring -- so their 96 gain parameters DO
    receive weight_decay (1e-4), contrary to the paper's description. Fixing the
    filter would change the optimizer state and invalidate every shipped
    checkpoint and printed number, so the behavior stays as trained. Anything
    retrained from scratch should match this filter, not the paper text.
    """
    decay_params = [p for n, p in model.named_parameters() if "bias" not in n and "norm" not in n]
    no_decay_params = [p for n, p in model.named_parameters() if "bias" in n or "norm" in n]
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def _reconstruct_log_predictions(
    model: nn.Module,
    loader: DataLoader,
    y_scaler: StandardScaler,
    device: str,
    log_fs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over ``loader`` and undo the y-scaler + add ``log_fs``.

    Returns ``(pred_log, tgt_log)`` in natural log space. The eval loader must
    be created with ``shuffle=False`` so the row order matches ``log_fs``.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    pred_norm = np.concatenate(preds).flatten()
    tgt_norm = np.concatenate(targets).flatten()
    pred_log_resid = y_scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
    tgt_log_resid = y_scaler.inverse_transform(tgt_norm.reshape(-1, 1)).flatten()
    if log_fs.shape[0] != pred_log_resid.shape[0]:
        raise ValueError(
            f"log_fs length {log_fs.shape[0]} does not match prediction length "
            f"{pred_log_resid.shape[0]}; did you forget to disable shuffling on the eval loader?"
        )
    pred_log = pred_log_resid + log_fs.astype(np.float64)
    tgt_log = tgt_log_resid + log_fs.astype(np.float64)
    return pred_log, tgt_log


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    y_scaler: StandardScaler,
    device: str,
    log_fs: np.ndarray,
) -> dict[str, float]:
    """Return ``{rmse_log, mape, max_ape}`` for the eval loader."""
    pred_log, tgt_log = _reconstruct_log_predictions(
        model, loader, y_scaler=y_scaler, device=device, log_fs=log_fs
    )
    rmse_log = float(np.sqrt(np.mean((pred_log - tgt_log) ** 2)))
    f_pred = np.exp(pred_log)
    f_tgt = np.exp(tgt_log)
    ape = np.abs(f_pred - f_tgt) / (f_tgt + 1e-12) * 100.0
    return {
        "rmse_log": rmse_log,
        "mape": float(np.mean(ape)),
        "max_ape": float(np.max(ape)),
    }


def collect_log_predictions(
    model: nn.Module,
    loader: DataLoader,
    y_scaler: StandardScaler,
    device: str,
    log_fs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Like ``evaluate_metrics`` but returns the raw ``(pred_log, tgt_log)``
    arrays for downstream worst-case export / scatter plots.
    """
    return _reconstruct_log_predictions(
        model, loader, y_scaler=y_scaler, device=device, log_fs=log_fs
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_scaler: StandardScaler,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    writer: SummaryWriter,
    model_save_path: Path,
    val_log_fs: np.ndarray,
) -> tuple[dict[str, list[float]], int]:
    """Huber-loss training loop with cosine LR + early stopping on val loss.

    Saves the best-by-val-loss checkpoint to ``model_save_path`` and returns
    ``(history, best_epoch)``. Uses ``evaluate_metrics`` every 10 epochs to
    log RMSE(log f) / MAPE on the validation split.
    """
    model.to(device)
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100.0
    )

    best_epoch = 0
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if epoch == 1 or epoch % 10 == 0:
            val_metrics = evaluate_metrics(
                model, val_loader, y_scaler=y_scaler, device=device, log_fs=val_log_fs
            )
            writer.add_scalar("val/rmse_log", val_metrics["rmse_log"], epoch)
            writer.add_scalar("val/mape", val_metrics["mape"], epoch)
            print(
                f"Epoch {epoch:4d} | train {train_loss:.5f} | val {val_loss:.5f} | "
                f"RMSE(log) {val_metrics['rmse_log']:.5f} | MAPE {val_metrics['mape']:.2f}%"
            )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_save_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, "
                f"best val loss: {best_val:.6f})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch

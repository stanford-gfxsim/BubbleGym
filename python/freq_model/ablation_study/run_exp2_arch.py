"""Experiment 2: network width (8 features fixed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_PYTHON_DIR = _THIS.parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import torch

from freq_model.ablation_study.ablation_common import (
    ALL_FEATURE_COLS,
    ensure_curated_eval,
    train_one,
)

RESULTS = _THIS.parents[2] / "results" / "experiments" / "supp_table02_width_ablation"

# 2:1 ratio per plan
ARCH_VARIANTS: dict[str, tuple[int, int]] = {
    "h16_h8": (16, 8),
    "h32_h16": (32, 16),
    "h64_h32": (64, 32),
    "h128_h64": (128, 64),
    "h256_h128": (256, 128),
}

DROPOUT = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/bubble_gym/dataset_bubblegym_10k.csv"),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Comma-separated subset of: " + ",".join(ARCH_VARIANTS.keys()),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()

    feature_keys = list(ALL_FEATURE_COLS)
    want = {v.strip() for v in args.variants.split(",") if v.strip()} if args.variants.strip() else None
    to_run = [k for k in ARCH_VARIANTS if want is None or k in want]
    if not to_run:
        raise SystemExit("No variants selected.")

    for name in to_run:
        h1, h2 = ARCH_VARIANTS[name]
        out = RESULTS / name
        metrics_path = out / "metrics.json"
        if metrics_path.is_file() and not args.force:
            print(f"[skip] {name}: exists {metrics_path}")
        else:
            train_one(
                variant_name=name,
                feature_keys=feature_keys,
                hidden_dim=h1,
                hidden_dim2=h2,
                dropout=DROPOUT,
                output_dir=out,
                dataset_path=args.dataset,
                epochs=args.epochs,
                device=args.device,
            )
        ensure_curated_eval(
            out,
            feature_keys,
            h1,
            h2,
            device=args.device,
            force=args.force,
        )

    print("Done exp2.")


if __name__ == "__main__":
    main()

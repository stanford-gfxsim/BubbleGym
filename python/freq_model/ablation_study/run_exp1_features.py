"""Experiment 1: input feature subsets (hidden 64/32 fixed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_PYTHON_DIR = _THIS.parents[1]  # .../python (repo's python package root)
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import torch

from freq_model.ablation_study.ablation_common import (
    ALL_FEATURE_COLS,
    ensure_curated_eval,
    train_one,
)

RESULTS = _THIS.parents[2] / "results" / "experiments" / "supp_table01_feature_ablation"

VARIANTS: dict[str, list[str]] = {
    "inertia": ["i11_over_i00", "i22_over_i00"],
    "nonsph": ["non_sph_va", "non_sph_vm", "non_sph_w"],
    "inertia_nonsph": [
        "i11_over_i00",
        "i22_over_i00",
        "non_sph_va",
        "non_sph_vm",
        "non_sph_w",
    ],
    "chull": ["eta_V", "eta_A", "eta_M"],
    "inertia_chull": [
        "i11_over_i00",
        "i22_over_i00",
        "eta_V",
        "eta_A",
        "eta_M",
    ],
}

HIDDEN = 64
HIDDEN2 = 32
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
        help="Comma-separated subset of: " + ",".join(VARIANTS.keys()),
    )
    parser.add_argument("--force", action="store_true", help="Re-train even if metrics.json exists.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()

    want = {v.strip() for v in args.variants.split(",") if v.strip()} if args.variants.strip() else None
    to_run = [k for k in VARIANTS if want is None or k in want]
    if not to_run:
        raise SystemExit("No variants selected.")

    for name in to_run:
        keys = VARIANTS[name]
        out = RESULTS / name
        metrics_path = out / "metrics.json"
        if metrics_path.is_file() and not args.force:
            print(f"[skip] {name}: exists {metrics_path}")
        else:
            train_one(
                variant_name=name,
                feature_keys=keys,
                hidden_dim=HIDDEN,
                hidden_dim2=HIDDEN2,
                dropout=DROPOUT,
                output_dir=out,
                dataset_path=args.dataset,
                epochs=args.epochs,
                device=args.device,
            )
        ensure_curated_eval(
            out,
            keys,
            HIDDEN,
            HIDDEN2,
            device=args.device,
            force=args.force,
        )

    print("Done exp1. All feature order reference:", ALL_FEATURE_COLS)


if __name__ == "__main__":
    main()

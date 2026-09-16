"""Build summary_exp{1,2}.csv and comparison PNGs from per-variant outputs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_PYTHON_DIR = _THIS.parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch

from freq_model.ablation_study.ablation_common import evaluate_on_curated

# results/ is grouped by the figure or table each artifact backs, so the two
# sweeps live in the folder named for the supplement table they produce.
EXPERIMENTS = _THIS.parents[2] / "results" / "experiments"
EXP_DIRS = {
    "exp1_features": EXPERIMENTS / "supp_table01_feature_ablation",
    "exp2_arch": EXPERIMENTS / "supp_table02_width_ablation",
}


def _read_json(p: Path) -> dict:
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _variant_dirs(exp_sub: str) -> list[Path]:
    """Variant directories, ordered by model size rather than by name.

    Plain alphabetical order puts the width sweep in the nonsense sequence
    h128, h16, h256, h32, h64, which makes the bar charts read as noise. Sorting
    on the parameter count from metrics.json puts the sweep in increasing-
    capacity order; the name breaks ties for the feature sweep, whose variants
    all have near-identical parameter counts.
    """
    root = EXP_DIRS[exp_sub]
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / "metrics.json").is_file()]

    def key(p: Path):
        try:
            return (int(_read_json(p / "metrics.json").get("n_params", 0)), p.name)
        except Exception:  # noqa: BLE001
            return (0, p.name)

    return sorted(dirs, key=key)


def _row_from_variant(vdir: Path) -> dict[str, object]:
    m = _read_json(vdir / "metrics.json")
    tm = m.get("test_metrics", {})
    per_src = m.get("test_metrics_per_source", {})
    tim = per_src.get("VOF", {})
    lbm = per_src.get("LBM", {})

    row: dict[str, object] = {
        "variant": vdir.name,
        "n_features": len(m.get("feature_cols", [])),
        "n_params": m.get("n_params", ""),
        "testset_rmse_log": tm.get("rmse_log", ""),
        "testset_mape": tm.get("mape", ""),
        "testset_max_ape": tm.get("max_ape", ""),
        "testset_mape_tim2016": tim.get("mape", ""),
        "testset_mape_lbm": lbm.get("mape", ""),
    }

    inf = m.get("inference_timing")
    if inf is None and (vdir / "curated_eval.json").is_file():
        inf = _read_json(vdir / "curated_eval.json").get("inference_timing")
    if isinstance(inf, dict):
        row["inference_forward_mean_ms"] = inf.get("forward_mean_ms", "")
        row["inference_forward_std_ms"] = inf.get("forward_std_ms", "")
        row["inference_forward_median_ms"] = inf.get("forward_median_ms", "")
        row["inference_per_sample_us"] = inf.get("per_sample_mean_us", "")
        row["inference_throughput_samples_per_s"] = inf.get("throughput_samples_per_s", "")
        row["inference_batch_size"] = inf.get("batch_size", "")
        row["inference_device"] = inf.get("device", "")
        row["inference_warmup"] = inf.get("warmup", "")
        row["inference_repeats"] = inf.get("repeats", "")
    else:
        row["inference_forward_mean_ms"] = ""
        row["inference_forward_std_ms"] = ""
        row["inference_forward_median_ms"] = ""
        row["inference_per_sample_us"] = ""
        row["inference_throughput_samples_per_s"] = ""
        row["inference_batch_size"] = ""
        row["inference_device"] = ""
        row["inference_warmup"] = ""
        row["inference_repeats"] = ""

    ce_path = vdir / "curated_eval.json"
    if ce_path.is_file():
        ce = _read_json(ce_path)
        ov = ce.get("metrics_overall", {})
        row["curated_overall_mape"] = ov.get("mape", "")
        row["curated_overall_max_ape"] = ov.get("max_ape", "")
        row["curated_overall_rmse_log"] = ov.get("rmse_log", "")
        for i, b in enumerate(ce.get("metrics_per_bin", [])):
            row[f"curated_bin{i}_mape"] = b.get("mape", "")
    else:
        row["curated_overall_mape"] = ""
        row["curated_overall_max_ape"] = ""
        row["curated_overall_rmse_log"] = ""
        for i in range(10):
            row[f"curated_bin{i}_mape"] = ""

    return row


def _write_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    if not rows:
        print(f"No rows for {out_csv}")
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# Display names for the bar charts, matching the row labels of supplement
# Table 1 so the figure and the table can be read against each other. Keys are
# the variant directory names; anything unlisted falls back to its own name.
VARIANT_DISPLAY_NAMES = {
    "inertia": "inertia only",
    "chull": "hull only",
    "nonsph": "non-sph. only",
    "inertia_chull": "inertia + hull",
    "inertia_nonsph": "inertia + non-sph.",
    "all8": "all8 (baseline)",
    "h16_h8": "h=16",
    "h32_h16": "h=32",
    "h64_h32": "h=64",
    "h128_h64": "h=128",
    "h256_h128": "h=256",
}


def _display(name: str) -> str:
    return VARIANT_DISPLAY_NAMES.get(name, name)


# Dark variants are written alongside the light ones, as <name>_dark.png, so a
# dark-themed web page can swap them in without post-processing the PNGs. Only
# the lettering and the paper change; the bar colours are identical.
_DARK = False
_DARK_INK = "#e8e8e8"
_DARK_PAPER = "#151515"


def _theme_path(out_png: Path) -> Path:
    """Where this theme's copy of a figure goes."""
    return out_png.with_name(out_png.stem + "_dark" + out_png.suffix) if _DARK else out_png


def _apply_theme() -> None:
    if _DARK:
        # savefig(facecolor=...) paints only the figure patch; the axes patch
        # and the grid have to be set separately or the plot area stays white.
        plt.rcParams.update({"text.color": _DARK_INK, "axes.labelcolor": _DARK_INK,
                             "xtick.color": _DARK_INK, "ytick.color": _DARK_INK,
                             "axes.edgecolor": "#555555",
                             "axes.facecolor": _DARK_PAPER,
                             "figure.facecolor": _DARK_PAPER,
                             "grid.color": "#4a4a4a",
                             "legend.facecolor": _DARK_PAPER,
                             "legend.edgecolor": "#555555"})
    else:
        plt.rcParams.update(plt.rcParamsDefault)


def _save(out_png: Path) -> None:
    out = _theme_path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, facecolor=(_DARK_PAPER if _DARK else "white"))
    plt.close()


def _bar_chart(labels: list[str], values: list[float], title: str, ylabel: str, out_png: Path) -> None:
    labels = [_display(l) for l in labels]
    x = np.arange(len(labels))
    v = [float(np.nan if (isinstance(val, str) and val == "") else val) for val in values]
    _apply_theme()
    plt.figure(figsize=(max(8.0, len(labels) * 0.9), 4.2), dpi=140)
    plt.bar(x, v, color="steelblue")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(out_png)


def _grouped_per_bin_chart(
    variant_labels: list[str],
    per_variant_bin_mapes: list[list[float]],
    title: str,
    out_png: Path,
) -> None:
    variant_labels = [_display(l) for l in variant_labels]
    n_var = len(variant_labels)
    n_bins = len(per_variant_bin_mapes[0]) if per_variant_bin_mapes else 0
    if n_bins == 0:
        return
    w = 0.8 / max(1, n_var)
    x = np.arange(n_bins)
    _apply_theme()
    plt.figure(figsize=(12, 4.5), dpi=140)
    for i, label in enumerate(variant_labels):
        offs = (i - (n_var - 1) / 2.0) * w
        plt.bar(x + offs, per_variant_bin_mapes[i], width=w * 0.95, label=label)
    plt.xlabel("bin")
    plt.ylabel("MAPE (%)")
    plt.title(title)
    plt.legend(fontsize=7, ncol=min(3, n_var))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(out_png)


def _collect_rows_exp1() -> list[dict[str, object]]:
    """Feature-subset rows only.

    The released production checkpoint used to be appended as an extra
    "all8_baseline" bar, which put two near-identical eight-feature bars side by
    side in every exp1 chart (0.089 vs 0.089 on the test set) and invited the
    reader to look for a difference that is not there. The sweep now trains its
    own `all8` under exactly the recipe the other variants use, which is the
    like-for-like comparison the ablation is about, so the production row is no
    longer added.
    """
    return [_row_from_variant(p) for p in _variant_dirs("exp1_features")]


def _collect_rows_exp2() -> list[dict[str, object]]:
    return [_row_from_variant(p) for p in _variant_dirs("exp2_arch")]


def _backfill_exp2_inference_timing(device: str) -> None:
    """Re-run curated eval (fast) so ``metrics.json`` gains ``inference_timing`` if missing."""
    for vdir in _variant_dirs("exp2_arch"):
        mj_path = vdir / "metrics.json"
        tc_path = vdir / "train_config.json"
        if not mj_path.is_file() or not tc_path.is_file():
            continue
        mj = _read_json(mj_path)
        if mj.get("inference_timing"):
            continue
        tc = _read_json(tc_path)
        evaluate_on_curated(
            variant_dir=vdir,
            feature_keys=list(tc["feature_cols"]),
            hidden_dim=int(tc["hidden_dim"]),
            hidden_dim2=int(tc["hidden_dim2"]),
            device=device,
        )
        print(f"[backfill inference] {vdir.name}")


def _plots_for_exp(rows: list[dict[str, object]], prefix: str, out_dir: Path) -> None:
    rows_plot = [r for r in rows if r.get("variant")]
    if not rows_plot:
        return
    labels = [str(r["variant"]) for r in rows_plot]
    test_mapes = [float(r.get("testset_mape", 0) or 0) for r in rows_plot]
    cur_mapes = [float(r.get("curated_overall_mape", 0) or 0) for r in rows_plot]

    _bar_chart(
        labels,
        test_mapes,
        f"{prefix}: test set MAPE (%)",
        "MAPE (%)",
        out_dir / f"summary_{prefix}_testset.png",
    )
    _bar_chart(
        labels,
        cur_mapes,
        f"{prefix}: curated 10x10 overall MAPE (%)",
        "MAPE (%)",
        out_dir / f"summary_{prefix}_curated_overall.png",
    )

    per_bin: list[list[float]] = []
    for r in rows_plot:
        per_bin.append([float(r.get(f"curated_bin{b}_mape", 0) or 0) for b in range(10)])
    _grouped_per_bin_chart(
        labels,
        per_bin,
        f"{prefix}: curated per-bin MAPE",
        out_dir / f"summary_{prefix}_curated_per_bin.png",
    )

    if prefix == "exp2":
        inf_ms = []
        for r in rows_plot:
            v = r.get("inference_forward_mean_ms", "")
            try:
                inf_ms.append(float(v) if v != "" else float("nan"))
            except (TypeError, ValueError):
                inf_ms.append(float("nan"))
        if any(np.isfinite(np.asarray(inf_ms, dtype=np.float64))):
            _bar_chart(
                labels,
                inf_ms,
                "exp2: inference — mean forward time (full curated batch per trial)",
                "mean time (ms)",
                out_dir / f"summary_{prefix}_inference_forward_ms.png",
            )
        inf_us = []
        for r in rows_plot:
            v = r.get("inference_per_sample_us", "")
            try:
                inf_us.append(float(v) if v != "" else float("nan"))
            except (TypeError, ValueError):
                inf_us.append(float("nan"))
        if any(np.isfinite(np.asarray(inf_us, dtype=np.float64))):
            _bar_chart(
                labels,
                inf_us,
                "exp2: inference — mean time per sample (batch mean / N)",
                "time (us)",
                out_dir / f"summary_{prefix}_inference_per_sample_us.png",
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for exp2 inference backfill (CUDA events on GPU).",
    )
    parser.add_argument(
        "--no-backfill-inference",
        action="store_true",
        help="Do not re-run curated eval to fill missing exp2 inference_timing.",
    )
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Write the dark-theme copies (<name>_dark.png) instead of the light ones.",
    )
    args = parser.parse_args()

    global _DARK
    _DARK = args.dark

    # Each sweep's summary lands in its own experiment folder, beside the
    # per-variant runs it was built from.
    out1 = EXP_DIRS["exp1_features"]
    out2 = EXP_DIRS["exp2_arch"]
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    r1 = _collect_rows_exp1()
    _write_csv(r1, out1 / "summary_exp1.csv")
    _plots_for_exp(r1, "exp1", out1)
    print(f"Wrote {out1 / 'summary_exp1.csv'}")

    if not args.no_backfill_inference:
        _backfill_exp2_inference_timing(args.device)
    r2 = _collect_rows_exp2()
    _write_csv(r2, out2 / "summary_exp2.csv")
    _plots_for_exp(r2, "exp2", out2)
    print(f"Wrote {out2 / 'summary_exp2.csv'}")


if __name__ == "__main__":
    main()

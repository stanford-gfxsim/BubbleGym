"""Plot the width-ablation inference cost (supplement Fig. 5).

Reads the JSON written by ``benchmark_width_timing.py`` and draws one grouped
bar chart: per-bubble cost at batch 512 for each width, for the forward pass
alone and for the whole path including host->device transfer and readback --
the two figures Table 1 of the main paper quotes.

Error bars are the standard deviation across the benchmark's interleaved
passes, and they are the point of the figure rather than decoration: the spread
across the five widths is comparable to the run-to-run scatter, which is what
"the call is dispatch-bound, so width is free" actually means. A version of this
plot without them would imply a precision the measurement does not have.

Usage (from the repository root):

    python -m python.freq_model.ablation_study.plot_width_timing \
        --timings results/experiments/supp_table02_width_ablation/width_timing.json \
        --out fig/ablation_exp2_inference_batch512.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
DEFAULT_TIMINGS = (
    _HERE.parents[2] / "results" / "experiments"
    / "supp_table02_width_ablation" / "width_timing.json"
)

# Order by capacity, not by name: alphabetical gives h128, h16, h256, h32, h64.
ORDER = ["h16_h8", "h32_h16", "h64_h32", "h128_h64", "h256_h128"]
LABEL = {"h16_h8": "h=16", "h32_h16": "h=32", "h64_h32": "h=64",
         "h128_h64": "h=128", "h256_h128": "h=256"}

# Matches the other supplement figures.
FWD_COLOR, FULL_COLOR = "#4878A8", "#A8C4DE"


def main() -> int:
    ap = argparse.ArgumentParser(description="Width-ablation inference cost figure.")
    ap.add_argument("--timings", type=Path, default=DEFAULT_TIMINGS,
                    help="JSON from benchmark_width_timing.py")
    ap.add_argument("--out", type=Path, required=True, help="output PNG")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    if not args.timings.is_file():
        print(f"ERROR: {args.timings} not found. Run benchmark_width_timing.py "
              "first (see README_timing.md).", file=sys.stderr)
        return 2

    blob = json.loads(args.timings.read_text())
    res = blob["results"]

    order = [v for v in ORDER if v in res] or list(res)
    fwd = np.array([res[v]["forward_us_per_bubble_mean"] for v in order])
    fwd_e = np.array([res[v]["forward_us_per_bubble_std"] for v in order])
    full = np.array([res[v]["fullpath_us_per_bubble_mean"] for v in order])
    full_e = np.array([res[v]["fullpath_us_per_bubble_std"] for v in order])
    params = [res[v]["n_params"] for v in order]

    fig, ax = plt.subplots(figsize=(1120 / 140, 588 / 140), dpi=args.dpi)
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, fwd, w, yerr=fwd_e, capsize=3, color=FWD_COLOR,
           edgecolor="none", label="forward pass only", error_kw=dict(lw=0.9))
    ax.bar(x + w / 2, full, w, yerr=full_e, capsize=3, color=FULL_COLOR,
           edgecolor="none", label="+ transfer and readback", error_kw=dict(lw=0.9))

    for xi, v, e in zip(x - w / 2, fwd, fwd_e):
        ax.text(xi, v + e + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v, e in zip(x + w / 2, full, full_e):
        ax.text(xi, v + e + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    # no thousands separator, to match the supplement's width table
    ax.set_xticklabels([f"{LABEL.get(v, v)}\n{p} params"
                        for v, p in zip(order, params)], fontsize=8.5)
    ax.set_ylabel("time per bubble (µs)")
    # No internal title: the figure carries its LaTeX caption, as in the other
    # ablation figures. The batch size and pass count live in that caption.
    ax.set_ylim(0, float(max(full + full_e)) * 1.28)
    ax.yaxis.grid(True, color="#DDDDDD", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")

    env = blob.get("environment", {})
    if env:
        print(f"  measured on: {env.get('gpu')} / {env.get('cpu')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

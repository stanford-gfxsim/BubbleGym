"""Regenerate Fig. 1's three frequency-curve panels from the shipped scenes.

Runs ``plot_overlay_from_tracked.py`` once per scene (same CLI semantics as
invoking it manually)::

    python python/scene/bubble_theater/batch_plot_overlay_procedural_results.py

INPUTS: ``dataset/bubble_theater/<scene>/trackedBubInfo_{Minnaert,Ellipsoid,NN,BEM}.txt``
for scene in ellipsoid, curl_noise, enright_test (``--base-dir``).

OUTPUTS: ``teaser_{ellipsoid,curl_noise,enright}_freq_curves.jpg`` in
``results/experiments/fig01_bubble_theater/`` (``--out-dir``), 1350 x 675 px.

Paths are resolved from this file's location, so the working directory does
not matter.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# <repo>/python/scene/bubble_theater/this_file.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAY_SCRIPT = Path(__file__).resolve().parent / "plot_overlay_from_tracked.py"

DEFAULT_BASE = REPO_ROOT / "dataset" / "bubble_theater"
DEFAULT_OUT = REPO_ROOT / "results" / "experiments" / "fig01_bubble_theater"

# scene folder under --base-dir -> panel name in --out-dir
DEFAULT_SCENES = {
    "ellipsoid": "teaser_ellipsoid_freq_curves.jpg",
    "curl_noise": "teaser_curl_noise_freq_curves.jpg",
    "enright_test": "teaser_enright_freq_curves.jpg",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run plot_overlay_from_tracked.py on the Bubble Theater scenes and "
            "write Fig. 1's three panels."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE,
        help=f"Directory holding one folder per scene. Default: {DEFAULT_BASE}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Where the panels are written. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--fontsize",
        type=float,
        default=15.0,
        metavar="PT",
        help="Forwarded to plot_overlay_from_tracked.py (default: 15).",
    )
    parser.add_argument(
        "--legend-loc",
        type=str,
        choices=("upper-left", "upper-right"),
        default="upper-left",
        help="Forwarded to plot_overlay_from_tracked.py (default: upper-left).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print subprocess commands without executing them.",
    )
    args = parser.parse_args()

    base = args.base_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not base.is_dir():
        print(f"Error: base-dir is not a directory: {base}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, int]] = []
    for scene, panel in DEFAULT_SCENES.items():
        rd = base / scene
        if not rd.is_dir():
            print(f"[skip] not a directory: {rd}", file=sys.stderr)
            failures.append((str(rd), 2))
            continue

        cmd = [
            sys.executable,
            str(OVERLAY_SCRIPT),
            str(rd),
            "--png-name",
            str(out_dir / panel),
            "--csv-name=",
            "--fontsize",
            str(args.fontsize),
            "--legend-loc",
            str(args.legend_loc),
        ]
        print(f"\n=== {scene} ===\n  {' '.join(cmd)}")
        if args.dry_run:
            continue

        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            failures.append((str(rd), proc.returncode))

    if args.dry_run:
        print("\n(dry-run: no commands executed)")
        return 0

    if failures:
        print("\n--- failures ---", file=sys.stderr)
        for path, code in failures:
            print(f"  exit {code}: {path}", file=sys.stderr)
        return 1

    print("\n=== All panels OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Batch-regenerate ``freq_curves.png`` for the standard procedural result folders.

Runs ``plot_overlay_from_tracked.py`` once per directory (same CLI semantics as
invoking it manually)::

    python python/scene/bubble_theater/batch_plot_overlay_procedural_results.py \\
        --fontsize 15

REQUIRED INPUTS: one ``<scene>_result/`` folder of ``trackedBubInfo_<tag>.txt``
files per scene, under ``--base-dir``. These are not shipped -- generate them
first with ``render_bubble_theater.py``, whose default ``--output-dir`` is the
same ``dataset/bubble_theater/`` this script defaults to::

    python python/scene/bubble_theater/render_bubble_theater.py \\
        dataset/bubble_theater/curl_noise/mesh

Default targets under ``--base-dir``: ``curl_noise_result/``,
``ellipsoid_result/``, ``enright_test_result/``. Paths are resolved from this
file's location, so the working directory does not matter.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# <repo>/python/scene/bubble_theater/this_file.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAY_SCRIPT = Path(__file__).resolve().parent / "plot_overlay_from_tracked.py"

# Matches render_bubble_theater.py's default --output-dir, which is where it
# writes each scene's <name>_result/ folder.
DEFAULT_BASE = REPO_ROOT / "results" / "experiments" / "fig01_bubble_theater"

DEFAULT_RESULT_DIRS = (
    "curl_noise_result",
    "ellipsoid_result",
    "enright_test_result",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run plot_overlay_from_tracked.py on the default "
            "bubble_theater *_result folders (or a custom list)."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the *_result folders. Default: "
            f"{DEFAULT_BASE} (resolved from this script's location)."
        ),
    )
    parser.add_argument(
        "--result-dirs",
        type=str,
        default=",".join(DEFAULT_RESULT_DIRS),
        help=(
            "Comma-separated folder names under --base-dir (default: the "
            "three standard procedural results)."
        ),
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

    base = (args.base_dir if args.base_dir is not None else DEFAULT_BASE).resolve()
    if not base.is_dir():
        print(f"Error: base-dir is not a directory: {base}", file=sys.stderr)
        return 2

    names = [tok.strip() for tok in args.result_dirs.split(",") if tok.strip()]
    if not names:
        print("Error: --result-dirs resolved to an empty list", file=sys.stderr)
        return 2

    if not OVERLAY_SCRIPT.is_file():
        print(
            f"Error: overlay script not found: {OVERLAY_SCRIPT}",
            file=sys.stderr,
        )
        return 2

    failures: list[tuple[str, int]] = []
    for name in names:
        rd = (base / name).resolve()
        if not rd.is_dir():
            print(f"[skip] not a directory: {rd}", file=sys.stderr)
            failures.append((str(rd), 2))
            continue

        cmd = [
            sys.executable,
            str(OVERLAY_SCRIPT),
            str(rd),
            "--fontsize",
            str(args.fontsize),
            "--legend-loc",
            str(args.legend_loc),
        ]
        print(f"\n=== {rd.name} ===\n  {' '.join(cmd)}")
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

    print("\n=== All overlays OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

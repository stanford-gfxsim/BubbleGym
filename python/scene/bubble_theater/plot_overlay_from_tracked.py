"""Re-render ``freq_curves.png`` from on-disk ``trackedBubInfo_<tag>.txt`` files.

Sister utility to ``render_bubble_theater.py``: scans a result
folder for ``trackedBubInfo_{Minnaert,Ellipsoid,NN,BEM}.txt`` files
(any subset present is fine), parses out the ``(time, frequency)``
columns from each, and writes a single overlay
``freq_curves.png`` (and ``freq_curves_overlay.csv``) using the same
``_plot_freq_curves`` helper that the main pipeline uses.

The overlay plot uses a friendlier legend label for the network
(``NN8`` → ``Learning model``); file names and CSV columns still use
the canonical tags. Legend position is
fixed to a corner via ``--legend-loc upper-left`` or ``upper-right``
(default: upper-left).

Useful for:

* refreshing ``freq_curves.png`` while a long BEM run is still in flight
  (it'll show whatever frames the most recent checkpoint flushed; rows
  not yet solved appear as ``nan`` in the trackedBubInfo and are
  skipped by the plot);
* combining outputs of multiple ``--methods`` invocations of the main
  script into one overlay without recomputing anything.

Example
-------
::

    python python/scene/bubble_theater/plot_overlay_from_tracked.py \\
        dataset/bubble_theater/curl_noise_result
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from scene.bubble_theater.render_bubble_theater import (  # noqa: E402
    METHOD_TAGS,
    _plot_freq_curves,
    _write_freq_curves_csv,
)

# Legend strings for the overlay PNG only. File tags are Minnaert / Ellipsoid /
# NN8 / BEM; the CSV column for the ellipsoid proxy is still ``f_strasberg``,
# after the analytic model it implements.
_OVERLAY_LEGEND_LABELS: dict[str, str] = {
    "NN8": "Learning model",
}

# Matplotlib ``loc`` strings for ``Axes.legend`` (overlay only; two corners).
_OVERLAY_LEGEND_LOC: dict[str, str] = {
    "upper-left": "upper left",
    "upper-right": "upper right",
}

_HEADER_RE = re.compile(r"^\s*Bub\s+\S+\s+(\S+)\s*$")


def _parse_tracked_bubinfo(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return ``(times, freqs, radius_eq)`` parsed from a trackedBubInfo file.

    ``radius_eq`` is taken from the leading ``Bub <id> <R_eq>`` header
    line; ``None`` if the file did not contain a parseable header.
    Sample rows are any leading-numeric lines (``time freq x y z p``);
    non-finite freq tokens (``nan``, ``inf``) are preserved as ``NaN``
    so the plot can skip them via ``np.isfinite``.
    """
    times: list[float] = []
    freqs: list[float] = []
    radius_eq: float | None = None
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        m = _HEADER_RE.match(raw)
        if m is not None:
            try:
                radius_eq = float(m.group(1))
            except ValueError:
                radius_eq = None
            continue
        if s.startswith("Start") or s.startswith("End"):
            continue
        toks = s.split()
        if len(toks) < 2:
            continue
        try:
            t = float(toks[0])
        except ValueError:
            continue
        try:
            f = float(toks[1])
        except ValueError:
            f = float("nan")
        if not math.isfinite(f):
            f = float("nan")
        times.append(t)
        freqs.append(f)
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(freqs, dtype=np.float64),
        radius_eq,
    )


def _align_to_reference(
    f_arr: np.ndarray, n_ref: int
) -> np.ndarray:
    """Pad with ``NaN`` (shorter) or truncate (longer) to length ``n_ref``."""
    if f_arr.size == n_ref:
        return f_arr
    if f_arr.size < n_ref:
        out = np.full(n_ref, np.nan, dtype=np.float64)
        out[: f_arr.size] = f_arr
        return out
    return f_arr[:n_ref]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read trackedBubInfo_<tag>.txt files in RESULT_DIR and write a "
            "single overlay freq_curves.png + freq_curves_overlay.csv"
        )
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help=(
            "A *_result/ folder produced by render_bubble_theater.py "
            "(e.g. dataset/bubble_theater/curl_noise_result)"
        ),
    )
    parser.add_argument(
        "--png-name",
        type=str,
        default="freq_curves.png",
        help="Output PNG filename inside RESULT_DIR (default: freq_curves.png)",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="freq_curves_overlay.csv",
        help=(
            "Output CSV filename inside RESULT_DIR with one row per frame "
            "(default: freq_curves_overlay.csv). Pass empty string to skip "
            "the CSV write."
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(METHOD_TAGS),
        help=(
            "Comma-separated subset of method tags to look for "
            f"(default: {','.join(METHOD_TAGS)}). Tags whose file is missing "
            "in RESULT_DIR are silently skipped."
        ),
    )
    parser.add_argument(
        "--title-extras",
        type=str,
        default="",
        help=(
            "Extra text appended in parentheses to the plot title. Default "
            "empty -> '<R_eq=... m, source=<result_dir basename>>' is used."
        ),
    )
    parser.add_argument(
        "--fontsize",
        type=float,
        default=15.0,
        metavar="PT",
        help=(
            "Font size (pt) for axis labels, tick labels, and legend in the "
            "overlay PNG (default: 15). Pass 0 to use matplotlib defaults."
        ),
    )
    parser.add_argument(
        "--legend-loc",
        type=str,
        choices=tuple(_OVERLAY_LEGEND_LOC.keys()),
        default="upper-left",
        metavar="CORNER",
        help=(
            "Legend corner for the overlay PNG only: upper-left or "
            "upper-right (default: upper-left). "
            "Matplotlib ``loc`` is fixed to these two options so the legend "
            "does not float to ``best``."
        ),
    )
    args = parser.parse_args()

    rd = args.result_dir.resolve()
    if not rd.is_dir():
        print(f"Error: not a directory: {rd}", file=sys.stderr)
        return 2

    requested = [tok.strip() for tok in args.methods.split(",") if tok.strip()]
    norm: list[str] = []
    for tok in requested:
        for canon in METHOD_TAGS:
            if tok.lower() == canon.lower():
                norm.append(canon)
                break
        else:
            print(
                f"Warning: ignoring unknown method tag {tok!r} "
                f"(valid: {','.join(METHOD_TAGS)})",
                file=sys.stderr,
            )

    found: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    radii: dict[str, float] = {}
    for tag in norm:
        p = rd / f"trackedBubInfo_{tag}.txt"
        if not p.is_file():
            print(f"  skip {tag}: {p.name} not present")
            continue
        times, freqs, r_eq = _parse_tracked_bubinfo(p)
        if times.size == 0:
            print(f"  skip {tag}: parsed 0 sample rows from {p.name}")
            continue
        n_finite = int(np.sum(np.isfinite(freqs)))
        print(
            f"  load {tag}: {times.size} rows ({n_finite} finite freq), "
            f"R_eq={r_eq if r_eq is not None else 'n/a'} from {p.name}"
        )
        found[tag] = (times, freqs)
        if r_eq is not None and math.isfinite(r_eq):
            radii[tag] = float(r_eq)

    if not found:
        print(
            f"Error: no trackedBubInfo files found in {rd}",
            file=sys.stderr,
        )
        return 1

    # Pick the longest time grid as the reference (handles transient cases
    # where some files have stale shorter row counts).
    ref_tag = max(found.keys(), key=lambda t: found[t][0].size)
    times_ref = found[ref_tag][0]
    n_ref = times_ref.size

    aligned: dict[str, np.ndarray] = {}
    for tag, (t, f) in found.items():
        if t.size != n_ref:
            print(
                f"  note {tag}: {t.size} rows; padding/truncating to "
                f"reference {ref_tag} ({n_ref} rows)"
            )
        aligned[tag] = _align_to_reference(f, n_ref)

    title_extras = args.title_extras.strip()
    if not title_extras:
        bits: list[str] = []
        bits.append(f"source={rd.name}")
        title_extras = ", ".join(bits)

    png_path = rd / args.png_name
    fs = float(args.fontsize)
    fontsize_kw = None if fs <= 0.0 else fs
    _plot_freq_curves(
        png_path,
        times_ref,
        aligned,
        title_extras=title_extras,
        legend_labels=_OVERLAY_LEGEND_LABELS,
        fontsize=fontsize_kw,
        legend_loc=_OVERLAY_LEGEND_LOC[args.legend_loc],
    )
    print(f"wrote {png_path}")

    if args.csv_name:
        csv_path = rd / args.csv_name
        # _write_freq_curves_csv writes columns k, t, then one column per
        # method in METHOD_TAGS order (only for tags present in the dict).
        _write_freq_curves_csv(csv_path, times_ref, aligned)
        print(f"wrote {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

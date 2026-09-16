"""Learned-model vs Minnaert frequency for the single rising bubble (paper figure).

Companion to the single-rising-bubble figure: the renders show the bubble
deforming from a sphere into a torus, and this plot shows what that does to the
predicted resonance. Both curves come from the delivered trackedBubInfo files
rather than from analysis of the rendered audio, so they are the frequencies
actually handed to the synthesizer.

The two files share one event graph and differ only in the frequency column.
Row format is::

    <time_s> <freq_Hz> <x> <y> <z> <pressure_avg> <depth_m>

preceded by ``Bub <id> <R_eq>`` and ``Start: <tag> <n>`` header lines. The
``_minnaert`` file is constant per bubble (f * R_header is fixed); the ``_nn``
file carries the time-varying mesh-NN frequency, PCHIP-interpolated between
per-bubble mesh keypoints, so it is C1 within a track and steps only at real
events -- the small ripples in the curve are those keypoints, not noise.

Bubble 1 is the rising bubble the figure follows (R = 23.5 mm, alive
0.033-1.204 s); everything else in the scene is later splash.

The data ships in this repo at ``dataset/single_rising_bubble/``, which is the
``--data`` default; point the flag elsewhere to plot another copy.

RUN (needs only numpy + matplotlib):
    python -u python/utils/plot_single_bubble_freq.py --out <path>.png
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
matplotlib.rcParams["font.size"] = 14.0
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

# trackedBubInfo_{NN,Minnaert,BEM}.txt, one event graph across the three.
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "dataset" / "single_rising_bubble"
BUB_ID = 1
TMAX_S = 1.2          # matches the spectrogram figure's time axis
FIGSIZE = (9.6, 4.0)  # same as the spectrogram figure, so the two composite
YTICK_HZ = 5.0        # y ticks every 5 Hz
FPS = 30.0
FRAMES = [0, 1, 2, 4, 7, 15]   # the frames shown in the figure above the plot

# The file has one row per LBM step (~12,048 Hz), but the NN runs once per
# exported per-bubble mesh -- every 50 LBM steps, ~241 Hz -- and the rows in
# between are the PCHIP interpolant. Verified by reconstruction: resampling onto
# a stride-50 grid and re-interpolating reproduces the series to 1e-4 Hz mean /
# 0.04 Hz max, while unaligned strides (20, 40, 60, 80) are 5-50x worse even
# with MORE knots. Plotting on this grid shows what the model predicted.
KEYPOINT_STRIDE = 50

# Styles follow the regressor-comparison figure so the two read as one system.
NN_STYLE = {"color": "#FF7F0E", "linestyle": "-", "linewidth": 3.4}
MN_STYLE = {"color": "#7F7F7F", "linestyle": "--", "linewidth": 3.0}
BEM_STYLE = {"color": "#2CA02C", "linestyle": "-", "linewidth": 2.2}
POLY2_STYLE = {"color": "#9467BD", "linestyle": (0, (1, 1)), "linewidth": 2.4}

# The BEM sweep writes per-(bubble, mesh) rows keyed by LBM iteration index with
# the frequency at UNIT volume, so overlaying them on a tracked series needs
#   t      = frame * DT_LBM
#   f_real = f_unit * R_unit / R_eq   (R_eq from the track's Bub header)
# Checked on bub 1: frame 400 -> t = 0.0333 s, its birth time in the track.
DT_LBM = 1.0 / 12000.0
R_UNIT = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)


def parse_radius_eq(path: Path, bub_id: int) -> float:
    """R_eq from the ``Bub <id> <R_eq>`` header of one record."""
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        tok = raw.split()
        if len(tok) == 3 and tok[0] == "Bub" and int(tok[1]) == bub_id:
            return float(tok[2])
    raise SystemExit(f"no Bub {bub_id} header in {path}")


def parse_bem_rows(srcs: list[Path], bub_id: int, r_eq: float):
    """(t, f_real) of the BEM solves for one bubble, from the sweep's rows.

    Each source is either a merged ``<label>_all_rows.csv`` or the ``chunks``
    directory of a sweep (running or finished). Rows whose ``f_hz_bem`` is
    blank are meshes above ``--bem-max-vertices``: the sweep records NN timing
    for them but no BEM solve, so they are simply absent from the curve.

    Several sources may be given and are merged by mesh frame, later sources
    winning. That is what lets a partial uncapped run be shown alongside an
    earlier capped one — the solver settings are identical, only the vertex
    cap differed, so a given mesh has the same solution in either.
    """
    by_frame: dict[int, float] = {}
    n_seen = 0
    for src in srcs:
        files = sorted(src.glob("chunk_*.csv")) if src.is_dir() else [src]
        if not files:
            raise SystemExit(f"no sweep CSVs under {src}")
        for f in files:
            with f.open(newline="", encoding="utf-8") as h:
                for row in csv.DictReader(h):
                    if row.get("bub_id") != str(bub_id):
                        continue
                    n_seen += 1
                    if not row.get("f_hz_bem"):
                        continue
                    by_frame[int(row["frame"])] = (
                        float(row["f_hz_bem"]) * R_UNIT / r_eq)
    frames = sorted(by_frame)
    t = np.array([fr * DT_LBM for fr in frames])
    f = np.array([by_frame[fr] for fr in frames])
    print(f"  BEM: {len(t)} distinct meshes solved (from {n_seen} rows across "
          f"{len(srcs)} source(s)); coverage t <= {t.max():.3f}s")
    return t, f


def parse_regressor_rows(src: Path, bub_id: int, r_eq: float, column: str):
    """(t, f_real) of one regressor column from apply_regressors_all_bubbles.py.

    That script is a post-pass over the BEM sweep: it re-extracts the eight
    shape features per mesh and applies the 10k-trained regressors, writing
    ``<label>_regressor_freqs.csv`` with f_hz_bem / f_hz_nn8_10k / f_hz_poly2
    at unit volume, keyed by the same LBM frame index. Same rescale as the
    BEM rows.
    """
    by_frame: dict[int, float] = {}
    with src.open(newline="", encoding="utf-8") as h:
        for row in csv.DictReader(h):
            if row.get("bub_id") != str(bub_id) or not row.get(column):
                continue
            by_frame[int(row["frame"])] = float(row[column]) * R_UNIT / r_eq
    frames = sorted(by_frame)
    print(f"  {column}: {len(frames)} points")
    return (np.array([fr * DT_LBM for fr in frames]),
            np.array([by_frame[fr] for fr in frames]))


def sweep_compute_times(srcs: list[Path], bub_id: int) -> tuple[float, float, int]:
    """(BEM seconds, learned-model seconds, n meshes) for this bubble's curve.

    Summed over the same rows the curves are drawn from, so the legend figure
    is the actual measured cost of this bubble, not a scene-level average.
    BEM is assemble+solve; the learned model is shape features + GPU inference,
    which is what the pipeline spends per mesh.
    """
    seen: dict[int, tuple[float, float]] = {}
    for src in srcs:
        files = sorted(src.glob("chunk_*.csv")) if src.is_dir() else [src]
        for f in files:
            with f.open(newline="", encoding="utf-8") as h:
                for row in csv.DictReader(h):
                    if row.get("bub_id") != str(bub_id):
                        continue
                    bem = float(row["t_bem_s"]) if row.get("t_bem_s") else 0.0
                    nn = 0.0
                    for col in ("t_feat_s", "t_inference_gpu_s"):
                        if row.get(col):
                            nn += float(row[col])
                    seen[int(row["frame"])] = (bem, nn)
    return (sum(b for b, _ in seen.values()),
            sum(n for _, n in seen.values()),
            len(seen))


def fmt_seconds(s: float) -> str:
    if s < 1.0:
        return f"{s * 1000:.0f} ms"
    if s < 60.0:
        return f"{s:.1f} s"
    if s < 3600.0:
        return f"{s / 60.0:.1f} min"
    return f"{s / 3600.0:.1f} h"


def parse_track(path: Path, bub_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Times and frequencies of one bubble record."""
    times: list[float] = []
    freqs: list[float] = []
    cur = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        tok = raw.split()
        if not tok:
            continue
        if tok[0] == "Bub":
            cur = int(tok[1])
        elif tok[0] == "Start:":
            continue
        elif cur == bub_id and len(tok) >= 2:
            try:
                times.append(float(tok[0]))
                freqs.append(float(tok[1]))
            except ValueError:
                pass
    if not times:
        raise SystemExit(f"no samples for bubble {bub_id} in {path}")
    return np.asarray(times), np.asarray(freqs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help=f"Directory holding the two trackedBubInfo files "
                         f"(default: {DEFAULT_DATA}).")
    ap.add_argument("--out", type=Path, required=True, help="Output image path.")
    ap.add_argument("--bubble", type=int, default=BUB_ID)
    ap.add_argument("--tmax", type=float, default=TMAX_S)
    ap.add_argument("--mark-frames", action="store_true",
                    help="Draw the figure's frame instants as vertical rules.")
    ap.add_argument("--stride", type=int, default=KEYPOINT_STRIDE,
                    help=f"Plot every Nth row. The default {KEYPOINT_STRIDE} is "
                         f"the mesh-keypoint grid, i.e. the rate the NN actually "
                         f"runs at (~241 Hz); pass 1 to draw the interpolated "
                         f"per-LBM-step series instead.")
    ap.add_argument("--bem-rows", type=Path, nargs="+", default=None,
                    help="Sweep rows CSV(s) or chunks/ dir(s); overlays BEM "
                         "ground truth. Multiple sources merge by mesh frame, "
                         "later ones winning.")
    ap.add_argument("--poly2-rows", type=Path, default=None,
                    help="<label>_regressor_freqs.csv; overlays the degree-2 "
                         "polynomial baseline.")
    ap.add_argument("--legend-times", action="store_true",
                    help="Append each method's total compute time for this "
                         "bubble to its legend entry, summed from the sweep rows.")
    ap.add_argument("--markers", action="store_true",
                    help="Dot each NN inference point.")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"--data is not a directory: {args.data}")
    def _pick(suffix: str) -> Path:
        """First *<suffix>.txt in --data, matched case-insensitively.

        The shipped files are trackedBubInfo_NN.txt / _Minnaert.txt; other
        copies use *_nn.txt / *_minnaert.txt, and only Windows treats those
        two globs as the same.
        """
        hits = sorted(p for p in args.data.glob("*.txt")
                      if p.stem.lower().endswith(suffix))
        if not hits:
            raise SystemExit(f"no *{suffix}.txt in {args.data}")
        return hits[0]

    nn_file = _pick("_nn")
    mn_file = _pick("_minnaert")

    t_nn, f_nn = parse_track(nn_file, args.bubble)
    t_mn, f_mn = parse_track(mn_file, args.bubble)
    row_dt = float(np.median(np.diff(t_nn)))
    if args.stride > 1:                     # keep only the NN inference points
        t_nn, f_nn = t_nn[::args.stride], f_nn[::args.stride]
        t_mn, f_mn = t_mn[::args.stride], f_mn[::args.stride]
    t_nn, f_nn = t_nn[t_nn <= args.tmax], f_nn[t_nn <= args.tmax]
    t_mn, f_mn = t_mn[t_mn <= args.tmax], f_mn[t_mn <= args.tmax]

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    if args.mark_frames:
        for fr in FRAMES:
            ax.axvline(fr / FPS, color="black", lw=1.0, ls="--", dashes=(4, 3), zorder=0)
    bem_lbl, nn_lbl = "BEM (ground truth)", "Learned model"
    if args.bem_rows:
        r_eq = parse_radius_eq(nn_file, args.bubble)
        t_bem, f_bem = parse_bem_rows(args.bem_rows, args.bubble, r_eq)
        if args.legend_times:
            t_b, t_n, n_mesh = sweep_compute_times(args.bem_rows, args.bubble)
            # per-mesh figures both in ms so the ratio is readable at a glance
            # Totals only. The per-mesh figure was the same fact divided by
            # 282 and made the legend entries long enough to crowd the curves.
            bem_lbl += f": {fmt_seconds(t_b)}"
            nn_lbl += f": {fmt_seconds(t_n)}"
            print(f"  compute over {n_mesh} meshes: BEM {t_b:,.0f}s "
                  f"({t_b / n_mesh:.1f}s/mesh) vs learned {t_n:,.1f}s "
                  f"({t_n / n_mesh * 1000:.1f}ms/mesh) -> {t_b / t_n:.0f}x")
        keep = t_bem <= args.tmax
        ax.plot(t_bem[keep], f_bem[keep], label=bem_lbl, **BEM_STYLE)
    if args.poly2_rows:
        r_eq2 = parse_radius_eq(nn_file, args.bubble)
        t_p2, f_p2 = parse_regressor_rows(args.poly2_rows, args.bubble, r_eq2, "f_hz_poly2")
        k2 = t_p2 <= args.tmax
        ax.plot(t_p2[k2], f_p2[k2], label="Poly (deg 2)", **POLY2_STYLE)
    ax.plot(t_mn, f_mn, label="Minnaert", **MN_STYLE)
    nn_style = dict(NN_STYLE)
    if args.markers:
        nn_style.update(marker="o", markersize=3.0, markevery=1)
    ax.plot(t_nn, f_nn, label=nn_lbl, **nn_style)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.yaxis.set_major_locator(MultipleLocator(YTICK_HZ))
    ax.set_xlim(0.0, args.tmax)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)

    cents = 1200.0 * np.log2(f_nn.max() / f_mn[0])
    rate = 1.0 / (args.stride * row_dt)
    print(f"bubble {args.bubble}: {len(t_nn)} points at stride {args.stride} "
          f"({args.stride * row_dt * 1000:.2f} ms, {rate:.0f} Hz), "
          f"{t_nn.min():.3f}-{t_nn.max():.3f} s")
    print(f"  Minnaert  {f_mn[0]:.2f} Hz (constant)")
    print(f"  learned   {f_nn[0]:.2f} -> {f_nn.max():.2f} Hz")
    print(f"  max separation {f_nn.max() - f_mn[0]:.2f} Hz = {cents:.0f} cents")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

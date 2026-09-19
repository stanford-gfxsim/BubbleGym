"""End-to-end pipeline timing breakdown as a per-scene pair of pies (paper Fig. 8).

One pipeline run costs LBM sim, bubble tracking, state-dump I/O, per-bubble mesh
extraction, frequency prediction and sound synthesis. Every stage costs the same
under both frequency solvers except the frequency leg, which is exactly what the
surrogate replaces, so the two pies differ in one slice and share one legend.

It also prints the totals, per-stage shares and the end-to-end speedup, so the
caption and prose can be restated from the same source rather than being
hand-maintained.

Stage costs are read from a timings JSON per scene (seconds; keys ``lbm``,
``tracking``, ``dump``, ``mesh``, ``freq_bem``, ``freq_ours``, ``audio``). The
two shipped under ``results/experiments/`` are the paper's, and are what gets
drawn when no ``--scene`` / ``--timings-json`` is given.

The figure is drawn at exactly the paper's ``\\columnwidth`` so it can be
included with ``width=\\linewidth`` at 1:1, which makes a matplotlib
``fontsize`` land as that many real points on the page.

RUN (needs only numpy + matplotlib):
    python -u python/utils/plot_end2end_timing_pies.py --out <path>.png
    python -u python/utils/plot_end2end_timing_pies.py --out <path>.png \
        --scene "Fruit Splash=results/experiments/fig08_end_to_end_timing/fruit08_end2end_timings.json" \
        --scene "Exhale=results/experiments/fig08_end_to_end_timing/exhale15_end2end_timings.json"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# \columnwidth in the paper's acmtog layout, in inches.
COLUMN_WIDTH_IN = 243.15 / 72.0

# (key, legend name, colour) in pie order.
STAGES: list[tuple[str, str, str]] = [
    ("lbm", "LBM simulation", "#3B7DD8"),
    ("tracking", "Bubble tracking", "#E8613C"),
    ("dump", "VOF field/bubble tag dump I/O", "#17BECF"),
    ("mesh", "Mesh extraction", "#F0A30A"),
    ("freq", "Frequency estimation", "#E377C2"),
    ("audio", "Audio synthesis", "#2CA02C"),
]

# Every key a timings JSON must carry, in seconds.
TIMING_KEYS = ("lbm", "tracking", "dump", "mesh", "freq_bem", "freq_ours", "audio")

RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results" / "experiments" / "fig08_end_to_end_timing"
)

# The paper's Fig. 8: the two shipped scene ledgers, drawn when the command line
# names no timings of its own. There are deliberately no hard-coded stage costs
# here -- an out-of-date literal draws a plausible figure that is not the
# paper's, and says nothing about it.
DEFAULT_SCENES: list[tuple[str, Path]] = [
    ("Fruit Splash", RESULTS_DIR / "fruit08_end2end_timings.json"),
    ("Exhalation", RESULTS_DIR / "exhale15_end2end_timings.json"),
]

# Multi-scene (paper) layout. Ink and panel tints for the column headers, the
# row labels, the totals and the legend frame, per theme.
INK = {"light": "#1E2A4A", "dark": "#E8ECF2"}
HEADER_FILL = {
    "light": {"bem": "#E2E8F0", "ours": "#D6E4F7"},
    "dark": {"bem": "#2A313B", "ours": "#23344D"},
}
LEGEND_FILL = {"light": "#F1F5F9", "dark": "#1E2329"}
LEGEND_EDGE = {"light": "#CBD5E1", "dark": "#3A424A"}
# Left strip that holds the scene names, as a fraction of the figure width.
ROW_LABEL_W = 0.205
# Pie axes data limits: room on the left for the stacked outside labels and
# below for the total; the unit-radius pie sits at x = 0. Only the surrogate
# column has outside labels, so the BEM column is cropped to the pie and its
# axes narrowed to keep the same scale -- otherwise the unused margin opens a
# wide gap between the scene names and the BEM pies.
PIE_XLIM = (-1.62, 1.08)
BEM_XLIM = (-1.08, 1.08)
PIE_YLIM = (-1.44, 1.30)

LABEL_FONTSIZE = 7.0
TITLE_FONTSIZE = 7.0
PCT_FONTSIZE = 5.5
LEGEND_FONTSIZE = 6.0
# The scene name sits at the column-title size and weight. What made the old
# header read as a heavy title bar was its length -- it carried both totals
# and the speedup on one line -- not the size or the weight.
SCENE_FONTSIZE = 7.0
# Each pie's total, set underneath it. At the title size: it is the headline
# number for its pie, not an annotation on it.
TOTAL_FONTSIZE = 7.0
# Where that total sits, in axis units. The pie has radius 1 and the axes run
# to PIE_LIMITS, so this is inside the existing margin -- no extra band height.
TOTAL_Y = -1.26
# Slices below this share get no in-pie label; the legend carries their cost.
PCT_LABEL_MIN = 4.0
# Multi-scene panels label every wedge with its share and duration. A wedge
# narrower than this cannot hold the text: white-on-wedge lettering that runs
# past the arc lands on the white page and simply disappears, so those wedges
# take an outside label on a leader instead. The threshold is set by the widest
# duration string, not by taste -- at this pie size "33.5 min" needs ~15 %.
INSIDE_MIN_PCT = 15.0
INSIDE_FONTSIZE = 5.0
INSIDE_RADIUS = 0.58
# Between this and INSIDE_MIN_PCT a wedge is too narrow for a duration string
# but wide enough for a bare percentage, so it keeps its share on the wedge and
# sends only the duration out to the leader.
PCT_ONLY_MIN_PCT = 7.0
PCT_ONLY_FONTSIZE = 4.2
OUTSIDE_FONTSIZE = 4.5
# Minimum vertical separation between stacked outside labels, in axis units.
OUTSIDE_MIN_DY = 0.19
# Multi-scene pies leave a margin for those outside labels.
PIE_LIMITS = (-1.50, 1.50)


def fmt_duration(seconds: float) -> str:
    """Human-readable duration, in the unit that keeps 2-3 significant digits."""
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.1f} h"


def _label_wedges(ax, values, total, only=None):
    """Write "share + duration" on each wedge of an already-drawn pie.

    Wide wedges carry the label inside, in white. Thin ones cannot -- see
    ``INSIDE_MIN_PCT`` -- so they get an outside label on a leader, stacked
    vertically because the small stages are adjacent in pie order and would
    otherwise pile up. Those outside labels are drawn in their stage's own
    colour: with three or four of them in a column, a leader line alone does not
    say which is which, and the legend is too far away to help.

    ``only`` restricts labelling to the named stage keys.
    """
    angle = 90.0
    inside, outside = [], []
    for (key, _name, color), val in zip(STAGES, values):
        pct = val / total * 100.0
        span = 360.0 * val / total
        mid = math.radians(angle - span / 2.0)
        angle -= span
        if only is not None and key not in only:
            continue
        if pct >= INSIDE_MIN_PCT:
            inside.append((mid, f"{pct:.0f}%\n{fmt_duration(val)}",
                           INSIDE_FONTSIZE))
        else:
            if pct >= PCT_ONLY_MIN_PCT:
                inside.append((mid, f"{pct:.0f}%", PCT_ONLY_FONTSIZE))
            outside.append((mid, fmt_duration(val), color))

    for mid, text, size in inside:
        ax.text(INSIDE_RADIUS * math.cos(mid), INSIDE_RADIUS * math.sin(mid),
                text, ha="center", va="center", color="white",
                fontsize=size, fontweight="bold", linespacing=1.05)

    # Walk down from the topmost so a run of thin wedges reads as a column.
    outside.sort(key=lambda t: -math.sin(t[0]))
    y_prev = None
    for mid, text, color in outside:
        y = math.sin(mid) * 1.12
        if y_prev is not None and y > y_prev - OUTSIDE_MIN_DY:
            y = y_prev - OUTSIDE_MIN_DY
        y_prev = y
        ax.annotate(
            text,
            xy=(math.cos(mid) * 0.97, math.sin(mid) * 0.97),
            xytext=(-1.04, y), ha="right", va="center",
            fontsize=OUTSIDE_FONTSIZE, color=color, fontweight="bold",
            arrowprops={"arrowstyle": "-", "linewidth": 0.3,
                        "color": color, "alpha": 0.7,
                        "shrinkA": 1.0, "shrinkB": 0.5},
        )


def build_variants(c: dict[str, float]) -> dict[str, dict[str, float]]:
    shared = {k: c[k] for k in ("lbm", "tracking", "dump", "mesh", "audio")}
    return {
        "bem": {**shared, "freq": c["freq_bem"]},
        "ours": {**shared, "freq": c["freq_ours"]},
    }


def plot(out_path: Path, scenes: list[tuple[str, dict[str, dict[str, float]]]],
         ours_label: str = "Our surrogate", dpi: int = 600,
         dark: bool = False) -> None:
    """One row of two pies per scene, sharing a single legend.

    With one scene the legend carries each stage's absolute cost, since every
    stage but frequency estimation is the same under both solvers. With several
    scenes it cannot: the shared stages differ from run to run, so a single
    legend row has no one number to show. The legend then names the stages only
    and the absolute costs move into the per-row headers, which give each
    scene's two totals and its speedup -- the numbers the prose actually quotes.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    panels = [("bem", "BEM reference"), ("ours", ours_label)]
    if dark:
        plt.rcParams.update({"text.color": "#e8e8e8",
                             "axes.labelcolor": "#e8e8e8",
                             "xtick.color": "#e8e8e8",
                             "ytick.color": "#e8e8e8"})

    n_rows = len(scenes)
    multi = n_rows > 1

    # Explicit layout rather than constrained_layout: the pie axes force a 1:1
    # aspect, which the layout engine resolves by leaving a large hole between
    # the pies and the legend and by clipping the two-line titles.
    theme = "dark" if dark else "light"
    ink = INK[theme]
    if multi:
        # Bands placed by hand, top to bottom: the boxed column headers, one
        # band of pies per scene with its name in a left strip, then the framed
        # legend. The pie axes take a 1:1 aspect, so each band's height follows
        # from the column width rather than from a grid.
        from matplotlib.patches import FancyBboxPatch

        titles_h, gap_h, legend_h = 0.20, 0.06, 0.40
        span_x = PIE_XLIM[1] - PIE_XLIM[0]
        bem_span_x = BEM_XLIM[1] - BEM_XLIM[0]
        span_y = PIE_YLIM[1] - PIE_YLIM[0]
        # Both columns share one data-to-figure scale; the BEM one is narrower.
        ours_w = (1.0 - ROW_LABEL_W - 0.004) / (1.0 + bem_span_x / span_x)
        col_w = [ours_w * bem_span_x / span_x, ours_w]
        pie_h = ours_w * COLUMN_WIDTH_IN * span_y / span_x
        fig_h = titles_h + n_rows * pie_h + (n_rows - 1) * gap_h + legend_h
        fig = plt.figure(figsize=(COLUMN_WIDTH_IN, fig_h))
        x0 = [ROW_LABEL_W, ROW_LABEL_W + col_w[0]]
        # Header boxes centre on the pies, not on the axes, and share one width.
        pie_cx = [x0[c] + col_w[c] * (0.0 - lim[0]) / (lim[1] - lim[0])
                  for c, lim in enumerate((BEM_XLIM, PIE_XLIM))]
        head_w = min(col_w[0], 2.0 * (1.0 - pie_cx[1])) - 0.02

        # Column headers: a tinted rounded box per solver, spanning its column.
        head_y = (fig_h - titles_h + 0.035) / fig_h
        head_hh = (titles_h - 0.065) / fig_h
        for c, (pkey, ptitle) in enumerate(panels):
            fig.patches.append(FancyBboxPatch(
                (pie_cx[c] - head_w / 2.0, head_y), head_w, head_hh,
                boxstyle="round,pad=0,rounding_size=0.012",
                transform=fig.transFigure, facecolor=HEADER_FILL[theme][pkey],
                edgecolor="none", zorder=0))
            fig.text(pie_cx[c], head_y + head_hh / 2.0, ptitle,
                     fontsize=TITLE_FONTSIZE, fontweight="bold", color=ink,
                     ha="center", va="center")

        rows, row_mid = [], []
        y = fig_h - titles_h
        for _r in range(n_rows):
            y -= pie_h
            # The pie's centre (data y = 0) inside this band, for the row label.
            row_mid.append((y + pie_h * (0.0 - PIE_YLIM[0]) / span_y) / fig_h)
            rows.append([fig.add_axes([x0[c], y / fig_h,
                                       col_w[c], pie_h / fig_h])
                         for c in range(2)])
            y -= gap_h
    else:
        fig = plt.figure(figsize=(COLUMN_WIDTH_IN, 2.34))
        axes_grid = fig.subplots(1, 2)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.885, bottom=0.355,
                            wspace=0.02)
        rows = [axes_grid]
    for r, ((scene_label, variants), row_axes) in enumerate(zip(scenes, rows)):
        for ax, (key, title) in zip(row_axes, panels):
            costs = variants[key]
            values = [costs[k] for k, _n, _c in STAGES]
            colors = [c for _k, _n, c in STAGES]
            total = sum(values)

            def autopct(pct: float) -> str:
                return f"{pct:.0f}%" if pct >= PCT_LABEL_MIN else ""

            ax.pie(
                values, colors=colors, startangle=90, counterclock=False,
                autopct=None if multi else autopct, pctdistance=0.62,
                radius=1.0,
                wedgeprops={"linewidth": 0.4, "edgecolor": "white"},
                textprops={"fontsize": PCT_FONTSIZE},
            )
            if not multi:
                for txt in ax.texts:
                    txt.set_color("white")
                    txt.set_fontweight("bold")
                # pie() leaves a ~10% margin around the unit circle, which at
                # this size is a visible band of dead space above the legend.
                ax.set_xlim(-1.04, 1.04)
                ax.set_ylim(-1.04, 1.04)
            else:
                # Multi-scene: the legend can no longer carry absolute costs, so
                # each wedge is labelled with its own share and duration. Wedges
                # too thin to hold two lines get an outside label on a leader.
                # In the BEM panel only the frequency wedge is labelled: every
                # other stage costs exactly the same as in the panel beside it,
                # where those wedges are large enough to read.
                _label_wedges(ax, values, total,
                              only=("freq",) if key == "bem" else None)
                ax.set_xlim(*(BEM_XLIM if key == "bem" else PIE_XLIM))
                ax.set_ylim(*PIE_YLIM)
                ax.set_aspect("equal")
                # Each pie carries its own total underneath, bold: it is the
                # headline number for that pie. The speedup is in the caption.
                ax.text(0.0, TOTAL_Y, fmt_duration(total), ha="center",
                        va="top", fontsize=TOTAL_FONTSIZE, fontweight="bold",
                        color=ink)
            if not multi:
                ax.set_title(f"{title}\n{fmt_duration(total)}",
                             fontsize=TITLE_FONTSIZE, pad=2.0)

        if multi:
            # Scene name in the left strip, centred on its pies. A two-word
            # name wraps so the strip can stay narrow.
            name = scene_label.replace(" ", "\n") if " " in scene_label else scene_label
            fig.text(0.012, row_mid[r], name, fontsize=SCENE_FONTSIZE,
                     fontweight="bold", color=ink, ha="left", va="center",
                     linespacing=0.95)

    handles, labels = [], []
    for key, name, color in STAGES:
        if multi:
            text = name
        elif key == "freq":
            # Only this stage differs between the two solvers, so it is the one
            # legend row that needs two numbers.
            text = (f"{name} — {fmt_duration(scenes[0][1]['bem'][key])} (BEM) "
                    f"→ {fmt_duration(scenes[0][1]['ours'][key])} (ours)")
        else:
            text = f"{name} — {fmt_duration(scenes[0][1]['bem'][key])}"
        handles.append(Patch(facecolor=color, edgecolor="none"))
        labels.append(text)
    if multi:
        # Framed three-column legend spanning the figure. Columns fill
        # top-to-bottom, so STAGES order gives (LBM, tracking), (dump, mesh),
        # (frequency, audio).
        leg = fig.legend(handles, labels, loc="lower center",
                         bbox_to_anchor=(0.5, 0.004), ncols=3, frameon=True,
                         fancybox=True, fontsize=LEGEND_FONTSIZE,
                         handlelength=0.9, handleheight=0.9, handletextpad=0.4,
                         labelspacing=0.35, columnspacing=0.7,
                         borderpad=0.45, borderaxespad=0.15)
        frame = leg.get_frame()
        frame.set_facecolor(LEGEND_FILL[theme])
        frame.set_edgecolor(LEGEND_EDGE[theme])
        frame.set_linewidth(0.5)
        for t in leg.get_texts():
            t.set_color(ink)
    else:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncols=1, frameon=False, fontsize=LEGEND_FONTSIZE,
                   handlelength=1.1, handleheight=0.9, handletextpad=0.5,
                   labelspacing=0.4, columnspacing=1.0, borderaxespad=0.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The wedge colours carry the meaning and read on either ground; only the
    # lettering and the paper have to change for a dark page.
    fig.savefig(out_path, dpi=dpi,
                facecolor=("#151515" if dark else "white"))
    plt.close(fig)


def report(variants: dict[str, dict[str, float]]) -> None:
    """Print every number the paper quotes, so prose can be restated from here."""
    totals = {k: sum(v.values()) for k, v in variants.items()}
    for key, title in (("bem", "BEM reference"), ("ours", "Our surrogate")):
        costs, total = variants[key], totals[key]
        print(f"\n{title}: total {total:,.1f} s = {fmt_duration(total)}")
        for stage, name, _c in STAGES:
            print(f"  {name:<22s} {costs[stage]:>10,.1f} s  "
                  f"{fmt_duration(costs[stage]):>9s}  "
                  f"{costs[stage] / total * 100.0:6.2f}%")
    print(f"\nend-to-end speedup: {totals['bem'] / totals['ours']:.1f}x")
    ours = variants["ours"]
    neck = max((s for s in STAGES if s[0] != "freq"), key=lambda s: ours[s[0]])
    print(f"bottleneck with the surrogate: {neck[1]} "
          f"({ours[neck[0]] / totals['ours'] * 100.0:.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="Output image path.")
    ap.add_argument("--timings-json", type=Path, default=None,
                    help=f"One scene's timings JSON, in seconds; must carry all "
                         f"of {', '.join(TIMING_KEYS)}. Per-stage flags override "
                         f"individual entries.")
    for key in ("lbm", "tracking", "dump", "mesh", "freq-bem", "freq-ours", "audio"):
        ap.add_argument(f"--{key}-s", type=float, default=None,
                        dest=key.replace("-", "_"),
                        help=f"Override the {key} stage cost, in seconds; "
                             f"the rest come from --timings-json, or from "
                             f"{DEFAULT_SCENES[0][1].name} if that is not given.")
    ap.add_argument("--scene", action="append", default=None,
                    metavar="LABEL=FILE.json",
                    help="A scene to draw as its own row of pies, from its own "
                         "timings JSON. Repeat for several scenes; they stack "
                         "in the order given and share one legend. Mutually "
                         "exclusive with --timings-json / the per-stage flags. "
                         "With none of the three, the paper's two scenes are "
                         "drawn from the JSONs shipped in results/.")
    ap.add_argument("--ours-label", default="Our surrogate")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--dark", action="store_true",
                    help="Light lettering on a dark ground, for a dark-themed page.")
    args = ap.parse_args()

    def load(path: Path) -> dict[str, float]:
        if not path.is_file():
            raise SystemExit(f"no such timings JSON: {path}")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        unknown = set(loaded) - set(TIMING_KEYS)
        if unknown:
            raise SystemExit(f"{path}: unknown timing keys: {sorted(unknown)}")
        # Every stage must be present: a silently-defaulted stage is a wedge of
        # the pie that came from somewhere other than this run's ledger.
        missing = set(TIMING_KEYS) - set(loaded)
        if missing:
            raise SystemExit(f"{path}: missing timing keys: {sorted(missing)}")
        return {k: float(loaded[k]) for k in TIMING_KEYS}

    overrides = {k: v for k in TIMING_KEYS
                 if (v := getattr(args, k, None)) is not None}

    if args.scene:
        if args.timings_json or overrides:
            raise SystemExit("--scene cannot be combined with --timings-json "
                             "or the per-stage flags: each scene carries its "
                             "own timings file.")
        scenes = []
        for spec in args.scene:
            label, _sep, path = spec.partition("=")
            if not _sep:
                raise SystemExit(f"--scene expects LABEL=FILE.json, got {spec!r}")
            scenes.append((label, build_variants(load(Path(path)))))
    elif args.timings_json or overrides:
        costs = load(args.timings_json or DEFAULT_SCENES[0][1])
        costs.update(overrides)
        scenes = [("", build_variants(costs))]
    else:
        scenes = [(label, build_variants(load(path)))
                  for label, path in DEFAULT_SCENES]

    plot(args.out.resolve(), scenes, ours_label=args.ours_label, dpi=args.dpi,
         dark=args.dark)
    for label, variants in scenes:
        if label:
            print(f"\n===== {label} =====")
        report(variants)
    print(f"\nwrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

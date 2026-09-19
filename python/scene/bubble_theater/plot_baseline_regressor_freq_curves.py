"""Overlay non-neural baseline regressors on the procedural single-bubble tests.

Classical regressors (linear / polynomial / RBF kernel ridge) fitted on the same
8D descriptors as the MLP are applied to the procedural OBJ sequences
(``ellipsoid``, ``curl_noise``, ``enright_test``) and drawn against Minnaert, the
ellipsoid (Strasberg) baseline, the learning model (NN8) and BEM. This is the
paper's regressor-comparison figure.

Per scene ``<name>``: frames ``dataset/bubble_theater/<name>/mesh/bub_{k:04d}.obj``
(k = 1..N) are normalized to unit volume, the same eight features the NN8
pipeline extracts (``_process_frame`` in ``render_bubble_theater.py``) are fed to
each baseline, and the predicted ``log f_unit`` is exponentiated and rescaled by
``f_real = f_unit * R_unit / R_eq``. The Minnaert / Strasberg / NN8 / BEM curves
are copied verbatim from the scene's existing ``freq_curves.csv`` -- nothing is
re-solved. Baselines are re-fitted on the paper's train split (seed 42, asserted
against ``split.json``) with the validation-selected hyperparameters.

REQUIRED INPUTS (none of the result-side files ship with this repo; generate
them before running, or point the flags at your own copies):

* ``<--results-root>/<name>_result/freq_curves.csv`` and
  ``trackedBubInfo_Minnaert.txt`` -- written by
  ``render_bubble_theater.py`` for that scene; the CSV's row count also sets N.
  Default results root: ``results/bubble_theater/``.
* ``results/experiments/table02_regressor_comparison/baseline_metrics.json`` --
  the hyperparameters chosen by
  ``python/freq_model/regression/baseline_regressors_8feat.py``
  (``--baseline-metrics`` to override).
* the training dataset CSV and its ``split.json`` (``--dataset``/``--split-json``).

Writes ``freq_curves_baselines.{csv,png}`` next to each scene's inputs, plus a
console MAPE-vs-BEM table. ``--stacked`` only re-reads those CSVs and stacks the
scenes into one column-width figure; it fits nothing, reads no meshes, and needs
only numpy + matplotlib.

RUN (from repo root, conda env ``bubblegym``):
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    python -u python/scene/bubble_theater/plot_baseline_regressor_freq_curves.py

RE-PLOT the stacked paper figure from an existing run:
    python -u python/scene/bubble_theater/plot_baseline_regressor_freq_curves.py \
        --stacked --results-root <dir holding the *_result folders> \
        --stacked-out <path>/baseline-regressors-vertical.png
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:  # torch before scipy on Windows; absent in a plotting-only env (--stacked)
    import torch as _TORCH  # noqa: E402,F401
except ImportError:  # pragma: no cover - only reached on the --stacked path
    _TORCH = None

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_THIS = Path(__file__).resolve()
PYTHON_ROOT = _THIS.parents[2]
REPO_ROOT = PYTHON_ROOT.parent
FREQ_MODEL_ROOT = PYTHON_ROOT / "freq_model"
for _p in (PYTHON_ROOT, FREQ_MODEL_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# baseline_regressors_8feat stubs torch.utils.tensorboard before importing the
# trainer, so this import chain works in the tensorboard-less bubblegym env.
#
# Only the fitting path needs any of this. --stacked just re-plots the
# freq_curves_baselines.csv files an earlier run already wrote, so keep it
# usable from an env that has nothing but numpy and matplotlib; fit_baselines()
# re-raises below if the fitting path is actually taken.
_FIT_DEPS_ERROR: ImportError | None = None
try:
    from freq_model.regression.baseline_regressors_8feat import (  # noqa: E402
        DEFAULT_DATASET,
        check_split_against_paper,
        load_xy,
    )
    from freq_model.NN.bubble_dataset import split_dataset  # noqa: E402
    from freq_model.NN.fit_shape_freq_model import (  # noqa: E402
        FEATURE_COLS,
    )
    from scene.bubble_theater.render_bubble_theater import (  # noqa: E402
        _process_frame,
    )

    from sklearn.kernel_ridge import KernelRidge  # noqa: E402
    from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
    from sklearn.pipeline import make_pipeline  # noqa: E402
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402
except ImportError as _exc:  # pragma: no cover - only reached on --stacked
    _FIT_DEPS_ERROR = _exc
    DEFAULT_DATASET = None

BASELINE_METRICS_JSON = (
    REPO_ROOT / "results" / "experiments" / "table02_regressor_comparison" / "baseline_metrics.json"
)
SCENES_DEFAULT = ("ellipsoid", "curl_noise", "enright_test")
PROCEDURAL_ROOT = REPO_ROOT / "dataset" / "bubble_theater"
# Scene OBJs sit in a "mesh" subfolder, and the rendered curves live under
# results/ -- the shipped layout keeps inputs in dataset/ and outputs in
# results/, so the two are not siblings.
MESH_SUBDIR = "mesh"
RESULTS_ROOT = REPO_ROOT / "results" / "experiments" / "fig01_bubble_theater"

# Equivalent-sphere radius of a unit-volume bubble; f_real = f_unit * R_UNIT/R_eq
R_UNIT = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

# Existing paper-figure styles (matches plot_overlay_from_tracked.py) plus the
# four baseline regressors. tab10 orange<->green is CVD-weak, so every added
# series carries a distinct dash pattern as a secondary encoding.
PLOT_SERIES: list[tuple[str, str, dict]] = [
    # (csv column, legend label, matplotlib style)
    ("f_minnaert", "Minnaert", {"color": "#7F7F7F", "linestyle": "--", "linewidth": 1.6}),
    ("f_strasberg", "Ellipsoid", {"color": "#1F77B4", "linestyle": "-", "linewidth": 1.6}),
    ("f_nn8", "Learning model", {"color": "#FF7F0E", "linestyle": "-", "linewidth": 1.8}),
    ("f_bem", "BEM", {"color": "#2CA02C", "linestyle": "-", "linewidth": 1.6}),
    ("f_linear", "Linear", {"color": "#E377C2", "linestyle": (0, (4, 2)), "linewidth": 1.6}),
    ("f_poly2", "Poly (deg 2)", {"color": "#9467BD", "linestyle": (0, (1, 1)), "linewidth": 1.8}),
    ("f_poly3", "Poly (deg 3)", {"color": "#D62728", "linestyle": (0, (5, 1, 1, 1)), "linewidth": 1.6}),
    ("f_rbf", "RBF kernel ridge", {"color": "#17BECF", "linestyle": (0, (2, 1)), "linewidth": 1.6}),
]


# ---------------------------------------------------------------------------
# Fit the baselines exactly as baseline_regressors_8feat.py selected them
# ---------------------------------------------------------------------------


def fit_baselines(
    dataset: Path = DEFAULT_DATASET,
    metrics_json: Path = BASELINE_METRICS_JSON,
    split_json: Path | None = None,
) -> tuple[StandardScaler, dict[str, object]]:
    """Re-fit the four baselines on the paper's train split.

    Hyperparameters are the validation-selected values recorded in
    ``baseline_metrics.json`` (no re-tuning here). ``split_json`` overrides the
    trained-model split the regenerated partition is asserted against (needed
    when ``dataset`` is not the paper's 9k CSV).
    """
    if _FIT_DEPS_ERROR is not None:
        raise SystemExit(
            "fitting the baselines needs torch, scikit-learn and the freq_model "
            f"package, which failed to import: {_FIT_DEPS_ERROR}"
        )
    with metrics_json.open("r", encoding="utf-8") as f:
        chosen = json.load(f)["models"]

    x_raw, y_raw, log_fs_raw, df_used = load_xy(dataset)
    split = split_dataset(
        x_raw, y_raw, log_fs_raw, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    if split_json is None:
        n_test = check_split_against_paper(df_used, split)
    else:
        n_test = check_split_against_paper(df_used, split, split_json)
    print(f"split matches split.json (n_test={n_test}); fitting baselines...")

    x_scaler = StandardScaler().fit(split.x_train)
    x_train = x_scaler.transform(split.x_train)
    y_train = split.y_train.astype(np.float64)

    def _poly(degree: int) -> object:
        alpha = float(chosen[f"poly{degree}"]["hparams"]["ridge_alpha"])
        reg = LinearRegression() if alpha == 0.0 else Ridge(alpha=alpha)
        return make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), reg)

    models: dict[str, object] = {
        "linear": LinearRegression(),
        "poly2": _poly(2),
        "poly3": _poly(3),
        "rbf": KernelRidge(
            kernel="rbf",
            alpha=float(chosen["rbf"]["hparams"]["alpha"]),
            gamma=float(chosen["rbf"]["hparams"]["gamma"]),
        ),
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        print(f"  fitted {name}")
    return x_scaler, models


# ---------------------------------------------------------------------------
# Per-scene feature extraction + prediction
# ---------------------------------------------------------------------------


def read_freq_curves_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as h:
        reader = csv.reader(h)
        header = next(reader)
        rows = [[float(tok) for tok in row] for row in reader if row]
    arr = np.asarray(rows, dtype=np.float64)
    return header, {name: arr[:, j] for j, name in enumerate(header)}


def parse_radius_eq(tracked_minnaert: Path) -> float:
    """Read R_eq from the leading ``Bub <id> <R_eq>`` header line."""
    for raw in tracked_minnaert.read_text(encoding="utf-8").splitlines():
        tok = raw.strip().split()
        if len(tok) == 3 and tok[0] == "Bub":
            return float(tok[2])
    raise SystemExit(f"no 'Bub <id> <R_eq>' header in {tracked_minnaert}")


def extract_features_per_frame(mesh_dir: Path, n_frames: int) -> np.ndarray:
    """Return an (n_frames, 8) matrix in FEATURE_COLS order (NaN on failure)."""
    x = np.full((n_frames, len(FEATURE_COLS)), np.nan, dtype=np.float64)
    chull_cols = FEATURE_COLS[2:]  # after the two inertia ratios
    for k in range(n_frames):
        obj_path = mesh_dir / f"bub_{k + 1:04d}.obj"
        try:
            st = _process_frame(obj_path, 1.0)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] frame {k + 1}: {type(exc).__name__}: {exc}")
            continue
        feats = st["chull_features"]
        if feats.get("status") != "ok":
            print(f"  [warn] frame {k + 1}: {feats.get('status')}")
            continue
        principal = np.asarray(st["inertia_principal_unit"], dtype=np.float64)
        if principal.shape != (3,) or principal[0] <= 0.0:
            print(f"  [warn] frame {k + 1}: bad principal moments {principal}")
            continue
        x[k, 0] = principal[1] / principal[0]
        x[k, 1] = principal[2] / principal[0]
        for j, c in enumerate(chull_cols, start=2):
            x[k, j] = float(feats[c])
        if (k + 1) % 30 == 0:
            print(f"  features {k + 1}/{n_frames}")
    return x


def predict_scene(
    x_feats: np.ndarray,
    x_scaler: StandardScaler,
    models: dict[str, object],
    freq_scale: float,
) -> dict[str, np.ndarray]:
    valid = np.all(np.isfinite(x_feats), axis=1)
    xn = x_scaler.transform(x_feats[valid])
    out: dict[str, np.ndarray] = {}
    for name, model in models.items():
        f_real = np.full(x_feats.shape[0], np.nan, dtype=np.float64)
        f_real[valid] = np.exp(model.predict(xn)) * freq_scale
        out[f"f_{name}"] = f_real
    return out


# ---------------------------------------------------------------------------
# Plot (paper freq_curves.png style: 9x4.5, fontsize 15, upper-left legend)
# ---------------------------------------------------------------------------


def plot_overlay(png_path: Path, t: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for col, label, style in PLOT_SERIES:
        y = curves.get(col)
        if y is None:
            continue
        finite = np.isfinite(y)
        if not finite.any():
            continue
        ax.plot(t[finite], y[finite], label=label, **style)
    ax.set_xlabel("Time (s)", fontsize=15)
    ax.set_ylabel("Frequency (Hz)", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.legend(loc="upper left", frameon=False, fontsize=12, ncols=2)
    if t.size:
        ax.set_xlim(float(t[0]), float(t[-1]))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stacked single-column variant (the paper's regressor-comparison figure)
# ---------------------------------------------------------------------------

# \columnwidth in the paper's acmtog layout is 243.15pt. Drawing the figure at
# exactly that width and including it with width=\linewidth places it at 1:1,
# so a matplotlib fontsize lands as that many real points on the page. The old
# figure was three 9in-wide panels stitched together and then squeezed into the
# column, which is why its 15pt type ended up under 6pt in print.
COLUMN_WIDTH_IN = 243.15 / 72.0

STACKED_PANEL_LABELS = {
    "ellipsoid": "Ellipsoid",
    "curl_noise": "Curl noise",
    "enright_test": "Enright deformation",
}

# Type sizes chosen against the 8pt acmtog caption: axis labels match it, ticks
# and legend sit just under. Line widths are scaled down from the single-panel
# figure, whose 1.6-1.8pt strokes go muddy at a third of the width.
STACKED_LABEL_FONTSIZE = 8.0
STACKED_TICK_FONTSIZE = 6.5
STACKED_LEGEND_FONTSIZE = 6.5
STACKED_TITLE_FONTSIZE = 7.0
STACKED_LINEWIDTH_SCALE = 0.55


def plot_stacked(
    out_path: Path,
    panels: list[tuple[str, np.ndarray, dict[str, np.ndarray]]],
    width_in: float = COLUMN_WIDTH_IN,
    panel_height_in: float = 1.12,
    dpi: int = 600,
) -> None:
    """Draw every scene as one column-width figure, one row per scene.

    ``panels`` is ``[(scene, t, curves), ...]`` in top-to-bottom order. The
    series legend is drawn once for the whole figure rather than repeated in
    every panel, and the axis labels are shared, which is most of the vertical
    space the stitched version wasted.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    n = len(panels)
    if n == 0:
        raise SystemExit("--stacked needs at least one scene")

    fig, axes = plt.subplots(
        n, 1, figsize=(width_in, panel_height_in * n + 0.85), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    handles: list[object] = []
    labels: list[str] = []
    for ax, (scene, t, curves) in zip(axes, panels):
        for col, label, style in PLOT_SERIES:
            y = curves.get(col)
            if y is None:
                continue
            finite = np.isfinite(y)
            if not finite.any():
                continue
            thin = dict(style)
            thin["linewidth"] = float(thin.get("linewidth", 1.6)) * STACKED_LINEWIDTH_SCALE
            (line,) = ax.plot(t[finite], y[finite], label=label, **thin)
            if label not in labels:
                handles.append(line)
                labels.append(label)
        ax.set_title(
            STACKED_PANEL_LABELS.get(scene, scene),
            fontsize=STACKED_TITLE_FONTSIZE,
            loc="left",
            pad=2.0,
        )
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(
            axis="both", which="major", labelsize=STACKED_TICK_FONTSIZE,
            length=2.0, width=0.6, pad=1.5,
        )
        # matplotlib's default tick density is set for a 6in-wide axes and
        # collides with itself at a third of that, so thin it out.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        if t.size:
            ax.set_xlim(float(t[0]), float(t[-1]))

    # Scene time ranges differ, so every panel keeps its own x ticks; only the
    # labels are shared.
    fig.supxlabel("Time (s)", fontsize=STACKED_LABEL_FONTSIZE)
    fig.supylabel("Frequency (Hz)", fontsize=STACKED_LABEL_FONTSIZE)
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=3,
        frameon=False,
        fontsize=STACKED_LEGEND_FONTSIZE,
        handlelength=2.4,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def mape_vs_bem(curves: dict[str, np.ndarray]) -> dict[str, float]:
    f_bem = curves["f_bem"]
    out: dict[str, float] = {}
    for col, _label, _style in PLOT_SERIES:
        if col == "f_bem" or col not in curves:
            continue
        y = curves[col]
        finite = np.isfinite(y) & np.isfinite(f_bem)
        if not finite.any():
            continue
        ape = np.abs(y[finite] - f_bem[finite]) / f_bem[finite] * 100.0
        out[col] = float(np.mean(ape))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scenes",
        type=str,
        default=",".join(SCENES_DEFAULT),
        help=f"Comma-separated scene names under {PROCEDURAL_ROOT} "
        f"(default: {','.join(SCENES_DEFAULT)})",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Training CSV the baselines are re-fitted on.")
    parser.add_argument("--baseline-metrics", type=Path, default=BASELINE_METRICS_JSON,
                        help="baseline_metrics.json with the selected hyperparameters.")
    parser.add_argument("--split-json", type=Path, default=None,
                        help="split.json to assert the regenerated split against "
                             "(defaults to the paper's; set when --dataset differs).")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT,
                        help=f"Directory holding the <scene>_result folders "
                             f"(default: {RESULTS_ROOT}).")
    parser.add_argument("--stacked", action="store_true",
                        help="Re-plot only: read each scene's existing "
                             "freq_curves_baselines.csv and draw all scenes as one "
                             "column-width figure, one row per scene, with a single "
                             "shared legend. Fits nothing and needs no meshes.")
    parser.add_argument("--stacked-out", type=Path, default=None,
                        help="Output path for --stacked (default: "
                             "<results-root>/baseline-regressors-vertical.png).")
    parser.add_argument("--stacked-width-in", type=float, default=COLUMN_WIDTH_IN,
                        help=f"Figure width in inches for --stacked; match the "
                             f"target \\columnwidth so type lands at its nominal "
                             f"point size (default: {COLUMN_WIDTH_IN:.4f}).")
    parser.add_argument("--stacked-panel-height-in", type=float, default=1.12,
                        help="Height in inches of each panel for --stacked "
                             "(default: 1.12).")
    parser.add_argument("--stacked-dpi", type=int, default=600,
                        help="Raster resolution for --stacked (default: 600).")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    results_root = args.results_root.resolve()

    if args.stacked:
        panels: list[tuple[str, np.ndarray, dict[str, np.ndarray]]] = []
        for scene in scenes:
            csv_path = results_root / f"{scene}_result" / "freq_curves_baselines.csv"
            if not csv_path.is_file():
                raise SystemExit(
                    f"[{scene}] missing {csv_path}\n"
                    "--stacked re-plots existing runs; run without it first."
                )
            _header, curves = read_freq_curves_csv(csv_path)
            panels.append((scene, curves["t"], curves))
            print(f"  read {csv_path} ({curves['t'].size} frames)")
        out_path = args.stacked_out or (results_root / "baseline-regressors-vertical.png")
        plot_stacked(
            out_path.resolve(),
            panels,
            width_in=args.stacked_width_in,
            panel_height_in=args.stacked_panel_height_in,
            dpi=args.stacked_dpi,
        )
        print(f"  wrote {out_path.resolve()}")
        return 0

    dataset = args.dataset or DEFAULT_DATASET
    if dataset is None:  # freq_model import failed, so DEFAULT_DATASET is unknown
        raise SystemExit(
            "cannot locate the training CSV: the freq_model package failed to "
            f"import ({_FIT_DEPS_ERROR}). Pass --dataset explicitly, or use "
            "--stacked to re-plot existing runs."
        )
    x_scaler, models = fit_baselines(
        dataset.resolve(), args.baseline_metrics.resolve(),
        args.split_json.resolve() if args.split_json else None,
    )

    for scene in scenes:
        mesh_dir = PROCEDURAL_ROOT / scene / MESH_SUBDIR
        result_dir = results_root / f"{scene}_result"
        curves_csv = result_dir / "freq_curves.csv"
        tracked_minnaert = result_dir / "trackedBubInfo_Minnaert.txt"
        for p in (mesh_dir, result_dir):
            if not p.is_dir():
                raise SystemExit(f"[{scene}] missing directory: {p}")
        for p in (curves_csv, tracked_minnaert):
            if not p.is_file():
                raise SystemExit(f"[{scene}] missing file: {p}")

        print(f"\n=== {scene} ===")
        _header, curves = read_freq_curves_csv(curves_csv)
        t = curves["t"]
        n_frames = t.size
        radius_eq = parse_radius_eq(tracked_minnaert)
        freq_scale = R_UNIT / radius_eq
        print(
            f"  frames={n_frames}  R_eq={radius_eq:.6f} m  "
            f"freq scale (unit->real) = {freq_scale:.4f}"
        )

        x_feats = extract_features_per_frame(mesh_dir, n_frames)
        curves.update(predict_scene(x_feats, x_scaler, models, freq_scale))

        out_csv = result_dir / "freq_curves_baselines.csv"
        cols = ["k", "t"] + [c for c, _l, _s in PLOT_SERIES if c in curves]
        with out_csv.open("w", newline="", encoding="utf-8") as h:
            writer = csv.writer(h)
            writer.writerow(cols)
            for k in range(n_frames):
                writer.writerow(
                    [int(curves["k"][k])] + [float(curves[c][k]) for c in cols[1:]]
                )
        out_png = result_dir / "freq_curves_baselines.png"
        plot_overlay(out_png, t, curves)
        print(f"  wrote {out_png}\n  wrote {out_csv}")

        print(f"  MAPE vs BEM over {n_frames} frames:")
        for col, m in mape_vs_bem(curves).items():
            print(f"    {col:<12s} {m:8.4f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())

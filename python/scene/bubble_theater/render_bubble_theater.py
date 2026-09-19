"""Per-frame resonance frequencies for a procedural single-bubble OBJ sequence.

Reads ``--frame-count`` meshes at ``{input_prefix}_{F:04d}.obj`` (``F`` from
``--frame-start``), normalizes each to unit volume and then to the common
``V_target = 1 / shrink_radius_scale**3``, so only the *shape* of each frame
affects the result. For every frame it evaluates the methods named by
``--methods``: ``Minnaert`` (spherical baseline, constant here), ``Ellipsoid``
(the moment-matching ellipsoidal proxy, from the volume-normalized inertia
tensor; ``strasberg`` is accepted as an alias), ``NN8`` (the
8-feature chull+inertia surrogate) and ``BEM`` (Galerkin P1-DP0 capacitance
solve -- seconds per frame, so it is opt-in).

Output lands in ``<--output-dir>/<input-parent-name>_result/``: one
single-bubble ``trackedBubInfo_<method>.txt`` on the uniform time grid
``t[k] = k * duration / (N - 1)`` with position pinned at the origin, plus
``freq_curves.csv`` and the overlay ``freq_curves.png``. Audio rendering is not
part of this repo. ``--replot-only`` redraws the CSV/PNG from trackedBubInfo
files already on disk.

Inputs are assumed to be clean closed manifolds with outward winding: no
component filter, winding flip or Laplacian smoothing is applied.

    python python/scene/bubble_theater/render_bubble_theater.py \\
        dataset/bubble_theater/curl_noise/mesh \\
        --shrink-radius-scale 48.5 --duration 6 --frame-count 240 \\
        --methods minnaert,strasberg,nn8,bem
"""

from __future__ import annotations

# IMPORTANT: torch must be imported BEFORE scipy / igl etc. on Windows + the
# soundlab conda env. Mirrors the workaround in
# ``python/scene/_common/write_trackedbubinfo_nn.py``.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch as _TORCH  # type: ignore  # noqa: E402,F401

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

PYTHON_ROOT = Path(__file__).resolve().parents[2]
FREQ_MODEL_ROOT = PYTHON_ROOT / "freq_model"
BASELINE_ROOT = PYTHON_ROOT / "baseline"
for _p in (PYTHON_ROOT, FREQ_MODEL_ROOT, BASELINE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from freq_model.analytical.minnaert_freq import MINNAERT_CONSTANT  # noqa: E402
from freq_model.analytical.strasberg_freq import strasberg_frequency_unit_volume  # noqa: E402
from freq_model.NN.nn_inference import (  # noqa: E402
    predict_nn_unit_for_frames_inertia_8feat_direct,
)
from shape_feature.nonspherical_features import (  # noqa: E402
    nonspherical_features_from_unit_mesh,
)
from shape_feature.nonspherical_features import (  # noqa: E402
    convex_hull_features_from_vf,
)
from shape_feature.mesh_utils import load_obj_mesh  # noqa: E402
from shape_feature.nonspherical_features import (  # noqa: E402
    curvature_integrals_from_vf,
)
from shape_feature.mesh_utils import (  # noqa: E402
    compute_mirtich_moments,
    mesh_signed_volume,
    vertices_scaled_to_target_volume,
)


REPO_ROOT = PYTHON_ROOT.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset" / "bubble_theater"
# 8-feature direct chull+inertia NN: the six chull features plus the two
# volume-normalized principal-moment ratios (I11/I00, I22/I00). On the held-out
# test set the inertia ratios lift MAPE from 0.25 % to ~0.09 % over the
# 6-feature chull-only model, which is why they are the production head.
DEFAULT_ARTIFACTS_NN8 = (
    FREQ_MODEL_ROOT / "output" / "output_8feature_direct_bubblegym_10k"
)

# "Ellipsoid" is the moment-matching ellipsoidal proxy the paper describes; it
# is implemented by analytical/strasberg_freq.py, so "strasberg" stays an
# accepted alias for anyone following the older naming.
METHOD_TAGS = ("Minnaert", "Ellipsoid", "NN8", "BEM")
METHOD_ALIASES = {
    "minnaert": "Minnaert",
    "ellipsoid": "Ellipsoid",
    "strasberg": "Ellipsoid",
    "nn8": "NN8",
    "nn_8": "NN8",
    "nn-8": "NN8",
    "nn8feature": "NN8",
    "nn_8feature": "NN8",
    "nn-8feature": "NN8",
    "nn_8feat": "NN8",
    "nn-8feat": "NN8",
    "bem": "BEM",
}


# ---------------------------------------------------------------------------
# Per-frame mesh feature extraction
# ---------------------------------------------------------------------------


def _load_signed_volume(obj_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load OBJ and return (v_raw, f, signed_volume) with cleanup checks."""
    v_raw, f = load_obj_mesh(obj_path)
    signed_vol_raw = mesh_signed_volume(v_raw, f)
    if not math.isfinite(signed_vol_raw) or signed_vol_raw == 0.0:
        raise ValueError(
            f"Degenerate mesh (signed_volume={signed_vol_raw!r}): {obj_path}"
        )
    if signed_vol_raw < 0.0:
        # Surface inside-out: Strasberg / Mirtich need positive principal
        # moments. Procedural meshes are assumed pre-cleaned, so we abort
        # rather than silently flipping.
        raise ValueError(
            f"Mesh has negative signed volume ({signed_vol_raw:.6e}); "
            f"input is inside-out. Fix the winding upstream: {obj_path}"
        )
    return v_raw, f, float(signed_vol_raw)


def _process_frame(
    obj_path: Path, volume_target: float
) -> dict[str, Any]:
    """Load one OBJ, rescale to ``volume_target`` (m^3), return frame state.

    Returns a dict with:

    * ``signed_volume_raw`` -- pre-rescale signed volume (sanity readout).
    * ``v_target``, ``f`` -- vertices rescaled to ``volume_target`` m^3.
    * ``v_unit`` -- the same mesh rescaled to V=1 (used for NN features
      and as the source of the volume-normalized inertia for Strasberg).
    * ``inertia_principal_unit`` -- principal moments of the unit-volume
      mesh, sorted ascending.
    * ``chull_features`` -- dict from ``nonspherical_features_from_unit_mesh``.
    """
    v_raw, f, signed_vol_raw = _load_signed_volume(obj_path)

    v_target = vertices_scaled_to_target_volume(
        v_raw, f, target_volume=float(volume_target)
    )
    v_unit = vertices_scaled_to_target_volume(v_raw, f, target_volume=1.0)

    _mass_unit, _com_unit, inertia_unit = compute_mirtich_moments(v_unit, f)
    # ``compute_mirtich_moments`` returns the volumetric inertia at unit
    # density; for the unit-volume mesh that already coincides with the
    # volume-normalized inertia (V=1) Strasberg expects.
    principal_unit = np.linalg.eigvalsh(inertia_unit).astype(np.float64)
    principal_unit.sort()

    chull = nonspherical_features_from_unit_mesh(v_unit, f)

    return {
        "signed_volume_raw": float(signed_vol_raw),
        "v_target": v_target,
        "f": f,
        "v_unit": v_unit,
        "inertia_principal_unit": principal_unit,
        "chull_features": chull,
    }


# ---------------------------------------------------------------------------
# Frequency models
# ---------------------------------------------------------------------------


def _strasberg_freq_real(
    inertia_principal_unit: np.ndarray, volume_target: float
) -> float:
    """Strasberg frequency at ``volume_target`` (m^3) from unit-volume I.

    Mirrors the rescale convention from
    ``scene.armaDrop.write_armadrop_freq_variants.strasberg_freq_isolated``::

        f_real = f_unit * V_target ** (-1/3)
    """
    f_unit = float(strasberg_frequency_unit_volume(inertia_principal_unit))
    if not math.isfinite(f_unit) or f_unit <= 0.0:
        raise ValueError(
            f"Strasberg unit-volume frequency invalid: {f_unit} for "
            f"I_principal={inertia_principal_unit.tolist()}"
        )
    return f_unit * (max(volume_target, 1e-24) ** (-1.0 / 3.0))


def _nn8_predict_per_frame(
    chull_features_per_frame: list[dict[str, Any]],
    inertia_principal_unit_per_frame: list[np.ndarray],
    artifacts_nn8: Path,
    volume_target: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the 8-feature direct chull+inertia NN; return (f_real, mask).

    The eight inputs are the six chull non-sphericity features plus
    ``I11/I00`` and ``I22/I00`` derived from the unit-volume principal
    moments (sorted ascending so the dataset's ``i00, i11, i22``
    convention is preserved). The network predicts ``log(f_BEM)`` at
    unit volume directly (no Strasberg residual), so we just
    exponentiate and rescale to ``volume_target`` via
    ``f_real = f_unit * V_target ** (-1/3)``.
    """
    f_unit, valid = predict_nn_unit_for_frames_inertia_8feat_direct(
        chull_features_per_frame,
        inertia_principal_unit_per_frame,
        artifacts_nn8,
    )
    f_real = np.full(f_unit.shape, np.nan, dtype=np.float64)
    scale = max(float(volume_target), 1e-24) ** (-1.0 / 3.0)
    f_real[valid] = f_unit[valid] * scale
    return f_real, valid


def _load_times_and_freqs_from_tracked_bubinfo(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse per-row ``(time, frequency)`` columns from a trackedBubInfo file.

    Returns ``(times, freqs)`` as a pair of length-N float64 arrays
    (NaN for any sample row whose frequency token is non-numeric or
    literally ``"nan"``). Used by ``--replot-only`` to reuse the
    per-method trackedBubInfo files written by an earlier full run
    instead of recomputing mesh features + NN8 / BEM / Strasberg
    frequencies from scratch.

    Format: see :func:`_write_tracked_bubinfo` -- ``Bub`` / ``Start:`` /
    ``End:`` lines are skipped, and each remaining indented line is
    parsed as ``<t> <f> 0 0 0 <p>``.
    """
    times: list[float] = []
    freqs: list[float] = []
    with path.open("r", encoding="utf-8") as h:
        for raw in h:
            line = raw.strip()
            if not line:
                continue
            tok = line.split()
            head = tok[0]
            if head in ("Bub", "Start:", "End:"):
                continue
            if head in ("N", "C", "S", "M") and len(tok) <= 2:
                continue
            if len(tok) < 2:
                continue
            try:
                t_val = float(tok[0])
            except ValueError:
                continue
            try:
                f_val = float(tok[1])
            except ValueError:
                f_val = float("nan")
            times.append(t_val)
            freqs.append(f_val)
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(freqs, dtype=np.float64),
    )


def _load_freqs_from_tracked_bubinfo(path: Path) -> np.ndarray:
    """Parse the per-frame frequency column from a ``trackedBubInfo_<tag>.txt``.

    Returns a length-N float64 array (NaN for any sample row whose
    frequency token is non-numeric or literally ``"nan"``). Used by
    ``--bem-reuse-tracked`` to skip the multi-minute BEM solve and just
    re-emit the curves on a new ``--duration`` time grid.

    Format produced by :func:`_write_tracked_bubinfo` (the only producer
    in this script):

    .. code-block:: text

        Bub <id> <radius_eq>
          Start: N <t0>
          <t0> <f0> 0 0 0 <p>
          <t1> <f1> 0 0 0 <p>
          ...
          End: C <tN>

    so we skip ``Bub`` / ``Start:`` / ``End:`` lines and read the second
    whitespace-token of every other indented line.
    """
    _, freqs = _load_times_and_freqs_from_tracked_bubinfo(path)
    return freqs


def _bem_predict_per_frame(
    frame_states: list[dict[str, Any]],
    *,
    volume_target: float,
    times: np.ndarray,
    radius_eq: float,
    trial_pair: str = "P1-DP0",
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 5,
    stride: int = 1,
    quadrature_regular: int = 6,
    quadrature_singular: int = 6,
    gmres_tol: float = 1e-12,
) -> np.ndarray:
    """Galerkin BEM frequency per frame, at ``volume_target``.

    ``solve_minnaert_frequency_galerkin`` runs once per frame on the
    unit-volume mesh from :func:`_process_frame`, and its unit-volume result is
    rescaled the same way Strasberg / NN8 are: ``f_unit * V_target ** (-1/3)``.

    Only frames with ``k % stride == 0``, plus the final frame (anchored so the
    tail needs no extrapolation), are actually solved; the rest are filled in at
    the end by ``np.interp`` through the solved frames.

    With ``checkpoint_path``, the whole accumulated trackedBubInfo is rewritten
    every ``checkpoint_every`` newly-solved frames and once at the end, with
    unsolved frames as NaN rows -- so an interrupted sweep still leaves every
    result it got. Frames the solver failed on, and that interpolation could not
    bridge, stay NaN in the returned length-N array.
    """
    n = len(frame_states)
    f_real = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return f_real
    if times.shape[0] != n:
        raise ValueError(
            f"_bem_predict_per_frame: len(frame_states)={n} but times has "
            f"shape {times.shape}"
        )
    stride = max(1, int(stride))

    # Build the explicit solve schedule: every multiple of stride plus
    # the last frame so endpoints are anchored. A sorted set keeps the
    # iteration order monotone and dedupes the n-1 anchor when stride
    # already divides n-1.
    solve_indices: list[int] = sorted(
        set(range(0, n, stride)) | {n - 1}
    )
    n_solve = len(solve_indices)

    # Lazy import: bempp / numba is heavy; only pay the cost when BEM is
    # actually requested. ``PYTHON_ROOT`` is already on sys.path (added at
    # module import), so the package import below resolves without extra
    # path munging.
    from bem.compute_freq_bempp_galerkin import (  # noqa: PLC0415
        solve_minnaert_frequency_galerkin,
    )

    try:
        from tqdm import tqdm  # noqa: PLC0415  # type: ignore
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    scale = max(float(volume_target), 1e-24) ** (-1.0 / 3.0)
    every = max(1, int(checkpoint_every))
    pending_since_last_dump = 0
    n_ok = 0
    n_failed = 0

    def _dump() -> None:
        if checkpoint_path is None:
            return
        try:
            _write_tracked_bubinfo(
                checkpoint_path, times, f_real, radius_eq
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            print(f"  [warn] BEM checkpoint write failed: {exc}")

    iterator: Any
    desc = f"BEM ({trial_pair})"
    if stride > 1:
        desc += f", stride={stride}"
    if tqdm is not None:
        iterator = tqdm(
            solve_indices,
            total=n_solve,
            desc=desc,
            unit="frame",
            mininterval=1.0,
        )
    else:
        iterator = solve_indices
        if stride > 1:
            print(
                f"  [BEM] solving {n_solve}/{n} frames "
                f"(stride={stride}); the rest will be linearly "
                f"interpolated."
            )

    for solve_pos, k in enumerate(iterator):
        st = frame_states[k]
        v_unit = st.get("v_unit")
        f_tris = st.get("f")
        if v_unit is None or f_tris is None:
            print(f"  [warn] BEM frame {k}: missing v_unit/f, skipping")
            n_failed += 1
            pending_since_last_dump += 1
        else:
            try:
                res = solve_minnaert_frequency_galerkin(
                    (v_unit, f_tris),
                    trial_pair=str(trial_pair),
                    precond="mass",
                    solver="gmres",
                    gmres_tol=float(gmres_tol),
                    quadrature_regular=int(quadrature_regular),
                    quadrature_singular=int(quadrature_singular),
                )
                f_unit = float(res["frequency"])
                if not math.isfinite(f_unit) or f_unit <= 0.0:
                    raise ValueError(f"non-finite/non-positive: {f_unit}")
                f_real[k] = f_unit * scale
                n_ok += 1
                pending_since_last_dump += 1
                if tqdm is not None:
                    iterator.set_postfix(
                        f_unit=f"{f_unit:.4f}",
                        f_real=f"{f_real[k]:.4f}",
                        ok=n_ok,
                        fail=n_failed,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] BEM frame {k} failed: {exc}")
                n_failed += 1
                pending_since_last_dump += 1
                if tqdm is None:
                    print(
                        f"  [BEM] solve {solve_pos + 1}/{n_solve} "
                        f"(frame {k + 1}/{n}): FAILED"
                    )

        if tqdm is None:
            print(
                f"  [BEM] solve {solve_pos + 1}/{n_solve} "
                f"(frame {k + 1}/{n}): f_real={f_real[k]:.4f} Hz "
                f"(ok={n_ok}, fail={n_failed})"
            )

        if pending_since_last_dump >= every:
            _dump()
            pending_since_last_dump = 0

    if pending_since_last_dump > 0:
        _dump()

    if stride > 1:
        # Linearly interpolate every non-solved (or solved-and-failed)
        # frame from its bracketing solved frames. ``np.interp`` clamps
        # to boundary values for x outside [xp.min, xp.max], which is
        # safe because we always include both 0 and n-1 in
        # ``solve_indices``. If *every* solve failed, leave f_real as
        # NaN so the failure is visible downstream rather than papered
        # over with garbage.
        finite_mask = np.isfinite(f_real)
        if np.any(finite_mask):
            xp = np.flatnonzero(finite_mask).astype(np.float64)
            fp = f_real[finite_mask]
            x_all = np.arange(n, dtype=np.float64)
            f_real_interp = np.interp(x_all, xp, fp)
            n_interp = int(np.sum(~finite_mask))
            f_real[~finite_mask] = f_real_interp[~finite_mask]
            print(
                f"  [BEM] stride={stride}: linearly interpolated "
                f"{n_interp}/{n} non-solved frame(s) from "
                f"{int(np.sum(finite_mask))} solved anchor(s)."
            )
            # Re-dump so the final on-disk file shows interpolated
            # values rather than the NaN placeholders the checkpoint
            # was happy to write.
            _dump()
        else:
            print(
                f"  [BEM] stride={stride}: ALL solved frames failed; "
                f"cannot interpolate. Output stays NaN."
            )

    print(
        f"  [BEM] done: {n_ok}/{n} solves succeeded, {n_failed} failed."
    )
    return f_real


# ---------------------------------------------------------------------------
# trackedBubInfo writer
# ---------------------------------------------------------------------------


def _write_tracked_bubinfo(
    out_path: Path,
    times: np.ndarray,
    freqs: np.ndarray,
    radius_eq: float,
    *,
    bub_id: int = 1,
    pressure: float = 1.0,
    freq_format: str = "{:.6f}",
    time_format: str = "{:.9g}",
) -> None:
    """Write a single-Bub trackedBubInfo with position pinned at (0,0,0).

    No leading ``#``-comment header is emitted: the downstream
    parser does not strip comment lines (it expects them removed before it
    runs), so any header line whose
    second whitespace-token equals ``Bub`` would be misread as the actual
    Bub block start. We bypass that whole class of issue by writing only
    the structural lines.
    """
    if times.shape != freqs.shape:
        raise ValueError(
            f"times {times.shape} vs freqs {freqs.shape} shape mismatch"
        )
    if times.size == 0:
        raise ValueError("times/freqs are empty")

    lines: list[str] = []
    lines.append(f"Bub {int(bub_id)} {radius_eq:.6f}")
    lines.append(f"  Start: N {time_format.format(float(times[0]))}")
    for t, f in zip(times, freqs):
        if not math.isfinite(float(f)):
            f_token = "nan"
        else:
            f_token = freq_format.format(float(f))
        lines.append(
            "  "
            + " ".join(
                [
                    time_format.format(float(t)),
                    f_token,
                    "0",
                    "0",
                    "0",
                    f"{pressure:g}",
                ]
            )
        )
    lines.append(f"  End: C {time_format.format(float(times[-1]))}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize(name: str, values: np.ndarray) -> str:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return f"  {name:<14} (no finite values)"
    return (
        f"  {name:<14} min={a.min():.4f}  median={float(np.median(a)):.4f}  "
        f"max={a.max():.4f}  Hz"
    )


def _write_freq_curves_csv(
    csv_path: Path,
    times: np.ndarray,
    freq_per_method: dict[str, np.ndarray],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    n = times.size
    method_order = [tag for tag in METHOD_TAGS if tag in freq_per_method]
    fieldnames = ["k", "t"] + [f"f_{tag.lower()}" for tag in method_order]
    with csv_path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(fieldnames)
        for k in range(n):
            row: list[Any] = [k, float(times[k])]
            for tag in method_order:
                row.append(float(freq_per_method[tag][k]))
            writer.writerow(row)


# Color scheme matches python/scene/bubble_theater/plot_overlay_from_tracked.py
# (Minnaert grey, NN orange) plus the ellipsoid proxy in steel blue for visual
# contrast. BEM in green is the high-accuracy reference.
_PLOT_STYLE: dict[str, dict[str, Any]] = {
    "Minnaert": {"color": "#7F7F7F", "linestyle": "--", "linewidth": 1.6},
    "Ellipsoid": {"color": "#1F77B4", "linestyle": "-", "linewidth": 1.6},
    "NN8": {"color": "#FF7F0E", "linestyle": "-", "linewidth": 1.8},
    # "NNResidual": {"color": "#D62728", "linestyle": "-", "linewidth": 1.8},
    # "NN": {"color": "#9467BD", "linestyle": "-", "linewidth": 1.8},
    "BEM": {"color": "#2CA02C", "linestyle": "-", "linewidth": 1.6},
}


def _plot_freq_curves(
    png_path: Path,
    times: np.ndarray,
    freq_per_method: dict[str, np.ndarray],
    *,
    title_extras: str = "",
    legend_labels: dict[str, str] | None = None,
    fontsize: float | None = None,
    legend_loc: str | None = None,
) -> None:
    """Render f-vs-t curves for every available method on a single axis.

    Lazy-imports matplotlib with the Agg backend so this never blocks on a
    display window. Skips gracefully (with a warning) when matplotlib is
    not installed -- the rest of the pipeline does not depend on the plot.

    ``legend_labels`` maps internal method tags (dict keys in
    ``freq_per_method``) to legend strings; omitted tags keep ``tag`` as
    the label. Used by ``plot_overlay_from_tracked.py`` for publication
    names without renaming trackedBubInfo files or CSV columns.

    When ``fontsize`` is set (e.g. 15), axis labels, tick labels, and the
    legend use that size; ``None`` keeps matplotlib defaults.

    ``legend_loc`` is forwarded to ``matplotlib.axes.Axes.legend`` as
    ``loc=...``. ``None`` means ``"best"``. For publication overlays prefer
    ``"upper left"`` or ``"upper right"`` so the legend stays in a corner.
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        print(
            f"  [warn] skipping freq_curves.png: matplotlib unavailable "
            f"({exc})"
        )
        return

    method_order = [tag for tag in METHOD_TAGS if tag in freq_per_method]
    if not method_order:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for tag in method_order:
        y = np.asarray(freq_per_method[tag], dtype=np.float64)
        finite = np.isfinite(y)
        if not bool(finite.any()):
            continue
        style = _PLOT_STYLE.get(tag, {})
        label = (legend_labels or {}).get(tag, tag)
        ax.plot(
            times[finite],
            y[finite],
            label=label,
            **style,
        )

    ax.set_xlabel("Time (s)", **({"fontsize": fontsize} if fontsize is not None else {}))
    ax.set_ylabel("Frequency (Hz)", **({"fontsize": fontsize} if fontsize is not None else {}))
    title = "Procedural single-bubble frequency curves"
    if title_extras:
        title += f"  ({title_extras})"
    # ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if legend_loc is None:
        mpl_legend_loc: str = "best"
    else:
        allowed = {"best", "upper left", "upper right"}
        if legend_loc not in allowed:
            raise ValueError(
                f"legend_loc must be one of {sorted(allowed)}, got {legend_loc!r}"
            )
        mpl_legend_loc = legend_loc
    legend_kw: dict[str, Any] = {"loc": mpl_legend_loc, "frameon": False}
    if fontsize is not None:
        legend_kw["fontsize"] = fontsize
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.legend(**legend_kw)
    if times.size:
        ax.set_xlim(float(times[0]), float(times[-1]))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _resolve_methods(value: str) -> list[str]:
    if not value:
        raise SystemExit("--methods cannot be empty")
    out: list[str] = []
    for tok in value.split(","):
        key = tok.strip().lower()
        if not key:
            continue
        if key not in METHOD_ALIASES:
            raise SystemExit(
                f"unknown --methods entry {tok!r}; expected one of "
                f"{sorted(METHOD_ALIASES)}"
            )
        tag = METHOD_ALIASES[key]
        if tag not in out:
            out.append(tag)
    if not out:
        raise SystemExit("--methods resolved to an empty list")
    return out


def _discover_obj_paths(
    input_prefix: Path, frame_start: int, frame_count: int
) -> list[Path]:
    if frame_count <= 0:
        raise SystemExit(f"--frame-count must be positive, got {frame_count}")
    parent = input_prefix.parent
    stem = input_prefix.name
    paths: list[Path] = []
    missing: list[Path] = []
    for k in range(int(frame_start), int(frame_start) + int(frame_count)):
        p = parent / f"{stem}_{k:04d}.obj"
        if not p.is_file():
            missing.append(p)
        paths.append(p)
    if missing:
        head = "\n  ".join(str(m) for m in missing[:10])
        more = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
        raise SystemExit(
            f"missing {len(missing)} of {len(paths)} mesh frames; e.g.:\n  "
            f"{head}{more}"
        )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_prefix",
        type=Path,
        help=(
            "Mesh path prefix; the script will look for "
            "{input_prefix}_{F:04d}.obj for F in "
            "[--frame-start, --frame-start + --frame-count)."
        ),
    )
    parser.add_argument(
        "--shrink-radius-scale",
        type=float,
        default=48.5,
        help=(
            "Multiplicative shrink factor applied to the bubble's "
            "equivalent-sphere radius (and uniformly to the mesh) AFTER "
            "the mesh has been normalized to unit volume. Concretely, "
            "every frame is uniformly rescaled (about its centroid) so "
            "its enclosed volume equals "
            "V_target = 1 / shrink_radius_scale**3 (m^3), equivalently "
            "R_eq = (3/(4 pi))**(1/3) / shrink_radius_scale. The raw "
            "input mesh volume is irrelevant -- it is always normalized "
            "to V=1 first. Default: 1.0 (V_target = 1 m^3, R_eq ~ 0.6204 m)."
        ),
    )
    parser.add_argument("--frame-count", type=int, default=90)
    parser.add_argument(
        "--frame-start",
        type=int,
        default=1,
        help=(
            "First frame index F used to build the OBJ path "
            "{input_prefix}_{F:04d}.obj. Default: 1 (so the sequence is "
            "_0001.obj ... _{frame_count:04d}.obj). Pass 0 if your "
            "sequence is zero-indexed (_0000.obj ... )."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help=(
            "Total time span (seconds) covered by the trackedBubInfo sample "
            "lines. The N=--frame-count rows are placed at "
            "t[k] = k * duration / (N - 1) for k = 0 ... N-1. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Base output directory. The actual artifacts land in "
            "<output_dir>/<input_parent_name>_result/, e.g. an input "
            "prefix .../curl_noise/bub yields .../curl_noise_result/. "
            f"Default base: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="minnaert,strasberg,nn,nn8",
        help=(
            "Comma-separated subset of "
            "{minnaert,strasberg,nn,nn_residual,nn8,bem}. The three NN "
            "entries select the chull-direct ('nn'), the chull "
            "log-residual ('nn_residual'), and the 8-feature "
            "chull+inertia direct ('nn8') artifact families "
            "respectively -- 'nn' and 'nn_residual' share the six chull "
            "features and 64/32 architecture (different training "
            "targets); 'nn8' adds two inertia ratios (I11/I00, I22/I00) "
            "on top and predicts log(f) directly. Default: "
            "'minnaert,strasberg,nn,nn8' (NN-residual and BEM are "
            "opt-in; for BEM, use --bem-reuse-tracked to skip the "
            "solver and re-load frequencies from an earlier "
            "trackedBubInfo_BEM.txt)."
        ),
    )
    parser.add_argument(
        "--bem-trial-pair",
        type=str,
        default="P1-DP0",
        choices=["P1-DP0", "DP0-DP0", "P1-DP1"],
        help=(
            "Galerkin trial pair forwarded to "
            "solve_minnaert_frequency_galerkin (only used when 'bem' is in "
            "--methods). Default: 'P1-DP0' (conforming P1 Dirichlet "
            "trace, DP0 Neumann trace -- the recommended pair)."
        ),
    )
    parser.add_argument(
        "--bem-checkpoint-every",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Flush the full accumulated trackedBubInfo_BEM.txt to disk "
            "after every N newly-finished BEM frames (default 5). "
            "Frames not yet solved appear as NaN rows so the on-disk "
            "file always reflects every BEM result produced through the "
            "last checkpoint."
        ),
    )
    parser.add_argument(
        "--bem-stride",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Subsample factor for the BEM solve grid: only frames whose "
            "index is a multiple of N (plus the final frame, anchored) "
            "are actually fed to the Galerkin solver; the remaining "
            "frames are filled in at the very end via piecewise-linear "
            "interpolation through the solved frames. With "
            "--frame-count 240 --bem-stride 10 you pay for ~25 BEM "
            "solves instead of 240. Default: 1 (solve every frame, "
            "no interpolation)."
        ),
    )
    parser.add_argument(
        "--bem-reuse-tracked",
        action="store_true",
        help=(
            "Skip the BEM solver entirely and load the per-frame BEM "
            "frequencies from the existing "
            "<output_dir>/trackedBubInfo_BEM.txt instead. Useful for "
            "re-emitting the curves on a new --duration time grid "
            "without paying the multi-minute Galerkin cost again. The "
            "cached file's "
            "row count must match --frame-count; the time column is "
            "discarded and replaced by the new linspace(0, --duration, "
            "--frame-count) grid."
        ),
    )
    parser.add_argument(
        "--artifacts-nn8",
        type=Path,
        default=DEFAULT_ARTIFACTS_NN8,
        help=(
            "Trained 8-feature chull+inertia NN artifact directory (the "
            "model that predicts log(f_unit) directly from the six chull "
            "features PLUS the two inertia ratios I11/I00, I22/I00 -- "
            "trained by fit_shape_freq_model.py). Used when 'nn8' is in "
            "--methods. "
            f"Default: {DEFAULT_ARTIFACTS_NN8}"
        ),
    )
    parser.add_argument(
        "--frequency-format", type=str, default="{:.6f}",
        help="Frequency formatter for the trackedBubInfo sample lines.",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help=(
            "Skip mesh discovery, per-frame feature extraction, every "
            "frequency model (Minnaert / Ellipsoid / NN8 / BEM) and the "
            "trackedBubInfo writing step entirely. Reuse the pre-existing "
            "trackedBubInfo_<method>.txt files already sitting in "
            "<output_dir>/<input_parent_name>_result/ (one per entry in "
            "--methods), re-emit the summary, and redraw freq_curves.csv / "
            ".png from their rows. In this mode the positional input_prefix "
            "is only used to derive the result folder; the OBJs and the NN "
            "artifact dir do not need to exist."
        ),
    )
    args = parser.parse_args()

    methods = _resolve_methods(args.methods)
    # --bem-reuse-tracked is meaningless unless BEM is actually in the
    # active methods list, so auto-include it (the cache-read branch
    # below still validates that the trackedBubInfo_BEM.txt file exists
    # before consuming it). This avoids the surprising no-op where the
    # user passes --bem-reuse-tracked but forgets to add 'bem' to
    # --methods (whose default no longer includes it).
    if args.bem_reuse_tracked and "BEM" not in methods:
        methods.append("BEM")
        print(
            "[info] --bem-reuse-tracked given; auto-including 'BEM' in "
            "methods so the cached BEM frequencies get re-rendered."
        )
    input_prefix = args.input_prefix.resolve()
    # Auto-bucket every render into a per-input-folder subdir so multiple
    # mesh sequences (curl_noise/, ellipsoid/, ...) under the same
    # --output-dir do not overwrite each other. Layout:
    #   <output_dir>/<input_parent_name>_result/{trackedBubInfo_*.txt,
    #                                            freq_curves.csv, .png}
    # Falls back to "default_result" when the input prefix sits at the
    # filesystem root (no parent name available).
    base_output_dir = args.output_dir.resolve()
    parent_name = input_prefix.parent.name or "default"
    output_dir = base_output_dir / f"{parent_name}_result"
    artifacts_nn8 = args.artifacts_nn8.resolve()

    if args.frame_count < 2:
        raise SystemExit(
            f"--frame-count must be >= 2 to define a time grid (got "
            f"{args.frame_count})"
        )
    if not (math.isfinite(args.duration) and args.duration > 0.0):
        raise SystemExit(f"--duration must be > 0, got {args.duration}")
    if not (
        math.isfinite(args.shrink_radius_scale)
        and args.shrink_radius_scale > 0.0
    ):
        raise SystemExit(
            f"--shrink-radius-scale must be > 0, got "
            f"{args.shrink_radius_scale}"
        )
    if not args.replot_only:
        if "NN8" in methods and not artifacts_nn8.is_dir():
            raise SystemExit(f"missing NN8 artifact dir: {artifacts_nn8}")
    if int(args.bem_stride) < 1:
        raise SystemExit(
            f"--bem-stride must be >= 1, got {args.bem_stride}"
        )

    if args.replot_only:
        # Mesh sequence is irrelevant in replot-only mode: we ingest the
        # pre-existing trackedBubInfo files instead.
        obj_paths: list[Path] = []
    else:
        obj_paths = _discover_obj_paths(
            input_prefix, args.frame_start, args.frame_count
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # The mesh is always normalized to unit volume first, then shrunk by
    # ``shrink_radius_scale``. Raw input volume is irrelevant to V_target;
    # we still record frame 0's measured volume below as a sanity readout.
    shrink = float(args.shrink_radius_scale)
    volume_target = 1.0 / (shrink ** 3)
    radius_eq_unit = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    radius_eq = radius_eq_unit / shrink

    print(f"=== Input prefix:   {input_prefix}")
    if args.replot_only:
        print(
            "    mode:           replot-only (mesh + frequency model + "
            "trackedBubInfo write steps skipped)"
        )
    else:
        print(
            f"    Frames:         {args.frame_count} starting at "
            f"{args.frame_start:04d}"
        )
        print(
            f"    Normalize:      every mesh -> V=1 m^3 first "
            f"(R_eq_unit={radius_eq_unit:.6f} m)"
        )
        print(
            f"    Shrink:         shrink_radius_scale={shrink:g} -> "
            f"V_target={volume_target:.6e} m^3 | R_eq={radius_eq:.6f} m"
        )
        print(f"    Duration:       {args.duration:g} s")
    print(f"    Methods:        {methods}")
    print(f"=== Output dir:     {output_dir}")

    if args.replot_only:
        # ---- Re-plot only: skip steps 1-3 and read the existing
        # trackedBubInfo files from disk.
        print()
        print("Loading per-method trackedBubInfo files from disk ...")
        tracked_paths_ao: dict[str, Path] = {}
        freq_per_method_ao: dict[str, np.ndarray] = {}
        times_ao: np.ndarray | None = None
        times_ref_tag: str | None = None
        missing: list[Path] = []
        # Iterate every user-requested method (not METHOD_TAGS) so a
        # missing trackedBubInfo_<tag>.txt always surfaces as a clear
        # SystemExit instead of getting silently dropped from the
        # downstream summary / CSV / PNG dispatch.
        for tag in methods:
            p = output_dir / f"trackedBubInfo_{tag}.txt"
            if not p.is_file():
                missing.append(p)
                continue
            t_arr, f_arr = _load_times_and_freqs_from_tracked_bubinfo(p)
            if times_ao is None:
                times_ao = t_arr
                times_ref_tag = tag
            elif t_arr.shape != times_ao.shape:
                raise SystemExit(
                    f"row-count mismatch in --replot-only mode: "
                    f"trackedBubInfo_{tag}.txt has {t_arr.size} sample row(s) "
                    f"but trackedBubInfo_{times_ref_tag}.txt has "
                    f"{times_ao.size}. Replot-only requires matching row "
                    f"counts across every method."
                )
            tracked_paths_ao[tag] = p
            freq_per_method_ao[tag] = f_arr
            print(f"  loaded {p}  ({f_arr.size} rows)")

        if missing:
            head = "\n  ".join(str(m) for m in missing)
            raise SystemExit(
                "--replot-only requires the following pre-built tracked "
                "file(s) to already exist:\n  " + head
            )
        if times_ao is None or times_ao.size == 0:
            raise SystemExit(
                f"--replot-only: no trackedBubInfo sample rows found "
                f"under {output_dir} for --methods {methods}; nothing "
                f"to render."
            )

        print()
        print("Per-method frequency summary (Hz):")
        for tag in methods:
            print(_summarize(tag, freq_per_method_ao[tag]))

        csv_path_ao = output_dir / "freq_curves.csv"
        _write_freq_curves_csv(
            csv_path_ao,
            times_ao,
            {tag: freq_per_method_ao[tag] for tag in methods},
        )
        print(f"  wrote {csv_path_ao}")

        png_path_ao = output_dir / "freq_curves.png"
        _plot_freq_curves(
            png_path_ao,
            times_ao,
            {tag: freq_per_method_ao[tag] for tag in methods},
            title_extras=(
                "re-plotted from existing trackedBubInfo files"
            ),
        )
        if png_path_ao.is_file():
            print(f"  wrote {png_path_ao}")

        print()
        print("=== Done.")
        print(f"    output dir: {output_dir}")
        return 0

    # ---- 1. Per-frame mesh processing ---------------------------------
    print()
    print("Loading + rescaling meshes ...")
    frame_states: list[dict[str, Any]] = []
    raw_volumes: list[float] = []
    for k, p in enumerate(obj_paths):
        st = _process_frame(p, volume_target)
        frame_states.append(st)
        raw_volumes.append(st["signed_volume_raw"])
    raw_arr = np.asarray(raw_volumes, dtype=np.float64)
    V_0_raw = float(raw_arr[0])
    rel = (raw_arr - V_0_raw) / max(abs(V_0_raw), 1e-30)

    print(
        "  pre-normalize signed volume per frame "
        "(rel = (V_k - V_0_raw) / V_0_raw):"
    )
    for k, (vk, rk, p) in enumerate(zip(raw_arr, rel, obj_paths)):
        print(
            f"    frame {k:4d}  V={float(vk):.6e}  rel={float(rk):+.3e}  "
            f"({p.name})"
        )
    print(
        f"  summary: V_0_raw={V_0_raw:.6e} m^3 | "
        f"rel-deviation min={float(rel.min()):+.3e}  "
        f"median={float(np.median(rel)):+.3e}  "
        f"max={float(rel.max()):+.3e}"
    )
    if float(np.max(np.abs(rel))) > 1e-2:
        print(
            "  [warn] input mesh sequence varies in raw volume by >1% "
            "relative to frame 0; this is informational only -- every "
            "frame is independently rescaled to V_target before any "
            "frequency computation, so no shape variation is lost."
        )

    print(
        f"  V_target (= 1 / shrink_radius_scale**3): "
        f"{volume_target:.6e} m^3 | R_eq: {radius_eq:.6f} m"
    )

    # ---- 2. Per-frame frequencies -------------------------------------
    n = len(frame_states)
    times = np.linspace(0.0, float(args.duration), n, dtype=np.float64)

    f_minnaert = np.full(n, MINNAERT_CONSTANT / radius_eq, dtype=np.float64)

    f_strasberg = np.full(n, np.nan, dtype=np.float64)
    n_strasberg_fb = 0
    for k, st in enumerate(frame_states):
        try:
            f_strasberg[k] = _strasberg_freq_real(
                st["inertia_principal_unit"], volume_target
            )
        except Exception as exc:  # noqa: BLE001
            f_strasberg[k] = float(f_minnaert[k])
            n_strasberg_fb += 1
            if n_strasberg_fb <= 3:
                print(
                    f"  [strasberg fallback] frame {k} ({obj_paths[k].name}): "
                    f"{type(exc).__name__}: {exc}"
                )
    if n_strasberg_fb > 0:
        print(
            f"  Strasberg fallback to Minnaert: {n_strasberg_fb}/{n} "
            f"frame(s)"
        )

    chull_per_frame = [st["chull_features"] for st in frame_states]
    n_feat_failed = sum(
        1 for fp in chull_per_frame if fp.get("status") != "ok"
    )
    if "NN8" in methods and n_feat_failed:
        print(
            f"  [nn] {n_feat_failed}/{n} frames failed chull feature "
            f"extraction; those frames will be NaN in NN8 output."
        )
        for k, fp in enumerate(chull_per_frame):
            if fp.get("status") != "ok" and k < 5:
                print(
                    f"    e.g. frame {k} ({obj_paths[k].name}): "
                    f"{fp.get('status')}"
                )

    # The 6-feature direct and Strasberg-residual heads were retired; NN8 (the
    # 8-feature production model, Eq. 16) is the only learned method left.
    if "NN8" in methods:
        inertia_per_frame = [st["inertia_principal_unit"] for st in frame_states]
    else:
        inertia_per_frame = []  # unused

    if "NN8" in methods:
        f_nn8, nn8_valid = _nn8_predict_per_frame(
            chull_per_frame,
            inertia_per_frame,
            artifacts_nn8,
            volume_target,
        )
        if not bool(nn8_valid.all()):
            n_nn8_invalid = int((~nn8_valid).sum())
            print(
                f"  [nn8] {n_nn8_invalid}/{n} total frames have NaN "
                f"frequency after NN8 inference."
            )
    else:
        f_nn8 = np.full(n, np.nan, dtype=np.float64)

    if "BEM" in methods:
        cached_bem_path = output_dir / "trackedBubInfo_BEM.txt"
        if args.bem_reuse_tracked:
            if not cached_bem_path.is_file():
                raise SystemExit(
                    f"--bem-reuse-tracked given but cached file is missing: "
                    f"{cached_bem_path}"
                )
            print()
            print(
                f"Reusing cached BEM frequencies from {cached_bem_path} "
                f"(skipping Galerkin solver) ..."
            )
            f_bem_cached = _load_freqs_from_tracked_bubinfo(cached_bem_path)
            if f_bem_cached.size != n:
                raise SystemExit(
                    f"--bem-reuse-tracked: cached file has "
                    f"{f_bem_cached.size} sample row(s) but --frame-count="
                    f"{n}; cannot remap. Re-run BEM or adjust --frame-count."
                )
            f_bem = f_bem_cached
            # Rewrite the file with the new time grid so any downstream
            # consumer gets the t[k] = k*duration/(N-1) grid the user just
            # asked for, not the stale one from the earlier run.
            _write_tracked_bubinfo(
                cached_bem_path,
                times,
                f_bem,
                radius_eq,
                freq_format=args.frequency_format,
            )
            print(
                f"  rewrote {cached_bem_path} with new time grid "
                f"(duration={args.duration:g}s)"
            )
        else:
            stride = max(1, int(args.bem_stride))
            n_solve_planned = len(
                sorted(set(range(0, n, stride)) | {n - 1})
            )
            stride_msg = (
                f"stride={stride} -> {n_solve_planned}/{n} solves + linear "
                f"interp"
                if stride > 1
                else f"every frame ({n} solves)"
            )
            print()
            print(
                f"Computing BEM frequencies (Galerkin {args.bem_trial_pair}, "
                f"{stride_msg}, checkpoint every "
                f"{args.bem_checkpoint_every} frames) ..."
            )
            f_bem = _bem_predict_per_frame(
                frame_states,
                volume_target=volume_target,
                times=times,
                radius_eq=radius_eq,
                trial_pair=str(args.bem_trial_pair),
                checkpoint_path=cached_bem_path,
                checkpoint_every=int(args.bem_checkpoint_every),
                stride=stride,
            )
        n_bem_invalid = int(np.sum(~np.isfinite(f_bem)))
        if n_bem_invalid > 0:
            print(
                f"  [bem] {n_bem_invalid}/{n} total frames have NaN "
                f"frequency after BEM solve."
            )
    else:
        f_bem = np.full(n, np.nan, dtype=np.float64)

    print()
    print("Per-method frequency summary (Hz):")
    if "Minnaert" in methods:
        print(_summarize("Minnaert", f_minnaert))
    if "Ellipsoid" in methods:
        print(_summarize("Ellipsoid", f_strasberg))
    if "NN8" in methods:
        print(_summarize("NN8", f_nn8))
    if "BEM" in methods:
        print(_summarize("BEM", f_bem))

    # ---- 3. Write trackedBubInfo per method ----------------------------
    tracked_paths: dict[str, Path] = {}
    freq_per_method: dict[str, np.ndarray] = {
        "Minnaert": f_minnaert,
        "Ellipsoid": f_strasberg,
        "NN8": f_nn8,
        "BEM": f_bem,
    }
    print()
    for tag in METHOD_TAGS:
        if tag not in methods:
            continue
        tracked_path = output_dir / f"trackedBubInfo_{tag}.txt"
        _write_tracked_bubinfo(
            tracked_path,
            times,
            freq_per_method[tag],
            radius_eq,
            freq_format=args.frequency_format,
        )
        tracked_paths[tag] = tracked_path
        print(f"  wrote {tracked_path}")

    csv_path = output_dir / "freq_curves.csv"
    _write_freq_curves_csv(
        csv_path,
        times,
        {tag: freq_per_method[tag] for tag in methods},
    )
    print(f"  wrote {csv_path}")

    png_path = output_dir / "freq_curves.png"
    _plot_freq_curves(
        png_path,
        times,
        {tag: freq_per_method[tag] for tag in methods},
        title_extras=(
            f"shrink_radius_scale={shrink:g}, R_eq={radius_eq:.4f} m, "
            f"V_target={volume_target:.3e} m^3"
        ),
    )
    if png_path.is_file():
        print(f"  wrote {png_path}")

    print()
    print("=== Done.")
    print(f"    output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

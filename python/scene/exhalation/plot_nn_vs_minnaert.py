"""Compare NN-predicted frequencies in trackedBubInfo-NN.txt against Minnaert.

Reads an existing ``trackedBubInfo-NN.txt`` (already produced by
``python/scene/_common/write_trackedbubinfo_nn.py``), restricts the comparison to
``Bub`` blocks that have a corresponding ``bub_<id>/`` mesh directory under
``--meshes-root`` (i.e. the lines that actually got NN predictions instead of
the Minnaert fallback), and emits four artifacts under ``--out-dir``:

  * ``<prefix>_aggregate.png``  / ``<prefix>_aggregate.html``
        Scatter of ``f_nn`` vs ``f_minnaert`` on log-log axes plus a histogram
        of the signed percent error. ``f_minnaert = MINNAERT_CONSTANT / r``
        for each bubble, where ``r`` is the per-bubble radius from the
        ``Bub <id> <radius>`` header.
  * ``<prefix>_per_bubble.png`` / ``<prefix>_per_bubble.html``
        Per-bubble frequency-vs-time traces for ``--num-bubbles`` randomly
        chosen NN-updated bubbles, with the bubble's Minnaert reference line
        overlaid.

The script does NOT re-run NN inference.

Example:
    python python/scene/exhalation/plot_nn_vs_minnaert.py
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


PYTHON_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = PYTHON_ROOT / "baseline"
for _p in (PYTHON_ROOT, BASELINE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from freq_model.analytical.minnaert_freq import MINNAERT_CONSTANT  # noqa: E402
from tracked_bubinfo import parse_trackedbubinfo_blocks


# Local copy of parse_bub_header_radii from scene/_common/write_trackedbubinfo_nn.py:
# importing that module would pull in torch (Windows DLL-ordering workaround)
# and this plot script doesn't need torch at all.
_BUB_HEADER_RE = re.compile(r"^\s*Bub\s+(\d+)\s+([+\-]?\d*\.?\d+(?:[eE][+\-]?\d+)?)")


def parse_bub_header_radii(lines: list[str]) -> dict[int, float]:
    radii: dict[int, float] = {}
    for ln in lines:
        m = _BUB_HEADER_RE.match(ln)
        if not m:
            continue
        try:
            radii[int(m.group(1))] = float(m.group(2))
        except ValueError:
            continue
    return radii


REPO_ROOT = PYTHON_ROOT.parent
DEFAULT_TRACKED_NN = REPO_ROOT / "dataset" / "exhalation" / "trackedBubInfo_NN.txt"
# No default mesh root: the raw per-bubble mesh tree is simulator output and is
# not part of the release, so any baked-in path would only be valid on the
# machine it was written on.
DEFAULT_OUT_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PREFIX = "lbm_exhale_nn_vs_minnaert"

BUB_DIR_RE = re.compile(r"^bub_(\d+)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def discover_nn_updated_bub_ids(meshes_root: Path) -> set[int]:
    """Return the set of bub_ids that have a ``bub_<id>/`` mesh directory."""
    bub_ids: set[int] = set()
    if not meshes_root.is_dir():
        raise FileNotFoundError(f"meshes_root does not exist: {meshes_root}")
    for entry in meshes_root.iterdir():
        if not entry.is_dir():
            continue
        m = BUB_DIR_RE.match(entry.name)
        if m:
            bub_ids.add(int(m.group(1)))
    return bub_ids


def parse_nn_samples(
    tracked_nn_path: Path,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[int, float]
]:
    """Read trackedBubInfo-NN.txt; return (bub_ids, t, f_nn, radii_by_bub).

    Sample lines whose frequency token isn't a finite positive float (e.g.
    leftover ``nan`` / ``-1`` sentinels) are dropped.
    """
    src_lines, sample_idxs, sample_meta = parse_trackedbubinfo_blocks(tracked_nn_path)
    radii_by_bub = parse_bub_header_radii(src_lines)

    n = len(sample_idxs)
    bub_ids = np.empty(n, dtype=np.int64)
    t_arr = np.empty(n, dtype=np.float64)
    f_nn = np.empty(n, dtype=np.float64)
    for i, (li, (bid, t_s, _y)) in enumerate(zip(sample_idxs, sample_meta)):
        bub_ids[i] = int(bid)
        t_arr[i] = float(t_s)
        parts = src_lines[li].split()
        try:
            f_nn[i] = float(parts[1])
        except (ValueError, IndexError):
            f_nn[i] = np.nan
    return bub_ids, t_arr, f_nn, radii_by_bub


def parse_minnaert_samples_aligned(
    tracked_minnaert_path: Path,
    bub_ids_ref: np.ndarray,
    t_ref: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> np.ndarray:
    """Read frequencies from a parallel trackedBubInfo file and align them
    sample-by-sample to ``(bub_ids_ref, t_ref)``.

    The two files must contain the same sample lines in the same order (this
    is the case when both come out of the iter cutoff pipeline). If the
    structure differs, a ``ValueError`` is raised pointing to the first
    mismatch.
    """
    src_lines, sample_idxs, sample_meta = parse_trackedbubinfo_blocks(
        tracked_minnaert_path
    )
    n_ref = len(bub_ids_ref)
    if len(sample_idxs) != n_ref:
        raise ValueError(
            f"sample-line count mismatch: NN file has {n_ref}, "
            f"Minnaert file ({tracked_minnaert_path.name}) has {len(sample_idxs)}"
        )

    f_minn = np.empty(n_ref, dtype=np.float64)
    for i, (li, (bid, t_s, _y)) in enumerate(zip(sample_idxs, sample_meta)):
        bid_i = int(bid)
        t_s_f = float(t_s)
        if int(bub_ids_ref[i]) != bid_i:
            raise ValueError(
                f"bub_id mismatch at sample {i}: NN={int(bub_ids_ref[i])} vs "
                f"Minnaert={bid_i}"
            )
        if not math.isclose(float(t_ref[i]), t_s_f, rel_tol=rtol, abs_tol=atol):
            raise ValueError(
                f"timestamp mismatch at sample {i} (Bub {bid_i}): "
                f"NN t={float(t_ref[i])!r} vs Minnaert t={t_s_f!r}"
            )
        parts = src_lines[li].split()
        try:
            f_minn[i] = float(parts[1])
        except (ValueError, IndexError):
            f_minn[i] = np.nan
    return f_minn


# ---------------------------------------------------------------------------
# Aggregate plot
# ---------------------------------------------------------------------------


def render_aggregate_png(
    f_nn: np.ndarray,
    f_minn: np.ndarray,
    err_pct: np.ndarray,
    radius: np.ndarray,
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ok = (
        np.isfinite(f_nn)
        & (f_nn > 0)
        & np.isfinite(f_minn)
        & (f_minn > 0)
        & np.isfinite(err_pct)
    )
    f_nn = f_nn[ok]
    f_minn = f_minn[ok]
    err_pct = err_pct[ok]
    radius = radius[ok]
    if f_nn.size == 0:
        raise RuntimeError("no finite (f_nn, f_minnaert) pairs to plot")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    lo = float(min(f_nn.min(), f_minn.min()))
    hi = float(max(f_nn.max(), f_minn.max()))
    diag = np.geomspace(lo, hi, 200)
    band_lo = 0.95 * diag
    band_hi = 1.05 * diag

    log_r = np.log10(np.maximum(radius, 1e-12))
    norm = Normalize(vmin=float(np.nanmin(log_r)), vmax=float(np.nanmax(log_r)))
    sc = axes[0].scatter(
        f_minn,
        f_nn,
        s=8,
        alpha=0.55,
        c=log_r,
        cmap="viridis",
        norm=norm,
        linewidths=0,
        zorder=2,
    )
    axes[0].fill_between(
        diag,
        band_lo,
        band_hi,
        color="lightgreen",
        alpha=0.25,
        linewidth=0,
        label="+/-5% band",
        zorder=0,
    )
    axes[0].plot(diag, diag, "k--", linewidth=1, label="y = x", zorder=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Minnaert frequency C / r  (Hz)")
    axes[0].set_ylabel("NN-predicted frequency  (Hz)")
    axes[0].set_title("NN vs Minnaert (log-log)")
    axes[0].legend(loc="upper left")
    cbar = fig.colorbar(sc, ax=axes[0], shrink=0.85)
    cbar.set_label("log10(radius [m])")

    median_err = float(np.median(err_pct))
    mean_err = float(np.mean(err_pct))
    p5 = float(np.mean(np.abs(err_pct) <= 5.0) * 100.0)
    p20 = float(np.mean(np.abs(err_pct) <= 20.0) * 100.0)
    bins = np.linspace(
        max(-200.0, float(np.percentile(err_pct, 0.5))),
        min(500.0, float(np.percentile(err_pct, 99.5))),
        80,
    )
    axes[1].hist(err_pct, bins=bins, color="steelblue", alpha=0.85)
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].axvline(median_err, color="crimson", linewidth=1.2, linestyle="--",
                    label=f"median = {median_err:+.2f}%")
    axes[1].axvline(mean_err, color="darkorange", linewidth=1.2, linestyle=":",
                    label=f"mean = {mean_err:+.2f}%")
    axes[1].set_xlabel("Signed percent error  100*(f_nn - f_minnaert) / f_minnaert")
    axes[1].set_ylabel("Sample-line count")
    axes[1].set_title(
        f"Signed % error  |  n={f_nn.size}  |  +/-5%: {p5:.1f}%  |  +/-20%: {p20:.1f}%"
    )
    axes[1].legend(loc="upper right")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def render_aggregate_html(
    f_nn: np.ndarray,
    f_minn: np.ndarray,
    err_pct: np.ndarray,
    radius: np.ndarray,
    bub_ids: np.ndarray,
    t_arr: np.ndarray,
    out_html: Path,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # pragma: no cover
        print(f"[html skip] plotly unavailable: {exc}")
        return

    ok = (
        np.isfinite(f_nn)
        & (f_nn > 0)
        & np.isfinite(f_minn)
        & (f_minn > 0)
        & np.isfinite(err_pct)
    )
    f_nn = f_nn[ok]
    f_minn = f_minn[ok]
    err_pct = err_pct[ok]
    radius = radius[ok]
    bub_ids = bub_ids[ok]
    t_arr = t_arr[ok]
    if f_nn.size == 0:
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "NN vs Minnaert (log-log)",
            "Signed percent error histogram",
        ),
        horizontal_spacing=0.12,
        column_widths=[0.55, 0.45],
    )

    lo = float(min(f_nn.min(), f_minn.min()))
    hi = float(max(f_nn.max(), f_minn.max()))
    diag = np.geomspace(lo, hi, 200)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([diag, diag[::-1]]),
            y=np.concatenate([0.95 * diag, (1.05 * diag)[::-1]]),
            fill="toself",
            fillcolor="rgba(144, 238, 144, 0.28)",
            line=dict(color="rgba(144, 238, 144, 0)"),
            name="+/-5% band",
            hoverinfo="skip",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="y = x",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    log_r = np.log10(np.maximum(radius, 1e-12))
    hover_lines = []
    for bid, t_s, r, fm, fn, e in zip(
        bub_ids, t_arr, radius, f_minn, f_nn, err_pct
    ):
        hover_lines.append(
            "<br>".join(
                [
                    f"<b>Bub {int(bid)}</b>",
                    f"t = {float(t_s):.6g} s",
                    f"radius = {float(r):.6g} m",
                    f"f_minnaert = {float(fm):.6g} Hz",
                    f"f_nn = {float(fn):.6g} Hz",
                    f"signed err = {float(e):+.3f}%",
                ]
            )
        )
    fig.add_trace(
        go.Scatter(
            x=f_minn,
            y=f_nn,
            mode="markers",
            marker=dict(
                size=5,
                opacity=0.6,
                color=log_r,
                colorscale="Viridis",
                colorbar=dict(
                    title="log10(radius [m])",
                    len=0.85,
                    x=0.495,
                    y=0.5,
                    thickness=12,
                ),
                line=dict(width=0),
            ),
            text=hover_lines,
            hovertemplate="%{text}<extra></extra>",
            name="samples",
        ),
        row=1,
        col=1,
    )

    median_err = float(np.median(err_pct))
    mean_err = float(np.mean(err_pct))
    p5 = float(np.mean(np.abs(err_pct) <= 5.0) * 100.0)
    p20 = float(np.mean(np.abs(err_pct) <= 20.0) * 100.0)
    fig.add_trace(
        go.Histogram(
            x=err_pct,
            nbinsx=80,
            marker=dict(color="steelblue"),
            opacity=0.85,
            name="signed % error",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_vline(x=0.0, line=dict(color="black", width=1), row=1, col=2)
    fig.add_vline(
        x=median_err,
        line=dict(color="crimson", width=1.4, dash="dash"),
        annotation_text=f"median {median_err:+.2f}%",
        annotation_position="top",
        row=1,
        col=2,
    )
    fig.add_vline(
        x=mean_err,
        line=dict(color="darkorange", width=1.4, dash="dot"),
        annotation_text=f"mean {mean_err:+.2f}%",
        annotation_position="bottom",
        row=1,
        col=2,
    )

    fig.update_xaxes(type="log", title_text="Minnaert frequency (Hz)", row=1, col=1)
    fig.update_yaxes(type="log", title_text="NN-predicted frequency (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="Signed percent error (%)", row=1, col=2)
    fig.update_yaxes(title_text="Sample-line count", row=1, col=2)

    fig.update_layout(
        title=(
            f"NN vs Minnaert  |  n={f_nn.size}  |  +/-5%: {p5:.1f}%"
            f"  |  +/-20%: {p20:.1f}%"
        ),
        width=1280,
        height=560,
        template="plotly_white",
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)


# ---------------------------------------------------------------------------
# Per-bubble plot
# ---------------------------------------------------------------------------


def _select_random_bubbles(
    bub_ids_arr: np.ndarray, allowed: set[int], num: int, seed: int
) -> list[int]:
    """Pick ``num`` bubble ids from ``allowed`` that actually appear in
    ``bub_ids_arr`` and have at least 2 samples (so we can draw a line)."""
    counts: dict[int, int] = {}
    for bid in bub_ids_arr:
        bid_i = int(bid)
        if bid_i in allowed:
            counts[bid_i] = counts.get(bid_i, 0) + 1
    eligible = sorted(b for b, c in counts.items() if c >= 2)
    if not eligible:
        return []
    rng = np.random.default_rng(int(seed))
    pick = min(int(num), len(eligible))
    chosen = rng.choice(np.array(eligible, dtype=np.int64), size=pick, replace=False)
    return sorted(int(x) for x in chosen)


def _per_bubble_traces(
    chosen: list[int],
    bub_ids_arr: np.ndarray,
    t_arr: np.ndarray,
    f_nn: np.ndarray,
    radii_by_bub: dict[int, float],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for bid in chosen:
        mask = bub_ids_arr == bid
        idx = np.where(mask)[0]
        order = np.argsort(t_arr[idx])
        t_b = t_arr[idx[order]]
        f_b = f_nn[idx[order]]
        good = np.isfinite(f_b) & (f_b > 0)
        t_b = t_b[good]
        f_b = f_b[good]
        if t_b.size == 0:
            continue
        r = float(radii_by_bub.get(bid, float("nan")))
        f_minn = (
            float(MINNAERT_CONSTANT) / r
            if (math.isfinite(r) and r > 0.0)
            else float("nan")
        )
        traces.append(
            {
                "bub_id": bid,
                "t": t_b,
                "f_nn": f_b,
                "radius": r,
                "f_minnaert": f_minn,
            }
        )
    return traces


def render_per_bubble_png(
    traces: list[dict[str, Any]],
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not traces:
        print("[per-bubble png] no eligible bubbles")
        return
    n = len(traces)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(13, 2.6 * rows), constrained_layout=True
    )
    axes_flat = np.atleast_1d(axes).reshape(-1)
    for i, tr in enumerate(traces):
        ax = axes_flat[i]
        ax.plot(
            tr["t"], tr["f_nn"], "-", color="#1f77b4", linewidth=1.2,
            marker="o", markersize=3, label="NN",
        )
        if math.isfinite(tr["f_minnaert"]):
            ax.axhline(
                tr["f_minnaert"], color="#d62728", linewidth=1.0, linestyle="--",
                label=f"Minnaert {tr['f_minnaert']:.1f} Hz",
            )
        r_str = f"{tr['radius']:.4g}" if math.isfinite(tr["radius"]) else "n/a"
        ax.set_title(f"Bub {tr['bub_id']}  |  r={r_str} m  |  n={tr['t'].size}",
                     fontsize=9)
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("f (Hz)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="best")
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(
        f"Per-bubble frequency: NN vs Minnaert  ({n} random NN-updated bubbles)",
        fontsize=11,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def render_per_bubble_html(
    traces: list[dict[str, Any]],
    out_html: Path,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # pragma: no cover
        print(f"[per-bubble html skip] plotly unavailable: {exc}")
        return
    if not traces:
        return
    n = len(traces)
    cols = 2
    rows = (n + cols - 1) // cols
    titles = []
    for tr in traces:
        r_str = f"{tr['radius']:.4g}" if math.isfinite(tr["radius"]) else "n/a"
        titles.append(f"Bub {tr['bub_id']} | r={r_str} m | n={tr['t'].size}")
    titles += [""] * (rows * cols - n)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
    )
    for i, tr in enumerate(traces):
        r = i // cols + 1
        c = i % cols + 1
        showleg = i == 0
        fig.add_trace(
            go.Scatter(
                x=tr["t"],
                y=tr["f_nn"],
                mode="lines+markers",
                name="NN",
                line=dict(color="#1f77b4", width=1.5),
                marker=dict(size=4),
                legendgroup="nn",
                showlegend=showleg,
                hovertemplate=(
                    f"<b>Bub {tr['bub_id']}</b><br>"
                    "t = %{x:.6g} s<br>f_NN = %{y:.6g} Hz<extra></extra>"
                ),
            ),
            row=r,
            col=c,
        )
        if math.isfinite(tr["f_minnaert"]) and tr["t"].size > 0:
            x0 = float(tr["t"].min())
            x1 = float(tr["t"].max())
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[tr["f_minnaert"], tr["f_minnaert"]],
                    mode="lines",
                    name="Minnaert C/r",
                    line=dict(color="#d62728", width=1.2, dash="dash"),
                    legendgroup="minnaert",
                    showlegend=showleg,
                    hovertemplate=(
                        f"<b>Bub {tr['bub_id']}</b><br>"
                        f"f_Minnaert = {tr['f_minnaert']:.6g} Hz<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )
        fig.update_xaxes(title_text="t (s)", row=r, col=c)
        fig.update_yaxes(title_text="f (Hz)", row=r, col=c)
    fig.update_layout(
        title=f"Per-bubble frequency: NN vs Minnaert ({n} random bubbles)",
        width=1280,
        height=320 * rows + 80,
        template="plotly_white",
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _summarize_top_bubbles(
    bub_ids_arr: np.ndarray,
    err_pct: np.ndarray,
    top_k: int,
) -> str:
    by_bub: dict[int, list[float]] = {}
    for bid, e in zip(bub_ids_arr, err_pct):
        if not math.isfinite(float(e)):
            continue
        by_bub.setdefault(int(bid), []).append(float(e))
    items: list[tuple[int, float, float, int]] = []
    for bid, errs in by_bub.items():
        a = np.asarray(errs, dtype=np.float64)
        if a.size == 0:
            continue
        items.append((bid, float(np.median(a)), float(np.mean(np.abs(a))), int(a.size)))
    items.sort(key=lambda r: -abs(r[1]))
    lines = [f"  Top-{top_k} bubbles by |median signed % error|:"]
    for bid, med, mae, n in items[:top_k]:
        lines.append(
            f"    Bub {bid:>6d}  n={n:>4d}  median_err={med:+8.2f}%  "
            f"mean_abs_err={mae:7.2f}%"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked-nn", type=Path, default=DEFAULT_TRACKED_NN)
    parser.add_argument(
        "--tracked-minnaert",
        type=Path,
        default=None,
        help=(
            "Optional companion trackedBubInfo file containing per-sample "
            "Minnaert frequencies (same sample-line structure as --tracked-nn). "
            "When omitted, Minnaert is computed on the fly as "
            "MINNAERT_CONSTANT / r using the per-bubble header radius."
        ),
    )
    parser.add_argument(
        "--meshes-root",
        type=Path,
        default=None,
        help=(
            "Mesh root used to identify NN-updated bub_ids by the presence of a "
            "``bub_<id>/`` directory. Optional when --tracked-minnaert is given "
            "(every paired sample line is then NN-updated by construction); "
            "required otherwise, since there is no portable default."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-prefix", type=str, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--num-bubbles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tracked = args.tracked_nn.resolve()
    out_dir = args.out_dir.resolve()
    if args.meshes_root is not None:
        meshes_root: Path | None = args.meshes_root.resolve()
    elif args.tracked_minnaert is None:
        raise SystemExit(
            "--meshes-root is required unless --tracked-minnaert is given: without "
            "either, there is no way to tell which bubbles the NN actually updated."
        )
    else:
        meshes_root = None  # mesh-root filter disabled

    if not tracked.is_file():
        raise SystemExit(f"missing tracked-nn: {tracked}")

    print(f"Loading {tracked} ...")
    bub_ids, t_arr, f_nn, radii_by_bub = parse_nn_samples(tracked)
    print(
        f"  {len(bub_ids)} sample lines; {len(radii_by_bub)} Bub headers with radius"
    )

    if meshes_root is not None:
        print(f"Scanning meshes: {meshes_root}")
        nn_updated_set = discover_nn_updated_bub_ids(meshes_root)
        print(f"  {len(nn_updated_set)} bub_<id>/ directories under meshes_root")
    else:
        nn_updated_set = None
        print(
            "Mesh-root filter disabled: every paired sample line is treated as "
            "NN-updated."
        )

    # Filter to NN-updated samples with positive radius and finite f_nn.
    radius_arr = np.array(
        [float(radii_by_bub.get(int(b), float("nan"))) for b in bub_ids],
        dtype=np.float64,
    )
    mask = (
        np.isfinite(f_nn)
        & (f_nn > 0.0)
        & np.isfinite(radius_arr)
        & (radius_arr > 0.0)
    )
    if nn_updated_set is not None:
        mask &= np.isin(bub_ids, np.fromiter(nn_updated_set, dtype=np.int64))
    n_used = int(mask.sum())
    print(f"  {n_used} NN-updated sample lines after filtering")
    if n_used == 0:
        raise SystemExit("no NN-updated sample lines to plot; aborting")

    if args.tracked_minnaert is not None:
        minn_path = args.tracked_minnaert.resolve()
        if not minn_path.is_file():
            raise SystemExit(f"missing tracked-minnaert: {minn_path}")
        print(f"Loading Minnaert source {minn_path} ...")
        f_minn_full = parse_minnaert_samples_aligned(minn_path, bub_ids, t_arr)
    else:
        f_minn_full = float(MINNAERT_CONSTANT) / np.where(
            (radius_arr > 0) & np.isfinite(radius_arr), radius_arr, np.nan
        )

    bub_ids_used = bub_ids[mask]
    t_used = t_arr[mask]
    f_nn_used = f_nn[mask]
    radius_used = radius_arr[mask]
    f_minn_used = f_minn_full[mask]
    err_pct = 100.0 * (f_nn_used - f_minn_used) / f_minn_used

    out_dir.mkdir(parents=True, exist_ok=True)
    agg_png = out_dir / f"{args.out_prefix}_aggregate.png"
    agg_html = out_dir / f"{args.out_prefix}_aggregate.html"
    pb_png = out_dir / f"{args.out_prefix}_per_bubble.png"
    pb_html = out_dir / f"{args.out_prefix}_per_bubble.html"

    print(f"Rendering aggregate PNG -> {agg_png}")
    render_aggregate_png(f_nn_used, f_minn_used, err_pct, radius_used, agg_png)
    print(f"Rendering aggregate HTML -> {agg_html}")
    render_aggregate_html(
        f_nn_used,
        f_minn_used,
        err_pct,
        radius_used,
        bub_ids_used,
        t_used,
        agg_html,
    )

    chosen = _select_random_bubbles(
        bub_ids_used, set(int(b) for b in np.unique(bub_ids_used)),
        args.num_bubbles, args.seed,
    )
    print(f"Per-bubble: chose bubbles {chosen}")
    traces = _per_bubble_traces(chosen, bub_ids_used, t_used, f_nn_used, radii_by_bub)
    print(f"Rendering per-bubble PNG -> {pb_png}")
    render_per_bubble_png(traces, pb_png)
    print(f"Rendering per-bubble HTML -> {pb_html}")
    render_per_bubble_html(traces, pb_html)

    median_err = float(np.median(err_pct))
    mean_err = float(np.mean(err_pct))
    pct5 = float(np.mean(np.abs(err_pct) <= 5.0) * 100.0)
    pct20 = float(np.mean(np.abs(err_pct) <= 20.0) * 100.0)
    n_bubs_used = int(np.unique(bub_ids_used).size)
    print()
    print("Summary:")
    print(f"  rows used:            {n_used}")
    print(f"  unique bubbles used:  {n_bubs_used}")
    print(f"  median signed err:    {median_err:+.3f}%")
    print(f"  mean signed err:      {mean_err:+.3f}%")
    print(f"  within +/-5%:         {pct5:.2f}%")
    print(f"  within +/-20%:        {pct20:.2f}%")
    print(_summarize_top_bubbles(bub_ids_used, err_pct, top_k=5))
    print(f"  outputs in {out_dir}")


if __name__ == "__main__":
    main()

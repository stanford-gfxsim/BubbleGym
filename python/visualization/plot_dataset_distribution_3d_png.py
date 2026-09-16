"""Render PNGs for old and convex-hull bubble shape spaces.

The two figures share the same frequency-ratio colour scale and camera angle.
By default, an interactive preview window opens first with both plots side by
side; rotate either plot and the other view follows. Closing the window writes
the two separate PNG files using the final synchronized camera.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from freq_model.analytical.minnaert_freq import minnaert_frequency_unit_volume  # noqa: E402


def numeric_column(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)


def wadell_phi(surface_area: np.ndarray, volume: np.ndarray) -> np.ndarray:
    phi = np.full_like(surface_area, np.nan, dtype=np.float64)
    ok = (
        np.isfinite(surface_area)
        & np.isfinite(volume)
        & (surface_area > 0.0)
        & (volume > 0.0)
    )
    phi[ok] = (np.pi ** (1.0 / 3.0)) * (6.0 * volume[ok]) ** (2.0 / 3.0) / surface_area[ok]
    return phi


def set_padded_3d_limits(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    def padded_limits(values: np.ndarray) -> tuple[float, float]:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        span = hi - lo
        if not np.isfinite(span) or span <= 0.0:
            span = max(abs(lo), 1.0)
        pad = 0.06 * span
        return lo - pad, hi + pad

    ax.set_xlim(*padded_limits(x))
    ax.set_ylim(*padded_limits(y))
    ax.set_zlim(*padded_limits(z))


def plot_shape_space_on_axis(
    *,
    ax,
    title: str,
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    axis_labels: tuple[str, str, str],
    freq_ratio: np.ndarray,
    color_min: float,
    color_max: float,
    elev: float,
    azim: float,
    marker_size: float,
    alpha: float,
    font_scale: float,
    font_bump: float = 0.0,
) -> tuple[object, int]:
    x, y, z = xyz
    ok = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & np.isfinite(freq_ratio)
        & (freq_ratio > 0.0)
    )
    if not np.any(ok):
        raise RuntimeError(f"No plottable rows for {title}")

    scatter = ax.scatter(
        x[ok],
        y[ok],
        z[ok],
        c=freq_ratio[ok],
        s=marker_size,
        cmap="rainbow",
        vmin=color_min,
        vmax=color_max,
        alpha=alpha,
        linewidths=0.0,
        depthshade=False,
    )

    if title:
        ax.set_title(title, pad=16, fontsize=14.0 * font_scale + font_bump)
    ax.set_xlabel(axis_labels[0], labelpad=10, fontsize=12.0 * font_scale + font_bump)
    ax.set_ylabel(axis_labels[1], labelpad=10, fontsize=12.0 * font_scale + font_bump)
    ax.set_zlabel(axis_labels[2], labelpad=10, fontsize=12.0 * font_scale + font_bump)
    ax.tick_params(axis="both", which="major", labelsize=10.0 * font_scale + font_bump)
    ax.view_init(elev=elev, azim=azim)
    set_padded_3d_limits(ax, x[ok], y[ok], z[ok])
    ax.grid(True, alpha=0.25)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    return scatter, int(ok.sum())


def sync_camera_angles(fig, axes: tuple[object, object]) -> None:
    """Keep two Matplotlib 3D axes at the same elev/azim while the user rotates."""

    syncing = {"active": False}

    def sync_from(source_ax) -> None:
        if syncing["active"]:
            return
        syncing["active"] = True
        try:
            for target_ax in axes:
                if target_ax is source_ax:
                    continue
                target_ax.view_init(elev=source_ax.elev, azim=source_ax.azim)
        finally:
            syncing["active"] = False
        fig.canvas.draw_idle()

    def on_motion(event) -> None:
        if event.inaxes in axes and getattr(event, "button", None) is not None:
            sync_from(event.inaxes)

    def on_release(event) -> None:
        if event.inaxes in axes:
            sync_from(event.inaxes)

    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)


def preview_synchronized_views(
    *,
    old_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    chull_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    freq_ratio: np.ndarray,
    color_min: float,
    color_max: float,
    elev: float,
    azim: float,
    marker_size: float,
    alpha: float,
    font_scale: float,
) -> tuple[float, float, int, int]:
    fig = plt.figure(figsize=(13.8, 6.2))
    old_ax = fig.add_subplot(121, projection="3d")
    chull_ax = fig.add_subplot(122, projection="3d")
    old_scatter, old_n = plot_shape_space_on_axis(
        ax=old_ax,
        title="Sphericity-Based Shape Space",
        xyz=old_xyz,
        axis_labels=(
            r"$1-\Phi_{VA}$",
            r"$1-\Phi_{VM}$",
            r"$1-\Phi_W$",
        ),
        freq_ratio=freq_ratio,
        color_min=color_min,
        color_max=color_max,
        elev=elev,
        azim=azim,
        marker_size=marker_size,
        alpha=alpha,
        font_scale=font_scale,
    )
    _, chull_n = plot_shape_space_on_axis(
        ax=chull_ax,
        title="Convex-Hull Shape Space",
        xyz=chull_xyz,
        axis_labels=(
            r"$\eta_V = V/V_{\mathrm{hull}}$",
            r"$\eta_A = A/A_{\mathrm{hull}}$",
            r"$\eta_M = M/M_{\mathrm{hull}}$",
        ),
        freq_ratio=freq_ratio,
        color_min=color_min,
        color_max=color_max,
        elev=elev,
        azim=azim,
        marker_size=marker_size,
        alpha=alpha,
        font_scale=font_scale,
    )
    cbar = fig.colorbar(old_scatter, ax=[old_ax, chull_ax], pad=0.04, shrink=0.78)
    cbar.set_label(r"Frequency ratio $f/f_M$", fontsize=12.0 * font_scale)
    cbar.ax.tick_params(labelsize=10.0 * font_scale)
    fig.suptitle(
        "Rotate either plot; camera angles are synchronized. Close the window to save PNGs.",
        y=0.98,
        fontsize=12.0 * font_scale,
    )
    sync_camera_angles(fig, (old_ax, chull_ax))
    plt.show()
    final_elev = float(old_ax.elev)
    final_azim = float(old_ax.azim)
    plt.close(fig)
    return final_elev, final_azim, old_n, chull_n


def save_shape_space_png(
    *,
    output: Path,
    title: str,
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    axis_labels: tuple[str, str, str],
    freq_ratio: np.ndarray,
    color_min: float,
    color_max: float,
    elev: float,
    azim: float,
    marker_size: float,
    alpha: float,
    dpi: int,
    font_scale: float,
    colorbar: bool = True,
    colorbar_pad: float = 0.12,
    font_bump: float = 0.0,
) -> int:
    fig = plt.figure(figsize=(7.2, 6.2), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    scatter, n_points = plot_shape_space_on_axis(
        ax=ax,
        title=title,
        xyz=xyz,
        axis_labels=axis_labels,
        freq_ratio=freq_ratio,
        color_min=color_min,
        color_max=color_max,
        elev=elev,
        azim=azim,
        marker_size=marker_size,
        alpha=alpha,
        font_scale=font_scale,
        font_bump=font_bump,
    )
    if colorbar:
        # At azim=135 the z-axis label sits where the bar wants to go: pad=0.09
        # overlaps it outright, 0.10 just touches, and every 0.01 above that buys
        # roughly a fifth of a bar-width of clearance. Checked against the
        # label's window extent rather than by eye.
        cbar = fig.colorbar(scatter, ax=ax, pad=colorbar_pad, shrink=0.72)
        cbar.set_label(r"Frequency ratio $f/f_M$", fontsize=12.0 * font_scale + font_bump)
        cbar.ax.tick_params(labelsize=10.0 * font_scale + font_bump)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # A 3D axis label sits outside the axes bbox, and bbox_inches="tight" does
    # not pick it up on its own -- without this the z label is silently cropped.
    fig.savefig(
        output,
        bbox_inches="tight",
        bbox_extra_artists=[ax.xaxis.label, ax.yaxis.label, ax.zaxis.label],
    )
    plt.close(fig)
    return n_points


def load_camera_state(path: Path) -> tuple[float, float] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        elev = float(data["elev"])
        azim = float(data["azim"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not (np.isfinite(elev) and np.isfinite(azim)):
        return None
    return elev, azim


def save_camera_state(path: Path, *, elev: float, azim: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "elev": float(elev),
        "azim": float(azim),
        "note": "Camera angle for python/visualization/plot_dataset_distribution_3d_png.py",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("dataset/bubble_gym/dataset_bubblegym_10k.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dataset_distribution_3d_png"))
    parser.add_argument("--old-output", type=Path, default=None)
    parser.add_argument("--chull-output", type=Path, default=None)
    parser.add_argument("--freq-col", default="frequency")
    parser.add_argument("--elev", type=float, default=None)
    parser.add_argument("--azim", type=float, default=None)
    parser.add_argument(
        "--no-panel-title",
        action="store_true",
        help="Drop the title from the two standalone panel PNGs. The published "
             "composite crops them, so a panel meant for it wants no title.",
    )
    parser.add_argument(
        "--panel-colorbar",
        choices=("both", "right-only", "none"),
        default="both",
        help="Which standalone panels carry a colorbar. 'right-only' leaves it "
             "on the convex-hull panel alone, which is where the composite's "
             "single shared bar sits.",
    )
    parser.add_argument(
        "--colorbar-pad",
        type=float,
        default=0.12,
        help="Gap between the 3D axes and the colorbar on a standalone panel. "
             "Below ~0.10 the bar lands on top of the z-axis label.",
    )
    parser.add_argument("--marker-size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-scale", type=float, default=1.18)
    parser.add_argument(
        "--font-bump",
        type=float,
        default=1.0,
        help="Points added to every label, tick and colorbar font AFTER "
             "--font-scale is applied. Additive rather than multiplicative so "
             "one step grows the small text as much as the large.",
    )
    parser.add_argument(
        "--camera-state",
        type=Path,
        default=None,
        help=(
            "JSON file used to persist the final camera angle. Defaults to "
            "<output-dir>/dataset_distribution_camera.json."
        ),
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip the interactive synchronized-camera preview and save immediately.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.lstrip("#").str.strip()

    required = [
        "surface_area",
        "volume",
        "Phi_VM",
        "W_vertex",
        "eta_V",
        "eta_A",
        "eta_M",
        args.freq_col,
    ]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"{args.csv} missing columns: {missing}")

    surface_area = numeric_column(df, "surface_area")
    volume = numeric_column(df, "volume")
    phi_va = wadell_phi(surface_area, volume)
    phi_vm = numeric_column(df, "Phi_VM")
    phi_w = 4.0 * np.pi / numeric_column(df, "W_vertex")

    old_xyz = (
        np.where(
            np.isfinite(phi_va) & (phi_va > 0.0) & (phi_va <= 1.0 + 1e-9),
            1.0 - phi_va,
            np.nan,
        ),
        1.0 - phi_vm,
        1.0 - phi_w,
    )
    chull_xyz = (
        numeric_column(df, "eta_V"),
        numeric_column(df, "eta_A"),
        numeric_column(df, "eta_M"),
    )

    freq_hz = numeric_column(df, args.freq_col)
    f_minnaert = float(minnaert_frequency_unit_volume())
    freq_ratio = freq_hz / f_minnaert
    freq_ok = np.isfinite(freq_ratio) & (freq_ratio > 0.0)
    color_min = float(np.nanmin(freq_ratio[freq_ok]))
    color_max = float(np.nanmax(freq_ratio[freq_ok]))

    old_output = args.old_output or (args.output_dir / "dataset_bubblegym_10k_old_nonsphericity.png")
    chull_output = args.chull_output or (args.output_dir / "dataset_bubblegym_10k_chull_nonsphericity.png")
    camera_state = args.camera_state or (args.output_dir / "dataset_distribution_camera.json")

    saved_camera = load_camera_state(camera_state)
    if args.elev is not None:
        final_elev = float(args.elev)
    elif saved_camera is not None:
        final_elev = saved_camera[0]
    else:
        final_elev = 25.0

    if args.azim is not None:
        final_azim = float(args.azim)
    elif saved_camera is not None:
        final_azim = saved_camera[1]
    else:
        final_azim = 45.0

    if saved_camera is not None and args.elev is None and args.azim is None:
        print(f"Loaded saved camera: elev={final_elev:.3g}, azim={final_azim:.3g} from {camera_state}")
    old_n = chull_n = 0
    if not args.no_preview:
        print("Opening synchronized 3D preview. Rotate either subplot, then close the window to save PNGs.")
        final_elev, final_azim, old_n, chull_n = preview_synchronized_views(
            old_xyz=old_xyz,
            chull_xyz=chull_xyz,
            freq_ratio=freq_ratio,
            color_min=color_min,
            color_max=color_max,
            elev=final_elev,
            azim=final_azim,
            marker_size=args.marker_size,
            alpha=args.alpha,
            font_scale=args.font_scale,
        )

    old_n = save_shape_space_png(
        output=old_output,
        title="" if args.no_panel_title else "Bubble Dataset: Sphericity-Based Shape Space",
        colorbar=args.panel_colorbar == "both",
        xyz=old_xyz,
        axis_labels=(
            r"$1-\Phi_{VA}$",
            r"$1-\Phi_{VM}$",
            r"$1-\Phi_W$",
        ),
        freq_ratio=freq_ratio,
        color_min=color_min,
        color_max=color_max,
        elev=final_elev,
        azim=final_azim,
        marker_size=args.marker_size,
        alpha=args.alpha,
        dpi=args.dpi,
        font_scale=args.font_scale,
        font_bump=args.font_bump,
    )
    chull_n = save_shape_space_png(
        output=chull_output,
        title="" if args.no_panel_title else "Bubble Dataset: Convex-Hull Shape Space",
        colorbar=args.panel_colorbar != "none",
        colorbar_pad=args.colorbar_pad,
        xyz=chull_xyz,
        axis_labels=(
            r"$\eta_V = V/V_{\mathrm{hull}}$",
            r"$\eta_A = A/A_{\mathrm{hull}}$",
            r"$\eta_M = M/M_{\mathrm{hull}}$",
        ),
        freq_ratio=freq_ratio,
        color_min=color_min,
        color_max=color_max,
        elev=final_elev,
        azim=final_azim,
        marker_size=args.marker_size,
        alpha=args.alpha,
        dpi=args.dpi,
        font_scale=args.font_scale,
        font_bump=args.font_bump,
    )

    print(f"Minnaert unit-volume f_M = {f_minnaert:.6f} Hz")
    print(f"Shared colour range: f/f_M in [{color_min:.6g}, {color_max:.6g}]")
    print(f"Final camera: elev={final_elev:.3g}, azim={final_azim:.3g}")
    save_camera_state(camera_state, elev=final_elev, azim=final_azim)
    print(f"Saved camera to {camera_state}")
    print(f"Wrote {old_n} points to {old_output}")
    print(f"Wrote {chull_n} points to {chull_output}")


if __name__ == "__main__":
    main()

"""Regression check: P1-DP0 Galerkin on the shipped procedural-ellipsoid track.

Every ``bub_NNNN.obj`` in ``--mesh-dir`` is solved at unit volume, then mapped
back to physical Hz using the equivalent-sphere radius recorded in the
``Bub <id> <radius>`` header of the reference ``trackedBubInfo`` file, since the
OBJ sequence is stored in simulation units. Under a uniform scale ``s`` the
Minnaert frequency goes as ``1/s``, so for a bubble of equivalent radius ``r``::

    f_physical = f_unit_volume / ( (4 pi / 3)^(1/3) * r )

The reference file is optional: without it the script just solves and times.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_ROOT.parent
for _p in (REPO_ROOT, PYTHON_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bem._threading import default_thread_count, limit_threads  # noqa: E402

DEFAULT_MESH_DIR = (
    REPO_ROOT / "dataset" / "bubble_theater" / "ellipsoid" / "mesh"
)
DEFAULT_REFERENCE = (
    REPO_ROOT / "dataset" / "bubble_theater" / "ellipsoid" / "trackedBubInfo_BEM.txt"
)
DEFAULT_TIMING_CSV = (
    REPO_ROOT / "results" / "ellipsoid_validation" / "validate_galerkin_timing.csv"
)

# V = (4/3) pi r^3, so the linear scale of a bubble of radius r relative to a
# unit-volume one is V^(1/3) = (4 pi / 3)^(1/3) * r.
_UNIT_VOLUME_SCALE_PER_RADIUS = (4.0 * math.pi / 3.0) ** (1.0 / 3.0)

_BUB_STEM_RE = re.compile(r"^bub_(\d+)$", re.IGNORECASE)
_REL_ERR_WARN_PCT = 2.0


def _mesh_curve_index(path: Path) -> int:
    m = _BUB_STEM_RE.match(path.stem)
    if not m:
        raise ValueError(f"expected bub_NNNN.obj name, got {path.name}")
    return int(m.group(1)) - 1


def _discover_meshes(mesh: Path | None, mesh_dir: Path) -> list[Path]:
    if mesh is not None:
        return [mesh.resolve()]
    if not mesh_dir.is_dir():
        return []
    return sorted(mesh_dir.glob("bub_*.obj"))


def _load_reference_by_k(ref_path: Path) -> dict[int, tuple[float, float]]:
    """Return ``{k: (f_reference_hz, equivalent_radius_m)}`` for each sample line.

    ``k`` is the position of the sample line within the file, which is the same
    ordering as ``bub_0001.obj, bub_0002.obj, ...`` in the mesh folder. The
    radius comes from the enclosing ``Bub <id> <radius>`` header and is what
    converts a unit-volume solve back into physical Hz.
    """
    from tracked_bubinfo.io import (
        parse_bub_header_radii,
        parse_trackedbubinfo_blocks,
    )

    lines, sample_idxs, meta = parse_trackedbubinfo_blocks(ref_path)
    radii = parse_bub_header_radii(lines)
    out: dict[int, tuple[float, float]] = {}
    for k, (line_idx, (bub_id, _t, _y)) in enumerate(zip(sample_idxs, meta)):
        parts = lines[line_idx].split()
        if len(parts) < 2:
            continue
        try:
            f_ref = float(parts[1])
        except ValueError:
            continue
        r = float(radii.get(bub_id, float("nan")))
        if math.isfinite(f_ref) and math.isfinite(r) and r > 0.0:
            out[k] = (f_ref, r)
    return out


def _timing_from_solve(out: dict) -> tuple[float, float, float]:
    t_asm = float(out.get("wall_time_assemble_s", 0.0))
    t_solve = float(out.get("wall_time_solve_s", 0.0))
    return t_asm, t_solve, t_asm + t_solve


def _write_timing_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mesh",
        "k",
        "f_new_hz",
        "f_ref_hz",
        "rel_err_pct",
        "t_assemble_s",
        "t_solve_s",
        "t_total_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Solve a single mesh only (default: all bub_*.obj in --mesh-dir).",
    )
    p.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_MESH_DIR,
        help=f"Folder of ellipsoid OBJ sequences (default: {DEFAULT_MESH_DIR}).",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=(
            "trackedBubInfo file holding the reference frequencies and the "
            f"per-bubble radius (default: {DEFAULT_REFERENCE}). Optional: when "
            "it is absent the script only solves and times."
        ),
    )
    p.add_argument("--threads", type=int, default=None)
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-solve Galerkin diagnostics (default: summary lines only).",
    )
    p.add_argument(
        "--timing-csv",
        type=Path,
        default=DEFAULT_TIMING_CSV,
        help=(
            "Write per-mesh wall times and frequencies to this CSV "
            f"(default: {DEFAULT_TIMING_CSV})."
        ),
    )
    args = p.parse_args()

    limit_threads(args.threads if args.threads is not None else default_thread_count())

    meshes = _discover_meshes(args.mesh, args.mesh_dir.resolve())
    if not meshes:
        if args.mesh is not None:
            print(f"error: mesh not found: {args.mesh}", file=sys.stderr)
        else:
            print(f"error: no bub_*.obj in {args.mesh_dir}", file=sys.stderr)
        return 2

    try:
        from bem.compute_freq_bempp_galerkin import solve_minnaert_frequency_galerkin
    except ModuleNotFoundError as exc:
        print(f"bempp not available: {exc}", file=sys.stderr)
        return 3

    ref_path = args.reference.resolve()
    ref_by_k: dict[int, tuple[float, float]] = {}
    if ref_path.is_file():
        ref_by_k = _load_reference_by_k(ref_path)
        if ref_by_k:
            k0 = min(ref_by_k)
            f0, r0 = ref_by_k[k0]
            print(
                f"Reference from {ref_path.name}: {len(ref_by_k)} sample(s), "
                f"k={k0} -> {f0:.6g} Hz at equivalent radius {r0:.6g} m"
            )
        else:
            print(f"warn: no usable sample lines in {ref_path}", file=sys.stderr)
    else:
        print(
            f"note: no reference file at {ref_path}; solving and timing only "
            "(pass --reference to compare against recorded frequencies).",
            file=sys.stderr,
        )

    print(f"Solving {len(meshes)} mesh(es) with P1-DP0 Galerkin ...")

    rows: list[tuple[str, int, float, float | None, float | None, float, float, float]] = []
    timing_records: list[dict] = []
    for i, mesh_path in enumerate(meshes, start=1):
        k = _mesh_curve_index(mesh_path)
        out = solve_minnaert_frequency_galerkin(
            str(mesh_path),
            trial_pair="P1-DP0",
            precond="mass",
            solver="gmres",
            verbose=args.verbose,
        )
        f_unit = float(out["frequency"])
        t_asm, t_solve, t_total = _timing_from_solve(out)

        ref = ref_by_k.get(k)
        f_ref: float | None = None
        rel: float | None = None
        if ref is not None:
            f_ref, r_eq = ref
            # The OBJ sequence is in simulation units, so report the solve at
            # the physical scale the reference file records.
            f_new = f_unit / (_UNIT_VOLUME_SCALE_PER_RADIUS * r_eq)
            rel = abs(f_new - f_ref) / max(abs(f_ref), 1e-12) * 100.0
        else:
            f_new = f_unit

        rows.append((mesh_path.name, k, f_new, f_ref, rel, t_asm, t_solve, t_total))
        timing_records.append(
            {
                "mesh": mesh_path.name,
                "k": k,
                "f_new_hz": f_new,
                "f_ref_hz": "" if f_ref is None else f_ref,
                "rel_err_pct": "" if rel is None else rel,
                "t_assemble_s": t_asm,
                "t_solve_s": t_solve,
                "t_total_s": t_total,
            }
        )

        time_str = (
            f"t_asm={t_asm:.3f}s  t_solve={t_solve:.3f}s  t_total={t_total:.3f}s"
        )
        if rel is not None:
            status = "WARN" if rel > _REL_ERR_WARN_PCT else "ok"
            print(
                f"[{i}/{len(meshes)}] {mesh_path.name}  k={k}  "
                f"f_ref={f_ref:.6g} Hz  f_new={f_new:.6g} Hz  "
                f"rel={rel:.4f}%  {time_str}  {status}"
            )
        else:
            print(
                f"[{i}/{len(meshes)}] {mesh_path.name}  k={k}  "
                f"f_new={f_new:.6g} Hz (at V = 1 m^3; no reference radius to "
                f"rescale with)  {time_str}"
            )

    timing_csv = args.timing_csv.resolve()
    _write_timing_csv(timing_csv, timing_records)
    print(f"Wrote per-mesh timings to {timing_csv}")

    totals = np.array([r[7] for r in rows], dtype=np.float64)
    slow_i = int(np.argmax(totals))
    slow = rows[slow_i]
    print(
        f"Timing: total={float(np.sum(totals)):.3f}s  "
        f"mean={float(np.mean(totals)):.3f}s  "
        f"slowest={slow[0]} (t_total={slow[7]:.3f}s)"
    )

    compared = [r for r in rows if r[4] is not None]
    if compared:
        rels = np.array([r[4] for r in compared], dtype=np.float64)
        worst_i = int(np.argmax(rels))
        worst = compared[worst_i]
        print(
            f"\nSummary ({len(compared)} compared): "
            f"median rel={float(np.median(rels)):.4f}%  "
            f"max rel={float(np.max(rels)):.4f}%  "
            f"worst={worst[0]} (k={worst[1]}, rel={worst[4]:.4f}%)"
        )
        n_warn = int(np.sum(rels > _REL_ERR_WARN_PCT))
        if n_warn:
            print(f"WARN: {n_warn} mesh(es) with relative error > {_REL_ERR_WARN_PCT:g}%")
            return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

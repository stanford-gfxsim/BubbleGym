"""Drop unusable-geometry bubbles from a trackedBubInfo *before* any estimator.

An open marching-cubes surface -- a sheet, or a cap clipped by the domain
boundary -- encloses no volume, so it has no equivalent radius, no capacitance
and no resonant frequency under any model. Filtering once here, instead of in
each estimator, keeps every estimator on the same bubble population and makes a
downstream failure a real anomaly rather than expected fallout.

``--check watertight`` (always on) uses edge incidence: a closed manifold shares
every edge between exactly two triangles. ``--check geometry`` additionally
catches closed-but-degenerate meshes, whose zero-area triangles make the
cotangent Laplacian -- and every Willmore-derived feature -- NaN or non-positive;
topology alone cannot see this, so it runs the pipeline's own shape-feature
integrals, at ~10 ms per mesh. ``--missing-mesh drop`` also drops ``Bub`` blocks
with no ``bub_<id>/`` directory (default ``keep``).

``--granularity sample`` (default) drops only the sample lines whose nearest
mesh frame is unusable; ``bubble`` drops the whole block if any frame is. Either
way a ``Bub`` block left with no samples is removed entirely.

Usage::

    python python/scene/_common/drop_nonwatertight_bubbles.py \\
        --tracked dataset/exhalation/trackedBubInfo.txt \\
        --meshes-root <.../mc_surface_per_bubble_smoothed_lap3> \\
        --check geometry --missing-mesh drop \\
        --out dataset/exhalation/trackedBubInfo_closed.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from tracked_bubinfo import (  # noqa: E402
    discover_mesh_jobs,
    infer_dt_lbm,
    parse_trackedbubinfo_blocks,
    truncate_and_rewrite_blocks,
    write_tracked_lines,
)
from shape_feature.mesh_utils import (  # noqa: E402
    edge_incidence_report,
    load_obj_mesh,
    vertices_scaled_to_target_volume,
)
from shape_feature.nonspherical_features import (  # noqa: E402
    curvature_integrals_from_vf,
    sphericities_from_integrals,
)


def inspect_mesh(job: tuple[str, bool]) -> tuple[str, bool, str, int, int]:
    """Return ``(path, usable, reason, n_boundary_edges, n_nonmanifold_edges)``.

    ``job`` is ``(obj_path, check_geometry)``. The mesh loads without the
    watertight gate so the edge report can be produced rather than raised; a
    mesh that fails to parse counts as unusable.

    With ``check_geometry`` the closed meshes additionally run the shape-feature
    integrals, mirroring what ``shape_feature.build_dataset.extract_features_one``
    does with an already-smoothed mesh: scale to unit volume, integrate, and
    derive the sphericities. That last step is what raises on a NaN or
    non-positive ``W_vertex``, i.e. on the zero-area triangles left behind by
    duplicated or collapsed vertices.
    """
    obj_path, check_geometry = job
    try:
        v, f = load_obj_mesh(Path(obj_path), require_watertight=False)
    except Exception as exc:  # noqa: BLE001 - unreadable mesh is not a bubble
        return obj_path, False, f"unreadable: {type(exc).__name__}", -1, -1

    rep = edge_incidence_report(f)
    n_b = int(rep["n_boundary_edges"])
    n_nm = int(rep["n_nonmanifold_edges"])
    if n_b or n_nm:
        return obj_path, False, "not closed", n_b, n_nm
    if not check_geometry:
        return obj_path, True, "ok", n_b, n_nm

    try:
        v_unit = vertices_scaled_to_target_volume(v, f, 1.0)
        curv = curvature_integrals_from_vf(v_unit, f, target_volume=None)
        sphericities_from_integrals(
            volume=float(curv["volume"]),
            area=float(curv["area"]),
            mean_curvature=float(curv["M"]),
            willmore_vertex=float(curv["W_vertex"]),
        )
    except Exception as exc:  # noqa: BLE001 - degenerate geometry, not a bubble
        return obj_path, False, f"degenerate: {type(exc).__name__}", n_b, n_nm
    return obj_path, True, "ok", n_b, n_nm


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tracked", type=Path, required=True)
    ap.add_argument("--meshes-root", type=Path, required=True,
                    help="Directory of bub_<id>/frame_<NNNNN>.obj.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--granularity", choices=["sample", "bubble"], default="sample")
    ap.add_argument(
        "--check",
        choices=["watertight", "geometry"],
        default="watertight",
        help="'watertight' tests edge incidence only. 'geometry' additionally "
             "runs the shape-feature integrals on the closed meshes and drops "
             "the frames whose Willmore energy comes out NaN or non-positive -- "
             "the signature of zero-area triangles from duplicated or collapsed "
             "vertices. Slower (~10 ms/mesh) but catches degeneracies topology "
             "cannot see.",
    )
    ap.add_argument(
        "--missing-mesh",
        choices=["keep", "drop"],
        default="keep",
        help="What to do with samples of a bubble that has no mesh directory at "
             "all. 'keep' (default) leaves the decision to the estimator. "
             "'drop' removes them, so every surviving sample is backed by real "
             "geometry.",
    )
    ap.add_argument("--sample-stride", type=int, default=50)
    ap.add_argument("--dt-lbm", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report-csv", type=Path, default=None,
                    help="Optional per-mesh watertightness report.")
    args = ap.parse_args()

    tracked = args.tracked.resolve()
    meshes_root = args.meshes_root.resolve()
    for p in (tracked, meshes_root):
        if not p.exists():
            raise SystemExit(f"not found: {p}")

    src_lines, sample_idxs, sample_meta = parse_trackedbubinfo_blocks(tracked)
    n_samples = len(sample_idxs)
    raw_jobs, job_meta, bub_ids_with_meshes = discover_mesh_jobs(meshes_root)
    print(f"tracked samples: {n_samples:,} | meshes: {len(raw_jobs):,} "
          f"| bubbles with meshes: {len(bub_ids_with_meshes):,}")

    dt_lbm = (
        float(args.dt_lbm)
        if args.dt_lbm is not None
        else infer_dt_lbm(sample_meta, int(args.sample_stride))
    )

    check_geometry = args.check == "geometry"
    what = "watertightness + geometry" if check_geometry else "watertightness"
    print(f"checking {what} (workers={args.workers}) ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(
            ex.map(
                inspect_mesh,
                [(p, check_geometry) for _b, p in raw_jobs],
                chunksize=64,
            )
        )
    usable_of = {p: ok for p, ok, _r, _b, _nm in results}
    reasons = Counter(r for _p, ok, r, _b, _nm in results if not ok)
    n_bad = sum(reasons.values())
    print(f"  {n_bad:,}/{len(results):,} meshes are unusable "
          f"({100.0 * n_bad / max(len(results), 1):.2f}%)")
    for reason, count in reasons.most_common():
        print(f"    {reason:<32} {count:,}")

    if args.report_csv is not None:
        import pandas as pd

        args.report_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"path": p, "usable": ok, "reason": r, "n_boundary_edges": b,
              "n_nonmanifold_edges": nm} for p, ok, r, b, nm in results]
        ).to_csv(args.report_csv, index=False)
        print(f"  wrote {args.report_csv}")

    # Per bubble: the frames it has, and whether each one is usable.
    frames_of: dict[int, list[int]] = {}
    ok_of: dict[tuple[int, int], bool] = {}
    for (bub_id, path), (b2, frame_idx) in zip(raw_jobs, job_meta):
        frames_of.setdefault(int(b2), []).append(int(frame_idx))
        ok_of[(int(b2), int(frame_idx))] = usable_of.get(path, False)
    for b in frames_of:
        frames_of[b].sort()

    bad_bubbles = {
        b for b, fr in frames_of.items() if any(not ok_of[(b, f)] for f in fr)
    }

    drop_mask = np.zeros(n_samples, dtype=bool)
    n_no_mesh = 0
    for i, (bub_id, t_s, _y) in enumerate(sample_meta):
        frames = frames_of.get(int(bub_id))
        if not frames:
            # No mesh at all for this bubble. Under --missing-mesh keep, leave
            # the decision to the estimator; under drop, it is not a bubble we
            # can say anything about from geometry.
            n_no_mesh += 1
            if args.missing_mesh == "drop":
                drop_mask[i] = True
            continue
        if args.granularity == "bubble":
            if int(bub_id) in bad_bubbles:
                drop_mask[i] = True
            continue
        arr = np.asarray(frames, dtype=np.float64)
        nearest = int(frames[int(np.argmin(np.abs(arr * dt_lbm - float(t_s))))])
        if not ok_of[(int(bub_id), nearest)]:
            drop_mask[i] = True

    # This filter only removes lines; it must not touch the frequency column.
    # truncate_and_rewrite_blocks always rewrites that field, so feed it the
    # original tokens to make the rewrite a no-op for every surviving sample.
    orig_freq_tokens = [
        (src_lines[li].split()[1] if len(src_lines[li].split()) >= 6 else "nan")
        for li in sample_idxs
    ]
    out_lines, n_blocks_dropped, n_samples_dropped = truncate_and_rewrite_blocks(
        src_lines, sample_idxs, drop_mask, orig_freq_tokens, float("inf")
    )
    write_tracked_lines(args.out, out_lines)

    print()
    print("Summary:")
    print(f"  check:                    {args.check}")
    print(f"  granularity:              {args.granularity}")
    print(f"  bubbles with a bad mesh:  {len(bad_bubbles):,}")
    print(f"  samples with no mesh:     {n_no_mesh:,} ({args.missing_mesh})")
    print(f"  sample lines dropped:     {n_samples_dropped:,} of {n_samples:,} "
          f"({100.0 * n_samples_dropped / max(n_samples, 1):.2f}%)")
    print(f"  Bub blocks dropped:       {n_blocks_dropped:,}")
    print(f"  wrote:                    {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

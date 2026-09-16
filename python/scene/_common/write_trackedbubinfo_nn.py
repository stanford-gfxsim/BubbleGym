"""Rewrite a ``trackedBubInfo.txt`` with the BubbleGym shape-to-frequency NN.

Each per-bubble marching-cubes mesh is reduced to its largest connected
component, winding-fixed and Laplacian-smoothed (cached under
``--smoothed-cache-root``), reduced to the 8-feature descriptor of Sec. 4.2 and
pushed through the surrogate (Eq. 16); the result replaces the tracker's own
frequency. Sister drivers beside it: ``write_trackedbubinfo_bem.py`` and
``replace_trackedbubinfo_freq_with_minnaert.py``.

TRAP: ``frame_NNNNN`` in an OBJ name is the **LBM iteration index**, in
multiples of ``--sample-stride``, not a render frame. ``dt_lbm`` resolves
``--dt-lbm`` > ``--dt-frame / sample_stride`` > auto-inference. Two checks run
before the expensive feature pool: every ``frame_idx`` must be a multiple of
``sample_stride`` (abort), and median ``|t_sample - frame_idx * dt_lbm|`` on the
longest bubble must be ``< 0.5 * dt_frame`` (warn).

``--missing-fill`` defaults to ``drop`` because a ``nan`` frequency is not an
inert marker downstream: one NaN oscillator poisons a whole bubble's
contribution to the mix. Non-watertight bubbles are dropped earlier, by
``drop_nonwatertight_bubbles.py``.
"""

from __future__ import annotations

# IMPORTANT: torch must be imported BEFORE scipy / igl etc. on Windows + the
# soundlab conda env, or torch fails to resolve shm.dll's OpenMP/MKL deps.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch as _TORCH  # type: ignore  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

PYTHON_ROOT = Path(__file__).resolve().parents[1]
FREQ_MODEL_ROOT = PYTHON_ROOT / "freq_model"
BASELINE_ROOT = PYTHON_ROOT / "baseline"
for _p in (PYTHON_ROOT, FREQ_MODEL_ROOT, BASELINE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from freq_model.analytical.minnaert_freq import MINNAERT_CONSTANT  # noqa: E402
from freq_model.NN.nn_inference import (  # noqa: E402
    predict_nn_unit_for_frames_inertia_8feat_direct,
)
from shape_feature.build_dataset import (  # noqa: E402
    extract_features_one,
    run_feature_pool,
)
from tracked_bubinfo import (  # noqa: E402
    MISSING_POLICIES,
    MISSING_POLICY_HELP,
    resolve_missing_samples,
    build_per_bubble_index,
    build_smoothing_jobs,
    check_stride_and_alignment,
    discover_mesh_jobs,
    format_freq,
    infer_dt_lbm,
    parse_bub_header_radii,
    parse_trackedbubinfo_blocks,
    summarize_freqs,
    truncate_and_rewrite_blocks,
)

REPO_ROOT = PYTHON_ROOT.parent
DEFAULT_ARTIFACTS = (
    FREQ_MODEL_ROOT / "output" / "output_8feature_direct_bubblegym_10k"
)
DEFAULT_SAMPLE_STRIDE = 50
DEFAULT_SMOOTH_ITERS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracked",
        type=Path,
        required=True,
        help="Input WaveBlender-format trackedBubInfo.txt.",
    )
    parser.add_argument(
        "--meshes-root",
        type=Path,
        required=True,
        help="Directory holding bub_<id>/frame_<NNNNN>.obj from the simulator.",
    )
    parser.add_argument(
        "--smoothed-cache-root",
        type=Path,
        default=None,
        help=(
            "Where to cache Laplacian-smoothed OBJs; mirrors the "
            "bub_<id>/frame_<NNNNN>.obj layout of --meshes-root. Defaults to a "
            "'<meshes-root>_smoothed_lap<N>' sibling directory."
        ),
    )
    parser.add_argument("--smooth-iters", type=int, default=DEFAULT_SMOOTH_ITERS)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--out", type=Path, required=True, help="Output trackedBubInfo path."
    )

    parser.add_argument(
        "--sample-stride",
        type=int,
        default=DEFAULT_SAMPLE_STRIDE,
        help=(
            "LBM iteration stride between consecutive sample lines AND "
            "consecutive mesh frames (matches --bub-track-sample-stride / "
            "--bub-mesh-iter-stride from the simulator)."
        ),
    )
    parser.add_argument(
        "--dt-lbm",
        type=float,
        default=None,
        help=(
            "Seconds per single LBM iteration. If omitted we derive it from "
            "--dt-frame or, failing that, the median consecutive sample dt in "
            "the trackedBubInfo file divided by --sample-stride."
        ),
    )
    parser.add_argument(
        "--dt-frame",
        type=float,
        default=None,
        help="Optional: seconds per stored mesh frame (= --sample-stride * --dt-lbm).",
    )

    parser.add_argument(
        "--missing-fill",
        type=str,
        default="drop",
        choices=list(MISSING_POLICIES),
        help=(
            "What to do with a sample the NN could not predict -- a bubble with "
            "no mesh at all, or a mesh the feature pipeline rejected. The "
            "default drops the sample line, because a 'nan' frequency is not an "
            "inert marker downstream: FluidSound turns it into a NaN oscillator "
            "that poisons the whole audio mix from that bubble's start time "
            "onward. " + MISSING_POLICY_HELP
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frequency-format", type=str, default="{:.6f}")
    parser.add_argument(
        "--feature-chunksize",
        type=int,
        default=0,
        help="ProcessPoolExecutor.map chunksize. 0 (default) = auto.",
    )
    parser.add_argument(
        "--limit-bubs",
        type=int,
        default=0,
        help="Debug: process at most this many bubble dirs (0 = all).",
    )
    args = parser.parse_args()

    tracked = args.tracked.resolve()
    meshes_root = args.meshes_root.resolve()
    smoothed_root = (
        args.smoothed_cache_root.resolve()
        if args.smoothed_cache_root is not None
        else meshes_root.with_name(
            f"{meshes_root.name}_smoothed_lap{int(args.smooth_iters)}"
        )
    )
    artifacts = args.artifacts.resolve()
    out_path = args.out.resolve()

    if not tracked.is_file():
        raise SystemExit(f"missing tracked: {tracked}")
    if not meshes_root.is_dir():
        raise SystemExit(f"missing meshes_root: {meshes_root}")
    if not artifacts.is_dir():
        raise SystemExit(f"missing artifacts: {artifacts}")
    smoothed_root.mkdir(parents=True, exist_ok=True)

    # ---- 1. Parse trackedBubInfo --------------------------------------
    print(f"Parsing trackedBubInfo: {tracked}")
    src_lines, sample_idxs, sample_meta = parse_trackedbubinfo_blocks(tracked)
    bub_radii = parse_bub_header_radii(src_lines)
    print(
        f"  {len(sample_idxs)} sample lines | {len(bub_radii)} Bub headers "
        f"(radius extracted)"
    )

    # ---- 2. Resolve dt_lbm --------------------------------------------
    sample_stride = int(args.sample_stride)
    if sample_stride <= 0:
        raise SystemExit(f"--sample-stride must be positive, got {sample_stride}")

    if args.dt_lbm is not None:
        dt_lbm = float(args.dt_lbm)
        dt_lbm_source = "--dt-lbm"
    elif args.dt_frame is not None:
        dt_lbm = float(args.dt_frame) / float(sample_stride)
        dt_lbm_source = (
            f"--dt-frame / sample_stride ({args.dt_frame} / {sample_stride})"
        )
    else:
        try:
            dt_lbm = infer_dt_lbm(sample_meta, sample_stride)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        dt_lbm_source = "auto-inferred from trackedBubInfo"

    if not (math.isfinite(dt_lbm) and dt_lbm > 0.0):
        raise SystemExit(f"resolved dt_lbm is non-positive/non-finite: {dt_lbm}")
    dt_frame = dt_lbm * float(sample_stride)
    print(
        f"  dt_lbm = {dt_lbm:.9g} s ({dt_lbm_source}); "
        f"dt_frame = sample_stride * dt_lbm = {dt_frame:.9g} s"
    )

    # ---- 3. Discover meshes -------------------------------------------
    print(f"Scanning meshes: {meshes_root}")
    raw_jobs, job_meta, bub_ids_with_meshes = discover_mesh_jobs(meshes_root)
    if int(args.limit_bubs) > 0:
        keep = set(sorted(bub_ids_with_meshes)[: int(args.limit_bubs)])
        rj2: list[tuple[int, str]] = []
        meta2: list[tuple[int, int]] = []
        for (_orig_idx, path), (bub_id, frame_idx) in zip(raw_jobs, job_meta):
            if bub_id in keep:
                rj2.append((len(rj2), path))
                meta2.append((bub_id, frame_idx))
        raw_jobs, job_meta, bub_ids_with_meshes = rj2, meta2, keep

    if not raw_jobs:
        raise SystemExit(f"no meshes discovered under {meshes_root}")

    print(
        f"  {len(raw_jobs)} OBJs across {len(bub_ids_with_meshes)} bubbles "
        f"(of {len(bub_radii)} bubbles in trackedBubInfo)"
    )

    # ---- 4. Safeguards ------------------------------------------------
    try:
        check_stride_and_alignment(job_meta, sample_meta, sample_stride, dt_lbm)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Per-frame cutoff: drop sample lines past the last available mesh time.
    global_max_frame = int(max(frame_idx for _bub_id, frame_idx in job_meta))
    t_max = float(global_max_frame) * float(dt_lbm)
    print(
        f"  global_max_frame = {global_max_frame} | "
        f"t_max = {t_max:.9g} s (= dt_lbm * max_frame)"
    )

    eps = 0.5 * float(dt_frame)
    sample_t = np.array([t for (_b, t, _y) in sample_meta], dtype=np.float64)
    drop_mask = sample_t > (t_max + eps)
    print(
        f"  {int(drop_mask.sum())}/{len(sample_meta)} sample lines past t_max "
        f"will be dropped"
    )

    # ---- 5. Feature extraction (parallel; smooths + caches) -----------
    jobs = build_smoothing_jobs(
        raw_jobs, job_meta, meshes_root, smoothed_root, int(args.smooth_iters)
    )
    feats_list = run_feature_pool(
        extract_features_one,
        jobs,
        workers=int(args.workers),
        chunksize=int(args.feature_chunksize) or None,
    )
    n_fail = sum(1 for f in feats_list if f.get("status") != "ok")
    if n_fail:
        print(
            f"  [features] {n_fail}/{len(feats_list)} OBJs failed extraction; "
            f"those frames will be skipped at NN inference time"
        )

    # ---- 6. NN inference (8-feature descriptor -> log f, Eq. 16) -------
    print("Running NN inference (chull+inertia 8-feature, log(f) direct) ...")
    inertia_per_frame = [
        np.asarray(
            fe.get("inertia_principal_unit", [float("nan")] * 3), dtype=np.float64
        )
        for fe in feats_list
    ]
    f_unit, valid = predict_nn_unit_for_frames_inertia_8feat_direct(
        feats_list, inertia_per_frame, artifacts
    )
    print(f"  NN predicted {int(np.sum(valid))}/{len(feats_list)} frames")

    # ---- 7. Map sample lines -> nearest frame, rescale by per-bub radius
    by_bub = build_per_bubble_index(job_meta, f_unit, valid)

    n_samples = len(sample_meta)
    new_freq: list[float] = [float("nan")] * n_samples
    have_pred = np.zeros(n_samples, dtype=bool)
    bub_ids_used: set[int] = set()
    no_radius_warned: set[int] = set()

    for i, (bub_id, t_s, _y) in enumerate(sample_meta):
        if bool(drop_mask[i]):
            continue
        entry = by_bub.get(int(bub_id))
        if entry is None:
            continue
        frames_arr, freq_arr = entry
        idx_local = int(
            np.argmin(
                np.abs(frames_arr.astype(np.float64) * float(dt_lbm) - float(t_s))
            )
        )
        f_unit_pick = float(freq_arr[idx_local])

        r = float(bub_radii.get(int(bub_id), float("nan")))
        if not (math.isfinite(r) and r > 0.0):
            if int(bub_id) not in no_radius_warned:
                print(
                    f"  [warn] no positive radius in trackedBubInfo for Bub "
                    f"{int(bub_id)}; sample at t={t_s} will use sentinel"
                )
                no_radius_warned.add(int(bub_id))
            continue
        v_real = (4.0 / 3.0) * math.pi * (r**3)
        new_freq[i] = float(f_unit_pick * (v_real ** (-1.0 / 3.0)))
        have_pred[i] = True
        bub_ids_used.add(int(bub_id))

    # ---- 8. Frequency column + missing-sample policy -------------------
    # Shared with every other tracked-file writer; see
    # tracked_bubinfo.io.resolve_missing_samples.
    fmt = args.frequency_format
    orig_tokens = (
        [
            (src_lines[li].split()[1] if len(src_lines[li].split()) >= 6 else "nan")
            for li in sample_idxs
        ]
        if args.missing_fill == "keep_original"
        else None
    )
    freq_strings, drop_mask, miss_stats = resolve_missing_samples(
        have_value=have_pred,
        values=new_freq,
        drop_mask=drop_mask,
        policy=args.missing_fill,
        fmt=fmt,
        orig_tokens=orig_tokens,
        bub_ids=[int(m[0]) for m in sample_meta],
        radius_of=bub_radii,
        minnaert_constant=float(MINNAERT_CONSTANT),
    )
    if miss_stats["dropped"] or miss_stats["substituted"] or miss_stats["no_radius"]:
        print(
            f"  {miss_stats['dropped'] + miss_stats['substituted'] + miss_stats['no_radius']}"
            f"/{n_samples} sample lines had no NN prediction "
            f"(--missing-fill {args.missing_fill})"
        )

    # ---- 9. Truncating writer -----------------------------------------
    out_lines, n_blocks_dropped, n_samples_dropped = truncate_and_rewrite_blocks(
        src_lines, sample_idxs, drop_mask, freq_strings, t_max
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # ---- 10. Summary --------------------------------------------------
    n_filled = int(np.sum(have_pred))
    n_kept_samples = n_samples - n_samples_dropped
    print()
    print("Summary:")
    print(f"  trackedBubInfo:           {tracked}")
    print(f"  output:                   {out_path}")
    print(f"  smoothed cache:           {smoothed_root}")
    print(f"  smooth_iters:             {int(args.smooth_iters)}")
    print(f"  dt_lbm:                   {dt_lbm:.9g} s ({dt_lbm_source})")
    print(f"  sample_stride:            {sample_stride}")
    print(f"  dt_frame (= stride*dt):   {dt_frame:.9g} s")
    print(f"  global_max_frame:         {global_max_frame}")
    print(f"  t_max (cutoff):           {t_max:.9g} s")
    print(f"  missing_fill:             {args.missing_fill}")
    print(f"  Bub blocks total (input): {len(bub_radii)}")
    print(f"  Bub blocks with meshes:   {len(bub_ids_with_meshes)}")
    print(f"  Bub blocks NN-updated:    {len(bub_ids_used)}")
    print(f"    -> real predictions:    {miss_stats['filled']}")
    print(f"    -> dropped (no pred):   {miss_stats['dropped']}")
    print(f"    -> substituted:         {miss_stats['substituted']}")
    if miss_stats["no_radius"]:
        print(f"    -> nan (no radius):     {miss_stats['no_radius']}")
    print(f"  Bub blocks dropped:       {n_blocks_dropped}")
    print(f"  sample lines (input):     {n_samples}")
    print(f"  sample lines dropped:     {n_samples_dropped}")
    print(f"  sample lines NN-updated:  {n_filled}")
    print(f"  sample lines sentinel:    {n_kept_samples - n_filled}")
    print(
        summarize_freqs(
            "f_nn_real (Hz)",
            [new_freq[i] for i in range(n_samples) if not drop_mask[i] and have_pred[i]],
        )
    )


if __name__ == "__main__":
    main()

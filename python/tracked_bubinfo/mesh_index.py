"""Index the per-bubble mesh tree that accompanies a ``trackedBubInfo`` file.

The LBM simulator exports marching-cubes surfaces as::

    <meshes_root>/bub_<id>/frame_<NNNNN>.obj

where ``NNNNN`` is the **LBM iteration index** (a multiple of the simulator's
``--bub-mesh-iter-stride``), not a wall-clock frame counter. Turning that tree
into per-sample-line frequencies needs three things, all of which live here:

1. enumerate the OBJs (:func:`discover_mesh_jobs`),
2. resolve seconds-per-iteration (:func:`infer_dt_lbm`) and sanity-check it
   against the tracked file (:func:`check_stride_and_alignment`),
3. snap each sample line to its nearest mesh frame
   (:func:`build_per_bubble_index`, :func:`nearest_frame`).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

__all__ = [
    "BUB_DIR_RE",
    "FRAME_FILE_RE",
    "build_per_bubble_index",
    "build_smoothing_jobs",
    "check_stride_and_alignment",
    "cross_check_dt_lbm",
    "discover_mesh_jobs",
    "infer_dt_lbm",
    "nearest_frame",
]


BUB_DIR_RE = re.compile(r"^bub_(\d+)$")
FRAME_FILE_RE = re.compile(r"^frame_(\d+)\.obj$")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_mesh_jobs(
    meshes_root: Path,
) -> tuple[list[tuple[int, str]], list[tuple[int, int]], set[int]]:
    """Enumerate every ``bub_<id>/frame_<NNNNN>.obj`` under ``meshes_root``.

    Returns ``(jobs, job_meta, bub_ids_with_meshes)`` where

    * ``jobs[i] = (i, obj_path_str)`` -- ready for ``ProcessPoolExecutor.map``
    * ``job_meta[i] = (bub_id, frame_idx)``
    * ``bub_ids_with_meshes`` is the set of bub_ids with at least one OBJ.
    """
    meshes_root = Path(meshes_root)
    if not meshes_root.is_dir():
        raise FileNotFoundError(f"meshes_root does not exist: {meshes_root}")

    jobs: list[tuple[int, str]] = []
    job_meta: list[tuple[int, int]] = []
    bub_ids: set[int] = set()
    job_idx = 0
    for entry in sorted(meshes_root.iterdir()):
        if not entry.is_dir():
            continue
        m = BUB_DIR_RE.match(entry.name)
        if not m:
            continue
        bub_id = int(m.group(1))
        for obj_path in sorted(entry.iterdir()):
            if not obj_path.is_file():
                continue
            mf = FRAME_FILE_RE.match(obj_path.name)
            if not mf:
                continue
            jobs.append((job_idx, str(obj_path)))
            job_meta.append((bub_id, int(mf.group(1))))
            bub_ids.add(bub_id)
            job_idx += 1
    return jobs, job_meta, bub_ids


def build_smoothing_jobs(
    raw_jobs: list[tuple[int, str]],
    job_meta: list[tuple[int, int]],
    meshes_root: Path,
    smoothed_root: Path,
    smooth_iters: int,
) -> list[tuple[int, str, str, int]]:
    """Pair each raw OBJ with its smoothed-cache destination.

    Returns ``(job_idx, raw_path, cached_path, smooth_iters)`` tuples for
    :func:`shape_feature.build_dataset.extract_features_one`. The cache
    mirrors the ``bub_<id>/frame_<NNNNN>.obj`` layout of ``meshes_root``.
    """
    meshes_root = Path(meshes_root)
    smoothed_root = Path(smoothed_root)
    out: list[tuple[int, str, str, int]] = []
    for (job_idx, raw_path), (bub_id, _frame_idx) in zip(raw_jobs, job_meta):
        raw = Path(raw_path)
        try:
            rel = raw.relative_to(meshes_root)
        except ValueError:
            # Fallback: rebuild the conventional bub_<id>/frame_<NNNNN>.obj path.
            rel = Path(f"bub_{bub_id}") / raw.name
        out.append((int(job_idx), str(raw), str(smoothed_root / rel), int(smooth_iters)))
    return out


# ---------------------------------------------------------------------------
# Timestep resolution + safeguards
# ---------------------------------------------------------------------------


def infer_dt_lbm(
    sample_meta: list[tuple[int, float, float]], sample_stride: int
) -> float:
    """Infer seconds-per-LBM-iteration from consecutive sample-line spacing.

    Collects ``t_{i+1} - t_i`` for adjacent samples within the same ``Bub``
    block, keeps the runs whose spacing is uniform (dropping non-positive
    deltas, which split/merge events produce as duplicate timestamps, and the
    larger gaps that event lines leave), and returns
    ``total run span / total intervals / sample_stride``.

    Pooled spans rather than a median of deltas, because trackedBubInfo writes
    times to six significant digits: a true spacing of 8.3333e-5 s prints as
    8e-5 or 9e-5 once ``t > 1`` s, so the median of the deltas is quantised to
    the printing grid and can sit several percent low. Dividing a run's total
    span by its interval count divides that rounding error by the number of
    intervals instead, which is thousands for a long record.

    ``sample_stride`` is the number of LBM steps between consecutive *sample
    lines*, which is not always the number of steps between mesh exports --
    see :func:`cross_check_dt_lbm`, which catches the two being confused.
    """
    deltas: list[float] = []
    prev_bub = None
    prev_t = None
    for bub_id, t, _y in sample_meta:
        if prev_bub == bub_id and prev_t is not None:
            d = float(t) - float(prev_t)
            if d > 0.0:
                deltas.append(d)
        prev_bub = bub_id
        prev_t = float(t)

    if not deltas:
        raise ValueError(
            "Cannot infer dt_lbm from sample lines (no positive consecutive "
            "deltas). Pass --dt-lbm or --dt-frame explicitly."
        )

    arr = np.asarray(deltas, dtype=np.float64)
    cutoff = float(np.percentile(arr, 95.0))
    trimmed = arr[arr <= cutoff]
    if trimmed.size == 0:
        trimmed = arr
    coarse = float(np.median(trimmed))

    # Second pass: pool the deltas that sit within a quarter of the coarse
    # spacing, so the total is one long uniform run per record and the
    # six-digit rounding averages out over its intervals.
    uniform = arr[np.abs(arr - coarse) <= 0.25 * coarse]
    if uniform.size == 0:
        uniform = trimmed
    return float(uniform.sum() / uniform.size) / float(sample_stride)


def cross_check_dt_lbm(
    by_bub_frames: dict[int, "np.ndarray | list[int]"],
    sample_meta: list[tuple[int, float, float]],
    dt_lbm: float,
    q: float = 10.0,
) -> tuple[float, float, int]:
    """Second, independent estimate of seconds-per-frame-index, and its error.

    Sample lines and mesh frame indices are two sequences over the same LBM
    step counter, so for a bubble whose meshes span its life the two cover the
    same interval and their inner quantiles line up::

        alpha = (t_q .. t_1-q span) / (frame_q .. frame_1-q span)

    Quantiles rather than min/max because a single mesh outside the record --
    a ``frame_00000.obj`` written before the bubble registers, say -- moves the
    endpoints several percent while leaving the bulk untouched.

    This never consults ``sample_stride``, so it catches the failure that flag
    cannot express: a scene that writes a sample line every LBM step but
    exports a mesh every 50 has two different strides, and using the mesh one
    to divide the sample spacing puts ``dt_lbm`` out by 50x.

    Returns ``(alpha, relative_error, n_bubbles_used)`` where
    ``relative_error = dt_lbm / alpha - 1``.
    """
    times: dict[int, list[float]] = {}
    for b, t, _y in sample_meta:
        times.setdefault(int(b), []).append(float(t))

    t_span = 0.0
    f_span = 0.0
    n_used = 0
    for b, frames in by_bub_frames.items():
        ts = times.get(int(b))
        fr = np.asarray(list(frames), dtype=np.float64)
        if not ts or fr.size < 2:
            continue
        ta = np.asarray(ts, dtype=np.float64)
        if ta.size < 2:
            continue
        dt_q = float(np.percentile(ta, 100.0 - q) - np.percentile(ta, q))
        df_q = float(np.percentile(fr, 100.0 - q) - np.percentile(fr, q))
        if dt_q <= 0.0 or df_q <= 0.0:
            continue
        t_span += dt_q
        f_span += df_q
        n_used += 1

    if n_used == 0 or f_span <= 0.0:
        return (float("nan"), float("nan"), 0)
    alpha = t_span / f_span
    return (alpha, float(dt_lbm) / alpha - 1.0, n_used)


def check_stride_and_alignment(
    job_meta: list[tuple[int, int]],
    sample_meta: list[tuple[int, float, float]],
    sample_stride: int,
    dt_lbm: float,
) -> None:
    """Verify mesh frame indices and sample times agree with ``dt_lbm``.

    Raises :class:`ValueError` when a discovered ``frame_idx`` is not a multiple
    of ``sample_stride`` (a hard inconsistency). Prints a ``[warn]`` when the
    median nearest-frame residual exceeds half a mesh-frame spacing -- the
    nearest-frame snap still runs, but ``dt_lbm`` is probably wrong.
    """
    bad = [(b, fr) for b, fr in job_meta if int(fr) % int(sample_stride) != 0]
    if bad:
        head = ", ".join(f"(bub_{b}, frame_{fr})" for b, fr in bad[:5])
        raise ValueError(
            f"sample_stride={sample_stride} but {len(bad)} mesh frames are not "
            f"multiples of it (e.g. {head}). Pass --sample-stride / --dt-lbm "
            f"explicitly."
        )

    by_bub_frames: dict[int, list[int]] = {}
    for b, fr in job_meta:
        by_bub_frames.setdefault(int(b), []).append(int(fr))
    for k in by_bub_frames:
        by_bub_frames[k].sort()

    by_bub_samples: dict[int, list[float]] = {}
    for b, t, _y in sample_meta:
        by_bub_samples.setdefault(int(b), []).append(float(t))

    # Check the bubble with the most sample lines that also has meshes.
    candidates = [
        (len(by_bub_samples.get(k, [])), k) for k in by_bub_frames if k in by_bub_samples
    ]
    if not candidates:
        print(
            "  [warn] alignment sanity skipped: no bubble has both sample lines "
            "and meshes",
            flush=True,
        )
        return
    candidates.sort(reverse=True)
    _n, k = candidates[0]
    frames = np.asarray(by_bub_frames[k], dtype=np.float64) * float(dt_lbm)
    times = np.asarray(by_bub_samples[k], dtype=np.float64)
    if frames.size == 0 or times.size == 0:
        return
    # The threshold is half a *mesh-frame* spacing (sample_stride * dt_lbm),
    # not half an LBM iteration, because the snap is to the nearest stored mesh.
    diff = np.abs(times[:, None] - frames[None, :])
    med_resid = float(np.median(diff.min(axis=1)))
    half_frame = 0.5 * float(sample_stride) * float(dt_lbm)
    if med_resid > half_frame:
        print(
            f"  [warn] alignment sanity: bub {k} has median |t_sample - "
            f"frame_idx*dt_lbm| = {med_resid:.6g}s > 0.5*dt_frame "
            f"({half_frame:.6g}s). Output mapping uses nearest-frame snap; "
            f"consider passing --dt-lbm explicitly.",
            flush=True,
        )
    else:
        print(
            f"  [ok] alignment sanity: bub {k} median residual = "
            f"{med_resid:.6g}s (< 0.5*dt_frame = {half_frame:.6g}s)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Frame <-> sample mapping
# ---------------------------------------------------------------------------


def build_per_bubble_index(
    job_meta: list[tuple[int, int]],
    f_unit: np.ndarray,
    valid: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Group per-frame predictions into ``{bub_id: (frame_idx_arr, f_unit_arr)}``.

    Only frames whose feature extraction and NN inference both succeeded (and
    that carry a finite positive frequency) are kept. Both arrays are sorted by
    frame index so :func:`nearest_frame` can index them directly.
    """
    by_bub: dict[int, list[tuple[int, float]]] = {}
    for i, (bub_id, frame_idx) in enumerate(job_meta):
        if not bool(valid[i]):
            continue
        if not math.isfinite(float(f_unit[i])) or f_unit[i] <= 0.0:
            continue
        by_bub.setdefault(bub_id, []).append((frame_idx, float(f_unit[i])))
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for bub_id, items in by_bub.items():
        items.sort()
        out[bub_id] = (
            np.array([k for k, _ in items], dtype=np.int64),
            np.array([v for _, v in items], dtype=np.float64),
        )
    return out


def nearest_frame(
    frames: np.ndarray, t_sample: float, dt_frame: float
) -> tuple[int, int]:
    """Return ``(frame_idx, position_in_frames)`` nearest in time to ``t_sample``.

    ``dt_frame`` converts a frame index to simulation seconds. The caller looks
    up the corresponding frequency at the returned position.
    """
    times = frames.astype(np.float64) * float(dt_frame)
    idx = int(np.argmin(np.abs(times - float(t_sample))))
    return int(frames[idx]), idx

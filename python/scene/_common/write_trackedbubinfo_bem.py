"""Write ``dataset/fruit_splash/trackedBubInfo_BEM.txt`` from the all-bubbles BEM run.

**Not re-runnable from the repository alone.** It documents how the shipped
``trackedBubInfo_BEM.txt`` was produced. Its input is the per-row CSV of the
Table 1 all-bubbles sweep (``nn8_vs_bem_all_bubbles_chunked.py``), which solved
a P1-DP0 Galerkin BEM on every >100-vertex fruit bubble mesh (<= 5k verts) and
recorded the unit-volume Minnaert frequency per (bub_id, frame). That CSV runs
to hundreds of MB and does not ship; only its summary does, under
``results/experiments/table01_timing_across_scenes/all_bubbles/fruit08/``.
Pass ``--rows-csv`` to point it at a sweep you have run yourself.

This script maps those frequencies onto the fruit trackedBubInfo, mirroring the
NN pipeline's conventions exactly: the same nearest-frame snap and the same
``f_real = f_unit * V**(-1/3)`` radius rescale as
``write_trackedbubinfo_nn.py`` step 7.

* base file = ``trackedBubInfo_NN.txt`` (the selected bubble cloud used by
  the published NN / Minnaert renders), frequencies overwritten per sample line
  by the nearest BEM mesh frame of the same bubble;
* sample lines whose bubble has no BEM frame keep their NN frequency
  (``--missing-fill keep_original``; the BEM run skips meshes <= 100 verts,
  > 5k verts (NN-only) and degenerate geometry) — counted in the summary.

The output is a frequency column, not audio. Rendering it to sound is the job
of FluidSound, which is a separate project and is not driven from this repo.

RUN (repo root, soundlab env):
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    python -u python/scene/_common/write_trackedbubinfo_bem.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PYTHON_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PYTHON_ROOT.parent
for _p in (REPO_ROOT, PYTHON_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402, F401  (torch before numpy on this Windows stack)
import numpy as np  # noqa: E402

from tracked_bubinfo import (  # noqa: E402
    cross_check_dt_lbm,
    format_freq,
    infer_dt_lbm,
    parse_bub_header_radii,
    parse_trackedbubinfo_blocks,
    summarize_freqs,
    truncate_and_rewrite_blocks,
)

DEFAULT_TRACKED = REPO_ROOT / "dataset" / "fruit_splash" / "trackedBubInfo_NN.txt"
DEFAULT_ROWS_CSV = (
    REPO_ROOT
    / "results" / "experiments" / "table01_timing_across_scenes"
    / "all_bubbles" / "fruits" / "fruits_all_rows.csv"
)
DEFAULT_OUT = REPO_ROOT / "dataset" / "fruit_splash" / "trackedBubInfo_BEM.txt"


def _load_bem_f_unit(
    rows_csv: Path,
    freq_column: str = "f_hz_bem",
    f_unit_max: float = 0.0,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """(bub_id -> (sorted frame indices, f_unit at those frames)) from the run CSV.

    ``f_unit_max`` > 0 drops rows above that unit-volume frequency — needed for
    the poly-2 column, whose OOD extrapolations reach 1e240 Hz / inf and would
    blow up the FluidSound solver (dropped lines fall back to the base file).
    """
    by_bub: dict[int, list[tuple[int, float]]] = {}
    n_rows = 0
    n_capped = 0
    with rows_csv.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            fb = r.get(freq_column, "")
            if not fb:
                continue
            fu = float(fb)
            if not (math.isfinite(fu) and fu > 0.0):
                continue
            if f_unit_max > 0.0 and fu > f_unit_max:
                n_capped += 1
                continue
            try:
                bub_id = int(r["bub_id"])
                frame = int(r["frame"])
            except (KeyError, TypeError, ValueError):
                continue
            by_bub.setdefault(bub_id, []).append((frame, fu))
            n_rows += 1
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for bub_id, pairs in by_bub.items():
        pairs.sort(key=lambda x: x[0])
        out[bub_id] = (
            np.array([p[0] for p in pairs], dtype=np.int64),
            np.array([p[1] for p in pairs], dtype=np.float64),
        )
    print(
        f"  {freq_column} lookup: {n_rows} (bub, frame) frequencies across "
        f"{len(out)} bubbles"
        + (f" | {n_capped} rows above f_unit_max={f_unit_max} dropped" if n_capped else "")
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracked", type=Path, default=DEFAULT_TRACKED)
    ap.add_argument("--rows-csv", type=Path, default=DEFAULT_ROWS_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--freq-column",
        type=str,
        default="f_hz_bem",
        help="Unit-volume frequency column of --rows-csv (e.g. f_hz_poly2 or "
        "f_hz_nn8_10k from *_regressor_freqs.csv).",
    )
    ap.add_argument(
        "--f-unit-max",
        type=float,
        default=0.0,
        help="Drop lookup rows above this unit-volume frequency (0 = no cap). "
        "Use ~1000 for poly2 (OOD exp-overflow predictions).",
    )
    ap.add_argument(
        "--sample-stride",
        type=int,
        default=50,
        help="LBM steps between consecutive SAMPLE LINES -- not necessarily "
        "the steps between mesh exports. A scene that writes a line every step "
        "and exports a mesh every 50 needs 1 here, and gets a dt_lbm 50x too "
        "small with the default. Only used when --dt-lbm is absent; either way "
        "the result is cross-checked against the mesh frame indices.",
    )
    ap.add_argument(
        "--dt-tolerance",
        type=float,
        default=0.02,
        help="Warn when the resolved dt_lbm differs from the frame-index "
        "cross-check by more than this fraction (default 0.02).",
    )
    ap.add_argument(
        "--allow-dt-mismatch",
        action="store_true",
        help="Downgrade a failed dt_lbm cross-check from an error to a "
        "warning. Only for scenes whose meshes genuinely cover a small part "
        "of each bubble's life, which biases the cross-check.",
    )
    ap.add_argument("--dt-lbm", type=float, default=None)
    ap.add_argument("--frequency-format", type=str, default="{:.6f}")
    ap.add_argument(
        "--missing-fill",
        choices=["keep_original", "nan"],
        default="keep_original",
        help="Sample lines whose bubble has no BEM frame keep the base (NN) "
        "frequency (default) so the render shares the exact bubble cloud of "
        "the NN/Minnaert variants.",
    )
    args = ap.parse_args()

    tracked = args.tracked.resolve()
    out_path = args.out.resolve()
    print(f"Parsing trackedBubInfo base: {tracked}")
    src_lines, sample_idxs, sample_meta = parse_trackedbubinfo_blocks(tracked)
    bub_radii = parse_bub_header_radii(src_lines)
    print(f"  {len(sample_idxs)} sample lines | {len(bub_radii)} Bub headers")

    by_bub = _load_bem_f_unit(
        args.rows_csv.resolve(),
        freq_column=str(args.freq_column),
        f_unit_max=float(args.f_unit_max),
    )
    if not by_bub:
        raise SystemExit(f"no {args.freq_column} frequencies found in {args.rows_csv}")

    if args.dt_lbm is not None:
        dt_lbm = float(args.dt_lbm)
        src = "--dt-lbm"
    else:
        dt_lbm = infer_dt_lbm(sample_meta, int(args.sample_stride))
        src = f"auto-inferred, --sample-stride {int(args.sample_stride)}"
    print(f"  dt_lbm = {dt_lbm:.9g} s ({src})")

    # A wrong dt_lbm does not fail: it snaps every sample line to the wrong
    # mesh and writes a plausible-looking file. Check it against the frame
    # indices, which know nothing about --sample-stride.
    alpha, rel_err, n_used = cross_check_dt_lbm(
        {b: fr for b, (fr, _f) in by_bub.items()}, sample_meta, dt_lbm
    )
    if n_used == 0:
        print("  [warn] dt_lbm cross-check skipped: no bubble has both sample "
              "lines and >= 2 mesh frames")
    elif abs(rel_err) <= float(args.dt_tolerance):
        print(f"  [ok] dt_lbm cross-check: frame indices imply {alpha:.9g} s "
              f"({rel_err * 100:+.2f}%, {n_used} bubble(s))")
    else:
        # dt_lbm = pooled sample spacing / sample_stride, so if the flag is
        # wrong by an integer factor the cross-check recovers it:
        #   alpha ~= spacing / k_true  =>  k_true ~= sample_stride * dt_lbm / alpha
        ratio = alpha / dt_lbm
        how = f"{ratio:.4g}x too small" if ratio > 1.0 else f"{1.0 / ratio:.4g}x too large"
        msg = (
            f"dt_lbm cross-check FAILED: the mesh frame indices imply "
            f"{alpha:.9g} s per frame, but dt_lbm is {dt_lbm:.9g} s "
            f"({rel_err * 100:+.1f}%, {n_used} bubble(s)) -- {how}.\n"
            f"  Every sample line would snap to a mesh at the wrong time."
        )
        explained = False
        if args.dt_lbm is None:
            k = int(round(int(args.sample_stride) * dt_lbm / alpha))
            if k >= 1 and k != int(args.sample_stride):
                explained = True
                msg += (
                    f"\n  --sample-stride is the steps between SAMPLE LINES, not "
                    f"between mesh exports; the frame indices say it is {k}, not "
                    f"{int(args.sample_stride)}. Try --sample-stride {k}, or pass "
                    f"--dt-lbm {alpha:.9g} explicitly."
                )
        if not explained:
            # The cross-check assumes a bubble's meshes span its life. Where they
            # do not -- meshes start at a vertex floor and stop before collapse --
            # the frame span is short for the same elapsed time and alpha reads
            # high by roughly the reciprocal of the coverage fraction.
            msg += (
                f"\n  No integer sample stride explains it. If the meshes cover "
                f"only ~{100.0 / max(alpha / dt_lbm, 1e-9):.0f}% of each bubble's "
                f"life -- a vertex floor at birth, collapse after the last export "
                f"-- that alone produces this, and --allow-dt-mismatch is the right "
                f"answer. Otherwise pass --dt-lbm {alpha:.9g}."
            )
        if args.allow_dt_mismatch:
            print(f"  [warn] {msg}")
        else:
            raise SystemExit(
                f"  [error] {msg}\n"
                f"  Re-run with --allow-dt-mismatch only if the meshes "
                f"genuinely cover a small part of each bubble's life."
            )

    # Keep every sample line of the base file (it is already truncated to the
    # scene's mesh availability by the NN pipeline): no additional cutoff.
    n_samples = len(sample_meta)
    drop_mask = np.zeros(n_samples, dtype=bool)
    t_max = float("inf")

    fmt = args.frequency_format
    orig_tokens: list[str] = []
    for li in sample_idxs:
        parts = src_lines[li].split()
        orig_tokens.append(parts[1] if len(parts) >= 6 else "nan")

    new_freq = [float("nan")] * n_samples
    have_pred = np.zeros(n_samples, dtype=bool)
    bub_ids_used: set[int] = set()

    for i, (bub_id, t_s, _y) in enumerate(sample_meta):
        entry = by_bub.get(int(bub_id))
        if entry is None:
            continue
        frames_arr, freq_arr = entry
        idx_local = int(
            np.argmin(np.abs(frames_arr.astype(np.float64) * dt_lbm - float(t_s)))
        )
        f_unit_pick = float(freq_arr[idx_local])
        r = float(bub_radii.get(int(bub_id), float("nan")))
        if not (math.isfinite(r) and r > 0.0):
            continue
        v_real = (4.0 / 3.0) * math.pi * (r ** 3)
        new_freq[i] = f_unit_pick * (v_real ** (-1.0 / 3.0))
        have_pred[i] = True
        bub_ids_used.add(int(bub_id))

    freq_strings: list[str] = [""] * n_samples
    n_fallback = 0
    for i in range(n_samples):
        if have_pred[i]:
            freq_strings[i] = format_freq(new_freq[i], fmt)
        elif args.missing_fill == "keep_original":
            freq_strings[i] = orig_tokens[i]
            n_fallback += 1
        else:
            freq_strings[i] = "nan"
            n_fallback += 1

    out_lines, n_blocks_dropped, n_samples_dropped = truncate_and_rewrite_blocks(
        src_lines, sample_idxs, drop_mask, freq_strings, t_max
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    finite_freq = [new_freq[i] for i in range(n_samples) if have_pred[i]]
    print()
    print("Summary:")
    print(f"  base tracked:               {tracked}")
    print(f"  output:                     {out_path}")
    print(f"  Bub blocks (base):          {len(bub_radii)}")
    print(f"  Bub blocks BEM-updated:     {len(bub_ids_used)}")
    print(f"  blocks dropped:             {n_blocks_dropped}")
    print(f"  sample lines:               {n_samples}")
    print(f"  sample lines BEM-filled:    {int(np.sum(have_pred))}")
    print(f"  sample lines kept-original: {n_fallback} ({args.missing_fill})")
    print(summarize_freqs("f_bem_real (Hz)", finite_freq))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

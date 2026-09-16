from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def find_thumbnail_path(thumbnail_dir: Path, mesh_filename: str) -> Path | None:
    name = str(mesh_filename).strip()
    if not name:
        return None
    stem = Path(name).stem
    candidates = [
        thumbnail_dir / f"{stem}.png",
        thumbnail_dir / name.replace(".obj", ".png"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def sanitize_filename(s: str) -> str:
    s = re.sub(r"[<>:\"/\\\\|?*]+", "_", s)
    s = re.sub(r"\s+", "_", s).strip("._ ")
    return s


def export_worst_cases(
    df_used: pd.DataFrame,
    test_indices: np.ndarray,
    pred_f: np.ndarray,
    tgt_f: np.ndarray,
    out_dir: Path,
    *,
    top_k: int = 10,
    thumbnail_dir: Path | None = None,
    thumbnail_dir_by_source: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Export worst-APE rows with optional thumbnails.

    ``thumbnail_dir_by_source`` lets callers route per-row thumbnail lookups
    based on the row's ``source`` column (e.g. ``{"VOF": Path(...),
    "LBM": Path(...)}``). When provided, it takes precedence over
    ``thumbnail_dir`` for rows whose source matches a known key; rows with an
    unknown / missing source fall back to ``thumbnail_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ape = np.abs(pred_f - tgt_f) / np.maximum(tgt_f, 1e-12) * 100.0
    order = np.argsort(-ape)
    chosen_local = order[: min(top_k, len(order))]
    chosen_global = np.asarray(test_indices, dtype=np.int64)[chosen_local]

    worst_rows = df_used.iloc[chosen_global].copy().reset_index(drop=True)
    worst_rows["pred_freq_hz"] = pred_f[chosen_local]
    worst_rows["gt_freq_hz"] = tgt_f[chosen_local]
    worst_rows["abs_error_hz"] = np.abs(worst_rows["pred_freq_hz"] - worst_rows["gt_freq_hz"])
    worst_rows["ape_pct"] = ape[chosen_local]

    txt_path = out_dir / "worst_10_test_cases.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(worst_rows.to_string(index=False))
        f.write("\n")

    csv_path = out_dir / "worst_10_test_cases.csv"
    worst_rows.to_csv(csv_path, index=False)

    has_mesh = "mesh_filename" in worst_rows.columns
    has_source = "source" in worst_rows.columns
    by_source = thumbnail_dir_by_source or {}
    if has_mesh and (thumbnail_dir is not None or by_source):
        for _, row in worst_rows.iterrows():
            mesh_name = str(row["mesh_filename"])
            src = str(row["source"]) if has_source and pd.notna(row.get("source", None)) else ""
            # Strict per-source routing: if `src` matches a registered key, only
            # look in that directory. Cross-source fallback would silently emit a
            # wrong thumbnail because Tim2016 and LBM share `bubble.NNNN.M.png`
            # filenames.
            if src and src in by_source:
                chosen_dir = by_source[src]
            else:
                chosen_dir = thumbnail_dir
            if chosen_dir is None or not Path(chosen_dir).exists():
                continue
            thumb = find_thumbnail_path(Path(chosen_dir), mesh_name)
            if thumb is None:
                continue
            ds_idx = row["index"] if "index" in worst_rows.columns else (row.get("mesh_id", "na") if has_source else "na")
            src_tag = f"_{src}" if src else ""
            name = sanitize_filename(
                f"{ds_idx}{src_tag}_{float(row['gt_freq_hz']):.6f}_{float(row['ape_pct']):.3f}pct.png"
            )
            shutil.copy2(thumb, out_dir / name)

    return worst_rows

"""Reading, rewriting, and mesh-indexing of WaveBlender ``trackedBubInfo`` files.

``trackedBubInfo.txt`` is the bubble-tracker export format shared by the LBM
simulator and the FluidSound renderer, both of which are separate
projects. The format is documented by the layout in
:mod:`tracked_bubinfo.io`. This package holds the format-level code that every scene
driver needs:

* :mod:`tracked_bubinfo.io` -- parse blocks / sample lines, read the per-bubble
  header radii, and rewrite the frequency column (optionally truncating blocks
  at a cutoff time).
* :mod:`tracked_bubinfo.mesh_index` -- discover the per-bubble marching-cubes
  OBJ tree that accompanies a tracked file, resolve the simulation timestep,
  and map each sample line to its nearest mesh frame.

Nothing here depends on torch, a trained model, or any particular scene.
"""

from __future__ import annotations

from tracked_bubinfo.io import (
    MISSING_POLICIES,
    MISSING_POLICY_HELP,
    format_freq,
    iter_sample_line_indices,
    parse_bub_header_radii,
    parse_sample_y,
    parse_trackedbubinfo_blocks,
    resolve_missing_samples,
    rewrite_freq_column,
    rewrite_with_string_freqs,
    summarize_freqs,
    truncate_and_rewrite_blocks,
    write_tracked_lines,
)
from tracked_bubinfo.mesh_index import (
    BUB_DIR_RE,
    FRAME_FILE_RE,
    build_per_bubble_index,
    build_smoothing_jobs,
    check_stride_and_alignment,
    cross_check_dt_lbm,
    discover_mesh_jobs,
    infer_dt_lbm,
    nearest_frame,
)

__all__ = [
    # io
    "MISSING_POLICIES",
    "MISSING_POLICY_HELP",
    "format_freq",
    "iter_sample_line_indices",
    "parse_bub_header_radii",
    "parse_sample_y",
    "parse_trackedbubinfo_blocks",
    "resolve_missing_samples",
    "rewrite_freq_column",
    "rewrite_with_string_freqs",
    "summarize_freqs",
    "truncate_and_rewrite_blocks",
    "write_tracked_lines",
    # mesh_index
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

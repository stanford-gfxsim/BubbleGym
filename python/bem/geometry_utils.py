"""
Mesh helpers for the BEM pipeline.

All functions work directly on (V Nx3, F Mx3) numpy arrays so the downstream BEM
solver does not need to round-trip through temporary .obj files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .bempp_io import _compute_volume, _load_triangle_mesh


def load_bubble_unit_volume(bubble_path: str | Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load a bubble mesh, translate its centroid to the origin, and scale it to unit volume.

    Returns (V, F, scale) where `scale` is the multiplicative factor that was applied
    to the vertex coordinates (1.0 if already unit-volume).
    """
    verts, faces = _load_triangle_mesh(str(bubble_path))
    if verts.size == 0 or faces.size == 0:
        raise ValueError(f"Empty mesh: {bubble_path}")

    centroid = verts.mean(axis=0)
    verts = verts - centroid

    vol = _compute_volume(verts, faces)
    if vol <= 1e-12:
        raise ValueError(f"Degenerate (non-positive) volume for bubble mesh: {bubble_path}")
    scale = (1.0 / vol) ** (1.0 / 3.0)
    verts = verts * scale
    return verts.astype(np.float64), faces.astype(np.int64), float(scale)


def _flip_faces(faces: np.ndarray) -> np.ndarray:
    """Reverse triangle winding (and therefore the face normals)."""
    out = np.asarray(faces).copy()
    out[:, [1, 2]] = out[:, [2, 1]]
    return out


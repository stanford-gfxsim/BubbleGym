"""Mesh I/O, topology checks, geometry primitives, and Laplacian smoothing.

Single source of truth for everything that operates on a raw triangle mesh
before any shape feature is derived.

Contents
--------
I/O                 :func:`load_obj_mesh`, :func:`write_obj`
Topology            :func:`edge_incidence_report`, :func:`check_watertight`,
                    :func:`largest_connected_component_mesh`,
                    :func:`ensure_outward_orientation`
Geometry            :func:`mesh_signed_volume`, :func:`mesh_surface_area`,
                    :func:`vertices_scaled_to_target_volume`
Mass properties     :func:`compute_mirtich_moments`, :func:`principal_inertia`
Smoothing           :func:`laplacian_smooth_vf`, :func:`load_or_smooth_obj`

Watertightness
--------------
Every feature in this package assumes a *closed* surface: the divergence-theorem
volume, the Mirtich moment integrals, and the dihedral-based curvature integrals
are all meaningless on a mesh with boundary. :func:`load_obj_mesh` therefore
validates topology on load and raises :class:`NonWatertightMeshError` when the
mesh has holes or non-manifold edges, rather than silently returning a number
computed from an open surface.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np


__all__ = [
    "MeshError",
    "NonWatertightMeshError",
    "load_obj_mesh",
    "write_obj",
    "edge_incidence_report",
    "check_watertight",
    "largest_connected_component_mesh",
    "ensure_outward_orientation",
    "mesh_signed_volume",
    "mesh_surface_area",
    "vertices_scaled_to_target_volume",
    "compute_mirtich_moments",
    "principal_inertia",
    "laplacian_smooth_vf",
    "load_or_smooth_obj",
]


class MeshError(ValueError):
    """Base class for malformed-mesh errors raised by this package."""


class NonWatertightMeshError(MeshError):
    """Raised when a mesh has boundary (holes) or non-manifold edges.

    Attributes
    ----------
    n_boundary_edges
        Edges incident to exactly one triangle -- the signature of a hole.
    n_nonmanifold_edges
        Edges incident to three or more triangles.
    """

    def __init__(
        self,
        message: str,
        *,
        n_boundary_edges: int = 0,
        n_nonmanifold_edges: int = 0,
    ):
        super().__init__(message)
        self.n_boundary_edges = int(n_boundary_edges)
        self.n_nonmanifold_edges = int(n_nonmanifold_edges)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def edge_incidence_report(f: np.ndarray) -> dict[str, int | np.ndarray]:
    """Count how many triangles share each undirected edge.

    A closed manifold triangle mesh has every edge shared by exactly two
    triangles. Edges with one incident face bound a hole; edges with three or
    more are non-manifold (e.g. two sheets pinched along a seam).

    Returns a dict with ``n_edges``, ``n_boundary_edges``,
    ``n_nonmanifold_edges``, and ``boundary_vertices`` (the unique vertex ids
    touched by boundary edges, useful for reporting where a hole is).
    """
    f = np.ascontiguousarray(f, dtype=np.int64)
    if f.size == 0:
        return {
            "n_edges": 0,
            "n_boundary_edges": 0,
            "n_nonmanifold_edges": 0,
            "boundary_vertices": np.empty(0, dtype=np.int64),
        }

    # Three undirected edges per triangle, canonicalized as (min, max).
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)

    boundary = counts == 1
    nonmanifold = counts > 2
    return {
        "n_edges": int(uniq.shape[0]),
        "n_boundary_edges": int(np.count_nonzero(boundary)),
        "n_nonmanifold_edges": int(np.count_nonzero(nonmanifold)),
        "boundary_vertices": np.unique(uniq[boundary].ravel()),
    }


def check_watertight(f: np.ndarray, *, context: str = "") -> None:
    """Raise :class:`NonWatertightMeshError` unless every edge has two faces.

    ``context`` is prepended to the message (typically the source path) so the
    caller can tell which mesh in a 30k-frame tree is broken.
    """
    rep = edge_incidence_report(f)
    n_b = int(rep["n_boundary_edges"])
    n_nm = int(rep["n_nonmanifold_edges"])
    if n_b == 0 and n_nm == 0:
        return

    where = f"{context}: " if context else ""
    parts = []
    if n_b:
        bv = rep["boundary_vertices"]
        sample = ", ".join(str(int(x)) for x in np.asarray(bv)[:5])
        parts.append(
            f"{n_b} boundary edge(s) (holes) touching {len(np.asarray(bv))} "
            f"vertices [e.g. {sample}]"
        )
    if n_nm:
        parts.append(f"{n_nm} non-manifold edge(s) (>2 incident faces)")
    raise NonWatertightMeshError(
        f"{where}mesh is not a closed manifold surface: " + "; ".join(parts) +
        ". Volume, inertia, and curvature integrals are undefined on an open "
        "surface, so this mesh cannot be used for shape features.",
        n_boundary_edges=n_b,
        n_nonmanifold_edges=n_nm,
    )


def largest_connected_component_mesh(
    v: np.ndarray, f: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return ``(v_lcc, f_lcc, n_components, kept_face_count)``.

    Triangles are partitioned by vertex-edge connectivity and the component with
    the most triangles is kept, with vertices remapped to a contiguous range.
    The inputs are returned untouched when the mesh has a single component.

    Some simulation frames store several disjoint blobs under one label --
    typically the main bubble plus tiny satellite shells around split/merge
    events. Keeping the dominant component yields a single closed surface.
    """
    f = np.asarray(f, dtype=np.int64)
    if f.size == 0:
        return v, f, 0, 0
    n_v = int(v.shape[0])
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.int8)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        graph = coo_matrix((data, (rows, cols)), shape=(n_v, n_v)).tocsr()
        n_cc, labels = connected_components(graph, directed=False)
    except ImportError:
        # Pure-numpy union-find fallback (only when scipy is unavailable).
        parent = np.arange(n_v, dtype=np.int64)

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        for a, b, c in f:
            ra, rb, rc = _find(int(a)), _find(int(b)), _find(int(c))
            if ra != rb:
                parent[ra] = rb
            if rb != rc:
                parent[rb] = rc
        labels = np.array([_find(i) for i in range(n_v)], dtype=np.int64)
        n_cc = int(np.unique(labels).size)
    if n_cc <= 1:
        return v, f, 1, int(f.shape[0])
    face_labels = labels[f[:, 0]]
    counts = np.bincount(face_labels, minlength=int(labels.max()) + 1)
    main = int(np.argmax(counts))
    keep = face_labels == main
    f_keep = f[keep]
    used = np.unique(f_keep.ravel())
    remap = np.full(n_v, -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    f_local = remap[f_keep]
    v_local = v[used]
    return v_local, f_local.astype(np.int64), int(n_cc), int(f_keep.shape[0])


def ensure_outward_orientation(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Return ``f`` with winding flipped if the mesh is inside-out.

    Some marching-cubes / Gmsh-derived OBJs are wound inwards, which makes the
    signed volume negative and the Mirtich principal moments come out negative
    (Strasberg requires them positive).
    """
    f = np.ascontiguousarray(f, dtype=np.int64)
    if mesh_signed_volume(v, f) < 0.0:
        return np.ascontiguousarray(f[:, [0, 2, 1]], dtype=np.int64)
    return f


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_obj_mesh(
    obj_path: Path, *, require_watertight: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``(vertices, faces)`` from a Wavefront OBJ; polygons are fanned.

    Negative (relative) face indices are supported. By default the result is
    validated with :func:`check_watertight`; pass ``require_watertight=False``
    only when the caller genuinely wants an open surface (e.g. visualizing a
    partial mesh).
    """
    obj_path = Path(obj_path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                idxs: list[int] = []
                for p in parts:
                    t = p.split("/")[0]
                    if not t:
                        continue
                    idx = int(t)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx -= 1
                    idxs.append(idx)
                if len(idxs) < 3:
                    continue
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    if not vertices or not faces:
        raise MeshError(f"OBJ parse failed (no vertices or no faces): {obj_path}")

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if require_watertight:
        check_watertight(f, context=str(obj_path))
    return v, f


def write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    """Minimal OBJ writer (``%.7g`` vertices, 1-based faces), written atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as h:
            h.write("# laplacian-smoothed bubble OBJ\n")
            for i in range(v.shape[0]):
                h.write(f"v {v[i, 0]:.7g} {v[i, 1]:.7g} {v[i, 2]:.7g}\n")
            for i in range(f.shape[0]):
                a, b, c = int(f[i, 0]) + 1, int(f[i, 1]) + 1, int(f[i, 2]) + 1
                h.write(f"f {a} {b} {c}\n")
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


def mesh_signed_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Signed volume via the divergence theorem (one tetrahedron per face)."""
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


def mesh_surface_area(v: np.ndarray, f: np.ndarray) -> float:
    """Total triangle area."""
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return float(0.5 * np.sum(np.linalg.norm(cross, axis=1)))


def vertices_scaled_to_target_volume(
    v: np.ndarray, f: np.ndarray, target_volume: float
) -> np.ndarray:
    """Uniformly rescale ``v`` about its centroid to a target enclosed volume."""
    vol_abs = abs(mesh_signed_volume(v, f))
    if not math.isfinite(vol_abs) or vol_abs <= 0.0:
        raise MeshError("mesh volume is non-positive or non-finite")
    c = v.mean(axis=0)
    s = (float(target_volume) / vol_abs) ** (1.0 / 3.0)
    return c + (v - c) * s


# ---------------------------------------------------------------------------
# Mass properties (Mirtich 1996 / Eberly)
# ---------------------------------------------------------------------------

# Mirtich integral pre-factors (per src/moment.cpp).
_MIRTICH_MULT = np.array(
    [
        1.0 / 6.0,
        1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0,
        1.0 / 60.0, 1.0 / 60.0, 1.0 / 60.0,
        1.0 / 120.0, 1.0 / 120.0, 1.0 / 120.0,
    ],
    dtype=np.float64,
)


def _subexpressions(
    w0: float, w1: float, w2: float
) -> Tuple[float, float, float, float, float, float]:
    """Mirtich's polynomial intermediates for one coordinate axis."""
    temp0 = w0 + w1
    f1 = temp0 + w2
    temp1 = w0 * w0
    temp2 = temp1 + w1 * temp0
    f2 = temp2 + w2 * f1
    f3 = w0 * temp1 + w1 * temp2 + w2 * f2
    g0 = f2 + w0 * (f1 + w0)
    g1 = f2 + w1 * (f1 + w1)
    g2 = f2 + w2 * (f1 + w2)
    return f1, f2, f3, g0, g1, g2


def compute_mirtich_moments(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return ``(mass, com, inertia_about_com)`` for a closed triangle mesh.

    ``mass`` is the signed enclosed volume at unit density (positive when faces
    are outward-oriented). ``inertia`` is the volumetric inertia tensor about
    ``com``; divide by ``abs(mass)`` for the volume-normalized (Strasberg)
    convention.

    Reference: Brian Mirtich, "Fast and Accurate Computation of Polyhedral Mass
    Properties", Journal of Graphics Tools 1(2), 1996. Matches
    ``src/moment.cpp`` (``bubbleLab::computeMoment``).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise MeshError("vertices must have shape (N, 3)")
    if f.ndim != 2 or f.shape[1] != 3:
        raise MeshError("faces must have shape (M, 3)")

    intg = np.zeros(10, dtype=np.float64)

    idx0, idx1, idx2 = f[:, 0], f[:, 1], f[:, 2]
    x0, y0, z0 = v[idx0, 0], v[idx0, 1], v[idx0, 2]
    x1, y1, z1 = v[idx1, 0], v[idx1, 1], v[idx1, 2]
    x2, y2, z2 = v[idx2, 0], v[idx2, 1], v[idx2, 2]

    a1, b1, c1 = x1 - x0, y1 - y0, z1 - z0
    a2, b2, c2 = x2 - x0, y2 - y0, z2 - z0

    d0 = b1 * c2 - b2 * c1
    d1 = a2 * c1 - a1 * c2
    d2 = a1 * b2 - a2 * b1

    # Kept as an explicit per-face loop: the summation order defines the
    # floating-point result, and the shipped dataset was built with it.
    for j in range(len(f)):
        f1x, f2x, f3x, g0x, g1x, g2x = _subexpressions(
            float(x0[j]), float(x1[j]), float(x2[j])
        )
        f1y, f2y, f3y, g0y, g1y, g2y = _subexpressions(
            float(y0[j]), float(y1[j]), float(y2[j])
        )
        f1z, f2z, f3z, g0z, g1z, g2z = _subexpressions(
            float(z0[j]), float(z1[j]), float(z2[j])
        )

        dd0, dd1, dd2 = float(d0[j]), float(d1[j]), float(d2[j])

        intg[0] += dd0 * f1x
        intg[1] += dd0 * f2x
        intg[2] += dd1 * f2y
        intg[3] += dd2 * f2z
        intg[4] += dd0 * f3x
        intg[5] += dd1 * f3y
        intg[6] += dd2 * f3z
        intg[7] += dd0 * (
            float(y0[j]) * g0x + float(y1[j]) * g1x + float(y2[j]) * g2x
        )
        intg[8] += dd1 * (
            float(z0[j]) * g0y + float(z1[j]) * g1y + float(z2[j]) * g2y
        )
        intg[9] += dd2 * (
            float(x0[j]) * g0z + float(x1[j]) * g1z + float(x2[j]) * g2z
        )

    intg *= _MIRTICH_MULT
    mass = float(intg[0])
    if not np.isfinite(mass) or abs(mass) < 1e-18:
        raise MeshError("Degenerate mass/volume from Mirtich integration.")

    cm = np.array([intg[1] / mass, intg[2] / mass, intg[3] / mass], dtype=np.float64)

    inertia = np.zeros((3, 3), dtype=np.float64)
    inertia[0, 0] = intg[5] + intg[6] - mass * (cm[1] * cm[1] + cm[2] * cm[2])
    inertia[1, 1] = intg[4] + intg[6] - mass * (cm[2] * cm[2] + cm[0] * cm[0])
    inertia[2, 2] = intg[4] + intg[5] - mass * (cm[0] * cm[0] + cm[1] * cm[1])
    inertia[0, 1] = inertia[1, 0] = -(intg[7] - mass * cm[0] * cm[1])
    inertia[1, 2] = inertia[2, 1] = -(intg[8] - mass * cm[1] * cm[2])
    inertia[0, 2] = inertia[2, 0] = -(intg[9] - mass * cm[2] * cm[0])
    return mass, cm, inertia


def principal_inertia(inertia: np.ndarray) -> np.ndarray:
    """Return the three principal moments, sorted ascending."""
    eig = np.linalg.eigvalsh(np.asarray(inertia, dtype=np.float64))
    return np.sort(eig)


# ---------------------------------------------------------------------------
# Laplacian smoothing
# ---------------------------------------------------------------------------


def _build_vertex_neighbors(
    f: np.ndarray, n_verts: int
) -> Tuple[np.ndarray, np.ndarray]:
    """CSR-style ``(indptr, indices)`` of the symmetric vertex adjacency."""
    f = np.ascontiguousarray(f, dtype=np.int64)
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    rows = np.concatenate([a, b, b, c, c, a])
    cols = np.concatenate([b, a, c, b, a, c])

    try:
        from scipy.sparse import coo_matrix

        data = np.ones(rows.shape[0], dtype=np.int8)
        graph = coo_matrix((data, (rows, cols)), shape=(n_verts, n_verts)).tocsr()
        graph.sum_duplicates()
        graph.data = np.ones_like(graph.data, dtype=np.int8)
        return graph.indptr.astype(np.int64), graph.indices.astype(np.int64)
    except ImportError:
        # Pure-numpy fallback: lexsort + dedup; functionally identical.
        order = np.lexsort((cols, rows))
        rows_s, cols_s = rows[order], cols[order]
        keep = np.ones(rows_s.shape[0], dtype=bool)
        if rows_s.size > 1:
            keep[1:] = (rows_s[1:] != rows_s[:-1]) | (cols_s[1:] != cols_s[:-1])
        rows_u, cols_u = rows_s[keep], cols_s[keep]
        counts = np.bincount(rows_u, minlength=n_verts)
        indptr = np.empty(n_verts + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        return indptr, cols_u.astype(np.int64)


def laplacian_smooth_vf(
    v: np.ndarray, f: np.ndarray, iters: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply ``iters`` rounds of uniform Laplacian smoothing.

    Each iteration replaces every vertex with the mean of its 1-ring neighbors.
    Isolated vertices keep their position; faces are returned unchanged, so
    topology (and therefore watertightness) is preserved.
    """
    v = np.ascontiguousarray(v, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.int64)
    n = int(v.shape[0])
    if n == 0 or iters <= 0:
        return v.copy(), f

    indptr, indices = _build_vertex_neighbors(f, n)
    counts = np.diff(indptr).astype(np.float64)
    has_neighbors = counts > 0.0
    safe_counts = np.where(has_neighbors, counts, 1.0)
    central = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr).astype(np.int64))

    cur = v.copy()
    for _ in range(int(iters)):
        sums = np.zeros_like(cur)
        np.add.at(sums, central, cur[indices])
        new_v = sums / safe_counts[:, None]
        new_v[~has_neighbors] = cur[~has_neighbors]
        cur = new_v

    return cur, f


def load_or_smooth_obj(
    raw_obj_path: Path, cached_obj_path: Path, iters: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a smoothed OBJ from cache, or build it from the raw OBJ.

    The cache is fresh when ``cached_obj_path`` exists and is no older than
    ``raw_obj_path``. Otherwise the raw mesh is loaded (and validated as
    watertight), reduced to its largest connected component, winding-fixed,
    smoothed, and written atomically to the cache.
    """
    raw_obj_path = Path(raw_obj_path)
    cached_obj_path = Path(cached_obj_path)

    if cached_obj_path.is_file():
        try:
            if cached_obj_path.stat().st_mtime >= raw_obj_path.stat().st_mtime:
                v_c, f_c = load_obj_mesh(cached_obj_path)
                return (
                    np.ascontiguousarray(v_c, dtype=np.float64),
                    np.ascontiguousarray(f_c, dtype=np.int64),
                )
        except OSError:
            pass

    v_raw, f_raw = load_obj_mesh(raw_obj_path)
    v_lcc, f_lcc, _n_cc, _kept = largest_connected_component_mesh(v_raw, f_raw)
    f_lcc = ensure_outward_orientation(v_lcc, f_lcc)

    v_smooth, f_smooth = laplacian_smooth_vf(v_lcc, f_lcc, iters=iters)
    write_obj(cached_obj_path, v_smooth, f_smooth)
    return v_smooth, f_smooth

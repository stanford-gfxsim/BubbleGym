"""The eight non-spherical shape features, and the integrals behind them.

The feature vector (Sec. 4.2)
-----------------------------
Two inertia ratios, from the volume-normalized principal moments::

    i11_over_i00 = I11 / I00        i22_over_i00 = I22 / I00

Three Wadell-style sphericity deviations, all zero for a sphere::

    non_sph_va = 1 - Phi_VA         Phi_VA = pi^(1/3) (6V)^(2/3) / A
    non_sph_vm = 1 - Phi_VM         Phi_VM = R_V / R_M,  R_M = M / (4 pi)
    non_sph_w  = 1 - Phi_W          Phi_W  = 4 pi / W_vertex

Three convex-hull ratios, measuring deviation from the convex envelope::

    eta_V = V(bubble) / V(hull)     in (0, 1]
    eta_A = A(bubble) / A(hull)     usually just below 1
    eta_M = M(bubble) / M(hull)     usually just above 1, rising steeply

All three are exactly 1 for a convex body. On the 10k benchmark ``eta_A`` is
below 1 for 99.5 % of bubbles (median 0.998, min 0.44): a dumbbell or a split
bubble has less area than the hull wrapped around it, which is geometrically
correct, not an error. ``eta_M`` is above 1 for 99.9 % (median 1.000, max 3.34),
and it is the ratio that grows with deformation, because concave regions add
mean curvature that the hull smooths away.

Validity of Phi_VA
------------------
``Phi_VA`` is bounded above by 1 for *any* closed surface -- that is the
isoperimetric inequality ``36 pi V^2 <= A^3``, and it holds exactly for the
polyhedron itself, not just its smooth limit. A value above 1 therefore cannot
be a discretization artifact; it means the volume or area is wrong, which in
practice means the mesh is not closed or is inside-out. Such values are
reported as errors rather than clipped into range.

``Phi_W`` has the analogous Willmore bound ``W >= 4 pi``, but ``W_vertex`` is a
*discrete cotangent estimate* that can undershoot on coarse meshes, so it is
only checked for positivity. ``Phi_VM`` has no upper bound for non-convex
shapes (concavities reduce ``M``), so it is likewise only checked for
positivity.

All integrals are scale-sensitive; callers rescale to unit volume first (the
convention used by every column in ``dataset_bubblegym_10k.csv``). The three
eta ratios are scale-invariant regardless.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Relative so the package works under both spellings in use here --
# ``import shape_feature`` (python/ on sys.path) and ``import
# python.shape_feature`` (repo root on sys.path).
from .mesh_utils import (
    MeshError,
    load_obj_mesh,
    mesh_signed_volume,
    mesh_surface_area,
    vertices_scaled_to_target_volume,
)


__all__ = [
    # feature-column conventions
    "WADELL_FEATURE_COLS",
    "CHULL_FEATURE_COLS",
    "INERTIA_FEATURE_COLS",
    "NONSPH_CHULL_FEATURE_COLS",
    "FEATURE_COLS_8",
    # curvature integrals
    "mean_curvature_integral",
    "willmore_energy_vertex",
    "curvature_integrals_from_vf",
    # convex hull
    "convex_hull_vf",
    "convex_hull_features_from_vf",
    # sphericities / features
    "InvalidSphericityError",
    "wadell_phi",
    "non_sph_va_from_area_volume",
    "sphericities_from_integrals",
    "nonspherical_features_from_unit_mesh",
    "mesh_feature_row",
    "compute_nonsphericity_columns",
]


WADELL_FEATURE_COLS = ["non_sph_va", "non_sph_vm", "non_sph_w"]
CHULL_FEATURE_COLS = ["eta_V", "eta_A", "eta_M"]
INERTIA_FEATURE_COLS = ["i11_over_i00", "i22_over_i00"]
NONSPH_CHULL_FEATURE_COLS = WADELL_FEATURE_COLS + CHULL_FEATURE_COLS
FEATURE_COLS_8 = INERTIA_FEATURE_COLS + NONSPH_CHULL_FEATURE_COLS

_EPS = 1e-30
# Slack for float round-off only. Phi_VA <= 1 is exact for a closed polyhedron,
# so anything beyond this is a real defect, not discretization.
_PHI_VA_TOL = 1e-9


class InvalidSphericityError(MeshError):
    """Raised when a sphericity falls outside its mathematically valid range."""


# ---------------------------------------------------------------------------
# Discrete curvature integrals
# ---------------------------------------------------------------------------


def _face_normals_and_areas(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(cross, axis=1)
    areas = 0.5 * norm
    safe = np.where(norm > 0.0, norm, 1.0)
    return cross / safe[:, None], areas


def _edge_dihedral_table(v: np.ndarray, f: np.ndarray) -> dict[str, np.ndarray]:
    """Interior edges with lengths, signed dihedrals, and mean incident area.

    Returns equal-length arrays indexed by interior edge: ``l`` (edge length),
    ``theta`` (signed dihedral, convex-positive, radians), and ``A_inc`` (mean
    of the two incident triangle areas).
    """
    normals, areas = _face_normals_and_areas(v, f)

    n_faces = f.shape[0]
    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    e_a = np.stack([i1, i2, np.arange(n_faces), i0], axis=1)
    e_b = np.stack([i2, i0, np.arange(n_faces), i1], axis=1)
    e_c = np.stack([i0, i1, np.arange(n_faces), i2], axis=1)
    E = np.concatenate([e_a, e_b, e_c], axis=0)

    vs = np.minimum(E[:, 0], E[:, 1])
    vl = np.maximum(E[:, 0], E[:, 1])
    key = np.stack([vs, vl], axis=1)

    order = np.lexsort((key[:, 1], key[:, 0]))
    E_sorted = E[order]
    key_sorted = key[order]

    same_as_prev = np.zeros(len(E_sorted), dtype=bool)
    same_as_prev[1:] = (key_sorted[1:, 0] == key_sorted[:-1, 0]) & (
        key_sorted[1:, 1] == key_sorted[:-1, 1]
    )
    pair_second = np.where(same_as_prev)[0]
    pair_start = pair_second - 1

    pairs_a = E_sorted[pair_start]
    pairs_b = E_sorted[pair_second]

    edge_vec = v[pairs_a[:, 1]] - v[pairs_a[:, 0]]
    l = np.linalg.norm(edge_vec, axis=1)

    fa, fb = pairs_a[:, 2], pairs_b[:, 2]
    n1, n2 = normals[fa], normals[fb]
    cross12 = np.cross(n1, n2)
    dot12 = np.clip(np.einsum("ij,ij->i", n1, n2), -1.0, 1.0)
    sin_mag = np.linalg.norm(cross12, axis=1)
    sign = np.where(np.einsum("ij,ij->i", cross12, edge_vec) >= 0.0, 1.0, -1.0)

    return {
        "l": l,
        "theta": np.arctan2(sign * sin_mag, dot12),
        "A_inc": 0.5 * (areas[fa] + areas[fb]),
    }


def mean_curvature_integral(v: np.ndarray, f: np.ndarray) -> float:
    """``M = sum_e l_e * theta_e / 2`` over interior edges.

    Precondition: a closed, 2-manifold mesh, so every edge is shared by
    exactly two faces. The edge table pairs consecutive sorted edge keys: an
    edge on three faces would be counted twice and a boundary edge silently
    dropped, with no error. ``load_obj_mesh`` enforces watertightness, so the
    normal path is safe; calling this directly on an unchecked mesh is not.
    """
    t = _edge_dihedral_table(v, f)
    return float(0.5 * np.sum(t["l"] * t["theta"]))


def willmore_energy_vertex(v: np.ndarray, f: np.ndarray) -> float:
    """Cotangent mean-curvature-normal Willmore estimate, ``sum_v H_v^2 A_v``.

    Raises ``ImportError`` if ``igl`` is unavailable. This is deliberate: a
    missing dependency is an environment error, not a property of the mesh.
    Returning NaN would flow into ``W_vertex`` for every row, then into
    ``non_sph_w``, and those rows would be silently dropped much later by the
    trainer's ``dropna`` -- a quietly degraded dataset instead of a clear fault.
    """
    try:
        import igl
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "willmore_energy_vertex requires the 'igl' package (pip install libigl) "
            "to compute the W_vertex column. Install it, or drop W_vertex-derived "
            "features (non_sph_w) from your feature set."
        ) from exc

    V = np.ascontiguousarray(v, dtype=np.float64)
    F = np.ascontiguousarray(f, dtype=np.int32)

    L = igl.cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    m_diag = np.asarray(M.diagonal(), dtype=np.float64)
    m_safe = np.where(m_diag > 0.0, m_diag, 1.0)

    HN = -(L @ V) / m_safe[:, None]
    H = 0.5 * np.linalg.norm(HN, axis=1)
    return float(np.sum((H ** 2) * m_diag))


def curvature_integrals_from_vf(
    v: np.ndarray, f: np.ndarray, *, target_volume: float | None = 1.0
) -> dict[str, float]:
    """Return ``{M, W_vertex, area, volume}`` on a (rescaled) mesh.

    Same precondition as :func:`mean_curvature_integral`: the mesh must be
    closed and 2-manifold, or ``M`` is wrong without an error.
    """
    if target_volume is not None:
        v = vertices_scaled_to_target_volume(v, f, target_volume)
    return {
        "M": mean_curvature_integral(v, f),
        "W_vertex": willmore_energy_vertex(v, f),
        "area": mesh_surface_area(v, f),
        "volume": abs(mesh_signed_volume(v, f)),
    }


# ---------------------------------------------------------------------------
# Convex hull
# ---------------------------------------------------------------------------


def convex_hull_vf(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(v_hull, f_hull)`` with consistently outward-oriented faces.

    Uses ``scipy.spatial.ConvexHull`` (Qhull). Faces are triangles whose
    right-hand-rule normals are aligned with ``ConvexHull.equations[:, :3]``,
    Qhull's canonical outward plane normals.
    """
    from scipy.spatial import ConvexHull

    pts = np.ascontiguousarray(points, dtype=np.float64)
    hull = ConvexHull(pts)

    used = np.unique(hull.simplices.ravel())
    remap = np.full(pts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])

    v_hull = pts[used]
    simplices = remap[hull.simplices].astype(np.int64)

    e0 = v_hull[simplices[:, 1]] - v_hull[simplices[:, 0]]
    e1 = v_hull[simplices[:, 2]] - v_hull[simplices[:, 0]]
    align = np.einsum("ij,ij->i", np.cross(e0, e1), np.asarray(hull.equations[:, :3]))
    flip = align < 0.0
    if np.any(flip):
        simplices[flip] = simplices[flip][:, [0, 2, 1]]

    return v_hull, simplices


def convex_hull_features_from_vf(
    v: np.ndarray, f: np.ndarray, *, target_volume: float | None = 1.0
) -> dict[str, float]:
    """Bubble and hull integrals plus the three dimensionless eta ratios.

    The bubble mesh is assumed to be a closed surface with outward-oriented
    faces (use ``mesh_utils.largest_connected_component_mesh`` and
    ``ensure_outward_orientation`` first). Returns ``V, A, M``, ``V_hull,
    A_hull, M_hull``, and ``eta_V, eta_A, eta_M``.

    Willmore is deliberately not computed on the hull: it is ill-conditioned on
    polyhedra with sharp edges.
    """
    v = np.ascontiguousarray(v, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.int64)

    if target_volume is not None:
        v = vertices_scaled_to_target_volume(v, f, target_volume)

    V_bub = abs(mesh_signed_volume(v, f))
    A_bub = mesh_surface_area(v, f)
    M_bub = mean_curvature_integral(v, f)

    v_hull, f_hull = convex_hull_vf(v)
    V_hull = abs(mesh_signed_volume(v_hull, f_hull))
    A_hull = mesh_surface_area(v_hull, f_hull)
    M_hull = mean_curvature_integral(v_hull, f_hull)

    return {
        "V": float(V_bub),
        "A": float(A_bub),
        "M": float(M_bub),
        "V_hull": float(V_hull),
        "A_hull": float(A_hull),
        "M_hull": float(M_hull),
        "eta_V": float(V_bub / max(V_hull, _EPS)),
        "eta_A": float(A_bub / max(A_hull, _EPS)),
        "eta_M": float(M_bub / max(M_hull, _EPS)),
    }


# ---------------------------------------------------------------------------
# Sphericities -- one implementation, two interfaces
# ---------------------------------------------------------------------------


def wadell_phi(surface_area, volume):
    """Wadell sphericity ``Phi_VA = pi^(1/3) (6V)^(2/3) / A``.

    Accepts scalars or arrays. Returns NaN where the inputs are non-finite or
    non-positive. The result is **not** clipped: see the module docstring for
    why a value above 1 is an error rather than something to squash.
    """
    surface_area = np.asarray(surface_area, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    out = np.full(np.broadcast(surface_area, volume).shape, np.nan, dtype=np.float64)
    ok = (
        np.isfinite(surface_area)
        & np.isfinite(volume)
        & (surface_area > 0.0)
        & (volume > 0.0)
    )
    sa = np.broadcast_to(surface_area, out.shape)
    vol = np.broadcast_to(volume, out.shape)
    out[ok] = (
        (math.pi ** (1.0 / 3.0)) * (6.0 * vol[ok]) ** (2.0 / 3.0) / sa[ok]
    )
    return out


def _phi_va_violations(phi_va: np.ndarray) -> np.ndarray:
    """Boolean mask of Phi_VA values that are finite but out of range."""
    phi_va = np.asarray(phi_va, dtype=np.float64)
    return np.isfinite(phi_va) & ((phi_va <= 0.0) | (phi_va > 1.0 + _PHI_VA_TOL))


def non_sph_va_from_area_volume(surface_area, volume, *, warn: bool = True, context: str = ""):
    """``1 - Phi_VA`` for scalars or arrays, with out-of-range values as NaN.

    Deliberately does not clip: clipping turns an impossible ``Phi_VA > 1`` into
    ``non_sph_va = 0``, i.e. reports a defective mesh as a perfect sphere.
    Out-of-range values become NaN and are reported, so they are dropped
    downstream instead of being disguised.
    """
    phi = wadell_phi(surface_area, volume)
    bad = _phi_va_violations(phi)
    if np.any(bad):
        if warn:
            where = f"{context}: " if context else ""
            worst = float(np.nanmax(np.abs(np.asarray(phi)[bad])))
            print(
                f"WARNING: {where}{int(np.count_nonzero(bad))} of {np.size(phi)} "
                f"Phi_VA values are outside (0, 1] (worst |Phi_VA| = {worst:.6g}). "
                "Phi_VA <= 1 is exact for any closed polyhedron, so these come "
                "from an inconsistent surface_area/volume pair -- most likely a "
                "mesh that is not closed. Set to NaN rather than clipped."
            )
        phi = np.where(bad, np.nan, phi)
    return 1.0 - phi


def sphericities_from_integrals(
    *, volume: float, area: float, mean_curvature: float, willmore_vertex: float
) -> dict[str, float]:
    """Return ``{Phi_VA, Phi_VM, Phi_W}`` for one mesh.

    Raises :class:`InvalidSphericityError` if ``Phi_VA`` leaves ``(0, 1]``,
    which indicates a defective mesh rather than a discretization effect.
    ``Phi_VM`` and ``Phi_W`` are checked only for positivity (see module
    docstring).
    """
    phi_va = float(wadell_phi(area, volume))
    if not np.isfinite(phi_va):
        raise InvalidSphericityError(
            f"Phi_VA is not finite (volume={volume!r}, area={area!r})"
        )
    if _phi_va_violations(np.array([phi_va]))[0]:
        raise InvalidSphericityError(
            f"Phi_VA = {phi_va:.12g} is outside the valid range (0, 1]. "
            "Phi_VA <= 1 is the isoperimetric inequality and holds exactly for "
            "any closed polyhedron, so this indicates a defective mesh "
            "(open surface, inverted winding, or self-intersection) rather "
            f"than discretization error. volume={volume!r}, area={area!r}."
        )

    r_v = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    r_m = mean_curvature / (4.0 * math.pi)
    if not (np.isfinite(r_m) and r_m > 0.0):
        raise InvalidSphericityError(
            f"mean-curvature radius R_M = {r_m!r} is not positive "
            f"(M={mean_curvature!r})"
        )
    if not (np.isfinite(willmore_vertex) and willmore_vertex > 0.0):
        raise InvalidSphericityError(
            f"W_vertex = {willmore_vertex!r} is not positive"
        )

    return {
        "Phi_VA": phi_va,
        "Phi_VM": float(r_v / r_m),
        "Phi_W": float(4.0 * math.pi / willmore_vertex),
    }


def nonspherical_features_from_unit_mesh(
    v_unit: np.ndarray, f: np.ndarray
) -> dict[str, Any]:
    """The six shape axes for a mesh already rescaled to unit volume.

    Returns ``non_sph_va``, ``non_sph_vm``, ``non_sph_w``, ``eta_V``, ``eta_A``,
    ``eta_M``, the raw integrals they derive from, and ``status``.

    ``status`` is ``"ok"`` on success, or ``"failed: <reason>"`` -- the contract
    the inference predictors consume, so one bad frame cannot abort a 30k-frame
    scene. Callers that would rather see the exception should call
    :func:`curvature_integrals_from_vf`, :func:`convex_hull_features_from_vf`,
    and :func:`sphericities_from_integrals` directly.

    The two inertia ratios that complete the 8-feature vector come from
    ``mesh_utils.compute_mirtich_moments``; they are supplied separately
    because callers often already have them.
    """
    feats: dict[str, Any] = {}
    try:
        curv = curvature_integrals_from_vf(v_unit, f, target_volume=None)
        phi = sphericities_from_integrals(
            volume=float(curv["volume"]),
            area=float(curv["area"]),
            mean_curvature=float(curv["M"]),
            willmore_vertex=float(curv["W_vertex"]),
        )
        hull = convex_hull_features_from_vf(v_unit, f, target_volume=None)
        feats.update(
            {
                "volume_unit": float(curv["volume"]),
                "surface_area_unit": float(curv["area"]),
                "M_unit": float(curv["M"]),
                "W_vertex_unit": float(curv["W_vertex"]),
                "non_sph_va": 1.0 - phi["Phi_VA"],
                "non_sph_vm": 1.0 - phi["Phi_VM"],
                "non_sph_w": 1.0 - phi["Phi_W"],
                "eta_V": float(hull["eta_V"]),
                "eta_A": float(hull["eta_A"]),
                "eta_M": float(hull["eta_M"]),
                "status": "ok",
            }
        )
    except Exception as exc:  # noqa: BLE001
        feats["status"] = f"failed: {type(exc).__name__}: {exc}"
    return feats


def mesh_feature_row(obj_path: Path) -> dict[str, float | int | str]:
    """Per-mesh sphericity row for one OBJ, keyed for the analysis plots.

    Loads (validating watertightness), keeps the largest connected component,
    fixes winding, rescales to unit volume, and returns the raw integrals plus
    ``Phi_VA`` / ``Phi_VM`` / ``Phi_W_vertex``. Built entirely on the shared
    primitives above, so it cannot drift from the trainer's feature definitions.
    """
    from .mesh_utils import (
        ensure_outward_orientation,
        largest_connected_component_mesh,
    )

    obj_path = Path(obj_path)
    v_raw, f = load_obj_mesh(obj_path)
    v, f, n_components, kept_faces = largest_connected_component_mesh(v_raw, f)
    f = ensure_outward_orientation(v, f)

    curv = curvature_integrals_from_vf(v, f, target_volume=1.0)
    phi = sphericities_from_integrals(
        volume=float(curv["volume"]),
        area=float(curv["area"]),
        mean_curvature=float(curv["M"]),
        willmore_vertex=float(curv["W_vertex"]),
    )
    return {
        "mesh_file": obj_path.name,
        "mesh_path": str(obj_path),
        "surface_area": float(curv["area"]),
        "volume": float(curv["volume"]),
        "M": float(curv["M"]),
        "R_M": float(curv["M"]) / (4.0 * math.pi),
        "W_vertex": float(curv["W_vertex"]),
        "Phi_VA": phi["Phi_VA"],
        "Phi_VM": phi["Phi_VM"],
        "Phi_W_vertex": phi["Phi_W"],
        "n_components": int(n_components),
        "kept_faces": int(kept_faces),
    }


def compute_nonsphericity_columns(
    df: pd.DataFrame,
    *,
    surface_area_col: str = "surface_area",
    volume_col: str = "volume",
    phi_vm_col: str = "Phi_VM",
    willmore_col: str = "W_vertex",
    warn: bool = True,
) -> pd.DataFrame:
    """Add ``non_sph_va`` / ``non_sph_vm`` / ``non_sph_w`` to a dataset frame.

    Returns a copy; existing columns are untouched. Rows whose ``Phi_VA`` falls
    outside ``(0, 1]`` get NaN (and are reported when ``warn``), because such a
    value means the row's volume/area pair is inconsistent -- see the module
    docstring. Downstream trainers drop NaN feature rows.

    This does not raise, because it operates on a whole CSV where a handful of
    bad rows should not abort training; the per-mesh path
    (:func:`sphericities_from_integrals`) does raise.
    """
    out = df.copy()
    surface_area = pd.to_numeric(out[surface_area_col], errors="coerce").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(out[volume_col], errors="coerce").to_numpy(dtype=np.float64)
    phi_vm = pd.to_numeric(out[phi_vm_col], errors="coerce").to_numpy(dtype=np.float64)
    w_vertex = pd.to_numeric(out[willmore_col], errors="coerce").to_numpy(dtype=np.float64)

    phi_va = wadell_phi(surface_area, volume)
    bad = _phi_va_violations(phi_va)
    if np.any(bad):
        if warn:
            worst = float(np.nanmax(np.abs(phi_va[bad])))
            print(
                f"WARNING: {int(np.count_nonzero(bad))} of {len(phi_va)} rows have "
                f"Phi_VA outside (0, 1] (worst |Phi_VA| = {worst:.6g}). Phi_VA <= 1 "
                "is exact for any closed polyhedron, so these rows have an "
                "inconsistent surface_area/volume pair -- most likely the mesh was "
                "not closed. They are set to NaN and will be dropped downstream; "
                "investigate rather than ignore."
            )
        phi_va = np.where(bad, np.nan, phi_va)

    with np.errstate(divide="ignore", invalid="ignore"):
        phi_w = np.where(w_vertex > 0.0, 4.0 * math.pi / w_vertex, np.nan)

    out["non_sph_va"] = 1.0 - phi_va
    out["non_sph_vm"] = 1.0 - phi_vm
    out["non_sph_w"] = 1.0 - phi_w
    return out

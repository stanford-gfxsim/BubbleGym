"""
Mesh loading, bempp-cl import shims, and the whole-file CLI entry point.

This module owns the parts of the BEM pipeline that do not depend on a
discretization:

* ``_load_triangle_mesh`` -- meshio-based loader with a minimal OBJ fallback,
  plus ``_compute_volume`` / ``_signed_volume`` / ``_normalize_to_unit_volume``.
* ``_import_bempp_api`` / ``_make_bempp_grid`` -- shims over the two packagings
  of bempp-cl (``bempp.api`` and ``bempp_cl.api``) and their grid constructors.
* ``solve_minnaert_frequency_mesh`` / ``solve_minnaert_frequency`` -- thin
  wrappers that print a report and delegate the actual solve to
  ``compute_freq_bempp_galerkin.solve_minnaert_frequency_galerkin`` (P1-DP0).

The physics: exterior Laplace Dirichlet on a closed surface Γ with φ|_Γ = 1
gives the Neumann trace q = ∂φ/∂n, hence the capacitance C = -(1/(4π)) ∫_Γ q dS
and the Minnaert frequency f = (1/(2π)) sqrt(4π γ P0 C / (ρ V0)).

Usage (PowerShell):
  python python/bem/bempp_io.py path/to/mesh.obj
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


def _minnaert_constant() -> float:
    """The Minnaert constant ``f * R`` for air in water, from ``freq_model``.

    Single source of truth for the analytical reference these scripts print;
    the ``3.26`` that used to be hard-coded here is 0.7% low.
    """
    py_root = str(Path(__file__).resolve().parents[1])
    if py_root not in sys.path:
        sys.path.insert(0, py_root)
    from freq_model.analytical.minnaert_freq import MINNAERT_CONSTANT

    return float(MINNAERT_CONSTANT)


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed divergence-theorem volume; positive iff the winding is outward."""
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _compute_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed-tet volume magnitude, matching the C++ implementation."""
    return abs(_signed_volume(vertices, faces))


def _normalize_to_unit_volume(vertices: np.ndarray, volume: float) -> Tuple[np.ndarray, float]:
    if volume < 1e-12:
        return vertices, 1.0
    scale = (1.0 / volume) ** (1.0 / 3.0)
    return vertices * scale, float(scale)


def _load_triangle_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a triangle surface mesh as (V Nx3, F Mx3).

    Tries `meshio` first (OBJ/STL/PLY/etc). If unavailable, falls back to a tiny OBJ reader.
    """
    ext = os.path.splitext(path)[1].lower()

    try:
        import meshio  # type: ignore

        mesh = meshio.read(path)
        if mesh.points is None or len(mesh.points) == 0:
            raise ValueError("meshio: no points found")
        vertices = np.asarray(mesh.points, dtype=np.float64)
        if vertices.shape[1] != 3:
            raise ValueError(f"Expected 3D points, got shape {vertices.shape}")

        tris = []
        for cell_block in mesh.cells:
            if cell_block.type == "triangle":
                tris.append(np.asarray(cell_block.data, dtype=np.int64))
        if not tris:
            raise ValueError("meshio: no triangle cells found")
        faces = np.vstack(tris)
        return vertices, faces
    except ModuleNotFoundError:
        if ext != ".obj":
            raise RuntimeError(
                "meshio is not installed, and only .obj fallback loader is available. "
                "Please `pip install meshio` or provide an OBJ mesh."
            )
    except Exception:
        # If meshio exists but fails for any reason, try OBJ fallback if applicable
        if ext != ".obj":
            raise

    # Minimal OBJ loader (supports 'v' and triangular 'f' only)
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) != 3:
                    continue
                idx = []
                for p in parts:
                    # f entries can be: v, v/vt, v//vn, v/vt/vn
                    v_str = p.split("/")[0]
                    vi = int(v_str)
                    if vi < 0:
                        vi = len(verts) + 1 + vi
                    idx.append(vi - 1)
                faces.append(idx)

    vertices = np.asarray(verts, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("OBJ fallback loader failed to parse a triangular surface mesh.")
    return vertices, faces_arr


def _make_bempp_grid(vertices: np.ndarray, faces: np.ndarray):
    """
    Build a BEM++ grid from (V Nx3, F Mx3).

    BEM++ expects vertices as shape (3, N) and elements as shape (3, M).
    """
    verts = np.asarray(vertices, dtype=np.float64).T
    elems = np.asarray(faces, dtype=np.int64).T

    # bempp-cl API variants exist; try the common constructors.
    bempp_api = _import_bempp_api()
    if hasattr(bempp_api, "grid_from_element_data"):
        return bempp_api.grid_from_element_data(verts, elems)
    if hasattr(bempp_api, "Grid"):
        return bempp_api.Grid(verts, elems)
    if hasattr(bempp_api, "grid") and hasattr(bempp_api.grid, "grid_from_element_data"):
        return bempp_api.grid.grid_from_element_data(verts, elems)
    raise RuntimeError("Could not construct a BEM++ grid (unsupported bempp.api version).")


def _import_bempp_api():
    """
    Return the BEM++ API module.

    Depending on how `bempp-cl` is packaged in the current environment, the API is exposed as either:
      - `bempp.api` (most common)
      - `bempp_cl.api` (some installs)
    """
    try:
        import bempp.api as bempp_api  # type: ignore

        return bempp_api
    except ModuleNotFoundError:
        pass

    import bempp_cl.api as bempp_api  # type: ignore

    return bempp_api


def solve_minnaert_frequency_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    gamma: float = 1.4,
    p0: float = 101325.0,
    rho: float = 1000.0,
    verbose: bool = False,
) -> dict:
    """
    Exterior Dirichlet Laplace BEM on a closed bubble mesh: φ|_Γ = 1, no outer boundaries.

    Uses the mesh **as given** (no rescaling to unit volume). Implementation:
    :func:`python.bem.compute_freq_bempp_galerkin.solve_minnaert_frequency_galerkin`
    with **P1-DP0** (conforming P1 Dirichlet / DP0 Neumann) and GMRES with mass
    preconditioning.

    Returns
    -------
    dict with keys: frequency, capacitance, capacitance_raw, v0, gmres_info,
    n_panels, method, plus galerkin-specific keys forwarded when present.
    """
    from .compute_freq_bempp_galerkin import solve_minnaert_frequency_galerkin

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    v0 = _compute_volume(vertices, faces)
    if v0 <= 1e-12:
        raise ValueError("Bubble mesh has non-positive volume.")

    out = solve_minnaert_frequency_galerkin(
        (vertices, faces),
        trial_pair="P1-DP0",
        precond="mass",
        solver="gmres",
        gmres_tol=1e-12,
        gmres_maxiter=4000,
        gmres_restart=300,
        quadrature_regular=6,
        quadrature_singular=6,
        gamma=gamma,
        p0=p0,
        rho=rho,
        skip_volume_rescale=True,
        verbose=verbose,
    )

    return {
        "frequency": float(out["frequency"]),
        "capacitance": float(out["capacitance"]),
        "capacitance_raw": float(out["capacitance_raw"]),
        "v0": float(out["v0"]),
        "gmres_info": int(out.get("gmres_info", 0)),
        "n_panels": int(out.get("n_triangles", faces.shape[0])),
        "method": "galerkin_p1_dp0",
        "galerkin_trial_pair": out.get("trial_pair", "P1-DP0"),
    }


def solve_minnaert_frequency(
    mesh_path: str,
    gamma: float = 1.4,
    p0: float = 101325.0,
    rho: float = 1000.0,
    verbose: bool = True,
):
    try:
        _import_bempp_api()
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "BEM++ (bempp-cl) is not installed in this Python environment.\n"
            "\n"
            "Install options:\n"
            "  - If you already have a bempp-cl environment (often via conda/mamba), run this script with that Python.\n"
            "  - Otherwise, create a conda env and install bempp-cl there (recommended). On Windows, this is often easiest via WSL2.\n"
            "\n"
            "After installation, rerun:\n"
            "  python python/bem/bempp_io.py python/bem/test_meshes/sphere_res20_reverse_N.obj\n"
        ) from e

    vertices, faces = _load_triangle_mesh(mesh_path)
    m = faces.shape[0]
    if verbose:
        print(f"Loading mesh from: {mesh_path}")
        print(f"Mesh: {vertices.shape[0]} vertices, {m} triangles")

    vol = _compute_volume(vertices, faces)
    if verbose:
        print(f"  Volume (before normalize) = {vol}")
    if abs(vol - 1.0) > 1e-9:
        vertices, scale = _normalize_to_unit_volume(vertices, vol)
        if verbose:
            print(f"  Rescaled mesh to unit volume (scale = {scale})")

    # Recompute volume after scaling (matches C++ test behavior)
    out = solve_minnaert_frequency_mesh(
        vertices, faces, gamma=gamma, p0=p0, rho=rho, verbose=False
    )
    numerical_freq = out["frequency"]
    c_val = out["capacitance_raw"]
    v0 = out["v0"]
    info = out["gmres_info"]

    if info not in (0, None) and verbose:
        print(f"GMRES returned info={info} (0 means converged).")

    if verbose:
        print("\nResults:")
        print(f"  C (raw) = {c_val}")
        if c_val < 0:
            print("  Note: C is negative (likely normal/sign convention). Using |C| for frequency.")
        print(f"  Volume V0 = {v0}")

    if verbose:
        print(f"  Solved Minnaert frequency f = {numerical_freq} Hz")

        # Same analytical comparison as C++ test (unit-volume sphere)
        volume = 1.0
        radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        analytical_freq = _minnaert_constant() / radius
        print(f"\n  Analytical Minnaert frequency for unit sphere: f = {analytical_freq} Hz, volume V0 = {volume}")

        err = abs(numerical_freq - analytical_freq) / analytical_freq * 100.0
        vol_err = abs(v0 - volume) / volume * 100.0
        print(f"  Error = {err}%")
        print(f"  Volume Error = {vol_err}%")

    return numerical_freq, c_val, v0, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_file", help="Path to a closed triangle mesh (OBJ/STL/PLY/...; requires meshio for non-OBJ).")
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--p0", type=float, default=101325.0)
    parser.add_argument("--rho", type=float, default=1000.0)
    args = parser.parse_args()

    solve_minnaert_frequency(args.mesh_file, gamma=args.gamma, p0=args.p0, rho=args.rho)


if __name__ == "__main__":
    main()

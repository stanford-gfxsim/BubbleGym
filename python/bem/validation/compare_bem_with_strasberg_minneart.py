"""
Compare BEM (Galerkin P1-DP0) vs Strasberg vs Minnaert on two test meshes.

For each mesh we:

  1.  Load the raw triangle mesh.
  2.  Inspect face normals via the signed tetrahedral volume:
        V_signed > 0  =>  outward-facing normals (canonical, required by BEM)
        V_signed < 0  =>  inward-facing normals; flip triangle winding.
  3.  Centre the mesh at the centroid and rescale so |V| = 1
      (unit-volume convention shared by the rest of the repo).
  4.  Compute the volume-normalized volumetric inertia tensor via
      Mirtich's exact integration over triangles.
  5.  Run three frequency estimators:
        - Minnaert (volume-only, isotropic sphere formula at V = 1)
        - Strasberg (best-fit ellipsoid from the inertia tensor)
        - BEM      (Galerkin P1-DP0, using the unit-volume mesh directly)

  6.  Report capacitance, frequency and relative differences.

Air-in-water conditions are used throughout (gamma=1.4, p0=101325 Pa,
rho=1000 kg/m^3) so the three estimators are directly comparable.

Run from repo root:

    python -m python.bem.validation.compare_bem_with_strasberg_minneart
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Make `python.*` imports work when run as a script.
_ROOT = _repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from python.bem.bempp_io import _load_triangle_mesh  # noqa: E402
from python.bem.compute_freq_bempp_galerkin import (  # noqa: E402
    solve_minnaert_frequency_galerkin,
)
from python.freq_model.analytical.minnaert_freq import (  # noqa: E402
    minnaert_frequency_unit_volume,
)
from python.freq_model.analytical.strasberg_freq import (  # noqa: E402
    semi_axes_from_normalized_inertia,
    strasberg_frequency,
)
from python.shape_feature.mesh_utils import (  # noqa: E402
    compute_mirtich_moments as _mirtich_inertia,
    mesh_signed_volume as _signed_volume,
)

# Physical constants (air-in-water) -- kept in lockstep with both solvers.
GAMMA = 1.4
P0 = 101_325.0
RHO = 1000.0


# --------------------------------------------------------------------------- #
# Mesh helpers
# --------------------------------------------------------------------------- #
def _flip_winding(f: np.ndarray) -> np.ndarray:
    g = f.copy()
    g[:, [1, 2]] = g[:, [2, 1]]
    return g


# --------------------------------------------------------------------------- #
# Preprocessing: orient outward + unit volume + centred
# --------------------------------------------------------------------------- #
@dataclass
class PreparedMesh:
    path: Path
    V: np.ndarray                # (N, 3) unit-volume, centred
    F: np.ndarray                # (M, 3) outward-oriented
    V_raw_signed: float          # signed volume of the input mesh
    flipped: bool                # True iff winding was flipped
    centroid_raw: np.ndarray     # centroid that was translated to origin
    scale_applied: float         # vertex scale factor (s.t. V_final = 1)
    I_norm: np.ndarray           # volume-normalized volumetric inertia (3x3)


def _prepare_mesh(path: Path) -> PreparedMesh:
    v_raw, f_raw = _load_triangle_mesh(str(path))
    v_raw = np.asarray(v_raw, dtype=np.float64)
    f_raw = np.asarray(f_raw, dtype=np.int64)

    v_signed_raw = _signed_volume(v_raw, f_raw)
    if v_signed_raw == 0.0 or not math.isfinite(v_signed_raw):
        raise ValueError(f"Degenerate mesh volume for {path}: {v_signed_raw}")

    flipped = False
    f_oriented = f_raw
    if v_signed_raw < 0.0:
        f_oriented = _flip_winding(f_raw)
        flipped = True

    centroid = v_raw.mean(axis=0)
    v_centred = v_raw - centroid

    vol_abs = abs(_signed_volume(v_centred, f_oriented))
    if vol_abs < 1e-18:
        raise ValueError(f"Zero volume after centring for {path}")
    scale = (1.0 / vol_abs) ** (1.0 / 3.0)
    v_unit = v_centred * scale

    vol_check = _signed_volume(v_unit, f_oriented)
    if not (0.99 < vol_check < 1.01):
        raise ValueError(
            f"Unit-volume rescale failed for {path}: signed volume = {vol_check}"
        )

    mass, _cm_unit, I_unit = _mirtich_inertia(v_unit, f_oriented)
    # mass from Mirtich equals the signed volume under unit density; take |.| to
    # be safe against residual orientation ambiguity.
    I_norm = I_unit / abs(mass)

    return PreparedMesh(
        path=path,
        V=v_unit.astype(np.float64),
        F=f_oriented.astype(np.int64),
        V_raw_signed=float(v_signed_raw),
        flipped=flipped,
        centroid_raw=centroid.astype(np.float64),
        scale_applied=float(scale),
        I_norm=I_norm.astype(np.float64),
    )


# --------------------------------------------------------------------------- #
# Frequency comparison
# --------------------------------------------------------------------------- #
@dataclass
class ComparisonResult:
    mesh_name: str
    n_vertices: int
    n_triangles: int
    v_signed_raw: float
    flipped: bool
    principal_moments: np.ndarray
    semi_axes: np.ndarray
    f_minnaert: float
    f_strasberg: float
    f_bem: float
    c_strasberg: float
    c_bem: float
    bem_info: dict


def _compare_one(mesh_path: Path, *, trial_pair: str = "P1-DP0") -> ComparisonResult:
    prep = _prepare_mesh(mesh_path)

    # Strasberg: the implementation enforces a strict
    #   V == (4/3) pi prod(a)
    # consistency check, and real meshes are never *exactly* ellipsoidal. So we
    # evaluate at V_axes (the volume implied by the inertia-derived ellipsoid)
    # and then rescale back to V = 1 using f ~ V^{-1/3}. Matches
    # strasberg_frequency_unit_volume() but also exposes the intermediate
    # capacitance / semi-axes for reporting.
    a_axes = semi_axes_from_normalized_inertia(prep.I_norm)
    V_axes = (4.0 / 3.0) * math.pi * float(np.prod(a_axes))
    strasberg_res = strasberg_frequency(
        prep.I_norm, V_axes, gamma=GAMMA, p0=P0, rho_liquid=RHO
    )
    f_strasberg_unit_volume = float(strasberg_res.frequency_hz) * (V_axes ** (1.0 / 3.0))

    # Minnaert (volume-only).
    f_minnaert = minnaert_frequency_unit_volume(gamma=GAMMA, p0=P0, rho_liquid=RHO)

    # BEM -- feed a pre-built (V, F) tuple so the solver does NOT re-centre or
    # re-scale; we already produced an outward-oriented unit-volume mesh.
    bem_out = solve_minnaert_frequency_galerkin(
        (prep.V, prep.F),
        trial_pair=trial_pair,
        precond="mass",
        solver="gmres",
        gmres_tol=1e-12,
        gmres_maxiter=4000,
        quadrature_regular=6,
        quadrature_singular=6,
        gamma=GAMMA,
        p0=P0,
        rho=RHO,
        verbose=False,
    )

    principal = np.sort(np.linalg.eigvalsh(prep.I_norm))
    return ComparisonResult(
        mesh_name=str(mesh_path),
        n_vertices=int(prep.V.shape[0]),
        n_triangles=int(prep.F.shape[0]),
        v_signed_raw=float(prep.V_raw_signed),
        flipped=bool(prep.flipped),
        principal_moments=principal.astype(np.float64),
        semi_axes=strasberg_res.semi_axes.astype(np.float64),
        f_minnaert=float(f_minnaert),
        f_strasberg=float(f_strasberg_unit_volume),
        f_bem=float(bem_out["frequency"]),
        c_strasberg=float(strasberg_res.capacitance),
        c_bem=float(bem_out["capacitance"]),
        bem_info=bem_out,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_vec(v: np.ndarray, fmt: str = "{:+.6e}") -> str:
    return "[" + ", ".join(fmt.format(float(x)) for x in v) + "]"


def _print_report(res: ComparisonResult) -> None:
    r_eq_unit = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

    print("=" * 72)
    print(f"Mesh: {res.mesh_name}")
    print("=" * 72)
    print(f"  #vertices        = {res.n_vertices}")
    print(f"  #triangles       = {res.n_triangles}")
    print(f"  raw signed vol   = {res.v_signed_raw:+.6e}")
    print(
        f"  normals check    = {'INWARD (flipped to outward)' if res.flipped else 'outward (OK)'}"
    )
    print(f"  BEM DOFs         : dirichlet={res.bem_info['n_dirichlet_dofs']}, "
          f"neumann={res.bem_info['n_neumann_dofs']}")
    print(f"  GMRES            : info={res.bem_info['gmres_info']}, "
          f"iters={res.bem_info['gmres_iterations']}  "
          f"(assemble {res.bem_info['wall_time_assemble_s']:.2f}s, "
          f"solve {res.bem_info['wall_time_solve_s']:.2f}s)")
    print()
    print(f"  I_principal (V=1) = {_fmt_vec(res.principal_moments)}")
    print(f"  semi-axes  (V=1) = {_fmt_vec(res.semi_axes)}")
    print(f"  r_eq       (V=1) = {r_eq_unit:.6f}")
    print(f"  C_strasberg      = {res.c_strasberg:.9f}")
    print(f"  C_bem            = {res.c_bem:.9f}")
    print()
    print("  Frequencies (Hz, at V = 1 m^3, air-in-water):")
    print(f"    Minnaert  = {res.f_minnaert:.6f}")
    print(f"    Strasberg = {res.f_strasberg:.6f}   "
          f"(rel. diff vs Minnaert  = {(res.f_strasberg - res.f_minnaert) / res.f_minnaert * 100:+.4f}%)")
    print(f"    BEM       = {res.f_bem:.6f}   "
          f"(rel. diff vs Minnaert  = {(res.f_bem - res.f_minnaert) / res.f_minnaert * 100:+.4f}%)")
    print(f"                                 "
          f"(rel. diff vs Strasberg = {(res.f_bem - res.f_strasberg) / res.f_strasberg * 100:+.4f}%)")
    print()


def main() -> None:
    meshes = [
        _ROOT / "python" / "bem" / "test_meshes" / "sphere_res20_reverse_N.obj",
        _ROOT / "python" / "bem" / "test_meshes" / "ellipsoid_123_unnormalized.obj",
    ]
    results = []
    for m in meshes:
        if not m.exists():
            raise FileNotFoundError(m)
        print(f"[compare] processing {m.name} ...")
        res = _compare_one(m, trial_pair="P1-DP0")
        results.append(res)
        _print_report(res)

    # Compact side-by-side summary.
    print("=" * 72)
    print("Summary (all frequencies in Hz, at V = 1 m^3)")
    print("=" * 72)
    header = f"{'mesh':<40s} {'f_Minn':>10s} {'f_Stra':>10s} {'f_BEM':>10s}"
    print(header)
    print("-" * len(header))
    for res in results:
        name = Path(res.mesh_name).name
        print(
            f"{name:<40s} {res.f_minnaert:>10.4f} "
            f"{res.f_strasberg:>10.4f} {res.f_bem:>10.4f}"
        )


if __name__ == "__main__":
    main()

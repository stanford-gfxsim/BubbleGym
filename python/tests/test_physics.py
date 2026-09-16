"""Unit tests for the capacitance-frequency relations and the sphere baselines.

The shape-feature and BEM cases both run on
``python/bem/test_meshes/sphere_res20_reverse_N.obj`` (4002 vertices, 8000
triangles), where every quantity has a closed form: a sphere has Wadell
sphericity 1, is its own convex hull, has Willmore energy 4 pi, and at unit
volume has capacitance ``R_V = (3 / (4 pi))^(1/3)``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from python.freq_model.analytical.physics import (  # noqa: E402
    capacitance_to_frequency,
    frequency_to_capacitance,
)

SPHERE_MESH = _ROOT / "python" / "bem" / "test_meshes" / "sphere_res20_reverse_N.obj"
R_UNIT_VOLUME = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)  # 0.6203504908994001


def _has(module: str) -> bool:
    """Is a top-level module importable? Never executes the module body."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


_HAS_IGL = _has("igl") and _has("scipy")
# Top-level names only: probing "bempp_cl.api" would import the parent package
# before limit_threads() has had a chance to cap its thread pools.
_HAS_BEMPP = _has("bempp") or _has("bempp_cl")


class TestPhysics(unittest.TestCase):
    def test_round_trip(self) -> None:
        V = 1.0
        C0 = 0.62
        f = capacitance_to_frequency(C0, V)
        C1 = frequency_to_capacitance(f, V)
        self.assertAlmostEqual(C0, C1, places=10)


@unittest.skipUnless(_HAS_IGL, "shape features need igl + scipy")
class TestSphereShapeFeatures(unittest.TestCase):
    """A discretized sphere must land on every analytical shape value."""

    @classmethod
    def setUpClass(cls) -> None:
        from python.shape_feature.mesh_utils import (
            ensure_outward_orientation,
            largest_connected_component_mesh,
            load_obj_mesh,
            vertices_scaled_to_target_volume,
        )
        from python.shape_feature.nonspherical_features import (
            convex_hull_features_from_vf,
            curvature_integrals_from_vf,
            sphericities_from_integrals,
        )

        v, f = load_obj_mesh(SPHERE_MESH)
        v, f, _n_cc, _kept = largest_connected_component_mesh(v, f)
        f = ensure_outward_orientation(v, f)
        v_unit = vertices_scaled_to_target_volume(v, f, 1.0)

        cls.curv = curvature_integrals_from_vf(v_unit, f, target_volume=None)
        cls.hull = convex_hull_features_from_vf(v_unit, f, target_volume=None)
        cls.phi = sphericities_from_integrals(
            volume=float(cls.curv["volume"]),
            area=float(cls.curv["area"]),
            mean_curvature=float(cls.curv["M"]),
            willmore_vertex=float(cls.curv["W_vertex"]),
        )

    def test_wadell_sphericity_is_one(self) -> None:
        # An inscribed polyhedron is always slightly under-spherical, so this
        # approaches 1 from below; 8000 triangles get within ~1.6e-4.
        self.assertLessEqual(self.phi["Phi_VA"], 1.0)
        self.assertAlmostEqual(self.phi["Phi_VA"], 1.0, delta=1e-3)
        self.assertAlmostEqual(self.phi["Phi_VM"], 1.0, delta=1e-3)
        self.assertAlmostEqual(self.phi["Phi_W"], 1.0, delta=1e-3)

    def test_convex_hull_ratios_are_one(self) -> None:
        # A convex body is its own convex hull, so all three eta ratios are
        # exactly 1 up to Qhull's floating-point reconstruction.
        for key in ("eta_V", "eta_A", "eta_M"):
            self.assertAlmostEqual(float(self.hull[key]), 1.0, delta=1e-6, msg=key)

    def test_willmore_energy_is_four_pi(self) -> None:
        # The cotangent estimate undershoots on a coarse mesh: 12.5568 vs
        # 4 pi = 12.5664, i.e. 7.6e-4 relative.
        w = float(self.curv["W_vertex"])
        self.assertAlmostEqual(w / (4.0 * math.pi), 1.0, delta=2e-3)


@unittest.skipUnless(_HAS_BEMPP, "BEM solve needs bempp-cl")
class TestSphereBem(unittest.TestCase):
    def test_unit_volume_capacitance(self) -> None:
        """A unit-volume sphere has capacitance ``R_V = (3/(4 pi))^(1/3)``."""
        from python.bem._threading import default_thread_count, limit_threads

        limit_threads(default_thread_count())
        from python.bem.compute_freq_bempp_galerkin import (
            solve_minnaert_frequency_galerkin,
        )

        # Quadrature 4/4 rather than the solver's 6/6 default: on a sphere it
        # is already accurate to ~1e-6 in C and keeps the test to a few seconds.
        out = solve_minnaert_frequency_galerkin(
            str(SPHERE_MESH),
            quadrature_regular=4,
            quadrature_singular=4,
        )
        self.assertAlmostEqual(float(out["v0"]), 1.0, delta=1e-9)
        # b = (K - 1/2 I) 1 must be the constant -1 on a sound closed surface.
        self.assertLess(float(out["rhs_orientation_residual"]), 1e-4)
        self.assertAlmostEqual(
            float(out["capacitance"]) / R_UNIT_VOLUME, 1.0, delta=1e-4
        )


if __name__ == "__main__":
    unittest.main()

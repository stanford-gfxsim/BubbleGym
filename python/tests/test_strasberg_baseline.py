"""
Unit tests for the Strasberg residual baseline wiring.

Two checks:

1. Sphere sanity: for a unit-volume sphere, Strasberg reduces exactly to Minnaert.
2. Preprocessing idempotency: running
   python/freq_model/NN/add_baseline_frequencies_to_dataset.py on a small synthetic CSV
   twice is a no-op on the second pass and values are bit-for-bit stable.

Run from repo root:
    python -m unittest python.tests.test_strasberg_baseline
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_PYTHON_ROOT = _ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from freq_model.analytical.minnaert_freq import minnaert_frequency_unit_volume  # noqa: E402
from freq_model.analytical.strasberg_freq import (  # noqa: E402
    strasberg_frequency,
    strasberg_frequency_unit_volume,
)
from freq_model.NN.add_baseline_frequencies_to_dataset import (  # noqa: E402
    F_MINNAERT_COL,
    F_STRASBERG_COL,
    add_baseline_columns_in_place,
)


def _sphere_normalized_inertia() -> np.ndarray:
    """
    Volume-normalized inertia principal moments for a uniform sphere with V = 1.

    The dataset's inertia moments are computed volumetrically (not with unit density
    on the surface); for a uniform sphere of volume V and radius R = (3V/4π)^(1/3),
    the volumetric moment about a principal axis is

        I = (2/5) * V * R^2.

    At V = 1 this gives I = (2/5) * R^2 on all three principal axes.
    """
    R = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    I_value = (2.0 / 5.0) * 1.0 * R * R
    return np.array([I_value, I_value, I_value], dtype=np.float64)


class TestSphereStrasbergEqualsMinnaert(unittest.TestCase):
    def test_sphere_strasberg_equals_minnaert(self) -> None:
        I_sphere = _sphere_normalized_inertia()
        f_m = minnaert_frequency_unit_volume()
        f_s = strasberg_frequency_unit_volume(I_sphere)
        self.assertTrue(np.isfinite(f_m) and f_m > 0.0)
        self.assertTrue(np.isfinite(f_s) and f_s > 0.0)
        rel_err = abs(f_s - f_m) / f_m
        self.assertLess(
            rel_err,
            1e-6,
            msg=f"Sphere mismatch: f_minnaert={f_m:.6f}, f_strasberg={f_s:.6f}, rel_err={rel_err:.3e}",
        )


class TestPreprocessingIdempotent(unittest.TestCase):
    def _build_synthetic_csv(self, path: Path) -> None:
        """
        Synthetic 3-row dataset: a sphere and two mildly non-spherical ellipsoids.
        All volumes = 1 (unit-volume convention used across the repo).
        """
        R = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        I_sphere = (2.0 / 5.0) * R * R

        rows = [
            # Sphere at V=1: isotropic inertia, f_BEM ~ Minnaert.
            {
                "volume": 1.0,
                "i00": I_sphere,
                "i11": I_sphere,
                "i22": I_sphere,
                "frequency": minnaert_frequency_unit_volume(),
                "mesh_filename": "mesh_sphere.obj",
            },
            # Slightly prolate: bump i11, i22 modestly.
            {
                "volume": 1.0,
                "i00": I_sphere * 0.90,
                "i11": I_sphere * 1.05,
                "i22": I_sphere * 1.05,
                "frequency": minnaert_frequency_unit_volume() * 0.97,
                "mesh_filename": "mesh_prolate.obj",
            },
            # Slightly oblate.
            {
                "volume": 1.0,
                "i00": I_sphere * 1.10,
                "i11": I_sphere * 0.95,
                "i22": I_sphere * 0.95,
                "frequency": minnaert_frequency_unit_volume() * 1.02,
                "mesh_filename": "mesh_oblate.obj",
            },
        ]
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_preprocessing_adds_columns_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "synth.csv"
            self._build_synthetic_csv(csv_path)

            stats1 = add_baseline_columns_in_place(csv_path)
            df1 = pd.read_csv(csv_path)
            self.assertIn(F_MINNAERT_COL, df1.columns)
            self.assertIn(F_STRASBERG_COL, df1.columns)
            self.assertTrue(np.all(np.isfinite(df1[F_MINNAERT_COL].to_numpy())))
            self.assertTrue(np.all(np.isfinite(df1[F_STRASBERG_COL].to_numpy())))

            v1_min = df1[F_MINNAERT_COL].to_numpy(dtype=np.float64)
            v1_str = df1[F_STRASBERG_COL].to_numpy(dtype=np.float64)

            # Row 0 is the sphere: Strasberg must equal Minnaert within 1e-6 relative.
            rel_err = abs(v1_str[0] - v1_min[0]) / v1_min[0]
            self.assertLess(rel_err, 1e-6)

            # Second pass should be a no-op.
            stats2 = add_baseline_columns_in_place(csv_path)
            self.assertEqual(stats2["strasberg_seconds"], 0.0)

            df2 = pd.read_csv(csv_path)
            v2_min = df2[F_MINNAERT_COL].to_numpy(dtype=np.float64)
            v2_str = df2[F_STRASBERG_COL].to_numpy(dtype=np.float64)
            np.testing.assert_array_equal(v1_min, v2_min)
            np.testing.assert_array_equal(v1_str, v2_str)

            # Sigma stats should be reported on both passes, strictly finite.
            for stats in (stats1, stats2):
                self.assertTrue(np.isfinite(stats["sigma_minnaert"]))
                self.assertTrue(np.isfinite(stats["sigma_strasberg"]))


def _ellipsoid_normalized_inertia(axes: np.ndarray) -> np.ndarray:
    """Volume-normalized principal moments of a uniform ellipsoid."""
    a = np.asarray(axes, dtype=np.float64)
    return np.array(
        [
            (a[1] ** 2 + a[2] ** 2) / 5.0,
            (a[0] ** 2 + a[2] ** 2) / 5.0,
            (a[0] ** 2 + a[1] ** 2) / 5.0,
        ],
        dtype=np.float64,
    )


def test_sphere_recovers_axes_capacitance_and_unit_shape_factor() -> None:
    """R = 1 mm sphere: semi-axes = R, C = R, f = f_Minnaert exactly."""
    R = 1e-3
    V = (4.0 / 3.0) * np.pi * R ** 3
    res = strasberg_frequency((2.0 / 5.0) * R ** 2 * np.eye(3), V)
    np.testing.assert_allclose(res.semi_axes, [R, R, R], rtol=1e-12)
    assert abs(res.capacitance - R) < 1e-12 * R
    assert abs(res.shape_factor - 1.0) < 1e-10


def test_oblate_spheroid_matches_strasberg_1953_two_percent() -> None:
    """Strasberg's own worked example: e = 2 oblate spheroid resonates ~2% high."""
    R = 1e-3
    e = 2.0
    a_minor = R / e ** (2.0 / 3.0)
    a_major = e * a_minor
    axes = np.sort([a_minor, a_major, a_major])
    V = (4.0 / 3.0) * np.pi * float(np.prod(axes))
    res = strasberg_frequency(_ellipsoid_normalized_inertia(axes), V)
    np.testing.assert_allclose(res.semi_axes, axes, rtol=1e-9)
    assert abs(res.shape_factor - 1.020758) < 1e-6


def test_triaxial_ellipsoid_axes_roundtrip() -> None:
    """1:2:3 mm ellipsoid: the inertia -> semi-axes inversion is exact."""
    axes = np.array([1.0e-3, 2.0e-3, 3.0e-3])
    V = (4.0 / 3.0) * np.pi * float(np.prod(axes))
    res = strasberg_frequency(_ellipsoid_normalized_inertia(axes), V)
    np.testing.assert_allclose(res.semi_axes, axes, rtol=1e-9)
    assert abs(res.shape_factor - 1.0401617944178925) < 1e-9


def test_full_tensor_is_rotation_invariant() -> None:
    """A rotated 3x3 tensor must give the same frequency as its eigenvalues."""
    axes = np.array([1.0e-3, 2.0e-3, 3.0e-3])
    V = (4.0 / 3.0) * np.pi * float(np.prod(axes))
    I_principal = _ellipsoid_normalized_inertia(axes)
    theta, phi = 0.73, 1.21
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    Ry = np.array(
        [
            [np.cos(phi), 0.0, np.sin(phi)],
            [0.0, 1.0, 0.0],
            [-np.sin(phi), 0.0, np.cos(phi)],
        ]
    )
    Rot = Rz @ Ry
    f_principal = strasberg_frequency(I_principal, V).frequency_hz
    f_rotated = strasberg_frequency(Rot @ np.diag(I_principal) @ Rot.T, V).frequency_hz
    assert abs(f_rotated - f_principal) / f_principal < 1e-12


def test_triangle_inequality_violation_raises() -> None:
    """I_3 > I_1 + I_2 describes no real ellipsoid and must not silently pass."""
    I_bad = np.array([1e-6, 1e-6, 5e-6])
    try:
        strasberg_frequency(I_bad, 1e-9)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-ellipsoidal inertia")


def test_volume_inertia_mismatch_raises() -> None:
    """The volume-consistency guard is what catches an un-normalized inertia."""
    axes = np.array([1.0e-3, 2.0e-3, 3.0e-3])
    V = (4.0 / 3.0) * np.pi * float(np.prod(axes))
    try:
        strasberg_frequency(_ellipsoid_normalized_inertia(axes), V * 2.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for volume / inertia mismatch")


if __name__ == "__main__":
    unittest.main()

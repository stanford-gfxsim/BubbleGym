"""
Strasberg (1953) resonant frequency of an ellipsoidal gas bubble
from the volume-normalized inertia tensor of an arbitrary shape.

Strasberg, M. (1953). "The pulsation frequency of nonspherical gas bubbles
in liquids." J. Acoust. Soc. Am. 25, 536-537.

    mesh -> (V, I_normalized) -> semi-axes (a1, a2, a3) -> capacitance C
    f = (1 / 2 pi) sqrt( 4 pi gamma p0 C / (rho V) )

Input convention: `inertia` is the *volume-normalized* volumetric inertia
tensor, I_ij = (1 / V) integral_V (r^2 delta_ij - x_i x_j) dV, units length^2.
Passing the un-normalized Mirtich output is the single most common mistake
here; `strasberg_frequency` raises on the resulting volume mismatch.

The triaxial-ellipsoid capacitance uses the Landau-Lifshitz elliptic-integral
form via Carlson R_F (`scipy.special.elliprf`): numerically equivalent to the
Kraniotis 2013 Appell-F1 closed form but O(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import elliprf

# Physical constants (air-in-water defaults; all overridable per call) live in
# physics.py so this module, minnaert_freq.py, and the dataset scripts cannot
# drift apart. Re-exported here for backwards compatibility with callers that
# import them from this module.
from .physics import GAMMA_AIR, P0_ATM, RHO_WATER


# --------------------------------------------------------------------------- #
# Geometry: volume-normalized inertia  -->  ellipsoid semi-axes
# --------------------------------------------------------------------------- #
def semi_axes_from_normalized_inertia(I: np.ndarray) -> np.ndarray:
    """
    Recover ellipsoid semi-axes (a1, a2, a3) from the *volume-normalized*
    inertia tensor of a shape, assuming the shape is an ellipsoid.

    For a uniform-density solid ellipsoid, the volume-normalized principal
    moments are
        I_k = (1/5) (a_i^2 + a_j^2),   {i, j, k} = {1, 2, 3}
    which inverts directly (no dependence on mass or volume) to
        a_k^2 = (5/2) (I_i + I_j - I_k).

    Parameters
    ----------
    I : (3, 3) or (3,) array
        Volume-normalized volumetric inertia tensor (full 3x3 symmetric)
        or its three principal moments. Units: length^2.

    Returns
    -------
    a : (3,) ndarray, sorted ascending (a[0] <= a[1] <= a[2])
        Ellipsoid semi-axes in the same length units as sqrt(I).

    Raises
    ------
    ValueError
        If the tensor is not consistent with any real ellipsoid
        (any recovered a_k^2 <= 0).
    """
    I = np.asarray(I, dtype=float)

    if I.shape == (3, 3):
        principal = np.sort(np.linalg.eigvalsh(I))
    elif I.shape == (3,):
        principal = np.sort(I)
    else:
        raise ValueError(f"Inertia must be shape (3,3) or (3,); got {I.shape}")

    if np.any(principal <= 0):
        raise ValueError(
            f"Principal moments must all be positive; got {principal}"
        )

    I1, I2, I3 = principal   # ascending: I1 <= I2 <= I3

    # a_k^2 = (5/2) (I_i + I_j - I_k)
    a_sq = 2.5 * np.array([
        -I1 + I2 + I3,
        +I1 - I2 + I3,
        +I1 + I2 - I3,
    ])

    if np.any(a_sq <= 0):
        raise ValueError(
            f"Inertia tensor is not consistent with any real ellipsoid. "
            f"Principal moments {principal} give a_k^2 = {a_sq}. "
            f"The shape is likely non-ellipsoidal (e.g. strongly concave, "
            f"toroidal, or highly elongated beyond the triangle inequality "
            f"on moments)."
        )

    return np.sqrt(np.sort(a_sq))   # ascending


# --------------------------------------------------------------------------- #
# Electrostatic capacitance of a conducting triaxial ellipsoid
# --------------------------------------------------------------------------- #
def ellipsoid_capacitance(a: np.ndarray) -> float:
    """
    Geometric (CGS) capacitance of a conducting triaxial ellipsoid.

    The capacitance has units of length and satisfies
        1/C = (1/2) integral_0^inf  ds / sqrt((s+a1^2)(s+a2^2)(s+a3^2))
            = R_F(a1^2, a2^2, a3^2)                        (Carlson)
        C   = 1 / R_F(a1^2, a2^2, a3^2)

    Verified limits:
        sphere   a1=a2=a3=R           ->  C = R
        prolate  a1=a2=a < c=a3       ->  C = sqrt(c^2 - a^2) / arccosh(c/a)
        oblate   a1 < a2=a3=a         ->  C = sqrt(a^2 - a1^2) / arcsin(e_std)
        disk     a1 -> 0, a2=a3=a     ->  C = 2 a / pi
    """
    a = np.asarray(a, dtype=float)
    if a.shape != (3,) or np.any(a <= 0):
        raise ValueError(f"Semi-axes must be three positive floats; got {a}")
    return 1.0 / float(elliprf(a[0] ** 2, a[1] ** 2, a[2] ** 2))


# --------------------------------------------------------------------------- #
# Main API
# --------------------------------------------------------------------------- #
@dataclass
class StrasbergResult:
    """Return bundle from `strasberg_frequency`."""
    frequency_hz: float               # f, in Hz
    omega: float                      # 2 pi f, in rad/s
    semi_axes: np.ndarray             # (a1, a2, a3), ascending, in metres
    volume: float                     # V, in m^3 (echoed input)
    capacitance: float                # C, in metres (CGS geometric)
    equivalent_sphere_radius: float   # R_eq = (3V/4pi)^(1/3)
    minnaert_frequency_hz: float      # Minnaert frequency of same-volume sphere
    shape_factor: float               # f / f_Minnaert (>= 1, equality iff sphere)


def strasberg_frequency(
    inertia: np.ndarray,
    volume: float,
    *,
    gamma: float = GAMMA_AIR,
    p0: float = P0_ATM,
    rho_liquid: float = RHO_WATER,
) -> StrasbergResult:
    """
    Strasberg (1953) monopole pulsation frequency of a gas bubble
    whose shape is characterised by its volume-normalized inertia tensor.

        f = (1 / 2 pi) sqrt( 4 pi gamma p0 C / (rho_liquid V) )

    `inertia` is the *volume-normalized* tensor (3x3) or its principal moments
    (3,), in length^2; `volume` is in m^3 and is passed separately because it
    has been absorbed into that normalization. Raises ValueError if the tensor
    is not consistent with any real ellipsoid, or if `volume` disagrees with
    the volume implied by the inertia-derived semi-axes (which is what an
    un-normalized inertia tensor looks like from in here).
    """
    a = semi_axes_from_normalized_inertia(inertia)

    # Consistency check: the volume implied by the inertia-derived semi-axes
    # should match the passed-in volume. This catches two common mistakes:
    #   (1) inertia was not actually volume-normalized (scale mismatch), or
    #   (2) the shape really isn't ellipsoidal (principal moments suggest
    #       semi-axes whose product gives a wrong volume).
    V_from_axes = (4.0 / 3.0) * np.pi * float(np.prod(a))
    rel_err = abs(V_from_axes - volume) / volume
    if rel_err > 1e-3:
        raise ValueError(
            f"Inconsistent inputs: volume from inertia-derived semi-axes is "
            f"{V_from_axes:.6e}, but passed volume is {volume:.6e} "
            f"(relative error {rel_err:.2%}). Either the shape is not "
            f"ellipsoidal, or `inertia` is not actually volume-normalized."
        )

    C = ellipsoid_capacitance(a)

    omega = np.sqrt((4.0 * np.pi * gamma * p0 * C) / (rho_liquid * volume))
    f = omega / (2.0 * np.pi)

    R_eq = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    f_minnaert = (1.0 / (2.0 * np.pi * R_eq)) * np.sqrt(
        3.0 * gamma * p0 / rho_liquid
    )

    return StrasbergResult(
        frequency_hz=float(f),
        omega=float(omega),
        semi_axes=a,
        volume=float(volume),
        capacitance=float(C),
        equivalent_sphere_radius=float(R_eq),
        minnaert_frequency_hz=float(f_minnaert),
        shape_factor=float(f / f_minnaert),
    )


# --------------------------------------------------------------------------- #
# Unit-volume convenience wrappers (shared by preprocessing / training / eval)
# --------------------------------------------------------------------------- #

def strasberg_frequency_unit_volume(
    inertia: np.ndarray,
    *,
    gamma: float = GAMMA_AIR,
    p0: float = P0_ATM,
    rho_liquid: float = RHO_WATER,
) -> float:
    """
    Strasberg frequency at unit volume (V = 1) from the *volume-normalized*
    inertia tensor of a shape.

    Real meshes are not exact ellipsoids, so the inertia-implied semi-axes
    generally do not satisfy 4/3 pi prod(a) == V exactly. To sidestep the
    strict volume-consistency check inside `strasberg_frequency`, we:

      1. compute a = semi_axes_from_normalized_inertia(I)
      2. let V_axes = 4/3 pi prod(a)                 (the volume implied by a)
      3. compute f at V_axes via strasberg_frequency(I, V_axes)
      4. rescale: Strasberg f scales as V^(-1/3), so
             f(V=1) = f(V_axes) * V_axes^(1/3)

    This is the single canonical implementation used by the preprocessing
    script, the trainers, the eval script and the interactive viewer.
    """
    a = semi_axes_from_normalized_inertia(inertia)
    V_axes = (4.0 / 3.0) * np.pi * float(np.prod(a))
    res = strasberg_frequency(
        inertia, V_axes, gamma=gamma, p0=p0, rho_liquid=rho_liquid
    )
    return float(res.frequency_hz) * (V_axes ** (1.0 / 3.0))


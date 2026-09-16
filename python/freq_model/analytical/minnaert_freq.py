import numpy as np

# Physical constants (air-in-water defaults; all overridable per call) live in
# physics.py so this module, strasberg_freq.py, and the dataset scripts cannot
# drift apart. Re-exported here for backwards compatibility with callers that
# import them from this module.
from .physics import GAMMA_AIR, P0_ATM, RHO_WATER

MINNAERT_CONSTANT = 3.283243423687599


def minnaert_frequency_unit_volume(
    *,
    gamma: float = GAMMA_AIR,
    p0: float = P0_ATM,
    rho_liquid: float = RHO_WATER,
) -> float:
    """
    Minnaert frequency of a unit-volume sphere (V = 1 m^3).

    f_Minnaert(V=1) = (1 / (2 pi R)) sqrt(3 gamma p0 / rho)
    with R = (3 / (4 pi))^(1/3).

    Provided as the single source of truth for the Minnaert baseline on
    unit-volume datasets; avoids the `3.26 / R` magic constant scattered
    elsewhere and keeps constants in lockstep with strasberg_frequency().
    """
    R_eq = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    return (1.0 / (2.0 * np.pi * R_eq)) * float(
        np.sqrt(3.0 * gamma * p0 / rho_liquid)
    )


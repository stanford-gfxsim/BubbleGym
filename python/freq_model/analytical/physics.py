"""Capacitance-frequency relations.

Frequency relation (Minnaert-type, from capacitance C and bubble volume V):

    f = (1 / 2pi) sqrt(4 pi gamma P0 C / (rho_l V))

The physical constants below are the single source of truth shared by every
trainer, evaluator, and dataset script.
"""

from __future__ import annotations

import math

GAMMA_AIR = 1.4
P0_ATM = 101_325.0  # Pa
RHO_WATER = 1000.0  # kg / m^3


def capacitance_to_frequency(
    C: float,
    V: float,
    *,
    gamma: float = GAMMA_AIR,
    p0: float = P0_ATM,
    rho: float = RHO_WATER,
) -> float:
    """Convert capacitance C (length scale) to resonance frequency f (Hz). V: volume of the bubble (m^3)."""
    if C <= 0.0 or V <= 0.0:
        raise ValueError("C and V must be positive")
    return math.sqrt(4.0 * math.pi * gamma * p0 * C / (rho * V)) / (2.0 * math.pi)


def frequency_to_capacitance(
    f: float,
    V: float,
    *,
    gamma: float = GAMMA_AIR,
    p0: float = P0_ATM,
    rho: float = RHO_WATER,
) -> float:
    """Invert capacitance_to_frequency: C = (2 pi f)^2 rho V / (4 pi gamma P0)."""
    if f <= 0.0 or V <= 0.0:
        raise ValueError("f and V must be positive")
    two_pi_f = 2.0 * math.pi * f
    return (two_pi_f * two_pi_f) * rho * V / (4.0 * math.pi * gamma * p0)


__all__ = [
    "GAMMA_AIR",                    # heat capacity ratio (gamma) of air
    "P0_ATM",                       # standard atmospheric pressure
    "RHO_WATER",                    # density of water
    "capacitance_to_frequency",     # capacitance -> frequency
    "frequency_to_capacitance",     # frequency -> capacitance
]

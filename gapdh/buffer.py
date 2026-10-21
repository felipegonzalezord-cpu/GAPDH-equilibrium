"""Buffer capacity and the pH drift caused by the H⁺ released by GAPDH (spec §3.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from . import constants as C


@dataclass(frozen=True)
class Buffer:
    """A monoprotic buffer component acting near one pKa."""

    name: str
    conc_mM: float
    pKa: float


def protonation_alpha(ph: float, pKa: float) -> float:
    """Fraction in the base form, α = 1 / (1 + 10^(pKa − pH))."""
    return 1.0 / (1.0 + 10 ** (pKa - ph))


def buffer_capacity(conc_mM: float, pKa: float, ph: float) -> float:
    """β = 2.303 · C · α(1 − α) in mM per pH unit."""
    a = protonation_alpha(ph, pKa)
    return C.LN10 * conc_mM * a * (1 - a)


def total_capacity(buffers: Sequence[Buffer], ph: float) -> float:
    """Sum of the capacities of all buffers at the given pH (mM/pH)."""
    return sum(buffer_capacity(b.conc_mM, b.pKa, ph) for b in buffers)


def ph_after(xi: float, ph0: float, beta: float) -> float:
    """pH after extent ξ, assuming one H⁺ per turnover and constant β: pH₀ − ξ/β (clipped to 0–14)."""
    if beta <= 0:
        return C.PH_DRIFT_CLIP[0] if xi > 0 else (C.PH_DRIFT_CLIP[1] if xi < 0 else ph0)
    ph = ph0 - C.PROTONS_PER_TURNOVER * xi / beta
    return min(max(ph, C.PH_DRIFT_CLIP[0]), C.PH_DRIFT_CLIP[1])


def drifting_k(k_at_ph: Callable[[float], float], ph0: float, beta: float) -> Callable[[float], float]:
    """Return ξ → K′(pH(ξ)) for use by the self-consistent equilibrium solver."""

    def k_of_xi(xi: float) -> float:
        return k_at_ph(ph_after(xi, ph0, beta))

    return k_of_xi

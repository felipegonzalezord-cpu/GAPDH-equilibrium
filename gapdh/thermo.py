"""Thermodynamics of D-GAP + NAD⁺ + Pi ⇌ 1,3-BPG + NADH + H⁺.

Conventions (spec §1):
* concentrations in mM; K′ and Γ in M⁻¹ (1 M standard state, [H⁺] folded in);
* extent ξ in mM, positive in the forward (GAP-oxidation) direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
from scipy.optimize import brentq

from . import constants as C

KFunc = Callable[[float], float]
KLike = Union[float, KFunc]


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Mixture:
    """Concentrations of the five reacting species, all in mM."""

    gap: float  # D-GAP (the reactive isomer)
    nad: float
    pi: float
    bpg: float = 0.0
    nadh: float = 0.0

    def shifted(self, xi: float) -> "Mixture":
        """Return the mixture after an extent ξ (mM) of the forward reaction."""
        return Mixture(
            gap=self.gap - xi,
            nad=self.nad - xi,
            pi=self.pi - xi,
            bpg=self.bpg + xi,
            nadh=self.nadh + xi,
        )

    def as_dict(self) -> dict:
        return {"GAP": self.gap, "NAD⁺": self.nad, "Pi": self.pi, "1,3-BPG": self.bpg, "NADH": self.nadh}

    @property
    def forward_limit(self) -> float:
        """Largest physically possible forward extent (mM)."""
        return min(self.gap, self.nad, self.pi)

    @property
    def reverse_limit(self) -> float:
        """Most negative physically possible extent (mM)."""
        return -min(self.bpg, self.nadh)


@dataclass(frozen=True)
class Conditions:
    """Parameters of the K′(pH, T) model (spec §3.1)."""

    ph: float = C.DEFAULT_PH
    temp_c: float = C.DEFAULT_TEMP_C
    k_ref: float = C.K_PRIME_REF
    ph_ref: float = C.PH_REF
    slope: float = C.PROTON_SLOPE
    delta_h: float = C.DELTA_H_KJ

    @property
    def temp_k(self) -> float:
        return self.temp_c + C.KELVIN_OFFSET

    def k_prime(self, ph: float | None = None) -> float:
        """K′ (M⁻¹) at this temperature and the given pH (defaults to self.ph)."""
        return k_prime(self.ph if ph is None else ph, self.temp_k, self.k_ref, self.ph_ref, self.slope, self.delta_h)


# ----------------------------------------------------------------------------
# K′ and Gibbs energies
# ----------------------------------------------------------------------------
def k_prime(
    ph: float,
    temp_k: float = C.T_REF_K,
    k_ref: float = C.K_PRIME_REF,
    ph_ref: float = C.PH_REF,
    slope: float = C.PROTON_SLOPE,
    delta_h: float = C.DELTA_H_KJ,
    t_ref: float = C.T_REF_K,
) -> float:
    """Apparent equilibrium constant K′ in M⁻¹ (spec §3.1).

    K′ = K′ref · 10^(s·(pH − pHref)) · exp(−(ΔH′°/R)·(1/T − 1/Tref)).
    """
    return float(k_ref * 10 ** (slope * (ph - ph_ref)) * math.exp(-(delta_h / C.R_KJ) * (1 / temp_k - 1 / t_ref)))


def delta_g0(k: float, temp_k: float = C.T_REF_K) -> float:
    """Standard transformed Gibbs energy ΔG′° = −RT ln K′ (kJ/mol, 1 M standard state)."""
    return -C.R_KJ * temp_k * math.log(k)


def mass_action_ratio(m: Mixture) -> float:
    """Γ = [BPG][NADH] / ([GAP][NAD⁺][Pi]) in M⁻¹ (inputs in mM).

    Returns 0 when no product is present and +inf when a substrate is absent.
    """
    num = max(m.bpg, 0.0) * max(m.nadh, 0.0)
    den = max(m.gap, 0.0) * max(m.nad, 0.0) * max(m.pi, 0.0)
    if num == 0.0:
        return 0.0 if den > 0 else math.nan
    if den == 0.0:
        return math.inf
    return 1000.0 * num / den


def delta_g(m: Mixture, k: float, temp_k: float = C.T_REF_K) -> float:
    """ΔG′ = RT ln(Γ/K′) of a mixture (kJ/mol); −inf with no products, +inf with no substrates."""
    gamma = mass_action_ratio(m)
    if math.isnan(gamma):
        return math.nan
    if gamma == 0.0:
        return -math.inf
    if math.isinf(gamma):
        return math.inf
    return C.R_KJ * temp_k * math.log(gamma / k)


def flux_ratio_from_dg(dg: float, temp_k: float = C.T_REF_K) -> float:
    """Forward/backward flux ratio J₊/J₋ = exp(−ΔG′/RT)."""
    return math.exp(-dg / (C.R_KJ * temp_k))


# ----------------------------------------------------------------------------
# Equilibrium solver (spec §3.2)
# ----------------------------------------------------------------------------
def _as_kfunc(k: KLike) -> KFunc:
    if callable(k):
        return k
    kk = float(k)
    return lambda _xi: kk


def solve_extent(m: Mixture, k: KLike) -> float:
    """Equilibrium extent ξ (mM) of the forward reaction.

    Solves (B₀+ξ)(N₀+ξ) = k(ξ)·(G₀−ξ)(D₀−ξ)(P₀−ξ) with k = K′/1000 (mM⁻¹).
    `k` may be a constant K′ (M⁻¹) or a callable ξ → K′ (used for buffer drift).
    On the physical interval the left side rises and the right side falls, so
    the root is unique.
    """
    kf = _as_kfunc(k)
    hi = m.forward_limit
    lo = m.reverse_limit
    if hi <= 0.0 and lo >= 0.0:
        return 0.0  # a substrate and a product are both missing: nothing can happen
    if hi - lo <= 1e-14:
        return 0.0  # degenerate interval

    scale = max(m.gap, m.nad, m.pi, m.bpg, m.nadh, 1e-12)
    eps = 1e-15 * max(scale, 1.0)
    a = lo + eps if lo < 0 else lo
    b = hi - eps if hi > 0 else hi
    # Keep the bracket strictly inside the physical range but never inverted.
    if a >= b:
        return 0.5 * (lo + hi)

    def f(xi: float) -> float:
        return (m.bpg + xi) * (m.nadh + xi) - kf(xi) / 1000.0 * (m.gap - xi) * (m.nad - xi) * (m.pi - xi)

    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        # Floating-point edge (extremely large/small K′): the root sits at a bound.
        return a if abs(fa) < abs(fb) else b
    return float(brentq(f, a, b, xtol=1e-16 * scale + 1e-300, rtol=4 * np.finfo(float).eps, maxiter=500))


def quadratic_shortcut(m: Mixture, k: float) -> float:
    """Teaching approximation: Pi held constant, giving a quadratic in ξ (spec §3.2).

    a = 1 − κ, b = B₀ + N₀ + κ(G₀ + D₀), c = B₀N₀ − κG₀D₀, κ = k·P₀.
    """
    kappa = k / 1000.0 * m.pi
    a = 1.0 - kappa
    b = m.bpg + m.nadh + kappa * (m.gap + m.nad)
    c = m.bpg * m.nadh - kappa * m.gap * m.nad
    if abs(a) < 1e-12:
        return -c / b if b != 0 else math.nan
    disc = b * b - 4 * a * c
    if disc < 0:
        return math.nan
    return (-b + math.sqrt(disc)) / (2 * a)


def square_root_rule(m: Mixture, k: float) -> float:
    """Teaching approximation for small conversion with no products: ξ ≈ √(κ·G₀·D₀)."""
    kappa = k / 1000.0 * m.pi
    return math.sqrt(max(kappa * m.gap * m.nad, 0.0))


def kappa(m: Mixture, k: float) -> float:
    """κ = k·P₀ (dimensionless), with k = K′/1000 in mM⁻¹."""
    return k / 1000.0 * m.pi


# ----------------------------------------------------------------------------
# Energy profile (spec §3.3)
# ----------------------------------------------------------------------------
def profile_point(m0: Mixture, x: float, k: float, temp_k: float) -> dict:
    """Γ, Γ/K′, ΔG′, J₊/J₋ and net fraction (1 − Γ/K′) after extent x from m0."""
    m = m0.shifted(x)
    gamma = mass_action_ratio(m)
    ratio = gamma / k if not math.isnan(gamma) else math.nan
    dg = delta_g(m, k, temp_k)
    if math.isnan(gamma):
        flux = math.nan
    elif gamma == 0.0:
        flux = math.inf
    elif math.isinf(gamma):
        flux = 0.0
    else:
        flux = k / gamma
    return {"x": x, "gamma": gamma, "gamma_over_k": ratio, "dg": dg, "flux_ratio": flux, "net_fraction": 1 - ratio}


def energy_profile(m0: Mixture, k: float, temp_k: float, xi_eq: float, n: int = 300) -> list[dict]:
    """Points along the reaction coordinate from ~0 to about 1.3·ξ_eq (or the physical limit)."""
    if xi_eq == 0:
        return []
    sign = 1.0 if xi_eq > 0 else -1.0
    limit = m0.forward_limit if sign > 0 else -m0.reverse_limit
    x_max = min(1.3 * abs(xi_eq), 0.999 * limit) if limit > 0 else 1.3 * abs(xi_eq)
    x_min = min(1e-6, abs(xi_eq) / 1000)
    if x_max <= x_min:
        return []
    xs = np.unique(np.concatenate([np.geomspace(x_min, x_max, n // 2), np.linspace(x_min, x_max, n)]))
    pts = []
    for x in xs:
        p = profile_point(m0, sign * float(x), k, temp_k)
        if math.isfinite(p["dg"]):
            pts.append(p)
    return pts


def snapshots(m0: Mixture, k: float, temp_k: float, xi_eq: float, spike_mM: float) -> dict:
    """The four teaching snapshots a–d (spec §3.3).

    a: 1 μM NADH formed; b: half of ξ_eq; c: equilibrium;
    d: the equilibrium mixture plus an NADH spike, and how far it relaxes back.
    """
    sign = 1.0 if xi_eq >= 0 else -1.0
    out = {
        "a": profile_point(m0, sign * C.SNAPSHOT_A_MM, k, temp_k),
        "b": profile_point(m0, xi_eq / 2, k, temp_k),
        "c": profile_point(m0, xi_eq, k, temp_k),
    }
    m_eq = m0.shifted(xi_eq)
    m_spiked = Mixture(m_eq.gap, m_eq.nad, m_eq.pi, m_eq.bpg, m_eq.nadh + spike_mM)
    gamma_d = mass_action_ratio(m_spiked)
    dg_d = delta_g(m_spiked, k, temp_k)
    relax = solve_extent(m_spiked, k)
    out["d"] = {
        "x": xi_eq + spike_mM,  # NADH present relative to the start (reaction + spike)
        "gamma": gamma_d,
        "gamma_over_k": gamma_d / k,
        "dg": dg_d,
        "flux_ratio": k / gamma_d if gamma_d > 0 else math.inf,
        "net_fraction": 1 - gamma_d / k,
        "relaxation_xi": relax,
    }
    return out


# ----------------------------------------------------------------------------
# Inverse: K′ from a measured plateau (spec §3.6)
# ----------------------------------------------------------------------------
def k_from_extent(m0: Mixture, n: float) -> float:
    """K′ (M⁻¹) implied by an observed extent N (mM of NADH formed)."""
    den = (m0.gap - n) * (m0.nad - n) * (m0.pi - n)
    if den <= 0:
        return math.inf
    return 1000.0 * (m0.bpg + n) * (m0.nadh + n) / den


def k_from_plateau(m0: Mixture, delta_a: float, epsilon: float, path_cm: float) -> float:
    """K′ (M⁻¹) implied by a measured plateau ΔA₃₄₀; N = ΔA/(ε·l)."""
    return k_from_extent(m0, delta_a / (epsilon * path_cm))


def k_from_plateau_with_error(
    m0: Mixture, delta_a: float, sd_delta_a: float, epsilon: float, path_cm: float
) -> dict:
    """K′ with an uncertainty propagated by central finite difference, and the local exponent d ln K′ / d ln N."""
    n = delta_a / (epsilon * path_cm)
    k = k_from_extent(m0, n)
    out = {"n": n, "k": k, "sd_k": math.nan, "dg0": math.nan, "exponent": math.nan, "k_low": math.nan, "k_high": math.nan}
    if not math.isfinite(k) or k <= 0:
        return out
    out["dg0"] = delta_g0(k)
    if sd_delta_a > 0:
        k_hi = k_from_plateau(m0, delta_a + sd_delta_a, epsilon, path_cm)
        k_lo = k_from_plateau(m0, max(delta_a - sd_delta_a, 0.0), epsilon, path_cm)
        out["k_low"], out["k_high"] = k_lo, k_hi
        if math.isfinite(k_hi):
            out["sd_k"] = (k_hi - k_lo) / 2
    h = max(n * 1e-4, 1e-9)
    k_plus = k_from_extent(m0, n + h)
    k_minus = k_from_extent(m0, max(n - h, 1e-15))
    if k_minus > 0 and math.isfinite(k_plus):
        out["exponent"] = (math.log(k_plus) - math.log(k_minus)) / (math.log(n + h) - math.log(max(n - h, 1e-15)))
    return out

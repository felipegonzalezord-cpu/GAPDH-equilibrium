"""Time-course simulator (spec §3.7).

dξ/dt = V · G/(Km + G) · S_NAD · (1 − Γ/K′)

* S_NAD = [D/(Km_NAD(1 + N/Ki) + D)] / [D₀/(Km_NAD + D₀)] with competitive NADH
  inhibition switched on, and 1 otherwise.
* Arsenate mode drops (1 − Γ/K′): arsenolysis is irreversible because
  1-arseno-3-phosphoglycerate hydrolyses, so the reaction runs to completion.
The enzyme amount (V) changes only the rate, never the plateau.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from . import constants as C
from .thermo import KLike, Mixture, _as_kfunc


@dataclass(frozen=True)
class KineticParams:
    v_max: float = C.V_MAX_MM_MIN  # mM/min
    km_gap: float = C.KM_GAP_MM  # mM
    km_nad: float = C.KM_NAD_MM  # mM
    ki_nadh: float = C.KI_NADH_MM  # mM
    inhibition: bool = False


def rate(xi: float, m0: Mixture, k: KLike, p: KineticParams, arsenate: bool) -> float:
    """dξ/dt (mM/min) at extent ξ."""
    kf = _as_kfunc(k)
    g = m0.gap - xi
    d = m0.nad - xi
    pi = m0.pi if arsenate else m0.pi - xi  # arsenate replaces Pi; Pi is not consumed
    n = m0.nadh + xi
    if g <= 0 or d <= 0 or (not arsenate and pi <= 0):
        return 0.0
    v = p.v_max * g / (p.km_gap + g)
    if p.inhibition:
        if m0.nad <= 0:
            return 0.0
        s_now = d / (p.km_nad * (1 + max(n, 0.0) / p.ki_nadh) + d)
        s_ref = m0.nad / (p.km_nad + m0.nad)
        v *= s_now / s_ref
    if arsenate:
        return v
    kp = kf(xi)
    num = 1000.0 * (m0.bpg + xi) * n
    den = g * d * pi
    return v * (1 - num / (kp * den))


def simulate(
    m0: Mixture,
    k: KLike,
    p: KineticParams,
    arsenate: bool = False,
    t_end_min: float = C.T_END_MIN,
    epsilon: float = C.EPSILON_340,
    path_cm: float = C.PATH_CM,
    n_points: int = 401,
) -> dict:
    """Integrate ξ(t) from 0 to t_end and return arrays t, ξ, ΔA₃₄₀ and Γ/K′.

    ΔA₃₄₀ is the change from t = 0 (NADH formed × ε × l).
    """
    kf = _as_kfunc(k)
    t_eval = np.linspace(0.0, t_end_min, n_points)

    def rhs(_t, y):
        return [rate(float(y[0]), m0, kf, p, arsenate)]

    sol = solve_ivp(rhs, (0.0, t_end_min), [0.0], t_eval=t_eval, method="LSODA", rtol=1e-9, atol=1e-12)
    xi = sol.y[0] if sol.success else np.full_like(t_eval, np.nan)
    gk = np.empty_like(xi)
    for i, x in enumerate(xi):
        if arsenate:
            gk[i] = 0.0
            continue
        g, d, pi = m0.gap - x, m0.nad - x, m0.pi - x
        den = g * d * pi
        num = 1000.0 * (m0.bpg + x) * (m0.nadh + x)
        gk[i] = (num / den) / kf(x) if den > 0 else math.inf
    return {"t": t_eval, "xi": xi, "dA": epsilon * path_cm * xi, "gamma_over_k": gk, "ok": bool(sol.success)}


def initial_rate_window_end(t: np.ndarray, gamma_over_k: np.ndarray,
                            threshold: float = C.INITIAL_RATE_GAMMA_FRACTION) -> float:
    """First time (min) at which Γ/K′ reaches the threshold; the end of the initial-rate window."""
    idx = np.nonzero(np.asarray(gamma_over_k) >= threshold)[0]
    if idx.size == 0:
        return float(t[-1])
    i = int(idx[0])
    if i == 0:
        return 0.0
    # linear interpolation between the bracketing samples
    t0, t1 = t[i - 1], t[i]
    g0, g1 = gamma_over_k[i - 1], gamma_over_k[i]
    if not np.isfinite(g1) or g1 == g0:
        return float(t1)
    return float(t0 + (threshold - g0) * (t1 - t0) / (g1 - g0))

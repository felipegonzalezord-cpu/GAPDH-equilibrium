"""Ionic strength and an APPROXIMATE activity correction of K′ (spec §3.5).

Extended Debye–Hückel at 25 °C: ln γ = −A z² √I / (1 + B √I), A = 1.1758, B = 1.6 (Alberty).
Greinert et al. (2021) showed that this strongly underestimates the non-ideality of
GAPDH reaction mixtures, so every number from this module is an estimate.
"""

from __future__ import annotations

import math

from . import constants as C
from .thermo import Mixture


def pi_dibasic_fraction(ph: float, pKa2: float = C.PKA2_PI_SPECIATION) -> float:
    """Fraction of inorganic phosphate present as HPO₄²⁻ (the rest is H₂PO₄⁻)."""
    return 1.0 / (1.0 + 10 ** (pKa2 - ph))


def gamma(z: float, ionic_strength_M: float, A: float = C.DH_A, B: float = C.DH_B) -> float:
    """Extended Debye–Hückel activity coefficient of an ion of charge z at ionic strength I (M)."""
    s = math.sqrt(max(ionic_strength_M, 0.0))
    return math.exp(-A * z * z * s / (1 + B * s))


def gamma_pi(ionic_strength_M: float, dibasic_fraction: float) -> float:
    """Speciation-weighted γ of total Pi: ln γ_Pi = (1−f)·ln γ(z=1) + f·ln γ(z=2)."""
    f = min(max(dibasic_fraction, 0.0), 1.0)
    return math.exp((1 - f) * math.log(gamma(1, ionic_strength_M)) + f * math.log(gamma(2, ionic_strength_M)))


def k_gamma(ionic_strength_M: float, dibasic_fraction: float = 1.0) -> float:
    """Kγ = γ_BPG · γ_NADH / (γ_GAP · γ_NAD⁺ · γ_Pi)."""
    z = C.CHARGE
    g_bpg = gamma(z["BPG"], ionic_strength_M)
    g_nadh = gamma(z["NADH"], ionic_strength_M)
    g_gap = gamma(z["GAP"], ionic_strength_M)
    g_nad = gamma(z["NAD+"], ionic_strength_M)
    return g_bpg * g_nadh / (g_gap * g_nad * gamma_pi(ionic_strength_M, dibasic_fraction))


def activity_factor(ionic_strength_M: float, i_ref_M: float, dibasic_fraction: float) -> float:
    """Multiplier on K′: K′(I) = K′(I_ref) · Kγ(I_ref) / Kγ(I)."""
    return k_gamma(i_ref_M, dibasic_fraction) / k_gamma(ionic_strength_M, dibasic_fraction)


def ionic_strength_breakdown(
    m: Mixture,
    gap_total_mM: float,
    edta_mM: float,
    kcl_mM: float,
    ph: float,
    pKa2: float = C.PKA2_PI_SPECIATION,
) -> list[dict]:
    """Contributions ½·c·z² (mM) to the ionic strength, one row per ion.

    `gap_total_mM` is all GAP in solution (both isomers if DL), which counts toward I.
    Every anion except Cl⁻ is balanced by Na⁺; Cl⁻ by K⁺.
    """
    z = C.CHARGE
    f = pi_dibasic_fraction(ph, pKa2)
    anions = [
        ("NAD⁺ (net −1)", m.nad, z["NAD+"]),
        ("NADH", m.nadh, z["NADH"]),
        ("GAP (all isomers)", gap_total_mM, z["GAP"]),
        ("1,3-BPG", m.bpg, z["BPG"]),
        ("H₂PO₄⁻", m.pi * (1 - f), z["H2PO4-"]),
        ("HPO₄²⁻", m.pi * f, z["HPO4 2-"]),
        ("EDTA (≈ −3.5)", edta_mM, z["EDTA"]),
    ]
    na = sum(c * abs(q) for _, c, q in anions)
    rows = [{"ion": name, "conc_mM": c, "charge": q, "half_cz2_mM": 0.5 * c * q * q} for name, c, q in anions]
    rows.append({"ion": "Na⁺ (counter-ion)", "conc_mM": na, "charge": 1.0, "half_cz2_mM": 0.5 * na})
    rows.append({"ion": "K⁺ (from KCl)", "conc_mM": kcl_mM, "charge": 1.0, "half_cz2_mM": 0.5 * kcl_mM})
    rows.append({"ion": "Cl⁻ (from KCl)", "conc_mM": kcl_mM, "charge": -1.0, "half_cz2_mM": 0.5 * kcl_mM})
    return rows


def ionic_strength(m: Mixture, gap_total_mM: float, edta_mM: float, kcl_mM: float, ph: float,
                   pKa2: float = C.PKA2_PI_SPECIATION) -> float:
    """Ionic strength I = ½ Σ cᵢ zᵢ² in M."""
    rows = ionic_strength_breakdown(m, gap_total_mM, edta_mM, kcl_mM, ph, pKa2)
    return sum(r["half_cz2_mM"] for r in rows) / 1000.0

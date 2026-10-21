"""Every validation number of the build spec, section 4 (plus the expected tables of section 5).

Most checks use a 1 % relative tolerance. Numbers that the spec gives to only one or two
significant figures (e.g. γ(z=4) = 0.005) are checked by rounding instead, because 1 % of a
rounded number is tighter than the rounding itself.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from gapdh import constants as C
from gapdh.activity import gamma, ionic_strength, k_gamma, pi_dibasic_fraction
from gapdh.buffer import buffer_capacity
from gapdh.kinetics import KineticParams, simulate
from gapdh.model import (
    AssaySettings,
    experiment_plan,
    kcl_anchored_prediction,
    perturbation,
    predict,
    sensitivity,
)
from gapdh.thermo import (
    delta_g0,
    flux_ratio_from_dg,
    k_from_plateau,
    k_prime,
    profile_point,
    quadratic_shortcut,
    snapshots,
    solve_extent,
    square_root_rule,
)

T25 = C.T_REF_K
BASE = AssaySettings()  # default preset: "Current assay (Felipe, pH 8.5)"
MIX = BASE.mixture()
K85 = k_prime(8.5)


def close(actual: float, expected: float, rel: float = 0.01) -> None:
    assert actual == pytest.approx(expected, rel=rel), f"{actual} vs {expected}"


def rounds_to(actual: float, expected: float, decimals: int) -> None:
    assert abs(actual - expected) <= 0.5 * 10 ** (-decimals) + 1e-12, f"{actual} does not round to {expected}"


def xi_at(**changes) -> float:
    return predict(replace(BASE, **changes)).xi


# --- Base case -----------------------------------------------------------------------
def test_k_prime_base():
    close(K85, 2.4666, rel=1e-4)


def test_base_case():
    p = predict(BASE)
    close(p.xi, 0.0947)
    close(p.delta_a, 0.589)
    close(p.pct_gap, 18.9)
    close(p.pct_nad, 9.5)
    close(p.gamma_eq, 2.47)
    assert p.gamma_matches_k


def test_quadratic_shortcut():
    close(quadratic_shortcut(MIX, K85), 0.0951)


def test_square_root_rule():
    close(square_root_rule(MIX, K85), 0.111)


# --- pH dependence -------------------------------------------------------------------
@pytest.mark.parametrize("ph,dg", [(6.5, 9.18), (7.0, 6.32), (7.5, 3.47), (8.0, 0.62), (8.5, -2.24),
                                   (9.0, -5.09), (9.5, -7.95)])
def test_dg0_vs_ph(ph, dg):
    close(delta_g0(k_prime(ph), T25), dg)


@pytest.mark.parametrize("ph,xi", [(6.5, 0.0109), (7.0, 0.0192), (7.5, 0.0333), (8.0, 0.0569), (9.0, 0.1509),
                                   (9.5, 0.2261)])
def test_xi_vs_ph(ph, xi):
    close(xi_at(ph=ph), xi)


# --- Single changes --------------------------------------------------------------------
def test_k_half_and_double():
    p_half = predict(replace(BASE, k_ref=C.K_PRIME_REF / 2))
    p_double = predict(replace(BASE, k_ref=C.K_PRIME_REF * 2))
    close(p_half.xi, 0.0700)
    close(p_double.xi, 0.1261)
    close(p_half.delta_a, 0.435)
    close(p_double.delta_a, 0.785)


def test_dl_gap():
    p = predict(replace(BASE, dl_racemate=True))
    assert p.mix0.gap == pytest.approx(0.25)
    close(p.xi, 0.0651)
    close(p.pct_gap, 26.0)


def test_pi_50_100():
    close(xi_at(pi=50.0), 0.1797)
    close(xi_at(pi=100.0), 0.2276)


def test_nad_10():
    close(xi_at(nad=10.0), 0.2448)


def test_nadh_present_at_start():
    close(xi_at(nadh=0.15), 0.0517)


def test_37_celsius():
    close(k_prime(8.5, 310.15), 2.65)
    close(xi_at(temp_c=37.0), 0.0976)


def test_path_half_cm():
    close(predict(replace(BASE, path_cm=0.5)).delta_a, 0.294)


# --- Reverse direction ---------------------------------------------------------------
def test_reverse_preset():
    s = AssaySettings.from_preset("Reverse, PGK-coupled style")
    p = predict(s)
    close(p.k_start, 0.311)
    close(p.xi, -0.0997)
    close(p.pct_bpg_reduced, 99.7)
    assert p.dg_start == math.inf


def test_reverse_mixture_ph85():
    s = replace(AssaySettings.from_preset("Reverse, PGK-coupled style"), ph=8.5)
    close(predict(s).xi, -0.0977)


# --- Energy-profile snapshots -----------------------------------------------------------
def test_snapshot_b_at_005_mM():
    pt = profile_point(MIX, 0.05, K85, T25)
    close(pt["gamma"], 0.588)
    close(pt["gamma_over_k"], 0.238)
    close(pt["dg"], -3.56)
    close(pt["flux_ratio"], 4.2)


def test_snapshot_d_spike():
    xi = solve_extent(MIX, K85)
    d = snapshots(MIX, K85, T25, xi, 0.05)["d"]
    close(d["gamma"], 3.77)
    close(d["dg"], 1.05)
    close(d["relaxation_xi"], -0.0186)


def test_snapshot_c_is_equilibrium():
    xi = solve_extent(MIX, K85)
    c = snapshots(MIX, K85, T25, xi, 0.05)["c"]
    assert c["dg"] == pytest.approx(0.0, abs=1e-9)
    assert c["flux_ratio"] == pytest.approx(1.0, rel=1e-9)


# --- Buffer drift --------------------------------------------------------------------
def test_buffer_drift_phosphate_only():
    p = predict(replace(BASE, drift_on=True))
    close(p.beta, 1.05)
    close(p.xi, 0.0872)
    close(p.delta_a, 0.542)
    close(p.ph_final, 8.417, rel=1e-4)


def test_buffer_drift_with_taps():
    # Spec: "β ≈ 28.4 mM/pH". That is the TAPS contribution; the total adds phosphate's 1.05.
    close(buffer_capacity(50.0, 8.4, 8.5), 28.4)
    p = predict(replace(BASE, drift_on=True, buffer2_on=True))
    close(p.beta, 28.4 + 1.05)
    rounds_to(BASE.ph - p.ph_final, 0.003, 3)


# --- Debye–Hückel ------------------------------------------------------------------------
def test_debye_huckel_I_004():
    close(gamma(1, 0.04), 0.837)
    close(gamma(2, 0.04), 0.490)
    close(gamma(4, 0.04), 0.058)
    close(k_gamma(0.04, dibasic_fraction=1.0), 0.141)  # spec value: Pi taken as fully HPO₄²⁻


def test_debye_huckel_I_025():
    close(gamma(1, 0.25), 0.721)
    close(gamma(2, 0.25), 0.271)
    rounds_to(gamma(4, 0.25), 0.005, 3)
    rounds_to(k_gamma(0.25, dibasic_fraction=1.0), 0.028, 3)


def test_k_gamma_speciation_weighted():
    # With Pi speciation at pH 8.5 (pKa₂ 6.8, 98 % HPO₄²⁻), Kγ is ~1.4 % lower than the z = 2 limit.
    f = pi_dibasic_fraction(8.5)
    close(k_gamma(0.04, f), 0.139)
    rounds_to(k_gamma(0.25, f), 0.027, 3)


def test_default_ionic_strength():
    close(ionic_strength(MIX, BASE.gap, BASE.edta, BASE.kcl, BASE.ph), 0.040)


def test_debye_huckel_kcl_jump():
    s100 = kcl_anchored_prediction(BASE, 100.0)
    close(s100.ionic_strength_M, 0.14)
    rounds_to(s100.activity_factor, 2.9, 1)
    close(s100.xi, 0.146)
    rounds_to(s100.delta_a, 0.91, 2)


# --- Inverse ---------------------------------------------------------------------------
@pytest.mark.parametrize("da,k", [(0.40, 1.02), (0.62, 2.78), (0.80, 5.18)])
def test_inverse(da, k):
    close(k_from_plateau(MIX, da, 6.22, 1.0), k)


# --- Time course -----------------------------------------------------------------------
def _a_at(sim, t):
    import numpy as np

    return float(np.interp(t, sim["t"], sim["dA"]))


def test_time_course_phosphate_ph85():
    sim = simulate(MIX, K85, KineticParams(), arsenate=False)
    close(_a_at(sim, 1), 0.245)
    close(_a_at(sim, 2), 0.423)
    close(_a_at(sim, 10), 0.589)


def test_time_course_arsenate():
    sim = simulate(MIX, K85, KineticParams(), arsenate=True)
    close(_a_at(sim, 1), 0.257)
    close(_a_at(sim, 5), 1.239)
    close(_a_at(sim, 10), 2.285)


def test_time_course_phosphate_ph7():
    sim = simulate(MIX, k_prime(7.0), KineticParams(), arsenate=False)
    close(_a_at(sim, 1), 0.116)
    close(_a_at(sim, 10), 0.119)


def test_enzyme_amount_does_not_move_equilibrium():
    slow = simulate(MIX, K85, KineticParams(v_max=0.05), t_end_min=200)
    fast = simulate(MIX, K85, KineticParams(v_max=0.10), t_end_min=200)
    close(slow["dA"][-1], fast["dA"][-1], rel=1e-4)
    close(fast["dA"][-1], 0.589)


# --- Section-5 tables ------------------------------------------------------------------
def test_flux_ratio_table():
    for dg, expected in [(0, 1), (-1, 1.5), (-5.7, 10), (-11.4, 100), (-17.1, 1000)]:
        close(flux_ratio_from_dg(dg, T25), expected, rel=0.02 if expected == 1.5 else 0.01)
    rounds_to(flux_ratio_from_dg(-2.5, T25), 2.7, 1)


def test_pi_speciation_table():
    for ph, pct in [(7.0, 61), (7.5, 83), (8.0, 94), (8.5, 98)]:
        rounds_to(100 * pi_dibasic_fraction(ph), pct, 0)


def test_two_fold_table():
    rows = {r["change"]: r for r in sensitivity(BASE)}
    rounds_to(rows["pH"]["up"], 1.59, 2)
    rounds_to(rows["pH"]["down"], 0.60, 2)
    rounds_to(rows["GAP"]["up"], 1.43, 2)
    rounds_to(rows["GAP"]["down"], 0.69, 2)
    rounds_to(rows["NAD⁺"]["up"], 1.37, 2)
    rounds_to(rows["NAD⁺"]["down"], 0.71, 2)
    rounds_to(rows["Pi"]["up"], 1.34, 2)
    rounds_to(rows["Pi"]["down"], 0.74, 2)
    rounds_to(rows["K′"]["up"], 1.33, 2)


def test_perturbation_table():
    assert perturbation(BASE, "2× enzyme")["net"] == 0.0
    rounds_to(perturbation(BASE, "+ Pi", 40.0)["relaxation"], 0.53, 2)
    nadh = perturbation(BASE, "+ NADH", 0.05)
    rounds_to(nadh["jump"], 0.31, 2)
    rounds_to(nadh["relaxation"], -0.12, 2)
    rounds_to(perturbation(BASE, "+ GAP", 0.5)["relaxation"], 0.25, 2)
    close(perturbation(BASE, "+ arsenate")["a_after"], 3.11)


def test_experiment_planner_sets():
    rows = experiment_plan(BASE, KineticParams())
    by = {(r["set"][0:2], r["condition"]): r["delta_A340"] for r in rows}
    for g, a in [(0.1, 0.24), (0.25, 0.40), (0.5, 0.59), (1.0, 0.84)]:
        rounds_to(by[("A:", f"GAP {g:g} mM")], a, 2)
    for p, a in [(2, 0.28), (10, 0.59), (50, 1.12)]:
        rounds_to(by[("B1", f"Pi {p:g} mM")], a, 2)
    for ph, a in [(7.5, 0.21), (8.0, 0.35), (8.5, 0.59), (9.0, 0.94)]:
        rounds_to(by[("D:", f"pH {ph:g}")], a, 2)


# --- Robustness --------------------------------------------------------------------------
@pytest.mark.parametrize("changes", [
    dict(gap=0.0, nad=0.0, bpg=0.0, nadh=0.0),
    dict(gap=0.0),
    dict(pi=0.0),
    dict(nad=0.0, nadh=0.5),
    dict(gap=1e-9, nad=1e-9),
    dict(gap=100.0, nad=100.0, pi=1000.0),
    dict(ph=4.0),
    dict(ph=11.0, temp_c=80.0),
    dict(ph=11.0, drift_on=True, pi=0.01),
    dict(ph=4.0, drift_on=True, ionic_on=True, kcl=1000.0),
    dict(arsenate=True, pi=0.0),
])
def test_extreme_inputs_do_not_crash(changes):
    p = predict(replace(BASE, **changes))
    assert math.isfinite(p.xi)
    m = p.mix_eq
    for v in (m.gap, m.nad, m.pi, m.bpg, m.nadh):
        assert v >= -1e-9

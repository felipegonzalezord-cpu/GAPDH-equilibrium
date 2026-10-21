"""One prediction pipeline that combines thermo, buffer drift and the activity correction.

Every tab of the app calls `predict` (directly or through the helpers below), so all
numbers shown follow the same model and the same switches.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Callable

import numpy as np

from . import constants as C
from .activity import activity_factor, ionic_strength, pi_dibasic_fraction
from .buffer import Buffer, drifting_k, ph_after, total_capacity
from .kinetics import KineticParams, initial_rate_window_end, simulate
from .presets import values_for
from .thermo import (
    Conditions,
    Mixture,
    delta_g,
    delta_g0,
    kappa,
    mass_action_ratio,
    quadratic_shortcut,
    solve_extent,
    square_root_rule,
)


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class AssaySettings:
    """Every user input of the sidebar. Concentrations in mM, T in °C, I in M."""

    ph: float = C.DEFAULT_PH
    temp_c: float = C.DEFAULT_TEMP_C
    gap: float = C.DEFAULT_GAP_MM  # as entered (DL total when dl_racemate is on)
    dl_racemate: bool = False
    nad: float = C.DEFAULT_NAD_MM
    pi: float = C.DEFAULT_PI_MM
    bpg: float = C.DEFAULT_BPG_MM
    nadh: float = C.DEFAULT_NADH_MM
    edta: float = C.DEFAULT_EDTA_MM
    kcl: float = C.DEFAULT_KCL_MM
    epsilon: float = C.EPSILON_340
    path_cm: float = C.PATH_CM
    k_ref: float = C.K_PRIME_REF
    ph_ref: float = C.PH_REF
    slope: float = C.PROTON_SLOPE
    delta_h: float = C.DELTA_H_KJ
    i_ref: float = C.I_REF_M
    drift_on: bool = False
    ionic_on: bool = False
    pi_pka_buffer: float = C.PKA2_PI_BUFFER
    buffer2_on: bool = False
    buffer2_name: str = C.EXAMPLE_SECOND_BUFFER["name"]
    buffer2_conc: float = C.EXAMPLE_SECOND_BUFFER["conc_mM"]
    buffer2_pka: float = C.EXAMPLE_SECOND_BUFFER["pKa"]
    arsenate: bool = False

    # -- derived -----------------------------------------------------------------
    @property
    def gap_d(self) -> float:
        """Effective D-GAP (mM): half the entered value for a DL-racemate."""
        return self.gap * C.DL_D_FRACTION if self.dl_racemate else self.gap

    @property
    def temp_k(self) -> float:
        return self.temp_c + C.KELVIN_OFFSET

    def mixture(self) -> Mixture:
        return Mixture(gap=self.gap_d, nad=self.nad, pi=self.pi, bpg=self.bpg, nadh=self.nadh)

    def conditions(self) -> Conditions:
        return Conditions(self.ph, self.temp_c, self.k_ref, self.ph_ref, self.slope, self.delta_h)

    def buffers(self) -> list[Buffer]:
        """Phosphate (the Pi entered, which is also a reactant) plus an optional second buffer."""
        out = [Buffer("Phosphate", self.pi, self.pi_pka_buffer)]
        if self.buffer2_on and self.buffer2_conc > 0:
            out.append(Buffer(self.buffer2_name or "Buffer 2", self.buffer2_conc, self.buffer2_pka))
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AssaySettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_preset(cls, name: str) -> "AssaySettings":
        return cls.from_dict(values_for(name))


def validate(s: AssaySettings) -> tuple[AssaySettings, list[str]]:
    """Clamp impossible inputs and return plain-language warnings (spec §6)."""
    warnings: list[str] = []
    changes: dict = {}
    for name, label in [("gap", "GAP"), ("nad", "NAD⁺"), ("pi", "Pi"), ("bpg", "1,3-BPG"), ("nadh", "NADH"),
                        ("edta", "EDTA"), ("kcl", "KCl"), ("buffer2_conc", "second buffer")]:
        if getattr(s, name) < 0:
            changes[name] = 0.0
            warnings.append(f"{label} cannot be negative; using 0 mM.")
    lo, hi = C.PH_RANGE
    if not lo <= s.ph <= hi:
        changes["ph"] = min(max(s.ph, lo), hi)
        warnings.append(f"pH must be between {lo:g} and {hi:g}; using {changes['ph']:g}.")
    lo, hi = C.TEMP_RANGE_C
    if not lo <= s.temp_c <= hi:
        changes["temp_c"] = min(max(s.temp_c, lo), hi)
        warnings.append(f"Temperature must be between {lo:g} and {hi:g} °C; using {changes['temp_c']:g} °C.")
    if s.epsilon <= 0:
        changes["epsilon"] = C.EPSILON_340
        warnings.append("ε₃₄₀ must be positive; using 6.22 mM⁻¹ cm⁻¹.")
    if s.path_cm <= 0:
        changes["path_cm"] = C.PATH_CM
        warnings.append("Path length must be positive; using 1 cm.")
    if s.k_ref <= 0:
        changes["k_ref"] = C.K_PRIME_REF
        warnings.append("K′ref must be positive; using 0.078 M⁻¹.")
    if s.i_ref < 0:
        changes["i_ref"] = 0.0
        warnings.append("Reference ionic strength cannot be negative; using 0 M.")
    s2 = replace(s, **changes) if changes else s

    m = s2.mixture()
    has_fwd = m.gap > 0 and m.nad > 0 and (m.pi > 0 or s2.arsenate)
    has_rev = m.bpg > 0 and m.nadh > 0 and not s2.arsenate
    if not has_fwd and not has_rev:
        warnings.append("Nothing can react: the forward direction needs GAP, NAD⁺ and Pi (or arsenate), "
                        "and the reverse direction needs 1,3-BPG and NADH.")
    elif m.gap > 0 and m.nad > 0 and m.pi == 0 and not s2.arsenate:
        warnings.append("Pi is 0 mM, so GAP cannot be oxidised (phosphate is a substrate).")
    if s2.drift_on and total_capacity(s2.buffers(), s2.ph) < 1e-6:
        warnings.append("Buffer capacity is essentially zero, so the drift correction is extreme. "
                        "Add phosphate or a second buffer.")
    return s2, warnings


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Prediction:
    xi: float  # equilibrium extent (mM), forward positive
    mix0: Mixture
    mix_eq: Mixture
    k_ideal: float  # K′ (M⁻¹) at the starting pH with no correction
    k_start: float  # K′ actually used at the starting pH (activity factor applied if on)
    k_final: float  # K′ at the final pH (differs from k_start only with buffer drift)
    activity_factor: float  # factor applied to K′ (1 when the correction is off)
    activity_factor_estimate: float  # what the factor would be (shown even when off)
    ionic_strength_M: float
    beta: float  # total buffer capacity (mM/pH)
    ph_final: float
    pct_gap: float  # % of D-GAP oxidised
    pct_nad: float  # % of NAD⁺ reduced
    pct_bpg_reduced: float  # % of BPG reduced (reverse direction)
    nadh_eq: float  # total NADH at equilibrium (mM)
    delta_a: float  # ΔA₃₄₀ from t = 0
    gamma_eq: float  # Γ at equilibrium (M⁻¹)
    kappa: float
    dg0: float  # ΔG′° at the starting pH (kJ/mol)
    dg_start: float  # ΔG′ of the starting mixture (kJ/mol)
    xi_quadratic: float
    xi_sqrt: float
    arsenate: bool

    @property
    def direction(self) -> str:
        if self.xi > 0:
            return "forward"
        if self.xi < 0:
            return "reverse"
        return "none"

    @property
    def gamma_matches_k(self) -> bool:
        if self.arsenate or not math.isfinite(self.gamma_eq) or self.k_final <= 0:
            return False
        return abs(self.gamma_eq / self.k_final - 1) < 1e-6

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("mix0", "mix_eq")}
        d["initial_mM"] = self.mix0.as_dict()
        d["equilibrium_mM"] = self.mix_eq.as_dict()
        d["direction"] = self.direction
        return d


def k_at_ph_function(s: AssaySettings) -> Callable[[float], float]:
    """pH → K′ (M⁻¹) including the activity factor when that correction is on."""
    cond = s.conditions()
    factor = _activity(s)[1] if s.ionic_on else 1.0
    return lambda ph: cond.k_prime(ph) * factor


def k_function(s: AssaySettings) -> Callable[[float], float]:
    """ξ → K′ (M⁻¹) for the current switches (buffer drift and/or ionic strength)."""
    k_at_ph = k_at_ph_function(s)
    if s.drift_on:
        return drifting_k(k_at_ph, s.ph, total_capacity(s.buffers(), s.ph))
    k0 = k_at_ph(s.ph)
    return lambda _xi: k0


def _activity(s: AssaySettings) -> tuple[float, float]:
    """(ionic strength in M, activity factor Kγ(I_ref)/Kγ(I)) for the starting mixture."""
    m = s.mixture()
    i = ionic_strength(m, s.gap, s.edta, s.kcl, s.ph)
    f = pi_dibasic_fraction(s.ph)
    return i, activity_factor(i, s.i_ref, f)


def predict(s: AssaySettings) -> Prediction:
    """Equilibrium (or, with arsenate, completion) for one set of inputs."""
    m0 = s.mixture()
    cond = s.conditions()
    k_ideal = cond.k_prime()
    i_strength, factor_est = _activity(s)
    factor = factor_est if s.ionic_on else 1.0
    beta = total_capacity(s.buffers(), s.ph)
    kfun = k_function(s)
    k_start = k_ideal * factor

    if s.arsenate:
        xi = min(m0.gap, m0.nad) if (m0.gap > 0 and m0.nad > 0) else 0.0
        mix_eq = Mixture(m0.gap - xi, m0.nad - xi, m0.pi, m0.bpg, m0.nadh + xi)
    else:
        xi = solve_extent(m0, kfun)
        mix_eq = m0.shifted(xi)

    ph_final = ph_after(xi, s.ph, beta) if s.drift_on else s.ph
    k_final = kfun(xi)
    gamma_eq = math.nan if s.arsenate else mass_action_ratio(mix_eq)
    pct_gap = 100 * xi / m0.gap if m0.gap > 0 else math.nan
    pct_nad = 100 * xi / m0.nad if m0.nad > 0 else math.nan
    pct_bpg = -100 * xi / m0.bpg if (m0.bpg > 0 and xi < 0) else math.nan
    return Prediction(
        xi=xi,
        mix0=m0,
        mix_eq=mix_eq,
        k_ideal=k_ideal,
        k_start=k_start,
        k_final=k_final,
        activity_factor=factor,
        activity_factor_estimate=factor_est,
        ionic_strength_M=i_strength,
        beta=beta,
        ph_final=ph_final,
        pct_gap=pct_gap,
        pct_nad=pct_nad,
        pct_bpg_reduced=pct_bpg,
        nadh_eq=m0.nadh + xi,
        delta_a=s.epsilon * s.path_cm * xi,
        gamma_eq=gamma_eq,
        kappa=kappa(m0, k_start),
        dg0=delta_g0(k_start, s.temp_k),
        dg_start=delta_g(m0, k_start, s.temp_k),
        xi_quadratic=quadratic_shortcut(m0, k_start),
        xi_sqrt=square_root_rule(m0, k_start),
        arsenate=s.arsenate,
    )


# ----------------------------------------------------------------------------
# Derived tables used by several tabs
# ----------------------------------------------------------------------------
def sensitivity(s: AssaySettings) -> list[dict]:
    """'What a two-fold change buys you': ξ(changed)/ξ(current) for single changes."""
    base = predict(s).xi

    def ratio(s2: AssaySettings) -> float:
        return predict(s2).xi / base if base != 0 else math.nan

    rows = [
        ("pH", "+0.5 / −0.5", replace(s, ph=min(s.ph + 0.5, C.PH_RANGE[1])), replace(s, ph=max(s.ph - 0.5, C.PH_RANGE[0]))),
        ("GAP", "×2 / ÷2", replace(s, gap=s.gap * 2), replace(s, gap=s.gap / 2)),
        ("NAD⁺", "×2 / ÷2", replace(s, nad=s.nad * 2), replace(s, nad=s.nad / 2)),
        ("Pi", "×2 / ÷2", replace(s, pi=s.pi * 2), replace(s, pi=s.pi / 2)),
        ("K′", "×2 / ÷2", replace(s, k_ref=s.k_ref * 2), replace(s, k_ref=s.k_ref / 2)),
    ]
    return [{"change": name, "step": step, "up": ratio(up), "down": ratio(down)} for name, step, up, down in rows]


ADDITIONS = ["2× enzyme", "+ Pi", "+ NADH", "+ GAP", "+ arsenate"]
DEFAULT_ADDITION_AMOUNTS = {"+ Pi": 40.0, "+ NADH": 0.05, "+ GAP": 0.5}


def perturbation(s: AssaySettings, addition: str, amount: float = 0.0) -> dict:
    """Instantaneous jump and relaxed change of A₃₄₀ when something is added at the plateau.

    The relaxed state equals the equilibrium of the start mixture plus the addition, because
    equilibrium depends only on the totals (and the drift on total turnover).
    """
    base = predict(s)
    eps_l = s.epsilon * s.path_cm
    jump = 0.0
    if addition == "2× enzyme":
        new = base  # enzyme changes the rate, never the equilibrium
        note = "Enzyme changes only the rate: the plateau stays where it is."
    elif addition == "+ Pi":
        new = predict(replace(s, pi=s.pi + amount))
        note = f"Pi is a substrate: {amount:g} mM more pushes the reaction forward."
    elif addition == "+ NADH":
        jump = eps_l * amount
        new = predict(replace(s, nadh=s.nadh + amount))
        note = "NADH is a product: the absorbance jumps, then part of it runs back."
    elif addition == "+ GAP":
        new = predict(replace(s, gap=s.gap + amount))
        note = f"{amount:g} mM GAP" + (" (DL: half is D-GAP)" if s.dl_racemate else "") + " pushes the reaction forward."
    elif addition == "+ arsenate":
        new = predict(replace(s, arsenate=True))
        note = "Arsenolysis is irreversible, so the reaction runs to completion of the limiting substrate."
    else:
        raise ValueError(f"unknown addition {addition!r}")
    relax = eps_l * (new.xi - base.xi)
    return {
        "addition": addition,
        "amount": amount,
        "jump": jump,
        "relaxation": relax,
        "net": jump + relax,
        "a_before": base.delta_a,
        "a_after": base.delta_a + jump + relax,
        "relax_xi": new.xi - base.xi,
        "note": note,
    }


def perturbation_trace(s: AssaySettings, addition: str, amount: float, kp: KineticParams,
                       t_add: float = C.T_END_MIN, t_after: float = C.T_END_MIN) -> dict:
    """Simulated A₃₄₀ trace: reaction to t_add, then the addition, then t_after more minutes."""
    m0 = s.mixture()
    kfun = k_function(s)
    pre = simulate(m0, kfun, kp, s.arsenate, t_add, s.epsilon, s.path_cm)
    xi1 = float(pre["xi"][-1])
    m1 = Mixture(m0.gap - xi1, m0.nad - xi1, m0.pi if s.arsenate else m0.pi - xi1, m0.bpg if s.arsenate else m0.bpg + xi1,
                 m0.nadh + xi1)
    kp2, arsenate2, jump = kp, s.arsenate, 0.0
    if addition == "2× enzyme":
        kp2 = replace(kp, v_max=kp.v_max * 2)
    elif addition == "+ Pi":
        m1 = replace(m1, pi=m1.pi + amount)
    elif addition == "+ NADH":
        m1 = replace(m1, nadh=m1.nadh + amount)
        jump = s.epsilon * s.path_cm * amount
    elif addition == "+ GAP":
        m1 = replace(m1, gap=m1.gap + (amount * C.DL_D_FRACTION if s.dl_racemate else amount))
    elif addition == "+ arsenate":
        arsenate2 = True
    # with buffer drift and/or the ionic correction, K′ after the addition follows the full model
    s_after = _settings_with_addition(s, addition, amount)
    k_after_fun = k_function(s_after)
    post = simulate(m1, lambda x: k_after_fun(xi1 + x), kp2, arsenate2, t_after, s.epsilon, s.path_cm)
    t = np.concatenate([pre["t"], t_add + post["t"]])
    a = np.concatenate([pre["dA"], pre["dA"][-1] + jump + post["dA"]])
    return {"t": t, "dA": a, "t_add": t_add}


def _settings_with_addition(s: AssaySettings, addition: str, amount: float) -> AssaySettings:
    if addition == "+ Pi":
        return replace(s, pi=s.pi + amount)
    if addition == "+ NADH":
        return replace(s, nadh=s.nadh + amount)
    if addition == "+ GAP":
        return replace(s, gap=s.gap + amount)
    if addition == "+ arsenate":
        return replace(s, arsenate=True)
    return s


def _perturbation_note(r: dict) -> str:
    def f(x: float) -> str:
        return f"{x:+.2f}".replace("-", "−")

    if abs(r["jump"]) < 5e-4 and abs(r["relaxation"]) < 5e-4:
        return "No change"
    if abs(r["jump"]) < 5e-4:
        return f"Rises {f(r['relaxation'])} ΔA" if r["relaxation"] > 0 else f"Falls {f(r['relaxation'])} ΔA"
    return f"Jumps {f(r['jump'])}, then relaxes {f(r['relaxation'])} ΔA"


def kcl_anchored_prediction(s: AssaySettings, kcl_new: float) -> Prediction:
    """Prediction at a new KCl level with K′ anchored on the current mixture.

    The current mixture's own ionic strength is taken as I_ref, so only the change in I
    acts (the 'relative' Debye–Hückel estimate).
    """
    i0, _ = _activity(s)
    return predict(replace(s, kcl=kcl_new, ionic_on=True, i_ref=i0))


def experiment_plan(s: AssaySettings, kp: KineticParams) -> list[dict]:
    """Predicted readouts for experiment sets A–E (spec §5, tab 9)."""
    rows: list[dict] = []

    def add(set_name: str, condition: str, s2: AssaySettings, note: str = "") -> None:
        p = predict(s2)
        rows.append({"set": set_name, "condition": condition, "xi_mM": p.xi, "delta_A340": p.delta_a,
                     "pct_GAP_oxidised": p.pct_gap, "note": note})

    for g in (0.1, 0.25, 0.5, 1.0):
        add("A: GAP titration", f"GAP {g:g} mM" + (" (DL)" if s.dl_racemate else ""), replace(s, gap=g))
    for p_ in (2.0, 10.0, 50.0):
        add("B1: Pi titration", f"Pi {p_:g} mM", replace(s, pi=p_))
    add("B2: KCl", "KCl 0 mM", replace(s, kcl=0.0), "Current model")
    s0 = replace(s, kcl=0.0)
    p_ideal = predict(replace(s0, kcl=100.0))
    rows.append({"set": "B2: KCl", "condition": "KCl 100 mM (ideal)", "xi_mM": p_ideal.xi,
                 "delta_A340": p_ideal.delta_a, "pct_GAP_oxidised": p_ideal.pct_gap,
                 "note": "Concentrations as activities: KCl has no effect"})
    p_dh = kcl_anchored_prediction(s0, 100.0)
    rows.append({"set": "B2: KCl", "condition": "KCl 100 mM (Debye–Hückel estimate)", "xi_mM": p_dh.xi,
                 "delta_A340": p_dh.delta_a, "pct_GAP_oxidised": p_dh.pct_gap,
                 "note": f"K′ ×{p_dh.activity_factor:.2g} relative to 0 mM KCl; approximate"})
    for add_name in ("2× enzyme", "+ Pi", "+ NADH", "+ GAP", "+ arsenate"):
        amt = DEFAULT_ADDITION_AMOUNTS.get(add_name, 0.0)
        r = perturbation(s, add_name, amt)
        label = add_name if add_name in ("2× enzyme", "+ arsenate") else f"{add_name} {amt:g} mM"
        rows.append({"set": "C: plateau perturbation", "condition": label, "xi_mM": predict(s).xi + r["relax_xi"],
                     "delta_A340": r["a_after"],
                     "pct_GAP_oxidised": predict(_settings_with_addition(s, add_name, amt)).pct_gap,
                     "note": _perturbation_note(r)})
    for ph in (7.5, 8.0, 8.5, 9.0):
        add("D: pH series", f"pH {ph:g}", replace(s, ph=ph))
    add("E: phosphate vs arsenate", "Phosphate (plateau)", replace(s, arsenate=False))
    add("E: phosphate vs arsenate", "Arsenate (completion)", replace(s, arsenate=True),
        "Above ~1.2 A, dilute or use a shorter path")
    tc = simulate(s.mixture(), k_function(replace(s, arsenate=False)), kp, True, C.T_END_MIN, s.epsilon, s.path_cm)
    rows.append({"set": "E: phosphate vs arsenate", "condition": f"Arsenate at {C.T_END_MIN:g} min",
                 "xi_mM": float(tc["xi"][-1]), "delta_A340": float(tc["dA"][-1]), "pct_GAP_oxidised":
                 100 * float(tc["xi"][-1]) / s.gap_d if s.gap_d > 0 else math.nan,
                 "note": "Time-course simulator, default kinetics"})
    return rows


def time_courses(s: AssaySettings, kp: KineticParams, t_end: float = C.T_END_MIN) -> dict:
    """Three overlays for the time-course tab: current settings, phosphate at pH 7.0, arsenate."""
    out = {}
    s_p7 = replace(s, ph=7.0, arsenate=False)
    if s.arsenate:
        specs = (
            ("current", s, f"Arsenate, pH {s.ph:g} (current)"),
            ("phosphate", replace(s, arsenate=False), f"Phosphate, pH {s.ph:g}"),
            ("ph7", s_p7, "Phosphate, pH 7.0"),
        )
    else:
        specs = (
            ("current", s, f"Phosphate, pH {s.ph:g} (current)"),
            ("ph7", s_p7, "Phosphate, pH 7.0"),
            ("arsenate", replace(s, arsenate=True), f"Arsenate, pH {s.ph:g}"),
        )
    for key, ss, label in specs:
        sim = simulate(ss.mixture(), k_function(replace(ss, arsenate=False)), kp, ss.arsenate, t_end, ss.epsilon,
                       ss.path_cm)
        eq = predict(ss)
        sim["label"] = label
        sim["equilibrium_dA"] = eq.delta_a
        sim["window_end"] = (initial_rate_window_end(sim["t"], sim["gamma_over_k"]) if not ss.arsenate else None)
        out[key] = sim
    return out

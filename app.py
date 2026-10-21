"""GAPDH Equilibrium Explorer — Streamlit user interface.

All science lives in the `gapdh` package; this file only collects inputs and displays
results. Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from gapdh import __version__
from gapdh import constants as C
from gapdh.activity import (
    gamma,
    gamma_pi,
    ionic_strength_breakdown,
    k_gamma,
    pi_dibasic_fraction,
)
from gapdh.buffer import buffer_capacity, protonation_alpha
from gapdh.kinetics import KineticParams, simulate
from gapdh.model import (
    ADDITIONS,
    DEFAULT_ADDITION_AMOUNTS,
    AssaySettings,
    experiment_plan,
    k_at_ph_function,
    k_function,
    kcl_anchored_prediction,
    perturbation,
    perturbation_trace,
    predict,
    sensitivity,
    time_courses,
    validate,
)
from gapdh.presets import CUSTOM, DEFAULT_PRESET, PRESET_NAMES, PRESETS, values_for
from gapdh.thermo import (
    delta_g0,
    energy_profile,
    flux_ratio_from_dg,
    k_from_plateau,
    k_from_plateau_with_error,
    snapshots,
)

st.set_page_config(
    page_title="GAPDH Equilibrium Explorer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Style: one accent for the current condition, greys for comparisons
# -----------------------------------------------------------------------------
ACCENT = "#2a78d6"
GREYS = ["#b3b2ac", "#9a9993", "#82817b", "#6e6d67"]  # mid greys that read on light and dark
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "gapdh_chart", "scale": 2},
}

st.markdown(
    """
    <style>
    .gapdh-header {
        border: 1px solid rgba(128,128,128,0.30);
        background: rgba(42,120,214,0.06);
        border-radius: 8px;
        padding: 1.1rem 1.5rem 0.9rem 1.5rem;
        margin-bottom: 1rem;
    }
    .gapdh-header h1 { margin: 0; padding: 0; font-size: 2.1rem; font-weight: 650; }
    .gapdh-header p  { margin: 0.25rem 0 0 0; font-size: 1.05rem; opacity: 0.8; }
    .gapdh-header .rxn { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.92rem; opacity: 0.75; margin-top: 0.4rem; }
    [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important;
                                      text-overflow: clip !important; }
    .gapdh-accent { height: 3px; border-radius: 2px; margin-top: 0.7rem;
                    background: linear-gradient(90deg, #2a78d6, rgba(42,120,214,0.15)); }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def sig(x: float | None, n: int = 3) -> str:
    """Format with n significant figures (never more than 4 in the UI)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    if x == 0:
        return "0"
    mag = math.floor(math.log10(abs(x)))
    if -3 <= mag < 5:
        s = f"{x:.{max(n - 1 - mag, 0)}f}"
    else:
        s = f"{x:.{n - 1}e}"
    return s.replace("-", "−")


def signed(x: float, n: int = 3) -> str:
    """Signed value with n significant figures; exact zero shown as 0."""
    if x is None or not math.isfinite(x) or abs(x) < 5e-5:
        return "0" if x is not None and math.isfinite(x) else sig(x)
    return ("+" if x > 0 else "") + sig(x, n)


def round_sig(x, n: int = 4):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float, np.floating)) and math.isfinite(x) and x != 0:
        return float(f"{x:.{n}g}")
    return x


def df_sig(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda v: round_sig(v, n))
    return out


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df_sig(df).to_csv(index=False).encode("utf-8")


def download_csv(df: pd.DataFrame, name: str, label: str = "Download CSV", key: str | None = None) -> None:
    st.download_button(label, csv_bytes(df), file_name=name, mime="text/csv", key=key or f"dl_{name}",
                       icon=":material/download:")


def show_table(df: pd.DataFrame, fmt: str = "%.3g", height: int | str = "auto", **col_cfg) -> None:
    cfg = {c: st.column_config.NumberColumn(c, format=fmt) for c in df.columns
           if pd.api.types.is_numeric_dtype(df[c]) and c not in col_cfg}
    cfg.update(col_cfg)
    st.dataframe(df, hide_index=True, column_config=cfg, height=height, placeholder="—")


def style(fig: go.Figure, x_title: str, y_title: str, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        showlegend=legend,
        hovermode="closest",
        font=dict(family="Arial, sans-serif"),
    )
    fig.update_xaxes(title_text=x_title, showline=True, ticks="outside", showgrid=False, zeroline=False)
    fig.update_yaxes(title_text=y_title, showline=True, ticks="outside", gridcolor="rgba(128,128,128,0.15)",
                     zeroline=False)
    return fig


def plot(fig: go.Figure, key: str) -> None:
    cfg = dict(PLOT_CONFIG)
    cfg["toImageButtonOptions"] = {**PLOT_CONFIG["toImageButtonOptions"], "filename": f"gapdh_{key}"}
    st.plotly_chart(fig, config=cfg, key=key)


def settings_key(s: AssaySettings) -> str:
    return json.dumps(s.to_dict(), sort_keys=True)


def safe(render, *args) -> None:
    """Never let one tab crash the app."""
    try:
        render(*args)
    except Exception as exc:  # noqa: BLE001 - shown to the user instead of a traceback
        st.error(f"This section could not be computed for the current inputs "
                 f"({type(exc).__name__}: {exc}). Try less extreme values.")


# -----------------------------------------------------------------------------
# Preset handling: every sidebar input lives in st.session_state["in_<field>"]
# -----------------------------------------------------------------------------
FIELDS = list(values_for(DEFAULT_PRESET).keys())


def _apply_preset(name: str) -> None:
    for k, v in values_for(name).items():
        st.session_state[f"in_{k}"] = v
    st.session_state["base_preset"] = name


def _on_preset_change() -> None:
    name = st.session_state["preset_choice"]
    if name != CUSTOM:
        _apply_preset(name)


def _on_reset() -> None:
    base = st.session_state.get("base_preset", DEFAULT_PRESET)
    _apply_preset(base)
    st.session_state["preset_choice"] = base


def _is_modified() -> bool:
    base = values_for(st.session_state.get("base_preset", DEFAULT_PRESET))
    for k, v in base.items():
        cur = st.session_state.get(f"in_{k}", v)
        if isinstance(v, float) or isinstance(cur, float):
            if not math.isclose(float(cur), float(v), rel_tol=1e-9, abs_tol=1e-12):
                return True
        elif cur != v:
            return True
    return False


if "base_preset" not in st.session_state:
    _apply_preset(DEFAULT_PRESET)
    st.session_state["preset_choice"] = DEFAULT_PRESET
MODIFIED = _is_modified()
if MODIFIED and st.session_state.get("preset_choice") != CUSTOM:
    st.session_state["preset_choice"] = CUSTOM


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
def sidebar() -> AssaySettings:
    sb = st.sidebar
    base = st.session_state.get("base_preset", DEFAULT_PRESET)

    sb.header("Preset")
    sb.selectbox(
        "Preset",
        PRESET_NAMES,
        key="preset_choice",
        on_change=_on_preset_change,
        format_func=lambda n: (f"Custom (modified from {base})" if n == CUSTOM and MODIFIED else n),
        help="Named starting points from the build spec (§2). Editing any input switches to Custom.",
        label_visibility="collapsed",
    )
    if st.session_state["preset_choice"] == CUSTOM:
        sb.caption(f"Custom values, starting from **{base}**." + (" Modified." if MODIFIED else ""))
    else:
        sb.caption(PRESETS[st.session_state["preset_choice"]]["description"])
    sb.button("Reset to preset", on_click=_on_reset, icon=":material/restart_alt:",
              help=f"Restore every input to “{base}”.")

    sb.header("Conditions")
    c1, c2 = sb.columns(2)
    c1.number_input("pH", min_value=C.PH_RANGE[0], max_value=C.PH_RANGE[1], step=0.1, format="%.2f", key="in_ph",
                    help="Assay pH (allowed 4–11). Default 8.5 — the lab's current assay.")
    c2.number_input("T (°C)", min_value=C.TEMP_RANGE_C[0], max_value=C.TEMP_RANGE_C[1], step=1.0, format="%.1f",
                    key="in_temp_c", help="Assay temperature (allowed 0–80 °C). Default 25 °C — the lab's current assay.")

    sb.header("Initial concentrations")
    c1, c2 = sb.columns(2)
    c1.number_input("GAP (mM)", min_value=0.0, step=0.05, format="%.3f", key="in_gap",
                    help="Glyceraldehyde-3-phosphate as added. Default 0.5 mM — the lab's current assay.")
    c2.number_input("NAD⁺ (mM)", min_value=0.0, step=0.1, format="%.3f", key="in_nad",
                    help="Default 1.0 mM — the lab's current assay.")
    sb.toggle("GAP is DL-racemate", key="in_dl_racemate",
              help="Only D-GAP is a substrate. When on, effective D-GAP = 50 % of the entered value "
                   "(spec §2); the L-isomer still counts toward ionic strength.")
    gap_entered = st.session_state["in_gap"]
    if st.session_state["in_dl_racemate"]:
        sb.caption(f"Entered {sig(gap_entered)} mM DL-GAP → **{sig(gap_entered * C.DL_D_FRACTION)} mM D-GAP** used.")
    c1, c2 = sb.columns(2)
    c1.number_input("Pi (mM)", min_value=0.0, step=1.0, format="%.2f", key="in_pi",
                    help="Inorganic phosphate (sodium phosphate). It is both substrate and buffer. "
                         "Default 10 mM — the lab's current assay.")
    c2.number_input("1,3-BPG at t = 0 (mM)", min_value=0.0, step=0.01, format="%.3f", key="in_bpg",
                    help="Product present at the start. Default 0 mM (forward assay); 0.1 mM in the reverse preset.")
    sb.number_input("NADH at t = 0 (mM)", min_value=0.0, step=0.01, format="%.3f", key="in_nadh",
                    help="NADH present at the start, e.g. from the NAD⁺ stock. Default 0 mM; 0.2 mM in the reverse preset.")

    sb.header("Medium")
    c1, c2 = sb.columns(2)
    c1.number_input("EDTA (mM)", min_value=0.0, step=0.5, format="%.2f", key="in_edta",
                    help="No thermodynamic role; counts toward ionic strength only (charge ≈ −3.5). "
                         "Default 1 mM — the lab's current assay.")
    c2.number_input("KCl (mM)", min_value=0.0, step=10.0, format="%.1f", key="in_kcl",
                    help="Counts toward ionic strength only. Default 0 mM — the lab's current assay.")
    sb.number_input("Phosphate pKa₂ for buffering", min_value=4.0, max_value=10.0, step=0.05, format="%.2f",
                    key="in_pi_pka_buffer",
                    help="Used for the buffer capacity of the Pi entered above. Default 7.2 (spec §2, "
                         "near the I → 0 value). Speciation and charges use the apparent pKa₂ 6.8 (spec §3.5).")
    sb.toggle("Add a second buffer", key="in_buffer2_on",
              help="An extra buffer for the drift correction, e.g. 50 mM TAPS, pKa 8.4 (spec §3.4).")
    on2 = st.session_state["in_buffer2_on"]
    sb.text_input("Second buffer name", key="in_buffer2_name", disabled=not on2,
                  help="Label only. Example from spec §3.4: TAPS.")
    c1, c2 = sb.columns(2)
    c1.number_input("Conc. (mM)", min_value=0.0, step=5.0, format="%.1f", key="in_buffer2_conc", disabled=not on2,
                    help="Concentration of the second buffer. Example: 50 mM (spec §3.4).")
    c2.number_input("pKa", min_value=2.0, max_value=12.0, step=0.05, format="%.2f", key="in_buffer2_pka",
                    disabled=not on2, help="pKa of the second buffer. Example: TAPS 8.4 (spec §3.4).")

    sb.header("Thermodynamic constants")
    c1, c2 = sb.columns(2)
    c1.number_input("K′ref (M⁻¹)", min_value=1e-6, step=0.005, format="%.4f", key="in_k_ref",
                    help="Apparent K′ at pHref and 25 °C. Default 0.078 M⁻¹ — textbook value, "
                         "equivalent to ΔG′° = +6.3 kJ/mol at pH 7 (BOKU lecture notes).")
    c2.number_input("pHref", min_value=4.0, max_value=11.0, step=0.1, format="%.2f", key="in_ph_ref",
                    help="pH at which K′ref applies. Default 7.0 — textbook convention.")
    c1, c2 = sb.columns(2)
    c1.number_input("Proton slope s (log K′ / pH)", min_value=0.0, max_value=3.0, step=0.05, format="%.2f",
                    key="in_slope", help="Change in log K′ per pH unit. Default 1.0 — one H⁺ released per turnover.")
    c2.number_input("ΔH′° (kJ/mol)", min_value=-200.0, max_value=200.0, step=0.5, format="%.2f", key="in_delta_h",
                    help="Reaction enthalpy for the van 't Hoff temperature term. Default +4.6 kJ/mol — "
                         "Greinert et al. 2020, titration calorimetry.")
    sb.number_input("Reference ionic strength of K′ (M)", min_value=0.0, max_value=2.0, step=0.01, format="%.3f",
                    key="in_i_ref",
                    help="Ionic strength at which K′ref is assumed valid. Default 0.25 M — an ASSUMPTION: "
                         "the textbook source does not state it.")
    sb.caption("⚠️ The textbook K′ does not state its ionic strength; 0.25 M is an assumption.")

    sb.header("Optics")
    c1, c2 = sb.columns(2)
    c1.number_input("ε₃₄₀ (mM⁻¹ cm⁻¹)", min_value=0.01, step=0.01, format="%.3f", key="in_epsilon",
                    help="NADH molar absorptivity at 340 nm. Default 6.22 mM⁻¹ cm⁻¹ — standard value.")
    c2.number_input("Path length (cm)", min_value=0.001, step=0.1, format="%.3f", key="in_path_cm",
                    help="Cuvette or well path length. Default 1 cm — standard cuvette.")

    sb.header("Corrections")
    sb.toggle("Buffer-drift correction", key="in_drift_on",
              help="One H⁺ per turnover lowers the pH, which lowers K′ (spec §3.4). Default off.")
    sb.toggle("Ionic-strength correction (approximate)", key="in_ionic_on",
              help="Extended Debye–Hückel (Alberty), K′(I) = K′(I_ref)·Kγ(I_ref)/Kγ(I) (spec §3.5). "
                   "Default off. Greinert et al. 2021: this underestimates non-ideality.")
    sb.toggle("Arsenate instead of phosphate (irreversible)", key="in_arsenate",
              help="Arsenolysis: the product hydrolyses, so the reaction runs to completion "
                   "(the “Forward with arsenate” preset).")

    vals = {k: st.session_state[f"in_{k}"] for k in FIELDS}
    return AssaySettings.from_dict(vals)


RAW = sidebar()
S, WARNINGS = validate(RAW)
P = predict(S)
T_K = S.temp_k


def kinetic_params() -> KineticParams:
    ss = st.session_state
    return KineticParams(
        v_max=ss.get("tc_v", C.V_MAX_MM_MIN),
        km_gap=ss.get("tc_kmg", C.KM_GAP_MM),
        km_nad=ss.get("tc_kmn", C.KM_NAD_MM),
        ki_nadh=ss.get("tc_ki", C.KI_NADH_MM),
        inhibition=ss.get("tc_inh", False),
    )


# -----------------------------------------------------------------------------
# Cached grids
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def grid_vs_ph(key: str) -> pd.DataFrame:
    s = AssaySettings.from_dict(json.loads(key))
    k_at = k_at_ph_function(s)
    rows = []
    for ph in np.round(np.linspace(6.0, 10.0, 81), 3):
        k = k_at(float(ph))
        p = predict(replace(s, ph=float(ph)))
        rows.append({"pH": ph, "K′ (M⁻¹)": k, "ΔG′° (kJ/mol)": delta_g0(k, s.temp_k),
                     "% GAP oxidised": p.pct_gap, "ξ (mM)": p.xi, "ΔA₃₄₀": p.delta_a})
    return pd.DataFrame(rows)


MAP_PHS = [7.0, 7.5, 8.0, 8.5, 9.0]


@st.cache_data(show_spinner=False)
def grid_maps(key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = AssaySettings.from_dict(json.loads(key))
    phs = sorted(set(MAP_PHS + [round(s.ph, 3)]))
    gap_rows, pi_rows = [], []
    for ph in phs:
        for g in np.geomspace(0.05, 5.0, 40):
            p = predict(replace(s, ph=ph, gap=float(g)))
            gap_rows.append({"pH": ph, "GAP (mM)": g, "ΔA₃₄₀": p.delta_a, "% GAP oxidised": p.pct_gap})
        for pi in np.geomspace(1.0, 100.0, 40):
            p = predict(replace(s, ph=ph, pi=float(pi)))
            pi_rows.append({"pH": ph, "Pi (mM)": pi, "% GAP oxidised": p.pct_gap, "ΔA₃₄₀": p.delta_a})
    return pd.DataFrame(gap_rows), pd.DataFrame(pi_rows)


@st.cache_data(show_spinner=False)
def grid_heatmap(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = AssaySettings.from_dict(json.loads(key))
    phs = np.round(np.linspace(6.5, 9.5, 31), 3)
    pis = np.geomspace(1.0, 100.0, 25)
    z = np.array([[predict(replace(s, ph=float(ph), pi=float(pi))).pct_gap for ph in phs] for pi in pis])
    return phs, pis, z


@st.cache_data(show_spinner=False)
def cached_sensitivity(key: str) -> list[dict]:
    return sensitivity(AssaySettings.from_dict(json.loads(key)))


@st.cache_data(show_spinner=False)
def cached_plan(key: str, kp_key: str) -> pd.DataFrame:
    s = AssaySettings.from_dict(json.loads(key))
    kp = KineticParams(**json.loads(kp_key))
    return pd.DataFrame(experiment_plan(s, kp))


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="gapdh-header">
      <h1>GAPDH Equilibrium Explorer</h1>
      <p>Where the GAPDH reaction stops, and why.</p>
      <div class="rxn">D-GAP + NAD⁺ + Pi ⇌ 1,3-BPG + NADH + H⁺</div>
      <div class="gapdh-accent"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

reverse = P.xi < 0
cards = st.columns(6)
if reverse:
    cards[0].metric("% BPG reduced", f"{sig(P.pct_bpg_reduced)} %", border=True, height="stretch",
                    help="Reverse direction: share of the starting 1,3-BPG reduced at equilibrium.")
else:
    cards[0].metric("% GAP oxidised", f"{sig(P.pct_gap)} %", border=True, height="stretch",
                    help="100·ξ/G₀, relative to D-GAP." + (" Arsenate: runs to completion." if P.arsenate else ""))
cards[1].metric("NADH at equilibrium (mM)", sig(P.nadh_eq), border=True, height="stretch",
                help="Total NADH once the reaction stops (N₀ + ξ).")
cards[2].metric("ΔA₃₄₀", sig(P.delta_a), border=True, height="stretch",
                help=f"ε·l·ξ with ε = {sig(S.epsilon)} mM⁻¹ cm⁻¹ and l = {sig(S.path_cm)} cm. "
                     "Negative means NADH is consumed.")
cards[3].metric("K′ (M⁻¹)", sig(P.k_start), border=True, height="stretch",
                help="Apparent equilibrium constant at the starting pH and T"
                     + (", with the activity correction" if S.ionic_on else "") + ".")
cards[4].metric("ΔG′° (kJ/mol)", sig(P.dg0), border=True, height="stretch", help="−RT ln K′, 1 M standard state.")
cards[5].metric("κ", sig(P.kappa), border=True, height="stretch", help="κ = (K′/1000)·[Pi]₀, dimensionless.")

with st.container(horizontal=True):
    st.badge(f"Buffer drift: {'on' if S.drift_on else 'off'}", color="blue" if S.drift_on else "gray",
             icon=":material/water_drop:")
    st.badge(f"Ionic strength: {'on (approx.)' if S.ionic_on else 'off'}", color="blue" if S.ionic_on else "gray",
             icon=":material/bolt:")
    st.badge("Arsenate: irreversible" if S.arsenate else "Phosphate: reversible",
             color="orange" if S.arsenate else "gray", icon=":material/science:")
    if S.dl_racemate:
        st.badge(f"DL-GAP: {sig(S.gap_d)} mM D-GAP used", color="gray")
    if S.drift_on:
        st.badge(f"Final pH {sig(P.ph_final, 4)}", color="gray")

for w in WARNINGS:
    st.warning(w)

TABS = st.tabs([
    "Equilibrium", "Energy profile", "pH & buffer", "Outcome maps", "Ionic strength",
    "Time course", "Reverse & spikes", "From data to K′", "Experiment planner", "Theory & references",
])


# -----------------------------------------------------------------------------
# 1. Equilibrium
# -----------------------------------------------------------------------------
def tab_equilibrium() -> None:
    m0, me = P.mix0, P.mix_eq
    if P.arsenate:
        st.info("Arsenate mode: 1-arseno-3-phosphoglycerate hydrolyses, so there is no equilibrium. "
                "The reaction runs to completion of the limiting substrate (GAP or NAD⁺). Pi is not consumed.")
    names = ["D-GAP" if S.dl_racemate else "GAP", "NAD⁺", "Pi", "1,3-BPG" + (" (→ 3-PG)" if P.arsenate else ""),
             "NADH"]
    init = [m0.gap, m0.nad, m0.pi, m0.bpg, m0.nadh]
    eq = [me.gap, me.nad, me.pi, me.bpg, me.nadh]
    table = pd.DataFrame({"Species": names, "Initial (mM)": init,
                          "Change (mM)": [e - i for e, i in zip(eq, init)], "Equilibrium (mM)": eq})

    c1, c2 = st.columns([5, 4])
    with c1:
        st.subheader("Concentrations")
        show_table(table)
        download_csv(table, "gapdh_equilibrium_table.csv")
    with c2:
        fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2], horizontal_spacing=0.12)
        small = [0, 1, 3, 4]
        for name, vals, color in (("Initial", init, GREYS[1]), ("Equilibrium" if not P.arsenate else "Final", eq, ACCENT)):
            fig.add_trace(go.Bar(x=[names[i] for i in small], y=[vals[i] for i in small], name=name,
                                 marker_color=color, hovertemplate="%{x}: %{y:.3g} mM<extra>" + name + "</extra>"),
                          row=1, col=1)
            fig.add_trace(go.Bar(x=["Pi"], y=[vals[2]], name=name, marker_color=color, showlegend=False,
                                 hovertemplate="Pi: %{y:.4g} mM<extra>" + name + "</extra>"), row=1, col=2)
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
        style(fig, "", "Concentration (mM)", height=330)
        fig.update_yaxes(title_text="", row=1, col=2)
        plot(fig, "equilibrium_bars")
        st.caption("Pi is drawn on its own scale (right) because it is in large excess.")

    st.subheader("Result")
    a, b, c, d = st.columns(4)
    if reverse:
        a.metric("Extent ξ (mM)", sig(P.xi), help="Negative: the reaction ran backwards (BPG reduced).")
        b.metric("% BPG reduced", f"{sig(P.pct_bpg_reduced)} %")
    else:
        a.metric("Extent ξ = NADH formed (mM)", sig(P.xi))
        b.metric("% NAD⁺ reduced", f"{sig(P.pct_nad)} %")
    c.metric("ΔA₃₄₀", sig(P.delta_a))
    if P.dg_start == -math.inf:
        dg_txt, dg_note = "−∞", "−∞ (no products yet)"
    elif P.dg_start == math.inf:
        dg_txt, dg_note = "+∞", "+∞ (no substrates yet)"
    else:
        dg_txt, dg_note = sig(P.dg_start), ""
    d.metric("ΔG′ of the starting mixture (kJ/mol)", dg_txt,
             help="RT ln(Γ/K′) before the reaction starts; negative means the forward direction is downhill.")
    if dg_note:
        d.caption(f"ΔG′ start: {dg_note}")

    if not P.arsenate:
        if P.gamma_matches_k:
            st.success(f"Check: Γ at equilibrium = {sig(P.gamma_eq)} M⁻¹ = K′ = {sig(P.k_final)} M⁻¹ ✓"
                       + (f" (K′ at the final pH {sig(P.ph_final, 4)})" if S.drift_on else ""))
        elif P.xi == 0:
            st.info("No reaction is possible with these concentrations, so Γ = K′ cannot be reached.")
        else:
            st.warning(f"Γ at equilibrium = {sig(P.gamma_eq)} M⁻¹ differs from K′ = {sig(P.k_final)} M⁻¹.")

    if not P.arsenate:
        st.subheader("Teaching shortcuts")
        rows = [{"Method": "Exact (cubic, Pi consumed)", "ξ (mM)": P.xi, "Error vs exact (%)": 0.0}]
        if math.isfinite(P.xi_quadratic):
            rows.append({"Method": "Quadratic shortcut (Pi held constant)", "ξ (mM)": P.xi_quadratic,
                         "Error vs exact (%)": 100 * (P.xi_quadratic / P.xi - 1) if P.xi else math.nan})
        if m0.bpg == 0 and m0.nadh == 0 and P.xi > 0:
            rows.append({"Method": "Square-root rule √(κ·G₀·D₀)", "ξ (mM)": P.xi_sqrt,
                         "Error vs exact (%)": 100 * (P.xi_sqrt / P.xi - 1)})
        show_table(pd.DataFrame(rows))
        st.caption("The square-root rule assumes small conversion and no products at the start; it overestimates "
                   "ξ because it ignores substrate depletion. The quadratic is nearly exact because Pi is in excess.")

    st.subheader("Checks")
    flagged = False
    if abs(P.delta_a) > C.A340_LINEAR_LIMIT:
        st.warning(f"ΔA₃₄₀ = {sig(P.delta_a)} is above {C.A340_LINEAR_LIMIT}: check photometric linearity "
                   "(dilute, or use a shorter path).")
        flagged = True
    if m0.nadh > 0 and P.xi >= 0 and not P.arsenate:
        clean = predict(replace(S, nadh=0.0))
        st.warning(f"NADH present at t = 0 ({sig(m0.nadh)} mM, e.g. from the NAD⁺ stock) lowers the extent: "
                   f"ξ = {sig(P.xi)} mM instead of {sig(clean.xi)} mM without it. Read the blank's A₃₄₀ before "
                   "adding enzyme.")
        flagged = True
    pct = P.pct_bpg_reduced if reverse else P.pct_gap
    if math.isfinite(pct) and abs(pct) < C.SMALL_SIGNAL_PERCENT and P.xi != 0:
        st.warning(f"Only {sig(abs(pct))} % conversion: signal too small to measure reliably.")
        flagged = True
    if not flagged:
        st.caption("No warnings. Commercial NAD⁺ can contain traces of NADH; reading the blank's A₃₄₀ before adding "
                   "enzyme rules this out.")


# -----------------------------------------------------------------------------
# 2. Energy profile
# -----------------------------------------------------------------------------
def tab_energy() -> None:
    if P.arsenate:
        st.info("Arsenate makes the reaction irreversible. The profile below is for the phosphate reaction "
                "with the same mixture.")
    p = P if not P.arsenate else predict(replace(S, arsenate=False))
    if p.xi == 0:
        st.info("No reaction is possible with these concentrations, so there is no profile to draw.")
        return
    k = p.k_start
    spike = st.columns([1, 3])[0].number_input("NADH spike for snapshot d (mM)", min_value=0.0, value=C.DEFAULT_NADH_SPIKE_MM,
                            step=0.01, format="%.3f", key="spike",
                            help="Added to the plateau mixture. Default 0.05 mM (spec §3.3).")
    if S.drift_on:
        st.caption("The profile uses K′ at the starting pH (buffer drift is not drawn here).")
    pts = pd.DataFrame(energy_profile(p.mix0, k, T_K, p.xi))
    snaps = snapshots(p.mix0, k, T_K, p.xi, spike)
    labels = {"a": "a · 1 μM NADH " + ("consumed" if p.xi < 0 else "formed"), "b": "b · half of ξ_eq",
              "c": "c · equilibrium",
              "d": f"d · plateau + {spike:g} mM NADH"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pts["x"], y=pts["dg"], mode="lines", line=dict(color=ACCENT, width=2),
                             name="ΔG′ along the reaction",
                             hovertemplate="ξ %{x:.3g} mM<br>ΔG′ %{y:.3g} kJ/mol<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=GREYS[2], width=1, dash="dot"))
    for key in ("a", "b", "c"):
        sp = snaps[key]
        fig.add_trace(go.Scatter(x=[sp["x"]], y=[sp["dg"]], mode="markers+text", text=[key], textposition="top right",
                                 marker=dict(size=10, color=ACCENT, line=dict(width=2, color="white")),
                                 name=labels[key], showlegend=False,
                                 hovertemplate=labels[key] + "<br>ΔG′ %{y:.3g} kJ/mol<extra></extra>"))
    d = snaps["d"]
    if spike > 0:
        fig.add_trace(go.Scatter(x=[d["x"]], y=[d["dg"]], mode="markers+text", text=["d"], textposition="top right",
                                 marker=dict(size=10, color=GREYS[3], symbol="diamond", line=dict(width=2, color="white")),
                                 name=labels["d"], showlegend=False,
                                 hovertemplate=labels["d"] + "<br>ΔG′ %{y:.3g} kJ/mol<extra></extra>"))
        fig.add_annotation(x=d["x"] + d["relaxation_xi"], y=0, ax=d["x"], ay=d["dg"], xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=GREYS[3],
                           text="")
    finite = pts["dg"][np.isfinite(pts["dg"])]
    lo = max(float(finite.min()), -25.0) if len(finite) else -10
    hi = max(float(finite.max()), d["dg"] if spike > 0 and math.isfinite(d["dg"]) else 0) + 2.5
    fig.update_yaxes(range=[lo - 1, hi])
    style(fig, "Extent ξ = NADH formed (mM)", "ΔG′ (kJ/mol)", height=380, legend=False)

    flux = go.Figure()
    flux.add_trace(go.Scatter(x=pts["x"], y=pts["flux_ratio"], mode="lines", line=dict(color=ACCENT, width=2),
                              name="J₊/J₋", hovertemplate="ξ %{x:.3g} mM<br>J₊/J₋ %{y:.3g}<extra></extra>"))
    flux.add_hline(y=1, line=dict(color=GREYS[2], width=1, dash="dot"))
    for key in ("a", "b", "c"):
        sp = snaps[key]
        flux.add_trace(go.Scatter(x=[sp["x"]], y=[sp["flux_ratio"]], mode="markers+text", text=[key],
                                  textposition="top right", marker=dict(size=9, color=ACCENT), showlegend=False,
                                  hovertemplate=labels[key] + "<br>J₊/J₋ %{y:.3g}<extra></extra>"))
    flux.update_yaxes(type="log", dtick=2, exponentformat="power")
    style(flux, "Extent ξ = NADH formed (mM)", "J₊/J₋ (forward / backward flux)", height=380, legend=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Gibbs energy of the mixture along the reaction**")
        plot(fig, "energy_profile")
    with c2:
        st.markdown("**Forward/backward flux ratio** (log scale)")
        plot(flux, "flux_ratio")

    st.markdown(
        "**The thermodynamic brake.** The net rate is the forward rate multiplied by (1 − Γ/K′): with no product "
        "Γ/K′ = 0 and the enzyme runs at full speed, but as NADH and 1,3-BPG accumulate the backward flux grows until "
        "it cancels the forward flux and the net rate reaches zero at Γ = K′. Adding enzyme speeds up both "
        "directions equally, so it changes how fast the plateau is reached, never where it is."
    )

    rows = []
    for key in ("a", "b", "c", "d"):
        sp = snaps[key]
        clean = lambda v: 0.0 if (math.isfinite(v) and abs(v) < 1e-9) else v  # noqa: E731
        rows.append({"Snapshot": labels[key], "NADH formed (mM)": sp["x"], "Γ (M⁻¹)": sp["gamma"],
                     "Γ/K′": sp["gamma_over_k"], "ΔG′ (kJ/mol)": clean(sp["dg"]), "J₊/J₋": sp["flux_ratio"],
                     "1 − Γ/K′": clean(sp["net_fraction"])})
    snap_df = pd.DataFrame(rows)
    c1, c2 = st.columns([7, 3])
    with c1:
        st.markdown("**Snapshots**")
        show_table(snap_df)
        if spike > 0:
            st.caption(f"After the spike the reaction runs backwards by {sig(abs(d['relaxation_xi']))} mM "
                       f"(ΔA₃₄₀ {sig(S.epsilon * S.path_cm * d['relaxation_xi'])}) until Γ = K′ again.")
        download_csv(snap_df, "gapdh_snapshots.csv", key="dl_snap")
    with c2:
        st.markdown(f"**ΔG′ to flux ratio** (at {sig(S.temp_c)} °C)")
        fr = pd.DataFrame({"ΔG′ (kJ/mol)": [0, -1, -2.5, -5.7, -11.4, -17.1]})
        fr["J₊/J₋"] = [flux_ratio_from_dg(x, T_K) for x in fr["ΔG′ (kJ/mol)"]]
        show_table(fr)
    download_csv(pts.rename(columns={"x": "xi_mM", "gamma": "Gamma_per_M", "gamma_over_k": "Gamma_over_K",
                                     "dg": "dG_kJ_per_mol", "flux_ratio": "Jplus_over_Jminus",
                                     "net_fraction": "net_fraction"}),
                 "gapdh_energy_profile.csv", "Download profile CSV", key="dl_profile")


# -----------------------------------------------------------------------------
# 3. pH & buffer
# -----------------------------------------------------------------------------
def tab_ph() -> None:
    df = grid_vs_ph(settings_key(S))
    k_now = k_at_ph_function(S)(S.ph)

    c1, c2 = st.columns(2)
    with c1:
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=df["pH"], y=df["K′ (M⁻¹)"], mode="lines", line=dict(color=ACCENT, width=2),
                                name="K′", hovertemplate="pH %{x:.2f}<br>K′ %{y:.3g} M⁻¹<extra></extra>"))
        f1.add_trace(go.Scatter(x=[S.ph], y=[k_now], mode="markers", marker=dict(size=10, color=ACCENT),
                                name="current", hovertemplate="current<br>K′ %{y:.3g} M⁻¹<extra></extra>"))
        f1.update_yaxes(type="log", dtick=1, exponentformat="power")
        style(f1, "pH", "K′ (M⁻¹, log scale)", height=330, legend=False)
        st.markdown("**K′ vs pH**")
        plot(f1, "k_vs_ph")
    with c2:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=df["pH"], y=df["ΔG′° (kJ/mol)"], mode="lines", line=dict(color=ACCENT, width=2),
                                name="ΔG′°", hovertemplate="pH %{x:.2f}<br>ΔG′° %{y:.3g} kJ/mol<extra></extra>"))
        f2.add_trace(go.Scatter(x=[S.ph], y=[delta_g0(k_now, T_K)], mode="markers", marker=dict(size=10, color=ACCENT),
                                name="current", hovertemplate="current<br>ΔG′° %{y:.3g} kJ/mol<extra></extra>"))
        f2.add_hline(y=0, line=dict(color=GREYS[2], width=1, dash="dot"))
        style(f2, "pH", "ΔG′° (kJ/mol)", height=330, legend=False)
        st.markdown("**ΔG′° vs pH**")
        plot(f2, "dg0_vs_ph")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**% GAP oxidised vs pH** (current mixture and switches)")
        f3 = go.Figure()
        f3.add_trace(go.Scatter(x=df["pH"], y=df["% GAP oxidised"], mode="lines", line=dict(color=ACCENT, width=2),
                                hovertemplate="pH %{x:.2f}<br>%{y:.3g} % oxidised<extra></extra>"))
        if math.isfinite(P.pct_gap):
            f3.add_trace(go.Scatter(x=[S.ph], y=[P.pct_gap], mode="markers", marker=dict(size=10, color=ACCENT),
                                    hovertemplate="current<br>%{y:.3g} %<extra></extra>"))
        style(f3, "pH", "GAP oxidised (%)", height=320, legend=False)
        plot(f3, "pct_vs_ph")
        download_csv(df, "gapdh_vs_pH.csv", key="dl_ph")
    with c2:
        st.markdown("**Phosphate speciation** (pKa₂ 6.8)")
        phs = [7.0, 7.5, 8.0, 8.5] + ([round(S.ph, 2)] if round(S.ph, 2) not in (7.0, 7.5, 8.0, 8.5) else [])
        spec = pd.DataFrame({"pH": phs, "HPO₄²⁻ (%)": [100 * pi_dibasic_fraction(x) for x in phs]})
        spec["H₂PO₄⁻ (%)"] = 100 - spec["HPO₄²⁻ (%)"]
        show_table(spec, fmt="%.3g")
        st.caption("Pi in the reaction is the total of both forms; K′ already contains the pH dependence.")

    st.subheader("Buffer capacity and pH drift")
    rows = [{"Buffer": b.name, "Conc. (mM)": b.conc_mM, "pKa": b.pKa,
             "Base fraction α": protonation_alpha(S.ph, b.pKa), "β (mM/pH)": buffer_capacity(b.conc_mM, b.pKa, S.ph)}
            for b in S.buffers()]
    bt = pd.DataFrame(rows)
    pd_on = predict(replace(S, drift_on=True))
    pd_off = predict(replace(S, drift_on=False))
    c1, c2 = st.columns([3, 2])
    with c1:
        show_table(bt)
        st.caption(f"Total β = {sig(pd_on.beta)} mM per pH unit at pH {sig(S.ph, 3)}. One H⁺ is released per "
                   "turnover, so pH(ξ) = pH₀ − ξ/β.")
    with c2:
        dd = pd.DataFrame([
            {"Model": "No drift", "Plateau ξ (mM)": pd_off.xi, "ΔA₃₄₀": pd_off.delta_a, "Final pH": S.ph},
            {"Model": "With drift", "Plateau ξ (mM)": pd_on.xi, "ΔA₃₄₀": pd_on.delta_a, "Final pH": pd_on.ph_final},
        ])
        if not S.buffer2_on:
            ex = C.EXAMPLE_SECOND_BUFFER
            s_t = replace(S, drift_on=True, buffer2_on=True, buffer2_name=ex["name"], buffer2_conc=ex["conc_mM"],
                          buffer2_pka=ex["pKa"])
            pt = predict(s_t)
            dd.loc[len(dd)] = {"Model": f"Drift + {ex['conc_mM']:g} mM {ex['name']}", "Plateau ξ (mM)": pt.xi,
                               "ΔA₃₄₀": pt.delta_a, "Final pH": pt.ph_final}
        show_table(dd, fmt="%.4g")
    drift = S.ph - pd_on.ph_final
    st.caption(("The drift correction is **on**, so the headline numbers include it. " if S.drift_on else
                "The drift correction is **off**; switch it on in the sidebar to use the drifting plateau. ")
               + f"Expected drift: {sig(abs(drift))} pH units.")


# -----------------------------------------------------------------------------
# 4. Outcome maps
# -----------------------------------------------------------------------------
def _ph_lines(df: pd.DataFrame, xcol: str, ycol: str, xlabel: str, ylabel: str, current_x: float, key: str,
              yfmt: str) -> None:
    fig = go.Figure()
    phs = sorted(df["pH"].unique())
    others = [p for p in phs if not math.isclose(p, S.ph, abs_tol=1e-6)]
    grey_for = {p: GREYS[min(i * len(GREYS) // max(len(others), 1), len(GREYS) - 1)] for i, p in enumerate(others)}
    for ph in phs:
        sub = df[df["pH"] == ph]
        current = math.isclose(ph, S.ph, abs_tol=1e-6)
        color = ACCENT if current else grey_for[ph]
        fig.add_trace(go.Scatter(
            x=sub[xcol], y=sub[ycol], mode="lines", name=f"pH {ph:g}" + (" (current)" if current else ""),
            line=dict(color=color, width=3 if current else 2),
            hovertemplate=f"pH {ph:g}<br>{xcol} %{{x:.3g}}<br>{ycol} %{{y:{yfmt}}}<extra></extra>"))
        fig.add_annotation(x=math.log10(float(sub[xcol].iloc[-1])), y=float(sub[ycol].iloc[-1]), text=f"pH {ph:g}",
                           showarrow=False, xanchor="left", xshift=4, font=dict(size=11, color=color))
    fig.add_vline(x=current_x, line=dict(color=GREYS[2], width=1, dash="dot"))
    xmin, xmax = float(df[xcol].min()), float(df[xcol].max())
    ticks = [t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if xmin * 0.999 <= t <= xmax * 1.001]
    fig.update_xaxes(type="log", tickvals=ticks, ticktext=[f"{t:g}" for t in ticks])
    style(fig, xlabel, ylabel, height=380)
    fig.update_layout(margin=dict(r=60))
    plot(fig, key)


def tab_maps() -> None:
    st.caption("Every point uses the full model with the current switches; only the plotted variable and pH change.")
    gap_df, pi_df = grid_maps(settings_key(S))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ΔA₃₄₀ vs GAP**" + (" (entered DL-GAP)" if S.dl_racemate else ""))
        _ph_lines(gap_df, "GAP (mM)", "ΔA₃₄₀", "GAP (mM, log scale)", "ΔA₃₄₀", S.gap if S.gap > 0 else 0.5,
                  "map_gap", ".3g")
        download_csv(gap_df, "gapdh_map_gap.csv", key="dl_map_gap")
    with c2:
        st.markdown("**% GAP oxidised vs Pi**")
        _ph_lines(pi_df, "Pi (mM)", "% GAP oxidised", "Pi (mM, log scale)", "GAP oxidised (%)",
                  S.pi if S.pi > 0 else 10, "map_pi", ".3g")
        download_csv(pi_df, "gapdh_map_pi.csv", key="dl_map_pi")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**% GAP oxidised over pH × Pi**")
        phs, pis, z = grid_heatmap(settings_key(S))
        ly = np.log10(pis)
        fig = go.Figure(go.Heatmap(
            x=phs, y=ly, z=z, colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
            colorbar=dict(title=dict(text="% oxidised"), thickness=12),
            customdata=np.repeat(pis[:, None], len(phs), axis=1),
            hovertemplate="pH %{x:.2f}<br>Pi %{customdata:.3g} mM<br>%{z:.3g} % oxidised<extra></extra>"))
        if S.pi > 0:
            fig.add_trace(go.Scatter(x=[S.ph], y=[math.log10(S.pi)], mode="markers",
                                     marker=dict(size=12, color="white", line=dict(color="black", width=2)),
                                     name="current", hovertemplate="current<extra></extra>"))
        ticks = [1, 2, 5, 10, 20, 50, 100]
        fig.update_yaxes(tickvals=[math.log10(t) for t in ticks], ticktext=[str(t) for t in ticks])
        style(fig, "pH", "Pi (mM, log scale)", height=400, legend=False)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        plot(fig, "heatmap")
        hm = pd.DataFrame(z, columns=[f"pH {p:g}" for p in phs])
        hm.insert(0, "Pi (mM)", pis)
        download_csv(hm, "gapdh_heatmap.csv", key="dl_heat")
    with c2:
        st.markdown("**What a two-fold change buys you**")
        rows = cached_sensitivity(settings_key(S))
        tf = pd.DataFrame([{"Change": r["change"], "Step": r["step"], "ξ ratio (up)": r["up"],
                            "ξ ratio (down)": r["down"]} for r in rows])
        show_table(tf)
        st.caption("Ratio of the new plateau to the current one. pH changes by ±0.5 instead of ×2. "
                   "Recomputed live for the current inputs.")
        download_csv(tf, "gapdh_two_fold.csv", key="dl_twofold")


# -----------------------------------------------------------------------------
# 5. Ionic strength
# -----------------------------------------------------------------------------
def tab_ionic() -> None:
    st.warning(
        "**Approximate.** Extended Debye–Hückel strongly underestimates the non-ideality of GAPDH reaction "
        "mixtures (Greinert et al. 2021). Use these numbers for the direction and rough size of an effect, not "
        f"for quantitative K′ values. The textbook K′ does not state its ionic strength: this app assumes "
        f"I_ref = {sig(S.i_ref)} M (sidebar)."
    )
    i_now = P.ionic_strength_M
    f = pi_dibasic_fraction(S.ph)
    br = pd.DataFrame(ionic_strength_breakdown(S.mixture(), S.gap, S.edta, S.kcl, S.ph))
    br = br.rename(columns={"ion": "Ion", "conc_mM": "c (mM)", "charge": "z", "half_cz2_mM": "½·c·z² (mM)"})
    br["Share (%)"] = 100 * br["½·c·z² (mM)"] / br["½·c·z² (mM)"].sum() if br["½·c·z² (mM)"].sum() > 0 else 0.0

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Contributions to ionic strength** — I = **{sig(i_now)} M**")
        show_table(br)
        st.caption("Counter-ions: Na⁺ for every anion except Cl⁻ (K⁺). GAP counts both isomers when DL. "
                   "EDTA charge ≈ −3.5 is the pH 8.5 value and is held fixed. Computed for the starting mixture.")
        download_csv(br, "gapdh_ionic_strength.csv", key="dl_ion")
    with c2:
        st.markdown("**Activity coefficients**")
        gt = pd.DataFrame([
            {"Species": "NAD⁺", "z": -1, f"γ at I = {sig(i_now)} M": gamma(1, i_now), "γ at I_ref": gamma(1, S.i_ref)},
            {"Species": "NADH", "z": -2, f"γ at I = {sig(i_now)} M": gamma(2, i_now), "γ at I_ref": gamma(2, S.i_ref)},
            {"Species": "GAP", "z": -2, f"γ at I = {sig(i_now)} M": gamma(2, i_now), "γ at I_ref": gamma(2, S.i_ref)},
            {"Species": "1,3-BPG", "z": -4, f"γ at I = {sig(i_now)} M": gamma(4, i_now), "γ at I_ref": gamma(4, S.i_ref)},
            {"Species": f"Pi ({100 * f:.0f} % HPO₄²⁻)", "z": -(1 + f), f"γ at I = {sig(i_now)} M": gamma_pi(i_now, f),
             "γ at I_ref": gamma_pi(S.i_ref, f)},
        ])
        show_table(gt)
        st.markdown(f"Kγ(I) = **{sig(k_gamma(i_now, f))}**, Kγ(I_ref) = **{sig(k_gamma(S.i_ref, f))}** → K′ factor "
                    f"Kγ(I_ref)/Kγ(I) = **{sig(P.activity_factor_estimate)}** "
                    + ("(applied)" if S.ionic_on else "(not applied: correction off)"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Kγ vs ionic strength**")
        grid = np.geomspace(1e-3, 0.5, 120)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=[k_gamma(i, f) for i in grid], mode="lines",
                                 line=dict(color=ACCENT, width=2), name="Kγ",
                                 hovertemplate="I %{x:.3g} M<br>Kγ %{y:.3g}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[max(i_now, 1e-3)], y=[k_gamma(i_now, f)], mode="markers+text", text=["mixture"],
                                 textposition="top right", marker=dict(size=10, color=ACCENT), showlegend=False,
                                 hovertemplate="mixture<br>Kγ %{y:.3g}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[max(S.i_ref, 1e-3)], y=[k_gamma(S.i_ref, f)], mode="markers+text",
                                 text=["I_ref"], textposition="bottom left",
                                 marker=dict(size=10, color=GREYS[3], symbol="diamond"), showlegend=False,
                                 hovertemplate="I_ref<br>Kγ %{y:.3g}<extra></extra>"))
        fig.update_xaxes(type="log", tickvals=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                         ticktext=["0.001", "0.003", "0.01", "0.03", "0.1", "0.3"])
        fig.update_yaxes(type="log", dtick=1, exponentformat="power")
        style(fig, "Ionic strength I (M, log scale)", "Kγ (log scale)", height=360, legend=False)
        plot(fig, "kgamma")
    with c2:
        st.markdown("**Plateau vs KCl**")
        anchor = st.radio("Reference K′ to", [f"I_ref from the sidebar ({sig(S.i_ref)} M)",
                                              "The current mixture (relative change only)"],
                          key="ion_anchor", horizontal=True,
                          help="The first option is the sidebar correction. The second treats the current mixture "
                               "as the reference, so only the change in ionic strength acts.")
        kcl = st.slider("What-if KCl (mM)", 0.0, 250.0, value=float(min(S.kcl, 250.0)), step=5.0, key="ion_kcl")
        relative = anchor.startswith("The current")

        def dh_pred(k):
            return kcl_anchored_prediction(S, k) if relative else predict(replace(S, kcl=k, ionic_on=True))

        ks = np.arange(0, 255, 5.0)
        ideal = [predict(replace(S, kcl=float(k), ionic_on=False)).delta_a for k in ks]
        dh = [dh_pred(float(k)).delta_a for k in ks]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=ideal, mode="lines", line=dict(color=GREYS[2], width=2),
                                 name="Ideal (no correction)", hovertemplate="KCl %{x:g} mM<br>ΔA %{y:.3g}<extra></extra>"))
        fig.add_trace(go.Scatter(x=ks, y=dh, mode="lines", line=dict(color=ACCENT, width=2),
                                 name="Debye–Hückel estimate", hovertemplate="KCl %{x:g} mM<br>ΔA %{y:.3g}<extra></extra>"))
        sel = dh_pred(kcl)
        fig.add_trace(go.Scatter(x=[kcl], y=[sel.delta_a], mode="markers", marker=dict(size=10, color=ACCENT),
                                 showlegend=False, hovertemplate="selected<br>ΔA %{y:.3g}<extra></extra>"))
        style(fig, "KCl (mM)", "Plateau ΔA₃₄₀", height=320)
        plot(fig, "plateau_vs_kcl")
        ideal_sel = predict(replace(S, kcl=kcl, ionic_on=False))
        st.markdown(f"At {kcl:g} mM KCl: I = {sig(sel.ionic_strength_M)} M, K′ factor {sig(sel.activity_factor)}, "
                    f"plateau ξ = **{sig(sel.xi)} mM** (ΔA₃₄₀ **{sig(sel.delta_a)}**) vs ideal "
                    f"{sig(ideal_sel.xi)} mM (ΔA₃₄₀ {sig(ideal_sel.delta_a)}).")


# -----------------------------------------------------------------------------
# 6. Time course
# -----------------------------------------------------------------------------
def tab_time() -> None:
    with st.expander("Kinetic parameters", expanded=False, icon=":material/tune:"):
        c1, c2, c3 = st.columns(3)
        c1.number_input("V (mM/min)", min_value=1e-6, value=C.V_MAX_MM_MIN, step=0.01, format="%.4f", key="tc_v",
                        help="Maximal rate at this enzyme amount. Default 0.05 mM/min (spec §3.7).")
        c2.number_input("Km GAP (mM)", min_value=1e-6, value=C.KM_GAP_MM, step=0.01, format="%.4f", key="tc_kmg",
                        help="Default 0.1 mM (spec §3.7).")
        c3.number_input("Time span (min)", min_value=0.5, max_value=240.0, value=C.T_END_MIN, step=1.0,
                        format="%.1f", key="tc_tend", help="Default 10 min (spec §3.7).")
        c1, c2, c3 = st.columns(3)
        c1.toggle("Competitive NADH inhibition", value=False, key="tc_inh",
                  help="S_NAD = [D/(Km_NAD(1+N/Ki)+D)] / [D₀/(Km_NAD+D₀)] (spec §3.7). Default off.")
        c2.number_input("Km NAD⁺ (mM)", min_value=1e-6, value=C.KM_NAD_MM, step=0.01, format="%.4f", key="tc_kmn",
                        help="Used only with inhibition on. Default 0.05 mM (spec §3.7).")
        c3.number_input("Ki NADH (mM)", min_value=1e-6, value=C.KI_NADH_MM, step=0.005, format="%.4f", key="tc_ki",
                        help="Used only with inhibition on. Default 0.01 mM (spec §3.7).")
    kp = kinetic_params()
    t_end = st.session_state.get("tc_tend", C.T_END_MIN)
    if S.mixture().gap <= 0 or S.nad <= 0:
        st.info("The simulator models the forward reaction and needs GAP and NAD⁺. Use the Reverse & spikes tab "
                "for the reverse direction.")
        return
    tcs = time_courses(S, kp, t_end)
    fig = go.Figure()
    color_for = {"current": ACCENT}
    greys = iter([GREYS[2], GREYS[3]])
    for key in tcs:
        if key != "current":
            color_for[key] = next(greys)
    cur = tcs["current"]
    if cur.get("window_end") is not None and cur["window_end"] > 0:
        fig.add_vrect(x0=0, x1=cur["window_end"], fillcolor=ACCENT, opacity=0.08, line_width=0,
                      annotation_text="initial-rate window (Γ/K′ < 0.05)", annotation_position="top left",
                      annotation_font_size=11)
    for key, sim in tcs.items():
        color = color_for[key]
        fig.add_trace(go.Scatter(x=sim["t"], y=sim["dA"], mode="lines", name=sim["label"],
                                 line=dict(color=color, width=3 if key == "current" else 2),
                                 hovertemplate=sim["label"] + "<br>t %{x:.2f} min<br>ΔA %{y:.3g}<extra></extra>"))
        fig.add_hline(y=sim["equilibrium_dA"], line=dict(color=color, width=1.5, dash="dash"),
                      annotation_text=f"{'completion' if 'Arsenate' in sim['label'] else 'equilibrium'} "
                                      f"{sig(sim['equilibrium_dA'])}",
                      annotation_position="bottom right", annotation_font_size=11, annotation_font_color=color)
    style(fig, "Time (min)", "ΔA₃₄₀ (from t = 0)", height=430)
    plot(fig, "time_course")

    tpoints = [t for t in (1, 2, 5, 10) if t <= t_end] + ([t_end] if t_end not in (1, 2, 5, 10) else [])
    tab = pd.DataFrame({"t (min)": tpoints})
    for sim in tcs.values():
        tab[sim["label"]] = [float(np.interp(t, sim["t"], sim["dA"])) for t in tpoints]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**ΔA₃₄₀ at selected times**")
        show_table(tab)
        long = pd.DataFrame({"time_min": cur["t"]})
        for sim in tcs.values():
            long[sim["label"]] = sim["dA"]
        download_csv(long, "gapdh_time_course.csv", key="dl_tc")
    with c2:
        st.markdown(
            "**How to use the curves.** For kinetic constants (V, Km) use arsenate, or only the first ~5 % of "
            "conversion with phosphate, where the brake (1 − Γ/K′) is still close to 1. For thermodynamics (K′) "
            "use phosphate plateaus: the plateau height does not depend on how much enzyme you add."
        )
        if cur.get("window_end") is not None:
            st.caption(f"With phosphate at pH {sig(S.ph, 3)}, Γ/K′ reaches 0.05 after "
                       f"{sig(cur['window_end'])} min.")


# -----------------------------------------------------------------------------
# 7. Reverse & spikes
# -----------------------------------------------------------------------------
def tab_reverse() -> None:
    st.subheader("Reverse-direction calculator")
    st.caption("1,3-BPG + NADH + H⁺ → GAP + NAD⁺ + Pi, as in PGK-coupled assays. Uses the sidebar temperature, "
               "constants and switches.")
    rv = values_for("Reverse, PGK-coupled style")
    c = st.columns(6)
    bpg = c[0].number_input("1,3-BPG (mM)", min_value=0.0, value=rv["bpg"], step=0.01, format="%.3f", key="rev_bpg",
                            help="Default 0.1 mM (reverse preset, spec §2).")
    nadh = c[1].number_input("NADH (mM)", min_value=0.0, value=rv["nadh"], step=0.01, format="%.3f", key="rev_nadh",
                             help="Default 0.2 mM (reverse preset).")
    pi = c[2].number_input("Pi (mM)", min_value=0.0, value=rv["pi"], step=1.0, format="%.2f", key="rev_pi",
                           help="Default 10 mM (reverse preset).")
    gap = c[3].number_input("GAP (mM)", min_value=0.0, value=0.0, step=0.01, format="%.3f", key="rev_gap",
                            help="Default 0 (reverse preset).")
    nad = c[4].number_input("NAD⁺ (mM)", min_value=0.0, value=0.0, step=0.01, format="%.3f", key="rev_nad",
                            help="Default 0 (reverse preset).")
    ph = c[5].number_input("pH", min_value=C.PH_RANGE[0], max_value=C.PH_RANGE[1], value=rv["ph"], step=0.1,
                           format="%.2f", key="rev_ph", help="Default 7.6 (reverse preset).")
    sr = replace(S, bpg=bpg, nadh=nadh, pi=pi, gap=gap, nad=nad, ph=ph, dl_racemate=False, arsenate=False)
    pr = predict(sr)
    m = st.columns(4)
    m[0].metric("Extent ξ (mM)", sig(pr.xi), help="Negative = reverse.")
    m[1].metric("% BPG reduced", f"{sig(pr.pct_bpg_reduced)} %" if math.isfinite(pr.pct_bpg_reduced) else "—")
    m[2].metric("ΔA₃₄₀", sig(pr.delta_a), help="Negative: NADH is consumed.")
    m[3].metric("K′ (M⁻¹)", sig(pr.k_start))
    st.caption(f"NADH remaining at equilibrium: {sig(pr.nadh_eq)} mM.")

    st.divider()
    st.subheader("Perturb the plateau")
    if P.arsenate:
        st.info("Arsenate mode has no plateau to perturb; switch arsenate off in the sidebar.")
        return
    if P.xi <= 0:
        st.info("This panel needs a forward plateau (GAP, NAD⁺ and Pi in the sidebar).")
        return
    c1, c2 = st.columns([2, 3])
    with c1:
        addition = st.radio("Addition at the plateau", ADDITIONS, key="pert_add",
                            help="What you add once the forward reaction has stopped.")
        amount = 0.0
        if addition in DEFAULT_ADDITION_AMOUNTS:
            amount = st.number_input("Amount (mM)", min_value=0.0, value=DEFAULT_ADDITION_AMOUNTS[addition],
                                     step=0.01 if addition != "+ Pi" else 5.0, format="%.3f",
                                     key=f"pert_amt_{addition}",
                                     help="Defaults: +40 mM Pi, +0.05 mM NADH, +0.5 mM GAP (spec §5).")
        r = perturbation(S, addition, amount)
        st.metric("Instantaneous jump (ΔA)", signed(r["jump"]))
        st.metric("Relaxation (ΔA)", signed(r["relaxation"]))
        st.metric("New plateau A₃₄₀ (from t = 0)", sig(r["a_after"]))
        st.caption(r["note"])
        if r["a_after"] > C.A340_LINEAR_LIMIT:
            st.warning("The new plateau is above 1.2 A: check photometric linearity.")
    with c2:
        tr = perturbation_trace(S, addition, amount, kinetic_params())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr["t"], y=tr["dA"], mode="lines", line=dict(color=ACCENT, width=2),
                                 name="Simulated A₃₄₀", hovertemplate="t %{x:.2f} min<br>ΔA %{y:.3g}<extra></extra>"))
        fig.add_vline(x=tr["t_add"], line=dict(color=GREYS[2], width=1, dash="dot"), annotation_text=addition,
                      annotation_position="top left")
        fig.add_hline(y=r["a_after"], line=dict(color=ACCENT, width=1.5, dash="dash"),
                      annotation_text=f"predicted {sig(r['a_after'])}", annotation_position="bottom right")
        fig.add_hline(y=r["a_before"], line=dict(color=GREYS[2], width=1.5, dash="dash"))
        style(fig, "Time (min)", "ΔA₃₄₀ (from t = 0)", height=380, legend=False)
        plot(fig, "perturbation")
        st.caption("Trace from the time-course simulator (its kinetic parameters apply); the dashed lines are the "
                   "thermodynamic predictions, which do not depend on kinetics.")

    st.markdown("**All additions for the current settings**")
    rows = []
    for add in ADDITIONS:
        amt = DEFAULT_ADDITION_AMOUNTS.get(add, 0.0)
        rr = perturbation(S, add, amt)
        rows.append({"Addition": add if add not in DEFAULT_ADDITION_AMOUNTS else f"{add} {amt:g} mM",
                     "Jump (ΔA)": rr["jump"], "Relaxation (ΔA)": rr["relaxation"], "Net change (ΔA)": rr["net"],
                     "New plateau A₃₄₀": rr["a_after"]})
    pt = pd.DataFrame(rows)
    show_table(pt)
    download_csv(pt, "gapdh_perturbations.csv", key="dl_pert")


# -----------------------------------------------------------------------------
# 8. From data to K′
# -----------------------------------------------------------------------------
def _example_csv() -> bytes:
    sim = simulate(S.mixture(), k_function(replace(S, arsenate=False)), kinetic_params(), False, 10.0,
                   S.epsilon, S.path_cm, n_points=121)
    rng = np.random.default_rng(7)
    a = 0.05 + sim["dA"] + rng.normal(0, 0.003, sim["t"].size)
    return pd.DataFrame({"time_min": np.round(sim["t"], 3), "A340": np.round(a, 4)}).to_csv(index=False).encode()


def tab_inverse() -> None:
    m0 = S.mixture()
    st.caption("Uses the starting concentrations, ε and path from the sidebar. N = ΔA₃₄₀/(ε·l) is the NADH formed; "
               "K′ = 1000·(B₀+N)(N₀+N)/((G₀−N)(D₀−N)(P₀−N)).")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Plateau → K′")
        da = st.number_input("Measured plateau ΔA₃₄₀", min_value=0.0, value=round(max(P.delta_a, 0.0), 3) or 0.4,
                             step=0.01, format="%.3f", key="inv_da", help="Change in A₃₄₀ from t = 0 to the plateau.")
        sd = st.number_input("Uncertainty of ΔA₃₄₀ (±)", min_value=0.0, value=0.01, step=0.005, format="%.3f",
                             key="inv_sd", help="One standard deviation; propagated by finite difference.")
        res = k_from_plateau_with_error(m0, da, sd, S.epsilon, S.path_cm)
        if not math.isfinite(res["k"]) or res["k"] <= 0:
            st.warning("This ΔA implies more NADH than the substrates allow (or none at all); check the inputs.")
        else:
            st.metric("K′ (M⁻¹)", f"{sig(res['k'])} ± {sig(res['sd_k'], 2)}")
            st.metric("ΔG′° (kJ/mol)", sig(delta_g0(res["k"], T_K)))
            st.metric("Local exponent d ln K′ / d ln N", sig(res["exponent"]),
                      help="≈ 2 at low conversion: K′ scales roughly with N², so a 5 % error in ΔA becomes ~10 % in K′.")
            rel_a = sd / da if da > 0 else math.nan
            rel_k = res["sd_k"] / res["k"] if math.isfinite(res["sd_k"]) else math.nan
            st.caption(f"Relative error: {sig(100 * rel_a, 2)} % in ΔA → {sig(100 * rel_k, 2)} % in K′. "
                       f"Model K′ for these conditions: {sig(P.k_start)} M⁻¹.")
    with c2:
        limit = 0.98 * m0.forward_limit * S.epsilon * S.path_cm
        limit = min(limit, max(2.0 * max(P.delta_a, 0.0), 1.5 * da, 0.2))
        grid = np.linspace(max(limit / 25, 1e-4), limit, 200) if limit > 0 else np.array([])
        if grid.size:
            ks = [k_from_plateau(m0, x, S.epsilon, S.path_cm) for x in grid]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grid, y=ks, mode="lines", line=dict(color=ACCENT, width=2), name="K′(ΔA)",
                                     hovertemplate="ΔA %{x:.3g}<br>K′ %{y:.3g} M⁻¹<extra></extra>"))
            n0 = grid[0] / (S.epsilon * S.path_cm)
            k0 = k_from_plateau(m0, grid[0], S.epsilon, S.path_cm)
            guide = [k0 * ((x / (S.epsilon * S.path_cm)) / n0) ** 2 for x in grid]
            fig.add_trace(go.Scatter(x=grid, y=guide, mode="lines", line=dict(color=GREYS[2], width=1.5, dash="dot"),
                                     name="∝ N² guide", hoverinfo="skip"))
            if math.isfinite(res["k"]) and res["k"] > 0:
                fig.add_trace(go.Scatter(x=[da], y=[res["k"]], mode="markers", marker=dict(size=10, color=ACCENT),
                                         name="measured", error_y=dict(type="data", array=[res["sd_k"]] if
                                                                       math.isfinite(res["sd_k"]) else None),
                                         hovertemplate="measured<br>K′ %{y:.3g} M⁻¹<extra></extra>"))
            fig.add_hline(y=P.k_start, line=dict(color=GREYS[3], width=1, dash="dash"),
                          annotation_text=f"model K′ {sig(P.k_start)}", annotation_position="top left")
            fig.update_yaxes(type="log", dtick=1, exponentformat="power")
            style(fig, "Plateau ΔA₃₄₀", "K′ (M⁻¹, log scale)", height=380)
            plot(fig, "inverse")

    st.divider()
    st.subheader("Upload a trace")
    c1, c2 = st.columns([2, 3])
    with c1:
        up = st.file_uploader("CSV with columns time_min, A340", type=["csv", "txt"], key="inv_upload",
                              help="Comma-separated. Header row with time_min and A340.")
        st.download_button("Download an example CSV", _example_csv(), file_name="gapdh_example_trace.csv",
                           mime="text/csv", key="dl_example", icon=":material/download:",
                           help="A simulated phosphate trace for the current settings, with noise and a 0.05 baseline.")
    if up is None:
        with c2:
            st.caption("Upload a trace to estimate K′ from its plateau. The plateau is the mean of the last N points; "
                       "its standard deviation is used as the uncertainty.")
        return
    try:
        raw = pd.read_csv(io.BytesIO(up.getvalue()), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the file: {exc}")
        return
    cols = {c.lower().strip(): c for c in raw.columns}
    if "time_min" in cols and "a340" in cols:
        data = raw[[cols["time_min"], cols["a340"]]].copy()
    else:
        num = raw.select_dtypes("number")
        if num.shape[1] < 2:
            st.error("The file needs two numeric columns: time_min and A340.")
            return
        data = num.iloc[:, :2].copy()
        st.warning(f"Columns time_min / A340 not found; using “{num.columns[0]}” and “{num.columns[1]}”.")
    data.columns = ["time_min", "A340"]
    data = data.apply(pd.to_numeric, errors="coerce").dropna().sort_values("time_min")
    if len(data) < 3:
        st.error("Need at least 3 numeric rows.")
        return
    with c1:
        n_last = st.slider("Plateau = mean of the last N points", 2, int(len(data)), min(10, int(len(data))),
                           key="inv_n")
        base_mode = st.radio("Baseline", ["First point", "None (values are already ΔA)"], key="inv_base",
                             horizontal=True)
    tail = data["A340"].iloc[-n_last:]
    baseline = float(data["A340"].iloc[0]) if base_mode == "First point" else 0.0
    plateau = float(tail.mean())
    sd_tail = float(tail.std(ddof=1)) if n_last > 1 else 0.0
    da_meas = plateau - baseline
    res = k_from_plateau_with_error(m0, da_meas, sd_tail, S.epsilon, S.path_cm)
    with c1:
        st.metric("Plateau ΔA₃₄₀", f"{sig(da_meas)} ± {sig(sd_tail, 2)}")
        if math.isfinite(res["k"]) and res["k"] > 0:
            st.metric("K′ (M⁻¹)", f"{sig(res['k'])} ± {sig(res['sd_k'], 2)}")
            st.caption(f"ΔG′° = {sig(delta_g0(res['k'], T_K))} kJ/mol. Model: K′ = {sig(P.k_start)} M⁻¹, "
                       f"ΔA₃₄₀ = {sig(P.delta_a)}.")
        else:
            st.warning("This plateau is not consistent with the starting concentrations in the sidebar.")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["time_min"], y=data["A340"], mode="markers", marker=dict(size=6, color=GREYS[3]),
                                 name="data", hovertemplate="t %{x:.3g} min<br>A %{y:.4g}<extra></extra>"))
        fig.add_trace(go.Scatter(x=data["time_min"].iloc[-n_last:], y=tail, mode="markers",
                                 marker=dict(size=8, color=ACCENT), name=f"last {n_last} points",
                                 hovertemplate="t %{x:.3g} min<br>A %{y:.4g}<extra></extra>"))
        fig.add_hline(y=plateau, line=dict(color=ACCENT, width=1.5), annotation_text=f"plateau {sig(plateau)}",
                      annotation_position="bottom left")
        fig.add_hline(y=baseline + P.delta_a, line=dict(color=GREYS[2], width=1.5, dash="dash"),
                      annotation_text=f"predicted equilibrium {sig(baseline + P.delta_a)}",
                      annotation_position="top right")
        style(fig, "Time (min)", "A₃₄₀", height=380)
        plot(fig, "uploaded_trace")


# -----------------------------------------------------------------------------
# 9. Experiment planner
# -----------------------------------------------------------------------------
PLAN_NOTES = {
    "A": "Does the plateau rise less than proportionally with GAP, as mass action predicts?",
    "B1": "Pi is a substrate: the plateau must rise with Pi (a pure buffer effect would not).",
    "B2": "Ideal model: no effect. A clear rise points to activity effects (Debye–Hückel estimate shown).",
    "C": "Additions at the plateau test that it is an equilibrium, not enzyme death or substrate decay.",
    "D": "K′ rises ~10-fold per pH unit (one H⁺ released).",
    "E": "Arsenate removes the brake: the trace should keep rising toward completion.",
}


def tab_planner() -> None:
    kp = kinetic_params()
    kp_key = json.dumps({"v_max": kp.v_max, "km_gap": kp.km_gap, "km_nad": kp.km_nad, "ki_nadh": kp.ki_nadh,
                         "inhibition": kp.inhibition})
    plan = cached_plan(settings_key(S), kp_key)
    st.caption("Predicted readouts for the current sidebar settings. Each row changes one thing.")
    for code, note in PLAN_NOTES.items():
        sub = plan[plan["set"].str.startswith(code + ":")]
        if sub.empty:
            continue
        st.markdown(f"**{sub['set'].iloc[0]}** — {note}")
        view = sub.drop(columns=["set"]).rename(columns={
            "condition": "Condition", "xi_mM": "ξ (mM)", "delta_A340": "ΔA₃₄₀", "pct_GAP_oxidised": "% GAP oxidised",
            "note": "Note"})
        show_table(view)
    out = plan.rename(columns={"set": "Set", "condition": "Condition", "xi_mM": "xi_mM", "delta_A340": "dA340",
                               "pct_GAP_oxidised": "pct_GAP_oxidised", "note": "Note"})
    download_csv(out, "gapdh_experiment_plan.csv", "Download the full plan (CSV)", key="dl_plan")


# -----------------------------------------------------------------------------
# 10. Theory & references
# -----------------------------------------------------------------------------
def tab_theory() -> None:
    st.subheader("Reaction and conventions")
    st.latex(r"\text{D-GAP} + \text{NAD}^+ + \text{P}_i \;\rightleftharpoons\; \text{1,3-BPG} + \text{NADH} + \text{H}^+")
    st.latex(r"\Gamma = \frac{[\text{BPG}][\text{NADH}]}{[\text{GAP}][\text{NAD}^+][\text{P}_i]}\ (\text{M}^{-1}),"
             r"\qquad \Delta G'^{\circ} = -RT\ln K', \qquad \Delta G' = RT\ln\frac{\Gamma}{K'}")
    st.markdown("Concentrations in mM inside the app; K′ and Γ in M⁻¹ (1 M standard state, [H⁺] folded into K′). "
                "Extent ξ in mM, positive for GAP oxidation. Readout: NADH at 340 nm, ΔA₃₄₀ = ε·l·ξ.")

    st.subheader("K′ as a function of pH and temperature")
    st.latex(r"K'(\text{pH},T) = K'_{\text{ref}}\;10^{\,s\,(\text{pH}-\text{pH}_{\text{ref}})}\;"
             r"\exp\!\left[-\frac{\Delta H'^{\circ}}{R}\left(\frac{1}{T}-\frac{1}{T_{\text{ref}}}\right)\right]")

    st.subheader("Equilibrium")
    st.latex(r"(B_0+\xi)(N_0+\xi) = k\,(G_0-\xi)(D_0-\xi)(P_0-\xi),\qquad k = K'/1000\ (\text{mM}^{-1})")
    st.markdown("Solved with Brent's method on −min(B₀, N₀) < ξ < min(G₀, D₀, P₀), where the root is unique. "
                "Teaching shortcuts with κ = k·P₀:")
    st.latex(r"(1-\kappa)\,\xi^2 + \left[B_0+N_0+\kappa(G_0+D_0)\right]\xi + (B_0N_0-\kappa G_0D_0) = 0,"
             r"\qquad \xi \approx \sqrt{\kappa\,G_0D_0}")

    st.subheader("The thermodynamic brake")
    st.latex(r"\frac{J_+}{J_-} = \frac{K'}{\Gamma} = e^{-\Delta G'/RT},\qquad "
             r"\frac{d\xi}{dt} = V\,\frac{G}{K_m+G}\,S_{\text{NAD}}\left(1-\frac{\Gamma}{K'}\right)")
    st.latex(r"S_{\text{NAD}} = \frac{D/\left(K_{m,\text{NAD}}(1+N/K_i)+D\right)}{D_0/\left(K_{m,\text{NAD}}+D_0\right)}")
    st.markdown("Arsenate drops the (1 − Γ/K′) term: 1-arseno-3-phosphoglycerate hydrolyses, so the reaction "
                "is irreversible. Enzyme amount (V) changes the rate only, never the plateau.")

    st.subheader("Buffer drift")
    st.latex(r"\beta = \sum_i 2.303\,C_i\,\alpha_i(1-\alpha_i),\quad \alpha_i = \frac{1}{1+10^{\,\text{p}K_{a,i}-\text{pH}}},"
             r"\quad \text{pH}(\xi) = \text{pH}_0 - \xi/\beta")

    st.subheader("Ionic strength (approximate)")
    st.latex(r"I = \tfrac12\sum_i c_i z_i^2,\qquad \ln\gamma = -\frac{A z^2\sqrt I}{1+B\sqrt I}\ \ (A=1.1758,\ B=1.6)")
    st.latex(r"K_\gamma = \frac{\gamma_{\text{BPG}}\,\gamma_{\text{NADH}}}{\gamma_{\text{GAP}}\,\gamma_{\text{NAD}^+}\,"
             r"\gamma_{\text{P}_i}},\qquad K'(I) = K'(I_{\text{ref}})\,\frac{K_\gamma(I_{\text{ref}})}{K_\gamma(I)}")
    st.markdown("γ of total Pi is speciation-weighted: ln γ_Pi = (1 − f)·ln γ(z = 1) + f·ln γ(z = 2), with f the "
                "HPO₄²⁻ fraction (pKa₂ 6.8).")

    st.subheader("Inverse: K′ from a plateau")
    st.latex(r"N = \frac{\Delta A_{340}}{\varepsilon\,l},\qquad K' = 1000\,\frac{(B_0+N)(N_0+N)}{(G_0-N)(D_0-N)(P_0-N)}")

    st.subheader("Assumptions")
    st.markdown(
        f"""
1. One H⁺ is released per turnover.
2. Concentrations are used as activities unless the ionic-strength correction is on.
3. The enzyme binds a negligible share of the reactants.
4. No Mg²⁺ complexation.
5. No non-enzymatic decay of GAP or 1,3-BPG in the equilibrium solver (GAP t½ ≈ {C.GAP_HALF_LIFE_37C_MIN:g} min
   at 37 °C, Phillips & Thornalley 1993; 1,3-BPG also hydrolyses), so slow drifts after the plateau are not modelled.
6. The reference ionic strength of the textbook K′ is not stated in its source; the app assumes {C.I_REF_M:g} M by
   default (currently {sig(S.i_ref)} M, sidebar).
7. Buffer capacity is evaluated at the starting pH (constant β); EDTA charge is fixed at −3.5.
"""
    )

    st.subheader("Gibbs energy values")
    gv = pd.DataFrame([
        {"Quantity": "ΔG′° at pH 7.0, 25 °C (textbook)", "Value": f"+{C.DG0_TEXTBOOK_KJ} kJ/mol",
         "Note": f"K′ = {C.K_PRIME_REF} M⁻¹; the app's default"},
        {"Quantity": "ΔG′° at pH 8.5, 25 °C", "Value": f"{sig(delta_g0(k_at_ph_function(replace(S, ionic_on=False, temp_c=25.0))(8.5)))} kJ/mol"
         .replace("-", "−"), "Note": "From the pH model (current K′ref and slope)"},
        {"Quantity": "ΔRg′⁰ (Greinert et al. 2020, activity-based)",
         "Value": f"{C.GREINERT_DRG_KJ} ± {C.GREINERT_DRG_SD_KJ} kJ/mol",
         "Note": "Different standard state (H⁺ at unit activity). For information only — never used as ΔG′°"},
        {"Quantity": "ΔRh′⁰ (Greinert et al. 2020, calorimetry)", "Value": f"{C.DELTA_H_KJ} ± {C.DELTA_H_SD_KJ} kJ/mol",
         "Note": "Used for the temperature dependence"},
    ])
    st.dataframe(gv, hide_index=True)

    st.subheader("References")
    for r in C.REFERENCES:
        link = f" [{r['url']}]({r['url']})" if r["url"] else ""
        st.markdown(f"- {r['text']}{link} — *{r['use']}*")
    st.subheader("Companion literature guide")
    if C.COMPANION_GUIDE_URL:
        st.markdown(f"[Open the companion literature guide]({C.COMPANION_GUIDE_URL})")
    else:
        st.info("**Placeholder:** the link to the companion literature guide goes here "
                "(set `COMPANION_GUIDE_URL` in `gapdh/constants.py`).")
    st.caption(f"GAPDH Equilibrium Explorer v{__version__} · {C.APP_URL}")


# -----------------------------------------------------------------------------
# Render tabs
# -----------------------------------------------------------------------------
for tab, fn in zip(TABS, [tab_equilibrium, tab_energy, tab_ph, tab_maps, tab_ionic, tab_time, tab_reverse,
                          tab_inverse, tab_planner, tab_theory]):
    with tab:
        safe(fn)

# Sidebar downloads (inputs + results)
st.sidebar.header("Download")
payload = {
    "app": "GAPDH Equilibrium Explorer",
    "version": __version__,
    "preset": st.session_state.get("preset_choice"),
    "based_on_preset": st.session_state.get("base_preset"),
    "inputs": S.to_dict(),
    "effective_D_GAP_mM": S.gap_d,
    "results": {k: (None if isinstance(v, float) and not math.isfinite(v) else round_sig(v, 6))
                for k, v in P.summary().items() if not isinstance(v, dict)},
    "initial_mM": P.mix0.as_dict(),
    "equilibrium_mM": {k: round_sig(v, 6) for k, v in P.mix_eq.as_dict().items()},
    "warnings": WARNINGS,
}
st.sidebar.download_button("Inputs + results (JSON)", json.dumps(payload, indent=2, ensure_ascii=False),
                           file_name="gapdh_equilibrium.json", mime="application/json",
                           icon=":material/download:", key="dl_json")
st.sidebar.caption("Every table has its own CSV button; charts export as PNG from their toolbar (camera icon).")

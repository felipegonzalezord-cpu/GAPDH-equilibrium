"""Every constant used by the GAPDH Equilibrium Explorer, each with its source.

Concentrations inside the package are in mM. Equilibrium constants are in M⁻¹
(1 M standard state) and are converted to mM⁻¹ only where they are applied.
"""

from __future__ import annotations

# --- Physical constants -------------------------------------------------------
R_KJ: float = 8.314462618e-3  # kJ mol⁻¹ K⁻¹; CODATA 2018 molar gas constant
KELVIN_OFFSET: float = 273.15  # K; definition of the Celsius scale
T_REF_K: float = 298.15  # K; 25 °C, reference temperature of K′ref (spec §3.1)

# --- Reaction thermodynamics (spec §2 and §3.1) --------------------------------
# D-GAP + NAD⁺ + Pi ⇌ 1,3-BPG + NADH + H⁺, apparent constant with [H⁺] folded in.
K_PRIME_REF: float = 0.078  # M⁻¹ at pH 7.0, 25 °C; textbook value (BOKU glycolysis notes), = ΔG′° +6.3 kJ/mol
PH_REF: float = 7.0  # pH at which K′ref applies (textbook convention)
PROTON_SLOPE: float = 1.0  # Δlog K′ per pH unit; one H⁺ released per turnover
DELTA_H_KJ: float = 4.6  # kJ/mol; Greinert et al. 2020, titration calorimetry (ΔRh′⁰ = 4.6 ± 0.1)
DELTA_H_SD_KJ: float = 0.1  # kJ/mol; Greinert et al. 2020
DG0_TEXTBOOK_KJ: float = 6.3  # kJ/mol at pH 7; textbook ΔG′° (BOKU glycolysis notes)
GREINERT_DRG_KJ: float = 51.5  # kJ/mol; Greinert et al. 2020, activity-based ΔRg′⁰, H⁺ at unit activity. DISPLAY ONLY
GREINERT_DRG_SD_KJ: float = 0.4  # kJ/mol; Greinert et al. 2020
I_REF_M: float = 0.25  # M; assumed ionic strength of K′ref. NOT stated by the textbook source (spec §2)

# --- Optics (spec §1) -----------------------------------------------------------
EPSILON_340: float = 6.22  # mM⁻¹ cm⁻¹; NADH molar absorptivity at 340 nm (standard value)
PATH_CM: float = 1.0  # cm; standard cuvette

# --- Default mixture: "Current assay (Felipe, pH 8.5)" (spec §2) ----------------
DEFAULT_PH: float = 8.5
DEFAULT_TEMP_C: float = 25.0
DEFAULT_GAP_MM: float = 0.5
DEFAULT_NAD_MM: float = 1.0
DEFAULT_PI_MM: float = 10.0  # sodium phosphate; reactant and buffer
DEFAULT_BPG_MM: float = 0.0
DEFAULT_NADH_MM: float = 0.0
DEFAULT_EDTA_MM: float = 1.0
DEFAULT_KCL_MM: float = 0.0
DL_D_FRACTION: float = 0.5  # fraction of D-GAP in a DL-racemate

# --- Phosphate acid–base (spec §2, §3.4, §3.5) ----------------------------------
PKA2_PI_BUFFER: float = 7.2  # pKa₂ used for buffer capacity (spec §2 default buffer; near the I → 0 value)
PKA2_PI_SPECIATION: float = 6.8  # pKa₂ used for charge / speciation (spec §3.5; apparent value at I ≈ 0.1 M)

# --- Buffer drift (spec §3.4) --------------------------------------------------
LN10: float = 2.303  # the factor in β = 2.303·C·α(1 − α), as written in the spec
PROTONS_PER_TURNOVER: float = 1.0  # one H⁺ released per GAP oxidised
EXAMPLE_SECOND_BUFFER = {"name": "TAPS", "conc_mM": 50.0, "pKa": 8.4}  # spec §3.4 example
PH_DRIFT_CLIP = (0.0, 14.0)  # numerical guard on pH(ξ) for very weak buffering

# --- Ionic strength and activity (spec §3.5) ------------------------------------
DH_A: float = 1.1758  # kg^½ mol^-½ at 25 °C, extended Debye–Hückel (Alberty)
DH_B: float = 1.6  # kg^½ mol^-½, extended Debye–Hückel (Alberty)
CHARGE = {  # net charge at the working pH (spec §3.5)
    "NAD+": -1.0,
    "NADH": -2.0,
    "GAP": -2.0,
    "BPG": -4.0,
    "H2PO4-": -1.0,
    "HPO4 2-": -2.0,
    "EDTA": -3.5,  # approximate, at pH 8.5 (spec §3.5); held fixed at other pH
    "Cl-": -1.0,
    "Na+": 1.0,
    "K+": 1.0,
}

# --- Time-course simulator (spec §3.7) -------------------------------------------
V_MAX_MM_MIN: float = 0.05  # mM/min
KM_GAP_MM: float = 0.1  # mM
KM_NAD_MM: float = 0.05  # mM
KI_NADH_MM: float = 0.01  # mM, competitive vs NAD⁺
T_END_MIN: float = 10.0  # min
INITIAL_RATE_GAMMA_FRACTION: float = 0.05  # initial-rate window ends where Γ/K′ reaches this

# --- Energy profile (spec §3.3) --------------------------------------------------
SNAPSHOT_A_MM: float = 0.001  # 1 μM NADH formed
DEFAULT_NADH_SPIKE_MM: float = 0.05  # snapshot d spike

# --- Input validation (spec §6) -------------------------------------------------
PH_RANGE = (4.0, 11.0)
TEMP_RANGE_C = (0.0, 80.0)

# --- Display thresholds (spec §5) ----------------------------------------------
A340_LINEAR_LIMIT: float = 1.2  # above this, check photometric linearity
SMALL_SIGNAL_PERCENT: float = 2.0  # conversion below this: signal too small

# --- Non-enzymatic decay, mentioned in assumptions (spec §6) ---------------------
GAP_HALF_LIFE_37C_MIN: float = 75.0  # min; Phillips & Thornalley 1993

# --- Deployment -------------------------------------------------------------------
APP_URL: str = "https://gapdh-equilibrium.streamlit.app"
COMPANION_GUIDE_URL: str = ""  # TODO: paste the companion literature guide URL here

REFERENCES = [
    {
        "text": "Greinert T, Vogel K, Mühlenweg J-K, Sadowski G, Maskow T, Held C (2020). "
        "Fluid Phase Equilibria 517:112597.",
        "url": "https://doi.org/10.1016/j.fluid.2020.112597",
        "use": "ΔRh′⁰ = 4.6 kJ/mol (calorimetry); activity-based ΔRg′⁰ = 51.5 kJ/mol",
    },
    {
        "text": "Greinert T, Vogel K, Maskow T, Held C (2021). Sci Rep 11:6125.",
        "url": "https://doi.org/10.1038/s41598-021-85594-8",
        "use": "Extended Debye–Hückel strongly underestimates non-ideality",
    },
    {
        "text": "Cornell NW, Leadbetter M, Veech RL (1979). J Biol Chem 254:6522–6527.",
        "url": "https://doi.org/10.1016/S0021-9258(18)50399-9",
        "use": "Equilibrium constants of glycolytic reactions",
    },
    {
        "text": "Veech RL, Lawson JW, Cornell NW, Krebs HA (1979). J Biol Chem 254:6538–6547.",
        "url": "",
        "use": "Cytosolic phosphorylation potential",
    },
    {
        "text": "Trentham DR, McMurray CH, Pogson CI (1969). Biochem J 114:19–24.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/4309306",
        "use": "GAP hydrate : aldehyde = 29 : 1",
    },
    {
        "text": "Moniot S et al. (2008). J Biol Chem 283:21693–21702.",
        "url": "https://doi.org/10.1074/jbc.M802286200",
        "use": "Thioacyl intermediate, PDB 3CMC",
    },
    {
        "text": "Textbook ΔG′° = +6.3 kJ/mol (BOKU lecture notes, Glykolyse 2).",
        "url": "https://boku.ac.at/fileadmin/data/H03000/H77000/LVA/772111_6/05_Glykolyse2.pdf",
        "use": "K′ref = 0.078 M⁻¹ at pH 7",
    },
    {
        "text": "Phillips SA, Thornalley PJ (1993). Methylglyoxal formation from triose phosphates. "
        "Eur J Biochem.",
        "url": "",
        "use": "Non-enzymatic GAP decay, t½ ≈ 75 min at 37 °C",
    },
]

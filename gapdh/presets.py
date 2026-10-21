"""Named presets for the sidebar dropdown (spec §2).

Each preset is a complete set of `AssaySettings` field values, so choosing a
preset or pressing "Reset to preset" restores every input.
"""

from __future__ import annotations

from . import constants as C

DEFAULT_PRESET = "Current assay (Felipe, pH 8.5)"
CUSTOM = "Custom"

_BASE: dict = {
    "ph": C.DEFAULT_PH,
    "temp_c": C.DEFAULT_TEMP_C,
    "gap": C.DEFAULT_GAP_MM,
    "dl_racemate": False,
    "nad": C.DEFAULT_NAD_MM,
    "pi": C.DEFAULT_PI_MM,
    "bpg": C.DEFAULT_BPG_MM,
    "nadh": C.DEFAULT_NADH_MM,
    "edta": C.DEFAULT_EDTA_MM,
    "kcl": C.DEFAULT_KCL_MM,
    "epsilon": C.EPSILON_340,
    "path_cm": C.PATH_CM,
    "k_ref": C.K_PRIME_REF,
    "ph_ref": C.PH_REF,
    "slope": C.PROTON_SLOPE,
    "delta_h": C.DELTA_H_KJ,
    "i_ref": C.I_REF_M,
    "drift_on": False,
    "ionic_on": False,
    "pi_pka_buffer": C.PKA2_PI_BUFFER,
    "buffer2_on": False,
    "buffer2_name": C.EXAMPLE_SECOND_BUFFER["name"],
    "buffer2_conc": C.EXAMPLE_SECOND_BUFFER["conc_mM"],
    "buffer2_pka": C.EXAMPLE_SECOND_BUFFER["pKa"],
    "arsenate": False,
}

PRESETS: dict[str, dict] = {
    DEFAULT_PRESET: {
        "description": "The lab's current forward assay: 0.5 mM GAP, 1 mM NAD⁺, 10 mM sodium phosphate, "
        "1 mM EDTA, no KCl, pH 8.5, 25 °C.",
        "values": dict(_BASE),
    },
    "Textbook pH 7.0": {
        "description": "The same mixture at pH 7.0, where the textbook K′ = 0.078 M⁻¹ applies directly.",
        "values": {**_BASE, "ph": 7.0},
    },
    "Forward with arsenate": {
        "description": "Arsenolysis: the acyl-arsenate product hydrolyses, so the reaction is irreversible "
        "and runs to completion. Used by the time-course tab.",
        "values": {**_BASE, "arsenate": True},
    },
    "Reverse, PGK-coupled style": {
        "description": "Reverse direction: 0.1 mM 1,3-BPG and 0.2 mM NADH with 10 mM Pi, no GAP or NAD⁺, pH 7.6.",
        "values": {**_BASE, "ph": 7.6, "gap": 0.0, "nad": 0.0, "bpg": 0.1, "nadh": 0.2, "pi": 10.0},
    },
}

PRESET_NAMES: list[str] = list(PRESETS) + [CUSTOM]


def values_for(name: str) -> dict:
    """Copy of a preset's input values (the default preset for unknown names)."""
    return dict(PRESETS.get(name, PRESETS[DEFAULT_PRESET])["values"])

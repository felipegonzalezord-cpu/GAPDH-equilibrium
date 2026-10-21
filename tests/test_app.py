"""Smoke tests: the app renders every tab without errors at defaults, for every preset and at extremes."""

from __future__ import annotations

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

from pathlib import Path  # noqa: E402

from gapdh.presets import PRESETS  # noqa: E402

TIMEOUT = 120
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run(setup=None) -> "AppTest":
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    if setup:
        setup(at)
        at.run()
    return at


def _assert_clean(at: "AppTest") -> None:
    assert not at.exception, [e.value for e in at.exception]
    errors = [e.value for e in at.error]
    assert not errors, errors
    assert len(at.tabs) == 10


def _metric(at: "AppTest", label: str) -> str:
    for m in at.metric:
        if m.label == label:
            return m.value
    raise KeyError(label)


def test_defaults_render_and_match_spec():
    at = _run()
    _assert_clean(at)
    assert _metric(at, "% GAP oxidised") == "18.9 %"
    assert _metric(at, "NADH at equilibrium (mM)") == "0.0947"
    assert _metric(at, "ΔA₃₄₀") == "0.589"
    assert _metric(at, "K′ (M⁻¹)") == "2.47"
    assert _metric(at, "ΔG′° (kJ/mol)") == "−2.24"
    assert _metric(at, "κ") == "0.0247"


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_renders(name):
    def setup(at):
        at.selectbox(key="preset_choice").set_value(name)

    at = _run(setup)
    _assert_clean(at)


def test_editing_switches_label_to_custom():
    def setup(at):
        at.number_input(key="in_ph").set_value(9.0)

    at = _run(setup)
    _assert_clean(at)
    assert at.selectbox(key="preset_choice").value == "Custom"
    assert _metric(at, "ΔA₃₄₀") == "0.939"


@pytest.mark.parametrize("changes", [
    {"in_ph": 4.0, "in_temp_c": 0.0},
    {"in_ph": 11.0, "in_temp_c": 80.0, "in_gap": 5.0, "in_nad": 5.0, "in_pi": 200.0},
    {"in_gap": 0.0, "in_nad": 0.0},
    {"in_pi": 0.0},
    {"in_nadh": 0.5},
    {"in_drift_on": True, "in_ionic_on": True, "in_kcl": 1000.0, "in_pi": 0.1},
    {"in_dl_racemate": True, "in_buffer2_on": True, "in_drift_on": True},
    {"in_arsenate": True},
])
def test_extreme_inputs_render(changes):
    def setup(at):
        for key, value in changes.items():
            if isinstance(value, bool):
                at.toggle(key=key).set_value(value)
            else:
                at.number_input(key=key).set_value(value)

    at = _run(setup)
    _assert_clean(at)

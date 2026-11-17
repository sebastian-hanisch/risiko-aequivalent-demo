"""Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben) und Zufalls-Seed-Button."""

import math
import random
from dataclasses import dataclass

import streamlit as st

import rae_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None     # Auswahlregler: erlaubte Werte (Permalink rastet darauf ein)
    step: float | None = None


_B = C._BASE
SETTING_SPECS = {
    "rho_select": SettingSpec("rho", int, _B["rho"], min(C.RHO_OPTIONS), max(C.RHO_OPTIONS), options=C.RHO_OPTIONS),
    "x1_slider": SettingSpec("x1", float, _B["x1"], C.X_MIN, C.X_MAX, step=C.X_STEP),
    "x2_slider": SettingSpec("x2", float, _B["x2"], C.X_MIN, C.X_MAX, step=C.X_STEP),
    "p1_slider": SettingSpec("p1", float, _B["p1"], C.P_MIN, C.P_MAX, step=C.P_STEP),
    "fg_select": SettingSpec("fg", float, _B["fg"], min(C.FRAGE_G), max(C.FRAGE_G), options=C.FRAGE_G),
    "fl_slider": SettingSpec("fl", float, _B["fl"], C.L_MIN, C.L_MAX, step=C.L_STEP),
    "a1_slider": SettingSpec("a1", float, _B["a1"], C.A_MIN, C.A_MAX, step=C.A_STEP),
    "a2_slider": SettingSpec("a2", float, _B["a2"], C.A_MIN, C.A_MAX, step=C.A_STEP),
    "b1_slider": SettingSpec("b1", float, _B["b1"], C.A_MIN, C.A_MAX, step=C.A_STEP),
    "b2_slider": SettingSpec("b2", float, _B["b2"], C.A_MIN, C.A_MAX, step=C.A_STEP),
    "p2_slider": SettingSpec("p2", float, _B["p2"], C.P_MIN, C.P_MAX, step=C.P_STEP),
    "w3_select": SettingSpec("w3", int, _B["w3"], min(C.W3_OPTIONS), max(C.W3_OPTIONS), options=C.W3_OPTIONS),
    "g3_select": SettingSpec("g3", int, _B["g3"], min(C.GAMMA_OPTIONS), max(C.GAMMA_OPTIONS), options=C.GAMMA_OPTIONS),
    "sk4_select": SettingSpec("sk4", float, _B["sk4"], min(C.SCALE_OPTIONS), max(C.SCALE_OPTIONS), options=C.SCALE_OPTIONS),
    "n4_select": SettingSpec("n4", int, _B["n4"], min(C.N4_OPTIONS), max(C.N4_OPTIONS), options=C.N4_OPTIONS),
    "seed4_input": SettingSpec("seed4", int, _B["seed4"], 0, C.SEED_MAX),
    "g5_select": SettingSpec("g5", int, _B["g5"], min(C.GAMMA_OPTIONS), max(C.GAMMA_OPTIONS), options=C.GAMMA_OPTIONS),
    "w5_select": SettingSpec("w5", int, _B["w5"], min(C.W5_OPTIONS), max(C.W5_OPTIONS), options=C.W5_OPTIONS),
}
PRESET_KEYS = {
    "rho": "rho_select", "x1": "x1_slider", "x2": "x2_slider", "p1": "p1_slider", "fg": "fg_select", "fl": "fl_slider",
    "a1": "a1_slider", "a2": "a2_slider", "b1": "b1_slider", "b2": "b2_slider", "p2": "p2_slider",
    "w3": "w3_select", "g3": "g3_select", "sk4": "sk4_select", "n4": "n4_select", "seed4": "seed4_input", "g5": "g5_select", "w5": "w5_select",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _snap(spec, value):
    """Wert auf die erlaubten Stufen einrasten (Auswahlregler) bzw. auf die Reglergrenzen begrenzen und auf die Schrittweite runden."""
    if spec.options is not None:
        return min(spec.options, key=lambda o: abs(o - value))
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step is not None:
        value = round(round((value - spec.lo) / spec.step) * spec.step + spec.lo, 6)
        value = min(spec.hi, max(spec.lo, value))
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf die erlaubten Stufen eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            st.session_state[state_key] = _snap(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def _gleich(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen (Gleitkommawerte auf 1e-9), sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem
    Session State. tests/test_layout.py prüft für jeden Regler, dass seine Änderung die Markierung aufhebt."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            _gleich(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed4_input"] = random.randint(0, C.SEED_MAX)

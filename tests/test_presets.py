"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Einrasten; jedes Preset meldet die erwartete Wirkung (DEMO-PLAYBOOK §11a Punkt 7)."""

import pytest

import rae_constants as C
import rae_evaluation as ev
import rae_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            v = preset[key]
            if spec.options is not None:
                assert v in spec.options, (name, key, v)
            else:
                assert spec.lo <= v <= spec.hi, (name, key, v)
                if spec.step is not None:
                    assert abs((v - spec.lo) / spec.step - round((v - spec.lo) / spec.step)) < 1e-9, (name, key)


def test_default_preset_equals_the_default_settings():
    standard = next(iter(C.PRESETS.values()))
    assert all(standard[key] == P.SETTING_SPECS[state_key].default for key, state_key in P.PRESET_KEYS.items())


def test_presets_are_pairwise_different():
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_snap_rastet_auf_die_stufen_ein():
    assert P._snap(P.SETTING_SPECS["rho_select"], 560) == 556
    assert P._snap(P.SETTING_SPECS["p1_slider"], 0.52) == 0.5 and P._snap(P.SETTING_SPECS["p1_slider"], 7) == 0.95
    assert P._snap(P.SETTING_SPECS["x1_slider"], 204) == 200.0


def _kipp(name):
    p = C.PRESETS[name]
    return ev.kipp_analyse(p["a1"], p["a2"], p["b1"], p["b2"], p["p2"], p["rho"])


def test_jedes_preset_meldet_die_erwartete_verdikt_zeile():
    k = _kipp("Kapazitätsfall, ρ = 200")
    assert k["punkte"][0] == pytest.approx(555.80, abs=0.01) and k["wahl_sae"] == -1                          # bei ρ = 200 wählt der Kapazitätsfall K
    assert "ρ* = 555,80" in ev.verdict_kipp(k)[1]
    k = _kipp("Kipppunkt Fall 1 (ρ = 556)")
    assert k["wahl_sae"] == 1 and k["sae_verlust"] == 0.0                                                      # knapp über ρ*: G wird gewählt
    k = _kipp("Fall 2: weniger Verlust")
    assert k["punkte"][0] == pytest.approx(158.05, abs=0.01) and k["wahl_sae"] == 1 and "ρ* = 158,05" in ev.verdict_kipp(k)[1]
    k = _kipp("Goldener Schnitt")
    assert k["punkte"][0] == pytest.approx(207.81, abs=0.01) and k["wahl_sae"] == -1
    k = _kipp("Zwei Kipppunkte (A gegen B)")
    assert k["punkte"] == pytest.approx([28.87, 72.18], abs=0.01) and k["wahl_sae"] == -1                    # bei ρ = 50 wählt B
    assert "2 Kipppunkte" in ev.verdict_kipp(k)[1] and "im schlechtesten Fall besser" in ev.verdict_kipp(k)[1]


def test_vermoegen_preset_wechselt_von_exponential_k_zu_potenz_g():
    p = C.PRESETS["Vermögen: Potenz γ = 2"]
    v = ev.vermoegen_analyse(p["w3"], p["g3"], p["rho"])
    assert v["wahl_exp"] == "K" and v["wahl_pow"] == "G"


def test_voreinstellung_nutzt_nur_praesets_ohne_wirkungsbehauptung_bei_wirkungsloser_einstellung():
    """Ein Preset ohne Effekt darf keinen Effekt melden: im Kapazitätsfall bei ρ = 5000 gibt es keinen SÄ-Verlust, und das Verdikt nennt keinen."""
    k = ev.kipp_analyse(200.0, -100.0, 30.0, 30.0, 0.5, 5000.0)
    assert k["wahl_sae"] == 1 and k["wahl_ew"] == 1 and k["sae_verlust"] == 0.0 and "SÄ-Verlust" not in ev.verdict_kipp(k)[1]

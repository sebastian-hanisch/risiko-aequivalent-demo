"""Die vorgerechnete Messung (`data/rae_frozen.json`) passt zum Code: ein kleiner Teil wird neu gerechnet und verglichen (`tools/rae_make_frozen.py --check`), dazu Konsistenz
der Datei. Die ganze Messung rechnet nur `python tools/rae_make_frozen.py` neu (rund eine Minute)."""

import importlib.util
from pathlib import Path

import rae_constants as C
import rae_evaluation as ev

ROOT = Path(__file__).resolve().parent.parent


def test_datei_ist_vorhanden_und_nennt_seeds_und_zahl_der_probleme():
    fz = ev.load_frozen()
    assert fz is not None
    kipp = fz["kipp"]
    assert len(kipp) == 12 and {(e["art"], e["m"], e["seed"]) for e in kipp} == {(a, m, s) for a in ("sicher", "riskant") for m in (2, 3, 4) for s in C.SEED_KIPP}
    assert all(e["n"] == C.N_KIPP for e in kipp)
    kal = fz["kalibrierung"]
    assert len(kal) == 18 and all(e["n"] == C.N_FROZEN_CAL for e in kal) and {e["seed"] for e in kal} == {C.DEFAULT_SEED, C.SEED_BAND}


def test_zaehlungen_der_datei_sind_in_sich_stimmig():
    for e in ev.load_frozen()["kipp"]:
        if e["art"] == "sicher":
            assert e["w0"] + e["w1"] + e["w2plus"] == e["n"]
        else:
            assert e["w0"] + e["w1"] + e["w2plus"] == e["nicht_dominiert"] <= e["n"] and e["dominiert_mit_wechsel"] == 0


def test_check_rechnet_einen_teil_neu_und_vergleicht(capsys):
    spec = importlib.util.spec_from_file_location("rae_make_frozen", ROOT / "tools" / "rae_make_frozen.py")
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    assert modul.pruefen() == 0
    assert "GESAMT: 3 Prüfungen, 0 Fehler" in capsys.readouterr().out

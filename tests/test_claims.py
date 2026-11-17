"""Claims-Tests: jede Zahl der Befunde B1 bis B13 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a; die Tabelle „Zahl → Test“ der README nennt je Zahl die Funktion).

Plattform (Lehre 2026-11-15): exp und log kommen aus der Bibliothek des Systems und können sich in der letzten Binärstelle unterscheiden. Darum: Zählungen aus Neu-Rechnung ± 2,
Anteile auf ein halbes Zehntel Prozent (± 0,0006), Gleitkomma relativ 1e-9; exakt nur, was in Brüchen gerechnet ist (`test_oracle_risiko_aequivalent.py`, `test_utility.py`).
Zufall nur aus SplitMix64 (`rae_rng`, ganzzahlig). Zufallsprobleme: Zufalls-Seeds 101 und 20261116, je 200 000 Probleme (Zähleinheit: nicht dominierte Probleme)."""

import math

import numpy as np
import pytest

import rae_constants as C
import rae_evaluation as ev
import rae_scenario as S
import rae_utility as U
import rae_wealth as W

BAND = 6e-4
CAP_X, CAP_P = np.array([200.0, -100.0]), np.array([0.5, 0.5])


@pytest.fixture(scope="module")
def strom101():
    return ev.zufall_strom(101, 200000, 100, 1.0, mit_verhaeltnis=True)


@pytest.fixture(scope="module")
def strom2():
    return ev.zufall_strom(20261116, 200000, 100, 1.0, mit_verhaeltnis=True)


@pytest.fixture(scope="module")
def nae101():
    return ev.naeherung_strom(101, 200000, 100)


@pytest.fixture(scope="module")
def nae2():
    return ev.naeherung_strom(20261116, 200000, 100)


@pytest.fixture(scope="module")
def frozen():
    return ev.load_frozen()


# --- B1: Sicherheitsäquivalent von Hand -----------------------------------------------------------------------------------------------------------

def test_b1_zweipunkt_lotterie_und_skaleninvarianz():
    sk = ev.skalen_beispiel()
    assert round(sk["klein"], 2) == 37.99 and round(sk["gross"], 2) == 75.98 and sk["gross"] == pytest.approx(2 * sk["klein"], abs=1e-12)


# --- B2: Kapazitätsfall bei ρ = 200 -------------------------------------------------------------------------------------------------------------

def test_b2_kapazitaetsfall_rho_200():
    a = ev.lotterie_analyse(200, -100, 0.5, 200)
    assert (a["ew"], round(a["sae"], 2), round(a["praemie"], 2), round(a["praemie_naeherung"], 2), round(a["fehler"], 1)) == (50.0, -1.65, 51.65, 56.25, 8.9)


def test_b2_praemie_und_naeherung_ueber_rho():
    soll = {50.0: (115.47, 94.9), 100.0: (85.54, 31.5), 200.0: (51.65, 8.9), 300.0: (36.03, 4.1), 555.8: (20.0, 1.2), 1000.0: (11.21, 0.4), 5000.0: (2.25, 0.01)}
    for z in ev.sae_tabelle(200, -100, 0.5):
        pr, fe = soll[z["rho"]]
        assert z["praemie"] == pytest.approx(pr, abs=0.006) and z["fehler"] == pytest.approx(fe, abs=0.06), z["rho"]


# --- B3 und B4: ρ*, Vorsprung, dieselbe Zahl andere Wahl --------------------------------------------------------------------------------------

def test_b3_rho_stern_im_kapazitaetsfall_und_den_nachbarfaellen():
    r1, s1 = U.rho_stern_sicher(CAP_X, CAP_P, 30.0)
    r2, _ = U.rho_stern_sicher(np.array([200.0, -50.0]), CAP_P, 30.0)
    rg, _ = U.rho_stern_sicher(CAP_X, CAP_P, 0.0)
    ri, _ = U.rho_stern_sicher(np.array([100.0, -60.0]), np.array([0.45, 0.55]), 5.0)
    assert (round(r1, 2), round(r2, 2), round(rg, 2), round(ri, 2)) == (555.80, 158.05, 207.81, 444.87)
    assert r1 / 300 == pytest.approx(1.853, abs=5e-4) and r2 / 250 == pytest.approx(0.632, abs=5e-4) and ri / 160 == pytest.approx(2.78, abs=5e-3)
    assert rg == pytest.approx(100 / math.log((1 + math.sqrt(5)) / 2), rel=1e-9)


def test_b3_drei_zustaende():
    punkte, _, _ = U.kipp_punkte(np.array([250.0, 60.0, -150.0]), np.array([90.0, 50.0, 0.0]), np.array([0.3, 0.4, 0.3]))
    assert len(punkte) == 1 and round(punkte[0], 2) == 1631.16


def test_b4_dieselbe_zahl_andere_wahl_und_saeverlust():
    wahl = {r: [f["wahl"] for f in ev.faelle_vergleich(r)] for r in (100, 200, 300, 600)}
    assert wahl == {100: ["K", "K"], 200: ["K", "G"], 300: ["K", "G"], 600: ["G", "G"]}
    f1, f2 = ev.faelle_vergleich(200)
    assert round(f1["sae_verlust"], 2) == 31.65 and round(f1["sae"], 2) == -1.65 and f2["sae_verlust"] == 0.0 and round(f2["sae"], 2) == 38.24
    assert (f1["ew"], f2["ew"], f1["vorsprung"], f2["vorsprung"], f1["spanne"], f2["spanne"]) == (50.0, 75.0, 20.0, 45.0, 300.0, 250.0)


def test_b5_rho_stern_ueber_den_vorsprung():
    soll = {10: (65.0, 60.35, 0.287, 40.5), 50: (45.0, 158.05, 0.632, 9.8), 70: (35.0, 248.47, 0.920, 4.8), 100: (20.0, 555.80, 1.853, 1.2), 130: (5.0, 2720.83, 8.2449, 0.1), 139: (0.5, 28730.08, 84.75, 0.0)}
    p = CAP_P
    for L, (v, rs, q, nae) in soll.items():
        r, status = U.rho_stern_sicher(np.array([200.0, -float(L)]), p, 30.0)
        assert status == "ok" and r == pytest.approx(rs, abs=0.006) and r / (200 + L) == pytest.approx(q, abs=1e-3)
        assert 100 * (U.normal_rho_stern(200 + L, v) / r - 1) == pytest.approx(nae, abs=0.06)
    werte = [z["rho_stern"] for z in ev.rho_kurve()]
    assert werte == sorted(werte) and len(werte) == len(set(C.L_KURVE)) == 44           # 44 verschiedene Stützstellen (L = 10, 13, …, 139), ρ* wächst, wenn der Vorsprung schrumpft
    assert 50 not in C.L_KURVE and 130 in C.L_KURVE and C.L_KURVE[-1] == 139             # L = 50 (Vorsprung 45) ist einzeln gerechnet, nicht Teil der Kurve


def test_b5_skaleninvarianz_von_rho_stern():
    r, _ = U.rho_stern_sicher(CAP_X, CAP_P, 30.0)
    for c in (0.1, 10.0, 1000.0):
        rc, _ = U.rho_stern_sicher(CAP_X * c, CAP_P, 30.0 * c)
        assert rc == pytest.approx(c * r, rel=1e-9)


def test_b6_schwelle_der_wahrscheinlichkeit_wandert_mit_rho():
    soll = {math.inf: 0.4333, 5000.0: 0.4407, 1000.0: 0.4703, 555.8014308792: 0.5000, 300.0: 0.5563, 200.0: 0.6152, 100.0: 0.7656, 50.0: 0.9280}
    for z in ev.p_stern_tabelle(200, -100, 30, 30, tuple(soll)):
        assert z["p_stern"] == pytest.approx(soll[z["rho"]], abs=5e-5)


def test_b7_zwei_kipppunkte_im_gitter_und_trotz_besserem_mittel_und_schlechtestem_fall():
    k = ev.kipp_analyse(-70, 70, 10, -80, 0.6, 50)
    assert [round(r, 2) for r in k["punkte"]] == [28.87, 72.18] and k["wahl_sae"] == -1 and k["ew_a"] > k["ew_b"] and k["min_a"] > k["min_b"]


def test_b12_frage_statt_zahl():
    soll = {1.2: 6.02, 1.5: 3.04, 2.0: 2.08, 3.0: 1.64, 4.0: 1.52, 10.0: 1.44}
    for g, f in soll.items():
        assert ev.rho_aus_frage(g) == pytest.approx(f, abs=0.006)


# --- B8: Kipppunkt-Zählung (vorgerechnet) -------------------------------------------------------------------------------------------------------

def test_b8_gegen_sichere_alternative_nie_mehr_als_ein_wechsel(frozen):
    sicher = [e for e in frozen["kipp"] if e["art"] == "sicher"]
    assert len(sicher) == 6 and sum(e["n"] for e in sicher) == 240000
    assert all(e["w2plus"] == 0 and e["mono"] and e["groesster"] <= 1 for e in sicher)
    assert sum(e["zwischen"] for e in sicher) == 114022 and sum(e["w1"] for e in sicher) == 56991       # K im Bereich der Lotterie; Probleme mit genau einem Wechsel (alle 240 000 zählen)


def test_b8_zwischen_zwei_riskanten_alternativen_gibt_es_mehrere(frozen):
    soll = {(2, 1301): (154, 19798), (2, 1302): (176, 20092), (3, 1301): (696, 29969), (3, 1302): (681, 29828), (4, 1301): (1150, 34989), (4, 1302): (1155, 35016)}
    for e in frozen["kipp"]:
        if e["art"] == "riskant":
            mehr, nd = soll[(e["m"], e["seed"])]
            assert abs(e["w2plus"] - mehr) <= 2 and abs(e["nicht_dominiert"] - nd) <= 2, (e["m"], e["seed"])
    anteile = {m: [e["w2plus"] / e["nicht_dominiert"] for e in frozen["kipp"] if e["art"] == "riskant" and e["m"] == m] for m in (2, 3, 4)}
    assert all(0.007 <= a <= 0.009 for a in anteile[2]) and all(0.0225 <= a <= 0.0235 for a in anteile[3]) and all(0.0325 <= a <= 0.0335 for a in anteile[4])


# --- B9: wie oft ändert die Risikoeinstellung die Wahl? ---------------------------------------------------------------------------------------

def test_b9_zahl_der_nicht_dominierten_probleme(strom101, strom2):
    assert abs(strom101["n_nd"] - 99389) <= 2 and abs(strom2["n_nd"] - 99388) <= 2 and strom101["dominiert"] == pytest.approx(0.503, abs=1e-3)


def test_b9_anteil_anderer_wahl_ueber_rho(strom101, strom2):
    s1 = {10: 0.360, 20: 0.324, 50: 0.226, 100: 0.136, 300: 0.049, 1000: 0.015}
    s2 = {10: 0.361, 20: 0.325, 50: 0.227, 100: 0.137, 300: 0.049, 1000: 0.014}
    for r in s1:
        assert strom101["staffel"][r] == pytest.approx(s1[r], abs=BAND), r
        assert strom2["staffel"][r] == pytest.approx(s2[r], abs=BAND), r
        assert abs(strom101["staffel"][r] - strom2["staffel"][r]) <= 0.002              # die beiden Ströme unterscheiden sich um höchstens zwei Zehntel Prozentpunkte


def test_b9_skala_ist_dieselbe_zahl_wie_ein_anderes_rho(strom101):
    sk = {0.1: 0.015, 0.5: 0.072, 1.0: 0.136, 2.0: 0.226, 10.0: 0.360}
    for c, a in sk.items():
        assert strom101["skalen"][c] == pytest.approx(a, abs=BAND), c
    gleich200 = ev.zufall_strom(101, 200000, 200, 1.0)["anteil"]                                  # ρ = 200 steht nicht in der Staffel
    for c, a in ((0.1, strom101["staffel"][1000]), (0.5, gleich200), (1.0, strom101["staffel"][100]), (2.0, strom101["staffel"][50]), (10.0, strom101["staffel"][10])):
        assert abs(strom101["skalen"][c] - a) <= 3 / strom101["n_nd"], c           # Skala ×c bei ρ = 100 gleich Skala ×1 bei ρ = 100/c (höchstens drei Probleme Gleichstand)


def test_b9_skaleninvarianz_der_wahl_hoechstens_zwei_abweichungen():
    P, A, dom = S.zufallsprobleme(101, 200000)
    nd = ~dom
    assert int(np.sum(U.wahl_vorzeichen(P, A, 100.0)[nd] != U.wahl_vorzeichen(P, A * 10.0, 1000.0)[nd])) <= 2


def test_b9_rho_stern_im_verhaeltnis_zur_spanne(strom101, strom2):
    v, w = strom101["verhaeltnis"], strom2["verhaeltnis"]
    assert abs(v["n_kippt"] - 38768) <= 3 and v["anteil_kippt"] == pytest.approx(0.390, abs=BAND) and w["anteil_kippt"] == pytest.approx(0.393, abs=BAND)
    assert v["p10"] == pytest.approx(0.076, abs=0.003) and v["median"] == pytest.approx(0.348, abs=0.003) and v["p90"] == pytest.approx(2.015, abs=0.01)
    assert w["median"] == pytest.approx(0.349, abs=0.003) and w["p90"] == pytest.approx(1.969, abs=0.01)
    assert (v["unter_spanne"], v["unter_halb"], v["ueber_fuenf"], v["ab_kapazitaet"]) == pytest.approx((0.804, 0.627, 0.040, 0.109), abs=BAND)
    assert w["ab_kapazitaet"] == pytest.approx(0.107, abs=BAND)
    assert abs(v["anteil_kippt"] - w["anteil_kippt"]) <= 0.0031                                  # die Ströme unterscheiden sich um höchstens 0,3 Prozentpunkte (39,0 gegen 39,3 %)
    # „in weiteren 1,0 % wechselt die Wahl zweimal“ (README B10, Zeile 4 der Korrekturen): Generator der Zufallsprobleme (P in 0,1 bis 0,9), Gitterzählung, Zufalls-Seed 101
    P, A, dom = S.zufallsprobleme(101, 200000)
    nd = ~dom
    P, A = P[nd], A[nd]
    ds = np.empty((len(P), len(U.RHO_GRID)))
    for j, r in enumerate(U.RHO_GRID):
        ds[:, j] = U.ce(A[:, 0, :], P, r) - U.ce(A[:, 1, :], P, r)
    wechsel = U.wechsel_zaehlen(ds)
    gleich_an_beiden_enden = U.vorzeichen(ds[:, 0]) == U.vorzeichen(ds[:, -1])
    assert abs(int(np.sum(wechsel >= 2)) - 994) <= 3 and abs(len(P) - 99389) <= 2                 # 994 von 99 389 = 1,0 %
    assert int(np.sum((wechsel >= 2) & ~gleich_an_beiden_enden)) == 0                            # alle bei gleicher Wahl an beiden Enden (also gerade Zahl, hier genau zwei)


# --- B10: Mittelwert-Varianz-Näherung ------------------------------------------------------------------------------------------------------------

def test_b10_naeherung_gegen_exakt(nae101, nae2):
    s1 = {10: 0.127, 20: 0.106, 50: 0.068, 100: 0.032, 300: 0.004, 1000: 0.0004}
    s2 = {10: 0.128, 20: 0.105, 50: 0.069, 100: 0.032, 300: 0.005, 1000: 0.0004}
    for r in s1:
        assert nae101["staffel"][r] == pytest.approx(s1[r], abs=BAND), r
        assert nae2["staffel"][r] == pytest.approx(s2[r], abs=BAND), r
    assert abs(nae101["n_kippend"] - 13549) <= 3 and abs(nae101["n_kippend_falsch"] - 1254) <= 3
    assert nae101["n_kippend_falsch"] / nae101["n_kippend"] == pytest.approx(0.093, abs=0.002)


# --- B11: Exponential gegen Potenz ------------------------------------------------------------------------------------------------------------

def test_b11_kippvermoegen_und_wahl_je_vermoegen():
    t = ev.kipp_vermoegen_tabelle()
    assert (round(t[1], 1), round(t[2], 1), round(t[4], 2)) == (522.5, 1075.0, 2184.95)
    assert ev.vermoegen_analyse(1000, 2, 200)["wahl_pow"] == "K" and ev.vermoegen_analyse(2000, 2, 200)["wahl_pow"] == "G"
    assert ev.vermoegen_analyse(1000, 2, 200)["wahl_exp"] == ev.vermoegen_analyse(2000, 2, 200)["wahl_exp"] == "K"


def test_b11_kalibriertes_rho_und_anteil_anderer_wahl(frozen):
    rho_soll = {(1, 500): 532.6, (1, 1000): 1041.0, (1, 5000): 5048.1, (2, 500): 261.1, (2, 1000): 517.8, (2, 5000): 2523.5, (4, 500): 125.6, (4, 1000): 256.3, (4, 5000): 1261.2}
    anteil_max = {(1, 500): 0.0016, (1, 1000): 0.0004, (1, 5000): 0.00003, (2, 500): 0.0032, (2, 1000): 0.0009, (2, 5000): 0.00004, (4, 500): 0.0078, (4, 1000): 0.0018, (4, 5000): 0.00004}
    for e in frozen["kalibrierung"]:
        key = (e["gamma"], e["w"])
        assert e["rho_k"] == pytest.approx(rho_soll[key], abs=0.06)
        assert e["anteil"] <= anteil_max[key] + 6e-5, key
    alle = [e["anteil"] for e in frozen["kalibrierung"]]
    assert max(alle) == pytest.approx(0.0078, abs=1e-4) and min(alle) < 5e-5                    # „0,00 bis 0,78 %“


def test_b11_eine_zelle_neu_gerechnet_stimmt_mit_der_datei(frozen):
    z = ev.kalibrierung_zelle(101, 200000, 4, 500)
    e = next(e for e in frozen["kalibrierung"] if (e["gamma"], e["w"], e["seed"]) == (4, 500, 101))
    assert z["rho_k"] == pytest.approx(e["rho_k"], rel=1e-9) and abs(z["anteil"] - e["anteil"]) <= 5e-5 and abs(z["n_nd"] - e["n_nd"]) <= 2


def test_b13_voreinstellung_der_app_nennt_die_zahlen_der_readme():
    """Die Voreinstellungen und Presets nennen nur Zahlen, die hier belegt sind."""
    assert C.PRESETS["Kapazitätsfall, ρ = 200"]["rho"] == 200 and C.PRESETS["Skala ×10"]["sk4"] == 10.0 and C.PRESETS["Zufallsprobleme ρ = 100"]["seed4"] == 101
    assert C.KAPAZITAETSFALL_RATIO == pytest.approx(555.8014 / 300, abs=1e-3)
    assert ev.rho_kurve()[0]["L"] == 10 and ev.rho_kurve()[-1]["L"] == 139

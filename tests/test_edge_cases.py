"""Randfälle (DEMO-PLAYBOOK §12): ρ → 0 und ρ → ∞, p = 0 oder 1, ein Zustand, gleiche Auszahlungen, K gleich Minimum oder Maximum, Vorsprung ≤ 0, Erwartungswert gleich,
Potenznutzen mit w ≤ 100, sehr große Auszahlungen (x/ρ über 700) und die Ränder der Regler."""

import math

import numpy as np
import pytest

import rae_constants as C
import rae_evaluation as ev
import rae_scenario as S
import rae_utility as U
import rae_wealth as W


def test_rho_sehr_klein_laeuft_nicht_ueber(cap):
    x, p, _ = cap
    for rho in (0.5, 0.05, 1e-3):
        wert = float(U.ce(x, p, rho))
        assert wert == pytest.approx(-100.0 - rho * math.log(0.5), abs=1e-9)               # Minimum − ρ ln p_min; der gute Zustand verschwindet (e^(−200/ρ))
    assert float(U.ce(np.array([-1000.0, 5.0]), np.array([0.5, 0.5]), 0.5)) == pytest.approx(-1000.0 + 0.5 * math.log(2), abs=1e-9)   # x/ρ = −2000, e^2000 würde überlaufen


def test_auszahlungen_ueber_700_rho_einheiten_sind_endlich():
    assert float(U.ce(np.array([5000.0, -5000.0]), np.array([0.5, 0.5]), 2.0)) == pytest.approx(-5000.0 + 2.0 * math.log(2), abs=1e-9)
    assert float(U.ce(np.array([5000.0, 4000.0]), np.array([0.5, 0.5]), 2.0)) == pytest.approx(4000.0 + 2.0 * math.log(2), abs=1e-9)


def test_rho_sehr_gross_und_unendlich_geben_den_erwartungswert(cap):
    x, p, _ = cap
    assert U.ce(x, p, math.inf) == 50.0
    assert float(U.ce(x, p, 1e9)) == pytest.approx(50.0, abs=1e-4)
    assert float(U.risikopraemie(x, p, 1e9)) >= -1e-9


def test_wahrscheinlichkeit_null_oder_eins_und_ein_zustand():
    assert float(U.ce(np.array([200.0, -100.0]), np.array([1.0, 0.0]), 50.0)) == pytest.approx(200.0, abs=1e-9)     # der Zustand mit p = 0 zählt nicht
    assert float(U.ce(np.array([200.0, -10000.0]), np.array([1.0, 0.0]), 1.0)) == pytest.approx(200.0, abs=1e-9)    # und läuft nicht über
    assert float(U.ce(np.array([42.0]), np.array([1.0]), 30.0)) == pytest.approx(42.0, abs=1e-12)


def test_sichere_alternative_gleich_minimum_oder_maximum_hat_kein_rho_stern(cap):
    x, p, _ = cap
    assert U.rho_stern_sicher(x, p, -100.0) == (None, "lotterie_immer")          # K gleich dem Minimum: die Lotterie ist nie schlechter
    assert U.rho_stern_sicher(x, p, 200.0) == (None, "k_bleibt")                  # K gleich dem Maximum: die Lotterie holt es nie ein
    assert U.rho_stern_sicher(x, p, 300.0)[1] == "k_bleibt" and U.rho_stern_sicher(x, p, -300.0)[1] == "lotterie_immer"


def test_vorsprung_null_oder_negativ_gibt_kein_rho_stern(cap):
    x, p, _ = cap
    assert U.rho_stern_sicher(x, p, 50.0) == (None, "k_bleibt")                   # Erwartungswert gleich K: die sichere Alternative bleibt bei jedem endlichen ρ gewählt
    assert U.rho_stern_sicher(x, p, 60.0)[1] == "k_bleibt"
    k = ev.kipp_analyse(200.0, -100.0, 60.0, 60.0, 0.5, 200.0)
    assert k["status"] == "k_bleibt" and k["punkte"] == [] and k["vorsprung"] < 0 and k["naeherung"] is None


def test_sehr_kleiner_vorsprung_gibt_grosses_rho_stern():
    r, status = U.rho_stern_sicher(np.array([200.0, -100.0]), np.array([0.5, 0.5]), 49.999)
    assert status == "ok" and r > 1e5 and r == pytest.approx(U.normal_rho_stern(300.0, 0.001), rel=1e-3)


def test_gleiche_alternativen_und_dominierte_haben_keinen_kipppunkt():
    assert U.kipp_punkte([10.0, 20.0], [10.0, 20.0], [0.4, 0.6]) == ([], 0, 0)
    punkte, unten, oben = U.kipp_punkte([10.0, 30.0], [5.0, 20.0], [0.4, 0.6])        # A dominiert
    assert punkte == [] and unten == 1 and oben == 1


def test_gleicher_erwartungswert_bevorzugt_bei_jedem_endlichen_rho_die_sicherheit():
    x, p = np.array([100.0, -20.0]), np.array([0.25, 0.75])                          # Erwartungswert 10
    assert float(U.erwartung(x, p)) == 10.0
    assert U.rho_stern_sicher(x, p, 10.0) == (None, "k_bleibt")
    assert float(U.ce(x, p, 1e4)) < 10.0


def test_p_stern_ohne_schwelle_bei_ueberlauf():
    assert U.p_stern([1e6, -1e6], [0.0, 0.0], 1.0) is None


def test_potenznutzen_mit_vermoegen_bis_100_ist_nicht_definiert(cap):
    x, p, _ = cap
    assert not W.definiert(x, p, 100.0) and not W.definiert(x, p, 50.0) and W.definiert(x, p, 100.5)
    assert math.isnan(float(W.ce_pow(x, p, 100.0, 2.0)))
    v = ev.vermoegen_analyse(100.0, 2, 200.0)
    assert not v["definiert"] and v["ce_pow"] is None and ev.verdict_vermoegen(v, ev.kipp_vermoegen_tabelle())[0] == "warning"
    assert W.kipp_vermoegen(x, p, 60.0, 2.0) is None                                 # K über dem Erwartungswert: kein Kippvermögen


def test_regler_raender_der_analysen_sind_endlich():
    for rho in (C.RHO_OPTIONS[0], C.RHO_OPTIONS[-1]):
        for x1, x2 in ((C.X_MIN, C.X_MIN), (C.X_MAX, C.X_MIN), (C.X_MIN, C.X_MAX), (C.X_MAX, C.X_MAX)):
            a = ev.lotterie_analyse(x1, x2, C.P_MIN, rho)
            assert all(math.isfinite(a[k]) for k in ("ew", "sae", "praemie", "naeherung"))
            assert ev.verdict_sae(a)[0] == "info"
        for a1, a2, b1, b2 in ((C.A_MIN, C.A_MIN, C.A_MAX, C.A_MAX), (C.A_MAX, C.A_MIN, C.A_MIN, C.A_MAX), (C.A_MAX, C.A_MAX, C.A_MAX, C.A_MAX)):
            for p in (C.P_MIN, C.P_MAX):
                k = ev.kipp_analyse(a1, a2, b1, b2, p, rho)
                assert ev.verdict_kipp(k)[0] in ("info", "success", "warning")


def test_zufallsprobleme_liegen_in_den_grenzen_des_generators():
    P, A, dom = S.zufallsprobleme(5, 5000)
    assert A.min() >= S.X_LO and A.max() <= S.X_HI and np.all(P > 0) and np.all(P < 1) and P[:, 0].min() >= S.P_LO and P[:, 0].max() <= S.P_HI
    assert np.allclose(P.sum(axis=1), 1.0) and np.array_equal(A, np.round(A))
    assert {S.X_LO, S.X_HI} <= set(A.astype(int).ravel().tolist())                    # beide Ränder kommen vor (Bereich eingeschlossen)

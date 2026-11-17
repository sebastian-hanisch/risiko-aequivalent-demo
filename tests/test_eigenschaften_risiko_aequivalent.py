"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für Sicherheitsäquivalent und Wahl gelten müssen, ohne dass man das richtige Ergebnis kennt.
Je Eigenschaft 20 bis 60 Zufallsinstanzen mit festem Zufalls-Seed (SplitMix64 über `rae_scenario`), Gleichstände unter 1e-9 zählen nicht."""

import math

import numpy as np
import pytest

import rae_scenario as S
import rae_utility as U
import rae_wealth as W

P, A, DOM = S.zufallsprobleme(4242, 60)
RHOS = (15.0, 60.0, 250.0, 1500.0)


def _ce(i, alt, rho):
    return float(U.ce(A[i, alt], P[i], rho))


@pytest.mark.parametrize("rho", RHOS)
def test_skalieren_von_auszahlungen_und_rho_aendert_die_wahl_nicht(rho):
    for c in (0.1, 2.0, 10.0):
        w1 = U.wahl_vorzeichen(P, A, rho)
        w2 = U.wahl_vorzeichen(P, A * c, rho * c)
        assert np.sum(w1 != w2) <= 1, (rho, c)                      # höchstens ein Gleitkomma-Gleichstand
        assert U.ce(A[0, 0], P[0], rho) * c == pytest.approx(float(U.ce(A[0, 0] * c, P[0], rho * c)), abs=1e-9)


@pytest.mark.parametrize("w", (-300.0, 17.0, 5000.0))
def test_verschieben_aller_auszahlungen_verschiebt_das_aequivalent_um_genau_w_delta(w):
    for rho in RHOS:
        for i in range(60):
            for alt in (0, 1):
                assert U.ce(A[i, alt] + w, P[i], rho) == pytest.approx(_ce(i, alt, rho) + w, abs=1e-8), (i, alt)
        assert np.sum(U.wahl_vorzeichen(P, A + w, rho) != U.wahl_vorzeichen(P, A, rho)) <= 1


def test_aequivalent_liegt_zwischen_minimum_und_erwartungswert_und_praemie_ist_nicht_negativ():
    for rho in RHOS:
        for alt in (0, 1):
            ce = U.ce(A[:, alt, :], P, rho)
            assert np.all(ce >= A[:, alt, :].min(axis=1) - 1e-9) and np.all(ce <= U.erwartung(A[:, alt, :], P) + 1e-9)
            assert np.all(U.risikopraemie(A[:, alt, :], P, rho) >= -1e-9)


def test_risikopraemie_ist_null_bei_sicherer_auszahlung():
    for k in (-80.0, 0.0, 33.0, 190.0):
        for rho in RHOS:
            assert U.risikopraemie(np.array([k, k]), np.array([0.3, 0.7]), rho) == pytest.approx(0.0, abs=1e-9)


def test_aequivalent_waechst_mit_rho_auf_dem_gitter():
    gitter = np.exp(np.linspace(math.log(0.5), math.log(1e5), 120))
    for i in range(40):
        for alt in (0, 1):
            werte = [float(U.ce(A[i, alt], P[i], r)) for r in gitter]
            assert all(b >= a - 1e-9 for a, b in zip(werte, werte[1:])), (i, alt)


def test_vertauschen_der_zustaende_aendert_nichts():
    for rho in RHOS:
        assert np.array_equal(U.wahl_vorzeichen(P, A, rho), U.wahl_vorzeichen(P[:, ::-1], A[:, :, ::-1], rho))
        assert U.ce(A[3, 0][::-1], P[3][::-1], rho) == pytest.approx(float(U.ce(A[3, 0], P[3], rho)), abs=1e-12)


def test_zusammenlegen_gleicher_auszahlungen_aendert_nichts():
    x, p = np.array([40.0, 40.0, -20.0]), np.array([0.25, 0.35, 0.4])
    for rho in RHOS:
        assert float(U.ce(x, p, rho)) == pytest.approx(float(U.ce(np.array([40.0, -20.0]), np.array([0.6, 0.4]), rho)), abs=1e-12)


def test_dominierte_alternativen_werden_bei_keinem_rho_gewaehlt():
    zaehler = 0
    for i in range(60):
        if not DOM[i]:
            continue
        besser = 0 if np.all(A[i, 0] >= A[i, 1]) else 1
        for rho in RHOS:
            d = _ce(i, 1 - besser, rho) - _ce(i, besser, rho)
            assert d <= 1e-9, (i, rho)
        zaehler += 1
    assert zaehler >= 15                                              # die Stichprobe enthält dominierte Probleme (Anteil etwa die Hälfte)


def test_positiv_affine_transformation_des_nutzens_aendert_die_wahl_nicht():
    """Wahl über den Erwartungsnutzen von u' = 3u + 5 (statt u): gleiche Wahl wie über das Sicherheitsäquivalent."""
    for rho in (40.0, 400.0):
        for i in range(60):
            eu = [sum(p * (3.0 * (1.0 - math.exp(-x / rho)) + 5.0) for x, p in zip(A[i, alt], P[i])) for alt in (0, 1)]
            d = _ce(i, 1, rho) - _ce(i, 0, rho)
            if abs(d) > 1e-6:
                assert (eu[1] > eu[0]) == (d > 0), (rho, i)


def test_aequivalent_ist_additiv_ueber_unabhaengige_risiken():
    """SÄ(X + Y) = SÄ(X) + SÄ(Y) für unabhängige X, Y (Produktlotterie), bei gleichem ρ."""
    for i in range(20):
        px, py = P[i], P[i + 20]
        xs, ys = A[i, 0], A[i + 20, 1]
        summe = np.array([x + y for x in xs for y in ys])
        pp = np.array([a * b for a in px for b in py])
        for rho in (30.0, 300.0):
            assert float(U.ce(summe, pp, rho)) == pytest.approx(float(U.ce(xs, px, rho)) + float(U.ce(ys, py, rho)), abs=1e-9)


def test_gegen_eine_sichere_alternative_hoechstens_ein_wechsel_im_gitter():
    gitter = np.exp(np.linspace(math.log(0.5), math.log(1e5), 150))
    for m in (2, 3, 4):
        Pm, Am, K = S.zufallsprobleme_m(77, 50, m, "sicher")
        for i in range(50):
            d = np.array([float(U.ce(Am[i], Pm[i], r)) - K[i] for r in gitter])
            assert int(U.wechsel_zaehlen(d[None, :])[0]) <= 1, (m, i)


def test_rho_stern_waechst_wenn_der_vorsprung_schrumpft():
    p = np.array([0.5, 0.5])
    werte = []
    for verlust in (10, 30, 50, 70, 90, 100, 110, 120, 130, 139):                      # Vorsprung (200 − L)/2 − 30 sinkt von 65 auf 0,5
        r, status = U.rho_stern_sicher(np.array([200.0, -float(verlust)]), p, 30.0)
        assert status == "ok"
        werte.append(r)
    assert all(a < b for a, b in zip(werte, werte[1:]))


def test_gleicher_zufalls_seed_gleicher_strom_und_prefix_eigenschaft():
    P1, A1, _ = S.zufallsprobleme(11, 500)
    P2, A2, _ = S.zufallsprobleme(11, 500)
    P3, A3, _ = S.zufallsprobleme(11, 800)
    assert np.array_equal(A1, A2) and np.array_equal(P1, P2)
    assert np.array_equal(A1, A3[:500]) and np.array_equal(P1, P3[:500])                # Problem i hängt nicht von der Zahl der Probleme ab
    assert not np.array_equal(A1, S.zufallsprobleme(12, 500)[1])


def test_potenznutzen_wahl_haengt_vom_vermoegen_der_exponentialnutzen_nicht():
    x, p = np.array([200.0, -100.0]), np.array([0.5, 0.5])
    wahlen_pot = {w: float(W.ce_pow(x, p, w, 2.0)) > 30.0 for w in (500.0, 1000.0, 1100.0, 3000.0)}
    assert wahlen_pot == {500.0: False, 1000.0: False, 1100.0: True, 3000.0: True}     # Kippvermögen 1075 (γ = 2)
    assert float(U.ce(x + 500.0, p, 200.0)) - 500.0 == pytest.approx(float(U.ce(x + 3000.0, p, 200.0)) - 3000.0, abs=1e-9)

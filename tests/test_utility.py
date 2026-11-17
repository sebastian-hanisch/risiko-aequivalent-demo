"""Kern von Hand (DEMO-PLAYBOOK §1b): Sicherheitsäquivalent, Kipp-Risikotoleranz, Schwelle p* und Kipppunkte an kleinen Beispielen, deren Zahlen von Hand
nachgerechnet sind (Brüche, ganze Zahlen, geschlossene Formeln).

Technik der exakten Vorzeichen: Mit z = e^(−u0/ρ) und Auszahlungen als Vielfachen von u0 ist „A schlägt B“ genau dann, wenn E_A[z^(x/u0)] < E_B[z^(x/u0)];
beide Seiten sind Polynome in z, das Vorzeichen ist in Brüchen exakt (Ganzzahlarithmetik, auf jeder Plattform gleich)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import rae_utility as U

H = F(1, 2)
PHI = (1 + math.sqrt(5)) / 2


def erwartung_z(xs, ps, z, u0):
    return sum(F(p) * z ** (x // u0) for x, p in zip(xs, ps))


def a_schlaegt_b(xa, pa, xb, pb, z, u0):
    """+1: A hat den größeren Erwartungsnutzen, -1: B, 0 gleich (exakt)."""
    d = erwartung_z(xb, pb, z, u0) - erwartung_z(xa, pa, z, u0)
    return (d > 0) - (d < 0)


def test_zweipunkt_lotterie_bruchrechnung_h1():
    """u(x) = 1 − 2^(−x/100) (ρ = 100/ln 2), Lotterie (0 | 100): Erwartungsnutzen 1/4. K sicher schlägt sie, wenn 2^(K/100) > 4/3."""
    assert H * 0 + H * (1 - F(1, 2)) == F(1, 4)
    assert 972 < 1024                       # K = 40: (4/3)^5 = 1024/243 > 4: die Lotterie gewinnt
    assert 3 ** 100 < 2 ** 159              # K = 41: die Lotterie gewinnt
    assert 3 ** 50 > 2 ** 79                # K = 42: der sichere Betrag gewinnt
    erwartet = 100 * math.log2(F(4, 3))
    assert 41.0 < erwartet < 42.0
    assert U.ce([0.0, 100.0], [0.5, 0.5], 100 / math.log(2)) == pytest.approx(erwartet, abs=1e-12)


def test_zweipunkt_lotterie_rho_100_und_skaleninvarianz_h2():
    erwartet = -100 * math.log((1 + math.exp(-1)) / 2)
    assert erwartet == pytest.approx(37.988549304, abs=1e-9)
    assert U.ce([0.0, 100.0], [0.5, 0.5], 100.0) == pytest.approx(erwartet, abs=1e-12)
    assert U.ce([0.0, 200.0], [0.5, 0.5], 200.0) == pytest.approx(2 * erwartet, abs=1e-12)     # Auszahlungen und ρ verdoppelt: Sicherheitsäquivalent verdoppelt


def test_goldener_schnitt_h3():
    """G = (+200 | −100), je ½, gegen 0 sicher: mit y = e^(−100/ρ) ist ½y² + ½/y = 1 ⇔ y³ − 2y + 1 = (y − 1)(y² + y − 1) = 0, y = (√5 − 1)/2, ρ* = 100/ln φ."""
    produkt = [0, 0, 0, 0]
    for i, a in enumerate([1, -1]):                                                          # y − 1
        for j, b in enumerate([1, 1, -1]):                                                   # y² + y − 1
            produkt[i + j] += a * b
    assert produkt == [1, 0, -2, 1]                                                          # y³ − 2y + 1 (ganzzahlig)
    r, status = U.rho_stern_sicher([200.0, -100.0], [0.5, 0.5], 0.0)
    assert status == "ok" and r == pytest.approx(100 / math.log(PHI), rel=1e-9) and r == pytest.approx(207.8086921235, abs=1e-6)
    pa, pb = [H, H], [F(1)]
    assert a_schlaegt_b([200, -100], pa, [0], pb, F(3, 5), 100) == -1                        # z = 3/5 (ρ = 195,8): 0 sicher
    assert a_schlaegt_b([200, -100], pa, [0], pb, F(31, 50), 100) == 1                       # z = 31/50 (ρ = 209,2): die Lotterie


def test_kapazitaetsfall_sicherheitsaequivalent_h4(cap):
    x, p, k = cap
    assert float(U.erwartung(x, p)) == 50.0 and float(U.varianz(x, p)) == 22500.0            # ½·200 + ½·(−100); Streuung 150
    assert U.ce(x, p, 200.0) == pytest.approx(-1.6532, abs=1e-4)                              # Risikoprämie 51,65
    assert U.risikopraemie(x, p, 200.0) == pytest.approx(51.6532, abs=1e-4)
    assert U.ce(x, p, 555.0) < 30.0 < U.ce(x, p, 556.0)
    assert U.normal_naeherung(x, p, 200.0) == pytest.approx(50 - 22500 / 400)                 # 50 − 56,25 = −6,25
    assert U.ce(x, p, math.inf) == 50.0


def test_kapazitaetsfall_kipp_rho_exakt_in_brueche_h4(cap):
    """Exakt in Brüchen: z = 49/50 (ρ = 495) wählt K, z = 99/100 (ρ = 995) wählt G; Bisektion auf dem Polynom vom Grad 30 gibt ρ* = 555,8014309."""
    x, p, k = cap
    pa, pb = [H, H], [F(1)]
    assert a_schlaegt_b([200, -100], pa, [30], pb, F(49, 50), 10) == -1
    assert a_schlaegt_b([200, -100], pa, [30], pb, F(99, 100), 10) == 1
    lo, hi = F(49, 50), F(99, 100)
    slo = a_schlaegt_b([200, -100], pa, [30], pb, lo, 10)
    for _ in range(70):
        mid = (lo + hi) / 2
        if a_schlaegt_b([200, -100], pa, [30], pb, mid, 10) == slo:
            lo = mid
        else:
            hi = mid
    r_exakt = -10 / math.log(float(lo))
    r, status = U.rho_stern_sicher(x, p, k)
    assert status == "ok" and r_exakt == pytest.approx(555.8014309, abs=1e-6)
    assert r == pytest.approx(r_exakt, rel=1e-9)
    r2, _ = U.rho_stern_sicher([200.0, -50.0], p, k)
    assert r2 == pytest.approx(158.0545240, abs=1e-6) and r2 < 200 < r                       # Fall 2 gegen Fall 1: dieselbe Zahl ρ = 200, andere Wahl
    assert U.normal_rho_stern(300.0, 20.0) == 562.5                                           # Spanne²/(8·Vorsprung) = 90 000/160


def test_p_stern_ist_linear_und_wandert_mit_rho():
    assert U.p_stern([200.0, -100.0], [30.0, 30.0], math.inf) == pytest.approx(13 / 30, abs=1e-15)    # (30 + 100)/(200 + 100)
    assert U.p_stern([200.0, -100.0], [30.0, 30.0], 555.8014308792) == pytest.approx(0.5, abs=1e-9)  # bei ρ* sind beide gleich gut, wenn p = ½
    werte = [U.p_stern([200.0, -100.0], [30.0, 30.0], r) for r in (5000.0, 1000.0, 300.0, 200.0, 100.0, 50.0)]
    assert all(a < b for a, b in zip(werte, werte[1:]))                                       # p* steigt, wenn ρ sinkt
    assert U.p_stern([1.0, 1.0], [1.0, 1.0], 100.0) is None                                   # gleiche Alternativen: keine Schwelle


def test_zwei_kipppunkte_h7_exakt_und_im_gitter():
    """p = (3/5, 2/5), A = (−70 | 70), B = (10 | −80): Erwartungswerte −14 und −26, schlechtester Fall −70 und −80; trotzdem gewinnt B zwischen den Kipppunkten."""
    pa = [F(3, 5), F(2, 5)]
    xa, xb = [-70, 70], [10, -80]
    assert (sum(p * x for p, x in zip(pa, xa)), sum(p * x for p, x in zip(pa, xb))) == (-14, -26) and min(xa) > min(xb)
    assert [a_schlaegt_b(xa, pa, xb, pa, z, 10) for z in (F(1, 2), F(4, 5), F(9, 10))] == [1, -1, 1]
    punkte, unten, oben = U.kipp_punkte([-70.0, 70.0], [10.0, -80.0], [0.6, 0.4])
    assert punkte == pytest.approx([28.8702, 72.1826], abs=1e-3) and unten == 1 and oben == 1


def test_frage_statt_zahl_h8():
    import rae_evaluation as ev

    assert ev.rho_aus_frage(2.0) == pytest.approx(1 / math.log(PHI), abs=1e-9)                 # 2,078 · L bei Gewinn 2L
    assert ev.rho_aus_frage(1.2) > ev.rho_aus_frage(4.0) > ev.rho_aus_frage(10.0)             # größerer Gewinn: kleineres ρ/L


def test_normalverteilung_formel_h9():
    assert F(50) - F(40 ** 2, 2 * 100) == 42                                                  # μ − σ²/(2ρ) bei μ = 50, σ = 40, ρ = 100


def test_ce_haelt_die_genauigkeit_bei_grossem_betrag_gegen_rho():
    """Regression: bei x/ρ = 25 (sichere Auszahlung 100 GE, ρ = 4) lieferte die Rechnung über log1p/expm1 nur etwa 1e-4 Genauigkeit, weil Σ p·expm1 nahe −1 liegt;
    das fiel in der Kipppunkt-Zählung als scheinbar fallendes Sicherheitsäquivalent auf. Jetzt gilt die Verschiebung ab |x/ρ| > 3."""
    assert U.ce([100.0, 100.0], [0.5, 0.5], 4.0) == pytest.approx(100.0, abs=1e-12)
    assert U.ce([100.0, 100.0], [0.3, 0.7], 3.0) == pytest.approx(100.0, abs=1e-12)
    assert U.ce([-100.0, -100.0], [0.5, 0.5], 4.0) == pytest.approx(-100.0, abs=1e-12)


def test_ce_nicht_fallend_in_rho_und_zwischen_minimum_und_erwartungswert(cap):
    x, p, _ = cap
    rhos = np.exp(np.linspace(math.log(0.5), math.log(1e5), 400))
    werte = U.ce(np.broadcast_to(x, (400, 2)), np.broadcast_to(p, (400, 2)), rhos)
    assert np.all(np.diff(werte) >= -1e-9) and np.all(werte >= -100.0 - 1e-9) and np.all(werte <= 50.0 + 1e-9)

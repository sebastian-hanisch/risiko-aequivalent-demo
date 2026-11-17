"""Orakel-Tests (ORACLE-PLAYBOOK §5, DEMO-PLAYBOOK §12): Sicherheitsäquivalent, Kipp-Risikotoleranz, Schwelle p*, Potenz-Kippvermögen und die Zufallsprobleme gegen einen
ANDEREN Rechenweg als den der Demo.

* Sicherheitsäquivalent: `scipy.optimize.brentq` auf dem Nutzen u (ohne die Formel −ρ ln E e^(−X/ρ)), `scipy.integrate.quad` für die Normalverteilung, eigene Fassung der Gegenprobe
  über u(x) = −exp(−γx) und die Umkehrung (γ = 1/ρ).
* ρ*: exakte Bisektion in Brüchen auf dem Polynom in z = e^(−u0/ρ) (Ganzzahlarithmetik, plattformfest), gegen die Rechnung der Demo (relativ 1e-9) und `numpy.roots` (relativ 1e-6).
* Zahl der Nullstellen: exakte Vorzeichenwechsel in Brüchen gegen die Gitterzählung.
* p*(ρ): geschlossene Form gegen „Sicherheitsäquivalente gleich“ (1e-9). Potenz-Kippvermögen: exakt in Brüchen und `brentq`.
* Zufallsprobleme: die vektorisierte Wahl gegen eine Rechnung mit `math.exp` und `math.log` Problem für Problem.
Plattform: exp und log kommen aus der Bibliothek des Systems; Entscheidungen mit |ΔSÄ| unter 1e-9 gelten als gleich, Zählungen als Band."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import rae_scenario as S
import rae_utility as U
import rae_wealth as W

optimize = pytest.importorskip("scipy.optimize")
integrate = pytest.importorskip("scipy.integrate")


# --- exakte Route in Brüchen -------------------------------------------------------------------------------------------------------------------

def poly_d(xs, ps, ys, qs, u0):
    """Koeffizienten (Exponent -> Bruch) von D(z) · z^m, D(z) = E_B[z^(y/u0)] − E_A[z^(x/u0)], z = exp(−u0/ρ); alle Auszahlungen ganze Vielfache von u0.
    SÄ_A > SÄ_B ⇔ E_A[exp(−x/ρ)] < E_B[exp(−y/ρ)] ⇔ D(z) > 0."""
    ex = [F(x, u0) for x in xs] + [F(y, u0) for y in ys]
    assert all(e.denominator == 1 for e in ex)
    m = -min(int(e) for e in ex)
    coef = {}
    for x, p in zip(xs, ps):
        e = int(F(x, u0)) + m
        coef[e] = coef.get(e, F(0)) - F(p)
    for y, q in zip(ys, qs):
        e = int(F(y, u0)) + m
        coef[e] = coef.get(e, F(0)) + F(q)
    return coef


def auswerten(coef, z):
    return sum(c * z ** e for e, c in coef.items())


def vorz(v):
    return (v > 0) - (v < 0)


def wurzeln_exakt(coef, gitter=1500):
    """Alle Vorzeichenwechsel des Polynoms auf z = i/gitter (exakt) und je eine exakte dyadische Bisektion (2^-64); Rückgabe Liste von (lo, hi) in Brüchen."""
    zs = sorted({F(i, gitter) for i in range(1, gitter)} | {F(10 ** 6 - 1, 10 ** 6), F(10 ** 9 - 1, 10 ** 9)})
    sg = [vorz(auswerten(coef, z)) for z in zs]
    out = []
    for i in range(len(zs) - 1):
        if sg[i] != 0 and sg[i + 1] != 0 and sg[i] != sg[i + 1]:
            lo, hi, slo = zs[i], zs[i + 1], sg[i]
            for _ in range(64):
                mid = (lo + hi) / 2
                s = vorz(auswerten(coef, mid))
                if s == 0:
                    lo = hi = mid
                    break
                if s == slo:
                    lo = mid
                else:
                    hi = mid
            out.append((lo, hi))
    return out


def rho_aus_z(z, u0):
    return -u0 / math.log(float(z))


FAELLE = [
    ("Fall 1", [200, -100], [F(1, 2)] * 2, [30], [F(1)], 10, 555.8014308792),
    ("Fall 2", [200, -50], [F(1, 2)] * 2, [30], [F(1)], 10, 158.0545239709),
    ("Goldener Schnitt", [200, -100], [F(1, 2)] * 2, [0], [F(1)], 100, 207.8086921235),
    ("drei Zustände", [250, 60, -150], [F(3, 10), F(4, 10), F(3, 10)], [90, 50, 0], [F(3, 10), F(4, 10), F(3, 10)], 10, 1631.1555986723),
]


@pytest.mark.parametrize("name, xs, ps, ys, qs, u0, soll", FAELLE)
def test_rho_stern_drei_rechenwege_gleich(name, xs, ps, ys, qs, u0, soll):
    coef = poly_d(xs, ps, ys, qs, u0)
    iv = wurzeln_exakt(coef)
    assert len(iv) == 1                                                # genau ein Vorzeichenwechsel auf dem exakten Gitter
    r_exakt = rho_aus_z((iv[0][0] + iv[0][1]) / 2, u0)
    assert r_exakt == pytest.approx(soll, rel=1e-9)                    # Literatur: die Zahl der früheren Messung
    px = [float(p) for p in ps]
    if len(ys) == 1:
        r_demo, status = U.rho_stern_sicher(np.array(xs, dtype=float), np.array(px), float(ys[0]))
        assert status == "ok"
    else:
        punkte, _, _ = U.kipp_punkte(np.array(xs, dtype=float), np.array(ys, dtype=float), np.array(px))
        assert len(punkte) == 1
        r_demo = punkte[0]
    assert r_demo == pytest.approx(r_exakt, rel=1e-9)
    grad = max(coef)
    c = np.zeros(grad + 1)
    for e, v in coef.items():
        c[grad - e] = float(v)
    wurzeln = [r.real for r in np.roots(c) if abs(r.imag) < 1e-9 and 1e-9 < r.real < 1 - 1e-6]
    r_np = min((rho_aus_z(w, u0) for w in wurzeln), key=lambda v: abs(v - r_exakt))
    assert r_np == pytest.approx(r_exakt, rel=1e-6)                    # numpy.roots: weniger genau, daher 1e-6


def test_anzahl_der_kipppunkte_exakt_gegen_gitter():
    """H7: zwei exakte Nullstellen, die Gitterzählung findet beide; dazu ein Beispiel mit genau einem und eines ohne Wechsel (A dominiert B)."""
    pa = [F(3, 5), F(2, 5)]
    iv = wurzeln_exakt(poly_d([-70, 70], pa, [10, -80], pa, 10))
    assert len(iv) == 2
    exakt = [rho_aus_z((a + b) / 2, 10) for a, b in iv]
    punkte, _, _ = U.kipp_punkte([-70.0, 70.0], [10.0, -80.0], [0.6, 0.4])
    assert punkte == pytest.approx(sorted(exakt), rel=1e-9)
    pb = [F(1, 2), F(1, 2)]
    assert len(wurzeln_exakt(poly_d([200, -100], pb, [100, -60], pb, 10))) == 1       # B hat den besseren schlechtesten Fall, A den besseren Erwartungswert: ein Wechsel
    assert U.kipp_punkte([200.0, -100.0], [100.0, -60.0], [0.5, 0.5])[0] != []
    assert len(wurzeln_exakt(poly_d([200, -100], pb, [0, -110], pb, 10))) == 0        # A dominiert B (in beiden Zuständen besser): kein Wechsel
    assert U.kipp_punkte([200.0, -100.0], [0.0, -110.0], [0.5, 0.5])[0] == []


# --- Sicherheitsäquivalent -----------------------------------------------------------------------------------------------------------------------

def _ce_brentq(xs, ps, rho):
    """Sicherheitsäquivalent als Nullstelle von u(s) − E u(X) mit u = 1 − exp(−x/ρ): ohne die geschlossene Formel."""
    eu = sum(p * (1.0 - math.exp(-x / rho)) for x, p in zip(xs, ps))
    return optimize.brentq(lambda s: (1.0 - math.exp(-s / rho)) - eu, min(xs) - 1.0, max(xs) + 1.0, xtol=1e-13, rtol=1e-14)


@pytest.mark.parametrize("xs, ps, rho", [([0.0, 100.0], [0.5, 0.5], 100.0), ([200.0, -100.0], [0.5, 0.5], 200.0), ([200.0, -100.0], [0.5, 0.5], 555.8), ([-70.0, 70.0], [0.6, 0.4], 50.0),
                                          ([10.0, -80.0, 120.0], [0.2, 0.3, 0.5], 80.0)])
def test_sicherheitsaequivalent_gegen_brentq_auf_dem_nutzen(xs, ps, rho):
    assert U.ce(np.array(xs), np.array(ps), rho) == pytest.approx(_ce_brentq(xs, ps, rho), abs=1e-9)


def test_sicherheitsaequivalent_eigene_fassung_der_negativen_exponentialfunktion():
    """Gegenprobe wie in der Nachbar-Demo für das Teilen von Risiko, hier als eigene Fassung: u(x) = −exp(−γx) mit γ = 1/ρ, Umkehrung s = −ln(−EU)/γ (andere Nutzenfunktion, gleiche Wahl)."""
    gamma = 0.005
    xs, ps = [0.0, 200.0], [0.5, 0.5]
    eu = sum(p * -math.exp(-gamma * x) for x, p in zip(xs, ps))
    assert -math.log(-eu) / gamma == pytest.approx(75.97709860834449, abs=1e-9)
    assert U.ce(np.array(xs), np.array(ps), 1 / gamma) == pytest.approx(-math.log(-eu) / gamma, abs=1e-9)


@pytest.mark.parametrize("mu, sg, rho", [(50.0, 40.0, 100.0), (0.0, 25.0, 30.0), (10.0, 20.0, 100.0)])
def test_normalverteilung_quadratur_gegen_formel(mu, sg, rho):
    f = lambda t: math.exp(-t / rho) * math.exp(-0.5 * ((t - mu) / sg) ** 2) / (sg * math.sqrt(2 * math.pi))  # noqa: E731
    val, _ = integrate.quad(f, mu - 12 * sg, mu + 12 * sg, epsabs=1e-14, epsrel=1e-13, limit=400)
    assert -rho * math.log(val) == pytest.approx(mu - sg ** 2 / (2 * rho), abs=1e-8)


def test_ce_in_grenzen_klein_und_gross_gegen_geschlossene_form(cap):
    x, p, _ = cap
    assert U.ce(x, p, 1e7) == pytest.approx(50.0 - 22500 / 2e7, abs=1e-6)      # ρ → ∞: Erwartungswert minus σ²/(2ρ) = 49,998875
    assert U.ce(x, p, 0.5) == pytest.approx(-100.0 - 0.5 * math.log(0.5), abs=1e-9)    # ρ → 0: Minimum − ρ ln p_min (die Beimischung des guten Zustands ist e^(−600) und verschwindet)


# --- Schwelle p*, Potenz ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [50.0, 100.0, 300.0, 555.0, 1000.0, 5000.0])
def test_p_stern_gegen_gleiches_sicherheitsaequivalent(rho):
    ps = U.p_stern([200.0, -100.0], [30.0, 30.0], rho)
    assert 0 < ps < 1
    assert U.ce(np.array([200.0, -100.0]), np.array([ps, 1 - ps]), rho) == pytest.approx(30.0, abs=1e-9)


def test_potenz_kippvermoegen_exakt_in_brueche_und_gegen_brentq(cap):
    x, p, k = cap
    assert (F(1045, 2) + 30) ** 2 == (F(1045, 2) + 200) * (F(1045, 2) - 100)                       # log-Nutzen: geometrisches Mittel, w* = 522,5
    assert F(1, 2) * (1 / (F(1075) + 200) + 1 / (F(1075) - 100)) == 1 / (F(1075) + 30)             # γ = 2: harmonisches Mittel, w* = 1075
    for gamma, soll in ((1.0, 522.5), (2.0, 1075.0)):
        f = lambda w: float(W.ce_pow(x, p, w, gamma)) - k  # noqa: E731
        assert optimize.brentq(f, 100.0 + 1e-6, 1e9, xtol=1e-10) == pytest.approx(soll, abs=1e-6)
        assert W.kipp_vermoegen(x, p, k, gamma) == pytest.approx(soll, abs=1e-6)
    assert W.kipp_vermoegen(x, p, k, 4.0) == pytest.approx(2184.95, abs=0.01)
    for w, soll in ((1000, -1), (2000, 1)):                                                         # γ = 2, exakt: bei 1000 gewinnt K, bei 2000 G
        eu_g = -F(1, 2) * (1 / F(w + 200) + 1 / F(w - 100))
        eu_k = -1 / F(w + 30)
        assert ((eu_g > eu_k) - (eu_g < eu_k)) == soll


def test_delta_eigenschaft_exakt_in_brueche():
    """EU_G(w) − EU_K(w) = z^(w/10) (EU_G(0) − EU_K(0)) für u = 1 − z^(x/10): das Vorzeichen hängt nicht vom Vermögen w ab (nicht die 1,8e-12 der Gleitkomma-Rundung)."""
    for z in (F(49, 50), F(99, 100)):
        def eu(xs, ps, w):
            return sum(F(p) * (1 - z ** ((w + x) // 10)) for x, p in zip(xs, ps))
        d0 = eu([200, -100], [F(1, 2)] * 2, 0) - eu([30], [F(1)], 0)
        for w in (10, 100, 300, 10 ** 5):
            assert eu([200, -100], [F(1, 2)] * 2, w) - eu([30], [F(1)], w) == z ** (w // 10) * d0


# --- Zufallsprobleme ---------------------------------------------------------------------------------------------------------------------------

def _ce_naiv(xs, p, rho):
    return -rho * math.log(sum(pi * math.exp(-xi / rho) for xi, pi in zip(xs, (p, 1 - p))))


def test_wahl_der_zufallsprobleme_gegen_problem_fuer_problem_rechnung():
    P, A, dom = S.zufallsprobleme(101, 300)
    for rho in (20.0, 100.0, 500.0):
        wahl = U.wahl_vorzeichen(P, A, rho)
        for i in range(300):
            d = _ce_naiv(A[i, 1], P[i, 0], rho) - _ce_naiv(A[i, 0], P[i, 0], rho)
            if abs(d) > 1e-6:                                      # Gleichstände (Bibliothek, letzte Stelle) nicht werten
                assert wahl[i] == (1 if d > 0 else -1), (rho, i)
    assert np.array_equal(dom, [bool(all(A[i, 0] >= A[i, 1]) or all(A[i, 1] >= A[i, 0])) for i in range(300)])


def test_rho_stern_viele_findet_einen_gleichstand_und_wechselpunkt():
    P, A, dom = S.zufallsprobleme(101, 3000)
    unten = U.wahl_vorzeichen(P, A, 0.5)
    ew = U.wahl_vorzeichen(P, A, math.inf)
    idx = np.where(~dom & (unten != ew) & (unten != 0) & (ew != 0))[0][:150]
    rs = U.rho_stern_viele(P[idx], A[idx], unten[idx])
    for j, i in enumerate(idx):
        d = _ce_naiv(A[i, 1], P[i, 0], rs[j]) - _ce_naiv(A[i, 0], P[i, 0], rs[j])
        assert abs(d) < 1e-6, (i, rs[j], d)                      # bei ρ* sind beide Alternativen gleich gut
        assert 0.5 <= rs[j] <= 1e7


def test_skaleninvarianz_der_wahl_in_zufallsproblemen_hoechstens_zwei_abweichungen():
    P, A, dom = S.zufallsprobleme(101, 20000)
    nd = ~dom
    w1 = U.wahl_vorzeichen(P, A, 100.0)
    w2 = U.wahl_vorzeichen(P, A * 10.0, 1000.0)
    assert int(np.sum(w1[nd] != w2[nd])) <= 2                    # Gleitkomma-Gleichstände; in der früheren Messung 0 von 99 575

"""Kern: Exponentialnutzen u(x) = 1 - exp(-x/rho), Sicherheitsäquivalent, Risikoprämie, Normalnäherung, Kipp-Risikotoleranz rho* und Schwellenwahrscheinlichkeit.

Auszahlungen x in Geldeinheiten (GE), positiv = Gewinn; rho > 0 ist die Risikotoleranz in GE. Das Sicherheitsäquivalent einer Lotterie (x_i mit p_i) ist
SÄ = -rho ln sum_i p_i exp(-x_i / rho); für rho -> unendlich ist es der Erwartungswert, für rho -> 0 das Minimum. Nur numpy und die Standardbibliothek.

Felder: `x` und `p` haben die Zustände auf der letzten Achse; mehrere Probleme liegen auf den vorderen Achsen (alles vektorisiert).
Entscheidungen mit |Differenz der Sicherheitsäquivalente| unter `TIE` zählen als gleich (nicht als Wahl): exp und log kommen aus der Bibliothek des Systems und
können sich in der letzten Binärstelle unterscheiden."""

import math

import numpy as np

TIE = 1e-9
RHO_GRID_LO, RHO_GRID_HI, RHO_GRID_N = 0.5, 1e5, 500
RHO_GRID = np.exp(np.linspace(math.log(RHO_GRID_LO), math.log(RHO_GRID_HI), RHO_GRID_N))   # Gitter der Kipppunkt-Suche (logarithmisch, 500 Punkte)
RHO_MAX = 1e12                                                                               # darüber gilt rho* als „praktisch unendlich“
_MODERAT = 3.0                                                                               # |x/rho| bis hierhin: log1p/expm1 (stabil bei großem rho), darüber Verschiebung (stabil, wenn Σ p e^a nahe 0 liegt)


def ce(x, p, rho):
    """Sicherheitsäquivalent; rho = inf gibt den Erwartungswert. `rho` ist ein Skalar oder hat die Form der vorderen Achsen von `x`.

    Rechnung: für moderate x/rho über log1p und expm1 (stabil bei großem rho, wo x/rho winzig ist), sonst mit Verschiebung um das Maximum (kein Überlauf bei
    kleinem rho). Zustände mit p = 0 zählen nicht."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    if np.ndim(rho) == 0 and math.isinf(float(rho)):
        return np.sum(p * x, axis=-1)
    r = np.asarray(rho, dtype=float)
    aktiv = p > 0
    a = np.where(aktiv, -x / (r[..., None] if r.ndim else r), 0.0)      # Zustände mit p = 0 zählen nicht und dürfen nichts überlaufen lassen
    amax = np.max(np.where(aktiv, a, -np.inf), axis=-1, keepdims=True)
    amin = np.min(np.where(aktiv, a, np.inf), axis=-1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        moderat = -r * np.log1p(np.sum(p * np.expm1(a), axis=-1))
        verschoben = -r * (amax[..., 0] + np.log(np.sum(p * np.exp(a - amax), axis=-1)))
    ok = (amax[..., 0] <= _MODERAT) & (amin[..., 0] >= -_MODERAT)
    out = np.where(ok, moderat, verschoben)
    return out if out.ndim else float(out)


def erwartung(x, p):
    return np.sum(np.asarray(p, dtype=float) * np.asarray(x, dtype=float), axis=-1)


def varianz(x, p):
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    mu = np.sum(p * x, axis=-1, keepdims=True)
    return np.sum(p * (x - mu) ** 2, axis=-1)


def risikopraemie(x, p, rho):
    """Erwartungswert minus Sicherheitsäquivalent (nie negativ)."""
    return erwartung(x, p) - ce(x, p, rho)


def normal_naeherung(x, p, rho):
    """Näherung zweiter Ordnung SÄ ≈ mu - sigma^2 / (2 rho); exakt bei Normalverteilung, sonst eine Näherung."""
    return erwartung(x, p) - varianz(x, p) / (2.0 * rho)


def spanne(*felder):
    """Größte minus kleinste Auszahlung über alle übergebenen Felder."""
    alle = np.concatenate([np.asarray(f, dtype=float).ravel() for f in felder])
    return float(alle.max() - alle.min())


def vorzeichen(d):
    """+1, -1 oder 0 (|d| < TIE) je Eintrag."""
    d = np.asarray(d, dtype=float)
    return np.where(np.abs(d) < TIE, 0, np.sign(d)).astype(np.int8)


def bisect_log(f, lo, hi, iterationen=200):
    """Nullstelle einer Funktion f auf (lo, hi), lo > 0, mit f(lo) und f(hi) verschiedenen Vorzeichens, durch Halbieren von ln(rho)."""
    flo = f(lo)
    a, b = math.log(lo), math.log(hi)
    for _ in range(iterationen):
        m = 0.5 * (a + b)
        fm = f(math.exp(m))
        if (fm < 0) == (flo < 0):
            a = m
        else:
            b = m
        if b - a < 1e-15:
            break
    return math.exp(0.5 * (a + b))


def rho_stern_sicher(x, p, k):
    """Kipp-Risikotoleranz gegen eine SICHERE Alternative mit Auszahlung k: das rho, bei dem SÄ(x) = k ist.

    Rückgabe (rho_stern, status); status: "ok", "k_bleibt" (k >= Erwartungswert: die sichere Alternative gewinnt bei jedem rho), "lotterie_immer" (k <= Minimum:
    die Lotterie ist bei jedem rho mindestens so gut) oder "unendlich" (rho_stern über RHO_MAX). Das Sicherheitsäquivalent wächst mit rho; darum gibt es höchstens eine Nullstelle."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    mn = float(np.min(np.where(p > 0, x, np.inf)))
    if k >= float(erwartung(x, p)) - TIE:
        return None, "k_bleibt"
    if k <= mn + TIE:
        return None, "lotterie_immer"
    sp = max(float(np.max(x[p > 0]) - mn), 1.0)

    def f(r):
        return float(ce(x, p, r)) - k

    lo = sp
    while f(lo) >= 0 and lo > sp * 1e-12:
        lo /= 4.0
    hi = sp
    while f(hi) < 0:
        hi *= 4.0
        if hi > RHO_MAX:
            return None, "unendlich"
    return bisect_log(f, lo, hi), "ok"


def normal_rho_stern(sp, vorsprung):
    """Näherung rho* ≈ Spanne² / (8 · Vorsprung) für G = (hoch | tief) mit p = 1/2 gegen eine sichere Alternative (Näherung zweiter Ordnung)."""
    return sp ** 2 / (8.0 * vorsprung)


def p_stern(x, y, rho):
    """Schwellenwahrscheinlichkeit des ersten Zustands für zwei Zustände: bei p = p_stern sind SÄ(x) und SÄ(y) gleich (die Gleichung ist linear in p).

    x = (x1, x2), y = (y1, y2); rho = inf gibt die Schwelle des Erwartungswerts. Rückgabe None, wenn es keine eindeutige Schwelle gibt."""
    x1, x2 = float(x[0]), float(x[1])
    y1, y2 = float(y[0]), float(y[1])
    if math.isinf(rho):
        nenner = (x1 - x2) - (y1 - y2)
        return None if nenner == 0 else (y2 - x2) / nenner
    try:
        a1, a2, b1, b2 = (math.exp(-v / rho) for v in (x1, x2, y1, y2))
    except OverflowError:
        return None
    nenner = (a1 - a2) - (b1 - b2)
    return None if nenner == 0 else (b2 - a2) / nenner


def wechsel_zaehlen(ds):
    """Vorzeichenwechsel je Zeile von ds (Probleme x Gitter); Beträge unter TIE gelten als Gleichstand und werden übersprungen."""
    ds = np.atleast_2d(ds)
    s = np.sign(ds)
    s[np.abs(ds) < TIE] = 0.0
    wechsel = np.zeros(ds.shape[0], dtype=int)
    letzt = s[:, 0].copy()
    for j in range(1, ds.shape[1]):
        cur = s[:, j]
        neu = (cur != 0) & (letzt != 0) & (cur != letzt)
        wechsel += neu
        letzt = np.where(cur != 0, cur, letzt)
    return wechsel


def kipp_punkte(x, y, p, gitter=RHO_GRID):
    """Alle Kipppunkte rho zwischen zwei Alternativen mit gleichen Wahrscheinlichkeiten p: Vorzeichenwechsel von SÄ(x) - SÄ(y) auf dem Gitter, je Wechsel eine Bisektion.

    Rückgabe (Liste der rho, Vorzeichen am unteren Gitterrand, Vorzeichen am oberen Gitterrand); +1 heißt x besser. Eng beieinanderliegende Wechsel zwischen
    zwei Gitterpunkten kann das Gitter übersehen: die Liste ist eine Untergrenze der Zahl der Kipppunkte."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    g = len(gitter)
    d = ce(np.broadcast_to(x, (g, len(x))), np.broadcast_to(p, (g, len(x))), gitter) - ce(np.broadcast_to(y, (g, len(y))), np.broadcast_to(p, (g, len(y))), gitter)
    s = vorzeichen(d)
    punkte = []
    letzt_i = None
    for i in range(g):
        if s[i] == 0:
            continue
        if letzt_i is not None and s[i] != s[letzt_i]:
            punkte.append(bisect_log(lambda r: float(ce(x, p, r) - ce(y, p, r)), float(gitter[letzt_i]), float(gitter[i])))
        letzt_i = i
    nichtnull = [int(v) for v in s if v != 0]
    return punkte, (nichtnull[0] if nichtnull else 0), (nichtnull[-1] if nichtnull else 0)


def wahl_vorzeichen(P, A, rho):
    """Je Problem (P: Probleme x Zustände, A: Probleme x 2 x Zustände): +1 wenn Alternative 1 das größere Sicherheitsäquivalent hat, -1 Alternative 0, 0 gleich."""
    return vorzeichen(ce(A[:, 1, :], P, rho) - ce(A[:, 0, :], P, rho))


def wahl_naeherung(P, A, rho):
    """Wie `wahl_vorzeichen`, aber mit der Näherung mu - sigma²/(2 rho) statt des exakten Sicherheitsäquivalents."""
    return vorzeichen(normal_naeherung(A[:, 1, :], P, rho) - normal_naeherung(A[:, 0, :], P, rho))


def rho_stern_viele(P, A, w_unten, rho_lo=0.5, rho_hi=1e7, schritte=40):
    """Kipp-Risikotoleranz je Problem (2 Alternativen), vektorisiert: Bisektion von ln(rho) zwischen rho_lo und rho_hi.

    Voraussetzung: bei rho_lo wählt jedes Problem `w_unten` (+1/-1) und bei rho_hi die andere Alternative (ungerade Zahl von Wechseln dazwischen);
    gefunden wird dann ein Kipppunkt (bei mehreren irgendeiner von ihnen)."""
    n = len(P)
    lo = np.full(n, math.log(rho_lo))
    hi = np.full(n, math.log(rho_hi))
    for _ in range(schritte):
        mid = 0.5 * (lo + hi)
        v = vorzeichen(ce(A[:, 1, :], P, np.exp(mid)) - ce(A[:, 0, :], P, np.exp(mid)))
        wie_unten = v == w_unten
        lo = np.where(wie_unten, mid, lo)
        hi = np.where(wie_unten, hi, mid)
    return np.exp(0.5 * (lo + hi))

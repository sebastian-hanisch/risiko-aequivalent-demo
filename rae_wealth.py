"""Potenznutzen auf dem Endvermögen (Gegenstück zum Exponentialnutzen): u(v) = v^(1-gamma) / (1-gamma), gamma = 1: ln v, mit v = w + x.

Der Exponentialnutzen hängt nicht vom Anfangsvermögen w ab (das Sicherheitsäquivalent der Zuwächse bleibt gleich, wenn man w addiert: Delta-Eigenschaft); der
Potenznutzen tut es: das Sicherheitsäquivalent der Zuwächse x ist CE_pow = u^(-1)(E u(w + x)) - w und wächst mit w. Verlangt w + x > 0 in jedem Zustand mit p > 0."""

import math

import numpy as np

import rae_utility as U


def definiert(x, p, w):
    """Ist w + x in jedem Zustand mit p > 0 positiv (der Potenznutzen ist sonst nicht definiert)?"""
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    return bool(np.all(np.where(p > 0, w + x, 1.0) > 0))


def ce_pow(x, p, w, gamma):
    """Sicherheitsäquivalent der Zuwächse bei Potenznutzen auf dem Endvermögen; NaN, wo w + x nicht positiv ist. Felder wie in `rae_utility.ce`."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    v = w + x
    with np.errstate(invalid="ignore", divide="ignore"):
        if abs(gamma - 1.0) < 1e-12:
            wert = np.exp(np.sum(np.where(p > 0, p * np.log(np.where(v > 0, v, np.nan)), 0.0), axis=-1))
        else:
            wert = np.sum(np.where(p > 0, p * np.where(v > 0, v, np.nan) ** (1.0 - gamma), 0.0), axis=-1) ** (1.0 / (1.0 - gamma))
    out = wert - w
    return out if np.ndim(out) else float(out)


def kipp_vermoegen(x, p, k, gamma):
    """Vermögen w*, bei dem Potenznutzen die Lotterie x und die sichere Auszahlung k gleich bewertet (darüber gewinnt die Lotterie, wenn ihr Erwartungswert über k liegt).

    Rückgabe None, wenn k nicht zwischen Minimum und Erwartungswert der Lotterie liegt (dann gibt es keinen Kipppunkt)."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    mn = float(np.min(np.where(p > 0, x, np.inf)))
    if not (mn + U.TIE < k < float(U.erwartung(x, p)) - U.TIE):
        return None
    w_min = -mn                                    # darunter ist der Nutzen nicht definiert
    return w_min + U.bisect_log(lambda t: float(ce_pow(x, p, w_min + t, gamma)) - k, 1e-9, 1e12)


def kalibriertes_rho(x, p, w, gamma):
    """Risikotoleranz rho, bei der der Exponentialnutzen dem Bezugsrisiko x dasselbe Sicherheitsäquivalent gibt wie der Potenznutzen bei Vermögen w."""
    ziel = float(ce_pow(x, p, w, gamma))
    sp = max(U.spanne(x), 1.0)
    return U.bisect_log(lambda r: float(U.ce(x, p, r)) - ziel, sp * 1e-3, 1e9)


def wahl_potenz(P, A, w, gamma):
    """Wie `rae_utility.wahl_vorzeichen`, aber mit Potenznutzen bei Vermögen w; verlangt w größer als das Betragsminimum aller Auszahlungen."""
    return U.vorzeichen(ce_pow(A[:, 1, :], P, w, gamma) - ce_pow(A[:, 0, :], P, w, gamma))

"""Szenarien: der Kapazitätsfall (ausgedachtes Zahlenbeispiel, kein Fall) und die Zufallsprobleme.

Der Kapazitätsfall in Geldeinheiten (GE): G (groß bauen) bringt +200 GE bei hoher und -100 GE bei niedriger Nachfrage, je mit Wahrscheinlichkeit 1/2; K (klein
oder mieten) bringt sicher +30 GE. Die Auszahlungen sind ausgedacht: das Beispiel ist ein Modell, kein Fall mit Daten.

Zufallsprobleme (Generator der Messungen): zwei Alternativen, zwei Zustände, P(Zustand 1) gleichverteilt in 0,1 bis 0,9, jede Auszahlung ganzzahlig gleichverteilt in
-100 bis 200 GE. Dominiert heißt: eine Alternative ist in beiden Zuständen mindestens so gut wie die andere. Der Zufall kommt nur aus SplitMix64 (`rae_rng`, ganzzahlig);
Problem i hat denselben Inhalt, wie groß die Zahl der Probleme auch ist (die ersten n Probleme eines Stroms sind die Probleme n kleiner)."""

import numpy as np

import rae_rng as R
import rae_utility as U

# --- Kapazitätsfall -------------------------------------------------------------------------------------------------------------------------
CAP_HOCH, CAP_TIEF, CAP_SICHER, CAP_P = 200.0, -100.0, 30.0, 0.5
CAP_X = (CAP_HOCH, CAP_TIEF)
CAP_P2 = (CAP_P, 1.0 - CAP_P)

# --- Zufallsprobleme ------------------------------------------------------------------------------------------------------------------------
X_LO, X_HI = -100, 200            # Auszahlungen gleichverteilt, ganzzahlig, beide Grenzen eingeschlossen
P_LO, P_HI = 0.1, 0.9
_SPALTEN = 5                      # je Problem: 1 Zahl für p, 4 Zahlen für die Auszahlungen


def zufallsprobleme(seed: int, n: int):
    """(P, A, dominiert): P (n, 2) Wahrscheinlichkeiten, A (n, 2, 2) Auszahlungen [Problem, Alternative, Zustand], dominiert (n,) bool."""
    roh = R.stream_u64(seed, _SPALTEN * n).reshape(n, _SPALTEN)
    u = (roh[:, 0] >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    p = P_LO + (P_HI - P_LO) * u
    P = np.stack([p, 1.0 - p], axis=1)
    A = ((roh[:, 1:] % np.uint64(X_HI - X_LO + 1)).astype(np.int64) + X_LO).astype(float).reshape(n, 2, 2)
    return P, A, dominiert(A)


def dominiert(A):
    """Eine Alternative ist in beiden Zuständen mindestens so gut wie die andere."""
    return np.all(A[:, 0, :] >= A[:, 1, :], axis=1) | np.all(A[:, 1, :] >= A[:, 0, :], axis=1)


def zufallsprobleme_m(seed: int, n: int, m: int, art: str):
    """Zufallsprobleme mit m Zuständen für die Zählung der Kipppunkte. Wahrscheinlichkeiten Dirichlet(1,…,1) über die Abstände von m-1 sortierten Gleichverteilten.

    art "sicher": Lotterie A (m Auszahlungen) gegen eine sichere Auszahlung K; Rückgabe (P, A, K) mit A (n, m), K (n,).
    art "riskant": zwei Lotterien A und B mit denselben Wahrscheinlichkeiten; Rückgabe (P, A, B) mit A, B (n, m)."""
    if art not in ("sicher", "riskant"):
        raise ValueError(art)
    breit = (m - 1) + (m + 1 if art == "sicher" else 2 * m)
    roh = R.stream_u64(seed, breit * n).reshape(n, breit)
    schnitte = np.sort((roh[:, : m - 1] >> np.uint64(11)).astype(np.float64) / float(1 << 53), axis=1)
    kanten = np.concatenate([np.zeros((n, 1)), schnitte, np.ones((n, 1))], axis=1)
    P = np.diff(kanten, axis=1)
    zahlen = ((roh[:, m - 1:] % np.uint64(X_HI - X_LO + 1)).astype(np.int64) + X_LO).astype(float)
    if art == "sicher":
        return P, zahlen[:, :m], zahlen[:, m]
    return P, zahlen[:, :m], zahlen[:, m:]


def probleme_dominiert_m(A, B):
    return np.all(A >= B, axis=1) | np.all(B >= A, axis=1)

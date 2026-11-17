"""Konstanten: Regler-Grenzen, feste Annahmen, Zufalls-Seeds, Farben und Beispielszenarien (Schnellstart).

Die Auszahlungen des Kapazitätsfalls sind ausgedacht (Geldeinheiten GE): das Beispiel ist ein Modell, kein Fall mit Daten."""

GITHUB = "https://github.com/sebastian-hanisch/"   # Nachbar-Demos: GitHub-Links werden aus dem Namen gebaut


def repo_link(name):
    """Markdown-Link auf ein Demo-Repo."""
    return f"[{name}]({GITHUB}{name})"


# --- Regler ------------------------------------------------------------------------------------------------------------------------------
RHO_OPTIONS = (10, 20, 30, 50, 70, 100, 150, 200, 300, 400, 556, 700, 1000, 2000, 5000)   # Risikotoleranz in GE (die Stufe 556 ist ρ* des Kapazitätsfalls, gerundet)
DEFAULT_RHO = 200
X_MIN, X_MAX, X_STEP = -200.0, 400.0, 10.0           # Auszahlungen im ersten Experiment
A_MIN, A_MAX, A_STEP = -100.0, 200.0, 10.0           # Auszahlungen im zweiten Experiment (wie im Generator der Zufallsprobleme)
P_MIN, P_MAX, P_STEP = 0.05, 0.95, 0.05
FRAGE_G = (1.2, 1.5, 2.0, 3.0, 4.0, 10.0)            # „Gewinn g·L gegen Verlust L“
L_MIN, L_MAX, L_STEP = 10.0, 300.0, 10.0
W3_OPTIONS = (100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 10000)   # Anfangsvermögen in GE
GAMMA_OPTIONS = (1, 2, 4)
SCALE_OPTIONS = (0.1, 0.5, 1.0, 2.0, 10.0)
N4_OPTIONS = (10000, 20000, 50000, 100000, 200000)
W5_OPTIONS = (500, 1000, 5000)
SEED_MAX = 999999
DEFAULT_SEED = 101

# --- Beispiele ------------------------------------------------------------------------------------------------------------------------------
CAP_FALL2_VERLUST = 50.0                              # „Fall 2“: weniger Verlust (G = +200 | -50)
L_KURVE = tuple(range(10, 140, 3))       # Verluste der ρ*-Kurve (Familie G = (200 | -L) gegen 30 sicher, p = 1/2)
RHO_TABELLE = (50.0, 100.0, 200.0, 300.0, 555.8, 1000.0, 5000.0)     # Zeilen der Tabelle „Sicherheitsäquivalent über ρ“
RHO_STAFFEL = (10, 20, 50, 100, 300, 1000)            # Spalten der Wirkungstabelle (Experiment 4 und 5)
KAPAZITAETSFALL_RATIO = 1.853                         # ρ*/Spanne des Kapazitätsfalls (555,80 / 300), Marke im Histogramm

# --- Zufallsströme ------------------------------------------------------------------------------------------------------------------------
SEED_BAND = 20261116                                  # zweiter, fester Strom für das Band der Anteile (die Zählungen der README)
SEED_KIPP = (1301, 1302)                              # Ströme der vorgerechneten Kipppunkt-Zählung
N_KIPP = 40000                                        # Probleme je Zählung (vorgerechnet)
N_FROZEN_CAL = 200000                                 # Probleme der vorgerechneten Potenz-Kalibrierung
FROZEN_FILE = "data/rae_frozen.json"
SEED_SPLIT = 1_000_003                                # zweiter Strom der App = Zufalls-Seed + SEED_SPLIT (Band über zwei Ströme)

# --- Darstellung -------------------------------------------------------------------------------------------------------------------------
COLOR_A = "#2F6B65"
COLOR_B = "#c77700"
COLOR_LINE = "#4c78a8"
COLOR_RED = "#b5453a"
COLOR_GRID = "rgba(150,150,150,0.55)"
CHART_HEIGHT = 380

# --- Beispielszenarien (Schnellstart); jedes setzt die Regler seines Experiments, alle übrigen bleiben auf der Voreinstellung ----------------
_BASE = dict(
    rho=DEFAULT_RHO,
    x1=200.0, x2=-100.0, p1=0.5, fg=2.0, fl=100.0,
    a1=200.0, a2=-100.0, b1=30.0, b2=30.0, p2=0.5,
    w3=1000, g3=2,
    sk4=1.0, n4=200000, seed4=DEFAULT_SEED,
    g5=2, w5=1000,
)
PRESETS = {
    "Kapazitätsfall, ρ = 200": dict(_BASE),
    "Kipppunkt Fall 1 (ρ = 556)": dict(_BASE, rho=556),
    "Fall 2: weniger Verlust": dict(_BASE, a2=-CAP_FALL2_VERLUST),
    "Goldener Schnitt": dict(_BASE, b1=0.0, b2=0.0),
    "Zwei Kipppunkte (A gegen B)": dict(_BASE, rho=50, a1=-70.0, a2=70.0, b1=10.0, b2=-80.0, p2=0.6),
    "Vermögen: Potenz γ = 2": dict(_BASE, w3=2000, g3=2),
    "Zufallsprobleme ρ = 100": dict(_BASE, rho=100),
    "Skala ×10": dict(_BASE, rho=100, sk4=10.0),
}
PRESET_HELP = {
    "Kapazitätsfall, ρ = 200": "Voreinstellung: groß bauen (+200 GE bei hoher, −100 GE bei niedriger Nachfrage, je ½) gegen klein oder mieten (sicher +30 GE), Risikotoleranz ρ = 200 GE. Alle Zahlen sind ausgedacht.",
    "Kipppunkt Fall 1 (ρ = 556)": "Risikotoleranz knapp über dem Kipppunkt ρ* = 555,80 GE: groß bauen wird gerade gewählt (zweites Experiment).",
    "Fall 2: weniger Verlust": "Verlust 50 statt 100 GE: der Erwartungswert von G steigt von 50 auf 75, der Kipppunkt sinkt von 555,80 auf 158,05 GE; bei ρ = 200 wird G gewählt (zweites Experiment).",
    "Goldener Schnitt": "G (+200 | −100) gegen 0 sicher: der Kipppunkt ist ρ* = 100 / ln φ = 207,81 GE, bei ρ = 200 wird noch 0 gewählt (zweites Experiment).",
    "Zwei Kipppunkte (A gegen B)": "Zwei riskante Alternativen mit P(Zustand 1) = 0,6: A = (−70 | +70) ist im Mittel und im schlechtesten Fall besser als B = (+10 | −80), B gewinnt aber für ρ zwischen 28,87 und 72,18 GE (zweites Experiment).",
    "Vermögen: Potenz γ = 2": "Anfangsvermögen 2000 GE, Potenznutzen mit γ = 2: groß bauen gewinnt; der Exponentialnutzen bei ρ = 200 wählt klein (drittes Experiment).",
    "Zufallsprobleme ρ = 100": "200 000 Zufallsprobleme (Zufalls-Seed 101), ρ = 100 GE: in 13,6 % der nicht dominierten Probleme ändert die Risikoeinstellung die Wahl (viertes Experiment).",
    "Skala ×10": "Dieselben Zufallsprobleme mit zehnfachen Auszahlungen und ρ = 100 GE: derselbe Zahlenwert ρ, ein anderes Verhältnis zu den Auszahlungen (viertes Experiment).",
}

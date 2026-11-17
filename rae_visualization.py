"""Plotly-Figuren (alle Achsen fest, `fixedrange`, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import math

import numpy as np
import plotly.graph_objects as go

import rae_constants as C
import rae_scenario as S

_LAYOUT = dict(height=C.CHART_HEIGHT, margin=dict(l=55, r=20, t=45, b=50), legend=dict(orientation="h", y=-0.25), hovermode="closest")


def _fest(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _leer(text):
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(height=C.CHART_HEIGHT, margin=dict(l=20, r=20, t=20, b=20))
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    return fig


# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------

def build_nutzen(a):
    """Nutzenkurve (positiv affin auf 0 bis 1 zwischen kleinster und größter Auszahlung normiert: die Wahl ändert das nicht), Sehne, Erwartungsnutzen und Sicherheitsäquivalent."""
    x1, x2, p1, rho = a["x1"], a["x2"], a["p1"], a["rho"]
    if x1 == x2:
        return _leer("Sichere Auszahlung: keine Kurve zu zeigen")
    lo, hi = min(x1, x2), max(x1, x2)
    breite = hi - lo

    def v(x):
        x = np.asarray(x, dtype=float)
        return (np.expm1(-lo / rho) - np.expm1(-x / rho)) / (np.expm1(-lo / rho) - np.expm1(-hi / rho))

    xs = np.linspace(lo - 0.15 * breite, hi + 0.15 * breite, 200)
    eu = p1 * float(v(x1)) + (1 - p1) * float(v(x2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=v(xs), mode="lines", name="Nutzen u(x) = 1 − e^(−x/ρ), normiert", line=dict(color=C.COLOR_LINE, width=3)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 1], mode="lines", name="Sehne zwischen den beiden Auszahlungen", line=dict(color=C.COLOR_GRID, dash="dot")))
    fig.add_trace(go.Scatter(x=[x1, x2], y=[float(v(x1)), float(v(x2))], mode="markers", name="Auszahlungen", marker=dict(size=11, color=C.COLOR_B)))
    fig.add_trace(go.Scatter(x=[a["ew"]], y=[eu], mode="markers", name=f"Erwartungswert {a['ew']:.1f} GE, Erwartungsnutzen", marker=dict(size=11, color=C.COLOR_RED, symbol="diamond")))
    fig.add_trace(go.Scatter(x=[a["ew"], a["sae"]], y=[eu, eu], mode="lines", name="Prämie: vom Erwartungswert zum Sicherheitsäquivalent", line=dict(color=C.COLOR_RED, dash="dash")))
    fig.add_trace(go.Scatter(x=[a["sae"]], y=[eu], mode="markers", name=f"Sicherheitsäquivalent {a['sae']:.1f} GE", marker=dict(size=13, color=C.COLOR_A, symbol="star")))
    fig.update_layout(**_LAYOUT, xaxis_title="Auszahlung in GE", yaxis_title="Nutzen (normiert)", title="Nutzenkurve und Sicherheitsäquivalent")
    return _fest(fig)


# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------

def build_kipp(kurven, k):
    """Sicherheitsäquivalent von A und B über die Risikotoleranz (logarithmisch), Kipppunkte und aktuelles ρ markiert."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kurven["rho"], y=kurven["a"], mode="lines", name="Sicherheitsäquivalent A", line=dict(color=C.COLOR_A, width=3)))
    fig.add_trace(go.Scatter(x=kurven["rho"], y=kurven["b"], mode="lines", name="Sicherheitsäquivalent B", line=dict(color=C.COLOR_B, width=3)))
    for i, r in enumerate(k["punkte"]):
        if 10 <= r <= 10000:
            fig.add_vline(x=r, line=dict(color=C.COLOR_RED, dash="dash"), annotation_text=f"ρ* = {r:.1f}".replace(".", ","), annotation_position="top")
    fig.add_vline(x=k["rho"], line=dict(color=C.COLOR_GRID, dash="dot"), annotation_text="Ihr ρ", annotation_position="bottom")
    fig.update_xaxes(type="log", title="Risikotoleranz ρ in GE (logarithmisch)", tickvals=[10, 30, 100, 300, 1000, 3000, 10000], ticktext=["10", "30", "100", "300", "1000", "3000", "10 000"])
    fig.update_layout(**_LAYOUT, yaxis_title="Sicherheitsäquivalent in GE", title="Wo wechselt die Wahl?")
    return _fest(fig)


def build_rho_kurve(zeilen, faelle):
    """ρ* über den Erwartungswert-Vorsprung (Familie G = (200 | −L) gegen 30 sicher): Messung und Näherung Spanne²/(8·Vorsprung); Fall 1 und Fall 2 markiert."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[z["vorsprung"] for z in zeilen], y=[z["rho_stern"] for z in zeilen], mode="lines", name="ρ* (exakt)", line=dict(color=C.COLOR_A, width=3)))
    fig.add_trace(go.Scatter(x=[z["vorsprung"] for z in zeilen], y=[z["naeherung"] for z in zeilen], mode="lines", name="Näherung Spanne²/(8·Vorsprung)", line=dict(color=C.COLOR_GRID, dash="dash")))
    fig.add_trace(go.Scatter(x=[f["vorsprung"] for f in faelle], y=[f["rho_stern"] for f in faelle], mode="markers+text", text=[f["name"] for f in faelle], textposition="top right",
                             name="Fall 1 und Fall 2", marker=dict(size=12, color=C.COLOR_RED)))
    fig.update_yaxes(type="log", title="Kipp-Risikotoleranz ρ* in GE (logarithmisch)", tickvals=[100, 300, 1000, 3000, 10000, 30000], ticktext=["100", "300", "1000", "3000", "10 000", "30 000"])
    fig.update_xaxes(title="Vorsprung des Erwartungswerts von G gegenüber K in GE (nach links kleiner)", autorange="reversed")
    fig.update_layout(**_LAYOUT, title="Je kleiner der Vorsprung, desto größer ρ*")
    return _fest(fig)


# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------

def build_vermoegen(kurven, v, kipp_tab):
    fig = go.Figure()
    farben = {1: "#4c78a8", 2: "#7a3fb0", 4: "#c77700"}
    for g, y in kurven["kurven"].items():
        fig.add_trace(go.Scatter(x=kurven["w"], y=y, mode="lines", name=f"Potenznutzen γ = {g}", line=dict(color=farben[g], width=3 if g == v["gamma"] else 1.5)))
    fig.add_hline(y=v["ce_exp"], line=dict(color=C.COLOR_A, width=3), annotation_text=f"Exponentialnutzen (ρ = {v['rho']:.0f} GE): unabhängig vom Vermögen", annotation_position="bottom right")
    fig.add_hline(y=S.CAP_SICHER, line=dict(color=C.COLOR_RED, dash="dash"), annotation_text="K = 30 GE sicher", annotation_position="top left")
    if v["definiert"]:
        fig.add_vline(x=v["w"], line=dict(color=C.COLOR_GRID, dash="dot"), annotation_text="Ihr Vermögen", annotation_position="top")
    fig.update_xaxes(type="log", title="Anfangsvermögen w in GE (logarithmisch)", tickvals=[100, 300, 1000, 3000, 10000], ticktext=["100", "300", "1000", "3000", "10 000"])
    fig.update_yaxes(title="Sicherheitsäquivalent von G in GE", range=[-100, 55])
    fig.update_layout(**_LAYOUT, title="Sicherheitsäquivalent von G über das Vermögen")
    return _fest(fig)


# --- Experiment 4 und 5 ---------------------------------------------------------------------------------------------------------------------

def build_staffel(z):
    """Anteil mit anderer Wahl als beim Erwartungswert über ρ, zwei Zufallsströme."""
    rhos = list(C.RHO_STAFFEL)
    fig = go.Figure()
    for name, s, farbe in (("Strom 1", z["a"], C.COLOR_A), ("Strom 2", z["b"], C.COLOR_B)):
        fig.add_trace(go.Scatter(x=rhos, y=[100 * s["staffel"][r] for r in rhos], mode="lines+markers", name=f"{name} (Zufalls-Seed {s['seed']})", line=dict(color=farbe, width=3)))
    fig.update_xaxes(type="log", title="Risikotoleranz ρ in GE (logarithmisch; Auszahlungen ×1)", tickvals=rhos, ticktext=[str(r) for r in rhos])
    fig.update_yaxes(title="Anteil mit anderer Wahl in %", rangemode="tozero")
    fig.update_layout(**_LAYOUT, title="Wie oft ändert ρ die Wahl?")
    return _fest(fig)


def build_skalen(z):
    sk = list(C.SCALE_OPTIONS)
    fig = go.Figure()
    for name, s, farbe in (("Strom 1", z["a"], C.COLOR_A), ("Strom 2", z["b"], C.COLOR_B)):
        fig.add_trace(go.Bar(x=[f"×{str(c).replace('.', ',')}".replace(",0", "") for c in sk], y=[100 * s["skalen"][c] for c in sk], name=f"{name} (Zufalls-Seed {s['seed']})", marker_color=farbe))
    fig.update_xaxes(title=f"Auszahlungen mit diesem Faktor multipliziert, ρ = {z['a']['rho']:.0f} GE fest")
    fig.update_yaxes(title="Anteil mit anderer Wahl in %", rangemode="tozero")
    fig.update_layout(**_LAYOUT, barmode="group", title="Dieselbe Zahl ρ, andere Skala")
    return _fest(fig)


def build_verhaeltnis(z):
    """Verteilung von ρ*/Spanne (logarithmisch) in den Problemen, in denen Maximin und Erwartungswert verschieden wählen; Marke: Kapazitätsfall."""
    q = np.asarray(z["a"]["verhaeltnis"]["werte"], dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=np.log10(np.clip(q, 1e-3, 1e3)), nbinsx=50, marker_color=C.COLOR_A, name="Probleme"))
    fig.add_vline(x=math.log10(C.KAPAZITAETSFALL_RATIO), line=dict(color=C.COLOR_RED, dash="dash"), annotation_text="Kapazitätsfall 1,85", annotation_position="top")
    fig.add_vline(x=math.log10(z["a"]["verhaeltnis"]["median"]), line=dict(color=C.COLOR_GRID, dash="dot"), annotation_text="Median", annotation_position="top left")
    ticks = [-2, -1, 0, 1, 2]
    fig.update_xaxes(title="ρ*/Spanne (logarithmisch)", tickvals=ticks, ticktext=["0,01", "0,1", "1", "10", "100"])
    fig.update_yaxes(title="Zahl der Probleme")
    fig.update_layout(**_LAYOUT, title="Wo liegt ρ* im Verhältnis zur Spanne?", showlegend=False)
    return _fest(fig)

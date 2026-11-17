"""Risikoeinstellung: Sicherheitsäquivalent und Kipppunkt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Entscheider mit Exponentialnutzen bewertet eine Lotterie mit ihrem Sicherheitsäquivalent; fünf Experimente am ausgedachten Kapazitätsfall (Geldeinheiten GE):
das Sicherheitsäquivalent von Hand, die Kipp-Risikotoleranz ρ* und was die Zahl ρ bedeutet, das Anfangsvermögen (Exponential gegen Potenz), wie oft die Risikoeinstellung
die Wahl in Zufallsproblemen ändert und ob eine Näherung reicht.
Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import rae_constants as C
import rae_evaluation as ev
import rae_scenario as S
from rae_evaluation import de, pct, zahl
from rae_layout import dunkles_design, kopf_css
from rae_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from rae_visualization import build_kipp, build_nutzen, build_rho_kurve, build_skalen, build_staffel, build_verhaeltnis, build_vermoegen

st.set_page_config(page_title="Risikoeinstellung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _lotterie(x1, x2, p1, rho):
    return ev.lotterie_analyse(x1, x2, p1, rho)


@st.cache_data(show_spinner=False)
def _kipp(a1, a2, b1, b2, p1, rho):
    return ev.kipp_analyse(a1, a2, b1, b2, p1, rho)


@st.cache_data(show_spinner=False)
def _kurven(a1, a2, b1, b2, p1):
    return ev.ce_kurven(a1, a2, b1, b2, p1)


@st.cache_data(show_spinner=False)
def _rho_kurve():
    return ev.rho_kurve()


@st.cache_data(show_spinner=False)
def _vermoegen_kurven():
    return ev.vermoegen_kurven()


@st.cache_data(show_spinner="Rechne die Zufallsprobleme (zwei Ströme, bis 200 000 Probleme, einige Sekunden) …")
def _zufall(seed, n, rho, skala):
    return ev.zufall_analyse(seed, n, rho, skala)


@st.cache_data(show_spinner="Rechne Näherung und Potenznutzen (zwei Ströme, bis 200 000 Probleme, einige Sekunden) …")
def _naeherung(seed, n, rho, gamma, w):
    return ev.naeherung_analyse(seed, n, rho, gamma, w)


@st.cache_data(show_spinner=False)
def _frozen():
    return ev.load_frozen()


def _kennzeichen(art_text):
    art, text = art_text
    getattr(st, art)(text.replace("ρ*", "ρ\\*"))


# --- Kopf und Einleitung ---------------------------------------------------------------------------------------------------------------------

st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⚖️ Risikoeinstellung: Sicherheitsäquivalent und Kipppunkt")
st.markdown(
    """
Ein Entscheider mit **Exponentialnutzen** gibt jeder Lotterie ihr **Sicherheitsäquivalent**: den sicheren Betrag, den er ihr gleichstellt. Es hängt an einer einzigen Zahl, der **Risikotoleranz ρ** in Geldeinheiten (GE).
Am **Kapazitätsfall** (groß bauen: +200 GE bei hoher, −100 GE bei niedriger Nachfrage, je ½; klein oder mieten: sicher +30 GE) zeigt die Demo, ab welchem ρ die Wahl kippt (die **Kipp-Risikotoleranz ρ\\***),
warum die Zahl ρ allein nicht sagt, was sie für eine Wahl heißt (das hängt an der Spanne und am Vorsprung der Auszahlungen), und wie oft die Risikoeinstellung die Wahl in Zufallsproblemen überhaupt ändert.
**Die Auszahlungen sind ausgedacht: das Beispiel ist ein Modell, kein Fall mit Daten.**

Weiter unten: die **Kernfragen**, fünf Experimente, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/risiko-aequivalent-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich vor, Sie wählen zwischen einem sicheren Betrag und einem Wagnis, das im Mittel mehr bringt, aber auch einen Verlust bringen kann. Wie viel ist das Wagnis wert? Die Demo antwortet mit dem **Sicherheitsäquivalent**: dem sicheren Betrag, den ein vorsichtiger Entscheider dem Wagnis gleichstellt. Wie vorsichtig er ist, steckt in einer Zahl, der **Risikotoleranz** in Geldeinheiten: Je größer sie ist, desto weniger zählt die Vorsicht. Bei einem bestimmten Wert, dem **Kipppunkt**, wechselt seine Wahl. Dieselbe Zahl bedeutet bei anderen Auszahlungen etwas anderes. Die Demo zeigt das im ausgedachten Kapazitätsfall und zählt in Zufallsproblemen, wie oft die Vorsicht die Wahl ändert. Die Aussagen gelten in diesem Rechenmodell, nicht in einem echten Betrieb.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/risiko-aequivalent-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert das Sicherheitsäquivalent", expanded=True):
    st.markdown(
        """
1. **Nutzen.** Jede Auszahlung x (in GE, Gewinn positiv) bekommt den Nutzen u(x) = 1 − e^(−x/ρ). Je kleiner ρ, desto stärker krümmt sich die Kurve und desto schwerer wiegt ein möglicher Verlust.
2. **Erwartungsnutzen.** Der Nutzen jeder Auszahlung wird mit seiner Wahrscheinlichkeit gewichtet und summiert.
3. **Sicherheitsäquivalent (SÄ).** Der sichere Betrag, dessen Nutzen dem Erwartungsnutzen gleicht: SÄ = −ρ ln E[e^(−X/ρ)]. Die **Risikoprämie** ist Erwartungswert minus SÄ und nie negativ. Für ρ → ∞ ist das SÄ der Erwartungswert, für ρ → 0 die schlechteste Auszahlung.
4. **Wahl und Kipppunkt.** Gewählt wird die Alternative mit dem größeren SÄ. Die **Kipp-Risikotoleranz ρ\\*** ist das ρ, bei dem zwei Alternativen gleich gut sind. Gegen eine *sichere* Alternative fand die Zählung nie mehr als einen solchen Punkt; zwischen zwei *riskanten* kann es mehrere geben (beides unten gezählt).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Risikoeinstellung**")
    rho = st.select_slider(
        "Risikotoleranz ρ (GE)", options=C.RHO_OPTIONS, key="rho_select",
        help="Je kleiner ρ, desto risikoscheuer der Entscheider (konstante absolute Risikoaversion 1/ρ). Gilt für alle fünf Experimente. Die Zahl legt die Nutzenfunktion fest; was sie für eine Wahl heißt, hängt am Verhältnis zur Spanne der Auszahlungen.",
    )
    cap = _kipp(*S.CAP_X, S.CAP_SICHER, S.CAP_SICHER, S.CAP_P, float(rho))
    st.caption(
        f"Zum Vergleich im Kapazitätsfall: Spanne {de(cap['spanne'], 0)} GE, Vorsprung des Erwartungswerts {de(cap['vorsprung'], 0)} GE, ρ/Spanne = {de(rho / cap['spanne'])}. "
        "Dieselbe Zahl ρ führt bei anderen Auszahlungen zu einer anderen Wahl (zweites und viertes Experiment)."
    )

ss = st.session_state
sync_query_params({k: ss[k] for k in (
    "rho_select", "x1_slider", "x2_slider", "p1_slider", "fg_select", "fl_slider", "a1_slider", "a2_slider", "b1_slider", "b2_slider", "p2_slider",
    "w3_select", "g3_select", "sk4_select", "n4_select", "seed4_input", "g5_select", "w5_select")})
RHO = float(rho)

# --- Ergebnis: der Kapazitätsfall --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: der Kapazitätsfall")
cap_a = _lotterie(*S.CAP_X, S.CAP_P, RHO)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Erwartungswert von G (groß bauen)", f"{de(cap_a['ew'])} GE", delta=f"{de(cap_a['ew'] - S.CAP_SICHER)} gegen K (+30 GE sicher)", delta_color="off", delta_arrow="off")
m2.metric(f"Sicherheitsäquivalent von G bei ρ = {de(RHO, 0)}", f"{de(cap_a['sae'])} GE", delta=f"{de(cap_a['sae'] - S.CAP_SICHER)} gegen K", delta_color="normal")
m3.metric("Risikoprämie", f"{de(cap_a['praemie'])} GE", delta=f"Näherung σ²/(2ρ): {de(cap_a['praemie_naeherung'])}", delta_color="off", delta_arrow="off")
m4.metric("Kipp-Risikotoleranz ρ*", f"{de(cap['punkte'][0])} GE", delta=f"ρ*/Spanne = {de(cap['punkte'][0] / cap['spanne'])}", delta_color="off", delta_arrow="off")
st.plotly_chart(build_kipp(_kurven(*S.CAP_X, S.CAP_SICHER, S.CAP_SICHER, S.CAP_P), cap), width="stretch", key="main_chart")
st.caption(
    "Kapazitätsfall mit ausgedachten Auszahlungen: A = G (groß bauen, +200 | −100 GE, je ½), B = K (klein oder mieten, sicher +30 GE). "
    "Das Sicherheitsäquivalent von G wächst mit ρ und schneidet K genau einmal, bei ρ*."
)

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
k_dyn = _kipp(ss["a1_slider"], ss["a2_slider"], ss["b1_slider"], ss["b2_slider"], ss["p2_slider"], RHO)
z_dyn = _zufall(int(ss["seed4_input"]), int(ss["n4_select"]), RHO, float(ss["sk4_select"]))
st.markdown("**1. Was ist eine Lotterie dem Entscheider wert?**")
_kennzeichen(ev.verdict_sae(_lotterie(ss["x1_slider"], ss["x2_slider"], ss["p1_slider"], RHO)))
st.markdown("**2. Ab welcher Risikotoleranz kippt die Wahl, und was bedeutet die Zahl?**")
_kennzeichen(ev.verdict_kipp(k_dyn))
_kennzeichen(ev.verdict_fall_vergleich(ev.faelle_vergleich(RHO), RHO))
st.markdown("**3. Wie oft spielt das überhaupt eine Rolle?**")
_kennzeichen(ev.verdict_zufall(z_dyn))

st.markdown("---")

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das Sicherheitsäquivalent von Hand")
st.caption(
    "Eine Lotterie mit zwei Auszahlungen (Voreinstellung: groß bauen im Kapazitätsfall). Das ρ stellen Sie in der Seitenleiste ein. "
    "Die Näherung μ − σ²/(2ρ) ist exakt, wenn die Auszahlung normalverteilt ist, sonst eine Näherung zweiter Ordnung."
)
c1, c2, c3 = st.columns(3)
x1 = c1.slider("Auszahlung in Zustand 1 (GE)", *bounds("x1_slider"), key="x1_slider", step=C.X_STEP)
x2 = c2.slider("Auszahlung in Zustand 2 (GE)", *bounds("x2_slider"), key="x2_slider", step=C.X_STEP)
p1 = c3.slider("Wahrscheinlichkeit von Zustand 1", *bounds("p1_slider"), key="p1_slider", step=C.P_STEP)
la = _lotterie(float(x1), float(x2), float(p1), RHO)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Erwartungswert", f"{de(la['ew'])} GE")
k2.metric("Sicherheitsäquivalent", f"{de(la['sae'])} GE", delta=f"{de(la['sae'] - la['ew'])} gegen Erwartungswert", delta_color="off", delta_arrow="off")
k3.metric("Risikoprämie", f"{de(la['praemie'])} GE")
k4.metric("Näherung der Prämie σ²/(2ρ)", f"{de(la['praemie_naeherung'])} GE", delta="–" if la["fehler"] is None else ev.pm(la["fehler"]), delta_color="off", delta_arrow="off")
st.plotly_chart(build_nutzen(la), width="stretch", key="nutzen_chart")
_kennzeichen(ev.verdict_sae(la))
tab = ev.sae_tabelle(float(x1), float(x2), float(p1))
st.dataframe(
    pd.DataFrame([
        {"ρ (GE)": de(t["rho"], 1 if t["rho"] % 1 else 0), "Sicherheitsäquivalent (GE)": de(t["sae"]), "Risikoprämie (GE)": de(t["praemie"]), "Näherung der Prämie (GE)": de(t["praemie_naeherung"]),
         "Fehler der Näherung": "–" if t["fehler"] is None else ev.pm(t["fehler"])} for t in tab
    ]),
    hide_index=True, width="stretch",
)
sk = ev.skalen_beispiel()
st.markdown(
    f"**Skaleninvarianz.** Die Lotterie (0 | 100 GE, je ½) hat bei ρ = 100 GE das Sicherheitsäquivalent **{de(sk['klein'])} GE**; verdoppelt man Auszahlungen und ρ (0 | 200 GE, ρ = 200 GE), verdoppelt sich auch das Sicherheitsäquivalent: **{de(sk['gross'])} GE**."
)
st.markdown("**Frage statt Zahl.** Gewinn g·L gegen Verlust L, je ½, gegen 0 sicher: bei welchem Verlust L lehnen Sie ab? Daraus folgt ρ (rauschfrei gerechnet: die Antwort wird als genau genommen).")
f1c, f2c, f3c = st.columns(3)
fg = f1c.selectbox("Gewinn g · L", C.FRAGE_G, key="fg_select", format_func=lambda g: f"g = {de(g, 1)}", help="Wie viel mehr als der mögliche Verlust L die Lotterie im guten Fall bringt.")
fl = f2c.slider("Größter Verlust L, den Sie gerade noch annehmen (GE)", *bounds("fl_slider"), key="fl_slider", step=C.L_STEP)
faktor = ev.rho_aus_frage(float(fg))
f3c.metric("Daraus folgt ρ", f"{de(faktor * float(fl))} GE", delta=f"ρ/L = {de(faktor, 3)} bei g = {de(float(fg), 1)}", delta_color="off", delta_arrow="off")
st.caption(
    "Der größte akzeptable Verlust bestimmt ρ nur, wenn der Gewinn mitgenannt wird: bei g = 1,2 ist ρ/L gleich 6,02, bei g = 10 gleich 1,44 (Wert für g = 2: 1/ln φ = 2,078). "
    f"Wie unsicher solche Antworten sind und wie man ρ erhebt, behandelt {C.repo_link('nutzen-erheben-demo')}."
)

st.markdown("---")

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Kippt die Wahl, und wann?")
st.caption(
    "Alternative A gegen Alternative B bei P(Zustand 1) = p. Voreinstellung: Kapazitätsfall, A = G (groß bauen), B = K (sicher 30 GE). Sind beide Auszahlungen von B gleich, ist B sicher: dann gibt es höchstens ein ρ*. "
    "Sonst wird das ρ-Gitter (500 Punkte von 0,5 bis 100 000 GE) nach Wechseln abgesucht, je Wechsel genau bestimmt."
)
d1, d2, d3, d4, d5 = st.columns(5)
a1 = d1.slider("A in Zustand 1 (GE)", *bounds("a1_slider"), key="a1_slider", step=C.A_STEP)
a2 = d2.slider("A in Zustand 2 (GE)", *bounds("a2_slider"), key="a2_slider", step=C.A_STEP)
b1 = d3.slider("B in Zustand 1 (GE)", *bounds("b1_slider"), key="b1_slider", step=C.A_STEP)
b2 = d4.slider("B in Zustand 2 (GE)", *bounds("b2_slider"), key="b2_slider", step=C.A_STEP)
p2 = d5.slider("Wahrscheinlichkeit von Zustand 1", *bounds("p2_slider"), key="p2_slider", step=C.P_STEP)
kk = _kipp(float(a1), float(a2), float(b1), float(b2), float(p2), RHO)
e1, e2, e3, e4 = st.columns(4)
e1.metric("Kipp-Risikotoleranz ρ*", ", ".join(de(r) for r in kk["punkte"]) + " GE" if kk["punkte"] else "keine", delta=f"{len(kk['punkte'])} Kipppunkt(e)" if not kk["sicher"] else None, delta_color="off", delta_arrow="off")
e2.metric("Spanne beider Alternativen", f"{de(kk['spanne'], 0)} GE")
e3.metric("Vorsprung des Erwartungswerts (A minus B)", f"{de(kk['vorsprung'])} GE")
e4.metric(f"SÄ bei ρ = {de(RHO, 0)}: A | B", f"{de(kk['ce_a'])} | {de(kk['ce_b'])} GE", delta="gewählt: " + ("A" if kk["wahl_sae"] == 1 else "B" if kk["wahl_sae"] == -1 else "gleich"), delta_color="off", delta_arrow="off")
st.plotly_chart(build_kipp(_kurven(float(a1), float(a2), float(b1), float(b2), float(p2)), kk), width="stretch", key="kipp_chart")
_kennzeichen(ev.verdict_kipp(kk))
if kk["sae_verlust"] > 0:
    st.caption(
        f"SÄ-Verlust: Wer bei ρ = {de(RHO, 0)} GE nach dem Erwartungswert wählt, bekommt {de(kk['sae_verlust'])} GE weniger Sicherheitsäquivalent als mit der Wahl nach dem Sicherheitsäquivalent. "
        "Das ist ein Unterschied von Sicherheitsäquivalenten vor der Entscheidung, kein Vergleich mit einem Orakel, das den Zustand kennt."
    )
st.markdown("**Brücke zur Wahrscheinlichkeitsschwelle.** Ab welcher Wahrscheinlichkeit von Zustand 1 ist A bei diesem ρ besser als B? Die Gleichung ist linear in p, die Schwelle wandert mit ρ.")
st.dataframe(
    pd.DataFrame([
        {"ρ (GE)": "∞ (Erwartungswert)" if math.isinf(t["rho"]) else de(t["rho"], 1 if t["rho"] % 1 else 0), "Schwelle p* von Zustand 1": "keine in [0, 1]" if t["p_stern"] is None else de(t["p_stern"], 4)}
        for t in ev.p_stern_tabelle(float(a1), float(a2), float(b1), float(b2))
    ]),
    hide_index=True, width="stretch",
)
st.caption(f"Im Kapazitätsfall ist die Schwelle des Erwartungswerts 13/30 = 0,4333 und bei ρ = ρ* genau ½. Wie sich die Schwelle sonst auf die Wahl auswirkt, zeigt {C.repo_link('kipp-schwelle-demo')}.")
st.markdown("**Derselbe Kapazitätsfall mit weniger Verlust.** Familie G = (+200 | −L) gegen K = +30 sicher, p = ½: ρ* hängt am Vorsprung des Erwartungswerts, nicht allein an der Spanne.")
fv = ev.faelle_vergleich(RHO)
st.plotly_chart(build_rho_kurve(_rho_kurve(), fv), width="stretch", key="rho_kurve_chart")
st.dataframe(
    pd.DataFrame([
        {"Fall": f["name"], "Verlust (GE)": de(f["verlust"], 0), "Erwartungswert von G (GE)": de(f["ew"]), "Vorsprung (GE)": de(f["vorsprung"]), "Spanne (GE)": de(f["spanne"], 0), "ρ* (GE)": de(f["rho_stern"]),
         "ρ*/Spanne": de(f["rho_stern"] / f["spanne"]), f"SÄ von G bei ρ = {de(RHO, 0)} (GE)": de(f["sae"]), "Wahl": f["wahl"]} for f in fv
    ]),
    hide_index=True, width="stretch",
)
_kennzeichen(ev.verdict_fall_vergleich(fv, RHO))

st.markdown("---")

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Vermögen – Exponential gegen Potenz")
st.caption(
    "Kapazitätsfall G = (+200 | −100) gegen K = +30 sicher. Der Exponentialnutzen bewertet Zuwächse unabhängig vom Anfangsvermögen w; der Potenznutzen u(v) = v^(1−γ)/(1−γ) auf dem Endvermögen v = w + x (γ = 1: ln v) tut es nicht. "
    "Er verlangt w größer als 100 GE (sonst ist das Endvermögen im schlechten Zustand nicht positiv)."
)
g1, g2 = st.columns(2)
w3 = g1.selectbox("Anfangsvermögen w (GE)", C.W3_OPTIONS, key="w3_select", format_func=zahl)
g3 = g2.selectbox("Potenzexponent γ", C.GAMMA_OPTIONS, key="g3_select", help="γ = 1 ist der Logarithmus; größere γ sind risikoscheuer.")
vm = ev.vermoegen_analyse(float(w3), int(g3), RHO)
kt = ev.kipp_vermoegen_tabelle()
st.plotly_chart(build_vermoegen(_vermoegen_kurven(), vm, kt), width="stretch", key="vermoegen_chart")
_kennzeichen(ev.verdict_vermoegen(vm, kt))
st.dataframe(
    pd.DataFrame([{"Potenzexponent γ": g, "Kippvermögen w* (GE)": de(kt[g], 1 if g < 4 else 2), "Art": "exakt in Brüchen belegt" if g < 4 else "numerisch"} for g in C.GAMMA_OPTIONS]),
    hide_index=True, width="stretch",
)
st.caption("Kippvermögen: ab diesem Anfangsvermögen wählt der Potenznutzen G statt K. Der Exponentialnutzen kennt kein solches Vermögen: seine Wahl bleibt bei jedem w gleich (Delta-Eigenschaft).")

st.markdown("---")

# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Wie oft ändert die Risikoeinstellung die Wahl?")
st.caption(
    "Zufallsprobleme: zwei Alternativen, zwei Zustände, P(Zustand 1) gleichverteilt in 0,1 bis 0,9, jede Auszahlung ganzzahlig gleichverteilt in −100 bis 200 GE. Gezählt werden nur nicht dominierte Probleme "
    "(keine Alternative ist in beiden Zuständen mindestens so gut wie die andere). „Andere Wahl“: bei ρ hat die andere Alternative das größere Sicherheitsäquivalent als beim Erwartungswert. Zufall nur aus SplitMix64."
)
h1, h2, h3, h4 = st.columns([2, 2, 2, 2])
sk4 = h1.selectbox("Auszahlungen ×", C.SCALE_OPTIONS, key="sk4_select", format_func=lambda c: de(c, 1 if c < 1 else 0), help="Alle Auszahlungen mit diesem Faktor multipliziert, ρ bleibt: die Bezugsmenge ändert sich.")
n4 = h2.selectbox("Zahl der Probleme", C.N4_OPTIONS, key="n4_select", format_func=zahl)
seed4 = h3.number_input("Zufalls-Seed", *bounds("seed4_input"), key="seed4_input", step=1)
h4.markdown("&nbsp;")
h4.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)
z = _zufall(int(seed4), int(n4), RHO, float(sk4))
za, zb = z["a"], z["b"]
q1, q2, q3 = st.columns(3)
q1.metric(f"Anteil anderer Wahl bei ρ = {de(RHO, 0)}, ×{de(float(sk4), 1 if sk4 < 1 else 0)}", pct(za["anteil"]), delta=f"zweiter Strom {pct(zb['anteil'])}", delta_color="off", delta_arrow="off")
q2.metric("Nicht dominierte Probleme", zahl(za["n_nd"]), delta=f"von {zahl(za['n'])}; dominiert {pct(za['dominiert'])}", delta_color="off", delta_arrow="off")
q3.metric("Median ρ*/Spanne", de(za["verhaeltnis"]["median"]), delta=f"in {pct(za['verhaeltnis']['anteil_kippt'])} der Probleme wählen ρ → 0 und ρ = ∞ verschieden", delta_color="off", delta_arrow="off")
pa, pb = st.columns(2)
pa.plotly_chart(build_staffel(z), width="stretch", key="staffel_chart")
pb.plotly_chart(build_skalen(z), width="stretch", key="skalen_chart")
st.plotly_chart(build_verhaeltnis(z), width="stretch", key="verhaeltnis_chart")
_kennzeichen(ev.verdict_zufall(z))
st.caption(
    f"Strom 1: Zufalls-Seed {za['seed']}, Strom 2: Zufalls-Seed {zb['seed']}. Mit 200 000 Problemen und den Zufalls-Seeds 101 und 20261116 unterscheiden sich die beiden Ströme in den Anteilen der README (B9 bis B11) um höchstens 0,3 Prozentpunkte (größter Abstand: 39,0 gegen 39,3 % in B10); ρ*/Spanne gilt für die Probleme, in denen die schlechteste-Fall-Regel "
    "(ρ = 0,5 GE) und der Erwartungswert verschieden wählen, und ist unabhängig von der Skala. Der Kapazitätsfall liegt mit 1,85 weit oben: ein Beispiel mit kleinem Vorsprung."
)
fz = _frozen()
st.markdown("**Wie viele Kipppunkte kann ein Problem haben?** Vorgerechnet (tools/rae_make_frozen.py, data/rae_frozen.json): je 40 000 Zufallsprobleme, ρ-Gitter 500 Punkte von 0,5 bis 100 000 GE (Gitterbefund, kein Beweis).")
if fz is None:
    st.warning("Die vorgerechnete Datei data/rae_frozen.json fehlt.")
else:
    zeilen = []
    for e in fz["kipp"]:
        if e["art"] == "sicher":
            zeilen.append({"Vergleich": "gegen sichere Alternative", "Zustände": e["m"], "Zufalls-Seed": e["seed"], "Probleme": zahl(e["n"]), "ein Wechsel": zahl(e["w1"]), "mindestens zwei Wechsel": zahl(e["w2plus"]),
                           "SÄ wächst mit ρ": "ja, in allen" if e["mono"] else "nein"})
        else:
            zeilen.append({"Vergleich": "zwei riskante Alternativen", "Zustände": e["m"], "Zufalls-Seed": e["seed"], "Probleme": f"{zahl(e['nicht_dominiert'])} nicht dominiert", "ein Wechsel": zahl(e["w1"]),
                           "mindestens zwei Wechsel": f"{zahl(e['w2plus'])} ({pct(e['w2plus'] / e['nicht_dominiert'], 2)})", "SÄ wächst mit ρ": "–"})
    st.dataframe(pd.DataFrame(zeilen), hide_index=True, width="stretch")
    zw_sicher = [e for e in fz["kipp"] if e["art"] == "sicher"]
    st.caption(f"Zufallsproblem: Wahrscheinlichkeiten Dirichlet(1,…,1) (bei zwei Zuständen P(Zustand 1) in 0 bis 1, anders als im Generator der Zufallsprobleme oben), gleiche Wahrscheinlichkeiten in beiden Alternativen; gegen die sichere Alternative zählen alle Probleme, in {zahl(sum(e['zwischen'] for e in zw_sicher))} der {zahl(sum(e['n'] for e in zw_sicher))} liegt die sichere Auszahlung im Bereich der Lotterie (nur dort ist ein Wechsel möglich). Das exakte Beispiel mit zwei Kipppunkten (A = (−70 | +70), B = (+10 | −80), p = 0,6) steht im Schnellstart „Zwei Kipppunkte“.")

st.markdown("---")

# --- Experiment 5 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Reicht eine Näherung?")
st.caption(
    "Zwei Vereinfachungen gegen das exakte Sicherheitsäquivalent: die Mittelwert-Varianz-Näherung μ − σ²/(2ρ) und ein Potenznutzen, den man am Kapazitätsfall auf dasselbe Sicherheitsäquivalent von G kalibriert hat. "
    "Zufallsprobleme, Zufalls-Seed und Zahl der Probleme wie im vierten Experiment; ρ aus der Seitenleiste."
)
i1, i2 = st.columns(2)
g5 = i1.selectbox("Potenzexponent γ", C.GAMMA_OPTIONS, key="g5_select")
w5 = i2.selectbox("Anfangsvermögen w (GE)", C.W5_OPTIONS, key="w5_select", format_func=zahl)
z5 = _naeherung(int(seed4), int(n4), RHO, int(g5), int(w5))
r1, r2 = st.columns(2)
r1.metric(f"Näherung: Anteil anderer Wahl bei ρ = {de(RHO, 0)}", pct(z5["a"]["anteil"]), delta=f"zweiter Strom {pct(z5['b']['anteil'])}", delta_color="off", delta_arrow="off")
r2.metric(f"Potenznutzen γ = {g5}, w = {zahl(w5)}: kalibriertes ρ", f"{de(z5['kal_a']['rho_k'], 1)} GE", delta=f"Anteil anderer Wahl {pct(z5['kal_a']['anteil'], 2)}", delta_color="off", delta_arrow="off")
_kennzeichen(ev.verdict_naeherung(z5))
st.dataframe(
    pd.DataFrame([{"ρ (GE)": r, "Näherung gegen exakt, Strom 1": pct(z5["a"]["staffel"][r]), "Näherung gegen exakt, Strom 2": pct(z5["b"]["staffel"][r])} for r in C.RHO_STAFFEL if r != 10]),
    hide_index=True, width="stretch",
)
if fz is not None:
    st.markdown(f"**Potenznutzen, kalibriert: alle neun Fälle.** Vorgerechnet, je {zahl(C.N_FROZEN_CAL)} Zufallsprobleme, zwei Zufalls-Seeds; kalibriert am Kapazitätsfall G = (+200 | −100).")
    zeilen = []
    for gamma in C.GAMMA_OPTIONS:
        for w in C.W5_OPTIONS:
            ein = [e for e in fz["kalibrierung"] if e["gamma"] == gamma and e["w"] == w]
            zeilen.append({"γ": gamma, "Vermögen (GE)": zahl(w), "kalibriertes ρ (GE)": de(ein[0]["rho_k"], 1), **{f"Anteil anderer Wahl, Zufalls-Seed {e['seed']}": pct(e["anteil"], 2) for e in ein}})
    st.dataframe(pd.DataFrame(zeilen), hide_index=True, width="stretch")
    st.caption("Die Vermögensabhängigkeit wirkt vor allem nahe der Indifferenz (der Kapazitätsfall mit K = 30 liegt dort). Gemessen nur in diesem Zahlenraum (Auszahlungen −100 bis 200 GE, Vermögen 500 bis 5000 GE) und für ein am Kapazitätsfall kalibriertes ρ.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Auszahlungen sind ausgedacht** | Alle Zahlen gelten für dieses Modell (Kapazitätsfall, Exponentialnutzen, zwei Zustände, dieser Generator der Zufallsprobleme), nicht für einen Fall mit Daten. Andere Verteilungen der Zufallsprobleme sind nicht gemessen; der Anteil anderer Wahl hängt an Skala und Vorsprung. | – |
| **Konstante absolute Risikoaversion (Exponentialnutzen)** | Das Sicherheitsäquivalent hängt nicht vom Anfangsvermögen ab; ein Potenznutzen wählt bei kleinem Vermögen anders (Kippvermögen 522,5 / 1075,0 / 2184,95 GE für γ = 1 / 2 / 4). Am Kapazitätsfall kalibriert, ändert er die Wahl in den Zufallsproblemen nur selten (drittes und fünftes Experiment). | {C.repo_link('erwartungsnutzen-grenze-demo')} prüft, wo auch der Erwartungsnutzen selbst endet |
| **ρ ist bekannt und eine Zahl** | In der Praxis ist ρ eine Antwort auf Fragen und schwankt; eine Frage bestimmt ρ nur zusammen mit dem Gewinn der Lotterie (erstes Experiment). Wie sicher man ρ erheben kann, ist hier nicht gemessen. | {C.repo_link('nutzen-erheben-demo')} |
| **Ein einziges ρ\\* gegen eine sichere Alternative** | Zwischen zwei riskanten Alternativen kann es mehrere geben (0,78 und 0,88 % der nicht dominierten Zweizustandsprobleme haben mindestens zwei, in zwei Strömen mit P(Zustand 1) in 0 bis 1; mit dem Generator der Zufallsprobleme, P(Zustand 1) in 0,1 bis 0,9, sind es 1,0 % im Strom mit Zufalls-Seed 101; die Zählung läuft auf einem Gitter und kann eng benachbarte Wechsel übersehen). | – |
| **Wahrscheinlichkeiten sind bekannt** | Hier sind sie gegeben. Wie stark die Wahl an ihnen hängt, zeigt der Kipppunkt der Wahrscheinlichkeit; Information darüber zu kaufen, gehört zu einer anderen Frage. | {C.repo_link('kipp-schwelle-demo')}, {C.repo_link('information-risikoaversion-demo')} |
| **Ein Entscheider, ein Risikomaß** | Kein CVaR und kein Vergleich von Nutzenfunktion und CVaR; die Teilung von Risiko zwischen zwei Seiten (Borch-Regel) steht in {C.repo_link('kuchen-vergroessern-demo')}, Risiko als Ziel mit CVaR in {C.repo_link('cvar-demo')}. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/risiko-aequivalent-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind zwei Alternativen mit Auszahlungen in Geldeinheiten (GE) in $m$ Zuständen mit den Wahrscheinlichkeiten $p_i$; gesucht sind das Sicherheitsäquivalent jeder Alternative, die Wahl bei Risikotoleranz $\rho$ und die Risikotoleranz $\rho^*$, bei der die Wahl kippt.

| Symbol | Bedeutung |
|---|---|
| $x_i$, $y_i$ | Auszahlung von Alternative A bzw. B im Zustand $i$ (GE, Gewinn positiv) |
| $p_i$ | Wahrscheinlichkeit des Zustands $i$, Summe 1 |
| $\rho$ | Risikotoleranz in GE (konstante absolute Risikoaversion $1/\rho$) |
| $u$ | Nutzenfunktion $u(x) = 1 - e^{-x/\rho}$ |
| $\mathrm{S\ddot{A}}$ | Sicherheitsäquivalent: der sichere Betrag mit demselben Nutzen wie die Lotterie |
| $\mu$, $\sigma^2$ | Erwartungswert und Varianz der Auszahlung |
| Spanne | größte minus kleinste Auszahlung beider Alternativen |
| Vorsprung | Erwartungswert der riskanten minus Wert der sicheren Alternative |
| $\rho^*$ | Kipp-Risikotoleranz: Wert von $\rho$, bei dem beide Alternativen gleich gut sind |
| $w$, $\gamma$ | Anfangsvermögen und Exponent des Potenznutzens |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/risiko-aequivalent-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind zwei Alternativen mit Auszahlungen in Geldeinheiten (GE) in $m$ Zuständen mit den Wahrscheinlichkeiten $p_i$; gesucht sind das Sicherheitsäquivalent jeder Alternative, die Wahl bei Risikotoleranz $\rho$ und die Risikotoleranz $\rho^*$, bei der die Wahl kippt.

| Symbol | Bedeutung |
|---|---|
| $x_i$, $y_i$ | Auszahlung von Alternative A bzw. B im Zustand $i$ (GE, Gewinn positiv) |
| $p_i$ | Wahrscheinlichkeit des Zustands $i$, Summe 1 |
| $\rho$ | Risikotoleranz in GE (konstante absolute Risikoaversion $1/\rho$) |
| $u$ | Nutzenfunktion $u(x) = 1 - e^{-x/\rho}$ |
| $\mathrm{S\ddot{A}}$ | Sicherheitsäquivalent: der sichere Betrag mit demselben Nutzen wie die Lotterie |
| $\mu$, $\sigma^2$ | Erwartungswert und Varianz der Auszahlung |
| Spanne | größte minus kleinste Auszahlung beider Alternativen |
| Vorsprung | Erwartungswert der riskanten minus Wert der sicheren Alternative |
| $\rho^*$ | Kipp-Risikotoleranz: Wert von $\rho$, bei dem beide Alternativen gleich gut sind |
| $w$, $\gamma$ | Anfangsvermögen und Exponent des Potenznutzens |

**Nutzen und Sicherheitsäquivalent.**

$$u(x) = 1 - e^{-x/\rho}, \qquad \mathrm{S\ddot{A}}(X) = -\rho \ln \sum_i p_i\, e^{-x_i/\rho}$$

**Risikoprämie und Normalnäherung.** Die Prämie ist $\mu - \mathrm{S\ddot{A}} \ge 0$; für normalverteiltes $X$ gilt exakt, sonst als Näherung zweiter Ordnung:

$$\mathrm{S\ddot{A}}(X) \approx \mu - \frac{\sigma^2}{2\rho}$$

**Wahl und Kipppunkt.** A wird gewählt, wenn $\mathrm{S\ddot{A}}(X) > \mathrm{S\ddot{A}}(Y)$. Bei einer sicheren Alternative $k$ wächst $\mathrm{S\ddot{A}}(X)$ in allen gezählten Problemen mit $\rho$ von der kleinsten Auszahlung bis zum Erwartungswert (ein Beweis steht hier nicht); $\rho^*$ löst

$$\mathrm{S\ddot{A}}(X;\rho^*) = k$$

und ist dann die einzige Lösung. Für $p = \tfrac12$ und zwei Zustände gilt die Näherung

$$\rho^* \approx \frac{(\text{Spanne})^2}{8\,(\mu - k)}$$

**Schwelle der Wahrscheinlichkeit.** Bei zwei Zuständen ist die Gleichung $\mathrm{S\ddot{A}}(X) = \mathrm{S\ddot{A}}(Y)$ linear in $p_1$; mit $a_i = e^{-x_i/\rho}$ und $b_i = e^{-y_i/\rho}$:

$$p_1^* = \frac{b_2 - a_2}{(a_1 - a_2) - (b_1 - b_2)}$$

**Skaleninvarianz.** Multipliziert man alle Auszahlungen und $\rho$ mit $c > 0$, ändert sich die Wahl nicht; das Sicherheitsäquivalent wird mit $c$ multipliziert. Addiert man $w$ zu allen Auszahlungen, wird das Sicherheitsäquivalent um genau $w$ größer und die Wahl bleibt gleich (Delta-Eigenschaft).

**Potenznutzen.** Auf dem Endvermögen $v = w + x > 0$:

$$u(v) = \frac{v^{1-\gamma}}{1-\gamma} \quad (\gamma \ne 1), \qquad u(v) = \ln v \quad (\gamma = 1)$$

Das Sicherheitsäquivalent der Zuwächse ist $u^{-1}\!\left(\sum_i p_i\, u(w + x_i)\right) - w$ und wächst mit $w$. Das Modell ist vollständig für zwei Alternativen mit endlich vielen Zuständen; nicht enthalten sind stetige Verteilungen, mehrere Perioden und ein Risikomaß ohne Nutzenfunktion (siehe die Grenzen oben).

**Quellen und ihre Stufe.** Eigenschaften der Exponentialfunktion (Delta-Eigenschaft, Additivität über unabhängige Risiken, Normalformel): arXiv 1809.00817, als Zweitquelle gelesen. Näherung nach Pratt und die Aussage „risikoaverser heißt größere Prämie für jedes Risiko“:
Gollier, Hammitt und Treich 2013, S. 129 und 139, als Zweitquelle gelesen; Pratt 1964 und Arrow 1965 im Original nicht gelesen. Alle Zahlen dieser Seite sind eigene Rechnung in diesem Modell. Alles hier ist Lehrbuchwissen; die Demo macht es mit Zähleinheit und Zahl vorführbar.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Entscheidungsanalyse: Information, Risiko, Grenzen](https://sebastianhanisch.net/konzepte-entscheidungsanalyse.html)."
)

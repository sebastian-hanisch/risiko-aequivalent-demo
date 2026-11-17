"""Kennzahlen, Messreihen und Verdikte der fünf Experimente. Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`: die App ruft `getattr(st, Art)(Text)`,
`tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten.

Alle Auszahlungen in Geldeinheiten (GE). Das Beispiel (Kapazitätsfall) ist ausgedacht: ein Modell, kein Fall mit Daten. „Andere Wahl“ meint: bei der Risikotoleranz ρ
hat die andere Alternative das größere Sicherheitsäquivalent als beim Erwartungswert (ρ = unendlich); Gleichstände (Unterschied unter 1e-9) zählen nicht als Wahl.
Zähleinheit der Anteile: nicht dominierte Zufallsprobleme (zwei Alternativen, zwei Zustände, Auszahlungen -100 bis 200 GE)."""

import json
import math
from pathlib import Path

import numpy as np

import rae_constants as C
import rae_scenario as S
import rae_utility as U
import rae_wealth as W

ROOT = Path(__file__).resolve().parent


# --- Formatierung -------------------------------------------------------------------------------------------------------------------------

def de(x, nd=2):
    """Dezimalzahl mit Komma und echtem Minuszeichen; ab 10 000 mit schmalem Leerzeichen als Tausendertrenner."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    r = round(float(x), nd)
    ganz, _, rest = f"{abs(r):.{nd}f}".partition(".")
    if abs(r) >= 10000:
        ganz = f"{int(ganz):,}".replace(",", " ")
    return ("−" if r < 0 else "") + ganz + ("," + rest if rest else "")


def zahl(n):
    """Ganzzahl mit schmalem Leerzeichen als Tausendertrenner (200 000)."""
    return f"{int(n):,}".replace(",", " ")


def pct(anteil, nd=1):
    """Anteil (0 bis 1) als Prozentzahl mit Komma."""
    return de(100.0 * anteil, nd) + " %"


def pm(x, nd=1):
    """Prozentabweichung mit Vorzeichen."""
    return ("+" if x >= 0 else "−") + de(abs(x), nd) + " %"


# --- Experiment 1: Sicherheitsäquivalent -----------------------------------------------------------------------------------------------------

def lotterie_analyse(x1, x2, p1, rho):
    """Sicherheitsäquivalent, Risikoprämie und Normalnäherung einer Lotterie (x1 mit Wahrscheinlichkeit p1, sonst x2) bei Risikotoleranz rho."""
    x = np.array([x1, x2], dtype=float)
    p = np.array([p1, 1.0 - p1], dtype=float)
    ew = float(U.erwartung(x, p))
    sae = float(U.ce(x, p, float(rho)))
    pr = ew - sae
    nae = float(U.normal_naeherung(x, p, float(rho)))
    pr_nae = ew - nae
    return dict(x1=float(x1), x2=float(x2), p1=float(p1), rho=float(rho), ew=ew, sae=sae, praemie=pr, naeherung=nae, praemie_naeherung=pr_nae,
                fehler=(100.0 * (pr_nae / pr - 1.0) if pr > U.TIE else None), sigma=math.sqrt(float(U.varianz(x, p))), sicher=bool(pr < U.TIE and x1 == x2))


def sae_tabelle(x1, x2, p1, rhos=C.RHO_TABELLE):
    """Zeilen (ρ, SÄ, Risikoprämie, Normalnäherung der Prämie, Fehler der Näherung in %) der Lotterie über ρ."""
    zeilen = []
    for r in rhos:
        a = lotterie_analyse(x1, x2, p1, r)
        zeilen.append(dict(rho=r, sae=a["sae"], praemie=a["praemie"], praemie_naeherung=a["praemie_naeherung"], fehler=a["fehler"]))
    return zeilen


def skalen_beispiel():
    """Skaleninvarianz: (0 | 100) bei ρ = 100 und (0 | 200) bei ρ = 200 (je ½): das zweite Sicherheitsäquivalent ist das Doppelte des ersten."""
    p = np.array([0.5, 0.5])
    a = float(U.ce(np.array([0.0, 100.0]), p, 100.0))
    b = float(U.ce(np.array([0.0, 200.0]), p, 200.0))
    return dict(klein=a, gross=b)


def rho_aus_frage(g):
    """Risikotoleranz je GE Verlust L aus der Frage „Gewinn g·L gegen Verlust L, je ½, gegen 0 sicher: ab welchem L lehnen Sie ab?“.

    Indifferenz bei y = L/ρ mit ½ e^(−g y) + ½ e^y = 1; Rückgabe ρ/L = 1/y (rauschfrei: die Antwort wird als genau gegeben genommen). Nur für g > 1 (sonst ist der Erwartungswert negativ)."""
    if g <= 1.0:
        raise ValueError("g muss größer als 1 sein")
    y = U.bisect_log(lambda t: 0.5 * math.exp(-g * t) + 0.5 * math.exp(t) - 1.0, 1e-9, 5.0)
    return 1.0 / y


def verdict_sae(a):
    if a["sicher"]:
        return "info", f"Die Auszahlung ist in beiden Zuständen gleich ({de(a['x1'], 0)} GE): Sicherheitsäquivalent und Erwartungswert sind gleich, die Risikoprämie ist 0."
    fehler = "" if a["fehler"] is None else f" Die Näherung σ²/(2ρ) sagt {de(a['praemie_naeherung'])} ({pm(a['fehler'])})."
    return "info", (
        f"Bei ρ = {de(a['rho'], 0)} GE ist die Lotterie ({de(a['x1'], 0)} | {de(a['x2'], 0)}; P(Zustand 1) = {de(a['p1'])}) dem Entscheider **{de(a['sae'])} GE** sicher wert: "
        f"Erwartungswert {de(a['ew'])}, Risikoprämie **{de(a['praemie'])} GE**.{fehler}"
    )


# --- Experiment 2: Kipppunkte ----------------------------------------------------------------------------------------------------------------

def kipp_analyse(a1, a2, b1, b2, p1, rho):
    """Alternative A = (a1 | a2) gegen B = (b1 | b2), P(Zustand 1) = p1: Sicherheitsäquivalente bei rho, Kipp-Risikotoleranzen, Spanne, Vorsprung, Schwelle p*(ρ)."""
    x = np.array([a1, a2], dtype=float)
    y = np.array([b1, b2], dtype=float)
    p = np.array([p1, 1.0 - p1], dtype=float)
    rho = float(rho)
    ce_a, ce_b = float(U.ce(x, p, rho)), float(U.ce(y, p, rho))
    ew_a, ew_b = float(U.erwartung(x, p)), float(U.erwartung(y, p))
    sicher = b1 == b2
    sp = U.spanne(x, y)
    if sicher:
        r_stern, status = U.rho_stern_sicher(x, p, float(b1))
        punkte = [] if r_stern is None else [r_stern]
        vorz_unten = -1 if status in ("ok", "k_bleibt", "unendlich") else 1
        vorz_oben = 1 if status in ("ok", "lotterie_immer") else -1
    else:
        punkte, vorz_unten, vorz_oben = U.kipp_punkte(x, y, p)
        status = "ok" if punkte else "keiner"
    wahl = lambda d: 0 if abs(d) < U.TIE else (1 if d > 0 else -1)  # noqa: E731
    w_sae, w_ew = wahl(ce_a - ce_b), wahl(ew_a - ew_b)
    verlust = 0.0
    if w_sae != 0 and w_ew != 0 and w_sae != w_ew:
        verlust = abs(ce_a - ce_b)                  # SÄ-Verlust: Unterschied der Sicherheitsäquivalente, wenn man die Wahl des Erwartungswerts trifft
    vorsprung = ew_a - ew_b
    naeherung = None
    if sicher and status == "ok" and vorsprung > 0:
        naeherung = float(U.varianz(x, p)) / (2.0 * vorsprung)      # Näherung zweiter Ordnung σ²/(2·Vorsprung)
    return dict(
        a=(float(a1), float(a2)), b=(float(b1), float(b2)), p1=float(p1), rho=rho, sicher=sicher, ce_a=ce_a, ce_b=ce_b, ew_a=ew_a, ew_b=ew_b, wahl_sae=w_sae, wahl_ew=w_ew,
        sae_verlust=verlust, punkte=[float(v) for v in punkte], status=status, vorz_unten=vorz_unten, vorz_oben=vorz_oben, spanne=sp, vorsprung=vorsprung,
        naeherung=naeherung, min_a=float(min(a1, a2)), min_b=float(min(b1, b2)),
    )


def p_stern_tabelle(a1, a2, b1, b2, rhos=(math.inf, 5000.0, 1000.0, 555.8, 300.0, 200.0, 100.0, 50.0)):
    """Zeilen (ρ, Schwelle p*): ab welcher Wahrscheinlichkeit des ersten Zustands A bei diesem ρ besser ist als B (None: keine Schwelle in [0, 1])."""
    zeilen = []
    for r in rhos:
        ps = U.p_stern((a1, a2), (b1, b2), r)
        zeilen.append(dict(rho=r, p_stern=ps if ps is not None and 0.0 <= ps <= 1.0 else None))
    return zeilen


def ce_kurven(a1, a2, b1, b2, p1, n=90, lo=10.0, hi=10000.0):
    """Sicherheitsäquivalente von A und B über ρ (logarithmisches Gitter) für das Diagramm."""
    r = np.exp(np.linspace(math.log(lo), math.log(hi), n))
    p = np.array([p1, 1.0 - p1])
    xa, xb = np.broadcast_to(np.array([a1, a2], dtype=float), (n, 2)), np.broadcast_to(np.array([b1, b2], dtype=float), (n, 2))
    pp = np.broadcast_to(p, (n, 2))
    return dict(rho=r, a=U.ce(xa, pp, r), b=U.ce(xb, pp, r))


def rho_kurve():
    """ρ* über den Verlust L: Familie G = (+200 | −L), K = +30 sicher, P = ½; mit ρ*/Spanne und der Näherung Spanne²/(8·Vorsprung)."""
    p = np.array([0.5, 0.5])
    zeilen = []
    for L in C.L_KURVE:
        x = np.array([S.CAP_HOCH, -float(L)])
        vorsprung = 0.5 * (S.CAP_HOCH - L) - S.CAP_SICHER
        sp = S.CAP_HOCH + L
        r, status = U.rho_stern_sicher(x, p, S.CAP_SICHER)
        zeilen.append(dict(L=L, ew=0.5 * (S.CAP_HOCH - L), vorsprung=vorsprung, spanne=sp, rho_stern=r, verhaeltnis=r / sp, naeherung=U.normal_rho_stern(sp, vorsprung)))
    return zeilen


def faelle_vergleich(rho):
    """Fall 1 (G = +200 | −100) und Fall 2 (G = +200 | −50) gegen +30 sicher bei der Risikotoleranz rho."""
    zeilen = []
    for name, verlust in (("Fall 1", 100.0), ("Fall 2", C.CAP_FALL2_VERLUST)):
        a = kipp_analyse(S.CAP_HOCH, -verlust, S.CAP_SICHER, S.CAP_SICHER, S.CAP_P, rho)
        zeilen.append(dict(name=name, verlust=verlust, ew=a["ew_a"], vorsprung=a["vorsprung"], spanne=a["spanne"], rho_stern=a["punkte"][0], sae=a["ce_a"],
                           wahl="G" if a["wahl_sae"] == 1 else "K", sae_verlust=a["sae_verlust"]))
    return zeilen


def _bereiche(punkte, start, namen):
    """Text „unter ρ₁ wählt X, zwischen ρ₁ und ρ₂ Y, über ρₙ Z“ aus den Kipppunkten; `start` ist +1 (A) oder -1 (B) unterhalb des ersten Punktes."""
    wer = [namen[0] if start == 1 else namen[1], namen[1] if start == 1 else namen[0]]
    teile = [f"unter {de(punkte[0])} GE {wer[0]}"]
    for i in range(1, len(punkte)):
        teile.append(f"zwischen {de(punkte[i - 1])} und {de(punkte[i])} GE {wer[i % 2]}")
    teile.append(f"über {de(punkte[-1])} GE {wer[len(punkte) % 2]}")
    return ", ".join(teile)


def verdict_kipp(k):
    """Verdikt zum zweiten Experiment: Kipp-Risikotoleranz(en) und was die Zahl bedeutet."""
    na, nb = "A", "B"
    if k["sicher"]:
        s = k["status"]
        if s == "k_bleibt":
            return "info", (f"B (sicher {de(k['b'][0], 0)} GE) wird bei jeder Risikotoleranz gewählt: der Erwartungswert von A ({de(k['ew_a'])} GE) liegt nicht über {de(k['b'][0], 0)} GE, "
                            "und das Sicherheitsäquivalent liegt nie über dem Erwartungswert. Es gibt kein ρ*.")
        if s == "lotterie_immer":
            return "info", (f"A wird bei jeder Risikotoleranz mindestens so gut bewertet wie B (sicher {de(k['b'][0], 0)} GE): die schlechteste Auszahlung von A ({de(k['min_a'], 0)} GE) liegt nicht unter dieser Zahl. Es gibt kein ρ*.")
        if s == "unendlich":
            return "warning", (f"Der Vorsprung von A ({de(k['vorsprung'])} GE) ist so klein, dass ρ* über 10¹² GE läge: praktisch wird B bei jedem ρ gewählt, das ein Mensch nennen würde.")
        r = k["punkte"][0]
        wahl_jetzt = "A" if k["wahl_sae"] == 1 else "B"
        naeh = "" if k["naeherung"] is None else f" Die Näherung σ²/(2·Vorsprung) sagt {de(k['naeherung'])} GE ({pm(100 * (k['naeherung'] / r - 1))})."
        text = (f"A wird ab **ρ* = {de(r)} GE** gewählt (B sicher {de(k['b'][0], 0)} GE; Spanne {de(k['spanne'], 0)} GE, ρ*/Spanne **{de(r / k['spanne'])}**; Vorsprung des Erwartungswerts {de(k['vorsprung'])} GE).{naeh} "
                f"Bei ρ = {de(k['rho'], 0)} GE wird {wahl_jetzt} gewählt")
        if k["sae_verlust"] > 0:
            text += f"; wer stattdessen den Erwartungswert maximiert und A wählt, verliert {de(k['sae_verlust'])} GE Sicherheitsäquivalent (SÄ-Verlust)."
        else:
            text += "."
        return "success", text
    n = len(k["punkte"])
    if n == 0:
        gewinner = na if k["vorz_unten"] == 1 else nb
        if k["vorz_unten"] == 0:
            return "info", "Beide Alternativen sind bei jedem ρ des Gitters gleich bewertet."
        return "info", f"Im Gitter von 0,5 bis 100 000 GE (500 Punkte) gibt es keinen Kipppunkt: {gewinner} wird bei jedem ρ gewählt."
    text = _bereiche(k["punkte"], k["vorz_unten"], (na, nb))
    vorne = k["ew_a"] > k["ew_b"] and k["min_a"] > k["min_b"]
    vorne_b = k["ew_b"] > k["ew_a"] and k["min_b"] > k["min_a"]
    zusatz = ""
    if n >= 2 and (vorne or vorne_b):
        zusatz = f" Das, obwohl {na if vorne else nb} im Mittel ({de(k['ew_a'] if vorne else k['ew_b'])} gegen {de(k['ew_b'] if vorne else k['ew_a'])} GE) und im schlechtesten Fall besser ist."
    kennz = "Ein Kipppunkt" if n == 1 else f"{n} Kipppunkte"
    return "success", f"{kennz} (Gitter 0,5 bis 100 000 GE, 500 Punkte): {text}.{zusatz} ρ*/Spanne: {', '.join(de(r / k['spanne']) for r in k['punkte'])}."


def verdict_fall_vergleich(zeilen, rho):
    f1, f2 = zeilen
    return "info", (
        f"Dieselbe Zahl, andere Wahl: bei ρ = {de(rho, 0)} GE wählt Fall 1 (Verlust {de(f1['verlust'], 0)}, Erwartungswert {de(f1['ew'])}, ρ* = {de(f1['rho_stern'])}) {f1['wahl']} "
        f"und Fall 2 (Verlust {de(f2['verlust'], 0)}, Erwartungswert {de(f2['ew'])}, ρ* = {de(f2['rho_stern'])}) {f2['wahl']}. "
        f"Die Zahl ρ hat eine Einheit und einen Bezug: erst mit der Spanne ({de(f1['spanne'], 0)} und {de(f2['spanne'], 0)} GE) und dem Vorsprung ({de(f1['vorsprung'])} und {de(f2['vorsprung'])} GE) lässt sich sagen, was sie für die Wahl heißt (im Modell legt ρ die Nutzenfunktion fest; die Deutung braucht die Auszahlungen)."
    )


# --- Experiment 3: Vermögen -----------------------------------------------------------------------------------------------------------------

def vermoegen_analyse(w, gamma, rho):
    """Kapazitätsfall G = (+200 | −100) gegen K = +30 sicher: Wahl bei Exponentialnutzen (ρ) und bei Potenznutzen (γ) mit Anfangsvermögen w."""
    x, p = np.array(S.CAP_X), np.array(S.CAP_P2)
    ce_e = float(U.ce(x, p, float(rho)))
    ok = W.definiert(x, p, w)
    ce_p = float(W.ce_pow(x, p, float(w), float(gamma))) if ok else None
    return dict(w=float(w), gamma=int(gamma), rho=float(rho), ce_exp=ce_e, wahl_exp="G" if ce_e > S.CAP_SICHER else "K", definiert=ok, ce_pow=ce_p,
                wahl_pow=None if ce_p is None else ("G" if ce_p > S.CAP_SICHER else "K"), kipp_w=W.kipp_vermoegen(x, p, S.CAP_SICHER, float(gamma)))


def kipp_vermoegen_tabelle():
    """Kippvermögen w* des Kapazitätsfalls für γ = 1, 2, 4."""
    x, p = np.array(S.CAP_X), np.array(S.CAP_P2)
    return {g: W.kipp_vermoegen(x, p, S.CAP_SICHER, float(g)) for g in C.GAMMA_OPTIONS}


def vermoegen_kurven(n=80):
    """Sicherheitsäquivalent von G (Zuwächse) über das Vermögen w für Potenznutzen γ = 1, 2, 4 (der Exponentialnutzen ist waagerecht)."""
    x, p = np.array(S.CAP_X), np.array(S.CAP_P2)
    w = np.exp(np.linspace(math.log(110.0), math.log(10000.0), n))
    return dict(w=w, kurven={g: np.array([float(W.ce_pow(x, p, float(v), float(g))) for v in w]) for g in C.GAMMA_OPTIONS})


def verdict_vermoegen(v, kipp_tab):
    if not v["definiert"]:
        return "warning", (f"Bei Vermögen {de(v['w'], 0)} GE ist das Endvermögen im schlechten Zustand nicht positiv (w − 100 ≤ 0): der Potenznutzen ist nicht definiert, es gibt keine Zahl. "
                           f"Der Exponentialnutzen (ρ = {de(v['rho'], 0)} GE) wählt {v['wahl_exp']} (Sicherheitsäquivalent von G {de(v['ce_exp'])} GE), unabhängig vom Vermögen.")
    kw = kipp_tab[v["gamma"]]
    return "info", (
        f"Exponentialnutzen (ρ = {de(v['rho'], 0)} GE): wählt **{v['wahl_exp']}** bei jedem Vermögen, Sicherheitsäquivalent von G {de(v['ce_exp'])} GE (Delta-Eigenschaft, exakt in Brüchen belegt). "
        f"Potenznutzen γ = {v['gamma']} bei Vermögen {de(v['w'], 0)} GE: Sicherheitsäquivalent von G {de(v['ce_pow'])} GE, wählt **{v['wahl_pow']}**; "
        f"das Kippvermögen ist {de(kw, 1 if v['gamma'] < 4 else 2)} GE, darüber wird G gewählt."
    )


# --- Experiment 4: Wie oft ändert die Risikoeinstellung die Wahl? ---------------------------------------------------------------------------

def _anteil_anders(P, A, nd, rho, ew):
    w = U.wahl_vorzeichen(P, A, rho)
    return float(np.sum(nd & (w != ew) & (w != 0) & (ew != 0)) / max(int(nd.sum()), 1))


def zufall_strom(seed, n, rho, skala, mit_verhaeltnis=False):
    """Ein Strom Zufallsprobleme: Anteil mit anderer Wahl als beim Erwartungswert (bei rho und Skala der Auszahlungen), über die Staffel ρ und über die Skalen; optional ρ*/Spanne."""
    P, A, dom = S.zufallsprobleme(seed, n)
    nd = ~dom
    ew = U.wahl_vorzeichen(P, A, math.inf)
    out = dict(seed=int(seed), n=int(n), n_nd=int(nd.sum()), dominiert=float(dom.mean()), rho=float(rho), skala=float(skala))
    out["anteil"] = _anteil_anders(P, A * skala, nd, float(rho), ew)
    out["staffel"] = {r: _anteil_anders(P, A, nd, float(r), ew) for r in C.RHO_STAFFEL}
    out["skalen"] = {c: _anteil_anders(P, A * c, nd, float(rho), ew) for c in C.SCALE_OPTIONS}
    if mit_verhaeltnis:
        unten = U.wahl_vorzeichen(P, A, 0.5)
        kippt = nd & (ew != unten) & (ew != 0) & (unten != 0)
        idx = np.where(kippt)[0]
        rs = U.rho_stern_viele(P[idx], A[idx], unten[idx])
        flach = A[idx].reshape(len(idx), -1)
        q = rs / (flach.max(axis=1) - flach.min(axis=1))
        out["verhaeltnis"] = dict(
            n_kippt=int(len(idx)), anteil_kippt=float(len(idx) / max(int(nd.sum()), 1)), p10=float(np.percentile(q, 10)), median=float(np.median(q)), p90=float(np.percentile(q, 90)),
            unter_spanne=float(np.mean(q < 1)), unter_halb=float(np.mean(q < 0.5)), ueber_fuenf=float(np.mean(q > 5)), ab_kapazitaet=float(np.mean(q >= C.KAPAZITAETSFALL_RATIO)),
            werte=q,
        )
    return out


def zufall_analyse(seed, n, rho, skala):
    """Zwei Ströme: der gewählte Zufalls-Seed (mit ρ*/Spanne) und ein zweiter fester Strom für das Band."""
    zweiter = C.SEED_BAND if int(seed) != C.SEED_BAND else C.DEFAULT_SEED
    return dict(a=zufall_strom(seed, n, rho, skala, mit_verhaeltnis=True), b=zufall_strom(zweiter, n, rho, skala))


def verdict_zufall(z):
    a, b = z["a"], z["b"]
    v = a["verhaeltnis"]
    return "info", (
        f"ρ = {de(a['rho'], 0)} GE, Auszahlungen ×{de(a['skala'], 1 if a['skala'] < 1 else 0)}: in **{pct(a['anteil'])}** der {zahl(a['n_nd'])} nicht dominierten Probleme (von {zahl(a['n'])}, Zufalls-Seed {a['seed']}) wählt der Entscheider "
        f"anders als beim Erwartungswert (zweiter Strom, Zufalls-Seed {b['seed']}: {pct(b['anteil'])}). Dasselbe ρ bei den Skalen ×0,1 / ×1 / ×10: "
        f"{pct(a['skalen'][0.1])} / {pct(a['skalen'][1.0])} / {pct(a['skalen'][10.0])}. Bei {pct(v['anteil_kippt'])} der Probleme wählen Maximin und Erwartungswert verschieden, und ρ* hat einen Wert (bei einem kleinen weiteren Teil wechselt die Wahl zweimal, dort gibt es zwei ρ*, hier nicht ausgewertet): "
        f"für die erstgenannten Probleme ρ*/Spanne im Median {de(v['median'])} (P90 {de(v['p90'])}); der Kapazitätsfall ({de(C.KAPAZITAETSFALL_RATIO, 2)}) wird nur von {pct(v['ab_kapazitaet'])} dieser Probleme erreicht: er ist ein Beispiel mit kleinem Vorsprung, nicht der typische Fall."
    )


def load_frozen():
    pfad = ROOT / C.FROZEN_FILE
    if not pfad.exists():
        return None
    return json.loads(pfad.read_text(encoding="utf-8"))


def kipp_zaehlung(seed, n, m, art, block=4000):
    """Zählung der Kipppunkte auf dem Gitter RHO_GRID (500 Punkte, 0,5 bis 100 000 GE) in Zufallsproblemen mit m Zuständen.

    art "sicher": Lotterie gegen sichere Auszahlung: Zahl der Vorzeichenwechsel von SÄ − K je Problem und ob das Sicherheitsäquivalent auf dem Gitter nie fällt.
    art "riskant": zwei Lotterien mit denselben Wahrscheinlichkeiten, nur nicht dominierte Probleme zählen; Wechsel von SÄ(A) − SÄ(B)."""
    P, A, X = S.zufallsprobleme_m(seed, n, m, art)
    g = len(U.RHO_GRID)
    zaehler = np.zeros(8, dtype=int)
    mono = True
    dom_wechsel = 0
    groesster = 0
    zwischen = 0
    for i in range(0, n, block):
        sl = slice(i, min(i + block, n))
        ds = np.empty((sl.stop - sl.start, g))
        if art == "sicher":
            ces = np.empty_like(ds)
            for j, r in enumerate(U.RHO_GRID):
                ces[:, j] = U.ce(A[sl], P[sl], r)
            mono = mono and bool(np.all(np.diff(ces, axis=1) >= -U.TIE))
            ds = ces - X[sl][:, None]
            zwischen += int(np.sum((X[sl] > A[sl].min(axis=1)) & (X[sl] < A[sl].max(axis=1))))
            w = U.wechsel_zaehlen(ds)
            zaehler[0] += int(np.sum(w == 0))
            zaehler[1] += int(np.sum(w == 1))
            zaehler[2] += int(np.sum(w >= 2))
            groesster = max(groesster, int(w.max()))
        else:
            for j, r in enumerate(U.RHO_GRID):
                ds[:, j] = U.ce(A[sl], P[sl], r) - U.ce(X[sl], P[sl], r)
            w = U.wechsel_zaehlen(ds)
            dom = S.probleme_dominiert_m(A[sl], X[sl])
            nd = ~dom
            dom_wechsel += int(np.sum(w[dom] != 0))
            zaehler[0] += int(np.sum((w == 0) & nd))
            zaehler[1] += int(np.sum((w == 1) & nd))
            zaehler[2] += int(np.sum((w >= 2) & nd))
            zaehler[3] += int(nd.sum())
            groesster = max(groesster, int(w[nd].max()) if nd.any() else 0)
    if art == "sicher":
        return dict(art=art, m=m, seed=int(seed), n=int(n), mono=mono, w0=int(zaehler[0]), w1=int(zaehler[1]), w2plus=int(zaehler[2]), zwischen=zwischen, groesster=groesster)
    return dict(art=art, m=m, seed=int(seed), n=int(n), nicht_dominiert=int(zaehler[3]), w0=int(zaehler[0]), w1=int(zaehler[1]), w2plus=int(zaehler[2]), groesster=groesster, dominiert_mit_wechsel=dom_wechsel)


# --- Experiment 5: Näherung und Potenznutzen --------------------------------------------------------------------------------------------------

def naeherung_strom(seed, n, rho):
    """Anteil nicht dominierter Probleme, in denen die Näherung μ − σ²/(2ρ) anders wählt als das exakte Sicherheitsäquivalent, über die Staffel ρ; dazu bei rho der Anteil unter den
    Problemen, in denen die Risikoeinstellung die Wahl ändert."""
    P, A, dom = S.zufallsprobleme(seed, n)
    nd = ~dom
    ew = U.wahl_vorzeichen(P, A, math.inf)

    def anders(r):
        ex, na = U.wahl_vorzeichen(P, A, float(r)), U.wahl_naeherung(P, A, float(r))
        return float(np.sum(nd & (ex != na) & (ex != 0) & (na != 0)) / max(int(nd.sum()), 1))

    ex = U.wahl_vorzeichen(P, A, float(rho))
    na = U.wahl_naeherung(P, A, float(rho))
    kippend = nd & (ex != ew) & (ex != 0) & (ew != 0)
    return dict(seed=int(seed), n=int(n), n_nd=int(nd.sum()), staffel={r: anders(r) for r in C.RHO_STAFFEL}, anteil=anders(rho), n_kippend=int(kippend.sum()),
                n_kippend_falsch=int(np.sum(kippend & (ex != na) & (na != 0))))


def kalibrierung_zelle(seed, n, gamma, w):
    """Potenznutzen (γ, Vermögen w), am Kapazitätsfall auf dasselbe Sicherheitsäquivalent kalibriert: Risikotoleranz ρ_k und Anteil nicht dominierter Probleme mit anderer Wahl."""
    x, p = np.array(S.CAP_X), np.array(S.CAP_P2)
    rho_k = W.kalibriertes_rho(x, p, float(w), float(gamma))
    P, A, dom = S.zufallsprobleme(seed, n)
    nd = ~dom
    e = U.wahl_vorzeichen(P, A, rho_k)
    q = W.wahl_potenz(P, A, float(w), float(gamma))
    return dict(seed=int(seed), n=int(n), gamma=int(gamma), w=int(w), rho_k=float(rho_k), n_nd=int(nd.sum()), anteil=float(np.sum(nd & (e != q) & (e != 0) & (q != 0)) / int(nd.sum())))


def naeherung_analyse(seed, n, rho, gamma, w):
    zweiter = C.SEED_BAND if int(seed) != C.SEED_BAND else C.DEFAULT_SEED
    return dict(rho=float(rho), a=naeherung_strom(seed, n, rho), b=naeherung_strom(zweiter, n, rho), kal_a=kalibrierung_zelle(seed, n, gamma, w), kal_b=kalibrierung_zelle(zweiter, n, gamma, w))


def verdict_naeherung(z):
    a, b, ka, kb = z["a"], z["b"], z["kal_a"], z["kal_b"]
    kippend = f" (unter den {zahl(a['n_kippend'])} Problemen, in denen ρ die Wahl gegenüber dem Erwartungswert ändert, liegt sie in {zahl(a['n_kippend_falsch'])} falsch)" if a["n_kippend"] else ""
    return "info", (
        f"Näherung μ − σ²/(2ρ): bei ρ = {de(z['rho'], 0)} GE wählt sie in **{pct(a['anteil'])}** der {zahl(a['n_nd'])} nicht dominierten Probleme anders als das exakte Sicherheitsäquivalent{kippend} "
        f"(zweiter Strom: {pct(b['anteil'])}; bei ρ = 100 / 300 GE im ersten Strom {pct(a['staffel'][100])} / {pct(a['staffel'][300])}). "
        f"Potenznutzen γ = {ka['gamma']} bei Vermögen {zahl(ka['w'])} GE, am Kapazitätsfall auf ρ = {de(ka['rho_k'], 1)} GE kalibriert: in **{pct(ka['anteil'], 2)}** der Probleme eine andere Wahl als beim Exponentialnutzen (zweiter Strom {pct(kb['anteil'], 2)}); "
        "gemessen nur in diesem Zahlenraum (Auszahlungen −100 bis 200 GE, Vermögen ab 500 GE) und nur für dieses Bezugsrisiko."
    )

"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Ein Verdikt ist eine reine Funktion (Art, Text) und
bekommt Kennzahlen, keine Zufalls-Seeds."""

import rae_evaluation as ev


def _lot(**kw):
    a = dict(x1=200.0, x2=-100.0, p1=0.5, rho=200.0, ew=50.0, sae=-1.65, praemie=51.65, naeherung=-6.25, praemie_naeherung=56.25, fehler=8.9, sigma=150.0, sicher=False)
    a.update(kw)
    return a


def test_de_formatiert_mit_komma_minuszeichen_und_tausendertrenner():
    assert ev.de(555.8014, 2) == "555,80" and ev.de(-1.653, 2) == "−1,65" and ev.de(-0.001, 2) == "0,00" and ev.de(28730.083, 0) == "28 730"
    assert ev.de(None) == "–" and ev.de(float("inf")) == "–" and ev.zahl(200000) == "200 000" and ev.pct(0.1362, 1) == "13,6 %"
    assert ev.pm(8.9) == "+8,9 %" and ev.pm(-0.4) == "−0,4 %"


def test_verdict_sae_nennt_aequivalent_praemie_und_naeherung():
    art, text = ev.verdict_sae(_lot())
    assert art == "info"
    assert "ρ = 200 GE" in text and "**−1,65 GE**" in text and "Erwartungswert 50,00" in text and "**51,65 GE**" in text and "56,25 (+8,9 %)" in text


def test_verdict_sae_sichere_auszahlung_hat_praemie_null_und_keine_naeherung():
    art, text = ev.verdict_sae(_lot(x1=40.0, x2=40.0, sicher=True, ew=40.0, sae=40.0, praemie=0.0, fehler=None))
    assert art == "info" and "Risikoprämie ist 0" in text and "Näherung" not in text


def _kipp(**kw):
    k = dict(sicher=True, a=(200.0, -100.0), b=(30.0, 30.0), p1=0.5, rho=200.0, ce_a=-1.65, ce_b=30.0, ew_a=50.0, ew_b=30.0, wahl_sae=-1, wahl_ew=1, sae_verlust=31.65, punkte=[555.8],
             status="ok", vorz_unten=-1, vorz_oben=1, spanne=300.0, vorsprung=20.0, naeherung=562.5, min_a=-100.0, min_b=30.0)
    k.update(kw)
    return k


def test_verdict_kipp_sicher_nennt_rho_stern_spanne_verhaeltnis_und_saeverlust():
    art, text = ev.verdict_kipp(_kipp())
    assert art == "success"
    assert "**ρ* = 555,80 GE**" in text and "Spanne 300 GE" in text and "ρ*/Spanne **1,85**" in text and "Vorsprung des Erwartungswerts 20,00 GE" in text
    assert "562,50 GE (+1,2 %)" in text and "wird B gewählt" in text and "verliert 31,65 GE" in text


def test_verdict_kipp_ohne_saeverlust_nennt_keinen():
    art, text = ev.verdict_kipp(_kipp(wahl_sae=1, wahl_ew=1, sae_verlust=0.0, rho=1000.0))
    assert art == "success" and "wird A gewählt" in text and "SÄ-Verlust" not in text


def test_verdict_kipp_ohne_rho_stern_je_status():
    art, text = ev.verdict_kipp(_kipp(status="k_bleibt", punkte=[], ew_a=25.0))
    assert art == "info" and "bei jeder Risikotoleranz gewählt" in text and "(25,00 GE)" in text and "kein ρ*" in text
    art, text = ev.verdict_kipp(_kipp(status="lotterie_immer", punkte=[], min_a=40.0))
    assert art == "info" and "mindestens so gut" in text and "(40 GE)" in text
    art, text = ev.verdict_kipp(_kipp(status="unendlich", punkte=[], vorsprung=1e-9))
    assert art == "warning" and "10¹² GE" in text


def _riskant(**kw):
    k = _kipp(sicher=False, a=(-70.0, 70.0), b=(10.0, -80.0), p1=0.6, rho=50.0, punkte=[28.87, 72.18], vorz_unten=1, vorz_oben=1, ew_a=-14.0, ew_b=-26.0, min_a=-70.0, min_b=-80.0, spanne=150.0,
              vorsprung=12.0, naeherung=None, status="ok", sae_verlust=0.0, wahl_sae=-1)
    k.update(kw)
    return k


def test_verdict_kipp_zwei_punkte_beschreibt_bereiche_und_die_obwohl_aussage():
    art, text = ev.verdict_kipp(_riskant())
    assert art == "success" and text.startswith("2 Kipppunkte")
    assert "unter 28,87 GE A, zwischen 28,87 und 72,18 GE B, über 72,18 GE A" in text
    assert "obwohl A im Mittel (−14,00 gegen −26,00 GE) und im schlechtesten Fall besser ist" in text
    assert "ρ*/Spanne: 0,19, 0,48" in text


def test_verdict_kipp_ein_punkt_und_obwohl_nur_bei_besserem_mittel_und_schlechtestem_fall():
    art, text = ev.verdict_kipp(_riskant(punkte=[40.0], vorz_oben=-1))
    assert text.startswith("Ein Kipppunkt") and "unter 40,00 GE A, über 40,00 GE B" in text and "obwohl" not in text           # ein Punkt: keine „obwohl“-Aussage
    art, text = ev.verdict_kipp(_riskant(ew_a=-30.0, ew_b=-26.0))
    assert "obwohl" not in text                                                                                               # A ist im Mittel nicht besser


def test_verdict_kipp_ohne_punkt_nennt_den_gewinner_oder_gleichstand():
    art, text = ev.verdict_kipp(_riskant(punkte=[], vorz_unten=-1, vorz_oben=-1))
    assert art == "info" and "keinen Kipppunkt" in text and "B wird bei jedem ρ gewählt" in text
    art, text = ev.verdict_kipp(_riskant(punkte=[], vorz_unten=0, vorz_oben=0))
    assert "gleich bewertet" in text


def test_verdict_fall_vergleich_nennt_beide_faelle_spanne_und_vorsprung():
    z = [dict(name="Fall 1", verlust=100.0, ew=50.0, vorsprung=20.0, spanne=300.0, rho_stern=555.8, sae=-1.65, wahl="K", sae_verlust=31.65),
         dict(name="Fall 2", verlust=50.0, ew=75.0, vorsprung=45.0, spanne=250.0, rho_stern=158.05, sae=38.24, wahl="G", sae_verlust=0.0)]
    art, text = ev.verdict_fall_vergleich(z, 200.0)
    assert art == "info"
    assert "ρ = 200 GE wählt Fall 1" in text and "ρ* = 555,80) K" in text and "Fall 2 (Verlust 50, Erwartungswert 75,00, ρ* = 158,05) G" in text
    assert "(300 und 250 GE)" in text and "(20,00 und 45,00 GE)" in text


def _verm(**kw):
    v = dict(w=2000.0, gamma=2, rho=200.0, ce_exp=-1.65, wahl_exp="K", definiert=True, ce_pow=39.02, wahl_pow="G", kipp_w=1075.0)
    v.update(kw)
    return v


def test_verdict_vermoegen_nennt_beide_wahlen_und_das_kippvermoegen():
    art, text = ev.verdict_vermoegen(_verm(), {1: 522.5, 2: 1075.0, 4: 2184.9478})
    assert art == "info" and "wählt **K** bei jedem Vermögen" in text and "−1,65 GE" in text and "γ = 2 bei Vermögen 2000 GE" in text
    assert "39,02 GE, wählt **G**" in text and "1075,0 GE" in text
    assert "2184,95 GE" in ev.verdict_vermoegen(_verm(gamma=4), {1: 522.5, 2: 1075.0, 4: 2184.9478})[1]


def test_verdict_vermoegen_undefiniert_meldet_statt_zahl():
    art, text = ev.verdict_vermoegen(_verm(w=100.0, definiert=False, ce_pow=None, wahl_pow=None), {1: 522.5, 2: 1075.0, 4: 2184.9478})
    assert art == "warning" and "nicht definiert" in text and "wählt K" in text


def _strom(seed, anteil, n_nd=99389):
    return dict(seed=seed, n=200000, n_nd=n_nd, dominiert=0.503, rho=100.0, skala=1.0, anteil=anteil, staffel={r: 0.1 for r in (10, 20, 50, 100, 300, 1000)},
                skalen={0.1: 0.015, 0.5: 0.07, 1.0: anteil, 2.0: 0.22, 10.0: 0.36},
                verhaeltnis=dict(n_kippt=38768, anteil_kippt=0.39, p10=0.07, median=0.35, p90=2.0, unter_spanne=0.8, unter_halb=0.63, ueber_fuenf=0.04, ab_kapazitaet=0.109, werte=[]))


def test_verdict_zufall_nennt_anteile_skalen_und_den_kapazitaetsfall_als_ausnahme():
    art, text = ev.verdict_zufall(dict(a=_strom(101, 0.1363), b=_strom(20261116, 0.1371)))
    assert art == "info"
    assert "ρ = 100 GE, Auszahlungen ×1" in text and "**13,6 %** der 99 389 nicht dominierten Probleme (von 200 000, Zufalls-Seed 101)" in text
    assert "Zufalls-Seed 20261116: 13,7 %" in text and "×0,1 / ×1 / ×10: 1,5 % / 13,6 % / 36,0 %" in text
    assert "39,0 % der Probleme" in text and "Median 0,35 (P90 2,00)" in text and "(1,85) wird nur von 10,9 %" in text and "nicht der typische Fall" in text


def test_verdict_naeherung_nennt_beide_anteile_und_die_grenze_der_gueltigkeit():
    a = dict(seed=101, n=200000, n_nd=99389, staffel={100: 0.0316, 300: 0.0044}, anteil=0.0316, n_kippend=13549, n_kippend_falsch=1254)
    b = dict(a, seed=20261116, anteil=0.0322)
    ka = dict(seed=101, n=200000, gamma=4, w=500, rho_k=125.6, n_nd=99389, anteil=0.0078)
    kb = dict(ka, seed=20261116, anteil=0.0072)
    art, text = ev.verdict_naeherung(dict(rho=100.0, a=a, b=b, kal_a=ka, kal_b=kb))
    assert art == "info"
    assert "bei ρ = 100 GE" in text and "**3,2 %**" in text and "13 549 Problemen" in text and "in 1 254 falsch" in text and "zweiter Strom: 3,2 %" in text
    assert "γ = 4 bei Vermögen 500 GE" in text and "ρ = 125,6 GE" in text and "**0,78 %**" in text and "zweiter Strom 0,72 %" in text and "nur in diesem Zahlenraum" in text

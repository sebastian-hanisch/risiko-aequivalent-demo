# Risikoeinstellung: Sicherheitsäquivalent und Kipppunkt – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-risiko-aequivalent-demo.streamlit.app/)** -->

Demo der Konzepte-Gruppe „Entscheidungsanalyse“ für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Gruppe fragt, was man tut, wenn man unter Unsicherheit wählen muss und die Wahl an Zahlen hängt, die man erst
bestimmen muss. Diese Demo hebt die Annahme auf, die die einfachste Rechnung stillschweigend macht: dass der Entscheider jede Lotterie mit ihrem **Erwartungswert** bewertet. Stattdessen bewertet er sie mit ihrem
**Sicherheitsäquivalent** bei einem **Exponentialnutzen**, und die Wahl hängt dann an einer Zahl, der **Risikotoleranz ρ** in Geldeinheiten (GE). Die Nachbar-Demo [kipp-schwelle-demo](https://github.com/sebastian-hanisch/kipp-schwelle-demo) fragt bei
Erwartungswert-Bewertung, bei welcher *Wahrscheinlichkeit* die Wahl kippt; hier fragt die Demo, bei welcher *Risikotoleranz* sie kippt. Weitere Nachbarn:
[information-risikoaversion-demo](https://github.com/sebastian-hanisch/information-risikoaversion-demo) (was Information unter Risikoaversion wert ist),
[nutzen-erheben-demo](https://github.com/sebastian-hanisch/nutzen-erheben-demo) (wie man ρ erhebt und wie unsicher Antworten sind) und
[erwartungsnutzen-grenze-demo](https://github.com/sebastian-hanisch/erwartungsnutzen-grenze-demo) (wo der Erwartungsnutzen selbst endet).

**Das Beispiel ist ausgedacht.** Der **Kapazitätsfall**: groß bauen (G) bringt +200 GE bei hoher und −100 GE bei niedriger Nachfrage, je mit Wahrscheinlichkeit ½; klein oder mieten (K) bringt sicher +30 GE. Die Auszahlungen sind erfunden: das Beispiel ist ein Modell, kein
Fall mit Daten. Eine Investition mit Auszahlungen +100 | −60 gegen +5 sicher (P = 0,45) kommt nur als eine Zeile vor (B3).

## Warum dieses Problem

Eine Zahl wie „ρ = 200 GE“ klingt wie eine Eigenschaft des Entscheiders. Sie ist es nur mit Einheit und Bezug: im Modell legt ρ die Nutzenfunktion vollständig fest (absolute Risikoaversion 1/ρ je GE), aber was die Zahl für eine Wahl heißt, hängt an der Spanne und am Vorsprung der Auszahlungen. Im Kapazitätsfall (Erwartungswerte 50 gegen 30, Spanne 300 GE) wählt der Entscheider mit Exponentialnutzen erst ab **ρ\* = 555,80 GE** G.
Bei Verlust 50 statt 100 (Erwartungswert 75, Vorsprung 45 statt 20) kippt dieselbe Wahl schon bei ρ\* = 158,05 GE: dieselbe Zahl ρ = 200 GE wählt im ersten Fall K, im zweiten G. Kernfragen der Demo: (1) Was ist eine Lotterie dem Entscheider wert?
(2) Ab welcher Risikotoleranz kippt die Wahl, und was bedeutet die Zahl? (3) Wie oft spielt das überhaupt eine Rolle? Alle Aussagen gelten im Kapazitätsfall, bei Exponentialnutzen, bei zwei Zuständen (Hauptfall) und für den hier beschriebenen Generator der Zufallsprobleme; Zähleinheit sind
**nicht dominierte Zufallsprobleme** (Ausnahme: die Kipppunkt-Zählung gegen eine sichere Alternative in B8 zählt alle Probleme, dort gibt es keine Auswahl nach Dominanz). Alles hier ist Lehrbuchwissen; die Demo macht es mit Zähleinheit und Zahl vorführbar.

## Modell

**Konvention.** Auszahlungen x in Geldeinheiten (GE), Gewinn positiv. Nutzen u(x) = 1 − e^(−x/ρ), ρ > 0 die Risikotoleranz in GE (konstante absolute Risikoaversion 1/ρ). **Sicherheitsäquivalent** SÄ = −ρ ln E[e^(−X/ρ)]: der sichere Betrag, den der Entscheider der Lotterie
gleichstellt. **Risikoprämie** = Erwartungswert − SÄ. Grenzfälle: ρ → ∞ gibt den Erwartungswert, ρ → 0 die kleinste Auszahlung. **Spanne** = größte minus kleinste Auszahlung beider Alternativen (im Kapazitätsfall 300 GE). **Vorsprung** = Erwartungswert der riskanten minus Wert der sicheren Alternative.
**Kipp-Risikotoleranz ρ\*** = die Risikotoleranz, bei der zwei Alternativen gleich gut sind. **SÄ-Verlust** = Unterschied der Sicherheitsäquivalente vor der Entscheidung, wenn man statt der Wahl nach dem Sicherheitsäquivalent die Wahl nach dem Erwartungswert trifft;
er ist kein Vergleich mit einem Orakel, das den Zustand kennt (das Wort „Regret“ meint im Bestand etwas anderes, nämlich Mehrkosten gegen ein solches Orakel). „Andere Wahl“: bei ρ hat die andere Alternative das größere SÄ als beim Erwartungswert; Gleichstände (Unterschied unter 1e-9 GE) zählen nicht als Wahl.
Die Formeln stehen im 📐-Expander der App (Lesehilfe mit Symboltabelle).

**Zufallsprobleme.** Zwei Alternativen, zwei Zustände, P(Zustand 1) gleichverteilt in 0,1 bis 0,9, jede Auszahlung ganzzahlig gleichverteilt in −100 bis 200 GE. *Dominiert* heißt: eine Alternative ist in beiden Zuständen mindestens so gut wie die andere (50,3 % der Probleme,
B9); gezählt wird nur unter den übrigen. Für die Kipppunkt-Zählung (B8, alle Zustandszahlen, auch zwei): Wahrscheinlichkeiten Dirichlet(1, …, 1), bei zwei Zuständen also P(Zustand 1) gleichverteilt in 0 bis 1, gleiche Wahrscheinlichkeiten in beiden Alternativen; jede Auszahlung, auch die sichere, ganzzahlig gleichverteilt in −100 bis 200 GE wie oben. Der Zufall kommt nur aus **SplitMix64** (`rae_rng.py`, ganzzahlig, auf jeder Plattform dieselbe Folge); Zufalls-Seeds 101 und 20261116 für den Generator oben, 1301 und 1302 für die Kipppunkt-Zählung (B8).

**Potenznutzen** (nur Experiment 3 und 5): u(v) = v^(1−γ)/(1−γ) (γ = 1: ln v) auf dem Endvermögen v = w + x, verlangt w + x > 0.

## Methodik

1. **Sicherheitsäquivalent von Hand** (Experiment 1): Nutzenkurve mit Sehne, Erwartungswert, SÄ, Risikoprämie und die Näherung μ − σ²/(2ρ) daneben mit Fehler. Dazu eine Frage-Eingabe für ρ (rauschfrei: „Gewinn g·L gegen Verlust L: bis zu welchem L nehmen Sie an?“).
2. **Kippt die Wahl, und wann?** (Experiment 2): gegen eine *sichere* Alternative wird ρ\* durch Bisektion von ln ρ bestimmt (das SÄ wächst in allen gezählten Problemen mit ρ, B8; dann gibt es höchstens eine Nullstelle); zwischen zwei *riskanten* Alternativen wird ein Gitter (500 Punkte, logarithmisch von 0,5 bis
   100 000 GE) nach Vorzeichenwechseln von SÄ(A) − SÄ(B) abgesucht und jeder Wechsel durch Bisektion genau bestimmt. Die Schwelle der Wahrscheinlichkeit p\*(ρ) ist geschlossen (die Gleichung ist linear in p). ρ\* über den Vorsprung für die Familie G = (+200 | −L) gegen K = +30, p = ½.
3. **Vermögen** (Experiment 3): Exponentialnutzen gegen Potenznutzen am Kapazitätsfall; Kippvermögen (Anfangsvermögen, bei dem der Potenznutzen K und G gleich bewertet, B12) durch Bisektion, exakt in Brüchen belegt.
4. **Wie oft ändert die Risikoeinstellung die Wahl?** (Experiment 4): bis 200 000 Zufallsprobleme je Strom, zwei Ströme für das Band, Skala der Auszahlungen als Regler; Verteilung von ρ\*/Spanne; die Kipppunkt-Zählung für zwei bis vier Zustände ist **vorgerechnet** (`tools/rae_make_frozen.py`, `data/rae_frozen.json`).
5. **Reicht eine Näherung?** (Experiment 5): Anteil anderer Wahl bei μ − σ²/(2ρ) gegen exakt; Potenznutzen, am Kapazitätsfall auf dasselbe Sicherheitsäquivalent von G kalibriert (ρ so gewählt, dass der Exponentialnutzen G dasselbe SÄ gibt wie der Potenznutzen bei diesem Vermögen); alle neun Fälle (γ = 1, 2, 4; Vermögen 500, 1000, 5000 GE) sind **vorgerechnet**.

Rechnung: Sicherheitsäquivalent über `log1p`/`expm1` für |x/ρ| bis 3 und mit Verschiebung um das Maximum darüber (kein Überlauf bei kleinem ρ, keine Auslöschung bei großem ρ); nur numpy und die Standardbibliothek in der App, scipy und `fractions` nur in den Tests. Live-Rechnungen mit 200 000 Problemen und zwei Strömen dauern einige Sekunden
(`st.cache_data`, Spinner mit Hinweis); alles Schwerere ist vorgerechnet.

## Befunde (gemessen, keine Behauptungen)

Jede Zahl steht in einem Test (Spalte „Test“, Datei `tests/test_claims.py`, wo nichts anderes steht). Zählungen aus Neu-Rechnung gelten als Band (± 2, Anteile auf ein halbes Zehntel Prozent), exakt ist nur, was in Brüchen gerechnet ist.
Alle Zeilen mit Zufallsproblemen außer der Kipppunkt-Zählung B8 (eigene Ströme, siehe dort): Zufalls-Seeds 101 und 20261116, je 200 000 Probleme, davon 99 389 bzw. 99 388 nicht dominiert (50,3 % dominiert).

| Nr | Frage | Befund (Bedingung) | Test |
|---|---|---|---|
| B1 | Sicherheitsäquivalent der Lotterie (0 \| 100 GE, je ½) | bei ρ = 100 GE 37,99 GE; Auszahlungen und ρ verdoppelt (0 \| 200, ρ = 200): 75,98 GE, das Doppelte (Skaleninvarianz); [kuchen-vergroessern-demo](https://github.com/sebastian-hanisch/kuchen-vergroessern-demo) nennt 75,98 für dasselbe Beispiel (dort γ = 5 ‰ je Euro, Querprobe, nichts kopiert) | `test_b1_*`, `test_utility.py` (Brüche) |
| B2 | Kapazitätsfall bei ρ = 200 GE | Erwartungswert 50,00, SÄ von G −1,65, Risikoprämie 51,65; die Näherung σ²/(2ρ) sagt 56,25 (+8,9 %). Fehler der Näherung der Prämie bei ρ = 50 / 100 / 200 / 300 / 555,8 / 1000 / 5000: +94,9 / +31,5 / +8,9 / +4,1 / +1,2 / +0,4 / +0,01 % | `test_b2_*` |
| B3 | Kipp-Risikotoleranz ρ\* | Kapazitätsfall (G = +200 \| −100 gegen K = +30, p = ½): **ρ\* = 555,80 GE**, Spanne 300, ρ\*/Spanne 1,85; bei Verlust 50: 158,05 GE (0,63); Goldener-Schnitt-Fall (G = +200 \| −100, p = ½, gegen 0 sicher): 100/ln φ = 207,81 GE (mit z = e^(−100/ρ) wird daraus z³ + 1 = 2z, nach Abspalten von z = 1 also z² + z − 1 = 0, z = 1/φ, φ = (1 + √5)/2); drei Zustände (G = (250, 60, −150) gegen B = (90, 50, 0), p = (0,3; 0,4; 0,3)): 1631,16 GE; Investition (+100 \| −60 gegen +5, p = 0,45): 444,87 GE, ρ\*/Spanne 2,78. Drei Rechenwege (exakte Bisektion in Brüchen auf einem Polynom vom Grad 30 in z = e^(−u₀/ρ), wobei u₀ = 10 GE der größte gemeinsame Teiler der Auszahlungen 200, −100 und 30 ist und das Polynom z³⁰ + 1 = 2 z¹³ lautet; die Wurzel z = 1 ist trivial (sie entspricht ρ → ∞), gesucht ist die Wurzel mit 0 < z < 1, ρ\* = −u₀/ln z, und im Test gibt es dort genau einen Vorzeichenwechsel; dazu die Rechnung der Demo und `numpy.roots`) stimmen auf relativ 1e-9 bzw. 1e-6 | `test_b3_*`, `test_oracle_risiko_aequivalent.py` |
| B4 | Dieselbe Zahl ρ, andere Wahl | ρ = 100 GE: beide Fälle wählen K; **ρ = 200 GE: Fall 1 wählt K, Fall 2 wählt G** (SÄ von G 38,24); ρ = 300: wie 200; ρ = 600: beide G. Wer bei ρ = 200 im Fall 1 trotzdem G wählt, hat einen SÄ-Verlust von 31,65 GE | `test_b4_*` |
| B5 | ρ\* über den Vorsprung (Familie G = (+200 \| −L) gegen +30, p = ½; 44 verschiedene Stützstellen L = 10, 13, …, 139 in Dreierschritten; Vorsprung 45 (L = 50) liegt nicht auf diesem Gitter und ist einzeln gerechnet) | In dieser Familie wächst ρ\* auf diesen Stützstellen monoton, wenn der Vorsprung schrumpft (mit L ändert sich auch die Streuung, der Vorsprung allein bestimmt ρ\* also nicht): Vorsprung 65 / 45 / 35 / 20 / 5 / 0,5 GE gibt ρ\* = 60,35 / 158,05 / 248,47 / 555,80 / 2 720,83 / 28 730,08 GE, ρ\*/Spanne 0,29 / 0,63 / 0,92 / 1,85 / 8,24 / 84,75. Näherung Spanne²/(8·Vorsprung) liegt um +40,5 / +9,8 / +4,8 / +1,2 / +0,1 / 0,0 % über ρ\*. Auszahlungen ×c gibt ρ\* ×c (c = 0,1; 10; 1000; relativ 1e-9) | `test_b5_*` |
| B6 | Schwelle der Wahrscheinlichkeit p\*(ρ) im Kapazitätsfall | 13/30 = 0,4333 bei ρ = ∞; 0,4407 (5000); 0,4703 (1000); 0,5000 bei ρ\* = 555,80; 0,5563 (300); 0,6152 (200); 0,7656 (100); 0,9280 (50) | `test_b6_*`, `test_utility.py` |
| B7 | Mehr als ein Kipppunkt zwischen zwei riskanten Alternativen | Beispiel p = (0,6; 0,4), A = (−70 \| +70), B = (+10 \| −80): Erwartungswert A −14 gegen −26, schlechtester Fall −70 gegen −80, trotzdem gewinnt B für ρ zwischen **28,87 und 72,18 GE**; exakt in Brüchen (mit z = e^(−10/ρ), 10 GE als größtem gemeinsamem Teiler der Auszahlungen: z = 1/2, also ρ = 14,4 GE, wählt A; z = 4/5, ρ = 44,8, wählt B; z = 9/10, ρ = 94,9, wählt A) | `test_b7_*`, `test_oracle_*`, `test_utility.py` |
| B8 | Wie viele Kipppunkte? (vorgerechnet: je 40 000 Probleme, Zufalls-Seeds 1301 und 1302, ρ-Gitter 500 Punkte von 0,5 bis 100 000 GE; Zufallsprobleme mit Dirichlet-Wahrscheinlichkeiten, bei zwei Zuständen also P(Zustand 1) in 0 bis 1 statt in 0,1 bis 0,9 wie im Generator der übrigen Zeilen) | **Gegen eine sichere Alternative: 0 von 240 000 Problemen mit zwei oder mehr Wechseln** (alle 240 000 = 6 × 40 000 Probleme, ohne Auswahl nach Dominanz; in 114 022 davon liegt die sichere Auszahlung zwischen der kleinsten und der größten Auszahlung der Lotterie, nur dort ist ein Wechsel möglich; genau einen Wechsel haben 56 991 der 240 000; 2, 3 und 4 Zustände, sechs Läufe; das SÄ wächst auf dem Gitter in allen Problemen mit ρ; Gitterbefund, kein Beweis). **Zwischen zwei riskanten Alternativen** (nicht dominiert) mindestens zwei Wechsel in 154 von 19 798 und 176 von 20 092 Problemen bei zwei Zuständen (0,78 und 0,88 %), 696 von 29 969 und 681 von 29 828 bei drei (2,32 und 2,28 %), 1 150 von 34 989 und 1 155 von 35 016 bei vier (3,29 und 3,30 %); größte gefundene Zahl 2, 3 und 3 | `test_b8_*`, `test_frozen.py` |
| B9 | Wie oft ändert die Risikoeinstellung die Wahl? (Zufallsprobleme, nicht dominiert) | Anteil mit anderer Wahl als beim Erwartungswert bei ρ = 10 / 20 / 50 / 100 / 300 / 1000 GE: Zufalls-Seed 101: 36,0 / 32,4 / 22,6 / 13,6 / 4,9 / 1,5 %; Zufalls-Seed 20261116: 36,1 / 32,5 / 22,7 / 13,7 / 4,9 / 1,4 %. **Skala:** bei ρ = 100 und Auszahlungen ×0,1 / ×0,5 / ×1 / ×2 / ×10: 1,5 / 7,2 / 13,6 / 22,6 / 36,0 % (Zufalls-Seed 101); das sind dieselben Zahlen wie die Spalten ρ = 1000 / 200 / 100 / 50 / 10 (Skaleninvarianz; höchstens 2 Abweichungen der Wahl in 99 389 Problemen) | `test_b9_*` |
| B10 | Wo liegt ρ\* im Verhältnis zur Spanne? | In 39,0 % der nicht dominierten Probleme (38 768 von 99 389; zweiter Strom 39,3 %) wählen ρ → 0 (genauer ρ = 0,5 GE) und ρ = ∞ verschieden, dort gibt es mindestens ein ρ\*; in weiteren 1,0 % (994 von 99 389) wechselt die Wahl zweimal (zwei ρ\*, bei gleicher Wahl an beiden Enden). Für die 39,0 % mit verschiedener Wahl (Bezugsmenge der folgenden Prozente: diese 38 768 Probleme, nicht alle 99 389) ρ\*/Spanne: P10 0,08, **Median 0,35**, P90 2,0 (2,02 und 1,97); ρ\* unter einer Spanne in 80,4 %, unter einer halben in 62,7 %, über fünf Spannen in 4,0 %; mindestens 1,853 (Kapazitätsfall) in 10,9 % (zweiter Strom 10,7 %). **Der Kapazitätsfall ist ein Beispiel mit kleinem Vorsprung, nicht der typische Fall** | `test_b9_rho_stern_im_verhaeltnis_zur_spanne` |
| B11 | Mittelwert-Varianz-Näherung μ − σ²/(2ρ) | Anteil nicht dominierter Probleme mit anderer Wahl als beim exakten SÄ bei ρ = 10 / 20 / 50 / 100 / 300 / 1000 GE: 12,7 / 10,6 / 6,8 / 3,2 / 0,4 / 0,04 % (Zufalls-Seed 101; zweiter Strom 12,8 / 10,5 / 6,9 / 3,2 / 0,5 / 0,04 %). Unter den 13 549 Problemen, in denen ρ = 100 die Wahl gegenüber dem Erwartungswert ändert, liegt die Näherung in 1 254 (9,3 %) falsch | `test_b10_*` |
| B12 | Exponential gegen Potenz | Kippvermögen (das Anfangsvermögen w, bei dem der Potenznutzen K und G gleich bewertet; darüber wählt er G, darunter K) des Kapazitätsfalls bei Potenznutzen: **522,5 GE (γ = 1), 1 075,0 (γ = 2)**, beide exakt in Brüchen, und 2 184,95 (γ = 4, numerisch); bei γ = 2 wählt Vermögen 1000 K, Vermögen 2000 G (exakt). Der Exponentialnutzen wählt bei jedem Vermögen dasselbe (Delta-Eigenschaft, exakt in Brüchen; die Abweichung 1,8e-12 einer früheren Gleitkomma-Rechnung war Rundung). **Negativbefund:** am Kapazitätsfall auf dasselbe SÄ kalibriert (ρ = 532,6 / 1041,0 / 5048,1 für γ = 1; 261,1 / 517,8 / 2523,5 für γ = 2; 125,6 / 256,3 / 1261,2 für γ = 4, je bei Vermögen 500 / 1000 / 5000 GE) wählt der Potenznutzen nur in 0,00 bis 0,78 % der Zufallsprobleme anders als der Exponentialnutzen (größter Wert γ = 4, Vermögen 500: 0,78 % und 0,72 %; Vermögen 5000: höchstens 0,005 %; je 200 000 Probleme, zwei Zufalls-Seeds) | `test_b11_*`, `test_oracle_*` |
| B13 | Frage statt Zahl | Lotterie „Gewinn g·L oder Verlust L, je ½, gegen 0 sicher“, indifferent bei ρ/L = 6,02 (g = 1,2); 3,04 (1,5); **2,08 (2; geschlossen 1/ln φ)**; 1,64 (3); 1,52 (4); 1,44 (10). Der größte akzeptable Verlust bestimmt ρ also nur, wenn der Gewinn mitgenannt wird (rauschfrei gerechnet) | `test_b12_frage_statt_zahl` |

Die Preset-Texte der App nennen nur Zahlen aus dieser Tabelle; `tests/test_presets.py` prüft für jedes Preset die erwartete Verdikt-Zeile (kein Preset ohne Effekt meldet einen Effekt).

## Befunde und Korrekturen gegenüber der ersten Rechnung

Die erste Rechnung (ein Wegwerf-Skript mit `numpy.random`) gilt als Erwartung, nicht als Quelle; die Demo zieht aus ihrem eigenen Strom (SplitMix64) und hat neu gemessen. Abweichungen:

1. **Ein Rechenfehler der ersten Fassung dieser Demo wurde gefunden und behoben.** Das Sicherheitsäquivalent über `log1p`/`expm1` verliert Stellen, wenn alle Auszahlungen viele Einheiten ρ groß sind (x/ρ = 25, etwa 100 GE bei ρ = 4: Fehler rund 1e-4 statt 1e-13), weil die Summe nahe −1 liegt. In der Kipppunkt-Zählung erschien dadurch bei zwei Zuständen und Zufalls-Seed 1302 ein scheinbar *fallendes*
   Sicherheitsäquivalent und ein Problem mit 12 Wechseln. Jetzt gilt die Verschiebung ab |x/ρ| > 3; Regressionstest `test_ce_haelt_die_genauigkeit_bei_grossem_betrag_gegen_rho` (schlug auf dem alten Code fehl), die Zählung ist neu gerechnet (B8: 0 von 240 000).
2. **Andere Zufallsströme, gleiche Größenordnung.** Nicht dominierte Probleme: 99 389 statt 99 575 der ersten Rechnung; Anteil bei ρ = 100: 13,6 % (gleich), Skala ×10: 36,0 % statt 35,9 %; P90 von ρ\*/Spanne 2,02 statt 2,03; Anteil der nicht dominierten Zweizustandsprobleme mit mindestens zwei Wechseln zwischen zwei riskanten Alternativen (B8, P(Zustand 1) in 0 bis 1) 0,78 und 0,88 % (erste Rechnung 0,75 %, ein früherer Lauf 0,92 %): als Band „0,7 bis 0,9 %“ dieses Anteils zu lesen.
3. **Der Vorsprung im Fall 2 ist 45, nicht 75.** 75 ist der Erwartungswert von G bei Verlust 50; der Vorsprung gegen K ist 75 − 30 = 45 (B4, B5).
4. **ρ\* ist nur gegen eine sichere Alternative eine einzelne Zahl** (B8, B7); zwischen zwei riskanten Alternativen gibt es bei 0,78 und 0,88 % der nicht dominierten Zweizustandsprobleme mindestens zwei (bei P(Zustand 1) gleichverteilt in 0 bis 1; mit dem Generator der Zufallsprobleme, P(Zustand 1) in 0,1 bis 0,9, sind es 994 von 99 389 nicht dominierten Problemen, also 1,0 %, im Strom mit Zufalls-Seed 101; Gitterzählung, kann eng benachbarte Wechsel übersehen; das Beispiel in B7 ist exakt in Brüchen).
5. **Die Delta-Abweichung 1,8e-12 einer früheren Rechnung ist Gleitkomma-Rundung**, keine Verletzung der Eigenschaft: in Brüchen ist sie exakt 0 (B12).
6. **Der Kapazitätsfall liegt am oberen Rand** (ρ\*/Spanne 1,85 gegen Median 0,35, B10): „weit über der Spanne“ gilt für dieses Beispiel, nicht für Zufallsprobleme.
7. Zusatz gegenüber der ersten Rechnung: ρ = 10 in der Staffel (Anteil anderer Wahl 36,0 %; Näherung 12,7 %) und ρ = 300 in der Tabelle der Risikoprämien.

## Quellen

Nur so zitiert, wie gelesen. Eigenschaften der Exponentialfunktion (konstante absolute Risikoaversion 1/ρ; Sicherheitsäquivalent unabhängig vom Anfangsvermögen, „Delta-Eigenschaft“; additiv über unabhängige Risiken; Normalformel): arXiv 1809.00817, als Zweitquelle gelesen. Näherung ½·σ²·r nach Pratt und
die Aussage „risikoaverser heißt größere Prämie für jedes Risiko“: Gollier, Hammitt und Treich (2013), S. 129 und 139, als Zweitquelle gelesen; die Quelle weist darauf hin, dass daraus nicht folgt, dass der Risikoaversere auch für eine *teilweise* Risikominderung mehr zahlt. **Pratt (1964) und Arrow (1965) im Original nicht gelesen.**
Die Daumenregeln für ρ aus Interviews (ein Sechstel des Eigenkapitals bei Howard 1988, 20 % des Marktwerts bei McNamee und Celona 1990) stehen nur als zitierte Aussage bei Smith (2004), Zusammenfassung; die Originale sind nicht gelesen, die Demo rechnet nichts damit. Alle Zahlen und Formeln dieser README sind eigene Nachrechnung in diesem Modell.

## Ehrliche Grenzen

- **Die Auszahlungen sind ausgedacht**, die Zahlen gelten im Kapazitätsfall, bei Exponentialnutzen, bei zwei Zuständen (Hauptfall) und für diesen Generator der Zufallsprobleme (gleichverteilte Auszahlungen in −100 bis 200 GE, P(Zustand 1) in 0,1 bis 0,9). Andere Verteilungen sind **nicht gemessen**; B9 und B11 hängen an der Skala und an der Größe der Vorsprünge.
  B5 zeigt die Abhängigkeit vom Vorsprung nur für eine Familie.
- **Kipppunkt-Zählung auf einem Gitter** (500 Punkte, zwei Zufalls-Seeds je Fall); eng beieinanderliegende Wechsel können fehlen. Mehr als vier Zustände und ungleiche Wahrscheinlichkeiten in beiden Alternativen sind nicht gemessen. „0 von 240 000“ ist ein Gitterbefund, kein Beweis.
- **Potenznutzen am Kapazitätsfall kalibriert** (B12): 0,00 bis 0,78 % gilt nur in diesem Zahlenraum (Auszahlungen −100 bis 200 GE, Vermögen 500 bis 5000 GE) und für dieses Bezugsrisiko; die Vermögensabhängigkeit wirkt vor allem nahe der Indifferenz (K = 30 liegt dort), in anderen Räumen kann sie größer sein.
- **Der SÄ-Verlust ist ein Unterschied von Sicherheitsäquivalenten vor der Entscheidung** und gilt nur für die gemessene Nutzenfunktion; er sagt nichts über Auszahlungen im Nachhinein.
- **Die Frage-Eingabe ist rauschfrei**: die Antwort wird als genau genommen. Wie sicher man ρ aus Antworten erheben kann, ist hier nicht gemessen; die Sechstel-Regel und ähnliche Daumenregeln sind nicht eingebaut.
- **Plattform:** `exp` und `log` kommen aus der Bibliothek des Systems; Zählungen und Anteile aus Neu-Rechnung gelten deshalb als Band (Gleichstände unter 1e-9 GE zählen nicht). Der Zufallsstrom selbst ist ganzzahlig und auf jeder Plattform gleich.
- **Kein CVaR** und kein Vergleich von Nutzenfunktion und CVaR; Risiko als Ziel mit CVaR zeigt [cvar-demo](https://github.com/sebastian-hanisch/cvar-demo). Die Teilung von Risiko zwischen zwei Seiten (Borch-Regel, Wette) zeigt [kuchen-vergroessern-demo](https://github.com/sebastian-hanisch/kuchen-vergroessern-demo); hier wählt ein einzelner Entscheider zwischen zwei Alternativen.
  EVPI und VSS stehen in [newsvendor-evpi-demo](https://github.com/sebastian-hanisch/newsvendor-evpi-demo); der Informationswert unter Risikoaversion in [information-risikoaversion-demo](https://github.com/sebastian-hanisch/information-risikoaversion-demo).

## Verifikation

- **Kern von Hand:** `tests/test_utility.py` rechnet Lotterie (0 | 100) bei ρ = 100/ln 2 in Brüchen (972 < 1024, 3¹⁰⁰ < 2¹⁵⁹, 3⁵⁰ > 2⁷⁹), den Goldenen-Schnitt-Fall (y³ − 2y + 1 = (y − 1)(y² + y − 1), ρ\* = 100/ln φ), den Kapazitätsfall (z = 49/50 wählt K, z = 99/100 wählt G), die zwei Kipppunkte in Brüchen und die Genauigkeit bei großem x/ρ (Regression, siehe oben).
- **Orakel** (`tests/test_oracle_risiko_aequivalent.py`): ρ\* gegen exakte Bisektion in Brüchen auf dem Polynom in z = e^(−u₀/ρ) (u₀ = größter gemeinsamer Teiler der Auszahlungen in GE: 10 im Kapazitätsfall, 100 im Goldenen-Schnitt-Fall) gegen `numpy.roots` (vier Fälle); Zahl der Kipppunkte exakt gegen Gitter; Sicherheitsäquivalent gegen `scipy.optimize.brentq` auf dem Nutzen, gegen `scipy.integrate.quad` bei Normalverteilung und
  in einer eigenen Fassung über u(x) = −exp(−γx); p\*(ρ) geschlossen gegen „Sicherheitsäquivalente gleich“; Potenz-Kippvermögen exakt in Brüchen und gegen `brentq`; Delta-Eigenschaft exakt; die vektorisierte Wahl in Zufallsproblemen gegen eine Rechnung mit `math.exp` Problem für Problem.
- **Eigenschaften** (`tests/test_eigenschaften_risiko_aequivalent.py`): Skalieren von Auszahlungen und ρ, Verschieben (Delta), Jensen (SÄ zwischen Minimum und Erwartungswert), SÄ wächst mit ρ, Vertauschen und Zusammenlegen von Zuständen, dominierte Alternativen nie gewählt, positiv affiner Nutzen, Additivität über unabhängige Risiken, höchstens ein Wechsel gegen eine sichere Alternative, ρ\* wächst mit schrumpfendem Vorsprung, Zufallsstrom.
- **Randfälle** (`tests/test_edge_cases.py`): ρ → 0 und ρ → ∞, x/ρ über 700, p = 0 oder 1, ein Zustand, K gleich Minimum oder Maximum, Vorsprung ≤ 0, Erwartungswert gleich, Potenznutzen mit w ≤ 100, Reglerränder.
- **Claims** (`tests/test_claims.py`) und vorgerechnete Messung (`tests/test_frozen.py`, `tools/rae_make_frozen.py --check` rechnet einen kleinen Teil neu); **Verdikte** an von Hand gesetzten Werten (`tests/test_verdicts.py`); **Zufallsquelle** (`tests/test_rng.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Knopf, jeder Regler am Minimum und Maximum), Presets und Hervorhebung des aktiven Presets, Footer, Laufzeit-Importe.
- **Linux:** auf Linux (Docker-Container, Python 3.12, numpy 2.5.5) bestehen alle Tests (158 von 158); die Zählungen sind dort als Band geprüft.
- **Testzahl:** 158 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, Kernfragen, fünf Experimente, Grenzen, 📐-Expander |
| `rae_utility.py` | Kern: Sicherheitsäquivalent, Risikoprämie, Normalnäherung, ρ\* (Bisektion), Kipppunkte (Gitter und Bisektion), Schwelle p\* |
| `rae_wealth.py` | Potenznutzen: Sicherheitsäquivalent, Kippvermögen, kalibriertes ρ |
| `rae_scenario.py` | Kapazitätsfall und Generator der Zufallsprobleme (SplitMix64) |
| `rae_rng.py` | SplitMix64 (Kopie der geprüften Vorlage des Website-Repos, dazu der Strom als Feld) |
| `rae_evaluation.py` | Kennzahlen, Messreihen und Verdikte (reine Funktionen) |
| `rae_constants.py` | Regler-Grenzen, Zufalls-Seeds, Farben, Beispielszenarien (Schnellstart) |
| `rae_presets.py` | Permalink, Presets, Hervorhebung des aktiven Presets, Zufalls-Seed-Knopf |
| `rae_layout.py` | Fester Kopf (Stylesheet) |
| `rae_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `data/rae_frozen.json` | vorgerechnete Kipppunkt-Zählung und Potenz-Kalibrierung |
| `tools/rae_make_frozen.py` | erzeugt die Datei (`--check` rechnet einen kleinen Teil neu) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, pandas, plotly) und Tests (pytest, scipy) |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Claims, vorgerechnete Messung, Verdikte, Zufallsquelle, Presets, Layout, AppTest-Rauchtest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Mehrere Perioden, Baum mit Sicherheitsäquivalent im Knoten, stetige Verteilungen:** die Demo bleibt bei endlich vielen Zuständen einer Entscheidung.
- **Erhebung von ρ mit Rauschen, ein Rechner „ρ aus Eigenkapital“:** nur die rauschfreie Umrechnung „Frage zu ρ“ und ein Verweis auf [nutzen-erheben-demo](https://github.com/sebastian-hanisch/nutzen-erheben-demo).
- **Prüfung des Erwartungsnutzens selbst** (Allais, Ellsberg): die Annahme „Erwartungsnutzen“ gilt hier unverändert und wird nicht getestet; das zeigt [erwartungsnutzen-grenze-demo](https://github.com/sebastian-hanisch/erwartungsnutzen-grenze-demo).
- **Informationswert unter Risikoaversion, Rangumkehr bei mehreren Attributen:** andere Fragen, eigene Demos.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die vorgerechnete Messung neu erzeugen: `python tools/rae_make_frozen.py` (rund eine Minute); nur einen kleinen Teil prüfen: `python tools/rae_make_frozen.py --check`.

Gebaut mit Streamlit, Plotly, pandas und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Entscheidungsanalyse: Information, Risiko, Grenzen](https://sebastianhanisch.net/konzepte-entscheidungsanalyse.html).

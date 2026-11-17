"""Erzeugt data/rae_frozen.json: die vorgerechneten Messungen der App (viertes und fünftes Experiment).

Aufruf (aus dem Repo-Ordner, Python des Test-venv):
  python tools/rae_make_frozen.py            erzeugt die Datei: Kipppunkt-Zählung (2, 3 und 4 Zustände, je 40 000 Probleme, zwei Zufalls-Seeds, Lotterie gegen sicher und
                                             gegen riskant) und Kalibrierung des Potenznutzens (3 Werte γ, 3 Vermögen, zwei Zufalls-Seeds, je 200 000 Probleme); rund eine Minute
  python tools/rae_make_frozen.py --check    rechnet je einen kleinen Teil neu (4 000 Probleme je Zählung, 20 000 für die Kalibrierung) und vergleicht ihn mit dem
                                             Teil „pruef“ der Datei (Zählungen exakt, Gleitkomma relativ 1e-9, Zählungen sicherheitshalber ± 2: Plattform, SIMD)
Der Zufall kommt nur aus SplitMix64 (`rae_rng`); die Zählungen hängen an exp und log der Systembibliothek und gelten darum als Band (Gleichstände unter 1e-9 zählen nicht)."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rae_constants as C  # noqa: E402
import rae_evaluation as ev  # noqa: E402

PRUEF_N_KIPP = 4000
PRUEF_N_KAL = 20000
SEEDS_KAL = (C.DEFAULT_SEED, C.SEED_BAND)


def erzeugen():
    t0 = time.time()
    out = {"_hinweis": "erzeugt von tools/rae_make_frozen.py; Zufalls-Seed, Zahl der Probleme und Zustände stehen je Eintrag", "kipp": [], "kalibrierung": [], "pruef": {"kipp": [], "kalibrierung": []}}
    for art in ("sicher", "riskant"):
        for m in (2, 3, 4):
            for seed in C.SEED_KIPP:
                out["kipp"].append(ev.kipp_zaehlung(seed, C.N_KIPP, m, art))
                print(f"  Kipppunkte {art}, m = {m}, Zufalls-Seed {seed} ({time.time() - t0:.0f} s)", flush=True)
    for gamma in C.GAMMA_OPTIONS:
        for w in C.W5_OPTIONS:
            for seed in SEEDS_KAL:
                out["kalibrierung"].append(ev.kalibrierung_zelle(seed, C.N_FROZEN_CAL, gamma, w))
        print(f"  Kalibrierung γ = {gamma} ({time.time() - t0:.0f} s)", flush=True)
    for art in ("sicher", "riskant"):
        out["pruef"]["kipp"].append(ev.kipp_zaehlung(C.SEED_KIPP[0], PRUEF_N_KIPP, 2, art))
    out["pruef"]["kalibrierung"].append(ev.kalibrierung_zelle(C.DEFAULT_SEED, PRUEF_N_KAL, 2, 1000))
    pfad = ROOT / C.FROZEN_FILE
    pfad.parent.mkdir(exist_ok=True)
    pfad.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{pfad} geschrieben ({time.time() - t0:.0f} s)")


def _gleich_zaehlung(a, b):
    zahlen = [k for k, v in a.items() if isinstance(v, int) and not isinstance(v, bool) and k not in ("seed", "n", "m")]
    return all(abs(a[k] - b[k]) <= 2 for k in zahlen) and a.get("mono") == b.get("mono")


def pruefen():
    fz = ev.load_frozen()
    if fz is None:
        print("Datei fehlt")
        return 1
    n_pruef, fehler = 0, 0
    for alt in fz["pruef"]["kipp"]:
        neu = ev.kipp_zaehlung(alt["seed"], alt["n"], alt["m"], alt["art"])
        ok = _gleich_zaehlung(alt, neu)
        n_pruef += 1
        fehler += 0 if ok else 1
        print(f"Kipppunkte {alt['art']}, m = {alt['m']}, {alt['n']} Probleme: {'OK' if ok else 'ABWEICHUNG'}")
    for alt in fz["pruef"]["kalibrierung"]:
        neu = ev.kalibrierung_zelle(alt["seed"], alt["n"], alt["gamma"], alt["w"])
        ok = abs(alt["rho_k"] - neu["rho_k"]) <= 1e-9 * alt["rho_k"] and abs(alt["anteil"] - neu["anteil"]) <= 0.001 and alt["n_nd"] == neu["n_nd"]
        n_pruef += 1
        fehler += 0 if ok else 1
        print(f"Kalibrierung γ = {alt['gamma']}, w = {alt['w']}: {'OK' if ok else 'ABWEICHUNG'}")
    print(f"GESAMT: {n_pruef} Prüfungen, {fehler} Fehler")
    return 1 if fehler else 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(pruefen())
    erzeugen()

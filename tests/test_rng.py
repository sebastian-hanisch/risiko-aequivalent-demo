"""Zufallsquelle (SplitMix64): das Feld `stream_u64` ist Zahl für Zahl dieselbe Folge wie `SplitMix64.next_u64` (ganzzahlig, auf jeder Plattform gleich)."""

import numpy as np

import rae_rng as R


def test_feldstrom_gleich_sequentieller_folge():
    for seed in (0, 101, 20261116, 2 ** 64 - 1):
        g = R.SplitMix64(seed)
        folge = [g.next_u64() for _ in range(500)]
        assert [int(v) for v in R.stream_u64(seed, 500)] == folge


def test_gleichverteilte_und_below_wie_die_klasse():
    g = R.SplitMix64(7)
    sequentiell = [g.random() for _ in range(200)]
    assert np.array_equal(R.uniform_array(7, 200), np.array(sequentiell))
    g = R.SplitMix64(7)
    assert [g.below(301) for _ in range(200)] == R.below_array(7, 200, 301).tolist()


def test_below_trifft_beide_raender_und_ist_ungefaehr_gleichverteilt():
    z = R.below_array(3, 60000, 301)
    assert z.min() == 0 and z.max() == 300
    assert abs(z.mean() - 150.0) < 1.5                                                  # Standardfehler 0,35
    anzahl = np.bincount(z, minlength=301)
    assert anzahl.min() > 100 and anzahl.max() < 300                                    # Erwartung 199: grober Gleichverteilungstest


def test_erste_ausgabe_bei_seed_null_regressionswert():
    """Regressionswert: die erste Ausgabe bei Zustand 0 (aus dieser Implementierung; keine gelesene Quelle, kein Literaturwert)."""
    assert R.SplitMix64(0).next_u64() == 0xE220A8397B1DCDAF

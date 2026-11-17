"""Gemeinsame Fixtures: der Kapazitätsfall (ausgedachte Auszahlungen in GE), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def cap():
    """Kapazitätsfall: G = (+200 | −100), je ½, gegen K = +30 sicher. Erwartungswerte 50 und 30, Spanne 300, Vorsprung 20."""
    return np.array([200.0, -100.0]), np.array([0.5, 0.5]), 30.0

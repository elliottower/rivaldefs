"""The orientation rule: bands, the adjacent-cell rule at edges, worst-case qualification per
generator family, and the modal-probability threshold."""
from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from rivaldefs import (
    NOT_RESOLVED,
    RESOLVED,
    OCRecord,
    band,
    cell,
    cells_for,
    nesting_label,
    orientation_status,
    qualifying_cells,
    ratio_of,
)
from rivaldefs._direction import E0_EDGES, RATIO_EDGES


def test_bands_are_half_open_with_an_open_top():
    assert band(4.99, E0_EDGES) == 0 and band(5.0, E0_EDGES) == 1
    assert band(99.9, E0_EDGES) == 4 and band(100.0, E0_EDGES) == 5
    assert band(1.0, RATIO_EDGES) == 0 and band(1.1, RATIO_EDGES) == 1
    assert band(math.inf, RATIO_EDGES) == 3
    assert ratio_of(0, 10) == math.inf and ratio_of(30, 10) == 3.0


def test_a_value_near_an_edge_needs_both_neighbouring_cells():
    assert cells_for(30.0, 3.0) == [(3, 3)]
    assert sorted(cells_for(19.0, 3.0)) == [(2, 3), (3, 3)]              # within 10% of 20
    assert sorted(cells_for(19.0, 1.48)) == [(2, 1), (2, 2), (3, 1), (3, 2)]


def _records(cov1: float, err1: float, cov2: float, err2: float, e0: float = 30.0,
             ratio: float = 3.0) -> list[OCRecord]:
    return [OCRecord("G1", e0, ratio, cov1, err1), OCRecord("G2", e0, ratio, cov2, err2)]


def test_a_cell_qualifies_on_the_worst_scenario_in_each_family():
    recs = _records(0.95, 0.01, 0.94, 0.02) + [OCRecord("G2", 25.0, 2.5, 0.92, 0.0)]
    assert cell(30.0, 3.0) not in qualifying_cells(recs)       # worst G2 coverage is 0.92
    assert cell(30.0, 3.0) in qualifying_cells(_records(0.95, 0.01, 0.94, 0.05))
    assert cell(30.0, 3.0) not in qualifying_cells(_records(0.95, 0.06, 0.94, 0.01))


def test_a_cell_missing_a_family_does_not_qualify():
    assert qualifying_cells([OCRecord("G1", 30.0, 3.0, 0.99, 0.0)]) == set()


def test_an_undefined_population_orientation_does_not_count_as_error():
    recs = _records(0.95, 0.01, 0.95, 0.01) + [OCRecord("G1", 30.0, 3.0, 0.96, None)]
    assert cell(30.0, 3.0) in qualifying_cells(recs)


def test_resolved_needs_a_qualifying_cell_and_a_confident_mode():
    q = qualifying_cells(_records(0.95, 0.01, 0.95, 0.01))
    probs = {"a smaller": 0.96, "b smaller": 0.03, "equal": 0.01}
    assert orientation_status(30.0, 3.0, probs, q) == (RESOLVED, "a smaller")
    weak = {"a smaller": 0.94, "b smaller": 0.05, "equal": 0.01}
    assert orientation_status(30.0, 3.0, weak, q)[0] == NOT_RESOLVED
    assert orientation_status(80.0, 3.0, probs, q)[0] == NOT_RESOLVED        # cell never simulated
    assert orientation_status(20.5, 3.0, probs, q)[0] == NOT_RESOLVED        # near 20: needs (2,3)
    assert orientation_status(math.nan, 3.0, probs, q)[0] == NOT_RESOLVED


def test_nesting_labels_are_descriptive_cuts():
    assert nesting_label(0.85) == "nested" and nesting_label(0.75) == "crossing"
    assert nesting_label(0.75, cut=0.70) == "nested"
    with pytest.raises(ValueError):
        nesting_label(math.nan)

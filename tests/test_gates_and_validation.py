"""The registered gates on public results, symmetric handling of undefined within-rule terms,
the tie rule for weighted prevalences, and refusal of malformed input."""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from rivaldefs import (
    DISTINCT,
    EQUAL,
    INDEPENDENT,
    NOT_RESOLVED,
    OCRecord,
    PairTable,
    Rule,
    RulePair,
    bootstrap,
    compare_rules,
    count_bounds,
    delta_bounds,
    grid,
    orientation_status,
    pair_table,
    qualifying_cells,
    ratio_of,
    replicate_weights,
    summarize,
)
from rivaldefs._direction import E0_EDGES, RATIO_EDGES

CONFIDENT = {"a smaller": 1.0, "b smaller": 0.0, "equal": 0.0}
EVERY_CELL = {(i, j) for i in range(len(E0_EDGES) + 1) for j in range(len(RATIO_EDGES) + 1)}


# ------------------------------------------------------------- degenerate pairs never resolve

def test_a_pair_with_no_positives_or_all_positives_is_never_resolved():
    for t in (PairTable(0, 0, 40, 60), PairTable(40, 60, 0, 0), PairTable(0, 0, 0, 100)):
        assert t.e0 == 0 and math.isnan(t.h)
        status, _ = orientation_status(t.e0, ratio_of(t.x, t.y), CONFIDENT, EVERY_CELL)
        assert status == NOT_RESOLVED
    assert orientation_status(0.0, math.inf, CONFIDENT, EVERY_CELL)[0] == NOT_RESOLVED
    assert orientation_status(math.inf, 3.0, CONFIDENT, EVERY_CELL)[0] == NOT_RESOLVED


def test_a_ratio_below_one_or_undefined_is_refused():
    for bad in (math.nan, 0.5):
        with pytest.raises(ValueError):
            orientation_status(30.0, bad, CONFIDENT, EVERY_CELL)


@pytest.mark.parametrize("probs", [
    {"a smaller": 1.0, "b smaller": 0.0},
    {"a smaller": 0.5, "b smaller": 0.4, "equal": 0.0},
    {"a smaller": math.nan, "b smaller": 0.0, "equal": 1.0},
    {"a smaller": 1.2, "b smaller": -0.2, "equal": 0.0},
])
def test_malformed_orientation_probabilities_are_refused(probs):
    with pytest.raises(ValueError):
        orientation_status(30.0, 3.0, probs, EVERY_CELL)


# ------------------------------------------------------------- operating-characteristic records

@pytest.mark.parametrize("kwargs", [
    {"coverage": math.nan}, {"orientation_error": math.nan}, {"e0": math.nan}, {"e0": -1.0},
    {"ratio": math.nan}, {"ratio": 0.9}, {"coverage": 1.5}, {"orientation_error": -0.1},
    {"e0": math.inf},
])
def test_a_malformed_record_cannot_be_built(kwargs):
    base = {"family": "G1", "e0": 30.0, "ratio": 3.0, "coverage": 0.95, "orientation_error": 0.01}
    with pytest.raises(ValueError):
        OCRecord(**{**base, **kwargs})


def test_a_cell_cannot_qualify_through_a_record_that_is_not_one():
    good = OCRecord("G2", 30.0, 3.0, 0.99, 0.0)
    with pytest.raises(TypeError):
        qualifying_cells([good, {"family": "G1", "e0": 30.0, "ratio": 3.0, "coverage": math.nan}])


# ------------------------------------------------------------- gates on the public results

def _pair_labels(rng: np.random.Generator, n: int, pa: float, pb: float, rho: float) -> np.ndarray:
    z = rng.normal(size=n)
    a = z + rng.normal(scale=math.sqrt(1 / rho - 1), size=n)
    b = z + rng.normal(scale=math.sqrt(1 / rho - 1), size=n)
    return np.column_stack([a > np.quantile(a, 1 - pa), b > np.quantile(b, 1 - pb)]).astype(int)


def test_the_delta_interval_is_withheld_when_the_orientation_is_uncertain(rng):
    # Prevalences 0.300 and 0.301: every resample is close to a tie, so the modal orientation
    # probability is far below 0.95 although 0 < m < E0 holds in the full sample.
    x = _pair_labels(rng, 2000, 0.300, 0.301, 0.5)
    res = bootstrap(x, pairs=[(0, 1)], n_boot=400, rng=rng).pairs[0]
    assert 0 < res.table.m < res.table.e0
    assert max(res.orientation_probabilities.values()) < 0.95
    assert res.h_delta_interval_ungated is not None
    assert res.h_delta_interval is None


def test_the_delta_interval_is_given_when_the_orientation_is_certain(rng):
    x = _pair_labels(rng, 2000, 0.10, 0.40, 0.5)
    res = bootstrap(x, pairs=[(0, 1)], n_boot=400, rng=rng).pairs[0]
    assert 0 < res.table.m < res.table.e0
    assert max(res.orientation_probabilities.values()) >= 0.95
    assert res.h_delta_interval is not None
    assert res.h_delta_interval == res.h_delta_interval_ungated


def test_r_is_withheld_entirely_below_the_denominator_threshold(rng):
    z = rng.normal(size=500)
    x = np.column_stack([z + rng.normal(scale=0.5, size=500) > t for t in (0.3, 0.5)]
                        + [-z + rng.normal(size=500) > 0.4]).astype(int)
    varied = Rule("A", (0, 1), ((0,), (1,)))
    fixed = Rule("B", (2,))
    same = Rule("C", (0, 0), ((0,), (1,)))          # identical variants: no within spread
    res = bootstrap(x, rule_pairs=[RulePair(varied, fixed), RulePair(same, fixed)], n_boot=200,
                    rng=rng)
    shown, hidden = res.rule_pairs
    assert shown.r_reportable and shown.r.reported and shown.r == shown.r_ungated
    assert not hidden.r_reportable
    assert not hidden.r.reported
    assert math.isnan(hidden.r.estimate) and math.isnan(hidden.r.lo) and math.isnan(hidden.r.hi)


# ------------------------------------------------------------- undefined within terms are symmetric

def test_an_undefined_conditional_within_term_makes_delta_undefined_on_either_side():
    x = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 0]])
    g = grid(x)
    two = Rule("A", (0, 1), ((0,), (1,)))
    one = Rule("B", (2,))
    for first, second in ((two, one), (one, two)):
        c = compare_rules(g, first, second, within=DISTINCT)
        assert math.isnan(c.denominator) and math.isnan(c.delta) and math.isnan(c.r)


def test_swapping_the_rules_leaves_delta_and_r_unchanged(rng):
    z = rng.normal(size=300)
    x = np.column_stack([z + rng.normal(scale=s, size=300) > t
                         for s, t in ((0.3, 0.4), (0.6, 0.5), (0.4, 0.8), (1.0, 0.1), (0.8, 0.2))]
                        ).astype(int)
    a = Rule("A", (0, 1, 2), ((0,), (1,), (2,)))
    b = Rule("B", (3, 4), ((0,), (1,)))
    g = grid(x)
    for law in (INDEPENDENT, DISTINCT):
        ab, ba = compare_rules(g, a, b, within=law), compare_rules(g, b, a, within=law)
        assert ab.delta == pytest.approx(ba.delta, abs=1e-12)
        assert ab.r == pytest.approx(ba.r, abs=1e-12)
        assert ab.d_between == pytest.approx(ba.d_between, abs=1e-12)


def test_delta_bounds_are_symmetric_in_the_rules_too():
    units = [np.array([[1, 1, 1]]), np.array([[1, 0, 0], [1, 0, 1]]), np.array([[0, 1, 1]]),
             np.array([[0, 0, 0]])]
    two = Rule("A", (0, 1), ((0,), (1,)))
    one = Rule("B", (2,))
    for first, second in ((two, one), (one, two)):
        bd = delta_bounds(units, first, second, within=DISTINCT)
        assert math.isnan(bd.delta[0]) and math.isnan(bd.delta[1])


# ------------------------------------------------------------- ties of weighted prevalences

def test_weighted_prevalences_equal_up_to_rounding_are_a_tie():
    a = np.array([1, 1, 0])
    b = np.array([0, 0, 1])
    w = np.array([0.1, 0.2, 0.3])
    t = pair_table(a, b, w)
    assert t.x != t.y                     # 0.1 + 0.2 is not 0.3 in floating point
    assert t.orientation == EQUAL
    assert t.m == pytest.approx(0.3) and t.nodf == 0.0
    g = grid(np.column_stack([a, b]), w)
    assert g.orientation[0, 1] == EQUAL and g.orientation[1, 0] == EQUAL
    assert g.m[0, 1] == pytest.approx(t.m) and g.nodf[0, 1] == 0.0
    assert g.h[0, 1] == pytest.approx(t.h)


def test_integer_counts_one_apart_are_never_a_tie_even_when_n_is_large():
    t = PairTable(100_000, 1, 0, 1_900_000)
    assert t.orientation == "b smaller" and t.m == 0.0


# ------------------------------------------------------------- malformed input

def test_rule_columns_must_be_non_negative_and_in_range(rng):
    with pytest.raises(ValueError):
        Rule("A", (-1, 0))
    x = rng.integers(0, 2, size=(30, 3))
    with pytest.raises(ValueError):
        compare_rules(grid(x), Rule("A", (0, 3)), Rule("B", (1,)))
    with pytest.raises(ValueError):
        bootstrap(x, rule_pairs=[RulePair(Rule("A", (0, 5)), Rule("B", (1,)))], n_boot=5, rng=rng)
    units = [row[None, :] for row in x]
    with pytest.raises(ValueError):
        delta_bounds(units, Rule("A", (0, 7)), Rule("B", (1,)))
    for a, b in ((-1, 0), (0, 3)):
        with pytest.raises(ValueError):
            count_bounds(units, a, b)


def test_a_replicate_column_summing_to_zero_is_refused(rng):
    x = rng.integers(0, 2, size=(20, 2))
    rep = rng.uniform(0.5, 2.0, size=(20, 4))
    rep[:, 2] = 0.0
    with pytest.raises(ValueError, match="sum to zero"):
        replicate_weights(x, np.ones(20), rep, pairs=[(0, 1)])


@pytest.mark.parametrize("n_boot", [0, -3, 2.5, True])
def test_n_boot_must_be_a_positive_integer(rng, n_boot):
    with pytest.raises(ValueError):
        bootstrap(rng.integers(0, 2, size=(20, 2)), pairs=[(0, 1)], n_boot=n_boot, rng=rng)


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5, -0.2])
def test_level_must_lie_strictly_between_zero_and_one(rng, level):
    with pytest.raises(ValueError):
        summarize(0.1, np.array([0.1, 0.2]), level=level)
    with pytest.raises(ValueError):
        bootstrap(rng.integers(0, 2, size=(20, 2)), pairs=[(0, 1)], n_boot=5, rng=rng,
                  level=level)


@pytest.mark.parametrize("min_defined", [-0.1, 1.1])
def test_min_defined_must_lie_in_the_unit_interval(min_defined):
    with pytest.raises(ValueError):
        summarize(0.1, np.array([0.1, 0.2]), min_defined=min_defined)

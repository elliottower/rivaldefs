"""Between- versus within-rule disagreement: the implementation distributions, the exact
D sums on a hand-worked case, and the invariances the registration commits as software tests."""
from __future__ import annotations

import itertools
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from rivaldefs import (
    DISTINCT,
    INDEPENDENT,
    Rule,
    compare_rules,
    d_within,
    factor_balanced,
    grid,
    reference_weighted,
)
from rivaldefs._rules import _weighted_quantiles


def test_factor_balanced_is_uniform_on_a_full_factorial():
    levels = tuple(itertools.product((0, 1, 2), ("a", "b"), (True, False)))
    rule = Rule("r", tuple(range(len(levels))), levels)
    w = factor_balanced(rule)
    assert np.allclose(w, 1 / 12) and w.sum() == pytest.approx(1)


def test_factor_balanced_splits_a_duplicated_combination():
    levels = ((0, 0), (0, 1), (1, 0), (1, 1), (1, 1))
    w = factor_balanced(Rule("r", tuple(range(5)), levels))
    assert list(w) == pytest.approx([0.25, 0.25, 0.25, 0.125, 0.125])


def test_factor_balanced_refuses_an_incomplete_grid():
    with pytest.raises(ValueError, match="full factorial"):
        factor_balanced(Rule("r", (0, 1, 2), ((0, 0), (0, 1), (1, 0))))


def test_reference_weighted_gives_half_to_the_reference():
    w = reference_weighted(Rule("r", (0, 1, 2, 3), reference=2))
    assert list(w) == pytest.approx([1 / 6, 1 / 6, 0.5, 1 / 6])


def test_d_sums_on_a_hand_worked_case():
    # Four units; rule A has variants a0, a1; rule B has one variant b0.
    x = np.array([
        # a0 a1 b0
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ])
    g = grid(x)
    a = Rule("A", (0, 1), ((0,), (1,)))
    b = Rule("B", (2,))
    c = compare_rules(g, a, b)
    # d(a0,b0) = 2/4, d(a1,b0) = 0; D_between = (0.5 + 0) / 2 = 0.25 -> 25 pp.
    # d(a0,a1) = 2/4; D_within(A) = 2 * 0.25 * 0.5 = 0.25 -> 25 pp; D_within(B) = 0.
    assert c.d_between == pytest.approx(25.0)
    assert c.d_within_a == pytest.approx(25.0)
    assert c.d_within_b == 0.0
    assert c.delta == pytest.approx(0.0)
    assert c.r == pytest.approx(1.0)
    # Conditional on distinct draws, D_within(A) = d(a0,a1) = 50 pp; B has one implementation,
    # so its conditional term is undefined and so are the denominator, Delta and R.
    cd = compare_rules(g, a, b, within=DISTINCT)
    assert cd.d_within_a == pytest.approx(50.0)
    assert math.isnan(cd.d_within_b)
    assert math.isnan(cd.denominator) and math.isnan(cd.delta) and math.isnan(cd.r)


def _random_rules(rng: np.random.Generator, n: int = 300):
    theta = rng.normal(size=n)
    cols = []
    levels_a = list(itertools.product((0, 1), (0, 1)))
    for s, t in levels_a:
        cols.append(theta + 0.3 * s + rng.normal(scale=0.2 + 0.3 * t, size=n) > 0.5)
    levels_b = list(itertools.product((0, 1, 2),))
    for (u,) in levels_b:
        cols.append(-0.5 * theta + rng.normal(scale=1.0, size=n) > 0.2 * u)
    x = np.column_stack(cols).astype(int)
    a = Rule("A", (0, 1, 2, 3), tuple(levels_a), reference=0)
    b = Rule("B", (4, 5, 6), tuple(levels_b), reference=0)
    return x, a, b


def test_delta_r_and_s_are_unchanged_when_a_variant_is_duplicated(rng: np.random.Generator):
    for _ in range(30):
        x, a, b = _random_rules(rng)
        base = compare_rules(grid(x), a, b)
        base_d = compare_rules(grid(x), a, b, within=DISTINCT)
        j = int(rng.integers(len(a.columns)))
        x2 = np.column_stack([x, x[:, a.columns[j]]])
        a2 = Rule("A", a.columns + (x.shape[1],), a.levels + (a.levels[j],), reference=a.reference)
        g2 = grid(x2)
        dup = compare_rules(g2, a2, b)
        dup_d = compare_rules(g2, a2, b, within=DISTINCT)
        for f in ("d_between", "d_within_a", "d_within_b", "delta", "r", "s"):
            assert getattr(dup, f) == pytest.approx(getattr(base, f), abs=1e-10), f
            assert getattr(dup_d, f) == pytest.approx(getattr(base_d, f), abs=1e-10), f


def test_s_is_the_weight_of_variant_pairs_sharing_the_reference_orientation(rng: np.random.Generator):
    x, a, b = _random_rules(rng)
    g = grid(x)
    c = compare_rules(g, a, b)
    wa, wb = factor_balanced(a), factor_balanced(b)
    ref = g.orientation[a.columns[a.reference], b.columns[b.reference]]
    want = sum(wa[i] * wb[j] for i, ca in enumerate(a.columns) for j, cb in enumerate(b.columns)
               if g.orientation[ca, cb] == ref)
    assert c.s == pytest.approx(want)
    assert sum(c.orientation_probabilities.values()) == pytest.approx(1.0)


def test_s_is_undefined_when_the_reference_pair_has_equal_prevalence():
    x = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    c = compare_rules(grid(x), Rule("A", (0,)), Rule("B", (1,)))
    assert math.isnan(c.s)
    assert c.orientation_probabilities["equal"] == 1.0


def test_r_is_undefined_when_no_rule_varies():
    x = np.array([[1, 0], [0, 1], [0, 0], [1, 0]])
    c = compare_rules(grid(x), Rule("A", (0,)), Rule("B", (1,)))
    assert math.isnan(c.r) and c.delta == pytest.approx(c.d_between)


def test_d_within_rejects_an_unknown_law():
    with pytest.raises(ValueError):
        d_within(np.zeros((2, 2)), np.array([0.5, 0.5]), "sometimes")
    assert d_within(np.array([[0, 1.0], [1.0, 0]]), np.array([0.5, 0.5]), INDEPENDENT) == 0.5


def test_h_quantiles_skip_undefined_values_and_renormalize_the_weights():
    # Defined weights 0.1, 0.3, 0.2 renormalize to 1/6, 1/2, 1/3: cumulative 1/6, 2/3, 1.
    got = _weighted_quantiles(np.array([math.nan, 0.2, 0.5, 0.9]),
                              np.array([0.4, 0.1, 0.3, 0.2]), (0.10, 0.50, 0.90))
    assert got == [0.2, 0.5, 0.9]
    got = _weighted_quantiles(np.array([0.9, 0.2, 0.5]), np.array([0.2, 0.1, 0.3]),
                              (1 / 6, 0.17))
    assert got == [0.2, 0.5]                          # inverted CDF: the step is at 1/6
    assert _weighted_quantiles(np.array([0.1, 0.8]), np.array([0.0, 1.0]), (0.1,)) == [0.8]
    assert all(math.isnan(v) for v in
               _weighted_quantiles(np.array([math.nan, math.nan]), np.array([0.5, 0.5]), (0.5,)))

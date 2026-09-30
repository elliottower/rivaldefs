"""Resampling: the pattern-count bootstrap is the participant bootstrap, the undefined-replicate
rule, and survey replicate weights."""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from rivaldefs import (
    Rule,
    RulePair,
    bootstrap,
    compare_rules,
    grid,
    pair_table,
    replicate_weights,
    summarize,
)


def _labels(n: int = 400) -> np.ndarray:
    theta = np.random.normal(size=n)
    return np.column_stack([
        theta > 1.0,
        theta + np.random.normal(scale=0.5, size=n) > 0.3,
        theta + np.random.normal(scale=0.5, size=n) > 0.5,
        -theta + np.random.normal(size=n) > 0.4,
    ]).astype(int)


def test_pattern_count_bootstrap_matches_unit_resampling():
    # Both draw N units with replacement from the observed ones, so the package's interval for H
    # must match the percentile interval of an explicit unit-level bootstrap, within Monte Carlo
    # error, and the two replicate distributions must share mean and spread.
    x = _labels(300)
    a, b = 1, 3                     # a crossing pair, so H varies across replicates
    reps = 4000
    res = bootstrap(x, pairs=[(a, b)], n_boot=reps)
    n = x.shape[0]
    h_unit = np.array([pair_table(x[idx, a], x[idx, b]).h
                       for idx in (np.random.randint(0, n, n) for _ in range(reps))])
    patterns, counts = np.unique(x, axis=0, return_counts=True)
    h_pat = []
    for c in np.random.multinomial(n, counts / n, size=reps):
        rows = np.repeat(patterns, c, axis=0)
        h_pat.append(pair_table(rows[:, a], rows[:, b]).h)
    h_pat_arr = np.array(h_pat)
    assert h_unit.std() > 0.01
    se = math.sqrt(h_unit.var() / reps + h_pat_arr.var() / reps)
    assert abs(h_unit.mean() - h_pat_arr.mean()) < 5 * se
    assert h_pat_arr.std() == pytest.approx(h_unit.std(), rel=0.08)
    lo, hi = np.quantile(h_unit, [0.025, 0.975])
    width = hi - lo
    assert res.pairs[0].h.lo == pytest.approx(lo, abs=0.1 * width)
    assert res.pairs[0].h.hi == pytest.approx(hi, abs=0.1 * width)


def test_point_estimates_are_the_full_sample_values():
    x = _labels()
    a = Rule("A", (1, 2), ((0,), (1,)))
    b = Rule("B", (3,))
    res = bootstrap(x, pairs=[(0, 1)], rule_pairs=[RulePair(a, b)], n_boot=50)
    assert res.pairs[0].h.estimate == pytest.approx(pair_table(x[:, 0], x[:, 1]).h)
    assert res.rule_pairs[0].delta.estimate == pytest.approx(compare_rules(grid(x), a, b).delta)
    probs = res.pairs[0].orientation_probabilities
    assert sum(probs.values()) == pytest.approx(1.0)


def test_an_interval_needs_ninety_percent_of_replicates_defined():
    reps = np.array([0.1] * 89 + [math.nan] * 11)
    s = summarize(0.1, reps)
    assert not s.reported and s.fraction_undefined == pytest.approx(0.11)
    reps = np.array([0.1] * 90 + [math.nan] * 10)
    s = summarize(0.1, reps)
    assert s.reported and s.fraction_undefined == pytest.approx(0.10)
    assert s.lo == s.hi == pytest.approx(0.1)


def test_a_rare_labeler_yields_undefined_replicates_and_reports_them():
    n = 200
    x = np.zeros((n, 2), dtype=int)
    x[0, 0] = 1                       # one positive: most resamples lose it and H is undefined
    x[:100, 1] = 1
    res = bootstrap(x, pairs=[(0, 1)], n_boot=500)
    h = res.pairs[0].h
    assert h.fraction_undefined > 0.2 and not h.reported


def test_replicate_weights_equal_to_the_full_weights_reproduce_the_point():
    x = _labels(100)
    w = np.random.uniform(0.5, 3.0, size=100)
    rep = np.column_stack([w] * 20)
    res = replicate_weights(x, w, rep, pairs=[(0, 1)])
    t = pair_table(x[:, 0], x[:, 1], w)
    assert res.pairs[0].h.estimate == pytest.approx(t.h)
    assert res.pairs[0].h.lo == pytest.approx(t.h) and res.pairs[0].h.hi == pytest.approx(t.h)
    assert res.method == "survey replicate weights"


def test_r_is_reported_only_above_half_a_point_of_within_disagreement():
    x = _labels(500)
    varied = Rule("A", (1, 2), ((0,), (1,)))
    fixed = Rule("B", (3,))
    same = Rule("C", (1, 1), ((0,), (1,)))          # two identical variants: no within spread
    res = bootstrap(x, rule_pairs=[RulePair(varied, fixed), RulePair(same, fixed)], n_boot=200)
    assert res.rule_pairs[0].denominator_p025 > 0.5 and res.rule_pairs[0].r_reportable
    assert not res.rule_pairs[1].r_reportable


def test_a_pair_must_name_two_labelers():
    with pytest.raises(ValueError):
        bootstrap(_labels(), pairs=[(1, 1)], n_boot=5)

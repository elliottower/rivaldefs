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


def _labels(rng: np.random.Generator, n: int = 400) -> np.ndarray:
    theta = rng.normal(size=n)
    return np.column_stack([
        theta > 1.0,
        theta + rng.normal(scale=0.5, size=n) > 0.3,
        theta + rng.normal(scale=0.5, size=n) > 0.5,
        -theta + rng.normal(size=n) > 0.4,
    ]).astype(int)


def test_pattern_count_bootstrap_matches_unit_resampling(rng: np.random.Generator):
    # Both draw N units with replacement from the observed ones, so the package's interval for H
    # must match the percentile interval of an explicit unit-level bootstrap, within Monte Carlo
    # error, and the two replicate distributions must share mean and spread.
    x = _labels(rng, 300)
    a, b = 1, 3                     # a crossing pair, so H varies across replicates
    reps = 4000
    res = bootstrap(x, pairs=[(a, b)], n_boot=reps, rng=rng)
    n = x.shape[0]
    h_unit = np.array([pair_table(x[idx, a], x[idx, b]).h
                       for idx in (rng.integers(0, n, n) for _ in range(reps))])
    patterns, counts = np.unique(x, axis=0, return_counts=True)
    h_pat = []
    for c in rng.multinomial(n, counts / n, size=reps):
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


def test_point_estimates_are_the_full_sample_values(rng: np.random.Generator):
    x = _labels(rng)
    a = Rule("A", (1, 2), ((0,), (1,)))
    b = Rule("B", (3,))
    res = bootstrap(x, pairs=[(0, 1)], rule_pairs=[RulePair(a, b)], n_boot=50, rng=rng)
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


def test_a_rare_labeler_yields_undefined_replicates_and_reports_them(rng: np.random.Generator):
    n = 200
    x = np.zeros((n, 2), dtype=int)
    x[0, 0] = 1                       # one positive: most resamples lose it and H is undefined
    x[:100, 1] = 1
    res = bootstrap(x, pairs=[(0, 1)], n_boot=500, rng=rng)
    h = res.pairs[0].h
    assert h.fraction_undefined > 0.2 and not h.reported


def test_replicate_weights_equal_to_the_full_weights_reproduce_the_point(rng: np.random.Generator):
    x = _labels(rng, 100)
    w = rng.uniform(0.5, 3.0, size=100)
    rep = np.column_stack([w] * 20)
    res = replicate_weights(x, w, rep, pairs=[(0, 1)])
    t = pair_table(x[:, 0], x[:, 1], w)
    assert res.pairs[0].h.estimate == pytest.approx(t.h)
    assert res.pairs[0].h.lo == pytest.approx(t.h) and res.pairs[0].h.hi == pytest.approx(t.h)
    assert res.method == "survey replicate weights"


def test_r_is_reported_only_above_half_a_point_of_within_disagreement(rng: np.random.Generator):
    x = _labels(rng, 500)
    varied = Rule("A", (1, 2), ((0,), (1,)))
    fixed = Rule("B", (3,))
    same = Rule("C", (1, 1), ((0,), (1,)))          # two identical variants: no within spread
    res = bootstrap(x, rule_pairs=[RulePair(varied, fixed), RulePair(same, fixed)], n_boot=200, rng=rng)
    assert res.rule_pairs[0].denominator_p025 > 0.5 and res.rule_pairs[0].r_reportable
    assert not res.rule_pairs[1].r_reportable


def test_a_pair_must_name_two_labelers(rng: np.random.Generator):
    with pytest.raises(ValueError):
        bootstrap(_labels(rng), pairs=[(1, 1)], n_boot=5)


def test_delta_r_and_s_intervals_match_explicit_unit_resampling(rng: np.random.Generator):
    # The pattern-count bootstrap and an explicit unit-level bootstrap draw from the same
    # distribution, so the registered rule outputs must agree within Monte Carlo error.
    n, reps = 300, 2000
    z = rng.normal(size=n)
    cols = [z + rng.normal(scale=s, size=n) > t
            for s, t in ((0.3, 0.3), (0.6, 0.5), (0.4, 0.8), (0.9, 0.2))]
    cols += [-0.5 * z + rng.normal(size=n) > t for t in (0.0, 0.4, 0.8)]
    x = np.column_stack(cols).astype(int)
    a = Rule("A", (0, 1, 2, 3), ((0, 0), (0, 1), (1, 0), (1, 1)))
    b = Rule("B", (4, 5, 6), ((0,), (1,), (2,)))
    res = bootstrap(x, rule_pairs=[RulePair(a, b)], n_boot=reps, rng=rng).rule_pairs[0]
    unit = {"delta": [], "r": [], "s": []}
    for _ in range(reps):
        c = compare_rules(grid(x[rng.integers(0, n, n)]), a, b)
        unit["delta"].append(c.delta)
        unit["r"].append(c.r)
        unit["s"].append(c.s)
    for name, interval in (("delta", res.delta), ("r", res.r_ungated)):
        u = np.array(unit[name], dtype=float)
        u = u[~np.isnan(u)]
        lo, hi = np.quantile(u, [0.025, 0.975])
        width = max(hi - lo, 1e-9)
        assert interval.reported, name
        assert interval.lo == pytest.approx(lo, abs=0.12 * width), name
        assert interval.hi == pytest.approx(hi, abs=0.12 * width), name
    # S moves in steps of one variant pair's weight, so its percentiles sit on a lattice and
    # their endpoints are compared through the distribution instead: the share of unit-level
    # replicates at or below the package's lower limit, and strictly below its upper limit.
    s_unit = np.array([v for v in unit["s"] if not math.isnan(v)])
    assert res.s.reported
    assert np.mean(s_unit <= res.s.lo) >= 0.025 - 0.02
    assert np.mean(s_unit < res.s.hi) <= 0.975 + 0.02


def test_survey_replicate_intervals_are_the_percentiles_of_the_replicate_estimates():
    # Twelve units, two labelers, five genuinely different replicate columns; the expected
    # interval is computed here from pair_table on each column, without the package's loop.
    x = np.array([[1, 1], [1, 0], [0, 1], [0, 0], [1, 1], [0, 1],
                  [0, 0], [1, 1], [0, 1], [0, 0], [1, 0], [0, 1]])
    w = np.array([1.0, 2.0, 1.5, 3.0, 0.5, 1.0, 2.0, 1.0, 1.5, 2.5, 1.0, 1.0])
    rep = np.array([
        [2.0, 0.0, 1.0, 1.0, 2.0],
        [0.0, 4.0, 2.0, 2.0, 2.0],
        [3.0, 1.5, 0.0, 1.5, 1.5],
        [3.0, 3.0, 6.0, 0.0, 3.0],
        [1.0, 0.5, 0.5, 0.5, 0.0],
        [0.0, 1.0, 2.0, 1.0, 1.0],
        [2.0, 2.0, 0.0, 4.0, 2.0],
        [1.0, 2.0, 1.0, 1.0, 1.0],
        [1.5, 0.0, 3.0, 1.5, 1.5],
        [5.0, 2.5, 2.5, 0.0, 2.5],
        [1.0, 1.0, 1.0, 2.0, 0.0],
        [0.0, 1.0, 2.0, 1.0, 1.0],
    ])
    res = replicate_weights(x, w, rep, pairs=[(0, 1)]).pairs[0]
    hs = np.array([pair_table(x[:, 0], x[:, 1], rep[:, j]).h for j in range(rep.shape[1])])
    cs = np.array([pair_table(x[:, 0], x[:, 1], rep[:, j]).c for j in range(rep.shape[1])])
    assert len(set(np.round(hs, 12))) == 5            # the columns really differ
    assert res.h.estimate == pytest.approx(pair_table(x[:, 0], x[:, 1], w).h)
    assert (res.h.lo, res.h.hi) == pytest.approx(tuple(np.quantile(hs, [0.025, 0.975])))
    assert (res.c.lo, res.c.hi) == pytest.approx(tuple(np.quantile(cs, [0.025, 0.975])))

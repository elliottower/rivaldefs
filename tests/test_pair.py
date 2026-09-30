"""One pair of labelings: exact values on hand-worked tables, and the identities H must satisfy.

The worked pair is the registration's committed Indices check. H is checked against an
independent brute-force count of Loevinger's Guttman errors, written here without importing
anything from the package.
"""
from __future__ import annotations

import math
import pathlib
import random
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from rivaldefs import (
    A_SMALLER,
    B_SMALLER,
    EQUAL,
    PairTable,
    h_delta_interval,
    pair_table,
)

# ------------------------------------------------------------- worked pair
# N = 1,000. (x, y, n11) = (100, 300, 95): n10 = 5, n01 = 205, n00 = 695.
# kappa = 2(95*695 - 5*205) / (100*700 + 300*900) = 130000/340000 = 13/34.
# E0 = 100*700/1000 = 70, m = 5, H = 1 - 5/70 = 13/14, C = 95/100.
# (x, y, n11) = (200, 200, 101): n10 = n01 = 99, n00 = 701.
# kappa = 2(101*701 - 99*99) / (2*200*800) = 122000/320000 = 61/160.
# E0 = 200*800/1000 = 160, m = 99, H = 61/160 = kappa, C = 101/200.

def test_the_worked_pair_has_equal_kappa_and_different_h():
    nested = PairTable(95, 5, 205, 695)
    even = PairTable(101, 99, 99, 701)
    assert nested.kappa == pytest.approx(13 / 34)
    assert even.kappa == pytest.approx(61 / 160)
    assert round(nested.kappa, 2) == round(even.kappa, 2) == 0.38
    assert nested.h == pytest.approx(13 / 14)
    assert even.h == pytest.approx(61 / 160)
    assert nested.c == pytest.approx(0.95)
    assert even.c == pytest.approx(101 / 200)
    assert nested.e0 == pytest.approx(70) and nested.m == 5
    assert even.e0 == pytest.approx(160) and even.m == 99


def test_at_equal_prevalence_the_orientation_is_equal_and_h_is_kappa():
    t = PairTable(101, 99, 99, 701)
    assert t.orientation == EQUAL
    assert t.n10 == t.n01 == t.m
    assert t.h == pytest.approx(t.kappa)
    assert t.nodf == 0.0


def test_orientation_follows_the_smaller_positive_count():
    assert PairTable(10, 0, 30, 60).orientation == A_SMALLER
    assert PairTable(10, 30, 0, 60).orientation == B_SMALLER
    assert PairTable(10, 30, 0, 60).m == 0


# ------------------------------------------------------------- boundaries

def test_a_labeler_against_itself_has_h_one_and_no_discordance():
    for _ in range(200):
        n = random.randint(2, 500)
        a = np.array([random.random() < random.random() for _ in range(n)], dtype=int)
        if 0 < a.sum() < n:
            t = pair_table(a, a)
            assert t.h == 1.0 and t.d == 0.0 and t.c == 1.0


def test_forced_nesting_empties_the_prohibited_cell_and_gives_h_one():
    for _ in range(200):
        n = random.randint(3, 400)
        score = [random.random() for _ in range(n)]
        lo, hi = sorted(random.random() for _ in range(2))
        strict = np.array([s > hi for s in score], dtype=int)
        loose = np.array([s > lo for s in score], dtype=int)
        t = pair_table(strict, loose)
        assert t.n10 == 0
        if 0 < t.x and t.y < n and t.x < t.y:
            assert t.h == 1.0
            assert t.c == 1.0


def test_h_is_undefined_when_a_prevalence_is_zero_or_one():
    assert math.isnan(PairTable(0, 0, 40, 60).h)          # a has no positives
    assert math.isnan(PairTable(40, 60, 0, 0).h)          # b labels everyone
    assert math.isnan(PairTable(0, 0, 40, 60).c)
    assert PairTable(0, 0, 40, 60).n10 == 0               # the prohibited cell is still empty


def test_kappa_is_undefined_when_both_labelings_are_constant():
    assert math.isnan(PairTable(0, 0, 0, 10).kappa)


# ------------------------------------------------------------- identities

def _random_table(n_max: int = 300) -> PairTable:
    n = random.randint(4, n_max)
    cut = sorted(random.randint(0, n) for _ in range(3))
    cells = [cut[0], cut[1] - cut[0], cut[2] - cut[1], n - cut[2]]
    random.shuffle(cells)
    return PairTable(*map(float, cells))


def test_h_equals_kappa_over_kappa_max_on_random_tables():
    checked = 0
    for _ in range(5000):
        t = _random_table()
        if math.isnan(t.h) or math.isnan(t.kappa_max) or t.kappa_max == 0:
            continue
        assert t.h == pytest.approx(t.kappa / t.kappa_max, abs=1e-12)
        checked += 1
    assert checked > 3000


def _loevinger_brute_force(a: list[int], b: list[int]) -> Fraction | None:
    """Guttman errors counted unit by unit, against their independence expectation."""
    n = len(a)
    pa, pb = Fraction(sum(a), n), Fraction(sum(b), n)
    hard, easy = (a, b) if pa <= pb else (b, a)
    p_hard, p_easy = min(pa, pb), max(pa, pb)
    errors = sum(1 for h, e in zip(hard, easy) if h == 1 and e == 0)
    expected = n * p_hard * (1 - p_easy)
    if expected == 0:
        return None
    return 1 - Fraction(errors) / expected


def test_h_equals_a_brute_force_loevinger_count():
    for _ in range(500):
        n = random.randint(2, 120)
        pa, pb = random.random(), random.random()
        a = [int(random.random() < pa) for _ in range(n)]
        b = [int(random.random() < pb) for _ in range(n)]
        ref = _loevinger_brute_force(a, b)
        got = pair_table(np.array(a), np.array(b)).h
        if ref is None:
            assert math.isnan(got)
        else:
            assert got == pytest.approx(float(ref), abs=1e-12)


def test_permuting_one_labeler_centres_h_on_zero():
    # Under a random permutation with both margins fixed, m is hypergeometric with mean
    # x(N - y)/N = E0, so E[H] = 0 exactly; the mean over many permutations must sit near it.
    a = np.array([1] * 60 + [0] * 340)
    b = np.array([1] * 150 + [0] * 250)
    hs = []
    for _ in range(20000):
        hs.append(pair_table(a, np.random.permutation(b)).h)
    hs_arr = np.array(hs)
    se = hs_arr.std() / math.sqrt(len(hs_arr))
    assert abs(hs_arr.mean()) < 5 * se


def test_weights_act_as_counts():
    a = np.array([1, 1, 0, 0, 1])
    b = np.array([1, 0, 1, 0, 1])
    w = np.array([2.0, 1.0, 3.0, 4.0, 0.5])
    t = pair_table(a, b, w)
    assert (t.n11, t.n10, t.n01, t.n00) == (2.5, 1.0, 3.0, 4.0)
    rep = np.repeat(np.arange(5), [4, 2, 6, 8, 1])     # integer weights doubled
    u = pair_table(a[rep], b[rep])
    assert u.h == pytest.approx(pair_table(a, b, w * 2).h)


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        pair_table([0, 1, 2], [0, 1, 1])
    with pytest.raises(ValueError):
        pair_table([0, 1], [0, 1, 1])
    with pytest.raises(ValueError):
        PairTable(-1, 0, 0, 1)


def test_the_delta_interval_matches_the_bootstrap_spread():
    # The delta-method SD of H and the SD of H over multinomial resamples of the four cells
    # must agree to within the Monte Carlo error of the latter, on a large table.
    t = PairTable(300, 40, 500, 2160)
    lo, hi = h_delta_interval(t)
    sd_delta = (hi - lo) / (2 * 1.959963984540054)
    p = np.array([t.n11, t.n10, t.n01, t.n00]) / t.n
    hs = []
    for c in np.random.multinomial(int(t.n), p, size=20000):
        hs.append(PairTable(*map(float, c)).h)
    sd_boot = float(np.std(hs))
    assert sd_delta == pytest.approx(sd_boot, rel=0.05)


def test_no_delta_interval_at_equal_prevalence_or_at_the_boundary():
    assert h_delta_interval(PairTable(101, 99, 99, 701)) is None
    assert h_delta_interval(PairTable(95, 0, 205, 700)) is None     # m = 0

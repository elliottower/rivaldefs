"""Many labelers: the grid agrees with the pairwise table, scale H with a brute-force pooled
count, and the ladder with its definition."""
from __future__ import annotations

import math
import pathlib
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from rivaldefs import grid, ladder, pair_table, scale_h


def _random_labels(n: int, k: int) -> np.ndarray:
    theta = np.random.normal(size=n)
    cuts = np.random.normal(size=k)
    noise = np.random.normal(scale=np.random.uniform(0.1, 2.0), size=(n, k))
    return ((theta[:, None] + noise) > cuts[None, :]).astype(int)


def test_every_grid_cell_equals_the_pair_table():
    for _ in range(30):
        x = _random_labels(random.randint(20, 300), random.randint(2, 6))
        g = grid(x)
        for a in range(x.shape[1]):
            for b in range(x.shape[1]):
                if a == b:
                    continue
                t = pair_table(x[:, a], x[:, b])
                for name in ("h", "c", "kappa", "d", "nodf", "m", "e0"):
                    got, want = getattr(g, name)[a, b], getattr(t, name)
                    if math.isnan(want):
                        assert math.isnan(got), name
                    else:
                        assert got == pytest.approx(want, abs=1e-12), name
                assert g.orientation[a, b] == t.orientation


def test_scale_h_equals_a_brute_force_pooled_count():
    for _ in range(40):
        x = _random_labels(random.randint(10, 200), random.randint(2, 5))
        n, k = x.shape
        errors = expected = 0.0
        for a in range(k):
            for b in range(a + 1, k):
                pa, pb = x[:, a].mean(), x[:, b].mean()
                hard, easy = (a, b) if pa <= pb else (b, a)
                errors += float(np.sum((x[:, hard] == 1) & (x[:, easy] == 0)))
                expected += n * min(pa, pb) * (1 - max(pa, pb))
        want = math.nan if expected == 0 else 1 - errors / expected
        got = scale_h(x)
        assert (math.isnan(got) and math.isnan(want)) or got == pytest.approx(want, abs=1e-10)


def test_a_perfect_ladder_is_all_rungs_with_ladder_h_one():
    score = np.random.uniform(size=1000)
    x = np.column_stack([(score > c).astype(int) for c in (0.8, 0.2, 0.5, 0.65)])
    lad = ladder(x)
    assert lad.order == (1, 2, 3, 0)          # loosest first
    assert lad.on_ladder == 1.0
    assert lad.h == pytest.approx(1.0)
    assert scale_h(x) == pytest.approx(1.0)


def test_the_independence_value_of_the_ladder_matches_its_formula():
    x = _random_labels(500, 3)
    lad = ladder(x)
    pi = lad.prevalence
    l0 = sum(math.prod(pi[:j]) * math.prod(1 - p for p in pi[j:]) for j in range(4))
    assert lad.on_ladder_independent == pytest.approx(l0)


def test_permuting_every_labeler_sends_ladder_h_toward_zero():
    base = _random_labels(4000, 4)
    hs = []
    for _ in range(300):
        perm = np.column_stack([np.random.permutation(base[:, j]) for j in range(4)])
        hs.append(ladder(perm).h)
    arr = np.array(hs)
    assert abs(arr.mean()) < 0.03


def test_ladder_ties_keep_input_order():
    x = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]])
    assert ladder(x).order == (0, 1, 2)


def test_weighted_grid_equals_the_expanded_grid():
    x = _random_labels(40, 3)
    w = np.random.randint(1, 5, size=40)
    g_w = grid(x, w.astype(float))
    g_e = grid(np.repeat(x, w, axis=0))
    assert np.allclose(np.nan_to_num(g_w.h, nan=9), np.nan_to_num(g_e.h, nan=9))
    assert np.allclose(g_w.d, g_e.d)

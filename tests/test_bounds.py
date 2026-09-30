"""Conservative outer bounds: on small cases every joint completion is enumerated, and the
complete-data Delta of each must lie inside the bounds."""
from __future__ import annotations

import itertools
import pathlib
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from rivaldefs import (
    DISTINCT,
    Rule,
    compare_rules,
    count_bounds,
    delta_bounds,
    grid,
    pair_table,
)


def _case():
    """Units with a few missing binary inputs; each labeler is a rule applied to the inputs."""
    n = random.randint(4, 7)
    k_inputs = 3
    inputs = [[random.choice((0, 1, None)) if random.random() < 0.25 else random.randint(0, 1)
               for _ in range(k_inputs)] for _ in range(n)]
    labelers = [
        lambda v: int(v[0] and v[1]),          # A variant 0
        lambda v: int(v[0] or v[1]),           # A variant 1
        lambda v: int(v[2]),                   # B variant 0
        lambda v: int(v[1] and v[2]),          # B variant 1
    ]
    units = []
    for row in inputs:
        miss = [i for i, v in enumerate(row) if v is None]
        comps = []
        for fill in itertools.product((0, 1), repeat=len(miss)):
            v = list(row)
            for i, f in zip(miss, fill):
                v[i] = f
            comps.append([lab(v) for lab in labelers])
        units.append(np.unique(np.array(comps), axis=0))
    a = Rule("A", (0, 1), ((0,), (1,)))
    b = Rule("B", (2, 3), ((0,), (1,)))
    return units, a, b


def test_every_completion_lies_inside_the_delta_bounds():
    for _ in range(150):
        units, a, b = _case()
        for within in ("independent", DISTINCT):
            bd = delta_bounds(units, a, b, within=within)
            for choice in itertools.product(*[range(len(u)) for u in units]):
                x = np.array([u[c] for u, c in zip(units, choice)])
                cmp = compare_rules(grid(x), a, b, within=within)
                assert bd.delta[0] - 1e-9 <= cmp.delta <= bd.delta[1] + 1e-9
                assert bd.d_between[0] - 1e-9 <= cmp.d_between <= bd.d_between[1] + 1e-9


def test_complete_data_bounds_collapse_to_the_point():
    x = np.random.randint(0, 2, size=(50, 4))
    a = Rule("A", (0, 1), ((0,), (1,)))
    b = Rule("B", (2, 3), ((0,), (1,)))
    bd = delta_bounds([row[None, :] for row in x], a, b)
    c = compare_rules(grid(x), a, b)
    assert bd.delta[0] == pytest.approx(c.delta) and bd.delta[1] == pytest.approx(c.delta)


def test_count_bounds_contain_every_completion():
    for _ in range(100):
        units, _, _ = _case()
        cb = count_bounds(units, 0, 2)
        for choice in itertools.product(*[range(len(u)) for u in units]):
            x = np.array([u[c] for u, c in zip(units, choice)])
            t = pair_table(x[:, 0], x[:, 2])
            for name in ("x", "y", "n10", "n01"):
                assert cb[name][0] <= getattr(t, name) <= cb[name][1]

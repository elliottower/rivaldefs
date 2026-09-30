"""Conservative outer bounds for missing data.

A unit with missing inputs has several completions, each a full vector of labels under every
labeler. Per-unit minima and maxima over completions, summed, bound x, y, n10, n01 and each
discordance d_ab in [l_ab, u_ab]. Weighted sums of the limits bound D_between and each D_within,
and Delta_L = D_between,L - max(D_within(A),U, D_within(B),U),
Delta_U = D_between,U - max(D_within(A),L, D_within(B),L). The bounds are not claimed to be
attainable jointly, and no bound on H is given.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._rules import (
    INDEPENDENT,
    Rule,
    _check_w,
    check_columns,
    d_within,
    factor_balanced,
    larger_within,
)


def _completions(units: Sequence[np.typing.ArrayLike]) -> list[np.ndarray]:
    out = []
    k = None
    for i, u in enumerate(units):
        arr = np.atleast_2d(np.asarray(u))
        if arr.shape[0] == 0 or not np.isin(arr, (0, 1)).all():
            raise ValueError(f"unit {i}: completions must be a non-empty 0/1 array")
        if k is None:
            k = arr.shape[1]
        elif arr.shape[1] != k:
            raise ValueError(f"unit {i}: every completion needs one label per labeler")
        out.append(arr.astype(bool))
    if not out:
        raise ValueError("no units")
    return out


def discordance_bounds(units: Sequence[np.typing.ArrayLike]) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds on d_ab (proportions) for every labeler pair.

    `units` holds, per unit, an (M_u x labelers) array of its possible label vectors; a unit
    with no missing inputs has M_u = 1.
    """
    comps = _completions(units)
    k = comps[0].shape[1]
    lo = np.zeros((k, k))
    hi = np.zeros((k, k))
    for arr in comps:
        dis = arr[:, :, None] != arr[:, None, :]
        lo += dis.min(axis=0)
        hi += dis.max(axis=0)
    n = len(comps)
    return lo / n, hi / n


def count_bounds(units: Sequence[np.typing.ArrayLike], a: int, b: int) -> dict[str, tuple[int, int]]:
    """Bounds on x, y, n10 and n01 for one labeler pair, as sums of per-unit minima and maxima."""
    comps = _completions(units)
    k = comps[0].shape[1]
    for i in (a, b):
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < k:
            raise ValueError(f"labeler index {i!r} is outside 0..{k - 1}")
    out = {}
    for name, f in (("x", lambda c: c[:, a]), ("y", lambda c: c[:, b]),
                    ("n10", lambda c: c[:, a] & ~c[:, b]), ("n01", lambda c: ~c[:, a] & c[:, b])):
        vals = [f(c) for c in comps]
        out[name] = (int(sum(v.min() for v in vals)), int(sum(v.max() for v in vals)))
    return out


@dataclass(frozen=True)
class DeltaBounds:
    """Bounds in percentage points."""

    d_between: tuple[float, float]
    d_within_a: tuple[float, float]
    d_within_b: tuple[float, float]
    delta: tuple[float, float]


def delta_bounds(units: Sequence[np.typing.ArrayLike], rule_a: Rule, rule_b: Rule,
                 w_a: Sequence[float] | None = None, w_b: Sequence[float] | None = None,
                 within: str = INDEPENDENT) -> DeltaBounds:
    lo, hi = discordance_bounds(units)
    check_columns(rule_a, lo.shape[0])
    check_columns(rule_b, lo.shape[0])
    lo, hi = lo * 100.0, hi * 100.0
    wa = factor_balanced(rule_a) if w_a is None else _check_w(np.asarray(w_a), rule_a)
    wb = factor_balanced(rule_b) if w_b is None else _check_w(np.asarray(w_b), rule_b)
    ca, cb = list(rule_a.columns), list(rule_b.columns)
    db = (float(wa @ lo[np.ix_(ca, cb)] @ wb), float(wa @ hi[np.ix_(ca, cb)] @ wb))
    ga, gb = rule_a.levels or None, rule_b.levels or None
    dwa = (d_within(lo[np.ix_(ca, ca)], wa, within, ga), d_within(hi[np.ix_(ca, ca)], wa, within, ga))
    dwb = (d_within(lo[np.ix_(cb, cb)], wb, within, gb), d_within(hi[np.ix_(cb, cb)], wb, within, gb))
    delta = (db[0] - larger_within(dwa[1], dwb[1]), db[1] - larger_within(dwa[0], dwb[0]))
    return DeltaBounds(d_between=db, d_within_a=dwa, d_within_b=dwb, delta=delta)

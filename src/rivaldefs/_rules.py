"""Rules, their admissible implementations, and between- versus within-rule disagreement.

A rule (a criterion) is a family of labelers, its implementations (variants), each described by
a tuple of factor levels. An implementation distribution w puts probability on the variants.
Definitions follow the registration: D_between, D_within, Delta = D_between - max(D_within),
R = D_between / max(D_within), and orientation stability S. Discordances, D terms and Delta are
returned in percentage points.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ._grid import Grid
from ._pair import A_SMALLER, B_SMALLER, EQUAL

INDEPENDENT = "independent"
DISTINCT = "conditional-distinct-pair"


@dataclass(frozen=True)
class Rule:
    """A rule and its admissible implementations.

    columns: the labeler columns (of a units x labelers matrix) that implement the rule.
    levels: one tuple of factor levels per column, in the same order.
    reference: the position, within `columns`, of the reference implementation.
    """

    name: str
    columns: tuple[int, ...]
    levels: tuple[tuple[Hashable, ...], ...] = field(default=())
    reference: int = 0

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError(f"rule {self.name!r} has no implementations")
        for c in self.columns:
            if isinstance(c, bool) or not isinstance(c, (int, np.integer)) or c < 0:
                raise ValueError(f"rule {self.name!r}: column {c!r} is not a non-negative index")
        if self.levels and len(self.levels) != len(self.columns):
            raise ValueError(f"rule {self.name!r}: one level tuple per column is required")
        if self.levels and len({len(t) for t in self.levels}) != 1:
            raise ValueError(f"rule {self.name!r}: every level tuple needs the same factors")
        if not 0 <= self.reference < len(self.columns):
            raise ValueError(f"rule {self.name!r}: reference position out of range")


def factor_balanced(rule: Rule) -> np.ndarray:
    """Every level of every factor equally likely, factors independent.

    For a full factorial this is uniform over the variants. A level combination listed more
    than once shares its probability equally among its copies. A combination missing from the
    grid is refused: the factor-balanced law is then not a distribution on the admissible set.
    """
    if not rule.levels:
        return np.full(len(rule.columns), 1.0 / len(rule.columns))
    n_factors = len(rule.levels[0])
    factor_levels = [sorted({t[f] for t in rule.levels}, key=repr) for f in range(n_factors)]
    combos = {tuple(c) for c in itertools.product(*factor_levels)}
    present = {tuple(t) for t in rule.levels}
    missing = combos - present
    if missing:
        raise ValueError(f"rule {rule.name!r} is not a full factorial; missing {sorted(missing, key=repr)[:3]}")
    p_combo = 1.0
    for lv in factor_levels:
        p_combo /= len(lv)
    copies: dict[tuple[Hashable, ...], int] = {}
    for t in rule.levels:
        copies[tuple(t)] = copies.get(tuple(t), 0) + 1
    return np.array([p_combo / copies[tuple(t)] for t in rule.levels])


def reference_weighted(rule: Rule) -> np.ndarray:
    """The reference variant has probability 0.5 and the rest share 0.5 uniformly."""
    k = len(rule.columns)
    if k == 1:
        return np.ones(1)
    w = np.full(k, 0.5 / (k - 1))
    w[rule.reference] = 0.5
    return w


def _check_w(w: np.ndarray, rule: Rule) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if w.shape != (len(rule.columns),) or np.any(w < 0) or not math.isclose(w.sum(), 1.0,
                                                                             abs_tol=1e-9):
        raise ValueError(f"weights for rule {rule.name!r} must be a distribution over its variants")
    return w


def d_within(dmat: np.ndarray, w: np.ndarray, within: str = INDEPENDENT,
             groups: Sequence[Hashable] | None = None) -> float:
    """Expected discordance of two draws from w (proportion).

    independent: two independent draws, a = a' contributing 0.
    conditional-distinct-pair: two independent draws conditional on drawing different
    implementations. `groups` names the implementation of each variant (its factor-level tuple),
    so copies of one implementation count as the same draw; by default every variant is its own.
    """
    total = float(w @ dmat @ w)
    if within == INDEPENDENT:
        return total
    if within == DISTINCT:
        keys = list(range(len(w))) if groups is None else list(groups)
        mass: dict[Hashable, float] = {}
        for k, wi in zip(keys, w):
            mass[k] = mass.get(k, 0.0) + float(wi)
        same = np.array([[ka == kb for kb in keys] for ka in keys])
        z = 1.0 - sum(v * v for v in mass.values())
        return math.nan if z <= 0 else float(w @ np.where(same, 0.0, dmat) @ w) / z
    raise ValueError(f"unknown within law {within!r}")


@dataclass(frozen=True)
class RuleComparison:
    """Between- versus within-rule disagreement for rules A and B (percentage points)."""

    d_between: float
    d_within_a: float
    d_within_b: float
    delta: float
    r: float
    s: float
    orientation_probabilities: dict[str, float]
    h_quantiles: dict[str, float]

    @property
    def denominator(self) -> float:
        return larger_within(self.d_within_a, self.d_within_b)


def larger_within(a: float, b: float) -> float:
    """max of two within-rule terms, NaN when either is undefined (a one-implementation rule
    under the conditional-distinct-pair law), whichever side it is on."""
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return max(a, b)


def check_columns(rule: Rule, k: int) -> None:
    """Refuse a rule naming a labeler column outside 0..k-1."""
    for c in rule.columns:
        if not 0 <= c < k:
            raise ValueError(f"rule {rule.name!r}: column {c} is outside 0..{k - 1}")


def compare_rules(g: Grid, rule_a: Rule, rule_b: Rule, w_a: Sequence[float] | None = None,
                  w_b: Sequence[float] | None = None, within: str = INDEPENDENT) -> RuleComparison:
    """Delta, R and S for two rules, computed exactly over every variant pair of the grid g.

    R here is the raw ratio; the registered reporting gate on R is applied by `bootstrap`.
    """
    k = g.n11.shape[0]
    check_columns(rule_a, k)
    check_columns(rule_b, k)
    wa = factor_balanced(rule_a) if w_a is None else _check_w(np.asarray(w_a), rule_a)
    wb = factor_balanced(rule_b) if w_b is None else _check_w(np.asarray(w_b), rule_b)
    ca, cb = list(rule_a.columns), list(rule_b.columns)
    d = g.d * 100.0
    db = float(wa @ d[np.ix_(ca, cb)] @ wb)
    dwa = d_within(d[np.ix_(ca, ca)], wa, within, rule_a.levels or None)
    dwb = d_within(d[np.ix_(cb, cb)], wb, within, rule_b.levels or None)
    den = larger_within(dwa, dwb)
    r = math.nan if not den > 0 else db / den
    joint = np.outer(wa, wb)
    orient = g.orientation[np.ix_(ca, cb)]
    probs = {o: float(joint[orient == o].sum()) for o in (A_SMALLER, B_SMALLER, EQUAL)}
    ref = g.orientation[ca[rule_a.reference], cb[rule_b.reference]]
    s = math.nan if ref == EQUAL else probs[ref]
    hq = _weighted_quantiles(g.h[np.ix_(ca, cb)].ravel(), joint.ravel(), (0.10, 0.50, 0.90))
    return RuleComparison(d_between=db, d_within_a=dwa, d_within_b=dwb, delta=db - den, r=r,
                          s=s, orientation_probabilities=probs,
                          h_quantiles={"p10": hq[0], "median": hq[1], "p90": hq[2]})


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray,
                        qs: Sequence[float]) -> list[float]:
    """Inverted-CDF weighted quantiles over defined values, weights renormalized."""
    ok = ~np.isnan(values) & (weights > 0)
    if not ok.any():
        return [math.nan for _ in qs]
    v, w = values[ok], weights[ok]
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cw = np.cumsum(w) / w.sum()
    return [float(v[min(int(np.searchsorted(cw, q - 1e-12)), len(v) - 1)]) for q in qs]

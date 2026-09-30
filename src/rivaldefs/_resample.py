"""Intervals by resampling: the participant bootstrap and survey replicate weights.

The participant bootstrap resamples the counts of the observed label patterns (a unit's labels
under every labeler) from a multinomial with the observed pattern frequencies. That is the
participant bootstrap itself, computed from the pattern counts. Every replicate recomputes each
labeler's positives, every pair's orientation and H, and every rule comparison, so one set of
resamples gives the intervals for H, C, Delta, R and S together.

Survey data take a matrix of replicate weights instead (for example the Rao-Wu rescaled
bootstrap weights of R's `survey` package); each column is one replicate.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ._grid import Grid, _check_weights, as_label_matrix, grid_from_gram
from ._pair import A_SMALLER, B_SMALLER, EQUAL, PairTable, h_delta_interval
from ._rules import INDEPENDENT, Rule, RuleComparison, compare_rules, factor_balanced

MIN_DEFINED = 0.90          # an interval needs at least this share of defined replicates
R_MIN_DENOMINATOR = 0.5     # percentage points: R reported only above this 2.5th percentile


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile interval over replicates.

    Replicates where the quantity is undefined are excluded. The interval is reported only when
    at least 90% of replicates are defined; `fraction_undefined` is always given.
    """

    estimate: float
    lo: float
    hi: float
    fraction_undefined: float
    reported: bool


def summarize(estimate: float, replicates: np.ndarray, level: float = 0.95,
              min_defined: float = MIN_DEFINED) -> Interval:
    reps = np.asarray(replicates, dtype=float)
    ok = ~np.isnan(reps)
    frac_undef = 1.0 - float(ok.mean()) if reps.size else 1.0
    if reps.size == 0 or ok.mean() < min_defined:
        return Interval(estimate, math.nan, math.nan, frac_undef, False)
    alpha = (1.0 - level) / 2
    lo, hi = np.quantile(reps[ok], [alpha, 1.0 - alpha])
    return Interval(estimate, float(lo), float(hi), frac_undef, True)


@dataclass(frozen=True)
class RulePair:
    """Two rules to compare, with their implementation distributions (default factor-balanced)
    and the law of the two within-rule draws."""

    rule_a: Rule
    rule_b: Rule
    w_a: Sequence[float] | None = None
    w_b: Sequence[float] | None = None
    within: str = INDEPENDENT


@dataclass(frozen=True)
class PairResult:
    a: int
    b: int
    table: PairTable
    h: Interval
    c: Interval
    orientation_probabilities: dict[str, float]
    h_delta_interval: tuple[float, float] | None

    @property
    def modal_orientation(self) -> str:
        return max(self.orientation_probabilities, key=lambda o: self.orientation_probabilities[o])


@dataclass(frozen=True)
class RulePairResult:
    spec: RulePair
    comparison: RuleComparison
    delta: Interval
    r: Interval
    r_reportable: bool
    denominator_p025: float
    s: Interval


@dataclass(frozen=True)
class Resampled:
    pairs: tuple[PairResult, ...]
    rule_pairs: tuple[RulePairResult, ...]
    n_replicates: int
    method: str


def bootstrap(labels: np.typing.ArrayLike, pairs: Sequence[tuple[int, int]] = (),
              rule_pairs: Sequence[RulePair] = (), n_boot: int = 2000,
              rng: np.random.Generator | int | None = None, level: float = 0.95) -> Resampled:
    """Participant bootstrap of every requested labeler pair and rule pair, from one set of
    resamples of the pattern counts."""
    x = as_label_matrix(labels)
    patterns, counts = np.unique(x, axis=0, return_counts=True)
    n = int(counts.sum())
    freq = counts / n
    gen = np.random.default_rng(rng)

    def replicate_counts() -> Iterator[np.ndarray]:
        for _ in range(n_boot):
            yield gen.multinomial(n, freq).astype(float)

    return _run(patterns, counts.astype(float), replicate_counts(), pairs, rule_pairs, level,
                method="participant bootstrap (pattern counts)", n_reps=n_boot)


def replicate_weights(labels: np.typing.ArrayLike, weights: np.typing.ArrayLike,
                      replicates: np.typing.ArrayLike, pairs: Sequence[tuple[int, int]] = (),
                      rule_pairs: Sequence[RulePair] = (), level: float = 0.95) -> Resampled:
    """Intervals from survey replicate weights (units x replicates), full-sample `weights`
    giving the point estimates. The interval is the percentile interval over replicates."""
    x = as_label_matrix(labels)
    w = _check_weights(weights, x.shape[0])
    rep = np.asarray(replicates, dtype=float)
    if rep.ndim != 2 or rep.shape[0] != x.shape[0]:
        raise ValueError("replicate weights must be a units x replicates matrix")
    if np.any(rep < 0) or not np.all(np.isfinite(rep)):
        raise ValueError("replicate weights must be finite and non-negative")

    def cols() -> Iterator[np.ndarray]:
        for j in range(rep.shape[1]):
            yield rep[:, j]

    return _run(x, w, cols(), pairs, rule_pairs, level, method="survey replicate weights",
                n_reps=rep.shape[1])


def _grid(patterns: np.ndarray, counts: np.ndarray) -> Grid:
    p = patterns.astype(float)
    return grid_from_gram((p * counts[:, None]).T @ p, float(counts.sum()))


def _run(patterns: np.ndarray, counts: np.ndarray, replicate_counts: Iterator[np.ndarray],
         pairs: Sequence[tuple[int, int]], rule_pairs: Sequence[RulePair], level: float,
         method: str, n_reps: int) -> Resampled:
    k = patterns.shape[1]
    for a, b in pairs:
        if not (0 <= a < k and 0 <= b < k) or a == b:
            raise ValueError(f"pair ({a}, {b}) does not name two labelers")
    specs = [_resolve(rp) for rp in rule_pairs]

    g0 = _grid(patterns, counts)
    point_pairs = [_pair_from_grid(g0, a, b) for a, b in pairs]
    point_rules = [compare_rules(g0, rp.rule_a, rp.rule_b, rp.w_a, rp.w_b, rp.within)
                   for rp in specs]

    ph = np.empty((n_reps, len(pairs)))
    pc = np.empty((n_reps, len(pairs)))
    po = np.empty((n_reps, len(pairs)), dtype=object)
    rd = np.empty((n_reps, len(specs)))
    rr = np.empty((n_reps, len(specs)))
    rden = np.empty((n_reps, len(specs)))
    rs = np.empty((n_reps, len(specs)))
    i = -1
    for i, cnt in enumerate(replicate_counts):
        g = _grid(patterns, cnt)
        for j, (a, b) in enumerate(pairs):
            ph[i, j] = g.h[a, b]
            pc[i, j] = g.c[a, b]
            po[i, j] = g.orientation[a, b]
        for j, rp in enumerate(specs):
            cmp = compare_rules(g, rp.rule_a, rp.rule_b, rp.w_a, rp.w_b, rp.within)
            rd[i, j] = cmp.delta
            rr[i, j] = cmp.r
            rden[i, j] = cmp.denominator
            rs[i, j] = cmp.s
    if i + 1 != n_reps:
        raise RuntimeError(f"expected {n_reps} replicates, got {i + 1}")

    pair_results = []
    for j, (a, b) in enumerate(pairs):
        t = point_pairs[j]
        probs = {o: float(np.mean(po[:, j] == o)) for o in (A_SMALLER, B_SMALLER, EQUAL)}
        pair_results.append(PairResult(
            a=a, b=b, table=t, h=summarize(t.h, ph[:, j], level), c=summarize(t.c, pc[:, j], level),
            orientation_probabilities=probs, h_delta_interval=h_delta_interval(t, level)))
    rule_results = []
    for j, rp in enumerate(specs):
        cmp = point_rules[j]
        den_ok = rden[:, j][~np.isnan(rden[:, j])]
        p025 = float(np.quantile(den_ok, (1.0 - level) / 2)) if den_ok.size else math.nan
        rule_results.append(RulePairResult(
            spec=rp, comparison=cmp, delta=summarize(cmp.delta, rd[:, j], level),
            r=summarize(cmp.r, rr[:, j], level), r_reportable=bool(p025 > R_MIN_DENOMINATOR),
            denominator_p025=p025, s=summarize(cmp.s, rs[:, j], level)))
    return Resampled(pairs=tuple(pair_results), rule_pairs=tuple(rule_results),
                     n_replicates=n_reps, method=method)


def _resolve(rp: RulePair) -> RulePair:
    wa = factor_balanced(rp.rule_a) if rp.w_a is None else np.asarray(rp.w_a, dtype=float)
    wb = factor_balanced(rp.rule_b) if rp.w_b is None else np.asarray(rp.w_b, dtype=float)
    return RulePair(rp.rule_a, rp.rule_b, [float(v) for v in wa], [float(v) for v in wb],
                    rp.within)


def _pair_from_grid(g: Grid, a: int, b: int) -> PairTable:
    n11 = float(g.n11[a, b])
    x, y = float(g.positives[a]), float(g.positives[b])
    return PairTable(n11=n11, n10=x - n11, n01=y - n11, n00=g.n - x - y + n11)

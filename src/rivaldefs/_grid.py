"""Many labelers at once: the pairwise grids, scale H and the ladder.

All of these are functions of the Gram matrix N11[a, b] (units positive under both a and b)
and the total N, so the same code serves raw data, weighted data and bootstrap resamples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._pair import A_SMALLER, B_SMALLER, EQUAL, TIE_RTOL


def as_label_matrix(labels: np.typing.ArrayLike) -> np.ndarray:
    """A units x labelers matrix of 0/1, as uint8; refuses anything else."""
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError("labels must be a units x labelers matrix")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("labels must have at least one unit and one labeler")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("labels must hold only 0 and 1")
    return arr.astype(np.uint8)


def gram(labels: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """N11 (labelers x labelers, diagonal = positives) and N, optionally weighted."""
    x = labels.astype(float)
    if weights is None:
        return x.T @ x, float(labels.shape[0])
    w = np.asarray(weights, dtype=float)
    return (x * w[:, None]).T @ x, float(w.sum())


@dataclass(frozen=True)
class Grid:
    """Pairwise indices for every ordered pair of labelers, rows a and columns b."""

    n: float
    positives: np.ndarray       # x for each labeler
    n11: np.ndarray
    h: np.ndarray
    c: np.ndarray
    kappa: np.ndarray
    d: np.ndarray
    nodf: np.ndarray
    orientation: np.ndarray     # object array of 'a smaller' / 'b smaller' / 'equal'
    m: np.ndarray
    e0: np.ndarray


def grid_from_gram(n11: np.ndarray, n: float) -> Grid:
    """The grids from a Gram matrix N11 (labelers x labelers, diagonal = positives, possibly
    weighted) and the total N. Lets a caller holding population cell counts or probabilities
    (N = 1) compute every pairwise index without unit-level labels."""
    n11 = np.asarray(n11, dtype=float)
    if n11.ndim != 2 or n11.shape[0] != n11.shape[1] or not np.allclose(n11, n11.T):
        raise ValueError("n11 must be a symmetric labelers x labelers matrix")
    if not (np.isfinite(n) and n > 0):
        raise ValueError("n must be finite and positive")
    x = np.diag(n11).astype(float)
    xa = x[:, None]
    yb = x[None, :]
    n10 = xa - n11
    n01 = yb - n11
    n00 = n - xa - yb + n11
    lo = np.minimum(xa, yb)
    hi = np.maximum(xa, yb)
    tie = np.abs(xa - yb) <= TIE_RTOL * n
    a_smaller = ~tie & (xa < yb)
    b_smaller = ~tie & (xa > yb)
    m = np.where(tie, (n10 + n01) / 2, np.where(b_smaller, n01, n10))
    e0 = lo * (n - hi) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(e0 > 0, 1.0 - m / e0, np.nan)
        c = np.where(lo > 0, n11 / lo, np.nan)
        kden = xa * (n - yb) + yb * (n - xa)
        kappa = np.where(kden > 0, 2 * (n11 * n00 - n10 * n01) / kden, np.nan)
        nodf = np.where(tie | (lo == 0), 0.0, n11 / np.where(lo > 0, lo, 1))
    d = (n10 + n01) / n
    orient = np.full(n11.shape, EQUAL, dtype=object)
    orient[a_smaller] = A_SMALLER
    orient[b_smaller] = B_SMALLER
    return Grid(n=n, positives=x, n11=n11, h=h, c=c, kappa=kappa, d=d, nodf=nodf,
                orientation=orient, m=m, e0=e0)


def grid(labels: np.typing.ArrayLike, weights: np.typing.ArrayLike | None = None) -> Grid:
    """The K x K grids of H, C, kappa, d and NODF for a units x labelers matrix."""
    x = as_label_matrix(labels)
    w = None if weights is None else _check_weights(weights, x.shape[0])
    n11, n = gram(x, w)
    return grid_from_gram(n11, n)


def scale_h_from_gram(n11: np.ndarray, n: float) -> float:
    """Pooled H = 1 - sum(m)/sum(E0) over every unordered pair (Mokken's scale H)."""
    g = grid_from_gram(n11, n)
    iu = np.triu_indices(n11.shape[0], k=1)
    se0 = float(g.e0[iu].sum())
    return float("nan") if se0 == 0 else 1.0 - float(g.m[iu].sum()) / se0


def scale_h(labels: np.typing.ArrayLike, weights: np.typing.ArrayLike | None = None) -> float:
    x = as_label_matrix(labels)
    if x.shape[1] < 2:
        raise ValueError("scale H needs at least two labelers")
    w = None if weights is None else _check_weights(weights, x.shape[0])
    return scale_h_from_gram(*gram(x, w))


@dataclass(frozen=True)
class Ladder:
    """The prevalence ladder of a set of labelers.

    `order` lists the labelers loosest first (descending prevalence; ties keep input order).
    Rungs are the K + 1 patterns in which the first j labelers of the order are positive and
    the rest negative. `on_ladder` is L, `on_ladder_independent` is L0, `h` is 1 - (1-L)/(1-L0).
    """

    order: tuple[int, ...]
    prevalence: tuple[float, ...]
    on_ladder: float
    on_ladder_independent: float
    h: float


def ladder(labels: np.typing.ArrayLike, weights: np.typing.ArrayLike | None = None) -> Ladder:
    x = as_label_matrix(labels)
    w = np.ones(x.shape[0]) if weights is None else _check_weights(weights, x.shape[0])
    n = float(w.sum())
    prev = (x.astype(float) * w[:, None]).sum(axis=0) / n
    order = tuple(int(i) for i in sorted(range(x.shape[1]), key=lambda i: -prev[i]))
    ordered = x[:, order]
    # A pattern is a rung when its positives form a prefix: non-increasing along the order.
    rung = np.all(ordered[:, :-1] >= ordered[:, 1:], axis=1) if x.shape[1] > 1 else np.ones(
        x.shape[0], dtype=bool)
    on = float((w * rung).sum() / n)
    pi = prev[list(order)]
    k = len(order)
    l0 = 0.0
    for j in range(k + 1):
        l0 += float(np.prod(pi[:j]) * np.prod(1.0 - pi[j:]))
    lh = float("nan") if l0 == 1.0 else 1.0 - (1.0 - on) / (1.0 - l0)
    return Ladder(order=order, prevalence=tuple(float(p) for p in pi), on_ladder=on,
                  on_ladder_independent=l0, h=lh)


def _check_weights(weights: np.typing.ArrayLike, n_units: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape != (n_units,) or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite, non-negative and one per unit")
    if w.sum() == 0:
        raise ValueError("weights sum to zero")
    return w

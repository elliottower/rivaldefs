"""Indices of one pair of binary labelings: the 2x2 table and what it determines.

Definitions follow the Indices section of the registration (PREREG_v2, criteria-direction):
orientation, the minority cell m, its independence value E0, H = 1 - m/E0 (Loevinger's H,
equal to kappa/kappa_max), % inside C, discordance d and the NODF pair score. Every quantity
that the registration calls undefined is returned as NaN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

A_SMALLER = "a smaller"
B_SMALLER = "b smaller"
EQUAL = "equal"
ORIENTATIONS = (A_SMALLER, B_SMALLER, EQUAL)

#: Two prevalences are equal when they differ by at most TIE_RTOL times the total weight N.
#: TIE_RTOL = 2**20 machine epsilons (about 2.3e-10): it absorbs the rounding of sums of
#: floating survey weights (0.1 + 0.2 against 0.3), and integer counts, whose differences are
#: at least 1, are never merged while N < 4e9.
TIE_RTOL = 2.0**20 * float(np.finfo(float).eps)


def is_tie(x: float, y: float, n: float) -> bool:
    """Whether two (possibly weighted) positive counts are equal to within TIE_RTOL * n."""
    return abs(x - y) <= TIE_RTOL * n


def _orientation(x: float, y: float, n: float) -> str:
    if is_tie(x, y, n):
        return EQUAL
    return A_SMALLER if x < y else B_SMALLER


def check_level(level: float) -> float:
    if not (isinstance(level, (int, float)) and 0 < level < 1):
        raise ValueError(f"level must lie strictly between 0 and 1, got {level!r}")
    return float(level)


@dataclass(frozen=True)
class PairTable:
    """Counts, possibly weighted, of a pair of labelings a and b.

    n11 both positive, n10 a only, n01 b only, n00 neither.
    """

    n11: float
    n10: float
    n01: float
    n00: float

    def __post_init__(self) -> None:
        for name in ("n11", "n10", "n01", "n00"):
            v = getattr(self, name)
            if not (v >= 0) or math.isinf(v):
                raise ValueError(f"{name} must be a finite count >= 0, got {v!r}")
        if self.n == 0:
            raise ValueError("the table is empty")

    @property
    def n(self) -> float:
        return self.n11 + self.n10 + self.n01 + self.n00

    @property
    def x(self) -> float:
        """Positives under a."""
        return self.n11 + self.n10

    @property
    def y(self) -> float:
        """Positives under b."""
        return self.n11 + self.n01

    @property
    def orientation(self) -> str:
        """'a smaller', 'b smaller' or 'equal' (sample version; x and y within TIE_RTOL * N are
        'equal')."""
        return _orientation(self.x, self.y, self.n)

    @property
    def m(self) -> float:
        """The minority cell: n10 if x < y, n01 if x > y, and at a tie (n10 + n01)/2, which
        equals both to within the tie tolerance."""
        o = self.orientation
        if o == EQUAL:
            return (self.n10 + self.n01) / 2
        return self.n10 if o == A_SMALLER else self.n01

    @property
    def e0(self) -> float:
        """E0 = min(x, y)(N - max(x, y))/N, the minority cell expected under independence."""
        return min(self.x, self.y) * (self.n - max(self.x, self.y)) / self.n

    @property
    def h(self) -> float:
        """H = 1 - m/E0; NaN when E0 = 0 (a prevalence of 0 or 1)."""
        e0 = self.e0
        return math.nan if e0 == 0 else 1.0 - self.m / e0

    @property
    def c(self) -> float:
        """% inside, n11/min(x, y); NaN when min(x, y) = 0."""
        lo = min(self.x, self.y)
        return math.nan if lo == 0 else self.n11 / lo

    @property
    def d(self) -> float:
        """Discordance (n10 + n01)/N, as a proportion."""
        return (self.n10 + self.n01) / self.n

    @property
    def kappa(self) -> float:
        """Cohen's kappa, 2(n11 n00 - n10 n01)/[x(N - y) + y(N - x)]; NaN when undefined."""
        den = self.x * (self.n - self.y) + self.y * (self.n - self.x)
        return math.nan if den == 0 else 2 * (self.n11 * self.n00 - self.n10 * self.n01) / den

    @property
    def kappa_max(self) -> float:
        """The largest kappa these marginals allow; NaN when undefined."""
        n, x, y = self.n, self.x, self.y
        den = n * (x + y) - 2 * x * y
        return math.nan if den == 0 else 2 * (n * min(x, y) - x * y) / den

    @property
    def nodf(self) -> float:
        """NODF pair score as a proportion: C when prevalences differ, 0 when they are equal.

        A smaller set with no positives scores 0, the convention of the ecological software.
        """
        if self.orientation == EQUAL:
            return 0.0
        lo = min(self.x, self.y)
        return 0.0 if lo == 0 else self.n11 / lo

    def as_dict(self) -> dict[str, float | str]:
        return {
            "n11": self.n11, "n10": self.n10, "n01": self.n01, "n00": self.n00,
            "n": self.n, "x": self.x, "y": self.y, "orientation": self.orientation,
            "m": self.m, "e0": self.e0, "h": self.h, "c": self.c, "d": self.d,
            "kappa": self.kappa, "kappa_max": self.kappa_max, "nodf": self.nodf,
        }


def pair_table(a: np.typing.ArrayLike, b: np.typing.ArrayLike,
               weights: np.typing.ArrayLike | None = None) -> PairTable:
    """The (weighted) 2x2 table of two 0/1 labelings of the same units."""
    av = _as_binary(a, "a")
    bv = _as_binary(b, "b")
    if av.shape != bv.shape:
        raise ValueError(f"a and b label different numbers of units: {av.shape} vs {bv.shape}")
    w = np.ones(av.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != av.shape or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite, non-negative and one per unit")
    return PairTable(
        n11=float(np.sum(w * (av & bv))),
        n10=float(np.sum(w * (av & ~bv))),
        n01=float(np.sum(w * (~av & bv))),
        n00=float(np.sum(w * (~av & ~bv))),
    )


def _as_binary(v: np.typing.ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(v)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must hold only 0 and 1")
    return arr.astype(bool)


def h_delta_interval(t: PairTable, level: float = 0.95) -> tuple[float, float] | None:
    """Delta-method interval for H under multinomial sampling of the four cells.

    Low-level and ungated: defined only when the orientation is strict and 0 < m < E0, and None
    otherwise. The registered secondary interval also needs a modal bootstrap orientation
    probability of at least 0.95; `bootstrap` applies that gate in `PairResult.h_delta_interval`.
    """
    check_level(level)
    if t.orientation == EQUAL or not (0 < t.m < t.e0):
        return None
    n = t.n
    if t.orientation == A_SMALLER:
        p11, pm, po = t.n11 / n, t.n10 / n, t.n01 / n
    else:
        p11, pm, po = t.n11 / n, t.n01 / n, t.n10 / n
    a = p11 + pm            # prevalence of the smaller set
    bq = 1.0 - p11 - po     # one minus the prevalence of the larger set
    # f = pm / (a * bq), H = 1 - f; gradient in (p11, pm, po)
    g11 = -pm * (bq - a) / (a * bq) ** 2
    gm = p11 / (a * a * bq)
    go = pm / (a * bq * bq)
    g = np.array([g11, gm, go, 0.0])
    p = np.array([p11, pm, po, 1.0 - p11 - pm - po])
    cov = (np.diag(p) - np.outer(p, p)) / n
    sd = math.sqrt(max(float(g @ cov @ g), 0.0))
    z = _z(level)
    return (t.h - z * sd, t.h + z * sd)


def _z(level: float) -> float:
    # Normal quantile by bisection on erf; no scipy dependency.
    target = 0.5 + level / 2
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

"""Whether a pair's orientation is resolved, from a simulation's operating characteristics.

A pair is placed in a cell by its E0 band and its prevalence-ratio band (max/min). A cell
qualifies when, over every prespecified pair and scenario placed in it, the minimum bootstrap
coverage of H is >= 0.93 and the maximum orientation error is <= 0.05, separately in each
generator family; a cell missing a family does not qualify. A real pair is resolved when its
cell qualifies and the bootstrap probability of its modal orientation is >= 0.95. When its E0 or
ratio lies within 10% of a band edge, the cells on both sides of that edge must qualify.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

E0_EDGES: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0, 100.0)
RATIO_EDGES: tuple[float, ...] = (1.1, 1.5, 2.0)
FAMILIES: tuple[str, ...] = ("G1", "G2")
MIN_COVERAGE = 0.93
MAX_ORIENTATION_ERROR = 0.05
MIN_MODAL_PROBABILITY = 0.95
EDGE_MARGIN = 0.10

RESOLVED = "resolved"
NOT_RESOLVED = "not resolved"


def band(value: float, edges: Sequence[float]) -> int:
    """Index of the half-open band [edge_{i-1}, edge_i) holding value; the last band is open."""
    if math.isnan(value):
        raise ValueError("cannot place an undefined value in a band")
    i = 0
    while i < len(edges) and value >= edges[i]:
        i += 1
    return i


def cell(e0: float, ratio: float) -> tuple[int, int]:
    return band(e0, E0_EDGES), band(ratio, RATIO_EDGES)


def ratio_of(x: float, y: float) -> float:
    """max/min of two prevalences or counts; infinite when the smaller is 0."""
    lo, hi = min(x, y), max(x, y)
    return math.inf if lo == 0 else hi / lo


def _near(value: float, edges: Sequence[float], margin: float) -> list[int]:
    """The band of value, plus the band across any edge within margin (relative) of it."""
    b = band(value, edges)
    out = {b}
    for i, e in enumerate(edges):
        if abs(value - e) <= margin * e:
            out.update({i, i + 1})
    return sorted(out)


def cells_for(e0: float, ratio: float, margin: float = EDGE_MARGIN) -> list[tuple[int, int]]:
    """Every cell a real pair must qualify in: its own, and across any edge it lies near."""
    return [(i, j) for i in _near(e0, E0_EDGES, margin) for j in _near(ratio, RATIO_EDGES, margin)]


@dataclass(frozen=True)
class OCRecord:
    """One prespecified pair in one scenario: its population E0 and ratio, and its simulated
    bootstrap coverage of H and orientation error (None when the population orientation is
    undefined, since orientation error is computed only where it is defined)."""

    family: str
    e0: float
    ratio: float
    coverage: float
    orientation_error: float | None


def qualifying_cells(records: Iterable[OCRecord], min_coverage: float = MIN_COVERAGE,
                     max_error: float = MAX_ORIENTATION_ERROR,
                     families: Sequence[str] = FAMILIES) -> set[tuple[int, int]]:
    worst_cov: dict[tuple[tuple[int, int], str], float] = {}
    worst_err: dict[tuple[tuple[int, int], str], float] = {}
    for r in records:
        if r.family not in families:
            raise ValueError(f"unknown generator family {r.family!r}")
        key = (cell(r.e0, r.ratio), r.family)
        worst_cov[key] = min(worst_cov.get(key, math.inf), r.coverage)
        err = r.orientation_error
        worst_err[key] = max(worst_err.get(key, -math.inf), -math.inf if err is None else err)
    cells = {c for c, _ in worst_cov}
    ok = set()
    for c in cells:
        if all((c, f) in worst_cov and worst_cov[(c, f)] >= min_coverage
               and worst_err[(c, f)] <= max_error for f in families):
            ok.add(c)
    return ok


def orientation_status(e0: float, ratio: float, orientation_probabilities: dict[str, float],
                       qualifying: set[tuple[int, int]],
                       min_probability: float = MIN_MODAL_PROBABILITY,
                       margin: float = EDGE_MARGIN) -> tuple[str, str]:
    """('resolved' or 'not resolved', the modal orientation) for a real pair, from its plug-in
    E0 and ratio and its bootstrap orientation probabilities."""
    modal = max(orientation_probabilities, key=lambda o: orientation_probabilities[o])
    if math.isnan(e0):
        return NOT_RESOLVED, modal
    in_cells = all(c in qualifying for c in cells_for(e0, ratio, margin))
    ok = in_cells and orientation_probabilities[modal] >= min_probability
    return (RESOLVED if ok else NOT_RESOLVED), modal


def nesting_label(h: float, cut: float = 0.80) -> str:
    """Descriptive label of a resolved pair: 'nested' when H >= cut, else 'crossing'."""
    if math.isnan(h):
        raise ValueError("H is undefined")
    return "nested" if h >= cut else "crossing"

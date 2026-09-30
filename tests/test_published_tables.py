"""H = kappa / kappa_max in exact rational arithmetic on published 2x2 tables.

The registration's Foreknowledge names five published tables on which this identity was checked
(Guerra, the delirium table, Habek, Daghi, one Mana pair). Only tables whose four cells are
recorded in a committed file are used here; each fixture names that file and its commit.

- Guerra et al. 2026, J Neurol 273(3):193 (neurologist-assigned SPMS against the HERCULES
  definition), Table 3 as printed: diagnostic-criteria-overlap
  experiments/table_recovery/inputs.json, key guerra2026, published_cells (commit afd4ca6).
- Habek et al. 2018, Mult Scler Relat Disord 25 (2010 against 2017 McDonald criteria), cells as
  printed: neurology-nosology results/mcdonald_recovered_tables.json, key habek2018, route
  "printed" (commit 99a68ee).
- Meagher et al. 2014, BMC Med 12:164 (DSM-IV and DSM-5 delirium), the three tables recovered
  uniquely from the printed n_A, n_B, kappa and agreement: diagnostic-criteria-overlap
  results/recovery.json, keys meagher2014_strict, meagher2014_reading, meagher2014_permissive,
  status "unique" (commit afd4ca6). Which of the three was the registration's "delirium table"
  is not recorded, so all three are checked.

Daghi 2026 (no joint table recoverable: status "infeasible") and the Mana pair (not named) are not
pinned to recorded cells and are not included.
"""
from __future__ import annotations

import pathlib
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from rivaldefs import PairTable

# (source, n11, n10, n01, n00)
PUBLISHED = [
    ("guerra2026 Table 3", 326, 540, 1277, 18163),
    ("habek2018 printed", 39, 0, 44, 30),
    ("meagher2014 DSM-IV vs DSM-5 strict", 155, 355, 3, 255),
    ("meagher2014 DSM-5 strict vs permissive", 158, 0, 308, 302),
    ("meagher2014 DSM-IV vs DSM-5 permissive", 455, 55, 11, 247),
]


def _exact(n11: int, n10: int, n01: int, n00: int) -> tuple[Fraction, Fraction, Fraction]:
    n = n11 + n10 + n01 + n00
    x, y = n11 + n10, n11 + n01
    kappa = Fraction(2 * (n11 * n00 - n10 * n01), x * (n - y) + y * (n - x))
    kappa_max = Fraction(2 * (n * min(x, y) - x * y), n * (x + y) - 2 * x * y)
    m = n10 if x < y else n01 if x > y else n10
    h = 1 - Fraction(m) / Fraction(min(x, y) * (n - max(x, y)), n)
    return h, kappa, kappa_max


@pytest.mark.parametrize("source,n11,n10,n01,n00", PUBLISHED)
def test_h_equals_kappa_over_kappa_max_exactly_on_a_published_table(source, n11, n10, n01, n00):
    h, kappa, kappa_max = _exact(n11, n10, n01, n00)
    assert h == kappa / kappa_max, source
    t = PairTable(n11, n10, n01, n00)
    assert t.h == pytest.approx(float(h), abs=1e-12), source
    assert t.kappa == pytest.approx(float(kappa), abs=1e-12), source


def test_the_published_kappas_are_reproduced():
    # A fixture typed wrongly would still satisfy the identity; the printed kappa would not.
    printed = {"guerra2026 Table 3": 0.22, "meagher2014 DSM-IV vs DSM-5 strict": 0.22,
               "meagher2014 DSM-5 strict vs permissive": 0.29,
               "meagher2014 DSM-IV vs DSM-5 permissive": 0.82}
    for source, *cells in PUBLISHED:
        if source in printed:
            assert round(float(_exact(*cells)[1]), 2) == printed[source], source

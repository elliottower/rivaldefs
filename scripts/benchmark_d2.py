"""Time the registered D2 bootstrap at its planned size: 40,000 units, 188 labelers, 2,000
resamples, on synthetic labels (no patient data).

The labelers are laid out as the five D2 rules of the registration (126 + 18 + 36 + 4 + 4
implementations, full factorials of their factors). The rule comparisons are the six
clinical-administrative pairs under each of the three registered laws, and the labeler pairs
are the ten reference pairs, all from one set of resamples as in the analysis.

    uv run --no-project --with numpy --with-editable . python scripts/benchmark_d2.py [n_boot]

Writes results/benchmark_d2_<date>.json.
"""
from __future__ import annotations

import datetime
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rivaldefs import DISTINCT, INDEPENDENT, Rule, RulePair, bootstrap, reference_weighted

N_UNITS = 40_000
SEED = 20260930
RULES = {                      # name: factor sizes of the full factorial
    "sepsis3": (3, 3, 7, 2),
    "sirs": (3, 3, 2),
    "qsofa": (3, 3, 2, 2),
    "angus": (2, 2),
    "martin": (2, 2),
}
CLINICAL = ("sepsis3", "sirs", "qsofa")
ADMIN = ("angus", "martin")


def labels(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Rule]]:
    severity = rng.normal(size=N_UNITS)
    coded = 0.6 * severity + 0.8 * rng.normal(size=N_UNITS)
    cols: list[np.ndarray] = []
    rules: dict[str, Rule] = {}
    for name, sizes in RULES.items():
        base = coded if name in ADMIN else severity
        levels = list(itertools.product(*[range(s) for s in sizes]))
        start = len(cols)
        for lv in levels:
            shift = 0.05 * sum(lv)
            noise = rng.normal(scale=0.2, size=N_UNITS)
            cols.append((base + noise > 1.0 + shift).astype(np.uint8))
        rules[name] = Rule(name, tuple(range(start, len(cols))), tuple(levels), reference=0)
    return np.column_stack(cols), rules


def main() -> None:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rng = np.random.default_rng(SEED)
    x, rules = labels(rng)
    assert x.shape == (N_UNITS, 188), x.shape
    rule_pairs = []
    for a, b in itertools.product(CLINICAL, ADMIN):
        ra, rb = rules[a], rules[b]
        rule_pairs.append(RulePair(ra, rb))
        rule_pairs.append(RulePair(ra, rb, reference_weighted(ra), reference_weighted(rb)))
        rule_pairs.append(RulePair(ra, rb, within=DISTINCT))
    refs = [rules[r].columns[0] for r in RULES]
    pairs = list(itertools.combinations(refs, 2))
    n_patterns = int(np.unique(x, axis=0).shape[0])
    t0 = time.perf_counter()
    res = bootstrap(x, pairs=pairs, rule_pairs=rule_pairs, n_boot=n_boot, rng=rng)
    elapsed = time.perf_counter() - t0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=False).stdout.strip()
    out = {
        "what": "rivaldefs.bootstrap at the registered D2 size, synthetic labels",
        "date": datetime.date.today().isoformat(),
        "rivaldefs_commit": commit,
        "seed": SEED,
        "n_units": N_UNITS,
        "n_labelers": int(x.shape[1]),
        "rule_sizes": {k: len(v.columns) for k, v in rules.items()},
        "n_unique_patterns": n_patterns,
        "n_labeler_pairs": len(pairs),
        "n_rule_pairs": len(rule_pairs),
        "laws": [INDEPENDENT + " factor-balanced", INDEPENDENT + " reference-weighted", DISTINCT],
        "n_boot": n_boot,
        "seconds": round(elapsed, 1),
        "seconds_per_replicate": round(elapsed / n_boot, 4),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "python": platform.python_version(), "numpy": np.__version__,
                    "cpu_count": os.cpu_count()},
        "check": {"n_replicates": res.n_replicates,
                  "all_delta_reported": all(r.delta.reported for r in res.rule_pairs
                                            if r.spec.within == INDEPENDENT)},
    }
    path = Path("results") / f"benchmark_d2_{out['date']}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

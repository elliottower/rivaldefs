# rivaldefs

**Which of two rival labeling rules sits inside the other, and is the gap between the rules
larger than the gap between their implementations?**

[![Tests](https://github.com/elliottower/rivaldefs/actions/workflows/test.yml/badge.svg)](https://github.com/elliottower/rivaldefs/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Two rules applied to the same units split them into four groups: positive under both, under
the first only, under the second only, under neither. Agreement statistics such as κ reduce
the four to one number and lose the direction of the disagreement. Two pairs of rules can share
κ while one pair is nested (one rule is a stricter cut of the other) and the other crosses
(each rule picks out units the other misses).

`rivaldefs` reports the direction as a quantity, with its uncertainty, and compares the
disagreement between two rules with the disagreement among admissible implementations of each
rule. It implements the indices of a frozen preregistration (criteria-direction, PREREG_v2,
OSF https://osf.io/bpw6e/).

## The problem, concretely

N = 1,000, two pairs of rules:

| | positives (x, y) | both | only first | only second | κ | H | % inside |
|---|---|---:|---:|---:|---:|---:|---:|
| pair 1 | 100, 300 | 95 | 5 | 205 | 0.38 | 0.93 | 95% |
| pair 2 | 200, 200 | 101 | 99 | 99 | 0.38 | 0.38 | 51% |

κ is 13/34 for the first pair and 61/160 for the second, 0.38 both. In the first pair five
units break the nesting where independence would put seventy; in the second each rule has 99
units the other misses.

## Install

```bash
git clone https://github.com/elliottower/rivaldefs && cd rivaldefs && pip install -e ".[test]"
```

Python 3.10+, NumPy.

## Use

```python
import numpy as np
import rivaldefs as rd

t = rd.PairTable(n11=95, n10=5, n01=205, n00=695)
t.orientation   # 'a smaller'
t.h             # 0.928...  = 1 - m/E0 = 13/14, Loevinger's H (kappa / kappa_max)
t.c             # 0.95      share of the smaller set inside the larger
t.kappa         # 0.382...

# Many labelers at once: a units x labelers 0/1 matrix.
rng = np.random.default_rng()
theta = rng.normal(size=2000)
labels = np.column_stack([
    theta > 1.0,                                          # rule A, implementation 0
    theta > 0.9,                                          # rule A, implementation 1
    theta + rng.normal(scale=0.8, size=2000) > 0.6,       # rule B, implementation 0
    theta + rng.normal(scale=0.8, size=2000) > 0.7,       # rule B, implementation 1
]).astype(int)

g = rd.grid(labels)          # K x K grids of H, C, kappa, discordance, NODF, orientation
g.h[0, 2]

a = rd.Rule("A", columns=(0, 1), levels=((0,), (1,)))
b = rd.Rule("B", columns=(2, 3), levels=((0,), (1,)))
cmp = rd.compare_rules(g, a, b)      # factor-balanced implementation distribution
cmp.delta                    # D_between - max(D_within), percentage points
cmp.s                        # probability a random implementation pair keeps the reference orientation

res = rd.bootstrap(labels, pairs=[(0, 2)], rule_pairs=[rd.RulePair(a, b)], n_boot=2000)
res.pairs[0].h                         # Interval(estimate, lo, hi, fraction_undefined, reported)
res.pairs[0].orientation_probabilities
res.rule_pairs[0].delta
```

## What it computes

- **Pairs.** The 2×2 table (counts or weights), orientation (the smaller set, or `equal`),
  the minority cell m and its independence value E0, H = 1 − m/E0, % inside C, κ, κmax,
  discordance d and the NODF pair score. Undefined values are NaN: H when a prevalence is 0 or 1,
  C when the smaller set is empty.
- **Many labelers.** K × K grids, Mokken's scale H, and the prevalence ladder (on-ladder share,
  its independence value, ladder H).
- **Rules and implementations.** A `Rule` is a family of labelers with factor levels. Its
  implementation distribution is factor-balanced by default (uniform on a full factorial),
  reference-weighted as an alternative, and the within-rule draws can be conditioned on being
  distinct. `compare_rules` returns D_between, D_within, Δ, R, S and the weighted quantiles of H
  over implementation pairs, computed exactly over every variant pair.
- **Uncertainty.** `bootstrap` resamples the counts of observed label patterns (the
  participant bootstrap, from pattern counts) and recomputes every labeler, orientation, H and
  rule comparison per replicate. `replicate_weights` does the same from survey replicate weights.
  A replicate where H or R is undefined is excluded; an interval is reported only when at least
  90% of replicates are defined, and the fraction undefined is always returned.
- **Orientation rule.** `qualifying_cells` reads a simulation's operating characteristics by
  E0 band × prevalence-ratio band; `orientation_status` marks a real pair resolved when its
  cells qualify and its modal orientation has bootstrap probability ≥ 0.95.
- **Missing inputs.** `delta_bounds` gives conservative outer bounds on Δ from each unit's
  possible completions.

## Tests

```bash
make test
```

The tests check exact values on hand-worked tables, H against an independent count of
Guttman errors, H = κ/κmax on random tables, the invariance of Δ, R and S to duplicated
implementations, the equivalence of the pattern-count and unit bootstrap, and that every
completion of a small table with missing inputs falls inside the bounds.

## License

MIT

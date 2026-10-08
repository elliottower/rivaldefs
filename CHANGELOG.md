# Changelog

## 0.1.0 (2026-10-08)

First version, implementing the indices of the criteria-direction preregistration (PREREG_v2,
OSF https://osf.io/bpw6e/).

- `PairTable` and `pair_table`: orientation, m, E0, H = 1 − m/E0 (= κ/κmax), % inside C, κ, κmax,
  discordance and the NODF pair score, on counts or weights; undefined values are NaN.
- `h_delta_interval`: the delta-method interval for H under multinomial sampling of the cells.
- `grid`, `scale_h`, `ladder`: K × K grids, Mokken's scale H, and the prevalence ladder.
- `Rule`, `factor_balanced`, `reference_weighted`, `compare_rules`: D_between, D_within, Δ, R and
  S under a declared implementation distribution, with the conditional-distinct-pair law.
- `bootstrap` (pattern-count participant bootstrap) and `replicate_weights` (survey replicate
  weights), with the undefined-replicate rule and the R reporting rule.
- `qualifying_cells`, `orientation_status`, `nesting_label`: the orientation rule read from a
  simulation's operating characteristics.
- `delta_bounds`, `discordance_bounds`, `count_bounds`: conservative outer bounds for missing
  inputs.
- Result objects apply the registered gates: the delta-method H interval needs a modal
  orientation probability ≥ 0.95, and R is withheld below the denominator threshold; ungated
  values are kept under `*_ungated`.
- Ties of weighted prevalences within `TIE_RTOL` × N; symmetric NaN when a conditional-distinct
  within term is undefined; a prevalence-degenerate pair is never resolved; malformed
  operating-characteristic records, probabilities, indices, levels and replicate columns are
  refused.
- Randomized tests are reproducible from the seed printed on failure.
- `Interval.sd`: the bootstrap (or replicate-weight) standard deviation over defined replicates.
- `gram` and `grid_from_gram` are public, for grids from population cell probabilities.
- H = κ/κmax is checked in exact rational arithmetic on six recorded cross-classifications from
  four studies (Guerra, Habek, three Meagher tables, Mana), each pinned to a committed source
  file. The registration names five tables; the Daghi table is omitted because the article
  prints no cross-classification, a departure recorded in the implementation choices and the
  registration Log.
- Known limitations of this version are listed in the README.

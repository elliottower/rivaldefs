# Changelog

## 0.1.0 (unreleased)

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

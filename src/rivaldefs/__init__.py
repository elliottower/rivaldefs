"""rivaldefs: direction and implementation robustness of disagreement between rival rules."""
from ._bounds import DeltaBounds, count_bounds, delta_bounds, discordance_bounds
from ._direction import (E0_EDGES, NOT_RESOLVED, RATIO_EDGES, RESOLVED, OCRecord, band, cell,
                         cells_for, nesting_label, orientation_status, qualifying_cells, ratio_of)
from ._grid import Grid, Ladder, grid, ladder, scale_h
from ._pair import A_SMALLER, B_SMALLER, EQUAL, PairTable, h_delta_interval, pair_table
from ._resample import (Interval, PairResult, Resampled, RulePair, RulePairResult, bootstrap,
                        replicate_weights, summarize)
from ._rules import (DISTINCT, INDEPENDENT, Rule, RuleComparison, compare_rules, d_within,
                     factor_balanced, reference_weighted)

__version__ = "0.1.0"

__all__ = [
    "A_SMALLER", "B_SMALLER", "EQUAL", "PairTable", "pair_table", "h_delta_interval",
    "Grid", "grid", "scale_h", "Ladder", "ladder",
    "Rule", "factor_balanced", "reference_weighted", "d_within", "compare_rules",
    "RuleComparison", "INDEPENDENT", "DISTINCT",
    "Interval", "summarize", "RulePair", "PairResult", "RulePairResult", "Resampled",
    "bootstrap", "replicate_weights",
    "OCRecord", "qualifying_cells", "orientation_status", "cells_for", "cell", "band",
    "ratio_of", "nesting_label", "RESOLVED", "NOT_RESOLVED", "E0_EDGES", "RATIO_EDGES",
    "DeltaBounds", "delta_bounds", "discordance_bounds", "count_bounds",
]

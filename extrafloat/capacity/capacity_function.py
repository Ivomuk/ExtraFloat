"""
extrafloat/capacity/capacity_function.py
=========================================
Canonical, single-source implementation of the frozen Capacity(F) formula
and its weights-artifact loader.

Promoted from scripts/derive_capacity_function_and_backtest_frontier.py
(Deliverable 2, Stage 2) per docs/stage_4c_logger_implementation_plan.md
Section 2 ("Decision 1: APPROVED -- promote the formula"), so the live
Stage 4C shadow logger and any future scripts/ research re-run consume
the EXACT SAME frozen implementation, never two independently maintained
copies that could drift apart. This is the one case in this workstream
where code moves INTO extrafloat/ rather than being restated outward
from it -- refactoring/provenance hardening, not a model change. No
behavioral change from the pre-promotion implementation.

See docs/stage_4b_capacity_k_selection_findings.md for the frozen k
selection (k_shadow=0.05, k_sensitivity=0.04/0.06) this formula feeds,
and docs/deliverable_2_capacity_f_findings.md for the formula's own
design rationale (Cobb-Douglas geometric mean, +1 regularization,
approximate rather than exact scale-homogeneity).

THE FORMULA:
    Capacity(F) = k * ( exp(w_F*log1p(Float) + w_C*log1p(Commission)) - 1 )
                = k * ( (1+Float)^w_F * (1+Commission)^w_C - 1 )
i.e. k times a weighted geometric mean of (1+Float) and (1+Commission),
minus 1 (log-space implementation for numerical stability at real UGX
scale rather than chaining literal (1+x)**w products).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"float_activity_value_1m": 0.5, "commission": 0.5}


def compute_capacity_ugx(df: pd.DataFrame, cols: list, weights: dict, k: float) -> pd.Series:
    """Capacity(F) = k * (exp(sum_i w_i*log1p(x_i)) - 1). Valid wherever
    every x_i >= 0 (NaN elsewhere -- this function never zero-fills or
    otherwise decides missingness handling; callers gate that themselves,
    e.g. via a capacity_estimable mask, before or after calling this)."""
    valid_mask = pd.Series(True, index=df.index)
    for c in cols:
        valid_mask &= df[c].notna() & (df[c] >= 0)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    log1p_df = np.log1p(df.loc[valid_mask, cols])
    s_log = sum(weights[c] * log1p_df[c] for c in cols)
    out.loc[valid_mask] = k * (np.exp(s_log) - 1.0)
    return out


def load_capacity_weights(weights_file, fundamentals: list) -> tuple:
    """Reads derive_capacity_score_weights.py's recommended_weights and
    its top-level "version" field if a weights_file is given, else falls
    back to DEFAULT_WEIGHTS with a printed warning and version
    "default_unversioned". Validates sum-to-1, renormalizing defensively
    with a loud warning if not -- never silently wrong. Returns
    (weights_dict, version_string).

    Raises ValueError (never sys.exit) on a malformed/incomplete weights
    file -- this is a library function; CLI callers that want sys.exit
    behavior should catch ValueError themselves (see
    scripts/derive_capacity_function_and_backtest_frontier.py's own
    thin wrapper for the pattern)."""
    if weights_file is None:
        print(f"WARNING: no weights_file given -- using DEFAULT_WEIGHTS {DEFAULT_WEIGHTS}. "
              f"Run derive_capacity_score_weights.py first to produce a governed artifact.")
        weights = dict(DEFAULT_WEIGHTS)
        version = "default_unversioned"
    else:
        meta = json.loads(Path(weights_file).read_text())
        weights = dict(meta["recommended_weights"])
        missing = [c for c in fundamentals if c not in weights]
        if missing:
            raise ValueError(f"weights_file has no weight for {missing}.")
        weights = {c: weights[c] for c in fundamentals}
        version = str(meta.get("version", "unknown"))

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        print(f"WARNING: weights sum to {total}, not 1.0 -- renormalizing.")
        weights = {c: w / total for c, w in weights.items()}
    return weights, version

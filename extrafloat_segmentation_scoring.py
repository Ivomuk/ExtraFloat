"""
extrafloat_segmentation_scoring.py
=====================================
Deterministic, versioned capacity/business scoring for Uganda MTN MoMo
agent segmentation.

Why this module exists
-----------------------
The ensemble-clustering pipeline (`extrafloat_segmentation_pipeline.py`) and
the pack-profiling tier builder (`extrafloat_segmentation_profiling.py`)
both decide an agent's business tier by ranking *clusters* against the
current run's population and evenly splitting the ranked list across tier
buckets. That means an agent's tier can move even when its own behavior is
unchanged, purely because other agents' scores shifted the ranking or a
random seed reshaped the clustering — a real problem for a product whose
tier sizes a credit limit downstream, since the assignment is neither
individually reproducible nor auditable.

This module replaces that mechanism with a deterministic, per-agent
"capacity score": a small number of business-KPI factor groups (value,
activity, efficiency), each normalized against a *frozen* reference range
fixed at calibration time (not recomputed from the current run's
population), blended into a single [0, 1] score, and bucketed against
*frozen* cutoff thresholds. Given the same scorecard, the same agent
feature values always produce the same score and tier — regardless of
which other agents are present in the run. That invariant is the entire
point and is asserted directly in the test suite
(test_extrafloat_segmentation_scoring.py).

Governance model
-----------------
A "scorecard" (normalization ranges + group weights + tier cutoffs) is a
versioned artifact, produced offline by `calibrate_capacity_scorecard`
(typically via the `calibrate_scorecard.py` CLI) and reviewed/signed off by
a human before being pointed at from production config
(`scoring.scorecard_path`). Production scoring (`compute_capacity_score`,
`compute_agent_capacity`) only ever *reads* a scorecard — it never fits
one. This mirrors how `extrafloat_segmentation_drift.py` separates
baseline creation (`save_drift_baseline`) from baseline consumption
(`build_drift_report`).

This module is intentionally free of clustering-library imports (no
sklearn, hdbscan, umap) so it stays importable in minimal environments —
only numpy/pandas.

Market: Uganda (UG) — MTN Mobile Money agent segmentation.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from extrafloat_segmentation_pipeline import BUSINESS_SEGMENTS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SCORECARD_SCHEMA_VERSION: str = "1.0"

# Factor groups: each maps a business-meaning bucket to the raw KPI columns
# that feed it and the aggregation used to combine them into one raw value
# per agent. Lifted from extrafloat_segmentation_pipeline._compute_composite_score's
# value_cols / activity_cols / efficiency_cols (same columns, same intent —
# "how much value moves through this agent", "how active is this agent",
# "how efficiently does this agent convert activity into commission").
CAPACITY_FACTOR_GROUPS: dict[str, dict[str, Any]] = {
    "value": {
        "columns": [
            "commission", "cash_out_value_1m", "cash_out_value_3m",
            "cash_in_value_1m", "cash_in_value_6m",
            "payment_value_1m", "payment_value_3m",
        ],
        "agg": "sum",
    },
    "activity": {
        "columns": ["cash_out_vol_1m", "cash_out_vol_3m"],
        "agg": "sum",
    },
    "efficiency": {
        "columns": ["commission_per_value_3m", "commission_per_value_6m", "tenure_years"],
        "agg": "mean",
    },
}

DEFAULT_GROUP_WEIGHTS: dict[str, float] = {"value": 0.5, "activity": 0.3, "efficiency": 0.2}

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SCORING_CONFIG: dict[str, Any] = {
    # Upper quantile used as the frozen normalization ceiling at calibration
    # time (values beyond it are clipped to 1.0 rather than re-scaling
    # everyone else — see _apply_normalization). Using p99 rather than the
    # raw max keeps a handful of extreme outliers in the calibration sample
    # from compressing the useful range for everyone else.
    "normalization_upper_quantile": 0.99,
    # Minimum tenure (years) required to hold any tier above the lowest.
    # Agents below this are downgraded one tier regardless of score.
    "min_tenure_years": 0.25,
}


def _get_scoring_config(config: dict | None) -> dict[str, Any]:
    """Return merged scoring config, filling missing keys from defaults."""
    if config is None:
        return dict(DEFAULT_SCORING_CONFIG)
    merged = dict(DEFAULT_SCORING_CONFIG)
    merged.update(config)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# SHARED SCORING PRIMITIVES
# (used by both calibrate_capacity_scorecard and production scoring, so the
#  two paths cannot silently drift apart)
# ─────────────────────────────────────────────────────────────────────────────


def compute_raw_factor_frame(
    df: pd.DataFrame,
    factor_groups: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Aggregate raw KPI columns into one raw value per factor group per agent.

    Row-wise only (sum/mean across *columns* of the same row) — never
    touches other rows — which is what makes downstream scoring
    population-independent.

    Parameters
    ----------
    df : Agent feature DataFrame (raw, pre-normalization).
    factor_groups : Mapping of group_name -> {"columns": [...], "agg": "sum"|"mean"}.
                     Defaults to CAPACITY_FACTOR_GROUPS.

    Returns
    -------
    pd.DataFrame indexed like *df*, one column per factor group.
    """
    if factor_groups is None:
        factor_groups = CAPACITY_FACTOR_GROUPS

    out = pd.DataFrame(index=df.index)
    for group_name, spec in factor_groups.items():
        cols = [c for c in spec.get("columns", []) if c in df.columns]
        agg = spec.get("agg", "sum")

        if not cols:
            logger.warning(
                "compute_raw_factor_frame: none of %s found for group '%s' — "
                "raw value defaulted to 0.0.",
                spec.get("columns", []),
                group_name,
            )
            out[group_name] = 0.0
            continue

        block = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out[group_name] = block.sum(axis=1) if agg == "sum" else block.mean(axis=1)

    return out


def _fit_normalization_params(
    raw_frame: pd.DataFrame,
    upper_quantile: float = 0.99,
) -> dict[str, dict[str, float]]:
    """Fit a frozen [ref_min, ref_max] normalization range per factor-group column.

    ref_min is always 0.0 (these are non-negative business KPIs). ref_max is
    the *upper_quantile* of the calibration population, deterministic given
    the calibration data (no randomness).
    """
    params: dict[str, dict[str, float]] = {}
    for col in raw_frame.columns:
        ref_max = float(raw_frame[col].quantile(upper_quantile))
        params[col] = {"ref_min": 0.0, "ref_max": ref_max}
    return params


def _apply_normalization(
    raw_frame: pd.DataFrame,
    normalization: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Apply a frozen normalization range to each factor-group column.

    Values are clipped to [0, 1] — an agent scoring above the calibration
    population's frozen ceiling on a factor clips to 1.0 on that factor
    rather than silently shifting everyone else's normalization (which is
    what a live re-fit / live quantile would do).
    """
    out = pd.DataFrame(index=raw_frame.index)
    for col in raw_frame.columns:
        params = normalization.get(col, {"ref_min": 0.0, "ref_max": 0.0})
        ref_min = float(params.get("ref_min", 0.0))
        ref_max = float(params.get("ref_max", 0.0))
        if ref_max <= ref_min:
            out[col] = 0.0
            continue
        normalized = (raw_frame[col] - ref_min) / (ref_max - ref_min)
        out[col] = normalized.clip(lower=0.0, upper=1.0)
    return out


def _blend_group_score(
    normalized_frame: pd.DataFrame,
    group_weights: dict[str, float],
) -> pd.Series:
    """Weighted sum of normalized factor-group columns -> one score per agent, in [0, 1]."""
    score = pd.Series(0.0, index=normalized_frame.index)
    for group_name, weight in group_weights.items():
        if group_name not in normalized_frame.columns:
            continue
        score = score + float(weight) * normalized_frame[group_name]
    return score.clip(lower=0.0, upper=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# SCORECARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────


def validate_scorecard(scorecard: dict[str, Any]) -> None:
    """Raise ValueError if *scorecard* is structurally invalid.

    Checked invariants:
        - required top-level keys present
        - group_weights sum to ~1.0
        - cutoffs are strictly ascending and one shorter than tiers
        - normalization ranges are non-degenerate (ref_max >= ref_min)
    """
    required_keys = (
        "scorecard_version", "cutoff_version", "factor_groups", "group_weights",
        "normalization", "cutoffs", "tiers",
    )
    missing = [k for k in required_keys if k not in scorecard]
    if missing:
        raise ValueError(f"validate_scorecard: missing required key(s): {missing}")

    weights = scorecard["group_weights"]
    total = sum(float(w) for w in weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"validate_scorecard: group_weights must sum to 1.0 (got {total:.6f})."
        )

    tiers = scorecard["tiers"]
    cutoffs = scorecard["cutoffs"]
    if len(cutoffs) != len(tiers) - 1:
        raise ValueError(
            f"validate_scorecard: expected {len(tiers) - 1} cutoffs for "
            f"{len(tiers)} tiers, got {len(cutoffs)}."
        )
    if list(cutoffs) != sorted(cutoffs):
        raise ValueError(f"validate_scorecard: cutoffs must be strictly ascending: {cutoffs}")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError(f"validate_scorecard: cutoffs must be strictly ascending (no ties): {cutoffs}")

    for col, params in scorecard["normalization"].items():
        ref_min = float(params.get("ref_min", 0.0))
        ref_max = float(params.get("ref_max", 0.0))
        if ref_max < ref_min:
            raise ValueError(
                f"validate_scorecard: normalization['{col}'] has ref_max < ref_min "
                f"({ref_max} < {ref_min})."
            )


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION (offline, human-governed — never called from production scoring)
# ─────────────────────────────────────────────────────────────────────────────


def calibrate_capacity_scorecard(
    development_df: pd.DataFrame,
    factor_groups: dict[str, dict[str, Any]] | None = None,
    group_weights: dict[str, float] | None = None,
    cutoffs: list[float] | None = None,
    target_tier_proportions: dict[str, float] | list[float] | None = None,
    tiers: tuple[str, ...] | list[str] | None = None,
    cutoff_version: str = "provisional_v0",
    is_provisional: bool = True,
    population_description: str = "",
    config: dict | None = None,
) -> dict[str, Any]:
    """Fit a versioned capacity scorecard from a development population.

    This is an *offline* calibration step — a human decision, not something
    a production run does implicitly. It is never called from
    `run_extrafloat_segmentation`. Fits the normalization ranges once, then
    resolves tier cutoffs from one of three sources (in priority order):

    1. *cutoffs* passed explicitly — used verbatim.
    2. *target_tier_proportions* — cutoffs are the quantiles of the blended
       score that would produce those proportions on *development_df*. The
       resulting numeric thresholds are logged (and recorded in the
       scorecard) for human sign-off — they are resolved once here and then
       frozen, never recomputed against future run populations.
    3. Neither given — cutoffs default to an even split (matching the
       *proportions* the old live-quantile mechanism produced), but unlike
       that mechanism these are computed once now and then frozen forever.

    No randomness is used anywhere in this function — calibration is fully
    deterministic given the same *development_df* and arguments.

    Parameters
    ----------
    development_df : Agent feature DataFrame used to fit normalization ranges
                      and (if not given explicitly) tier cutoffs.
    factor_groups   : Defaults to CAPACITY_FACTOR_GROUPS.
    group_weights   : Defaults to DEFAULT_GROUP_WEIGHTS.
    cutoffs         : Optional explicit ascending cutoff list, length len(tiers)-1.
    target_tier_proportions : Optional dict {tier_name: proportion} or list of
                      proportions aligned with *tiers*, summing to ~1.0.
    tiers           : Defaults to BUSINESS_SEGMENTS (8-tier).
    cutoff_version  : Free-form version label stored in the scorecard.
    is_provisional  : Stored in calibration_metadata; True means this
                      scorecard has not yet been reviewed against real
                      production data and should not be treated as final.
    population_description : Free-text note on what *development_df* is
                      (sample size, date range, source) for audit purposes.
    config          : Scoring config; see DEFAULT_SCORING_CONFIG.

    Returns
    -------
    dict — a fully-populated scorecard artifact (see module docstring).
    """
    cfg = _get_scoring_config(config)
    if factor_groups is None:
        factor_groups = CAPACITY_FACTOR_GROUPS
    if group_weights is None:
        group_weights = dict(DEFAULT_GROUP_WEIGHTS)
    if tiers is None:
        tiers = tuple(BUSINESS_SEGMENTS)
    tiers = tuple(tiers)

    total_weight = sum(float(w) for w in group_weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"calibrate_capacity_scorecard: group_weights must sum to 1.0 "
            f"(got {total_weight:.6f})."
        )

    raw_frame = compute_raw_factor_frame(development_df, factor_groups)
    normalization = _fit_normalization_params(
        raw_frame, upper_quantile=float(cfg["normalization_upper_quantile"])
    )
    normalized_frame = _apply_normalization(raw_frame, normalization)
    blended_score = _blend_group_score(normalized_frame, group_weights)

    n_tiers = len(tiers)

    if cutoffs is not None:
        resolved_cutoffs = [float(c) for c in cutoffs]
        logger.info(
            "calibrate_capacity_scorecard: using explicit cutoffs=%s", resolved_cutoffs
        )
    elif target_tier_proportions is not None:
        if isinstance(target_tier_proportions, dict):
            proportions = [float(target_tier_proportions.get(t, 0.0)) for t in tiers]
        else:
            proportions = [float(p) for p in target_tier_proportions]
        if len(proportions) != n_tiers:
            raise ValueError(
                f"calibrate_capacity_scorecard: target_tier_proportions has "
                f"{len(proportions)} entries, expected {n_tiers} (one per tier)."
            )
        if abs(sum(proportions) - 1.0) > 1e-3:
            raise ValueError(
                f"calibrate_capacity_scorecard: target_tier_proportions must "
                f"sum to 1.0 (got {sum(proportions):.4f})."
            )
        cumulative = np.cumsum(proportions)[:-1]
        resolved_cutoffs = [float(blended_score.quantile(q)) for q in cumulative]
        logger.info(
            "calibrate_capacity_scorecard: derived cutoffs=%s from "
            "target_tier_proportions=%s",
            resolved_cutoffs,
            proportions,
        )
    else:
        quantile_points = [i / n_tiers for i in range(1, n_tiers)]
        resolved_cutoffs = [float(blended_score.quantile(q)) for q in quantile_points]
        logger.info(
            "calibrate_capacity_scorecard: no cutoffs or target proportions "
            "given — defaulting to an even split, cutoffs=%s",
            resolved_cutoffs,
        )

    # De-duplicate degenerate cutoffs (e.g. a heavily-zero-inflated factor
    # group can produce repeated quantiles) by nudging ties upward by an
    # epsilon, preserving strict ascending order required by validate_scorecard.
    for i in range(1, len(resolved_cutoffs)):
        if resolved_cutoffs[i] <= resolved_cutoffs[i - 1]:
            resolved_cutoffs[i] = resolved_cutoffs[i - 1] + 1e-9

    tier_assignment = assign_capacity_tier(
        blended_score,
        {"cutoffs": resolved_cutoffs, "tiers": list(tiers)},
    )
    achieved_proportions = {
        t: float((tier_assignment == t).mean()) for t in tiers
    }

    scorecard: dict[str, Any] = {
        "scorecard_version": SCORECARD_SCHEMA_VERSION,
        "cutoff_version": cutoff_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "factor_groups": factor_groups,
        "group_weights": group_weights,
        "normalization": normalization,
        "score_scale": "0_to_1",
        "cutoffs": resolved_cutoffs,
        "tiers": list(tiers),
        "safety": {"min_tenure_years": float(cfg["min_tenure_years"])},
        "calibration_metadata": {
            "population_description": population_description,
            "n_agents": int(len(development_df)),
            "expected_tier_proportions": achieved_proportions,
            "is_provisional": bool(is_provisional),
        },
    }
    validate_scorecard(scorecard)
    logger.info(
        "calibrate_capacity_scorecard: calibrated on %d agents — tier "
        "proportions=%s (is_provisional=%s)",
        len(development_df),
        {k: round(v, 4) for k, v in achieved_proportions.items()},
        is_provisional,
    )
    return scorecard


# ─────────────────────────────────────────────────────────────────────────────
# SCORECARD PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────


def save_scorecard(scorecard: dict[str, Any], path: str, overwrite: bool = False) -> str:
    """Persist a scorecard artifact as JSON.

    Parameters
    ----------
    scorecard : Scorecard dict, typically from calibrate_capacity_scorecard.
    path      : Destination file path.
    overwrite : When False (default) raises if *path* already exists — a
                scorecard is a governance artifact, not a scratch file, so
                clobbering one silently is the wrong default.

    Returns
    -------
    str  The path written to.
    """
    if os.path.isfile(path) and not overwrite:
        raise FileExistsError(
            f"save_scorecard: '{path}' already exists. Pass overwrite=True to "
            "replace it, or choose a new cutoff_version/path for the new scorecard."
        )
    validate_scorecard(scorecard)
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(scorecard, fh, indent=2, default=str)
    logger.info("save_scorecard: wrote scorecard '%s' to '%s'.", scorecard.get("cutoff_version"), path)
    return path


def load_scorecard(path: str) -> dict[str, Any]:
    """Load and validate a scorecard artifact from *path*.

    Raises
    ------
    FileNotFoundError if *path* does not exist.
    ValueError if the loaded scorecard fails `validate_scorecard`.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"load_scorecard: scorecard file '{path}' does not exist.")
    with open(path) as fh:
        scorecard = json.load(fh)
    validate_scorecard(scorecard)
    logger.info(
        "load_scorecard: loaded scorecard '%s' (cutoff_version=%s) from '%s'.",
        scorecard.get("scorecard_version"),
        scorecard.get("cutoff_version"),
        path,
    )
    return scorecard


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION SCORING (deterministic, per-agent, never fits anything)
# ─────────────────────────────────────────────────────────────────────────────


def compute_capacity_score(
    features_df: pd.DataFrame,
    scorecard: dict[str, Any],
) -> pd.Series:
    """Compute the deterministic capacity score for each agent in *features_df*.

    Uses only the frozen `scorecard["normalization"]` ranges and
    `scorecard["group_weights"]` — never anything derived from
    *features_df* itself, so the result for a given agent's feature values
    is identical regardless of which other agents are present.
    """
    raw_frame = compute_raw_factor_frame(features_df, scorecard["factor_groups"])
    normalized_frame = _apply_normalization(raw_frame, scorecard["normalization"])
    return _blend_group_score(normalized_frame, scorecard["group_weights"])


def assign_capacity_tier(
    capacity_score: pd.Series,
    scorecard: dict[str, Any],
) -> pd.Series:
    """Bucket *capacity_score* against the scorecard's frozen cutoff thresholds.

    Fixed-threshold bucketing — not a live quantile split — so an agent's
    tier depends only on its own score and the frozen cutoffs, never on the
    other agents present in the current run.
    """
    cutoffs = np.asarray(scorecard["cutoffs"], dtype=float)
    tiers = list(scorecard["tiers"])
    idx = np.searchsorted(cutoffs, capacity_score.values, side="right")
    idx = np.clip(idx, 0, len(tiers) - 1)
    return pd.Series(np.array(tiers, dtype=object)[idx], index=capacity_score.index, dtype="object")


def apply_tenure_safety_cap(
    tier: pd.Series,
    tenure_years: pd.Series,
    scorecard: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """Downgrade agents below the scorecard's minimum tenure by one tier step.

    Per-agent reapplication of the old cluster-level
    `_apply_safety_filters` tenure check
    (extrafloat_segmentation_profiling.py) — same intent (a newly active
    agent shouldn't land in a high tier purely on a short burst of activity),
    applied per agent against the tier ladder rather than per cluster.

    Returns
    -------
    (final_tier, flags) — flags is a Series of "" or a human-readable
    safety-flag message, aligned with *tier*.
    """
    min_tenure = float(scorecard.get("safety", {}).get("min_tenure_years", 0.0))
    tiers = list(scorecard["tiers"])
    tier_index = {t: i for i, t in enumerate(tiers)}

    tenure_filled = pd.to_numeric(tenure_years, errors="coerce").fillna(0.0)
    breach = tenure_filled < min_tenure

    current_idx = tier.map(tier_index).fillna(0).astype(int)
    downgraded_idx = np.where(breach.values, np.maximum(current_idx.values - 1, 0), current_idx.values)
    final_tier = pd.Series(np.array(tiers, dtype=object)[downgraded_idx], index=tier.index, dtype="object")

    flags = pd.Series("", index=tier.index, dtype="object")
    flag_msg = breach.apply(
        lambda b: f"tenure_years < min_tenure_years={min_tenure}" if b else ""
    )
    flags = flag_msg

    n_downgraded = int(breach.sum())
    if n_downgraded:
        logger.info(
            "apply_tenure_safety_cap: downgraded %d agent(s) below "
            "min_tenure_years=%.3f.",
            n_downgraded,
            min_tenure,
        )
    return final_tier, flags


def compute_agent_capacity(
    features_df: pd.DataFrame,
    scorecard: dict[str, Any],
    is_dormant: pd.Series | None = None,
) -> pd.DataFrame:
    """Top-level orchestrator: score -> raw tier -> tenure safety cap -> dormant override.

    Parameters
    ----------
    features_df : Agent feature DataFrame (raw KPI columns present, as
                   produced by `prepare_features`).
    scorecard    : A validated scorecard, e.g. from `load_scorecard`.
    is_dormant   : Optional boolean Series aligned with *features_df*. Dormant
                   agents are force-set to the lowest tier regardless of score
                   (matching the clustering pipeline's dormant handling).

    Returns
    -------
    pd.DataFrame indexed like *features_df* with columns:
        capacity_score, capacity_tier_raw, capacity_tier,
        capacity_safety_flags, scorecard_version, cutoff_version
    """
    validate_scorecard(scorecard)
    tiers = list(scorecard["tiers"])

    score = compute_capacity_score(features_df, scorecard)
    tier_raw = assign_capacity_tier(score, scorecard)

    tenure_years = (
        features_df["tenure_years"] if "tenure_years" in features_df.columns
        else pd.Series(0.0, index=features_df.index)
    )
    tier_final, flags = apply_tenure_safety_cap(tier_raw, tenure_years, scorecard)

    if is_dormant is not None:
        dormant_mask = is_dormant.reindex(features_df.index).fillna(False).astype(bool)
        tier_final = tier_final.where(~dormant_mask, tiers[0])
        flags = flags.where(~dormant_mask, "dormant")

    out = pd.DataFrame(
        {
            "capacity_score": score,
            "capacity_tier_raw": tier_raw,
            "capacity_tier": tier_final,
            "capacity_safety_flags": flags,
            "scorecard_version": scorecard.get("scorecard_version"),
            "cutoff_version": scorecard.get("cutoff_version"),
        },
        index=features_df.index,
    )

    tier_counts = out["capacity_tier"].value_counts().to_dict()
    logger.info("compute_agent_capacity: capacity_tier distribution — %s", tier_counts)
    return out

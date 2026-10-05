"""
extrafloat_shadow_risk_multiplier.py
=====================================
Shadow (read-only, additive, parallel) computation of a continuous,
PD-calibrated limit multiplier -- the "C3 hybrid" policy from the prior
backtest/sensitivity analysis -- run alongside the live discrete 4-tier
multiplier without ever changing it.

Hard requirements this module exists to satisfy:
  * No existing engine column's value may change. This module only reads
    columns already produced upstream (cal_pd, combined_cap, assigned_limit,
    policy_multiplier, agent_tier_ceiling_multiplier, ...) and only writes
    new columns named in SHADOW_OUTPUT_COLUMNS.
  * Shadow failure must never fail lending, but it must never fail silently.
    Every non-OK status is logged with enough context (artifact path,
    status, model version, affected row count, run_id) to notice a broken
    shadow pipeline in production, instead of it silently going stale.
  * Calibration (cal_pd -> calibrated_risk) and policy (calibrated_risk ->
    multiplier) are versioned SEPARATELY (shadow_isotonic_model_version vs.
    SHADOW_POLICY_VERSION) so a future "keep calibration v1, change the
    floor from 13.8% to 12%" change is reconstructable.

Must be called AFTER finalize_limits() -- it needs the live assigned_limit
for the transition-control clamp.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from extrafloat.engine.extrafloat_limit_engine_caps import (
    _clip_series,
    _get_config,
    _round_to_nearest,
    _safe_series,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Frozen scenario definitions, carried over verbatim from the prior
# backtest/sensitivity analysis. Promoting these to config.shadow.scenarios
# is deliberately deferred until shadow observations validate them.
# -----------------------------------------------------------------------------
SHADOW_SCENARIOS: dict = {
    "base": {"r_floor": 0.138, "m_min": 0.40, "m_max": 1.00},
    "conservative": {"r_floor": 0.12, "m_min": 0.40, "m_max": 1.00},
}

# Version of the *policy function* (_policy_3_hybrid + SHADOW_SCENARIOS),
# tracked separately from the calibration artifact's own version (read from
# its metadata JSON at load time). Bump this string whenever _policy_3_hybrid
# or SHADOW_SCENARIOS changes.
SHADOW_POLICY_VERSION = "c3_v1"

ISOTONIC_MODEL_FILENAME = "pd_isotonic_calibrated_risk.joblib"
ISOTONIC_METADATA_FILENAME = "pd_isotonic_calibrated_risk_metadata.json"

# Run-level statuses: an artifact-level failure that blocks every row.
STATUS_OK = "ok"
STATUS_NO_ARTIFACTS_DIR = "skipped_no_artifacts_dir"
STATUS_MISSING_ARTIFACT = "skipped_missing_artifact"
STATUS_LOAD_ERROR = "skipped_load_error"
STATUS_UNEXPECTED_ERROR = "skipped_unexpected_error"

# Row-level statuses: only reachable once the artifact itself loaded
# successfully. A handful of borrowers missing cal_pd must not blank out
# shadow values for the whole run.
ROW_STATUS_OK = "ok"
ROW_STATUS_MISSING_CAL_PD = "missing_cal_pd"
ROW_STATUS_MISSING_COMBINED_CAP = "missing_combined_cap"
ROW_STATUS_MISSING_ASSIGNED_LIMIT = "missing_assigned_limit"

_SCENARIO_NAMES = tuple(SHADOW_SCENARIOS.keys())

SHADOW_OUTPUT_COLUMNS = (
    [
        "shadow_status",
        "shadow_isotonic_model_version",
        "shadow_policy_version",
        "shadow_calibrated_risk",
        "live_tier_multiplier",
    ]
    + [f"shadow_multiplier_{name}" for name in _SCENARIO_NAMES]
    + [f"shadow_limit_pre_transition_{name}" for name in _SCENARIO_NAMES]
    + [f"shadow_limit_post_transition_{name}" for name in _SCENARIO_NAMES]
)

# Tolerance for the artifact self-consistency check: model.predict(cal_pd_plateau)
# must match the metadata's stored r_plateau to within this much.
_R_PLATEAU_TOLERANCE = 1e-4

DEFAULT_SHADOW_CONFIG = {
    "enabled": True,
    "transition_delta_up": 0.25,
    "transition_delta_down": 0.25,
}


def _policy_3_hybrid(r, r_plateau, r_floor, m_max, m_min):
    """Hybrid plateau+floor multiplier policy, continuous in calibrated risk.

    Moved verbatim from scripts/backtest_limit_multiplier_policies.py so that
    scripts/ can import it FROM here: this repo's convention is scripts/
    depends on extrafloat/engine/, never the reverse.

    Non-increasing in r: flat at m_max up to r_plateau, linear decline to
    m_min by r_floor, flat at m_min beyond r_floor.
    """
    r = np.asarray(r, dtype="float64")
    out = np.full_like(r, m_max, dtype="float64")
    mask = r > r_plateau
    span = max(r_floor - r_plateau, 1e-9)
    out[mask] = m_max - (m_max - m_min) * np.clip((r[mask] - r_plateau) / span, 0, 1)
    out[r > r_floor] = m_min
    return out


def load_isotonic_calibration(artifacts_dir):
    """Load the frozen isotonic calibration model + its metadata. Never raises.

    Returns (model_or_None, metadata_dict, status). status == STATUS_OK only
    when the model loaded AND its self-consistency check against the
    metadata's recorded plateau passed.
    """
    if not artifacts_dir:
        return None, {}, STATUS_NO_ARTIFACTS_DIR

    artifacts_dir = Path(artifacts_dir)
    model_path = artifacts_dir / ISOTONIC_MODEL_FILENAME
    meta_path = artifacts_dir / ISOTONIC_METADATA_FILENAME
    if not model_path.exists() or not meta_path.exists():
        return None, {}, STATUS_MISSING_ARTIFACT

    try:
        model = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text())
    except Exception:
        logger.exception(
            "load_isotonic_calibration: failed to load artifact at %s / %s", model_path, meta_path
        )
        return None, {}, STATUS_LOAD_ERROR

    try:
        cal_pd_plateau = float(metadata["cal_pd_plateau"])
        r_plateau = float(metadata["r_plateau"])
        predicted = float(model.predict([cal_pd_plateau])[0])
    except Exception:
        logger.exception(
            "load_isotonic_calibration: metadata missing/invalid required fields at %s", meta_path
        )
        return None, metadata if isinstance(metadata, dict) else {}, STATUS_LOAD_ERROR

    if abs(predicted - r_plateau) > _R_PLATEAU_TOLERANCE:
        logger.error(
            "load_isotonic_calibration: artifact-integrity check FAILED at %s -- "
            "model.predict(cal_pd_plateau=%.4f)=%.6f vs. metadata r_plateau=%.6f (version=%s)",
            meta_path, cal_pd_plateau, predicted, r_plateau, metadata.get("version"),
        )
        return None, metadata, STATUS_LOAD_ERROR

    return model, metadata, STATUS_OK


def _effective_ceiling(df, cfg):
    """Exact mirror of apply_policy_adjustments()'s effective_ceiling_policy."""
    agent_tier_cfg = cfg.get("agent_tier", {})
    global_ceiling = cfg["global_ceiling_limit"]
    if agent_tier_cfg.get("enabled", False):
        return _clip_series(
            _safe_series(df, "agent_tier_ceiling_multiplier", 1.0) * global_ceiling,
            cfg["global_floor_limit"],
            global_ceiling,
        )
    return pd.Series(global_ceiling, index=df.index, dtype="float64")


def _apply_shadow_finishing_steps(df, combined_cap, shadow_tier_cap, cfg):
    """Deliberate, documented duplicate of apply_policy_adjustments()'s
    finishing steps that actually reach its stored `policy_cap` column
    (proven-good floor, active-borrower floor, effective-ceiling clip),
    re-implemented rather than called into so shadow-only code never
    couples to the live decision path.

    IMPORTANT: apply_policy_adjustments() also computes a rounded
    `final_cap` + a rounding-floor correction internally, but never stores
    either on `df` -- `df["policy_cap"]` is assigned the UNROUNDED value.
    Rounding against the real live `assigned_limit` happens separately, in
    finalize_limits() (run_extrafloat_limit_engine.py), via a plain
    `_round_to_nearest` with NO floor correction. So this function
    deliberately mirrors only what live `policy_cap` actually is -- the
    caller applies finalize_limits()-equivalent rounding afterward (see
    compute_shadow_risk_multiplier) to get an assigned_limit-equivalent
    value. Mirroring the dead rounding-floor-correction code here would
    make the shadow number diverge from what live code actually produces.

    If apply_policy_adjustments()'s finishing logic ever changes, this must
    be hand-updated to match -- the parameterized anti-drift test in
    tests/engine/test_extrafloat_shadow_risk_multiplier.py (which feeds this
    function the SAME discrete tier multiplier the live function used, across
    every finishing-step branch) fails loudly on drift.
    """
    policy_cfg = cfg["policy"]
    global_ceiling = cfg["global_ceiling_limit"]
    effective_ceiling = _effective_ceiling(df, cfg)

    raw_policy_cap = shadow_tier_cap

    total_loans = _safe_series(df, "total_loans", 0.0)
    on_time_rate = _clip_series(_safe_series(df, "on_time_repayment_rate", 0.0), 0.0, 1.0)
    lifetime_default = _clip_series(_safe_series(df, "lifetime_default_rate", 1.0), 0.0, 1.0)

    proven_good_mask = (
        (total_loans >= policy_cfg["proven_good_borrower_min_loans"])
        & (on_time_rate >= policy_cfg["proven_good_borrower_min_on_time_rate"])
        & (lifetime_default <= policy_cfg["proven_good_borrower_max_lifetime_default"])
    )
    proven_good_floor = combined_cap * policy_cfg["proven_good_borrower_floor_pct_of_combined"]

    policy_cap = pd.Series(
        np.where(proven_good_mask, np.maximum(raw_policy_cap, proven_good_floor), raw_policy_cap),
        index=df.index,
        dtype="float64",
    )

    recent_activity = _safe_series(df, "recent_disbursement_amount_1m", 0.0) + _safe_series(
        df, "recent_repayment_amount_1m", 0.0
    )
    active_floor = policy_cfg.get("active_borrower_min_limit", 500.0)
    active_floor_min_activity = policy_cfg.get("active_borrower_min_activity_amount", 1.0)
    active_floor_eligible = recent_activity >= active_floor_min_activity

    policy_cap = pd.Series(
        np.where(active_floor_eligible, np.maximum(policy_cap, active_floor), policy_cap),
        index=df.index,
        dtype="float64",
    )

    policy_cap = pd.Series(
        np.minimum(policy_cap.values, effective_ceiling.values), index=df.index, dtype="float64"
    )
    policy_cap = _clip_series(policy_cap, cfg["global_floor_limit"], global_ceiling)
    return policy_cap


def _apply_shadow_rounding(policy_cap, cfg):
    """Mirror of finalize_limits()'s clip + round (run_extrafloat_limit_engine.py)
    -- the step that turns a policy_cap-equivalent value into an
    assigned_limit-equivalent value. Deliberately does NOT replicate the
    Bank-of-Uganda regulatory cap clip: at the current DEFAULT_CAP_CONFIG
    values (regulatory_cap=5,000,000 > global_ceiling_limit=1,000,000, and
    policy_cap is already clipped to global_ceiling_limit before this point)
    that clip is always a no-op, so skipping it does not create any
    divergence from live behavior today; revisit if those config values
    ever change relative to each other.
    """
    clipped = _clip_series(policy_cap, cfg["global_floor_limit"], cfg["global_ceiling_limit"])
    return _round_to_nearest(clipped, cfg["rounding"]["round_to_nearest"])


def _log_status_summary(out, run_id, model_version):
    counts = out["shadow_status"].value_counts(dropna=False).to_dict()
    n_total = len(out)
    n_ok = int(counts.get(ROW_STATUS_OK, 0))
    log_fn = logger.info if n_ok == n_total else logger.warning
    log_fn(
        "compute_shadow_risk_multiplier: shadow_status summary %s (model_version=%s run_id=%s)",
        counts,
        model_version,
        run_id,
    )


def compute_shadow_risk_multiplier(df, artifacts_dir, config=None, run_id=None):
    """Additive, read-only shadow computation of the continuous C3 multiplier.

    Must be called AFTER finalize_limits() -- needs the live assigned_limit
    for the transition-control clamp. Never raises; never writes to any
    column outside SHADOW_OUTPUT_COLUMNS; never mutates an existing column.
    """
    cfg = _get_config(config)
    shadow_cfg = {**DEFAULT_SHADOW_CONFIG, **cfg.get("shadow", {})}

    out = df.copy()
    for col in SHADOW_OUTPUT_COLUMNS:
        out[col] = np.nan

    model, metadata, run_status = load_isotonic_calibration(artifacts_dir)
    if run_status != STATUS_OK:
        out["shadow_status"] = run_status
        logger.warning(
            "compute_shadow_risk_multiplier: shadow skipped for all %d rows -- "
            "status=%s artifacts_dir=%s model_version=%s run_id=%s",
            len(out),
            run_status,
            artifacts_dir,
            metadata.get("version"),
            run_id,
        )
        return out

    model_version = metadata.get("version", "unknown")
    r_plateau = float(metadata["r_plateau"])

    cal_pd = (
        pd.to_numeric(_safe_series(df, "cal_pd", np.nan), errors="coerce")
        if "cal_pd" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    combined_cap = (
        pd.to_numeric(_safe_series(df, "combined_cap", np.nan), errors="coerce")
        if "combined_cap" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    assigned_limit = (
        pd.to_numeric(_safe_series(df, "assigned_limit", np.nan), errors="coerce")
        if "assigned_limit" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )

    valid_cal_pd = cal_pd.notna() & np.isfinite(cal_pd)
    valid_combined_cap = combined_cap.notna() & np.isfinite(combined_cap)
    valid_assigned_limit = assigned_limit.notna() & np.isfinite(assigned_limit)
    valid = valid_cal_pd & valid_combined_cap & valid_assigned_limit

    row_status = pd.Series(ROW_STATUS_OK, index=df.index, dtype="object")
    row_status = pd.Series(
        np.where(~valid_assigned_limit, ROW_STATUS_MISSING_ASSIGNED_LIMIT, row_status),
        index=df.index,
        dtype="object",
    )
    row_status = pd.Series(
        np.where(~valid_combined_cap, ROW_STATUS_MISSING_COMBINED_CAP, row_status),
        index=df.index,
        dtype="object",
    )
    row_status = pd.Series(
        np.where(~valid_cal_pd, ROW_STATUS_MISSING_CAL_PD, row_status),
        index=df.index,
        dtype="object",
    )
    out["shadow_status"] = row_status
    out["shadow_isotonic_model_version"] = np.where(valid, model_version, pd.NA)
    out["shadow_policy_version"] = np.where(valid, SHADOW_POLICY_VERSION, pd.NA)
    out["live_tier_multiplier"] = np.where(valid, _safe_series(df, "policy_multiplier", np.nan), np.nan)

    if not valid.any():
        _log_status_summary(out, run_id, model_version)
        return out

    calibrated_risk = pd.Series(np.nan, index=df.index, dtype="float64")
    calibrated_risk[valid] = model.predict(cal_pd[valid].to_numpy())
    out["shadow_calibrated_risk"] = calibrated_risk

    delta_up = float(shadow_cfg["transition_delta_up"])
    delta_down = float(shadow_cfg["transition_delta_down"])
    effective_ceiling = _effective_ceiling(df, cfg)
    global_floor = cfg["global_floor_limit"]

    for name, params in SHADOW_SCENARIOS.items():
        multiplier = pd.Series(np.nan, index=df.index, dtype="float64")
        multiplier[valid] = _policy_3_hybrid(
            calibrated_risk[valid].to_numpy(), r_plateau, params["r_floor"], params["m_max"], params["m_min"]
        )
        out[f"shadow_multiplier_{name}"] = multiplier

        shadow_tier_cap = combined_cap * multiplier
        shadow_policy_cap = _apply_shadow_finishing_steps(df, combined_cap, shadow_tier_cap, cfg)
        pre_transition = _apply_shadow_rounding(shadow_policy_cap, cfg)
        pre_transition = pd.Series(np.where(valid, pre_transition, np.nan), index=df.index, dtype="float64")
        out[f"shadow_limit_pre_transition_{name}"] = pre_transition

        lower = assigned_limit * (1.0 - delta_down)
        upper = assigned_limit * (1.0 + delta_up)
        post_transition = pre_transition.clip(lower=lower, upper=upper)
        # Final safety clip. The lower bound (>= 0) is not a new constraint:
        # global_floor_limit == 0.0 in DEFAULT_CAP_CONFIG and every live cap
        # already enforces it unconditionally, and _apply_shadow_finishing_steps
        # already applied it once above. The substantive guarantee added here
        # is the upper bound: ShadowLimit <= EffectiveCeiling holds by
        # construction regardless of how the transition window interacts with
        # this borrower's ceiling.
        post_transition = post_transition.clip(lower=global_floor, upper=effective_ceiling)
        post_transition = pd.Series(np.where(valid, post_transition, np.nan), index=df.index, dtype="float64")
        out[f"shadow_limit_post_transition_{name}"] = post_transition

    _log_status_summary(out, run_id, model_version)
    return out

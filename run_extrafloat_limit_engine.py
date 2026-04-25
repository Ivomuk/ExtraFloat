from extrafloat_limit_engine_caps import (
    _get_config,
    _safe_series,
    _clip_series,
    _round_to_nearest,
    _validate_config,        # was _validate_tier_config — now correctly named
    compute_capacity_cap,
    compute_recent_usage_cap,
    compute_prior_exposure_cap,
    compute_risk_cap,
    combine_caps,
    apply_policy_adjustments,
)

import numpy as np
import pandas as pd

# compute_risk_cap() derives risk_score internally from component features
# (on_time_repayment_rate, lifetime_default_rate, etc.), so no columns are
# strictly required at engine entry. Validation is kept for structural checks
# if required inputs are added in future.
REQUIRED_COLUMNS: list = []

OPTIONAL_BUT_EXPECTED_COLUMNS = [
    "avg_balance_30d",
    "avg_balance_90d",
    "net_cashflow_30d",
    "net_cashflow_90d",
    "txn_count_30d",
    "txn_count_90d",
    "payments_in_30d",
    "payments_in_90d",
    "active_customers_30d",
    "active_customers_90d",
    "avg_prior_loan_size",
    "max_prior_loan_size",
    "current_loan_size",
    "recent_repayment_performance",
    "recent_disbursement_volume",
    "recent_repayment_volume",
    "recent_penalty_count",
    "coverage_ratio",
    "prior_limit",
    "is_thin_file",
    "is_active_borrower",
    "lifetime_loan_count",
    "on_time_repayment_rate",
    "lifetime_default_rate",
]

FINAL_OUTPUT_COLUMNS = [
    "assigned_limit",
    "assigned_limit_pre_round",
    "final_decision_reason",
    "policy_reason",
    "combined_reason",
    "combined_top_driver",
    "risk_tier",
]


def validate_required_columns(features_df):
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in features_df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_cols))
        )


def validate_expected_columns(features_df):
    missing_expected_cols = [
        col for col in OPTIONAL_BUT_EXPECTED_COLUMNS if col not in features_df.columns
    ]
    return missing_expected_cols


def _final_limit_source_series(df):
    # Check column names in order of preference:
    # policy_cap   — written by apply_policy_adjustments() (primary path)
    # combined_cap — pre-policy fallback (e.g. if policy step was skipped)
    if "policy_cap" in df.columns:
        return _safe_series(df, "policy_cap", 0.0)
    if "combined_cap" in df.columns:
        return _safe_series(df, "combined_cap", 0.0)
    return pd.Series(0.0, index=df.index, dtype="float64")


def finalize_limits(features_df, config=None):
    cfg = _get_config(config)
    df = features_df.copy()

    final_cap = _clip_series(
        _final_limit_source_series(df),
        cfg["global_floor_limit"],
        cfg["global_ceiling_limit"],
    )

    rounded_limit = _round_to_nearest(
        final_cap,
        cfg["rounding"]["round_to_nearest"],
    )

    # ── Bank of Uganda regulatory cap (change E / L) ──────────────────────
    reg_cfg = cfg.get("regulatory", {})
    regulatory_cap_applied = pd.Series(False, index=df.index, dtype="bool")

    if reg_cfg.get("enabled", False):
        bou_cap = float(reg_cfg.get("regulatory_cap", cfg["global_ceiling_limit"]))
        regulatory_cap_applied = rounded_limit > bou_cap
        rounded_limit = pd.Series(
            np.minimum(rounded_limit.values, bou_cap),
            index=df.index,
            dtype="float64",
        )

    # ── Decision reason ────────────────────────────────────────────────────
    policy_reason_col = _safe_series(df, "policy_reason", "").astype(str)
    combined_reason_col = _safe_series(df, "combined_reason", "finalized_from_combined_cap").astype(str)

    final_decision_reason = pd.Series(
        np.where(
            policy_reason_col != "",
            policy_reason_col,
            combined_reason_col,
        ),
        index=df.index,
        dtype="object",
    )

    # Override reason for rows clipped by BoU regulatory cap
    if reg_cfg.get("enabled", False):
        final_decision_reason = pd.Series(
            np.where(
                regulatory_cap_applied,
                "bou_regulatory_cap_applied",
                final_decision_reason,
            ),
            index=df.index,
            dtype="object",
        )

    df["assigned_limit_pre_round"] = final_cap
    df["assigned_limit"] = rounded_limit
    df["final_decision_reason"] = final_decision_reason
    df["regulatory_cap_applied"] = regulatory_cap_applied.astype(int)

    return df


def _trim_output_columns(features_df, keep_intermediate=True):
    df = features_df.copy()
    if keep_intermediate:
        return df
    keep_cols = [col for col in FINAL_OUTPUT_COLUMNS if col in df.columns]
    return df[keep_cols]


def run_extrafloat_limit_engine(
    features_df,
    config=None,
    keep_intermediate=True,
    validate_inputs=True,
):
    cfg = _get_config(config)
    _validate_config(cfg)   # validates tier thresholds + weight sums

    if validate_inputs:
        validate_required_columns(features_df)

    df = features_df.copy()
    df = compute_capacity_cap(df, cfg)
    df = compute_recent_usage_cap(df, cfg)
    df = compute_prior_exposure_cap(df, cfg)
    df = compute_risk_cap(df, cfg)
    df = combine_caps(df, cfg)
    df = apply_policy_adjustments(df, cfg)
    df = finalize_limits(df, cfg)
    df = _trim_output_columns(df, keep_intermediate=keep_intermediate)
    return df


def summarize_engine_input_coverage(features_df):
    validate_required_columns(features_df)
    missing_expected_cols = validate_expected_columns(features_df)
    return {
        "row_count": int(len(features_df)),
        "required_columns_present": True,
        "missing_expected_columns": missing_expected_cols,
        "missing_expected_count": int(len(missing_expected_cols)),
    }


def extract_drift_snapshot(result_df):
    """
    Lightweight audit snapshot from an engine output DataFrame.

    Returns a flat dict suitable for persisting (log, file, DB) as a
    reference window to pass to extrafloat_drift_monitor.run_drift_monitor().
    Does NOT import from extrafloat_drift_monitor — no circular dependency.
    """
    def _rate(col):
        if col not in result_df.columns:
            return None
        s = pd.to_numeric(result_df[col], errors="coerce").fillna(0)
        return float((s > 0).mean())

    def _pct_stats(col):
        if col not in result_df.columns:
            return None
        s = pd.to_numeric(result_df[col], errors="coerce").dropna()
        if len(s) == 0:
            return None
        return {
            "mean": float(s.mean()),
            "p25":  float(s.quantile(0.25)),
            "p50":  float(s.quantile(0.50)),
            "p75":  float(s.quantile(0.75)),
            "p95":  float(s.quantile(0.95)),
        }

    def _cat_dist(col):
        if col not in result_df.columns:
            return None
        return result_df[col].value_counts(normalize=True).to_dict()

    return {
        "row_count":                          len(result_df),
        "run_timestamp":                      pd.Timestamp.utcnow().isoformat(),
        "assigned_limit":                     _pct_stats("assigned_limit"),
        "risk_score":                         _pct_stats("risk_score"),
        "risk_tier_distribution":             _cat_dist("risk_tier"),
        "combined_top_driver_distribution":   _cat_dist("combined_top_driver"),
        "capacity_top_driver_distribution":   _cat_dist("capacity_top_driver"),
        "policy_reason_distribution":         _cat_dist("policy_reason"),
        "regulatory_cap_rate":                _rate("regulatory_cap_applied"),
        "thin_file_rate":                     _rate("is_thin_file"),
        "proven_good_rate":                   _rate("is_proven_good_borrower"),
        "active_floor_applied_rate":          _rate("active_floor_applied"),
        "fallback_inputs_rate":               _rate("capacity_fallback_inputs"),
        "missing_inputs_rate":                _rate("capacity_missing_inputs"),
    }

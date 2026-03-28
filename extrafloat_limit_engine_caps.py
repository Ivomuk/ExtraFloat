from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CAP_CONFIG = {
    "global_floor_limit": 0.0,
    "global_ceiling_limit": 30000.0,
	
	
	
    "rounding": {
        "round_to_nearest": 100.0,
    },

    "capacity": {
        # Input multipliers
        "balance_multiplier": 0.80,
        "revenue_multiplier": 0.35,
        "txn_multiplier": 40.0,
        "payments_multiplier": 0.30,
        "customers_multiplier": 25.0,
        "volume_multiplier": 0.20,

        # Weighted contribution mix
        "balance_weight": 0.22,
        "revenue_weight": 0.28,
        "txn_weight": 0.12,
        "payments_weight": 0.14,
        "customers_weight": 0.09,
        "activity_weight": 0.15,

        # Operational activity attenuation
        "activity_inactive_floor": 0.35,
        "activity_active_weight": 0.65,
    },
	"experience": {
		"min_experience_factor": 0.30,
		"minimum_total_loans_for_full_trust": 10.0,
	},

    "recent_usage": {
        # Activity gate
        "minimum_activity_required": 100.0,

        # Base usage construction
        "disbursement_multiplier": 0.90,
        "repayment_multiplier": 1.00,
        "disbursement_weight": 0.45,
        "repayment_weight": 0.55,

        # Coverage lift
        "coverage_multiplier_floor": 0.85,
        "coverage_weight": 0.35,
        "coverage_bonus_cap": 1.20,

        # Penalty haircut
        "penalty_haircut_per_event": 0.10,

        # Repayment ratio attenuation
        "repayment_ratio_cap": 1.00,
        "repayment_ratio_floor_weight": 0.55,
    },

    "combination": {
        # Standard-file weights
        "capacity_weight": 0.40,
        "recent_usage_weight": 0.25,
        "risk_weight": 0.20,

        # Thin-file weights
        "thin_file_capacity_weight": 0.25,
        "thin_file_recent_usage_weight": 0.15,
        "thin_file_risk_weight": 0.50,
		"thin_file_threshold": 3.0,

        # Prior-limit smoothing
        "prior_limit_weight": 0.15,
        "prior_limit_max_upside": 1.25,
        "prior_limit_max_downside": 0.75,
    },
		"risk": {
		"base_limit": 30000.0,
		"on_time_weight": 0.30,
		"lifetime_default_weight": 0.20,
		"recent_default_weight": 0.20,
		"recent_window_default_weight": 0.10,
		"cure_weight": 0.08,
		"stability_weight": 0.07,
		"volatility_weight": 0.05,

		"max_cure_hours": 72.0,
		"max_volatility_hours": 48.0,
		"max_recent_default_rate": 1.0,
		"max_lifetime_default_rate": 1.0,

		"min_score": 0.0,
		"max_score": 1.0,
	},
		"prior_exposure": {
		"avg_weight": 0.60,
		"max_weight": 0.55,
		"avg_multiplier": 0.90,
		"max_multiplier": 1.05,
		"ratio_penalty_above_max": 0.85,
		"new_to_credit_factor": 0.30,
		"growth_ratio_floor": 0.75,
		"growth_ratio_ceiling": 1.15,
		"recent_performance_floor": 0.70,
		"recent_performance_ceiling": 1.00,
		"new_to_credit_factor": 0.50,
	},


    "policy": {
        # Risk-tier cutoffs
        "risk_tier_1_score_max": 0.15,
        "risk_tier_2_score_max": 0.35,
        "risk_tier_3_score_max": 0.60,

        # Risk-tier multipliers
        "risk_tier_1_multiplier": 1.00,
        "risk_tier_2_multiplier": 0.85,
        "risk_tier_3_multiplier": 0.65,
        "risk_tier_4_multiplier": 0.40,

        # Proven-good borrower override
        "proven_good_borrower_min_loans": 3.0,
       
       
        
		"proven_good_borrower_min_on_time_rate": 0.90,
		"proven_good_borrower_max_lifetime_default": 0.05,
		"proven_good_borrower_floor_pct_of_combined": 0.85,
        # Active borrower minimum floor
        "active_borrower_min_limit": 500.0,
        "active_borrower_min_activity_amount": 1.0,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_config(config):
    return config if config is not None else DEFAULT_CAP_CONFIG


def _safe_series(df, column, default_value=0.0):
    if column not in df.columns:
        return pd.Series(default_value, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default_value)


def _safe_string_series(df, column, default_value="unknown"):
    if column not in df.columns:
        return pd.Series(default_value, index=df.index, dtype="object")
    return df[column].fillna(default_value).astype(str)


def _clip_series(series_obj, lower, upper):
    return pd.Series(np.clip(series_obj, lower, upper), index=series_obj.index, dtype="float64")


def _round_to_nearest(series_obj, nearest):
    if nearest <= 0:
        return series_obj
    return (series_obj / nearest).round() * nearest


def _validate_tier_config(final_cfg):
    policy_cfg = final_cfg["policy"]
    t1 = policy_cfg["risk_tier_1_score_max"]
    t2 = policy_cfg["risk_tier_2_score_max"]
    t3 = policy_cfg["risk_tier_3_score_max"]
    assert t1 < t2 < t3, "risk tier thresholds must be strictly increasing"


def _fill_missing_caps(df, cfg):
    combo_cfg = cfg["combination"]
    fill_strategy = combo_cfg.get("missing_cap_fill_strategy", "ceiling")
    ceiling_val = cfg["global_ceiling_limit"]
    floor_val = cfg["global_floor_limit"]
    for col_name in combo_cfg.get("missing_cap_columns", []):
        if col_name not in df.columns:
            if fill_strategy == "ceiling":
                df[col_name] = pd.Series(ceiling_val, index=df.index, dtype="float64")
            elif fill_strategy == "floor":
                df[col_name] = pd.Series(floor_val, index=df.index, dtype="float64")
            else:
                df[col_name] = pd.Series(ceiling_val, index=df.index, dtype="float64")
        else:
            numeric_vals = pd.to_numeric(df[col_name], errors="coerce")
            if fill_strategy == "ceiling":
                df[col_name] = numeric_vals.fillna(ceiling_val)
            elif fill_strategy == "floor":
                df[col_name] = numeric_vals.fillna(floor_val)
            else:
                df[col_name] = numeric_vals.fillna(ceiling_val)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 1. RISK CAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_cap(features_df, config=None):
    cfg = _get_config(config)
    risk_cfg = cfg["risk"]
    df = features_df.copy()

    on_time_rate = _clip_series(_safe_series(df, "on_time_repayment_rate"), 0.0, 1.0)
    lifetime_default = _clip_series(_safe_series(df, "lifetime_default_rate"), 0.0, risk_cfg["max_lifetime_default_rate"])
    recent_default = _clip_series(_safe_series(df, "default_rate_last_10_loans"), 0.0, risk_cfg["max_recent_default_rate"])
    window_default = _clip_series(_safe_series(df, "default_rate_last_50_loans"), 0.0, risk_cfg["max_recent_default_rate"])
    cure_speed = _clip_series(_safe_series(df, "avg_cure_time_hours"), 0.0, risk_cfg["max_cure_hours"])
    cure_time_vol = _clip_series(_safe_series(df, "cure_time_volatility"), 0.0, risk_cfg["max_volatility_hours"])
    stability_score = _clip_series(_safe_series(df, "repayment_stability_score"), 0.0, 1.0)

    on_time_score = on_time_rate
    lifetime_def_score = 1.0 - lifetime_default / risk_cfg["max_lifetime_default_rate"]
    recent_def_score = 1.0 - recent_default / risk_cfg["max_recent_default_rate"]
    window_def_score = 1.0 - window_default / risk_cfg["max_recent_default_rate"]
    cure_score = 1.0 - cure_speed / risk_cfg["max_cure_hours"]
    volatility_penalty = 1.0 - _clip_series(cure_time_vol / risk_cfg["max_volatility_hours"], 0.0, 1.0)

    risk_score = (
        on_time_score * risk_cfg["on_time_weight"]
        + lifetime_def_score * risk_cfg["lifetime_default_weight"]
        + recent_def_score * risk_cfg["recent_default_weight"]
        + window_def_score * risk_cfg["recent_window_default_weight"]
        + cure_score * risk_cfg["cure_weight"]
        + stability_score * risk_cfg["stability_weight"]
        + volatility_penalty * risk_cfg["volatility_weight"]
    )
    risk_score = _clip_series(risk_score, risk_cfg["min_score"], risk_cfg["max_score"])
		
    risk_cap = risk_score * risk_cfg["base_limit"]

    exp_cfg = cfg["experience"]

    experience_factor = _clip_series(
        _safe_series(df, "total_loans") / exp_cfg["minimum_total_loans_for_full_trust"],
        exp_cfg["min_experience_factor"],
        1.00,
	)
	
    risk_cap = risk_cap * experience_factor
    risk_cap = _clip_series(risk_cap, cfg["global_floor_limit"], cfg["global_ceiling_limit"])

    df["risk_score"] = risk_score
    df["risk_cap"] = risk_cap
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. CAPACITY CAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_capacity_cap(features_df, config=None):
    cfg = _get_config(config)
    cap_cfg = cfg["capacity"]
    df = features_df.copy()

    avg_balance = (
        0.7 * _safe_series(df, "avg_daily_balance_30d", 0.0)
        + 0.3 * _safe_series(df, "avg_daily_balance_90d", 0.0)
    )
    monthly_revenue = (
        0.7 * _safe_series(df, "avg_monthly_revenue_30d", 0.0)
        + 0.3 * _safe_series(df, "avg_monthly_revenue_90d", 0.0)
    )
    txn_count = (
        0.7 * _safe_series(df, "avg_monthly_txn_count_30d", 0.0)
        + 0.3 * _safe_series(df, "avg_monthly_txn_count_90d", 0.0)
    )
    monthly_payments = (
        0.7 * _safe_series(df, "avg_monthly_payments_30d", 0.0)
        + 0.3 * _safe_series(df, "avg_monthly_payments_90d", 0.0)
    )
    active_customers = (
        0.7 * _safe_series(df, "active_customer_count_30d", 0.0)
        + 0.3 * _safe_series(df, "active_customer_count_90d", 0.0)
    )
    txn_volume = (
        0.7 * _safe_series(df, "avg_monthly_txn_volume_30d", 0.0)
        + 0.3 * _safe_series(df, "avg_monthly_txn_volume_90d", 0.0)
    )

    balance_amount = avg_balance * cap_cfg["balance_multiplier"]
    revenue_amount = monthly_revenue * cap_cfg["revenue_multiplier"]
    txn_amount = txn_count * cap_cfg["txn_multiplier"]
    payments_amount = monthly_payments * cap_cfg["payments_multiplier"]
    customers_amount = active_customers * cap_cfg["customers_multiplier"]
    volume_amount = txn_volume * cap_cfg["volume_multiplier"]

    balance_component = balance_amount * cap_cfg["balance_weight"]
    revenue_component = revenue_amount * cap_cfg["revenue_weight"]
    txn_component = txn_amount * cap_cfg["txn_weight"]
    payments_component = payments_amount * cap_cfg["payments_weight"]
    customers_component = customers_amount * cap_cfg["customers_weight"]
    volume_component = volume_amount * cap_cfg["activity_weight"]

    raw_capacity_amount = (
        balance_component
        + revenue_component
        + txn_component
        + payments_component
        + customers_component
        + volume_component
    )

    ceiling_val = cfg["global_ceiling_limit"]
    scaled_capacity = np.log1p(np.maximum(raw_capacity_amount, 0.0))
    scaled_capacity = scaled_capacity / np.log1p(ceiling_val)
    scaled_capacity = scaled_capacity * ceiling_val
    scaled_capacity = pd.Series(scaled_capacity, index=df.index, dtype="float64")
    scaled_capacity = _clip_series(
        scaled_capacity,
        cfg["global_floor_limit"],
        ceiling_val,
    )

    capacity_score = _clip_series(scaled_capacity / ceiling_val, 0.0, 1.0)
    capacity_cap = scaled_capacity.copy()

    activity_flag = _clip_series(
        _safe_series(df, "operational_activity_flag", 0.0),
        0.0,
        1.0,
    )
    activity_mult = (
        cap_cfg["activity_inactive_floor"]
        + cap_cfg["activity_active_weight"] * activity_flag
    )
    capacity_cap = capacity_cap * activity_mult
    capacity_cap = _clip_series(
        capacity_cap,
        cfg["global_floor_limit"],
        ceiling_val,
    )

    df["capacity_balance_component"] = pd.Series(
        balance_component,
        index=df.index,
        dtype="float64",
    )
    df["capacity_revenue_component"] = pd.Series(
        revenue_component,
        index=df.index,
        dtype="float64",
    )
    df["capacity_txn_component"] = pd.Series(
        txn_component,
        index=df.index,
        dtype="float64",
    )
    df["capacity_payments_component"] = pd.Series(
        payments_component,
        index=df.index,
        dtype="float64",
    )
    df["capacity_customers_component"] = pd.Series(
        customers_component,
        index=df.index,
        dtype="float64",
    )
    df["capacity_volume_component"] = pd.Series(
        volume_component,
        index=df.index,
        dtype="float64",
    )

    df["capacity_top_driver"] = df[
        [
            "capacity_balance_component",
            "capacity_revenue_component",
            "capacity_txn_component",
            "capacity_payments_component",
            "capacity_customers_component",
            "capacity_volume_component",
        ]
    ].idxmax(axis=1).str.replace("capacity_", "", regex=False).str.replace("_component", "", regex=False)

    df["capacity_score"] = capacity_score
    df["capacity_cap"] = capacity_cap

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. RECENT USAGE CAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_recent_usage_cap(features_df, config=None):
    cfg = _get_config(config)
    usage_cfg = cfg["recent_usage"]
    df = features_df.copy()

    recent_disbursement_1m_raw = _safe_series(df, "recent_disbursement_amount_1m", 0.0)
    recent_repayment_1m_raw = _safe_series(df, "recent_repayment_amount_1m", 0.0)

    recent_disbursement_3m = _safe_series(df, "recent_disbursement_amount_3m", 0.0)
    recent_repayment_3m = _safe_series(df, "recent_repayment_amount_3m", 0.0)

    recent_disbursement_amount = (
        0.7 * recent_disbursement_1m_raw
        + 0.3 * recent_disbursement_3m
    )
    recent_repayment_amount = (
        0.7 * recent_repayment_1m_raw
        + 0.3 * recent_repayment_3m
    )

    recent_coverage_1m = _clip_series(
        _safe_series(df, "recent_repayment_coverage_1m", 0.0),
        0.0,
        1.0,
    )
    penalty_events = _safe_series(df, "recent_penalty_events_1m", 0.0)

    active_mask = (
        recent_disbursement_amount + recent_repayment_amount
    ) >= usage_cfg["minimum_activity_required"]

    weight_denom = (
        usage_cfg["disbursement_weight"] + usage_cfg["repayment_weight"]
    )

    disbursement_component = (
        recent_disbursement_amount
        * usage_cfg["disbursement_multiplier"]
        * usage_cfg["disbursement_weight"]
    ) / weight_denom

    repayment_component = (
        recent_repayment_amount
        * usage_cfg["repayment_multiplier"]
        * usage_cfg["repayment_weight"]
    ) / weight_denom

    usage_base_amount = disbursement_component + repayment_component

    coverage_component = (
        usage_cfg["coverage_multiplier_floor"]
        + recent_coverage_1m * usage_cfg["coverage_weight"]
    )
    coverage_mult = _clip_series(
        coverage_component,
        usage_cfg["coverage_multiplier_floor"],
        usage_cfg["coverage_bonus_cap"],
    )

    penalty_mult = _clip_series(
        1.0 - penalty_events * usage_cfg["penalty_haircut_per_event"],
        0.0,
        1.0,
    )

    recent_usage_cap_before_ratio = usage_base_amount * coverage_mult * penalty_mult

    repayment_ratio = _clip_series(
        recent_repayment_amount / np.maximum(recent_disbursement_amount, 1.0),
        0.0,
        usage_cfg["repayment_ratio_cap"],
    )
    floor_weight = usage_cfg["repayment_ratio_floor_weight"]
    repayment_ratio_mult = (
        floor_weight
        + (1.0 - floor_weight)
        * (repayment_ratio / usage_cfg["repayment_ratio_cap"])
    )

    recent_usage_cap_pre_activity = (
        recent_usage_cap_before_ratio * repayment_ratio_mult
    )

    recent_usage_cap = pd.Series(
        np.where(active_mask, recent_usage_cap_pre_activity, 0.0),
        index=df.index,
        dtype="float64",
    )
    recent_usage_cap = _clip_series(
        recent_usage_cap,
        cfg["global_floor_limit"],
        cfg["global_ceiling_limit"],
    )

    component_frame = pd.DataFrame(
        {
            "recent_usage_disbursement_component": disbursement_component,
            "recent_usage_repayment_component": repayment_component,
        },
        index=df.index,
    )

    recent_usage_top_driver = (
        component_frame.idxmax(axis=1)
        .str.replace("recent_usage_", "", regex=False)
        .str.replace("_component", "", regex=False)
    )

    recent_usage_reason = pd.Series(
        np.where(active_mask, "active_recent_usage_policy", "inactive_recent_usage_gate"),
        index=df.index,
        dtype="object",
    )

    recent_usage_reason = pd.Series(
        np.where(
            active_mask & (coverage_mult > usage_cfg["coverage_multiplier_floor"]),
            "coverage_bonus_applied",
            recent_usage_reason,
        ),
        index=df.index,
        dtype="object",
    )

    recent_usage_reason = pd.Series(
        np.where(
            active_mask & (repayment_ratio_mult < 1.0),
            "repayment_ratio_haircut_applied",
            recent_usage_reason,
        ),
        index=df.index,
        dtype="object",
    )

    recent_usage_reason = pd.Series(
        np.where(
            active_mask & (penalty_mult < 1.0),
            "penalty_event_haircut_applied",
            recent_usage_reason,
        ),
        index=df.index,
        dtype="object",
    )

    df["recent_usage_disbursement_component"] = pd.Series(
        disbursement_component,
        index=df.index,
        dtype="float64",
    )
    df["recent_usage_repayment_component"] = pd.Series(
        repayment_component,
        index=df.index,
        dtype="float64",
    )
    df["recent_usage_coverage_multiplier"] = coverage_mult
    df["recent_usage_penalty_multiplier"] = penalty_mult
    df["recent_usage_repayment_ratio"] = repayment_ratio
    df["recent_usage_repayment_ratio_multiplier"] = repayment_ratio_mult
    df["recent_usage_active_flag"] = active_mask.astype(int)
    df["recent_usage_top_driver"] = recent_usage_top_driver
    df["recent_usage_reason"] = recent_usage_reason
    df["recent_usage_cap"] = recent_usage_cap

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4. PRIOR EXPOSURE CAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_prior_exposure_cap(features_df, config=None):
    cfg = _get_config(config)
    pri_cfg = cfg["prior_exposure"]
    df = features_df.copy()

    avg_prior_loan_size = _safe_series(df, "avg_prior_loan_size", 0.0)
    max_prior_loan_size = _safe_series(df, "max_prior_loan_size", 0.0)
    current_loan_size = _safe_series(df, "current_loan_size", 0.0)

    recent_performance = _clip_series(
        _safe_series(df, "recent_repayment_performance", 1.0),
        pri_cfg["recent_performance_floor"],
        pri_cfg["recent_performance_ceiling"],
    )

    is_new_to_credit = (avg_prior_loan_size <= 0.0) & (max_prior_loan_size <= 0.0)

    avg_component = (
        avg_prior_loan_size
        * pri_cfg["avg_multiplier"]
        * pri_cfg["avg_weight"]
    )
    max_component = (
        max_prior_loan_size
        * pri_cfg["max_multiplier"]
        * pri_cfg["max_weight"]
    )

    prior_exposure_base = avg_component + max_component

    above_max_mask = current_loan_size > max_prior_loan_size
    above_max_penalty_multiplier = pd.Series(
        np.where(
            above_max_mask,
            pri_cfg["ratio_penalty_above_max"],
            1.0,
        ),
        index=df.index,
        dtype="float64",
    )

    growth_penalty_multiplier = pd.Series(
        np.where(
            avg_prior_loan_size > 0.0,
            np.clip(
                max_prior_loan_size / np.maximum(avg_prior_loan_size, 1.0),
                pri_cfg["growth_ratio_floor"],
                pri_cfg["growth_ratio_ceiling"],
            ),
            1.0,
        ),
        index=df.index,
        dtype="float64",
    )

    prior_exposure_cap_existing = (
        prior_exposure_base
        * above_max_penalty_multiplier
        * growth_penalty_multiplier
        * recent_performance
    )

    new_to_credit_cap = current_loan_size * pri_cfg["new_to_credit_factor"]

    prior_exposure_cap = pd.Series(
        np.where(
            is_new_to_credit,
            new_to_credit_cap,
            prior_exposure_cap_existing,
        ),
        index=df.index,
        dtype="float64",
    )

    prior_exposure_cap = _clip_series(
        prior_exposure_cap,
        cfg["global_floor_limit"],
        cfg["global_ceiling_limit"],
    )

    component_frame = pd.DataFrame(
        {
            "prior_exposure_avg_component": pd.Series(
                avg_component,
                index=df.index,
                dtype="float64",
            ),
            "prior_exposure_max_component": pd.Series(
                max_component,
                index=df.index,
                dtype="float64",
            ),
            "prior_exposure_new_to_credit_component": pd.Series(
                np.where(is_new_to_credit, new_to_credit_cap, 0.0),
                index=df.index,
                dtype="float64",
            ),
        },
        index=df.index,
    )

    prior_exposure_top_driver = (
        component_frame.idxmax(axis=1)
        .str.replace("prior_exposure_", "", regex=False)
        .str.replace("_component", "", regex=False)
    )

    prior_exposure_reason = pd.Series(
        np.where(
            is_new_to_credit,
            "new_to_credit_proxy_cap",
            "existing_exposure_policy",
        ),
        index=df.index,
        dtype="object",
    )

    prior_exposure_reason = pd.Series(
        np.where(
            (~is_new_to_credit) & above_max_mask,
            "above_prior_max_penalty_applied",
            prior_exposure_reason,
        ),
        index=df.index,
        dtype="object",
    )

    prior_exposure_reason = pd.Series(
        np.where(
            (~is_new_to_credit) & (recent_performance < pri_cfg["recent_performance_ceiling"]),
            "recent_performance_haircut_applied",
            prior_exposure_reason,
        ),
        index=df.index,
        dtype="object",
    )

    df["prior_exposure_avg_component"] = pd.Series(
        avg_component,
        index=df.index,
        dtype="float64",
    )
    df["prior_exposure_max_component"] = pd.Series(
        max_component,
        index=df.index,
        dtype="float64",
    )
    df["prior_exposure_new_to_credit_component"] = pd.Series(
        np.where(is_new_to_credit, new_to_credit_cap, 0.0),
        index=df.index,
        dtype="float64",
    )
    df["prior_exposure_above_max_penalty_multiplier"] = above_max_penalty_multiplier
    df["prior_exposure_growth_penalty_multiplier"] = growth_penalty_multiplier
    df["prior_exposure_recent_performance_multiplier"] = recent_performance
    df["prior_exposure_existing_cap_before_new_to_credit_override"] = pd.Series(
        prior_exposure_cap_existing,
        index=df.index,
        dtype="float64",
    )
    df["prior_exposure_top_driver"] = prior_exposure_top_driver
    df["prior_exposure_reason"] = prior_exposure_reason
    df["prior_exposure_cap"] = prior_exposure_cap

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 5. COMBINE CAPS
# ─────────────────────────────────────────────────────────────────────────────

def combine_caps(features_df, config=None):
    cfg = _get_config(config)
    combo_cfg = cfg["combination"]
    df = features_df.copy()

    capacity_cap = _clip_series(
        _safe_series(df, "capacity_cap", 0.0),
        cfg["global_floor_limit"],
        cfg["global_ceiling_limit"],
    )
    recent_usage_cap = _clip_series(
        _safe_series(df, "recent_usage_cap", 0.0),
        cfg["global_floor_limit"],
        cfg["global_ceiling_limit"],
    )
    prior_exposure_cap = _clip_series(
        _safe_series(df, "prior_exposure_cap", 0.0),
        cfg["global_floor_limit"],
        cfg["global_ceiling_limit"],
    )
    risk_cap = _clip_series(
        _safe_series(df, "risk_cap", 0.0),
        cfg["global_floor_limit"],
        cfg["global_ceiling_limit"],
    )

    prior_limit = _clip_series(
        _safe_series(df, "prior_limit", 0.0),
        cfg["global_floor_limit"],
        cfg["global_ceiling_limit"],
    )

    thin_file_flag = _clip_series(
        _safe_series(df, "is_thin_file", 0.0),
        0.0,
        1.0,
    )

    capacity_weight_standard = combo_cfg.get("capacity_weight", 0.45)
    usage_weight_standard = combo_cfg.get("recent_usage_weight", 0.25)
    prior_exposure_weight_standard = combo_cfg.get("prior_exposure_weight", 0.10)
    risk_weight_standard = combo_cfg.get("risk_weight", 0.20)

    capacity_weight_thin = combo_cfg.get(
        "thin_file_capacity_weight",
        capacity_weight_standard,
    )
    usage_weight_thin = combo_cfg.get(
        "thin_file_recent_usage_weight",
        usage_weight_standard,
    )
    prior_exposure_weight_thin = combo_cfg.get(
        "thin_file_prior_exposure_weight",
        prior_exposure_weight_standard,
    )
    risk_weight_thin = combo_cfg.get(
        "thin_file_risk_weight",
        risk_weight_standard,
    )

    capacity_weight = pd.Series(
        np.where(thin_file_flag > 0.0, capacity_weight_thin, capacity_weight_standard),
        index=df.index,
        dtype="float64",
    )
    usage_weight = pd.Series(
        np.where(thin_file_flag > 0.0, usage_weight_thin, usage_weight_standard),
        index=df.index,
        dtype="float64",
    )
    prior_exposure_weight = pd.Series(
        np.where(
            thin_file_flag > 0.0,
            prior_exposure_weight_thin,
            prior_exposure_weight_standard,
        ),
        index=df.index,
        dtype="float64",
    )
    risk_weight = pd.Series(
        np.where(thin_file_flag > 0.0, risk_weight_thin, risk_weight_standard),
        index=df.index,
        dtype="float64",
    )

    weight_sum = (
        capacity_weight
        + usage_weight
        + prior_exposure_weight
        + risk_weight
    )
    safe_weight_sum = pd.Series(
        np.where(weight_sum > 0.0, weight_sum, 1.0),
        index=df.index,
        dtype="float64",
    )

    capacity_component = capacity_cap * capacity_weight / safe_weight_sum
    recent_usage_component = recent_usage_cap * usage_weight / safe_weight_sum
    prior_exposure_component = (
        prior_exposure_cap * prior_exposure_weight / safe_weight_sum
    )
    risk_component = risk_cap * risk_weight / safe_weight_sum

    weighted_combined_cap = (
        capacity_component
        + recent_usage_component
        + prior_exposure_component
        + risk_component
    )

    combined_cap_before_risk_guardrail = pd.Series(
        weighted_combined_cap,
        index=df.index,
        dtype="float64",
    )

    risk_cap_binding = pd.Series(
        risk_cap < weighted_combined_cap,
        index=df.index,
        dtype="bool",
    )

    combined_cap_after_risk_guardrail = pd.Series(
        np.minimum(weighted_combined_cap, risk_cap),
        index=df.index,
        dtype="float64",
    )

    prior_limit_weight = combo_cfg.get("prior_limit_weight", 0.15)
    prior_limit_available = prior_limit > 0.0

    smoothed_cap_raw = pd.Series(
        np.where(
            prior_limit_available,
            (1.0 - prior_limit_weight) * combined_cap_after_risk_guardrail
            + prior_limit_weight * prior_limit,
            combined_cap_after_risk_guardrail,
        ),
        index=df.index,
        dtype="float64",
    )

    prior_limit_max_upside = combo_cfg.get("prior_limit_max_upside", 1.25)
    prior_limit_max_downside = combo_cfg.get("prior_limit_max_downside", 0.75)

    smoothed_cap = pd.Series(
        np.where(
            prior_limit_available,
            np.minimum(
                np.maximum(
                    smoothed_cap_raw,
                    prior_limit * prior_limit_max_downside,
                ),
                prior_limit * prior_limit_max_upside,
            ),
            smoothed_cap_raw,
        ),
        index=df.index,
        dtype="float64",
    )

    combined_cap_before_smoothing = combined_cap_after_risk_guardrail.copy()
    prior_limit_smoothing_applied = pd.Series(
        prior_limit_available
        & (np.abs(smoothed_cap - combined_cap_after_risk_guardrail) > 1e-9),
        index=df.index,
        dtype="bool",
    )

    combined_cap = smoothed_cap.copy()

    active_user_floor = _safe_series(df, "active_borrower_floor_limit", 0.0)
    active_floor_applied = _clip_series(
        _safe_series(df, "active_floor_applied", 0.0),
        0.0,
        1.0,
    ) > 0.0

    policy_floor_limit = _safe_series(df, "policy_floor_limit", 0.0)
    policy_floor_applied = _clip_series(
        _safe_series(df, "policy_floor_applied", 0.0),
        0.0,
        1.0,
    ) > 0.0

    active_floor_preserved = active_floor_applied & (active_user_floor > combined_cap)
    combined_cap = pd.Series(
        np.where(active_floor_preserved, active_user_floor, combined_cap),
        index=df.index,
        dtype="float64",
    )

    policy_floor_preserved = policy_floor_applied & (policy_floor_limit > combined_cap)
    combined_cap = pd.Series(
        np.where(policy_floor_preserved, policy_floor_limit, combined_cap),
        index=df.index,
        dtype="float64",
    )

    combined_cap = _clip_series(
        combined_cap,
        cfg["global_floor_limit"],
        cfg["global_ceiling_limit"],
    )

    component_frame = pd.DataFrame(
        {
            "capacity_component": pd.Series(
                capacity_component,
                index=df.index,
                dtype="float64",
            ),
            "recent_usage_component": pd.Series(
                recent_usage_component,
                index=df.index,
                dtype="float64",
            ),
            "prior_exposure_component": pd.Series(
                prior_exposure_component,
                index=df.index,
                dtype="float64",
            ),
            "risk_component": pd.Series(
                risk_component,
                index=df.index,
                dtype="float64",
            ),
        }
    )

    combined_top_driver = component_frame.idxmax(axis=1).astype("object")

    combined_reason = pd.Series(
        np.where(
            thin_file_flag > 0.0,
            "thin_file_weighting_applied",
            "standard_weighting_applied",
        ),
        index=df.index,
        dtype="object",
    )

    combined_reason = pd.Series(
        np.where(
            prior_limit_smoothing_applied,
            "prior_limit_smoothing_applied",
            combined_reason,
        ),
        index=df.index,
        dtype="object",
    )

    combined_reason = pd.Series(
        np.where(
            risk_cap_binding,
            "risk_cap_guardrail_binding",
            combined_reason,
        ),
        index=df.index,
        dtype="object",
    )

    combined_reason = pd.Series(
        np.where(
            active_floor_preserved,
            "active_user_floor_preserved",
            combined_reason,
        ),
        index=df.index,
        dtype="object",
    )

    combined_reason = pd.Series(
        np.where(
            policy_floor_preserved,
            "proven_good_policy_floor_preserved",
            combined_reason,
        ),
        index=df.index,
        dtype="object",
    )

    df["capacity_component"] = pd.Series(
        capacity_component,
        index=df.index,
        dtype="float64",
    )
    df["recent_usage_component"] = pd.Series(
        recent_usage_component,
        index=df.index,
        dtype="float64",
    )
    df["prior_exposure_component"] = pd.Series(
        prior_exposure_component,
        index=df.index,
        dtype="float64",
    )
    df["risk_component"] = pd.Series(
        risk_component,
        index=df.index,
        dtype="float64",
    )

    df["combined_cap_before_risk_guardrail"] = pd.Series(
        combined_cap_before_risk_guardrail,
        index=df.index,
        dtype="float64",
    )
    df["risk_cap_binding"] = risk_cap_binding.astype(int)

    df["combined_cap_after_risk_guardrail"] = pd.Series(
        combined_cap_after_risk_guardrail,
        index=df.index,
        dtype="float64",
    )

    df["combined_cap_before_smoothing"] = pd.Series(
        combined_cap_before_smoothing,
        index=df.index,
        dtype="float64",
    )
    df["prior_limit_smoothing_applied"] = prior_limit_smoothing_applied.astype(int)

    df["active_floor_preserved"] = active_floor_preserved.astype(int)
    df["policy_floor_preserved"] = policy_floor_preserved.astype(int)

    df["combined_top_driver"] = combined_top_driver
    df["combined_reason"] = combined_reason
    df["combined_cap"] = combined_cap

    return df
# ─────────────────────────────────────────────────────────────────────────────
# 6. POLICY ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────

def apply_policy_adjustments(features_df, config=None):
    cfg = _get_config(config)
    policy_cfg = cfg["policy"]
    _validate_tier_config(cfg)

    df = features_df.copy()

    risk_score = _clip_series(_safe_series(df, "risk_score", 0.0), 0.0, 1.0)
    combined_cap = _clip_series(
    _safe_series(df, "combined_cap", 0.0),
    cfg["global_floor_limit"],
    cfg["global_ceiling_limit"],
    )
    
    tier_1_mask = risk_score <= policy_cfg["risk_tier_1_score_max"]
    tier_2_mask = (
        (risk_score > policy_cfg["risk_tier_1_score_max"])
        & (risk_score <= policy_cfg["risk_tier_2_score_max"])
    )
    tier_3_mask = (
        (risk_score > policy_cfg["risk_tier_2_score_max"])
        & (risk_score <= policy_cfg["risk_tier_3_score_max"])
    )
    tier_4_mask = risk_score > policy_cfg["risk_tier_3_score_max"]
    
    tier_multiplier = pd.Series(
    policy_cfg["risk_tier_4_multiplier"],
    index=df.index,
    dtype="float64",
    )
    tier_multiplier = pd.Series(
        np.where(tier_1_mask, policy_cfg["risk_tier_1_multiplier"], tier_multiplier),
        index=df.index,
        dtype="float64",
    )
    tier_multiplier = pd.Series(
        np.where(tier_2_mask, policy_cfg["risk_tier_2_multiplier"], tier_multiplier),
        index=df.index,
        dtype="float64",
    )
    tier_multiplier = pd.Series(
        np.where(tier_3_mask, policy_cfg["risk_tier_3_multiplier"], tier_multiplier),
        index=df.index,
        dtype="float64",
    )
    tier_multiplier = pd.Series(
        np.where(tier_4_mask, policy_cfg["risk_tier_4_multiplier"], tier_multiplier),
        index=df.index,
        dtype="float64",
    )

    risk_tier = pd.Series("tier_4", index=df.index, dtype="object")
    risk_tier = pd.Series(
        np.where(tier_1_mask, "tier_1", risk_tier),
        index=df.index,
        dtype="object",
    )
    risk_tier = pd.Series(
        np.where(tier_2_mask, "tier_2", risk_tier),
        index=df.index,
        dtype="object",
    )
    risk_tier = pd.Series(
        np.where(tier_3_mask, "tier_3", risk_tier),
        index=df.index,
        dtype="object",
    )
    risk_tier = pd.Series(
        np.where(tier_4_mask, "tier_4", risk_tier),
        index=df.index,
        dtype="object",
    )

    raw_policy_cap = combined_cap * tier_multiplier

    total_loans = _safe_series(df, "total_loans", 0.0)
    on_time_rate = _clip_series(
        _safe_series(df, "on_time_repayment_rate", 0.0),
        0.0,
        1.0,
    )
    lifetime_default = _clip_series(
        _safe_series(df, "lifetime_default_rate", 1.0),
        0.0,
        1.0,
    )

    proven_good_mask = (
        (total_loans >= policy_cfg["proven_good_borrower_min_loans"])
        & (on_time_rate >= policy_cfg["proven_good_borrower_min_on_time_rate"])
        & (lifetime_default <= policy_cfg["proven_good_borrower_max_lifetime_default"])
    )

    proven_good_floor = (
        combined_cap * policy_cfg["proven_good_borrower_floor_pct_of_combined"]
    )

    policy_cap = pd.Series(
        np.where(
            proven_good_mask,
            np.maximum(raw_policy_cap, proven_good_floor),
            raw_policy_cap,
        ),
        index=df.index,
        dtype="float64",
    )

    policy_floor_applied = proven_good_mask & (proven_good_floor > raw_policy_cap)

    policy_reason = pd.Series("tier_4_policy", index=df.index, dtype="object")
    policy_reason = pd.Series(
        np.where(tier_1_mask, "tier_1_policy", policy_reason),
        index=df.index,
        dtype="object",
    )
    policy_reason = pd.Series(
        np.where(tier_2_mask, "tier_2_policy", policy_reason),
        index=df.index,
        dtype="object",
    )
    policy_reason = pd.Series(
        np.where(tier_3_mask, "tier_3_policy", policy_reason),
        index=df.index,
        dtype="object",
    )
    policy_reason = pd.Series(
        np.where(tier_4_mask, "tier_4_policy", policy_reason),
        index=df.index,
        dtype="object",
    )
    policy_reason = pd.Series(
        np.where(policy_floor_applied, "proven_good_floor_override", policy_reason),
        index=df.index,
        dtype="object",
    )

    policy_cap_before_active_floor = policy_cap.copy()

    recent_activity = (
        _safe_series(df, "recent_disbursement_amount_1m", 0.0)
        + _safe_series(df, "recent_repayment_amount_1m", 0.0)
    )

    active_floor = policy_cfg.get("active_borrower_min_limit", 500.0)
    active_floor_min_activity = policy_cfg.get(
        "active_borrower_min_activity_amount",
        1.0,
    )

    active_floor_eligible = recent_activity >= active_floor_min_activity

    policy_cap = pd.Series(
        np.where(
            active_floor_eligible,
            np.maximum(policy_cap, active_floor),
            policy_cap,
        ),
        index=df.index,
        dtype="float64",
    )

    active_floor_applied = active_floor_eligible & (
        policy_cap > policy_cap_before_active_floor
    )

    policy_reason = pd.Series(
        np.where(
            active_floor_applied,
            "active_borrower_min_floor_override",
            policy_reason,
        ),
        index=df.index,
        dtype="object",
    )

    policy_cap = _clip_series(
        policy_cap,
        cfg["global_floor_limit"],
        cfg["global_ceiling_limit"],
    )

    final_cap = _round_to_nearest(policy_cap, cfg["rounding"]["round_to_nearest"])
    final_cap = _clip_series(
        final_cap,
        cfg["global_floor_limit"],
        cfg["global_ceiling_limit"],
    )

    df["risk_tier"] = risk_tier
    df["policy_multiplier"] = tier_multiplier
    df["is_proven_good_borrower"] = proven_good_mask.astype(int)
    df["proven_good_floor"] = proven_good_floor
    df["policy_floor_applied"] = policy_floor_applied.astype(int)
    df["active_floor_eligible"] = active_floor_eligible.astype(int)
    df["active_floor_applied"] = active_floor_applied.astype(int)
    df["policy_reason"] = policy_reason
    df["policy_cap"] = policy_cap
    df["final_limit"] = final_cap

    return df
# ─────────────────────────────────────────────────────────────────────────────
# 7. ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_limit_caps(features_df, config=None):
    df = compute_risk_cap(features_df, config=config)
    df = compute_capacity_cap(df, config=config)
    df = compute_recent_usage_cap(df, config=config)
    df = compute_prior_exposure_cap(df, config=config)
    df = combine_caps(df, config=config)
    df = apply_policy_adjustments(df, config=config)
    return df

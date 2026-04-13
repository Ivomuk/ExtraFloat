"""
extrafloat_limit_engine_features.py
=====================================
Feature engineering pipeline for the ExtraFloat credit limit engine.
 
Prepares three input data sources and merges them into a single flat
feature DataFrame ready for ``run_extrafloat_limit_engine()``.
 
Market: Uganda (UG) — Bank of Uganda supervised mobile money.
"""
 
from __future__ import annotations
 
import logging
 
import numpy as np
import pandas as pd
 
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────────────────────────────────────
# UGANDA PEAK-SEASON MONTHS  (East Africa / Bank of Uganda context)
# Jan = school fees, Aug/Sep = harvest + back-to-school, Dec = Christmas
# ─────────────────────────────────────────────────────────────────────────────
PEAK_SEASON_MONTHS: frozenset[int] = frozenset({1, 8, 9, 12})
 
# ─────────────────────────────────────────────────────────────────────────────
# AGENT TIER CEILING MULTIPLIERS
# Matched case-insensitively against agent_profile (str.lower()).
# multiplier = tier_limit / global_ceiling_limit (1,000,000 UGX)
# Order matters: "silver class" must precede "silver" so the substring
# match catches "silver class" agents before the shorter key fires.
# Applied to global_ceiling_limit in the caps engine to derive the
# per-agent effective ceiling in compute_capacity_cap() and
# apply_policy_adjustments().
# ─────────────────────────────────────────────────────────────────────────────
AGENT_TIER_CEILING_MULTIPLIERS: dict[str, float] = {
    "diamond":      1.00,   # 1,000,000 / 1,000,000
    "titanium":     0.75,   #   750,000 / 1,000,000
    "platinum":     0.50,   #   500,000 / 1,000,000
    "gold":         0.35,   #   350,000 / 1,000,000
    "silver class": 0.25,   #   250,000 / 1,000,000  ← must precede "silver"
    "silver":       0.25,   #   250,000 / 1,000,000
    "new bronze":   0.05,   #    50,000 / 1,000,000  ← must precede "bronze"
    "bronze":       0.10,   #   100,000 / 1,000,000
    "unknown":      0.05,   # conservative fallback
}
 
# ─────────────────────────────────────────────────────────────────────────────
# THIN-FILE THRESHOLD
# Borrowers with total_loans below this value are flagged as thin-file,
# which shifts the caps engine toward higher risk weighting.
# ─────────────────────────────────────────────────────────────────────────────
THIN_FILE_LOAN_THRESHOLD: int = 3
 
# ─────────────────────────────────────────────────────────────────────────────
# STABILITY NORMALISATION
# stability_proxy (on_time_streak - default_streak) is unbounded.
# Clip to [0, STABILITY_NORM_UPPER] then divide to produce a [0, 1] score.
# ─────────────────────────────────────────────────────────────────────────────
STABILITY_NORM_UPPER: float = 10.0
 
 
# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED COLUMN SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
 
BORROWER_LIMIT_REQUIRED_COLUMNS: list[str] = [
    "msisdn",
    "total_loans",
    "first_loan_ts",
    "latest_loan_ts",
    "total_disbursed_amount",
    "avg_loan_size_lifetime",
    "max_loan_size_lifetime",
    "lifetime_on_time_24h_rate",
    "lifetime_on_time_26h_rate",
    "lifetime_default_24h_rate",
    "lifetime_default_26h_rate",
    "lifetime_severe_default_48h_rate",
    "lifetime_zero_recovery_rate",
    "lifetime_avg_hours_to_principal_cure",
    "lifetime_worst_hours_to_principal_cure",
    "lifetime_cure_time_volatility",
    "latest_requestid",
    "latest_disbursement_ts",
    "latest_disbursed_amount",
    "num_prior_loans",
    "prior_on_time_24h_rate",
    "avg_prior_hours_to_cure",
    "worst_prior_hours_to_cure",
    "cure_time_volatility",
    "recent_3_on_time_rate",
    "recent_3_avg_cure_time",
    "lifetime_prior_default_24h_rate",
    "recent_5_default_24h_rate",
    "cure_time_trend",
    "borrower_trend",
    "borrower_profile_type",
    "loans_last_50_loans",
    "defaults_last_50_loans",
    "default_rate_last_50_loans",
    "prior_on_time_streak",
    "prior_default_streak",
    "avg_prior_loan_size",
    "max_prior_loan_size",
    "loan_size_vs_avg_ratio",
    "loan_size_vs_max_ratio",
    "loan_above_prior_max_flag",
]
 
# Full multi-horizon schema derived from transaction_capacity_features_sample.txt
# agent_msisdn is automatically renamed to msisdn in prepare_transaction_capacity_features.
TRANSACTION_CAPACITY_REQUIRED_COLUMNS: list[str] = [
    "msisdn",           # or agent_msisdn — renamed in prepare step
    "snapshot_dt",
    "agent_profile",
    "account_balance",
    "average_balance",
    "commission",
    # Cash-out: volume, value, customers, commissions — 3 horizons
    "cash_out_vol_1m",   "cash_out_vol_3m",   "cash_out_vol_6m",
    "cash_out_value_1m", "cash_out_value_3m", "cash_out_value_6m",
    "cash_out_cust_1m",  "cash_out_cust_3m",  "cash_out_cust_6m",
    "cash_out_comm_1m",  "cash_out_comm_3m",  "cash_out_comm_6m",
    # Cash-in: volume, value, customers, commissions — 3 horizons
    "cash_in_vol_1m",    "cash_in_vol_3m",    "cash_in_vol_6m",
    "cash_in_value_1m",  "cash_in_value_3m",  "cash_in_value_6m",
    "cash_in_cust_1m",   "cash_in_cust_3m",   "cash_in_cust_6m",
    "cash_in_comm_1m",   "cash_in_comm_3m",   "cash_in_comm_6m",
    # Payment: volume, value, customers, commissions — 3 horizons
    "payment_vol_1m",    "payment_vol_3m",    "payment_vol_6m",
    "payment_value_1m",  "payment_value_3m",  "payment_value_6m",
    "payment_cust_1m",   "payment_cust_3m",   "payment_cust_6m",
    "payment_comm_1m",   "payment_comm_3m",   "payment_comm_6m",
    # Aggregate totals — 3 horizons
    "cust_1m", "cust_3m", "cust_6m",
    "vol_1m",  "vol_3m",  "vol_6m",
]
 
LOAN_SUMMARY_REQUIRED_COLUMNS: list[str] = [
    "msisdn",
    "snapshot_dt",
    "last_disbursement_date",
    "last_repayment_date",
    "disbursement_vol_1m",
    "disbursement_val_1m",
    "repayment_vol_1m",
    "repayment_val_1m",
    "penalties_1m",
    # 3m variants for temporal blending in compute_recent_usage_cap()
    "disbursement_val_3m",
    "repayment_val_3m",
    "penalties_3m",
]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def _check_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    df_name: str,
) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {', '.join(missing)}"
        )
 
 
def _standardize_msisdn(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.replace(".0", "", regex=False)
    cleaned = cleaned.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    return cleaned
 
 
def _safe_divide(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
) -> pd.Series:
    num = pd.Series(numerator)
    den = pd.Series(denominator)
    result = np.where((den.notna()) & (den != 0), num / den, np.nan)
    return pd.Series(result, index=num.index)
 
 
def _coerce_numeric(
    df: pd.DataFrame,
    column_names: list[str],
    fill_value: float | None = None,
) -> pd.DataFrame:
    for col in column_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if fill_value is not None:
                df[col] = df[col].fillna(fill_value)
    return df
 
 
def _coerce_datetime(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    for col in column_names:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
 
 
def _clip_lower_zero(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    for col in column_names:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    return df
 
 
def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Return a numeric column by name, or a zero-filled Series if absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. BORROWER LIMIT FEATURES
# ─────────────────────────────────────────────────────────────────────────────
 
def prepare_borrower_limit_features(
    borrower_limit_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean, validate, and engineer features from borrower credit history.
 
    KYC note
    --------
    If ``kyc_verified_flag`` is present (1 = verified, 0 = unverified),
    unverified borrowers are flagged via ``is_kyc_blocked = 1`` and will
    receive ``assigned_limit = 0`` in the caps engine.  If the column is
    absent, all borrowers are treated as verified (flag defaults to 1).
    """
    logger.info(
        "prepare_borrower_limit_features: input rows = %d", len(borrower_limit_df)
    )
 
    _check_required_columns(
        borrower_limit_df, BORROWER_LIMIT_REQUIRED_COLUMNS, "borrower_limit_features"
    )
 
    df = borrower_limit_df.copy()
 
    # ── Column standardisation ──
    if "phonenumber" in df.columns and "msisdn" not in df.columns:
        df = df.rename(columns={"phonenumber": "msisdn"})
    df["msisdn"] = _standardize_msisdn(df["msisdn"])
 
    # ── KYC verification (optional column, defaults to verified) ──
    if "kyc_verified_flag" in df.columns:
        df["kyc_verified_flag"] = (
            pd.to_numeric(df["kyc_verified_flag"], errors="coerce")
            .fillna(0)
            .clip(0, 1)
            .astype(int)
        )
        n_unverified = int((df["kyc_verified_flag"] == 0).sum())
        if n_unverified > 0:
            logger.warning(
                "prepare_borrower_limit_features: %d borrowers have "
                "kyc_verified_flag=0 — these will be blocked (assigned_limit=0).",
                n_unverified,
            )
    else:
        logger.info(
            "kyc_verified_flag column absent — all borrowers treated as verified."
        )
        df["kyc_verified_flag"] = 1
 
    df["is_kyc_blocked"] = (df["kyc_verified_flag"] == 0).astype(int)
 
    # ── Datetime coercion ──
    df = _coerce_datetime(
        df, ["first_loan_ts", "latest_loan_ts", "latest_disbursement_ts"]
    )
 
    # ── Numeric coercion ──
    numeric_cols = [
        "total_loans", "total_disbursed_amount", "avg_loan_size_lifetime",
        "max_loan_size_lifetime", "lifetime_on_time_24h_rate",
        "lifetime_on_time_26h_rate", "lifetime_default_24h_rate",
        "lifetime_default_26h_rate", "lifetime_severe_default_48h_rate",
        "lifetime_zero_recovery_rate", "lifetime_avg_hours_to_principal_cure",
        "lifetime_worst_hours_to_principal_cure", "lifetime_cure_time_volatility",
        "latest_disbursed_amount", "num_prior_loans", "prior_on_time_24h_rate",
        "avg_prior_hours_to_cure", "worst_prior_hours_to_cure",
        "cure_time_volatility", "recent_3_on_time_rate", "recent_3_avg_cure_time",
        "lifetime_prior_default_24h_rate", "recent_5_default_24h_rate",
        "cure_time_trend", "borrower_trend", "loans_last_50_loans",
        "defaults_last_50_loans", "default_rate_last_50_loans",
        "prior_on_time_streak", "prior_default_streak",
        "avg_prior_loan_size", "max_prior_loan_size",
        "loan_size_vs_avg_ratio", "loan_size_vs_max_ratio",
        "loan_above_prior_max_flag",
    ]
    df = _coerce_numeric(df, numeric_cols)
 
    # ── Clip rates to [0, 1] ──
    rate_cols = [
        "lifetime_on_time_24h_rate", "lifetime_on_time_26h_rate",
        "lifetime_default_24h_rate", "lifetime_default_26h_rate",
        "lifetime_severe_default_48h_rate", "lifetime_zero_recovery_rate",
        "prior_on_time_24h_rate", "recent_3_on_time_rate",
        "lifetime_prior_default_24h_rate", "recent_5_default_24h_rate",
        "default_rate_last_50_loans",
    ]
    for col in rate_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=1)
 
    # ── Clip non-negative cols ──
    non_neg_cols = [
        "total_loans", "total_disbursed_amount", "avg_loan_size_lifetime",
        "max_loan_size_lifetime", "lifetime_avg_hours_to_principal_cure",
        "lifetime_worst_hours_to_principal_cure", "lifetime_cure_time_volatility",
        "latest_disbursed_amount", "num_prior_loans", "avg_prior_hours_to_cure",
        "worst_prior_hours_to_cure", "cure_time_volatility", "recent_3_avg_cure_time",
        "loans_last_50_loans", "defaults_last_50_loans",
        "prior_on_time_streak", "prior_default_streak",
        "avg_prior_loan_size", "max_prior_loan_size",
        "loan_size_vs_avg_ratio", "loan_size_vs_max_ratio", "loan_above_prior_max_flag",
    ]
    df = _clip_lower_zero(df, non_neg_cols)
 
    df["borrower_profile_type"] = (
        df["borrower_profile_type"].astype("string").fillna("unknown")
    )
    if "latest_requestid" in df.columns:
        df["latest_requestid"] = df["latest_requestid"].astype(str)
 
    # ── Deduplicate: keep most recent record per borrower ──
    df = df.sort_values(
        ["msisdn", "latest_disbursement_ts", "latest_requestid"],
        ascending=[True, False, False],
        na_position="last",
    )
    n_before = len(df)
    df = df.drop_duplicates(subset=["msisdn"], keep="first")
    logger.info(
        "prepare_borrower_limit_features: deduped %d → %d rows", n_before, len(df)
    )
 
    # ── Tenure and recency ──
    df["borrower_tenure_days"] = (
        (df["latest_loan_ts"] - df["first_loan_ts"]).dt.days.clip(lower=0)
    )
    df["days_since_latest_loan"] = (
        (
            pd.Timestamp.today().normalize()
            - df["latest_loan_ts"].dt.normalize()
        ).dt.days.clip(lower=0)
    )
 
    # ── Composite features ──
    df["exposure_tolerance_proxy"] = df[[
        "avg_prior_loan_size", "max_prior_loan_size",
        "avg_loan_size_lifetime", "max_loan_size_lifetime",
    ]].max(axis=1)
 
    df["recent_risk_proxy"] = (
        0.50 * df["recent_5_default_24h_rate"].fillna(0)
        + 0.30 * df["lifetime_prior_default_24h_rate"].fillna(0)
        + 0.20 * (
            1 - df["recent_3_on_time_rate"].fillna(
                df["prior_on_time_24h_rate"].fillna(0)
            )
        )
    )
 
    df["stability_proxy"] = (
        df["prior_on_time_streak"].fillna(0)
        - df["prior_default_streak"].fillna(0)
    )
 
    logger.info(
        "prepare_borrower_limit_features: output rows = %d", len(df)
    )
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSACTION CAPACITY FEATURES
# ─────────────────────────────────────────────────────────────────────────────
 
def prepare_transaction_capacity_features(
    transaction_capacity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean, validate, and engineer multi-horizon (1m / 3m / 6m) transaction
    capacity features.
 
    Computes the 30d and 90d primary signals that ``compute_capacity_cap()``
    needs as PRIMARY columns, eliminating all fallback paths.
 
    Handles ``agent_msisdn`` → ``msisdn`` rename automatically.
    """
    logger.info(
        "prepare_transaction_capacity_features: input rows = %d",
        len(transaction_capacity_df),
    )
 
    df = transaction_capacity_df.copy()
 
    # ── Rename agent_msisdn → msisdn if needed ──
    if "agent_msisdn" in df.columns and "msisdn" not in df.columns:
        df = df.rename(columns={"agent_msisdn": "msisdn"})
        logger.info("Renamed agent_msisdn → msisdn.")
 
    _check_required_columns(
        df, TRANSACTION_CAPACITY_REQUIRED_COLUMNS, "transaction_capacity_features"
    )
 
    df["msisdn"] = _standardize_msisdn(df["msisdn"])
    df["snapshot_dt"] = pd.to_datetime(df["snapshot_dt"], errors="coerce")
    df["agent_profile"] = df["agent_profile"].astype("string").fillna("unknown")
 
    numeric_cols = [
        "account_balance", "average_balance", "commission",
        "cash_out_vol_1m",   "cash_out_vol_3m",   "cash_out_vol_6m",
        "cash_out_value_1m", "cash_out_value_3m", "cash_out_value_6m",
        "cash_out_cust_1m",  "cash_out_cust_3m",  "cash_out_cust_6m",
        "cash_out_comm_1m",  "cash_out_comm_3m",  "cash_out_comm_6m",
        "cash_in_vol_1m",    "cash_in_vol_3m",    "cash_in_vol_6m",
        "cash_in_value_1m",  "cash_in_value_3m",  "cash_in_value_6m",
        "cash_in_cust_1m",   "cash_in_cust_3m",   "cash_in_cust_6m",
        "cash_in_comm_1m",   "cash_in_comm_3m",   "cash_in_comm_6m",
        "payment_vol_1m",    "payment_vol_3m",    "payment_vol_6m",
        "payment_value_1m",  "payment_value_3m",  "payment_value_6m",
        "payment_cust_1m",   "payment_cust_3m",   "payment_cust_6m",
        "payment_comm_1m",   "payment_comm_3m",   "payment_comm_6m",
        "cust_1m", "cust_3m", "cust_6m",
        "vol_1m",  "vol_3m",  "vol_6m",
    ]
    df = _coerce_numeric(df, numeric_cols, fill_value=0.0)
    df = _clip_lower_zero(df, numeric_cols)
 
    # ── Aggregate if multiple rows per (msisdn, snapshot_dt) ──
    agg_map: dict[str, str] = {
        "agent_profile":    "last",
        "account_balance":  "max",
        "average_balance":  "max",
        "commission":       "sum",
        **{f"cash_out_vol_{h}":   "sum" for h in ("1m", "3m", "6m")},
        **{f"cash_out_value_{h}": "sum" for h in ("1m", "3m", "6m")},
        **{f"cash_out_cust_{h}":  "max" for h in ("1m", "3m", "6m")},
        **{f"cash_out_comm_{h}":  "sum" for h in ("1m", "3m", "6m")},
        **{f"cash_in_vol_{h}":    "sum" for h in ("1m", "3m", "6m")},
        **{f"cash_in_value_{h}":  "sum" for h in ("1m", "3m", "6m")},
        **{f"cash_in_cust_{h}":   "max" for h in ("1m", "3m", "6m")},
        **{f"cash_in_comm_{h}":   "sum" for h in ("1m", "3m", "6m")},
        **{f"payment_vol_{h}":    "sum" for h in ("1m", "3m", "6m")},
        **{f"payment_value_{h}":  "sum" for h in ("1m", "3m", "6m")},
        **{f"payment_cust_{h}":   "max" for h in ("1m", "3m", "6m")},
        **{f"payment_comm_{h}":   "sum" for h in ("1m", "3m", "6m")},
        **{f"cust_{h}": "max" for h in ("1m", "3m", "6m")},
        **{f"vol_{h}":  "sum" for h in ("1m", "3m", "6m")},
    }
    df = df.groupby(["msisdn", "snapshot_dt"], as_index=False).agg(agg_map)
 
    # ── Derived revenue (sum of commissions across transaction categories) ──
    for h in ("1m", "3m", "6m"):
        df[f"revenue_{h}"] = (
            df[f"cash_out_comm_{h}"]
            + df[f"cash_in_comm_{h}"]
            + df[f"payment_comm_{h}"]
        )
 
    # ── Total transaction values per horizon ──
    for h in ("1m", "3m", "6m"):
        df[f"total_txn_value_{h}"] = (
            df[f"cash_out_value_{h}"]
            + df[f"cash_in_value_{h}"]
            + df[f"payment_value_{h}"]
        )
 
    # ── PRIMARY 30d / 90d signals for compute_capacity_cap() ──
    # Naming matches the PRIMARY column names the caps engine expects,
    # so no fallback paths are triggered.
    df["avg_daily_balance_30d"] = df["average_balance"]
    df["avg_daily_balance_90d"] = df["average_balance"]   # single snapshot
 
    df["avg_monthly_revenue_30d"] = df["revenue_1m"]
    df["avg_monthly_revenue_90d"] = df["revenue_3m"] / 3.0   # avg monthly over 90d
 
    df["avg_monthly_txn_count_30d"] = df["vol_1m"]
    df["avg_monthly_txn_count_90d"] = df["vol_3m"] / 3.0
 
    df["avg_monthly_payments_30d"] = df["payment_value_1m"]
    df["avg_monthly_payments_90d"] = df["payment_value_3m"] / 3.0
 
    df["active_customer_count_30d"] = df["cust_1m"]
    df["active_customer_count_90d"] = df["cust_3m"]
 
    df["avg_monthly_txn_volume_30d"] = df["total_txn_value_1m"]
    df["avg_monthly_txn_volume_90d"] = df["total_txn_value_3m"] / 3.0
 
    # ── Legacy 1m derived metrics (kept for output and reference) ──
    df["avg_value_per_txn_1m"] = _safe_divide(
        df["total_txn_value_1m"], df["vol_1m"]
    )
    df["revenue_to_throughput_1m"] = _safe_divide(
        df["revenue_1m"], df["total_txn_value_1m"]
    )
    df["revenue_to_balance_1m"] = _safe_divide(
        df["revenue_1m"], df["average_balance"]
    )
    df["customer_to_volume_ratio_1m"] = _safe_divide(
        df["cust_1m"], df["vol_1m"]
    )
 
    # Weighted capacity proxy (reference signal — not used directly by caps)
    df["capacity_proxy_1m"] = (
        0.35 * df["average_balance"].fillna(0)
        + 0.25 * df["revenue_1m"].fillna(0)
        + 0.20 * df["payment_value_1m"].fillna(0)
        + 0.10 * df["cash_in_value_1m"].fillna(0)
        + 0.10 * df["cash_out_value_1m"].fillna(0)
    )
 
    # ── Operational activity flag ──
    df["operational_activity_flag"] = (
        (df["vol_1m"] > 0)
        | (df["cust_1m"] > 0)
        | (df["total_txn_value_1m"] > 0)
    ).astype(int)
 
    # ── Agent tier ceiling multiplier ──
    agent_lower = df["agent_profile"].str.lower().fillna("unknown")
    df["agent_tier_ceiling_multiplier"] = agent_lower.apply(
        lambda p: next(
            (v for k, v in AGENT_TIER_CEILING_MULTIPLIERS.items() if k in p),
            0.65,
        )
    )
 
    logger.info(
        "prepare_transaction_capacity_features: output rows = %d", len(df)
    )
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAN SUMMARY RECENT FEATURES
# ─────────────────────────────────────────────────────────────────────────────
 
def prepare_loan_summary_recent_features(
    loan_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean and engineer recent loan activity features including 3m variants
    that feed ``compute_recent_usage_cap()``.
    """
    logger.info(
        "prepare_loan_summary_recent_features: input rows = %d",
        len(loan_summary_df),
    )
 
    _check_required_columns(
        loan_summary_df, LOAN_SUMMARY_REQUIRED_COLUMNS, "loan_summary_recent_features"
    )
 
    df = loan_summary_df.copy()
    df["msisdn"] = _standardize_msisdn(df["msisdn"])
    df["snapshot_dt"] = pd.to_datetime(df["snapshot_dt"], errors="coerce")
    df["last_disbursement_date"] = pd.to_datetime(
        df["last_disbursement_date"], errors="coerce"
    )
    df["last_repayment_date"] = pd.to_datetime(
        df["last_repayment_date"], errors="coerce"
    )
 
    numeric_cols = [
        "disbursement_vol_1m", "disbursement_val_1m",
        "repayment_vol_1m", "repayment_val_1m",
        "penalties_1m",
        "disbursement_val_3m", "repayment_val_3m", "penalties_3m",
    ]
    df = _coerce_numeric(df, numeric_cols, fill_value=0.0)
    df = _clip_lower_zero(df, numeric_cols)
 
    agg_map: dict = {
        "last_disbursement_date": "max",
        "last_repayment_date":    "max",
        "disbursement_vol_1m":    "sum",
        "disbursement_val_1m":    "sum",
        "repayment_vol_1m":       "sum",
        "repayment_val_1m":       "sum",
        "penalties_1m":           "sum",
        "disbursement_val_3m":    "sum",
        "repayment_val_3m":       "sum",
        "penalties_3m":           "sum",
    }
    df = df.groupby(["msisdn", "snapshot_dt"], as_index=False).agg(agg_map)
 
    df["repayment_coverage_1m"] = _safe_divide(
        df["repayment_val_1m"], df["disbursement_val_1m"]
    )
    df["repayment_count_coverage_1m"] = _safe_divide(
        df["repayment_vol_1m"], df["disbursement_vol_1m"]
    )
 
    df["credit_days_since_last_disbursement"] = (
        (df["snapshot_dt"] - df["last_disbursement_date"]).dt.days.clip(lower=0)
    )
    df["credit_days_since_last_repayment"] = (
        (df["snapshot_dt"] - df["last_repayment_date"]).dt.days.clip(lower=0)
    )
 
    df["recent_credit_active_flag"] = (
        (df["disbursement_vol_1m"] > 0) | (df["repayment_vol_1m"] > 0)
    ).astype(int)
 
    df["recent_penalty_flag"] = (df["penalties_1m"] > 0).astype(int)
    df["utilization_proxy_1m"] = df[
        ["disbursement_val_1m", "repayment_val_1m"]
    ].max(axis=1)
 
    logger.info(
        "prepare_loan_summary_recent_features: output rows = %d", len(df)
    )
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL CONSISTENCY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
 
def _validate_temporal_alignment(
    transaction_df: pd.DataFrame,
    loan_df: pd.DataFrame,
    max_misalignment_days: int = 7,
) -> None:
    """
    Warn (non-blocking) if snapshot dates across input DataFrames differ
    by more than ``max_misalignment_days``.
    """
    snapshots: dict[str, pd.Timestamp] = {}
    for name, df in [("transaction_capacity", transaction_df), ("loan_summary", loan_df)]:
        if "snapshot_dt" in df.columns:
            dt = pd.to_datetime(df["snapshot_dt"], errors="coerce").dropna()
            if not dt.empty:
                snapshots[name] = dt.max()
 
    if len(snapshots) < 2:
        return
 
    dates = list(snapshots.values())
    for i in range(len(dates)):
        for j in range(i + 1, len(dates)):
            delta = abs((dates[i] - dates[j]).days)
            if delta > max_misalignment_days:
                logger.warning(
                    "Temporal misalignment: snapshot dates differ by %d days "
                    "(threshold = %d). Ensure all inputs cover the same period.",
                    delta,
                    max_misalignment_days,
                )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
 
def build_extrafloat_limit_engine_features(
    borrower_limit_df: pd.DataFrame,
    transaction_capacity_df: pd.DataFrame,
    loan_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main feature pipeline.  Prepares, merges, and engineers all features
    required by ``run_extrafloat_limit_engine()``.
 
    Returns one row per unique msisdn with all computed features.
    """
    logger.info("build_extrafloat_limit_engine_features: starting")
 
    # ── Prepare each source ──
    borrower_df     = prepare_borrower_limit_features(borrower_limit_df)
    transaction_df  = prepare_transaction_capacity_features(transaction_capacity_df)
    loan_df         = prepare_loan_summary_recent_features(loan_summary_df)
 
    # ── Temporal consistency check (non-blocking) ──
    _validate_temporal_alignment(transaction_df, loan_df)
 
    # ── Merge: borrower × transaction on msisdn ──
    merged = borrower_df.merge(transaction_df, on="msisdn", how="left")
    logger.info("After borrower × transaction merge: %d rows", len(merged))
 
    # ── Merge: result × loan summary on [msisdn, snapshot_dt] ──
    merged = merged.merge(loan_df, on=["msisdn", "snapshot_dt"], how="left")
    logger.info("After loan summary merge: %d rows", len(merged))
 
    # ── Post-merge deduplication (keeps most recent snapshot per borrower) ──
    n_before = len(merged)
    merged = merged.sort_values(
        ["msisdn", "snapshot_dt"], ascending=[True, False], na_position="last"
    ).drop_duplicates(subset=["msisdn"], keep="first")
    logger.info("Post-merge dedup: %d → %d rows", n_before, len(merged))
 
    # ── Zero-fill all merged numeric columns ──
    zero_fill = [
        "account_balance", "average_balance", "commission",
        "revenue_1m", "revenue_3m", "revenue_6m",
        "cash_in_value_1m", "cash_out_value_1m", "payment_value_1m",
        "cash_in_value_3m", "cash_out_value_3m", "payment_value_3m",
        "cust_1m", "cust_3m", "vol_1m", "vol_3m",
        "total_txn_value_1m", "total_txn_value_3m",
        "avg_daily_balance_30d", "avg_daily_balance_90d",
        "avg_monthly_revenue_30d", "avg_monthly_revenue_90d",
        "avg_monthly_txn_count_30d", "avg_monthly_txn_count_90d",
        "avg_monthly_payments_30d", "avg_monthly_payments_90d",
        "active_customer_count_30d", "active_customer_count_90d",
        "avg_monthly_txn_volume_30d", "avg_monthly_txn_volume_90d",
        "avg_value_per_txn_1m", "revenue_to_throughput_1m",
        "revenue_to_balance_1m", "customer_to_volume_ratio_1m",
        "capacity_proxy_1m", "operational_activity_flag",
        "disbursement_vol_1m", "disbursement_val_1m",
        "repayment_vol_1m", "repayment_val_1m",
        "penalties_1m", "disbursement_val_3m",
        "repayment_val_3m", "penalties_3m",
        "repayment_coverage_1m", "repayment_count_coverage_1m",
        "recent_credit_active_flag", "recent_penalty_flag",
        "utilization_proxy_1m",
    ]
    for col in zero_fill:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
 
    if "agent_profile" in merged.columns:
        merged["agent_profile"] = merged["agent_profile"].fillna("unknown")
    if "agent_tier_ceiling_multiplier" in merged.columns:
        merged["agent_tier_ceiling_multiplier"] = (
            merged["agent_tier_ceiling_multiplier"].fillna(0.65)
        )
 
    # ── Cross-source composite features ──
    merged["size_tolerance_gap"] = (
        _col(merged, "exposure_tolerance_proxy")
        - _col(merged, "utilization_proxy_1m")
    )
 
    merged["business_to_exposure_ratio"] = _safe_divide(
        _col(merged, "capacity_proxy_1m"),
        _col(merged, "exposure_tolerance_proxy"),
    )
 
    merged["recent_repayment_strength"] = (
        0.70 * _col(merged, "repayment_coverage_1m")
        + 0.30 * _col(merged, "repayment_count_coverage_1m")
    )
 
    merged["activity_alignment_flag"] = (
        (_col(merged, "operational_activity_flag") > 0)
        & (_col(merged, "recent_credit_active_flag") > 0)
    ).astype(int)
 
    # ── Seasonality flag (Uganda / East Africa: Jan, Aug, Sep, Dec) ──
    if "snapshot_dt" in merged.columns:
        months = pd.to_datetime(merged["snapshot_dt"], errors="coerce").dt.month
        merged["is_peak_season_flag"] = (
            months.isin(PEAK_SEASON_MONTHS).fillna(False).astype(int)
        )
    else:
        merged["is_peak_season_flag"] = 0
 
    # ── Alias columns — align with caps engine expected column names ──
    # All aliases are explicitly computed here so the caps engine
    # always finds its expected column names in the output.
 
    merged["on_time_repayment_rate"] = _col(merged, "prior_on_time_24h_rate")
    merged["lifetime_default_rate"]  = _col(merged, "lifetime_default_24h_rate")
 
    # Closest 10-loan proxy available: recent_5_default_24h_rate
    merged["default_rate_last_10_loans"] = _col(merged, "recent_5_default_24h_rate")
 
    merged["avg_cure_time_hours"] = _col(merged, "avg_prior_hours_to_cure")
 
    # Normalise stability_proxy (unbounded) → [0, 1]
    stability_raw = _col(merged, "stability_proxy")
    merged["repayment_stability_score"] = (
        stability_raw.clip(lower=0, upper=STABILITY_NORM_UPPER)
        / STABILITY_NORM_UPPER
    )
 
    merged["recent_disbursement_amount_1m"] = _col(merged, "disbursement_val_1m")
    merged["recent_disbursement_amount_3m"] = _col(merged, "disbursement_val_3m")
    merged["recent_repayment_amount_1m"]    = _col(merged, "repayment_val_1m")
    merged["recent_repayment_amount_3m"]    = _col(merged, "repayment_val_3m")
    merged["recent_repayment_coverage_1m"]  = _col(merged, "repayment_coverage_1m")
    merged["recent_penalty_events_1m"]      = _col(merged, "penalties_1m")
    merged["current_loan_size"]             = _col(merged, "latest_disbursed_amount")
    merged["recent_repayment_performance"]  = _col(merged, "recent_repayment_strength")
 
    # Thin-file flag: fewer than THIN_FILE_LOAN_THRESHOLD lifetime loans
    merged["is_thin_file"] = (
        _col(merged, "total_loans") < THIN_FILE_LOAN_THRESHOLD
    ).astype(int)
 
    merged["is_active_borrower"] = _col(merged, "recent_credit_active_flag").astype(int)
 
    logger.info(
        "build_extrafloat_limit_engine_features: done — rows=%d, cols=%d",
        len(merged),
        len(merged.columns),
    )
    return merged

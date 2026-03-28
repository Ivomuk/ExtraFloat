import numpy as np
import pandas as pd

BORROWER_LIMIT_REQUIRED_COLUMNS = [
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

TRANSACTION_CAPACITY_REQUIRED_COLUMNS = [
    "msisdn",
    "snapshot_dt",
    "agent_profile",
    "average_balance",
    "revenue_1m",
    "cash_in_value_1m",
    "cash_out_value_1m",
    "payment_value_1m",
    "cust_1m",
    "vol_1m",
]

LOAN_SUMMARY_REQUIRED_COLUMNS = [
    "msisdn",
    "snapshot_dt",
    "last_disbursement_date",
    "last_repayment_date",
    "disbursement_vol_1m",
    "disbursement_val_1m",
    "repayment_vol_1m",
    "repayment_val_1m",
    "penalties_1m",
]

def _check_required_columns(df: pd.DataFrame, required_columns: list[str], df_name: str) -> None:
    missing_columns = [column_name for column_name in required_columns if column_name not in df.columns]
    if missing_columns:
        raise ValueError(df_name + " is missing required columns " + ", ".join(missing_columns))

def _standardize_msisdn(series: pd.Series) -> pd.Series:
    cleaned_series = series.astype(str).str.strip()
    cleaned_series = cleaned_series.str.replace(".0", "", regex=False)
    cleaned_series = cleaned_series.replace({"": np.nan, "None": np.nan, "": np.nan})
    return cleaned_series

def _safe_divide(numerator, denominator):
    numerator_series = pd.Series(numerator)
    denominator_series = pd.Series(denominator)
    result_series = np.where(
        (denominator_series.notna()) & (denominator_series != 0),
        numerator_series / denominator_series,
        np.nan
    )
    return pd.Series(result_series, index=numerator_series.index)

def _coerce_numeric(df: pd.DataFrame, column_names: list[str], fill_value: float | None = None) -> pd.DataFrame:
    for column_name in column_names:
        if column_name in df.columns:
            df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
            if fill_value is not None:
                df[column_name] = df[column_name].fillna(fill_value)
    return df

def _coerce_datetime(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    for column_name in column_names:
        if column_name in df.columns:
            df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
    return df

def _clip_lower_zero(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    for column_name in column_names:
        if column_name in df.columns:
            df[column_name] = df[column_name].clip(lower=0)
    return df

def prepare_borrower_limit_features(borrower_limit_df: pd.DataFrame) -> pd.DataFrame:
    _check_required_columns(
        borrower_limit_df,
        BORROWER_LIMIT_REQUIRED_COLUMNS,
        "borrower_limit_features"
    )

    df = borrower_limit_df.copy()
    if "phonenumber" in df.columns and "msisdn" not in df.columns:
        df = df.rename(columns={"phonenumber": "msisdn"})

    datetime_cols = [
        "first_loan_ts",
        "latest_loan_ts",
        "latest_disbursement_ts",
    ]
    df = _coerce_datetime(df, datetime_cols)

    numeric_cols = [
        "total_loans",
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
    df = _coerce_numeric(df, numeric_cols)

    rate_cols = [
        "lifetime_on_time_24h_rate",
        "lifetime_on_time_26h_rate",
        "lifetime_default_24h_rate",
        "lifetime_default_26h_rate",
        "lifetime_severe_default_48h_rate",
        "lifetime_zero_recovery_rate",
        "prior_on_time_24h_rate",
        "recent_3_on_time_rate",
        "lifetime_prior_default_24h_rate",
        "recent_5_default_24h_rate",
        "default_rate_last_50_loans",
    ]
    for column_name in rate_cols:
        if column_name in df.columns:
            df[column_name] = df[column_name].clip(lower=0, upper=1)

    non_negative_cols = [
        "total_loans",
        "total_disbursed_amount",
        "avg_loan_size_lifetime",
        "max_loan_size_lifetime",
        "lifetime_avg_hours_to_principal_cure",
        "lifetime_worst_hours_to_principal_cure",
        "lifetime_cure_time_volatility",
        "latest_disbursed_amount",
        "num_prior_loans",
        "avg_prior_hours_to_cure",
        "worst_prior_hours_to_cure",
        "cure_time_volatility",
        "recent_3_avg_cure_time",
        "loans_last_50_loans",
        "defaults_last_50_loans",
        "prior_on_time_streak",
        "prior_default_streak",
        "avg_prior_loan_size",
        "max_prior_loan_size",
        "loan_size_vs_avg_ratio",
        "loan_size_vs_max_ratio",
        "loan_above_prior_max_flag",
    ]
    df = _clip_lower_zero(df, non_negative_cols)

    df["borrower_profile_type"] = df["borrower_profile_type"].astype("string").fillna("unknown")

    if "latest_requestid" in df.columns:
        df["latest_requestid"] = df["latest_requestid"].astype(str)

    df = df.sort_values(["msisdn", "latest_disbursement_ts", "latest_requestid"], ascending=[True, False, False])
    df = df.drop_duplicates(subset=["msisdn"], keep="first")

    df["borrower_tenure_days"] = (df["latest_loan_ts"] - df["first_loan_ts"]).dt.days
    df["days_since_latest_loan"] = (pd.Timestamp.today().normalize() - df["latest_loan_ts"].dt.normalize()).dt.days
    df["borrower_tenure_days"] = df["borrower_tenure_days"].clip(lower=0)
    df["days_since_latest_loan"] = df["days_since_latest_loan"].clip(lower=0)

    df["exposure_tolerance_proxy"] = df[
        [
            "avg_prior_loan_size",
            "max_prior_loan_size",
            "avg_loan_size_lifetime",
            "max_loan_size_lifetime"
        ]
    ].max(axis=1)

    df["recent_risk_proxy"] = (
        0.50 * df["recent_5_default_24h_rate"].fillna(0) +
        0.30 * df["lifetime_prior_default_24h_rate"].fillna(0) +
        0.20 * (1 - df["recent_3_on_time_rate"].fillna(df["prior_on_time_24h_rate"].fillna(0)))
    )

    df["stability_proxy"] = (
        df["prior_on_time_streak"].fillna(0) -
        df["prior_default_streak"].fillna(0)
    )

    return df

def prepare_transaction_capacity_features(transaction_capacity_df: pd.DataFrame) -> pd.DataFrame:
    _check_required_columns(
        transaction_capacity_df,
        TRANSACTION_CAPACITY_REQUIRED_COLUMNS,
        "transaction_capacity_features"
    )

    df = transaction_capacity_df.copy()
    df["msisdn"] = _standardize_msisdn(df["msisdn"])
    df["snapshot_dt"] = pd.to_datetime(df["snapshot_dt"], errors="coerce")

    numeric_cols = [
        "average_balance",
        "revenue_1m",
        "cash_in_value_1m",
        "cash_out_value_1m",
        "payment_value_1m",
        "cust_1m",
        "vol_1m",
    ]
    df = _coerce_numeric(df, numeric_cols, fill_value=0.0)
    df = _clip_lower_zero(df, numeric_cols)

    df["agent_profile"] = df["agent_profile"].astype("string").fillna("unknown")

    aggregation_map = {
        "agent_profile": "last",
        "average_balance": "max",
        "revenue_1m": "sum",
        "cash_in_value_1m": "sum",
        "cash_out_value_1m": "sum",
        "payment_value_1m": "sum",
        "cust_1m": "max",
        "vol_1m": "sum",
    }

    df = df.groupby(["msisdn", "snapshot_dt"], as_index=False).agg(aggregation_map)

    df["total_txn_value_1m"] = (
        df["cash_in_value_1m"] +
        df["cash_out_value_1m"] +
        df["payment_value_1m"]
    )
    df["avg_value_per_txn_1m"] = _safe_divide(df["total_txn_value_1m"], df["vol_1m"])
    df["revenue_to_throughput_1m"] = _safe_divide(df["revenue_1m"], df["total_txn_value_1m"])
    df["revenue_to_balance_1m"] = _safe_divide(df["revenue_1m"], df["average_balance"])
    df["customer_to_volume_ratio_1m"] = _safe_divide(df["cust_1m"], df["vol_1m"])

    df["capacity_proxy_1m"] = (
        0.35 * df["average_balance"].fillna(0) +
        0.25 * df["revenue_1m"].fillna(0) +
        0.20 * df["payment_value_1m"].fillna(0) +
        0.10 * df["cash_in_value_1m"].fillna(0) +
        0.10 * df["cash_out_value_1m"].fillna(0)
    )

    df["operational_activity_flag"] = (
        (df["vol_1m"] > 0) |
        (df["cust_1m"] > 0) |
        (df["total_txn_value_1m"] > 0)
    ).astype(int)

    return df

def prepare_loan_summary_recent_features(loan_summary_df: pd.DataFrame) -> pd.DataFrame:
    _check_required_columns(
        loan_summary_df,
        LOAN_SUMMARY_REQUIRED_COLUMNS,
        "loan_summary_recent_features"
    )

    df = loan_summary_df.copy()
    df["msisdn"] = _standardize_msisdn(df["msisdn"])
    df["snapshot_dt"] = pd.to_datetime(df["snapshot_dt"], errors="coerce")
    df["last_disbursement_date"] = pd.to_datetime(df["last_disbursement_date"], errors="coerce")
    df["last_repayment_date"] = pd.to_datetime(df["last_repayment_date"], errors="coerce")

    numeric_cols = [
        "disbursement_vol_1m",
        "disbursement_val_1m",
        "repayment_vol_1m",
        "repayment_val_1m",
        "penalties_1m",
    ]
    df = _coerce_numeric(df, numeric_cols, fill_value=0.0)
    df = _clip_lower_zero(df, numeric_cols)

    aggregation_map = {
        "last_disbursement_date": "max",
        "last_repayment_date": "max",
        "disbursement_vol_1m": "sum",
        "disbursement_val_1m": "sum",
        "repayment_vol_1m": "sum",
        "repayment_val_1m": "sum",
        "penalties_1m": "sum",
    }

    df = df.groupby(["msisdn", "snapshot_dt"], as_index=False).agg(aggregation_map)

    df["repayment_coverage_1m"] = _safe_divide(df["repayment_val_1m"], df["disbursement_val_1m"])
    df["repayment_count_coverage_1m"] = _safe_divide(df["repayment_vol_1m"], df["disbursement_vol_1m"])

    df["credit_days_since_last_disbursement"] = (
        df["snapshot_dt"] - df["last_disbursement_date"]
    ).dt.days

    df["credit_days_since_last_repayment"] = (
        df["snapshot_dt"] - df["last_repayment_date"]
    ).dt.days

    if "credit_days_since_last_disbursement" in df.columns:
        df["credit_days_since_last_disbursement"] = df["credit_days_since_last_disbursement"].clip(lower=0)

    if "credit_days_since_last_repayment" in df.columns:
        df["credit_days_since_last_repayment"] = df["credit_days_since_last_repayment"].clip(lower=0)

    df["recent_credit_active_flag"] = (
        (df["disbursement_vol_1m"] > 0) |
        (df["repayment_vol_1m"] > 0)
    ).astype(int)

    df["recent_penalty_flag"] = (df["penalties_1m"] > 0).astype(int)
    df["utilization_proxy_1m"] = df[["disbursement_val_1m", "repayment_val_1m"]].max(axis=1)

    return df

def build_extrafloat_limit_engine_features(
    borrower_limit_df: pd.DataFrame,
    transaction_capacity_df: pd.DataFrame,
    loan_summary_df: pd.DataFrame
) -> pd.DataFrame:
    borrower_df = prepare_borrower_limit_features(borrower_limit_df)
    transaction_df = prepare_transaction_capacity_features(transaction_capacity_df)
    loan_df = prepare_loan_summary_recent_features(loan_summary_df)

    merged_df = borrower_df.merge(
        transaction_df,
        on="msisdn",
        how="left"
    )

    merged_df = merged_df.merge(
        loan_df,
        on=["msisdn", "snapshot_dt"],
        how="left"
    )

    zero_fill_cols = [
        "average_balance",
        "revenue_1m",
        "cash_in_value_1m",
        "cash_out_value_1m",
        "payment_value_1m",
        "cust_1m",
        "vol_1m",
        "total_txn_value_1m",
        "avg_value_per_txn_1m",
        "revenue_to_throughput_1m",
        "revenue_to_balance_1m",
        "customer_to_volume_ratio_1m",
        "capacity_proxy_1m",
        "operational_activity_flag",
        "disbursement_vol_1m",
        "disbursement_val_1m",
        "repayment_vol_1m",
        "repayment_val_1m",
        "penalties_1m",
        "repayment_coverage_1m",
        "repayment_count_coverage_1m",
        "recent_credit_active_flag",
        "recent_penalty_flag",
        "utilization_proxy_1m",
    ]

    for column_name in zero_fill_cols:
        if column_name in merged_df.columns:
            merged_df[column_name] = merged_df[column_name].fillna(0)

    if "agent_profile" in merged_df.columns:
        merged_df["agent_profile"] = merged_df["agent_profile"].fillna("unknown")

    merged_df["size_tolerance_gap"] = (
        merged_df["exposure_tolerance_proxy"].fillna(0) -
        merged_df["utilization_proxy_1m"].fillna(0)
    )

    merged_df["business_to_exposure_ratio"] = _safe_divide(
        merged_df["capacity_proxy_1m"],
        merged_df["exposure_tolerance_proxy"]
    )

    merged_df["recent_repayment_strength"] = (
        0.70 * merged_df["repayment_coverage_1m"].fillna(0) +
        0.30 * merged_df["repayment_count_coverage_1m"].fillna(0)
    )

    merged_df["activity_alignment_flag"] = (
        (merged_df["operational_activity_flag"].fillna(0) > 0) &
        (merged_df["recent_credit_active_flag"].fillna(0) > 0)
    ).astype(int)

    output_columns = [
        "msisdn",
        "borrower_profile_type",
        "agent_profile",
        "latest_requestid",
        "first_loan_ts",
        "latest_loan_ts",
        "latest_disbursement_ts",
        "snapshot_dt",
        "total_loans",
        "num_prior_loans",
        "borrower_tenure_days",
        "days_since_latest_loan",
        "prior_on_time_24h_rate",
        "recent_3_on_time_rate",
        "lifetime_prior_default_24h_rate",
        "recent_5_default_24h_rate",
        "avg_prior_hours_to_cure",
        "worst_prior_hours_to_cure",
        "cure_time_volatility",
        "cure_time_trend",
        "borrower_trend",
        "prior_on_time_streak",
        "prior_default_streak",
        "default_rate_last_50_loans",
        "avg_prior_loan_size",
        "max_prior_loan_size",
        "avg_loan_size_lifetime",
        "max_loan_size_lifetime",
        "loan_size_vs_avg_ratio",
        "loan_size_vs_max_ratio",
        "loan_above_prior_max_flag",
        "exposure_tolerance_proxy",
        "recent_risk_proxy",
        "stability_proxy",
        "average_balance",
        "revenue_1m",
        "cash_in_value_1m",
        "cash_out_value_1m",
        "payment_value_1m",
        "cust_1m",
        "vol_1m",
        "total_txn_value_1m",
        "avg_value_per_txn_1m",
        "revenue_to_throughput_1m",
        "revenue_to_balance_1m",
        "customer_to_volume_ratio_1m",
        "capacity_proxy_1m",
        "operational_activity_flag",
        "last_disbursement_date",
        "last_repayment_date",
        "disbursement_vol_1m",
        "disbursement_val_1m",
        "repayment_vol_1m",
        "repayment_val_1m",
        "penalties_1m",
        "repayment_coverage_1m",
        "repayment_count_coverage_1m",
        "credit_days_since_last_disbursement",
        "credit_days_since_last_repayment",
        "recent_credit_active_flag",
        "recent_penalty_flag",
        "utilization_proxy_1m",
        "size_tolerance_gap",
        "business_to_exposure_ratio",
        "recent_repayment_strength",
        "activity_alignment_flag",
    ]

    existing_output_columns = [column_name for column_name in output_columns if column_name in merged_df.columns]
    final_df = merged_df[existing_output_columns].copy()

    return final_df


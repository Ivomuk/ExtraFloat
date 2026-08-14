"""
Feature-level configuration for the PD model pipeline.

Lists, patterns, and column-name constants that control feature selection,
leakage detection, and identity-column exclusion are defined here so that
downstream functions receive them as parameters rather than embedding them
as inline literals.
"""

from __future__ import annotations

# ======================================================================== #
# Column name constants
# ======================================================================== #

AGENT_KEY: str = "agent_msisdn"
TARGET_COL: str = "bad_state"
LABEL_COL: str = "hard_bad_flag"
THIN_FILE_COL: str = "thin_file_flag"
THIN_FILE_PRIOR_COL: str = "thin_file_pd_prior"
SNAPSHOT_COL: str = "snapshot_dt"
SPLIT_COL: str = "split"

DATE_COLS: list[str] = [
    "tbl_dt",
    "activation_dt",
    "snapshot_dt",
    "Last_disbursement_date",
    "Last_repayment_date",
    "date_of_birth",
    # data/loan_state_query_updated_materialized.txt (loan-level training query)
    "loan_date",
    "disbursement_ts",
    "label_horizon_7d_end",
    "label_horizon_30d_end",
    "scoring_state_date",
    "last_closed_loan_closure_date",
    "first_outcome_state_date",
    "last_outcome_state_date",
]

# ======================================================================== #
# Feature blacklist -- columns that must never enter the model feature set
# ======================================================================== #

PD_FEATURE_BLACKLIST: frozenset[str] = frozenset(
    {
        # Identifiers / snapshot keys
        "agent_msisdn",
        "tbl_dt",
        "account_number",
        "snapshot_dt",
        "account_name",
        "split",
        # Raw date anchors
        "Last_disbursement_date",
        "Last_repayment_date",
        "activation_dt",
        "date_of_birth",
        "days_since_snapshot",
        # Labels
        "bad_state",
        "bad_state_30D",
        "hard_bad_flag",
        # Delinquency / penalty constructs
        "penalties_1M",
        "penalties_3M",
        "penalties_6M",
        "penalty_frequency_6M",
        "penalty_roll_forward",
        "ever_penalized",
        "ever_delinquent_flag",
        "cured_after_penalty",
        "current_dpd",
        "dpd_30_plus",
        "dpd_60_plus",
        "dpd_90_plus",
        # Exposure / history flags
        "has_ever_loan",
        "has_loan",
        "loan_history",
        "has_loan_history",
        "is_new_agent",
        # Thin-file / policy controls
        "thin_file_flag",
        "thin_file_pd_prior",
        "no_loan_history_flag",
        # Point-in-time, 180-day-bounded thin-file routing signals -- feed
        # thin_file_flag only, not the thick-file model (see
        # loan_history_features.py's derive_thin_file_flag).
        "prior_loan_count_180d",
        "prior_active_loan_months_180d",
        # Routing intermediates — derived from disbursement_vol_mN which are in the
        # feature set; these summaries are redundant and cause spurious leakage flags
        # because no-loan-history agents (bad_state≈0) always have value 0.
        "distinct_loan_months",
        "total_loans_6m",
        # Target aliases
        "target",
        # Outcome encoders
        "chronic_delinquency_flag",
        # Cluster labels
        "cluster_id_k4",
        "cluster_round1",
        "cluster_id_k6",
        "cluster_round2",
        "cluster_id_gmm",
        # Scorecard outputs -- derived PD proxies, must not feed into the model
        "never_loan_points",
        "never_loan_score_0_100",
        "never_loan_pd_like",
        "never_loan_top_drivers",
        # Business-process co-definition proxy -- binary alias of net_exposure_6M
        "currently_outstanding_flag",
        # Sample-selection filter -- present in repayments CSV, must never enter features
        "outcome_observed_30d",
        "outcome_observed_30D",
        # data/loan_state_query_updated_materialized.txt -- identifiers, join
        # keys, and audit columns (see pd_model/preprocessing/loan_history_features.py)
        "msisdn",
        "disbursement_fid",
        "disbursement_uid",
        "target_loan_uid",
        "target_loan_seq",
        "same_day_disbursement_position",
        "same_day_disbursement_count",
        "scoring_state_loan_uid",
        "scoring_state_date",
        "last_closed_loan_uid",
        "last_closed_loan_closure_date",
        "loan_seq_minus_observed_prior_loan_count",
        "sales_region",
        "sales_territory",
        "district",
        "loan_date",
        "disbursement_ts",
        "label_horizon_7d_end",
        "label_horizon_30d_end",
        # Loan-level labels
        "bad_state_3dpd_30d",
        "bad_state_1dpd_7d",
        # Label-diagnostic columns -- future-derived, retained by the SQL for
        # label auditing only (see data/Features_Consult.txt and
        # pd_model/preprocessing/loan_history_features.py's
        # LABEL_DIAGNOSTIC_COLUMNS, which this list mirrors). Only a few of
        # these are caught by LEAKAGE_PATTERNS's "outcome" substring below;
        # most (days_aging / rollover / terminal_state / the coverage-ratio
        # and eligibility columns) match no existing pattern and would
        # otherwise leak straight into the model.
        "max_days_aging_7d",
        "max_days_aging_30d",
        "rollover_observed_7d",
        "rollover_observed_30d",
        "terminal_state_observed_30d",
        "same_day_settlement_observed_30d",
        "outcome_state_row_count_30d",
        "outcome_observed_date_count_30d",
        "post_disbursement_observed_date_count_7d",
        "post_disbursement_observed_date_count_30d",
        "first_outcome_state_date",
        "last_outcome_state_date",
        "observed_closure_date_30d",
        "closure_observation_state_date_30d",
        "required_observation_end_date_30d",
        "expected_observed_date_count_30d",
        "observed_date_count_to_required_end_30d",
        "last_state_date_to_required_end_30d",
        "follow_up_coverage_ratio_30d",
        "meets_coverage_ratio_30d",
        "near_horizon_observation_30d",
        "confirmed_good_30d",
        "has_post_disbursement_state_30d",
        "fails_coverage_ratio_30d",
        "fails_near_horizon_30d",
        "label_eligible_30d",
        "label_eligibility_reason_30d",
    }
)

# ======================================================================== #
# Pattern-based exclusion lists
# ======================================================================== #

# Substrings whose presence in a column name indicates leakage
LEAKAGE_PATTERNS: tuple[str, ...] = (
    "has_ever",
    "ever_",
    "bad_state",
    "hard_bad",
    "writeoff",
    "write_off",
    "charged_off",
    "charge_off",
    "default",
    "delinq",
    "delinquency",
    "penalt",   # catches both "penalty" and "penalties"
    "future",   # catches any forward-looking column (e.g. future_penalties_30d)
    "outcome",  # catches outcome_observed_30d and similar
    "collections",
    "collection",
    "recovery",
    "label",
    "target",
    # Defense-in-depth for data/loan_state_query_updated_materialized.txt's
    # label-diagnostic columns -- the exact names are already in
    # PD_FEATURE_BLACKLIST; these substrings catch any similarly-named
    # future SQL columns automatically. NOTE: "max_days_aging" (not the
    # broader "days_aging") -- the legitimate feature
    # active_loan_days_aging_at_scoring / active_loan_days_aging_at_snapshot
    # also contains "days_aging" and must not be blocked by this pattern
    # (confirmed via a synthetic end-to-end run_pipeline smoke test, which
    # raised DataLeakageError on that exact column before this fix).
    "max_days_aging",
    "rollover",
    "terminal_state",
)

# Substrings indicating identity-like or bookkeeping columns
ID_LIKE_PATTERNS: tuple[str, ...] = (
    "msisdn",
    "imei",
    "imsi",
    "customer_id",
    "cust_id",
    "client_id",
    "account_id",
    "acct_id",
    "agent_id",
    "user_id",
    "device_id",
    "national_id",
    "nid",
    "tbl_dt",
    "as_of",
    "snapshot_dt",
    "report_dt",
    "run_dt",
    "date_",
    "outcome_observed",
)

# DPD substring patterns that are definitively leakage
DPD_BLOCK_PATTERNS: tuple[str, ...] = (
    "max_dpd",
    "worst_dpd",
    "ever_dpd",
    "dpd_ever",
    "dpd_max",
    "dpd_worst",
    "dpd90",
    "dpd_90",
    "ever90",
    "ever_90",
    "dpd120",
    "dpd_120",
    "ever120",
    "ever_120",
    "dpd30",
    "dpd_30",
    "ever30",
    "ever_30",
    "dpd60",
    "dpd_60",
    "ever60",
    "ever_60",
)

# DPD substrings that are explicitly allowed despite containing "dpd"
DPD_ALLOW_PATTERNS: tuple[str, ...] = (
    "current_dpd",
    "dpd_current",
    "dpd_now",
    "days_since",
    "days_since_last_dpd",
    "days_since_dpd",
)

# ======================================================================== #
# Transformation classification patterns
# ======================================================================== #

COUNT_PATTERNS: tuple[str, ...] = (
    "cash_in_vol",
    "cash_out_vol",
    "payment_vol",
    "repayment_vol",
    "disbursement_vol",
    "_peers_",
    "_cust_",
    "_txns_",
    "_cnt",
    "_count",
    "if_active",
    "vol_1m",
    "vol_3m",
    "vol_6m",
)

LOG_PATTERNS: tuple[str, ...] = (
    "_value",
    "_val",
    "commission",
    "balance",
    "revenue",
    "voucher",
    "rev_",
    "payment_comm_",
    # data/loan_state_query_updated_materialized.txt monetary columns
    # (disbursement_amount_ugx, all_prior_loans_disbursed_ugx,
    # active_loan_outstanding_ugx_at_scoring, avg_prior_loan_amount_30d, ...)
    # match none of the patterns above -- "_val"/"_value" require that exact
    # substring, which "ugx"/"amount" never contain. Without these two
    # patterns these right-skewed money columns fall through to
    # DEFAULT PROTECTED (left raw, unwinsorized) instead of getting the
    # same log1p + winsorize treatment the old *_val_6M columns got.
    "_ugx",
    "_amount",
)

SIGNED_AMOUNT_PATTERNS: tuple[str, ...] = (
    "net_cash_flow",
    "net_cashflow",
    "net_flow",
    "cash_flow",
    "cashflow",
    "delta_",
    "change_",
    "diff_",
    "net_",
    "pnl",
    "profit",
    "loss",
)

CAP_ONLY_PATTERNS: tuple[str, ...] = (
    "_ratio",
    "_intensity",
    "_per_",
    "_to_",
    "_vs_",
    "coverage",
    "growth",
    "share",
    "volatility",
    "_cv",
    "avg_monthly",
    "repayment_gap_days",
    "cust_1m",
    "cust_3m",
    "cust_6m",
)

PROTECTED_PATTERNS: tuple[str, ...] = (
    "_flag",
    "_indicator",
    "is_",
    "num_",
    "consistent_",
    "sharp_",
    "days_since",
    "thin_file",
    "cluster",
    "has_loan",
    "loan_history",
    "bucket",
)

# ======================================================================== #
# Non-behavioural columns excluded from repayment feature list
# ======================================================================== #

NON_BEHAVIOURAL_COLS: frozenset[str] = frozenset(
    {
        "agent_msisdn",
        "msisdn",
        "snapshot_dt",
        "tbl_dt",
        "split",
        "thin_file_flag",
        "thin_file_pd_prior",
        "bad_state",
        "bad_state_30d",
        "bad_state_30D",
        "future_penalties_30d",
        "future_penalties_30D",
        "hard_bad_flag",
        "has_ever_loan",
        "has_loan_history",
        "is_new_agent",
        "Last_disbursement_date",
        "last_disbursement_date",
        "Last_repayment_date",
        "last_repayment_date",
    }
)

# Pattern-based forward-looking / label substrings for repayment feature guard
REPAYMENT_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "bad_state",
    "write_off",
    "written_off",
    "collection",
    "dpd_target",
)

# ======================================================================== #
# Scoring / postprocessing column constants
# ======================================================================== #

RAW_SCORE_COL: str = "raw_score"
CAL_PD_COL: str = "cal_pd"
DECISION_SOURCE_COL: str = "decision_source"
POLICY_BUCKET_COL: str = "final_policy_bucket"

# Whitelist / blacklist evaluation
WL_BL_COL: str = "xtrafloat_list_type"
WL_BL_KEY: str = "agent_msisdn_key"
WL_CATEGORY_COL: str = "agent_category"
WL_REASON_COL: str = "reason"
NON_PERF_BLACKLIST_REASONS: tuple[str, ...] = (
    "As requested by Director",
    "Agent active less than 3 months",
)

"""
Feature-level configuration for the PD model pipeline.

Lists, patterns, and column-name constants that control feature selection,
leakage detection, and identity-column exclusion are defined here so that
downstream functions receive them as parameters rather than embedding them
as inline literals.
"""

from __future__ import annotations

from typing import FrozenSet, List, Tuple


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

DATE_COLS: List[str] = [
    "tbl_dt",
    "activation_dt",
    "snapshot_dt",
    "Last_disbursement_date",
    "Last_repayment_date",
    "date_of_birth",
]

# ======================================================================== #
# Feature blacklist — columns that must never enter the model feature set
# ======================================================================== #

PD_FEATURE_BLACKLIST: FrozenSet[str] = frozenset(
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
        # Scorecard outputs — derived PD proxies, must not feed into the model
        "never_loan_points",
        "never_loan_score_0_100",
        "never_loan_pd_like",
        "never_loan_top_drivers",
    }
)

# ======================================================================== #
# Pattern-based exclusion lists
# ======================================================================== #

# Substrings whose presence in a column name indicates leakage
LEAKAGE_PATTERNS: Tuple[str, ...] = (
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
    "penalty",
    "collections",
    "collection",
    "recovery",
    "label",
    "target",
)

# Substrings indicating identity-like or bookkeeping columns
ID_LIKE_PATTERNS: Tuple[str, ...] = (
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
DPD_BLOCK_PATTERNS: Tuple[str, ...] = (
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
DPD_ALLOW_PATTERNS: Tuple[str, ...] = (
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

COUNT_PATTERNS: Tuple[str, ...] = (
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

LOG_PATTERNS: Tuple[str, ...] = (
    "_value",
    "_val",
    "commission",
    "balance",
    "revenue",
    "voucher",
    "rev_",
    "payment_comm_",
)

SIGNED_AMOUNT_PATTERNS: Tuple[str, ...] = (
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

CAP_ONLY_PATTERNS: Tuple[str, ...] = (
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

PROTECTED_PATTERNS: Tuple[str, ...] = (
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

NON_BEHAVIOURAL_COLS: FrozenSet[str] = frozenset(
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
REPAYMENT_FORBIDDEN_SUBSTRINGS: Tuple[str, ...] = (
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
NON_PERF_BLACKLIST_REASONS: Tuple[str, ...] = (
    "As requested by Director",
    "Agent active less than 3 months",
)

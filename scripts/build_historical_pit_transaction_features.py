"""
build_historical_pit_transaction_features.py
==================================================
Stage 4A.1 -- RESEARCH-ONLY Phase 2.1 point-in-time (PIT) transaction-feature
reconstruction. Produces NO PD scores, NO C3 application, no `k` selection,
no Capacity(F) combination. Output is reconstructed Phase 2.1 feature
vectors plus audit/status columns only -- never named `cal_pd` or implied
equivalent to the live pipeline's actual output (which also depends on
other phases this script doesn't touch).

WHY THIS SCRIPT EXISTS: `audit_historical_pd_rescoring_feasibility.py`
found Phase 2.1 (transaction/commission/balance features) not PIT-
reconstructable with the current production pipeline
(`pd_model/preprocessing/transaction_features.py`), specifically because
`run_phase_2_1_richer_tx_behaviour` computes recency (`days_since_
snapshot`) relative to `df["tbl_dt"].max()` -- the max date in whatever
batch happens to be loaded, not a caller-supplied historical decision
date. A real multi-month mart export (7 distinct month-end snapshots,
2026-01 through 2026-07) was subsequently confirmed to exist, which
means the remaining blocker is purely an implementation gap, not a data-
availability wall. This script tests whether that gap is closeable by
reusing a proven, already-verified as-of-join pattern from elsewhere in
this exact repo.

**This script NEVER modifies, imports from, or calls**
`pd_model/preprocessing/transaction_features.py` or `pd_model/
run_pipeline.py` -- the production PD pipeline is protected while
historical reconstruction is established as scientifically valid, as a
separate, read-only research path. If this script's reconstruction is
later judged sound, `audit_historical_pd_rescoring_feasibility.py` may be
re-run (unmodified) to see whether Phase 2.1's disposition changes --
that decision and any resulting change to the audit script are
deliberately NOT made here.

CONFIRMED FROM READING `pd_model/preprocessing/transaction_features.py`
IN FULL: `run_phase_2_1_richer_tx_behaviour` has exactly ONE batch-
relative computation in the entire file --

    if "tbl_dt" in df.columns:
        ref_date = df["tbl_dt"].max()
        df["days_since_snapshot"] = (ref_date - df["tbl_dt"]).dt.days

Every other feature block is pure row-level arithmetic on that row's own
columns. This script restates every row-level block verbatim (same
conditions, same formulas, same defensive `if all(c in df.columns ...)`
gating), and replaces ONLY the recency block: `days_since_snapshot` here
comes from the as-of join below (each episode's own `loan_date` minus
the matched historical snapshot's date), never a batch maximum.

UPSTREAM PEER/CLUSTER PROVENANCE IS AN EXPLICIT, UNRESOLVED GATE -- not a
minor footnote (per review correction: an earlier draft under-weighted
this as an informational caveat). Several Phase 2.1 blocks consume, but
do not themselves compute, upstream cluster/peer aggregate columns
(`commission_cluster_mean`, `vol_3m_cluster_mean`, `cluster_avg_
commission`, `cluster_avg_vol_3m`, `cash_in_peers_3m`, `cash_in_vol_3m`).
Those columns are produced by an earlier, upstream phase this
investigation did not read. `fundamentals_snapshot_date < loan_date` is
NECESSARY but NOT SUFFICIENT for the full Phase 2.1 vector to be PIT-
valid: the matched historical mart row is dated correctly, but a cluster/
peer aggregate ON that row could itself have been computed from a
reference population or period that was not actually available as of
that historical date (e.g. a July snapshot's cluster mean computed across
agents or a window not yet observable in July). That is a DIFFERENT
leakage channel than the as-of join solves, and this script does not
claim to have solved it.

Accordingly this script reports FOUR SEPARATE, independent facts, never
collapsed into one combined verdict (per review -- an earlier draft
folded "present" and "provenance" into a single status; this round
splits them, because each answers a genuinely different question):
- `reconstruction_status` -- whether the as-of join itself found a valid,
  strictly-prior historical mart row (`PIT_PHASE21_RECONSTRUCTED` /
  `PIT_PHASE21_UNAVAILABLE_NO_PRIOR_SNAPSHOT`). This is "RowPIT."
- `cluster_peer_columns_present` (bool) -- were any upstream cluster/peer
  columns actually present and non-null on the matched historical row?
  Purely a data-presence fact.
- `cluster_peer_pit_provenance` -- `"NOT_APPLICABLE"` when no such column
  was present (no derived feature exists to be in question);
  `"NOT_VERIFIED"` -- **never a default pass** -- whenever at least one
  was, because this script does not audit those columns' own point-in-
  time construction. A `"VERIFIED"` value exists in the enum for a
  future lineage audit to use; this script never sets it.
- `required_by_scoring_schema` -- `"YES"` / `"NO"` / `"UNKNOWN"`, a
  SCHEMA-level fact (same for every row in a given run, broadcast for
  convenience) from `check_upstream_feature_schema_overlap`: does the
  champion model's `selected_features` actually include any cluster/
  peer-derived feature name at all? `"UNKNOWN"` when no populated
  champion schema exists yet (the real repo currently has only
  placeholder `pd_model/artifacts/*.json` -- Stage 4A already found
  this).

THE EVENTUAL GATE LOGIC, recorded here for the NEXT step, NOT computed by
this script: `Phase21PIT = RowPIT AND RequiredFeatureCoverage AND
UpstreamDerivedFeaturePIT`, where `UpstreamDerivedFeaturePIT` only
matters (a `NOT_VERIFIED` provenance only BLOCKS) when
`required_by_scoring_schema == "YES"` -- if no cluster/peer-derived
feature is actually required by the champion, its unresolved provenance
is irrelevant to historical scoring. This script deliberately does NOT
compute that combined AND, and does NOT invent a feature-requirement set
of its own: it reconstructs and inventories the four facts above; a
later, separate step updates Stage 4A's own Gate 1 to combine them, only
once a real champion schema exists to combine them against.

THE CORRECT CONCLUSION FOR THIS SCRIPT (per review -- the only phrasing
this script's own output may use, printed verbatim in `main`): "Historical
Phase 2.1 row selection and row-derived feature reconstruction succeeded.
Full Phase 2.1 scoring-vector PIT validity remains conditional on
required-feature coverage and PIT provenance of any required upstream-
derived peer/cluster inputs." Never anything resembling "Phase 2.1 PIT
reconstruction succeeded" unqualified -- that overclaims what has
actually been established.

REUSES (restates, never imports -- one-way `scripts/` layering
convention) the exact as-of-join pattern already built, tested, and
verified in `scripts/build_loan_episode_capacity_dataset.py`:
`pd.merge_asof(direction="backward", allow_exact_matches=False)`, with
the same `TypeError` fallback for older pandas, the same strict `<`
requirement (an exact-date snapshot never matches), and the same
never-backfill discipline (no eligible prior snapshot -> left as a
distinct, explicit status, never zero-filled or borrowed from a later
snapshot). `_parse_mart_date` is also restated verbatim, including its
documented YYYYMMDD-vs-plain-date dtype guard (a numeric `tbl_dt` column
parsed with a plain `pd.to_datetime` silently collapses every distinct
date to ~1970-01-01 with no error -- confirmed directly in the episode
builder's own history; this script routes numeric date columns through
the same explicit `format="%Y%m%d"` parse to avoid it).

LEAKAGE ASSERTION: `assert_no_leakage` is called unconditionally before
any feature is computed. For every row classified as reconstructed, it
asserts `fundamentals_snapshot_date < loan_date` and raises loudly (not a
warning) on any violation -- this is the single most load-bearing check
in the script.

Imports `pd_model.config.model_config.DEFAULT_CONFIG` directly for `eps`
(not a `scripts/` module, so this doesn't violate the layering
convention) -- reusing the real constant rather than risking value drift
from a hand-copied literal.

Usage:
    python scripts\\build_historical_pit_transaction_features.py ^
        --loan-training-file data\\state_data_20260910_retail_filtered.csv ^
        --transaction-mart-file data\\mfs_daily_agent_mart_202607_retail_filtered.csv ^
        --out-prefix stage4a1_phase21_pit
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pd_model.config.model_config import DEFAULT_CONFIG  # noqa: E402
from pd_model.preprocessing.loan_history_features import (  # noqa: E402
    LABEL_DIAGNOSTIC_COLUMNS,
    SNAPSHOT_TO_TRAINING_COLUMN_MAP,
)
from segmentation.borrower_persona_clustering import digits  # noqa: E402

LOAN_REQUIRED_COLS = ["disbursement_fid", "agent_msisdn", "loan_date"]

CLUSTER_PEER_COLS = [
    "commission_cluster_mean", "vol_3m_cluster_mean",
    "cluster_avg_commission", "cluster_avg_vol_3m",
    "cash_in_peers_3m", "cash_in_vol_3m",
]

# The ENTIRE raw-mart-column whitelist restate_phase_2_1_features can ever
# consume -- every column named in any `if ... in df.columns` check in that
# function, plus the base_names x {1m,3m,6m} sweep. A real mart export can
# carry dozens of OTHER columns Phase 2.1 never reads; merging all of them
# through merge_asof's by= grouped join on millions of rows is a real,
# observed memory blowup (a real run OOM'd inside pandas' merge_asof
# internals on a 90-numeric-column mart x 5.6M loan episodes). Narrowing
# load_mart to this whitelist is a memory fix, not a correctness change --
# restate_phase_2_1_features's own per-block `if c in df.columns` gating
# already behaves identically whether an unused column was dropped before
# the join or merely unused after it.
PHASE_21_RAW_COLS = sorted(set(
    ["vol_1m", "vol_3m", "vol_6m", "commission", "account_balance", "average_balance",
     "cust_1m", "cust_3m", "cash_in_value_3m", "cash_out_value_3m", "payment_value_3m"]
    + CLUSTER_PEER_COLS
    + [f"{base}_{horizon}" for base in ("cash_out_vol", "cash_in_vol", "payment_vol", "vol")
       for horizon in ("1m", "3m", "6m")]
))

# Output feature names this script derives FROM the CLUSTER_PEER_COLS above --
# the schema-overlap check below intersects this list against the champion's
# actual selected_features, never the raw upstream columns themselves.
CLUSTER_DERIVED_FEATURE_NAMES = [
    "commission_drop_flag",
    "commission_vs_cluster_mean_ratio", "commission_vs_cluster_mean_diff",
    "vol_3m_vs_cluster_mean_ratio", "vol_3m_vs_cluster_mean_diff",
    "cluster_commission_per_vol_3m", "commission_per_vol_vs_cluster_ratio",
    "peer_dependency_ratio", "high_peer_dependency_flag",
]

# ======================================================================== #
# Champion-feature-coverage check: canonical PIT-lineage name sets.
#
# These answer a DIFFERENT question than CLUSTER_DERIVED_FEATURE_NAMES
# above (which only tracks the 9 cluster-derived Phase 2.1 outputs for the
# schema-overlap check). Here, EVERY champion selected_feature must be
# classified by the PROVENANCE OF ITS GENERATION PATH -- never by whether
# a same-named column happens to exist in some mart/history export. Each
# set below is the full, code-verified output-name vocabulary of one
# legitimate point-in-time lineage. See check_champion_feature_coverage().
# ======================================================================== #

# The FULL raw output schema of data/agent_profile_snapshot_query.sql's
# final SELECT (that file's own header documents "Output columns (57
# total)"), confirmed by direct read -- NOT the same thing as
# PHASE_21_RAW_COLS above. PHASE_21_RAW_COLS is deliberately narrower: it
# exists only to bound load_mart's merge_asof memory footprint to the
# columns restate_phase_2_1_features actually consumes. This constant is
# for a different purpose -- classifying a champion-selected RAW mart
# column (one restate_phase_2_1_features never touches at all, e.g.
# cash_in_comm_1m, cash_out_cust_3m) as legitimately Phase-2.1-sourced,
# since it comes from the identical point-in-time snapshot query (same
# `tbl_dt > dt_mN AND tbl_dt <= snapshot_dt` window guard as the columns
# already in PHASE_21_RAW_COLS) -- never widen PHASE_21_RAW_COLS itself to
# match, that would reintroduce the exact OOM this script already fixed.
PHASE21_RAW_MART_SCHEMA_COLS = sorted(set(
    ["account_balance", "average_balance", "commission"]
    + CLUSTER_PEER_COLS
    + ["cust_1m", "cust_3m", "cust_6m", "vol_1m", "vol_3m", "vol_6m"]
    + [
        f"{txn_type}_{metric}_{horizon}"
        for txn_type in ("cash_out", "cash_in", "payment")
        for metric in ("vol", "value", "peers", "comm", "cust")
        for horizon in ("1m", "3m", "6m")
    ]
))

# Every name restate_phase_2_1_features() can produce, enumerated
# block-for-block from that function (mirrors transaction_features.py's
# run_phase_2_1_richer_tx_behaviour verbatim) -- union of the FULL raw
# mart schema (PHASE21_RAW_MART_SCHEMA_COLS, not the narrower
# PHASE_21_RAW_COLS) and every derived column name.
_PHASE21_BASE_NAMES = ("cash_out_vol", "cash_in_vol", "payment_vol", "vol")
PHASE21_PIT_FEATURE_NAMES = set(PHASE21_RAW_MART_SCHEMA_COLS) | {
    "is_fully_inactive_6m", "is_consecutively_inactive", "activity_restart_flag",
    "vol_3m_if_active", "commission_without_activity_flag",
    "consistent_volume_decline_flag", "consistent_volume_growth_flag",
    "low_balance_flag", "balance_to_vol_3m_ratio", "avg_balance_to_vol_3m_ratio",
    "balance_drawdown_flag", "cust_concentration_flag", "peer_dependency_ratio",
    "high_peer_dependency_flag", "net_cash_flow_3m", "net_cash_flow_negative_flag",
    "payment_intensity_ratio", "sharp_volume_drop_flag", "commission_drop_flag",
    "commission_vs_cluster_mean_ratio", "commission_vs_cluster_mean_diff",
    "vol_3m_vs_cluster_mean_ratio", "vol_3m_vs_cluster_mean_diff",
    "commission_per_vol_3m", "cluster_commission_per_vol_3m",
    "commission_per_vol_vs_cluster_ratio",
    "is_inactive_1m", "is_inactive_3m", "is_inactive_6m",
    "num_inactive_horizons", "max_inactivity_horizon_flag", "days_since_snapshot",
} | {
    f"{base}_{suffix}"
    for base in _PHASE21_BASE_NAMES
    for suffix in (
        "avg_monthly_3m", "avg_monthly_6m", "share_1m_of_3m", "growth_1m_vs_prev2m",
        "share_3m_of_6m", "growth_3m_vs_prev3m", "share_1m_of_6m",
        "monthly_volatility_proxy", "monthly_volatility_cv",
    )
}

# Phase 2.2 (loan-history) canonical output names: the training-side names
# from pd_model.preprocessing.loan_history_features.SNAPSHOT_TO_TRAINING_
# COLUMN_MAP's values, the pass-through-unrenamed columns documented in
# that module's own comment, the thin-file/no-history derived flags from
# _apply_thin_file_flag_inplace, and prior_loans_per_observed_month
# (confirmed PIT-safe: data/loan_state_query_updated_materialized.txt:2039-
# 2044 computes it purely from p.prior_disbursement_count / p.months_
# since_first_observed_loan, both from the strictly-prior-joined CTE at
# lines 1987-1996 -- "prior_loan.disbursement_ts < current_loan.
# disbursement_ts"). Deliberately EXCLUDES bad_state -- that is the label
# itself (see PROHIBITED_LABEL_LEAKAGE_NAMES below), never reachable here.
#
# IMPORTANT: SNAPSHOT_TO_TRAINING_COLUMN_MAP is NOT an exhaustive list of
# Phase 2.2 output names -- per its own module comment, it only lists
# columns that need RENAMING between the scoring-time and training-time
# queries. A real champion-coverage run found 16 genuinely PIT-safe
# training-side sibling columns this missed entirely (same CTEs as the
# already-mapped columns, confirmed by direct read of
# data/loan_state_query_updated_materialized.txt's final SELECT,
# lines 2060-2299, and each one's own source CTE):
#   - scoring_state_features CTE (lines 1087+), same `s.` alias already
#     producing scoring_state_date/observed_prior_loan_count etc.:
#     active_loan_aging_bucket_at_scoring, scoring_state_loan_disbursed_ugx,
#     scoring_state_loan_repaid_ugx, scoring_state_loan_gross_repaid_ugx,
#     scoring_state_loan_fees_paid_ugx, scoring_state_loan_interest_paid_ugx,
#     scoring_state_loan_gross_repayment_ratio,
#     scoring_state_repayment_to_disbursement_event_ratio.
#   - prior_loan_features CTE (lines 1357+), same aggregation already
#     producing all_prior_loans_disbursed_ugx etc.:
#     historical_anomaly_open_loan_count, avg_prior_loan_principal_
#     repayment_ratio.
#   - last_closed_loan_features CTE (lines 1488+), same `closed_prior_
#     loans_ranked WHERE closed_loan_recency = 1` source already producing
#     max_tenure_days_last_5_closed_loans's sibling:
#     last_closed_loan_tenure_days, last_closed_loan_aging_bucket,
#     last_closed_loan_disbursed_ugx, last_closed_loan_repaid_ugx,
#     last_closed_loan_gross_repaid_ugx,
#     last_closed_loan_principal_repayment_ratio.
PHASE22_PIT_FEATURE_NAMES = set(SNAPSHOT_TO_TRAINING_COLUMN_MAP.values()) | {
    "has_observed_prior_state", "has_pre_window_history", "days_since_last_repayment",
    "avg_tenure_days_closed_loans", "max_tenure_days_last_5_closed_loans",
    "historical_anomaly_open_loan_rate", "consecutive_on_time_loans",
    "months_since_first_observed_loan", "prior_loan_count_180d",
    "prior_active_loan_months_180d",
    "has_ever_loan", "has_loan_history", "is_new_agent", "no_loan_history_flag",
    "thin_file_flag", "thin_file_pd_prior",
    "prior_loans_per_observed_month",
    "active_loan_aging_bucket_at_scoring", "scoring_state_loan_disbursed_ugx",
    "scoring_state_loan_repaid_ugx", "scoring_state_loan_gross_repaid_ugx",
    "scoring_state_loan_fees_paid_ugx", "scoring_state_loan_interest_paid_ugx",
    "scoring_state_loan_gross_repayment_ratio",
    "scoring_state_repayment_to_disbursement_event_ratio",
    "historical_anomaly_open_loan_count", "avg_prior_loan_principal_repayment_ratio",
    "last_closed_loan_tenure_days", "last_closed_loan_aging_bucket",
    "last_closed_loan_disbursed_ugx", "last_closed_loan_repaid_ugx",
    "last_closed_loan_gross_repaid_ugx", "last_closed_loan_principal_repayment_ratio",
}

# Columns describing the TARGET loan/decision event itself -- no history
# join needed at all. Verified against data/loan_state_query_updated_
# materialized.txt: disbursement_amount_ugx (lines 94-97, 2075) is
# TRY_CAST(d.disbursement_amount_ugx ...) on the current target loan's own
# record (tmp_target_loans d / current_loan); disbursement_hour (2006-2007)
# and disbursement_day_of_week (2009-2010) are EXTRACT(... FROM d.
# disbursement_ts); late_afternoon_disbursement (2012-2016) is a CASE on
# that same timestamp. None of these touch a prior-loan or mart row.
DECISION_CONTEXT_PIT_FEATURE_NAMES = {
    "disbursement_amount_ugx", "disbursement_hour", "disbursement_day_of_week",
    "late_afternoon_disbursement",
}

# Hybrids: a decision-context value (the current loan's own amount) divided
# by a strictly-prior-window Phase 2.2 aggregate. Verified at data/
# loan_state_query_updated_materialized.txt:2018-2037:
# d.disbursement_amount_ugx / p.avg_prior_loan_amount_{30,90,180}d, where
# avg_prior_loan_amount_Nd (lines 1919-1939) is itself restricted to
# prior_loan.loan_date >= DATE_ADD('day', -N, current_loan.loan_date) AND
# joined via the strict guard (lines 1989-1992) "prior_loan.disbursement_ts
# < current_loan.disbursement_ts" -- the same pre-disbursement discipline
# Phase 2.2 uses elsewhere.
DECISION_CONTEXT_PLUS_PHASE22_PIT_FEATURE_NAMES = {
    "current_to_avg_prior_loan_amount_30d",
    "current_to_avg_prior_loan_amount_90d",
    "current_to_avg_prior_loan_amount_180d",
}

# Future-derived / label columns -- never legitimately knowable
# pre-disbursement, checked FIRST and exact-name-only (never substring),
# regardless of which pipeline happens to emit the name. "bad_state" is
# the label derived from bad_state_3dpd_30d (_apply_bad_flags_loan_level_
# inplace); bad_state_3dpd_30d and bad_state_1dpd_7d are themselves the
# raw primary/secondary labels straight from the materialized SQL's
# "Targets" section (data/loan_state_query_updated_materialized.txt:2265-
# 2266) -- loan_history_features.py's own module docstring states
# bad_state_1dpd_7d "stays in the returned frame as a monitoring signal,
# but must never be used as a feature." None of these three names are
# ever legitimately a champion-selected feature.
PROHIBITED_LABEL_LEAKAGE_NAMES = set(LABEL_DIAGNOSTIC_COLUMNS) | {
    "bad_state", "bad_state_3dpd_30d", "bad_state_1dpd_7d",
}

# Champion-feature-coverage classification labels (checked in this order;
# categories 2-5 are tested as set membership, not sequential elif, so a
# genuine name collision across sets surfaces as AMBIGUOUS rather than
# being silently resolved by whichever check happened to run first).
COVERAGE_LABEL_LEAKAGE = "PROHIBITED_LABEL_LEAKAGE"
COVERAGE_PHASE21 = "PHASE21_PIT_RECONSTRUCTABLE"
COVERAGE_PHASE22 = "PHASE22_PIT_RECONSTRUCTABLE"
COVERAGE_DECISION_CONTEXT = "DECISION_CONTEXT_PIT_AVAILABLE"
COVERAGE_DECISION_CONTEXT_PLUS_PHASE22 = "DECISION_CONTEXT_PLUS_PHASE22_PIT"
COVERAGE_AMBIGUOUS = "AMBIGUOUS_MULTI_PATH_SOURCE"
COVERAGE_UNACCOUNTED = "UNACCOUNTED_FOR"
COVERAGE_NOT_REQUIRED_UPSTREAM_DERIVED = "NOT_REQUIRED_UPSTREAM_DERIVED"

STATUS_RECONSTRUCTED = "PIT_PHASE21_RECONSTRUCTED"
STATUS_UNAVAILABLE = "PIT_PHASE21_UNAVAILABLE_NO_PRIOR_SNAPSHOT"

# cluster_peer_pit_provenance values. "VERIFIED" is never set by this
# script -- it exists in the enum only for a future lineage audit to use.
PROVENANCE_NOT_APPLICABLE = "NOT_APPLICABLE"
PROVENANCE_NOT_VERIFIED = "NOT_VERIFIED"
PROVENANCE_VERIFIED = "VERIFIED"

# required_by_scoring_schema values (schema-level, broadcast per row).
REQUIRED_YES = "YES"
REQUIRED_NO = "NO"
REQUIRED_UNKNOWN = "UNKNOWN"

SCHEMA_CHECK_UNAVAILABLE = "UNKNOWN_SCHEMA_NOT_AVAILABLE"
SCHEMA_CHECK_NOT_REQUIRED = "NOT_REQUIRED_BY_CHAMPION"
SCHEMA_CHECK_REQUIRED_UNKNOWN = "REQUIRED_BY_CHAMPION_PROVENANCE_UNKNOWN"


def _read_csv_fast(path: Path) -> pd.DataFrame:
    """Restated from build_loan_episode_capacity_dataset.py (one-way
    scripts/ layering convention) -- prefers pyarrow for large files,
    falls back to the plain engine if unavailable."""
    try:
        return pd.read_csv(path, sep=",", encoding="utf-8-sig", engine="pyarrow")
    except (ImportError, ValueError):
        return pd.read_csv(path, sep=",", encoding="utf-8-sig")


def _parse_mart_date(series: pd.Series) -> pd.Series:
    """Restated verbatim from build_loan_episode_capacity_dataset.py.
    A numeric YYYYMMDD column (e.g. tbl_dt == 20260731 as int64) must be
    routed through an explicit format="%Y%m%d" parse -- a plain
    pd.to_datetime on a raw numeric Series silently collapses every
    distinct date to ~1970-01-01 with no error, confirmed directly in
    this exact repo's history."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def load_loan_episodes(path: Path) -> pd.DataFrame:
    """Restated subset of build_loan_episode_capacity_dataset.py's loader
    -- only the columns this script needs (disbursement_fid, agent_msisdn,
    loan_date)."""
    df = _read_csv_fast(path)
    if "agent_msisdn" not in df.columns:
        alt_col = next((c for c in ("msisdn", "phonenumber") if c in df.columns), None)
        if alt_col is not None:
            df = df.rename(columns={alt_col: "agent_msisdn"})
            print(f"NOTE: loan-training-file has no 'agent_msisdn' column -- using '{alt_col}' instead.")
    missing = [c for c in LOAN_REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")
    df = df.copy()
    df["loan_date"] = pd.to_datetime(df["loan_date"], errors="coerce")
    df["_id"] = digits(df["agent_msisdn"])
    n_bad_dates = int(df["loan_date"].isna().sum())
    if n_bad_dates:
        print(f"NOTE: {n_bad_dates} episode(s) dropped for unparseable loan_date.")
        df = df[df["loan_date"].notna()]
    return df


def load_mart(path: Path) -> pd.DataFrame:
    """Restated from build_loan_episode_capacity_dataset.py's load_mart,
    narrowed to PHASE_21_RAW_COLS (the whitelist of raw columns
    restate_phase_2_1_features can ever consume), NOT every raw mart
    column -- a real mart export can carry dozens of columns Phase 2.1
    never reads, and merging all of them through merge_asof's grouped
    join at real scale (millions of rows) is a genuine memory blowup, not
    merely wasteful. Narrowing here is a memory fix, not a correctness
    change: restate_phase_2_1_features's own per-block `if c in
    df.columns` gating treats a column dropped here identically to one
    that was simply never present in the export."""
    txn = _read_csv_fast(path)
    msisdn_col = "agent_msisdn" if "agent_msisdn" in txn.columns else "msisdn"
    if msisdn_col not in txn.columns:
        sys.exit(f"ERROR: {path} has neither 'agent_msisdn' nor 'msisdn'. Columns present: {list(txn.columns)}")
    date_col = "snapshot_dt" if "snapshot_dt" in txn.columns else ("tbl_dt" if "tbl_dt" in txn.columns else None)
    if date_col is None:
        sys.exit(f"ERROR: {path} has neither 'snapshot_dt' nor 'tbl_dt' -- cannot perform a "
                  f"point-in-time join without a snapshot date.")
    present_phase21_cols = [c for c in PHASE_21_RAW_COLS if c in txn.columns]
    missing_phase21_cols = [c for c in PHASE_21_RAW_COLS if c not in txn.columns]
    if missing_phase21_cols:
        print(f"NOTE: {path.name} is missing {len(missing_phase21_cols)} Phase 2.1 raw column(s) "
              f"-- the corresponding feature blocks will be skipped (defensive gating, not an "
              f"error): {missing_phase21_cols}")
    txn = txn[[msisdn_col, date_col] + present_phase21_cols].copy()
    txn["_id"] = digits(txn[msisdn_col])
    txn["fundamentals_snapshot_date"] = _parse_mart_date(txn[date_col]).dt.normalize()
    n_bad = int(txn["fundamentals_snapshot_date"].isna().sum())
    if n_bad:
        print(f"NOTE: {n_bad} mart row(s) dropped for unparseable {date_col}.")
        txn = txn[txn["fundamentals_snapshot_date"].notna()]
    n_dates = txn["fundamentals_snapshot_date"].nunique()
    print(f"Transaction mart: {txn['_id'].nunique():,} unique agent(s) across {n_dates:,} "
          f"distinct snapshot date(s) ({sorted(d.date().isoformat() for d in txn['fundamentals_snapshot_date'].unique())}), "
          f"narrowed to {len(present_phase21_cols)} Phase-2.1-relevant raw column(s).")
    return txn[["_id", "fundamentals_snapshot_date"] + present_phase21_cols]


def attach_prior_snapshot(episodes: pd.DataFrame, mart: pd.DataFrame) -> pd.DataFrame:
    """The restated merge_asof pattern verbatim (both the primary call and
    the TypeError fallback) from attach_pretrade_fundamentals. Adds
    `days_since_snapshot` (loan_date - fundamentals_snapshot_date, in
    days -- computed per row from that row's OWN loan_date, never a batch
    max) and `reconstruction_status` -- never silently backfilled."""
    ep = episodes.sort_values("loan_date").reset_index(drop=True)
    mt = mart.sort_values("fundamentals_snapshot_date").reset_index(drop=True)
    mart_value_cols = [c for c in mt.columns if c not in ("_id", "fundamentals_snapshot_date")]

    try:
        merged = pd.merge_asof(
            ep, mt, left_on="loan_date", right_on="fundamentals_snapshot_date",
            left_by="_id", right_by="_id", direction="backward", allow_exact_matches=False,
        )
        join_method = "merge_asof(direction='backward', allow_exact_matches=False)"
    except TypeError:
        merged = pd.merge_asof(
            ep, mt, left_on="loan_date", right_on="fundamentals_snapshot_date",
            left_by="_id", right_by="_id", direction="backward",
        )
        exact_match = merged["fundamentals_snapshot_date"] == merged["loan_date"]
        for c in ["fundamentals_snapshot_date"] + mart_value_cols:
            merged.loc[exact_match, c] = np.nan
        join_method = "merge_asof(direction='backward') + explicit exact-match nulling"
    print(f"Fundamentals join method: {join_method}")

    merged["days_since_snapshot"] = (merged["loan_date"] - merged["fundamentals_snapshot_date"]).dt.days
    merged["reconstruction_status"] = np.where(
        merged["fundamentals_snapshot_date"].notna(), STATUS_RECONSTRUCTED, STATUS_UNAVAILABLE,
    )
    n_unavailable = int((merged["reconstruction_status"] == STATUS_UNAVAILABLE).sum())
    if n_unavailable:
        print(f"NOTE: {n_unavailable:,} of {len(merged):,} episode(s) "
              f"({n_unavailable / len(merged) * 100:.1f}%) have no eligible prior mart snapshot -- "
              f"status={STATUS_UNAVAILABLE}, features left NaN, never backfilled.")
    return merged


def assert_no_leakage(df: pd.DataFrame) -> None:
    """The single most load-bearing check in this script. Raises loudly
    (never a warning) if any reconstructed row has a snapshot on or after
    its own loan_date."""
    reconstructed = df[df["reconstruction_status"] == STATUS_RECONSTRUCTED]
    violations = reconstructed[reconstructed["fundamentals_snapshot_date"] >= reconstructed["loan_date"]]
    if len(violations):
        raise AssertionError(
            f"LEAKAGE DETECTED: {len(violations)} reconstructed row(s) have "
            f"fundamentals_snapshot_date >= loan_date. This must never happen -- aborting."
        )
    print(f"Leakage assertion passed: all {len(reconstructed):,} reconstructed row(s) have "
          f"fundamentals_snapshot_date < loan_date.")


def restate_phase_2_1_features(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Block-for-block restatement of pd_model/preprocessing/
    transaction_features.py's run_phase_2_1_richer_tx_behaviour, with the
    ONE deliberate change: `days_since_snapshot` is taken from the as-of
    join (already relative to this episode's own loan_date), never
    recomputed from a batch-wide tbl_dt.max(). Every other block is
    copied verbatim: same conditions, same formulas, same defensive
    `if all(c in df.columns ...)` gating per block."""
    df = df.copy()

    # A) Strong inactivity structure.
    if all(c in df.columns for c in ["vol_1m", "vol_3m", "vol_6m"]):
        df["is_fully_inactive_6m"] = (
            (df["vol_1m"].fillna(0) == 0) & (df["vol_3m"].fillna(0) == 0) & (df["vol_6m"].fillna(0) == 0)
        ).astype(int)
        df["is_consecutively_inactive"] = (
            (df["vol_1m"].fillna(0) == 0) & (df["vol_3m"].fillna(0) == 0)
        ).astype(int)

    # B) Activity restart / recovery signal.
    if all(c in df.columns for c in ["vol_1m", "vol_3m"]):
        df["activity_restart_flag"] = ((df["vol_1m"] > 0) & (df["vol_3m"] == df["vol_1m"])).astype(int)

    # C) Conditional activity intensity.
    if "vol_3m" in df.columns:
        df["vol_3m_if_active"] = df["vol_3m"].where(df["vol_3m"] > 0, np.nan)

    # D) Commission dependency risk.
    if all(c in df.columns for c in ["commission", "vol_3m"]):
        df["commission_without_activity_flag"] = (
            (df["commission"] > 0) & (df["vol_3m"].fillna(0) == 0)
        ).astype(int)

    # Trend direction flags.
    if all(c in df.columns for c in ["vol_1m", "vol_3m", "vol_6m"]):
        df["consistent_volume_decline_flag"] = (
            (df["vol_1m"] < df["vol_3m"] / 3.0) & (df["vol_3m"] < df["vol_6m"] / 2.0)
        ).astype(int)
        df["consistent_volume_growth_flag"] = (
            (df["vol_1m"] > df["vol_3m"] / 3.0) & (df["vol_3m"] > df["vol_6m"] / 2.0)
        ).astype(int)

    # Liquidity & balance stress.
    if "account_balance" in df.columns:
        df["low_balance_flag"] = (df["account_balance"] <= 0).astype(int)
    if all(c in df.columns for c in ["account_balance", "vol_3m"]):
        df["balance_to_vol_3m_ratio"] = df["account_balance"] / (df["vol_3m"] + eps)
    if all(c in df.columns for c in ["average_balance", "vol_3m"]):
        df["avg_balance_to_vol_3m_ratio"] = df["average_balance"] / (df["vol_3m"] + eps)
    if all(c in df.columns for c in ["account_balance", "average_balance"]):
        df["balance_drawdown_flag"] = (df["account_balance"] < 0.5 * df["average_balance"]).astype(int)

    # Customer & peer dependence.
    if all(c in df.columns for c in ["cust_1m", "cust_3m"]):
        df["cust_concentration_flag"] = ((df["cust_1m"] / (df["cust_3m"] + eps)) > 0.8).astype(int)
    if all(c in df.columns for c in ["cash_in_peers_3m", "cash_in_vol_3m"]):
        df["peer_dependency_ratio"] = df["cash_in_peers_3m"] / (df["cash_in_vol_3m"] + eps)
        df["high_peer_dependency_flag"] = (df["peer_dependency_ratio"] > 0.7).astype(int)

    # Transaction mix & net flow stress.
    if all(c in df.columns for c in ["cash_in_value_3m", "cash_out_value_3m"]):
        df["net_cash_flow_3m"] = df["cash_in_value_3m"] - df["cash_out_value_3m"]
        df["net_cash_flow_negative_flag"] = (df["net_cash_flow_3m"] < 0).astype(int)
    if all(c in df.columns for c in ["payment_value_3m", "vol_3m"]):
        df["payment_intensity_ratio"] = df["payment_value_3m"] / (df["vol_3m"] + eps)

    # Stress acceleration flags.
    if all(c in df.columns for c in ["vol_1m", "vol_3m"]):
        df["sharp_volume_drop_flag"] = ((df["vol_1m"] / (df["vol_3m"] + eps)) < 0.3).astype(int)
    if all(c in df.columns for c in ["commission", "commission_cluster_mean"]):
        df["commission_drop_flag"] = (df["commission"] < 0.5 * df["commission_cluster_mean"]).astype(int)

    # 1) Cluster-relative commission and volume.
    if "commission" in df.columns and "commission_cluster_mean" in df.columns:
        df["commission_vs_cluster_mean_ratio"] = df["commission"] / (df["commission_cluster_mean"] + eps)
        df["commission_vs_cluster_mean_diff"] = df["commission"] - df["commission_cluster_mean"]
    if "vol_3m" in df.columns and "vol_3m_cluster_mean" in df.columns:
        df["vol_3m_vs_cluster_mean_ratio"] = df["vol_3m"] / (df["vol_3m_cluster_mean"] + eps)
        df["vol_3m_vs_cluster_mean_diff"] = df["vol_3m"] - df["vol_3m_cluster_mean"]

    # 2) Commission intensity (per volume).
    if "commission" in df.columns and "vol_3m" in df.columns:
        df["commission_per_vol_3m"] = df["commission"] / (df["vol_3m"] + eps)
    if (
        "cluster_avg_commission" in df.columns
        and "cluster_avg_vol_3m" in df.columns
        and "commission_per_vol_3m" in df.columns
    ):
        df["cluster_commission_per_vol_3m"] = df["cluster_avg_commission"] / (df["cluster_avg_vol_3m"] + eps)
        df["commission_per_vol_vs_cluster_ratio"] = df["commission_per_vol_3m"] / (
            df["cluster_commission_per_vol_3m"] + eps
        )

    # 3) Volume trajectory, intensity and volatility (1m / 3m / 6m).
    base_names = ["cash_out_vol", "cash_in_vol", "payment_vol", "vol"]
    for base in base_names:
        col_1m, col_3m, col_6m = f"{base}_1m", f"{base}_3m", f"{base}_6m"
        have_1m, have_3m, have_6m = (col_1m in df.columns, col_3m in df.columns, col_6m in df.columns)

        if have_3m:
            df[f"{base}_avg_monthly_3m"] = df[col_3m] / 3.0
        if have_6m:
            df[f"{base}_avg_monthly_6m"] = df[col_6m] / 6.0
        if have_1m and have_3m:
            df[f"{base}_share_1m_of_3m"] = df[col_1m] / (df[col_3m] + eps)
            prev2m = (df[col_3m] - df[col_1m]) / 2.0
            df[f"{base}_growth_1m_vs_prev2m"] = df[col_1m] / (prev2m + eps)
        if have_3m and have_6m:
            df[f"{base}_share_3m_of_6m"] = df[col_3m] / (df[col_6m] + eps)
            prev3m = (df[col_6m] - df[col_3m]) / 3.0
            df[f"{base}_growth_3m_vs_prev3m"] = df[col_3m] / (prev3m + eps)
        if have_1m and have_6m:
            df[f"{base}_share_1m_of_6m"] = df[col_1m] / (df[col_6m] + eps)
        if have_1m and have_3m and have_6m:
            m1 = df[col_1m]
            m2 = (df[col_3m] - df[col_1m]) / 2.0
            m3 = (df[col_6m] - df[col_3m]) / 3.0
            monthly = np.vstack([m1.values, m2.values, m3.values]).T
            mean_monthly = monthly.mean(axis=1)
            std_monthly = monthly.std(axis=1)
            df[f"{base}_monthly_volatility_proxy"] = std_monthly
            df[f"{base}_monthly_volatility_cv"] = std_monthly / (mean_monthly + eps)

    # 4) Explicit inactivity flags per horizon.
    for horizon in ["1m", "3m", "6m"]:
        col, flag_col = f"vol_{horizon}", f"is_inactive_{horizon}"
        if col in df.columns:
            df[flag_col] = (df[col].fillna(0) == 0).astype(int)
    inactivity_flag_cols = [c for c in ["is_inactive_1m", "is_inactive_3m", "is_inactive_6m"] if c in df.columns]
    if inactivity_flag_cols:
        df["num_inactive_horizons"] = df[inactivity_flag_cols].sum(axis=1)
    if all(c in df.columns for c in ["is_inactive_1m", "is_inactive_3m", "is_inactive_6m"]):
        df["max_inactivity_horizon_flag"] = df["is_inactive_1m"] + df["is_inactive_3m"] + df["is_inactive_6m"]

    # 5) Recency -- THE ONE DELIBERATE CHANGE: already computed per-row by
    # attach_prior_snapshot from this episode's own loan_date, never a
    # batch-wide tbl_dt.max(). Nothing to do here except confirm presence.
    if "days_since_snapshot" not in df.columns:
        df["days_since_snapshot"] = np.nan

    # Upstream cluster/peer provenance: THREE separate, independent facts,
    # never collapsed into one combined verdict -- presence alone cannot
    # establish PIT validity, and "required_by_scoring_schema" is a
    # schema-level fact, not a per-row assessment.
    present_cluster_cols = [c for c in CLUSTER_PEER_COLS if c in df.columns]
    if present_cluster_cols:
        df["cluster_peer_columns_present"] = df[present_cluster_cols].notna().any(axis=1)
    else:
        df["cluster_peer_columns_present"] = False
    # "NOT_VERIFIED" -- never a default pass -- whenever a cluster/peer
    # column was present, because this script does not audit those
    # columns' own point-in-time construction. "NOT_APPLICABLE" when none
    # were present (no derived feature exists to be in question).
    df["cluster_peer_pit_provenance"] = np.where(
        df["cluster_peer_columns_present"], PROVENANCE_NOT_VERIFIED, PROVENANCE_NOT_APPLICABLE,
    )

    return df


def check_upstream_feature_schema_overlap(pd_model_artifacts_dir: Path | None) -> dict:
    """Schema-level (not per-row) check: does the champion model actually
    REQUIRE any of the cluster/peer-derived features this script produces?
    Narrows whether the per-row NOT_VERIFIED provenance is merely
    theoretical or actually blocks confident scoring -- per the eventual
    Gate-1 rule (not computed here): Phase21PIT = RowPIT AND
    RequiredFeatureCoverage AND RequiredUpstreamFeaturePIT, where a
    NOT_VERIFIED provenance only blocks when required_by_scoring_schema
    == YES. Never claims a derived feature is PIT-verified -- only ever
    reports whether the question is moot (not required), live (required,
    provenance unknown), or undeterminable (no populated champion schema
    exists yet -- the real repo's artifacts are currently placeholders).
    `required_by_scoring_schema` (YES/NO/UNKNOWN) is the simplified,
    row-broadcastable form of `status` below."""
    if pd_model_artifacts_dir is None:
        return {"status": SCHEMA_CHECK_UNAVAILABLE, "required_by_scoring_schema": REQUIRED_UNKNOWN,
                "overlap": [], "n_selected_features": None, "detail": "no --pd-model-artifacts-dir given"}
    path = Path(pd_model_artifacts_dir) / "feature_order.json"
    try:
        import json
        data = json.loads(path.read_text())
    except Exception as e:
        return {"status": SCHEMA_CHECK_UNAVAILABLE, "required_by_scoring_schema": REQUIRED_UNKNOWN,
                "overlap": [], "n_selected_features": None, "detail": f"could not read {path}: {e}"}
    selected_features = data.get("selected_features") or []
    if not selected_features:
        return {"status": SCHEMA_CHECK_UNAVAILABLE, "required_by_scoring_schema": REQUIRED_UNKNOWN,
                "overlap": [], "n_selected_features": 0,
                "detail": f"{path} has no selected_features (placeholder/unpopulated champion schema)"}
    overlap = sorted(set(selected_features) & set(CLUSTER_DERIVED_FEATURE_NAMES))
    if overlap:
        return {"status": SCHEMA_CHECK_REQUIRED_UNKNOWN, "required_by_scoring_schema": REQUIRED_YES,
                "overlap": overlap, "n_selected_features": len(selected_features),
                "detail": f"{len(overlap)} cluster-derived feature(s) required by the champion "
                          f"schema, with unresolved upstream PIT provenance"}
    return {"status": SCHEMA_CHECK_NOT_REQUIRED, "required_by_scoring_schema": REQUIRED_NO,
            "overlap": [], "n_selected_features": len(selected_features),
            "detail": "none of the cluster-derived features are in the champion's selected_features"}


def _load_selected_features(pd_model_artifacts_dir: Path | None) -> tuple[list[str] | None, str]:
    """Shared loader for the coverage check -- same read path and same
    unavailable/empty handling as check_upstream_feature_schema_overlap,
    factored out so both checks treat a missing/placeholder feature_
    order.json identically. Returns (selected_features_or_None, detail)."""
    if pd_model_artifacts_dir is None:
        return None, "no --pd-model-artifacts-dir given"
    path = Path(pd_model_artifacts_dir) / "feature_order.json"
    try:
        import json
        data = json.loads(path.read_text())
    except Exception as e:
        return None, f"could not read {path}: {e}"
    selected_features = data.get("selected_features") or []
    if not selected_features:
        return None, f"{path} has no selected_features (placeholder/unpopulated champion schema)"
    return list(selected_features), f"loaded {len(selected_features)} selected_features from {path}"


def classify_champion_feature(name: str) -> tuple[str, str, str, bool]:
    """Classifies a single selected_feature name into exactly one of the 7
    RequiredChampionFeaturePITCoverage categories. Order matters:
    PROHIBITED_LABEL_LEAKAGE is checked FIRST (exact-name match only,
    never substring) so a future-derived/label column is never classified
    PIT-reconstructable merely because some pipeline happens to emit the
    name -- pipeline membership is not evidence of PIT safety. The four
    legitimate-lineage sets are then tested as SET MEMBERSHIP (not
    sequential elif), so a name present in more than one is caught as
    AMBIGUOUS_MULTI_PATH_SOURCE and reported, never silently assigned to
    whichever check happened to run first.

    Returns (classification, pit_reconstructable, evidence, blocking)."""
    if name in PROHIBITED_LABEL_LEAKAGE_NAMES:
        return (
            COVERAGE_LABEL_LEAKAGE, "NO",
            "matches bad_state or a LABEL_DIAGNOSTIC_COLUMNS name (future-derived, label leakage)",
            True,
        )

    matched_sets = []
    if name in PHASE21_PIT_FEATURE_NAMES:
        matched_sets.append(COVERAGE_PHASE21)
    if name in PHASE22_PIT_FEATURE_NAMES:
        matched_sets.append(COVERAGE_PHASE22)
    if name in DECISION_CONTEXT_PIT_FEATURE_NAMES:
        matched_sets.append(COVERAGE_DECISION_CONTEXT)
    if name in DECISION_CONTEXT_PLUS_PHASE22_PIT_FEATURE_NAMES:
        matched_sets.append(COVERAGE_DECISION_CONTEXT_PLUS_PHASE22)

    if len(matched_sets) > 1:
        return (
            COVERAGE_AMBIGUOUS, "UNKNOWN",
            f"name appears in multiple canonical PIT-lineage sets: {matched_sets} -- "
            f"reported explicitly, never silently assigned to one",
            True,
        )
    if len(matched_sets) == 1:
        classification = matched_sets[0]
        evidence = {
            COVERAGE_PHASE21: "in PHASE21_PIT_FEATURE_NAMES (restated run_phase_2_1_richer_tx_behaviour "
                               "raw-passthrough + derived output names)",
            COVERAGE_PHASE22: "in PHASE22_PIT_FEATURE_NAMES (loan_history_features.py's "
                               "SNAPSHOT_TO_TRAINING_COLUMN_MAP values + pass-through + derived-flag names)",
            COVERAGE_DECISION_CONTEXT: "in DECISION_CONTEXT_PIT_FEATURE_NAMES (target loan's own "
                                        "request/disbursement record -- no history join needed)",
            COVERAGE_DECISION_CONTEXT_PLUS_PHASE22: "in DECISION_CONTEXT_PLUS_PHASE22_PIT_FEATURE_NAMES "
                                                     "(current-loan value / strictly-prior Phase 2.2 aggregate)",
        }[classification]
        if classification == COVERAGE_PHASE21 and name in CLUSTER_DERIVED_FEATURE_NAMES:
            evidence += (" -- NOTE: this is one of the 9 cluster-derived names; its upstream "
                         "cluster/peer PIT provenance is a SEPARATE, still-NOT_VERIFIED caveat "
                         "(see cluster_peer_pit_provenance), not resolved by this classification")
        return classification, "YES", evidence, False

    return COVERAGE_UNACCOUNTED, "NO", "name is not in any canonical PIT-lineage set", True


def check_champion_feature_coverage(pd_model_artifacts_dir: Path | None) -> dict:
    """Classifies EVERY champion selected_feature into exactly one of the 7
    categories (see classify_champion_feature) -- the broader follow-up to
    check_upstream_feature_schema_overlap, which only answers whether the
    9 cluster-derived names are required. This answers: can every selected
    feature be assigned to a legitimate, code-verified PIT reconstruction
    lineage? Classification is by PROVENANCE OF GENERATION PATH, never by
    whether a same-named column happens to exist in some mart/history
    export.

    Returns {"status", "n_selected_features", "rows": [...], "counts": {...},
    "required_feature_coverage": "PASS"|"FAIL"|None,
    "not_required_upstream_derived": [...], "detail"}.

    "RequiredChampionFeaturePITCoverage" (the narrative name for this
    result, used in printed banners) == PASS means every feature THIS
    CHAMPION actually selected has an identified, code-supported decision-
    time lineage with zero label-leaked, unresolved, or ambiguous
    features -- it does not audit the full universe of features either
    pipeline is capable of generating, only the subset this champion uses.
    required_feature_coverage is None (never a fabricated PASS/FAIL) when
    no populated champion schema exists yet."""
    selected_features, detail = _load_selected_features(pd_model_artifacts_dir)
    if selected_features is None:
        return {
            "status": SCHEMA_CHECK_UNAVAILABLE, "n_selected_features": None,
            "rows": [], "counts": {}, "required_feature_coverage": None,
            "not_required_upstream_derived": [], "detail": detail,
        }

    rows = []
    counts = {
        COVERAGE_LABEL_LEAKAGE: 0, COVERAGE_PHASE21: 0, COVERAGE_PHASE22: 0,
        COVERAGE_DECISION_CONTEXT: 0, COVERAGE_DECISION_CONTEXT_PLUS_PHASE22: 0,
        COVERAGE_AMBIGUOUS: 0, COVERAGE_UNACCOUNTED: 0,
    }
    for name in selected_features:
        classification, pit_reconstructable, evidence, blocking = classify_champion_feature(name)
        counts[classification] += 1
        rows.append({
            "selected_feature": name,
            "feature_source": classification,
            "pit_reconstructable": pit_reconstructable,
            "evidence": evidence,
            "blocking": "YES" if blocking else "NO",
            "detail": "",
        })

    n_selected = len(selected_features)
    assert sum(counts.values()) == n_selected, (
        f"INTERNAL ERROR: classification counts ({sum(counts.values())}) do not sum to "
        f"n_selected_features ({n_selected}) -- every selected feature must land in exactly "
        f"one of the 7 categories."
    )

    not_required_upstream_derived = sorted(set(CLUSTER_DERIVED_FEATURE_NAMES) - set(selected_features))
    for name in not_required_upstream_derived:
        rows.append({
            "selected_feature": name,
            "feature_source": COVERAGE_NOT_REQUIRED_UPSTREAM_DERIVED,
            "pit_reconstructable": "N/A",
            "evidence": "cluster-derived Phase 2.1 output name, NOT in the champion's selected_features",
            "blocking": "NO",
            "detail": "informational only -- excluded from the n_selected_features coverage sum",
        })

    required_feature_coverage = "PASS" if (
        counts[COVERAGE_UNACCOUNTED] == 0
        and counts[COVERAGE_LABEL_LEAKAGE] == 0
        and counts[COVERAGE_AMBIGUOUS] == 0
    ) else "FAIL"

    return {
        "status": "OK", "n_selected_features": n_selected, "rows": rows, "counts": counts,
        "required_feature_coverage": required_feature_coverage,
        "not_required_upstream_derived": not_required_upstream_derived,
        "detail": detail,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loan-training-file", type=Path, default=None,
                     help="required unless --schema-check-only is given")
    ap.add_argument("--transaction-mart-file", type=Path, default=None,
                     help="required unless --schema-check-only is given")
    ap.add_argument("--pd-model-artifacts-dir", type=Path, default=None,
                     help="optional; expects feature_order.json. Enables the schema-overlap check "
                          "(whether the champion model actually requires any cluster-derived feature).")
    ap.add_argument("--schema-check-only", action="store_true",
                     help="skip the loan/mart reconstruction entirely (no episodes loaded, no join, "
                          "no leakage assertion, no features CSV) and only run the schema-overlap "
                          "check against --pd-model-artifacts-dir, printing the result in seconds. "
                          "Requires --pd-model-artifacts-dir. Use this to get required_by_scoring_"
                          "schema without re-running the full, multi-GB-output reconstruction just "
                          "to see that one answer.")
    ap.add_argument("--champion-feature-coverage-only", action="store_true",
                     help="skip the loan/mart reconstruction entirely (same shape as "
                          "--schema-check-only) and classify EVERY champion selected_feature into "
                          "one of 7 PIT-lineage categories (PROHIBITED_LABEL_LEAKAGE, "
                          "PHASE21_PIT_RECONSTRUCTABLE, PHASE22_PIT_RECONSTRUCTABLE, "
                          "DECISION_CONTEXT_PIT_AVAILABLE, DECISION_CONTEXT_PLUS_PHASE22_PIT, "
                          "AMBIGUOUS_MULTI_PATH_SOURCE, UNACCOUNTED_FOR). Requires "
                          "--pd-model-artifacts-dir. Answers whether every selected feature has an "
                          "identified, code-verified decision-time lineage -- never whether a "
                          "same-named column merely exists somewhere in a mart/history export.")
    ap.add_argument("--out-prefix", type=str, default="stage4a1_phase21_pit")
    args = ap.parse_args(argv)

    if args.schema_check_only:
        if args.pd_model_artifacts_dir is None:
            sys.exit("ERROR: --schema-check-only requires --pd-model-artifacts-dir.")
        print(f"\n{'#' * 100}\nSTAGE 4A.1 -- SCHEMA-OVERLAP CHECK ONLY (no loan/mart reconstruction, "
              f"no features CSV)\n{'#' * 100}")
        schema_check = check_upstream_feature_schema_overlap(args.pd_model_artifacts_dir)
        print(f"Status: {schema_check['status']}")
        print(f"required_by_scoring_schema: {schema_check['required_by_scoring_schema']}")
        print(f"Detail: {schema_check['detail']}")
        if schema_check["overlap"]:
            print(f"Overlapping feature(s): {schema_check['overlap']}")
        schema_check_path = f"{args.out_prefix}_phase21_upstream_schema_check.json"
        import json
        Path(schema_check_path).write_text(json.dumps(schema_check, indent=2))
        print(f"Wrote {schema_check_path}")
        return

    if args.champion_feature_coverage_only:
        if args.pd_model_artifacts_dir is None:
            sys.exit("ERROR: --champion-feature-coverage-only requires --pd-model-artifacts-dir.")
        print(f"\n{'#' * 100}\nSTAGE 4A.1 -- CHAMPION FEATURE COVERAGE CHECK ONLY (no loan/mart "
              f"reconstruction, no features CSV)\nClassifies every champion selected_feature by "
              f"PROVENANCE OF GENERATION PATH, never by column-name coincidence with a mart/history "
              f"export.\n{'#' * 100}")
        schema_check = check_upstream_feature_schema_overlap(args.pd_model_artifacts_dir)
        print(f"(Context) cluster-derived schema-overlap status: {schema_check['status']} "
              f"/ required_by_scoring_schema={schema_check['required_by_scoring_schema']}")

        coverage = check_champion_feature_coverage(args.pd_model_artifacts_dir)
        print(f"Status: {coverage['status']}")
        print(f"n_selected_features: {coverage['n_selected_features']}")
        print(f"Detail: {coverage['detail']}")
        if coverage["counts"]:
            print("-- Classification counts --")
            for label, n in coverage["counts"].items():
                print(f"  {label}: {n}")
            print(f"  {COVERAGE_NOT_REQUIRED_UPSTREAM_DERIVED} (informational, excluded from sum): "
                  f"{len(coverage['not_required_upstream_derived'])}")
        print(f"RequiredChampionFeaturePITCoverage: {coverage['required_feature_coverage']}")

        coverage_csv_path = f"{args.out_prefix}_champion_feature_coverage.csv"
        pd.DataFrame(
            coverage["rows"],
            columns=["selected_feature", "feature_source", "pit_reconstructable", "evidence",
                     "blocking", "detail"],
        ).to_csv(coverage_csv_path, index=False)
        print(f"Wrote {coverage_csv_path}")

        coverage_json_path = f"{args.out_prefix}_champion_feature_coverage.json"
        import json
        counts = coverage["counts"]
        json_summary = {
            "champion_selected_features": coverage["n_selected_features"],
            "phase21_pit_covered": counts.get(COVERAGE_PHASE21),
            "phase22_pit_covered": counts.get(COVERAGE_PHASE22),
            "decision_context_covered": counts.get(COVERAGE_DECISION_CONTEXT),
            "decision_context_plus_phase22_covered": counts.get(COVERAGE_DECISION_CONTEXT_PLUS_PHASE22),
            "ambiguous_multi_path": counts.get(COVERAGE_AMBIGUOUS),
            "label_leakage_suspected": counts.get(COVERAGE_LABEL_LEAKAGE),
            "unaccounted_for": counts.get(COVERAGE_UNACCOUNTED),
            "not_required_upstream_derived_count": len(coverage["not_required_upstream_derived"]),
            "required_feature_coverage": coverage["required_feature_coverage"],
            "detail": coverage["detail"],
        }
        Path(coverage_json_path).write_text(json.dumps(json_summary, indent=2))
        print(f"Wrote {coverage_json_path}")
        return

    if args.loan_training_file is None or args.transaction_mart_file is None:
        sys.exit("ERROR: --loan-training-file and --transaction-mart-file are required unless "
                  "--schema-check-only or --champion-feature-coverage-only is given.")

    print(f"\n{'#' * 100}\nSTAGE 4A.1 -- RESEARCH-ONLY PHASE 2.1 PIT FEATURE RECONSTRUCTION\n"
          f"Produces NO PD scores, NO C3 application, no k selection, no Capacity(F) combination.\n"
          f"Never modifies production pd_model/preprocessing/transaction_features.py.\n"
          f"Reports THREE separate upstream cluster/peer facts (present, pit_provenance,\n"
          f"required_by_scoring_schema), never a single collapsed verdict. Does NOT compute the\n"
          f"eventual Phase21PIT = RowPIT AND RequiredFeatureCoverage AND RequiredUpstreamFeaturePIT\n"
          f"gate -- that belongs to a later Stage 4A Gate-1 update, not this script.\n{'#' * 100}")

    if not args.loan_training_file.exists():
        sys.exit(f"ERROR: {args.loan_training_file} not found.")
    if not args.transaction_mart_file.exists():
        sys.exit(f"ERROR: {args.transaction_mart_file} not found.")

    episodes = load_loan_episodes(args.loan_training_file)
    print(f"{len(episodes):,} loan episode(s) loaded.")
    mart = load_mart(args.transaction_mart_file)

    merged = attach_prior_snapshot(episodes, mart)
    assert_no_leakage(merged)
    featured = restate_phase_2_1_features(merged, DEFAULT_CONFIG.eps)

    status_counts = featured["reconstruction_status"].value_counts()
    print("\n-- Reconstruction status (RowPIT -- as-of join validity) --")
    print(status_counts.to_string())

    provenance_counts = featured["cluster_peer_pit_provenance"].value_counts()
    print("\n-- cluster_peer_pit_provenance (never VERIFIED by this script) --")
    print(provenance_counts.to_string())

    age_bins = [-1, 7, 30, 60, 90, np.inf]
    age_labels = ["0-7d", "8-30d", "31-60d", "61-90d", ">90d"]
    reconstructed = featured[featured["reconstruction_status"] == STATUS_RECONSTRUCTED]
    age_band_counts = pd.cut(reconstructed["days_since_snapshot"], bins=age_bins, labels=age_labels).value_counts()
    print("\n-- days_since_snapshot age bands (reconstructed rows only) --")
    print(age_band_counts.sort_index().to_string())

    schema_check = check_upstream_feature_schema_overlap(args.pd_model_artifacts_dir)
    featured["required_by_scoring_schema"] = schema_check["required_by_scoring_schema"]
    print(f"\n{'#' * 100}\nSCHEMA-OVERLAP CHECK: does the champion model actually require a "
          f"cluster-derived feature?\n{'#' * 100}")
    print(f"Status: {schema_check['status']}")
    print(f"required_by_scoring_schema: {schema_check['required_by_scoring_schema']}")
    print(f"Detail: {schema_check['detail']}")
    if schema_check["overlap"]:
        print(f"Overlapping feature(s): {schema_check['overlap']}")
    if schema_check["status"] == SCHEMA_CHECK_UNAVAILABLE:
        print("Cannot yet determine whether NOT_VERIFIED provenance actually blocks scoring -- "
              "re-run with --pd-model-artifacts-dir pointing at a populated champion "
              "feature_order.json once one exists.")
    elif schema_check["status"] == SCHEMA_CHECK_NOT_REQUIRED:
        print("The champion's selected_features do not include any cluster-derived feature -- the "
              "unresolved upstream provenance question is moot for scoring with this exact model.")
    else:
        print("The champion's selected_features REQUIRE at least one cluster-derived feature whose "
              "upstream provenance remains NOT_VERIFIED -- scoring with this exact model would "
              "depend on an unverified assumption. This is not resolved by this script.")

    print(f"\n{'#' * 100}\nCONCLUSION (the only claim this script's result supports)\n{'#' * 100}")
    print(
        "Historical Phase 2.1 row selection and row-derived feature reconstruction succeeded.\n"
        "Full Phase 2.1 scoring-vector PIT validity remains conditional on required-feature\n"
        "coverage and PIT provenance of any required upstream-derived peer/cluster inputs."
    )

    feature_cols = [c for c in featured.columns if c not in merged.columns or c == "days_since_snapshot"]
    fixed_cols = ["disbursement_fid", "agent_msisdn", "loan_date", "fundamentals_snapshot_date",
                  "days_since_snapshot", "reconstruction_status", "cluster_peer_columns_present",
                  "cluster_peer_pit_provenance", "required_by_scoring_schema"]
    out_cols = fixed_cols + [c for c in feature_cols if c not in fixed_cols]
    out_cols = [c for c in dict.fromkeys(out_cols) if c in featured.columns]

    features_path = f"{args.out_prefix}_phase21_pit_features.csv"
    print(f"\nWriting {len(featured):,} row(s) x {len(out_cols)} column(s) to {features_path} "
          f"(no further progress output during this step -- pandas.to_csv prints nothing while "
          f"running; this can legitimately take several minutes at this size)...")
    featured[out_cols].to_csv(features_path, index=False)
    print(f"Wrote {features_path}")

    summary_path = f"{args.out_prefix}_phase21_pit_summary.csv"
    summary_df = pd.DataFrame({
        "reconstruction_status": status_counts.index,
        "n_episodes": status_counts.values,
    })
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    schema_check_path = f"{args.out_prefix}_phase21_upstream_schema_check.json"
    import json
    Path(schema_check_path).write_text(json.dumps(schema_check, indent=2))
    print(f"Wrote {schema_check_path}")


if __name__ == "__main__":
    main()

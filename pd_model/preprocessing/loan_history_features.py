"""
Phase 2.2 (loan-level) -- loan history features, labelling, and column mapping.

Replaces ``run_phase_2_2_repayment_pd_features()`` (see
``pd_model/preprocessing/loan_features.py``) for training data sourced from
``data/loan_state_query_updated_materialized.txt``, whose grain is one row
per ``disbursement_fid`` (per target loan) rather than one row per
``(agent_msisdn, snapshot_dt)``. ``loan_features.py`` is left intact and
unused -- this module has zero dependency on it, so the old module can be
deleted later without touching this one.

Provides:
- ``LABEL_DIAGNOSTIC_COLUMNS``       -- the 9 future-derived columns that must
                                       never reach the model (label leakage);
                                       see ``data/Features_Consult.txt``.
- ``SNAPSHOT_TO_TRAINING_COLUMN_MAP`` -- column-name correspondence between
                                       ``data/loan_history_snapshot_query.txt``
                                       (scoring time) and
                                       ``data/loan_state_query_updated_materialized.txt``
                                       (training time). Not a uniform suffix
                                       swap -- verified column-by-column.
- ``compute_bad_flags_loan_level``   -- derives ``bad_state`` from
                                       ``bad_state_3dpd_30d``.
- ``derive_thin_file_flag``          -- ``thin_file_flag`` from
                                       ``observed_prior_loan_count`` (exact
                                       count, not a same-day-inflated proxy,
                                       so no additional threshold is needed).
- ``run_phase_2_2_loan_history_pd_features``            -- training path.
- ``run_phase_2_2_loan_history_pd_features_inference``  -- scoring-time
                                       counterpart (no label to derive).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pd_model.config.model_config import DEFAULT_CONFIG, ModelConfig
from pd_model.exceptions import DataAlignmentError
from pd_model.logging_config import get_logger
from pd_model.validation.schema import require_binary_column, require_columns

logger = get_logger(__name__)


# ======================================================================== #
# Column-name constants
# ======================================================================== #

# Future-derived diagnostic columns present in the training query's output.
# Retained by the SQL for label auditing only -- never feed these to the
# model. See data/Features_Consult.txt for what each one means and how to
# use it for label auditing.
LABEL_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "max_days_aging_7d",
    "max_days_aging_30d",
    "rollover_observed_7d",
    "rollover_observed_30d",
    "terminal_state_observed_30d",
    "outcome_state_row_count_30d",
    "outcome_observed_date_count_30d",
    "first_outcome_state_date",
    "last_outcome_state_date",
)

# Identifiers / labels that must be present on the incoming training frame.
_TRAINING_REQUIRED_COLUMNS: list[str] = [
    "msisdn",
    "disbursement_fid",
    "split",
    "loan_date",
    "bad_state_3dpd_30d",
]

# data/loan_history_snapshot_query.txt column -> data/loan_state_query_updated_materialized.txt
# column. Only columns that need renaming are listed; columns matching
# exactly (has_observed_prior_state, has_pre_window_history,
# days_since_last_repayment, avg_tenure_days_closed_loans,
# max_tenure_days_last_5_closed_loans, historical_anomaly_open_loan_rate,
# consecutive_on_time_loans, months_since_first_observed_loan) pass through
# unrenamed. Columns describing the *candidate loan being requested*
# (disbursement_amount_ugx, disbursement_hour, disbursement_day_of_week,
# late_afternoon_disbursement, current_to_avg_prior_loan_amount_*,
# prior_loans_per_observed_month) have no snapshot-side analog at all --
# those must come from the scoring request itself, not from this map.
SNAPSHOT_TO_TRAINING_COLUMN_MAP: dict[str, str] = {
    "has_unresolved_loan_at_snapshot": "has_unresolved_loan_at_scoring",
    "active_loan_days_aging_at_snapshot": "active_loan_days_aging_at_scoring",
    "active_loan_outstanding_ugx_at_snapshot": "active_loan_outstanding_ugx_at_scoring",
    "anomaly_open_at_snapshot": "anomaly_open_at_scoring",
    "loan_seq_at_snapshot": "loan_seq_at_scoring",
    "latest_loan_principal_repayment_ratio": "scoring_state_loan_principal_repayment_ratio",
    "latest_loan_uid": "scoring_state_loan_uid",
    "latest_state_date": "scoring_state_date",
    "latest_loan_status": "scoring_state_loan_status",
    "observed_loan_count": "observed_prior_loan_count",
    "closed_loan_count": "prior_closed_loan_count",
    "all_loans_disbursed_ugx": "all_prior_loans_disbursed_ugx",
    "all_loans_repaid_ugx": "all_prior_loans_repaid_ugx",
    "all_loans_gross_repaid_ugx": "all_prior_loans_gross_repaid_ugx",
    "all_loans_fees_paid_ugx": "all_prior_loans_fees_paid_ugx",
    "all_loans_interest_paid_ugx": "all_prior_loans_interest_paid_ugx",
    "all_loans_principal_repayment_ratio": "all_prior_loans_principal_repayment_ratio",
    "all_loans_gross_repayment_ratio": "all_prior_loans_gross_repayment_ratio",
    "closed_loans_principal_repayment_ratio": "closed_prior_loans_principal_repayment_ratio",
    "disbursement_count": "prior_disbursement_count",
    "disbursement_count_30d": "prior_disbursement_count_30d",
    "disbursement_count_90d": "prior_disbursement_count_90d",
    "disbursement_count_180d": "prior_disbursement_count_180d",
    "avg_loan_amount_30d": "avg_prior_loan_amount_30d",
    "avg_loan_amount_90d": "avg_prior_loan_amount_90d",
    "avg_loan_amount_180d": "avg_prior_loan_amount_180d",
    "disbursed_ugx_30d": "prior_disbursed_ugx_30d",
    "disbursed_ugx_90d": "prior_disbursed_ugx_90d",
    "disbursed_ugx_180d": "prior_disbursed_ugx_180d",
}


# ======================================================================== #
# Label computation
# ======================================================================== #


def compute_bad_flags_loan_level(df_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ``bad_state`` from the primary label ``bad_state_3dpd_30d``
    (already computed in SQL -- see data/loan_state_query_updated_materialized.txt).

    Unlike ``compute_bad_flags()`` in ``loan_features.py``, there is no
    ``hard_bad_flag`` here: that flag was defined from monthly-bucket
    penalty columns with no equivalent in the new schema.

    Args:
        df_pd: Modelling DataFrame; must contain ``bad_state_3dpd_30d``.

    Returns:
        Copy of *df_pd* with ``bad_state`` added.
    """
    require_columns(df_pd, ["bad_state_3dpd_30d"], context="compute_bad_flags_loan_level")
    df = df_pd.copy()

    df["bad_state"] = (pd.to_numeric(df["bad_state_3dpd_30d"], errors="coerce").fillna(0) > 0).astype(int)

    require_binary_column(df, "bad_state", context="compute_bad_flags_loan_level")

    logger.info(
        "compute_bad_flags_loan_level: bad_state=%d/%d (%.2f%%)",
        int(df["bad_state"].sum()),
        len(df),
        float(df["bad_state"].mean()) * 100,
    )
    return df


# ======================================================================== #
# Thin-file classification
# ======================================================================== #


def derive_thin_file_flag(
    df_pd: pd.DataFrame,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Add ``has_ever_loan``, ``has_loan_history``, ``is_new_agent``,
    ``thin_file_flag``, and ``thin_file_pd_prior`` from
    ``observed_prior_loan_count``.

    ``thin_file_flag = 1`` iff ``observed_prior_loan_count == 0``.
    ``observed_prior_loan_count`` is an exact count of distinct prior loans
    reconstructed from loan_state_daily, not a same-day-inflated monthly
    proxy, so no additional month/loan-count floor is needed (unlike the
    old ``thin_file_min_active_months`` / ``thin_file_min_lifetime_loans``
    dual threshold in ``loan_features.py``).

    Args:
        df_pd: Modelling DataFrame; must contain ``observed_prior_loan_count``.
        cfg:   Model config supplying ``thin_file_pd_prior``.

    Returns:
        Copy of *df_pd* with the flags above added.
    """
    require_columns(df_pd, ["observed_prior_loan_count"], context="derive_thin_file_flag")
    df = df_pd.copy()

    has_loan_history = (
        pd.to_numeric(df["observed_prior_loan_count"], errors="coerce").fillna(0) > 0
    ).astype(int)

    df["has_ever_loan"] = has_loan_history
    df["has_loan_history"] = has_loan_history
    df["is_new_agent"] = (has_loan_history == 0).astype(int)
    df["thin_file_flag"] = (has_loan_history == 0).astype(int)
    df["thin_file_pd_prior"] = np.where(df["thin_file_flag"] == 1, cfg.thin_file_pd_prior, 0.0)

    n_thin = int(df["thin_file_flag"].sum())
    logger.info(
        "derive_thin_file_flag: %d/%d thin-file agents (thin_file_pd_prior=%.2f)",
        n_thin,
        len(df),
        cfg.thin_file_pd_prior,
    )
    return df


# ======================================================================== #
# Training-time pipeline
# ======================================================================== #


def run_phase_2_2_loan_history_pd_features(
    df_loans: pd.DataFrame,
    cfg: ModelConfig = DEFAULT_CONFIG,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the loan-level training frame produced by
    ``data/loan_state_query_updated_materialized.txt`` for modelling.

    Steps
    -----
    1. Validate required id/label columns.
    2. Move the 9 label-diagnostic columns out of the feature-eligible frame
       before anything reaches feature classification -- they are retained
       separately, for label auditing only (see data/Features_Consult.txt).
       ``bad_state_1dpd_7d`` (the secondary label) is NOT stripped here --
       it stays in the returned frame as a monitoring signal, but must
       never be used as a feature (enforced via
       ``feature_config.PD_FEATURE_BLACKLIST``, not by this function).
    3. Normalise the join key: ``msisdn`` -> ``agent_msisdn``.
    4. Derive ``bad_state`` from ``bad_state_3dpd_30d``.
    5. Derive ``thin_file_flag`` and related flags from
       ``observed_prior_loan_count``.

    Args:
        df_loans: Loan-level training DataFrame (query output).
        cfg:      Model config.
        verbose:  If True, log extra diagnostics.

    Returns:
        Tuple of ``(df_pd_out, df_diagnostics)`` where *df_diagnostics*
        holds the 9 label-diagnostic columns (keyed by ``agent_msisdn``,
        ``disbursement_fid``) for label auditing, kept out of the modelling
        frame.
    """
    df = df_loans.copy()
    df.columns = [c.strip() for c in df.columns]

    require_columns(df, _TRAINING_REQUIRED_COLUMNS, context="run_phase_2_2_loan_history")

    # -- Normalise join key --
    if "agent_msisdn" not in df.columns:
        df = df.rename(columns={"msisdn": "agent_msisdn"})
    df["agent_msisdn"] = df["agent_msisdn"].astype(str).str.strip()

    # -- Split off label-diagnostic columns before feature classification --
    present_diagnostics = [c for c in LABEL_DIAGNOSTIC_COLUMNS if c in df.columns]
    if present_diagnostics:
        df_diagnostics = df[["agent_msisdn", "disbursement_fid"] + present_diagnostics].copy()
        df = df.drop(columns=present_diagnostics)
        logger.info(
            "run_phase_2_2_loan_history: moved %d label-diagnostic column(s) out of the "
            "feature-eligible frame (retained separately for label auditing only): %s",
            len(present_diagnostics),
            present_diagnostics,
        )
    else:
        df_diagnostics = pd.DataFrame(columns=["agent_msisdn", "disbursement_fid"])

    # -- Label --
    df = compute_bad_flags_loan_level(df)

    # -- Thin-file classification --
    df = derive_thin_file_flag(df, cfg=cfg)

    if verbose:
        logger.info(
            "run_phase_2_2_loan_history: %d rows | bad_state=%d (%.2f%%) | thin_file=%d",
            len(df),
            int(df["bad_state"].sum()),
            float(df["bad_state"].mean()) * 100,
            int(df["thin_file_flag"].sum()),
        )

    return df, df_diagnostics


# ======================================================================== #
# Inference-time pipeline
# ======================================================================== #


def apply_snapshot_to_training_column_map(df_snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Rename ``data/loan_history_snapshot_query.txt`` output columns to match
    the training-time ``_at_scoring`` naming convention, so the same
    downstream feature-alignment logic works for both training and
    inference.
    """
    df = df_snapshot.copy()
    rename_map = {k: v for k, v in SNAPSHOT_TO_TRAINING_COLUMN_MAP.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def run_phase_2_2_loan_history_pd_features_inference(
    df_pd: pd.DataFrame,
    df_loan_history: pd.DataFrame,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Scoring-time counterpart of ``run_phase_2_2_loan_history_pd_features()``.
    There is no label to derive at inference time.

    Args:
        df_pd:           Modelling DataFrame (agent_msisdn or msisdn key).
        df_loan_history: ``data/loan_history_snapshot_query.txt`` output
                         (e.g. via
                         ``extrafloat.io.extrafloat_data_loaders.load_loan_history_snapshot_features()``).
        cfg:             Model config.

    Returns:
        Copy of *df_pd* with loan-history columns (renamed to the training
        naming convention) left-joined on ``agent_msisdn``, plus thin-file
        flags.

    Raises:
        DataAlignmentError: If *df_loan_history* is not unique per
                            ``agent_msisdn``, or the merge changes row count.
    """
    df_pd = df_pd.copy()
    df_history = apply_snapshot_to_training_column_map(df_loan_history.copy())

    if "agent_msisdn" not in df_history.columns and "msisdn" in df_history.columns:
        df_history = df_history.rename(columns={"msisdn": "agent_msisdn"})
    if "agent_msisdn" not in df_pd.columns and "msisdn" in df_pd.columns:
        df_pd = df_pd.rename(columns={"msisdn": "agent_msisdn"})

    require_columns(df_pd, ["agent_msisdn"], context="run_phase_2_2_loan_history_inference")
    require_columns(df_history, ["agent_msisdn"], context="run_phase_2_2_loan_history_inference")

    df_pd["agent_msisdn"] = df_pd["agent_msisdn"].astype(str).str.strip()
    df_history["agent_msisdn"] = df_history["agent_msisdn"].astype(str).str.strip()

    dup = int(df_history.duplicated(subset=["agent_msisdn"], keep=False).sum())
    if dup:
        raise DataAlignmentError(
            f"[run_phase_2_2_loan_history_inference] loan history snapshot is not "
            f"unique per agent_msisdn ({dup} duplicate rows)"
        )

    join_cols = [c for c in df_history.columns if c != "agent_msisdn"]
    n_before = len(df_pd)
    df_out = df_pd.merge(df_history[["agent_msisdn"] + join_cols], on="agent_msisdn", how="left")

    if len(df_out) != n_before:
        raise DataAlignmentError(
            "[run_phase_2_2_loan_history_inference] merge changed row count: "
            f"{n_before} -> {len(df_out)}"
        )

    df_out = derive_thin_file_flag(df_out, cfg=cfg)

    return df_out

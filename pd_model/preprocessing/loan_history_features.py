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
- ``LABEL_DIAGNOSTIC_COLUMNS``       -- the future-derived columns that must
                                       never reach the model (label leakage);
                                       see ``data/Features_Consult.txt``.
                                       Includes the bounded coverage-ratio /
                                       label-eligibility columns
                                       (``follow_up_coverage_ratio_30d``,
                                       ``meets_coverage_ratio_30d``,
                                       ``label_eligible_30d``,
                                       ``label_eligibility_reason_30d``, etc.)
                                       that ``run_phase_2_2_loan_history_pd_features()``
                                       uses to filter the modelling population
                                       before stripping them.
- ``SNAPSHOT_TO_TRAINING_COLUMN_MAP`` -- column-name correspondence between
                                       ``data/loan_history_snapshot_query.txt``
                                       (scoring time) and
                                       ``data/loan_state_query_updated_materialized.txt``
                                       (training time). Not a uniform suffix
                                       swap -- verified column-by-column.
- ``compute_bad_flags_loan_level``   -- derives ``bad_state`` from
                                       ``bad_state_3dpd_30d``.
- ``derive_thin_file_flag``          -- ``thin_file_flag``, gated by
                                       ``cfg.thin_file_use_windowed_rule``
                                       (default False until the warehouse
                                       has a mature 180-day lookback -- see
                                       the function's own docstring):
                                       windowed mode requires BOTH
                                       ``prior_loan_count_180d`` and
                                       ``prior_active_loan_months_180d``
                                       to clear ``cfg.thin_file_min_lifetime_loans``
                                       / ``cfg.thin_file_min_active_months``;
                                       interim mode (default) falls back to
                                       the simpler lifetime
                                       ``no_loan_history_flag``. Also always
                                       adds ``no_loan_history_flag`` itself,
                                       a lifetime "never borrowed" signal
                                       from ``observed_prior_loan_count``.
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
)

# Identifiers / labels that must be present on the incoming training frame.
_TRAINING_REQUIRED_COLUMNS: list[str] = [
    "msisdn",
    "disbursement_fid",
    "split",
    "loan_date",
    "bad_state_3dpd_30d",
    "bad_state_1dpd_7d",
    "label_eligible_30d",
    "follow_up_coverage_ratio_30d",
]

# data/loan_history_snapshot_query.txt column -> data/loan_state_query_updated_materialized.txt
# column. Only columns that need renaming are listed; columns matching
# exactly (has_observed_prior_state, has_pre_window_history,
# days_since_last_repayment, avg_tenure_days_closed_loans,
# max_tenure_days_last_5_closed_loans, historical_anomaly_open_loan_rate,
# consecutive_on_time_loans, months_since_first_observed_loan,
# prior_loan_count_180d, prior_active_loan_months_180d) pass through
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
    "late_fee_1_count": "prior_late_fee_1_count",
    "late_fee_2_count": "prior_late_fee_2_count",
    "total_late_fee_owed_ugx": "prior_total_late_fee_owed_ugx",
}


# ======================================================================== #
# Label computation
# ======================================================================== #


def _apply_bad_flags_loan_level_inplace(df: pd.DataFrame) -> None:
    """
    Core logic of ``compute_bad_flags_loan_level()``, mutating *df* in place.

    Factored out so ``run_phase_2_2_loan_history_pd_features()`` can invoke
    it directly on its own already-exclusively-owned frame, instead of
    going through ``compute_bad_flags_loan_level()``'s own defensive
    ``.copy()`` -- which is real and necessary for that function's other
    callers, but redundant when the caller already owns a private object
    nothing else references (confirmed via grep: every current caller of
    this logic already exclusively owns its input at the point of call).
    """
    require_columns(df, ["bad_state_3dpd_30d"], context="compute_bad_flags_loan_level")

    df["bad_state"] = (pd.to_numeric(df["bad_state_3dpd_30d"], errors="coerce").fillna(0) > 0).astype(int)

    require_binary_column(df, "bad_state", context="compute_bad_flags_loan_level")

    logger.info(
        "compute_bad_flags_loan_level: bad_state=%d/%d (%.2f%%)",
        int(df["bad_state"].sum()),
        len(df),
        float(df["bad_state"].mean()) * 100,
    )


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
    df = df_pd.copy()
    _apply_bad_flags_loan_level_inplace(df)
    return df


# ======================================================================== #
# Thin-file classification
# ======================================================================== #


def _apply_thin_file_flag_inplace(df: pd.DataFrame, cfg: ModelConfig) -> None:
    """
    Core logic of ``derive_thin_file_flag()``, mutating *df* in place. See
    ``_apply_bad_flags_loan_level_inplace()`` for why this is factored out
    -- same rationale, same "every current caller already owns its input"
    verification.
    """
    require_columns(df, ["observed_prior_loan_count"], context="derive_thin_file_flag")

    # Lifetime "never borrowed at all" signal -- has_ever_loan/has_loan_history
    # keep their existing meaning (any history, ever); this is NOT the
    # thick-file routing decision, only thin_file_flag below is.
    no_loan_history = (
        pd.to_numeric(df["observed_prior_loan_count"], errors="coerce").fillna(0) == 0
    ).astype(int)

    df["has_ever_loan"] = 1 - no_loan_history
    df["has_loan_history"] = 1 - no_loan_history
    df["is_new_agent"] = no_loan_history
    df["no_loan_history_flag"] = no_loan_history

    if cfg.thin_file_use_windowed_rule:
        # Thick-file routing requires BOTH sufficient recent loan volume AND
        # sufficient temporal breadth over the trailing 180 days, point-in-time
        # per target loan (see data/loan_state_query_updated_materialized.txt's
        # prior_disbursement_features CTE). One prior loan, or a burst of loans
        # in a single month, is not enough evidence for the full behavioral
        # model -- fewer than cfg.thin_file_min_lifetime_loans loans OR fewer
        # than cfg.thin_file_min_active_months distinct active months routes to
        # the conservative thin-file LR path instead.
        require_columns(
            df,
            ["prior_loan_count_180d", "prior_active_loan_months_180d"],
            context="derive_thin_file_flag",
        )
        prior_loans_180d = pd.to_numeric(df["prior_loan_count_180d"], errors="coerce").fillna(0)
        active_months_180d = pd.to_numeric(df["prior_active_loan_months_180d"], errors="coerce").fillna(0)
        df["thin_file_flag"] = (
            (prior_loans_180d < cfg.thin_file_min_lifetime_loans)
            | (active_months_180d < cfg.thin_file_min_active_months)
        ).astype(int)
        rule_desc = "windowed (180d loan-count + active-months)"
    else:
        # Interim rule while the warehouse hasn't yet accumulated a full
        # 180-day lookback (needs cohort_start >= warehouse_start + 180d;
        # as of max_state_date=2026-06-16 the warehouse only has 166 days of
        # history, so the windowed rule would structurally misclassify the
        # entire training cohort -- every training-period loan has too
        # little *elapsed calendar time* to ever show 4 active months,
        # regardless of the agent's real borrowing frequency). Falls back
        # to the simpler lifetime "no history at all" rule, which has no
        # minimum-elapsed-time requirement. Set
        # cfg.thin_file_use_windowed_rule=True once the warehouse matures
        # past ~2026-06-30.
        df["thin_file_flag"] = no_loan_history
        rule_desc = "interim (no_loan_history only -- windowed rule disabled, see cfg.thin_file_use_windowed_rule)"

    df["thin_file_pd_prior"] = np.where(df["thin_file_flag"] == 1, cfg.thin_file_pd_prior, 0.0)

    n_thin = int(df["thin_file_flag"].sum())
    n_no_history = int(df["no_loan_history_flag"].sum())
    logger.info(
        "derive_thin_file_flag [%s]: %d/%d thin-file agents (%d with no loan history at all, "
        "thin_file_pd_prior=%.2f)",
        rule_desc,
        n_thin,
        len(df),
        n_no_history,
        cfg.thin_file_pd_prior,
    )


def derive_thin_file_flag(
    df_pd: pd.DataFrame,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Add ``has_ever_loan``, ``has_loan_history``, ``is_new_agent``,
    ``no_loan_history_flag``, ``thin_file_flag``, and ``thin_file_pd_prior``.

    ``no_loan_history_flag = 1`` iff ``observed_prior_loan_count == 0``
    (lifetime, unbounded) -- a genuine "never borrowed at all" signal,
    computed unconditionally.

    ``thin_file_flag`` has two modes, controlled by
    ``cfg.thin_file_use_windowed_rule``:

    - **True** (windowed rule): ``thin_file_flag = 1`` iff
      ``prior_loan_count_180d < cfg.thin_file_min_lifetime_loans`` OR
      ``prior_active_loan_months_180d < cfg.thin_file_min_active_months`` --
      both are point-in-time, 180-day-bounded counts reconstructed per
      target loan in SQL (``data/loan_state_query_updated_materialized.txt``'s
      ``prior_disbursement_features`` CTE). Requires those two columns to
      be present. Correctly catches a single prior loan, or a burst of
      loans concentrated in one month, as thin-file. Only meaningful once
      the warehouse has accumulated a genuine 180-day lookback for the
      cohort being scored (warehouse start + 180 days) -- otherwise every
      loan in an immature cohort is structurally unable to show
      ``thin_file_min_active_months`` distinct months, regardless of the
      agent's real borrowing frequency, and gets misclassified thin.
    - **False** (default; interim rule): ``thin_file_flag`` falls back to
      ``no_loan_history_flag`` (the old, simpler rule) -- no minimum
      elapsed-calendar-time requirement, so not subject to the warehouse-
      maturity bias above. Does not require the two windowed columns to
      be present. Set ``cfg.thin_file_use_windowed_rule=True`` once the
      warehouse has matured past ~180 days from its earliest data point.

    Args:
        df_pd: Modelling DataFrame; must contain ``observed_prior_loan_count``.
               If ``cfg.thin_file_use_windowed_rule`` is True, must also
               contain ``prior_loan_count_180d`` and
               ``prior_active_loan_months_180d``.
        cfg:   Model config supplying ``thin_file_pd_prior``,
               ``thin_file_min_lifetime_loans``, ``thin_file_min_active_months``,
               ``thin_file_use_windowed_rule``.

    Returns:
        Copy of *df_pd* with the flags above added.
    """
    df = df_pd.copy()
    _apply_thin_file_flag_inplace(df, cfg)
    return df


# ======================================================================== #
# Training-time pipeline
# ======================================================================== #


def _validate_label_observability(df: pd.DataFrame) -> None:
    """
    Hard-fail data-integrity checks on the full, unfiltered training frame,
    called before it is split into the audit frame and the modelling
    population. Python-side echo of the pre-flight SQL checks documented in
    ``data/loan_state_query_updated_materialized.txt`` -- catches the same
    class of problem without depending on someone having run those checks
    first.

    Mutates *df* in place: once every row has passed validation, writes the
    coerced numeric ``follow_up_coverage_ratio_30d`` back onto it, so
    ``df_diagnostics`` and the audit summary's ``.mean()`` aggregation
    aren't silently built from an object-typed column.

    Raises:
        DataAlignmentError: On a duplicated ``disbursement_fid``, a
                            non-numeric, null, or out-of-[0, 1]-range
                            ``follow_up_coverage_ratio_30d``.
    """
    dup_mask = df["disbursement_fid"].duplicated(keep=False)
    if dup_mask.any():
        raise DataAlignmentError(
            f"[run_phase_2_2_loan_history] {int(dup_mask.sum())} rows share "
            "a duplicated disbursement_fid -- a target loan must map to "
            "exactly one row"
        )

    raw_ratio = df["follow_up_coverage_ratio_30d"]
    coverage_ratio = pd.to_numeric(raw_ratio, errors="coerce")

    non_numeric = raw_ratio.notna() & coverage_ratio.isna()
    if non_numeric.any():
        raise DataAlignmentError(
            f"[run_phase_2_2_loan_history] {int(non_numeric.sum())} rows "
            "have a non-numeric follow_up_coverage_ratio_30d"
        )

    missing_ratio = raw_ratio.isna()
    if missing_ratio.any():
        raise DataAlignmentError(
            f"[run_phase_2_2_loan_history] {int(missing_ratio.sum())} rows "
            "have a null follow_up_coverage_ratio_30d -- the SQL only "
            "produces NULL here for CENSORED_INVALID_OBSERVATION_WINDOW, "
            "itself a build-blocking integrity failure (see the pre-flight "
            "checks in data/loan_state_query_updated_materialized.txt); a "
            "genuinely unobserved loan with a valid window yields "
            "0.0 / positive, never NULL"
        )

    # missing_ratio already ruled out nulls above, so coverage_ratio is
    # fully non-null here -- no separate notna() guard needed.
    out_of_range = (coverage_ratio > 1.0 + 1e-9) | (coverage_ratio < -1e-9)
    if out_of_range.any():
        raise DataAlignmentError(
            f"[run_phase_2_2_loan_history] {int(out_of_range.sum())} rows "
            "have follow_up_coverage_ratio_30d outside [0, 1] -- inspect "
            "closure dates, loan mapping, and date bounds in the source "
            "SQL export (see the pre-flight checks in "
            "data/loan_state_query_updated_materialized.txt)"
        )

    # Every row now has a finite, in-range ratio -- write the coerced
    # numeric dtype back so df_diagnostics and the audit summary's
    # coverage_ratio_mean=(..., "mean") aren't silently built from an
    # object-typed column.
    df["follow_up_coverage_ratio_30d"] = coverage_ratio


def run_phase_2_2_loan_history_pd_features(
    df_loans: pd.DataFrame,
    cfg: ModelConfig = DEFAULT_CONFIG,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the loan-level training frame produced by
    ``data/loan_state_query_updated_materialized.txt`` for modelling.

    The SQL export is unfiltered -- every target loan, including
    ``CENSORED_*`` rows, is present -- so eligibility filtering happens
    here, not in SQL, and must run *before* label-diagnostic columns are
    stripped (``label_eligible_30d`` is itself one of them).

    Steps
    -----
    1. Validate required id/label columns, including
       ``label_eligible_30d`` and ``follow_up_coverage_ratio_30d``.
    2. Normalise the join key: ``msisdn`` -> ``agent_msisdn``.
    3. ``_validate_label_observability()``: hard-fail on duplicated
       ``disbursement_fid`` or an invalid ``follow_up_coverage_ratio_30d``;
       normalizes the ratio to numeric dtype in place.
    4. Validate ``label_eligible_30d`` is strictly binary; normalize to
       ``int8`` in place.
    5. Build the full, unfiltered audit frame from the now-normalized
       frame (every target loan, ``CENSORED_*`` rows included).
    6. Filter to the modelling population via ``label_eligible_30d``.
    7. Derive ``bad_state`` from ``bad_state_3dpd_30d``.
    8. Strip label-diagnostic columns from the now-filtered modelling
       frame -- they are retained separately, in the audit frame, for
       label auditing only (see ``data/Features_Consult.txt``).
       ``bad_state_1dpd_7d`` (the secondary label) is NOT stripped here --
       it stays in the returned frame as a monitoring signal, but must
       never be used as a feature (enforced via
       ``feature_config.PD_FEATURE_BLACKLIST``, not by this function).
    9. Derive ``thin_file_flag`` and related flags from
       ``prior_loan_count_180d`` / ``prior_active_loan_months_180d``
       (see ``derive_thin_file_flag`` docstring).

    Args:
        df_loans: Loan-level training DataFrame (query output).
        cfg:      Model config.
        verbose:  If True, log extra diagnostics.

    Returns:
        Tuple of ``(df_pd_out, df_diagnostics)`` where *df_diagnostics*
        holds the full, unfiltered audit frame (every target loan,
        ``CENSORED_*`` rows included) for label auditing, kept out of the
        modelling frame.

    Raises:
        DataAlignmentError: See ``_validate_label_observability()``; also
                            raised if ``label_eligible_30d`` is not
                            strictly binary, or if censored rows remain
                            after eligibility filtering.
    """
    df = df_loans.copy()
    df.columns = [c.strip() for c in df.columns]

    require_columns(df, _TRAINING_REQUIRED_COLUMNS, context="run_phase_2_2_loan_history")

    # -- Normalise join key --
    if "agent_msisdn" not in df.columns:
        df = df.rename(columns={"msisdn": "agent_msisdn"})
    df["agent_msisdn"] = df["agent_msisdn"].astype(str).str.strip()

    # -- Validate + normalize on the full frame, before any filtering --
    _validate_label_observability(df)

    eligibility = pd.to_numeric(df["label_eligible_30d"], errors="coerce")
    if eligibility.isna().any() or not eligibility.isin([0, 1]).all():
        raise DataAlignmentError(
            "[run_phase_2_2_loan_history] label_eligible_30d must be "
            "strictly 0/1 with no nulls"
        )
    df["label_eligible_30d"] = eligibility.astype("int8")

    # -- Build the full audit frame from validated, normalized data --
    # every target loan, CENSORED_* rows included.
    audit_cols = list(
        dict.fromkeys(
            c
            for c in [
                "agent_msisdn",
                "disbursement_fid",
                "target_loan_uid",
                "split",
                "loan_date",
                "bad_state_3dpd_30d",
                "bad_state_1dpd_7d",
                *LABEL_DIAGNOSTIC_COLUMNS,
            ]
            if c in df.columns
        )
    )
    df_diagnostics = df[audit_cols].copy()

    # -- Filter to the modelling population --
    n_before = len(df)
    df = df.loc[df["label_eligible_30d"].eq(1)].copy()
    logger.info(
        "run_phase_2_2_loan_history: dropped %d/%d censored rows "
        "(label_eligible_30d=0) before label derivation",
        n_before - len(df),
        n_before,
    )
    if not df["label_eligible_30d"].eq(1).all():
        raise DataAlignmentError(
            "[run_phase_2_2_loan_history] censored rows remained after "
            "label eligibility filtering"
        )

    # -- Label --
    # df is already exclusively owned by this function (copied from
    # df_loans at the top) -- call the in-place core logic directly instead
    # of compute_bad_flags_loan_level(), which would re-copy the input again
    # only to protect a contract nothing here relies on.
    _apply_bad_flags_loan_level_inplace(df)

    # -- Strip label-diagnostic columns now that filtering is done --
    present_diagnostics = [c for c in LABEL_DIAGNOSTIC_COLUMNS if c in df.columns]
    df = df.drop(columns=present_diagnostics)
    logger.info(
        "run_phase_2_2_loan_history: stripped %d label-diagnostic column(s) from the "
        "modelling frame (retained separately in df_diagnostics): %s",
        len(present_diagnostics),
        present_diagnostics,
    )

    # -- Thin-file classification --
    _apply_thin_file_flag_inplace(df, cfg)

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

"""Tests for pd_model.preprocessing.loan_history_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pd_model.config.model_config import DEFAULT_CONFIG, ModelConfig
from pd_model.exceptions import DataAlignmentError
from pd_model.preprocessing.loan_history_features import (
    LABEL_DIAGNOSTIC_COLUMNS,
    SNAPSHOT_TO_TRAINING_COLUMN_MAP,
    apply_snapshot_to_training_column_map,
    compute_bad_flags_loan_level,
    derive_thin_file_flag,
    run_phase_2_2_loan_history_pd_features,
    run_phase_2_2_loan_history_pd_features_inference,
)


def _loan_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_train = n // 2
    split = ["train"] * n_train + ["validation"] * (n - n_train)

    bad_state_3dpd_30d = rng.integers(0, 2, n)
    # confirmed_good_30d can only be 1 where bad_state_3dpd_30d is 0 --
    # mirrors the SQL's confirmed_good_30d definition (bad_state_3dpd_30d=0
    # is one of its own conjuncts).
    confirmed_good_30d = np.where(bad_state_3dpd_30d == 1, 0, rng.integers(0, 2, n))
    label_eligible_30d = ((bad_state_3dpd_30d == 1) | (confirmed_good_30d == 1)).astype(int)
    label_eligibility_reason_30d = np.where(
        bad_state_3dpd_30d == 1,
        "KNOWN_BAD",
        np.where(confirmed_good_30d == 1, "CONFIRMED_GOOD_HORIZON", "CENSORED_SPARSE_FOLLOW_UP"),
    )

    return pd.DataFrame(
        {
            "msisdn": [f"256{i:07d}" for i in range(n)],
            "disbursement_fid": [f"D{i:06d}" for i in range(n)],
            "target_loan_uid": [f"L{i:06d}" for i in range(n)],
            "split": split,
            "loan_date": pd.to_datetime("2025-09-30"),
            "bad_state_3dpd_30d": bad_state_3dpd_30d,
            "bad_state_1dpd_7d": rng.integers(0, 2, n),
            "observed_prior_loan_count": rng.integers(0, 10, n),
            "prior_loan_count_180d": rng.integers(0, 10, n),
            "prior_active_loan_months_180d": rng.integers(0, 6, n),
            "active_loan_days_aging_at_scoring": rng.integers(0, 10, n).astype(float),
            "disbursement_amount_ugx": rng.exponential(50000, n),
            "max_days_aging_7d": rng.integers(0, 10, n).astype(float),
            "max_days_aging_30d": rng.integers(0, 30, n).astype(float),
            "rollover_observed_7d": rng.integers(0, 2, n),
            "rollover_observed_30d": rng.integers(0, 2, n),
            "terminal_state_observed_30d": rng.integers(0, 2, n),
            "same_day_settlement_observed_30d": rng.integers(0, 2, n),
            "outcome_state_row_count_30d": rng.integers(0, 30, n),
            "outcome_observed_date_count_30d": rng.integers(0, 30, n),
            "post_disbursement_observed_date_count_7d": rng.integers(0, 7, n),
            "post_disbursement_observed_date_count_30d": rng.integers(0, 30, n),
            "first_outcome_state_date": pd.to_datetime("2025-10-01"),
            "last_outcome_state_date": pd.to_datetime("2025-10-28"),
            "observed_closure_date_30d": pd.NaT,
            "closure_observation_state_date_30d": pd.NaT,
            "required_observation_end_date_30d": pd.to_datetime("2025-10-30"),
            "expected_observed_date_count_30d": rng.integers(1, 31, n),
            "observed_date_count_to_required_end_30d": rng.integers(0, 31, n),
            "last_state_date_to_required_end_30d": pd.to_datetime("2025-10-28"),
            "follow_up_coverage_ratio_30d": rng.uniform(0, 1, n).round(4),
            "meets_coverage_ratio_30d": rng.integers(0, 2, n),
            "near_horizon_observation_30d": rng.integers(0, 2, n),
            "confirmed_good_30d": confirmed_good_30d,
            "has_post_disbursement_state_30d": rng.integers(0, 2, n),
            "fails_coverage_ratio_30d": rng.integers(0, 2, n),
            "fails_near_horizon_30d": rng.integers(0, 2, n),
            "label_eligible_30d": label_eligible_30d,
            "label_eligibility_reason_30d": label_eligibility_reason_30d,
        }
    )


class TestComputeBadFlagsLoanLevel:
    def test_bad_state_is_binary(self):
        df = _loan_df()
        out = compute_bad_flags_loan_level(df)
        assert out["bad_state"].isin([0, 1]).all()

    def test_bad_state_matches_primary_label(self):
        df = _loan_df()
        out = compute_bad_flags_loan_level(df)
        expected = (out["bad_state_3dpd_30d"] > 0).astype(int)
        assert (out["bad_state"] == expected).all()

    def test_no_hard_bad_flag_column(self):
        """Unlike the old compute_bad_flags(), there is no monthly-bucket
        penalty equivalent, so hard_bad_flag must not be synthesized."""
        df = _loan_df()
        out = compute_bad_flags_loan_level(df)
        assert "hard_bad_flag" not in out.columns

    def test_raises_without_primary_label(self):
        df = _loan_df().drop(columns=["bad_state_3dpd_30d"])
        with pytest.raises(ValueError, match="bad_state_3dpd_30d"):
            compute_bad_flags_loan_level(df)


def _thin_file_cases_df() -> pd.DataFrame:
    """
    One row per case called out in the thin-file redesign:
    0: never borrowed at all (no_loan_history_flag=1, thin).
    1: exactly one prior loan (thin under the new rule -- was thick under
       the old observed_prior_loan_count==0 rule).
    2: 10 loans, all in a single active month (burst guard -- passes the
       loan-count floor but fails the active-months floor, still thin).
    3: 5 loans spread across 4 distinct months (thick -- clears both floors).
    4: 20 lifetime loans but none in the trailing 180 days (stale history --
       thin via the 180d-bounded columns, but NOT no_loan_history_flag,
       since the agent has genuinely borrowed before).
    """
    return pd.DataFrame(
        {
            "observed_prior_loan_count": [0, 1, 10, 5, 20],
            "prior_loan_count_180d": [0, 1, 10, 5, 0],
            "prior_active_loan_months_180d": [0, 1, 1, 4, 0],
        }
    )


_WINDOWED_CFG = ModelConfig(thin_file_use_windowed_rule=True)


class TestDeriveThinFileFlag:
    def test_thin_file_flag_binary(self):
        df = _loan_df()
        out = derive_thin_file_flag(df)
        assert out["thin_file_flag"].isin([0, 1]).all()

    def test_prior_only_for_thin_file(self):
        df = _loan_df()
        out = derive_thin_file_flag(df)
        assert (out.loc[out["thin_file_flag"] == 0, "thin_file_pd_prior"] == 0.0).all()
        assert (out.loc[out["thin_file_flag"] == 1, "thin_file_pd_prior"] > 0.0).all()

    def test_has_ever_loan_matches_has_loan_history(self):
        df = _loan_df()
        out = derive_thin_file_flag(df)
        assert (out["has_ever_loan"] == out["has_loan_history"]).all()

    def test_is_new_agent_matches_no_loan_history_flag(self):
        """is_new_agent tracks the lifetime no_loan_history signal, not the
        180d-bounded thin_file_flag -- these diverge for stale/burst cases."""
        df = _loan_df()
        out = derive_thin_file_flag(df)
        assert (out["is_new_agent"] == out["no_loan_history_flag"]).all()

    def test_one_prior_loan_is_thin_file_under_windowed_rule(self):
        """A single prior loan must route to thin-file under the windowed
        rule -- the interim/old rule (observed_prior_loan_count == 0)
        incorrectly treats this as thick."""
        out = derive_thin_file_flag(_thin_file_cases_df(), cfg=_WINDOWED_CFG)
        assert out.loc[1, "thin_file_flag"] == 1

    def test_burst_of_loans_in_one_month_is_thin_file_under_windowed_rule(self):
        """10 loans clearing the loan-count floor but concentrated in a
        single active month must still route to thin-file (active-months
        floor not met) under the windowed rule."""
        out = derive_thin_file_flag(_thin_file_cases_df(), cfg=_WINDOWED_CFG)
        assert out.loc[2, "thin_file_flag"] == 1

    def test_sufficient_volume_and_breadth_is_thick_file_under_windowed_rule(self):
        out = derive_thin_file_flag(_thin_file_cases_df(), cfg=_WINDOWED_CFG)
        assert out.loc[3, "thin_file_flag"] == 0

    def test_stale_history_is_thin_but_not_no_loan_history_under_windowed_rule(self):
        """20 lifetime loans, none in the trailing 180 days: thin_file_flag
        fires (recent evidence is stale) but no_loan_history_flag does not
        (the agent has genuinely borrowed before) under the windowed rule."""
        out = derive_thin_file_flag(_thin_file_cases_df(), cfg=_WINDOWED_CFG)
        assert out.loc[4, "thin_file_flag"] == 1
        assert out.loc[4, "no_loan_history_flag"] == 0

    def test_no_loan_history_flag_iff_zero_lifetime_loans(self):
        """no_loan_history_flag is computed unconditionally, regardless of
        which thin_file_flag rule is active."""
        out = derive_thin_file_flag(_thin_file_cases_df())
        expected = (out["observed_prior_loan_count"] == 0).astype(int)
        assert (out["no_loan_history_flag"] == expected).all()

    def test_windowed_thresholds_are_config_driven(self):
        """Under the windowed rule, thin_file_flag reads
        cfg.thin_file_min_lifetime_loans / cfg.thin_file_min_active_months,
        not hardcoded literals."""
        df = _thin_file_cases_df()
        assert _WINDOWED_CFG.thin_file_min_lifetime_loans == 5
        assert _WINDOWED_CFG.thin_file_min_active_months == 4
        out = derive_thin_file_flag(df, cfg=_WINDOWED_CFG)
        expected = (
            (df["prior_loan_count_180d"] < _WINDOWED_CFG.thin_file_min_lifetime_loans)
            | (df["prior_active_loan_months_180d"] < _WINDOWED_CFG.thin_file_min_active_months)
        ).astype(int)
        assert (out["thin_file_flag"] == expected).all()

    def test_raises_without_180d_columns_when_windowed_rule_enabled(self):
        df = _loan_df().drop(columns=["prior_loan_count_180d"])
        with pytest.raises(ValueError, match="prior_loan_count_180d"):
            derive_thin_file_flag(df, cfg=_WINDOWED_CFG)

    def test_default_config_uses_interim_rule(self):
        """Regression guard: DEFAULT_CONFIG.thin_file_use_windowed_rule must
        stay False until the warehouse has a mature 180-day lookback (see
        model_config.py comment) -- flipping this silently would
        structurally misclassify an immature training cohort."""
        assert DEFAULT_CONFIG.thin_file_use_windowed_rule is False

    def test_default_rule_matches_no_loan_history_flag(self):
        """With the windowed rule disabled (the default), thin_file_flag
        must exactly equal no_loan_history_flag -- the interim fallback."""
        out = derive_thin_file_flag(_thin_file_cases_df())
        assert (out["thin_file_flag"] == out["no_loan_history_flag"]).all()

    def test_default_rule_does_not_require_180d_columns(self):
        """The interim rule must not require prior_loan_count_180d /
        prior_active_loan_months_180d -- backward compatible with exports
        that predate the windowed-rule SQL columns."""
        df = _loan_df().drop(columns=["prior_loan_count_180d", "prior_active_loan_months_180d"])
        out = derive_thin_file_flag(df)  # default cfg -- must not raise
        assert (out["thin_file_flag"] == out["no_loan_history_flag"]).all()


class TestRunPhase22LoanHistoryPdFeatures:
    def test_renames_msisdn_to_agent_msisdn(self):
        df_out, _diag = run_phase_2_2_loan_history_pd_features(_loan_df())
        assert "agent_msisdn" in df_out.columns
        assert "msisdn" not in df_out.columns

    def test_output_row_count_matches_eligible_rows_only(self):
        """The SQL export is unfiltered -- censored rows must be dropped
        here, in Python, before label derivation. Filtering happens on
        label_eligible_30d, not on row count preservation."""
        df = _loan_df()
        n_eligible = int(df["label_eligible_30d"].eq(1).sum())
        assert 0 < n_eligible < len(df), "fixture must contain both eligible and censored rows"
        df_out, _diag = run_phase_2_2_loan_history_pd_features(df)
        assert len(df_out) == n_eligible

    def test_diagnostics_row_count_matches_full_unfiltered_input(self):
        """df_diagnostics is the full audit export -- every target loan,
        CENSORED_* rows included -- built before eligibility filtering."""
        df = _loan_df()
        _df_out, df_diag = run_phase_2_2_loan_history_pd_features(df)
        assert len(df_diag) == len(df)

    def test_censored_rows_present_in_diagnostics_but_not_output(self):
        df = _loan_df()
        censored_fids = set(df.loc[df["label_eligible_30d"].eq(0), "disbursement_fid"])
        df_out, df_diag = run_phase_2_2_loan_history_pd_features(df)
        assert censored_fids.issubset(set(df_diag["disbursement_fid"]))
        assert censored_fids.isdisjoint(set(df_out["disbursement_fid"]))

    def test_bad_state_and_thin_file_flag_present(self):
        df_out, _diag = run_phase_2_2_loan_history_pd_features(_loan_df())
        assert "bad_state" in df_out.columns
        assert "thin_file_flag" in df_out.columns

    def test_label_diagnostics_stripped_from_feature_frame(self):
        """Label-diagnostic columns must never reach the feature-eligible
        frame -- they are future-derived and would leak the label."""
        df_out, _diag = run_phase_2_2_loan_history_pd_features(_loan_df())
        for col in LABEL_DIAGNOSTIC_COLUMNS:
            assert col not in df_out.columns, (
                f"Label-diagnostic column '{col}' leaked into the feature-eligible frame"
            )

    def test_label_diagnostics_returned_separately(self):
        df = _loan_df()
        _df_out, df_diag = run_phase_2_2_loan_history_pd_features(df)
        for col in LABEL_DIAGNOSTIC_COLUMNS:
            assert col in df_diag.columns
        assert len(df_diag) == len(df)

    def test_label_eligible_matches_known_bad_or_confirmed_good_identity(self):
        """label_eligible_30d == (bad_state_3dpd_30d | confirmed_good_30d)
        for every row of the full audit frame -- pins the eligibility
        identity the SQL and Python layers must agree on."""
        df = _loan_df()
        _df_out, df_diag = run_phase_2_2_loan_history_pd_features(df)
        expected = (
            df_diag["bad_state_3dpd_30d"].astype(int) | df_diag["confirmed_good_30d"].astype(int)
        )
        assert (df_diag["label_eligible_30d"].astype(int) == expected).all()

    def test_same_day_settlement_survives_filtering_despite_zero_coverage_ratio(self):
        """A same-day-settled loan fails every normal coverage-ratio signal
        (zero observed post-disbursement dates by construction) but must
        still survive the eligibility filter via the independent
        same_day_settlement_observed_30d path -- proves Python's filter
        trusts label_eligible_30d rather than re-deriving eligibility from
        the coverage-ratio diagnostics, which would wrongly look censored."""
        df = _loan_df()
        target_fid = "D000000"
        row = df.index[df["disbursement_fid"].eq(target_fid)][0]
        df.loc[row, "bad_state_3dpd_30d"] = 0
        df.loc[row, "observed_date_count_to_required_end_30d"] = 0
        df.loc[row, "expected_observed_date_count_30d"] = 30
        df.loc[row, "follow_up_coverage_ratio_30d"] = 0.0
        df.loc[row, "meets_coverage_ratio_30d"] = 0
        df.loc[row, "near_horizon_observation_30d"] = 0
        df.loc[row, "has_post_disbursement_state_30d"] = 0
        df.loc[row, "fails_coverage_ratio_30d"] = 1
        df.loc[row, "fails_near_horizon_30d"] = 1
        df.loc[row, "same_day_settlement_observed_30d"] = 1
        df.loc[row, "confirmed_good_30d"] = 1
        df.loc[row, "label_eligible_30d"] = 1
        df.loc[row, "label_eligibility_reason_30d"] = "CONFIRMED_GOOD_SAME_DAY"

        df_out, df_diag = run_phase_2_2_loan_history_pd_features(df)

        assert target_fid in set(df_out["disbursement_fid"])
        out_row = df_out.loc[df_out["disbursement_fid"].eq(target_fid)].iloc[0]
        assert out_row["bad_state"] == 0
        assert "same_day_settlement_observed_30d" not in df_out.columns

        diag_row = df_diag.loc[df_diag["disbursement_fid"].eq(target_fid)].iloc[0]
        assert diag_row["label_eligibility_reason_30d"] == "CONFIRMED_GOOD_SAME_DAY"
        assert diag_row["same_day_settlement_observed_30d"] == 1

    def test_secondary_label_not_stripped(self):
        """bad_state_1dpd_7d is a monitoring signal, not a diagnostic -- it
        stays in the returned frame (blacklist keeps it out of features,
        not this function)."""
        df_out, _diag = run_phase_2_2_loan_history_pd_features(_loan_df())
        assert "bad_state_1dpd_7d" in df_out.columns

    def test_raises_without_required_columns(self):
        df = _loan_df().drop(columns=["disbursement_fid"])
        with pytest.raises(ValueError, match="disbursement_fid"):
            run_phase_2_2_loan_history_pd_features(df)

    def test_works_without_optional_diagnostics_present(self):
        """Only label_eligible_30d and follow_up_coverage_ratio_30d are
        required (needed for filtering); every other diagnostic column is
        an optional input -- an export missing them must still work."""
        optional_diagnostics = [
            c for c in LABEL_DIAGNOSTIC_COLUMNS if c not in ("label_eligible_30d", "follow_up_coverage_ratio_30d")
        ]
        df = _loan_df().drop(columns=optional_diagnostics)
        n_eligible = int(df["label_eligible_30d"].eq(1).sum())
        df_out, df_diag = run_phase_2_2_loan_history_pd_features(df)
        assert len(df_out) == n_eligible
        assert len(df_diag) == len(df)

    def test_raises_on_duplicate_disbursement_fid(self):
        df = _loan_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(DataAlignmentError, match="duplicated disbursement_fid"):
            run_phase_2_2_loan_history_pd_features(df)

    def test_raises_on_non_numeric_coverage_ratio(self):
        df = _loan_df()
        df["follow_up_coverage_ratio_30d"] = df["follow_up_coverage_ratio_30d"].astype(object)
        df.loc[0, "follow_up_coverage_ratio_30d"] = "not-a-number"
        with pytest.raises(DataAlignmentError, match="non-numeric follow_up_coverage_ratio_30d"):
            run_phase_2_2_loan_history_pd_features(df)

    def test_raises_on_null_coverage_ratio(self):
        """A genuinely null ratio (distinct from an unparseable string) must
        also hard-fail -- the SQL only produces NULL for
        CENSORED_INVALID_OBSERVATION_WINDOW, itself a build-blocking
        integrity failure, never for an ordinary unobserved-but-valid loan."""
        df = _loan_df()
        df.loc[0, "follow_up_coverage_ratio_30d"] = np.nan
        with pytest.raises(DataAlignmentError, match="null follow_up_coverage_ratio_30d"):
            run_phase_2_2_loan_history_pd_features(df)

    def test_raises_on_out_of_range_coverage_ratio(self):
        df = _loan_df()
        df.loc[0, "follow_up_coverage_ratio_30d"] = 1.5
        with pytest.raises(DataAlignmentError, match="outside \\[0, 1\\]"):
            run_phase_2_2_loan_history_pd_features(df)

    def test_raises_on_negative_coverage_ratio(self):
        df = _loan_df()
        df.loc[0, "follow_up_coverage_ratio_30d"] = -0.1
        with pytest.raises(DataAlignmentError, match="outside \\[0, 1\\]"):
            run_phase_2_2_loan_history_pd_features(df)

    def test_raises_on_non_binary_label_eligible(self):
        df = _loan_df()
        df.loc[0, "label_eligible_30d"] = 2
        with pytest.raises(DataAlignmentError, match="strictly 0/1"):
            run_phase_2_2_loan_history_pd_features(df)

    def test_coverage_ratio_and_eligibility_normalized_to_numeric_dtype(self):
        """Validated values are written back onto the frame before
        df_diagnostics is built -- otherwise a numeric-string-typed column
        can pass validation but stay object-dtyped, silently breaking a
        downstream .mean() aggregation (e.g. the audit summary)."""
        df = _loan_df()
        df["follow_up_coverage_ratio_30d"] = df["follow_up_coverage_ratio_30d"].astype(str)
        df_out, df_diag = run_phase_2_2_loan_history_pd_features(df)
        assert pd.api.types.is_numeric_dtype(df_diag["follow_up_coverage_ratio_30d"])
        assert pd.api.types.is_integer_dtype(df_diag["label_eligible_30d"])
        assert "bad_state_1dpd_7d" in df_out.columns  # sanity: pipeline still completed


class TestApplySnapshotToTrainingColumnMap:
    def test_renames_known_columns(self):
        df = pd.DataFrame(
            {
                "has_unresolved_loan_at_snapshot": [0, 1],
                "observed_loan_count": [1, 2],
            }
        )
        out = apply_snapshot_to_training_column_map(df)
        assert "has_unresolved_loan_at_scoring" in out.columns
        assert "observed_prior_loan_count" in out.columns
        assert "has_unresolved_loan_at_snapshot" not in out.columns

    def test_leaves_unmapped_columns_untouched(self):
        df = pd.DataFrame({"has_observed_prior_state": [1, 0]})
        out = apply_snapshot_to_training_column_map(df)
        assert "has_observed_prior_state" in out.columns

    def test_map_values_are_all_at_scoring_names(self):
        """Sanity check the map itself points at real training-side names."""
        for snapshot_col, training_col in SNAPSHOT_TO_TRAINING_COLUMN_MAP.items():
            assert snapshot_col != training_col


class TestRunPhase22LoanHistoryPdFeaturesInference:
    def _scoring_df(self, n: int = 20) -> pd.DataFrame:
        return pd.DataFrame({"agent_msisdn": [f"256{i:07d}" for i in range(n)]})

    def _history_df(self, n: int = 20) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            {
                "msisdn": [f"256{i:07d}" for i in range(n)],
                "observed_loan_count": rng.integers(0, 5, n),
                "prior_loan_count_180d": rng.integers(0, 5, n),
                "prior_active_loan_months_180d": rng.integers(0, 4, n),
                "has_unresolved_loan_at_snapshot": rng.integers(0, 2, n),
            }
        )

    def test_row_count_preserved(self):
        df_pd = self._scoring_df()
        df_history = self._history_df()
        out = run_phase_2_2_loan_history_pd_features_inference(df_pd, df_history)
        assert len(out) == len(df_pd)

    def test_columns_renamed_to_training_convention(self):
        df_pd = self._scoring_df()
        df_history = self._history_df()
        out = run_phase_2_2_loan_history_pd_features_inference(df_pd, df_history)
        assert "observed_prior_loan_count" in out.columns
        assert "has_unresolved_loan_at_scoring" in out.columns

    def test_thin_file_flag_derived(self):
        df_pd = self._scoring_df()
        df_history = self._history_df()
        out = run_phase_2_2_loan_history_pd_features_inference(df_pd, df_history)
        assert "thin_file_flag" in out.columns
        assert out["thin_file_flag"].isin([0, 1]).all()

    def test_no_label_derived(self):
        """There is no bad_state at inference time -- must not be synthesized."""
        df_pd = self._scoring_df()
        df_history = self._history_df()
        out = run_phase_2_2_loan_history_pd_features_inference(df_pd, df_history)
        assert "bad_state" not in out.columns

    def test_raises_on_duplicate_history_rows(self):
        df_pd = self._scoring_df()
        df_history = pd.concat([self._history_df(), self._history_df().iloc[[0]]], ignore_index=True)
        with pytest.raises(DataAlignmentError):
            run_phase_2_2_loan_history_pd_features_inference(df_pd, df_history)

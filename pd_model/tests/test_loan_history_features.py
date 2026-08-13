"""Tests for pd_model.preprocessing.loan_history_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    return pd.DataFrame(
        {
            "msisdn": [f"256{i:07d}" for i in range(n)],
            "disbursement_fid": [f"D{i:06d}" for i in range(n)],
            "split": split,
            "loan_date": pd.to_datetime("2025-09-30"),
            "bad_state_3dpd_30d": rng.integers(0, 2, n),
            "bad_state_1dpd_7d": rng.integers(0, 2, n),
            "observed_prior_loan_count": rng.integers(0, 10, n),
            "active_loan_days_aging_at_scoring": rng.integers(0, 10, n).astype(float),
            "disbursement_amount_ugx": rng.exponential(50000, n),
            "max_days_aging_7d": rng.integers(0, 10, n).astype(float),
            "max_days_aging_30d": rng.integers(0, 30, n).astype(float),
            "rollover_observed_7d": rng.integers(0, 2, n),
            "rollover_observed_30d": rng.integers(0, 2, n),
            "terminal_state_observed_30d": rng.integers(0, 2, n),
            "outcome_state_row_count_30d": rng.integers(0, 30, n),
            "outcome_observed_date_count_30d": rng.integers(0, 30, n),
            "first_outcome_state_date": pd.to_datetime("2025-10-01"),
            "last_outcome_state_date": pd.to_datetime("2025-10-28"),
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


class TestDeriveThinFileFlag:
    def test_thin_file_iff_zero_prior_loans(self):
        df = _loan_df()
        out = derive_thin_file_flag(df)
        expected = (out["observed_prior_loan_count"] == 0).astype(int)
        assert (out["thin_file_flag"] == expected).all()

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

    def test_is_new_agent_inverse_of_thin_file(self):
        df = _loan_df()
        out = derive_thin_file_flag(df)
        assert (out["is_new_agent"] == out["thin_file_flag"]).all()


class TestRunPhase22LoanHistoryPdFeatures:
    def test_renames_msisdn_to_agent_msisdn(self):
        df_out, _diag = run_phase_2_2_loan_history_pd_features(_loan_df())
        assert "agent_msisdn" in df_out.columns
        assert "msisdn" not in df_out.columns

    def test_row_count_preserved(self):
        df = _loan_df()
        df_out, _diag = run_phase_2_2_loan_history_pd_features(df)
        assert len(df_out) == len(df)

    def test_bad_state_and_thin_file_flag_present(self):
        df_out, _diag = run_phase_2_2_loan_history_pd_features(_loan_df())
        assert "bad_state" in df_out.columns
        assert "thin_file_flag" in df_out.columns

    def test_label_diagnostics_stripped_from_feature_frame(self):
        """The 9 label-diagnostic columns must never reach the feature-eligible
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

    def test_works_without_diagnostics_present(self):
        """Diagnostics are optional inputs -- an already-stripped file must
        still work, returning an empty (but correctly keyed) diagnostics frame."""
        df = _loan_df().drop(columns=list(LABEL_DIAGNOSTIC_COLUMNS))
        df_out, df_diag = run_phase_2_2_loan_history_pd_features(df)
        assert len(df_out) == len(df)
        assert list(df_diag.columns) == ["agent_msisdn", "disbursement_fid"]


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

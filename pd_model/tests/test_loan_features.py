"""Tests for pd_model.preprocessing.loan_features."""

import numpy as np
import pandas as pd
import pytest

from pd_model.preprocessing.loan_features import (
    add_thin_file_flags,
    classify_agent_loan_status,
    compute_bad_flags,
    leakage_audit_phase_2_2,
)


def _pd_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "agent_msisdn": [f"256{i:07d}" for i in range(n)],
            "snapshot_dt": pd.to_datetime("2025-09-30"),
            "has_ever_loan": rng.integers(0, 2, n),
            "bad_state_30D": rng.integers(0, 2, n),
            "repayment_coverage_1M": rng.uniform(0, 1, n),
            "penalties_1M": rng.integers(0, 3, n),
            "repayment_val_1M": rng.integers(0, 2, n),
        }
    )


class TestClassifyAgentLoanStatus:
    def test_adds_expected_columns(self):
        df = classify_agent_loan_status(_pd_df())
        assert "has_loan_history" in df.columns
        assert "is_new_agent" in df.columns

    def test_binary_values(self):
        df = classify_agent_loan_status(_pd_df())
        assert df["has_loan_history"].isin([0, 1]).all()
        assert df["is_new_agent"].isin([0, 1]).all()

    def test_is_new_agent_inverse_of_has_loan(self):
        df = classify_agent_loan_status(_pd_df())
        assert (df["is_new_agent"] == (1 - df["has_loan_history"])).all()

    def test_raises_without_has_ever_loan(self):
        df = _pd_df().drop(columns=["has_ever_loan"])
        with pytest.raises(ValueError, match="has_ever_loan"):
            classify_agent_loan_status(df)


class TestAddThinFileFlags:
    def test_adds_columns(self):
        df = classify_agent_loan_status(_pd_df())
        df = add_thin_file_flags(df)
        assert "thin_file_flag" in df.columns
        assert "thin_file_pd_prior" in df.columns

    def test_thin_file_flag_binary(self):
        df = classify_agent_loan_status(_pd_df())
        df = add_thin_file_flags(df)
        assert df["thin_file_flag"].isin([0, 1]).all()

    def test_prior_only_for_thin_file(self):
        df = classify_agent_loan_status(_pd_df())
        df = add_thin_file_flags(df)
        assert (df.loc[df["thin_file_flag"] == 0, "thin_file_pd_prior"] == 0.0).all()
        assert (df.loc[df["thin_file_flag"] == 1, "thin_file_pd_prior"] == 0.12).all()


class TestComputeBadFlags:
    def test_bad_state_is_binary(self):
        df = compute_bad_flags(_pd_df())
        assert df["bad_state"].isin([0, 1]).all()

    def test_hard_bad_flag_is_binary(self):
        df = compute_bad_flags(_pd_df())
        assert df["hard_bad_flag"].isin([0, 1]).all()

    def test_no_nans_in_bad_state(self):
        df = compute_bad_flags(_pd_df())
        assert not df["bad_state"].isna().any()

    def test_raises_without_bad_state_30d(self):
        df = _pd_df().drop(columns=["bad_state_30D"])
        with pytest.raises(RuntimeError, match="bad_state_30D"):
            compute_bad_flags(df)


class TestLeakageAudit:
    def _clean_df(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            {
                "agent_msisdn": range(n),
                "snapshot_dt": pd.to_datetime("2025-09-30"),
                "feature_a": rng.uniform(0, 1, n),
                "feature_b": rng.uniform(0, 100, n),
                "bad_state": rng.integers(0, 2, n),
            }
        )

    def test_clean_df_passes(self):
        report = leakage_audit_phase_2_2(self._clean_df(), hard_fail=False)
        assert isinstance(report, pd.DataFrame)

    def test_hard_fail_on_high_correlation(self):
        df = self._clean_df()
        # Create a near-perfect label proxy
        df["label_proxy"] = df["bad_state"] + np.random.default_rng(2).uniform(0, 0.001, len(df))
        with pytest.raises(RuntimeError, match="HARD LEAKAGE"):
            leakage_audit_phase_2_2(df, hard_fail=True)

    def test_no_hard_fail_when_disabled(self):
        df = self._clean_df()
        df["label_proxy"] = df["bad_state"]
        report = leakage_audit_phase_2_2(df, hard_fail=False)
        assert "HIGH" in report["severity"].values

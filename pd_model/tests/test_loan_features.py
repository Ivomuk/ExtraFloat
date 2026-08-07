"""Tests for pd_model.preprocessing.loan_features."""

import numpy as np
import pandas as pd
import pytest

from pd_model.preprocessing.loan_features import (
    add_thin_file_flags,
    classify_agent_loan_status,
    compute_bad_flags,
    leakage_audit_phase_2_2,
    run_phase_2_2_repayment_pd_features,
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

    def test_blacklisted_medium_downgraded_to_info(self):
        # A column with high-but-not-perfect correlation produces MEDIUM severity.
        # When that column is in the blacklist it should be downgraded to INFO.
        rng = np.random.default_rng(3)
        df = self._clean_df(200)
        df["routing_col"] = df["bad_state"] + rng.uniform(0, 0.3, 200)
        blacklist = frozenset({"routing_col"})
        report = leakage_audit_phase_2_2(df, hard_fail=False, pd_feature_blacklist=blacklist)
        routing_rows = report[report["column"] == "routing_col"]
        assert not routing_rows.empty
        assert (routing_rows["severity"] == "INFO").all(), (
            f"Expected INFO for blacklisted MEDIUM column, got: {routing_rows['severity'].tolist()}"
        )


class TestHasEverLoanDistinctMonths:
    """has_ever_loan should use distinct active months, not raw loan count."""

    def _base_pd_df(self, msisdn="256700000001"):
        return pd.DataFrame(
            {
                "agent_msisdn": [msisdn],
                "snapshot_dt": [pd.Timestamp("2025-11-15")],
                "bad_state_30D": [0],
                "repayment_coverage_1M": [0.0],
                "penalties_1M": [0],
                "repayment_val_1M": [0],
            }
        )

    def _repayment_df(self, m1=0, m2=0, m3=0, m4=0, msisdn="256700000001"):
        return pd.DataFrame(
            {
                "agent_msisdn": [msisdn],
                "snapshot_dt": [pd.Timestamp("2025-11-15")],
                "disbursement_vol_m1": [m1],
                "disbursement_vol_m2": [m2],
                "disbursement_vol_m3": [m3],
                "disbursement_vol_m4": [m4],
                "disbursement_val_1m": [float(m1 + m2 + m3 + m4) * 1000],
            }
        )

    def test_all_loans_same_month_is_thin_file(self):
        # 30 loans in month 1 only -> distinct_months=1 < 4 -> thin-file
        pd_df = self._base_pd_df()
        rep_df = self._repayment_df(m1=30, m2=0, m3=0)
        result, _ = run_phase_2_2_repayment_pd_features(pd_df, rep_df)
        assert result["has_ever_loan"].iloc[0] == 0

    def test_loans_across_4_months_is_thick_file(self):
        # 2 loans each in m1..m4 -> distinct_months=4 >= 4, total=8 >= 5 -> thick-file
        pd_df = self._base_pd_df()
        rep_df = self._repayment_df(m1=2, m2=2, m3=2, m4=2)
        result, _ = run_phase_2_2_repayment_pd_features(pd_df, rep_df)
        assert result["has_ever_loan"].iloc[0] == 1

    def test_3_months_is_thin_file(self):
        # threshold is 4 months: 3 distinct months with enough loans -> thin-file
        pd_df = self._base_pd_df()
        rep_df = self._repayment_df(m1=5, m2=5, m3=5, m4=0)
        result, _ = run_phase_2_2_repayment_pd_features(pd_df, rep_df)
        assert result["has_ever_loan"].iloc[0] == 0

    def test_4_months_but_too_few_loans_is_thin_file(self):
        # 1 loan each in m1..m4 -> distinct_months=4 but total=4 < 5 -> thin-file
        pd_df = self._base_pd_df()
        rep_df = self._repayment_df(m1=1, m2=1, m3=1, m4=1)
        result, _ = run_phase_2_2_repayment_pd_features(pd_df, rep_df)
        assert result["has_ever_loan"].iloc[0] == 0

    def test_5_loans_but_single_month_is_thin_file(self):
        # 5 loans all in m1 -> total >= 5 but distinct_months=1 < 4 -> thin-file
        pd_df = self._base_pd_df()
        rep_df = self._repayment_df(m1=5, m2=0, m3=0)
        result, _ = run_phase_2_2_repayment_pd_features(pd_df, rep_df)
        assert result["has_ever_loan"].iloc[0] == 0

    def test_distinct_loan_months_col_present(self):
        pd_df = self._base_pd_df()
        rep_df = self._repayment_df(m1=5, m2=3, m3=0)
        result, _ = run_phase_2_2_repayment_pd_features(pd_df, rep_df)
        assert "distinct_loan_months" in result.columns
        assert "total_loans_6m" in result.columns
        assert result["distinct_loan_months"].iloc[0] == 2
        assert result["total_loans_6m"].iloc[0] == 8

    def test_fallback_when_no_monthly_cols(self):
        # When only disbursement_val_1m is present (no _mN cols),
        # any disbursement activity -> has_ever_loan=1 (original binary logic)
        pd_df = self._base_pd_df()
        rep_df = pd.DataFrame(
            {
                "agent_msisdn": ["256700000001"],
                "snapshot_dt": [pd.Timestamp("2025-11-15")],
                "disbursement_val_1m": [5000.0],
            }
        )
        result, _ = run_phase_2_2_repayment_pd_features(pd_df, rep_df)
        assert result["has_ever_loan"].iloc[0] == 1

    def test_value_columns_not_counted_in_total_loans(self):
        # disbursement_val_m1 (monetary value in UGX) also ends in _m1 and matches
        # the monthly column regex. It must NOT be included in total_loans_6m.
        # Agent has 3 vol loans in m1/m2/m3 but 50,000 UGX value in m1 alone —
        # total_loans_6m must be 3, not 50,003.
        pd_df = self._base_pd_df()
        rep_df = pd.DataFrame(
            {
                "agent_msisdn": ["256700000001"],
                "snapshot_dt": [pd.Timestamp("2025-11-15")],
                "disbursement_vol_m1": [1],
                "disbursement_vol_m2": [1],
                "disbursement_vol_m3": [1],
                "disbursement_val_m1": [50000.0],  # UGX value — must not be summed into count
                "disbursement_val_1m": [50000.0],
            }
        )
        result, _ = run_phase_2_2_repayment_pd_features(pd_df, rep_df)
        assert result["total_loans_6m"].iloc[0] == 3, (
            f"total_loans_6m={result['total_loans_6m'].iloc[0]}; "
            "disbursement_val_m1 (UGX value) must not be added to loan count"
        )
        assert result["distinct_loan_months"].iloc[0] == 3

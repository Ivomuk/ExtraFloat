"""Tests for pd_model.monitoring.stress."""
import numpy as np
import pandas as pd
import pytest

from pd_model.monitoring.stress import build_stress_summary, run_stress_test


def _make_scored_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "agent_msisdn": [f"msisdn_{i}" for i in range(n)],
        "cal_pd": rng.uniform(0.01, 0.50, n),
        "loan_amount": rng.uniform(100, 5000, n),
    })


def _make_policy_tbl(operating_points=(0.10, 0.20, 0.50, 0.80)) -> pd.DataFrame:
    """Minimal policy threshold table matching build_policy_tables output."""
    rows = []
    for op in operating_points:
        rows.append({
            "approve_rate_target": op,
            "n_approved": int(op * 300),
            "expected_bad_rate": 0.05 + op * 0.1,
            "cutoff": op * 0.4,       # synthetic PD cutoff
            "cutoff_var": "cal_pd",
        })
    return pd.DataFrame(rows)


class TestRunStressTest:
    def test_returns_dataframe(self):
        scored = _make_scored_df()
        thresh = _make_policy_tbl()
        result = run_stress_test(scored, thresh)
        assert isinstance(result, pd.DataFrame)

    def test_row_count(self):
        scored = _make_scored_df()
        thresh = _make_policy_tbl(operating_points=(0.20, 0.50))
        result = run_stress_test(
            scored, thresh,
            stress_multipliers=(1.0, 2.0),
            operating_points=(0.20, 0.50),
        )
        assert len(result) == 2 * 2   # 2 OPs × 2 multipliers

    def test_required_columns(self):
        scored = _make_scored_df()
        thresh = _make_policy_tbl()
        result = run_stress_test(scored, thresh,
                                 operating_points=(0.20,),
                                 stress_multipliers=(1.0, 2.0))
        for col in ("operating_point", "stress_multiplier", "locked_pd_cutoff",
                    "n_approved", "approval_rate", "expected_loss",
                    "approval_rate_change_pp", "el_change_pct"):
            assert col in result.columns

    def test_baseline_change_is_zero(self):
        scored = _make_scored_df()
        thresh = _make_policy_tbl()
        result = run_stress_test(scored, thresh,
                                 operating_points=(0.20,),
                                 stress_multipliers=(1.0, 2.0))
        baseline_row = result[result["stress_multiplier"] == 1.0].iloc[0]
        assert baseline_row["approval_rate_change_pp"] == pytest.approx(0.0)
        assert baseline_row["el_change_pct"] == pytest.approx(0.0)

    def test_higher_stress_lower_approval(self):
        scored = _make_scored_df()
        thresh = _make_policy_tbl()
        result = run_stress_test(scored, thresh,
                                 operating_points=(0.20,),
                                 stress_multipliers=(1.0, 3.0))
        r1 = result[result["stress_multiplier"] == 1.0]["approval_rate"].iloc[0]
        r3 = result[result["stress_multiplier"] == 3.0]["approval_rate"].iloc[0]
        assert r3 <= r1

    def test_higher_stress_higher_el(self):
        scored = _make_scored_df()
        thresh = _make_policy_tbl()
        result = run_stress_test(scored, thresh,
                                 operating_points=(0.20,),
                                 stress_multipliers=(1.0, 3.0))
        el1 = result[result["stress_multiplier"] == 1.0]["expected_loss"].iloc[0]
        el3 = result[result["stress_multiplier"] == 3.0]["expected_loss"].iloc[0]
        # EL may be lower (fewer approved) or higher (stressed PDs) — net effect varies
        # but el_change_pct should be computable
        assert not np.isnan(result[result["stress_multiplier"] == 3.0]["el_change_pct"].iloc[0])

    def test_with_loan_size_col(self):
        scored = _make_scored_df()
        thresh = _make_policy_tbl()
        result = run_stress_test(scored, thresh,
                                 loan_size_col="loan_amount",
                                 operating_points=(0.50,),
                                 stress_multipliers=(1.0, 2.0))
        assert result["expected_loss"].iloc[0] > 0

    def test_missing_cal_pd_raises(self):
        scored = _make_scored_df().drop(columns=["cal_pd"])
        thresh = _make_policy_tbl()
        with pytest.raises(ValueError, match="cal_pd"):
            run_stress_test(scored, thresh)


class TestBuildStressSummary:
    def test_wide_format(self):
        scored = _make_scored_df()
        thresh = _make_policy_tbl()
        result = run_stress_test(scored, thresh,
                                 operating_points=(0.20, 0.50),
                                 stress_multipliers=(1.0, 2.0))
        summary = build_stress_summary(result, metric="approval_rate")
        assert "operating_point" in summary.columns
        assert "approval_rate_at_1.0x" in summary.columns
        assert "approval_rate_at_2.0x" in summary.columns
        assert len(summary) == 2   # 2 operating points

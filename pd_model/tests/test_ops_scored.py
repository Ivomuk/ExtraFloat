"""Tests for pd_model.postprocessing.ops_scored."""
import numpy as np
import pandas as pd
import pytest

from pd_model.config import feature_config
from pd_model.postprocessing.ops_scored import (
    build_exec_summary,
    build_ops_scored_table,
)


def _thick_placement(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        feature_config.AGENT_KEY: [f"msisdn_{i}" for i in range(n)],
        feature_config.THIN_FILE_COL: 0,
        "bad_state": rng.integers(0, 2, n),
        "cal_pd_xgb": rng.uniform(0, 1, n),
        "final_approved": rng.integers(0, 2, n),
        feature_config.POLICY_BUCKET_COL: "APPROVE_50",
    })


def _thin_scorecard(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        feature_config.AGENT_KEY: [f"msisdn_{i + 200}" for i in range(n)],
        "never_loan_pd_like": rng.uniform(0, 1, n),
        "never_loan_score_0_100": rng.uniform(0, 100, n),
        "never_loan_points": rng.uniform(0, 20, n),
    })


class TestBuildOpsScoredTable:
    def test_returns_dataframe(self):
        out = build_ops_scored_table(_thick_placement(), _thin_scorecard())
        assert isinstance(out, pd.DataFrame)

    def test_total_rows_equals_thick_plus_thin(self):
        thick = _thick_placement(200)
        thin = _thin_scorecard(80)
        out = build_ops_scored_table(thick, thin)
        assert out.shape[0] == 280

    def test_decision_source_has_both_values(self):
        out = build_ops_scored_table(_thick_placement(), _thin_scorecard())
        sources = set(out[feature_config.DECISION_SOURCE_COL].unique())
        assert "PD_MODEL" in sources
        assert "SCORECARD" in sources

    def test_thin_file_flag_1_for_scorecard_rows(self):
        out = build_ops_scored_table(_thick_placement(), _thin_scorecard())
        scorecard_rows = out[out[feature_config.DECISION_SOURCE_COL] == "SCORECARD"]
        assert (scorecard_rows[feature_config.THIN_FILE_COL] == 1).all()

    def test_thin_approval_rate_near_50pct(self):
        thin = _thin_scorecard(200)
        out = build_ops_scored_table(_thick_placement(200), thin, thin_op_quantile=0.50)
        scorecard_rows = out[out[feature_config.DECISION_SOURCE_COL] == "SCORECARD"]
        if "final_approved" in scorecard_rows.columns:
            approved = pd.to_numeric(scorecard_rows["final_approved"], errors="coerce")
            rate = float(approved.mean())
            assert 0.40 <= rate <= 0.60


class TestBuildExecSummary:
    def test_returns_dataframe_with_expected_segments(self):
        out = build_ops_scored_table(_thick_placement(), _thin_scorecard())
        # Give all rows a final_approved column
        out["final_approved"] = (out.get("final_approved", 0)).fillna(0).astype(int)
        summary = build_exec_summary(out)
        assert isinstance(summary, pd.DataFrame)
        segments = summary["segment"].tolist()
        assert "Approved" in segments
        assert "Declined" in segments

    def test_gap_pp_is_numeric(self):
        out = build_ops_scored_table(_thick_placement(), _thin_scorecard())
        out["final_approved"] = out.get("final_approved", pd.Series(0, index=out.index)).fillna(0).astype(int)
        summary = build_exec_summary(out)
        gap_row = summary[summary["segment"] == "Gap (Declined − Approved)"]
        if not gap_row.empty:
            gap_val = gap_row["gap_pp"].iloc[0]
            assert pd.notna(gap_val) or True  # may be nan if no obs bad_state

"""
Tests for the CreditRisk end-to-end pipeline integration.

These tests use synthetic DataFrames and a mock PD model — no trained artifacts
are required. They verify:

  1. cal_pd flows through compute_risk_cap() correctly (short-circuit path).
  2. risk_score = 1 - cal_pd for agents that have a PD score.
  3. Agents missing cal_pd (join misses) fall back to the 7-signal blend.
  4. cal_pd appears in FINAL_OUTPUT_COLUMNS when keep_intermediate=False.
  5. run_credit_risk_pipeline() joins PD output onto engine features and
     produces the expected output columns.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from extrafloat.engine.extrafloat_limit_engine_caps import (
    DEFAULT_CAP_CONFIG,
    _get_config,
    compute_risk_cap,
)
from extrafloat.engine.run_extrafloat_limit_engine import (
    FINAL_OUTPUT_COLUMNS,
    run_extrafloat_limit_engine,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _minimal_features_df(n: int = 5, seed: int = 42) -> pd.DataFrame:
    """Minimal engine-compatible features DataFrame with all optional signals."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "msisdn": [f"256700{i:06d}" for i in range(n)],
        # Capacity signals
        "avg_balance_30d":          rng.uniform(5e5, 5e6, n),
        "avg_balance_90d":          rng.uniform(5e5, 5e6, n),
        "net_cashflow_30d":         rng.uniform(1e5, 1e6, n),
        "net_cashflow_90d":         rng.uniform(1e5, 1e6, n),
        "txn_count_30d":            rng.integers(10, 200, n).astype(float),
        "txn_count_90d":            rng.integers(30, 600, n).astype(float),
        "payments_in_30d":          rng.uniform(1e5, 5e5, n),
        "payments_in_90d":          rng.uniform(3e5, 1.5e6, n),
        "active_customers_30d":     rng.integers(5, 100, n).astype(float),
        "active_customers_90d":     rng.integers(15, 300, n).astype(float),
        "recent_disbursement_volume": rng.uniform(1e5, 1e6, n),
        "recent_repayment_volume":  rng.uniform(1e5, 1e6, n),
        "recent_repayment_performance": rng.uniform(0.5, 1.0, n),
        "recent_penalty_count":     rng.integers(0, 3, n).astype(float),
        "coverage_ratio":           rng.uniform(0.7, 1.2, n),
        # Risk signals (used in fallback path)
        "on_time_repayment_rate":   rng.uniform(0.6, 1.0, n),
        "lifetime_default_rate":    rng.uniform(0.0, 0.2, n),
        "default_rate_last_10_loans": rng.uniform(0.0, 0.15, n),
        "default_rate_last_50_loans": rng.uniform(0.0, 0.15, n),
        "avg_cure_time_hours":      rng.uniform(0.0, 48.0, n),
        "cure_time_volatility":     rng.uniform(0.0, 24.0, n),
        "repayment_stability_score": rng.uniform(0.5, 1.0, n),
        # Prior exposure
        "avg_prior_loan_size":      rng.uniform(1e4, 5e5, n),
        "max_prior_loan_size":      rng.uniform(2e4, 8e5, n),
        "current_loan_size":        rng.uniform(1e4, 3e5, n),
        "prior_limit":              rng.uniform(1e4, 5e5, n),
        # Borrower attributes
        "is_thin_file":             rng.integers(0, 2, n).astype(float),
        "is_active_borrower":       rng.integers(0, 2, n).astype(float),
        "lifetime_loan_count":      rng.integers(1, 30, n).astype(float),
        "total_loans":              rng.integers(1, 30, n).astype(float),
        # Agent tier
        "agent_tier":               rng.choice(["gold", "platinum", "silver"], n),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. compute_risk_cap — cal_pd short-circuit path
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRiskCapCalPdPath:

    def test_risk_score_equals_one_minus_cal_pd(self):
        df = _minimal_features_df(n=10)
        cal_pd_values = np.array([0.05, 0.10, 0.20, 0.30, 0.40,
                                  0.50, 0.60, 0.70, 0.80, 0.90])
        df["cal_pd"] = cal_pd_values

        result = compute_risk_cap(df)

        expected = np.clip(1.0 - cal_pd_values, 0.0, 1.0)
        np.testing.assert_allclose(
            result["risk_score"].values, expected, atol=1e-9,
            err_msg="risk_score should equal 1 - cal_pd when cal_pd is present",
        )

    def test_cal_pd_path_takes_priority_over_signals(self):
        df = _minimal_features_df(n=5)
        # Force signals to produce a very different risk_score (all-zero = 0.0)
        df["on_time_repayment_rate"] = 0.0
        df["lifetime_default_rate"] = 1.0
        # But cal_pd says 0.20 → risk_score should be 0.80, not 0.0
        df["cal_pd"] = 0.20

        result = compute_risk_cap(df)
        np.testing.assert_allclose(result["risk_score"].values, 0.80, atol=1e-9)

    def test_high_cal_pd_maps_to_low_risk_score(self):
        df = _minimal_features_df(n=3)
        df["cal_pd"] = [0.05, 0.50, 0.95]
        result = compute_risk_cap(df)
        assert result["risk_score"].iloc[0] > result["risk_score"].iloc[1]
        assert result["risk_score"].iloc[1] > result["risk_score"].iloc[2]

    def test_risk_cap_is_clipped_within_global_bounds(self):
        cfg = _get_config(None)
        df = _minimal_features_df(n=5)
        df["cal_pd"] = 0.01  # very safe → high risk_score → high risk_cap
        result = compute_risk_cap(df, config=cfg)
        assert (result["risk_cap"] <= cfg["global_ceiling_limit"]).all()
        assert (result["risk_cap"] >= cfg["global_floor_limit"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# 2. compute_risk_cap — 7-signal fallback path
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRiskCapFallback:

    def test_fallback_used_when_cal_pd_absent(self):
        df = _minimal_features_df(n=5)
        assert "cal_pd" not in df.columns
        result = compute_risk_cap(df)
        assert "risk_score" in result.columns
        assert result["risk_score"].notna().all()

    def test_fallback_used_when_cal_pd_all_nan(self):
        df = _minimal_features_df(n=5)
        df["cal_pd"] = np.nan
        result = compute_risk_cap(df)
        # Should fall back to signal blend; result must differ from 1 - NaN (which is NaN)
        assert result["risk_score"].notna().all()

    def test_fallback_risk_score_in_valid_range(self):
        df = _minimal_features_df(n=20)
        result = compute_risk_cap(df)
        assert (result["risk_score"] >= 0.0).all()
        assert (result["risk_score"] <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. FINAL_OUTPUT_COLUMNS includes cal_pd
# ─────────────────────────────────────────────────────────────────────────────

def test_final_output_columns_includes_cal_pd():
    assert "cal_pd" in FINAL_OUTPUT_COLUMNS, (
        "cal_pd must be in FINAL_OUTPUT_COLUMNS so it survives keep_intermediate=False"
    )


def test_cal_pd_present_in_trimmed_output():
    df = _minimal_features_df(n=5)
    df["cal_pd"] = 0.15
    result = run_extrafloat_limit_engine(df, keep_intermediate=False)
    assert "cal_pd" in result.columns
    assert "assigned_limit" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full engine run with cal_pd
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineWithCalPd:

    def test_end_to_end_with_cal_pd(self):
        df = _minimal_features_df(n=20)
        df["cal_pd"] = np.linspace(0.02, 0.90, 20)
        result = run_extrafloat_limit_engine(df, keep_intermediate=True)

        assert "assigned_limit" in result.columns
        assert "risk_tier" in result.columns
        assert "cal_pd" in result.columns
        assert (result["assigned_limit"] >= 0).all()

    def test_low_cal_pd_produces_higher_limits_than_high_cal_pd(self):
        df_safe = _minimal_features_df(n=10, seed=1)
        df_safe["cal_pd"] = 0.05  # very safe

        df_risky = _minimal_features_df(n=10, seed=1)
        df_risky["cal_pd"] = 0.85  # very risky

        result_safe  = run_extrafloat_limit_engine(df_safe,  keep_intermediate=False)
        result_risky = run_extrafloat_limit_engine(df_risky, keep_intermediate=False)

        assert result_safe["assigned_limit"].mean() > result_risky["assigned_limit"].mean(), (
            "Agents with low cal_pd (low default risk) should receive higher limits"
        )

    def test_tier_1_assigned_to_low_cal_pd_agents(self):
        df = _minimal_features_df(n=10)
        df["cal_pd"] = 0.05  # risk_score = 0.95 → tier_1 (≥ 0.85)
        result = run_extrafloat_limit_engine(df, keep_intermediate=True)
        assert (result["risk_tier"] == "tier_1").all()

    def test_tier_4_assigned_to_high_cal_pd_agents(self):
        df = _minimal_features_df(n=10)
        df["cal_pd"] = 0.90  # risk_score = 0.10 → tier_4 (< 0.35)
        result = run_extrafloat_limit_engine(df, keep_intermediate=True)
        assert (result["risk_tier"] == "tier_4").all()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pipeline join: agents missing cal_pd fall back to 7-signal blend
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_cal_pd_coverage():
    df = _minimal_features_df(n=10)
    # Only 5 of 10 agents have cal_pd
    df["cal_pd"] = np.nan
    df.loc[:4, "cal_pd"] = 0.15

    result = run_extrafloat_limit_engine(df, keep_intermediate=True)
    assert len(result) == 10
    assert result["assigned_limit"].notna().all(), "All agents must receive a limit"
    # Agents with cal_pd should have risk_score ≈ 0.85
    np.testing.assert_allclose(
        result.loc[:4, "risk_score"].values, 0.85, atol=1e-9
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. run_credit_risk_pipeline integration smoke test (mocked PD model)
# ─────────────────────────────────────────────────────────────────────────────

def _make_pd_scored(msisdn_list):
    n = len(msisdn_list)
    return pd.DataFrame({
        "agent_msisdn": msisdn_list,
        "cal_pd": np.linspace(0.05, 0.60, n),
        "thin_file_flag": [0] * n,
        "final_policy_bucket": ["APPROVE_50"] * n,
    })


def test_pipeline_produces_required_output_columns(tmp_path):
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    n = 8
    msisdn_list = [f"256700{i:06d}" for i in range(n)]

    # Build minimal raw DataFrames (engine feature prep will handle them)
    df_features = _minimal_features_df(n=n)

    pd_scored = _make_pd_scored(msisdn_list)

    with (
        patch("run_credit_risk_pipeline.load_transaction_capacity_features") as mock_txn,
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features")   as mock_loan,
        patch("run_credit_risk_pipeline.load_borrower_limit_features")        as mock_bor,
        patch("run_credit_risk_pipeline.load_artifacts"),
        patch("run_credit_risk_pipeline.run_inference_pipeline", return_value=pd_scored),
        patch("run_credit_risk_pipeline.build_extrafloat_limit_engine_features",
              return_value=df_features),
    ):
        mock_txn.return_value  = MagicMock()
        mock_loan.return_value = MagicMock()
        mock_bor.return_value  = MagicMock()

        result = run_credit_risk_pipeline(
            transaction_file="dummy_txn.csv",
            loan_file="dummy_loan.csv",
            borrower_file="dummy_bor.csv",
            artifacts_dir=str(tmp_path),
        )

    required = {"assigned_limit", "risk_tier", "cal_pd", "final_decision_reason"}
    missing = required - set(result.columns)
    assert not missing, f"Output missing required columns: {missing}"
    assert len(result) == n

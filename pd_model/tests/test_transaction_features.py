"""Tests for pd_model.preprocessing.transaction_features."""

import numpy as np
import pandas as pd
import pytest

from pd_model.preprocessing.transaction_features import run_phase_2_1_richer_tx_behaviour


def _base_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "agent_msisdn": [f"256{i:07d}" for i in range(n)],
            "snapshot_dt": pd.to_datetime("2025-09-30"),
            "tbl_dt": pd.to_datetime("2025-09-30"),
            "vol_1m": rng.uniform(0, 1000, n),
            "vol_3m": rng.uniform(0, 3000, n),
            "vol_6m": rng.uniform(0, 6000, n),
            "commission": rng.uniform(0, 500, n),
            "commission_cluster_mean": rng.uniform(100, 400, n),
            "account_balance": rng.uniform(-100, 5000, n),
            "average_balance": rng.uniform(0, 5000, n),
            "cash_in_value_3m": rng.uniform(0, 3000, n),
            "cash_out_value_3m": rng.uniform(0, 3000, n),
        }
    )


class TestRunPhase21:
    def test_returns_dataframe(self):
        df = run_phase_2_1_richer_tx_behaviour(_base_df())
        assert isinstance(df, pd.DataFrame)

    def test_creates_inactivity_flags(self):
        df = run_phase_2_1_richer_tx_behaviour(_base_df())
        assert "is_fully_inactive_6m" in df.columns
        assert "is_consecutively_inactive" in df.columns
        assert df["is_fully_inactive_6m"].isin([0, 1]).all()

    def test_creates_volume_trajectory_features(self):
        df = run_phase_2_1_richer_tx_behaviour(_base_df())
        assert "vol_share_1m_of_3m" in df.columns
        assert "vol_share_3m_of_6m" in df.columns
        assert "vol_monthly_volatility_cv" in df.columns

    def test_creates_balance_stress_flags(self):
        df = run_phase_2_1_richer_tx_behaviour(_base_df())
        assert "low_balance_flag" in df.columns
        assert "balance_drawdown_flag" in df.columns

    def test_creates_net_cash_flow(self):
        df = run_phase_2_1_richer_tx_behaviour(_base_df())
        assert "net_cash_flow_3m" in df.columns
        assert "net_cash_flow_negative_flag" in df.columns

    def test_num_inactive_horizons_range(self):
        df = run_phase_2_1_richer_tx_behaviour(_base_df())
        if "num_inactive_horizons" in df.columns:
            assert df["num_inactive_horizons"].between(0, 3).all()

    def test_all_inactive_agent(self):
        """Agent with zero vol in all windows should be fully inactive."""
        df = _base_df(5)
        df["vol_1m"] = 0.0
        df["vol_3m"] = 0.0
        df["vol_6m"] = 0.0
        result = run_phase_2_1_richer_tx_behaviour(df)
        assert result["is_fully_inactive_6m"].eq(1).all()

    def test_missing_optional_columns_graceful(self):
        """Should not fail if optional columns are absent."""
        df = _base_df().drop(columns=["commission", "account_balance"])
        result = run_phase_2_1_richer_tx_behaviour(df)
        assert isinstance(result, pd.DataFrame)

    def test_requires_agent_msisdn(self):
        df = _base_df().drop(columns=["agent_msisdn"])
        with pytest.raises(ValueError, match="agent_msisdn"):
            run_phase_2_1_richer_tx_behaviour(df)

    def test_does_not_modify_input(self):
        df = _base_df()
        original_cols = set(df.columns)
        run_phase_2_1_richer_tx_behaviour(df)
        assert set(df.columns) == original_cols

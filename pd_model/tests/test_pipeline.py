"""
End-to-end smoke test for the PD model feature pipeline.

Uses a small synthetic dataset (~100 rows) to verify that all steps
can be chained without error and that outputs have the expected shapes
and column presence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pd_model.config import feature_config
from pd_model.config.model_config import DEFAULT_CONFIG
from pd_model.data_prep.pipeline_data import prepare_pd_training_and_validation_data
from pd_model.modeling.scorecard import add_never_loan_scorecard_from_phase_2_1
from pd_model.preprocessing.loan_features import (
    add_thin_file_flags,
    classify_agent_loan_status,
    compute_bad_flags,
)
from pd_model.preprocessing.transaction_features import run_phase_2_1_richer_tx_behaviour
from pd_model.preprocessing.transformations import (
    build_transformed_dataframe,
    get_and_classify_pd_features,
)
from pd_model.scoring.iv_selector import iv_filter_phase_2


# ======================================================================== #
# Synthetic data factory
# ======================================================================== #

def _make_synthetic_dataset(n: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_pd, df_repayments) synthetic DataFrames."""
    rng = np.random.default_rng(0)

    n_train = 80
    snapshots = pd.to_datetime(
        ["2025-09-30"] * n_train + ["2025-11-15"] * (n - n_train)
    )

    df_pd = pd.DataFrame(
        {
            "agent_msisdn": [f"25678{i:05d}" for i in range(n)],
            "snapshot_dt": snapshots,
            "tbl_dt": snapshots,
            "split": ["train"] * n_train + ["validation"] * (n - n_train),
            "has_ever_loan": rng.integers(0, 2, n),
            "bad_state_30D": rng.integers(0, 2, n),
            "vol_1m": rng.exponential(200, n),
            "vol_3m": rng.exponential(600, n),
            "vol_6m": rng.exponential(1200, n),
            "commission": rng.exponential(50, n),
            "commission_cluster_mean": rng.exponential(45, n),
            "account_balance": rng.normal(500, 300, n),
            "average_balance": rng.exponential(400, n),
            "cash_in_value_3m": rng.exponential(800, n),
            "cash_out_value_3m": rng.exponential(700, n),
            "cust_1m": rng.integers(1, 20, n).astype(float),
            "cust_3m": rng.integers(5, 60, n).astype(float),
            "repayment_coverage_1M": rng.uniform(0, 1, n),
            "penalties_1M": rng.integers(0, 3, n).astype(float),
            "repayment_val_1M": rng.integers(0, 2, n).astype(float),
        }
    )

    df_repayments = pd.DataFrame(
        {
            "agent_msisdn": [f"25678{i:05d}" for i in range(n)],
            "snapshot_dt": snapshots,
            "repayment_gap_days": rng.integers(0, 120, n).astype(float),
            "penalties_1M": rng.integers(0, 3, n).astype(float),
            "penalties_3M": rng.integers(0, 5, n).astype(float),
            "penalties_6M": rng.integers(0, 8, n).astype(float),
            "bad_state_30D": rng.integers(0, 2, n).astype(float),
        }
    )

    return df_pd, df_repayments


# ======================================================================== #
# Tests
# ======================================================================== #

class TestEndToEndPipeline:
    @pytest.fixture(scope="class")
    def pipeline_outputs(self):
        cfg = DEFAULT_CONFIG
        df_pd, _ = _make_synthetic_dataset()

        # Phase 2.1
        df_pd = run_phase_2_1_richer_tx_behaviour(df_pd, cfg=cfg)
        assert "is_fully_inactive_6m" in df_pd.columns

        # Classify + label (skip full repayment merge for smoke test)
        df_pd = classify_agent_loan_status(df_pd)
        df_pd = add_thin_file_flags(df_pd, cfg=cfg)
        df_pd = compute_bad_flags(df_pd)

        # Thin-file scorecard
        df_pd = add_never_loan_scorecard_from_phase_2_1(df_pd, cfg=cfg)

        # Feature classification + transformation
        pd_features, log_cols, cap_cols, _, signed_log_cols, _ = get_and_classify_pd_features(df_pd)
        assert len(pd_features) > 0

        df_pd_raw = df_pd.copy(deep=True)
        df_pd_transformed, pd_features_pruned, _ = build_transformed_dataframe(
            df_pd, pd_features, log_cols, cap_cols, signed_log_cols, cfg=cfg
        )

        # Split
        (
            X_train_raw, X_train_trans, y_train,
            X_val_raw, X_val_trans, y_val,
            candidate_features, thin_train, thin_val,
            agent_train, agent_val,
        ) = prepare_pd_training_and_validation_data(
            df_pd_raw=df_pd_raw,
            df_pd_transformed=df_pd_transformed,
            target_col=feature_config.TARGET_COL,
            train_cutoff=pd.Timestamp("2025-09-30"),
            id_cols=[feature_config.AGENT_KEY],
            protected_cols=list(feature_config.PD_FEATURE_BLACKLIST),
            pd_feature_blacklist=feature_config.PD_FEATURE_BLACKLIST,
            forbidden_feature_patterns=feature_config.LEAKAGE_PATTERNS,
            date_cols=feature_config.DATE_COLS,
        )

        # IV
        selected_features, iv_table = iv_filter_phase_2(
            X_train_raw, X_train_trans, y_train, cfg=cfg
        )

        return {
            "df_pd": df_pd,
            "X_train_raw": X_train_raw,
            "X_train_trans": X_train_trans,
            "y_train": y_train,
            "X_val_raw": X_val_raw,
            "y_val": y_val,
            "candidate_features": candidate_features,
            "selected_features": selected_features,
            "iv_table": iv_table,
        }

    def test_train_set_nonempty(self, pipeline_outputs):
        assert len(pipeline_outputs["X_train_raw"]) > 0

    def test_val_set_nonempty(self, pipeline_outputs):
        assert len(pipeline_outputs["X_val_raw"]) > 0

    def test_target_is_binary(self, pipeline_outputs):
        assert pipeline_outputs["y_train"].isin([0, 1]).all()
        assert pipeline_outputs["y_val"].isin([0, 1]).all()

    def test_candidate_features_nonempty(self, pipeline_outputs):
        assert len(pipeline_outputs["candidate_features"]) > 0

    def test_target_not_in_features(self, pipeline_outputs):
        assert feature_config.TARGET_COL not in pipeline_outputs["candidate_features"]

    def test_agent_key_not_in_features(self, pipeline_outputs):
        assert feature_config.AGENT_KEY not in pipeline_outputs["candidate_features"]

    def test_iv_table_has_expected_cols(self, pipeline_outputs):
        cols = pipeline_outputs["iv_table"].columns.tolist()
        assert "feature" in cols
        assert "iv_after" in cols

    def test_scorecard_columns_present(self, pipeline_outputs):
        df = pipeline_outputs["df_pd"]
        thin_agents = df[df["thin_file_flag"] == 1]
        if len(thin_agents) > 0:
            assert "never_loan_points" in df.columns
            assert "never_loan_pd_like" in df.columns

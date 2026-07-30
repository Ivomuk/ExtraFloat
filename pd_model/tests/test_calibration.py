"""Tests for pd_model.modeling.calibration."""
import numpy as np
import pandas as pd
import pytest

from pd_model.config.model_config import ModelConfig
from pd_model.modeling.calibration import (
    add_policy_flags,
    attach_cal_pd,
    build_pd_calibration_map,
    build_policy_tables,
    run_bootstrap_comparison,
    run_locked_policy_pipeline,
)


def _scored_df(n: int = 6000, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = (x + rng.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame({
        "agent_msisdn": [f"msisdn_{i}" for i in range(n)],
        "bad_state": y,
        "raw_score": 1 / (1 + np.exp(-x)),
        "thin_file_flag": rng.integers(0, 2, n),
    })


# Use a relaxed config for testing (lower min_n, min_bads)
_TEST_CFG = ModelConfig(cal_min_n=100, cal_min_bads=5, cal_min_coverage=0.90)


class TestBuildPdCalibrationMap:
    def test_returns_dataframe(self):
        df = _scored_df()
        cal_map = build_pd_calibration_map(df, "xgb", cfg=_TEST_CFG)
        assert isinstance(cal_map, pd.DataFrame)

    def test_has_required_columns(self):
        df = _scored_df()
        cal_map = build_pd_calibration_map(df, "xgb", cfg=_TEST_CFG)
        for col in ("model", "score_min", "score_max", "pd", "ascending_risk"):
            assert col in cal_map.columns

    def test_pd_is_monotone(self):
        df = _scored_df()
        cal_map = build_pd_calibration_map(df, "xgb", cfg=_TEST_CFG)
        pd_vals = cal_map.sort_values("score_min")["pd"].values
        # Isotonic — should be non-decreasing (ascending_risk=True for this data)
        if bool(cal_map["ascending_risk"].iloc[0]):
            assert all(pd_vals[i] <= pd_vals[i + 1] + 1e-9 for i in range(len(pd_vals) - 1))

    def test_fail_closed_small_n(self):
        small_df = _scored_df(50)
        with pytest.raises(ValueError, match="cal_min_n"):
            build_pd_calibration_map(small_df, "xgb")  # default cfg.cal_min_n=5000


class TestAttachCalPd:
    def test_adds_cal_pd_column(self):
        df = _scored_df()
        cal_map = build_pd_calibration_map(df, "xgb", cfg=_TEST_CFG)
        out = attach_cal_pd(df, cal_map, "xgb", cfg=_TEST_CFG)
        assert "cal_pd" in out.columns

    def test_coverage_above_threshold(self):
        df = _scored_df()
        cal_map = build_pd_calibration_map(df, "xgb", cfg=_TEST_CFG)
        out = attach_cal_pd(df, cal_map, "xgb", cfg=_TEST_CFG)
        coverage = out["cal_pd"].notna().mean()
        assert coverage >= _TEST_CFG.cal_min_coverage

    def test_fail_closed_missing_model_key(self):
        df = _scored_df()
        cal_map = build_pd_calibration_map(df, "xgb", cfg=_TEST_CFG)
        with pytest.raises(ValueError, match="no calibration mapping"):
            attach_cal_pd(df, cal_map, "lgb", cfg=_TEST_CFG)


class TestBuildPolicyTables:
    def test_returns_three_outputs(self):
        df = _scored_df()
        cal_map = build_pd_calibration_map(df, "xgb", cfg=_TEST_CFG)
        df_cal = attach_cal_pd(df, cal_map, "xgb", cfg=_TEST_CFG)
        thresh, band, df_sorted = build_policy_tables(df_cal, cfg=_TEST_CFG)
        assert isinstance(thresh, pd.DataFrame)
        assert isinstance(band, pd.DataFrame)
        assert isinstance(df_sorted, pd.DataFrame)

    def test_threshold_table_has_expected_cols(self):
        df = _scored_df()
        cal_map = build_pd_calibration_map(df, "xgb", cfg=_TEST_CFG)
        df_cal = attach_cal_pd(df, cal_map, "xgb", cfg=_TEST_CFG)
        thresh, _, _ = build_policy_tables(df_cal, cfg=_TEST_CFG)
        for col in ("approve_rate_target", "n_approved", "expected_bad_rate", "cutoff"):
            assert col in thresh.columns

    def test_tighter_approval_lower_bad_rate(self):
        df = _scored_df(8000, seed=9)
        cal_map = build_pd_calibration_map(df, "xgb", cfg=_TEST_CFG)
        df_cal = attach_cal_pd(df, cal_map, "xgb", cfg=_TEST_CFG)
        thresh, _, _ = build_policy_tables(df_cal, cfg=_TEST_CFG)
        # Bad rate should generally decrease as approval rate decreases
        br = thresh.sort_values("approve_rate_target")["expected_bad_rate"].values
        # At least first < last (tighter approval = lower bad rate)
        assert br[0] <= br[-1]


class TestAddPolicyFlags:
    def test_adds_approved_columns(self):
        df = _scored_df()
        cal_map = build_pd_calibration_map(df, "xgb", cfg=_TEST_CFG)
        df_cal = attach_cal_pd(df, cal_map, "xgb", cfg=_TEST_CFG)
        thresh, _, _ = build_policy_tables(df_cal, cfg=_TEST_CFG)
        out = add_policy_flags(df_cal, thresh, prefix="xgb", cfg=_TEST_CFG)
        for op in (10, 20, 50, 80):
            assert f"xgb_approved_at_{op}" in out.columns

    def test_approved_flags_are_binary(self):
        df = _scored_df()
        cal_map = build_pd_calibration_map(df, "xgb", cfg=_TEST_CFG)
        df_cal = attach_cal_pd(df, cal_map, "xgb", cfg=_TEST_CFG)
        thresh, _, _ = build_policy_tables(df_cal, cfg=_TEST_CFG)
        out = add_policy_flags(df_cal, thresh, prefix="xgb", cfg=_TEST_CFG)
        assert out["xgb_approved_at_50"].isin([0, 1]).all()


class TestBootstrap:
    def test_returns_dataframe_with_metric_column(self):
        df = _scored_df(1000, seed=3)
        cfg = ModelConfig(cal_min_n=100, cal_min_bads=5, bootstrap_n=50)
        result = run_bootstrap_comparison(df, df, cfg=cfg)
        assert isinstance(result, pd.DataFrame)
        assert "metric" in result.columns
        assert "value" in result.columns

    def test_has_ci_columns(self):
        df = _scored_df(1000)
        cfg = ModelConfig(cal_min_n=100, cal_min_bads=5, bootstrap_n=50)
        result = run_bootstrap_comparison(df, df, cfg=cfg)
        metrics = result["metric"].tolist()
        assert "xgb_ci95_lo" in metrics
        assert "xgb_ci95_hi" in metrics

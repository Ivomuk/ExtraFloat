"""Tests for pd_model.modeling.inference."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from pd_model.config import feature_config
from pd_model.config.model_config import ModelConfig
from pd_model.modeling.inference import ModelArtifacts, align_features, load_artifacts, score_new_agents


def _make_artifacts(tmp_dir: Path, n_features: int = 5) -> ModelArtifacts:
    """Create minimal fake artifacts for testing."""
    from pd_model.modeling.calibration import attach_cal_pd, build_pd_calibration_map, build_policy_tables
    from pd_model.modeling.lgbm_model import train_lgbm
    from pd_model.modeling.xgb_model import train_xgb

    rng = np.random.default_rng(99)
    n = 600
    feat_names = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.normal(0, 1, (n, n_features)), columns=feat_names)
    y = pd.Series((X["feat_0"] + rng.normal(0, 0.5, n) > 0).astype(int))
    split = int(n * 0.7)
    Xtr, ytr = X.iloc[:split], y.iloc[:split]
    Xva, yva = X.iloc[split:], y.iloc[split:]

    cfg = ModelConfig(cal_min_n=100, cal_min_bads=5, cal_min_coverage=0.90)

    xgb_model, _, xgb_val = train_xgb(Xtr, ytr, Xva, yva, cfg=cfg)
    lgb_model, _, lgb_val = train_lgbm(Xtr, ytr, Xva, yva, cfg=cfg)

    # Build combined calibration map for both models (same format as run_pipeline step 12)
    import pandas as _pd

    xgb_cal_map = build_pd_calibration_map(xgb_val, "xgb", cfg=cfg)
    lgb_cal_map = build_pd_calibration_map(lgb_val, "lgb", cfg=cfg)
    cal_map = _pd.concat([xgb_cal_map, lgb_cal_map], ignore_index=True)

    xgb_cal = attach_cal_pd(xgb_val, cal_map, "xgb", cfg=cfg)
    lgb_cal = attach_cal_pd(lgb_val, cal_map, "lgb", cfg=cfg)
    xgb_thresh, _, _ = build_policy_tables(xgb_cal, cfg=cfg)
    lgb_thresh, _, _ = build_policy_tables(lgb_cal, cfg=cfg)

    # Write all artifacts
    joblib.dump(xgb_model, tmp_dir / "xgb_model.joblib")
    joblib.dump(lgb_model, tmp_dir / "lgbm_model.joblib")
    (tmp_dir / "feature_order.json").write_text(json.dumps({"selected_features": feat_names}))
    cal_map.to_csv(tmp_dir / "pd_calibration_map.csv", index=False)
    xgb_thresh.to_csv(tmp_dir / "xgb_policy_thresholds.csv", index=False)
    lgb_thresh.to_csv(tmp_dir / "lgb_policy_thresholds.csv", index=False)
    pd.DataFrame({"feature": feat_names, "transform": "cap"}).to_csv(
        tmp_dir / "transform_report.csv", index=False
    )

    return load_artifacts(tmp_dir)


class TestLoadArtifacts:
    def test_loads_all_fields(self, tmp_path):
        artifacts = _make_artifacts(tmp_path)
        assert artifacts.xgb_model is not None
        assert artifacts.lgb_model is not None
        assert isinstance(artifacts.feature_order, list)
        assert isinstance(artifacts.cal_map, pd.DataFrame)
        assert isinstance(artifacts.xgb_policy_thresholds, pd.DataFrame)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifacts(tmp_path)

    def test_feature_order_is_list(self, tmp_path):
        artifacts = _make_artifacts(tmp_path)
        assert len(artifacts.feature_order) == 5


class TestAlignFeatures:
    def test_columns_match_feature_order(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        aligned = align_features(df, ["a", "c"])
        assert aligned.columns.tolist() == ["a", "c"]

    def test_missing_features_filled_with_nan(self):
        df = pd.DataFrame({"a": [1]})
        aligned = align_features(df, ["a", "missing"])
        assert "missing" in aligned.columns
        assert aligned["missing"].isna().all()

    def test_extra_columns_dropped(self):
        df = pd.DataFrame({"a": [1], "extra": [99]})
        aligned = align_features(df, ["a"])
        assert "extra" not in aligned.columns


class TestScoreNewAgents:
    def test_returns_one_row_per_agent(self, tmp_path):
        artifacts = _make_artifacts(tmp_path)
        n = 50
        rng = np.random.default_rng(11)
        df = pd.DataFrame(rng.normal(0, 1, (n, 5)), columns=[f"feat_{i}" for i in range(5)])
        df[feature_config.AGENT_KEY] = [f"msisdn_{i}" for i in range(n)]
        df[feature_config.THIN_FILE_COL] = 0
        result = score_new_agents(df, artifacts)
        assert result.shape[0] == n

    def test_result_has_score_columns(self, tmp_path):
        artifacts = _make_artifacts(tmp_path)
        n = 30
        rng = np.random.default_rng(12)
        df = pd.DataFrame(rng.normal(0, 1, (n, 5)), columns=[f"feat_{i}" for i in range(5)])
        df[feature_config.AGENT_KEY] = [f"msisdn_{i}" for i in range(n)]
        df[feature_config.THIN_FILE_COL] = 0
        result = score_new_agents(df, artifacts)
        assert "xgb_raw_score" in result.columns
        assert "lgb_raw_score" in result.columns
        # Both models should have independent calibrated PD columns
        assert "xgb_cal_pd" in result.columns
        assert "lgb_cal_pd" in result.columns
        # Champion's cal_pd is promoted to the shared cal_pd column
        assert feature_config.CAL_PD_COL in result.columns

    def test_thin_agents_get_pd_from_scorecard(self, tmp_path):
        artifacts = _make_artifacts(tmp_path)
        n = 20
        rng = np.random.default_rng(13)
        df = pd.DataFrame(rng.normal(0, 1, (n, 5)), columns=[f"feat_{i}" for i in range(5)])
        df[feature_config.AGENT_KEY] = [f"msisdn_{i}" for i in range(n)]
        df[feature_config.THIN_FILE_COL] = 1
        df["never_loan_pd_like"] = rng.uniform(0, 1, n)
        result = score_new_agents(df, artifacts)
        thin_mask = result[feature_config.THIN_FILE_COL] == 1
        # Both model cal_pd columns should be populated from never_loan_pd_like
        assert result.loc[thin_mask, "xgb_cal_pd"].notna().sum() > 0
        assert result.loc[thin_mask, "lgb_cal_pd"].notna().sum() > 0
        # Shared cal_pd column is the champion's (xgb by default)
        assert result.loc[thin_mask, feature_config.CAL_PD_COL].notna().sum() > 0

    def test_missing_features_warned_but_not_raised(self, tmp_path):
        artifacts = _make_artifacts(tmp_path)
        n = 20
        rng = np.random.default_rng(14)
        # Only provide 3 out of 5 features
        df = pd.DataFrame(rng.normal(0, 1, (n, 3)), columns=[f"feat_{i}" for i in range(3)])
        df[feature_config.AGENT_KEY] = [f"msisdn_{i}" for i in range(n)]
        df[feature_config.THIN_FILE_COL] = 0
        # Should not raise -- missing features filled with NaN
        result = score_new_agents(df, artifacts)
        assert result.shape[0] == n

    def test_invalid_champion_raises(self, tmp_path):
        artifacts = _make_artifacts(tmp_path)
        df = pd.DataFrame({"feat_0": [1.0]})
        with pytest.raises(ValueError, match="champion"):
            score_new_agents(df, artifacts, champion="unknown")


# =========================================================================== #
# run_inference_pipeline -- loan_history_df wiring
# =========================================================================== #


def _make_e2e_artifacts(tmp_dir: Path) -> None:
    """
    Write real artifacts to *tmp_dir* whose feature_order/transform_report
    match columns actually produced by Phase 2.1 (vol_1m) and the loan-history
    inference path (observed_prior_loan_count, via
    run_phase_2_2_loan_history_pd_features_inference's rename map), so
    run_inference_pipeline can be exercised fully end-to-end.
    """
    from pd_model.config.model_config import ModelConfig
    from pd_model.modeling.calibration import attach_cal_pd, build_pd_calibration_map, build_policy_tables
    from pd_model.modeling.lgbm_model import train_lgbm
    from pd_model.modeling.xgb_model import train_xgb
    from pd_model.preprocessing.transformations import build_transformed_dataframe, get_and_classify_pd_features

    rng = np.random.default_rng(7)
    n = 600
    df_train = pd.DataFrame(
        {
            "agent_msisdn": [f"256{i:08d}" for i in range(n)],
            "snapshot_dt": pd.to_datetime("2025-09-30"),
            "bad_state": rng.integers(0, 2, n),
            "vol_1m": rng.exponential(200, n),
            "observed_prior_loan_count": rng.integers(0, 10, n),
        }
    )
    pd_features, log_cols, cap_cols, _prot, signed_log_cols, _ = get_and_classify_pd_features(df_train)
    df_transformed, pd_features_pruned, transform_report = build_transformed_dataframe(
        df_train, pd_features, log_cols, cap_cols, signed_log_cols
    )

    X = df_transformed[pd_features_pruned]
    y = df_train["bad_state"]
    split = int(n * 0.7)
    Xtr, ytr = X.iloc[:split], y.iloc[:split]
    Xva, yva = X.iloc[split:], y.iloc[split:]

    cfg = ModelConfig(cal_min_n=100, cal_min_bads=5, cal_min_coverage=0.90)
    xgb_model, _, xgb_val = train_xgb(Xtr, ytr, Xva, yva, cfg=cfg)
    lgb_model, _, lgb_val = train_lgbm(Xtr, ytr, Xva, yva, cfg=cfg)

    xgb_cal_map = build_pd_calibration_map(xgb_val, "xgb", cfg=cfg)
    lgb_cal_map = build_pd_calibration_map(lgb_val, "lgb", cfg=cfg)
    cal_map = pd.concat([xgb_cal_map, lgb_cal_map], ignore_index=True)

    xgb_cal = attach_cal_pd(xgb_val, cal_map, "xgb", cfg=cfg)
    lgb_cal = attach_cal_pd(lgb_val, cal_map, "lgb", cfg=cfg)
    xgb_thresh, _, _ = build_policy_tables(xgb_cal, cfg=cfg)
    lgb_thresh, _, _ = build_policy_tables(lgb_cal, cfg=cfg)

    joblib.dump(xgb_model, tmp_dir / "xgb_model.joblib")
    joblib.dump(lgb_model, tmp_dir / "lgbm_model.joblib")
    (tmp_dir / "feature_order.json").write_text(json.dumps({"selected_features": pd_features_pruned}))
    cal_map.to_csv(tmp_dir / "pd_calibration_map.csv", index=False)
    xgb_thresh.to_csv(tmp_dir / "xgb_policy_thresholds.csv", index=False)
    lgb_thresh.to_csv(tmp_dir / "lgb_policy_thresholds.csv", index=False)
    transform_report.to_csv(tmp_dir / "transform_report.csv", index=False)


class TestRunInferencePipelineLoanHistory:
    def _raw_agents(self, n: int = 15) -> pd.DataFrame:
        rng = np.random.default_rng(21)
        return pd.DataFrame(
            {
                "agent_msisdn": [f"256{i:08d}" for i in range(n)],
                "vol_1m": rng.exponential(200, n),
                "vol_3m": rng.exponential(600, n),
                "vol_6m": rng.exponential(1200, n),
            }
        )

    def _loan_history_df(self, n: int = 15) -> pd.DataFrame:
        rng = np.random.default_rng(22)
        return pd.DataFrame(
            {
                "msisdn": [f"256{i:08d}" for i in range(n)],
                "observed_loan_count": rng.integers(0, 10, n),
                "prior_loan_count_180d": rng.integers(0, 10, n),
                "prior_active_loan_months_180d": rng.integers(0, 6, n),
                "has_unresolved_loan_at_snapshot": rng.integers(0, 2, n),
            }
        )

    def test_loan_history_df_path_scores_all_agents(self, tmp_path):
        from pd_model.modeling.inference import run_inference_pipeline

        _make_e2e_artifacts(tmp_path)
        df_raw = self._raw_agents()
        result = run_inference_pipeline(
            df_raw=df_raw,
            artifacts_dir=tmp_path,
            loan_history_df=self._loan_history_df(),
        )
        assert result.shape[0] == len(df_raw)
        assert "cal_pd" in result.columns

    def test_no_repayment_or_loan_history_falls_back_to_thin_file(self, tmp_path):
        """Regression guard for the pre-existing classify_agent_loan_status bug:
        calling with neither repayment_df nor loan_history_df must not crash --
        it must fall back to a genuine thin-file path."""
        from pd_model.modeling.inference import run_inference_pipeline

        _make_e2e_artifacts(tmp_path)
        df_raw = self._raw_agents()
        result = run_inference_pipeline(df_raw=df_raw, artifacts_dir=tmp_path)
        assert result.shape[0] == len(df_raw)

    def test_loan_history_df_takes_precedence_over_repayment_df(self, tmp_path):
        from pd_model.modeling.inference import run_inference_pipeline

        _make_e2e_artifacts(tmp_path)
        df_raw = self._raw_agents()
        n = len(df_raw)
        bogus_repayment_df = pd.DataFrame(
            {"agent_msisdn": df_raw["agent_msisdn"], "snapshot_dt": pd.to_datetime("2025-09-30")}
        )
        # bogus_repayment_df is missing columns run_phase_2_2_repayment_pd_features
        # requires -- if the loan_history_df branch didn't take precedence, this
        # would raise instead of scoring successfully.
        result = run_inference_pipeline(
            df_raw=df_raw,
            artifacts_dir=tmp_path,
            repayment_df=bogus_repayment_df,
            loan_history_df=self._loan_history_df(),
        )
        assert result.shape[0] == n

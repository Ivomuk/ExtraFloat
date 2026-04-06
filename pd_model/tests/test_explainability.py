"""Tests for pd_model.modeling.explainability."""
import numpy as np
import pandas as pd
import pytest

from pd_model.modeling.explainability import (
    build_adverse_action_df,
    compute_shap_values,
    shap_feature_importance,
)


def _make_xgb_model(n_features: int = 5):
    from pd_model.modeling.xgb_model import train_xgb
    from pd_model.config.model_config import ModelConfig

    rng = np.random.default_rng(42)
    n = 400
    feat_names = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.normal(0, 1, (n, n_features)), columns=feat_names)
    y = pd.Series((X["f0"] + rng.normal(0, 0.5, n) > 0).astype(int))
    cfg = ModelConfig(xgb_n_estimators=50)
    model, _, _ = train_xgb(X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:], cfg=cfg)
    return model, feat_names, X.iloc[300:].reset_index(drop=True)


class TestComputeShapValues:
    def test_shape_matches_input(self):
        model, feat_names, X_val = _make_xgb_model(n_features=5)
        shap_vals = compute_shap_values(model, X_val, model_key="xgb")
        assert shap_vals.shape == (len(X_val), len(feat_names))

    def test_returns_float32(self):
        model, feat_names, X_val = _make_xgb_model()
        shap_vals = compute_shap_values(model, X_val, model_key="xgb")
        assert shap_vals.dtype == np.float32

    def test_values_are_finite(self):
        model, feat_names, X_val = _make_xgb_model()
        shap_vals = compute_shap_values(model, X_val, model_key="xgb")
        assert np.all(np.isfinite(shap_vals))


class TestBuildAdverseActionDf:
    def _shap(self):
        rng = np.random.default_rng(7)
        return rng.normal(0, 0.1, (50, 5))

    def test_output_columns(self):
        shap_vals = self._shap()
        feat_names = [f"f{i}" for i in range(5)]
        df = build_adverse_action_df(shap_vals, feat_names, n_reasons=3)
        for i in range(1, 4):
            assert f"adverse_reason_{i}" in df.columns
            assert f"adverse_reason_{i}_shap" in df.columns

    def test_row_count_matches(self):
        shap_vals = self._shap()
        feat_names = [f"f{i}" for i in range(5)]
        df = build_adverse_action_df(shap_vals, feat_names, n_reasons=3)
        assert len(df) == 50

    def test_negative_shap_not_adverse(self):
        # All SHAP values negative → all adverse reason slots should be None
        shap_vals = -np.abs(np.random.default_rng(1).normal(0.1, 0.05, (10, 4)))
        feat_names = [f"f{i}" for i in range(4)]
        df = build_adverse_action_df(shap_vals, feat_names, n_reasons=2, min_shap=0.0)
        assert df["adverse_reason_1"].isna().all() or (df["adverse_reason_1"] == None).all()

    def test_n_reasons_respected(self):
        shap_vals = self._shap()
        feat_names = [f"f{i}" for i in range(5)]
        df2 = build_adverse_action_df(shap_vals, feat_names, n_reasons=2)
        assert "adverse_reason_3" not in df2.columns
        assert "adverse_reason_2" in df2.columns

    def test_shap_values_rounded(self):
        shap_vals = np.array([[0.123456789, -0.1]])
        feat_names = ["a", "b"]
        df = build_adverse_action_df(shap_vals, feat_names, n_reasons=1)
        shap_val = df["adverse_reason_1_shap"].iloc[0]
        assert round(shap_val, 4) == shap_val


class TestShapFeatureImportance:
    def test_returns_all_features(self):
        shap_vals = np.random.default_rng(3).normal(0, 1, (30, 4))
        feat_names = [f"f{i}" for i in range(4)]
        df = shap_feature_importance(shap_vals, feat_names)
        assert len(df) == 4
        assert "feature" in df.columns
        assert "mean_abs_shap" in df.columns

    def test_sorted_descending(self):
        shap_vals = np.random.default_rng(3).normal(0, 1, (30, 4))
        feat_names = [f"f{i}" for i in range(4)]
        df = shap_feature_importance(shap_vals, feat_names)
        assert df["mean_abs_shap"].is_monotonic_decreasing

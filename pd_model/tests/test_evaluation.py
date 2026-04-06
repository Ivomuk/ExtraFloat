"""Tests for pd_model.modeling.evaluation."""
import numpy as np
import pandas as pd
import pytest

from pd_model.modeling.evaluation import (
    build_decile_tables,
    compare_models_deciles,
    safe_auc_with_reason,
)


def _scored_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = (x + rng.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame({"raw_score": x, "bad_state": y,
                         "thin_file_flag": rng.integers(0, 2, n)})


class TestSafeAuc:
    def test_ok(self):
        df = _scored_df()
        auc, reason = safe_auc_with_reason(df["bad_state"], df["raw_score"])
        assert reason == "ok"
        assert auc > 0.5

    def test_single_class(self):
        y = pd.Series([0] * 100)
        p = pd.Series(np.random.rand(100))
        auc, reason = safe_auc_with_reason(y, p)
        assert np.isnan(auc)
        assert reason == "single_class"

    def test_no_rows(self):
        auc, reason = safe_auc_with_reason(pd.Series([], dtype=float), pd.Series([], dtype=float))
        assert np.isnan(auc)
        assert reason == "no_rows"


class TestBuildDecileTables:
    def test_returns_dict_with_expected_keys(self):
        df = _scored_df(500)
        result = build_decile_tables(df)
        assert isinstance(result, dict)
        for key in ("seg_col", "seg_summary", "overall", "by_segment"):
            assert key in result

    def test_overall_has_decile_column(self):
        result = build_decile_tables(_scored_df(500))
        assert "decile" in result["overall"].columns

    def test_seg_summary_has_auc(self):
        result = build_decile_tables(_scored_df(500))
        assert "auc" in result["seg_summary"].columns

    def test_thin_segment_built_automatically(self):
        result = build_decile_tables(_scored_df(500))
        assert result["seg_col"] == "thin_segment"
        assert "non_thin" in result["seg_summary"]["segment"].values

    def test_explicit_segment_col(self):
        df = _scored_df(400)
        df["my_seg"] = (df["thin_file_flag"] == 1).map({True: "A", False: "B"})
        result = build_decile_tables(df, segment_col="my_seg")
        assert result["seg_col"] == "my_seg"

    def test_lift_vs_overall_present(self):
        result = build_decile_tables(_scored_df(500))
        assert "lift_vs_overall" in result["overall"].columns

    def test_small_df_returns_note(self):
        df = pd.DataFrame({"raw_score": [0.1, 0.2], "bad_state": [0, 1]})
        result = build_decile_tables(df, n_bins=10)
        assert "note" in result["overall"].columns


class TestCompareModelsDeciles:
    def test_returns_summary_with_two_rows(self):
        df = _scored_df(500)
        summary, res_a, res_b = compare_models_deciles(df, df, name_a="xgb", name_b="lgb")
        assert isinstance(summary, pd.DataFrame)
        assert summary.shape[0] == 2
        assert set(summary["model"].values) == {"xgb", "lgb"}

    def test_summary_has_auc_column(self):
        df = _scored_df(500)
        summary, _, _ = compare_models_deciles(df, df)
        assert "auc" in summary.columns

    def test_res_dicts_have_overall_key(self):
        df = _scored_df(500)
        _, res_a, res_b = compare_models_deciles(df, df)
        assert "overall" in res_a
        assert "overall" in res_b

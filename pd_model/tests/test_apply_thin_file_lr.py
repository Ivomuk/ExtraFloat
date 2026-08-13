"""Tests for pd_model.modeling.scorecard.apply_thin_file_lr."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from pd_model.modeling.scorecard import apply_thin_file_lr


def _df(n: int = 50, never_loan_pd_like_dtype: str = "float64") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "agent_msisdn": [f"a{i}" for i in range(n)],
            "thin_file_flag": [1] * 30 + [0] * (n - 30),
            "feat1": rng.normal(0, 1, n).astype("float32"),
            "never_loan_pd_like": np.full(n, 0.5, dtype=never_loan_pd_like_dtype),
        }
    )


def _mock_pipeline(n_thin: int = 30) -> MagicMock:
    rng = np.random.default_rng(1)
    pipeline = MagicMock()
    p1 = rng.uniform(0, 1, n_thin)
    pipeline.predict_proba.return_value = np.column_stack([1 - p1, p1])
    return pipeline


class TestApplyThinFileLr:
    def test_no_crash_when_target_column_already_float32(self):
        """Regression guard: never_loan_pd_like may already be float32 (the
        training pipeline downcasts float64 -> float32 ahead of this call
        for memory reasons on large loan-level exports), while sklearn's
        predict_proba always returns float64. Pandas 3.0 raises
        LossySetitemError on a bare .loc float64 -> float32 assignment
        instead of silently narrowing -- confirmed via a real end-to-end
        run_pipeline crash at this exact line before the fix."""
        df = _df(never_loan_pd_like_dtype="float32")
        out = apply_thin_file_lr(df, _mock_pipeline(), ["feat1"])
        assert out["never_loan_pd_like"].dtype == np.float32

    def test_no_crash_when_target_column_float64(self):
        df = _df(never_loan_pd_like_dtype="float64")
        out = apply_thin_file_lr(df, _mock_pipeline(), ["feat1"])
        assert out["never_loan_pd_like"].dtype == np.float64

    def test_updates_only_thin_file_rows(self):
        df = _df(never_loan_pd_like_dtype="float32")
        out = apply_thin_file_lr(df, _mock_pipeline(), ["feat1"])
        thick_mask = out["thin_file_flag"] == 0
        assert (out.loc[thick_mask, "never_loan_pd_like"] == 0.5).all()

    def test_enforces_thin_file_pd_prior_floor(self):
        from pd_model.config.model_config import DEFAULT_CONFIG

        df = _df(never_loan_pd_like_dtype="float32")
        out = apply_thin_file_lr(df, _mock_pipeline(), ["feat1"])
        thin_mask = out["thin_file_flag"] == 1
        assert (out.loc[thin_mask, "never_loan_pd_like"] >= DEFAULT_CONFIG.thin_file_pd_prior - 1e-6).all()

    def test_returns_unchanged_copy_when_no_thin_file_rows(self):
        df = _df(never_loan_pd_like_dtype="float32")
        df["thin_file_flag"] = 0
        out = apply_thin_file_lr(df, _mock_pipeline(), ["feat1"])
        assert (out["never_loan_pd_like"] == 0.5).all()

    def test_missing_feature_column_filled_with_nan(self):
        df = _df(never_loan_pd_like_dtype="float32")
        out = apply_thin_file_lr(df, _mock_pipeline(), ["feat1", "missing_feature"])
        assert out["never_loan_pd_like"].notna().all()

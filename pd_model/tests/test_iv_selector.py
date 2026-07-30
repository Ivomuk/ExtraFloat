"""Tests for pd_model.scoring.iv_selector."""

import inspect

import numpy as np
import pandas as pd
import pytest

import pd_model.scoring.iv_selector as iv_mod
from pd_model.scoring.iv_selector import (
    compute_iv,
    compute_iv_fixed_bins,
    iv_filter_phase_2,
)


def _binary_feature(n: int = 500) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(99)
    # Use a continuous feature with strong separation so IV is reliably > 0.01
    x = pd.Series(rng.normal(0, 1, n))
    y = pd.Series((x + rng.normal(0, 0.3, n) > 0.5).astype(int))
    return x, y


def _continuous_feature(n: int = 500) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(42)
    x = pd.Series(rng.normal(0, 1, n))
    y = pd.Series((x > 0.5).astype(int).values + rng.binomial(1, 0.05, n))
    y = y.clip(0, 1)
    return x, y


class TestComputeIv:
    def test_returns_float(self):
        x, y = _continuous_feature()
        iv = compute_iv(x, y)
        assert isinstance(iv, float)

    def test_positive_iv_for_informative_feature(self):
        x, y = _binary_feature()
        iv = compute_iv(x, y)
        assert iv > 0.01, f"Expected IV > 0.01, got {iv:.4f}"

    def test_zero_iv_for_constant(self):
        x = pd.Series([1.0] * 100)
        y = pd.Series(np.random.default_rng(0).integers(0, 2, 100))
        assert compute_iv(x, y) == 0.0

    def test_return_bins_flag(self):
        x, y = _continuous_feature()
        iv, bins = compute_iv(x, y, return_bins=True)
        assert isinstance(iv, float)
        assert bins is not None
        assert len(bins) >= 2

    def test_handles_nans_gracefully(self):
        x, y = _continuous_feature()
        x_with_nan = x.copy()
        x_with_nan.iloc[:50] = np.nan
        iv = compute_iv(x_with_nan, y)
        assert isinstance(iv, float)


class TestComputeIvFixedBins:
    def test_no_duplicate_definition(self):
        """Verify there is exactly one definition of compute_iv_fixed_bins."""
        source = inspect.getsource(iv_mod)
        count = source.count("def compute_iv_fixed_bins(")
        assert count == 1, f"Expected 1 definition, found {count}"

    def test_returns_float(self):
        x, y = _continuous_feature()
        iv = compute_iv_fixed_bins(x, y)
        assert isinstance(iv, float)

    def test_quantile_vs_uniform_binning(self):
        x, y = _continuous_feature()
        iv_q = compute_iv_fixed_bins(x, y, binning="quantile")
        iv_u = compute_iv_fixed_bins(x, y, binning="uniform")
        # Both should be positive for an informative feature
        assert iv_q > 0
        assert iv_u > 0


class TestIvFilterPhase2:
    def _make_data(self, n: int = 400):
        rng = np.random.default_rng(55)
        y = pd.Series(rng.integers(0, 2, n).astype(float))
        # Informative feature
        x_good = pd.Series(rng.normal(0, 1, n) + y * 1.5)
        # Noise feature
        x_noise = pd.Series(rng.normal(0, 1, n))
        X_raw = pd.DataFrame({"x_good": x_good, "x_noise": x_noise})
        X_trans = X_raw.copy()
        return X_raw, X_trans, y

    def test_selects_informative_feature(self):
        X_raw, X_trans, y = self._make_data()
        selected, iv_table = iv_filter_phase_2(X_raw, X_trans, y)
        assert "x_good" in selected

    def test_returns_iv_table(self):
        X_raw, X_trans, y = self._make_data()
        _, iv_table = iv_filter_phase_2(X_raw, X_trans, y)
        assert "feature" in iv_table.columns
        assert "iv_after" in iv_table.columns
        assert "iv_before" in iv_table.columns

    def test_target_col_excluded(self):
        X_raw, X_trans, y = self._make_data()
        X_trans["bad_state"] = y
        selected, _ = iv_filter_phase_2(X_raw, X_trans, y, target_col="bad_state")
        assert "bad_state" not in selected

    def test_blacklisted_col_excluded(self):
        X_raw, X_trans, y = self._make_data()
        selected, _ = iv_filter_phase_2(
            X_raw, X_trans, y, pd_feature_blacklist=frozenset({"x_good"})
        )
        assert "x_good" not in selected

"""Tests for pd_model.monitoring.drift."""
import numpy as np
import pandas as pd
import pytest

from pd_model.monitoring.drift import (
    compute_csi,
    compute_psi,
    run_drift_report,
)


def _make_df(seed: int, n: int = 500, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "score": rng.normal(0.5 + shift, 0.15, n).clip(0, 1),
        "feat_a": rng.normal(shift, 1, n),
        "feat_b": rng.exponential(1 + shift, n),
        "feat_c": rng.normal(0, 1, n),
    })


class TestComputePsi:
    def test_identical_distributions_near_zero(self):
        s = pd.Series(np.random.default_rng(0).normal(0, 1, 1000))
        psi = compute_psi(s, s)
        assert psi < 0.01

    def test_large_shift_high_psi(self):
        rng = np.random.default_rng(1)
        ref = pd.Series(rng.normal(0, 1, 1000))
        mon = pd.Series(rng.normal(5, 1, 1000))   # big shift
        psi = compute_psi(ref, mon)
        assert psi > 0.25

    def test_returns_float(self):
        s = pd.Series(np.random.default_rng(2).uniform(0, 1, 200))
        assert isinstance(compute_psi(s, s.sample(100, random_state=0)), float)

    def test_nan_input_returns_nan(self):
        ref = pd.Series([np.nan, np.nan])
        mon = pd.Series([np.nan, np.nan])
        assert np.isnan(compute_psi(ref, mon))


class TestComputeCsi:
    def test_returns_one_row_per_feature(self):
        ref = _make_df(0)
        mon = _make_df(1)
        csi = compute_csi(ref, mon, feature_cols=["feat_a", "feat_b", "feat_c"])
        assert len(csi) == 3

    def test_has_expected_columns(self):
        ref = _make_df(0)
        mon = _make_df(1)
        csi = compute_csi(ref, mon, feature_cols=["feat_a"])
        assert {"feature", "csi", "stability"} <= set(csi.columns)

    def test_sorted_descending(self):
        ref = _make_df(0)
        mon = _make_df(1, shift=2.0)   # large shift on all features
        csi = compute_csi(ref, mon, feature_cols=["feat_a", "feat_b", "feat_c"])
        vals = csi["csi"].dropna().tolist()
        assert vals == sorted(vals, reverse=True)

    def test_missing_feature_returns_nan(self):
        ref = _make_df(0)
        mon = _make_df(1)
        csi = compute_csi(ref, mon, feature_cols=["feat_a", "nonexistent"])
        row = csi[csi["feature"] == "nonexistent"].iloc[0]
        assert np.isnan(row["csi"])

    def test_stability_labels(self):
        ref = _make_df(0, n=1000)
        mon_stable = _make_df(0, n=1000)         # same distribution
        mon_shifted = _make_df(0, n=1000, shift=3.0)  # big shift

        csi_stable = compute_csi(ref, mon_stable, ["feat_a"])
        csi_shifted = compute_csi(ref, mon_shifted, ["feat_a"])

        assert csi_stable.iloc[0]["stability"] == "stable"
        assert csi_shifted.iloc[0]["stability"] == "significant_shift"


class TestRunDriftReport:
    def test_keys_present(self):
        ref = _make_df(0)
        mon = _make_df(1)
        report = run_drift_report(ref, mon, feature_cols=["feat_a", "feat_b"], score_col="score")
        assert "score_psi" in report
        assert "score_stability" in report
        assert "csi_table" in report
        assert "n_features_monitored" in report

    def test_score_psi_is_float(self):
        ref = _make_df(0)
        mon = _make_df(1)
        report = run_drift_report(ref, mon, feature_cols=["feat_a"], score_col="score")
        assert isinstance(report["score_psi"], float)

    def test_missing_score_col_returns_nan(self):
        ref = _make_df(0)
        mon = _make_df(1)
        report = run_drift_report(ref, mon, feature_cols=["feat_a"], score_col="no_such_col")
        assert np.isnan(report["score_psi"])
        assert report["score_stability"] == "unknown"

    def test_significant_shift_counted(self):
        ref = _make_df(0, n=1000)
        mon = _make_df(0, n=1000, shift=4.0)
        report = run_drift_report(ref, mon, feature_cols=["feat_a", "feat_b"], score_col="score")
        assert report["n_features_significant"] >= 1

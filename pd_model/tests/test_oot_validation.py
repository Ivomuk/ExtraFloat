"""
Test 6: Out-of-time (OOT) Population Stability Index (PSI).

PSI measures distributional shift between training (2025-09-30) and validation
(2025-11-15) feature distributions. High PSI means the model is being evaluated
on a population that looks very different from what it was trained on.

Industry thresholds:
    PSI < 0.10  -- stable, distribution is similar
    0.10 <= PSI < 0.25  -- moderate shift, investigate
    PSI >= 0.25  -- severe shift, model may not generalise

A very high AUC on a severely shifted population is suspicious -- it may mean the
validation set coincidentally separates better, not that the model generalises well.

To run on real data:
    from pd_model.tests.test_oot_validation import compute_psi, psi_report
    train_df = pd.read_csv("path/to/X_train.csv")
    val_df   = pd.read_csv("path/to/X_val.csv")
    features = ["net_exposure_6M", "disbursement_vol_m5", ...]
    report   = psi_report(train_df, val_df, features=features)
    severe   = report[report["flag"] == "SEVERE"]
    print(severe)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

PSI_STABLE = 0.10
PSI_SEVERE = 0.25


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute PSI between training (expected) and validation (actual) distributions.

    Uses quantile bins derived from the expected (training) distribution.
    Returns PSI as a float; higher = more shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(expected, quantiles))

    if len(bin_edges) < 2:
        return 0.0  # constant feature: no shift possible

    exp_counts = np.histogram(expected, bins=bin_edges)[0].astype(float)
    act_counts = np.histogram(actual, bins=bin_edges)[0].astype(float)

    exp_pct = exp_counts / (exp_counts.sum() + eps)
    act_pct = act_counts / (act_counts.sum() + eps)

    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_report(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    features: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return a PSI report for each feature, sorted by PSI descending."""
    records = []
    for feat in features:
        if feat not in df_train.columns or feat not in df_val.columns:
            records.append({"feature": feat, "psi": np.nan, "flag": "missing"})
            continue
        psi = compute_psi(df_train[feat].values, df_val[feat].values, n_bins=n_bins)
        flag = "SEVERE" if psi >= PSI_SEVERE else ("MODERATE" if psi >= PSI_STABLE else "stable")
        records.append({"feature": feat, "psi": round(psi, 4), "flag": flag})
    return pd.DataFrame(records).sort_values("psi", ascending=False, ignore_index=True)


class TestComputePsi:
    def test_identical_distributions_near_zero_psi(self):
        """Identical arrays must produce PSI near 0."""
        rng = np.random.default_rng(20)
        arr = rng.normal(0, 1, 1000)
        psi = compute_psi(arr, arr.copy())
        assert psi < PSI_STABLE, (
            f"Identical distribution PSI should be < {PSI_STABLE}, got {psi:.4f}"
        )

    def test_same_distribution_low_psi(self):
        """Two samples from the same distribution must produce low PSI."""
        rng = np.random.default_rng(21)
        train = rng.normal(0, 1, 2000)
        val = rng.normal(0, 1, 500)
        psi = compute_psi(train, val)
        assert psi < PSI_STABLE, (
            f"Same-distribution samples should have PSI < {PSI_STABLE}, got {psi:.4f}"
        )

    def test_large_shift_produces_high_psi(self):
        """A 5-sigma distribution shift must produce PSI above the severe threshold."""
        rng = np.random.default_rng(22)
        train = rng.normal(0, 1, 2000)
        val = rng.normal(5, 1, 500)
        psi = compute_psi(train, val)
        assert psi >= PSI_SEVERE, (
            f"5-sigma shift should have PSI >= {PSI_SEVERE}, got {psi:.4f}"
        )

    def test_constant_feature_zero_psi(self):
        """A constant feature produces PSI of 0 (no distributional shift)."""
        psi = compute_psi(np.ones(100), np.ones(50))
        assert psi == 0.0 or np.isnan(psi), (
            f"Constant feature PSI should be 0 or NaN, got {psi}"
        )

    def test_psi_nonnegative(self):
        """PSI must always be >= 0."""
        rng = np.random.default_rng(23)
        train = rng.exponential(1, 500)
        val = rng.normal(1, 0.5, 200)
        psi = compute_psi(train, val)
        assert psi >= 0, f"PSI must be non-negative, got {psi}"

    def test_empty_array_returns_nan(self):
        """Empty arrays must return NaN."""
        psi = compute_psi(np.array([]), np.array([1.0, 2.0]))
        assert np.isnan(psi), f"Expected NaN for empty expected array, got {psi}"


class TestPsiReport:
    def test_returns_dataframe_with_expected_columns(self):
        rng = np.random.default_rng(24)
        n = 300
        df_train = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.uniform(0, 1, n)})
        df_val = pd.DataFrame({"f1": rng.normal(0, 1, 100), "f2": rng.uniform(0, 1, 100)})
        report = psi_report(df_train, df_val, features=["f1", "f2"])
        assert isinstance(report, pd.DataFrame)
        for col in ("feature", "psi", "flag"):
            assert col in report.columns, f"Missing column: {col}"

    def test_stable_feature_flagged_correctly(self):
        rng = np.random.default_rng(25)
        df_train = pd.DataFrame({"stable_feat": rng.normal(0, 1, 1000)})
        df_val = pd.DataFrame({"stable_feat": rng.normal(0, 1, 200)})
        report = psi_report(df_train, df_val, features=["stable_feat"])
        flag = report.loc[report["feature"] == "stable_feat", "flag"].iloc[0]
        assert flag == "stable", f"Expected 'stable', got '{flag}'"

    def test_missing_feature_flagged(self):
        """A feature absent from val must be flagged 'missing', not raise."""
        df_train = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
        df_val = pd.DataFrame({"other": [1.0, 2.0]})
        report = psi_report(df_train, df_val, features=["feat"])
        assert report.iloc[0]["flag"] == "missing"

    def test_sorted_descending_by_psi(self):
        """Report must be sorted by PSI descending."""
        rng = np.random.default_rng(26)
        df_train = pd.DataFrame({
            "stable": rng.normal(0, 1, 1000),
            "shifted": rng.normal(0, 1, 1000),
        })
        df_val = pd.DataFrame({
            "stable": rng.normal(0, 1, 200),
            "shifted": rng.normal(10, 1, 200),  # large shift
        })
        report = psi_report(df_train, df_val, features=["stable", "shifted"])
        psi_vals = report["psi"].tolist()
        assert psi_vals[0] >= psi_vals[-1], "Report must be sorted by PSI descending"

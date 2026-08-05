"""
Test 4: Feature-target Spearman / Pearson correlation analysis.

A Spearman correlation |r| > 0.80 between a single feature and the binary target
signals the feature may be a near-proxy for the label -- either temporal leakage
or business-process co-definition.

Thresholds:
    |r| > 0.80 -> ALERT (investigate or exclude)
    |r| > 0.70 -> WARN
    |r| <= 0.70 -> ok

To analyse the real training matrix:
    from pd_model.tests.test_feature_target_correlation import feature_label_correlations
    import pandas as pd
    X_train = pd.read_csv("path/to/X_train.csv")
    y_train = pd.read_csv("path/to/y_train.csv")["bad_state"]
    report = feature_label_correlations(X_train, y_train)
    print(report[report["flag"] != "ok"])
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

CORR_WARN_THRESHOLD = 0.70
CORR_ALERT_THRESHOLD = 0.80


def feature_label_correlations(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compute Spearman and Pearson correlation between each feature and binary target.

    Returns DataFrame with columns [feature, spearman_r, pearson_r, flag]
    sorted by |spearman_r| descending.
    """
    records = []
    y_arr = pd.to_numeric(y, errors="coerce")

    for col in X.columns:
        x_arr = pd.to_numeric(X[col], errors="coerce")
        valid = x_arr.notna() & y_arr.notna()
        if valid.sum() < 50:
            records.append({"feature": col, "spearman_r": np.nan, "pearson_r": np.nan, "flag": "skip"})
            continue
        sp_r, _ = spearmanr(x_arr[valid], y_arr[valid])
        pe_r = float(np.corrcoef(x_arr[valid], y_arr[valid])[0, 1])
        abs_r = max(abs(sp_r), abs(pe_r))
        flag = "ALERT" if abs_r > CORR_ALERT_THRESHOLD else ("WARN" if abs_r > CORR_WARN_THRESHOLD else "ok")
        records.append({
            "feature": col,
            "spearman_r": round(float(sp_r), 4),
            "pearson_r": round(pe_r, 4),
            "flag": flag,
        })

    return pd.DataFrame(records).sort_values(
        "spearman_r", key=lambda s: s.abs(), ascending=False, ignore_index=True
    )


class TestFeatureTargetCorrelation:
    def test_uncorrelated_feature_below_warn_threshold(self):
        """Random uncorrelated feature must have |r| well below the warn threshold."""
        rng = np.random.default_rng(7)
        n = 2000
        y = pd.Series(rng.integers(0, 2, n).astype(float))
        X = pd.DataFrame({"noise": rng.standard_normal(n)})
        result = feature_label_correlations(X, y)
        r = abs(result.loc[result["feature"] == "noise", "spearman_r"].iloc[0])
        assert r < CORR_WARN_THRESHOLD, (
            f"Random feature |r|={r:.4f} exceeds warn threshold {CORR_WARN_THRESHOLD}"
        )

    def test_label_proxy_flagged_as_alert(self):
        """A near-perfect label proxy must be flagged ALERT."""
        rng = np.random.default_rng(8)
        n = 1000
        y = pd.Series(rng.integers(0, 2, n).astype(float))
        proxy = y + rng.normal(0, 0.01, n)
        X = pd.DataFrame({"leaked_proxy": proxy})
        result = feature_label_correlations(X, y)
        row = result.loc[result["feature"] == "leaked_proxy"].iloc[0]
        assert abs(row["spearman_r"]) > CORR_ALERT_THRESHOLD, (
            f"Expected |r| > {CORR_ALERT_THRESHOLD}, got {row['spearman_r']:.4f}"
        )
        assert row["flag"] == "ALERT"

    def test_strongly_correlated_feature_above_alert(self):
        """A feature with correlation ~0.90 must be flagged ALERT."""
        rng = np.random.default_rng(9)
        n = 2000
        y = pd.Series(rng.integers(0, 2, n).astype(float))
        x = 0.9 * y.values + 0.1 * rng.standard_normal(n)
        X = pd.DataFrame({"strong": x})
        result = feature_label_correlations(X, y)
        row = result.loc[result["feature"] == "strong"].iloc[0]
        assert abs(row["spearman_r"]) > CORR_WARN_THRESHOLD, (
            f"Expected |r| > {CORR_WARN_THRESHOLD}, got {row['spearman_r']:.4f}"
        )

    def test_returns_expected_columns(self):
        rng = np.random.default_rng(10)
        X = pd.DataFrame({"a": rng.standard_normal(200), "b": rng.standard_normal(200)})
        y = pd.Series(rng.integers(0, 2, 200).astype(float))
        result = feature_label_correlations(X, y)
        for col in ("feature", "spearman_r", "pearson_r", "flag"):
            assert col in result.columns, f"Missing column: {col}"

    def test_all_features_present_in_output(self):
        """Every feature in X must appear exactly once in the output."""
        rng = np.random.default_rng(11)
        feature_names = [f"f{i}" for i in range(6)]
        X = pd.DataFrame({f: rng.standard_normal(300) for f in feature_names})
        y = pd.Series(rng.integers(0, 2, 300).astype(float))
        result = feature_label_correlations(X, y)
        assert len(result) == len(feature_names)
        assert set(result["feature"]) == set(feature_names)

    def test_tiny_sample_skipped(self):
        """Features with fewer than 50 valid rows must be flagged skip."""
        rng = np.random.default_rng(12)
        X = pd.DataFrame({"small": rng.standard_normal(30)})
        y = pd.Series(rng.integers(0, 2, 30).astype(float))
        result = feature_label_correlations(X, y)
        assert result.iloc[0]["flag"] == "skip"

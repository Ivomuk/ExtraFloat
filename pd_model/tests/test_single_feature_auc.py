"""
Test 1: Single-feature AUC analysis.

Verifies that individual features cannot replicate the full model's 0.989 AUC
on their own. A single feature with AUC > 0.90 signals either temporal leakage
or business-process co-definition with the label.

Clean result: top features individually produce AUC in the 0.55-0.85 range.
Leakage result: any feature with AUC > 0.90 warrants immediate investigation.

To analyse the real training matrix:
    from pd_model.tests.test_single_feature_auc import single_feature_aucs
    import pandas as pd
    X_train = pd.read_csv("path/to/X_train.csv")
    y_train = pd.read_csv("path/to/y_train.csv")["bad_state"]
    report = single_feature_aucs(X_train, y_train)
    print(report[report["flag"] != "ok"])
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

SINGLE_FEAT_AUC_WARN = 0.85
SINGLE_FEAT_AUC_ALERT = 0.90


def single_feature_aucs(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compute individual-feature AUC against binary target y.

    Returns DataFrame sorted by AUC descending with columns
    [feature, auc, n_valid, flag].
    """
    records = []
    y_arr = pd.to_numeric(y, errors="coerce")

    for col in X.columns:
        x_arr = pd.to_numeric(X[col], errors="coerce")
        valid = x_arr.notna() & y_arr.notna()
        n_valid = int(valid.sum())
        if n_valid < 50 or y_arr[valid].nunique() < 2:
            records.append({"feature": col, "auc": np.nan, "n_valid": n_valid, "flag": "skip"})
            continue
        try:
            auc = roc_auc_score(y_arr[valid], x_arr[valid])
            auc = max(auc, 1 - auc)  # reflect below-0.5 (anti-correlated)
        except Exception:
            auc = np.nan
        flag = "ALERT" if auc > SINGLE_FEAT_AUC_ALERT else ("WARN" if auc > SINGLE_FEAT_AUC_WARN else "ok")
        records.append({"feature": col, "auc": auc, "n_valid": n_valid, "flag": flag})

    return pd.DataFrame(records).sort_values("auc", ascending=False, ignore_index=True)


class TestSingleFeatureAuc:
    def test_uncorrelated_feature_auc_near_half(self):
        """A random uncorrelated feature must produce AUC close to 0.50."""
        rng = np.random.default_rng(0)
        n = 2000
        y = pd.Series(rng.integers(0, 2, n).astype(float))
        X = pd.DataFrame({"noise": rng.standard_normal(n)})
        result = single_feature_aucs(X, y)
        auc = result.loc[result["feature"] == "noise", "auc"].iloc[0]
        assert 0.40 <= auc <= 0.60, f"Expected AUC near 0.50 for noise, got {auc:.4f}"

    def test_label_proxy_detected_above_alert_threshold(self):
        """A near-perfect label proxy must be flagged as ALERT (AUC > 0.90)."""
        rng = np.random.default_rng(1)
        n = 1000
        y = pd.Series(rng.integers(0, 2, n).astype(float))
        proxy = y + rng.normal(0, 0.01, n)
        X = pd.DataFrame({"leaked_proxy": proxy})
        result = single_feature_aucs(X, y)
        row = result.loc[result["feature"] == "leaked_proxy"].iloc[0]
        assert row["auc"] > SINGLE_FEAT_AUC_ALERT, (
            f"Expected leaked proxy AUC > {SINGLE_FEAT_AUC_ALERT}, got {row['auc']:.4f}"
        )
        assert row["flag"] == "ALERT"

    def test_moderate_signal_not_flagged_as_alert(self):
        """A moderately predictive feature must not reach the ALERT threshold."""
        rng = np.random.default_rng(2)
        n = 1000
        y = pd.Series(rng.integers(0, 2, n).astype(float))
        feature = y.values * 0.4 + rng.normal(0, 1, n)
        X = pd.DataFrame({"moderate_signal": feature})
        result = single_feature_aucs(X, y)
        auc = result.loc[result["feature"] == "moderate_signal", "auc"].iloc[0]
        assert auc < SINGLE_FEAT_AUC_ALERT, (
            f"Moderate signal AUC should be < {SINGLE_FEAT_AUC_ALERT}, got {auc:.4f}"
        )

    def test_returns_dataframe_with_expected_columns(self):
        rng = np.random.default_rng(3)
        X = pd.DataFrame({
            "net_exposure_6M": rng.uniform(0, 1e6, 200),
            "disbursement_vol_m5": rng.uniform(0, 5e5, 200),
            "noise": rng.standard_normal(200),
        })
        y = pd.Series(rng.integers(0, 2, 200).astype(float))
        result = single_feature_aucs(X, y)
        for col in ("feature", "auc", "n_valid", "flag"):
            assert col in result.columns, f"Missing column: {col}"
        assert len(result) == len(X.columns)

    def test_small_sample_skipped(self):
        """Columns with fewer than 50 valid rows must be marked skip."""
        rng = np.random.default_rng(4)
        X = pd.DataFrame({"tiny": rng.standard_normal(30)})
        y = pd.Series(rng.integers(0, 2, 30).astype(float))
        result = single_feature_aucs(X, y)
        assert result.iloc[0]["flag"] == "skip"

    def test_sorted_descending_by_auc(self):
        """Output must be sorted by AUC descending."""
        rng = np.random.default_rng(5)
        n = 300
        y = pd.Series(rng.integers(0, 2, n).astype(float))
        X = pd.DataFrame({
            "strong": y.values * 0.9 + rng.normal(0, 0.1, n),
            "weak": rng.standard_normal(n),
        })
        result = single_feature_aucs(X, y)
        aucs = result["auc"].dropna().tolist()
        assert aucs == sorted(aucs, reverse=True)

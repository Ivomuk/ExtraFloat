"""
Post-deployment drift monitoring: PSI and CSI.

PSI (Population Stability Index) measures shift in the model score distribution.
CSI (Characteristic Stability Index) applies the same formula per feature to
detect which inputs have drifted.

Interpretation thresholds (standard industry convention):
    PSI / CSI < 0.10  → stable
    0.10 ≤ PSI / CSI < 0.25 → moderate shift — investigate
    PSI / CSI ≥ 0.25  → significant shift — retrain / escalate

Usage
-----
::

    from pd_model.monitoring.drift import run_drift_report

    report = run_drift_report(
        reference_df=train_scored,       # snapshot used at training time
        monitoring_df=new_scored,        # latest production snapshot
        feature_cols=feature_order,
        score_col="xgb_raw_score",
    )
    print(report["score_psi"], report["score_stability"])
    print(report["csi_table"].head(10))
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pd_model.logging_config import get_logger

logger = get_logger(__name__)

# Standard PSI/CSI stability thresholds
_PSI_STABLE = 0.10
_PSI_MODERATE = 0.25


# ======================================================================== #
# Core PSI formula
# ======================================================================== #

def _psi_single(
    reference: np.ndarray,
    monitoring: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Compute PSI between two 1-D arrays using reference-defined bin edges.

    Bin edges are derived from the reference distribution's percentiles so
    the expected frequencies are approximately equal per bin.  A small
    epsilon (1e-6) prevents log(0) on empty bins.

    Returns
    -------
    float PSI value (0 = identical distributions).
    """
    ref = reference[~np.isnan(reference)]
    mon = monitoring[~np.isnan(monitoring)]

    if len(ref) < 2 or len(mon) < 2:
        return np.nan

    # Build bin edges from reference percentiles; unique to handle degenerate data
    pct_points = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref, pct_points))

    if len(edges) < 2:
        return np.nan

    eps = 1e-6
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    mon_counts = np.histogram(mon, bins=edges)[0].astype(float)

    ref_pct = (ref_counts / len(ref)).clip(eps)
    mon_pct = (mon_counts / len(mon)).clip(eps)

    return float(np.sum((mon_pct - ref_pct) * np.log(mon_pct / ref_pct)))


def _stability_label(psi: float) -> str:
    if np.isnan(psi):
        return "unknown"
    if psi < _PSI_STABLE:
        return "stable"
    if psi < _PSI_MODERATE:
        return "moderate_shift"
    return "significant_shift"


# ======================================================================== #
# Score PSI
# ======================================================================== #

def compute_psi(
    reference_scores: pd.Series,
    monitoring_scores: pd.Series,
    bins: int = 10,
) -> float:
    """
    PSI on model score (or calibrated PD) distributions.

    Parameters
    ----------
    reference_scores : scores from the training / reference period
    monitoring_scores: scores from the latest production snapshot
    bins             : number of equal-frequency bins (default 10)

    Returns
    -------
    PSI float.  Threshold: <0.10 stable, 0.10–0.25 moderate, ≥0.25 significant.
    """
    return _psi_single(
        reference_scores.to_numpy(dtype=float, na_value=np.nan),
        monitoring_scores.to_numpy(dtype=float, na_value=np.nan),
        bins=bins,
    )


# ======================================================================== #
# Feature CSI
# ======================================================================== #

def compute_csi(
    reference_df: pd.DataFrame,
    monitoring_df: pd.DataFrame,
    feature_cols: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    """
    CSI for each feature — same PSI formula applied column-by-column.

    Features present in ``feature_cols`` but missing from either DataFrame
    are skipped with CSI = NaN.

    Parameters
    ----------
    reference_df  : DataFrame from training / reference period
    monitoring_df : DataFrame from latest production snapshot
    feature_cols  : list of feature names to monitor
    bins          : number of equal-frequency bins (default 10)

    Returns
    -------
    DataFrame sorted by csi descending:
        feature, csi, stability
    """
    rows = []
    for feat in feature_cols:
        if feat not in reference_df.columns or feat not in monitoring_df.columns:
            rows.append({"feature": feat, "csi": np.nan, "stability": "unknown"})
            continue

        ref = pd.to_numeric(reference_df[feat], errors="coerce").to_numpy()
        mon = pd.to_numeric(monitoring_df[feat], errors="coerce").to_numpy()
        csi_val = _psi_single(ref, mon, bins=bins)
        rows.append({
            "feature": feat,
            "csi": round(csi_val, 5) if not np.isnan(csi_val) else np.nan,
            "stability": _stability_label(csi_val),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("csi", ascending=False, na_position="last").reset_index(drop=True)


# ======================================================================== #
# Full drift report
# ======================================================================== #

def run_drift_report(
    reference_df: pd.DataFrame,
    monitoring_df: pd.DataFrame,
    feature_cols: list[str],
    score_col: str = "xgb_raw_score",
    bins: int = 10,
) -> dict[str, Any]:
    """
    Full drift report: PSI on model scores + CSI on all features.

    Parameters
    ----------
    reference_df  : training snapshot DataFrame (with score and feature columns)
    monitoring_df : latest production snapshot DataFrame
    feature_cols  : feature names to include in CSI
    score_col     : column name for the model score (default "xgb_raw_score")
    bins          : bins for PSI/CSI calculation (default 10)

    Returns
    -------
    dict with keys:
        score_psi               – float
        score_stability         – "stable" | "moderate_shift" | "significant_shift"
        csi_table               – DataFrame (feature, csi, stability) sorted by csi
        n_features_stable       – int
        n_features_moderate     – int
        n_features_significant  – int
        n_features_monitored    – int
    """
    # Score PSI
    score_psi = np.nan
    if score_col in reference_df.columns and score_col in monitoring_df.columns:
        score_psi = compute_psi(
            reference_df[score_col], monitoring_df[score_col], bins=bins
        )

    # Feature CSI
    csi_tbl = compute_csi(reference_df, monitoring_df, feature_cols, bins=bins)

    counts = csi_tbl["stability"].value_counts().to_dict()

    logger.info(
        "drift_report: score_psi=%.4f (%s) | features monitored=%d | "
        "stable=%d | moderate=%d | significant=%d",
        score_psi if not np.isnan(score_psi) else -1,
        _stability_label(score_psi),
        len(csi_tbl),
        counts.get("stable", 0),
        counts.get("moderate_shift", 0),
        counts.get("significant_shift", 0),
    )

    return {
        "score_psi": score_psi,
        "score_stability": _stability_label(score_psi),
        "csi_table": csi_tbl,
        "n_features_stable": counts.get("stable", 0),
        "n_features_moderate": counts.get("moderate_shift", 0),
        "n_features_significant": counts.get("significant_shift", 0),
        "n_features_monitored": len(csi_tbl),
    }

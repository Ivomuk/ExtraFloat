"""
extrafloat_segmentation_drift.py
=====================================
Feature drift monitoring for the Uganda MTN MoMo agent segmentation pipeline.

Computes Population Stability Index (PSI) and KL divergence for key feature
distributions to detect input drift that may invalidate segment assignments.

PSI interpretation (industry standard for financial / telecom models):
  PSI < 0.10    — no significant change          (stable)
  0.10–0.25     — moderate change                (warning — monitor closely)
  PSI > 0.25    — significant population shift   (critical — retrain recommended)

Typical usage
-------------
On the reference run (first production batch):

    from extrafloat_segmentation_drift import save_drift_baseline
    save_drift_baseline(features_df, selected_cols, config={"drift": {...}})

On each subsequent run:

    from extrafloat_segmentation_drift import build_drift_report, load_drift_baseline
    baseline = load_drift_baseline(config["drift"]["baseline_path"])
    report   = build_drift_report(baseline, features_df, selected_cols, config["drift"])
    # report["drift_detected"] == True  →  investigate before trusting segments
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PSI_STABLE: str = "stable"
PSI_WARNING: str = "warning"
PSI_CRITICAL: str = "critical"
PSI_UNKNOWN: str = "unknown"

# Default financial KPI features to monitor (subset of PROFILING_COLS + key volumes)
DEFAULT_DRIFT_FEATURES: tuple[str, ...] = (
    "cash_out_value_1m",
    "cash_out_value_3m",
    "cash_in_value_1m",
    "payment_value_1m",
    "commission",
    "cash_out_vol_1m",
    "cash_in_vol_1m",
    "tenure_years",
)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DRIFT_CONFIG: dict[str, Any] = {
    # Features to monitor.  Defaults to DEFAULT_DRIFT_FEATURES;
    # pass an explicit list to restrict or expand.
    "drift_features": list(DEFAULT_DRIFT_FEATURES),
    # Number of equal-frequency bins for PSI / KL computation.
    "n_bins": 10,
    # PSI thresholds — aligned with Basel / telecom industry conventions.
    "psi_warn_threshold": 0.10,
    "psi_critical_threshold": 0.25,
    # Path to a previously saved baseline Parquet (or CSV) file.
    # When empty/None the drift check is skipped.
    "baseline_path": "",
    # When True the current feature distributions are saved to baseline_save_path
    # after drift detection completes.
    "save_baseline": False,
    "baseline_save_path": "",
}

# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _get_drift_config(config: dict | None) -> dict[str, Any]:
    """Return deep-merged drift config, filling missing keys from defaults."""
    if config is None:
        return deepcopy(DEFAULT_DRIFT_CONFIG)
    merged = deepcopy(DEFAULT_DRIFT_CONFIG)
    merged.update(config)
    return merged


def _psi_status(psi: float, warn: float, critical: float) -> str:
    """Map a scalar PSI value to a status string."""
    if np.isnan(psi):
        return PSI_UNKNOWN
    if psi >= critical:
        return PSI_CRITICAL
    if psi >= warn:
        return PSI_WARNING
    return PSI_STABLE


def _quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return *n_bins* equal-frequency bin edges derived from *values*.

    Uses the expected (baseline) distribution to define edges so that
    bins contain roughly equal numbers of observations.  This is more
    robust than equal-width bins for right-skewed financial data.
    """
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.nanquantile(values, quantiles))
    # Pad extreme edges so all values fall inside the range
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Population Stability Index between two distributions.

    PSI = Σ (actual_pct_i − expected_pct_i) × ln(actual_pct_i / expected_pct_i)

    Bin edges are derived from the *expected* (baseline) distribution using
    equal-frequency quantiles, making this robust for skewed financial features.
    A small epsilon (1e-4) is added before computing proportions to avoid
    log(0) and 0/0.

    Parameters
    ----------
    expected : Baseline feature values (1-D).
    actual   : Current feature values (1-D).
    n_bins   : Number of equal-frequency bins (default 10).

    Returns
    -------
    float  PSI value ≥ 0, or NaN if either array is empty / constant.
    """
    eps = 1e-4
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        logger.debug("compute_psi: empty array(s) — returning NaN.")
        return float("nan")

    edges = _quantile_bin_edges(expected, n_bins)
    if len(edges) < 2:
        logger.debug("compute_psi: constant feature — PSI = 0.")
        return 0.0

    exp_counts = np.histogram(expected, bins=edges)[0].astype(float)
    act_counts = np.histogram(actual, bins=edges)[0].astype(float)

    exp_pct = (exp_counts / len(expected)) + eps
    act_pct = (act_counts / len(actual)) + eps

    # Renormalise after epsilon addition
    exp_pct /= exp_pct.sum()
    act_pct /= act_pct.sum()

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return max(0.0, psi)  # PSI is non-negative; guard against floating-point underflow


def compute_kl_divergence(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute KL divergence KL(expected ‖ actual) between two distributions.

    KL(P ‖ Q) = Σ P_i × ln(P_i / Q_i)

    Uses the same equal-frequency binning as :func:`compute_psi`.
    KL divergence is asymmetric and unbounded; use PSI as the primary
    monitoring metric and KL as a supplementary signal.

    Parameters
    ----------
    expected : Baseline feature values (1-D).
    actual   : Current feature values (1-D).
    n_bins   : Number of equal-frequency bins (default 10).

    Returns
    -------
    float  KL divergence ≥ 0, or NaN if either array is empty / constant.
    """
    eps = 1e-4
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    edges = _quantile_bin_edges(expected, n_bins)
    if len(edges) < 2:
        return 0.0

    p = np.histogram(expected, bins=edges)[0].astype(float) + eps
    q = np.histogram(actual, bins=edges)[0].astype(float) + eps

    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: BASELINE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────


def save_drift_baseline(
    df: pd.DataFrame,
    features: list[str],
    config: dict | None = None,
) -> str:
    """Persist feature distributions as the reference baseline for future drift checks.

    Saves only the *features* columns of *df* to either Parquet (preferred) or
    CSV (fallback).  The save path is taken from
    ``config["baseline_save_path"]`` or ``config["baseline_path"]``.

    Parameters
    ----------
    df       : DataFrame containing baseline feature columns.
    features : Feature column names to persist.
    config   : Drift config dict.  Uses DEFAULT_DRIFT_CONFIG if None.

    Returns
    -------
    str  Path the baseline was written to.

    Raises
    ------
    ValueError if no save path is configured.
    """
    cfg = _get_drift_config(config)
    save_path: str = cfg.get("baseline_save_path", "") or cfg.get("baseline_path", "")

    if not save_path:
        raise ValueError(
            "save_drift_baseline: config must contain 'baseline_save_path' or "
            "'baseline_path' to specify where to save the baseline."
        )

    valid_features = [c for c in features if c in df.columns]
    if not valid_features:
        raise ValueError(
            f"save_drift_baseline: none of the requested features {features} "
            "are present in the DataFrame."
        )

    baseline_df = df[valid_features].copy()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    if save_path.endswith(".parquet"):
        baseline_df.to_parquet(save_path, index=False)
    else:
        baseline_df.to_csv(save_path, index=False)

    logger.info(
        "save_drift_baseline: saved baseline (%d agents × %d features) to '%s'.",
        len(baseline_df),
        len(valid_features),
        save_path,
    )
    return save_path


def load_drift_baseline(path: str) -> pd.DataFrame:
    """Load a previously saved drift baseline DataFrame from *path*.

    Parameters
    ----------
    path : Path to a Parquet or CSV file previously saved by
           :func:`save_drift_baseline`.

    Returns
    -------
    pd.DataFrame  Baseline feature DataFrame.

    Raises
    ------
    FileNotFoundError if *path* does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"load_drift_baseline: baseline file '{path}' does not exist."
        )

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    logger.info(
        "load_drift_baseline: loaded baseline (%d agents × %d features) from '%s'.",
        len(df),
        len(df.columns),
        path,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: DRIFT REPORT
# ─────────────────────────────────────────────────────────────────────────────


def build_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str] | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Compute per-feature PSI and KL divergence, and return a structured drift report.

    For each feature in *features* that is present in both DataFrames, the
    function computes PSI and KL divergence, assigns a status string
    (stable / warning / critical), and aggregates an overall drift summary.

    Parameters
    ----------
    baseline_df : Reference (expected) feature distributions.
    current_df  : Current-run feature distributions.
    features    : Feature columns to evaluate.  If None, uses
                  ``config["drift_features"]`` ∩ columns present in both dfs.
    config      : Drift config dict.  Uses DEFAULT_DRIFT_CONFIG if None.

    Returns
    -------
    dict with structure::

        {
            "features": {
                "<feature_name>": {
                    "psi": float,
                    "kl_divergence": float,
                    "status": "stable" | "warning" | "critical" | "unknown",
                },
                ...
            },
            "overall_psi": float,          # mean PSI across checked features
            "n_features_checked": int,
            "n_critical": int,
            "n_warning": int,
            "drift_detected": bool,        # True if any feature is critical
            "baseline_n": int,
            "current_n": int,
        }
    """
    cfg = _get_drift_config(config)
    n_bins: int = int(cfg.get("n_bins", 10))
    warn_thr: float = float(cfg.get("psi_warn_threshold", 0.10))
    crit_thr: float = float(cfg.get("psi_critical_threshold", 0.25))

    if features is None:
        features = cfg.get("drift_features", list(DEFAULT_DRIFT_FEATURES))

    # Only evaluate features present in both DataFrames
    checkable = [
        f for f in features
        if f in baseline_df.columns and f in current_df.columns
    ]
    skipped = [f for f in features if f not in checkable]
    if skipped:
        logger.debug(
            "build_drift_report: skipping features absent in one or both DataFrames: %s",
            skipped,
        )

    feature_reports: dict[str, dict[str, Any]] = {}
    psi_values: list[float] = []

    for feat in checkable:
        baseline_vals = baseline_df[feat].values
        current_vals = current_df[feat].values

        psi = compute_psi(baseline_vals, current_vals, n_bins=n_bins)
        kl = compute_kl_divergence(baseline_vals, current_vals, n_bins=n_bins)
        status = _psi_status(psi, warn_thr, crit_thr)

        feature_reports[feat] = {
            "psi": psi,
            "kl_divergence": kl,
            "status": status,
        }

        if not np.isnan(psi):
            psi_values.append(psi)

        if status == PSI_CRITICAL:
            logger.warning(
                "build_drift_report: CRITICAL drift on '%s' — PSI=%.4f (threshold=%.2f). "
                "Segment assignments may be unreliable; consider retraining.",
                feat,
                psi,
                crit_thr,
            )
        elif status == PSI_WARNING:
            logger.info(
                "build_drift_report: WARNING drift on '%s' — PSI=%.4f (threshold=%.2f). "
                "Monitor this feature closely.",
                feat,
                psi,
                warn_thr,
            )

    n_critical = sum(1 for r in feature_reports.values() if r["status"] == PSI_CRITICAL)
    n_warning = sum(1 for r in feature_reports.values() if r["status"] == PSI_WARNING)
    overall_psi = float(np.mean(psi_values)) if psi_values else float("nan")

    report: dict[str, Any] = {
        "features": feature_reports,
        "overall_psi": overall_psi,
        "n_features_checked": len(checkable),
        "n_critical": n_critical,
        "n_warning": n_warning,
        "drift_detected": n_critical > 0,
        "baseline_n": len(baseline_df),
        "current_n": len(current_df),
    }

    logger.info(
        "build_drift_report: checked %d features — %d critical, %d warning, "
        "overall_psi=%.4f, drift_detected=%s",
        len(checkable),
        n_critical,
        n_warning,
        overall_psi,
        report["drift_detected"],
    )
    return report

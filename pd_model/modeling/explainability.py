"""
SHAP-based explainability and adverse action reasons.

Computes SHAP values for XGBoost and LightGBM models and extracts the
top N features driving each agent's PD higher (adverse action reasons).

Usage
-----
::

    from pd_model.modeling.explainability import compute_shap_values, build_adverse_action_df

    shap_vals = compute_shap_values(xgb_model, X_aligned, model_key="xgb")
    reasons   = build_adverse_action_df(shap_vals, feature_names=X_aligned.columns.tolist())
    # Returns DataFrame with adverse_reason_1/2/3 and their SHAP contributions
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pd_model.logging_config import get_logger

logger = get_logger(__name__)


# ======================================================================== #
# SHAP value computation
# ======================================================================== #

def compute_shap_values(model, X: pd.DataFrame, model_key: str) -> np.ndarray:
    """
    Compute SHAP values for a fitted XGBoost or LightGBM classifier.

    Uses ``shap.TreeExplainer`` which is exact (not approximate) for
    tree-based models and does not require a background dataset.

    Parameters
    ----------
    model     : fitted XGBClassifier or LGBMClassifier
    X         : aligned feature DataFrame (output of align_features)
    model_key : "xgb" or "lgb" — used only for logging

    Returns
    -------
    np.ndarray of shape (n_samples, n_features).
    Positive values increase predicted PD (adverse).
    Negative values decrease predicted PD (protective).
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for explainability. Install with: pip install shap"
        ) from exc

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)

    # Binary classifiers: shap_values returns a list [class_0, class_1] for some
    # libraries; extract class 1 (default / bad)
    if isinstance(raw, list):
        shap_arr = raw[1]
    else:
        shap_arr = raw

    logger.info(
        "compute_shap_values: model=%s | agents=%d | features=%d | "
        "mean_abs_shap=%.4f",
        model_key, shap_arr.shape[0], shap_arr.shape[1],
        float(np.abs(shap_arr).mean()),
    )
    return shap_arr.astype(np.float32)


# ======================================================================== #
# Adverse action reasons
# ======================================================================== #

def build_adverse_action_df(
    shap_values: np.ndarray,
    feature_names: list[str],
    n_reasons: int = 3,
    min_shap: float = 0.0,
) -> pd.DataFrame:
    """
    Extract the top N adverse (positive-SHAP) features per agent.

    Only features with SHAP > ``min_shap`` are considered adverse.
    If fewer than ``n_reasons`` features are adverse for an agent, the
    remaining reason slots are None / NaN.

    Parameters
    ----------
    shap_values  : (n_samples, n_features) array from compute_shap_values
    feature_names: ordered list matching shap_values columns
    n_reasons    : number of top adverse reasons to extract (default 3)
    min_shap     : minimum SHAP threshold to count as adverse (default 0.0)

    Returns
    -------
    DataFrame with columns:
        adverse_reason_1, adverse_reason_1_shap,
        adverse_reason_2, adverse_reason_2_shap,
        adverse_reason_3, adverse_reason_3_shap
    """
    feat_arr = np.array(feature_names)
    n = shap_values.shape[0]

    # Sort each row descending by SHAP — argsort on negated array
    top_idx = np.argsort(-shap_values, axis=1)[:, :n_reasons]  # (n, n_reasons)

    # Gather top SHAP values and corresponding feature names
    row_idx = np.arange(n)[:, None]                             # (n, 1) broadcast
    top_shap = shap_values[row_idx, top_idx]                    # (n, n_reasons)
    top_names = feat_arr[top_idx]                               # (n, n_reasons)

    # Mask out entries that don't clear min_shap
    adverse_mask = top_shap > min_shap

    result: dict[str, np.ndarray] = {}
    for i in range(n_reasons):
        col = i + 1
        names_col = np.where(adverse_mask[:, i], top_names[:, i], None)
        shap_col = np.where(adverse_mask[:, i], top_shap[:, i].round(4), np.nan)
        result[f"adverse_reason_{col}"] = names_col
        result[f"adverse_reason_{col}_shap"] = shap_col.astype(float)

    return pd.DataFrame(result)


# ======================================================================== #
# Feature importance summary (global)
# ======================================================================== #

def shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Global SHAP feature importance: mean absolute SHAP per feature.

    Returns
    -------
    DataFrame sorted by mean_abs_shap descending:
        feature, mean_abs_shap, mean_shap (signed average direction)
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs.round(5),
        "mean_shap": mean_signed.round(5),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df

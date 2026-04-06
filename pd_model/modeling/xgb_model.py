"""
XGBoost model training and evaluation for the PD pipeline.

Preserves all algorithms from file7.txt exactly; removes globals, print
statements, and hardcoded notebook state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from pd_model.config import feature_config
from pd_model.config.model_config import DEFAULT_CONFIG, ModelConfig
from pd_model.logging_config import get_logger
from pd_model.modeling.evaluation import build_decile_tables, safe_auc_with_reason

logger = get_logger(__name__)

_META_COLS = {feature_config.AGENT_KEY, feature_config.THIN_FILE_COL, feature_config.TARGET_COL}


# ======================================================================== #
# Monotone constraint builder
# ======================================================================== #

def build_monotone_constraints(
    feature_names: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[int]:
    """
    Derive monotone constraint direction (+1 / -1 / 0) per feature from the
    sign of Spearman rank correlation with the target.

    +1  → higher feature value → higher predicted probability (risk-increasing)
    -1  → higher feature value → lower predicted probability (risk-decreasing)
     0  → no monotone constraint (correlation is near-zero or undefined)

    A feature gets constraint 0 if fewer than 30 non-null paired observations
    are available or if the correlation is within ±0.05 of zero.
    """
    constraints: list[int] = []
    y_arr = pd.to_numeric(pd.Series(y_train), errors="coerce").values

    for col in feature_names:
        if col not in X_train.columns:
            constraints.append(0)
            continue
        x_arr = pd.to_numeric(X_train[col], errors="coerce").values
        mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
        n_valid = int(mask.sum())
        if n_valid < 30:
            constraints.append(0)
            continue
        corr_val, _ = spearmanr(x_arr[mask], y_arr[mask])
        if np.isnan(corr_val) or abs(corr_val) < 0.05:
            constraints.append(0)
        elif corr_val > 0:
            constraints.append(1)
        else:
            constraints.append(-1)

    logger.info(
        "build_monotone_constraints: +1=%d, -1=%d, 0=%d out of %d features",
        constraints.count(1),
        constraints.count(-1),
        constraints.count(0),
        len(constraints),
    )
    return constraints


# ======================================================================== #
# Training
# ======================================================================== #

def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: ModelConfig = DEFAULT_CONFIG,
    monotone_constraints: list[int] | None = None,
) -> tuple[xgb.XGBClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Fit an XGBClassifier and return scored DataFrames for train and val.

    X_train / X_val must NOT contain agent_msisdn, thin_file_flag, or
    bad_state — those must be provided separately in the meta columns.
    The caller is responsible for extracting feature-only matrices before
    calling this function.

    Parameters
    ----------
    X_train : feature matrix (pure numeric, no meta cols)
    y_train : binary target aligned to X_train
    X_val   : feature matrix aligned to y_val
    y_val   : binary target aligned to X_val
    cfg     : ModelConfig
    monotone_constraints : list of int (+1/-1/0) per feature column.
              If None, constraints are derived automatically.

    Returns
    -------
    (fitted_model, train_scored_df, val_scored_df)

    scored_df columns: agent_msisdn*, thin_file_flag*, bad_state, raw_score
    (* only present if those columns exist in the original X_train/X_val)
    """
    # Hard alignment checks (preserved from file7)
    assert X_train.columns.tolist() == X_val.columns.tolist(), \
        "[FATAL] Train/val column order mismatch"
    assert feature_config.THIN_FILE_COL not in X_train.columns.tolist(), \
        f"[FATAL] {feature_config.THIN_FILE_COL} leaked into model matrix"

    feature_cols = X_train.columns.tolist()

    if monotone_constraints is None:
        monotone_constraints = build_monotone_constraints(feature_cols, X_train, y_train)

    assert len(monotone_constraints) == len(feature_cols), \
        f"[FATAL] constraints length {len(monotone_constraints)} != features {len(feature_cols)}"

    constraints_str = "(" + ",".join(str(int(v)) for v in monotone_constraints) + ")"

    model = xgb.XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        learning_rate=cfg.xgb_learning_rate,
        max_depth=cfg.xgb_max_depth,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        reg_lambda=cfg.xgb_reg_lambda,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="auc",
        random_state=42,
    )
    model.set_params(monotone_constraints=constraints_str)

    y_train_arr = pd.Series(y_train, index=X_train.index).astype(int).values
    y_val_arr = pd.Series(y_val, index=X_val.index).astype(int).values

    logger.info(
        "XGBoost fit: train=%d rows | val=%d rows | features=%d",
        X_train.shape[0], X_val.shape[0], len(feature_cols),
    )
    model.fit(
        X_train,
        y_train_arr,
        eval_set=[(X_val, y_val_arr)],
        verbose=False,
    )

    train_raw = model.predict_proba(X_train)[:, 1]
    val_raw = model.predict_proba(X_val)[:, 1]

    # Hard row alignment checks (preserved from file7)
    assert len(train_raw) == X_train.shape[0] == len(y_train_arr), \
        "[FATAL] Train rows misaligned at scoring time"
    assert len(val_raw) == X_val.shape[0] == len(y_val_arr), \
        "[FATAL] Val rows misaligned at scoring time"

    train_scored = pd.DataFrame(
        {"bad_state": y_train_arr, "raw_score": train_raw},
        index=X_train.index,
    )
    val_scored = pd.DataFrame(
        {"bad_state": y_val_arr, "raw_score": val_raw},
        index=X_val.index,
    )

    train_auc = roc_auc_score(y_train_arr, train_raw)
    val_auc = roc_auc_score(y_val_arr, val_raw)
    logger.info("XGBoost train AUC=%.4f | val AUC=%.4f", train_auc, val_auc)

    return model, train_scored, val_scored


# ======================================================================== #
# Evaluation
# ======================================================================== #

def evaluate_xgb(
    model: xgb.XGBClassifier,
    train_scored: pd.DataFrame,
    val_scored: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute AUCs (overall + by thin_file segment) and feature importances.

    Returns
    -------
    dict with keys:
        train_auc, val_auc,
        segment_aucs  : list of dicts {split, thin, auc, reason}
        feature_importance : pd.Series (gain, sorted descending)
    """
    train_auc, train_why = safe_auc_with_reason(
        train_scored["bad_state"], train_scored["raw_score"]
    )
    val_auc, val_why = safe_auc_with_reason(
        val_scored["bad_state"], val_scored["raw_score"]
    )
    logger.info("XGBoost overall train AUC=%.4f (%s) | val AUC=%.4f (%s)",
                train_auc, train_why, val_auc, val_why)

    seg_aucs = []
    for split_name, scored_df in [("train", train_scored), ("val", val_scored)]:
        if feature_config.THIN_FILE_COL not in scored_df.columns:
            continue
        thin_ser = pd.to_numeric(scored_df[feature_config.THIN_FILE_COL], errors="coerce")
        for thin_value in [0, 1]:
            mask = thin_ser == thin_value
            auc_val, why = safe_auc_with_reason(
                scored_df.loc[mask, "bad_state"],
                scored_df.loc[mask, "raw_score"],
            )
            logger.info(
                "XGBoost %s thin=%d AUC=%.4f (%s)", split_name, thin_value,
                auc_val if not np.isnan(auc_val) else -1, why,
            )
            seg_aucs.append({"split": split_name, "thin": thin_value, "auc": auc_val, "reason": why})

    booster = model.get_booster()
    imp_gain = booster.get_score(importance_type="gain")
    imp_ser = pd.Series(imp_gain).sort_values(ascending=False)

    return {
        "train_auc": train_auc,
        "val_auc": val_auc,
        "segment_aucs": seg_aucs,
        "feature_importance": imp_ser,
    }


# ======================================================================== #
# Segment diagnostics (validation)
# ======================================================================== #

def segment_performance_table(val_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Build a thin / non-thin / missing segment performance summary for validation.
    Preserves logic from df_va_seg3 block in file7.
    """
    df = val_scored.copy()
    df["raw_score"] = pd.to_numeric(df["raw_score"], errors="coerce")
    df["bad_state"] = pd.to_numeric(df["bad_state"], errors="coerce").fillna(0).astype(int)

    if feature_config.THIN_FILE_COL in df.columns:
        thin_num = pd.to_numeric(df[feature_config.THIN_FILE_COL], errors="coerce")
        df["thin_file_flag_bin"] = np.where(
            pd.isna(thin_num), np.nan, (thin_num >= 0.5).astype(int)
        )
        df["thin_segment"] = (
            pd.Series(df["thin_file_flag_bin"]).map({0: "non_thin", 1: "thin"}).fillna("missing_flag")
        )
    else:
        df["thin_segment"] = "missing_flag"

    rows = []
    for seg_name in ["non_thin", "thin", "missing_flag"]:
        seg_df = df[df["thin_segment"] == seg_name]
        if seg_df.shape[0] == 0:
            continue
        y_vals = seg_df["bad_state"].values
        p_vals = seg_df["raw_score"].values
        ok_mask = ~pd.isna(p_vals)
        auc_val, _ = safe_auc_with_reason(
            pd.Series(y_vals[ok_mask]), pd.Series(p_vals[ok_mask])
        )
        rows.append({
            "segment": seg_name,
            "n": int(seg_df.shape[0]),
            "bad_rate": float(np.mean(y_vals)),
            "avg_raw_score": float(np.nanmean(p_vals)),
            "auc": auc_val,
            "score_null_rate": float(np.mean(pd.isna(df["raw_score"]))),
        })

    seg_tbl = pd.DataFrame(rows)
    if seg_tbl.shape[0] > 0:
        seg_order = pd.Categorical(
            seg_tbl["segment"], categories=["non_thin", "thin", "missing_flag"], ordered=True
        )
        seg_tbl = seg_tbl.assign(segment=seg_order).sort_values("segment")
    return seg_tbl

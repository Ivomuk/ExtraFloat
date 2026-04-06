"""
LightGBM model training and evaluation for the PD pipeline.

Preserves all algorithms from file8.txt exactly; removes globals, print
statements, and hardcoded notebook state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from pd_model.config import feature_config
from pd_model.config.model_config import DEFAULT_CONFIG, ModelConfig
from pd_model.logging_config import get_logger
from pd_model.modeling.evaluation import build_decile_tables, safe_auc_with_reason
from pd_model.modeling.xgb_model import build_monotone_constraints

logger = get_logger(__name__)


# ======================================================================== #
# Training
# ======================================================================== #

def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: ModelConfig = DEFAULT_CONFIG,
    monotone_constraints: list[int] | None = None,
) -> tuple[lgb.LGBMClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Fit an LGBMClassifier and return scored DataFrames for train and val.

    X_train / X_val must NOT contain agent_msisdn, thin_file_flag, or
    bad_state — those must be provided separately in the meta columns.

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

    scored_df columns: bad_state, raw_score
    """
    assert X_train.columns.tolist() == X_val.columns.tolist(), \
        "[FATAL] Train/val column order mismatch"

    feature_cols = X_train.columns.tolist()

    if monotone_constraints is None:
        monotone_constraints = build_monotone_constraints(feature_cols, X_train, y_train)

    assert len(monotone_constraints) == len(feature_cols), \
        f"[FATAL] constraints length {len(monotone_constraints)} != features {len(feature_cols)}"

    y_train_arr = pd.Series(y_train, index=X_train.index).astype(int).values
    y_val_arr = pd.Series(y_val, index=X_val.index).astype(int).values

    model = lgb.LGBMClassifier(
        n_estimators=cfg.lgb_n_estimators,
        learning_rate=cfg.lgb_learning_rate,
        num_leaves=cfg.lgb_num_leaves,
        subsample=cfg.lgb_subsample,
        colsample_bytree=cfg.lgb_colsample_bytree,
        min_child_samples=cfg.lgb_min_child_samples,
        reg_lambda=cfg.lgb_reg_lambda,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.set_params(monotone_constraints=monotone_constraints)

    logger.info(
        "LightGBM fit: train=%d rows | val=%d rows | features=%d",
        X_train.shape[0], X_val.shape[0], len(feature_cols),
    )
    model.fit(
        X_train,
        y_train_arr,
        eval_set=[(X_val, y_val_arr)],
        eval_metric="auc",
    )

    train_raw = model.predict_proba(X_train)[:, 1]
    val_raw = model.predict_proba(X_val)[:, 1]

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
    logger.info("LightGBM train AUC=%.4f | val AUC=%.4f", train_auc, val_auc)

    return model, train_scored, val_scored


# ======================================================================== #
# Evaluation
# ======================================================================== #

def evaluate_lgbm(
    model: lgb.LGBMClassifier,
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
        val_deciles   : dict from build_decile_tables on val_scored
    """
    train_auc, train_why = safe_auc_with_reason(
        train_scored["bad_state"], train_scored["raw_score"]
    )
    val_auc, val_why = safe_auc_with_reason(
        val_scored["bad_state"], val_scored["raw_score"]
    )
    logger.info("LightGBM overall train AUC=%.4f (%s) | val AUC=%.4f (%s)",
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
                "LightGBM %s thin=%d AUC=%.4f (%s)", split_name, thin_value,
                auc_val if not np.isnan(auc_val) else -1, why,
            )
            seg_aucs.append({"split": split_name, "thin": thin_value, "auc": auc_val, "reason": why})

    imp_ser = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=model.booster_.feature_name(),
    ).sort_values(ascending=False)

    val_deciles = build_decile_tables(val_scored, score_col="raw_score", target_col="bad_state")

    return {
        "train_auc": train_auc,
        "val_auc": val_auc,
        "segment_aucs": seg_aucs,
        "feature_importance": imp_ser,
        "val_deciles": val_deciles,
    }

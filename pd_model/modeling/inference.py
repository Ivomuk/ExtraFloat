"""
Inference module — score new agents not seen during training.

Loads serialized artifacts (models + calibration map + feature order) and
runs the complete pipeline end-to-end on raw agent DataFrames.

Usage
-----
::

    from pd_model.modeling.inference import load_artifacts, score_new_agents

    artifacts = load_artifacts(Path("artifacts/"))
    scored = score_new_agents(df_raw_new_agents, artifacts)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pd_model.config import feature_config
from pd_model.config.model_config import DEFAULT_CONFIG, ModelConfig
from pd_model.logging_config import get_logger
from pd_model.modeling.calibration import add_policy_flags, attach_cal_pd, make_policy_bucket

logger = get_logger(__name__)


# ======================================================================== #
# Artifact container
# ======================================================================== #

@dataclass
class ModelArtifacts:
    """Container for all serialized pipeline artifacts needed at inference time."""

    xgb_model: Any                          # xgb.XGBClassifier
    lgb_model: Any                          # lgb.LGBMClassifier
    feature_order: list[str]                # ordered list of model input features
    cal_map: pd.DataFrame                   # from build_pd_calibration_map (stacked)
    xgb_policy_thresholds: pd.DataFrame     # from build_policy_tables
    lgb_policy_thresholds: pd.DataFrame     # from build_policy_tables
    transform_report: pd.DataFrame          # from apply_pd_transformations


# ======================================================================== #
# Artifact loading
# ======================================================================== #

def load_artifacts(artifacts_dir: Path) -> ModelArtifacts:
    """
    Load all serialized artifacts from a directory.

    Expected files (created by run_pipeline.py):
        xgb_model.joblib
        lgbm_model.joblib
        feature_order.json
        pd_calibration_map.csv
        xgb_policy_thresholds.csv
        lgb_policy_thresholds.csv
        transform_report.csv

    Returns
    -------
    ModelArtifacts dataclass instance.
    """
    artifacts_dir = Path(artifacts_dir)

    def _require(fname: str) -> Path:
        p = artifacts_dir / fname
        if not p.exists():
            raise FileNotFoundError(
                f"[load_artifacts] Required artifact not found: {p}\n"
                "Run the full training pipeline first to generate all artifacts."
            )
        return p

    xgb_model = joblib.load(_require("xgb_model.joblib"))
    lgb_model = joblib.load(_require("lgbm_model.joblib"))

    with open(_require("feature_order.json")) as f:
        feature_order_data = json.load(f)
    feature_order = feature_order_data.get("selected_features", feature_order_data)
    if not isinstance(feature_order, list):
        raise ValueError("[load_artifacts] feature_order.json must contain a list of feature names")

    cal_map = pd.read_csv(_require("pd_calibration_map.csv"))
    xgb_policy_thresholds = pd.read_csv(_require("xgb_policy_thresholds.csv"))
    lgb_policy_thresholds = pd.read_csv(_require("lgb_policy_thresholds.csv"))
    transform_report = pd.read_csv(_require("transform_report.csv"))

    logger.info(
        "load_artifacts: %d features | cal_map bins=%d | xgb_thresh rows=%d",
        len(feature_order), len(cal_map), len(xgb_policy_thresholds),
    )

    return ModelArtifacts(
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        feature_order=feature_order,
        cal_map=cal_map,
        xgb_policy_thresholds=xgb_policy_thresholds,
        lgb_policy_thresholds=lgb_policy_thresholds,
        transform_report=transform_report,
    )


# ======================================================================== #
# Feature alignment
# ======================================================================== #

def align_features(
    df_transformed: pd.DataFrame,
    feature_order: list[str],
) -> pd.DataFrame:
    """
    Align a transformed DataFrame to the exact feature order used at training.

    - Features in feature_order but missing from df_transformed → filled with NaN
      (and a warning is logged for each)
    - Extra columns not in feature_order are silently dropped

    Returns
    -------
    DataFrame with columns == feature_order, in that order.
    """
    missing = [f for f in feature_order if f not in df_transformed.columns]
    if missing:
        logger.warning(
            "align_features: %d features in feature_order are missing from "
            "input DataFrame and will be filled with NaN: %s",
            len(missing), missing[:20],
        )
        for f in missing:
            df_transformed[f] = np.nan

    return df_transformed[feature_order]


# ======================================================================== #
# End-to-end inference
# ======================================================================== #

def score_new_agents(
    df_transformed: pd.DataFrame,
    artifacts: ModelArtifacts,
    agent_meta: pd.DataFrame | None = None,
    cfg: ModelConfig = DEFAULT_CONFIG,
    champion: str = "xgb",
) -> pd.DataFrame:
    """
    Score a transformed feature DataFrame with trained models.

    This function expects that the caller has already run the full feature
    engineering pipeline (Phase 2.1, Phase 2.2, scorecard, transformations)
    and passes the resulting transformed DataFrame here.

    For a fully automated end-to-end inference path that takes raw data and
    handles feature engineering internally, use ``run_inference_pipeline()``.

    Parameters
    ----------
    df_transformed : output of ``build_transformed_dataframe()`` for new agents
    artifacts      : loaded ModelArtifacts from ``load_artifacts()``
    agent_meta     : optional DataFrame with agent_msisdn, thin_file_flag,
                     never_loan_pd_like etc. (indexed like df_transformed).
                     If None, meta columns are extracted from df_transformed.
    cfg            : ModelConfig
    champion       : "xgb" or "lgb" — which model's cal_pd drives policy flags

    Returns
    -------
    DataFrame with one row per agent:
        agent_msisdn, thin_file_flag,
        xgb_raw_score, lgb_raw_score,
        cal_pd, final_policy_bucket,
        xgb_approved_at_10/20/50/80
    """
    if champion not in ("xgb", "lgb"):
        raise ValueError(f"champion must be 'xgb' or 'lgb', got '{champion}'")

    # Extract meta columns before aligning to feature matrix
    meta_cols = [feature_config.AGENT_KEY, feature_config.THIN_FILE_COL,
                 "never_loan_pd_like", "never_loan_score_0_100", "never_loan_points",
                 "never_loan_top_drivers"]

    if agent_meta is None:
        agent_meta = df_transformed[[c for c in meta_cols if c in df_transformed.columns]].copy()

    # Align feature matrix
    X = align_features(df_transformed.copy(), artifacts.feature_order)

    # Score with XGBoost
    xgb_raw = artifacts.xgb_model.predict_proba(X)[:, 1]
    # Score with LightGBM
    lgb_raw = artifacts.lgb_model.predict_proba(X)[:, 1]

    logger.info(
        "score_new_agents: %d agents scored | champion=%s", X.shape[0], champion
    )

    # Build output DataFrame
    out = agent_meta.copy().reset_index(drop=True)
    out["xgb_raw_score"] = xgb_raw
    out["lgb_raw_score"] = lgb_raw

    # Determine thin/thick split
    thin_col = feature_config.THIN_FILE_COL
    if thin_col in out.columns:
        thin_mask = pd.to_numeric(out[thin_col], errors="coerce").eq(1)
    else:
        thin_mask = pd.Series(False, index=out.index)

    # Calibrate and apply policy for thick-file agents using champion model
    thick_mask = ~thin_mask
    if thick_mask.sum() > 0:
        champ_score_col = "xgb_raw_score" if champion == "xgb" else "lgb_raw_score"
        thick_df = out.loc[thick_mask].copy()
        thick_df = thick_df.rename(columns={champ_score_col: feature_config.RAW_SCORE_COL})
        thick_df["bad_state"] = 0  # placeholder — not available at inference time

        try:
            thick_cal = attach_cal_pd(thick_df, artifacts.cal_map, champion, cfg=cfg)
            out.loc[thick_mask, feature_config.CAL_PD_COL] = thick_cal[feature_config.CAL_PD_COL].values

            thresh = (
                artifacts.xgb_policy_thresholds
                if champion == "xgb"
                else artifacts.lgb_policy_thresholds
            )
            out_thick = add_policy_flags(
                out.loc[thick_mask].copy(), thresh, prefix=champion, cfg=cfg
            )
            for col in out_thick.columns:
                if col not in out.columns:
                    out[col] = np.nan
                out.loc[thick_mask, col] = out_thick[col].values
        except Exception as exc:
            logger.warning("attach_cal_pd failed for thick-file agents: %s", exc)

    # Thin-file agents: use never_loan_pd_like as their PD estimate
    if thin_mask.sum() > 0 and "never_loan_pd_like" in out.columns:
        thin_pd = pd.to_numeric(out.loc[thin_mask, "never_loan_pd_like"], errors="coerce")
        out.loc[thin_mask, feature_config.CAL_PD_COL] = thin_pd.values
        logger.info(
            "score_new_agents: %d thin-file agents use never_loan_pd_like as cal_pd",
            thin_mask.sum(),
        )

    # Build policy bucket
    champ_col = f"{champion}_approved_at_50"
    if champ_col in out.columns:
        bucket = make_policy_bucket(out, prefix=champion, cfg=cfg)
        out[feature_config.POLICY_BUCKET_COL] = bucket

    return out


# ======================================================================== #
# Full end-to-end inference pipeline
# ======================================================================== #

def run_inference_pipeline(
    df_raw: pd.DataFrame,
    artifacts_dir: Path,
    repayment_df: pd.DataFrame | None = None,
    cfg: ModelConfig = DEFAULT_CONFIG,
    champion: str = "xgb",
) -> pd.DataFrame:
    """
    Full end-to-end inference: raw data → feature engineering → scoring.

    This is the entry point for scoring new agents at deployment time.
    It runs the exact same feature engineering steps as the training pipeline.

    Parameters
    ----------
    df_raw        : raw agent snapshot DataFrame (same schema as training data)
    artifacts_dir : path to directory containing model artifacts
    repayment_df  : optional repayment history DataFrame for Phase 2.2 features.
                    If None, loan repayment features are skipped (thin-file path).
    cfg           : ModelConfig
    champion      : "xgb" or "lgb"

    Returns
    -------
    Scored DataFrame (see score_new_agents for column details).
    """
    from pd_model.modeling.scorecard import add_never_loan_scorecard_from_phase_2_1
    from pd_model.preprocessing.loan_features import (
        add_thin_file_flags,
        classify_agent_loan_status,
        run_phase_2_2_repayment_pd_features,
    )
    from pd_model.preprocessing.transaction_features import run_phase_2_1_richer_tx_behaviour
    from pd_model.preprocessing.transformations import build_transformed_dataframe
    from pd_model.scoring.iv_selector import iv_filter_phase_2

    artifacts = load_artifacts(Path(artifacts_dir))

    logger.info("run_inference_pipeline: %d raw agents", df_raw.shape[0])

    # Phase 2.1 — transaction behaviour features
    df = run_phase_2_1_richer_tx_behaviour(df_raw.copy(), cfg=cfg)

    # Phase 2.2 — loan repayment features (optional)
    if repayment_df is not None:
        df, _ = run_phase_2_2_repayment_pd_features(df, repayment_df, cfg=cfg, verbose=False)
    df = classify_agent_loan_status(df)
    df = add_thin_file_flags(df, cfg=cfg)

    # Thin-file scorecard
    df = add_never_loan_scorecard_from_phase_2_1(df, cfg=cfg)

    # Feature transformation
    from pd_model.config import feature_config as fc
    from pd_model.preprocessing.transformations import get_and_classify_pd_features

    (pd_features, log_cols, cap_cols, _protected, signed_log_cols, _excluded_df) = (
        get_and_classify_pd_features(df)
    )
    df_transformed, _pruned, _report = build_transformed_dataframe(
        df,
        pd_features=pd_features,
        log_cols=log_cols,
        cap_cols=cap_cols,
        signed_log_cols=signed_log_cols,
        cfg=cfg,
    )

    return score_new_agents(
        df_transformed=df_transformed,
        artifacts=artifacts,
        agent_meta=df[[c for c in [
            fc.AGENT_KEY, fc.THIN_FILE_COL,
            "never_loan_pd_like", "never_loan_score_0_100",
            "never_loan_points", "never_loan_top_drivers",
        ] if c in df.columns]].copy(),
        cfg=cfg,
        champion=champion,
    )

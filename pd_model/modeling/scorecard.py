"""
Thin-file (never-loan) scorecard for agents without loan history.

Applies a points-based scorecard using Phase 2.1 transactional features,
normalises to a 0-100 scale, and converts to a PD-like probability via a
sigmoid transform.  Only applied to agents where ``thin_file_flag == 1``.

Also provides ``fit_thin_file_lr`` / ``apply_thin_file_lr`` which replace the
manual sigmoid with a regularised logistic regression fitted on training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pd_model.config.model_config import (
    DEFAULT_CONFIG,
    DEFAULT_SCORECARD_WEIGHTS,
    ModelConfig,
    ScorecardWeights,
)
from pd_model.logging_config import get_logger

logger = get_logger(__name__)

# Features available in the raw (pre-transformation) DataFrame that the LR uses.
# Mirror the inputs used by the manual scorecard so the same signals are available.
_THIN_FILE_LR_FEATURES: list[str] = [
    # Binary flags
    # Removed: is_fully_inactive_6m, is_consecutively_inactive, sharp_volume_drop_flag
    # Data shows flag=1 agents have near-zero bad rate (they cannot default if inactive/
    # declining) -- LR coefficients contradicted raw bad rates, indicating multicollinearity.
    "consistent_volume_decline_flag",
    "activity_restart_flag",
    "consistent_volume_growth_flag",
    "low_balance_flag",
    "balance_drawdown_flag",
    "net_cash_flow_negative_flag",
    "high_peer_dependency_flag",
    "cust_concentration_flag",
    "commission_without_activity_flag",
    "commission_drop_flag",
    # Continuous behavioural signals
    "num_inactive_horizons",
    "avg_balance_to_vol_3m_ratio",
    "net_cash_flow_3m",
    "vol_3m",
    "commission_vs_cluster_mean_ratio",
    "commission_per_vol_vs_cluster_ratio",
    "vol_monthly_volatility_cv",
]


def fit_thin_file_lr(
    df_train_raw: pd.DataFrame,
    df_val_raw: pd.DataFrame | None = None,
    label_col: str = "bad_state",
    feature_candidates: list[str] | None = None,
    random_state: int = 42,
    min_positives: int = 20,
) -> tuple[object | None, list[str]]:
    """Fit a balanced logistic regression on thin-file training agents.

    Replaces the manually-tuned sigmoid with a data-driven model.  Strong L2
    regularisation (C=0.1) compensates for the very low positive rate typical
    in thin-file populations (~0.04%).

    When ``df_val_raw`` is supplied the raw LR probabilities are probability-
    calibrated via isotonic regression (or Platt scaling when fewer than 50
    positives are available) fitted on the val thin-file subset.  This corrects
    the ~10-20x overestimation caused by ``class_weight='balanced'`` and
    ensures that ``cal_pd`` reflects the true ~0.04% default rate rather than
    the balanced-class ~0.5 artefact.

    Args:
        df_train_raw:       Raw training DataFrame containing ``thin_file_flag``,
                            the label column, and behavioural features.
        df_val_raw:         Optional raw validation DataFrame used to fit a
                            probability calibrator on held-out thin-file agents.
        label_col:          Binary target column name.
        feature_candidates: Columns to consider.  Defaults to ``_THIN_FILE_LR_FEATURES``.
        random_state:       RNG seed for reproducibility.
        min_positives:      Return ``(None, [])`` if fewer positives than this.

    Returns:
        ``(fitted_pipeline_or_calibrated_classifier, feature_cols)`` or
        ``(None, [])`` on failure.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    thin_flag = df_train_raw.get("thin_file_flag", pd.Series(0, index=df_train_raw.index))
    thin_mask = pd.to_numeric(thin_flag, errors="coerce").fillna(0).eq(1)
    df_thin = df_train_raw[thin_mask]

    if label_col not in df_thin.columns:
        logger.warning("fit_thin_file_lr: label_col '%s' not found -- skipping LR fit", label_col)
        return None, []

    y = pd.to_numeric(df_thin[label_col], errors="coerce").fillna(0).astype(int)
    n_pos = int(y.sum())
    n_thin = len(y)

    if n_pos < min_positives:
        logger.warning(
            "fit_thin_file_lr: only %d positives in %d thin-file train agents -- "
            "skipping LR fit (min_positives=%d)",
            n_pos, n_thin, min_positives,
        )
        return None, []

    candidates = feature_candidates if feature_candidates is not None else _THIN_FILE_LR_FEATURES
    feature_cols = [c for c in candidates if c in df_thin.columns]

    if not feature_cols:
        logger.warning("fit_thin_file_lr: no candidate features found -- skipping")
        return None, []

    X = df_thin[feature_cols].apply(pd.to_numeric, errors="coerce")

    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            C=0.1,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])
    lr_pipeline.fit(X, y)

    try:
        from sklearn.metrics import roc_auc_score
        train_auc = float(roc_auc_score(y, lr_pipeline.predict_proba(X)[:, 1]))
        logger.info(
            "fit_thin_file_lr: n_thin=%d, n_pos=%d (%.3f%%), train_auc=%.4f, n_features=%d",
            n_thin, n_pos, 100.0 * n_pos / max(n_thin, 1), train_auc, len(feature_cols),
        )
    except Exception:
        logger.info(
            "fit_thin_file_lr: fitted on %d thin-file agents (%d positives), n_features=%d",
            n_thin, n_pos, len(feature_cols),
        )

    # Probability calibration: correct the balanced-class overestimation using
    # held-out val thin-file agents so cal_pd reflects the true default rate.
    if df_val_raw is not None:
        try:
            from sklearn.calibration import CalibratedClassifierCV

            thin_col = "thin_file_flag"
            val_thin_flag = df_val_raw.get(thin_col, pd.Series(0, index=df_val_raw.index))
            val_thin_mask = pd.to_numeric(val_thin_flag, errors="coerce").fillna(0).eq(1)
            df_thin_val = df_val_raw.loc[val_thin_mask]

            X_cal = df_thin_val[[c for c in feature_cols if c in df_thin_val.columns]].reindex(
                columns=feature_cols
            ).apply(pd.to_numeric, errors="coerce")
            y_cal = pd.to_numeric(
                df_thin_val.get(label_col, pd.Series(dtype=float)), errors="coerce"
            ).fillna(0).astype(int)

            n_pos_cal = int(y_cal.sum())
            if n_pos_cal >= 10:
                method = "isotonic" if n_pos_cal >= 50 else "sigmoid"
                cal_clf = CalibratedClassifierCV(lr_pipeline, cv="prefit", method=method)
                cal_clf.fit(X_cal, y_cal)
                lr_pipeline = cal_clf
                logger.info(
                    "fit_thin_file_lr: calibrated (%s) on %d val thin-file agents, %d positives",
                    method, len(y_cal), n_pos_cal,
                )
            else:
                logger.warning(
                    "fit_thin_file_lr: only %d positives in val thin-file -- skipping calibration",
                    n_pos_cal,
                )
        except Exception as exc:
            logger.warning("fit_thin_file_lr: calibration failed (%s) -- using uncalibrated LR", exc)

    return lr_pipeline, feature_cols


def apply_thin_file_lr(
    df: pd.DataFrame,
    lr_pipeline: object,
    feature_cols: list[str],
    thin_file_col: str = "thin_file_flag",
) -> pd.DataFrame:
    """Replace ``never_loan_pd_like`` for thin-file agents using the fitted LR.

    Non-thin-file rows are left unchanged.  If any feature in ``feature_cols``
    is missing from *df*, it is filled with NaN so the pipeline's ``SimpleImputer``
    handles it as usual.

    Returns a copy of *df* with ``never_loan_pd_like`` updated.
    """
    df = df.copy()
    thin_flag = df.get(thin_file_col, pd.Series(0, index=df.index))
    thin_mask = pd.to_numeric(thin_flag, errors="coerce").fillna(0).eq(1)

    if not thin_mask.any():
        return df

    df_thin = df[thin_mask]
    X = pd.DataFrame(index=df_thin.index)
    for c in feature_cols:
        X[c] = pd.to_numeric(df_thin[c], errors="coerce") if c in df_thin.columns else np.nan

    y_prob = lr_pipeline.predict_proba(X)[:, 1]
    df.loc[thin_mask, "never_loan_pd_like"] = y_prob

    logger.info(
        "apply_thin_file_lr: updated never_loan_pd_like for %d thin-file agents "
        "(median=%.4f, p95=%.4f)",
        int(thin_mask.sum()),
        float(np.nanmedian(y_prob)),
        float(np.nanpercentile(y_prob, 95)),
    )
    return df


def add_never_loan_scorecard_from_phase_2_1(
    df_pd_in: pd.DataFrame,
    weights: ScorecardWeights = DEFAULT_SCORECARD_WEIGHTS,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Compute a risk scorecard for thin-file agents (``thin_file_flag == 1``).

    Adds three columns to the returned DataFrame:
    - ``never_loan_points``      - raw accumulated risk points.
    - ``never_loan_score_0_100`` - normalised 0-100 score (1st-99th percentile).
    - ``never_loan_pd_like``     - sigmoid-based PD probability.
    - ``never_loan_top_drivers`` - pipe-separated string of active risk drivers.

    For thick-file agents these columns are set to ``NaN``.

    Args:
        df_pd_in: Modelling DataFrame; must contain ``thin_file_flag`` or
                  ``has_loan_history`` to derive it.
        weights:  Scorecard point weights (from ``ScorecardWeights`` dataclass).
        cfg:      Model config supplying ``eps`` and normalisation quantiles.

    Returns:
        Copy of *df_pd_in* with scorecard columns added.
    """
    eps = cfg.eps
    df_sc = df_pd_in.copy()

    # Ensure thin_file_flag exists
    if "thin_file_flag" not in df_sc.columns:
        has_loan_num = pd.to_numeric(df_sc.get("has_loan_history", np.nan), errors="coerce").fillna(0)
        df_sc["thin_file_flag"] = (has_loan_num.eq(0)).astype(int)

    thin_mask = df_sc["thin_file_flag"].eq(1)
    n_thin = int(thin_mask.sum())
    logger.info("Scorecard: scoring %d thin-file agents", n_thin)

    if n_thin == 0:
        logger.warning("Scorecard: no thin-file agents found -- returning without scoring")
        df_sc["never_loan_points"] = np.nan
        df_sc["never_loan_score_0_100"] = np.nan
        df_sc["never_loan_pd_like"] = np.nan
        df_sc["never_loan_top_drivers"] = np.nan
        return df_sc

    # ------------------------------------------------------------------ #
    # Helper accessors
    # ------------------------------------------------------------------ #
    def _s_num(col: str) -> pd.Series:
        if col not in df_sc.columns:
            return pd.Series(np.nan, index=df_sc.index)
        return pd.to_numeric(df_sc[col], errors="coerce")

    def _s_flag(col: str) -> pd.Series:
        if col not in df_sc.columns:
            return pd.Series(0.0, index=df_sc.index)
        return pd.to_numeric(df_sc[col], errors="coerce").fillna(0).clip(0, 1)

    def _winsor(s: pd.Series, q_lo: float = 0.01, q_hi: float = 0.99) -> pd.Series:
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() < 2:
            return s_num
        lo = s_num.quantile(q_lo)
        hi = s_num.quantile(q_hi)
        return s_num.clip(lower=lo, upper=hi)

    def _safe_log1p_pos(s: pd.Series) -> pd.Series:
        """log1p of positive values only; non-positive -> NaN."""
        s_num = pd.to_numeric(s, errors="coerce")
        s_num = s_num.where(s_num > 0, np.nan)
        return np.log1p(s_num)

    # ------------------------------------------------------------------ #
    # Accumulate points
    # ------------------------------------------------------------------ #
    pts = pd.Series(0.0, index=df_sc.index)

    # Inactivity
    pts += weights.fully_inactive_6m * _s_flag("is_fully_inactive_6m")
    pts += weights.consecutively_inactive * _s_flag("is_consecutively_inactive")

    if "num_inactive_horizons" in df_sc.columns:
        inh = _s_num("num_inactive_horizons").fillna(0).clip(0, weights.inactive_horizon_cap)
        pts += weights.inactive_horizon_per_unit * inh

    # Volume trajectory
    pts += weights.sharp_volume_drop * _s_flag("sharp_volume_drop_flag")
    pts += weights.consistent_volume_decline * _s_flag("consistent_volume_decline_flag")
    pts += weights.activity_restart * _s_flag("activity_restart_flag")  # negative
    pts += weights.consistent_volume_growth * _s_flag("consistent_volume_growth_flag")  # negative

    # Balance / liquidity
    pts += weights.low_balance * _s_flag("low_balance_flag")
    pts += weights.balance_drawdown * _s_flag("balance_drawdown_flag")

    if "avg_balance_to_vol_3m_ratio" in df_sc.columns:
        bal_ratio = _winsor(_s_num("avg_balance_to_vol_3m_ratio"))
        bal_ratio_log = _safe_log1p_pos(bal_ratio)
        pts += weights.avg_bal_to_vol_log_coeff * bal_ratio_log.fillna(0)

    # Cash flow
    pts += weights.net_cash_flow_negative * _s_flag("net_cash_flow_negative_flag")

    if "net_cash_flow_3m" in df_sc.columns and "vol_3m" in df_sc.columns:
        net_flow = _s_num("net_cash_flow_3m")
        vol3 = _s_num("vol_3m").abs()
        net_flow_per_vol = net_flow / (vol3 + eps)
        neg = net_flow_per_vol.where(net_flow_per_vol < 0, 0)
        neg_w = _winsor(neg)
        pts += weights.net_flow_per_vol_coeff * np.log1p(np.abs(neg_w.fillna(0)))

    # Peer / customer concentration
    pts += weights.high_peer_dependency * _s_flag("high_peer_dependency_flag")
    pts += weights.cust_concentration * _s_flag("cust_concentration_flag")

    # Commission
    pts += weights.commission_without_activity * _s_flag("commission_without_activity_flag")
    pts += weights.commission_drop * _s_flag("commission_drop_flag")

    if "commission_vs_cluster_mean_ratio" in df_sc.columns:
        comm_vs = _winsor(_s_num("commission_vs_cluster_mean_ratio"))
        under = (1.0 - comm_vs).where(comm_vs < 1.0, 0)
        pts += weights.commission_vs_cluster_coeff * under.fillna(0)

    if "commission_per_vol_vs_cluster_ratio" in df_sc.columns:
        inten_vs = _winsor(_s_num("commission_per_vol_vs_cluster_ratio"))
        under_i = (1.0 - inten_vs).where(inten_vs < 1.0, 0)
        pts += weights.commission_per_vol_vs_cluster_coeff * under_i.fillna(0)

    # Volatility
    if "vol_monthly_volatility_cv" in df_sc.columns:
        vol_cv = _winsor(_s_num("vol_monthly_volatility_cv")).abs()
        q75 = vol_cv.quantile(0.75)
        excess = (vol_cv - q75).where(vol_cv > q75, 0)
        pts += weights.vol_cv_excess_coeff * np.log1p(excess.fillna(0))

    df_sc["never_loan_points"] = np.where(thin_mask, pts, np.nan)

    # ------------------------------------------------------------------ #
    # Normalise to 0-100
    # ------------------------------------------------------------------ #
    thin_pts = pd.Series(df_sc.loc[thin_mask, "never_loan_points"])
    n_valid = thin_pts.notna().sum()

    if n_valid < 2:
        logger.warning(
            "Scorecard: fewer than 2 valid thin-file point values (%d) -- skipping 0-100 normalisation",
            n_valid,
        )
        df_sc["never_loan_score_0_100"] = np.nan
    else:
        p1 = thin_pts.quantile(cfg.scorecard_norm_q_low)
        p99 = thin_pts.quantile(cfg.scorecard_norm_q_high)
        denom = (p99 - p1) + eps
        df_sc["never_loan_score_0_100"] = np.nan
        df_sc.loc[thin_mask, "never_loan_score_0_100"] = (
            (df_sc.loc[thin_mask, "never_loan_points"] - p1) / denom
        ).clip(0, 1) * 100.0

    # ------------------------------------------------------------------ #
    # Sigmoid PD-like probability
    # ------------------------------------------------------------------ #
    thin_pts_all = pd.Series(df_sc["never_loan_points"])
    med = thin_pts.median()
    iqr_val = thin_pts.quantile(0.75) - thin_pts.quantile(0.25)
    iqr_val = max(float(iqr_val), eps)  # guard IQR = 0

    z = (thin_pts_all - med) / iqr_val
    df_sc["never_loan_pd_like"] = np.where(
        thin_mask,
        1.0 / (1.0 + np.exp(-weights.sigmoid_coeff * z)),
        np.nan,
    )

    # ------------------------------------------------------------------ #
    # Top drivers
    # ------------------------------------------------------------------ #
    driver_map = [
        ("inactive6m", "is_fully_inactive_6m"),
        ("inactive_consec", "is_consecutively_inactive"),
        ("vol_drop", "sharp_volume_drop_flag"),
        ("vol_decline", "consistent_volume_decline_flag"),
        ("low_bal", "low_balance_flag"),
        ("drawdown", "balance_drawdown_flag"),
        ("net_outflow", "net_cash_flow_negative_flag"),
        ("peer_dep", "high_peer_dependency_flag"),
        ("cust_conc", "cust_concentration_flag"),
        ("comm_drop", "commission_drop_flag"),
        ("comm_wo_act", "commission_without_activity_flag"),
    ]

    tags_list = []
    for tag, col_name in driver_map:
        if col_name in df_sc.columns:
            tags_list.append(np.where(_s_flag(col_name) > 0, tag, ""))

    df_sc["never_loan_top_drivers"] = pd.Series(pd.NA, index=df_sc.index, dtype=object)
    if tags_list:
        tags_arr = np.vstack(tags_list).T
        tags_series = pd.Series(
            ["|".join([t for t in row if t != ""]) for row in tags_arr],
            index=df_sc.index,
        ).replace("", np.nan)
        df_sc.loc[thin_mask, "never_loan_top_drivers"] = tags_series.loc[thin_mask]

    logger.info(
        "Scorecard complete: median_score=%.1f, median_pd_like=%.3f",
        float(df_sc.loc[thin_mask, "never_loan_score_0_100"].median()),
        float(df_sc.loc[thin_mask, "never_loan_pd_like"].median()),
    )
    return df_sc

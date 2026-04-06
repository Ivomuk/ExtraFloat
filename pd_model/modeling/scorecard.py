"""
Thin-file (never-loan) scorecard for agents without loan history.

Applies a points-based scorecard using Phase 2.1 transactional features,
normalises to a 0-100 scale, and converts to a PD-like probability via a
sigmoid transform.  Only applied to agents where ``thin_file_flag == 1``.
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


def add_never_loan_scorecard_from_phase_2_1(
    df_pd_in: pd.DataFrame,
    weights: ScorecardWeights = DEFAULT_SCORECARD_WEIGHTS,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Compute a risk scorecard for thin-file agents (``thin_file_flag == 1``).

    Adds three columns to the returned DataFrame:
    - ``never_loan_points``      – raw accumulated risk points.
    - ``never_loan_score_0_100`` – normalised 0-100 score (1st–99th percentile).
    - ``never_loan_pd_like``     – sigmoid-based PD probability.
    - ``never_loan_top_drivers`` – pipe-separated string of active risk drivers.

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
        has_loan_num = pd.to_numeric(
            df_sc.get("has_loan_history", np.nan), errors="coerce"
        ).fillna(0)
        df_sc["thin_file_flag"] = (has_loan_num.eq(0)).astype(int)

    thin_mask = df_sc["thin_file_flag"].eq(1)
    n_thin = int(thin_mask.sum())
    logger.info("Scorecard: scoring %d thin-file agents", n_thin)

    if n_thin == 0:
        logger.warning("Scorecard: no thin-file agents found — returning without scoring")
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
        """log1p of positive values only; non-positive → NaN."""
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
        inh = _s_num("num_inactive_horizons").fillna(0).clip(
            0, weights.inactive_horizon_cap
        )
        pts += weights.inactive_horizon_per_unit * inh

    # Volume trajectory
    pts += weights.sharp_volume_drop * _s_flag("sharp_volume_drop_flag")
    pts += weights.consistent_volume_decline * _s_flag("consistent_volume_decline_flag")
    pts += weights.activity_restart * _s_flag("activity_restart_flag")          # negative
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
    # Normalise to 0–100
    # ------------------------------------------------------------------ #
    thin_pts = pd.Series(df_sc.loc[thin_mask, "never_loan_points"])
    n_valid = thin_pts.notna().sum()

    if n_valid < 2:
        logger.warning(
            "Scorecard: fewer than 2 valid thin-file point values (%d) — "
            "skipping 0-100 normalisation",
            n_valid,
        )
        df_sc["never_loan_score_0_100"] = np.nan
    else:
        p1 = thin_pts.quantile(cfg.scorecard_norm_q_low)
        p99 = thin_pts.quantile(cfg.scorecard_norm_q_high)
        denom = (p99 - p1) + eps
        df_sc["never_loan_score_0_100"] = np.nan
        df_sc.loc[thin_mask, "never_loan_score_0_100"] = (
            ((df_sc.loc[thin_mask, "never_loan_points"] - p1) / denom).clip(0, 1) * 100.0
        )

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

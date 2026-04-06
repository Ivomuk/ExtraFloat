"""
Phase 2.1 — Richer transactional behaviour features for PD modelling.

Transforms the raw agent mart snapshot into a rich feature set covering:
- Inactivity structure (consecutive, full, per-horizon)
- Activity restart / recovery signals
- Commission dependency risk
- Volume trend direction (decline / growth flags)
- Balance & liquidity stress
- Customer & peer dependency
- Net cash flow analysis
- Transaction mix stress indicators
- Cluster-relative commission and volume comparisons
- Monthly volatility proxies
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pd_model.config.model_config import DEFAULT_CONFIG, ModelConfig
from pd_model.logging_config import get_logger
from pd_model.validation.schema import require_columns

logger = get_logger(__name__)


def run_phase_2_1_richer_tx_behaviour(
    df_pd: pd.DataFrame,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Compute Phase 2.1 richer transactional behaviour features in-place on a copy.

    The function is defensive: every feature block checks that its required
    source columns exist before computing, so it is safe to call on partial
    snapshots.

    Args:
        df_pd: Combined train + validation snapshot DataFrame. Must contain
               ``agent_msisdn`` and ``snapshot_dt``.
        cfg:   Model configuration supplying numeric constants (e.g. ``cfg.eps``).

    Returns:
        A copy of *df_pd* enriched with Phase 2.1 features.
    """
    require_columns(df_pd, ["agent_msisdn"], context="run_phase_2_1")

    df = df_pd.copy()
    eps = cfg.eps

    cols_before = set(df.columns)

    # ------------------------------------------------------------------ #
    # A) Strong inactivity structure (consecutive & full inactivity)
    # ------------------------------------------------------------------ #
    if all(c in df.columns for c in ["vol_1m", "vol_3m", "vol_6m"]):
        df["is_fully_inactive_6m"] = (
            (df["vol_1m"].fillna(0) == 0)
            & (df["vol_3m"].fillna(0) == 0)
            & (df["vol_6m"].fillna(0) == 0)
        ).astype(int)

        df["is_consecutively_inactive"] = (
            (df["vol_1m"].fillna(0) == 0) & (df["vol_3m"].fillna(0) == 0)
        ).astype(int)

    # ------------------------------------------------------------------ #
    # B) Activity restart / recovery signal
    # ------------------------------------------------------------------ #
    if all(c in df.columns for c in ["vol_1m", "vol_3m"]):
        df["activity_restart_flag"] = (
            (df["vol_1m"] > 0) & (df["vol_3m"] == df["vol_1m"])
        ).astype(int)

    # ------------------------------------------------------------------ #
    # C) Conditional activity intensity
    # ------------------------------------------------------------------ #
    if "vol_3m" in df.columns:
        df["vol_3m_if_active"] = df["vol_3m"].where(df["vol_3m"] > 0, np.nan)

    # ------------------------------------------------------------------ #
    # D) Commission dependency risk
    # ------------------------------------------------------------------ #
    if all(c in df.columns for c in ["commission", "vol_3m"]):
        df["commission_without_activity_flag"] = (
            (df["commission"] > 0) & (df["vol_3m"].fillna(0) == 0)
        ).astype(int)

    # ------------------------------------------------------------------ #
    # Trend direction flags (volume decline / growth)
    # ------------------------------------------------------------------ #
    if all(c in df.columns for c in ["vol_1m", "vol_3m", "vol_6m"]):
        df["consistent_volume_decline_flag"] = (
            (df["vol_1m"] < df["vol_3m"] / 3.0)
            & (df["vol_3m"] < df["vol_6m"] / 2.0)
        ).astype(int)

        df["consistent_volume_growth_flag"] = (
            (df["vol_1m"] > df["vol_3m"] / 3.0)
            & (df["vol_3m"] > df["vol_6m"] / 2.0)
        ).astype(int)

    # ------------------------------------------------------------------ #
    # Liquidity & balance stress
    # ------------------------------------------------------------------ #
    if "account_balance" in df.columns:
        df["low_balance_flag"] = (df["account_balance"] <= 0).astype(int)

    if all(c in df.columns for c in ["account_balance", "vol_3m"]):
        df["balance_to_vol_3m_ratio"] = df["account_balance"] / (df["vol_3m"] + eps)

    if all(c in df.columns for c in ["average_balance", "vol_3m"]):
        df["avg_balance_to_vol_3m_ratio"] = df["average_balance"] / (df["vol_3m"] + eps)

    if all(c in df.columns for c in ["account_balance", "average_balance"]):
        df["balance_drawdown_flag"] = (
            df["account_balance"] < 0.5 * df["average_balance"]
        ).astype(int)

    # ------------------------------------------------------------------ #
    # Customer & peer dependence
    # ------------------------------------------------------------------ #
    if all(c in df.columns for c in ["cust_1m", "cust_3m"]):
        df["cust_concentration_flag"] = (
            (df["cust_1m"] / (df["cust_3m"] + eps)) > 0.8
        ).astype(int)

    if all(c in df.columns for c in ["cash_in_peers_3m", "cash_in_vol_3m"]):
        df["peer_dependency_ratio"] = df["cash_in_peers_3m"] / (
            df["cash_in_vol_3m"] + eps
        )
        df["high_peer_dependency_flag"] = (df["peer_dependency_ratio"] > 0.7).astype(int)

    # ------------------------------------------------------------------ #
    # Transaction mix & net flow stress
    # ------------------------------------------------------------------ #
    if all(c in df.columns for c in ["cash_in_value_3m", "cash_out_value_3m"]):
        df["net_cash_flow_3m"] = df["cash_in_value_3m"] - df["cash_out_value_3m"]
        df["net_cash_flow_negative_flag"] = (df["net_cash_flow_3m"] < 0).astype(int)

    if all(c in df.columns for c in ["payment_value_3m", "vol_3m"]):
        df["payment_intensity_ratio"] = df["payment_value_3m"] / (df["vol_3m"] + eps)

    # ------------------------------------------------------------------ #
    # Stress acceleration flags
    # ------------------------------------------------------------------ #
    if all(c in df.columns for c in ["vol_1m", "vol_3m"]):
        df["sharp_volume_drop_flag"] = (
            (df["vol_1m"] / (df["vol_3m"] + eps)) < 0.3
        ).astype(int)

    if all(c in df.columns for c in ["commission", "commission_cluster_mean"]):
        df["commission_drop_flag"] = (
            df["commission"] < 0.5 * df["commission_cluster_mean"]
        ).astype(int)

    # ------------------------------------------------------------------ #
    # Ensure tbl_dt is datetime (for recency calculations)
    # ------------------------------------------------------------------ #
    if "tbl_dt" in df.columns:
        if not str(df["tbl_dt"].dtype).startswith("datetime"):
            df["tbl_dt"] = pd.to_datetime(df["tbl_dt"], errors="coerce")

    # ------------------------------------------------------------------ #
    # 1) Cluster-relative commission and volume
    # ------------------------------------------------------------------ #
    if "commission" in df.columns and "commission_cluster_mean" in df.columns:
        df["commission_vs_cluster_mean_ratio"] = df["commission"] / (
            df["commission_cluster_mean"] + eps
        )
        df["commission_vs_cluster_mean_diff"] = (
            df["commission"] - df["commission_cluster_mean"]
        )

    if "vol_3m" in df.columns and "vol_3m_cluster_mean" in df.columns:
        df["vol_3m_vs_cluster_mean_ratio"] = df["vol_3m"] / (
            df["vol_3m_cluster_mean"] + eps
        )
        df["vol_3m_vs_cluster_mean_diff"] = df["vol_3m"] - df["vol_3m_cluster_mean"]

    # ------------------------------------------------------------------ #
    # 2) Commission intensity (per volume)
    # ------------------------------------------------------------------ #
    if "commission" in df.columns and "vol_3m" in df.columns:
        df["commission_per_vol_3m"] = df["commission"] / (df["vol_3m"] + eps)

    if (
        "cluster_avg_commission" in df.columns
        and "cluster_avg_vol_3m" in df.columns
        and "commission_per_vol_3m" in df.columns
    ):
        df["cluster_commission_per_vol_3m"] = df["cluster_avg_commission"] / (
            df["cluster_avg_vol_3m"] + eps
        )
        df["commission_per_vol_vs_cluster_ratio"] = df["commission_per_vol_3m"] / (
            df["cluster_commission_per_vol_3m"] + eps
        )

    # ------------------------------------------------------------------ #
    # 3) Volume trajectory, intensity and volatility (1m / 3m / 6m)
    # ------------------------------------------------------------------ #
    base_names = ["cash_out_vol", "cash_in_vol", "payment_vol", "vol"]

    for base in base_names:
        col_1m = f"{base}_1m"
        col_3m = f"{base}_3m"
        col_6m = f"{base}_6m"

        have_1m = col_1m in df.columns
        have_3m = col_3m in df.columns
        have_6m = col_6m in df.columns

        # Average monthly intensity
        if have_3m:
            df[f"{base}_avg_monthly_3m"] = df[col_3m] / 3.0
        if have_6m:
            df[f"{base}_avg_monthly_6m"] = df[col_6m] / 6.0

        # Recent vs longer-term shares / trajectory
        if have_1m and have_3m:
            df[f"{base}_share_1m_of_3m"] = df[col_1m] / (df[col_3m] + eps)
            prev2m = (df[col_3m] - df[col_1m]) / 2.0
            df[f"{base}_growth_1m_vs_prev2m"] = df[col_1m] / (prev2m + eps)

        if have_3m and have_6m:
            df[f"{base}_share_3m_of_6m"] = df[col_3m] / (df[col_6m] + eps)
            prev3m = (df[col_6m] - df[col_3m]) / 3.0
            df[f"{base}_growth_3m_vs_prev3m"] = df[col_3m] / (prev3m + eps)

        if have_1m and have_6m:
            df[f"{base}_share_1m_of_6m"] = df[col_1m] / (df[col_6m] + eps)

        # Monthly volatility proxy
        if have_1m and have_3m and have_6m:
            m1 = df[col_1m]
            m2 = (df[col_3m] - df[col_1m]) / 2.0
            m3 = (df[col_6m] - df[col_3m]) / 3.0

            monthly = np.vstack([m1.values, m2.values, m3.values]).T
            mean_monthly = monthly.mean(axis=1)
            std_monthly = monthly.std(axis=1)

            df[f"{base}_monthly_volatility_proxy"] = std_monthly
            df[f"{base}_monthly_volatility_cv"] = std_monthly / (mean_monthly + eps)

    # ------------------------------------------------------------------ #
    # 4) Explicit inactivity flags per horizon
    # ------------------------------------------------------------------ #
    for horizon in ["1m", "3m", "6m"]:
        col = f"vol_{horizon}"
        flag_col = f"is_inactive_{horizon}"
        if col in df.columns:
            df[flag_col] = (df[col].fillna(0) == 0).astype(int)

    inactivity_flag_cols = [
        c
        for c in ["is_inactive_1m", "is_inactive_3m", "is_inactive_6m"]
        if c in df.columns
    ]

    if inactivity_flag_cols:
        df["num_inactive_horizons"] = df[inactivity_flag_cols].sum(axis=1)

    if all(c in df.columns for c in inactivity_flag_cols):
        df["max_inactivity_horizon_flag"] = (
            df["is_inactive_1m"] + df["is_inactive_3m"] + df["is_inactive_6m"]
        )

    # ------------------------------------------------------------------ #
    # 5) Recency from snapshot date
    # ------------------------------------------------------------------ #
    if "tbl_dt" in df.columns:
        ref_date = df["tbl_dt"].max()
        df["days_since_snapshot"] = (ref_date - df["tbl_dt"]).dt.days

    # ------------------------------------------------------------------ #
    # Summary log
    # ------------------------------------------------------------------ #
    new_cols = sorted(set(df.columns) - cols_before)
    logger.info(
        "Phase 2.1: created %d transactional behaviour features "
        "(df shape: %s → %s)",
        len(new_cols),
        df_pd.shape,
        df.shape,
    )

    return df

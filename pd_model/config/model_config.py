"""
Central numeric configuration for the PD model pipeline.

All magic numbers, thresholds, and tunable hyperparameters live here.
Downstream modules receive a ModelConfig instance; they never hard-code values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ScorecardWeights:
    """Point weights for the thin-file (never-loan) scorecard."""

    # Inactivity
    fully_inactive_6m: float = 7.0
    consecutively_inactive: float = 4.0
    inactive_horizon_per_unit: float = 2.0       # multiplied by num_inactive_horizons (capped at 3)
    inactive_horizon_cap: float = 3.0

    # Volume trajectory
    sharp_volume_drop: float = 3.0
    consistent_volume_decline: float = 2.0
    activity_restart: float = -2.0               # protective (negative risk)
    consistent_volume_growth: float = -1.0       # protective

    # Balance / liquidity
    low_balance: float = 2.5
    balance_drawdown: float = 2.0
    avg_bal_to_vol_log_coeff: float = -0.5       # coefficient on log(avg_balance_to_vol_3m_ratio)

    # Cash flow
    net_cash_flow_negative: float = 2.5
    net_flow_per_vol_coeff: float = 1.5          # coefficient on log1p(|negative net_flow_per_vol|)

    # Peer / customer concentration
    high_peer_dependency: float = 1.5
    cust_concentration: float = 1.0

    # Commission
    commission_without_activity: float = 1.5
    commission_drop: float = 1.5
    commission_vs_cluster_coeff: float = 1.0     # coefficient on (1 - ratio) when ratio < 1
    commission_per_vol_vs_cluster_coeff: float = 0.75

    # Volatility
    vol_cv_excess_coeff: float = 0.75            # coefficient on log1p(excess vol_cv above p75)

    # Sigmoid calibration
    sigmoid_coeff: float = 1.25                  # z-score multiplier in 1/(1+exp(-sigmoid_coeff*z))


DEFAULT_SCORECARD_WEIGHTS = ScorecardWeights()


@dataclass(frozen=True)
class ModelConfig:
    """
    All numeric thresholds and tunable constants for the PD pipeline.

    Pass an instance of this class into every pipeline function that needs
    configuration, rather than hard-coding literals.
    """

    # ------------------------------------------------------------------ #
    # General numerics
    # ------------------------------------------------------------------ #
    eps: float = 1e-6                       # smoothing / division guard

    # ------------------------------------------------------------------ #
    # Thin-file policy
    # ------------------------------------------------------------------ #
    thin_file_pd_prior: float = 0.12        # fallback PD for agents with no loan history

    # ------------------------------------------------------------------ #
    # Winsorization / transformation
    # ------------------------------------------------------------------ #
    winsor_q_low: float = 0.005             # lower winsorization quantile
    winsor_q_high: float = 0.995            # upper winsorization quantile
    neg_frac_threshold: float = 0.01        # fraction of negatives that triggers signed-log
    finite_frac_min: float = 0.98           # minimum finite fraction; below this → revert
    min_unique_post_transform: int = 2      # minimum distinct non-null values post-transform

    # ------------------------------------------------------------------ #
    # Scorecard normalisation
    # ------------------------------------------------------------------ #
    scorecard_norm_q_low: float = 0.01      # 1st percentile for 0–100 normalisation
    scorecard_norm_q_high: float = 0.99     # 99th percentile for 0–100 normalisation

    # ------------------------------------------------------------------ #
    # Leakage detection
    # ------------------------------------------------------------------ #
    corr_high: float = 0.85                 # |corr| threshold for HIGH leakage
    corr_med: float = 0.65                  # |corr| threshold for MEDIUM leakage
    dominance_high: float = 0.70            # bad-rate jump threshold for binary dominance
    max_corr_cols: int = 500                # max numeric cols to include in correlation sweep

    # ------------------------------------------------------------------ #
    # IV / feature selection
    # ------------------------------------------------------------------ #
    iv_min: float = 0.02                    # minimum IV to select a feature
    iv_min_uplift: float = -0.05            # minimum IV uplift (after vs before transform)
    iv_high_flag: float = 10.0              # IV above this is flagged as suspicious
    iv_n_bins: int = 10                     # number of bins for IV computation

    # ------------------------------------------------------------------ #
    # XGBoost hyperparameters
    # ------------------------------------------------------------------ #
    xgb_n_estimators: int = 800
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 4
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_lambda: float = 1.0

    # ------------------------------------------------------------------ #
    # LightGBM hyperparameters
    # ------------------------------------------------------------------ #
    lgb_n_estimators: int = 800
    lgb_learning_rate: float = 0.03
    lgb_num_leaves: int = 31
    lgb_subsample: float = 0.85
    lgb_colsample_bytree: float = 0.85
    lgb_min_child_samples: int = 40
    lgb_reg_lambda: float = 1.0

    # ------------------------------------------------------------------ #
    # Calibration / bootstrap / policy
    # ------------------------------------------------------------------ #
    cal_n_bins: int = 50                      # isotonic calibration bins
    cal_min_n: int = 5000                     # fail-closed minimum rows for calibration
    cal_min_bads: int = 50                    # fail-closed minimum bads for calibration
    cal_min_coverage: float = 0.98            # minimum fraction of rows that receive cal_pd
    bootstrap_n: int = 2000                   # number of bootstrap samples for AUC CI
    policy_operating_points: tuple = (0.10, 0.20, 0.50, 0.80)  # approval rate thresholds

    # ------------------------------------------------------------------ #
    # Scorecard weights (embedded for single-config access)
    # ------------------------------------------------------------------ #
    scorecard: ScorecardWeights = field(default_factory=ScorecardWeights)


DEFAULT_CONFIG = ModelConfig()

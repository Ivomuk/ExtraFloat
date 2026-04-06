"""
Unified ops_scored table — thick-file PD model output + thin-file scorecard output.

Preserves all logic from file10.txt exactly:
  - Concatenates thick-file and thin-file rows
  - Applies thin-file operating point (approve lowest-PD agents)
  - Builds exec-ready summary tables

No calls to globals(), no hardcoded paths, no print() statements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pd_model.config import feature_config
from pd_model.logging_config import get_logger

logger = get_logger(__name__)

_THIN_SCORECARD_COLS = [
    "never_loan_points",
    "never_loan_score_0_100",
    "never_loan_pd_like",
    "never_loan_top_drivers",
]


# ======================================================================== #
# Unified ops table
# ======================================================================== #

def build_ops_scored_table(
    agent_policy_placement: pd.DataFrame,
    df_pd_sc: pd.DataFrame,
    thin_op_quantile: float = 0.50,
) -> pd.DataFrame:
    """
    Concatenate thick-file PD model output with thin-file scorecard output.

    Parameters
    ----------
    agent_policy_placement : agent-level policy DataFrame from the PD model
                             (thick-file agents, thin_file_flag==0)
    df_pd_sc               : scorecard DataFrame (contains never_loan_* columns)
    thin_op_quantile       : thin-file approval quantile on never_loan_pd_like.
                             Agents with pd_like <= this quantile are approved.
                             Default 0.50 → approve the 50% with lowest pd_like.

    Returns
    -------
    ops_scored : unified DataFrame with columns from both inputs + decision_source,
                 final_approved, final_policy_bucket.
    """
    # ---- 1) Thick-file block ----
    thick_out = agent_policy_placement.copy()
    thick_out[feature_config.DECISION_SOURCE_COL] = "PD_MODEL"

    # ---- 2) Thin-file block ----
    thin_cols = [feature_config.AGENT_KEY] + [
        c for c in _THIN_SCORECARD_COLS if c in df_pd_sc.columns
    ]
    thin_out = df_pd_sc[thin_cols].copy()
    thin_out[feature_config.THIN_FILE_COL] = 1
    thin_out[feature_config.DECISION_SOURCE_COL] = "SCORECARD"

    # ---- 3) Schema alignment ----
    for c in thick_out.columns:
        if c not in thin_out.columns:
            thin_out[c] = np.nan
    for c in thin_out.columns:
        if c not in thick_out.columns:
            thick_out[c] = np.nan
    final_cols = list(thick_out.columns)
    thin_out = thin_out[final_cols]
    thick_out = thick_out[final_cols]

    # ---- 4) Concat ----
    ops_scored = pd.concat([thick_out, thin_out], ignore_index=True)

    # ---- 5) Apply thin-file operating point ----
    ops_scored[feature_config.THIN_FILE_COL] = pd.to_numeric(
        ops_scored[feature_config.THIN_FILE_COL], errors="coerce"
    )
    thin_mask = ops_scored[feature_config.THIN_FILE_COL].eq(1)

    if thin_mask.sum() > 0 and "never_loan_pd_like" in ops_scored.columns:
        ops_scored["never_loan_pd_like"] = pd.to_numeric(
            ops_scored["never_loan_pd_like"], errors="coerce"
        )
        valid_thin_pd = ops_scored.loc[thin_mask, "never_loan_pd_like"].notna().sum()
        if valid_thin_pd > 0:
            thin_cutoff = ops_scored.loc[thin_mask, "never_loan_pd_like"].quantile(thin_op_quantile)
            thin_approved = (ops_scored.loc[thin_mask, "never_loan_pd_like"] <= thin_cutoff).astype(int)
            ops_scored.loc[thin_mask, "final_approved"] = thin_approved.values
            ops_scored.loc[thin_mask, feature_config.POLICY_BUCKET_COL] = np.where(
                thin_approved.values == 1,
                f"SCORECARD_APPROVE_{int(thin_op_quantile * 100)}",
                f"SCORECARD_DECLINE_{int(thin_op_quantile * 100)}",
            )
            logger.info(
                "build_ops_scored_table: thin-file op_point=%.0f%% | "
                "cutoff=%.4f | approved=%d / %d thin agents",
                thin_op_quantile * 100, thin_cutoff,
                int(thin_approved.sum()), int(thin_mask.sum()),
            )

    total = ops_scored.shape[0]
    thick_n = int((~thin_mask).sum())
    thin_n = int(thin_mask.sum())
    logger.info(
        "build_ops_scored_table: total=%d | thick=%d | thin=%d",
        total, thick_n, thin_n,
    )
    return ops_scored


# ======================================================================== #
# Exec summary
# ======================================================================== #

def build_exec_summary(
    ops_scored: pd.DataFrame,
    target_col: str = "bad_state",
    operating_point_col: str = "final_approved",
) -> pd.DataFrame:
    """
    Executive-ready 2×2 Approved / Declined summary.

    Returns
    -------
    DataFrame with columns: segment, n, share, obs_bad_rate, gap_pp
    """
    if operating_point_col not in ops_scored.columns:
        raise ValueError(
            f"[build_exec_summary] '{operating_point_col}' column not found. "
            "Run build_ops_scored_table first."
        )

    total_n = int(ops_scored.shape[0])
    approved_mask = ops_scored[operating_point_col] == 1
    declined_mask = ops_scored[operating_point_col] == 0

    approved_n = int(approved_mask.sum())
    declined_n = int(declined_mask.sum())

    overall_br = float(ops_scored[target_col].mean()) if target_col in ops_scored.columns else np.nan
    approved_br = float(ops_scored.loc[approved_mask, target_col].mean()) if target_col in ops_scored.columns else np.nan
    declined_br = float(ops_scored.loc[declined_mask, target_col].mean()) if target_col in ops_scored.columns else np.nan
    gap_pp = (declined_br - approved_br) * 100.0 if not (np.isnan(approved_br) or np.isnan(declined_br)) else np.nan

    exec_tbl = pd.DataFrame({
        "segment": ["Overall", "Approved", "Declined", "Gap (Declined − Approved)"],
        "n": [total_n, approved_n, declined_n, np.nan],
        "share": [1.0, approved_n / total_n if total_n > 0 else np.nan,
                  declined_n / total_n if total_n > 0 else np.nan, np.nan],
        "obs_bad_rate": [overall_br, approved_br, declined_br, np.nan],
        "gap_pp": [np.nan, np.nan, np.nan, gap_pp],
    })

    logger.info(
        "build_exec_summary: approved_rate=%.1f%% | approved_bad_rate=%.2f%% | gap=%.2f pp",
        (approved_n / total_n * 100) if total_n > 0 else 0,
        approved_br * 100 if not np.isnan(approved_br) else 0,
        gap_pp if not np.isnan(gap_pp) else 0,
    )
    return exec_tbl


# ======================================================================== #
# Bucket summary
# ======================================================================== #

def build_bucket_summary(
    ops_scored: pd.DataFrame,
    target_col: str = "bad_state",
    cal_pd_col: str = "cal_pd_xgb",
) -> pd.DataFrame:
    """
    Group-by final_policy_bucket × thin_file_flag → n, obs_bad_rate, avg_cal_pd.
    """
    grp_cols = [feature_config.POLICY_BUCKET_COL, feature_config.THIN_FILE_COL]
    grp_cols_present = [c for c in grp_cols if c in ops_scored.columns]

    agg: dict[str, Any] = {"n": (feature_config.AGENT_KEY, "count")}
    if target_col in ops_scored.columns:
        agg["obs_bad_rate"] = (target_col, "mean")
    if cal_pd_col in ops_scored.columns:
        agg["avg_cal_pd"] = (cal_pd_col, "mean")

    summary = (
        ops_scored.groupby(grp_cols_present, dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values(grp_cols_present)
    )
    summary["share"] = summary["n"] / summary["n"].sum()
    return summary

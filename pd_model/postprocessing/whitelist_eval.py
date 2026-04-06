"""
Whitelist / blacklist evaluation for thin-file scorecard agents.

Preserves all algorithms from file11.txt exactly:
  - MSISDN key normalisation (strip ".0" suffix)
  - Blacklist > whitelist deduplication
  - AUC via Mann–Whitney U (no sklearn dependency)
  - Decile lift table with deterministic tie-break
  - Cutoff sweep at percentile grid

No hardcoded paths, no globals(), no print() statements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pd_model.config.feature_config import (
    NON_PERF_BLACKLIST_REASONS,
    WL_BL_COL,
    WL_BL_KEY,
    WL_CATEGORY_COL,
    WL_REASON_COL,
)
from pd_model.logging_config import get_logger

logger = get_logger(__name__)


# ======================================================================== #
# Loading and merging
# ======================================================================== #

def _normalize_msisdn(ser: pd.Series) -> pd.Series:
    """Strip .0 suffix and whitespace from MSISDN strings."""
    return ser.astype(str).str.replace(".0", "", regex=False).str.strip()


def load_and_merge_lists(
    whitelist_path: str | Path,
    blacklist_path: str | Path,
) -> pd.DataFrame:
    """
    Load whitelist and blacklist CSVs, tag with list type, and deduplicate
    so blacklist rows take priority over whitelist rows for the same MSISDN.

    Returns
    -------
    DataFrame with agent_msisdn_key, xtrafloat_list_type, agent_category, reason
    """
    df_wl = pd.read_csv(whitelist_path)
    df_bl = pd.read_csv(blacklist_path)

    df_wl[WL_BL_COL] = "whitelist"
    df_bl[WL_BL_COL] = "blacklist"

    combined = pd.concat([df_wl, df_bl], ignore_index=True, sort=False)
    combined[WL_BL_KEY] = _normalize_msisdn(combined["agent_msisdn"])

    # Blacklist > whitelist priority
    combined[WL_BL_COL] = combined[WL_BL_COL].astype(str).str.lower()
    combined["_list_priority"] = np.where(combined[WL_BL_COL].eq("blacklist"), 1, 0)
    deduped = (
        combined.sort_values("_list_priority", ascending=False)
        .drop_duplicates(WL_BL_KEY)
        .drop(columns=["_list_priority"])
        .reset_index(drop=True)
    )

    logger.info(
        "load_and_merge_lists: whitelist=%d | blacklist=%d | deduped=%d",
        len(df_wl), len(df_bl), len(deduped),
    )
    return deduped


# ======================================================================== #
# AUC via Mann–Whitney U
# ======================================================================== #

def _mann_whitney_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute AUC via Mann–Whitney U statistic (no sklearn).
    Preserves the algorithm from file11.txt exactly.
    """
    ranks = pd.Series(scores).rank(method="average")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    sum_ranks_pos = float(ranks[labels == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ======================================================================== #
# Cutoff sweep
# ======================================================================== #

def cutoff_sweep(
    eval_df: pd.DataFrame,
    score_col: str = "never_loan_pd_like_num",
    label_col: str = "is_blacklisted",
    pct_grid: list[int] | None = None,
) -> pd.DataFrame:
    """
    Sweep approval percentile thresholds and compute blacklist capture vs.
    whitelist decline rates.

    Lower pd_like = safer agent = approved first.

    Parameters
    ----------
    eval_df   : DataFrame with score_col and label_col (1=blacklist, 0=whitelist)
    score_col : column with PD-like score (lower = safer)
    label_col : binary column (1 = blacklisted)
    pct_grid  : list of approval percentiles to sweep (default: 5, 10, …, 95)

    Returns
    -------
    DataFrame with columns:
        approval_pct, threshold_pd_like, approved_n,
        approved_blacklist_rate, approved_whitelist_rate,
        declined_blacklist_capture, declined_whitelist_rate,
        improvement_vs_baseline_black_rate
    """
    if pct_grid is None:
        pct_grid = list(range(5, 100, 5))

    sweep_df = eval_df[[score_col, label_col]].dropna().copy()
    sweep_df = sweep_df.sort_values(score_col).reset_index(drop=True)

    pd_vals = sweep_df[score_col].values
    y_vals = sweep_df[label_col].values.astype(int)

    n_total = int(sweep_df.shape[0])
    n_black_total = int(y_vals.sum())
    n_white_total = int(n_total - n_black_total)
    baseline_black_rate = n_black_total / n_total if n_total > 0 else np.nan

    rows = []
    for pct in pct_grid:
        idx = int(np.floor(n_total * pct / 100.0)) - 1
        idx = max(0, min(idx, n_total - 1))
        thr = float(pd_vals[idx])

        approved_mask = pd_vals <= thr
        declined_mask = ~approved_mask

        n_app = int(approved_mask.sum())
        n_dec = int(declined_mask.sum())

        app_black = int(y_vals[approved_mask].sum())
        app_white = int(n_app - app_black)
        dec_black = int(y_vals[declined_mask].sum())
        dec_white = int(n_dec - dec_black)

        rows.append({
            "approval_pct": pct,
            "threshold_pd_like": thr,
            "approved_n": n_app,
            "approved_blacklist_rate": app_black / n_app if n_app > 0 else np.nan,
            "approved_whitelist_rate": app_white / n_app if n_app > 0 else np.nan,
            "declined_blacklist_capture": dec_black / n_black_total if n_black_total > 0 else np.nan,
            "declined_whitelist_rate": dec_white / n_white_total if n_white_total > 0 else np.nan,
            "improvement_vs_baseline_black_rate": (
                baseline_black_rate - app_black / n_app
            ) if n_app > 0 and baseline_black_rate is not np.nan else np.nan,
        })

    return pd.DataFrame(rows)


# ======================================================================== #
# Full evaluation
# ======================================================================== #

def run_whitelist_blacklist_eval(
    ops_scored_thin: pd.DataFrame,
    whitelist_path: str | Path,
    blacklist_path: str | Path,
    non_perf_reasons: tuple[str, ...] = NON_PERF_BLACKLIST_REASONS,
    score_col: str = "never_loan_pd_like",
) -> dict[str, Any]:
    """
    Evaluate thin-file scorecard ranking against whitelist / blacklist ground truth.

    Parameters
    ----------
    ops_scored_thin  : thin-file rows from ops_scored (decision_source == SCORECARD)
    whitelist_path   : path to whitelist CSV
    blacklist_path   : path to blacklist CSV
    non_perf_reasons : blacklist reasons to exclude from performance evaluation
                       (e.g. "As requested by Director")
    score_col        : column with PD-like score (lower = safer)

    Returns
    -------
    dict with keys:
        auc              : overall AUC (Mann–Whitney) on full labeled set
        auc_perf_filtered: AUC after excluding non-performance blacklist reasons
        decile_table     : DataFrame — 10-decile lift table (all labeled)
        lift_table       : DataFrame — 10-decile forced equal-size table
                           (deterministic tie-break, perf-filtered)
        cutoff_sweep     : DataFrame — percentile sweep table (perf-filtered)
        eval_df          : merged scored + list labels DataFrame
        perf_eval_df     : eval_df after filtering non-performance reasons
    """
    # ---- Load and merge lists ----
    wl_bl = load_and_merge_lists(whitelist_path, blacklist_path)

    ops_eval = ops_scored_thin.copy()
    ops_eval[WL_BL_KEY] = _normalize_msisdn(ops_eval["agent_msisdn"])

    keep_cols = [WL_BL_KEY, WL_BL_COL]
    if WL_CATEGORY_COL in wl_bl.columns:
        keep_cols.append(WL_CATEGORY_COL)
    if WL_REASON_COL in wl_bl.columns:
        keep_cols.append(WL_REASON_COL)

    ops_eval = ops_eval.merge(wl_bl[keep_cols], on=WL_BL_KEY, how="left")
    ops_eval["is_blacklisted"] = (ops_eval[WL_BL_COL] == "blacklist").astype(int)
    ops_eval[f"{score_col}_num"] = pd.to_numeric(ops_eval[score_col], errors="coerce")
    score_num_col = f"{score_col}_num"

    logger.info(
        "run_whitelist_blacklist_eval: %d thin agents | whitelist=%d | blacklist=%d | unmatched=%d",
        len(ops_eval),
        int((ops_eval[WL_BL_COL] == "whitelist").sum()),
        int((ops_eval[WL_BL_COL] == "blacklist").sum()),
        int(ops_eval[WL_BL_COL].isna().sum()),
    )

    # ---- AUC on full labeled set ----
    perf_full = ops_eval[[score_num_col, "is_blacklisted"]].dropna().copy()
    auc_full = np.nan
    if perf_full.shape[0] > 0 and perf_full["is_blacklisted"].nunique() == 2:
        auc_full = _mann_whitney_auc(perf_full[score_num_col].values, perf_full["is_blacklisted"].values)
    logger.info("AUC (full labeled): %.4f", auc_full)

    # ---- Performance-filtered subset ----
    perf_eval = ops_eval[ops_eval[WL_BL_COL].isin(["whitelist", "blacklist"])].copy()
    if non_perf_reasons and WL_REASON_COL in perf_eval.columns:
        perf_eval["_reason_str"] = perf_eval[WL_REASON_COL].astype(str)
        perf_eval = perf_eval[
            ~((perf_eval[WL_BL_COL] == "blacklist") & perf_eval["_reason_str"].isin(non_perf_reasons))
        ].copy()
        perf_eval = perf_eval.drop(columns=["_reason_str"])

    logger.info(
        "perf_filtered: %d rows | whitelist=%d | blacklist=%d",
        len(perf_eval),
        int((perf_eval[WL_BL_COL] == "whitelist").sum()),
        int((perf_eval[WL_BL_COL] == "blacklist").sum()),
    )

    perf = perf_eval[[score_num_col, "is_blacklisted"]].dropna().copy()
    auc_perf = np.nan
    if perf.shape[0] > 0 and perf["is_blacklisted"].nunique() == 2:
        auc_perf = _mann_whitney_auc(perf[score_num_col].values, perf["is_blacklisted"].values)
    logger.info("AUC (perf-filtered): %.4f", auc_perf)

    # ---- Decile table (rank-based, perf-filtered) ----
    decile_tbl = pd.DataFrame()
    if perf.shape[0] > 0 and perf["is_blacklisted"].nunique() == 2:
        perf_rank = perf.copy()
        perf_rank["score_pct"] = perf_rank[score_num_col].rank(pct=True, method="average")
        eps_val = 1e-12
        perf_rank["decile"] = np.ceil(
            np.clip(perf_rank["score_pct"], eps_val, 1.0) * 10
        ).astype(int)
        decile_tbl = (
            perf_rank.groupby("decile")
            .agg(
                n=("is_blacklisted", "size"),
                blacklist_rate=("is_blacklisted", "mean"),
                avg_pd_like=(score_num_col, "mean"),
                min_pd_like=(score_num_col, "min"),
                max_pd_like=(score_num_col, "max"),
            )
            .sort_index()
            .reset_index()
        )

    # ---- Forced equal-size lift table (deterministic tie-break) ----
    lift_tbl = pd.DataFrame()
    if WL_BL_KEY in perf_eval.columns:
        perf_bins = perf_eval[[WL_BL_KEY, score_num_col, "is_blacklisted"]].dropna().copy()
    else:
        perf_bins = perf_eval[[score_num_col, "is_blacklisted"]].dropna().copy()
        perf_bins[WL_BL_KEY] = perf_bins.index.astype(str)

    if perf_bins.shape[0] > 0:
        perf_bins = perf_bins.sort_values(
            [score_num_col, WL_BL_KEY], ascending=[True, True]
        ).reset_index(drop=True)
        n_rows = perf_bins.shape[0]
        perf_bins["row_num"] = np.arange(n_rows)
        perf_bins["decile"] = (np.floor(perf_bins["row_num"] * 10.0 / n_rows).astype(int) + 1).clip(1, 10)
        lift_tbl = (
            perf_bins.groupby("decile")
            .agg(
                n=("is_blacklisted", "size"),
                blacklist_rate=("is_blacklisted", "mean"),
                avg_pd_like=(score_num_col, "mean"),
                min_pd_like=(score_num_col, "min"),
                max_pd_like=(score_num_col, "max"),
            )
            .sort_index()
            .reset_index()
        )

    # ---- Cutoff sweep ----
    sweep_tbl = cutoff_sweep(perf_eval, score_col=score_num_col, label_col="is_blacklisted")

    return {
        "auc": auc_full,
        "auc_perf_filtered": auc_perf,
        "decile_table": decile_tbl,
        "lift_table": lift_tbl,
        "cutoff_sweep": sweep_tbl,
        "eval_df": ops_eval,
        "perf_eval_df": perf_eval,
    }

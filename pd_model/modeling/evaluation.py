"""
Model evaluation utilities — decile tables, segment AUCs, model comparison.

All functions are pure (no globals, no print statements). Results are returned
as DataFrames or dicts; callers decide what to log or persist.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pd_model.logging_config import get_logger

logger = get_logger(__name__)


# ======================================================================== #
# AUC helpers
# ======================================================================== #

def safe_auc_with_reason(
    y: pd.Series,
    p: pd.Series,
) -> tuple[float, str]:
    """
    Compute AUC, returning (nan, reason) instead of raising on edge cases.

    Returns
    -------
    (auc_value, reason_str)  where reason_str is "ok" on success or a short
    diagnostic string otherwise.
    """
    df_tmp = pd.DataFrame({"y": pd.Series(y).reset_index(drop=True),
                           "p": pd.Series(p).reset_index(drop=True)}).dropna()
    if df_tmp.shape[0] == 0:
        return np.nan, "no_rows"
    if int(df_tmp["y"].nunique()) < 2:
        return np.nan, "single_class"
    return float(roc_auc_score(df_tmp["y"], df_tmp["p"])), "ok"


# ======================================================================== #
# Decile table builder
# ======================================================================== #

def build_decile_tables(
    df: pd.DataFrame,
    score_col: str = "raw_score",
    target_col: str = "bad_state",
    segment_col: str | None = None,
    n_bins: int = 10,
    thin_flag_col: str = "thin_file_flag",
    make_thin_segment_if_missing: bool = True,
    thin_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Build decile performance tables, optionally segmented.

    Parameters
    ----------
    df : scored DataFrame (must contain score_col and target_col)
    score_col : column with model scores / probabilities
    target_col : binary outcome column
    segment_col : optional pre-existing segment column; if None and
                  make_thin_segment_if_missing=True, a thin_segment column
                  is auto-constructed from thin_flag_col
    n_bins : number of quantile bins (default 10 = deciles)
    thin_flag_col : column used to build thin_segment when segment_col is None
    make_thin_segment_if_missing : auto-build thin_segment if segment_col absent
    thin_threshold : threshold to binarise thin_flag_col (default 0.5)

    Returns
    -------
    dict with keys:
        "seg_col"     : name of segment column used
        "seg_summary" : DataFrame — one row per segment with n, bad_rate, auc
        "overall"     : DataFrame — decile table over all rows
        "by_segment"  : dict[segment_name -> decile DataFrame]
    """
    df_work = df.copy()
    df_work[score_col] = pd.to_numeric(df_work[score_col], errors="coerce")
    df_work[target_col] = pd.to_numeric(df_work[target_col], errors="coerce").fillna(0).astype(int)

    # ---- segment resolution ----
    if segment_col is not None and segment_col in df_work.columns:
        seg_col_use = segment_col
        df_work[seg_col_use] = df_work[seg_col_use].astype("object").fillna("missing")
    elif make_thin_segment_if_missing and thin_flag_col in df_work.columns:
        thin_num = pd.to_numeric(df_work[thin_flag_col], errors="coerce")
        thin_bin = np.where(pd.isna(thin_num), np.nan, (thin_num >= thin_threshold).astype(int))
        df_work["thin_segment"] = (
            pd.Series(thin_bin, index=df_work.index)
            .map({0: "non_thin", 1: "thin"})
            .fillna("missing_flag")
        )
        seg_col_use = "thin_segment"
    else:
        df_work["__segment__"] = "all"
        seg_col_use = "__segment__"

    seg_vals = pd.Series(df_work[seg_col_use].unique()).dropna().tolist()

    # ---- segment summary ----
    seg_rows = []
    for seg_name in seg_vals:
        seg_df = df_work[df_work[seg_col_use] == seg_name]
        if seg_df.shape[0] == 0:
            continue
        y_vals = seg_df[target_col].values
        p_vals = seg_df[score_col].values
        mask = ~pd.isna(p_vals)
        auc_val, _ = safe_auc_with_reason(
            pd.Series(y_vals[mask]), pd.Series(p_vals[mask])
        )
        seg_rows.append({
            "segment": seg_name,
            "n": int(seg_df.shape[0]),
            "bad_rate": float(np.mean(y_vals)),
            "avg_score": float(np.nanmean(p_vals)),
            "auc": auc_val,
            "score_null_rate": float(np.mean(pd.isna(p_vals))),
        })

    seg_summary_df = pd.DataFrame(seg_rows)
    if seg_col_use == "thin_segment" and seg_summary_df.shape[0] > 0:
        seg_order = pd.Categorical(
            seg_summary_df["segment"],
            categories=["non_thin", "thin", "missing_flag"],
            ordered=True,
        )
        seg_summary_df = seg_summary_df.assign(segment=seg_order).sort_values("segment").reset_index(drop=True)
    elif seg_summary_df.shape[0] > 0:
        seg_summary_df = seg_summary_df.sort_values("segment").reset_index(drop=True)

    # ---- decile helper ----
    def _deciles_for(sub_df: pd.DataFrame, label: str) -> pd.DataFrame:
        tmp = sub_df.copy()
        tmp = tmp[pd.notna(tmp[score_col])]
        if tmp.shape[0] == 0:
            return pd.DataFrame({"note": [f"{label}: no non-null scores"], "n": [0]})
        if int(tmp[score_col].nunique()) < n_bins:
            return pd.DataFrame({
                "note": [f"{label}: not enough unique scores for {n_bins} bins"],
                "unique_scores": [int(tmp[score_col].nunique())],
                "n": [int(tmp.shape[0])],
            })
        tmp["decile"] = pd.qcut(tmp[score_col], n_bins, labels=False, duplicates="drop").astype(int) + 1
        overall_bad = float(tmp[target_col].mean())
        tbl = (
            tmp.groupby("decile")
            .agg(
                n=(target_col, "size"),
                bad_rate=(target_col, "mean"),
                avg_score=(score_col, "mean"),
                min_score=(score_col, "min"),
                max_score=(score_col, "max"),
            )
            .reset_index()
            .sort_values("decile")
        )
        tbl["lift_vs_overall"] = tbl["bad_rate"] / (overall_bad + 1e-12)
        return tbl

    overall_df = _deciles_for(df_work, "overall")
    by_segment: dict[str, pd.DataFrame] = {}
    for seg_name in seg_vals:
        seg_df = df_work[df_work[seg_col_use] == seg_name]
        by_segment[str(seg_name)] = _deciles_for(seg_df, f"segment={seg_name}")

    logger.info(
        "build_decile_tables: %d rows | segments=%s | bins=%d",
        df_work.shape[0],
        seg_vals,
        n_bins,
    )

    return {
        "seg_col": seg_col_use,
        "seg_summary": seg_summary_df,
        "overall": overall_df,
        "by_segment": by_segment,
    }


# ======================================================================== #
# Model comparison
# ======================================================================== #

def compare_models_deciles(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    score_a: str = "raw_score",
    score_b: str = "raw_score",
    target_col: str = "bad_state",
    segment_col: str = "thin_segment",
    n_bins: int = 10,
    name_a: str = "xgb",
    name_b: str = "lgb",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    Compare two scored DataFrames via decile tables and AUC.

    Returns
    -------
    (summary_df, res_a, res_b)  where summary_df has one row per model with
    auc, top_decile_bad_rate, bottom_decile_bad_rate; res_a / res_b are the
    full dicts returned by build_decile_tables.
    """

    def _build(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
        seg_col = segment_col if segment_col in df.columns else None
        return build_decile_tables(
            df,
            score_col=score_col,
            target_col=target_col,
            segment_col=seg_col,
            n_bins=n_bins,
            make_thin_segment_if_missing=True,
        )

    res_a = _build(df_a, score_a)
    res_b = _build(df_b, score_b)

    def _overall_auc(df: pd.DataFrame, sc: str) -> float:
        tmp = df.copy()
        tmp[sc] = pd.to_numeric(tmp[sc], errors="coerce")
        tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce").fillna(0).astype(int)
        mask = ~pd.isna(tmp[sc])
        auc_val, _ = safe_auc_with_reason(tmp.loc[mask, target_col], tmp.loc[mask, sc])
        return auc_val

    def _top_decile_bad(dec_df: pd.DataFrame) -> float:
        if "decile" not in dec_df.columns:
            return np.nan
        top = int(dec_df["decile"].max())
        vals = dec_df.loc[dec_df["decile"] == top, "bad_rate"].values
        return float(vals[0]) if len(vals) > 0 else np.nan

    def _bot_decile_bad(dec_df: pd.DataFrame) -> float:
        if "decile" not in dec_df.columns:
            return np.nan
        bot = int(dec_df["decile"].min())
        vals = dec_df.loc[dec_df["decile"] == bot, "bad_rate"].values
        return float(vals[0]) if len(vals) > 0 else np.nan

    auc_a = _overall_auc(df_a, score_a)
    auc_b = _overall_auc(df_b, score_b)

    summary = pd.DataFrame([
        {
            "model": name_a,
            "auc": auc_a,
            "top_decile_bad_rate": _top_decile_bad(res_a["overall"]),
            "bottom_decile_bad_rate": _bot_decile_bad(res_a["overall"]),
            "seg_col_used": res_a.get("seg_col"),
        },
        {
            "model": name_b,
            "auc": auc_b,
            "top_decile_bad_rate": _top_decile_bad(res_b["overall"]),
            "bottom_decile_bad_rate": _bot_decile_bad(res_b["overall"]),
            "seg_col_used": res_b.get("seg_col"),
        },
    ])

    logger.info(
        "compare_models_deciles: %s AUC=%.4f | %s AUC=%.4f",
        name_a, auc_a if not np.isnan(auc_a) else -1,
        name_b, auc_b if not np.isnan(auc_b) else -1,
    )

    return summary, res_a, res_b

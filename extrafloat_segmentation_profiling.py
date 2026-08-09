"""
extrafloat_segmentation_profiling.py
=====================================
Pack-based cluster profiling and whitelist / blacklist reference-list
merging for Uganda MTN MoMo agents.

This module no longer assigns business tiers. Earlier versions built a
second, independent quantile-ranked tier (Platinum/Gold/Silver/Bronze) on
top of pack-lift scores, alongside the ensemble-cluster `segment` in
extrafloat_segmentation_pipeline.py — two live, population-relative tiering
mechanisms computed every run. Business tiering now belongs solely to
`extrafloat_segmentation_scoring.py`'s deterministic capacity scorecard.
`build_cluster_pack_profiles` remains: it's still useful as a read-only
research tool (e.g. profiling agents grouped by `capacity_tier` to see what
KPIs actually characterise each tier), it just no longer feeds a tier
decision back into the pipeline.

Refactors cl_file8.txt (lines 1–660) and cl_file9.txt.

Market: Uganda (UG) — MTN Mobile Money agent segmentation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE PACK DEFINITIONS
# Each pack groups semantically related agent KPI columns.
# ─────────────────────────────────────────────────────────────────────────────

PROFILING_PACKS: dict[str, list[str]] = {
    "cash_in": [
        "cash_in_comm_1m", "cash_in_cust_1m", "cash_in_peers_1m",
        "cash_in_peers_3m", "cash_in_cust_3m", "cash_in_cust_6m",
        "cash_in_comm_3m", "cash_in_peers_6m", "cash_in_comm_6m",
        "cash_in_value_1m", "cash_in_value_3m", "cash_in_value_6m",
        "cash_in_vol_1m", "cash_in_vol_3m", "cash_in_vol_6m",
        "cash_in_revenue_3m", "cash_in_revenue_6m",
    ],
    "cash_out": [
        "cash_out_value_6m", "cash_out_value_3m", "cash_out_cust_3m",
        "cash_out_cust_6m", "cash_out_comm_3m", "cash_out_peers_6m",
        "cash_out_peers_3m", "cash_out_comm_6m", "cash_out_vol_3m",
        "cash_out_vol_6m",
    ],
    "customer_reach": [
        "cust_6m", "cust_3m", "cust_1m", "vol_1m", "vol_6m", "vol_3m",
    ],
    "payments": [
        "payment_revenue_6m", "payment_revenue_3m", "payment_revenue_1m",
        "payment_value_3m", "payment_value_1m", "payment_comm_3m",
        "payment_cust_3m", "payment_cust_6m", "payment_peers_1m",
        "payment_value_6m", "payment_comm_1m", "payment_peers_6m",
        "payment_cust_1m", "payment_peers_3m", "payment_comm_6m",
        "payment_vol_1m", "payment_vol_3m", "payment_vol_6m",
    ],
    "balances": ["account_balance", "average_balance"],
    "revenue": [
        "commission", "voucher_cust_6m", "voucher_cust_3m",
        "voucher_vol_6m", "voucher_comm_3m", "voucher_value_6m",
        "voucher_comm_6m", "voucher_vol_1m", "voucher_vol_3m",
        "voucher_comm_1m", "voucher_peers_1m", "voucher_value_1m",
        "voucher_value_3m", "voucher_revenue_6m", "voucher_revenue_3m",
        "voucher_cust_1m", "revenue_3m", "revenue_1m",
        "cash_out_revenue_6m", "cash_out_revenue_3m", "voucher_revenue_1m",
        "cash_in_revenue_1m", "revenue_6m", "voucher_peers_6m",
        "cash_out_revenue_1m", "voucher_peers_3m",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    """Strip whitespace, remove non-digit characters, cast to Int64 (nullable)."""
    cleaned = s.astype(str).str.strip().str.replace(r"\D", "", regex=True)
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce").astype("Int64")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────


def build_cluster_pack_profiles(
    df: pd.DataFrame,
    cluster_col: str,
    packs: dict[str, list[str]] | None = None,
    melt_for_heatmap: bool = True,
) -> dict[str, dict]:
    """
    Compute mean and lift profiles per cluster for each feature pack.

    Parameters
    ----------
    df :
        Agent-level DataFrame. Must contain *cluster_col* and at least some
        feature columns referenced in *packs*.
    cluster_col :
        Column containing cluster labels (int or str).
    packs :
        Mapping of pack_name → list of feature column names.
        Defaults to ``PROFILING_PACKS``.
    melt_for_heatmap :
        When True, also produce long-form DataFrames suitable for seaborn
        heatmaps (columns: [cluster_col, "feature", "value"]).

    Returns
    -------
    dict with keys:
        "means"  : {pack_name: DataFrame (index=cluster, cols=features)}
        "lifts"  : {pack_name: DataFrame (lift vs overall mean)}
        "melted" : {pack_name: DataFrame (long-form)} or ``{}`` when
                   *melt_for_heatmap* is False
    """
    if cluster_col not in df.columns:
        raise ValueError(
            f"build_cluster_pack_profiles: cluster_col '{cluster_col}' "
            f"not found in DataFrame (columns: {list(df.columns)[:20]})"
        )

    if packs is None:
        packs = PROFILING_PACKS

    logger.info(
        "build_cluster_pack_profiles: %d rows, %d clusters, %d packs",
        len(df),
        df[cluster_col].nunique(),
        len(packs),
    )

    means_out: dict[str, pd.DataFrame] = {}
    lifts_out: dict[str, pd.DataFrame] = {}
    melted_out: dict[str, pd.DataFrame] = {}

    for pack_name, pack_cols in packs.items():
        # Keep only columns that exist in df and are numeric
        valid_cols = [
            c for c in pack_cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not valid_cols:
            logger.warning(
                "build_cluster_pack_profiles: pack '%s' — no valid numeric columns "
                "found in DataFrame; skipping.",
                pack_name,
            )
            continue

        pack_df = df[[cluster_col] + valid_cols].copy()
        cluster_means = pack_df.groupby(cluster_col)[valid_cols].mean()

        # Overall (population) mean per feature
        overall_mean = pack_df[valid_cols].mean()

        # Lift = cluster_mean / overall_mean; guard against zero denominator
        lift_df = cluster_means.copy()
        for col in valid_cols:
            denom = overall_mean[col]
            if denom == 0 or np.isnan(denom):
                lift_df[col] = np.nan
            else:
                lift_df[col] = cluster_means[col] / denom

        means_out[pack_name] = cluster_means
        lifts_out[pack_name] = lift_df

        logger.info(
            "build_cluster_pack_profiles: pack '%s' — %d features, "
            "%d clusters profiled.",
            pack_name,
            len(valid_cols),
            len(cluster_means),
        )

        if melt_for_heatmap:
            melted = (
                cluster_means
                .reset_index()
                .melt(id_vars=cluster_col, var_name="feature", value_name="value")
            )
            melted_out[pack_name] = melted

    return {"means": means_out, "lifts": lifts_out, "melted": melted_out}


def merge_reference_lists(
    df: pd.DataFrame,
    whitelist_df: pd.DataFrame,
    blacklist_df: pd.DataFrame | None = None,
    msisdn_col: str = "agent_msisdn",
) -> pd.DataFrame:
    """
    Left-join whitelist and blacklist reference lists onto *df*.

    Whitelist takes priority when an agent appears in both lists.

    Parameters
    ----------
    df :
        Main agent DataFrame.  Must contain *msisdn_col*.
    whitelist_df :
        Reference DataFrame for whitelisted agents.  Must contain
        *msisdn_col* and at least one of
        ``agent_category`` / ``CommissionDecision``.
    blacklist_df :
        Optional reference DataFrame for blacklisted agents. Same column
        expectations as *whitelist_df*.
    msisdn_col :
        MSISDN column name (present in all three DataFrames).

    Returns
    -------
    *df* with added/updated columns:
        CommissionDecision : ``"whitelist"`` | ``"blacklist"`` | ``NaN``
        agent_category     : from reference lists; whitelist takes priority
    """
    if msisdn_col not in df.columns:
        raise ValueError(
            f"merge_reference_lists: msisdn_col '{msisdn_col}' not in df "
            f"(columns: {list(df.columns)[:20]})"
        )
    if msisdn_col not in whitelist_df.columns:
        raise ValueError(
            f"merge_reference_lists: msisdn_col '{msisdn_col}' not in whitelist_df."
        )

    result = df.copy()
    result[msisdn_col] = _normalize_msisdn(result[msisdn_col])

    # ── Prepare whitelist ────────────────────────────────────────────────────
    wl = whitelist_df.copy()
    wl[msisdn_col] = _normalize_msisdn(wl[msisdn_col])
    wl = wl.drop_duplicates(subset=[msisdn_col])

    if "CommissionDecision" not in wl.columns:
        wl["CommissionDecision"] = "whitelist"
    else:
        wl["CommissionDecision"] = wl["CommissionDecision"].fillna("whitelist")

    wl_cols = [msisdn_col, "CommissionDecision"]
    if "agent_category" in wl.columns:
        wl_cols.append("agent_category")

    result = result.merge(
        wl[wl_cols].rename(
            columns={
                "CommissionDecision": "_wl_decision",
                "agent_category": "_wl_category",
            }
        ),
        on=msisdn_col,
        how="left",
        validate="many_to_one",
    )

    logger.info(
        "merge_reference_lists: whitelist matched %d / %d agents.",
        result["_wl_decision"].notna().sum(),
        len(result),
    )

    # ── Prepare blacklist ────────────────────────────────────────────────────
    if blacklist_df is not None:
        if msisdn_col not in blacklist_df.columns:
            raise ValueError(
                f"merge_reference_lists: msisdn_col '{msisdn_col}' "
                f"not in blacklist_df."
            )
        bl = blacklist_df.copy()
        bl[msisdn_col] = _normalize_msisdn(bl[msisdn_col])
        bl = bl.drop_duplicates(subset=[msisdn_col])

        if "CommissionDecision" not in bl.columns:
            bl["CommissionDecision"] = "blacklist"
        else:
            bl["CommissionDecision"] = bl["CommissionDecision"].fillna("blacklist")

        bl_cols = [msisdn_col, "CommissionDecision"]
        if "agent_category" in bl.columns:
            bl_cols.append("agent_category")

        result = result.merge(
            bl[bl_cols].rename(
                columns={
                    "CommissionDecision": "_bl_decision",
                    "agent_category": "_bl_category",
                }
            ),
            on=msisdn_col,
            how="left",
            validate="many_to_one",
        )

        logger.info(
            "merge_reference_lists: blacklist matched %d / %d agents.",
            result["_bl_decision"].notna().sum(),
            len(result),
        )
    else:
        result["_bl_decision"] = np.nan
        result["_bl_category"] = np.nan

    # ── Resolve priority: whitelist > blacklist ──────────────────────────────
    # CommissionDecision: whitelist wins if present, else blacklist, else NaN
    result["CommissionDecision"] = result["_wl_decision"].combine_first(
        result["_bl_decision"]
    )

    # agent_category: whitelist wins if present, else blacklist, else existing
    existing_cat = result["agent_category"].copy() if "agent_category" in result.columns else pd.Series(np.nan, index=result.index)
    wl_cat = result.get("_wl_category", pd.Series(np.nan, index=result.index))
    bl_cat = result.get("_bl_category", pd.Series(np.nan, index=result.index))

    result["agent_category"] = (
        wl_cat
        .combine_first(bl_cat)
        .combine_first(existing_cat)
    )

    # ── Drop intermediate merge columns ─────────────────────────────────────
    drop_cols = [c for c in ("_wl_decision", "_wl_category", "_bl_decision", "_bl_category") if c in result.columns]
    result = result.drop(columns=drop_cols)

    wl_count = (result["CommissionDecision"] == "whitelist").sum()
    bl_count = (result["CommissionDecision"] == "blacklist").sum()
    logger.info(
        "merge_reference_lists: final — whitelist=%d, blacklist=%d, "
        "unmatched=%d.",
        wl_count,
        bl_count,
        result["CommissionDecision"].isna().sum(),
    )

    return result

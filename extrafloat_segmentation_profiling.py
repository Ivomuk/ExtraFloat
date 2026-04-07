"""
extrafloat_segmentation_profiling.py
=====================================
Pack-based cluster profiling, tier building with safety filters, and
whitelist / blacklist reference-list merging for Uganda MTN MoMo agents.

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
# DEFAULT PROFILING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROFILING_CONFIG: dict[str, Any] = {
    "min_agents_per_tier": 30,
    "min_tenure_months": 3,
    "activity_quantile_threshold": 0.25,
    "cash_out_lift_platinum": 2.0,
    "cash_in_lift_platinum": 1.8,
    "cash_out_lift_gold": 1.5,
    "tier_quantile_high": 0.90,
    "tier_quantile_mid": 0.75,
    "zero_fraction_threshold": 0.99,
    "random_state": 42,
}

# ─────────────────────────────────────────────────────────────────────────────
# TIER LABELS  (ordered: highest → lowest)
# ─────────────────────────────────────────────────────────────────────────────

_TIER_ORDER: list[str] = ["Platinum", "Gold", "Silver", "Bronze"]

# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _get_config(config: dict | None) -> dict[str, Any]:
    """Return merged profiling config, filling missing keys from defaults."""
    if config is None:
        return dict(DEFAULT_PROFILING_CONFIG)
    merged = dict(DEFAULT_PROFILING_CONFIG)
    merged.update(config)
    return merged


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    """Strip whitespace, remove non-digit characters, cast to Int64 (nullable)."""
    cleaned = s.astype(str).str.strip().str.replace(r"\D", "", regex=True)
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce").astype("Int64")


def _apply_safety_filters(
    tier: str,
    n_agents: int,
    mean_tenure_months: float,
    cfg: dict[str, Any],
) -> tuple[str, list[str]]:
    """
    Optionally downgrade a cluster tier when safety thresholds are breached.

    Returns
    -------
    (final_tier, flags)
        final_tier : possibly downgraded tier string
        flags      : list of human-readable safety flag descriptions
    """
    flags: list[str] = []

    if n_agents < cfg["min_agents_per_tier"]:
        flags.append(
            f"n_agents={n_agents} < min_agents_per_tier={cfg['min_agents_per_tier']}"
        )

    if mean_tenure_months < cfg["min_tenure_months"]:
        flags.append(
            f"mean_tenure_months={mean_tenure_months:.1f} "
            f"< min_tenure_months={cfg['min_tenure_months']}"
        )

    if not flags:
        return tier, flags

    # Downgrade one step in the tier order for each triggered flag
    current_idx = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else len(_TIER_ORDER) - 1
    new_idx = min(current_idx + len(flags), len(_TIER_ORDER) - 1)
    downgraded = _TIER_ORDER[new_idx]

    if downgraded != tier:
        logger.info(
            "_apply_safety_filters: downgraded %s → %s (flags: %s)",
            tier,
            downgraded,
            "; ".join(flags),
        )

    return downgraded, flags


def _assign_sub_label(row: pd.Series, cfg: dict[str, Any]) -> str:  # noqa: ARG001
    """
    Determine sub-label from pack lift ratios present in *row*.

    Priority: High Cash-Out > High Cash-In > Payments-Led > Balanced.
    """
    co_lift = float(row.get("cash_out", np.nan) if not isinstance(row.get("cash_out"), float)
                    else row.get("cash_out", np.nan))
    ci_lift = float(row.get("cash_in", np.nan) if not isinstance(row.get("cash_in"), float)
                    else row.get("cash_in", np.nan))
    pay_lift = float(row.get("payments", np.nan) if not isinstance(row.get("payments"), float)
                     else row.get("payments", np.nan))

    co = co_lift if not np.isnan(co_lift) else 0.0
    ci = ci_lift if not np.isnan(ci_lift) else 0.0
    pay = pay_lift if not np.isnan(pay_lift) else 0.0

    co_thresh = cfg.get("cash_out_lift_gold", 1.5)
    ci_thresh = cfg.get("cash_in_lift_platinum", 1.8)

    if co >= co_thresh and co >= ci and co >= pay:
        return "High Cash-Out"
    if ci >= ci_thresh and ci >= co and ci >= pay:
        return "High Cash-In"
    if pay > 1.0 and pay >= co and pay >= ci:
        return "Payments-Led"
    return "Balanced"


def _compute_cluster_stats(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """
    Compute per-cluster descriptive statistics used by tier building.

    Returns
    -------
    DataFrame with index = cluster and columns:
        n_agents, mean_tenure_months, mean_activity
    """
    agg: dict[str, Any] = {"n_agents": (cluster_col, "count")}

    if "tenure_years" in df.columns:
        df = df.copy()
        df["_tenure_months_tmp"] = (
            pd.to_numeric(df["tenure_years"], errors="coerce") * 12.0
        )
        agg["mean_tenure_months"] = ("_tenure_months_tmp", "mean")

    if "cash_out_vol_1m" in df.columns:
        agg["mean_activity"] = ("cash_out_vol_1m", "mean")

    stats = df.groupby(cluster_col).agg(**agg)

    # Fill optional columns with safe defaults if absent
    if "mean_tenure_months" not in stats.columns:
        stats["mean_tenure_months"] = DEFAULT_PROFILING_CONFIG["min_tenure_months"]
        logger.info(
            "_compute_cluster_stats: 'tenure_years' absent — "
            "mean_tenure_months defaulted to %s",
            DEFAULT_PROFILING_CONFIG["min_tenure_months"],
        )
    if "mean_activity" not in stats.columns:
        stats["mean_activity"] = 0.0
        logger.info(
            "_compute_cluster_stats: 'cash_out_vol_1m' absent — "
            "mean_activity defaulted to 0.0"
        )

    return stats


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


def build_cluster_tiers(
    cluster_pack_scores: pd.DataFrame,
    cluster_stats: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Assign business tiers to clusters using quantile-based scoring and safety
    filters.

    Parameters
    ----------
    cluster_pack_scores :
        DataFrame whose index is the cluster label and whose columns are pack
        lift scores (one column per pack, e.g. "cash_out", "cash_in", …).
        Alternatively, must contain columns
        ["co_lift", "cr_lift", "n_agents", "mean_tenure_months",
        "mean_activity"] if pre-aggregated externally.
    cluster_stats :
        Optional DataFrame with per-cluster stats (n_agents,
        mean_tenure_months, mean_activity).  When provided, its values
        override any matching columns in *cluster_pack_scores*.
    config :
        Profiling configuration dict.  Missing keys fall back to
        ``DEFAULT_PROFILING_CONFIG``.

    Returns
    -------
    DataFrame with columns:
        cluster, tier, sub_label, n_agents, safety_flags
    """
    cfg = _get_config(config)

    scores = cluster_pack_scores.copy()
    if scores.index.name is None:
        scores.index.name = "cluster"

    logger.info(
        "build_cluster_tiers: %d clusters to tier.", len(scores)
    )

    # ── Merge cluster_stats if supplied ──────────────────────────────────────
    if cluster_stats is not None:
        for col in ("n_agents", "mean_tenure_months", "mean_activity"):
            if col in cluster_stats.columns:
                scores[col] = cluster_stats[col].reindex(scores.index).values

    # ── Ensure stat columns exist with safe defaults ─────────────────────────
    if "n_agents" not in scores.columns:
        scores["n_agents"] = cfg["min_agents_per_tier"]
        logger.warning(
            "build_cluster_tiers: 'n_agents' column absent — defaulted to %d.",
            cfg["min_agents_per_tier"],
        )
    if "mean_tenure_months" not in scores.columns:
        scores["mean_tenure_months"] = cfg["min_tenure_months"]
        logger.warning(
            "build_cluster_tiers: 'mean_tenure_months' column absent — "
            "defaulted to %.1f.",
            cfg["min_tenure_months"],
        )
    if "mean_activity" not in scores.columns:
        scores["mean_activity"] = 0.0

    # ── Composite score = row-wise mean of available lift columns ────────────
    stat_cols = {"n_agents", "mean_tenure_months", "mean_activity"}
    lift_cols = [c for c in scores.columns if c not in stat_cols]

    if not lift_cols:
        raise ValueError(
            "build_cluster_tiers: no lift columns found in cluster_pack_scores. "
            f"Available columns: {list(scores.columns)}"
        )

    scores["_composite"] = scores[lift_cols].mean(axis=1)

    # ── Quantile-based initial tier assignment ───────────────────────────────
    q_high = cfg["tier_quantile_high"]  # e.g. 0.90
    q_mid = cfg["tier_quantile_mid"]    # e.g. 0.75

    composite = scores["_composite"]
    thresh_high = composite.quantile(q_high)
    thresh_mid = composite.quantile(q_mid)
    thresh_low = composite.quantile(cfg.get("activity_quantile_threshold", 0.25))

    def _initial_tier(val: float) -> str:
        if val >= thresh_high:
            return "Platinum"
        if val >= thresh_mid:
            return "Gold"
        if val >= thresh_low:
            return "Silver"
        return "Bronze"

    scores["_initial_tier"] = composite.apply(_initial_tier)

    logger.info(
        "build_cluster_tiers: tier thresholds — Platinum≥%.3f, "
        "Gold≥%.3f, Silver≥%.3f.",
        thresh_high,
        thresh_mid,
        thresh_low,
    )

    # ── Apply safety filters and assign sub-labels ───────────────────────────
    records: list[dict[str, Any]] = []

    for cluster_label, row in scores.iterrows():
        initial = row["_initial_tier"]
        n_agents = int(row["n_agents"])
        mean_tenure = float(row["mean_tenure_months"])

        final_tier, flags = _apply_safety_filters(
            initial, n_agents, mean_tenure, cfg
        )
        sub_label = _assign_sub_label(row, cfg)

        records.append(
            {
                "cluster": cluster_label,
                "tier": final_tier,
                "sub_label": sub_label,
                "n_agents": n_agents,
                "safety_flags": "; ".join(flags) if flags else "",
            }
        )

    result = pd.DataFrame(records)
    tier_counts = result["tier"].value_counts().to_dict()
    logger.info("build_cluster_tiers: tier distribution — %s", tier_counts)

    return result


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

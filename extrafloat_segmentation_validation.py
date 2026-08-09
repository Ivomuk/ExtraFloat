"""
extrafloat_segmentation_validation.py
======================================
Segmentation quality validation for Uganda MTN MoMo mobile money agent
clustering.

Compares cluster assignments against a labeled reference dataset
(whitelist/blacklist with known ``agent_category`` labels) and provides
purity metrics, KPI comparisons, and feature importance analysis.

Market: Uganda (UG) — MTN Mobile Money agent segmentation.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# KPI COLUMNS
# Full set of agent profiling KPIs used across all segmentation analyses.
# ─────────────────────────────────────────────────────────────────────────────

KPI_COLUMNS_V1: tuple[str, ...] = (
    "account_balance", "average_balance", "commission",
    "revenue_1m", "revenue_3m", "revenue_6m",
    "cash_out_vol_1m", "cash_out_vol_3m", "cash_out_vol_6m",
    "cash_out_value_1m", "cash_out_value_3m", "cash_out_value_6m",
    "cash_out_peers_1m", "cash_out_peers_3m", "cash_out_peers_6m",
    "cash_out_revenue_1m", "cash_out_revenue_3m", "cash_out_revenue_6m",
    "cash_out_comm_1m", "cash_out_comm_3m", "cash_out_comm_6m",
    "cash_out_cust_1m", "cash_out_cust_3m", "cash_out_cust_6m",
    "cash_in_vol_1m", "cash_in_vol_3m", "cash_in_vol_6m",
    "cash_in_value_1m", "cash_in_value_3m", "cash_in_value_6m",
    "cash_in_peers_1m", "cash_in_peers_3m", "cash_in_peers_6m",
    "cash_in_revenue_1m", "cash_in_revenue_3m", "cash_in_revenue_6m",
    "cash_in_comm_1m", "cash_in_comm_3m", "cash_in_comm_6m",
    "cash_in_cust_1m", "cash_in_cust_3m", "cash_in_cust_6m",
    "voucher_vol_1m", "voucher_vol_3m", "voucher_vol_6m",
    "voucher_value_1m", "voucher_value_3m", "voucher_value_6m",
    "voucher_revenue_1m", "voucher_revenue_3m", "voucher_revenue_6m",
    "voucher_comm_1m", "voucher_comm_3m", "voucher_comm_6m",
    "payment_vol_1m", "payment_vol_3m", "payment_vol_6m",
    "payment_value_1m", "payment_value_3m", "payment_value_6m",
    "payment_revenue_1m", "payment_revenue_3m", "payment_revenue_6m",
    "payment_comm_1m", "payment_comm_3m", "payment_comm_6m",
    "payment_cust_1m", "payment_cust_3m", "payment_cust_6m",
    "cust_1m", "cust_3m", "cust_6m",
    "vol_1m", "vol_3m", "vol_6m",
)

# Default RandomForest hyperparameters
_RF_DEFAULT_N_ESTIMATORS: int = 300
_RF_DEFAULT_RANDOM_STATE: int = 42
_RF_DEFAULT_TEST_SIZE: float = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    """Raise ValueError if any of *columns* are absent from *df*."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _resolve_kpi_cols(
    df: pd.DataFrame,
    kpi_cols: list[str] | None,
    context: str,
) -> list[str]:
    """
    Return *kpi_cols* if provided, otherwise default to KPI_COLUMNS_V1
    filtered to columns actually present in *df*.
    """
    if kpi_cols is not None:
        _require_columns(df, kpi_cols, context)
        return kpi_cols
    available = [c for c in KPI_COLUMNS_V1 if c in df.columns]
    if not available:
        raise ValueError(
            f"[{context}] None of the default KPI_COLUMNS_V1 are present in the "
            "DataFrame. Pass kpi_cols explicitly."
        )
    return available


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def cluster_purity_table(
    df: pd.DataFrame,
    cluster_col: str = "ensemble_cluster",
    label_col: str = "agent_category",
    suffix: str = "",
) -> pd.DataFrame:
    """
    Compute per-cluster purity against a known reference label column.

    For each cluster, identifies the dominant label (mode), the count of
    agents with that label, and the purity (dominant_count / cluster_total).
    Also computes overall purity (weighted average across clusters).

    Parameters
    ----------
    df : DataFrame with cluster_col and label_col columns.
    cluster_col : Column containing cluster/segment labels.
    label_col : Column containing reference labels (e.g. agent_category).
    suffix : Optional suffix appended to output column names.

    Returns
    -------
    DataFrame with columns:
        cluster, n_agents, dominant_label, dominant_count, purity,
        overall_purity (scalar repeated on every row for convenience)
    """
    _require_columns(df, [cluster_col, label_col], "cluster_purity_table")

    logger.info(
        "cluster_purity_table: computing purity for %d agents across "
        "column '%s' vs label '%s'",
        len(df),
        cluster_col,
        label_col,
    )

    rows: list[dict[str, Any]] = []
    for cluster_id, group in df.groupby(cluster_col, sort=True):
        n_total = len(group)
        label_counts = group[label_col].value_counts(dropna=False)
        dominant_label = label_counts.index[0]
        dominant_count = int(label_counts.iloc[0])
        purity = dominant_count / n_total if n_total > 0 else 0.0
        rows.append(
            {
                "cluster": cluster_id,
                "n_agents": n_total,
                "dominant_label": dominant_label,
                "dominant_count": dominant_count,
                "purity": purity,
            }
        )

    result = pd.DataFrame(rows)

    total_agents = result["n_agents"].sum()
    overall_purity = (
        result["dominant_count"].sum() / total_agents if total_agents > 0 else 0.0
    )
    result["overall_purity"] = overall_purity

    if suffix:
        rename_map = {
            c: f"{c}{suffix}"
            for c in ["n_agents", "dominant_label", "dominant_count", "purity", "overall_purity"]
        }
        result = result.rename(columns=rename_map)

    logger.info(
        "cluster_purity_table: overall purity = %.4f across %d clusters",
        overall_purity,
        len(result),
    )
    return result


def compute_adjusted_rand_score(
    df: pd.DataFrame,
    cluster_col: str,
    label_col: str,
) -> float:
    """
    Compute sklearn adjusted_rand_score between cluster and reference label columns.

    Handles NaN by dropping rows where either column is NaN before computing.

    Parameters
    ----------
    df : DataFrame with cluster_col and label_col.
    cluster_col : Predicted cluster labels column.
    label_col : True reference labels column.

    Returns
    -------
    float : Adjusted Rand Index in [-1, 1].
    """
    _require_columns(df, [cluster_col, label_col], "compute_adjusted_rand_score")

    clean = df[[cluster_col, label_col]].dropna()
    n_dropped = len(df) - len(clean)
    if n_dropped:
        logger.info(
            "compute_adjusted_rand_score: dropped %d rows with NaN in '%s' or '%s'",
            n_dropped,
            cluster_col,
            label_col,
        )

    if len(clean) == 0:
        raise ValueError(
            "compute_adjusted_rand_score: no non-NaN rows remain after dropping "
            f"NaN values in '{cluster_col}' and '{label_col}'."
        )

    ari = float(adjusted_rand_score(clean[label_col], clean[cluster_col]))
    logger.info(
        "compute_adjusted_rand_score: ARI = %.6f (n=%d)", ari, len(clean)
    )
    return ari


def below_threshold_counts(
    df: pd.DataFrame,
    cluster_col: str = "ensemble_cluster",
    label_col: str = "agent_category",
    below_threshold_label: str = "Below Threshold",
) -> pd.DataFrame:
    """
    Count and share of agents labeled 'Below Threshold' per cluster.

    Parameters
    ----------
    df : Agent-level DataFrame.
    cluster_col : Cluster label column.
    label_col : Reference label column (contains below_threshold_label).
    below_threshold_label : The label string that denotes below-threshold agents.

    Returns
    -------
    DataFrame with columns: cluster, n_total, n_below_threshold, bt_share_pct
    """
    _require_columns(df, [cluster_col, label_col], "below_threshold_counts")

    logger.info(
        "below_threshold_counts: counting '%s' agents per cluster in '%s'",
        below_threshold_label,
        cluster_col,
    )

    is_bt = df[label_col] == below_threshold_label

    rows: list[dict[str, Any]] = []
    for cluster_id, group in df.groupby(cluster_col, sort=True):
        n_total = len(group)
        n_bt = int(is_bt.loc[group.index].sum())
        bt_share = (n_bt / n_total * 100.0) if n_total > 0 else 0.0
        rows.append(
            {
                "cluster": cluster_id,
                "n_total": n_total,
                "n_below_threshold": n_bt,
                "bt_share_pct": round(bt_share, 4),
            }
        )

    result = pd.DataFrame(rows)
    logger.info(
        "below_threshold_counts: %d clusters summarised; "
        "total below-threshold agents = %d",
        len(result),
        int(result["n_below_threshold"].sum()),
    )
    return result


def compare_cluster_kpis(
    df: pd.DataFrame,
    good_clusters: list[str],
    suspect_clusters: list[str],
    kpi_cols: list[str] | None = None,
    commission_decision_col: str = "CommissionDecision",
    whitelist_label: str = "whitelist",
) -> pd.DataFrame:
    """
    Side-by-side mean/median KPI comparison for good vs suspect whitelist agents.

    Parameters
    ----------
    df : Agent-level DataFrame (should contain CommissionDecision column).
    good_clusters : Ensemble cluster labels considered 'good' (high-performing whitelist).
    suspect_clusters : Ensemble cluster labels to investigate.
    kpi_cols : KPI columns to compare. Defaults to KPI_COLUMNS_V1.
    commission_decision_col : Column name for whitelist/blacklist label.
    whitelist_label : Value in commission_decision_col indicating whitelist agents.

    Returns
    -------
    DataFrame: multi-level columns (Good_whitelist, Suspect_whitelist) × (mean, median, std)
    """
    _require_columns(df, [commission_decision_col], "compare_cluster_kpis")

    # Resolve cluster column — default ensemble_cluster if present
    cluster_col = "ensemble_cluster"
    _require_columns(df, [cluster_col], "compare_cluster_kpis")

    resolved_kpi_cols = _resolve_kpi_cols(df, kpi_cols, "compare_cluster_kpis")

    if not good_clusters:
        raise ValueError("compare_cluster_kpis: good_clusters must not be empty.")
    if not suspect_clusters:
        raise ValueError("compare_cluster_kpis: suspect_clusters must not be empty.")

    is_whitelist = df[commission_decision_col] == whitelist_label

    good_mask = is_whitelist & df[cluster_col].isin(good_clusters)
    suspect_mask = is_whitelist & df[cluster_col].isin(suspect_clusters)

    good_df = df.loc[good_mask, resolved_kpi_cols]
    suspect_df = df.loc[suspect_mask, resolved_kpi_cols]

    logger.info(
        "compare_cluster_kpis: good whitelist n=%d, suspect whitelist n=%d",
        len(good_df),
        len(suspect_df),
    )

    def _summary(frame: pd.DataFrame) -> pd.DataFrame:
        stats = frame.describe().T[["mean", "50%", "std"]].rename(columns={"50%": "median"})
        return stats

    good_stats = _summary(good_df)
    suspect_stats = _summary(suspect_df)

    result = pd.concat(
        {"Good_whitelist": good_stats, "Suspect_whitelist": suspect_stats},
        axis=1,
    )
    return result


def rank_feature_importance(
    df: pd.DataFrame,
    good_clusters: list[str],
    suspect_clusters: list[str],
    kpi_cols: list[str] | None = None,
    commission_decision_col: str = "CommissionDecision",
    whitelist_label: str = "whitelist",
    config: dict | None = None,
) -> pd.Series:
    """
    RandomForest feature importance for distinguishing good vs suspect whitelist agents.

    Trains a binary classifier (1=good whitelist, 0=suspect whitelist) and returns
    feature importances sorted descending.

    Parameters
    ----------
    df : Agent-level DataFrame.
    good_clusters : Cluster labels for 'good' whitelist agents.
    suspect_clusters : Cluster labels for suspect whitelist agents.
    kpi_cols : Feature columns. Defaults to KPI_COLUMNS_V1.
    commission_decision_col : Column for whitelist/blacklist decision.
    whitelist_label : Value identifying whitelist agents.
    config : Optional dict with "random_state" (default 42), "n_estimators" (default 300).

    Returns
    -------
    pd.Series : Feature importances (0..1), sorted descending, indexed by feature name.
    """
    _require_columns(df, [commission_decision_col], "rank_feature_importance")

    cluster_col = "ensemble_cluster"
    _require_columns(df, [cluster_col], "rank_feature_importance")

    resolved_kpi_cols = _resolve_kpi_cols(df, kpi_cols, "rank_feature_importance")

    if not good_clusters:
        raise ValueError("rank_feature_importance: good_clusters must not be empty.")
    if not suspect_clusters:
        raise ValueError("rank_feature_importance: suspect_clusters must not be empty.")

    cfg = config or {}
    n_estimators: int = int(cfg.get("n_estimators", _RF_DEFAULT_N_ESTIMATORS))
    random_state: int = int(cfg.get("random_state", _RF_DEFAULT_RANDOM_STATE))

    is_whitelist = df[commission_decision_col] == whitelist_label

    good_mask = is_whitelist & df[cluster_col].isin(good_clusters)
    suspect_mask = is_whitelist & df[cluster_col].isin(suspect_clusters)

    good_df = df.loc[good_mask, resolved_kpi_cols].copy()
    suspect_df = df.loc[suspect_mask, resolved_kpi_cols].copy()

    good_df["_target"] = 1
    suspect_df["_target"] = 0

    combined = pd.concat([good_df, suspect_df], axis=0, ignore_index=True)
    combined[resolved_kpi_cols] = combined[resolved_kpi_cols].fillna(0)

    X = combined[resolved_kpi_cols].values
    y = combined["_target"].values

    logger.info(
        "rank_feature_importance: training RandomForest(n_estimators=%d, "
        "random_state=%d) on %d samples (%d good, %d suspect)",
        n_estimators,
        random_state,
        len(combined),
        int(good_mask.sum()),
        int(suspect_mask.sum()),
    )

    if len(np.unique(y)) < 2:
        raise ValueError(
            "rank_feature_importance: both class labels must be present. "
            "Check that good_clusters and suspect_clusters each match agents in df."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=_RF_DEFAULT_TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    logger.info(
        "rank_feature_importance: test accuracy = %.4f (n_test=%d)",
        test_acc,
        len(y_test),
    )

    importances = pd.Series(
        clf.feature_importances_,
        index=resolved_kpi_cols,
        name="feature_importance",
    ).sort_values(ascending=False)

    return importances


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY GATE
#
# The functions above (purity, ARI, KPI comparison, feature importance) are a
# validation toolkit, but until this section was added nothing in
# run_extrafloat_segmentation.py ever called them — a degenerate clustering
# run (collapsed GMM components, HDBSCAN mostly noise, poor purity against
# known agent categories) flowed straight through to output with no
# automated check before those tiers reached downstream consumers.
# `run_quality_gate` is the check meant to be wired into the main
# orchestration after clustering + profiling.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_QUALITY_GATE_CONFIG: dict[str, Any] = {
    "enabled": True,
    # When True, a breached check raises ValueError instead of only logging
    # CRITICAL. Defaults to False so turning the gate on doesn't immediately
    # break existing production runs; tighten once thresholds are calibrated
    # against real data.
    "fail_on_breach": False,
    # ── Unsupervised checks (always run, no ground truth required) ───────────
    "min_distinct_segments": 2,
    "max_hdb_noise_pct": 0.50,
    "max_active_below_threshold_pct": 0.50,
    # ── Supervised checks (only run when label_col is present in the data) ───
    "min_purity": 0.0,
    "min_ari": 0.0,
}


def _get_quality_gate_config(config: dict | None) -> dict[str, Any]:
    """Return deep-merged quality-gate config, filling missing keys from defaults."""
    merged = deepcopy(DEFAULT_QUALITY_GATE_CONFIG)
    if config:
        merged.update(config)
    return merged


def run_quality_gate(
    df: pd.DataFrame,
    config: dict | None = None,
    segment_col: str = "segment",
    cluster_col: str = "ensemble_cluster",
    label_col: str = "agent_category",
    active_mask: pd.Series | None = None,
    hdb_tier_col: str = "hdb_tier",
) -> dict[str, Any]:
    """
    Automated post-clustering sanity/quality gate.

    Always runs unsupervised sanity checks (segment diversity, HDBSCAN
    noise/unavailable share, "Below Threshold" share among *active* agents
    specifically — dormant agents are expected to land there, so mixing them
    in would make this check meaningless). When *label_col* (ground-truth
    agent category, e.g. from a whitelist/blacklist merge) is present, also
    runs supervised purity and ARI checks via `cluster_purity_table` /
    `compute_adjusted_rand_score`.

    This function only evaluates; it never raises unless
    ``config["fail_on_breach"]`` is True. Intended to be called from
    `run_extrafloat_segmentation` after clustering + profiling and before
    output trimming (so diagnostic columns like `diag_hdb_tier` /
    `diag_ensemble_cluster` are still present when diagnostics ran), so a
    degenerate run can be caught automatically instead of relying on
    someone manually running this module after the fact.

    Parameters
    ----------
    df : Segmentation output DataFrame (post capacity scoring / optional
         diagnostic clustering).
    config : Quality-gate config. See DEFAULT_QUALITY_GATE_CONFIG.
    segment_col : Business segment column.
    cluster_col : Ensemble cluster column (used for purity/ARI).
    label_col : Ground-truth reference label column, if available.
    active_mask : Boolean mask of non-dormant agents. Defaults to
                  `df["cluster_round2"].notna()` when that column is present,
                  else all agents are treated as active. Callers with a
                  dormant mask computed elsewhere (e.g.
                  `extrafloat_segmentation_pipeline._identify_dormant_mask`)
                  should pass it explicitly.
    hdb_tier_col : Column holding HDBSCAN diagnostic tier labels, used for
                  the noise/unavailable-share check. Only evaluated when
                  present in *df* (diagnostic clustering is optional).

    Returns
    -------
    dict with keys:
        passed : bool — True iff every evaluated check passed.
        checks : dict[str, dict] — per-check {value, threshold, passed}.
        has_ground_truth : bool — whether supervised checks ran.
        skipped : bool — True if the gate was disabled via config.

    Raises
    ------
    ValueError
        If ``config["fail_on_breach"]`` is True and ``passed`` is False.
    """
    cfg = _get_quality_gate_config(config)

    if not cfg.get("enabled", True):
        logger.info("run_quality_gate: disabled via config — skipping.")
        return {"passed": True, "checks": {}, "has_ground_truth": False, "skipped": True}

    _require_columns(df, [segment_col], "run_quality_gate")

    if active_mask is None:
        if "cluster_round2" in df.columns:
            active_mask = df["cluster_round2"].notna()
        else:
            active_mask = pd.Series(True, index=df.index)
    active = df.loc[active_mask]

    checks: dict[str, Any] = {}

    # ── Unsupervised checks ───────────────────────────────────────────────
    n_distinct = int(df[segment_col].nunique(dropna=True))
    checks["n_distinct_segments"] = {
        "value": n_distinct,
        "threshold": cfg["min_distinct_segments"],
        "passed": n_distinct >= cfg["min_distinct_segments"],
    }

    if hdb_tier_col in df.columns and len(active) > 0:
        noise_mask = active[hdb_tier_col].isin(["Noise / Irregular", "Unavailable"])
        noise_pct = float(noise_mask.mean())
        checks["hdb_noise_or_unavailable_pct"] = {
            "value": noise_pct,
            "threshold": cfg["max_hdb_noise_pct"],
            "passed": noise_pct <= cfg["max_hdb_noise_pct"],
        }

    active_bt_pct = float((active[segment_col] == "Below Threshold").mean()) if len(active) > 0 else 0.0
    checks["active_below_threshold_pct"] = {
        "value": active_bt_pct,
        "threshold": cfg["max_active_below_threshold_pct"],
        "passed": active_bt_pct <= cfg["max_active_below_threshold_pct"],
    }

    # ── Supervised checks (only when ground truth is present) ────────────────
    has_ground_truth = label_col in df.columns and bool(df[label_col].notna().any())
    if has_ground_truth and cluster_col in df.columns:
        try:
            purity_table = cluster_purity_table(df, cluster_col=cluster_col, label_col=label_col)
            overall_purity = float(purity_table["overall_purity"].iloc[0]) if len(purity_table) else 0.0
            checks["overall_purity"] = {
                "value": overall_purity,
                "threshold": cfg["min_purity"],
                "passed": overall_purity >= cfg["min_purity"],
            }
            ari = compute_adjusted_rand_score(df, cluster_col=cluster_col, label_col=label_col)
            checks["ari"] = {
                "value": ari,
                "threshold": cfg["min_ari"],
                "passed": ari >= cfg["min_ari"],
            }
        except ValueError as exc:
            logger.warning("run_quality_gate: skipping supervised checks — %s", exc)

    passed = all(bool(c["passed"]) for c in checks.values())

    for name, result in checks.items():
        log_fn = logger.info if result["passed"] else logger.critical
        log_fn(
            "run_quality_gate: %-32s value=%.4f threshold=%.4f -> %s",
            name,
            float(result["value"]),
            float(result["threshold"]),
            "PASS" if result["passed"] else "FAIL",
        )

    report = {
        "passed": passed,
        "checks": checks,
        "has_ground_truth": has_ground_truth,
        "skipped": False,
    }

    if not passed and cfg.get("fail_on_breach"):
        failed = [name for name, c in checks.items() if not c["passed"]]
        raise ValueError(
            f"run_quality_gate: quality gate FAILED (fail_on_breach=True). "
            f"Breached checks: {failed}. Full report: {report}"
        )

    return report

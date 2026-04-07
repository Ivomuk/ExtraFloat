"""
extrafloat_segmentation_pipeline.py
=====================================
Two-stage KMeans -> GMM+HDBSCAN ensemble clustering pipeline for
Uganda MTN MoMo agent segmentation.

Refactors cl_file3.txt + cl_file4.txt + cl_file5.txt.

Market: Uganda (UG) — MTN Mobile Money.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

try:
    import hdbscan as _hdbscan_module
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DORMANT_FILL_LABEL: str = "Dormant_Cluster_kmeans1"
ENSEMBLE_MISSING_LABEL: str = "ENSEMBLE_MISSING"
HDBSCAN_NOISE_LABEL: int = -1

BUSINESS_SEGMENTS: tuple[str, ...] = (
    "Below Threshold",
    "New Bronze",
    "Bronze",
    "Silver",
    "Gold",
    "Platinum",
    "Titanium",
    "Diamond",
)

# HDB tier names ordered from lowest to highest activity (data-driven assignment)
_HDB_TIER_NAMES: list[str] = [
    "Emerging / Low Activity",
    "Bronze Active",
    "Developing Active",
    "Silver Strong",
    "Gold Power",
    "Platinum Power",
]

# Profiling columns used for composite scoring (must exist in features_df)
PROFILING_COLS: tuple[str, ...] = (
    "commission",
    "cash_out_value_1m",
    "cash_out_value_3m",
    "cash_in_value_1m",
    "cash_in_value_6m",
    "payment_value_1m",
    "payment_value_3m",
    "cash_out_vol_1m",
    "cash_out_vol_3m",
    "commission_per_value_3m",
    "commission_per_value_6m",
    "tenure_years",
)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CLUSTERING_CONFIG: dict[str, Any] = {
    "random_state": 42,
    "kmeans_round1_k": 6,
    "kmeans_round2_k": 6,
    "dormant_col": "cash_out_vol_1m",
    "dormant_threshold": 0.0,
    "gmm_min_k": 2,
    "gmm_max_k": 12,
    "gmm_covariance_type": "full",
    "hdbscan_min_cluster_size": 1000,
    "hdbscan_min_samples": 150,
    "hdbscan_metric": "euclidean",
    "sample_cap_silhouette": 20000,
    "target_pca_variance": 0.90,
    "composite_weights": {
        "value": 0.5,
        "activity": 0.3,
        "efficiency": 0.2,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _get_clustering_config(config: dict | None) -> dict[str, Any]:
    """Return deep-merged clustering config, filling missing keys from defaults."""
    if config is None:
        return deepcopy(DEFAULT_CLUSTERING_CONFIG)
    merged = deepcopy(DEFAULT_CLUSTERING_CONFIG)
    for k, v in config.items():
        if k == "composite_weights" and isinstance(v, dict):
            merged["composite_weights"].update(v)
        else:
            merged[k] = v
    return merged


def _run_kmeans_round1(
    X_pca: np.ndarray, cfg: dict[str, Any], rng: np.random.RandomState
) -> np.ndarray:
    """Fit KMeans on all agents (Round 1).

    Parameters
    ----------
    X_pca : PCA-reduced feature array (all agents).
    cfg   : Clustering config.
    rng   : Random state.

    Returns
    -------
    Integer cluster label array of length n_agents.
    """
    k = cfg["kmeans_round1_k"]
    logger.info("_run_kmeans_round1: fitting KMeans k=%d on %d agents", k, len(X_pca))
    km = KMeans(
        n_clusters=k,
        random_state=rng.randint(0, 2**31),
        n_init=20,
        max_iter=300,
    )
    labels = km.fit_predict(X_pca)
    for cluster_id, count in zip(*np.unique(labels, return_counts=True)):
        logger.debug("_run_kmeans_round1: cluster %d -> %d agents", cluster_id, count)
    return labels


def _identify_dormant_mask(
    features_df: pd.DataFrame, cfg: dict[str, Any]
) -> pd.Series:
    """Identify dormant agents using a configurable activity column threshold.

    Parameters
    ----------
    features_df : Agent feature DataFrame.
    cfg         : Clustering config.

    Returns
    -------
    Boolean Series (True = dormant).
    """
    col = cfg["dormant_col"]
    threshold = cfg["dormant_threshold"]

    if col not in features_df.columns:
        logger.warning(
            "_identify_dormant_mask: dormant_col '%s' not found — treating all agents as active.",
            col,
        )
        return pd.Series(False, index=features_df.index)

    mask = features_df[col].fillna(0.0) <= threshold
    logger.info(
        "_identify_dormant_mask: %d dormant agents (%.1f%%) identified via %s <= %s",
        int(mask.sum()),
        mask.mean() * 100,
        col,
        threshold,
    )
    return mask


def _run_kmeans_round2(
    features_df: pd.DataFrame,
    active_mask: pd.Series,
    selected_cols: list[str],
    cfg: dict[str, Any],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Fit KMeans on active agents with fresh scaling + PCA (Round 2).

    Parameters
    ----------
    features_df  : Full agent feature DataFrame.
    active_mask  : Boolean mask of active (non-dormant) agents.
    selected_cols: Feature columns for re-scaling.
    cfg          : Clustering config.
    rng          : Random state.

    Returns
    -------
    Integer cluster label array of length active_mask.sum().
    """
    k = cfg["kmeans_round2_k"]
    target_var = cfg["target_pca_variance"]
    rs = rng.randint(0, 2**31)

    valid_cols = [c for c in selected_cols if c in features_df.columns]
    X_active = features_df.loc[active_mask, valid_cols].fillna(0.0).astype(float).values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_active)

    pca_full = PCA(random_state=rs)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cum_var >= target_var) + 1)

    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=rs)
    X_active_pca = pca.fit_transform(X_scaled)

    logger.info(
        "_run_kmeans_round2: k=%d, n_active=%d, n_components=%d",
        k,
        len(X_active),
        n_comp,
    )
    km = KMeans(n_clusters=k, random_state=rs, n_init=20, max_iter=300)
    labels = km.fit_predict(X_active_pca)
    return labels


def _get_active_pca(
    features_df: pd.DataFrame,
    active_mask: pd.Series,
    selected_cols: list[str],
    cfg: dict[str, Any],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Produce shared PCA-reduced feature array for GMM and HDBSCAN stages.

    Parameters
    ----------
    features_df  : Full agent feature DataFrame.
    active_mask  : Boolean mask of active agents.
    selected_cols: Feature columns for scaling.
    cfg          : Clustering config.
    rng          : Random state.

    Returns
    -------
    PCA-reduced array of shape (n_active, n_components).
    """
    target_var = cfg["target_pca_variance"]
    rs = rng.randint(0, 2**31)

    valid_cols = [c for c in selected_cols if c in features_df.columns]
    X_active = features_df.loc[active_mask, valid_cols].fillna(0.0).astype(float).values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_active)

    pca_full = PCA(random_state=rs)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cum_var >= target_var) + 1)

    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=rs)
    X_pca_active = pca.fit_transform(X_scaled)

    logger.info(
        "_get_active_pca: n_active=%d, n_components=%d (%.0f%% variance)",
        len(X_active),
        n_comp,
        target_var * 100,
    )
    return X_pca_active


def _run_gmm(
    X_pca_active: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.RandomState,
) -> tuple[np.ndarray, GaussianMixture]:
    """BIC-based GMM search then refit with best k.

    Parameters
    ----------
    X_pca_active : PCA-reduced active-agent array.
    cfg          : Clustering config.
    rng          : Random state.

    Returns
    -------
    (labels, fitted_gmm)
    """
    rs = rng.randint(0, 2**31)
    min_k = cfg["gmm_min_k"]
    max_k = cfg["gmm_max_k"]
    cov_type = cfg["gmm_covariance_type"]

    logger.info(
        "_run_gmm: BIC search k=%d..%d on %d active agents",
        min_k,
        max_k,
        len(X_pca_active),
    )

    bic_scores: list[float] = []
    ks = list(range(min_k, max_k + 1))
    for k in ks:
        g = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            random_state=rs,
            max_iter=200,
        )
        g.fit(X_pca_active)
        bic_scores.append(g.bic(X_pca_active))

    best_k = ks[int(np.argmin(bic_scores))]
    logger.info("_run_gmm: best_k=%d (min BIC=%.1f)", best_k, min(bic_scores))

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type=cov_type,
        random_state=rs,
        max_iter=300,
    )
    labels = gmm.fit_predict(X_pca_active)
    for cid, cnt in zip(*np.unique(labels, return_counts=True)):
        logger.debug("_run_gmm: cluster %d -> %d agents", cid, cnt)
    return labels, gmm


def _run_hdbscan(
    X_pca_active: np.ndarray, cfg: dict[str, Any]
) -> np.ndarray:
    """Fit HDBSCAN on active agents.

    Parameters
    ----------
    X_pca_active : PCA-reduced active-agent array.
    cfg          : Clustering config.

    Returns
    -------
    Integer label array (-1 = noise).

    Raises
    ------
    ImportError if hdbscan is not installed.
    """
    if not _HDBSCAN_AVAILABLE:
        raise ImportError(
            "HDBSCAN is not installed. Install with: pip install hdbscan"
        )

    logger.info(
        "_run_hdbscan: min_cluster_size=%d, min_samples=%d on %d agents",
        cfg["hdbscan_min_cluster_size"],
        cfg["hdbscan_min_samples"],
        len(X_pca_active),
    )
    model = _hdbscan_module.HDBSCAN(
        min_cluster_size=cfg["hdbscan_min_cluster_size"],
        min_samples=cfg["hdbscan_min_samples"],
        metric=cfg["hdbscan_metric"],
        cluster_selection_epsilon=0.0,
    )
    labels = model.fit_predict(X_pca_active)
    noise_pct = (labels == HDBSCAN_NOISE_LABEL).mean() * 100
    n_clusters = len(set(labels) - {HDBSCAN_NOISE_LABEL})
    logger.info(
        "_run_hdbscan: %d clusters, %.1f%% noise", n_clusters, noise_pct
    )
    return labels


def _map_hdb_to_tier(
    hdb_labels_active: np.ndarray,
    X_pca_active: np.ndarray,
    cfg: dict[str, Any],  # noqa: ARG001
) -> pd.Series:
    """Assign data-driven tier names to HDBSCAN cluster IDs.

    Ranks non-noise clusters by their mean PC1 score (ascending) and assigns
    tier names evenly from _HDB_TIER_NAMES (lowest -> highest activity).
    Noise label (-1) -> "Noise / Irregular".

    Parameters
    ----------
    hdb_labels_active : HDBSCAN label array for active agents.
    X_pca_active      : Corresponding PCA array (used for mean PC1 ranking).
    cfg               : Clustering config (reserved for future tuning).

    Returns
    -------
    pd.Series of tier name strings, same length as hdb_labels_active.
    """
    unique_clusters = sorted(
        c for c in np.unique(hdb_labels_active) if c != HDBSCAN_NOISE_LABEL
    )

    if not unique_clusters:
        logger.warning(
            "_map_hdb_to_tier: no non-noise HDBSCAN clusters found."
        )
        return pd.Series(
            ["Noise / Irregular"] * len(hdb_labels_active), dtype="object"
        )

    # Sort clusters by mean PC1 ascending (low activity -> high activity)
    pc1 = X_pca_active[:, 0]
    cluster_means = {
        cid: float(pc1[hdb_labels_active == cid].mean())
        for cid in unique_clusters
    }
    sorted_clusters = sorted(unique_clusters, key=lambda c: cluster_means[c])

    # Evenly distribute tier names across clusters
    tier_groups = np.array_split(sorted_clusters, min(len(_HDB_TIER_NAMES), len(sorted_clusters)))
    cluster_to_tier: dict[int, str] = {}
    tier_list = _HDB_TIER_NAMES[-len(tier_groups):]  # use highest tiers if fewer clusters
    for tier_name, group in zip(tier_list, tier_groups):
        for cid in group:
            cluster_to_tier[cid] = tier_name

    logger.info(
        "_map_hdb_to_tier: mapping %d clusters -> %d tiers (data-driven)",
        len(unique_clusters),
        len(set(cluster_to_tier.values())),
    )
    for cid in sorted_clusters:
        logger.debug(
            "_map_hdb_to_tier: cluster %d (mean_pc1=%.3f) -> %s",
            cid,
            cluster_means[cid],
            cluster_to_tier[cid],
        )

    tier_labels = [
        cluster_to_tier.get(int(lbl), "Noise / Irregular")
        if lbl != HDBSCAN_NOISE_LABEL
        else "Noise / Irregular"
        for lbl in hdb_labels_active
    ]
    return pd.Series(tier_labels, dtype="object")


def _build_ensemble_labels(
    gmm_labels: pd.Series,
    hdb_tier_col: pd.Series,
) -> pd.Series:
    """Combine GMM ID and HDB tier into "GMM_{id}__{tier}" string labels.

    Parameters
    ----------
    gmm_labels   : Integer GMM cluster IDs (Int64 series).
    hdb_tier_col : String HDB tier labels.

    Returns
    -------
    pd.Series of ensemble label strings.
    """
    result = np.where(
        gmm_labels.isna() | hdb_tier_col.isna(),
        ENSEMBLE_MISSING_LABEL,
        "GMM_" + gmm_labels.astype(str) + "__" + hdb_tier_col.astype(str),
    )
    return pd.Series(result, index=gmm_labels.index, dtype="object")


def _compute_composite_score(
    profile_means: pd.DataFrame, cfg: dict[str, Any]
) -> pd.Series:
    """Compute weighted composite score per ensemble cluster from mean profiles.

    Groups:
    - value     : commission + cash_out/in + payment values (sum, normalized)
    - activity  : cash_out/in volumes (sum, normalized)
    - efficiency: commission_per_value + tenure (mean, normalized)

    Parameters
    ----------
    profile_means : DataFrame (index=ensemble_cluster, cols=KPI features).
    cfg           : Clustering config with composite_weights dict.

    Returns
    -------
    pd.Series of composite scores, index=ensemble_cluster.
    """
    weights = cfg.get("composite_weights", {"value": 0.5, "activity": 0.3, "efficiency": 0.2})

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(0.0, index=s.index)
        return (s - lo) / (hi - lo)

    value_cols = [
        c for c in (
            "commission", "cash_out_value_1m", "cash_out_value_3m",
            "cash_in_value_1m", "cash_in_value_6m",
            "payment_value_1m", "payment_value_3m",
        ) if c in profile_means.columns
    ]
    activity_cols = [
        c for c in ("cash_out_vol_1m", "cash_out_vol_3m")
        if c in profile_means.columns
    ]
    efficiency_cols = [
        c for c in ("commission_per_value_3m", "commission_per_value_6m", "tenure_years")
        if c in profile_means.columns
    ]

    value_score = _norm(profile_means[value_cols].sum(axis=1)) if value_cols else pd.Series(0.0, index=profile_means.index)
    activity_score = _norm(profile_means[activity_cols].sum(axis=1)) if activity_cols else pd.Series(0.0, index=profile_means.index)
    efficiency_score = _norm(profile_means[efficiency_cols].mean(axis=1)) if efficiency_cols else pd.Series(0.0, index=profile_means.index)

    composite = (
        float(weights.get("value", 0.5)) * value_score
        + float(weights.get("activity", 0.3)) * activity_score
        + float(weights.get("efficiency", 0.2)) * efficiency_score
    )
    return composite


def _map_ensemble_to_segment(
    ensemble_col: pd.Series,
    features_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    """Assign business segments to agents via data-driven composite score ranking.

    Algorithm:
    1. Compute mean profile per ensemble cluster over PROFILING_COLS.
    2. Score each cluster with a composite score.
    3. Rank clusters by score (ascending).
    4. Assign BUSINESS_SEGMENTS by quantile (evenly distributed).
    5. Dormant cluster (DORMANT_FILL_LABEL) always -> "Below Threshold".

    Parameters
    ----------
    ensemble_col : Series of ensemble cluster labels (aligned with features_df).
    features_df  : Agent feature DataFrame.
    cfg          : Clustering config.

    Returns
    -------
    pd.Series of business segment strings.
    """
    valid_profiling_cols = [c for c in PROFILING_COLS if c in features_df.columns]
    tmp = features_df.copy()
    tmp["__ensemble__"] = ensemble_col.values

    profile_means = tmp.groupby("__ensemble__")[valid_profiling_cols].mean()

    composite = _compute_composite_score(profile_means, cfg)
    sorted_clusters = composite.sort_values(ascending=True).index.tolist()
    n = len(sorted_clusters)

    segment_map: dict[str, str] = {}
    for rank, cluster_label in enumerate(sorted_clusters):
        seg_idx = min(int(rank / n * len(BUSINESS_SEGMENTS)), len(BUSINESS_SEGMENTS) - 1)
        segment_map[cluster_label] = BUSINESS_SEGMENTS[seg_idx]

    # Dormant cluster always -> "Below Threshold"
    segment_map[DORMANT_FILL_LABEL] = BUSINESS_SEGMENTS[0]
    segment_map[ENSEMBLE_MISSING_LABEL] = BUSINESS_SEGMENTS[0]

    logger.info("_map_ensemble_to_segment: cluster -> segment mapping:")
    for cl, seg in sorted(segment_map.items(), key=lambda x: composite.get(x[0], -1)):
        logger.info("  %-45s -> %s", cl, seg)

    return ensemble_col.map(segment_map).fillna(BUSINESS_SEGMENTS[0])


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────


def run_clustering_pipeline(
    features_df: pd.DataFrame,
    X_pca: np.ndarray,
    selected_cols: list[str],
    config: dict | None = None,
) -> pd.DataFrame:
    """Two-stage KMeans -> GMM+HDBSCAN -> ensemble -> business segment mapping.

    Parameters
    ----------
    features_df :
        Agent-level feature DataFrame (index must align with rows of X_pca).
        Must contain the dormant detection column (default: cash_out_vol_1m).
    X_pca :
        PCA-reduced feature array aligned with features_df.
    selected_cols :
        Feature column names for re-scaling active agents in Round 2 / HDBSCAN.
    config :
        Clustering configuration dict. Missing keys fall back to
        DEFAULT_CLUSTERING_CONFIG.

    Returns
    -------
    pd.DataFrame
        features_df with added columns:
            cluster_round1  : Round-1 KMeans label (int, all agents)
            cluster_round2  : Round-2 KMeans label (Int64, NaN for dormant)
            cluster_id_gmm  : GMM label (Int64, -1 for dormant)
            cluster_hdb_raw : Raw HDBSCAN integer label (Int64)
            hdb_tier        : Named HDBSCAN tier string (data-driven)
            ensemble_cluster: Combined "GMM_{id}__{tier}" label
            segment         : Business segment from BUSINESS_SEGMENTS
    """
    cfg = _get_clustering_config(config)
    rng = np.random.RandomState(cfg["random_state"])

    if X_pca.shape[0] != len(features_df):
        raise ValueError(
            f"run_clustering_pipeline: X_pca has {X_pca.shape[0]} rows but "
            f"features_df has {len(features_df)} rows."
        )

    out = features_df.copy()

    logger.info(
        "run_clustering_pipeline: starting on %d agents, X_pca shape=%s",
        len(out),
        X_pca.shape,
    )

    # ── Round 1: KMeans on all agents ────────────────────────────────────────
    labels_r1 = _run_kmeans_round1(X_pca, cfg, rng)
    out["cluster_round1"] = labels_r1

    # ── Dormant detection (data-driven) ──────────────────────────────────────
    dormant_mask = _identify_dormant_mask(out, cfg)
    active_mask = ~dormant_mask

    # ── Round 2: KMeans on active agents ─────────────────────────────────────
    out["cluster_round2"] = pd.array([pd.NA] * len(out), dtype="Int64")
    if active_mask.sum() > 0:
        labels_r2 = _run_kmeans_round2(out, active_mask, selected_cols, cfg, rng)
        out.loc[active_mask, "cluster_round2"] = labels_r2

    # ── Active PCA for GMM + HDBSCAN ─────────────────────────────────────────
    X_pca_active: np.ndarray | None = None
    if active_mask.sum() > 0:
        X_pca_active = _get_active_pca(out, active_mask, selected_cols, cfg, rng)

    # ── GMM on active agents ──────────────────────────────────────────────────
    out["cluster_id_gmm"] = pd.array([-1] * len(out), dtype="Int64")
    if X_pca_active is not None:
        gmm_labels, _ = _run_gmm(X_pca_active, cfg, rng)
        active_indices = out.index[active_mask]
        out.loc[active_indices, "cluster_id_gmm"] = gmm_labels

    # ── HDBSCAN on active agents ──────────────────────────────────────────────
    out["cluster_hdb_raw"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["hdb_tier"] = pd.NA

    if X_pca_active is not None:
        try:
            hdb_labels = _run_hdbscan(X_pca_active, cfg)
            active_indices = out.index[active_mask]
            out.loc[active_indices, "cluster_hdb_raw"] = hdb_labels
            hdb_tier_active = _map_hdb_to_tier(hdb_labels, X_pca_active, cfg)
            out.loc[active_indices, "hdb_tier"] = hdb_tier_active.values
        except ImportError as exc:
            logger.error(
                "run_clustering_pipeline: HDBSCAN unavailable — %s. "
                "Skipping HDBSCAN stage; hdb_tier will be 'Unavailable'.",
                exc,
            )
            out.loc[active_mask, "hdb_tier"] = "Unavailable"

    # ── Ensemble labels ───────────────────────────────────────────────────────
    out["ensemble_cluster"] = DORMANT_FILL_LABEL

    gmm_active = out.loc[active_mask, "cluster_id_gmm"]
    hdb_active = out.loc[active_mask, "hdb_tier"]

    if active_mask.sum() > 0:
        ensemble_active = _build_ensemble_labels(gmm_active, hdb_active)
        # Replace any ENSEMBLE_MISSING with dormant fill label
        ensemble_active = ensemble_active.replace(
            {ENSEMBLE_MISSING_LABEL: DORMANT_FILL_LABEL}
        )
        out.loc[active_mask, "ensemble_cluster"] = ensemble_active.values

    # ── Business segment mapping (data-driven) ────────────────────────────────
    out["segment"] = _map_ensemble_to_segment(out["ensemble_cluster"], out, cfg)

    # Dormant agents always -> "Below Threshold"
    out.loc[dormant_mask, "segment"] = BUSINESS_SEGMENTS[0]

    # Log summary
    seg_counts = out["segment"].value_counts().to_dict()
    logger.info("run_clustering_pipeline: segment distribution — %s", seg_counts)

    return out

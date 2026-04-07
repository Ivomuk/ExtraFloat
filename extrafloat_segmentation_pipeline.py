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

try:
    import umap as _umap_module
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

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
    # ── Dormant detection (multi-product composite inactivity score) ──────────
    "dormant_inactivity_cols": [
        "cash_out_vol_1m",
        "cash_in_vol_1m",
        "payment_vol_1m",
        "voucher_volume_1m",
    ],
    "dormant_inactivity_weights": [0.5, 0.25, 0.15, 0.10],
    "dormant_composite_threshold": 0.05,  # normalized score ≤ this → dormant
    # ── GMM ───────────────────────────────────────────────────────────────────
    "gmm_min_k": 2,
    "gmm_max_k": 12,
    "gmm_covariance_type": "full",
    # Regularisation added to the GMM covariance diagonal to prevent singular
    # matrices when clusters collapse (ill-conditioned data / too many components).
    # Increase to 1e-3 or 1e-2 if you still see LinAlgError on your dataset.
    "gmm_reg_covar": 1e-4,
    # ── HDBSCAN ───────────────────────────────────────────────────────────────
    "hdbscan_min_cluster_size": 1000,
    "hdbscan_min_samples": 150,
    "hdbscan_metric": "euclidean",
    # ── UMAP (used as input to HDBSCAN when umap-learn is installed) ──────────
    "use_umap_for_hdbscan": True,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "umap_n_components": 2,
    # ── HDB tier ranking: use first available col as business KPI centroid ────
    "hdb_tier_ranking_cols": ["total_value_1m", "cash_out_value_1m", "commission"],
    # ── PCA variance target ───────────────────────────────────────────────────
    "target_pca_variance": 0.90,
    # ── Composite score weights ───────────────────────────────────────────────
    "composite_weights": {
        "value": 0.5,
        "activity": 0.3,
        "efficiency": 0.2,
    },
    # ── ARI-based weight optimisation (opt-in; requires agent_category col) ──
    "optimize_composite_weights": False,
    "weight_grid_step": 0.1,
    # ── Cluster stability reporting ───────────────────────────────────────────
    "stability_n_seeds": 3,
    "sample_cap_silhouette": 20000,
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
    """Identify dormant agents via a multi-product composite inactivity score.

    For each configured inactivity column the column is normalised by its 95th
    percentile (clipped to [0, 1]) and then the weighted sum is computed.
    Agents whose composite score is at or below *dormant_composite_threshold*
    are classified as dormant.

    Parameters
    ----------
    features_df : Agent feature DataFrame.
    cfg         : Clustering config.

    Returns
    -------
    Boolean Series (True = dormant).
    """
    inactivity_cols: list[str] = cfg.get(
        "dormant_inactivity_cols",
        ["cash_out_vol_1m", "cash_in_vol_1m", "payment_vol_1m", "voucher_volume_1m"],
    )
    raw_weights: list[float] = cfg.get(
        "dormant_inactivity_weights", [0.5, 0.25, 0.15, 0.10]
    )
    threshold: float = float(cfg.get("dormant_composite_threshold", 0.05))

    # Filter to columns that actually exist in the DataFrame
    present = [(col, w) for col, w in zip(inactivity_cols, raw_weights) if col in features_df.columns]

    if not present:
        logger.warning(
            "_identify_dormant_mask: none of %s found in features_df — "
            "treating all agents as active.",
            inactivity_cols,
        )
        return pd.Series(False, index=features_df.index)

    present_cols, present_weights = zip(*present)
    weight_sum = sum(present_weights)
    norm_weights = [w / weight_sum for w in present_weights]

    # Build composite score: weighted sum of per-column normalised activity
    composite = pd.Series(0.0, index=features_df.index)
    for col, w in zip(present_cols, norm_weights):
        series = features_df[col].fillna(0.0).clip(lower=0.0)
        p95 = series.quantile(0.95)
        normalised = (series / p95).clip(upper=1.0) if p95 > 0 else pd.Series(0.0, index=series.index)
        composite += w * normalised

    mask = composite <= threshold
    logger.info(
        "_identify_dormant_mask: %d dormant agents (%.1f%%) via composite inactivity "
        "score (cols=%s, threshold=%.3f)",
        int(mask.sum()),
        mask.mean() * 100,
        list(present_cols),
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
) -> tuple[np.ndarray, np.ndarray]:
    """Produce shared scaled + PCA-reduced feature arrays for GMM/HDBSCAN stages.

    Parameters
    ----------
    features_df  : Full agent feature DataFrame.
    active_mask  : Boolean mask of active agents.
    selected_cols: Feature columns for scaling.
    cfg          : Clustering config.
    rng          : Random state.

    Returns
    -------
    (X_pca_active, X_scaled_active) — both of shape (n_active, n_components/n_features).
    X_scaled_active is returned so UMAP can receive the full scaled space.
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
    return X_pca_active, X_scaled


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
    reg_covar = float(cfg.get("gmm_reg_covar", 1e-4))

    # Cap max_k: need at least 10 samples per component to avoid degenerate
    # clusters during EM initialisation.  With fewer samples, components
    # collapse to singletons and the covariance becomes singular no matter
    # how large reg_covar is.
    n_active = len(X_pca_active)
    safe_max_k = max(int(min_k), min(int(max_k), n_active // 10))
    if safe_max_k < max_k:
        logger.warning(
            "_run_gmm: capping gmm_max_k from %d to %d "
            "(need >= 10 agents per component, n_active=%d).",
            max_k, safe_max_k, n_active,
        )
    max_k = safe_max_k

    # Covariance type fallback chain: full → diag → spherical.
    # "full" can fail when components receive too few samples even after capping k.
    # "diag" is numerically stable and appropriate for high-dimensional PCA spaces.
    _COV_FALLBACK = ["full", "diag", "spherical"]
    cov_candidates = [cov_type] + [c for c in _COV_FALLBACK if c != cov_type]

    def _bic_search(cov: str) -> tuple[list[float], list[int]]:
        scores, valid_ks = [], []
        for k in range(min_k, max_k + 1):
            g = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                reg_covar=reg_covar,
                random_state=rs,
                max_iter=200,
            )
            try:
                g.fit(X_pca_active)
                scores.append(g.bic(X_pca_active))
                valid_ks.append(k)
            except ValueError:
                logger.warning(
                    "_run_gmm: k=%d cov=%s failed (singular covariance) — skipping.", k, cov
                )
        return scores, valid_ks

    logger.info(
        "_run_gmm: BIC search k=%d..%d on %d active agents "
        "(cov=%s, reg_covar=%.2e)",
        min_k, max_k, n_active, cov_type, reg_covar,
    )

    chosen_cov = cov_type
    bic_scores, valid_ks = _bic_search(cov_type)

    if not valid_ks:
        for fallback_cov in cov_candidates[1:]:
            logger.warning(
                "_run_gmm: all fits failed with cov='%s' — retrying with cov='%s'.",
                chosen_cov, fallback_cov,
            )
            bic_scores, valid_ks = _bic_search(fallback_cov)
            if valid_ks:
                chosen_cov = fallback_cov
                break

    if not valid_ks:
        raise ValueError(
            "_run_gmm: GMM fitting failed for all covariance types and k values. "
            "Check your data for duplicate rows or near-zero-variance features. "
            "You can also try config={'clustering': {'gmm_min_k': 2, 'gmm_max_k': 4}}."
        )

    best_k = valid_ks[int(np.argmin(bic_scores))]
    logger.info(
        "_run_gmm: best_k=%d cov='%s' (BIC=%.1f)", best_k, chosen_cov, min(bic_scores)
    )

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type=chosen_cov,
        reg_covar=reg_covar,
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


def _get_active_umap(
    X_scaled_active: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Reduce scaled active-agent features to a UMAP embedding for HDBSCAN.

    UMAP preserves local density structure (unlike PCA) so HDBSCAN produces
    more meaningful clusters.  If *umap-learn* is not installed the function
    falls back to using the first *umap_n_components* columns of
    *X_scaled_active* (a PCA-like linear slice) and emits a warning.

    Parameters
    ----------
    X_scaled_active : RobustScaler-scaled active-agent array (n_active × n_features).
    cfg             : Clustering config.
    rng             : Random state.

    Returns
    -------
    Embedding of shape (n_active, umap_n_components).
    """
    n_components: int = int(cfg.get("umap_n_components", 2))

    if not _UMAP_AVAILABLE:
        logger.warning(
            "_get_active_umap: umap-learn is not installed — falling back to "
            "first %d scaled columns for HDBSCAN input. "
            "Install with: pip install umap-learn",
            n_components,
        )
        return X_scaled_active[:, :n_components]

    reducer = _umap_module.UMAP(
        n_neighbors=int(cfg.get("umap_n_neighbors", 15)),
        min_dist=float(cfg.get("umap_min_dist", 0.1)),
        n_components=n_components,
        random_state=int(rng.randint(0, 2**31)),
        metric="euclidean",
    )
    embedding = reducer.fit_transform(X_scaled_active)
    logger.info(
        "_get_active_umap: UMAP embedding shape=%s (n_neighbors=%d, min_dist=%.2f)",
        embedding.shape,
        cfg.get("umap_n_neighbors", 15),
        cfg.get("umap_min_dist", 0.1),
    )
    return embedding


def _map_hdb_to_tier(
    hdb_labels_active: np.ndarray,
    features_df_active: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    """Assign data-driven tier names to HDBSCAN cluster IDs.

    Ranks non-noise clusters by their mean value of the first available
    business KPI column from *hdb_tier_ranking_cols* (default: total_value_1m
    → cash_out_value_1m → commission).  Using a business KPI centroid rather
    than PC1 ensures tier order reflects commercial value, not variance.
    Noise label (-1) -> "Noise / Irregular".

    Parameters
    ----------
    hdb_labels_active  : HDBSCAN label array for active agents.
    features_df_active : Raw feature DataFrame for active agents, aligned with
                         hdb_labels_active (used for KPI centroid ranking).
    cfg                : Clustering config.

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

    # Rank clusters by mean business KPI centroid (ascending = low → high value)
    ranking_cols: list[str] = cfg.get(
        "hdb_tier_ranking_cols",
        ["total_value_1m", "cash_out_value_1m", "commission"],
    )
    available_ranking_cols = [c for c in ranking_cols if c in features_df_active.columns]

    if available_ranking_cols:
        rank_col = available_ranking_cols[0]
        rank_values = features_df_active[rank_col].fillna(0.0).values
        cluster_means = {
            cid: float(rank_values[hdb_labels_active == cid].mean())
            for cid in unique_clusters
        }
        logger.info(
            "_map_hdb_to_tier: ranking %d clusters by '%s' centroid",
            len(unique_clusters),
            rank_col,
        )
    else:
        # Last-resort: warn and skip ranking (assign tiers in discovery order)
        logger.warning(
            "_map_hdb_to_tier: none of %s found in features_df_active — "
            "assigning tiers in cluster-ID order (no business-KPI ranking).",
            ranking_cols,
        )
        cluster_means = {cid: float(cid) for cid in unique_clusters}

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
            "_map_hdb_to_tier: cluster %d (mean_kpi=%.3f) -> %s",
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


def _find_optimal_composite_weights(
    profile_means: pd.DataFrame,
    features_df: pd.DataFrame,
    ensemble_col: pd.Series,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Grid-search composite weight combinations that maximise ARI vs agent_category.

    Searches all (value, activity, efficiency) triples in steps of
    *weight_grid_step* that sum to 1.0.  For each triple the resulting segment
    assignment is compared to ``features_df["agent_category"]`` using ARI.
    Returns the weight triple with the highest ARI; falls back to
    ``cfg["composite_weights"]`` if no improvement is found or if the
    ``agent_category`` column is absent.

    Parameters
    ----------
    profile_means : DataFrame (index=ensemble_cluster, cols=KPI features).
    features_df   : Agent-level feature DataFrame containing ``agent_category``.
    ensemble_col  : Series of ensemble cluster labels aligned with features_df.
    cfg           : Clustering config.

    Returns
    -------
    dict with keys ``value``, ``activity``, ``efficiency``.
    """
    from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415

    default_weights: dict[str, float] = dict(cfg.get("composite_weights", {"value": 0.5, "activity": 0.3, "efficiency": 0.2}))

    if "agent_category" not in features_df.columns:
        logger.info(
            "_find_optimal_composite_weights: agent_category column absent — "
            "using default weights %s",
            default_weights,
        )
        return default_weights

    step: float = float(cfg.get("weight_grid_step", 0.1))
    steps = [round(i * step, 10) for i in range(int(1.0 / step) + 1)]

    true_labels = features_df["agent_category"].astype(str).values

    best_ari: float = -2.0
    best_weights = default_weights

    for v in steps:
        for a in steps:
            e = round(1.0 - v - a, 10)
            if e < 0 or e > 1.0 + 1e-9:
                continue
            trial_cfg = {**cfg, "composite_weights": {"value": v, "activity": a, "efficiency": e}}
            composite = _compute_composite_score(profile_means, trial_cfg)
            sorted_clusters = composite.sort_values(ascending=True).index.tolist()
            n = len(sorted_clusters)
            seg_map: dict[str, str] = {}
            for rank, cl in enumerate(sorted_clusters):
                seg_idx = min(int(rank / n * len(BUSINESS_SEGMENTS)), len(BUSINESS_SEGMENTS) - 1)
                seg_map[cl] = BUSINESS_SEGMENTS[seg_idx]
            seg_map[DORMANT_FILL_LABEL] = BUSINESS_SEGMENTS[0]
            seg_map[ENSEMBLE_MISSING_LABEL] = BUSINESS_SEGMENTS[0]
            pred_labels = ensemble_col.map(seg_map).fillna(BUSINESS_SEGMENTS[0]).values
            try:
                ari = adjusted_rand_score(true_labels, pred_labels)
            except Exception:  # noqa: BLE001
                continue
            if ari > best_ari:
                best_ari = ari
                best_weights = {"value": v, "activity": a, "efficiency": round(e, 10)}

    logger.info(
        "_find_optimal_composite_weights: best weights=%s (ARI=%.4f vs default ARI=%.4f)",
        best_weights,
        best_ari,
        adjusted_rand_score(
            true_labels,
            ensemble_col.map(
                {
                    **{
                        cl: BUSINESS_SEGMENTS[min(int(i / len(profile_means) * len(BUSINESS_SEGMENTS)), len(BUSINESS_SEGMENTS) - 1)]
                        for i, cl in enumerate(_compute_composite_score(profile_means, cfg).sort_values().index)
                    },
                    DORMANT_FILL_LABEL: BUSINESS_SEGMENTS[0],
                    ENSEMBLE_MISSING_LABEL: BUSINESS_SEGMENTS[0],
                }
            ).fillna(BUSINESS_SEGMENTS[0]).values,
        ),
    )
    return best_weights


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

    # Optionally optimise composite weights via ARI against known labels
    if cfg.get("optimize_composite_weights") and "agent_category" in features_df.columns:
        optimised_weights = _find_optimal_composite_weights(
            profile_means, features_df, ensemble_col, cfg
        )
        cfg = {**cfg, "composite_weights": optimised_weights}
        logger.info(
            "_map_ensemble_to_segment: using ARI-optimised weights=%s", optimised_weights
        )

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


DEFAULT_STABILITY_REPORT: dict[str, Any] = {
    "silhouette_score": float("nan"),
    "ari_mean": float("nan"),
    "ari_std": float("nan"),
    "n_seeds": 0,
}


def _compute_stability_metrics(
    X_pca: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate cluster reproducibility across multiple random seeds.

    Runs KMeans (round-1 settings) ``stability_n_seeds`` times with different
    random seeds, then computes:

    - **silhouette_score**: averaged over seeds on a subsample of size
      ``sample_cap_silhouette``.
    - **ari_mean / ari_std**: mean and standard deviation of pairwise
      Adjusted Rand Index between all seed-pair label assignments.

    Parameters
    ----------
    X_pca : PCA-reduced feature array (all agents).
    cfg   : Clustering config.

    Returns
    -------
    dict with keys: ``silhouette_score``, ``ari_mean``, ``ari_std``, ``n_seeds``.
    """
    from itertools import combinations  # noqa: PLC0415
    from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: PLC0415

    n_seeds: int = int(cfg.get("stability_n_seeds", 3))
    k: int = int(cfg.get("kmeans_round1_k", 6))
    cap: int = int(cfg.get("sample_cap_silhouette", 20000))
    base_seed: int = int(cfg.get("random_state", 42))

    if n_seeds < 2:
        logger.warning(
            "_compute_stability_metrics: stability_n_seeds=%d < 2 — "
            "returning default report.",
            n_seeds,
        )
        return deepcopy(DEFAULT_STABILITY_REPORT)

    n = len(X_pca)
    seed_labels: list[np.ndarray] = []
    sil_scores: list[float] = []

    for i in range(n_seeds):
        seed = base_seed + i * 137  # deterministic but varied seeds
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=200)
        labels_i = km.fit_predict(X_pca)
        seed_labels.append(labels_i)

        # Silhouette on a subsample
        cap_i = min(n, cap)
        rng_i = np.random.RandomState(seed)
        idx = rng_i.choice(n, size=cap_i, replace=False) if cap_i < n else np.arange(n)
        try:
            sil = silhouette_score(X_pca[idx], labels_i[idx])
            sil_scores.append(sil)
        except Exception:  # noqa: BLE001
            pass

    # Pairwise ARI
    ari_values: list[float] = [
        adjusted_rand_score(seed_labels[a], seed_labels[b])
        for a, b in combinations(range(n_seeds), 2)
    ]

    report = {
        "silhouette_score": float(np.mean(sil_scores)) if sil_scores else float("nan"),
        "ari_mean": float(np.mean(ari_values)) if ari_values else float("nan"),
        "ari_std": float(np.std(ari_values)) if ari_values else float("nan"),
        "n_seeds": n_seeds,
    }
    logger.info(
        "_compute_stability_metrics: silhouette=%.3f, ARI mean=%.3f ± %.3f (%d seeds)",
        report["silhouette_score"],
        report["ari_mean"],
        report["ari_std"],
        n_seeds,
    )
    return report


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

    # ── Active PCA (GMM) + scaled features (UMAP→HDBSCAN) ───────────────────
    X_pca_active: np.ndarray | None = None
    X_scaled_active: np.ndarray | None = None
    if active_mask.sum() > 0:
        X_pca_active, X_scaled_active = _get_active_pca(
            out, active_mask, selected_cols, cfg, rng
        )

    # ── Stability metrics (run before main clustering to use same X_pca) ─────
    stability_report = _compute_stability_metrics(X_pca, cfg)

    # ── GMM on active agents (PCA space) ─────────────────────────────────────
    out["cluster_id_gmm"] = pd.array([-1] * len(out), dtype="Int64")
    if X_pca_active is not None:
        gmm_labels, _ = _run_gmm(X_pca_active, cfg, rng)
        active_indices = out.index[active_mask]
        out.loc[active_indices, "cluster_id_gmm"] = gmm_labels

    # ── HDBSCAN on active agents (UMAP space when available) ─────────────────
    out["cluster_hdb_raw"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["hdb_tier"] = pd.NA

    if X_pca_active is not None and X_scaled_active is not None:
        try:
            if cfg.get("use_umap_for_hdbscan", True):
                X_hdbscan = _get_active_umap(X_scaled_active, cfg, rng)
            else:
                X_hdbscan = X_pca_active

            hdb_labels = _run_hdbscan(X_hdbscan, cfg)
            active_indices = out.index[active_mask]
            out.loc[active_indices, "cluster_hdb_raw"] = hdb_labels
            features_df_active = out.loc[active_mask].reset_index(drop=True)
            hdb_tier_active = _map_hdb_to_tier(hdb_labels, features_df_active, cfg)
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

    # ── Attach stability report as DataFrame metadata ─────────────────────────
    out.attrs["stability_report"] = stability_report

    # Log summary
    seg_counts = out["segment"].value_counts().to_dict()
    logger.info("run_clustering_pipeline: segment distribution — %s", seg_counts)

    return out

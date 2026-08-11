"""
extrafloat_segmentation_pipeline.py
=====================================
GMM + HDBSCAN diagnostic clustering for Uganda MTN MoMo agent segmentation.

This module is diagnostics-only. It no longer decides any agent's business
tier — that is `extrafloat_segmentation_scoring.py`'s job, via a
deterministic, versioned capacity scorecard. This pipeline exists to
support two things a frozen scorecard cannot do on its own:

  1. **Anomaly flagging** — HDBSCAN noise points (diag_is_anomaly) surface
     agents whose behavior doesn't resemble any dense cluster, worth a
     human look regardless of what their capacity_tier says.
  2. **Archetype research** — GMM/HDBSCAN groupings and pack profiles let
     an analyst ask "what does a Gold-tier agent's feature profile
     actually look like" when reviewing/recalibrating a scorecard.

Earlier versions of this module also ran two rounds of KMeans and used a
composite-score cluster ranking to assign the business `segment`/`tier`
directly from cluster membership. That mechanism is gone: an agent's tier
moved whenever the *population* changed, even with the agent's own
behavior unchanged, which is not defensible for a product whose tier sizes
a real credit limit. See `extrafloat_segmentation_scoring.py`'s module
docstring for the replacement and the reasoning.

Because this is diagnostics rather than production scoring, it is entirely
optional: `run_extrafloat_segmentation` only calls
`run_diagnostic_clustering` when `clustering.enable_diagnostics=True`, and
missing optional dependencies (hdbscan / umap-learn) can only ever block a
deliberately-requested diagnostics run, never a production capacity-scoring
run.

Market: Uganda (UG) — MTN Mobile Money.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
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

DORMANT_FILL_LABEL: str = "Dormant_Cluster"
ENSEMBLE_MISSING_LABEL: str = "ENSEMBLE_MISSING"
HDBSCAN_NOISE_LABEL: int = -1
# scikit-learn's outlier-detector convention (LocalOutlierFactor.fit_predict,
# IsolationForest, ...): -1 = outlier, 1 = inlier. Same integer as
# HDBSCAN_NOISE_LABEL by coincidence of two different libraries' conventions
# — kept as a separate constant so the two are never confused for the same
# concept in code that reads both.
LOF_OUTLIER_LABEL: int = -1

# HDB tier names ordered from lowest to highest activity (data-driven assignment).
# Purely descriptive labels for diagnostic clusters — NOT business tiers.
# Business tiers (BUSINESS_SEGMENTS) live in extrafloat_segmentation_scoring.py.
_HDB_TIER_NAMES: list[str] = [
    "Emerging / Low Activity",
    "Bronze Active",
    "Developing Active",
    "Silver Strong",
    "Gold Power",
    "Platinum Power",
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CLUSTERING_CONFIG: dict[str, Any] = {
    "random_state": 42,
    # When False (default), run_diagnostic_clustering is never called from
    # run_extrafloat_segmentation — capacity scoring (deterministic scorecard)
    # is the only thing that decides an agent's tier. Set True to also run
    # the full GMM/HDBSCAN diagnostics bundle (archetype research, plus a
    # second, UMAP-space anomaly signal: diag_is_anomaly) alongside.
    "enable_diagnostics": False,
    # When True (default), flag_anomalies runs on every production run —
    # HDBSCAN only (no GMM, no UMAP) directly on the same PCA space used
    # elsewhere in this module, cheap enough to run unconditionally. Unlike
    # enable_diagnostics, a missing hdbscan install degrades gracefully
    # here (is_anomaly=False) rather than raising. See flag_anomalies().
    "enable_anomaly_detection": True,
    # ── LOF stage 2 (two-stage anomaly filter) ──────────────────────────────
    # flag_anomalies runs HDBSCAN first as a coarse global filter
    # (is_global_anomaly), then — only on the agents HDBSCAN actually
    # placed in a cluster, not the full active population — runs Local
    # Outlier Factor to catch subtler local anomalies within an otherwise
    # dense cluster (is_local_anomaly). Restricting LOF to the HDBSCAN
    # survivor subset is what keeps this a "filter" rather than doubling
    # the cost of anomaly detection outright. scikit-learn is already a
    # hard dependency of this whole package, so this stage needs no
    # optional-dependency guard the way HDBSCAN/UMAP do.
    "lof_enabled": True,
    "lof_n_neighbors": 20,
    "lof_contamination": "auto",
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
    # ── Diagnostic ensemble stability reporting (opt-in; expensive) ──────────
    "stability_n_seeds": 3,
    # O(stability_n_seeds) full active-diagnostics re-runs — off by default;
    # meant for scheduled/offline validation, not every production run.
    # See compute_diagnostic_ensemble_stability().
    "full_stability_check": False,
    # ── Optional dependency requirements ───────────────────────────────────────
    # HDBSCAN/UMAP are "soft" (importable-or-not) dependencies. Only matter
    # when enable_diagnostics=True: a missing install then raises RuntimeError
    # at the start of run_diagnostic_clustering instead of silently degrading
    # diagnostic quality. Never blocks production capacity scoring, which
    # does not depend on this module at all.
    "require_hdbscan": True,
    "require_umap": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _get_clustering_config(config: dict | None) -> dict[str, Any]:
    """Return deep-merged clustering config, filling missing keys from defaults."""
    if config is None:
        return deepcopy(DEFAULT_CLUSTERING_CONFIG)
    merged = deepcopy(DEFAULT_CLUSTERING_CONFIG)
    merged.update(config)
    return merged


def _check_optional_dependencies(cfg: dict[str, Any]) -> dict[str, bool]:
    """Verify optional ML dependencies (hdbscan, umap-learn) against config.

    Only ever called from `run_diagnostic_clustering`, i.e. only when a
    caller has deliberately opted into diagnostics
    (`clustering.enable_diagnostics=True`). Production capacity scoring
    never imports this module's public function and so can never be blocked
    by a missing optional dependency here.

    Parameters
    ----------
    cfg : Clustering config (post `_get_clustering_config`).

    Returns
    -------
    dict with keys ``hdbscan_available`` / ``umap_available``.

    Raises
    ------
    RuntimeError
        If ``require_hdbscan``/``require_umap`` is True (the default) and
        the corresponding package is not installed.
    """
    status = {
        "hdbscan_available": _HDBSCAN_AVAILABLE,
        "umap_available": _UMAP_AVAILABLE,
    }

    if cfg.get("require_hdbscan", True) and not _HDBSCAN_AVAILABLE:
        raise RuntimeError(
            "run_diagnostic_clustering: hdbscan is not installed but "
            "require_hdbscan=True (the default). Install with "
            "'pip install hdbscan', or explicitly set "
            "config['clustering']['require_hdbscan']=False to allow a "
            "degraded diagnostics run (diag_hdb_tier will be 'Unavailable' "
            "for all active agents)."
        )
    if cfg.get("require_umap", True) and not _UMAP_AVAILABLE:
        raise RuntimeError(
            "run_diagnostic_clustering: umap-learn is not installed but "
            "require_umap=True (the default). Install with "
            "'pip install umap-learn', or explicitly set "
            "config['clustering']['require_umap']=False to allow a "
            "degraded diagnostics run (HDBSCAN will receive a raw "
            "PCA-scaled column slice instead of a UMAP embedding)."
        )
    return status


def _identify_dormant_mask(
    features_df: pd.DataFrame, cfg: dict[str, Any]
) -> pd.Series:
    """Identify dormant agents via a multi-product composite inactivity score.

    For each configured inactivity column the column is normalised by its 95th
    percentile (clipped to [0, 1]) and then the weighted sum is computed.
    Agents whose composite score is at or below *dormant_composite_threshold*
    are classified as dormant.

    This is the single source of truth for dormancy across the whole
    segmentation pipeline — both `extrafloat_segmentation_scoring.py`
    (dormant agents forced to the lowest capacity tier) and this module's
    diagnostics call it with the same config.

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
    """Assign data-driven diagnostic labels to HDBSCAN cluster IDs.

    Ranks non-noise clusters by their mean value of the first available
    business KPI column from *hdb_tier_ranking_cols* (default: total_value_1m
    → cash_out_value_1m → commission).  Using a business KPI centroid rather
    than PC1 makes the diagnostic label order reflect commercial value, not
    variance. These labels are descriptive research aids only — they are
    NOT business tiers and never feed into capacity_tier.
    Noise label (-1) -> "Noise / Irregular".

    Parameters
    ----------
    hdb_labels_active  : HDBSCAN label array for active agents.
    features_df_active : Raw feature DataFrame for active agents, aligned with
                         hdb_labels_active (used for KPI centroid ranking).
    cfg                : Clustering config.

    Returns
    -------
    pd.Series of diagnostic label strings, same length as hdb_labels_active.
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
        # Last-resort: warn and skip ranking (assign labels in discovery order)
        logger.warning(
            "_map_hdb_to_tier: none of %s found in features_df_active — "
            "assigning labels in cluster-ID order (no business-KPI ranking).",
            ranking_cols,
        )
        cluster_means = {cid: float(cid) for cid in unique_clusters}

    sorted_clusters = sorted(unique_clusters, key=lambda c: cluster_means[c])

    # Evenly distribute descriptive labels across clusters
    tier_groups = np.array_split(sorted_clusters, min(len(_HDB_TIER_NAMES), len(sorted_clusters)))
    cluster_to_tier: dict[int, str] = {}
    tier_list = _HDB_TIER_NAMES[-len(tier_groups):]  # use highest labels if fewer clusters
    for tier_name, group in zip(tier_list, tier_groups):
        for cid in group:
            cluster_to_tier[cid] = tier_name

    logger.info(
        "_map_hdb_to_tier: mapping %d clusters -> %d diagnostic labels",
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
    """Combine GMM ID and HDB diagnostic label into "GMM_{id}__{tier}" string labels.

    Parameters
    ----------
    gmm_labels   : Integer GMM cluster IDs (Int64 series).
    hdb_tier_col : String HDB diagnostic label.

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


DEFAULT_DIAGNOSTIC_STABILITY_REPORT: dict[str, Any] = {
    "ensemble_consistency_rate": float("nan"),
    "ensemble_ari_mean": float("nan"),
    "ensemble_ari_std": float("nan"),
    "n_seeds": 0,
    "n_active": 0,
}


def _run_active_pipeline_once(
    features_df: pd.DataFrame,
    active_mask: pd.Series,
    selected_cols: list[str],
    cfg: dict[str, Any],
    rng: np.random.RandomState,
) -> pd.Series:
    """Run GMM -> UMAP/HDBSCAN -> ensemble label once for active agents.

    Factored out of the inline steps in `run_diagnostic_clustering` so
    `compute_diagnostic_ensemble_stability` can re-run just the active-agent
    stage under a fresh seed without duplicating that orchestration logic.

    Returns
    -------
    pd.Series of ensemble-cluster label strings, indexed like *features_df*
    (agents outside *active_mask* get DORMANT_FILL_LABEL).
    """
    out_ensemble = pd.Series(DORMANT_FILL_LABEL, index=features_df.index, dtype="object")

    if active_mask.sum() == 0:
        return out_ensemble

    X_pca_active, X_scaled_active = _get_active_pca(
        features_df, active_mask, selected_cols, cfg, rng
    )
    gmm_labels, _ = _run_gmm(X_pca_active, cfg, rng)
    active_index = features_df.index[active_mask]
    gmm_series = pd.Series(gmm_labels, index=active_index, dtype="Int64")

    try:
        if cfg.get("use_umap_for_hdbscan", True):
            X_hdbscan = _get_active_umap(X_scaled_active, cfg, rng)
        else:
            X_hdbscan = X_pca_active
        hdb_labels = _run_hdbscan(X_hdbscan, cfg)
        features_df_active = features_df.loc[active_mask].reset_index(drop=True)
        hdb_tier_active = _map_hdb_to_tier(hdb_labels, features_df_active, cfg)
        hdb_series = pd.Series(hdb_tier_active.values, index=active_index, dtype="object")
    except ImportError:
        hdb_series = pd.Series("Unavailable", index=active_index, dtype="object")

    ensemble_active = _build_ensemble_labels(gmm_series, hdb_series)
    ensemble_active = ensemble_active.replace({ENSEMBLE_MISSING_LABEL: DORMANT_FILL_LABEL})
    out_ensemble.loc[active_index] = ensemble_active.values

    return out_ensemble


def compute_diagnostic_ensemble_stability(
    features_df: pd.DataFrame,
    active_mask: pd.Series,
    selected_cols: list[str],
    config: dict | None = None,
    n_seeds: int | None = None,
) -> dict[str, Any]:
    """Measure reproducibility of the GMM+HDBSCAN diagnostic ensemble label.

    Since this module no longer decides any agent's business tier, this
    function measures whether the *diagnostic* ensemble_cluster grouping
    (used for anomaly flagging / archetype research) is itself reproducible
    across random seeds — useful context when trusting an anomaly flag or an
    archetype profile derived from a single run.

    This is O(n_seeds) full active-diagnostics re-runs, so it is intended
    for scheduled/offline validation runs rather than every production
    invocation — call it explicitly; it is not part of
    `run_diagnostic_clustering`.

    Parameters
    ----------
    features_df : Agent feature DataFrame (post feature engineering).
    active_mask : Boolean mask of non-dormant agents, e.g. from
                  `_identify_dormant_mask(features_df, cfg)`.
    selected_cols : Feature columns used for scaling/PCA.
    config : Clustering config (same shape as DEFAULT_CLUSTERING_CONFIG).
    n_seeds : Override `stability_n_seeds` from config.

    Returns
    -------
    dict with keys: ensemble_consistency_rate (fraction of agents assigned
    the *same* ensemble_cluster in every seed run), ensemble_ari_mean,
    ensemble_ari_std (pairwise Adjusted Rand Index across seed pairs),
    n_seeds, n_active.
    """
    from itertools import combinations  # noqa: PLC0415
    from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415

    cfg = _get_clustering_config(config)
    seeds = int(n_seeds) if n_seeds is not None else int(cfg.get("stability_n_seeds", 3))
    base_seed = int(cfg.get("random_state", 42))

    if seeds < 2:
        logger.warning(
            "compute_diagnostic_ensemble_stability: n_seeds=%d < 2 — "
            "returning default report.",
            seeds,
        )
        return deepcopy(DEFAULT_DIAGNOSTIC_STABILITY_REPORT)

    n_active = int(active_mask.sum())
    if n_active == 0:
        logger.warning(
            "compute_diagnostic_ensemble_stability: no active agents — "
            "returning default report."
        )
        return deepcopy(DEFAULT_DIAGNOSTIC_STABILITY_REPORT)

    seed_ensembles: list[pd.Series] = []
    for i in range(seeds):
        seed = base_seed + i * 137  # deterministic but varied seeds
        rng = np.random.RandomState(seed)
        seed_ensembles.append(
            _run_active_pipeline_once(features_df, active_mask, selected_cols, cfg, rng)
        )

    matrix = pd.concat(seed_ensembles, axis=1)
    matrix.columns = [f"seed_{i}" for i in range(seeds)]

    consistency = float((matrix.nunique(axis=1) == 1).mean())
    ari_values = [
        adjusted_rand_score(matrix.iloc[:, a], matrix.iloc[:, b])
        for a, b in combinations(range(seeds), 2)
    ]

    report = {
        "ensemble_consistency_rate": consistency,
        "ensemble_ari_mean": float(np.mean(ari_values)),
        "ensemble_ari_std": float(np.std(ari_values)),
        "n_seeds": seeds,
        "n_active": n_active,
    }
    logger.info(
        "compute_diagnostic_ensemble_stability: consistency=%.3f, ensemble ARI "
        "mean=%.3f ± %.3f (%d seeds, %d active agents)",
        report["ensemble_consistency_rate"],
        report["ensemble_ari_mean"],
        report["ensemble_ari_std"],
        seeds,
        n_active,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────


def flag_anomalies(
    features_df: pd.DataFrame,
    selected_cols: list[str],
    active_mask: pd.Series,
    config: dict | None = None,
) -> pd.DataFrame:
    """Two-stage anomaly filter: HDBSCAN global pass, then LOF local refinement.

    Stage 1 (HDBSCAN, dependency-graceful) runs alone — no GMM, no UMAP —
    directly on the same PCA space (`_get_active_pca`) used elsewhere in
    this module, flagging agents that don't belong to *any* dense cluster
    at all (`is_global_anomaly`). That's what makes this cheap enough to
    run on every production run by default
    (`clustering.enable_anomaly_detection`, default True), independent of
    the full diagnostics bundle (`clustering.enable_diagnostics`, default
    False, which additionally runs GMM + UMAP and computes its own,
    differently-embedded anomaly signal, `diag_is_anomaly`).

    Stage 2 (LOF, `clustering.lof_enabled`, default True) runs Local
    Outlier Factor — but only on the agents stage 1 actually placed inside
    an HDBSCAN cluster, not the full active population — to catch subtler
    *local* anomalies: agents that look unremarkable globally but are
    still off relative to their immediate cluster neighborhood
    (`is_local_anomaly`). Restricting LOF to the HDBSCAN-survivor subset
    is what keeps this a two-stage *filter* rather than doubling the cost
    of anomaly detection outright. scikit-learn is already a hard
    dependency of this package (unlike hdbscan/umap-learn), so stage 2
    needs no optional-dependency guard — it simply doesn't run if stage 1
    didn't (no hdbscan) or left too few in-cluster agents to be meaningful.

    `is_anomaly` (the column production code should read by default) is
    the OR of both stages — flagged if either algorithm flags it.

    Unlike `run_diagnostic_clustering`, a missing `hdbscan` install
    degrades gracefully here (`is_anomaly=False` for every agent, logged as
    a warning) rather than raising `RuntimeError`. This path is meant to
    run unconditionally as part of every production run; refusing to
    compute `capacity_tier` over a missing *optional* dependency for an
    *auxiliary* signal would defeat that.

    Parameters
    ----------
    features_df : Agent feature DataFrame (post feature engineering).
    selected_cols : Feature columns used for scaling/PCA.
    active_mask : Boolean mask of non-dormant agents, e.g. from
                  `_identify_dormant_mask(features_df, cfg)`.
    config : Clustering config (same shape as DEFAULT_CLUSTERING_CONFIG).

    Returns
    -------
    pd.DataFrame indexed like *features_df* with columns:
        is_anomaly              : bool, True if flagged by either stage
                                   (always False for dormant agents and
                                   whenever hdbscan is unavailable or there
                                   are no active agents).
        is_global_anomaly       : bool, True for HDBSCAN noise points
                                   among active agents (stage 1).
        is_local_anomaly        : bool, True for LOF-flagged outliers
                                   among agents that landed in an HDBSCAN
                                   cluster (stage 2); always False when
                                   stage 2 didn't run.
        anomaly_cluster_hdb_raw : Int64, raw HDBSCAN label (pd.NA for
                                   dormant agents / when hdbscan unavailable).
        lof_score               : float, LOF's `negative_outlier_factor_`
                                   (more negative = more anomalous), NaN
                                   wherever stage 2 didn't run for that agent.
    """
    cfg = _get_clustering_config(config)

    out = pd.DataFrame(index=features_df.index)
    out["is_global_anomaly"] = False
    out["is_local_anomaly"] = False
    out["anomaly_cluster_hdb_raw"] = pd.array([pd.NA] * len(features_df), dtype="Int64")
    out["lof_score"] = np.nan

    if not _HDBSCAN_AVAILABLE:
        logger.warning(
            "flag_anomalies: hdbscan is not installed — is_anomaly "
            "defaulting to False for all agents. Install with "
            "'pip install hdbscan' to enable anomaly flagging, or set "
            "config['clustering']['enable_anomaly_detection']=False to "
            "silence this warning."
        )
        out["is_anomaly"] = False
        return out

    if active_mask.sum() == 0:
        logger.info("flag_anomalies: no active agents — nothing to flag.")
        out["is_anomaly"] = False
        return out

    rng = np.random.RandomState(cfg["random_state"])
    X_pca_active, _ = _get_active_pca(features_df, active_mask, selected_cols, cfg, rng)
    hdb_labels = _run_hdbscan(X_pca_active, cfg)

    active_index = features_df.index[active_mask]
    out.loc[active_index, "anomaly_cluster_hdb_raw"] = hdb_labels
    out.loc[active_index, "is_global_anomaly"] = hdb_labels == HDBSCAN_NOISE_LABEL

    # ── Stage 2: LOF on HDBSCAN survivors only ────────────────────────────
    min_required_for_lof = 2
    if cfg.get("lof_enabled", True):
        in_cluster_local_mask = hdb_labels != HDBSCAN_NOISE_LABEL
        n_in_cluster = int(in_cluster_local_mask.sum())

        if n_in_cluster >= min_required_for_lof:
            n_neighbors = max(1, min(int(cfg.get("lof_n_neighbors", 20)), n_in_cluster - 1))
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=cfg.get("lof_contamination", "auto"),
            )
            X_in_cluster = X_pca_active[in_cluster_local_mask]
            lof_pred = lof.fit_predict(X_in_cluster)  # -1 = outlier, 1 = inlier

            in_cluster_index = active_index[in_cluster_local_mask]
            out.loc[in_cluster_index, "is_local_anomaly"] = lof_pred == LOF_OUTLIER_LABEL
            out.loc[in_cluster_index, "lof_score"] = lof.negative_outlier_factor_

            logger.info(
                "flag_anomalies: LOF stage flagged %d/%d in-cluster agent(s) "
                "as local anomalies (n_neighbors=%d).",
                int((lof_pred == LOF_OUTLIER_LABEL).sum()),
                n_in_cluster,
                n_neighbors,
            )
        else:
            logger.info(
                "flag_anomalies: only %d in-cluster agent(s) after HDBSCAN — "
                "skipping LOF stage (needs >= %d).",
                n_in_cluster,
                min_required_for_lof,
            )
    else:
        logger.info("flag_anomalies: LOF stage disabled (clustering.lof_enabled=False).")

    out["is_anomaly"] = out["is_global_anomaly"] | out["is_local_anomaly"]

    n_anomalies = int(out["is_anomaly"].sum())
    n_global = int(out["is_global_anomaly"].sum())
    n_local = int(out["is_local_anomaly"].sum())
    logger.info(
        "flag_anomalies: %d/%d active agents flagged as anomalies "
        "(%d global via HDBSCAN, %d local via LOF).",
        n_anomalies,
        int(active_mask.sum()),
        n_global,
        n_local,
    )
    return out


def run_diagnostic_clustering(
    features_df: pd.DataFrame,
    X_pca: np.ndarray,
    selected_cols: list[str],
    config: dict | None = None,
) -> pd.DataFrame:
    """GMM + HDBSCAN diagnostic clustering — anomaly flagging and archetype research.

    Does NOT assign any business tier. Only called from
    `run_extrafloat_segmentation` when `clustering.enable_diagnostics=True`;
    the deterministic capacity scorecard
    (`extrafloat_segmentation_scoring.compute_agent_capacity`) is what
    decides `capacity_tier`, independent of this function.

    Parameters
    ----------
    features_df :
        Agent-level feature DataFrame (index must align with rows of X_pca).
    X_pca :
        PCA-reduced feature array aligned with features_df. Accepted for
        interface symmetry with the rest of the pipeline, but this function
        recomputes its own active-agent PCA (`_get_active_pca`) rather than
        reusing *X_pca*, since dormant agents must be excluded first.
    selected_cols :
        Feature column names for re-scaling active agents for GMM/HDBSCAN.
    config :
        Clustering configuration dict. Missing keys fall back to
        DEFAULT_CLUSTERING_CONFIG.

    Returns
    -------
    pd.DataFrame
        features_df with added diagnostic-only columns:
            diag_cluster_id_gmm  : GMM label (Int64, -1 for dormant)
            diag_cluster_hdb_raw : Raw HDBSCAN integer label (Int64)
            diag_hdb_tier        : Descriptive HDBSCAN label (data-driven, NOT a business tier)
            diag_ensemble_cluster: Combined "GMM_{id}__{tier}" label
            diag_is_anomaly      : True where diag_cluster_hdb_raw == HDBSCAN_NOISE_LABEL
    """
    cfg = _get_clustering_config(config)
    dep_status = _check_optional_dependencies(cfg)
    rng = np.random.RandomState(cfg["random_state"])

    if X_pca.shape[0] != len(features_df):
        raise ValueError(
            f"run_diagnostic_clustering: X_pca has {X_pca.shape[0]} rows but "
            f"features_df has {len(features_df)} rows."
        )

    out = features_df.copy()

    logger.info(
        "run_diagnostic_clustering: starting on %d agents, X_pca shape=%s",
        len(out),
        X_pca.shape,
    )

    dormant_mask = _identify_dormant_mask(out, cfg)
    active_mask = ~dormant_mask

    # ── Active PCA (GMM) + scaled features (UMAP→HDBSCAN) ───────────────────
    X_pca_active: np.ndarray | None = None
    X_scaled_active: np.ndarray | None = None
    if active_mask.sum() > 0:
        X_pca_active, X_scaled_active = _get_active_pca(
            out, active_mask, selected_cols, cfg, rng
        )

    # ── GMM on active agents (PCA space) ─────────────────────────────────────
    out["diag_cluster_id_gmm"] = pd.array([-1] * len(out), dtype="Int64")
    if X_pca_active is not None:
        gmm_labels, _ = _run_gmm(X_pca_active, cfg, rng)
        active_indices = out.index[active_mask]
        out.loc[active_indices, "diag_cluster_id_gmm"] = gmm_labels

    # ── HDBSCAN on active agents (UMAP space when available) ─────────────────
    out["diag_cluster_hdb_raw"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["diag_hdb_tier"] = pd.NA

    if X_pca_active is not None and X_scaled_active is not None:
        try:
            if cfg.get("use_umap_for_hdbscan", True):
                X_hdbscan = _get_active_umap(X_scaled_active, cfg, rng)
            else:
                X_hdbscan = X_pca_active

            hdb_labels = _run_hdbscan(X_hdbscan, cfg)
            active_indices = out.index[active_mask]
            out.loc[active_indices, "diag_cluster_hdb_raw"] = hdb_labels
            features_df_active = out.loc[active_mask].reset_index(drop=True)
            hdb_tier_active = _map_hdb_to_tier(hdb_labels, features_df_active, cfg)
            out.loc[active_indices, "diag_hdb_tier"] = hdb_tier_active.values
        except ImportError as exc:
            logger.error(
                "run_diagnostic_clustering: HDBSCAN unavailable — %s. "
                "Skipping HDBSCAN stage; diag_hdb_tier will be 'Unavailable'.",
                exc,
            )
            out.loc[active_mask, "diag_hdb_tier"] = "Unavailable"

    # ── Ensemble labels ───────────────────────────────────────────────────────
    out["diag_ensemble_cluster"] = DORMANT_FILL_LABEL

    gmm_active = out.loc[active_mask, "diag_cluster_id_gmm"]
    hdb_active = out.loc[active_mask, "diag_hdb_tier"]

    if active_mask.sum() > 0:
        ensemble_active = _build_ensemble_labels(gmm_active, hdb_active)
        ensemble_active = ensemble_active.replace(
            {ENSEMBLE_MISSING_LABEL: DORMANT_FILL_LABEL}
        )
        out.loc[active_mask, "diag_ensemble_cluster"] = ensemble_active.values

    # ── Anomaly flag: HDBSCAN noise points among active agents ──────────────
    out["diag_is_anomaly"] = out["diag_cluster_hdb_raw"] == HDBSCAN_NOISE_LABEL

    # ── Attach degraded-mode flag as DataFrame metadata ──────────────────────
    out.attrs["diag_degraded_mode"] = {
        "hdbscan_unavailable": not dep_status["hdbscan_available"],
        "umap_unavailable": not dep_status["umap_available"],
    }
    if any(out.attrs["diag_degraded_mode"].values()):
        logger.warning(
            "run_diagnostic_clustering: completed in DEGRADED MODE — %s. "
            "Diagnostic quality is reduced; see out.attrs['diag_degraded_mode'].",
            out.attrs["diag_degraded_mode"],
        )

    n_anomalies = int(out["diag_is_anomaly"].sum())
    logger.info(
        "run_diagnostic_clustering: %d/%d active agents flagged as anomalies "
        "(HDBSCAN noise).",
        n_anomalies,
        int(active_mask.sum()),
    )

    return out

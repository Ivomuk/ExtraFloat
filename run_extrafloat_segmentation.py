"""
run_extrafloat_segmentation.py
================================
Orchestration entry point for the MTN MoMo agent segmentation pipeline.

Chains: feature engineering → clustering → pack profiling → tier building →
optional whitelist/blacklist enrichment → output trimming.

Market: Uganda (UG) — MTN Mobile Money agent segmentation.

Usage
-----
>>> from run_extrafloat_segmentation import run_extrafloat_segmentation, DEFAULT_SEGMENTATION_CONFIG
>>> import pandas as pd
>>> agents_df = pd.read_csv("path/to/kpi_mart.csv")
>>> config = dict(DEFAULT_SEGMENTATION_CONFIG)
>>> result = run_extrafloat_segmentation(agents_df, config=config)
>>> result[["agent_msisdn", "segment", "hdb_tier", "ensemble_cluster"]].head()
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from extrafloat_segmentation_features import prepare_features, REQUIRED_COLUMNS
from extrafloat_segmentation_pipeline import run_clustering_pipeline, BUSINESS_SEGMENTS
from extrafloat_segmentation_profiling import (
    build_cluster_pack_profiles,
    build_cluster_tiers,
    merge_reference_lists,
    PROFILING_PACKS,
    DEFAULT_PROFILING_CONFIG,
)
from extrafloat_segmentation_pipeline import DEFAULT_CLUSTERING_CONFIG
from extrafloat_segmentation_drift import (
    build_drift_report,
    load_drift_baseline,
    save_drift_baseline,
    DEFAULT_DRIFT_CONFIG,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

SEGMENT_OUTPUT_COLUMNS: tuple[str, ...] = (
    "agent_msisdn",
    "pos_msisdn",
    "cluster_round1",
    "cluster_round2",
    "cluster_id_gmm",
    "cluster_hdb_raw",
    "hdb_tier",
    "ensemble_cluster",
    "segment",
    "tier",
    "tier_sub_label",
    "safety_flags",
)

INTERMEDIATE_COLUMNS: tuple[str, ...] = (
    "cluster_round1",
    "cluster_round2",
    "cluster_id_gmm",
    "cluster_hdb_raw",
)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SEGMENTATION_CONFIG: dict[str, Any] = {
    "data": {
        # Path to the main KPI mart CSV.  Required when loading from disk.
        "agents_path": "",
        # Optional reference list paths.
        "whitelist_path": "",
        "blacklist_path": "",
        "date_format": "%Y%m%d",
        "msisdn_col": "agent_msisdn",
        # MSISDNs of commission/super agents to exclude before clustering.
        "commission_agents_msisdn": [256789760000, 256783891111, 256781872222, 256772453333],
    },
    "features": {
        "date_format": "%Y%m%d",
        "recency_cap_days": 3650,
        "skew_threshold": 1.0,
        "winsorize_lower_pct": 0.005,
        "winsorize_upper_pct": 0.995,
        "corr_threshold": 0.97,
        "target_pca_variance": 0.90,
        "random_state": 42,
        "max_interaction_cols": 10,
    },
    "clustering": {
        **DEFAULT_CLUSTERING_CONFIG,
    },
    "profiling": {
        **DEFAULT_PROFILING_CONFIG,
    },
    "output": {
        # Where to write optional CSV exports (empty string = no export).
        "output_dir": "segmentation_outputs",
        # When False only the SEGMENT_OUTPUT_COLUMNS are returned.
        "keep_intermediate_cols": False,
        "final_segment_col": "segment",
    },
    "drift": {
        **DEFAULT_DRIFT_CONFIG,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _get_config(config: dict | None) -> dict[str, Any]:
    """Return a deep-merged config, filling missing keys from defaults."""
    if config is None:
        return deepcopy(DEFAULT_SEGMENTATION_CONFIG)

    merged = deepcopy(DEFAULT_SEGMENTATION_CONFIG)
    for section, values in config.items():
        if isinstance(values, dict) and section in merged and isinstance(merged[section], dict):
            merged[section].update(values)
        else:
            merged[section] = values

    return merged


def _validate_config(cfg: dict[str, Any]) -> None:
    """Raise ValueError if required config keys are missing or invalid."""
    required_sections = ("data", "features", "clustering", "profiling", "output")
    for s in required_sections:
        if s not in cfg:
            raise ValueError(
                f"run_extrafloat_segmentation: config missing required section '{s}'."
            )

    weights = cfg["clustering"].get("composite_weights", {})
    total = sum(float(w) for w in weights.values()) if weights else 0.0
    if weights and abs(total - 1.0) > 0.01:
        logger.warning(
            "_validate_config: composite_weights sum to %.3f (expected 1.0). "
            "Scores will still be computed but rankings may be unexpected.",
            total,
        )


def _validate_input(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Raise ValueError if required columns are absent from *df*."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"run_extrafloat_segmentation: required columns missing from agents_df: "
            f"{missing}. Available columns: {list(df.columns)[:30]}"
        )


def _join_msisdn(result: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure agent_msisdn and pos_msisdn columns are present on the result.

    The clustering pipeline works on the feature matrix (which drops raw
    identifier columns).  This helper joins them back from the original df
    using the shared index.
    """
    for col in ("agent_msisdn", "pos_msisdn"):
        if col not in result.columns and col in original_df.columns:
            result[col] = original_df.loc[result.index, col]
    return result


def _trim_output_columns(
    result: pd.DataFrame,
    keep_intermediate: bool,
    original_cols: list[str],
) -> pd.DataFrame:
    """
    Keep the original agent columns plus the new segmentation output columns.

    When *keep_intermediate* is False, intermediate cluster columns
    (cluster_round1, cluster_round2, etc.) are dropped from the output.
    """
    new_seg_cols = [c for c in SEGMENT_OUTPUT_COLUMNS if c in result.columns]

    if not keep_intermediate:
        new_seg_cols = [c for c in new_seg_cols if c not in INTERMEDIATE_COLUMNS]

    # Preserve original agent columns that are not being replaced
    pass_through = [c for c in original_cols if c not in new_seg_cols and c in result.columns]

    final_cols = pass_through + [c for c in new_seg_cols if c not in pass_through]
    # Only keep columns that actually exist
    final_cols = [c for c in final_cols if c in result.columns]

    return result[final_cols].copy()


def _maybe_save_output(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Optionally write the segmented agent DataFrame to CSV."""
    out_dir = cfg["output"].get("output_dir", "")
    if not out_dir:
        return

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "agent_segments.csv")
    df.to_csv(out_path, index=False)
    logger.info("_maybe_save_output: wrote %d rows to %s", len(df), out_path)


def _load_reference_list(path: str, label: str) -> pd.DataFrame | None:
    """Load a whitelist or blacklist CSV from *path*; return None if path empty."""
    if not path:
        return None
    if not os.path.isfile(path):
        logger.warning(
            "_load_reference_list: %s path '%s' does not exist — skipping.",
            label,
            path,
        )
        return None
    df_ref = pd.read_csv(path)
    logger.info(
        "_load_reference_list: loaded %s with %d rows from '%s'.",
        label,
        len(df_ref),
        path,
    )
    return df_ref


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────


def run_extrafloat_segmentation(
    agents_df: pd.DataFrame,
    config: dict | None = None,
    whitelist_df: pd.DataFrame | None = None,
    blacklist_df: pd.DataFrame | None = None,
    validate_inputs: bool = True,
) -> pd.DataFrame:
    """
    End-to-end agent segmentation pipeline.

    Chains:
        1. Feature engineering (prepare_features)
        2. Two-stage clustering (run_clustering_pipeline)
        3. Pack-based profiling (build_cluster_pack_profiles)
        4. Tier assignment (build_cluster_tiers)
        5. Optional whitelist/blacklist enrichment (merge_reference_lists)
        6. Output trimming

    Parameters
    ----------
    agents_df :
        Raw agent KPI DataFrame from the MTN MoMo daily agent mart.
        Must contain at minimum: agent_msisdn, commission, cash_out_vol_1m,
        cash_out_value_1m, cash_in_value_1m.
    config :
        Nested segmentation configuration dict.  Missing top-level sections
        fall back to DEFAULT_SEGMENTATION_CONFIG.  Example override::

            config = {
                "features": {"corr_threshold": 0.95},
                "clustering": {"kmeans_round1_k": 8},
            }

    whitelist_df :
        Optional reference DataFrame for whitelisted agents.  If None and
        config["data"]["whitelist_path"] is set, the file is loaded from disk.
    blacklist_df :
        Optional reference DataFrame for blacklisted agents.  Same loading
        logic as whitelist_df.
    validate_inputs :
        When True, checks for required columns before processing.

    Returns
    -------
    pd.DataFrame
        agents_df enriched with segmentation columns:
            segment         : final business segment (from BUSINESS_SEGMENTS)
            ensemble_cluster: GMM×HDBSCAN ensemble label
            hdb_tier        : HDBSCAN tier label (data-driven)
            tier            : pack-profile based tier (Platinum/Gold/Silver/Bronze)
            tier_sub_label  : activity mix sub-label (High Cash-Out, etc.)
            safety_flags    : semi-colon separated safety filter messages (if any)
        Plus intermediate cluster columns when keep_intermediate_cols=True.
    """
    cfg = _get_config(config)
    _validate_config(cfg)

    original_cols = list(agents_df.columns)
    df = agents_df.copy()

    if validate_inputs:
        _validate_input(df, cfg)

    logger.info(
        "run_extrafloat_segmentation: starting pipeline on %d agents.", len(df)
    )

    # ── Step 1: Feature Engineering ──────────────────────────────────────────
    logger.info("run_extrafloat_segmentation: step 1 — feature engineering.")
    features_df, X_scaled, X_pca, selected_cols = prepare_features(
        df,
        config=cfg.get("features"),
    )
    logger.info(
        "run_extrafloat_segmentation: features ready — %d agents, %d features, "
        "%d PCA components.",
        len(features_df),
        len(selected_cols),
        X_pca.shape[1],
    )

    # ── Step 1b: Feature Drift Detection (optional) ──────────────────────────
    drift_cfg = cfg.get("drift", {})
    baseline_path: str = drift_cfg.get("baseline_path", "")
    if baseline_path and os.path.isfile(baseline_path):
        logger.info(
            "run_extrafloat_segmentation: step 1b — drift detection against '%s'.",
            baseline_path,
        )
        try:
            baseline_df = load_drift_baseline(baseline_path)
            drift_report = build_drift_report(
                baseline_df=baseline_df,
                current_df=features_df,
                features=drift_cfg.get("drift_features"),
                config=drift_cfg,
            )
            features_df.attrs["drift_report"] = drift_report
            if drift_report.get("drift_detected"):
                logger.warning(
                    "run_extrafloat_segmentation: DRIFT DETECTED — %d feature(s) "
                    "show critical PSI shift. Segment assignments may be unreliable. "
                    "Critical: %s",
                    drift_report["n_critical"],
                    [
                        f for f, r in drift_report["features"].items()
                        if r["status"] == "critical"
                    ],
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "run_extrafloat_segmentation: drift detection failed — %s. "
                "Continuing without drift report.",
                exc,
            )
    else:
        if baseline_path:
            logger.info(
                "run_extrafloat_segmentation: step 1b — baseline '%s' not found, "
                "skipping drift detection. Run with save_baseline=True to create one.",
                baseline_path,
            )

    # ── Step 2: Clustering ────────────────────────────────────────────────────
    logger.info("run_extrafloat_segmentation: step 2 — clustering pipeline.")
    features_df = run_clustering_pipeline(
        features_df=features_df,
        X_pca=X_pca,
        selected_cols=selected_cols,
        config=cfg.get("clustering"),
    )

    stability = features_df.attrs.get("stability_report", {})
    if stability and stability.get("n_seeds", 0) > 0:
        logger.info(
            "run_extrafloat_segmentation: cluster stability — "
            "silhouette=%.3f, ARI mean=%.3f ± %.3f (%d seeds)",
            stability.get("silhouette_score", float("nan")),
            stability.get("ari_mean", float("nan")),
            stability.get("ari_std", float("nan")),
            stability.get("n_seeds", 0),
        )

    seg_col = cfg["output"].get("final_segment_col", "segment")
    if seg_col != "segment" and "segment" in features_df.columns:
        features_df = features_df.rename(columns={"segment": seg_col})

    # ── Step 3: Pack Profiling ────────────────────────────────────────────────
    logger.info("run_extrafloat_segmentation: step 3 — pack profiling.")
    pack_profiles = build_cluster_pack_profiles(
        df=features_df,
        cluster_col="ensemble_cluster",
        packs=PROFILING_PACKS,
        melt_for_heatmap=False,
    )

    # ── Step 4: Tier Assignment ───────────────────────────────────────────────
    logger.info("run_extrafloat_segmentation: step 4 — tier assignment.")
    from extrafloat_segmentation_profiling import _compute_cluster_stats  # noqa: PLC0415

    # Build per-cluster lift summary (mean lift across all packs)
    lifts = pack_profiles.get("lifts", {})
    if lifts:
        lift_frames = []
        for pack_name, lift_df in lifts.items():
            mean_lift = lift_df.mean(axis=1).rename(pack_name)
            lift_frames.append(mean_lift)
        cluster_lift_summary = pd.concat(lift_frames, axis=1)
    else:
        # Fallback: use raw means for the first pack
        means = pack_profiles.get("means", {})
        first_pack = next(iter(means.values()), pd.DataFrame())
        cluster_lift_summary = first_pack.mean(axis=1).rename("overall").to_frame()

    cluster_stats = _compute_cluster_stats(features_df, "ensemble_cluster")

    tier_table = build_cluster_tiers(
        cluster_pack_scores=cluster_lift_summary,
        cluster_stats=cluster_stats,
        config=cfg.get("profiling"),
    )

    # Map tier back onto agents via ensemble_cluster
    tier_map = tier_table.set_index("cluster")[["tier", "sub_label", "safety_flags"]]
    features_df["tier"] = features_df["ensemble_cluster"].map(tier_map["tier"])
    features_df["tier_sub_label"] = features_df["ensemble_cluster"].map(tier_map["sub_label"])
    features_df["safety_flags"] = features_df["ensemble_cluster"].map(tier_map["safety_flags"])

    # ── Step 5: Reference List Enrichment (optional) ─────────────────────────
    wl = whitelist_df
    bl = blacklist_df

    if wl is None:
        wl = _load_reference_list(cfg["data"].get("whitelist_path", ""), "whitelist")
    if bl is None:
        bl = _load_reference_list(cfg["data"].get("blacklist_path", ""), "blacklist")

    if wl is not None:
        logger.info("run_extrafloat_segmentation: step 5 — merging reference lists.")
        msisdn_col = cfg["data"].get("msisdn_col", "agent_msisdn")
        features_df = merge_reference_lists(
            df=features_df,
            whitelist_df=wl,
            blacklist_df=bl,
            msisdn_col=msisdn_col,
        )
    else:
        logger.info(
            "run_extrafloat_segmentation: step 5 — no reference lists provided, skipping."
        )

    # ── Step 6: Join MSISDN + Trim Output ────────────────────────────────────
    logger.info("run_extrafloat_segmentation: step 6 — output trimming.")
    features_df = _join_msisdn(features_df, df)

    keep_intermediate = cfg["output"].get("keep_intermediate_cols", False)
    result = _trim_output_columns(
        result=features_df,
        keep_intermediate=keep_intermediate,
        original_cols=original_cols,
    )

    # ── Optional: save current distributions as new baseline ─────────────────
    if drift_cfg.get("save_baseline") and drift_cfg.get("baseline_save_path"):
        try:
            save_drift_baseline(
                df=features_df,
                features=drift_cfg.get("drift_features", []),
                config=drift_cfg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "run_extrafloat_segmentation: failed to save drift baseline — %s.", exc
            )

    _maybe_save_output(result, cfg)

    n_segments = result[seg_col].nunique() if seg_col in result.columns else 0
    logger.info(
        "run_extrafloat_segmentation: pipeline complete — %d agents, "
        "%d distinct segments.",
        len(result),
        n_segments,
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: load from disk and run
# ─────────────────────────────────────────────────────────────────────────────


def run_extrafloat_segmentation_from_config(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load agent data from disk (config["data"]["agents_path"]) and run the
    full segmentation pipeline.

    Parameters
    ----------
    config :
        Full segmentation config dict.  ``config["data"]["agents_path"]``
        must be a valid path to a CSV file.

    Returns
    -------
    pd.DataFrame : Segmented agent DataFrame.
    """
    cfg = _get_config(config)
    agents_path = cfg["data"].get("agents_path", "")
    if not agents_path:
        raise ValueError(
            "run_extrafloat_segmentation_from_config: "
            "config['data']['agents_path'] is empty. "
            "Provide a valid CSV file path."
        )
    if not os.path.isfile(agents_path):
        raise ValueError(
            f"run_extrafloat_segmentation_from_config: "
            f"agents_path '{agents_path}' does not exist."
        )

    logger.info(
        "run_extrafloat_segmentation_from_config: loading agents from '%s'.",
        agents_path,
    )
    agents_df = pd.read_csv(agents_path)

    wl = _load_reference_list(cfg["data"].get("whitelist_path", ""), "whitelist")
    bl = _load_reference_list(cfg["data"].get("blacklist_path", ""), "blacklist")

    return run_extrafloat_segmentation(
        agents_df=agents_df,
        config=cfg,
        whitelist_df=wl,
        blacklist_df=bl,
    )

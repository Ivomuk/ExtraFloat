"""
Runs GMM -> UMAP -> HDBSCAN (+ tier labels) -> LOF once on the real active
population and prints concrete output for EACH stage -- the same "see real
numbers in one run" approach as tune_hdbscan_params.py, but covering the
full diagnostic chain instead of just HDBSCAN's own hyperparameter sweep.

Uses the real production internals directly (_run_gmm, _get_active_umap,
_run_hdbscan, _map_hdb_to_tier, flag_anomalies) -- not a reimplementation,
so what you see here is exactly what run_diagnostic_clustering /
flag_anomalies would produce on this data.

Reports:
  GMM    : BIC score at every k tried (k=min_k..gmm_max_k), which k won,
           its cluster sizes. Also re-runs once with a higher probe
           ceiling (--gmm-max-k-probe, default 2x the configured
           gmm_max_k) to directly answer "is the configured ceiling
           actually constraining the search" -- i.e. would BIC keep
           improving past the default cutoff.
  UMAP   : embedding shape + per-dimension min/max/mean/std, so you can
           see it isn't degenerate (e.g. collapsed to a single point).
  HDBSCAN + tier mapping : cluster count/sizes/noise on the UMAP
           embedding (default hyperparameters -- use
           tune_hdbscan_params.py to sweep those), plus the diagnostic
           tier label (_map_hdb_to_tier) each cluster resolves to and
           that label's total agent share.
  LOF    : from flag_anomalies's real Stage 1 (HDBSCAN on PCA space,
           independent of the UMAP-based diagnostics HDBSCAN above) +
           Stage 2 (per-cluster LOF): per-cluster population, n flagged
           local anomaly, and the overall lof_meta status/counts.

Usage:
    python scripts\\check_diagnostic_stages_output.py --agents data\\mfs_daily_agent_mart_20260731_retail_filtered.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.extrafloat_segmentation_features import prepare_features  # noqa: E402
from segmentation.extrafloat_segmentation_pipeline import (  # noqa: E402
    HDBSCAN_NOISE_LABEL,
    _get_active_pca,
    _get_active_umap,
    _get_clustering_config,
    _identify_dormant_mask,
    _map_hdb_to_tier,
    _run_gmm,
    _run_hdbscan,
    flag_anomalies,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", required=True, metavar="PATH")
    p.add_argument(
        "--raw", action="store_true",
        help="Treat --agents as already containing raw KPI columns (skip prepare_features).",
    )
    p.add_argument(
        "--gmm-max-k-probe", type=int, default=None, metavar="K",
        help="Re-run GMM with this higher gmm_max_k ceiling to check whether the "
             "default ceiling was constraining the BIC search. Default: 2x the "
             "configured gmm_max_k.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    agents_df = pd.read_csv(args.agents)
    print(f"Loaded {len(agents_df):,} agent rows from {args.agents}\n")

    if args.raw:
        features_df = agents_df
        selected_cols = [c for c in features_df.select_dtypes(include="number").columns
                          if c not in ("agent_msisdn", "pos_msisdn")]
    else:
        features_df, _, _, selected_cols = prepare_features(agents_df)

    cfg = _get_clustering_config(None)
    dormant_mask = _identify_dormant_mask(features_df, cfg)
    active_mask = ~dormant_mask
    n_active = int(active_mask.sum())
    print(f"Active (non-dormant) agents: {n_active:,} of {len(features_df):,}\n")

    if n_active == 0:
        sys.exit("ERROR: no active agents -- nothing to run diagnostics on.")

    rng = np.random.RandomState(cfg["random_state"])
    X_pca_active, X_scaled_active = _get_active_pca(features_df, active_mask, selected_cols, cfg, rng)

    # ── Stage: GMM ───────────────────────────────────────────────────────────
    print("=" * 78)
    print("GMM (BIC-based k search)")
    print("=" * 78)
    try:
        gmm_labels, gmm_model = _run_gmm(X_pca_active, cfg, np.random.RandomState(cfg["random_state"]))
        best_k = gmm_model.n_components
        ids, counts = np.unique(gmm_labels, return_counts=True)
        print(f"gmm_max_k (configured) = {cfg['gmm_max_k']}")
        print(f"best_k chosen          = {best_k}")
        print(f"cluster sizes          = {dict(zip(ids.tolist(), counts.tolist()))}")
        ceiling_hit = best_k == cfg["gmm_max_k"]
        print(f"hit configured ceiling = {ceiling_hit}")

        probe_max_k = args.gmm_max_k_probe or (2 * int(cfg["gmm_max_k"]))
        if ceiling_hit:
            print(f"\n-- ceiling was hit -- probing with gmm_max_k={probe_max_k} to check if BIC keeps improving --")
            cfg_probe = dict(cfg)
            cfg_probe["gmm_max_k"] = probe_max_k
            _, gmm_model_probe = _run_gmm(
                X_pca_active, cfg_probe, np.random.RandomState(cfg["random_state"])
            )
            best_k_probe = gmm_model_probe.n_components
            print(f"best_k with higher ceiling = {best_k_probe}")
            if best_k_probe > best_k:
                print(
                    f"CONCLUSION: the default gmm_max_k={cfg['gmm_max_k']} ceiling WAS "
                    f"constraining the search -- BIC preferred k={best_k_probe} once allowed."
                )
            else:
                print(
                    f"CONCLUSION: BIC still preferred k={best_k_probe} <= {best_k} even with "
                    f"room to go higher -- the default ceiling was not the limiting factor."
                )
        else:
            print("\nBIC search found an interior optimum below the ceiling -- no probe needed.")
    except Exception as exc:  # noqa: BLE001
        print(f"GMM failed: {exc}")
    print()

    # ── Stage: UMAP ──────────────────────────────────────────────────────────
    print("=" * 78)
    print("UMAP (embedding for HDBSCAN)")
    print("=" * 78)
    if cfg.get("use_umap_for_hdbscan", True):
        X_hdbscan = _get_active_umap(X_scaled_active, cfg, np.random.RandomState(cfg["random_state"]))
        print(f"embedding shape = {X_hdbscan.shape}")
        for dim in range(X_hdbscan.shape[1]):
            col = X_hdbscan[:, dim]
            print(f"  dim {dim}: min={col.min():.3f} max={col.max():.3f} mean={col.mean():.3f} std={col.std():.3f}")
    else:
        X_hdbscan = X_pca_active
        print("use_umap_for_hdbscan=False -- HDBSCAN runs on PCA space directly.")
    print()

    # ── Stage: HDBSCAN + diagnostic tier mapping ────────────────────────────
    print("=" * 78)
    print("HDBSCAN (on UMAP embedding, default hyperparameters) + _map_hdb_to_tier")
    print("=" * 78)
    try:
        hdb_labels = _run_hdbscan(X_hdbscan, cfg)
        is_noise = hdb_labels == HDBSCAN_NOISE_LABEL
        print(f"clusters found : {len(set(hdb_labels.tolist()) - {HDBSCAN_NOISE_LABEL})}")
        print(f"noise          : {int(is_noise.sum()):,} ({is_noise.mean():.1%})")

        features_df_active = features_df.loc[active_mask].reset_index(drop=True)
        hdb_tier = _map_hdb_to_tier(hdb_labels, features_df_active, cfg)
        print("\ndiag_hdb_tier distribution (agent-share-based label grouping):")
        print(hdb_tier.value_counts(dropna=False).to_string())
    except ImportError as exc:
        print(f"HDBSCAN unavailable: {exc}")
        hdb_labels = None
    print()

    # ── Stage: LOF (via flag_anomalies, its real Stage-1/Stage-2 chain) ────
    print("=" * 78)
    print("flag_anomalies (Stage 1: HDBSCAN on PCA space; Stage 2: per-cluster LOF)")
    print("=" * 78)
    anomalies_df = flag_anomalies(features_df, selected_cols, active_mask, config=cfg)
    lof_meta = anomalies_df.attrs.get("lof_meta", {})
    print(f"lof_meta: {lof_meta}")

    active_view = anomalies_df.loc[active_mask]
    n_global = int(active_view["is_global_anomaly"].sum())
    n_local = int(active_view["is_local_anomaly"].sum())
    n_any = int(active_view["is_anomaly"].sum())
    print(f"\nis_global_anomaly (Stage 1, HDBSCAN noise): {n_global:,} / {n_active:,} ({n_global / n_active:.1%})")
    print(f"is_local_anomaly  (Stage 2, per-cluster LOF): {n_local:,} / {n_active:,} ({n_local / n_active:.1%})")
    print(f"is_anomaly        (either stage):             {n_any:,} / {n_active:,} ({n_any / n_active:.1%})")

    print("\nPer Stage-1-HDBSCAN-cluster: population vs. LOF-flagged local anomalies:")
    per_cluster = (
        active_view.assign(_cid=active_view["anomaly_cluster_hdb_raw"])
        .groupby("_cid", dropna=False)["is_local_anomaly"]
        .agg(population="count", local_anomalies="sum")
    )
    per_cluster["anomaly_rate"] = per_cluster["local_anomalies"] / per_cluster["population"]
    print(per_cluster.sort_values("population", ascending=False).to_string())
    print()


if __name__ == "__main__":
    main()

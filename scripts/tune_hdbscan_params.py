"""
Sweeps HDBSCAN's min_cluster_size/min_samples against several candidates in
ONE run, reusing the real production internals (_get_active_pca,
_get_active_umap, _run_hdbscan from extrafloat_segmentation_pipeline.py --
not a reimplementation, so results match exactly what run_diagnostic_clustering
would produce for each candidate) rather than re-running the full,
expensive segmentation pipeline once per candidate.

Why this is efficient: feature engineering, the active/dormant split, PCA,
and the UMAP embedding HDBSCAN actually clusters on are ALL independent of
HDBSCAN's own hyperparameters -- they're computed exactly ONCE here (the
expensive part: single-threaded UMAP on hundreds of thousands of agents,
same as what you already watched take several minutes), then only the
cheap part (HDBSCAN.fit_predict on the same precomputed embedding) is
repeated per candidate.

Motivation: the default (min_cluster_size=1000, min_samples=150) was found
to collapse ~90%+ of active agents into one dominant cluster ("Platinum
Power") across every real capacity_tier, on the real retail-filtered
population -- not very informative for archetype research. This lets you
see, in one run, whether smaller (or larger) settings produce a more
differentiated clustering, before committing to a change in
DEFAULT_CLUSTERING_CONFIG.

Reports, per candidate (min_cluster_size, min_samples):
  - number of non-noise clusters found
  - noise (-1) count/percentage
  - the largest cluster's size as a percentage of ACTIVE (non-dormant)
    agents -- directly answers "does this candidate avoid the one-giant-
    cluster problem"
  - sizes of the top 5 largest clusters, for a fuller picture

Usage:
    python scripts\\tune_hdbscan_params.py --agents data\\mfs_daily_agent_mart_20260731_retail_filtered.csv

    REM Compare specific candidates instead of the built-in default sweep:
    python scripts\\tune_hdbscan_params.py --agents data\\mfs_daily_agent_mart_20260731_retail_filtered.csv ^
        --candidate 1000:150 --candidate 500:100 --candidate 250:50 --candidate 100:25
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.extrafloat_segmentation_features import prepare_features  # noqa: E402
from segmentation.extrafloat_segmentation_pipeline import (  # noqa: E402
    DEFAULT_CLUSTERING_CONFIG,
    HDBSCAN_NOISE_LABEL,
    _get_active_pca,
    _get_active_umap,
    _get_clustering_config,
    _identify_dormant_mask,
    _run_hdbscan,
)

DEFAULT_CANDIDATES = [
    (1000, 150),  # current DEFAULT_CLUSTERING_CONFIG (baseline)
    (500, 100),
    (250, 50),
    (100, 25),
    (5000, 500),  # larger, for the full trend picture
]


def _parse_candidate(raw: str) -> tuple[int, int]:
    try:
        mcs_str, ms_str = raw.split(":")
        return int(mcs_str), int(ms_str)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"--candidate must be MIN_CLUSTER_SIZE:MIN_SAMPLES (e.g. 500:100), got {raw!r}"
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", required=True, metavar="PATH")
    p.add_argument(
        "--raw", action="store_true",
        help="Treat --agents as already containing raw KPI columns (skip prepare_features).",
    )
    p.add_argument(
        "--candidate", type=_parse_candidate, action="append", default=None,
        metavar="MIN_CLUSTER_SIZE:MIN_SAMPLES",
        help="A (min_cluster_size, min_samples) pair to test. May be given multiple times. "
             "Omit for a built-in default sweep.",
    )
    args = p.parse_args(argv)

    agents_df = pd.read_csv(args.agents)
    print(f"Loaded {len(agents_df):,} agent rows from {args.agents}")

    if args.raw:
        features_df = agents_df
        selected_cols = [c for c in features_df.select_dtypes(include="number").columns
                          if c not in ("agent_msisdn", "pos_msisdn")]
    else:
        features_df, _, _, selected_cols = prepare_features(agents_df)

    cfg = _get_clustering_config(None)
    rng = np.random.RandomState(cfg["random_state"])

    dormant_mask = _identify_dormant_mask(features_df, cfg)
    active_mask = ~dormant_mask
    n_active = int(active_mask.sum())
    print(f"Active (non-dormant) agents: {n_active:,} of {len(features_df):,}\n")

    if n_active == 0:
        sys.exit("ERROR: no active agents -- nothing to cluster.")

    print("Computing PCA + UMAP embedding ONCE (shared across all candidates below)...")
    X_pca_active, X_scaled_active = _get_active_pca(features_df, active_mask, selected_cols, cfg, rng)
    if cfg.get("use_umap_for_hdbscan", True):
        X_hdbscan = _get_active_umap(X_scaled_active, cfg, rng)
    else:
        X_hdbscan = X_pca_active
    print("Embedding ready -- sweeping HDBSCAN candidates (cheap from here on):\n")

    candidates = args.candidate or DEFAULT_CANDIDATES

    for min_cluster_size, min_samples in candidates:
        candidate_cfg = _get_clustering_config({
            "hdbscan_min_cluster_size": min_cluster_size,
            "hdbscan_min_samples": min_samples,
        })
        try:
            labels = _run_hdbscan(X_hdbscan, candidate_cfg)
        except ImportError as exc:
            sys.exit(f"ERROR: {exc}")

        is_noise = labels == HDBSCAN_NOISE_LABEL
        n_noise = int(is_noise.sum())
        non_noise_labels = labels[~is_noise]
        cluster_ids, cluster_sizes = np.unique(non_noise_labels, return_counts=True)
        n_clusters = len(cluster_ids)
        order = np.argsort(cluster_sizes)[::-1]
        top_sizes = cluster_sizes[order][:5]
        largest_pct = (top_sizes[0] / n_active * 100) if len(top_sizes) else 0.0

        print(f"=== min_cluster_size={min_cluster_size}, min_samples={min_samples} ===")
        print(f"  clusters found      : {n_clusters}")
        print(f"  noise               : {n_noise:,} ({n_noise / n_active:.1%})")
        print(f"  largest cluster     : {top_sizes[0]:,} agents ({largest_pct:.1f}% of active) " if len(top_sizes) else "  largest cluster     : n/a")
        print(f"  top 5 cluster sizes : {[int(s) for s in top_sizes]}")
        print()


if __name__ == "__main__":
    main()

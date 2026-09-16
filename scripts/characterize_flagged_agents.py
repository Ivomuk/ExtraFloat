"""
Characterizes two agent subsets found during diagnostics this session, by
comparing each subset's mean KPI values against the overall active
population's mean (as a ratio and a z-score), so the columns that actually
distinguish the subset surface to the top rather than requiring a manual
column-by-column comparison:

  1. The recurring ~2,700-3,400-agent GMM minority subgroup -- this exact
     size cluster showed up independently in GMM-full (3,416 agents),
     GMM-diag (2,747 agents), and the PCA-based HDBSCAN anomaly Stage 1
     (2,747 agents, which also had the highest local-anomaly rate at
     10.2% of any cluster) -- three different methods converging on a
     similarly-sized group is a much stronger signal than any one alone.
     Identified here by re-running the real _run_gmm (cov='full', the
     configured default) and picking the largest non-dominant, non-
     degenerate cluster (excludes the >50%-share dominant cluster and any
     cluster under --min-cluster-size, default 50, which are almost
     certainly single-point outlier-fitting artifacts, not real groups).

  2. The is_anomaly=True agents from a real segmentation run (global and
     local anomalies profiled separately, since they're different
     mechanisms -- global = doesn't fit any dense cluster at all, local =
     fits a cluster but stands out within it).

Uses the real production internals (_get_active_pca, _run_gmm,
_identify_dormant_mask) -- not a reimplementation.

Usage:
    python scripts\\characterize_flagged_agents.py ^
        --agents data\\mfs_daily_agent_mart_20260731_retail_filtered.csv ^
        --segments segmentation_outputs\\agent_segments.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.extrafloat_segmentation_features import prepare_features  # noqa: E402
from segmentation.extrafloat_segmentation_pipeline import (  # noqa: E402
    _get_active_pca,
    _get_clustering_config,
    _identify_dormant_mask,
    _run_gmm,
)

TOP_N = 20


def _profile_subset(
    label: str,
    subset_mask: pd.Series,
    active_features: pd.DataFrame,
    cols: list[str],
) -> None:
    n = int(subset_mask.sum())
    n_active = len(active_features)
    print(f"\n=== {label}: {n:,} agents ({n / n_active:.1%} of {n_active:,} active) ===")
    if n == 0:
        print("  (empty subset -- nothing to profile)")
        return

    rows = []
    for col in cols:
        vals = active_features[col].fillna(0.0).astype(float)
        subset_vals = vals[subset_mask]
        active_mean = vals.mean()
        active_std = vals.std()
        subset_mean = subset_vals.mean()
        ratio = (subset_mean / active_mean) if active_mean not in (0, np.nan) else np.nan
        zscore = ((subset_mean - active_mean) / active_std) if active_std > 0 else np.nan
        rows.append({
            "column": col,
            "subset_mean": subset_mean,
            "active_mean": active_mean,
            "ratio": ratio,
            "zscore": zscore,
        })

    profile = pd.DataFrame(rows).sort_values("zscore", key=lambda s: s.abs(), ascending=False)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(profile.head(TOP_N).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", required=True, metavar="PATH")
    p.add_argument("--segments", metavar="PATH",
                   help="agent_segments.csv from a real run -- enables the is_anomaly profiling. "
                        "Omit to only profile the GMM minority subgroup.")
    p.add_argument("--min-cluster-size", type=int, default=50, metavar="N",
                   help="GMM clusters smaller than this are treated as degenerate outlier-fitting "
                        "artifacts, not real subgroups, and excluded from consideration (default 50).")
    args = p.parse_args(argv)

    agents_df = pd.read_csv(args.agents)
    print(f"Loaded {len(agents_df):,} agent rows from {args.agents}")

    features_df, _, _, selected_cols = prepare_features(agents_df)
    cfg = _get_clustering_config(None)
    dormant_mask = _identify_dormant_mask(features_df, cfg)
    active_mask = ~dormant_mask
    active_features = features_df.loc[active_mask].reset_index(drop=True)
    n_active = len(active_features)
    print(f"Active (non-dormant) agents: {n_active:,} of {len(features_df):,}")

    rng = np.random.RandomState(cfg["random_state"])
    X_pca_active, _ = _get_active_pca(features_df, active_mask, selected_cols, cfg, rng)

    # ── Part 1: the recurring GMM minority subgroup ─────────────────────────
    print("\n" + "=" * 78)
    print("PART 1: recurring GMM minority subgroup (cov='full', configured default)")
    print("=" * 78)
    gmm_labels, gmm_model = _run_gmm(X_pca_active, cfg, np.random.RandomState(cfg["random_state"]))
    ids, counts = np.unique(gmm_labels, return_counts=True)
    sizes = dict(zip(ids.tolist(), counts.tolist()))
    print(f"Cluster sizes: {sizes}")

    dominant_id = max(sizes, key=sizes.get)
    candidates = {cid: n for cid, n in sizes.items() if cid != dominant_id and n >= args.min_cluster_size}
    if not candidates:
        print("No non-dominant, non-degenerate cluster found -- nothing to profile for Part 1.")
    else:
        minority_id = max(candidates, key=candidates.get)
        minority_mask = pd.Series(gmm_labels == minority_id, index=active_features.index)
        print(f"Selected cluster {minority_id} ({sizes[minority_id]:,} agents) as the minority subgroup "
              f"(excluded dominant cluster {dominant_id}, {sizes[dominant_id]:,} agents, "
              f"and any cluster under {args.min_cluster_size} agents).")
        _profile_subset(
            f"GMM cluster {minority_id}", minority_mask, active_features, selected_cols,
        )

    # ── Part 2: is_anomaly agents from a real run ───────────────────────────
    if args.segments:
        print("\n" + "=" * 78)
        print("PART 2: is_anomaly agents (from --segments)")
        print("=" * 78)
        segments_path = Path(args.segments)
        if not segments_path.exists():
            sys.exit(f"ERROR: file not found: {segments_path}")
        segments_df = pd.read_csv(segments_path)

        merge_keys = [c for c in ("agent_msisdn", "pos_msisdn")
                      if c in segments_df.columns and c in agents_df.columns]
        if not merge_keys:
            sys.exit("ERROR: no shared key columns between --segments and --agents.")

        # Align segments_df to features_df's row order via the same keys,
        # then subset to active agents so masks line up with active_features.
        anomaly_cols = ["is_anomaly", "is_global_anomaly", "is_local_anomaly"]
        key_df = agents_df[merge_keys].reset_index(drop=True)
        aligned = key_df.merge(
            segments_df[merge_keys + anomaly_cols], on=merge_keys, how="left"
        )
        for col in anomaly_cols:
            aligned[col] = aligned[col].fillna(False).astype(bool)
        aligned_active = aligned.loc[active_mask.values].reset_index(drop=True)

        _profile_subset(
            "is_global_anomaly agents", aligned_active["is_global_anomaly"], active_features, selected_cols,
        )
        _profile_subset(
            "is_local_anomaly agents", aligned_active["is_local_anomaly"], active_features, selected_cols,
        )
    else:
        print("\n(--segments not given -- skipping Part 2, the is_anomaly agent profile.)")


if __name__ == "__main__":
    main()

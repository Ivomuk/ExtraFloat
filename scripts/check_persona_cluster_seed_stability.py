"""
Multi-seed stability check for the borrower-persona HDBSCAN candidates that
cleared the dominant-cluster/noise/cluster-count guards in
segmentation/borrower_persona_clustering.py's candidate sweep.

Why this exists: the 160-260 min_cluster_size sweep at umap_min_dist=0.0
found only isolated points (min_cluster_size=100, 160, 200) that satisfy
all three hard bounds, surrounded on both sides by neighbors that fail at
~60%+ dominant-cluster share. That's consistent with either (a) a narrow
but real parameter region, or (b) a result that's sensitive to UMAP's own
stochastic embedding rather than reflecting stable structure in the data --
this script distinguishes the two by re-running each candidate across
several random seeds and reporting how much n_clusters/dominant_share move.

Only HDBSCAN's UMAP input varies with the seed (HDBSCAN itself is
deterministic given its input); flag_anomalies derives PCA's and UMAP's
own random states from cfg["random_state"] via a single np.random.RandomState,
so overriding that one config key is sufficient to vary the embedding.

Reuses load_and_join()/build_features()/evaluate_cluster_labels() from
borrower_persona_clustering.py directly (not a reimplementation), and
replicates main()'s own Step 3 (log-transform + correlation pruning) so
df_features here is built exactly the same way the real pipeline builds it.

Usage:
    python scripts\\check_persona_cluster_seed_stability.py
    python scripts\\check_persona_cluster_seed_stability.py --seeds 42 7 123 2024 31337
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from segmentation.borrower_persona_clustering import (  # noqa: E402
    build_features,
    evaluate_cluster_labels,
    load_and_join,
)
from segmentation.extrafloat_segmentation_features import (  # noqa: E402
    _apply_log_winsorize,
    _get_features_config,
    _prune_correlated_features,
)
from segmentation.extrafloat_segmentation_pipeline import (  # noqa: E402
    _get_clustering_config,
    flag_anomalies,
)

DEFAULT_SEEDS = [42, 7, 123, 2024, 31337]

# The three isolated candidates that cleared all three hard bounds in the
# real 160-260 sweep -- the only ones where instability vs. genuine signal
# is actually in question (every neighboring MCS value failed outright).
CANDIDATES = [
    ("cs100", {"hdbscan_min_cluster_size": 100, "hdbscan_min_samples": 15, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
    ("cs160", {"hdbscan_min_cluster_size": 160, "hdbscan_min_samples": 24, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
    ("cs200", {"hdbscan_min_cluster_size": 200, "hdbscan_min_samples": 30, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help=f"random_state values to test per candidate (default: {DEFAULT_SEEDS})")
    args = p.parse_args(argv)

    print("=== Load + build features (same as borrower_persona_clustering.py Steps 1-3) ===")
    df = load_and_join()
    feat, all_cols = build_features(df)
    numeric_cols = [c for c in all_cols if not c.startswith("has_")]
    feat_cfg = _get_features_config(None)
    X_log = _apply_log_winsorize(feat[numeric_cols].copy(), feat_cfg)
    X_pruned, selected_cols = _prune_correlated_features(X_log, feat_cfg)
    X_final = X_pruned.fillna(X_pruned.median()).fillna(0.0)
    df_features = X_final.copy()
    active_mask = pd.Series(True, index=df_features.index)
    n_active = int(active_mask.sum())
    print(f"  {n_active:,} active agents, {len(selected_cols)} features")

    print(f"\n=== Seed stability check: {len(CANDIDATES)} candidates x {len(args.seeds)} seeds "
          f"= {len(CANDIDATES) * len(args.seeds)} runs ===")
    rows = []
    for name, base_overrides in CANDIDATES:
        for seed in args.seeds:
            overrides = dict(base_overrides, random_state=seed)
            cfg = _get_clustering_config(overrides)
            out = flag_anomalies(df_features, selected_cols, active_mask, config=cfg)
            labels = out["anomaly_cluster_hdb_raw"]
            noise_pct, n_clusters, dominant_share, top5_share_pct, n_tiny, tiny_pop_pct = \
                evaluate_cluster_labels(labels, n_active)
            print(f"  {name} seed={seed}: n_clusters={n_clusters}, noise={noise_pct:.1f}%, "
                  f"largest_cluster={dominant_share:.1%}, top5={top5_share_pct}%")
            rows.append({
                "candidate": name,
                "seed": seed,
                "n_clusters": n_clusters,
                "noise_pct": round(noise_pct, 1),
                "dominant_pct": round(dominant_share * 100, 1),
                "top5_pct": top5_share_pct,
                "n_tiny_clusters": n_tiny,
                "tiny_clusters_pop_pct": tiny_pop_pct,
            })

    results_df = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent.parent / "segmentation_outputs" / "persona_cluster_seed_stability.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\n  wrote {out_path} ({len(results_df)} rows)")

    print("\n=== Per-candidate stability summary (across seeds) ===")
    summary = results_df.groupby("candidate").agg(
        n_clusters_min=("n_clusters", "min"),
        n_clusters_max=("n_clusters", "max"),
        n_clusters_std=("n_clusters", "std"),
        dominant_pct_min=("dominant_pct", "min"),
        dominant_pct_max=("dominant_pct", "max"),
        dominant_pct_std=("dominant_pct", "std"),
        noise_pct_min=("noise_pct", "min"),
        noise_pct_max=("noise_pct", "max"),
    ).round(2)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary.to_string())

    print(
        "\nReading this: a STABLE candidate should show a narrow "
        "dominant_pct_min-max range comfortably under 50% across every "
        "seed. If dominant_pct swings from, say, 35% at one seed to 85% "
        "at another for the same candidate, that candidate's apparent "
        "pass in the single-seed sweep was likely a coincidence of that "
        "one seed's UMAP embedding, not a real, reproducible structure in "
        "the data -- treat it as unstable rather than as a foundation for "
        "personas."
    )


if __name__ == "__main__":
    main()

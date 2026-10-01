"""
PCA + K-means/GMM benchmark for borrower persona segmentation -- the
deliberate pivot away from UMAP+HDBSCAN as the primary clustering approach.

Why this exists: a 5-seed stability check on the 3 HDBSCAN candidates that
cleared every guard in segmentation/borrower_persona_clustering.py's search
(min_cluster_size=100/160/200, all at umap_n_neighbors=10, umap_min_dist=0.0)
found the dominant-cluster share swinging from ~20% to ~85% across seeds for
every one of them -- the apparent "pass" at the default seed was luck of the
draw on UMAP's own stochastic embedding, not reproducible structure. A
persona system needs much more stability than that: the same agent
shouldn't become a different "type" just because the embedding seed changed.

This does NOT replace or modify borrower_persona_clustering.py -- that
UMAP+HDBSCAN result is kept as a documented experimental finding (fine-
grained density clustering was investigated and rejected for production
personas because cluster structure was highly sensitive to UMAP
initialization), not deleted.

This script instead asks a different question than HDBSCAN's "how many
density islands exist": at what granularity can we get distinct,
interpretable, REPRODUCIBLE personas via a deterministic PCA representation
(no UMAP, no embedding stochasticity) feeding two independent clustering
methods (K-means, GMM)? If both independently agree at some K, that's much
stronger evidence of real structure than anything the HDBSCAN sweep found.

Reuses load_and_join()/build_features() from borrower_persona_clustering.py
directly, and the same log-transform + correlation-pruning Step 3 that
script uses -- same clustering inputs (capacity + credit engagement + credit
quality + justified continuous behavioural features), same discipline of
keeping borrower_trend/borrower_profile_type OUT of clustering and reserved
for post-clustering profiling.

Part A prints PCA diagnostics (explained variance, loadings, PC1-3 shape)
and saves a plot BEFORE any clustering, so the component structure can be
inspected on its own merits. Part B benchmarks K-means and GMM across
K=3,4,5,6,7,8,10,12,15 on four axes: statistical quality (silhouette,
Davies-Bouldin, Calinski-Harabasz, BIC/AIC for GMM), population balance
(cluster-size distribution), stability (ARI/NMI across repeated fits with
different random seeds), and cross-method agreement (ARI/NMI between
K-means and GMM at the same K -- the strongest signal if it's high).
Deliberately does NOT auto-select a winning K or impose the HDBSCAN
pipeline's 50%-dominant-share rule -- "stable + differentiated +
interpretable" is the goal, not mathematically balanced clusters. Business-
meaning profiling (capacity/credit-quality/trend/outcome cross-tabs) is left
for a follow-up step once this benchmark narrows down which K values are
even worth profiling.

Usage:
    python scripts\\benchmark_persona_pca_clustering.py
    python scripts\\benchmark_persona_pca_clustering.py --k-values 4 5 6 8 --stability-seeds 42 7 123
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from segmentation.borrower_persona_clustering import build_features, load_and_join  # noqa: E402
from segmentation.extrafloat_segmentation_features import (  # noqa: E402
    _apply_log_winsorize,
    _get_features_config,
    _prune_correlated_features,
)

DEFAULT_K_VALUES = [3, 4, 5, 6, 7, 8, 10, 12, 15]
DEFAULT_STABILITY_SEEDS = [42, 7, 123, 2024, 31337]
RANDOM_STATE = 42
N_INIT = 20  # KMeans/GMM restarts per fit -- a real stability benchmark needs
             # more than sklearn's own default (10) to trust a single fit's result.

OUT_DIR = Path(__file__).resolve().parent.parent / "segmentation_outputs" / "persona_pca_benchmark"


def _cluster_balance(labels: np.ndarray) -> tuple[list[float], float, float]:
    """Return (sorted cluster-size percentages desc, max%, min%)."""
    n = len(labels)
    sizes = pd.Series(labels).value_counts()
    pct = (sizes / n * 100).round(1).sort_values(ascending=False).tolist()
    return pct, pct[0], pct[-1]


def _fit_kmeans(X: np.ndarray, k: int, random_state: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=random_state).fit_predict(X)


def _fit_gmm(X: np.ndarray, k: int, random_state: int) -> tuple[np.ndarray, GaussianMixture]:
    gmm = GaussianMixture(
        n_components=k, covariance_type="full", n_init=max(1, N_INIT // 4),
        random_state=random_state, reg_covar=1e-4,
    )
    labels = gmm.fit_predict(X)
    return labels, gmm


def _stability_ari_nmi(X: np.ndarray, k: int, method: str, seeds: list[int]) -> tuple[float, float, float, float]:
    """Refit at each seed, return (mean_ari, min_ari, mean_nmi, min_nmi) across all seed pairs."""
    labelings = []
    for seed in seeds:
        if method == "kmeans":
            labelings.append(_fit_kmeans(X, k, seed))
        else:
            labels, _ = _fit_gmm(X, k, seed)
            labelings.append(labels)
    aris, nmis = [], []
    for a, b in combinations(labelings, 2):
        aris.append(adjusted_rand_score(a, b))
        nmis.append(normalized_mutual_info_score(a, b))
    return float(np.mean(aris)), float(np.min(aris)), float(np.mean(nmis)), float(np.min(nmis))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    p.add_argument("--stability-seeds", type=int, nargs="+", default=DEFAULT_STABILITY_SEEDS)
    args = p.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Load + build features (same inputs as borrower_persona_clustering.py) ===")
    df = load_and_join()
    feat, all_cols = build_features(df)
    numeric_cols = [c for c in all_cols if not c.startswith("has_")]
    feat_cfg = _get_features_config(None)
    X_log = _apply_log_winsorize(feat[numeric_cols].copy(), feat_cfg)
    X_pruned, selected_cols = _prune_correlated_features(X_log, feat_cfg)
    X_final = X_pruned.fillna(X_pruned.median()).fillna(0.0)
    n_active = len(X_final)
    print(f"  {n_active:,} agents, {len(selected_cols)} clustering features: {selected_cols}")

    # ── Part A: PCA diagnostics, inspected BEFORE any clustering ───────────
    print("\n" + "=" * 78)
    print("PART A: PCA diagnostics")
    print("=" * 78)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_final.astype(float))

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)
    explained = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(explained)

    print("\nExplained variance by component:")
    for i, (ev, cv) in enumerate(zip(explained, cum_var), 1):
        print(f"  PC{i}: {ev:.1%} (cumulative {cv:.1%})")

    target_variance = feat_cfg["target_pca_variance"]
    n_components = int(np.argmax(cum_var >= target_variance) + 1)
    print(f"\n  -> {n_components} components reach the repo's {target_variance:.0%} "
          f"variance target (used for clustering in Part B)")

    n_loading_pcs = min(5, len(selected_cols))
    loadings = pd.DataFrame(
        pca_full.components_[:n_loading_pcs].T,
        index=selected_cols,
        columns=[f"PC{i + 1}" for i in range(n_loading_pcs)],
    )
    print(f"\nTop-5 loadings per component (first {n_loading_pcs} PCs):")
    for pc in loadings.columns:
        top = loadings[pc].abs().sort_values(ascending=False).head(5)
        print(f"  {pc}:")
        for feat_name in top.index:
            print(f"    {feat_name:45s} {loadings.loc[feat_name, pc]:+.3f}")

    pca_coords_full = pca_full.transform(X_scaled)
    print("\nPC1-3 distribution (percentiles):")
    for i in range(min(3, pca_coords_full.shape[1])):
        v = pca_coords_full[:, i]
        print(f"  PC{i + 1}: min={v.min():.2f} p10={np.percentile(v,10):.2f} "
              f"p25={np.percentile(v,25):.2f} median={np.median(v):.2f} "
              f"p75={np.percentile(v,75):.2f} p90={np.percentile(v,90):.2f} max={v.max():.2f} "
              f"skew={pd.Series(v).skew():.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        ax = axes[0, 0]
        n_scree = min(20, len(explained))
        ax.bar(range(1, n_scree + 1), explained[:n_scree] * 100, color="#4C72B0", alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(range(1, n_scree + 1), cum_var[:n_scree] * 100, color="#C44E52", marker="o", markersize=3)
        ax2.axhline(target_variance * 100, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained variance %")
        ax2.set_ylabel("Cumulative %")
        ax.set_title("Scree plot")

        ax = axes[0, 1]
        subsample = np.random.RandomState(RANDOM_STATE).choice(
            n_active, size=min(50_000, n_active), replace=False
        )
        ax.scatter(pca_coords_full[subsample, 0], pca_coords_full[subsample, 1],
                   s=3, alpha=0.3, linewidths=0, color="#4C72B0")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PC1 vs PC2 ({len(subsample):,} agents)")

        for idx, pc_i in enumerate([0, 1]):
            ax = axes[1, idx]
            ax.hist(pca_coords_full[:, pc_i], bins=80, color="#55A868", alpha=0.85)
            ax.set_xlabel(f"PC{pc_i + 1}")
            ax.set_ylabel("Count")
            ax.set_title(f"PC{pc_i + 1} distribution (multimodality check)")

        plt.tight_layout()
        plot_path = OUT_DIR / "pca_overview.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  wrote {plot_path}")
    except ImportError:
        print("\n  WARNING: matplotlib not installed -- skipping pca_overview.png "
              "(pip install matplotlib to get it; numeric diagnostics above are unaffected)")

    X_pca = pca_coords_full[:, :n_components]

    # ── Part B: K-means / GMM benchmark across K values ─────────────────────
    print("\n" + "=" * 78)
    print(f"PART B: K-means/GMM benchmark, K={args.k_values}, "
          f"{len(args.stability_seeds)} stability seeds")
    print("=" * 78)

    rows = []
    for k in args.k_values:
        print(f"\n-- K={k} --")
        kmeans_labels = _fit_kmeans(X_pca, k, RANDOM_STATE)
        gmm_labels, gmm_model = _fit_gmm(X_pca, k, RANDOM_STATE)

        kmeans_sil = silhouette_score(X_pca, kmeans_labels, sample_size=min(20_000, n_active), random_state=RANDOM_STATE)
        kmeans_db = davies_bouldin_score(X_pca, kmeans_labels)
        kmeans_ch = calinski_harabasz_score(X_pca, kmeans_labels)
        kmeans_pct, kmeans_max_pct, kmeans_min_pct = _cluster_balance(kmeans_labels)

        gmm_sil = silhouette_score(X_pca, gmm_labels, sample_size=min(20_000, n_active), random_state=RANDOM_STATE)
        gmm_db = davies_bouldin_score(X_pca, gmm_labels)
        gmm_ch = calinski_harabasz_score(X_pca, gmm_labels)
        gmm_pct, gmm_max_pct, gmm_min_pct = _cluster_balance(gmm_labels)
        gmm_bic = gmm_model.bic(X_pca)
        gmm_aic = gmm_model.aic(X_pca)

        kmeans_ari_mean, kmeans_ari_min, kmeans_nmi_mean, kmeans_nmi_min = _stability_ari_nmi(
            X_pca, k, "kmeans", args.stability_seeds
        )
        gmm_ari_mean, gmm_ari_min, gmm_nmi_mean, gmm_nmi_min = _stability_ari_nmi(
            X_pca, k, "gmm", args.stability_seeds
        )
        cross_ari = adjusted_rand_score(kmeans_labels, gmm_labels)
        cross_nmi = normalized_mutual_info_score(kmeans_labels, gmm_labels)

        print(f"  KMeans : silhouette={kmeans_sil:.3f} davies_bouldin={kmeans_db:.3f} "
              f"calinski_harabasz={kmeans_ch:.0f} sizes%={kmeans_pct}")
        print(f"           stability ARI mean/min={kmeans_ari_mean:.3f}/{kmeans_ari_min:.3f}, "
              f"NMI mean/min={kmeans_nmi_mean:.3f}/{kmeans_nmi_min:.3f}")
        print(f"  GMM    : silhouette={gmm_sil:.3f} davies_bouldin={gmm_db:.3f} "
              f"calinski_harabasz={gmm_ch:.0f} bic={gmm_bic:.0f} aic={gmm_aic:.0f} sizes%={gmm_pct}")
        print(f"           stability ARI mean/min={gmm_ari_mean:.3f}/{gmm_ari_min:.3f}, "
              f"NMI mean/min={gmm_nmi_mean:.3f}/{gmm_nmi_min:.3f}")
        print(f"  Cross-method agreement (KMeans vs GMM, same K): ARI={cross_ari:.3f} NMI={cross_nmi:.3f}")

        rows.append({
            "k": k, "method": "kmeans", "silhouette": round(kmeans_sil, 3),
            "davies_bouldin": round(kmeans_db, 3), "calinski_harabasz": round(kmeans_ch, 1),
            "bic": None, "aic": None,
            "max_cluster_pct": kmeans_max_pct, "min_cluster_pct": kmeans_min_pct,
            "cluster_sizes_pct": kmeans_pct,
            "stability_ari_mean": round(kmeans_ari_mean, 3), "stability_ari_min": round(kmeans_ari_min, 3),
            "stability_nmi_mean": round(kmeans_nmi_mean, 3), "stability_nmi_min": round(kmeans_nmi_min, 3),
            "cross_method_ari": round(cross_ari, 3), "cross_method_nmi": round(cross_nmi, 3),
        })
        rows.append({
            "k": k, "method": "gmm", "silhouette": round(gmm_sil, 3),
            "davies_bouldin": round(gmm_db, 3), "calinski_harabasz": round(gmm_ch, 1),
            "bic": round(gmm_bic, 1), "aic": round(gmm_aic, 1),
            "max_cluster_pct": gmm_max_pct, "min_cluster_pct": gmm_min_pct,
            "cluster_sizes_pct": gmm_pct,
            "stability_ari_mean": round(gmm_ari_mean, 3), "stability_ari_min": round(gmm_ari_min, 3),
            "stability_nmi_mean": round(gmm_nmi_mean, 3), "stability_nmi_min": round(gmm_nmi_min, 3),
            "cross_method_ari": round(cross_ari, 3), "cross_method_nmi": round(cross_nmi, 3),
        })

    results_df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "kmeans_gmm_benchmark.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv} ({len(results_df)} rows)")

    print("\n=== Summary table (statistical quality + stability + cross-method agreement) ===")
    display_cols = ["k", "method", "silhouette", "davies_bouldin", "calinski_harabasz",
                     "bic", "aic", "max_cluster_pct", "min_cluster_pct",
                     "stability_ari_mean", "stability_ari_min", "cross_method_ari"]
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(results_df[display_cols].to_string(index=False))

    print(
        "\nReading this: no K is auto-selected here on purpose -- 'stable + "
        "differentiated + interpretable' is the goal, not a single metric or "
        "a balanced-cluster rule. Look for K values where (a) stability_ari_mean "
        "is high (close to 1.0) for BOTH methods -- a candidate that's unstable "
        "even on its own repeated fits can't be trusted regardless of anything "
        "else -- and (b) cross_method_ari is also high, meaning K-means and GMM "
        "independently agree on the same partition from different geometric "
        "assumptions. That combination is much stronger evidence of real "
        "structure than any single statistical-quality metric alone. Business "
        "interpretability (capacity/credit-quality/trend profiles) is the next "
        "step once a short list of candidate K values comes out of this table, "
        "not computed here."
    )


if __name__ == "__main__":
    main()

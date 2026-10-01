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
(no UMAP, no embedding stochasticity)?

Reuses load_and_join()/build_features() from borrower_persona_clustering.py
directly, and the same log-transform + correlation-pruning Step 3 that
script uses -- same clustering inputs (capacity + credit engagement + credit
quality + justified continuous behavioural features), same discipline of
keeping borrower_trend/borrower_profile_type OUT of clustering and reserved
for post-clustering profiling.

Part A prints PCA diagnostics (explained variance, loadings, PC1-3 shape)
and saves a plot BEFORE any clustering. It also runs a KDE-based modality
analysis on PC1-3: real clusters should show density VALLEYS between modes,
not just a spread-out distribution -- 72.4% of variance sitting in 3
components means 3 dominant DIRECTIONS of variation, not 3 clusters, and a
population that's genuinely continuous along those directions (e.g. low to
high capacity, with no natural break) will make every clustering algorithm
disagree about where to draw boundaries, however internally consistent each
one is on its own.

Part B benchmarks K-means against three GMM covariance types (full, diag,
spherical) across K=3,4,5,6,7,8,10,12,15. A real run found K-means highly
reproducible (ARI ~0.99 at K=3-8) while full-covariance GMM mostly agreed
with itself too, yet the two methods only ever reached moderate agreement
with each other (ARI 0.26-0.52, never approaching 1.0) -- and GMM-full
consistently found one large elliptical component where K-means split into
two comparable groups. That's a specific, testable hypothesis: does
agreement with K-means rise as GMM's covariance assumption is constrained
toward K-means' own (spherical, equal-variance)? diag and spherical GMM
variants test exactly that, instead of adding another unrelated algorithm.

For each (K, method): statistical quality (silhouette, Davies-Bouldin,
Calinski-Harabasz, BIC/AIC for GMM variants), population balance
(cluster-size distribution), within-method stability (ARI/NMI across
repeated fits with different random seeds), and cross-method agreement vs.
K-means specifically (not all-pairs -- the diagnostic question is about
each GMM variant's relationship to K-means as geometry constrains). Does
NOT auto-select a winning K, and does NOT require cross-method ARI close to
1.0 to trust a result -- different algorithms have different objectives, so
strong agreement is excellent evidence when it occurs, but its absence
isn't automatic evidence of an invalid clustering. Within-method stability +
interpretable separation + business meaning matter more for a production
persona system. Business-meaning profiling (capacity/credit-quality/
trend/outcome cross-tabs) is left for a follow-up step.

Usage:
    python scripts\\benchmark_persona_pca_clustering.py
    python scripts\\benchmark_persona_pca_clustering.py --k-values 4 6 --stability-seeds 42 7 123
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
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

from segmentation.borrower_persona_clustering import (  # noqa: E402
    build_features,
    load_and_join,
    quantile_normalize_all_columns,
)
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
GMM_COVARIANCE_TYPES = ["full", "diag", "spherical"]
# A valley whose density is above this fraction of its smaller neighboring
# peak isn't a real separation, just a shoulder on an otherwise continuous
# distribution. Printed alongside the raw ratio either way so this threshold
# is a labeling convenience, not a hidden cutoff -- judge the actual numbers.
VALLEY_REAL_THRESHOLD = 0.8

OUT_DIR = Path(__file__).resolve().parent.parent / "segmentation_outputs" / "persona_pca_benchmark"


def _cluster_balance(labels: np.ndarray) -> tuple[list[float], float, float]:
    """Return (sorted cluster-size percentages desc, max%, min%)."""
    n = len(labels)
    sizes = pd.Series(labels).value_counts()
    pct = (sizes / n * 100).round(1).sort_values(ascending=False).tolist()
    return pct, pct[0], pct[-1]


def _fit_kmeans(X: np.ndarray, k: int, random_state: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=random_state).fit_predict(X)


def _fit_gmm(X: np.ndarray, k: int, random_state: int, covariance_type: str) -> tuple[np.ndarray, GaussianMixture]:
    gmm = GaussianMixture(
        n_components=k, covariance_type=covariance_type, n_init=max(1, N_INIT // 4),
        random_state=random_state, reg_covar=1e-4,
    )
    labels = gmm.fit_predict(X)
    return labels, gmm


def _stability_ari_nmi(fit_fn, seeds: list[int]) -> tuple[float, float, float, float]:
    """fit_fn(seed) -> labels. Refit at each seed, return (mean_ari, min_ari,
    mean_nmi, min_nmi) across all seed pairs."""
    labelings = [fit_fn(seed) for seed in seeds]
    aris, nmis = [], []
    for a, b in combinations(labelings, 2):
        aris.append(adjusted_rand_score(a, b))
        nmis.append(normalized_mutual_info_score(a, b))
    return float(np.mean(aris)), float(np.min(aris)), float(np.mean(nmis)), float(np.min(nmis))


def _analyze_modality(values: np.ndarray, grid_size: int = 512) -> dict:
    """KDE-based mode/valley detection for one PC's distribution.

    A real sub-population boundary shows up as a density VALLEY between two
    peaks -- not just "the distribution is spread out". For each valley,
    depth_ratio = valley density / the SMALLER of its two neighboring peaks'
    density: close to 0 means a deep, real separation; close to 1 means the
    "valley" barely dips below its neighbors, i.e. no real boundary there,
    just a continuous spread (consistent with a capacity/engagement gradient
    rather than discrete populations).
    """
    kde = gaussian_kde(values)
    grid = np.linspace(values.min(), values.max(), grid_size)
    density = kde(grid)
    peak_idx = list(argrelextrema(density, np.greater)[0])
    valley_idx = list(argrelextrema(density, np.less)[0])

    valleys = []
    for vi in valley_idx:
        left_peaks = [pi for pi in peak_idx if pi < vi]
        right_peaks = [pi for pi in peak_idx if pi > vi]
        if not left_peaks or not right_peaks:
            continue
        neighbor_min = min(density[max(left_peaks)], density[min(right_peaks)])
        depth_ratio = float(density[vi] / neighbor_min) if neighbor_min > 0 else 1.0
        valleys.append({"position": float(grid[vi]), "depth_ratio": depth_ratio})

    return {
        "grid": grid, "density": density,
        "n_peaks": len(peak_idx),
        "n_real_valleys": sum(1 for v in valleys if v["depth_ratio"] < VALLEY_REAL_THRESHOLD),
        "valleys": valleys,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    p.add_argument("--stability-seeds", type=int, nargs="+", default=DEFAULT_STABILITY_SEEDS)
    args = p.parse_args(argv)
    if len(args.stability_seeds) < 2:
        p.error(f"--stability-seeds needs at least 2 values to compare pairwise "
                 f"ARI/NMI across refits (got {args.stability_seeds}).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Load + build features (same inputs as borrower_persona_clustering.py) ===")
    df = load_and_join()
    feat, all_cols = build_features(df)
    numeric_cols = [c for c in all_cols if not c.startswith("has_")]
    feat_cfg = _get_features_config(None)
    X_log = _apply_log_winsorize(feat[numeric_cols].copy(), feat_cfg)
    X_pruned, selected_cols = _prune_correlated_features(X_log, feat_cfg)
    X_imputed = X_pruned.fillna(X_pruned.median()).fillna(0.0)
    # Rank-transform EVERY column to standard normal -- see
    # quantile_normalize_all_columns()'s docstring. A low-skew, negative-
    # capable column (cure_time_trend) with an extreme raw tail can
    # single-handedly dominate PCA's total variance regardless of its skew
    # statistic, since _apply_log_winsorize only transforms what it also
    # log-transforms -- and a fixed-percentile winsorize clip isn't
    # aggressive enough for a tail this heavy (verified empirically).
    X_final = quantile_normalize_all_columns(X_imputed)
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
    n_modality_pcs = min(3, pca_coords_full.shape[1])
    print("\nPC1-3 distribution (percentiles):")
    for i in range(n_modality_pcs):
        v = pca_coords_full[:, i]
        print(f"  PC{i + 1}: min={v.min():.2f} p10={np.percentile(v,10):.2f} "
              f"p25={np.percentile(v,25):.2f} median={np.median(v):.2f} "
              f"p75={np.percentile(v,75):.2f} p90={np.percentile(v,90):.2f} max={v.max():.2f} "
              f"skew={pd.Series(v).skew():.2f}")

    print(f"\nPC1-3 modality (KDE-based peak/valley detection, "
          f"valley 'real' if depth_ratio < {VALLEY_REAL_THRESHOLD}):")
    modality = []
    for i in range(n_modality_pcs):
        m = _analyze_modality(pca_coords_full[:, i])
        modality.append(m)
        print(f"  PC{i + 1}: {m['n_peaks']} peak(s), {len(m['valleys'])} valley(s) detected, "
              f"{m['n_real_valleys']} judged real (ratio < {VALLEY_REAL_THRESHOLD})")
        for v in m["valleys"]:
            tag = "REAL separation" if v["depth_ratio"] < VALLEY_REAL_THRESHOLD else "shoulder only"
            print(f"    valley at {v['position']:+.2f}: depth_ratio={v['depth_ratio']:.2f} ({tag})")
    if all(m["n_real_valleys"] == 0 for m in modality):
        print(
            "\n  No PC among PC1-3 shows a real density valley -- consistent with a "
            "population that's continuous along these dimensions (e.g. low-to-high "
            "capacity/engagement with no natural break) rather than discrete "
            "sub-populations. This does not mean clustering is wrong to attempt, but "
            "it does mean different algorithms should be expected to disagree about "
            "where to draw boundaries, however internally consistent each one is."
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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
        axes[0, 2].axis("off")

        for pc_i in range(n_modality_pcs):
            ax = axes[1, pc_i]
            v = pca_coords_full[:, pc_i]
            ax.hist(v, bins=80, density=True, color="#55A868", alpha=0.6)
            m = modality[pc_i]
            ax.plot(m["grid"], m["density"], color="#2A4D3A", linewidth=1.5)
            for val in m["valleys"]:
                color = "#C44E52" if val["depth_ratio"] < VALLEY_REAL_THRESHOLD else "#999999"
                ax.axvline(val["position"], color=color, linestyle="--", linewidth=1)
            ax.set_xlabel(f"PC{pc_i + 1}")
            ax.set_ylabel("Density")
            ax.set_title(f"PC{pc_i + 1}: {m['n_peaks']} peak(s), {m['n_real_valleys']} real valley(s)")

        plt.tight_layout()
        plot_path = OUT_DIR / "pca_overview.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  wrote {plot_path}")
    except ImportError:
        print("\n  WARNING: matplotlib not installed -- skipping pca_overview.png "
              "(pip install matplotlib to get it; numeric diagnostics above are unaffected)")

    X_pca = pca_coords_full[:, :n_components]

    # ── Part B: K-means vs constrained-GMM benchmark across K values ────────
    print("\n" + "=" * 78)
    print(f"PART B: K-means vs GMM({'/'.join(GMM_COVARIANCE_TYPES)}) benchmark, "
          f"K={args.k_values}, {len(args.stability_seeds)} stability seeds")
    print("=" * 78)

    rows = []
    for k in args.k_values:
        print(f"\n-- K={k} --")
        kmeans_labels = _fit_kmeans(X_pca, k, RANDOM_STATE)
        kmeans_sil = silhouette_score(X_pca, kmeans_labels, sample_size=min(20_000, n_active), random_state=RANDOM_STATE)
        kmeans_db = davies_bouldin_score(X_pca, kmeans_labels)
        kmeans_ch = calinski_harabasz_score(X_pca, kmeans_labels)
        kmeans_pct, kmeans_max_pct, kmeans_min_pct = _cluster_balance(kmeans_labels)
        kmeans_ari_mean, kmeans_ari_min, kmeans_nmi_mean, kmeans_nmi_min = _stability_ari_nmi(
            lambda s: _fit_kmeans(X_pca, k, s), args.stability_seeds
        )
        print(f"  KMeans      : silhouette={kmeans_sil:.3f} davies_bouldin={kmeans_db:.3f} "
              f"calinski_harabasz={kmeans_ch:.0f} sizes%={kmeans_pct}")
        print(f"                stability ARI mean/min={kmeans_ari_mean:.3f}/{kmeans_ari_min:.3f}, "
              f"NMI mean/min={kmeans_nmi_mean:.3f}/{kmeans_nmi_min:.3f}")
        rows.append({
            "k": k, "method": "kmeans", "silhouette": round(kmeans_sil, 3),
            "davies_bouldin": round(kmeans_db, 3), "calinski_harabasz": round(kmeans_ch, 1),
            "bic": None, "aic": None,
            "max_cluster_pct": kmeans_max_pct, "min_cluster_pct": kmeans_min_pct,
            "cluster_sizes_pct": kmeans_pct,
            "stability_ari_mean": round(kmeans_ari_mean, 3), "stability_ari_min": round(kmeans_ari_min, 3),
            "stability_nmi_mean": round(kmeans_nmi_mean, 3), "stability_nmi_min": round(kmeans_nmi_min, 3),
            "cross_vs_kmeans_ari": None, "cross_vs_kmeans_nmi": None,
        })

        for cov_type in GMM_COVARIANCE_TYPES:
            gmm_labels, gmm_model = _fit_gmm(X_pca, k, RANDOM_STATE, cov_type)
            gmm_sil = silhouette_score(X_pca, gmm_labels, sample_size=min(20_000, n_active), random_state=RANDOM_STATE)
            gmm_db = davies_bouldin_score(X_pca, gmm_labels)
            gmm_ch = calinski_harabasz_score(X_pca, gmm_labels)
            gmm_pct, gmm_max_pct, gmm_min_pct = _cluster_balance(gmm_labels)
            gmm_bic = gmm_model.bic(X_pca)
            gmm_aic = gmm_model.aic(X_pca)
            gmm_ari_mean, gmm_ari_min, gmm_nmi_mean, gmm_nmi_min = _stability_ari_nmi(
                lambda s, ct=cov_type: _fit_gmm(X_pca, k, s, ct)[0], args.stability_seeds
            )
            cross_ari = adjusted_rand_score(kmeans_labels, gmm_labels)
            cross_nmi = normalized_mutual_info_score(kmeans_labels, gmm_labels)

            print(f"  GMM-{cov_type:10s}: silhouette={gmm_sil:.3f} davies_bouldin={gmm_db:.3f} "
                  f"calinski_harabasz={gmm_ch:.0f} bic={gmm_bic:.0f} aic={gmm_aic:.0f} sizes%={gmm_pct}")
            print(f"                stability ARI mean/min={gmm_ari_mean:.3f}/{gmm_ari_min:.3f}, "
                  f"NMI mean/min={gmm_nmi_mean:.3f}/{gmm_nmi_min:.3f}")
            print(f"                vs. KMeans: ARI={cross_ari:.3f} NMI={cross_nmi:.3f}")

            rows.append({
                "k": k, "method": f"gmm_{cov_type}", "silhouette": round(gmm_sil, 3),
                "davies_bouldin": round(gmm_db, 3), "calinski_harabasz": round(gmm_ch, 1),
                "bic": round(gmm_bic, 1), "aic": round(gmm_aic, 1),
                "max_cluster_pct": gmm_max_pct, "min_cluster_pct": gmm_min_pct,
                "cluster_sizes_pct": gmm_pct,
                "stability_ari_mean": round(gmm_ari_mean, 3), "stability_ari_min": round(gmm_ari_min, 3),
                "stability_nmi_mean": round(gmm_nmi_mean, 3), "stability_nmi_min": round(gmm_nmi_min, 3),
                "cross_vs_kmeans_ari": round(cross_ari, 3), "cross_vs_kmeans_nmi": round(cross_nmi, 3),
            })

    results_df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "kmeans_gmm_benchmark.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv} ({len(results_df)} rows)")

    print("\n=== Summary table ===")
    display_cols = ["k", "method", "silhouette", "davies_bouldin", "calinski_harabasz",
                     "bic", "aic", "max_cluster_pct", "min_cluster_pct",
                     "stability_ari_mean", "stability_ari_min", "cross_vs_kmeans_ari"]
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(results_df[display_cols].to_string(index=False))

    print("\n=== Does GMM-vs-KMeans agreement rise as GMM's covariance constrains toward spherical? ===")
    pivot = results_df[results_df["method"] != "kmeans"].pivot(
        index="k", columns="method", values="cross_vs_kmeans_ari"
    )
    pivot_cols = [c for c in ["gmm_full", "gmm_diag", "gmm_spherical"] if c in pivot.columns]
    with pd.option_context("display.width", 200):
        print(pivot[pivot_cols].to_string())

    print(
        "\nReading this: look at the gmm_full -> gmm_diag -> gmm_spherical trend per "
        "row above. Rising substantially (e.g. toward 0.8+) means full-covariance GMM "
        "was describing genuinely compact structure through a different (elliptical) "
        "lens than K-means' spherical one -- the underlying partition is more "
        "defensible than cross-method ARI alone suggested. Rising only modestly means "
        "some geometric effect exists but no uniquely compelling partition. Staying "
        "flat (roughly 0.3-0.5 throughout) is the strongest signal that this is NOT "
        "primarily a covariance-assumption artifact -- worth weighing against the "
        "PC1-3 modality results above, which speak to the same question from a "
        "different angle (continuous gradients vs. discrete populations).\n\n"
        "Per-method within-method stability (ARI close to 1.0) remains valuable "
        "evidence on its own even where cross-method agreement stays moderate -- "
        "different algorithms have different objectives, so strong cross-method "
        "agreement is excellent evidence when present, but its absence isn't "
        "automatic evidence of an invalid clustering. No K is auto-selected here; "
        "business-meaning profiling (capacity/credit-quality/trend/outcome "
        "cross-tabs) on specific candidate K values is the next step, not computed "
        "here."
    )


if __name__ == "__main__":
    main()

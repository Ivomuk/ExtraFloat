"""
K=8 borrower-persona profiling -- the frozen production segmentation.

Why K=8: scripts/benchmark_persona_pca_clustering.py's real-data K=3..15
x {kmeans, gmm-full, gmm-diag, gmm-spherical} sweep showed K=8 with strong
within-method stability (KMeans ARI ~0.991) AND the strongest cross-method
convergence as GMM's covariance constrains toward spherical (ARI vs KMeans
~0.903) -- satisfying the sweep's own "Outcome A" criterion (sharp
convergence = robust structure, not a covariance-assumption artifact).
K=8 was chosen over the similarly-strong K=7 because the business already
operates 8 borrower profiles, giving the technical and business evidence a
rare, deliberate alignment. No further K/algorithm tuning happens here --
this script FREEZES that choice and profiles it.

Production vs. challenger: K-means is the production assignment method
(deterministic centroids, directly operationalizable). Spherical-covariance
GMM is fit at the same K as a challenger/validation model, not a second
candidate to pick between -- its agreement with K-means (reported here) is
itself evidence for or against trusting the K-means partition, per the
sweep's cross-method-agreement framework.

Reuses load_and_join()/build_features()/quantile_normalize_all_columns()
from borrower_persona_clustering.py and the identical Step 3 feature
pipeline (log-winsorize -> correlation-prune -> impute -> quantile-
normalize -> RobustScaler -> PCA) the benchmark script used, so the K=8
fit here is the exact same input space that produced the sweep's numbers.

borrower_trend / borrower_profile_type / thin-file status / commission tier
and every outcome variable (risk_score, risk_tier, assigned_limit,
repayment/default rates not used as clustering features) are brought in
ONLY for the cross-tab validation artifact below -- never fed back into
the KMeans/GMM fit. is_thin_file and the commission tier are both derived
locally rather than requiring engine_test_output.csv, so this validation
artifact doesn't depend on a keep_intermediate=True engine run existing:
  - is_thin_file: total_loans < 3, matching the engine's own definition in
    extrafloat_limit_engine_features.py / docs/engine_output_data_dictionary.md.
  - commission tier: the business's actual 8-category tier scheme
    (diamond/titanium/platinum/gold/silver/bronze/new bronze/below_threshold)
    -- NOT borrower_profile_type (confirmed to be a data-maturity flag --
    no_history/thin_file/insufficient_history/thick_file -- not a persona
    taxonomy) and NOT agent_profile (MTN's own top-level classification,
    a different, unrelated field). This is XtraFloat's own "authoritative
    second-level classification" (extrafloat_limit_engine_features.py:583-585),
    computed purely from 6-month commission via DEFAULT_CAP_CONFIG["agent_tier"]
    in extrafloat_limit_engine_caps.py -- imported directly here (not
    duplicated) so this stays in sync if the business ever changes the
    thresholds. Reuses the same "commission" column already loaded from
    the MoMo mart for clustering, since that IS the 6-month commission
    figure the engine's own tier assignment reads (confirmed via
    extrafloat_limit_engine_features.py:589-598).

PERSONA_NAMES/PERSONA_RATIONALE (below) carry the provisional persona
names and their one-line "why" drafted from this profiling work; every
summary-level output artifact includes persona_name + persona_rationale
columns/rows alongside the raw cluster ID, so both travel with the data
rather than staying only in chat/commit history. The per-borrower
assignments file carries persona_name only (rationale is cluster-level,
not worth repeating 143k times).

Outputs (segmentation_outputs/persona_k8_profile/):
  k8_cluster_profile.csv       tidy cluster x feature profile (medians,
                                ratios, standardized diffs, percentiles),
                                with persona_name + persona_rationale
  pca_loadings.csv             Feature x PC loadings
  k8_persona_fingerprint.csv   wide cluster x feature standardized-deviation
                                matrix (the same standardized diffs, pivoted)
  k8_validation_crosstabs.xlsx multi-sheet cross-tabs: borrower_profile_type,
                                borrower_trend, thin/thick file, commission
                                tier (diamond/.../below_threshold), held-out
                                outcome summary, KMeans-vs-GMM agreement
                                (falls back to one CSV per sheet if
                                openpyxl isn't installed)
  k8_cluster_assignments.csv   id, kmeans cluster, gmm_spherical cluster --
                                supporting infra for later differentiation-
                                matrix / business-profile-mapping work, not
                                one of the four requested artifacts itself

Usage:
    python scripts\\profile_persona_k8.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from segmentation.borrower_persona_clustering import (  # noqa: E402
    build_features,
    digits,
    load_and_join,
    quantile_normalize_all_columns,
)
from segmentation.extrafloat_segmentation_features import (  # noqa: E402
    _apply_log_winsorize,
    _get_features_config,
    _prune_correlated_features,
)
from extrafloat.engine.extrafloat_limit_engine_caps import DEFAULT_CAP_CONFIG  # noqa: E402

K = 8
RANDOM_STATE = 42
N_INIT = 20
GMM_COVARIANCE_TYPE = "spherical"
N_LOADING_PCS = 8  # printed/saved loadings table width; PC1-3 are the ones
                    # actually asked for, the rest are included for free.

# PROVISIONAL persona names, assigned from the real run's fingerprint +
# outcome_summary + commission_tier_crosstab (cluster IDs confirmed against
# that actual run, not synthetic data) -- not yet validated against forward
# realized outcomes (Persona(t) -> Performance(t+1..n)), so treat these as
# revisable working labels, not final business names. KMeans with a fixed
# random_state/n_init is deterministic given the same input, so cluster ID 0
# will keep meaning the same thing across reruns on the SAME underlying
# data/feature set -- re-verify this mapping against a fresh fingerprint
# before trusting it if the upstream snapshot or selected_cols ever changes.
PERSONA_NAMES = {
    0: "Core Reliable Majority",
    1: "Strained High-Earners",
    2: "Established Repeat Risk",
    3: "New & Already Struggling",
    4: "Elite Quality, Underleveraged",
    5: "Flagship Power Users",
    6: "Dormant Legacy Borrowers",
    7: "Mainstream Elevated Risk",
}

# One-line rationale behind each name above, so the "why" travels with the
# name into every output artifact rather than staying only in chat/commit
# history. Same provisional caveat as PERSONA_NAMES.
PERSONA_RATIONALE = {
    0: "High commission tier (90% top-3), solidly good on-time/default -- the backbone segment",
    1: "Meaningfully higher commission tier than C7, but below-average quality and a worsening trend",
    2: "Thick-file, ~6 prior loans, consistently poor on-time/default throughout",
    3: "Newest, mostly no-history, one large first loan, poor early signal",
    4: "Near-perfect on-time/default, fastest cure, high tier, but very few loans",
    5: "Near-pure diamond tier, best quality, by far the highest loan volume",
    6: "Zero commission/balance/limit, 74.5% below-threshold, yet real loan history",
    7: "Ordinary tier, below-average quality, largest at-risk population by sheer scale",
}

REPO = Path(__file__).resolve().parent.parent
ENGINE_OUTPUT_PATH = REPO / "output" / "engine_test_output.csv"
OUT_DIR = REPO / "segmentation_outputs" / "persona_k8_profile"

# commission_thresholds is ordered highest -> lowest in DEFAULT_CAP_CONFIG
# (first match wins, matching extrafloat_limit_engine_features.py's own
# _commission_to_multiplier()/agent_category logic exactly) -- imported
# live rather than hardcoded so this never drifts from the engine's real
# tier cutoffs.
_COMMISSION_TIER_THRESHOLDS = DEFAULT_CAP_CONFIG["agent_tier"]["commission_thresholds"]


def _commission_tier(commission: float) -> str:
    for tier_name, threshold in _COMMISSION_TIER_THRESHOLDS.items():
        if commission >= threshold:
            return tier_name
    return "below_threshold"


def _standardized_profile(raw_features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Tidy (persona_cluster, feature) profile vs. the whole population.

    standardized_diff = (cluster_mean - population_mean) / population_std,
    computed on the RAW (pre-log, pre-quantile-normalize) engineered feature
    values deliberately -- these are business units (UGX, days, rates),
    not PCA-space or rank-space units, so "+1.2 SD in commission" means
    something a reader can act on without knowing this pipeline's internals.
    """
    pop_median = raw_features.median()
    pop_mean = raw_features.mean()
    pop_std = raw_features.std(ddof=0).replace(0, np.nan)

    rows = []
    for cluster, idx in labels.groupby(labels).groups.items():
        sub = raw_features.loc[idx]
        n = len(idx)
        cluster_median = sub.median()
        cluster_mean = sub.mean()
        for feature in raw_features.columns:
            rows.append({
                "persona_cluster": cluster,
                "feature": feature,
                "n_borrowers": n,
                "pct_of_population": round(n / len(raw_features) * 100, 2),
                "cluster_median": cluster_median[feature],
                "population_median": pop_median[feature],
                "ratio_to_population_median": (
                    cluster_median[feature] / pop_median[feature]
                    if pop_median[feature] not in (0, None) and not pd.isna(pop_median[feature])
                    else np.nan
                ),
                "cluster_mean": cluster_mean[feature],
                "population_mean": pop_mean[feature],
                "population_std": pop_std[feature],
                "standardized_diff": (
                    (cluster_mean[feature] - pop_mean[feature]) / pop_std[feature]
                    if pd.notna(pop_std[feature]) else np.nan
                ),
                "p10": sub[feature].quantile(0.10),
                "p25": sub[feature].quantile(0.25),
                "p50": sub[feature].quantile(0.50),
                "p75": sub[feature].quantile(0.75),
                "p90": sub[feature].quantile(0.90),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Load + build features (same inputs as the benchmark sweep) ===")
    df = load_and_join()
    feat, all_cols = build_features(df)
    numeric_cols = [c for c in all_cols if not c.startswith("has_")]
    feat_cfg = _get_features_config(None)
    X_log = _apply_log_winsorize(feat[numeric_cols].copy(), feat_cfg)
    X_pruned, selected_cols = _prune_correlated_features(X_log, feat_cfg)
    X_imputed = X_pruned.fillna(X_pruned.median()).fillna(0.0)
    X_final = quantile_normalize_all_columns(X_imputed)
    n_active = len(X_final)
    print(f"  {n_active:,} agents, {len(selected_cols)} clustering features: {selected_cols}")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_final.astype(float))

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    target_variance = feat_cfg["target_pca_variance"]
    n_components = int(np.argmax(cum_var >= target_variance) + 1)
    pca_coords = pca_full.transform(X_scaled)
    X_pca = pca_coords[:, :n_components]
    print(f"  PCA: {n_components} components reach {target_variance:.0%} variance "
          f"(explained: {cum_var[n_components - 1]:.1%})")

    print(f"\n=== Fit K={K} (production: KMeans, challenger: GMM-{GMM_COVARIANCE_TYPE}) ===")
    kmeans = KMeans(n_clusters=K, n_init=N_INIT, random_state=RANDOM_STATE)
    kmeans_labels = pd.Series(kmeans.fit_predict(X_pca), index=X_final.index, name="persona_cluster")

    gmm = GaussianMixture(
        n_components=K, covariance_type=GMM_COVARIANCE_TYPE, n_init=max(1, N_INIT // 4),
        random_state=RANDOM_STATE, reg_covar=1e-4,
    )
    gmm_labels = pd.Series(gmm.fit_predict(X_pca), index=X_final.index, name=f"gmm_{GMM_COVARIANCE_TYPE}_cluster")

    ari = adjusted_rand_score(kmeans_labels, gmm_labels)
    nmi = normalized_mutual_info_score(kmeans_labels, gmm_labels)
    print("  KMeans cluster sizes:")
    for cluster_id, size in kmeans_labels.value_counts().sort_index().items():
        name = PERSONA_NAMES.get(cluster_id, "(unnamed)")
        rationale = PERSONA_RATIONALE.get(cluster_id, "")
        print(f"    C{cluster_id} {name:32s} n={size:,}")
        print(f"       {rationale}")
    print(f"  KMeans vs GMM-{GMM_COVARIANCE_TYPE} agreement at K={K}: ARI={ari:.3f} NMI={nmi:.3f}")

    # ── Artifact 1: PCA loadings ────────────────────────────────────────────
    n_loading_pcs = min(N_LOADING_PCS, pca_full.components_.shape[0])
    loadings = pd.DataFrame(
        pca_full.components_[:n_loading_pcs].T,
        index=selected_cols,
        columns=[f"PC{i + 1}" for i in range(n_loading_pcs)],
    )
    loadings_path = OUT_DIR / "pca_loadings.csv"
    loadings.to_csv(loadings_path)
    print(f"\n  wrote {loadings_path} ({loadings.shape[0]} features x {loadings.shape[1]} PCs)")
    print("\n  Top-5 |loading| per PC1-3:")
    for pc in loadings.columns[:3]:
        top = loadings[pc].abs().sort_values(ascending=False).head(5)
        for feature in top.index:
            print(f"    {pc} {feature:45s} {loadings.loc[feature, pc]:+.3f}")

    # ── Artifact 2 + 3: cluster profile (tidy) + persona fingerprint (wide) ─
    raw_features = feat[selected_cols].copy()
    profile = _standardized_profile(raw_features, kmeans_labels)
    profile.insert(1, "persona_name", profile["persona_cluster"].map(PERSONA_NAMES))
    profile.insert(2, "persona_rationale", profile["persona_cluster"].map(PERSONA_RATIONALE))
    profile_path = OUT_DIR / "k8_cluster_profile.csv"
    profile.to_csv(profile_path, index=False)
    print(f"\n  wrote {profile_path} ({len(profile)} rows = {K} clusters x {len(selected_cols)} features)")

    fingerprint = profile.pivot(index="persona_cluster", columns="feature", values="standardized_diff")
    fingerprint = fingerprint[selected_cols]  # stable, deliberate column order (not alphabetical)
    sizes = kmeans_labels.value_counts().sort_index()
    fingerprint.insert(0, "persona_name", pd.Series(PERSONA_NAMES))
    fingerprint.insert(1, "persona_rationale", pd.Series(PERSONA_RATIONALE))
    fingerprint.insert(2, "n_borrowers", sizes)
    fingerprint.insert(3, "pct_of_population", (sizes / n_active * 100).round(2))
    fingerprint_path = OUT_DIR / "k8_persona_fingerprint.csv"
    fingerprint.to_csv(fingerprint_path)
    print(f"  wrote {fingerprint_path} ({fingerprint.shape[0]} clusters x {fingerprint.shape[1]} columns)")

    # ── Artifact 4: validation cross-tabs ───────────────────────────────────
    print("\n=== Assembling validation cross-tabs (none of these fed the clustering fit) ===")
    val = pd.DataFrame(index=X_final.index)
    val["persona_cluster"] = kmeans_labels
    val[f"gmm_{GMM_COVARIANCE_TYPE}_cluster"] = gmm_labels
    val["borrower_profile_type"] = df["borrower_profile_type"]
    val["borrower_trend"] = df["borrower_trend"]
    val["total_loans"] = df["total_loans"]
    val["is_thin_file"] = (df["total_loans"].fillna(0) < 3).astype(int)
    val["file_status"] = np.where(val["is_thin_file"] == 1, "thin_file", "thick_file")
    val["commission_tier"] = df["commission"].apply(_commission_tier)

    # Outcome / descriptive variables NOT used as clustering inputs. Several
    # repayment-behavior columns (lifetime_on_time_24h_rate etc.) WERE used
    # as clustering features -- included below anyway for descriptive
    # continuity but explicitly labeled, since they can't serve as
    # independent validation of the clusters that were built from them.
    used_in_clustering = set(selected_cols)
    outcome_candidates = {
        "lifetime_on_time_24h_rate": df["lifetime_on_time_24h_rate"],
        "lifetime_default_24h_rate": df["lifetime_default_24h_rate"],
        "recent_5_default_24h_rate": df["recent_5_default_24h_rate"],
        "lifetime_avg_hours_to_principal_cure": df["lifetime_avg_hours_to_principal_cure"],
        "avg_loan_size_lifetime": df["avg_loan_size_lifetime"],
        "commission": df["commission"],
        "account_balance": df["account_balance"],
    }
    for name, series in outcome_candidates.items():
        val[name] = series

    engine_path = Path(ENGINE_OUTPUT_PATH)
    held_out_outcome_cols = []
    if engine_path.exists():
        wanted_cols = ["msisdn", "assigned_limit", "risk_tier", "risk_score"]
        available_cols = set(pd.read_csv(engine_path, nrows=0).columns)
        missing_cols = [c for c in wanted_cols if c not in available_cols]
        read_cols = [c for c in wanted_cols if c in available_cols]
        if missing_cols:
            print(f"  WARNING: {ENGINE_OUTPUT_PATH} is missing {missing_cols} -- continuing without them.")
        eng = pd.read_csv(engine_path, usecols=read_cols)
        eng["_id"] = digits(eng["msisdn"])
        eng = eng.drop(columns=["msisdn"]).set_index("_id")
        val = val.join(eng, how="left")
        held_out_outcome_cols = [c for c in ["assigned_limit", "risk_tier", "risk_score"] if c in val.columns]
        print(f"  joined {ENGINE_OUTPUT_PATH.name}: {held_out_outcome_cols} "
              f"(these are TRUE held-out validation signals -- never touched the clustering fit)")
    else:
        print(f"  WARNING: {ENGINE_OUTPUT_PATH} not found -- skipping assigned_limit/risk_tier/risk_score "
              f"held-out validation columns. Run a credit-engine pass first for the strongest validation signal.")

    sheets: dict[str, pd.DataFrame] = {}

    size_tbl = pd.DataFrame({
        "persona_name": pd.Series(PERSONA_NAMES),
        "persona_rationale": pd.Series(PERSONA_RATIONALE),
        "n_borrowers": sizes,
        "pct_of_population": (sizes / n_active * 100).round(2),
    })
    size_tbl.index.name = "persona_cluster"
    sheets["cluster_sizes"] = size_tbl.reset_index()

    # Highest-to-lowest tier order (matching DEFAULT_CAP_CONFIG's own
    # ordering), not crosstab's default alphabetical column order -- makes
    # the commission_tier sheet readable at a glance.
    tier_order = list(_COMMISSION_TIER_THRESHOLDS.keys()) + ["below_threshold"]

    for cat_col in ("borrower_profile_type", "borrower_trend", "file_status", "commission_tier"):
        pct = pd.crosstab(val["persona_cluster"], val[cat_col], normalize="index").round(4) * 100
        cnt = pd.crosstab(val["persona_cluster"], val[cat_col])
        if cat_col == "commission_tier":
            present_order = [t for t in tier_order if t in cnt.columns]
            pct, cnt = pct[present_order], cnt[present_order]
        pct.columns = [f"{c}_pct" for c in pct.columns]
        cnt.columns = [f"{c}_n" for c in cnt.columns]
        sheets[f"{cat_col}_crosstab"] = pd.concat([cnt, pct], axis=1).reset_index()

    outcome_rows = []
    all_outcome_cols = list(outcome_candidates.keys()) + held_out_outcome_cols + ["total_loans"]
    for cluster, idx in val.groupby("persona_cluster").groups.items():
        sub = val.loc[idx]
        row = {"persona_cluster": cluster, "n_borrowers": len(idx)}
        for col in all_outcome_cols:
            if col not in val.columns:
                continue
            if col == "risk_tier":
                mode = sub[col].mode()
                row[f"{col}_mode"] = mode.iat[0] if not mode.empty else None
            else:
                row[f"{col}_median"] = sub[col].median()
            row[f"{col}_used_in_clustering"] = col in used_in_clustering
        outcome_rows.append(row)
    sheets["outcome_summary"] = pd.DataFrame(outcome_rows)

    kmeans_vs_gmm = pd.crosstab(val["persona_cluster"], val[f"gmm_{GMM_COVARIANCE_TYPE}_cluster"])
    kmeans_vs_gmm.loc["AGREEMENT"] = [None] * kmeans_vs_gmm.shape[1]
    sheets["kmeans_vs_gmm_agreement"] = pd.concat([
        pd.DataFrame({"metric": ["ARI", "NMI"], "value": [round(ari, 4), round(nmi, 4)]}),
        pd.DataFrame({"metric": [""], "value": [""]}),
        kmeans_vs_gmm.reset_index(),
    ], ignore_index=False)

    xlsx_path = OUT_DIR / "k8_validation_crosstabs.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for sheet_name, sheet_df in sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f"\n  wrote {xlsx_path} ({len(sheets)} sheets)")
    except ImportError:
        print(f"\n  WARNING: openpyxl not installed -- writing one CSV per sheet instead of "
              f"{xlsx_path.name} (pip install openpyxl to get a single workbook).")
        for sheet_name, sheet_df in sheets.items():
            csv_path = OUT_DIR / f"k8_validation_crosstabs__{sheet_name}.csv"
            sheet_df.to_csv(csv_path, index=False)
            print(f"    wrote {csv_path}")

    # ── Supporting infra (not one of the 4 requested artifacts) ─────────────
    assignments = pd.DataFrame({
        "agent_msisdn": df["agent_msisdn"],
        "phonenumber": df["phonenumber"],
        "persona_cluster": kmeans_labels,
        "persona_name": kmeans_labels.map(PERSONA_NAMES),
        f"gmm_{GMM_COVARIANCE_TYPE}_cluster": gmm_labels,
    })
    assignments_path = OUT_DIR / "k8_cluster_assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    print(f"\n  wrote {assignments_path} ({len(assignments):,} rows) -- supporting infra for "
          f"the 8x8 differentiation matrix / business-profile mapping, not one of the 4 requested artifacts")

    print("\nDone. Cluster sizes:")
    print(size_tbl.to_string())


if __name__ == "__main__":
    main()

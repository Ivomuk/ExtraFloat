"""
borrower_persona_clustering.py
================================
Combines MoMo transaction-capacity data (mfs_daily_agent_mart) with loan/
credit-behavior data (borrower_history_features) to cluster the population
that has BOTH, into "borrower personas" -- something no existing module in
AgentFloat-integrated-solution does (segmentation/ only ever looks at MoMo
KPIs, never joins loan data).

Reuses the generic (column-list-driven) stages of segmentation's feature
engineering and clustering pipeline via direct import -- nothing under
AgentFloat-integrated-solution is modified. See the approved plan for the
full design rationale (feature framework, missing-value handling, why
dormancy filtering and hdb-to-tier mapping are bypassed).

Run order: after pipeline/run_pipeline.py (needs borrower_history_features.csv)
and after a credit-engine run (needs engine_output.csv for profiling-only
risk/limit columns -- optional, degrades gracefully if absent).
"""

import sys
from pathlib import Path

REPO = r"F:\AGENT DATA\AgentFloat-integrated-solution"
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

from segmentation.extrafloat_segmentation_features import (
    _apply_log_winsorize,
    _get_features_config,
    _prune_correlated_features,
    _scale_and_reduce,
)
from segmentation.extrafloat_segmentation_pipeline import (
    _get_clustering_config,
    flag_anomalies,
)

MOMO_PATH = r"F:\AGENT DATA\mfs_daily_agent_mart_20260831.csv"
LOANS_PATH = r"F:\AGENT DATA\pipeline\output\borrower_history_features.csv"
ENGINE_OUTPUT_PATH = r"F:\AGENT DATA\pipeline\output\engine_output.csv"
OUT_DIR = Path(r"F:\AGENT DATA\pipeline\output\borrower_persona_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_DATE = pd.Timestamp("2026-08-31")
EXPECTED_JOIN_COUNT = 28063


def digits(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True)


def load_and_join() -> pd.DataFrame:
    momo_cols = [
        "agent_msisdn", "commission", "account_balance",
        "cash_out_vol_3m", "cash_out_value_3m",
        "payment_vol_3m", "voucher_vol_3m", "cust_3m",
        "rev_1m", "rev_3m", "activation_dt",
    ]
    loan_cols = [
        "phonenumber", "total_loans", "avg_loan_size_lifetime",
        "lifetime_on_time_rate", "lifetime_default_rate", "recent_5_default_rate",
        "lifetime_avg_days_to_principal_cure", "lifetime_cure_time_volatility",
        "cure_time_trend", "borrower_trend", "borrower_profile_type", "latest_product",
    ]

    momo = pd.read_csv(MOMO_PATH, usecols=momo_cols)
    loans = pd.read_csv(LOANS_PATH, usecols=loan_cols)

    momo["_id"] = digits(momo["agent_msisdn"])
    loans["_id"] = digits(loans["phonenumber"])

    df = momo.merge(loans, on="_id", how="inner", suffixes=("_momo", "_loan"))
    print(f"Joined population: {len(df):,} (expected {EXPECTED_JOIN_COUNT:,})")
    assert len(df) == EXPECTED_JOIN_COUNT, (
        f"Join count changed since plan-time verification: got {len(df)}, "
        f"expected {EXPECTED_JOIN_COUNT}. Investigate before proceeding."
    )
    df = df.set_index("_id")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    eps = 1.0
    feat = pd.DataFrame(index=df.index)

    # -- Capacity --
    feat["commission"] = df["commission"]
    feat["account_balance"] = df["account_balance"]
    feat["cash_out_vol_3m"] = df["cash_out_vol_3m"]
    feat["cash_out_value_3m"] = df["cash_out_value_3m"]
    feat["payment_vol_3m"] = df["payment_vol_3m"]
    feat["voucher_vol_3m"] = df["voucher_vol_3m"]
    feat["cust_3m"] = df["cust_3m"]
    feat["rev_1m_to_3m_ratio"] = df["rev_1m"] / (df["rev_3m"] + eps)

    # activation_dt is stored as a plain YYYYMMDD float (e.g. 20240913.0) --
    # pd.to_datetime without an explicit format silently treats a bare
    # numeric Series as nanoseconds-since-epoch, collapsing every date to
    # ~1970-01-01. Must parse via the actual YYYYMMDD format.
    activation_str = df["activation_dt"].dropna().astype("Int64").astype(str)
    activation_dt = pd.to_datetime(
        df["activation_dt"].astype("Int64").astype(str).replace("<NA>", None),
        format="%Y%m%d", errors="coerce",
    )
    median_activation = activation_dt.median()
    activation_dt = activation_dt.fillna(median_activation)
    feat["tenure_days"] = (SNAPSHOT_DATE - activation_dt).dt.days

    # -- Credit engagement --
    feat["total_loans"] = df["total_loans"]
    feat["avg_loan_size_lifetime"] = df["avg_loan_size_lifetime"]

    # -- Credit quality --
    feat["lifetime_on_time_rate"] = df["lifetime_on_time_rate"]
    feat["lifetime_default_rate"] = df["lifetime_default_rate"]

    feat["has_recent_history"] = df["recent_5_default_rate"].notna().astype(int)
    feat["recent_5_default_rate"] = df["recent_5_default_rate"].fillna(0.0)

    feat["lifetime_avg_days_to_principal_cure"] = df["lifetime_avg_days_to_principal_cure"]

    feat["has_volatility_history"] = df["lifetime_cure_time_volatility"].notna().astype(int)
    feat["lifetime_cure_time_volatility"] = df["lifetime_cure_time_volatility"].fillna(0.0)

    feat["has_cure_trend"] = df["cure_time_trend"].notna().astype(int)
    feat["cure_time_trend"] = df["cure_time_trend"].fillna(0.0)

    numeric_cols = [c for c in feat.columns if not c.startswith("has_")]
    for c in numeric_cols:
        na_rate = feat[c].isna().mean()
        if na_rate > 0:
            print(f"  median-fallback: {c} had {na_rate:.4%} NaN, filling with median")
            feat[c] = feat[c].fillna(feat[c].median())

    flag_cols = [c for c in feat.columns if c.startswith("has_")]
    return feat, numeric_cols + flag_cols


def try_hdbscan_configs(df_features: pd.DataFrame, selected_cols: list[str], active_mask: pd.Series):
    """Empirically compare a few HDBSCAN sizings; return (chosen_cfg, chosen_result)."""
    candidates = [
        ("defaults_1000_150", {"hdbscan_min_cluster_size": 1000, "hdbscan_min_samples": 150}),
        ("scaled_300_40", {"hdbscan_min_cluster_size": 300, "hdbscan_min_samples": 40}),
        ("scaled_150_20", {"hdbscan_min_cluster_size": 150, "hdbscan_min_samples": 20}),
    ]
    results = {}
    for name, overrides in candidates:
        cfg = _get_clustering_config(overrides)
        out = flag_anomalies(df_features, selected_cols, active_mask, config=cfg)
        labels = out["anomaly_cluster_hdb_raw"]
        n_active = int(active_mask.sum())
        noise_pct = (labels == -1).sum() / n_active * 100
        n_clusters = labels[labels != -1].nunique()
        print(f"  {name}: n_clusters={n_clusters}, noise={noise_pct:.1f}%")
        results[name] = (cfg, out, noise_pct, n_clusters)

    # Pick the config with the lowest noise% among those producing >=3 clusters;
    # fall back to lowest noise% overall if none qualify.
    qualifying = {k: v for k, v in results.items() if v[3] >= 3}
    pool = qualifying if qualifying else results
    best_name = min(pool, key=lambda k: pool[k][2])
    print(f"  -> selected: {best_name}")
    cfg, out, noise_pct, n_clusters = results[best_name]
    return best_name, cfg, out


def main():
    print("=== Step 1: load + join ===")
    df = load_and_join()

    print("\n=== Step 2: build feature matrix ===")
    feat, all_cols = build_features(df)
    numeric_cols = [c for c in all_cols if not c.startswith("has_")]
    print(f"  {len(numeric_cols)} numeric features before pruning: {numeric_cols}")

    print("\n=== Step 3: log/winsorize + correlation pruning + scale ===")
    feat_cfg = _get_features_config(None)
    X_log = _apply_log_winsorize(feat[numeric_cols].copy(), feat_cfg)
    X_pruned, selected_cols = _prune_correlated_features(X_log, feat_cfg)
    X_final = X_pruned.fillna(X_pruned.median()).fillna(0.0)
    X_scaled, X_pca = _scale_and_reduce(X_final, feat_cfg)
    print(f"  final PCA shape: {X_pca.shape}")

    # Rebuild the full feature frame flag_anomalies expects (raw, unscaled --
    # it does its own RobustScaler+PCA internally via _get_active_pca).
    df_features = feat[selected_cols].copy()

    print("\n=== Step 4: HDBSCAN sizing comparison ===")
    active_mask = pd.Series(True, index=df_features.index)
    chosen_name, chosen_cfg, anomaly_out = try_hdbscan_configs(df_features, selected_cols, active_mask)

    print("\n=== Step 5: assemble output ===")
    result = df[["agent_msisdn", "phonenumber", "borrower_trend", "borrower_profile_type", "latest_product"]].copy()
    for c in all_cols:
        result[c] = feat[c]
    result["persona_cluster"] = anomaly_out["anomaly_cluster_hdb_raw"]
    result["is_anomaly"] = anomaly_out["is_anomaly"]
    result["is_global_anomaly"] = anomaly_out["is_global_anomaly"]
    result["is_local_anomaly"] = anomaly_out["is_local_anomaly"]
    result["lof_score"] = anomaly_out["lof_score"]

    engine_path = Path(ENGINE_OUTPUT_PATH)
    if engine_path.exists():
        eng = pd.read_csv(engine_path, usecols=["msisdn", "assigned_limit", "risk_tier", "capacity_cap", "combined_cap"])
        eng["_id"] = digits(eng["msisdn"])
        eng = eng.drop(columns=["msisdn"]).set_index("_id")
        result = result.join(eng, how="left")
        print(f"  joined engine_output.csv risk/limit columns ({eng.shape[1]} cols)")
    else:
        print(f"  WARNING: {ENGINE_OUTPUT_PATH} not found -- skipping risk/limit profiling columns")

    out_path = OUT_DIR / "borrower_persona_clusters.csv"
    result.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(result):,} rows, {result.shape[1]} cols)")

    print("\n=== Step 6: cluster profile ===")
    profile_cols = [c for c in numeric_cols if c in result.columns]
    profile = result.groupby("persona_cluster")[profile_cols].median()
    profile["n_borrowers"] = result.groupby("persona_cluster").size()
    if "assigned_limit" in result.columns:
        profile["median_assigned_limit"] = result.groupby("persona_cluster")["assigned_limit"].median()
        profile["median_risk_tier"] = result.groupby("persona_cluster")["risk_tier"].agg(
            lambda s: s.mode().iat[0] if not s.mode().empty else None
        )

    profile_path = OUT_DIR / "borrower_persona_cluster_profile.csv"
    profile.to_csv(profile_path)
    print(f"  wrote {profile_path} ({len(profile)} clusters)")
    print("\nCluster sizes:")
    print(result["persona_cluster"].value_counts(dropna=False).sort_index())

    print("\nCategorical cross-tabs (share within cluster):")
    for cat_col in ("borrower_profile_type", "borrower_trend"):
        print(f"\n-- {cat_col} --")
        print(pd.crosstab(result["persona_cluster"], result[cat_col], normalize="index").round(3))

    print("\nDone.")


if __name__ == "__main__":
    main()

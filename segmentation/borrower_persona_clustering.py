"""
borrower_persona_clustering.py
================================
Combines MoMo transaction-capacity data (mfs_daily_agent_mart) with loan/
credit-behavior data (borrower_history_retail_filtered.csv) to cluster the
population that has BOTH, into "borrower personas" -- something no existing
module in this repo does (segmentation/ only ever looks at MoMo KPIs, never
joins loan data).

Reuses the generic (column-list-driven) stages of segmentation's feature
engineering and clustering pipeline via direct import -- nothing under
segmentation/ is modified. See the approved plan for the full design
rationale (feature framework, missing-value handling, why dormancy
filtering and hdb-to-tier mapping are bypassed).

Run order: after run_retail_filtered.bat (needs the borrower_history_retail_filtered.csv
it produces) and after a credit-engine run (needs output/engine_test_output.csv
for profiling-only risk/limit columns -- optional, degrades gracefully if absent).
"""

import sys
from pathlib import Path

# Repo root: this file lives directly under segmentation/, same depth as
# scripts/*.py -- .parent.parent resolves the same way those scripts do,
# so `from segmentation....` (this package's own parent) is importable
# regardless of the current working directory this is run from.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

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

# data/, segmentation_outputs/, and the repo-root borrower_history_retail_
# filtered.csv all match this repo's established conventions -- see
# run_retail_filtered.bat (produces borrower_history_retail_filtered.csv at
# the repo root, data/mfs_daily_agent_mart_*.csv, output/engine_test_output.csv)
# and run_segmentation_standalone.bat (segmentation_outputs/).
MOMO_PATH = REPO / "data" / "mfs_daily_agent_mart_20260731.csv"
LOANS_PATH = REPO / "borrower_history_retail_filtered.csv"
ENGINE_OUTPUT_PATH = REPO / "output" / "engine_test_output.csv"
OUT_DIR = REPO / "segmentation_outputs" / "borrower_persona_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_DATE = pd.Timestamp("2026-07-31")


def digits(s: pd.Series) -> pd.Series:
    """Strip to digits only.

    A missing/blank/non-numeric input strips down to an empty string, not
    NaN -- masked to NA here so it can never become a shared join key. Left
    unmasked, every row with a missing agent_msisdn/phonenumber on either
    side would collapse onto the same "" key and silently many-to-many
    join against every other missing-msisdn row on the opposite side.

    If even one row anywhere in the source column is missing, pandas reads
    the WHOLE column as float64, not just that row -- str() on a float
    appends ".0", and naive digit-stripping keeps that trailing zero as
    a spurious extra digit on every otherwise-valid id in the column.
    Round-tripping through nullable Int64 first strips the decimal point
    (and preserves NaN as NA) before the id ever becomes a string.
    """
    if pd.api.types.is_float_dtype(s):
        s = s.astype("Int64")
    out = s.astype(str).str.replace(r"\D", "", regex=True)
    return out.mask(out == "")


def load_and_join() -> pd.DataFrame:
    momo_cols = [
        "agent_msisdn", "commission", "account_balance",
        "cash_out_vol_3m", "cash_out_value_3m",
        "payment_vol_3m", "cust_3m",
        "rev_1m", "rev_3m", "activation_dt",
    ]
    # Column names confirmed against BORROWER_LIMIT_REQUIRED_COLUMNS in
    # extrafloat/engine/extrafloat_limit_engine_features.py, this repo's own
    # established schema for borrower_history-family files. The 24h variant
    # is used for on-time/default rate (the schema's dominant convention --
    # recent_5_default_24h_rate, prior_on_time_24h_rate, etc. are all 24h;
    # 26h only exists as an alternate for the two "lifetime" rates
    # specifically -- swap to lifetime_on_time_26h_rate/lifetime_default_26h_rate
    # if that's the window this analysis should actually use). Cure time is
    # tracked in HOURS, not days, in this schema. latest_product has no
    # equivalent column anywhere in this schema -- dropped, not renamed.
    loan_cols = [
        "phonenumber", "total_loans", "avg_loan_size_lifetime",
        "lifetime_on_time_24h_rate", "lifetime_default_24h_rate", "recent_5_default_24h_rate",
        "lifetime_avg_hours_to_principal_cure", "lifetime_cure_time_volatility",
        "cure_time_trend", "borrower_trend", "borrower_profile_type",
        # Tiebreakers for the dedup below -- not analysis features themselves.
        "latest_disbursement_ts", "latest_requestid",
    ]

    momo = pd.read_csv(MOMO_PATH, usecols=momo_cols)
    loans = pd.read_csv(LOANS_PATH, usecols=loan_cols)

    # Both files are designed to be one row per entity: agent_msisdn unique
    # per agent in the mart, phonenumber unique per borrower in
    # borrower_history_retail_filtered.csv (it aggregates each borrower's
    # loan history into lifetime_* columns). Report raw-column duplicates
    # BEFORE any key normalization, so a genuine violation of that design is
    # visible on its own, distinct from anything digits() does below.
    n_momo_raw_dupes = momo["agent_msisdn"].dropna().shape[0] - momo["agent_msisdn"].nunique(dropna=True)
    n_loans_raw_dupes = loans["phonenumber"].dropna().shape[0] - loans["phonenumber"].nunique(dropna=True)
    if n_momo_raw_dupes:
        print(f"  WARNING: {n_momo_raw_dupes:,} duplicate raw agent_msisdn "
              f"values in {MOMO_PATH.name} -- expected unique per agent.")
    if n_loans_raw_dupes:
        print(f"  WARNING: {n_loans_raw_dupes:,} duplicate raw phonenumber "
              f"values in {LOANS_PATH.name} -- expected unique per borrower.")

    momo["_id"] = digits(momo["agent_msisdn"])
    loans["_id"] = digits(loans["phonenumber"])

    # If normalization collapses MORE ids than the raw columns already had
    # duplicated, digits() itself is merging genuinely different phone
    # numbers onto the same key -- a much more serious problem than a raw
    # data duplicate, since the dedup below would then silently keep one
    # real borrower's record and discard another's rather than resolving a
    # true duplicate. Surface it loudly rather than let the dedup mask it.
    n_momo_id_dupes = momo["_id"].dropna().shape[0] - momo["_id"].nunique(dropna=True)
    n_loans_id_dupes = loans["_id"].dropna().shape[0] - loans["_id"].nunique(dropna=True)
    if n_momo_id_dupes > n_momo_raw_dupes:
        print(f"  WARNING: digit-normalization collapsed "
              f"{n_momo_id_dupes - n_momo_raw_dupes:,} additional distinct "
              f"agent_msisdn values onto shared ids -- investigate the raw "
              f"values before trusting anything downstream of this join.")
    if n_loans_id_dupes > n_loans_raw_dupes:
        print(f"  WARNING: digit-normalization collapsed "
              f"{n_loans_id_dupes - n_loans_raw_dupes:,} additional distinct "
              f"phonenumber values onto shared ids -- investigate the raw "
              f"values before trusting anything downstream of this join.")

    # Guard: a null join key must never reach the merge -- drop (not fill)
    # rows with a missing/unparseable msisdn on either side, and say how
    # many, so a genuine data problem upstream stays visible instead of
    # silently vanishing into the join.
    n_momo_null = int(momo["_id"].isna().sum())
    n_loans_null = int(loans["_id"].isna().sum())
    if n_momo_null:
        print(f"  WARNING: {n_momo_null:,} momo rows have a missing/unparseable "
              f"agent_msisdn -- dropping before join.")
    if n_loans_null:
        print(f"  WARNING: {n_loans_null:,} loan rows have a missing/unparseable "
              f"phonenumber -- dropping before join.")
    momo = momo.dropna(subset=["_id"])
    loans = loans.dropna(subset=["_id"])

    # Defensive dedup, kept as a safety net matching
    # prepare_borrower_limit_features()'s own guard for this file family --
    # a no-op when the id is actually unique (n_loans_id_dupes == 0 above).
    # If the WARNINGs above fired, this is masking a real problem rather
    # than fixing one; don't trust the join count below until those are
    # resolved.
    loans["latest_disbursement_ts"] = pd.to_datetime(loans["latest_disbursement_ts"], errors="coerce")
    loans["latest_requestid"] = loans["latest_requestid"].astype(str)
    n_loan_rows_before = len(loans)
    loans = loans.sort_values(
        ["_id", "latest_disbursement_ts", "latest_requestid"],
        ascending=[True, False, False],
        na_position="last",
    ).drop_duplicates(subset=["_id"], keep="first")
    if len(loans) != n_loan_rows_before:
        print(f"  deduped loans: {n_loan_rows_before:,} -> {len(loans):,} rows "
              f"(kept most recent record per agent)")

    df = momo.merge(loans, on="_id", how="inner", suffixes=("_momo", "_loan"))
    print(f"Joined population: {len(df):,}")
    df = df.set_index("_id")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    eps = 1.0
    feat = pd.DataFrame(index=df.index)

    # -- Capacity --
    # voucher_vol_3m deliberately excluded: real-data check via
    # characterize_persona_feature_distributions.py showed p1 through p95
    # are all exactly 0 (>=95% of active agents used zero vouchers in 3
    # months). log1p can't fix that -- it compresses long right tails, not a
    # zero-spike-plus-rare-outlier shape -- and RobustScaler on a near-
    # constant column risks amplifying noise from the tiny nonzero subset
    # rather than contributing real signal for the rest of the population.
    feat["commission"] = df["commission"]
    feat["account_balance"] = df["account_balance"]
    feat["cash_out_vol_3m"] = df["cash_out_vol_3m"]
    feat["cash_out_value_3m"] = df["cash_out_value_3m"]
    feat["payment_vol_3m"] = df["payment_vol_3m"]
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
    feat["lifetime_on_time_24h_rate"] = df["lifetime_on_time_24h_rate"]
    feat["lifetime_default_24h_rate"] = df["lifetime_default_24h_rate"]

    feat["has_recent_history"] = df["recent_5_default_24h_rate"].notna().astype(int)
    feat["recent_5_default_24h_rate"] = df["recent_5_default_24h_rate"].fillna(0.0)

    feat["lifetime_avg_hours_to_principal_cure"] = df["lifetime_avg_hours_to_principal_cure"]

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


# Reject a config if its single largest non-noise cluster holds more than
# this share of active agents. "13 clusters, lowest noise%" can still be a
# useless segmentation if one of those 13 swallows 93% of the population --
# this guard is what actually catches that case in the selection below.
DOMINANT_CLUSTER_MAX_SHARE = 0.50

# A DISCOVERY safeguard, not a claim that this many clusters are usable as
# final business personas. Real run: umap_min_dist=0.0 is what actually
# breaks the 85-94% dominant-cluster problem every other setting hits, but
# on raw (pre-log-transform) features it overcorrected to 204 clusters/
# 40.1% noise -- technically under DOMINANT_CLUSTER_MAX_SHARE but useless.
# This bounds candidates on cluster count too, so an extreme like that
# can't win just because no competitor satisfies the dominant-cluster
# check -- without pretending 80 (or even 52) is an acceptable final
# persona count. Treat HDBSCAN's output here as micro-segments for a later
# profiling-based consolidation pass (group behaviourally-equivalent
# micro-segments using borrower_trend/borrower_profile_type -- deliberately
# excluded from clustering itself -- plus capacity/credit-quality outcomes)
# down to a business-facing persona count, not the final segmentation.
MIN_CLUSTERS = 3
MAX_DISCOVERY_CLUSTERS = 80

# Hard qualification alongside cluster-count and dominant-share bounds --
# rules out a candidate that "wins" on those two only by dumping a large
# share of agents into unclassified noise (the 204-cluster/40.1%-noise
# case would have failed this even though it passed DOMINANT_CLUSTER_MAX_SHARE).
NOISE_MAX_PCT = 15.0


def try_hdbscan_configs(df_features: pd.DataFrame, selected_cols: list[str], active_mask: pd.Series):
    """Empirically compare a few HDBSCAN/UMAP sizings; return (chosen_name, chosen_cfg, chosen_result).

    Varies HDBSCAN's min_cluster_size/min_samples, UMAP's n_neighbors (lower
    preserves more local structure), and UMAP's min_dist (lower packs points
    by local similarity instead of spreading them out for visualization --
    umap-learn's own docs recommend min_dist=0.0 specifically for a
    downstream clustering task like this, not its 0.1 default). All aimed
    at splitting a population that would otherwise collapse into one
    dominant blob (see DOMINANT_CLUSTER_MAX_SHARE), without overcorrecting
    into unusably many micro-clusters (see MIN_CLUSTERS/MAX_DISCOVERY_CLUSTERS).

    Real runs found umap_min_dist=0.0 is not one end of a smooth dial -- any
    nonzero value (0.01, 0.02, 0.05 were all tried) lands back near the
    85%+ dominant-cluster baseline, while 0.0 itself is what actually finds
    structure. hdbscan_min_cluster_size at min_dist=0.0 is also non-monotonic
    (100->71 clusters/36.0% dominant, 150->29/85.1%, 200->52/33.7%,
    300->37/61.4%, 500->8/87.2%) -- the 160-260 sweep below exists to check
    whether 200's result is a real local optimum or sampling noise around a
    noisy landscape.
    """
    candidates = [
        ("defaults_1000_150_umap15", {"hdbscan_min_cluster_size": 1000, "hdbscan_min_samples": 150, "umap_n_neighbors": 15}),
        ("scaled_300_40_umap15", {"hdbscan_min_cluster_size": 300, "hdbscan_min_samples": 40, "umap_n_neighbors": 15}),
        ("scaled_150_20_umap15", {"hdbscan_min_cluster_size": 150, "hdbscan_min_samples": 20, "umap_n_neighbors": 15}),
        ("scaled_150_20_umap10", {"hdbscan_min_cluster_size": 150, "hdbscan_min_samples": 20, "umap_n_neighbors": 10}),
        ("scaled_100_15_umap10", {"hdbscan_min_cluster_size": 100, "hdbscan_min_samples": 15, "umap_n_neighbors": 10}),
        ("scaled_150_20_umap10_mindist0", {"hdbscan_min_cluster_size": 150, "hdbscan_min_samples": 20, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
        ("scaled_100_15_umap10_mindist0", {"hdbscan_min_cluster_size": 100, "hdbscan_min_samples": 15, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
        ("scaled_160_24_umap10_mindist0", {"hdbscan_min_cluster_size": 160, "hdbscan_min_samples": 24, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
        ("scaled_180_27_umap10_mindist0", {"hdbscan_min_cluster_size": 180, "hdbscan_min_samples": 27, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
        ("scaled_200_30_umap10_mindist0", {"hdbscan_min_cluster_size": 200, "hdbscan_min_samples": 30, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
        ("scaled_220_33_umap10_mindist0", {"hdbscan_min_cluster_size": 220, "hdbscan_min_samples": 33, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
        ("scaled_240_36_umap10_mindist0", {"hdbscan_min_cluster_size": 240, "hdbscan_min_samples": 36, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
        ("scaled_260_39_umap10_mindist0", {"hdbscan_min_cluster_size": 260, "hdbscan_min_samples": 39, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
        ("scaled_300_40_umap10_mindist0", {"hdbscan_min_cluster_size": 300, "hdbscan_min_samples": 40, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
        ("scaled_500_75_umap10_mindist0", {"hdbscan_min_cluster_size": 500, "hdbscan_min_samples": 75, "umap_n_neighbors": 10, "umap_min_dist": 0.0}),
        ("scaled_100_15_umap10_mindist001", {"hdbscan_min_cluster_size": 100, "hdbscan_min_samples": 15, "umap_n_neighbors": 10, "umap_min_dist": 0.01}),
        ("scaled_100_15_umap10_mindist002", {"hdbscan_min_cluster_size": 100, "hdbscan_min_samples": 15, "umap_n_neighbors": 10, "umap_min_dist": 0.02}),
        ("scaled_100_15_umap10_mindist005", {"hdbscan_min_cluster_size": 100, "hdbscan_min_samples": 15, "umap_n_neighbors": 10, "umap_min_dist": 0.05}),
    ]
    results = {}
    n_active = int(active_mask.sum())
    for name, overrides in candidates:
        cfg = _get_clustering_config(overrides)
        out = flag_anomalies(df_features, selected_cols, active_mask, config=cfg)
        labels = out["anomaly_cluster_hdb_raw"]
        noise_pct = (labels == -1).sum() / n_active * 100
        cluster_sizes = labels[labels != -1].value_counts()
        n_clusters = len(cluster_sizes)
        dominant_share = (cluster_sizes.max() / n_active) if n_clusters else 1.0
        # Top-5 cluster shares (not just the largest) -- distinguishes "52
        # meaningful micro-segments" from "3 large clusters + 49 fragments",
        # which the dominant-cluster guard alone can't tell apart.
        top5_share_pct = (cluster_sizes.sort_values(ascending=False).head(5) / n_active * 100).round(1).tolist()
        print(f"  {name}: n_clusters={n_clusters}, noise={noise_pct:.1f}%, "
              f"largest_cluster={dominant_share:.1%} of active agents, "
              f"top5={top5_share_pct}%")
        results[name] = (cfg, out, noise_pct, n_clusters, dominant_share, top5_share_pct)

    # Full ranked table of every candidate (not just the winner) -- same
    # sort order the selector below uses (qualifies first, then the
    # hierarchical tiebreak), so it's legible at a glance which candidates
    # were even in contention and why one beat another near the boundary.
    print("\n  --- candidate summary (all configs) ---")
    summary_rows = []
    for name, (cfg, out, noise_pct, n_clusters, dominant_share, top5_share_pct) in results.items():
        qualifies = (
            MIN_CLUSTERS <= n_clusters <= MAX_DISCOVERY_CLUSTERS
            and dominant_share <= DOMINANT_CLUSTER_MAX_SHARE
            and noise_pct <= NOISE_MAX_PCT
        )
        summary_rows.append({
            "name": name,
            "min_cluster_size": cfg["hdbscan_min_cluster_size"],
            "min_samples": cfg["hdbscan_min_samples"],
            "umap_n_neighbors": cfg["umap_n_neighbors"],
            "umap_min_dist": cfg["umap_min_dist"],
            "n_clusters": n_clusters,
            "noise_pct": round(noise_pct, 1),
            "dominant_pct": round(dominant_share * 100, 1),
            "top5_pct": top5_share_pct,
            "qualifies": qualifies,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["qualifies", "n_clusters", "dominant_pct", "noise_pct"],
        ascending=[False, True, True, True],
    )
    with pd.option_context("display.max_rows", None, "display.width", 220, "display.max_colwidth", 80):
        print(summary_df.to_string(index=False))

    # Hard qualification on three independent bounds -- cluster count in a
    # discovery-safe range, dominant-cluster share capped, noise capped --
    # then, among whatever qualifies, a HIERARCHICAL tiebreak: fewer
    # clusters first, then lower dominant-cluster share, then lower noise,
    # each only breaking ties left by the one before it. Deliberately not a
    # single blended score: once a candidate already clears the dominant-
    # share and noise bounds, further minimizing either one just rewards
    # additional fragmentation (more clusters) for a marginal gain on an
    # axis that no longer matters as much as cluster count does.
    #
    # Nuance worth knowing: because fewer clusters is the FIRST tiebreak, a
    # candidate at "12 clusters / 49% dominant / 14.8% noise" beats one at
    # "40 clusters / 30% dominant / 5% noise" the moment both clear the
    # hard bounds -- the selector stops caring how much better the second
    # one did on dominant-share/noise once cluster count alone decides it.
    # That's intentional for now (fewer clusters is a real business-value
    # axis, not just tie-breaking noise), but it means the three hard
    # bounds are carrying real weight: a candidate barely inside them still
    # wins outright against one that cleared them by a wide margin. Revisit
    # if the full candidate table above shows this producing a bad pick
    # near the boundary.
    qualified = {
        k: v for k, v in results.items()
        if MIN_CLUSTERS <= v[3] <= MAX_DISCOVERY_CLUSTERS
        and v[4] <= DOMINANT_CLUSTER_MAX_SHARE
        and v[2] <= NOISE_MAX_PCT
    }
    if qualified:
        pool = qualified
        best_name = min(pool, key=lambda k: (pool[k][3], pool[k][4], pool[k][2]))
    else:
        # Relax the three bounds one at a time, strictest first, so
        # degenerate data still returns something rather than crashing.
        # Lowest noise% is the tiebreaker within whichever tier is used.
        no_noise_bound = {k: v for k, v in results.items()
                           if MIN_CLUSTERS <= v[3] <= MAX_DISCOVERY_CLUSTERS
                           and v[4] <= DOMINANT_CLUSTER_MAX_SHARE}
        in_range = {k: v for k, v in results.items() if MIN_CLUSTERS <= v[3] <= MAX_DISCOVERY_CLUSTERS}
        min_only = {k: v for k, v in results.items() if v[3] >= MIN_CLUSTERS}
        pool = no_noise_bound or in_range or min_only or results
        if no_noise_bound:
            tier = (f"{MIN_CLUSTERS}-{MAX_DISCOVERY_CLUSTERS} clusters and <= "
                    f"{DOMINANT_CLUSTER_MAX_SHARE:.0%} dominant share (none also stayed "
                    f"under {NOISE_MAX_PCT:.0f}% noise)")
        elif in_range:
            tier = (f"{MIN_CLUSTERS}-{MAX_DISCOVERY_CLUSTERS} clusters (none also stayed "
                    f"under {DOMINANT_CLUSTER_MAX_SHARE:.0%} dominant-cluster share)")
        elif min_only:
            tier = f">={MIN_CLUSTERS} clusters (none stayed within the {MIN_CLUSTERS}-{MAX_DISCOVERY_CLUSTERS} discovery range)"
        else:
            tier = "all candidates"
        print(f"  WARNING: no candidate satisfied all three hard bounds "
              f"({MIN_CLUSTERS}-{MAX_DISCOVERY_CLUSTERS} clusters, <= "
              f"{DOMINANT_CLUSTER_MAX_SHARE:.0%} dominant-cluster share, <= "
              f"{NOISE_MAX_PCT:.0f}% noise) -- falling back to {tier} and picking "
              f"lowest noise%. Segmentation quality may still be poor; consider "
              f"widening the candidate grid further.")
        best_name = min(pool, key=lambda k: pool[k][2])
    print(f"  -> selected: {best_name}")
    cfg, out, noise_pct, n_clusters, dominant_share, top5_share_pct = results[best_name]
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

    # DEVIATION from run_extrafloat_segmentation.py's own production
    # convention: it feeds flag_anomalies raw, un-logged features (confirmed
    # by code trace -- prepare_features() returns features_df_raw, and that
    # exact raw frame is what reaches flag_anomalies there, never its own
    # log-transformed X_scaled/X_pca). That convention was validated against
    # segmentation's own MoMo-KPI-only feature set. This script also mixes in
    # heavy-tailed monetary columns (avg_loan_size_lifetime, account_balance,
    # commission, ...) that RobustScaler's median/IQR centering alone doesn't
    # fix the skew of. An exhaustive real-data tuning pass (8 UMAP/HDBSCAN
    # parameter combinations) found only two regimes -- ~93% of agents in one
    # cluster, or umap_min_dist=0.0 overcorrecting to 204 micro-clusters --
    # with no middle ground, which is what motivated trying the feature space
    # itself next: X_final is the same selected_cols, log1p+winsorized (see
    # _apply_log_winsorize) and NaN-imputed (median, matching this repo's own
    # "safety re-impute after pruning" pattern in prepare_features()), instead
    # of the raw feat[selected_cols] this used to pass. flag_anomalies still
    # does its own RobustScaler+PCA/UMAP internally via _get_active_pca --
    # only the values handed to that scaler change here, not the pipeline
    # shape.
    df_features = X_final.copy()

    print("\n=== Step 4: HDBSCAN sizing comparison ===")
    active_mask = pd.Series(True, index=df_features.index)
    chosen_name, chosen_cfg, anomaly_out = try_hdbscan_configs(df_features, selected_cols, active_mask)

    print("\n=== Step 5: assemble output ===")
    result = df[["agent_msisdn", "phonenumber", "borrower_trend", "borrower_profile_type"]].copy()
    for c in all_cols:
        result[c] = feat[c]
    result["persona_cluster"] = anomaly_out["anomaly_cluster_hdb_raw"]
    result["is_anomaly"] = anomaly_out["is_anomaly"]
    result["is_global_anomaly"] = anomaly_out["is_global_anomaly"]
    result["is_local_anomaly"] = anomaly_out["is_local_anomaly"]
    result["lof_score"] = anomaly_out["lof_score"]

    engine_path = Path(ENGINE_OUTPUT_PATH)
    if engine_path.exists():
        # capacity_cap/combined_cap are intermediate-stage columns (see
        # docs/engine_output_data_dictionary.md) only present when the
        # engine run used keep_intermediate=True -- assigned_limit/risk_tier
        # are the only ones guaranteed by every run. Check the real header
        # first so a run without keep_intermediate degrades gracefully
        # (fewer profiling columns) instead of crashing on a missing usecol.
        wanted_cols = ["msisdn", "assigned_limit", "risk_tier", "capacity_cap", "combined_cap"]
        available_cols = set(pd.read_csv(engine_path, nrows=0).columns)
        missing_cols = [c for c in wanted_cols if c not in available_cols]
        read_cols = [c for c in wanted_cols if c in available_cols]
        if missing_cols:
            print(f"  WARNING: {ENGINE_OUTPUT_PATH} is missing {missing_cols} "
                  f"(likely produced without keep_intermediate=True) -- "
                  f"continuing without them.")
        eng = pd.read_csv(engine_path, usecols=read_cols)
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

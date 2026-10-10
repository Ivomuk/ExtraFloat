"""
build_historical_pit_transaction_features.py
==================================================
Stage 4A.1 -- RESEARCH-ONLY Phase 2.1 point-in-time (PIT) transaction-feature
reconstruction. Produces NO PD scores, NO C3 application, no `k` selection,
no Capacity(F) combination. Output is reconstructed Phase 2.1 feature
vectors plus audit/status columns only -- never named `cal_pd` or implied
equivalent to the live pipeline's actual output (which also depends on
other phases this script doesn't touch).

WHY THIS SCRIPT EXISTS: `audit_historical_pd_rescoring_feasibility.py`
found Phase 2.1 (transaction/commission/balance features) not PIT-
reconstructable with the current production pipeline
(`pd_model/preprocessing/transaction_features.py`), specifically because
`run_phase_2_1_richer_tx_behaviour` computes recency (`days_since_
snapshot`) relative to `df["tbl_dt"].max()` -- the max date in whatever
batch happens to be loaded, not a caller-supplied historical decision
date. A real multi-month mart export (7 distinct month-end snapshots,
2026-01 through 2026-07) was subsequently confirmed to exist, which
means the remaining blocker is purely an implementation gap, not a data-
availability wall. This script tests whether that gap is closeable by
reusing a proven, already-verified as-of-join pattern from elsewhere in
this exact repo.

**This script NEVER modifies, imports from, or calls**
`pd_model/preprocessing/transaction_features.py` or `pd_model/
run_pipeline.py` -- the production PD pipeline is protected while
historical reconstruction is established as scientifically valid, as a
separate, read-only research path. If this script's reconstruction is
later judged sound, `audit_historical_pd_rescoring_feasibility.py` may be
re-run (unmodified) to see whether Phase 2.1's disposition changes --
that decision and any resulting change to the audit script are
deliberately NOT made here.

CONFIRMED FROM READING `pd_model/preprocessing/transaction_features.py`
IN FULL: `run_phase_2_1_richer_tx_behaviour` has exactly ONE batch-
relative computation in the entire file --

    if "tbl_dt" in df.columns:
        ref_date = df["tbl_dt"].max()
        df["days_since_snapshot"] = (ref_date - df["tbl_dt"]).dt.days

Every other feature block is pure row-level arithmetic on that row's own
columns. This script restates every row-level block verbatim (same
conditions, same formulas, same defensive `if all(c in df.columns ...)`
gating), and replaces ONLY the recency block: `days_since_snapshot` here
comes from the as-of join below (each episode's own `loan_date` minus
the matched historical snapshot's date), never a batch maximum.

ONE REAL, UNRESOLVED CAVEAT, flagged rather than hidden: several Phase
2.1 blocks consume, but do not themselves compute, upstream cluster/peer
aggregate columns (`commission_cluster_mean`, `vol_3m_cluster_mean`,
`cluster_avg_commission`, `cluster_avg_vol_3m`, `cash_in_peers_3m`,
`cash_in_vol_3m`). Those columns are produced by an earlier, upstream
phase this investigation did not read. Their OWN point-in-time validity
(computed fresh per historical month, vs. from some other reference
batch) is NOT verified by this script -- it passes through whatever is
present on the matched historical mart row as-is and reports, per row,
whether any such column was actually present and non-null
(`cluster_columns_present`), rather than silently assuming their
provenance is fine.

REUSES (restates, never imports -- one-way `scripts/` layering
convention) the exact as-of-join pattern already built, tested, and
verified in `scripts/build_loan_episode_capacity_dataset.py`:
`pd.merge_asof(direction="backward", allow_exact_matches=False)`, with
the same `TypeError` fallback for older pandas, the same strict `<`
requirement (an exact-date snapshot never matches), and the same
never-backfill discipline (no eligible prior snapshot -> left as a
distinct, explicit status, never zero-filled or borrowed from a later
snapshot). `_parse_mart_date` is also restated verbatim, including its
documented YYYYMMDD-vs-plain-date dtype guard (a numeric `tbl_dt` column
parsed with a plain `pd.to_datetime` silently collapses every distinct
date to ~1970-01-01 with no error -- confirmed directly in the episode
builder's own history; this script routes numeric date columns through
the same explicit `format="%Y%m%d"` parse to avoid it).

LEAKAGE ASSERTION: `assert_no_leakage` is called unconditionally before
any feature is computed. For every row classified as reconstructed, it
asserts `fundamentals_snapshot_date < loan_date` and raises loudly (not a
warning) on any violation -- this is the single most load-bearing check
in the script.

Imports `pd_model.config.model_config.DEFAULT_CONFIG` directly for `eps`
(not a `scripts/` module, so this doesn't violate the layering
convention) -- reusing the real constant rather than risking value drift
from a hand-copied literal.

Usage:
    python scripts\\build_historical_pit_transaction_features.py ^
        --loan-training-file data\\state_data_20260910_retail_filtered.csv ^
        --transaction-mart-file data\\mfs_daily_agent_mart_202607_retail_filtered.csv ^
        --out-prefix stage4a1_phase21_pit
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pd_model.config.model_config import DEFAULT_CONFIG  # noqa: E402
from segmentation.borrower_persona_clustering import digits  # noqa: E402

LOAN_REQUIRED_COLS = ["disbursement_fid", "agent_msisdn", "loan_date"]

CLUSTER_PEER_COLS = [
    "commission_cluster_mean", "vol_3m_cluster_mean",
    "cluster_avg_commission", "cluster_avg_vol_3m",
    "cash_in_peers_3m", "cash_in_vol_3m",
]

STATUS_RECONSTRUCTED = "PIT_PHASE21_RECONSTRUCTED"
STATUS_UNAVAILABLE = "PIT_PHASE21_UNAVAILABLE_NO_PRIOR_SNAPSHOT"


def _read_csv_fast(path: Path) -> pd.DataFrame:
    """Restated from build_loan_episode_capacity_dataset.py (one-way
    scripts/ layering convention) -- prefers pyarrow for large files,
    falls back to the plain engine if unavailable."""
    try:
        return pd.read_csv(path, sep=",", encoding="utf-8-sig", engine="pyarrow")
    except (ImportError, ValueError):
        return pd.read_csv(path, sep=",", encoding="utf-8-sig")


def _parse_mart_date(series: pd.Series) -> pd.Series:
    """Restated verbatim from build_loan_episode_capacity_dataset.py.
    A numeric YYYYMMDD column (e.g. tbl_dt == 20260731 as int64) must be
    routed through an explicit format="%Y%m%d" parse -- a plain
    pd.to_datetime on a raw numeric Series silently collapses every
    distinct date to ~1970-01-01 with no error, confirmed directly in
    this exact repo's history."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def load_loan_episodes(path: Path) -> pd.DataFrame:
    """Restated subset of build_loan_episode_capacity_dataset.py's loader
    -- only the columns this script needs (disbursement_fid, agent_msisdn,
    loan_date)."""
    df = _read_csv_fast(path)
    if "agent_msisdn" not in df.columns:
        alt_col = next((c for c in ("msisdn", "phonenumber") if c in df.columns), None)
        if alt_col is not None:
            df = df.rename(columns={alt_col: "agent_msisdn"})
            print(f"NOTE: loan-training-file has no 'agent_msisdn' column -- using '{alt_col}' instead.")
    missing = [c for c in LOAN_REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")
    df = df.copy()
    df["loan_date"] = pd.to_datetime(df["loan_date"], errors="coerce")
    df["_id"] = digits(df["agent_msisdn"])
    n_bad_dates = int(df["loan_date"].isna().sum())
    if n_bad_dates:
        print(f"NOTE: {n_bad_dates} episode(s) dropped for unparseable loan_date.")
        df = df[df["loan_date"].notna()]
    return df


def load_mart(path: Path) -> pd.DataFrame:
    """Restated from build_loan_episode_capacity_dataset.py's load_mart,
    EXCEPT this script keeps EVERY raw mart column present (not just the
    4 capacity fundamentals) -- Phase 2.1's row-level feature blocks need
    vol_1m/3m/6m, commission, account_balance, cust_1m/3m, cash_in/
    cash_out/payment columns, and (optionally) the upstream cluster/peer
    columns, whichever of these the real export actually contains."""
    txn = _read_csv_fast(path)
    msisdn_col = "agent_msisdn" if "agent_msisdn" in txn.columns else "msisdn"
    if msisdn_col not in txn.columns:
        sys.exit(f"ERROR: {path} has neither 'agent_msisdn' nor 'msisdn'. Columns present: {list(txn.columns)}")
    date_col = "snapshot_dt" if "snapshot_dt" in txn.columns else ("tbl_dt" if "tbl_dt" in txn.columns else None)
    if date_col is None:
        sys.exit(f"ERROR: {path} has neither 'snapshot_dt' nor 'tbl_dt' -- cannot perform a "
                  f"point-in-time join without a snapshot date.")
    txn = txn.copy()
    txn["_id"] = digits(txn[msisdn_col])
    txn["fundamentals_snapshot_date"] = _parse_mart_date(txn[date_col]).dt.normalize()
    n_bad = int(txn["fundamentals_snapshot_date"].isna().sum())
    if n_bad:
        print(f"NOTE: {n_bad} mart row(s) dropped for unparseable {date_col}.")
        txn = txn[txn["fundamentals_snapshot_date"].notna()]
    n_dates = txn["fundamentals_snapshot_date"].nunique()
    print(f"Transaction mart: {txn['_id'].nunique():,} unique agent(s) across {n_dates:,} "
          f"distinct snapshot date(s) ({sorted(d.date().isoformat() for d in txn['fundamentals_snapshot_date'].unique())}).")
    return txn


def attach_prior_snapshot(episodes: pd.DataFrame, mart: pd.DataFrame) -> pd.DataFrame:
    """The restated merge_asof pattern verbatim (both the primary call and
    the TypeError fallback) from attach_pretrade_fundamentals. Adds
    `days_since_snapshot` (loan_date - fundamentals_snapshot_date, in
    days -- computed per row from that row's OWN loan_date, never a batch
    max) and `reconstruction_status` -- never silently backfilled."""
    ep = episodes.sort_values("loan_date").reset_index(drop=True)
    mt = mart.sort_values("fundamentals_snapshot_date").reset_index(drop=True)
    mart_value_cols = [c for c in mt.columns if c not in ("_id", "fundamentals_snapshot_date")]

    try:
        merged = pd.merge_asof(
            ep, mt, left_on="loan_date", right_on="fundamentals_snapshot_date",
            left_by="_id", right_by="_id", direction="backward", allow_exact_matches=False,
        )
        join_method = "merge_asof(direction='backward', allow_exact_matches=False)"
    except TypeError:
        merged = pd.merge_asof(
            ep, mt, left_on="loan_date", right_on="fundamentals_snapshot_date",
            left_by="_id", right_by="_id", direction="backward",
        )
        exact_match = merged["fundamentals_snapshot_date"] == merged["loan_date"]
        for c in ["fundamentals_snapshot_date"] + mart_value_cols:
            merged.loc[exact_match, c] = np.nan
        join_method = "merge_asof(direction='backward') + explicit exact-match nulling"
    print(f"Fundamentals join method: {join_method}")

    merged["days_since_snapshot"] = (merged["loan_date"] - merged["fundamentals_snapshot_date"]).dt.days
    merged["reconstruction_status"] = np.where(
        merged["fundamentals_snapshot_date"].notna(), STATUS_RECONSTRUCTED, STATUS_UNAVAILABLE,
    )
    n_unavailable = int((merged["reconstruction_status"] == STATUS_UNAVAILABLE).sum())
    if n_unavailable:
        print(f"NOTE: {n_unavailable:,} of {len(merged):,} episode(s) "
              f"({n_unavailable / len(merged) * 100:.1f}%) have no eligible prior mart snapshot -- "
              f"status={STATUS_UNAVAILABLE}, features left NaN, never backfilled.")
    return merged


def assert_no_leakage(df: pd.DataFrame) -> None:
    """The single most load-bearing check in this script. Raises loudly
    (never a warning) if any reconstructed row has a snapshot on or after
    its own loan_date."""
    reconstructed = df[df["reconstruction_status"] == STATUS_RECONSTRUCTED]
    violations = reconstructed[reconstructed["fundamentals_snapshot_date"] >= reconstructed["loan_date"]]
    if len(violations):
        raise AssertionError(
            f"LEAKAGE DETECTED: {len(violations)} reconstructed row(s) have "
            f"fundamentals_snapshot_date >= loan_date. This must never happen -- aborting."
        )
    print(f"Leakage assertion passed: all {len(reconstructed):,} reconstructed row(s) have "
          f"fundamentals_snapshot_date < loan_date.")


def restate_phase_2_1_features(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Block-for-block restatement of pd_model/preprocessing/
    transaction_features.py's run_phase_2_1_richer_tx_behaviour, with the
    ONE deliberate change: `days_since_snapshot` is taken from the as-of
    join (already relative to this episode's own loan_date), never
    recomputed from a batch-wide tbl_dt.max(). Every other block is
    copied verbatim: same conditions, same formulas, same defensive
    `if all(c in df.columns ...)` gating per block."""
    df = df.copy()

    # A) Strong inactivity structure.
    if all(c in df.columns for c in ["vol_1m", "vol_3m", "vol_6m"]):
        df["is_fully_inactive_6m"] = (
            (df["vol_1m"].fillna(0) == 0) & (df["vol_3m"].fillna(0) == 0) & (df["vol_6m"].fillna(0) == 0)
        ).astype(int)
        df["is_consecutively_inactive"] = (
            (df["vol_1m"].fillna(0) == 0) & (df["vol_3m"].fillna(0) == 0)
        ).astype(int)

    # B) Activity restart / recovery signal.
    if all(c in df.columns for c in ["vol_1m", "vol_3m"]):
        df["activity_restart_flag"] = ((df["vol_1m"] > 0) & (df["vol_3m"] == df["vol_1m"])).astype(int)

    # C) Conditional activity intensity.
    if "vol_3m" in df.columns:
        df["vol_3m_if_active"] = df["vol_3m"].where(df["vol_3m"] > 0, np.nan)

    # D) Commission dependency risk.
    if all(c in df.columns for c in ["commission", "vol_3m"]):
        df["commission_without_activity_flag"] = (
            (df["commission"] > 0) & (df["vol_3m"].fillna(0) == 0)
        ).astype(int)

    # Trend direction flags.
    if all(c in df.columns for c in ["vol_1m", "vol_3m", "vol_6m"]):
        df["consistent_volume_decline_flag"] = (
            (df["vol_1m"] < df["vol_3m"] / 3.0) & (df["vol_3m"] < df["vol_6m"] / 2.0)
        ).astype(int)
        df["consistent_volume_growth_flag"] = (
            (df["vol_1m"] > df["vol_3m"] / 3.0) & (df["vol_3m"] > df["vol_6m"] / 2.0)
        ).astype(int)

    # Liquidity & balance stress.
    if "account_balance" in df.columns:
        df["low_balance_flag"] = (df["account_balance"] <= 0).astype(int)
    if all(c in df.columns for c in ["account_balance", "vol_3m"]):
        df["balance_to_vol_3m_ratio"] = df["account_balance"] / (df["vol_3m"] + eps)
    if all(c in df.columns for c in ["average_balance", "vol_3m"]):
        df["avg_balance_to_vol_3m_ratio"] = df["average_balance"] / (df["vol_3m"] + eps)
    if all(c in df.columns for c in ["account_balance", "average_balance"]):
        df["balance_drawdown_flag"] = (df["account_balance"] < 0.5 * df["average_balance"]).astype(int)

    # Customer & peer dependence.
    if all(c in df.columns for c in ["cust_1m", "cust_3m"]):
        df["cust_concentration_flag"] = ((df["cust_1m"] / (df["cust_3m"] + eps)) > 0.8).astype(int)
    if all(c in df.columns for c in ["cash_in_peers_3m", "cash_in_vol_3m"]):
        df["peer_dependency_ratio"] = df["cash_in_peers_3m"] / (df["cash_in_vol_3m"] + eps)
        df["high_peer_dependency_flag"] = (df["peer_dependency_ratio"] > 0.7).astype(int)

    # Transaction mix & net flow stress.
    if all(c in df.columns for c in ["cash_in_value_3m", "cash_out_value_3m"]):
        df["net_cash_flow_3m"] = df["cash_in_value_3m"] - df["cash_out_value_3m"]
        df["net_cash_flow_negative_flag"] = (df["net_cash_flow_3m"] < 0).astype(int)
    if all(c in df.columns for c in ["payment_value_3m", "vol_3m"]):
        df["payment_intensity_ratio"] = df["payment_value_3m"] / (df["vol_3m"] + eps)

    # Stress acceleration flags.
    if all(c in df.columns for c in ["vol_1m", "vol_3m"]):
        df["sharp_volume_drop_flag"] = ((df["vol_1m"] / (df["vol_3m"] + eps)) < 0.3).astype(int)
    if all(c in df.columns for c in ["commission", "commission_cluster_mean"]):
        df["commission_drop_flag"] = (df["commission"] < 0.5 * df["commission_cluster_mean"]).astype(int)

    # 1) Cluster-relative commission and volume.
    if "commission" in df.columns and "commission_cluster_mean" in df.columns:
        df["commission_vs_cluster_mean_ratio"] = df["commission"] / (df["commission_cluster_mean"] + eps)
        df["commission_vs_cluster_mean_diff"] = df["commission"] - df["commission_cluster_mean"]
    if "vol_3m" in df.columns and "vol_3m_cluster_mean" in df.columns:
        df["vol_3m_vs_cluster_mean_ratio"] = df["vol_3m"] / (df["vol_3m_cluster_mean"] + eps)
        df["vol_3m_vs_cluster_mean_diff"] = df["vol_3m"] - df["vol_3m_cluster_mean"]

    # 2) Commission intensity (per volume).
    if "commission" in df.columns and "vol_3m" in df.columns:
        df["commission_per_vol_3m"] = df["commission"] / (df["vol_3m"] + eps)
    if (
        "cluster_avg_commission" in df.columns
        and "cluster_avg_vol_3m" in df.columns
        and "commission_per_vol_3m" in df.columns
    ):
        df["cluster_commission_per_vol_3m"] = df["cluster_avg_commission"] / (df["cluster_avg_vol_3m"] + eps)
        df["commission_per_vol_vs_cluster_ratio"] = df["commission_per_vol_3m"] / (
            df["cluster_commission_per_vol_3m"] + eps
        )

    # 3) Volume trajectory, intensity and volatility (1m / 3m / 6m).
    base_names = ["cash_out_vol", "cash_in_vol", "payment_vol", "vol"]
    for base in base_names:
        col_1m, col_3m, col_6m = f"{base}_1m", f"{base}_3m", f"{base}_6m"
        have_1m, have_3m, have_6m = (col_1m in df.columns, col_3m in df.columns, col_6m in df.columns)

        if have_3m:
            df[f"{base}_avg_monthly_3m"] = df[col_3m] / 3.0
        if have_6m:
            df[f"{base}_avg_monthly_6m"] = df[col_6m] / 6.0
        if have_1m and have_3m:
            df[f"{base}_share_1m_of_3m"] = df[col_1m] / (df[col_3m] + eps)
            prev2m = (df[col_3m] - df[col_1m]) / 2.0
            df[f"{base}_growth_1m_vs_prev2m"] = df[col_1m] / (prev2m + eps)
        if have_3m and have_6m:
            df[f"{base}_share_3m_of_6m"] = df[col_3m] / (df[col_6m] + eps)
            prev3m = (df[col_6m] - df[col_3m]) / 3.0
            df[f"{base}_growth_3m_vs_prev3m"] = df[col_3m] / (prev3m + eps)
        if have_1m and have_6m:
            df[f"{base}_share_1m_of_6m"] = df[col_1m] / (df[col_6m] + eps)
        if have_1m and have_3m and have_6m:
            m1 = df[col_1m]
            m2 = (df[col_3m] - df[col_1m]) / 2.0
            m3 = (df[col_6m] - df[col_3m]) / 3.0
            monthly = np.vstack([m1.values, m2.values, m3.values]).T
            mean_monthly = monthly.mean(axis=1)
            std_monthly = monthly.std(axis=1)
            df[f"{base}_monthly_volatility_proxy"] = std_monthly
            df[f"{base}_monthly_volatility_cv"] = std_monthly / (mean_monthly + eps)

    # 4) Explicit inactivity flags per horizon.
    for horizon in ["1m", "3m", "6m"]:
        col, flag_col = f"vol_{horizon}", f"is_inactive_{horizon}"
        if col in df.columns:
            df[flag_col] = (df[col].fillna(0) == 0).astype(int)
    inactivity_flag_cols = [c for c in ["is_inactive_1m", "is_inactive_3m", "is_inactive_6m"] if c in df.columns]
    if inactivity_flag_cols:
        df["num_inactive_horizons"] = df[inactivity_flag_cols].sum(axis=1)
    if all(c in df.columns for c in ["is_inactive_1m", "is_inactive_3m", "is_inactive_6m"]):
        df["max_inactivity_horizon_flag"] = df["is_inactive_1m"] + df["is_inactive_3m"] + df["is_inactive_6m"]

    # 5) Recency -- THE ONE DELIBERATE CHANGE: already computed per-row by
    # attach_prior_snapshot from this episode's own loan_date, never a
    # batch-wide tbl_dt.max(). Nothing to do here except confirm presence.
    if "days_since_snapshot" not in df.columns:
        df["days_since_snapshot"] = np.nan

    # Cluster/peer-column provenance caveat: per-row, not assumed.
    present_cluster_cols = [c for c in CLUSTER_PEER_COLS if c in df.columns]
    if present_cluster_cols:
        df["cluster_columns_present"] = df[present_cluster_cols].notna().any(axis=1)
    else:
        df["cluster_columns_present"] = False

    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loan-training-file", type=Path, required=True)
    ap.add_argument("--transaction-mart-file", type=Path, required=True)
    ap.add_argument("--out-prefix", type=str, default="stage4a1_phase21_pit")
    args = ap.parse_args(argv)

    print(f"\n{'#' * 100}\nSTAGE 4A.1 -- RESEARCH-ONLY PHASE 2.1 PIT FEATURE RECONSTRUCTION\n"
          f"Produces NO PD scores, NO C3 application, no k selection, no Capacity(F) combination.\n"
          f"Never modifies production pd_model/preprocessing/transaction_features.py.\n"
          f"Upstream cluster/peer-column provenance (commission_cluster_mean etc.) is NOT verified "
          f"by this script -- reported per row via cluster_columns_present, never assumed.\n{'#' * 100}")

    if not args.loan_training_file.exists():
        sys.exit(f"ERROR: {args.loan_training_file} not found.")
    if not args.transaction_mart_file.exists():
        sys.exit(f"ERROR: {args.transaction_mart_file} not found.")

    episodes = load_loan_episodes(args.loan_training_file)
    print(f"{len(episodes):,} loan episode(s) loaded.")
    mart = load_mart(args.transaction_mart_file)

    merged = attach_prior_snapshot(episodes, mart)
    assert_no_leakage(merged)
    featured = restate_phase_2_1_features(merged, DEFAULT_CONFIG.eps)

    status_counts = featured["reconstruction_status"].value_counts()
    print("\n-- Reconstruction status --")
    print(status_counts.to_string())

    age_bins = [-1, 7, 30, 60, 90, np.inf]
    age_labels = ["0-7d", "8-30d", "31-60d", "61-90d", ">90d"]
    reconstructed = featured[featured["reconstruction_status"] == STATUS_RECONSTRUCTED]
    age_band_counts = pd.cut(reconstructed["days_since_snapshot"], bins=age_bins, labels=age_labels).value_counts()
    print("\n-- days_since_snapshot age bands (reconstructed rows only) --")
    print(age_band_counts.sort_index().to_string())

    feature_cols = [c for c in featured.columns if c not in merged.columns or c == "days_since_snapshot"]
    out_cols = (["disbursement_fid", "agent_msisdn", "loan_date", "fundamentals_snapshot_date",
                 "days_since_snapshot", "reconstruction_status", "cluster_columns_present"]
                + [c for c in feature_cols if c not in
                   ("days_since_snapshot", "reconstruction_status", "cluster_columns_present")])
    out_cols = [c for c in dict.fromkeys(out_cols) if c in featured.columns]

    features_path = f"{args.out_prefix}_phase21_pit_features.csv"
    featured[out_cols].to_csv(features_path, index=False)
    print(f"\nWrote {features_path}")

    summary_path = f"{args.out_prefix}_phase21_pit_summary.csv"
    summary_df = pd.DataFrame({
        "reconstruction_status": status_counts.index,
        "n_episodes": status_counts.values,
    })
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

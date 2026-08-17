"""
Profile the commission_raw==0 / commission_6m_recomputed>=substantial bucket
(the ~2,099-agent group found earlier) against every other agent, across a
set of columns, to see what distinguishes them -- e.g. do they look like
large aggregator/corporate/system accounts rather than ordinary retail
agents?

Uses the same bucket definition as check_commission_raw_vs_recomputed.py /
sample_commission_mismatch_rows.py:
    commission_raw == 0  AND  commission_6m_recomputed >= 40,806

Default profiled columns: agent_profile, account_balance, average_balance,
plus anything else in the file matching a broad keyword list (region,
district, territory, product, status, ova, category, segment, activation,
tenure, kyc, verified, date/dt, birth/dob) -- only columns that actually
exist in the file are used. Pass --profile-cols to add more by name.

Usage:
    python profile_commission_mismatch_bucket.py ^
        --transaction-file data\\mfs_daily_agent_mart_20260731.csv ^
        --output-file output\\engine_test_output.csv ^
        --profile-cols some_other_column,another_column
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_AUTO_KEYWORDS = [
    "region", "district", "territory", "product", "status", "ova",
    "category", "segment", "activation", "tenure", "kyc", "verified",
    "date", "dt", "birth", "dob", "profile", "balance", "channel",
    "type", "class", "grade", "network",
]


def _norm_msisdn(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def _profile_numeric(col: str, bucket: pd.Series, rest: pd.Series):
    b = pd.to_numeric(bucket, errors="coerce")
    r = pd.to_numeric(rest, errors="coerce")
    print(f"\n--- {col} (numeric) ---")
    summary = pd.DataFrame({
        "bucket": b.describe(),
        "rest_of_population": r.describe(),
    })
    print(summary.to_string())


def _profile_categorical(col: str, bucket: pd.Series, rest: pd.Series, top_n: int = 10):
    print(f"\n--- {col} (categorical, top {top_n}) ---")
    b_counts = bucket.value_counts(normalize=True, dropna=False).head(top_n)
    r_counts = rest.value_counts(normalize=True, dropna=False).head(top_n)
    combined = pd.DataFrame({"bucket_pct": b_counts, "rest_pct": r_counts}).fillna(0.0)
    combined["bucket_pct"] = (combined["bucket_pct"] * 100).round(1)
    combined["rest_pct"] = (combined["rest_pct"] * 100).round(1)
    print(combined.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transaction-file", default="data/mfs_daily_agent_mart_20260731.csv")
    ap.add_argument("--output-file", default="output/engine_test_output.csv")
    ap.add_argument("--profile-cols", default="", help="Comma-separated extra column names to profile")
    ap.add_argument("--out", default="commission_mismatch_bucket_profile.csv")
    ap.add_argument("--categorical-max-unique", type=int, default=30)
    args = ap.parse_args()

    txn_path = Path(args.transaction_file)
    if not txn_path.exists():
        sys.exit(f"ERROR: transaction file not found: {txn_path}")

    txn = pd.read_csv(txn_path, sep=",")
    if "agent_msisdn" not in txn.columns:
        sys.exit(f"ERROR: transaction file has no 'agent_msisdn' column. Found: {list(txn.columns)[:20]}")
    if "commission" not in txn.columns:
        sys.exit(f"ERROR: transaction file has no 'commission' column. Found: {list(txn.columns)[:20]}")

    parts_6m = ["cash_out_comm_6m", "cash_in_comm_6m", "voucher_comm_6m", "payment_comm_6m"]
    missing_parts = [c for c in parts_6m if c not in txn.columns]
    if missing_parts:
        sys.exit(f"ERROR: transaction file missing 6m commission component columns: {missing_parts}")

    txn["_msisdn_norm"] = _norm_msisdn(txn["agent_msisdn"])
    txn = txn[txn["_msisdn_norm"].notna()].drop_duplicates("_msisdn_norm", keep="first")

    txn["commission_raw"] = pd.to_numeric(txn["commission"], errors="coerce").fillna(0)
    txn["commission_6m_recomputed"] = sum(
        pd.to_numeric(txn[c], errors="coerce").fillna(0) for c in parts_6m
    )

    substantial_floor = 6_801.0 * 6
    in_bucket = (txn["commission_raw"] == 0) & (txn["commission_6m_recomputed"] >= substantial_floor)
    txn["in_mismatch_bucket"] = in_bucket

    n_bucket = int(in_bucket.sum())
    n_rest = int((~in_bucket).sum())
    print(f"Bucket (commission_raw==0, commission_6m_recomputed>={substantial_floor:,.0f}): {n_bucket:,} agents")
    print(f"Rest of population: {n_rest:,} agents\n")

    if n_bucket == 0:
        sys.exit("No agents in the bucket -- nothing to profile.")

    # -- Optional: bring in agent_category for context ----------------------
    out_path = Path(args.output_file) if args.output_file else None
    if out_path and out_path.exists():
        out = pd.read_csv(out_path)
        if "msisdn" in out.columns and "agent_category" in out.columns:
            out_small = out[["msisdn", "agent_category"]].copy()
            out_small["_msisdn_norm"] = _norm_msisdn(out_small["msisdn"])
            out_small = out_small.dropna(subset=["_msisdn_norm"]).drop_duplicates("_msisdn_norm")
            txn = txn.merge(out_small[["_msisdn_norm", "agent_category"]], on="_msisdn_norm", how="left")
            print("=== agent_category, bucket vs rest ===")
            _profile_categorical("agent_category", txn.loc[in_bucket, "agent_category"], txn.loc[~in_bucket, "agent_category"])

    # -- Build the column list to profile ------------------------------------
    explicit_defaults = ["agent_profile", "account_balance", "average_balance"]
    extra = [c.strip() for c in args.profile_cols.split(",") if c.strip()]
    auto_matched = [
        c for c in txn.columns
        if any(kw in c.lower() for kw in _AUTO_KEYWORDS)
        and c not in ("agent_msisdn", "_msisdn_norm", "commission", "commission_raw",
                      "commission_6m_recomputed", "in_mismatch_bucket")
    ]
    candidate_cols = list(dict.fromkeys(explicit_defaults + extra + auto_matched))
    present_cols = [c for c in candidate_cols if c in txn.columns]
    missing_cols = [c for c in candidate_cols if c not in txn.columns]
    if missing_cols:
        print(f"\nNOTE: these requested/auto-matched columns aren't in the file, skipping: {missing_cols}")

    print(f"\nProfiling columns: {present_cols}")

    bucket_df = txn.loc[in_bucket]
    rest_df = txn.loc[~in_bucket]

    for col in present_cols:
        series_all = txn[col]
        n_unique = series_all.nunique(dropna=True)
        is_numeric = pd.api.types.is_numeric_dtype(series_all) or (
            pd.to_numeric(series_all, errors="coerce").notna().mean() > 0.9
        )
        # Continuous financial quantities are always profiled numerically,
        # even if a small/unrepresentative row count makes nunique look low
        # (e.g. a quick smoke test) -- cardinality alone shouldn't demote a
        # balance/amount column to a categorical value_counts() dump.
        always_numeric = any(kw in col.lower() for kw in ("balance", "amount", "_ugx", "_value"))
        if is_numeric and (always_numeric or n_unique > args.categorical_max_unique):
            _profile_numeric(col, bucket_df[col], rest_df[col])
        else:
            _profile_categorical(col, bucket_df[col], rest_df[col])

    # -- Also always show the core commission fields for context ------------
    print("\n=== Core commission fields, bucket vs rest (for reference) ===")
    for col in ["commission_raw", "commission_6m_recomputed"] + parts_6m:
        _profile_numeric(col, bucket_df[col], rest_df[col])

    txn.to_csv(args.out, index=False)
    print(f"\nFull transaction file with 'in_mismatch_bucket' flag written to: {args.out}")


if __name__ == "__main__":
    main()

"""
Full, untruncated agent_profile breakdown for just the commission-mismatch
bucket (commission_raw == 0 AND commission_6m_recomputed >= 40,806) --
closes the gap left by profile_commission_mismatch_bucket.py's top-10
display cutoff, so every category (down to n=1) is visible and the counts
are confirmed to sum to the full bucket size.

Usage:
    python full_profile_breakdown_mismatch_bucket.py ^
        --transaction-file data\\mfs_daily_agent_mart_20260731.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _norm_msisdn(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transaction-file", default="data/mfs_daily_agent_mart_20260731.csv")
    ap.add_argument("--out", default="commission_mismatch_bucket_full_profile_breakdown.csv")
    args = ap.parse_args()

    txn_path = Path(args.transaction_file)
    if not txn_path.exists():
        sys.exit(f"ERROR: transaction file not found: {txn_path}")

    txn = pd.read_csv(txn_path, sep=",")
    for req in ("agent_msisdn", "commission", "agent_profile"):
        if req not in txn.columns:
            sys.exit(f"ERROR: transaction file has no '{req}' column. Found: {list(txn.columns)[:20]}")

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
    bucket = txn[(txn["commission_raw"] == 0) & (txn["commission_6m_recomputed"] >= substantial_floor)]

    print(f"Bucket size: {len(bucket):,}\n")

    counts = bucket["agent_profile"].value_counts(dropna=False)
    pct = (counts / len(bucket) * 100).round(2)
    breakdown = pd.DataFrame({"n": counts, "pct_of_bucket": pct})

    print("=== Full agent_profile breakdown, no truncation ===")
    print(breakdown.to_string())

    print(f"\nSum of n across all categories: {int(counts.sum()):,} (should equal bucket size {len(bucket):,})")
    print(f"Number of distinct agent_profile values represented: {breakdown.shape[0]}")

    breakdown.to_csv(args.out)
    print(f"\nFull breakdown written to: {args.out}")


if __name__ == "__main__":
    main()

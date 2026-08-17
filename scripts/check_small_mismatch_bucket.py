"""
Extends the same two checks already run against the 2,099-agent "large
mismatch" bucket (commission_raw==0 & commission_6m_recomputed>=40,806) to
the 8,749-agent "small mismatch" bucket -- commission_raw==0 but
0 < commission_6m_recomputed < 40,806 (some real component activity, just
below the "substantial" bar). This is the 80.7% of the 10,848-agent
"activity despite raw==0" population that hasn't been examined yet.

Reports, for this bucket specifically:
  1. Full, untruncated agent_profile breakdown (same format as
     full_profile_breakdown_mismatch_bucket.py) -- is this also
     concentrated in Super/Master Agent-type profiles, or a different mix?
  2. customers_served_3m distribution, bucket vs. rest of population (same
     as check_customers_served_90d.py) -- does this group also mostly serve
     real customers, or does it look more like genuine low-activity agents?

Usage:
    python check_small_mismatch_bucket.py ^
        --transaction-file data\\mfs_daily_agent_mart_20260731.csv ^
        --output-file output\\engine_test_output.csv
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
    ap.add_argument("--output-file", default="output/engine_test_output.csv")
    ap.add_argument("--out", default="small_mismatch_bucket.csv")
    args = ap.parse_args()

    txn_path = Path(args.transaction_file)
    if not txn_path.exists():
        sys.exit(f"ERROR: transaction file not found: {txn_path}")

    txn = pd.read_csv(txn_path, sep=",")
    for req in ("agent_msisdn", "commission"):
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

    # Same "substantial" floor used throughout this investigation:
    # population median 1-month commission (~6,801) x 6.
    substantial_floor = 6_801.0 * 6
    in_bucket = (
        (txn["commission_raw"] == 0)
        & (txn["commission_6m_recomputed"] > 0)
        & (txn["commission_6m_recomputed"] < substantial_floor)
    )
    n_bucket = int(in_bucket.sum())
    print(f"Small-mismatch bucket size: {n_bucket:,} (expected ~8,749)\n")

    if n_bucket == 0:
        sys.exit("No agents in this bucket -- nothing to profile.")

    # -- Check 1: full, untruncated agent_profile breakdown -----------------
    if "agent_profile" in txn.columns:
        bucket_df = txn.loc[in_bucket]
        counts = bucket_df["agent_profile"].value_counts(dropna=False)
        pct = (counts / n_bucket * 100).round(2)
        breakdown = pd.DataFrame({"n": counts, "pct_of_bucket": pct})
        print("=== Full agent_profile breakdown, no truncation ===")
        print(breakdown.to_string())
        print(f"\nSum of n across all categories: {int(counts.sum()):,} (should equal bucket size {n_bucket:,})")
        print(f"Number of distinct agent_profile values represented: {breakdown.shape[0]}")
    else:
        print("NOTE: 'agent_profile' column not found -- skipping profile breakdown.")

    # -- Check 2: customers_served_3m, bucket vs. rest -----------------------
    cust_parts = ["cash_out_cust_3m", "cash_in_cust_3m", "voucher_cust_3m", "payment_cust_3m"]
    present_cust_parts = [c for c in cust_parts if c in txn.columns]
    if present_cust_parts:
        txn["customers_served_3m"] = sum(
            pd.to_numeric(txn[c], errors="coerce").fillna(0) for c in present_cust_parts
        )
        print(f"\n=== customers_served_3m: small-mismatch bucket (n={n_bucket:,}) "
              f"vs. rest (n={int((~in_bucket).sum()):,}) ===")
        print(
            pd.DataFrame({
                "bucket": txn.loc[in_bucket, "customers_served_3m"].describe(
                    percentiles=[.1, .25, .5, .75, .9]
                ),
                "rest_of_population": txn.loc[~in_bucket, "customers_served_3m"].describe(
                    percentiles=[.1, .25, .5, .75, .9]
                ),
            }).to_string()
        )
        n_zero_bucket = int((txn.loc[in_bucket, "customers_served_3m"] == 0).sum())
        print(
            f"\nWithin the small-mismatch bucket, {n_zero_bucket:,}/{n_bucket:,} "
            f"({n_zero_bucket/n_bucket:.1%}) have customers_served_3m == 0."
        )

        if "agent_profile" in txn.columns:
            print("\n=== customers_served_3m by agent_profile, within the bucket only "
                  "(mean / median / % zero) ===")
            grp = txn.loc[in_bucket].groupby("agent_profile")["customers_served_3m"].agg(
                n="count", mean="mean", median="median",
                pct_zero=lambda s: (s == 0).mean() * 100,
            ).sort_values("median")
            print(grp.round(2).to_string())
    else:
        print("\nNOTE: no *_cust_3m columns found -- skipping customers_served_3m check.")

    # -- Optional: bring in agent_category for context -----------------------
    out_path = Path(args.output_file) if args.output_file else None
    if out_path and out_path.exists():
        out = pd.read_csv(out_path)
        if "msisdn" in out.columns and "agent_category" in out.columns:
            out_small = out[["msisdn", "agent_category"]].copy()
            out_small["_msisdn_norm"] = _norm_msisdn(out_small["msisdn"])
            out_small = out_small.dropna(subset=["_msisdn_norm"]).drop_duplicates("_msisdn_norm")
            txn = txn.merge(out_small[["_msisdn_norm", "agent_category"]], on="_msisdn_norm", how="left")
            print("\n=== agent_category, small-mismatch bucket vs rest ===")
            b_counts = txn.loc[in_bucket, "agent_category"].value_counts(normalize=True, dropna=False) * 100
            r_counts = txn.loc[~in_bucket, "agent_category"].value_counts(normalize=True, dropna=False) * 100
            print(pd.DataFrame({"bucket_pct": b_counts, "rest_pct": r_counts}).fillna(0.0).round(1).to_string())

    txn.loc[in_bucket].to_csv(args.out, index=False)
    print(f"\nFull small-mismatch bucket written to: {args.out}")


if __name__ == "__main__":
    main()

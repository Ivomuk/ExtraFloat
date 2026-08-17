"""
Business feedback: filtering by agent_profile string isn't reliable enough
(confirmed empirically -- 87% of the earlier "commission_raw==0" bucket was
still labeled "Agent Silver Class", the same as ordinary retail agents).
Suggested alternative: filter on customers served over a trailing window --
a genuine retail agent transacts with many distinct end customers; a
rebalancing-focused super/master agent mostly moves float between agents and
should show few or none.

This uses the *_cust_3m columns (cash_out_cust_3m, cash_in_cust_3m,
voucher_cust_3m, payment_cust_3m -- the customer-COUNT siblings of the
*_comm_3m commission columns, confirmed to exist under this naming
convention earlier this session) as the closest available proxy for
"customers served in 90 days" -- the file's windows are 1m/3m/6m, not a
literal 90d bucket, and 3 months is the closest match. NOTE: summing the
four service-type counts is a proxy for "customer-transaction instances",
not a strictly deduplicated unique-customer count across service types (an
agent doing both a cash-in and cash-out for the same customer would count
twice) -- flagged here rather than silently assumed away.

Reports:
  - percentile distribution of customers_served_3m across the full population
    (to help pick a cutoff)
  - the same distribution for the earlier commission-mismatch bucket vs. the
    rest of the population (to check whether this metric actually separates
    them, validating the business's hypothesis empirically)
  - cross-tab against agent_profile, so both signals (profile label and
    behavior) can be compared side by side

Usage:
    python check_customers_served_90d.py ^
        --transaction-file data\\mfs_daily_agent_mart_20260731.csv ^
        --output-file output\\engine_test_output.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_msisdn(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transaction-file", default="data/mfs_daily_agent_mart_20260731.csv")
    ap.add_argument("--output-file", default="output/engine_test_output.csv")
    ap.add_argument("--out", default="customers_served_90d.csv")
    args = ap.parse_args()

    txn_path = Path(args.transaction_file)
    if not txn_path.exists():
        sys.exit(f"ERROR: transaction file not found: {txn_path}")

    txn = pd.read_csv(txn_path, sep=",")
    if "agent_msisdn" not in txn.columns:
        sys.exit(f"ERROR: transaction file has no 'agent_msisdn' column. Found: {list(txn.columns)[:20]}")

    cust_parts = ["cash_out_cust_3m", "cash_in_cust_3m", "voucher_cust_3m", "payment_cust_3m"]
    present_cust_parts = [c for c in cust_parts if c in txn.columns]
    missing_cust_parts = [c for c in cust_parts if c not in txn.columns]
    if missing_cust_parts:
        print(f"NOTE: these expected customer-count columns are absent, skipped: {missing_cust_parts}")
    if not present_cust_parts:
        # Fall back: report every column containing 'cust' so the real names
        # can be confirmed and the script re-pointed if the naming differs.
        cust_like = [c for c in txn.columns if "cust" in c.lower()]
        sys.exit(
            "ERROR: none of the expected *_cust_3m columns were found.\n"
            f"Columns containing 'cust' in this file: {cust_like}\n"
            "Re-run with the correct names once confirmed."
        )

    txn["_msisdn_norm"] = _norm_msisdn(txn["agent_msisdn"])
    txn = txn[txn["_msisdn_norm"].notna()].drop_duplicates("_msisdn_norm", keep="first")

    txn["customers_served_3m"] = sum(
        pd.to_numeric(txn[c], errors="coerce").fillna(0) for c in present_cust_parts
    )

    print(f"Agents: {len(txn):,}")
    print(f"Customer-count columns used: {present_cust_parts}\n")

    print("=== customers_served_3m percentiles, full population ===")
    print(txn["customers_served_3m"].describe(
        percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]
    ).to_string())

    n_zero = int((txn["customers_served_3m"] == 0).sum())
    print(f"\nAgents with customers_served_3m == 0: {n_zero:,} ({n_zero/len(txn):.1%})")

    # -- Recompute the earlier commission-mismatch bucket, to check whether
    #    this metric actually separates it from the rest -----------------
    if "commission" in txn.columns:
        parts_6m = ["cash_out_comm_6m", "cash_in_comm_6m", "voucher_comm_6m", "payment_comm_6m"]
        if all(c in txn.columns for c in parts_6m):
            txn["commission_raw"] = pd.to_numeric(txn["commission"], errors="coerce").fillna(0)
            txn["commission_6m_recomputed"] = sum(
                pd.to_numeric(txn[c], errors="coerce").fillna(0) for c in parts_6m
            )
            substantial_floor = 6_801.0 * 6
            in_bucket = (txn["commission_raw"] == 0) & (txn["commission_6m_recomputed"] >= substantial_floor)

            print(f"\n=== customers_served_3m: earlier commission-mismatch bucket (n={int(in_bucket.sum()):,}) "
                  f"vs. rest (n={int((~in_bucket).sum()):,}) ===")
            print(
                pd.DataFrame({
                    "bucket": txn.loc[in_bucket, "customers_served_3m"].describe(),
                    "rest_of_population": txn.loc[~in_bucket, "customers_served_3m"].describe(),
                }).to_string()
            )
            n_zero_bucket = int((txn.loc[in_bucket, "customers_served_3m"] == 0).sum())
            print(
                f"\nWithin the bucket, {n_zero_bucket:,}/{int(in_bucket.sum()):,} "
                f"({n_zero_bucket/max(1,int(in_bucket.sum())):.1%}) have customers_served_3m == 0."
            )

    # -- Cross-reference against agent_profile, if present ------------------
    if "agent_profile" in txn.columns:
        print("\n=== customers_served_3m by agent_profile (mean / median / % zero) ===")
        grp = txn.groupby("agent_profile")["customers_served_3m"].agg(
            n="count", mean="mean", median="median",
            pct_zero=lambda s: (s == 0).mean() * 100,
        ).sort_values("median")
        print(grp.round(2).to_string())

    # -- Optional: bring in agent_category for context -----------------------
    out_path = Path(args.output_file) if args.output_file else None
    if out_path and out_path.exists():
        out = pd.read_csv(out_path)
        if "msisdn" in out.columns and "agent_category" in out.columns:
            out_small = out[["msisdn", "agent_category"]].copy()
            out_small["_msisdn_norm"] = _norm_msisdn(out_small["msisdn"])
            out_small = out_small.dropna(subset=["_msisdn_norm"]).drop_duplicates("_msisdn_norm")
            txn = txn.merge(out_small[["_msisdn_norm", "agent_category"]], on="_msisdn_norm", how="left")

    txn.to_csv(args.out, index=False)
    print(f"\nFull transaction file with 'customers_served_3m' column written to: {args.out}")


if __name__ == "__main__":
    main()

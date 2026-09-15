"""
Pulls the agents driving the Pearson/Spearman gap seen in
check_commission_raw_vs_recomputed.py's full-population run (Pearson=0.44
vs Spearman=0.99 -- rank order agrees almost perfectly, but a handful of
extreme-magnitude points are dragging the linear correlation down;
commission_6m_recomputed's max (~728M) is ~6.7x commission_raw's max
(~108M)).

That script's mismatch buckets only catch commission_raw==0-vs-substantial
cases and >5% disagreement among moderate values -- neither is built to
surface "both large, but wildly different from each other" cases, which is
exactly what a few extreme-tail outliers would look like. This pulls the
top N by absolute gap AND by ratio (a huge account with a modest % gap and
a modest account with a huge multiplier are both worth seeing -- one view
alone would miss one or the other), so it's possible to tell whether the
tail is a few legitimately huge agents or a real data issue.

Usage:
    python scripts\\top_commission_mismatch_outliers.py ^
        --transaction-file data\\mfs_daily_agent_mart_20260731.csv ^
        --top-n 30
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
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--out", default="commission_mismatch_outliers.csv")
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
    n_null_msisdn = int(txn["_msisdn_norm"].isna().sum())
    if n_null_msisdn:
        print(f"WARNING: {n_null_msisdn} rows have an unparseable/blank agent_msisdn -- dropped.")
    txn = txn[txn["_msisdn_norm"].notna()].copy()

    n_dupe = int(txn["_msisdn_norm"].duplicated().sum())
    if n_dupe:
        print(f"WARNING: {n_dupe} duplicate agent_msisdn rows -- keeping first occurrence.")
        txn = txn.drop_duplicates("_msisdn_norm", keep="first")

    txn["commission_raw"] = pd.to_numeric(txn["commission"], errors="coerce").fillna(0)
    txn["commission_6m_recomputed"] = sum(
        pd.to_numeric(txn[c], errors="coerce").fillna(0) for c in parts_6m
    )
    txn["abs_gap"] = (txn["commission_6m_recomputed"] - txn["commission_raw"]).abs()
    # Ratio of the larger to the smaller (>=1). A one-sided zero (e.g.
    # commission_raw==0 but commission_6m_recomputed>0 -- the "concerning
    # bucket" from check_commission_raw_vs_recomputed.py) is a genuine,
    # maximally-extreme mismatch and must sort to the TOP as ratio=inf, not
    # disappear as NaN. Only a true both-zero row (no mismatch at all) gets
    # ratio=1.
    hi = txn[["commission_raw", "commission_6m_recomputed"]].max(axis=1)
    lo = txn[["commission_raw", "commission_6m_recomputed"]].min(axis=1)
    both_zero = (txn["commission_raw"] == 0) & (txn["commission_6m_recomputed"] == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = hi / lo.replace(0, np.nan)
    txn["ratio"] = ratio.fillna(np.inf).mask(both_zero, 1.0)

    context_cols = [
        c for c in ("agent_msisdn", "agent_profile", "account_balance", "average_balance", "tbl_dt")
        if c in txn.columns
    ]
    display_cols = context_cols + [
        "commission_raw", "commission_6m_recomputed", "abs_gap", "ratio"
    ] + parts_6m

    print(f"Transaction file: {len(txn):,} unique agents after cleanup\n")

    print(f"=== Top {args.top_n} by ABSOLUTE gap |commission_6m_recomputed - commission_raw| ===")
    top_abs = txn.sort_values("abs_gap", ascending=False).head(args.top_n)
    print(top_abs[display_cols].to_string(index=False))

    print(f"\n=== Top {args.top_n} by RATIO (larger / smaller, excluding both-zero) ===")
    top_ratio = (
        txn[~((txn["commission_raw"] == 0) & (txn["commission_6m_recomputed"] == 0))]
        .sort_values("ratio", ascending=False)
        .head(args.top_n)
    )
    print(top_ratio[display_cols].to_string(index=False))

    combined = (
        pd.concat([top_abs.assign(matched_by="abs_gap"), top_ratio.assign(matched_by="ratio")])
        .drop_duplicates(subset="_msisdn_norm")
    )
    combined[["matched_by"] + display_cols].to_csv(args.out, index=False)
    print(f"\n{len(combined)} unique outlier agent(s) (top-abs ∪ top-ratio) written to: {args.out}")


if __name__ == "__main__":
    main()

"""
Exact count of agents with commission_raw == 0 across the full transaction
file -- i.e. the file's own pre-existing `commission` column reads 0,
regardless of what the recomputed commission_6m looks like.

Usage:
    python count_zero_commission_agents.py --transaction-file data\\mfs_daily_agent_mart_20260731.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _norm_msisdn(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transaction-file", default="data/mfs_daily_agent_mart_20260731.csv")
    args = ap.parse_args()

    txn_path = Path(args.transaction_file)
    if not txn_path.exists():
        sys.exit(f"ERROR: transaction file not found: {txn_path}")

    txn = pd.read_csv(txn_path, sep=",")
    for req in ("agent_msisdn", "commission"):
        if req not in txn.columns:
            sys.exit(f"ERROR: transaction file has no '{req}' column. Found: {list(txn.columns)[:20]}")

    txn["_msisdn_norm"] = _norm_msisdn(txn["agent_msisdn"])
    txn = txn[txn["_msisdn_norm"].notna()].drop_duplicates("_msisdn_norm", keep="first")

    txn["commission_raw"] = pd.to_numeric(txn["commission"], errors="coerce").fillna(0)

    n_total = len(txn)
    n_zero = int((txn["commission_raw"] == 0).sum())
    n_nonzero = n_total - n_zero

    print(f"Total agents: {n_total:,}")
    print(f"commission_raw == 0: {n_zero:,} ({n_zero/n_total:.1%})")
    print(f"commission_raw >  0: {n_nonzero:,} ({n_nonzero/n_total:.1%})")

    # Split the zero-commission group by whether the 6m component parts
    # also show zero activity, if available -- gives context on how much
    # of it is "genuinely inactive" vs. the anomalous mismatch bucket.
    parts_6m = ["cash_out_comm_6m", "cash_in_comm_6m", "voucher_comm_6m", "payment_comm_6m"]
    if all(c in txn.columns for c in parts_6m):
        txn["commission_6m_recomputed"] = sum(
            pd.to_numeric(txn[c], errors="coerce").fillna(0) for c in parts_6m
        )
        zero_mask = txn["commission_raw"] == 0
        n_zero_also_no_activity = int((zero_mask & (txn["commission_6m_recomputed"] == 0)).sum())
        n_zero_but_some_activity = n_zero - n_zero_also_no_activity
        print(f"\nOf the {n_zero:,} agents with commission_raw == 0:")
        print(f"  commission_6m_recomputed also == 0 (genuinely inactive): {n_zero_also_no_activity:,} "
              f"({n_zero_also_no_activity/n_zero:.1%})")
        print(f"  commission_6m_recomputed > 0 (some component activity despite raw==0): {n_zero_but_some_activity:,} "
              f"({n_zero_but_some_activity/n_zero:.1%})")


if __name__ == "__main__":
    main()

"""
Issue #2: pull sample rows for the "commission_raw == 0 but
commission_6m_recomputed is substantial" bucket, for manual root-cause
inspection (join gap, timing issue, excluded product code, etc.).

Uses the exact same bucket definition as
check_commission_raw_vs_recomputed.py's full-population run:
  commission_raw == 0  AND  commission_6m_recomputed >= 40,806
    (40,806 = population median 1-month commission (~6,801) x 6, used only
    as a rough "meaningfully active over 6 months" bar)

Usage:
    python sample_commission_mismatch_rows.py ^
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
    ap.add_argument("--out", default="commission_mismatch_sample.csv")
    ap.add_argument("--preview-n", type=int, default=25, help="Rows to print to console (largest recomputed first)")
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
    bucket = txn[(txn["commission_raw"] == 0) & (txn["commission_6m_recomputed"] >= substantial_floor)].copy()
    print(f"Agents in the commission_raw==0 / commission_6m_recomputed>={substantial_floor:,.0f} bucket: {len(bucket):,}\n")

    if len(bucket) == 0:
        print("No matching rows -- nothing to sample.")
        return

    out_path = Path(args.output_file) if args.output_file else None
    if out_path and out_path.exists():
        out = pd.read_csv(out_path)
        if "msisdn" in out.columns:
            keep = ["msisdn", "agent_category", "risk_tier", "assigned_limit", "is_thin_file"]
            keep = [c for c in keep if c in out.columns]
            out_small = out[keep].copy()
            out_small["_msisdn_norm"] = _norm_msisdn(out_small["msisdn"])
            out_small = out_small.dropna(subset=["_msisdn_norm"]).drop_duplicates("_msisdn_norm")
            bucket = bucket.merge(out_small.drop(columns=["msisdn"]), on="_msisdn_norm", how="left")

    bucket = bucket.sort_values("commission_6m_recomputed", ascending=False)

    # -- Console preview: the essential commission columns + any obviously
    #    relevant date/id columns, so a pattern (a specific date, product
    #    code, region, etc.) might jump out without opening the full CSV. --
    preview_priority_cols = [
        "agent_msisdn", "commission_raw", "commission_6m_recomputed",
        "cash_out_comm_6m", "cash_in_comm_6m", "voucher_comm_6m", "payment_comm_6m",
        "agent_category", "risk_tier", "assigned_limit",
    ]
    date_like_cols = [c for c in txn.columns if "date" in c.lower() or c.lower().endswith("_dt")]
    other_relevant = [c for c in txn.columns if any(k in c.lower() for k in ("region", "district", "territory", "product", "status", "ova"))]
    preview_cols = list(dict.fromkeys(
        [c for c in preview_priority_cols if c in bucket.columns]
        + [c for c in date_like_cols if c in bucket.columns]
        + [c for c in other_relevant if c in bucket.columns]
    ))

    print(f"=== Preview: top {min(args.preview_n, len(bucket))} rows by commission_6m_recomputed ===")
    print(bucket[preview_cols].head(args.preview_n).to_string(index=False))

    bucket.to_csv(args.out, index=False)
    print(f"\nFull bucket ({len(bucket):,} rows, all original columns retained) written to: {args.out}")
    print("Open that CSV for the complete row-level detail on each of these agents.")


if __name__ == "__main__":
    main()

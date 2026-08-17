"""
Full-population check: does the transaction file's pre-existing `commission`
column (the field agent_category/agent_tier_ceiling_multiplier is actually
computed from -- see prepare_transaction_capacity_features() in
extrafloat_limit_engine_features.py, which reads it directly and never
recomputes it) agree with the formula-reconstructed 6-month commission
(cash_out_comm_6m + cash_in_comm_6m + voucher_comm_6m + payment_comm_6m,
the formula verified against real sample rows earlier this session)?

This runs across the WHOLE transaction file, not just the 40-agent
fin-log/output intersection checked previously -- the goal is to find out
whether that mismatch was a narrow edge case or a systemic data problem
affecting agent_category for the full population (which would mean the
diamond/silver/gold tier distribution investigated earlier in this session
may itself be resting on the same broken input).

Usage:
    python check_commission_raw_vs_recomputed.py ^
        --transaction-file data\\mfs_daily_agent_mart_20260731.csv ^
        --output-file output\\engine_test_output.csv

--output-file is optional -- if given and it has an `agent_category` column
(requires --keep-intermediate on the run.bat run that produced it), the
mismatch breakdown is also shown by agent_category.
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
    ap.add_argument("--out", default="commission_raw_vs_recomputed_full.csv")
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
    n_total = len(txn)
    n_null_msisdn = int(txn["_msisdn_norm"].isna().sum())
    if n_null_msisdn:
        print(f"WARNING: {n_null_msisdn} rows have an unparseable/blank agent_msisdn -- dropped.")
    txn = txn[txn["_msisdn_norm"].notna()].copy()

    n_dupe = int(txn["_msisdn_norm"].duplicated().sum())
    if n_dupe:
        print(f"WARNING: {n_dupe} duplicate agent_msisdn rows in transaction file -- keeping first occurrence.")
        txn = txn.drop_duplicates("_msisdn_norm", keep="first")

    txn["commission_raw"] = pd.to_numeric(txn["commission"], errors="coerce").fillna(0)
    txn["commission_6m_recomputed"] = sum(
        pd.to_numeric(txn[c], errors="coerce").fillna(0) for c in parts_6m
    )

    print(f"Transaction file rows: {n_total:,}  ->  {len(txn):,} unique agents after cleanup\n")

    print("=== Distribution: commission_raw vs. commission_6m_recomputed, full population ===")
    print(txn[["commission_raw", "commission_6m_recomputed"]].describe().to_string())

    pearson_r = txn["commission_raw"].corr(txn["commission_6m_recomputed"], method="pearson")
    spearman_r = txn["commission_raw"].corr(txn["commission_6m_recomputed"], method="spearman")
    print(f"\nPearson correlation:  {pearson_r:.4f}")
    print(f"Spearman correlation: {spearman_r:.4f}  (rank-based -- less sensitive to outlier magnitude)")

    # -- Mismatch buckets -------------------------------------------------
    # "substantial" reference: population median 1-month commission from
    # earlier analysis (~6,801 UGX) x 6, as a rough "meaningfully active
    # over a 6-month window" bar -- not an actual tier threshold, just a
    # descriptive cutoff for bucketing.
    substantial_floor = 6_801.0 * 6

    raw = txn["commission_raw"]
    recomputed = txn["commission_6m_recomputed"]

    both_zero = (raw == 0) & (recomputed == 0)
    raw_zero_recomputed_substantial = (raw == 0) & (recomputed >= substantial_floor)
    raw_substantial_recomputed_zero = (raw >= substantial_floor) & (recomputed == 0)

    both_nonzero = (raw > 0) & (recomputed > 0)
    denom = np.maximum(raw, recomputed).replace(0, np.nan)
    rel_diff = (raw - recomputed).abs() / denom
    close_match = both_nonzero & (rel_diff <= 0.05)
    mismatched_nonzero = both_nonzero & (rel_diff > 0.05)

    other = ~(both_zero | raw_zero_recomputed_substantial | raw_substantial_recomputed_zero
              | close_match | mismatched_nonzero)

    n = len(txn)
    print(f"\n=== Mismatch buckets (n={n:,}) ===")
    print(f"Both zero:                                                    {int(both_zero.sum()):,} ({both_zero.mean():.1%})")
    print(f"commission_raw==0 but commission_6m_recomputed>={substantial_floor:,.0f} "
          f"(THE concerning bucket):                {int(raw_zero_recomputed_substantial.sum()):,} "
          f"({raw_zero_recomputed_substantial.mean():.1%})")
    print(f"commission_raw>={substantial_floor:,.0f} but commission_6m_recomputed==0 "
          f"(reverse case):                {int(raw_substantial_recomputed_zero.sum()):,} "
          f"({raw_substantial_recomputed_zero.mean():.1%})")
    print(f"Both nonzero, within 5% of each other (consistent):           {int(close_match.sum()):,} ({close_match.mean():.1%})")
    print(f"Both nonzero, disagree by >5%:                                {int(mismatched_nonzero.sum()):,} ({mismatched_nonzero.mean():.1%})")
    print(f"Other / doesn't fit above buckets:                            {int(other.sum()):,} ({other.mean():.1%})")

    txn["mismatch_bucket"] = np.select(
        [both_zero, raw_zero_recomputed_substantial, raw_substantial_recomputed_zero, close_match, mismatched_nonzero],
        ["both_zero", "raw_zero_recomputed_substantial", "raw_substantial_recomputed_zero", "close_match", "mismatched_nonzero"],
        default="other",
    )

    # -- Optional: enrich with agent_category from the scoring output -----
    out_path = Path(args.output_file) if args.output_file else None
    if out_path and out_path.exists():
        out = pd.read_csv(out_path)
        if "msisdn" in out.columns and "agent_category" in out.columns:
            out["_msisdn_norm"] = _norm_msisdn(out["msisdn"])
            out_small = out[["_msisdn_norm", "agent_category"]].dropna(subset=["_msisdn_norm"]).drop_duplicates("_msisdn_norm")
            txn = txn.merge(out_small, on="_msisdn_norm", how="left")

            print("\n=== Mismatch bucket distribution by agent_category ===")
            print(
                pd.crosstab(txn["agent_category"], txn["mismatch_bucket"], dropna=False)
                .to_string()
            )

            below = txn[txn["agent_category"] == "Below Threshold"]
            if len(below):
                n_misclassified = int((below["commission_6m_recomputed"] >= substantial_floor).sum())
                print(
                    f"\nOf {len(below):,} agents currently classified 'Below Threshold', "
                    f"{n_misclassified:,} ({n_misclassified/len(below):.1%}) have "
                    f"commission_6m_recomputed >= {substantial_floor:,.0f} -- i.e. look like real "
                    f"earners by the formula, despite commission_raw putting them at the bottom tier."
                )
        else:
            print(
                "\nNOTE: --output-file given but missing 'msisdn'/'agent_category' columns "
                "(need --keep-intermediate) -- skipping agent_category breakdown."
            )
    elif args.output_file:
        print(f"\nNOTE: --output-file '{args.output_file}' not found -- skipping agent_category breakdown.")

    txn.to_csv(args.out, index=False)
    print(f"\nFull per-agent comparison written to: {args.out}")


if __name__ == "__main__":
    main()

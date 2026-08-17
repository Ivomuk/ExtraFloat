"""
Issue #3: why are 292,813 transaction-file agents absent from the scoring
output (agent_category == NaN)?

Hypothesis, from reading build_extrafloat_limit_engine_features()
(extrafloat_limit_engine_features.py:858):

    merged = borrower_df.merge(transaction_df, on="msisdn", how="left")

The base population for the WHOLE engine output is borrower_df (from
--borrower-file), not the transaction file -- transaction data is
left-joined ONTO the borrower population. So any agent present in the
transaction mart but absent from the borrower file never enters `merged`
at all, no matter how much transaction/commission activity they have.

This script checks that directly: how many transaction-file agents are
missing from the borrower file, and (if --output-file is given) whether
that count/set matches the agents missing from the actual scoring output.

Usage:
    python check_transaction_vs_borrower_coverage.py ^
        --transaction-file data\\mfs_daily_agent_mart_20260731.csv ^
        --borrower-file data\\borrower_history.csv ^
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
    ap.add_argument("--borrower-file", default="data/borrower_history.csv")
    ap.add_argument("--output-file", default="output/engine_test_output.csv")
    ap.add_argument("--out", default="transaction_agents_missing_from_borrower_file.csv")
    args = ap.parse_args()

    txn_path = Path(args.transaction_file)
    bor_path = Path(args.borrower_file)
    if not txn_path.exists():
        sys.exit(f"ERROR: transaction file not found: {txn_path}")
    if not bor_path.exists():
        sys.exit(f"ERROR: borrower file not found: {bor_path}")

    txn = pd.read_csv(txn_path, sep=",")
    if "agent_msisdn" not in txn.columns:
        sys.exit(f"ERROR: transaction file has no 'agent_msisdn' column. Found: {list(txn.columns)[:20]}")
    txn["_msisdn_norm"] = _norm_msisdn(txn["agent_msisdn"])
    txn = txn[txn["_msisdn_norm"].notna()]
    txn_msisdns = set(txn["_msisdn_norm"].drop_duplicates())

    bor = pd.read_csv(bor_path)
    bor_msisdn_col = "msisdn" if "msisdn" in bor.columns else ("phonenumber" if "phonenumber" in bor.columns else None)
    if bor_msisdn_col is None:
        sys.exit(f"ERROR: borrower file has neither 'msisdn' nor 'phonenumber' column. Found: {list(bor.columns)[:20]}")
    bor["_msisdn_norm"] = _norm_msisdn(bor[bor_msisdn_col])
    bor = bor[bor["_msisdn_norm"].notna()]
    bor_msisdns = set(bor["_msisdn_norm"].drop_duplicates())

    print(f"Transaction file: {len(txn_msisdns):,} unique agents")
    print(f"Borrower file:    {len(bor_msisdns):,} unique agents")

    txn_not_in_bor = txn_msisdns - bor_msisdns
    bor_not_in_txn = bor_msisdns - txn_msisdns
    both = txn_msisdns & bor_msisdns

    print(f"\nIn transaction file but NOT in borrower file: {len(txn_not_in_bor):,} "
          f"({len(txn_not_in_bor)/len(txn_msisdns):.1%} of transaction-file agents)")
    print(f"In borrower file but NOT in transaction file: {len(bor_not_in_txn):,} "
          f"({len(bor_not_in_txn)/len(bor_msisdns):.1%} of borrower-file agents)")
    print(f"In both: {len(both):,}")

    # -- Cross-check against the actual scoring output, if given -----------
    out_path = Path(args.output_file) if args.output_file else None
    if out_path and out_path.exists():
        out = pd.read_csv(out_path)
        if "msisdn" in out.columns:
            out["_msisdn_norm"] = _norm_msisdn(out["msisdn"])
            out_msisdns = set(out["_msisdn_norm"].dropna().drop_duplicates())
            print(f"\nScoring output: {len(out_msisdns):,} unique agents")

            txn_not_in_output = txn_msisdns - out_msisdns
            print(f"In transaction file but NOT in scoring output: {len(txn_not_in_output):,}")

            # How well does "missing from borrower file" explain "missing from output"?
            overlap = txn_not_in_bor & txn_not_in_output
            print(
                f"\nOf the {len(txn_not_in_output):,} agents missing from the scoring output, "
                f"{len(overlap):,} ({len(overlap)/max(1,len(txn_not_in_output)):.1%}) are also "
                f"missing from the borrower file -- i.e. explained by the borrower_df.merge(...,"
                f" how='left') base-population hypothesis."
            )
            unexplained = txn_not_in_output - txn_not_in_bor
            print(
                f"{len(unexplained):,} agents are missing from the output despite being present "
                f"in the borrower file -- NOT explained by this hypothesis; would need a separate "
                f"look (e.g. dropped during merge dedup, or excluded further downstream)."
            )
        else:
            print("\nNOTE: --output-file given but has no 'msisdn' column -- skipping output cross-check.")
    elif args.output_file:
        print(f"\nNOTE: --output-file '{args.output_file}' not found -- skipping output cross-check.")

    # -- Export the missing-from-borrower list for review -------------------
    missing_df = txn[txn["_msisdn_norm"].isin(txn_not_in_bor)].copy()
    missing_df.to_csv(args.out, index=False)
    print(f"\nFull list of {len(missing_df):,} transaction-file agents missing from the borrower "
          f"file (all transaction-file columns retained) written to: {args.out}")


if __name__ == "__main__":
    main()

"""
Post-fix audit of the 5 closed-loan features that were confirmed universally
degenerate (100% NULL or constant) before the prior_loan_state_candidates
same-day-closure JOIN fix:

    avg_tenure_days_closed_loans
    closed_prior_loans_principal_repayment_ratio
    max_tenure_days_last_5_closed_loans
    last_closed_loan_tenure_days
    consecutive_on_time_loans

Reads only these columns (plus disbursement_fid for a row-count sanity
check) from a freshly re-exported training CSV, using usecols so this stays
cheap even on a multi-GB file, and reports non-null rate, nunique, and basic
stats for each.

Usage:
    python scripts\\check_closed_loan_features.py --file data\\state_data_....csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_COLUMNS = [
    "avg_tenure_days_closed_loans",
    "closed_prior_loans_principal_repayment_ratio",
    "max_tenure_days_last_5_closed_loans",
    "last_closed_loan_tenure_days",
    "consecutive_on_time_loans",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="Path to the exported training/scoring CSV")
    ap.add_argument("--id-col", default="disbursement_fid", help="Row-identity column, for a row-count sanity check")
    ap.add_argument("--columns", nargs="*", default=None, help="Override the default 5 audited columns")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    columns = args.columns if args.columns else DEFAULT_COLUMNS
    missing = [c for c in columns if c not in header]
    if missing:
        print(f"WARNING: not found in header, skipping: {missing}")
        columns = [c for c in columns if c in header]
    if not columns:
        sys.exit("ERROR: none of the requested columns are present in this file's header.")

    usecols = columns + ([args.id_col] if args.id_col in header else [])
    df = pd.read_csv(path, usecols=usecols)

    n = len(df)
    print(f"Rows read: {n:,}")
    if args.id_col in df.columns:
        print(f"Distinct {args.id_col}: {df[args.id_col].nunique():,}")
    print()

    print("=" * 90)
    print(f"{'column':<52} {'non_null':>10} {'non_null_%':>10} {'nunique':>9}")
    print("=" * 90)
    for col in columns:
        s = df[col]
        non_null = int(s.notna().sum())
        nunique = int(s.nunique(dropna=True))
        print(f"{col:<52} {non_null:>10,} {non_null / n * 100:>9.2f}% {nunique:>9,}")

    print()
    print("=" * 90)
    print("Detail per column")
    print("=" * 90)
    for col in columns:
        s = df[col]
        print(f"\n--- {col} ---")
        non_null_s = s.dropna()
        if non_null_s.empty:
            print("  all NULL")
            continue
        numeric_s = pd.to_numeric(non_null_s, errors="coerce")
        if numeric_s.notna().any():
            print(f"  min={numeric_s.min():.4f}  max={numeric_s.max():.4f}  "
                  f"mean={numeric_s.mean():.4f}  median={numeric_s.median():.4f}")
        print("  top values:")
        print(non_null_s.value_counts().head(8).to_string())


if __name__ == "__main__":
    main()

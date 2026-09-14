"""
Quick breakdown of the row-level CSV exported from find_hidden_old_open_loans.sql's
first query (after excluding ANOMALY_OPEN, i.e. status filter narrowed to
'OPEN', 'OPEN_PRE_WINDOW'). Reports:
  - status_as_of_snapshot value counts (OPEN vs OPEN_PRE_WINDOW split)
  - current_loan_start_date value counts (checking for the same kind of
    mass single-date clustering already confirmed for the ANOMALY_OPEN case)
  - loan_age_at_snapshot_days distribution and its most common values
    (a few repeated exact values dominating would also indicate clustering
    rather than organic, independently-varying agent behavior)

Usage:
    python scripts\\summarize_hidden_open_loan_export.py --file <exported_csv>
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")

    df = pd.read_csv(path)
    print(f"Rows: {len(df):,}")
    if "msisdn" in df.columns:
        print(f"Distinct msisdns: {df['msisdn'].nunique():,}\n")

    if "status_as_of_snapshot" in df.columns:
        print("=" * 78)
        print("status_as_of_snapshot value counts")
        print("=" * 78)
        vc = df["status_as_of_snapshot"].value_counts()
        print(pd.DataFrame({"n": vc, "pct": (vc / len(df) * 100).round(1)}).to_string())
        print()
    else:
        print("NOTE: 'status_as_of_snapshot' column not found.")

    if "current_loan_start_date" in df.columns:
        print("=" * 78)
        print(f"current_loan_start_date -- top {args.top_n} most common dates")
        print("=" * 78)
        vc = df["current_loan_start_date"].value_counts().head(args.top_n)
        print(pd.DataFrame({"n": vc, "pct": (vc / len(df) * 100).round(1)}).to_string())
        print(f"\nDistinct start dates: {df['current_loan_start_date'].nunique():,}\n")
    else:
        print("NOTE: 'current_loan_start_date' column not found.")

    if "loan_age_at_snapshot_days" in df.columns:
        print("=" * 78)
        print("loan_age_at_snapshot_days -- distribution and most common exact values")
        print("=" * 78)
        ages = pd.to_numeric(df["loan_age_at_snapshot_days"], errors="coerce")
        print(f"min={ages.min():.0f}  median={ages.median():.0f}  mean={ages.mean():.1f}  "
              f"p90={ages.quantile(0.9):.0f}  max={ages.max():.0f}\n")
        vc = ages.value_counts().head(args.top_n)
        print(pd.DataFrame({"n": vc, "pct": (vc / len(df) * 100).round(1)}).to_string())
        print(f"\nDistinct age values: {ages.nunique():,}")
    else:
        print("NOTE: 'loan_age_at_snapshot_days' column not found.")


if __name__ == "__main__":
    main()

"""
Pulls out the specific agents behind the two severe-breach figures quoted
from check_monthly_summary_vs_assigned_limit.py's output:
  - breach by 25% or more (pct_of_limit >= 1.25)
  - breach by more than double (pct_of_limit >= 2.00)

For each severity group, prints how it breaks down by risk_tier and
agent_category (to check whether severe breaches are still concentrated in
the riskiest agents, same as the overall violation rate was), and writes
the actual agent list to CSV, worst-first, so specific msisdns can be
inspected.

Usage:
    python scripts\\summarize_severe_limit_breaches.py ^
        --file monthly_summary_vs_assigned_limit.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _breakdown(df: pd.DataFrame, col: str, total_violations: int) -> None:
    if col not in df.columns:
        return
    g = df.groupby(col, observed=True).size().sort_values(ascending=False)
    print(f"\nBy {col}:")
    for val, n in g.items():
        print(f"  {val}: {n:,} ({n / len(df):.1%} of this group, {n / total_violations:.1%} of all violations)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="monthly_summary_vs_assigned_limit.csv")
    ap.add_argument("--out-25pct", default="severe_breaches_25pct_plus.csv")
    ap.add_argument("--out-2x", default="severe_breaches_2x_plus.csv")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")

    df = pd.read_csv(path)
    if "pct_of_limit" not in df.columns:
        sys.exit(f"ERROR: {path} has no 'pct_of_limit' column -- re-run "
                  f"check_monthly_summary_vs_assigned_limit.py first.")

    violations = df[df["exceeds_limit"] == True].copy() if "exceeds_limit" in df.columns \
        else df[df["pct_of_limit"] > 1.0].copy()
    n_violations = len(violations)
    print(f"Total violations (pct_of_limit > 100%): {n_violations:,}\n")

    breach_25 = violations[violations["pct_of_limit"] >= 1.25].copy()
    breach_2x = violations[violations["pct_of_limit"] >= 2.00].copy()

    print("=" * 78)
    print(f"Breach by 25%+ (pct_of_limit >= 125%): {len(breach_25):,} "
          f"({len(breach_25) / n_violations:.1%} of all violations)")
    print("=" * 78)
    cols = ["msisdn", "month", "disbursed_amount", "assigned_limit", "pct_of_limit",
            "excess_amount", "risk_tier", "agent_category", "cal_pd"]
    cols = [c for c in cols if c in breach_25.columns]
    print(breach_25[cols].sort_values("pct_of_limit", ascending=False).head(15).to_string(index=False))
    _breakdown(breach_25, "risk_tier", n_violations)
    _breakdown(breach_25, "agent_category", n_violations)
    breach_25[cols].sort_values("pct_of_limit", ascending=False).to_csv(args.out_25pct, index=False)
    print(f"\nFull list written: {args.out_25pct}  ({len(breach_25):,} rows)")

    print(f"\n{'=' * 78}")
    print(f"Breach by more than double (pct_of_limit >= 200%): {len(breach_2x):,} "
          f"({len(breach_2x) / n_violations:.1%} of all violations)")
    print("=" * 78)
    print(breach_2x[cols].sort_values("pct_of_limit", ascending=False).head(15).to_string(index=False))
    _breakdown(breach_2x, "risk_tier", n_violations)
    _breakdown(breach_2x, "agent_category", n_violations)
    breach_2x[cols].sort_values("pct_of_limit", ascending=False).to_csv(args.out_2x, index=False)
    print(f"\nFull list written: {args.out_2x}  ({len(breach_2x):,} rows)")


if __name__ == "__main__":
    main()

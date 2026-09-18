"""
Breaks down check_disbursements_vs_assigned_limit.py's per-transaction
export by risk_tier and cal_pd decile: does the 30.4% limit-violation rate
concentrate among the riskiest agents (the worst version of this problem --
the control failing exactly where it matters most), or is it spread evenly
across risk levels (suggesting the limit isn't used operationally at all,
regardless of risk)?

Usage:
    python scripts\\summarize_limit_violations_by_risk.py ^
        --file disbursements_vs_limit_per_transaction.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col, observed=True)
    return pd.DataFrame({
        "n_transactions": g.size(),
        "violation_rate": g["exceeds_limit"].mean(),
        "n_violations": g["exceeds_limit"].sum(),
        "median_pct_of_limit": g["pct_of_limit"].median(),
        "median_excess_when_violating": df[df["exceeds_limit"]].groupby(group_col, observed=True)["excess_amount"].median(),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="disbursements_vs_limit_per_transaction.csv")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")

    df = pd.read_csv(path)
    df["exceeds_limit"] = df["exceeds_limit"].astype(bool)
    print(f"Rows: {len(df):,}\n")

    if "risk_tier" in df.columns:
        print("=" * 78)
        print("Violation rate by risk_tier")
        print("=" * 78)
        tbl = _summarize(df, "risk_tier").sort_index()
        print(tbl.round(4).to_string())
        print()
    else:
        print("NOTE: 'risk_tier' not in this file -- was the engine output missing that column?\n")

    if "pd_decile" in df.columns:
        print("=" * 78)
        print("Violation rate by pd_decile (1 = safest, 10 = riskiest)")
        print("=" * 78)
        tbl = _summarize(df, "pd_decile").sort_index()
        print(tbl.round(4).to_string())
        print()
    else:
        print("NOTE: 'pd_decile' not in this file.\n")

    print(
        "Interpretation: if violation_rate climbs from low tiers/deciles to high ones, "
        "the control is failing worst exactly where the risk is highest -- the most "
        "concerning shape. If violation_rate looks similar across all tiers/deciles, "
        "the limit isn't being respected operationally at all, independent of the "
        "agent's actual risk level -- still a serious finding, but a different one."
    )


if __name__ == "__main__":
    main()

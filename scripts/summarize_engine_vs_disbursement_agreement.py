"""
Fuller picture of where the credit-limit engine's assigned_limit agrees or
disagrees with what was actually disbursed (Aug 1-14), using check_
disbursements_vs_assigned_limit.py's per-transaction export. Rather than
just a binary over/under-limit split, buckets pct_of_limit (disbursed /
assigned_limit) across its whole range:

  - A cluster near 100% (say 90-110%) is the strongest evidence of genuine
    AGREEMENT -- disbursements actively sized against the engine's limit.
  - A cluster well BELOW 100% (e.g. <75%) is also agreement, just
    conservative -- the engine's cap wasn't binding, the agent took less
    than allowed.
  - Mass ABOVE 110%, especially far above (150%+), is DISAGREEMENT -- the
    limit was exceeded, sometimes substantially.
  - A broad, flat spread across all buckets with no concentration near
    100% would suggest the limit isn't being actively consulted at all --
    disbursement amounts are unrelated to it, agreement or violation
    happening by coincidence rather than by the limit acting as a real cap.

Usage:
    python scripts\\summarize_engine_vs_disbursement_agreement.py ^
        --file disbursements_vs_limit_per_transaction.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

BUCKET_EDGES = [-0.01, 0.25, 0.50, 0.75, 0.90, 1.10, 1.25, 1.50, 2.00, 100]
BUCKET_LABELS = [
    "0-25% (well under)", "25-50% (under)", "50-75% (under)",
    "75-90% (near, under)", "90-110% (AGREEMENT band)", "110-125% (near, over)",
    "125-150% (over)", "150-200% (well over)", "200%+ (far over)",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="disbursements_vs_limit_per_transaction.csv")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")

    df = pd.read_csv(path)
    df["exceeds_limit"] = df["exceeds_limit"].astype(bool)
    n = len(df)
    print(f"Transactions: {n:,}\n")

    buckets = pd.cut(df["pct_of_limit"], bins=BUCKET_EDGES, labels=BUCKET_LABELS)
    counts = buckets.value_counts().reindex(BUCKET_LABELS)
    pct = (counts / n * 100).round(1)

    print("=" * 78)
    print("Disbursed amount as % of assigned_limit -- full distribution")
    print("=" * 78)
    print(pd.DataFrame({"n": counts, "pct": pct}).to_string())

    near_agreement = int((buckets == "90-110% (AGREEMENT band)").sum())
    well_under = int(df["pct_of_limit"] <= 0.75).sum()
    well_over = int(df["pct_of_limit"] >= 1.50).sum()

    print(f"\n{'=' * 78}")
    print("Summary")
    print("=" * 78)
    print(f"Within the 90-110% 'actively sized to the limit' band: {near_agreement:,} ({near_agreement/n:.1%})")
    print(f"Well under limit (<=75%, conservative agreement): {well_under:,} ({well_under/n:.1%})")
    print(f"Any overage at all (>100%): {int(df['exceeds_limit'].sum()):,} ({df['exceeds_limit'].mean():.1%})")
    print(f"Well over limit (>=150% of limit): {well_over:,} ({well_over/n:.1%})")

    print(
        "\nInterpretation: a large 90-110% band means the limit IS actively used to size "
        "disbursements for a real share of transactions -- genuine agreement, not "
        "coincidence. A large, spread-out mass across many buckets with no concentration "
        "near 100% suggests disbursement amounts are set independently of the engine's "
        "limit -- the limit isn't functioning as an active cap on typical lending "
        "decisions, only incidentally correlated with them."
    )


if __name__ == "__main__":
    main()

"""
Follow-up to check_post_snapshot_disbursement.py, which ruled out "the bad
loan was disbursed after the snapshot" as the explanation for why the
closed-loan-tenure feature fix didn't move AUC on the "closed-loan
defaulter" subgroup (confirmed Defaulter blacklist agents whose loan
already shows resolved, zero-balance, at the July-31 snapshot).

Tests a different hypothesis: the August blacklist's "Defaulter not paid
back in the last 30 days" tag is stale for a meaningful share of this
subgroup -- i.e. the loan genuinely closed a long time before the blacklist
date, making "still delinquent" implausible. We already know from earlier
this session that 98.3% of the blacklist's overall coverage gap is
"confirmed benign" (agents who never borrowed), so blacklist staleness/noise
at scale is an established pattern here, not a new assumption.

Reports, for the same WITHOUT-unresolved Defaulter population used by
check_post_snapshot_disbursement.py:
  - the distribution of days_since_last_repayment (days between the last
    actual repayment event and the July-31 snapshot) -- passes through
    unrenamed from data/loan_history_snapshot_query.txt per
    pd_model/preprocessing/loan_history_features.py's column-mapping
    comment, so it should already be a column in --matched-file.
  - if scoring_state_date is present, days between that state date and
    --blacklist-date -- a direct "how long before being blacklisted did
    this loan's tracked state last update" measure.

A distribution skewed toward large day counts (weeks/months) supports the
stale-label hypothesis; a distribution clustered near 0 would argue against
it (the loan closed right before the blacklist, so "still delinquent" isn't
obviously implausible).

Usage:
    python scripts\\check_stale_defaulter_label.py --matched-file wl_bl_eval_matched_agents.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULTER_REASON = "Defaulter not paid back in the last 30 days"
BUCKET_EDGES = [-1, 0, 7, 30, 90, 180, 10_000]
BUCKET_LABELS = ["0 days", "1-7 days", "8-30 days", "31-90 days", "91-180 days", "180+ days"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched-file", default="wl_bl_eval_matched_agents.csv")
    ap.add_argument("--blacklist-date", default="2026-08-04")
    args = ap.parse_args()

    matched_path = Path(args.matched_file)
    if not matched_path.exists():
        sys.exit(f"ERROR: matched file not found: {matched_path}")

    df = pd.read_csv(matched_path)
    required = ["is_blacklisted", "reason", "has_unresolved_loan_at_snapshot"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        sys.exit(f"ERROR: {matched_path} is missing required column(s): {missing_req}")

    target = df[
        (df["is_blacklisted"] == 1)
        & (df["reason"] == DEFAULTER_REASON)
        & (df["has_unresolved_loan_at_snapshot"] == 0)
    ].copy()
    if target.empty:
        sys.exit(f"ERROR: no rows matched the WITHOUT-unresolved Defaulter population in {matched_path}")
    print(f"WITHOUT-unresolved Defaulter population: {len(target):,} rows\n")

    if "days_since_last_repayment" not in target.columns:
        print(f"NOTE: 'days_since_last_repayment' not found in {matched_path}.")
        print(f"Columns present: {list(target.columns)}")
    else:
        dslr = pd.to_numeric(target["days_since_last_repayment"], errors="coerce")
        n_non_null = dslr.notna().sum()
        print("=" * 78)
        print(f"days_since_last_repayment (days between last repayment and the "
              f"July-31 snapshot) -- n_non_null={n_non_null:,} / {len(target):,}")
        print("=" * 78)
        print(f"min={dslr.min():.1f}  p25={dslr.quantile(0.25):.1f}  median={dslr.median():.1f}  "
              f"p75={dslr.quantile(0.75):.1f}  p90={dslr.quantile(0.90):.1f}  max={dslr.max():.1f}\n")

        buckets = pd.cut(dslr, bins=BUCKET_EDGES, labels=BUCKET_LABELS)
        bucket_counts = buckets.value_counts().reindex(BUCKET_LABELS)
        bucket_pct = (bucket_counts / n_non_null * 100).round(1)
        bucket_tbl = pd.DataFrame({"n": bucket_counts, "pct": bucket_pct})
        print(bucket_tbl.to_string())
        print()

    if "scoring_state_date" in target.columns:
        state_date = pd.to_datetime(target["scoring_state_date"], errors="coerce")
        blacklist_date = pd.to_datetime(args.blacklist_date)
        days_before_blacklist = (blacklist_date - state_date).dt.days
        n_non_null = days_before_blacklist.notna().sum()
        print("=" * 78)
        print(f"Days between scoring_state_date (last tracked loan-state update) "
              f"and --blacklist-date ({args.blacklist_date}) -- n_non_null={n_non_null:,}")
        print("=" * 78)
        print(f"min={days_before_blacklist.min():.1f}  median={days_before_blacklist.median():.1f}  "
              f"p90={days_before_blacklist.quantile(0.90):.1f}  max={days_before_blacklist.max():.1f}")
    else:
        print(f"NOTE: 'scoring_state_date' not found in {matched_path} -- skipping that section.")

    print(
        "\nInterpretation: if most of this population clusters in the larger buckets "
        "(31+ days, especially 90+), that supports the blacklist's 'Defaulter' tag being "
        "stale for this subgroup -- the loan closed well before the blacklist date, making "
        "'still delinquent 30 days later' implausible on its face, consistent with the "
        "blacklist-noise pattern already confirmed elsewhere this session (98.3% of the "
        "overall coverage gap was 'confirmed benign'). If instead most cluster near 0-7 "
        "days, the label is harder to dismiss as stale and the AUC=0.47 result may instead "
        "reflect a genuine limit of what this loan-history data can predict for this "
        "specific pattern."
    )


if __name__ == "__main__":
    main()

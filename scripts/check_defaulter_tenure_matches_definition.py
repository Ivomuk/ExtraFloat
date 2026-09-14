"""
Follow-up to check_stale_defaulter_label.py, after the product owner
clarified the real definition: "Defaulter not paid back in the last 30
days" means a loan reached 30+1 days unpaid since ITS OWN disbursement date
-- a permanent flag, applied once and never cleared, so a wide spread of
days_since_last_repayment (30, 90, 109, 200+ days before the blacklist
snapshot) is expected and does not by itself indicate a stale/wrong label.

This script tests the definition directly against observable data: for the
same WITHOUT-unresolved Defaulter population (confirmed Defaulter agents
whose loan already shows resolved at the scoring snapshot), does their own
last CLOSED loan's tenure (disbursement date -> closure/repayment date,
now computed correctly per this session's tenure_days fix) actually show a
30+ day gap? If most do, the "Defaulter" tag is directly validated against
their visible loan history. If most show a SHORT tenure (quick repayment),
that loan can't be the one that triggered the flag -- either a different,
not-visible-here loan/product caused it, or there's a genuine data-linkage
gap worth investigating further.

Also reports observed_prior_loan_count / closed_loan_count, since an agent
with several closed loans could have one old bad (30+ day) loan buried in
an otherwise-clean, faster-repaying history -- last_closed_loan_tenure_days
(most recent) could look short/clean even while avg_tenure_days_closed_loans
(averaged across all their closed loans) reflects the old bad one.

Usage:
    python scripts\\check_defaulter_tenure_matches_definition.py ^
        --matched-file wl_bl_eval_matched_agents.csv ^
        --loan-history-file data\\loan_history_snapshot_20260817_retail_filtered.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULTER_REASON = "Defaulter not paid back in the last 30 days"
TENURE_COLS = [
    "last_closed_loan_tenure_days",
    "avg_tenure_days_closed_loans",
    "max_tenure_days_last_5_closed_loans",
]
COUNT_COLS = ["observed_loan_count", "closed_loan_count"]
BUCKET_EDGES = [-1, 0, 3, 7, 14, 30, 60, 90, 10_000]
BUCKET_LABELS = ["0 days", "1-3 days", "4-7 days", "8-14 days", "15-30 days",
                 "31-60 days", "61-90 days", "90+ days"]


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched-file", default="wl_bl_eval_matched_agents.csv")
    ap.add_argument("--loan-history-file", required=True)
    args = ap.parse_args()

    matched_path = Path(args.matched_file)
    if not matched_path.exists():
        sys.exit(f"ERROR: matched file not found: {matched_path}")
    lh_path = Path(args.loan_history_file)
    if not lh_path.exists():
        sys.exit(f"ERROR: loan-history file not found: {lh_path}")

    df = pd.read_csv(matched_path)
    required = ["is_blacklisted", "reason", "has_unresolved_loan_at_snapshot"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        sys.exit(f"ERROR: {matched_path} is missing required column(s): {missing_req}")
    msisdn_col = "msisdn" if "msisdn" in df.columns else ("agent_msisdn" if "agent_msisdn" in df.columns else None)
    if msisdn_col is None:
        sys.exit(f"ERROR: {matched_path} has neither 'msisdn' nor 'agent_msisdn' column")

    target = df[
        (df["is_blacklisted"] == 1)
        & (df["reason"] == DEFAULTER_REASON)
        & (df["has_unresolved_loan_at_snapshot"] == 0)
    ].copy()
    if target.empty:
        sys.exit(f"ERROR: no rows matched the WITHOUT-unresolved Defaulter population in {matched_path}")
    target["_key"] = _normalize_msisdn(target[msisdn_col])
    print(f"WITHOUT-unresolved Defaulter population: {target['_key'].nunique():,} unique msisdns\n")

    lh_df = pd.read_csv(lh_path)
    lh_msisdn_col = "msisdn" if "msisdn" in lh_df.columns else ("phonenumber" if "phonenumber" in lh_df.columns else None)
    if lh_msisdn_col is None:
        sys.exit(f"ERROR: {lh_path} has neither 'msisdn' nor 'phonenumber' column. "
                  f"Columns present: {list(lh_df.columns)}")
    lh_df["_key"] = _normalize_msisdn(lh_df[lh_msisdn_col])

    matched_lh = target[["_key"]].drop_duplicates().merge(lh_df, on="_key", how="left")
    n_found = matched_lh[lh_msisdn_col].notna().sum()
    print(f"Found in {lh_path.name}: {n_found:,} / {matched_lh['_key'].nunique():,}\n")

    present_count_cols = [c for c in COUNT_COLS if c in matched_lh.columns]
    if present_count_cols:
        print("=" * 78)
        print("Prior-loan counts (is this a one-loan agent or several?)")
        print("=" * 78)
        for col in present_count_cols:
            vals = pd.to_numeric(matched_lh[col], errors="coerce")
            print(f"{col}: median={vals.median():.0f}  mean={vals.mean():.2f}  "
                  f"pct with >1 = {(vals > 1).mean():.1%}")
        print()

    for col in TENURE_COLS:
        if col not in matched_lh.columns:
            print(f"NOTE: '{col}' not found in {lh_path}.")
            continue
        vals = pd.to_numeric(matched_lh[col], errors="coerce")
        n_non_null = vals.notna().sum()
        n_ge_30 = int((vals >= 30).sum())
        print("=" * 78)
        print(f"{col} -- n_non_null={n_non_null:,} / {len(matched_lh):,}")
        print("=" * 78)
        if n_non_null == 0:
            print("(all NULL)\n")
            continue
        print(f"min={vals.min():.1f}  median={vals.median():.1f}  mean={vals.mean():.2f}  "
              f"p90={vals.quantile(0.90):.1f}  max={vals.max():.1f}")
        print(f">= 30 days: {n_ge_30:,} / {n_non_null:,} ({n_ge_30 / n_non_null:.1%})\n")

        buckets = pd.cut(vals, bins=BUCKET_EDGES, labels=BUCKET_LABELS)
        bucket_counts = buckets.value_counts().reindex(BUCKET_LABELS)
        bucket_pct = (bucket_counts / n_non_null * 100).round(1)
        print(pd.DataFrame({"n": bucket_counts, "pct": bucket_pct}).to_string())
        print()

    print(
        "Interpretation: if last_closed_loan_tenure_days is >= 30 for most of this "
        "population, their MOST RECENT closed loan directly matches the '30+1 days unpaid' "
        "Defaulter definition -- the tag is accurate and current. If last_closed_loan_tenure_days "
        "is mostly short (quick repayment) while avg_tenure_days_closed_loans is pulled up by a "
        "higher max, and closed_loan_count > 1 is common, that supports 'one old bad loan buried "
        "in an otherwise-clean, multi-loan history' -- a permanently-flagged agent whose CURRENT "
        "behavior has genuinely improved, which the model would correctly score as low-risk even "
        "though the old flag persists."
    )


if __name__ == "__main__":
    main()

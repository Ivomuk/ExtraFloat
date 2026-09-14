"""
Checks how much two independently-derived "disbursed after the blacklist
date" groups overlap by msisdn:

  Group A: last_disbursement_date (from --loan-summary-file, e.g.
  data/loan_summary.csv) > --blacklist-date. This is the 197-agent group
  from check_disbursement_vs_blacklist_window.py.

  Group B: within the "old flag, clean now" subgroup (last_closed_loan_
  tenure_days < 30), a disbursement date derived from --loan-history-file
  as last_closed_loan_closure_date - last_closed_loan_tenure_days >
  --blacklist-date. This is the ~217-agent August-bucket group from
  check_old_flag_group_by_period.py, now cut at the exact blacklist date
  rather than a full calendar month.

Both are meant to capture the same real-world fact (a new loan disbursed
after the agent was already blacklisted) from two independently-built data
sources -- this reports whether they substantially agree (mostly the same
msisdns) or diverge (each catching cases the other misses), which matters
for how confidently the "new lending after blacklist" finding can be
reported.

Usage:
    python scripts\\check_post_blacklist_group_overlap.py ^
        --matched-file wl_bl_eval_matched_agents.csv ^
        --loan-summary-file data\\loan_summary.csv ^
        --loan-history-file data\\loan_history_snapshot_20260817_retail_filtered.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULTER_REASON = "Defaulter not paid back in the last 30 days"

CANDIDATE_DATE_COLS = [
    "last_disbursement_date",
    "Last_disbursement_date",
    "most_recent_disbursement_date",
    "latest_disbursement_date",
]


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def _parse_date_flexible(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce")
    if parsed.notna().any() and parsed.dropna().dt.year.max() <= 1971:
        parsed = pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return parsed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched-file", default="wl_bl_eval_matched_agents.csv")
    ap.add_argument("--loan-summary-file", required=True)
    ap.add_argument("--loan-history-file", required=True)
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--blacklist-date", default="2026-08-04")
    ap.add_argument("--tenure-threshold", type=float, default=30.0)
    args = ap.parse_args()

    matched_path = Path(args.matched_file)
    if not matched_path.exists():
        sys.exit(f"ERROR: matched file not found: {matched_path}")
    ls_path = Path(args.loan_summary_file)
    if not ls_path.exists():
        sys.exit(f"ERROR: loan-summary file not found: {ls_path}")
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
    target_keys = set(target["_key"].dropna())
    print(f"WITHOUT-unresolved Defaulter population: {len(target_keys):,}\n")

    blacklist_date = pd.to_datetime(args.blacklist_date)

    # Group A: loan_summary.csv
    ls_df = pd.read_csv(ls_path)
    ls_msisdn_col = "msisdn" if "msisdn" in ls_df.columns else ("phonenumber" if "phonenumber" in ls_df.columns else None)
    if ls_msisdn_col is None:
        sys.exit(f"ERROR: {ls_path} has neither 'msisdn' nor 'phonenumber' column. "
                  f"Columns present: {list(ls_df.columns)}")
    date_col = args.date_col or next((c for c in CANDIDATE_DATE_COLS if c in ls_df.columns), None)
    if date_col is None:
        sys.exit(f"ERROR: could not auto-detect a disbursement-date column in {ls_path}. "
                  f"Columns present: {list(ls_df.columns)}")
    ls_df["_key"] = _normalize_msisdn(ls_df[ls_msisdn_col])
    ls_df["_date"] = _parse_date_flexible(ls_df[date_col])
    group_a = set(
        ls_df.loc[(ls_df["_key"].isin(target_keys)) & (ls_df["_date"] > blacklist_date), "_key"].dropna()
    )

    # Group B: loan-history-snapshot file, old-flag-clean-now subset,
    # derived disbursement date > blacklist_date
    lh_df = pd.read_csv(lh_path)
    lh_msisdn_col = "msisdn" if "msisdn" in lh_df.columns else ("phonenumber" if "phonenumber" in lh_df.columns else None)
    if lh_msisdn_col is None:
        sys.exit(f"ERROR: {lh_path} has neither 'msisdn' nor 'phonenumber' column. "
                  f"Columns present: {list(lh_df.columns)}")
    needed = ["last_closed_loan_tenure_days", "last_closed_loan_closure_date"]
    missing_lh = [c for c in needed if c not in lh_df.columns]
    if missing_lh:
        sys.exit(f"ERROR: {lh_path} is missing {missing_lh}. Columns present: {list(lh_df.columns)}")
    lh_df["_key"] = _normalize_msisdn(lh_df[lh_msisdn_col])
    lh_df["_tenure"] = pd.to_numeric(lh_df["last_closed_loan_tenure_days"], errors="coerce")
    lh_df["_closure_date"] = _parse_date_flexible(lh_df["last_closed_loan_closure_date"])
    lh_df["_disbursement_date"] = lh_df["_closure_date"] - pd.to_timedelta(lh_df["_tenure"], unit="D")
    group_b = set(
        lh_df.loc[
            (lh_df["_key"].isin(target_keys))
            & (lh_df["_tenure"] < args.tenure_threshold)
            & (lh_df["_disbursement_date"] > blacklist_date),
            "_key",
        ].dropna()
    )

    overlap = group_a & group_b
    only_a = group_a - group_b
    only_b = group_b - group_a

    print("=" * 78)
    print(f"Group A (loan_summary.csv last_disbursement_date > {args.blacklist_date}): {len(group_a):,}")
    print(f"Group B (loan-history-file derived disbursement date > {args.blacklist_date}, "
          f"tenure < {args.tenure_threshold:.0f}d): {len(group_b):,}")
    print("=" * 78)
    print(f"Overlap (in both):     {len(overlap):,}")
    print(f"Only in Group A:       {len(only_a):,}")
    print(f"Only in Group B:       {len(only_b):,}")
    union = group_a | group_b
    if union:
        print(f"Jaccard overlap (intersection / union): {len(overlap) / len(union):.1%}")


if __name__ == "__main__":
    main()

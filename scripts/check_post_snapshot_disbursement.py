"""
Tests a specific hypothesis for why the closed-loan-tenure feature fix
didn't move the AUC for the "closed-loan defaulter" subgroup (confirmed
Defaulter blacklist agents whose loan already shows resolved at the July 31
scoring snapshot): maybe the loan that actually caused the August blacklist
was disbursed AFTER the July 31 snapshot -- in which case the snapshot's
closed-loan-tenure features are correctly describing a different, unrelated,
already-repaid PRIOR loan, and the model was never going to see the loan
that actually went bad. That would make the AUC=0.47 result a measurement-
timing artifact, not a real model failure.

Population under test: agents from --matched-file with
is_blacklisted == 1, reason == the Defaulter reason, and
has_unresolved_loan_at_snapshot == 0 -- the exact "WITHOUT unresolved loan"
KEY TEST population from check_defaulter_visibility_at_snapshot.py (derived
independently here rather than depending on that script's own msisdn export,
which can fail to write on Windows if the file is locked).

Cross-checks that population's msisdns against --loan-summary-file for a
disbursement-date column (auto-detects common names; override with
--date-col) and reports how many show a last-disbursement date AFTER
--snapshot-date -- i.e. a loan the snapshot could not possibly have known
about.

Usage:
    python scripts\\check_post_snapshot_disbursement.py ^
        --matched-file wl_bl_eval_matched_agents.csv ^
        --loan-summary-file data\\loan_summary.csv
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
    "loan_date",
    "disbursement_date",
]


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched-file", default="wl_bl_eval_matched_agents.csv")
    ap.add_argument("--loan-summary-file", required=True)
    ap.add_argument("--date-col", default=None,
                     help="Override auto-detection of the disbursement-date column in --loan-summary-file")
    ap.add_argument("--snapshot-date", default="2026-07-31")
    args = ap.parse_args()

    matched_path = Path(args.matched_file)
    if not matched_path.exists():
        sys.exit(f"ERROR: matched file not found: {matched_path}")
    ls_path = Path(args.loan_summary_file)
    if not ls_path.exists():
        sys.exit(f"ERROR: loan-summary file not found: {ls_path}")

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
    target_keys = set(_normalize_msisdn(target[msisdn_col]).dropna())
    print(f"WITHOUT-unresolved Defaulter population: {len(target_keys):,} unique msisdns\n")

    ls_df = pd.read_csv(ls_path)
    ls_msisdn_col = "msisdn" if "msisdn" in ls_df.columns else ("phonenumber" if "phonenumber" in ls_df.columns else None)
    if ls_msisdn_col is None:
        sys.exit(f"ERROR: {ls_path} has neither 'msisdn' nor 'phonenumber' column. "
                  f"Columns present: {list(ls_df.columns)}")

    date_col = args.date_col
    if date_col is None:
        for cand in CANDIDATE_DATE_COLS:
            if cand in ls_df.columns:
                date_col = cand
                break
    if date_col is None:
        print(f"NOTE: could not auto-detect a disbursement-date column in {ls_path}.")
        print(f"Columns present: {list(ls_df.columns)}")
        print(f"Re-run with --date-col <name> once you've identified the right one.")
        sys.exit(1)
    if date_col not in ls_df.columns:
        sys.exit(f"ERROR: --date-col '{date_col}' not found. Columns present: {list(ls_df.columns)}")

    print(f"Using date column: {date_col}\n")

    ls_df["_key"] = _normalize_msisdn(ls_df[ls_msisdn_col])
    ls_df["_date"] = pd.to_datetime(ls_df[date_col], errors="coerce")
    snapshot_date = pd.to_datetime(args.snapshot_date)

    matched_ls = ls_df[ls_df["_key"].isin(target_keys)].copy()
    n_found_in_loan_summary = matched_ls["_key"].nunique()
    print(f"Found in {ls_path.name}: {n_found_in_loan_summary:,} / {len(target_keys):,} "
          f"of the target population")

    has_date = matched_ls.dropna(subset=["_date"])
    n_with_valid_date = has_date["_key"].nunique()
    print(f"Have a parseable {date_col}: {n_with_valid_date:,}\n")

    post_snapshot = has_date[has_date["_date"] > snapshot_date]
    n_post_snapshot = post_snapshot["_key"].nunique()

    print("=" * 78)
    print(f"KEY RESULT: agents with a disbursement AFTER the {args.snapshot_date} snapshot")
    print("=" * 78)
    print(f"{n_post_snapshot:,} / {n_with_valid_date:,} "
          f"({n_post_snapshot / max(1, n_with_valid_date):.1%}) of the WITHOUT-unresolved "
          f"Defaulter population with a known {date_col} show a disbursement AFTER the "
          f"snapshot date.")
    print(
        "\nIf this is a large share, it supports the hypothesis that the July-31 snapshot's "
        "closed-loan-tenure features are describing an earlier, unrelated, already-repaid "
        "loan -- not the loan that actually caused the blacklist -- which would explain why "
        "fixing those features didn't move this subgroup's AUC: the risk event postdates "
        "the scoring snapshot entirely."
    )

    out_path = f"{matched_path.stem}_post_snapshot_disbursement_check.csv"
    matched_ls[["_key", date_col]].rename(columns={"_key": "msisdn_key"}).to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()

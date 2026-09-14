"""
Breaks down the "old flag, clean now" subgroup (WITHOUT-unresolved Defaulter
agents whose last_closed_loan_tenure_days < 30 -- 4,272 of 4,852, AUC=0.4307
per check_recent_bad_loan_vs_old_flag_auc.py) by disbursement month and
closure month of their most recent closed loan.

Derives both dates from the SAME source (the loan-history-snapshot file's
last_closed_loan_closure_date and last_closed_loan_tenure_days) rather than
cross-referencing loan_summary.csv's separately-built last_disbursement_date
-- avoids relying on two independently-constructed populations for what
should be a single, self-consistent loan record:
    disbursement_date = last_closed_loan_closure_date - last_closed_loan_tenure_days

Usage:
    python scripts\\check_old_flag_group_by_period.py ^
        --matched-file wl_bl_eval_matched_agents.csv ^
        --loan-history-file data\\loan_history_snapshot_20260817_retail_filtered.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULTER_REASON = "Defaulter not paid back in the last 30 days"


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def _parse_date_flexible(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce")
    if parsed.notna().any() and parsed.dropna().dt.year.max() <= 1971:
        parsed = pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return parsed


def _print_month_table(dates: pd.Series, label: str):
    print("=" * 78)
    print(f"{label} -- n_non_null={dates.notna().sum():,} / {len(dates):,}")
    print("=" * 78)
    if dates.notna().any():
        print(f"min={dates.min().date()}  median={dates.median().date()}  max={dates.max().date()}\n")
        counts = dates.dt.to_period("M").value_counts().sort_index()
        pct = (counts / dates.notna().sum() * 100).round(1)
        print(pd.DataFrame({"n": counts, "pct": pct}).to_string())
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched-file", default="wl_bl_eval_matched_agents.csv")
    ap.add_argument("--loan-history-file", required=True)
    ap.add_argument("--tenure-threshold", type=float, default=30.0)
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

    matched = target[["_key"]].drop_duplicates().merge(
        lh_df[["_key", "_tenure", "_closure_date", "_disbursement_date"]].drop_duplicates(subset="_key"),
        on="_key", how="left",
    )

    old_flag = matched[matched["_tenure"] < args.tenure_threshold].copy()
    print(f"WITHOUT-unresolved Defaulter population: {len(matched):,}")
    print(f"'Old flag, clean now' group (last_closed_loan_tenure_days < {args.tenure_threshold:.0f}): "
          f"{len(old_flag):,}\n")

    _print_month_table(old_flag["_disbursement_date"], "Disbursement month (derived: closure_date - tenure_days)")
    _print_month_table(old_flag["_closure_date"], "Closure month (last_closed_loan_closure_date)")

    out_path = f"{Path(args.matched_file).stem}_old_flag_group_by_period.csv"
    old_flag.rename(columns={
        "_key": "msisdn", "_tenure": "last_closed_loan_tenure_days",
        "_closure_date": "closure_date", "_disbursement_date": "disbursement_date",
    }).to_csv(out_path, index=False)
    print(f"Per-agent detail written: {out_path}")


if __name__ == "__main__":
    main()

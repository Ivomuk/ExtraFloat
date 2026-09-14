"""
Exports the specific list of agents requested: on the Defaulter blacklist,
but whose last (most recent) closed loan closed off in a day or two --
meaning that loan cannot be the one that triggered "not paid back in the
last 30 days" (not enough elapsed time between disbursement and closure),
so if the flag is valid it must trace to an older loan in their history.

This is the sharpest-cut, ready-to-hand-off version of the "old flag,
clean now" group from check_recent_bad_loan_vs_old_flag_auc.py / check_
old_flag_group_by_period.py, filtered to a tight tenure cutoff and written
out with per-agent detail (msisdn, cal_pd, risk_tier, the last closed
loan's tenure/closure/derived-disbursement dates) rather than just
aggregate counts.

Usage:
    python scripts\\export_defaulters_with_fast_last_loan.py ^
        --matched-file wl_bl_eval_matched_agents.csv ^
        --loan-history-file data\\loan_history_snapshot_20260817_retail_filtered.csv ^
        --max-tenure-days 2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULTER_REASON = "Defaulter not paid back in the last 30 days"

EXTRA_MATCHED_COLS = ["cal_pd", "pd_decile", "risk_tier", "assigned_limit", "final_decision_reason"]


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
    ap.add_argument("--loan-history-file", required=True)
    ap.add_argument("--max-tenure-days", type=float, default=2.0,
                     help="Keep agents whose last_closed_loan_tenure_days is <= this many days")
    ap.add_argument("--out", default=None)
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
    keep_cols = ["_key", msisdn_col] + [c for c in EXTRA_MATCHED_COLS if c in target.columns]
    target = target[keep_cols].drop_duplicates(subset="_key")

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
    lh_df["last_closed_loan_tenure_days"] = pd.to_numeric(lh_df["last_closed_loan_tenure_days"], errors="coerce")
    lh_df["last_closed_loan_closure_date"] = _parse_date_flexible(lh_df["last_closed_loan_closure_date"])
    lh_df["derived_last_disbursement_date"] = (
        lh_df["last_closed_loan_closure_date"]
        - pd.to_timedelta(lh_df["last_closed_loan_tenure_days"], unit="D")
    )

    lh_cols = ["_key", "last_closed_loan_tenure_days", "last_closed_loan_closure_date",
               "derived_last_disbursement_date"]
    if "avg_tenure_days_closed_loans" in lh_df.columns:
        lh_cols.append("avg_tenure_days_closed_loans")
    if "closed_loan_count" in lh_df.columns:
        lh_cols.append("closed_loan_count")
    elif "observed_loan_count" in lh_df.columns:
        lh_cols.append("observed_loan_count")

    merged = target.merge(
        lh_df[lh_cols].drop_duplicates(subset="_key"), on="_key", how="left"
    )

    result = merged[merged["last_closed_loan_tenure_days"] <= args.max_tenure_days].copy()
    result = result.drop(columns=["_key"]).sort_values("last_closed_loan_tenure_days")

    print(f"WITHOUT-unresolved Defaulter population: {len(target):,}")
    print(f"Agents whose last closed loan closed in <= {args.max_tenure_days:.0f} day(s): {len(result):,} "
          f"({len(result) / len(target):.1%})\n")
    print(result.head(20).to_string(index=False))
    if len(result) > 20:
        print(f"... ({len(result) - 20:,} more rows in the exported file)")

    out_path = args.out or f"{matched_path.stem}_fast_last_loan_le{int(args.max_tenure_days)}d.csv"
    result.to_csv(out_path, index=False)
    print(f"\nFull list written: {out_path}")


if __name__ == "__main__":
    main()

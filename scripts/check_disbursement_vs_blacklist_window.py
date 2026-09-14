"""
Sharper follow-up to check_post_snapshot_disbursement.py's "disbursed after
the snapshot" check. Per the "Defaulter not paid back in the last 30+1
days" definition (a specific loan crosses 30+1 days unpaid since ITS OWN
disbursement date), a loan disbursed less than 31 days before the blacklist
date cannot possibly be the loan that triggered the flag -- there hasn't
been enough time for it to reach 30+1 days unpaid yet.

Splits the WITHOUT-unresolved Defaulter population three ways by their
last_disbursement_date relative to [blacklist_date - 31 days, blacklist_date]:

  - BEFORE the window (disbursed on/before blacklist_date - 31 days):
    old enough that an EARLIER loan (not necessarily this "last" one)
    plausibly crossed 30+1 days unpaid and triggered the flag at the time --
    consistent with the flag being a permanent record of a past event,
    unrelated to their current (already-closed) most recent loan.

  - INSIDE the window (disbursed within 31 days of the blacklist date):
    this specific loan mathematically CANNOT be the trigger (not enough
    elapsed time) -- if it already shows closed/repaid, it's a separate,
    apparently fine transaction. Doesn't by itself disprove the flag (these
    agents typically have ~39 total loans -- the real trigger could still be
    an older one), but does mean new credit was extended close to an
    already-standing-or-imminent Defaulter flag, worth a business-process
    look.

  - AFTER the window (disbursed after the blacklist date itself):
    the check_post_snapshot_disbursement.py population -- new credit
    extended to an agent already on the blacklist.

Usage:
    python scripts\\check_disbursement_vs_blacklist_window.py ^
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
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--blacklist-date", default="2026-08-04")
    ap.add_argument("--min-unpaid-days", type=int, default=31,
                     help="Days since disbursement required to trigger the Defaulter flag")
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
    target["_key"] = _normalize_msisdn(target[msisdn_col])
    print(f"WITHOUT-unresolved Defaulter population: {target['_key'].nunique():,} unique msisdns\n")

    ls_df = pd.read_csv(ls_path)
    ls_msisdn_col = "msisdn" if "msisdn" in ls_df.columns else ("phonenumber" if "phonenumber" in ls_df.columns else None)
    if ls_msisdn_col is None:
        sys.exit(f"ERROR: {ls_path} has neither 'msisdn' nor 'phonenumber' column. "
                  f"Columns present: {list(ls_df.columns)}")
    date_col = args.date_col or next((c for c in CANDIDATE_DATE_COLS if c in ls_df.columns), None)
    if date_col is None:
        sys.exit(f"ERROR: could not auto-detect a disbursement-date column in {ls_path}. "
                  f"Columns present: {list(ls_df.columns)}. Re-run with --date-col.")

    ls_df["_key"] = _normalize_msisdn(ls_df[ls_msisdn_col])
    ls_df["_date"] = _parse_date_flexible(ls_df[date_col])

    blacklist_date = pd.to_datetime(args.blacklist_date)
    window_start = blacklist_date - pd.Timedelta(days=args.min_unpaid_days)

    matched = target[["_key"]].drop_duplicates().merge(
        ls_df[["_key", "_date"]].drop_duplicates(subset="_key"), on="_key", how="left"
    )
    n_with_date = matched["_date"].notna().sum()
    print(f"Have a parseable {date_col}: {n_with_date:,} / {len(matched):,}\n")

    before = matched[matched["_date"] <= window_start]
    inside = matched[(matched["_date"] > window_start) & (matched["_date"] <= blacklist_date)]
    after = matched[matched["_date"] > blacklist_date]

    print("=" * 78)
    print(f"Three-way split by last_disbursement_date vs. "
          f"[{window_start.date()}, {blacklist_date.date()}]")
    print("=" * 78)
    print(f"BEFORE window (<= {window_start.date()}, an earlier loan could plausibly have "
          f"triggered the flag): {len(before):,} ({len(before) / n_with_date:.1%})")
    print(f"INSIDE window ({window_start.date()} to {blacklist_date.date()}, this specific "
          f"loan CANNOT be the trigger -- not enough elapsed time): "
          f"{len(inside):,} ({len(inside) / n_with_date:.1%})")
    print(f"AFTER window (> {blacklist_date.date()}, new credit extended after the "
          f"blacklist date itself): {len(after):,} ({len(after) / n_with_date:.1%})")

    print(
        "\nInterpretation: BEFORE-window agents are consistent with the flag being a "
        "permanent record of an OLDER loan (plausible given ~39 median prior loans -- the "
        "trigger doesn't have to be their most recent one). INSIDE-window agents received a "
        "new, apparently-repaid loan close to their blacklist date -- that loan itself can't "
        "be the trigger, so if the flag is valid it must trace to a still-older loan; if the "
        "business's lending process should be blocking new credit to already-flagged agents, "
        "this group (and definitely the AFTER-window group) is worth raising directly."
    )

    out_path = f"{Path(args.matched_file).stem}_disbursement_vs_blacklist_window.csv"
    matched.rename(columns={"_key": "msisdn", "_date": date_col}).to_csv(out_path, index=False)
    print(f"\nPer-agent detail written: {out_path}")


if __name__ == "__main__":
    main()

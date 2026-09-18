"""
Two things, against the raw fin_log disbursements export:

1. Check whether any single agent (msisdn) is ever recorded under more than
   one distinct instruct_to_fro_user_prf profile value. Since
   summarize_limit_violations_by_agent_profile.py found this field is
   almost entirely a single constant value ("MTNU Agent Silver Class"),
   this confirms whether profile can safely be treated as one fixed
   attribute per agent, or whether some agents genuinely switch profiles.

2. Build a monthly per-agent summary file: one row per (msisdn, profile,
   month), with total disbursed amount and transaction count for that
   month. If an agent used more than one profile within the same month,
   that agent simply gets more than one row for that month -- this is not
   collapsed or hidden.

Usage:
    python scripts\\build_monthly_disbursement_summary.py ^
        --disbursements-file data\\fin_log_202609181256 ^
        --out monthly_disbursement_summary.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLS = ["instruct_to_fro_msisdn", "instruct_amount", "tbl_dt", "instruct_to_fro_user_prf"]


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".csv", "") or path.suffix.lower() not in (".tsv", ".txt"):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    for sep in ["\t", "|", ";"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    sys.exit(f"ERROR: could not parse {path} as CSV/TSV. Check its delimiter and re-run with "
              f"the right pandas.read_csv args, or tell me the actual format.")


def _parse_date_flexible(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce")
    if parsed.dropna().dt.year.le(1971).all():
        parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return parsed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disbursements-file", required=True)
    ap.add_argument("--out", default="monthly_disbursement_summary.csv")
    args = ap.parse_args()

    path = Path(args.disbursements_file)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")

    df = _read_any(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}. "
                  f"Columns present: {list(df.columns)}")

    df["_msisdn"] = _normalize_msisdn(df["instruct_to_fro_msisdn"])
    df["instruct_amount"] = pd.to_numeric(df["instruct_amount"], errors="coerce")
    df["_date"] = _parse_date_flexible(df["tbl_dt"])
    df["_month"] = df["_date"].dt.to_period("M").astype(str)
    df = df[df["_msisdn"].notna()]

    n_rows = len(df)
    n_agents = df["_msisdn"].nunique()
    print(f"Transactions: {n_rows:,}")
    print(f"Distinct agents (msisdn): {n_agents:,}\n")

    # -- Check 1: does any agent carry more than one distinct profile value? --
    profiles_per_agent = df.groupby("_msisdn")["instruct_to_fro_user_prf"].nunique()
    multi_profile_agents = profiles_per_agent[profiles_per_agent > 1]

    print("=" * 78)
    print("Check: agents with more than one distinct instruct_to_fro_user_prf value")
    print("=" * 78)
    if len(multi_profile_agents) == 0:
        print("NONE. Every agent in this file is recorded under exactly one profile value. "
              "Profile can be treated as a fixed per-agent attribute.")
    else:
        print(f"{len(multi_profile_agents):,} / {n_agents:,} agents have more than one distinct "
              f"profile value:")
        detail = (
            df[df["_msisdn"].isin(multi_profile_agents.index)]
            .groupby("_msisdn")["instruct_to_fro_user_prf"]
            .unique()
        )
        print(detail.to_string())

    # -- Build the monthly summary, one row per (msisdn, profile, month) --
    monthly = (
        df.groupby(["_msisdn", "instruct_to_fro_user_prf", "_month"], as_index=False)
        .agg(
            disbursed_amount=("instruct_amount", "sum"),
            n_transactions=("instruct_amount", "size"),
        )
        .rename(columns={"_msisdn": "msisdn", "instruct_to_fro_user_prf": "profile", "_month": "month"})
        .sort_values(["msisdn", "month"])
    )

    n_agent_months_multi_profile = (
        monthly.groupby(["msisdn", "month"])["profile"].transform("nunique") > 1
    ).sum()
    if n_agent_months_multi_profile:
        print(f"\nNOTE: {n_agent_months_multi_profile:,} row(s) belong to an agent/month that has "
              f"more than one profile row for that same month -- not collapsed, kept as separate rows.")

    monthly.to_csv(args.out, index=False)
    print(f"\nMonthly summary written: {args.out}  ({len(monthly):,} rows: one per msisdn/profile/month)")


if __name__ == "__main__":
    main()

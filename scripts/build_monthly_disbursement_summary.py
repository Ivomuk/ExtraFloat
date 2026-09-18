"""
Against the raw fin_log disbursements export, checks -- per agent per month
-- whether instruct_to_fro_user_prf (profile) and instruct_amount (disbursed
amount) are BOTH constant across all of that agent's transactions.

This is not a monthly total: the point is to compare what an agent actually
received against the engine's assigned_limit, and a summed total isn't
comparable to a per-transaction limit. So for agent/months where every
transaction disbursed the exact same amount under the exact same profile,
that single repeated amount unambiguously IS "the disbursement" for that
agent that month, and gets printed as one row:

    msisdn  profile  month  disbursed_amount  n_transactions

Agent/months where the profile or the amount varies across transactions are
NOT collapsed into a single row (no sum, no average) -- they're written to a
separate file for inspection, since a single "disbursed_amount" wouldn't
mean anything for them. For those, two extra columns show each distinct
amount and the date it was first received, in chronological order and
positionally aligned, e.g.:

    dates_received:   2026-08-02; 2026-08-13
    distinct_amounts: 750,000; 200,000

Usage:
    python scripts\\build_monthly_disbursement_summary.py ^
        --disbursements-file data\\fin_log_202609181256 ^
        --out monthly_disbursement_summary.csv ^
        --variable-out variable_disbursement_agents.csv
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
    ap.add_argument("--variable-out", default="variable_disbursement_agents.csv")
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

    # -- Per (msisdn, month): is profile constant? Is instruct_amount constant? --
    grp = df.groupby(["_msisdn", "_month"])
    agg = grp.agg(
        n_distinct_profiles=("instruct_to_fro_user_prf", "nunique"),
        n_distinct_amounts=("instruct_amount", "nunique"),
        n_transactions=("instruct_amount", "size"),
        profile=("instruct_to_fro_user_prf", "first"),
        disbursed_amount=("instruct_amount", "first"),
    ).reset_index().rename(columns={"_msisdn": "msisdn", "_month": "month"})

    is_consistent = (agg["n_distinct_profiles"] == 1) & (agg["n_distinct_amounts"] == 1)
    consistent = agg[is_consistent].copy()
    variable = agg[~is_consistent].copy()

    print("=" * 78)
    print("Check: agent/months where profile AND disbursed amount are both constant")
    print("=" * 78)
    print(f"Consistent agent/months (single profile, single amount): {len(consistent):,}")
    print(f"Variable agent/months (profile and/or amount changes):   {len(variable):,}")

    out_cols = ["msisdn", "profile", "month", "disbursed_amount", "n_transactions"]
    consistent[out_cols].sort_values(["msisdn", "month"]).to_csv(args.out, index=False)
    print(f"\nConsistent agent/months written: {args.out}  ({len(consistent):,} rows)")

    if len(variable):
        df["_key"] = list(zip(df["_msisdn"], df["_month"]))
        var_keys = set(zip(variable["msisdn"], variable["month"]))
        df_var = df[df["_key"].isin(var_keys)]

        amt_dates = (
            df_var.groupby(["_msisdn", "_month", "instruct_amount"], as_index=False)["_date"]
            .min()
            .rename(columns={"_date": "first_date"})
            .sort_values(["_msisdn", "_month", "first_date"])
        )
        detail = (
            amt_dates.groupby(["_msisdn", "_month"])
            .agg(
                dates_received=("first_date", lambda s: "; ".join(d.strftime("%Y-%m-%d") for d in s)),
                distinct_amounts=("instruct_amount", lambda s: "; ".join(f"{v:,.0f}" for v in s)),
            )
            .reset_index()
            .rename(columns={"_msisdn": "msisdn", "_month": "month"})
        )
        variable = variable.merge(detail, on=["msisdn", "month"], how="left")

        variable_cols = ["msisdn", "month", "n_distinct_profiles", "n_distinct_amounts",
                          "n_transactions", "dates_received", "distinct_amounts"]
        variable[variable_cols].sort_values(["msisdn", "month"]).to_csv(args.variable_out, index=False)
        print(f"Variable agent/months written for inspection: {args.variable_out}  ({len(variable):,} rows)")
        print(
            "\nThese were NOT summed or averaged -- a single 'disbursed_amount' figure "
            "isn't meaningful for an agent/month where the profile or amount actually "
            "changed across transactions. Inspect them separately before deciding how "
            "(or whether) to compare them to the engine's assigned_limit."
        )
    else:
        print("\nEvery agent/month in this file has a single constant profile and amount -- "
              "no variable cases to inspect separately.")


if __name__ == "__main__":
    main()

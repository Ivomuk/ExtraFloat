"""
check_mart_column_completeness.py
==================================
Checks real missing-value rates for a fixed list of MoMo-mart columns,
directly against the actual file this repo's pipelines read -- not a
same-named proxy from a different month's EDA report.

Why this exists: a May-2026 EDA profiling report showed revenue_1m/
revenue_3m/revenue_6m at 0% missing while rev_1m/rev_3m/rev_6m were 100%
missing -- looked like a simple "wrong column name" bug in
segmentation/borrower_persona_clustering.py's load_and_join(). The real
July file then showed BOTH naming variants 100% missing, contradicting
the May proxy. That means the May report can't be trusted for any other
column either -- this script checks the real file directly instead of
inferring from a different month's sample.

Covers: the revenue columns (both naming variants, for the record), every
column already used in persona clustering (as a sanity baseline), and the
float-utilization candidate columns business asked about (cash_in_*,
payment_value_*, payment_cust_*, cust_1m/6m, cash_out_cust_*) -- so a
single run tells you exactly which of these are safe to add as clustering
features and which aren't.

Also flags constant/zero-variance columns (a single distinct value across
the whole file), not just NaN-missingness. A May-2026 EDA sample showed
revenue_1m/revenue_3m/revenue_6m at 0% missing but Distinct=1, value
always "0.0" -- i.e. functionally as dead as 100% missing despite passing
a naive missingness check. Checking only isna() would silently repeat
that mistake against the real file.

Usage:
    python scripts\\check_mart_column_completeness.py
    python scripts\\check_mart_column_completeness.py --transaction-file data\\mfs_daily_agent_mart_20260731.csv
"""

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TRANSACTION_FILE = REPO / "data" / "mfs_daily_agent_mart_20260731.csv"

CANDIDATE_COLUMNS = [
    # Revenue -- both naming variants, both suspected/confirmed dead.
    "rev_1m", "rev_3m", "rev_6m",
    "revenue_1m", "revenue_3m", "revenue_6m",
    # Already used in persona clustering -- sanity baseline.
    "commission", "account_balance",
    "cash_out_vol_3m", "cash_out_value_3m", "payment_vol_3m", "cust_3m",
    # Float-utilization candidates (cash-in side -- currently entirely
    # absent from the clustering feature set).
    "cash_in_vol_1m", "cash_in_vol_3m", "cash_in_vol_6m",
    "cash_in_value_1m", "cash_in_value_3m", "cash_in_value_6m",
    "cash_in_cust_1m", "cash_in_cust_3m", "cash_in_cust_6m",
    # Float-utilization candidates (payment side -- vol already used at
    # 3m; value and per-channel customer counts are not).
    "payment_vol_1m", "payment_vol_6m",
    "payment_value_1m", "payment_value_3m", "payment_value_6m",
    "payment_cust_1m", "payment_cust_3m", "payment_cust_6m",
    # Other candidates business named directly.
    "cust_1m", "cust_6m",
    "cash_out_cust_1m", "cash_out_cust_3m", "cash_out_cust_6m",
    "cash_out_vol_1m", "cash_out_vol_6m",
    "cash_out_value_1m", "cash_out_value_6m",
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transaction-file", type=Path, default=DEFAULT_TRANSACTION_FILE)
    args = p.parse_args(argv)

    print(f"=== Checking column completeness in {args.transaction_file} ===")
    available_cols = set(pd.read_csv(args.transaction_file, nrows=0).columns)

    present = [c for c in CANDIDATE_COLUMNS if c in available_cols]
    absent = [c for c in CANDIDATE_COLUMNS if c not in available_cols]
    if absent:
        print(f"  NOT IN FILE AT ALL ({len(absent)}): {absent}")

    df = pd.read_csv(args.transaction_file, usecols=present)
    n_rows = len(df)
    print(f"  {n_rows:,} rows, checking {len(present)} columns\n")

    rows = []
    for col in present:
        s = df[col]
        n_missing = int(s.isna().sum())
        pct_missing = round(n_missing / n_rows * 100, 2) if n_rows else float("nan")
        non_null = s.dropna()
        n_distinct = int(non_null.nunique())
        is_constant = n_distinct <= 1
        constant_value = non_null.iloc[0] if is_constant and len(non_null) else None
        row = {
            "column": col,
            "pct_missing": pct_missing,
            "n_missing": n_missing,
            "dtype": str(s.dtype),
            "n_distinct": n_distinct,
            "is_constant": is_constant,
            "constant_value": constant_value,
        }
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().any():
            row["median"] = numeric.median()
            row["max"] = numeric.max()
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("pct_missing", ascending=False)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(summary.to_string(index=False))

    fully_missing = summary[summary["pct_missing"] == 100.0]["column"].tolist()
    constant = summary[summary["is_constant"] & (summary["pct_missing"] < 100.0)]
    if fully_missing:
        print(f"\n  100% MISSING -- do not use these: {fully_missing}")
    else:
        print("\n  No column in this list is 100% missing.")

    if len(constant):
        details = [f"{r.column} (always {r.constant_value!r})" for r in constant.itertuples()]
        print(f"  CONSTANT / ZERO-VARIANCE -- not missing but functionally dead, do not use these: {details}")
    else:
        print("  No column in this list is constant.")


if __name__ == "__main__":
    main()

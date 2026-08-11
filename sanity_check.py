"""
sanity_check.py
===============
Post-run sanity checks on the credit risk pipeline output CSV.
Prints a PASS / FAIL report for each check and shows violating rows.

Usage:
    python sanity_check.py
    python sanity_check.py --output output/engine_test_output.csv
"""

import argparse
import sys
import pandas as pd

REGULATORY_CAP   = 5_000_000   # Bank of Uganda hard ceiling
THIN_FILE_CAP    = 100_000     # Bronze flat — thin-file agents
NEW_BRONZE_CAP   =  50_000     # New Bronze flat
GLOBAL_FLOOR     =       0

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def check(label: str, mask: pd.Series, df: pd.DataFrame, show_cols: list[str]) -> bool:
    violations = df[mask]
    n = len(violations)
    if n == 0:
        print(f"  {PASS}  {label}")
        return True
    print(f"  {FAIL}  {label}  ({n:,} agents)")
    print(violations[show_cols].head(10).to_string(index=False))
    print()
    return False


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="output/engine_test_output.csv")
    args = p.parse_args(argv)

    print(f"\nLoading: {args.output}")
    try:
        df = pd.read_csv(args.output)
    except FileNotFoundError:
        print(f"ERROR: file not found — {args.output}")
        sys.exit(1)

    print(f"Rows loaded: {len(df):,}\n")

    base_cols   = ["msisdn", "assigned_limit", "risk_tier", "cal_pd", "final_decision_reason", "score_source"]
    thin_cols   = base_cols + ["thin_file_flag"] if "thin_file_flag" in df.columns else base_cols
    tier_cols   = base_cols + ["capacity_tier"]  if "capacity_tier"  in df.columns else base_cols

    results = []
    print("=" * 60)
    print(" Limit range checks")
    print("=" * 60)

    results.append(check(
        "No limit above regulatory cap (5,000,000 UGX)",
        df["assigned_limit"] > REGULATORY_CAP, df, base_cols,
    ))
    results.append(check(
        "No limit above 250,000 UGX (Silver ceiling)",
        df["assigned_limit"] > 250_000, df, base_cols,
    ))
    results.append(check(
        "No limit below 0",
        df["assigned_limit"] < GLOBAL_FLOOR, df, base_cols,
    ))
    results.append(check(
        "No NaN assigned_limit",
        df["assigned_limit"].isna(), df, base_cols,
    ))

    print("=" * 60)
    print(" Thin-file checks")
    print("=" * 60)

    if "thin_file_flag" in df.columns:
        results.append(check(
            "Thin-file agents not above Bronze cap (100,000 UGX)",
            (df["thin_file_flag"] == 1) & (df["assigned_limit"] > THIN_FILE_CAP),
            df, thin_cols,
        ))
        results.append(check(
            "Thin-file agents with cal_pd below 0.12 floor",
            (df["thin_file_flag"] == 1) & (df["cal_pd"] < 0.12) & df["cal_pd"].notna(),
            df, thin_cols,
        ))
    else:
        print("  (thin_file_flag column not in output — skipped)")

    print("=" * 60)
    print(" Risk tier consistency")
    print("=" * 60)

    if "cal_pd" in df.columns:
        scored = df[df["cal_pd"].notna()]
        results.append(check(
            "tier_1 agents all have cal_pd < 0.15",
            (scored["risk_tier"] == "tier_1") & (scored["cal_pd"] >= 0.15),
            df, base_cols,
        ))
        results.append(check(
            "tier_4 agents all have cal_pd >= 0.65",
            (scored["risk_tier"] == "tier_4") & (scored["cal_pd"] < 0.65),
            df, base_cols,
        ))

    nan_pd = df["cal_pd"].isna().sum()
    print(f"  {'!' if nan_pd else ' '}  Agents with NaN cal_pd (7-signal fallback): {nan_pd:,}")
    print()

    print("=" * 60)
    print(" Score source breakdown")
    print("=" * 60)

    if "score_source" in df.columns:
        print(df["score_source"].value_counts().to_string())
        print()

    print("=" * 60)
    print(" Limit distribution")
    print("=" * 60)

    print(df["assigned_limit"].describe().apply(lambda x: f"{x:,.0f}").to_string())
    print()
    print("Limit value counts (top 15):")
    print(df["assigned_limit"].value_counts().head(15).to_string())
    print()

    print("=" * 60)
    print(" Risk tier distribution")
    print("=" * 60)

    print(df["risk_tier"].value_counts().to_string())
    print()

    if "capacity_tier" in df.columns:
        print("=" * 60)
        print(" Capacity tier distribution")
        print("=" * 60)
        print(df["capacity_tier"].value_counts().to_string())
        print()

    print("=" * 60)
    passed = sum(results)
    total  = len(results)
    status = "ALL CHECKS PASSED" if passed == total else f"{total - passed} CHECK(S) FAILED"
    print(f" {status}  ({passed}/{total} passed)")
    print("=" * 60)
    print()

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()

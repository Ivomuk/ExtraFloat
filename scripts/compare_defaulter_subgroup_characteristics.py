"""
Follow-up to check_defaulter_visibility_at_snapshot.py's KEY TEST: within
the "Defaulter not paid back in the last 30 days" cohort, compares every
available characteristic between the two subgroups split by
has_unresolved_loan_at_snapshot -- the well-discriminated group (AUC~0.75,
loan still shows unresolved) vs. the poorly-discriminated group (AUC~0.47,
loan shows resolved/closed at snapshot despite the agent being a confirmed
non-payer days later).

Answers "what does the model actually see differently for these two
groups" using columns already present in --matched-file
(wl_bl_eval_matched_agents.csv, which carries every engine_test_output.csv
column via its own merge) -- no new data export needed.

Usage:
    python scripts\\compare_defaulter_subgroup_characteristics.py ^
        --matched-file wl_bl_eval_matched_agents.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULTER_REASON = "Defaulter not paid back in the last 30 days"

NUMERIC_CHARACTERISTIC_COLS = [
    "cal_pd", "assigned_limit", "assigned_limit_pre_round", "pd_decile",
    "capacity_score", "active_loan_days_aging_at_snapshot",
]
CATEGORICAL_CHARACTERISTIC_COLS = [
    "risk_tier", "capacity_tier", "capacity_tier_raw", "combined_top_driver",
    "combined_reason", "policy_reason", "final_decision_reason",
    "score_source", "is_anomaly", "risk_unresolved_loan_haircut_reason",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched-file", default="wl_bl_eval_matched_agents.csv")
    ap.add_argument("--top-n-categorical", type=int, default=8,
                     help="Max distinct values shown per categorical column")
    args = ap.parse_args()

    matched_path = Path(args.matched_file)
    if not matched_path.exists():
        sys.exit(f"ERROR: matched file not found: {matched_path}")

    df = pd.read_csv(matched_path)
    required = ["is_blacklisted", "reason", "has_unresolved_loan_at_snapshot"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        sys.exit(
            f"ERROR: {matched_path} is missing required column(s): {missing_req}. "
            f"has_unresolved_loan_at_snapshot requires engine_test_output.csv to have "
            f"been built with --loan-history-file."
        )

    defaulters = df[(df["is_blacklisted"] == 1) & (df["reason"] == DEFAULTER_REASON)].copy()
    if defaulters.empty:
        sys.exit(f"ERROR: no rows with reason == '{DEFAULTER_REASON}' found in {matched_path}")

    with_unresolved = defaulters[defaulters["has_unresolved_loan_at_snapshot"] == 1]
    without_unresolved = defaulters[defaulters["has_unresolved_loan_at_snapshot"] == 0]
    print(f"Defaulter cohort: {len(defaulters):,} total "
          f"(WITH unresolved loan: {len(with_unresolved):,} | WITHOUT: {len(without_unresolved):,})\n")

    print("=" * 78)
    print("Numeric characteristics: WITH unresolved loan  vs.  WITHOUT unresolved loan")
    print("=" * 78)
    rows = []
    for col in NUMERIC_CHARACTERISTIC_COLS:
        if col not in defaulters.columns:
            continue
        w = pd.to_numeric(with_unresolved[col], errors="coerce")
        wo = pd.to_numeric(without_unresolved[col], errors="coerce")
        rows.append({
            "characteristic": col,
            "with_unresolved_mean": w.mean(),
            "without_unresolved_mean": wo.mean(),
            "with_unresolved_median": w.median(),
            "without_unresolved_median": wo.median(),
        })
    if rows:
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    else:
        print("(none of the expected numeric columns are present)")

    print("\n" + "=" * 78)
    print("Categorical characteristics -- top values, WITH vs. WITHOUT unresolved loan")
    print("=" * 78)
    for col in CATEGORICAL_CHARACTERISTIC_COLS:
        if col not in defaulters.columns:
            continue
        w_counts = with_unresolved[col].fillna("(null)").value_counts(normalize=True).head(args.top_n_categorical)
        wo_counts = without_unresolved[col].fillna("(null)").value_counts(normalize=True).head(args.top_n_categorical)
        cmp_tbl = pd.DataFrame({
            "with_unresolved_pct": w_counts,
            "without_unresolved_pct": wo_counts,
        }).fillna(0.0).sort_values("without_unresolved_pct", ascending=False)
        print(f"\n--- {col} ---")
        print((cmp_tbl * 100).round(1).to_string())


if __name__ == "__main__":
    main()

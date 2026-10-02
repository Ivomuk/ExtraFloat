"""
Persona(t) -> Performance(t+1..n): joins the already-frozen K=8 persona
assignments (k8_cluster_assignments.csv, t = the 2026-07-31 snapshot) against
the forward-window loan outcomes pulled by
data/persona_k8_forward_outcomes_query.sql, and summarizes realized
performance per persona.

This is the step that upgrades the provisional persona names from
"supported by snapshot-level evidence" to "supported by forward-looking
evidence" -- or surfaces where they don't hold up, which is just as useful.

Does NOT retrain or re-touch the K=8 clustering itself -- persona_cluster
assignments are read as fixed, already-decided labels; only the forward
outcome data is new here.

Usage:
    python scripts\\analyze_persona_k8_forward_outcomes.py --forward-outcomes-file data\\persona_k8_forward_outcomes.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from segmentation.borrower_persona_clustering import digits  # noqa: E402
from scripts.profile_persona_k8 import PERSONA_NAMES, PERSONA_RATIONALE  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ASSIGNMENTS_PATH = REPO / "segmentation_outputs" / "persona_k8_profile" / "k8_cluster_assignments.csv"
OUT_DIR = REPO / "segmentation_outputs" / "persona_k8_profile"

RATE_COLS = ["fwd_any_bad_3dpd", "fwd_any_anomaly_open", "fwd_still_active_at_window_end"]
# Meaningful across the WHOLE assigned population, zero included -- "zero
# days past due" is a valid, correct statement even for a borrower with no
# forward activity at all.
WHOLE_POPULATION_MEDIAN_COLS = ["fwd_worst_days_aging"]
# Zero-dominated if computed across everyone (most borrowers take no new
# loan in a 39-day window) -- computed only among borrowers who actually
# took >=1 new loan, alongside a take-rate, so the comparison across
# personas is informative rather than "0.0 for every cluster."
NEW_LOAN_CONDITIONAL_MEDIAN_COLS = [
    "fwd_new_loan_count", "fwd_new_loans_disbursed_ugx", "fwd_new_loans_repaid_ugx",
    "fwd_new_loans_repayment_ratio",
]
COUNT_COLS = ["fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forward-outcomes-file", type=Path, required=True,
                   help="CSV exported from data/persona_k8_forward_outcomes_query.sql")
    p.add_argument("--assignments-file", type=Path, default=ASSIGNMENTS_PATH,
                   help=f"default: {ASSIGNMENTS_PATH}")
    args = p.parse_args(argv)

    print("=== Load persona assignments (t = the frozen K=8 snapshot) ===")
    assignments = pd.read_csv(args.assignments_file)
    assignments["_id"] = digits(assignments["phonenumber"])
    n_assigned = len(assignments)
    print(f"  {n_assigned:,} borrowers with a persona assignment")

    print("\n=== Load forward-window outcomes ===")
    fwd = pd.read_csv(args.forward_outcomes_file)
    fwd["_id"] = digits(fwd["customer_msisdn"])
    window_days = fwd["fwd_window_days"].iloc[0] if "fwd_window_days" in fwd.columns and len(fwd) else None
    window_start = fwd["fwd_window_start_exclusive"].iloc[0] if "fwd_window_start_exclusive" in fwd.columns and len(fwd) else None
    window_end = fwd["fwd_window_end"].iloc[0] if "fwd_window_end" in fwd.columns and len(fwd) else None
    print(f"  {len(fwd):,} borrowers with ANY loan-state activity in the forward window")
    print(f"  forward window: ({window_start}, {window_end}] -- {window_days} days")
    if window_days is not None and window_days < 30:
        print(f"  WARNING: only {window_days} days of forward data -- early read, treat conclusions as "
              f"directional, not final. Re-run this analysis as more time passes.")

    print("\n=== Join (left join on assignments -- every assigned borrower gets a row) ===")
    merged = assignments.merge(
        fwd.drop(columns=["customer_msisdn"]), on="_id", how="left", validate="one_to_one",
    )
    # Captured BEFORE zero-fill below -- fwd_new_loan_count.notna() would
    # always be True afterward since the fillna makes every row non-null.
    merged["_had_fwd_activity"] = merged["fwd_new_loan_count"].notna()
    had_activity = merged["_had_fwd_activity"]
    print(f"  {had_activity.sum():,} / {n_assigned:,} assigned borrowers ({had_activity.mean():.1%}) "
          f"had any loan-state activity in the forward window")
    print(f"  {(~had_activity).sum():,} ({(~had_activity).mean():.1%}) had NONE -- dormant through "
          f"window_end, or the window hasn't caught up to them yet. Reported as a population split "
          f"below, not silently zero-filled.")

    # Zero-fill only the columns where "no row in fwd" genuinely means "zero
    # activity" -- never zero-fill fwd_new_loans_repayment_ratio (undefined,
    # not zero, when there was no new borrowing) or fwd_worst_days_aging
    # (no loan activity is not the same claim as "zero days past due").
    zero_fill_cols = [
        "fwd_new_loan_count", "fwd_new_loans_disbursed_ugx", "fwd_new_loans_repaid_ugx",
        "fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count",
        "fwd_any_bad_3dpd", "fwd_any_anomaly_open", "fwd_still_active_at_window_end",
    ]
    for col in zero_fill_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    print("\n=== Per-persona forward-outcome summary ===")
    rows = []
    for cluster_id, idx in merged.groupby("persona_cluster").groups.items():
        sub = merged.loc[idx]
        took_new_loan = sub["fwd_new_loan_count"] > 0
        row = {
            "persona_cluster": cluster_id,
            "persona_name": PERSONA_NAMES.get(cluster_id, "(unnamed)"),
            "n_borrowers": len(sub),
            "pct_with_fwd_activity": round(sub["_had_fwd_activity"].mean() * 100, 1),
            "pct_took_new_loan": round(took_new_loan.mean() * 100, 1),
        }
        for col in RATE_COLS:
            if col in sub.columns:
                row[f"{col}_rate_pct"] = round(sub[col].mean() * 100, 2)
        for col in WHOLE_POPULATION_MEDIAN_COLS:
            if col in sub.columns:
                row[f"{col}_median"] = sub[col].median()
        for col in NEW_LOAN_CONDITIONAL_MEDIAN_COLS:
            if col in sub.columns:
                row[f"{col}_median_among_takers"] = sub.loc[took_new_loan, col].median() if took_new_loan.any() else None
        for col in COUNT_COLS:
            if col in sub.columns:
                row[f"{col}_total"] = int(sub[col].sum())
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("persona_cluster")
    out_path = OUT_DIR / "persona_k8_forward_outcomes_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary.to_string(index=False))

    print(
        "\nReading this: fwd_any_bad_3dpd_rate_pct is the closest thing here to a true forward bad "
        "rate (any loan, new or carried over, reaching >3 days past due during the window) -- compare "
        "this across personas directly. fwd_new_loans_repayment_ratio_median is coverage on NEW "
        "borrowing only, undefined (NaN, not 0) for borrowers who didn't take a new loan in the "
        "window. A short fwd_window_days means this is an early, directional read, not a final "
        "verdict -- re-run as more forward data accumulates before revising PERSONA_NAMES from "
        "provisional to validated."
    )


if __name__ == "__main__":
    main()

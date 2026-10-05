"""
compare_persona_severity_diagnostic.py
========================================
Descriptive-only diagnostic, not a validation of a borrower-level severity
model (Axis 2 did not earn that on the evidence from
test_axis2_pd_interaction.py). The question here is narrower and different:

    "What is the segmentation detecting that produces such different
    forward severity outcomes between these personas?"

NOT: "does this validate Axis 2?" -- it can't; a persona summarizes a
population, and a population-level difference (e.g. C4's 44-day median
aging vs. C6's 0-day) does not mean persona membership predicts any one
borrower's outcome. That requires borrower-level predictive validation,
which the Axis-2 work already tried and did not demonstrate.

Covers four personas across the diagnostic spectrum (not just the C4/C6
contrast), in this fixed order so any ordinal gradient is visible at a
glance:
    Elite Quality, Few Loans  ->  Mainstream Modest Activity
    ->  Churning Activity, Latent Risk  ->  Churning Activity, Severe Tail Risk

Two parts:

1. Severity-specific measures, computed fresh here (median, IQR,
   standardized diff vs. the full active population): cal_pd, the four
   borrower_history cure-history candidates, forward closed-loan bad
   rate, forward severe-tail rate (>= --severe-tail-threshold days,
   population-wide -- NOT conditioned on distress, since this is a
   portfolio-level descriptive comparison, not the Axis-2 predictive
   test), and median worst-days-aging.

2. Every OTHER clustering feature (capacity, utilization, momentum,
   tenure, loan frequency, etc.), pulled directly from
   k8_cluster_profile.csv -- profile_persona_k8.py's own already-computed,
   trusted per-persona standardized profile. Deliberately NOT
   rebuilt from the mart file here: that would duplicate
   build_features()'s logic and risk silently drifting from the frozen
   clustering features. This just filters and reorders what already
   exists.

Usage:
    python scripts\\compare_persona_severity_diagnostic.py --forward-outcomes-file data\\persona_k8_forward_outcomes.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from segmentation.borrower_persona_clustering import digits  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ASSIGNMENTS_PATH = REPO / "segmentation_outputs" / "persona_k8_profile" / "k8_cluster_assignments.csv"
CLUSTER_PROFILE_PATH = REPO / "segmentation_outputs" / "persona_k8_profile" / "k8_cluster_profile.csv"
BORROWER_HISTORY_PATH = REPO / "borrower_history_retail_filtered.csv"
ENGINE_OUTPUT_PATH = REPO / "output" / "engine_test_output.csv"
OUT_DIR = REPO / "segmentation_outputs" / "persona_k8_profile"

SPECTRUM_PERSONAS = [
    "Elite Quality, Few Loans",
    "Mainstream Modest Activity",
    "Churning Activity, Latent Risk",
    "Churning Activity, Severe Tail Risk",
]

SEVERITY_CANDIDATES = [
    "lifetime_cure_time_volatility", "cure_time_trend",
    "recent_5_default_24h_rate", "lifetime_avg_hours_to_principal_cure",
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assignments-file", type=Path, default=ASSIGNMENTS_PATH, help=f"default: {ASSIGNMENTS_PATH}")
    p.add_argument("--cluster-profile-file", type=Path, default=CLUSTER_PROFILE_PATH, help=f"default: {CLUSTER_PROFILE_PATH}")
    p.add_argument("--borrower-history-file", type=Path, default=BORROWER_HISTORY_PATH, help=f"default: {BORROWER_HISTORY_PATH}")
    p.add_argument("--engine-output-file", type=Path, default=ENGINE_OUTPUT_PATH, help=f"default: {ENGINE_OUTPUT_PATH}")
    p.add_argument("--forward-outcomes-file", type=Path, required=True,
                   help="CSV exported from data/persona_k8_forward_outcomes_query.sql")
    p.add_argument("--personas", nargs="+", default=SPECTRUM_PERSONAS, help=f"default: {SPECTRUM_PERSONAS}")
    p.add_argument("--severe-tail-threshold", type=float, default=30.0, help="days aging; default: 30")
    args = p.parse_args(argv)

    print("=== Load + merge (assignments, borrower_history, cal_pd, forward outcomes) ===")
    assignments = pd.read_csv(args.assignments_file)
    assignments["_id"] = digits(assignments["phonenumber"])

    bh_available = set(pd.read_csv(args.borrower_history_file, nrows=0).columns)
    bh_cols = [c for c in SEVERITY_CANDIDATES if c in bh_available]
    bh = pd.read_csv(args.borrower_history_file, usecols=["phonenumber"] + bh_cols)
    bh["_id"] = digits(bh["phonenumber"])
    bh = bh.drop(columns=["phonenumber"])

    eng = pd.read_csv(args.engine_output_file, usecols=lambda c: c in {"msisdn", "cal_pd"})
    eng["_id"] = digits(eng["msisdn"])
    eng = eng.drop(columns=["msisdn"])

    fwd = pd.read_csv(args.forward_outcomes_file)
    fwd["_id"] = digits(fwd["customer_msisdn"])
    fwd_cols = [c for c in ["fwd_worst_days_aging", "fwd_new_loans_closed_good_count",
                             "fwd_new_loans_closed_bad_count"] if c in fwd.columns]

    merged = (assignments.merge(bh, on="_id", how="left")
                          .merge(eng, on="_id", how="left")
                          .merge(fwd[["_id"] + fwd_cols], on="_id", how="left"))
    print(f"  {len(merged):,} assigned borrowers total (population baseline for standardized diffs)")

    severity_cols = bh_cols + ["cal_pd"]
    pop_median = merged[severity_cols].median()
    pop_mean = merged[severity_cols].mean()
    pop_std = merged[severity_cols].std(ddof=0).replace(0, np.nan)

    print("\n=== Part 1: severity-specific measures by persona (fixed spectrum order) ===")
    rows = []
    for persona in args.personas:
        sub = merged[merged["persona_name"] == persona]
        if len(sub) == 0:
            print(f"  WARNING: no borrowers found for persona_name == {persona!r} -- check spelling.")
            continue
        good = sub.get("fwd_new_loans_closed_good_count", pd.Series(dtype=float)).sum()
        bad = sub.get("fwd_new_loans_closed_bad_count", pd.Series(dtype=float)).sum()
        aging = sub.get("fwd_worst_days_aging", pd.Series(dtype=float))
        row = {
            "persona_name": persona, "n": len(sub),
            "fwd_closed_loan_bad_rate_pct": round(bad / (good + bad) * 100, 2) if (good + bad) > 0 else None,
            "fwd_closed_loan_n": int(good + bad),
            "fwd_severe_tail_rate_pct": round((aging >= args.severe_tail_threshold).mean() * 100, 2) if aging.notna().any() else None,
            "fwd_severe_tail_n_with_aging_data": int(aging.notna().sum()),
            "fwd_worst_days_aging_median": aging.median(),
        }
        for col in severity_cols:
            s = sub[col]
            row[f"{col}_median"] = s.median()
            row[f"{col}_p25"] = s.quantile(0.25)
            row[f"{col}_p75"] = s.quantile(0.75)
            row[f"{col}_standardized_diff"] = (
                (s.mean() - pop_mean[col]) / pop_std[col] if pd.notna(pop_std[col]) else None
            )
        rows.append(row)
    part1 = pd.DataFrame(rows)
    out1 = OUT_DIR / "persona_severity_diagnostic_part1.csv"
    part1.to_csv(out1, index=False)
    print(f"  wrote {out1}")
    # Print in readable blocks rather than one wide table.
    id_cols = ["persona_name", "n", "fwd_closed_loan_bad_rate_pct", "fwd_closed_loan_n",
               "fwd_severe_tail_rate_pct", "fwd_severe_tail_n_with_aging_data", "fwd_worst_days_aging_median"]
    print(part1[id_cols].to_string(index=False))
    for col in severity_cols:
        cols = ["persona_name"] + [f"{col}_{suffix}" for suffix in ("median", "p25", "p75", "standardized_diff")]
        print(f"\n  -- {col} --")
        print(part1[cols].to_string(index=False))

    print("\n=== Part 2: other clustering features, from k8_cluster_profile.csv (already computed, not rebuilt) ===")
    if not args.cluster_profile_file.exists():
        print(f"  WARNING: {args.cluster_profile_file} not found -- skipping Part 2. Run profile_persona_k8.bat first.")
        return
    profile = pd.read_csv(args.cluster_profile_file)
    name_to_cluster = merged[["persona_cluster", "persona_name"]].drop_duplicates().set_index("persona_name")["persona_cluster"]
    target_clusters = [name_to_cluster.get(p) for p in args.personas if p in name_to_cluster.index]
    already_covered = set(severity_cols) | {"commission", "account_balance"}  # commission/account_balance already seen in earlier outcome_summary work; not excluded here, just not double-labeled as "new"
    other = profile[profile["persona_cluster"].isin(target_clusters)]
    pivot_median = other.pivot(index="feature", columns="persona_cluster", values="cluster_median")
    pivot_sd = other.pivot(index="feature", columns="persona_cluster", values="standardized_diff")
    cluster_to_name = {v: k for k, v in name_to_cluster.items()}
    order = [c for c in target_clusters if c in pivot_median.columns]
    pivot_median = pivot_median[order].rename(columns=cluster_to_name)
    pivot_sd = pivot_sd[order].rename(columns=cluster_to_name)
    out2a = OUT_DIR / "persona_severity_diagnostic_part2_median.csv"
    out2b = OUT_DIR / "persona_severity_diagnostic_part2_standardized_diff.csv"
    pivot_median.to_csv(out2a)
    pivot_sd.to_csv(out2b)
    print(f"  wrote {out2a} and {out2b}")
    print("\n  -- median, by feature x persona --")
    print(pivot_median.to_string())
    print("\n  -- standardized diff (SD from population mean), by feature x persona --")
    print(pivot_sd.to_string())

    print(
        "\nReading this: scan Part 2's standardized-diff table for features where the four personas, in "
        "this fixed left-to-right order, show a roughly ORDERED gradient (e.g. steadily more negative, or "
        "steadily more positive) rather than a scattered pattern. An ordered gradient across several "
        "features at once is what 'the segmentation captures a real combination of behaviors, even without "
        "a single predictive severity feature' would look like -- consistent with Part 1's severity "
        "measures also being ordered (or not) across the same four personas."
    )


if __name__ == "__main__":
    main()

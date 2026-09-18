"""
For agent/months in variable_disbursement_agents.csv (built by
build_monthly_disbursement_summary.py -- agents whose disbursed amount
changed across transactions within the month, e.g. 100,000 / 50,000 /
250,000), checks their MAX disbursed amount against:

  1. the engine's own assigned_limit (cal_pd-based, per-transaction cap), and
  2. the fixed, business-rule agent-category ceiling table below -- the same
     flat amounts already encoded in
     extrafloat/engine/extrafloat_limit_engine_caps.py's
     DEFAULT_CAP_CONFIG["agent_tier"]["commission_thresholds"].

The category table is restated explicitly here (rather than imported) so
this check is an independent cross-check against the engine's own config,
not a tautology that would trivially agree with it.

Usage:
    python scripts\\check_variable_agents_vs_category_limit.py ^
        --variable-file variable_disbursement_agents.csv ^
        --engine-output output\\engine_test_output.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORY_LIMITS = {
    "new bronze": 50_000,
    "bronze": 100_000,
    "silver": 250_000,
    "gold": 350_000,
    "platinum": 500_000,
    "titanium": 750_000,
    "diamond": 1_000_000,
}

# Same bucket edges/labels as check_monthly_summary_vs_assigned_limit.py, so
# the two populations (consistent vs. variable agents) can be compared
# directly on how closely their disbursed amount tracks assigned_limit.
BUCKET_EDGES = [-0.01, 0.25, 0.50, 0.75, 0.90, 1.10, 1.25, 1.50, 2.00, 100]
BUCKET_LABELS = [
    "0-25% (well under)", "25-50% (under)", "50-75% (under)",
    "75-90% (near, under)", "90-110% (AGREEMENT band)", "110-125% (near, over)",
    "125-150% (over)", "150-200% (well over)", "200%+ (far over)",
]
AGREEMENT_EDGES = [-0.01, 0.50, 0.90, 1.10, 100]
AGREEMENT_LABELS = [
    "Well under limit (0-50%)", "Moderately under (50-90%)",
    "Agreement band (90-110%)", "Over limit (110%+)",
]


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def _max_amount(distinct_amounts: str) -> float:
    parts = [p.strip().replace(",", "") for p in str(distinct_amounts).split(";")]
    values = [float(p) for p in parts if p]
    return max(values) if values else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variable-file", default="variable_disbursement_agents.csv")
    ap.add_argument("--engine-output", default="output/engine_test_output.csv")
    ap.add_argument("--out", default="variable_agents_vs_category_limit.csv")
    args = ap.parse_args()

    var_path = Path(args.variable_file)
    eng_path = Path(args.engine_output)
    if not var_path.exists():
        sys.exit(f"ERROR: variable-agents file not found: {var_path}")
    if not eng_path.exists():
        sys.exit(f"ERROR: engine output file not found: {eng_path}")

    var = pd.read_csv(var_path)
    if "distinct_amounts" not in var.columns:
        sys.exit(f"ERROR: {var_path} has no 'distinct_amounts' column -- re-run "
                  f"build_monthly_disbursement_summary.py first.")

    eng = pd.read_csv(eng_path)
    eng_msisdn_col = "msisdn" if "msisdn" in eng.columns else ("agent_msisdn" if "agent_msisdn" in eng.columns else None)
    if eng_msisdn_col is None or "agent_category" not in eng.columns:
        sys.exit(f"ERROR: {eng_path} needs a msisdn/agent_msisdn column and agent_category. "
                  f"Columns present: {list(eng.columns)}")

    var["max_disbursed_amount"] = var["distinct_amounts"].apply(_max_amount)
    var["_key"] = _normalize_msisdn(var["msisdn"])
    eng["_key"] = _normalize_msisdn(eng[eng_msisdn_col])

    eng_cols = ["_key", "agent_category", "assigned_limit"]
    for extra in ["cal_pd", "risk_tier", "pd_decile"]:
        if extra in eng.columns:
            eng_cols.append(extra)
    eng_small = eng[eng_cols].drop_duplicates(subset="_key")

    merged = var.merge(eng_small, on="_key", how="left")
    n = len(merged)
    n_matched = merged["agent_category"].notna().sum()
    print(f"Variable agent/months: {n:,}")
    print(f"Matched to an agent_category in {eng_path.name}: {n_matched:,} / {n:,} ({n_matched / n:.1%})\n")

    scored = merged[merged["agent_category"].notna()].copy()
    scored["_category_key"] = scored["agent_category"].astype(str).str.strip().str.lower()
    scored["category_limit"] = scored["_category_key"].map(CATEGORY_LIMITS)

    n_unmapped = int(scored["category_limit"].isna().sum())
    if n_unmapped:
        unmapped_values = sorted(scored.loc[scored["category_limit"].isna(), "agent_category"].unique())
        print(f"NOTE: {n_unmapped:,} agent/month(s) have an agent_category not in the fixed table "
              f"(likely 'Below Threshold'): {unmapped_values}\n")

    known = scored[scored["category_limit"].notna()].copy()
    known["exceeds_category_limit"] = known["max_disbursed_amount"] > known["category_limit"]
    known["excess_over_category_limit"] = (known["max_disbursed_amount"] - known["category_limit"]).clip(lower=0)
    known["exceeds_assigned_limit"] = known["max_disbursed_amount"] > known["assigned_limit"]
    safe_assigned = known["assigned_limit"].astype(float).replace(0, np.nan)
    known["pct_of_assigned_limit"] = known["max_disbursed_amount"] / safe_assigned
    known["pct_of_assigned_limit_band"] = pd.cut(known["pct_of_assigned_limit"], bins=BUCKET_EDGES, labels=BUCKET_LABELS)
    known["vs_assigned_limit_band"] = pd.cut(known["pct_of_assigned_limit"], bins=AGREEMENT_EDGES, labels=AGREEMENT_LABELS)

    n_exceed_cat = int(known["exceeds_category_limit"].sum())
    n_exceed_assigned = int(known["exceeds_assigned_limit"].sum())
    print("=" * 78)
    print("Max disbursed amount vs. fixed agent-category ceiling")
    print("=" * 78)
    print(f"Exceeds category limit: {n_exceed_cat:,} / {len(known):,} ({n_exceed_cat / len(known):.1%})")
    print(f"Exceeds engine's assigned_limit (for comparison): {n_exceed_assigned:,} / {len(known):,} "
          f"({n_exceed_assigned / len(known):.1%})")

    print(f"\n{'=' * 78}")
    print("How closely max_disbursed_amount tracks assigned_limit (max of several draws)")
    print("=" * 78)
    counts = known["pct_of_assigned_limit_band"].value_counts().reindex(BUCKET_LABELS)
    pct = (counts / len(known) * 100).round(1)
    print(pd.DataFrame({"n": counts, "pct": pct}).to_string())
    print("\nCoarser vs_assigned_limit_band grouping:")
    agr_counts = known["vs_assigned_limit_band"].value_counts().reindex(AGREEMENT_LABELS)
    print(pd.DataFrame({"n": agr_counts, "pct": (agr_counts / len(known) * 100).round(1)}).to_string())
    print(
        "\nCompare this AGREEMENT band share against the same band's share in "
        "monthly_summary_vs_assigned_limit.csv (the consistent-agent population) -- "
        "if variable agents cluster near 100% distinctly tighter, that's evidence their "
        "max disbursement genuinely tracks assigned_limit; a similar spread suggests the "
        "earlier 'they tend to agree' impression was just a few salient rows."
    )

    print(f"\n{'=' * 78}")
    print("Breakdown by agent_category")
    print("=" * 78)
    g = known.groupby("agent_category", observed=True)
    tbl = pd.DataFrame({
        "n_agent_months": g.size(),
        "category_limit": g["category_limit"].first(),
        "violation_rate_vs_category": g["exceeds_category_limit"].mean(),
        "median_excess_when_violating": known[known["exceeds_category_limit"]].groupby("agent_category", observed=True)["excess_over_category_limit"].median(),
    })
    print(tbl.round(4).to_string())

    out_cols = ["msisdn", "month", "distinct_profiles", "agent_category", "category_limit",
                "max_disbursed_amount", "exceeds_category_limit", "excess_over_category_limit",
                "assigned_limit", "exceeds_assigned_limit", "pct_of_assigned_limit",
                "pct_of_assigned_limit_band", "vs_assigned_limit_band", "dates_received",
                "distinct_amounts", "n_transactions"]
    out_cols = [c for c in out_cols if c in known.columns]
    known[out_cols].sort_values("excess_over_category_limit", ascending=False).to_csv(args.out, index=False)
    print(f"\nPer-agent-month detail written: {args.out}")


if __name__ == "__main__":
    main()

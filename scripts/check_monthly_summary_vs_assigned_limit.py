"""
Checks whether the PD engine's assigned_limit agrees with what was actually
disbursed, using monthly_disbursement_summary.csv (built by
build_monthly_disbursement_summary.py) -- i.e. only agent/months where the
disbursed amount was constant across all of that agent's transactions, so
"disbursed_amount" is unambiguous.

Usage:
    python scripts\\check_monthly_summary_vs_assigned_limit.py ^
        --summary-file monthly_disbursement_summary.csv ^
        --engine-output output\\engine_test_output.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

# Same fixed, business-rule agent-category ceiling table as
# check_variable_agents_vs_category_limit.py -- restated independently here
# (rather than imported) as a cross-check against the engine's own config in
# extrafloat_limit_engine_caps.py's DEFAULT_CAP_CONFIG["agent_tier"].
CATEGORY_LIMITS = {
    "new bronze": 50_000,
    "bronze": 100_000,
    "silver": 250_000,
    "gold": 350_000,
    "platinum": 500_000,
    "titanium": 750_000,
    "diamond": 1_000_000,
}


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-file", default="monthly_disbursement_summary.csv")
    ap.add_argument("--engine-output", default="output/engine_test_output.csv")
    ap.add_argument("--out", default="monthly_summary_vs_assigned_limit.csv")
    args = ap.parse_args()

    summary_path = Path(args.summary_file)
    eng_path = Path(args.engine_output)
    if not summary_path.exists():
        sys.exit(f"ERROR: summary file not found: {summary_path}")
    if not eng_path.exists():
        sys.exit(f"ERROR: engine output file not found: {eng_path}")

    summary = pd.read_csv(summary_path)
    eng = pd.read_csv(eng_path)

    eng_msisdn_col = "msisdn" if "msisdn" in eng.columns else ("agent_msisdn" if "agent_msisdn" in eng.columns else None)
    if eng_msisdn_col is None or "assigned_limit" not in eng.columns:
        sys.exit(f"ERROR: {eng_path} needs a msisdn/agent_msisdn column and assigned_limit. "
                  f"Columns present: {list(eng.columns)}")

    summary["_key"] = _normalize_msisdn(summary["msisdn"])
    eng["_key"] = _normalize_msisdn(eng[eng_msisdn_col])

    eng_cols = ["_key", "assigned_limit"]
    for extra in ["cal_pd", "risk_tier", "pd_decile", "agent_category"]:
        if extra in eng.columns:
            eng_cols.append(extra)
    eng_small = eng[eng_cols].drop_duplicates(subset="_key")

    merged = summary.merge(eng_small, on="_key", how="left")
    n = len(merged)
    n_matched = merged["assigned_limit"].notna().sum()
    print(f"Agent/months in {summary_path.name}: {n:,}")
    print(f"Matched to an assigned_limit in {eng_path.name}: {n_matched:,} / {n:,} ({n_matched / n:.1%})")
    n_unmatched = n - n_matched
    if n_unmatched:
        print(f"NOTE: {n_unmatched:,} agent/month(s) have no matching agent in the engine output -- "
              f"never scored/limited by this model run.\n")

    scored = merged[merged["assigned_limit"].notna()].copy()
    scored["exceeds_limit"] = scored["disbursed_amount"] > scored["assigned_limit"]
    scored["excess_amount"] = (scored["disbursed_amount"] - scored["assigned_limit"]).clip(lower=0)
    safe_limit = scored["assigned_limit"].astype(float).replace(0, np.nan)
    scored["pct_of_limit"] = scored["disbursed_amount"] / safe_limit

    n_exceed = int(scored["exceeds_limit"].sum())
    n_zero_limit_disbursed = int(((scored["assigned_limit"] == 0) & (scored["disbursed_amount"] > 0)).sum())

    print("=" * 78)
    print("Agent/month disbursed_amount vs. assigned_limit")
    print("=" * 78)
    print(f"Agent/months exceeding assigned_limit: {n_exceed:,} / {len(scored):,} ({n_exceed / len(scored):.1%})")
    print(f"  of which, assigned_limit == 0 but still disbursed: {n_zero_limit_disbursed:,}")
    if "exceeds_category_limit" in scored.columns:
        known_cat = scored[scored["category_limit"].notna()]
        n_exceed_cat = int(known_cat["exceeds_category_limit"].sum())
        print(f"Agent/months exceeding the fixed agent-category ceiling: {n_exceed_cat:,} / {len(known_cat):,} "
              f"({n_exceed_cat / len(known_cat):.1%})")

    if "agent_category" in scored.columns:
        scored["category_limit"] = scored["agent_category"].astype(str).str.strip().str.lower().map(CATEGORY_LIMITS)
        scored["exceeds_category_limit"] = scored["disbursed_amount"] > scored["category_limit"]
        n_unmapped_cat = int(scored["category_limit"].isna().sum())
        if n_unmapped_cat:
            unmapped_values = sorted(scored.loc[scored["category_limit"].isna(), "agent_category"].unique())
            print(f"NOTE: {n_unmapped_cat:,} agent/month(s) have an agent_category not in the fixed "
                  f"table (likely 'Below Threshold'): {unmapped_values}")

    scored["pct_of_limit_band"] = pd.cut(scored["pct_of_limit"], bins=BUCKET_EDGES, labels=BUCKET_LABELS)
    scored["vs_limit_band"] = pd.cut(scored["pct_of_limit"], bins=AGREEMENT_EDGES, labels=AGREEMENT_LABELS)
    scored["severe_breach_25pct_plus"] = scored["pct_of_limit"] >= 1.25
    scored["severe_breach_2x_plus"] = scored["pct_of_limit"] >= 2.00

    counts = scored["pct_of_limit_band"].value_counts().reindex(BUCKET_LABELS)
    pct = (counts / len(scored) * 100).round(1)
    print(f"\nFull distribution of disbursed_amount as % of assigned_limit:")
    print(pd.DataFrame({"n": counts, "pct": pct}).to_string())

    print(f"\nCoarser vs_limit_band grouping:")
    agr_counts = scored["vs_limit_band"].value_counts().reindex(AGREEMENT_LABELS)
    print(pd.DataFrame({"n": agr_counts, "pct": (agr_counts / len(scored) * 100).round(1)}).to_string())

    n_25 = int(scored["severe_breach_25pct_plus"].sum())
    n_2x = int(scored["severe_breach_2x_plus"].sum())
    print(f"\nSevere breaches: {n_25:,} at 25%+ over limit ({n_25 / n_exceed:.1%} of all violations), "
          f"{n_2x:,} at 2x+ over limit ({n_2x / n_exceed:.1%} of all violations)")
    for label, col in [("25%+ breach", "severe_breach_25pct_plus"), ("2x+ breach", "severe_breach_2x_plus")]:
        subset = scored[scored[col]]
        if len(subset) and "risk_tier" in subset.columns:
            dist = subset["risk_tier"].value_counts(normalize=True).sort_index()
            print(f"  {label} by risk_tier: " + ", ".join(f"{t}={p:.1%}" for t, p in dist.items()))

    if "risk_tier" in scored.columns:
        print(f"\n{'=' * 78}")
        print("Violation rate by risk_tier")
        print("=" * 78)
        g = scored.groupby("risk_tier", observed=True)
        tbl = pd.DataFrame({
            "n_agent_months": g.size(),
            "violation_rate": g["exceeds_limit"].mean(),
            "median_pct_of_limit": g["pct_of_limit"].median(),
        }).sort_index()
        print(tbl.round(4).to_string())

    out_cols = ["msisdn", "profile", "month", "disbursed_amount", "n_transactions", "assigned_limit",
                "exceeds_limit", "excess_amount", "pct_of_limit", "pct_of_limit_band", "vs_limit_band",
                "severe_breach_25pct_plus", "severe_breach_2x_plus"] + \
        [c for c in ["cal_pd", "risk_tier", "pd_decile", "agent_category", "category_limit",
                      "exceeds_category_limit"] if c in scored.columns]
    scored[out_cols].sort_values("excess_amount", ascending=False).to_csv(args.out, index=False)
    print(f"\nPer-agent-month detail written: {args.out}")


if __name__ == "__main__":
    main()

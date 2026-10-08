"""
check_ceiling_binding_by_category.py
=======================================
Direct test of the ceiling-clip hypothesis (not inferred from correlation):
for one agent_category, splits agents into over-limit ("Diamond A"-style,
actual exposure > 110% of combined_cap) vs. control (within 110%), and
reports what fraction of each group is ceiling-bound
(combined_cap ~= capacity_effective_ceiling) plus the median suppression
(capacity_structural - combined_cap -- how much structural capacity the
tier ceiling is holding back).

If the over-limit group is disproportionately ceiling-bound AND shows
larger median suppression than the control group, that is direct
evidence for the ceiling hypothesis -- stronger than the correlation
finding from analyze_capacity_dimension_redundancy.py, which only
established that combined_cap and capacity_structural/capacity_effective_ceiling
move together less tightly in RELATIVE terms than in absolute terms
(consistent with clipping, but not proof of it).

Agents with no actual_exposure_ugx (no disbursement match) cannot be
classified into either group and are excluded, reported separately.

Usage:
    python scripts\\check_ceiling_binding_by_category.py ^
        --research-dataset capacity_research_dataset.csv ^
        --agent-category diamond
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

OVER_LIMIT_THRESHOLD = 1.10  # same convention as every other script this session
CEILING_BINDING_TOLERANCE = 0.02  # within 2% of the effective ceiling counts as "bound"

REQUIRED_COLS = [
    "agent_category", "combined_cap", "capacity_structural", "capacity_effective_ceiling",
    "exposure_intensity_vs_combined_cap",
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--research-dataset", default="capacity_research_dataset.csv")
    ap.add_argument("--agent-category", default="diamond",
                     help="agent_category to analyze (case-insensitive)")
    ap.add_argument("--over-limit-threshold", type=float, default=OVER_LIMIT_THRESHOLD)
    ap.add_argument("--ceiling-binding-tolerance", type=float, default=CEILING_BINDING_TOLERANCE,
                     help="combined_cap >= effective_ceiling * (1 - tolerance) counts as ceiling-bound")
    ap.add_argument("--out", default="ceiling_binding_by_category.csv")
    args = ap.parse_args(argv)

    path = Path(args.research_dataset)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_capacity_research_dataset.py first.")

    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")

    cat_mask = df["agent_category"].astype(str).str.strip().str.lower() == args.agent_category.strip().lower()
    cat_df = df[cat_mask].copy()
    print(f"{args.agent_category!r} agents: {len(cat_df):,} / {len(df):,} total.\n")
    if cat_df.empty:
        sys.exit(f"ERROR: no agents found with agent_category == {args.agent_category!r}. "
                  f"Values present: {sorted(df['agent_category'].dropna().astype(str).unique())}")

    n_unclassifiable = int(cat_df["exposure_intensity_vs_combined_cap"].isna().sum())
    classifiable = cat_df[cat_df["exposure_intensity_vs_combined_cap"].notna()].copy()
    print(f"Excluded {n_unclassifiable:,} agent(s) with no actual_exposure_ugx match -- cannot "
          f"classify over-limit vs. control without it. {len(classifiable):,} remain.\n")

    classifiable["_ceiling_bound"] = classifiable["combined_cap"] >= (
        classifiable["capacity_effective_ceiling"] * (1 - args.ceiling_binding_tolerance)
    )
    classifiable["_suppression"] = classifiable["capacity_structural"] - classifiable["combined_cap"]

    over_limit = classifiable[classifiable["exposure_intensity_vs_combined_cap"] > args.over_limit_threshold]
    control = classifiable[classifiable["exposure_intensity_vs_combined_cap"] <= args.over_limit_threshold]

    rows = []
    for label, sub in [(f"{args.agent_category} over-limit (>{args.over_limit_threshold * 100:.0f}% of combined_cap)", over_limit),
                        (f"{args.agent_category} control (<={args.over_limit_threshold * 100:.0f}% of combined_cap)", control)]:
        n = len(sub)
        rows.append({
            "population": label,
            "n_agents": n,
            "pct_ceiling_bound": round(sub["_ceiling_bound"].mean() * 100, 1) if n else float("nan"),
            "median_structural_cap": sub["capacity_structural"].median() if n else float("nan"),
            "median_effective_ceiling": sub["capacity_effective_ceiling"].median() if n else float("nan"),
            "median_combined_cap": sub["combined_cap"].median() if n else float("nan"),
            "median_suppression": sub["_suppression"].median() if n else float("nan"),
        })
    result = pd.DataFrame(rows).set_index("population")

    print("=" * 100)
    print("Ceiling-binding comparison")
    print("=" * 100)
    with pd.option_context("display.float_format", "{:,.1f}".format, "display.max_columns", None, "display.width", 200):
        print(result.to_string())

    result.reset_index().to_csv(args.out, index=False)
    print(f"\nWritten: {args.out}")

    print(f"\n{'=' * 100}")
    print("What this does and does not establish")
    print("=" * 100)
    if len(over_limit) and len(control):
        ceiling_gap = result.iloc[0]["pct_ceiling_bound"] - result.iloc[1]["pct_ceiling_bound"]
        suppression_gap = result.iloc[0]["median_suppression"] - result.iloc[1]["median_suppression"]
        if ceiling_gap > 0 and suppression_gap > 0:
            print(f"The over-limit group is {ceiling_gap:.1f} percentage points more ceiling-bound and has "
                  f"{suppression_gap:,.0f} UGX higher median suppression than the control group -- DIRECT, "
                  f"descriptive evidence consistent with the ceiling hypothesis (not a correlation inference).")
        else:
            print(f"The over-limit group is NOT more ceiling-bound and/or does not show higher suppression than "
                  f"the control group here -- the ceiling hypothesis is NOT supported by this comparison for "
                  f"this category.")
    print("This is a descriptive comparison of two observational groups, not a controlled experiment -- the "
          "over-limit and control groups may differ on other dimensions too (this is the same identification "
          "caveat as every over-limit comparison this session: observed exposure is not randomly assigned).")


if __name__ == "__main__":
    main()

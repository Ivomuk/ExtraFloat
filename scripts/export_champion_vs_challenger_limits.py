"""
export_champion_vs_challenger_limits.py
==========================================
The decision-MECHANICS table (not an outcome validation): for one logged
scoring cycle in output/shadow_multiplier_log.csv, shows every agent's
pre-policy capacity base and what each policy actually does to it --

    Agent        Current base   Live multiplier   Live limit   C3 multiplier   C3 shadow limit
    <msisdn>     combined_cap   live_tier_mult.   assigned_lim  shadow_mult_base  shadow_limit_post_transition_base

i.e. literally combined_cap x live_tier_multiplier = assigned_limit, and
combined_cap x shadow_multiplier_base = (pre-transition) shadow limit,
side by side, for every agent -- not just a cohort-level summary. This
is purely mechanical: what each policy ASSIGNS. It says nothing about
subsequent loan performance -- see evaluate_champion_vs_challenger_decision.py
for the outcome-validation comparison (cohort-level, joined against
forward loan outcomes).

combined_cap ("Current base") needs the pipeline run with
--keep-intermediate (see run_retail_filtered.bat/.sh) -- without it,
this table still runs but "Current base" is blank for every row, with
a loud warning rather than a silent gap.

Usage:
    python scripts\\export_champion_vs_challenger_limits.py ^
        --shadow-log output\\shadow_multiplier_log.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

AGENT_CATEGORY_ORDER = ["diamond", "titanium", "platinum", "gold", "silver", "bronze", "new bronze"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shadow-log", default="output/shadow_multiplier_log.csv")
    ap.add_argument("--run-id", default=None,
                     help="which logged cycle (run_id = scored_at) to export; defaults to the most "
                          "recent run_id present in --shadow-log")
    ap.add_argument("--out", default="champion_vs_challenger_limits.csv")
    args = ap.parse_args(argv)

    log_path = Path(args.shadow_log)
    if not log_path.exists():
        sys.exit(f"ERROR: {log_path} not found -- run log_shadow_multiplier_cycle.py first.")

    log = pd.read_csv(log_path, dtype={"run_id": str, "msisdn": str})
    required = ["msisdn", "live_tier_multiplier", "assigned_limit",
                "shadow_multiplier_base", "shadow_limit_post_transition_base", "shadow_status"]
    missing_required = [c for c in required if c not in log.columns]
    if missing_required:
        sys.exit(f"ERROR: {log_path} is missing required column(s): {missing_required}")

    run_id = args.run_id or log["run_id"].max()
    cycle = log[log["run_id"] == run_id].copy()
    if cycle.empty:
        sys.exit(f"ERROR: run_id {run_id!r} not found in {log_path}. "
                  f"Available run_ids: {sorted(log['run_id'].unique())}")
    cycle_date = cycle["cycle_date"].iloc[0] if "cycle_date" in cycle.columns else "(unknown)"
    print(f"Exporting logged cycle: run_id={run_id}  cycle_date={cycle_date}  n_agents={len(cycle):,}")

    if "combined_cap" not in cycle.columns or cycle["combined_cap"].isna().all():
        print("WARNING: combined_cap unavailable for this cycle -- 'Current base' will be blank for "
              "every row. Re-run the pipeline with --keep-intermediate (already added to "
              "run_retail_filtered.bat/.sh), then re-log this cycle, to populate it.")
        cycle["combined_cap"] = pd.NA

    n_shadow_failed = int((cycle["shadow_status"] != "ok").sum())
    if n_shadow_failed:
        print(f"NOTE: {n_shadow_failed:,} agent(s) have shadow_status != 'ok' -- their C3 columns "
              f"will be blank below (shadow could not be computed for them this cycle).")

    out = pd.DataFrame({
        "Agent": cycle["msisdn"],
        "Agent category": cycle["agent_category"] if "agent_category" in cycle.columns else pd.NA,
        "Current base": cycle["combined_cap"],
        "Live multiplier": cycle["live_tier_multiplier"],
        "Live limit": cycle["assigned_limit"],
        "C3 multiplier": cycle["shadow_multiplier_base"].where(cycle["shadow_status"] == "ok"),
        "C3 shadow limit": cycle["shadow_limit_post_transition_base"].where(cycle["shadow_status"] == "ok"),
    })
    if "pct_change_base" in cycle.columns:
        out["Pct change (C3 vs live)"] = (cycle["pct_change_base"] * 100).round(1)
    if "cohort_base" in cycle.columns:
        out["Cohort"] = cycle["cohort_base"]

    if "Agent category" in out.columns:
        cat_rank = out["Agent category"].str.lower().map(
            {cat: i for i, cat in enumerate(AGENT_CATEGORY_ORDER)}
        )
        out = out.assign(_cat_rank=cat_rank).sort_values(
            ["_cat_rank", "Pct change (C3 vs live)"], ascending=[True, False], na_position="last"
        ).drop(columns=["_cat_rank"])

    out.to_csv(args.out, index=False)
    print(f"\nPer-agent decision table written: {args.out}  ({len(out):,} rows)")

    if "Agent category" in out.columns:
        print(f"\n{'=' * 70}")
        print("Row counts by agent category")
        print("=" * 70)
        print(out["Agent category"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

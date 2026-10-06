"""
check_live_shadow_vs_category_limit.py
=========================================
Checks the engine's live assigned_limit AND the shadow continuous
multiplier's candidate limits (both scenarios) against the fixed,
business-rule agent-category ceiling table -- the same flat amounts
already encoded in extrafloat_limit_engine_caps.py's
DEFAULT_CAP_CONFIG["agent_tier"]["commission_thresholds"].

The category table is restated explicitly here (rather than imported),
matching the same convention as check_monthly_summary_vs_assigned_limit.py
and check_variable_agents_vs_category_limit.py -- so this is an
independent cross-check against the engine's own config, not a tautology
that would trivially agree with it.

Upper-bound only, by design (confirmed, not assumed): there is no
business-rule LOWER bound on an agent's limit within their category --
combine_caps()'s min() across capacity/recent-usage/prior-exposure/risk
caps can and does legitimately push a high-commission agent's limit well
below their category ceiling; that's normal, not a violation. The only
invariant worth checking is that neither the live nor the shadow limit
ever EXCEEDS the ceiling for the agent's commission-based category.

Optionally enriches the per-agent detail with monthly_disbursement_summary.csv
(built by build_monthly_disbursement_summary.py from the raw fin_log
disbursements export) -- i.e. what the agent ACTUALLY received, per
month, for agent/months where the disbursed amount was constant across
all of that agent's transactions. Joined on normalized msisdn; an agent
with disbursement records in more than one month appears once per month
(the live/shadow/category columns are a single snapshot and simply
repeat across that agent's rows). Skipped gracefully (with a NOTE, not
an error) if the file isn't found -- the category/live/shadow check
above works standalone either way.

Usage:
    python scripts\\check_live_shadow_vs_category_limit.py ^
        --engine-output output\\engine_test_output.csv ^
        --monthly-summary-file monthly_disbursement_summary.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)

CATEGORY_LIMITS = {
    "new bronze": 50_000,
    "bronze": 100_000,
    "silver": 250_000,
    "gold": 350_000,
    "platinum": 500_000,
    "titanium": 750_000,
    "diamond": 1_000_000,
}

# name -> engine-output column. "live" always applies to every scored row;
# the shadow columns only apply to rows where shadow_status == "ok".
LIMIT_COLUMNS = {
    "live": "assigned_limit",
    "shadow_base": "shadow_limit_post_transition_base",
    "shadow_conservative": "shadow_limit_post_transition_conservative",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-output", default="output/engine_test_output.csv")
    ap.add_argument("--monthly-summary-file", default="monthly_disbursement_summary.csv",
                     help="optional -- built by build_monthly_disbursement_summary.py; skipped with a "
                          "NOTE if not found")
    ap.add_argument("--out", default="live_shadow_vs_category_limit.csv")
    args = ap.parse_args(argv)

    eng_path = Path(args.engine_output)
    if not eng_path.exists():
        sys.exit(f"ERROR: engine output file not found: {eng_path}")

    eng = pd.read_csv(eng_path)
    required = {"agent_category", "assigned_limit"}
    missing = required - set(eng.columns)
    if missing:
        sys.exit(f"ERROR: {eng_path} is missing required columns {missing}. "
                  f"Columns present: {list(eng.columns)}")

    present = {name: col for name, col in LIMIT_COLUMNS.items() if col in eng.columns}
    absent = set(LIMIT_COLUMNS) - set(present)
    if absent:
        print(f"NOTE: columns for {sorted(absent)} not found in {eng_path.name} -- was this engine run "
              f"scored with the shadow multiplier active (artifacts_dir pointing at the shadow "
              f"calibration)? Those checks will be skipped.\n")

    df = eng.copy()
    df["_category_key"] = df["agent_category"].astype(str).str.strip().str.lower()
    df["category_limit"] = df["_category_key"].map(CATEGORY_LIMITS)

    n_total = len(df)
    n_unmapped = int(df["category_limit"].isna().sum())
    if n_unmapped:
        unmapped_values = sorted(df.loc[df["category_limit"].isna(), "agent_category"].astype(str).unique())
        print(f"NOTE: {n_unmapped:,}/{n_total:,} agent(s) have an agent_category not in the fixed table "
              f"(likely 'Below Threshold' or unscored): {unmapped_values}\n")

    known = df[df["category_limit"].notna()].copy()
    print(f"Agents with a known category ceiling: {len(known):,} / {n_total:,}\n")

    summary_rows = []
    for name, col in present.items():
        subset = known.dropna(subset=[col])
        if name != "live" and "shadow_status" in subset.columns:
            subset = subset[subset["shadow_status"] == "ok"]

        exceeds_col = f"exceeds_category_limit_{name}"
        excess_col = f"excess_over_category_limit_{name}"
        known[exceeds_col] = pd.NA
        known[excess_col] = pd.NA
        known.loc[subset.index, exceeds_col] = subset[col] > subset["category_limit"]
        known.loc[subset.index, excess_col] = (subset[col] - subset["category_limit"]).clip(lower=0)

        n_checked = len(subset)
        n_exceed = int((subset[col] > subset["category_limit"]).sum())
        summary_rows.append({
            "limit_source": name,
            "n_checked": n_checked,
            "n_exceeds_category_ceiling": n_exceed,
            "pct_exceeds_category_ceiling": round(n_exceed / n_checked * 100, 4) if n_checked else float("nan"),
        })

    print("=" * 78)
    print("Live vs. shadow limits -- violations of the fixed agent-category ceiling")
    print("=" * 78)
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print(f"\n{'=' * 78}")
    print("Breakdown by agent_category")
    print("=" * 78)
    for name in present:
        exceeds_col = f"exceeds_category_limit_{name}"
        checked = known.dropna(subset=[exceeds_col]).copy()
        if checked.empty:
            continue
        checked[exceeds_col] = checked[exceeds_col].astype(bool)
        g = checked.groupby("agent_category", observed=True)[exceeds_col]
        tbl = g.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "violation_rate"})
        print(f"\n-- {name} --")
        print(tbl.round(4).to_string())

    # -- Optional enrichment: what the agent ACTUALLY received -----------------
    monthly_path = Path(args.monthly_summary_file)
    disbursement_cols: list[str] = []
    reference_cols = {"category_limit": "category_limit", **present}
    if not monthly_path.exists():
        print(f"\nNOTE: {monthly_path} not found -- skipping disbursement enrichment "
              f"(pass --monthly-summary-file, or run build_monthly_disbursement_summary.py first).")
    else:
        monthly = pd.read_csv(monthly_path)
        required_monthly = {"msisdn", "profile", "month", "disbursed_amount", "n_transactions"}
        missing_monthly = required_monthly - set(monthly.columns)
        if missing_monthly:
            print(f"\nNOTE: {monthly_path} is missing {missing_monthly} -- skipping disbursement "
                  f"enrichment. Columns present: {list(monthly.columns)}")
        else:
            monthly["_key"] = _normalize_msisdn(monthly["msisdn"])
            known["_key"] = _normalize_msisdn(known["msisdn"])
            n_before = len(known)
            known = known.merge(
                monthly[["_key", "profile", "month", "disbursed_amount", "n_transactions"]],
                on="_key", how="left",
            )
            n_after = len(known)
            n_with_disbursement = int(known["disbursed_amount"].notna().sum())
            print(f"\nMerged {monthly_path.name}: {n_with_disbursement:,} agent-month row(s) matched a "
                  f"disbursement record ({n_after - n_before:+,} rows vs. before this merge -- an agent "
                  f"with records in more than one month now appears once per month, with the snapshot's "
                  f"live/shadow/category columns repeated across their rows).")
            disbursement_cols = ["profile", "month", "disbursed_amount", "n_transactions"]

            # Did the agent's ACTUAL disbursement fit under each reference ceiling/limit?
            for name, ref_col in reference_cols.items():
                exceeds_col = f"exceeds_{name}_disbursed"
                excess_col = f"excess_over_{name}_disbursed"
                valid = known["disbursed_amount"].notna() & known[ref_col].notna()
                if name != "category_limit" and name != "live" and "shadow_status" in known.columns:
                    valid &= known["shadow_status"] == "ok"
                known[exceeds_col] = pd.NA
                known[excess_col] = pd.NA
                known.loc[valid, exceeds_col] = known.loc[valid, "disbursed_amount"] > known.loc[valid, ref_col]
                known.loc[valid, excess_col] = (
                    known.loc[valid, "disbursed_amount"] - known.loc[valid, ref_col]
                ).clip(lower=0)
                disbursement_cols += [exceeds_col, excess_col]

            disb_summary_rows = []
            for name in reference_cols:
                exceeds_col = f"exceeds_{name}_disbursed"
                checked = known.dropna(subset=[exceeds_col])
                n_checked = len(checked)
                n_exceed = int(checked[exceeds_col].astype(bool).sum()) if n_checked else 0
                disb_summary_rows.append({
                    "reference": name,
                    "n_checked": n_checked,
                    "n_exceeds": n_exceed,
                    "pct_exceeds": round(n_exceed / n_checked * 100, 4) if n_checked else float("nan"),
                })
            print(f"\n{'=' * 78}")
            print("Actual disbursed_amount vs. category ceiling / live limit / shadow limits")
            print("=" * 78)
            print(pd.DataFrame(disb_summary_rows).to_string(index=False))

    detail_cols = ["msisdn", "agent_category", "category_limit"]
    for name, col in present.items():
        detail_cols += [col, f"exceeds_category_limit_{name}", f"excess_over_category_limit_{name}"]
    detail_cols += disbursement_cols
    detail_cols += [c for c in ["risk_tier", "cal_pd", "shadow_status"] if c in known.columns]
    detail_cols = [c for c in dict.fromkeys(detail_cols) if c in known.columns]

    sort_col = next(
        (f"excess_over_category_limit_{n}" for n in present if f"excess_over_category_limit_{n}" in known.columns),
        None,
    )
    out_df = known[detail_cols]
    if sort_col:
        out_df = out_df.sort_values(sort_col, ascending=False, na_position="last")
    out_df.to_csv(args.out, index=False)
    print(f"\nPer-agent detail written: {args.out}")


if __name__ == "__main__":
    main()

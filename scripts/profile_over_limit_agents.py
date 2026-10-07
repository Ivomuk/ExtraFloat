"""
profile_over_limit_agents.py
=========================================
Investigates WHO the "Over limit (110%+)" population is, using only
columns already present in the per-agent detail CSV written by
check_live_shadow_vs_category_limit.py (live_shadow_vs_category_limit.csv)
-- no new source file required.

"Over limit" here means the same thing it has throughout this analysis:
disbursed_amount > 110% of the chosen reference (default: the live
model's assigned_limit). An agent-month row counts as "checked" if it
has a non-null pct_of_<reference>_disbursed value at all.

This does NOT repeat the risk_tier / K=8 persona breakdown already
printed by check_live_shadow_vs_category_limit.py -- it adds four more
angles on the same population, each using a column already in the file:

  1. agent_category    -- is the gap concentrated in particular commission tiers?
  2. cal_pd (risk band) -- are over-limit agents actually low-risk, confirming
                           the "model under-recommends for good agents" story?
  3. n_transactions     -- does transaction volume/frequency predict the gap?
  4. profile            -- the raw disbursement-instruction profile label
                           (e.g. "MTNU Agent Silver Commission") -- a finer-
                           grained cut than agent_category, top-N by volume.

Also prints a plain business-impact number: the total, mean, and median
excess shillings disbursed beyond the reference, for the over-limit
group only -- "how much extra money is this gap worth," not just a rate.

Usage:
    python scripts\\profile_over_limit_agents.py ^
        --live-shadow-file live_shadow_vs_category_limit.csv ^
        --reference live
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

OVER_LIMIT_THRESHOLD = 1.10  # same convention as every other check_*.py script in this repo

CAL_PD_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 1.0]
CAL_PD_LABELS = ["<2%", "2-5%", "5-10%", "10-20%", "20-30%", "30%+"]

N_TXN_EDGES = [0, 1, 2, 3, 5, 10, 1_000_000]
N_TXN_LABELS = ["1", "2", "3", "4-5", "6-10", "11+"]

TOP_N_PROFILES = 15


def _group_breakdown(df: pd.DataFrame, group_col: str, over_col: str) -> pd.DataFrame:
    g = df.groupby(group_col, observed=True)[over_col]
    tbl = g.agg(["size", "sum"]).rename(columns={"size": "n_checked", "sum": "n_over"})
    tbl["pct_over"] = (tbl["n_over"] / tbl["n_checked"] * 100).round(1)
    n_over_total = int(df[over_col].sum())
    tbl["share_of_all_over"] = (tbl["n_over"] / n_over_total * 100).round(1) if n_over_total else 0.0
    return tbl.sort_values("n_over", ascending=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-shadow-file", default="live_shadow_vs_category_limit.csv",
                     help="per-agent detail CSV written by check_live_shadow_vs_category_limit.py")
    ap.add_argument("--reference", default="live", choices=["live", "shadow_base", "shadow_conservative"])
    ap.add_argument("--out", default="over_limit_agent_profile.csv")
    args = ap.parse_args(argv)

    ls_path = Path(args.live_shadow_file)
    if not ls_path.exists():
        sys.exit(f"ERROR: {ls_path} not found (run check_live_shadow_vs_category_limit.py first).")

    df = pd.read_csv(ls_path)
    pct_col = f"pct_of_{args.reference}_disbursed"
    excess_col = f"excess_over_{args.reference}_disbursed"
    if pct_col not in df.columns:
        sys.exit(f"ERROR: {pct_col} not found in {ls_path}. Columns present: {list(df.columns)}")

    df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")
    checked = df[df[pct_col].notna()].copy()
    checked["_over"] = checked[pct_col] > OVER_LIMIT_THRESHOLD
    n_checked = len(checked)
    n_over = int(checked["_over"].sum())

    print("=" * 78)
    print(f"Profiling the Over-limit (>110% of {args.reference}) population")
    print("=" * 78)
    print(f"Checked agent-months: {n_checked:,} | Over limit: {n_over:,} ({n_over / n_checked * 100:.1f}%)\n")

    # -- Business-impact number: how much extra money is this gap actually worth? --
    if excess_col in checked.columns:
        checked[excess_col] = pd.to_numeric(checked[excess_col], errors="coerce")
        over_rows = checked[checked["_over"]]
        total_excess = over_rows[excess_col].sum()
        print(f"{'=' * 78}")
        print("Business impact: excess disbursed beyond the reference (over-limit rows only)")
        print("=" * 78)
        print(f"Total excess:  {total_excess:,.0f}")
        print(f"Mean excess:   {over_rows[excess_col].mean():,.0f}")
        print(f"Median excess: {over_rows[excess_col].median():,.0f}\n")

    # -- Are over-limit agents actually LOW risk? Direct check of the "model --
    # -- under-recommends for good agents" hypothesis, independent of persona. --
    if "cal_pd" in checked.columns:
        checked["cal_pd"] = pd.to_numeric(checked["cal_pd"], errors="coerce")
        print(f"{'=' * 78}")
        print("cal_pd (model risk score): over-limit group vs. everyone else checked")
        print("=" * 78)
        cmp = checked.groupby("_over")["cal_pd"].agg(["count", "mean", "median"]).rename(
            index={False: "not over limit", True: "over limit"}
        )
        print(cmp.round(4).to_string())
        print()

        checked["_cal_pd_band"] = pd.cut(checked["cal_pd"], bins=CAL_PD_EDGES, labels=CAL_PD_LABELS)
        print(f"{'=' * 78}")
        print("Breakdown by cal_pd band")
        print("=" * 78)
        print(_group_breakdown(checked, "_cal_pd_band", "_over").to_string())
        print()

    if "agent_category" in checked.columns:
        print(f"{'=' * 78}")
        print("Breakdown by agent_category")
        print("=" * 78)
        print(_group_breakdown(checked, "agent_category", "_over").to_string())
        print()

    if "n_transactions" in checked.columns:
        checked["n_transactions"] = pd.to_numeric(checked["n_transactions"], errors="coerce")
        checked["_n_txn_band"] = pd.cut(checked["n_transactions"], bins=N_TXN_EDGES, labels=N_TXN_LABELS)
        print(f"{'=' * 78}")
        print("Breakdown by n_transactions (disbursement count that month)")
        print("=" * 78)
        print(_group_breakdown(checked, "_n_txn_band", "_over").to_string())
        print()

    if "profile" in checked.columns:
        top_profiles = checked["profile"].value_counts().head(TOP_N_PROFILES).index
        subset = checked[checked["profile"].isin(top_profiles)]
        print(f"{'=' * 78}")
        print(f"Breakdown by profile (top {TOP_N_PROFILES} by volume; "
              f"{len(checked) - len(subset):,} row(s) in smaller profiles omitted)")
        print("=" * 78)
        print(_group_breakdown(subset, "profile", "_over").to_string())
        print()

    # -- Write a single tidy CSV with every breakdown stacked, for pivoting/charting. --
    # (profile uses the FULL population here, not just the printed top-N by volume.)
    out_rows = []
    for dim_label, group_col in [
        ("agent_category", "agent_category"),
        ("cal_pd_band", "_cal_pd_band"),
        ("n_transactions_band", "_n_txn_band"),
        ("profile", "profile"),
    ]:
        if group_col not in checked.columns:
            continue
        tbl = _group_breakdown(checked, group_col, "_over").reset_index()
        tbl = tbl.rename(columns={group_col: "group_value"})
        tbl.insert(0, "dimension", dim_label)
        out_rows.append(tbl)
    if out_rows:
        pd.concat(out_rows, ignore_index=True).to_csv(args.out, index=False)
        print(f"Breakdown tables written: {args.out}")


if __name__ == "__main__":
    main()

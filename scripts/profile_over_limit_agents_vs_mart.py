"""
profile_over_limit_agents_vs_mart.py
=========================================
Profiles the Over-limit (110%+) population -- and its two distinct
drivers found by profile_over_limit_agents.py's agent_category x
cal_pd_band crosstab -- against the FULL engineered feature set the
real segmentation pipeline builds from the raw KPI agent mart (e.g.
data\\mfs_daily_agent_mart_20260731.csv).

Uses the real production feature engineering
(segmentation.extrafloat_segmentation_features.prepare_features) --
not a reimplementation -- so this sees the same tenure, balance,
cash-in/out, voucher/payment, and network-density features the
segmentation model itself uses, not just the handful of columns already
in live_shadow_vs_category_limit.csv.

Three groups, matching the crosstab finding (agent_category x cal_pd_band):
  - driver_a: ever over-limit, agent_category == diamond, cal_pd BELOW
    --high-risk-cal-pd (default 0.30) -- the "genuinely low/moderate-
    risk, high-volume, under-recommended" population (~49% of all
    over-limit cases).
  - driver_b: ever over-limit, cal_pd AT/ABOVE --high-risk-cal-pd, any
    tier -- the "model's own riskiest agents going over anyway"
    population (~51% of all over-limit cases).
  - all_over: everyone ever over-limit, for a single combined view.

Each group is profiled against the "not over limit" agents from the
SAME scored file (live_shadow_vs_category_limit.csv) -- not the raw
mart's full population -- so the baseline is drawn from exactly the
population that was actually checked, not an unrelated, unscored
superset of agents sitting in the raw mart.

For each engineered feature, prints (group mean / baseline mean) and a
z-score (how many baseline standard deviations apart the two means
are), sorted by |z-score| so the columns that actually distinguish each
group surface automatically -- without having to scan every column by
hand. Same method as scripts/characterize_flagged_agents.py.

Usage:
    python scripts\\profile_over_limit_agents_vs_mart.py ^
        --agents-file data\\mfs_daily_agent_mart_20260731.csv ^
        --live-shadow-file live_shadow_vs_category_limit.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.extrafloat_segmentation_features import prepare_features  # noqa: E402

OVER_LIMIT_THRESHOLD = 1.10
TOP_N = 25


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def _profile_subset(
    label: str,
    subset_mask: pd.Series,
    baseline_mask: pd.Series,
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame | None:
    n = int(subset_mask.sum())
    n_baseline = int(baseline_mask.sum())
    print(f"\n=== {label}: {n:,} agents (baseline: {n_baseline:,} not-over-limit agents) ===")
    if n == 0 or n_baseline == 0:
        print("  (empty group -- nothing to profile)")
        return None

    rows = []
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        baseline_vals = vals[baseline_mask]
        subset_vals = vals[subset_mask]
        baseline_mean = baseline_vals.mean()
        baseline_std = baseline_vals.std()
        subset_mean = subset_vals.mean()
        ratio = (subset_mean / baseline_mean) if baseline_mean != 0 else np.nan
        zscore = ((subset_mean - baseline_mean) / baseline_std) if baseline_std > 0 else np.nan
        rows.append({
            "group": label, "column": col,
            "subset_mean": subset_mean, "baseline_mean": baseline_mean,
            "ratio": ratio, "zscore": zscore,
        })
    profile = pd.DataFrame(rows).sort_values("zscore", key=lambda s: s.abs(), ascending=False)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(profile.drop(columns="group").head(TOP_N).to_string(index=False))
    return profile


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents-file", required=True, metavar="PATH",
                     help="raw KPI agent mart, e.g. data\\mfs_daily_agent_mart_20260731.csv")
    ap.add_argument("--live-shadow-file", default="live_shadow_vs_category_limit.csv",
                     help="per-agent detail CSV written by check_live_shadow_vs_category_limit.py")
    ap.add_argument("--reference", default="live", choices=["live", "shadow_base", "shadow_conservative"])
    ap.add_argument("--high-risk-cal-pd", type=float, default=0.30,
                     help="cal_pd threshold separating driver_a (below) from driver_b (at/above); "
                          "default 0.30 matches the '30%%+' band boundary used in "
                          "profile_over_limit_agents.py's CAL_PD_EDGES.")
    ap.add_argument("--out", default="over_limit_vs_mart_profile.csv")
    args = ap.parse_args(argv)

    ls_path = Path(args.live_shadow_file)
    agents_path = Path(args.agents_file)
    if not ls_path.exists():
        sys.exit(f"ERROR: {ls_path} not found (run check_live_shadow_vs_category_limit.py first).")
    if not agents_path.exists():
        sys.exit(f"ERROR: {agents_path} not found.")

    # -- Collapse the (possibly multi-month) live/shadow detail to one row --
    # -- per agent: "ever over limit" = ANY matched month exceeded the threshold. --
    ls = pd.read_csv(ls_path)
    pct_col = f"pct_of_{args.reference}_disbursed"
    if pct_col not in ls.columns:
        sys.exit(f"ERROR: {pct_col} not found in {ls_path}. Columns present: {list(ls.columns)}")
    ls["_key"] = _normalize_msisdn(ls["msisdn"])
    ls[pct_col] = pd.to_numeric(ls[pct_col], errors="coerce")
    if "cal_pd" in ls.columns:
        ls["cal_pd"] = pd.to_numeric(ls["cal_pd"], errors="coerce")

    summary_rows = []
    for key, grp in ls.groupby("_key", dropna=True):
        vals = grp[pct_col].dropna()
        if vals.empty:
            continue
        summary_rows.append({
            "_key": key,
            "ever_over": bool((vals > OVER_LIMIT_THRESHOLD).any()),
            "cal_pd": grp["cal_pd"].dropna().iloc[0] if "cal_pd" in grp.columns and grp["cal_pd"].notna().any() else np.nan,
            "agent_category": grp["agent_category"].dropna().iloc[0]
            if "agent_category" in grp.columns and grp["agent_category"].notna().any() else None,
        })
    summary = pd.DataFrame(summary_rows)
    n_checked_agents = len(summary)
    print(f"Agents with a checkable {args.reference} disbursement ratio: {n_checked_agents:,}")

    threshold = args.high_risk_cal_pd
    is_diamond = summary["agent_category"].astype(str).str.strip().str.lower() == "diamond"
    summary["driver_a"] = summary["ever_over"] & is_diamond & (summary["cal_pd"] < threshold)
    summary["driver_b"] = summary["ever_over"] & (summary["cal_pd"] >= threshold)
    summary["all_over"] = summary["ever_over"]
    summary["not_over"] = ~summary["ever_over"]

    # -- Real production feature engineering on the raw mart -- not a reimplementation. --
    agents_df = pd.read_csv(agents_path)
    print(f"Loaded {len(agents_df):,} agent rows from {agents_path}")
    features_df, _, _, selected_cols = prepare_features(agents_df)
    key_col = "agent_msisdn" if "agent_msisdn" in features_df.columns else "pos_msisdn"
    features_df = features_df.copy()
    features_df["_key"] = _normalize_msisdn(features_df[key_col])

    merged = features_df.merge(summary, on="_key", how="inner")
    print(f"Matched {len(merged):,} / {n_checked_agents:,} checkable agents to a row in {agents_path.name} "
          f"({len(merged) / n_checked_agents * 100:.1f}%).")
    if merged.empty:
        sys.exit("ERROR: no agents matched between the live/shadow file and the agent mart -- "
                  "check that --agents-file covers the same population (msisdn format, snapshot date).")

    cols = [c for c in selected_cols if c in merged.columns]
    print(f"Profiling {len(cols)} engineered feature(s) from prepare_features.\n")

    baseline_mask = merged["not_over"]
    profiles = []
    for label, group_col in [
        ("driver_a (diamond, below high-risk cal_pd)", "driver_a"),
        ("driver_b (high-risk cal_pd, any tier)", "driver_b"),
        ("all_over (everyone ever over limit)", "all_over"),
    ]:
        print("=" * 78)
        print(label)
        print("=" * 78)
        profile = _profile_subset(label, merged[group_col], baseline_mask, merged, cols)
        if profile is not None:
            profiles.append(profile)

    if profiles:
        pd.concat(profiles, ignore_index=True).to_csv(args.out, index=False)
        print(f"\nFull feature profile (all groups, all columns) written: {args.out}")


if __name__ == "__main__":
    main()

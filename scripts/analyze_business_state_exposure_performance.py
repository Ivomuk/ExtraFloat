"""
analyze_business_state_exposure_performance.py
==================================================
Level 1 (primary) evidence of the business-state-period evidence hierarchy
(cadence-gate pivot; formerly Deliverable 3, "analyze_episode_exposure_
scale_overlap.py" -- renamed and reframed, logic unchanged, after real-data
cadence diagnostics showed loan-to-loan deltas were the wrong estimand for
Level 2/3; this analysis never depended on that estimand and remains
correct as-is). Answers the identification question that must be settled
before any exposure-response claim can be trusted: were comparable measured
business states (by float activity / commission decile) actually observed
across MULTIPLE exposure tiers, or is exposure tier almost deterministically
assigned by scale? If high-scale agents almost exclusively received 750K/1M
and small agents almost exclusively received 50K/100K, then
BusinessState x Exposure -> Performance cannot reliably separate the two
effects in much of the feature space -- no amount of correct temporal
alignment or sample size fixes that. This is treated as a genuine go/no-go
input to the rest of this workstream, not mere descriptive color.

Level hierarchy this script anchors (see the plan's "Cadence-gate pivot"
section for the full three-level design): Level 1 compares comparable
measured business states at different exposures (cross-sectional, this
script); Level 2 (`analyze_business_state_exposure_variation.py`) tightens
to the same agent + same measured business-state anchor; Level 3
(`analyze_business_state_evolution.py`) adds genuine temporal movement in
the agent's measured business state. None of the three establish a causal
effect of increasing a limit -- progressively tighter observational
evidence, not a causal estimate.

TWO SEPARATE TABLES, never blended (same assignment/performance separation
this session has used throughout):
  - Exposure-ASSIGNMENT overlap: every episode with a valid fundamentals
    match, regardless of label_eligible_30d. Answers "were comparable
    measured business states historically observed at multiple exposure
    tiers at all."
  - Exposure-PERFORMANCE overlap: the same cross-tab restricted to
    label_eligible_30d==1, with bad rate added. Answers the (narrower,
    outcome-conditioned) question.

Run under THREE snapshot-freshness cuts (all / <=30 days / <=60 days -- the
same cuts used throughout this rebuild): if overlap or conclusions
disappear once restricted to recent snapshots, that is itself a reportable
finding, not something to quietly drop.

Restated independently (one-way scripts/ layering convention): the 7-tier
exposure set, consistent with every other script in this rebuild.

Usage:
    python scripts\\analyze_business_state_exposure_performance.py ^
        --episode-dataset loan_episode_capacity_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPOSURE_TIERS_UGX = [50_000, 100_000, 250_000, 350_000, 500_000, 750_000, 1_000_000]
SCALE_VARS = {"float_activity_value_1m": "Float activity", "commission": "Earnings (commission)"}
AGE_CUTS = {"all": None, "<=30d": 30, "<=60d": 60}
N_DECILES = 10
MIN_CELL_N = 10


def _qcut_safe(s: pd.Series, q: int) -> tuple:
    """qcut with duplicate bin edges dropped; if there's too little data/
    variation to form q bins at all, falls back to as many bins as the
    data supports (minimum 1 -- a single bucket covering everyone), rather
    than silently dropping every row the way a bare pd.qcut would (n_bins==0
    produces an all-NaN code column, which pd.crosstab then excludes
    entirely). Returns (labeled band series, n_bins actually produced)."""
    try:
        codes, bins = pd.qcut(s, q, duplicates="drop", retbins=True, labels=False)
    except ValueError:
        codes, bins = None, None
    n_bins = (len(bins) - 1) if bins is not None else 0
    if n_bins <= 0:
        return pd.Series("D1", index=s.index), 1
    labels = [f"D{i + 1}" for i in range(n_bins)]
    return codes.map(dict(enumerate(labels))), n_bins


def _apply_age_cut(df: pd.DataFrame, max_age_days) -> pd.DataFrame:
    if max_age_days is None:
        return df
    return df[df["fundamentals_age_days"].notna() & (df["fundamentals_age_days"] <= max_age_days)]


def _decile_assigned_population(df: pd.DataFrame, scale_col: str, max_age_days) -> tuple:
    """Shared decile assignment, computed ONCE on every episode with a
    valid fundamentals match (label_eligible_30d irrelevant at this stage)
    -- so the assignment and performance tables use the IDENTICAL decile
    boundaries. Deciles are deliberately NOT recomputed on the smaller
    eligible-only subset: doing so would let the same agent land in a
    different-numbered decile in each table purely because the population
    defining the quantile cut points changed, undermining the "same
    cross-tab, two populations" comparison this script exists to make."""
    working = df[df[scale_col].notna() & (df[scale_col] > 0) & df["disbursement_amount_ugx"].notna()].copy()
    working = _apply_age_cut(working, max_age_days)
    if working.empty:
        return working, []
    working["_decile"], n_bins = _qcut_safe(working[scale_col], N_DECILES)
    labels = [f"D{i + 1}" for i in range(n_bins)]
    return working, labels


def assignment_overlap_table(df: pd.DataFrame, scale_col: str, max_age_days) -> pd.DataFrame:
    """ALL episodes with a valid fundamentals match (label_eligible_30d
    irrelevant). Decile of scale_col x exposure-tier count cross-tab."""
    working, labels = _decile_assigned_population(df, scale_col, max_age_days)
    if working.empty:
        return pd.DataFrame()
    cross = pd.crosstab(working["_decile"], working["disbursement_amount_ugx"])
    cross = cross.reindex(index=labels, columns=EXPOSURE_TIERS_UGX, fill_value=0)
    return cross


def performance_overlap_table(df: pd.DataFrame, scale_col: str, max_age_days) -> tuple:
    """Same cross-tab (SAME decile boundaries as assignment_overlap_table --
    see _decile_assigned_population) restricted to label_eligible_30d==1,
    with bad rate added. Returns (count_table, bad_rate_table)."""
    working, labels = _decile_assigned_population(df, scale_col, max_age_days)
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()
    working = working[working["label_eligible_30d"] == 1]
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()
    count = pd.crosstab(working["_decile"], working["disbursement_amount_ugx"])
    count = count.reindex(index=labels, columns=EXPOSURE_TIERS_UGX, fill_value=0)
    bad_rate = working.groupby(["_decile", "disbursement_amount_ugx"], observed=False)["bad_state_3dpd_30d"].mean().unstack()
    bad_rate = bad_rate.reindex(index=labels, columns=EXPOSURE_TIERS_UGX)
    bad_rate = bad_rate.where(count >= MIN_CELL_N)
    return count, bad_rate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", default="loan_episode_capacity_dataset.csv")
    ap.add_argument("--out-prefix", default="episode_exposure_scale_overlap")
    args = ap.parse_args(argv)

    path = Path(args.episode_dataset)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_loan_episode_capacity_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)

    required = ["disbursement_amount_ugx", "label_eligible_30d", "bad_state_3dpd_30d", "fundamentals_age_days"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing_req}")
    missing_scale = [c for c in SCALE_VARS if c not in df.columns]
    if missing_scale:
        print(f"NOTE: scale variable(s) not found, skipped: {missing_scale}")

    n_unknown_exposure = int((~df["disbursement_amount_ugx"].isin(EXPOSURE_TIERS_UGX)).sum())
    pct_unknown_exposure = n_unknown_exposure / len(df) * 100 if len(df) else float("nan")
    print(f"n_unknown_exposure (disbursement_amount_ugx outside the known 7-tier set): "
          f"{n_unknown_exposure:,} ({pct_unknown_exposure:.2f}% of all episodes). These episodes are "
          f"excluded from every tier-indexed table below (reindexed to the 7-tier set) -- reported here "
          f"explicitly, never silently dropped. No unit conversion is assumed or applied.")

    for scale_col, label in SCALE_VARS.items():
        if scale_col not in df.columns:
            continue
        for cut_label, max_age in AGE_CUTS.items():
            print(f"\n{'#' * 100}")
            print(f"# {label} [{scale_col}] -- snapshot freshness: {cut_label}")
            print(f"{'#' * 100}")

            print(f"\n{'=' * 100}\nExposure-ASSIGNMENT overlap (all episodes with valid fundamentals)\n{'=' * 100}")
            assign = assignment_overlap_table(df, scale_col, max_age)
            if assign.empty:
                print("  (no qualifying episodes)")
            else:
                print(assign.to_string())
                assign.to_csv(f"{args.out_prefix}_assignment_{scale_col}_{cut_label.replace('<=', 'le')}.csv")

            print(f"\n{'=' * 100}\nExposure-PERFORMANCE overlap (label_eligible_30d==1 only)\n{'=' * 100}")
            count, bad_rate = performance_overlap_table(df, scale_col, max_age)
            if count.empty:
                print("  (no qualifying episodes)")
            else:
                print("-- N (eligible) --")
                print(count.to_string())
                print("-- Bad rate (NaN where N < {}) --".format(MIN_CELL_N))
                with pd.option_context("display.float_format", "{:.3f}".format):
                    print(bad_rate.to_string())
                count.to_csv(f"{args.out_prefix}_performance_n_{scale_col}_{cut_label.replace('<=', 'le')}.csv")
                bad_rate.to_csv(f"{args.out_prefix}_performance_badrate_{scale_col}_{cut_label.replace('<=', 'le')}.csv")

    print(f"\n{'#' * 100}")
    print("What this does and does not establish")
    print(f"{'#' * 100}")
    print("This is a pure identification diagnostic -- it does not fit anything. A cross-tab dominated\n"
          "by zeros off a narrow diagonal (high-scale deciles only at high tiers, low-scale deciles only\n"
          "at low tiers) means exposure tier is close to deterministically assigned by business scale in\n"
          "this data, and no later model can reliably separate exposure's effect from scale's. If overlap\n"
          "looks healthy under 'all' snapshots but collapses under <=30d/<=60d, that is itself a finding\n"
          "about data sufficiency, not something to average away.")


if __name__ == "__main__":
    main()

"""
Computes the actual blended capacity-score distribution for a candidate
development population, using the SAME primitives calibrate_capacity_scorecard
would use (compute_raw_kpi_frame -> _fit_normalization_params ->
_apply_normalization -> compute_group_scores -> _blend_group_score) -- not a
reimplementation, so the distribution shown here is exactly what
calibrate_scorecard.py's --target-proportions would be cutting quantiles out
of.

Why this exists: calibrate_capacity_scorecard's target_tier_proportions
argument is a business decision fed INTO calibration (it does not read from
any existing segmentation output, since none exists pre-calibration) -- but
that decision should still be made with visibility into the population's
actual score distribution, not picked blind. This script gives that
visibility before anyone runs calibrate_scorecard.py for real.

Reports, for the blended [0,1] capacity score and each of its 3 component
group scores (value/activity/efficiency):
  - percentile table (1/5/10/25/50/75/90/95/99)
  - what tier proportions an EVEN split (1/8 each) would cut at, for reference
  - histogram-style decile bucket counts

Usage:
    python scripts\\check_capacity_score_distribution.py --agents data\\mfs_daily_agent_mart_20260731.csv
    python scripts\\check_capacity_score_distribution.py --agents data\\mfs_daily_agent_mart_20260731.csv --raw
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.extrafloat_segmentation_features import prepare_features  # noqa: E402
from segmentation.extrafloat_segmentation_scoring import (  # noqa: E402
    CAPACITY_FACTOR_GROUPS,
    DEFAULT_GROUP_WEIGHTS,
    DEFAULT_SCORING_CONFIG,
    _apply_normalization,
    _blend_group_score,
    _fit_normalization_params,
    compute_group_scores,
    compute_raw_kpi_frame,
)

PERCENTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def _print_percentile_table(series: pd.Series, label: str) -> None:
    print(f"\n{label} -- percentiles")
    print("-" * 40)
    for q in PERCENTILES:
        print(f"  p{int(q * 100):>2}  {series.quantile(q):.4f}")
    print(f"  mean  {series.mean():.4f}")
    print(f"  std   {series.std():.4f}")


def _print_decile_buckets(series: pd.Series, label: str) -> None:
    print(f"\n{label} -- decile bucket counts (equal-width on [0,1], not equal-count)")
    print("-" * 40)
    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(series.clip(0, 1), bins=edges)
    total = len(series)
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        pct = 100.0 * c / total if total else 0.0
        bar = "#" * int(pct / 2)
        print(f"  [{lo:.1f}, {hi:.1f})  {c:>7}  ({pct:5.1f}%)  {bar}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", required=True, metavar="PATH")
    p.add_argument(
        "--raw", action="store_true",
        help="Treat --agents as already containing the raw KPI columns (skip prepare_features).",
    )
    p.add_argument(
        "--allow-missing-columns", action="store_true",
        help="Zero-fill any scorecard-declared KPI column missing from --agents instead of failing closed.",
    )
    args = p.parse_args(argv)

    agents_df = pd.read_csv(args.agents)
    print(f"Loaded {len(agents_df):,} agent rows from {args.agents}")

    if args.raw:
        development_df = agents_df
    else:
        development_df, _, _, _ = prepare_features(agents_df)

    on_missing = "zero" if args.allow_missing_columns else "raise"
    raw_kpi_frame = compute_raw_kpi_frame(development_df, CAPACITY_FACTOR_GROUPS, on_missing_column=on_missing)

    print("\nRaw KPI columns feeding the scorecard (pre-normalization):")
    print("-" * 60)
    print(raw_kpi_frame.describe(percentiles=PERCENTILES).T[["mean", "std", "min", "50%", "99%", "max"]])

    normalization = _fit_normalization_params(
        raw_kpi_frame, upper_quantile=float(DEFAULT_SCORING_CONFIG["normalization_upper_quantile"])
    )
    print("\nFrozen normalization ranges this population would produce (ref_min, ref_max @ p99):")
    print("-" * 60)
    for col, params in normalization.items():
        print(f"  {col:<28} ref_min={params['ref_min']:.4f}  ref_max={params['ref_max']:.4f}")

    normalized_kpi_frame = _apply_normalization(raw_kpi_frame, normalization)
    group_scores = compute_group_scores(normalized_kpi_frame, CAPACITY_FACTOR_GROUPS)
    blended_score = _blend_group_score(group_scores, DEFAULT_GROUP_WEIGHTS)

    for group_name in CAPACITY_FACTOR_GROUPS:
        _print_percentile_table(group_scores[group_name], f"Group score: {group_name}")

    _print_percentile_table(blended_score, "BLENDED capacity score (value*0.5 + activity*0.3 + efficiency*0.2)")
    _print_decile_buckets(blended_score, "BLENDED capacity score")

    n_tiers = 8
    even_cutoffs = [blended_score.quantile(i / n_tiers) for i in range(1, n_tiers)]
    print(f"\nFor reference -- an EVEN 1/8 split of THIS population would cut at:")
    print("-" * 60)
    tier_names = [
        "Below Threshold", "New Bronze", "Bronze", "Silver",
        "Gold", "Platinum", "Titanium", "Diamond",
    ]
    bounds = [0.0] + even_cutoffs + [1.0]
    for name, lo, hi in zip(tier_names, bounds[:-1], bounds[1:]):
        print(f"  {name:<16} [{lo:.4f}, {hi:.4f})")


if __name__ == "__main__":
    main()

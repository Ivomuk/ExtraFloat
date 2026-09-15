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
  - for each --target-proportions given (may be repeated -- e.g. 3 candidates
    in one run, so they can be compared side by side without re-running):
    the exact cutoffs and achieved agent counts THAT candidate proportion
    set would produce on this population -- the same quantile computation
    calibrate_capacity_scorecard's target_tier_proportions branch does
    internally, run here as a look-before-you-leap preview. Writes nothing
    -- calibrate_scorecard.py is still the only thing that persists a
    scorecard. Give each candidate an optional "_name" key for a readable
    label in the output; otherwise it's labeled "Candidate 1", "Candidate 2", ...

Usage:
    python scripts\\check_capacity_score_distribution.py --agents data\\mfs_daily_agent_mart_20260731.csv
    python scripts\\check_capacity_score_distribution.py --agents data\\mfs_daily_agent_mart_20260731.csv --raw

    REM Compare 3 candidates in one run -- repeat --target-proportions:
    python scripts\\check_capacity_score_distribution.py --agents data\\mfs_daily_agent_mart_20260731.csv ^
        --target-proportions "{\\"_name\\": \\"Even\\", \\"Below Threshold\\": 0.125, \\"New Bronze\\": 0.125, \\"Bronze\\": 0.125, \\"Silver\\": 0.125, \\"Gold\\": 0.125, \\"Platinum\\": 0.125, \\"Titanium\\": 0.125, \\"Diamond\\": 0.125}" ^
        --target-proportions "{\\"_name\\": \\"Mild pyramid\\", \\"Below Threshold\\": 0.15, \\"New Bronze\\": 0.20, \\"Bronze\\": 0.20, \\"Silver\\": 0.15, \\"Gold\\": 0.12, \\"Platinum\\": 0.10, \\"Titanium\\": 0.05, \\"Diamond\\": 0.03}" ^
        --target-proportions "{\\"_name\\": \\"Steep pyramid\\", \\"Below Threshold\\": 0.30, \\"New Bronze\\": 0.25, \\"Bronze\\": 0.15, \\"Silver\\": 0.10, \\"Gold\\": 0.08, \\"Platinum\\": 0.06, \\"Titanium\\": 0.04, \\"Diamond\\": 0.02}"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.extrafloat_segmentation_features import prepare_features  # noqa: E402
from segmentation.extrafloat_segmentation_scoring import (  # noqa: E402
    BUSINESS_SEGMENTS,
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


def _print_cutoff_preview(
    blended_score: pd.Series,
    tiers: list[str],
    proportions: list[float],
    label: str,
) -> None:
    """Print the cutoffs/tier ranges/achieved counts *proportions* would produce.

    Mirrors calibrate_capacity_scorecard's target_tier_proportions branch
    (cumulative quantiles of the blended score) exactly, but only prints --
    never writes a scorecard.
    """
    cumulative = np.cumsum(proportions)[:-1]
    cutoffs = [float(blended_score.quantile(q)) for q in cumulative]
    for i in range(1, len(cutoffs)):
        if cutoffs[i] <= cutoffs[i - 1]:
            cutoffs[i] = cutoffs[i - 1] + 1e-9

    bounds = [0.0] + cutoffs + [1.0]
    total = len(blended_score)
    print(f"\n{label}")
    print("-" * 60)
    for name, target_prop, lo, hi in zip(tiers, proportions, bounds[:-1], bounds[1:]):
        n_agents = int(((blended_score >= lo) & (blended_score < hi)).sum()) if hi < 1.0 else int(
            (blended_score >= lo).sum()
        )
        achieved_pct = 100.0 * n_agents / total if total else 0.0
        print(
            f"  {name:<16} target={target_prop * 100:5.1f}%  "
            f"cutoff=[{lo:.4f}, {hi:.4f})  "
            f"n={n_agents:>7,}  achieved={achieved_pct:5.1f}%"
        )

    # A large block of agents sharing the exact same blended score AT a cutoff
    # value silently breaks the achieved-vs-target match: assign_capacity_tier
    # (and this preview) route score == cutoff to the tier ABOVE the cutoff, so
    # the whole tied block lands in one tier regardless of how the target
    # proportions were split across that boundary. Flag it here instead of
    # leaving it to be spotted by eye in the achieved-percentage column.
    tie_threshold = max(1, int(0.005 * total))
    flagged_values: list[float] = []
    tie_warnings = []
    for cutoff in cutoffs:
        # Degenerate-cutoff nudging (+1e-9) can produce near-duplicate cutoffs
        # sitting on the SAME tie mass -- flag each distinct plateau once.
        if any(abs(cutoff - v) <= 1e-9 for v in flagged_values):
            continue
        n_tied = int(np.isclose(blended_score, cutoff, rtol=0, atol=1e-9).sum())
        if n_tied >= tie_threshold:
            tie_warnings.append((cutoff, n_tied))
            flagged_values.append(cutoff)
    if tie_warnings:
        print(
            "  WARNING: large tie mass sitting exactly at a cutoff -- the whole "
            "tied block goes to the tier ABOVE it, so achieved proportions on "
            "either side of these cutoffs will deviate from target:"
        )
        for cutoff, n_tied in tie_warnings:
            pct = 100.0 * n_tied / total if total else 0.0
            print(f"    cutoff={cutoff:.4f}  {n_tied:,} agents tied exactly here ({pct:.1f}% of population)")


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
    p.add_argument(
        "--target-proportions", metavar="JSON", action="append", default=None,
        help=(
            "JSON object {tier_name: proportion} summing to ~1.0, with an "
            "optional \"_name\" key for a readable label. May be given "
            "multiple times to preview several candidates side by side in "
            "one run. Previews the exact score cutoffs and achieved agent "
            "counts each candidate would produce on --agents -- same "
            "computation calibrate_scorecard.py would do, but writes nothing."
        ),
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

    tiers = list(BUSINESS_SEGMENTS)
    n_tiers = len(tiers)
    even_proportions = [1.0 / n_tiers] * n_tiers
    _print_cutoff_preview(
        blended_score, tiers, even_proportions,
        "For reference -- an EVEN split would cut at:",
    )

    for i, raw_json in enumerate(args.target_proportions or [], start=1):
        try:
            proportions_dict = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            print(f"error: --target-proportions #{i} is not valid JSON -- {exc}", file=sys.stderr)
            sys.exit(1)
        candidate_name = proportions_dict.pop("_name", f"Candidate {i}")
        proportions = [float(proportions_dict.get(t, 0.0)) for t in tiers]
        unknown = [t for t in proportions_dict if t not in tiers]
        if unknown:
            print(
                f"error: --target-proportions #{i} ({candidate_name!r}) has "
                f"unknown tier name(s): {unknown}",
                file=sys.stderr,
            )
            sys.exit(1)
        if abs(sum(proportions) - 1.0) > 1e-3:
            print(
                f"error: --target-proportions #{i} ({candidate_name!r}) must "
                f"sum to 1.0 (got {sum(proportions):.4f})",
                file=sys.stderr,
            )
            sys.exit(1)
        _print_cutoff_preview(
            blended_score, tiers, proportions,
            f"Candidate {i} -- {candidate_name} (preview only -- nothing written):",
        )


if __name__ == "__main__":
    main()

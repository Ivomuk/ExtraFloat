"""
calibrate_scorecard.py
========================
CLI wrapper for the offline, human-governed step that produces a versioned
capacity scorecard (extrafloat_segmentation_scoring.calibrate_capacity_scorecard)
and persists it as a JSON artifact.

This is deliberately a separate step from `run_extrafloat_segmentation.py`:
calibration is a decision a human reviews and signs off on, not something a
production run does implicitly. Point `run_extrafloat_segmentation`'s
`config["scoring"]["scorecard_path"]` at the resulting file once it's
reviewed.

Usage
-----
    # Even split across BUSINESS_SEGMENTS (provisional default)
    python calibrate_scorecard.py --agents development_agents.csv \\
        --out scorecards/capacity_scorecard_v0.json

    # Target specific tier proportions
    python calibrate_scorecard.py --agents development_agents.csv \\
        --out scorecards/capacity_scorecard_v0.json \\
        --target-proportions '{"Below Threshold": 0.15, "New Bronze": 0.15, \\
            "Bronze": 0.15, "Silver": 0.15, "Gold": 0.15, "Platinum": 0.1, \\
            "Titanium": 0.1, "Diamond": 0.05}'

    # Mark as reviewed / final (clears is_provisional)
    python calibrate_scorecard.py --agents development_agents.csv \\
        --out scorecards/capacity_scorecard_v1.json \\
        --cutoff-version reviewed_2026q1 --final
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from extrafloat_segmentation_features import prepare_features
from extrafloat_segmentation_scoring import calibrate_capacity_scorecard, save_scorecard


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python calibrate_scorecard.py",
        description=(
            "Calibrate a versioned, deterministic capacity scorecard from a "
            "development agent population."
        ),
    )
    p.add_argument(
        "--agents", metavar="PATH", required=True,
        help="Path to the development agent KPI CSV (same schema as run_extrafloat_segmentation input).",
    )
    p.add_argument(
        "--out", metavar="PATH", required=True,
        help="Output path for the scorecard JSON.",
    )
    p.add_argument(
        "--raw", action="store_true",
        help=(
            "Treat --agents as already containing the raw KPI columns "
            "(commission, cash_out_value_1m, ...) with no feature "
            "engineering needed. By default the CSV is run through "
            "prepare_features first, matching what run_extrafloat_segmentation "
            "would see."
        ),
    )
    p.add_argument(
        "--target-proportions", metavar="JSON", default=None,
        help='JSON object {tier_name: proportion} summing to 1.0. Omit for an even split.',
    )
    p.add_argument(
        "--cutoff-version", metavar="LABEL", default="provisional_v0",
        help="Version label stored in the scorecard (default: provisional_v0).",
    )
    p.add_argument(
        "--final", action="store_true",
        help="Mark the scorecard as reviewed/final (is_provisional=False). Omit for a provisional scorecard.",
    )
    p.add_argument(
        "--population-description", metavar="TEXT", default="",
        help="Free-text note on the development population (sample size, date range, source).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite --out if it already exists (default: refuse).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    agents_df = pd.read_csv(args.agents)

    if args.raw:
        development_df = agents_df
    else:
        development_df, _, _, _ = prepare_features(agents_df)

    target_proportions = None
    if args.target_proportions:
        try:
            target_proportions = json.loads(args.target_proportions)
        except json.JSONDecodeError as exc:
            print(f"error: --target-proportions is not valid JSON — {exc}", file=sys.stderr)
            sys.exit(1)

    scorecard = calibrate_capacity_scorecard(
        development_df,
        target_tier_proportions=target_proportions,
        cutoff_version=args.cutoff_version,
        is_provisional=not args.final,
        population_description=args.population_description,
    )

    try:
        save_scorecard(scorecard, args.out, overwrite=args.force)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Scorecard written to: {args.out}")
    print(f"  cutoff_version   : {scorecard['cutoff_version']}")
    print(f"  is_provisional   : {scorecard['calibration_metadata']['is_provisional']}")
    print(f"  n_agents         : {scorecard['calibration_metadata']['n_agents']}")
    print("  tier proportions :")
    for tier, prop in scorecard["calibration_metadata"]["expected_tier_proportions"].items():
        print(f"    {tier:<20} {prop:.4f}")


if __name__ == "__main__":
    main()

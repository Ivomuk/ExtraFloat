"""
Computes the actual dormancy composite-score distribution on a candidate
population, using the SAME formula _identify_dormant_mask
(extrafloat_segmentation_scoring.py) uses -- each of the 4 inactivity
columns normalized by its own 95th percentile (clipped to [0,1]), combined
via the configured weights -- so this is a direct look at why 55% of the
retail-filtered population is landing at or below the 0.05 dormancy
threshold, not a guess.

Why this exists: after fixing the voucher_vol_1m naming bug
(voucher_volume_1m didn't exist in the real mart, so it was silently
dropped from the composite every run), the dormancy rate on the real
population actually went UP slightly (54.3% -> 55.0%), meaning that bug
was real but small -- not the driver of the 55% rate. The more likely
explanation is structural: normalizing each column against its OWN p95
means a right-skewed distribution (typical for transaction volumes) can
trivially put most of the population under a 0.05 composite bar just from
the shape of the distribution, independent of whether those agents are
"dormant" in any real business sense. This script gives visibility into
that before anyone decides whether/how to change the threshold or the
normalization approach.

Reports, for each raw inactivity column and for the blended composite
score:
  - percentile table
  - what fraction of the population falls at/below several candidate
    dormancy thresholds (0.01, 0.02, 0.05, 0.10, 0.15, 0.20), so the
    sensitivity of the dormant count to the threshold choice is visible
  - cross-checks against the real _identify_dormant_mask output, so this
    script's own composite calculation is verified to match production
    exactly rather than silently drifting from it

  - "all columns near-zero" cross-check: for a few absolute tolerances
    (0, 1, 2 transactions), the count/share of agents where EVERY one of
    the 4 raw activity columns is simultaneously <= that tolerance -- an
    independent, non-normalized definition of "genuinely inactive" to
    compare against the composite-score-based dormant count. If the
    composite threshold flags far more agents than are actually near-zero
    on every input, that's further evidence the threshold (not just one
    column's weight) needs revisiting.

  - for each --target-dormant-proportion given (may be repeated): the
    composite-score threshold VALUE that would achieve that dormant rate on
    THIS population (composite.quantile(target)), mirroring how
    calibrate_capacity_scorecard derives cutoffs from target_tier_proportions
    -- i.e. computed from the real distribution, not guessed. Prints nothing
    to any file and changes no code default; it's the number you'd plug into
    --dormancy-config '{"dormant_composite_threshold": ...}' once you've
    picked a target.

Usage:
    python scripts\\check_dormancy_composite_distribution.py --agents data\\mfs_daily_agent_mart_20260731_retail_filtered.csv

    REM Compare candidate dormancy-rate targets and see what threshold each implies:
    python scripts\\check_dormancy_composite_distribution.py --agents data\\mfs_daily_agent_mart_20260731_retail_filtered.csv ^
        --target-dormant-proportion 0.05 --target-dormant-proportion 0.10 ^
        --target-dormant-proportion 0.15 --target-dormant-proportion 0.20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.extrafloat_segmentation_scoring import _identify_dormant_mask  # noqa: E402

PERCENTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
CANDIDATE_THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
NEAR_ZERO_TOLERANCES = [0, 1, 2, 5]

DEFAULT_COLS = ["cash_out_vol_1m", "cash_in_vol_1m", "payment_vol_1m", "voucher_vol_1m"]
DEFAULT_WEIGHTS = [0.5, 0.25, 0.15, 0.10]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", required=True, metavar="PATH")
    p.add_argument(
        "--target-dormant-proportion", type=float, action="append", default=None,
        metavar="FLOAT",
        help=(
            "Target dormant rate (e.g. 0.10 for 10%%). May be given multiple "
            "times to compare candidates. For each, prints the composite-"
            "score threshold that achieves it on --agents."
        ),
    )
    args = p.parse_args(argv)

    df = pd.read_csv(args.agents)
    print(f"Loaded {len(df):,} agent rows from {args.agents}\n")

    present = [(c, w) for c, w in zip(DEFAULT_COLS, DEFAULT_WEIGHTS) if c in df.columns]
    missing = [c for c in DEFAULT_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: missing from --agents (excluded, matching production behavior): {missing}\n")
    present_cols, present_weights = zip(*present)
    norm_weights = [w / sum(present_weights) for w in present_weights]

    print("Dormancy inputs and their weights (renormalized over present columns):")
    for col, w in zip(present_cols, norm_weights):
        print(f"  {col:<20} weight={w:.4f}")

    raw_frame = pd.DataFrame(index=df.index)
    normalized_frame = pd.DataFrame(index=df.index)
    composite = pd.Series(0.0, index=df.index)
    degenerate_cols: list[str] = []
    for col, w in zip(present_cols, norm_weights):
        series = df[col].fillna(0.0).clip(lower=0.0)
        raw_frame[col] = series
        p95 = series.quantile(0.95)
        normalized = (series / p95).clip(upper=1.0) if p95 > 0 else pd.Series(0.0, index=series.index)
        normalized_frame[col] = normalized
        composite += w * normalized

        pct_zero = float((series == 0).mean())
        print(f"\n=== Raw {col} -- percentiles ===")
        print(series.describe(percentiles=PERCENTILES).to_string())
        print(f"  {pct_zero:.1%} of agents are exactly 0 on this column")
        if p95 <= 0:
            degenerate_cols.append(col)
            print(
                f"  *** DEAD COLUMN: p95={p95:.4f} -- at least 95% of agents are 0 here, "
                f"so this column contributes EXACTLY ZERO to every agent's composite score "
                f"(the code's own div-by-zero guard), for agents with real activity on it too. "
                f"Its weight ({w:.1%}) is being wasted, capping the maximum achievable composite "
                f"below 1.0. Not a candidate signal for dormancy on this population as configured. ***"
            )
        else:
            print(f"  (p95 used as normalization ceiling: {p95:.4f})")
        print(f"\n=== Normalized {col} (value / own p95, clipped to 1.0) -- percentiles ===")
        for q in PERCENTILES:
            print(f"  p{int(q * 100):>2}  {normalized_frame[col].quantile(q):.4f}")

    if degenerate_cols:
        print(
            f"\n*** SUMMARY: {len(degenerate_cols)} dead/non-meaningful column(s) found: "
            f"{degenerate_cols} -- consider dropping from dormant_inactivity_cols or "
            f"replacing with a column that actually varies on this population. ***"
        )

    print("\n" + "=" * 70)
    print("=== BLENDED dormancy composite score -- percentiles ===")
    print("=" * 70)
    for q in PERCENTILES:
        print(f"  p{int(q * 100):>2}  {composite.quantile(q):.4f}")
    print(f"  mean  {composite.mean():.4f}")

    total = len(composite)
    print("\n=== Share of population at/below candidate dormancy thresholds ===")
    for t in CANDIDATE_THRESHOLDS:
        n = int((composite <= t).sum())
        print(f"  threshold <= {t:.2f}   n={n:>7,}  ({n / total:.1%})")

    print("\n=== Agents where EVERY activity column is near-zero (independent, non-normalized check) ===")
    print(f"(columns checked: {list(present_cols)})")
    for tol in NEAR_ZERO_TOLERANCES:
        all_near_zero = (raw_frame <= tol).all(axis=1)
        n = int(all_near_zero.sum())
        print(f"  all columns <= {tol}   n={n:>7,}  ({n / total:.1%})")
    current_dormant_n = int((composite <= 0.05).sum())
    all_zero_n = int((raw_frame <= 0).all(axis=1).sum())
    print(
        f"\n  For comparison: composite <= 0.05 (current threshold) flags "
        f"{current_dormant_n:,} ({current_dormant_n / total:.1%}), vs. "
        f"{all_zero_n:,} ({all_zero_n / total:.1%}) genuinely all-zero on "
        f"every column -- the gap between these is agents the composite "
        f"calls dormant despite having SOME real activity on at least one input."
    )

    targets = args.target_dormant_proportion or [0.05, 0.10, 0.15, 0.20]
    print("\n=== Threshold VALUE implied by each target dormant proportion ===")
    print("(computed as composite.quantile(target) on THIS population --")
    print(" plug the chosen value into --dormancy-config for calibration)")
    for target in targets:
        threshold_value = float(composite.quantile(target))
        n = int((composite <= threshold_value).sum())
        print(
            f"  target={target:.1%}   threshold={threshold_value:.4f}   "
            f"n={n:>7,}  achieved={n / total:.1%}"
        )

    # Cross-check: this script's own composite calculation must agree
    # exactly with the real production function's boolean output.
    production_mask = _identify_dormant_mask(df, {})
    my_mask = composite <= 0.05
    mismatches = int((production_mask != my_mask).sum())
    print(f"\nCross-check vs. real _identify_dormant_mask (threshold=0.05): "
          f"{'MATCH' if mismatches == 0 else f'{mismatches} MISMATCHES -- something is wrong with this script'}")


if __name__ == "__main__":
    main()

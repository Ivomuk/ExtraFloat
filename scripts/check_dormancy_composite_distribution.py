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

Usage:
    python scripts\\check_dormancy_composite_distribution.py --agents data\\mfs_daily_agent_mart_20260731_retail_filtered.csv
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

DEFAULT_COLS = ["cash_out_vol_1m", "cash_in_vol_1m", "payment_vol_1m", "voucher_vol_1m"]
DEFAULT_WEIGHTS = [0.5, 0.25, 0.15, 0.10]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", required=True, metavar="PATH")
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

    normalized_frame = pd.DataFrame(index=df.index)
    composite = pd.Series(0.0, index=df.index)
    for col, w in zip(present_cols, norm_weights):
        series = df[col].fillna(0.0).clip(lower=0.0)
        p95 = series.quantile(0.95)
        normalized = (series / p95).clip(upper=1.0) if p95 > 0 else pd.Series(0.0, index=series.index)
        normalized_frame[col] = normalized
        composite += w * normalized

        print(f"\n=== Raw {col} -- percentiles ===")
        print(series.describe(percentiles=PERCENTILES).to_string())
        print(f"  (p95 used as normalization ceiling: {p95:.4f})")
        print(f"\n=== Normalized {col} (value / own p95, clipped to 1.0) -- percentiles ===")
        for q in PERCENTILES:
            print(f"  p{int(q * 100):>2}  {normalized_frame[col].quantile(q):.4f}")

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

    # Cross-check: this script's own composite calculation must agree
    # exactly with the real production function's boolean output.
    production_mask = _identify_dormant_mask(df, {})
    my_mask = composite <= 0.05
    mismatches = int((production_mask != my_mask).sum())
    print(f"\nCross-check vs. real _identify_dormant_mask (threshold=0.05): "
          f"{'MATCH' if mismatches == 0 else f'{mismatches} MISMATCHES -- something is wrong with this script'}")


if __name__ == "__main__":
    main()

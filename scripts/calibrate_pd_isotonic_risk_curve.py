"""
calibrate_pd_isotonic_risk_curve.py
====================================
Axis 1 only (probability/frequency) of the two-axis risk architecture:

    cal_pd -> isotonic calibration -> calibrated realized bad rate

Fits a monotonic (isotonic) regression of REALIZED closed-loan bad rate
against cal_pd, weighted by each borrower's closed-loan volume. This is
the objective, data-only step -- it answers "given the data we actually
have, what does a smooth, monotonic realized-risk curve against cal_pd
look like" without imposing a straight line between the engine's current
tier cutoffs, and without treating the raw table's local wobbles (e.g.
the 8.21% -> 7.68% -> 6.90% -> 6.71% dip just above cal_pd=0.05 found in
analyze_risk_tier_pd_resolution.py's band breakdown) as genuine risk
reversals.

Deliberately does NOT:
- derive a limit multiplier from this curve (that needs a business
  risk-appetite decision -- how much exposure to give up per unit of
  calibrated risk -- or real LGD/EAD/revenue economics, neither of which
  exists in this repo; invented numbers here would be exactly the kind
  of "convenient round number" this whole exercise was trying to avoid).
- touch severity/duration-of-delinquency at all (deliberately kept off
  this axis -- severity needs its own point-in-time-safe indicator, per
  the architecture this script's output feeds into).
- run a portfolio simulation (needs a candidate multiplier curve first,
  which needs the policy decision above).

Weighting: isotonic regression is fit on one row PER BORROWER (not per
loan) with y = fwd_new_loans_closed_bad_count / (good+bad) and
sample_weight = good+bad, which is mathematically equivalent to fitting
on every individual closed loan but far cheaper -- loan-count weighting
is exactly what makes this volume-robust, the same principle as
fwd_closed_loan_bad_rate_pct in analyze_risk_tier_pd_resolution.py.

Usage:
    python scripts\\calibrate_pd_isotonic_risk_curve.py --forward-outcomes-file data\\persona_k8_forward_outcomes.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from segmentation.borrower_persona_clustering import digits  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ENGINE_OUTPUT_PATH = REPO / "output" / "engine_test_output.csv"
OUT_DIR = REPO / "segmentation_outputs" / "persona_k8_profile"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine-output-file", type=Path, default=ENGINE_OUTPUT_PATH, help=f"default: {ENGINE_OUTPUT_PATH}")
    p.add_argument("--forward-outcomes-file", type=Path, required=True,
                   help="CSV exported from data/persona_k8_forward_outcomes_query.sql")
    p.add_argument("--grid-step", type=float, default=0.01, help="PD grid resolution for the output curve (default 0.01)")
    p.add_argument("--min-loans-per-borrower", type=int, default=0,
                   help="drop borrowers with fewer than this many closed loans in the window (default 0 = keep all with >=1)")
    args = p.parse_args(argv)

    print("=== Load engine output (cal_pd) ===")
    if not args.engine_output_file.exists():
        print(f"  ERROR: {args.engine_output_file} not found -- nothing to calibrate.")
        return
    eng = pd.read_csv(args.engine_output_file, usecols=lambda c: c in {"msisdn", "cal_pd"})
    eng["_id"] = digits(eng["msisdn"])
    eng = eng.drop(columns=["msisdn"]).dropna(subset=["cal_pd"])
    print(f"  {len(eng):,} borrowers with a cal_pd score")

    print("\n=== Load forward-window closed-loan outcomes ===")
    fwd = pd.read_csv(args.forward_outcomes_file)
    fwd["_id"] = digits(fwd["customer_msisdn"])
    needed = {"fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count"}
    if not needed <= set(fwd.columns):
        print(f"  ERROR: {args.forward_outcomes_file} is missing {needed - set(fwd.columns)} -- cannot calibrate.")
        return

    merged = eng.merge(fwd[["_id"] + list(needed)], on="_id", how="inner")
    merged["_closed_n"] = merged["fwd_new_loans_closed_good_count"] + merged["fwd_new_loans_closed_bad_count"]
    merged = merged[merged["_closed_n"] >= max(1, args.min_loans_per_borrower)]
    merged["_bad_rate"] = merged["fwd_new_loans_closed_bad_count"] / merged["_closed_n"]
    n_borrowers = len(merged)
    n_closed_loans = int(merged["_closed_n"].sum())
    print(f"  {n_borrowers:,} borrowers with >=1 closed loan in the forward window "
          f"({n_closed_loans:,} closed loans total)")
    if n_borrowers < 50:
        print("  WARNING: very small sample -- the fitted curve below will be unstable. Treat as directional only.")

    print("\n=== Fit isotonic regression: cal_pd -> calibrated bad rate (loan-count weighted) ===")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(merged["cal_pd"], merged["_bad_rate"], sample_weight=merged["_closed_n"])

    pd_min, pd_max = merged["cal_pd"].min(), merged["cal_pd"].max()
    grid = np.arange(0.0, 1.0 + args.grid_step, args.grid_step)
    grid = grid[(grid >= pd_min) & (grid <= pd_max)]
    calibrated = iso.predict(grid)
    curve = pd.DataFrame({"cal_pd": grid.round(4), "calibrated_bad_rate_pct": (calibrated * 100).round(3)})

    # Flag where the curve's slope changes most -- candidate "the risk
    # regime actually shifts here" points, instead of picking round PD
    # numbers. Reported, not acted on: where to draw governance-tier
    # boundaries (if any) is a policy call, this just shows the data.
    curve["step_pct"] = curve["calibrated_bad_rate_pct"].diff().round(4)
    out_path = OUT_DIR / "pd_isotonic_calibration_curve.csv"
    curve.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")

    # Print a coarser view (every ~10th grid point) so the console output
    # stays readable; the full-resolution curve is in the CSV.
    coarse = curve.iloc[::max(1, int(0.05 / args.grid_step))]
    with pd.option_context("display.max_rows", None):
        print(coarse.to_string(index=False))

    # Raw-vs-calibrated comparison at a few representative points, so the
    # smoothing effect is visible directly rather than asserted.
    print("\n=== Raw weighted bad rate vs. calibrated, at decile cutpoints of observed cal_pd ===")
    deciles = merged["cal_pd"].quantile(np.arange(0, 1.01, 0.1))
    comp_rows = []
    for q, cutoff in deciles.items():
        raw_at_cutoff = merged.loc[(merged["cal_pd"] >= cutoff - 0.01) & (merged["cal_pd"] <= cutoff + 0.01)]
        raw_rate = (
            raw_at_cutoff["fwd_new_loans_closed_bad_count"].sum() / raw_at_cutoff["_closed_n"].sum() * 100
            if raw_at_cutoff["_closed_n"].sum() > 0 else None
        )
        comp_rows.append({
            "pd_decile": round(q, 2), "cal_pd_cutoff": round(cutoff, 4),
            "raw_bad_rate_pct_near_cutoff": round(raw_rate, 2) if raw_rate is not None else None,
            "calibrated_bad_rate_pct": round(float(iso.predict([cutoff])[0]) * 100, 2),
        })
    print(pd.DataFrame(comp_rows).to_string(index=False))

    print(
        "\nThis is Axis 1 (frequency) only -- deliberately stops before any multiplier. Translating this "
        "calibrated curve into a limit multiplier needs either a business risk-appetite decision (how much "
        "exposure to give up per point of calibrated risk) or real LGD/EAD/revenue economics -- neither is "
        "in this repo, and guessing them would reintroduce exactly the 'convenient round number' problem "
        "this calibration step exists to avoid. Severity (Axis 2) is intentionally untouched here; it needs "
        "its own point-in-time-safe indicator, not cal_pd."
    )


if __name__ == "__main__":
    main()

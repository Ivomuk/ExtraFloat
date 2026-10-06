"""
compare_shadow_vs_live_multiplier.py
=======================================
"Policy disagreement" check -- not just confirming the shadow continuous
multiplier (C3 hybrid) RAN, but showing where it would actually change
decisions relative to the live discrete 4-tier policy, on a single
scoring snapshot (output/engine_test_output.csv, already carrying both
live_tier_multiplier and the shadow_* columns side by side -- no refit,
no forward outcomes needed for this comparison).

This is a point-in-time structural comparison, not a backtest: it
compares what shadow WOULD assign today against what live DID assign
today, for the same borrowers, on the same snapshot. It does not use
realized outcomes -- that comparison (backtest_limit_multiplier_policies.py /
frontier_c3_sensitivity.py) already exists and used a historical forward
window. This script is the natural per-cycle companion once shadow is
running for real: cheap to run on every scoring snapshot, and the basis
for the "operational exceptions" / "policy disagreement" shadow-deployment
metrics described in the architecture write-up.

Reports, for each scenario (base, conservative), restricted to rows with
shadow_status == "ok":
  - multiplier delta (shadow - live): mean, median, direction split
  - limit delta (shadow_limit_post_transition - assigned_limit): mean,
    median, %, direction split
  - disagreement rate: share of rows where |limit delta| / assigned_limit
    exceeds 10% and 20%
  - all of the above broken down by live risk_tier, so the answer to
    "where would this change decisions" is concrete, not just an average.

Usage:
    python scripts\\compare_shadow_vs_live_multiplier.py --engine-output-file output\\engine_test_output.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ENGINE_OUTPUT_PATH = REPO / "output" / "engine_test_output.csv"
OUT_DIR = REPO / "output"

SCENARIOS = ["base", "conservative"]


def _disagreement_row(df: pd.DataFrame, group_label) -> dict:
    n = len(df)
    pct_change = (df["_limit_delta"] / df["assigned_limit"].replace(0, np.nan))
    return {
        "group": group_label,
        "n": n,
        "pct_shadow_gt_live_multiplier": round((df["_mult_delta"] > 1e-9).mean() * 100, 2),
        "pct_shadow_lt_live_multiplier": round((df["_mult_delta"] < -1e-9).mean() * 100, 2),
        "mean_multiplier_delta": round(df["_mult_delta"].mean(), 4),
        "mean_limit_delta": round(df["_limit_delta"].mean(), 1),
        "median_limit_delta": round(df["_limit_delta"].median(), 1),
        "pct_abs_limit_change_gt_10pct": round((pct_change.abs() > 0.10).mean() * 100, 2),
        "pct_abs_limit_change_gt_20pct": round((pct_change.abs() > 0.20).mean() * 100, 2),
        "pct_limit_up": round((df["_limit_delta"] > 0).mean() * 100, 2),
        "pct_limit_down": round((df["_limit_delta"] < 0).mean() * 100, 2),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine-output-file", type=Path, default=ENGINE_OUTPUT_PATH, help=f"default: {ENGINE_OUTPUT_PATH}")
    args = p.parse_args(argv)

    print(f"=== Load {args.engine_output_file} ===")
    df = pd.read_csv(args.engine_output_file)
    required = {"shadow_status", "live_tier_multiplier", "assigned_limit", "risk_tier"}
    missing = required - set(df.columns)
    if missing:
        print(f"  ERROR: missing required columns {missing} -- was this run scored with the shadow "
              "multiplier active (artifacts_dir pointing at a directory with the shadow calibration)?")
        return

    n_total = len(df)
    df_ok = df[df["shadow_status"] == "ok"].copy()
    n_ok = len(df_ok)
    print(f"  {n_total:,} total rows -- {n_ok:,} with shadow_status == 'ok' "
          f"({n_total - n_ok:,} excluded: {df.loc[df['shadow_status'] != 'ok', 'shadow_status'].value_counts().to_dict()})")
    if n_ok == 0:
        print("  No rows with a usable shadow value -- nothing to compare.")
        return

    for scenario in SCENARIOS:
        mult_col = f"shadow_multiplier_{scenario}"
        limit_col = f"shadow_limit_post_transition_{scenario}"
        if mult_col not in df_ok.columns or limit_col not in df_ok.columns:
            print(f"\n  WARNING: {mult_col}/{limit_col} not found -- skipping scenario '{scenario}'")
            continue

        sub = df_ok.dropna(subset=[mult_col, limit_col, "live_tier_multiplier", "assigned_limit"]).copy()
        sub["_mult_delta"] = sub[mult_col] - sub["live_tier_multiplier"]
        sub["_limit_delta"] = sub[limit_col] - sub["assigned_limit"]

        print(f"\n=== Scenario: {scenario} (n={len(sub):,}) ===")
        rows = [_disagreement_row(sub, "OVERALL")]
        for tier, tier_sub in sub.groupby("risk_tier", observed=True):
            rows.append(_disagreement_row(tier_sub, tier))
        result = pd.DataFrame(rows)
        out_path = OUT_DIR / f"shadow_vs_live_disagreement_{scenario}.csv"
        result.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(result.to_string(index=False))

    print(
        "\nReading this: pct_abs_limit_change_gt_10pct / _20pct is the 'policy disagreement' rate -- "
        "how often the shadow policy would actually move a borrower's limit by a material amount, not "
        "just whether it ran. Broken down by live risk_tier, this answers 'where would adopting this "
        "policy actually change decisions' for a credit committee, rather than hiding behind a portfolio "
        "average. This is a point-in-time snapshot comparison, not a backtest against realized outcomes -- "
        "see backtest_limit_multiplier_policies.py / frontier_c3_sensitivity.py for that."
    )


if __name__ == "__main__":
    main()

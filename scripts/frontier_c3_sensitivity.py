"""
frontier_c3_sensitivity.py
============================
The small, deliberate C3 sensitivity grid agreed as the next step -- NOT
another modelling exercise, and NOT optimized against the historical bad
rate (that would just reintroduce overfitting after a disciplined
analysis). r_plateau stays FIXED at its calibration-derived value
(iso.predict(0.25)); only the policy choices in the declining/floor
region are varied, across a small, named set of variants (not a grid
search over hundreds of combinations).

For each variant, reports the exposure/risk FRONTIER:
    delta_exposure_pct  vs.  delta_exposure_weighted_bad_rate_pp (REALIZED)
alongside a second, genuinely different metric requested explicitly:
    expected_bad_exposure_proxy = sum(limit_i * calibrated_risk_i) / sum(limit_i)
This is a forward-looking, MODEL-based proxy (uses the isotonic-calibrated
risk score, not what actually happened historically) -- distinct from the
realized exposure-weighted bad rate, and the natural precursor to a real
expected-loss figure (sum(PD*LGD*EAD)) once LGD becomes available.

Also reports exposure CONCENTRATION in the higher-risk region under each
variant's resulting limits:
    exposure_share_calibrated_risk_gt_10pct
    exposure_share_calibrated_risk_gt_hard_floor_ref  (fixed reference
        threshold across all rows, default 0.138, for comparability)

Two reference rows are included alongside the variant grid, per the
agreed framing -- Current policy (baseline) and Challenger 1 (the
growth-oriented upper-exposure benchmark) -- Challenger 2 is deliberately
excluded here: it already served its purpose as the cautionary "what
goes wrong without plateau-awareness" baseline and doesn't belong in a
C3-focused frontier.

Usage:
    python scripts\\frontier_c3_sensitivity.py --forward-outcomes-file data\\persona_k8_forward_outcomes.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from segmentation.borrower_persona_clustering import digits  # noqa: E402
from scripts.backtest_limit_multiplier_policies import (  # noqa: E402
    _fit_isotonic, _policy_1_plateau, _policy_3_hybrid,
    TIER_MULTIPLIER, BASE_LIMIT, GLOBAL_FLOOR, GLOBAL_CEILING,
)

REPO = Path(__file__).resolve().parent.parent
ENGINE_OUTPUT_PATH = REPO / "output" / "engine_test_output.csv"
OUT_DIR = REPO / "segmentation_outputs" / "persona_k8_profile"

# name, r_floor (absolute calibrated risk, None = use r_plateau + fraction
# of range like the original C3), m_min, delta_up, delta_down (None = no
# transition control)
VARIANTS = [
    {"name": "C3_floor0.12",        "r_floor": 0.12,  "m_min": 0.40, "delta": 0.25},
    {"name": "C3_floor0.138_base",  "r_floor": 0.138, "m_min": 0.40, "delta": 0.25},
    {"name": "C3_floor0.15",        "r_floor": 0.15,  "m_min": 0.40, "delta": 0.25},
    {"name": "C3_mmin0.50",         "r_floor": 0.138, "m_min": 0.50, "delta": 0.25},
    {"name": "C3_mmin0.30",         "r_floor": 0.138, "m_min": 0.30, "delta": 0.25},
    {"name": "C3_delta0.15",        "r_floor": 0.138, "m_min": 0.40, "delta": 0.15},
    {"name": "C3_delta0.20",        "r_floor": 0.138, "m_min": 0.40, "delta": 0.20},
    {"name": "C3_no_transition_cap","r_floor": 0.138, "m_min": 0.40, "delta": None},
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine-output-file", type=Path, default=ENGINE_OUTPUT_PATH, help=f"default: {ENGINE_OUTPUT_PATH}")
    p.add_argument("--forward-outcomes-file", type=Path, required=True,
                   help="CSV exported from data/persona_k8_forward_outcomes_query.sql")
    p.add_argument("--m-max", type=float, default=1.00, help="default: 1.00")
    p.add_argument("--concentration-threshold", type=float, default=0.10,
                   help="calibrated-risk threshold for the first concentration metric; default: 0.10")
    p.add_argument("--hard-floor-reference", type=float, default=0.138,
                   help="fixed reference threshold for the second concentration metric, held constant "
                        "across all rows for comparability; default: 0.138 (the original C3's floor)")
    args = p.parse_args(argv)

    print("=== Load engine output ===")
    eng_available = set(pd.read_csv(args.engine_output_file, nrows=0).columns)
    wanted = ["msisdn", "cal_pd", "risk_tier", "assigned_limit", "capacity_cap"]
    eng_cols = [c for c in wanted if c in eng_available]
    has_capacity_cap = "capacity_cap" in eng_cols
    if not has_capacity_cap:
        print("  NOTE: capacity_cap not available -- using the assigned_limit-rescaling approximation.")
    eng = pd.read_csv(args.engine_output_file, usecols=eng_cols)
    eng["_id"] = digits(eng["msisdn"])

    print("\n=== Load forward outcomes + fit isotonic calibration ===")
    fwd = pd.read_csv(args.forward_outcomes_file)
    fwd["_id"] = digits(fwd["customer_msisdn"])
    needed = ["fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count"]
    iso = _fit_isotonic(eng, fwd)

    df = eng.merge(fwd[["_id"] + needed], on="_id", how="left")
    df["_closed_n"] = df[needed[0]].fillna(0) + df[needed[1]].fillna(0)
    df["_bad_rate"] = np.where(df["_closed_n"] > 0, df[needed[1]] / df["_closed_n"].replace(0, np.nan), np.nan)
    df = df.dropna(subset=["cal_pd", "risk_tier", "assigned_limit"])
    df["calibrated_risk"] = iso.predict(df["cal_pd"])
    df["current_multiplier"] = df["risk_tier"].map(TIER_MULTIPLIER)
    df = df.dropna(subset=["current_multiplier"])

    r = df["calibrated_risk"].to_numpy()
    r_plateau = float(iso.predict([0.25])[0])  # FIXED, per the agreed design -- not varied
    r_max_obs = df["calibrated_risk"].max()
    print(f"  r_plateau (fixed, calibration-derived): {r_plateau:.4f}")
    print(f"  observed max calibrated risk: {r_max_obs:.4f}")

    current_proxy_denom = df["assigned_limit"].sum()
    current_proxy = (df["assigned_limit"] * df["calibrated_risk"]).sum() / current_proxy_denom * 100
    current_bad_rate = (df.loc[df["_bad_rate"].notna(), "assigned_limit"] * df.loc[df["_bad_rate"].notna(), "_bad_rate"]).sum() \
        / df.loc[df["_bad_rate"].notna(), "assigned_limit"].sum() * 100
    current_exposure = df["assigned_limit"].sum()

    def compute_limit(mult: np.ndarray, delta: float | None) -> pd.Series:
        risk_cap = mult * BASE_LIMIT
        if has_capacity_cap:
            raw = np.minimum(df["capacity_cap"], risk_cap).clip(GLOBAL_FLOOR, GLOBAL_CEILING)
        else:
            raw = df["assigned_limit"] * (mult / df["current_multiplier"])
        if delta is not None:
            raw = raw.clip(df["assigned_limit"] * (1 - delta), df["assigned_limit"] * (1 + delta))
        return raw

    def row_metrics(name: str, limit: pd.Series) -> dict:
        has_bad = df["_bad_rate"].notna()
        exposure = limit.sum()
        bad_rate = (limit[has_bad] * df.loc[has_bad, "_bad_rate"]).sum() / limit[has_bad].sum() * 100
        proxy = (limit * df["calibrated_risk"]).sum() / exposure * 100
        pct_change = (limit - df["assigned_limit"]) / df["assigned_limit"].replace(0, np.nan)
        conc1 = limit[df["calibrated_risk"] > args.concentration_threshold].sum() / exposure * 100
        conc2 = limit[df["calibrated_risk"] > args.hard_floor_reference].sum() / exposure * 100
        return {
            "variant": name,
            "total_exposure": exposure,
            "delta_exposure_pct": round((exposure - current_exposure) / current_exposure * 100, 2),
            "exposure_weighted_bad_rate_pct": round(bad_rate, 4),
            "delta_bad_rate_pp": round(bad_rate - current_bad_rate, 4),
            "expected_bad_exposure_proxy_pct": round(proxy, 4),
            "delta_proxy_pp": round(proxy - current_proxy, 4),
            "pct_up_gt10pct": round((pct_change > 0.10).mean() * 100, 2),
            "pct_down_gt10pct": round((pct_change < -0.10).mean() * 100, 2),
            f"exposure_share_risk_gt_{args.concentration_threshold:.3f}": round(conc1, 2),
            f"exposure_share_risk_gt_{args.hard_floor_reference:.3f}": round(conc2, 2),
        }

    rows = [{
        "variant": "Current (baseline)", "total_exposure": current_exposure, "delta_exposure_pct": 0.0,
        "exposure_weighted_bad_rate_pct": round(current_bad_rate, 4), "delta_bad_rate_pp": 0.0,
        "expected_bad_exposure_proxy_pct": round(current_proxy, 4), "delta_proxy_pp": 0.0,
        "pct_up_gt10pct": 0.0, "pct_down_gt10pct": 0.0,
        f"exposure_share_risk_gt_{args.concentration_threshold:.3f}": round(
            df.loc[df["calibrated_risk"] > args.concentration_threshold, "assigned_limit"].sum() / current_exposure * 100, 2),
        f"exposure_share_risk_gt_{args.hard_floor_reference:.3f}": round(
            df.loc[df["calibrated_risk"] > args.hard_floor_reference, "assigned_limit"].sum() / current_exposure * 100, 2),
    }]

    c1_limit = compute_limit(_policy_1_plateau(r, r_plateau, r_max_obs, args.m_max, 0.40), 0.25)
    rows.append(row_metrics("C1 (growth-oriented, reference)", c1_limit))

    for v in VARIANTS:
        mult = _policy_3_hybrid(r, r_plateau, v["r_floor"], args.m_max, v["m_min"])
        limit = compute_limit(mult, v["delta"])
        rows.append(row_metrics(v["name"], limit))

    frontier = pd.DataFrame(rows).sort_values("delta_exposure_pct", ascending=False)
    out_path = OUT_DIR / "c3_sensitivity_frontier.csv"
    frontier.to_csv(out_path, index=False)
    print(f"\n=== Frontier (sorted most-generous to most-conservative) ===")
    print(f"  wrote {out_path}")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(frontier.to_string(index=False))

    print(
        "\nReading this: delta_exposure_pct vs. delta_bad_rate_pp is the historical-outcomes frontier; "
        "delta_proxy_pp is the model-based (calibrated-risk) view of the same tradeoff, independent of "
        "whether this specific 42-day forward window happened to be representative. The two "
        "exposure_share_risk_gt_* columns show whether a variant is actually shrinking the book's exposure "
        "to its riskiest region, or just moving money around in the safe zone. Pick the variant whose "
        "position on this frontier matches the business's risk appetite -- none of these rows is "
        "'correct' on the data alone."
    )


if __name__ == "__main__":
    main()

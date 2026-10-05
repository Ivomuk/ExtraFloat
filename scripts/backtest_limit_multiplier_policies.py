"""
backtest_limit_multiplier_policies.py
========================================
Multiplier backtest: three challenger policies vs. the current
{1.00, 0.85, 0.65, 0.40} tiered policy, all operating on ISOTONIC
CALIBRATED RISK (fit fresh here, same approach as
calibrate_pd_isotonic_risk_curve.py), not raw cal_pd -- this separates
model calibration from business risk appetite per the agreed design:

    cal_pd --isotonic--> calibrated_risk --policy function--> multiplier

IMPORTANT LABELLING: the three challenger curves below are explicitly
POLICY SCENARIOS for comparison, not optimized or recommended multiplier
values. The isotonic curve tells us relative realized risk; it cannot by
itself justify any specific multiplier value -- that translation needs
LGD/EAD/revenue/funding-cost/risk-appetite inputs this repo does not have.

Challenger 1 -- Plateau-aware (conservative): flat at M_max through the
    empirical plateau (the calibrated-risk value at cal_pd=0.25, read off
    the fitted curve, not a round number), then a GENTLE linear decline
    across the full remaining range down to M_min at the highest observed
    calibrated risk.
Challenger 2 -- Naive smooth linear (baseline for contrast, NOT
    recommended): linear in calibrated risk across the ENTIRE observed
    range, no plateau privilege -- included specifically to show what
    goes wrong if you skip the plateau-aware step: it needlessly
    discounts borrowers sitting in the empirically flat region.
Challenger 3 -- Hybrid (flat + continuous decline + hard floor): flat at
    M_max through the same plateau, a STEEPER linear decline reaching
    M_min well before the top of the observed range (at
    --hybrid-floor-fraction of the way from the plateau to the max,
    default 0.6), then flat at the M_min hard floor for the remaining
    high-risk tail. Likely the most governable of the three.

LIMIT SIMULATION, both paths honestly labelled:
- If engine_output has `capacity_cap` (keep_intermediate=True on the
  engine run): challenger_limit_raw = min(capacity_cap, challenger
  multiplier x base_limit), clipped to [global_floor_limit,
  global_ceiling_limit]. This still ignores recent_usage_cap,
  prior_exposure_cap and apply_policy_adjustments()'s other haircuts --
  a closer approximation than the fallback below, not an exact
  reproduction of run_limit_caps().
- Else (the common case): challenger_limit_raw = assigned_limit x
  (challenger_multiplier / current_multiplier) -- assumes the risk cap is
  the effective binding constraint for every borrower. This OVERSTATES
  the impact for capacity-constrained borrowers (where capacity_cap, not
  risk_cap, actually binds) -- flagged loudly in the output, not silently
  assumed away.

Optional transition-control clamp (--delta-up/--delta-down, illustrative
defaults): challenger_limit = clip(challenger_limit_raw,
assigned_limit*(1-delta_down), assigned_limit*(1+delta_up)) -- separates
"is the target policy better" from "how fast could we safely migrate the
existing portfolio to it."

BACKTEST ASSUMPTION, stated explicitly: exposure-weighted bad rate uses
each borrower's ALREADY-REALIZED closed-loan bad rate from the forward
window, reweighted by the counterfactual limit. This assumes realized
behavior is independent of the limit assigned -- a standard, necessary
backtesting simplification (we cannot re-run history with a different
limit actually in force), not a claim that limits don't affect behavior.

Usage:
    python scripts\\backtest_limit_multiplier_policies.py --forward-outcomes-file data\\persona_k8_forward_outcomes.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from extrafloat.engine.extrafloat_limit_engine_caps import DEFAULT_CAP_CONFIG  # noqa: E402
from extrafloat.engine.extrafloat_shadow_risk_multiplier import _policy_3_hybrid  # noqa: E402
from segmentation.borrower_persona_clustering import digits  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ASSIGNMENTS_PATH = REPO / "segmentation_outputs" / "persona_k8_profile" / "k8_cluster_assignments.csv"
ENGINE_OUTPUT_PATH = REPO / "output" / "engine_test_output.csv"
BORROWER_HISTORY_PATH = REPO / "borrower_history_retail_filtered.csv"
OUT_DIR = REPO / "segmentation_outputs" / "persona_k8_profile"

TIER_MULTIPLIER = {"tier_1": 1.00, "tier_2": 0.85, "tier_3": 0.65, "tier_4": 0.40}
BASE_LIMIT = DEFAULT_CAP_CONFIG["risk"]["base_limit"]
GLOBAL_FLOOR = DEFAULT_CAP_CONFIG["global_floor_limit"]
GLOBAL_CEILING = DEFAULT_CAP_CONFIG["global_ceiling_limit"]


def _fit_isotonic(eng: pd.DataFrame, fwd: pd.DataFrame) -> IsotonicRegression:
    merged = eng[["_id", "cal_pd"]].dropna().merge(
        fwd[["_id", "fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count"]], on="_id", how="inner")
    merged["_closed_n"] = merged["fwd_new_loans_closed_good_count"] + merged["fwd_new_loans_closed_bad_count"]
    merged = merged[merged["_closed_n"] >= 1]
    merged["_bad_rate"] = merged["fwd_new_loans_closed_bad_count"] / merged["_closed_n"]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(merged["cal_pd"], merged["_bad_rate"], sample_weight=merged["_closed_n"])
    return iso


def _policy_1_plateau(r: np.ndarray, r_plateau: float, r_max: float, m_max: float, m_min: float) -> np.ndarray:
    out = np.full_like(r, m_max, dtype=float)
    mask = r > r_plateau
    span = max(r_max - r_plateau, 1e-9)
    out[mask] = m_max - (m_max - m_min) * np.clip((r[mask] - r_plateau) / span, 0, 1)
    return out


def _policy_2_linear(r: np.ndarray, r_min_obs: float, r_max: float, m_max: float, m_min: float) -> np.ndarray:
    span = max(r_max - r_min_obs, 1e-9)
    return m_max - (m_max - m_min) * np.clip((r - r_min_obs) / span, 0, 1)


# _policy_3_hybrid moved to extrafloat.engine.extrafloat_shadow_risk_multiplier
# (imported above) -- it's now also the live shadow-deployment policy
# function, so this repo's scripts/-depends-on-extrafloat/engine/ layering
# convention means the production copy is the source of truth and this
# script imports it rather than defining its own.


def _summarize(df: pd.DataFrame, group_col: str | None) -> pd.DataFrame:
    rows = []
    groups = [(None, df)] if group_col is None else list(df.groupby(group_col, observed=True))
    for key, sub in groups:
        row = {"group": key if key is not None else "OVERALL", "n": len(sub)}
        has_bad_rate = sub["_bad_rate"].notna()
        for label, lim_col in [("current", "assigned_limit"), ("c1", "limit_c1"), ("c2", "limit_c2"), ("c3", "limit_c3")]:
            lim = sub[lim_col]
            row[f"total_exposure_{label}"] = lim.sum()
            row[f"avg_limit_{label}"] = lim.mean()
            row[f"median_limit_{label}"] = lim.median()
            w = sub.loc[has_bad_rate, lim_col]
            br = sub.loc[has_bad_rate, "_bad_rate"]
            row[f"exposure_weighted_bad_rate_pct_{label}"] = round((w * br).sum() / w.sum() * 100, 3) if w.sum() > 0 else None
            if label != "current":
                pct_change = (lim - sub["assigned_limit"]) / sub["assigned_limit"].replace(0, np.nan)
                row[f"pct_up_gt10pct_{label}"] = round((pct_change > 0.10).mean() * 100, 2)
                row[f"pct_down_gt10pct_{label}"] = round((pct_change < -0.10).mean() * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assignments-file", type=Path, default=ASSIGNMENTS_PATH, help=f"default: {ASSIGNMENTS_PATH}")
    p.add_argument("--engine-output-file", type=Path, default=ENGINE_OUTPUT_PATH, help=f"default: {ENGINE_OUTPUT_PATH}")
    p.add_argument("--borrower-history-file", type=Path, default=BORROWER_HISTORY_PATH, help=f"default: {BORROWER_HISTORY_PATH}")
    p.add_argument("--forward-outcomes-file", type=Path, required=True,
                   help="CSV exported from data/persona_k8_forward_outcomes_query.sql")
    p.add_argument("--m-max", type=float, default=1.00, help="default: 1.00 (matches current tier_1)")
    p.add_argument("--m-min", type=float, default=0.40, help="default: 0.40 (matches current tier_4)")
    p.add_argument("--hybrid-floor-fraction", type=float, default=0.6,
                   help="Challenger 3: fraction of the way from the plateau to the max observed risk "
                        "where the hard floor starts. default: 0.6")
    p.add_argument("--delta-up", type=float, default=0.25, help="transition-control cap, illustrative default: 0.25 (25%% max increase)")
    p.add_argument("--delta-down", type=float, default=0.25, help="transition-control cap, illustrative default: 0.25 (25%% max decrease)")
    p.add_argument("--no-transition-control", action="store_true", help="report raw challenger limits, unclamped")
    args = p.parse_args(argv)

    print("=== Load engine output ===")
    eng_available = set(pd.read_csv(args.engine_output_file, nrows=0).columns)
    wanted = ["msisdn", "cal_pd", "risk_tier", "assigned_limit", "capacity_cap"]
    eng_cols = [c for c in wanted if c in eng_available]
    missing = [c for c in wanted if c not in eng_available]
    if "capacity_cap" in missing:
        print("  NOTE: capacity_cap not in engine output (needs keep_intermediate=True on the engine run) -- "
              "falling back to the assigned_limit-rescaling approximation (see module docstring).")
    if {"msisdn", "cal_pd", "risk_tier", "assigned_limit"} - set(eng_cols):
        print(f"  ERROR: engine output is missing required columns {missing} -- cannot backtest.")
        return
    eng = pd.read_csv(args.engine_output_file, usecols=eng_cols)
    eng["_id"] = digits(eng["msisdn"])
    has_capacity_cap = "capacity_cap" in eng.columns

    print("\n=== Load forward outcomes ===")
    fwd = pd.read_csv(args.forward_outcomes_file)
    fwd["_id"] = digits(fwd["customer_msisdn"])
    needed = {"fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count"}
    if not needed <= set(fwd.columns):
        print(f"  ERROR: forward-outcomes file is missing {needed - set(fwd.columns)} -- cannot backtest.")
        return

    print("\n=== Fit isotonic calibration (same approach as calibrate_pd_isotonic_risk_curve.py) ===")
    iso = _fit_isotonic(eng, fwd)

    df = eng.merge(fwd[["_id"] + list(needed)], on="_id", how="left")
    df["_closed_n"] = df["fwd_new_loans_closed_good_count"].fillna(0) + df["fwd_new_loans_closed_bad_count"].fillna(0)
    df["_bad_rate"] = np.where(df["_closed_n"] > 0,
                                df["fwd_new_loans_closed_bad_count"] / df["_closed_n"].replace(0, np.nan), np.nan)
    df = df.dropna(subset=["cal_pd", "risk_tier", "assigned_limit"])
    df["calibrated_risk"] = iso.predict(df["cal_pd"])
    df["current_multiplier"] = df["risk_tier"].map(TIER_MULTIPLIER)
    n_unmapped = df["current_multiplier"].isna().sum()
    if n_unmapped:
        print(f"  WARNING: {n_unmapped:,} rows have a risk_tier not in {list(TIER_MULTIPLIER)} -- dropping them.")
        df = df.dropna(subset=["current_multiplier"])

    r_plateau = float(iso.predict([0.25])[0])
    r_min_obs, r_max_obs = df["calibrated_risk"].min(), df["calibrated_risk"].max()
    r_floor = r_plateau + args.hybrid_floor_fraction * (r_max_obs - r_plateau)
    print(f"  r_plateau (calibrated risk at cal_pd=0.25): {r_plateau:.4f}")
    print(f"  observed calibrated risk range: [{r_min_obs:.4f}, {r_max_obs:.4f}]")
    print(f"  Challenger 3 hard-floor starts at calibrated risk: {r_floor:.4f}")

    r = df["calibrated_risk"].to_numpy()
    df["mult_c1"] = _policy_1_plateau(r, r_plateau, r_max_obs, args.m_max, args.m_min)
    df["mult_c2"] = _policy_2_linear(r, r_min_obs, r_max_obs, args.m_max, args.m_min)
    df["mult_c3"] = _policy_3_hybrid(r, r_plateau, r_floor, args.m_max, args.m_min)

    print("\n=== Simulate challenger limits ===")
    for c in ("c1", "c2", "c3"):
        challenger_risk_cap = df[f"mult_{c}"] * BASE_LIMIT
        if has_capacity_cap:
            raw = np.minimum(df["capacity_cap"], challenger_risk_cap)
            raw = raw.clip(GLOBAL_FLOOR, GLOBAL_CEILING)
        else:
            raw = df["assigned_limit"] * (df[f"mult_{c}"] / df["current_multiplier"])
        if not args.no_transition_control:
            raw = raw.clip(df["assigned_limit"] * (1 - args.delta_down), df["assigned_limit"] * (1 + args.delta_up))
        df[f"limit_{c}"] = raw
    print(f"  limit simulation method: {'min(capacity_cap, challenger risk cap)' if has_capacity_cap else 'assigned_limit rescaled by multiplier ratio (approximation -- see docstring)'}")
    print(f"  transition control: {'OFF (raw challenger limits)' if args.no_transition_control else f'+{args.delta_up:.0%} / -{args.delta_down:.0%} vs. current assigned_limit'}")

    print("\n=== Overall backtest summary ===")
    overall = _summarize(df, None)
    out_overall = OUT_DIR / "multiplier_backtest_overall.csv"
    overall.to_csv(out_overall, index=False)
    print(f"  wrote {out_overall}")
    print(overall.to_string(index=False))

    print("\n=== By current risk tier ===")
    by_tier = _summarize(df, "risk_tier")
    (OUT_DIR / "multiplier_backtest_by_tier.csv").write_text(by_tier.to_csv(index=False))
    print(by_tier.to_string(index=False))

    print("\n=== By calibrated-risk decile ===")
    # duplicates="drop" can collapse below 10 bins when the isotonic curve's
    # plateau region creates many repeated values -- label AFTER binning,
    # sized to however many distinct bins actually resulted, rather than
    # assuming exactly 10 labels will fit.
    _decile_raw = pd.qcut(df["calibrated_risk"], 10, duplicates="drop")
    _n_bins = _decile_raw.cat.categories.size
    if _n_bins < 10:
        print(f"  NOTE: calibrated risk has repeated values (isotonic plateau) -- collapsed to {_n_bins} "
              f"distinct bins instead of 10.")
    df["_risk_decile"] = _decile_raw.cat.rename_categories([f"D{i+1}" for i in range(_n_bins)])
    by_decile = _summarize(df, "_risk_decile")
    (OUT_DIR / "multiplier_backtest_by_pd_decile.csv").write_text(by_decile.to_csv(index=False))
    print(by_decile.to_string(index=False))

    if args.assignments_file.exists():
        print("\n=== By K=8 persona ===")
        assignments = pd.read_csv(args.assignments_file, usecols=["phonenumber", "persona_name"])
        assignments["_id"] = digits(assignments["phonenumber"])
        df_p = df.merge(assignments[["_id", "persona_name"]], on="_id", how="left")
        by_persona = _summarize(df_p, "persona_name")
        (OUT_DIR / "multiplier_backtest_by_persona.csv").write_text(by_persona.to_csv(index=False))
        print(by_persona.to_string(index=False))
        print(
            "\n  Reading this: check whether C6 (Elite Quality) naturally lands on a higher average "
            "multiplier/limit under the challengers WITHOUT persona ever being a model input -- that's the "
            "desired result (risk drives the individual limit; persona tells us whether the portfolio "
            "consequence makes business sense), not something to force."
        )
    else:
        print(f"\n  NOTE: {args.assignments_file} not found -- skipping persona breakdown.")

    if args.borrower_history_file.exists():
        print("\n=== By thin/thick file status ===")
        bh = pd.read_csv(args.borrower_history_file, usecols=["phonenumber", "total_loans"])
        bh["_id"] = digits(bh["phonenumber"])
        bh["file_status"] = np.where(bh["total_loans"].fillna(0) < 3, "thin_file", "thick_file")
        df_t = df.merge(bh[["_id", "file_status"]], on="_id", how="left")
        by_thin = _summarize(df_t, "file_status")
        (OUT_DIR / "multiplier_backtest_by_file_status.csv").write_text(by_thin.to_csv(index=False))
        print(by_thin.to_string(index=False))
    else:
        print(f"\n  NOTE: {args.borrower_history_file} not found -- skipping thin/thick breakdown.")

    print(
        "\nThese three challenger curves are POLICY SCENARIOS for comparison, not recommended multiplier "
        "values -- the isotonic curve justifies relative risk ordering, not the specific M_max/M_min/shape "
        "chosen here. Setting the actual values needs LGD/EAD/revenue/risk-appetite input from the "
        "business, per the agreed design."
    )


if __name__ == "__main__":
    main()

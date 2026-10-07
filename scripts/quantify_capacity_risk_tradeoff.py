"""
quantify_capacity_risk_tradeoff.py
=========================================
The economic follow-up to validate_capacity_risk_quadrant_outcomes.py:
now that Driver A (lower-risk, over-limit) and Driver B (high-risk,
over-limit) both show real, non-confounded business capacity AND real,
non-volume-driven repayment consequences, management's actual question
is unit economics -- how much additional credit shortfall are we
taking on for each additional shilling of exposure extended beyond the
model's recommendation?

WHAT THIS DOES NOT COMPUTE, AND WHY:
  - True revenue per incremental shilling: this extract has no
    fee/commission-per-loan field, so revenue from the extra lending
    cannot be isolated. This script reports exposure and repayment
    shortfall only -- a unit-economics-of-RISK metric, not a full
    revenue-vs-loss P&L. A true ROI calculation needs a fee/commission
    schedule joined in separately.
  - True lifetime LGD/EAD: fwd_new_loans_repaid_ugx / disbursed_ugx is
    repayment coverage WITHIN the 42-day forward window only, on loans
    newly originated in that window -- not the lifetime loss on the
    specific disbursement that drove the original over-limit flag.
    Loans still open at window_end may still cure or still worsen;
    this is an early, directional read, same caveat as the window-days
    warning below.

WHAT IT DOES COMPUTE -- an ECONOMIC PROXY, not a profitability or
expected-loss analysis:
  - excess exposure: sum(excess_over_<reference>_disbursed) -- the
    actual extra shillings disbursed beyond the recommended limit, for
    the over-limit group. (Within-limit groups should sum to ~0 by
    construction; reported as a sanity check, not a headline.)
  - forward shortfall: sum(fwd_new_loans_disbursed_ugx) -
    sum(fwd_new_loans_repaid_ugx) on NEW loans originated in the
    forward window -- unrepaid amount SO FAR, not a final loss figure.
    Deliberately called "shortfall," never "loss"/"credit loss"/"NPL":
    with only a 42-day window, some of Disbursed-Repaid is principal
    not yet contractually due or a loan still legitimately open, not
    necessarily money that will never come back. See the maturity
    caveat below.
  - incremental shortfall, per risk population SEPARATELY (never one
    portfolio-wide baseline), computed TWO ways:
      coarse: that population's own within-limit reference group's
        single pooled shortfall RATE, applied to the over-limit
        group's own forward-disbursed volume.
      PD-band standardized (preferred): the within-limit reference's
        shortfall rate computed SEPARATELY within each finer cal_pd
        band (LOWER_RISK_CAL_PD_LABELS / HIGH_RISK_CAL_PD_LABELS),
        each applied to the over-limit group's OWN forward-disbursed
        volume in that SAME band, then summed -- a single "cal_pd>=30%"
        baseline treats a 0.31 and a 0.65 borrower as interchangeable,
        which the isotonic calibration work already showed they are
        not, so the coarse version alone risks comparing the over-limit
        group against a baseline that doesn't match its own risk
        composition.
    Either way, incremental shortfall = actual - expected -- answers
    "how much more shortfall did the over-limit population generate
    than expected if the same forward-lending volume had performed at
    the SAME-cal_pd-band within-limit rate," not "how much shortfall
    is there in total."
  - two denominators, each answering a different question:
      incremental shortfall / forward-disbursed volume -- the
          incremental performance penalty relative to subsequent
          lending overall.
      incremental shortfall / excess exposure -- "for every 100
          shillings of identified excess exposure, how many shillings
          of above-baseline forward shortfall did the subsequent
          period contain." NOT "loss caused by each extra shilling" --
          the numerator is forward-period repayment performance across
          a population, the denominator is excess exposure measured at
          the original observation point; related through the same
          borrowers, but not necessarily the same loans. An
          association-based economic-intensity measure, stated as one.
  - the reciprocal, excess exposure / incremental shortfall, ONLY when
    incremental shortfall is positive -- "UGX X of excess lending was
    associated with each UGX 1 of additional forward shortfall."
    Reported as "no positive incremental shortfall observed" otherwise
    -- never force a reciprocal out of a zero or negative numerator.

MATURITY / CENSORING CAVEAT -- the biggest technical concern with this
measure: the over-limit groups have higher still-active rates (see
validate_capacity_risk_quadrant_outcomes.py's censoring check), so
Disbursed-Repaid may partly reflect loans that simply haven't had time
to repay yet, not degraded performance. The clean fix -- restricting to
loans with current_loan_start_date <= window_end - tenor - a
performance buffer, so every included loan had ~equal opportunity to
repay -- needs per-loan origination dates that are NOT present in this
borrower-level-aggregated extract (data/persona_k8_forward_outcomes.csv
sums fwd_new_loans_disbursed_ugx/repaid_ugx per borrower, with no
per-loan date). Building that version requires a SQL change to
data/persona_k8_forward_outcomes_query.sql to add a maturity-restricted
aggregate -- not done here. If this product's tenor is short enough
that essentially every loan in the window has already matured, this
caveat matters less in practice; this script does not assume that
either way.

Usage:
    python scripts\\quantify_capacity_risk_tradeoff.py ^
        --live-shadow-file live_shadow_vs_category_limit.csv ^
        --forward-outcomes-file data\\persona_k8_forward_outcomes.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.borrower_persona_clustering import digits  # noqa: E402

OVER_LIMIT_THRESHOLD = 1.10
ZERO_FILL_COLS = ["fwd_new_loans_disbursed_ugx", "fwd_new_loans_repaid_ugx"]
QUADRANT_ORDER = ["Reference", "Capacity test", "Risk reference", "Critical risk test"]

# A single baseline shortfall rate for the whole "cal_pd>=30%" (or "<30%") population treats
# a 0.31 and a 0.65 borrower as interchangeable, which the isotonic calibration work already
# showed they are not. These finer bands -- same convention as the lower-risk bands used
# elsewhere in this analysis, extended for the high-risk side per the >=30% sub-bands used in
# the matched-baseline feature profile -- let the standardization below compare each
# over-limit borrower's shortfall against reference borrowers in the SAME cal_pd band, not one
# pooled rate for the whole risk side.
LOWER_RISK_CAL_PD_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
LOWER_RISK_CAL_PD_LABELS = ["<2%", "2-5%", "5-10%", "10-20%", "20-30%"]
HIGH_RISK_CAL_PD_EDGES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 1.0]
HIGH_RISK_CAL_PD_LABELS = ["30-35%", "35-40%", "40-45%", "45-50%", "50-60%", "60%+"]


def _pd_band_standardized_expected_shortfall(
    df: pd.DataFrame, over_mask: pd.Series, within_mask: pd.Series, band_col: str, band_labels: list[str],
) -> tuple[float, pd.DataFrame, list[str]]:
    """Dollar-weighted standardization: E[Shortfall_over] = sum_j (ForwardDisbursed_over,j x
    ShortfallRate_within,j) -- the over-limit group's OWN forward-disbursed dollar volume in
    each cal_pd band, multiplied by the within-limit reference's shortfall RATE in that SAME
    band, summed across bands. Weights by dollar volume (not agent count), since the quantity
    being reconstructed is a dollar total, not a mean rate -- unlike the agent-count-weighted
    standardization used for the DPD rate elsewhere in this analysis.
    Returns (expected_shortfall_total, per_band_detail, bands_missing_a_within-limit_comparator).
    """
    rows = []
    missing = []
    for band in band_labels:
        over_band_mask = over_mask & (df[band_col] == band)
        within_band_mask = within_mask & (df[band_col] == band)
        over_disbursed = df.loc[over_band_mask, "fwd_new_loans_disbursed_ugx"].sum()
        over_repaid = df.loc[over_band_mask, "fwd_new_loans_repaid_ugx"].sum()
        over_actual_shortfall = over_disbursed - over_repaid
        within_disbursed = df.loc[within_band_mask, "fwd_new_loans_disbursed_ugx"].sum()
        within_shortfall = (
            within_disbursed - df.loc[within_band_mask, "fwd_new_loans_repaid_ugx"].sum()
        )
        if over_disbursed == 0:
            continue
        if within_disbursed == 0:
            missing.append(band)
            rows.append({"cal_pd_band": band, "n_over_agents": int(over_band_mask.sum()),
                         "over_forward_disbursed": over_disbursed, "over_actual_shortfall": over_actual_shortfall,
                         "n_within_agents": 0, "within_shortfall_rate_pct": float("nan"),
                         "expected_shortfall": float("nan"), "band_incremental_shortfall": float("nan")})
            continue
        within_rate = within_shortfall / within_disbursed
        expected = within_rate * over_disbursed
        rows.append({
            "cal_pd_band": band, "n_over_agents": int(over_band_mask.sum()),
            "over_forward_disbursed": over_disbursed, "over_actual_shortfall": over_actual_shortfall,
            "n_within_agents": int(within_band_mask.sum()),
            "within_shortfall_rate_pct": round(within_rate * 100, 3), "expected_shortfall": expected,
            "band_incremental_shortfall": over_actual_shortfall - expected,
        })
    detail = pd.DataFrame(rows, columns=["cal_pd_band", "n_over_agents", "over_forward_disbursed",
                                          "over_actual_shortfall", "n_within_agents",
                                          "within_shortfall_rate_pct", "expected_shortfall",
                                          "band_incremental_shortfall"])
    total_expected = detail["expected_shortfall"].sum(skipna=True) if len(detail) else 0.0
    return total_expected, detail, missing


def _excess_exposure_allocation(
    df: pd.DataFrame, over_mask: pd.Series, band_col: str, band_labels: list[str],
    band_detail: pd.DataFrame, total_excess_exposure: float, total_incremental_shortfall: float,
) -> pd.DataFrame:
    """Where is the excess exposure actually going, and does it line up with where the
    incremental shortfall is appearing? Two populations can have the same total excess
    exposure for very different reasons -- many agents each getting a modest increase, or
    a few agents getting a very large one -- which matters for whether the right policy
    response is a systematic capacity adjustment or an override/concentration control.
    """
    rows = []
    for band in band_labels:
        band_mask = over_mask & (df[band_col] == band)
        n_agents = int(band_mask.sum())
        if n_agents == 0:
            continue
        excess = df.loc[band_mask, "incremental_exposure"].sum()
        match = band_detail[band_detail["cal_pd_band"] == band]
        if len(match):
            forward_disbursed = match["over_forward_disbursed"].iloc[0]
            band_incremental = match["band_incremental_shortfall"].iloc[0]
        else:
            forward_disbursed = df.loc[band_mask, "fwd_new_loans_disbursed_ugx"].sum()
            band_incremental = float("nan")
        rows.append({
            "cal_pd_band": band,
            "n_over_agents": n_agents,
            "excess_exposure": excess,
            "pct_of_driver_excess_exposure": (excess / total_excess_exposure * 100) if total_excess_exposure else float("nan"),
            "excess_exposure_per_agent": (excess / n_agents) if n_agents else float("nan"),
            "forward_disbursed": forward_disbursed,
            "band_incremental_shortfall": band_incremental,
            "pct_of_driver_incremental_shortfall": (
                band_incremental / total_incremental_shortfall * 100
            ) if total_incremental_shortfall else float("nan"),
            "incremental_shortfall_pct_of_forward": (
                band_incremental / forward_disbursed * 100
            ) if forward_disbursed else float("nan"),
        })
    return pd.DataFrame(rows)


def _quadrant_label(is_high_risk: bool, is_over_limit: bool) -> str:
    if not is_high_risk and not is_over_limit:
        return "Reference"
    if not is_high_risk and is_over_limit:
        return "Capacity test"
    if is_high_risk and not is_over_limit:
        return "Risk reference"
    return "Critical risk test"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-shadow-file", default="live_shadow_vs_category_limit.csv")
    ap.add_argument("--forward-outcomes-file", default="data/persona_k8_forward_outcomes.csv")
    ap.add_argument("--reference", default="live", choices=["live", "shadow_base", "shadow_conservative"])
    ap.add_argument("--high-risk-cal-pd", type=float, default=0.30)
    ap.add_argument("--out", default="capacity_risk_tradeoff.csv")
    args = ap.parse_args(argv)

    ls_path = Path(args.live_shadow_file)
    fwd_path = Path(args.forward_outcomes_file)
    if not ls_path.exists():
        sys.exit(f"ERROR: {ls_path} not found.")
    if not fwd_path.exists():
        sys.exit(f"ERROR: {fwd_path} not found.")

    ls = pd.read_csv(ls_path)
    pct_col = f"pct_of_{args.reference}_disbursed"
    excess_col = f"excess_over_{args.reference}_disbursed"
    for col in (pct_col, excess_col):
        if col not in ls.columns:
            sys.exit(f"ERROR: {col} not found in {ls_path}. Columns present: {list(ls.columns)}")
    ls["_id"] = digits(ls["msisdn"])
    ls[pct_col] = pd.to_numeric(ls[pct_col], errors="coerce")
    ls[excess_col] = pd.to_numeric(ls[excess_col], errors="coerce").fillna(0.0)
    if "cal_pd" in ls.columns:
        ls["cal_pd"] = pd.to_numeric(ls["cal_pd"], errors="coerce")

    summary_rows = []
    for key, grp in ls.groupby("_id", dropna=True):
        vals = grp[pct_col].dropna()
        if vals.empty:
            continue
        summary_rows.append({
            "_id": key,
            "ever_over": bool((vals > OVER_LIMIT_THRESHOLD).any()),
            # Total extra shillings disbursed beyond the reference, summed across every
            # matched agent-month -- not just the month that tripped "ever_over".
            "incremental_exposure": grp[excess_col].sum(),
            "cal_pd": grp["cal_pd"].dropna().iloc[0] if "cal_pd" in grp.columns and grp["cal_pd"].notna().any() else np.nan,
        })
    summary = pd.DataFrame(summary_rows).dropna(subset=["cal_pd"])
    n_checked = len(summary)
    print(f"Agents with a checkable {args.reference} disbursement ratio AND a cal_pd: {n_checked:,}")

    threshold = args.high_risk_cal_pd
    summary["is_high_risk"] = summary["cal_pd"] >= threshold
    summary["quadrant"] = [
        _quadrant_label(hr, ov) for hr, ov in zip(summary["is_high_risk"], summary["ever_over"])
    ]

    fwd = pd.read_csv(fwd_path)
    fwd["_id"] = digits(fwd["customer_msisdn"])
    merged = summary.merge(fwd.drop(columns=["customer_msisdn"]), on="_id", how="left")
    for col in ZERO_FILL_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # Finer cal_pd band per agent -- different edge sets on each side of the high-risk
    # threshold, since "30-35%" is meaningless for a lower-risk agent and vice versa.
    merged["_cal_pd_band"] = np.where(
        merged["is_high_risk"],
        pd.cut(merged["cal_pd"], bins=HIGH_RISK_CAL_PD_EDGES, labels=HIGH_RISK_CAL_PD_LABELS).astype(str),
        pd.cut(merged["cal_pd"], bins=LOWER_RISK_CAL_PD_EDGES, labels=LOWER_RISK_CAL_PD_LABELS).astype(str),
    )

    agg = {}
    for quadrant in QUADRANT_ORDER:
        sub = merged[merged["quadrant"] == quadrant]
        total_exposure = float(sub["incremental_exposure"].sum())
        total_disbursed = float(sub["fwd_new_loans_disbursed_ugx"].sum())
        total_repaid = float(sub["fwd_new_loans_repaid_ugx"].sum())
        shortfall = total_disbursed - total_repaid
        agg[quadrant] = {
            "n_agents": len(sub),
            "total_incremental_exposure": total_exposure,
            "total_fwd_new_loans_disbursed_ugx": total_disbursed,
            "total_fwd_new_loans_repaid_ugx": total_repaid,
            "shortfall": shortfall,
            "shortfall_rate_pct": round(shortfall / total_disbursed * 100, 3) if total_disbursed else float("nan"),
        }

    result = pd.DataFrame(agg).T.reindex(QUADRANT_ORDER)
    print(f"\n{'=' * 100}")
    print("Quadrant totals: incremental exposure and forward repayment shortfall")
    print("=" * 100)
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 220):
        print(result.to_string())

    print(f"\nSanity check: within-limit quadrants' incremental exposure should be ~0 by construction -- "
          f"Reference: {result.loc['Reference', 'total_incremental_exposure']:,.0f}, "
          f"Risk reference: {result.loc['Risk reference', 'total_incremental_exposure']:,.0f}")

    print(f"\n{'=' * 100}")
    print("Economic proxy: incremental shortfall, computed separately per risk population")
    print("=" * 100)
    print("(Each risk population is compared ONLY against its own within-limit baseline rate --")
    print(" lower risk vs. Reference, high risk vs. Risk reference -- never one portfolio-wide rate.)\n")

    compact_rows = {}
    reciprocal_notes = {}
    for risk_label, over_q, within_q, band_labels in [
        ("Lower risk: Capacity test", "Capacity test", "Reference", LOWER_RISK_CAL_PD_LABELS),
        ("High risk: Critical risk test", "Critical risk test", "Risk reference", HIGH_RISK_CAL_PD_LABELS),
    ]:
        over = result.loc[over_q]
        within = result.loc[within_q]
        excess_exposure = over["total_incremental_exposure"]
        forward_disbursed = over["total_fwd_new_loans_disbursed_ugx"]
        forward_repaid = over["total_fwd_new_loans_repaid_ugx"]
        actual_shortfall = over["shortfall"]

        # Coarse: one pooled baseline rate for the whole risk side.
        coarse_rate_pct = within["shortfall_rate_pct"]
        coarse_expected = (coarse_rate_pct / 100) * forward_disbursed
        coarse_incremental = actual_shortfall - coarse_expected

        # PD-band standardized: compare each over-limit borrower's shortfall against
        # reference borrowers in the SAME cal_pd band, not one pooled rate -- a 0.31 and a
        # 0.65 borrower are not comparable, and the isotonic calibration work already
        # showed realized risk changes materially across this range.
        over_mask = merged["quadrant"] == over_q
        within_mask = merged["quadrant"] == within_q
        standardized_expected, band_detail, missing_bands = _pd_band_standardized_expected_shortfall(
            merged, over_mask, within_mask, "_cal_pd_band", band_labels,
        )
        standardized_incremental = actual_shortfall - standardized_expected

        print(f"\n-- {risk_label}: per-cal_pd-band detail --")
        with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 220):
            print(band_detail.to_string(index=False))
        if missing_bands:
            print(f"  NOTE: band(s) {missing_bands} have over-limit forward lending but NO within-limit "
                  f"comparator in that band -- excluded from the standardized total (their shortfall is "
                  f"still in the coarse total above).")

        allocation = _excess_exposure_allocation(
            merged, over_mask, "_cal_pd_band", band_labels, band_detail,
            excess_exposure, standardized_incremental,
        )
        print(f"\n-- {risk_label}: excess exposure vs. incremental shortfall, by cal_pd band --")
        print("(does where the excess exposure is GOING line up with where the incremental shortfall")
        print(" is APPEARING? incremental_shortfall_pct_of_forward is the preferred per-band ratio --")
        print(" not incremental/excess, for the same mismatched-denominator reason noted below.)")
        with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 240):
            print(allocation.to_string(index=False))
        allocation.insert(0, "driver", risk_label)
        allocation.to_csv(
            str(Path(args.out).with_name(Path(args.out).stem + f"_allocation_{over_q.replace(' ', '_')}.csv")),
            index=False,
        )

        incr_over_forward_pct = (standardized_incremental / forward_disbursed * 100) if forward_disbursed else float("nan")
        incr_over_excess_pct = (standardized_incremental / excess_exposure * 100) if excess_exposure else float("nan")

        if standardized_incremental > 0 and excess_exposure:
            reciprocal = excess_exposure / standardized_incremental
            reciprocal_notes[risk_label] = f"UGX {reciprocal:,.1f} of excess lending per UGX 1 of incremental shortfall"
        else:
            reciprocal_notes[risk_label] = "no positive incremental shortfall observed"

        compact_rows[risk_label] = {
            "Excess exposure (UGX)": excess_exposure,
            "Forward new-loan disbursement (UGX)": forward_disbursed,
            "Forward repayment (UGX)": forward_repaid,
            "Actual shortfall (UGX)": actual_shortfall,
            "-- Coarse (one pooled baseline rate) --": np.nan,
            "Coarse reference shortfall rate (%)": coarse_rate_pct,
            "Coarse expected shortfall (UGX)": coarse_expected,
            "Coarse incremental shortfall (UGX)": coarse_incremental,
            "-- PD-band standardized (preferred) --": np.nan,
            "Standardized expected shortfall (UGX)": standardized_expected,
            "Standardized incremental shortfall (UGX)": standardized_incremental,
            "Std. incremental shortfall / forward exposure (%)": incr_over_forward_pct,
            "Std. incremental shortfall / excess exposure (%)": incr_over_excess_pct,
        }

    compact = pd.DataFrame(compact_rows)
    print(f"\n{'=' * 100}")
    print("Compact comparison: coarse vs. PD-band-standardized")
    print("=" * 100)
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 220):
        print(compact.to_string())

    print("\nReciprocal, using the PD-band-standardized incremental shortfall (only when positive --")
    print("'UGX X of excess lending was associated with each UGX 1 of additional forward shortfall').")
    print("Treat as a secondary, order-of-magnitude figure, not a headline number -- see caveats below:")
    for risk_label, note in reciprocal_notes.items():
        print(f"  {risk_label}: {note}")

    print(f"\n{'=' * 100}")
    print("What this does and does not establish")
    print("=" * 100)
    print("This measures whether additional exposure is ASSOCIATED WITH repayment shortfall above the\n"
          "level expected from comparable within-limit borrowers in the SAME cal_pd band. It is an\n"
          "economic-risk PROXY, not a profitability or lifetime-credit-loss estimate. In particular,\n"
          "this analysis cannot say incremental profit = revenue - incremental shortfall, because\n"
          "shortfall is not LGD, and interest/fees, funding cost, operating cost, and capital cost are\n"
          "not included here.\n\n"
          "DENOMINATOR-MISMATCH WARNING on 'incremental shortfall / excess exposure': the numerator is\n"
          "generated from the group's TOTAL forward-lending volume (tens of billions of UGX), not from\n"
          "the excess-exposure amount itself (a much smaller number). A small percentage-point\n"
          "performance difference across a very large lending base, divided by a much smaller excess-\n"
          "exposure base, can produce a ratio that LOOKS like a large share of the excess money was\n"
          "lost -- it is not that. Do not present this ratio to management as 'X% of the excess\n"
          "exposure was lost'; it is an association-based economic-intensity measure relating two\n"
          "different quantities, not a loss-causation estimate. Treat both this ratio and its\n"
          "reciprocal as secondary, order-of-magnitude figures.\n\n"
          "Still outstanding before this becomes a management-level result: maturity normalization.\n"
          "See the module docstring's MATURITY / CENSORING CAVEAT -- 'shortfall' is not yet 'loss' at\n"
          "only 42 days, and the over-limit groups' higher still-active rates mean this matters more\n"
          "for them than for the within-limit baseline.")

    result.reset_index().rename(columns={"index": "quadrant"}).to_csv(args.out, index=False)
    print(f"\nQuadrant totals written: {args.out}")
    compact.to_csv(str(Path(args.out).with_name(Path(args.out).stem + "_compact.csv")))
    print(f"Compact comparison table written: {Path(args.out).with_name(Path(args.out).stem + '_compact.csv')}")


if __name__ == "__main__":
    main()

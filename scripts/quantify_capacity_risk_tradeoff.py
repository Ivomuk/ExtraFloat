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
    portfolio-wide baseline): that population's own within-limit
    reference group's shortfall RATE, applied to the over-limit
    group's own forward-disbursed volume, gives the shortfall that
    volume would be EXPECTED to produce at the baseline rate.
    Incremental shortfall = actual - expected -- answers "how much
    more shortfall did the over-limit population generate than
    expected if the same forward-lending volume had performed at its
    OWN population's within-limit baseline rate," not "how much
    shortfall is there in total."
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
    for risk_label, over_q, within_q in [("Lower risk: Capacity test", "Capacity test", "Reference"),
                                           ("High risk: Critical risk test", "Critical risk test", "Risk reference")]:
        over = result.loc[over_q]
        within = result.loc[within_q]
        excess_exposure = over["total_incremental_exposure"]
        forward_disbursed = over["total_fwd_new_loans_disbursed_ugx"]
        forward_repaid = over["total_fwd_new_loans_repaid_ugx"]
        actual_shortfall = over["shortfall"]
        baseline_rate_pct = within["shortfall_rate_pct"]
        expected_shortfall = (baseline_rate_pct / 100) * forward_disbursed
        incremental_shortfall = actual_shortfall - expected_shortfall

        incr_over_forward_pct = (incremental_shortfall / forward_disbursed * 100) if forward_disbursed else float("nan")
        incr_over_excess_pct = (incremental_shortfall / excess_exposure * 100) if excess_exposure else float("nan")

        if incremental_shortfall > 0 and excess_exposure:
            reciprocal = excess_exposure / incremental_shortfall
            reciprocal_notes[risk_label] = f"UGX {reciprocal:,.1f} of excess lending per UGX 1 of incremental shortfall"
        else:
            reciprocal_notes[risk_label] = "no positive incremental shortfall observed"

        compact_rows[risk_label] = {
            "Excess exposure (UGX)": excess_exposure,
            "Forward new-loan disbursement (UGX)": forward_disbursed,
            "Forward repayment (UGX)": forward_repaid,
            "Actual shortfall (UGX)": actual_shortfall,
            "Reference shortfall rate (%)": baseline_rate_pct,
            "Expected shortfall at reference rate (UGX)": expected_shortfall,
            "Incremental shortfall (UGX)": incremental_shortfall,
            "Incremental shortfall / forward exposure (%)": incr_over_forward_pct,
            "Incremental shortfall / excess exposure (%)": incr_over_excess_pct,
        }

    compact = pd.DataFrame(compact_rows)
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 220):
        print(compact.to_string())

    print("\nReciprocal (only when incremental shortfall > 0 -- 'UGX X of excess lending was associated")
    print("with each UGX 1 of additional forward shortfall'):")
    for risk_label, note in reciprocal_notes.items():
        print(f"  {risk_label}: {note}")

    print(f"\n{'=' * 100}")
    print("What this does and does not establish")
    print("=" * 100)
    print("This measures whether additional exposure is ASSOCIATED WITH repayment shortfall above the\n"
          "level expected from comparable within-limit borrowers. It is an economic-risk PROXY, not a\n"
          "profitability or lifetime-credit-loss estimate. In particular, this analysis cannot say\n"
          "incremental profit = revenue - incremental shortfall, because shortfall is not LGD, and\n"
          "interest/fees, funding cost, operating cost, and capital cost are not included here.\n"
          "The 'incremental shortfall / excess exposure' ratio above is an association-based economic\n"
          "intensity measure, not a loss-causation estimate -- its numerator is forward-period\n"
          "repayment performance across a population; its denominator is excess exposure measured at\n"
          "the original observation point. Related through the same borrowers, not necessarily the\n"
          "same loans. See the module docstring's MATURITY / CENSORING CAVEAT for why 'shortfall' is\n"
          "not yet 'loss' at only 42 days.")

    result.reset_index().rename(columns={"index": "quadrant"}).to_csv(args.out, index=False)
    print(f"\nQuadrant totals written: {args.out}")
    compact.to_csv(str(Path(args.out).with_name(Path(args.out).stem + "_compact.csv")))
    print(f"Compact comparison table written: {Path(args.out).with_name(Path(args.out).stem + '_compact.csv')}")


if __name__ == "__main__":
    main()

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

WHAT IT DOES COMPUTE:
  - incremental_exposure: sum(excess_over_<reference>_disbursed) --
    the actual extra shillings disbursed beyond the recommended limit,
    for the over-limit quadrant. (Within-limit quadrants should sum to
    ~0 by construction; reported as a sanity check, not a headline.)
  - shortfall: sum(fwd_new_loans_disbursed_ugx) - sum(fwd_new_loans_repaid_ugx)
    on NEW loans originated in the forward window -- unrepaid amount,
    not yet a final loss.
  - incremental_shortfall: actual shortfall MINUS the shortfall that
    quadrant's own within-limit comparator's shortfall RATE would
    predict on the SAME forward-lending volume -- isolates the portion
    of shortfall associated with being over-limit, net of this
    population's own baseline risk level.
  - the headline ratio: incremental_shortfall / incremental_exposure --
    "shillings of extra shortfall per shilling of extra exposure
    extended beyond recommendation," separately for the lower-risk and
    high-risk populations.

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
    print("Incremental shortfall per unit of incremental exposure")
    print("=" * 100)
    print("(incremental_shortfall = actual shortfall in the over-limit quadrant MINUS what its own\n"
          " within-limit comparator's shortfall RATE would predict on the SAME forward-lending volume\n"
          " -- isolates shortfall associated with being over-limit, net of this population's baseline risk.)\n")
    for risk_label, over_q, within_q in [("Lower risk", "Capacity test", "Reference"),
                                           ("High risk", "Critical risk test", "Risk reference")]:
        over = result.loc[over_q]
        within = result.loc[within_q]
        baseline_rate = within["shortfall_rate_pct"] / 100
        expected_shortfall = baseline_rate * over["total_fwd_new_loans_disbursed_ugx"]
        incremental_shortfall = over["shortfall"] - expected_shortfall
        incremental_exposure = over["total_incremental_exposure"]
        ratio_pct = (incremental_shortfall / incremental_exposure * 100) if incremental_exposure else float("nan")
        print(f"-- {risk_label}: {over_q} vs. {within_q} baseline rate ({within['shortfall_rate_pct']:.2f}%) --")
        print(f"   Incremental exposure (extra shillings lent beyond recommendation): {incremental_exposure:,.0f}")
        print(f"   Actual shortfall:                                                  {over['shortfall']:,.0f}")
        print(f"   Expected shortfall at baseline rate on same forward volume:        {expected_shortfall:,.0f}")
        print(f"   Incremental shortfall (actual - expected):                         {incremental_shortfall:,.0f}")
        print(f"   --> {ratio_pct:.2f} shillings of incremental shortfall per 100 shillings of "
              f"incremental exposure\n")

    print("NOTE: this is repayment shortfall on NEW loans within a 42-day forward window, not a "
          "lifetime loss figure, and does not include fee/commission revenue -- a unit-economics-of-"
          "risk read, not a full revenue-vs-loss P&L. See module docstring for what would be needed "
          "to extend this to a true ROI calculation.")

    result.reset_index().rename(columns={"index": "quadrant"}).to_csv(args.out, index=False)
    print(f"\nQuadrant totals written: {args.out}")


if __name__ == "__main__":
    main()

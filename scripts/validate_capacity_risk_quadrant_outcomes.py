"""
validate_capacity_risk_quadrant_outcomes.py
=========================================
The follow-up the matched agent_category x cal_pd_band feature profile
(profile_over_limit_agents_vs_mart.py) was building toward: now that
"over-limit" agents -- in both the Diamond/lower-risk population
(driver_a) and the high-risk population (driver_b) -- show a real,
non-confounded business-capacity signal (elevated commission, cash
flow, balances vs. matched peers), the open question is what actually
HAPPENED to repayment when these agents received exposure above the
model's recommendation.

Builds the 2x2 grid:

                    Lower risk              High risk (cal_pd>=threshold)
    Within limit    Reference               Risk reference
    >110% limit     Capacity test           Critical risk test

and reports forward-window loan performance for each cell, joining the
scored population (live_shadow_vs_category_limit.csv) against the real
forward-outcomes extract (data/persona_k8_forward_outcomes.csv) on
customer_msisdn -- same digits()-based join convention as
analyze_persona_k8_forward_outcomes.py, same zero-fill rules (a
borrower with no row in the forward file is zero-filled only for
count-like columns; ratio/aging columns stay undefined, not zero,
since "no new borrowing" is not the same claim as "zero days past
due" or "0% repayment").

Headline comparisons this answers directly:
  P(bad | cal_pd>=threshold, over limit)   vs  P(bad | cal_pd>=threshold, within limit)
      -- "Critical risk test" vs "Risk reference": does the risk adjustment
         bind strongly enough despite legitimate business demand?
  P(bad | cal_pd<threshold,  over limit)   vs  P(bad | cal_pd<threshold,  within limit)
      -- "Capacity test" vs "Reference": does extra exposure to lower-risk,
         high-capacity agents actually perform worse?

"Bad" is reported two ways, since neither alone is the full picture:
  - bad_closure_rate_pct: fwd_new_loans_closed_bad_count / (closed_good + closed_bad)
    -- the same "closed_loan_bad" definition used throughout the shadow-multiplier
    calibration work, restricted to borrowers with >=1 closed new loan in the window.
  - fwd_any_bad_3dpd_rate_pct: fwd_any_bad_3dpd -- any loan (new or carried over)
    reaching >3 days past due during the window, across the WHOLE quadrant
    (not just borrowers who took a new loan) -- the "3+ DPD" measure.

Exposure-weighted loss (requires LGD/EAD) and cure time are NOT computed here --
not available in this forward-outcomes extract; noted, not silently guessed at.

Usage:
    python scripts\\validate_capacity_risk_quadrant_outcomes.py ^
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

ZERO_FILL_COLS = [
    "fwd_new_loan_count", "fwd_new_loans_disbursed_ugx", "fwd_new_loans_repaid_ugx",
    "fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count",
    "fwd_any_bad_3dpd", "fwd_any_anomaly_open", "fwd_still_active_at_window_end",
]
QUADRANT_ORDER = ["Reference", "Capacity test", "Risk reference", "Critical risk test"]

# fwd_any_bad_3dpd is a MAX across EVERY loan (new + carried-over) touching the window
# (confirmed by reading data/persona_k8_forward_outcomes_query.sql directly, step 7) --
# a borrower taking more new loans has strictly more chances to satisfy that MAX, with
# no per-loan DPD count exposed in this extract to build a true per-loan rate instead.
# This is the best available substitute: stratify by how many new loans a borrower took,
# so the gap can be read within matched loan-count bins rather than across all of them
# pooled together.
LOAN_COUNT_EDGES = [0, 1, 2, 3, 5, 10, 1_000_000]
LOAN_COUNT_LABELS = ["1", "2", "3", "4-5", "6-10", "11+"]

# fwd_worst_days_aging > 3, among STILL-ACTIVE borrowers specifically, is the right
# available proxy for "this open loan is currently running overdue, not just open and
# current" -- confirmed correctly computed as a MAX across every state row in the
# window (the SQL file's own fix-history note documents this was bug-fixed), unlike
# fwd_any_anomaly_open, which that same SQL file's header explicitly documents as
# "ever flagged during the window" and NOT a debt-transfer or still-unresolved signal --
# wrong tool for a censoring check, so it is reported separately as a secondary
# diagnostic only, never used to define "unresolved" here.
UNRESOLVED_AGING_THRESHOLD = 3


def _standardized_rate(
    df: pd.DataFrame, subset_mask: pd.Series, baseline_mask: pd.Series, value_col: str, bin_col: str,
) -> tuple[float, float, list[str], list[str]]:
    """Direct standardization: reweights baseline's per-bin rate using the SUBSET's own
    bin sizes as weights, so "does the baseline look as bad as the subset once matched
    on loan-count mix" can be answered directly. Returns
    (subset_rate, standardized_baseline_rate, bins_used, bins_missing_from_baseline).
    """
    subset_bin_n = df.loc[subset_mask, bin_col].value_counts()
    baseline_bin_n = df.loc[baseline_mask, bin_col].value_counts()
    usable = [b for b in subset_bin_n.index if baseline_bin_n.get(b, 0) > 0]
    missing = [b for b in subset_bin_n.index if baseline_bin_n.get(b, 0) == 0]
    weights = subset_bin_n.loc[usable]
    n_used = int(weights.sum())
    if n_used == 0:
        return float("nan"), float("nan"), usable, missing

    subset_rate = df.loc[subset_mask & df[bin_col].isin(usable), value_col].mean() * 100
    bin_rates = pd.Series(
        [df.loc[baseline_mask & (df[bin_col] == b), value_col].mean() for b in usable], index=usable,
    )
    standardized_baseline_rate = (bin_rates * weights).sum() / n_used * 100
    return subset_rate, standardized_baseline_rate, usable, missing


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
    ap.add_argument("--live-shadow-file", default="live_shadow_vs_category_limit.csv",
                     help="per-agent detail CSV written by check_live_shadow_vs_category_limit.py")
    ap.add_argument("--forward-outcomes-file", default="data/persona_k8_forward_outcomes.csv",
                     help="CSV exported from data/persona_k8_forward_outcomes_query.sql")
    ap.add_argument("--reference", default="live", choices=["live", "shadow_base", "shadow_conservative"])
    ap.add_argument("--high-risk-cal-pd", type=float, default=0.30,
                     help="cal_pd threshold separating 'lower risk' from 'high risk'; default 0.30 "
                          "matches the '30%%+' band boundary used elsewhere in this analysis.")
    ap.add_argument("--out", default="capacity_risk_quadrant_outcomes.csv")
    ap.add_argument("--detail-out", default="capacity_risk_quadrant_outcomes_detail.csv")
    args = ap.parse_args(argv)

    ls_path = Path(args.live_shadow_file)
    fwd_path = Path(args.forward_outcomes_file)
    if not ls_path.exists():
        sys.exit(f"ERROR: {ls_path} not found (run check_live_shadow_vs_category_limit.py first).")
    if not fwd_path.exists():
        sys.exit(f"ERROR: {fwd_path} not found.")

    # -- Collapse the (possibly multi-month) live/shadow detail to one row per agent. --
    ls = pd.read_csv(ls_path)
    pct_col = f"pct_of_{args.reference}_disbursed"
    if pct_col not in ls.columns:
        sys.exit(f"ERROR: {pct_col} not found in {ls_path}. Columns present: {list(ls.columns)}")
    ls["_id"] = digits(ls["msisdn"])
    ls[pct_col] = pd.to_numeric(ls[pct_col], errors="coerce")
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
            "cal_pd": grp["cal_pd"].dropna().iloc[0] if "cal_pd" in grp.columns and grp["cal_pd"].notna().any() else np.nan,
            "agent_category": grp["agent_category"].dropna().iloc[0]
            if "agent_category" in grp.columns and grp["agent_category"].notna().any() else None,
        })
    summary = pd.DataFrame(summary_rows)
    summary = summary.dropna(subset=["cal_pd"])
    n_checked = len(summary)
    print(f"Agents with a checkable {args.reference} disbursement ratio AND a cal_pd: {n_checked:,}")

    threshold = args.high_risk_cal_pd
    summary["is_high_risk"] = summary["cal_pd"] >= threshold
    summary["quadrant"] = [
        _quadrant_label(hr, ov) for hr, ov in zip(summary["is_high_risk"], summary["ever_over"])
    ]

    # -- Real forward-window loan outcomes -- same join/zero-fill convention as --
    # -- analyze_persona_k8_forward_outcomes.py. --
    fwd = pd.read_csv(fwd_path)
    fwd["_id"] = digits(fwd["customer_msisdn"])
    window_days = fwd["fwd_window_days"].iloc[0] if "fwd_window_days" in fwd.columns and len(fwd) else None
    window_start = fwd["fwd_window_start_exclusive"].iloc[0] if "fwd_window_start_exclusive" in fwd.columns and len(fwd) else None
    window_end = fwd["fwd_window_end"].iloc[0] if "fwd_window_end" in fwd.columns and len(fwd) else None
    print(f"Forward window: ({window_start}, {window_end}] -- {window_days} days")
    if window_days is not None and window_days < 30:
        print(f"WARNING: only {window_days} days of forward data -- treat as an early, directional "
              f"read, not a final verdict.")

    merged = summary.merge(fwd.drop(columns=["customer_msisdn"]), on="_id", how="left")
    # Captured BEFORE zero-fill below -- fwd_new_loan_count.notna() would always be True
    # afterward since the fillna makes every row non-null (same pitfall this comment
    # documents in analyze_persona_k8_forward_outcomes.py).
    merged["_had_fwd_activity"] = merged["fwd_new_loan_count"].notna()
    n_matched_any_activity = int(merged["_had_fwd_activity"].sum())
    print(f"Matched {n_matched_any_activity:,} / {n_checked:,} agents ({n_matched_any_activity / n_checked * 100:.1f}%) "
          f"to ANY forward-window loan-state activity.\n")

    for col in ZERO_FILL_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    merged["_loan_count_bin"] = pd.cut(merged["fwd_new_loan_count"], bins=LOAN_COUNT_EDGES, labels=LOAN_COUNT_LABELS)
    merged["_is_unresolved_aging"] = (
        (merged["fwd_still_active_at_window_end"] == 1) & (merged["fwd_worst_days_aging"] > UNRESOLVED_AGING_THRESHOLD)
    )
    merged["_is_bad_or_unresolved"] = (merged["fwd_new_loans_closed_bad_count"] > 0) | merged["_is_unresolved_aging"]

    rows = []
    for quadrant in QUADRANT_ORDER:
        sub = merged[merged["quadrant"] == quadrant]
        n = len(sub)
        row = {"quadrant": quadrant, "n_agents": n}
        if n == 0:
            rows.append(row)
            continue
        row["pct_with_fwd_activity"] = round(sub["_had_fwd_activity"].mean() * 100, 1)
        took_new_loan = sub["fwd_new_loan_count"] > 0 if "fwd_new_loan_count" in sub.columns else pd.Series(False, index=sub.index)
        row["pct_took_new_loan"] = round(took_new_loan.mean() * 100, 1)

        if "fwd_any_bad_3dpd" in sub.columns:
            row["fwd_any_bad_3dpd_rate_pct"] = round(sub["fwd_any_bad_3dpd"].mean() * 100, 2)
        if "fwd_any_anomaly_open" in sub.columns:
            row["fwd_any_anomaly_open_rate_pct"] = round(sub["fwd_any_anomaly_open"].mean() * 100, 2)
        if "fwd_still_active_at_window_end" in sub.columns:
            row["fwd_still_active_rate_pct"] = round(sub["fwd_still_active_at_window_end"].mean() * 100, 2)

        if {"fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count"} <= set(sub.columns):
            n_good = int(sub["fwd_new_loans_closed_good_count"].sum())
            n_bad = int(sub["fwd_new_loans_closed_bad_count"].sum())
            n_closed = n_good + n_bad
            row["n_closed_new_loans"] = n_closed
            row["bad_closure_rate_pct"] = round(n_bad / n_closed * 100, 2) if n_closed else float("nan")

        if "fwd_new_loans_repayment_ratio" in sub.columns and took_new_loan.any():
            row["fwd_new_loans_repayment_ratio_median_among_takers"] = round(
                sub.loc[took_new_loan, "fwd_new_loans_repayment_ratio"].median(), 4
            )
        if "fwd_worst_days_aging" in sub.columns:
            row["fwd_worst_days_aging_median"] = sub["fwd_worst_days_aging"].median()
        # Supplementary, censoring-robust borrower-level rate -- bad_closure_rate_pct
        # above only ever looks at CLOSED new loans; this adds back borrowers whose
        # loan hasn't closed yet but is already running overdue (see
        # UNRESOLVED_AGING_THRESHOLD), over the WHOLE quadrant as the denominator
        # (not just those with a closed loan) -- a check against the gap between
        # pct_with_fwd_activity and n_closed_new_loans silently favoring whichever
        # quadrant has more loans still open.
        row["bad_or_unresolved_rate_pct"] = round(sub["_is_bad_or_unresolved"].mean() * 100, 2)
        rows.append(row)

    result = pd.DataFrame(rows).set_index("quadrant").reindex(QUADRANT_ORDER)
    print("=" * 100)
    print("2x2 grid: lower risk / high risk  x  within limit / over 110% of recommended limit")
    print("=" * 100)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(result.to_string())

    print(f"\n{'=' * 100}")
    print("Headline comparisons")
    print("=" * 100)
    for bad_col, bad_label in [("bad_closure_rate_pct", "bad-closure rate"),
                                ("fwd_any_bad_3dpd_rate_pct", "3+ DPD rate")]:
        if bad_col not in result.columns:
            continue
        rr = result.loc["Risk reference", bad_col]
        crt = result.loc["Critical risk test", bad_col]
        ref = result.loc["Reference", bad_col]
        ct = result.loc["Capacity test", bad_col]
        print(f"\n-- {bad_label} --")
        print(f"  High risk:   Critical risk test {crt:.2f}%  vs.  Risk reference {rr:.2f}%  "
              f"(delta {crt - rr:+.2f} pp)")
        print(f"  Lower risk:  Capacity test     {ct:.2f}%  vs.  Reference       {ref:.2f}%  "
              f"(delta {ct - ref:+.2f} pp)")

    # -- Is the 3+DPD gap just a loan-volume artifact? Stratify by how many new loans --
    # -- a borrower took, then standardize the within-limit comparator to the --
    # -- over-limit group's OWN loan-count mix -- if the gap survives, volume alone --
    # -- doesn't explain it. --
    print(f"\n{'=' * 100}")
    print("Is the 3+DPD gap explained by loan volume? Rate by loan-count bin")
    print("=" * 100)
    bin_pivot = merged.groupby(["quadrant", "_loan_count_bin"], observed=True)["fwd_any_bad_3dpd"].mean() * 100
    bin_pivot = bin_pivot.unstack("_loan_count_bin").reindex(QUADRANT_ORDER).reindex(columns=LOAN_COUNT_LABELS)
    with pd.option_context("display.float_format", "{:.1f}".format):
        print(bin_pivot.to_string())

    print("\n-- Standardized comparison: within-limit rate, reweighted to the over-limit "
          "group's own loan-count mix --")
    for risk_label, over_q, within_q in [("Lower risk", "Capacity test", "Reference"),
                                           ("High risk", "Critical risk test", "Risk reference")]:
        over_rate, standardized_within_rate, bins_used, missing = _standardized_rate(
            merged, merged["quadrant"] == over_q, merged["quadrant"] == within_q,
            "fwd_any_bad_3dpd", "_loan_count_bin",
        )
        raw_within_rate = result.loc[within_q, "fwd_any_bad_3dpd_rate_pct"]
        print(f"  {risk_label}: {over_q} {over_rate:.2f}%  vs.  {within_q} standardized to same "
              f"loan-count mix {standardized_within_rate:.2f}%  (raw {within_q} was {raw_within_rate:.2f}%)")
        if missing:
            print(f"    NOTE: loan-count bin(s) {missing} present in {over_q} have no {within_q} "
                  f"agents to standardize against -- those agents excluded from this comparison only.")

    # -- Censoring check: among STILL-ACTIVE borrowers only, are the over-limit --
    # -- groups' open loans actually running MORE overdue, or just as current as --
    # -- the within-limit groups' open loans? fwd_worst_days_aging, not --
    # -- fwd_any_anomaly_open -- see module docstring / constant comments for why. --
    print(f"\n{'=' * 100}")
    print("Still-active (unresolved-exposure) check -- among borrowers with an open loan at window end")
    print("=" * 100)
    active_rows = []
    for quadrant in QUADRANT_ORDER:
        sub = merged[(merged["quadrant"] == quadrant) & (merged["fwd_still_active_at_window_end"] == 1)]
        n = len(sub)
        if n == 0:
            active_rows.append({"quadrant": quadrant, "n_still_active": 0})
            continue
        active_rows.append({
            "quadrant": quadrant,
            "n_still_active": n,
            "pct_worst_aging_gt3": round((sub["fwd_worst_days_aging"] > UNRESOLVED_AGING_THRESHOLD).mean() * 100, 1),
            "worst_aging_median": sub["fwd_worst_days_aging"].median(),
            "fwd_any_anomaly_open_rate_pct (secondary -- 'ever flagged', not 'unresolved')":
                round(sub["fwd_any_anomaly_open"].mean() * 100, 1),
        })
    active_result = pd.DataFrame(active_rows).set_index("quadrant")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(active_result.to_string())

    print(f"\nNOTE: exposure-weighted loss (requires LGD/EAD) and cure time are not computed here -- "
          f"not available in {fwd_path.name}.")

    result.reset_index().to_csv(args.out, index=False)
    print(f"\nQuadrant summary written: {args.out}")

    detail_cols = ["_id", "quadrant", "cal_pd", "agent_category", "ever_over"] + \
        [c for c in ["fwd_new_loan_count", "_loan_count_bin", "fwd_any_bad_3dpd", "fwd_any_anomaly_open",
                      "fwd_still_active_at_window_end", "fwd_new_loans_closed_good_count",
                      "fwd_new_loans_closed_bad_count", "fwd_new_loans_repayment_ratio",
                      "fwd_worst_days_aging", "_is_unresolved_aging", "_is_bad_or_unresolved"]
         if c in merged.columns]
    merged[detail_cols].rename(columns={"_id": "msisdn"}).to_csv(args.detail_out, index=False)
    print(f"Per-agent detail written: {args.detail_out}")


if __name__ == "__main__":
    main()

"""
evaluate_champion_vs_challenger_decision.py
=============================================
The first concrete champion (live discrete tier policy) vs. challenger
(C3 shadow continuous multiplier, base scenario) comparison, built from
one logged scoring cycle in output/shadow_multiplier_log.csv joined
against the real forward-outcomes extract
(data/persona_k8_forward_outcomes.csv).

WHAT THIS ANSWERS: for the agents the challenger would treat
differently from the champion -- cohort_base == "uplift" (C3 wants a
materially HIGHER limit than live) or "tighten" (C3 wants a materially
LOWER limit) -- does their REALIZED subsequent loan performance, at
the limit they actually received (the live one; C3 is shadow-only),
look consistent with the challenger's judgment, relative to the
"neutral" cohort (where the two policies roughly agree) in the SAME
cal_pd band?

WHAT THIS DOES NOT ANSWER (same causal boundary as the logger itself):
this cannot say what would have happened if these agents had actually
been LENT UNDER the challenger's limit -- every agent here was scored
by the live tier policy and received the live limit. It can only ask
whether the agents the challenger would flag already show a
risk/capacity signal consistent with its judgment, under the exposure
they already had. That is strong policy-validation evidence, not
causal deployment evidence -- a controlled pilot is still the only way
to answer the causal question. See log_shadow_multiplier_cycle.py's
docstring for the same split.

REUSED METHODOLOGY (not re-derived -- same vetted functions as the two
scripts this is built on, adapted to compare cohort_base groups instead
of live-tier over/under-limit quadrants):
  - PD-band-standardized dollar shortfall: quantify_capacity_risk_tradeoff.py's
    _pd_band_standardized_expected_shortfall -- reweights the neutral
    cohort's per-cal_pd-band shortfall RATE using the uplift/tighten
    cohort's OWN forward-disbursed dollar volume in that band.
  - Loan-count-bin-standardized 3+DPD rate: validate_capacity_risk_quadrant_outcomes.py's
    _standardized_rate -- same direct-standardization technique, same
    caveat about fwd_any_bad_3dpd being a MAX across every loan
    touching the window (more loans = more chances to hit the max).
  - Same zero-fill rules, same "ever flagged" vs. "still unresolved"
    distinction for censoring, same maturity/censoring cavein as both
    source scripts: this reads forward RESULTS, not lifetime losses.

IMPORTANT ALIGNMENT CAVEAT this script cannot verify automatically: the
logged cycle (keyed by run_id = scored_at) and the forward-outcomes
file (keyed to whatever snapshot date data/persona_k8_forward_outcomes_query.sql
was run against) are two independently-dated artifacts. This script
prints both dates so a human can confirm they correspond to the SAME
underlying population snapshot -- it does not and cannot enforce that
match itself.

Usage:
    python scripts\\evaluate_champion_vs_challenger_decision.py ^
        --shadow-log output\\shadow_multiplier_log.csv ^
        --forward-outcomes-file data\\persona_k8_forward_outcomes.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.borrower_persona_clustering import digits  # noqa: E402

COHORT_ORDER = ["tighten", "neutral", "uplift"]

ZERO_FILL_COLS = [
    "fwd_new_loan_count", "fwd_new_loans_disbursed_ugx", "fwd_new_loans_repaid_ugx",
    "fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count",
    "fwd_any_bad_3dpd", "fwd_any_anomaly_open", "fwd_still_active_at_window_end",
]

# Same bands/threshold as quantify_capacity_risk_tradeoff.py and
# validate_capacity_risk_quadrant_outcomes.py -- restated independently per this
# session's convention (scripts/ don't import from each other).
LOWER_RISK_CAL_PD_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
LOWER_RISK_CAL_PD_LABELS = ["<2%", "2-5%", "5-10%", "10-20%", "20-30%"]
HIGH_RISK_CAL_PD_EDGES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 1.0]
HIGH_RISK_CAL_PD_LABELS = ["30-35%", "35-40%", "40-45%", "45-50%", "50-60%", "60%+"]

LOAN_COUNT_EDGES = [0, 1, 2, 3, 5, 10, 1_000_000]
LOAN_COUNT_LABELS = ["1", "2", "3", "4-5", "6-10", "11+"]
UNRESOLVED_AGING_THRESHOLD = 3


def _pd_band_standardized_expected_shortfall(
    df: pd.DataFrame, subset_mask: pd.Series, baseline_mask: pd.Series, band_col: str, band_labels: list[str],
) -> tuple[float, pd.DataFrame, list[str]]:
    """Dollar-weighted standardization, identical logic to
    quantify_capacity_risk_tradeoff.py's function of the same name: E[Shortfall_subset]
    = sum_j (ForwardDisbursed_subset,j x ShortfallRate_baseline,j), using the subset's
    OWN forward-disbursed dollar volume per band, the baseline's shortfall RATE in
    that same band.
    """
    rows = []
    missing = []
    for band in band_labels:
        subset_band_mask = subset_mask & (df[band_col] == band)
        baseline_band_mask = baseline_mask & (df[band_col] == band)
        subset_disbursed = df.loc[subset_band_mask, "fwd_new_loans_disbursed_ugx"].sum()
        subset_repaid = df.loc[subset_band_mask, "fwd_new_loans_repaid_ugx"].sum()
        subset_actual_shortfall = subset_disbursed - subset_repaid
        baseline_disbursed = df.loc[baseline_band_mask, "fwd_new_loans_disbursed_ugx"].sum()
        baseline_shortfall = (
            baseline_disbursed - df.loc[baseline_band_mask, "fwd_new_loans_repaid_ugx"].sum()
        )
        if subset_disbursed == 0:
            continue
        if baseline_disbursed == 0:
            missing.append(band)
            rows.append({"cal_pd_band": band, "n_subset_agents": int(subset_band_mask.sum()),
                         "subset_forward_disbursed": subset_disbursed, "subset_actual_shortfall": subset_actual_shortfall,
                         "n_baseline_agents": 0, "baseline_shortfall_rate_pct": float("nan"),
                         "expected_shortfall": float("nan"), "band_incremental_shortfall": float("nan")})
            continue
        baseline_rate = baseline_shortfall / baseline_disbursed
        expected = baseline_rate * subset_disbursed
        rows.append({
            "cal_pd_band": band, "n_subset_agents": int(subset_band_mask.sum()),
            "subset_forward_disbursed": subset_disbursed, "subset_actual_shortfall": subset_actual_shortfall,
            "n_baseline_agents": int(baseline_band_mask.sum()),
            "baseline_shortfall_rate_pct": round(baseline_rate * 100, 3), "expected_shortfall": expected,
            "band_incremental_shortfall": subset_actual_shortfall - expected,
        })
    detail = pd.DataFrame(rows, columns=["cal_pd_band", "n_subset_agents", "subset_forward_disbursed",
                                          "subset_actual_shortfall", "n_baseline_agents",
                                          "baseline_shortfall_rate_pct", "expected_shortfall",
                                          "band_incremental_shortfall"])
    total_expected = detail["expected_shortfall"].sum(skipna=True) if len(detail) else 0.0
    return total_expected, detail, missing


def _standardized_rate(
    df: pd.DataFrame, subset_mask: pd.Series, baseline_mask: pd.Series, value_col: str, bin_col: str,
) -> tuple[float, float, list[str], list[str]]:
    """Direct standardization, identical logic to
    validate_capacity_risk_quadrant_outcomes.py's function of the same name: reweights
    the baseline's per-bin rate using the SUBSET's own bin sizes as weights.
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shadow-log", default="output/shadow_multiplier_log.csv")
    ap.add_argument("--run-id", default=None,
                     help="which logged cycle (run_id = scored_at) to evaluate; defaults to the most "
                          "recent run_id present in --shadow-log")
    ap.add_argument("--scenario", default="base", choices=["base", "conservative"],
                     help="base is the primary challenger; conservative is a secondary sensitivity "
                          "(prior analysis on this branch showed it produces materially more migration)")
    ap.add_argument("--forward-outcomes-file", default="data/persona_k8_forward_outcomes.csv")
    ap.add_argument("--high-risk-cal-pd", type=float, default=0.30)
    ap.add_argument("--out", default="champion_vs_challenger_decision.csv")
    ap.add_argument("--detail-out", default="champion_vs_challenger_decision_detail.csv")
    args = ap.parse_args(argv)

    log_path = Path(args.shadow_log)
    fwd_path = Path(args.forward_outcomes_file)
    if not log_path.exists():
        sys.exit(f"ERROR: {log_path} not found -- run log_shadow_multiplier_cycle.py first.")
    if not fwd_path.exists():
        sys.exit(f"ERROR: {fwd_path} not found.")

    cohort_col = f"cohort_{args.scenario}"
    log = pd.read_csv(log_path, dtype={"run_id": str, "msisdn": str}, low_memory=False)
    if cohort_col not in log.columns:
        sys.exit(f"ERROR: {cohort_col} not found in {log_path}. Columns present: {list(log.columns)}")

    run_id = args.run_id or log["run_id"].max()
    cycle = log[log["run_id"] == run_id].copy()
    if cycle.empty:
        sys.exit(f"ERROR: run_id {run_id!r} not found in {log_path}. "
                  f"Available run_ids: {sorted(log['run_id'].unique())}")
    cycle_date = cycle["cycle_date"].iloc[0] if "cycle_date" in cycle.columns else "(unknown)"
    print(f"Evaluating logged cycle: run_id={run_id}  cycle_date={cycle_date}  n_agents={len(cycle):,}")
    print(f"Scenario: {args.scenario} (challenger)")

    n_unclassified = int((cycle[cohort_col] == "unclassified").sum())
    cycle = cycle[cycle[cohort_col].isin(COHORT_ORDER)].copy()
    print(f"Excluded {n_unclassified:,} agent(s) with {cohort_col} == 'unclassified' (shadow failed, "
          f"live limit not positive, or no post-transition value) -- {len(cycle):,} remain for comparison.\n")

    cycle["cal_pd"] = pd.to_numeric(cycle["cal_pd"], errors="coerce")
    cycle = cycle.dropna(subset=["cal_pd"])

    fwd = pd.read_csv(fwd_path)
    fwd["_id"] = digits(fwd["customer_msisdn"])
    window_days = fwd["fwd_window_days"].iloc[0] if "fwd_window_days" in fwd.columns and len(fwd) else None
    window_start = fwd["fwd_window_start_exclusive"].iloc[0] if "fwd_window_start_exclusive" in fwd.columns and len(fwd) else None
    window_end = fwd["fwd_window_end"].iloc[0] if "fwd_window_end" in fwd.columns and len(fwd) else None
    print(f"Forward-outcomes window: ({window_start}, {window_end}] -- {window_days} days")
    print(f"ALIGNMENT CHECK (manual): confirm this window corresponds to the same population snapshot "
          f"as cycle_date={cycle_date} above -- this script cannot verify that automatically.\n")
    if window_days is not None and window_days < 30:
        print(f"WARNING: only {window_days} days of forward data -- treat as an early, directional "
              f"read, not a final verdict.\n")

    merged = cycle.merge(fwd.drop(columns=["customer_msisdn"]), left_on="msisdn", right_on="_id", how="left")
    merged["_had_fwd_activity"] = merged["fwd_new_loan_count"].notna()
    n_matched = int(merged["_had_fwd_activity"].sum())
    print(f"Matched {n_matched:,} / {len(merged):,} agents ({n_matched / len(merged) * 100:.1f}%) to ANY "
          f"forward-window loan-state activity.\n")

    for col in ZERO_FILL_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    merged["_is_high_risk"] = merged["cal_pd"] >= args.high_risk_cal_pd
    merged["_cal_pd_band"] = np.where(
        merged["_is_high_risk"],
        pd.cut(merged["cal_pd"], bins=HIGH_RISK_CAL_PD_EDGES, labels=HIGH_RISK_CAL_PD_LABELS).astype(str),
        pd.cut(merged["cal_pd"], bins=LOWER_RISK_CAL_PD_EDGES, labels=LOWER_RISK_CAL_PD_LABELS).astype(str),
    )
    merged["_loan_count_bin"] = pd.cut(merged["fwd_new_loan_count"], bins=LOAN_COUNT_EDGES, labels=LOAN_COUNT_LABELS)
    merged["_is_unresolved_aging"] = (
        (merged["fwd_still_active_at_window_end"] == 1) & (merged["fwd_worst_days_aging"] > UNRESOLVED_AGING_THRESHOLD)
    )
    merged["_is_bad_or_unresolved"] = (merged["fwd_new_loans_closed_bad_count"] > 0) | merged["_is_unresolved_aging"]

    pct_change_col = f"pct_change_{args.scenario}"
    post_col = f"shadow_limit_post_transition_{args.scenario}"

    rows = []
    for cohort in COHORT_ORDER:
        sub = merged[merged[cohort_col] == cohort]
        n = len(sub)
        row = {"cohort": cohort, "n_agents": n}
        if n == 0:
            rows.append(row)
            continue
        row["mean_pct_change"] = round(sub[pct_change_col].mean() * 100, 1)
        row["prospective_exposure_change_ugx"] = (sub[post_col] - sub["assigned_limit"]).sum()
        row["pct_with_fwd_activity"] = round(sub["_had_fwd_activity"].mean() * 100, 1)
        took_new_loan = sub["fwd_new_loan_count"] > 0
        row["pct_took_new_loan"] = round(took_new_loan.mean() * 100, 1)
        row["fwd_any_bad_3dpd_rate_pct"] = round(sub["fwd_any_bad_3dpd"].mean() * 100, 2)
        row["fwd_still_active_rate_pct"] = round(sub["fwd_still_active_at_window_end"].mean() * 100, 2)
        n_good = int(sub["fwd_new_loans_closed_good_count"].sum())
        n_bad = int(sub["fwd_new_loans_closed_bad_count"].sum())
        n_closed = n_good + n_bad
        row["n_closed_new_loans"] = n_closed
        row["bad_closure_rate_pct"] = round(n_bad / n_closed * 100, 2) if n_closed else float("nan")
        row["bad_or_unresolved_rate_pct"] = round(sub["_is_bad_or_unresolved"].mean() * 100, 2)
        rows.append(row)

    headline = pd.DataFrame(rows).set_index("cohort").reindex(COHORT_ORDER)
    print("=" * 100)
    print(f"Headline: {cohort_col} comparison (live limit received by all agents; C3 base is shadow-only)")
    print("=" * 100)
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.float_format", "{:,.2f}".format):
        print(headline.to_string())

    print(f"\n{'=' * 100}")
    print("PD-band-standardized dollar shortfall and loan-count-bin-standardized 3+DPD rate")
    print("(each challenger-divergent cohort compared ONLY against 'neutral' -- never one pooled rate)")
    print("=" * 100)

    compact_rows = {}
    for label, subset_cohort in [("uplift (C3 wants MORE than live)", "uplift"),
                                  ("tighten (C3 wants LESS than live)", "tighten")]:
        if subset_cohort not in headline.index or pd.isna(headline.loc[subset_cohort, "n_agents"]) \
                or headline.loc[subset_cohort, "n_agents"] == 0:
            print(f"\n-- {label}: no agents in this cohort for this cycle -- skipped.")
            continue
        subset_mask = merged[cohort_col] == subset_cohort
        baseline_mask = merged[cohort_col] == "neutral"

        band_labels_all = LOWER_RISK_CAL_PD_LABELS + HIGH_RISK_CAL_PD_LABELS
        standardized_expected, band_detail, missing_bands = _pd_band_standardized_expected_shortfall(
            merged, subset_mask, baseline_mask, "_cal_pd_band", band_labels_all,
        )
        subset_disbursed = float(merged.loc[subset_mask, "fwd_new_loans_disbursed_ugx"].sum())
        subset_repaid = float(merged.loc[subset_mask, "fwd_new_loans_repaid_ugx"].sum())
        actual_shortfall = subset_disbursed - subset_repaid
        standardized_incremental = actual_shortfall - standardized_expected

        print(f"\n-- {label}: cal_pd-band shortfall detail (vs. neutral baseline) --")
        with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 220):
            print(band_detail.to_string(index=False))
        if missing_bands:
            print(f"  NOTE: band(s) {missing_bands} have {subset_cohort} forward lending but NO "
                  f"'neutral' comparator in that band -- excluded from the standardized total.")

        rate_subset, rate_baseline_std, bins_used, bins_missing = _standardized_rate(
            merged, subset_mask, baseline_mask, "fwd_any_bad_3dpd", "_loan_count_bin",
        )
        print(f"\n-- {label}: 3+DPD rate, standardized to {subset_cohort}'s own loan-count mix --")
        print(f"  {subset_cohort}: {rate_subset:.2f}%   vs.   neutral (standardized): {rate_baseline_std:.2f}%  "
              f"(delta {rate_subset - rate_baseline_std:+.2f} pp)")
        if bins_missing:
            print(f"  NOTE: loan-count bin(s) {bins_missing} present in {subset_cohort} have no 'neutral' "
                  f"agents to standardize against -- excluded from this comparison only.")

        incr_pct_of_forward = (standardized_incremental / subset_disbursed * 100) if subset_disbursed else float("nan")
        compact_rows[label] = {
            "n_agents": int(subset_mask.sum()),
            "Forward new-loan disbursement (UGX)": subset_disbursed,
            "Forward repayment (UGX)": subset_repaid,
            "Actual shortfall (UGX)": actual_shortfall,
            "Standardized expected shortfall vs. neutral (UGX)": standardized_expected,
            "Standardized incremental shortfall (UGX)": standardized_incremental,
            "Incremental shortfall / forward exposure (%)": incr_pct_of_forward,
            "3+DPD rate, this cohort (%)": rate_subset,
            "3+DPD rate, neutral standardized to this cohort's loan-count mix (%)": rate_baseline_std,
            "3+DPD rate delta (pp)": rate_subset - rate_baseline_std,
        }

    compact = pd.DataFrame(compact_rows)
    if len(compact.columns):
        print(f"\n{'=' * 100}")
        print("Compact comparison")
        print("=" * 100)
        with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 240):
            print(compact.to_string())

    print(f"\n{'=' * 100}")
    print("What this does and does not establish")
    print("=" * 100)
    print("A standardized incremental shortfall at or below zero, and a 3+DPD rate no worse than the\n"
          "PD-matched 'neutral' cohort, means the agents the challenger would flag do not show a WORSE\n"
          "forward performance than their risk-matched peers under the exposure they ALREADY received.\n"
          "That supports the challenger's judgment as consistent with observed behavior -- it is not\n"
          "proof that lending them MORE (uplift) or LESS (tighten) under C3 would itself produce that\n"
          "same relative performance, since every agent here was scored and lent under the LIVE policy,\n"
          "not the challenger's. The only way to test that causally is a controlled pilot.\n\n"
          "Same maturity/censoring caveat as quantify_capacity_risk_tradeoff.py: 'shortfall' is forward-\n"
          "window-to-date, not a final loss figure, and still-active loans may simply not have had time\n"
          "to repay yet. See the still-active rate and bad_or_unresolved_rate_pct columns above as a\n"
          "partial check; this is an early, directional read at this window length, not a final verdict.")

    headline.reset_index().to_csv(args.out, index=False)
    print(f"\nHeadline table written: {args.out}")
    if len(compact.columns):
        compact.to_csv(str(Path(args.out).with_name(Path(args.out).stem + "_compact.csv")))
        print(f"Compact comparison written: {Path(args.out).with_name(Path(args.out).stem + '_compact.csv')}")

    detail_cols = ["msisdn", "run_id", "cycle_date", cohort_col, pct_change_col, "cal_pd", "_cal_pd_band",
                   "agent_category", "risk_tier"] + \
        [c for c in ["fwd_new_loan_count", "_loan_count_bin", "fwd_any_bad_3dpd", "fwd_any_anomaly_open",
                      "fwd_still_active_at_window_end", "fwd_new_loans_closed_good_count",
                      "fwd_new_loans_closed_bad_count", "fwd_new_loans_repayment_ratio",
                      "fwd_worst_days_aging", "_is_unresolved_aging", "_is_bad_or_unresolved"]
         if c in merged.columns]
    detail_cols = [c for c in dict.fromkeys(detail_cols) if c in merged.columns]
    merged[detail_cols].to_csv(args.detail_out, index=False)
    print(f"Per-agent detail written: {args.detail_out}")


if __name__ == "__main__":
    main()

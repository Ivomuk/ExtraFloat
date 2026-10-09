"""
DEPRECATED after cadence diagnostics (scripts/analyze_loan_frequency_for_snapshot_cadence.py)
showed 91.3% of consecutive loan pairs occur within the same calendar month. The loan-to-loan
Δfundamentals estimand this script implements was superseded by business-state-period analysis
(scripts/analyze_business_state_exposure_variation.py, scripts/analyze_business_state_evolution.py).
Retained for audit/reproducibility -- not part of the active Analysis 3 pipeline.

----------------------------------------------------------------------------------------------

analyze_episode_agent_transitions.py
========================================
Deliverable 4 of the episode-grain rebuild of Analysis 3 ("Level 2:
consecutive within-agent" evidence). The question this script answers,
stated first and deliberately NOT framed as "prove underfunding":

    When the same agent's business scale changes, how does historical
    credit exposure move? Then: what subsequent same-loan performance
    accompanies those exposure/scale combinations?

POPULATION BASIS: transitions are built from ALL episodes with usable
exposure/fundamental information, ordered per-agent by disbursement_ts
(preferred) -> target_loan_seq (deterministic tiebreak) -> loan_date
(fallback only if disbursement_ts is unavailable) -- NOT restricted to
"eligible" episodes at the construction stage. Outcome eligibility gates
only the *performance* statistic, attached afterward: the first question
above doesn't need an outcome at all; only the second does. Every grid
cell therefore reports THREE figures, never blended into one ambiguous
"bad rate among eligible": n_transitions_total, n_transitions_outcome_
eligible, bad_rate_current_episode.

EXPLICIT, FROZEN outcome-attachment rule -- do not average episode j-1 and
j, do not leave this ambiguous:

    OutcomeTransition_{j-1,j} = bad_state_3dpd_30d(j), attached ONLY when
    label_eligible_30d(j) == 1.

L_j (episode j's own exposure) is the thing being evaluated, so its own
outcome is the economically relevant one:
    (X_{j-1}, L_{j-1}) -> (X_j, L_j) -> Y_j.

TWO INDEPENDENT 3x3 direction grids (float and commission are never
collapsed into one "fundamentals" axis -- an agent whose float grew while
commission fell is real, useful information a combined axis would hide):
ExposureDirection x FloatDirection, ExposureDirection x CommissionDirection.
Direction bands (named constants below, adjustable): ratio > 1.10 = "up",
0.90-1.10 = "flat", < 0.90 = "down".

Cell naming is neutral (e.g. exposure_down_float_up), never a loaded
hypothesis name like "policy-suppression candidate." The printed summary
calls out two specific cells BY DESCRIPTION for each grid:
  - exposure_flat_*_up: "agent's business grew while exposure stayed fixed
    -- a candidate limit-stickiness signature."
  - exposure_down_*_up: "exposure fell while business grew -- candidate
    population for investigating a possible financing constraint or policy
    change, among other explanations; does not by itself establish policy
    suppression."
Those two cells, for BOTH grids, additionally get a duration-stratified
breakout (<=30 / 31-90 / >90 days between loans), since a given business-
scale change over 3 days means something very different from the same
change over 180 days.

CAUSAL CAVEAT (printed explicitly, every run): a within-agent performance
difference across transition cells is observational, not causal. Report as
"agents who experienced this transition showed X difference in subsequent
performance" -- never "this transition caused X."

Restated independently (one-way scripts/ layering convention).

Usage:
    python scripts\\analyze_episode_agent_transitions.py ^
        --episode-dataset loan_episode_capacity_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FUNDAMENTALS_FOR_GRIDS = {"float_activity_value_1m": "float", "commission": "commission"}
SECONDARY_FUNDAMENTALS = {"cust_1m": "customer", "average_balance": "balance"}

DIRECTION_UP_THRESHOLD = 1.10
DIRECTION_DOWN_THRESHOLD = 0.90
DIRECTION_LABELS = ["down", "flat", "up"]

DURATION_BAND_EDGES = [-0.5, 30.5, 90.5, np.inf]
DURATION_BAND_LABELS = ["<=30", "31-90", ">90"]

REQUIRED_COLS = [
    "agent_msisdn", "loan_date", "disbursement_ts", "target_loan_seq", "disbursement_amount_ugx",
    "float_activity_value_1m", "commission", "cust_1m", "average_balance",
    "bad_state_3dpd_30d", "label_eligible_30d",
]

CALLOUT_CELLS = [("flat", "up"), ("down", "up")]


def _ratio(cur, prev):
    if pd.isna(prev) or prev <= 0 or pd.isna(cur):
        return np.nan
    return cur / prev


def _direction(ratio):
    if pd.isna(ratio):
        return np.nan
    if ratio > DIRECTION_UP_THRESHOLD:
        return "up"
    if ratio < DIRECTION_DOWN_THRESHOLD:
        return "down"
    return "flat"


def _exposure_intensity(exposure, fundamental):
    if pd.isna(exposure) or pd.isna(fundamental) or fundamental <= 0:
        return np.nan, np.nan, "missing_denominator"
    ei = exposure / fundamental
    log_ei = np.log(exposure) - np.log(fundamental)
    return ei, log_ei, "ok"


def order_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """disbursement_ts preferred, loan_date fallback, target_loan_seq as
    deterministic tiebreak -- same ordering priority used throughout this
    rebuild."""
    df = df.copy()
    df["_sort_ts"] = pd.to_datetime(df["disbursement_ts"], errors="coerce")
    fallback = pd.to_datetime(df["loan_date"], errors="coerce")
    df["_sort_ts"] = df["_sort_ts"].fillna(fallback)
    return df.sort_values(["agent_msisdn", "_sort_ts", "target_loan_seq"]).reset_index(drop=True)


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per consecutive (j-1, j) episode pair, per agent. Every
    episode with a parseable sort timestamp participates -- not restricted
    to outcome-eligible episodes (see module docstring)."""
    ordered = order_episodes(df)
    ordered = ordered[ordered["_sort_ts"].notna()]
    rows = []
    for agent, g in ordered.groupby("agent_msisdn"):
        g = g.reset_index(drop=True)
        for j in range(1, len(g)):
            prev, cur = g.iloc[j - 1], g.iloc[j]
            exposure_change_ratio = _ratio(cur["disbursement_amount_ugx"], prev["disbursement_amount_ugx"])
            float_change_ratio = _ratio(cur["float_activity_value_1m"], prev["float_activity_value_1m"])
            commission_change_ratio = _ratio(cur["commission"], prev["commission"])
            customer_change_ratio = _ratio(cur["cust_1m"], prev["cust_1m"])
            balance_change_ratio = _ratio(cur["average_balance"], prev["average_balance"])

            days_between = (cur["_sort_ts"] - prev["_sort_ts"]).days if pd.notna(cur["_sort_ts"]) and pd.notna(prev["_sort_ts"]) else np.nan
            same_timestamp = bool(days_between == 0) if pd.notna(days_between) else False

            ei_float_prev, log_ei_float_prev, _ = _exposure_intensity(prev["disbursement_amount_ugx"], prev["float_activity_value_1m"])
            ei_float_cur, log_ei_float_cur, _ = _exposure_intensity(cur["disbursement_amount_ugx"], cur["float_activity_value_1m"])
            ei_comm_prev, log_ei_comm_prev, _ = _exposure_intensity(prev["disbursement_amount_ugx"], prev["commission"])
            ei_comm_cur, log_ei_comm_cur, _ = _exposure_intensity(cur["disbursement_amount_ugx"], cur["commission"])
            delta_ei_float = (ei_float_cur - ei_float_prev) if pd.notna(ei_float_cur) and pd.notna(ei_float_prev) else np.nan
            delta_ei_commission = (ei_comm_cur - ei_comm_prev) if pd.notna(ei_comm_cur) and pd.notna(ei_comm_prev) else np.nan

            outcome_eligible = bool(cur["label_eligible_30d"] == 1)
            outcome = cur["bad_state_3dpd_30d"] if outcome_eligible else np.nan

            rows.append({
                "agent_msisdn": agent, "days_between_loans": days_between, "same_timestamp": same_timestamp,
                "exposure_change_ratio": exposure_change_ratio, "exposure_direction": _direction(exposure_change_ratio),
                "float_change_ratio": float_change_ratio, "float_direction": _direction(float_change_ratio),
                "commission_change_ratio": commission_change_ratio, "commission_direction": _direction(commission_change_ratio),
                "customer_change_ratio": customer_change_ratio, "balance_change_ratio": balance_change_ratio,
                "ei_float_prev": ei_float_prev, "ei_float_cur": ei_float_cur, "log_ei_float_prev": log_ei_float_prev,
                "log_ei_float_cur": log_ei_float_cur, "delta_ei_float": delta_ei_float,
                "ei_commission_prev": ei_comm_prev, "ei_commission_cur": ei_comm_cur,
                "log_ei_commission_prev": log_ei_comm_prev, "log_ei_commission_cur": log_ei_comm_cur,
                "delta_ei_commission": delta_ei_commission,
                "outcome_eligible": outcome_eligible, "outcome_current_episode": outcome,
            })
    return pd.DataFrame(rows)


def direction_grid(transitions: pd.DataFrame, fund_col: str, fund_name: str) -> pd.DataFrame:
    """ExposureDirection x <fund_name>Direction, 9 cells. For the float
    grid, also attaches median customer/balance change ratios descriptively
    (never used to define a category)."""
    fund_dir_col = f"{fund_name}_direction"
    rows = []
    for exp_d in DIRECTION_LABELS:
        for fund_d in DIRECTION_LABELS:
            cell = transitions[(transitions["exposure_direction"] == exp_d) & (transitions[fund_dir_col] == fund_d)]
            n_total = len(cell)
            elig = cell[cell["outcome_eligible"]]
            n_elig = len(elig)
            bad_rate = elig["outcome_current_episode"].mean() if n_elig else np.nan
            row = {
                "cell_name": f"exposure_{exp_d}_{fund_name}_{fund_d}",
                "exposure_direction": exp_d, f"{fund_name}_direction": fund_d,
                "n_transitions_total": n_total, "n_transitions_outcome_eligible": n_elig,
                "bad_rate_current_episode": bad_rate,
                "median_days_between_loans": cell["days_between_loans"].median() if n_total else np.nan,
            }
            if fund_name == "float":
                row["median_customer_change_ratio"] = cell["customer_change_ratio"].median() if n_total else np.nan
                row["median_balance_change_ratio"] = cell["balance_change_ratio"].median() if n_total else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def duration_stratified_callout(transitions: pd.DataFrame, fund_name: str, exp_dir: str, fund_dir: str) -> pd.DataFrame:
    fund_dir_col = f"{fund_name}_direction"
    cell = transitions[(transitions["exposure_direction"] == exp_dir) & (transitions[fund_dir_col] == fund_dir)].copy()
    rows = []
    if cell.empty:
        return pd.DataFrame()
    cell["_duration_band"] = pd.cut(cell["days_between_loans"], bins=DURATION_BAND_EDGES, labels=DURATION_BAND_LABELS)
    for band in DURATION_BAND_LABELS:
        sub = cell[cell["_duration_band"] == band]
        n_total = len(sub)
        elig = sub[sub["outcome_eligible"]]
        n_elig = len(elig)
        bad_rate = elig["outcome_current_episode"].mean() if n_elig else np.nan
        rows.append({"duration_band": band, "n_transitions_total": n_total,
                     "n_transitions_outcome_eligible": n_elig, "bad_rate_current_episode": bad_rate})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", default="loan_episode_capacity_dataset.csv")
    ap.add_argument("--out-prefix", default="episode_agent_transitions")
    args = ap.parse_args(argv)

    path = Path(args.episode_dataset)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_loan_episode_capacity_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")

    transitions = build_transitions(df)
    n_agents_with_transitions = transitions["agent_msisdn"].nunique() if not transitions.empty else 0
    print(f"Built {len(transitions):,} consecutive-episode transition(s) across {n_agents_with_transitions:,} agent(s).")
    if transitions.empty:
        sys.exit("ERROR: no agent has >=2 orderable episodes -- nothing to analyze.")

    pct_same_ts = transitions["same_timestamp"].mean() * 100
    print(f"Same-timestamp transitions (both episodes land on the identical sort timestamp): "
          f"{pct_same_ts:.1f}% of all transitions.")
    if pct_same_ts > 5.0:
        print("  NOTE: this is frequent enough that ordering granularity may be insufficient for some agents.")

    transitions.to_csv(f"{args.out_prefix}_pairs.csv", index=False)

    for fund_col, fund_name in FUNDAMENTALS_FOR_GRIDS.items():
        print(f"\n{'#' * 100}")
        print(f"# ExposureDirection x {fund_name.capitalize()}Direction grid")
        print(f"{'#' * 100}")
        grid = direction_grid(transitions, fund_col, fund_name)
        with pd.option_context("display.float_format", "{:,.3f}".format, "display.max_columns", None, "display.width", 240):
            print(grid.to_string(index=False))
        grid.to_csv(f"{args.out_prefix}_grid_{fund_name}.csv", index=False)

        for exp_dir, fund_dir in CALLOUT_CELLS:
            if exp_dir == "flat" and fund_dir == "up":
                desc = f"exposure_flat_{fund_name}_up: agent's business grew while exposure stayed fixed -- a candidate limit-stickiness signature."
            else:
                desc = (f"exposure_down_{fund_name}_up: exposure fell while business grew -- candidate population "
                        f"for investigating a possible financing constraint or policy change, among other "
                        f"explanations; does not by itself establish policy suppression.")
            print(f"\n-- Callout cell: {desc} --")
            strat = duration_stratified_callout(transitions, fund_name, exp_dir, fund_dir)
            if strat.empty:
                print("  (no transitions in this cell)")
            else:
                print(strat.to_string(index=False))
                strat.to_csv(f"{args.out_prefix}_duration_{fund_name}_exposure_{exp_dir}_{fund_name}_{fund_dir}.csv", index=False)

    print(f"\n{'#' * 100}")
    print("Causal caveat (printed every run)")
    print(f"{'#' * 100}")
    print("A within-agent performance difference across transition cells is OBSERVATIONAL, not causal.\n"
          "Report as 'agents who experienced this transition showed X difference in subsequent performance' --\n"
          "never as 'this transition caused X.' Later exposure often coincides with more lender information,\n"
          "calendar effects, and selection that remain live explanations.")


if __name__ == "__main__":
    main()

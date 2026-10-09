"""
analyze_loan_frequency_for_snapshot_cadence.py
==================================================
Ad hoc, decision-informing diagnostic -- NOT one of the five frozen
episode-grain-rebuild deliverables. Answers a question that came up while
planning a multi-date re-pull of the agent fundamentals mart (which only
retains its LATEST snapshot, not history -- see
build_loan_episode_capacity_dataset.py's module docstring and the
Gate-0-style investigation that led to it): given loans are frequent
enough that one agent can take several in a single week, how much would a
monthly (or any other cadence of) point-in-time fundamentals snapshot
actually help?

A `_1m` trailing-window fundamental (float_activity_value_1m, commission,
cust_1m, average_balance) barely moves over 3-7 days by construction --
and if two of an agent's loans land in the SAME calendar month, a monthly
snapshot pull matches BOTH of them to the identical snapshot regardless of
how many days apart they actually are. So the real question before
spending warehouse compute on N snapshot-date re-pulls is: what fraction
of this population's consecutive-loan pairs are close enough together
that monthly cadence would still collapse them onto one snapshot (little
to no benefit), versus far enough apart that it would genuinely help?

This script answers that directly from the loan-training file ALONE --
no mart file needed -- by computing the distribution of
days-between-consecutive-loans, per agent, ordered the same way as every
other script in this rebuild (disbursement_ts preferred, loan_date
fallback, target_loan_seq as deterministic tiebreak).

NOT a judgment on whether to proceed -- purely descriptive. The actual
cadence decision (monthly vs weekly vs something else, and whether the
compute cost is worth it) is a decision for whoever owns the warehouse
re-pull, informed by these numbers, not an automatic recommendation
printed here.

Usage:
    python scripts\\analyze_loan_frequency_for_snapshot_cadence.py ^
        --loan-training-file data\\state_data_20260910_retail_filtered.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = ["loan_date"]
DAY_BAND_EDGES = [-0.5, 7.5, 30.5, 90.5, np.inf]
DAY_BAND_LABELS = ["<=7", "8-30", "31-90", ">90"]


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "agent_msisdn" not in df.columns:
        alt_col = next((c for c in ("msisdn", "phonenumber") if c in df.columns), None)
        if alt_col is not None:
            df = df.rename(columns={alt_col: "agent_msisdn"})
            print(f"NOTE: loan-training-file has no 'agent_msisdn' column -- using '{alt_col}' instead.")
    missing = [c for c in REQUIRED_COLS + ["agent_msisdn"] if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")
    df = df.copy()
    df["loan_date"] = pd.to_datetime(df["loan_date"], errors="coerce")
    df["_sort_ts"] = pd.to_datetime(df["disbursement_ts"], errors="coerce") if "disbursement_ts" in df.columns else pd.NaT
    df["_sort_ts"] = df["_sort_ts"].fillna(df["loan_date"])
    if "target_loan_seq" not in df.columns:
        df["target_loan_seq"] = np.arange(len(df))
    n_bad = int(df["_sort_ts"].isna().sum())
    if n_bad:
        print(f"NOTE: {n_bad} loan(s) dropped for unparseable loan_date/disbursement_ts.")
        df = df[df["_sort_ts"].notna()]
    return df


def compute_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """One row per consecutive (j-1, j) loan pair, per agent -- the same
    population basis Deliverable 4 uses for its own transitions."""
    ordered = df.sort_values(["agent_msisdn", "_sort_ts", "target_loan_seq"]).reset_index(drop=True)
    rows = []
    for agent, g in ordered.groupby("agent_msisdn"):
        g = g.reset_index(drop=True)
        for j in range(1, len(g)):
            days = (g.loc[j, "_sort_ts"] - g.loc[j - 1, "_sort_ts"]).days
            rows.append({"agent_msisdn": agent, "days_between_loans": days})
    return pd.DataFrame(rows)


def monthly_cadence_collapse_rate(df: pd.DataFrame, gaps: pd.DataFrame) -> float:
    """Of all consecutive loan pairs, what fraction fall in the SAME
    calendar month (year, month) -- i.e. would be matched to the identical
    snapshot under a once-per-calendar-month mart cadence, regardless of
    how many days apart they actually are. Direct, not inferred from the
    day-gap distribution, since a monthly boundary can fall anywhere
    inside a <=30-day gap."""
    ordered = df.sort_values(["agent_msisdn", "_sort_ts", "target_loan_seq"]).reset_index(drop=True)
    same_month_count = 0
    total = 0
    for agent, g in ordered.groupby("agent_msisdn"):
        g = g.reset_index(drop=True)
        months = g["_sort_ts"].dt.to_period("M")
        for j in range(1, len(g)):
            total += 1
            if months.loc[j] == months.loc[j - 1]:
                same_month_count += 1
    return (same_month_count / total * 100) if total else np.nan


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loan-training-file", required=True)
    ap.add_argument("--out-prefix", default="loan_frequency_for_snapshot_cadence")
    args = ap.parse_args(argv)

    path = Path(args.loan_training_file)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")
    df = _load(path)

    n_agents = df["agent_msisdn"].nunique()
    n_loans = len(df)
    print(f"Loan-training file: {n_loans:,} loan(s), {n_agents:,} unique agent(s).")

    gaps = compute_gaps(df)
    n_pairs = len(gaps)
    n_agents_with_pairs = gaps["agent_msisdn"].nunique() if n_pairs else 0
    print(f"Built {n_pairs:,} consecutive-loan-pair gap(s) across {n_agents_with_pairs:,} agent(s) with >=2 loans.")
    if gaps.empty:
        sys.exit("ERROR: no agent has >=2 loans -- nothing to analyze.")
    gaps.to_csv(f"{args.out_prefix}_pairs.csv", index=False)

    print(f"\n{'#' * 100}")
    print("# Days-between-consecutive-loans -- overall distribution (all agents pooled)")
    print(f"{'#' * 100}")
    pct = gaps["days_between_loans"].quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    with pd.option_context("display.float_format", "{:,.1f}".format):
        print(pct.to_string())

    print(f"\n{'=' * 100}\nBanded distribution (how far apart are consecutive loans, in days)\n{'=' * 100}")
    gaps["_band"] = pd.cut(gaps["days_between_loans"], bins=DAY_BAND_EDGES, labels=DAY_BAND_LABELS)
    band_counts = gaps["_band"].value_counts(normalize=True).reindex(DAY_BAND_LABELS) * 100
    with pd.option_context("display.float_format", "{:,.1f}%".format):
        print(band_counts.to_string())

    print(f"\n{'=' * 100}\nPer-agent median days-between-loans -- how many agents are frequent vs occasional borrowers\n{'=' * 100}")
    per_agent_median = gaps.groupby("agent_msisdn")["days_between_loans"].median()
    agent_band_counts = pd.cut(per_agent_median, bins=DAY_BAND_EDGES, labels=DAY_BAND_LABELS).value_counts(normalize=True).reindex(DAY_BAND_LABELS) * 100
    with pd.option_context("display.float_format", "{:,.1f}%".format):
        print(agent_band_counts.to_string())
    print(f"(n_agents_with_pairs = {n_agents_with_pairs:,})")

    collapse_rate = monthly_cadence_collapse_rate(df, gaps)
    print(f"\n{'#' * 100}")
    print("# Monthly-cadence collapse rate (the number that actually answers the question)")
    print(f"{'#' * 100}")
    print(f"{collapse_rate:.1f}% of consecutive loan pairs fall within the SAME calendar month -- "
          f"a once-per-calendar-month mart snapshot would match BOTH loans in those pairs to the "
          f"identical snapshot, regardless of how many days apart they actually are.")
    print("This is descriptive only. It does not recommend a cadence -- it quantifies how much of this "
          "population a monthly re-pull would, and would not, actually help.")


if __name__ == "__main__":
    main()

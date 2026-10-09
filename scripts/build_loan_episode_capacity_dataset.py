"""
build_loan_episode_capacity_dataset.py
==========================================
Deliverable 1 of the episode-grain rebuild of Analysis 3 (capacity-challenger
workstream). Builds the one dataset every subsequent script in this rebuild
depends on:

    Loan_{i,t} + latest Fundamentals_{i, s<t} + Y_{i,t}^{30d}

WHY THIS REBUILD EXISTS (so a future reader doesn't have to re-derive it):
an earlier attempt used `capacity_research_dataset.csv`, whose
`actual_exposure_ugx` (a monthly-summary snapshot) and forward-outcome
columns (`fwd_any_bad_3dpd`, `fwd_new_loans_*`) were confirmed, by reading
`data/persona_k8_forward_outcomes_query.sql` directly, to be computed over
DIFFERENT, LATER loans than the one captured in `actual_exposure_ugx` --
"Gate 0" failure. This script instead builds on `data/loan_state_query_updated.txt`
(materialized), grain = one row per `disbursement_fid`, where
`bad_state_3dpd_30d` is confirmed computed on the SAME `target_loan_uid`
over a fixed 30-day window, with pre-disbursement features guarded to
`state_date < loan_date` and proper censoring (`label_eligible_30d`). That
table carries no business-fundamental (float/commission/customer/balance)
columns of its own -- they come from the same transaction-mart pipeline
`build_capacity_research_dataset.py` already uses. THIS script performs
that join at the correct grain: once per LOAN, against the agent's most
recent PRIOR fundamentals snapshot -- not once per agent against a single
latest snapshot, which is what `build_capacity_research_dataset.py` does
for its own (different) purpose.

JOIN RULE (frozen after three review rounds -- do not loosen without
re-confirming): for loan j at `loan_date`, find the LATEST mart snapshot
with `snapshot_date < loan_date`, STRICT. No backfilling from a snapshot
dated on or after the loan. No snapshot found -> all fundamentals columns
NaN for that episode (never zero-filled, never borrowed from a later
snapshot). `fundamentals_age_days` is always persisted so staleness is
visible downstream, never silently assumed away. The RAW snapshot column
(whatever precision it was found in) is preserved separately from the
date-truncated value used for the join, so a genuinely-daily mart can be
told apart from a timestamp that was merely truncated to a date.

`disbursement_ts` (the loan table's own highest-resolution time field) is
carried through even though the fundamentals join itself only needs
date granularity -- downstream scripts (transitions, cross-tier escalation)
need it for temporal ordering, and this workstream has already paid once
for dropping a timing field too early (the Gate-0 failure above).

THE SNAPSHOT-SHARING DIAGNOSTIC BELOW IS A VIABILITY GATE, not a nice-to-have:
if most multi-loan agents map every one of their loans to the SAME mart
snapshot, any downstream delta-fundamentals analysis measures artificial
zeros, not real business change. This script surfaces that loudly rather
than letting it hide in a quiet table.

NOT in scope here (frozen, see this session's plan): no PD column, no
loan-history/repayment-behavior features, no modeling of any kind. Pure
data construction.

Usage:
    python scripts\\build_loan_episode_capacity_dataset.py ^
        --loan-training-file loan_state_query_updated_export.csv ^
        --transaction-file retail_agents_filtered.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.borrower_persona_clustering import digits  # noqa: E402

LOAN_REQUIRED_COLS = [
    "disbursement_fid", "agent_msisdn", "loan_date", "disbursement_amount_ugx",
    "target_loan_seq", "bad_state_3dpd_30d", "label_eligible_30d", "label_eligibility_reason_30d",
]
MART_NUMERIC_COLS = ["cash_in_value_1m", "payment_value_1m", "commission", "cust_1m", "average_balance"]
FUNDAMENTALS_COLS = ["float_activity_value_1m", "commission", "cust_1m", "average_balance"]

SHARING_MIN_EPISODES = 2  # an agent needs >=2 episodes to have any "sharing" to measure


def _read_csv_fast(path: Path) -> pd.DataFrame:
    """pd.read_csv, preferring the multi-threaded pyarrow engine. Restated
    independently from scripts/filter_borrower_file_by_retail_agents.py's
    identically-named helper (one-way scripts/ layering convention) --
    --loan-training-file is exactly the "several GB, state_data_*.csv"
    file that helper's own docstring names as its reason for existing: the
    single-threaded default C engine with low_memory=False can exhaust
    available memory on a file that size (observed directly: a real
    --loan-training-file run raised `pandas.errors.ParserError: ... C
    error: out of memory` from this exact low_memory=False call). Falls
    back to the plain engine (default low_memory=True, i.e. chunked dtype
    inference -- NOT low_memory=False, which is the memory-heavier choice
    that caused the failure in the first place) if pyarrow isn't
    importable, so this never hard-fails on a slower read instead of not
    running at all."""
    try:
        return pd.read_csv(path, sep=",", encoding="utf-8-sig", engine="pyarrow")
    except (ImportError, ValueError):
        return pd.read_csv(path, sep=",", encoding="utf-8-sig")


def load_loan_episodes(path: Path) -> pd.DataFrame:
    df = _read_csv_fast(path)
    if "agent_msisdn" not in df.columns:
        # data/loan_state_query_updated.txt's own final SELECT names this
        # column `msisdn` (aliased from `customer_msisdn` -- in this
        # domain the "customer" taking the loan IS the retail agent, not
        # a separate end-customer). Same 3-way fallback order already
        # used for this exact file by
        # scripts/filter_borrower_file_by_retail_agents.py (restated
        # independently, one-way scripts/ layering convention), and by
        # load_mart() below for the transaction-mart file.
        alt_col = next((c for c in ("msisdn", "phonenumber") if c in df.columns), None)
        if alt_col is not None:
            df = df.rename(columns={alt_col: "agent_msisdn"})
            print(f"NOTE: loan-training-file has no 'agent_msisdn' column -- using '{alt_col}' instead.")
    missing = [c for c in LOAN_REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")
    df = df.copy()
    df["loan_date"] = pd.to_datetime(df["loan_date"], errors="coerce")
    if "disbursement_ts" in df.columns:
        df["disbursement_ts"] = pd.to_datetime(df["disbursement_ts"], errors="coerce")
    else:
        print("NOTE: loan-training-file has no disbursement_ts column -- downstream scripts will "
              "fall back to loan_date/target_loan_seq for temporal ordering.")
        df["disbursement_ts"] = pd.NaT
    df["_id"] = digits(df["agent_msisdn"])
    n_bad_dates = int(df["loan_date"].isna().sum())
    if n_bad_dates:
        print(f"NOTE: {n_bad_dates} episode(s) dropped for unparseable loan_date.")
        df = df[df["loan_date"].notna()]
    return df


def load_mart(path: Path) -> tuple:
    """Loads the raw transaction mart at FULL granularity -- every dated row
    per agent, NOT deduplicated to one row per agent (build_capacity_research_dataset.py
    deduplicates; this script needs the opposite, so each loan can match its
    own most-recent-prior snapshot)."""
    txn = _read_csv_fast(path)
    msisdn_col = "agent_msisdn" if "agent_msisdn" in txn.columns else "msisdn"
    if msisdn_col not in txn.columns:
        sys.exit(f"ERROR: {path} has neither 'agent_msisdn' nor 'msisdn'. Columns present: {list(txn.columns)}")
    date_col = "snapshot_dt" if "snapshot_dt" in txn.columns else ("tbl_dt" if "tbl_dt" in txn.columns else None)
    if date_col is None:
        sys.exit(f"ERROR: {path} has neither 'snapshot_dt' nor 'tbl_dt' -- cannot perform a "
                  f"point-in-time join without a snapshot date.")
    txn = txn.copy()
    txn["_id"] = digits(txn[msisdn_col])
    missing_raw = [c for c in MART_NUMERIC_COLS if c not in txn.columns]
    if missing_raw:
        print(f"NOTE: {path.name} is missing raw column(s): {missing_raw} -- corresponding "
              f"fundamentals will be NaN.")
    for c in MART_NUMERIC_COLS:
        if c in txn.columns:
            txn[c] = pd.to_numeric(txn[c], errors="coerce")
    if {"cash_in_value_1m", "payment_value_1m"} <= set(txn.columns):
        txn["float_activity_value_1m"] = txn["cash_in_value_1m"].fillna(0) + txn["payment_value_1m"].fillna(0)
    else:
        txn["float_activity_value_1m"] = np.nan

    txn["fundamentals_snapshot_raw"] = txn[date_col]
    txn["fundamentals_snapshot_date"] = pd.to_datetime(txn[date_col], errors="coerce").dt.normalize()
    n_bad = int(txn["fundamentals_snapshot_date"].isna().sum())
    if n_bad:
        print(f"NOTE: {n_bad} mart row(s) dropped for unparseable {date_col}.")
        txn = txn[txn["fundamentals_snapshot_date"].notna()]
    keep_cols = ["_id", "fundamentals_snapshot_raw", "fundamentals_snapshot_date"] + FUNDAMENTALS_COLS
    return txn[keep_cols].copy(), date_col


def attach_pretrade_fundamentals(episodes: pd.DataFrame, mart: pd.DataFrame) -> pd.DataFrame:
    """As-of join: for each episode, the LATEST mart row with
    fundamentals_snapshot_date STRICTLY before loan_date, for that agent.
    Uses merge_asof(direction="backward", allow_exact_matches=False) -- not
    an epsilon subtraction, which would be an avoidable ambiguity given
    mixed date/timestamp precision across sources. Falls back to a
    post-filter if the installed pandas lacks allow_exact_matches (prints
    which path was used either way)."""
    ep = episodes.sort_values("loan_date").reset_index(drop=True)
    mt = mart.sort_values("fundamentals_snapshot_date").reset_index(drop=True)

    try:
        merged = pd.merge_asof(
            ep, mt, left_on="loan_date", right_on="fundamentals_snapshot_date",
            left_by="_id", right_by="_id", direction="backward", allow_exact_matches=False,
        )
        join_method = "merge_asof(direction='backward', allow_exact_matches=False)"
    except TypeError:
        merged = pd.merge_asof(
            ep, mt, left_on="loan_date", right_on="fundamentals_snapshot_date",
            left_by="_id", right_by="_id", direction="backward",
        )
        exact_match = merged["fundamentals_snapshot_date"] == merged["loan_date"]
        for c in ["fundamentals_snapshot_raw", "fundamentals_snapshot_date"] + FUNDAMENTALS_COLS:
            merged.loc[exact_match, c] = np.nan
        join_method = "merge_asof(direction='backward') + explicit exact-match nulling (allow_exact_matches unavailable)"
    print(f"Fundamentals join method: {join_method}")

    merged["fundamentals_age_days"] = (merged["loan_date"] - merged["fundamentals_snapshot_date"]).dt.days
    n_unmatched = int(merged["fundamentals_snapshot_date"].isna().sum())
    if n_unmatched:
        print(f"NOTE: {n_unmatched:,} of {len(merged):,} episode(s) ({n_unmatched / len(merged) * 100:.1f}%) "
              f"have no eligible prior fundamentals snapshot -- fundamentals left NaN, not zero-filled "
              f"or back-filled.")
    return merged


def report_snapshot_sharing(df: pd.DataFrame) -> None:
    """Viability gate for the downstream transition/escalation analyses:
    if most multi-loan agents reuse the same fundamentals snapshot across
    every loan, delta-fundamentals analysis measures artificial zeros."""
    counts = df.groupby("_id").agg(
        n_episodes=("disbursement_fid", "size"),
        n_unique_snapshots=("fundamentals_snapshot_date", lambda s: s.dropna().nunique()),
    )
    multi = counts[counts["n_episodes"] >= SHARING_MIN_EPISODES]
    print(f"\nSnapshot-sharing diagnostic ({len(multi):,} agent(s) with >= {SHARING_MIN_EPISODES} episodes):")
    if multi.empty:
        print("  (no multi-episode agents -- nothing to report)")
        return
    pct_single_snapshot = (multi["n_unique_snapshots"] <= 1).mean() * 100
    median_unique = multi["n_unique_snapshots"].median()
    print(f"  {pct_single_snapshot:.1f}% of multi-episode agents have only ONE distinct fundamentals "
          f"snapshot across ALL their episodes.")
    print(f"  Median distinct snapshots per multi-episode agent: {median_unique:.1f}.")

    df_sorted = df.sort_values(["_id", "loan_date"])
    df_sorted["_prev_snapshot"] = df_sorted.groupby("_id")["fundamentals_snapshot_date"].shift(1)
    has_prev = df_sorted["_prev_snapshot"].notna() & df_sorted["fundamentals_snapshot_date"].notna()
    shares_prev = has_prev & (df_sorted["fundamentals_snapshot_date"] == df_sorted["_prev_snapshot"])
    pct_shares_prev = shares_prev.sum() / has_prev.sum() * 100 if has_prev.sum() else float("nan")
    print(f"  {pct_shares_prev:.1f}% of episodes (with a comparable predecessor) share the IDENTICAL "
          f"snapshot as their immediately preceding episode.")
    if pct_single_snapshot >= 80.0:
        print("  WARNING: the large majority of multi-episode agents reuse a single fundamentals "
              "snapshot across all loans. Delta-fundamentals analysis (Deliverables 4-5) will mostly "
              "measure artificial zeros, not real business change, for this population.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loan-training-file", required=True)
    ap.add_argument("--transaction-file", required=True)
    ap.add_argument("--out", default="loan_episode_capacity_dataset.csv")
    args = ap.parse_args(argv)

    loan_path = Path(args.loan_training_file)
    txn_path = Path(args.transaction_file)
    if not loan_path.exists():
        sys.exit(f"ERROR: {loan_path} not found.")
    if not txn_path.exists():
        sys.exit(f"ERROR: {txn_path} not found.")

    episodes = load_loan_episodes(loan_path)
    print(f"Loan-training file: {len(episodes):,} episode(s), {episodes['_id'].nunique():,} unique agent(s).")
    mart, date_col = load_mart(txn_path)
    print(f"Transaction mart: {len(mart):,} dated snapshot row(s) (on {date_col}), "
          f"{mart['_id'].nunique():,} unique agent(s).")

    joined = attach_pretrade_fundamentals(episodes, mart)
    report_snapshot_sharing(joined)

    out_cols = [
        "disbursement_fid", "agent_msisdn", "loan_date", "disbursement_ts", "target_loan_seq",
        "disbursement_amount_ugx", "fundamentals_snapshot_raw", "fundamentals_snapshot_date",
        "fundamentals_age_days", "float_activity_value_1m", "commission", "cust_1m", "average_balance",
        "bad_state_3dpd_30d", "label_eligible_30d", "label_eligibility_reason_30d",
    ]
    result = joined[out_cols].copy()
    result.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(result):,} rows, {len(out_cols)} columns).")


if __name__ == "__main__":
    main()

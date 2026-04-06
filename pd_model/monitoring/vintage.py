"""
Vintage / cohort analysis for credit model ongoing monitoring.

Tracks bad rate by disbursement month as loans mature (Months on Book = MOB).
This is the standard credit model monitoring tool — it answers "does the model's
risk ranking hold up as loans season?"

Definitions
-----------
- Cohort        : group of loans disbursed in the same calendar month
- MOB           : Months on Book = integer months elapsed since disbursement
- Bad           : outcome label (1 = default / bad, 0 = good)
- Vintage curve : bad rate at each MOB for a given cohort

Usage
-----
::

    from pd_model.monitoring.vintage import build_vintage_table, build_cohort_matrix

    vtbl  = build_vintage_table(loans_df,
                                disbursement_col="disbursement_date",
                                observation_col="observation_date",
                                outcome_col="is_bad")
    matrix = build_cohort_matrix(vtbl)   # cohort × MOB pivot of bad rates
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pd_model.logging_config import get_logger

logger = get_logger(__name__)


# ======================================================================== #
# Long-format vintage table
# ======================================================================== #

def build_vintage_table(
    loans_df: pd.DataFrame,
    disbursement_col: str = "disbursement_date",
    observation_col: str = "observation_date",
    outcome_col: str = "is_bad",
    agent_col: str = "agent_msisdn",
    model_bucket_col: str | None = None,
) -> pd.DataFrame:
    """
    Build a long-format vintage table with MOB derived from dates.

    MOB is computed as the number of complete calendar months between
    disbursement_date and observation_date.

    Parameters
    ----------
    loans_df         : one row per loan observation, must contain
                       disbursement_col, observation_col, outcome_col
    disbursement_col : date column for loan origination
    observation_col  : date column for performance observation (snapshot date)
    outcome_col      : binary bad/default label (1 = bad)
    agent_col        : agent identifier (for n_agents count)
    model_bucket_col : optional policy bucket column for stratified analysis

    Returns
    -------
    DataFrame (long format) with columns:
        cohort_month, mob, n_loans, n_bads, bad_rate, cumulative_bad_rate
        (+ model_bucket if model_bucket_col provided)
    """
    df = loans_df.copy()

    for col in (disbursement_col, observation_col):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=[disbursement_col, observation_col, outcome_col])
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0)

    # Cohort month = year-month of disbursement
    df["cohort_month"] = df[disbursement_col].dt.to_period("M")

    # MOB = complete calendar months elapsed
    df["mob"] = (
        (df[observation_col].dt.year - df[disbursement_col].dt.year) * 12
        + (df[observation_col].dt.month - df[disbursement_col].dt.month)
    ).clip(lower=0)

    group_cols = ["cohort_month", "mob"]
    if model_bucket_col and model_bucket_col in df.columns:
        group_cols.append(model_bucket_col)

    agg = (
        df.groupby(group_cols, observed=True)
        .agg(
            n_loans=(outcome_col, "count"),
            n_bads=(outcome_col, "sum"),
        )
        .reset_index()
    )
    agg["bad_rate"] = agg["n_bads"] / agg["n_loans"].clip(lower=1)

    # Cumulative bad rate within each cohort (sorted by MOB)
    agg = agg.sort_values(["cohort_month", "mob"]).reset_index(drop=True)
    agg["cumulative_bad_rate"] = (
        agg.groupby("cohort_month", observed=True)
        .apply(lambda g: g["n_bads"].cumsum() / g["n_loans"].cumsum())
        .reset_index(level=0, drop=True)
    )

    logger.info(
        "build_vintage_table: %d cohorts | MOB range=[%d, %d] | total loans=%d",
        agg["cohort_month"].nunique(),
        int(agg["mob"].min()),
        int(agg["mob"].max()),
        int(agg["n_loans"].sum()),
    )
    return agg


# ======================================================================== #
# Cohort × MOB pivot matrix
# ======================================================================== #

def build_cohort_matrix(
    vintage_tbl: pd.DataFrame,
    value_col: str = "cumulative_bad_rate",
) -> pd.DataFrame:
    """
    Pivot vintage table to a cohort × MOB matrix.

    Rows    = cohort_month (origination month)
    Columns = MOB (0, 1, 2, …)
    Values  = cumulative_bad_rate (or bad_rate, n_loans etc. via value_col)

    Empty cells (cohort not yet reached that MOB) are NaN.

    Returns
    -------
    DataFrame indexed by cohort_month with MOB as columns.
    """
    matrix = vintage_tbl.pivot_table(
        index="cohort_month",
        columns="mob",
        values=value_col,
        aggfunc="mean",
    )
    matrix.columns.name = "mob"
    matrix.index.name = "cohort_month"
    return matrix


# ======================================================================== #
# Cohort summary
# ======================================================================== #

def build_vintage_summary(
    vintage_tbl: pd.DataFrame,
    mature_mob: int = 6,
) -> pd.DataFrame:
    """
    Summary statistics per cohort.

    Parameters
    ----------
    vintage_tbl : output of build_vintage_table
    mature_mob  : MOB considered "mature" for the mature bad rate column.
                  Cohorts that have not yet reached this MOB will have NaN.

    Returns
    -------
    DataFrame with one row per cohort:
        cohort_month, n_loans_total, n_bads_total,
        overall_bad_rate, mature_bad_rate (at mature_mob),
        max_mob_observed
    """
    totals = (
        vintage_tbl.groupby("cohort_month", observed=True)
        .agg(
            n_loans_total=("n_loans", "sum"),
            n_bads_total=("n_bads", "sum"),
            max_mob_observed=("mob", "max"),
        )
        .reset_index()
    )
    totals["overall_bad_rate"] = totals["n_bads_total"] / totals["n_loans_total"].clip(lower=1)

    mature_rows = vintage_tbl[vintage_tbl["mob"] == mature_mob][
        ["cohort_month", "cumulative_bad_rate"]
    ].rename(columns={"cumulative_bad_rate": "mature_bad_rate"})

    summary = totals.merge(mature_rows, on="cohort_month", how="left")

    logger.info(
        "build_vintage_summary: %d cohorts | avg overall_bad_rate=%.4f | "
        "cohorts at MOB%d=%d",
        len(summary),
        float(summary["overall_bad_rate"].mean()),
        mature_mob,
        int(summary["mature_bad_rate"].notna().sum()),
    )
    return summary.sort_values("cohort_month").reset_index(drop=True)

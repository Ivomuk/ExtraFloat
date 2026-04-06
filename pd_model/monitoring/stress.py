"""
Stress testing — what happens to approval rates and expected loss
when bad rates scale up.

Uses *locked thresholds* from the training-time policy table: the PD cutoff
that achieved a given approval rate at training time is held fixed while
agent PDs are multiplied by a stress factor.  This answers the deployment
question: "if defaults are twice as frequent as we estimated, how many agents
would no longer qualify and what is our expected loss?"

Usage
-----
::

    from pd_model.monitoring.stress import run_stress_test

    stress_tbl = run_stress_test(
        scored_df=val_scored_with_cal_pd,
        policy_threshold_tbl=xgb_thresh,        # from build_policy_tables
        stress_multipliers=(1.0, 1.5, 2.0, 3.0),
        loan_size_col="loan_amount",             # optional
    )
    print(stress_tbl)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pd_model.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0)
_DEFAULT_OPERATING_POINTS = (0.10, 0.20, 0.50, 0.80)


# ======================================================================== #
# Core stress test
# ======================================================================== #

def run_stress_test(
    scored_df: pd.DataFrame,
    policy_threshold_tbl: pd.DataFrame,
    cal_pd_col: str = "cal_pd",
    loan_size_col: str | None = None,
    stress_multipliers: tuple[float, ...] = _DEFAULT_MULTIPLIERS,
    operating_points: tuple[float, ...] = _DEFAULT_OPERATING_POINTS,
) -> pd.DataFrame:
    """
    Stress test using locked PD thresholds from the training-time policy table.

    For each (operating_point, stress_multiplier) combination:
      - Retrieve the locked PD cutoff from policy_threshold_tbl
      - Multiply each agent's cal_pd by the stress multiplier (capped at 1.0)
      - Count agents still approved (stressed_pd <= locked cutoff)
      - Compute expected loss (EL) = sum(stressed_pd) for approved agents,
        or sum(stressed_pd × loan_size) if loan_size_col is provided

    Parameters
    ----------
    scored_df             : DataFrame with cal_pd (and optionally loan_size_col)
    policy_threshold_tbl  : output of build_policy_tables — must contain
                            approve_rate_target and cutoff columns
    cal_pd_col            : calibrated PD column name (default "cal_pd")
    loan_size_col         : optional column for loan amounts; if None EL uses
                            count-weighted PD (assumes unit loan size)
    stress_multipliers    : PD scaling factors to test (default 1×, 1.5×, 2×, 3×)
    operating_points      : approval-rate targets to lock (default 10/20/50/80%)

    Returns
    -------
    DataFrame with columns:
        operating_point, stress_multiplier,
        locked_pd_cutoff,
        n_agents, n_approved, approval_rate,
        expected_loss, el_per_approved,
        approval_rate_change_pp  (vs 1× baseline)
        el_change_pct            (vs 1× baseline)
    """
    if cal_pd_col not in scored_df.columns:
        raise ValueError(
            f"[run_stress_test] cal_pd column '{cal_pd_col}' not found in scored_df"
        )

    base_pds = pd.to_numeric(scored_df[cal_pd_col], errors="coerce").fillna(0.0).to_numpy()

    loan_sizes = None
    if loan_size_col and loan_size_col in scored_df.columns:
        loan_sizes = pd.to_numeric(scored_df[loan_size_col], errors="coerce").fillna(0.0).to_numpy()

    n_agents = len(base_pds)

    # Index policy table by approximate operating point for fast lookup
    thresh_tbl = policy_threshold_tbl.copy()
    thresh_tbl["approve_rate_target"] = pd.to_numeric(
        thresh_tbl["approve_rate_target"], errors="coerce"
    )

    rows = []
    baseline: dict[float, float] = {}  # operating_point → baseline approval_rate

    for op in operating_points:
        # Find the row in policy table closest to the requested operating point
        idx = (thresh_tbl["approve_rate_target"] - op).abs().idxmin()
        locked_cutoff = float(thresh_tbl.loc[idx, "cutoff"])

        for m in stress_multipliers:
            stressed_pds = np.clip(base_pds * m, 0.0, 1.0)
            approved_mask = stressed_pds <= locked_cutoff
            n_approved = int(approved_mask.sum())
            approval_rate = n_approved / max(n_agents, 1)

            if loan_sizes is not None:
                el = float((stressed_pds * loan_sizes * approved_mask).sum())
                el_per = el / max(n_approved, 1)
            else:
                el = float(stressed_pds[approved_mask].sum())
                el_per = float(stressed_pds[approved_mask].mean()) if n_approved > 0 else np.nan

            rows.append({
                "operating_point": op,
                "stress_multiplier": m,
                "locked_pd_cutoff": round(locked_cutoff, 6),
                "n_agents": n_agents,
                "n_approved": n_approved,
                "approval_rate": round(approval_rate, 4),
                "expected_loss": round(el, 4),
                "el_per_approved": round(el_per, 6) if not np.isnan(el_per) else np.nan,
            })

            if m == 1.0:
                baseline[op] = approval_rate

    df = pd.DataFrame(rows)

    # Compute change vs 1× baseline
    df["approval_rate_change_pp"] = df.apply(
        lambda r: round((r["approval_rate"] - baseline.get(r["operating_point"], np.nan)) * 100, 2),
        axis=1,
    )

    # EL change vs baseline
    baseline_el: dict[float, float] = {}
    for op in operating_points:
        base_row = df[(df["operating_point"] == op) & (df["stress_multiplier"] == 1.0)]
        baseline_el[op] = float(base_row["expected_loss"].iloc[0]) if len(base_row) else np.nan

    df["el_change_pct"] = df.apply(
        lambda r: round(
            (r["expected_loss"] / baseline_el.get(r["operating_point"], np.nan) - 1) * 100, 2
        )
        if baseline_el.get(r["operating_point"], 0) > 0
        else np.nan,
        axis=1,
    )

    logger.info(
        "run_stress_test: %d agents | %d operating points | %d multipliers",
        n_agents, len(operating_points), len(stress_multipliers),
    )
    return df.sort_values(["operating_point", "stress_multiplier"]).reset_index(drop=True)


# ======================================================================== #
# Wide-format summary
# ======================================================================== #

def build_stress_summary(
    stress_tbl: pd.DataFrame,
    metric: str = "approval_rate",
) -> pd.DataFrame:
    """
    Pivot stress table to operating_point × stress_multiplier wide format.

    Parameters
    ----------
    stress_tbl : output of run_stress_test
    metric     : column to pivot (default "approval_rate";
                 also useful: "expected_loss", "el_per_approved",
                 "approval_rate_change_pp", "el_change_pct")

    Returns
    -------
    DataFrame: rows = operating_point, columns = stress_multiplier values.
    """
    pivot = stress_tbl.pivot_table(
        index="operating_point",
        columns="stress_multiplier",
        values=metric,
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_at_{m}x" for m in pivot.columns]
    pivot.columns.name = None
    return pivot.reset_index()

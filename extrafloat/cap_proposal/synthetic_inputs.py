"""
extrafloat/cap_proposal/synthetic_inputs.py
===========================================
Synthetic raw input tables for a dry run of the burden limit design.

Produces the four raw tables the live engine's own loaders/feature builder
consume (transaction capacity, borrower history, loan summary, loan history
snapshot), so the trial exercises the REAL feature pipeline end to end.
Values are illustrative only: agent size is lognormal and every UGX column
of a template row is scaled by it. Nothing here is calibrated to the real
population; use it to check that the trial runs and that the design's rules
behave as written, never to judge the design's numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SNAPSHOT_DT = "2026-08-31"

# Template rows copied from tests/engine/test_extrafloat_pipeline.py (a small Silver agent).
_TXN_TEMPLATE = {
    "agent_profile": "Silver Class", "account_balance": 9403.0, "average_balance": 37646.15, "commission": 14402.80,
    "cash_out_vol_1m": 2.0, "cash_out_vol_3m": 7.0, "cash_out_vol_6m": 12.0,
    "cash_out_value_1m": 40000.0, "cash_out_value_3m": 195000.0, "cash_out_value_6m": 680000.0,
    "cash_out_cust_1m": 2.0, "cash_out_cust_3m": 7.0, "cash_out_cust_6m": 12.0,
    "cash_out_comm_1m": 547.0, "cash_out_comm_3m": 2017.0, "cash_out_comm_6m": 5757.0,
    "cash_in_vol_1m": 3.0, "cash_in_vol_3m": 7.0, "cash_in_vol_6m": 12.0,
    "cash_in_value_1m": 36000.0, "cash_in_value_3m": 119500.0, "cash_in_value_6m": 170500.0,
    "cash_in_cust_1m": 3.0, "cash_in_cust_3m": 7.0, "cash_in_cust_6m": 12.0,
    "cash_in_comm_1m": 473.0, "cash_in_comm_3m": 1193.0, "cash_in_comm_6m": 1993.0,
    "payment_vol_1m": 9.0, "payment_vol_3m": 25.0, "payment_vol_6m": 36.0,
    "payment_value_1m": 42000.0, "payment_value_3m": 117500.0, "payment_value_6m": 151200.0,
    "payment_cust_1m": 9.0, "payment_cust_3m": 25.0, "payment_cust_6m": 36.0,
    "payment_comm_1m": 1848.0, "payment_comm_3m": 5170.0, "payment_comm_6m": 6652.80,
    "cust_1m": 14.0, "cust_3m": 39.0, "cust_6m": 60.0, "vol_1m": 14.0, "vol_3m": 39.0, "vol_6m": 60.0,
}
_TXN_COUNT_PREFIXES = ("vol", "cust", "cash_out_vol", "cash_out_cust", "cash_in_vol", "cash_in_cust",
                       "payment_vol", "payment_cust")


def _is_count_col(col: str) -> bool:
    return any(col.startswith(p + "_") for p in _TXN_COUNT_PREFIXES)


def make_synthetic_inputs(n: int = 2_000, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Return {'transaction', 'borrower', 'loan_summary', 'loan_history'} raw frames for n agents."""
    rng = np.random.default_rng(seed)
    msisdn = [f"2567{i:08d}" for i in range(n)]

    # -- transaction capacity: scale the template by a lognormal size factor --
    size = rng.lognormal(mean=np.log(20.0), sigma=1.3, size=n)       # ~20x the template agent at the median
    txn = pd.DataFrame({c: np.full(n, v) for c, v in _TXN_TEMPLATE.items()})
    for c, v in _TXN_TEMPLATE.items():
        if isinstance(v, float):
            scale = np.sqrt(size) if _is_count_col(c) else size
            txn[c] = np.round(v * scale, 2)
    txn.insert(0, "agent_msisdn", msisdn)
    txn.insert(1, "snapshot_dt", SNAPSHOT_DT)

    # -- loan history: experience, streaks, current arrears --------------------
    total_loans = rng.choice([0, 1, 2, 3, 5, 8, 15, 30], size=n, p=[.12, .10, .08, .10, .15, .15, .15, .15])
    closed = np.maximum(total_loans - rng.integers(0, 2, size=n), 0)
    defaults = rng.binomial(np.maximum(closed, 0), 0.04)
    streak = np.where(defaults == 0, closed, rng.integers(0, np.maximum(closed, 1) + 1))
    aging = np.where(rng.random(n) < 0.06, rng.choice([3, 12, 25, 45, 90], size=n), 0)
    anomaly = (rng.random(n) < 0.01).astype(int)
    has_open = ((total_loans > closed) | (aging > 0)).astype(int)
    loan_hist = pd.DataFrame({
        "msisdn": msisdn, "snapshot_dt": SNAPSHOT_DT,
        "has_unresolved_loan_at_snapshot": has_open,
        "active_loan_days_aging_at_snapshot": aging,
        "active_loan_outstanding_ugx_at_snapshot": np.where(has_open == 1, 200_000.0, 0.0),
        "anomaly_open_at_snapshot": anomaly,
        "historical_anomaly_open_loan_rate": 0.0,
        "observed_loan_count": total_loans,
        "closed_loan_count": closed,
        "consecutive_on_time_loans": streak,
        # columns of the newer loan history extract, used by the s4-s6 candidate settings (arrears source
        # "snapshot" / "any_loan", reactivation); not produced by data/loan_history_snapshot_query.txt
        "current_dpd_at_snapshot": aging,
        "written_off_at_snapshot": anomaly,
        "worst_dpd_material_any_loan_at_snapshot": aging,
        "days_since_last_repayment_any_loan_at_snapshot": np.where(total_loans > 0, rng.choice([5, 20, 60, 200], size=n, p=[.5, .3, .15, .05]), np.nan),
    })

    # -- borrower history: loan sizes scale with the agent ---------------------
    typical_loan = np.clip(np.round(30_000 * size, -3), 50_000, 3_000_000)
    max_loan = np.where(total_loans > 0, np.round(typical_loan * rng.uniform(1.0, 1.8, size=n), -3), 0.0)
    default_rate = np.where(closed > 0, defaults / np.maximum(closed, 1), np.nan)
    on_time = np.where(closed > 1, np.clip(1 - default_rate - rng.choice([0, 0, 0.1], size=n), 0, 1), np.nan)
    worst_hours = np.where(defaults > 0, rng.choice([30, 200, 1_200], size=n), rng.choice([6, 18, 24], size=n))
    borrower = pd.DataFrame({
        "msisdn": msisdn, "total_loans": total_loans,
        "first_loan_ts": "2024-01-15", "latest_loan_ts": "2026-08-01",
        "total_disbursed_amount": typical_loan * total_loans,
        "avg_loan_size_lifetime": np.where(total_loans > 0, typical_loan, 0.0),
        "max_loan_size_lifetime": max_loan,
        "lifetime_on_time_24h_rate": on_time, "lifetime_on_time_26h_rate": on_time,
        "lifetime_default_24h_rate": default_rate, "lifetime_default_26h_rate": default_rate,
        "lifetime_severe_default_48h_rate": default_rate, "lifetime_zero_recovery_rate": 0.0,
        "lifetime_avg_hours_to_principal_cure": 12.0,
        "lifetime_worst_hours_to_principal_cure": np.where(total_loans > 0, worst_hours, np.nan),
        "lifetime_cure_time_volatility": 6.0,
        "latest_requestid": [f"REQ{i}" for i in range(n)],
        "latest_disbursement_ts": "2026-08-01", "latest_disbursed_amount": typical_loan,
        "num_prior_loans": np.maximum(total_loans - 1, 0),
        "prior_on_time_24h_rate": on_time, "avg_prior_hours_to_cure": 12.0,
        "worst_prior_hours_to_cure": worst_hours, "cure_time_volatility": 6.0,
        "recent_3_on_time_rate": on_time, "recent_3_avg_cure_time": 10.0,
        "lifetime_prior_default_24h_rate": default_rate, "recent_5_default_24h_rate": default_rate,
        "cure_time_trend": 0.0, "borrower_trend": 0.0, "borrower_profile_type": "standard",
        "loans_last_50_loans": total_loans, "defaults_last_50_loans": defaults,
        "default_rate_last_50_loans": default_rate,
        "prior_on_time_streak": streak, "prior_default_streak": 0,
        "avg_prior_loan_size": np.where(total_loans > 1, typical_loan, 0.0),
        "max_prior_loan_size": max_loan,
        "loan_size_vs_avg_ratio": 1.0, "loan_size_vs_max_ratio": 0.7, "loan_above_prior_max_flag": 0,
    })
    borrower = borrower[total_loans > 0].reset_index(drop=True)   # never-borrowed agents have no history row

    # -- loan summary: recent disbursement/repayment ----------------------------
    active = (total_loans > 0) & (rng.random(n) < 0.7)
    d1 = np.where(active, typical_loan * rng.integers(1, 4, size=n), 0.0)
    loan_summary = pd.DataFrame({
        "msisdn": msisdn, "snapshot_dt": SNAPSHOT_DT,
        "last_disbursement_date": "2026-08-20", "last_repayment_date": "2026-08-25",
        "disbursement_vol_1m": np.where(active, 2, 0), "disbursement_val_1m": d1,
        "repayment_vol_1m": np.where(active, 2, 0), "repayment_val_1m": d1 * 1.02,
        "penalties_1m": 0, "disbursement_val_3m": d1 * 3, "repayment_val_3m": d1 * 3.06, "penalties_3m": 0,
    })
    return {"transaction": txn, "borrower": borrower, "loan_summary": loan_summary, "loan_history": loan_hist}

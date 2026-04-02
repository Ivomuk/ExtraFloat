"""
tests/test_extrafloat_pipeline.py
==================================
Pytest test suite for the ExtraFloat mobile-money credit limit pipeline.

Covers all 10 requested test scenarios:
  1.  prepare_transaction_capacity_features() with sample data
  2a. prepare_borrower_limit_features() — all-null edge case
  2b. prepare_borrower_limit_features() — new-to-credit edge case
  3.  build_extrafloat_limit_engine_features() alias columns exist
  4.  compute_risk_cap() — score range 0-1, risk_cap bounded
  5.  compute_capacity_cap() — primary and fallback paths
  6.  compute_recent_usage_cap() — activity gate, penalty haircut
  7.  compute_prior_exposure_cap() — new-to-credit vs existing
  8.  combine_caps() — thin-file vs standard weighting
  9.  apply_policy_adjustments() — tier classification, proven-good override
  10. End-to-end: features → caps → assigned_limit sensible values
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from extrafloat_limit_engine_features import (
    THIN_FILE_LOAN_THRESHOLD,
    build_extrafloat_limit_engine_features,
    prepare_borrower_limit_features,
    prepare_transaction_capacity_features,
)
from extrafloat_limit_engine_caps import (
    apply_policy_adjustments,
    combine_caps,
    compute_capacity_cap,
    compute_prior_exposure_cap,
    compute_recent_usage_cap,
    compute_risk_cap,
)
from run_extrafloat_limit_engine import run_extrafloat_limit_engine


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def _txn_row(**overrides) -> dict:
    """Minimal valid transaction-capacity row using sample-file values."""
    base = {
        "agent_msisdn": "256785",
        "snapshot_dt": "2025-11-15",
        "agent_profile": "Silver Class",
        "account_balance": 9403.0,
        "average_balance": 37646.15,
        "commission": 14402.80,
        # cash-out
        "cash_out_vol_1m": 2.0,   "cash_out_vol_3m": 7.0,   "cash_out_vol_6m": 12.0,
        "cash_out_value_1m": 40000.0, "cash_out_value_3m": 195000.0, "cash_out_value_6m": 680000.0,
        "cash_out_cust_1m": 2.0,  "cash_out_cust_3m": 7.0,  "cash_out_cust_6m": 12.0,
        "cash_out_comm_1m": 547.0, "cash_out_comm_3m": 2017.0, "cash_out_comm_6m": 5757.0,
        # cash-in
        "cash_in_vol_1m": 3.0,    "cash_in_vol_3m": 7.0,    "cash_in_vol_6m": 12.0,
        "cash_in_value_1m": 36000.0, "cash_in_value_3m": 119500.0, "cash_in_value_6m": 170500.0,
        "cash_in_cust_1m": 3.0,   "cash_in_cust_3m": 7.0,   "cash_in_cust_6m": 12.0,
        "cash_in_comm_1m": 473.0,  "cash_in_comm_3m": 1193.0, "cash_in_comm_6m": 1993.0,
        # payment
        "payment_vol_1m": 9.0,    "payment_vol_3m": 25.0,   "payment_vol_6m": 36.0,
        "payment_value_1m": 42000.0, "payment_value_3m": 117500.0, "payment_value_6m": 151200.0,
        "payment_cust_1m": 9.0,   "payment_cust_3m": 25.0,  "payment_cust_6m": 36.0,
        "payment_comm_1m": 1848.0, "payment_comm_3m": 5170.0, "payment_comm_6m": 6652.80,
        # totals
        "cust_1m": 14.0, "cust_3m": 39.0, "cust_6m": 60.0,
        "vol_1m": 14.0,  "vol_3m": 39.0,  "vol_6m": 60.0,
    }
    base.update(overrides)
    return base


def _borrower_row(**overrides) -> dict:
    """Minimal valid borrower row with clean 5-loan history."""
    base = {
        "msisdn": "256785",
        "total_loans": 5,
        "first_loan_ts": "2024-01-15",
        "latest_loan_ts": "2025-10-01",
        "total_disbursed_amount": 25000.0,
        "avg_loan_size_lifetime": 5000.0,
        "max_loan_size_lifetime": 8000.0,
        "lifetime_on_time_24h_rate": 0.80,
        "lifetime_on_time_26h_rate": 0.85,
        "lifetime_default_24h_rate": 0.05,
        "lifetime_default_26h_rate": 0.05,
        "lifetime_severe_default_48h_rate": 0.02,
        "lifetime_zero_recovery_rate": 0.00,
        "lifetime_avg_hours_to_principal_cure": 12.0,
        "lifetime_worst_hours_to_principal_cure": 48.0,
        "lifetime_cure_time_volatility": 6.0,
        "latest_requestid": "REQ001",
        "latest_disbursement_ts": "2025-10-01",
        "latest_disbursed_amount": 5000.0,
        "num_prior_loans": 5,
        "prior_on_time_24h_rate": 0.80,
        "avg_prior_hours_to_cure": 12.0,
        "worst_prior_hours_to_cure": 48.0,
        "cure_time_volatility": 6.0,
        "recent_3_on_time_rate": 0.80,
        "recent_3_avg_cure_time": 10.0,
        "lifetime_prior_default_24h_rate": 0.05,
        "recent_5_default_24h_rate": 0.05,
        "cure_time_trend": 0.0,
        "borrower_trend": 0.0,
        "borrower_profile_type": "standard",
        "loans_last_50_loans": 5,
        "defaults_last_50_loans": 0,
        "default_rate_last_50_loans": 0.0,
        "prior_on_time_streak": 3,
        "prior_default_streak": 0,
        "avg_prior_loan_size": 5000.0,
        "max_prior_loan_size": 8000.0,
        "loan_size_vs_avg_ratio": 1.0,
        "loan_size_vs_max_ratio": 0.625,
        "loan_above_prior_max_flag": 0,
    }
    base.update(overrides)
    return base


def _loan_row(**overrides) -> dict:
    """Minimal valid loan-summary row."""
    base = {
        "msisdn": "256785",
        "snapshot_dt": "2025-11-15",
        "last_disbursement_date": "2025-10-01",
        "last_repayment_date": "2025-10-15",
        "disbursement_vol_1m": 1,
        "disbursement_val_1m": 5000.0,
        "repayment_vol_1m": 1,
        "repayment_val_1m": 5100.0,
        "penalties_1m": 0,
        "disbursement_val_3m": 10000.0,
        "repayment_val_3m": 10200.0,
        "penalties_3m": 0,
    }
    base.update(overrides)
    return base


def _features(**overrides) -> pd.DataFrame:
    """
    Single-row features DataFrame with all columns needed by cap functions
    and the full pipeline. Represents a healthy, mid-file borrower.
    """
    row = {
        # ── risk cap inputs ──────────────────────────────────────────────────
        "on_time_repayment_rate": 0.80,
        "lifetime_default_rate": 0.05,
        "default_rate_last_10_loans": 0.05,
        "default_rate_last_50_loans": 0.05,
        "avg_cure_time_hours": 12.0,
        "cure_time_volatility": 6.0,
        "repayment_stability_score": 0.70,
        "total_loans": 5.0,
        # ── capacity cap inputs — PRIMARY 30d/90d signals ────────────────────
        "avg_daily_balance_30d": 37646.15,
        "avg_daily_balance_90d": 37646.15,
        "avg_monthly_revenue_30d": 2868.0,
        "avg_monthly_revenue_90d": 2793.0,
        "avg_monthly_txn_count_30d": 14.0,
        "avg_monthly_txn_count_90d": 13.0,
        "avg_monthly_payments_30d": 42000.0,
        "avg_monthly_payments_90d": 39167.0,
        "active_customer_count_30d": 14.0,
        "active_customer_count_90d": 39.0,
        "avg_monthly_txn_volume_30d": 118000.0,
        "avg_monthly_txn_volume_90d": 145333.0,
        "operational_activity_flag": 1.0,
        "recent_credit_active_flag": 1.0,
        "is_peak_season_flag": 0.0,
        "agent_tier_ceiling_multiplier": 0.85,
        # ── recent usage cap inputs ──────────────────────────────────────────
        "recent_disbursement_amount_1m": 5000.0,
        "recent_disbursement_amount_3m": 10000.0,
        "recent_repayment_amount_1m": 5100.0,
        "recent_repayment_amount_3m": 10200.0,
        "recent_repayment_coverage_1m": 1.0,
        "recent_penalty_events_1m": 0.0,
        # ── prior exposure cap inputs ────────────────────────────────────────
        "avg_prior_loan_size": 5000.0,
        "max_prior_loan_size": 8000.0,
        "current_loan_size": 5000.0,
        "recent_repayment_performance": 0.95,
        # ── combine caps inputs ──────────────────────────────────────────────
        "capacity_cap": 15000.0,
        "recent_usage_cap": 8000.0,
        "prior_exposure_cap": 7000.0,
        "risk_cap": 20000.0,
        "is_thin_file": 0.0,
        "prior_limit": 0.0,
        # ── policy adjustment inputs ─────────────────────────────────────────
        "combined_cap": 10000.0,
        "risk_score": 0.20,
        "is_active_borrower": 1.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: prepare_transaction_capacity_features — sample data
# ─────────────────────────────────────────────────────────────────────────────

def test_prepare_transaction_capacity_features_sample_data():
    """
    agent_msisdn is renamed to msisdn; PRIMARY 30d/90d signals are derived;
    operational_activity_flag is binary; Silver Class → multiplier 0.85.
    """
    df = pd.DataFrame([_txn_row()])
    result = prepare_transaction_capacity_features(df)

    # Rename: msisdn present, agent_msisdn gone
    assert "msisdn" in result.columns
    assert "agent_msisdn" not in result.columns

    # All PRIMARY 30d/90d signal columns must exist
    for col in [
        "avg_daily_balance_30d", "avg_daily_balance_90d",
        "avg_monthly_revenue_30d", "avg_monthly_revenue_90d",
        "avg_monthly_txn_count_30d", "avg_monthly_txn_count_90d",
        "avg_monthly_payments_30d", "avg_monthly_payments_90d",
        "active_customer_count_30d", "active_customer_count_90d",
        "avg_monthly_txn_volume_30d", "avg_monthly_txn_volume_90d",
    ]:
        assert col in result.columns, f"Missing PRIMARY signal: {col}"

    row = result.iloc[0]

    # avg_daily_balance_30d == average_balance (direct assignment in code)
    assert row["avg_daily_balance_30d"] == pytest.approx(37646.15, rel=1e-3)

    # avg_monthly_revenue_30d == revenue_1m = sum of 1m commissions
    expected_revenue_1m = 547.0 + 473.0 + 1848.0  # cash_out + cash_in + payment comm
    assert row["avg_monthly_revenue_30d"] == pytest.approx(expected_revenue_1m, rel=1e-3)

    # avg_monthly_txn_count_30d == vol_1m
    assert row["avg_monthly_txn_count_30d"] == pytest.approx(14.0)

    # operational_activity_flag is binary {0, 1}
    assert result["operational_activity_flag"].isin([0, 1]).all()
    assert row["operational_activity_flag"] == 1  # vol_1m=14 > 0

    # Silver Class → 0.85
    assert row["agent_tier_ceiling_multiplier"] == pytest.approx(0.85)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2a: prepare_borrower_limit_features — all-null optional fields
# ─────────────────────────────────────────────────────────────────────────────

def test_prepare_borrower_limit_features_all_null_optional():
    """
    Function must not raise when all optional numeric fields are NaN;
    output contains msisdn and the engineered derived features.
    """
    row = _borrower_row()
    # Overwrite every nullable numeric column with None
    for col in [
        "total_disbursed_amount", "avg_loan_size_lifetime", "max_loan_size_lifetime",
        "lifetime_avg_hours_to_principal_cure", "lifetime_worst_hours_to_principal_cure",
        "lifetime_cure_time_volatility", "avg_prior_hours_to_cure",
        "worst_prior_hours_to_cure", "cure_time_volatility",
        "recent_3_avg_cure_time", "cure_time_trend", "borrower_trend",
        "loans_last_50_loans", "defaults_last_50_loans", "default_rate_last_50_loans",
        "prior_on_time_streak", "prior_default_streak",
        "loan_size_vs_avg_ratio", "loan_size_vs_max_ratio",
    ]:
        row[col] = None

    result = prepare_borrower_limit_features(pd.DataFrame([row]))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "msisdn" in result.columns
    for derived in ("exposure_tolerance_proxy", "recent_risk_proxy",
                    "stability_proxy", "borrower_tenure_days"):
        assert derived in result.columns, f"Missing derived column: {derived}"
    assert result.iloc[0]["msisdn"] == "256785"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2b: prepare_borrower_limit_features — new-to-credit borrower
# ─────────────────────────────────────────────────────────────────────────────

def test_prepare_borrower_limit_features_new_to_credit():
    """
    First-time borrower (0 prior loans, all history = 0) processes without error;
    total_loans < THIN_FILE_LOAN_THRESHOLD, stability_proxy = 0.
    """
    row = _borrower_row(
        total_loans=1,
        num_prior_loans=0,
        avg_prior_loan_size=0.0,
        max_prior_loan_size=0.0,
        avg_loan_size_lifetime=0.0,
        max_loan_size_lifetime=0.0,
        total_disbursed_amount=0.0,
        lifetime_on_time_24h_rate=0.0,
        lifetime_default_24h_rate=0.0,
        prior_on_time_24h_rate=0.0,
        avg_prior_hours_to_cure=0.0,
        recent_5_default_24h_rate=0.0,
        loans_last_50_loans=0,
        defaults_last_50_loans=0,
        default_rate_last_50_loans=0.0,
        prior_on_time_streak=0,
        prior_default_streak=0,
    )
    result = prepare_borrower_limit_features(pd.DataFrame([row]))

    assert len(result) == 1
    r = result.iloc[0]

    # Thin-file condition
    assert r["total_loans"] < THIN_FILE_LOAN_THRESHOLD

    # stability_proxy = on_time_streak - default_streak = 0 - 0 = 0
    assert r["stability_proxy"] == pytest.approx(0.0)

    # exposure_tolerance_proxy = max of all size cols = 0
    assert r["exposure_tolerance_proxy"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: build_extrafloat_limit_engine_features — alias columns
# ─────────────────────────────────────────────────────────────────────────────

def test_build_extrafloat_limit_engine_features_alias_columns():
    """
    All alias columns required by the caps engine are present;
    their values mirror the source borrower columns.
    """
    result = build_extrafloat_limit_engine_features(
        pd.DataFrame([_borrower_row()]),
        pd.DataFrame([_txn_row()]),
        pd.DataFrame([_loan_row()]),
    )

    assert len(result) == 1
    r = result.iloc[0]

    aliases = {
        "on_time_repayment_rate": "prior_on_time_24h_rate",
        "lifetime_default_rate": "lifetime_default_24h_rate",
        "default_rate_last_10_loans": "recent_5_default_24h_rate",
        "avg_cure_time_hours": "avg_prior_hours_to_cure",
        "current_loan_size": "latest_disbursed_amount",
    }
    for alias, source in aliases.items():
        assert alias in result.columns, f"Alias column missing: {alias}"
        assert r[alias] == pytest.approx(r[source], abs=1e-6), (
            f"{alias} ({r[alias]}) != {source} ({r[source]})"
        )

    # Derived flags
    assert "repayment_stability_score" in result.columns
    assert "is_thin_file" in result.columns
    assert r["is_thin_file"] == 0  # total_loans=5 >= 3


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: compute_risk_cap — score range [0,1], risk_cap bounded
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_risk_cap_score_bounds():
    """
    risk_score always in [0,1] and risk_cap always in [0,30000];
    perfect borrower reaches risk_score ≈ 1.0 and risk_cap ≈ 30000.
    """
    cases = [
        dict(on_time_repayment_rate=0.0, lifetime_default_rate=1.0,
             default_rate_last_10_loans=1.0, repayment_stability_score=0.0),
        dict(on_time_repayment_rate=0.5, lifetime_default_rate=0.5,
             default_rate_last_10_loans=0.5, repayment_stability_score=0.5),
        dict(on_time_repayment_rate=1.0, lifetime_default_rate=0.0,
             default_rate_last_10_loans=0.0, repayment_stability_score=1.0),
    ]
    for case in cases:
        result = compute_risk_cap(_features(**case))
        rs = float(result["risk_score"].iloc[0])
        rc = float(result["risk_cap"].iloc[0])
        assert 0.0 <= rs <= 1.0, f"risk_score={rs} out of [0,1] for {case}"
        assert 0.0 <= rc <= 30000.0, f"risk_cap={rc} out of [0,30000] for {case}"

    # Perfect borrower with enough loans for full experience factor
    perfect = _features(
        on_time_repayment_rate=1.0,
        lifetime_default_rate=0.0,
        default_rate_last_10_loans=0.0,
        default_rate_last_50_loans=0.0,
        avg_cure_time_hours=0.0,
        cure_time_volatility=0.0,
        repayment_stability_score=1.0,
        total_loans=20.0,
    )
    result_perfect = compute_risk_cap(perfect)
    assert float(result_perfect["risk_score"].iloc[0]) == pytest.approx(1.0, abs=0.01)
    assert float(result_perfect["risk_cap"].iloc[0]) == pytest.approx(30000.0, abs=200.0)

    # Worst borrower should score lower than mid borrower
    assert (
        float(compute_risk_cap(_features(**cases[0]))["risk_score"].iloc[0])
        < float(compute_risk_cap(_features(**cases[2]))["risk_score"].iloc[0])
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: compute_capacity_cap — primary path and fallback path
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_capacity_cap_primary_and_fallback():
    """
    Primary path: avg_daily_balance_30d present → capacity_cap > 0,
                  capacity_fallback_inputs == 0.
    Fallback path: only legacy columns (average_balance etc.) →
                   capacity_cap > 0, capacity_fallback_inputs > 0.
    """
    # Primary
    result_primary = compute_capacity_cap(_features())
    assert float(result_primary["capacity_cap"].iloc[0]) > 0.0
    assert float(result_primary["capacity_fallback_inputs"].iloc[0]) == 0.0

    # Fallback: drop all primary 30d/90d columns, add legacy fallback columns
    fallback_df = pd.DataFrame([{
        "average_balance": 37646.15,
        "revenue_1m": 2868.0,
        "vol_1m": 14.0,
        "payment_value_1m": 42000.0,
        "cust_1m": 14.0,
        "total_txn_value_1m": 118000.0,
        "operational_activity_flag": 1.0,
        "recent_credit_active_flag": 1.0,
        "is_peak_season_flag": 0.0,
        "agent_tier_ceiling_multiplier": 0.85,
    }])
    result_fallback = compute_capacity_cap(fallback_df)
    assert float(result_fallback["capacity_cap"].iloc[0]) > 0.0
    assert float(result_fallback["capacity_fallback_inputs"].iloc[0]) > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: compute_recent_usage_cap — activity gate and penalty haircut
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_recent_usage_cap_activity_gate_and_penalty():
    """
    Activity gate zeros the cap when blended total < 100.
    Penalty haircut: 2 events → multiplier = 0.80 → cap ~20% lower.

    Blended formula: 0.7*1m + 0.3*3m (both horizons affect the gate).
    """
    # Inactive: blended disb+repayment = 0.7*5 + 0.3*5 + 0.7*5 + 0.3*5 = 10 < 100
    inactive = _features(
        recent_disbursement_amount_1m=5.0,
        recent_disbursement_amount_3m=5.0,
        recent_repayment_amount_1m=5.0,
        recent_repayment_amount_3m=5.0,
    )
    result_inactive = compute_recent_usage_cap(inactive)
    assert float(result_inactive["recent_usage_active_flag"].iloc[0]) == 0
    assert float(result_inactive["recent_usage_cap"].iloc[0]) == pytest.approx(0.0)

    # Active, no penalties (blended ≈ 5000+5000 >> 100)
    active_clean = _features(
        recent_disbursement_amount_1m=5000.0,
        recent_disbursement_amount_3m=5000.0,
        recent_repayment_amount_1m=5000.0,
        recent_repayment_amount_3m=5000.0,
        recent_penalty_events_1m=0.0,
    )
    # Active, 2 penalty events → penalty_mult = clip(1 - 2*0.10, 0, 1) = 0.80
    active_penalty = _features(
        recent_disbursement_amount_1m=5000.0,
        recent_disbursement_amount_3m=5000.0,
        recent_repayment_amount_1m=5000.0,
        recent_repayment_amount_3m=5000.0,
        recent_penalty_events_1m=2.0,
    )

    result_clean = compute_recent_usage_cap(active_clean)
    result_penalty = compute_recent_usage_cap(active_penalty)

    cap_clean = float(result_clean["recent_usage_cap"].iloc[0])
    cap_penalty = float(result_penalty["recent_usage_cap"].iloc[0])
    penalty_mult = float(result_penalty["recent_usage_penalty_multiplier"].iloc[0])

    assert float(result_penalty["recent_usage_active_flag"].iloc[0]) == 1
    assert penalty_mult == pytest.approx(0.80, abs=0.001)
    assert cap_penalty < cap_clean
    # The 0.80 haircut should reduce cap by ~20%
    assert abs(cap_penalty / cap_clean - 0.80) < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: compute_prior_exposure_cap — new-to-credit vs existing
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_prior_exposure_cap_new_vs_existing():
    """
    New-to-credit (avg_prior=0, max_prior=0): cap = current_loan × 0.50.
    Existing borrower (avg_prior=5000, max_prior=8000): cap > new-to-credit cap.
    """
    new_df = _features(avg_prior_loan_size=0.0, max_prior_loan_size=0.0,
                       current_loan_size=5000.0, recent_repayment_performance=1.0)
    result_new = compute_prior_exposure_cap(new_df)
    cap_new = float(result_new["prior_exposure_cap"].iloc[0])

    # 5000 × 0.50 (new_to_credit_factor)
    assert cap_new == pytest.approx(2500.0, abs=1.0)
    assert result_new["prior_exposure_reason"].iloc[0] == "new_to_credit_proxy_cap"

    existing_df = _features(avg_prior_loan_size=5000.0, max_prior_loan_size=8000.0,
                            current_loan_size=5000.0, recent_repayment_performance=1.0)
    result_existing = compute_prior_exposure_cap(existing_df)
    cap_existing = float(result_existing["prior_exposure_cap"].iloc[0])

    assert cap_existing > cap_new
    assert result_existing["prior_exposure_reason"].iloc[0] != "new_to_credit_proxy_cap"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: combine_caps — thin-file vs standard weighting
# ─────────────────────────────────────────────────────────────────────────────

def test_combine_caps_thin_file_vs_standard():
    """
    Thin-file weights: capacity=0.25, recent_usage=0.15, prior_exposure=0.10, risk=0.50.
    Standard weights:  capacity=0.40, recent_usage=0.25, prior_exposure=0.15, risk=0.20.

    Scenario A (low risk_cap=2000): thin pre-guardrail < standard pre-guardrail.
      thin  = 0.25*10k + 0.15*10k + 0.10*10k + 0.50*2k = 6000
      std   = 0.40*10k + 0.25*10k + 0.15*10k + 0.20*2k = 8400

    Scenario B (risk_cap=30000, not binding): combined_reason reflects weighting type.
    """
    base = dict(capacity_cap=10000.0, recent_usage_cap=10000.0,
                prior_exposure_cap=10000.0, prior_limit=0.0)

    # Scenario A — guardrail binds; check pre-guardrail difference
    thin_A = combine_caps(_features(is_thin_file=1.0, risk_cap=2000.0, **base))
    std_A = combine_caps(_features(is_thin_file=0.0, risk_cap=2000.0, **base))

    thin_pre = float(thin_A["combined_cap_before_risk_guardrail"].iloc[0])
    std_pre = float(std_A["combined_cap_before_risk_guardrail"].iloc[0])
    assert thin_pre == pytest.approx(6000.0, abs=1.0)
    assert std_pre == pytest.approx(8400.0, abs=1.0)
    assert thin_pre < std_pre

    # Scenario B — guardrail doesn't bind; reason column distinguishes thin vs standard
    thin_B = combine_caps(_features(is_thin_file=1.0, risk_cap=30000.0, **base))
    std_B = combine_caps(_features(is_thin_file=0.0, risk_cap=30000.0, **base))

    assert thin_B["combined_reason"].iloc[0] == "thin_file_weighting_applied"
    assert std_B["combined_reason"].iloc[0] == "standard_weighting_applied"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: apply_policy_adjustments — tier classification, proven-good override
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_policy_adjustments_tier_and_proven_good():
    """
    Tier 1 (score ≤ 0.15) → multiplier 1.00; tier_4 (score > 0.60) → multiplier 0.40.
    Proven-good override (loans≥3, on_time≥0.90, lifetime_default≤0.05) floors
    policy_cap at 85% of combined_cap, overriding lower tier multipliers.
    """
    combined = 10000.0
    df = pd.DataFrame([
        # Tier 1
        dict(risk_score=0.10, combined_cap=combined, risk_cap=30000.0,
             total_loans=5.0, on_time_repayment_rate=0.70,
             lifetime_default_rate=0.15,
             recent_disbursement_amount_1m=5000.0,
             recent_repayment_amount_1m=5000.0,
             agent_tier_ceiling_multiplier=1.0),
        # Tier 4
        dict(risk_score=0.75, combined_cap=combined, risk_cap=30000.0,
             total_loans=5.0, on_time_repayment_rate=0.30,
             lifetime_default_rate=0.40,
             recent_disbursement_amount_1m=5000.0,
             recent_repayment_amount_1m=5000.0,
             agent_tier_ceiling_multiplier=1.0),
        # Tier 3 but proven-good → floor override
        # raw_cap = 10000 * 0.65 = 6500; proven_floor = 10000 * 0.85 = 8500
        dict(risk_score=0.50, combined_cap=combined, risk_cap=30000.0,
             total_loans=10.0, on_time_repayment_rate=0.95,
             lifetime_default_rate=0.02,
             recent_disbursement_amount_1m=5000.0,
             recent_repayment_amount_1m=5000.0,
             agent_tier_ceiling_multiplier=1.0),
    ])

    result = apply_policy_adjustments(df)

    assert result["risk_tier"].iloc[0] == "tier_1"
    assert result["risk_tier"].iloc[1] == "tier_4"

    tier1_cap = float(result["policy_cap"].iloc[0])
    tier4_cap = float(result["policy_cap"].iloc[1])
    assert tier1_cap > tier4_cap  # 10000 > 4000

    # Proven-good: policy_cap must be ≥ 85% of combined_cap
    proven_cap = float(result["policy_cap"].iloc[2])
    proven_floor = combined * 0.85
    assert proven_cap >= proven_floor - 1.0
    assert result["is_proven_good_borrower"].iloc[2] == 1
    assert result["policy_reason"].iloc[2] == "proven_good_floor_override"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: End-to-end — features → caps → assigned_limit
# ─────────────────────────────────────────────────────────────────────────────

def test_run_extrafloat_limit_engine_end_to_end():
    """
    Full pipeline: assigned_limit is present, in [0,30000], a multiple of 100,
    and non-NaN.  A healthy borrower with realistic signals gets limit > 0.
    """
    # risk_score must be pre-supplied for validate_required_columns
    df = _features(risk_score=0.20)

    result = run_extrafloat_limit_engine(df, validate_inputs=True)

    assert "assigned_limit" in result.columns
    assert "final_decision_reason" in result.columns

    assigned = result["assigned_limit"]
    assert assigned.notna().all()
    assert (assigned >= 0.0).all()
    assert (assigned <= 30000.0).all()

    # All non-zero limits must be multiples of 100
    non_zero = assigned[assigned > 0]
    if len(non_zero) > 0:
        assert (non_zero % 100 == 0).all(), (
            f"assigned_limit not rounded to 100: {non_zero.values}"
        )

    # A healthy borrower should receive a positive limit
    assert float(assigned.iloc[0]) > 0, (
        "Healthy borrower with realistic signals should get assigned_limit > 0"
    )

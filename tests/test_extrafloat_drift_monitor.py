"""
tests/test_extrafloat_drift_monitor.py
=======================================
10 tests for the ExtraFloat drift monitoring module.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

import extrafloat_drift_monitor as edm
from extrafloat_drift_monitor import (
    DEFAULT_DRIFT_CONFIG,
    CompositionDriftResult,
    DriftReport,
    FeatureDriftResult,
    PolicyHealthResult,
    SEVERITY_ALERT,
    SEVERITY_MONITOR,
    SEVERITY_STABLE,
    _SCIPY_AVAILABLE,
    _aggregate_severity,
    _compute_psi,
    monitor_cap_driver_drift,
    monitor_composition_drift,
    monitor_input_drift,
    monitor_output_drift,
    monitor_policy_health,
    run_drift_monitor,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _engine_output_df(
    n: int = 200,
    *,
    rng_seed: int = 42,
    risk_tier_dist: dict = None,
    reg_cap_frac: float = 0.0,
    thin_file_frac: float = 0.10,
    combined_top_driver: str = None,
    proven_good_frac: float = 0.15,
    balance_mean: float = 50_000.0,
    assigned_limit_mean: float = 80_000.0,
) -> pd.DataFrame:
    """Build a synthetic engine-output DataFrame for drift monitor testing."""
    rng = np.random.default_rng(rng_seed)

    if risk_tier_dist is None:
        risk_tier_dist = {
            "tier_1": 0.40, "tier_2": 0.30,
            "tier_3": 0.20, "tier_4": 0.10,
        }

    tiers = list(risk_tier_dist.keys())
    probs = [risk_tier_dist[t] for t in tiers]
    tier_col = rng.choice(tiers, size=n, p=probs)

    reg_cap = (rng.random(n) < reg_cap_frac).astype(int)
    thin    = (rng.random(n) < thin_file_frac).astype(int)
    proven  = (rng.random(n) < proven_good_frac).astype(int)

    if combined_top_driver is None:
        driver_col = rng.choice(
            ["capacity_component", "risk_component",
             "recent_usage_component", "prior_exposure_component"],
            size=n, p=[0.50, 0.25, 0.15, 0.10],
        )
    else:
        driver_col = np.full(n, combined_top_driver, dtype=object)

    return pd.DataFrame({
        # input features
        "avg_daily_balance_30d":        rng.normal(balance_mean, 10_000, n).clip(0),
        "avg_monthly_revenue_30d":      rng.normal(5_000, 1_000, n).clip(0),
        "avg_monthly_txn_count_30d":    rng.normal(20, 5, n).clip(1),
        "avg_monthly_payments_30d":     rng.normal(30_000, 5_000, n).clip(0),
        "active_customer_count_30d":    rng.normal(15, 4, n).clip(1),
        "avg_monthly_txn_volume_30d":   rng.normal(100_000, 20_000, n).clip(0),
        "on_time_repayment_rate":       rng.beta(8, 2, n),
        "lifetime_default_rate":        rng.beta(1, 20, n),
        "default_rate_last_10_loans":   rng.beta(1, 20, n),
        "avg_cure_time_hours":          rng.uniform(0, 24, n),
        "repayment_stability_score":    rng.beta(7, 3, n),
        "recent_disbursement_amount_1m": rng.normal(5_000, 1_000, n).clip(0),
        "recent_repayment_amount_1m":   rng.normal(5_100, 1_000, n).clip(0),
        "recent_repayment_coverage_1m": rng.beta(8, 2, n),
        "recent_penalty_events_1m":     rng.choice([0, 1, 2], n, p=[0.85, 0.10, 0.05]),
        # output features
        "assigned_limit":               rng.normal(assigned_limit_mean, 15_000, n).clip(0),
        "risk_score":                   rng.beta(7, 3, n),
        "capacity_cap":                 rng.normal(90_000, 20_000, n).clip(0),
        "recent_usage_cap":             rng.normal(70_000, 15_000, n).clip(0),
        "prior_exposure_cap":           rng.normal(75_000, 15_000, n).clip(0),
        "risk_cap":                     rng.normal(85_000, 18_000, n).clip(0),
        "combined_cap":                 rng.normal(80_000, 16_000, n).clip(0),
        "policy_cap":                   rng.normal(78_000, 15_000, n).clip(0),
        # categorical / flag columns
        "risk_tier":                    tier_col,
        "combined_top_driver":          driver_col,
        "capacity_top_driver":          rng.choice(["balance", "revenue", "txn"], n),
        "policy_reason":                rng.choice(["tier_1_policy", "tier_2_policy",
                                                     "tier_3_policy", "tier_4_policy"], n),
        "is_thin_file":                 thin,
        "regulatory_cap_applied":       reg_cap,
        "is_proven_good_borrower":      proven,
        "active_floor_applied":         (rng.random(n) < 0.05).astype(int),
        "is_kyc_blocked":               (rng.random(n) < 0.02).astype(int),
        "recent_usage_active_flag":     (rng.random(n) > 0.05).astype(int),
        "capacity_fallback_inputs":     (rng.random(n) < 0.03).astype(int),
        "capacity_missing_inputs":      (rng.random(n) < 0.01).astype(int),
    })


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — PSI stable population
# ─────────────────────────────────────────────────────────────────────────────

def test_psi_stable_population():
    """Same distribution → PSI < 0.10 (stable)."""
    rng = np.random.default_rng(42)
    ref = rng.normal(50_000, 10_000, 1000)
    cur = rng.normal(50_000, 10_000, 1000)

    psi = _compute_psi(ref, cur, n_bins=10)
    assert psi < 0.10, f"Expected PSI < 0.10 for identical distributions, got {psi:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — PSI moderate drift
# ─────────────────────────────────────────────────────────────────────────────

def test_psi_moderate_drift():
    """Mean shifted by 0.35 std → PSI in monitor zone [0.10, 0.25)."""
    rng = np.random.default_rng(99)
    std = 10_000
    ref = rng.normal(50_000, std, 2000)
    cur = rng.normal(50_000 + 0.35 * std, std, 2000)

    psi = _compute_psi(ref, cur, n_bins=10)
    assert 0.10 <= psi < 0.25, (
        f"Expected PSI in [0.10, 0.25) for 0.35-std shift, got {psi:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — PSI severe drift
# ─────────────────────────────────────────────────────────────────────────────

def test_psi_severe_drift():
    """Completely different distributions → PSI > 0.25 (alert)."""
    rng = np.random.default_rng(42)
    ref = rng.normal(50_000, 5_000, 2000)
    cur = rng.uniform(0, 400_000, 2000)

    psi = _compute_psi(ref, cur, n_bins=10)
    assert psi > 0.25, f"Expected PSI > 0.25 for severely different distributions, got {psi:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Population composition: no drift
# ─────────────────────────────────────────────────────────────────────────────

def test_population_composition_no_drift():
    """Same tier mix in ref and cur → composition result is stable."""
    rng = np.random.default_rng(42)
    tier_mix = (
        ["tier_1"] * 400 + ["tier_2"] * 300 +
        ["tier_3"] * 200 + ["tier_4"] * 100
    )
    ref_df = pd.DataFrame({"risk_tier": rng.permutation(tier_mix)})
    cur_df = pd.DataFrame({"risk_tier": rng.permutation(tier_mix)})

    results = monitor_composition_drift(ref_df, cur_df)
    tier_r = next((r for r in results if r.feature == "risk_tier"), None)
    assert tier_r is not None, "risk_tier result missing"
    assert tier_r.severity == SEVERITY_STABLE, (
        f"Identical tier mix should be stable, got {tier_r.severity}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Population composition: drift detected
# ─────────────────────────────────────────────────────────────────────────────

def test_population_composition_drift():
    """Dramatically shifted tier mix triggers monitor or alert."""
    ref_df = pd.DataFrame({
        "risk_tier": (
            ["tier_1"] * 400 + ["tier_2"] * 300 +
            ["tier_3"] * 200 + ["tier_4"] * 100
        )
    })
    cur_df = pd.DataFrame({
        "risk_tier": (
            ["tier_1"] * 50 + ["tier_2"] * 100 +
            ["tier_3"] * 250 + ["tier_4"] * 600
        )
    })

    results = monitor_composition_drift(ref_df, cur_df)
    tier_r = next((r for r in results if r.feature == "risk_tier"), None)
    assert tier_r is not None, "risk_tier result missing"
    assert tier_r.severity in (SEVERITY_MONITOR, SEVERITY_ALERT), (
        f"Shifted tier mix should not be stable, got {tier_r.severity}"
    )
    assert tier_r.max_absolute_shift > 0.10, (
        f"Expected absolute shift > 10 pp, got {tier_r.max_absolute_shift:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Policy calibration: regulatory cap alert
# ─────────────────────────────────────────────────────────────────────────────

def test_policy_calibration_regulatory_cap_alert():
    """More than 20% of rows hitting regulatory cap → alert."""
    n = 500
    ref_df = pd.DataFrame({"regulatory_cap_applied": np.zeros(n, dtype=int)})
    cur_df = pd.DataFrame({
        "regulatory_cap_applied": np.array(
            [1] * 120 + [0] * 380, dtype=int
        )
    })

    results = monitor_policy_health(ref_df, cur_df)
    r = next((x for x in results if x.metric == "regulatory_cap_applied_rate"), None)
    assert r is not None, "regulatory_cap_applied_rate metric missing"
    assert r.severity == SEVERITY_ALERT, (
        f"24% regulatory cap rate should be alert, got {r.severity}"
    )
    assert r.cur_rate == pytest.approx(0.24, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Cap driver composition shift detected
# ─────────────────────────────────────────────────────────────────────────────

def test_cap_driver_composition_shift():
    """Binding cap shifts from capacity_component → risk_component: detected."""
    ref_df = pd.DataFrame({
        "combined_top_driver": (
            ["capacity_component"] * 600 +
            ["risk_component"]     * 200 +
            ["recent_usage_component"]  * 150 +
            ["prior_exposure_component"] * 50
        )
    })
    cur_df = pd.DataFrame({
        "combined_top_driver": (
            ["capacity_component"] * 200 +
            ["risk_component"]     * 600 +
            ["recent_usage_component"]  * 150 +
            ["prior_exposure_component"] * 50
        )
    })

    results = monitor_cap_driver_drift(ref_df, cur_df)
    r = next((x for x in results if x.feature == "combined_top_driver"), None)
    assert r is not None, "combined_top_driver result missing"
    assert r.severity in (SEVERITY_MONITOR, SEVERITY_ALERT), (
        f"Binding-cap shift should not be stable, got {r.severity}"
    )
    assert r.max_absolute_shift >= 0.35, (
        f"Expected absolute shift >= 35 pp, got {r.max_absolute_shift:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — DriftReport severity aggregation
# ─────────────────────────────────────────────────────────────────────────────

def test_drift_report_severity_aggregation():
    """overall_severity is the maximum severity across all sub-results."""
    stable_r = FeatureDriftResult(
        feature="f1", psi=0.05, psi_severity=SEVERITY_STABLE,
        ks_statistic=None, ks_pvalue=None, ks_significant=None,
        percentile_shift=None, severity=SEVERITY_STABLE,
    )
    alert_r = FeatureDriftResult(
        feature="f2", psi=0.30, psi_severity=SEVERITY_ALERT,
        ks_statistic=None, ks_pvalue=None, ks_significant=None,
        percentile_shift=None, severity=SEVERITY_ALERT,
    )
    monitor_r = PolicyHealthResult(
        metric="thin_file_rate", ref_rate=0.10, cur_rate=0.17,
        absolute_change=0.07, relative_change=0.70,
        threshold=0.05, severity=SEVERITY_MONITOR,
    )

    overall = _aggregate_severity([[stable_r, alert_r], [monitor_r]])
    assert overall == SEVERITY_ALERT, (
        f"alert > monitor > stable, expected alert got {overall}"
    )

    report = DriftReport(
        input_drift=[stable_r, alert_r],
        output_drift=[],
        composition_drift=[],
        policy_health=[monitor_r],
        cap_driver_drift=[],
        overall_severity=overall,
        scipy_available=_SCIPY_AVAILABLE,
        ref_row_count=1000,
        cur_row_count=1000,
        skipped_features=[],
    )

    summary = report.summary_dict()
    assert summary["overall_severity"] == SEVERITY_ALERT
    assert summary["input_features_alert_count"] == 1
    assert summary["input_features_stable_count"] == 1
    assert summary["policy_health_monitor_count"] == 1
    assert "top_input_alerts" in summary
    assert "f2" in summary["top_input_alerts"]


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — Graceful degradation without scipy
# ─────────────────────────────────────────────────────────────────────────────

def test_graceful_degradation_without_scipy():
    """Module works without scipy: PSI computed, KS fields are None."""
    rng = np.random.default_rng(42)
    ref_df = pd.DataFrame({
        "avg_daily_balance_30d":  rng.normal(50_000, 10_000, 200),
        "on_time_repayment_rate": rng.beta(8, 2, 200),
    })
    cur_df = pd.DataFrame({
        "avg_daily_balance_30d":  rng.normal(60_000, 10_000, 200),
        "on_time_repayment_rate": rng.beta(7, 3, 200),
    })

    original = edm._SCIPY_AVAILABLE
    try:
        edm._SCIPY_AVAILABLE = False
        results = monitor_input_drift(ref_df, cur_df)

        for r in results:
            assert r.psi is not None, f"{r.feature}: PSI must be computed even without scipy"
            assert r.ks_statistic is None, f"{r.feature}: ks_statistic must be None without scipy"
            assert r.ks_pvalue is None,    f"{r.feature}: ks_pvalue must be None without scipy"
            assert r.ks_significant is None
    finally:
        edm._SCIPY_AVAILABLE = original


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — Full drift run with engine output
# ─────────────────────────────────────────────────────────────────────────────

def test_full_drift_run_with_engine_output():
    """End-to-end: build two synthetic output DataFrames, run run_drift_monitor,
    verify DriftReport is fully populated and structurally valid."""
    ref_df = _engine_output_df(n=300, rng_seed=1, balance_mean=50_000,
                                assigned_limit_mean=80_000)
    # Cur window: slight input shift (higher balances → some output shift)
    cur_df = _engine_output_df(n=300, rng_seed=2, balance_mean=65_000,
                                assigned_limit_mean=90_000)

    report = run_drift_monitor(ref_df, cur_df, monitor_inputs=True, monitor_outputs=True)

    assert isinstance(report, DriftReport)
    assert report.ref_row_count == 300
    assert report.cur_row_count == 300
    assert report.overall_severity in (SEVERITY_STABLE, SEVERITY_MONITOR, SEVERITY_ALERT)
    assert isinstance(report.scipy_available, bool)
    assert len(report.input_drift)  > 0, "input_drift should not be empty"
    assert len(report.output_drift) > 0, "output_drift should not be empty"
    assert len(report.policy_health) > 0, "policy_health should not be empty"

    for r in report.input_drift:
        assert r.psi is not None
        assert r.severity in (SEVERITY_STABLE, SEVERITY_MONITOR, SEVERITY_ALERT)

    for r in report.output_drift:
        assert r.psi is not None
        assert r.severity in (SEVERITY_STABLE, SEVERITY_MONITOR, SEVERITY_ALERT)

    summary = report.summary_dict()
    assert "overall_severity" in summary
    assert summary["ref_row_count"] == 300
    assert summary["cur_row_count"] == 300
    assert isinstance(summary["top_input_alerts"], list)
    assert isinstance(summary["top_output_alerts"], list)

    # Identical ref == cur → overall must be stable
    same_report = run_drift_monitor(ref_df, ref_df, monitor_inputs=True, monitor_outputs=True)
    assert same_report.overall_severity == SEVERITY_STABLE, (
        "Identical ref and cur should produce stable overall severity"
    )

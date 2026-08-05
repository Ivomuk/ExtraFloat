"""
Test 8: Business-process co-definition diagnostic.

Hypothesis: net_exposure_6M > 0 (= currently_outstanding_flag) is a near-deterministic
precondition for an XtraFloat penalty. If the penalty system automatically triggers
when outstanding debt exists at reconciliation time, then:

    P(bad_state = 1 | outstanding = 1) >> P(bad_state = 1 | outstanding = 0)

This is NOT temporal leakage — both the outstanding flag and the label are correctly
dated to the pre-snapshot and post-snapshot periods respectively. However, it IS
conceptual co-definition: the feature and the label share a common business-process
ancestor (the lending and penalty enforcement system).

Evidence of strong co-definition (warrants investigation):
  - bad_rate[outstanding=1] > 0.30  AND  bad_rate[outstanding=0] < 0.05
  - lift for outstanding segment > 3×

Evidence that co-definition is tolerable (model learns real credit risk):
  - bad_rate[outstanding=1] < 0.30 — being outstanding does NOT guarantee a penalty
  - Many features combine to explain variance not explained by net_exposure_6M alone

The fix applied in this commit removes `currently_outstanding_flag` from the model
(it is added to PD_FEATURE_BLACKLIST). `net_exposure_6M` itself is retained as a
legitimate continuous behavioral signal.

To run the diagnostic on real training data:
    from pd_model.tests.test_business_process_codef import business_process_codef_report
    import pandas as pd
    ops = pd.read_csv("path/to/ops_scored.csv")
    # Join with net_exposure_6M from the feature matrix if needed
    report = business_process_codef_report(ops)
    print(report)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def business_process_codef_report(
    df: pd.DataFrame,
    outstanding_col: str = "currently_outstanding_flag",
    net_exposure_col: str = "net_exposure_6M",
    label_col: str = "bad_state",
) -> pd.DataFrame:
    """Compute bad rates by outstanding status to assess co-definition risk.

    Returns DataFrame with columns:
        [segment, outstanding_flag, n, bad_rate, share_of_population, lift]
    """
    df = df.copy()

    if outstanding_col not in df.columns:
        if net_exposure_col not in df.columns:
            raise KeyError(
                f"Neither {outstanding_col!r} nor {net_exposure_col!r} found in df"
            )
        df[outstanding_col] = (pd.to_numeric(df[net_exposure_col], errors="coerce") > 0).astype(int)

    label = pd.to_numeric(df[label_col], errors="coerce")
    outstanding = pd.to_numeric(df[outstanding_col], errors="coerce")
    overall_bad_rate = float(label.mean())

    records = []
    for seg_val, seg_name in [(0, "not_outstanding"), (1, "outstanding")]:
        mask = outstanding == seg_val
        n = int(mask.sum())
        bad_rate = float(label[mask].mean()) if n > 0 else np.nan
        lift = (bad_rate / overall_bad_rate) if overall_bad_rate > 0 and not np.isnan(bad_rate) else np.nan
        records.append({
            "segment": seg_name,
            "outstanding_flag": seg_val,
            "n": n,
            "bad_rate": round(bad_rate, 4) if not np.isnan(bad_rate) else np.nan,
            "share_of_population": round(n / len(df), 4),
            "lift": round(lift, 3) if not np.isnan(lift) else np.nan,
        })

    return pd.DataFrame(records)


def simulate_codef_scenario(
    n: int = 2000,
    outstanding_to_bad_rate: float = 0.60,
    non_outstanding_to_bad_rate: float = 0.02,
    outstanding_prevalence: float = 0.25,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate synthetic data with controlled business-process co-definition.

    Args:
        n: Number of agents.
        outstanding_to_bad_rate: P(bad_state=1 | outstanding=1).
        non_outstanding_to_bad_rate: P(bad_state=1 | outstanding=0).
        outstanding_prevalence: Fraction of agents with outstanding debt.
        seed: RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    outstanding = (rng.uniform(0, 1, n) < outstanding_prevalence).astype(int)
    bad_rate_per_agent = np.where(
        outstanding == 1, outstanding_to_bad_rate, non_outstanding_to_bad_rate
    )
    bad_state = (rng.uniform(0, 1, n) < bad_rate_per_agent).astype(int)
    net_exposure = outstanding * rng.uniform(1_000, 500_000, n)  # >0 iff outstanding

    return pd.DataFrame({
        "agent_msisdn": [f"256{i:07d}" for i in range(n)],
        "currently_outstanding_flag": outstanding,
        "net_exposure_6M": net_exposure,
        "bad_state": bad_state,
        "disbursement_vol_m5": rng.uniform(0, 1_000_000, n),
    })


class TestBusinessProcessCodefinition:
    def test_report_correct_bad_rates_under_strong_codef(self):
        """Report must accurately separate bad rates when co-definition is strong."""
        df = simulate_codef_scenario(
            n=5000,
            outstanding_to_bad_rate=0.60,
            non_outstanding_to_bad_rate=0.02,
            outstanding_prevalence=0.25,
            seed=1,
        )
        report = business_process_codef_report(df)
        out_br = report.loc[report["segment"] == "outstanding", "bad_rate"].iloc[0]
        not_out_br = report.loc[report["segment"] == "not_outstanding", "bad_rate"].iloc[0]
        assert 0.50 <= out_br <= 0.70, (
            f"Outstanding bad rate expected ~0.60, got {out_br:.4f}"
        )
        assert 0.00 <= not_out_br <= 0.08, (
            f"Non-outstanding bad rate expected ~0.02, got {not_out_br:.4f}"
        )

    def test_lift_exceeds_three_under_strong_codef(self):
        """Under strong co-definition, lift for outstanding segment must exceed 3×."""
        df = simulate_codef_scenario(
            n=5000,
            outstanding_to_bad_rate=0.60,
            non_outstanding_to_bad_rate=0.02,
            outstanding_prevalence=0.25,
            seed=2,
        )
        report = business_process_codef_report(df)
        out_lift = report.loc[report["segment"] == "outstanding", "lift"].iloc[0]
        assert out_lift > 3.0, (
            f"Expected lift > 3 under strong co-definition, got {out_lift:.2f}"
        )

    def test_no_codef_both_lifts_near_one(self):
        """When both groups share the same bad rate, all lifts must be ~1.0."""
        df = simulate_codef_scenario(
            n=5000,
            outstanding_to_bad_rate=0.15,
            non_outstanding_to_bad_rate=0.15,
            outstanding_prevalence=0.25,
            seed=3,
        )
        report = business_process_codef_report(df)
        for _, row in report.iterrows():
            assert 0.5 <= row["lift"] <= 2.0, (
                f"Segment '{row['segment']}' lift={row['lift']:.2f} — "
                "expected near 1.0 when no co-definition exists"
            )

    def test_report_derives_flag_from_net_exposure_when_missing(self):
        """When currently_outstanding_flag is absent, net_exposure_6M is used as fallback."""
        df = simulate_codef_scenario(n=500, seed=4)
        df = df.drop(columns=["currently_outstanding_flag"])
        report = business_process_codef_report(df, outstanding_col="currently_outstanding_flag")
        assert len(report) == 2
        assert set(report["segment"]) == {"outstanding", "not_outstanding"}

    def test_report_edge_case_all_outstanding(self):
        """Edge case: entire population outstanding — not_outstanding has n=0."""
        rng = np.random.default_rng(5)
        n = 100
        df = pd.DataFrame({
            "currently_outstanding_flag": np.ones(n, dtype=int),
            "net_exposure_6M": rng.uniform(1, 100, n),
            "bad_state": rng.integers(0, 2, n),
        })
        report = business_process_codef_report(df)
        not_out_n = report.loc[report["segment"] == "not_outstanding", "n"].iloc[0]
        assert not_out_n == 0

    def test_simulation_outstanding_prevalence_correct(self):
        """Simulation must produce roughly the requested outstanding prevalence."""
        df = simulate_codef_scenario(n=10_000, outstanding_prevalence=0.30, seed=6)
        actual = df["currently_outstanding_flag"].mean()
        assert 0.25 <= actual <= 0.35, (
            f"Expected outstanding prevalence ~0.30, got {actual:.3f}"
        )

    def test_simulation_net_exposure_zero_when_not_outstanding(self):
        """In the simulation, net_exposure_6M must be 0 for non-outstanding agents."""
        df = simulate_codef_scenario(n=1000, seed=7)
        not_outstanding = df[df["currently_outstanding_flag"] == 0]
        assert (not_outstanding["net_exposure_6M"] == 0).all(), (
            "net_exposure_6M must be 0 for non-outstanding agents in the simulation"
        )

    def test_report_returns_two_rows(self):
        """Report must always have exactly two rows (outstanding / not_outstanding)."""
        df = simulate_codef_scenario(n=200, seed=8)
        report = business_process_codef_report(df)
        assert len(report) == 2
        assert set(report["segment"]) == {"outstanding", "not_outstanding"}

    def test_currently_outstanding_flag_not_in_model_features(self):
        """
        Regression guard: currently_outstanding_flag must be absent from model features.
        This test documents the fix applied alongside this test file:
        the column was added to PD_FEATURE_BLACKLIST to prevent it from entering the model.
        """
        from pd_model.config.feature_config import PD_FEATURE_BLACKLIST
        assert "currently_outstanding_flag" in PD_FEATURE_BLACKLIST, (
            "'currently_outstanding_flag' is not in PD_FEATURE_BLACKLIST. "
            "This is a business-process co-definition proxy that must be excluded "
            "from model features. See loan_features.py:605 for where it is computed."
        )

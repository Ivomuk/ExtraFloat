"""
Tests for the CreditRisk end-to-end pipeline integration.

These tests use synthetic DataFrames and a mock PD model -- no trained artifacts
are required. They verify:

  1. cal_pd flows through compute_risk_cap() correctly (short-circuit path).
  2. risk_score = 1 - cal_pd for agents that have a PD score.
  3. Agents missing cal_pd (join misses) fall back to the 7-signal blend.
  4. cal_pd appears in FINAL_OUTPUT_COLUMNS when keep_intermediate=False.
  5. run_credit_risk_pipeline() joins PD output onto engine features and
     produces the expected output columns.
  6. experience_factor is NOT applied on the cal_pd path (fix 3).
  7. Preflight check raises a clear error when artifacts are missing (fix 2).
  8. Preflight check passes silently when all artifacts are present (fix 2).
"""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from extrafloat.engine.extrafloat_limit_engine_caps import (
    _get_config,
    compute_risk_cap,
)
from extrafloat.engine.run_extrafloat_limit_engine import (
    FINAL_OUTPUT_COLUMNS,
    run_extrafloat_limit_engine,
)
from pd_model.exceptions import DataAlignmentError

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _minimal_features_df(n: int = 5, seed: int = 42) -> pd.DataFrame:
    """Minimal engine-compatible features DataFrame with all optional signals."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "msisdn": [f"256700{i:06d}" for i in range(n)],
            # Capacity signals
            "avg_balance_30d": rng.uniform(5e5, 5e6, n),
            "avg_balance_90d": rng.uniform(5e5, 5e6, n),
            "net_cashflow_30d": rng.uniform(1e5, 1e6, n),
            "net_cashflow_90d": rng.uniform(1e5, 1e6, n),
            "txn_count_30d": rng.integers(10, 200, n).astype(float),
            "txn_count_90d": rng.integers(30, 600, n).astype(float),
            "payments_in_30d": rng.uniform(1e5, 5e5, n),
            "payments_in_90d": rng.uniform(3e5, 1.5e6, n),
            "active_customers_30d": rng.integers(5, 100, n).astype(float),
            "active_customers_90d": rng.integers(15, 300, n).astype(float),
            "recent_disbursement_volume": rng.uniform(1e5, 1e6, n),
            "recent_repayment_volume": rng.uniform(1e5, 1e6, n),
            "recent_repayment_performance": rng.uniform(0.5, 1.0, n),
            "recent_penalty_count": rng.integers(0, 3, n).astype(float),
            "coverage_ratio": rng.uniform(0.7, 1.2, n),
            # Risk signals (used in fallback path)
            "on_time_repayment_rate": rng.uniform(0.6, 1.0, n),
            "lifetime_default_rate": rng.uniform(0.0, 0.2, n),
            "default_rate_last_10_loans": rng.uniform(0.0, 0.15, n),
            "default_rate_last_50_loans": rng.uniform(0.0, 0.15, n),
            "avg_cure_time_hours": rng.uniform(0.0, 48.0, n),
            "cure_time_volatility": rng.uniform(0.0, 24.0, n),
            "repayment_stability_score": rng.uniform(0.5, 1.0, n),
            # Prior exposure
            "avg_prior_loan_size": rng.uniform(1e4, 5e5, n),
            "max_prior_loan_size": rng.uniform(2e4, 8e5, n),
            "current_loan_size": rng.uniform(1e4, 3e5, n),
            "prior_limit": rng.uniform(1e4, 5e5, n),
            # Borrower attributes
            "is_thin_file": rng.integers(0, 2, n).astype(float),
            "is_active_borrower": rng.integers(0, 2, n).astype(float),
            "lifetime_loan_count": rng.integers(1, 30, n).astype(float),
            "total_loans": rng.integers(1, 30, n).astype(float),
            # Agent tier
            "agent_tier": rng.choice(["gold", "platinum", "silver"], n),
        }
    )


# -----------------------------------------------------------------------------
# 1. compute_risk_cap -- cal_pd short-circuit path
# -----------------------------------------------------------------------------


class TestComputeRiskCapCalPdPath:
    def test_risk_score_equals_one_minus_cal_pd(self):
        df = _minimal_features_df(n=10)
        cal_pd_values = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
        df["cal_pd"] = cal_pd_values

        result = compute_risk_cap(df)

        expected = np.clip(1.0 - cal_pd_values, 0.0, 1.0)
        np.testing.assert_allclose(
            result["risk_score"].values,
            expected,
            atol=1e-9,
            err_msg="risk_score should equal 1 - cal_pd when cal_pd is present",
        )

    def test_cal_pd_path_takes_priority_over_signals(self):
        df = _minimal_features_df(n=5)
        # Force signals to produce a very different risk_score (all-zero = 0.0)
        df["on_time_repayment_rate"] = 0.0
        df["lifetime_default_rate"] = 1.0
        # But cal_pd says 0.20 -> risk_score should be 0.80, not 0.0
        df["cal_pd"] = 0.20

        result = compute_risk_cap(df)
        np.testing.assert_allclose(result["risk_score"].values, 0.80, atol=1e-9)

    def test_high_cal_pd_maps_to_low_risk_score(self):
        df = _minimal_features_df(n=3)
        df["cal_pd"] = [0.05, 0.50, 0.95]
        result = compute_risk_cap(df)
        assert result["risk_score"].iloc[0] > result["risk_score"].iloc[1]
        assert result["risk_score"].iloc[1] > result["risk_score"].iloc[2]

    def test_risk_cap_is_clipped_within_global_bounds(self):
        cfg = _get_config(None)
        df = _minimal_features_df(n=5)
        df["cal_pd"] = 0.01  # very safe -> high risk_score -> high risk_cap
        result = compute_risk_cap(df, config=cfg)
        assert (result["risk_cap"] <= cfg["global_ceiling_limit"]).all()
        assert (result["risk_cap"] >= cfg["global_floor_limit"]).all()


# -----------------------------------------------------------------------------
# 2. compute_risk_cap -- 7-signal fallback path
# -----------------------------------------------------------------------------


class TestComputeRiskCapFallback:
    def test_fallback_used_when_cal_pd_absent(self):
        df = _minimal_features_df(n=5)
        assert "cal_pd" not in df.columns
        result = compute_risk_cap(df)
        assert "risk_score" in result.columns
        assert result["risk_score"].notna().all()

    def test_fallback_used_when_cal_pd_all_nan(self):
        df = _minimal_features_df(n=5)
        df["cal_pd"] = np.nan
        result = compute_risk_cap(df)
        # Should fall back to signal blend; result must differ from 1 - NaN (which is NaN)
        assert result["risk_score"].notna().all()

    def test_fallback_risk_score_in_valid_range(self):
        df = _minimal_features_df(n=20)
        result = compute_risk_cap(df)
        assert (result["risk_score"] >= 0.0).all()
        assert (result["risk_score"] <= 1.0).all()


# -----------------------------------------------------------------------------
# 3. FINAL_OUTPUT_COLUMNS includes cal_pd
# -----------------------------------------------------------------------------


def test_final_output_columns_includes_cal_pd():
    assert "cal_pd" in FINAL_OUTPUT_COLUMNS, (
        "cal_pd must be in FINAL_OUTPUT_COLUMNS so it survives keep_intermediate=False"
    )


def test_cal_pd_present_in_trimmed_output():
    df = _minimal_features_df(n=5)
    df["cal_pd"] = 0.15
    result = run_extrafloat_limit_engine(df, keep_intermediate=False)
    assert "cal_pd" in result.columns
    assert "assigned_limit" in result.columns


# -----------------------------------------------------------------------------
# 4. Full engine run with cal_pd
# -----------------------------------------------------------------------------


class TestEngineWithCalPd:
    def test_end_to_end_with_cal_pd(self):
        df = _minimal_features_df(n=20)
        df["cal_pd"] = np.linspace(0.02, 0.90, 20)
        result = run_extrafloat_limit_engine(df, keep_intermediate=True)

        assert "assigned_limit" in result.columns
        assert "risk_tier" in result.columns
        assert "cal_pd" in result.columns
        assert (result["assigned_limit"] >= 0).all()

    def test_low_cal_pd_produces_higher_limits_than_high_cal_pd(self):
        df_safe = _minimal_features_df(n=10, seed=1)
        df_safe["cal_pd"] = 0.05  # very safe

        df_risky = _minimal_features_df(n=10, seed=1)
        df_risky["cal_pd"] = 0.85  # very risky

        result_safe = run_extrafloat_limit_engine(df_safe, keep_intermediate=False)
        result_risky = run_extrafloat_limit_engine(df_risky, keep_intermediate=False)

        assert result_safe["assigned_limit"].mean() > result_risky["assigned_limit"].mean(), (
            "Agents with low cal_pd (low default risk) should receive higher limits"
        )

    def test_tier_1_assigned_to_low_cal_pd_agents(self):
        df = _minimal_features_df(n=10)
        df["cal_pd"] = 0.05  # risk_score = 0.95 -> tier_1 (>= 0.85)
        result = run_extrafloat_limit_engine(df, keep_intermediate=True)
        assert (result["risk_tier"] == "tier_1").all()

    def test_tier_4_assigned_to_high_cal_pd_agents(self):
        df = _minimal_features_df(n=10)
        df["cal_pd"] = 0.90  # risk_score = 0.10 -> tier_4 (< 0.35)
        result = run_extrafloat_limit_engine(df, keep_intermediate=True)
        assert (result["risk_tier"] == "tier_4").all()


# -----------------------------------------------------------------------------
# 5. Pipeline join: agents missing cal_pd fall back to 7-signal blend
# -----------------------------------------------------------------------------


def test_partial_cal_pd_coverage():
    df = _minimal_features_df(n=10)
    # Only 5 of 10 agents have cal_pd
    df["cal_pd"] = np.nan
    df.loc[:4, "cal_pd"] = 0.15

    result = run_extrafloat_limit_engine(df, keep_intermediate=True)
    assert len(result) == 10
    assert result["assigned_limit"].notna().all(), "All agents must receive a limit"
    # Agents with cal_pd should have risk_score ≈ 0.85
    np.testing.assert_allclose(result.loc[:4, "risk_score"].values, 0.85, atol=1e-9)


# -----------------------------------------------------------------------------
# 6. run_credit_risk_pipeline integration smoke test (mocked PD model)
# -----------------------------------------------------------------------------


def _make_pd_scored(msisdn_list):
    n = len(msisdn_list)
    return pd.DataFrame(
        {
            "agent_msisdn": msisdn_list,
            "cal_pd": np.linspace(0.05, 0.60, n),
            "thin_file_flag": [0] * n,
            "final_policy_bucket": ["APPROVE_50"] * n,
        }
    )


def test_pipeline_produces_required_output_columns(tmp_path):
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    n = 8
    msisdn_list = [f"256700{i:06d}" for i in range(n)]

    df_features = _minimal_features_df(n=n)
    pd_scored = _make_pd_scored(msisdn_list)

    df_raw = pd.DataFrame(
        {
            "agent_msisdn": msisdn_list,
            "commission": np.random.default_rng(0).uniform(1e4, 1e5, n),
            "account_balance": np.random.default_rng(1).uniform(1e5, 5e6, n),
        }
    )

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch("run_credit_risk_pipeline.pd.read_csv", return_value=df_raw),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features") as mock_txn,
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features") as mock_loan,
        patch("run_credit_risk_pipeline.load_borrower_limit_features") as mock_bor,
        patch("run_credit_risk_pipeline.run_inference_pipeline", return_value=pd_scored),
        patch("run_credit_risk_pipeline.build_extrafloat_limit_engine_features", return_value=df_features),
    ):
        mock_txn.return_value = MagicMock()
        mock_loan.return_value = MagicMock()
        mock_bor.return_value = MagicMock()

        result = run_credit_risk_pipeline(
            transaction_file="dummy_txn.csv",
            loan_file="dummy_loan.csv",
            borrower_file="dummy_bor.csv",
            artifacts_dir=str(tmp_path),
        )

    required = {"assigned_limit", "risk_tier", "cal_pd", "final_decision_reason"}
    missing = required - set(result.columns)
    assert not missing, f"Output missing required columns: {missing}"
    assert len(result) == n


def test_inference_receives_raw_data_with_agent_msisdn(tmp_path):
    """run_inference_pipeline must receive the raw CSV (agent_msisdn key),
    not the engine-loader output where the key has been renamed to msisdn."""
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    n = 4
    msisdn_list = [f"256700{i:06d}" for i in range(n)]
    df_features = _minimal_features_df(n=n)
    pd_scored = _make_pd_scored(msisdn_list)

    # Raw read returns agent_msisdn; loader returns msisdn (engine rename)
    raw_df = pd.DataFrame({"agent_msisdn": msisdn_list, "commission": [1.0] * n})
    engine_df = pd.DataFrame({"msisdn": msisdn_list, "commission": [1.0] * n})

    captured = {}

    def capture_inference(df_raw, **kwargs):
        captured["df_raw"] = df_raw
        return pd_scored

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch("run_credit_risk_pipeline.pd.read_csv", return_value=raw_df),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features", return_value=engine_df),
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_borrower_limit_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.run_inference_pipeline", side_effect=capture_inference),
        patch("run_credit_risk_pipeline.build_extrafloat_limit_engine_features", return_value=df_features),
    ):
        run_credit_risk_pipeline(
            transaction_file="dummy_txn.csv",
            loan_file="dummy_loan.csv",
            borrower_file="dummy_bor.csv",
            artifacts_dir=str(tmp_path),
        )

    assert "agent_msisdn" in captured["df_raw"].columns, (
        "run_inference_pipeline must receive the raw DataFrame with agent_msisdn, "
        "not the engine-loader version where the key is renamed to msisdn"
    )
    assert "msisdn" not in captured["df_raw"].columns


def test_loan_history_file_reaches_stage3_inference(tmp_path):
    """--loan-history-file must reach both Stage 3 (PD inference, as
    loan_history_df) and Stage 4 (engine features, as loan_history_snapshot_df)
    -- the same loaded DataFrame, two consumers."""
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    n = 4
    msisdn_list = [f"256700{i:06d}" for i in range(n)]
    df_features = _minimal_features_df(n=n)
    pd_scored = _make_pd_scored(msisdn_list)
    raw_df = pd.DataFrame({"agent_msisdn": msisdn_list, "commission": [1.0] * n})
    loan_history_df = pd.DataFrame(
        {"msisdn": msisdn_list, "observed_loan_count": [1] * n}
    )

    captured_inference = {}

    def capture_inference(df_raw, **kwargs):
        captured_inference.update(kwargs)
        return pd_scored

    captured_engine_kwargs = {}

    def capture_engine_features(*args, **kwargs):
        captured_engine_kwargs.update(kwargs)
        return df_features

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch("run_credit_risk_pipeline.pd.read_csv", return_value=raw_df),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_borrower_limit_features", return_value=MagicMock()),
        patch(
            "run_credit_risk_pipeline.load_loan_history_snapshot_features",
            return_value=loan_history_df,
        ),
        patch("run_credit_risk_pipeline.run_inference_pipeline", side_effect=capture_inference),
        patch(
            "run_credit_risk_pipeline.build_extrafloat_limit_engine_features",
            side_effect=capture_engine_features,
        ),
    ):
        run_credit_risk_pipeline(
            transaction_file="dummy_txn.csv",
            loan_file="dummy_loan.csv",
            borrower_file="dummy_bor.csv",
            artifacts_dir=str(tmp_path),
            loan_history_file="dummy_loan_history.csv",
        )

    assert captured_inference.get("loan_history_df") is loan_history_df
    assert captured_engine_kwargs.get("loan_history_snapshot_df") is loan_history_df


def test_loan_history_file_absent_is_none_in_both_stages(tmp_path):
    """Omitting --loan-history-file must leave both consumers untouched
    (clean no-op, as documented)."""
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    n = 4
    msisdn_list = [f"256700{i:06d}" for i in range(n)]
    df_features = _minimal_features_df(n=n)
    pd_scored = _make_pd_scored(msisdn_list)
    raw_df = pd.DataFrame({"agent_msisdn": msisdn_list, "commission": [1.0] * n})

    captured_inference = {}

    def capture_inference(df_raw, **kwargs):
        captured_inference.update(kwargs)
        return pd_scored

    captured_engine_kwargs = {}

    def capture_engine_features(*args, **kwargs):
        captured_engine_kwargs.update(kwargs)
        return df_features

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch("run_credit_risk_pipeline.pd.read_csv", return_value=raw_df),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_borrower_limit_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.run_inference_pipeline", side_effect=capture_inference),
        patch(
            "run_credit_risk_pipeline.build_extrafloat_limit_engine_features",
            side_effect=capture_engine_features,
        ),
    ):
        run_credit_risk_pipeline(
            transaction_file="dummy_txn.csv",
            loan_file="dummy_loan.csv",
            borrower_file="dummy_bor.csv",
            artifacts_dir=str(tmp_path),
        )

    assert captured_inference.get("loan_history_df") is None
    assert captured_engine_kwargs.get("loan_history_snapshot_df") is None


# -----------------------------------------------------------------------------
# 7. Experience factor NOT applied on the cal_pd path (fix 3)
# -----------------------------------------------------------------------------


def test_experience_factor_not_applied_on_cal_pd_path():
    """
    A thin-file agent (total_loans=1) and an experienced agent (total_loans=50)
    with the same cal_pd must get the same risk_cap on the cal_pd path.

    If the experience_factor were still applied, the thin-file agent would get
    a lower risk_cap (factor = 1/10 = 0.10 vs 1.00 for experienced agent),
    double-penalising what the PD model already handled via never_loan_pd_like.
    """
    cal_pd = 0.20  # risk_score = 0.80 for both agents

    df_thin = _minimal_features_df(n=1)
    df_thin["cal_pd"] = cal_pd
    df_thin["total_loans"] = 1  # thin-file: very few loans

    df_experienced = _minimal_features_df(n=1)
    df_experienced["cal_pd"] = cal_pd
    df_experienced["total_loans"] = 50  # experienced borrower

    result_thin = compute_risk_cap(df_thin)
    result_experienced = compute_risk_cap(df_experienced)

    np.testing.assert_allclose(
        result_thin["risk_cap"].values,
        result_experienced["risk_cap"].values,
        atol=1e-6,
        err_msg=(
            "risk_cap must be identical for thin-file and experienced agents "
            "when cal_pd is the same -- experience_factor must not be applied "
            "on the cal_pd path"
        ),
    )


# -----------------------------------------------------------------------------
# 7b. Unresolved-loan-at-snapshot haircut (data/loan_history_snapshot_query.txt)
# -----------------------------------------------------------------------------


def test_unresolved_loan_haircut_reduces_risk_cap_on_fallback_path():
    df_baseline = _minimal_features_df(n=1, seed=7)
    df_unresolved = _minimal_features_df(n=1, seed=7)
    df_unresolved["has_unresolved_loan_at_snapshot"] = 1
    df_unresolved["active_loan_days_aging_at_snapshot"] = 10

    result_baseline = compute_risk_cap(df_baseline)
    result_unresolved = compute_risk_cap(df_unresolved)

    assert (result_unresolved["risk_cap"].values < result_baseline["risk_cap"].values).all()
    assert result_unresolved["risk_unresolved_loan_haircut_reason"].iloc[0] == "unresolved_loan_haircut"
    assert result_unresolved["risk_unresolved_loan_multiplier"].iloc[0] < 1.0


def test_unresolved_loan_columns_do_not_affect_cal_pd_path():
    """
    Mirrors test_experience_factor_not_applied_on_cal_pd_path: an agent
    flagged has_unresolved_loan_at_snapshot=1 and an otherwise-identical
    agent with no unresolved loan must get the same risk_cap when both
    have the same cal_pd -- the PD model's own training query already
    computes has_unresolved_loan_at_scoring / anomaly_open_at_scoring as
    predictors, so a second haircut here would double-count.
    """
    cal_pd = 0.20

    df_clean = _minimal_features_df(n=1)
    df_clean["cal_pd"] = cal_pd

    df_unresolved = _minimal_features_df(n=1)
    df_unresolved["cal_pd"] = cal_pd
    df_unresolved["has_unresolved_loan_at_snapshot"] = 1
    df_unresolved["active_loan_days_aging_at_snapshot"] = 40
    df_unresolved["anomaly_open_at_snapshot"] = 1

    result_clean = compute_risk_cap(df_clean)
    result_unresolved = compute_risk_cap(df_unresolved)

    np.testing.assert_allclose(
        result_clean["risk_cap"].values,
        result_unresolved["risk_cap"].values,
        atol=1e-6,
        err_msg="risk_cap must be identical on the cal_pd path regardless of unresolved-loan columns",
    )
    assert result_unresolved["risk_unresolved_loan_multiplier"].iloc[0] == 1.0
    assert result_unresolved["risk_unresolved_loan_haircut_reason"].iloc[0] == "cal_pd_path_not_applicable"


def test_unresolved_loan_columns_absent_is_no_op():
    """When the loan history snapshot input was never supplied, the
    haircut columns are absent from features_df -- _safe_series defaults
    them to 0.0, so risk_cap must be byte-identical to a run before this
    feature existed."""
    df = _minimal_features_df(n=5, seed=3)
    assert "has_unresolved_loan_at_snapshot" not in df.columns

    result = compute_risk_cap(df)

    assert (result["risk_unresolved_loan_multiplier"] == 1.0).all()
    assert (result["risk_unresolved_loan_haircut_reason"] == "none").all()


def test_anomaly_open_haircut_harsher_than_unresolved_alone():
    df_baseline = _minimal_features_df(n=1, seed=11)

    df_unresolved_only = _minimal_features_df(n=1, seed=11)
    df_unresolved_only["has_unresolved_loan_at_snapshot"] = 1
    df_unresolved_only["active_loan_days_aging_at_snapshot"] = 0

    df_anomaly = _minimal_features_df(n=1, seed=11)
    df_anomaly["has_unresolved_loan_at_snapshot"] = 1
    df_anomaly["anomaly_open_at_snapshot"] = 1

    cap_baseline = compute_risk_cap(df_baseline)["risk_cap"].iloc[0]
    cap_unresolved = compute_risk_cap(df_unresolved_only)["risk_cap"].iloc[0]
    cap_anomaly = compute_risk_cap(df_anomaly)["risk_cap"].iloc[0]

    assert cap_anomaly < cap_unresolved < cap_baseline


def test_unresolved_loan_aging_haircut_monotonic_and_capped():
    cfg = _get_config(None)
    caps = []
    for aging_days in (0, 10, 50, 500):
        df = _minimal_features_df(n=1, seed=5)
        df["has_unresolved_loan_at_snapshot"] = 1
        df["active_loan_days_aging_at_snapshot"] = aging_days
        result = compute_risk_cap(df)
        caps.append(result["risk_cap"].iloc[0])
        assert result["risk_cap"].iloc[0] >= cfg["global_floor_limit"]

    # Monotonically non-increasing as aging grows.
    for earlier, later in zip(caps, caps[1:]):
        assert later <= earlier
    # The per-day haircut is capped, so 50 and 500 days must produce the
    # same risk_cap (both saturate unresolved_loan_aging_haircut_cap).
    assert caps[2] == pytest.approx(caps[3], abs=1e-6)


def test_unresolved_loan_haircut_never_pushes_risk_cap_negative():
    cfg = deepcopy(_get_config(None))
    cfg["risk"]["unresolved_loan_base_haircut"] = 5.0
    cfg["risk"]["anomaly_open_haircut"] = 5.0
    cfg["risk"]["unresolved_loan_aging_haircut_per_day"] = 5.0
    cfg["risk"]["unresolved_loan_aging_haircut_cap"] = 5.0

    df = _minimal_features_df(n=3, seed=9)
    df["has_unresolved_loan_at_snapshot"] = 1
    df["active_loan_days_aging_at_snapshot"] = 100
    df["anomaly_open_at_snapshot"] = 1

    result = compute_risk_cap(df, config=cfg)
    assert (result["risk_cap"] >= cfg["global_floor_limit"]).all()
    assert (result["risk_unresolved_loan_multiplier"] >= 0.0).all()


# -----------------------------------------------------------------------------
# 8. Preflight artifact check (fix 2)
# -----------------------------------------------------------------------------


def test_preflight_raises_on_missing_artifacts(tmp_path):
    from run_credit_risk_pipeline import _REQUIRED_ARTIFACTS, _check_artifacts

    # Empty artifacts dir -- all files missing
    with pytest.raises(FileNotFoundError) as exc_info:
        _check_artifacts(tmp_path)

    msg = str(exc_info.value)
    # Error must name at least one missing file and show the training command
    assert any(f in msg for f in _REQUIRED_ARTIFACTS)
    assert "pd_model.run_pipeline" in msg
    assert "--output-dir" in msg


def test_preflight_passes_when_all_artifacts_present(tmp_path):
    import json as _json

    from run_credit_risk_pipeline import _REQUIRED_ARTIFACTS, _check_artifacts

    for fname in _REQUIRED_ARTIFACTS:
        if fname == "model_metadata.json":
            # Empty checksums -- use allow_unverified=True (dev/test mode)
            (tmp_path / fname).write_text(_json.dumps({}))
        else:
            (tmp_path / fname).touch()

    # allow_unverified=True: placeholder metadata warns but does not raise
    _check_artifacts(tmp_path, allow_unverified=True)


# -----------------------------------------------------------------------------
# 8. pd_decile -- population-relative risk rank from cal_pd
# -----------------------------------------------------------------------------


def test_pd_decile_computed_when_cal_pd_present():
    """pd_decile (1-10) is present and correctly ordered when cal_pd is available."""
    from extrafloat.engine.extrafloat_limit_engine_caps import apply_policy_adjustments

    n = 50
    rng = np.random.default_rng(42)
    df = _minimal_features_df(n=n, seed=42)
    df["cal_pd"] = rng.uniform(0.01, 0.95, n)
    # Build a risk_score column so apply_policy_adjustments can find it
    df["risk_score"] = 1.0 - df["cal_pd"]
    df["combined_cap"] = rng.uniform(100_000, 2_000_000, n)
    df["agent_profile"] = "Gold"

    result = apply_policy_adjustments(df)

    assert "pd_decile" in result.columns, "pd_decile must be in output"
    assert result["pd_decile"].notna().all(), "all rows should have pd_decile when cal_pd is fully present"

    d1_median = result.loc[result["pd_decile"] == 1, "cal_pd"].median()
    d10_median = result.loc[result["pd_decile"] == 10, "cal_pd"].median()
    assert d1_median < d10_median, "decile 1 (lowest risk) must have lower cal_pd than decile 10"


def test_pd_decile_nan_when_cal_pd_absent():
    """pd_decile is NA for all agents when cal_pd is not present (7-signal fallback)."""
    from extrafloat.engine.extrafloat_limit_engine_caps import apply_policy_adjustments

    n = 20
    rng = np.random.default_rng(7)
    df = _minimal_features_df(n=n, seed=7)
    # Provide risk_score directly (no cal_pd -- simulates 7-signal path)
    df["risk_score"] = rng.uniform(0.1, 0.9, n)
    df["combined_cap"] = rng.uniform(100_000, 2_000_000, n)
    df["agent_profile"] = "Gold"
    # Ensure cal_pd is absent
    assert "cal_pd" not in df.columns

    result = apply_policy_adjustments(df)

    assert "pd_decile" in result.columns, "pd_decile column must still be present"
    assert result["pd_decile"].isna().all(), "pd_decile must be NA on the 7-signal path"


def test_pd_decile_in_final_output_columns():
    """pd_decile survives keep_intermediate=False trimming."""
    assert "pd_decile" in FINAL_OUTPUT_COLUMNS


# -----------------------------------------------------------------------------
# 9. F5 -- Calibration fail-closed
# -----------------------------------------------------------------------------


def test_calibration_exception_propagates():
    """After F5 fix: calibration failure must raise, not be swallowed."""
    from pd_model.modeling.inference import ModelArtifacts, score_new_agents

    n = 6
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.uniform(0, 1, (n, len(feature_cols))), columns=feature_cols)
    df["agent_msisdn"] = [f"256{i:09d}" for i in range(n)]
    df["thin_file_flag"] = 0  # all thick-file -> triggers calibration path

    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])
    mock_lgb = MagicMock()
    mock_lgb.predict_proba.return_value = np.column_stack([np.full(n, 0.7), np.full(n, 0.3)])

    artifacts = ModelArtifacts(
        xgb_model=mock_xgb,
        lgb_model=mock_lgb,
        feature_order=feature_cols,
        cal_map=pd.DataFrame(),
        xgb_policy_thresholds=pd.DataFrame(),
        lgb_policy_thresholds=pd.DataFrame(),
        transform_report=pd.DataFrame({"feature": [], "action": []}),
    )

    with patch(
        "pd_model.modeling.inference.attach_cal_pd", side_effect=RuntimeError("Calibration map corrupt")
    ):
        with pytest.raises(RuntimeError, match="Calibration map corrupt"):
            score_new_agents(df, artifacts)


def test_score_source_pd_model_when_cal_pd_present(tmp_path):
    """score_source == 'pd_model' for every agent that has cal_pd."""
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    n = 5
    msisdn_list = [f"256700{i:06d}" for i in range(n)]
    df_features = _minimal_features_df(n=n)
    pd_scored = _make_pd_scored(msisdn_list)  # all agents have cal_pd

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch(
            "run_credit_risk_pipeline.pd.read_csv", return_value=pd.DataFrame({"agent_msisdn": msisdn_list})
        ),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_borrower_limit_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.run_inference_pipeline", return_value=pd_scored),
        patch("run_credit_risk_pipeline.build_extrafloat_limit_engine_features", return_value=df_features),
    ):
        result = run_credit_risk_pipeline(
            transaction_file="dummy.csv",
            loan_file="dummy.csv",
            borrower_file="dummy.csv",
            artifacts_dir=str(tmp_path),
        )

    assert "score_source" in result.columns
    assert "scored_at" in result.columns
    assert (result["score_source"] == "pd_model").all(), (
        "All agents with cal_pd must have score_source='pd_model'"
    )


def test_score_source_fallback_when_cal_pd_absent(tmp_path):
    """score_source == '7_signal_fallback' for agents without cal_pd."""
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    n = 6
    msisdn_list = [f"256700{i:06d}" for i in range(n)]
    df_features = _minimal_features_df(n=n)

    # PD model returns NaN cal_pd for all agents (e.g. not in PD output)
    pd_scored_no_pd = pd.DataFrame(
        {
            "agent_msisdn": msisdn_list,
            "cal_pd": [float("nan")] * n,
            "thin_file_flag": [0] * n,
        }
    )

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch(
            "run_credit_risk_pipeline.pd.read_csv", return_value=pd.DataFrame({"agent_msisdn": msisdn_list})
        ),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_borrower_limit_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.run_inference_pipeline", return_value=pd_scored_no_pd),
        patch("run_credit_risk_pipeline.build_extrafloat_limit_engine_features", return_value=df_features),
    ):
        result = run_credit_risk_pipeline(
            transaction_file="dummy.csv",
            loan_file="dummy.csv",
            borrower_file="dummy.csv",
            artifacts_dir=str(tmp_path),
        )

    assert (result["score_source"] == "7_signal_fallback").all(), (
        "Agents with no cal_pd must have score_source='7_signal_fallback'"
    )


# -----------------------------------------------------------------------------
# 10. F6 -- Join integrity
# -----------------------------------------------------------------------------


def test_join_raises_on_duplicate_engine_msisdn(tmp_path):
    """Duplicate msisdn in engine features must raise DataAlignmentError before the join."""
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    n = 4
    msisdn_list = [f"256700{i:06d}" for i in range(n)]
    pd_scored = _make_pd_scored(msisdn_list)

    # Engine features with a duplicate msisdn
    df_features_dup = _minimal_features_df(n=n)
    df_features_dup.loc[1, "msisdn"] = df_features_dup.loc[0, "msisdn"]  # duplicate

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch(
            "run_credit_risk_pipeline.pd.read_csv", return_value=pd.DataFrame({"agent_msisdn": msisdn_list})
        ),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_borrower_limit_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.run_inference_pipeline", return_value=pd_scored),
        patch(
            "run_credit_risk_pipeline.build_extrafloat_limit_engine_features", return_value=df_features_dup
        ),
    ):
        with pytest.raises(DataAlignmentError, match="Duplicate msisdn in engine features"):
            run_credit_risk_pipeline(
                transaction_file="dummy.csv",
                loan_file="dummy.csv",
                borrower_file="dummy.csv",
                artifacts_dir=str(tmp_path),
            )


def test_join_raises_on_duplicate_pd_msisdn(tmp_path):
    """Duplicate agent_msisdn in PD output must raise DataAlignmentError before the join."""
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    n = 4
    msisdn_list = [f"256700{i:06d}" for i in range(n)]
    df_features = _minimal_features_df(n=n)

    # PD output with a duplicate agent_msisdn
    pd_scored_dup = _make_pd_scored(msisdn_list)
    pd_scored_dup = pd.concat([pd_scored_dup, pd_scored_dup.iloc[[0]]], ignore_index=True)

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch(
            "run_credit_risk_pipeline.pd.read_csv", return_value=pd.DataFrame({"agent_msisdn": msisdn_list})
        ),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_borrower_limit_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.run_inference_pipeline", return_value=pd_scored_dup),
        patch("run_credit_risk_pipeline.build_extrafloat_limit_engine_features", return_value=df_features),
    ):
        with pytest.raises(DataAlignmentError, match="Duplicate agent_msisdn in PD output"):
            run_credit_risk_pipeline(
                transaction_file="dummy.csv",
                loan_file="dummy.csv",
                borrower_file="dummy.csv",
                artifacts_dir=str(tmp_path),
            )


@pytest.mark.parametrize("null_value", [None, np.nan, "None", "none", "nan"])
def test_join_raises_on_null_engine_msisdn(tmp_path, null_value):
    """Null msisdn in engine features (None, np.nan, or sentinel strings) raises DataAlignmentError."""
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    n = 4
    msisdn_list = [f"256700{i:06d}" for i in range(n)]
    pd_scored = _make_pd_scored(msisdn_list)

    df_features_null = _minimal_features_df(n=n)
    df_features_null.loc[2, "msisdn"] = null_value

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch(
            "run_credit_risk_pipeline.pd.read_csv", return_value=pd.DataFrame({"agent_msisdn": msisdn_list})
        ),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_borrower_limit_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.run_inference_pipeline", return_value=pd_scored),
        patch(
            "run_credit_risk_pipeline.build_extrafloat_limit_engine_features", return_value=df_features_null
        ),
    ):
        with pytest.raises(DataAlignmentError, match="null msisdn"):
            run_credit_risk_pipeline(
                transaction_file="dummy.csv",
                loan_file="dummy.csv",
                borrower_file="dummy.csv",
                artifacts_dir=str(tmp_path),
            )


def test_msisdn_dot_zero_normalized(tmp_path):
    """'256700123.0' in the engine must match '256700123' in PD output after normalisation."""
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    n = 3
    canonical = [f"256700{i:06d}" for i in range(n)]

    # PD output uses the canonical form
    pd_scored = _make_pd_scored(canonical)

    # Engine features use the .0-suffix form (common when read from float CSV column)
    df_features_dot0 = _minimal_features_df(n=n)
    df_features_dot0["msisdn"] = [f + ".0" for f in canonical]

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch("run_credit_risk_pipeline.pd.read_csv", return_value=pd.DataFrame({"agent_msisdn": canonical})),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_borrower_limit_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.run_inference_pipeline", return_value=pd_scored),
        patch(
            "run_credit_risk_pipeline.build_extrafloat_limit_engine_features", return_value=df_features_dot0
        ),
    ):
        result = run_credit_risk_pipeline(
            transaction_file="dummy.csv",
            loan_file="dummy.csv",
            borrower_file="dummy.csv",
            artifacts_dir=str(tmp_path),
        )

    # All agents must have cal_pd -- normalisation resolved the mismatch
    assert result["cal_pd"].notna().all(), (
        "MSISDN .0-suffix normalisation failed: some agents have no cal_pd "
        "despite matching canonical MSISDNs in PD output"
    )


def test_is_thin_file_preserved_for_agents_absent_from_pd_output(tmp_path):
    """Regression test: an agent present in the engine features (borrower-file
    population) but absent from the PD model output (transaction-file population)
    must keep the engine's own is_thin_file -- it must not be silently forced to 0
    by the thin-file reconciliation step.

    build_extrafloat_limit_engine_features roots its population in the borrower
    file, while the PD model only scores agents present in the transaction file
    (df_agent). A borrower-only agent therefore has thin_file_flag == NaN after
    the left join in Stage 5; fillna(0) on the whole column would previously wipe
    out a genuinely thin-file agent's protective status.
    """
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    scored = [f"256700{i:06d}" for i in range(4)]  # agents the PD model scored
    unscored = "256799999999"  # borrower-only agent, absent from PD output

    pd_scored = _make_pd_scored(scored)  # thin_file_flag = 0 for all scored agents

    df_features = _minimal_features_df(n=5)
    df_features["msisdn"] = scored + [unscored]
    # Engine's own computation (e.g. total_loans < 3): the unscored agent is
    # genuinely thin-file; the scored agents are not.
    df_features["is_thin_file"] = [0, 0, 0, 0, 1]

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch(
            "run_credit_risk_pipeline.pd.read_csv",
            return_value=pd.DataFrame({"agent_msisdn": scored}),
        ),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.load_borrower_limit_features", return_value=MagicMock()),
        patch("run_credit_risk_pipeline.run_inference_pipeline", return_value=pd_scored),
        patch("run_credit_risk_pipeline.build_extrafloat_limit_engine_features", return_value=df_features),
    ):
        result = run_credit_risk_pipeline(
            transaction_file="dummy.csv",
            loan_file="dummy.csv",
            borrower_file="dummy.csv",
            artifacts_dir=str(tmp_path),
            keep_intermediate=True,
        )

    unscored_row = result.loc[result["msisdn"] == unscored].iloc[0]
    assert pd.isna(unscored_row["cal_pd"]), "unscored agent should have no cal_pd"
    assert unscored_row["is_thin_file"] == 1, (
        "engine-computed is_thin_file must survive for agents absent from PD output -- "
        "it must not be forced to 0 by the thin-file reconciliation step"
    )

    scored_rows = result.loc[result["msisdn"] != unscored]
    assert (scored_rows["is_thin_file"] == 0).all(), (
        "PD-scored agents should still be reconciled to the PD model's thin_file_flag"
    )


# -----------------------------------------------------------------------------
# Checksum integrity tests (NF2)
# -----------------------------------------------------------------------------


def test_checksum_passes_with_valid_artifacts(tmp_path):
    """_check_artifacts does not raise when all stored checksums match."""
    import hashlib
    import json as _json

    from run_credit_risk_pipeline import _REQUIRED_ARTIFACTS, _check_artifacts

    artifact_hashes = {}
    for fname in _REQUIRED_ARTIFACTS:
        if fname == "model_metadata.json":
            continue
        fpath = tmp_path / fname
        fpath.write_bytes(b"dummy content for " + fname.encode())
        artifact_hashes[fname] = hashlib.sha256(fpath.read_bytes()).hexdigest()

    meta = {"artifact_sha256": artifact_hashes}
    (tmp_path / "model_metadata.json").write_text(_json.dumps(meta))

    _check_artifacts(tmp_path)  # must not raise


def test_checksum_mismatch_raises_on_preflight(tmp_path):
    """_check_artifacts raises RuntimeError when a stored checksum does not match."""
    import hashlib
    import json as _json

    from run_credit_risk_pipeline import _REQUIRED_ARTIFACTS, _check_artifacts

    artifact_hashes = {}
    for fname in _REQUIRED_ARTIFACTS:
        if fname == "model_metadata.json":
            continue
        fpath = tmp_path / fname
        fpath.write_bytes(b"original content")
        artifact_hashes[fname] = hashlib.sha256(fpath.read_bytes()).hexdigest()

    # Corrupt one artifact after hashing
    (tmp_path / "xgb_model.joblib").write_bytes(b"corrupted content")

    meta = {"artifact_sha256": artifact_hashes}
    (tmp_path / "model_metadata.json").write_text(_json.dumps(meta))

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _check_artifacts(tmp_path)


def test_preflight_warns_when_no_checksums_stored(tmp_path, caplog):
    """With allow_unverified=True, absent checksums produce a warning but not an error."""
    import json as _json
    import logging

    from run_credit_risk_pipeline import _REQUIRED_ARTIFACTS, _check_artifacts

    for fname in _REQUIRED_ARTIFACTS:
        if fname == "model_metadata.json":
            (tmp_path / fname).write_text(_json.dumps({}))
        else:
            (tmp_path / fname).touch()

    with caplog.at_level(logging.WARNING, logger="credit_risk_pipeline"):
        _check_artifacts(tmp_path, allow_unverified=True)  # must not raise

    assert any("no artifact_sha256 checksums" in r.message for r in caplog.records), (
        "Expected a warning about missing checksums"
    )


def test_preflight_raises_when_no_checksums_and_not_allow_unverified(tmp_path):
    """Without allow_unverified, absent checksums raise RuntimeError (production default)."""
    import json as _json

    from run_credit_risk_pipeline import _REQUIRED_ARTIFACTS, _check_artifacts

    for fname in _REQUIRED_ARTIFACTS:
        if fname == "model_metadata.json":
            (tmp_path / fname).write_text(_json.dumps({}))
        else:
            (tmp_path / fname).touch()

    with pytest.raises(RuntimeError, match="no artifact_sha256 checksums"):
        _check_artifacts(tmp_path, allow_unverified=False)


def test_preflight_raises_when_artifact_missing_from_manifest(tmp_path):
    """Without allow_unverified, an artifact absent from the hash map raises RuntimeError."""
    import hashlib
    import json as _json

    from run_credit_risk_pipeline import _REQUIRED_ARTIFACTS, _check_artifacts

    artifact_hashes = {}
    for fname in _REQUIRED_ARTIFACTS:
        if fname == "model_metadata.json":
            continue
        fpath = tmp_path / fname
        fpath.write_bytes(b"content")
        artifact_hashes[fname] = hashlib.sha256(fpath.read_bytes()).hexdigest()

    # Remove one artifact from the manifest
    del artifact_hashes["xgb_model.joblib"]
    (tmp_path / "model_metadata.json").write_text(_json.dumps({"artifact_sha256": artifact_hashes}))

    with pytest.raises(RuntimeError, match="No stored checksum"):
        _check_artifacts(tmp_path, allow_unverified=False)

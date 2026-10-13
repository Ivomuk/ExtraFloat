"""
tests/pipeline/test_stage_4c_live_path_unchanged.py
=====================================================
The literal hard release gate for the Stage 4C Decision Log insertion
point (docs/stage_4c_logger_implementation_plan.md Section 4/5 item 10,
elevated from a verification nicety to a named hard release gate per the
plan's review-correction round): run_credit_risk_pipeline() must return
an IDENTICAL result at keep_intermediate=False whether the Stage 9
Decision Log block is enabled or disabled -- same columns, same column
order, same dtypes, same values, same row order.

Mirrors tests/engine/test_shadow_live_path_unchanged.py's exact pattern
for the analogous C3 shadow-multiplier insertion point.
"""

from __future__ import annotations

import datetime as _dt_real
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pandas.testing as pdt

from tests.pipeline.test_credit_risk_pipeline import _make_pd_scored, _minimal_features_df

# Frozen "now" so the two pipeline invocations under comparison produce the
# SAME run_id/scored_at -- real invocations a few milliseconds apart would
# otherwise differ there for a reason that has nothing to do with Stage 4C.
_FIXED_NOW = _dt_real.datetime(2026, 10, 13, 6, 0, 0)


class _FakeDtNamespace:
    """Rebinds run_credit_risk_pipeline's OWN `_dt` name (a local module-
    level reference to the real datetime module) to this fake namespace,
    rather than patching the real, globally-shared datetime.datetime
    class in-place -- an earlier version of this test did the latter and
    broke pandas internals process-wide (pandas does its own
    isinstance(val, (np.datetime64, dt.datetime)) checks against the
    real datetime module, which must stay untouched)."""

    class datetime:
        @staticmethod
        def utcnow():
            return _FIXED_NOW


class _FakeSink:
    """Minimal sink double -- confirms the Decision Log block actually
    runs (and writes) when enabled, without needing a real storage
    backend (Stage 4C's sink is deliberately unresolved -- see
    docs/stage_4c_logger_implementation_plan.md Section 3.4)."""

    def __init__(self):
        self.written_rows = None
        self.manifests = []

    def write_decision_log_rows(self, df):
        self.written_rows = df

    def write_manifest(self, manifest):
        self.manifests.append(manifest)


def _df_raw_with_capacity_inputs(msisdn_list, seed=7):
    """Raw agent snapshot carrying BOTH the columns the existing pipeline
    already reads (agent_msisdn, commission, account_balance) AND the
    Gate I0 capacity-input columns (cash_in_value_1m, payment_value_1m,
    snapshot_dt) the Stage 9 block needs -- confirms the live path is
    unaffected even when those extra columns are present on df_agent."""
    n = len(msisdn_list)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "agent_msisdn": msisdn_list,
            "commission": rng.uniform(1e4, 1e5, n),
            "account_balance": rng.uniform(1e5, 5e6, n),
            "cash_in_value_1m": rng.uniform(1e5, 2e6, n),
            "payment_value_1m": rng.uniform(1e4, 5e5, n),
            "snapshot_dt": pd.Timestamp("2026-01-31"),
        }
    )


def _run_pipeline(tmp_path, df_raw, df_features, pd_scored, stage_4c_logging_enabled, sink=None):
    from run_credit_risk_pipeline import run_credit_risk_pipeline

    with (
        patch("run_credit_risk_pipeline._check_artifacts"),
        patch("run_credit_risk_pipeline.pd.read_csv", return_value=df_raw),
        patch("run_credit_risk_pipeline.load_transaction_capacity_features") as mock_txn,
        patch("run_credit_risk_pipeline.load_loan_summary_recent_features") as mock_loan,
        patch("run_credit_risk_pipeline.load_borrower_limit_features") as mock_bor,
        patch("run_credit_risk_pipeline.run_inference_pipeline", return_value=pd_scored),
        patch("run_credit_risk_pipeline.build_extrafloat_limit_engine_features", return_value=df_features),
        patch("run_credit_risk_pipeline._dt", _FakeDtNamespace),
    ):
        mock_txn.return_value = MagicMock()
        mock_loan.return_value = MagicMock()
        mock_bor.return_value = MagicMock()

        return run_credit_risk_pipeline(
            transaction_file="dummy_txn.csv",
            loan_file="dummy_loan.csv",
            borrower_file="dummy_bor.csv",
            artifacts_dir=str(tmp_path),
            stage_4c_logging_enabled=stage_4c_logging_enabled,
            stage_4c_sink=sink,
            stage_4c_artifact_versions={
                "champion_model_version": "xgb_test_v1",
                "pd_calibration_artifact_version": "cal_test_v1",
                "c3_shadow_artifact_version": None,
                "c3_policy_version": None,
                "fundamentals_source_version": "mart_test_v1",
            },
        )


def test_stage_4c_disabled_matches_stage_4c_enabled_at_keep_intermediate_false(tmp_path):
    """The hard release gate: identical result_df (columns, column order,
    dtypes, values, row order) at keep_intermediate=False regardless of
    whether Stage 9 (Stage 4C Decision Log) is enabled or disabled."""
    n = 6
    msisdn_list = [f"256700{i:06d}" for i in range(n)]
    df_features = _minimal_features_df(n=n)
    pd_scored = _make_pd_scored(msisdn_list)
    df_raw = _df_raw_with_capacity_inputs(msisdn_list)

    result_off = _run_pipeline(tmp_path, df_raw, df_features, pd_scored, stage_4c_logging_enabled=False)

    sink = _FakeSink()
    result_on = _run_pipeline(tmp_path, df_raw, df_features, pd_scored, stage_4c_logging_enabled=True, sink=sink)

    # Confirms the block actually ran (not a vacuous "both sides skipped it" pass).
    assert sink.written_rows is not None, "Stage 4C block did not run / did not write when enabled"
    assert len(sink.manifests) == 1
    assert sink.manifests[0]["shadow_logging_status"] == "SUCCESS"
    assert sink.manifests[0]["logged_agent_count"] == n

    pdt.assert_frame_equal(result_off, result_on, check_exact=True)


def test_stage_4c_disabled_output_columns_unchanged_from_pre_stage_4c_baseline(tmp_path):
    """Explicit column-level check (not just equality between the two new
    runs): with Stage 4C disabled, (a) no engine-intermediate column the
    Section 4 internal keep_intermediate=True + restore-afterward
    mechanism touches leaks into the output, and (b) the columns this
    pipeline has always unconditionally produced are still present.
    (Columns that depend on OPTIONAL_BUT_EXPECTED_COLUMNS being present
    in the input fixture -- e.g. the unresolved-loan-at-snapshot audit
    trail, or segmentation's capacity_tier/agent_category -- are a
    property of this minimal fixture, not of the Stage 4C change, and
    are deliberately not asserted here; test 1 above, comparing the SAME
    fixture with the flag on vs off, is the precise regression check.)"""
    n = 5
    msisdn_list = [f"256700{i:06d}" for i in range(n)]
    df_features = _minimal_features_df(n=n)
    pd_scored = _make_pd_scored(msisdn_list)
    df_raw = _df_raw_with_capacity_inputs(msisdn_list)

    result = _run_pipeline(tmp_path, df_raw, df_features, pd_scored, stage_4c_logging_enabled=False)

    forbidden_intermediate = {
        "capacity_cap", "policy_multiplier", "combined_cap", "recent_usage_cap",
        "prior_exposure_cap", "risk_cap", "thin_file_flag",
    }
    leaked = forbidden_intermediate & set(result.columns)
    assert not leaked, f"Engine-intermediate columns leaked into keep_intermediate=False output: {leaked}"

    always_present = {"assigned_limit", "risk_tier", "cal_pd", "final_decision_reason", "score_source", "scored_at"}
    missing_required = always_present - set(result.columns)
    assert not missing_required, f"Pre-existing, unconditional output columns went missing: {missing_required}"

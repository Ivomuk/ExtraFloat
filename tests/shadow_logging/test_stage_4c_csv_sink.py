"""
tests/shadow_logging/test_stage_4c_csv_sink.py
================================================
Synthetic sink failure/retry tests for the Stage 4C-S1 CSV sink adapter
(extrafloat/shadow_logging/stage_4c_csv_sink.py) -- the step named
explicitly in the Stage 4C roadmap after the sink adapter itself:

    sink reconnaissance -> sink adapter -> synthetic sink failure/retry
    tests -> deployment-readiness review -> first live shadow run

Covers the sink in isolation (atomicity, idempotency, one-file-per-run,
input validation) and, in the final test class, the full retry-after-an-
uncertain-write scenario at the orchestration level
(run_decision_log_stage), not just the sink's own methods in isolation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from extrafloat.shadow_logging.stage_4c_csv_sink import (
    CsvDecisionLogSink,
    _sanitize_run_id_for_filename,
)
from extrafloat.shadow_logging.stage_4c_decision_logger import (
    MANIFEST_STATUS_FAILED,
    MANIFEST_STATUS_SUCCESS,
    run_decision_log_stage,
)

# -----------------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------------

WEIGHTS = {"float_activity_value_1m": 0.5, "commission": 0.5}
SCORING_TS = pd.Timestamp("2026-03-01")
SNAPSHOT_OK_DATE = pd.Timestamp("2026-01-31")

ARTIFACT_VERSIONS = {
    "champion_model_version": "xgb_v1",
    "pd_calibration_artifact_version": "cal_v1",
    "c3_shadow_artifact_version": "c3_v1",
    "c3_policy_version": "c3_policy_v1",
    "fundamentals_source_version": "mart_v2026_07",
}


def _df_agent(msisdn):
    return pd.DataFrame({
        "agent_msisdn": [msisdn],
        "cash_in_value_1m": [1_000_000.0],
        "payment_value_1m": [200_000.0],
        "commission": [50_000.0],
        "snapshot_dt": [SNAPSHOT_OK_DATE],
    })


def _result_df(msisdn):
    return pd.DataFrame({
        "msisdn": [msisdn],
        "cal_pd": [0.12],
        "thin_file_flag": [0],
        "raw_model_score": [-1.1],
        "assigned_limit": [150_000.0],
        "policy_multiplier": [0.85],
        "capacity_cap": [200_000.0],
        "final_decision_reason": ["finalized_from_combined_cap"],
        "shadow_multiplier_base": [0.9],
        "shadow_multiplier_conservative": [0.8],
    })


# -----------------------------------------------------------------------------
# Sink-level tests
# -----------------------------------------------------------------------------


def test_round_trip_integrity(tmp_path):
    sink = CsvDecisionLogSink(tmp_path)
    df = pd.DataFrame({
        "run_id": ["run_a"] * 3,
        "agent_msisdn": ["1", "2", "3"],
        "l_current": [100_000.0, 200_000.0, 0.0],
    })
    sink.write_decision_log_rows(df)

    written_path = tmp_path / "stage_4c_decision_log_run_a.csv"
    assert written_path.exists()
    roundtrip = pd.read_csv(written_path)
    assert len(roundtrip) == 3
    assert list(roundtrip["agent_msisdn"].astype(str)) == ["1", "2", "3"]
    # No stray temp file left behind by a clean write.
    assert list(tmp_path.glob("*.tmp*")) == []


def test_mkdir_parents_true_creates_nested_output_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    sink = CsvDecisionLogSink(nested)
    df = pd.DataFrame({"run_id": ["run_a"], "agent_msisdn": ["1"], "l_current": [1.0]})
    sink.write_decision_log_rows(df)
    assert nested.exists()
    assert (nested / "stage_4c_decision_log_run_a.csv").exists()


def test_one_file_per_run_id_never_clobbers_another_run(tmp_path):
    sink = CsvDecisionLogSink(tmp_path)
    df_run_a = pd.DataFrame({"run_id": ["run_a"], "agent_msisdn": ["1"], "l_current": [1.0]})
    df_run_b = pd.DataFrame({"run_id": ["run_b"], "agent_msisdn": ["2"], "l_current": [2.0]})

    sink.write_decision_log_rows(df_run_a)
    sink.write_decision_log_rows(df_run_b)

    assert pd.read_csv(tmp_path / "stage_4c_decision_log_run_a.csv")["agent_msisdn"].iloc[0] == 1
    assert pd.read_csv(tmp_path / "stage_4c_decision_log_run_b.csv")["agent_msisdn"].iloc[0] == 2


def test_idempotent_retry_does_not_raise_and_does_not_overwrite(tmp_path):
    """The exact idempotency property asked for: a retry keyed by run_id
    must be a safe no-op, never an error, and never silently corrupt the
    already-committed file with different data from the retry attempt."""
    sink = CsvDecisionLogSink(tmp_path)
    original = pd.DataFrame({"run_id": ["run_a"] * 2, "agent_msisdn": ["1", "2"], "l_current": [10.0, 20.0]})
    sink.write_decision_log_rows(original)

    different_retry_data = pd.DataFrame({
        "run_id": ["run_a"] * 5, "agent_msisdn": ["9", "8", "7", "6", "5"],
        "l_current": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    sink.write_decision_log_rows(different_retry_data)  # must not raise

    final = pd.read_csv(tmp_path / "stage_4c_decision_log_run_a.csv")
    assert len(final) == 2
    assert list(final["agent_msisdn"].astype(str)) == ["1", "2"]


def test_simulated_crash_leaves_stray_tmp_file_that_is_not_mistaken_for_complete(tmp_path):
    """Simulates a process killed between writing the temp file and the
    atomic rename: a stray .tmp file must never be mistaken for a
    completed write, and a subsequent (retried) write must still
    succeed cleanly despite the leftover garbage."""
    sink = CsvDecisionLogSink(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    stray_tmp = tmp_path / "stage_4c_decision_log_run_crash.csv.tmp99999"
    stray_tmp.write_text("garbage,from,a,dead,process\n1,2,3,4,5")

    assert not sink.has_decision_log("run_crash")

    retry_df = pd.DataFrame({"run_id": ["run_crash"], "agent_msisdn": ["1"], "l_current": [1.0]})
    sink.write_decision_log_rows(retry_df)

    assert sink.has_decision_log("run_crash")
    final = pd.read_csv(tmp_path / "stage_4c_decision_log_run_crash.csv")
    assert final["agent_msisdn"].iloc[0] == 1
    # The real retry succeeded; the stray garbage file is simply orphaned
    # (a real deployment's cleanup job sweeps *.tmp* files -- not this
    # sink's job to guess at a safe-deletion policy for someone else's
    # leftover temp file from an unrelated crash).


def test_refuses_dataframe_spanning_multiple_run_ids(tmp_path):
    sink = CsvDecisionLogSink(tmp_path)
    mixed = pd.DataFrame({"run_id": ["run_a", "run_b"], "agent_msisdn": ["1", "2"], "l_current": [1.0, 2.0]})
    with pytest.raises(ValueError):
        sink.write_decision_log_rows(mixed)


def test_refuses_empty_dataframe(tmp_path):
    sink = CsvDecisionLogSink(tmp_path)
    empty = pd.DataFrame({"run_id": [], "agent_msisdn": [], "l_current": []})
    with pytest.raises(ValueError):
        sink.write_decision_log_rows(empty)


def test_manifest_round_trip_and_overwrite_allowed(tmp_path):
    sink = CsvDecisionLogSink(tmp_path)
    manifest_v1 = {
        "run_id": "run_a", "shadow_logging_status": MANIFEST_STATUS_FAILED,
        "expected_agent_count": 3, "logged_agent_count": 0,
        "failure_stage": "ValueError", "error_code": "boom",
    }
    sink.write_manifest(manifest_v1)
    assert sink.read_manifest("run_a") == manifest_v1

    # A later attempt's outcome (SUCCESS) must be able to correct the
    # earlier FAILED manifest for the same run_id -- manifests are
    # metadata, not the historical data itself, and always reflect the
    # latest attempt.
    manifest_v2 = dict(manifest_v1, shadow_logging_status=MANIFEST_STATUS_SUCCESS, logged_agent_count=3,
                        failure_stage=None, error_code=None)
    sink.write_manifest(manifest_v2)
    assert sink.read_manifest("run_a") == manifest_v2
    assert list(tmp_path.glob("*.tmp*")) == []


def test_run_id_sanitization_keeps_filenames_safe_and_traceable():
    sanitized = _sanitize_run_id_for_filename("2026-10-13T06:04:53.255990Z")
    assert ":" not in sanitized
    assert sanitized == _sanitize_run_id_for_filename("2026-10-13T06:04:53.255990Z")


# -----------------------------------------------------------------------------
# Orchestration-level retry test: run_decision_log_stage + a sink that
# fails once, then succeeds on retry -- the actual "retry after an
# uncertain write" scenario, not just the sink's own methods in isolation.
# -----------------------------------------------------------------------------


class _FlakyOnceSink(CsvDecisionLogSink):
    """Wraps the real CsvDecisionLogSink but raises on the FIRST call to
    write_decision_log_rows, then behaves normally on every subsequent
    call -- simulating a transient failure (e.g. a network blip to a
    mounted share) that a retry should recover from cleanly."""

    def __init__(self, output_dir):
        super().__init__(output_dir)
        self._calls = 0

    def write_decision_log_rows(self, df):
        self._calls += 1
        if self._calls == 1:
            raise OSError("simulated transient write failure (e.g. network share blip)")
        super().write_decision_log_rows(df)


def test_retry_after_uncertain_write_recovers_via_run_decision_log_stage(tmp_path):
    """End-to-end: the first run_decision_log_stage call fails during the
    sink write (status=FAILED, logged_agent_count=0, no file written --
    the all-or-nothing atomicity contract from the implementation plan).
    A retry with the SAME run_id against the SAME (now-recovered) sink
    must then succeed cleanly: status=SUCCESS, the Decision Log file is
    actually written, and the earlier FAILED manifest is corrected to
    SUCCESS for that run_id."""
    sink = _FlakyOnceSink(tmp_path)
    df_agent = _df_agent("256700000001")
    result_df = _result_df("256700000001")

    first_manifest = run_decision_log_stage(
        df_agent_raw=df_agent, result_df=result_df, run_id="run_retry_001",
        scoring_run_timestamp=SCORING_TS, capacity_weights=WEIGHTS, capacity_weights_version="w_v1",
        artifact_versions=ARTIFACT_VERSIONS, sink=sink,
    )
    assert first_manifest["shadow_logging_status"] == MANIFEST_STATUS_FAILED
    assert first_manifest["logged_agent_count"] == 0
    assert not (tmp_path / "stage_4c_decision_log_run_retry_001.csv").exists()
    assert sink.read_manifest("run_retry_001")["shadow_logging_status"] == MANIFEST_STATUS_FAILED

    second_manifest = run_decision_log_stage(
        df_agent_raw=df_agent, result_df=result_df, run_id="run_retry_001",
        scoring_run_timestamp=SCORING_TS, capacity_weights=WEIGHTS, capacity_weights_version="w_v1",
        artifact_versions=ARTIFACT_VERSIONS, sink=sink,
    )
    assert second_manifest["shadow_logging_status"] == MANIFEST_STATUS_SUCCESS
    assert second_manifest["logged_agent_count"] == 1
    written_path = tmp_path / "stage_4c_decision_log_run_retry_001.csv"
    assert written_path.exists()
    assert pd.read_csv(written_path)["agent_msisdn"].iloc[0] == 256700000001
    # The manifest for this run_id now reflects the LATEST (successful) attempt.
    assert sink.read_manifest("run_retry_001")["shadow_logging_status"] == MANIFEST_STATUS_SUCCESS


def test_retry_of_an_already_succeeded_run_is_a_safe_no_op(tmp_path):
    """The simpler, more common retry case: an orchestrator retries a run
    that actually DID succeed (e.g. it only saw a late/lost acknowledgment).
    Must not raise, must not corrupt the existing file, and the manifest
    stays SUCCESS."""
    sink = CsvDecisionLogSink(tmp_path)
    df_agent = _df_agent("256700000002")
    result_df = _result_df("256700000002")

    first_manifest = run_decision_log_stage(
        df_agent_raw=df_agent, result_df=result_df, run_id="run_retry_002",
        scoring_run_timestamp=SCORING_TS, capacity_weights=WEIGHTS, capacity_weights_version="w_v1",
        artifact_versions=ARTIFACT_VERSIONS, sink=sink,
    )
    assert first_manifest["shadow_logging_status"] == MANIFEST_STATUS_SUCCESS

    second_manifest = run_decision_log_stage(
        df_agent_raw=df_agent, result_df=result_df, run_id="run_retry_002",
        scoring_run_timestamp=SCORING_TS, capacity_weights=WEIGHTS, capacity_weights_version="w_v1",
        artifact_versions=ARTIFACT_VERSIONS, sink=sink,
    )
    assert second_manifest["shadow_logging_status"] == MANIFEST_STATUS_SUCCESS
    written_path = tmp_path / "stage_4c_decision_log_run_retry_002.csv"
    assert pd.read_csv(written_path)["agent_msisdn"].iloc[0] == 256700000002

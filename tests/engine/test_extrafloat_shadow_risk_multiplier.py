"""
tests/engine/test_extrafloat_shadow_risk_multiplier.py
=========================================================
Unit tests for the shadow continuous risk multiplier module. Covers:
  1. Artifact-level failure statuses (missing dir/artifact/corrupt/integrity
     check), each logged and never raising.
  2. Row-level partial missingness (cal_pd/combined_cap/assigned_limit) --
     a few bad rows must not blank out the whole run.
  3. SHADOW_SCENARIOS / SHADOW_POLICY_VERSION literal assertions.
  4. Monotonicity/boundary tests for _policy_3_hybrid, both scenarios.
  5. Transition-clamp + explicit ceiling-invariant tests.
  6. The parameterized anti-drift test: _apply_shadow_finishing_steps(...)
     fed the SAME discrete tier multiplier apply_policy_adjustments() used,
     across every finishing-step branch, must reproduce its policy_cap
     exactly.
"""

import json
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from extrafloat.engine.extrafloat_limit_engine_caps import DEFAULT_CAP_CONFIG, apply_policy_adjustments
from extrafloat.engine.extrafloat_shadow_risk_multiplier import (
    ISOTONIC_METADATA_FILENAME,
    ISOTONIC_MODEL_FILENAME,
    ROW_STATUS_MISSING_ASSIGNED_LIMIT,
    ROW_STATUS_MISSING_CAL_PD,
    ROW_STATUS_MISSING_COMBINED_CAP,
    ROW_STATUS_OK,
    SHADOW_POLICY_VERSION,
    SHADOW_SCENARIOS,
    STATUS_LOAD_ERROR,
    STATUS_MISSING_ARTIFACT,
    STATUS_NO_ARTIFACTS_DIR,
    STATUS_OK,
    _apply_shadow_finishing_steps,
    _apply_shadow_rounding,
    _policy_3_hybrid,
    compute_shadow_risk_multiplier,
    load_isotonic_calibration,
)
from extrafloat.engine.run_extrafloat_limit_engine import finalize_limits

# -----------------------------------------------------------------------------
# Fixture helpers
# -----------------------------------------------------------------------------


def _write_artifact(tmp_path, cal_pd_plateau=0.25, r_plateau_override=None, version="test_v0"):
    """Fit a small real isotonic model and write a valid (or deliberately
    corrupted) artifact pair to tmp_path. Returns tmp_path."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit([0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0], [0.0, 0.03, 0.0754, 0.12, 0.18, 0.22, 0.25])
    r_plateau = float(iso.predict([cal_pd_plateau])[0]) if r_plateau_override is None else r_plateau_override
    joblib.dump(iso, tmp_path / ISOTONIC_MODEL_FILENAME)
    (tmp_path / ISOTONIC_METADATA_FILENAME).write_text(
        json.dumps({"version": version, "cal_pd_plateau": cal_pd_plateau, "r_plateau": r_plateau})
    )
    return tmp_path


def _shadow_row(**overrides):
    base = {
        "cal_pd": 0.3,
        "combined_cap": 100_000.0,
        "assigned_limit": 100_000.0,
        "policy_multiplier": 0.65,
        "agent_tier_ceiling_multiplier": 1.0,
        "total_loans": 5.0,
        "on_time_repayment_rate": 0.70,
        "lifetime_default_rate": 0.15,
        "recent_disbursement_amount_1m": 5000.0,
        "recent_repayment_amount_1m": 5000.0,
    }
    base.update(overrides)
    return base


def _policy_row(**overrides):
    """Row shape matching apply_policy_adjustments()'s expected inputs.

    Includes "combined_reason" (normally set upstream by combine_caps(),
    not called in these finishing-steps-only fixtures) because
    finalize_limits()'s _safe_series(df, "combined_reason", <string
    default>) hits a real quirk when the column is absent: _safe_series
    always builds a float64 Series for a missing column regardless of the
    default's type, so a string default on a missing column raises. Never
    triggered in the live pipeline (combine_caps always sets this column
    first), only in a fixture that calls apply_policy_adjustments() in
    isolation -- supplying the column sidesteps it rather than exercising it.
    """
    base = {
        "risk_score": 0.50,
        "combined_cap": 10_000.0,
        "risk_cap": 30_000.0,
        "total_loans": 5.0,
        "on_time_repayment_rate": 0.70,
        "lifetime_default_rate": 0.15,
        "recent_disbursement_amount_1m": 5000.0,
        "recent_repayment_amount_1m": 5000.0,
        "agent_tier_ceiling_multiplier": 1.0,
        "combined_reason": "test_combined_reason",
    }
    base.update(overrides)
    return base


# -----------------------------------------------------------------------------
# 1. Artifact-level failure statuses
# -----------------------------------------------------------------------------


def test_none_artifacts_dir_skips_shadow_safely(caplog):
    df = pd.DataFrame([_shadow_row()])
    with caplog.at_level("WARNING"):
        out = compute_shadow_risk_multiplier(df, artifacts_dir=None, run_id="run1")
    assert (out["shadow_status"] == STATUS_NO_ARTIFACTS_DIR).all()
    assert out["shadow_calibrated_risk"].isna().all()
    assert any("run1" in r.message and STATUS_NO_ARTIFACTS_DIR in r.message for r in caplog.records)


def test_missing_artifact_skips_shadow_safely(tmp_path, caplog):
    df = pd.DataFrame([_shadow_row()])
    with caplog.at_level("WARNING"):
        out = compute_shadow_risk_multiplier(df, artifacts_dir=tmp_path, run_id="run2")
    assert (out["shadow_status"] == STATUS_MISSING_ARTIFACT).all()
    assert any(str(tmp_path) in r.message for r in caplog.records)


def test_corrupt_artifact_skips_shadow_safely(tmp_path, caplog):
    (tmp_path / ISOTONIC_MODEL_FILENAME).write_bytes(b"not a joblib file")
    (tmp_path / ISOTONIC_METADATA_FILENAME).write_text("{}")
    df = pd.DataFrame([_shadow_row()])
    with caplog.at_level("WARNING"):
        out = compute_shadow_risk_multiplier(df, artifacts_dir=tmp_path, run_id="run3")
    assert (out["shadow_status"] == STATUS_LOAD_ERROR).all()


def test_artifact_integrity_check_failure(tmp_path, caplog):
    """Metadata's stored r_plateau deliberately doesn't match the model's
    own prediction at cal_pd_plateau -> STATUS_LOAD_ERROR, logged."""
    _write_artifact(tmp_path, r_plateau_override=0.999, version="bad_v0")
    with caplog.at_level("ERROR"):
        model, metadata, status = load_isotonic_calibration(tmp_path)
    assert model is None
    assert status == STATUS_LOAD_ERROR
    assert any("integrity check FAILED" in r.message for r in caplog.records)


def test_valid_artifact_loads_ok(tmp_path):
    _write_artifact(tmp_path, version="good_v0")
    model, metadata, status = load_isotonic_calibration(tmp_path)
    assert status == STATUS_OK
    assert model is not None
    assert metadata["version"] == "good_v0"


# -----------------------------------------------------------------------------
# 2. Row-level partial missingness
# -----------------------------------------------------------------------------


def test_partial_missing_cal_pd_does_not_blank_whole_run(tmp_path, caplog):
    _write_artifact(tmp_path, version="v1")
    df = pd.DataFrame([_shadow_row(), _shadow_row(cal_pd=np.nan), _shadow_row()])
    with caplog.at_level("WARNING"):
        out = compute_shadow_risk_multiplier(df, artifacts_dir=tmp_path, run_id="run4")
    assert out["shadow_status"].tolist() == [ROW_STATUS_OK, ROW_STATUS_MISSING_CAL_PD, ROW_STATUS_OK]
    assert out["shadow_calibrated_risk"].notna().sum() == 2
    assert out.loc[1, "shadow_multiplier_base"] != out.loc[1, "shadow_multiplier_base"]  # NaN
    assert out.loc[0, "shadow_multiplier_base"] == out.loc[0, "shadow_multiplier_base"]  # not NaN
    # Run-level status counts logged, not a run-wide failure.
    assert any("missing_cal_pd" in r.message for r in caplog.records)


def test_partial_missing_combined_cap_isolated_per_row(tmp_path):
    _write_artifact(tmp_path, version="v1")
    df = pd.DataFrame([_shadow_row(), _shadow_row(combined_cap=np.nan)])
    out = compute_shadow_risk_multiplier(df, artifacts_dir=tmp_path)
    assert out.loc[0, "shadow_status"] == ROW_STATUS_OK
    assert out.loc[1, "shadow_status"] == ROW_STATUS_MISSING_COMBINED_CAP
    assert pd.isna(out.loc[1, "shadow_limit_post_transition_base"])


def test_partial_missing_assigned_limit_isolated_per_row(tmp_path):
    _write_artifact(tmp_path, version="v1")
    df = pd.DataFrame([_shadow_row(), _shadow_row(assigned_limit=np.nan)])
    out = compute_shadow_risk_multiplier(df, artifacts_dir=tmp_path)
    assert out.loc[0, "shadow_status"] == ROW_STATUS_OK
    assert out.loc[1, "shadow_status"] == ROW_STATUS_MISSING_ASSIGNED_LIMIT


def test_no_cal_pd_column_at_all(tmp_path):
    _write_artifact(tmp_path, version="v1")
    df = pd.DataFrame([{k: v for k, v in _shadow_row().items() if k != "cal_pd"}])
    out = compute_shadow_risk_multiplier(df, artifacts_dir=tmp_path)
    assert (out["shadow_status"] == ROW_STATUS_MISSING_CAL_PD).all()


# -----------------------------------------------------------------------------
# 3. Literal scenario/version assertions
# -----------------------------------------------------------------------------


def test_shadow_scenarios_match_agreed_business_parameters():
    assert SHADOW_SCENARIOS == {
        "base": {"r_floor": 0.138, "m_min": 0.40, "m_max": 1.00},
        "conservative": {"r_floor": 0.12, "m_min": 0.40, "m_max": 1.00},
    }


def test_shadow_policy_version_is_set():
    assert isinstance(SHADOW_POLICY_VERSION, str) and len(SHADOW_POLICY_VERSION) > 0


# -----------------------------------------------------------------------------
# 4. Monotonicity / boundary tests for _policy_3_hybrid
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_name,params", SHADOW_SCENARIOS.items())
def test_policy_3_hybrid_is_non_increasing(scenario_name, params):
    r_plateau = 0.0754
    r_grid = np.linspace(0.0, 0.30, 200)
    m = _policy_3_hybrid(r_grid, r_plateau, params["r_floor"], params["m_max"], params["m_min"])
    assert np.all(np.diff(m) <= 1e-12), f"{scenario_name}: multiplier increased somewhere in r"


@pytest.mark.parametrize("scenario_name,params", SHADOW_SCENARIOS.items())
def test_policy_3_hybrid_plateau_and_floor_boundaries(scenario_name, params):
    r_plateau = 0.0754
    below_plateau = np.array([0.0, 0.02, r_plateau])
    above_floor = np.array([params["r_floor"], params["r_floor"] + 0.05, 1.0])
    m_below = _policy_3_hybrid(below_plateau, r_plateau, params["r_floor"], params["m_max"], params["m_min"])
    m_above = _policy_3_hybrid(above_floor, r_plateau, params["r_floor"], params["m_max"], params["m_min"])
    assert np.allclose(m_below, params["m_max"]), f"{scenario_name}: not flat at m_max through plateau"
    assert np.allclose(m_above, params["m_min"]), f"{scenario_name}: not flat at m_min beyond floor"


# -----------------------------------------------------------------------------
# 5. Transition clamp + ceiling-invariant tests
# -----------------------------------------------------------------------------


def test_transition_clamp_actually_binds_and_stays_in_bounds(tmp_path):
    _write_artifact(tmp_path, version="v1")
    # High cal_pd -> multiplier near m_min -> a huge drop from assigned_limit
    # that the +-25% transition window must clamp.
    df = pd.DataFrame([_shadow_row(cal_pd=0.9, combined_cap=100_000.0, assigned_limit=100_000.0)])
    out = compute_shadow_risk_multiplier(df, artifacts_dir=tmp_path)
    pre = float(out.loc[0, "shadow_limit_pre_transition_base"])
    post = float(out.loc[0, "shadow_limit_post_transition_base"])
    lower = 100_000.0 * 0.75
    upper = 100_000.0 * 1.25
    assert pre < lower, "fixture should actually need clamping (test would be vacuous otherwise)"
    assert lower - 1e-6 <= post <= upper + 1e-6


def test_post_transition_never_exceeds_effective_ceiling(tmp_path):
    """Engineered to push the post-finishing, post-clamp value above a
    borrower's effective ceiling: a borrower near their agent-tier ceiling
    with a large assigned_limit and a scenario that would want to push up."""
    _write_artifact(tmp_path, version="v1")
    df = pd.DataFrame(
        [
            _shadow_row(
                cal_pd=0.01,  # deep in the plateau -> multiplier = m_max = 1.0
                combined_cap=1_000_000.0,
                assigned_limit=100_000.0,  # transition window alone would allow up to 125,000
                agent_tier_ceiling_multiplier=0.10,  # effective ceiling = 100,000
            )
        ]
    )
    out = compute_shadow_risk_multiplier(df, artifacts_dir=tmp_path)
    post = float(out.loc[0, "shadow_limit_post_transition_base"])
    assert post <= 100_000.0 + 1e-6
    assert post >= 0.0


def test_post_transition_never_below_zero(tmp_path):
    _write_artifact(tmp_path, version="v1")
    df = pd.DataFrame([_shadow_row(cal_pd=0.9, combined_cap=0.0, assigned_limit=1000.0)])
    out = compute_shadow_risk_multiplier(df, artifacts_dir=tmp_path)
    post = float(out.loc[0, "shadow_limit_post_transition_base"])
    assert post >= 0.0


# -----------------------------------------------------------------------------
# 6. Parameterized anti-drift test
# -----------------------------------------------------------------------------

_default_cfg = DEFAULT_CAP_CONFIG

_rounding_floor_cfg = deepcopy(DEFAULT_CAP_CONFIG)
_rounding_floor_cfg["policy"]["active_borrower_min_limit"] = 30.0

ANTI_DRIFT_CASES = [
    pytest.param(
        _policy_row(risk_score=0.50, combined_cap=10_000.0, total_loans=5.0,
                    on_time_repayment_rate=0.70, lifetime_default_rate=0.15),
        _default_cfg,
        id="ordinary_borrower",
    ),
    pytest.param(
        _policy_row(risk_score=0.50, combined_cap=10_000.0, total_loans=10.0,
                    on_time_repayment_rate=0.95, lifetime_default_rate=0.02),
        _default_cfg,
        id="proven_good_floor_binding",
    ),
    pytest.param(
        _policy_row(risk_score=0.10, combined_cap=100.0, total_loans=5.0,
                    on_time_repayment_rate=0.30, lifetime_default_rate=0.40,
                    recent_disbursement_amount_1m=10.0, recent_repayment_amount_1m=10.0),
        _default_cfg,
        id="active_borrower_min_floor_binding",
    ),
    pytest.param(
        _policy_row(risk_score=0.95, combined_cap=900_000.0, agent_tier_ceiling_multiplier=0.05),
        _default_cfg,
        id="effective_ceiling_binding",
    ),
    pytest.param(
        _policy_row(risk_score=1.00, combined_cap=12_345.0),
        _default_cfg,
        id="rounding_boundary",
    ),
    pytest.param(
        _policy_row(risk_score=1.00, combined_cap=40.0,
                    recent_disbursement_amount_1m=10.0, recent_repayment_amount_1m=10.0),
        _rounding_floor_cfg,
        id="rounding_floor_correction",
    ),
    pytest.param(
        _policy_row(risk_score=0.50, combined_cap=0.0,
                    recent_disbursement_amount_1m=0.0, recent_repayment_amount_1m=0.0),
        _default_cfg,
        id="zero_combined_cap",
    ),
    pytest.param(
        _policy_row(risk_score=1.00, combined_cap=1_000_000.0, agent_tier_ceiling_multiplier=1.0),
        _default_cfg,
        id="highest_cap_at_ceiling",
    ),
]


@pytest.mark.parametrize("row,cfg", ANTI_DRIFT_CASES)
def test_shadow_finishing_steps_matches_policy_adjustments(row, cfg):
    df = pd.DataFrame([row])
    live = apply_policy_adjustments(df, config=cfg)
    live_tier_multiplier = live["policy_multiplier"]
    shadow_tier_cap = df["combined_cap"] * live_tier_multiplier

    shadow_final = _apply_shadow_finishing_steps(df, df["combined_cap"], shadow_tier_cap, cfg)

    pd.testing.assert_series_equal(
        shadow_final.reset_index(drop=True),
        live["policy_cap"].reset_index(drop=True),
        check_exact=True,
        check_names=False,
    )


@pytest.mark.parametrize("row,cfg", ANTI_DRIFT_CASES)
def test_shadow_rounding_matches_finalize_limits(row, cfg):
    """Second-stage anti-drift check: _apply_shadow_rounding(policy_cap, cfg)
    must reproduce finalize_limits()'s assigned_limit when fed the SAME
    (unrounded) policy_cap finalize_limits() itself would have used.

    This caught a real discovery while writing these tests: apply_policy_
    adjustments() computes a rounded final_cap + a rounding-floor
    correction internally, but never stores either -- df["policy_cap"] is
    the unrounded value, and actual rounding against assigned_limit
    happens separately in finalize_limits() via a plain round with NO
    floor correction. Mirroring the dead in-function rounding would have
    made the shadow numbers silently diverge from live assigned_limit.
    """
    df = pd.DataFrame([row])
    live_policy = apply_policy_adjustments(df, config=cfg)
    live_final = finalize_limits(live_policy, config=cfg)

    shadow_rounded = _apply_shadow_rounding(live_policy["policy_cap"], cfg)

    pd.testing.assert_series_equal(
        shadow_rounded.reset_index(drop=True),
        live_final["assigned_limit"].reset_index(drop=True),
        check_exact=True,
        check_names=False,
    )

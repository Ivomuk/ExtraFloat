"""
tests/engine/test_shadow_live_path_unchanged.py
=================================================
The literal regression test for the hard requirement running through the
whole shadow-multiplier feature: no existing engine column's value may
change. Runs run_extrafloat_limit_engine() on an identical fixture both
with shadow off (shadow_artifacts_dir=None) and with a real fitted shadow
artifact, and asserts every pre-existing live column is check_exact=True
identical between the two runs.

Deliberately does NOT modify tests/engine/test_extrafloat_pipeline.py's own
test_run_extrafloat_limit_engine_end_to_end -- that test continuing to pass
unmodified is itself part of the regression evidence that this feature is
purely additive.
"""

import json

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from extrafloat.engine.extrafloat_shadow_risk_multiplier import (
    ISOTONIC_METADATA_FILENAME,
    ISOTONIC_MODEL_FILENAME,
)
from extrafloat.engine.run_extrafloat_limit_engine import run_extrafloat_limit_engine
from tests.engine.test_extrafloat_pipeline import _features

# Every column the live engine produces today, independent of this feature.
# cal_pd is deliberately included -- some fixtures below set it so the
# shadow path actually runs, and it must still pass through unchanged.
LIVE_COLUMNS_SNAPSHOT = [
    "risk_score",
    "risk_cap",
    "risk_unresolved_loan_multiplier",
    "risk_unresolved_loan_haircut_reason",
    "capacity_cap",
    "recent_usage_cap",
    "prior_exposure_cap",
    "combined_cap",
    "combined_reason",
    "combined_top_driver",
    "risk_tier",
    "pd_decile",
    "policy_multiplier",
    "is_proven_good_borrower",
    "proven_good_floor",
    "policy_floor_applied",
    "active_floor_eligible",
    "active_floor_applied",
    "policy_reason",
    "policy_cap",
    "assigned_limit_pre_round",
    "assigned_limit",
    "final_decision_reason",
    "regulatory_cap_applied",
]


def _write_fitted_artifact(tmp_path):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit([0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0], [0.0, 0.03, 0.0754, 0.12, 0.18, 0.22, 0.25])
    r_plateau = float(iso.predict([0.25])[0])
    joblib.dump(iso, tmp_path / ISOTONIC_MODEL_FILENAME)
    (tmp_path / ISOTONIC_METADATA_FILENAME).write_text(
        json.dumps({"version": "live_path_test_v0", "cal_pd_plateau": 0.25, "r_plateau": r_plateau})
    )
    return tmp_path


def _fixture_df():
    """A handful of borrowers spanning a range of cal_pd (and therefore,
    once compute_risk_cap() derives risk_score from cal_pd on this path,
    a range of live risk tiers) plus a proven-good borrower, so the shadow
    path has real, varied work to do -- not just one trivial row."""
    rows = [
        _features(cal_pd=0.05),  # low cal_pd -> risk_score=0.95 -> tier_1
        _features(cal_pd=0.20),  # risk_score=0.80 -> tier_2
        _features(cal_pd=0.45),  # risk_score=0.55 -> tier_3
        _features(cal_pd=0.70),  # risk_score=0.30 -> tier_4
        _features(total_loans=10.0, on_time_repayment_rate=0.95,
                  lifetime_default_rate=0.02, cal_pd=0.45),  # proven-good
    ]
    return pd.concat(rows, ignore_index=True)


def test_shadow_computation_does_not_alter_live_columns(tmp_path):
    df = _fixture_df()

    without_shadow = run_extrafloat_limit_engine(df.copy(), keep_intermediate=True, shadow_artifacts_dir=None)

    artifacts_dir = _write_fitted_artifact(tmp_path)
    with_shadow = run_extrafloat_limit_engine(df.copy(), keep_intermediate=True, shadow_artifacts_dir=artifacts_dir)

    for col in LIVE_COLUMNS_SNAPSHOT:
        assert col in without_shadow.columns, f"{col} missing from shadow-off run"
        assert col in with_shadow.columns, f"{col} missing from shadow-on run"
        pd.testing.assert_series_equal(
            without_shadow[col].reset_index(drop=True),
            with_shadow[col].reset_index(drop=True),
            check_exact=True,
            check_dtype=True,
            check_names=False,
        )


def test_shadow_actually_ran_ok_on_at_least_some_rows(tmp_path):
    """Non-vacuous check: the comparison above isn't meaningful if shadow
    silently never produced real values."""
    df = _fixture_df()
    artifacts_dir = _write_fitted_artifact(tmp_path)
    result = run_extrafloat_limit_engine(df.copy(), keep_intermediate=True, shadow_artifacts_dir=artifacts_dir)
    assert (result["shadow_status"] == "ok").any()
    assert result.loc[result["shadow_status"] == "ok", "shadow_calibrated_risk"].notna().all()


def test_shadow_off_fills_nan_without_affecting_live_columns():
    df = _fixture_df()
    result = run_extrafloat_limit_engine(df.copy(), keep_intermediate=True, shadow_artifacts_dir=None)
    assert (result["shadow_status"] == "skipped_no_artifacts_dir").all()
    assert result["assigned_limit"].notna().all()
    assert result["risk_tier"].notna().all()

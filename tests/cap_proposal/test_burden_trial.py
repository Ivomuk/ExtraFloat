"""
Tests for the burden limit design trial (extrafloat/cap_proposal/).

The guarantees that matter most:
  1. running the design never changes what the live engine produces (same inputs -> same limits,
     before and after a design run), and the engine functions it swaps are always put back;
  2. the design's limits obey its own rules (ladder multiple x monthly commission, governor,
     engine ceiling, rounding).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import extrafloat.engine.run_extrafloat_limit_engine as R
from extrafloat.cap_proposal import limit_design as ld
from extrafloat.cap_proposal import run_burden_trial as trial
from extrafloat.cap_proposal.synthetic_inputs import SNAPSHOT_DT, make_synthetic_inputs

logging.disable(logging.WARNING)


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    paths = {}
    for name, frame in make_synthetic_inputs(400, seed=11).items():
        frame.to_csv(raw / f"{name}.csv", index=False)
        paths[name] = str(raw / f"{name}.csv")
    return trial.load_features(**paths, as_of_date=SNAPSHOT_DT)


@pytest.fixture(scope="module")
def settings():
    return ld.load_design_config(trial.DEFAULT_SETTINGS)


@pytest.fixture(scope="module")
def proposed(features, settings):
    return trial.run_proposed(features[1], settings)


# -- 1. live engine untouched ---------------------------------------------------


def test_live_engine_output_identical_before_and_after_design_run(features, settings):
    production, design = features
    before = trial.run_current(production)
    trial.run_proposed(design, settings)
    after = trial.run_current(production)
    pdt.assert_frame_equal(before, after)


def test_engine_functions_restored_after_design_run(features, settings):
    originals = (R.compute_capacity_cap, R.compute_risk_cap)
    trial.run_proposed(features[1], settings)
    assert (R.compute_capacity_cap, R.compute_risk_cap) == originals


def test_engine_functions_restored_when_design_raises(features, settings, monkeypatch):
    originals = (R.compute_capacity_cap, R.compute_risk_cap)

    def boom(*a, **k):
        raise RuntimeError("downstream failure")

    monkeypatch.setattr(ld, "stage_and_k", boom)
    with pytest.raises(RuntimeError, match="downstream failure"):
        trial.run_proposed(features[1], settings)
    assert (R.compute_capacity_cap, R.compute_risk_cap) == originals


def test_production_features_have_no_loan_history(features):
    production, design = features
    assert "closed_loan_count" not in production.columns
    assert "closed_loan_count" in design.columns


# -- 2. the design's own rules --------------------------------------------------


def test_limit_never_exceeds_ladder_cap_or_engine_ceiling(proposed, settings):
    lim = proposed["assigned_limit"]
    assert (lim <= proposed["design_ladder_cap"] + 50).all()        # +50: rounding to nearest 100
    assert (lim <= settings["engine_ceiling"]).all()
    assert (lim >= 0).all()
    assert (lim % 100 == 0).all()


def test_ladder_multiple_never_above_capacity_ceiling_multiple(proposed, settings):
    assert (proposed["design_k_applied"] <= settings["capacity_ceiling_multiple"]).all()


def test_governor_bounds_the_increase_over_the_reference_ceiling(proposed, settings):
    g = settings["governor"]["multiple_of_largest_past_loan"]
    gov = np.maximum(proposed["design_limit_at_reference_ceiling"], g * proposed["max_prior_loan_size"].fillna(0))
    assert (proposed["assigned_limit"] <= gov + 50).all()


def test_monthly_commission_is_six_month_commission_over_six(proposed):
    np.testing.assert_allclose(proposed["design_monthly_commission"], proposed["commission"].clip(lower=0) / 6.0)


def _stage_frame(**cols):
    base = {"closed_loan_count": 5, "lifetime_default_rate": 0.0, "on_time_repayment_rate": 1.0,
                "lifetime_worst_days_to_principal_cure": 0, "consecutive_on_time_loans": 5,
                "active_loan_days_aging_at_snapshot": 0, "anomaly_open_at_snapshot": 0}
    base.update(cols)
    return pd.DataFrame([base])


@pytest.mark.parametrize("cols, stage_prefix, k", [
    ({"closed_loan_count": 0}, "S0", 2.0),
    ({"closed_loan_count": 2}, "S1 ", 3.0),
    ({"closed_loan_count": 2, "on_time_repayment_rate": 0.5}, "S1h", 2.0),
    ({"closed_loan_count": 1, "lifetime_default_rate": np.nan}, "S1p", 2.0),
    ({"lifetime_worst_days_to_principal_cure": 45}, "S2s", 2.0),
    ({}, "S2a", 4.0),
    ({"consecutive_on_time_loans": 1}, "S2b", 3.0),
    ({"active_loan_days_aging_at_snapshot": 10}, "S2a", 3.0),        # minor arrears holds at 3x
    ({"active_loan_days_aging_at_snapshot": 45}, "S2a", 2.0),        # severe arrears drops to 2x
    ({"anomaly_open_at_snapshot": 1}, "S2a", 2.0),                   # open rollover counts as severe
])
def test_experience_ladder_stages(settings, cols, stage_prefix, k):
    stage, k_applied, *_ = ld.stage_and_k(_stage_frame(**cols), settings)
    assert stage.iloc[0].startswith(stage_prefix)
    assert k_applied.iloc[0] == k


# -- 3. trial plumbing ----------------------------------------------------------


def test_adapt_converts_worst_cure_hours_to_days(settings):
    f = _stage_frame().drop(columns="lifetime_worst_days_to_principal_cure")
    f["lifetime_worst_hours_to_principal_cure"] = 72.0
    f["commission"], f["max_prior_loan_size"] = 600_000.0, 100_000.0
    out = trial.adapt_for_design(f, settings)
    assert out["lifetime_worst_days_to_principal_cure"].iloc[0] == 3.0


def test_adapt_raises_on_missing_design_input(settings):
    f = _stage_frame(commission=600_000.0, max_prior_loan_size=0.0).drop(columns="closed_loan_count")
    with pytest.raises(ValueError, match="closed_loan_count"):
        trial.adapt_for_design(f, settings)


def test_compare_one_row_per_agent(features, settings, proposed):
    agents = trial.compare(trial.run_current(features[0]), proposed)
    assert agents["msisdn"].is_unique
    assert len(agents) == len(proposed)
    assert set(agents["change_direction"]) <= {"up", "down", "same", "not scored by current"}


def test_cli_synthetic_all_candidates(tmp_path):
    assert trial.main(["--synthetic", "300", "--all-candidates", "--out-dir", str(tmp_path)]) == 0
    idx = pd.read_csv(tmp_path / "index.csv")
    assert len(idx) == len(list(trial.HERE.glob("limit_design_*.json")))
    assert "skipped" not in idx.columns or idx["skipped"].isna().all()
    assert (tmp_path / "limit_design_proposal" / "summary.md").exists()

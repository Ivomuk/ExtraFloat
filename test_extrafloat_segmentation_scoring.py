"""
test_extrafloat_segmentation_scoring.py
=========================================
Pytest test suite for extrafloat_segmentation_scoring.

Run with:
    python -m pytest test_extrafloat_segmentation_scoring.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from extrafloat_segmentation_scoring import (
    BUSINESS_SEGMENTS,
    CAPACITY_FACTOR_GROUPS,
    DEFAULT_GROUP_WEIGHTS,
    apply_tenure_safety_cap,
    assign_capacity_tier,
    calibrate_capacity_scorecard,
    compute_agent_capacity,
    compute_capacity_score,
    compute_raw_factor_frame,
    load_scorecard,
    save_scorecard,
    validate_scorecard,
    _apply_normalization,
    _blend_group_score,
    _fit_normalization_params,
)

# ─────────────────────────────────────────────────────────────────────────────
# TEST DATA FACTORY
# ─────────────────────────────────────────────────────────────────────────────


def _make_dev_df(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Synthetic development population covering all CAPACITY_FACTOR_GROUPS columns."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "commission": rng.uniform(0, 5000, n),
            "cash_out_value_1m": rng.uniform(0, 2_000_000, n),
            "cash_out_value_3m": rng.uniform(0, 6_000_000, n),
            "cash_in_value_1m": rng.uniform(0, 1_500_000, n),
            "cash_in_value_6m": rng.uniform(0, 8_000_000, n),
            "payment_value_1m": rng.uniform(0, 500_000, n),
            "payment_value_3m": rng.uniform(0, 1_500_000, n),
            "cash_out_vol_1m": rng.uniform(0, 100, n),
            "cash_out_vol_3m": rng.uniform(0, 300, n),
            "commission_per_value_3m": rng.uniform(0, 0.01, n),
            "commission_per_value_6m": rng.uniform(0, 0.01, n),
            "tenure_years": rng.uniform(0, 8, n),
        }
    )


def _minimal_scorecard(tiers=("Low", "Mid", "High")) -> dict:
    """Small hand-built valid scorecard for unit-level tests that don't need calibration."""
    return {
        "scorecard_version": "1.0",
        "cutoff_version": "unit_test_v0",
        "factor_groups": CAPACITY_FACTOR_GROUPS,
        "group_weights": dict(DEFAULT_GROUP_WEIGHTS),
        "normalization": {
            "value": {"ref_min": 0.0, "ref_max": 100.0},
            "activity": {"ref_min": 0.0, "ref_max": 50.0},
            "efficiency": {"ref_min": 0.0, "ref_max": 10.0},
        },
        "score_scale": "0_to_1",
        "cutoffs": [1.0 / 3, 2.0 / 3],
        "tiers": list(tiers),
        "safety": {"min_tenure_years": 0.25},
        "calibration_metadata": {
            "population_description": "unit test",
            "n_agents": 0,
            "expected_tier_proportions": {t: 1.0 / len(tiers) for t in tiers},
            "is_provisional": True,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# compute_raw_factor_frame
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeRawFactorFrame:
    def test_basic_aggregation(self):
        df = pd.DataFrame({"commission": [10.0, 20.0], "cash_out_value_1m": [5.0, 5.0]})
        groups = {"value": {"columns": ["commission", "cash_out_value_1m"], "agg": "sum"}}
        out = compute_raw_factor_frame(df, groups)
        assert list(out["value"]) == [15.0, 25.0]

    def test_mean_aggregation(self):
        df = pd.DataFrame({"a": [10.0, 20.0], "b": [30.0, 40.0]})
        groups = {"eff": {"columns": ["a", "b"], "agg": "mean"}}
        out = compute_raw_factor_frame(df, groups)
        assert list(out["eff"]) == [20.0, 30.0]

    def test_missing_columns_default_to_zero(self):
        df = pd.DataFrame({"unrelated": [1.0, 2.0]})
        groups = {"value": {"columns": ["commission", "cash_out_value_1m"], "agg": "sum"}}
        out = compute_raw_factor_frame(df, groups)
        assert (out["value"] == 0.0).all()

    def test_nan_filled_with_zero(self):
        df = pd.DataFrame({"a": [10.0, np.nan]})
        groups = {"g": {"columns": ["a"], "agg": "sum"}}
        out = compute_raw_factor_frame(df, groups)
        assert out["g"].iloc[1] == 0.0

    def test_row_wise_independent_of_other_rows(self):
        """A row's raw factor value must not depend on which other rows are present."""
        df = _make_dev_df(50)
        full = compute_raw_factor_frame(df)
        single = compute_raw_factor_frame(df.iloc[[10]])
        assert np.allclose(full.iloc[10].values, single.iloc[0].values)


# ─────────────────────────────────────────────────────────────────────────────
# normalization round trip
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalization:
    def test_fit_apply_round_trip_within_range(self):
        raw = pd.DataFrame({"value": [0.0, 50.0, 100.0]})
        params = _fit_normalization_params(raw, upper_quantile=1.0)
        normed = _apply_normalization(raw, params)
        assert np.allclose(normed["value"].values, [0.0, 0.5, 1.0])

    def test_clips_values_outside_frozen_range(self):
        raw_fit = pd.DataFrame({"value": [0.0, 10.0, 20.0]})
        params = _fit_normalization_params(raw_fit, upper_quantile=1.0)  # ref_max = 20.0

        raw_new = pd.DataFrame({"value": [-5.0, 40.0]})
        normed = _apply_normalization(raw_new, params)
        # Below ref_min clips to 0.0, above ref_max clips to 1.0 — never re-scales.
        assert normed["value"].iloc[0] == 0.0
        assert normed["value"].iloc[1] == 1.0

    def test_degenerate_range_guard_returns_zero(self):
        raw = pd.DataFrame({"value": [5.0, 5.0, 5.0]})
        params = _fit_normalization_params(raw, upper_quantile=1.0)  # ref_max == ref_min == 5.0... actually min fixed at 0
        # ref_min is always 0.0; ref_max=5.0 here so this isn't degenerate.
        # Force an explicit degenerate case instead:
        degenerate_params = {"value": {"ref_min": 5.0, "ref_max": 5.0}}
        normed = _apply_normalization(raw, degenerate_params)
        assert (normed["value"] == 0.0).all()

    def test_unknown_column_defaults_to_zero_range(self):
        raw = pd.DataFrame({"unseen": [1.0, 2.0]})
        normed = _apply_normalization(raw, {})
        assert (normed["unseen"] == 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# _blend_group_score
# ─────────────────────────────────────────────────────────────────────────────


class TestBlendGroupScore:
    def test_weighted_sum(self):
        normed = pd.DataFrame({"value": [1.0], "activity": [0.0], "efficiency": [0.0]})
        weights = {"value": 0.5, "activity": 0.3, "efficiency": 0.2}
        score = _blend_group_score(normed, weights)
        assert np.isclose(score.iloc[0], 0.5)

    def test_missing_group_in_frame_contributes_zero_not_nan(self):
        """A group entirely absent from the normalized frame must not poison the score with NaN."""
        normed = pd.DataFrame({"value": [0.6]})
        weights = {"value": 0.5, "activity": 0.3, "efficiency": 0.2}
        score = _blend_group_score(normed, weights)
        assert not np.isnan(score.iloc[0])
        assert np.isclose(score.iloc[0], 0.3)

    def test_result_clipped_to_unit_interval(self):
        normed = pd.DataFrame({"value": [1.0]})
        weights = {"value": 1.5}
        score = _blend_group_score(normed, weights)
        assert score.iloc[0] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# validate_scorecard
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateScorecard:
    def test_valid_scorecard_passes(self):
        validate_scorecard(_minimal_scorecard())  # should not raise

    def test_missing_key_raises(self):
        sc = _minimal_scorecard()
        del sc["cutoffs"]
        with pytest.raises(ValueError):
            validate_scorecard(sc)

    def test_weights_not_summing_to_one_raises(self):
        sc = _minimal_scorecard()
        sc["group_weights"] = {"value": 0.5, "activity": 0.3, "efficiency": 0.3}
        with pytest.raises(ValueError):
            validate_scorecard(sc)

    def test_wrong_cutoff_count_raises(self):
        sc = _minimal_scorecard()
        sc["cutoffs"] = [0.3]  # 3 tiers need 2 cutoffs
        with pytest.raises(ValueError):
            validate_scorecard(sc)

    def test_non_ascending_cutoffs_raise(self):
        sc = _minimal_scorecard()
        sc["cutoffs"] = [0.7, 0.3]
        with pytest.raises(ValueError):
            validate_scorecard(sc)

    def test_tied_cutoffs_raise(self):
        sc = _minimal_scorecard()
        sc["cutoffs"] = [0.5, 0.5]
        with pytest.raises(ValueError):
            validate_scorecard(sc)

    def test_ref_max_below_ref_min_raises(self):
        sc = _minimal_scorecard()
        sc["normalization"]["value"] = {"ref_min": 10.0, "ref_max": 5.0}
        with pytest.raises(ValueError):
            validate_scorecard(sc)


# ─────────────────────────────────────────────────────────────────────────────
# calibrate_capacity_scorecard
# ─────────────────────────────────────────────────────────────────────────────


class TestCalibrateCapacityScorecard:
    def test_deterministic_across_repeated_calls(self):
        df = _make_dev_df(150)
        sc1 = calibrate_capacity_scorecard(df)
        sc2 = calibrate_capacity_scorecard(df)
        # generated_at_utc is a wall-clock timestamp; everything else must match exactly.
        sc1.pop("generated_at_utc")
        sc2.pop("generated_at_utc")
        assert sc1 == sc2

    def test_default_even_split_cutoffs(self):
        df = _make_dev_df(400)
        sc = calibrate_capacity_scorecard(df)
        assert len(sc["cutoffs"]) == len(BUSINESS_SEGMENTS) - 1
        props = sc["calibration_metadata"]["expected_tier_proportions"]
        for tier in BUSINESS_SEGMENTS:
            assert props[tier] == pytest.approx(1.0 / len(BUSINESS_SEGMENTS), abs=0.02)

    def test_explicit_cutoffs_used_verbatim(self):
        df = _make_dev_df(100)
        tiers = ("Low", "High")
        sc = calibrate_capacity_scorecard(df, cutoffs=[0.42], tiers=tiers)
        assert sc["cutoffs"] == [0.42]

    def test_target_tier_proportions_path(self):
        df = _make_dev_df(500)
        tiers = ("Low", "Mid", "High")
        target = {"Low": 0.2, "Mid": 0.3, "High": 0.5}
        sc = calibrate_capacity_scorecard(df, tiers=tiers, target_tier_proportions=target)
        props = sc["calibration_metadata"]["expected_tier_proportions"]
        assert props["Low"] == pytest.approx(0.2, abs=0.03)
        assert props["Mid"] == pytest.approx(0.3, abs=0.03)
        assert props["High"] == pytest.approx(0.5, abs=0.03)

    def test_target_proportions_not_summing_to_one_raises(self):
        df = _make_dev_df(50)
        with pytest.raises(ValueError):
            calibrate_capacity_scorecard(
                df, tiers=("Low", "High"), target_tier_proportions={"Low": 0.5, "High": 0.6}
            )

    def test_weights_not_summing_to_one_raises(self):
        df = _make_dev_df(50)
        with pytest.raises(ValueError):
            calibrate_capacity_scorecard(df, group_weights={"value": 0.5, "activity": 0.6})

    def test_is_provisional_flag_recorded(self):
        df = _make_dev_df(50)
        sc = calibrate_capacity_scorecard(df, is_provisional=False, cutoff_version="reviewed_v1")
        assert sc["calibration_metadata"]["is_provisional"] is False
        assert sc["cutoff_version"] == "reviewed_v1"

    def test_output_passes_validation(self):
        df = _make_dev_df(120)
        sc = calibrate_capacity_scorecard(df)
        validate_scorecard(sc)  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# save_scorecard / load_scorecard
# ─────────────────────────────────────────────────────────────────────────────


class TestScorecardPersistence:
    def test_round_trip(self, tmp_path):
        df = _make_dev_df(80)
        sc = calibrate_capacity_scorecard(df)
        path = str(tmp_path / "scorecard.json")
        save_scorecard(sc, path)
        loaded = load_scorecard(path)
        assert loaded["cutoffs"] == sc["cutoffs"]
        assert loaded["group_weights"] == sc["group_weights"]
        assert loaded["tiers"] == sc["tiers"]

    def test_refuses_overwrite_by_default(self, tmp_path):
        df = _make_dev_df(40)
        sc = calibrate_capacity_scorecard(df)
        path = str(tmp_path / "scorecard.json")
        save_scorecard(sc, path)
        with pytest.raises(FileExistsError):
            save_scorecard(sc, path)

    def test_overwrite_true_allows_replace(self, tmp_path):
        df = _make_dev_df(40)
        sc = calibrate_capacity_scorecard(df)
        path = str(tmp_path / "scorecard.json")
        save_scorecard(sc, path)
        save_scorecard(sc, path, overwrite=True)  # should not raise

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_scorecard(str(tmp_path / "does_not_exist.json"))

    def test_load_invalid_scorecard_raises(self, tmp_path):
        import json

        bad_path = tmp_path / "bad.json"
        bad_path.write_text(json.dumps({"not": "a scorecard"}))
        with pytest.raises(ValueError):
            load_scorecard(str(bad_path))


# ─────────────────────────────────────────────────────────────────────────────
# compute_capacity_score / assign_capacity_tier — core determinism invariant
# ─────────────────────────────────────────────────────────────────────────────


class TestProductionScoringDeterminism:
    """The core invariant this module exists to guarantee: given a frozen
    scorecard, the same agent feature values always produce the same score
    and tier, regardless of which other agents are present in the run."""

    def test_score_independent_of_population(self):
        df = _make_dev_df(300)
        sc = calibrate_capacity_scorecard(df)

        full_scores = compute_capacity_score(df, sc)

        for idx in (0, 37, 150, 299):
            solo = df.iloc[[idx]]
            solo_score = compute_capacity_score(solo, sc).iloc[0]
            assert np.isclose(solo_score, full_scores.iloc[idx])

    def test_tier_independent_of_population(self):
        df = _make_dev_df(300)
        sc = calibrate_capacity_scorecard(df)

        full_tiers = assign_capacity_tier(compute_capacity_score(df, sc), sc)

        subset = df.iloc[::3].reset_index(drop=True)  # every third row, different population
        subset_scores = compute_capacity_score(subset, sc)
        subset_tiers = assign_capacity_tier(subset_scores, sc)

        for local_i, orig_i in enumerate(range(0, len(df), 3)):
            assert subset_tiers.iloc[local_i] == full_tiers.iloc[orig_i]

    def test_reordering_rows_does_not_change_individual_results(self):
        df = _make_dev_df(100)
        sc = calibrate_capacity_scorecard(df)

        shuffled = df.sample(frac=1.0, random_state=123)
        orig_scores = compute_capacity_score(df, sc)
        shuffled_scores = compute_capacity_score(shuffled, sc)

        for idx in shuffled.index:
            assert np.isclose(shuffled_scores.loc[idx], orig_scores.loc[idx])

    def test_assign_capacity_tier_respects_cutoffs(self):
        sc = _minimal_scorecard(tiers=("Low", "Mid", "High"))
        scores = pd.Series([0.0, 0.32, 0.34, 0.6, 0.99])
        tiers = assign_capacity_tier(scores, sc)
        assert list(tiers) == ["Low", "Low", "Mid", "Mid", "High"]


# ─────────────────────────────────────────────────────────────────────────────
# apply_tenure_safety_cap
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyTenureSafetyCap:
    def test_short_tenure_downgraded_one_step(self):
        sc = _minimal_scorecard(tiers=("Low", "Mid", "High"))
        tier = pd.Series(["High", "High"])
        tenure = pd.Series([0.1, 5.0])  # first breaches min_tenure_years=0.25
        final, flags = apply_tenure_safety_cap(tier, tenure, sc)
        assert list(final) == ["Mid", "High"]
        assert flags.iloc[0] != ""
        assert flags.iloc[1] == ""

    def test_lowest_tier_cannot_downgrade_further(self):
        sc = _minimal_scorecard(tiers=("Low", "Mid", "High"))
        tier = pd.Series(["Low"])
        tenure = pd.Series([0.0])
        final, _ = apply_tenure_safety_cap(tier, tenure, sc)
        assert final.iloc[0] == "Low"

    def test_nan_tenure_treated_as_zero_and_downgraded(self):
        sc = _minimal_scorecard(tiers=("Low", "Mid", "High"))
        tier = pd.Series(["High"])
        tenure = pd.Series([np.nan])
        final, flags = apply_tenure_safety_cap(tier, tenure, sc)
        assert final.iloc[0] == "Mid"
        assert flags.iloc[0] != ""


# ─────────────────────────────────────────────────────────────────────────────
# compute_agent_capacity — top-level orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeAgentCapacity:
    def test_output_columns(self):
        df = _make_dev_df(60)
        sc = calibrate_capacity_scorecard(df)
        out = compute_agent_capacity(df, sc)
        for col in (
            "capacity_score", "capacity_tier_raw", "capacity_tier",
            "capacity_safety_flags", "scorecard_version", "cutoff_version",
        ):
            assert col in out.columns

    def test_row_count_and_index_preserved(self):
        df = _make_dev_df(60)
        sc = calibrate_capacity_scorecard(df)
        out = compute_agent_capacity(df, sc)
        assert len(out) == len(df)
        assert list(out.index) == list(df.index)

    def test_dormant_agents_forced_to_lowest_tier(self):
        df = _make_dev_df(60)
        sc = calibrate_capacity_scorecard(df)
        is_dormant = pd.Series(False, index=df.index)
        is_dormant.iloc[:5] = True

        out = compute_agent_capacity(df, sc, is_dormant=is_dormant)
        assert (out["capacity_tier"].iloc[:5] == sc["tiers"][0]).all()
        assert (out["capacity_safety_flags"].iloc[:5] == "dormant").all()

    def test_non_dormant_agents_unaffected_by_dormant_flag_presence(self):
        df = _make_dev_df(60)
        sc = calibrate_capacity_scorecard(df)
        no_dormant = pd.Series(False, index=df.index)

        out_without = compute_agent_capacity(df, sc)
        out_with = compute_agent_capacity(df, sc, is_dormant=no_dormant)
        assert (out_without["capacity_tier"] == out_with["capacity_tier"]).all()

    def test_scorecard_version_columns_populated(self):
        df = _make_dev_df(30)
        sc = calibrate_capacity_scorecard(df, cutoff_version="v42")
        out = compute_agent_capacity(df, sc)
        assert (out["cutoff_version"] == "v42").all()
        assert (out["scorecard_version"] == sc["scorecard_version"]).all()

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
    SCORECARD_SCHEMA_VERSION,
    apply_tenure_safety_cap,
    assign_capacity_tier,
    calibrate_capacity_scorecard,
    compute_agent_capacity,
    compute_capacity_score,
    compute_group_scores,
    compute_raw_kpi_frame,
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
    """Synthetic development population covering exactly the KPI columns
    CAPACITY_FACTOR_GROUPS references (the widest-window variant of each
    KPI only — no overlapping 1m/3m or 1m/6m pairs)."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "commission": rng.uniform(0, 5000, n),
            "cash_out_value_3m": rng.uniform(0, 6_000_000, n),
            "cash_in_value_6m": rng.uniform(0, 8_000_000, n),
            "payment_value_3m": rng.uniform(0, 1_500_000, n),
            "cash_out_vol_3m": rng.uniform(0, 300, n),
            "commission_per_value_3m": rng.uniform(0, 0.01, n),
            "commission_per_value_6m": rng.uniform(0, 0.01, n),
            "tenure_years": rng.uniform(0, 8, n),
        }
    )


def _minimal_scorecard(tiers=("Low", "Mid", "High")) -> dict:
    """Small hand-built valid scorecard for unit-level tests that don't need calibration.

    Reuses the real CAPACITY_FACTOR_GROUPS (per-KPI within-group weights)
    so structural tests exercise the actual default schema shape.
    """
    kpi_cols = sorted(
        {col for spec in CAPACITY_FACTOR_GROUPS.values() for col in spec["columns"]}
    )
    n_tiers = len(tiers)
    return {
        "scorecard_version": SCORECARD_SCHEMA_VERSION,
        "cutoff_version": "unit_test_v0",
        "factor_groups": CAPACITY_FACTOR_GROUPS,
        "group_weights": dict(DEFAULT_GROUP_WEIGHTS),
        "normalization": {col: {"ref_min": 0.0, "ref_max": 100.0} for col in kpi_cols},
        "score_scale": "0_to_1",
        "cutoffs": [i / n_tiers for i in range(1, n_tiers)],
        "tiers": list(tiers),
        "safety": {"min_tenure_years": 0.25},
        "calibration_metadata": {
            "population_description": "unit test",
            "n_agents": 0,
            "expected_tier_proportions": {t: 1.0 / n_tiers for t in tiers},
            "is_provisional": True,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# compute_raw_kpi_frame
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeRawKpiFrame:
    def test_basic_extraction(self):
        df = pd.DataFrame({"commission": [10.0, 20.0], "cash_out_value_3m": [5.0, 6.0]})
        groups = {"value": {"columns": {"commission": 0.5, "cash_out_value_3m": 0.5}}}
        out = compute_raw_kpi_frame(df, groups)
        assert list(out["commission"]) == [10.0, 20.0]
        assert list(out["cash_out_value_3m"]) == [5.0, 6.0]

    def test_output_has_one_column_per_unique_kpi_across_groups(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        groups = {
            "g1": {"columns": {"a": 1.0}},
            "g2": {"columns": {"b": 0.5, "c": 0.5}},
        }
        out = compute_raw_kpi_frame(df, groups)
        assert set(out.columns) == {"a", "b", "c"}

    def test_missing_column_raises_by_default(self):
        df = pd.DataFrame({"unrelated": [1.0, 2.0]})
        groups = {"value": {"columns": {"commission": 1.0}}}
        with pytest.raises(ValueError, match="missing"):
            compute_raw_kpi_frame(df, groups)

    def test_partially_missing_columns_raises_by_default(self):
        """Even when SOME declared columns are present, a missing one must
        still fail closed — a partial computation is not comparable to a
        fully-populated one."""
        df = pd.DataFrame({"commission": [10.0, 20.0]})
        groups = {"value": {"columns": {"commission": 0.5, "cash_out_value_3m": 0.5}}}
        with pytest.raises(ValueError, match="cash_out_value_3m"):
            compute_raw_kpi_frame(df, groups)

    def test_missing_column_zero_fills_with_explicit_escape_hatch(self):
        df = pd.DataFrame({"commission": [10.0, 20.0]})
        groups = {"value": {"columns": {"commission": 0.5, "cash_out_value_3m": 0.5}}}
        out = compute_raw_kpi_frame(df, groups, on_missing_column="zero")
        assert (out["cash_out_value_3m"] == 0.0).all()
        assert list(out["commission"]) == [10.0, 20.0]

    def test_invalid_on_missing_column_value_raises(self):
        df = pd.DataFrame({"commission": [10.0]})
        groups = {"value": {"columns": {"commission": 1.0}}}
        with pytest.raises(ValueError, match="on_missing_column"):
            compute_raw_kpi_frame(df, groups, on_missing_column="bogus")

    def test_nan_filled_with_zero(self):
        df = pd.DataFrame({"a": [10.0, np.nan]})
        groups = {"g": {"columns": {"a": 1.0}}}
        out = compute_raw_kpi_frame(df, groups)
        assert out["a"].iloc[1] == 0.0

    def test_row_wise_independent_of_other_rows(self):
        """A row's raw KPI value must not depend on which other rows are present."""
        df = _make_dev_df(50)
        full = compute_raw_kpi_frame(df, CAPACITY_FACTOR_GROUPS)
        single = compute_raw_kpi_frame(df.iloc[[10]], CAPACITY_FACTOR_GROUPS)
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
        degenerate_params = {"value": {"ref_min": 5.0, "ref_max": 5.0}}
        normed = _apply_normalization(raw, degenerate_params)
        assert (normed["value"] == 0.0).all()

    def test_unknown_column_defaults_to_zero_range(self):
        raw = pd.DataFrame({"unseen": [1.0, 2.0]})
        normed = _apply_normalization(raw, {})
        assert (normed["unseen"] == 0.0).all()

    def test_kpis_normalized_independently(self):
        """Each KPI's ref range is fit from its own distribution only —
        an outlier in one KPI must not affect another's normalization."""
        raw = pd.DataFrame({"small_scale": [0.0, 0.005, 0.01], "large_scale": [0.0, 500.0, 1000.0]})
        params = _fit_normalization_params(raw, upper_quantile=1.0)
        normed = _apply_normalization(raw, params)
        assert np.allclose(normed["small_scale"].values, [0.0, 0.5, 1.0])
        assert np.allclose(normed["large_scale"].values, [0.0, 0.5, 1.0])


# ─────────────────────────────────────────────────────────────────────────────
# compute_group_scores — the unit-mixing fix
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeGroupScores:
    def test_weighted_combination_within_group(self):
        normed = pd.DataFrame({"a": [1.0], "b": [0.0]})
        groups = {"g": {"columns": {"a": 0.75, "b": 0.25}}}
        out = compute_group_scores(normed, groups)
        assert np.isclose(out["g"].iloc[0], 0.75)

    def test_equal_weights_average_normalized_values(self):
        normed = pd.DataFrame({"a": [1.0], "b": [0.0]})
        groups = {"g": {"columns": {"a": 0.5, "b": 0.5}}}
        out = compute_group_scores(normed, groups)
        assert np.isclose(out["g"].iloc[0], 0.5)

    def test_one_group_per_column(self):
        normed = pd.DataFrame({"a": [0.2], "b": [0.8]})
        groups = {"g1": {"columns": {"a": 1.0}}, "g2": {"columns": {"b": 1.0}}}
        out = compute_group_scores(normed, groups)
        assert set(out.columns) == {"g1", "g2"}
        assert np.isclose(out["g1"].iloc[0], 0.2)
        assert np.isclose(out["g2"].iloc[0], 0.8)

    def test_result_clipped_to_unit_interval(self):
        normed = pd.DataFrame({"a": [1.0]})
        groups = {"g": {"columns": {"a": 1.5}}}  # deliberately mis-specified weight
        out = compute_group_scores(normed, groups)
        assert out["g"].iloc[0] == 1.0

    def test_unit_mixing_no_longer_lets_large_raw_scale_dominate(self):
        """Regression test for the fixed bug: a large-raw-scale KPI (e.g.
        tenure_years, range 0..20) and a small-raw-scale KPI (e.g. a
        commission ratio, range 0..0.01) must contribute *equally* to a
        group's score when they carry equal within-group weight and are
        each individually at their own midpoint — because each is
        normalized against its own frozen range *before* being combined,
        not averaged together in raw units."""
        # Two agents: one at the low end of both KPIs' own ranges, one at
        # the high end of both — after independent normalization both KPIs
        # read the same [0, 1] value for a given agent, so equal weights
        # must produce the exact midpoint regardless of raw scale.
        raw = pd.DataFrame(
            {
                "small_scale_ratio": [0.0, 0.005, 0.01],
                "large_scale_tenure": [0.0, 10.0, 20.0],
            }
        )
        normalization = _fit_normalization_params(raw, upper_quantile=1.0)
        normalized = _apply_normalization(raw, normalization)
        groups = {
            "efficiency": {"columns": {"small_scale_ratio": 0.5, "large_scale_tenure": 0.5}}
        }
        out = compute_group_scores(normalized, groups)
        assert np.allclose(out["efficiency"].values, [0.0, 0.5, 1.0])


# ─────────────────────────────────────────────────────────────────────────────
# _blend_group_score
# ─────────────────────────────────────────────────────────────────────────────


class TestBlendGroupScore:
    def test_weighted_sum(self):
        group_scores = pd.DataFrame({"value": [1.0], "activity": [0.0], "efficiency": [0.0]})
        weights = {"value": 0.5, "activity": 0.3, "efficiency": 0.2}
        score = _blend_group_score(group_scores, weights)
        assert np.isclose(score.iloc[0], 0.5)

    def test_missing_group_in_frame_contributes_zero_not_nan(self):
        """A group entirely absent from the frame must not poison the score with NaN."""
        group_scores = pd.DataFrame({"value": [0.6]})
        weights = {"value": 0.5, "activity": 0.3, "efficiency": 0.2}
        score = _blend_group_score(group_scores, weights)
        assert not np.isnan(score.iloc[0])
        assert np.isclose(score.iloc[0], 0.3)

    def test_result_clipped_to_unit_interval(self):
        group_scores = pd.DataFrame({"value": [1.0]})
        weights = {"value": 1.5}
        score = _blend_group_score(group_scores, weights)
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

    def test_incompatible_schema_version_raises(self):
        sc = _minimal_scorecard()
        sc["scorecard_version"] = "1.0"
        with pytest.raises(ValueError, match="scorecard_version"):
            validate_scorecard(sc)

    def test_weights_not_summing_to_one_raises(self):
        sc = _minimal_scorecard()
        sc["group_weights"] = {"value": 0.5, "activity": 0.3, "efficiency": 0.3}
        with pytest.raises(ValueError):
            validate_scorecard(sc)

    def test_negative_group_weight_raises(self):
        sc = _minimal_scorecard()
        sc["group_weights"] = {"value": 1.5, "activity": -0.5, "efficiency": 0.0}
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_scorecard(sc)

    def test_non_finite_group_weight_raises(self):
        sc = _minimal_scorecard()
        sc["group_weights"] = {"value": float("nan"), "activity": 0.3, "efficiency": 0.2}
        with pytest.raises(ValueError, match="finite"):
            validate_scorecard(sc)

    def test_group_weight_referencing_unknown_group_raises(self):
        sc = _minimal_scorecard()
        sc["group_weights"] = {"value": 0.5, "activity": 0.3, "not_a_real_group": 0.2}
        with pytest.raises(ValueError, match="unknown factor group"):
            validate_scorecard(sc)

    def test_empty_columns_factor_group_raises(self):
        sc = _minimal_scorecard()
        sc["factor_groups"] = {**sc["factor_groups"], "value": {"columns": {}}}
        with pytest.raises(ValueError, match="no columns"):
            validate_scorecard(sc)

    def test_within_group_weights_not_summing_to_one_raises(self):
        sc = _minimal_scorecard()
        sc["factor_groups"] = {
            **sc["factor_groups"],
            "activity": {"columns": {"cash_out_vol_3m": 0.5}},  # should be 1.0
        }
        with pytest.raises(ValueError, match="within-group weights"):
            validate_scorecard(sc)

    def test_negative_within_group_weight_raises(self):
        sc = _minimal_scorecard()
        sc["factor_groups"] = {
            **sc["factor_groups"],
            "value": {
                "columns": {"commission": 1.5, "cash_out_value_3m": -0.5,
                            "cash_in_value_6m": 0.0, "payment_value_3m": 0.0},
            },
        }
        with pytest.raises(ValueError, match="must be >= 0"):
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

    def test_non_finite_cutoff_raises(self):
        sc = _minimal_scorecard()
        sc["cutoffs"] = [float("nan"), 0.7]
        with pytest.raises(ValueError, match="finite"):
            validate_scorecard(sc)

    def test_cutoff_above_one_raises_for_0_to_1_scale(self):
        sc = _minimal_scorecard(tiers=("Low", "High"))
        sc["cutoffs"] = [1.5]
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_scorecard(sc)

    def test_cutoff_below_zero_raises_for_0_to_1_scale(self):
        sc = _minimal_scorecard(tiers=("Low", "High"))
        sc["cutoffs"] = [-0.1]
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_scorecard(sc)

    def test_empty_tiers_raises(self):
        sc = _minimal_scorecard()
        sc["tiers"] = []
        sc["cutoffs"] = []
        with pytest.raises(ValueError, match="tiers must not be empty"):
            validate_scorecard(sc)

    def test_duplicate_tier_names_raise(self):
        sc = _minimal_scorecard(tiers=("Low", "Mid", "Low"))
        with pytest.raises(ValueError, match="duplicate"):
            validate_scorecard(sc)

    def test_empty_tier_name_raises(self):
        sc = _minimal_scorecard(tiers=("Low", "", "High"))
        with pytest.raises(ValueError, match="empty names"):
            validate_scorecard(sc)

    def test_normalization_missing_entry_for_referenced_kpi_raises(self):
        sc = _minimal_scorecard()
        del sc["normalization"]["tenure_years"]
        with pytest.raises(ValueError, match="tenure_years"):
            validate_scorecard(sc)

    def test_ref_max_below_ref_min_raises(self):
        sc = _minimal_scorecard()
        sc["normalization"]["commission"] = {"ref_min": 10.0, "ref_max": 5.0}
        with pytest.raises(ValueError):
            validate_scorecard(sc)

    def test_non_finite_normalization_value_raises(self):
        sc = _minimal_scorecard()
        sc["normalization"]["commission"] = {"ref_min": 0.0, "ref_max": float("inf")}
        with pytest.raises(ValueError, match="finite"):
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

    def test_within_group_weights_not_summing_to_one_raises(self):
        df = _make_dev_df(50)
        bad_groups = {
            **CAPACITY_FACTOR_GROUPS,
            "activity": {"columns": {"cash_out_vol_3m": 0.5}},
        }
        with pytest.raises(ValueError, match="within-group weights"):
            calibrate_capacity_scorecard(df, factor_groups=bad_groups)

    def test_empty_columns_factor_group_raises(self):
        df = _make_dev_df(50)
        bad_groups = {**CAPACITY_FACTOR_GROUPS, "activity": {"columns": {}}}
        with pytest.raises(ValueError, match="no columns"):
            calibrate_capacity_scorecard(df, factor_groups=bad_groups)

    def test_missing_column_raises_by_default(self):
        df = _make_dev_df(50).drop(columns=["tenure_years"])
        with pytest.raises(ValueError, match="tenure_years"):
            calibrate_capacity_scorecard(df)

    def test_missing_column_escape_hatch_allows_calibration(self):
        df = _make_dev_df(50).drop(columns=["tenure_years"])
        sc = calibrate_capacity_scorecard(df, on_missing_column="zero")  # should not raise
        validate_scorecard(sc)

    def test_is_provisional_flag_recorded(self):
        df = _make_dev_df(50)
        sc = calibrate_capacity_scorecard(df, is_provisional=False, cutoff_version="reviewed_v1")
        assert sc["calibration_metadata"]["is_provisional"] is False
        assert sc["cutoff_version"] == "reviewed_v1"

    def test_output_passes_validation(self):
        df = _make_dev_df(120)
        sc = calibrate_capacity_scorecard(df)
        validate_scorecard(sc)  # should not raise

    def test_output_scorecard_version_matches_current_schema(self):
        df = _make_dev_df(30)
        sc = calibrate_capacity_scorecard(df)
        assert sc["scorecard_version"] == SCORECARD_SCHEMA_VERSION

    def test_normalization_keyed_by_individual_kpi_not_group_name(self):
        df = _make_dev_df(30)
        sc = calibrate_capacity_scorecard(df)
        expected_kpis = {
            col for spec in CAPACITY_FACTOR_GROUPS.values() for col in spec["columns"]
        }
        assert set(sc["normalization"].keys()) == expected_kpis
        assert not expected_kpis & {"value", "activity", "efficiency"}


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

    def test_load_incompatible_schema_version_raises(self, tmp_path):
        import json

        df = _make_dev_df(40)
        sc = calibrate_capacity_scorecard(df)
        sc["scorecard_version"] = "1.0"
        bad_path = tmp_path / "old_schema.json"
        bad_path.write_text(json.dumps(sc, default=str))
        with pytest.raises(ValueError, match="scorecard_version"):
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

    def test_missing_column_in_production_raises_by_default(self):
        df = _make_dev_df(50)
        sc = calibrate_capacity_scorecard(df)
        prod_df = df.drop(columns=["tenure_years"])
        with pytest.raises(ValueError, match="tenure_years"):
            compute_capacity_score(prod_df, sc)

    def test_missing_column_in_production_escape_hatch(self):
        df = _make_dev_df(50)
        sc = calibrate_capacity_scorecard(df)
        prod_df = df.drop(columns=["tenure_years"])
        score = compute_capacity_score(prod_df, sc, config={"on_missing_column": "zero"})
        assert not score.isna().any()


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

    def test_missing_column_raises_by_default(self):
        df = _make_dev_df(60)
        sc = calibrate_capacity_scorecard(df)
        prod_df = df.drop(columns=["commission"])
        with pytest.raises(ValueError, match="commission"):
            compute_agent_capacity(prod_df, sc)

    def test_missing_column_escape_hatch_via_config(self):
        df = _make_dev_df(60)
        sc = calibrate_capacity_scorecard(df)
        prod_df = df.drop(columns=["commission"])
        out = compute_agent_capacity(prod_df, sc, config={"on_missing_column": "zero"})
        assert len(out) == len(prod_df)

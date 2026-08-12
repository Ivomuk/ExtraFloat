"""
Tests for commission-based agent tier assignment.

Verifies that agent_category and agent_tier_ceiling_multiplier are derived
solely from the commission column (6-month rolling sum from mfs_daily_agent_mart),
NOT from agent_profile (MTN's top-level classification).
"""

import pandas as pd
import pytest

from extrafloat.engine.extrafloat_limit_engine_caps import DEFAULT_CAP_CONFIG
from extrafloat.engine.extrafloat_limit_engine_features import (
    TRANSACTION_CAPACITY_REQUIRED_COLUMNS,
    prepare_transaction_capacity_features,
)


def _minimal_df(commission, agent_profile="Agent Bronze Class"):
    """Build the smallest valid DataFrame for prepare_transaction_capacity_features."""
    row = {col: [0] for col in TRANSACTION_CAPACITY_REQUIRED_COLUMNS}
    row["msisdn"] = ["256700000001"]
    row["snapshot_dt"] = [pd.Timestamp("2026-07-31")]
    row["agent_profile"] = [agent_profile]
    row["commission"] = [commission]
    return pd.DataFrame(row)


def _tier_result(commission, agent_profile="Agent Bronze Class"):
    df = _minimal_df(commission, agent_profile)
    out = prepare_transaction_capacity_features(df)
    return out["agent_category"].iloc[0], out["agent_tier_ceiling_multiplier"].iloc[0]


# ---------------------------------------------------------------------------
# All 8 tiers — exact threshold values
# ---------------------------------------------------------------------------


class TestCommissionToAgentTierExact:
    def test_diamond(self):
        cat, mult = _tier_result(1_000_000)
        assert cat == "diamond"
        assert mult == pytest.approx(1.00)

    def test_titanium(self):
        cat, mult = _tier_result(750_000)
        assert cat == "titanium"
        assert mult == pytest.approx(0.75)

    def test_platinum(self):
        cat, mult = _tier_result(500_000)
        assert cat == "platinum"
        assert mult == pytest.approx(0.50)

    def test_gold(self):
        cat, mult = _tier_result(350_000)
        assert cat == "gold"
        assert mult == pytest.approx(0.35)

    def test_silver(self):
        cat, mult = _tier_result(250_000)
        assert cat == "silver"
        assert mult == pytest.approx(0.25)

    def test_bronze(self):
        cat, mult = _tier_result(100_000)
        assert cat == "bronze"
        assert mult == pytest.approx(0.10)

    def test_new_bronze(self):
        cat, mult = _tier_result(50_000)
        assert cat == "new bronze"
        assert mult == pytest.approx(0.05)

    def test_below_threshold(self):
        cat, mult = _tier_result(49_999)
        assert cat == "Below Threshold"
        assert mult == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Boundary values — one UGX below each tier threshold
# ---------------------------------------------------------------------------


class TestCommissionBoundaries:
    def test_just_below_diamond_is_titanium(self):
        cat, mult = _tier_result(999_999)
        assert cat == "titanium"
        assert mult == pytest.approx(0.75)

    def test_just_below_titanium_is_platinum(self):
        cat, mult = _tier_result(749_999)
        assert cat == "platinum"
        assert mult == pytest.approx(0.50)

    def test_just_below_platinum_is_gold(self):
        cat, mult = _tier_result(499_999)
        assert cat == "gold"
        assert mult == pytest.approx(0.35)

    def test_just_below_gold_is_silver(self):
        cat, mult = _tier_result(349_999)
        assert cat == "silver"
        assert mult == pytest.approx(0.25)

    def test_just_below_silver_is_bronze(self):
        cat, mult = _tier_result(249_999)
        assert cat == "bronze"
        assert mult == pytest.approx(0.10)

    def test_just_below_bronze_is_new_bronze(self):
        cat, mult = _tier_result(99_999)
        assert cat == "new bronze"
        assert mult == pytest.approx(0.05)

    def test_just_below_new_bronze_is_below_threshold(self):
        cat, mult = _tier_result(49_999)
        assert cat == "Below Threshold"
        assert mult == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Edge cases: null, negative, very high commission
# ---------------------------------------------------------------------------


class TestCommissionEdgeCases:
    def test_null_commission_is_below_threshold(self):
        cat, mult = _tier_result(None)
        assert cat == "Below Threshold"
        assert mult == pytest.approx(0.0)

    def test_negative_commission_is_below_threshold(self):
        cat, mult = _tier_result(-1)
        assert cat == "Below Threshold"
        assert mult == pytest.approx(0.0)

    def test_very_high_commission_is_diamond(self):
        cat, mult = _tier_result(5_000_000)
        assert cat == "diamond"
        assert mult == pytest.approx(1.00)

    def test_zero_commission_is_below_threshold(self):
        cat, mult = _tier_result(0)
        assert cat == "Below Threshold"
        assert mult == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# agent_profile column cannot override commission-based assignment
# ---------------------------------------------------------------------------


class TestAgentProfileIrrelevant:
    def test_silver_commission_with_diamond_profile_yields_silver(self):
        """High agent_profile cannot promote an agent beyond their commission tier."""
        cat, mult = _tier_result(250_000, agent_profile="Agent Diamond Class")
        assert cat == "silver"
        assert mult == pytest.approx(0.25)

    def test_diamond_commission_with_bronze_profile_yields_diamond(self):
        """Low agent_profile cannot demote an agent with high commission."""
        cat, mult = _tier_result(1_000_000, agent_profile="Agent Bronze Class")
        assert cat == "diamond"
        assert mult == pytest.approx(1.00)

    def test_below_threshold_commission_with_diamond_profile_yields_below(self):
        """Even 'Diamond' agent_profile cannot grant XtraFloat eligibility below commission threshold."""
        cat, mult = _tier_result(10_000, agent_profile="Agent Diamond Class")
        assert cat == "Below Threshold"
        assert mult == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Config invariant: commission_thresholds ordered highest → lowest
# ---------------------------------------------------------------------------


def test_threshold_ordering_is_highest_first():
    """Detects accidental reordering of the commission_thresholds config dict."""
    thresholds = list(
        DEFAULT_CAP_CONFIG["agent_tier"]["commission_thresholds"].values()
    )
    for i in range(len(thresholds) - 1):
        assert thresholds[i] > thresholds[i + 1], (
            f"commission_thresholds out of order at index {i}: "
            f"{thresholds[i]} is not > {thresholds[i + 1]}"
        )

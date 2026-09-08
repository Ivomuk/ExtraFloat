"""
Test 5: Agent overlap across training and validation snapshots.

An agent appearing in BOTH training (2025-09-30) AND validation (2025-11-15)
snapshots is acceptable for a cross-sectional model -- it is the same entity
measured at two different points in time. However, very high overlap (>95%)
means the OOT validation is not truly independent.

Additionally, if an agent's label is persistent across periods (e.g., a
delinquent agent stays delinquent), high overlap combined with correlated labels
can inflate OOT AUC.

This test verifies the overlap computation utility is correct on synthetic data
and documents the interpretation thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pd_model.validation.schema import compute_agent_overlap


class TestComputeAgentOverlap:
    def test_zero_overlap(self):
        """Disjoint agent sets must produce 0% overlap."""
        df_train = pd.DataFrame({"agent_msisdn": [f"256{i:07d}" for i in range(100)]})
        df_val = pd.DataFrame({"agent_msisdn": [f"256{i:07d}" for i in range(100, 150)]})
        result = compute_agent_overlap(df_train, df_val)
        assert result["n_overlap"] == 0
        assert result["overlap_pct_of_train"] == 0.0
        assert result["overlap_pct_of_val"] == 0.0

    def test_full_overlap(self):
        """Identical agent sets must produce 100% overlap."""
        agents = [f"256{i:07d}" for i in range(100)]
        df_train = pd.DataFrame({"agent_msisdn": agents})
        df_val = pd.DataFrame({"agent_msisdn": agents})
        result = compute_agent_overlap(df_train, df_val)
        assert result["n_overlap"] == 100
        assert result["overlap_pct_of_train"] == 1.0
        assert result["overlap_pct_of_val"] == 1.0

    def test_partial_overlap_computed_correctly(self):
        """50% overlap must be computed correctly for both train and val denominators."""
        agents_train = [f"256{i:07d}" for i in range(100)]
        agents_val = [f"256{i:07d}" for i in range(50, 130)]  # 50 common
        df_train = pd.DataFrame({"agent_msisdn": agents_train})
        df_val = pd.DataFrame({"agent_msisdn": agents_val})
        result = compute_agent_overlap(df_train, df_val)
        assert result["n_overlap"] == 50
        assert abs(result["overlap_pct_of_train"] - 0.50) < 1e-9
        assert abs(result["overlap_pct_of_val"] - 50 / 80) < 1e-9

    def test_returns_correct_keys(self):
        df_train = pd.DataFrame({"agent_msisdn": ["a", "b", "c"]})
        df_val = pd.DataFrame({"agent_msisdn": ["b", "c", "d"]})
        result = compute_agent_overlap(df_train, df_val)
        expected = {"n_train", "n_val", "n_overlap", "overlap_pct_of_train", "overlap_pct_of_val"}
        assert set(result.keys()) == expected

    def test_high_overlap_detectable(self):
        """Test that near-100% overlap is correctly computed and flagged."""
        agents = [f"256{i:07d}" for i in range(1000)]
        df_train = pd.DataFrame({"agent_msisdn": agents})
        # 99% of val are in training, 1% new
        df_val = pd.DataFrame({
            "agent_msisdn": agents[:990] + [f"999{i:07d}" for i in range(10)]
        })
        result = compute_agent_overlap(df_train, df_val)
        assert result["n_overlap"] == 990
        assert result["overlap_pct_of_val"] > 0.98

    def test_empty_val_set(self):
        """Empty validation set must produce 0 overlap."""
        df_train = pd.DataFrame({"agent_msisdn": ["a", "b"]})
        df_val = pd.DataFrame({"agent_msisdn": pd.Series([], dtype=str)})
        result = compute_agent_overlap(df_train, df_val)
        assert result["n_overlap"] == 0
        assert result["n_val"] == 0
        assert result["overlap_pct_of_val"] == 0.0

    def test_custom_id_col(self):
        """Custom id_col parameter must be respected."""
        df_train = pd.DataFrame({"msisdn": ["a", "b", "c"]})
        df_val = pd.DataFrame({"msisdn": ["b", "c", "d"]})
        result = compute_agent_overlap(df_train, df_val, id_col="msisdn")
        assert result["n_overlap"] == 2

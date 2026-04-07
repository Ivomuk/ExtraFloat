"""
test_extrafloat_segmentation.py
================================
Pytest test suite for the extrafloat agent segmentation package.

Run with:
    python -m pytest test_extrafloat_segmentation.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# CAPABILITY PROBES
# ─────────────────────────────────────────────────────────────────────────────


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def _has_hdbscan() -> bool:
    try:
        import hdbscan  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TEST DATA FACTORIES
# ─────────────────────────────────────────────────────────────────────────────


def _agent_row(**overrides) -> dict:
    """Base agent row with all commonly required columns set to safe defaults."""
    base: dict = {
        "agent_msisdn": 256780001234,
        "pos_msisdn": "256780005678",
        "tbl_dt": "20251116",
        "activation_dt": "20200101",
        "date_of_birth": "19850601",
        "commission": 1500.0,
        "account_balance": 200000.0,
        "average_balance": 180000.0,
        "cash_out_vol_1m": 45.0,
        "cash_out_vol_3m": 130.0,
        "cash_out_vol_6m": 260.0,
        "cash_out_value_1m": 900000.0,
        "cash_out_value_3m": 2600000.0,
        "cash_out_value_6m": 5100000.0,
        "cash_in_vol_1m": 30.0,
        "cash_in_vol_3m": 88.0,
        "cash_in_vol_6m": 175.0,
        "cash_in_value_1m": 700000.0,
        "cash_in_value_3m": 2000000.0,
        "cash_in_value_6m": 4000000.0,
        "voucher_vol_1m": 5.0,
        "voucher_vol_3m": 15.0,
        "voucher_vol_6m": 30.0,
        "voucher_value_1m": 50000.0,
        "voucher_value_3m": 150000.0,
        "voucher_value_6m": 300000.0,
        "payment_vol_1m": 10.0,
        "payment_vol_3m": 30.0,
        "payment_vol_6m": 60.0,
        "payment_value_1m": 100000.0,
        "payment_value_3m": 300000.0,
        "payment_value_6m": 600000.0,
        "cash_out_comm_1m": 900.0,
        "cash_out_comm_3m": 2600.0,
        "cash_out_comm_6m": 5100.0,
        "cash_in_comm_1m": 700.0,
        "cash_in_comm_3m": 2000.0,
        "cash_in_comm_6m": 4000.0,
        "voucher_comm_1m": 50.0,
        "voucher_comm_3m": 150.0,
        "voucher_comm_6m": 300.0,
        "payment_comm_1m": 100.0,
        "payment_comm_3m": 300.0,
        "payment_comm_6m": 600.0,
        "cash_out_cust_1m": 20.0,
        "cash_out_cust_3m": 60.0,
        "cash_out_cust_6m": 120.0,
        "cash_in_cust_1m": 15.0,
        "cash_in_cust_3m": 45.0,
        "cash_in_cust_6m": 90.0,
        "payment_cust_1m": 8.0,
        "payment_cust_3m": 24.0,
        "payment_cust_6m": 48.0,
        "cash_out_peers_1m": 18.0,
        "cash_out_peers_3m": 54.0,
        "cash_out_peers_6m": 108.0,
        "cash_in_peers_1m": 12.0,
        "cash_in_peers_3m": 36.0,
        "cash_in_peers_6m": 72.0,
        "payment_peers_1m": 6.0,
        "payment_peers_3m": 18.0,
        "payment_peers_6m": 36.0,
        "voucher_peers_1m": 3.0,
        "voucher_peers_3m": 9.0,
        "voucher_peers_6m": 18.0,
        "voucher_cust_1m": 4.0,
        "voucher_cust_3m": 12.0,
        "voucher_cust_6m": 24.0,
        "cust_1m": 35.0,
        "cust_3m": 105.0,
        "cust_6m": 210.0,
        "vol_1m": 90.0,
        "vol_3m": 263.0,
        "vol_6m": 525.0,
        "revenue_1m": 1750.0,
        "revenue_3m": 5050.0,
        "revenue_6m": 10000.0,
        "cash_out_revenue_1m": 900.0,
        "cash_out_revenue_3m": 2600.0,
        "cash_out_revenue_6m": 5100.0,
        "cash_in_revenue_1m": 700.0,
        "cash_in_revenue_3m": 2000.0,
        "cash_in_revenue_6m": 4000.0,
        "payment_revenue_1m": 100.0,
        "payment_revenue_3m": 300.0,
        "payment_revenue_6m": 600.0,
        "voucher_revenue_1m": 50.0,
        "voucher_revenue_3m": 150.0,
        "voucher_revenue_6m": 300.0,
        "tenure_years": 4.5,
        "gender": "Male",
        "agent_profile": "MTNU AGENT BRONZE CLASS",
    }
    base.update(overrides)
    return base


def _make_agents_df(n: int = 80, **col_overrides) -> pd.DataFrame:
    """Build a deterministic synthetic agents DataFrame with *n* rows."""
    rng = np.random.RandomState(99)
    base = _agent_row()
    rows = []
    for i in range(n):
        row = dict(base)
        for k, v in row.items():
            if isinstance(v, float) and v > 0:
                row[k] = abs(v * (1.0 + 0.4 * rng.randn()))
        row["agent_msisdn"] = 256780000000 + i
        rows.append(row)
    df = pd.DataFrame(rows)
    for col, val in col_overrides.items():
        df[col] = val
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES MODULE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestPrepareFeatures:
    """Tests for extrafloat_segmentation_features.prepare_features."""

    def test_returns_correct_types(self):
        from extrafloat_segmentation_features import prepare_features

        df = _make_agents_df(60)
        result = prepare_features(df)
        feat_df, X_scaled, X_pca, sel_cols = result

        assert isinstance(feat_df, pd.DataFrame)
        assert isinstance(X_scaled, np.ndarray)
        assert isinstance(X_pca, np.ndarray)
        assert isinstance(sel_cols, list)

    def test_pca_col_count_matches_selected_cols(self):
        from extrafloat_segmentation_features import prepare_features

        df = _make_agents_df(60)
        _, X_scaled, X_pca, sel_cols = prepare_features(df)

        assert X_scaled.shape[1] == len(sel_cols)
        assert X_pca.ndim == 2

    def test_missing_required_col_raises(self):
        from extrafloat_segmentation_features import prepare_features

        df = _make_agents_df(30).drop(columns=["commission"])
        with pytest.raises(ValueError, match="commission"):
            prepare_features(df)

    def test_commission_agents_excluded(self):
        from extrafloat_segmentation_features import (
            COMMISSION_AGENTS_MSISDN,
            prepare_features,
        )

        df = _make_agents_df(50)
        commission_msisdn = next(iter(COMMISSION_AGENTS_MSISDN))
        df.loc[0, "agent_msisdn"] = commission_msisdn

        feat_df, _, _, _ = prepare_features(df)
        msisdns = pd.to_numeric(
            feat_df.get("agent_msisdn", pd.Series(dtype=float)), errors="coerce"
        )
        assert commission_msisdn not in msisdns.values

    def test_no_nan_in_scaled_output(self):
        from extrafloat_segmentation_features import prepare_features

        df = _make_agents_df(50)
        _, X_scaled, _, _ = prepare_features(df)
        assert not np.isnan(X_scaled).any(), "X_scaled must not contain NaN"

    def test_row_count_preserved(self):
        from extrafloat_segmentation_features import prepare_features

        df = _make_agents_df(40)
        feat_df, X_scaled, X_pca, _ = prepare_features(df)
        assert X_scaled.shape[0] == len(feat_df)
        assert X_pca.shape[0] == len(feat_df)

    def test_required_columns_constant(self):
        from extrafloat_segmentation_features import REQUIRED_COLUMNS

        assert "agent_msisdn" in REQUIRED_COLUMNS
        assert "commission" in REQUIRED_COLUMNS
        assert "cash_out_vol_1m" in REQUIRED_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE MODULE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestClusteringPipeline:
    """Tests for extrafloat_segmentation_pipeline.run_clustering_pipeline."""

    def _get_small_inputs(self, n=80):
        from extrafloat_segmentation_features import prepare_features

        df = _make_agents_df(n)
        feat_df, _, X_pca, sel_cols = prepare_features(df)
        return feat_df, X_pca, sel_cols

    def test_output_has_required_columns(self):
        from extrafloat_segmentation_pipeline import run_clustering_pipeline

        feat_df, X_pca, sel_cols = self._get_small_inputs()
        result = run_clustering_pipeline(feat_df, X_pca, sel_cols)

        for col in ("cluster_round1", "ensemble_cluster", "segment"):
            assert col in result.columns, f"Missing column: {col}"

    def test_segment_col_values_are_valid(self):
        from extrafloat_segmentation_pipeline import BUSINESS_SEGMENTS, run_clustering_pipeline

        feat_df, X_pca, sel_cols = self._get_small_inputs()
        result = run_clustering_pipeline(feat_df, X_pca, sel_cols)

        seg_values = result["segment"].dropna().unique()
        invalid = [s for s in seg_values if s not in BUSINESS_SEGMENTS]
        assert not invalid, f"Invalid segment values found: {invalid}"

    def test_dormant_agents_get_below_threshold(self):
        from extrafloat_segmentation_pipeline import (
            BUSINESS_SEGMENTS,
            DORMANT_FILL_LABEL,
            run_clustering_pipeline,
        )

        feat_df, X_pca, sel_cols = self._get_small_inputs(80)
        # Set first 10 agents as dormant
        feat_df = feat_df.copy()
        feat_df.loc[feat_df.index[:10], "cash_out_vol_1m"] = 0.0

        result = run_clustering_pipeline(feat_df, X_pca, sel_cols)
        dormant_segs = result.iloc[:10]["segment"].unique()

        for seg in dormant_segs:
            assert seg == BUSINESS_SEGMENTS[0], (
                f"Dormant agent got segment '{seg}' instead of 'Below Threshold'"
            )

    def test_row_count_preserved(self):
        from extrafloat_segmentation_pipeline import run_clustering_pipeline

        feat_df, X_pca, sel_cols = self._get_small_inputs()
        result = run_clustering_pipeline(feat_df, X_pca, sel_cols)
        assert len(result) == len(feat_df)

    def test_mismatched_shapes_raise(self):
        from extrafloat_segmentation_pipeline import run_clustering_pipeline

        feat_df, X_pca, sel_cols = self._get_small_inputs(60)
        X_wrong = X_pca[: len(X_pca) // 2]
        with pytest.raises(ValueError):
            run_clustering_pipeline(feat_df, X_wrong, sel_cols)

    def test_business_segments_has_8_entries(self):
        from extrafloat_segmentation_pipeline import BUSINESS_SEGMENTS

        assert len(BUSINESS_SEGMENTS) == 8


# ─────────────────────────────────────────────────────────────────────────────
# PROFILING MODULE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestClusterPackProfiles:
    """Tests for extrafloat_segmentation_profiling.build_cluster_pack_profiles."""

    def _make_df_with_clusters(self):
        df = _make_agents_df(60)
        rng = np.random.RandomState(7)
        df["ensemble_cluster"] = rng.choice(["A", "B", "C"], size=len(df))
        return df

    def test_returns_expected_keys(self):
        from extrafloat_segmentation_profiling import build_cluster_pack_profiles

        df = self._make_df_with_clusters()
        result = build_cluster_pack_profiles(df, cluster_col="ensemble_cluster")
        assert "means" in result
        assert "lifts" in result
        assert "melted" in result

    def test_invalid_cluster_col_raises(self):
        from extrafloat_segmentation_profiling import build_cluster_pack_profiles

        df = _make_agents_df(30)
        with pytest.raises(ValueError, match="cluster_col"):
            build_cluster_pack_profiles(df, cluster_col="nonexistent_col")

    def test_means_shape(self):
        from extrafloat_segmentation_profiling import PROFILING_PACKS, build_cluster_pack_profiles

        df = self._make_df_with_clusters()
        result = build_cluster_pack_profiles(
            df, cluster_col="ensemble_cluster", packs={"balances": ["account_balance", "average_balance"]}
        )
        means_df = result["means"]["balances"]
        assert means_df.shape[0] == df["ensemble_cluster"].nunique()
        assert "account_balance" in means_df.columns


class TestBuildClusterTiers:
    """Tests for extrafloat_segmentation_profiling.build_cluster_tiers."""

    def _make_scores(self):
        scores = pd.DataFrame(
            {
                "cash_in": [2.5, 0.5, 1.0, 0.3],
                "cash_out": [2.2, 0.6, 0.9, 0.2],
            },
            index=["ClusterA", "ClusterB", "ClusterC", "ClusterD"],
        )
        stats = pd.DataFrame(
            {
                "n_agents": [500, 20, 200, 300],
                "mean_tenure_months": [24, 2, 18, 12],
            },
            index=scores.index,
        )
        return scores, stats

    def test_returns_expected_columns(self):
        from extrafloat_segmentation_profiling import build_cluster_tiers

        scores, stats = self._make_scores()
        result = build_cluster_tiers(scores, cluster_stats=stats)
        for col in ("cluster", "tier", "sub_label", "n_agents", "safety_flags"):
            assert col in result.columns, f"Missing column: {col}"

    def test_small_cluster_triggers_safety_flag(self):
        from extrafloat_segmentation_profiling import build_cluster_tiers

        scores, stats = self._make_scores()
        # ClusterB has n_agents=20 < min_agents_per_tier=30
        result = build_cluster_tiers(scores, cluster_stats=stats)
        clusterb_row = result[result["cluster"] == "ClusterB"].iloc[0]
        assert clusterb_row["safety_flags"] != "", (
            "ClusterB (n=20) should have a safety flag"
        )

    def test_no_lift_columns_raises(self):
        from extrafloat_segmentation_profiling import build_cluster_tiers

        scores = pd.DataFrame(
            {"n_agents": [100, 200], "mean_tenure_months": [12, 24]},
            index=["A", "B"],
        )
        with pytest.raises(ValueError, match="no lift columns"):
            build_cluster_tiers(scores)

    def test_tier_values_are_valid(self):
        from extrafloat_segmentation_profiling import build_cluster_tiers

        scores, stats = self._make_scores()
        result = build_cluster_tiers(scores, cluster_stats=stats)
        valid_tiers = {"Platinum", "Gold", "Silver", "Bronze"}
        for tier in result["tier"]:
            assert tier in valid_tiers, f"Unexpected tier: {tier}"


class TestMergeReferenceLists:
    """Tests for extrafloat_segmentation_profiling.merge_reference_lists."""

    def _make_lists(self):
        wl = pd.DataFrame(
            {
                "agent_msisdn": [256780000001, 256780000002, 256780000003],
                "agent_category": ["Gold", "Silver", "Bronze"],
            }
        )
        bl = pd.DataFrame(
            {
                "agent_msisdn": [256780000003, 256780000004],
                "agent_category": ["Bronze", "Below Threshold"],
            }
        )
        return wl, bl

    def test_whitelist_priority(self):
        from extrafloat_segmentation_profiling import merge_reference_lists

        df = _make_agents_df(5)
        df["agent_msisdn"] = [256780000001, 256780000002, 256780000003, 256780000004, 256780000005]
        wl, bl = self._make_lists()

        result = merge_reference_lists(df, whitelist_df=wl, blacklist_df=bl)
        # Agent 256780000003 is in both -> whitelist should win
        row = result[result["agent_msisdn"].astype(str) == "256780000003"]
        assert not row.empty
        assert row.iloc[0]["CommissionDecision"] == "whitelist"

    def test_blacklist_only_agent_gets_blacklist(self):
        from extrafloat_segmentation_profiling import merge_reference_lists

        df = _make_agents_df(5)
        df["agent_msisdn"] = [256780000001, 256780000002, 256780000003, 256780000004, 256780000005]
        wl, bl = self._make_lists()

        result = merge_reference_lists(df, whitelist_df=wl, blacklist_df=bl)
        row = result[result["agent_msisdn"].astype(str) == "256780000004"]
        assert not row.empty
        assert row.iloc[0]["CommissionDecision"] == "blacklist"

    def test_invalid_msisdn_col_raises(self):
        from extrafloat_segmentation_profiling import merge_reference_lists

        df = _make_agents_df(5)
        wl = pd.DataFrame({"agent_msisdn": [1, 2]})
        with pytest.raises(ValueError, match="msisdn_col"):
            merge_reference_lists(df, whitelist_df=wl, msisdn_col="nonexistent")


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION MODULE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestClusterPurityTable:
    """Tests for extrafloat_segmentation_validation.cluster_purity_table."""

    def _make_df(self):
        return pd.DataFrame(
            {
                "ensemble_cluster": ["A", "A", "A", "B", "B", "C"],
                "agent_category": ["Gold", "Gold", "Silver", "Bronze", "Bronze", "Gold"],
            }
        )

    def test_output_columns(self):
        from extrafloat_segmentation_validation import cluster_purity_table

        df = self._make_df()
        result = cluster_purity_table(df, cluster_col="ensemble_cluster", label_col="agent_category")
        for col in ("cluster", "n_agents", "dominant_label", "dominant_count", "purity"):
            assert col in result.columns

    def test_purity_in_range(self):
        from extrafloat_segmentation_validation import cluster_purity_table

        df = self._make_df()
        result = cluster_purity_table(df)
        assert (result["purity"] >= 0.0).all()
        assert (result["purity"] <= 1.0).all()

    def test_overall_purity_in_range(self):
        from extrafloat_segmentation_validation import cluster_purity_table

        df = self._make_df()
        result = cluster_purity_table(df)
        assert "overall_purity" in result.columns
        val = result["overall_purity"].iloc[0]
        assert 0.0 <= float(val) <= 1.0

    def test_missing_cols_raises(self):
        from extrafloat_segmentation_validation import cluster_purity_table

        df = pd.DataFrame({"wrong_col": ["A", "B"]})
        with pytest.raises(ValueError):
            cluster_purity_table(df, cluster_col="ensemble_cluster")


class TestComputeARI:
    """Tests for extrafloat_segmentation_validation.compute_adjusted_rand_score."""

    def test_perfect_agreement_gives_1(self):
        from extrafloat_segmentation_validation import compute_adjusted_rand_score

        df = pd.DataFrame({"pred": ["A", "A", "B", "B"], "true": ["A", "A", "B", "B"]})
        ari = compute_adjusted_rand_score(df, cluster_col="pred", label_col="true")
        assert abs(ari - 1.0) < 1e-9

    def test_missing_col_raises(self):
        from extrafloat_segmentation_validation import compute_adjusted_rand_score

        df = pd.DataFrame({"pred": ["A", "B"]})
        with pytest.raises(ValueError):
            compute_adjusted_rand_score(df, cluster_col="pred", label_col="nonexistent")


class TestBelowThresholdCounts:
    """Tests for extrafloat_segmentation_validation.below_threshold_counts."""

    def test_output_columns(self):
        from extrafloat_segmentation_validation import below_threshold_counts

        df = pd.DataFrame(
            {
                "ensemble_cluster": ["A", "A", "B", "B", "B"],
                "agent_category": ["Below Threshold", "Gold", "Below Threshold", "Silver", "Below Threshold"],
            }
        )
        result = below_threshold_counts(df)
        for col in ("cluster", "n_total", "n_below_threshold", "bt_share_pct"):
            assert col in result.columns

    def test_share_pct_in_range(self):
        from extrafloat_segmentation_validation import below_threshold_counts

        df = pd.DataFrame(
            {
                "ensemble_cluster": ["A", "A", "A"],
                "agent_category": ["Below Threshold", "Below Threshold", "Gold"],
            }
        )
        result = below_threshold_counts(df)
        assert (result["bt_share_pct"] >= 0).all()
        assert (result["bt_share_pct"] <= 100).all()


class TestCompareClusterKpis:
    """Tests for extrafloat_segmentation_validation.compare_cluster_kpis."""

    def test_returns_multi_level_columns(self):
        from extrafloat_segmentation_validation import compare_cluster_kpis

        df = _make_agents_df(40)
        df["ensemble_cluster"] = ["good"] * 20 + ["suspect"] * 20
        df["CommissionDecision"] = "whitelist"

        result = compare_cluster_kpis(
            df,
            good_clusters=["good"],
            suspect_clusters=["suspect"],
            kpi_cols=["commission", "cash_out_vol_1m"],
        )
        assert isinstance(result.columns, pd.MultiIndex)


class TestRankFeatureImportance:
    """Tests for extrafloat_segmentation_validation.rank_feature_importance."""

    def test_returns_series_summing_to_one(self):
        from extrafloat_segmentation_validation import rank_feature_importance

        df = _make_agents_df(80)
        df["ensemble_cluster"] = ["good"] * 40 + ["suspect"] * 40
        df["CommissionDecision"] = "whitelist"

        result = rank_feature_importance(
            df,
            good_clusters=["good"],
            suspect_clusters=["suspect"],
            kpi_cols=["commission", "cash_out_vol_1m", "cash_in_value_1m"],
        )
        assert isinstance(result, pd.Series)
        assert abs(result.sum() - 1.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION MODULE TESTS
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
class TestVizModule:
    """Tests for extrafloat_segmentation_viz — require matplotlib."""

    def test_plot_pca_scatter_returns_figure(self):
        from extrafloat_segmentation_viz import plot_pca_scatter
        import matplotlib.pyplot as plt

        X_2d = np.random.RandomState(1).randn(200, 2)
        labels = np.repeat(["A", "B", "C", "D"], 50)
        fig = plot_pca_scatter(X_2d, labels, max_points=100)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_plot_cluster_distribution_returns_figure(self):
        from extrafloat_segmentation_viz import plot_cluster_distribution
        import matplotlib.pyplot as plt
        import matplotlib.figure

        df = _make_agents_df(30)
        df["segment"] = np.random.choice(["A", "B", "C"], size=len(df))
        fig = plot_cluster_distribution(df, cluster_col="segment")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_plot_cluster_distribution_invalid_col_raises(self):
        from extrafloat_segmentation_viz import plot_cluster_distribution

        df = _make_agents_df(20)
        with pytest.raises(ValueError, match="nonexistent"):
            plot_cluster_distribution(df, cluster_col="nonexistent")

    def test_plot_purity_heatmap_returns_figure(self):
        from extrafloat_segmentation_viz import plot_purity_heatmap
        import matplotlib.pyplot as plt
        import matplotlib.figure

        crosstab = pd.DataFrame(
            {"Gold": [30, 5], "Bronze": [5, 20]},
            index=["ClusterA", "ClusterB"],
        )
        fig = plot_purity_heatmap(crosstab)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION / ORCHESTRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestDefaultConfig:
    """Tests for run_extrafloat_segmentation.DEFAULT_SEGMENTATION_CONFIG."""

    def test_all_sections_present(self):
        from run_extrafloat_segmentation import DEFAULT_SEGMENTATION_CONFIG

        for section in ("data", "features", "clustering", "profiling", "output"):
            assert section in DEFAULT_SEGMENTATION_CONFIG, f"Missing section: {section}"

    def test_config_round_trip(self):
        from run_extrafloat_segmentation import _get_config

        cfg = _get_config(None)
        for section in ("data", "features", "clustering", "profiling", "output"):
            assert section in cfg

    def test_partial_override_preserved(self):
        from run_extrafloat_segmentation import _get_config

        cfg = _get_config({"features": {"corr_threshold": 0.85}})
        assert cfg["features"]["corr_threshold"] == 0.85
        # Other feature keys should still be present
        assert "skew_threshold" in cfg["features"]


class TestPipelineConstants:
    """Verify critical module constants are correctly defined."""

    def test_business_segments_length(self):
        from extrafloat_segmentation_pipeline import BUSINESS_SEGMENTS

        assert len(BUSINESS_SEGMENTS) == 8

    def test_business_segments_first_is_below_threshold(self):
        from extrafloat_segmentation_pipeline import BUSINESS_SEGMENTS

        assert BUSINESS_SEGMENTS[0] == "Below Threshold"

    def test_business_segments_last_is_diamond(self):
        from extrafloat_segmentation_pipeline import BUSINESS_SEGMENTS

        assert BUSINESS_SEGMENTS[-1] == "Diamond"

    def test_dormant_fill_label_constant(self):
        from extrafloat_segmentation_pipeline import DORMANT_FILL_LABEL

        assert isinstance(DORMANT_FILL_LABEL, str)
        assert len(DORMANT_FILL_LABEL) > 0

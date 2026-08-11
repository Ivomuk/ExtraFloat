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
        # Vary activation_dt per agent so tenure_years isn't a constant
        # column — _engineer_date_features silently drops constant columns,
        # which would otherwise make tenure_years vanish from every
        # scorecard's "efficiency" factor without anyone noticing (this is
        # exactly the failure mode compute_raw_kpi_frame's fail-closed
        # default is meant to catch — it caught it here).
        days_ago = int(rng.randint(30, 3650))
        activation_date = pd.Timestamp("2025-11-16") - pd.Timedelta(days=days_ago)
        row["activation_dt"] = activation_date.strftime("%Y%m%d")
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


class TestFlagAnomalies:
    """Tests for extrafloat_segmentation_pipeline.flag_anomalies — the
    two-stage anomaly filter (HDBSCAN global pass, then LOF local
    refinement on HDBSCAN survivors only) that runs by default
    (clustering.enable_anomaly_detection) independent of the full
    diagnostics bundle, and degrades gracefully instead of raising when
    hdbscan is unavailable."""

    def _get_small_inputs(self, n=80):
        from extrafloat_segmentation_features import prepare_features
        from extrafloat_segmentation_pipeline import _identify_dormant_mask, DEFAULT_CLUSTERING_CONFIG

        df = _make_agents_df(n)
        feat_df, _, _, sel_cols = prepare_features(df)
        active_mask = ~_identify_dormant_mask(feat_df, DEFAULT_CLUSTERING_CONFIG)
        return feat_df, sel_cols, active_mask

    def _get_inputs_with_real_clusters(self, n=300):
        """Default hdbscan_min_cluster_size=1000 always calls everything
        noise for a synthetic population this small, which never exercises
        the LOF stage (nothing survives stage 1). Lower the thresholds so
        HDBSCAN actually finds clusters, matching what a real production
        population would produce at default settings."""
        from extrafloat_segmentation_pipeline import DEFAULT_CLUSTERING_CONFIG

        feat_df, sel_cols, active_mask = self._get_small_inputs(n)
        cfg = dict(DEFAULT_CLUSTERING_CONFIG)
        cfg["hdbscan_min_cluster_size"] = 5
        cfg["hdbscan_min_samples"] = 1
        return feat_df, sel_cols, active_mask, cfg

    def test_output_has_required_columns(self):
        from extrafloat_segmentation_pipeline import flag_anomalies

        feat_df, sel_cols, active_mask = self._get_small_inputs()
        result = flag_anomalies(feat_df, sel_cols, active_mask)
        for col in ("is_anomaly", "anomaly_cluster_hdb_raw"):
            assert col in result.columns

    def test_row_count_and_index_preserved(self):
        from extrafloat_segmentation_pipeline import flag_anomalies

        feat_df, sel_cols, active_mask = self._get_small_inputs()
        result = flag_anomalies(feat_df, sel_cols, active_mask)
        assert len(result) == len(feat_df)
        assert list(result.index) == list(feat_df.index)

    def test_is_anomaly_matches_hdb_noise_label(self):
        from extrafloat_segmentation_pipeline import HDBSCAN_NOISE_LABEL, flag_anomalies

        feat_df, sel_cols, active_mask = self._get_small_inputs()
        result = flag_anomalies(feat_df, sel_cols, active_mask)
        active_result = result.loc[active_mask]
        expected = active_result["anomaly_cluster_hdb_raw"] == HDBSCAN_NOISE_LABEL
        assert (active_result["is_anomaly"] == expected).all()

    def test_dormant_agents_never_flagged_as_anomaly(self):
        from extrafloat_segmentation_pipeline import flag_anomalies

        feat_df, sel_cols, active_mask = self._get_small_inputs()
        result = flag_anomalies(feat_df, sel_cols, active_mask)
        dormant_result = result.loc[~active_mask]
        assert not dormant_result["is_anomaly"].any()
        assert dormant_result["anomaly_cluster_hdb_raw"].isna().all()

    def test_no_active_agents_returns_all_false(self):
        import pandas as pd
        from extrafloat_segmentation_pipeline import flag_anomalies

        feat_df, sel_cols, _ = self._get_small_inputs()
        no_active = pd.Series(False, index=feat_df.index)
        result = flag_anomalies(feat_df, sel_cols, no_active)
        assert not result["is_anomaly"].any()
        assert result["anomaly_cluster_hdb_raw"].isna().all()

    def test_missing_hdbscan_degrades_gracefully_not_raise(self, monkeypatch):
        import extrafloat_segmentation_pipeline as pipe

        monkeypatch.setattr(pipe, "_HDBSCAN_AVAILABLE", False)
        feat_df, sel_cols, active_mask = self._get_small_inputs()
        result = pipe.flag_anomalies(feat_df, sel_cols, active_mask)  # must not raise
        assert not result["is_anomaly"].any()
        assert result["anomaly_cluster_hdb_raw"].isna().all()

    def test_enabled_by_default_in_clustering_config(self):
        from extrafloat_segmentation_pipeline import DEFAULT_CLUSTERING_CONFIG

        assert DEFAULT_CLUSTERING_CONFIG["enable_anomaly_detection"] is True

    def test_lof_enabled_by_default_in_clustering_config(self):
        from extrafloat_segmentation_pipeline import DEFAULT_CLUSTERING_CONFIG

        assert DEFAULT_CLUSTERING_CONFIG["lof_enabled"] is True

    def test_output_has_two_stage_columns(self):
        from extrafloat_segmentation_pipeline import flag_anomalies

        feat_df, sel_cols, active_mask, cfg = self._get_inputs_with_real_clusters()
        result = flag_anomalies(feat_df, sel_cols, active_mask, config=cfg)
        for col in ("is_anomaly", "is_global_anomaly", "is_local_anomaly", "lof_score"):
            assert col in result.columns

    def test_is_anomaly_is_or_of_both_stages(self):
        from extrafloat_segmentation_pipeline import flag_anomalies

        feat_df, sel_cols, active_mask, cfg = self._get_inputs_with_real_clusters()
        result = flag_anomalies(feat_df, sel_cols, active_mask, config=cfg)
        expected = result["is_global_anomaly"] | result["is_local_anomaly"]
        assert (result["is_anomaly"] == expected).all()

    def test_global_anomaly_matches_hdb_noise_label(self):
        from extrafloat_segmentation_pipeline import HDBSCAN_NOISE_LABEL, flag_anomalies

        feat_df, sel_cols, active_mask, cfg = self._get_inputs_with_real_clusters()
        result = flag_anomalies(feat_df, sel_cols, active_mask, config=cfg)
        active_result = result.loc[active_mask]
        expected = active_result["anomaly_cluster_hdb_raw"] == HDBSCAN_NOISE_LABEL
        assert (active_result["is_global_anomaly"] == expected).all()

    def test_lof_stage_actually_runs_on_in_cluster_agents(self):
        """With real clusters present, LOF must produce a non-NaN score for
        every in-cluster (non-global-anomaly, active) agent — proving stage
        2 actually ran rather than being silently skipped."""
        from extrafloat_segmentation_pipeline import flag_anomalies

        feat_df, sel_cols, active_mask, cfg = self._get_inputs_with_real_clusters()
        result = flag_anomalies(feat_df, sel_cols, active_mask, config=cfg)
        in_cluster = active_mask & ~result["is_global_anomaly"]
        assert in_cluster.sum() > 0, "test setup produced no in-cluster agents"
        assert result.loc[in_cluster, "lof_score"].notna().all()

    def test_lof_score_nan_for_global_anomalies_and_dormant(self):
        from extrafloat_segmentation_pipeline import flag_anomalies

        feat_df, sel_cols, active_mask, cfg = self._get_inputs_with_real_clusters()
        result = flag_anomalies(feat_df, sel_cols, active_mask, config=cfg)
        assert result.loc[result["is_global_anomaly"], "lof_score"].isna().all()
        assert result.loc[~active_mask, "lof_score"].isna().all()

    def test_local_anomaly_never_true_for_global_anomalies(self):
        """LOF only ever runs on HDBSCAN survivors — a global anomaly can
        never also be flagged as a local anomaly."""
        from extrafloat_segmentation_pipeline import flag_anomalies

        feat_df, sel_cols, active_mask, cfg = self._get_inputs_with_real_clusters()
        result = flag_anomalies(feat_df, sel_cols, active_mask, config=cfg)
        assert not (result["is_global_anomaly"] & result["is_local_anomaly"]).any()

    def test_lof_disabled_skips_stage_two_only(self):
        from extrafloat_segmentation_pipeline import flag_anomalies

        feat_df, sel_cols, active_mask, cfg = self._get_inputs_with_real_clusters()
        cfg_no_lof = dict(cfg)
        cfg_no_lof["lof_enabled"] = False

        with_lof = flag_anomalies(feat_df, sel_cols, active_mask, config=cfg)
        without_lof = flag_anomalies(feat_df, sel_cols, active_mask, config=cfg_no_lof)

        assert not without_lof["is_local_anomaly"].any()
        assert without_lof["lof_score"].isna().all()
        # Stage 1 is untouched by disabling stage 2.
        assert (with_lof["is_global_anomaly"] == without_lof["is_global_anomaly"]).all()
        assert (without_lof["is_anomaly"] == without_lof["is_global_anomaly"]).all()

    def test_too_few_in_cluster_agents_skips_lof_without_raising(self):
        """A degenerate case (e.g. exactly one non-noise cluster point)
        must not crash LOF — it should just skip stage 2 for that run."""
        from extrafloat_segmentation_pipeline import flag_anomalies, DEFAULT_CLUSTERING_CONFIG

        feat_df, sel_cols, active_mask = self._get_small_inputs(n=10)
        cfg = dict(DEFAULT_CLUSTERING_CONFIG)
        cfg["hdbscan_min_cluster_size"] = 2
        cfg["hdbscan_min_samples"] = 1
        result = flag_anomalies(feat_df, sel_cols, active_mask, config=cfg)  # must not raise
        assert "is_local_anomaly" in result.columns

    def test_missing_hdbscan_degrades_all_two_stage_columns(self, monkeypatch):
        import extrafloat_segmentation_pipeline as pipe

        monkeypatch.setattr(pipe, "_HDBSCAN_AVAILABLE", False)
        feat_df, sel_cols, active_mask, cfg = self._get_inputs_with_real_clusters()
        result = pipe.flag_anomalies(feat_df, sel_cols, active_mask, config=cfg)  # must not raise
        assert not result["is_anomaly"].any()
        assert not result["is_global_anomaly"].any()
        assert not result["is_local_anomaly"].any()
        assert result["lof_score"].isna().all()


class TestDiagnosticClustering:
    """Tests for extrafloat_segmentation_pipeline.run_diagnostic_clustering —
    GMM/HDBSCAN diagnostics only. Never assigns a business tier; that is
    extrafloat_segmentation_scoring.py's job."""

    def _get_small_inputs(self, n=80):
        from extrafloat_segmentation_features import prepare_features

        df = _make_agents_df(n)
        feat_df, _, X_pca, sel_cols = prepare_features(df)
        return feat_df, X_pca, sel_cols

    def test_output_has_required_columns(self):
        from extrafloat_segmentation_pipeline import run_diagnostic_clustering

        feat_df, X_pca, sel_cols = self._get_small_inputs()
        result = run_diagnostic_clustering(feat_df, X_pca, sel_cols)

        for col in ("diag_cluster_id_gmm", "diag_ensemble_cluster", "diag_hdb_tier", "diag_is_anomaly"):
            assert col in result.columns, f"Missing column: {col}"

    def test_no_business_tier_columns_produced(self):
        """Diagnostics must never produce a segment/tier/capacity_tier column itself."""
        from extrafloat_segmentation_pipeline import run_diagnostic_clustering

        feat_df, X_pca, sel_cols = self._get_small_inputs()
        result = run_diagnostic_clustering(feat_df, X_pca, sel_cols)

        for col in ("segment", "tier", "capacity_tier"):
            assert col not in result.columns, f"Diagnostics unexpectedly produced '{col}'"

    def test_dormant_agents_get_dormant_fill_label(self):
        from extrafloat_segmentation_pipeline import DORMANT_FILL_LABEL, run_diagnostic_clustering

        feat_df, X_pca, sel_cols = self._get_small_inputs(80)
        # Zero out all multi-product inactivity cols so composite score = 0
        feat_df = feat_df.copy()
        for col in ("cash_out_vol_1m", "cash_in_vol_1m", "payment_vol_1m", "voucher_volume_1m"):
            if col in feat_df.columns:
                feat_df.loc[feat_df.index[:10], col] = 0.0

        result = run_diagnostic_clustering(feat_df, X_pca, sel_cols)
        dormant_ensembles = result.iloc[:10]["diag_ensemble_cluster"].unique()

        for val in dormant_ensembles:
            assert val == DORMANT_FILL_LABEL, (
                f"Dormant agent got diag_ensemble_cluster '{val}' instead of '{DORMANT_FILL_LABEL}'"
            )

    def test_row_count_preserved(self):
        from extrafloat_segmentation_pipeline import run_diagnostic_clustering

        feat_df, X_pca, sel_cols = self._get_small_inputs()
        result = run_diagnostic_clustering(feat_df, X_pca, sel_cols)
        assert len(result) == len(feat_df)

    def test_mismatched_shapes_raise(self):
        from extrafloat_segmentation_pipeline import run_diagnostic_clustering

        feat_df, X_pca, sel_cols = self._get_small_inputs(60)
        X_wrong = X_pca[: len(X_pca) // 2]
        with pytest.raises(ValueError):
            run_diagnostic_clustering(feat_df, X_wrong, sel_cols)

    def test_is_anomaly_matches_hdb_noise_label(self):
        from extrafloat_segmentation_pipeline import HDBSCAN_NOISE_LABEL, run_diagnostic_clustering

        feat_df, X_pca, sel_cols = self._get_small_inputs()
        result = run_diagnostic_clustering(feat_df, X_pca, sel_cols)
        expected = result["diag_cluster_hdb_raw"] == HDBSCAN_NOISE_LABEL
        assert (result["diag_is_anomaly"] == expected).all()

    def test_diag_degraded_mode_in_attrs(self):
        from extrafloat_segmentation_pipeline import run_diagnostic_clustering

        feat_df, X_pca, sel_cols = self._get_small_inputs()
        result = run_diagnostic_clustering(feat_df, X_pca, sel_cols)
        assert "diag_degraded_mode" in result.attrs
        for key in ("hdbscan_unavailable", "umap_unavailable"):
            assert key in result.attrs["diag_degraded_mode"]

    def test_hdb_tier_ranking_uses_business_kpi(self):
        """_map_hdb_to_tier should rank by total_value_1m not PC1 when available."""
        import numpy as np
        from extrafloat_segmentation_pipeline import _map_hdb_to_tier

        n = 50
        hdb_labels = np.array([0] * 25 + [1] * 25)
        features_df_active = _make_agents_df(n)
        features_df_active = features_df_active.reset_index(drop=True)
        features_df_active["total_value_1m"] = (
            [100.0] * 25 + [10000.0] * 25
        )
        cfg = {"hdb_tier_ranking_cols": ["total_value_1m", "cash_out_value_1m", "commission"]}
        tiers = _map_hdb_to_tier(hdb_labels, features_df_active, cfg)
        # cluster 0 (low value) should get a lower diagnostic label than cluster 1 (high value)
        tier_0 = tiers.iloc[0]
        tier_1 = tiers.iloc[25]
        assert tier_0 != tier_1, "Both clusters got the same diagnostic label"

    def test_umap_fallback_when_not_available(self):
        """_get_active_umap returns array of correct shape even without umap."""
        import numpy as np
        from extrafloat_segmentation_pipeline import _get_active_umap

        rng = np.random.RandomState(42)
        X = rng.randn(40, 10)
        cfg = {"umap_n_components": 2, "umap_n_neighbors": 5, "umap_min_dist": 0.1}
        result = _get_active_umap(X, cfg, rng)
        assert result.shape[0] == 40
        assert result.shape[1] == 2


class TestDiagnosticEnsembleStability:
    """Tests for extrafloat_segmentation_pipeline.compute_diagnostic_ensemble_stability."""

    def _get_small_inputs(self, n=80):
        from extrafloat_segmentation_features import prepare_features
        from extrafloat_segmentation_pipeline import _identify_dormant_mask, DEFAULT_CLUSTERING_CONFIG

        df = _make_agents_df(n)
        feat_df, _, X_pca, sel_cols = prepare_features(df)
        active_mask = ~_identify_dormant_mask(feat_df, DEFAULT_CLUSTERING_CONFIG)
        return feat_df, active_mask, sel_cols

    def test_returns_expected_keys(self):
        from extrafloat_segmentation_pipeline import compute_diagnostic_ensemble_stability

        feat_df, active_mask, sel_cols = self._get_small_inputs()
        report = compute_diagnostic_ensemble_stability(feat_df, active_mask, sel_cols, n_seeds=2)
        for key in ("ensemble_consistency_rate", "ensemble_ari_mean", "ensemble_ari_std", "n_seeds", "n_active"):
            assert key in report, f"report missing key '{key}'"
        assert report["n_seeds"] == 2

    def test_default_report_when_n_seeds_below_two(self):
        from extrafloat_segmentation_pipeline import (
            compute_diagnostic_ensemble_stability,
            DEFAULT_DIAGNOSTIC_STABILITY_REPORT,
        )

        feat_df, active_mask, sel_cols = self._get_small_inputs()
        report = compute_diagnostic_ensemble_stability(feat_df, active_mask, sel_cols, n_seeds=1)
        assert report == DEFAULT_DIAGNOSTIC_STABILITY_REPORT

    def test_default_report_when_no_active_agents(self):
        import pandas as pd
        from extrafloat_segmentation_pipeline import (
            compute_diagnostic_ensemble_stability,
            DEFAULT_DIAGNOSTIC_STABILITY_REPORT,
        )

        feat_df, _, sel_cols = self._get_small_inputs()
        no_active = pd.Series(False, index=feat_df.index)
        report = compute_diagnostic_ensemble_stability(feat_df, no_active, sel_cols, n_seeds=2)
        assert report == DEFAULT_DIAGNOSTIC_STABILITY_REPORT


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

        for section in ("data", "features", "scoring", "clustering", "profiling", "output", "drift"):
            assert section in DEFAULT_SEGMENTATION_CONFIG, f"Missing section: {section}"

    def test_config_round_trip(self):
        from run_extrafloat_segmentation import _get_config

        cfg = _get_config({"scoring": {"allow_missing_scorecard": True}})
        for section in ("data", "features", "scoring", "clustering", "profiling", "output", "drift"):
            assert section in cfg

    def test_scorecard_required_by_default(self):
        """A run with no scorecard_path and no allow_missing_scorecard override
        must fail fast, since capacity_tier is this pipeline's sole output."""
        from run_extrafloat_segmentation import DEFAULT_SEGMENTATION_CONFIG

        assert DEFAULT_SEGMENTATION_CONFIG["scoring"]["scorecard_path"] == ""
        assert DEFAULT_SEGMENTATION_CONFIG["scoring"]["allow_missing_scorecard"] is False

    def test_partial_override_preserved(self):
        from run_extrafloat_segmentation import _get_config

        cfg = _get_config({"features": {"corr_threshold": 0.85}})
        assert cfg["features"]["corr_threshold"] == 0.85
        # Other feature keys should still be present
        assert "skew_threshold" in cfg["features"]


class TestPipelineConstants:
    """Verify critical module constants are correctly defined.

    BUSINESS_SEGMENTS is owned by extrafloat_segmentation_scoring.py — the
    only module that assigns a business tier now.
    extrafloat_segmentation_pipeline.py is diagnostics-only and has no
    business tier concept of its own.
    """

    def test_business_segments_length(self):
        from extrafloat_segmentation_scoring import BUSINESS_SEGMENTS

        assert len(BUSINESS_SEGMENTS) == 8

    def test_business_segments_first_is_below_threshold(self):
        from extrafloat_segmentation_scoring import BUSINESS_SEGMENTS

        assert BUSINESS_SEGMENTS[0] == "Below Threshold"

    def test_business_segments_last_is_diamond(self):
        from extrafloat_segmentation_scoring import BUSINESS_SEGMENTS

        assert BUSINESS_SEGMENTS[-1] == "Diamond"

    def test_pipeline_module_has_no_business_segments(self):
        import extrafloat_segmentation_pipeline as pipeline_mod

        assert not hasattr(pipeline_mod, "BUSINESS_SEGMENTS")

    def test_dormant_fill_label_constant(self):
        from extrafloat_segmentation_pipeline import DORMANT_FILL_LABEL

        assert isinstance(DORMANT_FILL_LABEL, str)
        assert len(DORMANT_FILL_LABEL) > 0


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT MODULE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestDriftModule:
    """Tests for extrafloat_segmentation_drift."""

    def _make_dist(self, n: int, seed: int, scale: float = 1.0) -> np.ndarray:
        return np.random.RandomState(seed).exponential(scale=scale, size=n)

    # ── compute_psi ──────────────────────────────────────────────────────────

    def test_psi_zero_for_identical(self):
        from extrafloat_segmentation_drift import compute_psi

        vals = self._make_dist(500, seed=0)
        psi = compute_psi(vals, vals.copy())
        assert psi < 0.01, f"PSI should be near 0 for identical distributions, got {psi:.4f}"

    def test_psi_high_for_very_different(self):
        from extrafloat_segmentation_drift import compute_psi

        expected = self._make_dist(1000, seed=0, scale=1.0)
        actual = self._make_dist(1000, seed=1, scale=100.0)  # 100× larger scale
        psi = compute_psi(expected, actual)
        assert psi > 0.25, f"PSI should be critical (>0.25) for very different distributions, got {psi:.4f}"

    def test_psi_non_negative(self):
        from extrafloat_segmentation_drift import compute_psi

        rng = np.random.RandomState(42)
        for _ in range(10):
            a = rng.randn(200)
            b = rng.randn(200) + rng.uniform(0, 3)
            assert compute_psi(a, b) >= 0.0

    def test_psi_empty_array_returns_nan(self):
        from extrafloat_segmentation_drift import compute_psi

        assert np.isnan(compute_psi(np.array([]), np.array([1.0, 2.0])))
        assert np.isnan(compute_psi(np.array([1.0, 2.0]), np.array([])))

    def test_psi_constant_feature_returns_zero(self):
        from extrafloat_segmentation_drift import compute_psi

        psi = compute_psi(np.full(100, 5.0), np.full(100, 5.0))
        assert psi == 0.0

    # ── compute_kl_divergence ────────────────────────────────────────────────

    def test_kl_zero_for_identical(self):
        from extrafloat_segmentation_drift import compute_kl_divergence

        vals = self._make_dist(500, seed=0)
        kl = compute_kl_divergence(vals, vals.copy())
        assert kl < 0.01, f"KL should be near 0 for identical distributions, got {kl:.4f}"

    def test_kl_non_negative(self):
        from extrafloat_segmentation_drift import compute_kl_divergence

        expected = self._make_dist(200, seed=0)
        actual = self._make_dist(200, seed=1, scale=2.0)
        assert compute_kl_divergence(expected, actual) >= 0.0

    # ── build_drift_report ───────────────────────────────────────────────────

    def test_drift_report_structure(self):
        from extrafloat_segmentation_drift import build_drift_report

        rng = np.random.RandomState(7)
        baseline_df = pd.DataFrame({
            "commission": rng.exponential(100, 300),
            "cash_out_value_1m": rng.exponential(5000, 300),
        })
        current_df = pd.DataFrame({
            "commission": rng.exponential(100, 280),
            "cash_out_value_1m": rng.exponential(5000, 280),
        })
        report = build_drift_report(
            baseline_df, current_df,
            features=["commission", "cash_out_value_1m"],
            config={"n_bins": 5, "psi_warn_threshold": 0.10, "psi_critical_threshold": 0.25},
        )
        for key in ("features", "overall_psi", "n_features_checked", "n_critical", "n_warning", "drift_detected", "baseline_n", "current_n"):
            assert key in report, f"drift report missing key '{key}'"

    def test_drift_report_stable_for_same_data(self):
        from extrafloat_segmentation_drift import build_drift_report

        rng = np.random.RandomState(9)
        df = pd.DataFrame({"commission": rng.exponential(100, 400)})
        report = build_drift_report(df, df.copy(), features=["commission"])
        assert report["features"]["commission"]["psi"] < 0.10
        assert report["drift_detected"] is False

    def test_drift_report_critical_for_shifted_data(self):
        from extrafloat_segmentation_drift import build_drift_report

        rng = np.random.RandomState(3)
        baseline = pd.DataFrame({"commission": rng.exponential(100, 500)})
        current = pd.DataFrame({"commission": rng.exponential(5000, 500)})
        report = build_drift_report(
            baseline, current, features=["commission"],
            config={"psi_warn_threshold": 0.10, "psi_critical_threshold": 0.25},
        )
        assert report["n_critical"] >= 1
        assert report["drift_detected"] is True

    def test_drift_report_skips_missing_features(self):
        from extrafloat_segmentation_drift import build_drift_report

        rng = np.random.RandomState(5)
        df = pd.DataFrame({"commission": rng.exponential(100, 100)})
        # Request a feature not in either df
        report = build_drift_report(df, df.copy(), features=["commission", "nonexistent_col"])
        assert report["n_features_checked"] == 1
        assert "nonexistent_col" not in report["features"]

    def test_drift_report_baseline_current_n(self):
        from extrafloat_segmentation_drift import build_drift_report

        rng = np.random.RandomState(11)
        b = pd.DataFrame({"commission": rng.randn(150)})
        c = pd.DataFrame({"commission": rng.randn(120)})
        report = build_drift_report(b, c, features=["commission"])
        assert report["baseline_n"] == 150
        assert report["current_n"] == 120

    # ── save/load baseline ───────────────────────────────────────────────────

    def test_save_load_baseline_roundtrip(self, tmp_path):
        from extrafloat_segmentation_drift import load_drift_baseline, save_drift_baseline

        rng = np.random.RandomState(0)
        df = pd.DataFrame({
            "commission": rng.exponential(100, 200),
            "cash_out_value_1m": rng.exponential(5000, 200),
        })
        save_path = str(tmp_path / "baseline.csv")
        cfg = {"baseline_save_path": save_path}
        save_drift_baseline(df, ["commission", "cash_out_value_1m"], config=cfg)

        loaded = load_drift_baseline(save_path)
        assert list(loaded.columns) == ["commission", "cash_out_value_1m"]
        assert len(loaded) == 200

    def test_save_baseline_raises_without_path(self):
        from extrafloat_segmentation_drift import save_drift_baseline

        df = pd.DataFrame({"commission": [1.0, 2.0]})
        with pytest.raises(ValueError, match="baseline_save_path"):
            save_drift_baseline(df, ["commission"], config={})

    def test_load_baseline_raises_for_missing_file(self):
        from extrafloat_segmentation_drift import load_drift_baseline

        with pytest.raises(FileNotFoundError):
            load_drift_baseline("/nonexistent/path/baseline.csv")


class TestCategoricalDrift:
    """Tests for extrafloat_segmentation_drift.compute_categorical_psi /
    build_tier_drift_report — monitors capacity_tier proportions against a
    scorecard's frozen calibration expectations."""

    def test_identical_distributions_have_zero_psi(self):
        from extrafloat_segmentation_drift import compute_categorical_psi

        labels = pd.Series(["A"] * 50 + ["B"] * 50)
        expected = {"A": 0.5, "B": 0.5}
        psi = compute_categorical_psi(expected, labels, ["A", "B"])
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distribution_has_positive_psi(self):
        from extrafloat_segmentation_drift import compute_categorical_psi

        labels = pd.Series(["A"] * 90 + ["B"] * 10)
        expected = {"A": 0.5, "B": 0.5}
        psi = compute_categorical_psi(expected, labels, ["A", "B"])
        assert psi > 0.0

    def test_missing_category_defaults_to_zero_expected(self):
        from extrafloat_segmentation_drift import compute_categorical_psi

        labels = pd.Series(["A"] * 50 + ["B"] * 50)
        expected = {"A": 1.0}  # "B" absent -> expected 0.0
        psi = compute_categorical_psi(expected, labels, ["A", "B"])
        assert psi > 0.0

    def test_empty_labels_return_nan(self):
        from extrafloat_segmentation_drift import compute_categorical_psi

        labels = pd.Series([], dtype="object")
        psi = compute_categorical_psi({"A": 1.0}, labels, ["A"])
        assert np.isnan(psi)

    def test_nan_labels_dropped_before_computing(self):
        from extrafloat_segmentation_drift import compute_categorical_psi

        labels = pd.Series(["A"] * 50 + ["B"] * 50 + [np.nan] * 20)
        expected = {"A": 0.5, "B": 0.5}
        psi = compute_categorical_psi(expected, labels, ["A", "B"])
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_build_tier_drift_report_structure(self):
        from extrafloat_segmentation_drift import build_tier_drift_report

        labels = pd.Series(["A"] * 50 + ["B"] * 50)
        expected = {"A": 0.5, "B": 0.5}
        report = build_tier_drift_report(expected, labels, ["A", "B"])
        for key in ("psi", "status", "categories", "expected_proportions",
                    "actual_proportions", "n_actual", "drift_detected"):
            assert key in report
        assert report["n_actual"] == 100
        assert report["drift_detected"] is False

    def test_build_tier_drift_report_flags_critical_shift(self):
        from extrafloat_segmentation_drift import build_tier_drift_report

        labels = pd.Series(["A"] * 99 + ["B"])
        expected = {"A": 0.1, "B": 0.9}
        report = build_tier_drift_report(
            expected, labels, ["A", "B"],
            config={"psi_warn_threshold": 0.10, "psi_critical_threshold": 0.25},
        )
        assert report["status"] == "critical"
        assert report["drift_detected"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Optional dependency guard rails (require_hdbscan / require_umap)
# ─────────────────────────────────────────────────────────────────────────────


class TestOptionalDependencyGuards:
    """Tests for extrafloat_segmentation_pipeline._check_optional_dependencies
    and the require_hdbscan/require_umap config flags wired into
    run_diagnostic_clustering. A missing hdbscan/umap install used to degrade
    cluster quality silently (hdb_tier="Unavailable" for every active agent,
    or a raw column slice fed to HDBSCAN instead of a real embedding) with
    only a log line marking the difference. These tests confirm that is a
    loud, opt-in failure mode instead of a silent one — and, since this
    module is diagnostics-only now, that it can only ever block a
    deliberately-requested diagnostics run, never production capacity
    scoring (which does not import this module's public function at all).
    """

    def _small_inputs(self, n=50):
        rng = np.random.RandomState(0)
        X_pca = rng.randn(n, 5)
        df = pd.DataFrame({
            "cash_out_vol_1m": rng.rand(n) * 10,
            "cash_in_vol_1m": rng.rand(n) * 10,
            "payment_vol_1m": rng.rand(n) * 10,
            "voucher_volume_1m": rng.rand(n) * 10,
        })
        return df, X_pca, list(df.columns)

    def test_defaults_require_both(self):
        from extrafloat_segmentation_pipeline import DEFAULT_CLUSTERING_CONFIG

        assert DEFAULT_CLUSTERING_CONFIG["require_hdbscan"] is True
        assert DEFAULT_CLUSTERING_CONFIG["require_umap"] is True

    def test_diagnostics_disabled_by_default(self):
        from extrafloat_segmentation_pipeline import DEFAULT_CLUSTERING_CONFIG

        assert DEFAULT_CLUSTERING_CONFIG["enable_diagnostics"] is False

    def test_missing_hdbscan_raises_by_default(self, monkeypatch):
        import extrafloat_segmentation_pipeline as pipe

        monkeypatch.setattr(pipe, "_HDBSCAN_AVAILABLE", False)
        df, X_pca, cols = self._small_inputs()
        with pytest.raises(RuntimeError, match="require_hdbscan"):
            pipe.run_diagnostic_clustering(df, X_pca, cols)

    def test_missing_umap_raises_by_default(self, monkeypatch):
        import extrafloat_segmentation_pipeline as pipe

        monkeypatch.setattr(pipe, "_UMAP_AVAILABLE", False)
        df, X_pca, cols = self._small_inputs()
        with pytest.raises(RuntimeError, match="require_umap"):
            pipe.run_diagnostic_clustering(df, X_pca, cols)

    def test_missing_hdbscan_allowed_when_not_required(self, monkeypatch):
        import extrafloat_segmentation_pipeline as pipe

        monkeypatch.setattr(pipe, "_HDBSCAN_AVAILABLE", False)
        df, X_pca, cols = self._small_inputs()
        result = pipe.run_diagnostic_clustering(
            df, X_pca, cols, config={"require_hdbscan": False},
        )
        assert result.attrs["diag_degraded_mode"]["hdbscan_unavailable"] is True

    def test_degraded_mode_false_when_all_available(self):
        from extrafloat_segmentation_pipeline import run_diagnostic_clustering

        df, X_pca, cols = self._small_inputs()
        result = run_diagnostic_clustering(df, X_pca, cols)
        assert result.attrs["diag_degraded_mode"] == {
            "hdbscan_unavailable": False,
            "umap_unavailable": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quality gate (extrafloat_segmentation_validation.run_quality_gate)
# ─────────────────────────────────────────────────────────────────────────────


class TestQualityGate:
    """Tests for run_quality_gate — the automated post-clustering sanity
    check that was previously missing from the main orchestration entirely
    (the validation module existed but nothing called it)."""

    def _clean_df(self, n=100):
        """Agents with perfect agreement between ensemble_cluster and
        agent_category, no HDBSCAN noise, and a healthy segment spread."""
        rng = np.random.RandomState(0)
        segments = (["Silver"] * (n // 2)) + (["Gold"] * (n - n // 2))
        return pd.DataFrame({
            "segment": segments,
            "ensemble_cluster": segments,  # cluster == category -> perfect purity/ARI
            "agent_category": segments,
            "hdb_tier": ["Silver Strong"] * (n // 2) + ["Gold Power"] * (n - n // 2),
            "cluster_round2": np.arange(n),  # all "active" (non-NaN)
        })

    def test_clean_data_passes(self):
        from extrafloat_segmentation_validation import run_quality_gate

        report = run_quality_gate(self._clean_df())
        assert report["passed"] is True
        assert report["has_ground_truth"] is True
        assert report["checks"]["overall_purity"]["value"] == pytest.approx(1.0)
        assert report["checks"]["ari"]["value"] == pytest.approx(1.0)

    def test_all_noise_fails_hdb_check(self):
        from extrafloat_segmentation_validation import run_quality_gate

        df = self._clean_df()
        df["hdb_tier"] = "Noise / Irregular"
        report = run_quality_gate(df)
        assert report["passed"] is False
        assert report["checks"]["hdb_noise_or_unavailable_pct"]["passed"] is False

    def test_single_segment_fails_distinct_check(self):
        from extrafloat_segmentation_validation import run_quality_gate

        df = self._clean_df()
        df["segment"] = "Below Threshold"
        report = run_quality_gate(df)
        assert report["checks"]["n_distinct_segments"]["passed"] is False

    def test_disabled_via_config_is_skipped(self):
        from extrafloat_segmentation_validation import run_quality_gate

        report = run_quality_gate(self._clean_df(), config={"enabled": False})
        assert report["skipped"] is True
        assert report["passed"] is True
        assert report["checks"] == {}

    def test_fail_on_breach_raises(self):
        from extrafloat_segmentation_validation import run_quality_gate

        df = self._clean_df()
        df["hdb_tier"] = "Noise / Irregular"
        with pytest.raises(ValueError, match="quality gate FAILED"):
            run_quality_gate(df, config={"fail_on_breach": True})

    def test_no_ground_truth_skips_supervised_checks(self):
        from extrafloat_segmentation_validation import run_quality_gate

        df = self._clean_df().drop(columns=["agent_category"])
        report = run_quality_gate(df)
        assert report["has_ground_truth"] is False
        assert "overall_purity" not in report["checks"]
        assert "ari" not in report["checks"]

    def test_missing_segment_col_raises(self):
        from extrafloat_segmentation_validation import run_quality_gate

        df = self._clean_df().drop(columns=["segment"])
        with pytest.raises(ValueError, match="Missing required column"):
            run_quality_gate(df)


# ─────────────────────────────────────────────────────────────────────────────
# Run manifest + structured alerts (run_extrafloat_segmentation.py)
# ─────────────────────────────────────────────────────────────────────────────


class TestCollectAlerts:
    """Tests for run_extrafloat_segmentation._collect_alerts — rolls the
    diag_degraded_mode/quality_gate/drift/tier_drift/diagnostic_stability
    .attrs reports into a flat, structured list any caller can route
    without knowing each report's shape."""

    def _base_df(self):
        return pd.DataFrame({"capacity_tier": ["Gold", "Silver"]})

    def test_no_attrs_no_alerts(self):
        from run_extrafloat_segmentation import _collect_alerts

        df = self._base_df()
        assert _collect_alerts(df) == []

    def test_diag_degraded_mode_produces_warning_alert(self):
        from run_extrafloat_segmentation import _collect_alerts

        df = self._base_df()
        df.attrs["diag_degraded_mode"] = {"hdbscan_unavailable": True, "umap_unavailable": False}
        alerts = _collect_alerts(df)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["source"] == "diag_degraded_mode"

    def test_anomaly_detection_missing_hdbscan_produces_warning_alert(self):
        from run_extrafloat_segmentation import _collect_alerts

        df = self._base_df()
        df.attrs["anomaly_report"] = {
            "hdbscan_available": False, "n_active": 10, "n_anomalies": 0, "anomaly_rate": 0.0,
        }
        alerts = _collect_alerts(df)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["source"] == "anomaly_detection"

    def test_anomaly_detection_with_hdbscan_available_produces_no_alert(self):
        from run_extrafloat_segmentation import _collect_alerts

        df = self._base_df()
        df.attrs["anomaly_report"] = {
            "hdbscan_available": True, "n_active": 10, "n_anomalies": 2, "anomaly_rate": 0.2,
        }
        assert _collect_alerts(df) == []

    def test_failed_quality_gate_produces_critical_alert(self):
        from run_extrafloat_segmentation import _collect_alerts

        df = self._base_df()
        df.attrs["quality_gate"] = {
            "passed": False,
            "skipped": False,
            "checks": {"n_distinct_segments": {"value": 1, "threshold": 2, "passed": False}},
        }
        alerts = _collect_alerts(df)
        assert len(alerts) == 1
        assert alerts[0]["source"] == "quality_gate"

    def test_skipped_quality_gate_produces_no_alert(self):
        from run_extrafloat_segmentation import _collect_alerts

        df = self._base_df()
        df.attrs["quality_gate"] = {"passed": True, "skipped": True, "checks": {}}
        assert _collect_alerts(df) == []

    def test_drift_detected_produces_warning_alert(self):
        from run_extrafloat_segmentation import _collect_alerts

        df = self._base_df()
        df.attrs["drift_report"] = {"drift_detected": True, "n_critical": 2, "n_warning": 1}
        alerts = _collect_alerts(df)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["source"] == "drift"

    def test_low_diagnostic_consistency_produces_warning_alert(self):
        from run_extrafloat_segmentation import _collect_alerts

        df = self._base_df()
        df.attrs["diagnostic_stability_report"] = {
            "ensemble_consistency_rate": 0.2,
            "n_seeds": 3,
        }
        alerts = _collect_alerts(df)
        assert len(alerts) == 1
        assert alerts[0]["source"] == "diagnostic_stability"

    def test_high_diagnostic_consistency_produces_no_alert(self):
        from run_extrafloat_segmentation import _collect_alerts

        df = self._base_df()
        df.attrs["diagnostic_stability_report"] = {
            "ensemble_consistency_rate": 0.95,
            "n_seeds": 3,
        }
        assert _collect_alerts(df) == []

    def test_tier_drift_detected_produces_warning_alert(self):
        from run_extrafloat_segmentation import _collect_alerts

        df = self._base_df()
        df.attrs["tier_drift_report"] = {"drift_detected": True, "psi": 0.9, "status": "critical"}
        alerts = _collect_alerts(df)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["source"] == "tier_drift"

    def test_tier_drift_stable_produces_no_alert(self):
        from run_extrafloat_segmentation import _collect_alerts

        df = self._base_df()
        df.attrs["tier_drift_report"] = {"drift_detected": False, "psi": 0.01, "status": "stable"}
        assert _collect_alerts(df) == []


class TestRunManifest:
    """Tests for _build_run_manifest / _maybe_save_run_manifest /
    _collect_package_versions / _get_git_commit_sha."""

    def test_collect_package_versions_returns_expected_keys(self):
        from run_extrafloat_segmentation import _collect_package_versions

        versions = _collect_package_versions()
        for pkg in ("numpy", "pandas", "scikit-learn", "hdbscan", "umap-learn"):
            assert pkg in versions
            assert isinstance(versions[pkg], str)

    def test_get_git_commit_sha_returns_str_or_none(self):
        from run_extrafloat_segmentation import _get_git_commit_sha

        sha = _get_git_commit_sha()
        assert sha is None or (isinstance(sha, str) and len(sha) == 40)

    def test_build_run_manifest_structure(self):
        from run_extrafloat_segmentation import _build_run_manifest, DEFAULT_SEGMENTATION_CONFIG

        df = pd.DataFrame({"capacity_tier": ["Gold", "Silver", "Gold"]})
        df.attrs["quality_gate"] = {"passed": True, "skipped": False, "checks": {}}
        manifest = _build_run_manifest(df, DEFAULT_SEGMENTATION_CONFIG, n_input_rows=3)

        for key in (
            "generated_at_utc", "git_commit_sha", "package_versions", "config",
            "n_input_rows", "n_output_rows", "n_distinct_tiers",
            "drift_report", "tier_drift_report", "anomaly_report",
            "diagnostic_stability_report", "quality_gate", "diag_degraded_mode", "alerts",
        ):
            assert key in manifest, f"manifest missing key '{key}'"
        assert manifest["n_input_rows"] == 3
        assert manifest["n_output_rows"] == 3
        assert manifest["n_distinct_tiers"] == 2

    def test_maybe_save_run_manifest_writes_json(self, tmp_path):
        from run_extrafloat_segmentation import _maybe_save_run_manifest
        import json

        df = pd.DataFrame({"capacity_tier": ["Gold", "Silver"]})
        cfg = {"output": {"output_dir": str(tmp_path), "save_run_manifest": True, "primary_tier_col": "capacity_tier"}}
        _maybe_save_run_manifest(df, cfg, n_input_rows=2)

        manifest_path = tmp_path / "run_manifest.json"
        assert manifest_path.exists()
        loaded = json.loads(manifest_path.read_text())
        assert loaded["n_input_rows"] == 2

    def test_maybe_save_run_manifest_skips_when_disabled(self, tmp_path):
        from run_extrafloat_segmentation import _maybe_save_run_manifest

        df = pd.DataFrame({"capacity_tier": ["Gold"]})
        cfg = {"output": {"output_dir": str(tmp_path), "save_run_manifest": False, "primary_tier_col": "capacity_tier"}}
        _maybe_save_run_manifest(df, cfg, n_input_rows=1)

        assert not (tmp_path / "run_manifest.json").exists()

    def test_maybe_save_run_manifest_skips_without_output_dir(self, tmp_path):
        from run_extrafloat_segmentation import _maybe_save_run_manifest

        df = pd.DataFrame({"capacity_tier": ["Gold"]})
        cfg = {"output": {"output_dir": "", "save_run_manifest": True, "primary_tier_col": "capacity_tier"}}
        _maybe_save_run_manifest(df, cfg, n_input_rows=1)

        assert not (tmp_path / "run_manifest.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic capacity scoring orchestration (opt-in, shadow-run alongside
# the existing ensemble-cluster segment/tier — see DEFAULT_SEGMENTATION_CONFIG
# ["scoring"]).
# ─────────────────────────────────────────────────────────────────────────────


class TestScoringOrchestration:
    """Tests for the run_extrafloat_segmentation capacity-scoring wiring:
    a scorecard is REQUIRED by default (fails fast without one), diagnostic
    clustering is opt-in and independent, and capacity_tier is deterministic
    regardless of which other agents are in the run."""

    def _small_agents_df(self, n=80):
        return _make_agents_df(n)

    @staticmethod
    def _no_output_cfg(**overrides) -> dict:
        """Config dict with output_dir disabled, so these tests don't write
        agent_segments.csv / run_manifest.json into the repo working tree."""
        cfg = {"output": {"output_dir": ""}}
        cfg.update(overrides)
        return cfg

    def _calibrated_scorecard_path(self, agents_df, tmp_path, **kwargs):
        from extrafloat_segmentation_features import prepare_features
        from extrafloat_segmentation_scoring import calibrate_capacity_scorecard, save_scorecard

        dev_df, _, _, _ = prepare_features(agents_df)
        scorecard = calibrate_capacity_scorecard(dev_df, population_description="test fixture", **kwargs)
        scorecard_path = str(tmp_path / "scorecard.json")
        save_scorecard(scorecard, scorecard_path)
        return scorecard_path, scorecard

    @staticmethod
    def _scoring_cfg(scorecard_path, **overrides) -> dict:
        """scoring config pointed at a calibrated (provisional-by-default)
        scorecard, with allow_provisional_scorecard explicitly opted in —
        these tests exercise the scoring mechanism itself, not the
        provisional-scorecard guard (see test_provisional_scorecard_*)."""
        cfg = {"scorecard_path": scorecard_path, "allow_provisional_scorecard": True}
        cfg.update(overrides)
        return cfg

    def test_default_config_requires_scorecard(self):
        from run_extrafloat_segmentation import DEFAULT_SEGMENTATION_CONFIG

        assert "scoring" in DEFAULT_SEGMENTATION_CONFIG
        assert DEFAULT_SEGMENTATION_CONFIG["scoring"]["scorecard_path"] == ""
        assert DEFAULT_SEGMENTATION_CONFIG["scoring"]["allow_missing_scorecard"] is False

    def test_no_scorecard_configured_raises_by_default(self):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        with pytest.raises(ValueError, match="scorecard"):
            run_extrafloat_segmentation(agents_df, config=self._no_output_cfg())

    def test_no_scorecard_with_allow_missing_runs_without_capacity_tier(self):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        cfg = self._no_output_cfg(scoring={"allow_missing_scorecard": True})
        result = run_extrafloat_segmentation(agents_df, config=cfg)  # should not raise
        for col in ("capacity_score", "capacity_tier", "capacity_tier_raw"):
            assert col not in result.columns

    def test_missing_scorecard_file_with_hard_fail_raises(self):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        cfg = self._no_output_cfg(
            scoring={"scorecard_path": "/nonexistent/scorecard.json", "allow_missing_scorecard": False}
        )
        with pytest.raises(ValueError):
            run_extrafloat_segmentation(agents_df, config=cfg)

    def test_missing_scorecard_file_with_allow_missing_skips_gracefully(self):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        cfg = self._no_output_cfg(
            scoring={"scorecard_path": "/nonexistent/scorecard.json", "allow_missing_scorecard": True}
        )
        result = run_extrafloat_segmentation(agents_df, config=cfg)  # should not raise
        assert "capacity_tier" not in result.columns

    def test_provisional_scorecard_rejected_by_default(self, tmp_path):
        """A provisional scorecard (the default from calibrate_capacity_scorecard)
        must never run silently in what looks like a production config."""
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, scorecard = self._calibrated_scorecard_path(agents_df, tmp_path)
        assert scorecard["calibration_metadata"]["is_provisional"] is True

        cfg = self._no_output_cfg(scoring={"scorecard_path": scorecard_path})
        with pytest.raises(ValueError, match="provisional"):
            run_extrafloat_segmentation(agents_df, config=cfg)

    def test_provisional_scorecard_allowed_with_explicit_opt_in(self, tmp_path):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)

        cfg = self._no_output_cfg(
            scoring={"scorecard_path": scorecard_path, "allow_provisional_scorecard": True}
        )
        result = run_extrafloat_segmentation(agents_df, config=cfg)  # should not raise
        assert "capacity_tier" in result.columns

    def test_non_provisional_scorecard_runs_without_opt_in(self, tmp_path):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, scorecard = self._calibrated_scorecard_path(
            agents_df, tmp_path, is_provisional=False, cutoff_version="reviewed_v1"
        )
        assert scorecard["calibration_metadata"]["is_provisional"] is False

        cfg = self._no_output_cfg(scoring={"scorecard_path": scorecard_path})
        result = run_extrafloat_segmentation(agents_df, config=cfg)  # should not raise
        assert "capacity_tier" in result.columns

    def test_configured_scorecard_produces_capacity_columns(self, tmp_path):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, scorecard = self._calibrated_scorecard_path(agents_df, tmp_path)

        cfg = self._no_output_cfg(scoring=self._scoring_cfg(scorecard_path))
        result = run_extrafloat_segmentation(agents_df, config=cfg)

        for col in ("capacity_score", "capacity_tier_raw", "capacity_tier",
                    "capacity_safety_flags", "scorecard_version", "cutoff_version"):
            assert col in result.columns
        assert result["capacity_tier"].isin(scorecard["tiers"]).all()
        # Legacy aliases present by default (emit_legacy_aliases=True) and
        # equal to capacity_tier — not an independently-computed mechanism.
        assert "segment" in result.columns
        assert "tier" in result.columns
        assert (result["segment"] == result["capacity_tier"]).all()
        assert (result["tier"] == result["capacity_tier"]).all()

    def test_emit_legacy_aliases_false_omits_deprecated_columns(self, tmp_path):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)

        cfg = self._no_output_cfg(
            scoring=self._scoring_cfg(scorecard_path),
            output={"output_dir": "", "emit_legacy_aliases": False},
        )
        result = run_extrafloat_segmentation(agents_df, config=cfg)
        assert "capacity_tier" in result.columns
        assert "segment" not in result.columns
        assert "tier" not in result.columns

    def test_diagnostics_disabled_by_default_no_diag_columns(self, tmp_path):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)

        cfg = self._no_output_cfg(scoring=self._scoring_cfg(scorecard_path))
        result = run_extrafloat_segmentation(agents_df, config=cfg)
        assert not any(c.startswith("diag_") for c in result.columns)

    def test_anomaly_detection_on_by_default_without_diagnostics(self, tmp_path):
        """is_anomaly must appear even with clustering.enable_diagnostics
        left at its default False — it's independent of the full
        diagnostics bundle."""
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)

        cfg = self._no_output_cfg(scoring=self._scoring_cfg(scorecard_path))
        result = run_extrafloat_segmentation(agents_df, config=cfg)
        assert "is_anomaly" in result.columns
        assert "diag_is_anomaly" not in result.columns
        assert "anomaly_report" in result.attrs
        assert result.attrs["anomaly_report"]["hdbscan_available"] is True

    def test_anomaly_detection_disabled_via_config(self, tmp_path):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)

        cfg = self._no_output_cfg(
            scoring=self._scoring_cfg(scorecard_path),
            clustering={"enable_anomaly_detection": False},
        )
        result = run_extrafloat_segmentation(agents_df, config=cfg)
        assert "is_anomaly" not in result.columns
        assert "anomaly_report" not in result.attrs

    def test_anomaly_detection_missing_hdbscan_degrades_and_alerts(self, tmp_path, monkeypatch):
        import extrafloat_segmentation_pipeline as pipe
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)
        cfg = self._no_output_cfg(scoring=self._scoring_cfg(scorecard_path))

        monkeypatch.setattr(pipe, "_HDBSCAN_AVAILABLE", False)
        result = run_extrafloat_segmentation(agents_df, config=cfg)

        assert "is_anomaly" in result.columns
        assert not result["is_anomaly"].any()
        assert result.attrs["anomaly_report"]["hdbscan_available"] is False
        alert_sources = [a["source"] for a in result.attrs.get("alerts", [])]
        assert "anomaly_detection" in alert_sources

    def test_capacity_tier_unaffected_by_anomaly_detection_toggle(self, tmp_path):
        """capacity_tier must be identical whether or not anomaly detection
        ran — it's a pure aside, never an input to scoring."""
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)

        cfg_on = self._no_output_cfg(scoring=self._scoring_cfg(scorecard_path))
        cfg_off = self._no_output_cfg(
            scoring=self._scoring_cfg(scorecard_path),
            clustering={"enable_anomaly_detection": False},
        )
        result_on = run_extrafloat_segmentation(agents_df, config=cfg_on)
        result_off = run_extrafloat_segmentation(agents_df, config=cfg_off)
        assert (
            result_on.set_index("agent_msisdn")["capacity_tier"]
            == result_off.set_index("agent_msisdn")["capacity_tier"]
        ).all()

    def test_drift_report_survives_past_capacity_scoring_concat(self, tmp_path):
        """Regression test: pd.concat used to silently drop .attrs set
        earlier in the pipeline (e.g. drift_report from step 1b) once
        capacity scoring's pd.concat ran, because pandas doesn't propagate
        DataFrame.attrs through concat by default."""
        from run_extrafloat_segmentation import run_extrafloat_segmentation
        from extrafloat_segmentation_features import prepare_features
        from extrafloat_segmentation_drift import save_drift_baseline

        agents_df = self._small_agents_df()
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)

        dev_df, _, _, _ = prepare_features(agents_df)
        baseline_path = str(tmp_path / "baseline.csv")
        drift_features = ["commission", "cash_out_value_1m", "tenure_years"]
        save_drift_baseline(dev_df, features=drift_features, config={"baseline_save_path": baseline_path})

        cfg = self._no_output_cfg(
            scoring=self._scoring_cfg(scorecard_path),
            drift={"baseline_path": baseline_path, "drift_features": drift_features},
        )
        result = run_extrafloat_segmentation(agents_df, config=cfg)
        assert "drift_report" in result.attrs
        assert "overall_psi" in result.attrs["drift_report"]

    def test_diagnostics_enabled_produces_diag_columns(self, tmp_path):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)

        cfg = self._no_output_cfg(
            scoring=self._scoring_cfg(scorecard_path),
            clustering={"enable_diagnostics": True},
        )
        result = run_extrafloat_segmentation(agents_df, config=cfg)
        for col in ("diag_hdb_tier", "diag_ensemble_cluster", "diag_is_anomaly"):
            assert col in result.columns
        # capacity_tier must be identical whether or not diagnostics ran —
        # diagnostics never feeds capacity scoring.
        cfg_no_diag = self._no_output_cfg(scoring=self._scoring_cfg(scorecard_path))
        result_no_diag = run_extrafloat_segmentation(agents_df, config=cfg_no_diag)
        assert (
            result.set_index("agent_msisdn")["capacity_tier"]
            == result_no_diag.set_index("agent_msisdn")["capacity_tier"]
        ).all()

    def test_capacity_tier_deterministic_across_different_populations(self, tmp_path):
        """The whole point: an agent's capacity_tier must not depend on which
        other agents are in the same run, unlike the old ensemble segment/tier."""
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df(80)
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)

        cfg = self._no_output_cfg(scoring=self._scoring_cfg(scorecard_path))
        full_result = run_extrafloat_segmentation(agents_df, config=cfg)

        subset_df = agents_df.iloc[:40].reset_index(drop=True)
        subset_result = run_extrafloat_segmentation(subset_df, config=cfg)

        full_by_msisdn = full_result.set_index("agent_msisdn")["capacity_tier"]
        subset_by_msisdn = subset_result.set_index("agent_msisdn")["capacity_tier"]
        common = full_by_msisdn.index.intersection(subset_by_msisdn.index)
        assert len(common) > 0
        assert (full_by_msisdn.loc[common] == subset_by_msisdn.loc[common]).all()

    def test_tier_drift_report_attached_when_scorecard_used(self, tmp_path):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)

        cfg = self._no_output_cfg(scoring=self._scoring_cfg(scorecard_path))
        result = run_extrafloat_segmentation(agents_df, config=cfg)
        assert "tier_drift_report" in result.attrs
        assert "psi" in result.attrs["tier_drift_report"]

    def test_quality_gate_uses_capacity_tier(self, tmp_path):
        from run_extrafloat_segmentation import run_extrafloat_segmentation

        agents_df = self._small_agents_df()
        scorecard_path, _ = self._calibrated_scorecard_path(agents_df, tmp_path)

        cfg = self._no_output_cfg(scoring=self._scoring_cfg(scorecard_path))
        result = run_extrafloat_segmentation(agents_df, config=cfg)
        quality_gate = result.attrs.get("quality_gate", {})
        assert quality_gate.get("skipped") is False
        assert "n_distinct_segments" in quality_gate.get("checks", {})

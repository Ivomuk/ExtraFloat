"""Tests for pd_model.postprocessing.whitelist_eval."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pd_model.postprocessing.whitelist_eval import (
    _normalize_msisdn,
    cutoff_sweep,
    load_and_merge_lists,
    run_whitelist_blacklist_eval,
)


def _make_list_csvs(tmp_dir: Path, n_wl: int = 100, n_bl: int = 40):
    rng = np.random.default_rng(42)
    wl = pd.DataFrame({
        "agent_msisdn": [f"2547{i:08d}" for i in range(n_wl)],
        "agent_category": "agent",
        "reason": "good_performance",
    })
    bl = pd.DataFrame({
        "agent_msisdn": [f"2547{i + n_wl:08d}" for i in range(n_bl)],
        "agent_category": "agent",
        "reason": "bad_debt",
    })
    wl_path = tmp_dir / "whitelist.csv"
    bl_path = tmp_dir / "blacklist.csv"
    wl.to_csv(wl_path, index=False)
    bl.to_csv(bl_path, index=False)
    return wl_path, bl_path


def _make_scored_thin(wl_path: Path, bl_path: Path, n_extra: int = 50) -> pd.DataFrame:
    wl = pd.read_csv(wl_path)
    bl = pd.read_csv(bl_path)
    all_msisdn = wl["agent_msisdn"].tolist() + bl["agent_msisdn"].tolist()
    rng = np.random.default_rng(7)
    # Assign lower pd_like to whitelist, higher to blacklist (so AUC > 0.5)
    pd_like_wl = rng.uniform(0.0, 0.4, len(wl))
    pd_like_bl = rng.uniform(0.5, 1.0, len(bl))
    pd_like = list(pd_like_wl) + list(pd_like_bl)
    return pd.DataFrame({
        "agent_msisdn": all_msisdn,
        "never_loan_pd_like": pd_like,
        "final_approved": rng.integers(0, 2, len(all_msisdn)),
        "final_policy_bucket": "SCORECARD_APPROVE_50",
    })


class TestNormalizeMsisdn:
    def test_strips_dot_zero(self):
        s = pd.Series(["2547000001.0", "2547000002"])
        result = _normalize_msisdn(s)
        assert result.tolist() == ["2547000001", "2547000002"]

    def test_strips_whitespace(self):
        s = pd.Series(["  2547000001 "])
        assert _normalize_msisdn(s).iloc[0] == "2547000001"


class TestLoadAndMergeLists:
    def test_returns_dataframe(self, tmp_path):
        wl, bl = _make_list_csvs(tmp_path)
        df = load_and_merge_lists(wl, bl)
        assert isinstance(df, pd.DataFrame)

    def test_deduplication_blacklist_priority(self, tmp_path):
        # Make one MSISDN appear in both
        wl = pd.DataFrame({"agent_msisdn": ["111"], "agent_category": "a", "reason": "r"})
        bl = pd.DataFrame({"agent_msisdn": ["111"], "agent_category": "a", "reason": "b"})
        wl_path = tmp_path / "wl.csv"
        bl_path = tmp_path / "bl.csv"
        wl.to_csv(wl_path, index=False)
        bl.to_csv(bl_path, index=False)
        df = load_and_merge_lists(wl_path, bl_path)
        assert df.shape[0] == 1
        assert df["xtrafloat_list_type"].iloc[0] == "blacklist"


class TestCutoffSweep:
    def test_returns_dataframe_with_expected_cols(self):
        rng = np.random.default_rng(3)
        n = 200
        df = pd.DataFrame({
            "never_loan_pd_like_num": rng.uniform(0, 1, n),
            "is_blacklisted": rng.integers(0, 2, n),
        })
        result = cutoff_sweep(df)
        for col in ("approval_pct", "threshold_pd_like", "approved_n",
                    "approved_blacklist_rate", "declined_blacklist_capture"):
            assert col in result.columns

    def test_custom_pct_grid(self):
        rng = np.random.default_rng(4)
        df = pd.DataFrame({
            "never_loan_pd_like_num": rng.uniform(0, 1, 200),
            "is_blacklisted": rng.integers(0, 2, 200),
        })
        result = cutoff_sweep(df, pct_grid=[25, 50, 75])
        assert list(result["approval_pct"]) == [25, 50, 75]


class TestRunWhitelistBlacklistEval:
    def test_returns_dict_with_expected_keys(self, tmp_path):
        wl_path, bl_path = _make_list_csvs(tmp_path)
        thin_df = _make_scored_thin(wl_path, bl_path)
        result = run_whitelist_blacklist_eval(thin_df, wl_path, bl_path)
        for key in ("auc", "auc_perf_filtered", "decile_table", "lift_table", "cutoff_sweep"):
            assert key in result

    def test_auc_is_float(self, tmp_path):
        wl_path, bl_path = _make_list_csvs(tmp_path)
        thin_df = _make_scored_thin(wl_path, bl_path)
        result = run_whitelist_blacklist_eval(thin_df, wl_path, bl_path)
        assert isinstance(result["auc"], float)

    def test_auc_above_05_on_separable_data(self, tmp_path):
        wl_path, bl_path = _make_list_csvs(tmp_path)
        thin_df = _make_scored_thin(wl_path, bl_path)
        result = run_whitelist_blacklist_eval(thin_df, wl_path, bl_path)
        assert result["auc"] > 0.5

    def test_non_perf_reasons_filtered(self, tmp_path):
        wl_path, bl_path = _make_list_csvs(tmp_path)
        thin_df = _make_scored_thin(wl_path, bl_path)
        result_full = run_whitelist_blacklist_eval(thin_df, wl_path, bl_path, non_perf_reasons=())
        result_filt = run_whitelist_blacklist_eval(
            thin_df, wl_path, bl_path,
            non_perf_reasons=("bad_debt",),  # filter all blacklist rows
        )
        # perf_eval_df should have fewer rows when filtering
        assert result_filt["perf_eval_df"].shape[0] <= result_full["perf_eval_df"].shape[0]

    def test_cutoff_sweep_has_rows(self, tmp_path):
        wl_path, bl_path = _make_list_csvs(tmp_path)
        thin_df = _make_scored_thin(wl_path, bl_path)
        result = run_whitelist_blacklist_eval(thin_df, wl_path, bl_path)
        assert result["cutoff_sweep"].shape[0] > 0

"""Tests for pd_model.preprocessing.transformations."""

import numpy as np
import pandas as pd
import pytest

from pd_model.preprocessing.transformations import (
    apply_pd_transformations,
    get_and_classify_pd_features,
    prune_post_transform_features,
)


def _make_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "agent_msisdn": range(n),
            "snapshot_dt": pd.to_datetime("2025-09-30"),
            "bad_state": rng.integers(0, 2, n),
            # LOG column
            "average_balance": rng.exponential(1000, n),
            # SIGNED LOG column
            "net_cash_flow_3m": rng.normal(0, 500, n),
            # CAP column
            "vol_share_1m_of_3m": rng.uniform(0, 2, n),
            # PROTECTED column
            "is_fully_inactive_6m": rng.integers(0, 2, n),
            # Count column
            "vol_1m": rng.exponential(200, n),
            "vol_3m": rng.exponential(600, n),
        }
    )


class TestGetAndClassifyPdFeatures:
    def test_returns_six_items(self):
        result = get_and_classify_pd_features(_make_df())
        assert len(result) == 6

    def test_bad_state_excluded(self):
        pd_features, *_ = get_and_classify_pd_features(_make_df())
        assert "bad_state" not in pd_features

    def test_agent_msisdn_excluded(self):
        pd_features, *_ = get_and_classify_pd_features(_make_df())
        assert "agent_msisdn" not in pd_features

    def test_buckets_are_disjoint(self):
        _, log_cols, cap_cols, protected_cols, signed_log_cols, _ = get_and_classify_pd_features(_make_df())
        log_set = set(log_cols)
        cap_set = set(cap_cols)
        prot_set = set(protected_cols)
        signed_set = set(signed_log_cols)
        assert not (log_set & cap_set), "LOG ∩ CAP must be empty"
        assert not (log_set & prot_set), "LOG ∩ PROTECTED must be empty"
        assert not (cap_set & prot_set), "CAP ∩ PROTECTED must be empty"
        assert not (signed_set & (log_set | cap_set | prot_set)), "SIGNED_LOG overlaps other buckets"

    def test_excluded_df_has_columns(self):
        *_, excluded_df = get_and_classify_pd_features(_make_df())
        assert "feature" in excluded_df.columns
        assert "reason" in excluded_df.columns


class TestApplyPdTransformations:
    def test_output_shape_unchanged(self):
        df = _make_df()
        pd_features, log_cols, cap_cols, *_ = get_and_classify_pd_features(df)
        df_out, _, _ = apply_pd_transformations(df, pd_features, log_cols, cap_cols)
        assert df_out.shape[0] == df.shape[0]

    def test_signed_log_preserves_sign(self):
        """sign(x) * log1p(|x|) should preserve the sign of each element."""
        df = _make_df()
        pd_features, log_cols, cap_cols, _, signed_log_cols, _ = get_and_classify_pd_features(df)
        df_out, _, _ = apply_pd_transformations(
            df, pd_features, log_cols, cap_cols, signed_log_cols=signed_log_cols
        )
        if "net_cash_flow_3m" in signed_log_cols and "net_cash_flow_3m" in df.columns:
            raw = df["net_cash_flow_3m"].dropna()
            transformed = df_out["net_cash_flow_3m"].dropna()
            # Signs must agree for non-zero values
            common = raw.index.intersection(transformed.index)
            nonzero = raw.loc[common][raw.loc[common] != 0]
            assert (np.sign(nonzero.values) == np.sign(transformed.loc[nonzero.index].values)).all()

    def test_output_float32(self):
        df = _make_df()
        pd_features, log_cols, cap_cols, _, signed_log_cols, _ = get_and_classify_pd_features(df)
        df_out, _, _ = apply_pd_transformations(
            df, pd_features, log_cols, cap_cols, signed_log_cols=signed_log_cols
        )
        for col in log_cols:
            if col in df_out.columns:
                assert df_out[col].dtype == np.float32

    def test_all_nan_column_skipped_gracefully(self):
        df = _make_df()
        df["average_balance"] = np.nan
        pd_features, log_cols, cap_cols, *_ = get_and_classify_pd_features(df)
        df_out, _, report = apply_pd_transformations(df, pd_features, log_cols, cap_cols)
        actions = report["action"].tolist()
        assert any("skip_all_nan" in a for a in actions)

    def test_transform_report_returned(self):
        df = _make_df()
        pd_features, log_cols, cap_cols, *_ = get_and_classify_pd_features(df)
        _, _, report = apply_pd_transformations(df, pd_features, log_cols, cap_cols)
        assert isinstance(report, pd.DataFrame)
        assert "feature" in report.columns
        assert "action" in report.columns


class TestPrunePostTransformFeatures:
    def test_drops_all_nan_column(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan, np.nan]})
        df_pruned, kept, dropped = prune_post_transform_features(df, ["a", "b"])
        assert "b" in dropped
        assert "a" in kept

    def test_drops_constant_column(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
        df_pruned, kept, dropped = prune_post_transform_features(df, ["a", "b"])
        assert "b" in dropped

    def test_keeps_valid_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        _, kept, dropped = prune_post_transform_features(df, ["a", "b"])
        assert not dropped
        assert "a" in kept and "b" in kept

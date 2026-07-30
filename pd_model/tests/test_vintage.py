"""Tests for pd_model.monitoring.vintage."""
import numpy as np
import pandas as pd
import pytest

from pd_model.monitoring.vintage import (
    build_cohort_matrix,
    build_vintage_summary,
    build_vintage_table,
)


def _make_loans(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    disburse_months = rng.choice(
        pd.date_range("2025-01-01", periods=6, freq="MS"), n
    )
    obs_months = [
        d + pd.DateOffset(months=int(m))
        for d, m in zip(disburse_months, rng.integers(1, 7, n))
    ]
    return pd.DataFrame({
        "agent_msisdn": [f"msisdn_{i}" for i in range(n)],
        "disbursement_date": pd.to_datetime(disburse_months),
        "observation_date": pd.to_datetime(obs_months),
        "is_bad": rng.integers(0, 2, n),
        "loan_amount": rng.uniform(100, 1000, n),
    })


class TestBuildVintageTable:
    def test_returns_dataframe(self):
        df = _make_loans()
        vtbl = build_vintage_table(df)
        assert isinstance(vtbl, pd.DataFrame)

    def test_required_columns_present(self):
        vtbl = build_vintage_table(_make_loans())
        for col in ("cohort_month", "mob", "n_loans", "n_bads", "bad_rate", "cumulative_bad_rate"):
            assert col in vtbl.columns

    def test_bad_rate_bounded(self):
        vtbl = build_vintage_table(_make_loans())
        assert (vtbl["bad_rate"] >= 0).all()
        assert (vtbl["bad_rate"] <= 1).all()

    def test_mob_non_negative(self):
        vtbl = build_vintage_table(_make_loans())
        assert (vtbl["mob"] >= 0).all()

    def test_n_bads_le_n_loans(self):
        vtbl = build_vintage_table(_make_loans())
        assert (vtbl["n_bads"] <= vtbl["n_loans"]).all()

    def test_cumulative_bad_rate_bounded(self):
        vtbl = build_vintage_table(_make_loans(n=400))
        assert (vtbl["cumulative_bad_rate"] >= 0).all()
        assert (vtbl["cumulative_bad_rate"] <= 1).all()

    def test_drops_bad_dates(self):
        df = _make_loans(n=50)
        df.loc[0, "disbursement_date"] = pd.NaT
        vtbl = build_vintage_table(df)
        total = vtbl["n_loans"].sum()
        assert total <= 50


class TestBuildCohortMatrix:
    def test_shape(self):
        vtbl = build_vintage_table(_make_loans())
        matrix = build_cohort_matrix(vtbl)
        n_cohorts = vtbl["cohort_month"].nunique()
        assert matrix.shape[0] == n_cohorts

    def test_mob_as_columns(self):
        vtbl = build_vintage_table(_make_loans())
        matrix = build_cohort_matrix(vtbl)
        assert matrix.columns.name == "mob"

    def test_values_bounded(self):
        vtbl = build_vintage_table(_make_loans())
        matrix = build_cohort_matrix(vtbl)
        vals = matrix.values.flatten()
        vals = vals[~np.isnan(vals)]
        assert (vals >= 0).all() and (vals <= 1).all()


class TestBuildVintageSummary:
    def test_one_row_per_cohort(self):
        vtbl = build_vintage_table(_make_loans())
        summary = build_vintage_summary(vtbl)
        assert len(summary) == vtbl["cohort_month"].nunique()

    def test_required_columns(self):
        vtbl = build_vintage_table(_make_loans())
        summary = build_vintage_summary(vtbl)
        for col in ("cohort_month", "n_loans_total", "n_bads_total",
                    "overall_bad_rate", "max_mob_observed"):
            assert col in summary.columns

    def test_bad_rate_bounded(self):
        vtbl = build_vintage_table(_make_loans())
        summary = build_vintage_summary(vtbl)
        assert (summary["overall_bad_rate"] >= 0).all()
        assert (summary["overall_bad_rate"] <= 1).all()

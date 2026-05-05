"""
tests/io/test_extrafloat_data_loaders.py
=========================================
10 tests for the three production CSV loaders.
"""

import pandas as pd
import pytest

from extrafloat.io.extrafloat_data_loaders import (
    load_borrower_limit_features,
    load_loan_summary_recent_features,
    load_transaction_capacity_features,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(tmp_path, filename, df):
    p = tmp_path / filename
    df.to_csv(p, index=False)
    return p


def _txn_df(**extra_cols):
    base = {
        "agent_msisdn": ["256785"],
        "tbl_dt": ["2025-11-15"],
        "agent_profile": ["Silver Class"],
        "account_balance": [9403.0],
        "average_balance": [37646.15],
        "commission": [14402.80],
        "cash_out_vol_1m": [2], "cash_out_vol_3m": [5], "cash_out_vol_6m": [9],
        "cash_out_value_1m": [40000], "cash_out_value_3m": [100000], "cash_out_value_6m": [180000],
        "cash_out_cust_1m": [2], "cash_out_cust_3m": [5], "cash_out_cust_6m": [9],
        "cash_out_comm_1m": [547], "cash_out_comm_3m": [1200], "cash_out_comm_6m": [2000],
        "cash_in_vol_1m": [3], "cash_in_vol_3m": [8], "cash_in_vol_6m": [15],
        "cash_in_value_1m": [30000], "cash_in_value_3m": [80000], "cash_in_value_6m": [150000],
        "cash_in_cust_1m": [3], "cash_in_cust_3m": [8], "cash_in_cust_6m": [14],
        "cash_in_comm_1m": [400], "cash_in_comm_3m": [900], "cash_in_comm_6m": [1700],
        "payment_vol_1m": [5], "payment_vol_3m": [12], "payment_vol_6m": [22],
        "payment_value_1m": [25000], "payment_value_3m": [60000], "payment_value_6m": [110000],
        "payment_cust_1m": [4], "payment_cust_3m": [10], "payment_cust_6m": [20],
        "payment_comm_1m": [300], "payment_comm_3m": [700], "payment_comm_6m": [1300],
        "cust_1m": [14], "cust_3m": [39], "cust_6m": [60],
        "vol_1m": [14], "vol_3m": [39], "vol_6m": [60],
    }
    base.update(extra_cols)
    return pd.DataFrame(base)


def _loan_df(**extra_cols):
    base = {
        "msisdn": ["256785"],
        "snapshot_dt": ["2025-11-15"],
        "last_disbursement_date": ["2025-11-01"],
        "last_repayment_date": ["2025-11-10"],
        "disbursement_vol_1m": [3],
        "disbursement_val_1m": [15000.0],
        "repayment_vol_1m": [3],
        "repayment_val_1m": [15000.0],
        "penalties_1m": [0.0],
    }
    base.update(extra_cols)
    return pd.DataFrame(base)


def _borrower_df(**extra_cols):
    base = {
        "phonenumber": ["256785"],
        "total_loans": [12],
        "first_loan_ts": ["2023-01-15 08:00:00"],
        "latest_loan_ts": ["2025-11-01 09:00:00"],
        "latest_disbursement_ts": ["2025-11-01 09:00:00"],
    }
    base.update(extra_cols)
    return pd.DataFrame(base)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTION CAPACITY
# ─────────────────────────────────────────────────────────────────────────────

def test_load_transaction_missing_file():
    with pytest.raises(FileNotFoundError, match="file not found"):
        load_transaction_capacity_features("/no/such/file.csv")


def test_load_transaction_agent_msisdn_renamed(tmp_path):
    p = _write_csv(tmp_path, "txn.csv", _txn_df())
    df = load_transaction_capacity_features(p)
    assert "msisdn" in df.columns
    assert "agent_msisdn" not in df.columns


def test_load_transaction_tbl_dt_renamed_to_snapshot_dt(tmp_path):
    p = _write_csv(tmp_path, "txn.csv", _txn_df())
    df = load_transaction_capacity_features(p)
    assert "snapshot_dt" in df.columns
    assert "tbl_dt" not in df.columns


def test_load_transaction_snapshot_dt_kept_when_present(tmp_path):
    raw = _txn_df()
    raw["snapshot_dt"] = raw["tbl_dt"]  # both present — snapshot_dt wins
    p = _write_csv(tmp_path, "txn.csv", raw)
    df = load_transaction_capacity_features(p)
    assert "snapshot_dt" in df.columns
    # tbl_dt should remain untouched since rename was skipped
    assert "tbl_dt" in df.columns


def test_load_transaction_all_columns_preserved(tmp_path):
    raw = _txn_df()
    p = _write_csv(tmp_path, "txn.csv", raw)
    df = load_transaction_capacity_features(p)
    # Every source column (after rename) must be present — no subsetting
    original_cols = {
        "msisdn" if c == "agent_msisdn" else ("snapshot_dt" if c == "tbl_dt" else c)
        for c in raw.columns
    }
    assert original_cols.issubset(set(df.columns))


def test_load_transaction_snapshot_dt_is_datetime(tmp_path):
    p = _write_csv(tmp_path, "txn.csv", _txn_df())
    df = load_transaction_capacity_features(p)
    assert pd.api.types.is_datetime64_any_dtype(df["snapshot_dt"])


# ─────────────────────────────────────────────────────────────────────────────
# LOAN SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def test_load_loan_summary_missing_file():
    with pytest.raises(FileNotFoundError, match="file not found"):
        load_loan_summary_recent_features("/no/such/file.csv")


def test_load_loan_summary_3m_cols_zero_filled_when_absent(tmp_path):
    p = _write_csv(tmp_path, "loan.csv", _loan_df())  # no 3m cols
    df = load_loan_summary_recent_features(p)
    for col in ["disbursement_val_3m", "repayment_val_3m", "penalties_3m"]:
        assert col in df.columns
        assert (df[col] == 0.0).all()


def test_load_loan_summary_3m_cols_preserved_when_present(tmp_path):
    raw = _loan_df(disbursement_val_3m=[45000.0], repayment_val_3m=[44000.0], penalties_3m=[100.0])
    p = _write_csv(tmp_path, "loan.csv", raw)
    df = load_loan_summary_recent_features(p)
    assert df["disbursement_val_3m"].iloc[0] == 45000.0
    assert df["repayment_val_3m"].iloc[0] == 44000.0
    assert df["penalties_3m"].iloc[0] == 100.0


def test_load_loan_summary_dates_parsed(tmp_path):
    p = _write_csv(tmp_path, "loan.csv", _loan_df())
    df = load_loan_summary_recent_features(p)
    for col in ["snapshot_dt", "last_disbursement_date", "last_repayment_date"]:
        assert pd.api.types.is_datetime64_any_dtype(df[col]), f"{col} not datetime"


# ─────────────────────────────────────────────────────────────────────────────
# BORROWER LIMIT
# ─────────────────────────────────────────────────────────────────────────────

def test_load_borrower_limit_missing_file():
    with pytest.raises(FileNotFoundError, match="file not found"):
        load_borrower_limit_features("/no/such/file.csv")


def test_load_borrower_limit_phonenumber_renamed(tmp_path):
    p = _write_csv(tmp_path, "borrow.csv", _borrower_df())
    df = load_borrower_limit_features(p)
    assert "msisdn" in df.columns
    assert "phonenumber" not in df.columns


def test_load_borrower_limit_msisdn_kept_when_present(tmp_path):
    raw = _borrower_df()
    raw["msisdn"] = raw["phonenumber"]  # both present — msisdn wins
    p = _write_csv(tmp_path, "borrow.csv", raw)
    df = load_borrower_limit_features(p)
    assert "msisdn" in df.columns
    assert int(df["msisdn"].iloc[0]) == 256785  # type cast left to prepare_borrower_limit_features


def test_load_borrower_limit_timestamps_parsed(tmp_path):
    p = _write_csv(tmp_path, "borrow.csv", _borrower_df())
    df = load_borrower_limit_features(p)
    for col in ["first_loan_ts", "latest_loan_ts", "latest_disbursement_ts"]:
        assert pd.api.types.is_datetime64_any_dtype(df[col]), f"{col} not datetime"

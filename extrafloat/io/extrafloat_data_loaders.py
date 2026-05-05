"""
extrafloat_data_loaders.py
===========================
Production CSV loaders for the three ExtraFloat input DataFrames.

Responsibility boundary
-----------------------
These loaders do the minimum needed before handing off to the
``prepare_*`` functions in ``extrafloat_limit_engine_features``:

  1. Validate the file exists.
  2. Read the CSV.
  3. Rename legacy column names to the canonical names the pipeline expects.
  4. Parse known date/timestamp columns.
  5. Zero-fill optional 3m columns in loan_summary if absent (with a warning).
  6. Return the full DataFrame — no column subsetting.

All validation of required columns, numeric coercion, aggregation, and
feature engineering is handled downstream by the ``prepare_*`` functions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 3m columns required by prepare_loan_summary_recent_features().
# If missing from the source file they are zero-filled with a warning so the
# pipeline can still run (temporal blending falls back to 1m values).
_LOAN_SUMMARY_3M_COLS = [
    "disbursement_val_3m",
    "repayment_val_3m",
    "penalties_3m",
]


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read CSV after confirming the file exists."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ExtraFloat data loader: file not found — {p.resolve()}"
        )
    df = pd.read_csv(p, low_memory=False)
    logger.info("Loaded %s — %d rows, %d columns", p.name, len(df), len(df.columns))
    return df


def _parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce known date columns to datetime, leaving unrecognised values as NaT."""
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\.\d+$", "", regex=True)  # strip microseconds / ".0"
                .replace({"": None, "None": None, "nan": None, "<NA>": None})
            )
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRANSACTION CAPACITY FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def load_transaction_capacity_features(path: str | Path) -> pd.DataFrame:
    """
    Load the agent transaction capacity CSV.

    Column renames applied
    ----------------------
    - ``agent_msisdn``  → ``msisdn``      (if ``msisdn`` absent)
    - ``tbl_dt``        → ``snapshot_dt`` (if ``snapshot_dt`` absent)

    Date columns parsed
    -------------------
    ``snapshot_dt``, ``tbl_dt``, ``activation_dt``

    The returned DataFrame contains every column from the source file.
    Pass it directly to ``prepare_transaction_capacity_features()``.
    """
    df = _read_csv(path)

    # ── Column renames ──────────────────────────────────────────────────────
    if "agent_msisdn" in df.columns and "msisdn" not in df.columns:
        df = df.rename(columns={"agent_msisdn": "msisdn"})
        logger.info("Renamed agent_msisdn → msisdn")

    if "tbl_dt" in df.columns and "snapshot_dt" not in df.columns:
        df = df.rename(columns={"tbl_dt": "snapshot_dt"})
        logger.info("Renamed tbl_dt → snapshot_dt")

    # ── Date parsing ────────────────────────────────────────────────────────
    df = _parse_dates(df, ["snapshot_dt", "tbl_dt", "activation_dt"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAN SUMMARY RECENT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def load_loan_summary_recent_features(path: str | Path) -> pd.DataFrame:
    """
    Load the loan summary CSV.

    Date columns parsed
    -------------------
    ``snapshot_dt``, ``tbl_dt``, ``last_disbursement_date``, ``last_repayment_date``

    Missing 3m columns
    ------------------
    If ``disbursement_val_3m``, ``repayment_val_3m``, or ``penalties_3m`` are
    absent from the source file they are zero-filled and a warning is logged.
    The pipeline requires these columns for temporal blending in
    ``compute_recent_usage_cap()``; zero-filling causes it to fall back
    gracefully to 1m values.

    The returned DataFrame contains every column from the source file plus any
    zero-filled 3m columns. Pass it directly to
    ``prepare_loan_summary_recent_features()``.
    """
    df = _read_csv(path)

    # ── Date parsing ────────────────────────────────────────────────────────
    df = _parse_dates(
        df,
        ["snapshot_dt", "tbl_dt", "last_disbursement_date", "last_repayment_date"],
    )

    # ── Zero-fill missing 3m columns ────────────────────────────────────────
    missing_3m = [c for c in _LOAN_SUMMARY_3M_COLS if c not in df.columns]
    if missing_3m:
        logger.warning(
            "load_loan_summary_recent_features: 3m columns absent from source "
            "and zero-filled — %s. Temporal blending in compute_recent_usage_cap() "
            "will fall back to 1m values.",
            ", ".join(missing_3m),
        )
        for col in missing_3m:
            df[col] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. BORROWER LIMIT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def load_borrower_limit_features(path: str | Path) -> pd.DataFrame:
    """
    Load the borrower credit history CSV.

    Column renames applied
    ----------------------
    - ``phonenumber`` → ``msisdn`` (if ``msisdn`` absent)

    Timestamp columns parsed
    ------------------------
    ``first_loan_ts``, ``latest_loan_ts``, ``latest_disbursement_ts``

    The returned DataFrame contains every column from the source file.
    Pass it directly to ``prepare_borrower_limit_features()``.
    """
    df = _read_csv(path)

    # ── Column renames ──────────────────────────────────────────────────────
    if "phonenumber" in df.columns and "msisdn" not in df.columns:
        df = df.rename(columns={"phonenumber": "msisdn"})
        logger.info("Renamed phonenumber → msisdn")

    # ── Timestamp parsing ───────────────────────────────────────────────────
    df = _parse_dates(
        df,
        ["first_loan_ts", "latest_loan_ts", "latest_disbursement_ts"],
    )

    return df

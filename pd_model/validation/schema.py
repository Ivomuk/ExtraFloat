"""
Input / output validation helpers for the PD model pipeline.

These functions raise clear, context-rich exceptions rather than letting
silent failures propagate downstream.  They are designed to be called at
the entry and exit points of each pipeline step.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from pd_model.logging_config import get_logger

logger = get_logger(__name__)


def require_columns(df: pd.DataFrame, required: List[str], context: str = "") -> None:
    """
    Assert that *df* contains every column in *required*.

    Args:
        df:       DataFrame to validate.
        required: Column names that must be present.
        context:  Human-readable label for the calling step (e.g. ``"run_phase_2_1"``).

    Raises:
        ValueError: If any required column is absent.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {missing}"
        )


def check_missing_rates(
    df: pd.DataFrame,
    cols: List[str],
    max_rate: float = 0.95,
    context: str = "",
) -> None:
    """
    Warn when a column's missing rate exceeds *max_rate*.

    Args:
        df:       DataFrame to inspect.
        cols:     Column names to check.
        max_rate: Threshold above which a column is flagged (default 0.95).
        context:  Human-readable label for the calling step.

    Raises:
        ValueError: If any column's missing rate exceeds *max_rate*.
    """
    for col in cols:
        if col not in df.columns:
            continue
        rate = df[col].isna().mean()
        if rate > max_rate:
            raise ValueError(
                f"[{context}] Column '{col}' has {rate:.1%} missing "
                f"(threshold {max_rate:.0%})"
            )


def assert_output_not_empty(df: pd.DataFrame, context: str = "") -> None:
    """
    Assert that *df* has at least one row.

    Args:
        df:      DataFrame to check.
        context: Human-readable label for the calling step.

    Raises:
        ValueError: If *df* is empty.
    """
    if len(df) == 0:
        raise ValueError(f"[{context}] Output DataFrame is empty")


def assert_binary_column(df: pd.DataFrame, col: str, context: str = "") -> None:
    """
    Assert that *col* contains only 0 and 1 (no NaNs, no other values).

    Args:
        df:      DataFrame containing the column.
        col:     Column name to validate.
        context: Human-readable label for the calling step.

    Raises:
        ValueError: If the column has NaNs or values outside {0, 1}.
    """
    require_columns(df, [col], context=context)
    if df[col].isna().any():
        raise ValueError(f"[{context}] Column '{col}' contains NaN values")
    bad_vals = set(df[col].unique()) - {0, 1}
    if bad_vals:
        raise ValueError(
            f"[{context}] Column '{col}' contains values outside {{0, 1}}: {bad_vals}"
        )


def assert_index_aligned(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    context: str = "",
) -> None:
    """
    Assert that two DataFrames have identical indices.

    Args:
        df_a:    First DataFrame.
        df_b:    Second DataFrame.
        context: Human-readable label for the calling step.

    Raises:
        ValueError: If the indices differ.
    """
    if not df_a.index.equals(df_b.index):
        raise ValueError(
            f"[{context}] Index mismatch: "
            f"df_a has {len(df_a)} rows, df_b has {len(df_b)} rows; "
            "indices are not identical."
        )

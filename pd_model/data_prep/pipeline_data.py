"""
Final training/validation data preparation for PD modelling (file6).

Provides:
- ``prepare_pd_training_and_validation_data`` – time-split the modelling
  DataFrame into train and validation sets, enforce leakage guards, and
  return aligned X / y / metadata objects ready for IV filtering and model
  training.
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Tuple

import pandas as pd

from pd_model.config import feature_config
from pd_model.logging_config import get_logger

logger = get_logger(__name__)


def prepare_pd_training_and_validation_data(
    df_pd_raw: pd.DataFrame,
    df_pd_transformed: pd.DataFrame,
    target_col: str,
    train_cutoff: pd.Timestamp,
    id_cols: List[str],
    protected_cols: List[str],
    pd_feature_blacklist: FrozenSet[str],
    forbidden_feature_patterns: Tuple[str, ...],
    date_cols: List[str],
    split_date_col: str = "snapshot_dt",
    allowed_features: Optional[List[str]] = None,
) -> Tuple[
    pd.DataFrame,  # X_train_raw
    pd.DataFrame,  # X_train_trans
    pd.Series,     # y_train
    pd.DataFrame,  # X_val_raw
    pd.DataFrame,  # X_val_trans
    pd.Series,     # y_val
    List[str],     # candidate_features
    pd.Series,     # thin_train
    pd.Series,     # thin_val
    pd.Series,     # agent_train
    pd.Series,     # agent_val
]:
    """
    Split the modelling DataFrame into aligned train / validation feature matrices.

    Steps
    -----
    1. Force index alignment between raw and transformed DataFrames.
    2. Validate the split date column and coerce to datetime.
    3. Build the candidate numeric feature list (exclude IDs, protected,
       blacklisted, date, target, pattern-matched columns).
    4. Assert no pattern-based leakage in candidate features.
    5. Time-split at ``train_cutoff``.
    6. Build X / y matrices and apply final schema assertions.

    Args:
        df_pd_raw:                 Raw modelling DataFrame (contains ID, date, and
                                   non-numeric passthrough columns).
        df_pd_transformed:         Transformed numeric features (aligned index with raw).
        target_col:                Binary target column name.
        train_cutoff:              Inclusive upper bound for training rows.
        id_cols:                   Columns to exclude (agent keys, etc.).
        protected_cols:            Non-feature columns to exclude (flags, etc.).
        pd_feature_blacklist:      Exact-match exclusion set (case-insensitive).
        forbidden_feature_patterns:Substring exclusion patterns (case-insensitive).
        date_cols:                 Date/datetime columns to exclude.
        split_date_col:            Column used to perform the train/val split.
        allowed_features:          If provided, restrict candidates to this list.

    Returns:
        11-tuple:
        ``(X_train_raw, X_train_trans, y_train,
           X_val_raw, X_val_trans, y_val,
           candidate_features, thin_train, thin_val, agent_train, agent_val)``
    """
    agent_key = feature_config.AGENT_KEY
    thin_col = feature_config.THIN_FILE_COL

    # ------------------------------------------------------------------ #
    # 1) Force index alignment
    # ------------------------------------------------------------------ #
    common_idx = df_pd_raw.index.intersection(df_pd_transformed.index)
    df_pd_raw = df_pd_raw.loc[common_idx].copy()
    df_pd_transformed = df_pd_transformed.loc[common_idx].copy()
    assert df_pd_raw.index.equals(df_pd_transformed.index), (
        "[prepare_pd_data] Raw and transformed snapshots misaligned after alignment"
    )

    # ------------------------------------------------------------------ #
    # 2) Split date validation
    # ------------------------------------------------------------------ #
    assert (split_date_col in df_pd_raw.columns) or (
        split_date_col in df_pd_transformed.columns
    ), f"[prepare_pd_data] split_date_col '{split_date_col}' missing from both DataFrames"

    if split_date_col in df_pd_raw.columns:
        df_pd_raw[split_date_col] = pd.to_datetime(
            df_pd_raw[split_date_col], errors="coerce"
        )
        split_series = df_pd_raw[split_date_col]
    else:
        df_pd_transformed[split_date_col] = pd.to_datetime(
            df_pd_transformed[split_date_col], errors="coerce"
        )
        split_series = df_pd_transformed[split_date_col]

    assert split_series.notna().all(), (
        f"[prepare_pd_data] split_date_col '{split_date_col}' has NaT values"
    )

    # Coerce optional date columns if present
    for c in date_cols:
        for df_ref in [df_pd_raw, df_pd_transformed]:
            if c in df_ref.columns and not pd.api.types.is_datetime64_any_dtype(df_ref[c]):
                df_ref[c] = pd.to_datetime(df_ref[c], errors="coerce")

    # ------------------------------------------------------------------ #
    # 3) Candidate numeric feature selection
    # ------------------------------------------------------------------ #
    forbidden_patterns_low = [str(p).lower() for p in forbidden_feature_patterns]
    blacklist_low = {str(c).lower() for c in pd_feature_blacklist}
    id_cols_low = {str(c).lower() for c in id_cols}
    protected_low = {str(c).lower() for c in protected_cols}

    base_cols = allowed_features if allowed_features is not None else df_pd_transformed.columns
    candidate_features: List[str] = []

    for c in base_cols:
        if c not in df_pd_transformed.columns:
            continue
        c_low = str(c).lower()
        if c_low in id_cols_low:
            continue
        if c_low in protected_low:
            continue
        if c_low in blacklist_low:
            continue
        if c == target_col or c == split_date_col:
            continue
        if not pd.api.types.is_numeric_dtype(df_pd_transformed[c]):
            continue
        candidate_features.append(c)

    logger.info(
        "prepare_pd_data: %d candidate numeric features identified",
        len(candidate_features),
    )

    # ------------------------------------------------------------------ #
    # 4) Pattern-based leakage guard
    # ------------------------------------------------------------------ #
    leakage_hits = [
        c
        for c in candidate_features
        if any(p in c.lower() for p in forbidden_patterns_low)
    ]
    assert not leakage_hits, (
        "[prepare_pd_data] Pattern-based leakage detected. "
        "Add to PD_FEATURE_BLACKLIST explicitly:\n" + ", ".join(leakage_hits)
    )
    assert agent_key not in candidate_features, (
        f"[prepare_pd_data] {agent_key} leaked into candidate_features"
    )

    # ------------------------------------------------------------------ #
    # 5) Time-based split
    # ------------------------------------------------------------------ #
    if split_date_col in df_pd_raw.columns:
        split_series = pd.to_datetime(df_pd_raw[split_date_col], errors="coerce")
    else:
        split_series = pd.to_datetime(df_pd_transformed[split_date_col], errors="coerce")

    train_mask = split_series <= train_cutoff
    val_mask = ~train_mask

    df_train_trans = df_pd_transformed.loc[train_mask].copy()
    df_train_raw = df_pd_raw.loc[train_mask].copy()
    df_val_trans = df_pd_transformed.loc[val_mask].copy()
    df_val_raw = df_pd_raw.loc[val_mask].copy()

    # ------------------------------------------------------------------ #
    # 6) Schema assertions
    # ------------------------------------------------------------------ #
    assert agent_key in df_train_raw.columns, f"[prepare_pd_data] {agent_key} missing in df_train_raw"
    assert agent_key in df_val_raw.columns, f"[prepare_pd_data] {agent_key} missing in df_val_raw"
    assert df_train_raw[agent_key].notna().all(), f"[prepare_pd_data] {agent_key} nulls in train"
    assert df_val_raw[agent_key].notna().all(), f"[prepare_pd_data] {agent_key} nulls in val"

    agent_train = df_train_raw[agent_key].copy()
    agent_val = df_val_raw[agent_key].copy()

    assert not df_train_raw.empty, "[prepare_pd_data] Training set is empty"
    assert df_train_raw.shape[0] == df_train_trans.shape[0], (
        "[prepare_pd_data] Raw and transformed train rows misaligned"
    )
    assert df_train_raw.index.equals(df_train_trans.index), (
        "[prepare_pd_data] Raw and transformed train indexes misaligned"
    )
    assert df_val_raw.shape[0] == df_val_trans.shape[0], (
        "[prepare_pd_data] Raw and transformed val rows misaligned"
    )
    assert df_val_raw.index.equals(df_val_trans.index), (
        "[prepare_pd_data] Raw and transformed val indexes misaligned"
    )

    assert target_col in df_train_trans.columns, (
        f"[prepare_pd_data] target_col '{target_col}' missing in train transformed"
    )
    assert target_col in df_val_trans.columns, (
        f"[prepare_pd_data] target_col '{target_col}' missing in val transformed"
    )

    missing_train = [c for c in candidate_features if c not in df_train_trans.columns]
    assert not missing_train, (
        f"[prepare_pd_data] Candidate features missing in train transformed: {missing_train}"
    )
    missing_val = [c for c in candidate_features if c not in df_val_trans.columns]
    assert not missing_val, (
        f"[prepare_pd_data] Candidate features missing in val transformed: {missing_val}"
    )

    X_train_raw = df_train_raw[candidate_features]
    X_train_trans = df_train_trans[candidate_features]
    y_train = df_train_trans[target_col]

    X_val_raw = df_val_raw[candidate_features]
    X_val_trans = df_val_trans[candidate_features]
    y_val = df_val_trans[target_col]

    assert X_train_raw.shape == X_train_trans.shape, (
        "[prepare_pd_data] Train raw/trans shapes differ"
    )
    assert X_val_raw.shape == X_val_trans.shape, (
        "[prepare_pd_data] Val raw/trans shapes differ"
    )
    assert y_train.notna().all(), "[prepare_pd_data] Training target contains NaNs"
    assert y_val.notna().all(), "[prepare_pd_data] Validation target contains NaNs"

    thin_train = df_train_raw[thin_col] if thin_col in df_train_raw.columns else pd.Series(0, index=df_train_raw.index)
    thin_val = df_val_raw[thin_col] if thin_col in df_val_raw.columns else pd.Series(0, index=df_val_raw.index)

    logger.info(
        "prepare_pd_data: Train=%d rows | Val=%d rows | Features=%d",
        len(X_train_raw),
        len(X_val_raw),
        len(candidate_features),
    )
    logger.info(
        "prepare_pd_data: Bad rate — train=%.2f%% | val=%.2f%%",
        float(y_train.mean()) * 100,
        float(y_val.mean()) * 100,
    )

    return (
        X_train_raw,
        X_train_trans,
        y_train,
        X_val_raw,
        X_val_trans,
        y_val,
        candidate_features,
        thin_train,
        thin_val,
        agent_train,
        agent_val,
    )

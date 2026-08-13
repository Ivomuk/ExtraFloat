"""
Final training/validation data preparation for PD modelling (file6).

Provides:
- ``prepare_pd_training_and_validation_data`` - time-split the modelling
  DataFrame into train and validation sets, enforce leakage guards, and
  return aligned X / y / metadata objects ready for IV filtering and model
  training.
"""

from __future__ import annotations

import gc

import pandas as pd

from pd_model.config import feature_config
from pd_model.exceptions import DataAlignmentError, DataLeakageError, SchemaValidationError
from pd_model.logging_config import get_logger
from pd_model.validation.schema import require_index_alignment

logger = get_logger(__name__)


def prepare_pd_training_and_validation_data(
    df_pd_raw: pd.DataFrame,
    df_pd_transformed: pd.DataFrame,
    target_col: str,
    train_cutoff: pd.Timestamp | None,
    id_cols: list[str],
    protected_cols: list[str],
    pd_feature_blacklist: frozenset[str],
    forbidden_feature_patterns: tuple[str, ...],
    date_cols: list[str],
    split_date_col: str = "snapshot_dt",
    allowed_features: list[str] | None = None,
    precomputed_train_mask: pd.Series | None = None,
) -> tuple[
    pd.DataFrame,  # X_train_raw
    pd.DataFrame,  # X_train_trans
    pd.Series,  # y_train
    pd.DataFrame,  # X_val_raw
    pd.DataFrame,  # X_val_trans
    pd.Series,  # y_val
    list[str],  # candidate_features
    pd.Series,  # thin_train
    pd.Series,  # thin_val
    pd.Series,  # agent_train
    pd.Series,  # agent_val
]:
    """
    Split the modelling DataFrame into aligned train / validation feature matrices.

    Steps
    -----
    1. Build the candidate numeric feature list (exclude IDs, protected,
       blacklisted, date, target, pattern-matched columns). Done first,
       ahead of index alignment -- only needs column names/dtypes, not
       row data, so there's no reason to reindex every column in the
       input before knowing which ~40-50 of them are even needed.
    2. Narrow both DataFrames to just candidate_features plus the handful
       of other columns this function reads (agent key, thin-file flag,
       target, split date column), then force index alignment. Narrowing
       before the reindex (not after) is what keeps this from allocating
       a single contiguous block across every raw/engineered column in
       the input -- confirmed as the fix for an ArrayMemoryError on a
       loan-level, millions-of-rows export.
    3. Validate the split date column and coerce to datetime.
    4. Assert no pattern-based leakage in candidate features.
    5. Time-split at ``train_cutoff``.
    6. Build X / y matrices and apply final schema assertions.

    Args:
        df_pd_raw:                 Raw modelling DataFrame (contains ID, date, and
                                   non-numeric passthrough columns).
        df_pd_transformed:         Transformed numeric features (aligned index with raw).
        target_col:                Binary target column name.
        train_cutoff:              Inclusive upper bound for training rows. Ignored
                                   when ``precomputed_train_mask`` is supplied.
        id_cols:                   Columns to exclude (agent keys, etc.).
        protected_cols:            Non-feature columns to exclude (flags, etc.).
        pd_feature_blacklist:      Exact-match exclusion set (case-insensitive).
        forbidden_feature_patterns:Substring exclusion patterns (case-insensitive).
        date_cols:                 Date/datetime columns to exclude.
        split_date_col:            Column used to perform the train/val split.
                                   Ignored when ``precomputed_train_mask`` is
                                   supplied (still coerced to datetime if present,
                                   for downstream diagnostics only).
        allowed_features:          If provided, restrict candidates to this list.
        precomputed_train_mask:    Optional boolean Series, aligned to
                                   ``df_pd_raw``'s index, used directly as the
                                   train/validation split instead of comparing
                                   ``split_date_col`` against ``train_cutoff``.
                                   Use this when the split is already
                                   authoritative upstream (e.g. the ``split``
                                   column baked into
                                   data/loan_state_query_updated_materialized.txt's
                                   ``cohorts`` CTE) -- more robust than
                                   re-deriving a date cutoff that SQL already
                                   computed. When ``None`` (default), behaviour
                                   is unchanged from before this parameter
                                   existed.

    Returns:
        11-tuple:
        ``(X_train_raw, X_train_trans, y_train,
           X_val_raw, X_val_trans, y_val,
           candidate_features, thin_train, thin_val, agent_train, agent_val)``
    """
    agent_key = feature_config.AGENT_KEY
    thin_col = feature_config.THIN_FILE_COL

    # ------------------------------------------------------------------ #
    # 1) Candidate numeric feature selection -- moved ahead of index
    #    alignment/reindex below. Only needs column names/dtypes (cheap,
    #    metadata-only), not row data, so there's no reason to pay for a
    #    full reindex of every column before knowing which ~40-50 of them
    #    are even needed downstream.
    # ------------------------------------------------------------------ #
    forbidden_patterns_low = [str(p).lower() for p in forbidden_feature_patterns]
    blacklist_low = {str(c).lower() for c in pd_feature_blacklist}
    id_cols_low = {str(c).lower() for c in id_cols}
    protected_low = {str(c).lower() for c in protected_cols}
    # Explicit date-column exclusion. Previously this was covered only
    # indirectly (the date-coercion loop used to run BEFORE this
    # selection, so date columns already failed the is_numeric_dtype
    # check below by the time this loop saw them). Now that selection
    # runs first, that implicit protection is gone, so exclude by name
    # directly -- strictly more robust than the coercion-order trick it
    # replaces (works even for still-object/string-encoded dates).
    date_cols_low = {str(c).lower() for c in date_cols}

    base_cols = allowed_features if allowed_features is not None else df_pd_transformed.columns
    candidate_features: list[str] = []

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
        if c_low in date_cols_low:
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
    # 2) Narrow to only what this function reads, then force index
    #    alignment. Narrowing BEFORE the reindex (not after) is the fix --
    #    .loc[index] fancy indexing already returns an independent object
    #    under pandas' copy-on-write (confirmed: no shared memory, safe to
    #    mutate), but reindexing the FULL raw/engineered column set before
    #    narrowing still forces one giant contiguous allocation across
    #    every column in the input. Confirmed as the cause of an
    #    ArrayMemoryError on a loan-level, millions-of-rows export
    #    (~169 raw+engineered columns narrowed down to ~45-55 needed here).
    # ------------------------------------------------------------------ #
    need_split_col = precomputed_train_mask is None
    raw_needed = list(
        dict.fromkeys(
            candidate_features
            + [agent_key, thin_col, target_col]
            + ([split_date_col] if need_split_col else [])
        )
    )
    trans_needed = list(
        dict.fromkeys(
            candidate_features + [target_col] + ([split_date_col] if need_split_col else [])
        )
    )
    raw_needed = [c for c in raw_needed if c in df_pd_raw.columns]
    trans_needed = [c for c in trans_needed if c in df_pd_transformed.columns]

    common_idx = df_pd_raw.index.intersection(df_pd_transformed.index)
    df_pd_raw = df_pd_raw.loc[common_idx, raw_needed]
    df_pd_transformed = df_pd_transformed.loc[common_idx, trans_needed]
    require_index_alignment(df_pd_raw, df_pd_transformed, context="prepare_pd_data")

    # ------------------------------------------------------------------ #
    # 3) Split date validation (skipped when precomputed_train_mask is given)
    # ------------------------------------------------------------------ #
    if precomputed_train_mask is None:
        if split_date_col not in df_pd_raw.columns and split_date_col not in df_pd_transformed.columns:
            raise SchemaValidationError(
                f"[prepare_pd_data] split_date_col '{split_date_col}' missing from both DataFrames"
            )

        if split_date_col in df_pd_raw.columns:
            df_pd_raw[split_date_col] = pd.to_datetime(df_pd_raw[split_date_col], errors="coerce")
            split_series = df_pd_raw[split_date_col]
        else:
            df_pd_transformed[split_date_col] = pd.to_datetime(
                df_pd_transformed[split_date_col], errors="coerce"
            )
            split_series = df_pd_transformed[split_date_col]

        if not split_series.notna().all():
            raise SchemaValidationError(f"[prepare_pd_data] split_date_col '{split_date_col}' has NaT values")

    # Coerce optional date columns if present (regardless of split strategy --
    # downstream diagnostics may still want these as real datetimes). Now a
    # no-op for everything except split_date_col, since Step 2 above already
    # narrowed away every other date column -- left in place rather than
    # deleted since it's effectively free and still correct.
    for c in date_cols:
        for df_ref in [df_pd_raw, df_pd_transformed]:
            if c in df_ref.columns and not pd.api.types.is_datetime64_any_dtype(df_ref[c]):
                df_ref[c] = pd.to_datetime(df_ref[c], errors="coerce")

    # ------------------------------------------------------------------ #
    # 4) Pattern-based leakage guard
    # ------------------------------------------------------------------ #
    leakage_hits = [c for c in candidate_features if any(p in c.lower() for p in forbidden_patterns_low)]
    if leakage_hits:
        raise DataLeakageError(
            "[prepare_pd_data] Pattern-based leakage detected. "
            "Add to PD_FEATURE_BLACKLIST explicitly:\n" + ", ".join(leakage_hits)
        )
    if agent_key in candidate_features:
        raise DataLeakageError(f"[prepare_pd_data] {agent_key} leaked into candidate_features")

    # ------------------------------------------------------------------ #
    # 5) Train/validation split
    # ------------------------------------------------------------------ #
    if precomputed_train_mask is not None:
        train_mask = precomputed_train_mask.reindex(df_pd_raw.index)
        if train_mask.isna().any():
            raise SchemaValidationError(
                "[prepare_pd_data] precomputed_train_mask does not cover every row "
                "in the aligned index"
            )
        train_mask = train_mask.astype(bool)
    else:
        if split_date_col in df_pd_raw.columns:
            split_series = pd.to_datetime(df_pd_raw[split_date_col], errors="coerce")
        else:
            split_series = pd.to_datetime(df_pd_transformed[split_date_col], errors="coerce")

        train_mask = split_series <= train_cutoff
    val_mask = ~train_mask

    # Same reasoning as Step 1 above -- boolean-mask .loc indexing is already
    # independent under copy-on-write, so no explicit .copy() needed here.
    df_train_trans = df_pd_transformed.loc[train_mask]
    df_train_raw = df_pd_raw.loc[train_mask]
    df_val_trans = df_pd_transformed.loc[val_mask]
    df_val_raw = df_pd_raw.loc[val_mask]

    # df_pd_raw/df_pd_transformed (this function's own locally-narrowed
    # copies, rebound at the Step-2 reindex above) are dead weight from
    # here on -- everything downstream reads from the just-built
    # df_train_raw/df_train_trans/df_val_raw/df_val_trans instead.
    del df_pd_raw, df_pd_transformed
    gc.collect()

    # ------------------------------------------------------------------ #
    # 6) Schema assertions
    # ------------------------------------------------------------------ #
    if agent_key not in df_train_raw.columns:
        raise SchemaValidationError(f"[prepare_pd_data] {agent_key} missing in df_train_raw")
    if agent_key not in df_val_raw.columns:
        raise SchemaValidationError(f"[prepare_pd_data] {agent_key} missing in df_val_raw")
    if not df_train_raw[agent_key].notna().all():
        raise SchemaValidationError(f"[prepare_pd_data] {agent_key} nulls in train")
    if not df_val_raw[agent_key].notna().all():
        raise SchemaValidationError(f"[prepare_pd_data] {agent_key} nulls in val")

    agent_train = df_train_raw[agent_key].copy()
    agent_val = df_val_raw[agent_key].copy()

    if df_train_raw.empty:
        raise SchemaValidationError("[prepare_pd_data] Training set is empty")
    if df_train_raw.shape[0] != df_train_trans.shape[0]:
        raise DataAlignmentError("[prepare_pd_data] Raw and transformed train rows misaligned")
    require_index_alignment(df_train_raw, df_train_trans, context="prepare_pd_data/train")
    if df_val_raw.shape[0] != df_val_trans.shape[0]:
        raise DataAlignmentError("[prepare_pd_data] Raw and transformed val rows misaligned")
    require_index_alignment(df_val_raw, df_val_trans, context="prepare_pd_data/val")

    if target_col not in df_train_trans.columns:
        raise SchemaValidationError(
            f"[prepare_pd_data] target_col '{target_col}' missing in train transformed"
        )
    if target_col not in df_val_trans.columns:
        raise SchemaValidationError(f"[prepare_pd_data] target_col '{target_col}' missing in val transformed")

    missing_train = [c for c in candidate_features if c not in df_train_trans.columns]
    if missing_train:
        raise SchemaValidationError(
            f"[prepare_pd_data] Candidate features missing in train transformed: {missing_train}"
        )
    missing_val = [c for c in candidate_features if c not in df_val_trans.columns]
    if missing_val:
        raise SchemaValidationError(
            f"[prepare_pd_data] Candidate features missing in val transformed: {missing_val}"
        )

    X_train_raw = df_train_raw[candidate_features]
    X_train_trans = df_train_trans[candidate_features]
    y_train = df_train_trans[target_col]

    X_val_raw = df_val_raw[candidate_features]
    X_val_trans = df_val_trans[candidate_features]
    y_val = df_val_trans[target_col]

    # df_train_trans/df_val_trans are fully superseded now (their only
    # further reads, X_*_trans and y_*, are already extracted above) --
    # df_train_raw/df_val_raw stay alive a bit longer for thin_train/thin_val.
    del df_train_trans, df_val_trans
    gc.collect()

    if X_train_raw.shape != X_train_trans.shape:
        raise DataAlignmentError("[prepare_pd_data] Train raw/trans shapes differ")
    if X_val_raw.shape != X_val_trans.shape:
        raise DataAlignmentError("[prepare_pd_data] Val raw/trans shapes differ")
    if not y_train.notna().all():
        raise SchemaValidationError("[prepare_pd_data] Training target contains NaNs")
    if not y_val.notna().all():
        raise SchemaValidationError("[prepare_pd_data] Validation target contains NaNs")

    thin_train = (
        df_train_raw[thin_col] if thin_col in df_train_raw.columns else pd.Series(0, index=df_train_raw.index)
    )
    thin_val = (
        df_val_raw[thin_col] if thin_col in df_val_raw.columns else pd.Series(0, index=df_val_raw.index)
    )

    # df_train_raw/df_val_raw are fully superseded now -- agent_train/agent_val
    # were already extracted as independent copies earlier, and X_train_raw/
    # X_val_raw/thin_train/thin_val hold everything else still needed.
    del df_train_raw, df_val_raw
    gc.collect()

    logger.info(
        "prepare_pd_data: Train=%d rows | Val=%d rows | Features=%d",
        len(X_train_raw),
        len(X_val_raw),
        len(candidate_features),
    )
    logger.info(
        "prepare_pd_data: Bad rate -- train=%.2f%% | val=%.2f%%",
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
